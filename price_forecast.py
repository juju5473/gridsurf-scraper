"""
GridSurf Price Forecaster
=========================
Time-series decomposition, 24-hour price forecasts, availability
projections, and cross-provider spread alerts — all from the local DB.

Usage:
    python price_forecast.py                       # RTX_4090 sample + export
    python price_forecast.py --gpu H100_SXM        # Different GPU focus
    python price_forecast.py --no-export           # Print only, no file write
    python price_forecast.py --spread-only         # Spread alerts only
"""

import argparse
import json
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

DB_PATH       = Path(__file__).parent / "data" / "provider_snapshots.db"
FORECAST_JSON = Path(__file__).parent / "data" / "forecast_windows.json"

SPREAD_PROVIDERS = {"vast", "runpod", "spheron"}


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load_price_series(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return one row per (hourly-bucket, provider, gpu_name):
      captured_at  — truncated to the hour (UTC)
      price        — min price_per_gpu_hr in that bucket
      offer_count  — number of distinct offers
    """
    df = pd.read_sql_query(
        """
        SELECT
            STRFTIME('%Y-%m-%dT%H:00:00', captured_at) AS captured_at,
            provider,
            gpu_name,
            MIN(price_per_gpu_hr)  AS price,
            COUNT(*)               AS offer_count
        FROM provider_offers
        WHERE price_per_gpu_hr IS NOT NULL
          AND price_per_gpu_hr > 0
        GROUP BY 1, 2, 3
        ORDER BY 1
        """,
        conn,
    )
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


# ── 2. Time-series decomposition ──────────────────────────────────────────────

def decompose(series: pd.Series) -> dict:
    """
    Additive decomposition:  observed = trend + seasonal + noise

    Trend    — centered rolling mean (window ≈ 6 h).
    Seasonal — mean residual by hour-of-day (intraday pattern).
    Noise    — remainder after removing both components.

    Returns a dict with the three component Series plus summary scalars.
    """
    s = series.dropna().sort_index()
    n = len(s)

    window = min(6, max(2, n // 4))
    trend = s.rolling(window=window, center=True, min_periods=1).mean()

    detrended = s - trend
    hour_index = s.index.hour
    seasonal_by_hour = detrended.groupby(hour_index).mean()
    seasonal = pd.Series(
        [float(seasonal_by_hour.get(h, 0.0)) for h in hour_index],
        index=s.index,
    )
    noise = s - trend - seasonal

    # Trend slope via OLS on the trend component
    trend_vals = trend.ffill().values
    slope = float(np.polyfit(np.arange(len(trend_vals)), trend_vals, 1)[0])

    # Seasonal strength: std(seasonal) / std(detrended);  0 = flat, 1 = strong
    detrended_std = float(detrended.std()) or 1e-9
    seasonal_strength = min(1.0, float(seasonal.std()) / detrended_std)

    return {
        "trend":             trend,
        "seasonal":          seasonal,
        "noise":             noise,
        "trend_slope":       slope,
        "seasonal_strength": seasonal_strength,
        "noise_std":         float(noise.std()),
    }


# ── 3. 24-hour price forecast ─────────────────────────────────────────────────

def _time_features(ts_index: pd.DatetimeIndex, base_n: int = 0) -> np.ndarray:
    """Linear trend + intraday sin/cos + day-of-week sin/cos."""
    t    = np.arange(base_n, base_n + len(ts_index), dtype=float)
    hour = ts_index.hour + ts_index.minute / 60.0
    dow  = ts_index.dayofweek.astype(float)
    return np.column_stack([
        t,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  /  7),
        np.cos(2 * np.pi * dow  /  7),
    ])


def forecast_24h(series: pd.Series) -> pd.DataFrame:
    """
    Ridge regression on historical price data.
    Returns 24-row DataFrame: timestamp, forecast, lower_95, upper_95.
    """
    s = series.dropna().sort_index()
    if len(s) < 6:
        return pd.DataFrame()

    X_hist = _time_features(s.index)
    y      = s.values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_hist)
    model   = Ridge(alpha=1.0)
    model.fit(X_sc, y)

    resid_std = float(np.std(y - model.predict(X_sc)))

    last_ts   = s.index[-1]
    future_ts = pd.date_range(
        start  = last_ts + pd.Timedelta(hours=1),
        periods= 24,
        freq   = "h",
        tz     = last_ts.tzinfo,
    )
    X_fut  = _time_features(future_ts, base_n=len(s))
    X_fut_sc = scaler.transform(X_fut)

    preds = np.maximum(model.predict(X_fut_sc), 0.0)
    ci    = 1.96 * resid_std

    return pd.DataFrame({
        "timestamp": future_ts,
        "forecast":  preds,
        "lower_95":  np.maximum(preds - ci, 0.0),
        "upper_95":  preds + ci,
    })


# ── 4. Availability forecast ──────────────────────────────────────────────────

def forecast_availability_6h(offer_count_series: pd.Series) -> list[dict]:
    """
    Predict number of available offers for the next 6 hours.
    Strategy: hour-of-day mean, smoothed with an exponential weight toward
    recent observations to capture short-term trends.
    """
    s = offer_count_series.dropna().sort_index()
    if len(s) < 3:
        return []

    by_hour    = s.groupby(s.index.hour).mean()
    recent_avg = float(s.tail(6).mean())
    last_ts    = s.index[-1]

    result = []
    for h in range(1, 7):
        future_ts  = last_ts + timedelta(hours=h)
        hour_avg   = float(by_hour.get(future_ts.hour, s.mean()))
        # Blend historical pattern (70 %) with recent level (30 %)
        blended    = 0.7 * hour_avg + 0.3 * recent_avg
        result.append({
            "timestamp":       future_ts.isoformat(),
            "predicted_offers": round(blended, 1),
        })
    return result


# ── 5. Cross-provider spread analysis ────────────────────────────────────────

def analyze_spread(df: pd.DataFrame) -> list[dict]:
    """
    For each GPU class with data in 2+ providers:
      - Compute hourly spread (max_price - min_price across providers)
      - Flag when latest spread > 1.5 × historical mean spread
    """
    records = []

    prov_df = df[df["provider"].isin(SPREAD_PROVIDERS)].copy()

    # Hourly min price per (hour-bucket, provider, gpu_name)
    hourly = (
        prov_df
        .groupby([pd.Grouper(key="captured_at", freq="h"), "provider", "gpu_name"])
        ["price"].min()
        .reset_index()
    )

    for gpu_name, gdf in hourly.groupby("gpu_name"):
        providers = gdf["provider"].unique()
        if len(providers) < 2:
            continue

        pivot = (
            gdf.pivot_table(index="captured_at", columns="provider", values="price")
            .dropna(how="all")
        )
        if len(pivot) < 4:
            continue

        row_spread = (pivot.max(axis=1) - pivot.min(axis=1)).dropna()
        if row_spread.empty:
            continue

        hist_avg   = float(row_spread.mean())
        hist_std   = float(row_spread.std())
        latest_sp  = float(row_spread.iloc[-1])
        threshold  = hist_avg * 1.5
        flagged    = latest_sp > threshold

        latest_row = pivot.iloc[-1].dropna()
        cheapest   = str(latest_row.idxmin()) if not latest_row.empty else None
        priciest   = str(latest_row.idxmax()) if not latest_row.empty else None

        records.append({
            "gpu_name":             gpu_name,
            "providers_compared":   sorted(providers.tolist()),
            "latest_spread":        round(latest_sp, 4),
            "hist_avg_spread":      round(hist_avg, 4),
            "hist_std_spread":      round(hist_std, 4),
            "threshold_1_5x":       round(threshold, 4),
            "spread_alert":         flagged,
            "cheapest_provider":    cheapest,
            "most_expensive_provider": priciest,
            "latest_prices": {
                str(k): round(float(v), 4)
                for k, v in latest_row.items()
            },
        })

    records.sort(key=lambda r: r["latest_spread"], reverse=True)
    return records


# ── 6. Export forecast_windows.json ──────────────────────────────────────────

def build_forecast_windows(
    df: pd.DataFrame,
    generated_at: datetime,
) -> list[dict]:
    """
    One entry per (provider, gpu_class, forecast_hour).
    Schema extends scheduler_windows.json with forecast fields and
    future timestamps, so the scheduler can consume it as a drop-in.
    """
    windows: list[dict] = []

    for (provider, gpu_name), gdf in df.groupby(["provider", "gpu_name"]):
        price_s  = gdf.set_index("captured_at")["price"].sort_index()
        avail_s  = gdf.set_index("captured_at")["offer_count"].sort_index()

        if len(price_s) < 6:
            continue

        fc = forecast_24h(price_s)
        if fc.empty:
            continue

        decomp   = decompose(price_s)
        prices   = price_s.dropna().values
        avail_fc = forecast_availability_6h(avail_s)
        avail_map = {e["timestamp"][:13]: e["predicted_offers"] for e in avail_fc}

        p10 = float(np.percentile(prices, 10))
        p50 = float(np.percentile(prices, 50))
        p90 = float(np.percentile(prices, 90))
        vol = float(np.std(prices) / (np.mean(prices) + 1e-9))
        avail_prob = float((avail_s > 0).mean())
        last_actual = float(price_s.dropna().iloc[-1])

        for _, row in fc.iterrows():
            ts_str   = row["timestamp"].isoformat()
            ts_key   = ts_str[:13]
            h_ahead  = int(
                (row["timestamp"] - price_s.index[-1]).total_seconds() // 3600
            )

            windows.append({
                # ── identity ──────────────────────────────────────────────
                "provider":               provider,
                "gpu_class":              gpu_name,
                # ── timing ────────────────────────────────────────────────
                "forecast_timestamp":     ts_str,
                "generated_at":           generated_at.isoformat(),
                "forecast_hours_ahead":   h_ahead,
                # ── price forecast ────────────────────────────────────────
                "price_forecast":         round(float(row["forecast"]), 4),
                "price_forecast_lower":   round(float(row["lower_95"]),  4),
                "price_forecast_upper":   round(float(row["upper_95"]),  4),
                "last_actual_price":      round(last_actual, 4),
                # ── historical percentiles (scheduler_windows schema) ─────
                "price_per_gpu_hour_p10": round(p10, 4),
                "price_per_gpu_hour_p50": round(p50, 4),
                "price_per_gpu_hour_p90": round(p90, 4),
                "price_volatility":       round(vol, 4),
                "availability_probability": round(avail_prob, 4),
                # ── decomposition signals ─────────────────────────────────
                "trend_slope_per_hour":   round(decomp["trend_slope"],       6),
                "seasonal_strength":      round(decomp["seasonal_strength"], 4),
                # ── availability projection ───────────────────────────────
                "predicted_offers": avail_map.get(ts_key, None),
                "n_price_samples":  len(prices),
            })

    return windows


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_decomp(df: pd.DataFrame, gpu_name: str):
    subset = df[df["gpu_name"] == gpu_name]
    if subset.empty:
        print(f"  No data for {gpu_name}")
        return
    print(f"\n{'─'*68}")
    print(f"  Decomposition — {gpu_name}")
    print(f"{'─'*68}")
    print(f"  {'Provider':<12} {'Trend $/hr':>12} {'Seasonal str':>14} {'Noise std':>11}")
    print(f"{'─'*68}")
    for provider, gdf in subset.groupby("provider"):
        s = gdf.set_index("captured_at")["price"].sort_index()
        if len(s) < 4:
            continue
        d = decompose(s)
        print(
            f"  {provider:<12}"
            f"  {d['trend_slope']:>+10.5f}/hr"
            f"  {d['seasonal_strength']:>12.3f}"
            f"  {d['noise_std']:>10.4f}"
        )
    print(f"{'─'*68}")


def _print_forecast(df: pd.DataFrame, gpu_name: str, provider: str):
    gdf = df[(df["gpu_name"] == gpu_name) & (df["provider"] == provider)]
    s   = gdf.set_index("captured_at")["price"].sort_index()
    fc  = forecast_24h(s)
    if fc.empty:
        print(f"  Insufficient data for {provider}/{gpu_name}")
        return
    last = float(s.dropna().iloc[-1])
    print(f"\n{'─'*62}")
    print(f"  24 h forecast — {provider} / {gpu_name}   (last actual: ${last:.4f})")
    print(f"{'─'*62}")
    print(f"  {'Timestamp (UTC)':<24} {'Forecast':>10} {'Low 95%':>9} {'High 95%':>10}")
    print(f"{'─'*62}")
    for _, row in fc.iterrows():
        ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        print(
            f"  {ts:<24}"
            f"  ${row['forecast']:>8.4f}"
            f"  ${row['lower_95']:>7.4f}"
            f"  ${row['upper_95']:>8.4f}"
        )
    print(f"{'─'*62}")


def _print_availability(df: pd.DataFrame, gpu_name: str):
    subset = df[df["gpu_name"] == gpu_name]
    if subset.empty:
        return
    print(f"\n{'─'*52}")
    print(f"  Availability forecast (next 6 h) — {gpu_name}")
    print(f"{'─'*52}")
    for provider, gdf in subset.groupby("provider"):
        avail = gdf.set_index("captured_at")["offer_count"].sort_index()
        fc    = forecast_availability_6h(avail)
        if not fc:
            continue
        print(f"  {provider}:")
        for e in fc:
            print(f"    {e['timestamp'][:16]}  {e['predicted_offers']:5.1f} offers")
    print(f"{'─'*52}")


def _print_spread(spread_results: list[dict]):
    alerts = [r for r in spread_results if r["spread_alert"]]
    print(f"\n{'═'*76}")
    print(f"  Cross-provider spread analysis — {len(spread_results)} GPU classes, "
          f"{len(alerts)} alert(s)")
    print(f"{'═'*76}")
    print(
        f"  {'GPU':<22} {'Spread':>8} {'HistAvg':>8} "
        f"{'1.5x thr':>9} {'Alert':>6}  {'Cheapest':<10}"
    )
    print(f"{'─'*76}")
    for r in spread_results:
        flag = "YES" if r["spread_alert"] else "—"
        print(
            f"  {r['gpu_name']:<22}"
            f"  ${r['latest_spread']:>6.4f}"
            f"  ${r['hist_avg_spread']:>6.4f}"
            f"  ${r['threshold_1_5x']:>7.4f}"
            f"  {flag:>5}"
            f"  {r['cheapest_provider'] or '—':<10}"
        )
    if alerts:
        print(f"\n  Alert detail:")
        for r in alerts:
            print(f"    {r['gpu_name']}: spread=${r['latest_spread']:.4f}  "
                  f"prices={r['latest_prices']}")
    print(f"{'═'*76}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridSurf price forecaster")
    parser.add_argument("--db",          default=str(DB_PATH))
    parser.add_argument("--out",         default=str(FORECAST_JSON))
    parser.add_argument("--gpu",         default="RTX_4090",
                        help="GPU class for the printed sample (default: RTX_4090)")
    parser.add_argument("--no-export",   action="store_true",
                        help="Print analysis only, skip writing forecast_windows.json")
    parser.add_argument("--spread-only", action="store_true",
                        help="Print cross-provider spread table only")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    print("Loading price history from DB...")
    df = load_price_series(conn)
    print(
        f"  {len(df):,} hourly observations  |  "
        f"{df['gpu_name'].nunique()} GPU classes  |  "
        f"{df['provider'].nunique()} providers  |  "
        f"{df['captured_at'].min().date()} → {df['captured_at'].max().date()}"
    )

    if args.spread_only:
        _print_spread(analyze_spread(df))
    else:
        _print_decomp(df, args.gpu)

        for provider in sorted(df[df["gpu_name"] == args.gpu]["provider"].unique()):
            _print_forecast(df, args.gpu, provider)

        _print_availability(df, args.gpu)
        _print_spread(analyze_spread(df))

    if not args.no_export:
        generated_at = datetime.now(timezone.utc)
        print(f"\nBuilding 24 h forecast windows for all GPU classes...")
        windows = build_forecast_windows(df, generated_at)
        out_path = Path(args.out)
        out_path.write_text(json.dumps(windows, indent=2, default=str))
        gpu_classes = len({w["gpu_class"] for w in windows})
        providers   = len({w["provider"]  for w in windows})
        print(f"  Wrote {len(windows):,} entries → {out_path}")
        print(f"  {gpu_classes} GPU classes × {providers} providers × 24 h")
