"""
GridSurf Price Forecaster
=========================
Time-series decomposition, 24-hour price forecasts, availability
projections, and cross-provider spread alerts — all from the local DB.

Feature set (no data leakage — all lags/rolling computed from t-1 and earlier):
  Time     : hour_sin, hour_cos, dow_sin, dow_cos, t (linear trend)
  Lags     : price_lag_1h, price_lag_6h, price_lag_24h, price_lag_168h
  Rolling  : rolling_mean_24h, rolling_std_24h, rolling_mean_7d
  Supply   : offer_count, offer_count_lag_1h

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

# Canonical feature order — used for training and walk-forward prediction
FEATURE_NAMES = [
    "t",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "price_lag_1h",
    "price_lag_6h",
    "price_lag_24h",
    "price_lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_7d",
    "offer_count",
    "offer_count_lag_1h",
]


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

    trend_vals = trend.ffill().values
    slope = float(np.polyfit(np.arange(len(trend_vals)), trend_vals, 1)[0])

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


# ── 3. Feature engineering ────────────────────────────────────────────────────

def build_features(
    price: pd.Series,
    offer_count: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Construct the full feature matrix for model training.

    All lag and rolling features are shifted so that row t only ever
    sees data from t-1 and earlier — no same-row or future leakage.

    Columns match FEATURE_NAMES exactly.
    """
    s = price.copy()

    # Align offer_count to price index; fill gaps with forward-fill then 0
    if offer_count is not None:
        oc = offer_count.reindex(s.index).ffill().fillna(0.0)
    else:
        oc = pd.Series(np.nan, index=s.index)

    df = pd.DataFrame(index=s.index)

    # ── Time features ─────────────────────────────────────────────────────────
    df["t"]        = np.arange(len(s), dtype=float)
    df["hour_sin"] = np.sin(2 * np.pi * s.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * s.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * s.index.dayofweek.astype(float) / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * s.index.dayofweek.astype(float) / 7)

    # ── Lag features — shift(k) means "price k hours ago", no leakage ─────────
    df["price_lag_1h"]   = s.shift(1)
    df["price_lag_6h"]   = s.shift(6)
    df["price_lag_24h"]  = s.shift(24)
    df["price_lag_168h"] = s.shift(168)

    # ── Rolling statistics — shift(1) before rolling excludes the current row ──
    shifted = s.shift(1)
    df["rolling_mean_24h"] = shifted.rolling(24,  min_periods=6).mean()
    df["rolling_std_24h"]  = shifted.rolling(24,  min_periods=6).std().fillna(0.0)
    df["rolling_mean_7d"]  = shifted.rolling(168, min_periods=24).mean()

    # ── Supply features ───────────────────────────────────────────────────────
    df["offer_count"]        = oc.values
    df["offer_count_lag_1h"] = oc.shift(1).values

    return df[FEATURE_NAMES]


# ── 4. 24-hour price forecast ─────────────────────────────────────────────────

def forecast_24h(
    price: pd.Series,
    offer_count: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Ridge regression trained on lag + rolling + time + supply features.

    Training uses only rows where all features are available (NaN rows at
    the head of the series are dropped, not imputed).

    Prediction uses walk-forward: each forecast step appends its own
    prediction to the price buffer so that subsequent lag features
    reflect previously predicted values rather than stale actuals.

    Returns
    -------
    forecast_df : pd.DataFrame  — columns: timestamp, forecast, lower_95, upper_95
    feat_importance : dict      — feature → normalized |coefficient|  (sums to 1)
    """
    s = price.dropna().sort_index()
    if len(s) < 30:
        return pd.DataFrame(), {}

    feat_df = build_features(s, offer_count)

    # Drop the leading rows that lack full lag/rolling history
    valid = feat_df.dropna()
    if len(valid) < 10:
        return pd.DataFrame(), {}

    y_train = s.loc[valid.index].values
    X_train = valid.values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_sc, y_train)

    resid_std = float(np.std(y_train - model.predict(X_sc)))

    # Feature importance: normalized absolute standardized coefficients
    # (valid because features are all on the same scale after StandardScaler)
    abs_coef   = np.abs(model.coef_)
    importance = abs_coef / (abs_coef.sum() + 1e-9)
    feat_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, importance)
    }

    # ── Walk-forward prediction ───────────────────────────────────────────────
    # Seed the price buffer with the last 168 actuals (1 week)
    price_buffer = list(s.values[-168:])

    oc_val      = 0.0
    oc_lag_val  = 0.0
    if offer_count is not None:
        oc_aligned = offer_count.reindex(s.index).ffill().fillna(0.0)
        if len(oc_aligned) >= 2:
            oc_val     = float(oc_aligned.iloc[-1])
            oc_lag_val = float(oc_aligned.iloc[-2])
        elif len(oc_aligned) == 1:
            oc_val = oc_lag_val = float(oc_aligned.iloc[-1])

    last_ts   = s.index[-1]
    future_ts = pd.date_range(
        start  = last_ts + pd.Timedelta(hours=1),
        periods= 24,
        freq   = "h",
        tz     = last_ts.tzinfo,
    )

    preds = []
    for i, ts in enumerate(future_ts):
        buf = price_buffer  # alias for readability
        n   = len(buf)

        row = [
            float(len(s) + i),                          # t
            np.sin(2 * np.pi * ts.hour / 24),           # hour_sin
            np.cos(2 * np.pi * ts.hour / 24),           # hour_cos
            np.sin(2 * np.pi * ts.dayofweek / 7),       # dow_sin
            np.cos(2 * np.pi * ts.dayofweek / 7),       # dow_cos
            buf[-1]   if n >= 1   else buf[0],           # price_lag_1h
            buf[-6]   if n >= 6   else buf[0],           # price_lag_6h
            buf[-24]  if n >= 24  else buf[0],           # price_lag_24h
            buf[-168] if n >= 168 else buf[0],           # price_lag_168h
            float(np.mean(buf[-24:]  if n >= 6  else buf)),  # rolling_mean_24h
            float(np.std(buf[-24:]   if n >= 6  else buf)),  # rolling_std_24h
            float(np.mean(buf[-168:] if n >= 24 else buf)),  # rolling_mean_7d
            oc_val,                                      # offer_count
            oc_lag_val,                                  # offer_count_lag_1h
        ]

        X_row = scaler.transform(np.array([row]))
        pred  = float(max(model.predict(X_row)[0], 0.0))
        preds.append(pred)

        # Append prediction so the next step's lags reflect it
        price_buffer.append(pred)
        oc_lag_val = oc_val  # shift offer_count forward by one step

    preds_arr = np.array(preds)
    ci        = 1.96 * resid_std

    return pd.DataFrame({
        "timestamp": future_ts,
        "forecast":  preds_arr,
        "lower_95":  np.maximum(preds_arr - ci, 0.0),
        "upper_95":  preds_arr + ci,
    }), feat_importance


# ── 5. Availability forecast ──────────────────────────────────────────────────

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
        blended    = 0.7 * hour_avg + 0.3 * recent_avg
        result.append({
            "timestamp":        future_ts.isoformat(),
            "predicted_offers": round(blended, 1),
        })
    return result


# ── 6. Cross-provider spread analysis ────────────────────────────────────────

def analyze_spread(df: pd.DataFrame) -> list[dict]:
    """
    For each GPU class with data in 2+ providers:
      - Compute hourly spread (max_price - min_price across providers)
      - Flag when latest spread > 1.5 × historical mean spread
    """
    records = []

    prov_df = df[df["provider"].isin(SPREAD_PROVIDERS)].copy()

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

        hist_avg  = float(row_spread.mean())
        hist_std  = float(row_spread.std())
        latest_sp = float(row_spread.iloc[-1])
        threshold = hist_avg * 1.5
        flagged   = latest_sp > threshold

        latest_row = pivot.iloc[-1].dropna()
        cheapest   = str(latest_row.idxmin()) if not latest_row.empty else None
        priciest   = str(latest_row.idxmax()) if not latest_row.empty else None

        records.append({
            "gpu_name":                gpu_name,
            "providers_compared":      sorted(providers.tolist()),
            "latest_spread":           round(latest_sp, 4),
            "hist_avg_spread":         round(hist_avg,  4),
            "hist_std_spread":         round(hist_std,  4),
            "threshold_1_5x":          round(threshold, 4),
            "spread_alert":            flagged,
            "cheapest_provider":       cheapest,
            "most_expensive_provider": priciest,
            "latest_prices": {
                str(k): round(float(v), 4)
                for k, v in latest_row.items()
            },
        })

    records.sort(key=lambda r: r["latest_spread"], reverse=True)
    return records


# ── 7. Export forecast_windows.json ──────────────────────────────────────────

def build_forecast_windows(
    df: pd.DataFrame,
    generated_at: datetime,
) -> list[dict]:
    """
    One entry per (provider, gpu_class, forecast_hour).
    Extends scheduler_windows.json schema with forecast fields and the
    top-3 most important features for that GPU/provider combination.
    """
    windows: list[dict] = []

    for (provider, gpu_name), gdf in df.groupby(["provider", "gpu_name"]):
        price_s = gdf.set_index("captured_at")["price"].sort_index()
        avail_s = gdf.set_index("captured_at")["offer_count"].sort_index()

        if len(price_s) < 30:
            continue

        fc, feat_imp = forecast_24h(price_s, avail_s)
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
        avail_prob  = float((avail_s > 0).mean())
        last_actual = float(price_s.dropna().iloc[-1])

        # Top-3 features by importance for this model instance
        top_features = sorted(feat_imp, key=feat_imp.get, reverse=True)[:3]

        for _, row in fc.iterrows():
            ts_str  = row["timestamp"].isoformat()
            ts_key  = ts_str[:13]
            h_ahead = int(
                (row["timestamp"] - price_s.index[-1]).total_seconds() // 3600
            )

            windows.append({
                "provider":               provider,
                "gpu_class":              gpu_name,
                "forecast_timestamp":     ts_str,
                "generated_at":           generated_at.isoformat(),
                "forecast_hours_ahead":   h_ahead,
                "price_forecast":         round(float(row["forecast"]), 4),
                "price_forecast_lower":   round(float(row["lower_95"]),  4),
                "price_forecast_upper":   round(float(row["upper_95"]),  4),
                "last_actual_price":      round(last_actual, 4),
                "price_per_gpu_hour_p10": round(p10, 4),
                "price_per_gpu_hour_p50": round(p50, 4),
                "price_per_gpu_hour_p90": round(p90, 4),
                "price_volatility":       round(vol, 4),
                "availability_probability": round(avail_prob, 4),
                "trend_slope_per_hour":   round(decomp["trend_slope"],       6),
                "seasonal_strength":      round(decomp["seasonal_strength"], 4),
                "predicted_offers":       avail_map.get(ts_key, None),
                "n_price_samples":        len(prices),
                "top_features":           top_features,
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


def _print_feature_importance(feat_importance: dict):
    if not feat_importance:
        return
    print(f"\n  Feature importance (normalized |coef| after StandardScaler):")
    print(f"  {'Feature':<22} {'Importance':>11}  {'Bar'}")
    print(f"  {'─'*54}")
    sorted_feats = sorted(feat_importance.items(), key=lambda x: -x[1])
    for name, imp in sorted_feats:
        bar = "█" * int(imp * 40)
        print(f"  {name:<22}  {imp:>9.4f}  {bar}")


def _print_forecast(df: pd.DataFrame, gpu_name: str, provider: str):
    gdf = df[(df["gpu_name"] == gpu_name) & (df["provider"] == provider)]
    s   = gdf.set_index("captured_at")["price"].sort_index()
    oc  = gdf.set_index("captured_at")["offer_count"].sort_index()
    fc, feat_imp = forecast_24h(s, oc)
    if fc.empty:
        print(f"  Insufficient data for {provider}/{gpu_name}")
        return
    last = float(s.dropna().iloc[-1])
    n_train = len(build_features(s.dropna().sort_index(), oc).dropna())
    print(f"\n{'─'*62}")
    print(f"  24 h forecast — {provider} / {gpu_name}   (last actual: ${last:.4f})")
    print(f"  Training rows: {n_train}  |  Features: {len(FEATURE_NAMES)}")
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
    _print_feature_importance(feat_imp)
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
    print(f"  Feature set ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")

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
