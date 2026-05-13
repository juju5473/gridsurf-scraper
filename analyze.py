"""
GridSurf Provider Data Analyzer
================================
Query and analyze the historical GPU pricing data collected by scraper.py.

Usage:
    python analyze.py --summary                    # Price summary across all snapshots
    python analyze.py --gpu H100_PCIE              # Price history for one GPU
    python analyze.py --cheapest --vram 40         # Cheapest offers with >=40GB VRAM
    python analyze.py --windows --budget 2.00      # Windows where price was under $2/hr
    python analyze.py --volatility                 # Price volatility by GPU class
    python analyze.py --export-scheduler           # Export provider windows JSON for scheduler
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "provider_snapshots.db"


def get_conn(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Run scraper.py first to collect data.")
        raise SystemExit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Summary ──────────────────────────────────────────────────────────────────

def cmd_summary(conn: sqlite3.Connection):
    """Show current price landscape across all GPU classes and providers."""
    cur = conn.execute("""
        SELECT
            provider,
            gpu_name,
            COUNT(DISTINCT snapshot_id)           AS snapshots_seen,
            COUNT(*)                              AS total_offers,
            ROUND(MIN(price_per_gpu_hr), 4)       AS all_time_min,
            ROUND(AVG(price_per_gpu_hr), 4)       AS all_time_avg,
            ROUND(MAX(price_per_gpu_hr), 4)       AS all_time_max,
            ROUND(AVG(CASE WHEN interruptible=1
                THEN price_per_gpu_hr END), 4)    AS avg_spot_price,
            ROUND(AVG(CASE WHEN interruptible=0
                THEN price_per_gpu_hr END), 4)    AS avg_ondemand_price,
            MIN(captured_at)                      AS first_seen,
            MAX(captured_at)                      AS last_seen
        FROM provider_offers
        GROUP BY provider, gpu_name
        ORDER BY gpu_name, provider
    """)
    rows = cur.fetchall()
    if not rows:
        print("No data yet. Run scraper.py first.")
        return

    print(f"\n{'═'*100}")
    print("  GridSurf — GPU Price Summary (all collected snapshots)")
    print(f"{'═'*100}")
    print(
        f"  {'Provider':<10} {'GPU':<18} {'Snaps':>6} {'Offers':>7} "
        f"{'Min':>8} {'Avg':>8} {'Max':>8} "
        f"{'Spot avg':>10} {'OD avg':>10}"
    )
    print(f"{'─'*100}")
    for r in rows:
        spot = f"{r['avg_spot_price']:.4f}" if r['avg_spot_price'] else "  —"
        od   = f"{r['avg_ondemand_price']:.4f}" if r['avg_ondemand_price'] else "  —"
        print(
            f"  {r['provider']:<10} {r['gpu_name']:<18} "
            f"{r['snapshots_seen']:>6} {r['total_offers']:>7} "
            f"{r['all_time_min']:>8.4f} {r['all_time_avg']:>8.4f} {r['all_time_max']:>8.4f} "
            f"{spot:>10} {od:>10}"
        )
    print(f"{'═'*100}\n")


# ── GPU price history ─────────────────────────────────────────────────────────

def cmd_gpu_history(conn: sqlite3.Connection, gpu: str):
    """Show price trend over time for a specific GPU class."""
    cur = conn.execute("""
        SELECT
            captured_at,
            provider,
            COUNT(*)                              AS offers,
            ROUND(MIN(price_per_gpu_hr), 4)       AS min_price,
            ROUND(AVG(price_per_gpu_hr), 4)       AS avg_price,
            ROUND(MAX(price_per_gpu_hr), 4)       AS max_price
        FROM provider_offers
        WHERE UPPER(gpu_name) LIKE UPPER(?)
        GROUP BY DATE(captured_at), provider
        ORDER BY captured_at DESC
        LIMIT 100
    """, (f"%{gpu}%",))
    rows = cur.fetchall()
    if not rows:
        print(f"No data for GPU: {gpu}")
        return

    print(f"\n{'─'*70}")
    print(f"  Price history: {gpu}")
    print(f"{'─'*70}")
    print(f"  {'Timestamp':<28} {'Provider':<10} {'Offers':>6} {'Min':>8} {'Avg':>8} {'Max':>8}")
    print(f"{'─'*70}")
    for r in rows:
        ts = r['captured_at'][:19].replace("T", " ")
        print(
            f"  {ts:<28} {r['provider']:<10} {r['offers']:>6} "
            f"{r['min_price']:>8.4f} {r['avg_price']:>8.4f} {r['max_price']:>8.4f}"
        )
    print(f"{'─'*70}\n")


# ── Cheapest offers ───────────────────────────────────────────────────────────

def cmd_cheapest(conn: sqlite3.Connection, min_vram: float = 0):
    """Show the cheapest current offers meeting a VRAM threshold."""
    cur = conn.execute("""
        SELECT
            o.provider,
            o.gpu_name,
            o.gpu_name_raw,
            o.gpu_count,
            o.vram_gb,
            o.price_per_gpu_hr,
            o.price_total_hr,
            o.interruptible,
            o.reliability_score,
            o.geography,
            o.captured_at
        FROM provider_offers o
        INNER JOIN (
            SELECT MAX(captured_at) AS latest FROM provider_offers
        ) latest ON o.captured_at >= latest.latest
        WHERE o.vram_gb >= ?
        ORDER BY o.price_per_gpu_hr ASC
        LIMIT 30
    """, (min_vram,))
    rows = cur.fetchall()
    if not rows:
        print(f"No current offers with VRAM >= {min_vram}GB")
        return

    print(f"\n{'─'*85}")
    print(f"  Cheapest current offers (VRAM >= {min_vram}GB)")
    print(f"{'─'*85}")
    print(
        f"  {'Provider':<10} {'GPU':<18} {'VRAM':>6} {'$/gpu-hr':>9} "
        f"{'Type':<14} {'Reliability':>11} {'Geo':<20}"
    )
    print(f"{'─'*85}")
    for r in rows:
        offer_type = "interruptible" if r['interruptible'] else "on-demand"
        rel = f"{r['reliability_score']:.2f}" if r['reliability_score'] else "  —"
        geo = (r['geography'] or "unknown")[:18]
        print(
            f"  {r['provider']:<10} {r['gpu_name']:<18} "
            f"{r['vram_gb']:>5.0f}G {r['price_per_gpu_hr']:>9.4f} "
            f"{offer_type:<14} {rel:>11} {geo:<20}"
        )
    print(f"{'─'*85}\n")


# ── Budget windows ────────────────────────────────────────────────────────────

def cmd_windows(conn: sqlite3.Connection, budget: float):
    """
    Show historical windows where GPU was available at or below a bid price.
    This is the core input GridSurf's scheduler needs.
    """
    cur = conn.execute("""
        SELECT
            provider,
            gpu_name,
            COUNT(*)                              AS qualifying_offers,
            COUNT(DISTINCT DATE(captured_at))     AS days_with_offers,
            ROUND(MIN(price_per_gpu_hr), 4)       AS cheapest_seen,
            ROUND(AVG(price_per_gpu_hr), 4)       AS avg_when_cheap,
            MIN(captured_at)                      AS first_window,
            MAX(captured_at)                      AS last_window
        FROM provider_offers
        WHERE price_per_gpu_hr <= ?
          AND interruptible = 1
        GROUP BY provider, gpu_name
        ORDER BY qualifying_offers DESC
    """, (budget,))
    rows = cur.fetchall()
    if not rows:
        print(f"No interruptible offers found at or below ${budget:.2f}/hr")
        return

    print(f"\n{'─'*85}")
    print(f"  Budget windows: interruptible offers ≤ ${budget:.4f}/gpu-hr")
    print(f"{'─'*85}")
    print(
        f"  {'Provider':<10} {'GPU':<18} {'Offers':>8} "
        f"{'Days':>5} {'Cheapest':>9} {'Avg when cheap':>15}"
    )
    print(f"{'─'*85}")
    for r in rows:
        print(
            f"  {r['provider']:<10} {r['gpu_name']:<18} "
            f"{r['qualifying_offers']:>8} {r['days_with_offers']:>5} "
            f"{r['cheapest_seen']:>9.4f} {r['avg_when_cheap']:>15.4f}"
        )
    print(f"{'─'*85}\n")


# ── Volatility ────────────────────────────────────────────────────────────────

def cmd_volatility(conn: sqlite3.Connection):
    """
    Measure price volatility by GPU class.
    High volatility = harder to predict, wider confidence intervals needed.
    """
    cur = conn.execute("""
        SELECT
            provider,
            gpu_name,
            COUNT(DISTINCT snapshot_id)                       AS snapshots,
            ROUND(AVG(price_per_gpu_hr), 4)                   AS mean_price,
            ROUND(
                SQRT(AVG(price_per_gpu_hr * price_per_gpu_hr)
                     - AVG(price_per_gpu_hr) * AVG(price_per_gpu_hr))
            , 4)                                              AS std_dev,
            ROUND(MIN(price_per_gpu_hr), 4)                   AS p0,
            ROUND(MAX(price_per_gpu_hr), 4)                   AS p100
        FROM provider_offers
        WHERE interruptible = 1
        GROUP BY provider, gpu_name
        HAVING snapshots >= 3
        ORDER BY std_dev DESC
    """)
    rows = cur.fetchall()
    if not rows:
        print("Not enough data for volatility analysis (need ≥3 snapshots).")
        return

    print(f"\n{'─'*80}")
    print("  Price volatility — interruptible/spot offers")
    print("  (higher std_dev = more volatile = wider scheduler confidence intervals)")
    print(f"{'─'*80}")
    print(
        f"  {'Provider':<10} {'GPU':<18} {'Snaps':>6} "
        f"{'Mean':>8} {'StdDev':>8} {'Min':>8} {'Max':>8}"
    )
    print(f"{'─'*80}")
    for r in rows:
        print(
            f"  {r['provider']:<10} {r['gpu_name']:<18} "
            f"{r['snapshots']:>6} {r['mean_price']:>8.4f} "
            f"{r['std_dev']:>8.4f} {r['p0']:>8.4f} {r['p100']:>8.4f}"
        )
    print(f"{'─'*80}\n")


# ── Scheduler export ──────────────────────────────────────────────────────────

def cmd_export_scheduler(conn: sqlite3.Connection, output_path: Path = None):
    """
    Export provider windows in the format expected by the GridSurf scheduler.
    This is the JSON interface described in the product report (section 6).
    """
    cur = conn.execute("""
        SELECT
            snapshot_id || '_' || provider || '_' || gpu_name || '_' || ROWID
                                                      AS window_id,
            provider,
            gpu_name                                  AS gpu_class,
            captured_at                               AS start_time,
            price_per_gpu_hr                          AS price_per_gpu_hour_p50,
            interruptible,
            reliability_score,
            geography
        FROM provider_offers
        ORDER BY captured_at DESC
        LIMIT 5000
    """)
    rows = cur.fetchall()

    windows = []
    for r in rows:
        windows.append({
            "window_id":               r["window_id"],
            "provider":                r["provider"],
            "gpu_class":               r["gpu_class"],
            "start_time":              r["start_time"],
            "price_per_gpu_hour_p50":  r["price_per_gpu_hour_p50"],
            "price_per_gpu_hour_p90":  round(float(r["price_per_gpu_hour_p50"] or 0) * 1.25, 6),
            "availability_probability": float(r["reliability_score"] or 0.6),
            "interruption_probability": 0.15 if r["interruptible"] else 0.02,
        })

    out = output_path or Path(__file__).parent / "data" / "scheduler_windows.json"
    with open(out, "w") as f:
        json.dump(windows, f, indent=2)

    print(f"Exported {len(windows)} provider windows → {out}")


# ── DB stats ──────────────────────────────────────────────────────────────────

def cmd_db_stats(conn: sqlite3.Connection):
    stats = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM snapshots)       AS total_snapshots,
            (SELECT COUNT(*) FROM provider_offers) AS total_offers,
            (SELECT MIN(started_at) FROM snapshots) AS oldest_snapshot,
            (SELECT MAX(started_at) FROM snapshots) AS latest_snapshot
    """).fetchone()
    print(f"\n  Database: {DB_PATH}")
    print(f"  Snapshots collected : {stats['total_snapshots']}")
    print(f"  Total offer rows    : {stats['total_offers']}")
    print(f"  Oldest snapshot     : {stats['oldest_snapshot'] or 'none'}")
    print(f"  Latest snapshot     : {stats['latest_snapshot'] or 'none'}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridSurf provider data analyzer")
    parser.add_argument("--summary",          action="store_true", help="Price summary across all data")
    parser.add_argument("--gpu",              type=str,            help="Price history for a GPU class (e.g. H100_PCIE)")
    parser.add_argument("--cheapest",         action="store_true", help="Show cheapest current offers")
    parser.add_argument("--vram",             type=float, default=0, help="Minimum VRAM filter in GB (use with --cheapest)")
    parser.add_argument("--windows",          action="store_true", help="Show historical windows below a budget")
    parser.add_argument("--budget",           type=float, default=2.0, help="Max price threshold for --windows")
    parser.add_argument("--volatility",       action="store_true", help="Price volatility analysis")
    parser.add_argument("--export-scheduler", action="store_true", help="Export windows JSON for scheduler")
    parser.add_argument("--stats",            action="store_true", help="Show database stats")
    parser.add_argument("--db",               type=str, default=str(DB_PATH), help="Path to SQLite database")
    args = parser.parse_args()

    conn = get_conn(Path(args.db))

    if args.stats:
        cmd_db_stats(conn)
    if args.summary:
        cmd_summary(conn)
    if args.gpu:
        cmd_gpu_history(conn, args.gpu)
    if args.cheapest:
        cmd_cheapest(conn, args.min_vram if hasattr(args, "min_vram") else args.vram)
    if args.windows:
        cmd_windows(conn, args.budget)
    if args.volatility:
        cmd_volatility(conn)
    if args.export_scheduler:
        cmd_export_scheduler(conn)

    if not any([args.stats, args.summary, args.gpu, args.cheapest,
                args.windows, args.volatility, args.export_scheduler]):
        cmd_db_stats(conn)
        cmd_summary(conn)


if __name__ == "__main__":
    main()
