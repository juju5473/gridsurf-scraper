"""
GridSurf Price Model
====================
Reads provider_snapshots.db and produces scheduler-ready provider windows
with price percentiles, availability scores, and interruption probabilities.

Usage:
    python price_model.py                          # Print summary table
    python price_model.py --budget 0.50            # Filter by max $/GPU/hr
    python price_model.py --budget 0.50 --vram 24  # Also filter by min VRAM
    python price_model.py --export                 # Write scheduler_windows.json
"""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

DB_PATH      = Path(__file__).parent / "data" / "provider_snapshots.db"
WINDOWS_PATH = Path(__file__).parent / "data" / "scheduler_windows.json"

# Provider-level interruption priors (spot/preemption rate estimates).
# Vast is a peer-to-peer marketplace where hosts can reclaim machines;
# RunPod community-cloud pods can be interrupted; Spheron and Akash are
# relatively stable decentralised leases.
INTERRUPTION_PRIORS = {
    "vast":    0.20,
    "runpod":  0.05,
    "akash":   0.10,
    "spheron": 0.02,
}

DEFAULT_INTERRUPTION = 0.15  # fallback for unknown providers


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── 1. Price statistics ───────────────────────────────────────────────────────

def get_price_stats(gpu_name: str, provider: str) -> dict:
    """
    Return p10 / p50 / p90 prices and a volatility coefficient (std/mean)
    for a given GPU + provider combination.

    Returns None if fewer than 3 price observations exist.
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT price_per_gpu_hr
        FROM   provider_offers
        WHERE  gpu_name  = ?
          AND  provider  = ?
          AND  price_per_gpu_hr IS NOT NULL
          AND  price_per_gpu_hr > 0
        ORDER  BY captured_at
    """, (gpu_name, provider)).fetchall()
    conn.close()

    prices = [r["price_per_gpu_hr"] for r in rows]
    if len(prices) < 3:
        return None

    prices_sorted = sorted(prices)
    n = len(prices_sorted)

    def percentile(p):
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return prices_sorted[lo] + (idx - lo) * (prices_sorted[hi] - prices_sorted[lo])

    mean = statistics.mean(prices)
    std  = statistics.stdev(prices) if len(prices) > 1 else 0.0

    return {
        "gpu_name":    gpu_name,
        "provider":    provider,
        "n_samples":   n,
        "p10":         round(percentile(10), 6),
        "p50":         round(percentile(50), 6),
        "p90":         round(percentile(90), 6),
        "mean":        round(mean, 6),
        "volatility":  round(std / mean, 4) if mean > 0 else 0.0,
    }


# ── 2. Availability score ─────────────────────────────────────────────────────

def get_availability_score(gpu_name: str, provider: str) -> float:
    """
    Return a 0-1 score: fraction of total snapshots in which this
    GPU + provider combination had at least one offer.

    A score of 1.0 means the GPU appeared in every snapshot ever taken;
    0.0 means it never appeared (shouldn't happen if called for a known GPU).
    """
    conn = _connect()

    total_snaps = conn.execute(
        "SELECT COUNT(*) FROM snapshots WHERE finished_at IS NOT NULL"
    ).fetchone()[0]

    snaps_with_offer = conn.execute("""
        SELECT COUNT(DISTINCT snapshot_id)
        FROM   provider_offers
        WHERE  gpu_name = ?
          AND  provider = ?
          AND  price_per_gpu_hr IS NOT NULL
    """, (gpu_name, provider)).fetchone()[0]

    conn.close()

    if total_snaps == 0:
        return 0.0
    return round(snaps_with_offer / total_snaps, 4)


# ── 3. Interruption probability ───────────────────────────────────────────────

def get_interruption_probability(provider: str) -> float:
    """
    Return the estimated per-job interruption probability for a provider.
    Blends the static prior with the observed spot fraction from the DB
    when enough data exists (≥10 rows), otherwise returns the prior directly.
    """
    prior = INTERRUPTION_PRIORS.get(provider.lower(), DEFAULT_INTERRUPTION)

    conn = _connect()
    row = conn.execute("""
        SELECT
            COUNT(*)                          AS total,
            SUM(CASE WHEN interruptible = 1
                     THEN 1 ELSE 0 END)       AS spot_count
        FROM provider_offers
        WHERE provider = ?
          AND price_per_gpu_hr IS NOT NULL
    """, (provider,)).fetchone()
    conn.close()

    total, spot_count = row["total"], row["spot_count"]
    if total < 10:
        return prior

    observed = spot_count / total
    # Blend: 70 % prior, 30 % observed (prior is domain knowledge)
    return round(0.70 * prior + 0.30 * observed, 4)


# ── 4. Export provider windows ────────────────────────────────────────────────

def export_provider_windows(
    budget: float | None = None,
    min_vram: float | None = None,
    path: Path = WINDOWS_PATH,
) -> list[dict]:
    """
    Build scheduler windows for every GPU × provider combination that has
    enough price history, optionally filtered by budget (max p50) and
    minimum VRAM.  Writes results to path and returns the list.
    """
    conn = _connect()
    query = """
        SELECT DISTINCT gpu_name, provider, MAX(vram_gb) AS vram_gb
        FROM   provider_offers
        WHERE  price_per_gpu_hr IS NOT NULL
    """
    params = []
    if min_vram is not None:
        query += " AND vram_gb >= ?"
        params.append(min_vram)
    query += " GROUP BY gpu_name, provider"

    combos = conn.execute(query, params).fetchall()
    conn.close()

    windows = []
    for row in combos:
        gpu_name = row["gpu_name"]
        provider = row["provider"]
        vram_gb  = row["vram_gb"]

        stats = get_price_stats(gpu_name, provider)
        if stats is None:
            continue  # not enough history

        if budget is not None and stats["p50"] > budget:
            continue

        window = {
            "provider":                   provider,
            "gpu_class":                  gpu_name,
            "vram_gb":                    vram_gb,
            "price_per_gpu_hour_p10":     stats["p10"],
            "price_per_gpu_hour_p50":     stats["p50"],
            "price_per_gpu_hour_p90":     stats["p90"],
            "price_volatility":           stats["volatility"],
            "availability_probability":   get_availability_score(gpu_name, provider),
            "interruption_probability":   get_interruption_probability(provider),
            "n_price_samples":            stats["n_samples"],
        }
        windows.append(window)

    windows.sort(key=lambda w: (w["gpu_class"], w["price_per_gpu_hour_p50"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(windows, f, indent=2)

    print(f"Exported {len(windows)} provider windows -> {path}")
    return windows


# ── 5. Summary printer ────────────────────────────────────────────────────────

def _print_summary(windows: list[dict]):
    if not windows:
        print("No windows to display.")
        return

    W = 96
    print(f"\n{'─' * W}")
    print(f"  {'Provider':<10} {'GPU':<18} {'VRAM':>5} {'p50 $/hr':>9} {'p90 $/hr':>9} "
          f"{'Avail':>6} {'Intr':>6} {'Samples':>8}")
    print(f"{'─' * W}")
    for w in windows:
        print(
            f"  {w['provider']:<10} {w['gpu_class']:<18} "
            f"{(str(int(w['vram_gb'])) + 'G') if w['vram_gb'] else '?':>5} "
            f"{w['price_per_gpu_hour_p50']:>9.4f} "
            f"{w['price_per_gpu_hour_p90']:>9.4f} "
            f"{w['availability_probability']:>6.2f} "
            f"{w['interruption_probability']:>6.2f} "
            f"{w['n_price_samples']:>8}"
        )
    print(f"{'─' * W}")
    print(f"  {len(windows)} windows\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridSurf price model")
    parser.add_argument("--budget",  type=float, default=None,
                        help="Max p50 price per GPU/hr (e.g. 0.50)")
    parser.add_argument("--vram",    type=float, default=None,
                        help="Min VRAM in GB (e.g. 24)")
    parser.add_argument("--export",  action="store_true",
                        help="Write scheduler_windows.json")
    args = parser.parse_args()

    windows = export_provider_windows(
        budget=args.budget,
        min_vram=args.vram,
    ) if args.export else _build_windows(args.budget, args.vram)

    _print_summary(windows)


def _build_windows(budget=None, min_vram=None) -> list[dict]:
    """Like export_provider_windows but without writing to disk."""
    conn = _connect()
    query = """
        SELECT DISTINCT gpu_name, provider, MAX(vram_gb) AS vram_gb
        FROM   provider_offers
        WHERE  price_per_gpu_hr IS NOT NULL
    """
    params = []
    if min_vram is not None:
        query += " AND vram_gb >= ?"
        params.append(min_vram)
    query += " GROUP BY gpu_name, provider"

    combos = conn.execute(query, params).fetchall()
    conn.close()

    windows = []
    for row in combos:
        gpu_name = row["gpu_name"]
        provider = row["provider"]
        vram_gb  = row["vram_gb"]

        stats = get_price_stats(gpu_name, provider)
        if stats is None:
            continue

        if budget is not None and stats["p50"] > budget:
            continue

        windows.append({
            "provider":                   provider,
            "gpu_class":                  gpu_name,
            "vram_gb":                    vram_gb,
            "price_per_gpu_hour_p10":     stats["p10"],
            "price_per_gpu_hour_p50":     stats["p50"],
            "price_per_gpu_hour_p90":     stats["p90"],
            "price_volatility":           stats["volatility"],
            "availability_probability":   get_availability_score(gpu_name, provider),
            "interruption_probability":   get_interruption_probability(provider),
            "n_price_samples":            stats["n_samples"],
        })

    windows.sort(key=lambda w: (w["gpu_class"], w["price_per_gpu_hour_p50"]))
    return windows


if __name__ == "__main__":
    main()
