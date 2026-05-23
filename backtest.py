"""
GridSurf AWS Spot Backtest
==========================
Simulates running GPU jobs on spot instances using historical price data.

Usage:
    python backtest.py                          # Sample backtest with defaults
    python backtest.py --budget 1.0 --gpu g5.xlarge --hours 4
    python backtest.py --budget 0.5 --all-gpus
"""

import argparse
import glob
import io
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import zstandard

DATA_DIR = Path.home() / "gridsurf_data" / "aws_spot_history"

GPU_PREFIXES = ("p2.", "p3.", "p4.", "g3.", "g4.", "g5.", "g6.")


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class PricePoint:
    az: str
    instance_type: str
    price: float
    timestamp: datetime


def _iter_file(path: str) -> Iterator[PricePoint]:
    with open(path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as sr:
            reader = io.TextIOWrapper(sr)
            for line in reader:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                az, itype, product, price_str, ts_str = parts[:5]
                if product not in ("Linux/UNIX", ""):
                    continue
                if not itype.startswith(GPU_PREFIXES):
                    continue
                try:
                    price = float(price_str)
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    yield PricePoint(az, itype, price, ts)
                except (ValueError, TypeError):
                    continue


def load_prices(gpu_type: str | None = None) -> list[PricePoint]:
    """Load all price points, optionally filtered to one instance type."""
    files = sorted(glob.glob(str(DATA_DIR / "*.tsv.zst")))
    if not files:
        raise FileNotFoundError(f"No .tsv.zst files found in {DATA_DIR}")

    points: list[PricePoint] = []
    for path in files:
        for p in _iter_file(path):
            if gpu_type is None or p.instance_type == gpu_type:
                points.append(p)

    points.sort(key=lambda p: p.timestamp)
    return points


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    gpu_type: str
    budget: float
    job_duration_hours: float
    # counts
    total_windows: int = 0
    completed_jobs: int = 0
    interrupted_jobs: int = 0
    # time accumulators (hours)
    total_wait_hours: float = 0.0
    total_runtime_hours: float = 0.0
    total_cost: float = 0.0
    # per-job completion times (hours from submission to done)
    completion_times: list[float] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        total = self.completed_jobs + self.interrupted_jobs
        return self.completed_jobs / total if total else 0.0

    @property
    def interruption_rate(self) -> float:
        total = self.completed_jobs + self.interrupted_jobs
        return self.interrupted_jobs / total if total else 0.0

    @property
    def avg_wait_hours(self) -> float:
        total = self.completed_jobs + self.interrupted_jobs
        return self.total_wait_hours / total if total else 0.0

    @property
    def avg_actual_cost(self) -> float:
        return self.total_cost / self.completed_jobs if self.completed_jobs else 0.0

    @property
    def p50_completion_hours(self) -> float:
        if not self.completion_times:
            return float("nan")
        s = sorted(self.completion_times)
        return s[len(s) // 2]

    @property
    def p90_completion_hours(self) -> float:
        if not self.completion_times:
            return float("nan")
        s = sorted(self.completion_times)
        return s[int(len(s) * 0.9)]


def run_backtest(
    budget: float,
    gpu_type: str,
    job_duration_hours: float = 1.0,
    az: str | None = None,
) -> BacktestResult:
    """
    Simulate job scheduling against historical spot prices.

    Each unique (az, timestamp) price observation is treated as a 1-hour
    window. Windows are replayed in order. A job is submitted, waits until
    the spot price is at or below budget, then runs. If the price rises above
    budget mid-job, it is interrupted and must restart from checkpoint
    (modelled as restarting from scratch — pessimistic assumption).
    """
    result = BacktestResult(
        gpu_type=gpu_type,
        budget=budget,
        job_duration_hours=job_duration_hours,
    )

    points = load_prices(gpu_type)
    if not points:
        print(f"  No data found for {gpu_type}")
        return result

    # Build hourly price series per AZ: list of (timestamp, price)
    az_series: dict[str, list[tuple[datetime, float]]] = {}
    for p in points:
        if az and p.az != az:
            continue
        az_series.setdefault(p.az, []).append((p.timestamp, p.price))

    if not az_series:
        print(f"  No data for az={az}")
        return result

    # Pick the AZ with the most data
    best_az = max(az_series, key=lambda k: len(az_series[k]))
    series = az_series[best_az]

    result.total_windows = len(series)

    # Sliding window simulation: replay every starting hour as a job submission
    # Sample every 24th window to keep runtime reasonable on large datasets
    step = max(1, len(series) // 500)

    for start_idx in range(0, len(series) - 1, step):
        job_submitted = series[start_idx][0]
        hours_into_job = 0.0
        wait_hours = 0.0
        cost = 0.0
        interrupted = False
        idx = start_idx

        while idx < len(series):
            ts, price = series[idx]

            if price > budget:
                if hours_into_job > 0:
                    # Interrupted mid-job — restart from checkpoint
                    result.interrupted_jobs += 1
                    result.total_cost += cost
                    interrupted = True
                    hours_into_job = 0.0
                    cost = 0.0
                wait_hours += 1.0
            else:
                hours_into_job += 1.0
                cost += price

                if hours_into_job >= job_duration_hours:
                    # Job completed
                    result.completed_jobs += 1
                    result.total_cost += cost
                    result.total_wait_hours += wait_hours
                    result.total_runtime_hours += hours_into_job
                    elapsed = wait_hours + hours_into_job
                    if interrupted:
                        # Add the wasted runtime from interruptions
                        elapsed += result.total_runtime_hours - hours_into_job
                    result.completion_times.append(elapsed)
                    break

            idx += 1

    return result


# ── Formatting ────────────────────────────────────────────────────────────────

def print_result(r: BacktestResult):
    print(f"\n{'─' * 52}")
    print(f"  GPU:              {r.gpu_type}")
    print(f"  Budget:           ${r.budget:.2f}/hr")
    print(f"  Job duration:     {r.job_duration_hours:.1f} hr")
    print(f"  Price windows:    {r.total_windows:,}")
    print(f"{'─' * 52}")
    print(f"  Completion rate:  {r.completion_rate * 100:.1f}%")
    print(f"  Interruption rate:{r.interruption_rate * 100:.1f}%")
    print(f"  Avg wait:         {r.avg_wait_hours:.1f} hr")
    print(f"  Avg actual cost:  ${r.avg_actual_cost:.4f}")
    print(f"  p50 completion:   {r.p50_completion_hours:.1f} hr")
    print(f"  p90 completion:   {r.p90_completion_hours:.1f} hr")
    print(f"{'─' * 52}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _gpu_choices() -> list[str]:
    """Return sorted unique GPU instance types found in the data directory."""
    seen: set[str] = set()
    files = sorted(glob.glob(str(DATA_DIR / "*.tsv.zst")))[:2]  # sample 2 files
    for path in files:
        with open(path, "rb") as f:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f) as sr:
                for line in io.TextIOWrapper(sr):
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[1].startswith(GPU_PREFIXES):
                        seen.add(parts[1])
    return sorted(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS Spot GPU backtest")
    parser.add_argument("--budget", type=float, default=0.5,
                        help="Max $/hr budget (default: 0.5)")
    parser.add_argument("--gpu", type=str, default="g5.xlarge",
                        help="Instance type (default: g5.xlarge)")
    parser.add_argument("--hours", type=float, default=1.0,
                        help="Job duration in hours (default: 1.0)")
    parser.add_argument("--az", type=str, default=None,
                        help="Limit to a specific AZ global ID")
    parser.add_argument("--all-gpus", action="store_true",
                        help="Run backtest for every GPU type found in data")
    args = parser.parse_args()

    if args.all_gpus:
        print("Discovering GPU types in dataset...")
        gpu_types = _gpu_choices()
        print(f"Found {len(gpu_types)} GPU types. Running backtests...\n")
        results = []
        for gpu in gpu_types:
            print(f"  Backtesting {gpu}...", end="", flush=True)
            r = run_backtest(args.budget, gpu, args.hours, args.az)
            results.append(r)
            print(f" done ({r.completion_rate * 100:.0f}% completion)")

        # Sort by completion rate descending
        results.sort(key=lambda r: r.completion_rate, reverse=True)
        print("\n=== Summary (sorted by completion rate) ===")
        print(f"{'GPU':<20} {'Complete%':>10} {'AvgWait':>9} {'AvgCost':>10} {'p90hr':>7}")
        print("─" * 60)
        for r in results:
            print(
                f"{r.gpu_type:<20} "
                f"{r.completion_rate * 100:>9.1f}% "
                f"{r.avg_wait_hours:>8.1f}h "
                f"${r.avg_actual_cost:>9.4f} "
                f"{r.p90_completion_hours:>6.1f}h"
            )
    else:
        print(f"Loading prices for {args.gpu}...")
        r = run_backtest(args.budget, args.gpu, args.hours, args.az)
        print_result(r)
