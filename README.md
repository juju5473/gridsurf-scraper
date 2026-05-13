# GridSurf Provider Scraper

Collects GPU pricing snapshots from **Vast.ai** and **RunPod** on a schedule
and stores them in SQLite. This is Dataset C from the GridSurf product report —
your proprietary time-series of GPU marketplace pricing.

---

## Setup

```bash
# 1. Clone or copy this folder somewhere permanent on your machine
cd ~/gridsurf_scraper   # or wherever you put it

# 2. Install dependencies (one time)
pip install requests schedule

# 3. Run a single collection immediately to verify everything works
python scraper.py
```

You should see output like:

```
2026-05-12 10:00:01  INFO      Starting collection run: snap_20260512_100001
2026-05-12 10:00:02  INFO      Fetching Vast.ai offers…
2026-05-12 10:00:03  INFO        Vast.ai returned 312 raw offers
2026-05-12 10:00:04  INFO      Fetching RunPod GPU types…
2026-05-12 10:00:05  INFO        RunPod returned 28 GPU types

  ───────────────────────────────────────────────────────────────────────────
  Snapshot: snap_20260512_100001
  ───────────────────────────────────────────────────────────────────────────
  Provider   GPU                Offers   Min $/hr   Avg $/hr   Max $/hr  Spot
  ───────────────────────────────────────────────────────────────────────────
  runpod     A100_80GB               2     1.9900     2.2450     2.4900     1
  runpod     H100_PCIE               2     2.4900     2.7450     2.9900     1
  vast       A100_80GB              14     1.1200     1.8743     2.6400    14
  vast       H100_PCIE              23     1.8700     2.3102     3.1200    23
  vast       RTX_4090               87     0.3400     0.5812     0.9800    87
  ...
```

---

## Run continuously (recommended — start this today)

### Option A: Built-in scheduler (simplest)

```bash
# Collect every 60 minutes, runs indefinitely
python scraper.py --schedule

# Or every 30 minutes for denser data
python scraper.py --schedule --interval 30
```

Keep this running in a terminal, tmux session, or screen. Every day of
data you collect today is irreplaceable for backtesting the scheduler.

### Option B: cron job (recommended for reliability)

Add to your crontab (`crontab -e`):

```cron
# GridSurf provider scraper — every hour
0 * * * * cd /path/to/gridsurf_scraper && python scraper.py >> data/cron.log 2>&1
```

Replace `/path/to/gridsurf_scraper` with your actual path.

### Option C: Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task → "GridSurf Scraper"
3. Trigger: Daily, repeat every 1 hour
4. Action: Start a program
   - Program: `python`
   - Arguments: `C:\path\to\gridsurf_scraper\scraper.py`
   - Start in: `C:\path\to\gridsurf_scraper`

---

## Analyze collected data

```bash
# Overall price summary
python analyze.py --summary

# Price history for a specific GPU
python analyze.py --gpu H100_PCIE
python analyze.py --gpu RTX_4090
python analyze.py --gpu A100_80GB

# Cheapest current offers with at least 40GB VRAM
python analyze.py --cheapest --vram 40

# Historical windows where price was at or below your bid
python analyze.py --windows --budget 2.00

# Volatility analysis (how much do prices swing?)
python analyze.py --volatility

# Export to CSV for Excel/Tableau
python scraper.py --export-csv

# Export provider windows JSON for the GridSurf scheduler
python analyze.py --export-scheduler

# Database stats
python analyze.py --stats
```

---

## File structure

```
gridsurf_scraper/
├── scraper.py              ← Main data collector
├── analyze.py              ← Query and analysis tool
├── README.md               ← This file
└── data/
    ├── provider_snapshots.db   ← SQLite database (auto-created)
    ├── scraper.log             ← Collection run logs
    ├── latest_snapshot.csv     ← CSV export of most recent run
    └── scheduler_windows.json  ← Provider windows for the scheduler
```

---

## Database schema

### `provider_offers` — every captured offer

| Column | Type | Description |
|---|---|---|
| `snapshot_id` | TEXT | Batch ID for the collection run |
| `captured_at` | TEXT | ISO-8601 UTC timestamp |
| `provider` | TEXT | `vast` or `runpod` |
| `offer_id` | TEXT | Provider's internal offer ID |
| `gpu_name` | TEXT | Normalized GPU label (e.g. `H100_PCIE`) |
| `gpu_name_raw` | TEXT | Exactly as provider returned |
| `gpu_count` | INTEGER | Number of GPUs in this offer |
| `vram_gb` | REAL | VRAM in GB |
| `price_per_gpu_hr` | REAL | USD per GPU per hour |
| `price_total_hr` | REAL | USD per hour for whole offer |
| `interruptible` | INTEGER | 1=spot/interruptible, 0=on-demand |
| `reliability_score` | REAL | 0–1 host reliability (Vast only) |
| `geography` | TEXT | Country/region if known |
| `inet_down_mbps` | REAL | Download bandwidth (Vast only) |
| `inet_up_mbps` | REAL | Upload bandwidth (Vast only) |
| `disk_space_gb` | REAL | Available disk (Vast only) |
| `gpu_utilization_pct` | REAL | Current GPU utilization % |
| `extra_json` | TEXT | Full raw offer JSON for anything not in schema |

### `snapshots` — one row per collection run

| Column | Description |
|---|---|
| `snapshot_id` | Unique batch ID |
| `started_at` | When the run started |
| `finished_at` | When the run completed |
| `vast_offers` | Number of Vast offers captured |
| `runpod_offers` | Number of RunPod offers captured |
| `errors` | JSON list of any errors encountered |

---

## Adding more providers later

To add Akash, Spheron, Lambda, etc., follow the same pattern as `fetch_vast()`
or `fetch_runpod()` in `scraper.py`:

1. Add a `fetch_<provider>()` function that returns a list of dicts matching
   the `provider_offers` schema
2. Call it inside the `collect()` function
3. The `normalize_gpu_name()` function will handle most naming variations

---

## What this data is used for

This SQLite database is the foundation for:

- **Price/availability model** — your side of the GridSurf system
- **Scheduler backtesting** — "if a user had bid $X, when would their job complete?"
- **Budget window estimation** — how often does H100 drop below $2/hr?
- **Volatility modeling** — how wide should confidence intervals be?
- **Provider comparison** — Vast vs RunPod pricing dynamics over time

The longer you run this, the more valuable it becomes. One week of data gives
you a baseline. One month gives you volatility. Three months gives you enough
for meaningful backtests.
