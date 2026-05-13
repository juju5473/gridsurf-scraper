"""
GridSurf Provider Scraper
=========================
Collects GPU pricing snapshots from four providers on a schedule:
  • Vast.ai        — real-time marketplace offers (public API, no key)
  • RunPod         — GPU type pricing (public GraphQL, no key)
  • Akash Network  — decentralised marketplace (public REST + blockchain API)
  • Spheron        — neocloud GPU pricing (public pricing page scrape)

All data lands in a single SQLite database for historical analysis and
backtest input for the GridSurf price/availability model.

Usage:
    python scraper.py                          # Run once immediately
    python scraper.py --schedule               # Run every 60 min continuously
    python scraper.py --schedule --interval 30 # Every 30 min
    python scraper.py --export-csv             # Export latest snapshot to CSV
    python scraper.py --providers vast runpod  # Run only specific providers

Requirements:
    pip install requests schedule beautifulsoup4
"""

import argparse
import csv
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import schedule

# ── Paths ────────────────────────────────────────────────────────────────────

DB_PATH    = Path(__file__).parent / "data" / "provider_snapshots.db"
LOG_PATH   = Path(__file__).parent / "data" / "scraper.log"
CSV_EXPORT = Path(__file__).parent / "data" / "latest_snapshot.csv"

# ── Provider endpoints ────────────────────────────────────────────────────────

VAST_URL    = "https://console.vast.ai/api/v0/bundles/"
VAST_BODY   = {
    "limit":     100,
    "type":      "on-demand",
    "rentable":  {"eq": True},
    "rented":    {"eq": False},
    "num_gpus":  {"gte": 1},
    "gpu_ram":   {"gte": 16},
    "order":     [["dph_total", "asc"]],
}

RUNPOD_URL   = "https://api.runpod.io/graphql"
RUNPOD_QUERY = """
{
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: { gpuCount: 1 }) {
      minimumBidPrice
      uninterruptablePrice
    }
  }
}
"""

AKASH_PROVIDERS_URL  = "https://api.akash.network/akash/providers/v1/providers?limit=200"
AKASH_GPU_PRICES_URL = "https://console.akash.network/api/gpu-prices"
AKASH_BACKUP_URL     = "https://api.akash.network/akash/market/v1beta4/leases/list?filters.state=active&pagination.limit=100"

SPHERON_PRICING_URL  = "https://spheron.network/pricing"
SPHERON_API_URL      = "https://api.spheron.network/v1/compute/gpu-pricing"

# Static fallback — sourced from spheron.network/blog/gpu-cloud-pricing-comparison-2026/
# timestamped April 2026. Overwrites itself each run so you get a dated record
# even when live scraping fails.
SPHERON_STATIC_PRICES = [
    {"gpu": "RTX_4090",  "vram": 24,  "on_demand": 0.49,  "spot": 0.20},
    {"gpu": "L40S",      "vram": 48,  "on_demand": 0.86,  "spot": 0.35},
    {"gpu": "A100_80GB", "vram": 80,  "on_demand": 1.07,  "spot": 0.60},
    {"gpu": "H100_PCIE", "vram": 80,  "on_demand": 2.01,  "spot": 0.80},
    {"gpu": "H100_SXM",  "vram": 80,  "on_demand": 2.49,  "spot": 1.10},
    {"gpu": "H200",      "vram": 141, "on_demand": 3.20,  "spot": 1.45},
    {"gpu": "B200",      "vram": 192, "on_demand": 6.02,  "spot": 2.12},
    {"gpu": "B300",      "vram": 288, "on_demand": 6.80,  "spot": 2.45},
]

HEADERS = {
    "User-Agent":   "GridSurf-Scraper/0.2 (research data collection)",
    "Accept":       "application/json",
    "Content-Type": "application/json",
}
REQUEST_TIMEOUT = 30

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )

log = logging.getLogger("gridsurf.scraper")

# ── Database ──────────────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS provider_offers (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id             TEXT    NOT NULL,
            captured_at             TEXT    NOT NULL,
            provider                TEXT    NOT NULL,
            offer_id                TEXT,
            gpu_name                TEXT    NOT NULL,
            gpu_name_raw            TEXT,
            gpu_count               INTEGER,
            vram_gb                 REAL,
            price_per_gpu_hr        REAL,
            price_total_hr          REAL,
            interruptible           INTEGER,
            reliability_score       REAL,
            geography               TEXT,
            inet_down_mbps          REAL,
            inet_up_mbps            REAL,
            disk_space_gb           REAL,
            gpu_utilization_pct     REAL,
            extra_json              TEXT
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id     TEXT PRIMARY KEY,
            started_at      TEXT NOT NULL,
            finished_at     TEXT,
            vast_offers     INTEGER DEFAULT 0,
            runpod_offers   INTEGER DEFAULT 0,
            akash_offers    INTEGER DEFAULT 0,
            spheron_offers  INTEGER DEFAULT 0,
            errors          TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_offers_provider_gpu
            ON provider_offers (provider, gpu_name, captured_at);
        CREATE INDEX IF NOT EXISTS idx_offers_price
            ON provider_offers (price_per_gpu_hr, captured_at);
        CREATE INDEX IF NOT EXISTS idx_offers_snapshot
            ON provider_offers (snapshot_id);
    """)
    conn.commit()
    return conn

# ── GPU name normalisation ────────────────────────────────────────────────────

def normalize_gpu_name(raw: str) -> str:
    if not raw:
        return "UNKNOWN"
    s = raw.upper().replace(" ", "_").replace("-", "_")
    for prefix in ("NVIDIA_", "GEFORCE_", "TESLA_", "QUADRO_"):
        s = s.replace(prefix, "")
    if "H100" in s:
        return "H100_SXM" if "SXM" in s else "H100_PCIE"
    if "H200" in s:
        return "H200"
    if "B200" in s:
        return "B200"
    if "B300" in s:
        return "B300"
    if "A100" in s:
        return "A100_80GB" if "80" in s else "A100_40GB"
    if "L40S" in s:
        return "L40S"
    if "L40" in s:
        return "L40"
    if "RTX_4090" in s:
        return "RTX_4090"
    if "RTX_3090" in s:
        return "RTX_3090"
    if "RTX_4080" in s:
        return "RTX_4080"
    if "RTX_3080" in s:
        return "RTX_3080"
    if "A40" in s:
        return "A40"
    if "A6000" in s:
        return "RTX_A6000"
    if "T4" in s:
        return "T4"
    if "V100" in s:
        return "V100"
    return s[:64]

def make_snapshot_id() -> str:
    return datetime.now(timezone.utc).strftime("snap_%Y%m%d_%H%M%S")

# ── INSERT helper ─────────────────────────────────────────────────────────────

INSERT_SQL = """
    INSERT INTO provider_offers (
        snapshot_id, captured_at, provider, offer_id,
        gpu_name, gpu_name_raw, gpu_count, vram_gb,
        price_per_gpu_hr, price_total_hr, interruptible,
        reliability_score, geography,
        inet_down_mbps, inet_up_mbps, disk_space_gb,
        gpu_utilization_pct, extra_json
    ) VALUES (
        :snapshot_id, :captured_at, :provider, :offer_id,
        :gpu_name, :gpu_name_raw, :gpu_count, :vram_gb,
        :price_per_gpu_hr, :price_total_hr, :interruptible,
        :reliability_score, :geography,
        :inet_down_mbps, :inet_up_mbps, :disk_space_gb,
        :gpu_utilization_pct, :extra_json
    )
"""

def write_offers(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany(INSERT_SQL, rows)
    conn.commit()

# ── Vast.ai ───────────────────────────────────────────────────────────────────

def fetch_vast(snapshot_id: str, captured_at: str) -> list[dict]:
    log.info("Fetching Vast.ai offers…")
    try:
        resp = requests.post(VAST_URL, json=VAST_BODY,
                     headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        offers_raw = resp.json().get("offers", [])
    except Exception as e:
        log.error(f"Vast.ai error: {e}")
        return []

    log.info(f"  Vast.ai -> {len(offers_raw)} raw offers")
    rows = []
    for o in offers_raw:
        gpu_raw       = o.get("gpu_name", "") or ""
        gpu_count     = int(o.get("num_gpus", 1) or 1)
        vram_gb       = float(o.get("gpu_ram", 0) or 0) / 1024
        price_total   = float(o.get("dph_total", 0) or 0)
        price_per_gpu = round(price_total / max(gpu_count, 1), 6)
        rows.append({
            "snapshot_id":         snapshot_id,
            "captured_at":         captured_at,
            "provider":            "vast",
            "offer_id":            str(o.get("id", "")),
            "gpu_name":            normalize_gpu_name(gpu_raw),
            "gpu_name_raw":        gpu_raw,
            "gpu_count":           gpu_count,
            "vram_gb":             vram_gb,
            "price_per_gpu_hr":    price_per_gpu,
            "price_total_hr":      price_total,
            "interruptible":       1,
            "reliability_score":   float(o.get("reliability2", 0) or 0),
            "geography":           o.get("geolocation", ""),
            "inet_down_mbps":      float(o.get("inet_down", 0) or 0),
            "inet_up_mbps":        float(o.get("inet_up", 0) or 0),
            "disk_space_gb":       float(o.get("disk_space", 0) or 0),
            "gpu_utilization_pct": float(o.get("gpu_util", 0) or 0),
            "extra_json":          json.dumps({
                "machine_id":  o.get("machine_id"),
                "cuda_vers":   o.get("cuda_max_good"),
                "compute_cap": o.get("compute_cap"),
                "verified":    o.get("verified"),
            }),
        })
    return rows

# ── RunPod ────────────────────────────────────────────────────────────────────

def fetch_runpod(snapshot_id: str, captured_at: str) -> list[dict]:
    log.info("Fetching RunPod GPU types…")
    try:
        resp = requests.post(RUNPOD_URL, json={"query": RUNPOD_QUERY},
                             headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        gpu_types = resp.json().get("data", {}).get("gpuTypes", [])
    except Exception as e:
        log.error(f"RunPod error: {e}")
        return []

    log.info(f"  RunPod -> {len(gpu_types)} GPU types")
    rows = []
    for g in gpu_types:
        lowest    = g.get("lowestPrice") or {}
        bid_price = lowest.get("minimumBidPrice")
        od_price  = lowest.get("uninterruptablePrice")
        gpu_raw   = g.get("displayName", "") or g.get("id", "")
        vram_gb   = float(g.get("memoryInGb", 0) or 0)

        if bid_price is None and od_price is None:
            continue

        base = dict(
            snapshot_id=snapshot_id, captured_at=captured_at,
            provider="runpod",
            gpu_name=normalize_gpu_name(gpu_raw), gpu_name_raw=gpu_raw,
            gpu_count=1, vram_gb=vram_gb,
            reliability_score=None, geography=None,
            inet_down_mbps=None, inet_up_mbps=None,
            disk_space_gb=None, gpu_utilization_pct=None,
            extra_json=json.dumps({
                "gpu_id":          g.get("id"),
                "secure_cloud":    g.get("secureCloud"),
                "community_cloud": g.get("communityCloud"),
            }),
        )
        if bid_price is not None:
            rows.append({**base,
                "offer_id":         f"{g.get('id','')}_spot",
                "price_per_gpu_hr": float(bid_price),
                "price_total_hr":   float(bid_price),
                "interruptible":    1,
            })
        if od_price is not None:
            rows.append({**base,
                "offer_id":         f"{g.get('id','')}_ondemand",
                "price_per_gpu_hr": float(od_price),
                "price_total_hr":   float(od_price),
                "interruptible":    0,
            })
    return rows

# ── Akash Network ─────────────────────────────────────────────────────────────

def _parse_vram(ram_str: str) -> float:
    if not ram_str:
        return 0.0
    s = ram_str.strip().upper()
    try:
        if s.endswith("GI"):
            return float(s[:-2]) * (1024**3) / (1000**3)
        if s.endswith("G"):
            return float(s[:-1])
        if s.endswith("MI"):
            return float(s[:-2]) * (1024**2) / (1000**3)
        if s.endswith("M"):
            return float(s[:-1]) / 1000
        return float(re.sub(r"[^\d.]", "", s))
    except ValueError:
        return 0.0

def _attr_float(attrs: list, key: str):
    for a in attrs:
        if a.get("key", "").lower() == key.lower():
            try:
                return float(a.get("value", 0))
            except (ValueError, TypeError):
                return None
    return None

def _extract_geo(prov: dict) -> str:
    attrs = prov.get("attributes", []) or []
    for a in attrs:
        if a.get("key", "").lower() in ("region", "location", "country", "geo"):
            return str(a.get("value", ""))
    return prov.get("region", "") or ""

def fetch_akash(snapshot_id: str, captured_at: str) -> list[dict]:
    """
    Two data sources:
      1. console.akash.network/api/gpu-prices — aggregated bid prices per GPU model
      2. api.cloudmos.io/v1/providers        — individual provider attributes
    Akash prices are reverse-auction bids denominated USD/month on-chain;
    we convert to USD/hr by dividing by 730.
    """
    log.info("Fetching Akash Network GPU data…")
    rows = []

    # Source 1: aggregated GPU prices
    try:
        resp = requests.get(AKASH_GPU_PRICES_URL, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            prices = resp.json()
            if isinstance(prices, list):
                for p in prices:
                    gpu_raw = f"{p.get('gpu','')} {p.get('ram','')} {p.get('interface','')}".strip()
                    price_d = p.get("price", {})
                    avg_p   = price_d.get("avg") or price_d.get("min")
                    if avg_p is None:
                        continue
                    price_hr = round(float(avg_p) / 730, 6)
                    rows.append({
                        "snapshot_id":         snapshot_id,
                        "captured_at":         captured_at,
                        "provider":            "akash",
                        "offer_id":            f"akash_{p.get('gpu','')}_{p.get('ram','')}",
                        "gpu_name":            normalize_gpu_name(p.get("gpu", "")),
                        "gpu_name_raw":        gpu_raw,
                        "gpu_count":           1,
                        "vram_gb":             _parse_vram(p.get("ram", "")),
                        "price_per_gpu_hr":    price_hr,
                        "price_total_hr":      price_hr,
                        "interruptible":       1,
                        "reliability_score":   None,
                        "geography":           None,
                        "inet_down_mbps":      None,
                        "inet_up_mbps":        None,
                        "disk_space_gb":       None,
                        "gpu_utilization_pct": None,
                        "extra_json":          json.dumps({
                            "price_min_monthly": price_d.get("min"),
                            "price_avg_monthly": price_d.get("avg"),
                            "price_max_monthly": price_d.get("max"),
                            "interface":         p.get("interface"),
                            "source":            "akash_console_prices",
                        }),
                    })
                log.info(f"  Akash console prices -> {len(rows)} GPU models")
    except Exception as e:
        log.warning(f"  Akash console prices: {e}")

    # Source 2: active provider list
    provider_rows = []
    try:
        resp = requests.get(AKASH_PROVIDERS_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            providers = data.get("providers", data) if isinstance(data, dict) else data
            if not isinstance(providers, list):
                providers = []
            for prov in providers:
                attrs = prov.get("attributes", []) or []
                gpu_models = []
                for attr in attrs:
                    k = attr.get("key", "").lower()
                    if "gpu" in k and "model" in k:
                        gpu_models.append(attr.get("value", ""))
                caps = prov.get("capabilities", {}) or {}
                for m in caps.get("gpu", {}).get("models", []):
                    gpu_models.append(m.get("model", ""))
                pricing = prov.get("pricing", {}) or {}
                gpu_price_monthly = pricing.get("gpu", 0) or 0
                gpu_price_hr = round(float(gpu_price_monthly) / 730, 6) if gpu_price_monthly else None
                geo = _extract_geo(prov)
                for gpu_raw in set(filter(None, gpu_models)):
                    provider_rows.append({
                        "snapshot_id":         snapshot_id,
                        "captured_at":         captured_at,
                        "provider":            "akash",
                        "offer_id":            f"akash_prov_{str(prov.get('owner',''))[:16]}_{gpu_raw[:20]}",
                        "gpu_name":            normalize_gpu_name(gpu_raw),
                        "gpu_name_raw":        gpu_raw,
                        "gpu_count":           1,
                        "vram_gb":             None,
                        "price_per_gpu_hr":    gpu_price_hr,
                        "price_total_hr":      gpu_price_hr,
                        "interruptible":       1,
                        "reliability_score":   float(prov.get("uptime", 0) or 0) / 100,
                        "geography":           geo,
                        "inet_down_mbps":      _attr_float(attrs, "network-speed-down"),
                        "inet_up_mbps":        _attr_float(attrs, "network-speed-up"),
                        "disk_space_gb":       None,
                        "gpu_utilization_pct": None,
                        "extra_json":          json.dumps({
                            "owner":        prov.get("owner"),
                            "host_uri":     prov.get("hostUri"),
                            "active_leases":prov.get("leaseCount"),
                            "audited":      prov.get("isAudited"),
                            "source":       "akash_provider_list",
                        }),
                    })
            if provider_rows:
                log.info(f"  Akash provider list -> {len(provider_rows)} provider-GPU rows")
                rows.extend(provider_rows)
    except Exception as e:
        log.warning(f"  Akash provider list: {e}")

    if not rows:
        log.warning("  Akash: no data from any source")
    return rows

# ── Spheron ───────────────────────────────────────────────────────────────────

def _deep_find(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _deep_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _deep_find(item, key)
            if r is not None:
                return r
    return None

def fetch_spheron(snapshot_id: str, captured_at: str) -> list[dict]:
    """
    Three-stage strategy:
      1. Try Spheron REST API (may need auth — will fail gracefully).
      2. Try scraping __NEXT_DATA__ from the pricing page HTML.
      3. Fall back to the static price table (Apr 2026 baseline).
    Stage 3 always runs when stages 1 & 2 fail, so there's always a row
    timestamped at collection time even if live data is unavailable.
    """
    log.info("Fetching Spheron GPU pricing…")
    rows = []

    def _make_rows(gpu_raw, vram_gb, od_price, sp_price, source):
        base = dict(
            snapshot_id=snapshot_id, captured_at=captured_at,
            provider="spheron",
            gpu_name=normalize_gpu_name(gpu_raw), gpu_name_raw=gpu_raw,
            gpu_count=1, vram_gb=float(vram_gb or 0),
            reliability_score=0.99,
            geography="multi-region",
            inet_down_mbps=None, inet_up_mbps=None,
            disk_space_gb=None, gpu_utilization_pct=None,
            extra_json=json.dumps({"source": source}),
        )
        out = []
        if od_price:
            out.append({**base,
                "offer_id":         f"spheron_{gpu_raw}_od",
                "price_per_gpu_hr": float(od_price),
                "price_total_hr":   float(od_price),
                "interruptible":    0,
            })
        if sp_price:
            out.append({**base,
                "offer_id":         f"spheron_{gpu_raw}_spot",
                "price_per_gpu_hr": float(sp_price),
                "price_total_hr":   float(sp_price),
                "interruptible":    1,
            })
        return out

    # Stage 1: API
    try:
        resp = requests.get(SPHERON_API_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            gpus = data.get("gpus", data if isinstance(data, list) else [])
            for g in gpus:
                gpu_raw  = g.get("name") or g.get("gpu") or ""
                od_price = g.get("onDemandPrice") or g.get("hourlyPrice") or g.get("price")
                sp_price = g.get("spotPrice") or g.get("interruptiblePrice")
                vram_gb  = g.get("vram") or g.get("memoryGb") or 0
                rows.extend(_make_rows(gpu_raw, vram_gb, od_price, sp_price, "spheron_api"))
            if rows:
                log.info(f"  Spheron API -> {len(rows)} entries")
                return rows
    except Exception as e:
        log.debug(f"  Spheron API: {e}")

    # Stage 2: HTML scrape
    try:
        resp = requests.get(
            SPHERON_PRICING_URL,
            headers={**HEADERS, "Accept": "text/html,application/xhtml+xml"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            html = resp.text
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
            )
            if match:
                page_data = json.loads(match.group(1))
                gpu_data = (
                    _deep_find(page_data, "gpuPricing") or
                    _deep_find(page_data, "gpus") or
                    _deep_find(page_data, "pricing")
                )
                if gpu_data and isinstance(gpu_data, list):
                    for g in gpu_data:
                        gpu_raw  = g.get("name") or g.get("gpu") or g.get("model") or ""
                        od_price = g.get("hourlyPrice") or g.get("onDemandPrice") or g.get("price")
                        sp_price = g.get("spotPrice") or g.get("interruptiblePrice")
                        vram_gb  = g.get("vram") or g.get("memory") or 0
                        rows.extend(_make_rows(gpu_raw, vram_gb, od_price, sp_price, "spheron_html"))
                    if rows:
                        log.info(f"  Spheron HTML scrape -> {len(rows)} entries")
                        return rows
    except Exception as e:
        log.debug(f"  Spheron HTML scrape: {e}")

    # Stage 3: Static fallback
    log.info("  Spheron -> static price table (Apr 2026 baseline)")
    for entry in SPHERON_STATIC_PRICES:
        rows.extend(_make_rows(
            entry["gpu"], entry["vram"],
            entry["on_demand"], entry["spot"],
            "static_fallback_2026-04-15",
        ))
    log.info(f"  Spheron static -> {len(rows)} entries")
    return rows

# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(conn: sqlite3.Connection, snapshot_id: str):
    cur = conn.execute("""
        SELECT provider, gpu_name,
            COUNT(*)                        AS offers,
            ROUND(MIN(price_per_gpu_hr),4)  AS min_p,
            ROUND(AVG(price_per_gpu_hr),4)  AS avg_p,
            ROUND(MAX(price_per_gpu_hr),4)  AS max_p,
            SUM(interruptible)              AS spot
        FROM provider_offers
        WHERE snapshot_id = ?
        GROUP BY provider, gpu_name
        ORDER BY gpu_name, provider
    """, (snapshot_id,))
    rows = cur.fetchall()
    W = 82
    print(f"\n{'─'*W}")
    print(f"  Snapshot: {snapshot_id}")
    print(f"{'─'*W}")
    print(f"  {'Provider':<10} {'GPU':<18} {'Offers':>6} {'Min':>9} {'Avg':>9} {'Max':>9} {'Spot':>5}")
    print(f"{'─'*W}")
    for r in rows:
        print(
            f"  {r['provider']:<10} {r['gpu_name']:<18} "
            f"{r['offers']:>6} {r['min_p']:>9.4f} "
            f"{r['avg_p']:>9.4f} {r['max_p']:>9.4f} {r['spot']:>5}"
        )
    print(f"{'─'*W}\n")

# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(conn: sqlite3.Connection, path: Path):
    cur = conn.execute("""
        SELECT * FROM provider_offers
        WHERE snapshot_id = (
            SELECT snapshot_id FROM snapshots ORDER BY started_at DESC LIMIT 1
        )
        ORDER BY provider, gpu_name, price_per_gpu_hr
    """)
    rows = cur.fetchall()
    if not rows:
        log.warning("No rows to export.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([dict(r) for r in rows])
    log.info(f"Exported {len(rows)} rows -> {path}")

# ── Main collection run ───────────────────────────────────────────────────────

ALL_PROVIDERS = ["vast", "runpod", "akash", "spheron"]

def collect(conn: sqlite3.Connection, providers: list = None):
    if providers is None:
        providers = ALL_PROVIDERS

    snapshot_id = make_snapshot_id()
    started_at  = captured_at = datetime.now(timezone.utc).isoformat()
    errors      = []

    conn.execute(
        "INSERT INTO snapshots (snapshot_id, started_at) VALUES (?, ?)",
        (snapshot_id, started_at),
    )
    conn.commit()
    log.info(f"Collection run: {snapshot_id}  providers={providers}")

    counts = {p: 0 for p in ALL_PROVIDERS}
    fetchers = {
        "vast":    fetch_vast,
        "runpod":  fetch_runpod,
        "akash":   fetch_akash,
        "spheron": fetch_spheron,
    }

    for provider in providers:
        try:
            rows = fetchers[provider](snapshot_id, captured_at)
            write_offers(conn, rows)
            counts[provider] = len(rows)
        except Exception as e:
            msg = f"{provider}: {e}"
            log.error(msg)
            errors.append(msg)

    finished_at = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE snapshots
        SET finished_at    = ?,
            vast_offers    = ?,
            runpod_offers  = ?,
            akash_offers   = ?,
            spheron_offers = ?,
            errors         = ?
        WHERE snapshot_id = ?
    """, (
        finished_at,
        counts["vast"], counts["runpod"],
        counts["akash"], counts["spheron"],
        json.dumps(errors) if errors else None,
        snapshot_id,
    ))
    conn.commit()

    total = sum(counts.values())
    log.info(
        f"Done: vast={counts['vast']} runpod={counts['runpod']} "
        f"akash={counts['akash']} spheron={counts['spheron']}  total={total}"
    )
    print_summary(conn, snapshot_id)
    return snapshot_id

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GridSurf provider data scraper")
    parser.add_argument("--schedule",   action="store_true")
    parser.add_argument("--interval",   type=int, default=60)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--providers",  nargs="+", default=None, choices=ALL_PROVIDERS)
    parser.add_argument("--db",         type=str, default=str(DB_PATH))
    args = parser.parse_args()

    setup_logging()
    conn     = init_db(Path(args.db))
    providers = args.providers or ALL_PROVIDERS

    if args.export_csv:
        export_csv(conn, CSV_EXPORT)
        return

    if args.schedule:
        log.info(f"Scheduled mode: every {args.interval} min")
        collect(conn, providers)
        schedule.every(args.interval).minutes.do(collect, conn=conn, providers=providers)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        collect(conn, providers)

if __name__ == "__main__":
    main()
