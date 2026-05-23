#!/usr/bin/env python3
"""
GridSurf Dashboard API
======================
Run:  python dashboard_api.py
Then open dashboard.html in a browser.
"""

import json
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR       = Path(__file__).parent
DB_PATH        = BASE_DIR / "data" / "provider_snapshots.db"
SCHEDULER_JSON = BASE_DIR / "data" / "scheduler_windows.json"
FORECAST_JSON  = BASE_DIR / "data" / "forecast_windows.json"
BACKTEST_JSON  = BASE_DIR / "data" / "backtest_results.json"

PROVIDERS = {"vast", "runpod", "spheron"}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_json(path: Path) -> list | dict:
    if path.exists():
        return json.loads(path.read_text())
    return []


def recent_where(hours: int = 2) -> str:
    return f"AND SUBSTR(captured_at,1,19) >= STRFTIME('%Y-%m-%dT%H:%M:%S','now','-{hours} hours')"


# ── endpoints ─────────────────────────────────────────────────────────────────

def api_stats() -> dict:
    conn = get_conn()
    r = conn.execute("""
        SELECT COUNT(DISTINCT snapshot_id) AS total_snapshots,
               COUNT(*)                   AS total_rows,
               COUNT(DISTINCT provider)   AS providers,
               MAX(captured_at)           AS last_updated,
               MIN(captured_at)           AS first_snapshot
        FROM provider_offers
    """).fetchone()
    first = datetime.fromisoformat(r["first_snapshot"].replace("Z", "+00:00"))
    last  = datetime.fromisoformat(r["last_updated"].replace("Z", "+00:00"))
    return {
        "total_snapshots": r["total_snapshots"],
        "total_rows":      r["total_rows"],
        "providers":       r["providers"],
        "last_updated":    r["last_updated"],
        "days_of_data":    max(1, (last - first).days + 1),
    }


def api_summary() -> list:
    conn = get_conn()
    # Try recent 4 h, fall back to 48 h so table is never empty
    for window in ["-4 hours", "-48 hours", "-9999 hours"]:
        rows = conn.execute(f"""
            SELECT gpu_name, provider,
                   ROUND(MIN(price_per_gpu_hr), 4) AS min_price,
                   ROUND(AVG(price_per_gpu_hr), 4) AS avg_price,
                   COUNT(*)                         AS offer_count
            FROM provider_offers
            WHERE price_per_gpu_hr IS NOT NULL
              AND SUBSTR(captured_at,1,19) >= STRFTIME('%Y-%m-%dT%H:%M:%S','now','{window}')
            GROUP BY gpu_name, provider
        """).fetchall()
        if rows:
            break

    by_gpu: dict[str, dict] = {}
    for r in rows:
        gpu = r["gpu_name"]
        by_gpu.setdefault(gpu, {})[r["provider"]] = {
            "min_price":   r["min_price"],
            "avg_price":   r["avg_price"],
            "offer_count": r["offer_count"],
        }

    result = []
    for gpu, pvs in sorted(by_gpu.items()):
        prices = {p: v["avg_price"] for p, v in pvs.items() if v["avg_price"]}
        if not prices:
            continue
        cheapest = min(prices, key=prices.get)
        result.append({
            "gpu_name":          gpu,
            "providers":         pvs,
            "cheapest_provider": cheapest,
            "cheapest_price":    prices[cheapest],
            "provider_count":    len(pvs),
        })

    result.sort(key=lambda x: (-x["provider_count"], x["gpu_name"]))
    return result[:20]


def api_volatility() -> list:
    data = load_json(SCHEDULER_JSON)
    by_gpu: dict[str, dict] = {}
    for item in data:
        gpu = item["gpu_class"]
        entry = by_gpu.setdefault(gpu, {"gpu_class": gpu, "vols": [], "providers": []})
        entry["vols"].append(item.get("price_volatility", 0))
        entry["providers"].append(item["provider"])

    result = []
    for gpu, info in by_gpu.items():
        avg_vol = sum(info["vols"]) / len(info["vols"])
        result.append({
            "gpu_class":      gpu,
            "avg_volatility": round(avg_vol, 4),
            "max_volatility": round(max(info["vols"]), 4),
            "providers":      info["providers"],
        })
    result.sort(key=lambda x: -x["avg_volatility"])
    return result


def api_forecast(gpu: str | None = None) -> list:
    data = load_json(FORECAST_JSON)
    if not data:
        return []
    priority = [gpu] if gpu else ["RTX_4090", "H100_SXM", "A100_40GB", "L40S"]
    return [d for d in data if d["gpu_class"] in priority]


def api_alerts() -> list:
    conn = get_conn()
    # Current prices per GPU per provider (last 4 h)
    for window in ["-4 hours", "-48 hours", "-9999 hours"]:
        rows = conn.execute(f"""
            SELECT gpu_name, provider, MIN(price_per_gpu_hr) AS min_price
            FROM provider_offers
            WHERE price_per_gpu_hr IS NOT NULL
              AND SUBSTR(captured_at,1,19) >= STRFTIME('%Y-%m-%dT%H:%M:%S','now','{window}')
            GROUP BY gpu_name, provider
        """).fetchall()
        if rows:
            break

    current: dict[str, dict] = {}
    for r in rows:
        current.setdefault(r["gpu_name"], {})[r["provider"]] = float(r["min_price"])

    # Historical p50 spreads from scheduler for threshold
    hist_spreads: dict[str, float] = {}
    for item in load_json(SCHEDULER_JSON):
        gpu = item["gpu_class"]
        hist_spreads.setdefault(gpu, []).append(item.get("price_per_gpu_hour_p50", 0))

    alerts = []
    for gpu, prices in current.items():
        if len(prices) < 2:
            continue
        vals = list(prices.values())
        spread = max(vals) - min(vals)
        if spread < 0.005:
            continue
        cheapest  = min(prices, key=prices.get)
        priciest  = max(prices, key=prices.get)
        hist_vals = hist_spreads.get(gpu, [])
        if len(hist_vals) >= 2:
            hist_spread = max(hist_vals) - min(hist_vals)
            threshold   = hist_spread * 1.5
            is_alert    = spread > threshold and spread > 0.05
        else:
            is_alert = spread / prices[priciest] > 0.15
        alerts.append({
            "gpu_name":         gpu,
            "spread":           round(spread, 4),
            "cheapest_provider": cheapest,
            "cheapest_price":   round(prices[cheapest], 4),
            "priciest_provider": priciest,
            "priciest_price":   round(prices[priciest], 4),
            "savings_per_100h": round(spread * 100, 2),
            "prices":           {p: round(v, 4) for p, v in prices.items()},
            "is_alert":         is_alert,
        })

    alerts.sort(key=lambda x: -x["spread"])
    return alerts


def api_windows(budget: float) -> list:
    data = load_json(SCHEDULER_JSON)
    result = []
    for item in data:
        p50 = item.get("price_per_gpu_hour_p50", 9999)
        if p50 <= budget:
            result.append({
                "provider":          item["provider"],
                "gpu_class":         item["gpu_class"],
                "price_p50":         p50,
                "price_p10":         item.get("price_per_gpu_hour_p10", p50),
                "availability":      item.get("availability_probability", 0),
                "volatility":        item.get("price_volatility", 0),
                "savings_vs_budget": round(budget - p50, 4),
                "n_samples":         item.get("n_price_samples", 0),
            })
    result.sort(key=lambda x: x["price_p50"])
    return result


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    ROUTES = {
        "/api/stats":      lambda _: api_stats(),
        "/api/summary":    lambda _: api_summary(),
        "/api/volatility": lambda _: api_volatility(),
        "/api/alerts":     lambda _: api_alerts(),
    }

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path   = parsed.path

        try:
            if path in ("/", "/dashboard.html"):
                html_path = BASE_DIR / "dashboard.html"
                if not html_path.exists():
                    self._respond(404, {"error": "dashboard.html not found"})
                    return
                payload = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if path in self.ROUTES:
                data = self.ROUTES[path](params)
            elif path == "/api/forecast":
                gpu = params.get("gpu", [None])[0]
                data = api_forecast(gpu)
            elif path == "/api/windows":
                budget = float(params.get("budget", [1.0])[0])
                data = api_windows(budget)
            else:
                self._respond(404, {"error": "Not found"})
                return
            self._respond(200, data)
        except Exception as exc:
            self._respond(500, {"error": str(exc)})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _respond(self, code: int, body):
        payload = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {args[0]}  {args[1]}  {args[2]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("GridSurf Dashboard API  →  http://0.0.0.0:8080")
    print("Open dashboard.html in your browser (or http://<public-ip>:8080 remotely)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
