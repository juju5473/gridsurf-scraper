#!/usr/bin/env python3
"""Writes STATS.md with current DB metrics. Called by daily_push.sh."""

import datetime
import os
import sqlite3

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(REPO_DIR, "data", "provider_snapshots.db")
STATS_PATH = os.path.join(REPO_DIR, "STATS.md")


def main():
    if not os.path.exists(DB_PATH):
        print(f"DB not found: {DB_PATH}")
        raise SystemExit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    overall = conn.execute("""
        SELECT COUNT(DISTINCT snapshot_id) AS snapshots,
               COUNT(*)                   AS rows,
               COUNT(DISTINCT provider)   AS providers,
               MIN(SUBSTR(captured_at,1,10)) AS since,
               MAX(SUBSTR(captured_at,1,10)) AS latest
        FROM provider_offers
    """).fetchone()

    by_provider = conn.execute("""
        SELECT provider,
               COUNT(DISTINCT gpu_name) AS gpu_types,
               COUNT(*)                 AS offers
        FROM provider_offers
        WHERE SUBSTR(captured_at,1,19) >= STRFTIME('%Y-%m-%dT%H:%M:%S','now','-48 hours')
        GROUP BY provider
        ORDER BY offers DESC
    """).fetchall()

    cheapest = conn.execute("""
        SELECT provider, gpu_name, ROUND(MIN(price_per_gpu_hr), 4) AS min_price
        FROM provider_offers
        WHERE price_per_gpu_hr IS NOT NULL
          AND SUBSTR(captured_at,1,19) >= STRFTIME('%Y-%m-%dT%H:%M:%S','now','-48 hours')
        GROUP BY provider, gpu_name
        ORDER BY provider, min_price
        LIMIT 15
    """).fetchall()

    conn.close()

    today = datetime.date.today().isoformat()

    lines = [
        "# GridSurf — Live Price Stats",
        "",
        f"> Auto-updated daily &nbsp;·&nbsp; Last update: **{today}**",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Snapshots collected | {overall['snapshots']:,} |",
        f"| Total price observations | {overall['rows']:,} |",
        f"| Providers tracked | {overall['providers']} |",
        f"| Data since | {overall['since']} |",
        f"| Latest snapshot | {overall['latest']} |",
        "",
        "## Provider Coverage (last 48h)",
        "",
        "| Provider | GPU types | Offers |",
        "|----------|-----------|--------|",
    ]
    for r in by_provider:
        lines.append(f"| {r['provider']} | {r['gpu_types']} | {r['offers']} |")

    lines += [
        "",
        "## Cheapest GPUs by Provider (last 48h)",
        "",
        "| Provider | GPU | Min $/hr |",
        "|----------|-----|----------|",
    ]
    for r in cheapest:
        lines.append(f"| {r['provider']} | {r['gpu_name']} | ${r['min_price']:.4f} |")

    lines += [
        "",
        "---",
        "Dashboard: [http://40.233.121.227:8080](http://40.233.121.227:8080)",
    ]

    with open(STATS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"STATS.md written ({overall['rows']:,} rows, {overall['snapshots']:,} snapshots).")


if __name__ == "__main__":
    main()
