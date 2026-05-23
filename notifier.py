"""
GridSurf Email Notifier
=======================
Sends a per-run summary email after each scraper collection via Gmail SMTP.

Configuration — create gridsurf_scraper/.env with:
    GRIDSURF_GMAIL_USER=you@gmail.com
    GRIDSURF_GMAIL_APP_PASS=xxxx xxxx xxxx xxxx

Usage:
    python notifier.py          # send a test email
"""

import os
import smtplib
import sqlite3
from email.mime.text import MIMEText
from pathlib import Path

RECIPIENT  = "ankitjhurani@gmail.com"
SMTP_HOST  = "smtp.gmail.com"
SMTP_PORT  = 587
_ENV_FILE  = Path(__file__).parent / ".env"


def _load_env() -> None:
    if not _ENV_FILE.exists():
        return
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _credentials() -> tuple[str, str] | None:
    _load_env()
    user = os.environ.get("GRIDSURF_GMAIL_USER", "").strip()
    pw   = os.environ.get("GRIDSURF_GMAIL_APP_PASS", "").strip()
    return (user, pw) if user and pw else None


def _smtp_send(user: str, pw: str, subject: str, body: str) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = user
    msg["To"]      = RECIPIENT
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(user, pw)
        smtp.sendmail(user, [RECIPIENT], msg.as_string())


def _build_subject_and_body(
    conn: sqlite3.Connection,
    snapshot_id: str,
    errors: list[str],
) -> tuple[str, str]:
    snap = conn.execute(
        "SELECT finished_at FROM snapshots WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()

    total_snaps = conn.execute(
        "SELECT COUNT(*) FROM snapshots WHERE finished_at IS NOT NULL"
    ).fetchone()[0]

    total_rows = conn.execute(
        "SELECT COUNT(*) FROM provider_offers"
    ).fetchone()[0]

    per_provider = conn.execute(
        """
        SELECT provider, COUNT(*) AS cnt
        FROM provider_offers
        WHERE snapshot_id = ?
        GROUP BY provider ORDER BY provider
        """,
        (snapshot_id,),
    ).fetchall()

    ts_raw  = ((snap["finished_at"] or "")[:16]).replace("T", " ")
    subject = f"GridSurf Scraper — Snapshot #{total_snaps} | {ts_raw} UTC"

    lines = [
        "GridSurf Scraper — Collection Summary",
        "=" * 46,
        "",
        f"Snapshot ID   : {snapshot_id}",
        f"Finished at   : {ts_raw} UTC",
        f"Snapshot #    : {total_snaps}",
        "",
        "Database totals",
        "-" * 30,
        f"  Total snapshots : {total_snaps}",
        f"  Total DB rows   : {total_rows:,}",
        "",
        "This run — rows by provider",
        "-" * 30,
    ]
    for r in per_provider:
        lines.append(f"  {r['provider']:<10} : {r['cnt']}")

    if errors:
        lines += ["", "Errors", "-" * 30]
        for e in errors:
            lines.append(f"  • {e}")
    else:
        lines += ["", "No errors."]

    lines += ["", "-" * 46, "GridSurf Scraper"]
    return subject, "\n".join(lines)


def send_summary(
    conn: sqlite3.Connection,
    snapshot_id: str,
    errors: list[str] | None = None,
) -> None:
    """
    Send a post-collection summary email.
    Silently no-ops when credentials are not configured.
    """
    creds = _credentials()
    if creds is None:
        return
    user, pw = creds
    subject, body = _build_subject_and_body(conn, snapshot_id, errors or [])
    _smtp_send(user, pw, subject, body)


def send_test() -> bool:
    """Send a connectivity test email. Returns True on success."""
    creds = _credentials()
    if creds is None:
        print(
            "No credentials found.\n"
            "Create gridsurf_scraper/.env with:\n\n"
            "  GRIDSURF_GMAIL_USER=you@gmail.com\n"
            "  GRIDSURF_GMAIL_APP_PASS=xxxx xxxx xxxx xxxx\n"
        )
        return False

    user, pw = creds
    _smtp_send(
        user, pw,
        subject="GridSurf Scraper — Test Email",
        body=(
            "This is a test from your GridSurf scraper.\n\n"
            "Email notifications are working correctly.\n\n"
            "— GridSurf"
        ),
    )
    print(f"Test email sent to {RECIPIENT}")
    return True


if __name__ == "__main__":
    send_test()
