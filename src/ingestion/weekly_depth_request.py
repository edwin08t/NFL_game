# game_request.py
# --- project root import shim (so `from src...` works when run directly) -----
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # <- parent of 'src'
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

from pathlib import Path
import os, time, json, http.client
from src.config_loader import settings, api_keys


def weekly_depth(
    week: int,
    year: int,
    key: str | None = None,
    *,
    min_bytes: int = 100_000,
    max_tries: int = 12,
    sleep_s: float = 0.8,
    overwrite: bool = False,
) -> str:
    """
    Pull weekly depth charts from Sportradar and save to:
      <paths.weekly_depth_dir>/{year}_{week}.json
    """
    week = int(week)
    year = int(year)

    # ---------- paths from config.yaml ----------
    cfg = settings() or {}
    paths = cfg.get("paths") or {}
    weekly_dir_str = paths.get("weekly_depth_dir")
    if not weekly_dir_str:
        raise KeyError("config.yaml is missing paths.weekly_depth_dir")
    weekly_dir = Path(weekly_dir_str)
    weekly_dir.mkdir(parents=True, exist_ok=True)

    file_path = weekly_dir / f"{year}_{week}.json"

    # ---------- API key (arg -> secrets -> env) ----------
    try:
        sec = api_keys()
    except Exception:
        sec = {}
    key = (
        key or sec.get("SPORTRADAR_API_KEY") or os.getenv("SPORTRADAR_API_KEY") or ""
    ).strip()
    if not key:
        raise ValueError("Missing API key: pass key= or set SPORTRADAR_API_KEY")

    request_url = f"/nfl/official/trial/v7/en/seasons/{year}/REG/{week}/depth_charts.json?api_key={key}"

    # Skip if an apparently complete file exists
    if not overwrite and file_path.exists() and file_path.stat().st_size >= min_bytes:
        print(f"[INFO] Already present: {file_path}")
        return str(file_path)

    # ---------- GET with retries ----------
    conn = http.client.HTTPSConnection("api.sportradar.us")
    try:
        tries, last_status = 0, None
        while tries < max_tries:
            tries += 1
            conn.request("GET", request_url)
            res = conn.getresponse()
            last_status = res.status
            data = res.read() or b""

            if res.status in (429, 503):
                print(f"[WARN] HTTP {res.status} (try {tries}) — backing off…")
                time.sleep(max(1.5, sleep_s))
                continue
            if not data.strip():
                print(f"[WARN] Blank payload (try {tries}) — retrying…")
                time.sleep(sleep_s)
                continue
            if len(data) < min_bytes:
                print(
                    f"[WARN] Small payload {len(data)} (<{min_bytes}) (try {tries}) — retrying…"
                )
                time.sleep(sleep_s)
                continue

            # Validate JSON, then save
            text = data.decode("utf-8", errors="strict")
            json.loads(text)
            file_path.write_text(text, encoding="utf-8")
            print(f"[OK] Saved {file_path} (tries={tries}, bytes={len(data)})")
            return str(file_path)

        raise RuntimeError(
            f"Failed after {max_tries} attempts (last HTTP status={last_status})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    # quick manual test
    weekly_depth(week=13, year=2025)
