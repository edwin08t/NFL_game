# src/config_loader.py
"""
Tiny config/secret loader used by all scripts.

- db_creds():     returns only Postgres credentials (from .env)
- api_keys():     returns only API keys (from .env)
- settings():     loads YAML config (config.yaml) for the selected profile

Conventions:
- Put .env in project root (same level as config.yaml).
- Put config.yaml in project root. You can override path with CONFIG_YAML env var.
- Select profile via APP_PROFILE env var (default: "dev").

Example config.yaml
-------------------
profiles:
  dev:
    paths:
      schedule_dir: "Z:/NFL Project/NFL_Two/Seasonal_Updates/yearlySchedule"
      dvoa_dir:     "Z:/NFL Project/NFL_Two/Completed/DVOA"
      odds_dir:     "Z:/NFL Project/NFL_Two/Current/oddBetting"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv

# Load .env once on import (safe to call multiple times)
load_dotenv()


# ----------------------------- helpers ----------------------------------------
def _get(name: str, default: Any = None, *, required: bool = False, cast=None) -> Any:
    """Fetch env var with optional requirement and casting."""
    val = os.getenv(name, default)
    if required and (val is None or f"{val}".strip() == ""):
        raise RuntimeError(f"Missing env var: {name}")
    if cast and val is not None:
        return cast(val)
    return val


# ----------------------------- DB creds ---------------------------------------
def db_creds(*, required: bool = True) -> Dict[str, Any]:
    """
    Return Postgres connection parameters from env (.env).
    If required=True, all must be present.
    """
    return {
        "PG_HOST": _get("PG_HOST", "localhost", required=required),
        "PG_PORT": _get("PG_PORT", "5432", required=required, cast=int),
        "PG_DB": _get("PG_DB", required=required),
        "PG_USER": _get("PG_USER", required=required),
        "PG_PASSWORD": _get("PG_PASSWORD", required=required),
    }


# ----------------------------- API keys ---------------------------------------
def api_keys(required: Iterable[str] = ()) -> Dict[str, Optional[str]]:
    """
    Return API keys from env (.env).

    `required` is a list/tuple of key names you want to enforce, e.g.:
      api_keys(required=("ODDS_API_KEY",))
    Only those will be enforced as present.
    """
    req = set(required or ())
    return {
        "ODDS_API_KEY": _get("ODDS_API_KEY", required=("ODDS_API_KEY" in req)),
        "SPORTRADAR_API_KEY": _get("SPORTRADAR_API_KEY", required=("SPORTRADAR_API_KEY" in req)),
        # Add more as needed:
        # "OTHER_API_KEY": _get("OTHER_API_KEY", required=("OTHER_API_KEY" in req)),
    }


# ----------------------------- settings (YAML) --------------------------------
def settings(profile: Optional[str] = None, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load config.yaml and return the dict for the selected profile.

    Lookup order for config path:
      1) `config_path` arg if provided
      2) ENV: CONFIG_YAML
      3) Project root default: <repo_root>/config.yaml  (parent of this file's folder)

    Profile selection:
      - arg `profile`, else ENV APP_PROFILE, else "dev"

    Returns the profile block as-is (e.g., {"paths": {...}, ...}).
    """
    try:
        import yaml  # PyYAML
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

    # Resolve config path
    env_cfg = os.getenv("CONFIG_YAML")
    cfg_path = Path(env_cfg) if env_cfg else None
    if config_path is not None:
        cfg_path = Path(config_path)
    if cfg_path is None:
        # Default: project root (parent of src/)
        cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    prof = profile or os.getenv("APP_PROFILE", "dev")

    # Support both structured and flat configs
    profiles = cfg.get("profiles")
    if isinstance(profiles, dict) and prof in profiles:
        return profiles[prof] or {}
    return cfg  # fall back to whole file if no profiles block


# ----------------------------- backward-compat --------------------------------
def secrets(required: Iterable[str] = ()):
    """
    Backward-compatible shim:
    - returns union of DB creds (NOT required) + API keys (with `required` applied).
    Useful while migrating older scripts that import `secrets()`.
    """
    out = {}
    out.update(db_creds(required=False))
    out.update(api_keys(required=required))
    return out


__all__ = ["db_creds", "api_keys", "settings", "secrets"]
