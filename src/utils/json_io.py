# src/utils/json_io.py
from __future__ import annotations
from pathlib import Path
import json, gzip, tempfile
from typing import Any, Iterable

# gzip magic bytes; lets us detect compressed files
_GZIP_MAGIC = b"\x1f\x8b"

# Encodings we'll try in order; covers UTF-8, BOM, and common Windows encodings
_DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

def _maybe_decompress(raw: bytes) -> bytes:
    """Return decompressed bytes if input is gzip; otherwise return as-is."""
    return gzip.decompress(raw) if raw.startswith(_GZIP_MAGIC) else raw

def load_json(path: str | Path, encodings: Iterable[str] = _DEFAULT_ENCODINGS) -> Any:
    """
    Load JSON from a path, handling gzip (.json.gz) and multiple text encodings.
    Raises:
      - UnicodeDecodeError if none of the encodings work
      - json.JSONDecodeError if the decoded text isn't valid JSON
    """
    p = Path(path)
    raw = p.read_bytes()
    raw = _maybe_decompress(raw)

    last_err: Exception | None = None
    for enc in encodings:
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        "unknown", b"", 0, 0,
        f"Could not decode {p} using any of: {', '.join(encodings)}"
    ) from last_err

def save_json(obj: Any, path: str | Path, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Write JSON atomically in UTF-8:
      - create parent dir if needed
      - write to a temp file in the same folder, then replace
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(p.parent), delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)

    tmp_path.replace(p)
