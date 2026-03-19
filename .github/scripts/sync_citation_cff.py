#!/usr/bin/env python3
"""Sync CITATION.cff version from pyproject.toml; refresh date-released when version changes."""
from __future__ import annotations

import datetime
import re
import sys
import tomllib
from pathlib import Path


def _strip_version_value(raw: str) -> str:
    s = raw.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    pyproject_path = root / "pyproject.toml"
    citation_path = root / "CITATION.cff"

    if not pyproject_path.is_file():
        print(f"Missing {pyproject_path}", file=sys.stderr)
        return 1
    if not citation_path.is_file():
        print(f"Missing {citation_path}", file=sys.stderr)
        return 1

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    project = data.get("project") or {}
    py_version = project.get("version")
    if not py_version or not isinstance(py_version, str):
        print("pyproject.toml [project] must contain a string 'version'", file=sys.stderr)
        return 1
    py_version = py_version.strip()

    text = citation_path.read_text(encoding="utf-8")
    m = re.search(r"^version:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        print("CITATION.cff: no version: line found", file=sys.stderr)
        return 1
    cff_version = _strip_version_value(m.group(1))

    if cff_version == py_version:
        print(f"CITATION.cff already at version {py_version}; no changes.")
        return 0

    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    text = re.sub(r"^version:\s*.*$", f"version: {py_version}", text, count=1, flags=re.MULTILINE)
    if re.search(r"^date-released:\s*.*$", text, flags=re.MULTILINE):
        text = re.sub(
            r"^date-released:\s*.*$",
            f"date-released: {today}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = re.sub(
            r"^(version:\s*.+)$",
            rf"\1\ndate-released: {today}",
            text,
            count=1,
            flags=re.MULTILINE,
        )

    citation_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated CITATION.cff: version {cff_version!r} -> {py_version!r}, date-released -> {today}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
