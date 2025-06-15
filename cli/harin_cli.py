"""
harin.cli.harin_cli
~~~~~~~~~~~~~~~~~~~

Command‑line interface for Harin utilities **(no feature abbreviation)**

Sub‑commands
────────────
* `telemetry view [--limit N] [--event EVT] [--user USER]` – print log lines
* `telemetry flush`                                  – truncate log file safely
* `plugins list`                                     – show loaded plugins with tags

Usage
─────
```bash
python -m harin.cli.harin_cli telemetry view --limit 20 --event reply
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

from harin.plugins.plugin_manager import PluginManager

TELEMETRY_PATH = Path("harin_telemetry.log")
PLUGINS_PATH = Path("harin_plugins")


# ──────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────

def _read_telemetry(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not TELEMETRY_PATH.exists():
        return []
    lines = TELEMETRY_PATH.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(l) for l in lines if l.strip()]
    if limit:
        records = records[-limit:]
    return records


def _filter(records: List[Dict[str, Any]], key: str, value: str) -> List[Dict[str, Any]]:
    return [r for r in records if str(r.get(key, "")) == value]


def cmd_telemetry(args: argparse.Namespace):  # noqa: D401
    if args.action == "view":
        recs = _read_telemetry(limit=args.limit)
        if args.event:
            recs = _filter(recs, "event", args.event)
        if args.user:
            recs = _filter(recs, "user_id", args.user)
        for r in recs:
            print(json.dumps(r, ensure_ascii=False))
    elif args.action == "flush":
        TELEMETRY_PATH.write_text("", encoding="utf-8")
        print(f"Telemetry flushed: {TELEMETRY_PATH.resolve()}")


def cmd_plugins(_: argparse.Namespace):  # noqa: D401
    pm = PluginManager(PLUGINS_PATH)
    info = pm.info()
    if not info:
        print("(no plugins loaded)")
        return
    for name, meta in info.items():
        tags = ",".join(meta.get("tags", []))
        doc = meta.get("doc", "")
        print(f"{name:\<15} tags=[{tags}]  – {doc}")


# ──────────────────────────────────────────────────────────────────────────
# Argument parser setup
# ──────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:  # noqa: D401
    p = argparse.ArgumentParser(prog="harin_cli", description="Harin utility CLI (no abbreviation)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # telemetry cmd
    tele = sub.add_parser("telemetry", help="telemetry operations")
    tele_sub = tele.add_subparsers(dest="action", required=True)
    view = tele_sub.add_parser("view", help="view log")
    view.add_argument("--limit", type=int, default=20, help="max lines")
    view.add_argument("--event", type=str, help="filter event")
    view.add_argument("--user", type=str, help="filter user_id")
    flush = tele_sub.add_parser("flush", help="truncate log file")

    # plugins cmd
    plugins = sub.add_parser("plugins", help="plugin operations")
    plugins.set_defaults(func=cmd_plugins)

    tele.set_defaults(func=cmd_telemetry)
    return p


def main(argv: List[str] | None = None):  # noqa: D401
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help(sys.stderr)


if __name__ == "__main__":
    main()
"""
