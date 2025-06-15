"""
harin.tools.palantir_viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Command‑line utility to explore PalantirGraph without abbreviation.

Features
────────
* `print --node NODE_ID`                       – show full node JSON
* `traverse --start NODE_ID [--depth N]`       – DFS traversal print
* `filter --meta key=value [...] [--limit N]`  – meta_filter search using best_path_meta()

Example
───────
```bash
python -m harin.tools.palantir_viewer filter --meta mood=confused node_type=reflection --limit 5
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from harin.memory.palantirgraph import PalantirGraph

GRAPH_PATH = Path("palantir_graph.json")

def load_graph() -> PalantirGraph:
    return PalantirGraph(persist_path=GRAPH_PATH)

# ------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------

def cmd_print(args: argparse.Namespace):  # noqa: D401
    g = load_graph()
    node = g.nodes.get(args.node)
    if not node:
        print("Node not found", file=sys.stderr)
        return
    print(json.dumps(node.__dict__, ensure_ascii=False, indent=2))


def cmd_traverse(args: argparse.Namespace):  # noqa: D401
    g = load_graph()
    nodes = g.traverse(args.start, depth=args.depth)
    for n in nodes:
        print(f"{n.id}: {n.content[:80]}…  meta={n.meta}")


def cmd_filter(args: argparse.Namespace):  # noqa: D401
    g = load_graph()
    meta: Dict[str, Any] = dict(kv.split("=", 1) for kv in args.meta)
    limit = args.limit or 10
    nodes = g.best_path_meta(meta, limit=limit)
    for n in nodes:
        print(f"{n.id}: {n.content[:80]}…  meta={n.meta}")

# ------------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:  # noqa: D401
    p = argparse.ArgumentParser(prog="palantir_viewer", description="PalantirGraph exploration CLI (no abbreviation)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("print", help="show single node")
    pr.add_argument("--node", required=True)
    pr.set_defaults(func=cmd_print)

    tr = sub.add_parser("traverse", help="DFS traverse")
    tr.add_argument("--start", required=True)
    tr.add_argument("--depth", type=int, default=2)
    tr.set_defaults(func=cmd_traverse)

    fl = sub.add_parser("filter", help="meta filter search")
    fl.add_argument("--meta", nargs="+", required=True, help="key=value pairs")
    fl.add_argument("--limit", type=int, default=10)
    fl.set_defaults(func=cmd_filter)

    return p


def main(argv=None):  # noqa: D401
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
"""
