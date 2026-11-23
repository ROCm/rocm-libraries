#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Show the GPU faults that hold nodes, and reset a node after it was
rebooted and checked (lib/node_faults.py).

    node_faults.py show  [NODE] [--store DIR]
    node_faults.py reset NODE --note TEXT [--store DIR]

show   every fault of NODE (default: this node) and whether they hold it
reset  write NODE's reset marker with TEXT (who checked it, after which
       reboot): the faults recorded so far hold it no more, so runs bench
       on it again. Nothing else writes the marker.

--store defaults to $TILEWRIGHT_NODE_FAULTS, else
$XDG_STATE_HOME/tilewright/node_faults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import node_faults as nf  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show")
    show.add_argument("node", nargs="?", default=None)
    show.add_argument("--store", default=None)
    reset = sub.add_parser("reset")
    reset.add_argument("node")
    reset.add_argument("--note", required=True)
    reset.add_argument("--store", default=None)
    args = ap.parse_args(argv)
    store = nf.NodeStore(nf.store_dir(args.store))
    node = args.node or nf.node_name()
    if args.cmd == "reset":
        if not args.note.strip():
            ap.error("--note must say who checked the node, after which reboot")
        path = store.reset(node, args.note.strip())
        print(f"{node}: reset ({path})")
        return 0
    held: List[dict] = store.held(node)
    print(json.dumps({"node": node, "faults": store.faults(node)}, indent=2))
    print(nf.describe_hold(node, held, store.root) if held else f"{node}: not held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
