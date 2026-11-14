# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convert a facts file recorded before the qualname keys (tr/factkeys.py).

usage: python -m tr.rekey <module> [--rocke DIR] [--facts FILE]

Old files key functions by "<qualname>@<firstlineno>". Each key whose line
still matches the module's current source becomes its qualname key and gets
the `src` stamp of that source. A key whose line no longer matches is kept
as it is and listed; re-record those (tr.record). Exit 1 if any is left.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys

from .env import ROCKE_HELP, data_dir, rocke_root
from .factkeys import code_keys, first_line, src_hash


def rekey(facts: dict, src: str) -> tuple[dict, list[str]]:
    """(facts with qualname keys, old keys left unconverted)."""
    index = {
        f"{k.split('#')[0]}@{first_line(n)}": (k, n)
        for k, n in code_keys(ast.parse(src))
    }
    funcs, left = {}, []
    for old, v in facts.get("funcs", {}).items():
        if old in index:
            key, node = index[old]
            funcs[key] = dict(v, src=src_hash(node))
        else:
            funcs[old] = v
            if "@" in old:
                left.append(old)
    return dict(facts, funcs=dict(sorted(funcs.items()))), left


def main(argv=None) -> int:
    from .env import module_path

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module")
    ap.add_argument("--rocke", help=ROCKE_HELP)
    ap.add_argument("--facts", help="facts file (default: <data>/facts/<module>.json)")
    a = ap.parse_args(argv)
    path = a.facts or os.path.join(data_dir(), "facts", f"{a.module}.json")
    src = open(module_path(rocke_root(a.rocke), a.module)).read()
    out, left = rekey(json.load(open(path)), src)
    with open(path, "w") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k in left:
        print(f"not converted (line moved; re-run tr.record): {k}")
    print(f"{len(out['funcs']) - len(left)} keys current, {len(left)} left -> {path}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
