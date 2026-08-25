#!/usr/bin/env python3
"""Regex search across wiki/ and sources/.

python3 scripts/grep_wiki.py "compv4" --only wiki
python3 scripts/grep_wiki.py "ds_read_tr" --context 2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402


def iter_files(scope):
    dirs = {"wiki": ["wiki"], "sources": ["sources"], "all": ["wiki", "sources"]}
    for sub in dirs.get(scope, ["wiki", "sources"]):
        base = WIKI_ROOT / sub
        if not base.exists():
            continue
        yield from base.rglob("*.md")


def main():
    parser = argparse.ArgumentParser(description="Search the rocke optimization wiki")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--only", choices=["wiki", "sources", "all"], default="all")
    parser.add_argument("--any", action="store_true", help="match if ANY pattern hits")
    parser.add_argument("--context", type=int, default=0)
    args = parser.parse_args()

    compiled = [re.compile(p, re.IGNORECASE) for p in args.patterns]
    hits = 0
    for path in iter_files(args.only):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        matched_idx = []
        for i, line in enumerate(lines):
            ok = [c.search(line) is not None for c in compiled]
            if any(ok) if args.any else all(ok):
                matched_idx.append(i)
        if not matched_idx:
            continue
        rel = path.relative_to(WIKI_ROOT).as_posix()
        for i in matched_idx:
            lo = max(0, i - args.context)
            hi = min(len(lines), i + args.context + 1)
            for j in range(lo, hi):
                prefix = ">" if j == i else " "
                print(f"{rel}:{j+1}:{prefix} {lines[j]}")
            if args.context:
                print("--")
            hits += 1
    if hits == 0:
        print("No matches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
