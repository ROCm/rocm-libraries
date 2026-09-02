#!/usr/bin/env python3
"""Fetch one wiki/source page by id or path.

python3 scripts/get_page.py family-gemm
python3 scripts/get_page.py technique-lds-swizzle --follow-sources
python3 scripts/get_page.py wiki/families/overview.md --body-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402
from _wiki_root import WIKI_ROOT  # noqa: E402


def split_frontmatter(content):
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", content, re.DOTALL)
    if not m:
        return None, content
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        fm = None
    return fm, m.group(2)


def find_page(lookup):
    if "/" in lookup or lookup.endswith(".md"):
        p = WIKI_ROOT / lookup
        return p if p.exists() else None
    for subdir in ["wiki", "sources"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, _ = split_frontmatter(content)
            if isinstance(fm, dict) and fm.get("id") == lookup:
                return md
    return None


def find_by_id(pid):
    for subdir in ["wiki", "sources"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            fm, _ = split_frontmatter(md.read_text(encoding="utf-8"))
            if isinstance(fm, dict) and fm.get("id") == pid:
                return md
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch a rocke optimization wiki page")
    parser.add_argument("lookup", help="page id or path")
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--frontmatter-only", action="store_true")
    parser.add_argument("--follow-sources", action="store_true")
    args = parser.parse_args()

    path = find_page(args.lookup)
    if path is None:
        print(f"No page matching {args.lookup!r}", file=sys.stderr)
        sys.exit(1)

    content = path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(content)
    rel = path.relative_to(WIKI_ROOT).as_posix()

    if args.frontmatter_only:
        print(rel)
        print(yaml.safe_dump(fm, sort_keys=False) if fm else "")
        return
    if args.body_only:
        print(body.rstrip())
        return

    print(f"# {rel}")
    print()
    print(content.rstrip())
    print()

    if args.follow_sources and isinstance(fm, dict):
        for sid in fm.get("sources") or []:
            src = find_by_id(sid)
            print()
            print(f"----- source {sid} -----")
            if src is None:
                print("(missing)")
                continue
            s_content = src.read_text(encoding="utf-8")
            _, s_body = split_frontmatter(s_content)
            print((s_body or s_content)[:800].rstrip())
            print()


if __name__ == "__main__":
    main()
