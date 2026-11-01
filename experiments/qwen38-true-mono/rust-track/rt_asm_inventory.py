"""Inventory of the ``tile.inline_asm`` templates left in serialized rocKE IR files.

    python rt_asm_inventory.py DIR/*.ir

Prints (count, families, constraints, template with numbers as N) for every distinct template, most common first.
"""

import collections
import re
import sys

ATTR = re.compile(r'(\w+) = s:"((?:[^"\\]|\\.)*)"')


def main(paths):
    tot = collections.Counter()
    fams = collections.defaultdict(set)
    for f in paths:
        stem = f.replace("\\", "/").rsplit("/", 1)[-1]
        for ln in open(f, encoding="utf-8"):
            if "tile.inline_asm" not in ln:
                continue
            attrs = dict(ATTR.findall(ln))
            key = (attrs.get("constraints", "?"), re.sub(r"\d+", "N", attrs.get("template", "?"))[:140])
            tot[key] += 1
            fams[key].add(stem.split("_")[0])
    for k, v in tot.most_common():
        print(v, sorted(fams[k]), k)


if __name__ == "__main__":
    main(sys.argv[1:])
