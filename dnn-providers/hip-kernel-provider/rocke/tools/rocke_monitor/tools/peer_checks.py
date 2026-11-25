# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A review's peer checks (rkmon_lib/peer_checks.py), one per run or `all`; read-only.
Usage: python3 -I peer_checks.py CHECK --repo R --head SHA [--base REV] [--rocke-path K]
       [--meta FILE | --title TEXT] [--slug OWNER/NAME --pr N | --threads-json FILE] [--allow P,P]
       [--items-json FILE]
Prints the section, then one last line: the check's name in capitals, then "pass|fail|skip <detail>". With
--items-json, also writes each check's result as JSON (check -> its items, title and error), which a merge-only head's
threads, title and prnumbers flags are compared on (rkmon_lib/peer_carry.py)."""
import argparse
import json
import pathlib
import sys

sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rkmon_lib import prtree  # noqa: E402
from rkmon_lib.peer_checks import CHECKS, render, run, tracker_allow  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=sorted(CHECKS) + ["all"])
    ap.add_argument("--repo")
    ap.add_argument("--head")
    ap.add_argument("--base", default=prtree.DEVELOP)
    ap.add_argument(
        "--rocke-path",
        default="",
        help="default: the configuration's folder; the title check reads no tree and needs none",
    )
    ap.add_argument(
        "--meta",
        default=None,
        help="the staged pr/meta.md (its first line has the title)",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--slug", default=None, help="OWNER/NAME, for the threads")
    ap.add_argument("--pr", type=int, default=None)
    ap.add_argument(
        "--threads-json",
        default=None,
        help="PR data as rkmon_lib.followups.read returns it",
    )
    ap.add_argument(
        "--allow",
        default=None,
        help="P,P: tracker-like prefixes that are not trackers (the package's by default: config tracker_allow)",
    )
    ap.add_argument(
        "--items-json",
        default=None,
        help="FILE: also write each check's result (its items, the title, any error) as JSON, by check",
    )
    a = ap.parse_args(argv)
    if a.check != "title":
        try:
            a.rocke_path = prtree.folder_argument(a.rocke_path)
        except argparse.ArgumentTypeError as exc:
            ap.error(f"argument --rocke-path: {exc}")
    a.allow = (
        tuple(x.strip() for x in a.allow.split(",") if x.strip())
        if a.allow
        else tracker_allow({})
    )
    todo = sorted(CHECKS, key=list(CHECKS).index) if a.check == "all" else [a.check]
    results = {}
    for i, c in enumerate(todo):
        if a.check == "all":
            print(("\n" if i else "") + f"## {CHECKS[c][2]}\n")
        results[c] = run(c, a)
        print("\n".join(render(results[c])))
    if a.items_json:
        pathlib.Path(a.items_json).write_text(
            json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
