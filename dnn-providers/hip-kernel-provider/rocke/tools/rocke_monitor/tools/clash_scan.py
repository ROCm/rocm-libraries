# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The clash section of a review's checks: PR N at its head against every other open PR, both land orders, and the
silent clashes that merge clean (rkmon_lib/clashes.py). Reads the state and the mirror; writes only temporary merge
objects (no ref) and the facts cache.

Usage: python3 -I clash_scan.py N [--head SHA] [--heads N=SHA,N=SHA] [--develop REV] [--others N,N] [--budget S]
                                  [--max-fetch K] [--cache DIR] [--state FILE] [--config FILE] [--json FILE]
--heads sets other PRs' heads (a snapshot's; a PR the state does not list as open joins with it), --others limits
the PRs compared. Prints the section, then one last line: "CLASH pass|fail|skip <detail>".
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import clashes, config  # noqa: E402


def _numbers(s: str) -> list[int]:
    return [int(x.strip().lstrip("#")) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int)
    ap.add_argument("--head", default=None, help="PR N's head (default: the state's)")
    ap.add_argument("--heads", default="", help="N=SHA,N=SHA: other PRs' heads")
    ap.add_argument(
        "--develop",
        default=None,
        help="the develop commit (default: the mirror's develop)",
    )
    ap.add_argument("--others", default=None, help="N,N: compare only these PRs")
    ap.add_argument(
        "--budget",
        type=float,
        default=None,
        help="seconds (default: config clashes.budget_s)",
    )
    ap.add_argument(
        "--max-fetch", type=int, default=None, help="PRs whose blobs may be fetched"
    )
    ap.add_argument(
        "--cache",
        default=None,
        help="a directory for each PR's facts, by head and merge base",
    )
    ap.add_argument(
        "--state", default=None, help="default: the monitor home's state/state.json"
    )
    ap.add_argument(
        "--config",
        default=None,
        help="default: the monitor home's config.json (RKMON_HOME)",
    )
    ap.add_argument(
        "--json", default=None, help="also write the raw result to this file"
    )
    args = ap.parse_args(argv)
    cfg = config.load(
        pathlib.Path(args.config) if args.config else None, check=False
    )  # its paths resolved
    st = json.loads(
        pathlib.Path(args.state or config.state_dir() / "state.json").read_text()
    )
    heads = {}
    for item in filter(None, (x.strip() for x in args.heads.split(","))):
        k, _, sha = item.partition("=")
        heads[int(k.lstrip("#"))] = sha.strip()
    if args.head:
        heads[args.n] = args.head
    res, lines = clashes.run(
        cfg,
        st,
        args.n,
        heads=heads,
        develop=args.develop,
        others=_numbers(args.others) if args.others is not None else None,
        cache=args.cache,
        budget_s=args.budget,
        max_fetch=args.max_fetch,
    )
    print("\n".join(lines))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, default=sorted, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
