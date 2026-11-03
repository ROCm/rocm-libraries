#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which shards of a finished run need another ``--resume`` link (login-node tool, needs no GPU or torch).

    completion.py <results-dir> [--shards-file FILE]

``<results-dir>`` holds the shard directories of one run (``<suite>-<k>of<N>``, each a
``run_aotriton.py --out`` directory). A shard needs a link when work its ``--resume`` would still do
remains: a collected case of pass 1 without a row, a case of a pass directory's ``nodeids.txt`` without
a row, or a pass the shard's pass-1 rows call for (``run_aotriton.engine_work``, and the reference-forward
passes of ``forward_failed``) whose directory does not exist or lacks rows. A shard whose
``collected.txt`` is missing or empty left no list of cases and goes back whole. With
``--shards-file`` (``<suite> <k> <N>`` per line) a shard without a directory is listed too.

stdout: the shards to resubmit in ``submit_chains.sh --shards-file`` format, in file (else directory)
order. stderr: ``<shard>: <reason>`` lines. Exit 0 whatever the list holds; 2 on usage errors.
"""

import argparse
import sys
from pathlib import Path

import reconcile
import run_aotriton

MAX_REASONS = 5  # reasons printed per shard; the rest are counted


def shard_gaps(shard):
    """Reasons ``shard`` needs another resumed link; empty when it is complete."""
    shard = Path(shard)
    collected = reconcile.listing(shard / reconcile.COLLECTED_NAME)
    if not collected:
        return ["collected.txt is missing or empty"]
    try:
        rows = reconcile.read_rows(shard / reconcile.CSV_NAME)
        wanted = {shard: collected}
        for d, _ in reconcile.pass_dirs(shard):
            wanted[d] = reconcile.listing(d / reconcile.NODEIDS_NAME)

        def plan(rel, nodeids):
            d = shard / reconcile.ENGINES_DIR / rel
            wanted[d] = list(dict.fromkeys([*wanted.get(d, ()), *nodeids]))

        for (direction, engine), ids in run_aotriton.engine_work(rows).items():
            plan(Path(direction) / run_aotriton.engine_slug(engine), ids)
        failed = run_aotriton.forward_failed(shard)
        if failed:
            refwd = Path("refwd")
            plan(refwd / "discovery", failed)
            discovery = reconcile.read_rows(
                shard / reconcile.ENGINES_DIR / refwd / "discovery" / reconcile.CSV_NAME
            )
            for (_, engine), ids in run_aotriton.engine_work(discovery).items():
                plan(refwd / run_aotriton.engine_slug(engine), ids)
        gaps = []
        for d, ids in sorted(wanted.items()):
            name = reconcile.pass_name(shard, d)
            if not d.is_dir():
                gaps.append(f"{name}: not started ({len(ids)} cases)")
                continue
            recorded = run_aotriton.recorded_ids(d)
            missing = len([n for n in ids if n not in recorded])
            if missing:
                gaps.append(f"{name}: {missing} of {len(ids)} cases have no row")
        return gaps
    except ValueError as e:
        return [f"results unreadable: {e}"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--shards-file", type=Path)
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        print(f"no such results dir: '{args.results_dir}'", file=sys.stderr)
        return 2
    names = (
        reconcile.read_shards_file(args.shards_file)
        if args.shards_file
        else [d.name for d in reconcile.shard_dirs(args.results_dir)]
    )
    for name in names:
        shard = args.results_dir / name
        gaps = shard_gaps(shard) if shard.is_dir() else ["no shard directory"]
        if not gaps:
            continue
        suite, k, n = reconcile.SHARD_DIR.match(name).groups()
        print(f"{suite} {k} {n}")
        for gap in gaps[:MAX_REASONS]:
            print(f"{name}: {gap}", file=sys.stderr)
        if len(gaps) > MAX_REASONS:
            print(f"{name}: ... {len(gaps) - MAX_REASONS} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
