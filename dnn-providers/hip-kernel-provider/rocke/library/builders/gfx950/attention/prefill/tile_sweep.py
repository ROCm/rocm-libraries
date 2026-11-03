# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Times every legal tile for every shape, to fit the gfx950 dense tile heuristic.

The hipDNN ``hipkernel:Gfx950AttentionDense`` catalog is 52 kernels: one per
(head_size, dtype, causal, tile), with seven tiles at D64 and six at D128. Its
scorer currently prefers the 256x64 tile for every problem, which was the right
default while the catalog baked the shape and is not once it does not. This
sweep measures which tile actually wins per shape so the scorer can be fitted
against data rather than intuition.

Two shape sources, which do different jobs:

* the LLM suite (``--suite``), real model shapes, carries the geomean claim;
* the grid (``--grid``), synthetic, carries catalog coverage. The suite alone
  reaches 19 of the 52 kernels because every servable shape in it is causal, so
  the grid is what keeps the other 33 from being fitted on nothing.

Correctness first: every timed config is checked against torch SDPA by the same
``run`` the prefill harness uses, and a row records its own error. A row that
fails the check is recorded and excluded from any fit.

**The JSONL is raw measurement data, not a publishable artifact.** Absolute
times and TFLOPS stay in it for analysis; anything derived for a PR, ticket or
doc reports relative ratios only (``platform/AGENTS.md``, Compliance).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

from builders.gfx950.attention.prefill.tile_sweep_validate import (  # noqa: E402
    catalog_tiles,
    divides,
    provisional_grid,
    shape_id,
    spec_for,
)

# Shapes the gfx950 dense engine declines, keyed by the suite field that says so.
# Each is a graph-match refusal or a catalog gap, not a tuning choice, so a shape
# carrying one is out of scope for the heuristic rather than a shape it ranks badly.
_SUITE_SKIP_KEYS = ("sinks", "kv_dtype", "varlen")


def load_suite(path):
    """Shapes from ``attention_shapes_llm.txt`` the dense catalog can serve.

    One ``attention {json}`` per line. Returns ``(shapes, skipped)`` where
    ``skipped`` counts the exclusions by reason, so a run can report what the
    suite covers instead of quietly shrinking.
    """
    shapes, skipped = [], {}

    def drop(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line.partition(" ")[2])
        if record["D"] not in (64, 128):
            drop(f"head_size={record['D']}")
            continue
        if record["mask"] not in ("causal", "full"):
            drop(f"mask={record['mask']}")
            continue
        if any(record.get(key) for key in _SUITE_SKIP_KEYS):
            drop(next(k for k in _SUITE_SKIP_KEYS if record.get(k)))
            continue
        shape = {
            "batch": record["B"],
            "seqlen_q": record["Sq"],
            "seqlen_kv": record["Skv"],
            "num_query_heads": record["Hq"],
            "num_kv_heads": record["Hkv"],
            "head_size": record["D"],
            "dtype": record["dtype"],
            "causal": record["mask"] == "causal",
            "source": "suite",
            "model": record.get("model", ""),
        }
        if not any(divides(shape, *t) for t in catalog_tiles(shape["head_size"])):
            # Sq=1 decode, which no prefill tile divides. Expected, not a gap.
            drop("no tile divides")
            continue
        shapes.append(shape)
    return shapes, skipped


def configs(shapes):
    """``(shape, block_m, block_n)`` for every tile that can serve its shape."""
    for shape in shapes:
        for block_m, block_n in catalog_tiles(shape["head_size"]):
            if divides(shape, block_m, block_n):
                yield shape, block_m, block_n


def sweep(plan, *, rounds: int, warmup: int, iters: int, seed: int):
    """Time every planned config once per round. Yields one record per (config, round).

    Rounds alternate over the whole config list rather than repeating each config
    back to back, so a thermal or clock drift during the run shows up as spread
    between rounds instead of biasing whichever tile happened to be measured
    while the device was cool.
    """
    import torch

    from builders.gfx950.attention.prefill.attention_dense_prefill import run

    # Recorded per row so the fit reads the part it ran on rather than assuming one.
    # A grid-fill rule is meaningless against the wrong CU count.
    cu_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    for round_index in range(rounds):
        for shape, block_m, block_n in plan:
            record = {
                "round": round_index,
                "shape": shape_id(shape),
                "source": shape.get("source", ""),
                "model": shape.get("model", ""),
                "batch": shape["batch"],
                "seqlen_q": shape["seqlen_q"],
                "seqlen_kv": shape["seqlen_kv"],
                "num_query_heads": shape["num_query_heads"],
                "num_kv_heads": shape["num_kv_heads"],
                "head_size": shape["head_size"],
                "dtype": shape["dtype"],
                "causal": shape["causal"],
                "block_m": block_m,
                "block_n": block_n,
                "cu_count": cu_count,
            }
            try:
                spec = spec_for(shape, block_m, block_n)
                record["kernel"] = spec.kernel_name()
                ms, tflops, err = run(
                    spec, warmup=warmup, iters=iters, check=True, seed=seed
                )
                record.update(ms=ms, tflops=tflops, max_abs_err=err, status="ok")
            except Exception as exc:  # one bad config must not end the sweep
                record.update(status="error", detail=f"{type(exc).__name__}: {exc}")
            yield record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", help="attention_shapes_llm.txt")
    ap.add_argument(
        "--grid",
        action="store_true",
        help="include the synthetic coverage grid (needed to reach all 52 kernels)",
    )
    ap.add_argument("--rounds", type=int, default=3, help="alternating timing rounds")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N configs per round")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N shapes")
    ap.add_argument("--dry-run", action="store_true", help="plan only, launch nothing")
    args = ap.parse_args(argv)

    shapes = []
    if args.suite:
        suite, skipped = load_suite(args.suite)
        shapes += suite
        print(f"suite: {len(suite)} servable shapes")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"   excluded {count:4d}  {reason}")
    if args.grid:
        grid = [dict(s, source="grid") for s in provisional_grid()]
        grid = [
            s
            for s in grid
            if any(divides(s, *t) for t in catalog_tiles(s["head_size"]))
        ]
        shapes += grid
        print(f"grid: {len(grid)} servable shapes")
    if not shapes:
        ap.error("no shapes: pass --suite and/or --grid")

    # The grid is built to cover the catalog, so it restates shapes the suite already
    # has. Timing one twice wastes device time and, worse, double-weights it in any
    # geomean over the result. First wins, which keeps the suite's model label.
    seen = set()
    deduped = []
    for shape in shapes:
        key = shape_id(shape)
        if key not in seen:
            seen.add(key)
            deduped.append(shape)
    if len(deduped) != len(shapes):
        print(f"deduped: {len(shapes) - len(deduped)} shape(s) in both sources")
    shapes = deduped

    plan = list(configs(shapes[args.offset :]))
    if args.limit:
        plan = plan[: args.limit]
    kernels = {(s["head_size"], s["dtype"], s["causal"], bm, bn) for s, bm, bn in plan}
    planned_shapes = {shape_id(s) for s, _, _ in plan}
    print(
        f"plan: {len(plan)} configs x {args.rounds} rounds "
        f"over {len(planned_shapes)} shapes, reaching {len(kernels)} of 52 kernels"
    )
    if args.dry_run:
        return 0

    started = time.time()
    ok = failed = 0
    with open(args.output_jsonl, "w") as out:
        for record in sweep(
            plan,
            rounds=args.rounds,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        ):
            out.write(json.dumps(record) + "\n")
            out.flush()
            if record["status"] == "ok":
                ok += 1
            else:
                failed += 1
    print(
        f"done in {time.time() - started:.0f}s   timed: {ok}   failed: {failed}"
        f"   -> {args.output_jsonl}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
