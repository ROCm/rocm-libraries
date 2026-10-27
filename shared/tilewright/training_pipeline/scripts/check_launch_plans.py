#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check TensileLite's launch plans and keys against the pipeline's launch mirror.

MLREC_v3 models train with `lib/hipblaslt_launch.launch_plan` and
`launch_key`; TensileLite ranks them with the plans and keys of its own
launch code (TilewrightLaunch). This script writes the pool of a library for
tensilelite-tests, which dumps the launch key of every pool kernel and the
plans of random problems and execution contexts for it, then recomputes
every dumped key with `launch_key` and plan with `launch_plan` and reports
each one that differs.

One class of differing plans is known and counted apart (`known_class`):
a static Stream-K launch of batch 1 with one K iteration per tile and more
tiles than the CUs it may use, where TensileLite takes the grid Origami's
select_grid_size chooses, which can differ from launch rule version 1's
(lib/hipblaslt_launch.py). Such plans are reported but do not fail the
check; every other difference does.

Usage:
  check_launch_plans.py pool OUT --library-dir DIR --library-stem STEM
      --processor gfx950 --cu-count 256 [--chip-id 0x75a0] [--bank-stem STEM ...]
  TILEWRIGHT_LAUNCH_PLANS_POOL=OUT TILEWRIGHT_LAUNCH_PLANS_OUT=PLANS \\
      tensilelite-tests --gtest_filter=TilewrightLaunchTest.DumpsPoolPlans
  check_launch_plans.py compare PLANS --library-dir DIR --library-stem STEM
      --processor gfx950 --cu-count 256 [--chip-id 0x75a0] [--bank-stem STEM ...]

Without --bank-stem the bank is every file of the problem type. Exit code 0
when every key and plan matches but plans of the known class, 1 on another
mismatch, 2 on bad input.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from lib import bank_pool, hipblaslt_launch  # noqa: E402
from lib import features as fs  # noqa: E402
from lib.dat import library_logic_path  # noqa: E402
from lib.evaluate import pool_from_kernels  # noqa: E402
from lib.hardware import DATATYPE_NAME  # noqa: E402

SCHEDULES = {0: "default", 1: "dynamic", 2: "auto"}
FIELDS = ("dynamic_queue", "grid_wgs", "partial_reduction")


@dataclass
class Pool:
    stems: List[str]
    kernels: List[Dict[str, Any]]


def library_pool(args: argparse.Namespace) -> Pool:
    device = bank_pool.BankDevice(
        processor=args.processor, cu_count=args.cu_count, chip_id=args.chip_id
    )
    lp = bank_pool.library_pool(
        args.library_dir, args.library_stem, device, args.bank_stem or None
    )
    stems = [args.library_stem] + [s for s in lp.bank.stems if s != args.library_stem]
    return Pool(stems, pool_from_kernels(lp.kernels))


def pool_file(pool: Pool, library_dir: Path) -> str:
    """The pool file DumpsPoolPlans reads: the logic files (without their
    .dat[.zlib] suffix), then the solution indices in pool order."""
    lines = []
    for stem in pool.stems:
        path = Path(library_logic_path(library_dir, stem))
        name = path.name
        for suffix in (".dat.zlib", ".dat"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        lines.append(f"library {path.parent.resolve() / name}")
    lines.append(
        "pool " + " ".join(str(int(k["sol_idx_global"])) for k in pool.kernels)
    )
    return "\n".join(lines) + "\n"


@dataclass
class Row:
    pos: int
    solution: int
    problem: Dict[str, Any]
    context: fs.ExecutionContext
    plan: Tuple[bool, int, str]
    kernel: Tuple[int, int, int, int]


@dataclass
class KeyRow:
    pos: int
    solution: int
    key: int


def parse_keys(lines: Iterable[str]) -> List[KeyRow]:
    """The "key <pos> <solution> <launch key>" lines of a dump."""
    rows = []
    for line in lines:
        if not line.startswith("key "):
            continue
        w = line.split()
        if len(w) != 4:
            raise ValueError(f"bad key line: {line!r}")
        rows.append(KeyRow(pos=int(w[1]), solution=int(w[2]), key=int(w[3])))
    return rows


def parse_plans(lines: Iterable[str]) -> List[Row]:
    rows = []
    for line in lines:
        if not line.strip() or line.startswith(("#", "key ")):
            continue
        w = line.split()
        if len(w) != 16:
            raise ValueError(f"bad plan line: {line!r}")
        v = [int(x) for x in w[:11]]
        mi = DATATYPE_NAME.get(v[6])
        if mi is None:
            raise ValueError(f"unknown mi_dtype {v[6]} in {line!r}")
        rows.append(
            Row(
                pos=v[0],
                solution=v[1],
                problem=dict(m=v[2], n=v[3], k=v[4], batch=v[5], mi_dtype=mi),
                context=fs.ExecutionContext(cu_budget=v[7], schedule=SCHEDULES[v[8]]),
                plan=(bool(v[9]), v[10], w[11]),
                kernel=tuple(int(x) for x in w[12:16]),
            )
        )
    return rows


KNOWN_CLASS = "one K iteration per tile, batch 1, static Stream-K"


def known_class(
    row: Row, kernel: Mapping[str, Any], got: Tuple[bool, int, str], n_cu: int
) -> bool:
    """Whether a plan that differs from `launch_plan`'s (`got`) is of the
    known class: a Stream-K kernel, batch 1, one K iteration per tile (K at
    most the kernel's depth), more output tiles than the CUs the launch may
    use, and a static launch in both plans, where TensileLite takes the grid
    Origami's select_grid_size chooses."""
    attrs = kernel.get("attributes") or {}
    if int(attrs.get("tile_processing_strategy", 0)) != 2:
        return False
    if row.plan[0] or got[0] or int(row.problem.get("batch", 1)) != 1:
        return False
    k, mt_k = int(row.problem["k"]), max(int(kernel["mt_k"]), 1)
    if k > mt_k:
        return False
    tiles = fs.output_tiles(row.problem, kernel)
    granule = hipblaslt_launch.SM_TARGET_GRANULE
    sm_target = int(row.context.cu_budget) // granule * granule
    cu_count = sm_target if 0 < sm_target < int(n_cu) else int(n_cu)
    return tiles > cu_count


@dataclass
class PlanCheck:
    rows: int = 0
    kernels: int = 0
    field_mismatches: Dict[str, int] = field(default_factory=dict)
    kernel_mismatches: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    # Differing plans of the known class (`known_class`): reported, not failing.
    known_field_mismatches: Dict[str, int] = field(default_factory=dict)
    known_examples: List[str] = field(default_factory=list)

    @property
    def mismatched_rows(self) -> int:
        return len(self.examples)

    @property
    def known_rows(self) -> int:
        return len(self.known_examples)

    @property
    def ok(self) -> bool:
        return not self.examples and not self.kernel_mismatches


def check_plans(
    rows: Sequence[Row],
    pool: Sequence[Mapping[str, Any]],
    n_cu: int,
    arch: str,
) -> PlanCheck:
    out = PlanCheck(rows=len(rows), kernels=len({r.pos for r in rows}))
    out.field_mismatches = {f: 0 for f in FIELDS}
    out.known_field_mismatches = {f: 0 for f in FIELDS}
    for r in rows:
        if r.pos >= len(pool):
            out.kernel_mismatches.append(f"pos {r.pos}: past the pool's end")
            continue
        kernel = pool[r.pos]
        want = (
            int(kernel["sol_idx_global"]),
            int(kernel["mt_m"]),
            int(kernel["mt_n"]),
            int(kernel["mt_k"]),
            int(kernel["occupancy"]),
        )
        if (r.solution,) + r.kernel != want:
            out.kernel_mismatches.append(
                f"pos {r.pos}: TensileLite has {(r.solution,) + r.kernel}, "
                f"the pool {want}"
            )
            continue
        p = hipblaslt_launch.launch_plan(
            r.problem, kernel, kernel.get("attributes"), r.context, n_cu, arch
        )
        got = (bool(p.dynamic_queue), int(p.grid_wgs), str(p.partial_reduction))
        bad = [f for f, a, b in zip(FIELDS, r.plan, got) if a != b]
        if not bad:
            continue
        known = known_class(r, kernel, got, n_cu)
        counts = out.known_field_mismatches if known else out.field_mismatches
        for f in bad:
            counts[f] += 1
        (out.known_examples if known else out.examples).append(
            f"pos {r.pos} solution {r.solution} {r.problem} "
            f"cu_budget={r.context.cu_budget} schedule={r.context.schedule}: "
            f"TensileLite {r.plan}, launch_plan {got}"
        )
    return out


@dataclass
class KeyCheck:
    kernels: int = 0
    mismatches: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def check_keys(keys: Sequence[KeyRow], pool: Sequence[Mapping[str, Any]]) -> KeyCheck:
    """Every pool kernel has one key, equal to its `launch_key`."""
    out = KeyCheck(kernels=len(keys))
    seen = set()
    for r in keys:
        if r.pos >= len(pool) or r.pos in seen:
            out.mismatches.append(f"pos {r.pos}: past the pool's end or listed twice")
            continue
        seen.add(r.pos)
        kernel = pool[r.pos]
        if r.solution != int(kernel["sol_idx_global"]):
            out.mismatches.append(
                f"pos {r.pos}: TensileLite has solution {r.solution}, "
                f"the pool {int(kernel['sol_idx_global'])}"
            )
            continue
        want = hipblaslt_launch.launch_key(kernel, kernel.get("attributes"))
        if r.key != want:
            out.mismatches.append(
                f"pos {r.pos} solution {r.solution}: TensileLite key {r.key:#018x}, "
                f"launch_key {want:#018x}"
            )
    missing = len(pool) - len(seen)
    if missing:
        out.mismatches.append(f"pool kernels without a key: {missing}")
    return out


def _dedupe(kernel_mismatches: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(kernel_mismatches))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("pool", "compare"))
    ap.add_argument("path", type=Path, help="pool file to write, or plans to read")
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument("--library-stem", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--cu-count", type=int, required=True)
    ap.add_argument("--chip-id", type=lambda v: int(v, 0))
    ap.add_argument("--bank-stem", action="append", default=[])
    ap.add_argument(
        "--arch", help="arch of the launch model (default: --processor)", default=None
    )
    ap.add_argument("--show", type=int, default=20, help="mismatches to list")
    args = ap.parse_args(argv)
    try:
        pool = library_pool(args)
        if args.command == "pool":
            args.path.write_text(pool_file(pool, args.library_dir))
            print(f"pool: {len(pool.kernels)} kernels from {len(pool.stems)} files")
            return 0
        lines = args.path.read_text().splitlines()
        keys = parse_keys(lines)
        rows = parse_plans(lines)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    key_result = check_keys(keys, pool.kernels)
    result = check_plans(rows, pool.kernels, args.cu_count, args.arch or args.processor)
    print(f"pool: {len(pool.kernels)} kernels")
    print(f"keys: {key_result.kernels}, differing: {len(key_result.mismatches)}")
    for line in key_result.mismatches[: args.show]:
        print(f"  {line}")
    print(f"plans: {result.rows} of {result.kernels} kernels")
    kernel_mismatches = _dedupe(result.kernel_mismatches)
    print(f"kernels that differ from the pool: {len(kernel_mismatches)}")
    for line in kernel_mismatches[: args.show]:
        print(f"  {line}")
    print(f"plans that differ: {result.mismatched_rows}")
    for f in FIELDS:
        print(f"  {f}: {result.field_mismatches.get(f, 0)}")
    for line in result.examples[: args.show]:
        print(f"  {line}")
    print(f"plans that differ in the known class ({KNOWN_CLASS}): {result.known_rows}")
    for f in FIELDS:
        print(f"  {f}: {result.known_field_mismatches.get(f, 0)}")
    for line in result.known_examples[: args.show]:
        print(f"  {line}")
    ok = key_result.ok and result.ok and result.kernels == len(pool.kernels)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
