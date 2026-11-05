#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check a bench log's tilewright picks against the pipeline's kernel pool.

With `TILEWRIGHT_PICK_LOG=1` hipBLASLt prints a `[TILEWRIGHT_PICK]` line per
ranked problem whose `top1_index` is a position in the pool it ranked, and
hipblaslt-bench then prints the solution it ran (`--Solution index: N`). This
script builds the pool the pipeline uses (`lib/bank_pool.py`) and reports
every problem whose pool position does not name the solution that ran, and
every `n_configs` that differs from the pool size.

Usage:
  check_pick_log.py LOG [LOG ...] --library-dir DIR --library-stem STEM
      (--config-yaml CFG [--hardware-device N]
       | --processor gfx950 --cu-count 256 [--chip-id 0x75a0]
         [--bank-stem STEM ...])
      [--blocklist FILE] [--guards FILE]

With --config-yaml the bank is the one the config asks for on the device;
otherwise it is taken from --bank-stem files, or every file of the problem
type. The pool leaves out the kernels the runtime left out of the pool it
ranked (lib/correctness.py): those of --blocklist (a sidecar, a list or a
blocklist file), else of the sidecar `<STEM>.tilewright.blocklist` in
--library-dir, which the runtime reads next to the model, else of the
config's blocklist. A model trained with kernel guards is served with the
guards sidecar (--guards, else `<STEM>.tilewright.guards` in --library-dir
when the model the directory's `tilewright_index` names for STEM, on the
devices of --chip-id, declares them; the runtime reads the guards of no
other model): its kernels stay in the pool, and the engine's pick line
names the engine's top-1, which the runtime passes over where its guards
deny the problem. Such a pick whose bench ran another solution is counted
apart (`passed over below its guard`), and one where the bench ran a
guarded kernel below its guard is a violation. Exit code 0 when every pick
matches, 1 on a mismatch or a violation, 2 on bad input.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import yaml  # noqa: E402

from lib import bank_pool  # noqa: E402
from lib import correctness as cx  # noqa: E402
from lib import mlrec  # noqa: E402
from lib import tilewright_index as twi  # noqa: E402

_PICK_RE = re.compile(
    r"\[TILEWRIGHT_PICK\]\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)\s+b=(\d+)\s.*?"
    r"top1_index=(\d+)\s+n_configs=(\d+)"
)
_SOLUTION_RE = re.compile(r"^\s*--Solution index:\s*(-?\d+)")


@dataclass(frozen=True)
class Pick:
    shape: Tuple[int, int, int, int]
    top1_index: int
    n_configs: int
    solution: int


@dataclass
class PickCheck:
    picks: List[Pick] = field(default_factory=list)
    without_pick: int = 0
    n_configs: List[int] = field(default_factory=list)
    mismatches: List[Tuple[Pick, Optional[int]]] = field(default_factory=list)
    passed_over: List[Pick] = field(default_factory=list)
    violations: List[Pick] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.violations


def parse_picks(lines: Iterable[str]) -> Tuple[List[Pick], int]:
    """(picks paired with the solution index printed after them, number of
    solution indices without a pick line before them)."""
    picks: List[Pick] = []
    pending: Optional[re.Match] = None
    without = 0
    for line in lines:
        m = _PICK_RE.search(line)
        if m:
            pending = m
            continue
        s = _SOLUTION_RE.match(line)
        if s is None:
            continue
        if pending is None:
            without += 1
            continue
        g = [int(x) for x in pending.groups()]
        picks.append(Pick(tuple(g[:4]), g[4], g[5], int(s.group(1))))
        pending = None
    return picks, without


def check_picks(
    lines: Iterable[str],
    pool: Sequence[int],
    guards: Optional[Mapping[int, Any]] = None,
) -> PickCheck:
    """Map every pick's `top1_index` through `pool` (solution indices in pool
    order). `guards` ({solution: guards}, lib/correctness.py) are the guarded
    kernels of the pool: a pick naming one whose guards deny the problem is
    passed over by the runtime, and a bench that ran one there violates
    them."""
    picks, without = parse_picks(lines)
    out = PickCheck(picks=picks, without_pick=without)
    out.n_configs = sorted({p.n_configs for p in picks})
    guards = guards or {}

    def denied(sid: Optional[int], p: Pick) -> bool:
        g = guards.get(sid) if sid is not None else None
        return bool(g) and not cx.guards_admit(g, p.shape[0], p.shape[1])

    for p in picks:
        named = pool[p.top1_index] if p.top1_index < len(pool) else None
        if denied(p.solution, p):
            out.violations.append(p)
        elif named != p.solution and denied(named, p):
            out.passed_over.append(p)
        elif named != p.solution:
            out.mismatches.append((p, named))
    return out


def served_with_guards(library_dir: Path, stem: str, chip: Optional[int]) -> bool:
    """Whether the model `library_dir`'s tilewright_index names for `stem`
    on a device of PCI chip ID `chip` (None: unknown) declares kernel guards
    (lib/mlrec.py `with_kernel_guards`); false when no model is named or it
    cannot be read."""
    try:
        index = twi.parse(twi.read_text(Path(library_dir) / twi.INDEX_NAME))
        name = index.model_of(stem, int(chip or 0))
        if name is None:
            return False
        header = mlrec.read_header((Path(library_dir) / name).read_bytes())
    except (OSError, ValueError):
        return False
    return bool((header.get("serving") or {}).get("kernel_guards"))


def blocklist_of(
    args: argparse.Namespace, cfg: Optional[Mapping[str, Any]]
) -> Tuple[Optional[cx.Blocklist], str]:
    """(the kernels the runtime left out of the pool, where they come from):
    --blocklist, else the sidecar next to the model in --library-dir, else
    the config's blocklist; narrowed by --guards, else by the guards sidecar
    next to the model when the model declares kernel guards. ValueError when
    a file cannot be read."""
    guards = args.guards
    if guards is None:
        found = Path(args.library_dir) / cx.guards_name(args.library_stem)
        if found.is_file() and served_with_guards(
            args.library_dir, args.library_stem, args.chip_id
        ):
            guards = found
    if args.blocklist is not None:
        listed = cx.read_list(args.blocklist)
        if guards is not None:
            listed = cx.narrowed(listed, cx.read_guards(guards), str(guards))
        return listed, str(args.blocklist) + (f" with {guards}" if guards else "")
    side = Path(args.library_dir) / cx.sidecar_name(args.library_stem)
    if side.is_file():
        return cx.sidecar_blocklist(side, guards), str(side) + (
            f" with {guards}" if guards else ""
        )
    if cfg is not None:
        return cx.config_blocklist(cfg), "the config's blocklist"
    return None, "none"


def _pool(args: argparse.Namespace) -> Tuple[List[int], str, Dict[int, Any]]:
    cfg = None
    if args.config_yaml is not None:
        cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
    blocklist, where = blocklist_of(args, cfg)
    if cfg is not None:
        pool = bank_pool.config_pool(
            cfg,
            args.library_dir,
            args.library_stem,
            args.hardware_device,
            apply_blocklist=False,
        )
    else:
        device = bank_pool.BankDevice(
            processor=args.processor, cu_count=args.cu_count, chip_id=args.chip_id
        )
        pool = bank_pool.library_pool(
            args.library_dir, args.library_stem, device, args.bank_stem or None
        )
    kept = bank_pool.without_blocked(pool, blocklist)
    n_blocked = len(pool.kernels) - len(kept.kernels)
    guarded = {
        int(k["sol_idx_global"]): k[cx.POOL_GUARDS]
        for k in kept.kernels
        if k.get(cx.POOL_GUARDS)
    }
    text = f"{n_blocked} kernel(s) left out ({where})"
    if guarded:
        text += f", {len(guarded)} guarded"
    return kept.indices, text, guarded


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--library-dir", type=Path, required=True)
    ap.add_argument("--library-stem", required=True)
    ap.add_argument("--config-yaml", type=Path)
    ap.add_argument("--hardware-device", type=int, default=0)
    ap.add_argument("--processor")
    ap.add_argument("--cu-count", type=int)
    ap.add_argument("--chip-id", type=lambda v: int(v, 0))
    ap.add_argument("--bank-stem", action="append", default=[])
    ap.add_argument(
        "--blocklist",
        type=Path,
        help="kernels the runtime left out (default: the sidecar next to the "
        "model in --library-dir, else the config's blocklist)",
    )
    ap.add_argument(
        "--guards",
        type=Path,
        help="the guards sidecar the runtime served a model trained with kernel "
        "guards with (default: the one next to the model in --library-dir, "
        "when it is there and the model declares kernel guards)",
    )
    ap.add_argument("--show", type=int, default=20, help="mismatches to list")
    args = ap.parse_args(argv)
    if args.config_yaml is None and (args.processor is None or args.cu_count is None):
        ap.error("give --config-yaml, or --processor and --cu-count")
    try:
        pool, blocked, guarded = _pool(args)
        lines: List[str] = []
        for log in args.logs:
            lines += log.read_text(errors="replace").splitlines()
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    result = check_picks(lines, pool, guarded)
    print(f"pool: {len(pool)} kernels; blocklist: {blocked}")
    print(f"picks: {len(result.picks)} with a solution index after them")
    print(f"solution indices without a pick line: {result.without_pick}")
    sizes = ", ".join(str(n) for n in result.n_configs) or "-"
    print(f"n_configs in the log: {sizes}")
    if any(n != len(pool) for n in result.n_configs):
        print("n_configs differs from the pool size")
    if guarded:
        print(f"passed over below its guard: {len(result.passed_over)}")
        print(f"guarded kernels run below their guards: {len(result.violations)}")
        for p in result.violations[: args.show]:
            m, n, k, b = p.shape
            print(f"  m={m} n={n} k={k} b={b}: the bench ran {p.solution}")
    print(f"mismatches: {len(result.mismatches)}")
    for p, named in result.mismatches[: args.show]:
        m, n, k, b = p.shape
        print(
            f"  m={m} n={n} k={k} b={b} top1_index={p.top1_index}: pool names "
            f"{named}, the bench ran {p.solution}"
        )
    return 0 if result.ok and all(n == len(pool) for n in result.n_configs) else 1


if __name__ == "__main__":
    sys.exit(main())
