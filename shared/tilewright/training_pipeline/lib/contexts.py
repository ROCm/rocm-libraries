# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execution contexts the pipeline benches GEMMs under.

A context is what the engine's `ExecutionContext` describes: the CUs the
kernels may use (`cu_budget`, 0 = every CU) and the tile schedule. The bench
reproduces one with

  * `--sm_count_target <cu_budget>`: hipBLASLt sizes kernel selection and
    persistent grids for that many CUs (process wide);
  * `--hybrid_assignment_policy DynamicWorkQueue|Auto` for the dynamic and
    auto schedules;
  * with `cotenant`, `hipblaslt-cotenant --cus <device CUs - cu_budget>`
    around the bench: a resident busy kernel holds the other CUs, so the
    kernels really run on `cu_budget` CUs.

The cotenant pins its kernel to the device named by `--device` through
HIP_VISIBLE_DEVICES, so the bench it starts sees that device as device 0;
`bench_command` resolves the bench's device index through the
HIP_VISIBLE_DEVICES the stage runs with, so both stay on the stage's devices.

The exclusive context (every CU, default schedule, no cotenant) is the
round's own stage02 / stage03 bench; the contexts here are the others. A
`holdout` context is benched and enriched separately and never trained on:
stage04b reports the model on it (interpolation between trained budgets).
"""
from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEDULES: Tuple[str, ...] = ("default", "dynamic", "auto")
HYBRID_POLICY = {"dynamic": "DynamicWorkQueue", "auto": "Auto"}
SPEC_FILE = "context.json"
WRAPPER_NAME = "hipblaslt-bench"
ShapeKey = Tuple[int, int, int, int]


@dataclass(frozen=True)
class BenchContext:
    cu_budget: int = 0
    schedule: str = "default"
    cotenant: bool = False
    fraction: float = 1.0
    holdout: bool = False
    name: Optional[str] = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        base = f"cu{self.cu_budget}" if self.cu_budget else "allcu"
        if self.cotenant:
            base += "_cot"
        if self.schedule != "default":
            base += f"_{self.schedule}"
        return base

    def is_exclusive(self) -> bool:
        return self.cu_budget == 0 and self.schedule == "default" and not self.cotenant

    def cotenant_cus(self, n_cu: int) -> int:
        return max(0, int(n_cu) - self.cu_budget) if self.cotenant else 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["name"] = self.label
        return d

    @classmethod
    def from_mapping(
        cls, m: Mapping[str, Any], holdout: bool = False
    ) -> "BenchContext":
        return cls(
            cu_budget=int(m.get("cu_budget", 0) or 0),
            schedule=str(m.get("schedule", "default") or "default"),
            cotenant=bool(m.get("cotenant", False)),
            fraction=float(m.get("fraction", 1.0)),
            holdout=bool(m.get("holdout", holdout)),
            name=m.get("name") or None,
        )


def contexts_from_config(cfg: Mapping[str, Any]) -> List[BenchContext]:
    """The non-exclusive contexts of `contexts.list` and every
    `contexts.holdout` entry, in config order."""
    sec = cfg.get("contexts") or {}
    out = [BenchContext.from_mapping(m) for m in sec.get("list") or []]
    out = [c for c in out if not c.is_exclusive()]
    out += [
        BenchContext.from_mapping(m, holdout=True) for m in sec.get("holdout") or []
    ]
    return out


def context_errors(
    cfg: Mapping[str, Any], n_cu: Optional[int] = None, where: str = "contexts"
) -> List[str]:
    """Rules the schema cannot state. `n_cu` (when known) bounds budgets."""
    sec = cfg.get("contexts") or {}
    errors: List[str] = []
    seen: Dict[Tuple[int, str], str] = {}
    labels: Dict[str, str] = {}
    for key in ("list", "holdout"):
        for i, m in enumerate(sec.get(key) or []):
            if not isinstance(m, Mapping):
                continue
            at = f"{where}.{key}[{i}]"
            c = BenchContext.from_mapping(m, holdout=key == "holdout")
            if c.cotenant and c.cu_budget <= 0:
                errors.append(f"{at}: cotenant needs a cu_budget > 0")
            if c.is_exclusive():
                if key == "holdout":
                    errors.append(f"{at}: the exclusive context cannot be held out")
                elif c.fraction != 1.0:
                    errors.append(
                        f"{at}: the exclusive context benches every shape "
                        f"(fraction must be 1)"
                    )
                continue
            if n_cu is not None and c.cu_budget >= n_cu:
                errors.append(
                    f"{at}: cu_budget {c.cu_budget} must be below the device's "
                    f"{n_cu} CUs (0 means every CU)"
                )
            ident = (c.cu_budget, c.schedule)
            if ident in seen:
                errors.append(
                    f"{at}: cu_budget {c.cu_budget} / schedule {c.schedule} is "
                    f"also {seen[ident]}; rows carry only budget and schedule"
                )
            seen[ident] = at
            if c.label in labels:
                errors.append(f"{at}: name {c.label!r} is also {labels[c.label]}")
            labels[c.label] = at
    return errors


def trained_fractions(contexts: Sequence[BenchContext], n_cu: int) -> List[float]:
    """Usable-CU fractions the trained (non-holdout) contexts cover, the
    exclusive 1.0 included."""
    out = {1.0}
    for c in contexts:
        if not c.holdout and 0 < c.cu_budget < n_cu:
            out.add(c.cu_budget / float(n_cu))
    return sorted(out)


# ── shape selection ──────────────────────────────────────────────────────────


def select_shapes(
    shapes: Sequence[ShapeKey], ctx: BenchContext, seed: int
) -> List[ShapeKey]:
    """The shapes `ctx` benches: all of them at fraction 1, else
    ceil(fraction * n) per cell (at least one), drawn with `seed`; in the
    order of `shapes`."""
    from .planner import shape_cell, stable_rng

    distinct = list(dict.fromkeys(shapes))
    if ctx.fraction >= 1.0:
        return distinct
    by_cell: Dict[str, List[ShapeKey]] = {}
    for s in distinct:
        by_cell.setdefault(shape_cell(s), []).append(s)
    chosen = set()
    for cell, members in sorted(by_cell.items()):
        k = min(len(members), max(1, math.ceil(ctx.fraction * len(members))))
        rng = stable_rng(seed, "context", ctx.label, cell)
        chosen.update(rng.sample(sorted(members), k))
    return [s for s in distinct if s in chosen]


# ── bench wrapper ────────────────────────────────────────────────────────────


def bench_spec(
    ctx: BenchContext,
    n_cu: int,
    bench: str,
    cotenant: Optional[str],
    log_dir: str,
    library_path: Optional[str] = None,
    path_append: Sequence[str] = (),
) -> Dict[str, Any]:
    """What the wrapper of `ctx` runs (`context.json`)."""
    cus = ctx.cotenant_cus(n_cu)
    if cus and not cotenant:
        raise ValueError(f"context {ctx.label} needs hipblaslt-cotenant")
    if cus >= n_cu:
        raise ValueError(f"context {ctx.label} would leave the bench no CU")
    return {
        "context": ctx.to_dict(),
        "n_cu": int(n_cu),
        "bench": str(bench),
        "cotenant": str(cotenant) if cus else None,
        "cotenant_cus": cus,
        "sm_count_target": int(ctx.cu_budget),
        "hybrid_assignment_policy": HYBRID_POLICY.get(ctx.schedule),
        "log_dir": str(log_dir),
        "library_path": library_path,
        "path_append": [str(p) for p in path_append],
    }


def _device_of(argv: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """(--device value, argv without it)."""
    out: List[str] = []
    device = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
            i += 2
            continue
        if a.startswith("--device="):
            device = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return device, out


def cotenant_device(device: str, environ: Mapping[str, str]) -> str:
    """hipblaslt-cotenant's `--device` for the bench's `--device device` in
    a process with `environ`. The cotenant replaces HIP_VISIBLE_DEVICES by
    its `--device`, so an index into a HIP_VISIBLE_DEVICES (or
    CUDA_VISIBLE_DEVICES) list the process already has is resolved through
    that list; ValueError when it cannot be."""
    index = int(device)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = str(environ.get(var, "")).strip()
        if not raw:
            continue
        entries = [e.strip() for e in raw.split(",")]
        if not 0 <= index < len(entries) or not entries[index].isdigit():
            raise ValueError(
                f"--device {device} does not name a device of {var}={raw}; "
                f"the cotenant would run elsewhere"
            )
        return entries[index]
    return str(index)


def bench_command(
    spec: Mapping[str, Any],
    argv: Sequence[str],
    stamp: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """(command, environment changes) of one bench call `argv` (as stage02 /
    stage03 pass it: `--device D --yaml F`) under `spec`, from a process
    with `environ` (default: this one's)."""
    environ = os.environ if environ is None else environ
    device, rest = _device_of(list(argv))
    bench_args = list(rest)
    if int(spec.get("sm_count_target") or 0) > 0:
        bench_args += ["--sm_count_target", str(int(spec["sm_count_target"]))]
    if spec.get("hybrid_assignment_policy"):
        bench_args += [
            "--hybrid_assignment_policy",
            str(spec["hybrid_assignment_policy"]),
        ]
    env: Dict[str, str] = {}
    if spec.get("library_path"):
        env["HIPBLASLT_TENSILE_LIBPATH"] = str(spec["library_path"])
    if spec.get("path_append"):
        env["PATH"] = os.pathsep.join(
            [environ.get("PATH", "")] + list(spec["path_append"])
        ).lstrip(os.pathsep)
    cus = int(spec.get("cotenant_cus") or 0)
    if cus <= 0:
        dev = [] if device is None else ["--device", device]
        return [str(spec["bench"]), *dev, *bench_args], env
    log = Path(spec["log_dir"]) / f"cotenant_{cus}_{stamp}.log"
    cmd = [str(spec["cotenant"]), "--cus", str(cus)]
    if device is not None:
        cmd += ["--device", cotenant_device(device, environ)]
    cmd += ["--log", str(log), "--", str(spec["bench"]), "--device", "0", *bench_args]
    return cmd, env


def write_wrapper(
    directory: Path, spec: Mapping[str, Any], python: str, script: Path
) -> Path:
    """`<directory>/hipblaslt-bench`: runs `script` (lib/context_bench.py)
    with `<directory>/context.json`, for stage02 / stage03's --bench-binary."""
    directory.mkdir(parents=True, exist_ok=True)
    Path(spec["log_dir"]).mkdir(parents=True, exist_ok=True)
    spec_path = directory / SPEC_FILE
    spec_path.write_text(json.dumps(dict(spec), indent=2) + "\n")
    wrapper = directory / WRAPPER_NAME
    wrapper.write_text("#!/bin/sh\n" f'exec "{python}" "{script}" "{spec_path}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper
