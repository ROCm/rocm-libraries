# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parallel build-on-the-fly sweep driver for the CK DSL.

The dispatcher ships with a heuristic that ranks N kernel variants per
problem shape; we want to (a) be able to *build* every one of those
variants on the fly from the DSL, (b) *run* each one and record the
actual TFLOPS, then (c) beat the heuristic's choice. This module is
the harness that drives (a) and the manifest plumbing for (b).

Steps:

  1. `build_all_instances(specs, ...)` builds every supplied spec
     to LLVM IR and HSACO in parallel (one process per spec, falling
     back to one thread if multiprocessing is unavailable). Results
     are cached at `<cache_dir>/<spec_hash>.hsaco` so a re-sweep is
     essentially free.

  2. `write_run_plan(...)` emits a JSON run-plan that a downstream
     launcher (`example/ck_tile/dsl/common/launcher.cpp` or a future
     sweep launcher) consumes to benchmark each kernel and produce a
     CSV of `(name, M, N, K, ms_per_iter, tflops, gbps, ok)`.

  3. `pick_best(runs, ...)` reports the best correct kernel per shape
     and (when the dispatcher's prediction is available) the gap.

Acceptance gates (per the runbook):
  - Correctness validated at 1e-3 absolute (not 1e-2) on a small shape
    BEFORE perf is reported. Bit-exact is the goal.
  - Throw away the first run of a fresh process; report median over
    >=3 fresh-process attempts.
  - Sweep one lever family at a time; record VGPR/LDS for each variant
    by reading the HSACO ELF metadata.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .core.lower_llvm import lower_kernel_to_llvm
from .runtime.comgr import build_hsaco_from_llvm_ir
from .instances import UniversalGemmSpec, build_universal_gemm


# ---------------------------------------------------------------------
# Build records
# ---------------------------------------------------------------------


@dataclass
class BuildRecord:
    """One spec's build outcome."""

    name: str
    spec_dict: Dict[str, object]
    ok: bool
    error: str = ""
    hsaco_path: str = ""
    hsaco_bytes: int = 0
    block_m: int = 0
    block_n: int = 0
    block_k: int = 0
    threads_per_block: int = 0
    ir_build_ms: float = 0.0
    ir_lower_ms: float = 0.0
    comgr_ms: float = 0.0
    elf_meta: Dict[str, int] = field(default_factory=dict)

    @property
    def total_build_ms(self) -> float:
        return self.ir_build_ms + self.ir_lower_ms + self.comgr_ms


def _spec_to_dict(spec: UniversalGemmSpec) -> Dict[str, object]:
    return {
        "name": spec.name,
        "tile": asdict(spec.tile),
        "trait": asdict(spec.trait),
        "data": asdict(spec.data),
        "block_size": spec.block_size,
        "wave_size": spec.wave_size,
    }


def _spec_hash(spec: UniversalGemmSpec) -> str:
    d = _spec_to_dict(spec)
    blob = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------------------------------------------------------------
# ELF metadata extraction (VGPR / SGPR / LDS)
# ---------------------------------------------------------------------


def _extract_elf_meta(hsaco_path: Path) -> Dict[str, int]:
    """Pull VGPR/SGPR/LDS bytes from the HSA kernel descriptor.

    HSA-style code objects encode this in `.note` and `.rodata`; the
    most portable way without depending on `pyelftools` is to call
    `llvm-objdump` (installed alongside the ROCm toolchain) and grep
    the result. We keep this best-effort: if the tool isn't present
    the metadata stays empty (sweeping still proceeds).
    """
    import shutil
    import subprocess

    od = shutil.which("llvm-objdump") or shutil.which("/opt/rocm/llvm/bin/llvm-objdump")
    if not od:
        return {}
    try:
        r = subprocess.run(
            [
                od,
                "--disassemble-symbols=",
                "--mcpu=gfx950",
                "--triple=amdgcn-amd-amdhsa",
                "-S",
                str(hsaco_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = r.stdout + r.stderr
    except Exception:
        return {}
    meta: Dict[str, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(".amdhsa_next_free_vgpr"):
            try:
                meta["vgprs"] = int(line.split()[-1])
            except ValueError:
                pass
        elif line.startswith(".amdhsa_next_free_sgpr"):
            try:
                meta["sgprs"] = int(line.split()[-1])
            except ValueError:
                pass
        elif line.startswith(".amdhsa_group_segment_fixed_size"):
            try:
                meta["lds_bytes"] = int(line.split()[-1])
            except ValueError:
                pass
    return meta


# ---------------------------------------------------------------------
# Builder worker
# ---------------------------------------------------------------------


def _build_one(args: Tuple[str, Dict[str, object], str, str]) -> Dict[str, object]:
    """Build worker. Runs in the calling process or in a pool worker.

    args = (spec_hash, spec_dict, cache_dir, isa)
    Returns the BuildRecord as a dict (so it survives a fork/spawn).
    """
    spec_hash, spec_dict, cache_dir_str, isa = args
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec_from_dict(spec_dict)
    name = spec.kernel_name()
    out_path = cache_dir / f"{spec_hash}_{name[:120]}.hsaco"

    rec = BuildRecord(
        name=name,
        spec_dict=spec_dict,
        ok=False,
        block_m=spec.tile.tile_m,
        block_n=spec.tile.tile_n,
        block_k=spec.tile.tile_k,
        threads_per_block=spec.block_size,
    )

    # Fast cache hit.
    if out_path.exists() and out_path.stat().st_size > 0:
        rec.ok = True
        rec.hsaco_path = str(out_path)
        rec.hsaco_bytes = out_path.stat().st_size
        rec.elf_meta = _extract_elf_meta(out_path)
        return asdict(rec)

    try:
        t0 = time.perf_counter()
        kernel = build_universal_gemm(spec)
        t1 = time.perf_counter()
        ll = lower_kernel_to_llvm(kernel)
        t2 = time.perf_counter()
        hsaco, ct = build_hsaco_from_llvm_ir(ll, isa=isa)

        out_path.write_bytes(hsaco)
        rec.ok = True
        rec.hsaco_path = str(out_path)
        rec.hsaco_bytes = len(hsaco)
        rec.ir_build_ms = (t1 - t0) * 1000.0
        rec.ir_lower_ms = (t2 - t1) * 1000.0
        rec.comgr_ms = ct.total * 1000.0
        rec.elf_meta = _extract_elf_meta(out_path)
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"

    return asdict(rec)


def _spec_from_dict(d: Dict[str, object]) -> UniversalGemmSpec:
    """Inverse of `_spec_to_dict` — used by worker processes."""
    from .instances import DataSpec, TileSpec, TraitSpec

    return UniversalGemmSpec(
        name=d["name"],  # type: ignore[arg-type]
        tile=TileSpec(**d["tile"]),  # type: ignore[arg-type]
        trait=TraitSpec(**d["trait"]),  # type: ignore[arg-type]
        data=DataSpec(**d["data"]),  # type: ignore[arg-type]
        wave_size=int(d["wave_size"]),  # type: ignore[arg-type]
        block_size=int(d["block_size"]),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def build_all_instances(
    specs: Iterable[UniversalGemmSpec],
    *,
    cache_dir: Path,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    parallel: Optional[int] = None,
) -> List[BuildRecord]:
    """Build every spec to a cached HSACO blob.

    `parallel`:
      - None or > 1: use a multiprocessing Pool with that many workers
        (`os.cpu_count()` if None).
      - 1: build serially (useful for debugging crashes).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    specs = list(specs)
    if not specs:
        return []

    work = [(_spec_hash(s), _spec_to_dict(s), str(cache_dir), isa) for s in specs]

    if parallel == 1:
        out_dicts = [_build_one(w) for w in work]
    else:
        n = parallel or os.cpu_count() or 4
        ctx = mp.get_context("fork")  # avoid re-importing ck_dsl in spawn
        with ctx.Pool(n) as pool:
            out_dicts = pool.map(_build_one, work)

    return [BuildRecord(**d) for d in out_dicts]


def write_sweep_manifest(
    records: Sequence[BuildRecord],
    out_path: Path,
    *,
    shapes: Sequence[Tuple[int, int, int]],
    warmup: int = 5,
    iters: int = 100,
) -> None:
    """Emit a JSON manifest the runner uses to launch each HSACO.

    Layout matches what `example/ck_tile/dsl/common/launcher.cpp` reads
    per individual kernel; the sweep launcher (`tools/dsl_sweep_run.cpp`)
    reads the `instances` list and calls into the same launcher logic
    once per instance.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": "ck.dsl.sweep.manifest/v1",
        "warmup_iters": warmup,
        "timed_iters": iters,
        "shapes": [{"M": int(M), "N": int(N), "K": int(K)} for M, N, K in shapes],
        "instances": [
            {
                "name": r.name,
                "hsaco": r.hsaco_path,
                "block_m": r.block_m,
                "block_n": r.block_n,
                "block_k": r.block_k,
                "threads_per_block": r.threads_per_block,
                "build_ms": r.total_build_ms,
                "build_ok": r.ok,
                "build_error": r.error,
                "elf_meta": r.elf_meta,
                "spec": r.spec_dict,
            }
            for r in records
        ],
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))


# ---------------------------------------------------------------------
# Convenience: build the default dispatcher set in one call
# ---------------------------------------------------------------------


def build_default_dispatcher_set(
    *,
    cache_dir: Path,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    parallel: Optional[int] = None,
    pipelines: Sequence[str] = ("compv3", "compv4"),
    epilogues: Sequence[str] = ("default", "cshuffle"),
) -> List[BuildRecord]:
    """Sweep the dispatcher's `default_config.json` cartesian product.

    Returns one BuildRecord per spec, with `ok=True` for the ones that
    compiled (some configs may legitimately fail LDS budget; those are
    skipped by `is_valid_spec` upstream).
    """
    from .instances import all_dispatcher_configs

    specs = list(
        all_dispatcher_configs(
            pipeline=pipelines,  # type: ignore[arg-type]
            epilogue=epilogues,  # type: ignore[arg-type]
        )
    )
    return build_all_instances(specs, cache_dir=cache_dir, isa=isa, parallel=parallel)
