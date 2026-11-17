#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Live timing comparison: rocke's layernorm2d/batchnorm2d vs MIOpenDriver find-mode.

This is a **timing-only** comparison script -- it does not claim anything
about numeric correctness (that is the separate, already-existing on-GPU
numeric parity test in ``platform/tests/instances/differential/numeric.py``).

For each representative case it:
  1. Runs the matching rocke AOT benchmark script
     (``benchmark_layernorm2d_aot.py`` / ``benchmark_batchnorm2d_aot.py``)
     with ``--timing staged`` (the protocol ``batched_timing``'s docstring
     calls "what a number compared against hipDNN or MIOpen has to be") and
     takes its best (lowest-time) swept candidate.
  2. Shells out to ``MIOpenDriver {layernorm,bnorm}fp16`` with
     ``MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4`` set (find-mode, forced
     database re-search -- *not* ``-S 0``, which would pick a fixed solver
     and skip find() entirely) and parses MIOpenDriver's own reported GPU
     kernel time.
  3. Prints a comparison table: case, rocke time, MIOpen find-mode time,
     speedup.

Structural analogy only to ``benchmark_implicit_gemm_conv.py``'s
``_run_ckprofiler`` (subprocess + stdout-parsing + timeout pattern) -- that
function shells out to ckProfiler, a different binary with a different
output format; rocke has no pre-existing live-MIOpenDriver-invocation glue to
reuse, this is new.

Usage::

    PYTHONPATH=$ROCKE/python:<repo>/library:<repo>/library/benchmarks \\
        python library/benchmarks/common/compare_norm_vs_miopen.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_MIOPEN_DRIVER = "/opt/rocm-7.1.1/bin/MIOpenDriver"
_MIOPEN_ENV_EXTRA = {"MIOPEN_FIND_MODE": "1", "MIOPEN_FIND_ENFORCE": "4"}

_ROCKE_BEST_RE = re.compile(r"^Best:\s+\S+\s+\(([0-9.]+)\s*ms\)", re.MULTILINE)
_MIOPEN_LN_RE = re.compile(
    r"GPU Kernel Time Forward LayerNorm Elapsed:\s*([0-9.]+)\s*ms"
)
_MIOPEN_BN_RE = re.compile(
    r"GPU Kernel Min Time Forward Batch Normalization Elapsed:\s*([0-9.]+)\s*ms"
)


@dataclass(frozen=True)
class CaseResult:
    label: str
    rocke_ms: Optional[float]
    miopen_ms: Optional[float]
    note: str = ""

    @property
    def speedup(self) -> Optional[float]:
        if self.rocke_ms and self.miopen_ms:
            return self.miopen_ms / self.rocke_ms
        return None


def _run(cmd: list[str], *, env_extra: dict[str, str] | None = None, timeout: int = 120) -> str:
    import os

    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout
    )
    return proc.stdout + "\n" + proc.stderr


def _rocke_best_ms(stdout: str) -> Optional[float]:
    m = _ROCKE_BEST_RE.search(stdout)
    return float(m.group(1)) if m else None


def run_layernorm_case(rows: int, cols: int, *, dtype: str, arch: str) -> CaseResult:
    label = f"layernorm rows={rows} cols={cols} {dtype}"
    rocke_cmd = [
        sys.executable, str(_HERE / "benchmark_layernorm2d_aot.py"),
        "--rows", str(rows), "--cols", str(cols), "--kind", "layernorm",
        "--dtype", dtype, "--arch", arch, "--timing", "staged",
        "--warmup", "5", "--iters", "15",
    ]
    out = _run(rocke_cmd)
    rocke_ms = _rocke_best_ms(out)
    note = "" if rocke_ms is not None else f"rocke: no Best line ({out[-400:]!r})"

    miopen_cmd = [
        _MIOPEN_DRIVER, "layernormfp16", "-F", "1",
        "-X", f"{rows}x{cols}", "-o", "1", "-t", "1", "-w", "1", "-V", "0", "-i", "20",
    ]
    mout = _run(miopen_cmd, env_extra=_MIOPEN_ENV_EXTRA)
    m = _MIOPEN_LN_RE.search(mout)
    miopen_ms = float(m.group(1)) if m else None
    if miopen_ms is None:
        note = (note + "; " if note else "") + f"miopen: no match ({mout[-400:]!r})"

    return CaseResult(label, rocke_ms, miopen_ms, note)


def run_batchnorm_case(
    n: int, c: int, h: int, w: int, *, training: bool, dtype: str, arch: str
) -> CaseResult:
    mode = "training" if training else "inference"
    label = f"batchnorm n={n} c={c} h={h} w={w} {mode} {dtype}"
    rocke_cmd = [
        sys.executable, str(_HERE / "benchmark_batchnorm2d_aot.py"),
        "--n", str(n), "--c", str(c), "--h", str(h), "--w", str(w),
        "--training" if training else "--inference",
        "--dtype", dtype, "--arch", arch, "--timing", "staged",
        "--warmup", "5", "--iters", "15",
    ]
    out = _run(rocke_cmd, timeout=180)
    rocke_ms = _rocke_best_ms(out)
    note = "" if rocke_ms is not None else f"rocke: no Best line ({out[-400:]!r})"

    forw = "1" if training else "2"
    save = "1" if training else "0"
    miopen_cmd = [
        _MIOPEN_DRIVER, "bnormfp16", "-F", forw,
        "-n", str(n), "-c", str(c), "-H", str(h), "-W", str(w),
        "-m", "1", "-r", "1", "-s", save,
        "-t", "1", "-w", "1", "-V", "0", "-i", "20",
    ]
    mout = _run(miopen_cmd, env_extra=_MIOPEN_ENV_EXTRA)
    m = _MIOPEN_BN_RE.search(mout)
    miopen_ms = float(m.group(1)) if m else None
    if miopen_ms is None:
        note = (note + "; " if note else "") + f"miopen: no match ({mout[-400:]!r})"

    return CaseResult(label, rocke_ms, miopen_ms, note)


LAYERNORM_CASES = [
    (2048, 1024),
    (4096, 4096),
    (8192, 8192),
]

# n,c,h,w chosen so n*h*w divides rocke's block_size*vec grid (see
# norm_sweep_space.batchnorm_combos); not exact ResNet dims, but the same
# order of magnitude as a mid/late-stage conv feature map.
#
# N capped at 32: batchnorm2d's training-mode build Python-unrolls a loop
# over range(N) (one sweep_row_chunks call per batch element, see
# instances/common/batchnorm2d.py) and composes it with the two-pass
# streaming path for small-vec/large-elems_per_thread candidates. At N=256
# this combination made the small-vec candidates (observed:
# b128_v2, elems_per_thread=2048) take minutes to compile -- real CPU-bound
# time inside compile_kernel, not a hang, confirmed by isolating that one
# candidate. At N=32 the full 9-candidate sweep compiles in well under a
# second per candidate. This is a real scaling caveat of the current
# batchnorm2d implementation worth fixing (e.g. a device-side loop over N
# instead of a Python-unrolled one) -- flagged here, not silently worked
# around.
BATCHNORM_CASES = [
    (32, 64, 32, 32),
    (16, 128, 32, 32),
    (16, 256, 16, 16),
]


def main() -> int:
    dtype, arch = "fp16", "gfx950"
    results: list[CaseResult] = []

    for rows, cols in LAYERNORM_CASES:
        print(f"running layernorm rows={rows} cols={cols} ...", file=sys.stderr)
        results.append(run_layernorm_case(rows, cols, dtype=dtype, arch=arch))

    for n, c, h, w in BATCHNORM_CASES:
        for training in (True, False):
            print(f"running batchnorm n={n} c={c} h={h} w={w} training={training} ...",
                  file=sys.stderr)
            results.append(
                run_batchnorm_case(n, c, h, w, training=training, dtype=dtype, arch=arch)
            )

    print(f"\n{'case':<45} {'rocke (staged)':>16} {'miopen (find)':>16} {'speedup':>10}")
    print("-" * 90)
    for r in results:
        rocke_s = f"{r.rocke_ms:.4f} ms" if r.rocke_ms is not None else "FAIL"
        miopen_s = f"{r.miopen_ms:.4f} ms" if r.miopen_ms is not None else "FAIL"
        speedup_s = f"{r.speedup:.2f}x" if r.speedup is not None else "-"
        print(f"{r.label:<45} {rocke_s:>16} {miopen_s:>16} {speedup_s:>10}")
        if r.note:
            print(f"    note: {r.note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
