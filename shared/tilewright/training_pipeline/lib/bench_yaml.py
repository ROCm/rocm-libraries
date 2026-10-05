# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt-bench yaml writer and line-oriented reader.

A bench yaml holds one flow mapping per line, `- {function: matmul, ...}`,
describing one GEMM with all its bench knobs inline. Reading is regex based
(no yaml load) so large shape files stay cheap to process.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .fs import atomic_write_text
from .shapes import Shape, footprint_bytes


def strides_for(
    transA: str, transB: str, m: int, n: int, k: int
) -> Tuple[int, int, int, int]:
    """Default leading dimensions for hipBLASLt's column-major matrices."""
    lda = m if transA == "N" else k
    ldb = k if transB == "N" else n
    return lda, ldb, m, m


DEFAULT_LINE_KNOBS = {
    "alpha": 1,
    "beta": 0,
    "initialization": "trig_float",
    "iters": 50,
    "cold_iters": 50,
    "use_gpu_timer": 1,
    "flush": "true",
    "rotating": 512,
    "print_kernel_info": 1,
    "requested_solution_num": -1,
}

# ── adaptive timing ──────────────────────────────────────────────────────────
#
# With `adaptive: true` hipblaslt-bench times each kernel until the relative
# standard error of its mean drops under `noise_threshold`, instead of running
# a fixed iteration count. The yaml keys are the `Arguments` struct field names
# (hipblaslt_arguments.hpp), not the `--adaptive_*` command-line spellings: an
# unknown yaml key is silently ignored by the bench.
ADAPTIVE_KEYS = (
    "warmup_time",
    "sample_time",
    "measure_time",
    "max_measure_time",
    "min_iters",
    "max_iters",
    "noise_threshold",
    "stability_threshold",
    "stability_window",
    "stability_interval",
)

# The bench rejects `adaptive` combined with iters / cold_iters.
_ITER_KEYS = ("iters", "cold_iters")


def adaptive_cfg(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The `bench.adaptive` block when `enabled` is true, else None.

        bench:
          adaptive:
            enabled: true
            measure_time: 2.0      # ms per kernel
            ...

    Keys in ADAPTIVE_KEYS are passed through verbatim; anything else is
    dropped."""
    blk = (config.get("bench") or {}).get("adaptive") or {}
    if not blk or not blk.get("enabled", False):
        return None
    return {k: blk[k] for k in ADAPTIVE_KEYS if k in blk}


def _default_scale_for_dtype(dtype: str) -> int:
    """Default scaleA / scaleB mode when the config sets none: 1 (Scalar, the
    `UseScaleAB: Scalar` F8 libraries) for f8 operands, 0 otherwise. Mode 2
    would be Vector, which selects a different kernel family."""
    return 1 if "f8_r" in dtype else 0


# ── rotating buffer sizing ───────────────────────────────────────────────────
#
# `rotating` is a rotating-buffer budget in MiB; the bench fills it with as
# many copies of the problem as fit and cycles through them so consecutive
# runs read cold data. A fixed budget turns into millions of copies of a
# small problem, so the budget is sized to the problem instead:
#     rotating_mib = ceil(footprint * bench.rotating_target_blocks / MiB)
# clamped to [1, bench.rotating_mb]. Never 0: without rotation, candidates
# benched back to back on the same buffers read them warm, and the kernel
# measured last looks fastest.
_MIB = 1024 * 1024


def _scale_modes(hl: Dict[str, Any]) -> Tuple[int, int]:
    a_t = str(hl.get("a_type", "bf16_r"))
    b_t = str(hl.get("b_type", a_t))
    scale_a = int(hl["scaleA"]) if "scaleA" in hl else _default_scale_for_dtype(a_t)
    scale_b = int(hl["scaleB"]) if "scaleB" in hl else _default_scale_for_dtype(b_t)
    return scale_a, scale_b


def _problem_footprint(shape: Shape, hl: Dict[str, Any]) -> int:
    scale_a, scale_b = _scale_modes(hl)
    return footprint_bytes(
        shape.m,
        shape.n,
        shape.k,
        shape.batch,
        a_type=hl.get("a_type"),
        b_type=hl.get("b_type"),
        c_type=hl.get("c_type"),
        d_type=hl.get("d_type"),
        scale_a=scale_a,
        scale_b=scale_b,
    )


def rotating_mb_for(shape: Shape, config: Dict[str, Any]) -> int:
    """Rotating-buffer MiB for one shape (see the sizing note above)."""
    bench = config.get("bench") or {}
    cap = int(bench.get("rotating_mb", 512))
    target = int(bench.get("rotating_target_blocks", 16))
    if cap <= 0 or target <= 0:
        return 0
    want = _problem_footprint(shape, config["hipblaslt"]) * target
    want_mib = -(-want // _MIB)
    return int(min(cap, max(1, want_mib)))


def render_bench_line(
    shape: Shape, config: Dict[str, Any], knobs: Optional[Dict[str, Any]] = None
) -> str:
    """One hipblaslt-bench yaml line for `shape`, with dtype / layout from
    `config.hipblaslt` and `knobs` merged over DEFAULT_LINE_KNOBS.

    Precedence: `bench.skip_slow_solution_ratio` from the config overrides
    `knobs`; `rotating` is sized per shape unless `knobs` sets it; with
    `bench.adaptive` enabled the iteration knobs are dropped and the adaptive
    keys added. Strides are emitted as 0 so the bench derives them. scaleA /
    scaleB come from `config.hipblaslt` or default by dtype (1 for f8, else
    0)."""
    hl = config["hipblaslt"]
    transA, transB = hl["transA"], hl["transB"]
    m, n, k, b = shape.m, shape.n, shape.k, shape.batch
    lda, ldb, ldc, ldd = strides_for(transA, transB, m, n, k)
    scale_a, scale_b = _scale_modes(hl)
    k_dict = dict(DEFAULT_LINE_KNOBS)
    if knobs:
        k_dict.update(knobs)
    # The screen drops a candidate when warm_up * ratio > best warm-up so far;
    # 0 disables it and every candidate gets fully timed.
    ssr = (config.get("bench") or {}).get("skip_slow_solution_ratio")
    if ssr is not None:
        k_dict["skip_slow_solution_ratio"] = ssr
    if not (knobs and "rotating" in knobs):
        k_dict["rotating"] = rotating_mb_for(shape, config)
    ad = adaptive_cfg(config)
    if ad is not None:
        for key in _ITER_KEYS:
            k_dict.pop(key, None)
        k_dict["adaptive"] = "true"
        k_dict.update(ad)
    kv = ", ".join(f"{key}: {v}" for key, v in k_dict.items())
    return (
        f"- {{function: matmul, transA: {transA}, transB: {transB}, "
        f"a_type: {hl['a_type']}, b_type: {hl['b_type']}, "
        f"c_type: {hl['c_type']}, d_type: {hl['d_type']}, "
        f"scale_type: {hl['scale_type']}, compute_type: {hl['compute_type']}, "
        f"scaleA: {scale_a}, scaleB: {scale_b}, "
        f"M: {m}, N: {n}, K: {k}, "
        f"ldc: {ldc}, ldd: {ldd}, lda: {lda}, ldb: {ldb}, "
        f"stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, "
        f"batch_count: {b}, {kv}}}"
    )


def write_bench_yaml(
    shapes: Iterable[Shape],
    config: Dict[str, Any],
    out_path: Path,
    knobs: Optional[Dict[str, Any]] = None,
) -> int:
    """Write one bench line per shape (atomically); returns the line count."""
    lines = [render_bench_line(s, config, knobs) for s in shapes]
    atomic_write_text(Path(out_path), "".join(line + "\n" for line in lines))
    return len(lines)


# ── reading ──────────────────────────────────────────────────────────────────

_KEY_RE = {
    key: re.compile(rf"(?<![\w]){key}:\s*(\d+)")
    for key in ("M", "N", "K", "batch_count")
}


def read_bench_lines(path: Path) -> List[str]:
    """Every `- {...}` entry line of a bench yaml, newline-terminated, in
    file order (which is the bench's execution order)."""
    out: List[str] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip().startswith("-"):
                out.append(line if line.endswith("\n") else line + "\n")
    return out


def shape_of_line(line: str) -> Optional[Tuple[int, int, int, int]]:
    """(M, N, K, batch_count) of one bench line; batch_count defaults to 1 as
    in the bench. None when M, N or K is missing."""
    vals = {}
    for key, pat in _KEY_RE.items():
        m = pat.search(line)
        if m:
            vals[key] = int(m.group(1))
    if not all(k in vals for k in ("M", "N", "K")):
        return None
    return (vals["M"], vals["N"], vals["K"], vals.get("batch_count", 1))


def parse_shapes_yaml(path: Path) -> List[Tuple[int, int, int, int]]:
    """(M, N, K, batch) of every entry of a bench yaml, in file order."""
    out: List[Tuple[int, int, int, int]] = []
    for line in read_bench_lines(path):
        shape = shape_of_line(line)
        if shape is not None:
            out.append(shape)
    return out
