# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The AOT cache for the direct convolution kernels.

Direct conv splits its parameters differently from implicit GEMM. Batch,
spatial extents and the group count are kernargs, so one binary serves any
image; the filter geometry (``KH``/``KW``), ``stride``, ``PAD`` and the
per-group channel counts (``cpg``/``kpg``) shape the unrolled MFMA chain and
the LDS row layout, so they are *capabilities* baked into the binary. A cache
therefore holds one binary per (capability, tuning knob) pair, and
``--run-from-cache`` offers a binary only to problems with exactly its
capabilities.

Which capabilities to build is a product decision. :data:`DIRECT_CAPABILITIES`
follows the set the reference direct-conv library covers on CDNA and drops
what rocke's kernels cannot compute (see the table's comments).

The run side is split in two. :func:`direct_plans` is pure CPU: it picks the
cached kernels that fit a problem, re-validates each spec against the real
shape, and returns a launch plan -- binaries, signatures, grids and the
buffer each kernarg slot binds to. The benchmark only allocates memory and
launches the plan.
"""

from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from benchmarks.common.kernel_sweep import BuildJob, compile_jobs

# Operand dtypes the direct kernels build for. The cache is filled for one at a
# time (--compile-all --dtype); each binary records its dtype and is only
# offered to problems of that dtype. Variants without a kernel for a dtype
# (the 4c atom is fp16-only) are dropped by their spec validators.
DIRECT_DTYPES = ("fp16", "bf16")
DIRECT_DTYPE = "fp16"  # the default dtype

# ---------------------------------------------------------------------------
# Tuning grids -- shared with benchmark_direct_conv's JIT sweep so the cache
# and the sweep explore the same configurations.
# ---------------------------------------------------------------------------

# Grouped forward (DirectConvSpec) and the MFMA dgrad's fprop pass.
BLOCK_Q = (16, 32)
BLOCK_GROUPS = (1, 2, 4, 8, 16)
DOUBLE_BUFFER = (True, False)
# Depthwise (cpg == 1) forward and dgrad.
# block_w=32 is omitted from both directions: across the depthwise corpus, at
# both dtypes, it never won a single geometry -- every shape whose best config
# used a wide block_w landed on 8 or 16 -- while still costing a quarter of the
# candidate builds and tuning launches.
#
# The surviving values differ by direction, because the two directions do not
# have the same fallback.  Forward is overwhelmingly a block_w=4 story: the
# column-streamed variant (its own grid below) covers the wide-block cases, so
# the preloading kernel rarely needs to go wide, and the few shapes that do go
# wide land on 16 rather than 8.  Dropping 8 from the forward grid is therefore
# free, while dropping 16 is not.  Dgrad has no second variant to fall back on,
# so its tail keeps both 8 and 16 -- dropping either regresses small-spatial /
# large-filter shapes, 8 the more severely of the two.  Neither tuple is
# reducible further without giving up a shape's best config.
DW_BLOCK_W_FWD = (4, 16)
DW_BLOCK_W_DGRAD = (4, 8, 16)
DW_BLOCK_WAVES = (1, 2, 4)
# Column-streamed depthwise forward. It keeps only ``block_h*block_w + KH`` f32
# live per lane instead of ``KH*KW + KH*block_w``, so its sweet spot sits at far
# smaller block_w than the weight-preloading kernel's. block_h is the output-row
# tile: the row loop is unrolled at build time, so the tile height is a
# capability while the image height stays a kernarg. Pairs whose band does not
# fit the arch's live-f32 budget are dropped by the spec validator.
#
# Measured on gfx950 over 15 depthwise shapes (3x3..11x11, stride 1 and 2,
# 7x7..112x112 images, large enough batches to clear launch latency): block_w
# 8/16/32 never won a shape, and this 3x3x3 grid matches the best of the full
# 3x6x3 grid on every one. All three block_h values are needed -- dropping 32
# costs up to 16% on a shape, dropping 8 up to 74%.
DW_COL_BLOCK_W = (1, 2, 4)
DW_COL_BLOCK_H = (8, 16, 32)
# Grouped scalar-FMA dgrad.
DGRAD_BLOCK_Q = (4, 8, 16, 32)
# MFMA dgrad fprop pass: output rows per block, then
# (waves_q, waves_k, runtime_k_loop, persistent_grid, fold_k32).
DGRAD_BLOCK_H = (8, 16)
DGRAD_WAVES = (
    (1, 1, False, False, False),
    (1, 4, False, False, False),
    # runtime_k_loop: one K-atom at a time -> low VGPR use, more blocks/CU.
    (1, 4, True, False, False),
    (1, 2, False, False, True),
    (1, 6, False, False, True),
)

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectCaps:
    """What one binary is baked for; everything else about a shape is runtime."""

    KH: int
    PAD: int
    stride: int
    cpg: int
    kpg: int

    @property
    def KW(self) -> int:
        # Every capability below is a square filter.
        return self.KH


def _grouped_channels() -> Tuple[int, ...]:
    # The reference library covers cpg == kpg in {4, 8, 16, 32}.
    return (4, 8, 16, 32)


_DEPTHWISE_FILTERS = (3, 5, 7, 9, 11)
_STRIDES = (1, 2)


def _same_pad(kh: int) -> int:
    return (kh - 1) // 2


def _all_pads(kh: int) -> range:
    # The reference library takes the padding as a runtime value in [0, KH-1].
    return range(kh)


def _col_pads(kh: int) -> Tuple[int, ...]:
    # The column-streamed kernel can compute any PAD in [0, KH-1] (bar stride-1
    # over-padding, which its validator rejects), but every padding is its own
    # binary. The cache carries the two networks actually use -- "valid"
    # (PAD=0) and "same" -- rather than the whole range: the intermediate ones
    # would be two thirds of the variant's entries for shapes that are rare in
    # practice. Other paddings still build on demand (the JIT sweep).
    return tuple(sorted({0, _same_pad(kh)}))


# variant -> (direction, capability list). Forward rows only list "same"
# padding: every forward direct kernel streams input rows and flushes output
# rows by input-row index, which is only right for PAD == (KH-1)/2 (see
# kernels.common.conv_direct_grouped.forward_padding_reason). Dgrad kernels
# bound their rows by p_Ho and are padding-generic, so they get the full
# [0, KH-1] range like the reference.
DIRECT_CAPABILITIES: Dict[str, Tuple[str, Tuple[DirectCaps, ...]]] = {
    # Grouped forward, 3x3. The reference also serves stride 2; rocke's
    # grouped forward kernels are stride-1 only, so that row is absent.
    "direct_grouped": (
        "fwd",
        tuple(
            DirectCaps(KH=3, PAD=1, stride=1, cpg=c, kpg=c) for c in _grouped_channels()
        ),
    ),
    # Depthwise forward: square odd filters 3..11, stride 1 and 2.
    "direct_depthwise": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
        ),
    ),
    # Column-streamed depthwise forward: same filters and strides. It bounds
    # its output rows against p_Ho rather than streaming them by input row, so
    # unlike the other forward kernels it also serves "valid" padding.
    "direct_depthwise_col": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
            for pad in _col_pads(k)
        ),
    ),
    # Depthwise dgrad (row-streaming kernel): same filters and strides.
    "direct_depthwise_dgrad": (
        "dgrad",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
            for pad in _all_pads(k)
        ),
    ),
    # Grouped dgrad, scalar FMA: any stride and padding the reference covers.
    "direct_grouped_dgrad": (
        "dgrad",
        tuple(
            DirectCaps(KH=3, PAD=pad, stride=s, cpg=c, kpg=c)
            for c, s, pad in itertools.product(
                _grouped_channels(), _STRIDES, _all_pads(3)
            )
        ),
    ),
    # Grouped dgrad, MFMA pipeline (weight transpose -> reorganize -> fprop).
    # Its last pass is a forward kernel on the transposed problem, so it
    # inherits the forward limits: stride 1 and "same" padding.
    "direct_grouped_dgrad_mfma": (
        "dgrad",
        tuple(
            DirectCaps(KH=3, PAD=1, stride=1, cpg=c, kpg=c) for c in _grouped_channels()
        ),
    ),
}

# Weight-transform kernels of the MFMA dgrad pipeline, cached as their own
# entries and found by exact identity.
_TRANSPOSE = "direct_dgrad_transpose"
_REORGANIZE = "direct_dgrad_reorganize"


def _knob_grid(variant: str) -> Iterator[dict]:
    if variant == "direct_grouped":
        for bq, bg, db in itertools.product(BLOCK_Q, BLOCK_GROUPS, DOUBLE_BUFFER):
            yield dict(block_q=bq, block_groups=bg, double_buffer=db)
    elif variant == "direct_depthwise":
        for bw, waves in itertools.product(DW_BLOCK_W_FWD, DW_BLOCK_WAVES):
            yield dict(block_w=bw, block_waves=waves)
    elif variant == "direct_depthwise_dgrad":
        for bw, waves in itertools.product(DW_BLOCK_W_DGRAD, DW_BLOCK_WAVES):
            yield dict(block_w=bw, block_waves=waves)
    elif variant == "direct_depthwise_col":
        for bh, bw, waves in itertools.product(
            DW_COL_BLOCK_H, DW_COL_BLOCK_W, DW_BLOCK_WAVES
        ):
            yield dict(block_h=bh, block_w=bw, block_waves=waves)
    elif variant == "direct_grouped_dgrad":
        for bq, bg in itertools.product(DGRAD_BLOCK_Q, BLOCK_GROUPS):
            yield dict(block_q=bq, block_groups=bg)
    elif variant == "direct_grouped_dgrad_mfma":
        for bq, bg, bh, (wq, wk, rk, pg, k32) in itertools.product(
            BLOCK_Q, BLOCK_GROUPS, DGRAD_BLOCK_H, DGRAD_WAVES
        ):
            yield dict(
                block_q=bq,
                block_groups=bg,
                block_h=bh,
                waves_q=wq,
                waves_k=wk,
                runtime_k_loop=rk,
                persistent_grid=pg,
                fold_k32=k32,
            )
    else:
        raise ValueError(f"unknown direct conv variant {variant!r}")


# ---------------------------------------------------------------------------
# Spec construction -- the single place a (variant, problem, knobs) triple
# turns into a kernel spec, used by both the build and the run side.
# ---------------------------------------------------------------------------


def make_spec(variant: str, problem, knobs: dict):
    """The spec ``variant`` builds for ``problem``.

    For the MFMA dgrad pipeline this is the spec of its compute pass (a
    forward kernel on the transposed problem); the weight transforms are
    derived from the original problem separately.
    """
    from kernels.common import conv_direct_grouped as dc

    if variant == "direct_grouped":
        return dc.DirectConvSpec(problem=problem, name="rocke_direct_conv", **knobs)
    if variant == "direct_depthwise":
        return dc.DirectDepthwiseSpec(
            problem=problem, name="rocke_direct_depthwise", **knobs
        )
    if variant == "direct_depthwise_col":
        # The col spec carries its own element type; a binary serves only the
        # dtype it was built for, so it follows the problem's.
        return dc.DirectDepthwiseColSpec(
            problem=problem,
            name="rocke_direct_depthwise_col",
            dtype=problem.dtype,
            **knobs,
        )
    if variant == "direct_depthwise_dgrad":
        return dc.DirectDepthwiseDgradStreamSpec(
            problem=problem, name="rocke_direct_dw_dgrad", **knobs
        )
    if variant == "direct_grouped_dgrad":
        return dc.DirectConvDgradSpec(
            problem=problem, name="rocke_direct_dgrad", **knobs
        )
    if variant == "direct_grouped_dgrad_mfma":
        fold_k32 = knobs["fold_k32"]
        spec = dc.make_dgrad_fprop_spec(
            problem, **{k: v for k, v in knobs.items() if k != "fold_k32"}
        )
        return replace(spec, fold_k32=fold_k32) if fold_k32 else spec
    if variant == _TRANSPOSE:
        return dc.DirectTransposeWeightsDgradSpec(problem=problem)
    if variant == _REORGANIZE:
        return dc.DirectReorganizeWeightsSpec(problem=problem, **knobs)
    raise ValueError(f"unknown direct conv variant {variant!r}")


def _mfma_knobs_reason(problem, knobs: dict) -> Optional[str]:
    """Constraints of the MFMA dgrad pipeline that its spec cannot see.

    The fprop spec validates K-atoms as ``ceil(cpg/16)``; with ``fold_k32``
    the pass actually runs ``kpg/32`` atoms of 32, and each wave along K
    takes an equal slice, so the split has to divide that count instead.
    """
    if knobs["fold_k32"]:
        if problem.kpg % 32:
            return f"fold_k32 needs kpg % 32 == 0 (got kpg={problem.kpg})"
        n_atoms = problem.kpg // 32
    else:
        n_atoms = (problem.kpg + 15) // 16
    if n_atoms == 0 or n_atoms % knobs["waves_k"]:
        return f"{n_atoms} K-atoms do not split over waves_k={knobs['waves_k']}"
    return None


def _uses_coalesced_weights(knobs: dict) -> bool:
    """Does the MFMA dgrad's fprop pass read the reorganized weight layout?

    Only when K is split over waves or walked by the runtime K loop; otherwise
    it reads the plain transposed weights and the reorganize pass is skipped.
    """
    return knobs["waves_k"] > 1 or knobs["runtime_k_loop"]


def validate_spec(variant: str, spec, arch: str) -> Tuple[bool, str]:
    """``(ok, reason)`` for ``spec`` on ``arch``, from the kernel's own validator."""
    from kernels.common import conv_direct_grouped as dc

    try:
        if variant in ("direct_grouped", "direct_grouped_dgrad_mfma"):
            spec.validate()
            return dc.is_valid_spec(spec, arch=arch)
        if variant == "direct_depthwise":
            return dc.is_valid_depthwise_spec(spec, arch=arch)
        if variant == "direct_depthwise_col":
            spec.validate()
            return dc.is_valid_depthwise_col_spec(spec, arch=arch)
        if variant == "direct_depthwise_dgrad":
            return dc.is_valid_depthwise_dgrad_stream_spec(spec, arch=arch)
        if variant == "direct_grouped_dgrad":
            return dc.is_valid_dgrad_spec(spec, arch=arch)
        if variant in (_TRANSPOSE, _REORGANIZE):
            return True, "ok"
    except ValueError as e:
        return False, str(e)
    raise ValueError(f"unknown direct conv variant {variant!r}")


def _build_kernel(variant: str, spec, arch: str):
    from kernels.common import conv_direct_grouped as dc

    builders = {
        "direct_grouped": dc.build_direct_conv,
        "direct_grouped_dgrad_mfma": dc.build_direct_conv,
        "direct_depthwise": dc.build_direct_depthwise,
        "direct_depthwise_col": dc.build_direct_depthwise_col,
        "direct_depthwise_dgrad": dc.build_direct_depthwise_dgrad_streaming,
        "direct_grouped_dgrad": dc.build_direct_conv_dgrad,
        _TRANSPOSE: dc.build_direct_transpose_weights_dgrad,
        _REORGANIZE: dc.build_direct_reorganize_weights,
    }
    return builders[variant](spec, arch=arch)


# ---------------------------------------------------------------------------
# Identities and jobs
# ---------------------------------------------------------------------------


def _identity(
    arch: str,
    wave_size: int,
    variant: str,
    direction: str,
    caps: DirectCaps,
    knobs: dict,
    dtype: str = DIRECT_DTYPE,
) -> KernelIdentity:
    return KernelIdentity(
        arch=arch,
        direction=direction,
        algorithm=variant,
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_d=dtype,
        # Direct conv is not a GEMM: no tile, warp or vector fields.
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        pipeline="direct",
        epilogue="direct",
        wave_size=wave_size,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        filter_h=caps.KH,
        filter_w=caps.KW,
        stride_h=caps.stride,
        stride_w=caps.stride,
        dilation_h=1,
        dilation_w=1,
        pad_h=caps.PAD,
        pad_w=caps.PAD,
        cpg=caps.cpg,
        kpg=caps.kpg,
        knobs=json.dumps(knobs, sort_keys=True),
    )


def _helper_identity(
    arch: str,
    wave_size: int,
    variant: str,
    caps: DirectCaps,
    knobs: dict,
    dtype: str = DIRECT_DTYPE,
) -> KernelIdentity:
    # The weight transforms depend on the filter and channel counts only;
    # stride and padding are zeroed so every pipeline entry that shares a
    # filter shares the helper binary.
    return _identity(
        arch,
        wave_size,
        variant,
        "direct_dgrad_helper",
        replace(caps, PAD=0, stride=0),
        knobs,
        dtype,
    )


def _caps_of(identity: KernelIdentity) -> DirectCaps:
    return DirectCaps(
        KH=identity.filter_h,
        PAD=identity.pad_h,
        stride=identity.stride_h,
        cpg=identity.cpg,
        kpg=identity.kpg,
    )


def probe_problem(caps: DirectCaps, dtype: str = DIRECT_DTYPE):
    """A problem with exactly ``caps`` that every knob in the grid fits.

    The emitted IR does not depend on N, H, W or groups (the shape-invariance
    cases in test_conv_abi.py assert it), so the probe only has to satisfy the
    spec validators: 64 groups divide every block_groups, 256 depthwise
    channels fill the widest depthwise block, and 64x64 leaves every stride-2
    output wider than the widest W tile.
    """
    from kernels.common.conv_direct_grouped import DirectConvProblem

    groups = 256 if caps.cpg == 1 else 64
    return DirectConvProblem(
        N=1,
        H=64,
        W=64,
        groups=groups,
        cpg=caps.cpg,
        kpg=caps.kpg,
        KH=caps.KH,
        KW=caps.KW,
        PAD=caps.PAD,
        stride=caps.stride,
        dtype=dtype,
    )


def _job(identity: KernelIdentity, caps: DirectCaps, knobs: dict) -> BuildJob:
    return BuildJob(
        identity=identity,
        direction=identity.direction,
        spec_kwargs=knobs,
        caps=dict(
            KH=caps.KH, PAD=caps.PAD, stride=caps.stride, cpg=caps.cpg, kpg=caps.kpg
        ),
    )


def _caps_jobs(
    arch: str,
    wave_size: int,
    variant: str,
    direction: str,
    caps: DirectCaps,
    dtype: str,
) -> Tuple[int, List[Tuple[str, BuildJob]]]:
    """Walk one (variant, capability) cell of the grid.

    Returns the raw candidate count and the valid ``(key, job)`` pairs in
    generator order; duplicates are resolved by the caller's ordered merge.
    The MFMA dgrad entries pull in their weight-transform helpers.
    """
    out: List[Tuple[str, BuildJob]] = []
    n_raw = 0
    probe = probe_problem(caps, dtype)
    for knobs in _knob_grid(variant):
        n_raw += 1
        if variant == "direct_grouped_dgrad_mfma" and _mfma_knobs_reason(probe, knobs):
            continue
        spec = make_spec(variant, probe, knobs)
        if not validate_spec(variant, spec, arch)[0]:
            continue
        cell = [
            _job(
                _identity(
                    arch, wave_size, variant, f"direct_{direction}", caps, knobs, dtype
                ),
                caps,
                knobs,
            )
        ]
        if variant == "direct_grouped_dgrad_mfma":
            cell.append(
                _job(
                    _helper_identity(arch, wave_size, _TRANSPOSE, caps, {}, dtype),
                    caps,
                    {},
                )
            )
            if _uses_coalesced_weights(knobs):
                reorg = {"fold_k32": knobs["fold_k32"]}
                cell.append(
                    _job(
                        _helper_identity(
                            arch, wave_size, _REORGANIZE, caps, reorg, dtype
                        ),
                        caps,
                        reorg,
                    )
                )
        out.extend((job.identity.stable_hash(), job) for job in cell)
    return n_raw, out


def _caps_worker(payload):
    return _caps_jobs(*payload)


def count_direct_jobs(directions: Sequence[str]) -> int:
    """Raw candidate count :func:`enumerate_direct_jobs` walks (the progress
    denominator): capabilities times tuning knobs, before validation."""
    return sum(
        len(caps_list) * sum(1 for _ in _knob_grid(variant))
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
    )


def enumerate_direct_jobs(
    *,
    arch: str,
    target,
    directions: Sequence[str],
    dtype: str = DIRECT_DTYPE,
    jobs: int = 1,
    log=None,
    log_every_s: float = 5.0,
) -> List[BuildJob]:
    """Every direct kernel worth caching for ``arch``, pre-validated.

    ``directions`` takes the benchmark's names, ``fwd`` and ``dgrad``. The MFMA
    dgrad entries pull in their weight-transform helpers.

    The grid is walked one (variant, capability) cell at a time; with
    ``jobs > 1`` the cells run in a process pool. Cells are merged in order,
    so the result (including its order) does not depend on ``jobs``. With
    ``log`` set a progress line is emitted at most every ``log_every_s``
    seconds, as for the implicit-GEMM enumeration.
    """
    for d in directions:
        if d not in ("fwd", "dgrad"):
            raise ValueError(f"direct conv has no {d!r} kernel; expected fwd or dgrad")
    if dtype not in DIRECT_DTYPES:
        raise ValueError(f"direct conv builds {DIRECT_DTYPES}, not {dtype!r}")
    wave_size = target.wave_size
    jobs = max(1, int(jobs))
    payloads = [
        (arch, wave_size, variant, direction, caps, dtype)
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
        for caps in caps_list
    ]

    total = 0
    if log is not None:
        total = count_direct_jobs(directions)
        log(f"  {total} candidates to check with {jobs} process(es)")

    n_raw = n_valid = 0
    started = last_log = time.perf_counter()
    results: List[Optional[List[Tuple[str, BuildJob]]]] = [None] * len(payloads)

    def _done(idx: int, cell_raw: int, cell_out) -> None:
        nonlocal n_raw, n_valid, last_log
        results[idx] = cell_out
        n_raw += cell_raw
        n_valid += len(cell_out)
        if log is None:
            return
        now = time.perf_counter()
        if now - last_log >= log_every_s:
            last_log = now
            pct = 100.0 * n_raw / total if total else 0.0
            log(
                f"  enumerating: {n_raw}/{total} candidates checked "
                f"({pct:.1f}%), ~{n_valid} valid ({now - started:.0f}s)"
            )

    if jobs <= 1:
        for idx, payload in enumerate(payloads):
            _done(idx, *_caps_jobs(*payload))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_caps_worker, payload): idx
                for idx, payload in enumerate(payloads)
            }
            for fut in as_completed(futures):
                _done(futures[fut], *fut.result())

    seen = set()
    per_direction: Dict[str, List[BuildJob]] = {d: [] for d in directions}
    for payload, cell_out in zip(payloads, results):
        for key, job in cell_out:
            if key not in seen:
                seen.add(key)
                per_direction[payload[3]].append(job)
    # Round-robin across directions, as the implicit-GEMM enumeration does, so
    # a --limit smoke build covers every requested direction.
    out: List[BuildJob] = []
    streams = [iter(per_direction[d]) for d in directions]
    while streams:
        still_running = []
        for stream in streams:
            job = next(stream, None)
            if job is not None:
                out.append(job)
                still_running.append(stream)
        streams = still_running
    if log is not None:
        log(
            f"  enumerated {n_raw} candidates -> {len(out)} unique valid "
            f"({time.perf_counter() - started:.0f}s)"
        )
    return out


def build_direct_job(job: BuildJob, arch: str, dtype: str):
    """Build one direct job's kernel IR. Returns ``(kernel, meta)``.

    Compilation, deduplication and incremental rebuilds are
    :func:`compile_jobs`'s; this only decides what the kernel is.
    """
    if dtype not in DIRECT_DTYPES:
        raise ValueError(f"direct conv builds {DIRECT_DTYPES}, not {dtype!r}")
    caps = DirectCaps(**job.caps)
    variant = job.identity.algorithm
    spec = make_spec(variant, probe_problem(caps, dtype), job.spec_kwargs)
    ok, why = validate_spec(variant, spec, arch)
    if not ok:
        raise ValueError(why)
    return _build_kernel(variant, spec, arch), {}


def compile_all_direct(
    *,
    cache: KernelCache,
    arch: str,
    target,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
    dtype: str = DIRECT_DTYPE,
) -> int:
    """Populate ``cache`` with every direct kernel in :data:`DIRECT_CAPABILITIES`
    for operand dtype ``dtype``."""
    log(
        f"AOT compile-all: enumerating direct variants for {arch}/{dtype} "
        f"({', '.join(directions)})"
    )
    return compile_jobs(
        cache=cache,
        all_jobs=enumerate_direct_jobs(
            arch=arch,
            target=target,
            directions=directions,
            dtype=dtype,
            jobs=jobs,
            log=log,
        ),
        build=build_direct_job,
        arch=arch,
        dtype=dtype,
        directions=[f"direct_{d}" for d in directions],
        jobs=jobs,
        limit=limit,
        log=log,
    )


# ---------------------------------------------------------------------------
# Running a problem out of the cache (CPU half)
# ---------------------------------------------------------------------------


def _transform_signature(dtype: str) -> list:
    """Launch signature of the weight-transform kernels: source, destination
    and their byte sizes, nothing else."""
    elem = {"fp16": "f16", "bf16": "bf16"}[dtype]
    return [
        {"name": "A", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "D", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
    ]


@dataclass(frozen=True)
class DirectStep:
    """One kernel launch of a plan.

    ``buffers`` names the buffers bound to the A, B and D kernarg slots (B is
    ``None`` for the two-operand weight transforms). ``conv_args`` is ``None``
    for those too; a conv kernel takes its extents from it.
    """

    kernel_name: str
    hsaco_path: Path
    signature: list
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    buffers: Tuple[str, Optional[str], str]
    conv_args: object = None


@dataclass(frozen=True)
class DirectPlan:
    """Everything needed to run one cached kernel (or pipeline) on a problem.

    Buffer names: ``x``/``w``/``y`` for forward, ``dy``/``w``/``dx`` for
    dgrad, plus the scratch buffers in ``workspaces`` (name -> bytes).
    """

    identity: KernelIdentity
    steps: Tuple[DirectStep, ...]
    workspaces: Dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.identity.short_label()


def _conv_step(identity, hsaco_path, meta, spec, problem, direction, buffers):
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import direct_launch_geometry

    grid, block = direct_launch_geometry(spec)
    return DirectStep(
        kernel_name=meta["kernel_name"],
        hsaco_path=hsaco_path,
        signature=conv_direct_args_signature(identity.dtype_a, direction=direction),
        grid=grid,
        block=block,
        buffers=buffers,
        conv_args=ConvArgs.from_problem(problem, direction=direction),
    )


def plan_for(
    cache: KernelCache,
    identity: KernelIdentity,
    hsaco_path: Path,
    meta: dict,
    problem,
    arch: str,
) -> Tuple[Optional[DirectPlan], str]:
    """Launch plan for one cached kernel on ``problem``, or ``(None, why)``.

    The capability match (:meth:`KernelCache.supports_problem`) is necessary
    but not sufficient: tuning knobs have shape rules of their own (groups
    divisible by block_groups, a persistent grid's cell budget, ...). The
    spec is rebuilt on the real problem and run through the kernel's own
    validator, so a binary is only offered where a fresh build would accept
    the shape too.
    """
    from kernels.common import conv_direct_grouped as dc

    if not meta.get("kernel_name"):
        return None, "cache entry has no kernel_name (rebuild the cache)"
    variant = identity.algorithm
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if variant == "direct_grouped_dgrad_mfma":
        why = _mfma_knobs_reason(problem, knobs)
        if why:
            return None, why
    spec = make_spec(variant, problem, knobs)
    ok, why = validate_spec(variant, spec, arch)
    if not ok:
        return None, why

    if identity.direction == "direct_fwd":
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "fwd", ("x", "w", "y")
        )
        return DirectPlan(identity, (step,)), "ok"
    if variant != "direct_grouped_dgrad_mfma":
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "dgrad", ("dy", "w", "dx")
        )
        return DirectPlan(identity, (step,)), "ok"

    # MFMA pipeline: W -> W_T (transpose), optionally W_T -> W_coa
    # (reorganize), then a forward pass over the transposed problem.
    caps = _caps_of(identity)
    helpers = [(_TRANSPOSE, {}, "w", "ws_t")]
    use_coalesced = _uses_coalesced_weights(knobs)
    if use_coalesced:
        helpers.append((_REORGANIZE, {"fold_k32": knobs["fold_k32"]}, "ws_t", "ws_coa"))
    steps = []
    for helper, helper_knobs, src, dst in helpers:
        helper_id = _helper_identity(
            identity.arch,
            identity.wave_size,
            helper,
            caps,
            helper_knobs,
            identity.dtype_a,
        )
        entry = cache.get(helper_id)
        if entry is None:
            return None, f"missing cached {helper} helper (rebuild the cache)"
        helper_path = cache.hsaco_path(helper_id)
        helper_spec = make_spec(helper, problem, helper_knobs)
        grid, block = dc.direct_launch_geometry(helper_spec)
        steps.append(
            DirectStep(
                kernel_name=entry[1]["kernel_name"],
                hsaco_path=helper_path,
                signature=_transform_signature(identity.dtype_a),
                grid=grid,
                block=block,
                buffers=(src, None, dst),
            )
        )
    weights = "ws_coa" if use_coalesced else "ws_t"
    # The compute pass is a *forward* kernel: forward ABI, transposed extents.
    steps.append(
        _conv_step(
            identity, hsaco_path, meta, spec, spec.problem, "fwd", ("dy", weights, "dx")
        )
    )
    workspaces = {"ws_t": dc.direct_dgrad_workspace_bytes(problem)}
    if use_coalesced:
        workspaces["ws_coa"] = dc.direct_dgrad_coalesced_workspace_bytes(
            problem, fold_k32=knobs["fold_k32"]
        )
    return DirectPlan(identity, tuple(steps), workspaces), "ok"


def direct_plans(
    cache: KernelCache, problem, direction: str, arch: str
) -> Tuple[List[DirectPlan], List[Tuple[KernelIdentity, str]]]:
    """Launch plans for every cached kernel that can run ``problem``.

    Returns ``(plans, rejected)``; ``rejected`` pairs each capability-matching
    kernel that failed the per-shape re-validation with the reason, so an
    empty plan list is diagnosable.
    """
    plans: List[DirectPlan] = []
    rejected: List[Tuple[KernelIdentity, str]] = []
    for identity, hsaco_path, meta in cache.compatible(
        problem, direction=f"direct_{direction}"
    ):
        # A binary only serves the operand dtype it was built for.
        if identity.dtype_a != getattr(problem, "dtype", DIRECT_DTYPE):
            continue
        plan, why = plan_for(cache, identity, hsaco_path, meta, problem, arch)
        if plan is None:
            rejected.append((identity, why))
        else:
            plans.append(plan)
    return plans, rejected


def launch_values(
    step: DirectStep, ptrs: Dict[str, int], sizes: Dict[str, int]
) -> dict:
    """Kernarg values of ``step`` with its buffers bound to ``ptrs``/``sizes``."""
    a, b, d = step.buffers
    if step.conv_args is None:
        return {"A": ptrs[a], "D": ptrs[d], "A_bytes": sizes[a], "D_bytes": sizes[d]}
    return step.conv_args.to_launch_values(
        ptrs[a], ptrs[b], ptrs[d], sizes[a], sizes[b], sizes[d]
    )
