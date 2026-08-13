# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_gfx950_sweep.py -- what rolling is worth across ALL of kernels/gfx950.
#
#   python3 -m rocke.portable_ir.drivers.roll_gfx950_sweep [--family F] [--phase N]
#
# The other drivers gate a hand-picked set of axes that are known to work. This
# one is the opposite: point it at every build entry point in kernels/gfx950,
# every axis those kernels expose, and every feature flag that changes what they
# emit, then report what rolled and what did not. It is a survey, not a gate --
# a refusal here is a finding, not a failure.
#
# Four phases, each answering a different question:
#
#   1 DOMAIN   How many values does each axis legally take? Asked of the kernel
#              itself, through three layers of its own gating: the spec's
#              __post_init__, the supports_* admission function, and finally an
#              actual build. Each layer catches things the previous one does not,
#              and skipping them inflates every number downstream.
#   2 AXIS     Does each axis roll on its own, and if not, why?
#   3 CROSS    Does ONE recipe cover the cross product of the axes that rolled?
#              This is the roll_nd payoff, measured as points covered per trace
#              recorded.
#   4 FEATURE  Does that cross-product recipe still hold with each feature flag
#              flipped? Flags are not axes -- they are compile-time choices that
#              select different code -- so the question is not whether they roll
#              but whether rolling survives them.
#
# One finding is baked in as a `coherent` predicate rather than discovered each
# run: the tiled 2D and 3D kernels accept a `num_kv_heads` that does not divide
# `num_query_heads`. Neither the spec's __post_init__ nor supports_* rejects it
# (the admission check only bounds `num_queries_per_kv`, which is computed with a
# floor division that quietly swallows the remainder), and the kernel then builds
# and bakes in a group size that does not correspond to any real grouping.
# attention_dense rejects the same combination outright. The sweep filters those
# points so the rolling numbers are not polluted, but the underlying gap belongs
# to the kernels.

from __future__ import annotations

import argparse
import itertools
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rocke.portable_ir.drivers.roll_kernel import (
    Kernel,
    arch_refusals,
    recipe_ops,
    roll,
)
from rocke.portable_ir.src.recipe_bundle import cbor_encode
from rocke.portable_ir.src.recording_builder import record_kernel
from rocke.portable_ir.src.roll_regimes import legal_values

ARCH = "gfx950"

# Base configurations. Chosen to be realistic rather than minimal: each is a
# shape the kernel is actually deployed at, so the sweep measures the kernels as
# used rather than at some degenerate corner.
DENSE = dict(
    batch=1,
    seqlen_q=512,
    seqlen_kv=512,
    num_query_heads=128,
    num_kv_heads=8,
    head_size=128,
    causal=True,
    dtype="bf16",
    block_n=64,
    waves_per_eu=2,
)
T2D = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_warps=4,
    num_seqs=64,
)
T3D = dict(
    head_size=64,
    block_size=32,
    num_query_heads=64,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_segments=4,
    num_seqs=64,
)
RED = dict(
    head_size=64, num_query_heads=64, num_kv_heads=8, dtype="bf16", num_segments=4
)

# Candidate values probed for legality, per axis. Deliberately wider than any
# kernel accepts -- the point is to let the kernel do the rejecting.
CANDIDATES: Dict[str, List[int]] = {
    "batch": list(range(1, 65)),
    "seqlen_q": [256 * i for i in range(1, 17)],
    "seqlen_kv": [64 * i for i in range(1, 33)],
    "sliding_window": [0] + [64 * i for i in range(1, 17)],
    "num_persistent": [64 * i for i in range(1, 17)],
    "num_segments": list(range(1, 33)),
    "num_seqs": [16 * i for i in range(1, 33)],
    "num_warps": [1, 2, 4, 8],
    "block_m_per_warp": [16, 32],
    "kq_lds_pad_halves": list(range(1, 33)),
    "kv_ring_depth": [2, 3],
    "tile_size": [32 * i for i in range(1, 17)],
    "waves_per_eu": list(range(1, 9)),
    "block_n": [32 * i for i in range(1, 17)],
    "head_size": [16 * i for i in range(1, 17)],
    "block_size": [16 * i for i in range(1, 17)],
}
_DEFAULT_CANDIDATES = [16 * i for i in range(1, 33)]

# Sample values per axis; set from --samples.
_K = 2


class Family:
    """One build entry point, with everything needed to sweep it.

    The spec-making, gating and building are handed to a ``roll_kernel.Kernel``,
    which is the same binding the generic driver rolls through. These kernels do
    not follow the conventions that driver's module lookup assumes -- they gate
    through ``supports_*`` taking a dozen keyword arguments, and one of them
    builds its spec out of another kernel's spec -- so the binding is written
    out here rather than discovered. What matters is that it is written out
    once: every phase below then asks the same questions the same way."""

    def __init__(
        self,
        label: str,
        make_spec: Callable[..., Any],
        build: Callable[..., Any],
        axes: Sequence[str],
        flags: Sequence[Tuple[str, Dict[str, Any]]],
        admits: Optional[Callable[[Any], Any]] = None,
        coherent: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self.label = label
        self.make_spec = make_spec
        self.build = build
        self.axes = list(axes)
        self.flags = list(flags)
        self.admits = admits
        # Constraints the kernel does NOT enforce but that its emitted code
        # depends on -- see the note on num_kv_heads in the module docstring.
        self.coherent = coherent
        self.kernel = Kernel(
            label=label,
            make_spec=make_spec,
            build_at=build,
            gate=admits,
            coherent=coherent,
        )

    def legal_point(self, point: Dict[str, Any], *, build: bool = True) -> bool:
        """Is this whole combination legal, not just each value on its own?

        Axes interact. `num_query_heads` and `num_kv_heads` are each fine over a
        wide range and illegal together unless one divides the other, so a cross
        product built from two independently-legal axis lists can contain points
        the kernel refuses to build.

        `build=False` skips the build probe and consults only the declarative
        gates. That is the cheap form, used when scanning many combinations to
        choose a grid -- coupling between axes is declared there, while what the
        build probe adds is per-value constraints already covered in phase 1."""
        if arch_refusals(self.kernel, lambda p: p, [point], ARCH):
            return False
        if build:
            try:
                self.build(**point)
            except Exception:
                return False
        return True


def _families() -> List[Family]:
    from kernels.gfx950.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
        supports_attention_dense,
    )
    from kernels.gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
        supports_tiled_2d,
    )
    from kernels.gfx950.attention_tiled_2d_fastkv_regp import (
        build_unified_attention_2d_fastkv_register_p,
        make_fastkv_register_p_spec,
    )
    from kernels.gfx950.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec,
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_3d_tiled,
        build_unified_attention_reduce_tiled,
        supports_tiled_3d,
    )

    def spec_maker(cls, base):
        return lambda **kw: cls(**{**base, **kw})

    def builder(cls, base, fn):
        def b(**point):
            return fn(cls(**{**base, **point}), arch=ARCH)

        return b

    def admits_2d(spec):
        return supports_tiled_2d(
            head_size=spec.head_size,
            block_size=spec.block_size,
            dtype=spec.dtype,
            num_queries_per_kv=spec.num_queries_per_kv,
            use_alibi=spec.use_alibi,
            use_qq_bias=spec.use_qq_bias,
            use_fp8=bool(spec.use_fp8_mfma_qk),
            q_dtype=None,
            num_warps=spec.num_warps,
            block_m_per_warp=spec.block_m_per_warp,
            kv_storage_dtype=spec.kv_storage_dtype,
            tile_size=spec.tile_size,
            arch=ARCH,
            use_mfma_32x32x8=spec.use_mfma_32x32,
            use_transposed_qk_32x32=spec.use_transposed_qk_32x32,
            use_k_single_buffer=spec.use_k_single_buffer,
        )

    def admits_3d(spec):
        return supports_tiled_3d(
            head_size=spec.head_size,
            block_size=spec.block_size,
            dtype=spec.dtype,
            num_queries_per_kv=spec.num_queries_per_kv,
            use_alibi=spec.use_alibi,
            use_qq_bias=spec.use_qq_bias,
            use_fp8=False,
            q_dtype=None,
            kv_storage_dtype=spec.kv_storage_dtype,
            arch=ARCH,
        )

    def gqa_coherent(base):
        """`num_kv_heads` must divide `num_query_heads` -- unchecked by these two
        kernels, so the sweep has to check it to keep the domains honest."""

        def ok(point):
            nq = point.get("num_query_heads", base["num_query_heads"])
            nkv = point.get("num_kv_heads", base["num_kv_heads"])
            return nkv > 0 and nq % nkv == 0

        return ok

    def fastkv_spec(**kw):
        return make_fastkv_register_p_spec(UnifiedAttention2DTiledSpec(**{**T2D, **kw}))

    def fastkv_build(**point):
        return build_unified_attention_2d_fastkv_register_p(
            fastkv_spec(**point), arch=ARCH
        )

    return [
        Family(
            "attention_dense",
            spec_maker(AttentionDenseSpec, DENSE),
            builder(AttentionDenseSpec, DENSE, build_attention_dense),
            [
                "batch",
                "seqlen_q",
                "seqlen_kv",
                "num_query_heads",
                "num_kv_heads",
                "waves_per_eu",
                "num_persistent",
                "head_size",
                "block_n",
                "sliding_window",
            ],
            [
                ("causal=False", {"causal": False}),
                ("dtype=fp16", {"dtype": "fp16"}),
                ("ragged", {"ragged": True, "seqlen_q": 512, "seqlen_kv": 512}),
                ("varlen", {"varlen": True}),
                ("persistent", {"persistent": True}),
                ("persistent+interleave", {"persistent": True, "interleave": True}),
                (
                    "persist_decode=qb_major",
                    {"persistent": True, "persist_decode": "qb_major"},
                ),
                (
                    "persist_decode=hkv_major",
                    {"persistent": True, "persist_decode": "hkv_major"},
                ),
                ("lazy_rescale=False", {"lazy_rescale": False}),
                ("sliding_window=128", {"sliding_window": 128}),
            ],
            admits=lambda s: supports_attention_dense(s, arch=ARCH),
        ),
        Family(
            "attention_tiled_2d",
            spec_maker(UnifiedAttention2DTiledSpec, T2D),
            builder(UnifiedAttention2DTiledSpec, T2D, build_unified_attention_2d_tiled),
            [
                "head_size",
                "block_size",
                "num_query_heads",
                "num_kv_heads",
                "num_seqs",
                "num_warps",
                "tile_size",
                "block_m_per_warp",
                "kq_lds_pad_halves",
                "sliding_window",
                "kv_ring_depth",
            ],
            [
                ("dtype=fp16", {"dtype": "fp16"}),
                ("use_sinks", {"use_sinks": True}),
                ("has_softcap", {"has_softcap": True}),
                ("use_alibi", {"use_alibi": True}),
                ("use_qq_bias", {"use_qq_bias": True}),
                ("sliding_window=256", {"sliding_window": 256}),
                ("fp8 kv cache", {"kv_storage_dtype": "fp8e4m3"}),
                (
                    "fp8 mfma qk+pv",
                    {
                        "kv_storage_dtype": "fp8e4m3",
                        "use_fp8_mfma_qk": True,
                        "use_fp8_mfma_pv": True,
                    },
                ),
                ("use_v_double_buffer", {"use_v_double_buffer": True}),
                (
                    "v_double_buffer+stagger",
                    {"use_v_double_buffer": True, "use_staggered_iter_wait": True},
                ),
                ("use_kq_lds_pad", {"use_kq_lds_pad": True}),
                ("use_i64_kv_addr", {"use_i64_kv_addr": True}),
                (
                    "use_sched_barrier",
                    {"use_sched_barrier": True, "sched_barrier_mask": 1},
                ),
                ("mfma_32x32", {"use_mfma_32x32": True, "block_m_per_warp": 32}),
                (
                    "transposed_qk_32x32",
                    {
                        "use_mfma_32x32": True,
                        "block_m_per_warp": 32,
                        "use_transposed_qk_32x32": True,
                    },
                ),
                (
                    "transposed+scalar_state",
                    {
                        "use_mfma_32x32": True,
                        "block_m_per_warp": 32,
                        "use_transposed_qk_32x32": True,
                        "use_transposed_scalar_state": True,
                    },
                ),
                (
                    "transposed+mask_once",
                    {
                        "use_mfma_32x32": True,
                        "block_m_per_warp": 32,
                        "use_transposed_qk_32x32": True,
                        "use_transposed_mask_once": True,
                    },
                ),
                (
                    "transposed+half_local_pv",
                    {
                        "use_mfma_32x32": True,
                        "block_m_per_warp": 32,
                        "use_transposed_qk_32x32": True,
                        "use_transposed_half_local_pv": True,
                    },
                ),
                (
                    "kv_ring_depth=3",
                    {
                        "use_mfma_32x32": True,
                        "block_m_per_warp": 32,
                        "use_transposed_qk_32x32": True,
                        "kv_ring_depth": 3,
                    },
                ),
                ("use_k_single_buffer", {"use_k_single_buffer": True}),
                ("use_early_v_schedule", {"use_early_v_schedule": True}),
                ("use_register_pv", {"use_register_pv": True}),
                ("softmax_mfma_interleave", {"use_softmax_mfma_interleave": True}),
            ],
            admits=admits_2d,
            coherent=gqa_coherent(T2D),
        ),
        Family(
            "attention_tiled_3d",
            spec_maker(UnifiedAttention3DTiledSpec, T3D),
            builder(UnifiedAttention3DTiledSpec, T3D, build_unified_attention_3d_tiled),
            [
                "head_size",
                "block_size",
                "num_query_heads",
                "num_kv_heads",
                "num_segments",
                "num_seqs",
                "sliding_window",
            ],
            [
                ("dtype=fp16", {"dtype": "fp16"}),
                ("use_sinks", {"use_sinks": True}),
                ("has_softcap", {"has_softcap": True}),
                ("use_alibi", {"use_alibi": True}),
                ("use_qq_bias", {"use_qq_bias": True}),
                ("sliding_window=256", {"sliding_window": 256}),
                ("fp8 kv cache", {"kv_storage_dtype": "fp8e4m3"}),
                ("use_i64_kv_addr", {"use_i64_kv_addr": True}),
                ("use_invariant_hoist", {"use_invariant_hoist": True}),
                ("use_wide_kv_load", {"use_wide_kv_load": True}),
            ],
            admits=admits_3d,
            coherent=gqa_coherent(T3D),
        ),
        Family(
            "attention_reduce",
            spec_maker(UnifiedAttentionReduceTiledSpec, RED),
            builder(
                UnifiedAttentionReduceTiledSpec,
                RED,
                build_unified_attention_reduce_tiled,
            ),
            ["head_size", "num_query_heads", "num_kv_heads", "num_segments"],
            [
                ("dtype=fp16", {"dtype": "fp16"}),
                ("waves_per_eu=4", {"waves_per_eu": 4}),
            ],
            coherent=gqa_coherent(RED),
        ),
        Family(
            "fastkv_regp",
            fastkv_spec,
            fastkv_build,
            [
                "num_seqs",
                "sliding_window",
                "num_query_heads",
                "num_kv_heads",
                "head_size",
                "block_size",
            ],
            [
                ("use_sinks", {"use_sinks": True}),
                ("has_softcap", {"has_softcap": True}),
                ("use_alibi", {"use_alibi": True}),
                ("dtype=fp16", {"dtype": "fp16"}),
            ],
        ),
    ]


def _hb(n: float) -> str:
    """Bytes at a readable scale. Recipes here span 4KiB to tens of MiB."""
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return f"{n:.0f}B"


class AxisResult:
    """What one axis did on its own, including what its recipe costs on disk."""

    def __init__(
        self,
        ok: bool,
        why: str,
        samples: List[int],
        points: int = 0,
        cbor: int = 0,
        concrete: int = 0,
    ):
        self.ok = ok
        self.why = why
        self.samples = samples
        self.points = points  # points this one-axis recipe covers
        self.cbor = cbor  # bytes of the rolled recipe, CBOR-encoded
        self.concrete = concrete  # mean bytes of ONE concrete recipe on this axis


class CrossResult:
    """The roll_nd payoff for one family, in points and in bytes.

    `concrete` is the projection this driver reports as the no-rolling baseline:
    serving N shapes without a recipe means shipping N recipes, so it is the mean
    recorded trace size times the number of points covered. It is a projection
    rather than a measurement because encoding all N would mean re-recording every
    verified point; `spread` is carried alongside it so the reader can see how much
    the estimate could move -- for a constants-only roll the concrete recipes have
    identical op counts and differ only in integer widths, so it is nearly flat."""

    def __init__(
        self, points: int, traces: int, ops: int, cbor: int, trace_bytes: List[int]
    ):
        self.points = points
        self.traces = traces
        self.ops = ops
        self.cbor = cbor
        self.trace_bytes = trace_bytes
        self.mean = sum(trace_bytes) // max(1, len(trace_bytes))
        self.authoring = sum(trace_bytes)  # what the traces themselves cost
        self.concrete = self.mean * points

    @property
    def spread(self) -> str:
        if not self.trace_bytes:
            return "-"
        lo, hi = min(self.trace_bytes), max(self.trace_bytes)
        return f"{_hb(lo)}-{_hb(hi)}" if hi != lo else _hb(lo)

    @property
    def ratio(self) -> float:
        return self.concrete / max(1, self.cbor)

    @property
    def overhead(self) -> float:
        """What the parametric form costs per recipe, as a percentage.

        A constants-only recipe holds the same instructions as any one concrete
        recipe, with intexpr trees where plain integers used to be -- so it is
        slightly *larger* than what it replaces, and all of the win is in needing
        one instead of `points`. A negative number means the structural roller
        found repetition to fold away."""
        return 100.0 * (self.cbor - self.mean) / max(1, self.mean)


def _samples(vals: List[int], k: Optional[int] = None) -> Tuple[List[int], List[int]]:
    """Sample points and one extrapolated holdout, spread over the legal domain.

    Spread matters more than count. Adjacent values are often powers of two of
    each other, and a constant that is invariant across two powers of two (every
    magic multiplier is 1 there) reads as frozen -- a wrong model that verifies.
    Taking values from across the domain avoids that by construction."""
    k = _K if k is None else k
    if len(vals) <= k:
        return vals[: max(2, k - 1)], vals[k - 1 : k]
    idx = [(i * len(vals)) // (k + 1) for i in range(k)]
    picks = sorted({vals[i] for i in idx})
    hold = [vals[-1]] if vals[-1] not in picks else []
    return picks, hold


def phase_domain(fam: Family, verbose: bool = True) -> Dict[str, List[int]]:
    """How many values each axis legally takes, per the kernel's own gating."""
    out: Dict[str, List[int]] = {}
    for axis in fam.axes:
        cands = CANDIDATES.get(axis, _DEFAULT_CANDIDATES)
        ok = legal_values(
            axis, cands, fam.make_spec, admits=fam.admits, probe=fam.build
        )
        if fam.coherent:
            ok = [v for v in ok if fam.coherent({axis: v})]
        out[axis] = ok
        if verbose:
            span = f"{ok[0]}..{ok[-1]}" if ok else "-"
            print(
                f"    {axis:<20}{len(ok):>4} legal  {span:<14}"
                f"{'' if len(ok) >= 2 else '(too few to roll)'}"
            )
    return out


def phase_axis(
    fam: Family, domains: Dict[str, List[int]], verbose: bool = True
) -> Dict[str, AxisResult]:
    """Roll each axis on its own; record the refusal reason when it does not.

    Also sizes each single-axis recipe, because that is the honest baseline for
    what roll_nd buys: without it a family needs one of these per axis, and they
    cover a line through the space each rather than the volume together."""
    out: Dict[str, AxisResult] = {}
    for axis, vals in domains.items():
        if len(vals) < 2:
            out[axis] = AxisResult(False, f"only {len(vals)} legal value(s)", vals)
            continue
        samples, hold = _samples(vals)
        t0 = time.time()
        pts = nb = cb = 0
        r = roll(
            kernel=fam.kernel,
            arch=ARCH,
            axes={axis: samples},
            holdout={axis: hold} if hold else None,
            quiet=True,
        )
        ok, why = r.rolled, ("" if r.rolled else r.reason)
        if ok:
            pts = len(r.points)
            nb = len(r.cbor or b"")
            cb = sum(r.trace_bytes) // max(1, len(r.trace_bytes))
        out[axis] = AxisResult(ok, why, samples, pts, nb, cb)
        if verbose:
            dt = time.time() - t0
            note = f"{pts} pts, {nb / 1024:.1f}KiB recipe" if ok else why[:58]
            print(
                f"    {axis:<20}{str(samples):<22}"
                f"{'ROLLS' if ok else 'refused':<9}{dt:5.1f}s  {note}"
            )
    return out


def choose_grid(
    fam: Family,
    domains: Dict[str, List[int]],
    axis_results: Dict[str, AxisResult],
    overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """Pick sample values that are legal TOGETHER, not just one axis at a time.

    Naively crossing per-axis sample lists produces points the kernel rejects,
    because axes are coupled -- `num_kv_heads` has to divide `num_query_heads`,
    `seqlen_kv` has to be a multiple of `block_n`. Dropping whole axes on the
    first conflict is the wrong response: it throws away a cross product to fix a
    choice of values. So axes are added one at a time, each new axis keeping only
    the values that stay legal against every combination chosen so far.

    A feature flag can itself constrain the axes -- `ragged` requires
    `seqlen_q == seqlen_kv`, so a grid that varies them independently is illegal
    the moment it is set. Passing `overrides` re-chooses the grid under the flag,
    which keeps a harness artifact from being reported as a rolling failure.

    Returns (grid, holdout)."""
    overrides = overrides or {}
    order = sorted(
        (a for a, r in axis_results.items() if r.ok),
        key=lambda a: -len(domains[a]),
    )
    grid: Dict[str, List[int]] = {}
    hold: Dict[str, int] = {}
    for axis in order:
        combos = list(itertools.product(*grid.values())) or [()]
        feasible = [
            v
            for v in domains[axis]
            if all(
                fam.legal_point(
                    {**overrides, **dict(zip(grid, c)), axis: v}, build=False
                )
                for c in combos
            )
        ]
        if len(feasible) < 2:
            if verbose:
                print(
                    f"    skipping {axis}: only {len(feasible)} value(s) remain legal "
                    f"alongside {','.join(grid) or 'the base'}"
                )
            continue
        picks, held = _samples(feasible)
        if len(picks) < 2:
            continue
        grid[axis] = picks
        hold[axis] = held[0] if held else picks[-1]
    return grid, hold


def phase_cross(
    fam: Family,
    grid: Dict[str, List[int]],
    hold: Dict[str, int],
    overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    reasons: Optional[List[str]] = None,
) -> Optional[CrossResult]:
    """Roll ONE recipe over the given grid, and size what came out."""
    if not grid:
        if verbose:
            print("    no jointly-legal axis grid; nothing to cover")
        return None

    # A feature flag is just a spec field pinned for the whole roll, which is
    # what `fixed` means to the roller.
    t0 = time.time()
    r = roll(
        kernel=fam.kernel,
        arch=ARCH,
        axes=grid,
        fixed=dict(overrides or {}),
        holdout={a: [v] for a, v in hold.items()},
        quiet=True,
    )
    dt = time.time() - t0
    if not r.rolled:
        if reasons is not None:
            reasons.append(r.reason)
        if verbose:
            print(f"    REFUSED after {dt:.1f}s: {r.reason[:80]}")
        return None
    n_grid = len(list(itertools.product(*grid.values())))
    out = CrossResult(
        points=len(r.points),
        traces=r.n_recorded,
        ops=recipe_ops(r.recipe),
        cbor=len(r.cbor or b""),
        trace_bytes=list(r.trace_bytes),
    )
    if verbose:
        print(f"    {len(grid)} axes {','.join(grid)}")
        print(
            f"    {n_grid} grid + {len(r.points) - n_grid} held-out points verified "
            f"from {out.traces} traces  ({out.ops} ops, {dt:.1f}s)"
        )
        print(
            f"    one recipe = {_hb(out.cbor)} CBOR vs {_hb(out.concrete)} for "
            f"{out.points} concrete ({out.spread} each)  {out.ratio:.0f}x"
        )
    return out


def phase_feature(
    fam: Family,
    domains: Dict[str, List[int]],
    axis_results: Dict[str, AxisResult],
    verbose: bool = True,
) -> Tuple[int, int]:
    """Re-run the cross-product roll with each feature flag flipped.

    Flags are not axes: they pick different code at build time, so one recipe per
    setting is the expected answer. What matters is that rolling keeps working
    inside each setting -- a flag that silently breaks the cross product would
    mean the recipe only holds for default configurations."""
    held, total = 0, 0
    for name, over in fam.flags:
        try:
            _, recipe = record_kernel(lambda: fam.build(**over))
        except Exception as e:
            if verbose:
                print(
                    f"    {name:<26} unsupported at this base "
                    f"({type(e).__name__}: {str(e)[:34]})"
                )
            continue
        total += 1
        why: List[str] = []
        g, h = choose_grid(fam, domains, axis_results, overrides=over, verbose=False)
        got = phase_cross(fam, g, h, overrides=over, verbose=False, reasons=why)
        held += 1 if got else 0
        if verbose:
            n = recipe_ops(recipe)
            kib = len(cbor_encode(recipe)) / 1024
            verdict = (
                f"cross-roll {got.points:>4} pts / {got.traces:>2} traces  "
                f"{got.cbor / 1024:6.1f}KiB {got.ratio:>4.0f}x"
                if got
                else f"cross-roll REFUSED: {(why[0] if why else '')[:40]}"
            )
            print(f"    {name:<26}{n:>6} ops {kib:6.1f}KiB  {verdict}")
    return held, total


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", help="only sweep this family")
    ap.add_argument(
        "--phase",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="stop after this phase (each needs the ones before it). --phase 3 is "
        "the useful one: it measures the cross-product roll without paying for "
        "the feature pass, which re-rolls it once per flag",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=2,
        help="sample values per axis (default 2). 3 disambiguates candidate models "
        "and multiplies the verified grid, but attention_dense's 6-axis roll "
        "then verifies 729 points and takes minutes",
    )
    args = ap.parse_args(argv)
    global _K
    _K = args.samples

    print(f"gfx950 kernel rolling sweep  (arch={ARCH})")
    summary = []
    for fam in _families():
        if args.family and fam.label != args.family:
            continue
        print(f"\n================ {fam.label} ================")
        print("  [1] axis domains (spec validation + supports_* + a real build)")
        domains = phase_domain(fam)
        axis_results: Dict[str, AxisResult] = {}
        rolled: List[str] = []
        cross: Optional[CrossResult] = None
        held = total = 0
        if args.phase >= 2:
            print("  [2] one axis at a time")
            axis_results = phase_axis(fam, domains)
            rolled = [a for a, r in axis_results.items() if r.ok]
        if args.phase >= 3:
            print(f"  [3] cross product of the {len(rolled)} axes that rolled")
            grid, hold = choose_grid(fam, domains, axis_results)
            cross = phase_cross(fam, grid, hold)
        if args.phase >= 4:
            print("  [4] feature flags, each re-running the cross-product roll")
            held, total = phase_feature(fam, domains, axis_results)
        summary.append(
            (fam.label, len(domains), rolled, axis_results, cross, held, total)
        )

    print("\n\n================ summary: what one recipe covers ================")
    hdr = (
        f"{'family':<20}{'axes':>5}{'rolled':>7}{'points':>7}{'traces':>7}"
        f"{'flags held':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, n_axes, rolled, _, cross, held, total in summary:
        pts = str(cross.points) if cross else "-"
        tr = str(cross.traces) if cross else "-"
        print(
            f"{label:<20}{n_axes:>5}{len(rolled):>7}{pts:>7}{tr:>7}"
            f"{f'{held}/{total}':>12}"
        )
    print("\n'points' is what ONE recipe covers and 'traces' what it cost to infer;")
    print("'flags held' counts feature settings whose cross-product roll still held.")

    print("\n================ summary: CBOR bytes ================")
    hdr = (
        f"{'family':<20}{'rolled':>10}{'1 concrete':>12}{'vs 1':>8}"
        f"{'all concrete':>14}{'saved':>8}{'per point':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, _, _, _, cross, _, _ in summary:
        if not cross:
            print(f"{label:<20}{'-':>10}")
            continue
        print(
            f"{label:<20}{_hb(cross.cbor):>10}{_hb(cross.mean):>12}"
            f"{cross.overhead:>+7.1f}%{_hb(cross.concrete):>14}"
            f"{cross.ratio:>7.0f}x{_hb(cross.cbor / cross.points):>11}"
        )
    print(
        "\n'rolled' is the one parametric recipe and 'vs 1' what being parametric costs"
    )
    print(
        "in bytes; 'all concrete' is the mean concrete recipe times the points covered,"
    )
    print(
        "i.e. what shipping them individually costs. The win is the count, not the size."
    )

    print("\n================ summary: roll_nd vs one roll per axis ================")
    hdr = (
        f"{'family':<20}{'per-axis recipes':>18}{'their points':>14}"
        f"{'roll_nd':>10}{'its points':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, _, rolled, ax, cross, _, _ in summary:
        n = len(rolled)
        tot = sum(ax[a].cbor for a in rolled)
        # A one-axis recipe covers a LINE; the base point is shared by all of them.
        line_pts = sum(ax[a].points for a in rolled) - (n - 1 if n else 0)
        got = _hb(cross.cbor) if cross else "-"
        gpts = str(cross.points) if cross else "-"
        print(f"{label:<20}{f'{n} ({_hb(tot)})':>18}{line_pts:>14}{got:>10}{gpts:>12}")
    print("\nSame axes either way: N recipes covering a cross of lines, or one")
    print("covering the volume. The last two columns are the roll_nd payoff.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
