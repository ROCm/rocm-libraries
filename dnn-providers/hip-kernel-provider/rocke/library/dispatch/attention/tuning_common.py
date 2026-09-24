# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared machinery for explicit gfx942/gfx950 attention tuning candidates."""

from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)
from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import (
    Capability,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
    stable_json_hash,
)

from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_DTYPES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionTuningSpec,
    FAMILY,
    _problem,
    _request_errors,
)


TUNING_ALGORITHM = "unified_tuning"
_SWEEP_WAVES: Tuple[Optional[int], ...] = (None, 1, 2, 3, 4)


@dataclass(frozen=True)
class AttentionGeometryVariant:
    arch: str
    path: str
    codepath: str
    builder_kind: str
    tile_policy: str
    num_warps: int = 1
    block_m_per_warp: int = 16
    num_segments: int = 0
    compile_backend: str = "llvm"

    @property
    def variant_id(self) -> str:
        if self.path == "2d":
            tile = self.tile_policy.replace("x", "xb")
            return (
                f"{self.codepath}_nw{self.num_warps}_mw"
                f"{self.block_m_per_warp}_t{tile}_{self.compile_backend}"
            )
        tile = self.tile_policy.replace("x", "xb")
        return f"{self.codepath}_seg{self.num_segments}_t{tile}"

    @property
    def candidate_name(self) -> str:
        return f"attention_{self.arch}_u{self.path}_{self.variant_id}"

    @property
    def spec_id(self) -> str:
        return f"{self.arch}_u{self.path}_{self.variant_id}"


def _items(values: Mapping[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(values.items()))


Knobs = Tuple[Tuple[str, object], ...]


@dataclass(frozen=True)
class KnobAxis:
    """One tuning decision over kernel-spec fields; ``choices[0]`` is the default.

    ``enabler`` marks knobs that can turn an otherwise-illegal geometry legal
    (gfx942 ``num_warps=8, block_m_per_warp=32`` needs direct-Q + cfvst). They
    must lead the axis order; see :func:`_iter_knob_sets`.
    """

    name: str
    choices: Tuple[Knobs, ...]
    enabler: bool = False


def _flag(name: str, *, enabler: bool = False) -> KnobAxis:
    return KnobAxis(name, ((), ((name, True),)), enabler)


def _values(name: str, default: object, values: Sequence[object]) -> KnobAxis:
    return KnobAxis(name, ((),) + tuple(((name, v),) for v in values if v != default))


def _gated(
    gate: str, sub: Mapping[str, Sequence[object]], *, enabler: bool = False
) -> KnobAxis:
    """``gate`` off, or on with every combination of the sub-knobs it gates.

    The kernel reads the sub-knobs only when ``gate`` is set, so varying them
    while it is off would emit byte-identical duplicate kernels.
    """
    combos: list[Knobs] = [((gate, True),)]
    for name, values in sub.items():
        combos = [c + ((name, v),) for c in combos for v in values]
    return KnobAxis(gate, ((),) + tuple(combos), enabler)


# AMDGPU ``sched_barrier`` ABI: 0 is a full fence, and each bit lets one
# instruction class cross it (ALU, VALU, SALU, MFMA, VMEM, VMEM rd, VMEM wr,
# DS, DS rd, DS wr, transcendental). 0x108 is MFMA + DS-read.
_SCHED_BARRIER_MASKS = (0x0,) + tuple(1 << bit for bit in range(11)) + (0x108,)
# The kernel clamps the group count to the per-tile MFMA count.
_SOFTMAX_INTERLEAVE_GROUPS = (1, 2, 4, 8, 16)
# Pad width in halves; a multiple of 8 keeps b128 LDS access aligned.
_KQ_LDS_PAD_HALVES = (8, 16, 24, 32)
# Multiples of the 32x32x8 QK k-step; the kernel rejects widths that do not
# split head_size into at least two slices.
_K_SLICE_HD = (8, 16, 32, 64)

# The LDS-saving knobs lead as enablers: once they are decided, every later
# axis only grows the LDS footprint, so the LDS budget prunes like a validator.
_GFX950_2D_AXES: Tuple[KnobAxis, ...] = (
    _flag("use_fp8_mfma_qk", enabler=True),
    _flag("use_fp8_mfma_pv", enabler=True),
    _flag("use_register_pv", enabler=True),
    _flag("use_q_direct_reg", enabler=True),
    _flag("use_k_single_buffer", enabler=True),
    _flag("use_transposed_scalar_state"),
    _flag("use_transposed_invariant_hoist"),
    _flag("use_transposed_mask_once"),
    _flag("use_transposed_half_local_pv"),
    _flag("use_mfma32_skip_legacy_qreg"),
    _flag("use_transposed_mask_limit"),
    _flag("use_mask_phase_split"),
    _flag("use_agpr_alloc_zero"),
    _flag("use_grouped_kv2_softmax"),
    _flag("use_fast_paged_kv_desc"),
    _flag("use_early_v_schedule"),
    _flag("use_v_double_buffer"),
    _flag("use_staggered_iter_wait"),
    _values("kv_ring_depth", 2, (2, 3)),
    _flag("use_q_reread"),
    _gated("use_kq_lds_pad", {"kq_lds_pad_halves": _KQ_LDS_PAD_HALVES}),
    _gated("use_sched_barrier", {"sched_barrier_mask": _SCHED_BARRIER_MASKS}),
    KnobAxis(
        "use_softmax_mfma_interleave",
        ((),)
        + tuple(
            (("use_softmax_mfma_interleave", True), ("softmax_interleave_mode", m))
            for m in (0, 1)
        )
        + tuple(
            (
                ("use_softmax_mfma_interleave", True),
                ("softmax_interleave_mode", 2),
                ("softmax_interleave_groups", g),
            )
            for g in _SOFTMAX_INTERLEAVE_GROUPS
        ),
    ),
)

_GFX942_2D_AXES: Tuple[KnobAxis, ...] = (
    _flag("use_q_direct_global", enabler=True),
    _gated(
        "use_conflict_free_v_store",
        {
            "use_conflict_free_v_store_split": (True, False),
            "use_conflict_free_v_ck_vlds": (True, False),
        },
        enabler=True,
    ),
    _flag("use_conflict_free_v"),
    _flag("use_mfma_32x32"),
    _flag("use_fp8_mfma_qk"),
    _flag("use_fp8_mfma_pv"),
    _flag("use_register_pv"),
    _flag("use_transposed_scalar_state"),
    _flag("use_transposed_invariant_hoist"),
    _flag("use_transposed_mask_once"),
    _flag("use_transposed_half_local_pv"),
    _flag("use_mfma32_skip_legacy_qreg"),
    _flag("use_transposed_mask_limit"),
    _flag("use_grouped_kv2_softmax"),
    _flag("use_fast_paged_kv_desc"),
    _flag("use_early_v_schedule"),
    _flag("use_agpr_alloc_zero"),
    _flag("use_k_single_buffer"),
    _gated("use_k_sliced_ring", {"ring_depth": (2, 3), "k_slice_hd": _K_SLICE_HD}),
    _flag("use_k_sliced_ldsseq"),
    _flag("use_iglp_opt"),
    _flag("use_qk_pv_sched_group_barrier"),
    _flag("use_v_hbm_direct"),
    _flag("use_global_load_lds_k"),
    _values("kv_cache_policy", "stream", ("stream", "all", "global", "nt")),
    _flag("use_q_major_grid"),
    _flag("use_causal_mask_phase_split"),
)

_3D_AXES: Tuple[KnobAxis, ...] = (
    _flag("use_invariant_hoist"),
    _flag("use_wide_kv_load"),
)

# Axes cover every tuning field; ones an arch rejects are pruned at once.
_AXES: Mapping[Tuple[str, str], Tuple[KnobAxis, ...]] = {
    ("gfx950", "2d"): _GFX950_2D_AXES,
    ("gfx942", "2d"): _GFX942_2D_AXES,
    ("gfx950", "3d"): _3D_AXES,
    ("gfx942", "3d"): _3D_AXES,
}

# Knobs fixed by the geometry variant's codepath rather than enumerated.
_CODEPATH_KNOBS: Mapping[Tuple[str, str], Mapping[str, object]] = {
    ("gfx950", "wide32"): {"use_mfma_32x32": True},
    ("gfx950", "transposed32"): {
        "use_mfma_32x32": True,
        "use_transposed_qk_32x32": True,
    },
    ("gfx942", "wide32x8"): {"use_mfma_32x32x8": True},
    ("gfx942", "transposed_x8"): {
        "use_mfma_32x32x8": True,
        "use_transposed_qk_32x32": True,
    },
}

# Kernel knobs held out of the feasible space until an fp32-reference sweep
# passes: gfx942 transposed-x8 K-HBM-direct prefill produced wrong outputs.
KNOWN_WRONG_KNOBS: Mapping[str, frozenset] = {
    "gfx942": frozenset({"use_k_hbm_direct"}),
    "gfx950": frozenset(),
}

# Knobs the kernels document as measured dead ends. They are not
# KNOWN_WRONG_KNOBS: outputs are correct, so the full sweep still samples
# them. Production stacks simply do not turn them on.
#   gfx950 use_q_reread       -- "[TESTED: dead end, kept gated]": slower, no
#                                occupancy gain.
#   gfx942 use_conflict_free_v -- the synchronous gather store is several times
#                                slower; use_conflict_free_v_store supersedes it.
DEAD_END_KNOBS: Mapping[str, frozenset] = {
    "gfx950": frozenset({"use_q_reread"}),
    "gfx942": frozenset({"use_conflict_free_v"}),
}

SWEEP_LEVELS: Tuple[str, ...] = ("production", "full")
_SWEEP_LEVEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "attention_sweep_level", default="production"
)


def configure_sweep(level: str, tuning_sample: int = 0) -> int:
    """Select the sweep level and return the sample count for ``iter_combos``.

    ``production`` walks the curated stacks and ignores ``tuning_sample``.
    ``full`` samples ``tuning_sample`` specs per candidate (0 walks the full
    stream, which is millions of specs on the transposed paths).
    """
    if level not in SWEEP_LEVELS:
        raise ValueError(f"sweep level must be one of {SWEEP_LEVELS}, got {level!r}")
    _SWEEP_LEVEL.set(level)
    if level == "production":
        return 0
    return max(0, int(tuning_sample))


# Production sweep: the hand-curated stacks per (arch, codepath), walked
# exhaustively. Codepath base knobs (_CODEPATH_KNOBS) are applied on top.
_R4_S1 = {
    "use_transposed_scalar_state": True,
    "use_transposed_invariant_hoist": True,
    "use_transposed_mask_once": True,
}
_R4_MLIM = {**_R4_S1, "use_transposed_mask_limit": True}
_R4_HLPV = {
    **_R4_MLIM,
    "use_transposed_half_local_pv": True,
    "use_mfma32_skip_legacy_qreg": True,
}
_VDBUF = {"use_v_double_buffer": True}
_VDBUF_STGW = {**_VDBUF, "use_staggered_iter_wait": True}
_PROD_SCHED_BARRIER_MASKS = (0x0, 0x008, 0x108)

_PRODUCTION_STACKS: Mapping[Tuple[str, str], Tuple[Tuple[str, Mapping], ...]] = {
    ("gfx950", "narrow"): (
        ("baseline", {}),
        ("regpv", {"use_register_pv": True}),
        ("fp8qk", {"use_fp8_mfma_qk": True}),
        ("fp8pv", {"use_fp8_mfma_pv": True}),
        ("fp8both", {"use_fp8_mfma_qk": True, "use_fp8_mfma_pv": True}),
        ("early_v", {"use_early_v_schedule": True}),
        ("vdbuf", _VDBUF),
        ("vdbuf_stgw", _VDBUF_STGW),
    )
    + tuple(
        (
            f"{prefix}schedb_{mask:#x}",
            {**extra, "use_sched_barrier": True, "sched_barrier_mask": mask},
        )
        for mask in _PROD_SCHED_BARRIER_MASKS
        for prefix, extra in (("", {}), ("vdbuf_", _VDBUF))
    ),
    ("gfx950", "wide32"): (
        ("baseline", {}),
        ("early_v", {"use_early_v_schedule": True}),
        ("vdbuf", _VDBUF),
        ("vdbuf_stgw", _VDBUF_STGW),
    ),
    ("gfx950", "transposed32"): (
        ("baseline", {}),
        ("scalar", {"use_transposed_scalar_state": True}),
        ("r4_s1", _R4_S1),
        ("r4_s1_mlim", _R4_MLIM),
        ("r4_hlpv", _R4_HLPV),
        ("early_v", {"use_early_v_schedule": True}),
        ("vdbuf", _VDBUF),
        ("vdbuf_stgw", _VDBUF_STGW),
        ("ksb", {"use_k_single_buffer": True}),
        ("ksb_qdreg", {"use_k_single_buffer": True, "use_q_direct_reg": True}),
        ("ring3", {"kv_ring_depth": 3}),
        ("qdreg", {"use_q_direct_reg": True}),
        ("gkv2", {"use_grouped_kv2_softmax": True}),
        ("fastkv", {"use_fast_paged_kv_desc": True}),
        ("r4_mlim_vdbuf", {**_R4_MLIM, **_VDBUF}),
        ("r4_mlim_phase", {**_R4_MLIM, "use_mask_phase_split": True}),
        ("r4_hlpv_agpr0", {**_R4_HLPV, "use_agpr_alloc_zero": True}),
    ),
    ("gfx942", "narrow"): (
        ("baseline", {}),
        ("regpv", {"use_register_pv": True}),
        ("early_v", {"use_early_v_schedule": True}),
        ("iglp", {"use_iglp_opt": True}),
        ("qmajor", {"use_q_major_grid": True}),
        ("gldsk", {"use_global_load_lds_k": True}),
        ("fastkv", {"use_fast_paged_kv_desc": True}),
        ("vhbm", {"use_v_hbm_direct": True}),
    ),
    ("gfx942", "wide32x8"): (
        ("baseline", {}),
        ("iglp", {"use_iglp_opt": True}),
        ("qmajor", {"use_q_major_grid": True}),
    ),
    ("gfx942", "gfx942_4warp"): (("baseline", {}),),
    ("gfx942", "transposed_x8"): (
        ("baseline", {}),
        ("scalar", {"use_transposed_scalar_state": True}),
        ("r4_s1", _R4_S1),
        ("r4_s1_mlim", _R4_MLIM),
        ("cfvst", {"use_conflict_free_v_store": True}),
        (
            "cfvst_nosplit",
            {
                "use_conflict_free_v_store": True,
                "use_conflict_free_v_store_split": False,
            },
        ),
        ("ksb", {"use_k_single_buffer": True}),
        ("qdglob", {"use_q_direct_global": True}),
        ("vhbm", {"use_v_hbm_direct": True}),
        ("gldsk", {"use_global_load_lds_k": True}),
        ("qmajor", {"use_q_major_grid": True}),
        ("cphase", {"use_causal_mask_phase_split": True}),
        ("agpr0", {"use_agpr_alloc_zero": True}),
        ("iglp", {"use_iglp_opt": True}),
        ("schedg", {"use_qk_pv_sched_group_barrier": True}),
    )
    + tuple(
        (
            f"ring_d{depth}_w32{'_ldsseq' if seq else ''}",
            {
                "use_conflict_free_v_store": True,
                "use_k_sliced_ring": True,
                "ring_depth": depth,
                "k_slice_hd": 32,
                **({"use_k_sliced_ldsseq": True} if seq else {}),
            },
        )
        for depth, seq in ((2, False), (3, False), (3, True))
    )
    + (("kvcp_nt", {"kv_cache_policy": "nt"}),),
    ("gfx942", "splitkv"): (
        ("baseline", {}),
        ("hoist", {"use_invariant_hoist": True}),
        ("widekv", {"use_wide_kv_load": True}),
        ("hoist_widekv", {"use_invariant_hoist": True, "use_wide_kv_load": True}),
    ),
    ("gfx950", "splitkv"): (("baseline", {}),),
}
# Production micro-axes layered on some stacks: padded K for the single-K
# unaliased-Q stack, the sched_group_barrier interleave on three transposed
# stacks, and a reduced 2D waves-per-EU set.
_PROD_PAD_STACKS = frozenset({"ksb_qdreg"})
_PROD_PAD = {"use_kq_lds_pad": True, "kq_lds_pad_halves": 16}
_PROD_INTERLEAVE_STACKS = frozenset({"baseline", "r4_hlpv", "vdbuf"})
_PROD_INTERLEAVE = {
    "use_softmax_mfma_interleave": True,
    "softmax_interleave_mode": 2,
    "softmax_interleave_groups": 4,
}
_PROD_WAVES_2D: Tuple[Optional[int], ...] = (None, 2, 4)

# Kernel-documented constraints its ``__post_init__`` does not enforce.
# Softmax interleave and sched_barrier steer the scheduler in opposite
# directions; interleave is only emitted on the transposed-32x32 body and the
# sched_barrier fence only in the 16x16 QK loop, so elsewhere they are inert.
_EXCLUSIVE_KNOBS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "gfx950": (("use_softmax_mfma_interleave", "use_sched_barrier"),),
}
_TRANSPOSED_ONLY_KNOBS: Mapping[str, Tuple[str, ...]] = {
    "gfx950": ("use_softmax_mfma_interleave",),
}
_NARROW_ONLY_KNOBS: Mapping[str, Tuple[str, ...]] = {
    "gfx950": ("use_sched_barrier",),
}


def tuning_axes(arch: str, path: str) -> Tuple[KnobAxis, ...]:
    try:
        return _AXES[(arch, path)]
    except KeyError:
        raise ValueError(f"no explicit {path.upper()} tuning axes for arch {arch!r}")


def _policy_conflict(arch: str, knobs: Mapping[str, object]) -> Optional[str]:
    for a, b in _EXCLUSIVE_KNOBS.get(arch, ()):
        if knobs.get(a) and knobs.get(b):
            return f"{a} and {b} are mutually exclusive"
    if not knobs.get("use_transposed_qk_32x32"):
        for name in _TRANSPOSED_ONLY_KNOBS.get(arch, ()):
            if knobs.get(name):
                return f"{name} is only emitted on the transposed-32x32 path"
    if knobs.get("use_mfma_32x32"):
        for name in _NARROW_ONLY_KNOBS.get(arch, ()):
            if knobs.get(name):
                return f"{name} is only emitted in the 16x16 QK loop"
    return None


def _iter_knob_sets(
    axes: Tuple[KnobAxis, ...],
    base: Mapping[str, object],
    is_valid,
) -> Iterable[dict]:
    """Every legal assignment over ``axes`` on top of ``base``, depth first.

    ``is_valid`` sees the assignment so far with undecided axes at their
    defaults. Every "requires" relation in the kernel validators points at an
    earlier axis, so a failing prefix cannot be repaired later and its subtree
    is pruned. Only while leading ``enabler`` axes are undecided is an invalid
    prefix kept, because those knobs can make the geometry itself legal.
    """
    n_enablers = next((i for i, a in enumerate(axes) if not a.enabler), len(axes))
    if any(a.enabler for a in axes[n_enablers:]):
        raise ValueError("enabler axes must lead the axis order")

    def walk(i: int, knobs: dict, valid: bool):
        if i == len(axes):
            if valid:
                yield knobs
            return
        for choice in axes[i].choices:
            if choice:
                child = dict(knobs)
                child.update(choice)
                child_valid = is_valid(child)
            else:
                child, child_valid = knobs, valid
            if child_valid or i < n_enablers:
                yield from walk(i + 1, child, child_valid)

    root = dict(base)
    yield from walk(0, root, is_valid(root))


def _k_schedule_bufs(spec) -> int:
    if spec.use_k_single_buffer:
        return 1
    depth = int(spec.kv_ring_depth)
    return depth if depth > 2 else 2


def _kq_pad_eligible(spec) -> Tuple[bool, str]:
    """Shared KQ-pad decision for the LDS model and the support predicate."""
    if not spec.use_kq_lds_pad:
        return False, "KQ LDS pad is off"
    head = int(spec.head_size)
    tile = int(spec.tile_size_eff)
    pad = int(spec.kq_lds_pad_halves)
    fp8_qk = spec.kv_storage_dtype == "fp8e4m3" and bool(spec.use_fp8_mfma_qk)
    if fp8_qk:
        return False, "KQ LDS pad does not support native-FP8 K LDS"
    slab_rows = (512 // head) if head and 512 % head == 0 else 0
    if not slab_rows or tile % slab_rows != 0 or pad % 8 != 0:
        return (
            False,
            "KQ LDS pad requires an aligned slab layout "
            f"(head_size={head}, tile_size={tile}, pad={pad})",
        )
    if _k_schedule_bufs(spec) != 1:
        return False, "KQ LDS pad requires a single-K schedule"
    k_elem_bytes = 1 if fp8_qk else 2
    q_bytes = int(spec.block_m) * head * 2
    q_aliases_k = bool(
        not spec.use_q_reread
        and not spec.use_q_direct_reg
        and q_bytes <= 2 * tile * head * k_elem_bytes
    )
    if q_aliases_k:
        return False, "padded K LDS does not support aliased Q"
    return True, "ok"


def _gfx950_tuning_lds_bytes(spec) -> int:
    """Dispatcher-side static LDS model for one explicit gfx950 tuning spec."""
    tile = int(spec.tile_size_eff)
    head = int(spec.head_size)
    block_m = int(spec.block_m)
    kv_fp8 = spec.kv_storage_dtype == "fp8e4m3"
    fp8_qk = kv_fp8 and bool(spec.use_fp8_mfma_qk)
    fp8_pv = kv_fp8 and bool(spec.use_fp8_mfma_pv)
    k_elem_bytes = 1 if fp8_qk else 2
    v_elem_bytes = 1 if fp8_pv else 2
    k_bufs = _k_schedule_bufs(spec)
    v_bufs = 2 if spec.use_v_double_buffer else 1

    pad = int(spec.kq_lds_pad_halves) if spec.use_kq_lds_pad else 0
    slab_rows = (512 // head) if pad and 512 % head == 0 else 0
    pad_active, _pad_why = _kq_pad_eligible(spec)
    if pad_active:
        k_bytes = k_bufs * (tile // slab_rows) * (slab_rows * head + pad) * k_elem_bytes
    else:
        k_bytes = k_bufs * tile * head * k_elem_bytes
    v_bytes = v_bufs * tile * head * v_elem_bytes

    transposed_register_p = bool(spec.use_mfma_32x32 and spec.use_transposed_qk_32x32)
    p_bytes = (
        0
        if spec.use_register_pv or transposed_register_p
        else block_m * (tile + (16 if fp8_pv else 8)) * v_elem_bytes
    )

    q_bytes = block_m * head * 2
    q_aliases_k = bool(not fp8_qk and q_bytes <= 2 * tile * head * k_elem_bytes)
    if spec.use_q_reread or spec.use_q_direct_reg:
        q_aliases_k = False
    q_lds_bytes = 0 if spec.use_q_direct_reg or q_aliases_k else q_bytes

    out_stripe_cols = 32 if head <= 64 else head
    acc_bytes = block_m * out_stripe_cols * 2
    fp8_staging_bytes = 3 * tile * head if kv_fp8 and spec.use_fp8_mfma_qk else 0
    return k_bytes + v_bytes + p_bytes + q_lds_bytes + acc_bytes + fp8_staging_bytes


def _supports_tuning_spec(
    variant: AttentionGeometryVariant, kernel_spec
) -> Tuple[bool, str]:
    """Residual per-spec support that cannot be represented by Capability."""
    if variant.arch != "gfx950" or variant.path != "2d":
        return True, "supported"

    if kernel_spec.use_kq_lds_pad:
        pad_ok, pad_why = _kq_pad_eligible(kernel_spec)
        if not pad_ok:
            return False, pad_why

    from rocke.core.arch import ArchTarget

    capacity = ArchTarget.from_gfx(variant.arch).lds_capacity_bytes
    lds_bytes = _gfx950_tuning_lds_bytes(kernel_spec)
    if lds_bytes > capacity:
        return (
            False,
            f"estimated LDS {lds_bytes} B exceeds the {variant.arch} "
            f"{capacity} B LDS budget; hipcc/comgr codegen would fail",
        )
    return True, "supported"


def _canonical_payload(
    *,
    arch: str,
    path: str,
    builder_kind: str,
    compile_backend: str,
    fp8_fnuz: bool,
    kernel_spec,
    reduce_spec,
) -> dict:
    def _as_payload(value):
        if value is None:
            return None
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return repr(value)

    return {
        "abi": ATTENTION_ABI_VERSION,
        "arch": arch,
        "path": path,
        "builder_kind": builder_kind,
        "compile_backend": compile_backend,
        "fp8_fnuz": bool(fp8_fnuz),
        "kernel_spec": _as_payload(kernel_spec),
        "reduce_spec": _as_payload(reduce_spec),
    }


def _tuning_id_prefix(variant: AttentionGeometryVariant, kernel_spec) -> str:
    """Readable id stem stored on the spec. The hash suffix is display-only."""
    wpe = getattr(kernel_spec, "waves_per_eu", None)
    return f"{variant.variant_id}_wpe{wpe if wpe is not None else 'none'}"


def _make_tuning_id(
    variant: AttentionGeometryVariant,
    kernel_spec,
    reduce_spec=None,
    *,
    fp8_fnuz: bool = False,
) -> str:
    digest = stable_json_hash(
        _canonical_payload(
            arch=variant.arch,
            path=variant.path,
            builder_kind=variant.builder_kind,
            compile_backend=variant.compile_backend,
            fp8_fnuz=fp8_fnuz,
            kernel_spec=kernel_spec,
            reduce_spec=reduce_spec,
        ),
        n=16,
    )
    return f"{_tuning_id_prefix(variant, kernel_spec)}@{digest}"


def retarget_tuning_spec(
    spec: AttentionTuningSpec, *, num_kv_blocks: int
) -> AttentionTuningSpec:
    """Refresh i32/i64 paged addressing once the physical cache is known."""
    count = int(num_kv_blocks)
    if count < 0:
        raise ValueError("num_kv_blocks must be non-negative")
    kernel_spec = spec.kernel_spec
    if not hasattr(kernel_spec, "use_i64_kv_addr"):
        return replace(spec, num_kv_blocks=count)

    elem_bytes = 1 if kernel_spec.kv_storage_dtype == "fp8e4m3" else 2
    block_stride = (
        int(kernel_spec.block_size)
        * int(kernel_spec.num_kv_heads)
        * int(kernel_spec.head_size)
        * elem_bytes
    )
    use_i64 = count > 0 and count * block_stride > 0x8000_0000
    refreshed_kernel = replace(kernel_spec, use_i64_kv_addr=use_i64)
    if refreshed_kernel == kernel_spec and int(spec.num_kv_blocks) == count:
        return spec

    prefix = spec.tuning_id_prefix
    if not prefix:
        raise ValueError(
            f"tuning spec {spec.tuning_id!r} has no tuning_id_prefix; "
            "the display id is not parsed back into identity"
        )
    digest = stable_json_hash(
        _canonical_payload(
            arch=spec.arch,
            path=spec.path,
            builder_kind=spec.builder_kind,
            compile_backend=spec.compile_backend,
            fp8_fnuz=spec.fp8_fnuz,
            kernel_spec=refreshed_kernel,
            reduce_spec=spec.reduce_spec,
        ),
        n=16,
    )
    return replace(
        spec,
        tuning_id=f"{prefix}@{digest}",
        kernel_spec=refreshed_kernel,
        num_kv_blocks=count,
    )


def _explicit_3d_specs(problem, variant, knobs, wpe):
    config = ExplicitAttention3DConfig(
        num_segments=variant.num_segments,
        tile_policy=variant.tile_policy,
        waves_per_eu=wpe,
        knobs=_items(knobs),
    )
    return make_explicit_attention_3d_specs(problem, config, arch=variant.arch)


def _explicit_2d_spec(problem, variant, knobs, wpe):
    config = ExplicitAttention2DConfig(
        num_warps=variant.num_warps,
        block_m_per_warp=variant.block_m_per_warp,
        tile_policy=variant.tile_policy,
        waves_per_eu=wpe,
        compile_backend=variant.compile_backend,
        builder_kind=variant.builder_kind,
        knobs=_items(knobs),
    )
    return make_explicit_attention_2d_spec(problem, config, arch=variant.arch)


def _knob_space(problem, variant: AttentionGeometryVariant):
    """``(axes, base, is_valid)`` for one geometry variant and problem."""
    axes = tuning_axes(variant.arch, variant.path)
    if variant.builder_kind == "gfx942_4warp_gqa":
        axes = ()  # the 4-warp GQA builder reads no tuning knobs
    base = _CODEPATH_KNOBS.get((variant.arch, variant.codepath), {})
    build = _explicit_3d_specs if variant.path == "3d" else _explicit_2d_spec

    def is_valid(knobs) -> bool:
        if _policy_conflict(variant.arch, knobs):
            return False
        try:
            spec = build(problem, variant, knobs, None)
        except (ValueError, NotImplementedError):
            return False
        if variant.path == "2d":
            return _supports_tuning_spec(variant, spec)[0]
        return True

    return axes, base, is_valid


def _tuning_spec(problem, variant, knobs, wpe) -> Optional[AttentionTuningSpec]:
    """Wrap one legal knob set at one ``waves_per_eu``; ``None`` if unsupported."""
    try:
        return _build_tuning_spec(problem, variant, knobs, wpe)
    except (ValueError, TypeError, NotImplementedError):
        return None


def _build_tuning_spec(problem, variant, knobs, wpe) -> Optional[AttentionTuningSpec]:
    fp8_fnuz = bool(problem.fp8_fnuz)
    if variant.path == "3d":
        segment, reduce = _explicit_3d_specs(problem, variant, knobs, wpe)
        return AttentionTuningSpec(
            path="3d",
            arch=variant.arch,
            builder_kind="tiled_3d",
            compile_backend="llvm",
            candidate_name=variant.candidate_name,
            tuning_id=_make_tuning_id(variant, segment, reduce, fp8_fnuz=fp8_fnuz),
            tuning_id_prefix=_tuning_id_prefix(variant, segment),
            kernel_spec=segment,
            fp8_fnuz=fp8_fnuz,
            num_kv_blocks=int(problem.num_kv_blocks),
            reduce_spec=reduce,
        )
    kernel_spec = _explicit_2d_spec(problem, variant, knobs, wpe)
    if not _supports_tuning_spec(variant, kernel_spec)[0]:
        return None
    return AttentionTuningSpec(
        path="2d",
        arch=variant.arch,
        builder_kind=variant.builder_kind,
        compile_backend=variant.compile_backend,
        candidate_name=variant.candidate_name,
        tuning_id=_make_tuning_id(variant, kernel_spec, fp8_fnuz=fp8_fnuz),
        tuning_id_prefix=_tuning_id_prefix(variant, kernel_spec),
        kernel_spec=kernel_spec,
        fp8_fnuz=fp8_fnuz,
        num_kv_blocks=int(problem.num_kv_blocks),
    )


def _explicit_configs(
    problem,
    variant: AttentionGeometryVariant,
) -> Iterable[AttentionTuningSpec]:
    axes, base, is_valid = _knob_space(problem, variant)
    for knobs in _iter_knob_sets(axes, base, is_valid):
        for wpe in _SWEEP_WAVES:
            spec = _tuning_spec(problem, variant, knobs, wpe)
            if spec is not None:
                yield spec


def _production_knob_sets(variant: AttentionGeometryVariant):
    """Curated stacks for one geometry, plus their pad / interleave variants.

    A stack that turns on a dead-end or known-wrong knob is a data error: the
    production level exists to keep those out.
    """
    stacks = _PRODUCTION_STACKS.get((variant.arch, variant.codepath))
    if not stacks:
        return
    base = dict(_CODEPATH_KNOBS.get((variant.arch, variant.codepath), {}))
    banned = DEAD_END_KNOBS.get(variant.arch, frozenset()) | KNOWN_WRONG_KNOBS.get(
        variant.arch, frozenset()
    )
    for name, overrides in stacks:
        knobs = dict(base)
        knobs.update(overrides)
        if any(knobs.get(knob) for knob in banned):
            raise ValueError(
                f"production stack {variant.arch}/{variant.codepath}/{name} "
                f"sets a dead-end knob"
            )
        yield knobs
        if (
            name in _PROD_PAD_STACKS
            and variant.arch == "gfx950"
            and variant.codepath != "narrow"
        ):
            yield {**knobs, **_PROD_PAD}
        if (
            name in _PROD_INTERLEAVE_STACKS
            and variant.arch == "gfx950"
            and variant.codepath == "transposed32"
        ):
            yield {**knobs, **_PROD_INTERLEAVE}


def _production_configs(problem, variant: AttentionGeometryVariant):
    waves = _SWEEP_WAVES if variant.path == "3d" else _PROD_WAVES_2D
    seen: set[str] = set()
    for knobs in _production_knob_sets(variant):
        for waves_per_eu in waves:
            spec = _tuning_spec(problem, variant, knobs, waves_per_eu)
            if spec is not None and spec.tuning_id not in seen:
                seen.add(spec.tuning_id)
                yield spec


def _random_knob_set(axes, base, is_valid, rng) -> Optional[dict]:
    """One random walk down the pruned axis tree; ``None`` on a dead end.

    At every axis the walk picks uniformly among the choices the validator
    accepts, so every legal assignment is reachable. It is not uniform over
    the whole legal set, which would need the subtree sizes the full walk is
    too slow to count; it does sample each knob value at a useful rate.
    """
    n_enablers = next((i for i, a in enumerate(axes) if not a.enabler), len(axes))
    knobs, valid = dict(base), is_valid(base)
    for i, axis in enumerate(axes):
        for choice in rng.sample(axis.choices, len(axis.choices)):
            if choice:
                child = dict(knobs)
                child.update(choice)
                child_valid = is_valid(child)
            else:
                child, child_valid = knobs, valid
            if child_valid or i < n_enablers:
                knobs, valid = child, child_valid
                break
    return knobs if valid else None


def iter_tuning_specs(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    level: str = "production",
):
    """Specs for one geometry at ``level``.

    ``production`` walks the curated stacks exhaustively (no dead-end knobs).
    ``full`` walks every kernel knob; consume it through
    :func:`sample_tuning_specs` unless the space is known to be small.
    """
    if level not in SWEEP_LEVELS:
        raise ValueError(f"sweep level must be one of {SWEEP_LEVELS}, got {level!r}")
    problem = _problem(req)
    if level == "production":
        return _production_configs(problem, variant)
    return _explicit_configs(problem, variant)


def sample_tuning_specs(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    n: int,
    seed: int = 0,
) -> Iterable[AttentionTuningSpec]:
    """Up to ``n`` distinct random legal specs from the full knob space.

    This is the non-production sweep: every kernel knob except
    ``KNOWN_WRONG_KNOBS``, dead ends included. Stops early once ``20 * n``
    draws have been tried, which is what a space smaller than ``n`` looks like.
    """
    import random

    problem = _problem(req)
    axes, base, is_valid = _knob_space(problem, variant)
    rng = random.Random(f"{int(seed)}:{variant.candidate_name}")
    seen: set[str] = set()
    for _ in range(20 * int(n)):
        if len(seen) >= n:
            return
        knobs = _random_knob_set(axes, base, is_valid, rng)
        if knobs is None:
            continue
        spec = _tuning_spec(problem, variant, knobs, rng.choice(_SWEEP_WAVES))
        if spec is not None and spec.tuning_id not in seen:
            seen.add(spec.tuning_id)
            yield spec


def _sets_dead_end(spec: AttentionTuningSpec) -> bool:
    banned = DEAD_END_KNOBS.get(spec.arch, frozenset())
    return any(getattr(spec.kernel_spec, knob, False) for knob in banned)


def _find_tuning_spec(
    req: AttentionRequest,
    variant: AttentionGeometryVariant,
    tuning_id: str,
) -> Optional[AttentionTuningSpec]:
    """Resolve one spec, production stacks first.

    ``auto`` returns the first production spec, then the first full-space spec
    that does not turn a dead-end knob on. A pinned id is matched in the
    production set and, if absent, in the full space (dead ends included, so a
    sampled full-sweep id still replays).
    """
    wanted = tuning_id.strip()
    problem = _problem(req)
    for spec in _production_configs(problem, variant):
        if wanted in ("auto", "") or spec.tuning_id == wanted:
            return spec
    full = _explicit_configs(problem, variant)
    if wanted in ("auto", ""):
        return next((spec for spec in full if not _sets_dead_end(spec)), None)
    for spec in full:
        if spec.tuning_id == wanted:
            return spec
    return None


def _opted_in(req: AttentionRequest, variant: AttentionGeometryVariant) -> bool:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", TUNING_ALGORITHM):
        return False
    if spec_id not in ("auto", variant.spec_id):
        return False
    return algorithm == TUNING_ALGORITHM or spec_id == variant.spec_id


def make_tuning_candidate(
    variant: AttentionGeometryVariant,
) -> KernelCandidate:
    def support(req: OperatorRequest):
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if not _opted_in(req, variant):
            return False, "explicit unified tuning candidate is opt-in"
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem, arch=variant.arch)
        if not ok:
            return False, why
        tuning_id = req.attention_tuning_id.strip()
        if _find_tuning_spec(req, variant, tuning_id) is None:
            if tuning_id != "auto":
                return False, f"unknown or invalid attention_tuning_id {tuning_id!r}"
            return False, "no dependency-valid tuning spec for this request"
        return True, "ok"

    def sweep(req: OperatorRequest):
        ok, _ = candidate.admits(req)
        if not ok:
            return ()
        assert isinstance(req, AttentionRequest)
        return iter_tuning_specs(req, variant, level=_SWEEP_LEVEL.get())

    def sample(req: OperatorRequest, n: int, seed: int):
        ok, _ = candidate.admits(req)
        if not ok:
            return ()
        assert isinstance(req, AttentionRequest)
        return sample_tuning_specs(req, variant, n, seed)

    def select(req: OperatorRequest):
        assert isinstance(req, AttentionRequest)
        tuning_id = req.attention_tuning_id.strip()
        spec = _find_tuning_spec(req, variant, tuning_id)
        if spec is None:
            raise ValueError(f"{variant.candidate_name} does not support request")
        return spec

    def build(spec, arch):
        from .tuning_specs import validate_explicit_fp8_encoding

        validate_explicit_fp8_encoding(
            arch=arch,
            use_fp8=spec.kernel_spec.kv_storage_dtype == "fp8e4m3",
            fp8_fnuz=spec.fp8_fnuz,
        )
        return spec.build(arch)

    def signature(spec):
        from kernels.common.attention_unified import (
            _3d_signature,
            _attn_signature,
        )

        if spec.path == "3d":
            return _3d_signature(
                spec.kernel_spec.dtype,
                kv_dtype=spec.kernel_spec.kv_storage_dtype,
            )
        return _attn_signature(
            spec.kernel_spec.dtype,
            include_bt_stride=True,
            include_qq_bias_stride=True,
            kv_dtype=spec.kernel_spec.kv_storage_dtype,
        )

    def grid(spec, req):
        assert isinstance(req, AttentionRequest)
        return spec.launch_grid(_problem(req))

    def block(spec):
        return spec.launch_block()

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_tuning_attention_torch

        payload = dict(tensors)
        if "problem" not in payload:
            payload["problem"] = _problem(request)
        return bind_tuning_attention_torch(
            request,
            spec,
            payload,
            grid=grid(spec, request),
            block=block(spec),
            **kwargs,
        )

    candidate = KernelCandidate(
        name=variant.candidate_name,
        family=FAMILY,
        algorithm=TUNING_ALGORITHM,
        spec_id=variant.spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=30,
        capability=Capability(
            arches=(variant.arch,),
            dtypes=UNIFIED_DTYPES,
            shapes=(
                ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            supports_features=ATTENTION_FEATURES,
        ),
        _supports=support,
        select_spec=select,
        signature=signature,
        grid=grid,
        block=block,
        sweep_space=sweep,
        sample_space=sample,
        build=build,
        bind_torch=bind_torch,
        opt_in=True,
    )
    return candidate
