# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The attention tuning space as data: knob axes, codepath knobs,
production stacks, and the knobs held out of the space.

Nothing here walks or builds a spec: the axis helpers and the walk are the
shared :mod:`rocke.dispatch.tuning`, and :mod:`.unified_rules` /
:mod:`.dense_rules` decide legality.
"""

from __future__ import annotations

from typing import Mapping, Tuple

from rocke.dispatch.tuning.axes import (  # noqa: F401  (re-exported for the rules)
    KnobAxis,
    Knobs,
    axis_knob_names,
    choices as _choices_axis,
    flag as _flag,
    gated as _gated,
    knob_requirements,
    sorted_items as _items,
    values as _values,
)


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

# Dense-kernel spec fields the tuning space never varies. Problem fields come
# from the request; variant fields are fixed by the registered dense candidate
# (gfx950: the grid or persistent body, and wide DMA under persistent; gfx942
# registers one candidate and sweeps persistence too); ``waves_per_eu`` is
# walked by its own loop, as on the unified paths. The tile is swept on both.
DENSE_PROBLEM_FIELDS = frozenset(
    {
        "batch",
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "causal",
        "dtype",
        "sliding_window",
        "ragged",
        "varlen",
        "paged",
        "block_size",
        "num_kv_blocks",
        "use_sinks",
        "causal_bottom_right",
    }
)
DENSE_VARIANT_FIELDS: Mapping[str, frozenset] = {
    "gfx950": frozenset({"persistent", "wide_lds_dma"}),
    "gfx942": frozenset(),
}
DENSE_LOOP_FIELDS = frozenset({"waves_per_eu"})
# Knobs with nothing to sweep: the validator accepts a single value, or (gfx942
# ``lazy_rescale``) the body never reads the field and only the name changes.
DENSE_UNTUNABLE_KNOBS: Mapping[str, frozenset] = {
    "gfx950": frozenset({"lds_num_buffers"}),
    "gfx942": frozenset({"lds_num_buffers", "lazy_rescale"}),
}
# Axes whose values are relative to the base spec (request or policy derived),
# so a choice may equal a dataclass default; the base-equal choice is pruned.
DENSE_BASE_RELATIVE_KNOBS = frozenset(
    {"num_persistent", "persistent", "block_m", "block_n"}
)

_DENSE_LDS_PADS = (0, 8, 16, 24, 32)
_DENSE_LAZY_THRESHOLDS = (1.0, 2.0, 4.0)
_DENSE_EXP_PER_PV_STEP = (1, 2, 3, 4)
_DENSE_PV_SCHED_DS_READS = (1, 2, 3, 4)
# Symbolic persistent-CTA counts, resolved against the problem by
# :func:`resolve_dense_num_persistent` (the ``tile_policy="2x"`` pattern).
_NUM_PERSISTENT_MULTIPLES = ("half", "0.75x", "1.25x", "1.5x", "1.75x", "2x")
_GFX950_NUM_PERSISTENT_POLICIES = _NUM_PERSISTENT_MULTIPLES + (
    "gqa_pair",
    "gqa_pair_2phase",
    "work",
)
# gfx942 implements only the qb_major / hkv_major decodes.
_GFX942_NUM_PERSISTENT_POLICIES = _NUM_PERSISTENT_MULTIPLES + ("work",)
_GFX942_LDS_ROW_PADS = (0, 4, 8, 12, 16, 24, 32)
_GFX942_V_ROW_PADS = (0, 8, 16, 32, 64)
_GFX942_BLOCK_M = (32, 64, 128, 256, 512)
_GFX942_BLOCK_N = (32, 64, 128, 256)
# The gfx950 bodies implement 128- and 256-row query tiles (see
# supports_attention_dense); the validator prunes block_n the tile cannot take.
_GFX950_BLOCK_M = (128, 256)
_GFX950_BLOCK_N = (32, 64, 128)


def _num_persistent_axis(policies: Tuple[str, ...]) -> KnobAxis:
    return KnobAxis(
        "num_persistent", ((),) + tuple((("num_persistent", p),) for p in policies)
    )


# One axis per spec field. The PV scheduling knobs lead as enablers: IGLP needs
# the manual fence and sched_group template off, and a manual one needs IGLP
# off, so each can only be reached once the others are decided. Then
# prerequisites-first: the tile sets the query-block count the CTA counts and
# decodes are exact for, persist_decode's gqa_pair modes need an exact
# num_persistent, interleave is read only on the resolved qb_major decode, and
# the fence mask / DS-read count / rescale threshold follow their parents.
_GFX950_DENSE_AXES: Tuple[KnobAxis, ...] = (
    _values("pv_sched_fence", None, (True, False), enabler=True),
    _values("pv_sched_group_template", None, (True, False), enabler=True),
    _values("iglp_mode", None, (-1, 0, 1), enabler=True),
    _choices_axis("block_m", _GFX950_BLOCK_M),
    _choices_axis("block_n", _GFX950_BLOCK_N),
    _num_persistent_axis(_GFX950_NUM_PERSISTENT_POLICIES),
    _values(
        "persist_decode",
        "auto",
        ("qb_major", "hkv_major", "gqa_pair", "gqa_pair_2phase"),
    ),
    _flag("interleave"),
    _values("lazy_rescale", True, (False,)),
    _values("lazy_rescale_threshold", 8.0, _DENSE_LAZY_THRESHOLDS),
    _values("pv_sched_fence_mask", 0, _SCHED_BARRIER_MASKS),
    _values("pv_sched_group_ds_read", 2, _DENSE_PV_SCHED_DS_READS),
    _values("lds_k_row_pad", 8, _DENSE_LDS_PADS),
    _values("lds_v_row_pad", 32, _DENSE_LDS_PADS),
    _values("lds_k_group_pad", 8, _DENSE_LDS_PADS),
    _values("use_exp2_fast", True, (False,)),
    _values("exp_per_pv_step", None, _DENSE_EXP_PER_PV_STEP),
    _values("partial_vmcnt_prefetch", True, (False,)),
    _values("pv_priority", 1, (0, 1, 2, 3)),
    _values("pv_loop_order", None, ("d_major", "k_major")),
    _values("causal_diag_split", True, (False,)),
    _values("o_store_width", 4, (1, 2, 4)),
)

# gfx942 registers one dense candidate, so persistent is a knob here as well
# as the tile (block_m, block_n). The LDS-saving knobs
# lead as enablers: they can bring an otherwise over-budget tile under the LDS
# limit (e.g. D64 block_n=256 fits only with lds_k_group_pad=0). Then
# prerequisites-first: exp2 policy reads persistent, v_row_pad policy reads
# block_n, a non-derived v_row_pad needs the swizzle off, interleave is read
# only on the resolved qb_major decode, and iglp_mode / the PV fence follow iglp.
_GFX942_DENSE_AXES: Tuple[KnobAxis, ...] = (
    _values("use_cfvst", None, (True, False), enabler=True),
    _values("lds_row_pad", 8, _GFX942_LDS_ROW_PADS, enabler=True),
    _values("lds_k_group_pad", 8, _DENSE_LDS_PADS, enabler=True),
    _choices_axis("persistent", (True, False)),
    _choices_axis("block_m", _GFX942_BLOCK_M),
    _choices_axis("block_n", _GFX942_BLOCK_N),
    _num_persistent_axis(_GFX942_NUM_PERSISTENT_POLICIES),
    _values("persist_decode", "auto", ("qb_major", "hkv_major")),
    _flag("interleave"),
    _values("use_v_swizzle", None, (True, False)),
    _values("v_row_pad", None, _GFX942_V_ROW_PADS),
    _values("use_exp2_fast", None, (True, False)),
    _flag("iglp"),
    _values("iglp_mode", 0, (0, 1)),
    _values("pv_sched_fence_mask", None, _SCHED_BARRIER_MASKS),
    _values("pv_priority", 0, (0, 1, 2, 3)),
    _values("pv_loop_order", "d_major", ("d_major", "k_major")),
    _values("o_store_width", 4, (1, 2, 4)),
    _values("causal_diag_split", False, (True,)),
)

# Axes cover every tuning field; ones an arch rejects are pruned at once.
_AXES: Mapping[Tuple[str, str], Tuple[KnobAxis, ...]] = {
    ("gfx950", "2d"): _GFX950_2D_AXES,
    ("gfx942", "2d"): _GFX942_2D_AXES,
    ("gfx950", "3d"): _3D_AXES,
    ("gfx942", "3d"): _3D_AXES,
    ("gfx950", "dense"): _GFX950_DENSE_AXES,
    ("gfx942", "dense"): _GFX942_DENSE_AXES,
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


# Knobs the kernel emits only on one body and ignores elsewhere: softmax
# interleave only on the transposed-32x32 body, the sched_barrier fence only in
# the 16x16 QK loop. Setting them anywhere else re-emits the same kernel.
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
