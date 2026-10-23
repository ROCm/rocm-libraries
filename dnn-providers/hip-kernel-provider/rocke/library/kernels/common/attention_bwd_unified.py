# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unified SDPA backward: THD queries over a paged KV cache (dQ, dK, dV).

The unified family serves the vLLM-style layout of the unified forward:
``Q, O, dO, dQ`` are ``[total_q, h_q, D]`` with ``(token, head)`` element
strides and int32 ``cu_seqlens_q`` token offsets; ``LSE`` is natural-log fp32
``[total_q, h_q]``; ``K`` / ``V`` are paged caches ``[num_blocks, page, h_kv,
D]`` with element strides ``(block, token, head)`` (unit stride along ``D``),
int32 ``seq_lens_kv`` and an int32 ``block_table[num_seqs][max_blocks_per_seq]``
with a row stride; ``dK`` / ``dV`` are gradient caches with the layout, the
strides and the block table of ``K`` / ``V``.

Main kernel (``rocke.attn_bwd_unified.v1``): KV-parallel, one CTA per
(logical kv tile of ``kN0`` keys, kv head x ``g_split``, sequence). The CTA
runs the shared tile step (:mod:`._attention_bwd_body`) and the q loops of
the layout-general family (:func:`.attention_bwd._kv_tile`: head groups,
head packing, edge tiles, step order) with the paged address generator
(:mod:`._attention_bwd_paged`) for K / V, then one of two dK / dV epilogues:

* ``page_mode = "direct"`` (the caller asserts the block table is injective
  over live slots): every CTA stores its tile's dK / dV (output dtype,
  ``dk_mult`` on dK) into the physical slots of K / V; slots of the last live
  page past ``len_kv`` get exact zeros (tiles in ``[len_kv,
  roundup(len_kv, page))`` skip the q loop and store zeros); pages not
  referenced by a live block are not touched. Deterministic.
* ``page_mode = "atomic"`` (the default; correct with shared pages): fp32
  ``global_atomic_add_f32`` of every key row ``< len_kv`` into the page
  workspace ``WS_PDK`` / ``WS_PDV`` (``[h_kv][num_blocks * page][D]``, by
  physical slot), zeroed beforehand and converted afterwards by the paged
  aux roles of :mod:`.attention_bwd_aux`. Non-deterministic in the last bits.

Page classes: ``sub_page = None`` is the ``"multi"`` class (runtime page size,
``page % kN0 == 0``, one block-table lookup per tile); an integer ``sub_page =
p`` is the ``"sub"`` class (``kN0 % p == 0``, ``p < kN0``, compile-time page,
``kN0 / p`` lookups per tile). The one-wave ``kN0 = 16`` tile divides every
legal page and is the fallback of the host plan.

:class:`AttnBwdUnifiedSpec` carries the unified fields (``page_mode``,
``mask_class``, ``sub_page``, ``stage_vec``) plus every tile / residency /
path / schedule / codegen knob of :class:`.attention_bwd.AttnBwdSpec` as a
defaulted field (the knob fields are derived from that spec, so a knob added
to the general family is a unified knob too). ``None`` knobs resolve to the
general family's start point for THD (:func:`attn_bwd_unified_default_geometry`);
legality is the general family's (:func:`.attention_bwd.validate_attn_bwd_spec`
on :meth:`AttnBwdUnifiedSpec.general_spec`) plus the page-class rules.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Optional

from rocke.core.codegen_policy import (
    SCHEDULER_STRATEGY_ATTR,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.helpers.attention_band import AttnRuntimeBounds
from rocke.helpers.spec import non_default_fields, salted_kernel_name

from kernels.common._attention_bwd_addr import (
    StridedTensorAddr,
    WorkspaceAddr,
    chunk_coords,
    deal_chunks,
    decode_batched,
    decode_thd,
)
from kernels.common._attention_bwd_band import NoMaskKTail, RuntimeBand
from kernels.common._attention_bwd_body import (
    BwdLds,
    BwdTileGeometry,
    BwdTileStep,
    StatsView,
)
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_paged import PagedKv
from kernels.common._attention_bwd_sinks import (
    AtomicWorkspaceSink,
    PackedAtomicWorkspaceSink,
)
from kernels.common._attention_bwd_worklist import decode_cta
from kernels.common.attention_bwd import (
    AGPR_ALLOC_BACKEND,
    BWD_BARRIER_FENCE,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    SPLIT_AGPR_ROUTES,
    AttnBwdSpec,
    _kv_tile,
    _not_built,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
    lds_plan,
    static_unit_mults,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_aux import AttnBwdAuxSpec, attn_bwd_aux_params

__all__ = [
    "ATTN_BWD_UNIFIED_ABI",
    "PAGE_CLASSES",
    "PAGE_GRANULE",
    "PAGE_MODES",
    "UNIFIED_KNOBS",
    "AttnBwdUnifiedSpec",
    "attn_bwd_unified_default_geometry",
    "attn_bwd_unified_params",
    "build_attn_bwd_unified_main",
    "validate_attn_bwd_unified_spec",
]

ATTN_BWD_UNIFIED_ABI = "rocke.attn_bwd_unified.v1"
PAGE_MODES = ("atomic", "direct")
PAGE_CLASSES = ("multi", "sub")
MASK_CLASSES = ("band", "none")
# Every legal page size is a multiple of this (the one-wave tile's kN0).
PAGE_GRANULE = 16
# The aux kernels (q-side prep, dQ convert, page roles) are the shared ones of
# the backward family, under the family's aux name.
AUX_NAME = "rocke_attn_bwd"

# Fields of the general spec the unified spec does not carry as knobs: the
# problem keys it fixes (THD queries, atomic dQ) or carries itself.
_NOT_KNOBS = (
    "head_size",
    "dtype",
    "seq_mode",
    "dkv_mode",
    "mask_class",
    "dq_mode",
    "stage_vec",
    "name",
)


def _knob_fields() -> list:
    hints = typing.get_type_hints(AttnBwdSpec)
    out = []
    for f in dataclasses.fields(AttnBwdSpec):
        if f.name in _NOT_KNOBS:
            continue
        if f.default is dataclasses.MISSING:
            raise TypeError(f"general knob {f.name!r} has no default")
        out.append((f.name, hints[f.name], dataclasses.field(default=f.default)))
    return out


_KNOB_FIELDS = _knob_fields()
# General knobs that act in the general main kernel's own shell (its CTA
# decode, its kernarg attributes, its strided dK / dV slabs) rather than in the
# shared tile step / kv-tile body the unified kernel reuses: the unified shell
# does not build them, so only their default is legal here.
_SHELL_KNOBS = {
    "xcd_remap": False,
    "lb_interleave": False,
    "ptr_noalias": False,
    "dkv_slab_hoist": False,
    "lb_mirror": False,
    # its own epilogues, stats view and addressing (the paged shell): the slice
    # sink, the buffer-bounded stats, the per-slot quad epilogue, the access
    # width cap and the carried slab offsets are the general shell's
    "dkv_sink": "atomic",
    "ws_bounds": "clamp",
    "dkv_cell_vec": 1,
    "narrow_vec": 1,
    "slab_carry": False,
}
# The tile / residency / path / schedule / codegen knobs shared with the
# general family (field names of AttnBwdUnifiedSpec).
UNIFIED_KNOBS = tuple(name for name, _t, _f in _KNOB_FIELDS)
# Fields that form the unsalted base name of a main instance.
_BASE_FIELDS = (
    "head_size",
    "dtype",
    "page_mode",
    "mask_class",
    "stage_vec",
    "sub_page",
)


def _check_in(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {tuple(allowed)}")


class _UnifiedSpecMethods:
    """Methods of :class:`AttnBwdUnifiedSpec` (a generated dataclass)."""

    def __post_init__(self) -> None:
        _check_in("head_size", self.head_size, BWD_HEAD_SIZES)
        _check_in("dtype", self.dtype, BWD_DTYPES)
        _check_in("page_mode", self.page_mode, PAGE_MODES)
        _check_in("mask_class", self.mask_class, MASK_CLASSES)
        _check_in("stage_vec", self.stage_vec, (8, 1))
        sp = self.sub_page
        if sp is not None and (
            isinstance(sp, bool)
            or not isinstance(sp, int)
            or sp < PAGE_GRANULE
            or sp % PAGE_GRANULE
        ):
            raise ValueError(
                f"sub_page={sp!r} must be None or a positive multiple of {PAGE_GRANULE}"
            )
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"name={self.name!r} must be a non-empty identifier")
        # Knob values and arch-free couplings: the general spec's rules (it
        # also canonicalises tuples and the swizzle string).
        general = self.general_spec()
        for k in UNIFIED_KNOBS:
            object.__setattr__(self, k, getattr(general, k))

    # -- derived -----------------------------------------------------------

    @property
    def page_class(self) -> str:
        return "multi" if self.sub_page is None else "sub"

    def general_spec(self) -> AttnBwdSpec:
        """The layout-general spec of the same tile step (THD, atomic dQ).

        ``page_mode = "atomic"`` maps to ``dkv_mode = "atomic"`` (the epilogue
        adds fp32 values and stages nothing), ``"direct"`` to ``"direct"``.
        """
        return AttnBwdSpec(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode="thd",
            dkv_mode="atomic" if self.page_mode == "atomic" else "direct",
            stage_vec=self.stage_vec,
            mask_class=self.mask_class,
            name=self.name,
            **{k: getattr(self, k) for k in UNIFIED_KNOBS},
        )

    def resolved(self, arch: str) -> "AttnBwdUnifiedSpec":
        """A copy with every ``None`` knob replaced by the arch policy value."""
        g = self.general_spec().resolved(arch)
        return dataclasses.replace(self, **{k: getattr(g, k) for k in UNIFIED_KNOBS})

    def non_default_fields(self) -> tuple[tuple[str, object], ...]:
        """Compile-time fields that differ from their declared default."""
        return non_default_fields(self, skip=("head_size", "name"))

    def kernel_name(self, stage: str = "main") -> str:
        """Main kernel name: base key plus a salt over every non-default knob.

        The aux kernels are the shared backward ones: see :meth:`aux_spec`.
        """
        _check_in("stage", stage, ("main",))
        base = (
            f"{self.name}_main_{self.dtype}_d{self.head_size}_{self.page_mode}_"
            f"{self.mask_class}_sv{self.stage_vec}"
        )
        if self.sub_page is not None:
            base += f"_sp{self.sub_page}"
        return salted_kernel_name(
            base,
            [(k, v) for k, v in self.non_default_fields() if k not in _BASE_FIELDS],
        )

    def aux_spec(
        self,
        stage: str,
        *,
        rows_per_cta: Optional[int] = None,
        convert_vec: int = 8,
    ) -> AttnBwdAuxSpec:
        """The shared aux spec of ``stage``.

        ``"prep"`` / ``"convert"``: the THD q-side prep and dQ convert of the
        general family (same instances); ``"page_zero"`` / ``"page_convert"``:
        the paged workspace roles (``kv_target = "paged"``, head-major).
        """
        _check_in("stage", stage, ("prep", "convert", "page_zero", "page_convert"))
        if stage != "convert" and convert_vec != 8:
            raise ValueError("convert_vec applies to the dQ convert only")
        paged = stage in ("page_zero", "page_convert")
        return AttnBwdAuxSpec(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode="thd",
            stage_vec=self.stage_vec,
            ws_layout="head_major" if paged else self.ws_layout,
            kv_target="paged" if paged else "contiguous",
            rows_per_cta=rows_per_cta,
            convert_vec=convert_vec,
            name=AUX_NAME,
        )


AttnBwdUnifiedSpec = dataclasses.make_dataclass(
    "AttnBwdUnifiedSpec",
    [
        ("head_size", int),
        ("dtype", str, dataclasses.field(default="bf16")),
        ("page_mode", str, dataclasses.field(default="atomic")),
        ("mask_class", str, dataclasses.field(default="band")),
        ("sub_page", Optional[int], dataclasses.field(default=None)),
        ("stage_vec", int, dataclasses.field(default=8)),
        *_KNOB_FIELDS,
        ("name", str, dataclasses.field(default="rocke_attn_bwd_unified")),
    ],
    bases=(_UnifiedSpecMethods,),
    frozen=True,
    namespace={
        "__doc__": (
            "Compile-time spec of the unified backward main kernel.\n\n"
            "``head_size`` (32, 64, 128) is the only required field. "
            "``page_mode``: ``'atomic'`` (default, fp32 page workspace) or "
            "``'direct'`` (scatter; the caller asserts an injective table). "
            "``sub_page``: ``None`` (page class 'multi', runtime page, "
            "``page % block_n == 0``) or the compile-time page of the 'sub' "
            "class. Every other field is a knob of the general backward "
            "(:class:`AttnBwdSpec`), ``None`` resolved per arch."
        )
    },
)
AttnBwdUnifiedSpec.__module__ = __name__


def attn_bwd_unified_default_geometry(
    arch: str, head_size: int, dtype: str = "bf16", stage_vec: int = 8
) -> dict:
    """Start point of every ``None`` knob: the general family's THD start point."""
    return attn_bwd_default_geometry(arch, head_size, dtype, stage_vec, "thd")


def validate_attn_bwd_unified_spec(spec, arch: str):
    """Resolve ``spec`` for ``arch`` and apply every static legality rule.

    The general family's rules on :meth:`AttnBwdUnifiedSpec.general_spec`,
    plus: a ``sub_page`` must divide ``block_n`` and stay below it; the knobs
    of the general main-kernel shell (:data:`_SHELL_KNOBS`) and its Q / dO
    prefetch ring (``ring_depth > 1`` without the barrier merge, the DMA path)
    are not built here. Returns the resolved unified spec.
    """
    # the general main kernel's recorded resource failures (RESOURCE_REFUSED)
    # are of another code object: the unified kernel's gate rows cover its own
    g = validate_attn_bwd_spec(spec.general_spec(), arch, main_kernel_refusals=False)
    s = dataclasses.replace(spec, **{k: getattr(g, k) for k in UNIFIED_KNOBS})
    for name, default in _SHELL_KNOBS.items():
        if getattr(s, name) != default:
            raise ValueError(
                f"{name}={getattr(s, name)!r}: a knob of the general main-kernel "
                "shell (CTA decode, kernarg attributes or strided dK / dV "
                "epilogue), not built for the unified family"
            )
    if (s.ring_depth != 1 and s.barrier_merge != "on") or s.global_path != "vgpr":
        # the general shell builds the Q / dO prefetch ring (QRing) and hands
        # it to the kv-tile body; the unified shell does not
        raise ValueError(
            f"ring_depth={s.ring_depth}, global_path={s.global_path!r}: the "
            "prefetch ring of the general main-kernel shell is not built for the "
            "unified family"
        )
    if s.sub_page is not None:
        if s.block_n % s.sub_page or s.sub_page >= s.block_n:
            raise ValueError(
                f"sub_page={s.sub_page} must divide block_n={s.block_n} and be "
                "below it (otherwise the page is the 'multi' class)"
            )
    return s


# ---------------------------------------------------------------------------
# kernarg ABI rocke.attn_bwd_unified.v1
# ---------------------------------------------------------------------------
# ``page_rows`` is the cache's physical slot count ``num_blocks * page`` in both
# page modes: it bounds the atomic page workspace and turns on the block-id
# guard of :class:`._attention_bwd_paged.PagedKv` (an entry outside ``[0,
# num_blocks)`` is never dereferenced).

_MAIN_PTRS = (
    "Q KC VC dO DKC DVC WS_LSE2 WS_DSUM WS_DQ WS_PDK WS_PDV CU_SEQLENS_Q "
    "SEQ_LENS_KV BLOCK_TABLE WORKLIST"
).split()
_MAIN_F32 = ("scale_log2", "ds_mult", "dk_mult", "dq_mult")
_MAIN_I32 = (
    "q_t q_h do_t do_h kc_blk kc_tok kc_h vc_blk vc_tok vc_h bt_stride "
    "h_q h_k G num_seqs max_seqlen_q max_seqlen_kv max_blocks_per_seq page "
    "left right bottom_right ws_rows_q page_rows "
    "g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist"
).split()
_MAIN_READONLY = frozenset(
    "Q KC VC dO WS_LSE2 WS_DSUM CU_SEQLENS_Q SEQ_LENS_KV BLOCK_TABLE WORKLIST".split()
)
_IO_PTRS = ("Q", "KC", "VC", "dO", "DKC", "DVC")
_PAGE_STAGES = ("page_zero", "page_convert")


def attn_bwd_unified_params(stage: str = "main") -> tuple[tuple[str, str], ...]:
    """Ordered ``(name, struct format)`` kernargs of a unified stage.

    ``"main"``: the ``rocke.attn_bwd_unified.v1`` main list (pointers, f32,
    i32); ``"page_zero"`` / ``"page_convert"``: the paged aux roles (same ABI
    tag). The q-side prep and dQ convert use the general ``rocke.attn_bwd.v3``
    lists (:func:`.attention_bwd.attn_bwd_params`).
    """
    _check_in("stage", stage, ("main",) + _PAGE_STAGES)
    if stage in _PAGE_STAGES:
        return attn_bwd_aux_params(stage, AttnBwdAuxSpec(64, kv_target="paged"))
    return (
        tuple((p, "Q") for p in _MAIN_PTRS)
        + tuple((f, "f") for f in _MAIN_F32)
        + tuple((i, "i") for i in _MAIN_I32)
    )


def _declare_main(b, io) -> dict:
    args = {}
    for name, fmt in attn_bwd_unified_params("main"):
        if fmt == "Q":
            if name in _IO_PTRS:
                t = PtrType(io, "global")
            elif name.startswith("WS_"):
                t = PtrType(F32, "global")
            else:
                t = PtrType(I32, "global")
            attrs = {"readonly": True} if name in _MAIN_READONLY else {}
            args[name] = b.param(name, t, **attrs)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64


def _kernel_attrs(b, g: AttnBwdSpec, geom, facts) -> None:
    """The general main kernel's attributes for the resolved spec ``g``."""
    b.kernel.attrs["max_workgroup_size"] = max(geom.threads, _MIN_WG_ATTR)
    if facts.matrix_path == "mfma" and g.agpr_alloc not in (None, AGPR_ALLOC_BACKEND):
        b.kernel.attrs["agpr_alloc"] = tuple(g.agpr_alloc)
    if g.agpr_route in SPLIT_AGPR_ROUTES:
        b.kernel.attrs["mfma_vgpr_form"] = True
    if g.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = int(g.waves_per_eu)
    if g.scheduler_strategy is not None:
        b.kernel.attrs[SCHEDULER_STRATEGY_ATTR] = normalize_scheduler_strategy(
            g.scheduler_strategy
        )


def _emit_main(spec, arch: str, *, sentinel: bool = True):
    """Emit the unified main kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term of the dS select (negative
    controls only; not a spec field).
    """
    u = validate_attn_bwd_unified_spec(spec, arch)
    g = validate_attn_bwd_spec(u.general_spec(), arch, main_kernel_refusals=False)
    missing = _not_built(g)
    if missing:
        raise NotImplementedError(
            "the unified attention backward main kernel is not built for "
            + ", ".join(missing)
        )
    facts = attn_bwd_arch_facts(arch)
    geom = BwdTileGeometry.from_spec(g, arch)
    plan = lds_plan(g, arch)
    io = ir_type(u.dtype)
    D, kn = u.head_size, u.block_n

    b = IRBuilder(u.kernel_name())
    # the general family's barrier fence (same tile step and staging barriers)
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare_main(b, io)
    # Multipliers that are 1.0 under the compiled placement are not applied
    # (the general family's rule; the plan asserts they are 1.0).
    a.update(dict.fromkeys(static_unit_mults(g.static_scale, g.dkv_mode)))
    _kernel_attrs(b, g, geom, facts)

    tid, lane, wave = lane_and_wave(b, facts.wave_size)
    atomic = u.page_mode == "atomic"
    # CTA decode: kv tile fastest, then the kv head (and split), then the
    # sequence; the load-balance order permutes the kv tiles of one (head, seq).
    work = decode_cta(
        b, x=b.block_id_x(), y=b.block_id_y(), z=b.block_id_z(),
        n_kv_tiles=a["n_kv_tiles"], lb_order=a["lb_order"],
        g_split=a["g_split"] if atomic else None,
    )  # fmt: skip
    s_idx = work.batch
    # q side: THD rows by token (cu_seqlens_q, int32 token offsets; the
    # workspace rows are the tokens, ws_rows_q = total_q); kv side: the
    # sequence's kv length, clamped to [0, max_seqlen_kv].
    seq_q = decode_thd(b, s_idx, s_max=a["max_seqlen_q"], off_ptr=a["CU_SEQLENS_Q"])
    seq_kv = decode_batched(
        b, s_idx, s_max=a["max_seqlen_kv"], seq_ptr=a["SEQ_LENS_KV"], has_len=1
    )
    # The kv head of the CTA owns query heads [hk * G, hk * G + G); a split
    # (atomic page mode) owns G / g_split of them.
    G = a["G"]
    hk = work.unit
    if atomic:
        per = b.div(G, a["g_split"])
        h0 = b.add(b.mul(hk, G), b.mul(work.split, per))
        n_h = per
    else:
        h0, n_h = b.mul(hk, G), G

    def q_side(name, prefix):
        return StridedTensorAddr(
            b, a[name], dtype=io, seq=seq_q, stage_vec=u.stage_vec,
            strides=(0, a[f"{prefix}_h"], a[f"{prefix}_t"]),
        )  # fmt: skip

    q_addr = q_side("Q", "q")
    do_addr = q_side("dO", "do")
    nh = a["h_q"] if u.ws_layout == "token_major" else None

    def ws(name, width):
        return WorkspaceAddr(
            b, a[name], ws_rows=a["ws_rows_q"], width=width, row0=seq_q.ws_row0,
            layout=u.ws_layout, n_heads=nh,
        )  # fmt: skip

    # stats_vec > 1 reads <n x f32> windows kept inside the head-major arrays
    # (StatsView needs their query-head count, as in the general shell)
    stats = StatsView(
        ws("WS_LSE2", 1), ws("WS_DSUM", 1),
        n_heads=a["h_q"] if u.stats_vec != 1 else None,
    )  # fmt: skip
    sink_cls = PackedAtomicWorkspaceSink if u.head_pack else AtomicWorkspaceSink
    sink = sink_cls(b, ws("WS_DQ", D), seq=seq_q, scale=a["dq_mult"])
    zero = b.const_i32(0)
    if u.mask_class == "band":
        # Per-sequence diagonal: 0 (top-left) or len_kv - len_q (bottom-right).
        off = b.select(
            b.cmp_ne(a["bottom_right"], zero),
            b.sub(seq_kv.length, seq_q.length),
            zero,
        )
        bounds = AttnRuntimeBounds(
            seq_q.length, seq_kv.length, off, a["left"], a["right"]
        )
        mask = RuntimeBand(b, bounds, k_m0=u.block_m, k_n0=kn)
    else:
        mask = NoMaskKTail(b, seq_q.length, seq_kv.length, k_m0=u.block_m, k_n0=kn)
    lds = BwdLds(b, plan, u.dtype)
    step = BwdTileStep(
        b, geom, mask=mask, stats=stats, dq_sink=sink, lds=lds, tid=tid, lane=lane,
        wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
        sentinel=sentinel,
    )  # fmt: skip
    paged = PagedKv(
        b, bt_ptr=a["BLOCK_TABLE"], bt_stride=a["bt_stride"], seq_idx=s_idx,
        seq=seq_kv, k_n0=kn,
        page=a["page"] if u.sub_page is None else u.sub_page, sub_page=u.sub_page,
        page_rows=a["page_rows"],
    )  # fmt: skip

    def cache(ptr, prefix):
        return paged.cache(
            a[ptr], dtype=io, stage_vec=u.stage_vec,
            strides=(a[f"{prefix}_blk"], a[f"{prefix}_tok"], a[f"{prefix}_h"]),
        )  # fmt: skip

    k_cache, v_cache = cache("KC", "kc"), cache("VC", "vc")
    if atomic:

        def epilogue(b, step, acc, *, k0, live):
            _page_atomic_epilogue(
                b, u, a, step, acc, paged=paged, hk=hk, k0=k0, live=live
            )

    else:
        out = (cache("DKC", "kc"), cache("DVC", "vc"))

        def epilogue(b, step, acc, *, k0, live):
            _page_direct_epilogue(
                b, u, a, step, acc, paged=paged, out=out, hk=hk, k0=k0, tid=tid, io=io
            )

    _kv_tile(
        b, g, a, step, mask, kt=work.first_kt, hk=hk, hv=hk, h0=h0, n_h=n_h,
        seq_q=seq_q, seq_kv=seq_kv, q_addr=q_addr, do_addr=do_addr,
        k_addr=k_cache, v_addr=v_cache, io=io, tid=tid, epilogue=epilogue,
    )  # fmt: skip
    b.ret()
    return b.kernel


def _page_atomic_epilogue(b, u, a, step, acc, *, paged, hk, k0, live) -> None:
    """fp32 dK / dV of the key tile added into the page workspace.

    Inside ``scf_if(live)``. Every key row ``< len_kv`` of the lane adds its
    values (dK times ``dk_mult``) at its physical slot of ``WS_PDK`` /
    ``WS_PDV`` (kv head ``hk``); rows past ``len_kv`` add nothing, and a slot
    at or past ``page_rows`` is never written. Under ``dkv_atomic_guard =
    "select"`` a row outside adds ``+0.0`` at a clamped slot instead (no row
    branch; identical sums).
    """
    tile = paged.tile(k0)
    select = u.dkv_atomic_guard == "select"
    with b.scf_if(live):
        for which, ptr, mult in (
            ("dk", "WS_PDK", a["dk_mult"]),
            ("dv", "WS_PDV", None),
        ):
            row_slab = tile.workspace_row(
                a[ptr], page_rows=a["page_rows"], width=u.head_size, head=hk
            )
            for row, items in step.dkv_row_slots(b, which, acc):
                if mult is not None:
                    items = [(c, b.fmul(v, mult)) for c, v in items]
                in_seq = b.cmp_lt(b.add(k0, row), paged.seq.length)
                slab, rr = row_slab(row)
                if select:
                    slab.atomic_add_row_select(rr, items, in_seq)
                else:
                    slab.atomic_add_row(rr, items, in_seq)


def _page_direct_epilogue(b, u, a, step, acc, *, paged, out, hk, k0, tid, io) -> None:
    """dK / dV rows of the key tile scattered to the gradient caches.

    Rows with a physical slot in a live page (``k < ceil(len_kv / page) *
    page``) are stored: the gradient (``dk_mult`` on dK) for ``k < len_kv``,
    exact zeros past it. Other rows are not touched. ``dkv_epilogue`` picks
    the path as in the general family; every path stores the same values.
    """
    tile = paged.tile(k0)
    outputs = (("dk", out[0], a["dk_mult"]), ("dv", out[1], None))

    def slab(cache):
        return cache.slab(hk, k0, clamp=False)

    if u.dkv_epilogue == "direct":
        e0 = b.vec_extract(b.zero_vec(io, 1), 0)
        for which, cache, mult in outputs:
            dst = slab(cache)
            for row, items in step.dkv_cells(b, which, acc, mult):
                k = b.add(k0, row)
                store, row_live = tile.store_rule(k)
                with b.scf_if(store):
                    for col, v in items:
                        dst.store(k, col, b.select(row_live, v, e0))
        return
    vec = u.dkv_store_vec
    zero = b.zero_vec(io, vec)

    def store_rows(dst, read):
        D, threads = u.head_size, step.geom.threads
        per_row = D // vec
        total = u.block_n * per_row
        for chunk, tail in deal_chunks(b, tid, total, threads):
            r, c = chunk_coords(b, chunk, per_row, vec)
            k = b.add(k0, r)
            store, row_live = tile.store_rule(k)
            if tail:
                store = b.land(store, b.cmp_lt(chunk, b.const_i32(total)))
            with b.scf_if(store):
                dst.store_vec(k, c, b.select(row_live, read(r, c), zero))

    if u.dkv_epilogue == "lds_staged_pair":
        staged = []
        for slot, (which, cache, mult) in enumerate(outputs):
            last = slot == len(outputs) - 1
            read = step.dkv_rows(b, which, acc, mult, vec=vec, slot=slot, sync=last)
            staged.append((slab(cache), read))
        for dst, read in staged:
            store_rows(dst, read)
        b.sync()  # reads of the epilogue buffers done before they are reused
        return
    for which, cache, mult in outputs:
        read = step.dkv_rows(b, which, acc, mult, vec=vec)
        store_rows(slab(cache), read)
        b.sync()  # reads of the epilogue buffer done before it is reused


def build_attn_bwd_unified_main(spec: AttnBwdUnifiedSpec, *, arch: str):
    """Build the unified backward main kernel (``rocke.attn_bwd_unified.v1``).

    The spec is resolved and validated for ``arch`` first (an illegal
    geometry raises ``ValueError``); knob values whose emission is not built
    raise :class:`NotImplementedError` before any IR is emitted.
    """
    return _emit_main(spec, arch)
