# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Multi-wave FP16/BF16 WMMA attention forward (prefill) for gfx11 / gfx12.

One object per (dtype, head-width bucket, mask class, bias kind) per RDNA
family, compiled for ``gfx11-generic`` or ``gfx12-generic`` (or one of their
processors). The body lives in :mod:`rocke.helpers.rdna_attention`; this module
owns the kernel ABI, the CTA decode and the per-(batch, head) buffer
resources.

Runtime (kernel arguments): batch, Sq, Sk, Hq, Hkv and Hv (the K and V head
counts; each divides Hq, query head ``h`` reads K head ``h / (Hq / Hkv)`` and
V head ``h / (Hq / Hv)``), the Q/K
and V head widths within the bucket (multiples of 8), every batch / token /
head stride in elements (head-dim stride 1, token strides multiples of 8 so
rows stay 16-byte aligned), the softmax scale (``scale_log2 = scale * log2(e)
> 0``), causal alignment (top-left / bottom-right), window bounds, whether LSE
is written, and the bias broadcast strides (batch / head / query; 0
broadcasts, key stride 1).

Baked: dtype, Q/K and V head-width buckets, mask class (``none`` or ``causal``
-- the latter with runtime diagonal and window bounds), bias kind (none, f32,
Q dtype), and the CTA geometry. Buckets above 128 split the head widths
across a wave pair (gfx12 keeps an unsplit Q up to 256 when V <= 128 with
``wave_pair=False``).

Optional baked features (each appends its runtime arguments to the ABI, in
the order below, before the bias arguments): softcap (``softcap``), attention
sinks (``sink_ptr``,
[Hq] in the Q dtype), ALiBi (``alibi_slopes_ptr``, [Hq] f32), QQ-bias
(``qq_bias_ptr`` [rows, cols] f32 with ``qq_bias_rows / cols / stride``), a
packed layout -- ragged (``cu_seqlens_q``, ``cu_seqlens_k``) or paged
(``cu_seqlens_q``, ``seqused_k``, ``block_table``, ``block_table_stride``,
``stride_k_block``, ``stride_v_block``, ``num_blocks`` (the caches' page count;
a ``block_table`` entry outside ``[0, num_blocks)`` reads as zero K / V rows);
power-of-two ``page_block_size``) --
and OCP E4M3FN K/V (``k_scale``, ``v_scale``). Packed layouts take Q / O as
[total_q, Hq, D] and LSE per packed row; ``batch`` is the sequence count and
``seqlen_q`` / ``seqlen_k`` are the maxima (they size the grid; the
per-sequence lengths come from the offsets). Positions (causal diagonal,
ALiBi, QQ-bias, the bias query row) are sequence-local.

Grid: flat 1-D, ``batch * ceil(Sq / block_m) * Hq`` CTAs of
``WmmaFmhaFwdV2Spec.threads`` threads (:func:`wmma_fmha_fwd_v2_grid`). Heads
run fastest, then the query tile and batch in ``cta_order``; the causal class
walks query tiles heaviest-first (LPT).

C++ mirror: ``platform/cpp/instances/gfx1151/wmma_fmha_fwd_v2.cpp``
(byte-identical emission).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
)

__all__ = [
    "WmmaFmhaFwdV2Spec",
    "build_wmma_fmha_fwd_v2",
    "is_gfx12_family",
    "is_valid_spec",
    "wmma_fmha_fwd_v2_grid",
    "wmma_fmha_fwd_v2_lds_bytes",
    "wmma_fmha_fwd_v2_signature",
]

_MAX_THREADS = 1024
_MAX_LDS_BYTES = 65536


@dataclass(frozen=True)
class WmmaFmhaFwdV2Spec:
    """One compiled RDNA prefill object.

    Tuning knobs (all emitted IR changes; the defaults are the shipping
    shipping configuration): ``block_m``, ``block_n``, ``waves_per_eu``,
    ``scheduler_strategy``, ``lazy_rescale``, ``k_pad`` / ``v_pad``,
    ``wave_pair`` and ``cta_order``. Feature flags: ``use_softcap``,
    ``use_sinks``, ``use_alibi``, ``use_qq_bias``, ``layout`` (with
    ``page_block_size`` for ``paged``) and ``kv_dtype``.
    """

    dtype: str = "bf16"  # "fp16" | "bf16"
    head_dim: int = 128  # Q/K head-width bucket (multiple of 16, <= 256)
    head_dim_v: int = 0  # V head-width bucket (0 -> head_dim)
    mask: str = "none"  # "none" | "causal"
    bias: Optional[str] = None  # additive bias: None | "f32" | "q" (Q dtype)
    use_softcap: bool = False  # runtime ``softcap`` cap (tanh logit capping)
    use_sinks: bool = False  # per-query-head sink logit (Q dtype, [Hq])
    use_alibi: bool = False  # per-query-head ALiBi slopes (f32, [Hq])
    use_qq_bias: bool = False  # query-section [rows, cols] f32 bias
    layout: str = "dense"  # "dense" | "ragged" | "paged"
    page_block_size: int = 0  # paged only: tokens per page (power of two)
    kv_dtype: str = ""  # "" -> Q dtype; "fp8e4m3" -> OCP E4M3FN K/V bytes
    # Query rows per CTA (16 per wave or wave pair). 0 -> the shipping
    # default: 128 (8 waves), or 64 (8 waves as 4 pairs) with the wave pair.
    block_m: int = 0
    block_n: int = 32  # keys per tile (multiple of 16, <= 128)
    waves_per_eu: int = 2  # occupancy hint; 0 -> none
    scheduler_strategy: Optional[str] = None  # None -> LLVM default scheduler
    lazy_rescale: bool = True  # skip the O/l rescale while the max barely moves
    k_pad: int = 8  # LDS row padding of the K tile (elements)
    v_pad: int = 8  # LDS row padding of the V^T tile (elements)
    wave_pair: Optional[bool] = None  # None -> split head widths above 128
    # CTA order after the head: "hqb" (query tile, then batch: a (batch,
    # head)'s tiles run together and share its K/V in L2) or "hbq" (batch,
    # then query tile).
    cta_order: str = "hqb"
    name: str = "rocke_wmma_fmha_fwd_v2"

    def __post_init__(self) -> None:
        from rocke.core.codegen_policy import normalize_scheduler_strategy

        object.__setattr__(
            self,
            "scheduler_strategy",
            normalize_scheduler_strategy(self.scheduler_strategy),
        )
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError(f"dtype must be fp16/bf16, got {self.dtype!r}")
        if self.mask not in ("none", "causal"):
            raise ValueError(f"mask must be none/causal, got {self.mask!r}")
        if self.bias not in (None, "f32", "q"):
            raise ValueError(f"bias must be None, 'f32' or 'q', got {self.bias!r}")
        if self.cta_order not in ("hqb", "hbq"):
            raise ValueError(f"cta_order must be hqb/hbq, got {self.cta_order!r}")
        if self.head_dim_v < 0 or self.block_m < 0 or self.waves_per_eu < 0:
            raise ValueError("head_dim_v, block_m and waves_per_eu must be >= 0")
        if self.layout not in ("dense", "ragged", "paged"):
            raise ValueError(f"layout must be dense/ragged/paged, got {self.layout!r}")
        ps = self.page_block_size
        if self.layout == "paged":
            if ps <= 0 or ps & (ps - 1):
                raise ValueError("paged attention needs a power-of-two page size")
        elif ps:
            raise ValueError("page_block_size is only valid for the paged layout")
        if self.kv_dtype not in ("", "fp8e4m3"):
            raise ValueError(f"kv_dtype must be '' or 'fp8e4m3', got {self.kv_dtype!r}")

    @property
    def dv(self) -> int:
        """V / O head-width bucket."""
        return self.head_dim_v or self.head_dim

    @property
    def pair(self) -> bool:
        """Whether the head widths are split across a wave pair."""
        if self.wave_pair is not None:
            return self.wave_pair
        return max(self.head_dim, self.dv) > 128

    @property
    def bm(self) -> int:
        """Query rows per CTA."""
        return self.block_m or (64 if self.pair else 128)

    @property
    def threads(self) -> int:
        """Threads per CTA (one wave per 16 query rows, two with the pair)."""
        return 2 * self.bm * (2 if self.pair else 1)

    def kernel_name(self) -> str:
        """Symbol of the compiled object; distinct for every emitted variant."""
        dt = "bf16" if self.dtype == "bf16" else "f16"
        dv = f"v{self.dv}" if self.dv != self.head_dim else ""
        bias = f"_bias{self.bias}" if self.bias else ""
        feats = "".join(
            tag
            for tag, on in (
                ("_softcap", self.use_softcap),
                ("_sinks", self.use_sinks),
                ("_alibi", self.use_alibi),
                ("_qqbias", self.use_qq_bias),
                ("_ragged", self.layout == "ragged"),
                (f"_paged{self.page_block_size}", self.layout == "paged"),
                ("_kvfp8", bool(self.kv_dtype)),
            )
            if on
        )
        geo = f"m{self.bm}n{self.block_n}"
        geo += "p" if self.pair else ""
        geo += "_hbq" if self.cta_order == "hbq" else ""
        knobs = ""
        if self.waves_per_eu != 2:
            knobs += f"_wpe{self.waves_per_eu}"
        if self.scheduler_strategy:
            knobs += "_sched_" + self.scheduler_strategy.replace("-", "_")
        if not self.lazy_rescale:
            knobs += "_eager"
        if (self.k_pad, self.v_pad) != (8, 8):
            knobs += f"_pad{self.k_pad}x{self.v_pad}"
        return (
            f"{self.name}_{dt}_d{self.head_dim}{dv}_{self.mask}{bias}{feats}_{geo}"
            f"{knobs}"
        )

    def normalized(self) -> "WmmaFmhaFwdV2Spec":
        """The same object with the derived V width and block_m filled in."""
        return replace(self, head_dim_v=self.dv, block_m=self.bm)

    # ---- hipDNN catalog metadata ----
    # hipkernel:Gfx1151WmmaAttention ships this spec beside WmmaFmhaDecodeSpec,
    # and its IngestorGenerator specialization binds each matcher field to one
    # accessor both specs define (configs/gfx1151_wmma_attention.yaml).
    def catalog_kind(self) -> str:
        """``"prefill"``: this object runs the multi-wave prefill kernel."""
        return "prefill"

    def catalog_head_size(self) -> int:
        """Q/K head-width bucket."""
        return self.head_dim

    def catalog_head_size_v(self) -> int:
        """V/O head-width bucket."""
        return self.dv

    def catalog_mask(self) -> str:
        """Mask class: ``"none"`` or ``"causal"`` (runtime diagonal and window)."""
        return self.mask

    def catalog_bias(self) -> str:
        """Additive bias kind: ``"none"``, ``"f32"`` or ``"q"``."""
        return self.bias or "none"

    def catalog_block_size(self) -> int:
        """Threads per CTA."""
        return self.threads

    def catalog_block_m(self) -> int:
        """Query rows per CTA."""
        return self.bm

    def catalog_dv_split(self) -> int:
        """CTAs sharing one O row: always 1 for prefill."""
        return 1


def is_gfx12_family(arch: str) -> bool:
    """Whether ``arch`` uses the gfx12 WMMA operand layout."""
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    return target.mma.by_op_id("wmma_gfx12_f32_16x16x16_bf16") is not None


def _body_config(spec: WmmaFmhaFwdV2Spec, gfx12: bool):
    from rocke.helpers.rdna_attention import RdnaAttentionConfig

    return RdnaAttentionConfig(
        dtype=spec.dtype,
        head_dim=spec.head_dim,
        head_dim_v=spec.dv,
        block_m=spec.bm,
        block_n=spec.block_n,
        causal=spec.mask == "causal",
        gfx12=gfx12,
        lazy_rescale=spec.lazy_rescale,
        k_pad=spec.k_pad,
        v_pad=spec.v_pad,
        bias=spec.bias,
        wave_pair=spec.pair,
        softcap=spec.use_softcap,
        sinks=spec.use_sinks,
        alibi=spec.use_alibi,
        qq_bias=spec.use_qq_bias,
        kv_fp8=bool(spec.kv_dtype),
        page_block_size=spec.page_block_size,
    )


def _wmma_op_id(spec: WmmaFmhaFwdV2Spec, gfx12: bool) -> str:
    return ("wmma_gfx12_f32_16x16x16_" if gfx12 else "wmma_f32_16x16x16_") + (
        "bf16" if spec.dtype == "bf16" else "f16"
    )


def wmma_fmha_fwd_v2_lds_bytes(spec: WmmaFmhaFwdV2Spec, *, arch: str) -> int:
    """Static LDS bytes one CTA of ``spec`` allocates on ``arch``."""
    from rocke.helpers.rdna_attention import rdna_attention_lds_bytes

    return rdna_attention_lds_bytes(_body_config(spec, is_gfx12_family(arch)))


def is_valid_spec(spec: WmmaFmhaFwdV2Spec, *, arch: str) -> Tuple[bool, str]:
    """``(ok, reason)``: whether ``spec`` builds for ``arch``."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except (KeyError, ValueError) as e:
        return False, str(e)
    gfx12 = is_gfx12_family(arch)
    if target.wave_size != 32 or target.mma.by_op_id(_wmma_op_id(spec, gfx12)) is None:
        return False, f"{arch} has no wave32 {spec.dtype} WMMA atom"
    if spec.threads > _MAX_THREADS:
        return False, f"{spec.threads} threads per CTA exceed {_MAX_THREADS}"
    cfg = _body_config(spec, gfx12)
    try:
        cfg.validate()
    except ValueError as e:
        return False, str(e)
    if wmma_fmha_fwd_v2_lds_bytes(spec, arch=arch) > _MAX_LDS_BYTES:
        return False, f"LDS tiles exceed {_MAX_LDS_BYTES} bytes"
    return True, "ok"


def _declare_params(b: IRBuilder, spec: WmmaFmhaFwdV2Spec):
    elem = BF16 if spec.dtype == "bf16" else F16
    kv_elem, kv_align = (FP8E4M3, 1) if spec.kv_dtype else (elem, 2)
    p = {}
    for name, ro in (("Q", True), ("K", True), ("V", True), ("O", False)):
        kv = name in ("K", "V")
        attrs = dict(noalias=True, align=kv_align if kv else 2)
        attrs["readonly" if ro else "writeonly"] = True
        p[name] = b.param(name, PtrType(kv_elem if kv else elem, "global"), **attrs)
    p["LSE"] = b.param("LSE", PtrType(F32, "global"), writeonly=True, align=4)
    p["scale_log2"] = b.param("scale_log2", F32)
    for name in (
        "batch",
        "seqlen_q",
        "seqlen_k",
        "num_query_heads",
        "num_kv_heads",
        "num_v_heads",
        "head_dim_q",
        "head_dim_v",
        "bottom_right",
        "window_left",
        "window_right",
        "write_lse",
    ):
        p[name] = b.param(name, I32)
    for t in ("q", "k", "v", "o", "lse"):
        for axis in ("batch", "token", "head"):
            p[f"stride_{t}_{axis}"] = b.param(f"stride_{t}_{axis}", I32)

    def ptr(name, pointee, align):
        p[name] = b.param(name, PtrType(pointee, "global"), readonly=True, align=align)

    def scalars(dtype, *names):
        for name in names:
            p[name] = b.param(name, dtype)

    if spec.use_softcap:
        scalars(F32, "softcap")
    if spec.use_sinks:
        ptr("sink_ptr", elem, 2)
    if spec.use_alibi:
        ptr("alibi_slopes_ptr", F32, 4)
    if spec.use_qq_bias:
        ptr("qq_bias_ptr", F32, 4)
        scalars(I32, "qq_bias_rows", "qq_bias_cols", "qq_bias_stride")
    if spec.layout != "dense":
        ptr("cu_seqlens_q", I32, 4)
        if spec.layout == "ragged":
            ptr("cu_seqlens_k", I32, 4)
        else:
            ptr("seqused_k", I32, 4)
            ptr("block_table", I32, 4)
            scalars(
                I32,
                "block_table_stride",
                "stride_k_block",
                "stride_v_block",
                "num_blocks",
            )
    if spec.kv_dtype:
        scalars(F32, "k_scale", "v_scale")
    if spec.bias:
        bias_elem = F32 if spec.bias == "f32" else elem
        p["attn_bias_ptr"] = b.param(
            "attn_bias_ptr",
            PtrType(bias_elem, "global"),
            noalias=True,
            readonly=True,
            align=4 if spec.bias == "f32" else 2,
        )
        # Element strides; the key stride is 1, a 0 stride broadcasts.
        for axis in ("b", "h", "q"):
            p[f"bias_stride_{axis}"] = b.param(f"bias_stride_{axis}", I32)
    return p


def build_wmma_fmha_fwd_v2(spec: WmmaFmhaFwdV2Spec, *, arch: str) -> KernelDef:
    """Build one RDNA prefill attention kernel for ``arch``.

    Raises ``ValueError`` if ``is_valid_spec(spec, arch=arch)`` rejects it.
    """
    spec = spec.normalized()
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid wmma_fmha_fwd_v2 spec: {why}")
    from rocke.core.codegen_policy import CodegenPolicy, apply_codegen_policy
    from rocke.helpers.rdna_attention import rdna_attention_fwd_body

    cfg = _body_config(spec, is_gfx12_family(arch))
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = cfg.threads
    apply_codegen_policy(
        b.kernel, CodegenPolicy(scheduler_strategy=spec.scheduler_strategy)
    )
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, spec.waves_per_eu)
    p = _declare_params(b, spec)

    ci = b.const_i32
    c0, c1 = ci(0), ci(1)
    sq, sk = p["seqlen_q"], p["seqlen_k"]
    hq = p["num_query_heads"]
    bm = spec.bm

    # ---- CTA decode: head fastest, then (query tile, batch) in spec order ----
    bid = b.block_id_x()
    head = b.mod(bid, hq)
    rest = b.div(bid, hq)
    n_qt = b.div(b.add(sq, ci(bm - 1)), ci(bm))
    if spec.cta_order == "hqb":
        qt = b.mod(rest, n_qt)
        batch = b.div(rest, n_qt)
    else:
        batch = b.mod(rest, p["batch"])
        qt = b.div(rest, p["batch"])
    if spec.mask == "causal":
        qt = b.sub(b.sub(n_qt, c1), qt)  # heaviest tiles first (LPT)
    q_start = b.mul(qt, ci(bm))
    kv_head = b.div(head, b.div(hq, p["num_kv_heads"]))
    # V may carry its own head count (any Hv dividing Hq).
    v_head = b.div(head, b.div(hq, p["num_v_heads"]))

    # ---- packed layouts: this sequence's rows and lengths ----
    q_row0 = k_row0 = None
    if spec.layout != "dense":
        nxt = b.add(batch, c1)
        q_row0 = b.global_load_i32(p["cu_seqlens_q"], batch)
        sq = b.sub(b.global_load_i32(p["cu_seqlens_q"], nxt), q_row0)
        with b.scf_if(b.cmp_ge(q_start, sq)):
            b.ret()  # past this sequence's queries
        if spec.layout == "ragged":
            k_row0 = b.global_load_i32(p["cu_seqlens_k"], batch)
            sk = b.sub(b.global_load_i32(p["cu_seqlens_k"], nxt), k_row0)
        else:
            sk = b.global_load_i32(p["seqused_k"], batch)

    def slice_rsrc(
        ptr, tensor, head_idx, rows, width, elem_bytes, enabled=None, row0=None
    ):
        """Resource over one (batch, head) slice: ``(rows - 1) * token stride
        + width`` elements, empty when there are no rows (or not enabled). A
        packed sequence starts at token ``row0`` instead of at its batch."""
        if row0 is None:
            first = b.mul(batch, p[f"stride_{tensor}_batch"])
        else:
            first = b.mul(row0, p[f"stride_{tensor}_token"])
        base = b.add(first, b.mul(head_idx, p[f"stride_{tensor}_head"]))
        sliced = b.global_ptr_add(ptr, b.mul(base, ci(elem_bytes)))
        extent = b.add(b.mul(b.sub(rows, c1), p[f"stride_{tensor}_token"]), width)
        nbytes = b.mul(extent, ci(elem_bytes))
        live = b.cmp_gt(rows, c0)
        if enabled is not None:
            live = b.land(live, enabled)
        nbytes = b.select(live, nbytes, c0)
        return b.buffer_rsrc(sliced, nbytes), nbytes

    def page_rsrc(ptr, tensor, head_idx):
        """Paged K / V: the whole cache at this head (the binding bounds the
        cache to I32 bytes; dead rows are masked per load)."""
        off = b.mul(b.mul(head_idx, p[f"stride_{tensor}_head"]), ci(kv_eb))
        return b.buffer_rsrc(b.global_ptr_add(ptr, off), ci(0x7FFFFFFF))

    hdq, hdv = p["head_dim_q"], p["head_dim_v"]
    kv_eb = 1 if spec.kv_dtype else 2
    q_rsrc, _ = slice_rsrc(p["Q"], "q", head, sq, hdq, 2, row0=q_row0)
    if spec.layout == "paged":
        k_rsrc, k_bytes = page_rsrc(p["K"], "k", kv_head), None
        v_rsrc, v_bytes = page_rsrc(p["V"], "v", v_head), None
    else:
        k_rsrc, k_bytes = slice_rsrc(p["K"], "k", kv_head, sk, hdq, kv_eb, row0=k_row0)
        v_rsrc, v_bytes = slice_rsrc(p["V"], "v", v_head, sk, hdv, kv_eb, row0=k_row0)
    o_rsrc, _ = slice_rsrc(p["O"], "o", head, sq, hdv, 2, row0=q_row0)
    lse_rsrc, _ = slice_rsrc(
        p["LSE"],
        "lse",
        head,
        sq,
        c1,
        4,
        enabled=b.cmp_ne(p["write_lse"], c0),
        row0=q_row0,
    )
    bias_rsrc = bias_lead = bias_bytes = None
    if spec.bias:
        eb = 4 if spec.bias == "f32" else 2
        base = b.add(b.mul(batch, p["bias_stride_b"]), b.mul(head, p["bias_stride_h"]))
        sliced = b.global_ptr_add(p["attn_bias_ptr"], b.mul(base, ci(eb)))
        extent = b.add(b.mul(b.sub(sq, c1), p["bias_stride_q"]), sk)
        live = b.land(b.cmp_gt(sq, c0), b.cmp_gt(sk, c0))
        nbytes = b.mul(extent, ci(eb))
        if spec.bias == "q":
            # A Q-dtype slice may start or end mid-dword, and the hardware drops a
            # loaded dword that ends past num_records: the resource starts at the
            # dword below the slice and spans whole dwords (bias_lead elements in).
            lead_bytes = b.land(b.trunc(b.global_ptr_to_i64(sliced), I32), ci(2))
            sliced = b.global_ptr_add(
                sliced, b.sub(b.const_i64(0), b.zext(lead_bytes, I64))
            )
            nbytes = b.land(b.add(b.add(nbytes, lead_bytes), ci(3)), ci(-4))
            bias_lead = b.lshr(lead_bytes, c1)
        bias_bytes = b.select(live, nbytes, c0)
        bias_rsrc = b.buffer_rsrc(sliced, bias_bytes)

    # ---- score features ----
    qq_rsrc = sink = slope = table_row = None
    if spec.use_qq_bias:
        rows, cols = p["qq_bias_rows"], p["qq_bias_cols"]
        extent = b.add(b.mul(b.sub(rows, c1), p["qq_bias_stride"]), cols)
        live = b.land(b.cmp_gt(rows, c0), b.cmp_gt(cols, c0))
        nbytes = b.select(live, b.mul(extent, ci(4)), c0)
        qq_rsrc = b.buffer_rsrc(p["qq_bias_ptr"], nbytes)
    if spec.use_sinks or spec.use_alibi:
        log2e = b.const_f32(1.4426950408889634)
    if spec.use_sinks:
        elem = BF16 if spec.dtype == "bf16" else F16
        raw = b.global_load(p["sink_ptr"], head, elem, align=2)
        sink = b.fmul(b.cast_to_f32(raw), log2e)
    if spec.use_alibi:
        raw = b.global_load(p["alibi_slopes_ptr"], head, F32, align=4)
        slope = b.fmul(raw, log2e)
    if spec.layout == "paged":
        table_row = b.mul(batch, p["block_table_stride"])

    causal = spec.mask == "causal"
    positional = spec.use_alibi or spec.use_qq_bias
    diag = None
    if causal or positional:
        diag = b.select(b.cmp_ne(p["bottom_right"], c0), b.sub(sk, sq), c0)
    rdna_attention_fwd_body(
        b,
        cfg,
        q_rsrc=q_rsrc,
        k_rsrc=k_rsrc,
        v_rsrc=v_rsrc,
        o_rsrc=o_rsrc,
        lse_rsrc=lse_rsrc,
        stride_q=p["stride_q_token"],
        stride_k=p["stride_k_token"],
        stride_v=p["stride_v_token"],
        stride_o=p["stride_o_token"],
        stride_lse=p["stride_lse_token"],
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        head_dim_q=hdq,
        head_dim_v=hdv,
        seqlen_q=sq,
        seqlen_k=sk,
        q_start=q_start,
        scale_log2=p["scale_log2"],
        diag_offset=diag if causal else None,
        window_left=p["window_left"] if causal else None,
        window_right=p["window_right"] if causal else None,
        bias_rsrc=bias_rsrc,
        stride_bias_q=p.get("bias_stride_q"),
        bias_lead=bias_lead,
        bias_bytes=bias_bytes,
        softcap=p.get("softcap"),
        sink_log2=sink,
        alibi_slope_log2=slope,
        context=diag if positional else None,
        qq_rsrc=qq_rsrc,
        qq_rows=p.get("qq_bias_rows"),
        qq_cols=p.get("qq_bias_cols"),
        qq_stride=p.get("qq_bias_stride"),
        k_scale=p.get("k_scale"),
        v_scale=p.get("v_scale"),
        block_table=p.get("block_table"),
        table_row=table_row,
        stride_k_block=p.get("stride_k_block"),
        stride_v_block=p.get("stride_v_block"),
        num_blocks=p.get("num_blocks"),
    )
    b.ret()
    return b.kernel


def wmma_fmha_fwd_v2_grid(
    spec: WmmaFmhaFwdV2Spec, *, batch: int, seqlen_q: int, num_query_heads: int
):
    """``((grid_x, 1, 1), (threads, 1, 1))`` covering one problem."""
    n_qt = -(-seqlen_q // spec.bm)
    n = batch * n_qt * num_query_heads
    if not 0 < n <= 0x7FFFFFFF:
        raise ValueError(f"grid of {n} CTAs is empty or does not fit I32")
    return (n, 1, 1), (spec.threads, 1, 1)


def wmma_fmha_fwd_v2_signature(spec: WmmaFmhaFwdV2Spec):
    """The kernel ABI as ``({name, type}, ...)`` (no body emission)."""
    b = IRBuilder(spec.kernel_name())
    _declare_params(b, spec)
    return tuple({"name": q.name, "type": q.type.name} for q in b.kernel.params)
