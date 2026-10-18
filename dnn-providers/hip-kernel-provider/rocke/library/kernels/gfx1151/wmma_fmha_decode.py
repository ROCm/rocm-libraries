# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split-KV, GQA-packed WMMA attention decode for RDNA (gfx11 / gfx12), FP16/BF16.

Serves the small-``Sq`` end of SDPA forward: decode (``Sq = 1``), speculative
decode (``Sq = 4..8``) and cross attention with one query row, for
``[batch, token, head, dim]``-strided Q/K/V/O, ragged (varlen) batches and paged
KV caches. One object per (dtype, head-dim bucket, bias kind, feature flags);
everything else is a kernel argument.

* **Rows.** Query head ``h`` reads K head ``h / (Hq / Hkv)`` and V head
  ``h / (Hq / Hv)``; one of ``Hkv`` / ``Hv`` divides the other. The query heads
  are cut into ``P = max(Hkv, Hv)`` pack groups of ``g = Hq / P`` consecutive
  heads, each reading one K head and one V head, and a group's heads and query
  positions are packed into 16-row WMMA tiles: packed row ``r`` of group ``pg``
  is query head ``pg * g + r // Sq`` at position ``r % Sq``, valid while
  ``r < g * Sq``. Each K/V head is read once per row tile, not once per query
  head. ``ceil(g * Sq / 16)`` row tiles cover the group.
* **Scores transposed.** ``S^T = K @ Q^T`` puts the query row on the lane
  (``lane % 16``) and the keys on the accumulator slots, so the softmax row
  reduction is an in-lane tree plus one ``permlanex16``. ``O^T = V^T @ P``
  keeps the output in the same per-lane distribution as the softmax state. On
  gfx12 the score accumulator *is* the P operand; on gfx11 P is transposed in
  registers (``permlanex16`` + ``v_perm_b32``, CK ``PermuteWarpGemmCToA``).
* **Split-KV.** The workgroup's key range (the window's left edge to the
  diagonal, for the union of its rows) is cut into 16-key tiles shared by
  ``num_splits`` workgroups (a runtime argument the host picks from the shape
  and the CU count, :func:`wmma_fmha_decode_num_splits`): split ``s`` takes
  every ``num_splits``-th run of ``num_waves`` tiles, so the splits of a head
  sweep the keys as one front (one DRAM stream, not ``num_splits``). With one
  split the kernel writes O (and the natural-log LSE) directly; with more, each
  split writes a normalized f32 partial O and its log2-domain LSE to a
  workspace (:func:`wmma_fmha_decode_workspace`). The partials are combined
  either by the last split of each group to finish (fused merge: a non-null
  ``WS_CNT`` arrival counter) or by :func:`build_wmma_fmha_decode_merge`;
  :func:`wmma_fmha_decode_fused_merge` picks per launch.
  :func:`build_wmma_fmha_decode_module` packages both kernels as one code
  object (the merge kernel is the ``"merge"`` companion).
* **Waves.** ``num_waves`` waves per workgroup hold the same rows and take
  adjacent key tiles of each run; with ``lockstep`` a workgroup barrier per
  iteration keeps them there (else they drift apart into separate DRAM
  streams). K goes straight from global memory into the WMMA A operand (on gfx11
  each lane loads its own 8-column half and ``permlanex16`` fills the
  duplicated half); V is transposed through a wave-private LDS region (two
  keys per dword, written with ``v_perm_b32`` pairs, read back as 128-bit
  WMMA operands). The waves combine ``(m, l, O)`` through the same LDS at the
  end, each wave finishing a share of the output columns.
* **Masks.** Causal (top-left or bottom-right) and sliding-window bounds are
  runtime arguments (``window_left`` / ``window_right``, ``-1`` = unbounded,
  ``window_right = 0`` = causal) applied as one unsigned span compare per
  score; tiles outside the range are not visited. An additive bias
  (``bias = "q"`` in the query dtype or ``"f32"``) is a compile-time variant.
* **Head dims.** ``head_size`` is the bucket (64, 128 or 256): the largest
  Q/K and V width served. The actual widths (``head_dim_q``, ``head_dim_v``,
  multiples of 8) are arguments; K and V column chunks past them are loaded
  out of the buffer range (zero, no memory traffic), O columns past
  ``head_dim_v`` are not stored, and K/V reads stay inside a bounded buffer
  resource per (batch, KV head) slice (tail keys read zero). Above 128,
  ``dv_split`` workgroups each produce a slice of O's columns.
* **Per-family codegen.** :func:`resolve_wmma_fmha_decode_spec` fixes dv split,
  Q placement, V store order, prefetch, lockstep and scheduler per WMMA family
  and bucket.
* **Score features** (compile-time flags, runtime values; the prefill kernel's
  semantics and argument order): ``softcap`` (``cap * tanh(s / cap)`` on the
  scaled score), ``alibi`` (``+ slope[h] * (key - context)``), ``qq_bias``
  (``+ qq_bias[query, key - context]`` inside its bounds) and the additive
  ``bias``, in that order, before the masks. Any of them moves the loop's scores
  to natural-log units. ``sinks`` adds one per-head logit to the softmax
  denominator only: split 0 folds it into its partial (so the merge counts it
  once), and a row without visible keys yields O = 0 and LSE = sink.
* **FP8 KV** (``kv_dtype = "fp8e4m3"``, OCP E4M3FN): K / V bytes are decoded in
  registers to the query dtype exactly (gfx12: ``v_cvt_pk_f32_fp8``; gfx11: integer
  ops giving value / 256, NaN kept); ``k_scale`` (times that unit) scales Q.K before
  every feature and ``v_scale`` the normalized O.
* **Layouts.** ``ragged``: Q / O and K / V packed per sequence through
  ``cu_seqlens_q`` / ``cu_seqlens_k``. ``paged``: Q / O packed, K / V in
  ``[page, page_block_size, head, dim]`` caches through ``block_table`` with
  ``seqused_k`` lengths. Both take the maxima as ``seqlen_q`` / ``seqlen_k``
  (grid, workspace) and each sequence's own lengths for masks and positions;
  pages of at least one key tile cost one wave-uniform table read per tile,
  smaller pages one read per loaded row. A ``block_table`` entry outside
  ``[0, num_blocks)`` (``num_blocks``: the caches' page count, a runtime
  argument) is contained: its keys read as zero K / V rows and nothing outside
  the caches is addressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I16,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    VectorType,
)

__all__ = [
    "WmmaFmhaDecodeSpec",
    "build_wmma_fmha_decode",
    "build_wmma_fmha_decode_merge",
    "build_wmma_fmha_decode_module",
    "is_valid_spec",
    "wmma_fmha_decode_lds_bytes",
    "wmma_fmha_decode_bucket",
    "wmma_fmha_decode_kv_range",
    "wmma_fmha_decode_pack_groups",
    "wmma_fmha_decode_row_tiles",
    "wmma_fmha_decode_num_splits",
    "wmma_fmha_decode_grid",
    "wmma_fmha_decode_merge_grid",
    "wmma_fmha_decode_workspace",
    "wmma_fmha_decode_fused_merge",
    "wmma_fmha_decode_signature",
    "wmma_fmha_decode_merge_signature",
    "resolve_wmma_fmha_decode_spec",
    "DECODE_ROWS",
    "DECODE_TILE",
    "MERGE_THREADS",
]

DECODE_ROWS = 16  # packed query rows per WMMA tile
DECODE_TILE = 16  # keys per wave per key-loop iteration (one 16-key WMMA tile)
MERGE_THREADS = 64  # merge workgroup; each thread owns 4 consecutive output columns
_MERGE_COLS = 4
_MERGE_UNROLL = 8  # split partials loaded per merge-loop step
_MAX_SPLITS = 64
_FUSED_MAX_SPLITS = 4
_SPLIT_KEYS = 32  # key granularity of the split heuristic (two key-loop tiles)
_V_PAD = 4  # LDS V^T row padding in dwords
_MAX_LDS_BYTES = 65536
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_FP8_INT_UNIT = 256.0  # the integer FP8 decode yields value / 256 (see _fp8x8)
_BUCKETS = (64, 128, 256)
# Smallest width each bucket serves: K columns below it never need zeroing.
_BUCKET_MIN = {64: 16, 128: 72, 256: 136}


@dataclass(frozen=True)
class WmmaFmhaDecodeSpec:
    """One split-KV decode object.

    Baked axes: ``dtype``, ``head_size`` (bucket), ``bias`` and the feature flags
    (``softcap``, ``sinks``, ``alibi``, ``qq_bias``, ``kv_dtype``, ``layout`` with
    ``page_block_size``). The remaining fields are codegen knobs; their
    zero/``None``/``"auto"`` values resolve per WMMA family through
    :func:`resolve_wmma_fmha_decode_spec` (one measured setting per family and
    bucket), which also records the family in ``wmma_family``. Builders resolve
    the spec themselves; host helpers that depend on ``dv_split`` need a resolved
    spec (or an explicit ``dv_split``).
    """

    head_size: int
    dtype: str = "fp16"
    bias: str = ""  # "" | "q" (query dtype) | "f32"
    softcap: bool = (
        False  # s = cap * tanh(s / cap) on the scaled score (runtime ``softcap``)
    )
    sinks: bool = (
        False  # one per-query-head logit in the denominator only (``sink_ptr``)
    )
    alibi: bool = False  # s += slope[h] * (key - context) (``alibi_slopes_ptr``)
    qq_bias: bool = False  # s += qq_bias[query, key - context] inside its bounds
    kv_dtype: str = (
        ""  # "" (query dtype) | "fp8e4m3" (OCP E4M3FN K/V with k_scale / v_scale)
    )
    layout: str = "dense"  # "dense" | "ragged" (cu_seqlens_q/k) | "paged" (block_table)
    page_block_size: int = 0  # paged: keys per page, a power of two
    num_waves: int = 4  # waves per workgroup (1, 2, 4 or 8)
    dv_split: int = (
        0  # workgroups sharing one O row, head_size / dv_split columns each; 0 = auto
    )
    q_mode: str = (
        ""  # "reg" (resident registers) | "lds" (staged once per workgroup); "" = auto
    )
    v_store_early: Optional[bool] = (
        None  # V -> LDS before (True) / after the softmax; None = auto
    )
    prefetch: Optional[bool] = (
        None  # distance-1 K/V prefetch across iterations; None = auto
    )
    lockstep: Optional[bool] = (
        None  # workgroup barrier per key-loop iteration; None = auto
    )
    scheduler_strategy: Optional[str] = (
        "auto"  # "auto" | "default" / None (LLVM default) | a strategy
    )
    waves_per_eu: int = 0  # occupancy hint; 0 = none
    fp8_cvt: Optional[bool] = (
        None  # FP8 KV: hardware FP8 conversion (gfx12 only); None = auto
    )
    tanh_fast: Optional[bool] = (
        None  # softcap: exp2 + rcp tanh (else the libm form); None = auto
    )
    wmma_family: str = ""  # "" (unresolved) | "gfx11" | "gfx12": set by resolution
    name: str = "rocke_wmma_fmha_decode"

    def __post_init__(self) -> None:
        from rocke.core.codegen_policy import normalize_scheduler_strategy

        if self.head_size not in _BUCKETS:
            raise ValueError(
                f"head_size bucket must be one of {_BUCKETS} (got {self.head_size})"
            )
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError(f"dtype must be fp16 or bf16 (got {self.dtype!r})")
        if self.bias not in ("", "q", "f32"):
            raise ValueError(f"bias must be '', 'q' or 'f32' (got {self.bias!r})")
        if self.kv_dtype not in ("", "fp8e4m3"):
            raise ValueError(
                f"kv_dtype must be '' or 'fp8e4m3' (got {self.kv_dtype!r})"
            )
        if self.layout not in ("dense", "ragged", "paged"):
            raise ValueError(
                f"layout must be dense, ragged or paged (got {self.layout!r})"
            )
        ps = self.page_block_size
        if self.layout == "paged":
            if ps <= 0 or ps & (ps - 1):
                raise ValueError(
                    f"paged page_block_size must be a positive power of two (got {ps})"
                )
        elif ps:
            raise ValueError("page_block_size is only meaningful for the paged layout")
        if self.num_waves not in (1, 2, 4, 8):
            raise ValueError(f"num_waves must be 1, 2, 4 or 8 (got {self.num_waves})")
        # Each V load covers 32 columns of the workgroup's dv slice.
        if self.dv_split not in (0, 1, 2, 4) or (
            self.dv_split and self.head_size // self.dv_split < 32
        ):
            raise ValueError(
                f"invalid dv_split {self.dv_split} for head_size {self.head_size}"
            )
        if self.q_mode not in ("", "reg", "lds"):
            raise ValueError(f"q_mode must be '', 'reg' or 'lds' (got {self.q_mode!r})")
        if self.waves_per_eu < 0:
            raise ValueError(f"waves_per_eu must be >= 0 (got {self.waves_per_eu})")
        if self.wmma_family not in ("", "gfx11", "gfx12"):
            raise ValueError(
                f"wmma_family must be '', 'gfx11' or 'gfx12' (got {self.wmma_family!r})"
            )
        if self.scheduler_strategy not in ("auto", "default"):
            normalize_scheduler_strategy(self.scheduler_strategy)

    @property
    def resolved_dv_split(self) -> int:
        if not self.dv_split:
            raise ValueError(
                "dv_split is family-dependent: resolve_wmma_fmha_decode_spec first"
            )
        return self.dv_split

    @property
    def dv_tile(self) -> int:
        return self.head_size // self.resolved_dv_split

    @property
    def block_size(self) -> int:
        return 32 * self.num_waves

    @property
    def packed(self) -> bool:
        """Q / O / LSE rows packed per sequence through ``cu_seqlens_q`` (ragged, paged)."""
        return self.layout != "dense"

    @property
    def natural_scores(self) -> bool:
        """Scores move to natural-log units in the loop (an additive term or softcap)."""
        return bool(self.bias or self.softcap or self.alibi or self.qq_bias)

    # ---- hipDNN catalog metadata ----
    # hipkernel:Gfx1151WmmaAttention ships this spec beside WmmaFmhaFwdV2Spec, and
    # its IngestorGenerator specialization binds each matcher field to one accessor
    # both specs define (configs/gfx1151_wmma_attention.yaml).
    def catalog_kind(self) -> str:
        """``"decode"``: this object runs the split-KV decode (with its merge)."""
        return "decode"

    def catalog_head_size(self) -> int:
        """Q/K head-width bucket."""
        return self.head_size

    def catalog_head_size_v(self) -> int:
        """V/O head-width bucket (the same bucket as Q/K)."""
        return self.head_size

    def catalog_mask(self) -> str:
        """``"runtime"``: causal and window bounds are all kernel arguments."""
        return "runtime"

    def catalog_bias(self) -> str:
        """Additive bias kind: ``"none"``, ``"f32"`` or ``"q"``."""
        return self.bias or "none"

    def catalog_block_size(self) -> int:
        """Threads per workgroup."""
        return self.block_size

    def catalog_block_m(self) -> int:
        """Packed query rows per workgroup tile."""
        return DECODE_ROWS

    def catalog_dv_split(self) -> int:
        """Workgroups sharing one O row; the spec must state it (or be resolved)."""
        return self.resolved_dv_split

    def kernel_name(self) -> str:
        """Symbol name; a resolved spec names every resolved knob (``t16`` / ``khalf``
        / ``si`` mark the fixed 16-key tile, gfx11 half-K loads and interleaved splits).
        """
        parts = [self.name, f"d{self.head_size}", self.dtype, f"w{self.num_waves}"]
        if self.bias:
            parts.append(f"bias{self.bias}")
        for flag, tag in (
            (self.softcap, "cap"),
            (self.sinks, "sink"),
            (self.alibi, "alibi"),
            (self.qq_bias, "qq"),
        ):
            if flag:
                parts.append(tag)
        if self.kv_dtype:
            parts.append("kvfp8")
        if self.layout == "ragged":
            parts.append("ragged")
        elif self.layout == "paged":
            parts.append(f"paged{self.page_block_size}")
        if self.dv_split:
            parts.append(f"dvs{self.dv_split}")
        if self.wmma_family:
            parts.append(f"t{DECODE_TILE}")
        if self.q_mode:
            parts.append(f"q{self.q_mode}")
        if self.wmma_family == "gfx11":
            parts.append("khalf")
        if self.v_store_early is not None:
            parts.append("vearly" if self.v_store_early else "vlate")
        if self.prefetch is not None:
            parts.append("pf" if self.prefetch else "nopf")
        if self.wmma_family:
            parts.append("si")
        if self.lockstep:
            parts.append("ls")
        if self.scheduler_strategy != "auto":
            parts.append((self.scheduler_strategy or "default").replace("-", ""))
        if self.waves_per_eu:
            parts.append(f"wpe{self.waves_per_eu}")
        if self.kv_dtype and self.fp8_cvt is not None:
            parts.append("cvt8" if self.fp8_cvt else "int8")
        if self.softcap and self.tanh_fast is not None:
            parts.append("tanhf" if self.tanh_fast else "tanhm")
        return "_".join(parts)


def _layout(arch: str, dtype: str):
    """(op_id, gfx12) for the dtype's wave32 WMMA atom on ``arch``."""
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    elem = "bf16" if dtype == "bf16" else "f16"
    if target.wave_size == 32:
        gfx12 = f"wmma_gfx12_f32_16x16x16_{elem}"
        if target.mma.by_op_id(gfx12) is not None:
            return gfx12, True
        gfx11 = f"wmma_f32_16x16x16_{elem}"
        if target.mma.by_op_id(gfx11) is not None:
            return gfx11, False
    raise ValueError(f"{arch} has no wave32 RDNA WMMA {elem} atom")


def resolve_wmma_fmha_decode_spec(
    spec: WmmaFmhaDecodeSpec, *, arch: str
) -> WmmaFmhaDecodeSpec:
    """Fill the spec's auto knobs with the setting measured for ``arch``'s WMMA family.

    gfx11 (operands duplicated across lane halves, so K and Q cost twice the registers):
    own-half K loads, max-ilp, V stored after the softmax, no cross-tile prefetch (its
    carried registers spill under max-ilp); D256 stages Q in LDS and splits O over 2
    workgroups. gfx12: V stored before the softmax, max-ilp up to D128; distance-1 K/V
    prefetch at D64 and D256 (at D128 it buys nothing and leaves no VGPR headroom); D256
    splits O over 4 workgroups (64 columns each) and uses the LLVM default scheduler.

    DRAM locality (cold cache): the splits of a head interleave their key tiles on both
    families, and on gfx12 a barrier per key-loop iteration keeps a workgroup's waves on
    adjacent tiles. Without the barrier the waves drift apart and every workgroup becomes
    several scattered DRAM streams, which costs cold-cache throughput.

    Feature objects keep these knobs except: FP8 K/V prefetches on gfx11 below D256 (its
    raw K/V carry half the registers) and decodes with ``v_cvt_pk_f32_fp8`` on gfx12
    except at D128 (``fp8_cvt``); softcap uses the exp2/rcp tanh except on gfx11 D256
    without FP8 (``tanh_fast``).

    Explicit knobs are kept (``fp8_cvt`` only applies on gfx12). Resolving is idempotent;
    a spec already resolved for the other family is rejected.
    """
    from dataclasses import replace

    _, gfx12 = _layout(arch, spec.dtype)
    family = "gfx12" if gfx12 else "gfx11"
    if spec.wmma_family not in ("", family):
        raise ValueError(
            f"spec resolved for {spec.wmma_family} cannot build for {arch} ({family})"
        )
    d256 = spec.head_size > 128
    if gfx12:
        auto = dict(
            dv_split=4 if d256 else 1,
            q_mode="reg",
            v_store_early=True,
            prefetch=spec.head_size != 128,
            lockstep=True,
            scheduler_strategy=None if d256 else "max-ilp",
        )
    else:
        auto = dict(
            dv_split=2 if d256 else 1,
            q_mode="lds" if d256 else "reg",
            v_store_early=False,
            # FP8 K/V carry half the registers across the prefetch: on below D256.
            prefetch=bool(spec.kv_dtype) and not d256,
            lockstep=False,
            scheduler_strategy="max-ilp",
        )
    fields = dict(
        dv_split=spec.dv_split or auto["dv_split"],
        q_mode=spec.q_mode or auto["q_mode"],
        v_store_early=(
            auto["v_store_early"] if spec.v_store_early is None else spec.v_store_early
        ),
        prefetch=auto["prefetch"] if spec.prefetch is None else spec.prefetch,
        lockstep=auto["lockstep"] if spec.lockstep is None else spec.lockstep,
        # gfx12 D128 sits at the VGPR limit: the hardware conversion's F32 temporaries spill
        # there where the integer decode spills far fewer registers and is faster.
        fp8_cvt=(
            (gfx12 and spec.head_size != 128)
            if spec.fp8_cvt is None
            else (spec.fp8_cvt and gfx12)
        ),
        # The exp2/rcp softcap spills gfx11 D256 (already at 256 VGPRs) under max-ilp, so it
        # is slower there; with FP8 K/V (fewer live K/V registers) it wins there too.
        tanh_fast=(
            (gfx12 or not d256 or bool(spec.kv_dtype))
            if spec.tanh_fast is None
            else spec.tanh_fast
        ),
        scheduler_strategy=(
            auto["scheduler_strategy"]
            if spec.scheduler_strategy == "auto"
            else (
                None
                if spec.scheduler_strategy == "default"
                else spec.scheduler_strategy
            )
        ),
        wmma_family=family,
    )
    return replace(spec, **fields)


def _lds_dwords(spec: WmmaFmhaDecodeSpec) -> Tuple[int, int]:
    """(per-wave region, Q staging) LDS dwords of a resolved spec."""
    rs = DECODE_TILE // 2 + _V_PAD  # V^T row stride: one dword holds two keys
    rec = (
        64 + (spec.dv_tile // 16) * 256
    )  # combine record: (m, l) per lane + O fragments
    q = DECODE_ROWS * (spec.head_size // 2 + 4) if spec.q_mode == "lds" else 0
    return max(spec.dv_tile * rs, rec), q


def wmma_fmha_decode_lds_bytes(spec: WmmaFmhaDecodeSpec, *, arch: str) -> int:
    """Static LDS bytes one decode workgroup of ``spec`` allocates on ``arch``."""
    spec = resolve_wmma_fmha_decode_spec(spec, arch=arch)
    region, q = _lds_dwords(spec)
    return 4 * (spec.num_waves * region + q)


def is_valid_spec(spec: WmmaFmhaDecodeSpec, *, arch: str) -> Tuple[bool, str]:
    """``(ok, reason)``: whether ``spec`` builds for ``arch`` (wave32 WMMA atom, LDS)."""
    try:
        lds = wmma_fmha_decode_lds_bytes(spec, arch=arch)
    except (KeyError, ValueError) as e:
        return False, str(e)
    if lds > _MAX_LDS_BYTES:
        return False, f"LDS {lds} bytes exceed {_MAX_LDS_BYTES}"
    return True, "ok"


def _elem(spec):
    return BF16 if spec.dtype == "bf16" else F16


def _declare_params(b: IRBuilder, spec: WmmaFmhaDecodeSpec):
    elem = _elem(spec)
    p = {}
    kv_elem, kv_align = (FP8E4M3, 1) if spec.kv_dtype else (elem, 16)
    p["Q"] = b.param(
        "Q", PtrType(elem, "global"), noalias=True, readonly=True, align=16
    )
    for name in ("K", "V"):
        p[name] = b.param(
            name,
            PtrType(kv_elem, "global"),
            noalias=True,
            readonly=True,
            align=kv_align,
        )
    p["O"] = b.param(
        "O", PtrType(elem, "global"), noalias=True, writeonly=True, align=16
    )
    p["LSE"] = b.param(
        "LSE", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
    )
    p["WS_O"] = b.param("WS_O", PtrType(F32, "global"), noalias=True, align=16)
    p["WS_LSE"] = b.param("WS_LSE", PtrType(F32, "global"), noalias=True, align=4)
    p["WS_CNT"] = b.param("WS_CNT", PtrType(I32, "global"), noalias=True, align=4)
    p["scale_log2"] = b.param("scale_log2", F32)
    for name in (
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
        "num_splits",
    ):
        p[name] = b.param(name, I32)
    for tensor in ("q", "k", "v", "o", "lse"):
        for axis in ("batch", "token", "head"):
            p[f"stride_{tensor}_{axis}"] = b.param(f"stride_{tensor}_{axis}", I32)
    # Feature arguments in the v2 prefill kernel's order (wmma_fmha_fwd_v2).
    if spec.softcap:
        p["softcap"] = b.param("softcap", F32)
    if spec.sinks:
        p["sink_ptr"] = b.param(
            "sink_ptr", PtrType(elem, "global"), noalias=True, readonly=True, align=2
        )
    if spec.alibi:
        p["alibi_slopes_ptr"] = b.param(
            "alibi_slopes_ptr",
            PtrType(F32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
    if spec.qq_bias:
        p["qq_bias_ptr"] = b.param(
            "qq_bias_ptr", PtrType(F32, "global"), noalias=True, readonly=True, align=4
        )
        for name in ("qq_bias_rows", "qq_bias_cols", "qq_bias_stride"):
            p[name] = b.param(name, I32)
    i32p = PtrType(I32, "global")
    if spec.layout == "ragged":
        for name in ("cu_seqlens_q", "cu_seqlens_k"):
            p[name] = b.param(name, i32p, noalias=True, readonly=True, align=4)
    elif spec.layout == "paged":
        for name in ("cu_seqlens_q", "seqused_k", "block_table"):
            p[name] = b.param(name, i32p, noalias=True, readonly=True, align=4)
        for name in (
            "block_table_stride",
            "stride_k_block",
            "stride_v_block",
            "num_blocks",
        ):
            p[name] = b.param(name, I32)
    if spec.kv_dtype:
        p["k_scale"] = b.param("k_scale", F32)
        p["v_scale"] = b.param("v_scale", F32)
    if spec.bias:
        # Same names as the prefill kernel's bias; keys are contiguous (stride 1).
        bias_elem = F32 if spec.bias == "f32" else elem
        p["attn_bias_ptr"] = b.param(
            "attn_bias_ptr",
            PtrType(bias_elem, "global"),
            noalias=True,
            readonly=True,
            align=4 if spec.bias == "f32" else 2,
        )
        for axis in ("b", "h", "q"):
            p[f"bias_stride_{axis}"] = b.param(f"bias_stride_{axis}", I32)
    return p


def build_wmma_fmha_decode(spec: WmmaFmhaDecodeSpec, *, arch: str) -> KernelDef:
    """Build the split-KV decode kernel for ``arch`` (resolving ``spec`` first).

    Launch: grid :func:`wmma_fmha_decode_grid`, ``spec.block_size`` threads. The
    arguments are :func:`wmma_fmha_decode_signature`; ``WS_O`` / ``WS_LSE`` /
    ``WS_CNT`` follow :func:`wmma_fmha_decode_workspace` (unused with one split).
    """
    from rocke.core.codegen_policy import CodegenPolicy, apply_codegen_policy

    spec = resolve_wmma_fmha_decode_spec(spec, arch=arch)
    ok, reason = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid wmma_fmha_decode spec for {arch}: {reason}")
    op_id, gfx12 = _layout(arch, spec.dtype)
    elem = _elem(spec)
    NW = spec.num_waves
    T = DECODE_TILE
    NSUB = T // 16  # 16-key WMMA sub-tiles per iteration
    NP = T // 2  # V row pairs per tile (one LDS dword holds two keys)
    VG = 32 // NP  # lanes per V row pair: column groups of 8 per load
    D = spec.head_size
    DVS = spec.resolved_dv_split
    DVT = spec.dv_tile
    n_ks = D // 16
    n_dc = DVT // 16
    q_mode = spec.q_mode
    v_early = spec.v_store_early
    RS = NP + _V_PAD  # LDS V^T row stride (dwords)
    REG, _ = _lds_dwords(
        spec
    )  # per-wave LDS region: V^T staging, then the combine record

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    apply_codegen_policy(
        b.kernel, CodegenPolicy(scheduler_strategy=spec.scheduler_strategy)
    )
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, spec.waves_per_eu)
    p = _declare_params(b, spec)

    ci = b.const_i32
    c0, c1, c2 = ci(0), ci(1), ci(2)
    c16 = ci(16)
    f0 = b.const_f32(0.0)
    ninf = b.const_f32(float("-inf"))
    zero8 = b.zero_vec(elem, 8)

    Hq, Hkv = p["num_query_heads"], p["num_kv_heads"]
    Sq, Sk, ns = p["seqlen_q"], p["seqlen_k"], p["num_splits"]
    d_rt, dv_rt = p["head_dim_q"], p["head_dim_v"]

    # ---- workgroup -> (batch, split, pack group, dv slice, row tile); row tile fastest,
    # then the pack groups of one token range, then splits.
    Hv = p["num_v_heads"]
    n_pg = b.smax(Hkv, Hv)
    g = b.div(Hq, n_pg)
    rows = b.mul(g, Sq)
    n_rt = b.div(b.add(rows, ci(15)), c16)
    bid = b.block_id_x()
    rt = b.mod(bid, n_rt)
    rest = b.div(bid, n_rt)
    dvi = c0
    if DVS > 1:
        dvi = b.mod(rest, ci(DVS))
        rest = b.div(rest, ci(DVS))
    # The pack groups of one token range run together (adjacent BSHD rows).
    pg = b.mod(rest, n_pg)
    rest = b.div(rest, n_pg)
    split = b.mod(rest, ns)
    batch = b.div(rest, ns)
    kvh = b.div(pg, b.div(n_pg, Hkv))
    vh = b.div(pg, b.div(n_pg, Hv))
    dv0 = b.mul(dvi, ci(DVT))

    tid = b.thread_id_x()
    wave = b.readfirstlane(b.div(tid, ci(32)))
    lane = b.mod(tid, ci(32))
    col = b.mod(lane, c16)  # packed query row within the tile / lane's row or column
    half = b.div(lane, c16)
    lo_half = b.cmp_lt(lane, c16)

    # ---- per-sequence lengths. Packed layouts (ragged / paged) pass the maxima as
    # seqlen_q / seqlen_k (grid, workspace rows) and read each sequence's own lengths.
    if spec.packed:
        cuq = p["cu_seqlens_q"]
        q_start = b.global_load_i32(cuq, batch)
        Sq_b = b.sub(b.global_load_i32(cuq, b.add(batch, c1)), q_start)
        if spec.layout == "ragged":
            k_start = b.global_load_i32(p["cu_seqlens_k"], batch)
            Sk_b = b.sub(
                b.global_load_i32(p["cu_seqlens_k"], b.add(batch, c1)), k_start
            )
        else:
            Sk_b = b.global_load_i32(p["seqused_k"], batch)
        rows_b = b.mul(g, Sq_b)
        Sq_d = b.smax(
            Sq_b, c1
        )  # row -> (head, position) divisor; empty sequences have no rows
    else:
        Sq_b, Sk_b, rows_b, Sq_d = Sq, Sk, rows, Sq

    r = b.add(b.mul(rt, c16), col)
    row_ok = b.cmp_lt(r, rows_b)
    r = b.select(row_ok, r, c0)
    hq = b.add(b.mul(pg, g), b.div(r, Sq_d))
    qpos = b.mod(r, Sq_d)

    # ---- mask: this row's visible keys [first, end); the tile's union [lo, hi).
    ctx = b.select(b.cmp_ne(p["bottom_right"], c0), b.sub(Sk_b, Sq_b), c0)
    wl, wr = p["window_left"], p["window_right"]
    has_l = b.cmp_ge(wl, c0)
    has_r = b.cmp_ge(wr, c0)
    diag = b.add(qpos, ctx)
    first = b.select(has_l, b.smax(b.sub(diag, wl), c0), c0)
    end = b.select(has_r, b.smin(b.add(b.add(diag, wr), c1), Sk_b), Sk_b)
    span = b.smax(b.sub(end, first), c0)
    span = b.select(row_ok, span, c0)
    span64 = b.zext(span, I64)
    lo = b.select(has_l, b.smax(b.sub(ctx, wl), c0), c0)
    hi = b.select(has_r, b.smax(b.smin(b.add(b.add(ctx, wr), Sq_b), Sk_b), c0), Sk_b)
    if spec.packed:
        # A row tile past this sequence's rows visits no keys (and stores nothing).
        hi = b.select(b.cmp_lt(b.mul(rt, c16), rows_b), hi, c0)
    lo_t = b.div(lo, ci(T))
    hi_t = b.div(b.add(hi, ci(T - 1)), ci(T))
    # Split s takes every ns-th run of NW tiles: the splits of a head sweep the key range
    # together as one front instead of ns separate streams.
    step_v = b.mul(ns, ci(NW))
    first_t = b.add(lo_t, b.mul(split, ci(NW)))
    my_t0 = b.add(first_t, wave)
    my_end = hi_t
    wg_n = b.div(b.add(b.smax(b.sub(hi_t, first_t), c0), b.sub(step_v, c1)), step_v)
    if spec.lockstep:
        # Every wave runs the workgroup's trip count (a barrier per iteration keeps the
        # waves' streams together); tiles past the wave's end are masked and read nothing.
        n_it = wg_n
    else:
        n_it = b.div(b.add(b.smax(b.sub(my_end, my_t0), c0), b.sub(step_v, c1)), step_v)

    # ---- pointers: batch offsets in 64-bit, per-(batch, KV head) K/V slices. Packed
    # layouts address Q / O rows from the sequence's first token (batch strides unused).
    fp8 = bool(spec.kv_dtype)
    kv_eb = 1 if fp8 else 2  # K / V element bytes

    def kv_bytes(x):
        return x if kv_eb == 1 else b.mul(x, c2)

    def batch_ptr(ptr, stride, elem_bytes):
        off = b.mul(
            b.mul(b.zext(batch, I64), b.zext(stride, I64)), b.const_i64(elem_bytes)
        )
        return b.global_ptr_add(ptr, off)

    def token_ptr(ptr, start, stride, elem_bytes):
        off = b.mul(
            b.mul(b.zext(start, I64), b.zext(stride, I64)), b.const_i64(elem_bytes)
        )
        return b.global_ptr_add(ptr, off)

    if spec.packed:
        # An empty trailing sequence starts one past the last Q row; its (dead) Q loads
        # read the first row instead.
        q_base = b.select(b.cmp_gt(Sq_b, c0), q_start, c0)
        Qb = token_ptr(p["Q"], q_base, p["stride_q_token"], 2)
        Ob = token_ptr(p["O"], q_start, p["stride_o_token"], 2)
    else:
        Qb = batch_ptr(p["Q"], p["stride_q_batch"], 2)
        Ob = batch_ptr(p["O"], p["stride_o_batch"], 2)
    k_ss, v_ss = p["stride_k_token"], p["stride_v_token"]
    if spec.layout == "dense":
        k_base = lambda: batch_ptr(p["K"], p["stride_k_batch"], kv_eb)  # noqa: E731
        v_base = lambda: batch_ptr(p["V"], p["stride_v_batch"], kv_eb)  # noqa: E731
    elif spec.layout == "ragged":
        k_base = lambda: token_ptr(p["K"], k_start, k_ss, kv_eb)  # noqa: E731
        v_base = lambda: token_ptr(p["V"], k_start, v_ss, kv_eb)  # noqa: E731
    else:  # paged: the cache base; pages are added per tile / row
        k_base, v_base = (lambda: p["K"]), (lambda: p["V"])
    k_slice = b.global_ptr_add(k_base(), kv_bytes(b.mul(kvh, p["stride_k_head"])))
    v_slice = b.global_ptr_add(v_base(), kv_bytes(b.mul(vh, p["stride_v_head"])))
    sk1 = b.sub(Sk_b, c1)
    k_extent = kv_bytes(b.add(b.mul(sk1, k_ss), d_rt))
    v_extent = kv_bytes(b.add(b.mul(sk1, v_ss), b.add(dv0, ci(DVT))))
    v_extent = b.smin(v_extent, kv_bytes(b.add(b.mul(sk1, v_ss), dv_rt)))
    k_row_bytes = kv_bytes(k_ss)
    v_row_bytes = kv_bytes(v_ss)
    k_tile_bytes = b.mul(k_row_bytes, ci(T))
    v_tile_bytes = b.mul(v_row_bytes, ci(T))

    def tile_rsrc(slice_ptr, extent, tile_bytes, t, valid=None):
        """Buffer resource for key tile ``t``; an invalid tile gets no records (reads 0)."""
        off = b.mul(t, tile_bytes)
        nrec = b.smax(b.sub(extent, off), c0)
        if valid is not None:
            nrec = b.select(valid, nrec, c0)
        return b.buffer_rsrc(b.global_ptr_add(slice_ptr, off), nrec)

    paged = spec.layout == "paged"
    page_rows = (
        paged and spec.page_block_size < T
    )  # a tile spans pages: per-row lookups
    if paged:
        ps = spec.page_block_size
        page_shift = ci(ps.bit_length() - 1)
        page_mask = ci(ps - 1)
        bt_row = b.mul(batch, p["block_table_stride"])

        def page_of(key, valid):
            """Physical page holding logical ``key`` (table index 0 when not ``valid``)."""
            blk = b.lshr(key, page_shift)
            if valid is not None:
                blk = b.select(valid, blk, c0)
            return b.global_load_i32(p["block_table"], b.add(bt_row, blk))

        def page_ok(phys):
            """``phys`` names a page of the cache: ``0 <= phys < num_blocks``. The
            table's values are caller data; a page outside the cache reads as zeros."""
            return b.land(b.cmp_ge(phys, c0), b.cmp_lt(phys, p["num_blocks"]))

    def paged_tile_rsrc(slice_ptr, extent, row_bytes, blk_stride, t, valid):
        """A key tile is aligned and no larger than a page, so it lies in one page:
        one wave-uniform block-table read per tile moves the tile's resource base."""
        kv0_ = b.mul(t, ci(T))
        phys = b.readfirstlane(page_of(kv0_, valid))
        in_cache = page_ok(phys)
        page_off = b.mul(
            b.mul(b.zext(phys, I64), b.zext(blk_stride, I64)), b.const_i64(kv_eb)
        )
        page_off = b.add(
            page_off, b.zext(b.mul(b.land(kv0_, page_mask), row_bytes), I64)
        )
        nrec = b.smax(b.sub(extent, b.mul(kv0_, row_bytes)), c0)
        if valid is not None:
            nrec = b.select(valid, nrec, c0)
        nrec = b.select(in_cache, nrec, c0)
        return b.buffer_rsrc(b.global_ptr_add(slice_ptr, page_off), nrec)

    def k_rsrc(t, valid=None):
        if paged:
            return paged_tile_rsrc(
                k_slice, k_extent, k_row_bytes, p["stride_k_block"], t, valid
            )
        return tile_rsrc(k_slice, k_extent, k_tile_bytes, t, valid)

    def v_rsrc(t, valid=None):
        if paged:
            return paged_tile_rsrc(
                v_slice, v_extent, v_row_bytes, p["stride_v_block"], t, valid
            )
        return tile_rsrc(v_slice, v_extent, v_tile_bytes, t, valid)

    # Loop-invariant per-lane byte offsets inside a tile; column offsets are immediates.
    # Each lane loads its own 8-column half of a 16-column K step (gfx11: the duplicated
    # half comes from the lane^16 peer, see ``k_operand``).
    k_lane = [b.mul(b.add(ci(a * 16), col), k_row_bytes) for a in range(NSUB)]
    k_lane = [b.add(x, b.mul(half, c16 if kv_eb == 2 else ci(8))) for x in k_lane]
    # V loads: lane -> row pair vp (rows 2vp, 2vp+1) and column group vcg of 8 columns.
    vp = b.mod(lane, ci(NP))
    vcg = b.div(lane, ci(NP))
    v_lane0 = b.add(
        b.mul(b.mul(vp, c2), v_row_bytes), kv_bytes(b.add(b.mul(vcg, ci(8)), dv0))
    )
    v_lane = [v_lane0, b.add(v_lane0, v_row_bytes)]
    n_vj = DVT // (8 * VG)  # V loads per row per tile

    if page_rows:
        # Pages smaller than a tile: every K / V row a lane loads looks up its own page.
        # Rows past the sequence (or of an invalid tile) get an out-of-range offset.
        all_recs = ci(
            0x7FFF0000
        )  # the cache must fit I32 bytes; below the dead-row offset
        k_all = b.buffer_rsrc(k_slice, all_recs)
        v_all = b.buffer_rsrc(v_slice, all_recs)
        k_col_lane = kv_bytes(b.mul(half, ci(8)))
        v_col_lane = kv_bytes(b.add(b.mul(vcg, ci(8)), dv0))

        def row_offs(t, valid, rel_rows, blk_stride, row_stride, col_off):
            kv0_ = b.mul(t, ci(T))
            out = []
            for rel in rel_rows:
                key = b.add(kv0_, rel)
                ok = b.cmp_lt(key, Sk_b)
                if valid is not None:
                    ok = b.land(ok, valid)
                phys = page_of(key, ok)
                ok = b.land(ok, page_ok(phys))
                off = b.add(
                    b.mul(phys, blk_stride), b.mul(b.land(key, page_mask), row_stride)
                )
                # Dead rows: past all_recs, and + column bytes (< 1 KiB) stays below 2^31.
                out.append(b.select(ok, b.add(kv_bytes(off), col_off), ci(0x7FFF8000)))
            return out

        def k_row_offs(t, valid):
            rel = [b.add(ci(a * 16), col) for a in range(NSUB)]
            return row_offs(t, valid, rel, p["stride_k_block"], k_ss, k_col_lane)

        def v_row_offs(t, valid):
            rel = [b.mul(vp, c2), b.add(b.mul(vp, c2), c1)]
            return row_offs(t, valid, rel, p["stride_v_block"], v_ss, v_col_lane)

    # ---- Q: B operand of S^T = K Q^T (lane = query row, K = head dim).
    q_off = b.add(b.mul(hq, p["stride_q_head"]), b.mul(qpos, p["stride_q_token"]))

    def q_part(cbase, per_lane):
        """Eight Q columns from ``cbase`` (+8 * half when ``per_lane``)."""
        colv = ci(cbase)
        if per_lane:
            colv = b.add(colv, b.mul(half, ci(8)))
        ok = b.land(row_ok, b.cmp_lt(colv, d_rt))
        raw = b.global_load_vN(
            Qb, b.select(ok, b.add(q_off, colv), c0), elem, 8, align=16
        )
        return b.select(ok, raw, zero8)

    def q_pack(ks):
        if gfx12:
            return q_part(ks * 16, True)
        return b.vec_concat(q_part(ks * 16, False), q_part(ks * 16 + 8, False))

    q_packs = None  # "reg": resident operands; "lds": read back per tile
    if q_mode == "reg":
        q_packs = [q_pack(ks) for ks in range(n_ks)]
    else:
        # The workgroup's 16 x D Q tile, staged once and shared by the waves; each K step
        # reads its B operand back with 128-bit LDS loads.
        QRS = D // 2 + 4  # row stride in dwords (16-byte aligned rows)
        q_smem = b.smem_alloc(I32, [DECODE_ROWS * QRS], name_hint="dec_q")
        per_row = D // 8
        n_ch = DECODE_ROWS * per_row
        for k in range((n_ch + spec.block_size - 1) // spec.block_size):
            ch = b.add(tid, ci(k * spec.block_size))
            ch_ok = b.cmp_lt(ch, ci(n_ch))
            qrow = b.div(ch, ci(per_row))
            c8 = b.mod(ch, ci(per_row))
            rq = b.add(b.mul(rt, c16), qrow)
            rq_ok = b.cmp_lt(rq, rows_b)
            rq = b.select(rq_ok, rq, c0)
            colq = b.mul(c8, ci(8))
            ok = b.land(b.land(ch_ok, rq_ok), b.cmp_lt(colq, d_rt))
            off = b.add(
                b.add(
                    b.mul(b.add(b.mul(pg, g), b.div(rq, Sq_d)), p["stride_q_head"]),
                    b.mul(b.mod(rq, Sq_d), p["stride_q_token"]),
                ),
                colq,
            )
            raw = b.select(
                ok,
                b.global_load_vN(Qb, b.select(ok, off, c0), elem, 8, align=16),
                zero8,
            )
            with b.scf_if(ch_ok):
                b.smem_store_vN(
                    q_smem,
                    [b.add(b.mul(qrow, ci(QRS)), b.mul(c8, ci(4)))],
                    b.vec_bitcast(raw, VectorType(I32, 4)),
                    4,
                )
        b.sync()
        q_rd = b.mul(col, ci(QRS))
        if gfx12:
            q_rd = b.add(q_rd, b.mul(half, ci(4)))

        def q_pack(ks):  # noqa: F811 - the LDS form replaces the global one
            base = b.add(q_rd, ci(ks * 8))
            if gfx12:
                return b.vec_bitcast(
                    b.smem_load_vN(q_smem, base, dtype=I32, n=4), VectorType(elem, 8)
                )
            lo_ = b.smem_load_vN(q_smem, base, dtype=I32, n=4)
            hi_ = b.smem_load_vN(q_smem, b.add(base, ci(4)), dtype=I32, n=4)
            return b.vec_bitcast(b.vec_concat(lo_, hi_), VectorType(elem, 16))

    # ---- bias (compile-time variant): this lane's (head, position) bias row, keys
    # contiguous (key stride 1), read as vectors through a bounded per-batch resource.
    if spec.bias:
        bias_elem = F32 if spec.bias == "f32" else elem
        bias_bytes = 4 if spec.bias == "f32" else 2
        bias_base = b.global_ptr_add(
            p["attn_bias_ptr"],
            b.mul(
                b.mul(b.zext(batch, I64), b.zext(p["bias_stride_b"], I64)),
                b.const_i64(bias_bytes),
            ),
        )
        bias_extent = b.add(
            b.add(
                b.mul(b.sub(Hq, c1), p["bias_stride_h"]),
                b.mul(b.sub(Sq, c1), p["bias_stride_q"]),
            ),
            Sk,
        )
        bias_nbytes = b.mul(bias_extent, ci(bias_bytes))
        if spec.bias == "q":
            # A Q-dtype resource may start or end mid-dword, and the hardware drops a
            # loaded dword that ends past num_records: start at the dword below the
            # slice and span whole dwords (rows sit ``lead_bytes / 2`` elements in).
            lead_bytes = b.land(b.trunc(b.global_ptr_to_i64(bias_base), I32), ci(2))
            bias_base = b.global_ptr_add(
                bias_base, b.sub(b.const_i64(0), b.zext(lead_bytes, I64))
            )
            bias_nbytes = b.land(b.add(b.add(bias_nbytes, lead_bytes), ci(3)), ci(-4))
        bias_rsrc = b.buffer_rsrc(bias_base, bias_nbytes)
        bias_row = b.add(b.mul(hq, p["bias_stride_h"]), b.mul(qpos, p["bias_stride_q"]))
        if spec.bias == "q":
            # Each row is read from its dword-aligned key; ``bias_par`` is the 16-bit
            # half this lane's first key sits in (tile and sub-tile offsets are even).
            row = b.add(bias_row, b.lshr(lead_bytes, c1))
            if not gfx12:
                row = b.add(row, half)
            bias_par = b.land(row, c1)
            bias_row = b.sub(row, bias_par)
            if gfx12:
                bias_sel = b.select(
                    b.cmp_ne(bias_par, c0), ci(0x05040302), ci(0x03020100)
                )

        def bias_vals(kv0, a):
            """f32 bias of this lane's 8 score slots of sub-tile ``a``."""
            first_key = b.add(kv0, ci(a * 16))
            n = (
                8 if gfx12 else 16
            )  # gfx11 slots are every other key: load 16, pick parity
            base = b.add(bias_row, first_key)
            if gfx12:
                base = b.add(base, b.mul(half, ci(8)))
            if spec.bias == "q" and gfx12:
                # Five dwords from the aligned key, shifted by ``bias_par`` halves.
                off = b.mul(base, ci(bias_bytes))
                raw = b.buffer_load_vN(bias_rsrc, off, c0, I32, 4)
                words = [b.vec_extract(raw, i) for i in range(4)]
                raw = b.buffer_load_vN(bias_rsrc, b.add(off, ci(16)), c0, I32, 1)
                words.append(b.vec_extract(raw, 0))
                words = [b.perm_b32(words[i + 1], words[i], bias_sel) for i in range(4)]
                v8 = b.vec_bitcast(b.vec_pack(words, I32), VectorType(elem, 8))
                return [b.cast_to_f32(b.vec_extract(v8, i)) for i in range(8)]
            per = 4 if spec.bias == "f32" else 8
            vals = []
            for c in range(n // per):
                raw = b.buffer_load_vN(
                    bias_rsrc,
                    b.mul(b.add(base, ci(c * per)), ci(bias_bytes)),
                    c0,
                    bias_elem,
                    per,
                )
                vals += [b.vec_extract(raw, i) for i in range(per)]
            if bias_elem != F32:
                vals = [b.cast_to_f32(v) for v in vals]
            if gfx12:
                return vals
            odd = b.cmp_ne(bias_par if spec.bias == "q" else half, c0)
            return [b.select(odd, vals[2 * e + 1], vals[2 * e]) for e in range(8)]

    # ---- score units. FP8 K / V are decoded exactly to value / 256 (see _fp8x8): 256 *
    # k_scale joins the score scale (applied to Q.K before any feature, as scaling K is)
    # and 256 * v_scale the output normalization. Raw scores keep exp2(cs * (s - m));
    # an additive term or softcap moves them to natural-log units first (cs = log2 e).
    qk_scale = p["scale_log2"]
    v_mul = None
    if fp8:
        unit = b.const_f32(1.0 if spec.fp8_cvt else _FP8_INT_UNIT)
        qk_scale = b.fmul(qk_scale, b.fmul(p["k_scale"], unit))
        v_mul = b.fmul(p["v_scale"], unit)
    if spec.natural_scores:
        scale_nat = b.fmul(qk_scale, b.const_f32(1.0 / _LOG2E))
        cs = b.const_f32(_LOG2E)  # scores are natural-log units after the features
    else:
        cs = qk_scale  # scores stay raw; exp2(cs * (s - m))

    def slot_key(a, e):
        """Key offset (within the tile) of score slot ``e`` of sub-tile ``a``."""
        return a * 16 + (e if gfx12 else 2 * e)

    lane_key = b.mul(half, ci(8)) if gfx12 else half

    # ---- score features, in the prefill kernel's order: softcap, ALiBi, QQ-bias, then
    # the additive bias and the masks. Positions are sequence-local; ``context`` = ctx.
    if spec.softcap:
        # cap * tanh(x), x = s * scale / cap. ``tanh_fast``: cap - 2 cap / (1 + 2^(2 x log2 e)),
        # exp2 + rcp + fma per score, absolute error a few ulp of cap, +-inf exponentials
        # saturating to +-cap; else the libm tanh (~20 ops with a division).
        cap = p["softcap"]
        cap_in = b.fdiv(scale_nat, cap)
        if spec.tanh_fast:
            cap_e = b.fmul(cap_in, b.const_f32(2.0 * _LOG2E))
            cap_m2 = b.fmul(cap, b.const_f32(-2.0))
            f1 = b.const_f32(1.0)
    if spec.alibi:
        slope = b.global_load(p["alibi_slopes_ptr"], hq, F32, align=4)
    if spec.qq_bias:
        # qq_bias[query, key - ctx] inside [rows, cols), else 0; one bounded scalar read per
        # score (keys before the query section read nothing).
        qq_cols = p["qq_bias_cols"]
        qq_stride = p["qq_bias_stride"]
        qq_ext = b.add(b.mul(b.sub(p["qq_bias_rows"], c1), qq_stride), qq_cols)
        qq_rsrc = b.buffer_rsrc(p["qq_bias_ptr"], b.mul(b.smax(qq_ext, c0), ci(4)))
        qq_q_ok = b.land(row_ok, b.cmp_lt(qpos, p["qq_bias_rows"]))
        qq_row = b.mul(qpos, qq_stride)
        qq_cols64 = b.zext(qq_cols, I64)

    def feature_adds(kv0):
        """Per-tile closure: f32 additive term (or None) of score slot (a, e)."""
        if not (spec.alibi or spec.qq_bias):
            return lambda a, e: None
        rel0 = b.sub(b.add(kv0, lane_key), ctx)  # key - ctx of slot 0
        if spec.alibi:
            al0 = b.fmul(slope, b.sitofp_f32(rel0))

        def add(a, e):
            k = slot_key(a, e)
            out = None
            if spec.alibi:
                out = b.fma(slope, b.const_f32(float(k)), al0) if k else al0
            if spec.qq_bias:
                relk = b.add(rel0, ci(k))
                ok = b.land(
                    qq_q_ok, b.cmp_lt(b.zext(relk, I64), qq_cols64)
                )  # 0 <= relk < cols
                off = b.select(ok, b.mul(b.add(qq_row, relk), ci(4)), big_off)
                qv = b.vec_extract(b.buffer_load_vN(qq_rsrc, off, c0, F32, 1), 0)
                out = qv if out is None else b.fadd(out, qv)
            return out

        return add

    # ---- LDS: one region per wave (V^T staging, then the combine record).
    smem = b.smem_alloc(I32, [NW * REG], name_hint="dec_lds")
    wbase = b.mul(wave, ci(REG))
    vt_wr = b.add(wbase, b.add(b.mul(b.mul(vcg, ci(8)), ci(RS)), vp))
    vt_rd = b.add(wbase, b.mul(col, ci(RS)))
    if gfx12:
        vt_rd = b.add(vt_rd, b.mul(half, ci(4)))

    sel_lo = b.select(lo_half, ci(0x05040100), ci(0x01000504))
    sel_hi = b.select(lo_half, ci(0x07060302), ci(0x03020706))

    def peer_f32(v):
        return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)

    def tree(vals, op):
        while len(vals) > 1:
            nxt = [op(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    big_off = ci(
        0x7FFFFFF0
    )  # past any slice extent: the load reads zero, no memory traffic

    def live_off(voff, colv, width):
        """``voff`` while column chunk ``colv`` is inside ``width``, else out of range."""
        return b.select(b.cmp_lt(colv, width), voff, big_off)

    def k_parts(rsrc, lanes):
        """Raw K loads of one tile: per (sub-tile, K step), the 8-column parts (FP8: the
        raw 8 bytes, decoded at use so prefetch carries half the registers)."""

        def part(a, cbase):
            voff = b.add(lanes[a], ci(cbase * kv_eb))
            if cbase + 16 > _BUCKET_MIN[D]:
                colv = b.add(ci(cbase), b.mul(half, ci(8)))
                voff = live_off(voff, colv, d_rt)
            if fp8:
                return b.buffer_load_vN(rsrc, voff, c0, I32, 2)
            return b.buffer_load_vN(rsrc, voff, c0, elem, 8)

        return [part(a, ks * 16) for a in range(NSUB) for ks in range(n_ks)]

    def k_load(t, valid=None):
        if page_rows:
            return k_parts(k_all, k_row_offs(t, valid))
        return k_parts(k_rsrc(t, valid), k_lane)

    def k_dec(raw):
        if not fp8:
            return raw
        return b.vec_bitcast(
            b.vec_pack(_fp8x8(b, raw, elem, spec.fp8_cvt), I32), VectorType(elem, 8)
        )

    def k_operand(parts, a, ks):
        i = a * n_ks + ks
        if gfx12:
            return k_dec(parts[i])
        # gfx11 duplicates each operand across the lane halves: own 8 columns + the peer's.
        own = b.vec_bitcast(k_dec(parts[i]), VectorType(I32, 4))
        o_ = [b.vec_extract(own, j) for j in range(4)]
        pr = [b.permlanex16(x) for x in o_]
        lo_ = [b.select(lo_half, o_[j], pr[j]) for j in range(4)]
        hi_ = [b.select(lo_half, pr[j], o_[j]) for j in range(4)]
        return b.vec_bitcast(b.vec_pack(lo_ + hi_, I32), VectorType(elem, 16))

    def v_rows(rsrc, lanes, j):
        """Rows (2vp, 2vp+1) of V, columns dv0 + 8 * (VG * j + vcg) .. +7, as dwords
        (FP8: the raw 8 bytes)."""
        out = []
        for rr in range(2):
            voff = b.add(lanes[rr], ci(j * VG * 8 * kv_eb))
            if DVS > 1 or 8 * VG * (j + 1) > _BUCKET_MIN[D]:
                colv = b.add(dv0, b.add(ci(8 * VG * j), b.mul(vcg, ci(8))))
                voff = live_off(voff, colv, dv_rt)
            if fp8:
                out.append(b.buffer_load_vN(rsrc, voff, c0, I32, 2))
            else:
                out.append(
                    b.vec_bitcast(
                        b.buffer_load_vN(rsrc, voff, c0, elem, 8), VectorType(I32, 4)
                    )
                )
        return out

    def v_store(v_raw):
        """V rows -> LDS V^T, two keys per dword (``v_perm_b32`` pairs)."""
        for j in range(n_vj):
            ra, rb = v_raw[j]
            if fp8:
                ra, rb = _fp8x8(b, ra, elem, spec.fp8_cvt), _fp8x8(
                    b, rb, elem, spec.fp8_cvt
                )
            for k in range(4):
                if fp8:
                    xa, xb = ra[k], rb[k]
                else:
                    xa, xb = b.vec_extract(ra, k), b.vec_extract(rb, k)
                for tt, sel in ((0, 0x05040100), (1, 0x07060302)):
                    d_rel = 8 * VG * j + 2 * k + tt  # + 8 * vcg via vt_wr
                    b.smem_store_vN(
                        smem,
                        [b.add(vt_wr, ci(d_rel * RS))],
                        b.perm_b32(xb, xa, ci(sel)),
                        1,
                    )

    def p_operand(ps):
        if gfx12:
            return b.vec_cast_f32_to(b.vec_pack(ps, F32), elem)
        outs = []
        for m in range(4):
            lo_ = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m], elem), I16), I32)
            hi_ = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m + 1], elem), I16), I32)
            v = b.lor(lo_, b.shl(hi_, c16))
            w = b.permlanex16(v)
            outs.append(b.perm_b32(w, v, sel_lo))
            outs.append(b.perm_b32(w, v, sel_hi))
        return b.vec_bitcast(b.vec_pack(outs, I32), VectorType(elem, 16))

    def v_load(t, valid=None):
        if page_rows:
            rsrc, lanes = v_all, v_row_offs(t, valid)
        else:
            rsrc, lanes = v_rsrc(t, valid), v_lane
        return [x for j in range(n_vj) for x in v_rows(rsrc, lanes, j)]

    def v_pairs(flat):
        return [flat[2 * j : 2 * j + 2] for j in range(n_vj)]

    # Distance-1 prefetch: the next tile's raw K and V loads are issued as soon as the
    # current tile's registers retire (K after QK, V after its LDS store) and carried to
    # the next iteration, so a wave keeps two tiles of loads in flight across the
    # softmax and PV. A tile past the split's end gets an empty resource (no traffic).
    pf = spec.prefetch
    iter_args = [("m", ninf), ("l", f0)] + [
        (f"o{d}", b.zero_vec_f32(8)) for d in range(n_dc)
    ]
    if pf:
        ok0 = b.cmp_lt(my_t0, my_end)
        kp0 = k_load(my_t0, ok0)
        vp0 = v_load(my_t0, ok0)
        n_kp, n_vp = len(kp0), len(vp0)
        iter_args += [(f"pk{i}", x) for i, x in enumerate(kp0)] + [
            (f"pv{i}", x) for i, x in enumerate(vp0)
        ]
    loop = b.scf_for_iter(c0, n_it, c1, iter_args=iter_args, iv_name="it")
    with loop as (it, state):
        m_i, l_i = state[0], state[1]
        accs = list(state[2 : 2 + n_dc])
        t = b.add(my_t0, b.mul(it, step_v))
        kv0 = b.mul(t, ci(T))
        if pf:
            kp = list(state[2 + n_dc : 2 + n_dc + n_kp])
            vflat = list(state[2 + n_dc + n_kp :])
            t_n = b.add(t, step_v)
            ok_n = b.cmp_lt(t_n, my_end)
        else:
            # Every load of the tile issued up front, K first so the QK waits leave the
            # V loads in flight.
            act = b.cmp_lt(t, my_end) if spec.lockstep else None
            kp = k_load(t, act)
            vflat = v_load(t, act)
        v_raw = v_pairs(vflat)
        s_acc = [b.zero_vec_f32(8) for _ in range(NSUB)]
        for ks in range(n_ks):
            q_k = q_packs[ks] if q_packs is not None else q_pack(ks)
            for a in range(NSUB):
                s_acc[a] = b.mma(op_id, k_operand(kp, a, ks), q_k, s_acc[a])
        if pf:
            kp_next = k_load(t_n, ok_n)
        # V -> LDS does not depend on the scores: before the softmax (V registers retire
        # early; better on gfx12) or after it (better on gfx11), see ``v_store_early``.
        if v_early:
            v_store(v_raw)
            if pf:
                vp_next = v_load(t_n, ok_n)

        # ---- scores: features, mask, online softmax (lane = query row).
        rel0 = b.add(b.sub(kv0, first), lane_key)
        span_t = (
            b.select(b.cmp_lt(t, my_end), span64, b.const_i64(0))
            if spec.lockstep
            else span64
        )
        adds = feature_adds(kv0)
        svals = []
        for a in range(NSUB):
            for e in range(8):
                s = b.vec_extract(s_acc[a], e)
                if spec.natural_scores:
                    extra = adds(a, e)
                    if spec.bias:
                        if e == 0:
                            bvs = bias_vals(kv0, a)
                        extra = bvs[e] if extra is None else b.fadd(extra, bvs[e])
                    if spec.softcap:
                        if spec.tanh_fast:
                            e2x = b.exp2_fast(b.fmul(s, cap_e))
                            s = b.fma(cap_m2, b.rcp_fast(b.fadd(e2x, f1)), cap)
                        else:
                            s = b.fmul(cap, b.tanh(b.fmul(s, cap_in)))
                        if extra is not None:
                            s = b.fadd(s, extra)
                    else:
                        s = b.fma(s, scale_nat, extra)
                rel = b.zext(b.add(rel0, ci(slot_key(a, e))), I64)
                s = b.select(b.cmp_lt(rel, span_t), s, ninf)
                svals.append(s)
        m_loc = tree(list(svals), b.fmax)
        m_new = b.fmax(m_i, b.fmax(m_loc, peer_f32(m_loc)))
        m_safe = b.select(b.fcmp("oeq", m_new, ninf), f0, m_new)
        nmc = b.fneg(b.fmul(cs, m_safe))
        alpha = b.exp2_fast(b.fma(m_i, cs, nmc))
        ps = [b.exp2_fast(b.fma(s, cs, nmc)) for s in svals]
        l_loc = tree(list(ps), b.fadd)
        l_new = b.fma(l_i, alpha, b.fadd(l_loc, peer_f32(l_loc)))
        alpha_v = b.vector_splat(alpha, 8)
        accs = [b.fmul(acc, alpha_v) for acc in accs]
        p_ops = [p_operand(ps[a * 8 : a * 8 + 8]) for a in range(NSUB)]
        if not v_early:
            v_store(v_raw)
            if pf:
                vp_next = v_load(t_n, ok_n)

        # ---- O^T += V^T P (V^T read back from the wave's LDS region).
        for dc in range(n_dc):
            for a in range(NSUB):
                base = b.add(vt_rd, ci(dc * 16 * RS + a * 8))
                if gfx12:
                    vt = b.vec_bitcast(
                        b.smem_load_vN(smem, base, dtype=I32, n=4), VectorType(elem, 8)
                    )
                else:
                    lo_ = b.smem_load_vN(smem, base, dtype=I32, n=4)
                    hi_ = b.smem_load_vN(smem, b.add(base, ci(4)), dtype=I32, n=4)
                    vt = b.vec_bitcast(b.vec_concat(lo_, hi_), VectorType(elem, 16))
                accs[dc] = b.mma(op_id, vt, p_ops[a], accs[dc])
        if spec.lockstep:
            b.sync()
        b.scf_yield(m_new, l_new, *accs, *((kp_next + vp_next) if pf else ()))

    m_w, l_w = loop.results[0], loop.results[1]
    o_w = list(loop.results[2 : 2 + n_dc])

    # ---- combine the waves through LDS; wave w finishes chunks dc = w, w + NW, ...
    rec_ml = b.add(wbase, b.mul(lane, c2))
    b.smem_store_vN(
        smem, [rec_ml], b.vec_pack([b.bitcast(m_w, I32), b.bitcast(l_w, I32)], I32), 2
    )
    for dc in range(n_dc):
        frag = b.vec_bitcast(o_w[dc], VectorType(I32, 8))
        idx = b.add(wbase, b.add(ci(64 + dc * 256), b.mul(lane, ci(8))))
        b.smem_store_vN(
            smem, [idx], b.vec_pack([b.vec_extract(frag, i) for i in range(4)], I32), 4
        )
        b.smem_store_vN(
            smem,
            [b.add(idx, ci(4))],
            b.vec_pack([b.vec_extract(frag, i) for i in range(4, 8)], I32),
            4,
        )
    b.sync()

    ms, ls = [], []
    for w2 in range(NW):
        ml = b.smem_load_vN(smem, b.add(ci(w2 * REG), b.mul(lane, c2)), dtype=I32, n=2)
        ms.append(b.bitcast(b.vec_extract(ml, 0), F32))
        ls.append(b.bitcast(b.vec_extract(ml, 1), F32))
    m_all = tree(list(ms), b.fmax)
    m_all_safe = b.select(b.fcmp("oeq", m_all, ninf), f0, m_all)
    nmc = b.fneg(b.fmul(cs, m_all_safe))
    fs = [b.exp2_fast(b.fma(mw, cs, nmc)) for mw in ms]
    l_all = tree([b.fmul(lw, fw) for lw, fw in zip(ls, fs)], b.fadd)
    if spec.sinks:
        # The sink adds 2^(sink * log2 e) to the denominator of split 0's partial only, so
        # it enters the softmax exactly once: the merge weights each split by 2^lse_i and
        # sums them, which reproduces sum_k exp(s_k) + exp(sink) (and O = 0, LSE = sink
        # for a row without visible keys). Every dv slice of split 0 applies it alike.
        sink2 = b.fmul(
            b.cast_to_f32(b.global_load(p["sink_ptr"], hq, elem, align=2)),
            b.const_f32(_LOG2E),
        )
        sink2 = b.select(b.cmp_eq(split, c0), sink2, ninf)
        m2 = b.select(b.fcmp("ogt", l_all, f0), b.fmul(cs, m_all_safe), ninf)
        top = b.fmax(m2, sink2)
        top = b.select(b.fcmp("oeq", top, ninf), f0, top)
        keep = b.exp2_fast(b.fsub(m2, top))
        den = b.fadd(b.fmul(l_all, keep), b.exp2_fast(b.fsub(sink2, top)))
        has = b.fcmp("ogt", den, f0)
        inv_l = b.select(has, b.fdiv(keep, den), f0)
        lse2 = b.select(has, b.fadd(top, b.log2(den)), ninf)
    else:
        has = b.fcmp("ogt", l_all, f0)
        inv_l = b.select(has, b.fdiv(b.const_f32(1.0), l_all), f0)
        lse2 = b.select(has, b.fadd(b.fmul(cs, m_all_safe), b.log2(l_all)), ninf)
    if v_mul is not None:
        inv_l = b.fmul(inv_l, v_mul)
    fsv = [b.vector_splat(fw, 8) for fw in fs]
    inv_v = b.vector_splat(inv_l, 8)

    single = b.cmp_eq(ns, c1)
    prow = b.add(b.mul(b.add(b.mul(b.add(b.mul(batch, Hq), hq), Sq), qpos), ns), split)
    o_row = b.add(b.mul(hq, p["stride_o_head"]), b.mul(qpos, p["stride_o_token"]))
    ws_row = b.mul(prow, dv_rt)

    for kk in range((n_dc + NW - 1) // NW):
        dc = b.add(ci(kk * NW), wave)
        acc = None
        for w2 in range(NW):
            idx = b.add(
                b.add(ci(w2 * REG + 64), b.mul(dc, ci(256))), b.mul(lane, ci(8))
            )
            lo_ = b.smem_load_vN(smem, idx, dtype=I32, n=4)
            hi_ = b.smem_load_vN(smem, b.add(idx, ci(4)), dtype=I32, n=4)
            fr = b.vec_bitcast(b.vec_concat(lo_, hi_), VectorType(F32, 8))
            term = b.fmul(fr, fsv[w2])
            acc = term if acc is None else b.fadd(acc, term)
        acc = b.fmul(acc, inv_v)
        o = [b.vec_extract(acc, i) for i in range(8)]
        if not gfx12:
            # Slot i holds d = 2i + half; trade with the lane^16 peer so each lane holds
            # eight consecutive columns (8 * half + 0..7).
            send = [b.select(lo_half, o[4 + k], o[k]) for k in range(4)]
            recv = [peer_f32(x) for x in send]
            out = []
            for k in range(4):
                out.append(b.select(lo_half, o[k], recv[k]))
                out.append(b.select(lo_half, recv[k], o[4 + k]))
            o = out
        colv = b.add(b.add(dv0, b.mul(dc, c16)), b.mul(half, ci(8)))
        ok = b.land(b.land(row_ok, b.cmp_lt(colv, dv_rt)), b.cmp_lt(dc, ci(n_dc)))
        with b.scf_if(b.land(ok, single)):
            vec = b.vec_cast_f32_to(b.vec_pack(o, F32), elem)
            ptr = b.global_ptr_add(Ob, b.mul(b.add(o_row, colv), c2))
            b.global_store(ptr, c0, vec, align=16)
        with b.scf_if(b.land(ok, b.lnot(single))):
            ws_off = b.mul(b.zext(b.add(ws_row, colv), I64), b.const_i64(4))
            ptr = b.global_ptr_add(p["WS_O"], ws_off)
            b.global_store(ptr, c0, b.vec_pack(o[:4], F32), align=16)
            b.global_store(ptr, c1, b.vec_pack(o[4:], F32), align=16)

    def lse_seq():
        """LSE offset of the sequence's first row (packed: its first token)."""
        if spec.packed:
            return b.mul(q_start, p["stride_lse_token"])
        return b.mul(batch, p["stride_lse_batch"])

    stats = b.land(b.land(row_ok, lo_half), b.cmp_eq(wave, c0))
    with b.scf_if(
        b.land(
            b.land(stats, b.cmp_eq(dvi, c0)),
            b.land(single, b.cmp_ne(p["write_lse"], c0)),
        )
    ):
        lse_off = b.add(
            lse_seq(),
            b.add(b.mul(hq, p["stride_lse_head"]), b.mul(qpos, p["stride_lse_token"])),
        )
        b.global_store(p["LSE"], lse_off, b.fmul(lse2, b.const_f32(_LN2)), align=4)
    with b.scf_if(b.land(stats, b.lnot(single))):
        # Every dv slice stores the (identical) partial LSE: a fused merge waits only for
        # its own slice's splits, so it must not read an LSE another slice writes.
        b.global_store(p["WS_LSE"], prow, lse2, align=4)

    # Fused merge (WS_CNT non-null): the last split of each (batch, pack group, dv slice, row
    # tile) to finish merges the group's partials in place of the merge launch, then
    # re-arms its counter. A null WS_CNT leaves the merge to the merge kernel.
    with b.scf_if(b.cmp_ne(b.global_ptr_to_i64(p["WS_CNT"]), b.const_i64(0))):
        with b.scf_if(b.lnot(single)):
            b.fence(scope="agent", ordering="release")
            b.sync()
            grp = b.add(
                b.mul(b.add(b.mul(b.add(b.mul(batch, n_pg), pg), ci(DVS)), dvi), n_rt),
                rt,
            )
            with b.scf_if(b.cmp_eq(tid, c0)):
                old = b.global_atomic_add(p["WS_CNT"], grp, c1)
                b.smem_store_vN(smem, [c0], old, 1)
            b.sync()
            arrived = b.readfirstlane(
                b.vec_extract(b.smem_load_vN(smem, c0, dtype=I32, n=1), 0)
            )
            with b.scf_if(b.cmp_eq(arrived, b.sub(ns, c1))):
                b.fence(scope="agent", ordering="acquire")
                tpr = spec.block_size // DECODE_ROWS  # threads per merged row
                mrow = b.div(tid, ci(tpr))
                mcg = b.mod(tid, ci(tpr))
                rm = b.add(b.mul(rt, c16), mrow)
                rm_ok = b.cmp_lt(rm, rows_b)
                rm = b.select(rm_ok, rm, c0)
                hq_m = b.add(b.mul(pg, g), b.div(rm, Sq_d))
                qpos_m = b.mod(rm, Sq_d)
                row_ns = b.mul(
                    b.add(b.mul(b.add(b.mul(batch, Hq), hq_m), Sq), qpos_m), ns
                )
                n_c4 = max(1, (DVT // _MERGE_COLS + tpr - 1) // tpr)
                cols, oks = [], []
                for k in range(n_c4):
                    rel = b.mul(b.add(mcg, ci(k * tpr)), ci(_MERGE_COLS))
                    colv = b.add(dv0, rel)
                    ok = b.land(
                        rm_ok, b.land(b.cmp_lt(rel, ci(DVT)), b.cmp_lt(colv, dv_rt))
                    )
                    cols.append(b.select(ok, colv, c0))
                    oks.append(ok)
                m_sf, l_all, accs_m = _split_merge(
                    b, p["WS_O"], p["WS_LSE"], row_ns, ns, dv_rt, cols
                )
                has = b.fcmp("ogt", l_all, f0)
                inv_l = b.vector_splat(
                    b.select(has, b.fdiv(b.const_f32(1.0), l_all), f0), _MERGE_COLS
                )
                o_row_m = b.add(
                    b.mul(hq_m, p["stride_o_head"]), b.mul(qpos_m, p["stride_o_token"])
                )
                for colv, ok, acc in zip(cols, oks, accs_m):
                    with b.scf_if(ok):
                        vec = b.vec_cast_f32_to(b.fmul(acc, inv_l), elem)
                        b.global_store(
                            b.global_ptr_add(Ob, b.mul(b.add(o_row_m, colv), c2)),
                            c0,
                            vec,
                            align=8,
                        )
                lse_ok = b.land(
                    b.land(rm_ok, b.cmp_eq(mcg, c0)),
                    b.land(b.cmp_eq(dvi, c0), b.cmp_ne(p["write_lse"], c0)),
                )
                with b.scf_if(lse_ok):
                    lse_off = b.add(
                        lse_seq(),
                        b.add(
                            b.mul(hq_m, p["stride_lse_head"]),
                            b.mul(qpos_m, p["stride_lse_token"]),
                        ),
                    )
                    val = b.select(
                        has,
                        b.fmul(b.fadd(m_sf, b.log2(l_all)), b.const_f32(_LN2)),
                        ninf,
                    )
                    b.global_store(p["LSE"], lse_off, val, align=4)
                with b.scf_if(b.cmp_eq(tid, c0)):
                    b.global_store(p["WS_CNT"], grp, c0, align=4)
    b.ret()
    return b.kernel


def _split_merge(b, ws_o, ws_lse, row_ns, ns, dv, cols):
    """Online LSE merge of ``ns`` split partials of one row: ``(m, l, [acc per col])``.

    ``cols`` are in-range column offsets of ``_MERGE_COLS`` consecutive f32 each;
    ``_MERGE_UNROLL`` splits are loaded per loop step.
    """
    ci = b.const_i32
    c0, c1 = ci(0), ci(1)
    f0 = b.const_f32(0.0)
    ninf = b.const_f32(float("-inf"))
    U = _MERGE_UNROLL
    n_grp = b.div(b.add(ns, ci(U - 1)), ci(U))
    iter_args = [("mg_m", ninf), ("mg_l", f0)] + [
        (f"mg_a{k}", b.zero_vec_f32(_MERGE_COLS)) for k in range(len(cols))
    ]
    sloop = b.scf_for_iter(c0, n_grp, c1, iter_args=iter_args, iv_name="sg")
    with sloop as (sg, st):
        lses, parts = [], []
        for i in range(U):
            s = b.add(b.mul(sg, ci(U)), ci(i))
            ok = b.cmp_lt(s, ns)
            prow = b.add(row_ns, b.select(ok, s, c0))
            lses.append(b.select(ok, b.global_load_f32(ws_lse, prow), ninf))
            base = b.mul(prow, dv)
            parts.append(
                [
                    b.global_load_vN(ws_o, b.add(base, cv), F32, _MERGE_COLS, align=16)
                    for cv in cols
                ]
            )
        m_new = st[0]
        for x in lses:
            m_new = b.fmax(m_new, x)
        m_sf = b.select(b.fcmp("oeq", m_new, ninf), f0, m_new)
        alpha = b.exp2_fast(b.fsub(st[0], m_sf))
        l_new = b.fmul(st[1], alpha)
        av = b.vector_splat(alpha, _MERGE_COLS)
        accs = [b.fmul(a, av) for a in st[2:]]
        for x, prt in zip(lses, parts):
            wgt = b.exp2_fast(b.fsub(x, m_sf))
            l_new = b.fadd(l_new, wgt)
            wv = b.vector_splat(wgt, _MERGE_COLS)
            accs = [b.fadd(a, b.fmul(pv, wv)) for a, pv in zip(accs, prt)]
        b.scf_yield(m_new, l_new, *accs)
    res = sloop.results
    m_all = res[0]
    return b.select(b.fcmp("oeq", m_all, ninf), f0, m_all), res[1], list(res[2:])


def _fp8x8(b, raw, elem, native):
    """Eight OCP E4M3FN bytes (``<2 x i32>``) -> four dwords of ``elem`` pairs, exactly
    (every E4M3FN value, subnormals and NaN included, is an F16 / BF16 value): the value
    itself when ``native``, else value / 256 (:data:`_FP8_INT_UNIT`).

    ``native`` (gfx12): ``v_cvt_pk_f32_fp8`` (OCP E4M3 on gfx12) to F32, then F16 by a
    (exact) conversion or BF16 by keeping each F32's upper half.

    Integer form (gfx11, no FP8 conversion instructions): the F16 bits of value / 256 are
    the byte's sign and its 7 magnitude bits moved to F16 bits [13:7] (the 8-bit exponent
    bias gap is the 2^-8; E4M3 subnormals land on F16 subnormals). The two NaN encodings
    (magnitude 0x7F) also set the exponent's top bit, giving an F16 NaN. BF16 goes
    through F32 (exact) and keeps its upper half.
    """
    ci = b.const_i32
    if native:
        v8 = b.vec_bitcast(raw, VectorType(FP8E4M3, 8))
        f = []
        for h in range(2):
            q4 = b.vec_pack([b.vec_extract(v8, 4 * h + i) for i in range(4)], FP8E4M3)
            f4 = b.cvt_pk_f32_fp8x4(q4)
            f += [b.vec_extract(f4, i) for i in range(4)]
        if elem == BF16:
            fi = [b.bitcast(x, I32) for x in f]
            return [
                b.perm_b32(fi[2 * k + 1], fi[2 * k], ci(0x07060302)) for k in range(4)
            ]
        h16 = b.vec_bitcast(
            b.vec_cast_f32_to(b.vec_pack(f, F32), F16), VectorType(I32, 4)
        )
        return [b.vec_extract(h16, k) for k in range(4)]
    m80 = ci(0x80808080)
    out = []
    for w in range(2):
        x = b.vec_extract(raw, w)
        mag = b.land(x, ci(0x7F7F7F7F))
        # Per byte: hi = sign | mag >> 1 (| 0x40 for NaN), lo = (mag & 1) << 7. ``hm`` is
        # mag >> 1 with the next byte's bit 0 in each bit 7 (masked off in hi).
        hm = b.lshr(mag, ci(1))
        hi = b.lor(b.land(x, m80), b.land(hm, ci(0x3F3F3F3F)))
        # NaN (magnitude 0x7F): magnitude bits 1-6 carry into bit 6 when incremented, and bit
        # 0 is set. hm + 0x01010101 < 2^31 (no signed overflow) and carries stay in bytes.
        nan = b.land(b.land(b.add(hm, ci(0x01010101)), b.shl(x, ci(6))), ci(0x40404040))
        hi = b.lor(hi, nan)
        lo = b.land(b.shl(x, ci(7)), m80)
        out += [b.perm_b32(hi, lo, ci(0x05010400)), b.perm_b32(hi, lo, ci(0x07030602))]
    if elem == BF16:
        h = b.vec_bitcast(b.vec_pack(out, I32), VectorType(F16, 8))
        f = [b.bitcast(b.cast_to_f32(b.vec_extract(h, i)), I32) for i in range(8)]
        out = [b.perm_b32(f[2 * k + 1], f[2 * k], ci(0x07060302)) for k in range(4)]
    return out


def _merge_name(spec: WmmaFmhaDecodeSpec) -> str:
    return f"rocke_wmma_fmha_decode_merge_{spec.dtype}" + (
        "_packed" if spec.packed else ""
    )


def _declare_merge_params(b: IRBuilder, spec: WmmaFmhaDecodeSpec):
    elem = _elem(spec)
    p = {
        "O": b.param(
            "O", PtrType(elem, "global"), noalias=True, writeonly=True, align=8
        ),
        "LSE": b.param(
            "LSE", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
        ),
        "WS_O": b.param(
            "WS_O", PtrType(F32, "global"), noalias=True, readonly=True, align=16
        ),
        "WS_LSE": b.param(
            "WS_LSE", PtrType(F32, "global"), noalias=True, readonly=True, align=4
        ),
    }
    for name in (
        "seqlen_q",
        "num_query_heads",
        "head_dim_v",
        "num_splits",
        "write_lse",
    ):
        p[name] = b.param(name, I32)
    for tensor in ("o", "lse"):
        for axis in ("batch", "token", "head"):
            p[f"stride_{tensor}_{axis}"] = b.param(f"stride_{tensor}_{axis}", I32)
    if spec.packed:
        p["cu_seqlens_q"] = b.param(
            "cu_seqlens_q", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
    return p


def build_wmma_fmha_decode_merge(spec: WmmaFmhaDecodeSpec, *, arch: str) -> KernelDef:
    """Combine the split partials into O (and the natural-log LSE).

    One workgroup of :data:`MERGE_THREADS` per output row ``(batch, head,
    position)`` (grid :func:`wmma_fmha_decode_merge_grid`); thread ``t`` owns
    columns ``4t .. 4t+3`` (head dims up to 256). Only the output dtype and, for
    ragged / paged specs, the packed row layout are baked: a packed merge
    (``_packed``) takes ``cu_seqlens_q`` last, addresses O / LSE rows from each
    sequence's first token and skips positions past its length.
    """
    _layout(arch, spec.dtype)
    elem = _elem(spec)
    b = IRBuilder(_merge_name(spec))
    b.kernel.attrs["max_workgroup_size"] = MERGE_THREADS
    p = _declare_merge_params(b, spec)
    ci = b.const_i32
    c0, c1 = ci(0), ci(1)
    f0 = b.const_f32(0.0)
    ninf = b.const_f32(float("-inf"))
    Sq, Hq, ns, dv = (
        p["seqlen_q"],
        p["num_query_heads"],
        p["num_splits"],
        p["head_dim_v"],
    )
    row = b.block_id_x()
    qpos = b.mod(row, Sq)
    hq = b.mod(b.div(row, Sq), Hq)
    batch = b.div(row, b.mul(Sq, Hq))
    tid = b.thread_id_x()
    d = b.mul(tid, ci(_MERGE_COLS))
    d_ok = b.cmp_lt(d, dv)
    d_safe = b.select(d_ok, d, c0)
    row_ns = b.mul(row, ns)
    row_ok = None
    if spec.packed:
        q_start = b.global_load_i32(p["cu_seqlens_q"], batch)
        row_ok = b.cmp_lt(
            qpos, b.sub(b.global_load_i32(p["cu_seqlens_q"], b.add(batch, c1)), q_start)
        )
        o_seq = b.mul(q_start, p["stride_o_token"])
        lse_seq = b.mul(q_start, p["stride_lse_token"])
    else:
        o_seq = None

    m_safe, l_all, (acc,) = _split_merge(
        b, p["WS_O"], p["WS_LSE"], row_ns, ns, dv, [d_safe]
    )
    has = b.fcmp("ogt", l_all, f0)
    inv_l = b.select(has, b.fdiv(b.const_f32(1.0), l_all), f0)
    out = b.vec_cast_f32_to(b.fmul(acc, b.vector_splat(inv_l, _MERGE_COLS)), elem)
    o_off = b.add(
        b.add(
            b.mul(batch, p["stride_o_batch"]) if o_seq is None else o_seq,
            b.mul(qpos, p["stride_o_token"]),
        ),
        b.add(b.mul(hq, p["stride_o_head"]), d_safe),
    )
    with b.scf_if(d_ok if row_ok is None else b.land(d_ok, row_ok)):
        b.global_store(b.global_ptr_add(p["O"], b.mul(o_off, ci(2))), c0, out, align=8)
    lse_ok = b.land(b.cmp_eq(tid, c0), b.cmp_ne(p["write_lse"], c0))
    with b.scf_if(lse_ok if row_ok is None else b.land(lse_ok, row_ok)):
        lse = b.select(
            has, b.fmul(b.fadd(m_safe, b.log2(l_all)), b.const_f32(_LN2)), ninf
        )
        lse_off = b.add(
            lse_seq if spec.packed else b.mul(batch, p["stride_lse_batch"]),
            b.add(b.mul(hq, p["stride_lse_head"]), b.mul(qpos, p["stride_lse_token"])),
        )
        b.global_store(p["LSE"], lse_off, lse, align=4)
    b.ret()
    return b.kernel


def build_wmma_fmha_decode_module(spec: WmmaFmhaDecodeSpec, *, arch: str):
    """The decode kernel with its merge kernel as one code object.

    Returns ``rocke.helpers.KernelModule(primary=<decode>, companions={"merge":
    <merge>})`` for :func:`rocke.helpers.compile_module` / hkp_pack: the merge
    matches the spec's dtype and row layout (dense or packed), and a host that
    loaded the decode kernel finds the merge in the same module by name.
    """
    from rocke.helpers import KernelModule

    return KernelModule(
        primary=build_wmma_fmha_decode(spec, arch=arch),
        companions={"merge": build_wmma_fmha_decode_merge(spec, arch=arch)},
    )


# ---------------------------------------------------------------------------- host helpers
def wmma_fmha_decode_bucket(*, head_dim_q: int, head_dim_v: int) -> Optional[int]:
    """The ``head_size`` bucket that serves runtime widths ``(head_dim_q, head_dim_v)``, or
    None: the smallest of 64 / 128 / 256 holding both widths (positive multiples of 8)
    whose smallest served Q/K width (16 / 72 / 136) is at most ``head_dim_q``.

    K column chunks below a bucket's minimum width are loaded without masking (their Q
    columns past ``head_dim_q`` are zero, but NaN/Inf bytes past the row would poison the
    scores), so a narrower ``head_dim_q`` is not served. ``head_dim_v`` has no lower bound:
    V columns past it only reach output columns that are never stored.
    """
    dq, dv = head_dim_q, head_dim_v
    if dq <= 0 or dv <= 0 or dq % 8 or dv % 8:
        return None
    bucket = next((b for b in _BUCKETS if max(dq, dv) <= b), None)
    if bucket is None or dq < _BUCKET_MIN[bucket]:
        return None
    return bucket


def wmma_fmha_decode_kv_range(
    *,
    seqlen_q: int,
    seqlen_k: int,
    bottom_right: bool,
    window_left: int,
    window_right: int,
) -> Tuple[int, int]:
    """The keys any query row can see, ``[lo, hi)`` (mirrors the kernel)."""
    ctx = seqlen_k - seqlen_q if bottom_right else 0
    lo = max(ctx - window_left, 0) if window_left >= 0 else 0
    hi = (
        max(min(ctx + window_right + seqlen_q, seqlen_k), 0)
        if window_right >= 0
        else seqlen_k
    )
    return lo, hi


def wmma_fmha_decode_pack_groups(
    *, num_query_heads: int, num_kv_heads: int, num_v_heads: int
) -> int:
    """Pack groups ``P = max(Hkv, Hv)``: each covers ``Hq / P`` query heads that read one
    K head and one V head (the kernel derives the same from its head-count arguments).
    """
    hq, hk, hv = num_query_heads, num_kv_heads, num_v_heads
    if min(hq, hk, hv) <= 0 or hq % hk or hq % hv:
        raise ValueError(
            "num_query_heads must be a positive multiple of num_kv_heads and num_v_heads"
        )
    if max(hk, hv) % min(hk, hv):
        raise ValueError("one of num_kv_heads / num_v_heads must divide the other")
    return max(hk, hv)


def wmma_fmha_decode_row_tiles(
    *, num_query_heads: int, num_kv_heads: int, num_v_heads: int, seqlen_q: int
) -> int:
    """16-row tiles per pack group (:func:`wmma_fmha_decode_pack_groups`)."""
    groups = wmma_fmha_decode_pack_groups(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_v_heads=num_v_heads,
    )
    rows = (num_query_heads // groups) * seqlen_q
    return (rows + DECODE_ROWS - 1) // DECODE_ROWS


def wmma_fmha_decode_num_splits(
    spec: WmmaFmhaDecodeSpec,
    *,
    batch: int,
    pack_groups: int,
    row_tiles: int,
    kv_len: int,
    num_cus: int,
    wg_per_cu: int = 2,
    min_tiles_per_wave: int = 4,
) -> int:
    """Splits so ``batch * pack_groups * row_tiles * dv_split * splits`` workgroups fill
    ``wg_per_cu`` per CU (``num_cus``: the device's multiprocessor count), while each wave
    keeps ``min_tiles_per_wave`` runs of 32 keys of ``kv_len`` (the visible key range,
    ``hi - lo`` of :func:`wmma_fmha_decode_kv_range`). ``spec`` must be resolved or
    carry an explicit ``dv_split``; the result is at most 64."""
    groups = max(1, batch * pack_groups * row_tiles * spec.resolved_dv_split)
    tiles = (max(kv_len, 0) + _SPLIT_KEYS - 1) // _SPLIT_KEYS
    by_work = max(1, tiles // (spec.num_waves * min_tiles_per_wave))
    by_device = max(1, (max(num_cus, 1) * wg_per_cu) // groups)
    return max(1, min(by_work, by_device, _MAX_SPLITS))


def wmma_fmha_decode_grid(
    spec: WmmaFmhaDecodeSpec,
    *,
    batch: int,
    pack_groups: int,
    row_tiles: int,
    num_splits: int,
) -> Tuple[int, int, int]:
    """Workgroups along x. Ragged / paged: ``batch`` is the sequence count and the row
    tiles come from the longest query length (the kernel's ``seqlen_q``)."""
    n = batch * num_splits * pack_groups * spec.resolved_dv_split * row_tiles
    if not 0 < n <= 0x7FFFFFFF:
        raise ValueError(f"decode grid {n} does not fit I32")
    return (n, 1, 1)


def wmma_fmha_decode_merge_grid(
    *, batch: int, num_query_heads: int, seqlen_q: int
) -> Tuple[int, int, int]:
    """One merge workgroup (:data:`MERGE_THREADS` threads) per output row."""
    n = batch * num_query_heads * seqlen_q
    if not 0 < n <= 0x7FFFFFFF:
        raise ValueError(f"merge grid {n} does not fit I32")
    return (n, 1, 1)


def wmma_fmha_decode_workspace(
    spec: WmmaFmhaDecodeSpec,
    *,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    num_v_heads: int,
    seqlen_q: int,
    head_dim_v: int,
    num_splits: int,
) -> dict:
    """Split workspace layout: byte offsets into one allocation of ``bytes`` (all zero
    when ``num_splits == 1``, when the kernel takes null workspace pointers).

    ``ws_o``: f32 partial O, ``head_dim_v`` per (batch, head, position, split) row
    (``WS_O``, 16-byte aligned); ``ws_lse``: one f32 log2-domain LSE per row
    (``WS_LSE``); ``ws_cnt``: one i32 arrival counter per (batch, pack group, dv slice,
    row tile) for the fused merge (``WS_CNT``; pass null to use the merge kernel). The
    counters must be zero when the allocation is first used; the kernel re-arms them.
    Ragged / paged: ``batch`` is the sequence count and ``seqlen_q`` the longest query.
    """
    if num_splits <= 1:
        return {"ws_o": 0, "ws_lse": 0, "ws_cnt": 0, "bytes": 0}
    rows = batch * num_query_heads * seqlen_q * num_splits
    if rows * (head_dim_v + 1) > 0x7FFFFFFF:
        raise ValueError("split workspace does not fit I32 element offsets")
    lse = rows * head_dim_v * 4
    cnt = lse + rows * 4
    kw = dict(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_v_heads=num_v_heads,
    )
    row_tiles = wmma_fmha_decode_row_tiles(seqlen_q=seqlen_q, **kw)
    groups = (
        batch * wmma_fmha_decode_pack_groups(**kw) * spec.resolved_dv_split * row_tiles
    )
    return {"ws_o": 0, "ws_lse": lse, "ws_cnt": cnt, "bytes": cnt + groups * 4}


def wmma_fmha_decode_fused_merge(num_splits: int) -> bool:
    """Merge in the last split (pass WS_CNT) rather than launch the merge kernel.

    The fused merge saves the second launch but lengthens every workgroup's tail by an
    agent-scope release, a barrier and an atomic; it wins up to
    ``_FUSED_MAX_SPLITS`` splits and the merge kernel wins above.
    """
    return 1 < num_splits <= _FUSED_MAX_SPLITS


def _signature(kernel: KernelDef):
    return tuple({"name": prm.name, "type": prm.type.name} for prm in kernel.params)


def wmma_fmha_decode_signature(spec: WmmaFmhaDecodeSpec):
    """The decode kernel's ``({"name", "type"}, ...)`` argument list."""
    b = IRBuilder(spec.kernel_name())
    _declare_params(b, spec)
    return _signature(b.kernel)


def wmma_fmha_decode_merge_signature(spec: WmmaFmhaDecodeSpec):
    """The merge kernel's argument list (dtype and packed layout only)."""
    b = IRBuilder(_merge_name(spec))
    _declare_merge_params(b, spec)
    return _signature(b.kernel)
