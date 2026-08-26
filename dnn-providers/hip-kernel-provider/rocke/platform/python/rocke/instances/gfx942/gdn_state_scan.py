# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 gated-delta-rule / KDA chunked state scan (K5 + optional fused K6).

The compute, for one chunk of ``BT`` tokens, with state ``h`` of shape
``[V, K]``::

    h_snapshot = h                                  # drained for the K6 stage
    v_new      = u - w @ h^T                        # GEMM1
    v_new     *= decay                              # scalar-gate path only
    h          = Diag(gate) @ h + k^T @ v_new       # GEMM2

With ``COMPUTE_OUTPUT=True``, the same chunk also computes from the pre-update
snapshot::

    o_inter = q @ h_snapshot^T
    A       = tril(q @ k^T)
    o_intra = A @ v_new
    o       = scale * combine_gates(o_inter, o_intra)

**Why this is an arch-specific instance rather than a shared one.** ``k`` feeds
GEMM2 as an A-operand contracted along ``K``, which wants a ``[K, BT]`` fragment
read. CDNA4 gets that from a transposing LDS read (``ds_read_*_tr_*``); gfx942
has no such instruction, so this kernel transposes on the *store* side instead,
staging ``k`` into LDS as ``[K, BT]`` one row at a time
(:func:`_stage_k_transposed`). That staging, and the K-row prefetch schedule
built around it, is the part that genuinely differs on gfx942.

``MFMA_K`` is derived from the arch catalog, not hardcoded. On CDNA3 the widest
16x16 bf16 atom is K=16; on CDNA4 it is K=32. Every K-loop bound keys off
:attr:`GdnStateScanSpec.mfma_k`, so retargeting changes one lookup instead of a
dozen literals.

**The MFMA operand mapping**, for a 16x16xK atom on wave64 (verified empirically
against a torch matmul before being used here):

===========  ===============================================================
A operand    lane holds ``A[m = lane % 16, k = (lane // 16) * 4 + e]``
B operand    lane holds ``B[k = (lane // 16) * 4 + e, n = lane % 16]``
C / D        lane holds ``D[m = (lane // 16) * 4 + e, n = lane % 16]``
===========  ===============================================================

In GEMM1 ``B`` is ``h^T``, i.e. ``B[k, v] == h[v, k]``, so the B-operand read is
a run along K at fixed ``v`` — contiguous in the K-major LDS layout, the same
shape of access as the A-operand read. This mapping is the highest-risk part of
the kernel: a transposed operand load is a legal set of addresses and yields
silently wrong numbers rather than an error.

**The C-operand trap.** ``u`` is GEMM1's C operand, so it must be gathered in
the C-fragment layout: four *consecutive* BT rows at a fixed V column, which is
a strided read in a ``[T, V]``-major tensor. A row-contiguous load of the same
tile yields the same element count and stays in bounds — it just computes the
transpose. That is why ``u`` is read with scalar loads keyed off
``atom.lane_to_output`` rather than a vector load.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import ClassVar, List, Optional, Tuple

from ...core.arch.target import ArchTarget
from ...core.ir import (BF16, F32, I16, I32, IRBuilder, KernelDef, PtrType,
                        VectorType)
from ...helpers.atoms import mfma_atom
from ...helpers.spec import SignatureBuilder

#: log2(e) — the kernel works in the log2 domain so the gate can use the raw
#: ``v_exp_f32`` (``exp2``) rather than a range-reduced ``exp``.
LOG2E = 1.4426950408889634

#: LDS row padding, in bf16 elements, for the unswizzled fallback layout.
#: Padding is the always-correct answer; the XOR swizzle
#: (:attr:`GdnStateScanSpec.LDS_SWIZZLE`) is the numerics-neutral optimization
#: on top of it.
LDS_PAD = 8

#: Global->LDS staging vector width, in bf16 elements (8 * 2 B = 128 bit).
STAGE_VEC = 8

#: XCD count on this part; drives the P7 chiplet remap.
NXCD = 8


#: bf16 atoms we are willing to use, widest-K first. The 16x16 shape is what
#: the state-scan tiling assumes (a 32x32 atom would change N_REPEAT and the
#: whole wave decomposition), so only the K dimension varies by arch.
_BF16_ATOM_PREFERENCE = (
    "mfma_f32_16x16x32_bf16",   # CDNA4
    "mfma_f32_16x16x16_bf16",   # CDNA3
)


def pick_bf16_atom(arch: str) -> Tuple[str, int]:
    """Return ``(op_id, mfma_k)`` — the widest legal 16x16 bf16 atom on *arch*."""
    target = ArchTarget.from_gfx(arch)
    catalog = target.mma
    ops = catalog.ops if hasattr(catalog, "ops") else catalog
    available = {o.op_id for o in ops}
    for op_id in _BF16_ATOM_PREFERENCE:
        if op_id in available:
            return op_id, int(op_id.rsplit("x", 1)[-1].split("_")[0])
    raise ValueError(
        f"no 16x16 bf16 MFMA atom on {arch}; available: {sorted(available)}"
    )


@dataclass(frozen=True)
class GdnStateScanSpec:
    """Compile-time configuration for one build of the state-scan kernel.

    Every downstream quantity is a derived property rather than a stored field,
    and :meth:`__post_init__` rejects the illegal combinations — so an unusable
    configuration is impossible to construct rather than merely discouraged.
    Use :func:`is_valid_spec` for a non-throwing probe.
    """

    # -- problem shape --
    K: int = 128
    V: int = 128
    H: int = 12                  # value heads on this device (post-TP)
    Hg: int = 12                 # key heads on this device (post-TP); GQA = H // Hg
    BT: int = 64                 # chunk size; fixed by the upstream stages
    BV: int = 32                 # V-slice per CTA

    # -- gate selection: exactly one --
    USE_G: bool = True           # scalar gate
    USE_GK: bool = False         # per-channel gate

    # -- feature knobs, all default-off-or-benign --
    USE_INITIAL_STATE: bool = True
    STORE_FINAL_STATE: bool = True
    SAVE_NEW_VALUE: bool = True
    STORE_H: bool = True
    IS_VARLEN: bool = True
    WU_CONTIGUOUS: bool = True
    STATE_DTYPE_BF16: bool = False
    OUTPUT_DTYPE_BF16: bool = False
    G_IS_LOG2_SCALED: bool = False

    # -- wave widening --
    NR_SPLIT: int = 1

    #: Group-major + XOR swizzle for every LDS buffer, matching the FlyDSL
    #: parent. A logical ``[R, C]`` tile is stored as ``[R][C/4][4]`` and the
    #: 4-element group index is XORed by ``(row ^ (row >> 3)) & (ng - 1)``.
    #:
    #: When off, the buffers are row-major with :data:`LDS_PAD` trailing
    #: elements instead — always correct, but it costs LDS *and* leaves bank
    #: conflicts on the fragment reads. Padding does not fit the fused
    #: configuration, so this is a fit requirement there, not just a tuning
    #: knob. Default-on because the parent does it and the comparison is
    #: supposed to isolate the DSL, not the algorithm.
    LDS_SWIZZLE: bool = True

    #: Loop-carried prefetch: issue chunk i+1's ``w`` / ``u`` / gate reads at
    #: the end of chunk i and carry the raw values across the loop back edge.
    #: The FlyDSL parent does this, so matching it matters for a like-for-like
    #: comparison.
    #:
    #: **Default-off pending root cause.** With prefetch on *and* the LDS
    #: swizzle on, ``BV=64, NR_SPLIT=2, USE_GK`` produces wrong results —
    #: deterministically, and only when both ``u`` and the gate are prefetched.
    #: Reordering the carry list makes all 14 verify configs pass, but the
    #: reason the original order fails is not understood, so the flag stays off
    #: rather than shipping a fix nobody can explain. See PERF_PLAN.md P5.
    PREFETCH: bool = False

    #: Carry-order switch for the prefetch phis, kept only to reproduce the
    #: backend miscompile (see repro_phi_order.py / PERF_PLAN.md §P5). True (the
    #: default and committed layout) puts the ``<4 x float>`` gate phis before
    #: the ``bfloat`` u phis, which the backend compiles correctly. False
    #: reproduces the miscompiled scalar-first order. No effect unless PREFETCH.
    PREFETCH_VEC_FIRST: bool = True

    #: Issue the next chunk's ``w`` loads before the pre-GEMM2 barrier. ATT
    #: shows the default post-barrier issue still waiting on ``w`` at the next
    #: iteration's LDS store. This adds barrier overlap without keeping W live
    #: across GEMM1, while U and gate remain at the lower-pressure late point.
    PREFETCH_W_EARLY: bool = False

    #: Load the current chunk's four K rows before the first barrier, then
    #: gather and transpose-store them before GEMM2.
    PREFETCH_K_EARLY: bool = False

    #: Spread the four K-row loads across the first four GEMM1 MFMA steps.
    #: This is the lower-vmcnt-pressure schedule for wider blocks.
    PREFETCH_K_INTERLEAVE: bool = False

    #: Route global **loads** through bounds-checked buffer descriptors
    #: (``buffer_rsrc`` + ``buffer_load_*``) instead of raw pointers, letting
    #: hardware return 0 out of range so the explicit row clamps can go. The
    #: FlyDSL parent does this via ``make_buffer_tensor(max_size=False)``.
    #:
    #: **Default-off: implemented but NOT yet verified on hardware.**
    BUFFER_DESC: bool = False

    #: Chiplet (XCD) remap of the flat block id, so a head's whole run of
    #: ``GRID_V`` V-tiles lands on one XCD and shares its L2 copy of the
    #: V-independent ``w`` / ``k`` / gate slices. Ported from the parent,
    #: including its tail guard. ``NXCD = 8`` on this part.
    #:
    #: **Default-off: implemented but NOT yet verified on hardware.**
    XCD_REMAP: bool = False

    #: Force VGPR-form MFMA by pinning ``amdgpu-agpr-alloc`` to 0.
    #:
    #: The state accumulators are read and written by VALU every chunk (the
    #: gate multiply), so leaving them in AGPRs makes the backend insert
    #: ``accvgpr_read`` / ``accvgpr_write`` copies around every gate. Pinning
    #: AGPR allocation to zero removes those copies outright and *lowers* the
    #: register count. Default-off to match rocKE's default codegen; see
    #: PERF_PLAN.md item P0.
    MFMA_VGPR_FORM: bool = False

    # -- fused K6 stage --
    COMPUTE_OUTPUT: bool = False
    SCALE: Optional[float] = None

    arch: str = "gfx942"
    name: str = "rocke_gdn_state_scan"

    # ---------------------------------------------------------------- checks
    def __post_init__(self) -> None:
        if self.USE_G == self.USE_GK:
            raise ValueError("exactly one of USE_G / USE_GK must be set")
        if self.K % 64 != 0 or self.K > 256:
            raise ValueError(f"K must be a multiple of 64 and <= 256 (got {self.K})")
        if self.K & (self.K - 1):
            raise ValueError(f"K must be a power of two (got {self.K}); "
                             "the k store-transpose staging depends on it")
        if self.BV % 16 != 0:
            raise ValueError(f"BV must be a multiple of 16 (got {self.BV})")
        if self.BT % 4 or self.K % 4:
            raise ValueError("BT and K must both be multiples of 4")
        if self.H % self.Hg:
            raise ValueError(f"H={self.H} must be divisible by Hg={self.Hg} (GQA)")
        if (self.SCALE is not None) != self.COMPUTE_OUTPUT:
            raise ValueError("SCALE is required iff COMPUTE_OUTPUT")
        if self.PREFETCH_K_EARLY and self.PREFETCH_K_INTERLEAVE:
            raise ValueError("choose one K prefetch schedule")
        # group-XOR is a bank bijection only with >= 16 groups per row
        if self.K // 4 < 16 or self.BT // 4 < 16:
            raise ValueError("group-XOR swizzle needs >= 16 groups per row "
                             f"(K/4={self.K // 4}, BT/4={self.BT // 4})")

        if self.n_repeat % self.NR_SPLIT:
            legal = [s for s in (1, 2, 4, 8) if self.n_repeat % s == 0]
            raise ValueError(
                f"NR_SPLIT={self.NR_SPLIT} must divide N_REPEAT={self.n_repeat} "
                f"(=BV/16); BV={self.BV} supports {legal}"
            )
        if self.COMPUTE_OUTPUT and (self.BT // self.mfma_k) % self.NR_SPLIT:
            raise ValueError(
                f"NR_SPLIT={self.NR_SPLIT} must divide BT_STEPS="
                f"{self.BT // self.mfma_k} to split the attention matrix "
                "across the V-split waves"
            )

        target = ArchTarget.from_gfx(self.arch)
        if self.block_threads > target.max_threads_per_block:
            raise ValueError(
                f"BLOCK_THREADS={self.block_threads} exceeds the {self.arch} "
                f"limit ({target.max_threads_per_block}); reduce NR_SPLIT"
            )
        if not target.fits_lds(self.lds_total_bytes):
            raise ValueError(
                f"LDS {self.lds_total_bytes / 1024:.1f} KiB exceeds the "
                f"{self.arch} budget; reduce BV"
            )
        if self.STORE_H:
            if self.lds_h_ng % 2:
                raise ValueError(
                    f"h drain pairs k-groups; K/4={self.lds_h_ng} must be even")
            pairs = self.BV * (self.lds_h_ng // 2)
            if pairs % self.block_threads:
                raise ValueError(
                    f"h snapshot drain ({pairs} pairs) must tile "
                    f"BLOCK_THREADS={self.block_threads}")

    # ------------------------------------------------------------- derived
    @property
    def mfma_op_id(self) -> str:
        return pick_bf16_atom(self.arch)[0]

    @property
    def mfma_k(self) -> int:
        """Contraction depth of the chosen atom. **16 on CDNA3, 32 on CDNA4.**"""
        return pick_bf16_atom(self.arch)[1]

    @property
    def gqa_ratio(self) -> int:
        return self.H // self.Hg

    @property
    def num_k_blocks(self) -> int:
        return self.K // 64

    @property
    def n_repeat(self) -> int:
        return self.BV // 16

    @property
    def m_waves(self) -> int:
        return self.BT // 16

    @property
    def n_repeat_local(self) -> int:
        return self.n_repeat // self.NR_SPLIT

    @property
    def num_warps(self) -> int:
        return self.m_waves * self.NR_SPLIT

    @property
    def block_threads(self) -> int:
        return self.num_warps * ArchTarget.from_gfx(self.arch).wave_size

    @property
    def bt_steps(self) -> int:
        return self.BT // self.mfma_k

    @property
    def bt_steps_local(self) -> int:
        return self.bt_steps // self.NR_SPLIT

    @property
    def k_steps_per_block(self) -> int:
        return 64 // self.mfma_k

    @property
    def grid_v(self) -> int:
        return (self.V + self.BV - 1) // self.BV

    @property
    def kt_transposed(self) -> bool:
        """``k`` staged as ``[K, BT]``. True for K5-only; the fused build wants
        ``[BT, K]`` so the heavier reader (the attention GEMM) gets the
        contiguous access and the state GEMM takes the strided one."""
        return not self.COMPUTE_OUTPUT

    # -- LDS, in elements (bf16) --
    #: Trailing pad per row when the XOR swizzle is off. Swizzle needs none.
    LDS_PAD: ClassVar[int] = 8

    @property
    def _pad(self) -> int:
        return 0 if self.LDS_SWIZZLE else self.LDS_PAD

    @property
    def lds_w_elems(self) -> int:
        return self.BT * (self.K + self._pad)

    @property
    def lds_kt_elems(self) -> int:
        return self.K * (self.BT + self._pad)

    @property
    def lds_vnt_elems(self) -> int:
        return self.BV * (self.BT + self._pad)

    @property
    def lds_h_elems(self) -> int:
        return self.BV * (self.K + self._pad)

    @property
    def lds_a_elems(self) -> int:
        return self.BT * self.BT if self.COMPUTE_OUTPUT else 0

    @property
    def lds_h_ng(self) -> int:
        return self.K // 4

    @property
    def lds_total_bytes(self) -> int:
        elems = (self.lds_w_elems + self.lds_kt_elems + self.lds_vnt_elems
                 + self.lds_h_elems + self.lds_a_elems)
        return elems * 2

    @property
    def alias_a_onto_h(self) -> bool:
        """Reuse the ``h`` buffer for the attention matrix. Only engaged when
        the five buffers would otherwise overflow — at K=128/BV=64 they come to
        exactly the CDNA3 budget, so this is inactive there and the path exists
        for configurations that do overflow."""
        budget = 65536
        return (self.COMPUTE_OUTPUT
                and self.lds_total_bytes > budget
                and self.lds_a_elems <= self.lds_h_elems)

    #: Manifest dtype of the initial / final SSM state buffers (``H0`` / ``Ht``).
    @property
    def state_dtype(self) -> str:
        return "bf16" if self.STATE_DTYPE_BF16 else "f32"

    #: Manifest dtype of the materialized per-chunk outputs (``Vnew`` / ``Hout``).
    @property
    def output_dtype(self) -> str:
        return "bf16" if self.OUTPUT_DTYPE_BF16 else "f32"

    def kernel_name(self) -> str:
        gate = "gk" if self.USE_GK else "g"
        suffix = "_o" if self.COMPUTE_OUTPUT else ""
        return (f"{self.name}{suffix}_{gate}_K{self.K}_V{self.V}"
                f"_bt{self.BT}_bv{self.BV}_w{self.num_warps}")

    def grid(self, n_seq_heads: int) -> Tuple[int, int, int]:
        """``(V-tiles, sequence*head, 1)``."""
        return (self.grid_v, n_seq_heads, 1)

    def describe(self) -> str:
        return (
            f"{self.kernel_name()}\n"
            f"  arch={self.arch} atom={self.mfma_op_id} mfma_k={self.mfma_k}\n"
            f"  waves={self.num_warps} threads={self.block_threads} "
            f"m_waves={self.m_waves} nr_split={self.NR_SPLIT} "
            f"n_repeat={self.n_repeat}/{self.n_repeat_local}\n"
            f"  k_blocks={self.num_k_blocks} bt_steps={self.bt_steps} "
            f"k_steps_per_block={self.k_steps_per_block} "
            f"kt_transposed={self.kt_transposed}\n"
            f"  lds={self.lds_total_bytes / 1024:.1f} KiB "
            f"(w={self.lds_w_elems * 2 // 1024} kt={self.lds_kt_elems * 2 // 1024} "
            f"vnt={self.lds_vnt_elems * 2 // 1024} h={self.lds_h_elems * 2 // 1024} "
            f"A={self.lds_a_elems * 2 // 1024} KiB) alias_A={self.alias_a_onto_h}\n"
            f"  grid_v={self.grid_v}"
        )


def is_valid_config(**fields) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a candidate set of :class:`GdnStateScanSpec`
    fields, without requiring that they be constructible.

    :meth:`GdnStateScanSpec.__post_init__` rejects illegal combinations in the
    constructor, which is what makes a spec instance trustworthy but also means
    a sweep cannot build one in order to test it. This is the probe for that
    case: pass the candidate fields, get a reason instead of an exception. An
    unknown field name comes back as a reason too, so a stale sweep table
    reports itself rather than crashing the driver.
    """
    try:
        spec = GdnStateScanSpec(**fields)
        # Not covered by __post_init__: it only reaches spec.mfma_k when
        # COMPUTE_OUTPUT is set, so an arch without a 16x16 bf16 atom would
        # otherwise surface at the first mfma_k use inside the builder.
        pick_bf16_atom(spec.arch)
    except (ValueError, KeyError, TypeError) as e:
        return False, str(e).strip("'\"")
    return True, "ok"


def is_valid_spec(
    spec: GdnStateScanSpec, arch: Optional[str] = None
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for *spec*, optionally retargeted to *arch*.

    The non-throwing validity probe every instance module exposes, for
    dispatcher-style enumeration where an unsupported configuration is a skip
    rather than an error. Retargeting is the case that can actually fail for an
    already-constructed spec: the LDS budget, the thread cap, and the MFMA atom
    all come from the arch, so a spec that fits gfx942 need not fit elsewhere.
    """
    # dataclass_fields(), not __dataclass_fields__: the latter also reports the
    # LDS_PAD ClassVar, which is not a constructor argument.
    cfg = {f.name: getattr(spec, f.name) for f in dataclass_fields(spec) if f.init}
    if arch is not None:
        cfg["arch"] = arch
    return is_valid_config(**cfg)


def gdn_state_scan_signature(spec: GdnStateScanSpec) -> List[dict]:
    """The kernel's ABI, as the manifest-style list
    :class:`rocke.runtime.launcher.KernelLauncher` binds against.

    This is the interface contract for :func:`build_gdn_state_scan`: same names,
    same order, same dtypes as the ``b.param`` sequence the builder emits, which
    that builder asserts before returning. Two spec flags move the pointer
    dtypes — ``STATE_DTYPE_BF16`` for the SSM state carried across chunks
    (``H0`` / ``Ht``) and ``OUTPUT_DTYPE_BF16`` for the materialized per-chunk
    outputs (``Vnew`` / ``Hout``) — and ``IS_VARLEN`` appends the packed-sequence
    descriptors.
    """
    st, out = spec.state_dtype, spec.output_dtype
    sig = (
        SignatureBuilder()
        .ptr("Kt", "bf16")
        .ptr("Wt", "bf16")
        .ptr("Ut", "bf16")
        .ptr("Gate", "f32")
        .ptr("H0", st)
        .ptr("Vnew", out)
        .ptr("Hout", out)
        .ptr("Ht", st)
    )
    if spec.COMPUTE_OUTPUT:
        sig = sig.ptr("Qt", "bf16").ptr("O", "bf16")
    sig = (
        sig.scalar("T_val", "i32")
        .scalar("NT_val", "i32")
        .scalar("N_val", "i32")
    )
    if spec.IS_VARLEN:
        sig = (
            sig.ptr("cu_seqlens", "i32")
            .ptr("chunk_offsets", "i32")
            .scalar("T_flat", "i32")
        )
    return sig.build()


def gdn_state_scan_grid(
    spec: GdnStateScanSpec, n_seq_heads: int
) -> Tuple[int, int, int]:
    """``(ceil(V / BV), N * H, 1)``. ``n_seq_heads`` is ``N * H``."""
    return spec.grid(n_seq_heads)



class _Loader:
    """Global loads, either raw-pointer or through a bounds-checked descriptor.

    P6. With ``use_desc`` the resource carries ``num_records`` in *bytes*, so
    hardware returns 0 for any out-of-range access and the caller can drop its
    explicit row clamp. ``voffset`` is a byte offset; callers keep working in
    elements and the conversion happens here.
    """

    def __init__(self, b, ptr, *, n_elems=None, elem_bytes=2, use_desc=False):
        """``n_elems`` is a *thunk* returning a Value, so the size arithmetic is
        emitted only when a descriptor is actually built. With the flag off the
        emitted IR is identical to the raw-pointer version — no dead ops to
        perturb the already-verified default path."""
        self.b, self.ptr, self.eb = b, ptr, elem_bytes
        self.zero = b.const_i32(0) if use_desc else None
        self.rsrc = (b.buffer_rsrc(ptr, b.mul(n_elems(), b.const_i32(elem_bytes)))
                     if use_desc else None)

    @property
    def bounds_checked(self) -> bool:
        return self.rsrc is not None

    def _bytes(self, off):
        return self.b.mul(off, self.b.const_i32(self.eb))

    def vN(self, off, dtype, n):
        if self.rsrc is None:
            return self.b.global_load_vN(self.ptr, off, dtype, n)
        return self.b.buffer_load_vN(self.rsrc, self._bytes(off), self.zero, dtype, n)

    def scalar(self, off, dtype):
        if self.rsrc is None:
            return (self.b.global_load_bf16(self.ptr, off) if dtype is BF16
                    else self.b.global_load_f32(self.ptr, off))
        if dtype is BF16:
            return self.b.buffer_load_bf16(self.rsrc, self._bytes(off), self.zero)
        return self.b.buffer_load(self.rsrc, self._bytes(off), self.zero, dtype)


def _grp_col(b, row, col, ng):
    """Group-major + folded-XOR swizzled column.

    A logical ``[R, C]`` tile is stored ``[R][C/4][4]``: each row is ``ng = C/4``
    groups of 4 bf16 (8 B — one MFMA fragment), and the group index is XORed by
    a key that includes row bits 3+ to cover transposed store patterns.
    """
    grp = b.lshr(col, b.const_i32(2))
    mask = b.land(b.xor(row, b.lshr(row, b.const_i32(3))),
                  b.const_i32(ng - 1))
    return b.shl(b.xor(grp, mask), b.const_i32(2))


def _swz(b, row, col, ng):
    """``_grp_col`` when swizzling, identity when not."""
    return col if ng is None else _grp_col(b, row, col, ng)


def _swz_elem(b, row, col, ng):
    """Swizzled address of one element, preserving its in-group low bits."""
    if ng is None:
        return col
    return b.add(_grp_col(b, row, col, ng),
                 b.land(col, b.const_i32(3)))


def _regroup(b, v, base, n=4):
    """Extract ``n`` consecutive lanes of a wider vector into their own vector."""
    out = b.vector_splat(b.vec_extract(v, base), n)
    for j in range(1, n):
        out = b.vec_insert(out, b.vec_extract(v, base + j), j)
    return out


def _stage_tile(b, *, src_ptr, smem, rows, cols, row_stride_src, src_row_base,
                block_threads, tid, ng=None, elem_off=None, clamp=None):
    """Cooperatively stage a ``[rows, cols]`` bf16 tile from global into LDS.

    Slot ``s = it * block_threads + tid`` walks the tile in units of
    ``STAGE_VEC`` contiguous columns, so ``cols // STAGE_VEC`` consecutive tids
    cover one row segment — coalesced on the global side, and contiguous on the
    LDS side (the pad sits at the end of a row, never inside a segment).

    ``elem_off`` is added to the flat source offset (used to select a head).
    ``clamp`` bounds the *source row*: rows at or past it read row 0 instead, so
    a tail chunk never reads out of range. The values are garbage but harmless —
    the tail-chunk row mask (N2) zeroes anything they feed.
    """
    regs = _stage_tile_load(b, src_ptr=src_ptr, rows=rows, cols=cols,
                            row_stride_src=row_stride_src,
                            src_row_base=src_row_base,
                            block_threads=block_threads, tid=tid,
                            elem_off=elem_off, clamp=clamp)
    _stage_tile_store(b, smem=smem, regs=regs, rows=rows, cols=cols,
                      block_threads=block_threads, tid=tid, ng=ng)


def _tile_slots(b, *, rows, cols, block_threads, tid):
    """Per-iteration ``(row, col)`` for the cooperative tile walk.

    Shared by the load and store halves so the two agree by construction — they
    now run in *different loop iterations* (P5), so a drifting decomposition
    would be a silent corruption rather than a compile error.
    """
    vecs_per_row = cols // STAGE_VEC
    total_slots = rows * vecs_per_row
    assert total_slots % block_threads == 0, (
        f"tile [{rows},{cols}] = {total_slots} vec{STAGE_VEC} slots must tile "
        f"block_threads={block_threads}")
    c_vpr = b.const_i32(vecs_per_row)
    out = []
    for it in range(total_slots // block_threads):
        slot = b.add(b.const_i32(it * block_threads), tid)
        out.append((b.div(slot, c_vpr),
                    b.mul(b.mod(slot, c_vpr), b.const_i32(STAGE_VEC))))
    return out


def _stage_tile_load(b, *, src_ptr, rows, cols, row_stride_src, src_row_base,
                     block_threads, tid, elem_off=None, clamp=None, ldr=None):
    """Issue only the global loads for a tile; return the raw vectors.

    **Raw loads only** — no conversion, no select on the *value*, no packing.
    Anything that consumes a load here would force a wait in the issuing
    iteration and defeat the prefetch (P5).
    """
    c_srcstride = b.const_i32(row_stride_src)
    regs = []
    for row, col in _tile_slots(b, rows=rows, cols=cols,
                                block_threads=block_threads, tid=tid):
        src_row = b.add(row, src_row_base) if src_row_base is not None else row
        # A bounds-checked descriptor makes the clamp redundant: an out-of-range
        # row reads a hardware zero instead of row 0, and the tail-chunk mask
        # (N2) zeroes whatever it feeds either way.
        if clamp is not None and not (ldr is not None and ldr.bounds_checked):
            src_row = b.select(b.cmp_lt(src_row, clamp), src_row, b.const_i32(0))
        off = b.add(b.mul(src_row, c_srcstride), col)
        if elem_off is not None:
            off = b.add(off, elem_off)
        regs.append(ldr.vN(off, BF16, STAGE_VEC) if ldr is not None
                    else b.global_load_vN(src_ptr, off, BF16, STAGE_VEC))
    return regs


def _stage_tile_store(b, *, smem, regs, rows, cols, block_threads, tid, ng=None):
    """Write previously-loaded tile registers into LDS."""
    slots = _tile_slots(b, rows=rows, cols=cols,
                        block_threads=block_threads, tid=tid)
    assert len(slots) == len(regs)
    for (row, col), v in zip(slots, regs):
        if ng is None:
            b.smem_store_vN(smem, [row, col], v, STAGE_VEC)
        else:
            # 8 contiguous elements span two 4-element groups, and the XOR can
            # place them anywhere in the row — so they become two stores.
            for g in range(STAGE_VEC // 4):
                c = b.add(col, b.const_i32(g * 4))
                b.smem_store_vN(smem, [row, _grp_col(b, row, c, ng)],
                                _regroup(b, v, g * 4), 4)


def _init_k_transposed(b, *, spec, tid, nthreads):
    """Return per-thread K-transpose records with no global loads issued."""
    BT, K = spec.BT, spec.K
    KVW = min(STAGE_VEC, max(2, (BT // 4) * K // nthreads))
    assert KVW & (KVW - 1) == 0, f"K vector width {KVW} must be a power of two"
    assert K % KVW == 0, f"K={K} must be divisible by the vector width {KVW}"
    col_groups = K // KVW
    slots = (BT // 4) * col_groups
    assert slots % nthreads == 0, (
        f"k transpose slots ({slots}) must tile block_threads={nthreads}")
    records = []
    c_cg = b.const_i32(col_groups)
    for it in range(slots // nthreads):
        slot = b.add(b.const_i32(it * nthreads), tid)
        bt0 = b.mul(b.div(slot, c_cg), b.const_i32(4))
        k0 = b.mul(b.mod(slot, c_cg), b.const_i32(KVW))
        records.append((bt0, k0, []))
    return records


def _load_k_transposed_row(b, *, src, staged, row, spec, t_base, i_hg,
                           T_val, Hg, ldr=None, bos=None):
    """Append one of the four row loads to each K-transpose record."""
    KVW = min(STAGE_VEC, max(2, (spec.BT // 4) * spec.K // spec.block_threads))
    for bt0, k0, rows in staged:
        t_loc = b.add(b.add(t_base, bt0), b.const_i32(row))
        in_range = b.cmp_lt(t_loc, T_val)
        t_loc = (t_loc if (ldr is not None and ldr.bounds_checked)
                 else b.select(in_range, t_loc, b.const_i32(0)))
        t_glob = b.add(bos, t_loc) if bos is not None else t_loc
        off = b.add(b.mul(b.add(b.mul(t_glob, b.const_i32(Hg)), i_hg),
                          b.const_i32(spec.K)), k0)
        rows.append(ldr.vN(off, BF16, KVW) if ldr is not None
                    else b.global_load_vN(src, off, BF16, KVW))


def _stage_k_transposed(b, *, src, smem, spec, tid, nthreads, t_base, i_hg,
                        T_val, Hg, ng=None, ldr=None, bos=None, staged=None):
    """Stage ``k[BT, K]`` from global into ``sKT[K, BT]`` — a real transpose.

    gfx942 has no ``ds_read_*_tr_*`` (CDNA4 only), so a transposed operand must
    be built on the **store** side.

    **P2, matching the FlyDSL parent's decomposition.** A thread takes four
    **BT-consecutive** rows at the same K columns. Per K column those four
    values then form one ``bt``-group — contiguous in ``sKT[k, bt]`` — so an
    in-register gather turns what would be four scattered ``ds_write_b16`` into
    a single packed ``ds_write_b64``.

    The row quad stays 4 regardless of block size (that is what makes the store
    packed), so a wider block costs *load* width rather than store width::

        256 thr -> vec8 (dwordx4) | 512 thr -> vec4 | 1024 thr -> vec2
    """
    BT, K = spec.BT, spec.K
    # Widest K run a thread can take while every (row-quad, col-group) slot
    # still tiles the block. Same formula as the parent.
    KVW = min(STAGE_VEC, max(2, (BT // 4) * K // nthreads))
    assert KVW & (KVW - 1) == 0, f"K vector width {KVW} must be a power of two"
    assert K % KVW == 0, f"K={K} must be divisible by the vector width {KVW}"
    col_groups = K // KVW
    row_quads = BT // 4
    slots = row_quads * col_groups
    assert slots % nthreads == 0, (
        f"k transpose slots ({slots}) must tile block_threads={nthreads}")

    if staged is None:
        staged = _init_k_transposed(
            b, spec=spec, tid=tid, nthreads=nthreads,
        )
        for r in range(4):
            _load_k_transposed_row(
                b, src=src, staged=staged, row=r, spec=spec,
                t_base=t_base, i_hg=i_hg, T_val=T_val, Hg=Hg,
                ldr=ldr, bos=bos,
            )

    if smem is None:
        return staged

    for bt0, k0, rows in staged:
        for j in range(KVW):
            col = b.vector_splat(b.vec_extract(rows[0], j), 4)
            for r in range(1, 4):
                col = b.vec_insert(col, b.vec_extract(rows[r], j), r)
            row = b.add(k0, b.const_i32(j))
            b.smem_store_vN(smem, [row, _swz(b, row, bt0, ng)], col, 4)
    return staged


def _drain_h(b, *, sH, dst, spec, tid, nthreads, chunk, i_h, v_base, ng=None):
    """Drain the ``[BV, K]`` state snapshot from LDS to ``Hout[NT, H, V, K]``.

    All threads cooperate; each handles ``STAGE_VEC`` contiguous K values, which
    are contiguous in both LDS (K-major) and HBM (K innermost).
    """
    BV, K, V, H = spec.BV, spec.K, spec.V, spec.H
    vecs_per_row = K // STAGE_VEC
    total = BV * vecs_per_row
    assert total % nthreads == 0, (
        f"h drain ({total} vec slots) must tile block_threads={nthreads}")
    c_vpr = b.const_i32(vecs_per_row)
    for it in range(total // nthreads):
        slot = b.add(b.const_i32(it * nthreads), tid)
        row = b.div(slot, c_vpr)
        k0 = b.mul(b.mod(slot, c_vpr), b.const_i32(STAGE_VEC))
        v_abs = b.add(v_base, row)
        base = b.add(b.mul(b.add(b.mul(b.add(b.mul(chunk, b.const_i32(H)), i_h),
                                       b.const_i32(V)), v_abs),
                           b.const_i32(K)), k0)
        # P4: one wide LDS read, then stores in 4-wide groups. The values
        # are contiguous in both LDS (K-major) and HBM (K innermost), so the
        # only reason this was ever scalar was expedience.
        for g in range(STAGE_VEC // 4):
            c = b.add(k0, b.const_i32(g * 4))
            val = b.smem_load_vN(sH, row, _swz(b, row, c, ng), dtype=BF16, n=4)
            quad = (val if spec.OUTPUT_DTYPE_BF16 else _pack_f32x4(
                b, [b.cast_to_f32(b.vec_extract(val, j)) for j in range(4)]))
            b.global_store_vN(dst, b.add(base, b.const_i32(g * 4)), quad, 4)


# ---------------------------------------------------------------------------
# The full K5 chunk recurrence
# ---------------------------------------------------------------------------
#
# Operand assignments, derived from the atom contract at the top of this file:
#
#   GEMM1  bv[BT, BV] = w[BT, K] @ h[BV, K]^T          contract over K
#     A <- w    from sW  : A[m=bt,   kc=k ]   row=bt,   4 consecutive k
#     B <- h    from sH  : B[kc=k,   n=v  ] == h[v, k], row=v, 4 consecutive k
#     D          -> bv[bt = wid_m*16 + lmb*4 + e, v = tile*16 + lane_n]
#
#   GEMM2  h[BV, K] += k^T[K, BT] @ v_new^T[BV, BT]    contract over BT
#     A <- k    from sKT : A[m=kdim, kc=bt]   row=kdim, 4 consecutive bt
#     B <- vnew from sVN : B[kc=bt,  n=v  ]   row=v,    4 consecutive bt
#     D          -> h[k = kb*64 + wid_m*16 + lmb*4 + e, v = tile*16 + lane_n]
#
# Note the axis swap: `wid_m` selects the **BT** tile in GEMM1 and the **K**
# tile in GEMM2. That is what makes each lane's four accumulator slots four
# *consecutive K values at one V* — exactly the VK storage order — so writing
# the state back out is a contiguous 4-wide store rather than a scatter.


def _to_bf16_fast(b, x, n: int = 1):
    """Round-half-away-from-zero f32 -> bf16 (numerics item N1).

    ``(bitcast_u32(x) + 0x8000) >> 16``, truncated. CDNA3 has no
    ``v_cvt_pk_bf16_f32``, so the compiler's RNE path expands to several VALU
    ops per element; this is four. More importantly the bias is *symmetric* —
    plain truncation is one-sided, and over a serial scan that bias compounds.

    With ``n > 1`` the whole conversion runs on an ``<n x f32>`` vector, so the
    cost is four VALU ops for the *group* rather than four per element. That is
    the form the FlyDSL parent uses, and it is what makes a packed
    ``ds_write_b64`` worth doing (P1).
    """
    if n == 1:
        bits = b.add(b.bitcast(x, I32), b.const_i32(0x8000))
        return b.bitcast(b.trunc(b.lshr(bits, b.const_i32(16)), I16), BF16)
    bits = b.bitcast(x, VectorType(I32, n))
    bits = b.add(bits, b.vector_splat(b.const_i32(0x8000), n))
    bits = b.lshr(bits, b.vector_splat(b.const_i32(16), n))
    return b.bitcast(b.trunc(bits, VectorType(I16, n)), VectorType(BF16, n))


def _pack_f32x4(b, vals):
    """Assemble four f32 scalars into an ``<4 x f32>``."""
    v = b.vector_splat(vals[0], 4)
    for i in range(1, 4):
        v = b.vec_insert(v, vals[i], i)
    return v


def build_gdn_state_scan(spec: GdnStateScanSpec) -> KernelDef:
    """Full K5 recurrence with the optional fused K6 output stage.

    The emitted kernel binds against :func:`gdn_state_scan_signature` and
    launches on :func:`gdn_state_scan_grid`. Call :func:`is_valid_spec` first if
    *spec* comes from a sweep; an unsupported configuration raises here.

    Layouts (row-major, token-major)::

        Kt   bf16 [T, Hg, K]      Wt bf16 [T, H, K]      Ut bf16 [T, H, V]
        Gk   f32  [T, H,  K]      H0 f32  [N, H, V, K]
        Vnew out  [T, H,  V]      Hout out [NT, H, V, K]  Ht state [N, H, V, K]
        Qt   bf16 [T, Hg, K]      O bf16 [T, H, V]       (fused K6 only)

    Grid ``(ceil(V/BV), N*H, 1)``.
    """
    # STATE_DTYPE_BF16: the initial/final SSM state lives in bf16 in HBM to halve
    # its bandwidth/footprint; the kernel still accumulates in f32 and converts
    # only at the H0 load and the Ht store (matching the parent). The per-chunk
    # OUTPUT_DTYPE_BF16 independently controls the materialized K5 outputs.
    state_dt = BF16 if spec.STATE_DTYPE_BF16 else F32
    state_bytes = 2 if spec.STATE_DTYPE_BF16 else 4
    output_dt = BF16 if spec.OUTPUT_DTYPE_BF16 else F32

    atom = mfma_atom("bf16", 16, 16, spec.mfma_k)
    BT, K, BV, V, H, Hg = spec.BT, spec.K, spec.BV, spec.V, spec.H, spec.Hg
    NKB, NRL, MW = spec.num_k_blocks, spec.n_repeat_local, spec.m_waves
    nthreads = spec.block_threads
    # P3: swizzled buffers need no padding; the unswizzled fallback does.
    pad = 0 if spec.LDS_SWIZZLE else LDS_PAD
    wcols, tcols = K + pad, BT + pad
    ng_k = (K // 4) if spec.LDS_SWIZZLE else None    # sW, sH  (K-major rows)
    ng_t = (BT // 4) if spec.LDS_SWIZZLE else None   # sKT, sVN (BT-major rows)

    # P5 is gated: see PERF_PLAN.md for the unresolved ordering sensitivity.
    _PF_W = _PF_U = _PF_G = spec.PREFETCH

    def _exp_gate(x):
        """exp(x) for a gate value.

        The gates arrive as natural-log cumsums, and the kernel wants exp(). We
        route through the hardware ``exp2`` (``exp2_fast``, valid since gate
        args are <= 0), so exp(x) = exp2(x * log2(e)). When the upstream stages
        already emit gates in the **log2 domain** (``G_IS_LOG2_SCALED``), the
        multiply is redundant and dropped — matching the parent's
        ``_make_fast_exp``.
        """
        if spec.G_IS_LOG2_SCALED:
            return b.exp2_fast(x)
        return b.exp2_fast(b.fmul(x, b.const_f32(LOG2E)))

    b = IRBuilder(spec.kernel_name() + "_k5")
    b.kernel.attrs["max_workgroup_size"] = nthreads
    if spec.MFMA_VGPR_FORM:
        # Keep the accumulators in VGPRs: they are VALU-touched every chunk by
        # the gate multiply, and the AGPR form pays accvgpr copies for it.
        b.kernel.attrs["agpr_alloc"] = (0, 0)

    Kt = b.param("Kt", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Wt = b.param("Wt", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Ut = b.param("Ut", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    # One gate pointer, interpreted per spec: [T, H, K] on the per-channel
    # path, head-major [H, T] on the scalar path.
    Gate = b.param("Gate", PtrType(F32, "global"), noalias=True, readonly=True,
                   align=16)
    H0 = b.param("H0", PtrType(state_dt, "global"), noalias=True, readonly=True, align=16)
    Vn = b.param("Vnew", PtrType(output_dt, "global"), noalias=True, align=16)
    Ho = b.param("Hout", PtrType(output_dt, "global"), noalias=True, align=16)
    Ht = b.param("Ht", PtrType(state_dt, "global"), noalias=True, align=16)
    if spec.COMPUTE_OUTPUT:
        Qt = b.param(
            "Qt", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
        )
        O_ptr = b.param("O", PtrType(BF16, "global"), noalias=True, align=16)
    T_val = b.param("T_val", I32)
    NT_val = b.param("NT_val", I32)
    N_val = b.param("N_val", I32)          # sequences; grid.y == N_val * H
    if spec.IS_VARLEN:
        # cu_seqlens[i_n], cu_seqlens[i_n+1] bound this sequence; chunk_offsets
        # gives its base chunk index in the packed [sum(NT), H, V, K] snapshot.
        # T_flat is the packed token count (= cu_seqlens[-1]); w/u are addressed
        # against it in the head-major (WU_CONTIGUOUS) layout.
        CU = b.param("cu_seqlens", PtrType(I32, "global"), noalias=True, readonly=True)
        CO = b.param("chunk_offsets", PtrType(I32, "global"), noalias=True, readonly=True)
        T_flat = b.param("T_flat", I32)

    sW = b.smem_alloc(BF16, [BT, wcols], name_hint="sW")
    sH = b.smem_alloc(BF16, [BV, wcols], name_hint="sH")
    sKT = (
        b.smem_alloc(BF16, [K, tcols], name_hint="sKT")
        if spec.kt_transposed
        else b.smem_alloc(BF16, [BT, wcols], name_hint="sK")
    )
    sVN = b.smem_alloc(BF16, [BV, tcols], name_hint="sVN")
    if spec.COMPUTE_OUTPUT:
        # Keep A independent from q (sW) and the state snapshot (sH). Source
        # order alone is not a cross-wave lifetime boundary; `exclusive`
        # prevents the LDS pool packer from reusing either live tile.
        sA = b.smem_alloc(BF16, [BT, BT], name_hint="sA", exclusive=True)

    tid = b.thread_id_x()
    cH, cV, cK = b.const_i32(H), b.const_i32(V), b.const_i32(K)

    if spec.XCD_REMAP:
        # P7. A head's GRID_V V-tiles all read the same V-independent w / k /
        # gate slices. Under the hardware default (flat block `xy` runs on XCD
        # `xy % NXCD`) those tiles scatter across XCDs and the slices land in up
        # to NXCD separate private L2s. Invert the round-robin so runs of GRID_V
        # consecutive logical ids share an XCD, then unflatten.
        #
        # Ported from the parent, tail guard included: ids past the last full
        # NXCD*GRID_V cycle pass through unchanged, since remapping them would
        # collide or run out of range. So any grid is at least as good as the
        # round-robin baseline.
        GV = spec.grid_v
        cGV, cNX = b.const_i32(GV), b.const_i32(NXCD)
        grid_total = b.mul(cGV, b.mul(N_val, cH))
        xy = b.add(b.block_id_x(), b.mul(cGV, b.block_id_y()))
        xcd = b.mod(xy, cNX)
        cycle = b.const_i32(NXCD * GV)
        last_full = b.mul(b.div(grid_total, cycle), cycle)
        local_id = b.div(xy, cNX)
        remapped = b.add(b.mul(b.div(local_id, cGV), cycle),
                         b.add(b.mul(xcd, cGV), b.mod(local_id, cGV)))
        logical = b.select(b.cmp_lt(xy, last_full), remapped, xy)
        i_v = b.mod(logical, cGV)
        nh = b.div(logical, cGV)
    else:
        i_v = b.block_id_x()
        nh = b.block_id_y()

    i_n = b.div(nh, cH)
    i_h = b.mod(nh, cH)
    i_hg = b.div(i_h, b.const_i32(spec.gqa_ratio))
    v_base = b.mul(i_v, b.const_i32(BV))

    # ---- per-sequence prologue: bos / T_local / nt / chunk-snapshot base ----
    # Non-varlen collapses to a single sequence of T_val tokens per i_n; varlen
    # reads the sequence bounds from cu_seqlens and its snapshot base from
    # chunk_offsets, matching the parent exactly.
    if spec.IS_VARLEN:
        bos = b.buffer_load(b.buffer_rsrc(CU, b.mul(b.add(N_val, b.const_i32(1)),
                                                    b.const_i32(4))),
                            b.mul(i_n, b.const_i32(4)), b.const_i32(0), I32)
        eos = b.buffer_load(b.buffer_rsrc(CU, b.mul(b.add(N_val, b.const_i32(1)),
                                                    b.const_i32(4))),
                            b.mul(b.add(i_n, b.const_i32(1)), b.const_i32(4)),
                            b.const_i32(0), I32)
        T_local = b.sub(eos, bos)
        nt_local = b.div(b.add(T_local, b.const_i32(BT - 1)), b.const_i32(BT))
        chunk_base = b.buffer_load(
            b.buffer_rsrc(CO, b.mul(N_val, b.const_i32(4))),
            b.mul(i_n, b.const_i32(4)), b.const_i32(0), I32)
    else:
        bos = b.mul(i_n, T_val)
        T_local = T_val
        nt_local = NT_val
        chunk_base = b.mul(i_n, NT_val)

    c16, c4, c64 = b.const_i32(16), b.const_i32(4), b.const_i32(64)
    wid, lane = b.div(tid, c64), b.mod(tid, c64)
    wid_m, wid_n = b.mod(wid, b.const_i32(MW)), b.div(wid, b.const_i32(MW))
    lane_n, lmb = b.mod(lane, c16), b.div(lane, c16)
    k_lane = b.mul(lmb, c4)

    # this wave's V rows — shared by GEMM1's D, GEMM2's B, and the state layout
    v_tile = [b.mul(b.add(b.mul(wid_n, b.const_i32(NRL)), b.const_i32(nr)), c16)
              for nr in range(NRL)]
    v_row = [b.add(t, lane_n) for t in v_tile]

    # GEMM2 K tile owned by this wave: k = kb*64 + wid_m*16 + (lmb*4 + e)
    k_tile = [b.add(b.const_i32(kb * 64), b.mul(wid_m, c16)) for kb in range(NKB)]

    # ---- w / u / k / gate base offsets and per-row strides -----------------
    # Centralised so IS_VARLEN and WU_CONTIGUOUS live in one place. All are
    # element offsets; `tok` is the token index WITHIN the sequence (0..T_local),
    # so `bos + tok` is the packed/global token.
    #   w  head-major  : (i_h*T_flat + bos + tok)*K + col     stride K
    #      token-major : ((bos + tok)*H + i_h)*K + col        stride H*K
    #   u  = w with V and the CTA's V-window folded into the base
    #   k  always      : ((bos + tok)*Hg + i_hg)*K + col      stride Hg*K
    #   gk head-major  : ((bos + tok)*H + i_h)*K + col        (per-channel gate)
    #   g  head-major  : i_h*T_flat + bos + tok               (scalar gate)
    # Head-major row-base offset (in *rows*), matching the parent's two cases:
    #   varlen     : i_h*T_flat + bos        (one packed run per head)
    #   non-varlen : (i_n*H + i_h)*T_val      (per-(seq, head))
    # Token-major uses (bos + tok)*H + i_h and needs no flat length.
    if spec.WU_CONTIGUOUS:
        if spec.IS_VARLEN:
            _wu_row = b.add(b.mul(i_h, T_flat), bos)
        else:
            _wu_row = b.mul(b.add(b.mul(i_n, cH), i_h), T_val)
        w_stride, u_stride = K, V
        _w_base = b.mul(_wu_row, cK)
        _u_row_base = b.mul(_wu_row, cV)
    else:
        w_stride, u_stride = H * K, H * V
        _w_base = b.mul(b.add(b.mul(bos, cH), i_h), cK)
        _u_row_base = b.mul(b.add(b.mul(bos, cH), i_h), cV)
    # u: fold this CTA's V window into the base
    _u_base = b.add(_u_row_base, v_base)
    # Scalar gate is head-major. Preserve the established K5-only addressing;
    # fused K6 uses the full dense [N, H, T] batch/head row.
    if spec.IS_VARLEN:
        _g_base = b.add(b.mul(i_h, T_flat), bos)
    elif spec.COMPUTE_OUTPUT:
        _g_base = b.mul(b.add(b.mul(i_n, cH), i_h), T_val)
    else:
        _g_base = b.add(b.mul(i_h, T_val), bos)
    if spec.COMPUTE_OUTPUT:
        # q shares k's token-major [T, Hg, K] layout.
        _qk_base = b.mul(b.add(b.mul(bos, b.const_i32(Hg)), i_hg), cK)

    def wu_row_off(base, tok, stride):
        """Flat element offset of row `tok` (within-sequence) of a w/u tensor."""
        return b.add(base, b.mul(tok, b.const_i32(stride)))

    def state_off(kb, nr, seq):
        """Flat offset of this lane's 4 state values in an [N, H, V, K] tensor."""
        v_abs = b.add(v_base, v_row[nr])
        k0 = b.add(k_tile[kb], k_lane)
        return b.add(b.mul(b.add(b.mul(b.add(b.mul(seq, cH), i_h), cV), v_abs), cK), k0)

    # P6: one bounds-checked resource per loaded tensor. Sizes are exact, so an
    # out-of-range access returns 0 rather than reading a neighbour.
    _bd = spec.BUFFER_DESC
    ld_w = _Loader(b, Wt, elem_bytes=2, use_desc=_bd,
                   n_elems=lambda: b.mul(b.mul(T_val, cH), cK))
    ld_k = _Loader(b, Kt, elem_bytes=2, use_desc=_bd,
                   n_elems=lambda: b.mul(b.mul(T_val, b.const_i32(Hg)), cK))
    ld_u = _Loader(b, Ut, elem_bytes=2, use_desc=_bd,
                   n_elems=lambda: b.mul(b.mul(T_val, cH), cV))
    ld_g = _Loader(b, Gate, elem_bytes=4, use_desc=_bd,
                   n_elems=(lambda: b.mul(b.mul(T_val, cH), cK)) if spec.USE_GK
                           else (lambda: b.mul(cH, T_val)))
    ld_h0 = _Loader(b, H0, elem_bytes=state_bytes, use_desc=_bd,
                    n_elems=lambda: b.mul(b.mul(b.mul(N_val, cH), cV), cK))
    if spec.COMPUTE_OUTPUT:
        ld_q = _Loader(
            b,
            Qt,
            elem_bytes=2,
            use_desc=_bd,
            n_elems=lambda: b.mul(b.mul(T_val, b.const_i32(Hg)), cK),
        )

    # ---- P5: loop-carried prefetch -------------------------------------
    # Issue chunk i+1's HBM reads at the end of chunk i and carry the raw
    # values across the back edge, so their latency sits behind this chunk's
    # MFMA chain instead of in front of the next one.
    #
    # Two rules, both load-bearing:
    #   * RAW LOADS ONLY. No exp, no select-on-value, no packing at issue time
    #     — anything that consumes a load here forces a wait here.
    #   * issue/unpack are structural inverses. scf_yield ordering is unchecked
    #     and this list is long, so they walk the same sequence by construction.
    def _pf_issue_parts(i_t_n, *, issue_w=True, issue_ug=True):
        """Raw reads for one chunk, split into ``w`` / ``u`` / gate parts."""
        t_b = b.mul(i_t_n, b.const_i32(BT))
        w_part = [] if not (_PF_W and issue_w) else list(_stage_tile_load(
            b, src_ptr=Wt, rows=BT, cols=K, row_stride_src=w_stride,
            src_row_base=t_b, block_threads=nthreads, tid=tid,
            elem_off=_w_base, clamp=T_local, ldr=ld_w))
        u_part = []
        # u, gathered in the MMA's C-fragment layout
        for nr in (range(NRL) if (_PF_U and issue_ug) else []):
            for e in range(atom.c_per_lane):
                r_in, c_in = atom.lane_to_output(b, lane, e)
                t_abs = b.add(t_b, b.add(b.mul(wid_m, c16), r_in))
                t_safe = b.select(b.cmp_lt(t_abs, T_local), t_abs, b.const_i32(0))
                col = b.add(b.add(v_tile[nr], c_in), b.const_i32(0))
                u_part.append(ld_u.scalar(
                    b.add(wu_row_off(_u_base, t_safe, u_stride), col), BF16))
        g_part = []
        # gate
        t_last = b.sub(b.smin(b.add(t_b, b.const_i32(BT)), T_local), b.const_i32(1))
        if not (_PF_G and issue_ug):
            pass
        elif spec.USE_GK:
            row = b.mul(b.add(b.mul(b.add(bos, t_last), cH), i_h), cK)
            for kb in range(NKB):
                g_part.append(
                    ld_g.vN(b.add(row, b.add(k_tile[kb], k_lane)), F32, 4)
                )
        elif True:
            g_base = _g_base
            g_part.append(ld_g.scalar(b.add(g_base, t_last), F32))
            for e in range(atom.c_per_lane):
                r_in, _c = atom.lane_to_output(b, lane, e)
                t_abs = b.add(t_b, b.add(b.mul(wid_m, c16), r_in))
                t_safe = b.select(b.cmp_lt(t_abs, T_local), t_abs, b.const_i32(0))
                g_part.append(ld_g.scalar(b.add(g_base, t_safe), F32))
        return w_part, u_part, g_part

    def _pf_pack(w_part, u_part, g_part):
        """Pack prefetch parts in the backend-safe loop-carried phi order."""
        # Carry order. `[w][gate][u]` (vector-group phis before the bfloat u
        # phis) is the layout the AMDGPU backend compiles correctly; the
        # scalar-first `[w][u][gate]` is miscompiled (PERF_PLAN.md §P5,
        # repro_phi_order.py). The switch exists only to reproduce the bug.
        if spec.PREFETCH_VEC_FIRST:
            return w_part + g_part + u_part
        return w_part + u_part + g_part               # miscompiled repro order

    def _pf_issue(i_t_n):
        """Raw reads for chunk ``i_t_n``. Returns a flat list of Values."""
        return _pf_pack(*_pf_issue_parts(i_t_n))

    def _pf_unpack(vals):
        """Structural inverse of :func:`_pf_issue`."""
        it = iter(vals)
        n_w = len(_tile_slots(b, rows=BT, cols=K,
                              block_threads=nthreads, tid=tid))
        w_regs = [next(it) for _ in range(n_w)] if _PF_W else None

        def _rd_gate():
            if not _PF_G:
                return None
            n = NKB if spec.USE_GK else (1 + atom.c_per_lane)
            return [next(it) for _ in range(n)]

        def _rd_u():
            return ([[next(it) for _ in range(atom.c_per_lane)] for _ in range(NRL)]
                    if _PF_U else None)

        if spec.PREFETCH_VEC_FIRST:
            gate = _rd_gate()
            u_vals = _rd_u()
        else:
            u_vals = _rd_u()
            gate = _rd_gate()
        assert next(it, None) is None, "prefetch unpack did not consume every value"
        return w_regs, u_vals, gate

    _pf0 = _pf_issue(b.const_i32(0))

    # ---- initial state -> accumulators ---------------------------------
    inits = []
    for kb in range(NKB):
        for nr in range(NRL):
            if not spec.USE_INITIAL_STATE:
                v = atom.zero_acc(b)
            elif spec.STATE_DTYPE_BF16:
                # load bf16 state, widen each lane to the f32 accumulator
                bf = ld_h0.vN(state_off(kb, nr, i_n), BF16, 4)
                v = _pack_f32x4(b, [b.cast_to_f32(b.vec_extract(bf, j))
                                    for j in range(4)])
            else:
                v = ld_h0.vN(state_off(kb, nr, i_n), F32, 4)
            inits.append((f"h_{kb}_{nr}", v))

    n_acc = len(inits)
    inits = inits + [(f"pf_{i}", v) for i, v in enumerate(_pf0)]

    # ======================= chunk loop =================================
    for_op = b.scf_for_iter(b.const_i32(0), nt_local, b.const_i32(1), inits,
                            iv_name="i_t")
    with for_op as (i_t, carried):
        hacc, pf = carried[:n_acc], carried[n_acc:]
        w_regs, u_pf, gate_pf = _pf_unpack(pf)
        t_base = b.mul(i_t, b.const_i32(BT))

        # -- phase A: accumulators -> sH (bf16); stage w -> sW ------------
        for kb in range(NKB):
            for nr in range(NRL):
                # P1: the four accumulator slots are four consecutive k, and sH
                # is K-major, so this is one packed ds_write, not four scalars.
                k0 = b.add(k_tile[kb], k_lane)
                b.smem_store_vN(sH, [v_row[nr], _swz(b, v_row[nr], k0, ng_k)],
                                _to_bf16_fast(b, hacc[kb * NRL + nr], 4), 4)
        if _PF_W:
            _stage_tile_store(b, smem=sW, regs=w_regs, rows=BT, cols=K,
                              block_threads=nthreads, tid=tid, ng=ng_k)
        else:
            _stage_tile_store(b, smem=sW, rows=BT, cols=K,
                              block_threads=nthreads, tid=tid, ng=ng_k,
                              regs=_stage_tile_load(
                                  b, src_ptr=Wt, rows=BT, cols=K,
                                  row_stride_src=w_stride, src_row_base=t_base,
                                  block_threads=nthreads, tid=tid,
                                  elem_off=_w_base, clamp=T_local, ldr=ld_w))
        nxt_w = []
        k_staged = None
        if (not spec.COMPUTE_OUTPUT
                and (spec.PREFETCH_K_EARLY or spec.PREFETCH_K_INTERLEAVE)):
            k_staged = _init_k_transposed(
                b, spec=spec, tid=tid, nthreads=nthreads,
            )
        if spec.PREFETCH_K_EARLY and not spec.COMPUTE_OUTPUT:
            for r in range(4):
                _load_k_transposed_row(
                    b, src=Kt, staged=k_staged, row=r, spec=spec,
                    t_base=t_base, i_hg=i_hg, T_val=T_local, Hg=Hg,
                    ldr=ld_k, bos=bos,
                )
        b.sync()

        if spec.COMPUTE_OUTPUT:
            # Keep K's VMEM reads after the sW/sH handoff barrier. AMDGPU
            # barriers wait for vmcnt(0), so issuing these before sync would
            # serialize the entire K fetch instead of hiding it under GEMM1.
            # Fused K6 reads k most heavily as [BT, K], so retain that global
            # orientation and store it to LDS after GEMM1.
            k_staged = _stage_tile_load(
                b,
                src_ptr=Kt,
                rows=BT,
                cols=K,
                row_stride_src=Hg * K,
                src_row_base=t_base,
                block_threads=nthreads,
                tid=tid,
                elem_off=_qk_base,
                clamp=T_local,
                ldr=ld_k,
            )

        if spec.STORE_H:
            _drain_h(b, sH=sH, dst=Ho, spec=spec, tid=tid, nthreads=nthreads,
                     chunk=b.add(chunk_base, i_t), i_h=i_h, v_base=v_base, ng=ng_k)

        # -- GEMM1: bv = w @ h^T ------------------------------------------
        bt_row = b.add(b.mul(wid_m, c16), lane_n)
        bv = []
        for nr in range(NRL):
            acc = atom.zero_acc(b)
            for kb in range(NKB):
                for ks in range(spec.k_steps_per_block):
                    k_load_slot = kb * spec.k_steps_per_block + ks
                    if (spec.PREFETCH_K_INTERLEAVE and nr == 0
                            and k_load_slot < 4):
                        _load_k_transposed_row(
                            b, src=Kt, staged=k_staged, row=k_load_slot,
                            spec=spec, t_base=t_base, i_hg=i_hg,
                            T_val=T_local, Hg=Hg, ldr=ld_k, bos=bos,
                        )
                    k0 = b.add(b.const_i32(kb * 64 + ks * spec.mfma_k), k_lane)
                    af = b.smem_load_vN(sW, bt_row, _swz(b, bt_row, k0, ng_k),
                                        dtype=BF16, n=atom.a_per_lane)
                    bf = b.smem_load_vN(sH, v_row[nr], _swz(b, v_row[nr], k0, ng_k),
                                        dtype=BF16, n=atom.b_per_lane)
                    acc = atom.emit(b, af, bf, acc)
            bv.append(acc)

        # -- last valid token of this chunk (shared by both gate paths) ----
        t_bound = T_local if spec.COMPUTE_OUTPUT else T_val
        t_last = b.sub(b.smin(b.add(t_base, b.const_i32(BT)), t_bound),
                       b.const_i32(1))

        # -- C-fragment row coordinates, derived ONCE ----------------------
        # Five things key off this mapping (the row mask, the v_new store
        # guard, the scalar gate, the u gather, the state write). Deriving it
        # once is what keeps them from silently disagreeing.
        frag_bt, frag_t, frag_ok = [], [], []
        for e in range(atom.c_per_lane):
            r_in, _ = atom.lane_to_output(b, lane, e)
            bt = b.add(b.mul(wid_m, c16), r_in)
            t_abs = b.add(t_base, bt)
            frag_bt.append(bt)
            frag_t.append(t_abs)
            frag_ok.append(b.cmp_lt(t_abs, T_local))

        # -- scalar-gate factors (USE_G only) ------------------------------
        # gate[e] = exp(g_last - g[t_e]) applied to v_new; h decays by
        # exp(g_last). g is head-major [H, T].
        if spec.USE_G:
            gb = _g_base
            g_last = gate_pf[0] if _PF_G else ld_g.scalar(b.add(gb, t_last), F32)
            g_gate = []
            g_query = []
            for e in range(atom.c_per_lane):
                ge = (gate_pf[1+e] if _PF_G else ld_g.scalar(
                      b.add(gb, b.select(frag_ok[e], frag_t[e], b.const_i32(0))), F32))
                g_query.append(ge)
                g_gate.append(_exp_gate(b.fsub(g_last, ge)))
            h_decay = _exp_gate(g_last)

        # -- v_new = u - bv, with the tail-chunk row mask (N2) -------------
        # The mask is UNCONDITIONAL. On the scalar-gate path the gate happens to
        # zero out-of-range rows; on the per-channel path nothing else would.
        # Note v_new is reported UNGATED (matching the reference) while the LDS
        # copy that feeds GEMM2 is gated.
        vn = []
        for nr in range(NRL):
            per_e = []
            for e in range(atom.c_per_lane):
                _r, c_in = atom.lane_to_output(b, lane, e)
                ok, t_abs = frag_ok[e], frag_t[e]
                col = b.add(b.add(v_tile[nr], c_in), b.const_i32(0))
                t_safe = b.select(ok, t_abs, b.const_i32(0))
                off = b.add(wu_row_off(_u_base, t_safe, u_stride), col)
                u_f = b.cast_to_f32(u_pf[nr][e] if _PF_U else ld_u.scalar(off, BF16))
                val = b.select(ok, b.fsub(u_f, b.vec_extract(bv[nr], e)),
                               b.const_f32(0.0))
                gated = b.fmul(val, g_gate[e]) if spec.USE_G else val
                per_e.append((val, gated, off, ok))
                if spec.SAVE_NEW_VALUE:
                    with b.scf_if(ok):
                        if spec.OUTPUT_DTYPE_BF16:
                            b.global_store(Vn, off, _to_bf16_fast(b, val), align=2)
                        else:
                            b.global_store(Vn, off, val, align=4)
            vn.append(per_e)

        # -- v_new -> sVN [BV, BT]; k -> sKT [K, BT] ----------------------
        for nr in range(NRL):
            # P1: frag_bt[e] = wid_m*16 + lmb*4 + e -> four consecutive bt, and
            # sVN is BT-major, so the gated values pack into one ds_write.
            packed = _pack_f32x4(b, [vn[nr][e][1] for e in range(4)])
            b.smem_store_vN(sVN, [v_row[nr], _swz(b, v_row[nr], frag_bt[0], ng_t)],
                            _to_bf16_fast(b, packed, 4), 4)
        if spec.COMPUTE_OUTPUT:
            _stage_tile_store(
                b,
                smem=sKT,
                regs=k_staged,
                rows=BT,
                cols=K,
                block_threads=nthreads,
                tid=tid,
                ng=ng_k,
            )
        else:
            _stage_k_transposed(b, src=Kt, smem=sKT, spec=spec, tid=tid,
                                nthreads=nthreads, t_base=t_base, i_hg=i_hg,
                                T_val=T_local, Hg=Hg, ng=ng_t, ldr=ld_k, bos=bos,
                                staged=k_staged)
        if (not spec.COMPUTE_OUTPUT
                and spec.PREFETCH and spec.PREFETCH_W_EARLY):
            nxt_w, _, _ = _pf_issue_parts(
                b.add(i_t, b.const_i32(1)),
                issue_w=True,
                issue_ug=False,
            )
        b.sync()

        if spec.COMPUTE_OUTPUT:
            # q is independent of the state update. Issue it now so GEMM2 hides
            # its HBM latency; the LDS write follows the MFMA chain.
            q_staged = _stage_tile_load(
                b,
                src_ptr=Qt,
                rows=BT,
                cols=K,
                row_stride_src=Hg * K,
                src_row_base=t_base,
                block_threads=nthreads,
                tid=tid,
                elem_off=_qk_base,
                clamp=T_local,
                ldr=ld_q,
            )
            nxt = []
        # P5/P8: K5 issues chunk i+1 before GEMM2. Fused K5+K6 defers it
        # until after the output store, because this latency slot belongs to q.
        elif spec.PREFETCH and spec.PREFETCH_W_EARLY:
            _, nxt_u, nxt_g = _pf_issue_parts(
                b.add(i_t, b.const_i32(1)),
                issue_w=False,
                issue_ug=True,
            )
            nxt = _pf_pack(nxt_w, nxt_u, nxt_g)
        else:
            nxt = _pf_issue(b.add(i_t, b.const_i32(1))) if spec.PREFETCH else []

        # -- state decay, then GEMM2 ---------------------------------------
        # USE_GK: h[v, k] *= exp(gk_last[k]) — per channel. Slot e is
        #         k = tile + lmb*4 + e, so the four factors are one f32x4 load.
        # USE_G : h *= exp(g_last)          — one scalar for the whole state.
        gk_row = b.mul(b.add(b.mul(b.add(bos, t_last), cH), i_h), cK)
        out = []
        for kb in range(NKB):
            if spec.USE_GK:
                gk4 = (gate_pf[kb] if _PF_G else ld_g.vN(
                    b.add(gk_row, b.add(k_tile[kb], k_lane)), F32, 4))
            for nr in range(NRL):
                acc = hacc[kb * NRL + nr]
                dec = atom.zero_acc(b)
                for e in range(4):
                    f = (_exp_gate(b.vec_extract(gk4, e))
                         if spec.USE_GK else h_decay)
                    dec = b.vec_insert(dec, b.fmul(b.vec_extract(acc, e), f), e)
                k_row = b.add(k_tile[kb], lane_n)
                for bs in range(spec.bt_steps):
                    bt0 = b.add(b.const_i32(bs * spec.mfma_k), k_lane)
                    if spec.COMPUTE_OUTPUT:
                        # sKT is [BT, K] on the fused path. GEMM2 needs k^T,
                        # so gather the four contraction rows at fixed K.
                        af = None
                        for e in range(atom.a_per_lane):
                            bt = b.add(bt0, b.const_i32(e))
                            x = b.vec_extract(
                                b.smem_load_vN(
                                    sKT,
                                    bt,
                                    _swz_elem(b, bt, k_row, ng_k),
                                    dtype=BF16,
                                    n=1,
                                ),
                                0,
                            )
                            af = (
                                b.vector_splat(x, atom.a_per_lane)
                                if af is None
                                else b.vec_insert(af, x, e)
                            )
                    else:
                        af = b.smem_load_vN(
                            sKT,
                            k_row,
                            _swz(b, k_row, bt0, ng_t),
                            dtype=BF16,
                            n=atom.a_per_lane,
                        )
                    bf = b.smem_load_vN(sVN, v_row[nr], _swz(b, v_row[nr], bt0, ng_t),
                                        dtype=BF16, n=atom.b_per_lane)
                    dec = atom.emit(b, af, bf, dec)
                out.append(dec)

        # ================================================================
        # Fused K6 output. sH still holds the pre-update state snapshot,
        # while sKT and sVN hold this chunk's k and gated v_new.
        # ================================================================
        if spec.COMPUTE_OUTPUT:
            # q aliases the now-dead w tile.
            _stage_tile_store(
                b,
                smem=sW,
                regs=q_staged,
                rows=BT,
                cols=K,
                block_threads=nthreads,
                tid=tid,
                ng=ng_k,
            )
            b.sync()

            # -- GEMM3: o_inter = q @ h_snapshot^T -----------------------
            inter = []
            for nr in range(NRL):
                acc = atom.zero_acc(b)
                for kb in range(NKB):
                    for ks in range(spec.k_steps_per_block):
                        k0 = b.add(
                            b.const_i32(kb * 64 + ks * spec.mfma_k), k_lane
                        )
                        af = b.smem_load_vN(
                            sW,
                            bt_row,
                            _swz(b, bt_row, k0, ng_k),
                            dtype=BF16,
                            n=atom.a_per_lane,
                        )
                        bf = b.smem_load_vN(
                            sH,
                            v_row[nr],
                            _swz(b, v_row[nr], k0, ng_k),
                            dtype=BF16,
                            n=atom.b_per_lane,
                        )
                        acc = atom.emit(b, af, bf, acc)
                inter.append(acc)

            # -- GEMM4a: A = tril(q @ k^T) -------------------------------
            # A is V-independent. wid_n partitions its key-column tiles
            # across the V-split waves; every wave still owns one query tile.
            a_tiles = []
            for ns in range(spec.bt_steps_local):
                a_tile = b.mul(
                    b.add(b.const_i32(ns * spec.NR_SPLIT), wid_n),
                    c16,
                )
                key_row = b.add(a_tile, lane_n)
                acc = atom.zero_acc(b)
                for kb in range(NKB):
                    for ks in range(spec.k_steps_per_block):
                        k0 = b.add(
                            b.const_i32(kb * 64 + ks * spec.mfma_k), k_lane
                        )
                        af = b.smem_load_vN(
                            sW,
                            bt_row,
                            _swz(b, bt_row, k0, ng_k),
                            dtype=BF16,
                            n=atom.a_per_lane,
                        )
                        bf = b.smem_load_vN(
                            sKT,
                            key_row,
                            _swz(b, key_row, k0, ng_k),
                            dtype=BF16,
                            n=atom.b_per_lane,
                        )
                        acc = atom.emit(b, af, bf, acc)
                a_tiles.append((key_row, acc))

            for key_row, acc in a_tiles:
                key_abs = b.add(t_base, key_row)
                key_ok = b.cmp_lt(key_abs, T_local)
                for e in range(atom.c_per_lane):
                    causal = b.land(
                        b.cmp_ge(frag_bt[e], key_row),
                        b.land(frag_ok[e], key_ok),
                    )
                    a_val = b.select(
                        causal, b.vec_extract(acc, e), b.const_f32(0.0)
                    )
                    query_row = frag_bt[e]
                    b.smem_store_vN(
                        sA,
                        [query_row, _swz_elem(b, query_row, key_row, ng_t)],
                        _to_bf16_fast(b, a_val),
                        1,
                    )
            b.sync()

            # -- GEMM4b: o_intra = A @ v_new_gated ----------------------
            intra = []
            for nr in range(NRL):
                acc = atom.zero_acc(b)
                for bs in range(spec.bt_steps):
                    bt0 = b.add(b.const_i32(bs * spec.mfma_k), k_lane)
                    af = b.smem_load_vN(
                        sA,
                        bt_row,
                        _swz(b, bt_row, bt0, ng_t),
                        dtype=BF16,
                        n=atom.a_per_lane,
                    )
                    bf = b.smem_load_vN(
                        sVN,
                        v_row[nr],
                        _swz(b, v_row[nr], bt0, ng_t),
                        dtype=BF16,
                        n=atom.b_per_lane,
                    )
                    acc = atom.emit(b, af, bf, acc)
                intra.append(acc)

            # USE_G applies distinct per-query factors to the inter and
            # intra terms. USE_GK intentionally applies no K6 gate.
            scale = b.const_f32(spec.SCALE)
            o_base = b.add(
                b.mul(b.add(b.mul(bos, cH), i_h), cV),
                v_base,
            )
            for nr in range(NRL):
                v_abs = b.add(v_base, v_row[nr])
                v_ok = b.cmp_lt(v_abs, cV)
                for e in range(atom.c_per_lane):
                    oi = b.vec_extract(inter[nr], e)
                    oa = b.vec_extract(intra[nr], e)
                    if spec.USE_G:
                        oi = b.fmul(oi, _exp_gate(g_query[e]))
                        oa = b.fmul(
                            oa, _exp_gate(b.fsub(g_query[e], g_last))
                        )
                    value = b.fmul(b.fadd(oi, oa), scale)
                    store_ok = b.land(frag_ok[e], v_ok)
                    off = b.add(
                        b.add(
                            o_base,
                            b.mul(frag_t[e], b.const_i32(H * V)),
                        ),
                        v_row[nr],
                    )
                    with b.scf_if(store_ok):
                        b.global_store(
                            O_ptr,
                            off,
                            _to_bf16_fast(b, value),
                            align=2,
                        )

            # Match the fused schedule: next-chunk reads are issued only after
            # the output store, and the K6 MFMA chain hides their latency.
            nxt = (
                _pf_issue(b.add(i_t, b.const_i32(1)))
                if spec.PREFETCH
                else []
            )
        # No third barrier here. The two hazards that cross the loop back edge
        # are both already ordered:
        #   GEMM2(i) reads sKT/sVN  vs  writes to sKT/sVN in i+1 — a thread can
        #     only reach those writes after sync1(i+1), which every thread can
        #     only reach after finishing GEMM2(i).
        #   GEMM1(i) reads sW/sH    vs  writes to sW/sH in i+1 — separated by
        #     sync2(i), which is before the back edge.
        # Two barriers per chunk, not three.
        b.scf_yield(*out, *nxt)

    # ---- final state ----------------------------------------------------
    if spec.STORE_FINAL_STATE:
        res = for_op.results
        for kb in range(NKB):
            for nr in range(NRL):
                acc = res[kb * NRL + nr]
                if spec.STATE_DTYPE_BF16:
                    b.global_store_vN(Ht, state_off(kb, nr, i_n),
                                      _to_bf16_fast(b, acc, 4), 4)
                else:
                    b.global_store_vN(Ht, state_off(kb, nr, i_n), acc, 4)
    b.ret()

    # The kernelspec is the published interface; this is what keeps it honest.
    # Every arg the launcher binds is named and ordered here, so a param added
    # to the builder without a matching signature entry fails at build time
    # rather than as a misaligned kernarg buffer at launch.
    declared = [p.name for p in b.kernel.params]
    expected = [a["name"] for a in gdn_state_scan_signature(spec)]
    if declared != expected:
        raise AssertionError(
            f"kernel params {declared} do not match "
            f"gdn_state_scan_signature {expected}"
        )
    return b.kernel
