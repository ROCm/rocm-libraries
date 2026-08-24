# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Spec for the gated-delta-rule chunked state scan (the "K5" stage).

One frozen dataclass carrying the compile-time configuration, with every
downstream quantity exposed as a derived property rather than a stored field —
so an illegal combination is impossible to construct rather than merely
discouraged.

**MFMA_K is derived from the arch catalog, not hardcoded.** On CDNA3 the widest
16x16 bf16 atom is K=16; on CDNA4 it is K=32. Every K-loop bound in the kernel
keys off :attr:`GdnStateScanSpec.mfma_k`, so retargeting changes one lookup
instead of a dozen literals.

The compute, for one chunk of ``BT`` tokens, with state ``h`` of shape
``[V, K]``::

    h_snapshot = h                                  # drained for the K6 stage
    v_new      = u - w @ h^T                        # GEMM1
    v_new     *= decay                              # scalar-gate path only
    h          = Diag(gate) @ h + k^T @ v_new       # GEMM2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from rocke.core.arch.target import ArchTarget


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
    """Compile-time configuration for one build of the state-scan kernel."""

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

    # -- fused K6 stage (out of scope for the first study milestone) --
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


def is_valid_spec(spec: GdnStateScanSpec, arch: Optional[str] = None) -> Tuple[bool, str]:
    """Non-throwing validity probe, for dispatcher-style enumeration."""
    try:
        if arch is not None and arch != spec.arch:
            from dataclasses import replace
            spec = replace(spec, arch=arch)
        else:
            GdnStateScanSpec(**{f: getattr(spec, f) for f in spec.__dataclass_fields__})
        return True, ""
    except ValueError as e:
        return False, str(e)
