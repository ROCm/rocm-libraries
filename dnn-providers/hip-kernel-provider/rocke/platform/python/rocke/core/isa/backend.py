# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ISA backends: the per-gfx code that owns architecture-specific LLVM details.

The generic ``_Lowerer`` in ``core/lower_llvm.py`` keeps the target-neutral walk
(CFG, PHIs, scf.for, LLVM types, generic memory). Everything that is keyed by
the gfx target — datalayout / triple, ``s_waitcnt`` encoding, (later) MFMA call
emission, async copy, and LDS transpose reads — goes behind an
:class:`ISABackend` selected from the :class:`~rocke.core.arch.ArchTarget`.

Status (first milestone): for the CDNA targets wired up today (gfx942 / gfx950)
the datalayout, triple, and the ``s_waitcnt`` *layout* are identical — this is
hardware-verified for the base f16 GEMM path on both MI300X and MI350X. The
backend therefore selects the same shared constants for both, while exposing
distinct classes (``Gfx950Backend`` vs ``Gfx9MfmaBackend``). The
``arch.vmcnt_bits`` fact is load-bearing: :meth:`ISABackend.encode_waitcnt`
asserts it against the encoder's representable field width
(:data:`ISABackend._VMCNT_ENCODER_BITS`), so genuinely divergent codegen (e.g.
gfx908, or a future target whose VMCNT field is wider than the current 6-bit
encoders emit) fails loudly here instead of silently truncating a partial wait
into a full VMEM drain. See
``dsl_docs/architecture/multi_arch_data_layout.md`` ("ISA Backend").

This module imports only from ``core/arch`` at module load; the shared LLVM
constants are pulled from ``core/lower_llvm`` lazily inside methods to avoid an
import cycle (``lower_llvm`` imports :func:`backend_for` at module top).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Union

from ..arch import ArchTarget


class ISABackend:
    """Base ISA backend. Holds the :class:`ArchTarget` and exposes the
    gfx-keyed LLVM details the lowerer needs."""

    def __init__(self, arch: ArchTarget) -> None:
        self.arch = arch

    # --- module preamble -------------------------------------------------
    @property
    def triple(self) -> str:
        from ..lower_llvm import _TRIPLE

        return _TRIPLE

    def datalayout(self, llvm_flavor: str) -> str:
        """Module ``target datalayout`` for this target under ``llvm_flavor``.

        The string is LLVM-version-keyed, not gfx-keyed (every wired arch
        shares one datalayout; only the ``p8`` field drifts between LLVM
        20 and 22), so the backend defers to
        :func:`~rocke.core.lower_llvm._datalayout_for_flavor`.
        """
        from ..lower_llvm import _datalayout_for_flavor

        return _datalayout_for_flavor(llvm_flavor)

    def module_preamble(self, llvm_flavor: str) -> str:
        """The two leading IR lines: ``target datalayout`` + ``target triple``."""
        return (
            f'target datalayout = "{self.datalayout(llvm_flavor)}"\n'
            f'target triple = "{self.triple}"'
        )

    # --- buffer resource descriptor --------------------------------------
    @property
    def buffer_rsrc_word3(self) -> int:
        """DWORD3 of the buffer resource descriptor fed to
        ``llvm.amdgcn.make.buffer.rsrc`` as its ``flags`` operand.

        The format/OOB-select encoding in word3 is **ISA-specific**: the
        CDNA (gfx9) layout is *not* binary-compatible with the RDNA
        (gfx10/11) layout. The CDNA value ``0x00027000`` ("32-bit-uint,
        bounds-checked"; matches CK Tile's hardcoded gfx9 constant) places
        the resource in an out-of-bounds-everything state on gfx11, so a
        ``raw.ptr.buffer.load/store`` against it silently returns 0 / drops
        the write. RDNA backends override this with the gfx10/11 word3."""
        return 0x00027000

    # --- s_waitcnt -------------------------------------------------------
    #
    # Widest VMCNT field this backend's encoder can physically represent. The
    # gfx9/10 split layout (VMCNT across ``[3:0]`` and ``[15:14]``) emits a
    # 6-bit value; the RDNA contiguous layout (``[15:10]``) is also 6-bit. This
    # is the *encoder capability*, distinct from any one arch's declared
    # ``vmcnt_bits`` -- e.g. gfx942 declares a 4-bit field but is lowered
    # through this 6-bit-capable encoder (HW-verified on MI300X). The guard in
    # :meth:`encode_waitcnt` consults ``arch.vmcnt_bits`` so a future target
    # whose field is *wider* than the encoder can emit fails loudly instead of
    # silently truncating a wait into a full VMEM drain.
    _VMCNT_ENCODER_BITS: int = 6

    def encode_waitcnt(self, vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
        """Encode an ``s_waitcnt`` immediate.

        The gfx9/gfx10 split layout (VMCNT across ``[3:0]`` and ``[15:14]``) is
        shared across the CDNA targets we lower today. ``arch.vmcnt_bits`` is
        asserted against :data:`_VMCNT_ENCODER_BITS` (the field width this
        encoder produces) so divergence -- a target declaring a wider VMCNT than
        the encoder emits -- is caught here rather than miscompiled.
        """
        self._check_vmcnt_width()
        from ..lower_llvm import _encode_waitcnt_gfx9_10

        return _encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt)

    def _check_vmcnt_width(self) -> None:
        if self.arch.vmcnt_bits > self._VMCNT_ENCODER_BITS:
            raise NotImplementedError(
                f"{self.arch.gfx} declares a {self.arch.vmcnt_bits}-bit VMCNT "
                f"field but {type(self).__name__}.encode_waitcnt emits only "
                f"{self._VMCNT_ENCODER_BITS} bits; a wider s_waitcnt encoder is "
                f"needed before this target can lower correctly"
            )

    # --- matrix ops ------------------------------------------------------
    def emit_mma(self, lowerer, op) -> None:
        """Lower a target-neutral ``tile.mma`` op.

        The ``op_id`` attribute selects the concrete atom. The base (CDNA)
        implementation rebuilds the legacy ISA-named op (``tile.<op_id>``) and
        dispatches it through the lowerer's existing per-op handler, so the
        emitted text is **byte-identical** to the historical MFMA path. RDNA
        backends override this to route through :meth:`emit_wmma`.

        A WMMA ``op_id`` reaching a CDNA backend resolves to the WMMA handler,
        which calls :meth:`emit_wmma` and correctly raises ``NotImplementedError``
        (WMMA is an RDNA-only instruction).
        """
        from ..ir import Op

        op_id = op.attrs["op_id"]
        legacy = Op(
            name=f"tile.{op_id}",
            operands=list(op.operands),
            results=list(op.results),
            attrs={k: v for k, v in op.attrs.items() if k != "op_id"},
            loc=op.loc,
        )
        lowerer.lower_op(legacy)

    @property
    def emits_legacy_s_waitcnt(self) -> bool:
        """Whether ``llvm.amdgcn.s.waitcnt`` is selectable on this target.

        gfx9 / gfx10 / gfx11 support the monolithic ``s_waitcnt`` intrinsic, so
        the lowerer emits an explicit ``s_waitcnt`` before LDS barriers. gfx1250
        (gfx1250) removed it in favour of split wait counters and overrides this
        to ``False`` (see :class:`Gfx1250Backend`)."""
        return True

    @property
    def has_async_lds_counter(self) -> bool:
        """Whether the target has the gfx1250 dedicated async-DMA counter
        (``s_wait_asynccnt`` + ``global_load_async_to_lds``). Only gfx1250
        (gfx1250) overrides this to ``True``; elsewhere the async-to-LDS
        instructions do not exist, so ``s_wait_asynccnt`` lowers to nothing."""
        return False

    def ds_tr16_b128_spec(self, elem_type: str = "f16"):
        """``(need_key, intrinsic, llvm_ret_ty)`` for the wide 16-bit
        transpose-LDS read (logical element ``elem_type``) used to feed the
        MFMA/WMMA B operand.

        Default is the gfx950 ``ds_read_b128_tr_b16`` whose intrinsic returns a
        type-agnostic ``<8 x i16>``; the lowerer bitcasts those raw 16-bit lanes
        to the requested ``half`` / ``bfloat`` element, so a single spec serves
        both f16 and bf16. gfx1250 (gfx1250) overrides with an
        element-type-specific ``ds_load_tr16_b128`` intrinsic (its wave32
        per-lane data distribution also differs — kernels must use the layout
        appropriate to the target). Fail fast for any unsupported element."""
        if elem_type not in ("f16", "fp16", "bf16"):
            raise NotImplementedError(
                f"ds_read_tr16_b128 on {self.arch.gfx} supports f16/bf16 only, "
                f"got elem_type={elem_type!r}"
            )
        return ("ds.read.tr16.b128", "llvm.amdgcn.ds.read.tr16.b128", "<8 x i16>")

    def emit_lds_barrier_drain(self, lowerer, *, drain_vmem: bool) -> None:
        """Emit the memory wait that must precede an LDS workgroup barrier.

        An ``s_barrier`` only synchronises waves; it does not drain outstanding
        LDS (and, when ``drain_vmem``, VMEM->LDS) traffic. Without this wait a
        post-barrier reader can observe stale LDS — and on a single-wave
        workgroup (where the barrier is a NOP) the same-wave LDS write->read is
        unordered entirely. gfx9/10/11 use the monolithic ``s_waitcnt``
        (lgkmcnt[, vmcnt]); gfx1250 (gfx1250) overrides this with split counters."""
        mask = self.encode_waitcnt(vmcnt=0 if drain_vmem else -1, expcnt=-1, lgkmcnt=0)
        lowerer._need("s.waitcnt")
        lowerer._current().emit(f"  call void @llvm.amdgcn.s.waitcnt(i32 {mask})")

    def emit_wmma(self, lowerer, op) -> None:
        """Emit an RDNA WMMA matrix op. Only RDNA backends implement this;
        CDNA/MFMA targets reject it (MFMA ops lower inline in ``_Lowerer``)."""
        raise NotImplementedError(
            f"WMMA op {op.name!r} not available on {self.arch.gfx} "
            f"(WMMA is an RDNA/gfx11 instruction; this is a CDNA/MFMA target)"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(gfx={self.arch.gfx})"


class Gfx950Backend(ISABackend):
    """CDNA4 / MI350-MI355. 6-bit VMCNT, fp8/bf8/fp4 MFMA, ``ds_read_*_tr_*``,
    160 KB LDS. This is the historical default; its output is the byte-identical
    baseline."""


class Gfx9MfmaBackend(ISABackend):
    """CDNA gfx9 MFMA family (gfx908 / gfx90a / gfx942). Shares the gfx9/10
    waitcnt layout and (for the verified base GEMM path) the same datalayout /
    triple as gfx950. Per-arch divergence (4-bit VMCNT fields, no
    transpose-LDS) keys off ``self.arch``."""


# WMMA op -> (decl key in _INTRINSIC_DECLS, fully-mangled intrinsic, SSA operand
# element type, call-site operand element type). When the two element types
# differ, emit_wmma bitcasts each <16 x ssa_elt> operand to <16 x call_elt>
# before the call. Hardware-verified on gfx1151 (ctr-halo, ROCm 7.0.2 clang 20):
#   f16:  llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half>,<16 x half>,<8 x float>)
#   bf16: llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16>,<16 x i16>,<8 x float>)
# bf16 operands arrive as <16 x bfloat> and are bitcast to <16 x i16>.
_RDNA_WMMA = {
    "tile.wmma_f32_16x16x16_f16": (
        "wmma.f32.16x16x16.f16",
        "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16",
        "half",
        "half",
    ),
    "tile.wmma_f32_16x16x16_bf16": (
        "wmma.f32.16x16x16.bf16",
        "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16",
        "bfloat",
        "i16",
    ),
}


# Integer WMMA op -> (decl key, fully-mangled intrinsic, A/B operand vector
# width, accumulator/result vector width). Integer WMMA differs from the float
# path in two ways: (1) operands/accumulator are i32 vectors (A/B packed, C/D
# the i32 accumulator), and (2) the intrinsic signature carries i1 signedness
# flags before each matrix operand and a trailing i1 clamp. Operands arrive in
# SSA already as <N x i32> (the kernel packs int8/int4 into i32), so no bitcast
# is needed. Our quantized data is signed and within i32 range, so the flags are
# emitted as (unsignedA=0, unsignedB=0, clamp=0). Verified on gfx1151/gfx11-generic
# (ROCm 7.2.0): lowers to v_wmma_i32_16x16x16_iu8.
#   iu8:  A/B = <4 x i32> (16 int8 packed 4-per-i32), C/D = <8 x i32>
#   iu4:  A/B = <2 x i32> (16 int4 packed 8-per-i32), C/D = <8 x i32>
_RDNA_WMMA_INT = {
    "tile.wmma_i32_16x16x16_iu8": (
        "wmma.i32.16x16x16.iu8",
        "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32",
        4,
        8,
    ),
    "tile.wmma_i32_16x16x16_iu4": (
        "wmma.i32.16x16x16.iu4",
        "llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32",
        2,
        8,
    ),
}


# RDNA4 (gfx12) WMMA. Same instruction family as RDNA3/3.5 but the operand
# fragments dropped the cross-half duplication: A/B are <8 x ...> per lane (not
# <16 x ...>), so the intrinsic mangling is ``v8f16`` / ``v8i16``. The op_id is
# distinct (``wmma_gfx12_*``) so the fragment/lane-map tables stay flat-keyed.
_RDNA_GFX12_WMMA = {
    "tile.wmma_gfx12_f32_16x16x16_f16": (
        "wmma.gfx12.f32.16x16x16.f16",
        "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16",
        "half",
        "half",
    ),
    "tile.wmma_gfx12_f32_16x16x16_bf16": (
        "wmma.gfx12.f32.16x16x16.bf16",
        "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16",
        "bfloat",
        "i16",
    ),
}


# gfx1250 (gfx1250) WMMA. CDNA multi-chip on the GFX12 programming model. The
# primary fp16/bf16 atom is K=32 (not gfx1201's K=16): A/B are <16 x ...> per
# lane and the intrinsic takes the gfx1250 8-operand form
# (i1 negA, A, i1 negB, B, i16 fmt, C, i1 reuseA, i1 reuseB). Unlike gfx11/gfx12,
# bf16 lowers to <16 x bfloat> directly (v16bf16), so there is no i16 bitcast.
# op.name -> (decl key, fully-mangled intrinsic, SSA/call operand element type).
_GFX1250_WMMA = {
    "tile.wmma_gfx1250_f32_16x16x32_f16": (
        "wmma.gfx1250.f32.16x16x32.f16",
        "llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16",
        "half",
    ),
    "tile.wmma_gfx1250_f32_16x16x32_bf16": (
        "wmma.gfx1250.f32.16x16x32.bf16",
        "llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16",
        "bfloat",
    ),
}


# gfx1250 (gfx1250) FP8/BF8 WMMA, K=64. A/B are 32 low-bit bytes per lane carried
# as <8 x i32> at the intrinsic boundary; the 6-operand form is
# (A, B, i16 fmt, C, i1, i1) -- no leading neg flags, unlike the K=32 f16/bf16
# form. op.name -> (decl key, fully-mangled intrinsic).
_GFX1250_WMMA_FP8 = {
    "tile.wmma_gfx1250_f32_16x16x64_fp8_fp8": (
        "wmma.gfx1250.f32.16x16x64.fp8.fp8",
        "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32",
    ),
    "tile.wmma_gfx1250_f32_16x16x64_fp8_bf8": (
        "wmma.gfx1250.f32.16x16x64.fp8.bf8",
        "llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32",
    ),
    "tile.wmma_gfx1250_f32_16x16x64_bf8_fp8": (
        "wmma.gfx1250.f32.16x16x64.bf8.fp8",
        "llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32",
    ),
    "tile.wmma_gfx1250_f32_16x16x64_bf8_bf8": (
        "wmma.gfx1250.f32.16x16x64.bf8.bf8",
        "llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32",
    ),
}


class Gfx11RdnaBackend(ISABackend):
    """RDNA3 / RDNA3.5 (gfx11, e.g. gfx1151 Strix Halo). **wave32**, **WMMA**
    (no MFMA), and a distinct ``s_waitcnt`` layout from gfx9/10. Datalayout +
    triple are identical to CDNA on the ROCm releases we target (clang-verified
    on gfx1151), so those are inherited unchanged."""

    @property
    def buffer_rsrc_word3(self) -> int:
        """RDNA (gfx10/11/12) buffer resource DWORD3.

        ``0x31014000`` is the gfx10+ "raw" SRD word3 used by ROCm / CK Tile
        for ``gfx103`` / ``gfx11`` / ``gfx12`` (the format + OOB-select field
        encoding moved relative to gfx9). The CDNA value ``0x00027000`` makes
        every ``raw.ptr.buffer.load/store`` read 0 / drop on gfx11; this is
        the value that makes bounds-checked raw buffer access work on the
        gfx1151 Strix Halo box."""
        return 0x31014000

    def emit_mma(self, lowerer, op) -> None:
        """Lower ``tile.mma`` to an RDNA WMMA call.

        Rebuilds the legacy ``tile.<op_id>`` op so :meth:`emit_wmma`'s
        name-keyed lookup keeps working, then emits the WMMA call. This is the
        RDNA half of the neutral-MMA contract; the CDNA base emits MFMA.
        """
        from ..ir import Op

        op_id = op.attrs["op_id"]
        legacy = Op(
            name=f"tile.{op_id}",
            operands=list(op.operands),
            results=list(op.results),
            attrs={k: v for k, v in op.attrs.items() if k != "op_id"},
            loc=op.loc,
        )
        self.emit_wmma(lowerer, legacy)

    def emit_wmma(self, lowerer, op) -> None:
        int_spec = _RDNA_WMMA_INT.get(op.name)
        if int_spec is not None:
            self._emit_wmma_int(lowerer, op, int_spec)
            return
        spec = _RDNA_WMMA.get(op.name)
        if spec is None:
            raise NotImplementedError(
                f"WMMA op {op.name!r} not yet wired for {self.arch.gfx}; "
                f"known: {sorted(_RDNA_WMMA) + sorted(_RDNA_WMMA_INT)}"
            )
        decl_key, intrinsic, ssa_elt, call_elt = spec
        a, b, c = op.operands
        lowerer._need(decl_key)
        a_arg = lowerer._operand(a)
        b_arg = lowerer._operand(b)
        if call_elt != ssa_elt:
            # bf16 (and any future type whose SSA element differs from the
            # intrinsic's operand element): bitcast <16 x ssa_elt> -> <16 x call_elt>.
            a_cast = lowerer._fresh("wmma_a")
            b_cast = lowerer._fresh("wmma_b")
            lowerer._current().emit(
                f"  {a_cast} = bitcast <16 x {ssa_elt}> {a_arg} to <16 x {call_elt}>"
            )
            lowerer._current().emit(
                f"  {b_cast} = bitcast <16 x {ssa_elt}> {b_arg} to <16 x {call_elt}>"
            )
            a_arg, b_arg = a_cast, b_cast
        lowerer._current().emit(
            f"  {op.result.name} = call <8 x float> @{intrinsic}("
            f"<16 x {call_elt}> {a_arg}, "
            f"<16 x {call_elt}> {b_arg}, "
            f"<8 x float> {lowerer._operand(c)})"
        )

    def _emit_wmma_int(self, lowerer, op, spec) -> None:
        """Emit an integer WMMA (iu8/iu4) call.

        The integer intrinsic signature is
        ``(i1 signedA, <N x i32> A, i1 signedB, <N x i32> B, <8 x i32> C, i1 clamp)``
        with an ``<8 x i32>`` result. The leading i1 per operand selects the
        operand's *signedness*: ``1`` = signed, ``0`` = unsigned. This was
        verified empirically on gfx11-generic (iu8 GEMM probe): passing ``0``
        made the unit compute the **unsigned** dot product (all-positive
        results matching ``A.view(uint8) @ B.view(uint8).T``). Our quantized
        data is signed, so both flags are ``1``. Operands arrive as
        ``<N x i32>`` in SSA (int8/int4 packed into i32), so no bitcast is
        needed; values stay within i32 range -> ``clamp = 0`` (exact wrap).
        """
        decl_key, intrinsic, op_vec, acc_vec = spec
        a, b, c = op.operands
        lowerer._need(decl_key)
        a_arg = lowerer._operand(a)
        b_arg = lowerer._operand(b)
        c_arg = lowerer._operand(c)
        lowerer._current().emit(
            f"  {op.result.name} = call <{acc_vec} x i32> @{intrinsic}("
            f"i1 1, <{op_vec} x i32> {a_arg}, "
            f"i1 1, <{op_vec} x i32> {b_arg}, "
            f"<{acc_vec} x i32> {c_arg}, i1 0)"
        )

    def encode_waitcnt(self, vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
        # RDNA gfx11 uses a different s_waitcnt field layout than the gfx9/10
        # split the base encodes: contiguous expcnt[2:0] / lgkmcnt[9:4] /
        # vmcnt[15:10] (no split VMCNT, 6-bit LGKMCNT). The layout was read
        # off the ROCm 7.0.2 AMDGPU assembler on a gfx1151 node; see
        # _encode_waitcnt_gfx11 for the empirical encodings. Also a 6-bit VMCNT
        # field, so the base _VMCNT_ENCODER_BITS guard applies unchanged.
        self._check_vmcnt_width()
        from ..lower_llvm import _encode_waitcnt_gfx11

        return _encode_waitcnt_gfx11(vmcnt, expcnt, lgkmcnt)


class Gfx12RdnaBackend(Gfx11RdnaBackend):
    """RDNA4 (gfx12, e.g. gfx1201 Navi 48). **wave32**, **WMMA** with the gfx12
    fragment ABI: A/B operands are ``<8 x ...>`` per lane (the RDNA3/3.5
    cross-half duplication was removed) and the accumulator is column-distributed.
    Datalayout / triple / buffer SRD word3 / s_waitcnt layout are inherited from
    the RDNA3 backend (gfx11/gfx12 share the RDNA buffer word3 and contiguous
    waitcnt layout). Only :meth:`emit_wmma` diverges (8-wide operands, gfx12
    intrinsic mangling)."""

    def emit_wmma(self, lowerer, op) -> None:
        spec = _RDNA_GFX12_WMMA.get(op.name)
        if spec is None:
            raise NotImplementedError(
                f"WMMA op {op.name!r} not yet wired for {self.arch.gfx}; "
                f"known: {sorted(_RDNA_GFX12_WMMA)}"
            )
        decl_key, intrinsic, ssa_elt, call_elt = spec
        a, b, c = op.operands
        lowerer._need(decl_key)
        a_arg = lowerer._operand(a)
        b_arg = lowerer._operand(b)
        if call_elt != ssa_elt:
            # bf16: bitcast <8 x bfloat> -> <8 x i16> before the call.
            a_cast = lowerer._fresh("wmma_a")
            b_cast = lowerer._fresh("wmma_b")
            lowerer._current().emit(
                f"  {a_cast} = bitcast <8 x {ssa_elt}> {a_arg} to <8 x {call_elt}>"
            )
            lowerer._current().emit(
                f"  {b_cast} = bitcast <8 x {ssa_elt}> {b_arg} to <8 x {call_elt}>"
            )
            a_arg, b_arg = a_cast, b_cast
        lowerer._current().emit(
            f"  {op.result.name} = call <8 x float> @{intrinsic}("
            f"<8 x {call_elt}> {a_arg}, "
            f"<8 x {call_elt}> {b_arg}, "
            f"<8 x float> {lowerer._operand(c)})"
        )


class Gfx1250Backend(Gfx12RdnaBackend):
    """gfx1250-class (gfx1250) CDNA device on the GFX12 programming model.
    **wave32**, **WMMA** (no MFMA), with the K=32 fp16/bf16 atom. Datalayout /
    triple are byte-identical to gfx950/gfx1201 (verified on ROCm 7.13), so they
    are inherited. Only :meth:`emit_wmma` diverges: the gfx1250 16x16x32 form has
    16-wide operands and the 8-operand intrinsic signature, and bf16 lowers to
    ``<16 x bfloat>`` directly (no i16 bitcast).

    NOTE (gfx1250 model): the inherited buffer SRD word3 and gfx11
    ``s_waitcnt`` layout are placeholders adequate for flat-global WMMA GEMM
    bring-up; the gfx1250 57-bit SRD and split wait-counter model are deferred
    (see gfx1250_universal_attention_plan.md, Leg A Phase 2/3)."""

    @property
    def emits_legacy_s_waitcnt(self) -> bool:
        # gfx1250 (gfx1250) replaced the monolithic ``s_waitcnt`` with split wait
        # counters (``s_wait_dscnt`` / ``s_wait_loadcnt`` / ...). The
        # ``llvm.amdgcn.s.waitcnt`` intrinsic is NOT selectable on gfx1250.
        return False

    @property
    def has_async_lds_counter(self) -> bool:
        # gfx1250 (gfx1250) has ``global_load_async_to_lds`` tracked by a
        # dedicated ``ASYNCcnt`` drained via ``s_wait_asynccnt``.
        return True

    def ds_tr16_b128_spec(self, elem_type: str = "f16"):
        # gfx1250 (gfx1250) wave32 ``ds_load_tr16_b128`` is overloaded on the
        # result element type. Select the intrinsic matching the op's element
        # type so an f16 read does not reinterpret a bf16 payload (and vice
        # versa); fail fast for anything the opcode cannot carry.
        if elem_type == "bf16":
            return (
                "ds.load.tr16.b128.v8bf16",
                "llvm.amdgcn.ds.load.tr16.b128.v8bf16",
                "<8 x bfloat>",
            )
        if elem_type in ("f16", "fp16"):
            return (
                "ds.load.tr16.b128.v8f16",
                "llvm.amdgcn.ds.load.tr16.b128.v8f16",
                "<8 x half>",
            )
        raise NotImplementedError(
            f"ds_load_tr16_b128 on {self.arch.gfx} supports f16/bf16 only, "
            f"got elem_type={elem_type!r}"
        )

    def emit_lds_barrier_drain(self, lowerer, *, drain_vmem: bool) -> None:
        # gfx1250 split counters. The raw ``llvm.amdgcn.s.barrier`` does NOT get
        # an auto-inserted pre-barrier ``s_wait_dscnt``, so the LDS write->read
        # (e.g. P-staging -> PV read) would race and read stale LDS -> NaN. Emit
        # the explicit drain: dscnt for LDS, plus loadcnt for the VMEM->LDS chain.
        # clang fuses these into ``s_wait_loadcnt_dscnt 0`` before the barrier.
        if drain_vmem:
            lowerer._need("s.wait.loadcnt")
            lowerer._current().emit("  call void @llvm.amdgcn.s.wait.loadcnt(i16 0)")
        lowerer._need("s.wait.dscnt")
        lowerer._current().emit("  call void @llvm.amdgcn.s.wait.dscnt(i16 0)")

    def emit_wmma(self, lowerer, op) -> None:
        fp8_spec = _GFX1250_WMMA_FP8.get(op.name)
        if fp8_spec is not None:
            self._emit_wmma_fp8(lowerer, op, fp8_spec)
            return
        spec = _GFX1250_WMMA.get(op.name)
        if spec is None:
            raise NotImplementedError(
                f"WMMA op {op.name!r} not yet wired for {self.arch.gfx}; "
                f"known: {sorted(_GFX1250_WMMA) + sorted(_GFX1250_WMMA_FP8)}"
            )
        decl_key, intrinsic, elt = spec
        a, b, c = op.operands
        lowerer._need(decl_key)
        a_arg = lowerer._operand(a)
        b_arg = lowerer._operand(b)
        # gfx1250 8-operand form: (i1 negA, A, i1 negB, B, i16 fmt, C, i1, i1).
        # bf16 operands are <16 x bfloat> directly (no i16 bitcast).
        lowerer._current().emit(
            f"  {op.result.name} = call <8 x float> @{intrinsic}("
            f"i1 false, <16 x {elt}> {a_arg}, "
            f"i1 false, <16 x {elt}> {b_arg}, "
            f"i16 0, <8 x float> {lowerer._operand(c)}, "
            f"i1 false, i1 false)"
        )

    def _emit_wmma_fp8(self, lowerer, op, spec) -> None:
        """Emit a gfx1250 K=64 FP8/BF8 WMMA call.

        The A/B fragments arrive in SSA as ``<8 x i32>`` (32 low-bit bytes per
        lane). The 6-operand form is ``(A, B, i16 fmt, C, i1, i1)`` with the
        format / reuse immediates pinned to 0 (plain unscaled MMA).
        """
        decl_key, intrinsic = spec
        a, b, c = op.operands
        lowerer._need(decl_key)
        lowerer._current().emit(
            f"  {op.result.name} = call <8 x float> @{intrinsic}("
            f"<8 x i32> {lowerer._operand(a)}, "
            f"<8 x i32> {lowerer._operand(b)}, "
            f"i16 0, <8 x float> {lowerer._operand(c)}, "
            f"i1 false, i1 false)"
        )


# gfx -> backend class. Adding a CDNA gfx is one row here plus, when its codegen
# actually diverges, a new subclass.
#
# Some rows are forward-declarations: the backend class is wired here before the
# matching ``core/arch/data/arch_specs.json`` row exists (gfx908 / gfx90a reuse
# ``Gfx9MfmaBackend`` and are kept for upcoming enablement). ``backend_for``
# resolves the :class:`ArchTarget` first, so such a row reports a clean
# "metadata not yet present" error from this layer rather than a raw arch-layer
# ``KeyError``. ``wired_arches()`` returns only the rows that have arch metadata
# and can actually build a backend today.
BACKEND_REGISTRY: Dict[str, Callable[[ArchTarget], ISABackend]] = {
    "gfx908": Gfx9MfmaBackend,
    "gfx90a": Gfx9MfmaBackend,
    "gfx942": Gfx9MfmaBackend,
    "gfx950": Gfx950Backend,
    "gfx1151": Gfx11RdnaBackend,
    "gfx1201": Gfx12RdnaBackend,
    "gfx1250": Gfx1250Backend,
    "gfx11-generic": Gfx11RdnaBackend,
}


def wired_arches() -> Tuple[str, ...]:
    """gfx targets that have BOTH a backend row and arch metadata today.

    A backend row may be a forward-declaration (registered before its
    ``arch_specs.json`` row lands); those are excluded here because
    :func:`backend_for` cannot construct them yet.
    """
    from ..arch import known_arches

    known = set(known_arches())
    return tuple(sorted(g for g in BACKEND_REGISTRY if g in known))


def backend_for(arch: Union[str, ArchTarget]) -> ISABackend:
    """Resolve a gfx string or :class:`ArchTarget` to its ISA backend."""
    if isinstance(arch, ArchTarget):
        target = arch
    else:
        cls = BACKEND_REGISTRY.get(arch)
        if cls is None:
            raise KeyError(
                f"no ISA backend registered for {arch!r}; "
                f"known: {sorted(BACKEND_REGISTRY)}"
            )
        try:
            target = ArchTarget.from_gfx(arch)
        except KeyError as exc:
            # Backend row exists but the arch metadata does not: a
            # forward-declared target (e.g. gfx908 / gfx90a) that is not yet
            # buildable. Report it as such instead of leaking the arch-layer
            # "unknown gfx target" message.
            raise KeyError(
                f"ISA backend for {arch!r} is forward-declared but has no "
                f"arch_specs.json metadata yet; buildable now: "
                f"{list(wired_arches())}"
            ) from exc
    cls = BACKEND_REGISTRY.get(target.gfx)
    if cls is None:
        raise KeyError(
            f"no ISA backend registered for {target.gfx!r}; "
            f"known: {sorted(BACKEND_REGISTRY)}"
        )
    return cls(target)
