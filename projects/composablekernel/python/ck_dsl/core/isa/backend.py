# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ISA backends: the per-gfx code that owns architecture-specific LLVM details.

The generic ``_Lowerer`` in ``core/lower_llvm.py`` keeps the target-neutral walk
(CFG, PHIs, scf.for, LLVM types, generic memory). Everything that is keyed by
the gfx target — datalayout / triple, ``s_waitcnt`` encoding, (later) MFMA call
emission, async copy, and LDS transpose reads — goes behind an
:class:`ISABackend` selected from the :class:`~ck_dsl.core.arch.ArchTarget`.

Status (first milestone): for the CDNA targets wired up today (gfx942 / gfx950)
the datalayout, triple, and the ``s_waitcnt`` *layout* are identical — this is
hardware-verified for the base f16 GEMM path on both MI300X and MI350X. The
backend therefore selects the same shared constants for both, while exposing
distinct classes (``Gfx950Backend`` vs ``Gfx9MfmaBackend``) and the
``arch.vmcnt_bits`` fact so genuinely divergent codegen (e.g. gfx908, or a
gfx942 ``compv4`` partial-waitcnt that needs the 4-bit VMCNT field) plugs in
here without touching ``_Lowerer``. See
``dsl_docs/architecture/multi_arch_data_layout.md`` ("ISA Backend").

This module imports only from ``core/arch`` at module load; the shared LLVM
constants are pulled from ``core/lower_llvm`` lazily inside methods to avoid an
import cycle (``lower_llvm`` imports :func:`backend_for` at module top).
"""

from __future__ import annotations

from typing import Callable, Dict, Union

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

    @property
    def datalayout(self) -> str:
        from ..lower_llvm import _DATALAYOUT

        return _DATALAYOUT

    def module_preamble(self) -> str:
        """The two leading IR lines: ``target datalayout`` + ``target triple``."""
        return (
            f'target datalayout = "{self.datalayout}"\ntarget triple = "{self.triple}"'
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
    def encode_waitcnt(self, vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
        """Encode an ``s_waitcnt`` immediate. The gfx9/gfx10 split layout
        (VMCNT across ``[3:0]`` and ``[15:14]``) is shared across the CDNA
        targets we lower today; ``arch.vmcnt_bits`` records the field width
        for future divergence."""
        from ..lower_llvm import _encode_waitcnt_gfx9_10

        return _encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt)

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
_RDNA_WMMA_INT = {
    "tile.wmma_i32_16x16x16_iu8": (
        "wmma.i32.16x16x16.iu8",
        "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32",
        4,
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
        # _encode_waitcnt_gfx11 for the empirical encodings.
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


# gfx -> backend class. Adding a CDNA gfx is one row here plus, when its codegen
# actually diverges, a new subclass.
BACKEND_REGISTRY: Dict[str, Callable[[ArchTarget], ISABackend]] = {
    "gfx908": Gfx9MfmaBackend,
    "gfx90a": Gfx9MfmaBackend,
    "gfx942": Gfx9MfmaBackend,
    "gfx950": Gfx950Backend,
    "gfx1151": Gfx11RdnaBackend,
    "gfx1201": Gfx12RdnaBackend,
    "gfx11-generic": Gfx11RdnaBackend,
}


def backend_for(arch: Union[str, ArchTarget]) -> ISABackend:
    """Resolve a gfx string or :class:`ArchTarget` to its ISA backend."""
    target = arch if isinstance(arch, ArchTarget) else ArchTarget.from_gfx(arch)
    cls = BACKEND_REGISTRY.get(target.gfx)
    if cls is None:
        raise KeyError(
            f"no ISA backend registered for {target.gfx!r}; "
            f"known: {sorted(BACKEND_REGISTRY)}"
        )
    return cls(target)
