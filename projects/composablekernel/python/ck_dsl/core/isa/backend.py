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
        spec = _RDNA_WMMA.get(op.name)
        if spec is None:
            raise NotImplementedError(
                f"WMMA op {op.name!r} not yet wired for {self.arch.gfx}; "
                f"known: {sorted(_RDNA_WMMA)}"
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

    def encode_waitcnt(self, vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
        # RDNA gfx11 uses a different s_waitcnt field layout than the gfx9/10
        # split the base encodes: contiguous expcnt[2:0] / lgkmcnt[9:4] /
        # vmcnt[15:10] (no split VMCNT, 6-bit LGKMCNT). The layout was read
        # off the ROCm 7.0.2 AMDGPU assembler on a gfx1151 node; see
        # _encode_waitcnt_gfx11 for the empirical encodings.
        from ..lower_llvm import _encode_waitcnt_gfx11

        return _encode_waitcnt_gfx11(vmcnt, expcnt, lgkmcnt)


# gfx -> backend class. Adding a CDNA gfx is one row here plus, when its codegen
# actually diverges, a new subclass.
BACKEND_REGISTRY: Dict[str, Callable[[ArchTarget], ISABackend]] = {
    "gfx908": Gfx9MfmaBackend,
    "gfx90a": Gfx9MfmaBackend,
    "gfx942": Gfx9MfmaBackend,
    "gfx950": Gfx950Backend,
    "gfx1151": Gfx11RdnaBackend,
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
