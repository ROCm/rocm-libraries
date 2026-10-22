# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled-MFMA compute paths for MX GEMV/GEMM inner loops (the ``compute`` knob).

One call site, five interchangeable ways to run ``acc += A(fmt_a) @ B(fmt_b)``
with one E8M0 scale per 32 K on both sides:

==========================  ==========  =========  =======  =====================================
``compute``                 shape       engines    op_sel   instruction route
==========================  ==========  =========  =======  =====================================
``mfma_scaled_32x32x64``    32x32x64    both       no       catalog ``mfma_scaled`` atom (default)
``mfma_32x32x64_op``        32x32x64    python     yes      ``tile.mfma_scale_mx`` (free formats)
``mfma_16x16x128_op``       16x16x128   python     yes      ``tile.mfma_scale_mx``
``mfma_asm``                16x16x128   both       yes      ``tile.inline_asm`` MFMA chain
``mfma_asm_32x32x64``       32x32x64    both       yes      ``tile.inline_asm`` MFMA chain
==========================  ==========  =========  =======  =====================================

* ``engines="python"`` values use an op only the Python engine lowers: kernels
  using them compile with ``backend="python"``, carry a distinct kernel name
  (:func:`kernel_name_suffix`) and are excluded from the C++ byte-identity gate.
* ``op_sel`` values read the E8M0 byte ``opsel`` (0..3) of a scale dword, so
  one dword carries four K steps' scales. The catalog atom reads byte 0 only.
* 16x16x128 holds 16 A rows per MFMA (a decode GEMV with up to 16 rows wastes
  fewer rows than with 32) and has an fp4 operand of four VGPRs.
* The ``asm`` values are opaque to the compiler's scheduler and hazard
  recognizer: :class:`AsmHazard` holds the ``s_nop`` padding they need. The
  defaults are conservative; lowering them is a perf lever that must be
  re-verified on hardware. Two producer cases need lead padding before an asm
  chain: a VALU write of a source (``AsmHazard.lead``, default ``s_nop 1``)
  and an XDL write, i.e. a compiler-emitted MFMA (``op`` / ``catalog`` route)
  writing the accumulator or an operand just before, which needs far more
  wait states. :func:`mx_mma_chain` raises the lead to ``_XDL_LEAD`` when a
  source is defined by an MFMA op it can see; a chain whose accumulator is a
  loop-carried value fed by a compiler MFMA must set
  ``AsmHazard(after_xdl=True)``. After the chain, ``tail`` covers both VALU and
  MFMA readers of the result.
* Engine requirement of a built kernel: :func:`backend_for` with ``kernel=``
  reads :func:`rocke.core.ir.uses_python_only_ops`, so a kernel that also uses
  another Python-only op (e.g. the quantizer's ``cvt="intrinsic"``) is routed
  to the Python engine too.

Operand layouts per lane follow :mod:`rocke.core.arch.mfma_scale_mx` (the
16x16x128 fp8 operand is two 16-byte halves 64 K apart). ``pack_operands`` and
``unpack_acc`` here build and read those layouts on the host.

:func:`build_mx_mfma_probe` is a one-wave microkernel over host-preshuffled
fragments, used to verify every ``compute`` value bitwise and to
micro-benchmark issue rate (:func:`build_mx_mfma_rate`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from rocke.core.arch import ArchTarget
from rocke.core.arch import mfma_scale_mx as mx
from rocke.core.ir import (
    F32,
    I32,
    IRBuilder,
    PtrType,
    Value,
    VectorType,
    uses_python_only_ops,
)


@dataclass(frozen=True)
class MxCompute:
    """One value of the ``compute`` knob."""

    name: str
    shape: str
    route: str  # "catalog" | "op" | "asm"
    engines: str  # "both" | "python"
    opsel: bool

    @property
    def mx_shape(self) -> mx.MxShape:
        return mx.shape(self.shape)


COMPUTE_MODES: dict[str, MxCompute] = {
    c.name: c
    for c in (
        MxCompute("mfma_scaled_32x32x64", "32x32x64", "catalog", "both", False),
        MxCompute("mfma_32x32x64_op", "32x32x64", "op", "python", True),
        MxCompute("mfma_16x16x128_op", "16x16x128", "op", "python", True),
        MxCompute("mfma_asm", "16x16x128", "asm", "both", True),
        MxCompute("mfma_asm_32x32x64", "32x32x64", "asm", "both", True),
    )
}
DEFAULT_COMPUTE = "mfma_scaled_32x32x64"


def compute_mode(name: str) -> MxCompute:
    try:
        return COMPUTE_MODES[name]
    except KeyError:
        raise ValueError(
            f"compute must be one of {sorted(COMPUTE_MODES)}, got {name!r}"
        ) from None


def kernel_name_suffix(compute: str) -> str:
    """Kernel-name suffix marking builds outside the C++ byte-identity gate."""
    return "_pyonly" if compute_mode(compute).engines == "python" else ""


def backend_for(compute: str, kernel=None) -> str | None:
    """``compile_kernel`` backend a build with ``compute`` must use.

    With ``kernel`` (a built ``KernelDef``) the answer comes from the ops the
    kernel actually uses (:func:`rocke.core.ir.uses_python_only_ops`), and a
    ``python`` compute value whose kernel uses no Python-only op is an error
    (the table and the IR disagree).
    """
    static = "python" if compute_mode(compute).engines == "python" else None
    if kernel is None:
        return static
    used = uses_python_only_ops(kernel)
    if static == "python" and not used:
        raise ValueError(
            f"{compute} is Python-only but kernel {kernel.name!r} uses no "
            f"Python-only op"
        )
    return "python" if used else None


# ----------------------------------------------------------------- asm route


#: Wait states a VALU read of an XDL result needs: 8-pass (16x16x128) and
#: 16-pass (32x32x64) shapes, rounded up. ``s_nop n`` waits ``n + 1``.
_TAIL_NOPS = {"16x16x128": (15,), "32x32x64": (15, 7)}
#: Lead padding when an XDL instruction (a compiler-emitted MFMA) wrote a
#: source of the chain: 24 wait states, above the gfx950 XDL write -> XDL
#: SrcA/B/C read requirement of every pass count these shapes issue.
_XDL_LEAD = (15, 7)

#: Ops whose results an XDL instruction writes.
_XDL_OPS = frozenset({"tile.mma", "tile.mfma_scale_mx"})


def _wait_states(nops: Sequence[int]) -> int:
    return sum(int(n) + 1 for n in nops)


def _from_xdl(v: Value) -> bool:
    op = getattr(v, "op", None)
    if op is None:
        return False
    return op.name in _XDL_OPS or (
        op.name.startswith("tile.") and ("mfma" in op.name or "smfmac" in op.name)
    )


@dataclass(frozen=True)
class AsmHazard:
    """``s_nop`` padding around an inline-asm MFMA chain.

    ``lead`` before the first MFMA (VALU-written sources), ``inter`` between
    chained MFMAs, ``tail`` after the last one (its result and source reads).
    ``None`` picks the conservative default for the shape.

    ``after_xdl=True`` declares that a compiler-emitted MFMA may have written a
    source just before the chain (e.g. a loop-carried accumulator shared with
    the ``op`` / ``catalog`` route): the lead becomes at least ``_XDL_LEAD``.
    ``auto_xdl`` (default on) does the same when the chain can see that a
    source is an MFMA result; turning it off is for hazard tests only.
    """

    lead: tuple[int, ...] = (1,)
    inter: tuple[int, ...] = ()
    tail: tuple[int, ...] | None = None
    src: str = "v"  # register class of A/B: "v" or "a"
    after_xdl: bool = False
    auto_xdl: bool = True

    def tail_for(self, shape: str) -> tuple[int, ...]:
        return _TAIL_NOPS[shape] if self.tail is None else self.tail

    def lead_for(self, xdl_source: bool = False) -> tuple[int, ...]:
        xdl = self.after_xdl or (xdl_source and self.auto_xdl)
        if xdl and _wait_states(self.lead) < _wait_states(_XDL_LEAD):
            return _XDL_LEAD
        return self.lead


_DEFAULT_HAZARD = AsmHazard()


def _nops(values: Sequence[int]) -> list[str]:
    out = []
    for n in values:
        if not 0 <= int(n) <= 15:
            raise ValueError(f"s_nop count must be 0..15, got {n}")
        out.append(f"s_nop {int(n)}")
    return out


def _opsel_mods(opsel_a: int, opsel_b: int) -> str:
    # The 2-bit scale byte select: bit 0 in op_sel, bit 1 in op_sel_hi
    # (operand 0 = A scale, 1 = B scale).
    return (
        f"op_sel:[{opsel_a & 1},{opsel_b & 1},0] "
        f"op_sel_hi:[{(opsel_a >> 1) & 1},{(opsel_b >> 1) & 1},0]"
    )


def asm_template(
    shape: str,
    fmt_a: int,
    fmt_b: int,
    opsels: Sequence[tuple[int, int]],
    hazard: AsmHazard,
    *,
    xdl_source: bool = False,
) -> str:
    """Template of an ``n``-MFMA chain on one accumulator.

    Operands: ``$0`` acc out (tied to ``$1``), then per step ``i``:
    ``$(2+4i)`` A, ``$(3+4i)`` B, ``$(4+4i)`` A scale, ``$(5+4i)`` B scale.
    ``xdl_source``: a source is an MFMA result (see :meth:`AsmHazard.lead_for`).
    """
    mnem = f"v_mfma_scale_f32_{shape}_f8f6f4"
    fmt = (f" cbsz:{fmt_a}" if fmt_a else "") + (f" blgp:{fmt_b}" if fmt_b else "")
    lines = _nops(hazard.lead_for(xdl_source))
    for i, (sa, sb) in enumerate(opsels):
        if i:
            lines += _nops(hazard.inter)
        o = 2 + 4 * i
        lines.append(
            f"{mnem} $0, ${o}, ${o + 1}, $0, ${o + 2}, ${o + 3} "
            f"{_opsel_mods(sa, sb)}{fmt}"
        )
    lines += _nops(hazard.tail_for(shape))
    return "\n\t".join(lines)


# ----------------------------------------------------------------- emitters


@dataclass(frozen=True)
class MxStep:
    """One K step: operands, scale dwords and their byte selects."""

    a: Value
    b: Value
    sa: Value
    sb: Value
    opsel_a: int = 0
    opsel_b: int = 0


def _pad_words(b: IRBuilder, v: Value, words: int) -> Value:
    have = v.type.count
    if have == words:
        return v
    zero = b.const_i32(0)
    parts = [b.vec_extract(v, i) for i in range(have)] + [zero] * (words - have)
    return b.vec_pack(parts, I32)


def mx_mma_chain(
    b: IRBuilder,
    compute: str,
    acc: Value,
    steps: Sequence[MxStep],
    *,
    fmt_a="fp8",
    fmt_b="fp4",
    hazard: AsmHazard = _DEFAULT_HAZARD,
) -> Value:
    """``acc += sum_i A_i @ B_i`` in step order (one fp32 accumulation chain).

    ``a`` / ``b`` are ``<8|6|4 x i32>`` per their format (the catalog route
    pads to eight words); ``sa`` / ``sb`` are i32 scale dwords.
    """
    mode = compute_mode(compute)
    fa, fb = mx.format_code(fmt_a), mx.format_code(fmt_b)
    if not steps:
        return acc
    if mode.route == "asm":
        for s in steps:
            attrs = {
                "shape": mode.shape,
                "fmt_a": fa,
                "fmt_b": fb,
                "opsel_a": s.opsel_a,
                "opsel_b": s.opsel_b,
            }
            types = [v.type.name for v in (s.a, s.b, acc, s.sa, s.sb)]
            error = mx.check_operands(attrs, types)
            if error:
                raise ValueError(error)
        xdl = _from_xdl(acc) or any(
            _from_xdl(v) for s in steps for v in (s.a, s.b, s.sa, s.sb)
        )
        template = asm_template(
            mode.shape,
            fa,
            fb,
            [(s.opsel_a, s.opsel_b) for s in steps],
            hazard,
            xdl_source=xdl,
        )
        cls = hazard.src
        if cls not in ("v", "a"):
            raise ValueError("AsmHazard.src must be 'v' or 'a'")
        constraints = ",".join(["=v", "0"] + [cls, cls, "v", "v"] * len(steps))
        operands = [acc]
        for s in steps:
            operands += [s.a, s.b, s.sa, s.sb]
        return b.inline_asm(
            template,
            constraints,
            operands,
            acc.type,
            sideeffect=True,
            result_name_hint="mxasm",
        )
    for s in steps:
        if mode.route == "op":
            acc = b.mfma_scale_mx(
                s.a,
                s.b,
                acc,
                s.sa,
                s.sb,
                shape=mode.shape,
                fmt_a=fa,
                fmt_b=fb,
                opsel_a=s.opsel_a,
                opsel_b=s.opsel_b,
            )
        else:
            if s.opsel_a or s.opsel_b:
                raise ValueError(
                    f"{compute} reads scale byte 0 only; use an op_sel compute mode"
                )
            names = {v: k for k, v in mx.FORMAT_ALIASES.items() if len(k) == 3}
            atom = ArchTarget.from_gfx("gfx950").mma.op_for_shape(
                family="mfma_scaled",
                a_dtype=names[fa],
                b_dtype=names[fb],
                c_dtype="fp32",
                scales=("e8m0", "e8m0", 32),
                m=32,
                n=32,
                k=64,
            )
            acc = b.mma(
                atom, _pad_words(b, s.a, 8), _pad_words(b, s.b, 8), acc, s.sa, s.sb
            )
    return acc


def mx_mma(
    b: IRBuilder,
    compute: str,
    acc: Value,
    a: Value,
    w: Value,
    sa: Value,
    sb: Value,
    *,
    fmt_a="fp8",
    fmt_b="fp4",
    opsel_a: int = 0,
    opsel_b: int = 0,
    hazard: AsmHazard = _DEFAULT_HAZARD,
) -> Value:
    """One scaled MFMA step (``mx_mma_chain`` of one step)."""
    return mx_mma_chain(
        b,
        compute,
        acc,
        [MxStep(a, w, sa, sb, opsel_a, opsel_b)],
        fmt_a=fmt_a,
        fmt_b=fmt_b,
        hazard=hazard,
    )


# ----------------------------------------------------------------- host packing


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``codes[..., 32]`` (``bits`` each, LSB first) into i32 words."""
    codes = np.asarray(codes, dtype=np.uint64)
    shape = codes.shape[:-1]
    flat = codes.reshape(-1, 32)
    bitarr = ((flat[:, :, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    packed = np.packbits(bitarr.reshape(flat.shape[0], -1), axis=1, bitorder="little")
    return packed.view(np.uint32).reshape(*shape, -1)


def pack_a(codes: np.ndarray, shape: str, fmt, k_iters: int) -> np.ndarray:
    """A codes ``[M, K]`` -> per-lane words ``[k_iters, 64, words]`` (uint32)."""
    sh, f = mx.shape(shape), mx.format_code(fmt)
    out = np.zeros((k_iters, 64, 32), dtype=np.uint64)
    for lane in range(64):
        for s in range(32):
            r, k = mx.a_coord(mx.HostArith, shape, f, lane, s)
            out[:, lane, s] = codes[r, np.arange(k_iters) * sh.k + k]
    return _pack_codes(out, mx.FORMAT_BITS[f])


def pack_b(codes: np.ndarray, shape: str, fmt, k_iters: int) -> np.ndarray:
    """B codes given as ``[N, K]`` (column-major B) -> ``[k_iters, 64, words]``."""
    sh, f = mx.shape(shape), mx.format_code(fmt)
    out = np.zeros((k_iters, 64, 32), dtype=np.uint64)
    for lane in range(64):
        for s in range(32):
            k, c = mx.b_coord(mx.HostArith, shape, f, lane, s)
            out[:, lane, s] = codes[c, np.arange(k_iters) * sh.k + k]
    return _pack_codes(out, mx.FORMAT_BITS[f])


def pack_scales(
    scales: np.ndarray, shape: str, k_iters: int, *, opsel: bool, filler=None
) -> np.ndarray:
    """Scales ``[rows, K/32]`` (uint8; A rows or B columns) -> per-lane dwords.

    ``opsel=False``: ``[k_iters, 64]`` with the scale in byte 0. ``opsel=True``:
    ``[ceil(k_iters/4), 64]``, byte ``j`` = K step ``4q + j``. ``filler``
    (uint8 array broadcastable to the dwords' unused bytes) defaults to 127.
    """
    sh = mx.shape(shape)
    per = np.zeros((k_iters, 64), dtype=np.uint32)
    for lane in range(64):
        row, kb = mx.scale_a_coord(mx.HostArith, shape, lane)
        per[:, lane] = scales[row, np.arange(k_iters) * sh.k_blocks + kb]
    rng = np.random.default_rng(1234)
    if not opsel:
        hi = (
            rng.integers(0, 256, size=(k_iters, 64, 3), dtype=np.uint32)
            if filler is None
            else np.broadcast_to(np.asarray(filler, np.uint32), (k_iters, 64, 3))
        )
        return per | (hi[..., 0] << 8) | (hi[..., 1] << 16) | (hi[..., 2] << 24)
    q = (k_iters + 3) // 4
    out = np.full((q * 4, 64), 127, dtype=np.uint32)
    out[:k_iters] = per
    out = out.reshape(q, 4, 64)
    return out[:, 0] | (out[:, 1] << 8) | (out[:, 2] << 16) | (out[:, 3] << 24)


def unpack_acc(acc: np.ndarray, shape: str) -> np.ndarray:
    """Per-lane accumulators ``[64, c_len]`` -> C ``[M, N]``."""
    sh = mx.shape(shape)
    out = np.full((sh.m, sh.n), np.nan, dtype=acc.dtype)
    for lane in range(64):
        for i in range(sh.c_len):
            r, c = mx.c_coord(mx.HostArith, shape, lane, i)
            out[r, c] = acc[lane, i]
    return out


# ----------------------------------------------------------------- probe kernels


@dataclass(frozen=True)
class MxMfmaProbeSpec:
    """One-wave tile GEMM over preshuffled fragments: ``C[tile] = A @ B^T``.

    Inputs per tile ``t`` (``kit`` = K step, ``l`` = lane): ``A[t][kit][l]``
    and ``B[t][kit][l]`` operand words, ``SA[t][q][l]`` / ``SB[t][q][l]``
    scale dwords (``q = kit // 4`` with op_sel packing, else ``q = kit``),
    ``C[t][l][c_len]`` fp32 out. ``chain`` groups the K steps of one asm
    block (``mfma_asm*`` only). An fp6 operand is a ``<6 x i32>`` element,
    which LLVM lays out in 32 bytes: its buffer has a 32-byte lane stride.
    ``valu_operands`` adds an i32 kernel argument ``M`` XOR-ed into every
    operand and scale word just before its MFMA (launch with ``M = 0``), so
    the MFMA reads registers a VALU instruction has just written: the hazard
    case for the inline-asm route's ``AsmHazard.lead`` padding.
    ``compute_alt`` (same shape) alternates the route per ``chain`` group on
    the one accumulator: groups 0, 2, ... use ``compute``, 1, 3, ... use
    ``compute_alt`` (the XDL-write -> asm-MFMA and asm -> compiler-MFMA
    hazard cases).
    """

    k_iters: int = 4
    compute: str = DEFAULT_COMPUTE
    fmt_a: str = "fp8"
    fmt_b: str = "fp4"
    opsel: bool = False
    chain: int = 4
    hazard: AsmHazard = field(default_factory=AsmHazard)
    valu_operands: bool = False
    compute_alt: str | None = None
    name: str = "mx_mfma_probe"

    def computes(self) -> tuple[str, ...]:
        return (self.compute,) + ((self.compute_alt,) if self.compute_alt else ())

    def kernel_name(self) -> str:
        alt = f"_alt_{self.compute_alt}_c{self.chain}" if self.compute_alt else ""
        suffix = (
            "_pyonly" if any(kernel_name_suffix(c) for c in self.computes()) else ""
        )
        return (
            f"{self.name}_{self.compute}{alt}_{self.fmt_a}x{self.fmt_b}_k{self.k_iters}"
            f"{'_opsel' if self.opsel else ''}{'_valu' if self.valu_operands else ''}"
            f"{suffix}"
        )


def _validate_probe(spec: MxMfmaProbeSpec) -> MxCompute:
    mode = compute_mode(spec.compute)
    for c in spec.computes():
        m = compute_mode(c)
        if spec.opsel and not m.opsel:
            raise ValueError(f"{c} has no op_sel scale select")
        if m.shape != mode.shape:
            raise ValueError("compute_alt must have the shape of compute")
    if spec.k_iters < 1 or spec.chain < 1:
        raise ValueError("k_iters and chain must be >= 1")
    mx.format_code(spec.fmt_a)
    mx.format_code(spec.fmt_b)
    return mode


def build_mx_mfma_probe(spec: MxMfmaProbeSpec, *, arch: str = "gfx950"):
    if arch != "gfx950":
        raise ValueError("scaled MFMA probe requires gfx950")
    mode = _validate_probe(spec)
    sh = mode.mx_shape
    fa, fb = mx.format_code(spec.fmt_a), mx.format_code(spec.fmt_b)
    wa, wb = mx.OPERAND_WORDS[fa], mx.OPERAND_WORDS[fb]
    nsc = (spec.k_iters + 3) // 4 if spec.opsel else spec.k_iters
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64
    pa = b.param(
        "A",
        PtrType(VectorType(I32, wa), "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    pb = b.param(
        "B",
        PtrType(VectorType(I32, wb), "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    psa = b.param("SA", PtrType(I32, "global"), noalias=True, readonly=True, align=4)
    psb = b.param("SB", PtrType(I32, "global"), noalias=True, readonly=True, align=4)
    pc = b.param("C", PtrType(F32, "global"), noalias=True, align=4)
    mask = b.param("M", I32) if spec.valu_operands else None
    lane = b.thread_id_x()
    tile = b.block_id_x()
    c64 = b.const_i32(64)

    def touch(v):
        if mask is None:
            return v
        if isinstance(v.type, VectorType):
            n = v.type.count
            return b.vec_pack([b.xor(b.vec_extract(v, i), mask) for i in range(n)], I32)
        return b.xor(v, mask)

    def at(count, i):
        return b.add(
            b.mul(b.add(b.mul(tile, b.const_i32(count)), b.const_i32(i)), c64), lane
        )

    sas = [b.global_load(psa, at(nsc, q), I32, align=4) for q in range(nsc)]
    sbs = [b.global_load(psb, at(nsc, q), I32, align=4) for q in range(nsc)]
    steps = []
    for kit in range(spec.k_iters):
        a = b.global_load(pa, at(spec.k_iters, kit), VectorType(I32, wa), align=16)
        w = b.global_load(pb, at(spec.k_iters, kit), VectorType(I32, wb), align=16)
        q, sel = (kit // 4, kit % 4) if spec.opsel else (kit, 0)
        steps.append(MxStep(touch(a), touch(w), touch(sas[q]), touch(sbs[q]), sel, sel))
    acc = b.zero_vec_f32(sh.c_len)
    group = spec.chain if mode.route == "asm" or spec.compute_alt else len(steps)
    for i in range(0, len(steps), group):
        compute = spec.computes()[(i // group) % len(spec.computes())]
        acc = mx_mma_chain(
            b,
            compute,
            acc,
            steps[i : i + group],
            fmt_a=fa,
            fmt_b=fb,
            hazard=spec.hazard,
        )
    base = b.mul(b.add(b.mul(tile, c64), lane), b.const_i32(sh.c_len))
    for i in range(sh.c_len):
        b.global_store(pc, b.add(base, b.const_i32(i)), b.vec_extract(acc, i), align=4)
    b.ret()
    return b.kernel


@dataclass(frozen=True)
class MxMfmaRateSpec:
    """Issue-rate microkernel: each wave runs ``iters`` rounds of ``n_acc``
    independent MFMA chains (``n_acc = 1``: one dependent chain, latency bound).

    Operands come from registers loaded once; the accumulators are stored at
    the end so nothing is dead. ``waves`` waves per workgroup.
    """

    compute: str = DEFAULT_COMPUTE
    fmt_a: str = "fp8"
    fmt_b: str = "fp4"
    n_acc: int = 4
    iters: int = 256
    waves: int = 4
    hazard: AsmHazard = field(default_factory=lambda: AsmHazard(lead=(), tail=()))
    name: str = "mx_mfma_rate"

    def kernel_name(self) -> str:
        return (
            f"{self.name}_{self.compute}_{self.fmt_a}x{self.fmt_b}_a{self.n_acc}"
            f"_i{self.iters}{kernel_name_suffix(self.compute)}"
        )


def build_mx_mfma_rate(spec: MxMfmaRateSpec, *, arch: str = "gfx950"):
    if arch != "gfx950":
        raise ValueError("scaled MFMA rate probe requires gfx950")
    mode = compute_mode(spec.compute)
    sh = mode.mx_shape
    fa, fb = mx.format_code(spec.fmt_a), mx.format_code(spec.fmt_b)
    wa, wb = mx.OPERAND_WORDS[fa], mx.OPERAND_WORDS[fb]
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64 * spec.waves
    pin = b.param("X", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    pc = b.param("C", PtrType(F32, "global"), noalias=True, align=4)
    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))
    # X rows (64 lanes each): 0..7 A words, 8..15 B words, 16 A scale, 17 B scale.
    words = [
        b.global_load(pin, b.add(lane, b.const_i32(64 * i)), I32, align=4)
        for i in range(18)
    ]
    a = b.vec_pack(words[:wa], I32)
    w = b.vec_pack(words[8 : 8 + wb], I32)
    sa, sb = words[16], words[17]
    inits = [(f"acc{i}", b.zero_vec_f32(sh.c_len)) for i in range(spec.n_acc)]
    loop = b.scf_for_iter(
        b.const_i32(0), b.const_i32(spec.iters), b.const_i32(1), inits, iv_name="it"
    )
    with loop as (_iv, accs):
        if mode.route == "asm" and spec.n_acc > 1:
            # One asm block with n_acc independent MFMAs (one schedule node).
            outs = _asm_independent(b, mode, fa, fb, accs, a, w, sa, sb, spec.hazard)
        else:
            outs = [
                mx_mma(
                    b,
                    spec.compute,
                    acc,
                    a,
                    w,
                    sa,
                    sb,
                    fmt_a=fa,
                    fmt_b=fb,
                    hazard=spec.hazard,
                )
                for acc in accs
            ]
        b.scf_yield(*outs)
    if mode.route == "asm":
        # In the loop an asm MFMA only feeds the next one (no padding needed
        # between them); the stores after the loop wait out the last result.
        b.inline_asm("\n\t".join(_nops(_TAIL_NOPS[mode.shape])), "", [])
    gid = b.add(b.mul(b.block_id_x(), b.const_i32(64 * spec.waves)), tid)
    base = b.mul(gid, b.const_i32(sh.c_len * spec.n_acc))
    for j, res in enumerate(loop.results):
        for i in range(sh.c_len):
            idx = b.add(base, b.const_i32(j * sh.c_len + i))
            b.global_store(pc, idx, b.vec_extract(res, i), align=4)
    b.ret()
    return b.kernel


def _asm_independent(b, mode, fa, fb, accs, a, w, sa, sb, hazard):
    n = len(accs)
    mnem = f"v_mfma_scale_f32_{mode.shape}_f8f6f4"
    fmt = (f" cbsz:{fa}" if fa else "") + (f" blgp:{fb}" if fb else "")
    xdl = any(_from_xdl(v) for v in list(accs) + [a, w, sa, sb])
    lines = _nops(hazard.lead_for(xdl))
    for i in range(n):
        lines.append(
            f"{mnem} ${i}, ${2 * n}, ${2 * n + 1}, ${i}, ${2 * n + 2}, ${2 * n + 3} {_opsel_mods(0, 0)}{fmt}"
        )
    lines += _nops(hazard.tail_for(mode.shape))
    cls = hazard.src
    constraints = ",".join(
        ["=v"] * n + [str(i) for i in range(n)] + [cls, cls, "v", "v"]
    )
    return b.inline_asm_multi(
        "\n\t".join(lines),
        constraints,
        list(accs) + [a, w, sa, sb],
        result_types=[acc.type for acc in accs],
        sideeffect=True,
        result_name_hint="mxasm",
    )


__all__ = [
    "COMPUTE_MODES",
    "DEFAULT_COMPUTE",
    "AsmHazard",
    "MxCompute",
    "MxMfmaProbeSpec",
    "MxMfmaRateSpec",
    "MxStep",
    "asm_template",
    "backend_for",
    "build_mx_mfma_probe",
    "build_mx_mfma_rate",
    "compute_mode",
    "kernel_name_suffix",
    "mx_mma",
    "mx_mma_chain",
    "pack_a",
    "pack_b",
    "pack_scales",
    "unpack_acc",
]
