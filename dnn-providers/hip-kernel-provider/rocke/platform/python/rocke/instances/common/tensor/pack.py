# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor pack: one strided operand copied into a packed mode order.

``Y[modes_dst] = X[modes_src]`` (the same modes, ``Y`` packed row-major in
``modes_dst`` order, ``X`` with arbitrary element strides). The tensor GETT
plan uses it to give an operand a copy the fast GEMM bodies read directly:
free modes in one contiguous run (MN-layout bodies) or K-contiguous rows
(the RCR bodies).

Three forms (``form``, ``"auto"`` picks the first that applies):

* ``pairs``: the destination's fastest mode is the source's unit-stride mode
  with even runs. Each thread writes ``groups`` vectors of 8 consecutive ``Y``
  elements (16-byte stores) gathered in 2-element (4-byte) loads.
* ``tiled``: a transposing copy. The destination splits into an outer group
  ``O`` (holding the source's unit-stride mode, innermost in ``O``, even
  extent) and the trailing group ``I`` (``|I|`` a multiple of 8). A
  workgroup moves a 64 (O) x 64 (I) tile through LDS: 4-byte loads of
  consecutive ``O`` element pairs (coalesced along the source's contiguous
  runs), a padded ``[I][O]`` LDS tile, and 16-byte stores along ``I``.
* ``elements``: any order; 8 element loads per 16-byte store.

Offsets are mixed-radix digits of the destination indices (compile-time
extents and strides). Elements past the tensor fall outside the buffer ranges
(loads return zeros, stores drop), so there are no branches. f16 / bf16.
The C++ engine mirrors the builder (``cpp/instances/common/tensor_gett_pack.cpp``;
the ``tensor_pack`` parity family keeps the two engines byte-identical).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ....core.ir import BF16, F16, I32, IRBuilder, KernelDef, PtrType, VectorType
from ....helpers.spec import SignatureBuilder
from .gett_plan import merge_digits, packed_strides

ModeMap = Tuple[Tuple[str, int], ...]
TENSOR_PACK_DTYPES = ("fp16", "bf16")
TENSOR_PACK_FORMS = ("auto", "pairs", "tiled", "elements")
#: largest source / destination byte span the 32-bit buffer offsets address
MAX_PACK_BYTES = 0x7FFF0000
#: tiled form: tile edge (elements) along O and I, LDS row pad (elements)
TILE = 64
TILE_PAD = 2


def _ty(dtype: str):
    return {"f16": F16, "fp16": F16, "bf16": BF16}[dtype]


@dataclass(frozen=True)
class TensorPackSpec:
    """``Y[modes_dst] = X[modes_src]``. Mode lists are row-major (last mode
    fastest); ``strides_src`` are element strides (``None``: packed in
    ``modes_src`` order). ``groups``: 8-element output vectors per thread
    (pairs / elements forms); ``block``: threads per workgroup (256 for the
    tiled form); ``form``: see the module doc."""

    modes_src: Tuple[str, ...]
    modes_dst: Tuple[str, ...]
    extents: ModeMap
    strides_src: Optional[ModeMap] = None
    dtype: str = "fp16"
    groups: int = 2
    block: int = 256
    form: str = "auto"

    @staticmethod
    def make(
        modes_src, modes_dst, extents, *, strides_src=None, **kw
    ) -> "TensorPackSpec":
        return TensorPackSpec(
            modes_src=tuple(modes_src),
            modes_dst=tuple(modes_dst),
            extents=tuple(sorted((str(k), int(v)) for k, v in extents.items())),
            strides_src=(
                None
                if strides_src is None
                else tuple(sorted((str(k), int(v)) for k, v in strides_src.items()))
            ),
            **kw,
        )

    def ext(self) -> Dict[str, int]:
        return dict(self.extents)

    def src_strides(self) -> Dict[str, int]:
        if self.strides_src is None:
            return packed_strides(self.modes_src, self.ext())
        return dict(self.strides_src)

    def numel(self) -> int:
        n = 1
        for m in self.modes_dst:
            n *= self.ext()[m]
        return n

    def src_digits(self) -> List[Tuple[int, int]]:
        """(extent, source stride) digits, inner first, of the destination
        index."""
        return merge_digits(self.modes_dst, self.ext(), self.src_strides())

    def unit_mode(self) -> Optional[str]:
        """The source's unit-stride mode (extent > 1), if any."""
        st, ext = self.src_strides(), self.ext()
        u = [m for m in self.modes_src if st[m] == 1 and ext[m] > 1]
        return u[0] if u else None

    def pairs(self) -> bool:
        """``pairs`` form applies: the destination's fastest mode is the
        source's unit-stride mode and element pairs are contiguous."""
        st = self.src_strides()
        if st[self.modes_dst[-1]] != 1:
            return False
        sdig = self.src_digits()
        return sdig[0][0] % 2 == 0 and all(s % 2 == 0 for _e, s in sdig[1:])

    def tiled_groups(self) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """(O modes, I modes) of the ``tiled`` form, or None when it does not
        apply: I = the trailing destination modes after the source's
        unit-stride mode u, u innermost in O with an even extent, element
        pairs contiguous, |I| a multiple of 8."""
        u = self.unit_mode()
        if u is None or u not in self.modes_dst:
            return None
        k = self.modes_dst.index(u)
        O, I = self.modes_dst[: k + 1], self.modes_dst[k + 1 :]
        if not I:
            return None
        ext, st = self.ext(), self.src_strides()
        pi = 1
        for m in I:
            pi *= ext[m]
        if pi % 8:
            return None
        odig = merge_digits(O, ext, st)
        if odig[0][1] != 1 or odig[0][0] % 2 or any(s % 2 for _e, s in odig[1:]):
            return None
        if any(st[m] % 2 for m in I):
            return None
        return O, I

    def resolved_form(self) -> str:
        if self.form != "auto":
            return self.form
        if self.pairs():
            return "pairs"
        if self.tiled_groups() is not None:
            return "tiled"
        return "elements"

    def kernel_name(self) -> str:
        key = json.dumps(asdict(self), sort_keys=True)
        dt = "f16" if self.dtype in ("f16", "fp16") else self.dtype
        return f"tensor_pack_{dt}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def is_valid_spec(spec: TensorPackSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    if spec.dtype not in TENSOR_PACK_DTYPES + ("f16",):
        return False, f"dtype {spec.dtype!r} not in {TENSOR_PACK_DTYPES}"
    if sorted(spec.modes_src) != sorted(spec.modes_dst):
        return False, "source and destination modes differ"
    if set(spec.ext()) != set(spec.modes_src):
        return False, "extents must name exactly the tensor's modes"
    if spec.form not in TENSOR_PACK_FORMS:
        return False, f"form {spec.form!r} not in {TENSOR_PACK_FORMS}"
    if spec.groups not in (1, 2, 4):
        return False, f"groups {spec.groups} not in (1, 2, 4)"
    if spec.block not in (64, 128, 256, 512):
        return False, f"block {spec.block} not in (64, 128, 256, 512)"
    ext, st = spec.ext(), spec.src_strides()
    if any(v <= 0 for v in ext.values()):
        return False, "extents must be positive"
    if spec.numel() % 8:
        return False, f"{spec.numel()} elements is not a multiple of 8 (16-byte stores)"
    form = spec.resolved_form()
    if form == "pairs" and not spec.pairs():
        return (
            False,
            "pairs form: the destination's fastest mode is not the source's unit-stride run",
        )
    if form == "tiled":
        if spec.tiled_groups() is None:
            return False, "tiled form does not apply (see TensorPackSpec.tiled_groups)"
        if spec.block != 256:
            return False, "tiled form: block must be 256"
    span = 2 * (1 + sum((ext[m] - 1) * st[m] for m in spec.modes_src))
    if span > MAX_PACK_BYTES or 2 * spec.numel() > MAX_PACK_BYTES:
        return False, "tensor spans more than 2 GiB (32-bit buffer offsets)"
    return True, "ok"


def _digits(b: IRBuilder, x, digits: Sequence[Tuple[int, int]]):
    off = None
    q = x
    for j, (e, s) in enumerate(digits):
        last = j == len(digits) - 1
        d = q if last else b.mod(q, b.const_i32(e))
        term = d if s == 1 else b.mul(d, b.const_i32(s))
        off = term if off is None else b.add(off, term)
        if not last:
            q = b.div(q, b.const_i32(e))
    return off


def _src_bytes(spec: TensorPackSpec) -> int:
    ext, st = spec.ext(), spec.src_strides()
    return 2 * (1 + sum((ext[m] - 1) * st[m] for m in spec.modes_src))


def build_tensor_pack(spec: TensorPackSpec, *, arch: str = "gfx950") -> KernelDef:
    """Params: X (source), Y (destination); grid :func:`tensor_pack_grid`,
    block :func:`tensor_pack_block`."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid tensor pack spec: {why}")
    dt = _ty(spec.dtype)
    b = IRBuilder(spec.kernel_name())
    X = b.param("X", PtrType(dt, "global"), noalias=True, readonly=True, align=4)
    Y = b.param("Y", PtrType(dt, "global"), noalias=True, writeonly=True, align=16)
    n = spec.numel()
    rs_x = b.buffer_rsrc(X, b.const_i32(_src_bytes(spec)))
    rs_y = b.buffer_rsrc(Y, b.const_i32(2 * n))
    c0 = b.const_i32(0)
    big = b.const_i32(0x7FFFFFF0)
    st_fn = b.buffer_store_vN_bf16 if dt is BF16 else b.buffer_store_vN_f16
    form = spec.resolved_form()
    if form == "tiled":
        _build_tiled(b, spec, dt, rs_x, rs_y, c0, big, st_fn)
        return b.kernel
    dig = spec.src_digits()
    tid = b.thread_id_x()
    base = b.mul(
        b.add(b.mul(b.block_id_x(), b.const_i32(spec.block * spec.groups)), tid),
        b.const_i32(8),
    )
    vec8 = VectorType(dt, 8)
    for g in range(spec.groups):
        e0 = b.add(base, b.const_i32(g * spec.block * 8)) if g else base
        inb = b.cmp_lt(e0, b.const_i32(n))
        if form == "pairs":
            words = []
            for j in range(0, 8, 2):
                e = b.add(e0, b.const_i32(j)) if j else e0
                so = b.mul(_digits(b, e, dig), b.const_i32(2))
                so = b.select(inb, so, big)
                words.append(b.bitcast(b.buffer_load_vN(rs_x, so, c0, dt, 2), I32))
            vec = b.vec_bitcast(b.vec_pack(words, I32), vec8)
        else:
            els = []
            for j in range(8):
                e = b.add(e0, b.const_i32(j)) if j else e0
                so = b.mul(_digits(b, e, dig), b.const_i32(2))
                so = b.select(inb, so, big)
                els.append(b.buffer_load(rs_x, so, c0, dt))
            vec = b.vec_pack(els, dt)
        do = b.select(inb, b.mul(e0, b.const_i32(2)), big)
        st_fn(rs_y, do, c0, vec, 4)
    return b.kernel


def _build_tiled(b, spec, dt, rs_x, rs_y, c0, big, st_fn) -> None:
    """The ``tiled`` form: one 64 (O) x 64 (I) tile per workgroup through a
    padded [I][O] LDS tile (see the module doc)."""
    O, I = spec.tiled_groups()
    ext, st = spec.ext(), spec.src_strides()
    p_o = 1
    for m in O:
        p_o *= ext[m]
    p_i = 1
    for m in I:
        p_i *= ext[m]
    odig = merge_digits(O, ext, st)
    idig = merge_digits(I, ext, st)
    sm = b.smem_alloc(dt, [TILE, TILE + TILE_PAD], name_hint="pack_tile")
    tid = b.thread_id_x()
    i_blk = b.mul(b.block_id_x(), b.const_i32(TILE))
    o_blk = b.mul(b.block_id_y(), b.const_i32(TILE))
    # load: thread -> O pair op (32 per row) and I rows ib + 8 j
    op = b.mod(tid, b.const_i32(TILE // 2))
    ib = b.div(tid, b.const_i32(TILE // 2))
    o = b.add(o_blk, b.mul(op, b.const_i32(2)))
    ok_o = b.cmp_lt(o, b.const_i32(p_o))
    ro = _digits(b, o, odig)
    rows_per_pass = 256 // (TILE // 2)
    for j in range(TILE // rows_per_pass):
        il = b.add(ib, b.const_i32(j * rows_per_pass)) if j else ib
        i = b.add(i_blk, il)
        ok = b.land(ok_o, b.cmp_lt(i, b.const_i32(p_i)))
        so = b.mul(b.add(ro, _digits(b, i, idig)), b.const_i32(2))
        so = b.select(ok, so, big)
        pr = b.buffer_load_vN(rs_x, so, c0, dt, 2)
        b.smem_store_vN(sm, [il, b.mul(op, b.const_i32(2))], pr, 2)
    b.sync()
    # store: 16-byte vectors of 8 consecutive I per (o, chunk)
    chunks = TILE // 8
    for jj in range(TILE * chunks // 256):
        ci = b.add(tid, b.const_i32(256 * jj)) if jj else tid
        ol = b.div(ci, b.const_i32(chunks))
        ic = b.mul(b.mod(ci, b.const_i32(chunks)), b.const_i32(8))
        els = [
            b.vec_extract(
                b.smem_load_vN(
                    sm, b.add(ic, b.const_i32(e)) if e else ic, ol, dtype=dt, n=1
                ),
                0,
            )
            for e in range(8)
        ]
        vec = b.vec_pack(els, dt)
        o2 = b.add(o_blk, ol)
        i2 = b.add(i_blk, ic)
        ok = b.land(b.cmp_lt(o2, b.const_i32(p_o)), b.cmp_lt(i2, b.const_i32(p_i)))
        do = b.mul(b.add(b.mul(o2, b.const_i32(p_i)), i2), b.const_i32(2))
        st_fn(rs_y, b.select(ok, do, big), c0, vec, 4)


def tensor_pack_grid(spec: TensorPackSpec) -> Tuple[int, int, int]:
    if spec.resolved_form() == "tiled":
        O, I = spec.tiled_groups()
        ext = spec.ext()
        p_o = 1
        for m in O:
            p_o *= ext[m]
        p_i = 1
        for m in I:
            p_i *= ext[m]
        return (-(-p_i // TILE), -(-p_o // TILE), 1)
    per = spec.block * spec.groups * 8
    return (max(1, -(-spec.numel() // per)), 1, 1)


def tensor_pack_block(spec: TensorPackSpec) -> Tuple[int, int, int]:
    return (spec.block, 1, 1)


def tensor_pack_signature(spec: TensorPackSpec):
    dt = "f16" if spec.dtype in ("f16", "fp16") else spec.dtype
    return SignatureBuilder().ptr("X", dt).ptr("Y", dt).build()


__all__ = [
    "MAX_PACK_BYTES",
    "TENSOR_PACK_DTYPES",
    "TENSOR_PACK_FORMS",
    "TensorPackSpec",
    "build_tensor_pack",
    "is_valid_spec",
    "tensor_pack_block",
    "tensor_pack_grid",
    "tensor_pack_signature",
]
