# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One wave computes ``D = A @ B + C`` with a gfx942 f64 MFMA atom.

Each K step of 4 issues one MFMA, and operands are placed with the catalog lane
maps, so the probe checks the A/B and C/D layouts as well as the instruction.

* ``m=16`` (``mfma_f64_16x16x4_f64``): ``A`` is row-major ``16 x k``, ``B`` is
  row-major ``k x 16``, and ``C``/``D`` are row-major ``16 x 16``.
* ``m=4`` (``mfma_f64_4x4x4_f64``): four independent ``4 x 4`` blocks stacked
  along rows. ``A`` is row-major ``16 x k`` (block ``b`` is rows ``4b..4b+3``),
  ``B`` is ``k / 4`` row-major ``16 x 4`` slabs (slab ``s`` holds rows
  ``4s..4s+3`` of every block's ``k x 4`` operand, block ``b`` at rows
  ``4b..4b+3``), and ``C``/``D`` are row-major ``16 x 4``.
"""

from dataclasses import dataclass

from ...core.arch import ArchTarget
from ...core.ir import F64, IRBuilder, KernelDef, PtrType

# m -> (B slab rows, B/C/D columns) per K step of 4.
_B_SLAB = {16: (4, 16), 4: (16, 4)}


@dataclass(frozen=True)
class F64MmaProbeSpec:
    m: int = 16
    k: int = 4

    def __post_init__(self):
        if self.m not in _B_SLAB:
            raise ValueError("f64 MMA probe requires m=16 or m=4")
        if self.k <= 0 or self.k % 4:
            raise ValueError("f64 MMA probe requires k to be a positive multiple of 4")

    @property
    def name(self):
        return f"f64_mma_probe_m{self.m}_k{self.k}"


def build_f64_mma_probe(spec: F64MmaProbeSpec) -> KernelDef:
    """Build a probe for the fixed gfx942 F64 instruction and lane-map contract."""
    target = ArchTarget.from_gfx("gfx942")
    atom = target.mma.op_for_shape(
        family="mma",
        a_dtype="fp64",
        b_dtype="fp64",
        c_dtype="fp64",
        m=spec.m,
        n=spec.m,
        k=4,
    )
    # Reused output indices require the actual instruction maps, not just equal widths.
    expected_id = "mfma_f64_16x16x4_f64" if spec.m == 16 else "mfma_f64_4x4x4_f64"
    registered = target.mma.by_op_id(expected_id)

    def matches(desc, expected):
        layout = desc.layout
        return (
            desc.dtype == expected.dtype
            and type(desc.frag_len) is int
            and desc.frag_len == expected.frag_len
            and layout is not None
            and type(layout.frag_len) is int
            and layout.frag_len == expected.layout.frag_len
            and type(layout.wave_size) is int
            and layout.wave_size == expected.layout.wave_size
            and callable(layout.fn)
            and layout.fn is expected.layout.fn
        )

    if (
        atom is None
        or registered is None
        or atom.op_id != registered.op_id
        or atom.family != registered.family
        or atom.shape != registered.shape
        or any(type(size) is not int for size in (*atom.shape, atom.wave_size))
        or atom.wave_size != registered.wave_size
        or any(
            not matches(desc, expected)
            for desc, expected in zip(
                (*atom.srcs, atom.dst), (*registered.srcs, registered.dst)
            )
        )
    ):
        raise ValueError(
            "F64 probe requires the fixed gfx942 instruction and layout contract"
        )
    slab_rows, cols = _B_SLAB[spec.m]
    b = IRBuilder(spec.name)
    a = b.param("A", PtrType(F64, "global"), align=8)
    bb = b.param("B", PtrType(F64, "global"), align=8)
    cc = b.param("C", PtrType(F64, "global"), align=8)
    d = b.param("D", PtrType(F64, "global"), align=8)
    lane = b.thread_id_x()
    cn = b.const_i32(cols)
    ck = b.const_i32(spec.k)
    c_values, out_indices = [], []
    for slot in range(atom.srcs[2].frag_len):
        r, col = atom.src_layout(2).coord(b, lane, slot)
        index = b.add(b.mul(r, cn), col)
        out_indices.append(index)
        c_values.append(b.global_load(cc, index, F64, align=8))
    acc = b.vec_pack(c_values, F64)
    a_row, a_k = atom.src_layout(0).coord(b, lane, 0)
    b_row, b_col = atom.src_layout(1).coord(b, lane, 0)
    for step in range(spec.k // 4):
        a_index = b.add(b.mul(a_row, ck), b.add(b.const_i32(step * 4), a_k))
        b_slab_row = b.add(b.const_i32(step * slab_rows), b_row)
        b_index = b.add(b.mul(b_slab_row, cn), b_col)
        av = b.global_load(a, a_index, F64, align=8)
        bv = b.global_load(bb, b_index, F64, align=8)
        acc = b.mma(atom, av, bv, acc)
    for slot, index in enumerate(out_indices):
        b.global_store(d, index, b.vec_extract(acc, slot), align=8)
    b.ret()
    return b.kernel
