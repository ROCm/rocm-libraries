# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One wave per dense atom, with explicit FP32 input-preparation variants."""

from dataclasses import dataclass

from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I32, TF32, IRBuilder, PtrType, VectorType
from rocke.core.storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from rocke.helpers.mma_io import load_matrix_fragment

PREPARATIONS = ("raw", "carrier", "rne", "prepacked", "fp32")


@dataclass(frozen=True)
class Tf32MmaProbeSpec:
    m: int = 16
    preparation: str = "raw"

    def __post_init__(self):
        if self.m not in (16, 32) or self.preparation not in PREPARATIONS:
            raise ValueError("TF32 probe requires m=16/32 and a known preparation")

    @property
    def name(self):
        return f"tf32_probe_{self.m}_{self.preparation}"


def build_tf32_mma_probe(spec: Tf32MmaProbeSpec):
    m, mode = spec.m, spec.preparation
    k = 128 // m
    target = ArchTarget.from_gfx("gfx942")
    atom = target.mma.op_for_shape(
        family="mma",
        a_dtype="tf32",
        b_dtype="tf32",
        c_dtype="fp32",
        m=m,
        n=m,
        k=k,
    )
    b = IRBuilder(spec.name)
    ty = I32 if mode == "prepacked" else F32
    a = b.param("A", PtrType(ty, "global"), align=4)
    bb = b.param("B", PtrType(ty, "global"), align=4)
    cc = b.param("C", PtrType(F32, "global"), align=4)
    d = b.param("D", PtrType(F32, "global"), align=4)
    pa = b.param("PA", PtrType(I32, "global"), align=4)
    pb = b.param("PB", PtrType(I32, "global"), align=4)
    lane = b.thread_id_x()
    batch = b.block_id_x()
    cm = b.const_i32(m)
    axis = b.mod(lane, cm)
    group = b.div(lane, cm)
    ck = b.const_i32(k)
    row = b.mul(axis, ck)
    batch_input = b.mul(batch, b.const_i32(m * k))
    row_base = b.add(batch_input, row)
    batch_output = b.mul(batch, b.const_i32(m * m))
    c_values, out_indices = [], []
    for slot in range(atom.c_frag_len):
        r, col = atom.c_layout().coord(b, lane, slot)
        index = b.add(batch_output, b.add(b.mul(r, cm), col))
        out_indices.append(index)
        c_values.append(b.global_load(cc, index, F32, align=4))
    acc = b.vec_pack(c_values, F32)
    if mode == "fp32":
        full = target.mma.op_for_shape(
            family="mma",
            a_dtype="fp32",
            b_dtype="fp32",
            c_dtype="fp32",
            m=m,
            n=m,
            k=k // 2,
        )
        for step in range(2):
            index = b.add(row_base, b.add(group, b.const_i32(step * (64 // m))))
            av = b.global_load(a, index, F32, align=4)
            bv = b.global_load(bb, index, F32, align=4)
            b.global_store(pa, index, b.bitcast(av, I32), align=4)
            b.global_store(pb, index, b.bitcast(bv, I32), align=4)
            acc = b.mma(full, av, bv, acc)
    else:
        layout = MatrixFragmentLayout(
            FragmentPacking(BitPacking(32), 2, 32, 2), 2, 64 // m, m
        )
        fragments = []
        for ptr, prepared in ((a, pa), (bb, pb)):
            carrier = I32 if mode in ("carrier", "prepacked") else F32
            values = load_matrix_fragment(
                b,
                ptr,
                row_base,
                group,
                0,
                dtype="tf32" if mode == "prepacked" else "fp32",
                layout=layout,
                carrier_type=carrier,
                alignment_bytes=4,
            )
            if mode == "rne":
                values = b.vec_pack(
                    [b.cvt_f32_to_tf32(b.vec_extract(values, i)) for i in range(2)],
                    TF32,
                )
            else:
                values = b.bitcast(values, VectorType(TF32, 2))
            for slot in range(2):
                index = b.add(
                    row_base, b.add(b.mul(group, b.const_i32(2)), b.const_i32(slot))
                )
                bits = b.bitcast(b.vec_extract(values, slot), I32)
                b.global_store(prepared, index, bits, align=4)
            fragments.append(values)
        acc = b.mma(atom, *fragments, acc)
    for slot, index in enumerate(out_indices):
        b.global_store(d, index, b.vec_extract(acc, slot), align=4)
    b.ret()
    return b.kernel
