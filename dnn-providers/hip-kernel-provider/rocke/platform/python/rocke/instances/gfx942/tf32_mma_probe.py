# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One wave per dense atom, with explicit FP32 input-preparation variants."""

from dataclasses import dataclass

from rocke.core.arch import ArchTarget
from rocke.core.arch.target import _frag_info
from rocke.core.ir import F32, I32, TF32, IRBuilder, PtrType, VectorType
from rocke.core.storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from rocke.helpers.tiling import (
    Fragment,
    load_fragment,
    make_matrix_fragment_desc,
    make_mma_accumulator_desc,
    make_tensor_desc,
    make_window,
    store_fragment,
)

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
    full = None
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
    # Pairwise agreement does not prove the fixed instruction ABI or coordinates.
    width = 4 if m == 16 else 16

    def valid(selected, dtype, source_width, atom_k, suffix):
        expected_id = f"mfma_f32_{m}x{m}x{atom_k}_{suffix}"
        return (
            selected is not None
            and selected.op_id == expected_id
            and selected.family == "mma"
            and selected.shape == (m, m, atom_k)
            and type(selected.wave_size) is int
            and selected.wave_size == 64
            and all(
                src.dtype == dtype
                and type(src.frag_len) is int
                and src.frag_len == source_width
                for src in selected.srcs[:2]
            )
            and all(
                desc.dtype == "fp32"
                and type(desc.frag_len) is int
                and desc.frag_len == width
                for desc in (selected.srcs[2], selected.dst)
            )
        )

    expected_id = f"mfma_f32_{m}x{m}x{k}_xf32"
    registered = _frag_info(expected_id)
    if (
        not valid(atom, "tf32", 2, k, "xf32")
        or (mode == "fp32" and not valid(full, "fp32", 1, k // 2, "f32"))
        or any(
            layout is None
            or type(layout.frag_len) is not int
            or layout.frag_len != width
            or type(layout.wave_size) is not int
            or layout.wave_size != 64
            or layout.fn is None
            or layout.fn is not expected.fn
            for layout, expected in (
                (atom.srcs[2].layout, registered.srcs[2]),
                (atom.dst.layout, registered.dst),
            )
        )
    ):
        raise ValueError(
            "TF32 probe requires the fixed gfx942 instruction and layout contract"
        )
    c_desc = make_mma_accumulator_desc(atom, target="gfx942")
    d_desc = make_mma_accumulator_desc(atom, target="gfx942", result=True)
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
    input_dtype = TF32 if mode == "prepacked" else F32
    inputs = make_tensor_desc((None, m, k), (m * k, k, 1), input_dtype)
    diagnostics = make_tensor_desc((None, m, k), (m * k, k, 1), I32)
    outputs = make_tensor_desc((None, m, m), (m * m, m, 1), F32)
    output_window = make_window(outputs, (batch, 0, 0))
    acc = load_fragment(b, cc, output_window, c_desc, lane, alignment_bytes=4).value
    if mode == "fp32":
        scalar_layout = MatrixFragmentLayout(
            FragmentPacking(BitPacking(32), 1, 32, 1), 1, 64 // m, m
        )
        scalar_desc = make_matrix_fragment_desc(scalar_layout)
        for step in range(2):
            origin = (batch, 0, step * (64 // m))
            window = make_window(inputs, origin)
            af = load_fragment(b, a, window, scalar_desc, lane, alignment_bytes=4)
            bf = load_fragment(b, bb, window, scalar_desc, lane, alignment_bytes=4)
            av, bv = b.vec_extract(af.value, 0), b.vec_extract(bf.value, 0)
            # Preserve alias-visible A/B reads, PA/PB writes, then the instruction.
            bits = b.vector_splat(b.bitcast(av, I32), 1)
            store_fragment(
                b,
                pa,
                make_window(diagnostics, origin),
                Fragment(scalar_desc, I32, bits),
                lane,
                alignment_bytes=4,
            )
            bits = b.vector_splat(b.bitcast(bv, I32), 1)
            store_fragment(
                b,
                pb,
                make_window(diagnostics, origin),
                Fragment(scalar_desc, I32, bits),
                lane,
                alignment_bytes=4,
            )
            acc = b.mma(full, av, bv, acc)
    else:
        layout = MatrixFragmentLayout(
            FragmentPacking(BitPacking(32), 2, 32, 2), 2, 64 // m, m
        )
        desc = make_matrix_fragment_desc(layout)
        window = make_window(inputs, (batch, 0, 0))
        diagnostic_window = make_window(diagnostics, (batch, 0, 0))
        fragments = []
        for ptr, prepared in ((a, pa), (bb, pb)):
            values = load_fragment(b, ptr, window, desc, lane, alignment_bytes=4).value
            if mode == "carrier":
                values = b.bitcast(values, VectorType(I32, 2))
            if mode == "rne":
                values = b.vec_pack(
                    [b.cvt_f32_to_tf32(b.vec_extract(values, i)) for i in range(2)],
                    TF32,
                )
            else:
                values = b.bitcast(values, VectorType(TF32, 2))
            bits = b.bitcast(values, VectorType(I32, 2))
            store_fragment(
                b,
                prepared,
                diagnostic_window,
                Fragment(desc, I32, bits),
                lane,
                alignment_bytes=4,
            )
            fragments.append(values)
        acc = b.mma(atom, *fragments, acc)
    store_fragment(
        b, d, output_window, Fragment(d_desc, F32, acc), lane, alignment_bytes=4
    )
    b.ret()
    return b.kernel
