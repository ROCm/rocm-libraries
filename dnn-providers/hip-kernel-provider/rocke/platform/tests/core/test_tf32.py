# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Logical TF32 contracts and strict native parity, independent of GPU access."""
import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.dtypes import dtype_info, normalize_dtype
from rocke.core.ir import (
    F32,
    I32,
    TF32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
)
from rocke.core.ir_serialize import serialize, parse
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.gfx942.tf32_numerics import rne_bits
from rocke.instances.gfx942.tf32_mma_probe import (
    PREPARATIONS,
    Tf32MmaProbeSpec,
    build_tf32_mma_probe,
)


def test_dtype_identity():
    assert TF32 != F32 and TF32 != I32
    for alias in ("tf32", "xf32", " XF32 "):
        assert normalize_dtype(alias) == "tf32"
        assert dtype_to_ir_type(alias) == TF32
        assert dtype_info(alias).encoded_bits == 32


@pytest.mark.parametrize("m,k,acc", [(16, 8, 4), (32, 4, 16)])
def test_catalog_isolation(m, k, acc):
    catalog = ArchTarget.from_gfx("gfx942").mma
    atom = catalog.op_for_shape(
        family="mma", a_dtype="xf32", b_dtype="tf32", c_dtype="fp32", m=m, n=m, k=k
    )
    assert (atom.a_frag_len, atom.b_frag_len, atom.c_frag_len) == (2, 2, acc)
    assert atom.op_id == f"mfma_f32_{m}x{m}x{k}_xf32"
    full = catalog.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=m, n=m, k=k // 2
    )
    assert full.op_id == f"mfma_f32_{m}x{m}x{k//2}_f32"
    assert not catalog.has_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=m, n=m, k=k
    )
    for gfx in ("gfx950", "gfx1151", "gfx1250"):
        assert not any(op.a_dtype == "tf32" for op in ArchTarget.from_gfx(gfx).mma.ops)

    class Integers:
        const_i32 = staticmethod(int)
        mod = staticmethod(lambda x, y: x % y)
        div = staticmethod(lambda x, y: x // y)
        mul = staticmethod(lambda x, y: x * y)
        add = staticmethod(lambda x, y: x + y)

    calc = Integers()
    for layout, shape in (
        (atom.a_layout(), (m, k)),
        (atom.b_layout(), (k, m)),
        (atom.c_layout(), (m, m)),
    ):
        count = acc if shape == (m, m) else 2
        coords = [
            layout.coord(calc, lane, slot)
            for lane in range(64)
            for slot in range(count)
        ]
        assert len(set(coords)) == shape[0] * shape[1] == len(coords)
        assert set(coords) == {(i, j) for i in range(shape[0]) for j in range(shape[1])}


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("mode", PREPARATIONS)
def test_both_builders_and_lowerers(m, mode):
    native = pytest.importorskip("rocke_engine")
    kernel = build_tf32_mma_probe(Tf32MmaProbeSpec(m, mode))
    ir = serialize(kernel)
    assert serialize(parse(ir)) == ir
    assert native.tf32_mma_probe_serialize_ir(m, mode) == ir
    for flavor in ("llvm20", "llvm22", "llvm23"):
        ll = _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)
        assert native.lower_serialized_ir(ir, arch="gfx942", flavor=flavor) == ll
        if mode != "fp32":
            assert f"llvm.amdgcn.mfma.f32.{m}x{m}x{128//m}.xf32" in ll
            assert "sitofp" not in ll and "uitofp" not in ll
    if mode != "fp32":
        assert "__builtin_bit_cast(f32x2," in lower_kernel_to_hip(kernel, arch="gfx942")


@pytest.mark.parametrize("dtype,n", [(F32, 2), (I32, 2), (TF32, 1), (TF32, 4)])
def test_mma_requires_logical_tf32(dtype, n):
    b = IRBuilder("bad")
    a = b.param("a", VectorType(dtype, n))
    c = b.param("c", VectorType(F32, 4))
    with pytest.raises(ValueError, match="XF32 MMA requires"):
        b.mma("mfma_f32_16x16x8_xf32", a, a, c)


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1151", "gfx1250"])
def test_rejects_unsupported_target(gfx):
    kernel = build_tf32_mma_probe(Tf32MmaProbeSpec())
    with pytest.raises(ValueError, match="gfx942"):
        _lower_kernel_to_llvm_python(kernel, arch=gfx)
    with pytest.raises(ValueError, match="gfx942"):
        lower_kernel_to_hip(kernel, arch=gfx)
    native = pytest.importorskip("rocke_engine")
    with pytest.raises(Exception):
        native.lower_serialized_ir(serialize(kernel), arch=gfx)


def test_arithmetic_rejected_after_serialization():
    b = IRBuilder("bad_arithmetic")
    v = b.param("x", TF32)
    b.add(v, v)
    b.ret()
    with pytest.raises(ValueError, match="TF32 arithmetic"):
        _lower_kernel_to_llvm_python(b.kernel)
    native = pytest.importorskip("rocke_engine")
    with pytest.raises(Exception):
        native.lower_serialized_ir(serialize(b.kernel))


def test_transport():
    b = IRBuilder("transport")
    p = b.param("p", PtrType(TF32, "global"))
    i = b.const_i32(0)
    v = b.global_load(p, i, TF32, align=4)
    pair = b.vec_pack([v, v], TF32)
    v = b.vec_extract(pair, 1)
    b.global_store(p, i, v, align=4)
    b.ret()
    ir = serialize(b.kernel)
    assert serialize(parse(ir)) == ir
    ll = _lower_kernel_to_llvm_python(b.kernel)
    assert "load i32" in ll and "store i32" in ll
    native = pytest.importorskip("rocke_engine")
    assert native.lower_serialized_ir(ir) == ll


def test_rne_reference_edges():
    inputs = np.array(
        [
            0,
            0x80000000,
            0x3F801000,
            0x3F803000,
            0x3F801001,
            0xBF801001,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFF800001,
            0x1000,
            0x1001,
            0x7FFFFF,
        ],
        np.uint32,
    )
    expected = np.array(
        [
            0,
            0x80000000,
            0x3F800000,
            0x3F804000,
            0x3F802000,
            0xBF802000,
            0x7F800000,
            0xFF800000,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0,
            0x2000,
            0x800000,
        ],
        np.uint32,
    )
    assert np.array_equal(rne_bits(inputs), expected)
    assert np.array_equal(rne_bits(expected), expected)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_lds_and_loop_transport(n):
    b = IRBuilder("tf32_lds_loop")
    p = b.param("p", PtrType(TF32, "global"))
    zero = b.const_i32(0)
    one = b.const_i32(1)
    v = b.global_load(p, zero, TF32, align=4)
    pair = b.vec_pack([v] * n, TF32)
    shared = b.smem_alloc(TF32, (64,))
    b.smem_store_vN(shared, [zero], v if n == 1 else pair, n)
    loaded = b.smem_load_vN(shared, zero, dtype=TF32, n=n)
    loop = b.scf_for_iter(zero, one, one, [("payload", loaded)])
    with loop as (_, [payload]):
        b.scf_yield(payload)
    b.global_store(p, zero, b.vec_extract(loop.results[0], 0), align=4)
    b.ret()
    ir = serialize(b.kernel)
    assert serialize(parse(ir)) == ir
    ll = _lower_kernel_to_llvm_python(b.kernel)
    assert "[256 x i8]" in ll
    assert f"phi <{n} x i32>" in ll
    assert "int smem" in lower_kernel_to_hip(
        b.kernel
    ) or "int shared" in lower_kernel_to_hip(b.kernel)
    native = pytest.importorskip("rocke_engine")
    assert native.lower_serialized_ir(ir) == ll


def test_fp32_atom_rejects_tf32_carrier():
    b = IRBuilder("wrong_atom")
    a = b.param("a", TF32)
    c = b.param("c", VectorType(F32, 4))
    with pytest.raises(ValueError, match="TF32 operands require"):
        b.mma("mfma_f32_16x16x4_f32", a, a, c)


def test_parsed_mma_rejects_wrong_operands():
    native = pytest.importorskip("rocke_engine")
    for bad in ("vec<i32x2>", "vec<f32x2>", "vec<tf32x4>"):
        ir = serialize(build_tf32_mma_probe(Tf32MmaProbeSpec())).replace(
            "vec<tf32x2>", bad
        )
        with pytest.raises(ValueError, match="XF32 MMA requires"):
            _lower_kernel_to_llvm_python(parse(ir), arch="gfx942")
        with pytest.raises(Exception):
            native.lower_serialized_ir(ir, arch="gfx942")


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("mode", PREPARATIONS)
def test_recipe_replay_preserves_ir(m, mode, monkeypatch):
    from rocke.portable_ir.src.recording_builder import record_kernel
    from rocke.portable_ir.src import online, recipe_bundle

    spec = Tf32MmaProbeSpec(m, mode)
    recorded, recipe = record_kernel(lambda: build_tf32_mma_probe(spec))
    assert serialize(recorded) == serialize(build_tf32_mma_probe(spec))
    for flavor in ("llvm20", "llvm22", "llvm23"):
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
        replayed, _ = online.recipe_cbor_to_llvm(
            recipe_bundle.cbor_encode(recipe), arch="gfx942"
        )
        assert replayed == _lower_kernel_to_llvm_python(
            recorded, arch="gfx942", llvm_flavor=flavor
        )


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_global_vector_store(n):
    b = IRBuilder("tf32_vector_store")
    p = b.param("p", PtrType(TF32, "global"))
    i = b.const_i32(0)
    v = b.global_load(p, i, TF32, align=4)
    values = b.vec_pack([v] * n, TF32)
    if n == 16:
        with pytest.raises(ValueError, match="n=16 not supported for tf32"):
            b.global_store_vN(p, i, values, n)
        return
    b.global_store_vN(p, i, values, n)
    b.ret()
    ir = serialize(b.kernel)
    assert serialize(parse(ir)) == ir
    native = pytest.importorskip("rocke_engine")
    for flavor in ("llvm20", "llvm22", "llvm23"):
        ll = _lower_kernel_to_llvm_python(b.kernel, llvm_flavor=flavor)
        assert f"store <{n} x i32>" in ll
        assert f"align {n * 4}" in ll
        assert native.lower_serialized_ir(ir, flavor=flavor) == ll


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("role", ["accumulator", "result"])
@pytest.mark.parametrize("bad", ["integer", "tf32", "width"])
def test_parsed_mma_rejects_wrong_accumulator_and_result(m, role, bad):
    native = pytest.importorskip("rocke_engine")
    kernel = build_tf32_mma_probe(Tf32MmaProbeSpec(m))
    mma = next(op for op in kernel.body.ops if op.name == "tile.mma")
    value = mma.operands[2] if role == "accumulator" else mma.result
    count = 4 if m == 16 else 16
    value.type = VectorType(
        {"integer": I32, "tf32": TF32, "width": F32}[bad],
        count + 1 if bad == "width" else count,
    )
    ir = serialize(kernel)
    message = "XF32 MMA requires" if role == "accumulator" else "XF32 MMA result"
    with pytest.raises(ValueError, match=message):
        _lower_kernel_to_llvm_python(parse(ir), arch="gfx942")
    with pytest.raises(Exception, match=message):
        native.lower_serialized_ir(ir, arch="gfx942")


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_numeric_harness_auto_flavor(flavor, monkeypatch, tmp_path):
    pytest.importorskip("rocke_engine")
    from rocke.examples.gfx942 import tf32_numerics
    from rocke.runtime import comgr, hip_module

    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    monkeypatch.setattr(tf32_numerics, "_resolve_llvm_flavor", lambda: flavor)
    monkeypatch.setattr(hip_module, "get_device_arch", lambda: "gfx942")
    expected = _lower_kernel_to_llvm_python(
        build_tf32_mma_probe(Tf32MmaProbeSpec()),
        arch="gfx942",
        llvm_flavor=flavor,
    )

    class ReachedCompiler(Exception):
        pass

    def check_compiler_input(ll, *, isa):
        assert ll == expected
        raise ReachedCompiler

    # Exercise the actual builders/lowerers, then stop before GPU compilation.
    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", check_compiler_input)
    with pytest.raises(ReachedCompiler):
        tf32_numerics.run(tmp_path, backend="cpp", shapes=(16,))
