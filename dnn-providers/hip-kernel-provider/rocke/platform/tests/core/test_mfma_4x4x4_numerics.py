# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lane-layout and numerics check for the 4x4x4 MFMA atoms (f16 and bf16).

One wave issues a single ``mfma_f32_4x4x4_{f16,bf16}``. The test pins the
16-batch lane map the direct-conv 4c kernel relies on:

* lane ``l`` belongs to batch ``blk = l // 4`` and carries ``i = l % 4``;
* operand ``a`` of lane ``l`` is row ``i`` of ``A_blk`` (K = 0..3);
* operand ``b`` of lane ``l`` is column ``i`` of ``B_blk`` (K = 0..3);
* result slot ``m`` of lane ``l`` is ``D_blk[m][i]``.

The IR / lowering checks run everywhere; the on-device numeric check runs only
where a gfx942 / gfx950 GPU is visible (numpy only, torch is not required).

It also pins, for these atoms and the f64 4x4x4 atom, that the MMA catalog
lists each atom exactly on the targets whose compiler selects its intrinsic.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.arch import domain as arch_domain
from rocke.core.ir import BF16, F16, F32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.tests.engines import (
    assert_engine_rejects,
    assert_engine_rejects_hip,
    available_engines,
    lower_with_engine,
)

_INTRIN = {
    "f16": "@llvm.amdgcn.mfma.f32.4x4x4f16(",
    "bf16": "@llvm.amdgcn.mfma.f32.4x4x4bf16.1k(",
}

_BUILTIN = {
    "f16": "__builtin_amdgcn_mfma_f32_4x4x4f16(",
    "bf16": "__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(",
}

# The targets whose MMA catalog lists each batched 4x4x4 atom: gfx908 has the
# f16 atom but neither the bf16 _1k nor the f64 form, and no RDNA target has an
# MFMA.
_LISTED_ON = {
    "mfma_f32_4x4x4_f16": ["gfx908", "gfx90a", "gfx942", "gfx950"],
    "mfma_f32_4x4x4_bf16": ["gfx90a", "gfx942", "gfx950"],
    "mfma_f64_4x4x4_f64": ["gfx90a", "gfx942", "gfx950"],
}

# Each atom's intrinsic, as the declaration table and the arch domain key it.
_INTRINSIC_KEY = {
    "mfma_f32_4x4x4_f16": "mfma.f32.4x4x4f16",
    "mfma_f32_4x4x4_bf16": "mfma.f32.4x4x4bf16.1k",
    "mfma_f64_4x4x4_f64": "mfma.f64.4x4x4f64",
}


def _op_id(dtype: str) -> str:
    """The op_id of the f32-accumulating 4x4x4 atom of ``dtype``."""
    return f"mfma_f32_4x4x4_{dtype}"


def _listing_targets(op_id: str) -> list[str]:
    """The targets whose MMA catalog lists the atom ``op_id``."""
    return [
        arch for arch in known_arches() if ArchTarget.from_gfx(arch).mma.by_op_id(op_id)
    ]


def _selecting_targets(op_id: str) -> list[str]:
    """The targets on which the compiler of every committed LLVM flavor links and
    selects the intrinsic of the atom ``op_id``, as the measured arch domain
    records it."""
    columns = [arch_domain.load(flavor) for flavor in arch_domain.committed_flavors()]
    return [
        arch
        for arch in known_arches()
        if all(
            (cell := column.lookup(_INTRINSIC_KEY[op_id], arch)) is not None
            and cell.status == "ok"
            for column in columns
        )
    ]


def _named_for_its_atom(kernel):
    """``kernel`` with its ``tile.mma`` op renamed ``tile.<op_id>``, the other
    name the Python lowerers lower an MMA op under. The C++ engine's IR has no
    such op: its parser refuses the name."""
    for op in kernel.body.ops:
        if op.name == "tile.mma":
            op.name = f"tile.{op.attrs.pop('op_id')}"
    return kernel


def _catalog_refusal(dtype: str, arch: str) -> str:
    return (
        f"tile.{_op_id(dtype)} is not available on {arch}: "
        "its MMA catalog does not list it"
    )


def _build(dtype: str):
    elem = BF16 if dtype == "bf16" else F16
    b = IRBuilder(f"mfma_4x4x4_{dtype}_probe")
    b.kernel.attrs["max_workgroup_size"] = 64
    A = b.param("A", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    B = b.param("B", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    lane = b.thread_id_x()
    idx = b.mul(lane, b.const_i32(4))
    a = b.global_load_vN(A, idx, elem, 4)
    bv = b.global_load_vN(B, idx, elem, 4)
    zero = b.zero_vec_f32(4)
    if dtype == "bf16":
        acc = b.mfma_f32_4x4x4_bf16(a, bv, zero)
    else:
        acc = b.mfma_f32_4x4x4_f16(a, bv, zero)
    b.global_store_vN(D, idx, acc, 4)
    b.ret()
    return b.kernel


def _to_bits(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        # round-to-nearest-even f32 -> bf16 bit pattern
        u = x.astype(np.float32).view(np.uint32)
        u = u + 0x7FFF + ((u >> 16) & 1)
        return (u >> 16).astype(np.uint16)
    return x.astype(np.float16).view(np.uint16)


def _from_bits(u: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return (u.astype(np.uint32) << 16).view(np.float32)
    return u.view(np.float16).astype(np.float32)


def _reference(a_lanes: np.ndarray, b_lanes: np.ndarray) -> np.ndarray:
    """Per-lane expected result for the documented 4x4x4 lane map."""
    out = np.zeros((64, 4), dtype=np.float32)
    for blk in range(16):
        A = a_lanes[blk * 4 : blk * 4 + 4]  # A[i][k]
        Bt = b_lanes[blk * 4 : blk * 4 + 4]  # Bt[j][k] == B[k][j]
        Dm = A @ Bt.T  # D[i][j]
        for j in range(4):
            out[blk * 4 + j, :] = Dm[:, j]
    return out


@pytest.mark.parametrize("op_id", sorted(_LISTED_ON))
def test_catalog_lists_the_atom_exactly_where_the_compiler_selects_it(op_id):
    assert _listing_targets(op_id) == _LISTED_ON[op_id] == _selecting_targets(op_id)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_lowering_emits_expected_intrinsic(dtype):
    ll = lower_kernel_to_llvm(_build(dtype), arch="gfx950")
    assert _INTRIN[dtype] in ll
    if dtype == "bf16":
        assert "bitcast <4 x bfloat>" in ll
        assert "declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4bf16.1k(<4 x i16>" in ll
    # Each engine lowers the atom where the catalog lists it, and refuses it
    # with the catalog gate's text on every other target, under either name.
    listed_on = _LISTED_ON[_op_id(dtype)]
    for arch in known_arches():
        for engine in available_engines():
            if arch in listed_on:
                text = lower_with_engine(
                    _build(dtype), arch, engine=engine, llvm_flavor="llvm22"
                )
                assert _INTRIN[dtype] in text, (arch, engine)
            else:
                assert_engine_rejects(
                    _build(dtype),
                    arch,
                    engine=engine,
                    error_type=NotImplementedError,
                    message=_catalog_refusal(dtype, arch),
                    llvm_flavor="llvm22",
                )
        if arch not in listed_on:
            assert_engine_rejects(
                _named_for_its_atom(_build(dtype)),
                arch,
                engine="python",
                error_type=NotImplementedError,
                message=_catalog_refusal(dtype, arch),
                llvm_flavor="llvm22",
            )


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_hip_lowering_emits_expected_builtin(dtype):
    from rocke.core.lower_hip import lower_kernel_to_hip

    assert _BUILTIN[dtype] in lower_kernel_to_hip(_build(dtype))
    # The HIP lowering of each engine applies the same catalog gate. Both
    # engines' HIP text for the listing targets is pinned by
    # test_mma_operand_register_type.py (test_hip_batched_4x4x4_mfma_lowers_in_both_engines).
    for arch in known_arches():
        if arch in _LISTED_ON[_op_id(dtype)]:
            assert _BUILTIN[dtype] in lower_kernel_to_hip(_build(dtype), arch=arch)
            continue
        for engine in available_engines():
            assert_engine_rejects_hip(
                _build(dtype),
                arch,
                engine=engine,
                error_type=NotImplementedError,
                message=_catalog_refusal(dtype, arch),
            )
        assert_engine_rejects_hip(
            _named_for_its_atom(_build(dtype)),
            arch,
            engine="python",
            error_type=NotImplementedError,
            message=_catalog_refusal(dtype, arch),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_numerics_on_device(dtype):
    try:
        from rocke import compile_kernel
        from rocke.runtime import KernelLauncher, LaunchConfig, Runtime

        rt = Runtime()
    except (ImportError, OSError, RuntimeError) as e:  # pragma: no cover
        pytest.skip(f"no HIP runtime: {e}")
    try:
        from rocke.runtime import get_device_info

        arch = get_device_info().base_arch
    except (ImportError, OSError, RuntimeError) as e:  # pragma: no cover
        pytest.skip(f"no visible GPU: {e}")
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"4x4x4 MFMA check needs gfx942/gfx950 (got {arch})")
    art = compile_kernel(_build(dtype), arch=arch, capture_ir_text=False)

    rng = np.random.default_rng(1234)
    a = _from_bits(_to_bits(rng.uniform(-2, 2, (64, 4)), dtype), dtype)
    bm = _from_bits(_to_bits(rng.uniform(-2, 2, (64, 4)), dtype), dtype)
    a_bits = np.ascontiguousarray(_to_bits(a, dtype))
    b_bits = np.ascontiguousarray(_to_bits(bm, dtype))
    out = np.zeros((64, 4), dtype=np.float32)

    def u8(x):
        return (ctypes.c_uint8 * x.nbytes).from_address(x.ctypes.data)

    dA, dB, dD = rt.alloc(a_bits.nbytes), rt.alloc(b_bits.nbytes), rt.alloc(out.nbytes)
    rt.memcpy_h2d(dA, u8(a_bits), a_bits.nbytes)
    rt.memcpy_h2d(dB, u8(b_bits), b_bits.nbytes)
    pt = "bf16" if dtype == "bf16" else "f16"
    sig = [
        {"name": "A", "type": f"ptr<{pt}, global>", "size_bytes": 8},
        {"name": "B", "type": f"ptr<{pt}, global>", "size_bytes": 8},
        {"name": "D", "type": "ptr<f32, global>", "size_bytes": 8},
    ]
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
    )
    launcher(
        {"A": dA, "B": dB, "D": dD},
        config=LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)),
    )
    rt.sync()
    rt.memcpy_d2h(u8(out), dD, out.nbytes)
    for ptr in (dA, dB, dD):
        rt.free(ptr)
    ref = _reference(a, bm)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
