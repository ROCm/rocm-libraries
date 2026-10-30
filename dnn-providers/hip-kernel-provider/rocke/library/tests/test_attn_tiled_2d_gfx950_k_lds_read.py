# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 tiled-2D QK paths read K from LDS in the type K was stored in.

With an fp8 KV cache and ``use_fp8_mfma_qk`` the kernel keeps K in LDS as raw
fp8 bytes and dequantizes it (``* k_scale``) in registers before the bf16 QK
MFMA. Every QK path has to read those bytes as fp8: a bf16 read of the fp8
buffer hands the MFMA pairs of fp8 bytes reinterpreted as bf16, with no
k_scale, and the scores come out wrong. The IR verifier does not catch such a
read, so this test walks the built kernel and checks every LDS load against
the element type of the buffer it reads, and that every fp8 K read reaches the
QK MFMA only through a multiply by the ``k_scale`` kernel argument.

Pure Python: it builds the kernel IR and needs no GPU. The C++ engine tests
need the ``rocke_engine`` binding (``ROCKE_BUILD_PYBIND=ON``), which CI does
not build, so they skip there and run only in developer builds; the C++
evidence for these kernels in CI is the developer byte-identity run recorded
with the change.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import pytest

from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_BASE = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    tile_size=64,
)

_FP8_K_IN_LDS = dict(kv_storage_dtype="fp8e4m3", use_fp8_mfma_qk=True)

# One spec per QK path that can hold fp8 K in LDS.
_QK_PATHS = {
    "16x16": dict(),
    "32x32 non-transposed": dict(use_mfma_32x32=True, block_m_per_warp=32),
    "32x32 transposed": dict(
        use_mfma_32x32=True, use_transposed_qk_32x32=True, block_m_per_warp=32
    ),
}


def _walk(ops):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _walk(region.ops)


def _lds_loads(kernel):
    """``(buffer element type, loaded element type)`` of every LDS vector load
    that reads a buffer allocated in the kernel."""
    loads = []
    for op in _walk(kernel.body.ops):
        if op.name != "tile.smem_load_vN":
            continue
        buffer = op.operands[0]
        if buffer.op is None or buffer.op.name != "tile.smem_alloc":
            continue
        loads.append((buffer, buffer.type.elem.name, op.attrs["elem_type"]))
    return loads


def _build(**knobs):
    spec = UnifiedAttention2DTiledSpec(**_BASE, **knobs)
    return build_unified_attention_2d_tiled(spec, arch="gfx950")


@pytest.mark.parametrize("path", sorted(_QK_PATHS))
@pytest.mark.parametrize("fp8_k_in_lds", [False, True], ids=["bf16_k", "fp8_k"])
def test_lds_loads_read_the_buffer_element_type(path, fp8_k_in_lds):
    knobs = dict(_QK_PATHS[path], **(_FP8_K_IN_LDS if fp8_k_in_lds else {}))
    loads = _lds_loads(_build(**knobs))
    assert loads, "expected LDS loads in the kernel"
    wrong = sorted(
        {
            f"{buffer.name}: {stored} buffer read as {read}"
            for buffer, stored, read in loads
            if read != stored
        }
    )
    assert not wrong, f"{path}: LDS loads that reinterpret the buffer: {wrong}"


@pytest.mark.parametrize("path", sorted(_QK_PATHS))
def test_every_qk_path_reads_fp8_k_as_fp8(path):
    # The 32x32 non-transposed path used to read the fp8 K buffer as bf16.
    kernel = _build(**_QK_PATHS[path], **_FP8_K_IN_LDS)
    k_reads = [
        read
        for buffer, stored, read in _lds_loads(kernel)
        if stored == "fp8e4m3" and "Klds" in buffer.name
    ]
    assert k_reads, f"{path}: no QK read of the fp8 K buffer"
    assert set(k_reads) == {"fp8e4m3"}


def _users(kernel):
    """Map ``id(value)`` to the ops that take ``value`` as an operand."""
    users = defaultdict(list)
    for op in _walk(kernel.body.ops):
        for value in op.operands:
            users[id(value)].append(op)
    return users


def _forward(starts, users, stop):
    """The ops reached from the results of ``starts`` through operand uses,
    not continuing past an op for which ``stop(op)`` is true."""
    seen, reached, todo = set(), [], list(starts)
    while todo:
        op = todo.pop()
        for result in op.results:
            for user in users[id(result)]:
                if id(user) in seen:
                    continue
                seen.add(id(user))
                reached.append(user)
                if not stop(user):
                    todo.append(user)
    return reached


@pytest.mark.parametrize("path", sorted(_QK_PATHS))
def test_fp8_k_is_scaled_by_k_scale_before_the_mfma(path):
    # Reading the fp8 bytes as fp8 is not enough: each value has to be
    # multiplied by k_scale before it reaches the MFMA, or the scores are off
    # by that factor.
    kernel = _build(**_QK_PATHS[path], **_FP8_K_IN_LDS)
    assert "k_scale" in [param.name for param in kernel.params]
    users = _users(kernel)

    def is_k_scale(value):
        # A kernel parameter is a value with no defining op, named after it.
        return value.op is None and value.name == "%k_scale"

    def scales(op):
        return op.name == "arith.fmul" and any(map(is_k_scale, op.operands))

    k_reads = [
        op
        for op in _walk(kernel.body.ops)
        if op.name == "tile.smem_load_vN"
        and op.attrs["elem_type"] == "fp8e4m3"
        and "Klds" in op.operands[0].name
    ]
    assert k_reads, f"{path}: no QK read of the fp8 K buffer"
    for read in k_reads:
        reached = _forward([read], users, stop=scales)
        unscaled = [op.name for op in reached if op.name == "tile.mma"]
        assert not unscaled, f"{path}: fp8 K reaches the MFMA without * k_scale"
        multiplies = [op for op in reached if scales(op)]
        assert multiplies, f"{path}: fp8 K is never multiplied by k_scale"
        mfmas = _forward(multiplies, users, stop=lambda op: op.name == "tile.mma")
        assert any(
            op.name == "tile.mma" for op in mfmas
        ), f"{path}: scaled K does not reach an MFMA"


@pytest.mark.parametrize("path", sorted(_QK_PATHS))
def test_cpp_engine_emits_the_same_llvm_ir(path):
    # The C++ builder serves these kernels without Python; it must read and
    # scale fp8 K exactly as the Python builder does. Developer builds only:
    # CI does not build the rocke_engine binding.
    engine = pytest.importorskip("rocke_engine")
    spec = UnifiedAttention2DTiledSpec(**_BASE, **_QK_PATHS[path], **_FP8_K_IN_LDS)
    expected = _lower_kernel_to_llvm_python(
        build_unified_attention_2d_tiled(spec, arch="gfx950"), arch="gfx950"
    )
    assert (
        engine.gfx950_attention_tiled_2d_lower_llvm(asdict(spec), "gfx950") == expected
    )
