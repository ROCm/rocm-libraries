# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 tiled-2D QK paths read K from LDS in the type K was stored in.

With an fp8 KV cache and ``use_fp8_mfma_qk`` the kernel keeps K in LDS as raw
fp8 bytes and dequantizes it (``* k_scale``) in registers before the bf16 QK
MFMA. Every QK path has to read those bytes as fp8: a bf16 read of the fp8
buffer hands the MFMA pairs of fp8 bytes reinterpreted as bf16, with no
k_scale, and the scores come out wrong. The IR verifier does not catch such a
read, so this test walks the built kernel and checks every LDS load against
the element type of the buffer it reads.

Pure Python: it builds the kernel IR and needs no GPU.
"""

from __future__ import annotations

import pytest

from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)

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
