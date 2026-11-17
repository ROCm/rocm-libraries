# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free bounded decode numerics through the selected rocKE runtime pipeline.

This cohort checks shuffled page mapping and ordered metadata recovery on raw
HIP buffers against independent float64 SDPA over quantized inputs. It does not
replace the Torch binding, prepared-execution, or graph-capture suites.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from dispatch.attention import AttentionRequest, dispatch_attention
from dispatch.attention.common import AttentionMaskType, attention_problem
from kernels.common import attention_unified as au
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_num_cus
from rocke.runtime.host_buffers import as_u8_buffer
from rocke.runtime.launcher import DeviceMem

_BATCH = 2
_QUERY_HEADS = 4
_KV_HEADS = 2
_CAPACITY = 193
_WINDOW = 33
_CANARY_BYTES = 32


@pytest.fixture(scope="module")
def gpu():
    arch = get_device_arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("needs a gfx942 or gfx950 GPU")
    num_cus = get_device_num_cus()
    if num_cus is None:
        pytest.fail(f"cannot read the CU count of the {arch} GPU")
    return arch, num_cus


def _request(arch, num_cus, dtype, page, dim, emit_lse):
    return AttentionRequest(
        batch=_BATCH,
        nhead_q=_QUERY_HEADS,
        nhead_k=_KV_HEADS,
        seqlen_q=1,
        seqlen_k=_CAPACITY,
        hdim_q=dim,
        hdim_v=dim,
        dtype=dtype,
        arch=arch,
        kv_block_size=page,
        num_cus=num_cus,
        target_ctas=8,
        algorithm="paged_decode_t32",
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        sliding_window=_WINDOW,
        emit_lse=emit_lse,
    )


def _encode(values, dtype):
    return (
        np.asarray(values, np.float16)
        if dtype == "fp16"
        else f32_to_bf16(np.asarray(values, np.float32))
    )


def _decode(values, dtype):
    return values.astype(np.float32) if dtype == "fp16" else bf16_to_f32(values)


def _copy_to_device(memory, values):
    values = np.ascontiguousarray(values)
    assert values.nbytes == memory.nbytes()
    Runtime().memcpy_h2d(memory.ptr(), as_u8_buffer(values), values.nbytes)


def _device(values):
    values = np.ascontiguousarray(values)
    memory = DeviceMem(values.nbytes)
    _copy_to_device(memory, values)
    return memory


def _reference(q, k, v, lengths):
    output = np.empty(q.shape, np.float64)
    lse = np.empty((_BATCH, _QUERY_HEADS), np.float64)
    for sequence, length in enumerate(lengths):
        result, normalizer = sdpa_reference(
            q[sequence : sequence + 1],
            k[sequence, :, :length].transpose(1, 0, 2),
            v[sequence, :, :length].transpose(1, 0, 2),
            top_left=False,
            right_bound=0,
            left_bound=_WINDOW - 1,
            q_offsets=[0, 1],
            kv_offsets=[0, int(length)],
        )
        output[sequence] = result[0]
        lse[sequence] = normalizer[0]
    return output, lse


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("dim", [64, 256])
@pytest.mark.parametrize("emit_lse", [False, True])
def test_cohort_selects_bounded_specs(arch, dtype, page, dim, emit_lse):
    selected = dispatch_attention(_request(arch, 32, dtype, page, dim, emit_lse))
    assert selected.candidate.name == "attention_paged_decode_t32"
    assert selected.spec.kernel_spec.metadata_containment
    assert selected.spec.reduce_spec.metadata_containment
    assert selected.spec.reduce_spec.emit_lse is emit_lse
    segment, reducer = selected.spec.build(arch)
    signatures = au._tiled_3d_signatures(
        arch, dtype, metadata_containment=True, emit_lse=emit_lse
    )
    for kernel, signature in zip((segment, reducer), signatures):
        assert [param.name for param in kernel.params] == [
            param["name"] for param in signature
        ]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_cohort_rejects_top_left_band_alias(arch):
    request = _request(arch, 32, "fp16", 16, 64, False)
    selected = dispatch_attention(request)
    top_left = replace(
        request,
        mask_type=AttentionMaskType.NO_MASK,
        right_bound=0,
        diagonal_alignment=0,
    )
    ok, why = selected.candidate.admits(top_left)
    assert not ok and "bottom-right" in why
    bottom_right = replace(top_left, diagonal_alignment=1)
    assert selected.candidate.admits(bottom_right)[0]


def test_reference_empty_and_windowed_rows():
    q = np.ones((_BATCH, _QUERY_HEADS, 64), np.float32)
    k = np.zeros((_BATCH, _KV_HEADS, _CAPACITY, 64), np.float32)
    v = np.ones(k.shape, np.float32)
    out, lse = _reference(q, k, v, [0, _CAPACITY])
    assert np.all(out[0] == 0)
    assert np.isneginf(lse[0]).all()
    np.testing.assert_allclose(out[1], 1.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(lse[1], np.log(_WINDOW))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("dim", [64, 256])
@pytest.mark.parametrize("emit_lse", [False, True])
def test_bounded_mapping_and_metadata_recovery(gpu, dtype, page, dim, emit_lse):
    arch, num_cus = gpu
    columns = math.ceil(_CAPACITY / page)
    physical_pages = _BATCH * columns + 2
    request = _request(arch, num_cus, dtype, page, dim, emit_lse)
    selected = dispatch_attention(request)
    assert selected.candidate.name == "attention_paged_decode_t32"
    spec = selected.spec.with_num_kv_blocks(physical_pages)
    assert (
        spec.kernel_spec.metadata_containment and spec.reduce_spec.metadata_containment
    )
    problem = attention_problem(request)
    segments = spec.kernel_spec.num_segments
    prepared = au._get_3d_pipeline(
        problem, spec.cache_key(), segments, arch=arch, tuning_spec=spec
    )

    rng = np.random.default_rng(37)
    q_bits = _encode(rng.normal(size=(_BATCH, _QUERY_HEADS, dim)), dtype)
    logical_k = _decode(
        _encode(rng.normal(size=(_BATCH, _KV_HEADS, _CAPACITY, dim)), dtype), dtype
    )
    logical_v = _decode(
        _encode(rng.normal(size=(_BATCH, _KV_HEADS, _CAPACITY, dim)), dtype), dtype
    )
    mapping = rng.permutation(physical_pages)[: _BATCH * columns].reshape(
        _BATCH, columns
    )
    table = mapping.astype(np.int32)
    pools = []
    for logical in (logical_k, logical_v):
        pool = np.full((physical_pages, page, _KV_HEADS, dim), np.nan, np.float32)
        # A sanitized invalid ID reads finite page-zero data, so the rejection
        # test cannot pass merely because its fallback page happened to be NaN.
        pool[0] = 0
        for sequence in range(_BATCH):
            for token in range(_CAPACITY):
                physical = table[sequence, token // page]
                pool[physical, token % page] = logical[sequence, :, token]
        pools.append(_device(_encode(pool, dtype)))
    lengths = np.array([_CAPACITY, 97], np.int32)
    expected, expected_lse = _reference(
        _decode(q_bits, dtype), logical_k, logical_v, lengths
    )
    table_memory = _device(table)
    lengths_memory = _device(lengths)
    offsets_memory = _device(np.arange(_BATCH + 1, dtype=np.int32))
    workspace = tuple(
        _device(np.full(item.nbytes(), 0xFF, np.uint8))
        for item in au._attention_3d_workspace_specs(
            problem, segments, device=None, metadata_containment=True
        )
    )
    partial, maximum, expsum, status = workspace
    output_bytes = np.prod(q_bits.shape) * 2
    initial = np.full(output_bytes + 2 * _CANARY_BYTES, 0x5A, np.uint8)
    initial[_CANARY_BYTES:-_CANARY_BYTES] = 0xFF
    output = _device(initial)
    lse_memory = _device(np.full((_BATCH, _QUERY_HEADS, 1), np.nan, np.float32))
    segment_values = {
        "segm_output_ptr": partial,
        "segm_max_ptr": maximum,
        "segm_expsum_ptr": expsum,
        "query_ptr": _device(q_bits),
        "key_cache_ptr": pools[0],
        "value_cache_ptr": pools[1],
        "sink_ptr": 0,
        "block_tables_ptr": table_memory,
        "seq_lens_ptr": lengths_memory,
        "alibi_slopes_ptr": 0,
        "qq_bias_ptr": 0,
        "query_start_len_ptr": offsets_memory,
        "scale": 1 / math.sqrt(dim),
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": _BATCH,
        "block_table_stride": columns,
        "qq_bias_stride_0": 0,
        "metadata_status_ptr": status,
        "physical_pages": physical_pages,
        "max_seqlen_k": _CAPACITY,
        "table_cols": columns,
        "k_span_bytes": pools[0].nbytes(),
        "v_span_bytes": pools[1].nbytes(),
        **au._runtime_kernarg_values(problem),
    }
    reduce_values = {
        "output_ptr": output.ptr() + _CANARY_BYTES,
        "segm_output_ptr": partial,
        "segm_max_ptr": maximum,
        "segm_expsum_ptr": expsum,
        "seq_lens_ptr": lengths_memory,
        "metadata_status_ptr": status,
        "num_seqs": _BATCH,
        **au._runtime_kernarg_values(problem),
    }
    if emit_lse:
        reduce_values["lse_ptr"] = lse_memory

    def run():
        # The final-stage fence completes both kernels before the host reads.
        au._launch_3d_pipeline(
            prepared, segment_values, reduce_values, 0, capturing=False
        )
        raw = np.empty(initial.shape, np.uint8)
        Runtime().memcpy_d2h(as_u8_buffer(raw), output.ptr(), raw.nbytes)
        assert np.all(raw[:_CANARY_BYTES] == 0x5A)
        assert np.all(raw[-_CANARY_BYTES:] == 0x5A)
        bits = (
            raw[_CANARY_BYTES:-_CANARY_BYTES].view(q_bits.dtype).reshape(q_bits.shape)
        )
        actual_lse = np.empty((_BATCH, _QUERY_HEADS, 1), np.float32)
        if emit_lse:
            Runtime().memcpy_d2h(
                as_u8_buffer(actual_lse), lse_memory.ptr(), actual_lse.nbytes
            )
        return _decode(bits, dtype), actual_lse[..., 0]

    def compare(failed=False):
        actual, actual_lse = run()
        first = 1 if failed else 0
        if failed:
            assert np.isnan(actual[0]).all()
            if emit_lse:
                assert np.isnan(actual_lse[0]).all()
        np.testing.assert_allclose(
            actual[first:], expected[first:], atol=0.01, rtol=0.02
        )
        if emit_lse:
            np.testing.assert_allclose(
                actual_lse[first:], expected_lse[first:], atol=0.002, rtol=0.002
            )

    compare()
    consumed = (_CAPACITY - 1) // page
    for invalid in (-1, physical_pages):
        changed = table.copy()
        changed[0, consumed] = invalid
        _copy_to_device(table_memory, changed)
        compare(failed=True)
        _copy_to_device(table_memory, table)
        compare()
    for invalid in (-1, _CAPACITY + 1):
        changed = lengths.copy()
        changed[0] = invalid
        _copy_to_device(lengths_memory, changed)
        compare(failed=True)
        _copy_to_device(lengths_memory, lengths)
        compare()
    offsets = np.arange(_BATCH + 1, dtype=np.int32)
    changed_offsets = offsets.copy()
    changed_offsets[0] = 1
    _copy_to_device(offsets_memory, changed_offsets)
    compare(failed=True)
    _copy_to_device(offsets_memory, offsets)
    compare()
    # Every page here has its first page wholly outside the live bottom-right
    # window: 193 - 33 = 160, beyond the end of a 64-token page.
    excluded = table.copy()
    excluded[0, 0] = physical_pages
    _copy_to_device(table_memory, excluded)
    compare()
    _copy_to_device(table_memory, table)
    empty = lengths.copy()
    empty[0] = 0
    _copy_to_device(lengths_memory, empty)
    actual, actual_lse = run()
    assert np.all(actual[0] == 0)
    np.testing.assert_allclose(actual[1], expected[1], atol=0.01, rtol=0.02)
    if emit_lse:
        assert np.isneginf(actual_lse[0]).all()
        np.testing.assert_allclose(
            actual_lse[1], expected_lse[1], atol=0.002, rtol=0.002
        )
    _copy_to_device(lengths_memory, lengths)
    compare()
