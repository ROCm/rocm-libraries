# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value-range proofs and public boundaries for bounded paged decode."""

from dataclasses import replace, asdict
from importlib import import_module

import pytest

from tests import test_paged_kv_decode as paged
from kernels.common._attention_paged_bounds import PagedBounds


class Interpreter(paged.AddressInterpreter):
    sub = staticmethod(lambda a, b: a - b)
    smax = staticmethod(max)
    cmp_ge = staticmethod(lambda a, b: a >= b)
    cmp_ne = staticmethod(lambda a, b: a != b)
    cmp_eq = staticmethod(lambda a, b: a == b)
    land = staticmethod(lambda a, b: a and b)

    def __init__(self):
        self.reads = []

    def masked_global_load(self, table, index, mask, other, **kwargs):
        if not mask:
            return other
        assert 0 <= index < len(table)
        self.reads.append(index)
        return table[index]


class Bounds(PagedBounds):
    def __init__(self, b, pages, cols, sliding_window=0):
        self.b, self.pages, self.cols = b, pages, cols
        self.sliding_window = sliding_window
        self.failed = False

    def mark(self, invalid):
        self.failed |= invalid


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("physical", [-2147483648, -1, 2, 2147483647])
def test_physical_is_sanitized_before_multiplication(page, physical):
    b = Interpreter()
    bounds = Bounds(b, 2, 1)
    got = bounds.offset(b, [physical], 0, 0, 0, 0, 1, 64, page, 32, 1)
    assert got == 0 and bounds.failed
    assert b.reads == [0]


@pytest.mark.parametrize("page", [1, 16, 32, 64])
def test_tail_does_not_read_poison_table(page):
    b = Interpreter()
    bounds = Bounds(b, 2, 1)
    got = bounds.offset(b, [], 0, 0, 0, 64, 1, 64, page, 32, 1)
    assert got == 0x7FFFFFFF and not bounds.failed and not b.reads


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("window", [1, 33, 65, 66])
def test_window_excluded_tokens_do_not_read_table(page, window):
    length, hd = 65, 64
    begin = max(0, length - window)
    cols = (length + page - 1) // page
    table = [-1 if (i + 1) * page <= begin else 0 for i in range(cols)]
    for token in range(length + 1):
        b = Interpreter()
        bounds = Bounds(b, 1, cols, sliding_window=window)
        tile, row = divmod(token, 32)
        got = bounds.offset(b, table, 0, 0, tile, row * hd, length, hd, page, 32, 1)
        live = begin <= token < length
        assert b.reads == ([token // page] if live else [])
        assert not bounds.failed
        assert got == ((token % page) * hd * 2 if live else 0x7FFFFFFF)


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("window", [0, 1, 33, 65, 66])
@pytest.mark.parametrize("physical", [-2147483648, -1, 1, 2147483647])
def test_window_boundary_keeps_consumed_ids_checked(page, window, physical):
    length, hd = 65, 64
    begin = max(0, length - window) if window else 0
    cols = (length + page - 1) // page
    table = [0] * cols
    table[begin // page] = physical
    for token in (begin - 1, begin, length):
        b = Interpreter()
        bounds = Bounds(b, 1, cols, sliding_window=window)
        tile, row = divmod(token, 32)
        got = bounds.offset(b, table, 0, 0, tile, row * hd, length, hd, page, 32, 1)
        live = begin <= token < length
        assert bounds.failed == live
        assert b.reads == ([token // page] if live else [])
        assert got == ((token % page) * hd * 2 if live else 0x7FFFFFFF)


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("window", [0, 1, 33])
def test_zero_length_window_does_not_read_table(page, window):
    b = Interpreter()
    bounds = Bounds(b, 1, 1, sliding_window=window)
    assert bounds.offset(b, [-1], 0, 0, 0, 0, 0, 64, page, 32, 1) == 0x7FFFFFFF
    assert not bounds.failed and not b.reads


def test_table_row_escape_is_masked():
    b = Interpreter()
    bounds = Bounds(b, 2, 1)
    assert bounds.physical([1], 0, 1, True) == 0
    assert bounds.failed and not b.reads


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("hd", [64, 128, 256])
def test_valid_offsets_fit_the_actual_pool(page, hd):
    b = Interpreter()
    bounds = Bounds(b, 2, 2)
    for token in range(min(2 * page, 65)):
        for dim in (0, hd - 8):
            tile, row = divmod(token, 32)
            got = bounds.offset(
                b,
                [1, 0],
                0,
                1,
                tile,
                row * hd + dim,
                min(2 * page, 65),
                hd,
                page,
                32,
                2,
            )
            expected = (
                ([1, 0][token // page] * page + token % page) * 2 * hd + hd + dim
            ) * 2
            assert got == expected and got + 16 <= 2 * page * 2 * hd * 2
    assert not bounds.failed


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
def test_candidate_pair_bounds_and_signature(arch, page):
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common.attention_paged_decode import paged_metadata_bounds
    from kernels.common.attention_unified import _3d_signature, _reduce_signature

    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="fp16",
        kv_block_size=page,
        algorithm="paged_decode_t32",
    )
    selected = dispatch_attention(req)
    assert selected.candidate.abi_version == "rocke-attention-paged/v2"
    assert selected.spec.kernel_spec.metadata_containment
    assert selected.spec.reduce_spec.metadata_containment
    problem, tensors = paged.inputs(page)
    problem = replace(problem, clamp_arch=arch)
    bounds = paged_metadata_bounds(problem, tensors, selected.spec)
    assert bounds["physical_pages"] == tensors[1].shape[0]
    assert bounds["k_span_bytes"] == tensors[1].numel() * 2
    assert selected.spec.with_num_kv_blocks(99).kernel_spec.metadata_containment
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    segment = module.build_unified_attention_3d_tiled(
        selected.spec.kernel_spec, arch=arch
    )
    reducer = module.build_unified_attention_reduce_tiled(
        selected.spec.reduce_spec, arch=arch
    )
    # Kernel/signature order is a launch ABI, not just a parameter count.
    sig = _3d_signature("fp16", metadata_containment=True)
    redsig = _reduce_signature("fp16", metadata_containment=True)
    assert [p.name for p in segment.params] == [p["name"] for p in sig]
    assert [p.name for p in reducer.params] == [p["name"] for p in redsig]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("metadata_containment", 1),
        ("use_decode_grid", False),
        ("use_wide_kv_load", True),
        ("use_wave_page_id_load", True),
        ("use_i64_kv_addr", True),
        ("tile_size_override", 64),
    ],
)
def test_bounded_spec_cannot_select_unchecked_loader(arch, field, value):
    from tests.parity.attention_paged_kv_emit import spec_for_index

    spec, _ = spec_for_index(66 if arch == "gfx942" else 82)
    with pytest.raises(ValueError):
        replace(spec, **{field: value})


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_mismatched_pair_and_real_pool_rejected(arch):
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common.attention_paged_decode import (
        has_paged_metadata_containment,
        paged_metadata_bounds,
    )
    from tests.parity.attention_paged_kv_emit import spec_for_index

    seg, _ = spec_for_index(66 if arch == "gfx942" else 82)
    m = import_module(f"kernels.{arch}.attention_tiled_3d")
    red = m.UnifiedAttentionReduceTiledSpec(64, 8, 2, "fp16", 8)
    spec = AttentionTuningSpec(
        "3d", arch, "tiled_3d", "llvm", "test", "test", seg, reduce_spec=red
    )
    with pytest.raises(ValueError, match="must match"):
        has_paged_metadata_containment(spec)
    red = replace(red, metadata_containment=True)
    spec = replace(spec, reduce_spec=red)
    problem, tensors = paged.inputs()
    tensors[1] = paged.Tensor((1 << 24, 1, 2, 64))
    tensors[2] = paged.Tensor(tensors[1].shape)
    with pytest.raises(ValueError, match="pool exceeds"):
        paged_metadata_bounds(problem, tensors, spec)


def test_bounded_rejects_overlapping_views():
    from dispatch.attention import AttentionRequest, dispatch_attention
    from kernels.common.attention_paged_decode import paged_metadata_bounds

    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        kv_block_size=1,
        algorithm="paged_decode_t32",
    )
    spec = dispatch_attention(req).spec
    problem, tensors = paged.inputs()
    tensors[3].pointer = tensors[0].pointer
    with pytest.raises(ValueError, match="must not overlap"):
        paged_metadata_bounds(problem, tensors, spec)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_native_bounded_rejects_unsupported_geometry(arch):
    engine = pytest.importorskip("rocke_engine")
    from tests.parity.attention_paged_kv_emit import spec_for_index

    spec, _ = spec_for_index(66 if arch == "gfx942" else 82)
    for field, value in (
        ("head_size", 96),
        ("num_kv_heads", 0),
        ("num_query_heads", 6),
        ("num_segments", 129),
    ):
        raw = dict(asdict(spec), **{field: value})
        with pytest.raises((ValueError, RuntimeError)):
            getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(raw, arch)
    for field, value in (
        ("kv_layout", "strided"),
        ("use_additive_bias", True),
        ("causal_top_left", True),
        ("right_bound", -1),
    ):
        raw = dict(asdict(spec), **{field: value})
        for suffix in ("lower_llvm", "serialize_ir", "verify"):
            fn = getattr(engine, f"{arch}_attention_tiled_3d_{suffix}")
            with pytest.raises((ValueError, RuntimeError)):
                fn(raw, arch)
    with pytest.raises(ValueError, match="gfx942/gfx950"):
        engine.paged_decode_reduce_lower_llvm(dict(asdict(spec)), "gfx1201")


def test_selected_workspace_includes_status():
    from dispatch.attention import AttentionRequest, dispatch_attention
    from dispatch.attention.common import _problem
    from kernels.common.attention_unified import attention_3d_workspace_nbytes

    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        kv_block_size=1,
        algorithm="paged_decode_t32",
    )
    selected = dispatch_attention(req).spec
    legacy = replace(
        selected,
        kernel_spec=replace(selected.kernel_spec, metadata_containment=False),
        reduce_spec=replace(selected.reduce_spec, metadata_containment=False),
    )
    problem = _problem(req)
    args = dict(problem=problem, arch="gfx950")
    bounded_bytes = attention_3d_workspace_nbytes(**args, tuning_spec=selected)
    legacy_bytes = attention_3d_workspace_nbytes(**args, tuning_spec=legacy)
    assert bounded_bytes - legacy_bytes == 3 * 2 * selected.kernel_spec.num_segments * 4


@pytest.mark.parametrize("stage", ["bind", "run"])
def test_retarget_cannot_strip_bounded_pair(stage, monkeypatch):
    from dispatch.attention import AttentionRequest, dispatch_attention
    from kernels.common import attention_unified as au
    from dispatch.attention import bindings

    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch="gfx950",
        kv_block_size=1,
        algorithm="paged_decode_t32",
        target_ctas=8,
    )
    bounded = dispatch_attention(req).spec
    legacy = replace(
        bounded,
        kernel_spec=replace(bounded.kernel_spec, metadata_containment=False),
        reduce_spec=replace(bounded.reduce_spec, metadata_containment=False),
    )
    adapted = paged._structural_paged_spec(bounded, retarget=lambda count: legacy)
    adapted.fp8_fnuz = bounded.fp8_fnuz
    problem, ts = paged.inputs()
    tensors = dict(
        zip(("q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table"), ts)
    )
    tensors["problem"] = problem
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx950")
    with pytest.raises(ValueError, match="preserve metadata containment"):
        if stage == "bind":
            bindings.bind_tuning_attention_torch(
                req, adapted, tensors, unsafe_skip_paged_value_validation=True
            )
        else:
            au.run_unified_attention_torch(
                **tensors,
                tuning_spec=adapted,
                softmax_scale=0.125,
                softcap=0,
                backend="3d",
            )
