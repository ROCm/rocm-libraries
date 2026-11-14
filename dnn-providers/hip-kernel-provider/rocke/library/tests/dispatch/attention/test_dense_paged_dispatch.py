# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for paged K/V (``layout="paged"``) on the dense candidates (host
only)."""

import dataclasses
from types import SimpleNamespace
from unittest import mock

import pytest

from ..._dense_paged_stand_ins import CacheStandIn, I32DeviceTensor, pinned
from dispatch.attention import (
    AttentionRequest,
    dispatch_attention,
)

_PAGED_SPEC_IDS = {
    "gfx942": ("gfx942_dense",),
    "gfx950": ("gfx950_dense_grid", "gfx950_dense_qbminor"),
}
_CASES = [(arch, sid) for arch, ids in _PAGED_SPEC_IDS.items() for sid in ids]
_SQ, _HQ, _HKV, _D, _BS = 2048, 16, 4, 128, 16


def _request(arch, **overrides):
    fields = dict(
        batch=1,
        nhead_q=_HQ,
        nhead_k=_HKV,
        seqlen_q=_SQ,
        seqlen_k=_SQ,
        hdim_q=_D,
        hdim_v=_D,
        arch=arch,
        mask_type=1,
        dtype="fp16",
        sliding_window=1024,
        layout="paged",
        kv_block_size=_BS,
    )
    fields.update(overrides)
    return AttentionRequest(**fields)


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "overrides",
    [
        dict(),
        dict(sliding_window=0),
        dict(hdim_q=64, hdim_v=64),
        dict(seqlen_q=512),
        dict(seqlen_q=4096, sliding_window=0),
    ],
    ids=["swa", "plain_causal", "d64", "sq_lt_skv", "sq_gt_skv"],
)
def test_paged_request_builds_a_paged_spec(arch, spec_id, overrides):
    req = _request(arch, **overrides)
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged and spec.block_size == _BS
    assert spec.num_kv_blocks == _SQ // _BS  # the smallest cache, until binding
    assert not spec.persistent


@pytest.mark.parametrize(
    "spec_id", ["gfx950_dense_persist", "gfx950_dense_persist_widedma"]
)
def test_persistent_variants_decline_paged(spec_id):
    with pytest.raises(ValueError, match="layout_paged"):
        dispatch_attention(pinned(_request("gfx950"), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "overrides,error",
    [
        (dict(batch=2), "batch>1"),
        (dict(mask_type=0, sliding_window=0), "causal-only"),
        (dict(seqlen_q=1000, seqlen_k=1000, sliding_window=0), "whole KV tiles"),
        (dict(kv_block_size=8), "must be >= 16"),
        (
            dict(mask_type=2, seqlen_q=1024, sliding_window=0),
            "causal_bottom_right is not supported with paged",
        ),
    ],
    ids=["multi_seq", "non_causal", "partial_kv_tile", "page_8", "bottom_right"],
)
def test_paged_outside_the_cohort_is_refused(arch, spec_id, overrides, error):
    with pytest.raises(ValueError, match=error):
        dispatch_attention(pinned(_request(arch, **overrides), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_paged_ragged_query_tail_follows_the_contiguous_kernel(arch, spec_id):
    req = _request(arch, seqlen_q=1088, seqlen_k=1088, sliding_window=0)
    if arch == "gfx942":  # no partial-tile (ragged) body on gfx942
        with pytest.raises(ValueError, match="seqlen_q 1088 is not a multiple of"):
            dispatch_attention(pinned(req, spec_id))
    else:
        spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
        assert spec.paged and spec.ragged


_SMALL = dict(seqlen_q=512, seqlen_k=512, nhead_q=8, nhead_k=2, num_cus=256)
_D64_BF16 = dict(hdim_q=64, hdim_v=64, dtype="bf16")


def _tiles(arch, spec_id, **overrides):
    """``(block_m, block_n, window_wave_skip)`` dispatch picks for a small paged
    request and for its contiguous twin."""
    req = _request(arch, **{**_SMALL, **overrides})
    tiles = []
    for r in (req, dataclasses.replace(req, layout="auto")):
        s = dispatch_attention(pinned(r, spec_id)).spec.kernel_spec
        tiles.append((s.block_m, s.block_n, getattr(s, "window_wave_skip", False)))
    return tiles


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "overrides",
    [
        dict(sliding_window=0),
        dict(sliding_window=256),
        dict(sliding_window=256, kv_block_size=32, **_D64_BF16),
    ],
    ids=["causal", "swa", "swa_d64_bf16"],
)
def test_paged_takes_the_contiguous_tile_when_its_pages_fit(arch, spec_id, overrides):
    paged, contiguous = _tiles(arch, spec_id, **overrides)
    assert paged == contiguous
    if arch == "gfx942" or overrides["sliding_window"]:
        assert paged[0] < 256  # a smaller tile, not the default one


@pytest.mark.parametrize(
    "arch,spec_id,overrides",
    [
        ("gfx942", "gfx942_dense", dict(sliding_window=256, kv_block_size=64)),
        (
            "gfx942",
            "gfx942_dense",
            dict(sliding_window=256, kv_block_size=16, **_D64_BF16),
        ),
    ],
    ids=["128x32_page64", "64x64_page16"],
)
def test_paged_keeps_the_default_tile_when_its_pages_do_not_fit(
    arch, spec_id, overrides
):
    paged, contiguous = _tiles(arch, spec_id, **overrides)
    assert paged == (256, 64, False)
    assert contiguous[:2] != (256, 64)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_layout_is_matched_like_the_capability(arch, spec_id):
    """The request's layout spelling is normalized as in its feature set, so the
    candidate that matched ``layout_paged`` also builds the paged spec."""
    req = _request(arch, layout=" Paged ")
    assert dispatch_attention(pinned(req, spec_id)).spec.kernel_spec.paged


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize("block_size", [0, 16.9, "16"])
def test_malformed_kv_block_size_is_refused(arch, spec_id, block_size):
    with pytest.raises(ValueError, match="kv_block_size"):
        dispatch_attention(pinned(_request(arch, kv_block_size=block_size), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_retargets_the_cache_and_forwards_the_block_table(arch, spec_id):
    """The runner sees the real cache extent and the paged tensors; it stops at
    compile, so no GPU is needed; with unsafe_skip_paged_value_validation the
    runner skips its synchronized value checks."""
    import importlib

    ad = importlib.import_module(f"kernels.{arch}.attention_dense")
    result = dispatch_attention(pinned(_request(arch), spec_id))
    num_kv_blocks = 4 * (_SQ // _BS)
    q = SimpleNamespace(shape=(1, _SQ, _HQ, _D))
    cache = CacheStandIn((num_kv_blocks, _BS, _HKV, _D))
    tensors = {
        "q": q,
        "k": cache,
        "v": cache,
        "out": q,
        "block_tables": I32DeviceTensor([list(range(_SQ // _BS))]),
        "kv_lens": I32DeviceTensor([_SQ]),
    }
    seen = {}
    real_run = ad.run_attention_dense_torch

    def run(**kw):
        seen.update(kw)
        return real_run(**kw)

    sentinel = RuntimeError("reached-compile")
    for skip in (False, True):
        seen.clear()
        with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch.object(
            ad, "run_attention_dense_torch", run
        ), mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
            with pytest.raises(RuntimeError, match="reached-compile"):
                result.bind_torch(
                    tensors, unsafe_skip_paged_value_validation=skip
                ).launch()
        assert seen["spec"].num_kv_blocks == num_kv_blocks
        assert seen["block_tables"] is tensors["block_tables"]
        assert seen["kv_lens"] is tensors["kv_lens"]
        assert seen["validate_paged"] is not skip


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_and_spec_refuse_non_bool_paged_flags(arch, spec_id):
    result = dispatch_attention(pinned(_request(arch), spec_id))
    cache = CacheStandIn((_SQ // _BS, _BS, _HKV, _D))
    tensors = {"q": None, "k": cache, "v": cache, "out": None}
    with pytest.raises(ValueError, match="must be a bool"):
        result.bind_torch(tensors, unsafe_skip_paged_value_validation="False")
    with pytest.raises(ValueError, match="paged must be a bool"):
        dataclasses.replace(result.spec.kernel_spec, paged="no")


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_launch_refuses_unknown_keywords(arch, spec_id):
    """A bind-only option passed to launch() is refused, not dropped."""
    result = dispatch_attention(pinned(_request(arch), spec_id))
    cache = CacheStandIn((_SQ // _BS, _BS, _HKV, _D))
    launch = result.bind_torch({"q": None, "k": cache, "v": cache, "out": None}).launch
    for kw in ({"unsafe_skip_paged_value_validation": True}, {"softmax_scale": 0.5}):
        with pytest.raises(TypeError, match=next(iter(kw))):
            launch(**kw)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_bind_refuses_unknown_keywords(arch, spec_id):
    """The unified binding's softmax_scale is refused, not run with 1/sqrt(d)."""
    result = dispatch_attention(pinned(_request(arch), spec_id))
    cache = CacheStandIn((_SQ // _BS, _BS, _HKV, _D))
    with pytest.raises(TypeError, match="softmax_scale"):
        result.bind_torch(
            {"q": None, "k": cache, "v": cache, "out": None}, softmax_scale=0.5
        )


@pytest.mark.parametrize("arch", sorted(_PAGED_SPEC_IDS))
def test_contiguous_dense_spec_ignores_the_block_count(arch):
    spec_id = _PAGED_SPEC_IDS[arch][0]
    tuning = dispatch_attention(pinned(_request(arch, layout="auto"), spec_id)).spec
    assert not tuning.kernel_spec.paged
    assert tuning.with_num_kv_blocks(512) is tuning
