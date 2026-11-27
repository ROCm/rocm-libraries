# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for paged K/V (``layout="paged"``) on the dense candidates (host
only)."""

import dataclasses
import math
from types import SimpleNamespace
from unittest import mock

import pytest

from ..._dense_paged_stand_ins import CacheStandIn, I32DeviceTensor, pinned
from dispatch.attention import (
    AttentionRequest,
    dispatch_attention,
    tuning_spec_with_knobs,
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
    # gfx942 defaults windowed D128 paged requests to the persistent grid.
    windowed_d128 = req.sliding_window > 256 and req.hdim_q == 128
    assert spec.persistent == (arch == "gfx942" and windowed_d128)
    assert not spec.varlen  # whole KV tiles keep the single-sequence kernel


@pytest.mark.parametrize("window", [0, 1024], ids=["plain_causal", "swa"])
def test_wide_dma_takes_one_paged_sequence_with_tile_sized_pages(window):
    req = _request("gfx950", kv_block_size=64, sliding_window=window)
    spec = tuning_spec_with_knobs(req, "gfx950_dense_persist_widedma", {}).kernel_spec
    assert spec.paged and spec.wide_lds_dma and spec.block_size == spec.block_n


@pytest.mark.parametrize(
    "overrides,error",
    [
        (dict(), "block_size >= block_n"),  # pages smaller than the KV tile
        (dict(kv_block_size=64, batch=3), "varlen|single-sequence"),
    ],
    ids=["small_pages", "multi_seq"],
)
def test_wide_dma_declines_other_paged_requests(overrides, error):
    with pytest.raises(ValueError, match=error):
        tuning_spec_with_knobs(
            _request("gfx950", **overrides), "gfx950_dense_persist_widedma", {}
        )


_PINNED_PERSISTENT = pytest.mark.parametrize(
    "arch,spec_id,knobs",
    [
        ("gfx950", "gfx950_dense_persist", {}),
        ("gfx942", "gfx942_dense", {"persistent": True}),
    ],
    ids=["gfx950_persist", "gfx942_persistent_knob"],
)


@_PINNED_PERSISTENT
@pytest.mark.parametrize(
    "overrides",
    [dict(), dict(sliding_window=0), dict(hdim_q=64, hdim_v=64)],
    ids=["swa", "plain_causal", "d64"],
)
def test_pinned_persistent_takes_one_paged_sequence(arch, spec_id, knobs, overrides):
    req = _request(arch, **overrides)
    spec = tuning_spec_with_knobs(req, spec_id, knobs).kernel_spec
    assert spec.paged and spec.persistent and not spec.varlen
    assert spec.runtime_shape  # the cache bound is the num_kv_blocks kernarg


@_PINNED_PERSISTENT
def test_pinned_persistent_refuses_paged_varlen(arch, spec_id, knobs):
    """Several sequences need the varlen body, which has no persistent form."""
    with pytest.raises(ValueError, match="varlen"):
        tuning_spec_with_knobs(_request(arch, batch=3), spec_id, knobs)


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "overrides,error",
    [
        (dict(mask_type=0, sliding_window=0), "causal-only"),
        (dict(kv_block_size=8), "must be >= 16"),
    ],
    ids=["non_causal", "page_8"],
)
def test_paged_outside_the_cohort_is_refused(arch, spec_id, overrides, error):
    with pytest.raises(ValueError, match=error):
        dispatch_attention(pinned(_request(arch, **overrides), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_lse_is_refused_with_paged_varlen(arch, spec_id):
    req = _request(arch, batch=3, emit_lse=True)
    with pytest.raises(ValueError, match="emit_lse is not yet supported with paged"):
        dispatch_attention(pinned(req, spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "overrides",
    [
        dict(seqlen_q=512, seqlen_k=1000),
        dict(batch=3, seqlen_q=512),
        dict(batch=3, seqlen_q=1000, seqlen_k=512),
    ],
    ids=["partial_kv_tile_sq_lt_skv", "multi_seq_sq_lt_skv", "multi_seq_sq_gt_skv"],
)
def test_paged_top_left_cross_length_keeps_its_lengths(arch, spec_id, overrides):
    """Top-left needs no q_len == kv_len: the request's maxima reach the spec
    unchanged, still top-left."""
    req = _request(arch, **overrides)
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged and not spec.causal_bottom_right
    assert (spec.seqlen_q, spec.seqlen_kv) == (req.seqlen_q, req.seqlen_k)


@pytest.mark.parametrize("seqlen_q", [1000, 2600])
def test_gfx942_paged_partial_q_tile_cross_length_takes_varlen(seqlen_q):
    """No ragged body on gfx942: the varlen body bounds the partial Q tile."""
    req = _request("gfx942", seqlen_q=seqlen_q, sliding_window=0)
    spec = dispatch_attention(pinned(req, "gfx942_dense")).spec.kernel_spec
    assert spec.paged and spec.varlen and not spec.causal_bottom_right


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize("seqlen", [_SQ, 1000], ids=["aligned", "ragged_max"])
def test_multi_seq_paged_request_builds_a_paged_varlen_spec(arch, spec_id, seqlen):
    """Several sequences take packed varlen Q; the lengths are maxima, so a
    non-tile-multiple maximum stays on the varlen body instead of going ragged."""
    req = _request(arch, batch=3, seqlen_q=seqlen, seqlen_k=seqlen)
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged and spec.varlen and not spec.ragged
    assert not spec.causal_bottom_right  # top-left request
    assert spec.batch == 3
    assert spec.num_kv_blocks == 3 * -(-seqlen // _BS)


_GFX950_CASES = [("gfx950", sid) for sid in _PAGED_SPEC_IDS["gfx950"]]


# The band form of bottom-right causal: gfx942 dense serves only the explicit
# causal mask kinds for cross-length requests.
_BR_MASKS = [
    (arch, sid, mask)
    for arch, sid in _CASES
    for mask in (
        dict(mask_type=2),
        dict(mask_type=0, right_bound=0, diagonal_alignment=1),
    )
    if arch == "gfx950" or mask["mask_type"] == 2
]


@pytest.mark.parametrize("arch,spec_id,mask", _BR_MASKS)
@pytest.mark.parametrize(
    "batch,sq,sk",
    [(3, 512, _SQ), (3, _SQ, _SQ), (3, 1000, 1017), (1, 512, _SQ)],
    ids=["multi_seq", "multi_seq_equal_maxima", "multi_seq_ragged", "one_seq"],
)
def test_bottom_right_paged_request_takes_the_per_sequence_shift(
    arch, spec_id, mask, batch, sq, sk
):
    """Bottom-right over a paged cache runs on the varlen body, whose diagonal is
    each sequence's own kv_len - q_len. Equal maxima keep the shift: the
    sequences themselves may still differ."""
    req = _request(
        arch, batch=batch, seqlen_q=sq, seqlen_k=sk, sliding_window=0, **mask
    )
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged and spec.varlen and spec.causal_bottom_right
    assert (spec.batch, spec.seqlen_q, spec.seqlen_kv) == (batch, sq, sk)


@pytest.mark.parametrize(
    "batch,sq", [(3, _SQ), (1, 512)], ids=["multi_seq_equal_maxima", "one_seq"]
)
def test_gfx942_declines_a_bottom_right_paged_window(batch, sq):
    """The gfx942 window is anchored on the top-left diagonal; equal maxima do not
    make each sequence's q_len == kv_len, so a top-left fold would be wrong."""
    req = _request("gfx942", batch=batch, seqlen_q=sq, mask_type=2, sliding_window=1024)
    with pytest.raises(ValueError, match="causal_bottom_right"):
        dispatch_attention(pinned(req, "gfx942_dense"))


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize("seqlen", [_SQ, 1000], ids=["aligned", "ragged"])
def test_one_sequence_equal_lengths_bottom_right_window_is_served(
    arch, spec_id, seqlen
):
    """One sequence's request lengths are its own, so equal ones make the
    bottom-right window the top-left one."""
    req = _request(arch, mask_type=2, seqlen_q=seqlen, seqlen_k=seqlen)
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged and spec.sliding_window == 1024
    assert not spec.causal_bottom_right


@pytest.mark.parametrize("arch,spec_id", _GFX950_CASES)
@pytest.mark.parametrize(
    "mask",
    [
        dict(mask_type=2, sliding_window=1024),
        dict(mask_type=3, sliding_window=0, right_bound=64, diagonal_alignment=1),
    ],
    ids=["swa", "lookahead"],
)
def test_gfx950_declines_equal_maxima_bottom_right_band(arch, spec_id, mask):
    """The body anchors a window or lookahead top-left; over several sequences
    equal maxima do not make that the bottom-right band."""
    req = _request(arch, batch=3, **mask)
    with pytest.raises(ValueError, match="several paged sequences"):
        dispatch_attention(pinned(req, spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_one_sequence_ending_inside_a_kv_tile_takes_the_varlen_body(arch, spec_id):
    """The single-sequence kernel would read the padded keys of the partial last
    KV tile from stale page rows; the varlen body zeroes them."""
    req = _request(arch, seqlen_q=1000, seqlen_k=1000)
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged and spec.varlen and not spec.ragged
    assert spec.batch == 1 and spec.seqlen_kv % spec.block_n


@pytest.mark.parametrize(
    "arch,spec_id,seqlen,tile,varlen",
    [
        # Whole KV tiles at the default block_n, a partial last one at 128.
        ("gfx950", "gfx950_dense_grid", 1088, (256, 128), True),
        # A partial last KV tile at the default block_n, whole ones at 32.
        ("gfx950", "gfx950_dense_grid", 1056, (256, 32), False),
        # Whole Q tiles at the default block_m, a partial last one at 512.
        ("gfx942", "gfx942_dense", 1280, (512, 64), True),
    ],
    ids=["gfx950_partial_kv", "gfx950_whole_kv", "gfx942_partial_q"],
)
def test_swept_tile_picks_the_body_for_its_own_tails(
    arch, spec_id, seqlen, tile, varlen
):
    """A swept tile gets the body its own tails need, as if it were the default."""
    req = _request(
        arch, seqlen_q=seqlen, seqlen_k=seqlen, sliding_window=0, hdim_q=64, hdim_v=64
    )
    default = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert default.varlen is not varlen
    knobs = dict(zip(("block_m", "block_n"), tile))
    spec = tuning_spec_with_knobs(req, spec_id, knobs).kernel_spec
    assert spec.varlen is varlen
    assert (spec.block_m, spec.block_n) == tile


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize("window", [0, 1024], ids=["causal", "swa"])
def test_paged_ragged_query_tail_follows_the_contiguous_kernel(arch, spec_id, window):
    req = _request(arch, seqlen_q=1088, seqlen_k=1088, sliding_window=window)
    spec = dispatch_attention(pinned(req, spec_id)).spec.kernel_spec
    assert spec.paged
    # gfx942 has no ragged body and gfx950's refuses a window: the varlen body
    # takes those.
    if arch == "gfx942" or window:
        assert spec.varlen and not spec.ragged
    else:
        assert spec.ragged and not spec.varlen


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
    with pytest.raises(ValueError, match="use_i64_kv_addr must be a bool"):
        dataclasses.replace(result.spec.kernel_spec, use_i64_kv_addr=1)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_launch_refuses_unknown_keywords(arch, spec_id):
    """A bind-only option passed to launch() is refused, not dropped."""
    result = dispatch_attention(pinned(_request(arch), spec_id))
    cache = CacheStandIn((_SQ // _BS, _BS, _HKV, _D))
    q = SimpleNamespace(shape=(1, _SQ, _HQ, _D))
    launch = result.bind_torch({"q": q, "k": cache, "v": cache, "out": q}).launch
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


def _bind_until_compile(arch, result, tensors, **bind_kw):
    """Launch through the binding up to compile; return the runner's arguments."""
    import importlib

    ad = importlib.import_module(f"kernels.{arch}.attention_dense")
    seen = {}
    real_run = ad.run_attention_dense_torch

    def run(**kw):
        seen.update(kw)
        return real_run(**kw)

    sentinel = RuntimeError("reached-compile")
    with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch.object(
        ad, "run_attention_dense_torch", run
    ), mock.patch("rocke.helpers.compile.compile_kernel", side_effect=sentinel):
        with pytest.raises(RuntimeError, match="reached-compile"):
            result.bind_torch(tensors, **bind_kw).launch()
    return seen


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_forwards_packed_queries_and_cu_seqlens_q(arch, spec_id):
    lens = [_SQ, 1000, 17]
    cu = [0, lens[0], lens[0] + lens[1], sum(lens)]
    result = dispatch_attention(pinned(_request(arch, batch=3), spec_id))
    pages = [-(-n // _BS) for n in lens]
    width = _SQ // _BS
    table, base = [], 0
    for n in pages:
        table.append(list(range(base, base + n)) + [0x7FFF_FFFF] * (width - n))
        base += n
    packed = CacheStandIn((sum(lens), _HQ, _D))
    cache = CacheStandIn((base, _BS, _HKV, _D))
    tensors = {
        "q": packed,
        "k": cache,
        "v": cache,
        "out": packed,
        "block_tables": I32DeviceTensor(table),
        "kv_lens": I32DeviceTensor(lens),
        "cu_seqlens_q": I32DeviceTensor(cu),
    }
    seen = _bind_until_compile(arch, result, tensors)
    assert seen["spec"].varlen and seen["spec"].num_kv_blocks == base
    assert seen["q"] is packed and seen["out"] is packed
    assert seen["cu_seqlens_q"] is tensors["cu_seqlens_q"]
    assert "cu_seqlens_kv" not in seen


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_views_one_packed_sequence_as_the_single_sequence_layout(arch, spec_id):
    result = dispatch_attention(pinned(_request(arch), spec_id))
    packed = CacheStandIn((_SQ, _HQ, _D))
    tensors = {
        "q": packed,
        "k": CacheStandIn((_SQ // _BS, _BS, _HKV, _D)),
        "v": CacheStandIn((_SQ // _BS, _BS, _HKV, _D)),
        "out": packed,
        "block_tables": I32DeviceTensor([list(range(_SQ // _BS))]),
        "kv_lens": I32DeviceTensor([_SQ]),
    }
    seen = _bind_until_compile(arch, result, tensors)
    assert not seen["spec"].varlen
    assert seen["q"].shape == seen["out"].shape == (1, _SQ, _HQ, _D)
    # Unified-style callers also pass cu_seqlens_q for one sequence.
    tensors["cu_seqlens_q"] = I32DeviceTensor([0, _SQ])
    seen = _bind_until_compile(arch, result, tensors)
    assert "cu_seqlens_q" not in seen
    tensors["cu_seqlens_q"] = I32DeviceTensor([0, _SQ - 64])
    with pytest.raises(ValueError, match=r"must be \[0, "):
        result.bind_torch(tensors).launch()
    seen = _bind_until_compile(
        arch, result, tensors, unsafe_skip_paged_value_validation=True
    )
    assert "cu_seqlens_q" not in seen
    tensors["cu_seqlens_q"] = I32DeviceTensor([0, _SQ, _SQ])
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        result.bind_torch(tensors).launch()
    # The runner's metadata rules hold although the tensor is dropped, also
    # without value validation.
    for bad, error in (
        (I32DeviceTensor([0, _SQ], dtype="torch.int64"), "dtype must be int32"),
        (I32DeviceTensor([0, _SQ], is_cuda=False), "must be a GPU tensor"),
        (I32DeviceTensor([0, _SQ], device="cuda:1"), "on the K/V device"),
        (I32DeviceTensor([0, _SQ], inner_stride=2), "innermost stride"),
    ):
        tensors["cu_seqlens_q"] = bad
        with pytest.raises(ValueError, match=error):
            result.bind_torch(tensors, unsafe_skip_paged_value_validation=True).launch()
    del tensors["cu_seqlens_q"]
    tensors["q"] = tensors["out"] = CacheStandIn((_SQ - 64, _HQ, _D))
    with pytest.raises(ValueError, match="seqlen_q"):
        result.bind_torch(tensors)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_packs_one_sequence_for_the_varlen_body(arch, spec_id):
    """One sequence on the varlen body: its [1, Sq, Hq, D] form is viewed packed,
    and cu_seqlens_q = [0, Sq] is built once at bind time when none is given."""
    from dispatch.attention import bindings

    sq = 1000
    result = dispatch_attention(
        pinned(_request(arch, seqlen_q=sq, seqlen_k=sq), spec_id)
    )
    pages = -(-sq // _BS)
    cache = CacheStandIn((pages, _BS, _HKV, _D))
    tensors = {
        "q": CacheStandIn((1, sq, _HQ, _D)),
        "k": cache,
        "v": cache,
        "out": CacheStandIn((1, sq, _HQ, _D)),
        "block_tables": I32DeviceTensor([list(range(pages))]),
        "kv_lens": I32DeviceTensor([sq]),
    }
    built = []

    def one_sequence_cu(q):
        built.append(tuple(q.shape))
        return I32DeviceTensor([0, int(q.shape[0])])

    with mock.patch.object(bindings, "_one_sequence_cu_seqlens_q", one_sequence_cu):
        seen = _bind_until_compile(arch, result, tensors)
    assert built == [(sq, _HQ, _D)]
    assert seen["spec"].varlen
    assert seen["q"].shape == seen["out"].shape == (sq, _HQ, _D)
    assert seen["cu_seqlens_q"].tolist() == [0, sq]
    # A caller's packed Q and cu_seqlens_q go through unchanged.
    packed = CacheStandIn((sq, _HQ, _D))
    cu = I32DeviceTensor([0, sq])
    seen = _bind_until_compile(
        arch, result, {**tensors, "q": packed, "out": packed, "cu_seqlens_q": cu}
    )
    assert seen["q"] is packed and seen["cu_seqlens_q"] is cu


def test_one_sequence_cu_seqlens_q_is_int32_on_the_q_device():
    torch = pytest.importorskip("torch")
    from dispatch.attention.bindings import _one_sequence_cu_seqlens_q

    cu = _one_sequence_cu_seqlens_q(torch.empty(1000, 2, 8))
    assert cu.dtype == torch.int32 and cu.device.type == "cpu"
    assert cu.tolist() == [0, 1000]


# Blocks of one page that fill exactly the i32 K/V buffer range.
_I32_BLOCKS = 0x7FFF_0000 // (_BS * _HKV * _D * 2)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_cache_size_specializes_i64_but_keeps_the_id(arch, spec_id):
    """A cache past the i32 range is a runtime specialization: a different
    binary (identity), the same configuration (tuning_id)."""
    from rocke.dispatch.core import spec_identity

    base = dispatch_attention(pinned(_request(arch), spec_id)).spec
    at_limit = base.with_num_kv_blocks(_I32_BLOCKS)
    past = base.with_num_kv_blocks(_I32_BLOCKS + 1)
    assert not at_limit.kernel_spec.use_i64_kv_addr
    assert past.kernel_spec.use_i64_kv_addr
    assert past.kernel_spec.num_kv_blocks == past.num_kv_blocks == _I32_BLOCKS + 1
    assert at_limit.tuning_id == past.tuning_id == base.tuning_id
    assert spec_identity(past) != spec_identity(at_limit)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_forced_i64_survives_retargeting_to_a_small_cache(arch, spec_id):
    base = dispatch_attention(pinned(_request(arch), spec_id)).spec
    forced = dataclasses.replace(
        base, kernel_spec=dataclasses.replace(base.kernel_spec, use_i64_kv_addr=True)
    )
    small = forced.with_num_kv_blocks(_SQ // _BS)
    assert small.kernel_spec.use_i64_kv_addr
    assert small.kernel_spec.num_kv_blocks == _SQ // _BS
    assert small.tuning_id == base.tuning_id


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_a_cache_past_the_i32_range(arch, spec_id):
    """Binding switches the spec to i64 K/V addressing, which the arch's
    runner accepts (it gets as far as the compile)."""
    result = dispatch_attention(pinned(_request(arch), spec_id))
    q = SimpleNamespace(shape=(1, _SQ, _HQ, _D))
    cache = CacheStandIn((_I32_BLOCKS + 1, _BS, _HKV, _D))
    tensors = {
        "q": q,
        "k": cache,
        "v": cache,
        "out": q,
        "block_tables": I32DeviceTensor([list(range(_SQ // _BS))]),
        "kv_lens": I32DeviceTensor([_SQ]),
    }
    seen = _bind_until_compile(arch, result, tensors)
    assert seen["spec"].use_i64_kv_addr


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_packed_sequences_to_a_cache_past_the_i32_range(arch, spec_id):
    lens = [1000, 17, 512]
    cu = [0, lens[0], lens[0] + lens[1], sum(lens)]
    req = _request(arch, batch=3, seqlen_q=max(lens), seqlen_k=max(lens))
    result = dispatch_attention(pinned(req, spec_id))
    assert not result.spec.kernel_spec.use_i64_kv_addr
    num_kv_blocks = _I32_BLOCKS + 1
    width = -(-max(lens) // _BS)
    table, top = [], num_kv_blocks
    for n in lens:  # pages at the top of the cache, past the i32 range
        pages = -(-n // _BS)
        table.append(list(range(top - pages, top)) + [0x7FFF_FFFF] * (width - pages))
        top -= pages
    packed = CacheStandIn((sum(lens), _HQ, _D))
    cache = CacheStandIn((num_kv_blocks, _BS, _HKV, _D))
    tensors = {
        "q": packed,
        "k": cache,
        "v": cache,
        "out": packed,
        "block_tables": I32DeviceTensor(table),
        "kv_lens": I32DeviceTensor(lens),
        "cu_seqlens_q": I32DeviceTensor(cu),
    }
    seen = _bind_until_compile(arch, result, tensors)
    bound = result.spec.with_num_kv_blocks(num_kv_blocks)
    assert seen["spec"] == bound.kernel_spec
    assert seen["spec"].varlen and seen["spec"].use_i64_kv_addr
    assert seen["spec"].num_kv_blocks == num_kv_blocks
    assert bound.tuning_id == result.spec.tuning_id
    assert seen["q"] is packed and seen["cu_seqlens_q"] is tensors["cu_seqlens_q"]


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_request_whose_own_pages_pass_the_i32_range_takes_i64(arch, spec_id):
    """A short chunk over a long prefix needs more cache than the i32 K/V range
    holds: selection takes i64 page bases rather than refusing the request."""

    def spec_for(num_kv_blocks):
        req = _request(
            arch,
            seqlen_q=512,
            seqlen_k=num_kv_blocks * _BS,
            mask_type=2,
            sliding_window=0,
        )
        return dispatch_attention(pinned(req, spec_id)).spec.kernel_spec

    assert not spec_for(_I32_BLOCKS).use_i64_kv_addr
    # Just past the i64 threshold, and past 2 GiB of logical K/V, which a
    # contiguous spec would refuse.
    for blocks in (_I32_BLOCKS + 1, 2 * _I32_BLOCKS):
        past = spec_for(blocks)
        assert past.paged and past.use_i64_kv_addr
        assert past.num_kv_blocks == blocks


@pytest.mark.parametrize("arch", sorted(_PAGED_SPEC_IDS))
def test_contiguous_dense_spec_ignores_the_block_count(arch):
    spec_id = _PAGED_SPEC_IDS[arch][0]
    tuning = dispatch_attention(pinned(_request(arch, layout="auto"), spec_id)).spec
    assert not tuning.kernel_spec.paged
    assert tuning.with_num_kv_blocks(512) is tuning


def _check_paged_varlen_dispatch_on_gpu(
    arch, spec_id, lens, *, one_seq_4d=False, q_lens=None
):
    """Sequences of ``lens`` served end to end through dispatch: packed Q, pages
    permuted in a 2x cache whose unused pages and past-length slots are NaN, a
    table padded with garbage ids and a NaN-filled output, against per-sequence
    fp32 sliding-window references. ``one_seq_4d``: one sequence passed as
    [1, S, Hq, D] without cu_seqlens_q, which the binding then builds.
    ``q_lens``: shorter query lengths, requested as bottom-right causal (no
    window); the reference shifts each sequence by its kv_len - q_len."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != arch:
        pytest.skip(f"needs {arch}")
    bottom_right = q_lens is not None
    q_lens = q_lens if bottom_right else lens
    window, seqlen = (0 if bottom_right else 1024), max(lens)
    torch.manual_seed(0)
    total = sum(q_lens)
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    q = torch.randn(total, _HQ, _D, device="cuda", dtype=torch.float16)
    pages = [-(-n // _BS) for n in lens]
    width = -(-seqlen // _BS)
    num_blocks = 2 * sum(pages)
    phys = torch.randperm(num_blocks, device="cuda")[: sum(pages)]
    k_cache = torch.full(
        (num_blocks, _BS, _HKV, _D), float("nan"), device="cuda", dtype=torch.float16
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    table = torch.full(
        (len(lens), width), 0x7FFF_FFFF, device="cuda", dtype=torch.int32
    )
    ks, vs, first = [], [], 0
    for b, n in enumerate(lens):
        ids = phys[first : first + pages[b]]
        first += pages[b]
        table[b, : pages[b]] = ids.to(torch.int32)
        k = torch.randn(n, _HKV, _D, device="cuda", dtype=torch.float16)
        v = torch.randn(n, _HKV, _D, device="cuda", dtype=torch.float16)
        for p in range(pages[b]):
            rows = slice(p * _BS, min(n, (p + 1) * _BS))
            k_cache[ids[p], : rows.stop - rows.start] = k[rows]
            v_cache[ids[p], : rows.stop - rows.start] = v[rows]
        ks.append(k)
        vs.append(v)
    # Rows past the packed output catch a tail store the kernel's guard missed.
    out_buf = torch.full(
        (total + 256, _HQ, _D), float("nan"), device="cuda", dtype=torch.float16
    )
    out = out_buf[:total]
    tensors = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "block_tables": table,
        "kv_lens": torch.tensor(lens, device="cuda", dtype=torch.int32),
        "cu_seqlens_q": cu.to("cuda"),
    }
    if one_seq_4d:
        tensors["q"], tensors["out"] = q.unsqueeze(0), out.unsqueeze(0)
        del tensors["cu_seqlens_q"]
    req = _request(
        arch,
        batch=len(lens),
        seqlen_q=max(q_lens),
        seqlen_k=seqlen,
        sliding_window=window,
        mask_type=2 if bottom_right else 1,
    )
    result = dispatch_attention(pinned(req, spec_id))
    assert result.spec.kernel_spec.varlen
    assert result.spec.kernel_spec.causal_bottom_right == bottom_right
    result.bind_torch(tensors).launch()
    torch.cuda.synchronize()
    assert torch.isnan(out_buf[total:]).all(), "store past the packed output"

    rep = _HQ // _HKV
    for b, n in enumerate(lens):
        if q_lens[b] == 0:
            continue
        rows = slice(int(cu[b]), int(cu[b + 1]))
        qi = torch.arange(q_lens[b], device="cuda").view(-1, 1) + (n - q_lens[b])
        ki = torch.arange(n, device="cuda").view(1, -1)
        keep = ki <= qi
        if window:
            keep &= ki > qi - window
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[rows].float().transpose(0, 1),
            ks[b].float().transpose(0, 1).repeat_interleave(rep, 0),
            vs[b].float().transpose(0, 1).repeat_interleave(rep, 0),
            attn_mask=keep,
            scale=1.0 / math.sqrt(_D),
        ).transpose(0, 1)
        assert torch.isfinite(out[rows]).all(), f"sequence {b}"
        assert (out[rows].float() - ref).abs().max().item() < 2e-2, f"sequence {b}"


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_multi_seq_paged_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    _check_paged_varlen_dispatch_on_gpu(arch, spec_id, [_SQ, 1000, 17])


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_bottom_right_paged_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    """Prefix-cache style sequences: queries shorter than their keys (shift 0, mid
    tile and large), and an empty query over cached keys."""
    _check_paged_varlen_dispatch_on_gpu(
        arch, spec_id, [_SQ, 1000, 300, 17], q_lens=[300, 1000, 0, 17]
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_one_sequence_bottom_right_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    _check_paged_varlen_dispatch_on_gpu(
        arch, spec_id, [_SQ], q_lens=[512], one_seq_4d=True
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "seqlen", [1000, 1088], ids=["partial_kv_tile", "partial_q_tile"]
)
def test_one_sequence_partial_tile_dispatch_matches_the_reference_on_gpu(
    arch, spec_id, seqlen
):
    _check_paged_varlen_dispatch_on_gpu(arch, spec_id, [seqlen], one_seq_4d=True)
