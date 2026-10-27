# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher contract for paged K/V (``layout="paged"``) on the dense candidates."""

import dataclasses
import math
from types import SimpleNamespace
from unittest import mock

import pytest

from ..._dense_paged_stand_ins import CacheStandIn, I32DeviceTensor
from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
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


def _pinned(req, spec_id):
    candidate = next(
        c for c in attention_execution_candidates() if c.spec_id == spec_id
    )
    return dataclasses.replace(req, algorithm=candidate.algorithm, spec_id=spec_id)


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_paged_request_builds_a_paged_spec(arch, spec_id):
    spec = dispatch_attention(_pinned(_request(arch), spec_id)).spec.kernel_spec
    assert spec.paged and spec.block_size == _BS
    assert spec.num_kv_blocks == _SQ // _BS  # the smallest cache, until binding
    assert not spec.persistent


@pytest.mark.parametrize(
    "spec_id", ["gfx950_dense_persist", "gfx950_dense_persist_widedma"]
)
def test_persistent_variants_decline_paged(spec_id):
    with pytest.raises(ValueError, match="layout_paged"):
        dispatch_attention(_pinned(_request("gfx950"), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "overrides",
    [dict(batch=2), dict(sliding_window=0), dict(hdim_q=64, hdim_v=64)],
    ids=["multi_seq", "plain_causal", "d64"],
)
def test_paged_outside_the_cohort_is_refused(arch, spec_id, overrides):
    with pytest.raises(ValueError):
        dispatch_attention(_pinned(_request(arch, **overrides), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_layout_is_matched_like_the_capability(arch, spec_id):
    """The request's layout spelling is normalized as in its feature set, so the
    candidate that matched ``layout_paged`` also builds the paged spec."""
    req = _request(arch, layout=" Paged ")
    assert dispatch_attention(_pinned(req, spec_id)).spec.kernel_spec.paged


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_non_positive_kv_block_size_is_refused(arch, spec_id):
    with pytest.raises(ValueError, match="kv_block_size"):
        dispatch_attention(_pinned(_request(arch, kv_block_size=0), spec_id))


@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_binding_retargets_the_cache_and_forwards_the_block_table(arch, spec_id):
    """The runner sees the real cache extent and the paged tensors; it stops at
    compile, so no GPU is needed; with unsafe_skip_paged_value_validation the
    runner skips its synchronized value checks."""
    import importlib

    ad = importlib.import_module(f"kernels.{arch}.attention_dense")
    result = dispatch_attention(_pinned(_request(arch), spec_id))
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


@pytest.mark.parametrize("arch", sorted(_PAGED_SPEC_IDS))
def test_contiguous_dense_spec_ignores_the_block_count(arch):
    spec_id = _PAGED_SPEC_IDS[arch][0]
    tuning = dispatch_attention(_pinned(_request(arch, layout="auto"), spec_id)).spec
    assert not tuning.kernel_spec.paged
    assert tuning.with_num_kv_blocks(512) is tuning


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_paged_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    """A paged request served end to end through dispatch: permuted pages in a 2x
    cache whose unused pages are NaN, against the fp32 sliding-window reference."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != arch:
        pytest.skip(f"needs {arch}")
    window = 1024
    torch.manual_seed(0)
    q = torch.randn(1, _SQ, _HQ, _D, device="cuda", dtype=torch.float16)
    k = torch.randn(1, _SQ, _HKV, _D, device="cuda", dtype=torch.float16)
    v = torch.randn(1, _SQ, _HKV, _D, device="cuda", dtype=torch.float16)
    npages = _SQ // _BS
    phys = torch.randperm(2 * npages, device="cuda")[:npages]
    k_cache = torch.full(
        (2 * npages, _BS, _HKV, _D), float("nan"), device="cuda", dtype=torch.float16
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    k_cache[phys] = k[0].view(npages, _BS, _HKV, _D)
    v_cache[phys] = v[0].view(npages, _BS, _HKV, _D)
    out = torch.empty_like(q)
    tensors = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "block_tables": phys.to(torch.int32).view(1, npages).contiguous(),
        "kv_lens": torch.tensor([_SQ], device="cuda", dtype=torch.int32),
    }
    result = dispatch_attention(_pinned(_request(arch, sliding_window=window), spec_id))
    result.bind_torch(tensors).launch()
    torch.cuda.synchronize()

    rep = _HQ // _HKV
    qi = torch.arange(_SQ, device="cuda").view(-1, 1)
    ki = torch.arange(_SQ, device="cuda").view(1, -1)
    keep = (ki <= qi) & (ki > qi - window)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2).repeat_interleave(rep, 1),
        v.float().transpose(1, 2).repeat_interleave(rep, 1),
        attn_mask=keep,
        scale=1.0 / math.sqrt(_D),
    ).transpose(1, 2)
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max().item() < 2e-2
