# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The unified SDPA binding's data movement, on CPU.

The kernel is replaced by a stand-in that reads only what the binding hands
it -- packed Q, the paged K/V cache through its block table, ``cu_seqlens_q``
and ``seqused_k`` -- and computes attention with ``sdpa_reference`` using the
request's band and scale. Comparing the scattered-back O and LSE with the
case-table oracle then checks packing, paging, lengths and write-back for
every layout and length mode. The kernels themselves are checked on GPU by
``test_unified_sdpa_numeric.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rocke.numeric.sdpa_reference import hipdnn_left_to_window, sdpa_reference

from dispatch.sdpa.bind_unified import (
    _LaunchPlan,
    sdpa_unified_admits,
    sdpa_unified_bind,
)
from tests.sdpa import cases as C
from tests.sdpa import helpers as H
from tests.sdpa.binding_harness import bind_tensors, outputs, strided_tensor
from tests.sdpa.request_bridge import request_from_case
from tests.sdpa.request_cases import bhsd, make_request, packed, ragged_offset

ARCH = "gfx950"


def fake_kernel(plan, req):
    """Attention over exactly the tensors the binding prepared."""
    kt = plan.kernel_tensors()
    used = kt["seqused_k"].tolist()
    table = kt["block_table"]
    k_pages, v_pages = [], []
    for b, n in enumerate(used):
        k_pages.append(kt["k"][table[b].long()].reshape(-1, *kt["k"].shape[2:])[:n])
        v_pages.append(kt["v"][table[b].long()].reshape(-1, *kt["v"].shape[2:])[:n])
    o, lse = sdpa_reference(
        kt["q"].double().numpy(),
        torch.cat(k_pages).double().numpy(),
        torch.cat(v_pages).double().numpy(),
        scale=float(req.scale),
        left_bound=hipdnn_left_to_window(req.left_bound),
        right_bound=req.right_bound,
        top_left=req.diagonal == "top_left",
        q_offsets=np.asarray(kt["cu_seqlens_q"].tolist()),
        kv_offsets=np.concatenate([[0], np.cumsum(used)]),
    )
    kt["out"].copy_(torch.from_numpy(o).to(kt["out"].dtype))
    if plan.lse is not None:
        plan.lse.copy_(torch.from_numpy(lse)[..., None].float())


_CASES = [
    c
    for c in C.all_cases()
    if ARCH in c.archs
    and ("smoke" in c.tags or "full" in c.tags)
    and c.cost <= 5e6
    and sdpa_unified_admits(request_from_case(c, ARCH))[0]
]


def test_the_cpu_table_covers_every_layout_and_length_mode():
    assert {c.length_mode for c in _CASES} == set(H.LENGTH_MODES)
    assert {c.layout for c in _CASES} >= {"bhsd", "bshd", "thd"}
    assert any(c.emit_lse for c in _CASES)


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_binding_moves_data_faithfully(case):
    req = request_from_case(case, ARCH)
    inputs = H.materialize(case)
    tensors = bind_tensors(case, H.to_torch(inputs, device="cpu"), req)

    plan = _LaunchPlan(req, tensors)
    plan.pack()
    fake_kernel(plan, req)
    plan.unpack()

    o, lse = outputs(case, tensors, req)
    assert H.compare(case, o, lse, H.evaluate(inputs)) == []


@pytest.mark.parametrize("layout,inplace", [(None, True), (bhsd, False)])
def test_packed_bshd_q_and_o_are_used_in_place(layout, inplace):
    req = make_request(d=64, **({"layout": layout} if layout else {}))
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    plan = _LaunchPlan(req, tensors)
    assert plan.q_inplace is inplace and plan.o_inplace is inplace
    if inplace:
        assert plan.q_packed.data_ptr() == tensors["q"].data_ptr()


def test_misaligned_packed_bshd_q_and_o_fall_back_to_a_workspace_copy():
    """The gfx942/gfx950 tiled 2D kernels declare ``query_ptr`` / ``output_ptr``
    with ``align=16``. A packed-BSHD Q/O view that is contiguous but starts at
    an unaligned element must not be handed to the kernel in place -- it has
    to copy through the allocated workspace instead, as the dense twin's
    ``_DensePlan`` already does for its BSHD views."""
    req = make_request(d=64)
    tensors = {
        n: strided_tensor(getattr(req, n), misaligned=True)
        for n in ("q", "k", "v", "o")
    }
    plan = _LaunchPlan(req, tensors)
    assert plan.q_inplace is False and plan.o_inplace is False
    assert plan.q_packed.data_ptr() % 16 == 0
    assert plan.o_packed.data_ptr() % 16 == 0


def test_padded_rows_are_zero_and_neg_inf():
    lens = torch.tensor([1, 0], dtype=torch.int32).view(2, 1, 1, 1)
    req = replace(
        make_request(d=64, s_q=1, s_kv=16),
        padding_mask=True,
        seq_len_q=packed((2, 1, 1, 1), "int32"),
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
        stats=packed((2, 4, 1, 1), "fp32"),
    )
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["o"].fill_(7)
    tensors.update(
        seq_len_q=lens,
        seq_len_kv=torch.tensor([16, 16], dtype=torch.int32).view(2, 1, 1, 1),
        stats=torch.full((2, 4, 1, 1), 7.0),
    )
    plan = _LaunchPlan(req, tensors)
    assert plan.cu == [0, 1, 1]
    plan.pack()
    fake_kernel(plan, req)
    plan.unpack()
    assert torch.all(tensors["o"][1] == 0)
    assert torch.all(torch.isneginf(tensors["stats"][1]))
    assert torch.all(torch.isfinite(tensors["stats"][0]))


# seq_len_q / seq_len_kv become cu_seqlens_q / seqused_k, which bound the
# kernel's Q and K/V reads on the device; a bad value must be refused at bind.
# The range/count/dtype rule itself is ``validate_lengths``'s (shared with
# ``sdpa_seqlens`` / ``sdpa_cu_seqlens``, covered torch-free in
# ``tests/test_sdpa_seqlen.py``); these pin that ``sdpa_unified_bind`` reaches
# it before any kernel launch.
_BAD_LENGTHS = [
    ("seq_len_q", [2, 1], r"seq_len_q: every length must satisfy 0 <= len <= 1"),
    ("seq_len_q", [-1, 1], "seq_len_q: every length must be >= 0"),
    ("seq_len_q", [1, 1, 1], "seq_len_q: expected 2 per-batch lengths, got 3"),
    ("seq_len_kv", [17, 16], r"seq_len_kv: every length must satisfy 0 <= len <= 16"),
    ("seq_len_kv", [-1, 16], "seq_len_kv: every length must be >= 0"),
    ("seq_len_kv", [16], "seq_len_kv: expected 2 per-batch lengths, got 1"),
]


@pytest.mark.parametrize(
    "key,values,match", _BAD_LENGTHS, ids=[f"{k}={v}" for k, v, _ in _BAD_LENGTHS]
)
def test_invalid_seq_len_is_refused_before_launch(key, values, match):
    req = replace(
        make_request(d=64, s_q=1, s_kv=16),
        padding_mask=True,
        seq_len_q=packed((2, 1, 1, 1), "int32"),
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
    )
    assert sdpa_unified_admits(req)[0]
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["seq_len_q"] = torch.tensor([1, 1], dtype=torch.int32)
    tensors["seq_len_kv"] = torch.tensor([16, 16], dtype=torch.int32)
    tensors[key] = torch.tensor(values, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        sdpa_unified_bind(req, tensors)


def test_float_seq_len_is_refused_not_truncated():
    # [1.5, 1.0] truncates to the valid [1, 1]; the binding must refuse the
    # float32 tensor instead of launching on the truncated lengths.
    req = replace(
        make_request(d=64, s_q=1, s_kv=16),
        padding_mask=True,
        seq_len_q=packed((2, 1, 1, 1), "int32"),
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
    )
    assert sdpa_unified_admits(req)[0]
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["seq_len_q"] = torch.tensor([1.5, 1.0], dtype=torch.float32)
    tensors["seq_len_kv"] = torch.tensor([16, 16], dtype=torch.int32)
    with pytest.raises(ValueError, match="seq_len_q must be torch.int32"):
        sdpa_unified_bind(req, tensors)


def _ragged_request():
    def ragged(t):
        return replace(
            t,
            ragged_offset=ragged_offset(t.dims[0]),  # int32
            ragged_offset_multiplier=t.strides[2],
        )

    req = make_request(d=64)
    req = replace(req, **{n: ragged(getattr(req, n)) for n in ("q", "k", "v", "o")})
    assert sdpa_unified_admits(req)[0]
    tensors = {
        n: torch.randn(getattr(req, n).volume).half() for n in ("q", "k", "v", "o")
    }
    return req, tensors


@pytest.mark.parametrize(
    "offsets,dtype",
    [
        # Truncate to the valid int32 offsets [0, 4, 8].
        ([0.5, 4.5, 8.5], torch.float32),
        # Integer, but not the int32 the request's ragged_offset describes.
        ([0, 4, 8], torch.int64),
    ],
    ids=["float32", "int64"],
)
def test_ragged_offsets_of_another_dtype_are_refused(offsets, dtype):
    req, tensors = _ragged_request()
    tensors["ragged_offset_q"] = torch.tensor([0, 4, 8], dtype=torch.int32)
    tensors["ragged_offset_k"] = torch.tensor([0, 4, 8], dtype=torch.int32)
    _LaunchPlan(req, dict(tensors))  # int32 offsets bind
    tensors["ragged_offset_q"] = torch.tensor(offsets, dtype=dtype)
    with pytest.raises(ValueError, match="q: ragged offsets must be torch.int32"):
        _LaunchPlan(req, tensors)


def test_a_tensor_that_disagrees_with_the_request_is_refused():
    req = make_request(d=64)  # BSHD
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["q"] = torch.empty_strided(
        req.q.dims, bhsd(2, 4, 8, 64).strides, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="strides"):
        _LaunchPlan(req, tensors)


def _bind_with_a_fake_launch(monkeypatch, current=4242):
    """``sdpa_unified_bind`` on CPU, with Torch's current stream faked as
    ``current`` and the kernel replaced by a recorder of the stream it got."""
    from dispatch.attention import bindings
    from dispatch.sdpa import bind_shared
    from kernels.common import attention_unified

    # The real validator needs a GPU; these tests stop at the launch call.
    monkeypatch.setattr(
        bindings, "validate_tuning_attention_tensors", lambda problem, t, **kw: 1
    )
    monkeypatch.setattr(bind_shared, "resolve_stream", lambda s, device=None: current)
    streams = []
    monkeypatch.setattr(
        attention_unified,
        "run_unified_attention_torch",
        lambda **kw: streams.append(kw["stream"]),
    )
    req = make_request(d=64)
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    return sdpa_unified_bind(req, tensors), streams


@pytest.mark.parametrize("stream", [None, 0, 4242], ids=["unset", "zero", "named"])
def test_launch_runs_on_the_current_stream(monkeypatch, stream):
    """Graph capture names the current stream by handle; that is the stream
    the binding packs on, so it is honored, not refused."""
    binding, streams = _bind_with_a_fake_launch(monkeypatch)
    binding.launch(stream=stream)
    assert streams == [4242]


def test_launch_rejects_a_stream_other_than_the_current_one(monkeypatch):
    """Pack and unpack run on the current stream and the workspace belongs to
    it, so launching the kernel alone on another stream would race."""
    binding, streams = _bind_with_a_fake_launch(monkeypatch)
    with pytest.raises(NotImplementedError, match="cannot honor stream=123456"):
        binding.launch(stream=123456)
    assert streams == []


def test_launch_rejects_an_unknown_keyword(monkeypatch):
    binding, streams = _bind_with_a_fake_launch(monkeypatch)
    with pytest.raises(TypeError):
        binding.launch(grid=(1, 1, 1))
    assert streams == []


def test_misaligned_ragged_q_and_o_fall_back_to_a_workspace_copy():
    """The ragged twin of the BSHD test above: contiguous packed rows used in
    place only from a 16-byte-aligned base."""
    req, _ = _ragged_request()
    offsets = {
        "ragged_offset_q": torch.tensor([0, 4, 8], dtype=torch.int32),
        "ragged_offset_k": torch.tensor([0, 4, 8], dtype=torch.int32),
    }
    for misaligned in (False, True):
        tensors = {
            n: strided_tensor(
                packed((getattr(req, n).volume,), "fp16"), misaligned=misaligned
            )
            for n in ("q", "k", "v", "o")
        }
        plan = _LaunchPlan(req, {**tensors, **offsets})
        assert plan.q_inplace is plan.o_inplace is (not misaligned)
        assert plan.q_packed.data_ptr() % 16 == 0
        assert plan.o_packed.data_ptr() % 16 == 0


def test_seq_len_of_another_integer_dtype_is_refused():
    """The request names int32 lengths; int64 ones are refused, not narrowed."""
    req = replace(
        make_request(d=64, s_q=1, s_kv=16),
        padding_mask=True,
        seq_len_q=packed((2, 1, 1, 1), "int32"),
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
    )
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["seq_len_q"] = torch.tensor([1, 1], dtype=torch.int64)
    tensors["seq_len_kv"] = torch.tensor([16, 16], dtype=torch.int32)
    with pytest.raises(ValueError, match="seq_len_q must be torch.int32"):
        _LaunchPlan(req, tensors)


def test_stats_of_another_dtype_are_refused():
    """FP32 stats requested: FP16 ones would get the LSE silently narrowed."""
    req = replace(make_request(d=64), stats=packed((2, 4, 8, 1), "fp32"))
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["stats"] = torch.zeros(req.stats.dims, dtype=torch.float16)
    with pytest.raises(ValueError, match="stats must be torch.float32"):
        _LaunchPlan(req, tensors)


# --- K/V in place --------------------------------------------------------------
#
# Full-length packed K/V already is the identity-table paged cache, so the plan
# hands it to the kernel as is. Anything that would leave a caller's row past a
# sequence's length inside a kernel tile (padding, a partial last page) is
# copied into the zero-filled cache instead: a masked row still reaches P @ V.

_S_KV = 32  # two pages of the default 16-row KV block


def _kv_plan(*, misaligned=False, **overrides):
    req = make_request(d=64, s_kv=_S_KV, **overrides)
    tensors = {
        n: strided_tensor(getattr(req, n), misaligned=misaligned)
        for n in ("q", "k", "v", "o")
    }
    return req, tensors, _LaunchPlan(req, tensors)


def test_full_length_packed_bshd_k_and_v_are_the_paged_cache():
    req, tensors, plan = _kv_plan()
    assert plan.k_inplace and plan.v_inplace
    assert plan.k_cache.data_ptr() == tensors["k"].data_ptr()
    assert plan.v_cache.data_ptr() == tensors["v"].data_ptr()
    pages = _S_KV // plan.block
    for b in range(req.batch):
        for p in range(pages):
            rows = slice(p * plan.block, (p + 1) * plan.block)
            slot = int(plan.block_table[b, p])
            assert torch.equal(
                plan.k_cache[slot], tensors["k"][b, :, rows].transpose(0, 1)
            )
            assert torch.equal(
                plan.v_cache[slot], tensors["v"][b, :, rows].transpose(0, 1)
            )


@pytest.mark.parametrize(
    "why,kw",
    [
        ("bhsd", {"layout": bhsd}),
        ("partial_last_page", {"s_kv": _S_KV + 8}),
        ("misaligned", {"misaligned": True}),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_k_and_v_are_copied_into_a_zeroed_cache_otherwise(why, kw):
    misaligned = kw.pop("misaligned", False)
    s_kv = kw.pop("s_kv", _S_KV)
    req = make_request(d=64, s_kv=s_kv, **kw)
    tensors = {
        n: strided_tensor(getattr(req, n), misaligned=misaligned)
        for n in ("q", "k", "v", "o")
    }
    plan = _LaunchPlan(req, tensors)
    assert not plan.k_inplace and not plan.v_inplace
    assert torch.all(plan.k_cache == 0) and torch.all(plan.v_cache == 0)


def test_padded_k_and_v_are_copied_so_the_tail_reads_zero():
    """With a sequence shorter than S_kv the caller's tail rows would sit in
    the kernel's last tile; poison them and check none reaches the cache."""
    req = replace(
        make_request(d=64, s_kv=_S_KV),
        padding_mask=True,
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
    )
    tensors = {n: strided_tensor(getattr(req, n)) for n in ("q", "k", "v", "o")}
    tensors["k"][:, :, 20:] = float("nan")
    tensors["v"][:, :, 20:] = float("nan")
    tensors["seq_len_kv"] = torch.tensor([20, _S_KV], dtype=torch.int32).view(
        2, 1, 1, 1
    )
    plan = _LaunchPlan(req, tensors)
    assert not plan.k_inplace and not plan.v_inplace
    plan.pack()
    assert not torch.isnan(plan.k_cache[: plan.pages]).any()
    assert not torch.isnan(plan.v_cache[: plan.pages]).any()


@pytest.mark.parametrize("lengths,inplace", [((_S_KV, _S_KV), True), ((24, 8), False)])
def test_ragged_k_and_v_are_used_in_place_only_at_full_length(lengths, inplace):
    def ragged(t):
        return replace(
            t,
            ragged_offset=ragged_offset(t.dims[0]),
            ragged_offset_multiplier=t.strides[2],
        )

    req = make_request(d=64, s_q=_S_KV, s_kv=_S_KV)
    req = replace(req, **{n: ragged(getattr(req, n)) for n in ("q", "k", "v", "o")})
    starts = [0, lengths[0], lengths[0] + lengths[1]]
    offsets = {
        "ragged_offset_q": torch.tensor([0, _S_KV, 2 * _S_KV], dtype=torch.int32),
        "ragged_offset_k": torch.tensor(starts, dtype=torch.int32),
    }
    tensors = {
        n: strided_tensor(packed((getattr(req, n).volume,), "fp16"))
        for n in ("q", "k", "v", "o")
    }
    plan = _LaunchPlan(req, {**tensors, **offsets})
    assert plan.k_inplace is plan.v_inplace is inplace
    if inplace:
        assert plan.k_cache.data_ptr() == tensors["k"].data_ptr()


def test_the_cpu_table_binds_k_and_v_both_in_place_and_copied():
    """``test_binding_moves_data_faithfully`` runs both K/V paths."""
    seen = set()
    for case in _CASES:
        req = request_from_case(case, ARCH)
        tensors = bind_tensors(case, H.to_torch(H.materialize(case), device="cpu"), req)
        seen.add(_LaunchPlan(req, tensors).k_inplace)
        if seen == {True, False}:
            return
    pytest.fail(f"the CPU case table binds K only with k_inplace in {seen}")
