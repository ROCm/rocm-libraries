# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA requests the gfx942 and gfx950 kernels decline, refused on a live device.

A refused launch must raise a typed error and leave its outputs untouched: every
output starts as NaN, so a kernel that ran anyway shows up as written values.
Each family has controls that drop the offending feature and must run and match
the float64 reference, so the refusal, not a broken harness, is what stops the
launch.

* Dense softmax scale: the dispatched dense binding refuses a scale outside
  ``[DENSE_MIN_SCALE, DENSE_MAX_SCALE]`` (NaN and infinity as invalid input,
  zero and negatives as a kernel limit), whether it comes at bind or at launch.
* Unified attention with a forced backend: what the split-KV (``3d``), tiled
  and scalar paths do not implement (LSE with sinks, LSE on the scalar path,
  ALiBi under a non-causal mask, a non-default mask on gfx942, an unsupported
  head size, LSE on strided-KV decode).

The host-only scale contract is ``dispatch/attention/test_attention_dense_scale.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from kernels.common import attention_unified as au
from kernels.common.attention_dense_spec import DENSE_MAX_SCALE, DENSE_MIN_SCALE
from rocke.runtime.device_array import DeviceArray
from rocke.tests.gpu import current_device_arch_in

from ._attention_tolerance import LSE_TOLERANCE, attention_tolerance
from ._attention_unified_support import paged_layout, random_half
from ._dense_numeric_gpu import (
    DENSE_GRID_SPEC_ID,
    assert_close,
    dense_reference,
    filled,
    operand,
    upload,
    values,
)
from ._dense_paged_stand_ins import pinned


@pytest.fixture
def arch():
    """The current GPU's arch; skips unless it is gfx942 or gfx950."""
    return current_device_arch_in(
        tuple(DENSE_GRID_SPEC_ID), subject="the SDPA declines"
    )


def _assert_untouched(*outputs):
    """Every given output still holds only the NaN it started with."""
    for output in outputs:
        if output is not None:
            assert np.isnan(values(output)).all(), "a refused launch wrote its output"


# --------------------------------------------------------------- dense scale ----

_DENSE_SHAPE = (1, 512, 16, 64)  # (batch, seq, heads, head size), causal MHA

_BAD_SCALES = [
    pytest.param(float("nan"), ValueError, "finite", id="nan"),
    pytest.param(float("inf"), ValueError, "finite", id="inf"),
    pytest.param(float("-inf"), ValueError, "finite", id="-inf"),
    pytest.param(0.0, NotImplementedError, "NOT_YET_IMPLEMENTED", id="zero"),
    pytest.param(-0.0, NotImplementedError, "NOT_YET_IMPLEMENTED", id="neg-zero"),
    pytest.param(-0.125, NotImplementedError, "NOT_YET_IMPLEMENTED", id="negative"),
    pytest.param(
        math.nextafter(DENSE_MIN_SCALE, 0.0),
        NotImplementedError,
        "NOT_YET_IMPLEMENTED",
        id="below-min",
    ),
    pytest.param(
        math.nextafter(DENSE_MAX_SCALE, math.inf),
        NotImplementedError,
        "NOT_YET_IMPLEMENTED",
        id="above-max",
    ),
]


def _dense_binding(arch, **bind_options):
    """The default-grid dense kernel bound to seeded fp16 operands and a NaN
    output; returns ``(binding, out, (q, k, v))``."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    batch, seq, heads, head_size = _DENSE_SHAPE
    request = AttentionRequest(
        batch=batch,
        nhead_q=heads,
        nhead_k=heads,
        seqlen_q=seq,
        seqlen_k=seq,
        hdim_q=head_size,
        hdim_v=head_size,
        arch=arch,
        mask_type=1,
        dtype="fp16",
    )
    result = dispatch_attention(pinned(request, DENSE_GRID_SPEC_ID[arch]))
    rng = np.random.default_rng(0)
    q, k, v = (operand(rng, _DENSE_SHAPE, "fp16") for _ in range(3))
    out = filled(_DENSE_SHAPE, "fp16")
    binding = result.bind_torch({"q": q, "k": k, "v": v, "out": out}, **bind_options)
    return binding, out, (q, k, v)


def _assert_dense_matches(out, qkv, scale):
    reference, _ = dense_reference(*qkv, scale)
    assert_close(out, reference, attention_tolerance("fp16"))


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1 / math.sqrt(_DENSE_SHAPE[-1]), 0.25])
def test_dense_control_scale_runs(arch, scale):
    binding, out, qkv = _dense_binding(arch)
    binding.launch(scale=scale)
    _assert_dense_matches(out, qkv, scale)


@pytest.mark.gpu
@pytest.mark.parametrize("scale,error,match", _BAD_SCALES)
def test_dense_launch_scale_refused_without_a_launch(arch, scale, error, match):
    """A bad scale at launch is refused before the kernel runs; the same binding
    then launches with the default scale."""
    binding, out, qkv = _dense_binding(arch)
    with pytest.raises(error, match=match):
        binding.launch(scale=scale)
    _assert_untouched(out)
    binding.launch()
    _assert_dense_matches(out, qkv, 1 / math.sqrt(_DENSE_SHAPE[-1]))


@pytest.mark.gpu
@pytest.mark.parametrize("scale,error,match", _BAD_SCALES)
def test_dense_bind_scale_refused_without_a_launch(arch, scale, error, match):
    binding, out, _ = _dense_binding(arch, scale=scale)
    with pytest.raises(error, match=match):
        binding.launch()
    _assert_untouched(out)


# ----------------------------------------------------- unified, forced backend ----

_HQ, _HKV, _HEAD_SIZE, _BLOCK_SIZE = 8, 2, 64, 16
_KV_LENS = (40, 33)  # one decode query per sequence


class _DecodeScene:
    """A decode batch of two sequences over a paged cache whose unused table
    entries point at a NaN page, so a read past a sequence's keys shows."""

    def __init__(self, *, head_size=_HEAD_SIZE, lse=False, **problem_options):
        rng = np.random.default_rng(0)
        capacity = max(_KV_LENS)
        self.q = random_half(rng, (len(_KV_LENS), _HQ, head_size), "fp16")
        self.k, self.v = (
            random_half(rng, (len(_KV_LENS), _HKV, capacity, head_size), "fp16")
            for _ in range(2)
        )
        self.k_pages, self.table = paged_layout(
            self.k, _BLOCK_SIZE, poison_after=_KV_LENS
        )
        self.v_pages, _ = paged_layout(self.v, _BLOCK_SIZE, poison_after=_KV_LENS)
        self.problem = au.UnifiedAttentionProblem(
            total_q=len(_KV_LENS),
            num_seqs=len(_KV_LENS),
            num_query_heads=_HQ,
            num_kv_heads=_HKV,
            head_size=head_size,
            block_size=_BLOCK_SIZE,
            max_seqlen_q=1,
            max_seqlen_k=capacity,
            dtype="fp16",
            emit_lse=lse,
            **problem_options,
        )
        self.out = filled(self.q.shape, "fp16")
        self.lse = filled((len(_KV_LENS), _HQ, 1), "fp32") if lse else None

    def launch(self, *, backend, sinks=False, alibi=False, **options):
        return au.run_unified_attention_torch(
            problem=self.problem,
            q=upload(self.q, "fp16"),
            k=upload(self.k_pages, "fp16"),
            v=upload(self.v_pages, "fp16"),
            out=self.out,
            cu_seqlens_q=DeviceArray(np.arange(len(_KV_LENS) + 1, dtype=np.int32)),
            seqused_k=DeviceArray(np.array(_KV_LENS, dtype=np.int32)),
            softmax_scale=1.0 / math.sqrt(self.q.shape[-1]),
            block_table=DeviceArray(self.table),
            softcap=0.0,
            sinks=upload(np.zeros(_HQ), "fp16") if sinks else None,
            alibi_slopes=(
                DeviceArray(np.full(_HQ, 0.0625, np.float32)) if alibi else None
            ),
            lse=self.lse,
            backend=backend,
            **options,
        )

    def assert_matches_reference(self):
        """O (and LSE) of the bottom-right causal decode, against the oracle."""

        def packed(logical):  # [B, Hkv, capacity, D] -> the visible keys, [T, Hkv, D]
            return np.concatenate(
                [logical[b, :, :n].transpose(1, 0, 2) for b, n in enumerate(_KV_LENS)]
            )

        out, lse = dense_reference(
            self.q,
            packed(self.k),
            packed(self.v),
            1.0 / math.sqrt(self.q.shape[-1]),
            bottom_right=True,
            q_offsets=np.arange(len(_KV_LENS) + 1),
            kv_offsets=np.concatenate([[0], np.cumsum(_KV_LENS)]),
        )
        assert_close(self.out, out, attention_tolerance("fp16"))
        if self.lse is not None:
            assert_close(self.lse, lse, LSE_TOLERANCE)


_REFUSED = [
    pytest.param(
        {"backend": "3d", "lse": True, "use_sinks": True},
        {"sinks": True},
        "sinks",
        id="3d-lse-with-sinks",
    ),
    pytest.param(
        {"backend": "auto", "lse": True, "use_sinks": True},
        {"sinks": True},
        "sinks",
        id="auto-lse-with-sinks",
    ),
    pytest.param({"backend": "scalar", "lse": True}, {}, "LSE", id="scalar-lse"),
    pytest.param(
        {"backend": "3d", "use_alibi": True, "right_bound": -1},
        {"alibi": True},
        "only defined for causal",
        id="3d-alibi-non-causal",
    ),
    pytest.param(
        {"backend": "tiled", "use_alibi": True, "right_bound": -1},
        {"alibi": True},
        "only defined for causal",
        id="tiled-alibi-non-causal",
    ),
    pytest.param(
        {"backend": "3d", "head_size": 80}, {}, "head_size", id="3d-head-size-80"
    ),
    pytest.param(
        {"backend": "tiled", "head_size": 80}, {}, "head_size", id="tiled-head-size-80"
    ),
]


@pytest.mark.gpu
@pytest.mark.parametrize("setup,call,match", _REFUSED)
def test_unified_refusal_leaves_output_untouched(arch, setup, call, match):
    setup = dict(setup)
    backend = setup.pop("backend")
    scene = _DecodeScene(**setup)
    with pytest.raises(NotImplementedError, match=match):
        scene.launch(backend=backend, **call)
    _assert_untouched(scene.out, scene.lse)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["3d", "tiled"])
def test_gfx942_non_default_mask_refused(backend):
    current_device_arch_in(("gfx942",), subject="the gfx942 tiled mask refusal")
    scene = _DecodeScene(right_bound=-1)
    with pytest.raises(NotImplementedError, match="does not implement"):
        scene.launch(backend=backend)
    _assert_untouched(scene.out)


@pytest.mark.gpu
def test_strided_kv_lse_refused(arch):
    scene = _DecodeScene(lse=True)
    with pytest.raises(NotImplementedError, match="strided-KV decode"):
        scene.launch(backend="3d", kv_layout="strided")
    _assert_untouched(scene.out, scene.lse)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "setup",
    [
        pytest.param({"backend": "3d"}, id="3d"),
        pytest.param({"backend": "tiled", "lse": True}, id="tiled-lse"),
        pytest.param({"backend": "scalar"}, id="scalar"),
    ],
)
def test_unified_control_without_the_refused_feature_runs(arch, setup):
    setup = dict(setup)
    backend = setup.pop("backend")
    scene = _DecodeScene(**setup)
    scene.launch(backend=backend)
    scene.assert_matches_reference()
