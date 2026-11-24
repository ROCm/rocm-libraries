# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA requests the gfx942/gfx950 kernels decline, refused on a live device.

A refused launch must raise a typed error and leave the output buffer
untouched: every output is pre-filled with NaN, so a kernel that ran anyway
shows up as written values. Each refusal has a control that drops the
offending feature and must run and match the float64 reference, so the
refusal is what stops the launch and not a broken harness.

Two families are covered:

* dense softmax scale: the dispatched dense binding refuses a scale outside
  [2**-64, 2**4] (NaN and infinity as invalid input, zero and negatives as a
  kernel limit), at bind and at launch;
* unified attention with a forced backend: combinations the split-KV, tiled and
  scalar paths do not implement (LSE with sinks, LSE on the scalar path,
  ALiBi under a non-causal mask, a non-default mask on gfx942, an unsupported
  head size, LSE on strided-KV decode).

Torch-free. The host-only scale contract lives in
``dispatch/attention/test_attention_dense_scale.py``; the ASM engine's own
declines are C++ and are tested there.
"""

import math

import numpy as np
import pytest
from kernels.common import attention_unified as au
from kernels.common.attention_dense_spec import DENSE_MAX_SCALE, DENSE_MIN_SCALE
from rocke.core.arch.target import known_arches
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray
from ._dense_paged_stand_ins import pinned

_ARCH = get_device_arch()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH not in ("gfx942", "gfx950") or _ARCH not in known_arches(),
        reason=f"needs a gfx942 or gfx950 GPU, found {_ARCH}",
    ),
]


@pytest.fixture(scope="module")
def rt():
    return an.runtime()


# ---------------------------------------------------------------- dense scale

_DENSE_SPEC = {"gfx942": "gfx942_dense", "gfx950": "gfx950_dense_grid"}
_B, _S, _H, _D = 1, 512, 16, 64

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


def _dense_binding(rt, **bind_options):
    from dispatch.attention import AttentionRequest, dispatch_attention

    request = AttentionRequest(
        batch=_B,
        nhead_q=_H,
        nhead_k=_H,
        seqlen_q=_S,
        seqlen_k=_S,
        hdim_q=_D,
        hdim_v=_D,
        arch=_ARCH,
        mask_type=1,
        dtype="fp16",
    )
    result = dispatch_attention(pinned(request, _DENSE_SPEC[_ARCH]))
    rng = np.random.default_rng(0)
    q, k, v = (an.randn(rng, (_B, _S, _H, _D), "fp16") for _ in range(3))
    out = an.empty_device(rt, q.shape, "fp16")
    tensors = {
        "q": an.to_device(rt, q, "fp16"),
        "k": an.to_device(rt, k, "fp16"),
        "v": an.to_device(rt, v, "fp16"),
        "out": out,
    }
    return result.bind_torch(tensors, **bind_options), out, (q, k, v)


def _dense_reference(qkv, scale):
    q, k, v = (np.swapaxes(x, 1, 2) for x in qkv)
    o, _ = sdpa_reference(q, k, v, scale=scale, causal=True)
    return np.swapaxes(o, 1, 2)


@pytest.mark.parametrize("scale", [1 / math.sqrt(_D), 0.25])
def test_dense_control_scale_runs(rt, scale):
    binding, out, qkv = _dense_binding(rt)
    binding.launch(scale=scale)
    an.assert_close(an.read(out, "fp16"), _dense_reference(qkv, scale), "fp16")


@pytest.mark.parametrize("scale,error,match", _BAD_SCALES)
def test_dense_launch_scale_refused_without_a_launch(rt, scale, error, match):
    binding, out, qkv = _dense_binding(rt)
    with pytest.raises(error, match=match):
        binding.launch(scale=scale)
    assert np.isnan(an.read(out, "fp16")).all(), "output written by a refused launch"
    binding.launch()
    an.assert_close(
        an.read(out, "fp16"), _dense_reference(qkv, 1 / math.sqrt(_D)), "fp16"
    )


@pytest.mark.parametrize("scale,error,match", _BAD_SCALES)
def test_dense_bind_scale_refused_without_a_launch(rt, scale, error, match):
    binding, out, _ = _dense_binding(rt, scale=scale)
    with pytest.raises(error, match=match):
        binding.launch()
    assert np.isnan(an.read(out, "fp16")).all(), "output written by a refused launch"


# ------------------------------------------------------- unified, forced path

_HQ, _HKV, _UD, _BS = 8, 2, 64, 16
_Q_LENS, _KV_LENS = (1, 1), (40, 33)


class _Scene:
    """A decode batch of two sequences over a paged cache with a poison block."""

    def __init__(self, rt, *, head_size=_UD, lse=False, **problem_options):
        rng = np.random.default_rng(0)
        d = head_size
        needed = [math.ceil(n / _BS) for n in _KV_LENS]
        blocks = sum(needed) + 1
        order = iter(rng.permutation(blocks - 1))
        self.table = np.full((len(_KV_LENS), max(needed)), blocks - 1, dtype=np.int32)
        for s, n in enumerate(needed):
            self.table[s, :n] = [next(order) for _ in range(n)]
        self.q = an.randn(rng, (sum(_Q_LENS), _HQ, d), "fp16")
        self.k = an.randn(rng, (blocks, _BS, _HKV, d), "fp16")
        self.v = an.randn(rng, (blocks, _BS, _HKV, d), "fp16")
        self.problem = au.UnifiedAttentionProblem(
            total_q=sum(_Q_LENS),
            num_seqs=len(_Q_LENS),
            num_query_heads=_HQ,
            num_kv_heads=_HKV,
            head_size=d,
            block_size=_BS,
            max_seqlen_q=max(_Q_LENS),
            max_seqlen_k=max(_KV_LENS),
            dtype="fp16",
            emit_lse=lse,
            **problem_options,
        )
        self.rt = rt
        self.out = an.empty_device(rt, self.q.shape, "fp16")
        self.lse = an.lse_device(rt, (sum(_Q_LENS), _HQ, 1)) if lse else None

    def launch(self, *, backend, sinks=False, alibi=False, **extra):
        rt = self.rt
        cu = np.array([0, 1, 2], dtype=np.int32)
        d = self.q.shape[-1]
        return au.run_unified_attention_torch(
            problem=self.problem,
            q=an.to_device(rt, self.q, "fp16"),
            k=an.to_device(rt, self.k, "fp16"),
            v=an.to_device(rt, self.v, "fp16"),
            out=self.out,
            cu_seqlens_q=DeviceArray(rt, cu),
            seqused_k=DeviceArray(rt, np.array(_KV_LENS, dtype=np.int32)),
            softmax_scale=1.0 / math.sqrt(d),
            block_table=DeviceArray(rt, self.table),
            softcap=0.0,
            sinks=(
                an.to_device(rt, np.zeros(_HQ, np.float32), "fp16") if sinks else None
            ),
            alibi_slopes=(
                DeviceArray(rt, np.full(_HQ, 0.0625, np.float32)) if alibi else None
            ),
            lse=self.lse,
            backend=backend,
            **extra,
        )

    def reference(self):
        d = self.q.shape[-1]
        out = np.zeros(self.q.shape, dtype=np.float64)
        lse = np.zeros((self.q.shape[0], _HQ, 1), dtype=np.float64)
        for s, kv_len in enumerate(_KV_LENS):
            blocks = self.table[s, : math.ceil(kv_len / _BS)]
            keys = self.k[blocks].reshape(-1, _HKV, d)[:kv_len]
            values = self.v[blocks].reshape(-1, _HKV, d)[:kv_len]
            o, lse_s = sdpa_reference(
                self.q[s : s + 1].transpose(1, 0, 2)[None],
                keys.transpose(1, 0, 2)[None],
                values.transpose(1, 0, 2)[None],
                causal_bottom_right=True,
            )
            out[s] = o[0, :, 0]
            lse[s, :, 0] = lse_s[0, :, 0]
        return out, lse


def _untouched(scene):
    assert np.isnan(an.read(scene.out, "fp16")).all(), "output written"
    if scene.lse is not None:
        assert np.isnan(scene.lse.read()).all(), "lse written"


def _matches(scene):
    out, lse = scene.reference()
    an.assert_close(an.read(scene.out, "fp16"), out, "fp16")
    if scene.lse is not None:
        an.assert_close(scene.lse.read(), lse, "fp16", "lse", atol=1e-2)


_REFUSED = [
    pytest.param(
        dict(backend="3d", lse=True, use_sinks=True),
        dict(sinks=True),
        NotImplementedError,
        "sinks",
        id="3d-lse-with-sinks",
    ),
    pytest.param(
        dict(backend="auto", lse=True, use_sinks=True),
        dict(sinks=True),
        NotImplementedError,
        "sinks",
        id="auto-lse-with-sinks",
    ),
    pytest.param(
        dict(backend="scalar", lse=True),
        {},
        NotImplementedError,
        "LSE",
        id="scalar-lse",
    ),
    pytest.param(
        dict(backend="3d", use_alibi=True, right_bound=-1),
        dict(alibi=True),
        NotImplementedError,
        "only defined for causal",
        id="3d-alibi-non-causal",
    ),
    pytest.param(
        dict(backend="tiled", use_alibi=True, right_bound=-1),
        dict(alibi=True),
        NotImplementedError,
        "only defined for causal",
        id="tiled-alibi-non-causal",
    ),
    pytest.param(
        dict(backend="3d", head_size=80),
        {},
        NotImplementedError,
        "head_size",
        id="3d-head-size-80",
    ),
    pytest.param(
        dict(backend="tiled", head_size=80),
        {},
        NotImplementedError,
        "head_size",
        id="tiled-head-size-80",
    ),
]


@pytest.mark.parametrize("setup,call,error,match", _REFUSED)
def test_unified_refusal_leaves_output_untouched(rt, setup, call, error, match):
    setup = dict(setup)
    backend = setup.pop("backend")
    scene = _Scene(rt, **setup)
    with pytest.raises(error, match=match):
        scene.launch(backend=backend, **call)
    _untouched(scene)


@pytest.mark.skipif(_ARCH != "gfx942", reason="gfx950 implements non-default masks")
@pytest.mark.parametrize("backend", ["3d", "tiled"])
def test_gfx942_non_default_mask_refused(rt, backend):
    scene = _Scene(rt, right_bound=-1)
    with pytest.raises(NotImplementedError, match="does not implement"):
        scene.launch(backend=backend)
    _untouched(scene)


def test_strided_kv_lse_refused(rt):
    scene = _Scene(rt, lse=True)
    with pytest.raises(NotImplementedError, match="strided-KV decode"):
        scene.launch(backend="3d", kv_layout="strided")
    _untouched(scene)


# The split-KV (3d) launcher allocates its scratch with torch, so its controls
# cannot run here; its refusals above raise before that point.
_CONTROLS = [
    pytest.param(dict(backend="tiled", lse=True), id="tiled-lse"),
    pytest.param(dict(backend="scalar"), id="scalar"),
]


@pytest.mark.parametrize("setup", _CONTROLS)
def test_unified_control_without_the_refused_feature_runs(rt, setup):
    setup = dict(setup)
    backend = setup.pop("backend")
    scene = _Scene(rt, **setup)
    scene.launch(backend=backend)
    _matches(scene)
