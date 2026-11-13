# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host contract for the free-stride attention forward addressing layer.

The generalized forward kernel must read Q/K/V from any memory layout -- BSHD,
BHSD, general B/H/S strides and packed QKV -- addressed by per-tensor
``(batch, head, token)`` element strides with a unit head-dim stride (the
contract of the ``StridedTensorAddr`` layer).

These host tests pin that contract with no GPU and no kernel: they reconstruct
the logical ``(B, H, S, D)`` tensor from a backing buffer using the same scalar
element-address formula the kernel emits,

    addr = base + b*stride_batch + h*stride_head + s*stride_token + d*stride_dim

and confirm every layout recovers the same values and the same attention
output. This is exactly what the kernel ABI (the stride params we pass) has to
satisfy, so it is the host reference the emitter is matched against.

These are a contract check on the addressing formula, not on the kernel: they
gather through numpy and would pass with no kernel at all. The padded and
packed layouts are exercised against the real kernel on a GPU by
``test_fmha_mfma_free_stride_numeric.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernels.common._sdpa_fwd_strides import contiguous_strides
from rocke.numeric.references import dense_attention_reference


def _addr_gather(flat, base, strides, shape):
    """Reconstruct a tensor of ``shape`` from a flat element buffer using the
    kernel's scalar address formula ``base + sum(idx_i * stride_i)``."""
    s_b, s_h, s_t, s_d = strides
    g = np.indices(shape)
    addr = base + g[0] * s_b + g[1] * s_h + g[2] * s_t + g[3] * s_d
    return flat[addr]


def _view_base_strides(view, backing):
    """Element ``(flat, base, strides)`` of ``view`` into a C-contiguous
    ``backing``: byte offsets and strides divided by the item size."""
    it = backing.itemsize
    base = (view.ctypes.data - backing.ctypes.data) // it
    strides = tuple(s // it for s in view.strides)
    return backing.reshape(-1), base, strides


def _logical(B=2, H=4, S=16, D=16, seed=0xF00D):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16)


# Each builder returns ``(view, backing)`` where ``view`` presents the logical
# ``(B, H, S, D)`` order and ``backing`` is the C-contiguous buffer behind it.
def _bhsd(x):
    b = np.ascontiguousarray(x)  # [B, H, S, D]: token stride = D
    return b, b


def _bshd(x):
    # Store tokens-major [B, S, H, D]; the logical view transposes heads back.
    b = np.ascontiguousarray(x.transpose(0, 2, 1, 3))
    return b.transpose(0, 2, 1, 3), b


def _padded(x):
    # Pad the head dim: a non-contiguous slice, so batch/head/token strides all
    # carry the padding while the head dim keeps unit stride.
    B, H, S, D = x.shape
    b = np.zeros((B, H, S, D + 8), x.dtype)
    b[..., :D] = x
    return b[..., :D], b


LAYOUTS = {"bhsd": _bhsd, "bshd": _bshd, "padded": _padded}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layout_strides_recover_logical(name):
    x = _logical()
    view, backing = LAYOUTS[name](x)
    flat, base, strides = _view_base_strides(view, backing)
    # The addressing layer assumes a unit head-dim stride.
    assert strides[3] == 1, f"{name}: head-dim stride must be 1, got {strides[3]}"
    got = _addr_gather(flat, base, strides, x.shape)
    np.testing.assert_array_equal(got, x)


def test_packed_qkv_views_recover_logical():
    # Q, K, V interleaved in one buffer [B, H, S, 3, D]; each is a strided view
    # with its own base offset (0, D, 2D) into the shared buffer.
    B, H, S, D = 2, 4, 16, 16
    rng = np.random.default_rng(0xBEE5)
    qkv = [
        (rng.standard_normal((B, H, S, D)) * 0.3).astype(np.float16) for _ in range(3)
    ]
    packed = np.zeros((B, H, S, 3, D), np.float16)
    for i, t in enumerate(qkv):
        packed[..., i, :] = t
    for i, want in enumerate(qkv):
        view = packed[..., i, :]
        flat, base, strides = _view_base_strides(view, packed)
        assert strides[3] == 1
        assert base == i * D
        np.testing.assert_array_equal(
            _addr_gather(flat, base, strides, (B, H, S, D)), want
        )


@pytest.mark.parametrize("name", ["bhsd", "bshd"])
def test_contiguous_strides_match_numpy_views(name):
    # The launcher's analytical (batch, head, token) strides must equal what
    # numpy computes for the same contiguous layout.
    x = _logical()
    _B, H, S, D = x.shape
    view, backing = LAYOUTS[name](x)
    _flat, _base, np_strides = _view_base_strides(view, backing)
    got = contiguous_strides(name, heads=H, seqlen=S, head_size=D)
    assert got == np_strides[:3]  # (batch, head, token); dim is unit, excluded
    assert np_strides[3] == 1


def test_contiguous_strides_rejects_bad_input():
    with pytest.raises(ValueError, match="layout"):
        contiguous_strides("thd", heads=4, seqlen=16, head_size=16)
    with pytest.raises(ValueError, match="heads"):
        contiguous_strides("bhsd", heads=0, seqlen=16, head_size=16)


def _attention_bhsd(ql, kl, vl, *, causal):
    """Attention over logical ``(B, H, S, D)`` operands; the reference consumes
    one batch of shape ``(S, H, D)``."""
    return np.stack(
        [
            dense_attention_reference(
                ql[b].transpose(1, 0, 2),
                kl[b].transpose(1, 0, 2),
                vl[b].transpose(1, 0, 2),
                causal=causal,
            )
            for b in range(ql.shape[0])
        ]
    )


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_attention_output_is_layout_invariant(name):
    ql, kl, vl = _logical(seed=1), _logical(seed=2), _logical(seed=3)
    ref = _attention_bhsd(ql, kl, vl, causal=False)

    def through(x):
        view, backing = LAYOUTS[name](x)
        flat, base, strides = _view_base_strides(view, backing)
        return _addr_gather(flat, base, strides, x.shape)

    got = _attention_bhsd(through(ql), through(kl), through(vl), causal=False)
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)
