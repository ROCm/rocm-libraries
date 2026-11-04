# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The reference forward (mode ref-fwd) against ATen's math SDPA on CPU."""

import math

import pytest
import torch

from aotriton_flash import reference_attention
from conftest import Mask

B, HQ, SQ, SK, D = 2, 4, 9, 11, 16


def additive(visible):
    return torch.zeros(visible.shape, dtype=torch.float64).masked_fill(
        ~visible, float("-inf")
    )


def band(mask, sq=SQ, sk=SK):
    """Visible positions from the AOTriton definitions, not from the shim's own band function."""
    i = torch.arange(sq)[:, None]
    j = torch.arange(sk)[None, :]
    if mask.alignment == "tl" and mask.left is None and mask.right == 0:
        return j <= i  # causal, top-left
    if mask.alignment == "br" and mask.left is None and mask.right == 0:
        return j - (sk - sq) <= i  # causal, bottom-right
    left, right = mask.left - 1, mask.right  # window (l, r): i - l <= j <= i + r
    return (j >= i - left) & (j <= i + right)


def aten(
    q, k, v, attn_mask=None, dropout_p=0.0, dropout_mask=None, scale=None, gqa=False
):
    return torch.ops.aten._scaled_dot_product_attention_math(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        dropout_mask=dropout_mask,
        scale=scale,
        enable_gqa=gqa,
    )[0]


def scores_lse(q, k, scale, add):
    kk = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    return torch.logsumexp(q @ kk.transpose(-1, -2) * scale + add, dim=-1)


@pytest.fixture
def qkv():
    torch.manual_seed(3)
    return (
        torch.randn(B, HQ, SQ, D, dtype=torch.float64),
        torch.randn(B, HQ, SK, D, dtype=torch.float64),
        torch.randn(B, HQ, SK, D, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "mask",
    [
        None,
        Mask("tl", None, 0),
        Mask("br", None, 0),
        Mask("tl", 4, 2),  # window (3, 2)
        Mask("tl", 1, 0),  # window (0, 0)
        Mask("tl", 6, 8),
    ],
)
def test_matches_aten_for_every_mask_form(qkv, mask):
    q, k, v = qkv
    scale = 0.25  # exactly representable: ATen applies sqrt(scale) to Q and K
    visible = torch.ones(SQ, SK, dtype=torch.bool) if mask is None else band(mask)
    add = additive(visible)
    out, lse = reference_attention(q, k, v, scale, mask)
    live = visible.any(-1)
    want = aten(q, k, v, attn_mask=add, scale=scale)
    assert torch.allclose(out[:, :, live], want[:, :, live], atol=1e-12)
    assert torch.allclose(
        lse[:, :, live].double(), scores_lse(q, k, scale, add)[:, :, live], atol=1e-5
    )


def test_rows_without_visible_keys_are_dead(qkv):
    """Bottom-right causal with Sq > Sk: the first Sq - Sk rows see nothing: O = 0, LSE = +inf."""
    q, k, v = qkv
    k, v = k[:, :, :5], v[:, :, :5]
    out, lse = reference_attention(q, k, v, 0.25, Mask("br", None, 0))
    dead = SQ - 5
    assert dead > 0
    assert (out[:, :, :dead] == 0).all() and torch.isposinf(lse[:, :, :dead]).all()
    assert torch.isfinite(lse[:, :, dead:]).all()
    visible = band(Mask("br", None, 0), SQ, 5)
    want = aten(q, k, v, attn_mask=additive(visible), scale=0.25)
    assert torch.allclose(out[:, :, dead:], want[:, :, dead:], atol=1e-12)


def test_bias_added_before_masking(qkv):
    q, k, v = qkv
    bias = torch.randn(B, HQ, SQ, SK, dtype=torch.float64)
    out, lse = reference_attention(q, k, v, 0.25, None, bias=bias)
    assert torch.allclose(out, aten(q, k, v, attn_mask=bias, scale=0.25), atol=1e-12)
    assert torch.allclose(lse.double(), scores_lse(q, k, 0.25, bias), atol=1e-5)


def test_gqa_maps_query_head_h_to_kv_head_h_over_group(qkv):
    q, k, v = qkv
    k, v = k[:, :2], v[:, :2]
    out, lse = reference_attention(q, k, v, 0.25, Mask("tl", None, 0))
    add = additive(band(Mask("tl", None, 0)))
    assert torch.allclose(
        out, aten(q, k, v, attn_mask=add, scale=0.25, gqa=True), atol=1e-12
    )
    assert torch.allclose(lse.double(), scores_lse(q, k, 0.25, add), atol=1e-5)
    assert torch.allclose(
        out[:, 1],
        reference_attention(q[:, 1:2], k[:, :1], v[:, :1], 0.25, Mask("tl", None, 0))[
            0
        ][:, 0],
    )


def test_dropout_keeps_scaled_probabilities_and_undropped_lse(qkv):
    q, k, v = qkv
    p = 0.3
    keep = torch.rand(B, HQ, SQ, SK, generator=torch.Generator().manual_seed(1)) > p
    out, lse = reference_attention(q, k, v, 0.25, None, keep=keep, p=p)
    want = aten(q, k, v, scale=0.25, dropout_p=p, dropout_mask=keep)
    assert torch.allclose(out, want, atol=1e-12)
    assert torch.allclose(lse.double(), scores_lse(q, k, 0.25, 0.0), atol=1e-5)


def test_low_precision_inputs_accumulate_in_fp32_and_cast_back():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 16, dtype=torch.float16) for _ in range(3))
    out, lse = reference_attention(q, k, v, 1 / math.sqrt(16), Mask("tl", None, 0))
    assert out.dtype == torch.float16 and lse.dtype == torch.float32
    want = aten(
        q.double(),
        k.double(),
        v.double(),
        attn_mask=additive(band(Mask("tl", None, 0), 8, 8)),
        scale=1 / math.sqrt(16),
    )
    assert torch.allclose(out.double(), want, atol=2e-3)
