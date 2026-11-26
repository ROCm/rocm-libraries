# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""SDPA backward under ``torch.autocast`` (ALMIOPEN-2978): fp32 inputs, and
inputs mixed across dtypes, under bf16 autocast. Marked ``gpu`` and
auto-skipped by ``conftest.py`` unless a wired provider + HIP device is
present.

Under CUDA autocast, the override casts every CUDA floating tensor argument
(q, k, v, a float ``attn_mask``) that is not fp64 and not already the
autocast dtype to ``torch.get_autocast_dtype("cuda")`` with a differentiable
``.to`` (so grads return in the INPUT dtype), per tensor -- not from one
representative operand.
"""

import contextlib
import os

import pytest

pytestmark = pytest.mark.gpu

_TOL = dict(rtol=4e-2, atol=4e-2)
_GRAD_PREFIX = "grad "
_BWD_PREFIX = "bwd "


def _census_key(prefix, sq, skv, d, dtype_tok):
    return f"{prefix}Sq={sq},Skv={skv},D={d},dtype={dtype_tok}"


def _grad_row(ov, sq, skv, d, dtype_tok):
    return ov.census().get(_census_key(_GRAD_PREFIX, sq, skv, d, dtype_tok))


def _bwd_row(ov, sq, skv, d, dtype_tok):
    return ov.census().get(_census_key(_BWD_PREFIX, sq, skv, d, dtype_tok))


def _flydsl_state():
    v = os.environ.get("HIPDNN_TORCH_TEST_FLYDSL")
    return v.lower() if v else None


def _require_known_build():
    if _flydsl_state() is None:
        pytest.skip(
            "HIPDNN_TORCH_TEST_FLYDSL unset; whether this shape is served "
            "depends on which engine(s) the build loads"
        )


@contextlib.contextmanager
def _routed_sdpa():
    import hipdnn_torch

    ov = hipdnn_torch.overrides()["sdpa"]
    hipdnn_torch.install(["sdpa"])
    ov.reset()
    try:
        yield ov
    finally:
        hipdnn_torch.uninstall(["sdpa"])


def _qkv_bshd(torch, b, sq, skv, hq, hkv, d, dtype):
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(b, skv, hkv, d, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(b, skv, hkv, d, device="cuda", dtype=dtype).transpose(1, 2)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    return q, k, v


def _clone_leaf(t):
    return t.detach().clone().requires_grad_(t.requires_grad)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Fp32 inputs under autocast                                               #
# --------------------------------------------------------------------------- #
def test_fp32_inputs_under_autocast_census_dtype_is_autocast_dtype(torch):
    """Fp32 inputs under ``torch.autocast("cuda", dtype=bf16)``: the
    census key's dtype is the AUTOCAST dtype (bf16), not the caller's fp32,
    and grads come back in the input dtype (fp32, the differentiable
    ``.to`` the cast uses). MHA, D=128, no mask, p=0: an unmasked shape
    every build serves."""
    _require_known_build()
    import torch.nn.functional as F

    b, hq, sq, skv, d = 2, 4, 64, 64, 128
    q, k, v = _qkv_bshd(torch, b, sq, skv, hq, hq, d, torch.float32)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa() as ov:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = F.scaled_dot_product_attention(q, k, v)
        out.float().sum().backward()

    row = _grad_row(ov, sq, skv, d, "bf16")
    assert row is not None and row["aot"] > 0, "autocast dtype must key the census row"
    assert q.grad.dtype == torch.float32
    assert k.grad.dtype == torch.float32
    assert v.grad.dtype == torch.float32

    with torch.autocast("cuda", dtype=torch.bfloat16):
        want = F.scaled_dot_product_attention(q2, k2, v2)
    want.float().sum().backward()

    torch.testing.assert_close(out.float(), want.float(), **_TOL)
    for g, gn in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        torch.testing.assert_close(g.float(), gn.float(), **_TOL)


# --------------------------------------------------------------------------- #
# Mixed-dtype autocast input                                             #
# --------------------------------------------------------------------------- #
def test_mixed_dtype_autocast_input_casts_per_tensor(torch):
    """Q, k fp32 and v bf16 under bf16 autocast: the cast is decided
    PER TENSOR (a cast decided from q alone would be the bug this catches)
    -- census dtype bf16, a served grad row and a served bwd row, dq/dk
    fp32, dv bf16, and all three grads equal native autocast SDPA's. MHA,
    D=128, no mask, p=0: an unmasked shape every build serves."""
    _require_known_build()
    import torch.nn.functional as F

    b, hq, sq, skv, d = 2, 4, 64, 64, 128

    def _mk(s, dtype):
        t = torch.randn(b, s, hq, d, device="cuda", dtype=dtype).transpose(1, 2)
        t.requires_grad_(True)
        return t

    q, k, v = _mk(sq, torch.float32), _mk(skv, torch.float32), _mk(skv, torch.bfloat16)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa() as ov:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = F.scaled_dot_product_attention(q, k, v)
        out.float().sum().backward()

    row = _grad_row(ov, sq, skv, d, "bf16")
    bwd = _bwd_row(ov, sq, skv, d, "bf16")
    assert row is not None and row["aot"] > 0
    assert bwd is not None and bwd["aot"] > 0
    assert q.grad.dtype == torch.float32
    assert k.grad.dtype == torch.float32
    assert v.grad.dtype == torch.bfloat16

    with torch.autocast("cuda", dtype=torch.bfloat16):
        want = F.scaled_dot_product_attention(q2, k2, v2)
    want.float().sum().backward()

    torch.testing.assert_close(out.float(), want.float(), **_TOL)
    torch.testing.assert_close(q.grad.float(), q2.grad.float(), **_TOL)
    torch.testing.assert_close(k.grad.float(), k2.grad.float(), **_TOL)
    torch.testing.assert_close(v.grad.float(), v2.grad.float(), **_TOL)


def test_mixed_dtype_autocast_q_already_cast_kv_still_cast(torch):
    """Q already bf16 (the autocast dtype, so no cast is applied to it)
    and k, v fp32 (so both need casting) under bf16 autocast: proves the
    cast is decided PER TENSOR, not from q alone -- with q fp32 (the other
    direction, covered above), a cast decided from q's dtype alone would
    still correctly cast k too, since k matches q's dtype; only this q
    bf16 / k,v fp32 combination catches a cast wrongly decided from q and
    skipped for k/v because q already looked like the autocast dtype --
    census dtype bf16, a served grad row and a served bwd row, dq bf16
    (q itself is never cast, so autograd returns its own dtype), dk/dv
    fp32 (the differentiable cast returns grads in the INPUT dtype), and
    all three grads equal native autocast SDPA's. MHA, D=128, no mask,
    p=0: an unmasked shape every build serves."""
    _require_known_build()
    import torch.nn.functional as F

    b, hq, sq, skv, d = 2, 4, 64, 64, 128

    def _mk(s, dtype):
        t = torch.randn(b, s, hq, d, device="cuda", dtype=dtype).transpose(1, 2)
        t.requires_grad_(True)
        return t

    q, k, v = _mk(sq, torch.bfloat16), _mk(skv, torch.float32), _mk(skv, torch.float32)
    q2, k2, v2 = _clone_leaf(q), _clone_leaf(k), _clone_leaf(v)

    with _routed_sdpa() as ov:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = F.scaled_dot_product_attention(q, k, v)
        out.float().sum().backward()

    row = _grad_row(ov, sq, skv, d, "bf16")
    bwd = _bwd_row(ov, sq, skv, d, "bf16")
    assert row is not None and row["aot"] > 0
    assert bwd is not None and bwd["aot"] > 0
    assert q.grad.dtype == torch.bfloat16
    assert k.grad.dtype == torch.float32
    assert v.grad.dtype == torch.float32

    with torch.autocast("cuda", dtype=torch.bfloat16):
        want = F.scaled_dot_product_attention(q2, k2, v2)
    want.float().sum().backward()

    torch.testing.assert_close(out.float(), want.float(), **_TOL)
    torch.testing.assert_close(q.grad.float(), q2.grad.float(), **_TOL)
    torch.testing.assert_close(k.grad.float(), k2.grad.float(), **_TOL)
    torch.testing.assert_close(v.grad.float(), v2.grad.float(), **_TOL)


def test_mixed_dtype_autocast_real_qwen3_layer(torch):
    """A real Qwen3 attention layer with fp32 parameters under bf16
    autocast: rotary cos/sin promote q, k to fp32 while v stays bf16 -- the
    same per-tensor cast must apply end to end, reaching a served bwd row.
    The layer is GQA (4:1), so a served backward needs the kernel ingestor
    (the only engine whose backward builder accepts GQA): skip cleanly
    unless ``HIPDNN_TORCH_TEST_FLYDSL=on``."""
    if _flydsl_state() != "on":
        pytest.skip(
            "HIPDNN_TORCH_TEST_FLYDSL not 'on'; this GQA layer's backward is "
            "only served by the kernel-ingestor engine (the other backward "
            "builder rejects GQA)"
        )
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    config = Qwen3Config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=1,
        attn_implementation="sdpa",
    )
    layer = Qwen3Attention(config, layer_idx=0).to(device="cuda", dtype=torch.float32)
    b, s = 2, 16
    hidden = torch.randn(b, s, config.hidden_size, device="cuda", dtype=torch.float32)
    cos = torch.ones(b, s, config.head_dim, device="cuda", dtype=torch.float32)
    sin = torch.zeros(b, s, config.head_dim, device="cuda", dtype=torch.float32)

    with _routed_sdpa() as ov:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = layer(hidden, (cos, sin), None)
        out.float().sum().backward()

    assert any(p.grad is not None for p in layer.parameters())
    bwd_rows = {k: v for k, v in ov.census().items() if k.startswith(_BWD_PREFIX)}
    assert bwd_rows, "the layer's SDPA call never reached a served bwd row"
    assert all(
        row["aot"] > 0 and row["native"] == 0 for row in bwd_rows.values()
    ), bwd_rows
