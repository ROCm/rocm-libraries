# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU case (q): real transformers attention layers through SDPA backward
(ALMIOPEN-2978). Skips cleanly (``pytest.importorskip``) when ``transformers``
is not installed -- it is not a dependency of the GPU provider suite.

One layer each of Qwen3 (D=128, GQA 4:1), a Qwen2-family config at 14 query /
2 kv heads / D=64 (GQA 7:1 -- the Qwen2.5-0.5B shape: hidden 896 = 14*64; no
checkpoint is downloaded, only the layer's shape), OLMo-2 (D=128, MHA) and
GPT-2 (D=64), each with ``use_cache=False``, ``use_cache=True`` and a padded
``attention_mask``, dropout 0, ``attn_implementation="sdpa"`` (a bare Config
otherwise defaults to eager -- ``AttentionInterface.get_interface(None, ...)``
never reaches SDPA). Per variant: ``bwd`` aot == calls and native == 0 when
``HIPDNN_TORCH_TEST_FLYDSL=on`` (FlyDSL serves every shape here: no GQA/D=64
/bias restriction), or the exact M18 decline reason when ``=off`` (GQA and
D=64-without-#16337 -> ``grad: bwd:``; any mask -> ``grad: fwd+stats:``;
unmasked OLMo-2, MHA D=128 -> served by ASM); grads match native autograd;
the recorded q/k/v strides equal the layout class of that variant computed
from the layer's config by DESIGN.md's M18 formulas (not the D=16/Hq=8/S=16
table it was measured at); a served ``bwd`` row's engine is a name, not hex.
"""

import contextlib
import os

import pytest

pytestmark = pytest.mark.gpu

_TOL = {"bf16": dict(rtol=4e-2, atol=4e-2)}
_GRAD_PREFIX = "grad "
_BWD_PREFIX = "bwd "


def _census_key(prefix, sq, skv, d, dtype_tok):
    return f"{prefix}Sq={sq},Skv={skv},D={d},dtype={dtype_tok}"


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


@contextlib.contextmanager
def _capture_sdpa_args():
    """Layer a stride-capturing spy ON TOP of the (already-installed)
    hipdnn_torch wrapper: HF calls land on the spy first (whatever
    ``functional.scaled_dot_product_attention`` currently is, after
    ``install()``), which records the actual args of the LAST call and
    delegates. Restoring happens via ``hipdnn_torch.uninstall()`` in the
    caller, which replaces the module attribute with the TRUE original
    ``_real`` it captured at install time -- unaffected by this temporary
    extra layer."""
    import torch.nn.functional as F

    captured = {}
    patched = F.scaled_dot_product_attention

    def _spy(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        captured["q_stride"] = tuple(query.stride())
        captured["k_stride"] = tuple(key.stride())
        captured["v_stride"] = tuple(value.stride())
        captured["enable_gqa"] = enable_gqa
        return patched(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    F.scaled_dot_product_attention = _spy
    try:
        yield captured
    finally:
        F.scaled_dot_product_attention = patched


def _rope_identity(torch, b, s, head_dim, device):
    """Identity rotation (cos=1, sin=0) in bf16 -- the dtype a real rotary
    embedding module returns (``cos.to(dtype=x.dtype)``). A fp32 cos/sin
    would promote q, k to fp32 through ``apply_rotary_pos_emb`` while v
    stayed bf16, changing the census dtype and breaking SDPA's dtype
    agreement outside autocast; this is NOT case (g2)'s deliberate
    mixed-dtype scenario (test_sdpa_autocast.py), which runs under autocast
    with fp32 model parameters instead."""
    cos = torch.ones(b, s, head_dim, device=device, dtype=torch.bfloat16)
    sin = torch.zeros(b, s, head_dim, device=device, dtype=torch.bfloat16)
    return cos, sin


def _bshd_stride(s, h, d):
    return (s * h * d, d, h * d, 1)


def _bhsd_contig_stride(h, s, d):
    return (h * s * d, s * d, d, 1)


def _gpt2_fused_stride(s, hq, d):
    return (s * 3 * hq * d, d, 3 * hq * d, 1)


def _flydsl_state():
    v = os.environ.get("HIPDNN_TORCH_TEST_FLYDSL")
    return v.lower() if v else None


def _pr16337_state():
    v = os.environ.get("HIPDNN_TORCH_TEST_16337")
    return v.lower() if v else None


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def transformers():
    return pytest.importorskip("transformers")


def _qwen3_layer(torch, transformers):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    config = Qwen3Config(
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=128,
        num_hidden_layers=1,
        attn_implementation="sdpa",
    )
    layer = Qwen3Attention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    return config, layer


def _qwen25_0_5b_layer(torch, transformers):
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    config = Qwen2Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        attn_implementation="sdpa",
    )
    layer = Qwen2Attention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    return config, layer


def _olmo2_layer(torch, transformers):
    from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention

    config = Olmo2Config(
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        attn_implementation="sdpa",
    )
    layer = Olmo2Attention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    return config, layer


def _gpt2_layer(torch, transformers):
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    config = GPT2Config(
        n_embd=512,
        n_head=8,
        n_layer=1,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        attn_implementation="sdpa",
    )
    layer = GPT2Attention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    return config, layer


# name -> (builder, head_dim, hq, hkv); hq != hkv marks a GQA arch.
_ARCHS = {
    "qwen3": (_qwen3_layer, 128, 4, 1),
    "qwen25_0_5b": (_qwen25_0_5b_layer, 64, 14, 2),
    "olmo2": (_olmo2_layer, 128, 4, 4),
    "gpt2": (_gpt2_layer, 64, 8, 8),
}


def _run_rope_layer(torch, layer, hidden, attention_mask, cache, head_dim):
    cos, sin = _rope_identity(
        torch, hidden.shape[0], hidden.shape[1], head_dim, hidden.device
    )
    return layer(hidden, (cos, sin), attention_mask, cache=cache)


def _run_layer(torch, arch, layer, hidden, attention_mask, cache):
    if arch == "gpt2":
        return layer(hidden, cache=cache, attention_mask=attention_mask)
    head_dim = {"qwen3": 128, "qwen25_0_5b": 64, "olmo2": 128}[arch]
    return _run_rope_layer(torch, layer, hidden, attention_mask, cache, head_dim)


def _assert_served_or_declined(ov, sq, skv, d, *, arch, masked):
    """(q)/M18: with FlyDSL ON every variant here is served; with FlyDSL OFF
    only unmasked MHA D=128 (olmo2) is served by ASM -- GQA archs and D=64
    without #16337 decline at the bwd stage, and a mask always declines at
    the fwd+stats stage (both ASM builders reject any bias)."""
    aot, native = ov.totals()
    bwd_aot, bwd_native = ov.bwd_totals()
    flydsl = _flydsl_state()
    if flydsl is None:
        pytest.skip(
            "HIPDNN_TORCH_TEST_FLYDSL unset; case (q) states a build-specific "
            "census expectation per variant (DESIGN.md M18)"
        )

    if flydsl == "on":
        assert (
            aot > 0 and native == 0
        ), f"{arch}: expected served, got aot={aot} native={native}"
        assert bwd_aot > 0 and bwd_native == 0
        row = ov.census()[_census_key(_GRAD_PREFIX, sq, skv, d, "bf16")]
        bwd = ov.census()[_census_key(_BWD_PREFIX, sq, skv, d, "bf16")]
        assert bwd["engines"] and all(not e.startswith("0x") for e in bwd["engines"])
        return row, bwd

    # FlyDSL OFF.
    if masked:
        assert aot == 0 and native > 0
        reasons = ov.fallback_reasons()
        assert any(r.startswith("grad: fwd+stats:") for r in reasons), reasons
        return None, None
    if arch in ("qwen3", "qwen25_0_5b"):  # GQA: ASM's bwd builder rejects it
        assert aot == 0 and native > 0
        reasons = ov.fallback_reasons()
        assert any(r.startswith("grad: bwd:") for r in reasons), reasons
        return None, None
    if arch == "gpt2":  # D=64: depends on hip-kernel-provider#16337
        pr16337 = _pr16337_state()
        if pr16337 is None:
            pytest.skip(
                "HIPDNN_TORCH_TEST_16337 unset; GPT-2's FlyDSL-OFF D=64 "
                "expectation depends on whether #16337 is in the build"
            )
        if pr16337 == "out":
            assert aot == 0 and native > 0
            reasons = ov.fallback_reasons()
            assert any(r.startswith("grad: bwd:") for r in reasons), reasons
            return None, None
        # #16337 in: ASM now serves gfx950 D=64.
        assert aot > 0 and native == 0
        row = ov.census()[_census_key(_GRAD_PREFIX, sq, skv, d, "bf16")]
        bwd = ov.census()[_census_key(_BWD_PREFIX, sq, skv, d, "bf16")]
        return row, bwd
    # olmo2: unmasked MHA D=128 -- ASM serves it even with FlyDSL off.
    assert aot > 0 and native == 0
    row = ov.census()[_census_key(_GRAD_PREFIX, sq, skv, d, "bf16")]
    bwd = ov.census()[_census_key(_BWD_PREFIX, sq, skv, d, "bf16")]
    assert bwd["engines"] == {"ASM_SDPA_ENGINE"}
    return row, bwd


@pytest.mark.parametrize("arch", list(_ARCHS))
def test_use_cache_false_no_mask_matches_native_and_layout(torch, transformers, arch):
    """(q) ``use_cache=False``, no mask: q/k are the plain BSHD view of the
    projection output (q BSHD `(S*Hq*D, D, Hq*D, 1)`, k/v BSHD
    `(S*Hkv*D, D, Hkv*D, 1)`; GPT-2's q/k/v are fused `c_attn` slices,
    `(S*3*Hq*D, D, 3*Hq*D, 1)`)."""
    builder, head_dim, hq, hkv = _ARCHS[arch]
    config, layer = builder(torch, transformers)
    b, s = 2, 16
    hidden_data = torch.randn(
        b, s, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    hidden = hidden_data.clone().requires_grad_(True)
    hidden2 = hidden_data.clone().requires_grad_(True)

    with _routed_sdpa() as ov, _capture_sdpa_args() as captured:
        out, _ = _run_layer(torch, arch, layer, hidden, None, None)
        out.float().sum().backward()

    for p in layer.parameters():
        p.grad = None
    out2, _ = _run_layer(torch, arch, layer, hidden2, None, None)
    out2.float().sum().backward()

    torch.testing.assert_close(out.float(), out2.float(), **_TOL["bf16"])
    torch.testing.assert_close(
        hidden.grad.float(), hidden2.grad.float(), **_TOL["bf16"]
    )

    if arch == "gpt2":
        want_q = want_k = want_v = _gpt2_fused_stride(s, hq, head_dim)
    else:
        want_q = _bshd_stride(s, hq, head_dim)
        want_k = want_v = _bshd_stride(s, hkv, head_dim)
    assert captured["q_stride"] == want_q, (captured["q_stride"], want_q)
    assert captured["k_stride"] == want_k, (captured["k_stride"], want_k)
    assert captured["v_stride"] == want_v, (captured["v_stride"], want_v)
    if hkv != hq and arch != "gpt2":
        assert captured["enable_gqa"] is True, (
            "the unmasked, no-cache GQA variant is the only one that "
            "reaches the kernel with enable_gqa=True (sdpa_attention.py)"
        )

    _assert_served_or_declined(ov, s, s, head_dim, arch=arch, masked=False)


@pytest.mark.parametrize("arch", list(_ARCHS))
def test_use_cache_true_mixed_layout_matches_native(torch, transformers, arch):
    """(q) ``use_cache=True``: a prefill then a 1-token decode step appends
    to a ``DynamicCache``, producing a mixed per-tensor layout -- k/v become
    contiguous BHSD (`Hkv*S*D, S*D, D, 1`) while q stays a fresh BSHD view
    at the decode step (M18/M4: mixed layouts are structurally accepted,
    correctness is this test)."""
    from transformers.cache_utils import DynamicCache

    builder, head_dim, hq, hkv = _ARCHS[arch]
    config, layer = builder(torch, transformers)
    b, s_prefill = 2, 16
    hidden1_data = torch.randn(
        b, s_prefill, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    hidden2_data = torch.randn(
        b, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    def _two_step(h1, h2):
        cache = DynamicCache()
        out1, _ = _run_layer(torch, arch, layer, h1, None, cache)
        out2, _ = _run_layer(torch, arch, layer, h2, None, cache)
        return out1, out2

    hidden1 = hidden1_data.clone().requires_grad_(True)
    hidden2 = hidden2_data.clone().requires_grad_(True)
    with _routed_sdpa() as ov, _capture_sdpa_args() as captured:
        out1, out2 = _two_step(hidden1, hidden2)
        (out1.float().sum() + out2.float().sum()).backward()

    for p in layer.parameters():
        p.grad = None
    hidden1n = hidden1_data.clone().requires_grad_(True)
    hidden2n = hidden2_data.clone().requires_grad_(True)
    want1, want2 = _two_step(hidden1n, hidden2n)
    (want1.float().sum() + want2.float().sum()).backward()

    torch.testing.assert_close(out1.float(), want1.float(), **_TOL["bf16"])
    torch.testing.assert_close(out2.float(), want2.float(), **_TOL["bf16"])
    torch.testing.assert_close(
        hidden1.grad.float(), hidden1n.grad.float(), **_TOL["bf16"]
    )
    torch.testing.assert_close(
        hidden2.grad.float(), hidden2n.grad.float(), **_TOL["bf16"]
    )

    # the decode step's call is the mixed one: q is a fresh BSHD view (S=1)
    # for the rotary archs, but GPT-2's q is still a fused c_attn slice even
    # at decode (its projection never changes shape); k/v become the
    # cache's contiguous BHSD cat for every arch.
    if arch == "gpt2":
        assert captured["q_stride"] == _gpt2_fused_stride(1, hq, head_dim)
    else:
        assert captured["q_stride"] == _bshd_stride(1, hq, head_dim)
    assert captured["k_stride"] == _bhsd_contig_stride(hkv, s_prefill + 1, head_dim)
    assert captured["v_stride"] == _bhsd_contig_stride(hkv, s_prefill + 1, head_dim)

    _assert_served_or_declined(ov, 1, s_prefill + 1, head_dim, arch=arch, masked=False)


@pytest.mark.parametrize("arch", list(_ARCHS))
def test_padded_mask_matches_native(torch, transformers, arch):
    """(q) A padded ``attention_mask``: `repeat_kv`'s expanded-MHA path
    (``enable_gqa=False``, ``is_causal=False``, bias from the bool mask);
    always declines ``grad: fwd+stats:`` with FlyDSL off (both ASM builders
    reject any bias)."""
    builder, head_dim, hq, hkv = _ARCHS[arch]
    config, layer = builder(torch, transformers)
    b, s = 2, 16
    hidden_data = torch.randn(
        b, s, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    hidden = hidden_data.clone().requires_grad_(True)
    hidden2 = hidden_data.clone().requires_grad_(True)
    # a bool padding mask: the last 4 tokens of batch element 1 are padding.
    mask = torch.ones(b, 1, s, s, dtype=torch.bool, device="cuda")
    mask[1, :, :, -4:] = False

    with _routed_sdpa() as ov, _capture_sdpa_args() as captured:
        out, _ = _run_layer(torch, arch, layer, hidden, mask, None)
        out.float().sum().backward()

    for p in layer.parameters():
        p.grad = None
    out2, _ = _run_layer(torch, arch, layer, hidden2, mask, None)
    out2.float().sum().backward()

    torch.testing.assert_close(out.float(), out2.float(), **_TOL["bf16"])
    torch.testing.assert_close(
        hidden.grad.float(), hidden2.grad.float(), **_TOL["bf16"]
    )

    want_q = (
        _gpt2_fused_stride(s, hq, head_dim)
        if arch == "gpt2"
        else _bshd_stride(s, hq, head_dim)
    )
    assert captured["q_stride"] == want_q, (captured["q_stride"], want_q)
    if hkv != hq and arch != "gpt2":
        assert (
            captured["enable_gqa"] is False
        ), "a mask present forces repeat_kv (expanded MHA), never enable_gqa"

    _assert_served_or_declined(ov, s, s, head_dim, arch=arch, masked=True)
