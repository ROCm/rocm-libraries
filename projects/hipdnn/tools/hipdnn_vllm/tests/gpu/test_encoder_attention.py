# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""torch.ops.hipdnn_vllm.encoder_attention against torch SDPA, and the
MMEncoderAttention OOT layer against vLLM's own layer and torch SDPA.

A signature that no installed engine serves must decline with a taxonomy
reason; such cases xfail with the reason so the route per arch is visible in
the test report.
"""

import pytest
import torch
import torch.nn.functional as F

from hipdnn_vllm import _bootstrap, stats
from hipdnn_vllm.declines import DeclineError, Stage, parse_op_error, parse_reason

TOL = {
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=4e-3, rtol=4e-3),
    torch.float32: dict(atol=1e-4, rtol=1e-4),
}


def ref_attention(q, k, v, cu_seqlens, scale):
    """vLLM torch_sdpa_wrapper semantics in float32: images split along dim 1."""
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))  # BHSD
    gqa = q.shape[2] != k.shape[2]
    if cu_seqlens is None:
        o = F.scaled_dot_product_attention(qf, kf, vf, scale=scale, enable_gqa=gqa)
        return o.transpose(1, 2).to(q.dtype)
    out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[3], device=q.device)
    cu = cu_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        if b > a:
            o = F.scaled_dot_product_attention(
                qf[:, :, a:b], kf[:, :, a:b], vf[:, :, a:b], scale=scale, enable_gqa=gqa
            )
            out[:, a:b] = o.transpose(1, 2)
    return out.to(q.dtype)


def make_qkv(tokens, hq, hk, d, dtype, batch=1, fused=True, seed=0):
    """q, k, v as vLLM's ViT passes them: views of one fused projection, so
    the token stride is (hq + 2 hk) * d (when fused)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if fused:
        qkv = torch.randn(batch, tokens, (hq + 2 * hk) * d, device="cuda", dtype=dtype, generator=g)
        q = qkv[..., : hq * d].view(batch, tokens, hq, d)
        k = qkv[..., hq * d : (hq + hk) * d].view(batch, tokens, hk, d)
        v = qkv[..., (hq + hk) * d :].view(batch, tokens, hk, d)
        return q, k, v
    return tuple(
        torch.randn(batch, tokens, h, d, device="cuda", dtype=dtype, generator=g) for h in (hq, hk, hk)
    )


def cu(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


def run_or_xfail(q, k, v, cu_seqlens, scale):
    try:
        return torch.ops.hipdnn_vllm.encoder_attention(q, k, v, cu_seqlens, scale)
    except RuntimeError as exc:
        parsed = parse_op_error(exc)
        if parsed is None:
            raise
        parse_reason(parsed[1].reason)
        pytest.xfail(
            f"hipDNN declined {tuple(q.shape)} kv_heads={k.shape[2]} {q.dtype}: {parsed[1].reason}"
        )


def execute_calls():
    from hipdnn_vllm import _C

    return _C.stats_snapshot()["runtime"]["execute_calls"]


@pytest.fixture(autouse=True)
def _loaded(hipdnn_state):
    return hipdnn_state


DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("d", [64, 80, 128])
def test_uniform_images_one_launch(dtype, d):
    lengths = [256, 256, 256]
    q, k, v = make_qkv(sum(lengths), 16, 16, d, dtype, seed=d)
    c = cu(lengths)
    scale = d**-0.5
    out = run_or_xfail(q, k, v, c, scale)
    torch.testing.assert_close(out, ref_attention(q, k, v, c, scale), **TOL[dtype])
    assert out.shape == (1, sum(lengths), 16, d) and out.is_contiguous()
    before = execute_calls()
    torch.ops.hipdnn_vllm.encoder_attention(q, k, v, c, scale)
    assert execute_calls() - before == 1  # one batched graph for all images


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("d", [64, 80, 128])
def test_varied_images_per_length(dtype, d):
    # Qwen2.5-VL window attention shape: mostly 64-token windows, ragged edges.
    lengths = [64, 64, 48, 64, 16, 48]
    q, k, v = make_qkv(sum(lengths), 16, 16, d, dtype, seed=d + 1)
    c = cu(lengths)
    scale = d**-0.5
    out = run_or_xfail(q, k, v, c, scale)
    torch.testing.assert_close(out, ref_attention(q, k, v, c, scale), **TOL[dtype])
    before = execute_calls()
    torch.ops.hipdnn_vllm.encoder_attention(q, k, v, c, scale)
    assert execute_calls() - before == len(lengths)


# 64 is the SigLIP / InternViT head size; 128 is the size ASM_SDPA serves on CDNA.
HEAD_SIZES = [64, 128]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("d", HEAD_SIZES)
def test_padded_cu_seqlens_and_contiguous_inputs(dtype, d):
    """cu_seqlens padded with repeats of the last offset (vLLM's encoder
    cudagraph bucketing) adds empty images, which must be skipped."""
    lengths = [100, 100]
    q, k, v = make_qkv(sum(lengths), 8, 8, d, dtype, fused=False, seed=3)
    c = torch.cat([cu(lengths), torch.tensor([200, 200], dtype=torch.int32, device="cuda")])
    out = run_or_xfail(q, k, v, c, d**-0.5)
    torch.testing.assert_close(out, ref_attention(q, k, v, cu(lengths), d**-0.5), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("d", HEAD_SIZES)
def test_gqa(dtype, d):
    lengths = [128, 128]
    q, k, v = make_qkv(sum(lengths), 16, 4, d, dtype, seed=4)
    c = cu(lengths)
    out = run_or_xfail(q, k, v, c, d**-0.5)
    torch.testing.assert_close(out, ref_attention(q, k, v, c, d**-0.5), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("d", HEAD_SIZES)
def test_dense_batch_without_cu_seqlens(dtype, d):
    """SigLIP-style call: [B, S, H, D] with no cu_seqlens."""
    q, k, v = make_qkv(196, 12, 12, d, dtype, batch=3, seed=5)
    out = run_or_xfail(q, k, v, None, d**-0.5)
    torch.testing.assert_close(out, ref_attention(q, k, v, None, d**-0.5), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("d", HEAD_SIZES)
def test_batch_gt1_with_cu_seqlens_varied(dtype, d):
    lengths = [50, 30]
    q, k, v = make_qkv(sum(lengths), 8, 8, d, dtype, batch=2, seed=6)
    c = cu(lengths)
    out = run_or_xfail(q, k, v, c, d**-0.5)
    torch.testing.assert_close(out, ref_attention(q, k, v, c, d**-0.5), **TOL[dtype])


def test_prefilter_declines_are_counted():
    q, k, v = make_qkv(64, 4, 4, 64, torch.bfloat16, fused=False)
    strided = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
    stats.reset()
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.encoder_attention(strided, k, v, cu([64]), 0.125)
    op, decline = parse_op_error(ei.value)
    assert op == "encoder_attention" and decline.reason == "UNSUPPORTED_FEATURE:strided_head_dim"
    with pytest.raises(RuntimeError) as ei:
        torch.ops.hipdnn_vllm.encoder_attention(q, k.float(), v, cu([64]), 0.125)
    assert parse_op_error(ei.value)[1].reason == "UNSUPPORTED_FEATURE:mixed_dtype"
    rows = [r for r in stats.snapshot()["counters"] if r["op"] == "encoder_attention"]
    reasons = {r["reason"]: r["count"] for r in rows if r["route"] == "fallback"}
    assert reasons.get("UNSUPPORTED_FEATURE:strided_head_dim") == 1
    assert reasons.get("UNSUPPORTED_FEATURE:mixed_dtype") == 1


def test_capture_with_cu_seqlens_declines():
    q, k, v = make_qkv(128, 8, 8, 64, torch.bfloat16)
    c = cu([64, 64])
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    err = None
    with torch.cuda.stream(stream):
        try:
            with torch.cuda.graph(graph, stream=stream):
                try:
                    torch.ops.hipdnn_vllm.encoder_attention(q, k, v, c, 0.125)
                except RuntimeError as exc:
                    err = exc
        except RuntimeError:
            pass
    assert err is not None
    assert parse_op_error(err)[1].stage is Stage.CAPTURE_INVARIANT


@pytest.mark.parametrize("d", HEAD_SIZES)
def test_dense_capture_replay(d):
    q, k, v = make_qkv(196, 12, 12, d, torch.bfloat16, batch=2, fused=False)
    run_or_xfail(q, k, v, None, d**-0.5)  # builds the plan outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.hipdnn_vllm.encoder_attention(q, k, v, None, d**-0.5)
    for seed in (11, 12):
        nq, nk, nv = make_qkv(196, 12, 12, d, torch.bfloat16, batch=2, fused=False, seed=seed)
        q.copy_(nq), k.copy_(nk), v.copy_(nv)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref_attention(q, k, v, None, d**-0.5), **TOL[torch.bfloat16])


@pytest.mark.parametrize("d", HEAD_SIZES)
def test_torch_compile_fullgraph(d):
    lengths = [64, 64, 32]
    q, k, v = make_qkv(sum(lengths), 8, 8, d, torch.bfloat16)
    c = cu(lengths)
    eager = run_or_xfail(q, k, v, c, d**-0.5)

    @torch.compile(fullgraph=True, dynamic=True)
    def f(q, k, v, c):
        return torch.ops.hipdnn_vllm.encoder_attention(q, k, v, c, d**-0.5) * 2

    torch.testing.assert_close(f(q, k, v, c), eager * 2)


# --- the vLLM layer ------------------------------------------------------------

vllm = pytest.importorskip("vllm")


@pytest.fixture
def vllm_ctx():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def oot(vllm_ctx):
    from hipdnn_vllm.ops.encoder_attention import register_oot

    register_oot()
    from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

    return MMEncoderAttention


def make_layer(oot, num_heads=16, head_size=80, monkeypatch=None, strict=False, dtype=torch.bfloat16):
    from hipdnn_vllm.ops.encoder_attention import HipdnnMMEncoderAttention

    if monkeypatch is not None:
        monkeypatch.setenv("HIPDNN_VLLM_STRICT", "1" if strict else "0")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        layer = oot(num_heads=num_heads, head_size=head_size, scale=head_size**-0.5)
    finally:
        torch.set_default_dtype(prev)
    assert type(layer) is HipdnnMMEncoderAttention
    return layer


def parent_forward(layer, *args):
    from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

    return MMEncoderAttention.forward_hip(layer, *args)


def layer_route_counts():
    rows = [r for r in stats.snapshot()["counters"] if r["op"] == "encoder_attention" and r["count"]]
    out = {}
    for r in rows:
        out[r["route"]] = out.get(r["route"], 0) + r["count"]
    return out


def test_oot_registration_idempotent(oot):
    from vllm.model_executor.custom_op import op_registry_oot

    from hipdnn_vllm.ops.encoder_attention import HipdnnMMEncoderAttention, register_oot

    register_oot()
    assert op_registry_oot["MMEncoderAttention"] is HipdnnMMEncoderAttention


def test_cu_seqlens_hooks_are_parents(oot):
    """The OOT class inherits the hooks: MMEncoderAttention's classmethods
    dispatch to it without recursing and return the parent's layout."""
    import numpy as np

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cu_np = np.array([0, 64, 128, 176], dtype=np.int32)
    for backend in (
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
        AttentionBackendEnum.TORCH_SDPA,
    ):
        assert oot.maybe_compute_seq_lens(backend, cu_np, torch.device("cuda")) is None
        got = oot.maybe_recompute_cu_seqlens(backend, cu_np, 1280, 1, torch.device("cuda"))
        assert got.device.type == "cuda"
        assert got.cpu().tolist() == cu_np.tolist()


def op_accepts(q, k, v, c, scale):
    try:
        torch.ops.hipdnn_vllm.encoder_attention(q, k, v, c, scale)
        return True
    except RuntimeError as exc:
        if parse_op_error(exc) is None:
            raise
        return False


@pytest.mark.parametrize("head_size", [80, 128], ids=["d80_qwen_vit", "d128"])
@pytest.mark.parametrize("lengths", [[256, 256], [64, 64, 48, 16]], ids=["uniform", "varied"])
def test_layer_matches_parent(oot, monkeypatch, lengths, head_size):
    """The layer equals vLLM's own layer, and takes the hipDNN route exactly
    when the op accepts the signature (one count per graph launch)."""
    layer = make_layer(oot, head_size=head_size, monkeypatch=monkeypatch)
    q, k, v = make_qkv(sum(lengths), 16, 16, head_size, torch.bfloat16, seed=7)
    c = cu(lengths)
    max_seqlen = torch.tensor(max(lengths), dtype=torch.int32)
    accepted = op_accepts(q, k, v, c, layer.scale)
    stats.reset()
    got = layer(q, k, v, cu_seqlens=c, max_seqlen=max_seqlen)
    want = parent_forward(layer, q, k, v, c, max_seqlen, None)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, **TOL[torch.bfloat16])
    routes = layer_route_counts()
    if accepted:
        launches = 1 if len(set(lengths)) == 1 else len(lengths)
        assert routes == {"hipdnn": launches}, routes
    else:
        assert routes == {"fallback": 1}, routes


def expected_routes(accepted, launches):
    """Route counts for one layer call: hipDNN whenever the op accepts the
    signature (so a silent fallback fails), else one counted fallback."""
    return {"hipdnn": launches} if accepted else {"fallback": 1}


# float16 has a tight tolerance against the float32 reference, so one key
# leaking across an image boundary (or dropped) fails it.
LAYER_DTYPES = [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dtype", LAYER_DTYPES, ids=str)
def test_layer_3d_input(oot, monkeypatch, dtype):
    layer = make_layer(oot, num_heads=12, head_size=64, monkeypatch=monkeypatch, dtype=dtype)
    q, k, v = make_qkv(196, 12, 12, 64, dtype, batch=2, fused=False, seed=8)
    q3, k3, v3 = (t.reshape(2, 196, -1) for t in (q, k, v))
    accepted = op_accepts(q, k, v, None, layer.scale)
    stats.reset()
    got = layer(q3, k3, v3)
    assert got.shape == (2, 196, 12 * 64)
    want = ref_attention(q, k, v, None, layer.scale).reshape(2, 196, -1)
    torch.testing.assert_close(got, want, **TOL[dtype])
    assert layer_route_counts() == expected_routes(accepted, 1), layer_route_counts()


def test_layer_fallback_on_decline_is_counted(oot, monkeypatch):
    """A strided head dim is declined by the pre-filter; the layer must return
    the parent's result and count the fallback."""
    layer = make_layer(oot, num_heads=8, head_size=64, monkeypatch=monkeypatch)
    base = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
    q = base[..., ::2]
    k = base[..., 1::2]
    v = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    c = cu([64, 64])
    ms = torch.tensor(64, dtype=torch.int32)
    stats.reset()
    got = layer(q, k, v, cu_seqlens=c, max_seqlen=ms)
    want = parent_forward(layer, q, k, v, c, ms, None)
    torch.testing.assert_close(got, want, **TOL[torch.bfloat16])
    assert layer_route_counts().get("fallback", 0) == 1


def test_layer_strict_raises(oot, monkeypatch):
    layer = make_layer(oot, num_heads=8, head_size=64, monkeypatch=monkeypatch, strict=True)
    base = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(DeclineError):
        layer(base[..., ::2], base[..., 1::2], v, cu_seqlens=cu([128]), max_seqlen=torch.tensor(128))


@pytest.mark.parametrize("dtype", LAYER_DTYPES, ids=str)
@pytest.mark.parametrize("head_size", [80, 128])
def test_layer_under_torch_compile(oot, monkeypatch, head_size, dtype):
    layer = make_layer(oot, num_heads=16, head_size=head_size, monkeypatch=monkeypatch, dtype=dtype)
    lengths = [64, 64, 48, 16]
    q, k, v = make_qkv(sum(lengths), 16, 16, head_size, dtype, seed=9)
    c = cu(lengths)
    ms = torch.tensor(64, dtype=torch.int32)
    want = ref_attention(q, k, v, c, layer.scale)
    accepted = op_accepts(q, k, v, c, layer.scale)

    @torch.compile(fullgraph=True, dynamic=True)
    def f(q, k, v, c, ms):
        return layer(q, k, v, cu_seqlens=c, max_seqlen=ms) + 1

    stats.reset()
    got = f(q, k, v, c, ms)
    torch.testing.assert_close(got, (want.float() + 1).to(dtype), **TOL[dtype])
    # The opaque op runs eagerly inside the compiled graph and counts its route.
    assert layer_route_counts() == expected_routes(accepted, len(lengths)), layer_route_counts()


def test_bootstrap_loaded():
    assert _bootstrap.load() is not None
