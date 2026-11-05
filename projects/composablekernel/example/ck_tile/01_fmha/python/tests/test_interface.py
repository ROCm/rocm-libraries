# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch

from ck_tile_fmha import fmha_func, fmha_varlen_func, fmha_with_kvcache
from reference import apply_rotary_emb


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
def test_softcap_is_rejected(api):
    q = torch.randn(1, 4, 2, 32, dtype=torch.float16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    with pytest.raises(RuntimeError, match="do not support softcap"):
        if api == "fixed":
            fmha_func(q, k, v, softcap=1.0)
        elif api == "varlen":
            cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
            fmha_varlen_func(q[0], k[0], v[0], cu_seqlens, cu_seqlens, 4, 4, softcap=1.0)
        else:
            fmha_with_kvcache(q, k, v, softcap=1.0)


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_registered_operator_fake_metadata(varlen, backward):
    # Compare fake metadata with actual CK outputs, including the uint8 dropout
    # buffer and CK-specific variable-length layouts used by torch.compile.
    q = torch.randn(2, 97, 4, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 113, 2, 64, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    sequence_args = ()
    if varlen:
        q, k, v = (x.flatten(0, 1) for x in (q, k, v))
        cu_q = torch.tensor([0, 97, 194], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 113, 226], device="cuda", dtype=torch.int32)
        sequence_args = (cu_q, cu_k, 97, 113)
    name = "_fmha_varlen_" if varlen else "_fmha_"
    op = getattr(torch.ops.ck_tile_fmha, name + "forward")
    args = (q, k, v, *sequence_args, 0.17, 0.125, False, -1, -1, 0.0, None, True)
    if backward:
        out, lse, _, rng = op(*args)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        op = getattr(torch.ops.ck_tile_fmha, name + "backward")
        args = (
            torch.randn_like(out),
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            *sequence_args,
            0.17,
            0.125,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng,
        )
    torch.library.opcheck(op, args, test_utils=("test_faketensor",))


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_reference_batch_offsets(interleaved):
    # Batch zero rotates by pi/2; batch one uses the identity. The last two
    # features are outside rotary_dim and must survive unchanged.
    x = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda", dtype=torch.float16)
    x = x.expand(2, 1, 1, 6).contiguous()
    cos = torch.tensor([[0, 0], [1, 1]], device="cuda", dtype=x.dtype)
    sin = torch.tensor([[1, 1], [0, 0]], device="cuda", dtype=x.dtype)
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    actual = apply_rotary_emb(x, cos, sin, interleaved=interleaved, seqlen_offsets=offsets)
    first = [-2, 1, -4, 3, 5, 6] if interleaved else [-3, -4, 1, 2, 5, 6]
    expected = torch.tensor([first, [1, 2, 3, 4, 5, 6]], device="cuda", dtype=x.dtype)
    assert torch.equal(actual[:, 0, 0], expected)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride", "alibi"])
def test_bias_validation(invalid):
    q = torch.randn(1, 4, 2, 64, device="cuda", dtype=torch.float16)
    bias = torch.zeros(4, 4, device=q.device, dtype=q.dtype)
    kwargs = {}
    if invalid == "shape":
        bias = bias.unsqueeze(0)
        message = "bias shape"
    elif invalid == "dtype":
        bias = bias.float()
        message = "same dtype"
    elif invalid == "device":
        bias = bias.cpu()
        message = "same device"
    elif invalid == "stride":
        bias = bias.t()
        message = "contiguous last dimension"
    else:
        kwargs["alibi_slopes"] = torch.ones(2, device=q.device)
        message = "cannot apply bias and alibi"
    with pytest.raises(RuntimeError, match=message):
        fmha_func(q, q, q, bias=bias, **kwargs)


def test_bias_operator_metadata():
    q = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = torch.randn(32, 32, device=q.device, dtype=q.dtype)
    fwd = torch.ops.ck_tile_fmha._fmha_forward
    args = (q, k, v, 0.17, 0.125, False, -1, -1, 0.0, None, True, bias)
    torch.library.opcheck(fwd, args, test_utils=("test_schema", "test_faketensor"))
    out, lse, _, rng = fwd(*args)
    dq, dk, dv, dbias = (torch.empty_like(x) for x in (q, k, v, bias))
    args = (
        torch.randn_like(out),
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        0.17,
        0.125,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        rng,
        bias,
        dbias,
    )
    torch.library.opcheck(
        torch.ops.ck_tile_fmha._fmha_backward,
        args,
        test_utils=("test_schema", "test_faketensor"),
    )


def test_bias_torch_compile():
    q, k, v = (
        torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    bias = torch.randn(32, 32, device=q.device, dtype=q.dtype, requires_grad=True)
    compiled = torch.compile(fmha_func, backend="aot_eager", fullgraph=True)
    actual = compiled(q, k, v, bias=bias)
    expected = fmha_func(q, k, v, bias=bias)
    torch.testing.assert_close(actual, expected)
    dout = torch.randn_like(actual)
    args = (q, k, v, bias)
    for grad, ref in zip(
        torch.autograd.grad(actual, args, dout), torch.autograd.grad(expected, args, dout)
    ):
        torch.testing.assert_close(grad, ref)


def test_native_bias_defaults_preserve_positional_api():
    from ck_tile_fmha import _C

    q, k, v = (torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    args = (q, k, v, None, None, 0.0, 0.125, False, -1, -1, 0.0, False, None)
    out, lse, _, rng = _C.fwd(*args)
    explicit = _C.fwd(*args, None)
    torch.testing.assert_close(out, explicit[0])
    torch.testing.assert_close(lse, explicit[1])
    args = (
        torch.randn_like(out),
        q,
        k,
        v,
        out,
        lse,
        None,
        None,
        None,
        None,
        0.0,
        0.125,
        False,
        -1,
        -1,
        0.0,
        False,
        None,
        rng,
    )
    for actual, expected in zip(_C.bwd(*args), _C.bwd(*args, None, None)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
@pytest.mark.parametrize("invalid", ["dtype", "device", "shape", "stride"])
def test_alibi_validation(api, invalid):
    q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
    slopes = torch.ones(2, device=q.device)
    if invalid == "dtype":
        slopes = slopes.half()
        message = "float32"
    elif invalid == "device":
        slopes = slopes.cpu()
        message = "same device"
    elif invalid == "shape":
        slopes = slopes[0]
        message = "shape"
    else:
        slopes = torch.ones(4, device=q.device)[::2]
        message = "contiguous last dimension"
    with pytest.raises(RuntimeError, match=message):
        if api == "fixed":
            fmha_func(q, q, q, alibi_slopes=slopes)
        elif api == "varlen":
            cu = torch.tensor([0, 8], device=q.device, dtype=torch.int32)
            fmha_varlen_func(q[0], q[0], q[0], cu, cu, 8, 8, alibi_slopes=slopes)
        else:
            fmha_with_kvcache(q, q, q, cache_seqlens=8, alibi_slopes=slopes)


def test_kvcache_leftpad_is_rejected():
    q = torch.randn(1, 4, 2, 64, device="cuda", dtype=torch.float16)
    leftpad = torch.ones(1, device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="leftpad"):
        fmha_with_kvcache(q, q, q, cache_seqlens=4, cache_leftpad=leftpad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("argument", ["seqused_k", "leftpad_k", "num_splits"])
def test_native_varlen_unsupported_arguments(dtype, argument):
    q = torch.randn(8, 2, 64, device="cuda", dtype=dtype)
    cu = torch.tensor([0, 8], device=q.device, dtype=torch.int32)
    value = 2 if argument == "num_splits" else torch.ones(1, device=q.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match=argument):
        torch.ops.ck_tile_fmha._fmha_varlen_forward(
            q, q, q, cu, cu, 8, 8, 0.0, 0.125, False, **{argument: value}
        )


@pytest.mark.parametrize("varlen", [False, True])
def test_native_backward_without_rng(varlen):
    q, k, v = (torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    sequence_args = ()
    if varlen:
        q, k, v = (x[0] for x in (q, k, v))
        cu = torch.tensor([0, 32], device=q.device, dtype=torch.int32)
        sequence_args = (cu, cu, 32, 32)
    prefix = "_fmha_varlen_" if varlen else "_fmha_"
    forward = getattr(torch.ops.ck_tile_fmha, prefix + "forward")
    out, lse, _, rng = forward(q, k, v, *sequence_args, 0.0, 0.125, False, -1, -1, 0.0, None, False)
    dout = torch.randn_like(out)
    actual = _native_backward(varlen, q, k, v, out, lse, dout)
    expected = _native_backward(varlen, q, k, v, out, lse, dout, rng)
    for grad, ref in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(grad, ref)


def _native_backward(varlen, q, k, v, out, lse, dout, rng=None, slopes=None, deterministic=False):
    from ck_tile_fmha import _C

    args = (dout, q, k, v, out, lse, None, None, None)
    if varlen:
        cu_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
        cu_k = torch.tensor([0, k.shape[0]], device=q.device, dtype=torch.int32)
        return _C.varlen_bwd(
            *args,
            cu_q,
            cu_k,
            slopes,
            q.shape[0],
            k.shape[0],
            0.0,
            0.125,
            False,
            False,
            -1,
            -1,
            0.0,
            deterministic,
            None,
            rng,
        )
    return _C.bwd(*args, slopes, 0.0, 0.125, False, -1, -1, 0.0, deterministic, None, rng)


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
@pytest.mark.parametrize("zero_q", [False, True])
def test_zero_heads_are_rejected(api, zero_q):
    q = torch.empty(1, 8, 0 if zero_q else 2, 64, device="cuda", dtype=torch.float16)
    k = torch.empty(1, 8, 2 if zero_q else 0, 64, device=q.device, dtype=q.dtype)
    v = torch.empty_like(k)
    lse = torch.empty(1, q.shape[2], 8, device=q.device)
    with pytest.raises(RuntimeError, match="Number of heads"):
        if api == "kvcache":
            fmha_with_kvcache(q, k, v, cache_seqlens=8)
        else:
            if api == "varlen":
                q, k, v, lse = q[0], k[0], v[0], lse[0]
            _native_backward(api == "varlen", q, k, v, q, lse, q)


@pytest.mark.parametrize("varlen", [False, True])
def test_native_backward_alibi_dtype(varlen):
    q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
    lse = torch.empty(1, 2, 8, device=q.device)
    slopes = torch.ones(2, device=q.device, dtype=q.dtype)
    if varlen:
        q, lse = q[0], lse[0]
    with pytest.raises(RuntimeError, match="float32"):
        _native_backward(varlen, q, q, q, q, lse, q, slopes=slopes)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_native_backward_off_current_device(varlen, deterministic):
    from reference import attention_ref

    with torch.cuda.device(1):
        q, k, v = (
            torch.randn(1, 128, 2, 64, device="cuda:1", dtype=torch.float16, requires_grad=True)
            for _ in range(3)
        )
        out_ref, _, lse = attention_ref(q, k, v, return_lse=True)
        dout = torch.randn_like(out_ref)
        expected = torch.autograd.grad(out_ref, (q, k, v), dout)
        if varlen:
            q, k, v, out_ref, lse, dout = (x[0] for x in (q, k, v, out_ref, lse, dout))
            expected = tuple(x[0] for x in expected)
        torch.cuda.synchronize()
    with torch.cuda.device(0):
        actual = _native_backward(varlen, q, k, v, out_ref, lse, dout, deterministic=deterministic)
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(1)
    for grad, ref in zip(actual[:3], expected):
        torch.testing.assert_close(grad, ref, atol=4e-3, rtol=4e-3)


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_dropout_current_stream(varlen, capture):
    from reference import attention_ref
    from utils import ck_randval_to_dropout_mask

    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    )
    dout = torch.randn_like(q)
    cu = torch.tensor([0, 128], device=q.device, dtype=torch.int32)
    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in (q, k, v))

    def run():
        if varlen:
            out, _, randval = fmha_varlen_func(
                q[0], k[0], v[0], cu, cu, 128, 128, dropout_p=0.3, return_attn_probs=True
            )
            out, randval = out.unsqueeze(0), randval.unsqueeze(0)
        else:
            out, _, randval = fmha_func(q, k, v, dropout_p=0.3, return_attn_probs=True)
        grads = torch.autograd.grad(out, (q, k, v), dout)
        return out, randval, grads

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out, randval, grads = run()
    previous = None
    for _ in range(2):
        with torch.cuda.stream(stream):
            if capture:
                graph.replay()
            else:
                out, randval, grads = run()
        torch.cuda.current_stream().wait_stream(stream)
        if previous is not None:
            assert not torch.equal(randval, previous)
        previous = randval.clone()
        mask = ck_randval_to_dropout_mask(randval, 0.3) >= 0
        ref, _ = attention_ref(*reference_inputs, dropout_p=0.3, dropout_mask=mask)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
        for grad, ref_grad in zip(grads, torch.autograd.grad(ref, reference_inputs, dout)):
            torch.testing.assert_close(grad, ref_grad, atol=4e-3, rtol=4e-3)
        # Finish reference reads before a replay overwrites the captured buffers.
        stream.wait_stream(torch.cuda.current_stream())


def _call_sink_api(api, q, sink, **kwargs):
    import ck_tile_fmha

    if api == "fixed":
        return fmha_func(q, q, q, sink=sink, **kwargs)
    if api == "kvpacked":
        return ck_tile_fmha.fmha_kvpacked_func(q, torch.stack((q, q), dim=2), sink=sink, **kwargs)
    if api == "qkvpacked":
        return ck_tile_fmha.fmha_qkvpacked_func(torch.stack((q, q, q), dim=2), sink=sink, **kwargs)
    if api == "kvcache":
        return fmha_with_kvcache(q, q.clone(), q.clone(), sink=sink, **kwargs)
    cu = torch.arange(q.shape[0] + 1, device=q.device, dtype=torch.int32) * q.shape[1]
    flat = q.flatten(0, 1)
    if api == "varlen":
        return fmha_varlen_func(
            flat, flat, flat, cu, cu, q.shape[1], q.shape[1], sink=sink, **kwargs
        )
    if api == "varlen_kvpacked":
        return ck_tile_fmha.fmha_varlen_kvpacked_func(
            flat,
            torch.stack((flat, flat), dim=1),
            cu,
            cu,
            q.shape[1],
            q.shape[1],
            sink=sink,
            **kwargs,
        )
    return ck_tile_fmha.fmha_varlen_qkvpacked_func(
        torch.stack((flat, flat, flat), dim=1),
        cu,
        q.shape[1],
        sink=sink,
        **kwargs,
    )


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
@pytest.mark.parametrize("logit", [False, True])
@pytest.mark.parametrize(
    "causal,window",
    [
        pytest.param(False, (-1, -1), id="unmasked"),
        pytest.param(True, (-1, -1), id="causal"),
        pytest.param(False, (-1, 3), id="right-bounded"),
        pytest.param(False, (16, 0), id="clamped-left"),
    ],
)
def test_sink_prefix_without_left_window_matches_default(api, logit, causal, window):
    torch.manual_seed(45)
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
    sink = torch.tensor([-0.7, 4.0], device=q.device) if logit else None
    kwargs = dict(causal=causal, window_size=window)
    baseline = _call_sink_api(api, q, sink, **kwargs)
    # Even a prefix beyond split-KV's local-window limit is inert here.
    actual = _call_sink_api(api, q, sink, sink_size=80, **kwargs)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)


@pytest.mark.parametrize("api", ["fixed", "varlen", "kvcache"])
@pytest.mark.parametrize(
    "invalid", ["shape", "heads", "dtype", "device", "stride", "sink_size", "scale"]
)
def test_sink_validation(api, invalid):
    q = torch.randn(1, 4, 2, 64, device="cuda", dtype=torch.float16)
    sink = torch.zeros(2, device=q.device)
    kwargs = {}
    if invalid == "shape":
        sink = sink[None]
        message = "shape"
    elif invalid == "heads":
        sink = sink[:1]
        message = "shape"
    elif invalid == "dtype":
        sink = sink.half()
        message = "float32"
    elif invalid == "device":
        sink = sink.cpu()
        message = "same device"
    elif invalid == "stride":
        sink = torch.zeros(4, device=q.device)[::2]
        message = "contiguous"
    elif invalid == "sink_size":
        kwargs["sink_size"] = -1
        message = "nonnegative"
    else:
        kwargs["softmax_scale"] = 0.0
        message = "positive finite"
    with pytest.raises(RuntimeError, match=message):
        _call_sink_api(api, q, sink, **kwargs)


@pytest.mark.parametrize(
    "api",
    ["fixed", "kvpacked", "qkvpacked", "varlen", "varlen_kvpacked", "varlen_qkvpacked", "kvcache"],
)
@pytest.mark.parametrize("grad_input", ["q", "sink", "prefix"])
def test_sink_forward_only(api, grad_input):
    q = torch.randn(
        1, 4, 2, 64, device="cuda", dtype=torch.float16, requires_grad=grad_input != "sink"
    )
    sink = (
        None
        if grad_input == "prefix"
        else torch.zeros(2, device=q.device, requires_grad=grad_input == "sink")
    )
    kwargs = dict(sink_size=2, causal=True, window_size=(1, 0))
    with pytest.raises(ValueError, match="forward only"):
        _call_sink_api(api, q, sink, **kwargs)
    with torch.no_grad():
        result = _call_sink_api(api, q, sink, **kwargs)
        assert not result.requires_grad


@pytest.mark.parametrize("varlen", [False, True])
def test_sink_operator_metadata_and_compile(varlen):
    torch.manual_seed(48)
    q = torch.randn(2, 33, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 65, 2, 64, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    sink = torch.tensor([-torch.inf, 0.0, 4.0, 7.0], device=q.device)
    sequence_args = ()
    if varlen:
        q, k, v = (x.flatten(0, 1) for x in (q, k, v))
        cu_q = torch.tensor([0, 33, 66], device=q.device, dtype=torch.int32)
        cu_k = torch.tensor([0, 65, 130], device=q.device, dtype=torch.int32)
        sequence_args = (cu_q, cu_k, 33, 65)
    op = (
        torch.ops.ck_tile_fmha._fmha_varlen_forward
        if varlen
        else torch.ops.ck_tile_fmha._fmha_forward
    )
    args = (q, k, v, *sequence_args, 0.0, 0.125, True, 17, 0, 0.0, None, False)
    torch.library.opcheck(
        op, args, kwargs=dict(sink_size=4, sink=sink), test_utils=("test_schema", "test_faketensor")
    )

    def forward(q, k, v, sink):
        kwargs = dict(
            causal=True, window_size=(17, 0), sink_size=4, sink=sink, return_attn_probs=True
        )
        if varlen:
            return fmha_varlen_func(q, k, v, *sequence_args, **kwargs)[:2]
        return fmha_func(q, k, v, **kwargs)[:2]

    compiled = torch.compile(forward, backend="inductor", fullgraph=True)
    # Reuse the graph with different logits so the sink tensor cannot be captured as a constant.
    for values in (sink, sink + 1.0):
        actual = compiled(q, k, v, values)
        expected = forward(q, k, v, values)
        for result, ref in zip(actual, expected):
            torch.testing.assert_close(result, ref, atol=0, rtol=0)
