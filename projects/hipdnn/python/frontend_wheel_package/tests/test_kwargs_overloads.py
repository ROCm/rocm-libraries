# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the cuDNN-compatible keyword overloads of the Graph op methods.

Each op is built twice, once through its ``*Attributes`` object and once
through the cuDNN-shaped keyword overload, with identically named and
UID-tagged tensors; the serialized graphs must match. Dispatch tests pin that
the attributes overloads keep winning for every call they accepted before.
"""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import scalar_tensor, unpack
from .helpers import (
    build_all_plans,
    canonical_graph_json,
    create_float_graph,
    execute_zeros,
)

_HAS_SDPA = hasattr(hipdnn.Graph, "sdpa")
_requires_sdpa = pytest.mark.skipif(
    not _HAS_SDPA, reason="SDPA is compiled out of the bindings"
)

FLOAT = hipdnn.DataType.FLOAT
INT32 = hipdnn.DataType.INT32
INT64 = hipdnn.DataType.INT64

# Parity graphs compute in FLOAT, and validate() gives an op whose compute type
# is NOT_SET the graph's. Ops are given a different compute type, so an overload
# that drops compute_data_type changes the serialized node.
OP_COMPUTE = hipdnn.DataType.HALF


class _Tensors:
    """Creates tensors with names and sequential UIDs, so two builds that
    create tensors in the same order assign the same UIDs."""

    def __init__(self):
        self._next_uid = 1

    def adopt(self, tensor, name, output=False):
        if tensor is None:
            return None
        tensor.set_name(name).set_uid(self._next_uid)
        self._next_uid += 1
        if output:
            tensor.set_output(True)
        return tensor

    def input(self, name, dim, data_type=FLOAT):
        return self.adopt(hipdnn.Tensor.create(dim, data_type), name)

    def scalar(self, name, value=1e-5):
        return self.adopt(scalar_tensor(value), name)


def _adopt_outputs(tensors, outputs, names, kind):
    for output, name in zip(unpack(outputs, len(names), kind), names):
        tensors.adopt(output, name, output=True)


# Compute type of the graph _graph(None) builds; the default-compute parity test
# also runs with FLOAT. See _graph.
_graph_compute_without_op_compute = OP_COMPUTE


def _graph(compute):
    """compute None builds both forms without a compute type, which validate() fills
    with the graph's. The graph computes in OP_COMPUTE, so a keyword default other
    than NOT_SET changes the serialized node, except a default of OP_COMPUTE itself,
    which the FLOAT-graph run of the same test catches."""
    graph = create_float_graph()
    if compute is None:
        graph.set_compute_data_type(_graph_compute_without_op_compute)
    return graph, _Tensors()


def _compute_keywords(compute):
    return {} if compute is None else {"compute_data_type": compute}


def _compute_args(compute):
    return () if compute is None else (compute,)


def _new_attributes(cls, compute):
    attributes = cls()
    if compute is not None:
        attributes.set_compute_data_type(compute)
    return attributes


def _assert_parity(build, compute):
    attributes_graph = build(False, compute)
    keywords_graph = build(True, compute)
    for graph in (attributes_graph, keywords_graph):
        status = graph.validate()
        assert status.is_good(), status.get_message()
    assert canonical_graph_json(keywords_graph) == canonical_graph_json(
        attributes_graph
    )


# --- Builders: each takes ``keywords`` and ``compute`` (see _graph) and returns
# an unvalidated graph. ---


def _peer_stats(t):
    return [t.input(f"peer_stats{index}", [2, 8, 1, 1]) for index in range(2)]


def _batchnorm(keywords, compute):
    graph, t = _graph(compute)
    x = t.input("x", [4, 8, 8, 8])
    scale = t.input("scale", [1, 8, 1, 1])
    bias = t.input("bias", [1, 8, 1, 1])
    running_mean = t.input("running_mean", [1, 8, 1, 1])
    running_var = t.input("running_var", [1, 8, 1, 1])
    epsilon = t.scalar("epsilon")
    momentum = t.scalar("momentum", 0.1)
    peer_stats = _peer_stats(t)
    if keywords:
        outputs = graph.batchnorm(
            x,
            scale,
            bias,
            running_mean,
            running_var,
            epsilon,
            momentum,
            peer_stats=peer_stats,
            **_compute_keywords(compute),
            name="bn",
        )
    else:
        outputs = graph.batchnorm(
            x,
            scale,
            bias,
            _new_attributes(hipdnn.BatchnormAttributes, compute)
            .set_name("bn")
            .set_epsilon(epsilon)
            .set_previous_running_stats(running_mean, running_var, momentum)
            .set_peer_stats(peer_stats),
        )
    _adopt_outputs(
        t, outputs, ["y", "mean", "inv_variance", "next_mean", "next_var"], list
    )
    return graph


def _batchnorm_backward(keywords, compute):
    graph, t = _graph(compute)
    dy = t.input("dy", [4, 8, 8, 8])
    x = t.input("x", [4, 8, 8, 8])
    scale = t.input("scale", [1, 8, 1, 1])
    mean = t.input("mean", [1, 8, 1, 1])
    inv_variance = t.input("inv_variance", [1, 8, 1, 1])
    peer_stats = _peer_stats(t)
    if keywords:
        outputs = graph.batchnorm_backward(
            grad=dy,
            input=x,
            scale=scale,
            mean=mean,
            inv_variance=inv_variance,
            peer_stats=peer_stats,
            **_compute_keywords(compute),
            name="bnb",
        )
    else:
        outputs = graph.batchnorm_backward(
            dy,
            x,
            scale,
            _new_attributes(hipdnn.BatchnormBackwardAttributes, compute)
            .set_name("bnb")
            .set_saved_mean_and_inv_variance(mean, inv_variance)
            .set_peer_stats(peer_stats),
        )
    _adopt_outputs(t, outputs, ["dx", "dscale", "dbias"], list)
    return graph


def _batchnorm_inference(keywords, compute):
    graph, t = _graph(compute)
    tensors = [
        t.input("x", [4, 8, 8, 8]),
        t.input("mean", [1, 8, 1, 1]),
        t.input("inv_variance", [1, 8, 1, 1]),
        t.input("scale", [1, 8, 1, 1]),
        t.input("bias", [1, 8, 1, 1]),
    ]
    if keywords:
        y = graph.batchnorm_inference(
            *tensors, **_compute_keywords(compute), name="bni"
        )
    else:
        attributes = _new_attributes(hipdnn.BatchnormInferenceAttributes, compute)
        y = graph.batchnorm_inference(*tensors, attributes.set_name("bni"))
    t.adopt(y, "y", output=True)
    return graph


# Distinct, non-default stride and dilation, so a swapped or dropped mapping in
# either overload changes the serialized graph. With an 8x8 input, a 3x3 filter
# and padding 1, they give a 4x3 output.
_STRIDE = [2, 2]
_DILATION = [1, 2]
_CONV_OUT = [2, 4, 4, 3]


def _conv_attributes(cls, pre_padding, post_padding, mode, compute):
    return (
        _new_attributes(cls, compute)
        .set_name("conv")
        .set_pre_padding(pre_padding)
        .set_post_padding(post_padding)
        .set_stride(_STRIDE)
        .set_dilation(_DILATION)
        .set_convolution_mode(mode)
    )


def _conv_kwargs(split, pre_padding, post_padding, mode, compute):
    """cuDNN keywords for either padding shape. The symmetric shape requires
    pre_padding == post_padding."""
    common = dict(
        stride=_STRIDE,
        dilation=_DILATION,
        convolution_mode=mode,
        **_compute_keywords(compute),
        name="conv",
    )
    if split:
        return dict(pre_padding=pre_padding, post_padding=post_padding, **common)
    assert pre_padding == post_padding
    return dict(padding=pre_padding, **common)


def _conv_fprop(split, pre_padding=(1, 1), post_padding=(1, 1), mode=None):
    mode = hipdnn.ConvolutionMode.CROSS_CORRELATION if mode is None else mode
    pre_padding, post_padding = list(pre_padding), list(post_padding)

    def build(keywords, compute):
        graph, t = _graph(compute)
        x = t.input("x", [2, 4, 8, 8])
        weight = t.input("w", [4, 4, 3, 3])
        if keywords:
            y = graph.conv_fprop(
                image=x,
                weight=weight,
                **_conv_kwargs(split, pre_padding, post_padding, mode, compute),
            )
        else:
            y = graph.conv_fprop(
                x,
                weight,
                _conv_attributes(
                    hipdnn.ConvFpropAttributes, pre_padding, post_padding, mode, compute
                ),
            )
        t.adopt(y, "y", output=True)
        return graph

    return build


def _conv_dgrad(split):
    def build(keywords, compute):
        graph, t = _graph(compute)
        dy = t.input("dy", _CONV_OUT)
        weight = t.input("w", [4, 4, 3, 3])
        mode = hipdnn.ConvolutionMode.CROSS_CORRELATION
        if keywords:
            dx = graph.conv_dgrad(
                loss=dy,
                filter=weight,
                **_conv_kwargs(split, [1, 1], [1, 1], mode, compute),
            )
        else:
            dx = graph.conv_dgrad(
                dy,
                weight,
                _conv_attributes(
                    hipdnn.ConvDgradAttributes, [1, 1], [1, 1], mode, compute
                ),
            )
        dx.set_dim([2, 4, 8, 8])
        t.adopt(dx, "dx", output=True)
        return graph

    return build


def _conv_wgrad(split):
    def build(keywords, compute):
        graph, t = _graph(compute)
        dy = t.input("dy", _CONV_OUT)
        x = t.input("x", [2, 4, 8, 8])
        mode = hipdnn.ConvolutionMode.CROSS_CORRELATION
        if keywords:
            dw = graph.conv_wgrad(
                image=x,
                loss=dy,
                **_conv_kwargs(split, [1, 1], [1, 1], mode, compute),
            )
        else:
            dw = graph.conv_wgrad(
                dy,
                x,
                _conv_attributes(
                    hipdnn.ConvWgradAttributes, [1, 1], [1, 1], mode, compute
                ),
            )
        dw.set_dim([4, 4, 3, 3])
        t.adopt(dw, "dw", output=True)
        return graph

    return build


def _matmul(keywords, compute):
    graph, t = _graph(compute)
    a = t.input("a", [4, 3])
    b = t.input("b", [3, 5])
    if keywords:
        c = graph.matmul(A=a, B=b, **_compute_keywords(compute), name="mm")
    else:
        attributes = _new_attributes(hipdnn.MatmulAttributes, compute)
        c = graph.matmul(a, b, attributes.set_name("mm"))
    t.adopt(c, "c", output=True)
    return graph


def _layernorm(keywords, compute):
    graph, t = _graph(compute)
    x = t.input("x", [2, 6, 4])
    scale = t.input("scale", [6, 4])
    bias = t.input("bias", [6, 4])
    epsilon = t.scalar("epsilon")
    phase = hipdnn.NormFwdPhase.TRAINING
    if keywords:
        outputs = graph.layernorm(
            phase,
            x,
            scale,
            bias,
            epsilon,
            **_compute_keywords(compute),
            name="ln",
        )
    else:
        outputs = graph.layernorm(
            x,
            scale,
            bias,
            _new_attributes(hipdnn.LayernormAttributes, compute)
            .set_name("ln")
            .set_epsilon(epsilon)
            .set_forward_phase(phase),
        )
    _adopt_outputs(t, outputs, ["y", "mean", "inv_variance"], tuple)
    return graph


def _layernorm_backward(with_stats):
    def build(keywords, compute):
        graph, t = _graph(compute)
        dy = t.input("dy", [4, 8, 2, 2])
        x = t.input("x", [4, 8, 2, 2])
        scale = t.input("scale", [1, 8, 2, 2])
        mean = t.input("mean", [4, 1, 1, 1]) if with_stats else None
        inv_variance = t.input("inv_variance", [4, 1, 1, 1]) if with_stats else None
        epsilon = t.scalar("epsilon")
        if keywords:
            outputs = graph.layernorm_backward(
                dy,
                x,
                scale,
                mean,
                inv_variance,
                *_compute_args(compute),
                name="lnb",
                epsilon=epsilon,
                normalized_dim_count=3,
            )
        else:
            attributes = (
                _new_attributes(hipdnn.LayernormBackwardAttributes, compute)
                .set_name("lnb")
                .set_epsilon(epsilon)
                .set_normalized_dim_count(3)
            )
            if with_stats:
                attributes.set_saved_mean_and_inv_variance(mean, inv_variance)
            outputs = graph.layernorm_backward(dy, x, scale, attributes)
        _adopt_outputs(t, outputs, ["dx", "dscale", "dbias"], tuple)
        return graph

    return build


def _rmsnorm(with_bias):
    def build(keywords, compute):
        graph, t = _graph(compute)
        x = t.input("x", [2, 8, 2, 2])
        scale = t.input("scale", [1, 8, 2, 2])
        bias = t.input("bias", [1, 8, 2, 2]) if with_bias else None
        epsilon = t.scalar("epsilon")
        phase = hipdnn.NormFwdPhase.TRAINING
        if keywords and with_bias:
            # cuDNN's positional order: the required epsilon follows the defaulted bias.
            args = (phase, x, scale, bias, epsilon)
            if compute is None:
                outputs = graph.rmsnorm(*args, name="rms")
            else:
                outputs = graph.rmsnorm(*args, compute, "rms")
        elif keywords:
            outputs = graph.rmsnorm(
                norm_forward_phase=phase,
                input=x,
                scale=scale,
                epsilon=epsilon,
                **_compute_keywords(compute),
                name="rms",
            )
        else:
            attributes = (
                _new_attributes(hipdnn.RMSNormAttributes, compute)
                .set_name("rms")
                .set_epsilon(epsilon)
                .set_forward_phase(phase)
            )
            if with_bias:
                attributes.set_bias(bias)
            outputs = graph.rmsnorm(x, scale, attributes)
        _adopt_outputs(t, outputs, ["y", "inv_rms"], tuple)
        return graph

    return build


def _rmsnorm_backward(has_dbias):
    def build(keywords, compute):
        graph, t = _graph(compute)
        dy = t.input("dy", [1, 8, 2, 2])
        x = t.input("x", [1, 8, 2, 2])
        scale = t.input("scale", [1, 8, 2, 2])
        inv_rms = t.input("inv_rms", [1, 1, 1, 1])
        if keywords:
            outputs = graph.rmsnorm_backward(
                dy, x, scale, inv_rms, has_dbias, *_compute_args(compute), name="rmsb"
            )
        else:
            outputs = graph.rmsnorm_backward(
                dy,
                x,
                scale,
                inv_rms,
                _new_attributes(hipdnn.RMSNormBackwardAttributes, compute)
                .set_name("rmsb")
                .set_compute_dbias(has_dbias),
            )
        assert (outputs[2] is not None) == has_dbias
        _adopt_outputs(t, outputs, ["dx", "dscale", "dbias"], tuple)
        return graph

    return build


def _block_scale_dequantize(keywords, compute):
    graph, t = _graph(compute)
    x = t.input("x", [2, 64, 32, 32])
    scale = t.input("scale", [2, 2, 32, 32])
    if keywords:
        y = graph.block_scale_dequantize(
            x,
            scale,
            [32],
            is_negative_scale=True,
            **_compute_keywords(compute),
            name="dq",
        )
    else:
        y = graph.block_scale_dequantize(
            x,
            scale,
            _new_attributes(hipdnn.BlockScaleDequantizeAttributes, compute)
            .set_name("dq")
            .set_block_size([32])
            .set_is_negative_scale(True),
        )
    # Dequantize is fused-only, so its output stays virtual.
    t.adopt(y, "y")
    return graph


def _block_scale_quantize(keywords, compute):
    graph, t = _graph(compute)
    x = t.input("x", [2, 64, 32, 32])
    if keywords:
        outputs = graph.block_scale_quantize(
            x,
            32,
            axis=1,
            transpose=True,
            **_compute_keywords(compute),
            name="q",
        )
    else:
        outputs = graph.block_scale_quantize(
            x,
            _new_attributes(hipdnn.BlockScaleQuantizeAttributes, compute)
            .set_name("q")
            .set_block_size(32)
            .set_axis(1)
            .set_transpose(True),
        )
    _adopt_outputs(t, outputs, ["y", "scale"], tuple)
    for output in outputs:
        output.set_data_type(FLOAT)
    return graph


def _reduction(keywords, compute):
    graph, t = _graph(compute)
    x = t.input("x", [4, 8])
    if keywords:
        y = graph.reduction(
            x,
            hipdnn.ReductionMode.ADD,
            **_compute_keywords(compute),
            name="red",
        )
    else:
        y = graph.reduction(
            x,
            _new_attributes(hipdnn.ReductionAttributes, compute)
            .set_name("red")
            .set_mode(hipdnn.ReductionMode.ADD),
        )
    y.set_dim([1, 8]).set_stride([8, 1])
    t.adopt(y, "y", output=True)
    return graph


def _moe_grouped_matmul(keywords, compute):
    """Unlike the other keyword overloads, this one defaults compute_data_type to
    FLOAT, as cuDNN's does, not NOT_SET."""
    graph, t = _graph(compute)
    token = t.input("token", [1, 8, 16])
    weight = t.input("weight", [2, 16, 32])
    first_token_offset = t.input("first_token_offset", [2, 1, 1], INT32)
    token_index = t.input("token_index", [1, 8, 1], INT32)
    token_ks = t.input("token_ks", [1, 8, 1], INT32)
    mode = hipdnn.MoeGroupedMatmulMode.SCATTER
    if keywords:
        output = graph.moe_grouped_matmul(
            token,
            weight,
            first_token_offset,
            token_index=token_index,
            token_ks=token_ks,
            mode=mode,
            top_k=2,
            name="moe",
            **_compute_keywords(compute),
        )
    else:
        output = graph.moe_grouped_matmul(
            token,
            weight,
            first_token_offset,
            token_index,
            token_ks,
            hipdnn.MoeGroupedMatmulAttributes()
            .set_name("moe")
            .set_mode(mode)
            .set_top_k(2)
            .set_compute_data_type(FLOAT if compute is None else compute),
        )
    t.adopt(output, "output", output=True)
    output.set_data_type(FLOAT)
    return graph


# SDPA options: each takes the _Tensors factory, creates any tensors it needs
# and returns (keywords, configure), the cuDNN keywords and a function applying
# the same settings to the attributes object. unfuse_fma has no case: it is an
# advisory hint that does not reach to_json(), so parity cannot observe it.
_SCORE_DIM = [2, 8, 16, 32]
_STATS_DIM = [2, 8, 16, 1]
_SEQ_LEN_DIM = [2, 1, 1, 1]
_SEED_DIM = [1, 1, 1, 1]


def _causal_mask(t):
    return (
        dict(attn_scale=0.125, use_causal_mask=True),
        lambda a: a.set_attn_scale(0.125).set_causal_mask(True),
    )


def _causal_mask_bottom_right(t):
    return (
        dict(use_causal_mask_bottom_right=True),
        lambda a: a.set_causal_mask_bottom_right(True),
    )


def _sliding_window(t):
    return (
        dict(sliding_window_length=7),
        lambda a: a.set_diagonal_band_left_bound(7),
    )


def _diagonal_band(t):
    return (
        dict(
            diagonal_alignment=hipdnn.DiagonalAlignment.BOTTOM_RIGHT,
            diagonal_band_left_bound=5,
            diagonal_band_right_bound=3,
        ),
        lambda a: a.set_diagonal_alignment(hipdnn.DiagonalAlignment.BOTTOM_RIGHT)
        .set_diagonal_band_left_bound(5)
        .set_diagonal_band_right_bound(3),
    )


def _attn_scale_tensor(t):
    scale = t.scalar("attn_scale", 0.125)
    return dict(attn_scale=scale), lambda a: a.set_attn_scale(scale)


def _padding_mask(t):
    seq_len_q = t.input("seq_len_q", _SEQ_LEN_DIM, INT32)
    seq_len_kv = t.input("seq_len_kv", _SEQ_LEN_DIM, INT32)
    return (
        dict(seq_len_q=seq_len_q, seq_len_kv=seq_len_kv, use_padding_mask=True),
        lambda a: a.set_seq_len_q(seq_len_q)
        .set_seq_len_kv(seq_len_kv)
        .set_padding_mask(True),
    )


def _alibi_mask(t):
    return dict(use_alibi_mask=True), lambda a: a.set_alibi_mask(True)


def _sdpa_bias(t):
    bias = t.input("bias", [1, 8, 16, 32])
    return dict(bias=bias), lambda a: a.set_bias(bias)


def _sdpa_philox_dropout(t):
    seed = t.input("seed", _SEED_DIM, INT64)
    offset = t.input("offset", _SEED_DIM, INT64)
    rng_dump = t.input("rng_dump", _SCORE_DIM)
    return (
        dict(dropout=(0.25, seed, offset), rng_dump=rng_dump),
        lambda a: a.set_dropout(0.25, seed, offset).set_rng_dump(rng_dump),
    )


def _sdpa_mask_dropout(t):
    mask = t.input("dropout_mask", _SCORE_DIM)
    scale = t.scalar("dropout_scale", 1.25)
    return (
        dict(dropout=(mask, scale)),
        lambda a: a.set_dropout_mask(mask).set_dropout_scale(scale),
    )


def _block_mask(t):
    block_mask = t.input("block_mask", [2, 8, 1, 1], hipdnn.DataType.UINT8)
    return dict(block_mask=block_mask), lambda a: a.set_block_mask(block_mask)


def _paged_attention(t):
    seq_len_keywords, seq_len_configure = _padding_mask(t)
    k_table = t.input("k_table", [2, 1, 4, 1], INT32)
    v_table = t.input("v_table", [2, 1, 4, 1], INT32)
    return (
        dict(
            paged_attention_k_table=k_table,
            paged_attention_v_table=v_table,
            paged_attention_max_seq_len_kv=128,
            **seq_len_keywords,
        ),
        lambda a: seq_len_configure(a)
        .set_paged_attention_k_table(k_table)
        .set_paged_attention_v_table(v_table)
        .set_paged_attention_max_seq_len_kv(128),
    )


def _score_outputs(t):
    score_max = t.input("score_max", _STATS_DIM)
    score_sum_exp = t.input("score_sum_exp", _STATS_DIM)
    return (
        dict(score_max=score_max, score_sum_exp=score_sum_exp),
        lambda a: a.set_logit_max(score_max).set_score_sum_exp(score_sum_exp),
    )


def _sink_token(t):
    sink_token = t.input("sink_token", [1, 8, 1, 1])
    return dict(sink_token=sink_token), lambda a: a.set_sink_token(sink_token)


def _implementation(t):
    implementation = hipdnn.AttentionImplementation.COMPOSITE
    return (
        dict(implementation=implementation),
        lambda a: a.set_implementation(implementation),
    )


def _sdpa_backward_bias(t):
    bias = t.input("bias", [1, 8, 16, 32])
    dbias = t.input("dbias", [1, 8, 16, 32])
    return dict(bias=bias, dBias=dbias), lambda a: a.set_bias(bias).set_dbias(dbias)


def _sdpa_backward_philox_dropout(t):
    seed = t.input("seed", _SEED_DIM, INT64)
    offset = t.input("offset", _SEED_DIM, INT64)
    return (
        dict(dropout=(0.25, seed, offset)),
        lambda a: a.set_dropout(0.25, seed, offset),
    )


def _sdpa_backward_mask_dropout(t):
    mask = t.input("dropout_mask", _SCORE_DIM)
    scale = t.scalar("dropout_scale", 1.25)
    scale_inv = t.scalar("dropout_scale_inv", 0.8)
    return (
        dict(dropout=(mask, scale, scale_inv)),
        lambda a: a.set_dropout_mask(mask)
        .set_dropout_scale(scale)
        .set_dropout_scale_inv(scale_inv),
    )


def _sdpa(options):
    def build(keywords, compute):
        graph, t = _graph(compute)
        q = t.input("q", [2, 8, 16, 64])
        k = t.input("k", [2, 8, 32, 64])
        v = t.input("v", [2, 8, 32, 64])
        option_keywords, configure = options(t)
        if keywords:
            outputs = graph.sdpa(
                q,
                k,
                v,
                is_inference=False,
                **_compute_keywords(compute),
                name="sdpa",
                **option_keywords,
            )
        else:
            attributes = (
                _new_attributes(hipdnn.SdpaAttributes, compute)
                .set_name("sdpa")
                .set_generate_stats(True)
            )
            configure(attributes)
            outputs = graph.sdpa(q, k, v, attributes)
        _adopt_outputs(t, outputs, ["o", "stats"], tuple)
        return graph

    return build


def _sdpa_backward(options):
    def build(keywords, compute):
        graph, t = _graph(compute)
        q = t.input("q", [2, 8, 16, 64])
        k = t.input("k", [2, 8, 32, 64])
        v = t.input("v", [2, 8, 32, 64])
        o = t.input("o", [2, 8, 16, 64])
        d_o = t.input("do", [2, 8, 16, 64])
        stats = t.input("stats", _STATS_DIM)
        option_keywords, configure = options(t)
        if keywords:
            outputs = graph.sdpa_backward(
                q,
                k,
                v,
                o,
                d_o,
                stats,
                **_compute_keywords(compute),
                name="sdpab",
                **option_keywords,
            )
        else:
            attributes = _new_attributes(
                hipdnn.SdpaBackwardAttributes, compute
            ).set_name("sdpab")
            configure(attributes)
            outputs = graph.sdpa_backward(q, k, v, o, d_o, stats, attributes)
        _adopt_outputs(t, outputs, ["dq", "dk", "dv"], tuple)
        return graph

    return build


def _sdpa_case(options, case_id):
    return pytest.param(_sdpa(options), id=f"sdpa-{case_id}", marks=_requires_sdpa)


def _sdpa_backward_case(options, case_id):
    return pytest.param(
        _sdpa_backward(options), id=f"sdpa_backward-{case_id}", marks=_requires_sdpa
    )


_PARITY_CASES = [
    pytest.param(_batchnorm, id="batchnorm"),
    pytest.param(_batchnorm_backward, id="batchnorm_backward"),
    pytest.param(_batchnorm_inference, id="batchnorm_inference"),
    pytest.param(_conv_fprop(split=False), id="conv_fprop-padding"),
    pytest.param(_conv_fprop(split=True), id="conv_fprop-pre_post_padding"),
    pytest.param(
        _conv_fprop(
            split=True,
            pre_padding=(1, 1),
            post_padding=(2, 2),
            mode=hipdnn.ConvolutionMode.CONVOLUTION,
        ),
        id="conv_fprop-asymmetric-convolution_mode",
    ),
    pytest.param(_conv_dgrad(split=False), id="conv_dgrad-padding"),
    pytest.param(_conv_dgrad(split=True), id="conv_dgrad-pre_post_padding"),
    pytest.param(_conv_wgrad(split=False), id="conv_wgrad-padding"),
    pytest.param(_conv_wgrad(split=True), id="conv_wgrad-pre_post_padding"),
    pytest.param(_matmul, id="matmul"),
    pytest.param(_layernorm, id="layernorm"),
    pytest.param(_layernorm_backward(with_stats=True), id="layernorm_backward"),
    pytest.param(
        _layernorm_backward(with_stats=False), id="layernorm_backward-no_stats"
    ),
    pytest.param(_rmsnorm(with_bias=False), id="rmsnorm"),
    pytest.param(_rmsnorm(with_bias=True), id="rmsnorm-bias"),
    pytest.param(_rmsnorm_backward(has_dbias=True), id="rmsnorm_backward-dbias"),
    pytest.param(_rmsnorm_backward(has_dbias=False), id="rmsnorm_backward-no_dbias"),
    pytest.param(_block_scale_dequantize, id="block_scale_dequantize"),
    pytest.param(_block_scale_quantize, id="block_scale_quantize"),
    pytest.param(_reduction, id="reduction"),
    pytest.param(_moe_grouped_matmul, id="moe_grouped_matmul"),
    _sdpa_case(_causal_mask, "causal_mask"),
    _sdpa_case(_causal_mask_bottom_right, "causal_mask_bottom_right"),
    _sdpa_case(_sliding_window, "sliding_window"),
    _sdpa_case(_diagonal_band, "diagonal_band"),
    _sdpa_case(_attn_scale_tensor, "attn_scale_tensor"),
    _sdpa_case(_padding_mask, "padding_mask"),
    _sdpa_case(_alibi_mask, "alibi_mask"),
    _sdpa_case(_sdpa_bias, "bias"),
    _sdpa_case(_sdpa_philox_dropout, "philox_dropout"),
    _sdpa_case(_sdpa_mask_dropout, "mask_dropout"),
    _sdpa_case(_block_mask, "block_mask"),
    _sdpa_case(_paged_attention, "paged_attention"),
    _sdpa_case(_score_outputs, "score_outputs"),
    _sdpa_case(_sink_token, "sink_token"),
    _sdpa_case(_implementation, "implementation"),
    _sdpa_backward_case(_causal_mask, "causal_mask"),
    _sdpa_backward_case(_causal_mask_bottom_right, "causal_mask_bottom_right"),
    _sdpa_backward_case(_sliding_window, "sliding_window"),
    _sdpa_backward_case(_diagonal_band, "diagonal_band"),
    _sdpa_backward_case(_attn_scale_tensor, "attn_scale_tensor"),
    _sdpa_backward_case(_padding_mask, "padding_mask"),
    _sdpa_backward_case(_alibi_mask, "alibi_mask"),
    _sdpa_backward_case(_sdpa_backward_bias, "bias_dbias"),
    _sdpa_backward_case(_sdpa_backward_philox_dropout, "philox_dropout"),
    _sdpa_backward_case(_sdpa_backward_mask_dropout, "mask_dropout"),
]


@pytest.mark.parametrize("build", _PARITY_CASES)
def test_keyword_overload_serializes_like_attributes_overload(build):
    _assert_parity(build, OP_COMPUTE)


@pytest.mark.parametrize(
    "graph_compute", [OP_COMPUTE, FLOAT], ids=["op_compute_graph", "float_graph"]
)
@pytest.mark.parametrize("build", _PARITY_CASES)
def test_keyword_overload_default_compute_type_matches_attributes_overload(
    build, graph_compute, monkeypatch
):
    monkeypatch.setitem(globals(), "_graph_compute_without_op_compute", graph_compute)
    _assert_parity(build, None)


# --- Dispatch ---


class TestDispatch:
    """The attributes overloads keep every call shape; keyword overloads take the rest."""

    def test_attributes_mixed_with_keywords_raises_type_error(self):
        graph = create_float_graph()
        dy = hipdnn.Tensor.create([4, 8, 2, 2], FLOAT)
        x = hipdnn.Tensor.create([4, 8, 2, 2], FLOAT)
        scale = hipdnn.Tensor.create([1, 8, 2, 2], FLOAT)
        a = hipdnn.Tensor.create([4, 3], FLOAT)
        b = hipdnn.Tensor.create([3, 5], FLOAT)

        with pytest.raises(TypeError):
            graph.layernorm_backward(
                dy, x, scale, hipdnn.LayernormBackwardAttributes(), name="n"
            )
        with pytest.raises(TypeError):
            graph.conv_fprop(x, scale, hipdnn.ConvFpropAttributes(), padding=[1, 1])
        with pytest.raises(TypeError):
            graph.matmul(a, b, hipdnn.MatmulAttributes(), name="mm")

    def test_layernorm_backward_dispatches_by_fourth_argument(self):
        graph = create_float_graph()
        dy = hipdnn.Tensor.create([4, 8, 2, 2], FLOAT)
        x = hipdnn.Tensor.create([4, 8, 2, 2], FLOAT)
        scale = hipdnn.Tensor.create([1, 8, 2, 2], FLOAT)
        mean = hipdnn.Tensor.create([4, 1, 1, 1], FLOAT)
        inv_variance = hipdnn.Tensor.create([4, 1, 1, 1], FLOAT)

        by_attributes = graph.layernorm_backward(
            dy, x, scale, hipdnn.LayernormBackwardAttributes().set_name("attrs")
        )
        by_keywords = graph.layernorm_backward(
            dy, x, scale, mean, inv_variance, name="kw"
        )
        assert isinstance(by_attributes, tuple) and len(by_attributes) == 3
        assert isinstance(by_keywords, tuple) and len(by_keywords) == 3
        assert by_attributes[0].get_name() == "attrs::DX"
        assert by_keywords[0].get_name() == "kw::DX"

    def test_none_accepted_for_optional_tensors(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 8, 2, 2], FLOAT)
        channel = [1, 8, 1, 1]

        outputs = graph.batchnorm(
            x,
            hipdnn.Tensor.create(channel, FLOAT),
            hipdnn.Tensor.create(channel, FLOAT),
            None,
            None,
            scalar_tensor(),
            None,
        )
        assert isinstance(outputs, list) and len(outputs) == 5
        assert outputs[3] is None and outputs[4] is None

        outputs = graph.batchnorm_backward(
            x,
            x,
            hipdnn.Tensor.create(channel, FLOAT),
            mean=None,
            inv_variance=None,
        )
        assert isinstance(outputs, list) and len(outputs) == 3

        outputs = graph.layernorm_backward(
            x,
            x,
            hipdnn.Tensor.create([1, 8, 2, 2], FLOAT),
            mean=None,
            inv_variance=None,
            epsilon=None,
            normalized_dim_count=None,
        )
        assert isinstance(outputs, tuple) and len(outputs) == 3

        outputs = graph.rmsnorm(
            hipdnn.NormFwdPhase.INFERENCE,
            x,
            hipdnn.Tensor.create([1, 8, 2, 2], FLOAT),
            None,
            scalar_tensor(),
        )
        assert isinstance(outputs, tuple) and len(outputs) == 2

        outputs = graph.block_scale_quantize(x, 2, axis=None)
        assert isinstance(outputs, tuple) and len(outputs) == 2

        output = graph.moe_grouped_matmul(
            hipdnn.Tensor.create([1, 8, 16], FLOAT),
            hipdnn.Tensor.create([2, 16, 32], FLOAT),
            hipdnn.Tensor.create([2, 1, 1], hipdnn.DataType.INT32),
            token_index=None,
            token_ks=None,
        )
        assert isinstance(output, hipdnn.Tensor)

    def test_matmul_rejects_nonzero_padding(self):
        graph = create_float_graph()
        a = hipdnn.Tensor.create([4, 3], FLOAT)
        b = hipdnn.Tensor.create([3, 5], FLOAT)

        assert isinstance(graph.matmul(a, b, padding=0.0), hipdnn.Tensor)
        with pytest.raises(NotImplementedError, match="padding"):
            graph.matmul(a, b, padding=1.0)

    def test_reduction_dispatches_by_second_argument(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 8], FLOAT)
        y = hipdnn.Tensor.create([1, 8], FLOAT)
        attributes = hipdnn.ReductionAttributes().set_mode(hipdnn.ReductionMode.ADD)

        assert isinstance(graph.reduction(x, attributes), hipdnn.Tensor)
        assert graph.reduction(x, y.set_name("out"), attributes).get_name() == "out"
        assert isinstance(graph.reduction(x, hipdnn.ReductionMode.MAX), hipdnn.Tensor)

    def test_named_batchnorm_inference_attributes_call_still_resolves(self):
        graph = create_float_graph()
        channel = [1, 8, 1, 1]
        y = graph.batchnorm_inference(
            x=hipdnn.Tensor.create([4, 8, 2, 2], FLOAT),
            mean=hipdnn.Tensor.create(channel, FLOAT),
            inv_variance=hipdnn.Tensor.create(channel, FLOAT),
            scale=hipdnn.Tensor.create(channel, FLOAT),
            bias=hipdnn.Tensor.create(channel, FLOAT),
            attributes=hipdnn.BatchnormInferenceAttributes().set_name("named"),
        )
        assert y.get_name() == "named::Y"


@_requires_sdpa
class TestSdpaDispatch:
    """cuDNN's SDPA argument rules and the parameters hipDNN cannot honour."""

    @staticmethod
    def _qkv():
        return (
            hipdnn.Tensor.create([2, 8, 16, 64], FLOAT),
            hipdnn.Tensor.create([2, 8, 32, 64], FLOAT),
            hipdnn.Tensor.create([2, 8, 32, 64], FLOAT),
        )

    @staticmethod
    def _backward_inputs():
        q, k, v = TestSdpaDispatch._qkv()
        o = hipdnn.Tensor.create([2, 8, 16, 64], FLOAT)
        d_o = hipdnn.Tensor.create([2, 8, 16, 64], FLOAT)
        stats = hipdnn.Tensor.create([2, 8, 16, 1], FLOAT)
        return q, k, v, o, d_o, stats

    def test_exactly_one_stats_flag_is_required(self):
        graph = create_float_graph()
        with pytest.raises(RuntimeError):
            graph.sdpa(*self._qkv())
        with pytest.raises(RuntimeError):
            graph.sdpa(*self._qkv(), is_inference=True, generate_stats=False)

        o, stats = graph.sdpa(*self._qkv(), generate_stats=False)
        assert stats is None
        o, stats = graph.sdpa(*self._qkv(), is_inference=False)
        assert stats is not None

    @pytest.mark.parametrize(
        "options, message",
        [
            (
                dict(use_causal_mask=True, diagonal_band_right_bound=0),
                "use_causal_mask and diagonal_band_right_bound",
            ),
            (
                dict(use_causal_mask=True, use_causal_mask_bottom_right=True),
                "cannot both be true",
            ),
            (
                dict(use_causal_mask_bottom_right=True, diagonal_band_right_bound=0),
                "use_causal_mask_bottom_right and diagonal_band_right_bound",
            ),
            (
                dict(sliding_window_length=4, diagonal_band_left_bound=4),
                "sliding window and left_bound",
            ),
        ],
        ids=[
            "causal-right_bound",
            "causal-causal_bottom_right",
            "causal_bottom_right-right_bound",
            "sliding_window-left_bound",
        ],
    )
    def test_conflicting_mask_options_raise(self, options, message):
        graph = create_float_graph()
        with pytest.raises(RuntimeError, match=message):
            graph.sdpa(*self._qkv(), generate_stats=True, **options)

    def test_optional_inputs_accept_none(self):
        graph = create_float_graph()
        outputs = graph.sdpa(
            *self._qkv(),
            is_inference=False,
            attn_scale=None,
            bias=None,
            seq_len_q=None,
            seq_len_kv=None,
            dropout=None,
            rng_dump=None,
            sink_token=None,
        )
        assert isinstance(outputs, tuple) and len(outputs) == 2

        outputs = graph.sdpa_backward(
            *self._backward_inputs(),
            attn_scale=None,
            bias=None,
            dBias=None,
            dropout=None,
        )
        assert isinstance(outputs, tuple) and len(outputs) == 3

    def test_attn_scale_accepts_tensor(self):
        graph = create_float_graph()
        scale = scalar_tensor()
        outputs = graph.sdpa(*self._qkv(), generate_stats=True, attn_scale=scale)
        assert isinstance(outputs, tuple) and len(outputs) == 2

    def test_unsupported_parameters_raise_not_implemented(self):
        graph = create_float_graph()
        with pytest.raises(NotImplementedError, match="score_mod"):
            graph.sdpa(*self._qkv(), generate_stats=True, score_mod=lambda *args: None)
        with pytest.raises(NotImplementedError, match="use_deterministic_algorithm"):
            graph.sdpa_backward(
                *self._backward_inputs(), use_deterministic_algorithm=True
            )
        with pytest.raises(NotImplementedError, match="max_total_seq_len_q"):
            graph.sdpa_backward(*self._backward_inputs(), max_total_seq_len_q=64)


# --- GPU tier: keyword-built graphs through plan build and stub execution ---


@pytest.mark.gpu
class TestKeywordGraphsExecute:
    def test_conv_fprop(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([16, 16, 16, 16], FLOAT)
        weight = hipdnn.Tensor.create([16, 16, 3, 3], FLOAT)
        y = graph.conv_fprop(x, weight, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
        y.set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(
            graph, [(x, np.float32), (weight, np.float32), (y, np.float32)], handle
        )

    def test_matmul(self):
        graph = create_float_graph()
        a = hipdnn.Tensor.create([4, 3], FLOAT)
        b = hipdnn.Tensor.create([3, 5], FLOAT)
        c = graph.matmul(a, b)
        c.set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(
            graph, [(a, np.float32), (b, np.float32), (c, np.float32)], handle
        )

    def test_layernorm(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 6, 4], FLOAT)
        scale = hipdnn.Tensor.create([6, 4], FLOAT)
        bias = hipdnn.Tensor.create([6, 4], FLOAT)
        y, mean, inv_variance = graph.layernorm(
            hipdnn.NormFwdPhase.INFERENCE, x, scale, bias, scalar_tensor()
        )
        assert (mean, inv_variance) == (None, None)
        y.set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (scale, np.float32), (bias, np.float32), (y, np.float32)],
            handle,
        )

    def test_rmsnorm(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 64, 32, 32], FLOAT)
        scale = hipdnn.Tensor.create([1, 64, 32, 32], FLOAT)
        y, inv_rms = graph.rmsnorm(
            hipdnn.NormFwdPhase.TRAINING, x, scale, epsilon=scalar_tensor()
        )
        y.set_output(True)
        inv_rms.set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (x, np.float32),
                (scale, np.float32),
                (y, np.float32),
                (inv_rms, np.float32),
            ],
            handle,
        )
