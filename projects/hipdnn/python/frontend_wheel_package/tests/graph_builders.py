# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared graph builders for hipDNN Python binding tests.

One ``build_<op>_graph()`` per graph-construction operation exposed by the
bindings. Each returns ``(graph, *tensors)`` -- the graph first, then every
tensor a caller needs to assemble a variant pack (inputs, then outputs).

Graphs come back execution-ready: outputs are marked ``set_output(True)`` and
carry whatever ``set_dim``/``set_stride``/``set_data_type`` the operation needs
to build plans. The one deliberate exception is
:func:`build_block_scale_dequantize_graph`, whose ``y`` must stay virtual
because dequantize is a fused-only operation.

Callers that only need topology (serialization round-trips) take ``[0]``.
"""

import hipdnn_frontend as hipdnn

from .helpers import create_float_graph


def _scalar_tensor(value=1e-5):
    """A [1]-shaped FLOAT tensor holding a compile-time scalar (epsilon)."""
    tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
    tensor.set_value(value)
    return tensor


def _unpack(outputs, count, kind=tuple):
    """Return a node's multi-output container, asserting its bound type and arity.

    ``kind`` is the Python type the binding is contracted to produce, and it
    differs by how the node is bound. Lambdas that call ``nb::make_tuple``
    yield a ``tuple``; methods bound directly off a C++ signature returning
    ``std::array``/``std::vector`` (batchnorm, batchnorm_backward, custom_op)
    yield a ``list``. Unpacking alone catches a changed arity but not a
    changed container, and it reports as a bare ValueError from inside a
    builder.
    """
    assert isinstance(
        outputs, kind
    ), f"expected {kind.__name__}, got {type(outputs).__name__}"
    assert len(outputs) == count, f"expected {count} outputs, got {len(outputs)}"
    return outputs


def build_pointwise_add_graph(n=16, c=16, h=16, w=16):
    """Build an elementwise-add pointwise graph returning (graph, a, b, out)."""
    graph = create_float_graph()
    graph.set_name("pointwise_add_test")

    a = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    a.set_name("in_0")

    b = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    b.set_name("in_1")

    attrs = hipdnn.PointwiseAttributes()
    attrs.set_name("pointwise_add_node")
    attrs.set_mode(hipdnn.PointwiseMode.ADD)

    out = graph.pointwise(a, b, attrs)
    out.set_name("out_0")
    out.set_output(True)

    return graph, a, b, out


def build_matmul_graph(m=4, k=3, n=5):
    """Build a matmul graph (A [M, K] x B [K, N] -> C [M, N])."""
    graph = create_float_graph()
    graph.set_name("matmul_test")

    a = hipdnn.Tensor.create([m, k], hipdnn.DataType.FLOAT)
    a.set_name("a")

    b = hipdnn.Tensor.create([k, n], hipdnn.DataType.FLOAT)
    b.set_name("b")

    attrs = hipdnn.MatmulAttributes()
    attrs.set_name("matmul_node")

    c = graph.matmul(a, b, attrs)
    c.set_name("c")
    c.set_output(True)

    return graph, a, b, c


def build_conv_fprop_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_fprop graph returning (graph, x, weight, y)."""
    graph = create_float_graph()
    graph.set_name("conv_fprop_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("w")

    conv_attrs = hipdnn.ConvFpropAttributes()
    conv_attrs.set_name("conv_fprop_node")
    conv_attrs.set_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    y = graph.conv_fprop(x, weight, conv_attrs)
    y.set_name("y")
    y.set_output(True)

    return graph, x, weight, y


def build_conv_dgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_dgrad graph returning (graph, dy, weight, dx)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_dgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("dy")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("w")

    conv_attrs = hipdnn.ConvDgradAttributes()
    conv_attrs.set_name("conv_dgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dx = graph.conv_dgrad(dy, weight, conv_attrs)
    dx.set_dim([n, c, h, w])
    dx.set_name("dx")
    dx.set_output(True)

    return graph, dy, weight, dx


def build_conv_wgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_wgrad graph returning (graph, dy, x, dw)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_wgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("dy")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    conv_attrs = hipdnn.ConvWgradAttributes()
    conv_attrs.set_name("conv_wgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dw = graph.conv_wgrad(dy, x, conv_attrs)
    dw.set_dim([k, c, r, s])
    dw.set_name("dw")
    dw.set_output(True)

    return graph, dy, x, dw


def build_batchnorm_training_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm (training) graph.

    Returns (graph, x, scale, bias, y, mean, inv_variance). Per-channel
    scale/bias use [1, C, 1, 1] shapes; mean and inv_variance are produced as
    per-channel outputs.
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_training_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias.set_name("bias")

    epsilon = hipdnn.Tensor()
    epsilon.set_name("epsilon")
    epsilon.set_value(1e-5)

    attrs = hipdnn.BatchnormAttributes()
    attrs.set_name("batchnorm_node")
    attrs.set_epsilon(epsilon)

    outputs = graph.batchnorm(x, scale, bias, attrs)
    y, mean, inv_variance, _next_mean, _next_var = _unpack(outputs, 5, list)
    y.set_name("y")
    y.set_output(True)
    mean.set_name("mean")
    mean.set_output(True)
    inv_variance.set_name("inv_variance")
    inv_variance.set_output(True)

    return graph, x, scale, bias, y, mean, inv_variance


def build_batchnorm_backward_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm_backward graph.

    Returns (graph, dy, x, scale, dx, dscale, dbias).
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_backward_test")

    dy = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    dy.set_name("dy")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    attrs = hipdnn.BatchnormBackwardAttributes()
    attrs.set_name("batchnorm_backward_node")

    outputs = graph.batchnorm_backward(dy, x, scale, attrs)
    dx, dscale, dbias = _unpack(outputs, 3, list)
    dx.set_name("dx")
    dx.set_output(True)
    dscale.set_name("dscale")
    dscale.set_output(True)
    dbias.set_name("dbias")
    dbias.set_output(True)

    return graph, dy, x, scale, dx, dscale, dbias


def build_batchnorm_inference_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm inference graph.

    Returns (graph, x, mean, inv_variance, scale, bias, y). Per-channel
    mean/inv_variance/scale/bias use [1, C, 1, 1] shapes.
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_inference_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    mean = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    mean.set_name("mean")

    inv_variance = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    inv_variance.set_name("inv_variance")

    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias.set_name("bias")

    attrs = hipdnn.BatchnormInferenceAttributes()
    attrs.set_name("batchnorm_inference_node")

    y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attrs)
    y.set_name("y")
    y.set_output(True)

    return graph, x, mean, inv_variance, scale, bias, y


def build_batchnorm_inference_variance_graph(n=4, c=8, h=8, w=8):
    """Build batchnorm inference from variance and a compile-time epsilon.

    Returns (graph, x, mean, variance, scale, bias, y).
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_inference_variance_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")
    mean = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    mean.set_name("mean")
    variance = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    variance.set_name("variance")
    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")
    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias.set_name("bias")
    epsilon = hipdnn.Tensor()
    epsilon.set_value(1e-5)
    epsilon.set_name("epsilon")

    y = graph.batchnorm_inference_variance_ext(
        x,
        mean,
        variance,
        scale,
        bias,
        epsilon,
        hipdnn.BatchnormInferenceAttributesVarianceExt(),
    )
    y.set_name("y")
    y.set_output(True)

    return graph, x, mean, variance, scale, bias, y


def build_layernorm_graph():
    """Build an inference-phase layernorm graph.

    Returns (graph, x, scale, bias, y, mean, inv_variance); ``mean`` and
    ``inv_variance`` are None in the INFERENCE phase.
    """
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 6, 4], hipdnn.DataType.FLOAT)
    x.set_name("x")
    scale = hipdnn.Tensor.create([6, 4], hipdnn.DataType.FLOAT)
    scale.set_name("scale")
    bias = hipdnn.Tensor.create([6, 4], hipdnn.DataType.FLOAT)
    bias.set_name("bias")
    epsilon = _scalar_tensor()
    epsilon.set_name("epsilon")

    outputs = graph.layernorm(
        x,
        scale,
        bias,
        hipdnn.LayernormAttributes()
        .set_epsilon(epsilon)
        .set_forward_phase(hipdnn.NormFwdPhase.INFERENCE),
    )
    y, mean, inv_variance = _unpack(outputs, 3)
    y.set_name("y")
    y.set_output(True)

    return graph, x, scale, bias, y, mean, inv_variance


def build_layernorm_backward_graph():
    """Build a layernorm_backward graph.

    Returns (graph, dy, x, scale, dx, dscale, dbias).
    """
    graph = create_float_graph()
    dy = hipdnn.Tensor.create([16, 64, 32, 32], hipdnn.DataType.FLOAT)
    dy.set_name("dy")
    x = hipdnn.Tensor.create([16, 64, 32, 32], hipdnn.DataType.FLOAT)
    x.set_name("x")
    scale = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
    scale.set_name("scale")
    mean = hipdnn.Tensor.create([16, 1, 1, 1], hipdnn.DataType.FLOAT)
    mean.set_name("mean")
    inv_variance = hipdnn.Tensor.create([16, 1, 1, 1], hipdnn.DataType.FLOAT)
    inv_variance.set_name("inv_variance")
    epsilon = _scalar_tensor()
    epsilon.set_name("epsilon")

    outputs = graph.layernorm_backward(
        dy,
        x,
        scale,
        hipdnn.LayernormBackwardAttributes()
        .set_mean(mean)
        .set_inv_variance(inv_variance)
        .set_epsilon(epsilon),
    )
    dx, dscale, dbias = _unpack(outputs, 3)
    dx.set_name("dx")
    dscale.set_name("dscale")
    dbias.set_name("dbias")
    dx.set_output(True)
    dscale.set_output(True)
    dbias.set_output(True)

    return graph, dy, x, scale, dx, dscale, dbias


def build_rmsnorm_graph():
    """Build a training-phase rmsnorm graph returning (graph, x, scale, y, inv_rms)."""
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT)
    x.set_name("x")
    scale = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
    scale.set_name("scale")
    epsilon = _scalar_tensor()
    epsilon.set_name("epsilon")

    outputs = graph.rmsnorm(
        x,
        scale,
        hipdnn.RMSNormAttributes()
        .set_epsilon(epsilon)
        .set_forward_phase(hipdnn.NormFwdPhase.TRAINING),
    )
    y, inv_rms = _unpack(outputs, 2)
    y.set_name("y")
    inv_rms.set_name("inv_rms")
    y.set_output(True)
    inv_rms.set_output(True)

    return graph, x, scale, y, inv_rms


def build_rmsnorm_backward_graph():
    """Build an rmsnorm_backward graph.

    Returns (graph, dy, x, scale, inv_rms, dx, dscale, dbias); ``dbias`` is
    None because ``set_compute_dbias(True)`` is not requested.
    """
    graph = create_float_graph()
    dy = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
    dy.set_name("dy")
    x = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
    x.set_name("x")
    scale = hipdnn.Tensor.create([1, 64, 32, 32], hipdnn.DataType.FLOAT)
    scale.set_name("scale")
    inv_rms = hipdnn.Tensor.create([1, 1, 1, 1], hipdnn.DataType.FLOAT)
    inv_rms.set_name("inv_rms")

    outputs = graph.rmsnorm_backward(
        dy, x, scale, inv_rms, hipdnn.RMSNormBackwardAttributes()
    )
    dx, dscale, dbias = _unpack(outputs, 3)
    dx.set_name("dx")
    dscale.set_name("dscale")
    dx.set_output(True)
    dscale.set_output(True)

    return graph, dy, x, scale, inv_rms, dx, dscale, dbias


def build_block_scale_dequantize_graph():
    """Build a block_scale_dequantize graph returning (graph, x, scale, y).

    ``y`` is deliberately left virtual: dequantize is a fused-only operation
    with no standalone execution, so marking it an output breaks plan build.
    """
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT)
    x.set_name("x")
    scale = hipdnn.Tensor.create([2, 2, 32, 32], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    y = graph.block_scale_dequantize(
        x,
        scale,
        hipdnn.BlockScaleDequantizeAttributes().set_block_size([32]),
    )
    y.set_name("y")

    return graph, x, scale, y


def build_block_scale_quantize_graph():
    """Build a block_scale_quantize graph returning (graph, x, y, scale)."""
    graph = create_float_graph()
    x = hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT)
    x.set_name("x")

    outputs = graph.block_scale_quantize(
        x,
        hipdnn.BlockScaleQuantizeAttributes().set_block_size(32),
    )
    y, scale = _unpack(outputs, 2)
    y.set_name("y")
    scale.set_name("scale")
    for output in (y, scale):
        output.set_output(True)
        output.set_data_type(hipdnn.DataType.FLOAT)

    return graph, x, y, scale


def build_reduction_graph(with_output_tensor=False):
    """Build an add-reduction graph over [4, 8] -> [1, 8].

    Returns (graph, x, output, node_output), where ``node_output`` is whatever
    ``graph.reduction`` handed back. With ``with_output_tensor`` the output
    tensor is created here and passed into the node (which returns that same
    object); otherwise the node allocates it and it is shaped here.
    """
    graph = create_float_graph()
    x = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)
    x.set_name("in")
    attrs = hipdnn.ReductionAttributes().set_mode(hipdnn.ReductionMode.ADD)

    if with_output_tensor:
        output = hipdnn.Tensor.create([1, 8], hipdnn.DataType.FLOAT)
        output.set_name("out")
        output.set_output(True)
        node_output = graph.reduction(x, output, attrs)
    else:
        output = node_output = graph.reduction(x, attrs)
        output.set_name("out")
        output.set_dim([1, 8]).set_stride([8, 1]).set_output(True)

    return graph, x, output, node_output


def build_moe_grouped_matmul_graph():
    """Build a scatter-mode MoE grouped matmul graph.

    Returns (graph, token, weight, first_token_offset, token_index, token_ks,
    output).
    """
    graph = create_float_graph()
    token = hipdnn.Tensor.create([1, 8, 16], hipdnn.DataType.FLOAT)
    token.set_name("token")
    weight = hipdnn.Tensor.create([2, 16, 32], hipdnn.DataType.FLOAT)
    weight.set_name("weight")
    first_token_offset = hipdnn.Tensor.create([2, 1, 1], hipdnn.DataType.INT32)
    first_token_offset.set_name("first_token_offset")
    token_index = hipdnn.Tensor.create([1, 8, 1], hipdnn.DataType.INT32)
    token_index.set_name("token_index")
    token_ks = hipdnn.Tensor.create([1, 8, 1], hipdnn.DataType.INT32)
    token_ks.set_name("token_ks")

    output = graph.moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        token_index,
        token_ks,
        hipdnn.MoeGroupedMatmulAttributes()
        .set_mode(hipdnn.MoeGroupedMatmulMode.SCATTER)
        .set_top_k(2),
    )
    output.set_name("output")
    output.set_output(True)
    output.set_data_type(hipdnn.DataType.FLOAT)

    return graph, token, weight, first_token_offset, token_index, token_ks, output


def build_custom_op_graph():
    """Build a two-in/two-out custom-op graph returning (graph, a, b, y0, y1)."""
    graph = create_float_graph()
    a = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)
    b = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)

    outputs = graph.custom_op(
        [a, b],
        2,
        hipdnn.CustomOpAttributes().set_custom_op_id("example.identity"),
    )
    y0, y1 = _unpack(outputs, 2, list)
    for output in (y0, y1):
        output.set_dim([4, 8]).set_stride([8, 1]).set_output(True)
        output.set_data_type(hipdnn.DataType.FLOAT)

    return graph, a, b, y0, y1


def _resample_fwd_attributes():
    return (
        hipdnn.ResampleFwdAttributes()
        .set_resample_mode(hipdnn.ResampleMode.MAXPOOL)
        .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
        .set_pre_padding([0, 0])
        .set_post_padding([0, 0])
        .set_stride([2, 2])
        .set_window([2, 2])
    )


def build_resample_graph():
    """Build a maxpool graph via the tuple-returning ``resample`` overload.

    Returns (graph, x, y, index); ``index`` is None because
    ``set_generate_index(True)`` is not requested.
    """
    graph = create_float_graph()
    x = hipdnn.Tensor.create([1, 3, 4, 4], hipdnn.DataType.FLOAT)
    x.set_name("x")

    y, index = _unpack(graph.resample(x, _resample_fwd_attributes()), 2)
    y.set_name("y")
    y.set_output(True)

    return graph, x, y, index


def build_resample_fwd_graph():
    """Build a maxpool ``resample_fwd`` graph returning (graph, x, y)."""
    graph = create_float_graph()
    x = hipdnn.Tensor.create([1, 3, 4, 4], hipdnn.DataType.FLOAT)
    x.set_name("x")

    y = graph.resample_fwd(x, _resample_fwd_attributes())
    y.set_name("y")
    y.set_output(True)
    y.set_data_type(hipdnn.DataType.FLOAT)

    return graph, x, y


def build_resample_bwd_graph(with_index=False):
    """Build a ``resample_bwd`` graph returning (graph, dy, index, dx).

    By default this is an average-pool backward pass and ``index`` is None.
    With ``with_index`` it becomes a maxpool backward pass driven by a real
    INT32 index tensor passed through the optional ``index=`` argument.
    """
    graph = create_float_graph()
    dy = hipdnn.Tensor.create([1, 3, 16, 16], hipdnn.DataType.FLOAT)
    dy.set_name("dy")

    if with_index:
        attrs = (
            hipdnn.ResampleBwdAttributes()
            .set_resample_mode(hipdnn.ResampleMode.MAXPOOL)
            .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
            .set_pre_padding([0, 0])
            .set_post_padding([0, 0])
            .set_stride([2, 2])
            .set_window([2, 2])
        )
        index = hipdnn.Tensor.create([1, 3, 16, 16], hipdnn.DataType.INT32)
        index.set_name("index")
        dx = graph.resample_bwd(dy, attrs, index=index)
    else:
        attrs = (
            hipdnn.ResampleBwdAttributes()
            .set_resample_mode(hipdnn.ResampleMode.AVGPOOL_EXCLUDE_PADDING)
            .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
            .set_pre_padding([1, 1])
            .set_post_padding([1, 1])
            .set_stride([2, 2])
            .set_window([3, 3])
        )
        index = None
        dx = graph.resample_bwd(dy, attrs)

    dx.set_name("dx")
    dx.set_output(True)
    dx.set_data_type(hipdnn.DataType.FLOAT)

    return graph, dy, index, dx


def build_sdpa_graph():
    """Build an SDPA graph returning (graph, q, k, v, o, stats).

    Construction stays inside the function body so importing this module is
    safe when SDPA is compiled out of the bindings.
    """
    graph = create_float_graph()
    q = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
    q.set_name("q")
    k = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
    k.set_name("k")
    v = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
    v.set_name("v")

    outputs = graph.sdpa(q, k, v, hipdnn.SdpaAttributes().set_generate_stats(True))
    o, stats = _unpack(outputs, 2)
    o.set_name("o")
    stats.set_name("stats")
    o.set_output(True)

    return graph, q, k, v, o, stats


def build_sdpa_backward_graph():
    """Build an SDPA backward graph.

    Returns (graph, q, k, v, o, d_o, stats, dq, dk, dv). Construction stays
    inside the function body so importing this module is safe when SDPA is
    compiled out of the bindings.
    """
    graph = create_float_graph()
    q = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
    q.set_name("q")
    k = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
    k.set_name("k")
    v = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
    v.set_name("v")
    o = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
    o.set_name("o")
    d_o = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
    d_o.set_name("do")
    stats = hipdnn.Tensor.create([2, 8, 16, 1], hipdnn.DataType.FLOAT)
    stats.set_name("stats")

    outputs = graph.sdpa_backward(
        q, k, v, o, d_o, stats, hipdnn.SdpaBackwardAttributes()
    )
    dq, dk, dv = _unpack(outputs, 3)
    dq.set_name("dq")
    dk.set_name("dk")
    dv.set_name("dv")
    for output in (dq, dk, dv):
        output.set_output(True)

    return graph, q, k, v, o, d_o, stats, dq, dk, dv
