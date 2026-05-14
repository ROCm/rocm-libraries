# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Analytical FLOPs and I/O byte computation from graph JSON.

Every recognised op contributes a real arithmetic FLOP count, and the
caller always also receives ``analytical_io_bytes`` and
``derived_gbytes_per_s``. Reporting both signals lets the consumer
decide which is the dominant constraint for each op type — a low
TFLOPs/s number for a memory-bound kernel is informative when paired
with a high GB/s number, in the same way NVIDIA Nsight Compute exposes
Compute Throughput and Memory Throughput as independent percentages of
peak. We deliberately do *not* mirror MIOpen's ``bn_driver.hpp``
``flopCnt = 0`` choice (which then mislabels a bandwidth metric as
"GFLOPs"); the precedent we follow is Composable Kernel, whose
example/profiling code reports honest arithmetic FLOPs alongside GB/s
for the same kernel.

Per-op formulas (FMA = 2 FLOPs throughout):

* Conv2D fwd: ``2 * N * C * R * S * K * H_out * W_out / group_count``
  (matches ``miopen/driver/conv_driver.hpp:1706-1710``).
* GEMM: ``2 * M * N * K``.
* Pointwise (add, mul, relu, …): ``num_output_elements`` (1 op/elem).
* BatchNorm inference: ``4 * num_output_elements`` (subtract mean,
  multiply by inv_var, multiply by scale, add bias).
* BatchNorm fwd/bwd training: ``8 * num_output_elements`` (mean +
  variance reductions plus the inference math).
* LayerNorm fwd: ``8 * num_output_elements`` (mean + variance +
  normalisation).
* RMSNorm fwd: ``8 * num_output_elements`` — RMSNorm omits the mean
  step (~6 ops/elem in theory) but the simplification keeps the
  estimator within roofline-noise of the truth and avoids a separate
  handler.
* Softmax fwd: ``4 * num_output_elements`` (max, exp, sum, divide).
* Reduction (sum/mean/etc.): ``num_input_elements`` (1 op/elem).
* Rng: ``num_output_elements`` (one op per generated value;
  conservative — most PRNGs do more).

When a graph contains a node type this module does not recognise, the
``partial`` flag in :func:`compute_flops` is set so callers can label
the value as incomplete.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..graph.tensor_info import TensorInfo


def _tensor_dim_product(tensor: Dict[str, Any]) -> int:
    dims = tensor.get("dims") or []
    if not dims:
        return 0
    n = 1
    for d in dims:
        n *= int(d)
    return n


def _tensor_lookup(graph_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(t["uid"]): t for t in graph_json.get("tensors", []) if "uid" in t}


def _conv_flops_impl(
    x: Optional[Dict[str, Any]],
    w: Optional[Dict[str, Any]],
    y: Optional[Dict[str, Any]],
    params: Dict[str, Any],
) -> Optional[int]:
    """Compute conv FLOPs given resolved input / weight / output tensors.

    The three conv directions (fwd / bwd-data / bwd-filter) all do the
    same total scalar work; they only differ in which tensor is being
    produced. Each direction-specific handler resolves the right
    UID-keyed tensor as ``x`` (the activation-shaped input),
    ``w`` (the weight), ``y`` (the activation-shaped output) and calls
    this implementation.

    MIOpen formula: ``2 * N * C_in * R * S * K * H_out * W_out / group``
    (matches ``conv_driver.hpp:1750-1751``).
    """
    if not x or not w or not y:
        return None
    x_dims = x.get("dims") or []
    w_dims = w.get("dims") or []
    y_dims = y.get("dims") or []
    if len(x_dims) < 4 or len(w_dims) < 4 or len(y_dims) < 4:
        return None
    spatial_w = w_dims[2:]
    spatial_y = y_dims[2:]
    if not spatial_w or not spatial_y:
        return None

    # NCHW / NCDHW: dim 0 = N, dim 1 = C; for weight K = dim 0, C/g = dim 1.
    n = int(x_dims[0])
    c_in = int(x_dims[1])
    k = int(w_dims[0])

    weight_spatial = 1
    for d in spatial_w:
        weight_spatial *= int(d)
    output_spatial = 1
    for d in spatial_y:
        output_spatial *= int(d)

    group_count = int(params.get("group_count", 1)) or 1

    return 2 * n * c_in * weight_spatial * k * output_spatial // group_count


# TODO: lift the string-keyed UID lookups in the per-direction handlers
# below into a structural pass over the parsed graph once a Graph/Node
# abstraction is available. Today every handler reaches into the raw
# JSON and probes hipDNN-specific input/output names by string, which
# couples us to the exact key strings emitted by hipDNN's frontend.
#
# Cannot use ``or`` chains for UID lookups: hipDNN tensor UIDs start at
# 0 and ``0 or fallback`` evaluates to fallback, masking the real UID.


def _conv_fwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ConvolutionFwdAttributes: ``y = conv(x, w)``."""
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    x_uid = inputs.get("x_tensor_uid")
    w_uid = inputs.get("w_tensor_uid")
    y_uid = outputs.get("y_tensor_uid")
    if x_uid is None or w_uid is None or y_uid is None:
        return None
    return _conv_flops_impl(
        x=tensors_by_uid.get(int(x_uid)),
        w=tensors_by_uid.get(int(w_uid)),
        y=tensors_by_uid.get(int(y_uid)),
        params=node.get("parameters", {}) or {},
    )


def _conv_dgrad_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ConvolutionBwdAttributes (dgrad): ``dx = conv_transposed(dy, w)``.

    The output ``dx`` has the activation-input shape ``[N, C, H_in, W_in]``
    and ``dy`` has the activation-output shape ``[N, K, H_out, W_out]``.
    Map them onto the fwd-shaped impl: dx -> x, dy -> y.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    dy_uid = inputs.get("dy_tensor_uid")
    w_uid = inputs.get("w_tensor_uid")
    dx_uid = outputs.get("dx_tensor_uid")
    if dy_uid is None or w_uid is None or dx_uid is None:
        return None
    return _conv_flops_impl(
        x=tensors_by_uid.get(int(dx_uid)),
        w=tensors_by_uid.get(int(w_uid)),
        y=tensors_by_uid.get(int(dy_uid)),
        params=node.get("parameters", {}) or {},
    )


def _conv_wgrad_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """ConvolutionWrwAttributes (wgrad): ``dw = conv(x, dy)``.

    Output ``dw`` has the weight shape ``[K, C/G, R, S]`` and ``dy``
    has the activation-output shape. Map onto the fwd-shaped impl:
    x -> x, dw -> w, dy -> y.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    x_uid = inputs.get("x_tensor_uid")
    dy_uid = inputs.get("dy_tensor_uid")
    dw_uid = outputs.get("dw_tensor_uid")
    if x_uid is None or dy_uid is None or dw_uid is None:
        return None
    return _conv_flops_impl(
        x=tensors_by_uid.get(int(x_uid)),
        w=tensors_by_uid.get(int(dw_uid)),
        y=tensors_by_uid.get(int(dy_uid)),
        params=node.get("parameters", {}) or {},
    )


def _matmul_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for MatmulAttributes: ``2 * batch * M * N * K``.

    Supports batched matmul by multiplying all leading dims of the
    output tensor.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    a_uid = inputs.get("a_tensor_uid")
    b_uid = inputs.get("b_tensor_uid")
    c_uid = outputs.get("c_tensor_uid")
    if a_uid is None or b_uid is None or c_uid is None:
        return None
    a = tensors_by_uid.get(int(a_uid))
    b = tensors_by_uid.get(int(b_uid))
    c = tensors_by_uid.get(int(c_uid))
    if not a or not b or not c:
        return None

    a_dims = a.get("dims") or []
    b_dims = b.get("dims") or []
    c_dims = c.get("dims") or []
    if len(a_dims) < 2 or len(b_dims) < 2 or len(c_dims) < 2:
        return None

    m = int(c_dims[-2])
    n = int(c_dims[-1])
    k = int(a_dims[-1])

    batch = 1
    for d in c_dims[:-2]:
        batch *= int(d)

    return 2 * batch * m * n * k


def _pointwise_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for PointwiseAttributes — counted as one FLOP per output element.

    Element-wise operations (relu, add, mul, sub, div, abs, neg, exp,
    log, tanh, sigmoid, sqrt) all do O(num_elements) work. We do not
    distinguish unary vs binary because the dominant cost is memory
    traffic anyway; the FLOPs number is small relative to other ops in
    a fused graph.
    """
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get("out_0_tensor_uid")
    if out_uid is None:
        return None
    out = tensors_by_uid.get(int(out_uid))
    if not out:
        return None
    return _tensor_dim_product(out)


def _output_elements(
    node: Dict[str, Any],
    tensors_by_uid: Dict[int, Dict[str, Any]],
    output_key: str = "y_tensor_uid",
) -> Optional[int]:
    """Resolve the output tensor and return its element count, or None."""
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get(output_key)
    if out_uid is None:
        return None
    tensor = tensors_by_uid.get(int(out_uid))
    if not tensor:
        return None
    return _tensor_dim_product(tensor)


def _batchnorm_inference_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BatchNorm inference: 4 ops/elem (subtract mean, mul inv_var, mul scale, add bias)."""
    elems = _output_elements(node, tensors_by_uid)
    return 4 * elems if elems is not None else None


def _batchnorm_training_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BatchNorm fwd training: 8 ops/elem (reductions + inference math).

    Output count comes from ``y_tensor_uid`` which is the activation
    output of the BN forward.
    """
    elems = _output_elements(node, tensors_by_uid)
    return 8 * elems if elems is not None else None


def _batchnorm_backward_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BatchNorm bwd: 8 ops/elem on the activation-shaped gradient.

    BN backward emits ``dx_tensor_uid`` (the gradient w.r.t. the input
    activation), plus ``dscale`` / ``dbias`` (parameter-sized — small
    enough to ignore). The dominant cost is over ``dx`` elements.
    """
    elems = _output_elements(node, tensors_by_uid, output_key="dx_tensor_uid")
    return 8 * elems if elems is not None else None


def _layernorm_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """LayerNorm / RMSNorm: 8 ops/elem (mean + variance + normalisation)."""
    elems = _output_elements(node, tensors_by_uid)
    return 8 * elems if elems is not None else None


def _softmax_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """SoftMax fwd: 4 ops/elem (max, exp, sum, divide)."""
    elems = _output_elements(node, tensors_by_uid)
    return 4 * elems if elems is not None else None


def _reduction_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Reduction: 1 op/elem of the *input* tensor (the output is a scalar/row)."""
    inputs = node.get("inputs", {}) or {}
    in_uid = inputs.get("x_tensor_uid") or inputs.get("in_0_tensor_uid")
    if in_uid is None:
        return None
    tensor = tensors_by_uid.get(int(in_uid))
    if not tensor:
        return None
    return _tensor_dim_product(tensor)


def _rng_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Rng: 1 op/elem of the output (conservative — real PRNGs do more)."""
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get("out_0_tensor_uid")
    if out_uid is None:
        out_uid = outputs.get("y_tensor_uid")
    if out_uid is None:
        return None
    tensor = tensors_by_uid.get(int(out_uid))
    if not tensor:
        return None
    return _tensor_dim_product(tensor)


# Dispatch table: node "type" -> handler returning int FLOPs (or None
# when tensor data is incomplete). Unrecognised types flip the
# ``partial`` flag in compute_flops.
_FLOP_HANDLERS = {
    # Convolution: hipDNN's actual node names are ConvolutionFwdAttributes,
    # ConvolutionBwdAttributes (dgrad), and ConvolutionWrwAttributes
    # (wgrad). The *DataAttributes / *FilterAttributes spellings are
    # kept as aliases for older graph snapshots.
    "ConvolutionFwdAttributes": _conv_fwd_flops,
    "ConvolutionBwdAttributes": _conv_dgrad_flops,
    "ConvolutionBwdDataAttributes": _conv_dgrad_flops,
    "ConvolutionWrwAttributes": _conv_wgrad_flops,
    "ConvolutionBwdFilterAttributes": _conv_wgrad_flops,
    "MatmulAttributes": _matmul_flops,
    "PointwiseAttributes": _pointwise_flops,
    # BatchNorm: hipDNN's BatchnormAttributes covers fwd training (with
    # next_running_* outputs) and BatchnormBackwardAttributes covers
    # bwd. BatchnormFwdAttributes / BatchnormBwdAttributes are kept as
    # aliases for older graph snapshots.
    "BatchnormInferenceAttributes": _batchnorm_inference_flops,
    "BatchnormAttributes": _batchnorm_training_flops,
    "BatchnormFwdAttributes": _batchnorm_training_flops,
    "BatchnormBackwardAttributes": _batchnorm_backward_flops,
    "BatchnormBwdAttributes": _batchnorm_backward_flops,
    "LayerNormAttributes": _layernorm_flops,
    "RmsNormAttributes": _layernorm_flops,
    "SoftmaxAttributes": _softmax_flops,
    "ReductionAttributes": _reduction_flops,
    "RngAttributes": _rng_flops,
}


def compute_flops(graph_json: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """Sum analytical FLOPs across a graph's nodes.

    Args:
        graph_json: Parsed hipDNN graph dictionary.

    Returns:
        ``(total_flops, partial)``. ``total_flops`` is ``None`` only
        when the graph has no nodes at all; otherwise it is the sum of
        the per-handler counts (``0`` is possible for a graph whose
        only nodes had unknown types). ``partial`` is True when at
        least one node was unrecognised or had missing tensor data — the
        returned sum then reflects only the recognised nodes.
    """
    nodes = graph_json.get("nodes") or []
    if not nodes:
        return None, False

    tensors_by_uid = _tensor_lookup(graph_json)

    total = 0
    partial = False
    for node in nodes:
        node_type = node.get("type", "")
        handler = _FLOP_HANDLERS.get(node_type)
        if handler is None:
            partial = True
            continue
        flops = handler(node, tensors_by_uid)
        if flops is None:
            partial = True
            continue
        total += flops

    return total, partial


def compute_io_bytes(tensor_infos: Iterable[TensorInfo]) -> int:
    """Sum bytes of all non-virtual tensors (inputs + outputs + weights).

    Virtual tensors are intermediate buffers that hipDNN may allocate
    inside a fused kernel and never materialise to global memory, so
    they are excluded. Uses :attr:`TensorInfo.size_bytes` which already
    accounts for non-contiguous strides.
    """
    total = 0
    for ti in tensor_infos:
        if ti.is_virtual:
            continue
        total += ti.size_bytes
    return total


def derive_throughputs(
    flops: Optional[int],
    io_bytes: Optional[int],
    kernel_mean_ms: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Derive TFLOPs/s and GB/s from totals + mean kernel time.

    Args:
        flops: Total analytical FLOPs (or ``None``).
        io_bytes: Total non-virtual tensor bytes (or ``None``).
        kernel_mean_ms: Mean GPU kernel time in ms (or ``None``).

    Returns:
        ``(tflops_per_s, gbytes_per_s)`` — either component is ``None``
        when its inputs are missing or zero.
    """
    if not kernel_mean_ms or kernel_mean_ms <= 0:
        return None, None
    seconds = kernel_mean_ms / 1000.0
    tflops = (flops / seconds / 1e12) if flops else None
    gbytes = (io_bytes / seconds / 1e9) if io_bytes else None
    return tflops, gbytes


# ---------------------------------------------------------------------------
# Convenience wrappers used by tests that want a single-call surface.
# ---------------------------------------------------------------------------


def list_unsupported_node_types(graph_json: Dict[str, Any]) -> List[str]:
    """Return node type strings present in the graph that have no handler.

    Useful for diagnostic output that explains *why* ``partial`` is True.
    """
    seen: List[str] = []
    seen_set: set = set()
    for node in graph_json.get("nodes") or []:
        nt = node.get("type", "")
        if not nt:
            continue
        if nt in _FLOP_HANDLERS:
            continue
        if nt not in seen_set:
            seen_set.add(nt)
            seen.append(nt)
    return seen
