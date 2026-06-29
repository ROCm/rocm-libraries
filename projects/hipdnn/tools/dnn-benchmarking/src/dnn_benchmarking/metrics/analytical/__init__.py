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

* Conv2D fwd / dgrad / wgrad:
  ``2 * N * C_in * R * S * K * H_out * W_out / group_count``
  — see :mod:`.conv`. Matches MIOpen's
  ``conv_driver.hpp:1706-1710``.
* GEMM: ``2 * batch * M * N * K`` — see :mod:`.matmul`. Standard
  textbook formula (e.g. NVIDIA's perf model docs).
* Pointwise (relu, add, mul, …): ``num_output_elements`` (1 op/elem)
  — see :mod:`.elementwise`.
* Rng: ``num_output_elements`` (conservative; most PRNGs do more
  per draw) — see :mod:`.elementwise`.
* BatchNorm inference: ``4 * num_output_elements`` (subtract mean,
  multiply by inv_var, multiply by scale, add bias). Training fwd /
  bwd: ``8 * num_output_elements`` (above plus mean / variance
  reductions). LayerNorm / RMSNorm fwd: ``8 * num_output_elements``.
  SoftMax fwd: ``4 * num_output_elements`` (max, exp, sum, divide).
  See :mod:`.normalization`. Multipliers follow Composable Kernel's
  norm benchmarks and PyTorch's profiler conventions.
* Reduction (sum/mean/etc.): ``num_input_elements`` (1 op/elem on
  the input — output is typically scalar/row). See :mod:`.reduction`.

Per-op FLOP / IO handlers live in this directory split by op family:
``conv.py``, ``matmul.py``, ``elementwise.py``, ``normalization.py``,
``reduction.py``. The dispatch table is the single source of truth
for which node types we recognise.

When a graph contains a node type this module does not recognise, the
``partial`` flag in :func:`compute_flops` is set so callers can label
the value as incomplete, and a one-shot warning surfaces the unknown
type via :mod:`.._diagnostic.warn_once`.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...graph.tensor_info import TensorInfo
from .._diagnostic import warn_once
from ._common import tensor_lookup
from .conv import conv_dgrad_flops, conv_fwd_flops, conv_wgrad_flops
from .elementwise import pointwise_flops, rng_flops
from .matmul import matmul_flops
from .normalization import (
    batchnorm_backward_flops,
    batchnorm_inference_flops,
    batchnorm_training_flops,
    layernorm_flops,
    softmax_flops,
)
from .reduction import reduction_flops

# Dispatch table: node "type" -> handler returning int FLOPs (or None
# when tensor data is incomplete). Unrecognised types flip the
# ``partial`` flag in compute_flops.
_FLOP_HANDLERS = {
    # Convolution: hipDNN's actual node names are ConvolutionFwdAttributes,
    # ConvolutionBwdAttributes (dgrad), and ConvolutionWrwAttributes
    # (wgrad). The *DataAttributes / *FilterAttributes spellings are
    # kept as aliases for older graph snapshots.
    "ConvolutionFwdAttributes": conv_fwd_flops,
    "ConvolutionBwdAttributes": conv_dgrad_flops,
    "ConvolutionBwdDataAttributes": conv_dgrad_flops,
    "ConvolutionWrwAttributes": conv_wgrad_flops,
    "ConvolutionBwdFilterAttributes": conv_wgrad_flops,
    "MatmulAttributes": matmul_flops,
    "PointwiseAttributes": pointwise_flops,
    # BatchNorm: hipDNN's BatchnormAttributes covers fwd training (with
    # next_running_* outputs) and BatchnormBackwardAttributes covers
    # bwd. BatchnormFwdAttributes / BatchnormBwdAttributes are kept as
    # aliases for older graph snapshots.
    "BatchnormInferenceAttributes": batchnorm_inference_flops,
    "BatchnormAttributes": batchnorm_training_flops,
    "BatchnormFwdAttributes": batchnorm_training_flops,
    "BatchnormBackwardAttributes": batchnorm_backward_flops,
    "BatchnormBwdAttributes": batchnorm_backward_flops,
    "LayernormAttributes": layernorm_flops,
    "RMSNormAttributes": layernorm_flops,
    "SoftmaxAttributes": softmax_flops,
    "ReductionAttributes": reduction_flops,
    "RngAttributes": rng_flops,
}


def compute_flops(graph_json: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """Sum analytical FLOPs across a graph's nodes.

    Args:
        graph_json: Parsed hipDNN graph dictionary.

    Returns:
        ``(total_flops, partial)``. ``total_flops`` is ``None`` when the
        graph has no nodes at all, or when no node could be modelled
        analytically (every node was unrecognised or lacked tensor
        data) — a count of ``0`` in that case would be indistinguishable
        from a genuine zero-FLOP graph, so ``None`` ("unknown") is
        returned instead. ``partial`` is True when at least one node was
        unrecognised or had missing tensor data; the returned sum then
        reflects only the recognised nodes.

    Unrecognised node types also surface a one-shot warning via
    :func:`warn_once` so the user notices during a run; the structured
    ``partial`` flag stays as the machine-readable signal.
    """
    nodes = graph_json.get("nodes") or []
    if not nodes:
        return None, False

    tensors_by_uid = tensor_lookup(graph_json)

    total = 0
    partial = False
    modelled_any = False
    for node in nodes:
        node_type = node.get("type", "")
        handler = _FLOP_HANDLERS.get(node_type)
        if handler is None:
            partial = True
            warn_once(
                "analytical",
                f"unrecognised node type {node_type!r}; FLOPs marked partial",
            )
            continue
        flops = handler(node, tensors_by_uid)
        if flops is None:
            partial = True
            continue
        total += flops
        modelled_any = True

    if not modelled_any:
        # Nothing could be modelled analytically: report unknown (None)
        # rather than a misleading 0 that reads as a real FLOP count.
        return None, partial

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

    Note: ``kernel_mean_ms`` is the plain arithmetic mean over post-warmup
    iterations (no trimming, no outlier rejection — see
    :class:`reporting.statistics.BenchmarkStats`). A single noisy
    iteration can therefore skew the derived throughput; for tighter
    signal use ``gpu_kernel_stats.min_ms`` or ``p95_ms``.
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


__all__ = [
    "compute_flops",
    "compute_io_bytes",
    "derive_throughputs",
    "list_unsupported_node_types",
]
