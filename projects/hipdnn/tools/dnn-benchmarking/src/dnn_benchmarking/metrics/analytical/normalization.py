# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for normalization-family ops.

All of these are output-element-count driven and use a small fixed
ops/elem multiplier:

* BN inference: 4 (subtract mean, mul inv_var, mul scale, add bias)
* BN training fwd / bwd: 8 (above + mean & variance reductions)
* LayerNorm / RMSNorm fwd: 8 (mean + variance + normalisation)
* Softmax fwd: 4 (max, exp, sum, divide)

RMSNorm omits the mean step (~6 ops/elem in theory) but the
8-ops/elem simplification keeps the estimator within roofline noise
and avoids a separate handler.
"""

from typing import Any, Dict, Optional

from ._common import output_elements


def batchnorm_inference_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BatchNorm inference: 4 ops/elem."""
    elems = output_elements(node, tensors_by_uid)
    return 4 * elems if elems is not None else None


def batchnorm_training_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BatchNorm fwd training: 8 ops/elem.

    Output count comes from ``y_tensor_uid`` which is the activation
    output of the BN forward.
    """
    elems = output_elements(node, tensors_by_uid)
    return 8 * elems if elems is not None else None


def batchnorm_backward_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """BatchNorm bwd: 8 ops/elem on the activation-shaped gradient.

    BN backward emits ``dx_tensor_uid`` (the gradient w.r.t. the input
    activation), plus ``dscale`` / ``dbias`` (parameter-sized — small
    enough to ignore). The dominant cost is over ``dx`` elements.
    """
    elems = output_elements(node, tensors_by_uid, output_key="dx_tensor_uid")
    return 8 * elems if elems is not None else None


def layernorm_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """LayerNorm / RMSNorm: 8 ops/elem (mean + variance + normalisation)."""
    elems = output_elements(node, tensors_by_uid)
    return 8 * elems if elems is not None else None


def softmax_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """SoftMax fwd: 4 ops/elem (max, exp, sum, divide)."""
    elems = output_elements(node, tensors_by_uid)
    return 4 * elems if elems is not None else None
