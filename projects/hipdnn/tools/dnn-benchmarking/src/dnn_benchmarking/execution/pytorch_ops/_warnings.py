# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Static warnings describing manual (non-built-in) PyTorch reference paths."""

from typing import Any, Dict, List

import torch.nn.functional as F

from ._common import _node_param, _node_uid
from .handlers.norm import _rmsnorm_graph_can_use_builtin
from .handlers.reduction import _reduction_mode_name
from .handlers.resample import _resample_has_asymmetric_padding, _resample_mode_name


def get_reference_warnings(graph_json: Dict[str, Any]) -> List[str]:
    """Describe manual/non-built-in portions of the PyTorch reference graph.

    The timed PyTorch reference row is useful as a baseline only when users can
    tell whether it is measuring a public PyTorch primitive or local reference
    glue.  This helper is intentionally static: it only inspects graph metadata
    and reports paths whose handler is not solely a built-in PyTorch operator.
    """

    warnings: List[str] = []
    for node in graph_json.get("nodes", []):
        op_type = str(node.get("type", ""))
        name = str(node.get("name") or op_type)

        if op_type == "LayernormAttributes":
            if (
                _node_uid(node, "mean_tensor_uid", ("outputs",), required=False)
                is not None
                or _node_uid(
                    node, "inv_variance_tensor_uid", ("outputs",), required=False
                )
                is not None
            ):
                warnings.append(
                    f"{name}: LayernormAttributes uses torch.nn.functional.layer_norm "
                    "for y but computes requested mean/inv-variance outputs manually; "
                    "PyTorch reference timing is not solely built-in PyTorch operator time."
                )

        elif op_type == "RMSNormAttributes":
            reasons: List[str] = []
            if not hasattr(F, "rms_norm"):
                reasons.append("torch.nn.functional.rms_norm is unavailable")
            if (
                _node_uid(node, "bias_tensor_uid", ("inputs",), required=False)
                is not None
            ):
                reasons.append("optional bias is applied manually")
            if (
                _node_uid(node, "inv_rms_tensor_uid", ("outputs",), required=False)
                is not None
            ):
                reasons.append("requested inv_rms output is computed manually")
            if not _rmsnorm_graph_can_use_builtin(node, graph_json):
                reasons.append("per-channel layout uses a manual RMSNorm formula")
            if reasons:
                warnings.append(
                    f"{name}: RMSNormAttributes includes manual reference work "
                    f"({'; '.join(dict.fromkeys(reasons))}); PyTorch reference timing "
                    "is not solely built-in PyTorch operator time."
                )

        elif op_type == "RMSNormBackwardAttributes":
            warnings.append(
                f"{name}: RMSNormBackwardAttributes uses a manual RMSNorm backward "
                "formula because PyTorch has no public operator matching hipDNN's "
                "saved-inv_rms backward node; PyTorch reference timing is not solely "
                "built-in PyTorch operator time."
            )

        elif op_type == "SdpaBackwardAttributes":
            warnings.append(
                f"{name}: SdpaBackwardAttributes uses a manual flash-attention "
                "backward formula that consumes hipDNN's saved stats (log-sum-exp); "
                "torch.nn.functional.scaled_dot_product_attention autograd cannot "
                "consume external stats, so PyTorch reference timing is not solely "
                "built-in PyTorch operator time."
            )

        elif op_type == "ReductionAttributes":
            if (
                _reduction_mode_name(_node_param(node, "mode", "NOT_SET"))
                == "MUL_NO_ZEROS"
            ):
                warnings.append(
                    f"{name}: ReductionAttributes mode MUL_NO_ZEROS uses a manual "
                    "masked product; PyTorch reference timing is not solely built-in "
                    "PyTorch operator time."
                )

        elif op_type == "ResampleFwdAttributes":
            mode = _resample_mode_name(_node_param(node, "resample_mode", "NOT_SET"))
            if mode == "AVGPOOL_EXCLUDE_PADDING" and _resample_has_asymmetric_padding(
                node, graph_json
            ):
                warnings.append(
                    f"{name}: ResampleFwdAttributes AVGPOOL_EXCLUDE_PADDING with "
                    "asymmetric padding uses manual valid-count correction around "
                    "torch.nn.functional.avg_pool; PyTorch reference timing is not "
                    "solely built-in PyTorch operator time."
                )

    return warnings
