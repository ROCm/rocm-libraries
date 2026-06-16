# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Matmul reference handler."""

from typing import Any, Dict

import torch

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


@register_handler("MatmulAttributes")
def handle_matmul(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle MatmulAttributes (matrix multiplication)."""
    a_uid = _required_input_uid(node, "a_tensor_uid")
    b_uid = _required_input_uid(node, "b_tensor_uid")
    c_uid = _required_output_uid(node, "c_tensor_uid")

    c = torch.matmul(_tensor(tensors, a_uid, node), _tensor(tensors, b_uid, node))
    _store_tensor(tensors, c_uid, c)
