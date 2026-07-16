# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Matmul reference handler."""

from typing import Any, Dict

import torch

from .._common import *  # noqa: F401,F403
from .._registry import CompiledOp, register_handler


@register_handler("MatmulAttributes")
def compile_matmul(
    node: Dict[str, Any],
    graph_json: Dict[str, Any],
) -> CompiledOp:
    """Plan MatmulAttributes (matrix multiplication)."""
    a_uid = _required_input_uid(node, "a_tensor_uid")
    b_uid = _required_input_uid(node, "b_tensor_uid")
    c_uid = _required_output_uid(node, "c_tensor_uid")

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        c = torch.matmul(_tensor(tensors, a_uid, node), _tensor(tensors, b_uid, node))
        _store_tensor(tensors, c_uid, c)

    return run
