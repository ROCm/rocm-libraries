# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tensor and utility helpers for hipDNN JSON graph construction."""

from typing import Any, Dict, List


def _make_tensor(
    uid: int,
    name: str,
    dims: List[int],
    strides: List[int],
    data_type: str = "bfloat16",
    virtual: bool = False,
) -> Dict[str, Any]:
    return {
        "uid": uid,
        "name": name,
        "dims": dims,
        "strides": strides,
        "data_type": data_type,
        "virtual": virtual,
    }


def _make_scalar_tensor(
    uid: int,
    name: str,
    value: float,
    data_type: str = "float",
) -> Dict[str, Any]:
    """Build a pass-by-value scalar tensor (e.g. epsilon for batchnorm)."""
    value_type_map = {
        "float": "Float32Value",
        "half": "Float16Value",
        "bfloat16": "BFloat16Value",
        "double": "Float64Value",
    }
    return {
        "uid": uid,
        "name": name,
        "dims": [1],
        "strides": [1],
        "data_type": data_type,
        "virtual": False,
        "value_type": value_type_map.get(data_type, "Float32Value"),
        "value": value,
    }


def _join_prefix(prefix: str, rest: str) -> str:
    """Join prefix and rest with '_', omitting the separator when prefix is empty."""
    return f"{prefix}_{rest}" if prefix else rest
