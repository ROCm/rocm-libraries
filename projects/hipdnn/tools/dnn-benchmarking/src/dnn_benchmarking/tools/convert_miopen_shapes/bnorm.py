# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Batchnorm type resolution and hipDNN JSON graph construction."""

from typing import Any, Dict, Optional

from .parsing import BNORM_FLAG_ALIASES, _int, normalize_args
from .strides import _input_strides
from .tensors import _join_prefix, _make_scalar_tensor, _make_tensor

# ---------------------------------------------------------------------------
# Batchnorm type resolution
# ---------------------------------------------------------------------------

# MIOpen BN driver template: BatchNormDriver<TInput, Tref, TAcc, TScaleBias, TOut>
# The operation name encodes the IO type; stat/affine tensors follow MIOpen internals:
#
#   bnorm          → TInput=float,    TAcc=float,  TScaleBias=float
#   bnormfp16      → TInput=float16,  TAcc=float,  TScaleBias=float
#   bnormbfp16     → TInput=bfloat16, TAcc=float,  TScaleBias=float
#   bnormfp16fp32  → TInput=float16,  TAcc=float,  TScaleBias=float16  (TOut=float, rare)
#   bnormbfp16fp32 → TInput=bfloat16, TAcc=float,  TScaleBias=bfloat16 (TOut=float, rare)
#
# For hipDNN graphs, TAcc drives the stat-tensor dtype and TScaleBias drives scale/bias.
# dscale/dbias are always TAcc (float) regardless of TScaleBias.

_BNORM_IO_TYPE: Dict[str, str] = {
    "bnorm": "float",
    "bnormfp16": "half",
    "bnormbfp16": "bfloat16",
    "bnormfp16fp32": "half",
    "bnormbfp16fp32": "bfloat16",
    # short aliases without data-type suffix default to float
    "bn": "float",
    "bnfp16": "half",
    "bnbfp16": "bfloat16",
}

_BNORM_SCALE_BIAS_TYPE: Dict[str, str] = {
    "bnorm": "float",
    "bnormfp16": "float",
    "bnormbfp16": "float",
    "bnormfp16fp32": "half",
    "bnormbfp16fp32": "bfloat16",
    "bn": "float",
    "bnfp16": "float",
    "bnbfp16": "float",
}


def _bnorm_io_type(operation: str) -> str:
    return _BNORM_IO_TYPE.get(operation, "bfloat16")


def _bnorm_scale_bias_type(operation: str) -> str:
    return _BNORM_SCALE_BIAS_TYPE.get(operation, "float")


# ---------------------------------------------------------------------------
# Batchnorm conversion
# ---------------------------------------------------------------------------


def build_bnorm_json(operation: str, args: Dict[str, str]) -> Dict[str, Any]:
    """Build a hipDNN JSON graph dict from parsed bnorm* driver args.

    MIOpen --forw / --back semantics (from bn_driver.hpp):
      --forw 1 (default) → forward training
      --forw 2           → forward inference
      --back 1           → backward (requires --forw 0)
    """
    args = normalize_args(args, BNORM_FLAG_ALIASES)
    N = _int(args, "-n", 1)
    C = _int(args, "-c", 1)
    H = _int(args, "-H", 1)
    W = _int(args, "-W", 1)
    layout = args.get("--layout", args.get("-L", "NCHW"))
    forw = _int(args, "--forw", 1)
    back = _int(args, "--back", 0)

    io_type = _bnorm_io_type(operation)
    # TAcc is always float for all supported driver variants
    stat_type = "float"
    # TScaleBias depends on the driver variant
    scale_bias_type = _bnorm_scale_bias_type(operation)

    is_3d = "-D" in args
    D: Optional[int] = _int(args, "-D", 1) if is_3d else None

    if is_3d and D is not None:
        x_dims = [N, C, D, H, W]
        x_strides = _input_strides(layout, N, C, H, W, D)
        scale_dims = [1, C, 1, 1, 1]
        scale_strides = [C, 1, 1, 1, 1]
    else:
        x_dims = [N, C, H, W]
        x_strides = _input_strides(layout, N, C, H, W)
        scale_dims = [1, C, 1, 1]
        scale_strides = [C, 1, 1, 1]

    # Determine direction.  When both forw and back are 0, MIOpen defaults to
    # forw=1 (training).  back=1 takes priority over forw for backward.
    if back == 1:
        direction = "backward"
    elif forw == 2:
        direction = "inference"
    else:
        # forw == 1 (or default 0 which MIOpen remaps to 1)
        direction = "fwd_training"

    if direction == "inference":
        # Inference: x, mean, inv_variance, scale, bias → y
        node_type = "BatchnormInferenceAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
            _make_tensor(2, "mean", scale_dims, scale_strides, data_type=stat_type),
            _make_tensor(
                3, "inv_variance", scale_dims, scale_strides, data_type=stat_type
            ),
            _make_tensor(
                4, "scale", scale_dims, scale_strides, data_type=scale_bias_type
            ),
            _make_tensor(
                5, "bias", scale_dims, scale_strides, data_type=scale_bias_type
            ),
            _make_tensor(6, "output_y", x_dims, x_strides, data_type=io_type),
        ]
        nodes = [
            {
                "name": "batchnorm_inference_node",
                "type": node_type,
                "compute_data_type": "float",
                "inputs": {
                    "x_tensor_uid": 1,
                    "mean_tensor_uid": 2,
                    "inv_variance_tensor_uid": 3,
                    "scale_tensor_uid": 4,
                    "bias_tensor_uid": 5,
                },
                "outputs": {"y_tensor_uid": 6},
            }
        ]
    elif direction == "fwd_training":
        # Forward training: x, scale, bias, epsilon → y, mean, inv_variance
        # Optional: prev_running_mean/variance + momentum → next_running_mean/variance
        # peer_stats_tensor_uid is required by the schema (empty list = no peers).
        node_type = "BatchnormAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
            _make_tensor(
                2, "scale", scale_dims, scale_strides, data_type=scale_bias_type
            ),
            _make_tensor(
                3, "bias", scale_dims, scale_strides, data_type=scale_bias_type
            ),
            _make_scalar_tensor(4, "epsilon", 1e-5, data_type="float"),
            _make_tensor(5, "output_y", x_dims, x_strides, data_type=io_type),
            _make_tensor(
                6, "saved_mean", scale_dims, scale_strides, data_type=stat_type
            ),
            _make_tensor(
                7, "saved_inv_variance", scale_dims, scale_strides, data_type=stat_type
            ),
        ]
        nodes = [
            {
                "name": "batchnorm_fwd_node",
                "type": node_type,
                "compute_data_type": "float",
                "inputs": {
                    "x_tensor_uid": 1,
                    "scale_tensor_uid": 2,
                    "bias_tensor_uid": 3,
                    "epsilon_tensor_uid": 4,
                    "peer_stats_tensor_uid": [],
                    "prev_running_mean_tensor_uid": None,
                    "prev_running_variance_tensor_uid": None,
                    "momentum_tensor_uid": None,
                },
                "outputs": {
                    "y_tensor_uid": 5,
                    "mean_tensor_uid": 6,
                    "inv_variance_tensor_uid": 7,
                    "next_running_mean_tensor_uid": None,
                    "next_running_variance_tensor_uid": None,
                },
            }
        ]
    else:
        # Backward: dy, x, mean, inv_variance, scale → dx, dscale, dbias
        # mean and inv_variance are optional (null if not available).
        # peer_stats_tensor_uid is required by the schema (empty list = no peers).
        # scale/bias type matches TScaleBias; dscale/dbias are TAcc (always float)
        node_type = "BatchnormBackwardAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
            _make_tensor(2, "input_dy", x_dims, x_strides, data_type=io_type),
            _make_tensor(3, "mean", scale_dims, scale_strides, data_type=stat_type),
            _make_tensor(
                4, "inv_variance", scale_dims, scale_strides, data_type=stat_type
            ),
            _make_tensor(
                5, "scale", scale_dims, scale_strides, data_type=scale_bias_type
            ),
            _make_tensor(6, "output_dx", x_dims, x_strides, data_type=io_type),
            # dscale and dbias accumulate in TAcc (float) regardless of TScaleBias
            _make_tensor(
                7, "output_dscale", scale_dims, scale_strides, data_type=stat_type
            ),
            _make_tensor(
                8, "output_dbias", scale_dims, scale_strides, data_type=stat_type
            ),
        ]
        nodes = [
            {
                "name": "batchnorm_backward_node",
                "type": node_type,
                "compute_data_type": "float",
                "inputs": {
                    "dy_tensor_uid": 2,
                    "x_tensor_uid": 1,
                    "mean_tensor_uid": 3,
                    "inv_variance_tensor_uid": 4,
                    "scale_tensor_uid": 5,
                    "peer_stats_tensor_uid": [],
                },
                "outputs": {
                    "dx_tensor_uid": 6,
                    "dscale_tensor_uid": 7,
                    "dbias_tensor_uid": 8,
                },
            }
        ]

    return {
        "compute_data_type": "float",
        "io_data_type": io_type,
        "intermediate_data_type": "float",
        "tensors": tensors,
        "nodes": nodes,
    }


def _bnorm_filename(prefix: str, operation: str, args: Dict[str, str]) -> str:
    args = normalize_args(args, BNORM_FLAG_ALIASES)
    N = _int(args, "-n", 1)
    C = _int(args, "-c", 1)
    H = _int(args, "-H", 1)
    W = _int(args, "-W", 1)
    forw = _int(args, "--forw", 1)
    back = _int(args, "--back", 0)

    if back == 1:
        direction = "backward"
    elif forw == 2:
        direction = "inference"
    else:
        direction = "fwd"

    is_3d = "-D" in args
    if is_3d:
        D = _int(args, "-D", 1)
        return _join_prefix(prefix, f"bnorm_{direction}_n{N}c{C}D{D}H{H}W{W}")
    return _join_prefix(prefix, f"bnorm_{direction}_n{N}c{C}H{H}W{W}")
