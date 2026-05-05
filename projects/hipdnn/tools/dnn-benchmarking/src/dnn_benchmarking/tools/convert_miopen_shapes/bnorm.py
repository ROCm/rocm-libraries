# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Batchnorm type resolution and hipDNN JSON graph construction."""

import dataclasses
import enum
import warnings
from typing import Any, Dict, List

from .parsing import BNORM_FLAG_ALIASES, get_int_arg, normalize_args
from .strides import _input_strides, nchw_strides, ncdhw_strides
from .tensors import _join_prefix, _make_scalar_tensor, _make_tensor

# ---------------------------------------------------------------------------
# Batchnorm type resolution
# ---------------------------------------------------------------------------

class BnormDirection(enum.Enum):
    FORWARD_TRAINING = "fwd"
    FORWARD_INFERENCE = "inference"
    BACKWARD_TRAINING = "backward"


_BNORM_IO_TYPE: Dict[str, str] = {
    "bnorm": "float",
    "bnormfp16": "half",
    "bnormbfp16": "bfloat16",
}


# ---------------------------------------------------------------------------
# Batchnorm parsed parameters
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BnormParams:
    """Parsed batchnorm parameters extracted from MIOpen driver args."""

    N: int
    C: int
    H: int
    W: int
    D: int
    layout: str
    mode: int
    forw: int
    back: int

    @classmethod
    def from_args(cls, args: Dict[str, str]) -> "BnormParams":
        args = normalize_args(args, BNORM_FLAG_ALIASES)
        D = get_int_arg(args, "-D", 0)
        default_layout = "NCDHW" if D > 0 else "NCHW"

        activ_mode = args.get("--activ_mode", args.get("-f", "0"))
        if activ_mode != "0":
            warnings.warn(
                "Fused activation (--activ_mode / -f) is not supported in "
                "hipDNN batchnorm nodes; use a separate pointwise node",
                stacklevel=2,
            )
        alpha = args.get("-A", "1.0")
        beta = args.get("-B", "0.")
        if alpha not in ("1", "1.0") or beta not in ("0", "0.", "0.0"):
            warnings.warn(
                f"Non-default alpha={alpha}/beta={beta} scaling is not "
                "supported in hipDNN batchnorm graphs",
                stacklevel=2,
            )
        if args.get("-s", "0") != "0":
            warnings.warn(
                "Save mode (-s/--save) for running mean/variance is not "
                "supported; hipDNN batchnorm graphs do not populate "
                "next_running_mean/variance tensors",
                stacklevel=2,
            )
        if args.get("-r", "0") != "0":
            warnings.warn(
                "Run mode (-r/--run) for running mean/variance is not "
                "supported; hipDNN batchnorm graphs do not use "
                "prev_running_mean/variance tensors",
                stacklevel=2,
            )
        if args.get("-I", "0") != "0":
            warnings.warn(
                "Inverse variance flag (-I/--inverse_variance) is ignored; "
                "hipDNN batchnorm always uses inv_variance semantics",
                stacklevel=2,
            )

        return cls(
            N=get_int_arg(args, "-n", 32),
            C=get_int_arg(args, "-c", 3),
            H=get_int_arg(args, "-H", 32),
            W=get_int_arg(args, "-W", 32),
            D=D,
            layout=args.get("-L", default_layout),
            mode=get_int_arg(args, "-m", 0),
            forw=get_int_arg(args, "--forw", 1),
            back=get_int_arg(args, "--back", 0),
        )

    @property
    def is_3d(self) -> bool:
        return self.D > 0

    @property
    def is_spatial(self) -> bool:
        return self.mode == 1

    @property
    def direction(self) -> BnormDirection:
        if self.back == 1:
            return BnormDirection.BACKWARD_TRAINING
        if self.forw == 2:
            return BnormDirection.FORWARD_INFERENCE
        return BnormDirection.FORWARD_TRAINING

    def scale_dims_and_strides(self) -> tuple[List[int], List[int]]:
        """Return (dims, strides) for scale/bias/mean/variance tensors.

        mode=0 (per-activation): dims match input spatial shape with N=1.
        mode=1 (spatial): dims are [1, C, 1, ...].
        """
        if self.is_3d:
            if self.is_spatial:
                return [1, self.C, 1, 1, 1], [self.C, 1, 1, 1, 1]
            return (
                [1, self.C, self.D, self.H, self.W],
                ncdhw_strides(1, self.C, self.D, self.H, self.W),
            )
        if self.is_spatial:
            return [1, self.C, 1, 1], [self.C, 1, 1, 1]
        return (
            [1, self.C, self.H, self.W],
            nchw_strides(1, self.C, self.H, self.W),
        )


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
    p = BnormParams.from_args(args)
    io_type = _BNORM_IO_TYPE[operation]

    if p.is_3d:
        x_dims = [p.N, p.C, p.D, p.H, p.W]
        x_strides = _input_strides(p.layout, p.N, p.C, p.H, p.W, p.D)
    else:
        x_dims = [p.N, p.C, p.H, p.W]
        x_strides = _input_strides(p.layout, p.N, p.C, p.H, p.W)

    scale_dims, scale_strides = p.scale_dims_and_strides()

    def _stat(uid: int, name: str) -> Dict[str, Any]:
        return _make_tensor(uid, name, scale_dims, scale_strides, data_type="float")

    direction = p.direction

    if direction is BnormDirection.FORWARD_INFERENCE:
        # Inference: x, mean, inv_variance, scale, bias → y
        node_type = "BatchnormInferenceAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
            _stat(2, "mean"),
            _stat(3, "inv_variance"),
            _stat(4, "scale"),
            _stat(5, "bias"),
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
    elif direction is BnormDirection.FORWARD_TRAINING:
        # Forward training: x, scale, bias, epsilon → y, mean, inv_variance
        # Optional: prev_running_mean/variance + momentum → next_running_mean/variance
        # peer_stats_tensor_uid is required by the schema (empty list = no peers).
        node_type = "BatchnormAttributes"
        tensors = [
            _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
            _stat(2, "scale"),
            _stat(3, "bias"),
            _make_scalar_tensor(4, "epsilon", 1e-5, data_type="float"),
            _make_tensor(5, "output_y", x_dims, x_strides, data_type=io_type),
            _stat(6, "saved_mean"),
            _stat(7, "saved_inv_variance"),
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
            _stat(3, "mean"),
            _stat(4, "inv_variance"),
            _stat(5, "scale"),
            _make_tensor(6, "output_dx", x_dims, x_strides, data_type=io_type),
            _stat(7, "output_dscale"),
            _stat(8, "output_dbias"),
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
    p = BnormParams.from_args(args)

    direction = p.direction.value

    if p.is_3d:
        return _join_prefix(
            prefix, f"bnorm_{direction}_n{p.N}c{p.C}D{p.D}H{p.H}W{p.W}"
        )
    return _join_prefix(prefix, f"bnorm_{direction}_n{p.N}c{p.C}H{p.H}W{p.W}")
