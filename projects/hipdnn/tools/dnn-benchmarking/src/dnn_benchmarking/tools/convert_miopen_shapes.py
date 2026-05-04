# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convert MIOpen driver shape files to hipDNN JSON graph files.

Usage:
    python convert_miopen_shapes.py graphs/shapes.txt graphs/shapes_3D.txt
    python convert_miopen_shapes.py shapes.txt --outdir graphs/generic_convolutions/
"""

import argparse
import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Stride helpers
# ---------------------------------------------------------------------------


def nchw_strides(N: int, C: int, H: int, W: int) -> List[int]:
    return [C * H * W, H * W, W, 1]


def nhwc_strides(N: int, C: int, H: int, W: int) -> List[int]:
    # Dims stay as [N, C, H, W]; strides reflect NHWC memory order
    return [H * W * C, 1, W * C, C]


def ncdhw_strides(N: int, C: int, D: int, H: int, W: int) -> List[int]:
    return [C * D * H * W, D * H * W, H * W, W, 1]


def ndhwc_strides(N: int, C: int, D: int, H: int, W: int) -> List[int]:
    # Dims stay as [N, C, D, H, W]; strides reflect NDHWC memory order
    return [D * H * W * C, 1, H * W * C, W * C, C]


def _input_strides(
    layout: str, N: int, C: int, H: int, W: int, D: Optional[int] = None
) -> List[int]:
    """Return strides for an input tensor given its memory layout."""
    if D is not None:
        if layout == "NDHWC":
            return ndhwc_strides(N, C, D, H, W)
        return ncdhw_strides(N, C, D, H, W)
    if layout == "NHWC":
        return nhwc_strides(N, C, H, W)
    return nchw_strides(N, C, H, W)


def _weight_strides(
    K: int, Cg: int, R: int, S: int, D: Optional[int] = None, layout: str = "NCHW"
) -> List[int]:
    """Weight strides for dims [K, Cg, R, S] (or [K, Cg, D, R, S] for 3D).

    NCHW/NCDHW → row-major KCRS / KCDRS (Cg innermost after spatial).
    NHWC/NDHWC → KRSC / KDRSC (Cg is the fastest-moving dimension).
    """
    if D is not None:
        if layout in ("NHWC", "NDHWC"):
            # KDRSC: stride[K]=D*R*S*Cg, stride[Cg]=1, stride[D]=R*S*Cg,
            #        stride[R]=S*Cg, stride[S]=Cg
            return [D * R * S * Cg, 1, R * S * Cg, S * Cg, Cg]
        return [Cg * D * R * S, D * R * S, R * S, S, 1]
    if layout in ("NHWC",):
        # KRSC: stride[K]=R*S*Cg, stride[Cg]=1, stride[R]=S*Cg, stride[S]=Cg
        return [R * S * Cg, 1, S * Cg, Cg]
    return [Cg * R * S, R * S, S, 1]


# ---------------------------------------------------------------------------
# Output dimension formula
# ---------------------------------------------------------------------------


def conv_out_dim(dim_in: int, pad: int, dilation: int, kernel: int, stride: int) -> int:
    return math.floor((dim_in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)


# ---------------------------------------------------------------------------
# MIOpen argument parsers
# ---------------------------------------------------------------------------


def is_flag(token: str) -> bool:
    """Return True if token looks like a CLI flag (e.g. -n, --layout).

    Negative numbers like -1 or -0.5 are values, not flags.
    """
    if not token.startswith("-"):
        return False
    rest = token[1:]
    if rest and (rest[0].isdigit() or rest[0] == "."):
        return False
    return True


def parse_args(tokens: List[str]) -> Dict[str, str]:
    """Parse a flat list of flag/value tokens into a dict."""
    result: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if is_flag(tok):
            # Check if next token is a value (not a flag)
            if i + 1 < len(tokens) and not is_flag(tokens[i + 1]):
                result[tok] = tokens[i + 1]
                i += 2
            else:
                result[tok] = "1"  # boolean flag
                i += 1
        else:
            i += 1
    return result


def _int(args: Dict[str, str], key: str, default: int = 0) -> int:
    return int(args.get(key, default))


# ---------------------------------------------------------------------------
# Flag alias normalization
# ---------------------------------------------------------------------------
# MIOpenDriver accepts both short and long flag forms.  The code below maps
# the *alternative* form to the *canonical* form already used in this file so
# that inputs written with either style are handled identically.


def normalize_args(args: Dict[str, str], aliases: Dict[str, str]) -> Dict[str, str]:
    """Merge flag aliases so both short and long forms are recognized.

    The canonical key (value in *aliases*) takes precedence when both the
    alternative and canonical forms appear in *args*.
    """
    normalized = dict(args)
    for alt, canonical in aliases.items():
        if alt in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alt)
        elif alt in normalized:
            del normalized[alt]
    return normalized


# Convolution: maps every alternative flag to the canonical key used by
# ConvParams.from_args().
CONV_FLAG_ALIASES: Dict[str, str] = {
    # Long → short (2D parameters)
    "--batchsize": "-n",
    "--in_channels": "-c",
    "--in_h": "-H",
    "--in_w": "-W",
    "--out_channels": "-k",
    "--fil_h": "-y",
    "--fil_w": "-x",
    "--pad_h": "-p",
    "--pad_w": "-q",
    "--conv_stride_h": "-u",
    "--conv_stride_w": "-v",
    "--dilation_h": "-l",
    "--dilation_w": "-j",
    "--group_count": "-g",
    "--forw": "-F",
    # Short → long (3D / layout parameters)
    "-_": "--spatial_dim",
    "-!": "--in_d",
    "-@": "--fil_d",
    "-$": "--pad_d",
    "-#": "--conv_stride_d",
    "-^": "--dilation_d",
    "-I": "--in_layout",
    "-f": "--fil_layout",
    "-O": "--out_layout",
}

# Batchnorm: maps every alternative flag to the canonical key used by
# build_bnorm_json() and _bnorm_filename().
BNORM_FLAG_ALIASES: Dict[str, str] = {
    # Long → short
    "--batchsize": "-n",
    "--in_channels": "-c",
    "--in_h": "-H",
    "--in_w": "-W",
    "--in_d": "-D",
    # Short → long
    "-F": "--forw",
    "-b": "--back",
}


# ---------------------------------------------------------------------------
# Build tensor objects
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Convolution parameter container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ConvParams:
    """Parsed convolution parameters extracted from MIOpen driver args."""

    N: int
    C: int
    H: int
    W: int
    K: int
    R: int
    S: int
    pad_h: int
    pad_w: int
    stride_h: int
    stride_w: int
    dil_h: int
    dil_w: int
    groups: int
    F: int
    spatial_dim: int
    in_layout: str
    fil_layout: str
    out_layout: str
    D: Optional[int] = None
    D_f: Optional[int] = None
    pad_d: int = 0
    stride_d: int = 1
    dil_d: int = 1

    @classmethod
    def from_args(cls, args: Dict[str, str]) -> "ConvParams":
        """Parse MIOpen convolution args into a ConvParams instance."""
        args = normalize_args(args, CONV_FLAG_ALIASES)
        spatial_dim = _int(args, "--spatial_dim", 2)
        is_3d = spatial_dim == 3
        D: Optional[int] = None
        D_f: Optional[int] = None
        pad_d = 0
        stride_d = 1
        dil_d = 1
        if is_3d:
            D = _int(args, "--in_d", 1)
            D_f = _int(args, "--fil_d", 1)
            pad_d = _int(args, "--pad_d", 0)
            stride_d = _int(args, "--conv_stride_d", 1)
            dil_d = _int(args, "--dilation_d", 1)
        C = _int(args, "-c", 1)
        K = _int(args, "-k", 1)
        groups = _int(args, "-g", 1)
        if C % groups != 0:
            raise ValueError(
                f"Invalid convolution: C={C} is not divisible by groups={groups}"
            )
        if K % groups != 0:
            raise ValueError(
                f"Invalid convolution: K={K} is not divisible by groups={groups}"
            )
        return cls(
            N=_int(args, "-n", 1),
            C=C,
            H=_int(args, "-H", 1),
            W=_int(args, "-W", 1),
            K=K,
            R=_int(args, "-y", 1),
            S=_int(args, "-x", 1),
            pad_h=_int(args, "-p", 0),
            pad_w=_int(args, "-q", 0),
            stride_h=_int(args, "-u", 1),
            stride_w=_int(args, "-v", 1),
            dil_h=_int(args, "-l", 1),
            dil_w=_int(args, "-j", 1),
            groups=groups,
            F=_int(args, "-F", 1),
            spatial_dim=spatial_dim,
            in_layout=args.get("--in_layout", "NCHW"),
            fil_layout=args.get("--fil_layout", "NCHW"),
            out_layout=args.get("--out_layout", "NCHW"),
            D=D,
            D_f=D_f,
            pad_d=pad_d,
            stride_d=stride_d,
            dil_d=dil_d,
        )


# ---------------------------------------------------------------------------
# Convolution conversion
# ---------------------------------------------------------------------------


def _conv_direction_label(F: int) -> str:
    return {1: "fwd", 2: "dgrad", 4: "wgrad"}.get(F, f"F{F}")


def _conv_node_type(F: int) -> str:
    return {
        1: "ConvolutionFwdAttributes",
        2: "ConvolutionBwdAttributes",
        4: "ConvolutionWrwAttributes",
    }.get(F, "ConvolutionFwdAttributes")


def build_conv_json(p: ConvParams, io_type: str = "bfloat16") -> Dict[str, Any]:
    """Build a hipDNN JSON graph dict from a ConvParams instance."""
    is_3d = p.spatial_dim == 3
    Cg = p.C // p.groups  # channels per group for weight tensor

    # Compute output spatial dims
    H_out = conv_out_dim(p.H, p.pad_h, p.dil_h, p.R, p.stride_h)
    W_out = conv_out_dim(p.W, p.pad_w, p.dil_w, p.S, p.stride_w)
    D_out: Optional[int] = None
    if is_3d and p.D is not None and p.D_f is not None:
        D_out = conv_out_dim(p.D, p.pad_d, p.dil_d, p.D_f, p.stride_d)

    # Build dims in canonical NCHW / NCDHW order
    if is_3d and p.D is not None and p.D_f is not None and D_out is not None:
        x_dims = [p.N, p.C, p.D, p.H, p.W]
        w_dims = [p.K, Cg, p.D_f, p.R, p.S]
        y_dims = [p.N, p.K, D_out, H_out, W_out]
    else:
        x_dims = [p.N, p.C, p.H, p.W]
        w_dims = [p.K, Cg, p.R, p.S]
        y_dims = [p.N, p.K, H_out, W_out]

    x_strides = _input_strides(p.in_layout, p.N, p.C, p.H, p.W, p.D)
    w_strides = _weight_strides(p.K, Cg, p.R, p.S, p.D_f, p.fil_layout)
    y_strides = _input_strides(p.out_layout, p.N, p.K, H_out, W_out, D_out)

    node_type = _conv_node_type(p.F)

    if is_3d and p.D_f is not None:
        pre_pad = [p.pad_d, p.pad_h, p.pad_w]
        post_pad = [p.pad_d, p.pad_h, p.pad_w]
        stride_list = [p.stride_d, p.stride_h, p.stride_w]
        dil_list = [p.dil_d, p.dil_h, p.dil_w]
    else:
        pre_pad = [p.pad_h, p.pad_w]
        post_pad = [p.pad_h, p.pad_w]
        stride_list = [p.stride_h, p.stride_w]
        dil_list = [p.dil_h, p.dil_w]

    # Wire up inputs/outputs differently per direction
    if p.F == 1:  # forward: x, w → y
        tensors = [
            _make_tensor(0, "output_y", y_dims, y_strides, data_type=io_type),
            _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
            _make_tensor(2, "weight_w", w_dims, w_strides, data_type=io_type),
        ]
        node_inputs = {"x_tensor_uid": 1, "w_tensor_uid": 2}
        node_outputs = {"y_tensor_uid": 0}
    elif p.F == 2:  # dgrad: dy, w → dx
        tensors = [
            _make_tensor(0, "output_dx", x_dims, x_strides, data_type=io_type),
            _make_tensor(1, "input_dy", y_dims, y_strides, data_type=io_type),
            _make_tensor(2, "weight_w", w_dims, w_strides, data_type=io_type),
        ]
        node_inputs = {"dy_tensor_uid": 1, "w_tensor_uid": 2}
        node_outputs = {"dx_tensor_uid": 0}
    else:  # wgrad (F==4): dy, x → dw
        tensors = [
            _make_tensor(0, "output_dw", w_dims, w_strides, data_type=io_type),
            _make_tensor(1, "input_dy", y_dims, y_strides, data_type=io_type),
            _make_tensor(2, "input_x", x_dims, x_strides, data_type=io_type),
        ]
        node_inputs = {"dy_tensor_uid": 1, "x_tensor_uid": 2}
        node_outputs = {"dw_tensor_uid": 0}

    nodes = [
        {
            "name": "conv_node",
            "type": node_type,
            "compute_data_type": "float",
            "inputs": node_inputs,
            "outputs": node_outputs,
            "parameters": {
                "conv_mode": "CROSS_CORRELATION",
                "pre_padding": pre_pad,
                "post_padding": post_pad,
                "stride": stride_list,
                "dilation": dil_list,
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


def _join_prefix(prefix: str, rest: str) -> str:
    """Join prefix and rest with '_', omitting the separator when prefix is empty."""
    return f"{prefix}_{rest}" if prefix else rest


def _conv_filename(prefix: str, p: ConvParams) -> str:
    direction = _conv_direction_label(p.F)

    if p.spatial_dim == 3 and p.D is not None and p.D_f is not None:
        name = _join_prefix(
            prefix,
            f"conv_{direction}"
            f"_n{p.N}c{p.C}D{p.D}H{p.H}W{p.W}"
            f"_k{p.K}Df{p.D_f}R{p.R}S{p.S}"
            f"_pd{p.pad_d}p{p.pad_h}q{p.pad_w}"
            f"_sd{p.stride_d}u{p.stride_h}v{p.stride_w}"
            f"_g{p.groups}",
        )
    else:
        name = _join_prefix(
            prefix,
            f"conv_{direction}"
            f"_n{p.N}c{p.C}H{p.H}W{p.W}"
            f"_k{p.K}R{p.R}S{p.S}"
            f"_p{p.pad_h}q{p.pad_w}"
            f"_u{p.stride_h}v{p.stride_w}"
            f"_l{p.dil_h}j{p.dil_w}"
            f"_g{p.groups}",
        )
    return name


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


# ---------------------------------------------------------------------------
# Supported operation sets
# ---------------------------------------------------------------------------

_BNORM_OPERATIONS = {
    "bnorm",
    "bnormfp16",
    "bnormbfp16",
    "bnormfp16fp32",
    "bnormbfp16fp32",
    "bn",
    "bnfp16",
    "bnbfp16",
}

_CONV_OPERATIONS = {"convbfp16", "conv", "convfp16", "convfp32"}

_ALL_OPERATIONS = _CONV_OPERATIONS | _BNORM_OPERATIONS


# ---------------------------------------------------------------------------
# Line parser dispatcher
# ---------------------------------------------------------------------------


def parse_line(line: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse one shape file line. Returns (operation, args_dict) or None.

    Accepts lines with or without an executable prefix::

        ./bin/MIOpenDriver convbfp16 -n 16 ...   # executable + operation
        convbfp16 -n 16 ...                       # bare operation
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # Strip leading repeat count (e.g. "     5  ./bin/MIOpenDriver ...")
    m = re.match(r"^\s*\d+\s+", line)
    if m:
        line = line[m.end() :]

    parts = line.split()
    if not parts:
        return None

    if parts[0] in _ALL_OPERATIONS:
        operation = parts[0]
        flag_tokens = parts[1:]
    elif len(parts) < 2:
        return None
    else:
        operation = parts[1]
        flag_tokens = parts[2:]

    args = parse_args(flag_tokens)
    return operation, args


_CONV_IO_TYPE: Dict[str, str] = {
    "conv": "float",
    "convfp16": "half",
    "convbfp16": "bfloat16",
    "convfp32": "float",
}


def conv_io_type(operation: str) -> str:
    return _CONV_IO_TYPE.get(operation, "bfloat16")


def convert_line(
    operation: str, args: Dict[str, str], prefix: str
) -> Tuple[str, Dict[str, Any]]:
    """Convert parsed MIOpen args to (filename_stem, json_dict)."""
    if operation in _CONV_OPERATIONS:
        p = ConvParams.from_args(args)
        graph = build_conv_json(p, io_type=conv_io_type(operation))
        name_stem = _conv_filename(prefix, p)
    elif operation in _BNORM_OPERATIONS:
        graph = build_bnorm_json(operation, args)
        name_stem = _bnorm_filename(prefix, operation, args)
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    graph["name"] = name_stem
    return name_stem, graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MIOpen driver shape files or inline args to hipDNN JSON graph files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="SHAPES_FILE",
        help="One or more MIOpen shape .txt files to convert.",
    )
    parser.add_argument(
        "-A",
        "--args",
        metavar="MIOPEN_ARGS",
        default=None,
        help=(
            "Inline MIOpen driver arguments (everything after the executable path), "
            "e.g. 'convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 ...'. "
            "Use with --output to write to a specific file."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSON file path (used with --args; ignored for file inputs).",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        default=None,
        help=(
            "Output directory for JSON files. "
            "Defaults to the same directory as each input file."
        ),
    )
    return parser


def _process_inline_args(args_str: str, output: Optional[str]) -> int:
    """Convert a single inline MIOpen driver argument string to a JSON file."""
    parts = args_str.split()
    if not parts:
        print("ERROR: --args is empty.", file=sys.stderr)
        return 1

    operation = parts[0]
    flag_tokens = parts[1:]
    parsed_args = parse_args(flag_tokens)

    try:
        name_stem, graph = convert_line(operation, parsed_args, "")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(f"{name_stem}.json")

    out_path.write_text(json.dumps(graph, indent=4) + "\n")
    print(f"Written: {out_path}")
    return 0


def _process_file(input_path: Path, outdir: Path) -> Tuple[int, int, int]:
    """Process one shape file. Returns (written, skipped, warnings)."""
    prefix = input_path.stem.lower().replace("-", "_")
    lines = input_path.read_text().splitlines()

    written = 0
    skipped = 0
    warnings = 0
    seen_stems: Dict[str, int] = {}

    for lineno, raw_line in enumerate(lines, start=1):
        parsed = parse_line(raw_line)
        if parsed is None:
            skipped += 1
            continue

        operation, args = parsed
        try:
            name_stem, graph = convert_line(operation, args, prefix)
        except Exception as exc:
            print(f"  WARNING line {lineno}: {exc}", file=sys.stderr)
            warnings += 1
            continue

        # Deduplicate: if same stem appears more than once, append a counter
        count = seen_stems.get(name_stem, 0)
        seen_stems[name_stem] = count + 1
        if count > 0:
            unique_stem = f"{name_stem}_{count}"
        else:
            unique_stem = name_stem

        out_path = outdir / f"{unique_stem}.json"
        out_path.write_text(json.dumps(graph, indent=4) + "\n")
        written += 1

    return written, skipped, warnings


def main() -> int:
    parser = _build_arg_parser()
    ns = parser.parse_args()

    if ns.args:
        if ns.inputs:
            print(
                "ERROR: cannot combine --args with positional SHAPES_FILE arguments.",
                file=sys.stderr,
            )
            return 1
        return _process_inline_args(ns.args, ns.output)

    if not ns.inputs:
        parser.print_help(sys.stderr)
        return 1

    total_written = 0
    total_skipped = 0
    total_warnings = 0

    for input_str in ns.inputs:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            return 1

        outdir = Path(ns.outdir) if ns.outdir else input_path.parent
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {input_path} → {outdir}/")
        written, skipped, warns = _process_file(input_path, outdir)
        print(f"  Written: {written}  Skipped: {skipped}  Warnings: {warns}")
        total_written += written
        total_skipped += skipped
        total_warnings += warns

    print(f"\nTotal: {total_written} files written, {total_warnings} warnings.")
    return 0 if total_warnings == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
