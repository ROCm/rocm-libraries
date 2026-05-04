# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convolution parameter parsing and hipDNN JSON graph construction."""

import dataclasses
from typing import Any, Dict, List, Optional

from .parsing import CONV_FLAG_ALIASES, _int, normalize_args
from .strides import _input_strides, _weight_strides, conv_out_dim
from .tensors import _join_prefix, _make_tensor


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


def _conv_direction_label(F: int) -> str:
    return {1: "fwd", 2: "dgrad", 4: "wgrad"}.get(F, f"F{F}")


def _conv_node_type(F: int) -> str:
    return {
        1: "ConvolutionFwdAttributes",
        2: "ConvolutionBwdAttributes",
        4: "ConvolutionWrwAttributes",
    }.get(F, "ConvolutionFwdAttributes")


_CONV_IO_TYPE: Dict[str, str] = {
    "conv": "float",
    "convfp16": "half",
    "convbfp16": "bfloat16",
    "convfp32": "float",
}


def conv_io_type(operation: str) -> str:
    return _CONV_IO_TYPE.get(operation, "bfloat16")


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
