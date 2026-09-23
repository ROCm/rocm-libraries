# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convolution parameter parsing and hipDNN JSON graph construction."""

import dataclasses
import enum
import warnings
from typing import Any, Dict, List

from .parsing import CONV_FLAG_ALIASES, get_int_arg, normalize_args
from .strides import Layout, _input_strides, _weight_strides, conv_out_dim
from .tensors import _join_prefix, _make_tensor


class ConvDirection(enum.IntEnum):
    FORWARD = 1
    BACKWARD_DATA = 2
    BACKWARD_WEIGHTS = 4

    @property
    def label(self) -> str:
        return {
            ConvDirection.FORWARD: "fwd",
            ConvDirection.BACKWARD_DATA: "dgrad",
            ConvDirection.BACKWARD_WEIGHTS: "wgrad",
        }[self]

    @property
    def node_type(self) -> str:
        return {
            ConvDirection.FORWARD: "ConvolutionFwdAttributes",
            ConvDirection.BACKWARD_DATA: "ConvolutionBwdAttributes",
            ConvDirection.BACKWARD_WEIGHTS: "ConvolutionWrwAttributes",
        }[self]


class ConvMode(enum.Enum):
    CROSS_CORRELATION = "conv"
    CONVOLUTION = "trans"


CONV_IO_TYPE: Dict[str, str] = {
    "conv": "float",
    "convfp16": "half",
    "convbfp16": "bfloat16",
}


class PadMode(enum.Enum):
    DEFAULT = "default"
    SAME = "same"
    VALID = "valid"


def _same_padding(dim_in: int, kernel: int, dilation: int) -> int:
    effective_kernel = (kernel - 1) * dilation + 1
    return (effective_kernel - 1) // 2


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
    in_layout: Layout
    fil_layout: Layout
    out_layout: Layout
    conv_mode: ConvMode = ConvMode.CROSS_CORRELATION
    D: int = 0
    D_f: int = 0
    pad_d: int = 0
    stride_d: int = 1
    dil_d: int = 1

    @property
    def is_3d(self) -> bool:
        return self.spatial_dim == 3

    @property
    def direction(self) -> ConvDirection:
        return ConvDirection(self.F)

    @classmethod
    def from_args(cls, args: Dict[str, str]) -> "ConvParams":
        """Parse MIOpen convolution args into a ConvParams instance."""
        args = normalize_args(args, CONV_FLAG_ALIASES)
        spatial_dim = get_int_arg(args, "--spatial_dim", 2)
        is_3d = spatial_dim == 3
        C = get_int_arg(args, "-c", 3)
        K = get_int_arg(args, "-k", 32)
        groups = get_int_arg(args, "-g", 1)
        if C % groups != 0:
            raise ValueError(
                f"Invalid convolution: C={C} is not divisible by groups={groups}"
            )
        if K % groups != 0:
            raise ValueError(
                f"Invalid convolution: K={K} is not divisible by groups={groups}"
            )
        try:
            conv_mode = ConvMode(args.get("-m", "conv"))
        except ValueError:
            raise ValueError(
                f"Unknown conv mode {args.get('-m')!r}, "
                f"expected one of {[m.value for m in ConvMode]}"
            )

        try:
            pad_mode = PadMode(args.get("-z", "default"))
        except ValueError:
            raise ValueError(
                f"Unknown pad_mode {args.get('-z')!r}, "
                f"expected one of {[m.value for m in PadMode]}"
            )

        R = get_int_arg(args, "-y", 3)
        S = get_int_arg(args, "-x", 3)
        dil_h = get_int_arg(args, "-l", 1)
        dil_w = get_int_arg(args, "-j", 1)

        match pad_mode:
            case PadMode.SAME:
                pad_h = _same_padding(get_int_arg(args, "-H", 32), R, dil_h)
                pad_w = _same_padding(get_int_arg(args, "-W", 32), S, dil_w)
            case PadMode.VALID:
                pad_h = 0
                pad_w = 0
            case PadMode.DEFAULT:
                pad_h = get_int_arg(args, "-p", 0)
                pad_w = get_int_arg(args, "-q", 0)

        D_f = get_int_arg(args, "--fil_d", 3) if is_3d else 0
        dil_d = get_int_arg(args, "--dilation_d", 1) if is_3d else 1
        if is_3d:
            match pad_mode:
                case PadMode.SAME:
                    pad_d = _same_padding(get_int_arg(args, "--in_d", 32), D_f, dil_d)
                case PadMode.VALID:
                    pad_d = 0
                case PadMode.DEFAULT:
                    pad_d = get_int_arg(args, "--pad_d", 0)
        else:
            pad_d = 0

        if args.get("-b", "0") != "0":
            warnings.warn(
                "Conv bias (-b) is not supported in hipDNN conv nodes; "
                "use a separate pointwise ADD node",
                stacklevel=2,
            )
        if args.get("-Z", "0") != "0":
            warnings.warn(
                "Tensor vectorization (-Z) is not supported in hipDNN graphs",
                stacklevel=2,
            )
        if args.get("-L", "1") != "1":
            warnings.warn(
                "Tensor vector_length (-L) is not supported in hipDNN graphs",
                stacklevel=2,
            )
        if args.get("-r", "0") != "0":
            warnings.warn(
                "Non-zero pad_val (-r) is not supported in hipDNN; "
                "padding always uses zeros",
                stacklevel=2,
            )
        for flag, name in [
            ("-R", "wei_cast_type"),
            ("-T", "out_cast_type"),
            ("-U", "in_cast_type"),
        ]:
            if flag in args:
                warnings.warn(
                    f"Cast type {name} ({flag}) is not mapped to hipDNN "
                    "tensor data types; all tensors use the operation's io_type",
                    stacklevel=2,
                )
        for flag, name in [("-X", "width"), ("-Y", "height"), ("-%", "depth")]:
            if args.get(flag, "0") != "0":
                warnings.warn(
                    f"Transpose output padding for {name} ({flag}) is not "
                    "supported in hipDNN graphs",
                    stacklevel=2,
                )

        default_layout = Layout.NCDHW if is_3d else Layout.NCHW

        return cls(
            N=get_int_arg(args, "-n", 100),
            C=C,
            H=get_int_arg(args, "-H", 32),
            W=get_int_arg(args, "-W", 32),
            K=K,
            R=R,
            S=S,
            pad_h=pad_h,
            pad_w=pad_w,
            stride_h=get_int_arg(args, "-u", 1),
            stride_w=get_int_arg(args, "-v", 1),
            dil_h=dil_h,
            dil_w=dil_w,
            groups=groups,
            F=get_int_arg(args, "-F", 0),
            spatial_dim=spatial_dim,
            in_layout=Layout.parse(args.get("--in_layout"), default_layout),
            fil_layout=Layout.parse(args.get("--fil_layout"), default_layout),
            out_layout=Layout.parse(args.get("--out_layout"), default_layout),
            conv_mode=conv_mode,
            D=get_int_arg(args, "--in_d", 32) if is_3d else 0,
            D_f=D_f,
            pad_d=pad_d,
            stride_d=get_int_arg(args, "--conv_stride_d", 1) if is_3d else 1,
            dil_d=dil_d,
        )


def build_conv_json(p: ConvParams, io_type: str = "bfloat16") -> Dict[str, Any]:
    """Build a hipDNN JSON graph dict from a ConvParams instance."""
    Cg = p.C // p.groups  # channels per group for weight tensor

    # Compute output spatial dims
    H_out = conv_out_dim(p.H, p.pad_h, p.dil_h, p.R, p.stride_h)
    W_out = conv_out_dim(p.W, p.pad_w, p.dil_w, p.S, p.stride_w)

    # Build dims in canonical NCHW / NCDHW order
    if p.is_3d:
        D_out = conv_out_dim(p.D, p.pad_d, p.dil_d, p.D_f, p.stride_d)
        x_dims = [p.N, p.C, p.D, p.H, p.W]
        w_dims = [p.K, Cg, p.D_f, p.R, p.S]
        y_dims = [p.N, p.K, D_out, H_out, W_out]
        x_strides = _input_strides(p.in_layout, p.N, p.C, p.H, p.W, p.D)
        w_strides = _weight_strides(p.K, Cg, p.R, p.S, p.D_f, p.fil_layout)
        y_strides = _input_strides(p.out_layout, p.N, p.K, H_out, W_out, D_out)
        pre_pad = [p.pad_d, p.pad_h, p.pad_w]
        post_pad = [p.pad_d, p.pad_h, p.pad_w]
        stride_list = [p.stride_d, p.stride_h, p.stride_w]
        dil_list = [p.dil_d, p.dil_h, p.dil_w]
    else:
        x_dims = [p.N, p.C, p.H, p.W]
        w_dims = [p.K, Cg, p.R, p.S]
        y_dims = [p.N, p.K, H_out, W_out]
        x_strides = _input_strides(p.in_layout, p.N, p.C, p.H, p.W)
        w_strides = _weight_strides(p.K, Cg, p.R, p.S, layout=p.fil_layout)
        y_strides = _input_strides(p.out_layout, p.N, p.K, H_out, W_out)
        pre_pad = [p.pad_h, p.pad_w]
        post_pad = [p.pad_h, p.pad_w]
        stride_list = [p.stride_h, p.stride_w]
        dil_list = [p.dil_h, p.dil_w]

    # Wire up inputs/outputs differently per direction
    match p.direction:
        case ConvDirection.FORWARD:  # x, w → y
            tensors = [
                _make_tensor(0, "output_y", y_dims, y_strides, data_type=io_type),
                _make_tensor(1, "input_x", x_dims, x_strides, data_type=io_type),
                _make_tensor(2, "weight_w", w_dims, w_strides, data_type=io_type),
            ]
            node_inputs = {"x_tensor_uid": 1, "w_tensor_uid": 2}
            node_outputs = {"y_tensor_uid": 0}
        case ConvDirection.BACKWARD_DATA:  # dy, w → dx
            tensors = [
                _make_tensor(0, "output_dx", x_dims, x_strides, data_type=io_type),
                _make_tensor(1, "input_dy", y_dims, y_strides, data_type=io_type),
                _make_tensor(2, "weight_w", w_dims, w_strides, data_type=io_type),
            ]
            node_inputs = {"dy_tensor_uid": 1, "w_tensor_uid": 2}
            node_outputs = {"dx_tensor_uid": 0}
        case ConvDirection.BACKWARD_WEIGHTS:  # dy, x → dw
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
            "type": p.direction.node_type,
            "compute_data_type": "float",
            "inputs": node_inputs,
            "outputs": node_outputs,
            "parameters": {
                "conv_mode": p.conv_mode.name,
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
    if p.is_3d:
        return _join_prefix(
            prefix,
            f"conv_{p.direction.label}"
            f"_n{p.N}c{p.C}D{p.D}H{p.H}W{p.W}"
            f"_k{p.K}Df{p.D_f}R{p.R}S{p.S}"
            f"_pd{p.pad_d}p{p.pad_h}q{p.pad_w}"
            f"_sd{p.stride_d}u{p.stride_h}v{p.stride_w}"
            f"_g{p.groups}",
        )
    return _join_prefix(
        prefix,
        f"conv_{p.direction.label}"
        f"_n{p.N}c{p.C}H{p.H}W{p.W}"
        f"_k{p.K}R{p.R}S{p.S}"
        f"_p{p.pad_h}q{p.pad_w}"
        f"_u{p.stride_h}v{p.stride_w}"
        f"_l{p.dil_h}j{p.dil_w}"
        f"_g{p.groups}",
    )
