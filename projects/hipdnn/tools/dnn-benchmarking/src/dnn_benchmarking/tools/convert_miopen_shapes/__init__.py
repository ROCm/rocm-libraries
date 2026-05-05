# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Convert MIOpen driver shape files to hipDNN JSON graph files.

Usage:
    python -m dnn_benchmarking.tools.convert_miopen_shapes graphs/shapes.txt
    dnn-convert-shapes shapes.txt --outdir graphs/generic_convolutions/
"""

from .bnorm import BnormParams, build_bnorm_json
from .cli import convert_line, main, parse_line
from .conv import CONV_IO_TYPE, ConvMode, ConvParams, build_conv_json
from .parsing import (
    BNORM_FLAG_ALIASES,
    CONV_FLAG_ALIASES,
    get_int_arg,
    is_flag,
    normalize_args,
    parse_args,
)
from .strides import (
    Layout,
    conv_out_dim,
    nchw_strides,
    ncdhw_strides,
    ndhwc_strides,
    nhwc_strides,
)

__all__ = [
    "BNORM_FLAG_ALIASES",
    "CONV_FLAG_ALIASES",
    "BnormParams",
    "ConvParams",
    "build_bnorm_json",
    "build_conv_json",
    "ConvMode",
    "CONV_IO_TYPE",
    "Layout",
    "conv_out_dim",
    "convert_line",
    "get_int_arg",
    "is_flag",
    "main",
    "nchw_strides",
    "ncdhw_strides",
    "ndhwc_strides",
    "nhwc_strides",
    "normalize_args",
    "parse_args",
    "parse_line",
]
