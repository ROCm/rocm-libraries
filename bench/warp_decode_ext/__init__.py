# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""JIT-loadable PyTorch extension exposing the CKTile warp-decode kernels.

Usage:
    import warp_decode_ext
    warp_decode_ext.warp_decode_gate_up_fp8(...)
    warp_decode_ext.warp_decode_gate_up_bf16(...)
    warp_decode_ext.warp_decode_down_reduce(...)

Builds against the CKTile headers under
``warp-decode-moe/projects/composablekernel/include`` with ``CK_TILE_USE_OCP_FP8``
enabled (required for OCP FP8 on gfx950). The first import triggers the build;
subsequent imports reuse the cached shared object.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent
_CK_INCLUDE = (_THIS_DIR.parent.parent / "projects" / "composablekernel" / "include").resolve()

if not _CK_INCLUDE.is_dir():
    raise RuntimeError(
        f"Could not locate CKTile include directory at {_CK_INCLUDE}. "
        "warp_decode_ext must live under warp-decode-moe/bench/."
    )

_SRC = _THIS_DIR / "warp_decode_ext.cu"

_EXTRA_CFLAGS = [
    "-O3",
    "-std=c++17",
    "-DCK_TILE_USE_OCP_FP8=1",
    "-Wno-unused-variable",
    "-Wno-unused-parameter",
]

_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "-std=c++17",
    "-DCK_TILE_USE_OCP_FP8=1",
    "-Wno-unused-variable",
    "-Wno-unused-parameter",
    "--offload-arch=gfx950",
    "-ffast-math",
]

_extra_include_paths = [str(_CK_INCLUDE)]


def _load_extension():
    build_dir = Path(os.environ.get(
        "WARP_DECODE_EXT_BUILD_DIR",
        str(_THIS_DIR / "build"),
    )).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name="warp_decode_ext",
        sources=[str(_SRC)],
        extra_include_paths=_extra_include_paths,
        extra_cflags=_EXTRA_CFLAGS,
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        build_directory=str(build_dir),
        verbose=bool(int(os.environ.get("WARP_DECODE_EXT_VERBOSE", "1"))),
        with_cuda=True,
    )


_ext = _load_extension()

warp_decode_gate_up_fp8 = _ext.warp_decode_gate_up_fp8
warp_decode_gate_up_bf16 = _ext.warp_decode_gate_up_bf16
warp_decode_down_reduce = _ext.warp_decode_down_reduce

__all__ = [
    "warp_decode_gate_up_fp8",
    "warp_decode_gate_up_bf16",
    "warp_decode_down_reduce",
]
