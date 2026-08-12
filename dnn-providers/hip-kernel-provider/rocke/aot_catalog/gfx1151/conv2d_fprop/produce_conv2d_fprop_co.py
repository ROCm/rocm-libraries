# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# One-shot, build-time producer: emit gfx1151 2-D forward-convolution (implicit-
# GEMM) .co (HSACO) kernels for the AOT catalog engine. Runtime never touches
# rocKE -- this is run by hand by a kernel author to drop .co files beside their
# family.json.
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_conv2d_fprop_co.py <out_dir>
#
# Runtime-generic model (parity with rmsnorm2d_dynamic / layernorm2d_dynamic):
# every kernel here is FULLY dynamic -- ALL convolution geometry (N, C, K, Hi,
# Wi, R, S, stride, pad, dilation) is read from runtime i32 args, and only the
# tile/perf config is baked. So ONE .co per tile config serves ANY 2-D forward
# conv shape; partial tiles at the M/N/K boundaries are masked (transform-DAG
# validity + hardware buffer-OOB clamp + m<M / n<N_gemm store predication), not
# mis-addressed. The catalog selector keys on dtype / groups / vec-alignment
# only (see family.json), and the tuner picks the winning tile config per shape.
#
# gfx1151 note: wave_size MUST be 32 (WMMA). The wave32 cross-lane reduction
# gotcha does not apply to conv -- the WMMA accumulation path does not go through
# the Welford/shuffle helpers -- but the WMMA atom itself is a wave32 primitive.

import os
import sys

from rocke.helpers.compile import compile_kernel
from rocke.instances.common.conv_implicit_gemm import ImplicitGemmConvSpec
from rocke.instances.common.conv_implicit_gemm_dynamic import (
    DYNAMIC_CONV_PLACEHOLDER,
    DynamicConvGeometry,
    build_implicit_gemm_conv_dynamic,
)
from rocke.instances.common._conv_implicit_gemm_common import ConvDataSpec

ARCH = "gfx1151"

# rocKE ConvDataSpec dtype token per catalog dtype. The catalog / family.json
# `dtype` constraint uses "f16"/"bf16" (the ConvFpropAdapter's providerDtype
# vocabulary); the rocKE builder uses "fp16"/"bf16" -- distinct namespaces, so
# the exported .co symbol carries "fp16" while family.json constrains "f16".
_ROCKE_DTYPE = {"f16": "fp16", "bf16": "bf16"}

# WMMA warp geometry, fixed for gfx1151: 2x2 warp grid over the 16x16x16 WMMA
# atom at wave_size 32 -> block_size = 2*2*32 = 128.
WARP_M = 2
WARP_N = 2
WARP_TILE_M = 16
WARP_TILE_N = 16
WARP_TILE_K = 16
WAVE_SIZE = 32

# Per-operand vector widths. A/B load `vec` channel-contiguous elements (NHWC /
# KYXC innermost is C), so C must be a multiple of `vec` (enforced at selection
# time by family.json, NOT in the kernel). The default epilogue stores per-lane
# scalars, so vector_size_c MUST be 1 (a wider C store fails is_valid_spec).
VEC_A = 8
VEC_B = 8
VEC_C = 1

# Small tile-config table: NOT per-shape. Each entry is one shape-generic .co;
# the tuner runs each and caches the winner per problem. Kept small to bound
# build time -- add configs to widen the tuner's choice.
#
# (tile_m, tile_n, tile_k)
CONFIGS = [
    (64, 64, 64),
    (64, 64, 32),
    (128, 64, 32),
]


def _emit(out_dir, dtype):
    rocke_dtype = _ROCKE_DTYPE[dtype]
    for tile_m, tile_n, tile_k in CONFIGS:
        spec = ImplicitGemmConvSpec(
            problem=DYNAMIC_CONV_PLACEHOLDER,
            name="conv_igemm_fprop_dyn",
            data=ConvDataSpec(
                dtype_a=rocke_dtype, dtype_b=rocke_dtype, dtype_d=rocke_dtype
            ),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=WARP_M,
            warp_n=WARP_N,
            warp_tile_m=WARP_TILE_M,
            warp_tile_n=WARP_TILE_N,
            warp_tile_k=WARP_TILE_K,
            wave_size=WAVE_SIZE,
            pipeline="mem",
            epilogue="default",
            vector_size_a=VEC_A,
            vector_size_b=VEC_B,
            vector_size_c=VEC_C,
            groups=1,
        )

        geom = DynamicConvGeometry()
        symbol = geom.kernel_name(spec)
        kernel = build_implicit_gemm_conv_dynamic(spec, arch=ARCH, geom=geom)
        artifact = compile_kernel(kernel, arch=ARCH)

        # A zero-byte .co passes the fs::exists gate at catalog load and is
        # catalogued as valid, failing only later at hipModuleLoad; fail loudly.
        if not artifact.hsaco:
            raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
        out_path = os.path.join(out_dir, symbol + ".co")
        with open(out_path, "wb") as f:
            f.write(artifact.hsaco)

        print(
            f"symbol={symbol} shape=runtime tile={tile_m}x{tile_n}x{tile_k} "
            f"block={spec.block_size} bytes={len(artifact.hsaco)} path={out_path}"
        )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # f16 + bf16: diffusion / vision transformers run both. Same tile table --
    # dtype does not move the WMMA tile geometry.
    _emit(out_dir, "f16")
    _emit(out_dir, "bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
