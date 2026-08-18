# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# One-shot, build-time producer: emit gfx950 pointwise activation .co (HSACO) for the
# AOT catalog engine. Runtime never touches rocKE.
#
# Normally run by this family's CMakeLists via the rocKE build interpreter
# (${ROCKE_PYENV_PYTHON}). To run standalone:
#   <build>/rocke-pyenv/bin/python produce_activation_co.py <out_dir>
#
# This is the most portable family in the catalog: ElementwiseSpec has no wave_size
# field at all (the kernel is per-element f32 math with a scalar tail), so the gfx950
# port is the arch string and nothing else. block_size/vec are pure perf knobs.

import os
import sys

from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise
from rocke.helpers.compile import compile_kernel

ARCH = "gfx950"

# The two activations the injection's pointwise path can express: F.silu maps to
# PointwiseMode::SWISH_FWD (beta == 1) and F.gelu(approximate="tanh") to
# GELU_APPROX_TANH_FWD. Exact-erf gelu has no builder and is left to fall back.
OPS = ["silu", "gelu_tanh"]
DTYPES = ["f16", "bf16"]

# (block_size, vec) -- three points so measure-and-cache has a real choice.
TUNING = [(256, 8), (512, 8), (256, 4)]


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    for op in OPS:
        for dtype in DTYPES:
            for block_size, vec in TUNING:
                spec = ElementwiseSpec(op=op, dtype=dtype, block_size=block_size, vec=vec)
                artifact = compile_kernel(build_elementwise(spec), arch=ARCH)
                symbol = spec.kernel_name()
                if not artifact.hsaco:
                    print(f"ERROR {symbol}: compiled .co is empty", file=sys.stderr)
                    return 1
                path = os.path.join(out_dir, symbol + ".co")
                with open(path, "wb") as f:
                    f.write(artifact.hsaco)
                print(
                    f"symbol={symbol} op={op} dtype={dtype} block_size={block_size} "
                    f"vec={vec} bytes={len(artifact.hsaco)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
