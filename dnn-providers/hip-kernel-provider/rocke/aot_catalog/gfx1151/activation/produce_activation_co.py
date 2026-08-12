# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# One-shot, build-time producer: emit gfx1151 elementwise-activation .co (HSACO)
# for the AOT catalog engine. Runtime never touches rocKE -- this is run by hand
# by a kernel author to drop .co files beside their family.json.
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_activation_co.py <out_dir>
#
# gfx1151 note: elementwise activations are purely per-element (compute in f32,
# no cross-lane reduction), so there is no wave_size gotcha at all -- the
# ElementwiseSpec has no wave_size field. A single .co serves ANY numel: the CK
# Tile 21_elementwise parity kernel vectorises full block_size*vec slabs and
# falls through to a per-element scalar tail for the ragged remainder, so numel
# need not be a multiple of vec (family.json carries only numel {min: 1}).

import sys
import os

from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise
from rocke.helpers.compile import compile_kernel

ARCH = "gfx1151"

# v1 activations: the two unary ops the elementwise builder implements that map
# to real hipDNN PointwiseModes our ActivationAdapter accepts.
#   * "silu"      <- PointwiseMode::SWISH_FWD (beta == 1)
#   * "gelu_tanh" <- PointwiseMode::GELU_APPROX_TANH_FWD
# Exact erf GELU (PointwiseMode::GELU_FWD) has no builder op yet and is declined
# by the adapter -- a documented follow-up (add an `erf` op to elementwise.py).
OPS = ["silu", "gelu_tanh"]

DTYPES = ["f16", "bf16"]

# Perf-only tuning spread per (op, dtype): identical correct output, different
# block_size/vec so the catalog engine's measure-and-cache selection has a real
# choice. Every variant serves every numel (scalar tail handles the remainder);
# the grid is ceil_div(numel, block_size*vec).
#
# (block_size, vec)
TUNING = [
    (256, 8),
    (512, 8),
    (256, 4),
]


def _emit(out_dir, ops, dtypes, tuning):
    for op in ops:
        for dtype in dtypes:
            for block_size, vec in tuning:
                spec = ElementwiseSpec(
                    op=op,
                    dtype=dtype,
                    block_size=block_size,
                    vec=vec,
                )
                kernel = build_elementwise(spec)
                artifact = compile_kernel(kernel, arch=ARCH)

                symbol = spec.kernel_name()
                # A zero-byte .co passes the fs::exists gate at catalog load and is
                # catalogued as valid, failing only later at hipModuleLoad; fail
                # loudly here.
                if not artifact.hsaco:
                    raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
                out_path = os.path.join(out_dir, symbol + ".co")
                with open(out_path, "wb") as f:
                    f.write(artifact.hsaco)

                print(
                    f"symbol={symbol} op={op} dtype={dtype} block_size={block_size} "
                    f"vec={vec} bytes={len(artifact.hsaco)} path={out_path}"
                )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    _emit(out_dir, OPS, DTYPES, TUNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
