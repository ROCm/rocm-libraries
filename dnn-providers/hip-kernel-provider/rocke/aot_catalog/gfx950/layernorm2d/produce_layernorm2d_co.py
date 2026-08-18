# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# One-shot, build-time producer: emit gfx950 layernorm2d forward .co (HSACO) for the
# AOT catalog engine. Runtime never touches rocKE.
#
# Normally run by this family's CMakeLists via the rocKE build interpreter
# (${ROCKE_PYENV_PYTHON}). To run standalone:
#   <build>/rocke-pyenv/bin/python produce_layernorm2d_co.py <out_dir>
#
# gfx950 is CDNA4 / wave64, so wave_size=64 -- the opposite of the gfx1151 family.
# The STATIC path's block reduction is a pure-LDS stable-Welford tree keyed only on
# block_size and is wave-size inert; the DYNAMIC (runtime-N) path's Welford merge has
# a wave prologue where the size is load-bearing. 64 is the dataclass default here.

import os
import sys

from rocke.instances.common.layernorm2d import LayerNorm2DSpec, build_layernorm2d
from rocke.instances.common.layernorm2d_dynamic import (
    LayerNorm2DDynamicSpec,
    build_layernorm2d_dynamic,
)
from rocke.helpers.compile import compile_kernel

ARCH = "gfx950"
WAVE_SIZE = 64

# (N, block_size, vec). Every config must satisfy N % (block_size*vec) == 0,
# block_size <= 1024, LDS = 3 * block_size * 4 bytes.
CONFIGS = [
    (2048, 256, 4),
    (2048, 512, 4),
    (2048, 128, 8),
    (2048, 64, 8),
    (1024, 256, 4),
    (1024, 128, 8),
    (4096, 512, 4),
    (4096, 256, 8),
]
CONFIGS_BF16 = list(CONFIGS)

# (block_size, vec) -- runtime-N binaries, one per vec alignment class.
DYNAMIC_CONFIGS = [(256, 4), (128, 8)]


def _write(out_dir, symbol, artifact):
    if not artifact.hsaco:
        raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
    path = os.path.join(out_dir, symbol + ".co")
    with open(path, "wb") as f:
        f.write(artifact.hsaco)
    return path


def _emit_static(out_dir, configs, dtype):
    for n_per_block, block_size, vec in configs:
        spec = LayerNorm2DSpec(
            n_per_block=n_per_block,
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            save_mean_invstd=False,
            wave_size=WAVE_SIZE,
        )
        artifact = compile_kernel(build_layernorm2d(spec), arch=ARCH)
        symbol = spec.kernel_name()
        path = _write(out_dir, symbol, artifact)
        print(
            f"symbol={symbol} N={n_per_block} block_size={block_size} vec={vec} "
            f"bytes={len(artifact.hsaco)} path={path}"
        )


def _emit_dynamic(out_dir, configs, dtype):
    for block_size, vec in configs:
        spec = LayerNorm2DDynamicSpec(
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            save_mean_invstd=False,
            wave_size=WAVE_SIZE,
        )
        artifact = compile_kernel(build_layernorm2d_dynamic(spec), arch=ARCH)
        symbol = spec.kernel_name()
        path = _write(out_dir, symbol, artifact)
        print(
            f"symbol={symbol} N=runtime block_size={block_size} vec={vec} "
            f"bytes={len(artifact.hsaco)} path={path}"
        )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    _emit_static(out_dir, CONFIGS, "f16")
    _emit_dynamic(out_dir, DYNAMIC_CONFIGS, "f16")
    _emit_static(out_dir, CONFIGS_BF16, "bf16")
    _emit_dynamic(out_dir, DYNAMIC_CONFIGS, "bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
