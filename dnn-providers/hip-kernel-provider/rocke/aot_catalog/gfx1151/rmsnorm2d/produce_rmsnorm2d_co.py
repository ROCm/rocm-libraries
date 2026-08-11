# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# One-shot, build-time producer: emit a gfx1151 rmsnorm2d forward .co (HSACO)
# for the AOT catalog engine. Runtime never touches rocKE -- this is run by hand
# by a kernel author to drop a .co beside its family.json.
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_rmsnorm2d_co.py <out_dir>
#
# CRITICAL gfx1151 gotcha: wave_size MUST be 32. The default 64 emits an invalid
# lane-mask-32 shuffle on wave32 hardware and silently returns wrong results.

import sys
import os

from rocke.instances.common.rmsnorm2d import RMSNorm2DSpec, build_rmsnorm2d
from rocke.instances.common.rmsnorm2d_dynamic import (
    RMSNorm2DDynamicSpec,
    build_rmsnorm2d_dynamic,
)
from rocke.helpers.compile import compile_kernel

ARCH = "gfx1151"

# Phase 2 variant table: multiple perf configs per N so the catalog engine's
# measure-and-cache selection has a real choice. block_size/vec are perf-only
# (identical correct output); elems_per_thread = N/block_size selects a
# VGPR-cached single-pass body vs. a two-pass streaming body, so the spread is
# genuinely large. Every config must satisfy N % (block_size*vec) == 0 and the
# gfx1151 caps (block_size <= 1024, LDS = block_size*4 bytes).
#
# (N, block_size, vec) -> elems_per_thread
CONFIGS = [
    # N=2048 per-shape perf spread (single-pass through two-pass)
    (2048, 256, 4),  # ept=8  (baseline; already shipped)
    (2048, 512, 4),  # ept=4
    (2048, 128, 8),  # ept=16
    (2048, 64, 8),  # ept=32 (streaming two-pass)
    # N=1024 shape tier
    (1024, 256, 4),  # ept=4
    (1024, 128, 8),  # ept=8
    # N=4096 shape tier
    (4096, 512, 4),  # ept=8
    (4096, 256, 8),  # ept=16
]

# bf16 static tier. LTX-Video (and most diffusion transformers) run in bf16, and
# every LTX RMSNorm normalizes over the full hidden dim -- N=2048 (inner_dim =
# 32 heads * 64) for the block/QK norms and N=4096 for the text projection. So we
# specialize exactly those two N in bf16, reusing the same block_size/vec perf
# spread as f16 (dtype does not change the single-pass vs two-pass boundary).
#
# (N, block_size, vec)
CONFIGS_BF16 = [
    # N=2048 (LTX block + QK norms)
    (2048, 256, 4),
    (2048, 512, 4),
    (2048, 128, 8),
    (2048, 64, 8),
    # N=4096 (LTX text-projection norm)
    (4096, 512, 4),
    (4096, 256, 8),
]

# Phase 3 runtime-N variant table: N is a runtime kernel argument, so each of
# these binaries serves EVERY N that is a multiple of `vec` (flat row-major
# addressing needs vec-aligned row starts -> N % vec == 0; enforced by a
# `multiple_of` constraint in family.json, not in the kernel). Every real
# ComfyUI RMSNorm hidden size is a multiple of 8, so this is not a limit in
# practice. They compete with the static specializations on the listed N tiers
# and are the sole match for any other multiple-of-vec N (Flux 3072, SD3.5 2432,
# ...). block_size/vec are the only knobs; wave_size MUST be 32 for the same
# gfx1151 reason as above.
#
# (block_size, vec)
DYNAMIC_CONFIGS = [
    (256, 4),
    (128, 8),
]


def _emit_static(out_dir, configs, dtype):
    for n_per_block, block_size, vec in configs:
        spec = RMSNorm2DSpec(
            n_per_block=n_per_block,
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            save_inv_rms=False,
            wave_size=32,  # gfx1151 is wave32; 64 miscompiles the cross-lane reduction
        )
        kernel = build_rmsnorm2d(spec)
        artifact = compile_kernel(kernel, arch=ARCH)

        symbol = spec.kernel_name()
        # A zero-byte .co passes the fs::exists gate at catalog load and is
        # catalogued as valid, failing only later at hipModuleLoad; fail loudly here.
        if not artifact.hsaco:
            raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
        out_path = os.path.join(out_dir, symbol + ".co")
        with open(out_path, "wb") as f:
            f.write(artifact.hsaco)

        print(
            f"symbol={symbol} N={n_per_block} block_size={block_size} vec={vec} "
            f"ept={spec.elems_per_thread} bytes={len(artifact.hsaco)} path={out_path}"
        )


def _emit_dynamic(out_dir, configs, dtype):
    for block_size, vec in configs:
        spec = RMSNorm2DDynamicSpec(
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            save_inv_rms=False,
            wave_size=32,  # gfx1151 is wave32; 64 miscompiles the cross-lane reduction
        )
        kernel = build_rmsnorm2d_dynamic(spec)
        artifact = compile_kernel(kernel, arch=ARCH)

        symbol = spec.kernel_name()
        # A zero-byte .co passes the fs::exists gate at catalog load and is
        # catalogued as valid, failing only later at hipModuleLoad; fail loudly here.
        if not artifact.hsaco:
            raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
        out_path = os.path.join(out_dir, symbol + ".co")
        with open(out_path, "wb") as f:
            f.write(artifact.hsaco)

        print(
            f"symbol={symbol} N=runtime block_size={block_size} vec={vec} "
            f"bytes={len(artifact.hsaco)} path={out_path}"
        )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # f16 kernels (rmsnorm2d/ family, dtype=f16 constraint per kernel)
    _emit_static(out_dir, CONFIGS, "f16")
    _emit_dynamic(out_dir, DYNAMIC_CONFIGS, "f16")

    # bf16 kernels -- LTX-Video runs bf16; the runtime-N dynamic
    # kernels share the same DYNAMIC_CONFIGS table (dtype is the only difference).
    _emit_static(out_dir, CONFIGS_BF16, "bf16")
    _emit_dynamic(out_dir, DYNAMIC_CONFIGS, "bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
