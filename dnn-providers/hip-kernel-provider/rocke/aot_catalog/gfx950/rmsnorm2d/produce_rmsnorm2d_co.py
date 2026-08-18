# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# One-shot, build-time producer: emit a gfx950 rmsnorm2d forward .co (HSACO)
# for the AOT catalog engine. Runtime never touches rocKE -- this is run by hand
# by a kernel author to drop a .co beside its family.json.
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_rmsnorm2d_co.py <out_dir>
#
# gfx950 is CDNA4 / wave64, so wave_size=64 -- the opposite of the gfx1151
# family, whose XOR-butterfly reduction must be built at 32. 64 is also the
# RMSNorm2D*Spec default, so this arch is the un-special case.

import sys
import os

from rocke.instances.common.rmsnorm2d import RMSNorm2DSpec, build_rmsnorm2d
from rocke.instances.common.rmsnorm2d_dynamic import (
    RMSNorm2DDynamicSpec,
    build_rmsnorm2d_dynamic,
)
from rocke.helpers.compile import compile_kernel

ARCH = "gfx950"
WAVE_SIZE = 64  # CDNA4 wave64

# Static per-N specializations. block_size/vec are perf-only (identical correct
# output); elems_per_thread = N/block_size selects a VGPR-cached single-pass body
# vs. a two-pass streaming body, so the measure-and-cache selector has a real
# choice. Every config must satisfy N % (block_size*vec) == 0, block_size <= 1024,
# LDS = block_size*4 bytes.
#
# The N tiers are the hidden sizes real transformer blocks normalize over; they
# are shape facts, not arch facts, so they carry over from the gfx1151 table
# unchanged. block_size stays >= 64 so every config is at least one full wave64.
#
# (N, block_size, vec)
CONFIGS = [
    # N=2048 per-shape perf spread (single-pass through two-pass)
    (2048, 256, 4),  # ept=8
    (2048, 512, 4),  # ept=4
    (2048, 128, 8),  # ept=16
    (2048, 64, 8),  # ept=32 (streaming two-pass, one wave64)
    # N=1024 shape tier
    (1024, 256, 4),  # ept=4
    (1024, 128, 8),  # ept=8
    # N=4096 shape tier
    (4096, 512, 4),  # ept=8
    (4096, 256, 8),  # ept=16
]

# bf16 static tier: bf16 is the default dtype for CDNA inference workloads, so it
# gets the same N coverage as f16 here (the gfx1151 family trimmed bf16 to the two
# LTX-Video tiers; on gfx950 there is no reason to prefer f16).
CONFIGS_BF16 = list(CONFIGS)

# Runtime-N variants: N is a runtime kernel argument, so each of these binaries
# serves EVERY N that is a multiple of `vec` (flat row-major addressing needs
# vec-aligned row starts -> N % vec == 0; enforced by a `multiple_of` constraint
# in family.json, not in the kernel). They compete with the static specializations
# on the listed N tiers and are the sole match for any other multiple-of-vec N.
#
# (block_size, vec)
DYNAMIC_CONFIGS = [
    (256, 4),
    (128, 8),
]


def _write_co(out_dir, symbol, artifact):
    # A zero-byte .co passes the fs::exists gate at catalog load and is catalogued
    # as valid, failing only later at hipModuleLoad; fail loudly here instead.
    if not artifact.hsaco:
        raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
    out_path = os.path.join(out_dir, symbol + ".co")
    with open(out_path, "wb") as f:
        f.write(artifact.hsaco)
    return out_path


def _emit_static(out_dir, configs, dtype):
    for n_per_block, block_size, vec in configs:
        spec = RMSNorm2DSpec(
            n_per_block=n_per_block,
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            save_inv_rms=False,
            wave_size=WAVE_SIZE,
        )
        kernel = build_rmsnorm2d(spec)
        artifact = compile_kernel(kernel, arch=ARCH)
        symbol = spec.kernel_name()
        out_path = _write_co(out_dir, symbol, artifact)
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
            wave_size=WAVE_SIZE,
        )
        kernel = build_rmsnorm2d_dynamic(spec)
        artifact = compile_kernel(kernel, arch=ARCH)
        symbol = spec.kernel_name()
        out_path = _write_co(out_dir, symbol, artifact)
        print(
            f"symbol={symbol} N=runtime block_size={block_size} vec={vec} "
            f"bytes={len(artifact.hsaco)} path={out_path}"
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
