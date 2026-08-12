# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# One-shot, build-time producer: emit a gfx1151 layernorm2d forward .co (HSACO)
# for the AOT catalog engine. Runtime never touches rocKE -- this is run by hand
# by a kernel author to drop a .co beside its family.json.
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_layernorm2d_co.py <out_dir>
#
# gfx1151 note: unlike rmsnorm2d, layernorm2d has NO wave64 gotcha. Its block
# reduction is a pure-LDS stable-Welford tree keyed only on block_size -- there
# is no cross-lane XOR-butterfly shuffle to miscompile at wave64. The spec's
# `wave_size` field is currently INERT in the builder body; we still pass 32 for
# consistency with the other gfx1151 families and to future-proof against a
# builder that starts honoring it.

import sys
import os

from rocke.instances.common.layernorm2d import LayerNorm2DSpec, build_layernorm2d
from rocke.helpers.compile import compile_kernel

ARCH = "gfx1151"

# Static-N variant table, mirroring rmsnorm2d: multiple perf configs per N so the
# catalog engine's measure-and-cache selection has a real choice. block_size/vec
# are perf-only (identical correct output); elems_per_thread = N/block_size
# selects a VGPR-cached single-pass body vs. a two-pass streaming body, so the
# spread is genuinely large. Every config must satisfy N % (block_size*vec) == 0
# and the gfx1151 caps (block_size <= 1024, LDS = 3*block_size*4 bytes for the
# Welford mean/M2/count triple).
#
# There is no layernorm2d_dynamic builder (only rmsnorm has one), so this family
# is static-N only: one .co per (N, block_size, vec). Runtime-N is a documented
# follow-up (see family.json _comment).
#
# (N, block_size, vec)
CONFIGS = [
    # N=2048 per-shape perf spread (single-pass through two-pass)
    (2048, 256, 4),
    (2048, 512, 4),
    (2048, 128, 8),
    (2048, 64, 8),
    # N=1024 shape tier
    (1024, 256, 4),
    (1024, 128, 8),
    # N=4096 shape tier
    (4096, 512, 4),
    (4096, 256, 8),
]

# bf16 static tier. Diffusion transformers run in bf16 and normalize over the
# full hidden dim; specialize the same N tiers LTX-Video and friends hit, reusing
# the f16 block_size/vec perf spread (dtype does not move the single-pass vs
# two-pass boundary).
#
# (N, block_size, vec)
CONFIGS_BF16 = [
    (2048, 256, 4),
    (2048, 512, 4),
    (2048, 128, 8),
    (2048, 64, 8),
    (4096, 512, 4),
    (4096, 256, 8),
]


def _emit_static(out_dir, configs, dtype):
    for n_per_block, block_size, vec in configs:
        spec = LayerNorm2DSpec(
            n_per_block=n_per_block,
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            save_mean_invstd=False,  # forward inference only; no stat outputs
            wave_size=32,  # inert for layernorm (pure-LDS Welford); set for consistency
        )
        kernel = build_layernorm2d(spec)
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


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # f16 kernels (layernorm2d/ family, dtype=f16 constraint per kernel)
    _emit_static(out_dir, CONFIGS, "f16")
    # bf16 kernels -- diffusion transformers run bf16.
    _emit_static(out_dir, CONFIGS_BF16, "bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
