# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Top-level CLI for `python -m rocke`.

The DSL no longer ships a single hard-coded "emit this kernel" path.
Each kernel family has its own runnable module:

    python -m rocke.examples.common.ck_tile_parity           --arch <gfx...>

    python -m rocke.run_manifest <hsaco> <manifest.json> [--verify]
    python -m rocke.sweep_bench <sweep_manifest.json> [--csv ...]

Carved-out kernel verticals ship their own runnable modules under the rocke
library instead (convolution bake-offs, for example, are
``python -m builders.common.bake_off_implicit_gemm`` /
``builders.common.bake_off_direct_conv_{16c,4c}``).

This top-level entry point just prints those discoverable modules.
"""

from __future__ import annotations


def main() -> int:
    print(__doc__.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
