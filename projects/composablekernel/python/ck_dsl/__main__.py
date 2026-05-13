# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Top-level CLI for `python -m ck_dsl`.

The DSL no longer ships a single hard-coded "emit this kernel" path.
Each kernel family has its own runnable module:

    python -m ck_dsl.examples.bake_off_implicit_gemm   --output-dir <dir>
    python -m ck_dsl.examples.bake_off_direct_conv_16c --output-dir <dir>
    python -m ck_dsl.examples.bake_off_direct_conv_4c  --output-dir <dir>

    python -m ck_dsl.run_manifest <hsaco> <manifest.json> [--verify]
    python -m ck_dsl.sweep_bench <sweep_manifest.json> [--csv ...]

This top-level entry point just prints those discoverable modules.
"""

from __future__ import annotations


def main() -> int:
    print(__doc__.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
