#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Run one gfx942 unified 3D table generator for the build.

The engine's host tests check its C++ restatements of rocKE rules (admission,
split rule, kernel arguments) against tables that rocKE itself
answers. The tables are generated into the build tree on every build that touches
their inputs and are never committed, so they cannot go stale.

    gen_gfx942_attention_unified_3d_table.py GENERATOR --out TABLE.inc --depfile TABLE.d

GENERATOR is one of the ``gen_gfx942_attention_unified_3d_*.py`` scripts beside this
one; it prints its table to stdout. This driver puts the rocKE source tree on
``sys.path`` (the generators need only the standard library and rocKE's Python),
runs it, writes the table, and writes a Makefile-style depfile listing every
rocKE module the generator imported, so a change to any of them regenerates the
table.
"""

import argparse
import contextlib
import io
import os
import runpy
import sys
from pathlib import Path

# <provider>/src/tests/engines/kernel_ingestor_engine/packs/<this>
_PROVIDER = Path(__file__).resolve().parents[5]
_ROCKE_PATHS = (
    _PROVIDER / "rocke" / "library",
    _PROVIDER / "rocke" / "platform" / "python",
)


def _depfile_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(" ", "\\ ").replace("#", "\\#")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("generator", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depfile", type=Path, required=True)
    args = parser.parse_args()

    # The pack step lowers through the Python engine too (HkpPackaging.cmake).
    os.environ.setdefault("ROCKE_BACKEND", "python")
    sys.path[:0] = [str(p) for p in _ROCKE_PATHS]

    generator = args.generator.resolve()
    table = io.StringIO()
    with contextlib.redirect_stdout(table):
        runpy.run_path(str(generator), run_name="__main__")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table.getvalue(), encoding="utf-8")

    inputs = {generator, Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source and Path(source).resolve().is_relative_to(_PROVIDER / "rocke"):
            inputs.add(Path(source).resolve())
    args.depfile.write_text(
        f"{_depfile_path(args.out)}: "
        + " \\\n  ".join(_depfile_path(p) for p in sorted(inputs))
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
