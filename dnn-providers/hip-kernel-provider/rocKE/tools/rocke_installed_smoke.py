#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Smoke-test an installed rocKE engine package.

The script is intended to run from an installed staging prefix, where it lives in
``bin/`` and the Python package/extension live under ``lib/pythonX.Y/site-packages``.
It adds those installed paths to ``sys.path`` before importing anything, so RockCI
can execute it directly from a staged test package. By default it checks only the
pure Python lowering path; the optional pybind backend is opt-in.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _add_installed_python_paths() -> None:
    script = Path(__file__).resolve()
    prefix = script.parents[1] if script.parent.name == "bin" else script.parent
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        prefix / "lib" / pyver / "site-packages",
        prefix / "lib64" / pyver / "site-packages",
        prefix / "lib" / "python" / "site-packages",
        prefix / "python",
        script.parent,
    ]
    for path in reversed(candidates):
        if path.exists():
            sys.path.insert(0, str(path))


def _build_smoke_kernel():
    from ck_dsl.instances.common.gemm_universal import (
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    return build_universal_gemm(
        UniversalGemmSpec(
            name="rocke_install_smoke",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4",
                scheduler="intrawave",
                epilogue="cshuffle",
            ),
        )
    )


def _smoke_python_engine() -> tuple[str, str]:
    from ck_dsl.core.ir_serialize import serialize
    from ck_dsl.core.lower_llvm import lower_kernel_to_llvm

    kernel = _build_smoke_kernel()
    ir_text = serialize(kernel)
    ll_text = lower_kernel_to_llvm(kernel)
    if "rocke_install_smoke" not in ll_text:
        raise RuntimeError("Python engine smoke did not lower the expected kernel")
    return ir_text, ll_text


def _smoke_pybind_engine(ir_text: str, py_ll_text: str) -> str:
    import ckc_engine  # type: ignore

    cpp_ll_text = ckc_engine.lower_serialized_ir(
        ir_text,
        arch="gfx950",
        flavor=os.environ.get("CK_DSL_LLVM_FLAVOR", ""),
    )
    if cpp_ll_text != py_ll_text:
        raise RuntimeError(
            "Installed ckc_engine output differs from Python engine output"
        )
    build_id = ckc_engine.build_id()
    if not build_id:
        raise RuntimeError("Installed ckc_engine did not report a build id")
    return build_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-pybind",
        action="store_true",
        help="Also check the optional ckc_engine pybind backend.",
    )
    args = parser.parse_args()

    _add_installed_python_paths()
    ir_text, py_ll_text = _smoke_python_engine()
    print("rocKE installed Python engine smoke: PASS")

    if args.check_pybind:
        build_id = _smoke_pybind_engine(ir_text, py_ll_text)
        print(f"rocKE installed pybind backend smoke: PASS ({build_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
