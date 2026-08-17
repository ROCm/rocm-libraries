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
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    rel_site_packages = (
        Path("lib") / pyver / "site-packages",
        Path("lib64") / pyver / "site-packages",
        Path("lib") / "python" / "site-packages",
        Path("python"),
    )
    # The script may be installed at <prefix>/bin/ (standalone) or nested under a
    # provider test subdir such as <prefix>/bin/<provider>/. Probe a bounded set
    # of ancestor directories as candidate install prefixes, and always include
    # the script's own directory (covers rocke co-located next to the script in
    # the provider test bucket).
    candidate_prefixes = [script.parents[i] for i in range(min(4, len(script.parents)))]
    found: list[Path] = []
    for prefix in candidate_prefixes:
        for rel in rel_site_packages:
            path = prefix / rel
            if path.is_dir() and path not in found:
                found.append(path)
    if script.parent not in found:
        found.append(script.parent)
    # Insert closest-prefix matches at the front of sys.path (highest priority).
    for path in reversed(found):
        sys.path.insert(0, str(path))


def _build_smoke_kernel():
    from rocke.instances.common.gemm_universal import (
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
    from rocke.core.ir_serialize import serialize
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    kernel = _build_smoke_kernel()
    ir_text = serialize(kernel)
    ll_text = lower_kernel_to_llvm(kernel)
    if "rocke_install_smoke" not in ll_text:
        raise RuntimeError("Python engine smoke did not lower the expected kernel")
    return ir_text, ll_text


def _smoke_pybind_engine(ir_text: str, py_ll_text: str) -> str:
    import rocke_engine  # type: ignore

    cpp_ll_text = rocke_engine.lower_serialized_ir(
        ir_text,
        arch="gfx950",
        flavor=os.environ.get("ROCKE_LLVM_FLAVOR", ""),
    )
    if cpp_ll_text != py_ll_text:
        raise RuntimeError(
            "Installed rocke_engine output differs from Python engine output"
        )
    build_id = rocke_engine.build_id()
    if not build_id:
        raise RuntimeError("Installed rocke_engine did not report a build id")
    return build_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-pybind",
        action="store_true",
        help="Also check the optional rocke_engine pybind backend.",
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
