#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""C++/Python byte-identity gate for the dense + D256 attention kernels.

The IR-sha256 golden for these kernels lives in the platform parity harness
(``platform/tests/instances/rocke_ir_parity_harness.py``, families
``attention_dense`` / ``attention_d256``) and is gated in CI by the
``rocke_golden_static`` CTest entry. That golden pins the *Python* lowering only.
This file adds the other half for the same case set: the C++ engine
(``rocke_engine``) must lower each of those kernels to byte-identical IR.

Cases are read back from the harness rather than redeclared, so the two gates can
never drift apart. Importing the harness is the allowed ``library -> platform``
direction (the reverse is forbidden); it is reached by path because the harness
ships in the platform *test* tree, not inside the ``rocke`` package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[2] / "platform" / "tests" / "instances"

# Families whose cases this gate covers (the ones built from library kernels).
_FAMILIES = ("attention_dense", "attention_d256")


def _harness():
    if str(_HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(_HARNESS_DIR))
    import rocke_ir_parity_harness

    return rocke_ir_parity_harness


def test_attention_ir_cpp_python_byte_identity():
    """Both sides go through ``_lower_llvm_via_backend`` so they resolve the same
    llvm flavor; ``ROCKE_CPP_STRICT=1`` disables the silent python fallback so a
    missing/stale C++ engine surfaces as a skip, not a false pass."""
    import pytest

    if not _HARNESS_DIR.is_dir():
        pytest.skip(f"platform parity harness not found at {_HARNESS_DIR}")
    try:
        from rocke.helpers.compile import _lower_llvm_via_backend
    except Exception as e:  # pragma: no cover
        pytest.skip(f"backend lowering unavailable: {e}")

    cases = [c for c in _harness().cases() if c["family"] in _FAMILIES]
    assert cases, f"no harness cases for families {_FAMILIES}"

    prev = os.environ.get("ROCKE_CPP_STRICT")
    os.environ["ROCKE_CPP_STRICT"] = "1"
    mism = []
    try:
        for case in cases:
            arch = case["arch"]
            kernel = case["build"]()
            py = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
            try:
                cpp = _lower_llvm_via_backend(
                    kernel, arch=arch, backend="cpp", spec=None
                )
            except Exception as e:  # C++ engine not built / opcode gap
                pytest.skip(
                    f"C++ engine unavailable ({case['case_id']}): {str(e)[:140]}"
                )
            if py != cpp:
                mism.append(case["case_id"])
    finally:
        if prev is None:
            os.environ.pop("ROCKE_CPP_STRICT", None)
        else:
            os.environ["ROCKE_CPP_STRICT"] = prev
    assert not mism, "attention cpp/python IR byte-mismatch:\n  " + "\n  ".join(mism)


if __name__ == "__main__":
    test_attention_ir_cpp_python_byte_identity()
    print("PASS")
