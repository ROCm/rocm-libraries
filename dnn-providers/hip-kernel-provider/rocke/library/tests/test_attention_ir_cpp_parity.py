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
    missing/stale C++ engine cannot produce a false pass.

    A per-case failure is classified by exception type rather than lumped
    together, because the three outcomes need three different verdicts:

    * ``BackendCoverageGap`` -- an arch named in ``backend.CPP_UNPORTED_ARCHES``,
      the one sanctioned gap taxonomy. Reported as a skip.
    * ``BackendError`` -- the engine is not importable at all. Whole-run skip;
      an absent engine cannot become present mid-loop.
    * anything else -- the engine is present and this case would not lower.
      That is a defect, so it propagates and the gate goes red.

    The classification is what makes the loop safe to continue through. Bailing
    out on the first failure used to hide the ``attention_d256`` cases entirely:
    the ``attention_dense`` cases sort ahead of them, so an engine that could not
    lower dense skipped the test before d256 was ever built.
    """
    import pytest

    if not _HARNESS_DIR.is_dir():
        pytest.skip(f"platform parity harness not found at {_HARNESS_DIR}")
    try:
        from rocke.core.backend import BackendCoverageGap, BackendError
        from rocke.helpers.compile import _lower_llvm_via_backend
    except Exception as e:  # pragma: no cover
        pytest.skip(f"backend lowering unavailable: {e}")

    cases = [c for c in _harness().cases() if c["family"] in _FAMILIES]
    # Stronger than ``assert cases``: a renamed or dropped family would still
    # leave a non-empty list and the gate would go green having never looked at
    # it. Harness drift must not be reported as an engine gap.
    empty = set(_FAMILIES) - {c["family"] for c in cases}
    assert not empty, f"harness declares no cases for families {sorted(empty)}"

    prev = os.environ.get("ROCKE_CPP_STRICT")
    os.environ["ROCKE_CPP_STRICT"] = "1"
    mism = []
    gaps = []
    try:
        for case in cases:
            arch = case["arch"]
            kernel = case["build"]()
            py = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
            try:
                cpp = _lower_llvm_via_backend(
                    kernel, arch=arch, backend="cpp", spec=None
                )
            except BackendCoverageGap:  # declared gap; subclass, so catch first
                gaps.append(f"{case['case_id']} ({arch})")
                continue
            except BackendError as e:  # engine absent -- same for every case
                pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
            if py != cpp:
                mism.append(case["case_id"])
    finally:
        if prev is None:
            os.environ.pop("ROCKE_CPP_STRICT", None)
        else:
            os.environ["ROCKE_CPP_STRICT"] = prev
    assert not mism, "attention cpp/python IR byte-mismatch:\n  " + "\n  ".join(mism)
    if gaps:
        pytest.skip("CPP_UNPORTED_ARCHES cases not gated:\n  " + "\n  ".join(gaps))


if __name__ == "__main__":
    test_attention_ir_cpp_python_byte_identity()
    print("PASS")
