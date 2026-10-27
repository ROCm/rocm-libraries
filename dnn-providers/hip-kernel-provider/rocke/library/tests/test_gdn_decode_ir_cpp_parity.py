# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""C++/Python byte-identity gate for the GDN/KDA decode kernels.

GDN decode is a Python-only family: its spec and emitter have no C++ twin, and
``platform/tools/check_byte_identity.py`` runs no GDN case. The C++ engine
still lowers every GDN kernel a caller builds (any kernel goes through the
serialized-IR seam, ``rocke_engine.lower_serialized_ir``), so the two engines'
lowering of this family's IR must agree. The family golden
(``test_gdn_decode_golden.py``) pins the *Python* lowering only; this file is
the other half for the same case set and every golden LLVM flavor: the C++
engine must produce byte-identical LLVM IR.

Engine absence is a skip on an ordinary CPU run. Under ``ROCKE_BACKEND=cpp``
or ``both`` -- the differential lane, where the engine is expected -- it is a
failure instead, so that lane cannot go green without comparing anything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import test_gdn_decode_golden as golden  # noqa: E402

_ARCH = "gfx950"


def _engine():
    expected = os.environ.get("ROCKE_BACKEND") in ("cpp", "both")
    try:
        import rocke_engine
    except ImportError as e:
        if expected:
            pytest.fail(
                f"ROCKE_BACKEND={os.environ['ROCKE_BACKEND']} expects the C++ "
                f"engine, but rocke_engine is not importable: {e}"
            )
        pytest.skip(f"rocke_engine extension not built/importable: {e}")
    return rocke_engine


def test_gdn_decode_ir_cpp_python_byte_identity():
    engine = _engine()
    from rocke.core.ir_golden import GOLDEN_FLAVORS
    from rocke.core.ir_serialize import serialize
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    cases = golden._cases()
    # Stronger than ``assert cases``: every case the golden records must be
    # compared, so a case list that silently shrank cannot pass here.
    recorded = set(json.loads(golden._GOLDEN.read_text())["flavors"]["llvm20"]["cases"])
    assert set(cases) == recorded, sorted(set(cases) ^ recorded)

    mismatches, compared = [], 0
    for flavor in GOLDEN_FLAVORS:
        for case_id, build in cases.items():
            kernel = build()
            py = _lower_kernel_to_llvm_python(kernel, arch=_ARCH, llvm_flavor=flavor)
            cpp = engine.lower_serialized_ir(
                serialize(kernel), arch=_ARCH, flavor=flavor
            )
            compared += 1
            if cpp != py:
                mismatches.append(f"{flavor}/{case_id}")
    assert compared == len(cases) * len(GOLDEN_FLAVORS)
    assert not mismatches, "gdn_decode C++/Python IR byte-mismatch:\n  " + "\n  ".join(
        mismatches
    )
