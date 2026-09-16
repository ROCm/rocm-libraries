################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 2 — helper-kernel emit goldens.

The GEMM solutions also generate *helper* kernels (beta-only, type-conversion,
activation-enum headers) emitted as HIP C++ via a separate path
(``KernelWriterBetaOnly`` / ``KernelWriterConversion`` / ``KernelWriterModules``)
from the assembly GEMM kernels. This suite drives that path across curated
configs whose solutions request such helpers (UseBeta, GSU/conversion, fused
activation), covering otherwise-unreached emit code. Order-invariant golden
(name + emit return code); see ``target.md``.
"""

import glob
import os

import pytest

from codegen_harness import emit_helpers_from_logic

pytestmark = pytest.mark.unit

_DATA = os.path.join(os.path.dirname(__file__), "data")

# Configs whose solutions tend to request helper kernels: beta (UseBeta),
# conversion (GSU / mixed precision), and fused activation.
_FILES = [
    "gfx942/HSS_BH_Bias.yaml",       # UseBeta + activation -> beta-only / enum
    "gfx942/GSU.yaml",               # GlobalSplitU -> conversion kernels
    "gfx942/F8N_multi.yaml",         # mixed precision + amaxD
    "gfx950/HHS.yaml",
    "gfx950/I8_GSU.yaml",            # int8 + GSU conversion
    "gfx90a/HSS.yaml",
]


def test_helper_emit_digests(snapshot):
    out = {}
    for rel in _FILES:
        path = os.path.join(_DATA, rel)
        if not os.path.exists(path):
            continue
        helpers = emit_helpers_from_logic(path)
        assert all(err == 0 for _name, err in helpers)
        out[rel] = helpers
    assert out
    assert out == snapshot
