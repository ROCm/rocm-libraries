################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S11a - GlobalWriteBatch _prolog alpha-before-loadC characterization.

Drives the designed alpha-before-loadC config
(``data/test_data/_designed/gfx942/s11a_prolog_alpha_before_loadc_packe.yaml``)
through the config-driven emit harness. Targets the ``_prolog`` alpha handling in
``Tensile/Components/GlobalWriteBatch.py``:

  - the ``codeMulAlpha`` not-None branch (int8 MI-out -> fp32 replaceHolder):
    line 836 (``srcRegName = rh.getParams()[2].getCompleteRegName()``) and
    line 837 (``module.add(VCvtI32toF32(...))``).

The alphaBeforeLoadC path requires MIArchVgpr=True, applyAlpha, beta, and
StorePriorityOpt (KernelWriterAssembly.py:16506-16520); 836/837 additionally
require I8 in / single compute / GlobalSplitU==1. ``emit_kernels_from_config``
runs assignDerivedParameters + emission, so the target lines fire during emit.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import assert_config_emits_golden

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s11a_prolog_alpha_before_loadc_packe.yaml",
)


def test_s11a_prolog_alpha_before_loadc_packe_golden(snapshot):
    """The alpha-before-load-C path converts its int32 accumulator to fp32."""
    assert_config_emits_golden(
        _CONFIG,
        _ARCH,
        snapshot,
        limit=8,
        validate_source=True,
        required_source_patterns=(
            (
                "int32-to-fp32 alpha conversion",
                r"^\s*v_cvt_f32_i32\b.*Convert MI out reg to fp32",
            ),
        ),
    )
