# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The universal GEMM test helper refuses an engine name it does not know.

A typo in the engine name must fail with a clear error instead of silently picking
the C++ engine.
"""

from __future__ import annotations

import pytest

from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)
from rocke.tests.universal_gemm import lower_universal_gemm


def test_rejects_an_unknown_engine():
    spec = UniversalGemmSpec(
        name="engine_name_check",
        tile=TileSpec(32, 32, 8, 1, 1, 1, 16, 16, 4),
        trait=TraitSpec(pipeline="mem", epilogue="default"),
        data=DataSpec("fp64", "fp64", "fp64", "fp64"),
    )
    with pytest.raises(ValueError) as error:
        lower_universal_gemm(spec, "c++", arch="gfx942")
    assert str(error.value) == "engine must be 'python' or 'cpp', got 'c++'"
