# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import os
import pytest
from rocke.runtime.hip_module import get_device_arch
from rocke.examples.gfx942.tf32_numerics import run


def test_gfx942_tf32_numeric(tmp_path):
    arch = get_device_arch()
    if not arch or arch.split(":")[0] != "gfx942":
        if os.getenv("ROCKE_REQUIRE_GFX942") == "1":
            pytest.fail(f"Required gfx942 device absent: {arch}")
        pytest.skip("requires gfx942")
    result = run(tmp_path, backend="both")
    assert result["status"] == "pass"
    assert len(result["results"]) == 24
