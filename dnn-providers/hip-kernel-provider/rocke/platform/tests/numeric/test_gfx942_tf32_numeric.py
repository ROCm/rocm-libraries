# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import importlib.util
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
    # The installed wheel has no pybind extension. Keep numerical coverage in
    # that layout and demand native parity whenever the binding is supplied.
    backend = "python" if importlib.util.find_spec("rocke_engine") is None else "both"
    result = run(tmp_path, backend=backend)
    assert result["status"] == "pass"
    engines = {"python"} if backend == "python" else {"python", "cpp"}
    assert {row["engine"] for row in result["results"]} == engines
    assert len(result["results"]) == 12 * len(engines)
