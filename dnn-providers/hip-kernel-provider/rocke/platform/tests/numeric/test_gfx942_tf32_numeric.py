# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from rocke.examples.gfx942.tf32_numerics import run
from rocke.tests.engines import all_engines_backend, available_engines
from rocke.tests.gpu import current_device_arch_in


def test_gfx942_tf32_numeric(tmp_path):
    current_device_arch_in(("gfx942",))
    # The installed wheel has no pybind extension. Keep numerical coverage in
    # that layout and demand native parity whenever the binding is supplied.
    result = run(tmp_path, backend=all_engines_backend())
    assert result["status"] == "pass"
    engines = set(available_engines())
    assert {row["engine"] for row in result["results"]} == engines
    assert len(result["results"]) == 12 * len(engines)
