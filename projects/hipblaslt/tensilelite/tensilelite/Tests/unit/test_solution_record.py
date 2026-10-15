# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A real logic file's solution record is built by solution_from_state."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).parent
    / "characterization"
    / "TensileCreateLibraryRun"
    / "data"
    / "logic_gfx942_HSS_BH_tiny.yaml"
)


def test_library_logic_solution_record_has_kernel_name_and_size_mapping():
    if importlib.util.find_spec("tensilelite_write") is None:
        pytest.skip("tensilelite_write extension was not built")

    from Tensile.Common.Architectures import SUPPORTED_ISA
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.Utilities import state
    from Tensile.Contractions import _solution_record
    from Tensile.LibraryIO import parseLibraryLogicFile
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    logic = parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isaInfoMap, False
    )
    logic.library.applyNaming(False)
    solution = next(iter(logic.library.solutions.values()))
    solution.sizeMapping.CUOccupancy = 12345

    record = state(solution)
    assert record["kernelName"]
    assert record["name"]
    assert isinstance(record["sizeMapping"], dict)
    assert record["sizeMapping"]["macroTile"]
    assert record["sizeMapping"]["CUOccupancy"] == 12345

    cpp = _solution_record(solution.originalSolution._state)
    assert record["problemType"] == cpp["problemType"]
    assert record["problemPredicate"] == cpp["problemPredicate"]
    assert record["taskPredicate"] == cpp["taskPredicate"]
    assert record["hardwarePredicate"] == cpp["hardwarePredicate"]
    cpp["sizeMapping"]["CUOccupancy"] = 12345
    assert record["sizeMapping"] == cpp["sizeMapping"]
