# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import re
from copy import deepcopy

import pytest

from Tensile import LibraryIO, __version__
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.TensileLogic.Run import Check, _runChecks


pytestmark = pytest.mark.unit

_KERNEL_NAME = (
    "Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_"
    "MI16x16x1_UserArgs_shortname0_gfx950"
)


def _plain_bbs_problem_type():
    return {
        "OperationType": "GEMM",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "S",
        "TransposeA": True,
        "TransposeB": False,
        "Batched": True,
        "UseBeta": True,
        "HighPrecisionAccumulate": True,
        "Activation": False,
    }


def _bias_sav_bbs_problem_type():
    problem_type = _plain_bbs_problem_type()
    problem_type.update(
        {
            "UseBias": 1,
            "UseScaleAlphaVec": 1,
            "Activation": True,
            "ActivationType": "hipblaslt_all",
        }
    )
    return problem_type


def _custom_config():
    return {
        "CustomKernelName": _KERNEL_NAME,
        "KernelLanguage": "Assembly",
        "MatrixInstruction": [16, 16, 16, 1],
        "ProblemType": _plain_bbs_problem_type(),
    }


def _logic_data():
    return {
        "MinimumRequiredVersion": __version__,
        "ScheduleName": "gfx950",
        "ArchitectureName": "gfx950",
        "ProblemType": _bias_sav_bbs_problem_type(),
        "Solutions": [
            {
                "SolutionIndex": 0,
                "CustomKernelName": _KERNEL_NAME,
            }
        ],
        "LibraryType": "FreeSize",
    }


def _mismatch_fields(message):
    return set(re.findall(r"field=([^ ]+)", message))


@pytest.mark.parametrize(
    "check",
    [
        Check(OnlyCustomKernels=False, All=True),
        Check(OnlyCustomKernels=True, All=False),
    ],
)
def test_tensile_logic_and_library_io_reject_same_bias_sav_regression(
    check, tmp_path, monkeypatch, capsys
):
    logic_path = tmp_path / "logic"
    logic_path.mkdir()
    logic_file = logic_path / "regression.yaml"
    logic_file.write_text("CustomKernelName: regression\n")

    monkeypatch.setattr(
        "Tensile.TensileLogic.Run._validateChipId",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "Tensile.TensileLogic.Run.readYAML",
        lambda *args, **kwargs: deepcopy(_logic_data()),
    )
    monkeypatch.setattr(
        "Tensile.TensileLogic.Run.hasCustomKernel",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "Tensile.TensileLogic.HandleCustomKernel.getCustomKernelConfig",
        lambda *args, **kwargs: deepcopy(_custom_config()),
    )

    result = _runChecks(
        logic_path,
        {},
        check,
        frozenset(),
        [logic_file],
    )
    run_message = capsys.readouterr().out

    monkeypatch.setattr(
        LibraryIO,
        "getCustomKernelConfig",
        lambda *args, **kwargs: deepcopy(_custom_config()),
    )
    with pytest.raises(ValueError) as exc_info:
        LibraryIO.parseLibraryLogicData(
            deepcopy(_logic_data()),
            str(logic_file),
            None,
            False,
            False,
            False,
            {},
            False,
        )
    library_io_message = str(exc_info.value)

    expected_fields = {
        "Activation",
        "UseBias",
        "UseScaleAlphaVec",
    }
    assert result == (0, 1, 0, 0, 0)
    assert _mismatch_fields(run_message) == expected_fields
    assert _mismatch_fields(library_io_message) == expected_fields
    assert f"logic path={logic_file}" in run_message
    assert f"logic path={logic_file}" in library_io_message
    assert "solution index=0" in run_message
    assert "solution index=0" in library_io_message
    assert f"custom kernel={_KERNEL_NAME}" in run_message
    assert f"custom kernel={_KERNEL_NAME}" in library_io_message


def test_library_io_uses_normalized_problem_type_and_preserves_type_collector(
    monkeypatch
):
    data = _logic_data()
    data["ProblemType"] = _plain_bbs_problem_type()
    data["ProblemType"]["Batched"] = 1
    custom_config = _custom_config()
    custom_config["ProblemType"]["Batched"] = True
    seen_library_problem_types = []
    original_compare = LibraryIO.compareCustomKernelProblemTypes

    def compare_with_observation(library_problem_type, kernel_problem_type):
        seen_library_problem_types.append(library_problem_type)
        return original_compare(library_problem_type, kernel_problem_type)

    monkeypatch.setattr(
        LibraryIO,
        "compareCustomKernelProblemTypes",
        compare_with_observation,
    )
    monkeypatch.setattr(
        LibraryIO,
        "getCustomKernelConfig",
        lambda *args, **kwargs: deepcopy(custom_config),
    )
    monkeypatch.setattr(
        LibraryIO,
        "Solution",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        LibraryIO.SolutionLibrary.MasterSolutionLibrary,
        "FromOriginalState",
        lambda *args, **kwargs: (object(), None),
    )

    logic = LibraryIO.parseLibraryLogicData(
        data,
        "normalized-type-mismatch.yaml",
        None,
        False,
        False,
        False,
        {},
        False,
    )

    assert len(seen_library_problem_types) == 1
    assert isinstance(seen_library_problem_types[0], ProblemType)
    assert ("Batched", "int", "bool") in logic.typeMismatches
