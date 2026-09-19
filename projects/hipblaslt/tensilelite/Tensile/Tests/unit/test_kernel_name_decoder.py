# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import io
import json

import pytest

from Tensile.SolutionStructs.KernelNameDecoder import (
    KernelNameDecodeError,
    decode_kernel_name,
)
from Tensile.TensileDecodeKernelName import run


pytestmark = pytest.mark.unit


KERNEL_NAME = (
    "Cijk_Alik_Bljk_BBS_BH_Bias_SAV_UserArgs_"
    "MT96x96x128_MI16x16x1_SN_"
    "LDSB0_AFC1_AFEM1_AFEM4_ASEM1_GSUAMB_ISA950_LWPMn1_"
    "MIWT3_3_WG32_8_1"
)


def _by_name(result):
    return {parameter.name: parameter for parameter in result.parameters}


def test_decodes_current_kernel_name_with_typed_values():
    result = decode_kernel_name(KERNEL_NAME)

    assert result.complete
    assert result.problem_type == "Cijk_Alik_Bljk_BBS_BH_Bias_SAV_UserArgs"
    parameters = _by_name(result)
    assert parameters["MacroTile"].value == [96, 96, 128]
    assert parameters["MatrixInstruction"].value == [16, 16, 1]
    assert parameters["ActivationFuncCall"].value is True
    assert parameters["AssertFree0ElementMultiple"].value == 1
    assert parameters["AssertFree1ElementMultiple"].value == 4
    assert parameters["GlobalSplitUAlgorithm"].value == "MultipleBuffer"
    assert parameters["ISA"].value == [9, 5, 0]
    assert parameters["LocalWritePerMfma"].value == -1
    assert parameters["MIWaveTile"].value == [3, 3]
    assert parameters["WorkGroup"].value == [32, 8, 1]
    assert not result.warnings


def test_decodes_legacy_aliases_and_four_field_matrix_instruction():
    result = decode_kernel_name(
        "Cijk_Alik_Bljk_HHS_BH_"
        "MT64x16x32_MI16x16x16x1_SN_1LDSB0_GRVW2_VAW2"
    )

    assert result.complete
    parameters = _by_name(result)
    assert parameters["MatrixInstruction"].value == [16, 16, 16, 1]
    assert parameters["1LDSBuffer"].value == 0
    assert parameters["GlobalReadVectorWidth"].value == 2
    assert parameters["VectorAtomicWidth"].value == 2


def test_decodes_compatibility_parameters_from_shipped_logic():
    result = decode_kernel_name(
        "Cijk_Alik_Bljk_CB_UserArgs_MT128x128x4_MI32x32x2x1_SN_"
        "AF0EM1_AF1EM4_CB0_FL0_LBSPP128_LSU2_UMLDSA1_UMLDSB0_VW2"
    )

    assert result.complete
    parameters = _by_name(result)
    assert parameters["AssertFree0ElementMultiple"].value == 1
    assert parameters["AssertFree1ElementMultiple"].value == 4
    assert parameters["ClusterBarrier"].value is False
    assert parameters["FractionalLoad"].value == 0
    assert parameters["LdsBlockSizePerPad"].value == 128
    assert parameters["LocalSplitU"].value == 2
    assert parameters["UnrollMajorLDSA"].value is True
    assert parameters["UnrollMajorLDSB"].value is False
    assert parameters["VectorWidth"].value == 2


def test_single_colliding_abbreviation_is_reported_as_ambiguous():
    result = decode_kernel_name("Cij_Aik_Bjk_S_B_UserArgs_SN_AFEM1")

    assert not result.complete
    assert result.parameters[0].status == "ambiguous"
    assert result.parameters[0].candidates == (
        "AssertFree0ElementMultiple",
        "AssertFree1ElementMultiple",
    )


def test_unknown_component_is_preserved_and_strict_decode_is_incomplete():
    result = decode_kernel_name("Cij_Aik_Bjk_S_B_UserArgs_SN_FUTURE42")

    assert not result.complete
    assert result.parameters[0].component == "FUTURE42"
    assert result.parameters[0].status == "unknown"


def test_shortened_name_reports_irreversible_suffix():
    result = decode_kernel_name(
        "Cij_Aik_Bjk_S_B_UserArgs_MT128x128x16_MI16x16x1_SN_"
        "J5fN-PqZZHaG3sf4Ra41sB3oPau-xn6LxQN5r_42D8A="
    )

    assert result.truncated
    assert not result.complete
    assert any("irreversible" in warning for warning in result.warnings)


@pytest.mark.parametrize("kernel_name", ["", "   ", "name with spaces"])
def test_rejects_invalid_input(kernel_name):
    with pytest.raises(KernelNameDecodeError):
        decode_kernel_name(kernel_name)


def test_result_is_json_serializable():
    payload = decode_kernel_name(KERNEL_NAME).to_dict()

    assert json.loads(json.dumps(payload)) == payload


def test_cli_text_output():
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = run([KERNEL_NAME], stdout=stdout, stderr=stderr)

    assert status == 0
    assert "Problem type: Cijk_Alik_Bljk_BBS_BH_Bias_SAV_UserArgs" in stdout.getvalue()
    assert "GlobalSplitUAlgorithm" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_cli_json_from_stdin():
    stdout = io.StringIO()

    status = run(
        ["--format", "json"],
        stdin=io.StringIO(KERNEL_NAME + "\n"),
        stdout=stdout,
    )

    assert status == 0
    assert json.loads(stdout.getvalue())["complete"] is True


def test_cli_strict_mode_fails_for_unknown_component():
    status = run(
        ["--strict", "Cij_Aik_Bjk_S_B_UserArgs_SN_FUTURE42"],
        stdout=io.StringIO(),
    )

    assert status == 1


def test_cli_rejects_multiple_stdin_names():
    stderr = io.StringIO()

    status = run(
        [],
        stdin=io.StringIO(KERNEL_NAME + "\n" + KERNEL_NAME + "\n"),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert "exactly one" in stderr.getvalue()


def test_cli_does_not_block_on_empty_terminal_input():
    class TerminalInput(io.StringIO):
        def isatty(self):
            return True

    stderr = io.StringIO()

    status = run(
        [],
        stdin=TerminalInput(),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert status == 2
    assert "redirected on stdin" in stderr.getvalue()
