# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from Tensile.Common.DataType import DataType
from Tensile.KernelWriterConversion import KernelWriterConversion


def _half_argument_writer():
    writer = object.__new__(KernelWriterConversion)
    writer.kernelName = "PostGSU2"
    writer.language = "HIP"
    writer.endLine = "\n"
    writer.indexChars = ["0I", "1J", "K"]
    writer.datatype = "tensile_half"
    writer.actGradientPrefix = ""
    writer.state = {"ProblemType": {
        "DestDataType": DataType("half"),
        "ComputeDataType": DataType("half"),
        "StridedBatched": True,
        "UseE": False,
        "UseBias": 0,
        "Gradient": False,
        "BiasSrc": "D",
        "UseScaleAB": False,
        "UseScaleCD": False,
        "UseScaleAlphaVec": False,
        "UseGateResidual": False,
        "ActivationComputeDataType": DataType("half"),
        "ActivationType": "none",
        "UseInitialStridesCD": False,
        "NumIndicesC": 3,
    }, "ActivationFused": False}
    return writer


def test_half_alpha_beta_match_kernel_arguments_dword_abi():
    arguments = _half_argument_writer().functionArgument()

    assert "tensile_half2 alpha;" in arguments
    assert "tensile_half2 beta;" in arguments
    assert "tensile_half alpha;" not in arguments
    assert "tensile_half beta;" not in arguments


def test_scalar_half_load_uses_scalar_type_and_one_load_lane():
    writer = object.__new__(KernelWriterConversion)
    writer.state = {"ProblemType": {"DataType": DataType("half")}}
    writer.datatype = DataType("half").toDevice("HIP")

    writer.num_dword_load, writer.is_sub_dword_load = writer._loadWidth(
        1, DataType("half"))

    assert writer.num_dword_load == 1
    assert writer.is_sub_dword_load
    assert writer._loadType("float") == writer.datatype


def _stride_writer(use_initial_strides, use_e=False):
    writer = object.__new__(KernelWriterConversion)
    writer.state = {"ProblemType": {
        "UseE": use_e,
        "UseInitialStridesCD": use_initial_strides,
    }}
    writer.indexChars = ["0I"]
    writer.endLine = "\n"
    return writer


def test_conversion_uses_supplied_initial_strides():
    writer = _stride_writer(True, use_e=True)
    defines = writer._initialStrideDefines()
    undefines = writer._initialStrideUndefines()

    for tensor in ("E", "D", "W", "C"):
        assert f"#define stride{tensor}0I arg.stride{tensor}0I\n" in defines
        assert f"#undef stride{tensor}0I\n" in undefines
    assert "hard-coded initial strides" not in defines


def test_conversion_defaults_to_unit_initial_strides():
    defines = _stride_writer(False)._initialStrideDefines()

    for tensor in ("D", "W", "C"):
        assert f"#define stride{tensor}0I 1\n" in defines
    assert "strideE0I" not in defines
