# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch bounds and argument ABI for the specialized decode kernel."""
from pathlib import Path

import pytest
import yaml

from Tensile.Common.Utilities import state
from Tensile.Contractions import ProblemPredicate
from Tensile.CustomKernels import getCustomKernelConfigAndAssembly, readCustomKernelConfig

pytestmark = pytest.mark.unit

NAME = "Custom_W4A16_Decode_G128_ExLlama_gfx1151"
GENERAL = "Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB128ZPU8X_UserArgs_MT64x160x64_MI16x16x1_gfx1151"
DIRECTORY = Path(__file__).parents[2] / "CustomKernels"


def test_decode_selection_bounds():
    config = readCustomKernelConfig(NAME, DIRECTORY)
    equal = ProblemPredicate.FromOriginalKeyPair(("AssertSizeEqual", config["AssertSizeEqual"]))
    positive_k = ProblemPredicate.FromOriginalKeyPair(
        ("AssertSizeGreaterThan", config["AssertSizeGreaterThan"]))
    assert state(equal) == {
        "type": "And",
        "value": [
            {"type": "SizeEqual", "index": 1, "value": 1},
            {"type": "SizeEqual", "index": 2, "value": 1},
        ],
    }
    assert state(positive_k) == {"type": "SizeGreaterThan", "index": 3, "value": 0}
    assert config["AssertSummationElementMultiple"] == 256
    support = config["InternalSupportParams"]
    assert not support["SupportUserGSU"]
    assert not support["SupportCustomWGM"]
    assert not support["SupportCustomStaggerU"]


def test_decode_universal_arguments_match_matrix_kernel():
    def metadata(name):
        config, _ = getCustomKernelConfigAndAssembly(name, DIRECTORY)
        return yaml.safe_load(config)["amdhsa.kernels"][0]

    decode, general = metadata(NAME), metadata(GENERAL)
    # HIP compilation must preserve the universal layout that the existing
    # host library supplies, including unused fields and trailing offsets.
    for key in (".kernarg_segment_size", ".kernarg_segment_align"):
        assert decode[key] == general[key]
    assert [(arg[".offset"], arg[".size"]) for arg in decode[".args"]] == [
        (arg[".offset"], arg[".size"]) for arg in general[".args"]
    ]
    config = readCustomKernelConfig(NAME, DIRECTORY)
    x, y, z = config["WorkGroup"]
    assert x * y * z == decode[".max_flat_workgroup_size"]
