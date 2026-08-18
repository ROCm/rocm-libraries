# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct-lane coverage for the custom-kernel branch of
``LibraryIO.parseLibraryLogicData``.

The heavy parse path (live ``Solution`` construction) is driven against the
vendored production logic file used by the LibraryIO characterization suite,
with a real ``amdclang++`` assembler + per-ISA capability map. Unlike the
characterization copy, these live in a direct ``Tests/unit`` module so the
coverage lane credits the changed lines, and they exercise the *new* branches:
the ``CustomKernel`` mapping-name detection and the skip-on-missing-config path
(which returns ``None`` and drops the solution).
"""

import copy
from pathlib import Path
from typing import List

import pytest

import Tensile.LibraryIO as L
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

pytestmark = pytest.mark.unit


_FIXTURE = (Path(__file__).parent / "characterization" / "LibraryIO"
            / "data" / "logic_gfx942_HSS_BH.yaml")


@pytest.fixture(scope="module")
def cxx_compiler():
    return validateToolchain("amdclang++")


@pytest.fixture(scope="module")
def isa_info_map(cxx_compiler):
    return makeIsaInfoMap(SUPPORTED_ISA, cxx_compiler)


@pytest.fixture(scope="module")
def assembler(cxx_compiler):
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx_compiler, bundler, "default").assembler


def _raw_dict():
    data = L.read(str(_FIXTURE), True)
    assert isinstance(data, List)
    return L.parseLibraryLogicList(copy.deepcopy(data), str(_FIXTURE))


def test_custom_kernel_legacy_name_is_resolved(assembler, isa_info_map, monkeypatch):
    # Legacy flat CustomKernelName + an (empty) config -> the merge branch runs
    # and the solution constructs normally (no handwritten-kernel path).
    seen = []
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp: seen.append(name) or {})
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_legacy_kernel"
    # Present InternalSupportParams -> the isp-extraction branch also runs.
    data["Solutions"][0]["InternalSupportParams"] = {"KernelLanguage": "Assembly"}

    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )

    assert "synthetic_legacy_kernel" in seen
    assert len(logic.solutions) >= 1


def test_custom_kernel_mapping_name_skipped_on_missing_config(
        assembler, isa_info_map, monkeypatch, capsys):
    # A solution using the new CustomKernel: { name: ... } mapping form whose
    # config can't be read is dropped (returns None), emitting a per-kernel and
    # an aggregate "Skipped N" warning. Appending (rather than replacing) keeps
    # the vendored fixture's solution-index table intact so only the new,
    # dropped kernel is exercised.
    def _raise(name, isp):
        raise RuntimeError("no custom.config")

    monkeypatch.setattr(L, "getCustomKernelConfig", _raise)
    data = _raw_dict()
    total = len(data["Solutions"])
    data["Solutions"].append(
        {"KernelLanguage": "Assembly", "CustomKernel": {"name": "map_broken_kernel"}}
    )

    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )

    out = capsys.readouterr().out
    assert "map_broken_kernel" in out              # mapping-name resolved
    assert "Skipped 1 solution(s)" in out          # aggregate skip warning
    assert len(logic.solutions) == total           # only the appended kernel dropped


def test_custom_kernel_bad_matrix_instruction_raises(assembler, isa_info_map, monkeypatch):
    monkeypatch.setattr(
        L, "getCustomKernelConfig", lambda name, isp: {"MatrixInstruction": [1, 2, 3]}
    )
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "bad_mi_kernel"

    with pytest.raises(ValueError):
        L.parseLibraryLogicData(
            data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
        )
