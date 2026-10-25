# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Naming import (
    getKernelCompileKey,
    getKernelNameMin,
)
from tensilelite.TensileCreateLibrary.Run import (
    groupKernelRecords,
    passPostKernelInfoToSolution,
)


class _Solution:
    def __init__(self, kernel):
        self._kernel = kernel
        self._state = {}

    def getKernels(self):
        return [self._kernel]


def _kernel():
    kernel = deepcopy(defaultSolution)
    kernel["ProblemType"] = {
        "OperationIdentifier": "Cijk_Ailk_Bjlk",
        "DataType": 0,
        "DestDataType": 0,
        "ComputeDataType": 0,
        "GroupedGemm": False,
        "UseBeta": False,
        "UseBias": 0,
    }
    kernel.update(
        {
            "KernelLanguage": "Assembly",
            "MacroTile0": 64,
            "MacroTile1": 32,
            "DepthU": 256,
            "MatrixInstM": 16,
            "MatrixInstN": 16,
            "MatrixInstB": 1,
            "MatrixInstruction": [16, 16, 1, 1],
            "MIWaveTile": [2, 2],
            "WorkGroup": [32, 4, 2],
            "ISA": (9, 5, 0),
            "TileProcessingStrategy": "StreamK",
            "WorkAssignment": "StaticGrid",
        }
    )
    return kernel


@pytest.mark.parametrize(
    "parameter,value,tag",
    [
        ("DebugStreamK", 1, "DSK1"),
        ("StreamKAtomic", 1, "SKA1"),
        ("MbskPrefetchMethod", 1, "MPM1"),
        ("DebugPersistentKernelLoopForever", True, "DPKLF1"),
    ],
)
def test_code_changing_modes_have_distinct_names(parameter, value, tag):
    baseline = _kernel()
    variant = _kernel()
    variant[parameter] = value

    assert getKernelNameMin(baseline, False) != getKernelNameMin(variant, False)
    assert getKernelCompileKey(baseline, False) != getKernelCompileKey(variant, False)
    assert tag in getKernelNameMin(variant, False)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("DebugStreamK", 1),
        ("StreamKAtomic", 1),
        ("MbskPrefetchMethod", 1),
        ("DebugPersistentKernelLoopForever", True),
    ],
)
def test_code_changing_modes_form_two_compile_groups(parameter, value):
    baseline = _kernel()
    variant = _kernel()
    variant[parameter] = value

    groups = groupKernelRecords([baseline, variant], False)

    assert len(groups) == 2
    assert groups[0].baseName != groups[1].baseName
    assert all(len(group.aliases) == 1 for group in groups)


def test_runtime_internal_args_share_one_compile_group():
    first = _kernel()
    second = _kernel()
    second["WorkGroupMapping"] = 8
    second["StaggerU"] = 32

    groups = groupKernelRecords([first, second], False)

    assert getKernelCompileKey(first, False) == getKernelCompileKey(second, False)
    assert len(groups) == 1
    assert groups[0].aliases == (first, second)


@pytest.mark.parametrize(
    "parameter,first,second",
    [
        ("NoReject", False, True),
        ("AssertAIGreaterThanEqual", -1, 4),
        # Shipped gfx90a Equality/GridBased records retain different legacy
        # ThreadTile values, while matrix instructions use MIWaveTile instead.
        ("ThreadTile", [1, 1], [1, 16]),
    ],
)
def test_validation_only_variants_share_one_compile_group(
    parameter, first, second
):
    kernels = [_kernel(), _kernel()]
    kernels[0][parameter] = first
    kernels[1][parameter] = second

    groups = groupKernelRecords(kernels, False)

    assert len(groups) == 1
    assert groups[0].aliases == tuple(kernels)


def test_wgmxcc_names_follow_codegen_categories():
    ordinary = [_kernel(), _kernel()]
    ordinary[0]["WorkGroupMappingXCC"] = 1
    ordinary[1]["WorkGroupMappingXCC"] = 8
    assert getKernelNameMin(ordinary[0], False) == getKernelNameMin(ordinary[1], False)

    skxcc = [_kernel(), _kernel()]
    for kernel, value in zip(skxcc, (1, 8)):
        kernel["WorkAssignment"] = "DynamicWorkQueue"
        kernel["PersistentXCCMapping"] = 1
        kernel["WorkGroupMappingXCC"] = value
    assert getKernelNameMin(skxcc[0], False) != getKernelNameMin(skxcc[1], False)
    assert getKernelCompileKey(skxcc[0], False) != getKernelCompileKey(skxcc[1], False)
    # WorkAssignment.usesRawQueueRank tests > 1; the exact fixed count is
    # supplied at runtime, so 2 and 8 follow the same generated branch.
    another_fixed = deepcopy(skxcc[1])
    another_fixed["WorkGroupMappingXCC"] = 2
    assert len(groupKernelRecords([skxcc[1], another_fixed], False)) == 1


def test_distinct_compile_keys_cannot_share_an_artifact(monkeypatch):
    import tensilelite.TensileCreateLibrary.Run as run

    kernels = [_kernel(), _kernel()]
    kernels[1]["ISA"] = (9, 4, 2)
    keys = iter(("first", "second"))
    monkeypatch.setattr(run, "getKernelCompileKey", lambda kernel, split: next(keys))
    monkeypatch.setattr(run, "getKernelNameMin", lambda kernel, split: "same")
    monkeypatch.setattr(run, "getKernelFileBase", lambda split, kernel: "same")

    with pytest.raises(RuntimeError, match="Distinct kernel compile keys"):
        run.groupKernelRecords(kernels, False)


def _register_allocation_warnings(capsys):
    return [
        line
        for line in capsys.readouterr().out.splitlines()
        if "StinkyTofuRegisterAllocation" in line
    ]


def test_register_allocation_configs_with_one_mode_share_a_kernel(capsys):
    # The name records only the mode, so these are one kernel, built with the
    # first config. classes=vs is the default, and the two spellings of the
    # scalar config are one warning.
    kernels = [_kernel(), _kernel(), _kernel()]
    kernels[0]["StinkyTofuRegisterAllocation"] = "2,classes=vs"
    kernels[1]["StinkyTofuRegisterAllocation"] = "2,classes=s"
    kernels[2]["StinkyTofuRegisterAllocation"] = "apply,classes=s"

    [group] = groupKernelRecords(kernels, False)

    assert group.representative is kernels[0]
    assert group.aliases == tuple(kernels)
    [warning] = _register_allocation_warnings(capsys)
    assert "'2,classes=s' is not built for 2 kernel(s)" in warning
    assert "an earlier kernel with 2," in warning


def test_spellings_of_one_register_allocation_config_are_not_merges(capsys):
    kernels = [_kernel(), _kernel()]
    kernels[0]["StinkyTofuRegisterAllocation"] = "apply"
    kernels[1]["StinkyTofuRegisterAllocation"] = 2

    assert len(groupKernelRecords(kernels, False)) == 1
    assert _register_allocation_warnings(capsys) == []


def test_metadata_fans_out_to_all_placement_aliases():
    kernels = [_kernel(), _kernel()]
    kernels[0]["codeObjectFile"] = "first"
    kernels[1]["codeObjectFile"] = "second"
    groups = groupKernelRecords(kernels, False)
    solutions = [_Solution(kernel) for kernel in kernels]
    result = SimpleNamespace(
        cuoccupancy=3,
        pgr=2,
        mathclk=4,
        customKernelDef=None,
    )

    passPostKernelInfoToSolution(
        [result], [groups[0].representative], solutions, False
    )

    assert [solution._state["CUOccupancy"] for solution in solutions] == [3, 3]
    assert [solution._state["PrefetchGlobalRead"] for solution in solutions] == [2, 2]


def test_thread_tile_distinguishes_non_matrix_kernels():
    kernels = [_kernel(), _kernel()]
    for kernel, tile in zip(kernels, ([1, 1], [1, 16])):
        del kernel["MatrixInstM"]
        kernel["ThreadTile"] = tile
    assert len(groupKernelRecords(kernels, False)) == 2


def test_collection_preserves_all_placement_aliases():
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions

    kernels = [_kernel(), _kernel()]
    kernels[1]["codeObjectFile"] = "other"
    collected = generateKernelObjectsFromSolutions([_Solution(k) for k in kernels])
    groups = groupKernelRecords(collected, False)
    assert len(groups) == 1
    assert groups[0].aliases == tuple(kernels)
