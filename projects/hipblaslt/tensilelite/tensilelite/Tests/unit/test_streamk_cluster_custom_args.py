# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The external clustered StreamK descriptor has one explicit scheduling ABI."""

from copy import deepcopy

import pytest

from Tensile.CustomKernels import validateCustomPersistentArgs

pytestmark = pytest.mark.unit


def config():
    return {
        "TileProcessingStrategy": "StreamK",
        "WorkAssignment": "StaticGrid",
        "ClusterDim": [4, 2],
        "InternalSupportParams": {"KernArgsVersion": 3, "PersistentLoopArgsVersion": 2},
        "CustomKernel": {
            "name": "external_cluster",
            "args": [{"type": "address", "semantic": "AddressFlags"}]
            + [{"type": "uint32", "semantic": semantic} for semantic in (
                "ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                "SKItersPerWG", "SKGrid", "SKTilesAndSplit")]
            + [{"type": "address", "semantic": "AddressWorkspace"}],
        },
    }


@pytest.mark.parametrize("workspace", ["None", "StreamKWithReduction"])
def test_accepts_explicit_or_inferred_workspace_without_reordering(workspace):
    state = config()
    state["CustomKernel"]["workspaceType"] = workspace
    before = deepcopy(state)
    validateCustomPersistentArgs(state)
    assert state == before


@pytest.mark.parametrize("mutation", ["wide", "padding", "separated", "reverse", "duplicate",
    "missing_grid", "missing_flags", "duplicate_flags", "wide_flags", "legacy_total",
    "dp_grid", "workspace", "old_outer", "missing_descriptor"])
def test_rejects_incompatible_scheduling_payload(mutation):
    state = config()
    args = state["CustomKernel"]["args"]
    if mutation == "wide":
        args[3]["type"] = "uint64"
    elif mutation == "padding":
        args[3]["padding"] = 4
    elif mutation == "separated":
        args.insert(3, {"type": "float32", "semantic": "Alpha"})
    elif mutation == "reverse":
        args[2], args[3] = args[3], args[2]
    elif mutation == "duplicate":
        args.append(deepcopy(args[2]))
    elif mutation == "missing_grid":
        del args[5]
    elif mutation == "missing_flags":
        del args[0]
    elif mutation == "duplicate_flags":
        args.append(deepcopy(args[0]))
    elif mutation == "wide_flags":
        args[0]["type"] = "uint64"
    elif mutation == "legacy_total":
        args.insert(4, {"type": "uint32", "semantic": "TotalIters"})
    elif mutation == "dp_grid":
        args[5]["semantic"] = "PersistentGrid"
    elif mutation == "workspace":
        state["CustomKernel"]["workspaceType"] = "GSU"
    elif mutation == "old_outer":
        state["InternalSupportParams"]["KernArgsVersion"] = 2
    elif mutation == "missing_descriptor":
        del state["CustomKernel"]
        state["CustomKernelName"] = "external_cluster"
    with pytest.raises(ValueError):
        validateCustomPersistentArgs(state)
