# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every checked-in rocRoller Stream-K kernel launches as one.

A build without rocRoller treats a checked-in kernel as Stream-K when its
``CustomKernel`` block declares ``workspaceType: RocRollerStreamK``: that is
what the rocRoller-off walk matches against a ranked Stream-K tile, what sizes
the partials and flags workspace, and what sizes the grid with
``RocRollerStreamK``. The kernel name says ``_SK_WGT_`` for the same thing.
If the two disagreed, the walk would serve a Stream-K kernel for a
data-parallel tile (or the reverse) and launch it with the wrong grid and no
workspace.

A Stream-K kernel also must not take a workgroup-mapping kernarg (rocRoller
turns workgroup mapping off for Stream-K), must take the kernargs that tell it
how many tiles are Stream-K and how many data-parallel (the launch refuses
grids that leave data-parallel tiles), must declare its occupancy (the grid
is sized from it), and must not declare uniform summation order (its partial
sums are added in arrival order).
"""

from pathlib import Path

import pytest
import yaml

from logic_corpus import needs_logic_corpus
from rocroller_logic_corpus import corpus, rocroller_logic_files
from tensilelite.CustomKernels import readCustomKernelConfig

pytestmark = pytest.mark.unit

RR_KERNEL_PREFIX = "RR_GEMM_"
STREAMK = "RocRollerStreamK"
STREAMK_NAME = "_SK_WGT_"
STREAMK_ARGS = {
    "RR_SKNumWGs",
    "RR_SKNumTilesN",
    "RR_SKNumSKTiles",
    "RR_SKNumSKTilesPerWG",
    "RR_SKNumDPTiles",
    "RR_SKNumDPTilesPerWG",
    "RR_SKAddressPartials",
    "RR_SKAddressFlags",
}
# The fields the launch reads from the CustomKernel block.
LAUNCH_FIELDS = ("grid", "workspaceType", "occupancy", "macrotile")

_KERNELS = Path(__file__).resolve().parents[2] / "CustomKernels" / "rocroller"


@pytest.fixture(scope="module")
def shipped():
    """CustomKernel block of every shipped rocRoller kernel, from its custom.config."""
    kernels = {}
    for path in sorted(_KERNELS.glob(RR_KERNEL_PREFIX + "*.s")):
        kernels[path.stem] = readCustomKernelConfig(path.stem, str(_KERNELS))["CustomKernel"]
    assert kernels, f"no {RR_KERNEL_PREFIX}*.s kernels under {_KERNELS}"
    return kernels


def _is_streamk(ck):
    return ck.get("workspaceType") == STREAMK


def _semantics(ck):
    return {arg["semantic"] for arg in ck.get("args", [])}


def test_streamk_name_workspace_and_grid_agree(shipped):
    disagree = []
    for name, ck in shipped.items():
        by_name = STREAMK_NAME in name
        by_grid = (ck.get("grid") or [None])[0] == STREAMK
        if not by_name == by_grid == _is_streamk(ck):
            disagree.append(
                f"{name}: name {by_name}, grid {ck.get('grid')}, workspaceType {ck.get('workspaceType')}"
            )
    assert not disagree, f"Stream-K name, grid and workspace disagree: {disagree}"


def test_streamk_kernels_take_the_streamk_kernargs(shipped):
    streamk = {name: ck for name, ck in shipped.items() if _is_streamk(ck)}
    assert streamk, f"no {STREAMK} kernels under {_KERNELS}"
    missing = {name: sorted(STREAMK_ARGS - _semantics(ck)) for name, ck in streamk.items()}
    missing = {name: args for name, args in missing.items() if args}
    assert not missing, f"Stream-K kernels lack Stream-K kernargs: {missing}"
    mapped = sorted(name for name, ck in streamk.items() if "RR_WorkgroupMapping" in _semantics(ck))
    assert not mapped, f"Stream-K kernels take a workgroup-mapping kernarg: {mapped}"


def test_only_streamk_kernels_take_streamk_kernargs(shipped):
    stray = sorted(
        name
        for name, ck in shipped.items()
        if not _is_streamk(ck) and _semantics(ck) & STREAMK_ARGS
    )
    assert not stray, f"data-parallel kernels take Stream-K kernargs: {stray}"


def test_streamk_kernels_declare_occupancy(shipped):
    bad = sorted(
        f"{name}: {ck.get('occupancy')!r}"
        for name, ck in shipped.items()
        if _is_streamk(ck)
        and not (isinstance(ck.get("occupancy"), int) and ck["occupancy"] > 0)
    )
    assert not bad, f"Stream-K kernels without a positive occupancy: {bad}"


def test_no_streamk_kernel_declares_uniform_summation_order(shipped):
    flagged = sorted(
        name for name, ck in shipped.items() if _is_streamk(ck) and ck.get("uniformSummationOrder")
    )
    assert not flagged, f"Stream-K kernels declare uniformSummationOrder: {flagged}"


@needs_logic_corpus
def test_logic_launch_fields_agree_with_custom_config(shipped):
    disagree, seen = [], set()
    for path in rocroller_logic_files():
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        for solution in doc["Solutions"]:
            ck = solution.get("CustomKernel") or {}
            name = ck.get("name", "")
            if not name.startswith(RR_KERNEL_PREFIX) or not _is_streamk(shipped.get(name, {})):
                continue
            seen.add(name)
            for field in LAUNCH_FIELDS:
                if ck.get(field) != shipped[name].get(field):
                    disagree.append(f"{path.name}: {name}.{field}")
    assert seen, f"no {STREAMK} solutions under {corpus()}"
    assert not disagree, f"logic CustomKernel launch fields differ from custom.config: {disagree}"
