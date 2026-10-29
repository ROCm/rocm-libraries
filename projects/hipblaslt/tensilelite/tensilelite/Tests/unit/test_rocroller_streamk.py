# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocRoller Stream-K launch values match the kernel's own expression trees.

rocRoller serialises, into every Stream-K kernel's .amdgpu_metadata, the
expression tree it would have evaluated on the host to produce each hoisted
kernarg. Those trees are the contract: the kernel prologue recomputes
nothing, so a host value that disagrees with its tree silently corrupts the
Stream-K fixup pass and leaves the flag region dirty, which the next launch
on that stream then spins on.

RocRollerStreamK.evaluate_streamk is a closed form of those trees, because
the C++ launcher cannot carry a tree evaluator. This module keeps the two
honest: it evaluates the real captured trees generically and asserts the
closed form agrees, over a grid of problem shapes and grid sizes, for two
kernels with deliberately different macrotiles -- including MT_N = 192,
which is not a power of two.

tensilelite/tests/RocRollerStreamKKernargs_test.cpp checks the C++ mirror
against the same vectors; edit both together.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tensilelite.CustomKernels import _metadataArgToCustomArg, rocRollerOccupancy
from tensilelite.RocRollerStreamK import (
    SK_POINTER_ROLES,
    SK_ROLE_ORDER,
    STABLE_NAME_TO_SEMANTIC,
    classify_streamk_expression,
    evaluate_streamk,
    iters_per_tile,
)
from tensilelite.RocRollerWgm import u32

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent / "rocroller_streamk_metadata.yaml"

# Command arguments the Stream-K trees read, bound per evaluation.
_M = "Tensor_0_size_0"
_K = "Tensor_0_size_1"
_N = "Tensor_2_size_1"
_NUM_WGS = "numWGs"


# --------------------------------------------------------------------------
# Generic evaluator for the serialised trees. This is the oracle; it exists
# only here, so there is exactly one shipped implementation.
#
# Values carry a "is this already uint32" flag so truncation happens where
# rocRoller's type propagation puts it: a Convert to UInt32 truncates, and a
# binary node truncates only when both operands are uint32. The tile-count
# ceil-divides mix an Int64 tensor size with the macrotile literal and so
# stay exact until the Convert that wraps them.
# --------------------------------------------------------------------------


class _Val:
    __slots__ = ("v", "u")

    def __init__(self, v, u):
        self.v = u32(v) if u else int(v)
        self.u = u


def _binary(node, kids, op):
    lhs, rhs = kids
    wide = op(lhs.v, rhs.v)
    return _Val(wide, lhs.u and rhs.u)


_BINARY_OPS = {
    "Add": lambda a, b: a + b,
    "Subtract": lambda a, b: a - b,
    "Multiply": lambda a, b: a * b,
    "Divide": lambda a, b: a // b,
    "Modulo": lambda a, b: a % b,
    "ArithmeticShiftR": lambda a, b: a >> b,
}


def _evaluate(node, binding):
    from tensilelite.RocRollerWgm import magic_multiple_u32, magic_shifts_u32

    if not isinstance(node, dict):
        raise AssertionError(f"unexpected expression node {node!r}")
    kind = node.get("type")

    if kind == "CommandArgument":
        name = node.get("name")
        if name not in binding:
            raise AssertionError(f"unbound command argument {name!r}")
        wide = node.get("variableType", {}).get("dataType") != "UInt32"
        return _Val(binding[name], not wide)

    if kind == "Kernel Argument":
        # A kernarg that another kernarg reads. Its own tree is inlined.
        return _evaluate(node["expression"], binding)

    if isinstance(kind, str) and kind.startswith("LiteralValue"):
        return _Val(node["value"], node.get("dataType") == "UInt32")

    if kind == "Convert":
        inner = _evaluate(node["arg"], binding)
        return _Val(inner.v, node.get("dataType") == "UInt32")

    if kind in _BINARY_OPS:
        kids = (_evaluate(node["lhs"], binding), _evaluate(node["rhs"], binding))
        return _binary(node, kids, _BINARY_OPS[kind])

    if kind == "GreaterThan":
        lhs = _evaluate(node["lhs"], binding)
        rhs = _evaluate(node["rhs"], binding)
        return _Val(1 if lhs.v > rhs.v else 0, True)

    if kind == "Conditional":
        cond = _evaluate(node["lhs"], binding)
        taken = node["r1hs"] if cond.v else node["r2hs"]
        return _evaluate(taken, binding)

    if kind == "MagicMultiple":
        return _Val(magic_multiple_u32(_evaluate(node["arg"], binding).v), True)

    if kind == "MagicShifts":
        return _Val(magic_shifts_u32(_evaluate(node["arg"], binding).v), True)

    raise AssertionError(f"unhandled expression node type {kind!r}")


# --------------------------------------------------------------------------


def _fixture():
    return yaml.safe_load(_FIXTURE.read_text())


def _kernels():
    return sorted(_fixture().items())


def _kernel_ids():
    return [name for name, _ in _kernels()]


# Problem shapes chosen to exercise both arms of the tiles-vs-grid branch,
# a K that divides the macrotile exactly and one that does not, and a grid
# larger than the tile count.
_SHAPES = [
    (4096, 4096, 8192),
    (256, 256, 32768),
    (32, 4096, 16384),
    (8192, 8192, 1024),
    (1024, 2048, 4096),
    (16, 16, 128),
    (1, 1, 128),
    (4097, 4097, 8321),
    (65536, 128, 256),
]
_GRIDS = [1, 7, 224, 256, 304, 2048]


@pytest.mark.parametrize("name,kernel", _kernels(), ids=_kernel_ids())
def test_closed_form_matches_serialised_trees(name, kernel):
    """The shipped closed form reproduces every hoisted kernarg's own tree."""
    tile_m, tile_n, tile_k = kernel["macrotile"]
    trees = {
        arg[".name"]: arg[".expression"]
        for arg in kernel["args"]
        if ".expression" in arg
    }
    # Name -> semantic for the computed arguments only. The Magic* names
    # carry an unstable numeric suffix, so go through the classifier.
    semantics = {}
    for arg in kernel["args"]:
        argname = arg[".name"]
        if argname not in trees:
            continue
        semantic = STABLE_NAME_TO_SEMANTIC.get(argname)
        if semantic is None:
            semantic = classify_streamk_expression(
                trees[argname], (arg.get(".variableType") or {}).get("dataType")
            )
        if semantic in SK_ROLE_ORDER:
            semantics[argname] = semantic

    assert set(semantics.values()) == set(SK_ROLE_ORDER), (
        f"{name}: classified {sorted(semantics.values())}"
    )

    checked = 0
    for (m, n, k) in _SHAPES:
        for grid in _GRIDS:
            binding = {_M: m, _N: n, _K: k, _NUM_WGS: grid}
            closed = evaluate_streamk(m, n, k, tile_m, tile_n, tile_k, grid)
            for argname, semantic in semantics.items():
                expected = _evaluate(trees[argname], binding).v
                assert closed[semantic] == u32(expected), (
                    f"{name} {argname} ({semantic}) at "
                    f"M={m} N={n} K={k} grid={grid}: "
                    f"closed form {closed[semantic]} != tree {u32(expected)}"
                )
                checked += 1
    assert checked == len(_SHAPES) * len(_GRIDS) * len(SK_ROLE_ORDER)


@pytest.mark.parametrize("name,kernel", _kernels(), ids=_kernel_ids())
def test_every_kernarg_maps_to_a_semantic(name, kernel):
    """CustomKernels no longer raises on a Stream-K kernel's arguments."""
    seen = []
    for arg in kernel["args"]:
        mapped = _metadataArgToCustomArg(arg, name)
        assert mapped["semantic"], f"{name}: {arg['.name']} produced {mapped}"
        seen.append(mapped["semantic"])

    for role in SK_ROLE_ORDER + SK_POINTER_ROLES:
        assert role in seen, f"{name}: missing {role}"

    # The two scratch pointers must be addresses; everything Stream-K
    # computes is a uint32.
    by_semantic = {}
    for arg, mapped in zip(kernel["args"], (_metadataArgToCustomArg(a, name) for a in kernel["args"])):
        by_semantic[mapped["semantic"]] = mapped["type"]
    for role in SK_POINTER_ROLES:
        assert by_semantic[role] == "address"
    # Every computed argument is a four-byte slot. rocRoller declares the
    # shift constants Int32 even though the value is an unsigned shift
    # count, so take the declared type rather than assuming.
    for role in SK_ROLE_ORDER:
        assert by_semantic[role] in ("uint32", "int32")


def test_magic_roles_are_classified_by_divisor_not_name():
    """The same constant has a different suffix on each kernel."""
    suffixes = {}
    for name, kernel in _kernels():
        for arg in kernel["args"]:
            tree = arg.get(".expression")
            if tree is None:
                continue
            semantic = classify_streamk_expression(
                tree, (arg.get(".variableType") or {}).get("dataType")
            )
            if semantic is None:
                continue
            suffixes.setdefault(semantic, set()).add(arg[".name"])

    assert set(suffixes) == {
        "RR_SKMagicMultipleIters",
        "RR_SKMagicShiftsIters",
        "RR_SKMagicMultipleNumTiles1",
        "RR_SKMagicShiftsNumTiles1",
    }
    # If the names were stable, each role would have exactly one spelling.
    assert any(len(names) > 1 for names in suffixes.values()), (
        "fixture no longer demonstrates suffix drift; name-based mapping "
        "would now pass by accident"
    )


def test_all_streamk_when_fewer_tiles_than_workgroups():
    """tiles <= numWGs - 1 puts every iteration in the Stream-K phase."""
    # 1x1 tiles against a 224-wide grid.
    out = evaluate_streamk(16, 16, 4096, 256, 256, 128, 224)
    iters = iters_per_tile(4096, 128)
    assert out["RR_SKNumSKTiles"] == iters
    assert out["RR_SKNumDPTiles"] == 0
    assert out["RR_SKNumDPTilesPerWG"] == 0
    assert out["RR_SKNumSKTilesPerWG"] == 1


def test_grid_must_be_positive():
    with pytest.raises(ValueError):
        evaluate_streamk(4096, 4096, 8192, 16, 16, 128, 0)


def test_macrotile_must_be_positive():
    with pytest.raises(ValueError):
        evaluate_streamk(4096, 4096, 8192, 0, 16, 128, 224)
    with pytest.raises(ValueError):
        evaluate_streamk(4096, 4096, 8192, 16, 16, 0, 224)


def test_non_power_of_two_macrotile_n():
    """MT_N = 192 ships, so the tile count cannot be a shift."""
    out = evaluate_streamk(8192, 8192, 1024, 256, 192, 128, 224)
    assert out["RR_SKNumTiles1"] == (8192 + 191) // 192


@pytest.mark.parametrize("name,kernel", _kernels(), ids=_kernel_ids())
def test_occupancy_is_plausible(name, kernel):
    """Captured occupancy is a positive workgroups-per-CU count."""
    meta = {
        ".vgpr_count": kernel["vgpr_count"],
        ".agpr_count": kernel["agpr_count"],
        ".group_segment_fixed_size": kernel["group_segment_fixed_size"],
        ".max_flat_workgroup_size": kernel["max_flat_workgroup_size"],
        ".wavefront_size": kernel["wavefront_size"],
    }
    occupancy = rocRollerOccupancy(meta)
    assert 1 <= occupancy <= 32, f"{name}: occupancy {occupancy}"


def test_occupancy_unknown_without_metadata():
    assert rocRollerOccupancy({}) == 0
    assert rocRollerOccupancy(None) == 0


# (vgpr, agpr, lds, flatWorkgroupSize, expected). Measured against
# hipModuleOccupancyMaxActiveBlocksPerMultiprocessor on gfx950 over 4552
# kernels drawn from a Tensile library, the rocRoller custom-kernel code
# object, and synthetic kernels built to sweep AGPR and LDS usage; the model
# agreed on every one. These rows are the witnesses that pin each constant.
_OCCUPANCY = [
    # Binding constraint is LDS.
    (44, 0, 12480, 64, 13),
    (20, 0, 24832, 256, 6),
    (142, 0, 65536, 256, 2),
    (188, 0, 99328, 256, 1),
    (256, 0, 163840, 256, 1),
    # Granule is 8, not 4: 512/88 = 5, where granule 4 would give 6.
    (84, 0, 14336, 256, 5),
    # Granule is 8, not 16: 512/72 = 7, where granule 16 would give 6.
    (72, 0, 0, 256, 7),
    # .agpr_count must not be added. This kernel really ships in
    # rr_custom_kernels_gfx950.co; 254 + 192 would give 1, HIP says 2.
    (254, 192, 69632, 256, 2),
    (48, 32, 0, 256, 8),
    (272, 256, 0, 256, 1),
    # No LDS at all, so the wave term binds; capped at 8 waves per SIMD.
    (10, 0, 0, 1024, 2),
    (128, 0, 0, 1024, 1),
]


@pytest.mark.parametrize("vgpr,agpr,lds,flat,expected", _OCCUPANCY)
def test_occupancy_matches_hip(vgpr, agpr, lds, flat, expected):
    meta = {
        ".vgpr_count": vgpr,
        ".agpr_count": agpr,
        ".group_segment_fixed_size": lds,
        ".max_flat_workgroup_size": flat,
        ".wavefront_size": 64,
    }
    assert rocRollerOccupancy(meta) == expected
