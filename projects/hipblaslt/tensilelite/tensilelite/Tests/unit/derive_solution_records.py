# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Writes the derived states in test_data/solution_records/<name>.yaml.

This is not a test module. A derived state is the map the library record is
built from: what Solution.FromOriginalState reads and what solutionFromState
reads in C++. Each state comes from the Python pipeline, not from a hand-built
dict: a solution of a checked-in library logic file or of a benchmark config
runs through the same derivation the library build uses, and the state is
written the way LibraryIO.writeSolutions writes it.

    python tensilelite/Tests/unit/derive_solution_records.py [name ...]

then regenerate the record, predicate and msgpack files:

    python tensilelite/Tests/unit/test_solution_state_record.py

The three *-plain states are not made here; they are the custom.config blocks
of three generated kernels.

A few fixtures change one field of a derived state, because no checked-in
logic file or config reaches that field on a real kernel (an MX free-block
count other than 1, the occupancy probe, ReuseAcrossPersistent). The edit is
listed next to the source, and test_solution_state_record.py builds both
records from the edited state, so the two builders see the same input.
"""

import copy
import sys
from pathlib import Path

import yaml

_UNIT = Path(__file__).resolve().parent
_CODEGEN = _UNIT / "characterization" / "_codegen"
for _path in (_UNIT.parent, _UNIT, _CODEGEN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_LOGIC = (
    _UNIT.parents[3]
    / "library"
    / "src"
    / "amd_detail"
    / "rocblaslt"
    / "src"
    / "Tensile"
    / "Logic"
    / "asm_full"
)


def records_dir():
    return _UNIT / "test_data" / "solution_records"


def _serialize(solution):
    """One solution as LibraryIO.writeSolutions writes it."""
    from tensilelite.SolutionStructs.Problem import problemTypeToEnum

    state = copy.copy(solution.getAttributes())
    state["ProblemType"] = copy.deepcopy(state["ProblemType"].state)
    problemTypeToEnum(state["ProblemType"])
    isa = state["ISA"]
    state["ISA"] = [isa[0], isa[1], isa[2]]
    return copy.deepcopy(state)


def _logic_solution(relative, arch, pick):
    """The first solution of a logic file, after derivation, that `pick` accepts."""
    from config_harness import _isolated_globals_with_isa, _toolchain_for
    from tensilelite import LibraryIO

    path = _LOGIC / relative
    data = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
    if isinstance(data, list):
        data = LibraryIO.parseLibraryLogicList(data, str(path))
    assembler, isa_info_map = _toolchain_for(arch)
    with _isolated_globals_with_isa(isa_info_map):
        logic = LibraryIO.parseLibraryLogicData(
            data, str(path), assembler, False, False, False, isa_info_map, False
        )
    matches = [solution for solution in logic.solutions if pick(solution.getAttributes())]
    if not matches:
        raise RuntimeError(f"no solution of {relative} matches")
    return _serialize(matches[0])


def _config_solution(config, arch, index):
    from config_harness import solutions_from_config

    solutions = solutions_from_config(config, arch=arch, limit_solutions=index + 1)
    return _serialize(solutions[index])


def _tile(mt0, mt1, depth_u, grvw_a, grvw_b):
    return lambda s: (
        s.get("MacroTile0"),
        s.get("MacroTile1"),
        s.get("DepthU"),
        s.get("GlobalReadVectorWidthA"),
        s.get("GlobalReadVectorWidthB"),
    ) == (mt0, mt1, depth_u, grvw_a, grvw_b)


def _mx_free_blocks(state):
    state["ProblemType"]["MXBlockFreeA"] = 7
    state["ProblemType"]["MXBlockFreeB"] = 4
    return state


def _rap_and_probe(state):
    state["ReuseAcrossPersistent"] = 1
    state["_RAPNumResidentKTiles"] = 3
    state["InternalSupportParams"]["SupportOccupancyProbe"] = True
    return state


# name -> (build state, edit applied to the state or None)
_SOURCES = {
    # fp8 x bf8 StreamK on gfx942: a 256x224 tile, DepthU 32, GlobalReadVectorWidth 8 for A
    # and 4 for B. A swap of the two operands or of M and N changes the predicates.
    "gfx942-streamk-fp8": (
        lambda: _logic_solution(
            "aquavanjaram/gfx942/StreamK/gfx942_Cijk_Alik_Bjlk_F8B8NSS_BH_UserArgs.yaml",
            "gfx942",
            _tile(256, 224, 32, 8, 4),
        ),
        None,
    ),
    # fp8 MX on gfx950: MXBlockA/B 32, HostPreSwizzle scale format.
    "gfx950-mx": (
        lambda: _logic_solution(
            "gfx950/gfx950/GridBased/gfx950_Cijk_Alik_Bljk_F8SS_MXAE8B32_MXBE8B32_BH_Bias_GG_HA_S_SAB_SAV_UserArgs.yaml",
            "gfx950",
            lambda s: True,
        ),
        None,
    ),
    # The same kernel with MX free-block counts that differ per operand.
    "gfx950-mx-free-blocks": (
        lambda: _logic_solution(
            "gfx950/gfx950/GridBased/gfx950_Cijk_Alik_Bljk_F8SS_MXAE8B32_MXBE8B32_BH_Bias_GG_HA_S_SAB_SAV_UserArgs.yaml",
            "gfx950",
            lambda s: True,
        ),
        _mx_free_blocks,
    ),
    # int4 A with three scale group sizes and the UnsignedBias8 encoding on gfx1151:
    # a 64x160 tile, DepthU 64.
    "gfx1151-int4": (
        lambda: _logic_solution(
            "gfx1151/FreeSize/gfx1151_Cijk_Alik_Bljk_I4B_BBS_BH_SABBGU8_UserArgs.yaml",
            "gfx1151",
            lambda s: True,
        ),
        None,
    ),
    # fp8 with stochastic rounding on gfx942: a 16x16 tile, GlobalReadVectorWidth 4 and 8.
    "gfx942-stochastic-rounding": (
        lambda: _logic_solution(
            "aquavanjaram/gfx942/GridBased/aquavanjaram_Cijk_Ailk_Bljk_HH_F8NHS_BH_BiasSH_HAS_SAV_UserArgs.yaml",
            "gfx942",
            _tile(16, 16, 32, 4, 8),
        ),
        None,
    ),
    # StreamK with a spatial cluster on gfx1250: PersistentLoopArgsVersion 2.
    "gfx1250-streamk-cluster": (
        lambda: _config_solution(
            "tensilelite/Tests/common/streamk/gfx1250/static_streamk_cluster_alpha0.yaml",
            "gfx1250",
            0,
        ),
        None,
    ),
    # The cluster kernel with ReuseAcrossPersistent and the occupancy probe set.
    "gfx1250-streamk-cluster-rap-probe": (
        lambda: _config_solution(
            "tensilelite/Tests/common/streamk/gfx1250/static_streamk_cluster_alpha0.yaml",
            "gfx1250",
            0,
        ),
        _rap_and_probe,
    ),
}


def derived_sources():
    return sorted(_SOURCES)


def derive(name):
    build, edit = _SOURCES[name]
    state = build()
    return edit(state) if edit else state


def main(names):
    for name in names or derived_sources():
        state = derive(name)
        path = records_dir() / f"{name}.yaml"
        path.write_text(yaml.safe_dump(state, sort_keys=True))
        print(name, len(state))


if __name__ == "__main__":
    main(sys.argv[1:])
