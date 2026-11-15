# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CompactSolution: the parent-side stand-in for a parsed Solution."""

import copy
import pickle
from pathlib import Path

import pytest

from tensilelite import LibraryIO
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Common.GlobalParameters import assignGlobalParameters
from tensilelite.KernelHelperNaming import kernelObjectNameCallables
from tensilelite.SolutionLibrary import MasterSolutionLibrary
from tensilelite.SolutionStructs import CompactSolution as CS
from tensilelite.SolutionStructs.CompactSolution import (
    CompactSolution,
    compactLibrarySolutions,
    hydrateSolution,
    kernelDedupeKey,
    kernelFileBase,
)
from tensilelite.SolutionStructs.Naming import getKernelFileBase, getKeyNoInternalArgs
from tensilelite.SolutionStructs.Solution import Solution

_LOGIC_YAML = (
    Path(__file__).parent
    / "characterization"
    / "TensileCreateLibraryRun"
    / "data"
    / "logic_gfx942_HSS_BH_tiny.yaml"
)


@pytest.fixture(scope="module")
def parsed():
    """(LibraryLogic with real Solutions, assembler, isaInfoMap) for one small logic file."""
    import shutil

    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Toolchain.Component import Assembler

    if not _LOGIC_YAML.is_file():
        pytest.skip(f"missing fixture {_LOGIC_YAML}")
    clang = shutil.which("clang++") or "/opt/rocm/lib/llvm/bin/clang++"
    isaInfoMap = makeIsaInfoMap([gfxToIsa("gfx942")], clang)
    assignGlobalParameters({}, isaInfoMap)
    assembler = Assembler(Path(clang), "4")
    logic = LibraryIO.parseLibraryLogicFile(
        str(_LOGIC_YAML), assembler, False, True, False, isaInfoMap, True
    )
    if not logic.library.lazyLibraries and not logic.library.solutions:
        pytest.skip("fixture produced no solutions")
    return logic, assembler, isaInfoMap


def _solutions(library: MasterSolutionLibrary):
    out = list(library.solutions.values())
    for lazy in library.lazyLibraries.values():
        out.extend(_solutions(lazy))
    return out


def _realAndCompact(parsed):
    logic, assembler, isaInfoMap = parsed
    real = _solutions(logic.library)[0].originalSolution
    assert isinstance(real, Solution)
    compact = CompactSolution.fromSolution(
        copy.deepcopy(real), False, kernelObjectNameCallables()
    )
    return real, compact


def test_summary_serves_parent_keys_without_hydrating(parsed):
    real, compact = _realAndCompact(parsed)
    before = CS.fallbackReads
    for key in CS.SUMMARY_KEYS:
        if key in real._state:
            assert compact[key] == real._state[key]
            assert key in compact
        else:
            with pytest.raises(KeyError):
                compact[key]
            assert key not in compact
            assert compact.get(key, "dflt") == "dflt"
    assert CS.fallbackReads == before


def test_names_and_keys_match_the_naming_helpers(parsed):
    real, compact = _realAndCompact(parsed)
    kernel = real.getKernels()[0]
    assert str(compact) == str(real)
    assert compact.baseName == getKernelFileBase(False, kernel)
    assert kernelFileBase(False, compact) == kernelFileBase(False, real)
    assert compact.dedupeKey() == getKeyNoInternalArgs(kernel, False)
    assert kernelDedupeKey(False, compact) == kernelDedupeKey(False, real)
    # codeObjectFile is assigned by the parent after compaction and is part of the key.
    compact["codeObjectFile"] = "lib_z"
    real._state["codeObjectFile"] = "lib_z"
    assert kernelDedupeKey(False, compact) == getKeyNoInternalArgs(kernel, False)
    assert kernelDedupeKey(False, compact).endswith(
        "lib_z" + str(real._state.get("DeviceNames", ""))
    )
    assert compact.helperNames == [(t, c(real)) for t, c in kernelObjectNameCallables()]


def test_hash_and_eq_follow_solution_semantics(parsed):
    real, compact = _realAndCompact(parsed)
    other = CompactSolution.fromSolution(copy.deepcopy(real))
    assert hash(compact) == hash(real) == hash(other)
    assert compact == other
    assert len(dict.fromkeys([compact, other])) == 1
    other["codeObjectFile"] = "somewhere"
    assert hash(other) != hash(
        compact
    )  # same as Solution: codeObjectFile is part of the hash


def test_hydrate_restores_the_solution_with_overlay(parsed):
    real, compact = _realAndCompact(parsed)
    compact["codeObjectFile"] = "lib_x"
    compact["BaseName"] = "base_x"
    compact.duplicate = True
    kernels = compact.getKernels()
    assert kernels == [compact]
    sol = compact.hydrate()
    assert isinstance(sol, Solution) and sol is not real
    assert sol._state["codeObjectFile"] == "lib_x"
    assert sol._state["BaseName"] == "base_x"
    assert sol._state["Kernel"] is True
    assert sol.duplicate is True
    assert sol.assembler is compact.assembler
    assert sol.isaInfoMap is compact.isaInfoMap
    stripped = {
        k: v
        for k, v in sol._state.items()
        if k not in ("codeObjectFile", "BaseName", "Kernel")
    }
    realState = {
        k: v
        for k, v in real._state.items()
        if k not in ("codeObjectFile", "BaseName", "Kernel")
    }
    assert set(stripped) == set(realState)
    for k in realState:
        if k != "ProblemType":
            assert stripped[k] == realState[k], k
    assert hydrateSolution(real) is real


def test_unknown_key_falls_back_and_promotes(parsed):
    real, compact = _realAndCompact(parsed)
    key = next(
        k for k in real._state if k not in CS.SUMMARY_KEYS and k != "ProblemType"
    )
    before = CS.fallbackReads
    for _ in range(CS.PROMOTE_AFTER):
        assert compact[key] == real._state[key]
    assert CS.fallbackReads == before + CS.PROMOTE_AFTER
    assert compact._hydrated is not None
    assert compact[key] == real._state[key]
    assert CS.fallbackReads == before + CS.PROMOTE_AFTER
    compact["codeObjectFile"] = "later"
    assert compact._hydrated._state["codeObjectFile"] == "later"


def test_pickle_round_trip_keeps_summary_absence_and_overlay(parsed):
    real, compact = _realAndCompact(parsed)
    compact["codeObjectFile"] = "lib_y"
    clone = pickle.loads(pickle.dumps(compact, pickle.HIGHEST_PROTOCOL))
    assert clone == compact and hash(clone) == hash(compact)
    assert clone["codeObjectFile"] == "lib_y"
    for key in CS.SUMMARY_KEYS:
        assert (key in clone) == (key in compact)
    assert clone.hydrate()._state["codeObjectFile"] == "lib_y"


def test_compact_is_much_smaller_than_the_pickled_solution(parsed):
    real, compact = _realAndCompact(parsed)
    # The blob holds everything but the process-wide assembler/isaInfoMap,
    # which travel once per pickle rather than once per solution.
    stateSize = len(pickle.dumps(real._state, pickle.HIGHEST_PROTOCOL))
    assert len(compact._blob) < stateSize / 2, (len(compact._blob), stateSize)


def test_compact_library_solutions_replaces_every_original(parsed):
    logic, assembler, isaInfoMap = parsed
    library = copy.deepcopy(logic.library)
    compactLibrarySolutions(library, False, kernelObjectNameCallables())
    sols = _solutions(library)
    assert sols and all(isinstance(s.originalSolution, CompactSolution) for s in sols)
    # Naming, which hydrates, still produces what the real solutions produce.
    library.applyNaming(False)
    for lazy in library.lazyLibraries.values():
        lazy.applyNaming(False)
    ref = copy.deepcopy(logic.library)
    ref.applyNaming(False)
    for lazy in ref.lazyLibraries.values():
        lazy.applyNaming(False)
    assert [(s.name, s.kernelName) for s in _solutions(library)] == [
        (s.name, s.kernelName) for s in _solutions(ref)
    ]
