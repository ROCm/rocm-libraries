################################################################################
# Characterization tests for Tensile.SolutionLibrary
#
# ADD-ONLY: pins the library (de)serialization class tree. The heavy
# integration entry points — MasterSolutionLibrary.FromOriginalState (and its
# nested level-builders) and BenchmarkingLibrary — require fully-derived
# Contractions.Solution / ProblemType objects, an assembler and an isaInfoMap;
# they are documented as resistance in target.md. Everything reachable with
# lightweight fakes is pinned here.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.SolutionLibrary")


class FakeSol:
    """Stand-in for a Contractions.Solution: has a mutable `.index`."""

    def __init__(self, index):
        self.index = index


# ---------------------------------------------------------------------------
# SingleSolutionLibrary / IndexSolutionLibrary
# ---------------------------------------------------------------------------
def test_single_solution_library():
    s = FakeSol(7)
    lib = M.SingleSolutionLibrary(s)
    assert lib.tag == "Single"
    assert lib.state() == {"type": "Single", "index": 7}
    assert lib.remapSolutionIndices({}) is None


def test_index_solution_library_state_is_bare_index():
    lib = M.IndexSolutionLibrary(FakeSol(3))
    assert lib.tag == "Index"
    assert lib.state() == 3


# ---------------------------------------------------------------------------
# PlaceholderLibrary
# ---------------------------------------------------------------------------
def test_placeholder_library_state_and_merge_ok():
    a = M.PlaceholderLibrary("foo")
    b = M.PlaceholderLibrary("foo")
    assert a.tag == "Placeholder"
    assert a.state() == {"type": "Placeholder", "value": "foo"}
    a.merge(b)  # same name -> no raise


def test_placeholder_library_merge_collision_raises():
    a = M.PlaceholderLibrary("foo")
    b = M.PlaceholderLibrary("bar")
    with pytest.raises(RuntimeError, match="COLLISION"):
        a.merge(b)


# ---------------------------------------------------------------------------
# MatchingLibrary.FromOriginalState — all distance variants
# ---------------------------------------------------------------------------
def _matching_d(distance, indexOrder, table):
    return {"indexOrder": indexOrder, "distance": distance, "table": table}


def test_matching_from_state_equality():
    sols = [FakeSol(0), FakeSol(1)]
    # indexOrder maps positions to property keys; key 0 valid for Equality
    d = _matching_d("Equality", [0, 1], [[[5, 9], [1, 0.8]]])
    lib = M.MatchingLibrary.FromOriginalState(d, sols)
    assert lib.tag == "Matching"
    assert lib.distance == "Equality"
    assert len(lib.table) == 1
    assert lib.table[0]["speed"] == 0.8
    assert isinstance(lib.table[0]["index"], M.IndexSolutionLibrary)


def test_matching_from_state_gridbased_has_no_speed():
    sols = [FakeSol(0), FakeSol(1)]
    d = _matching_d("GridBased", [0, 1], [[[5, 9], [1, 0.8]]])
    lib = M.MatchingLibrary.FromOriginalState(d, sols)
    assert "speed" not in lib.table[0]


def test_matching_from_state_range_property_map():
    sols = [FakeSol(0), FakeSol(1)]
    d = _matching_d("Range", [0, 2, 6], [[[5, 9, 13], [1, 0.5]]])
    lib = M.MatchingLibrary.FromOriginalState(d, sols)
    assert lib.distance == "Range"
    assert len(lib.table) == 1


def test_matching_merge_and_remap():
    sols = [FakeSol(0), FakeSol(1)]
    d1 = _matching_d("Equality", [0], [[[5], [1, 0.8]]])
    d2 = _matching_d("Equality", [0], [[[3], [0, 0.6]]])
    a = M.MatchingLibrary.FromOriginalState(d1, sols)
    b = M.MatchingLibrary.FromOriginalState(d2, sols)
    a.merge(b)
    assert len(a.table) == 2
    # sorted by key
    assert a.table[0]["key"] <= a.table[1]["key"]
    assert a.remapSolutionIndices({}) is None


# ---------------------------------------------------------------------------
# FreeSize / Prediction / MLPClassification FromOriginalState + merge
# ---------------------------------------------------------------------------
def test_free_size_library():
    sols = [FakeSol(i) for i in range(5)]
    lib = M.FreeSizeLibrary.FromOriginalState({"table": [1, 3]}, sols)  # idx 1,2,3
    assert lib.tag == "FreeSize"
    assert len(lib.table) == 3
    other = M.FreeSizeLibrary.FromOriginalState({"table": [0, 1]}, sols)
    lib.merge(other)
    assert len(lib.table) == 4


def test_prediction_library():
    sols = [FakeSol(i) for i in range(4)]
    lib = M.PredictionLibrary.FromOriginalState({"table": [0, 2]}, sols)
    assert lib.tag == "Prediction"
    assert len(lib.table) == 2
    lib.merge(M.PredictionLibrary.FromOriginalState({"table": [2, 1]}, sols))
    assert len(lib.table) == 3


def test_mlp_classification_library_and_merge_raises():
    sols = [FakeSol(i) for i in range(3)]
    d = {"table": [0, 2], "mlp": {"w": 1}, "problemFeatures": ["m", "n"]}
    lib = M.MLPClassificationLibrary.FromOriginalState(d, sols)
    assert lib.tag == "MLPClassification"
    assert lib.mlp == {"w": 1}
    assert lib.problemFeatures == ["m", "n"]
    with pytest.raises(RuntimeError, match="does not support merging"):
        lib.merge(lib)


# ---------------------------------------------------------------------------
# ProblemMapLibrary
# ---------------------------------------------------------------------------
def test_problem_map_library_merge_and_remap():
    inner_a = M.FreeSizeLibrary([{"index": M.IndexSolutionLibrary(FakeSol(0))}])
    inner_b = M.FreeSizeLibrary([{"index": M.IndexSolutionLibrary(FakeSol(1))}])
    inner_c = M.FreeSizeLibrary([{"index": M.IndexSolutionLibrary(FakeSol(2))}])
    a = M.ProblemMapLibrary("prop", {"k1": inner_a})
    b = M.ProblemMapLibrary("prop", {"k1": inner_b, "k2": inner_c})
    a.merge(b)  # k1 merges, k2 added
    assert set(a.mapping.keys()) == {"k1", "k2"}
    assert len(a.mapping["k1"].table) == 2
    assert a.remapSolutionIndices({}) is None  # delegates (no-op children)


# ---------------------------------------------------------------------------
# PredicateLibrary
# ---------------------------------------------------------------------------
def test_predicate_library_merge_predicate_sort():
    a = M.PredicateLibrary(
        tag="Problem",
        rows=[{"predicate": "p2", "library": M.FreeSizeLibrary([])}],
    )
    b = M.PredicateLibrary(
        tag="Problem",
        rows=[
            {"predicate": "p2", "library": M.FreeSizeLibrary([{"index": 1}])},  # merges
            {"predicate": "p1", "library": M.FreeSizeLibrary([])},               # appended
        ],
    )
    a.merge(b)
    # not placeholder -> sorted by predicate string
    assert [r["predicate"] for r in a.rows] == ["p1", "p2"]
    # p2's library was merged (table grew)
    p2 = next(r for r in a.rows if r["predicate"] == "p2")
    assert len(p2["library"].table) == 1


def test_predicate_library_merge_placeholder_sort_by_name_length():
    a = M.PredicateLibrary(
        tag="Hardware",
        rows=[{"predicate": "x", "library": M.PlaceholderLibrary("longname")}],
    )
    b = M.PredicateLibrary(
        tag="Hardware",
        rows=[{"predicate": "y", "library": M.PlaceholderLibrary("ab")}],
    )
    a.merge(b)
    # placeholder rows sort by len(filenamePrefix): "ab" before "longname"
    assert a.rows[0]["library"].filenamePrefix == "ab"


def test_predicate_library_remap_delegates():
    called = []

    class _Lib:
        def remapSolutionIndices(self, m):
            called.append(m)

    pl = M.PredicateLibrary(tag="Problem", rows=[{"predicate": "p", "library": _Lib()}])
    pl.remapSolutionIndices({"a": 1})
    assert called == [{"a": 1}]


# ---------------------------------------------------------------------------
# MasterSolutionLibrary.hardware (staticmethod)
# ---------------------------------------------------------------------------
def test_hardware_fallback_truepred():
    lib = M.PlaceholderLibrary("inner")
    newLib, name = M.MasterSolutionLibrary.hardware(
        {"ArchitectureName": "fallback", "CUCount": 0}, lib, "Base", lazyLibrary=False
    )
    assert newLib.tag == "Hardware"
    assert newLib.rows[0]["library"] is lib
    assert name == "Base"  # not lazy -> unchanged


def test_hardware_real_arch_lazy_suffixes():
    lib = M.PlaceholderLibrary("inner")
    d = {"ArchitectureName": "gfx942", "CUCount": 104, "DeviceNames": ["Device 75a0"]}
    newLib, name = M.MasterSolutionLibrary.hardware(d, lib, "Base", lazyLibrary=True)
    assert newLib.tag == "Hardware"
    assert "_CU104" in name
    assert name.endswith("_gfx942")


# ---------------------------------------------------------------------------
# FixSolutionIndices
# ---------------------------------------------------------------------------
def test_fix_solution_indices_fills_none_and_dedupes():
    sols = [FakeSol(0), FakeSol(None), FakeSol(0), FakeSol(2)]
    M.MasterSolutionLibrary.FixSolutionIndices(sols)
    idxs = [s.index for s in sols]
    assert len(set(idxs)) == 4  # all unique now
    assert None not in idxs


def test_fix_solution_indices_all_none():
    sols = [FakeSol(None), FakeSol(None)]
    M.MasterSolutionLibrary.FixSolutionIndices(sols)
    assert sorted(s.index for s in sols) == [0, 1]


# ---------------------------------------------------------------------------
# MasterSolutionLibrary instance: __init__/state/cpp/remap/merge
# ---------------------------------------------------------------------------
def test_master_init_and_state_empty():
    lib = M.PlaceholderLibrary("inner")
    msl = M.MasterSolutionLibrary({}, lib)
    st = msl.state()
    assert st["library"] == {"type": "Placeholder", "value": "inner"}
    assert "version" not in st
    assert msl.lazyLibraries == {}


def test_master_state_includes_version():
    msl = M.MasterSolutionLibrary({}, M.PlaceholderLibrary("inner"), version="v1")
    assert msl.state()["version"] == "v1"


def test_master_cpp_class_names():
    msl = M.MasterSolutionLibrary({}, M.PlaceholderLibrary("inner"))
    assert "MasterSolutionLibrary" in msl.cpp_class
    assert "SolutionLibrary<" in msl.cpp_base_class


def test_master_remap_starting_from():
    s0, s1 = FakeSol(5), FakeSol(9)

    class _Lib:
        def __init__(self):
            self.seen = None

        def remapSolutionIndices(self, m):
            self.seen = m

    lib = _Lib()
    msl = M.MasterSolutionLibrary({5: s0, 9: s1}, lib)
    msl.remapSolutionIndicesStartingFrom(100)
    assert sorted(msl.solutions.keys()) == [100, 101]
    assert s0.index == 100 and s1.index == 101
    assert lib.seen == {5: 100, 9: 101}


def test_master_merge_reindexes_and_merges_library():
    # two masters whose libraries are PredicateLibraries with mergeable rows
    s_self = FakeSol(0)
    s_other = FakeSol(0)

    lib_self = M.PredicateLibrary(tag="Problem", rows=[{"predicate": "p", "library": M.FreeSizeLibrary([])}])
    lib_other = M.PredicateLibrary(tag="Problem", rows=[{"predicate": "q", "library": M.FreeSizeLibrary([])}])

    a = M.MasterSolutionLibrary({0: s_self}, lib_self)
    b = M.MasterSolutionLibrary({0: s_other}, lib_other)
    nextIndex = a.merge(b)
    # other's solution reindexed to 1; both predicates present after merge
    assert s_other.index == 1
    assert 1 in a.solutions
    assert nextIndex == 2
    assert sorted(r["predicate"] for r in a.library.rows) == ["p", "q"]


def test_master_apply_naming(monkeypatch):
    monkeypatch.setattr(M, "getSolutionNameMin", lambda k, split: "SNAME")
    monkeypatch.setattr(M, "getKernelNameMin", lambda k, split: "KNAME")

    class _Orig:
        def getKernels(self):
            return [{"k": 1}]

    class _S:
        def __init__(self):
            self.index = 0
            self.originalSolution = _Orig()

    s = _S()
    msl = M.MasterSolutionLibrary({0: s}, M.PlaceholderLibrary("x"))
    msl.applyNaming(splitGSU=False)
    assert s.name == "SNAME"
    assert s.kernelName == "KNAME"


# ---------------------------------------------------------------------------
# MasterSolutionLibrary.FromOriginalState — non-lazy core, all selection types.
# The Contractions.ProblemType derivation is stubbed; the solutionClass is a
# fake whose FromSolutionStruct yields index-bearing fakes. This drives the
# level-builders (hardware/operationIdentifier/performanceMetric/predicates/
# selection) and the bottom-up assembly loop without a real assembler/ISA.
# ---------------------------------------------------------------------------
class _FakeProblemType:
    operationIdentifier = "OP_ID"

    def predicates(self, includeBatch=False, includeType=False):
        return []


class _FakeSolClass:
    @classmethod
    def FromSolutionStruct(cls, s, *a, **k):
        return FakeSol(s)


def _run_from_state(monkeypatch, library_type, library_spec):
    monkeypatch.setattr(
        M.Contractions.ProblemType, "FromOriginalState", staticmethod(lambda d: _FakeProblemType())
    )
    origData = {
        "ProblemType": {},
        "ArchitectureName": "fallback",
        "CUCount": 0,
        "LibraryType": library_type,
        "Library": library_spec,
    }
    rv, name = M.MasterSolutionLibrary.FromOriginalState(
        origData,
        [0, 1],                 # origSolutions -> FakeSol(0), FakeSol(1)
        splitGSU=False,
        printSolutionRejectionReason=False,
        printIndexAssignmentInfo=False,
        assembler=None,
        isaInfoMap={},
        lazyLibraryLoading=False,
        solutionClass=_FakeSolClass,
    )
    return rv, name


def test_from_state_freesize_nonlazy(monkeypatch):
    rv, name = _run_from_state(monkeypatch, "FreeSize", {"table": [0, 2]})
    assert isinstance(rv, M.MasterSolutionLibrary)
    # top library is the Hardware PredicateLibrary
    assert rv.library.tag == "Hardware"
    assert len(rv.solutions) == 2


def test_from_state_matching_nonlazy(monkeypatch):
    rv, _ = _run_from_state(
        monkeypatch,
        "Matching",
        {"distance": "Equality", "indexOrder": [0], "table": [[[5], [0, 0.9]]]},
    )
    assert isinstance(rv, M.MasterSolutionLibrary)


def test_from_state_prediction_nonlazy(monkeypatch):
    rv, _ = _run_from_state(monkeypatch, "Prediction", {"table": [0, 1]})
    assert isinstance(rv, M.MasterSolutionLibrary)


def test_from_state_mlp_nonlazy(monkeypatch):
    rv, _ = _run_from_state(
        monkeypatch,
        "MLPClassification",
        {"table": [0, 1], "mlp": {}, "problemFeatures": []},
    )
    assert isinstance(rv, M.MasterSolutionLibrary)


def test_from_state_unrecognized_library_type_asserts(monkeypatch):
    with pytest.raises(AssertionError):
        _run_from_state(monkeypatch, "Bogus", {"table": [0, 1]})


# ---- lazy path: placeholder recursion + the placeholderName suffix block ----
class _FakeType:
    def __init__(self, s, single=True):
        self._s = s
        self._single = single

    def __str__(self):
        return self._s

    def __eq__(self, other):
        return isinstance(other, _FakeType) and other._s == self._s

    def isSingle(self):
        return self._single


class _FakeAct:
    def __ne__(self, other):
        return other != "relu"

    def __str__(self):
        return "relu"


class _LazyProblemType:
    operationIdentifier = "OP_ID"
    # selection() lazy-suffix attributes
    computeInputTypeA = _FakeType("f32")
    computeInputTypeB = _FakeType("f32")
    aType = _FakeType("f16")
    bType = _FakeType("f16")
    cType = _FakeType("f16")
    activationType = _FakeAct()
    mxBlockA = 0
    mxBlockB = 0
    mxTypeA = _FakeType("x")
    mxTypeB = _FakeType("x")
    swizzleTensorA = False
    swizzleTensorB = False
    useBias = False
    useE = False
    useGradient = False
    groupedGemm = False
    stridedBatched = True
    useScaleAB = "None"
    useScaleCD = False
    useScaleAlphaVec = False
    sparse = 0
    metadataLayout = 0
    f32XdlMathOp = _FakeType("f32", single=True)
    supportDeviceUserArguments = False

    def predicates(self, includeBatch=False, includeType=False):
        return []

    def placeholderStr(self, includeBatch=False, includeType=False):
        return "_PS"


class _FakeActAll:
    def __ne__(self, other):
        return other != "all"

    def __str__(self):
        return "all"


class _LazyProblemType2:
    """All placeholder-suffix toggles ON, to cover the alternate branches."""

    operationIdentifier = "OP_ID"
    computeInputTypeA = _FakeType("f32")
    computeInputTypeB = _FakeType("f64")     # != A -> combined-type branch
    aType = _FakeType("f16")
    bType = _FakeType("f16")
    cType = _FakeType("f16")
    activationType = _FakeActAll()           # 'all' -> _A branch
    mxBlockA = 4
    mxBlockB = 4
    mxTypeA = _FakeType("e8")
    mxTypeB = _FakeType("e5")
    swizzleTensorA = True
    swizzleTensorB = True
    useBias = True
    useE = True
    useGradient = True                       # -> _Grad
    groupedGemm = True                       # -> _GG
    stridedBatched = True
    useScaleAB = "Scalar"                    # -> _SAB
    useScaleCD = True
    useScaleAlphaVec = True
    sparse = 2                               # -> _SPB
    metadataLayout = 1
    f32XdlMathOp = _FakeType("xf32", single=False)
    supportDeviceUserArguments = True

    def predicates(self, includeBatch=False, includeType=False):
        return []

    def placeholderStr(self, includeBatch=False, includeType=False):
        return "_PS"


def test_from_state_lazy_all_suffix_toggles(monkeypatch):
    monkeypatch.setattr(
        M.Contractions.ProblemType, "FromOriginalState", staticmethod(lambda d: _LazyProblemType2())
    )
    origData = {
        "ProblemType": {},
        "ArchitectureName": "fallback",
        "CUCount": 0,
        "LibraryType": "FreeSize",
        "Library": {"table": [0, 1]},
    }
    rv, _ = M.MasterSolutionLibrary.FromOriginalState(
        origData, [0, 1], False, False, False, None, {}, True, solutionClass=_FakeSolClass
    )
    name = next(iter(rv.lazyLibraries))
    for frag in ("_A", "_MXA", "_MXB", "_STA", "_STB", "_Bias", "_Grad", "_GG",
                 "_SAB", "_SCD", "_SAV", "_SPB", "_UA"):
        assert frag in name, frag


def test_from_state_lazy_placeholder_recursion(monkeypatch):
    monkeypatch.setattr(
        M.Contractions.ProblemType, "FromOriginalState", staticmethod(lambda d: _LazyProblemType())
    )
    origData = {
        "ProblemType": {},
        "ArchitectureName": "fallback",
        "CUCount": 0,
        "LibraryType": "FreeSize",
        "Library": {"table": [0, 1]},
    }
    rv, name = M.MasterSolutionLibrary.FromOriginalState(
        origData,
        [0, 1],
        splitGSU=False,
        printSolutionRejectionReason=False,
        printIndexAssignmentInfo=False,
        assembler=None,
        isaInfoMap={},
        lazyLibraryLoading=True,        # -> default lazy libraryOrder incl. placeholder
        solutionClass=_FakeSolClass,
    )
    assert isinstance(rv, M.MasterSolutionLibrary)
    # the recursion populated a lazy library keyed by the derived placeholder name
    assert rv.lazyLibraries
    derivedName = next(iter(rv.lazyLibraries))
    assert "_RELU" in derivedName        # activation suffix branch
    assert derivedName.endswith("_fallback") or "_fallback" in derivedName


# ---------------------------------------------------------------------------
# merge with lazyLibraries (both the merge-existing and reindex-new branches)
# ---------------------------------------------------------------------------
class _LazyLib:
    """Minimal stand-in for a per-file MasterSolutionLibrary inside lazyLibraries."""

    def __init__(self, solutions):
        self.solutions = solutions
        self.library = self
        self.merged_with = None

    def remapSolutionIndices(self, m):
        pass

    def merge(self, other, curIndex):
        self.merged_with = other
        return curIndex + len(other.solutions)


def test_master_merge_with_lazy_libraries():
    # PredicateLibrary.merge dereferences rows[0], so give each a (same-predicate)
    # row to merge cleanly.
    a = M.MasterSolutionLibrary(
        {0: FakeSol(0)},
        M.PredicateLibrary(tag="P", rows=[{"predicate": "p", "library": M.FreeSizeLibrary([])}]),
    )
    b = M.MasterSolutionLibrary(
        {0: FakeSol(0)},
        M.PredicateLibrary(tag="P", rows=[{"predicate": "p", "library": M.FreeSizeLibrary([])}]),
    )

    shared = _LazyLib({0: FakeSol(0)})
    a.lazyLibraries = {"shared": shared}
    # b has a shared (-> merge branch) and a brand-new (-> reindex branch)
    b.lazyLibraries = {
        "shared": _LazyLib({0: FakeSol(0)}),
        "fresh": _LazyLib({0: FakeSol(0)}),
    }
    a.merge(b)
    assert shared.merged_with is not None        # existing name merged
    assert "fresh" in a.lazyLibraries            # new name adopted


# ---------------------------------------------------------------------------
# hardware() chip-id suffix branch (gfx950 is the only chip-id-aware arch)
# ---------------------------------------------------------------------------
def test_hardware_chipid_suffix_gfx950():
    lib = M.PlaceholderLibrary("inner")
    d = {
        "ArchitectureName": "gfx950",
        "CUCount": 64,
        "DeviceNames": ["Device 75a0", "Device 75b0"],
    }
    newLib, name = M.MasterSolutionLibrary.hardware(d, lib, "Base", lazyLibrary=True)
    assert "_ID75a0-75b0" in name
    assert name.endswith("_gfx950")


# ---------------------------------------------------------------------------
# BenchmarkingLibrary (Contractions.Solution.FromOriginalState stubbed)
# ---------------------------------------------------------------------------
def test_benchmarking_library(monkeypatch):
    class _Built:
        _next = 0

        def __init__(self):
            self.index = None
            self.problemPredicate = "PRED"

    monkeypatch.setattr(
        M.Contractions.Solution, "FromOriginalState", staticmethod(lambda *a, **k: _Built())
    )

    class _InSol:
        _state = {"dummy": 1}

    msl = M.MasterSolutionLibrary.BenchmarkingLibrary(
        [_InSol(), _InSol()],
        assembler=None,
        splitGSU=False,
        printSolutionRejectionReason=False,
        printIndexAssignmentInfo=False,
        isaInfoMap={},
    )
    assert isinstance(msl, M.MasterSolutionLibrary)
    assert msl.library.tag == "Problem"
    assert len(msl.library.rows) == 2
    # FixSolutionIndices assigned 0 and 1
    assert sorted(msl.solutions.keys()) == [0, 1]
