# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv
from unittest.mock import patch

import pytest

import tensilelite.ClientWriter as CW
import tensilelite.SolutionLibrary as SL
from tensilelite.Contractions import ProblemType as ContractionsProblemType
from tensilelite.SolutionStructs.Problem import Problem, ProblemSizes, ProblemType
from tensilelite.TensileMergeLibrary import fixSizeInconsistencies

pytestmark = pytest.mark.unit

_FUSED_CONFIG = {
    "OperationType": "GEMM",
    "DataType": "b",
    "DestDataType": "b",
    "ComputeDataType": "s",
    "HighPrecisionAccumulate": True,
    "TransposeA": True,
    "TransposeB": False,
    "UseBeta": True,
    "Batched": True,
    "FusedGemmA2A": True,
}


def _problemType(**overrides):
    return ProblemType({**_FUSED_CONFIG, **overrides}, False)


def _sizes(problemType, *exacts):
    return ProblemSizes(problemType, [{"Exact": e} for e in exacts])


class TestExactDictWorld:
    def test_extent_defaults_to_m(self):
        ps = _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": 2})
        assert ps.exacts[0].a2aExtent == 3072

    def test_fused_requires_world(self):
        with pytest.raises(RuntimeError, match="must set 'world'"):
            _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048]})

    @pytest.mark.parametrize("world", [0, 9])
    def test_world_must_be_in_range(self, world):
        with pytest.raises(RuntimeError, match=r"world=%d must be in \[1, 8\]" % world):
            _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": world})

    @pytest.mark.parametrize("world", [1, 8])
    def test_world_bounds_are_accepted(self, world):
        assert _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": world}).exacts[0].world == world

    @pytest.mark.parametrize("extent", [0, 3073])
    def test_extent_must_be_within_m(self, extent):
        with pytest.raises(RuntimeError, match="a2aExtent=%d" % extent):
            _sizes(_problemType(), {"sizes": [3072, 512, 1, 2048], "world": 2, "a2aExtent": extent})

    def test_fused_rejects_batch(self):
        with pytest.raises(RuntimeError, match="batch=2"):
            _sizes(_problemType(), {"sizes": [3072, 512, 2, 2048], "world": 2})

    def test_non_fused_rejects_world(self):
        with pytest.raises(RuntimeError, match="only to FusedGemmA2A"):
            _sizes(_problemType(FusedGemmA2A=False), {"sizes": [3072, 512, 1, 2048], "world": 2})

    def test_fused_rejects_exact_list(self):
        with pytest.raises(RuntimeError, match="Exact dict"):
            _sizes(_problemType(), [3072, 512, 1, 2048])

    def test_fused_rejects_range(self):
        with pytest.raises(RuntimeError, match="Range"):
            ProblemSizes(_problemType(), [{"Range": [[3072], [512], [1], [2048]]}])


class TestFusedProblemTypeRejects:
    @pytest.mark.parametrize("overrides, name", [
        ({"UseBias": 1}, "UseBias"),
        ({"UseScaleAB": "Scalar"}, "UseScaleAB"),
        ({"Activation": True, "ActivationType": "hipblaslt_all"}, "Activation"),
    ])
    def test_rejects(self, overrides, name):
        with pytest.raises(Exception, match="FusedGemmA2A does not support .*%s" % name):
            _problemType(**overrides)


class _FakeSol:
    def __init__(self, index):
        self.index = index


class TestFusedMatchingKey:
    _ROW = [[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 4, 2048], [0, 1000.0]]

    def test_world_and_extent_join_the_key(self):
        d = {"indexOrder": [2, 3, 0, 1], "distance": "Equality", "table": [self._ROW]}
        lib = SL.MatchingLibrary.FromOriginalState(d, [_FakeSol(0)], fusedGemmA2A=True)
        assert [p.tag for p in lib.properties][-2:] == ["World", "FusedA2AExtent"]
        assert lib.table[0]["key"] == [3072, 512, 1, 2048, 4, 2048]

    def test_non_fused_key_unchanged(self):
        row = [self._ROW[0][:8], self._ROW[1]]
        d = {"indexOrder": [2, 3, 0, 1], "distance": "Equality", "table": [row]}
        lib = SL.MatchingLibrary.FromOriginalState(d, [_FakeSol(0)])
        assert lib.table[0]["key"] == [3072, 512, 1, 2048]
        assert len(lib.properties) == 4


class TestMergeKeepsWorld:
    def test_strips_lds_only(self):
        rows, n = fixSizeInconsistencies(
            [[[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 2, 2048], [0, 1.0]],
             [[3072, 512, 1, 2048, 3072, 3072, 2048, 2048, 4, 2048], [0, 2.0]]], "base", True)
        assert n == 2
        assert sorted(r[0] for r in rows) == [[3072, 512, 1, 2048, 2, 2048],
                                              [3072, 512, 1, 2048, 4, 2048]]
        assert fixSizeInconsistencies(rows, "base", True) == (rows, 2)


class TestClientWriterWorld:
    def test_fused_writes_world_and_extent(self):
        cpt = ContractionsProblemType.FromOriginalState(_problemType().state)
        problem = Problem(sizes=[3072, 512, 1, 2048, 3072, 3072, 2048, 2048], world=4, a2aExtent=2048)
        params = CW.problemSizeParams(cpt, problem, [0])
        assert ("world", 4) in params and ("fused-a2a-extent", 2048) in params


class _MockDataType:
    def flopsPerMac(self):
        return 2


class _FusedMockProblemType(dict):
    def __init__(self):
        super().__init__({
            "Index0": 0, "Index1": 1, "IndexUnroll": 3, "TotalIndices": 4, "NumIndicesLD": 0,
            "NumIndicesC": 3, "IndexAssignmentsA": [3, 0, 2], "IndexAssignmentsB": [3, 1, 2],
            "IndexAssignmentsLD": [], "DataType": _MockDataType(), "OperationType": "GEMM",
            "TileAwareSelection": False, "GroupedGemm": False, "FusedGemmA2A": True,
        })

    def __hash__(self):
        return id(self)


class _MockSolutionDict(dict):
    def __init__(self, idx):
        super().__init__({"MacroTile0": 256, "MacroTile1": 256, "SolutionIndex": idx})

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _FusedSizes:
    def __init__(self, keys):
        self.problems = []
        self.ranges = []
        self.exacts = [Problem(sizes=k[:4], world=k[4], a2aExtent=k[5]) for k in keys]


def _analyze(tmp_path, header, rows, keys, numSolutions):
    from tensilelite.Common.GlobalParameters import defaultAnalysisParameters
    from tensilelite.LibraryLogic import LogicAnalyzer
    path = tmp_path / "bench.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    name = lambda sol, *a, **k: "Sol%d" % sol["SolutionIndex"]
    with patch("tensilelite.LibraryLogic.getSolutionNameMin", name), \
         patch("tensilelite.LibraryLogic.getKernelNameMin", name), \
         patch("tensilelite.LibraryLogic.getSolutionNameFull", name):
        return LogicAnalyzer(_FusedMockProblemType(), [_FusedSizes(keys)],
                             [[_MockSolutionDict(i) for i in range(numSolutions)]], [str(path)],
                             dict(defaultAnalysisParameters), splitGSU=False)


class TestLibraryLogicWorldKey:
    _HEADER = ["GFlops", "SizeI", "SizeJ", "SizeK", "SizeL", "World", "A2AExtent", "TotalFlops"]

    def test_winner_per_world(self, tmp_path):
        keys = [(3072, 512, 1, 2048, 2, 2048), (3072, 512, 1, 2048, 4, 2048), (3072, 512, 1, 2048, 8, 2048)]
        la = _analyze(tmp_path, self._HEADER + ["S0", "S1"],
                      [["GFlops", *keys[0], 1, 20.0, 10.0], ["GFlops", *keys[1], 1, 10.0, 30.0]], keys, 2)
        assert {k: w[0] for k, w in la.exactWinners.items()} == {keys[0]: 0, keys[1]: 1}

    def test_missing_world_header_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _analyze(tmp_path, [c for c in self._HEADER if c != "World"] + ["S0"],
                     [["GFlops", 3072, 512, 1, 2048, 2048, 1, 20.0]], [(3072, 512, 1, 2048, 2, 2048)], 1)
        assert "World" in capsys.readouterr().out


class TestSyntheticCsvWorld:
    def test_columns(self, tmp_path):
        from tensilelite.BenchmarkProblems import _writeSyntheticResultsCSV

        class _PS:
            problems = [Problem(sizes=[3072, 512, 1, 2048, 3072, 3072, 2048, 2048], world=4, a2aExtent=2048)]

        path = tmp_path / "s.csv"
        _writeSyntheticResultsCSV(str(path), _PS(), "gfx950", 1)
        rows = list(csv.reader(open(path)))
        assert rows[1][1:11] == ["3072", "512", "1", "2048", "3072", "3072", "2048", "2048", "4", "2048"]


class TestBenchmarkDataRoundTrip:
    def test_exact_line_reloads_with_world(self):
        import yaml
        pt = _problemType()
        written = _sizes(pt, {"sizes": [3072, 512, 1, 2048], "world": 4, "a2aExtent": 2048}).exacts[0]
        loaded = yaml.load("- Exact: {}\n".format(written), yaml.CSafeLoader)
        ps = ProblemSizes(pt, loaded)
        assert (ps.exacts[0].world, ps.exacts[0].a2aExtent) == (4, 2048)


class TestLibLogicToYamlWorld:
    _ROW = [3072, 512, 1, 2048, 3072, 3072, 2048, 2048]

    @staticmethod
    def _reload(problemType, data):
        import yaml
        written = yaml.dump(data["BenchmarkFinalParameters"][0]["ProblemSizes"])
        return ProblemSizes(problemType, yaml.load(written, yaml.CSafeLoader)).exacts

    def test_fused_logic_row_becomes_an_exact_dict(self):
        from tensilelite.TensileLibLogicToYaml import formProblemSize
        logic = [(self._ROW + [4, 2048], [0, 1.0]), (self._ROW + [8, 1024], [1, 2.0])]
        data = formProblemSize(logic, 1, dict(_FUSED_CONFIG))
        p = self._reload(_problemType(), data)[0]
        assert (list(p.sizes), p.world, p.a2aExtent) == (self._ROW, 8, 1024)

    def test_benchmark_exact_dict_keeps_its_fields(self):
        from tensilelite.TensileLibLogicToYaml import formProblemSize
        exact = {"sizes": [3072, 512, 1, 2048], "world": 2, "a2aExtent": 2048}
        data = formProblemSize(None, 0, {}, problemSizes=[{"Exact": exact}])
        p = self._reload(_problemType(), data)[0]
        assert (list(p.sizes[:4]), p.world, p.a2aExtent) == ([3072, 512, 1, 2048], 2, 2048)
