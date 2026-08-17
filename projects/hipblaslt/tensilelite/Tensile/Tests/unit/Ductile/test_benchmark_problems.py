# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for BenchmarkProblems._build_and_validate_solution and related helpers.

Tests cover: MI expansion (len=9 and len=0), WavefrontSize=-1 substitution,
silent vs. verbose rejection, None returned on exception, _generate_single_solution
pass-through, _generate_ga_solutions alignment/deduplication.
"""

import types

import pytest

import Tensile.BenchmarkProblems as BP

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

def _isa_info(has_wave32=True):
    return types.SimpleNamespace(archCaps={"HasWave32": has_wave32})


def _isa_info_map(isa="gfx942", has_wave32=True):
    return {isa: _isa_info(has_wave32)}


def _debug_config(silent_rejection=True):
    return types.SimpleNamespace(
        splitGSU=False,
        printSolutionRejectionReason=not silent_rejection,
        printIndexAssignmentInfo=False,
    )


class _ValidSolution:
    """Minimal Solution-like that is always valid."""
    def __init__(self, *args, **kwargs):
        self._valid = True

    def __getitem__(self, key):
        if key == "Valid":
            return self._valid
        return None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _InvalidSolution(_ValidSolution):
    def __getitem__(self, key):
        if key == "Valid":
            return False
        return None


# ---------------------------------------------------------------------------
# _build_and_validate_solution
# ---------------------------------------------------------------------------

class TestBuildAndValidateSolution:
    def _base_solution(self, mi=(), wavefront=-1):
        return {
            "ProblemType": {"DataType": "f32"},
            "ISA": "gfx942",
            "MatrixInstruction": list(mi),
            "WavefrontSize": wavefront,
            "WorkGroup": [16, 16, 1],
        }

    def test_returns_none_when_mi_validation_fails(self, monkeypatch):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: False)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is None

    def test_returns_none_when_solution_invalid(self, monkeypatch):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})
        monkeypatch.setattr(BP, "Solution", _InvalidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is None

    def test_returns_solution_when_valid(self, monkeypatch):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})
        monkeypatch.setattr(BP, "Solution", _ValidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is not None

    def test_wavefront_minus1_resolved_to_32_when_has_wave32(self, monkeypatch):
        captured = {}

        def fake_mi_params(mi, isa, wavefront, ptype, workgroup, isa_map):
            captured["wavefront"] = wavefront
            return {}

        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", fake_mi_params)
        monkeypatch.setattr(BP, "Solution", _ValidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=-1)
        BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map(has_wave32=True))
        assert captured["wavefront"] == 32

    def test_wavefront_minus1_resolved_to_64_when_no_wave32(self, monkeypatch):
        captured = {}

        def fake_mi_params(mi, isa, wavefront, ptype, workgroup, isa_map):
            captured["wavefront"] = wavefront
            return {}

        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", fake_mi_params)
        monkeypatch.setattr(BP, "Solution", _ValidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=-1)
        BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map(has_wave32=False))
        assert captured["wavefront"] == 64

    def test_empty_mi_disables_matrix_instruction(self, monkeypatch):
        captured = {}

        class _CapSolution(_ValidSolution):
            def __init__(self, sol, *a, **kw):
                super().__init__(sol, *a, **kw)
                captured["emi"] = sol.get("EnableMatrixInstruction")

        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "Solution", _CapSolution)

        sol = self._base_solution(mi=[], wavefront=64)
        BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert captured.get("emi") is False

    def test_returns_none_on_exception(self, monkeypatch):
        def raise_exc(*a, **kw):
            raise RuntimeError("simulated error")

        monkeypatch.setattr(BP, "validateMIParameters", raise_exc)

        sol = self._base_solution(mi=[], wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map())
        assert result is None

    def test_silent_mode_suppresses_rejection_print(self, monkeypatch, capsys):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: False)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        debug = types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=True,
            printIndexAssignmentInfo=False,
        )
        BP._build_and_validate_solution(sol, object(), debug, _isa_info_map(), silent=True)
        out = capsys.readouterr().out
        assert "rejecting" not in out

    def test_invalid_solution_prints_rejection_when_verbose(self, monkeypatch, capsys):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: True)
        monkeypatch.setattr(BP, "matrixInstructionToMIParameters", lambda *a, **kw: {})
        monkeypatch.setattr(BP, "Solution", _InvalidSolution)

        sol = self._base_solution(mi=[0] * 9, wavefront=64)
        debug = types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=True,
            printIndexAssignmentInfo=False,
        )
        result = BP._build_and_validate_solution(sol, object(), debug, _isa_info_map(), silent=False)
        out = capsys.readouterr().out
        assert result is None
        assert "rejecting solution" in out

    def test_exception_path_prints_error_when_not_silent(self, monkeypatch, capsys):
        def raise_exc(*_a, **_kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(BP, "validateMIParameters", raise_exc)

        sol = self._base_solution(mi=[], wavefront=64)
        result = BP._build_and_validate_solution(sol, object(), _debug_config(), _isa_info_map(), silent=False)
        out = capsys.readouterr().out
        assert result is None
        assert "Error processing permutation: boom" in out

    def test_validate_mi_false_prints_rejection_when_verbose(self, monkeypatch, capsys):
        monkeypatch.setattr(BP, "validateMIParameters", lambda sol, isa_map: False)

        sol = self._base_solution(mi=[], wavefront=64)
        debug = types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=True,
            printIndexAssignmentInfo=False,
        )
        result = BP._build_and_validate_solution(sol, object(), debug, _isa_info_map(), silent=False)
        out = capsys.readouterr().out
        assert result is None
        assert "rejecting solution" in out


# ---------------------------------------------------------------------------
# _generate_single_solution — thin wrapper over _build_and_validate_solution
# ---------------------------------------------------------------------------

class TestGenerateSingleSolution:
    def test_passes_perm_and_constant_params_to_build(self, monkeypatch):
        captured = {}

        def fake_build(solution, *a, **kw):
            captured["solution"] = solution
            return None

        monkeypatch.setattr(BP, "_build_and_validate_solution", fake_build)

        perm = {"DepthU": 32}
        constant = {"WorkGroup": [16, 16, 1]}
        problem_type = types.SimpleNamespace(state={"DataType": "f32"})

        BP._generate_single_solution(perm, problem_type, constant, object(), _debug_config(), _isa_info_map())

        sol = captured["solution"]
        assert sol["DepthU"] == 32
        assert sol["WorkGroup"] == [16, 16, 1]
        assert "ProblemType" in sol
        assert "ISA" in sol


# ---------------------------------------------------------------------------
# _generate_ga_solutions (in ductile_backend.py)
# ---------------------------------------------------------------------------

class TestGenerateGaSolutions:
    def test_valid_solutions_appended(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        monkeypatch.setattr(
            dbmod,
            "_generate_single_solution_with_groups",
            lambda perm, *a, **kw: _ValidSolution(),
        )
        monkeypatch.setattr(dbmod, "getKernelFileBase", lambda splitGSU, sol: str(id(sol)))

        individuals = [{"a": 0}, {"a": 1}]
        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            individuals,
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        assert len(result) == 2
        assert all(r is not None for r in result)

    def test_invalid_candidates_become_none(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        monkeypatch.setattr(
            dbmod,
            "_generate_single_solution_with_groups",
            lambda perm, *a, **kw: None,  # always invalid
        )

        individuals = [{"a": 0}, {"a": 1}]
        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            individuals,
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        assert result == [None, None]

    def test_duplicate_solutions_become_none(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        shared_sol = _ValidSolution()
        call_count = [0]

        def _gen(*a, **kw):
            call_count[0] += 1
            return shared_sol  # same object → same hash → duplicate

        monkeypatch.setattr(dbmod, "_generate_single_solution_with_groups", _gen)
        monkeypatch.setattr(dbmod, "getKernelFileBase", lambda splitGSU, sol: "same_base")

        individuals = [{"a": 0}, {"a": 1}]
        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            individuals,
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        # First is kept; second has same base → None
        assert result[0] is not None
        assert result[1] is None

    def test_empty_individuals_returns_empty_list(self, monkeypatch):
        import Tensile.backends.ductile_backend as dbmod

        result = dbmod._generate_ga_solutions(
            types.SimpleNamespace(state={}),
            {},
            [],
            object(),
            types.SimpleNamespace(splitGSU=False),
            {"gfx942": {}},
        )
        assert result == []


# ---------------------------------------------------------------------------
# _benchmarkProblemType backend integration (patch-focused)
# ---------------------------------------------------------------------------

class _FakeBackendForBenchmark:
    def __init__(self):
        self.calls = []

    def run(self, backend_config, benchmark_config, benchmark_runner, cacheValid=False, buildOnly=False):
        self.calls.append((backend_config, benchmark_config, cacheValid, buildOnly))
        benchmark_runner([{}], isCached=False, buildOnly=False)


class _FakeBackendForBenchmarkCached:
    def __init__(self):
        self.calls = []

    def run(self, backend_config, benchmark_config, benchmark_runner, cacheValid=False, buildOnly=False):
        self.calls.append((backend_config, benchmark_config, cacheValid, buildOnly))
        benchmark_runner([{}], isCached=True, buildOnly=True)


def _minimal_problem_size_group_cfg():
    return {
        "ForkParameters": False,
        "BenchmarkFinalParameters": [{"ProblemSizes": [[64, 64, 1, 64]]}],
    }


def _make_fake_step():
    class _Step:
        problemSizes = types.SimpleNamespace(totalProblemSizes=1)
        factorDimArgs = types.SimpleNamespace(totalProblemSizes=1)
        biasTypeArgs = types.SimpleNamespace(totalProblemSizes=1)
        activationArgs = types.SimpleNamespace(totalProblemSizes=1)
        icacheFlushArgs = []
        forkParams = {}
        paramGroups = []
        constantParams = {}
        customKernels = []
        internalSupportParams = {}
        customKernelWildcard = False

        def isFinal(self):
            return True

        def __str__(self):
            return "step0"

    return _Step()


def test_benchmark_problem_type_passes_backend_config_and_runner(monkeypatch, tmp_path):
    fake_backend = _FakeBackendForBenchmark()
    fake_step = _make_fake_step()

    class _FakeBP:
        def __init__(self, *_args, **_kwargs):
            self.problemType = {"TileAwareSelection": False}
            self._step = fake_step

        def __iter__(self):
            yield self._step

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            assert idx == 0
            return self._step

    monkeypatch.setattr(BP, "BenchmarkProcess", _FakeBP)
    monkeypatch.setattr(BP.BackendFactory, "create", lambda _name: fake_backend)
    monkeypatch.setattr(BP, "_computeCacheKey", lambda _step: "abc123")
    monkeypatch.setattr(BP, "_resetCacheDir", lambda _d: None)
    monkeypatch.setattr(BP, "runClient", lambda *_a, **_kw: 0)
    monkeypatch.setattr(BP, "writeBenchmarkFiles", lambda *_a, **_kw: (["k.co"], "lib.yaml"))
    monkeypatch.setattr(BP.LibraryIO, "writeYAML", lambda *_a, **_kw: None)
    monkeypatch.setattr(BP.LibraryIO, "writeSolutions", lambda *_a, **_kw: None)
    monkeypatch.setattr(BP, "getSolutionNameMin", lambda *_a, **_kw: "s")
    monkeypatch.setattr(BP, "getKernelNameMin", lambda *_a, **_kw: "k")
    monkeypatch.setattr(BP, "startTime", 0.0)

    monkeypatch.setitem(BP.globalParameters, "ConfigPath", ["/tmp/cfg.yaml"])
    monkeypatch.setitem(BP.globalParameters, "ForceRedoBenchmarkProblems", True)
    monkeypatch.setitem(BP.globalParameters, "LibraryFormat", "yaml")

    backend_cfg = {"Name": "tensile", "Config": {"x": 1}}
    result_base, fails = BP._benchmarkProblemType(
        backendConfig=backend_cfg,
        problemTypeConfig={"OperationType": "GEMM", "DataType": "f32"},
        problemSizeGroupConfig=_minimal_problem_size_group_cfg(),
        problemSizeGroupIdx=0,
        outerBenchmarkIdx=0,
        configPath="/tmp/cfg.yaml",
        useCache=False,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        buildTmpPath=tmp_path / "build_tmp",
        benchmarkProblemsPath=tmp_path / "bench",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        gfxName="gfx942",
        isaInfoMap={"gfx942": {}},
        probSolMap={},
        buildOnly=False,
        solutionPoolIndex={},
    )

    assert fails == 0
    assert result_base is not None
    assert len(fake_backend.calls) == 1
    seen_backend_cfg, seen_bench_cfg, seen_cache, seen_build_only = fake_backend.calls[0]
    assert seen_backend_cfg == {"x": 1}
    assert seen_cache is False
    assert seen_build_only is False
    assert seen_bench_cfg["configName"] == "cfg"


def test_main_ignores_solution_pool_for_backend_without_support(monkeypatch, tmp_path):
    class _NoPoolBackend:
        def supports_solution_pool(self):
            return False

    monkeypatch.setattr(BP.BackendFactory, "create", lambda _name: _NoPoolBackend())
    monkeypatch.setattr(BP, "_loadSolutionPool", lambda _files: (_ for _ in ()).throw(AssertionError("should not load pool")))
    monkeypatch.setattr(BP, "_benchmarkProblemType", lambda **_kw: (str(tmp_path / "res"), 0))
    monkeypatch.setattr(BP, "printWarning", lambda *_a, **_kw: None)
    monkeypatch.setitem(BP.globalParameters, "ForceRedoBenchmarkProblems", True)
    monkeypatch.setitem(BP.globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(BP.globalParameters, "ExitOnFails", False)
    monkeypatch.setitem(BP.globalParameters, "ConfigPath", ["/tmp/cfg.yaml"])

    cfg = [[{"OperationType": "GEMM", "DataType": "s", "DestDataType": "s"}, _minimal_problem_size_group_cfg()]]

    BP.main(
        backend={"Name": "ductile", "Config": {}},
        config=cfg,
        useCache=False,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        outputPath=tmp_path / "out",
        buildTmpPath=tmp_path / "build_tmp",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        gfxName="gfx942",
        isaInfoMap={"gfx942": {}},
        probSolMap={},
        buildOnly=True,
        solutionPoolFiles=["pool.yaml"],
    )


def test_main_with_none_config_returns_early(monkeypatch):
    seen = []
    monkeypatch.setitem(BP.globalParameters, "ConfigPath", ["/tmp/cfg.yaml"])
    monkeypatch.setattr("builtins.print", lambda msg: seen.append(msg))

    BP.main(
        backend={"Name": "tensile", "Config": {}},
        config=None,
        useCache=False,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        outputPath=types.SimpleNamespace(__truediv__=lambda self, _x: self),
        buildTmpPath=types.SimpleNamespace(__truediv__=lambda self, _x: self),
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        gfxName="gfx942",
        isaInfoMap={"gfx942": {}},
        probSolMap={},
        buildOnly=True,
        solutionPoolFiles=None,
    )

    assert any("No config specified" in s for s in seen)


def test_main_invalid_backend_type_exits(monkeypatch, tmp_path):
    # Monkeypatch printExit in both BP and backends.config modules
    exit_func = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    monkeypatch.setattr(BP, "printExit", exit_func)
    from Tensile.backends import config as backend_config_module
    monkeypatch.setattr(backend_config_module, "printExit", exit_func)

    with pytest.raises(RuntimeError, match="'Backend' must be a dictionary"):
        BP.main(
            backend="bad",
            config=[],
            useCache=False,
            asmToolchain=types.SimpleNamespace(assembler=object()),
            srcToolchain=types.SimpleNamespace(compiler="cc"),
            cCompiler="cc",
            outputPath=tmp_path / "out",
            buildTmpPath=tmp_path / "build_tmp",
            debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
            deviceId=0,
            gfxName="gfx942",
            isaInfoMap={"gfx942": {}},
            probSolMap={},
            buildOnly=True,
            solutionPoolFiles=None,
        )


def test_main_backend_config_none_is_normalized(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(BP, "_benchmarkProblemType", lambda **kw: (captured.setdefault("backend", kw["backendConfig"]), ("results.csv", 0))[1])
    monkeypatch.setitem(BP.globalParameters, "ForceRedoBenchmarkProblems", True)
    monkeypatch.setitem(BP.globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(BP.globalParameters, "ExitOnFails", False)
    monkeypatch.setattr(BP, "writeBenchmarkFiles", lambda *args, **kwargs: None)

    cfg = [[{"OperationType": "GEMM", "DataType": "s", "DestDataType": "s"}, _minimal_problem_size_group_cfg()]]

    BP.main(
        backend={"Name": "tensile", "Config": None},
        config=cfg,
        useCache=False,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        outputPath=tmp_path / "out",
        buildTmpPath=tmp_path / "build_tmp",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        gfxName="gfx942",
        isaInfoMap={"gfx942": {}},
        probSolMap={},
        buildOnly=True,
        solutionPoolFiles=None,
    )

    assert captured["backend"]["Config"] == {}


def test_main_backend_config_invalid_type_exits(monkeypatch, tmp_path):
    # Monkeypatch printExit in both BP and backends.config modules
    exit_func = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    monkeypatch.setattr(BP, "printExit", exit_func)
    from Tensile.backends import config as backend_config_module
    monkeypatch.setattr(backend_config_module, "printExit", exit_func)

    with pytest.raises(RuntimeError, match="'Backend.Config' must be a dictionary"):
        BP.main(
            backend={"Name": "tensile", "Config": "bad"},
            config=[],
            useCache=False,
            asmToolchain=types.SimpleNamespace(assembler=object()),
            srcToolchain=types.SimpleNamespace(compiler="cc"),
            cCompiler="cc",
            outputPath=tmp_path / "out",
            buildTmpPath=tmp_path / "build_tmp",
            debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
            deviceId=0,
            gfxName="gfx942",
            isaInfoMap={"gfx942": {}},
            probSolMap={},
            buildOnly=True,
            solutionPoolFiles=None,
        )


def test_main_loads_solution_pool_when_backend_supports_it(monkeypatch, tmp_path):
    class _PoolBackend:
        def supports_solution_pool(self):
            return True

    loaded = {}
    monkeypatch.setattr(BP.BackendFactory, "create", lambda _name: _PoolBackend())
    monkeypatch.setattr(BP, "_loadSolutionPool", lambda files: loaded.setdefault("files", files) or {"k": []})
    monkeypatch.setattr(BP, "_benchmarkProblemType", lambda **_kw: (str(tmp_path / "res"), 0))
    monkeypatch.setitem(BP.globalParameters, "ForceRedoBenchmarkProblems", True)
    monkeypatch.setitem(BP.globalParameters, "CSVExportWinner", False)
    monkeypatch.setitem(BP.globalParameters, "ExitOnFails", False)
    monkeypatch.setitem(BP.globalParameters, "ConfigPath", ["/tmp/cfg.yaml"])

    cfg = [[{"OperationType": "GEMM", "DataType": "s", "DestDataType": "s"}, _minimal_problem_size_group_cfg()]]

    BP.main(
        backend={"Name": "tensile", "Config": {}},
        config=cfg,
        useCache=False,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        outputPath=tmp_path / "out",
        buildTmpPath=tmp_path / "build_tmp",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        gfxName="gfx942",
        isaInfoMap={"gfx942": {}},
        probSolMap={},
        buildOnly=True,
        solutionPoolFiles=["pool.yaml"],
    )

    assert loaded["files"] == ["pool.yaml"]


def test_benchmark_problem_type_cached_runner_build_only_path(monkeypatch, tmp_path):
    fake_backend = _FakeBackendForBenchmarkCached()
    fake_step = _make_fake_step()

    class _FakeBP:
        def __init__(self, *_args, **_kwargs):
            self.problemType = {"TileAwareSelection": False}
            self._step = fake_step

        def __iter__(self):
            yield self._step

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            assert idx == 0
            return self._step

    monkeypatch.setattr(BP, "BenchmarkProcess", _FakeBP)
    monkeypatch.setattr(BP.BackendFactory, "create", lambda _name: fake_backend)
    monkeypatch.setattr(BP, "_computeCacheKey", lambda _step: "abc123")
    monkeypatch.setattr(BP, "_loadCacheIfMatches", lambda *_a, **_kw: {"CodeObjectFiles": ["k.co"], "LibraryFile": "lib.yaml"})
    monkeypatch.setattr(BP, "runClient", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("runClient should not run in buildOnly")))
    monkeypatch.setattr(BP, "writeClientConfigIni", lambda *_a, **_kw: None)
    monkeypatch.setattr(BP.LibraryIO, "writeSolutions", lambda *_a, **_kw: None)
    monkeypatch.setattr(BP, "ProblemType", lambda *_a, **_kw: object())
    monkeypatch.setattr(BP, "ContractionsProblemType", types.SimpleNamespace(FromOriginalState=lambda *_a, **_kw: object()))
    monkeypatch.setattr(BP, "tensileLibraryFile", lambda *_a, **_kw: str(tmp_path / "dummy_lib.yaml"))
    monkeypatch.setattr(BP.os.path, "isfile", lambda _p: True)
    monkeypatch.setattr(BP, "startTime", 0.0)
    monkeypatch.setitem(BP.globalParameters, "ConfigPath", ["/tmp/cfg.yaml"])
    monkeypatch.setitem(BP.globalParameters, "ForceRedoBenchmarkProblems", False)
    monkeypatch.setitem(BP.globalParameters, "LibraryFormat", "yaml")

    result_base, fails = BP._benchmarkProblemType(
        backendConfig={"Name": "tensile", "Config": {"x": 2}},
        problemTypeConfig={"OperationType": "GEMM", "DataType": "f32"},
        problemSizeGroupConfig=_minimal_problem_size_group_cfg(),
        problemSizeGroupIdx=0,
        outerBenchmarkIdx=0,
        configPath="/tmp/cfg.yaml",
        useCache=True,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        cCompiler="cc",
        buildTmpPath=tmp_path / "build_tmp",
        benchmarkProblemsPath=tmp_path / "bench",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        gfxName="gfx942",
        isaInfoMap={"gfx942": {}},
        probSolMap={},
        buildOnly=False,
        solutionPoolIndex={},
    )

    assert fails == 0
    assert result_base is not None
    assert len(fake_backend.calls) == 1
