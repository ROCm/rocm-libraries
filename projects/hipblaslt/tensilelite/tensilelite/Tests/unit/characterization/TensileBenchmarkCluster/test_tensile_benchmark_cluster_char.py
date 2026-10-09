################################################################################
# Characterization tests for Tensile.TensileBenchmarkCluster
#
# ADD-ONLY: pins the SLURM cluster-benchmark orchestrator. Subprocess/docker/
# gzip/file-template work is stubbed at the module seams (subprocess, gzip,
# ScriptWriter, BenchmarkSplitter, mergePartialLogics) so we drive control flow
# and argv/config construction without a cluster.
#
# Pinned quirk: TensileBenchmarkCluster.__parseArgs IGNORES its `cmdlineArgs`
# parameter and parses sys.argv directly (argparse .parse_args() with no list).
# Tests therefore set sys.argv to control parsing.
################################################################################
import importlib
import os
import sys

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileBenchmarkCluster")
TBC = M.TensileBenchmarkCluster
SLURM = M.BenchmarkImplSLURM


@pytest.fixture
def make_cluster(monkeypatch):
    """Build a TensileBenchmarkCluster with a controlled argv."""

    def _make(extra=None):
        argv = ["prog", "/logic/path", "/deploy/path"]
        if extra:
            argv += extra
        monkeypatch.setattr(sys, "argv", argv)
        return TBC(sys.argv[1:])

    return _make


# ---------------------------------------------------------------------------
# __parseArgs (via sys.argv) + full config initialization
# ---------------------------------------------------------------------------
def test_default_config_and_accessors(make_cluster):
    c = make_cluster()
    assert c.config()["BenchmarkLogicPath"] == "/logic/path"
    assert c.baseDir() == "/deploy/path"
    assert c.tasksDir() == os.path.join("/deploy/path", "Tasks")
    assert c.imageDir() == os.path.join("/deploy/path", "Image")
    assert c.resultsDir() == os.path.join("/deploy/path", "Results")
    assert c.finalLogicDir() == os.path.join("/deploy/path", "Results", "Final")
    assert c.logsDir() == os.path.join("/deploy/path", "Logs")
    assert c.rootTensileDir()  # non-empty path string
    # default: all three workflow steps enabled
    assert c.workflowSteps() == (True, True, True)


def test_deploy_only_steps(make_cluster):
    c = make_cluster(["--deploy-only"])
    assert c.workflowSteps() == (True, False, False)


def test_run_only_steps(make_cluster):
    c = make_cluster(["--run-only"])
    assert c.workflowSteps() == (False, True, False)


def test_results_only_raises_due_to_boolop_bug(monkeypatch):
    # LATENT BUG (pinned): --results-only sets only RunResultsStep True, i.e. the
    # *third* operand of the constraint "RunDeployStep or RunBenchmarkStep or
    # RunResultsStep". ExpressionEvaluator's BoolOp handler
    # (Configuration.py:651-652) evaluates only values[0]/values[1] and ignores
    # values[2:], so the 3-way `or` reduces to `False or False` -> constraint
    # fails -> AssertionError during construction. Thus --results-only ALONE is
    # currently unusable. See DECISIONS D12.
    monkeypatch.setattr(sys, "argv", ["prog", "/logic", "/deploy", "--results-only"])
    with pytest.raises(AssertionError, match="Constraint evaluation failed"):
        TBC(sys.argv[1:])


def test_run_and_results_only_steps(make_cluster):
    c = make_cluster(["--run-and-results-only"])
    # deploy disabled; benchmark + results enabled
    assert c.workflowSteps() == (False, True, True)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["prog", "/logic", "/deploy", "--cluster-backend", "k8s"]
    )
    with pytest.raises(NotImplementedError):
        TBC(sys.argv[1:])


def test_benchmark_parameters_override_recognized(make_cluster, capsys):
    # BenchmarkTaskSize is a recognized key; override it via --benchmark-parameters
    c = make_cluster(["--benchmark-parameters", "BenchmarkTaskSize=5"])
    assert c.config()["BenchmarkTaskSize"] == 5


def test_benchmark_parameters_unrecognized_warns(make_cluster, capsys):
    c = make_cluster(["--benchmark-parameters", "NotAKey=1"])
    out = capsys.readouterr().out
    assert "unrecognised" in out


def test_parse_args_ignores_passed_list(monkeypatch):
    # The documented quirk: the cmdlineArgs argument is ignored; sys.argv wins.
    monkeypatch.setattr(sys, "argv", ["prog", "/real/logic", "/real/deploy"])
    c = TBC(["bogus", "values", "ignored"])
    assert c.config()["BenchmarkLogicPath"] == "/real/logic"


# ---------------------------------------------------------------------------
# initializeConfig (SLURM backend, pure config)
# ---------------------------------------------------------------------------
def test_slurm_initialize_config_values(make_cluster):
    c = make_cluster()
    cfg = c.config()
    docker = cfg["SLURM"]["DOCKER"]
    assert docker["DockerImageName"] == "tensile-tuning-cluster-executable"
    assert docker["TensileBranch"] == "develop"
    scripts = cfg["SLURM"]["SCRIPTS"]
    assert scripts["JobScriptName"] == "runBenchmark.sh"
    assert scripts["TaskScriptName"] == "enqueueTask.sh"


# ---------------------------------------------------------------------------
# ensurePath
# ---------------------------------------------------------------------------
def test_ensure_path_creates_and_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b"
    assert TBC.ensurePath(str(target)) == str(target)
    assert target.is_dir()
    # second call hits the OSError-swallow branch
    assert TBC.ensurePath(str(target)) == str(target)


# ---------------------------------------------------------------------------
# preInvoke / postInvoke are no-ops
# ---------------------------------------------------------------------------
def test_pre_and_post_invoke_noops():
    assert SLURM.preInvokeBenchmark({}) is None
    assert SLURM.postInvokeBenchmark({}) is None


# ---------------------------------------------------------------------------
# main() driver — stub the three private orchestration steps
# ---------------------------------------------------------------------------
def _patch_steps(c, monkeypatch, log):
    monkeypatch.setattr(
        c, "_TensileBenchmarkCluster__generateClusterBenchmark", lambda: log.append("gen")
    )
    monkeypatch.setattr(
        c, "_TensileBenchmarkCluster__runClusterBenchmark", lambda: log.append("run")
    )
    monkeypatch.setattr(
        c,
        "_TensileBenchmarkCluster__combineClusterBenchmarkResults",
        lambda: log.append("combine"),
    )


def test_main_runs_all_steps(make_cluster, monkeypatch):
    c = make_cluster()
    log = []
    _patch_steps(c, monkeypatch, log)
    c.main()
    assert log == ["gen", "run", "combine"]


def test_main_deploy_only(make_cluster, monkeypatch):
    c = make_cluster(["--deploy-only"])
    log = []
    _patch_steps(c, monkeypatch, log)
    c.main()
    assert log == ["gen"]


# ---------------------------------------------------------------------------
# __runClusterBenchmark — delegates to backend (3 hooks)
# ---------------------------------------------------------------------------
def test_run_cluster_benchmark_delegates(make_cluster, monkeypatch):
    c = make_cluster()
    seen = []
    for hook in ("preInvokeBenchmark", "invokeBenchmark", "postInvokeBenchmark"):
        monkeypatch.setattr(
            c._backendImpl, hook, classmethod(lambda cls, cfg, h=hook: seen.append(h))
        )
    c._TensileBenchmarkCluster__runClusterBenchmark()
    assert seen == ["preInvokeBenchmark", "invokeBenchmark", "postInvokeBenchmark"]


# ---------------------------------------------------------------------------
# __generateClusterBenchmark — stub splitter + backend generate
# ---------------------------------------------------------------------------
def test_generate_cluster_benchmark(make_cluster, monkeypatch, tmp_path):
    c = make_cluster(["--benchmark-parameters", "BenchmarkBaseDir=" + repr(str(tmp_path))])
    # redirect all dirs under tmp_path
    cfg = c.config()
    cfg["BenchmarkBaseDir"] = str(tmp_path / "base")
    cfg["BenchmarkTasksDir"] = str(tmp_path / "tasks")
    cfg["BenchmarkImageDir"] = str(tmp_path / "img")
    cfg["BenchmarkResultsDir"] = str(tmp_path / "res")
    cfg["BenchmarkFinalLogicDir"] = str(tmp_path / "res" / "final")
    cfg["BenchmarkLogsDir"] = str(tmp_path / "logs")

    splits = []
    monkeypatch.setattr(
        M.BenchmarkSplitter,
        "splitBenchmarkBySizes",
        staticmethod(lambda *a, **k: splits.append((a, k))),
    )
    gen = []
    monkeypatch.setattr(
        M.BenchmarkImplSLURM, "generateBenchmark", classmethod(lambda cls, cfg: gen.append(cfg))
    )
    c._TensileBenchmarkCluster__generateClusterBenchmark()
    assert (tmp_path / "tasks").is_dir()
    assert len(splits) == 1
    assert len(gen) == 1


# ---------------------------------------------------------------------------
# __combineClusterBenchmarkResults — build a fake results tree
# ---------------------------------------------------------------------------
def test_combine_results_gathers_files(make_cluster, monkeypatch, tmp_path):
    c = make_cluster()
    resultsDir = tmp_path / "res"
    finalDir = tmp_path / "final"
    # one partial result with a 3_LibraryLogic/foo.yaml
    part = resultsDir / "part0" / "3_LibraryLogic"
    part.mkdir(parents=True)
    (part / "foo.yaml").write_text("logic")
    c.config()["BenchmarkResultsDir"] = str(resultsDir)
    c.config()["BenchmarkFinalLogicDir"] = str(finalDir)

    captured = {}
    monkeypatch.setattr(
        M, "mergePartialLogics", lambda files, fd, force, trim: captured.update(
            files=files, fd=fd, force=force, trim=trim
        )
    )
    c._TensileBenchmarkCluster__combineClusterBenchmarkResults()
    assert captured["files"] == [str(part / "foo.yaml")]
    assert captured["fd"] == str(finalDir)


def test_combine_results_warns_on_inconsistent(make_cluster, monkeypatch, tmp_path, capsys):
    c = make_cluster()
    resultsDir = tmp_path / "res"
    # a 3_LibraryLogic dir with NO files -> dirs(1) != files(0) -> warning
    (resultsDir / "part0" / "3_LibraryLogic").mkdir(parents=True)
    c.config()["BenchmarkResultsDir"] = str(resultsDir)
    c.config()["BenchmarkFinalLogicDir"] = str(tmp_path / "final")
    monkeypatch.setattr(M, "mergePartialLogics", lambda *a, **k: None)
    c._TensileBenchmarkCluster__combineClusterBenchmarkResults()
    assert "inconsistent number of expected results" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# generateBenchmark / invokeBenchmark / script + container builders
# (stub subprocess, gzip, ScriptWriter)
# ---------------------------------------------------------------------------
def test_slurm_generate_benchmark(make_cluster, monkeypatch, tmp_path):
    c = make_cluster()
    cfg = c.config()
    cfg["BenchmarkBaseDir"] = str(tmp_path / "base")
    cfg["BenchmarkTasksDir"] = str(tmp_path / "tasks")
    cfg["BenchmarkImageDir"] = str(tmp_path / "img")
    cfg["BenchmarkLogsDir"] = str(tmp_path / "logs")
    for d in ("base", "tasks", "img", "logs"):
        (tmp_path / d).mkdir()
    # one task config to be moved into a subdir
    (tmp_path / "tasks" / "cfg0000.yaml").write_text("x")

    called = []
    monkeypatch.setattr(
        SLURM,
        "_BenchmarkImplSLURM__createTensileBenchmarkContainer",
        staticmethod(lambda *a, **k: called.append("container")),
    )
    monkeypatch.setattr(M.ScriptWriter, "writeBenchmarkJobScript", staticmethod(lambda *a: None))
    monkeypatch.setattr(M.ScriptWriter, "writeBenchmarkTaskScript", staticmethod(lambda *a: None))
    monkeypatch.setattr(M.ScriptWriter, "writeBenchmarkNodeScript", staticmethod(lambda *a: None))

    SLURM.generateBenchmark(cfg)
    assert called == ["container"]
    # the task config was relocated into its own subdir
    assert (tmp_path / "tasks" / "cfg0000" / "cfg0000.yaml").is_file()


def test_slurm_invoke_benchmark(make_cluster, monkeypatch, tmp_path):
    c = make_cluster()
    cfg = c.config()
    cfg["BenchmarkBaseDir"] = str(tmp_path / "base")
    cfg["BenchmarkTasksDir"] = str(tmp_path / "tasks")
    cfg["BenchmarkImageDir"] = str(tmp_path / "img")
    cfg["BenchmarkResultsDir"] = str(tmp_path / "res")
    cfg["BenchmarkLogsDir"] = str(tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    cmds = []
    monkeypatch.setattr(M.subprocess, "check_call", lambda args, **k: cmds.append(args))
    SLURM.invokeBenchmark(cfg)
    assert len(cmds) == 1
    # first token is the runBenchmark.sh path under baseDir
    assert cmds[0][0].endswith("runBenchmark.sh")


def test_slurm_create_container(monkeypatch, tmp_path):
    logDir = tmp_path / "logs"
    outDir = tmp_path / "out"
    logDir.mkdir()
    outDir.mkdir()

    calls = {"check_call": 0}
    monkeypatch.setattr(M.subprocess, "check_call", lambda *a, **k: calls.__setitem__("check_call", calls["check_call"] + 1))

    class _Proc:
        def __init__(self):
            self.stdout = self

        def read(self):
            return b"image-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Proc())

    class _Zip:
        def __init__(self):
            self.data = b""

        def write(self, b):
            self.data += b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    zips = []

    def _gzip_open(path, mode):
        z = _Zip()
        zips.append((path, z))
        return z

    monkeypatch.setattr(M.gzip, "open", _gzip_open)

    SLURM._BenchmarkImplSLURM__createTensileBenchmarkContainer(
        "base-image",
        "/path/Dockerfile",
        "myimage:TEST",
        str(outDir),
        str(logDir),
        "fork",
        "branch",
        "commit",
    )
    assert calls["check_call"] == 1  # docker build
    assert zips and zips[0][1].data == b"image-bytes"
    # archive named after image base name
    assert zips[0][0].endswith("myimage.tar.gz")


# ---------------------------------------------------------------------------
# module-level main()
# ---------------------------------------------------------------------------
def test_module_main_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "/logic", "/deploy"])
    ran = []
    monkeypatch.setattr(TBC, "main", lambda self: ran.append(True))
    M.main()
    assert ran == [True]
