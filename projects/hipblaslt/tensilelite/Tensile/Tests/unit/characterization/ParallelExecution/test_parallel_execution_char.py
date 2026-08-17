################################################################################
# Characterization tests for Tensile.ParallelExecution
#
# ADD-ONLY: pins the multi-GPU client orchestration helpers. Subprocess (rocm-smi
# / hipInfo / client launch), the ClientExecutionLock and globalParameters are
# stubbed so control flow and file munging are pinned without real GPUs.
################################################################################
import importlib
import subprocess

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.ParallelExecution")


# ---------------------------------------------------------------------------
# detectAvailableGpus
# ---------------------------------------------------------------------------
class _Run:
    def __init__(self, rc, out):
        self.returncode = rc
        self.stdout = out


def test_detect_gpus_from_amd_smi(monkeypatch):
    # `amd-smi list --json` returns one JSON object per GPU.
    monkeypatch.setattr(
        M.subprocess, "run", lambda *a, **k: _Run(0, '[{"gpu": 0}, {"gpu": 1}]')
    )
    assert M.detectAvailableGpus() == 2


def test_detect_gpus_fallback_hipinfo(monkeypatch):
    calls = []

    def fake_run(args, **k):
        calls.append(args[0])
        if args[0] == "amd-smi":
            return _Run(1, "")  # fail -> fallback
        return _Run(0, "Number of devices: 4\n")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    assert M.detectAvailableGpus() == 4
    assert calls == ["amd-smi", "hipInfo"]


def test_detect_gpus_default_one(monkeypatch):
    def boom(*a, **k):
        raise OSError("no such tool")

    monkeypatch.setattr(M.subprocess, "run", boom)
    assert M.detectAvailableGpus() == 1


# ---------------------------------------------------------------------------
# countProblemsInConfig
# ---------------------------------------------------------------------------
def test_count_problems(tmp_path):
    cfg = tmp_path / "c.cfg"
    cfg.write_text("problem-size=1\nother=x\n problem-size=2\nproblem-size=3\n")
    assert M.countProblemsInConfig(str(cfg)) == 3


# ---------------------------------------------------------------------------
# createPerGpuConfig
# ---------------------------------------------------------------------------
def test_create_per_gpu_config(tmp_path):
    orig = tmp_path / "orig.cfg"
    orig.write_text("device-idx=0\nresults-file=/old/res.csv\nkeepme=1\n")
    cfgPath, resPath = M.createPerGpuConfig(str(orig), 2, 10, 5, str(tmp_path))
    text = open(cfgPath).read()
    assert "device-idx=2" in text
    assert f"results-file={resPath}" in text
    assert "keepme=1" in text
    assert "problem-start-idx=10" in text
    assert "num-problems=5" in text


# ---------------------------------------------------------------------------
# mergeResultsCsv
# ---------------------------------------------------------------------------
def test_merge_results_csv(tmp_path):
    r0 = tmp_path / "r0.csv"
    r1 = tmp_path / "r1.csv"
    r0.write_text("head\na\nb\n")
    r1.write_text("head\nc\n")
    out = tmp_path / "out.csv"
    M.mergeResultsCsv([str(r0), str(tmp_path / "missing.csv"), str(r1)], str(out))
    # header once, then data rows from both (r1's header dropped)
    assert out.read_text() == "head\na\nb\nc\n"


def test_merge_results_csv_skips_empty(tmp_path):
    empty = tmp_path / "e.csv"
    empty.write_text("")
    good = tmp_path / "g.csv"
    good.write_text("head\nx\n")
    out = tmp_path / "out.csv"
    M.mergeResultsCsv([str(empty), str(good)], str(out))
    assert out.read_text() == "head\nx\n"


# ---------------------------------------------------------------------------
# runClientParallel
# ---------------------------------------------------------------------------
class _Proc:
    def __init__(self, returncode=0, raise_timeout=False):
        self.returncode = returncode
        self._raise_timeout = raise_timeout
        self.killed = False

    def wait(self, timeout=None):
        if self._raise_timeout:
            self._raise_timeout = False
            raise subprocess.TimeoutExpired(cmd="client", timeout=timeout)
        return self.returncode

    def kill(self):
        self.killed = True

    def poll(self):
        return self.returncode


class _DummyLock:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def _lock_and_gp(monkeypatch):
    monkeypatch.setattr(M, "ClientExecutionLock", _DummyLock)
    # ensure the lock-path key exists
    monkeypatch.setitem(M.globalParameters, "ClientExecutionLockPath", "/tmp/lock")


def _write_cfg(path, n, results="/res/out.csv"):
    lines = ["device-idx=0\n", f"results-file={results}\n"]
    lines += [f"problem-size={i}\n" for i in range(n)]
    path.write_text("".join(lines))


def test_run_client_parallel_basic(tmp_path, monkeypatch, _lock_and_gp):
    cfg = tmp_path / "c.cfg"
    resOut = tmp_path / "final.csv"
    _write_cfg(cfg, 4, results=str(resOut))
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Proc(returncode=0))
    rc = M.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=2, timingEnabled=True,
        getClientExecutablePath=lambda: "/bin/client",
    )
    assert rc == 0
    # merged results file created; per-GPU dir cleaned up
    assert resOut.exists()
    assert not (tmp_path / "parallel_gpu").exists()


def test_run_client_parallel_zero_problems_skipped(tmp_path, monkeypatch, _lock_and_gp, capsys):
    cfg = tmp_path / "c.cfg"
    _write_cfg(cfg, 0)
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Proc())
    rc = M.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=2, timingEnabled=False,
        getClientExecutablePath=lambda: "/bin/client",
    )
    assert rc == 0
    assert "No problems found" in capsys.readouterr().out


def test_run_client_parallel_nonzero_return_code(tmp_path, monkeypatch, _lock_and_gp):
    cfg = tmp_path / "c.cfg"
    _write_cfg(cfg, 2, results=str(tmp_path / "o.csv"))
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Proc(returncode=3))
    rc = M.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=2, timingEnabled=False,
        getClientExecutablePath=lambda: "/bin/client",
    )
    assert rc == 3


def test_run_client_parallel_timeout_kills(tmp_path, monkeypatch, _lock_and_gp):
    cfg = tmp_path / "c.cfg"
    _write_cfg(cfg, 1, results=str(tmp_path / "o.csv"))
    procs = []

    def make(*a, **k):
        p = _Proc(returncode=0, raise_timeout=True)
        procs.append(p)
        return p

    monkeypatch.setattr(M.subprocess, "Popen", make)
    M.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=1, timingEnabled=False,
        getClientExecutablePath=lambda: "/bin/client",
    )
    assert procs and procs[0].killed is True


def test_run_client_parallel_no_results_file_warns(tmp_path, monkeypatch, _lock_and_gp, capsys):
    cfg = tmp_path / "c.cfg"
    # no results-file= line -> preserve per-GPU dir + warn
    cfg.write_text("device-idx=0\nproblem-size=0\nproblem-size=1\n")
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **k: _Proc())
    M.runClientParallel(
        str(tmp_path), [str(cfg)], numGpus=1, timingEnabled=False,
        getClientExecutablePath=lambda: "/bin/client",
    )
    out = capsys.readouterr().out
    assert "No results-file found" in out
    assert (tmp_path / "parallel_gpu").exists()  # preserved
