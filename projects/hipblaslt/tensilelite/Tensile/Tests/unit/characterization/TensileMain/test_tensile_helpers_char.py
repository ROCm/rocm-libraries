################################################################################
# Characterization tests for Tensile.Tensile — helper layer.
#
# ADD-ONLY. Tensile.py's executeStepsInConfig / Tensile() are the top-level
# build+benchmark orchestrators (BenchmarkProblems / ClientWriter / LibraryLogic
# + GPU), out of scope here. This suite pins the pure / stubbable helpers:
# addCommonArguments, argUpdatedGlobalParameters, get_gpu_max_frequency_smi,
# get_user_max_frequency, store_max_frequency, restore_prob_sol_map,
# TensileConfigPath/TensileTestPath, and the thin entry-point wrappers.
################################################################################
import argparse
import importlib
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.Tensile")


# ---------------------------------------------------------------------------
# addCommonArguments (+ splitExtraParameters)
# ---------------------------------------------------------------------------
def test_add_common_arguments_defaults():
    p = argparse.ArgumentParser()
    M.addCommonArguments(p)
    args = p.parse_args(["--code-object-version", "5"])
    assert args.device == 0
    assert args.CodeObjectVersion == "5"
    assert args.LogicFormat == "yaml"
    assert args.global_parameters == []


def test_add_common_arguments_global_parameters_eval():
    p = argparse.ArgumentParser()
    M.addCommonArguments(p)
    args = p.parse_args(["--global-parameters", "Foo=5", "Bar='baz'"])
    assert ("Foo", 5) in args.global_parameters
    assert ("Bar", "baz") in args.global_parameters


# ---------------------------------------------------------------------------
# argUpdatedGlobalParameters
# ---------------------------------------------------------------------------
def _args(**over):
    base = dict(
        platform=None, RuntimeLanguage=None, CodeObjectVersion=None, debug=False,
        client_lock=None, prebuilt_client=None, MXScaleFormat=0, global_parameters=[],
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_arg_updated_global_parameters_all_overrides(monkeypatch):
    monkeypatch.delenv("PyTestBuildArchNames", raising=False)
    args = _args(
        platform=2, RuntimeLanguage="HIP", CodeObjectVersion="5", debug=True,
        client_lock="/lock", prebuilt_client="/client", MXScaleFormat=1,
        global_parameters=[("K", "V")],
    )
    rv = M.argUpdatedGlobalParameters(args)
    assert rv["Platform"] == 2
    assert rv["RuntimeLanguage"] == "HIP"
    assert rv["CodeObjectVersion"] == "5"
    assert rv["CMakeBuildType"] == "Debug"
    assert rv["ClientExecutionLockPath"] == "/lock"
    assert rv["PrebuiltClient"] == "/client"
    assert rv["MXScaleFormat"] == 1
    assert rv["K"] == "V"


def test_arg_updated_global_parameters_empty(monkeypatch):
    monkeypatch.delenv("PyTestBuildArchNames", raising=False)
    assert M.argUpdatedGlobalParameters(_args()) == {}


def test_arg_updated_global_parameters_pytest_arch_env(monkeypatch):
    monkeypatch.setenv("PyTestBuildArchNames", "gfx942")
    rv = M.argUpdatedGlobalParameters(_args())
    assert rv["Architecture"] == "gfx942"


# ---------------------------------------------------------------------------
# get_gpu_max_frequency_smi
# ---------------------------------------------------------------------------
def test_gpu_max_freq_smi_parses(monkeypatch):
    # `amd-smi metric --clock --json`: max GFX (sclk) max_clk across engines.
    out = (
        '{"gpu_data": [{"clock": {'
        '"gfx_0": {"max_clk": {"value": 700}}, '
        '"gfx_1": {"max_clk": {"value": 1200}}, '
        '"mem_0": {"max_clk": {"value": 1900}}'
        '}}]}'
    )
    monkeypatch.setattr(
        M.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr="")
    )
    assert M.get_gpu_max_frequency_smi(0) == 1200


def test_gpu_max_freq_smi_error_returncode(monkeypatch, capsys):
    monkeypatch.setattr(
        M.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom")
    )
    assert M.get_gpu_max_frequency_smi(0) is None
    assert "Error running amd-smi" in capsys.readouterr().out


def test_gpu_max_freq_smi_exception(monkeypatch):
    def boom(*a, **k):
        raise OSError("no smi")

    monkeypatch.setattr(M.subprocess, "run", boom)
    assert M.get_gpu_max_frequency_smi(0) is None


# ---------------------------------------------------------------------------
# get_user_max_frequency
# ---------------------------------------------------------------------------
def test_user_max_freq_non_tty_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(M.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert M.get_user_max_frequency() is None
    assert "non-interactive" in capsys.readouterr().out


def test_user_max_freq_valid(monkeypatch):
    monkeypatch.setattr(M.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "1500")
    assert M.get_user_max_frequency() == 1500


def test_user_max_freq_retries_on_bad_then_nonpositive_then_good(monkeypatch):
    monkeypatch.setattr(M.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter(["abc", "-5", "900"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert M.get_user_max_frequency() == 900


# ---------------------------------------------------------------------------
# store_max_frequency
# ---------------------------------------------------------------------------
def test_store_max_frequency(monkeypatch):
    monkeypatch.delenv("MAX_FREQ", raising=False)
    assert M.store_max_frequency(1234) is True
    assert os.environ["MAX_FREQ"] == "1234"


# ---------------------------------------------------------------------------
# restore_prob_sol_map
# ---------------------------------------------------------------------------
def _log_line(prob="0/2", sol="0/3", us="12.5"):
    # prefix"(quoted)"suffix; after de-quote+split(',') tokens[10] must be `us`
    return f'r,{prob},{sol},Contraction,"(1,2,3,4)",f5,f6,f7,f8,f9,{us},extra'


def test_restore_prob_sol_map_completed(tmp_path):
    log = tmp_path / "tune.log"
    log.write_text(
        "noise\n"
        "run,problem-progress,whatever\n"
        + _log_line(us="12.5") + "\n"
        + "####\n"                      # postProblem -> prob 0 completed
        + "clientExit done\n"
    )
    result = M.restore_prob_sol_map(str(log))
    assert result == {0: 0}


def test_restore_prob_sol_map_incomplete_last_prob_removed(tmp_path):
    log = tmp_path / "tune.log"
    # no #### line -> prob 0 never completes -> popped from map
    log.write_text(
        "run,problem-progress,x\n"
        + _log_line() + "\n"
        + "clientExit\n"
    )
    assert M.restore_prob_sol_map(str(log)) == {}


def test_restore_prob_sol_map_keeps_faster_winner(tmp_path):
    log = tmp_path / "tune.log"
    log.write_text(
        "run,problem-progress,x\n"
        + _log_line(sol="0/3", us="20.0") + "\n"   # prob0 sol0 @20us
        + _log_line(sol="1/3", us="5.0") + "\n"    # prob0 sol1 @5us  -> winner
        + "####\n"
        + "clientExit\n"
    )
    assert M.restore_prob_sol_map(str(log)) == {0: 1}


def test_restore_prob_sol_map_missing_file(tmp_path, capsys):
    assert M.restore_prob_sol_map(str(tmp_path / "nope.log")) == {}
    assert "was not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# path helpers + entry points
# ---------------------------------------------------------------------------
def test_config_and_test_paths():
    assert M.TensileConfigPath("a.yaml").endswith(os.path.join("Configs", "a.yaml"))
    assert M.TensileTestPath("b").endswith(os.path.join("Tests", "b"))


@pytest.mark.parametrize("fn,frag", [
    ("TensileROCBLASSGEMM", "rocblas_sgemm.yaml"),
    ("TensileROCBLASDGEMM", "rocblas_dgemm.yaml"),
    ("TensileROCBLASCGEMM", "rocblas_cgemm.yaml"),
    ("TensileROCBLASZGEMM", "rocblas_zgemm.yaml"),
    ("TensileSGEMM5760", "sgemm_5760.yaml"),
])
def test_entry_points_call_tensile(monkeypatch, fn, frag):
    calls = []
    monkeypatch.setattr(M, "Tensile", lambda argv: calls.append(argv))
    getattr(M, fn)()
    assert len(calls) == 1
    assert frag in calls[0][0]
    assert calls[0][1] == "."


def test_main_calls_tensile(monkeypatch):
    monkeypatch.setattr(M.sys, "argv", ["prog", "cfg.yaml", "out"])
    calls = []
    monkeypatch.setattr(M, "Tensile", lambda argv: calls.append(argv))
    M.main()
    assert calls == [["cfg.yaml", "out"]]
