################################################################################
# Characterization tests for Tensile.TensileLogic — ParseArguments + small bits.
#
# ADD-ONLY. TensileLogic/Run._runChecks/_setup/main are the validation driver
# (resistance). This pins TensileLogic/ParseArguments.parseArguments, the Check
# NamedTuple, _progress_loop's stop path, and the TensileCreateLibrary/__main__
# import shim.
################################################################################
import importlib
import sys
import threading

import pytest

pytestmark = pytest.mark.unit

PA = importlib.import_module("Tensile.TensileLogic.ParseArguments")
RUN = importlib.import_module("Tensile.TensileLogic.Run")


# ---------------------------------------------------------------------------
# parseArguments (reads sys.argv)
# ---------------------------------------------------------------------------
def test_parse_arguments_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "/logic"])
    args = PA.parseArguments()
    assert args.LogicPath == "/logic"
    assert args.Verbose == 1
    assert args.Jobs == 48
    assert args.CheckAll is False
    assert args.CheckOnlyCustomKernels is False


def test_parse_arguments_check_all(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "/logic", "-v", "2", "--check-all"])
    args = PA.parseArguments()
    assert args.Verbose == 2
    assert args.CheckAll is True


def test_parse_arguments_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["prog", "/logic", "--check-all", "--check-only-custom-kernels"]
    )
    with pytest.raises(SystemExit):
        PA.parseArguments()


# ---------------------------------------------------------------------------
# Check NamedTuple
# ---------------------------------------------------------------------------
def test_check_namedtuple():
    c = RUN.Check(OnlyCustomKernels=True, All=False)
    assert c.OnlyCustomKernels is True
    assert c.All is False


# ---------------------------------------------------------------------------
# _progress_loop stop path (pre-set event -> returns immediately)
# ---------------------------------------------------------------------------
def test_progress_loop_stops(capsys):
    ev = threading.Event()
    ev.set()  # already stopped -> loop body skipped, clears line, returns
    RUN._progress_loop(ev, interval=0.01)
    # no assertion on content; just that it returns without hanging
    capsys.readouterr()


# ---------------------------------------------------------------------------
# TensileCreateLibrary/__main__ import shim
# ---------------------------------------------------------------------------
def test_create_library_main_import():
    mod = importlib.import_module("Tensile.TensileCreateLibrary.__main__")
    assert hasattr(mod, "run")
