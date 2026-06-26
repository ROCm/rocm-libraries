################################################################################
# Characterization tests for Tensile.LibraryLogic — pure helper layer.
#
# ADD-ONLY. LibraryLogic.py is dominated by the LogicAnalyzer class (~1200 lines
# of benchmark-data analysis over per-problem winner arrays) plus
# analyzeProblemType / generateLogic, all of which require fully-derived
# Solution/ProblemType state and parsed CSV benchmark data — out of scope here.
# This suite pins the two pure helpers: handle_frequency_issue and read_max_freq.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.LibraryLogic")


# ---------------------------------------------------------------------------
# handle_frequency_issue (interactive input loop)
# ---------------------------------------------------------------------------
def test_handle_frequency_issue_valid(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1400")
    assert M.handle_frequency_issue("msg") == 1400.0


def test_handle_frequency_issue_retries(monkeypatch):
    # empty -> non-positive -> invalid -> valid
    answers = iter(["", "0", "-3", "abc", "950"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert M.handle_frequency_issue("msg") == 950.0


# ---------------------------------------------------------------------------
# read_max_freq (MAX_FREQ env var)
# ---------------------------------------------------------------------------
def test_read_max_freq_set(monkeypatch):
    monkeypatch.setenv("MAX_FREQ", "1234")
    assert M.read_max_freq() == 1234.0


def test_read_max_freq_unset(monkeypatch, capsys):
    monkeypatch.delenv("MAX_FREQ", raising=False)
    assert M.read_max_freq() is None
    assert "not set" in capsys.readouterr().out


def test_read_max_freq_empty(monkeypatch):
    monkeypatch.setenv("MAX_FREQ", "   ")
    assert M.read_max_freq() is None


def test_read_max_freq_invalid(monkeypatch):
    monkeypatch.setenv("MAX_FREQ", "not-a-number")
    assert M.read_max_freq() is None
