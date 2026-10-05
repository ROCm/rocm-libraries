################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.Utilities``: the small pure
helpers (verbosity/printing, param search, exe location, version compat, state/
hash serialization, math) over crafted inputs / tmp dirs."""

import contextlib
import importlib
import os

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


@contextlib.contextmanager
def _verbosity(v):
    saved = U.getVerbosity()
    U.setVerbosity(v)
    try:
        yield
    finally:
        U.setVerbosity(saved)


def test_fastdeepcopy():
    src = {"a": [1, 2], "b": {"c": 3}}
    cp = U.fastdeepcopy(src)
    assert cp == src and cp is not src and cp["a"] is not src["a"]


def test_verbosity_and_prints(capsys):
    with _verbosity(2):
        U.print1("one"); U.print2("two")
        out = capsys.readouterr().out
        assert "one" in out and "two" in out
    with _verbosity(0):
        U.print1("hidden1"); U.print2("hidden2")
        assert capsys.readouterr().out == ""


def test_print_warning(capsys):
    U.printWarning("careful")
    assert "Tensile::WARNING: careful" in capsys.readouterr().out


def test_print_exit(capsys):
    with pytest.raises(SystemExit):
        U.printExit("fatal")
    assert "Tensile::FATAL: fatal" in capsys.readouterr().out


def test_has_param():
    assert U.hasParam("x", {"x": 1}) is True
    assert U.hasParam("x", [{"y": 1}, {"x": 2}]) is True
    assert U.hasParam("x", [{"y": 1}]) is False
    assert U.hasParam("x", "x") is True
    assert U.hasParam("x", "z") is False


def test_is_exe_and_locate_exe(tmp_path, monkeypatch):
    exe = tmp_path / "tool"
    exe.write_text("#!/bin/sh\n")
    os.chmod(exe, 0o755)
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    assert U.isExe(str(exe)) is True
    assert U.isExe(str(plain)) is False
    # found in defaultPath
    assert U.locateExe(str(tmp_path), "tool") == str(exe)
    # found via PATH
    monkeypatch.setenv("PATH", str(tmp_path))
    assert U.locateExe("", "tool") == str(exe)
    # not found
    with pytest.raises(OSError):
        U.locateExe(str(tmp_path), "nonexistent-tool")


def test_ensure_path(tmp_path):
    p = tmp_path / "a" / "b"
    assert U.ensurePath(str(p)) == str(p) and p.is_dir()
    # already-exists -> no error
    assert U.ensurePath(str(p)) == str(p)


def test_round_up_and_math():
    assert U.roundUp(3.1) == 4
    assert U.log2(8) == 3
    assert U.ceilDivide(10, 3) == 4
    assert U.roundUpToNearestMultiple(10, 4) == 12


def test_ceil_divide_negative_and_zero(capsys):
    # Negative -> caught, prints, returns 0. Divide-by-zero -> prints, returns 0.
    assert U.ceilDivide(-1, 3) == 0
    assert U.ceilDivide(10, 0) == 0
    assert "ERROR" in capsys.readouterr().out


def test_version_is_compatible():
    from Tensile import __version__
    assert U.versionIsCompatible(__version__) is True
    major = int(__version__.split(".")[0])
    assert U.versionIsCompatible(f"{major + 1}.0.0") is False   # major mismatch
    assert U.versionIsCompatible("9999.0.0" if major != 9999 else "0.0.0") is False


def test_elineno_is_file_line():
    rv = U.elineno()
    assert isinstance(rv, str) and ":" in rv  # "file.py:NN"


def test_progress_bar(capsys):
    pb = U.ProgressBar(maxValue=10, width=80)
    for _ in range(10):
        pb.increment()
    pb.finish()
    assert "%" in capsys.readouterr().out


def test_spinny_thing(capsys):
    s = U.SpinnyThing()
    s.increment(); s.increment(); s.finish()
    assert capsys.readouterr().out  # wrote spinner chars


def test_data_direction_enum(snapshot):
    assert {m.name: m.value for m in U.DataDirection} == snapshot


def test_iterate_progress_with_len(capsys):
    assert list(U.iterate_progress([1, 2, 3])) == [1, 2, 3]


def test_iterate_progress_without_len(capsys):
    # A generator has no len() -> SpinnyThing path.
    assert list(U.iterate_progress(x for x in [4, 5])) == [4, 5]


def test_is_rhel8(monkeypatch, tmp_path, capsys):
    rhel = tmp_path / "os-release-rhel"
    rhel.write_text('NAME="Red Hat Enterprise Linux"\nVERSION_ID="8.9"\n')
    other = tmp_path / "os-release-other"
    other.write_text('NAME="Ubuntu"\nVERSION_ID="24.04"\n')
    missing = tmp_path / "nope"

    monkeypatch.setattr(U, "Path", lambda p: rhel)
    assert U.isRhel8() is True
    monkeypatch.setattr(U, "Path", lambda p: other)
    assert U.isRhel8() is False
    monkeypatch.setattr(U, "Path", lambda p: missing)
    assert U.isRhel8() is False


def test_state_variants():
    class WithState:
        def state(self):
            return {"s": 1}

    class WithKeys:
        StateKeys = ["a", "b"]
        a, b = 1, 2

    assert U.state(WithState()) == {"s": 1}
    assert U.state(WithKeys()) == {"a": 1, "b": 2}
    assert U.state({"k": 5}) == {"k": 5}
    assert U.state(7) == 7
    assert U.state([1, 2]) == [1, 2]


def test_state_key_ordering():
    @U.state_key_ordering
    class Pt:
        StateKeys = ["x", "y"]

        def __init__(self, x, y):
            self.x, self.y = x, y

    assert Pt(1, 2) < Pt(1, 3)
    assert Pt(1, 2) == Pt(1, 2)
    assert Pt(2, 0) > Pt(1, 9)  # from total_ordering


def test_hash_helpers():
    assert U.hash_combine([1, 2, 3]) == ((1 << 1) ^ 2) << 1 ^ 3
    assert U.hash_combine(5) == 5            # non-iterable -> returned as-is
    assert isinstance(U.hash_objs(1, "a", (2, 3)), int)


def test_client_execution_lock_empty_and_real(tmp_path):
    with U.ClientExecutionLock("") as f:
        assert f is not None                 # devnull handle
    lock = U.ClientExecutionLock(str(tmp_path / "x.lock"))
    assert hasattr(lock, "acquire")          # a filelock.FileLock


def test_assign_parameter_with_default():
    dest = {}
    U.assignParameterWithDefault(dest, "k", {"k": 1}, {"k": 99})
    U.assignParameterWithDefault(dest, "m", {}, {"m": 42})
    assert dest == {"k": 1, "m": 42}


@pytest.mark.parametrize(
    "wmma,bits,expected",
    [
        ((16, 16, 4, 1), None, (1, 16, 2, 2)),
        ((16, 16, 32, 1), None, (2, 16, 2, 8)),
        ((16, 16, 64, 1), None, (2, 16, 2, 16)),
        ((16, 16, 128, 1), 8, (4, 16, 2, 16)),
        ((16, 16, 128, 1), 4, (2, 16, 2, 32)),
    ],
    ids=["w4", "w32", "w64", "w128_b8", "w128_b4"],
)
def test_wmma_v3_input_vgpr_layout(wmma, bits, expected):
    assert U.wmmaV3InputVgprLayout(wmma, bits) == expected


def test_wmma_v3_input_vgpr_layout_unhandled():
    with pytest.raises(AssertionError):
        U.wmmaV3InputVgprLayout((1, 2, 3, 4))


def test_choose_multiplier(snapshot):
    assert U.choose_multiplier(3, 32, 8) == snapshot
