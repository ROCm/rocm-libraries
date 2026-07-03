################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os

import pytest
from unittest.mock import mock_open, patch
from pathlib import Path

from Tensile import __version__
from Tensile.Common import Utilities as UtilitiesModule
from Tensile.Common.Utilities import (
    ClientExecutionLock,
    ProgressBar,
    ceilDivide,
    choose_multiplier,
    elineno,
    ensurePath,
    fastdeepcopy,
    getVerbosity,
    hasParam,
    hash_combine,
    hash_objs,
    isRhel8,
    iterate_progress,
    locateExe,
    print1,
    print2,
    printExit,
    printWarning,
    roundUp,
    roundUpToNearestMultiple,
    setVerbosity,
    state,
    state_key_ordering,
    versionIsCompatible,
    wmmaV3InputVgprLayout,
)

@pytest.fixture
def mock_openFile():
    with patch("builtins.open", mock_open()) as mock:
        yield mock

@pytest.fixture
def mock_exists():
    with patch.object(Path, "exists", return_value=True) as mock:
        yield mock

@pytest.fixture
def mock_notExists():
    with patch.object(Path, "exists", return_value=False) as mock:
        yield mock

# Test cases for isRhel8
def test_isRhel8_true(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="8.4"'
    assert isRhel8() is True
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_false_non_rhel(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Ubuntu" VERSION_ID="20.04"'
    assert isRhel8() is False
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_false_new_version(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="9.0"'
    assert isRhel8() is False
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_true_with_warning(mock_exists, mock_openFile):
    mock_openFile.return_value.read.return_value = 'NAME="Red Hat Enterprise Linux" VERSION_ID="8.5"'
    assert isRhel8() is True
    mock_exists.assert_called_once_with()
    mock_openFile.assert_called_once_with(Path("/etc/os-release"), "r")

def test_isRhel8_file_not_found(mock_notExists, mock_openFile):
    assert isRhel8() is False
    mock_notExists.assert_called_once_with()
    mock_openFile.assert_not_called()  # No file open attempt if the file doesn't exist


class TestHasParam:
    """hasParam searches scalars, dicts (by key), and lists (recursively)."""

    @pytest.mark.parametrize("structure", [
        "Foo",
        {"Foo": 1, "Bar": 2},
        ["Bar", {"Foo": 1}],
    ])
    def test_finds_param(self, structure):
        assert hasParam("Foo", structure) is True

    @pytest.mark.parametrize("structure", [
        "Bar",
        {"Bar": 2},
        ["Bar", "Baz"],
        [],
    ])
    def test_absent_param_returns_false(self, structure):
        assert hasParam("Foo", structure) is False


class TestRoundUp:
    @pytest.mark.parametrize("value,expected", [
        (2.1, 3),
        (2.0, 2),
        (-1.5, -1),
    ])
    def test_round_up(self, value, expected):
        result = roundUp(value)
        assert result == expected
        assert type(result) is int


class TestCeilDivide:
    @pytest.mark.parametrize("num,den,expected", [
        (10, 3, 4),
        (9, 3, 3),
        (0, 5, 0),
    ])
    def test_happy_path(self, num, den, expected):
        assert ceilDivide(num, den) == expected

    def test_negative_numerator_returns_zero(self):
        assert ceilDivide(-1, 3) == 0

    def test_negative_denominator_returns_zero(self):
        assert ceilDivide(3, -1) == 0

    def test_divide_by_zero_returns_zero(self):
        assert ceilDivide(5, 0) == 0


class TestRoundUpToNearestMultiple:
    @pytest.mark.parametrize("num,den,expected", [
        (10, 4, 12),
        (8, 4, 8),
        (1, 8, 8),
    ])
    def test_round_up_to_multiple(self, num, den, expected):
        assert roundUpToNearestMultiple(num, den) == expected


_MAJOR, _MINOR, _STEP = (int(p) for p in __version__.split("."))


class TestVersionIsCompatible:
    """Compatibility is keyed off the live Tensile __version__ so these stay
    correct across version bumps."""

    def test_exact_version_is_compatible(self):
        assert versionIsCompatible(f"{_MAJOR}.{_MINOR}.{_STEP}") is True

    def test_newer_major_is_incompatible(self):
        assert versionIsCompatible(f"{_MAJOR + 1}.{_MINOR}.{_STEP}") is False

    def test_older_major_is_incompatible(self):
        assert versionIsCompatible(f"{_MAJOR - 1}.{_MINOR}.{_STEP}") is False

    def test_newer_minor_is_incompatible(self):
        assert versionIsCompatible(f"{_MAJOR}.{_MINOR + 1}.{_STEP}") is False

    def test_newer_step_same_minor_is_incompatible(self):
        assert versionIsCompatible(f"{_MAJOR}.{_MINOR}.{_STEP + 1}") is False

    def test_older_minor_same_major_is_compatible(self, monkeypatch):
        monkeypatch.setattr(UtilitiesModule, "__version__", f"{_MAJOR}.{_MINOR + 1}.0")
        assert versionIsCompatible(f"{_MAJOR}.{_MINOR}.99") is True


class TestHashCombine:
    def test_two_values(self):
        assert hash_combine(1, 2) == 0  # (1 << 1) ^ 2

    def test_three_values(self):
        assert hash_combine(1, 2, 3) == 3

    @pytest.mark.parametrize("args,expected", [
        (([1, 2, 3],), 3),
        ((5,), 5),
        ((), 0),
    ])
    def test_edge_shapes(self, args, expected):
        assert hash_combine(*args) == expected

    def test_custom_shift(self):
        assert hash_combine(1, 2, shift=2) == 6  # (1 << 2) ^ 2


class TestHashObjs:
    def test_matches_tuple_hash(self):
        assert hash_objs(1, 2, 3) == hash((1, 2, 3))

    def test_order_sensitive(self):
        assert hash_objs(1, 2) != hash_objs(2, 1)


class TestChooseMultiplier:
    @pytest.mark.parametrize("d,N,p,expected", [
        (3, 8, 8, (171, 1, 2)),
        (2, 8, 8, (257, 1, 1)),
        (7, 16, 16, (74899, 3, 3)),
    ])
    def test_known_constants(self, d, N, p, expected):
        assert choose_multiplier(d, N, p) == expected


class TestWmmaV3InputVgprLayout:
    @pytest.mark.parametrize("wmma,expected", [
        ((16, 16, 4, 1), (1, 16, 2, 2)),
        ((16, 16, 32, 1), (2, 16, 2, 8)),
        ((16, 16, 64, 1), (2, 16, 2, 16)),
    ])
    def test_layouts_without_bitwidth(self, wmma, expected):
        assert wmmaV3InputVgprLayout(wmma) == expected

    @pytest.mark.parametrize("wmma,bitwidth,expected", [
        ((16, 16, 128, 1), 8, (4, 16, 2, 16)),
        ((32, 16, 128, 1), 8, (4, 16, 2, 16)),
        ((16, 16, 128, 1), 4, (2, 16, 2, 32)),
        ((16, 16, 128, 1), 6, (2, 16, 2, 32)),
    ])
    def test_bitwidth_dependent_layouts(self, wmma, bitwidth, expected):
        assert wmmaV3InputVgprLayout(wmma, bitwidth) == expected

    def test_unknown_wmma_raises(self):
        with pytest.raises(AssertionError):
            wmmaV3InputVgprLayout((1, 2, 3, 4))

    def test_missing_bitwidth_raises(self):
        with pytest.raises(AssertionError):
            wmmaV3InputVgprLayout((16, 16, 128, 1))

    def test_unsupported_bitwidth_raises(self):
        with pytest.raises(AssertionError):
            wmmaV3InputVgprLayout((16, 16, 128, 1), 16)


class TestState:
    def test_primitives_pass_through(self):
        assert state(5) == 5
        assert state("gfx942") == "gfx942"
        assert state(1.5) == 1.5

    def test_dict_is_recursively_serialized(self):
        assert state({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}

    def test_list_is_recursively_serialized(self):
        assert state([1, "x", 2.0]) == [1, "x", 2.0]

    def test_statekeys_object_is_serialized(self):
        class WithStateKeys:
            StateKeys = ["a", ("renamed", "b")]

            def __init__(self):
                self.a = 1
                self.b = [2, 3]

        assert state(WithStateKeys()) == {"a": 1, "renamed": [2, 3]}

    def test_object_with_state_method_delegates(self):
        class WithState:
            def state(self):
                return {"kind": "custom"}

        assert state(WithState()) == {"kind": "custom"}

    def test_non_serializable_object_returned_as_is(self):
        sentinel = object()
        assert state(sentinel) is sentinel


class TestStateKeyOrdering:
    def test_ordering_and_equality(self):
        @state_key_ordering
        class Pair:
            StateKeys = ["a", "b"]

            def __init__(self, a, b):
                self.a = a
                self.b = b

        assert Pair(1, 2) == Pair(1, 2)
        assert Pair(1, 2) != Pair(1, 3)
        assert Pair(1, 2) < Pair(1, 3)
        assert Pair(2, 0) > Pair(1, 9)
        # total_ordering derives <= and >= from __lt__ and __eq__.
        assert Pair(1, 2) <= Pair(1, 2)
        assert Pair(2, 0) >= Pair(2, 0)


@pytest.fixture
def restore_verbosity():
    original = getVerbosity()
    yield
    setVerbosity(original)


class TestFastDeepCopy:
    def test_returns_equal_but_distinct(self):
        original = {"a": [1, 2], "b": {"c": 3}}
        copy = fastdeepcopy(original)
        assert copy == original
        assert copy is not original

    def test_nested_mutation_is_independent(self):
        original = {"a": [1, 2]}
        copy = fastdeepcopy(original)
        copy["a"].append(99)
        assert original["a"] == [1, 2]


class TestElineno:
    def test_returns_file_and_line(self):
        result = elineno()  # reports this call site
        assert result.startswith("test_Utilities.py:")
        assert result.split(":")[1].isdigit()


class TestPrintHelpers:
    def test_print1_emits_when_verbose(self, capsys, restore_verbosity):
        setVerbosity(1)
        print1("hello")
        assert "hello" in capsys.readouterr().out

    def test_print1_silent_when_quiet(self, capsys, restore_verbosity):
        setVerbosity(0)
        print1("hello")
        assert capsys.readouterr().out == ""

    def test_print2_needs_higher_verbosity(self, capsys, restore_verbosity):
        setVerbosity(1)
        print2("deep")
        assert capsys.readouterr().out == ""
        setVerbosity(2)
        print2("deep")
        assert "deep" in capsys.readouterr().out

    def test_print_warning_includes_message(self, capsys):
        printWarning("careful")
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "careful" in out

    def test_print_exit_raises_system_exit(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            printExit("fatal thing")
        assert exc_info.value.code == -1
        out = capsys.readouterr().out
        assert "FATAL" in out
        assert "fatal thing" in out


class TestLocateExe:
    @staticmethod
    def _make_executable(path):
        path.write_text("#!/bin/sh\n")
        os.chmod(path, 0o755)

    def test_found_in_default_path(self, tmp_path):
        self._make_executable(tmp_path / "mytool")
        assert locateExe(str(tmp_path), "mytool") == str(tmp_path / "mytool")

    def test_found_on_PATH_when_absent_from_default_path(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self._make_executable(bindir / "mytool")
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(bindir))
        # defaultPath is searched first and misses, so the lookup falls
        # through to PATH.
        assert locateExe(str(empty), "mytool") == str(bindir / "mytool")

    def test_not_found_raises_oserror(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        with pytest.raises(OSError):
            locateExe("", "does_not_exist")


class TestEnsurePath:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "a" / "b"
        assert ensurePath(str(target)) == str(target)
        assert target.is_dir()

    def test_existing_directory_is_ok(self, tmp_path):
        target = tmp_path / "exists"
        target.mkdir()
        # FileExistsError is swallowed; the path is returned.
        assert ensurePath(str(target)) == str(target)

    def test_oserror_is_reraised(self):
        with patch("os.makedirs", side_effect=OSError("denied")):
            with pytest.raises(OSError):
                ensurePath("/some/path")


class TestClientExecutionLock:
    def test_empty_path_returns_devnull_handle(self):
        handle = ClientExecutionLock("")
        try:
            assert handle.name == os.devnull
        finally:
            handle.close()

    def test_path_returns_filelock(self, tmp_path):
        lock = ClientExecutionLock(str(tmp_path / "client.lock"))
        acquired = lock.acquire(timeout=0)
        try:
            assert lock.lock_file == str(tmp_path / "client.lock")
            assert acquired is not None
        finally:
            lock.release()


class TestIterateProgress:
    def test_yields_all_items_for_sized_iterable(self, capsys):
        # list has __len__ -> ProgressBar path
        assert list(iterate_progress([1, 2, 3])) == [1, 2, 3]
        assert "100%" in capsys.readouterr().out

    def test_yields_all_items_for_generator(self, capsys):
        # generator has no __len__ -> SpinnyThing path
        gen = (x * 2 for x in range(3))
        assert list(iterate_progress(gen)) == [0, 2, 4]
        assert "*" in capsys.readouterr().out


class TestProgressBar:
    def test_update_without_new_tick_does_not_print(self, capsys):
        progress = ProgressBar(maxValue=100, width=10)
        progress.update(50)
        first_output = capsys.readouterr().out

        progress.update(51)
        assert first_output
        assert capsys.readouterr().out == ""
