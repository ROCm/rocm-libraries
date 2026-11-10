# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``rocke.tests.ctest_shard`` pytest plugin.

Unit tests cover the shard source. The end-to-end tests run pytest with the
plugin in child processes on small generated test projects. This file itself
runs in sharded CI jobs, so each child process starts with no shard variable
and sets only the variables that its case needs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from rocke.tests import ctest_shard

PACKAGE_ROOT = Path(ctest_shard.__file__).resolve().parents[2]
SPLIT_LINE = re.compile(
    r"^rocke-ctest-shard (\S+): split ([\d/]+) of (\d+); kept (\d+)$"
)
SKIPPED_LINE = "reported as skipped"


# ----------------------------------------------------------------------- resolve_shard
@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, None),
        ({"ROCKE_TEST_SHARD": "  "}, None),
        ({"GTEST_SHARD_INDEX": "2"}, None),
        ({"GTEST_TOTAL_SHARDS": " ", "GTEST_SHARD_INDEX": "2"}, None),
        ({"GTEST_TOTAL_SHARDS": "1", "GTEST_SHARD_INDEX": "0"}, None),
        ({"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "0"}, "0/4"),
        ({"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "3"}, "3/4"),
        ({"ROCKE_TEST_SHARD": "2/4"}, "2/4"),
        ({"ROCKE_TEST_SHARD": "0,2/4"}, "0,2/4"),
        (
            {
                "ROCKE_TEST_SHARD": "1/4",
                "GTEST_TOTAL_SHARDS": "4",
                "GTEST_SHARD_INDEX": "0",
            },
            "1/4",
        ),
        (
            {
                "ROCKE_TEST_SHARD": "1/4",
                "GTEST_TOTAL_SHARDS": "x",
                "GTEST_SHARD_INDEX": "9",
            },
            "1/4",
        ),
        ({"SHARD_INDEX": "1", "TOTAL_SHARDS": "4"}, None),
    ],
)
def test_resolve_shard(env, expected):
    resolved = ctest_shard.resolve_shard(env)
    assert (None if resolved is None else resolved[0]) == expected


@pytest.mark.parametrize(
    "env",
    [
        {"GTEST_TOTAL_SHARDS": "1", "GTEST_SHARD_INDEX": "1"},
        {"GTEST_TOTAL_SHARDS": "1"},
        {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "4"},
        {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "-1"},
        {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "x"},
        {"GTEST_TOTAL_SHARDS": "4"},
        {"GTEST_TOTAL_SHARDS": "0", "GTEST_SHARD_INDEX": "0"},
        {"GTEST_TOTAL_SHARDS": "x", "GTEST_SHARD_INDEX": "0"},
        {"ROCKE_TEST_SHARD": "4/4"},
        {"ROCKE_TEST_SHARD": "1"},
        {"ROCKE_TEST_SHARD": "a/2"},
    ],
)
def test_resolve_shard_rejects(env):
    with pytest.raises(ValueError):
        ctest_shard.resolve_shard(env)


def test_resolve_shard_names_its_source():
    gtest = {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "1"}
    assert ctest_shard.resolve_shard(gtest) == (
        "1/4",
        "GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4",
    )
    same = {**gtest, "ROCKE_TEST_SHARD": "1/4"}
    assert ctest_shard.resolve_shard(same) == ("1/4", "ROCKE_TEST_SHARD=1/4")
    other = {**gtest, "ROCKE_TEST_SHARD": "0/4"}
    assert ctest_shard.resolve_shard(other) == (
        "0/4",
        "ROCKE_TEST_SHARD=0/4 overrides GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4",
    )


def test_resolve_shard_reads_the_process_environment(monkeypatch):
    for name in ctest_shard.CHILD_ENV_EXCLUDED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GTEST_TOTAL_SHARDS", "4")
    monkeypatch.setenv("GTEST_SHARD_INDEX", "2")
    assert ctest_shard.resolve_shard()[0] == "2/4"


# -------------------------------------------------------------------------- end to end
def _child_env(**shard):
    env = ctest_shard.unsharded_environment()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    env.update(shard)
    return env


def _project(tmp_path, source, conftest=None):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(source, encoding="utf-8")
    if conftest is not None:
        (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    return tmp_path


def _pytest(project, *args, **shard):
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    command += ["-p", ctest_shard.PLUGIN_NAME, *args]
    return subprocess.run(
        command,
        cwd=project,
        env=_child_env(**shard),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _gtest(index, total=4):
    return {"GTEST_SHARD_INDEX": str(index), "GTEST_TOTAL_SHARDS": str(total)}


def _split_line(out):
    lines = [m for m in map(SPLIT_LINE.match, out.splitlines()) if m]
    assert len(lines) == 1, out
    return lines[0]


_PARAMETRIZED = textwrap.dedent(
    """
    import pytest

    @pytest.mark.parametrize("i", range(40))
    def test_a(i):
        pass

    @pytest.mark.parametrize("i", range(40))
    def test_b(i):
        pass
    """
)
_PLAIN = "".join(f"\n\ndef test_plain{i}():\n    pass\n" for i in range(10))
_MANY = _PARAMETRIZED + _PLAIN

_ONE = "def test_one():\n    pass\n"


def _collected(out):
    return {line for line in out.splitlines() if line.startswith("test_mod.py::")}


def test_shards_partition_the_selection(tmp_path):
    project = _project(tmp_path, _MANY)
    whole = _pytest(project, "--collect-only", "-q")
    assert whole.returncode == 0, whole.stdout + whole.stderr
    every = _collected(whole.stdout)
    assert len(every) == 90

    shards, splits = [], set()
    for index in range(4):
        run = _pytest(project, "--collect-only", "-q", **_gtest(index))
        assert run.returncode == 0, run.stdout + run.stderr
        ids = _collected(run.stdout)
        shards.append(ids)
        for nodeid in ids:  # both tests of one parameter set share a shard
            if "::test_a[" in nodeid:
                assert nodeid.replace("::test_a[", "::test_b[") in ids
        line = _split_line(run.stdout)
        assert line.group(1) == f"{index}/4"
        assert int(line.group(4)) == len(ids)
        splits.add((line.group(2), line.group(3)))

    assert sum(len(ids) for ids in shards) == len(every)
    assert set().union(*shards) == every
    assert all(shards)
    (split, total), *rest = splits
    assert not rest, splits  # every shard prints the same whole split
    counts = [int(c) for c in split.split("/")]
    assert int(total) == sum(counts) == len(every)
    assert counts == [len(ids) for ids in shards]


def test_an_empty_shard_is_reported_as_skipped(tmp_path):
    project = _project(tmp_path, _ONE)
    codes = []
    for index in range(4):
        run = _pytest(project, **_gtest(index))
        codes.append(run.returncode)
        out = run.stdout + run.stderr
        if run.returncode == ctest_shard.SKIP_EXIT_CODE:
            assert SKIPPED_LINE in out
            assert f"no test of this entry falls in shard {index}/4" in out
        else:
            assert run.returncode == 0, out
            assert SKIPPED_LINE not in out
            assert "1 passed" in out
        assert _split_line(run.stdout).group(3) == "1"
    assert sorted(codes) == [0, 77, 77, 77]


def test_a_collection_error_fails_every_shard(tmp_path):
    """A module that fails to import makes every shard fail (exit 2), also a
    shard whose share of the other tests is empty."""
    project = _project(tmp_path, _ONE)
    (project / "test_broken.py").write_text(
        "raise ImportError('broken on purpose')\n", encoding="utf-8"
    )
    kept = []
    for index in range(4):
        run = _pytest(project, **_gtest(index))
        out = run.stdout + run.stderr
        assert run.returncode == pytest.ExitCode.INTERRUPTED, out
        assert SKIPPED_LINE not in out
        kept.append(int(_split_line(run.stdout).group(4)))
    assert sorted(kept) == [0, 0, 0, 1]


def test_ctest_reports_the_empty_shard_as_skipped(tmp_path):
    ctest = shutil.which("ctest")
    if ctest is None:
        pytest.skip("ctest is not on PATH")
    project = tmp_path / "project"
    project.mkdir()
    _project(project, _ONE)
    python = Path(sys.executable).as_posix()
    entries = []
    for index in range(4):
        name = f"shard_{index}"
        entries.append(
            f'add_test({name} "{python}" -m pytest -p no:cacheprovider '
            f'-p {ctest_shard.PLUGIN_NAME} "{project.as_posix()}")\n'
            f"set_tests_properties({name} PROPERTIES SKIP_RETURN_CODE 77 "
            f'ENVIRONMENT "GTEST_SHARD_INDEX={index};GTEST_TOTAL_SHARDS=4")\n'
        )
    build = tmp_path / "build"
    build.mkdir()
    (build / "CTestTestfile.cmake").write_text("".join(entries), encoding="utf-8")
    run = subprocess.run(
        [ctest, "--output-on-failure"],
        cwd=build,
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "0 tests failed out of 4" in out
    assert len(re.findall(r"shard_\d \(Skipped\)", out)) == 3, out


def test_owns_selection_only_when_loaded_as_a_plugin(tmp_path):
    conftest = (
        "from rocke.tests import ctest_shard\n"
        "\n"
        "\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    print('owns_selection=%s' % ctest_shard.owns_selection(config))\n"
    )
    project = _project(tmp_path, _ONE, conftest)
    loaded = _pytest(project, "-s")
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    assert "owns_selection=True" in loaded.stdout
    imported = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-s"],
        cwd=project,
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert "owns_selection=False" in imported.stdout


def test_a_shard_outside_the_range_is_an_error(tmp_path):
    conftest = (
        "import rocke.tests.ctest_shard as ctest_shard\n"
        "\n"
        "ctest_shard.shard_of = lambda key, n: n\n"
    )
    project = _project(tmp_path, _ONE, conftest)
    run = _pytest(project, **_gtest(0))
    assert run.returncode == pytest.ExitCode.USAGE_ERROR, run.stdout + run.stderr
    assert "maps to shard 4, not in 0..3" in run.stderr


def test_an_empty_selection_still_fails(tmp_path):
    project = _project(tmp_path, _ONE)
    run = _pytest(project, "-k", "nomatch", **_gtest(0))
    assert run.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, run.stdout
    assert SKIPPED_LINE not in run.stdout + run.stderr
    assert _split_line(run.stdout).group(3) == "0"

    empty = tmp_path / "empty"
    empty.mkdir()
    _project(empty, "")
    run = _pytest(empty)
    assert run.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, run.stdout


def test_no_shard_runs_every_test(tmp_path):
    project = _project(tmp_path, _ONE)
    run = _pytest(project)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "rocke-ctest-shard: off" in run.stdout
    assert "1 passed" in run.stdout
    assert not [m for m in map(SPLIT_LINE.match, run.stdout.splitlines()) if m]


def test_the_header_names_the_shard_source(tmp_path):
    project = _project(tmp_path, _MANY)
    run = _pytest(project, **_gtest(1))
    assert run.returncode == 0, run.stdout + run.stderr
    assert (
        "rocke-ctest-shard: 1/4 (GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4)"
        in run.stdout
    )


@pytest.mark.parametrize(
    "shard, message",
    [
        ({"ROCKE_TEST_SHARD": "4/4"}, "shard '4/4' out of range"),
        ({"GTEST_TOTAL_SHARDS": "4"}, "GTEST_TOTAL_SHARDS=4 needs GTEST_SHARD_INDEX"),
    ],
)
def test_a_malformed_shard_is_a_usage_error(tmp_path, shard, message):
    project = _project(tmp_path, _ONE)
    run = _pytest(project, **shard)
    assert run.returncode == pytest.ExitCode.USAGE_ERROR, run.stdout + run.stderr
    assert f"rocke-ctest-shard: {message}" in run.stderr


def test_an_import_of_the_plugin_loads_no_gpu_library():
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc/self/maps")
    probe = textwrap.dedent(
        """
        import rocke.tests.ctest_shard

        with open("/proc/self/maps", encoding="utf-8") as maps:
            text = maps.read()
        print([name for name in ("libamd_comgr", "libamdhip64") if name in text])
        """
    )
    run = subprocess.run(
        [sys.executable, "-c", probe],
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"
