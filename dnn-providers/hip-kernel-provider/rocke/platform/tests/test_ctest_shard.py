# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``rocke.tests.ctest_shard`` pytest plugin.

Unit tests cover the shard source and the shard rule. The end-to-end tests run
pytest with the plugin in child processes on small generated test projects.
This file itself runs in sharded CI jobs, so each child process starts with no
shard variable and sets only the variables that its case needs. The children
of one case run side by side to keep the cost of this file low.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from rocke.tests import ctest_shard

PACKAGE_ROOT = Path(ctest_shard.__file__).resolve().parents[2]
FLAG = ctest_shard.ACTIVATION_FLAG
SPLIT_LINE = re.compile(
    r"^rocke-ctest-shard (\S+): split ([\d/]+) of (\d+); kept (\d+)$"
)
PASSED_LINE = re.compile(r"^(test_mod\.py::\S+) PASSED")
EMPTY_SHARE = "no test of this entry falls in this shard; the entry passes"


def _gtest(index, total=4):
    return {"GTEST_SHARD_INDEX": str(index), "GTEST_TOTAL_SHARDS": str(total)}


# ----------------------------------------------------------------------- resolve_shard
@pytest.mark.parametrize(
    "env, without_flag, with_flag",
    [
        ({}, None, None),
        ({"ROCKE_TEST_SHARD": "  "}, None, None),
        (_gtest(0), None, "0/4"),
        (_gtest(3), None, "3/4"),
        (_gtest(0, 1), None, "0/1"),
        ({"ROCKE_TEST_SHARD": "2/4"}, "2/4", "2/4"),
        ({"ROCKE_TEST_SHARD": "0,2/4"}, "0,2/4", "0,2/4"),
        ({"ROCKE_TEST_SHARD": "1/4", **_gtest(0)}, "1/4", "1/4"),
        ({"SHARD_INDEX": "1", "TOTAL_SHARDS": "4"}, None, None),
    ],
)
def test_resolve_shard(env, without_flag, with_flag):
    for from_gtest, expected in ((False, without_flag), (True, with_flag)):
        resolved = ctest_shard.resolve_shard(env, from_gtest=from_gtest)
        assert (None if resolved is None else resolved[0]) == expected, from_gtest


BAD_GTEST_PAIRS = [
    {"GTEST_SHARD_INDEX": "2"},
    {"GTEST_TOTAL_SHARDS": "4"},
    {"GTEST_TOTAL_SHARDS": " ", "GTEST_SHARD_INDEX": "2"},
    {"GTEST_TOTAL_SHARDS": "0", "GTEST_SHARD_INDEX": "0"},
    {"GTEST_TOTAL_SHARDS": "x", "GTEST_SHARD_INDEX": "0"},
    {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "4"},
    {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "-1"},
    {"GTEST_TOTAL_SHARDS": "4", "GTEST_SHARD_INDEX": "x"},
    {"GTEST_TOTAL_SHARDS": "1", "GTEST_SHARD_INDEX": "1"},
]


@pytest.mark.parametrize("env", BAD_GTEST_PAIRS)
def test_a_bad_gtest_pair_is_an_error_only_when_it_gives_the_shard(env):
    for blank in ({}, {"ROCKE_TEST_SHARD": " "}):
        with pytest.raises(ValueError):
            ctest_shard.resolve_shard({**env, **blank}, from_gtest=True)
        assert ctest_shard.resolve_shard({**env, **blank}, from_gtest=False) is None


@pytest.mark.parametrize("value", ["4/4", "1", "a/2"])
@pytest.mark.parametrize("from_gtest", [False, True])
def test_a_bad_rocke_test_shard_is_an_error(value, from_gtest):
    with pytest.raises(ValueError):
        ctest_shard.resolve_shard({"ROCKE_TEST_SHARD": value}, from_gtest=from_gtest)


def test_resolve_shard_names_its_source():
    gtest = _gtest(1)
    assert ctest_shard.resolve_shard(gtest, from_gtest=True) == (
        "1/4",
        "GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4",
    )
    same = {**gtest, "ROCKE_TEST_SHARD": "1/4"}
    assert ctest_shard.resolve_shard(same, from_gtest=True) == (
        "1/4",
        "ROCKE_TEST_SHARD=1/4",
    )
    other = {**gtest, "ROCKE_TEST_SHARD": "0/4"}
    for from_gtest in (False, True):
        assert ctest_shard.resolve_shard(other, from_gtest=from_gtest) == (
            "0/4",
            "ROCKE_TEST_SHARD=0/4",
        )


@pytest.mark.parametrize("env", BAD_GTEST_PAIRS)
def test_rocke_test_shard_does_not_read_a_bad_gtest_pair(env):
    for from_gtest in (False, True):
        assert ctest_shard.resolve_shard(
            {**env, "ROCKE_TEST_SHARD": "1/4"}, from_gtest=from_gtest
        ) == ("1/4", "ROCKE_TEST_SHARD=1/4")


def test_resolve_shard_reads_the_process_environment(monkeypatch):
    for name in ctest_shard.CHILD_ENV_EXCLUDED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GTEST_TOTAL_SHARDS", "4")
    monkeypatch.setenv("GTEST_SHARD_INDEX", "2")
    assert ctest_shard.resolve_shard() is None
    assert ctest_shard.resolve_shard(from_gtest=True)[0] == "2/4"


# ------------------------------------------------------------------------- shard rule
# Shard of each key for n = 2, 3, 4 and 8. The values pin the hash, so that
# developer runs and CI map a test to the same shard on every change.
GOLDEN_SHARDS = {
    "tests/core/test_arch_mma_ssot.py::"
    "test_raw_mma_rejects_present_unsupported_destination_dtype": (1, 1, 3, 7),
    "tests/sdpa/test_sdpa_fwd.py[fp16-64]": (0, 0, 0, 4),
    "tests/sdpa/test_sdpa_fwd.py[bf16-128]": (1, 1, 3, 7),
    "tests/conv/test_conv_fwd.py[nhwc-3x3]": (1, 2, 1, 5),
    "tests/test_attention_bwd_hsaco_cache.py::test_shard_parse_and_partition": (
        1,
        1,
        1,
        1,
    ),
    "test_mod.py[7]": (1, 2, 1, 1),
}


@pytest.mark.parametrize("key, shards", GOLDEN_SHARDS.items())
def test_the_shard_of_a_key_does_not_change(key, shards):
    assert tuple(ctest_shard.shard_of(key, n) for n in (2, 3, 4, 8)) == shards


@pytest.mark.parametrize("dtype", ["fp16-64"])
def test_a_shard_key_is_the_module_and_the_parameter_id(request, dtype):
    module = request.node.nodeid.split("::", 1)[0]
    assert ctest_shard.shard_key(request.node) == f"{module}[{dtype}]"


@pytest.mark.parametrize(
    "nodeid, param_id, key",
    [
        ("tests/x.py::test_b[p1]", "p1", "tests/x.py[p1]"),
        ("tests/x.py::TestA::test_b[p1]", "p1", "tests/x.py[p1]"),
        ("tests/x.py::TestA::TestB::test_c[p1-2]", "p1-2", "tests/x.py[p1-2]"),
    ],
)
def test_a_shard_key_drops_the_class_and_the_function(nodeid, param_id, key):
    item = types.SimpleNamespace(
        nodeid=nodeid, callspec=types.SimpleNamespace(id=param_id)
    )
    assert ctest_shard.shard_key(item) == key


def test_a_shard_key_without_parameters_is_the_node_id(request):
    assert ctest_shard.shard_key(request.node) == request.node.nodeid


# -------------------------------------------------------------------------- end to end
_INI = "[pytest]\nmarkers =\n    slow: a test that a run can leave out with -m\n"


def _child_env(**shard):
    env = ctest_shard.unsharded_environment()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p
    )
    env.update(shard)
    return env


def _project(path, source, conftest=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "pytest.ini").write_text(_INI, encoding="utf-8")
    (path / "test_mod.py").write_text(source, encoding="utf-8")
    if conftest is not None:
        (path / "conftest.py").write_text(conftest, encoding="utf-8")
    return path


@dataclasses.dataclass
class _Run:
    args: tuple = ()
    env: dict = dataclasses.field(default_factory=dict)
    flag: bool = True
    plugin: bool = True


def _command(run):
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    if run.plugin:
        command += ["-p", ctest_shard.PLUGIN_NAME]
    if run.flag:
        command.append(FLAG)
    return command + list(run.args)


def _pytest_side_by_side(project, runs):
    """One child pytest per run, all started before the first is awaited."""
    children = []
    try:
        for run in runs:
            children.append(
                subprocess.Popen(
                    _command(run),
                    cwd=project,
                    env=_child_env(**run.env),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        results = []
        for child in children:
            out, err = child.communicate(timeout=120)
            results.append(
                subprocess.CompletedProcess(child.args, child.returncode, out, err)
            )
        return results
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


def _split_line(out):
    lines = [m for m in map(SPLIT_LINE.match, out.splitlines()) if m]
    assert len(lines) == 1, out
    return lines[0]


def _passed(out):
    return {m.group(1) for m in map(PASSED_LINE.match, out.splitlines()) if m}


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
_SLOW = "\n\n@pytest.mark.slow\ndef test_slow():\n    pass\n"
_MANY = _PARAMETRIZED + _PLAIN + _SLOW
_MANY_COUNT = 91
# These filters leave out one test with -m and one with --deselect.
_FILTERS = ("-m", "not slow", "--deselect", "test_mod.py::test_plain1")
_FILTERED_COUNT = _MANY_COUNT - 2

_ONE = "def test_one():\n    pass\n"


def _assert_partition(shards, every):
    assert sum(len(ids) for ids in shards) == len(every)
    assert set().union(*shards) == every
    for ids in shards:  # both tests of one parameter set share a shard
        for nodeid in ids:
            if "::test_a[" in nodeid:
                assert nodeid.replace("::test_a[", "::test_b[") in ids


def test_gtest_shards_partition_the_selection(tmp_path):
    project = _project(tmp_path, _MANY)
    pairs = [(k, n) for n in (1, 2, 3, 4) for k in range(n)]
    results = _pytest_side_by_side(
        project, [_Run(("-v", *_FILTERS), _gtest(k, n)) for k, n in pairs]
    )
    by_total = {}
    for (k, n), run in zip(pairs, results):
        assert run.returncode == 0, run.stdout + run.stderr
        assert (
            f"rocke-ctest-shard: {k}/{n} "
            f"(GTEST_SHARD_INDEX={k}, GTEST_TOTAL_SHARDS={n})"
        ) in run.stdout
        line = _split_line(run.stdout)
        assert line.group(1) == f"{k}/{n}"
        ids = _passed(run.stdout)
        assert int(line.group(4)) == len(ids)
        by_total.setdefault(n, []).append((ids, line.group(2), line.group(3)))

    every = by_total[1][0][0]
    assert len(every) == _FILTERED_COUNT
    assert "test_mod.py::test_slow" not in every
    assert "test_mod.py::test_plain1" not in every
    assert by_total[1][0][1:] == (str(_FILTERED_COUNT), str(_FILTERED_COUNT))
    for n, shards in by_total.items():
        _assert_partition([ids for ids, _split, _total in shards], every)
        splits = {(split, total) for _ids, split, total in shards}
        assert len(splits) == 1, splits  # every shard prints the same split
        ((split, total),) = splits
        counts = [int(c) for c in split.split("/")]
        assert len(counts) == n
        assert int(total) == sum(counts) == len(every)
        assert counts == [len(ids) for ids, _split, _total in shards]


def test_rocke_test_shard_partitions_and_wins_over_gtest(tmp_path):
    project = _project(tmp_path, _MANY)
    runs = [
        _Run(("-v",), {"ROCKE_TEST_SHARD": "0/3", **_gtest(1)}, flag=False),
        _Run(("-v",), {"ROCKE_TEST_SHARD": "1/3", **_gtest(1)}),
        _Run(("-v",), {"ROCKE_TEST_SHARD": "2/3", "GTEST_SHARD_INDEX": "x"}),
    ]
    results = _pytest_side_by_side(project, runs)
    for run in results:
        assert run.returncode == 0, run.stdout + run.stderr
    for index, run in enumerate(results):
        assert (
            f"rocke-ctest-shard: {index}/3 (ROCKE_TEST_SHARD={index}/3)\n" in run.stdout
        )
        assert "overrides" not in run.stdout
    shards = [_passed(run.stdout) for run in results]
    assert sum(len(ids) for ids in shards) == _MANY_COUNT
    _assert_partition(shards, set().union(*shards))
    assert len({_split_line(run.stdout).group(2) for run in results}) == 1


def test_gtest_variables_without_the_flag_run_every_test(tmp_path):
    project = _project(tmp_path, _MANY)
    (run,) = _pytest_side_by_side(project, [_Run(("-v",), _gtest(1), flag=False)])
    assert run.returncode == 0, run.stdout + run.stderr
    assert "rocke-ctest-shard: off\n" in run.stdout
    assert f"{_MANY_COUNT} passed" in run.stdout
    assert not [m for m in map(SPLIT_LINE.match, run.stdout.splitlines()) if m]


@pytest.fixture(scope="module")
def shared_tree(tmp_path_factory):
    """One test, one empty module and one module that fails to import."""
    tree = _project(tmp_path_factory.mktemp("shared"), _ONE)
    (tree / "test_empty.py").write_text("", encoding="utf-8")
    (tree / "test_broken.py").write_text(
        "raise ImportError('broken on purpose')\n", encoding="utf-8"
    )
    return tree


def test_an_empty_shard_passes(shared_tree):
    results = _pytest_side_by_side(
        shared_tree, [_Run(("test_mod.py",), _gtest(k)) for k in range(4)]
    )
    ran = 0
    for run in results:
        out = run.stdout + run.stderr
        assert run.returncode == 0, out
        line = _split_line(run.stdout)
        assert line.group(3) == "1"
        if line.group(4) == "1":
            ran += 1
            assert "1 passed" in out
            assert EMPTY_SHARE not in out
        else:
            assert line.group(4) == "0"
            assert f"rocke-ctest-shard {line.group(1)}: {EMPTY_SHARE}" in out
    assert ran == 1


def test_an_empty_selection_or_a_missing_path_fails_on_every_shard(shared_tree):
    runs = [_Run(("-k", "nomatch", "test_mod.py"), _gtest(k)) for k in range(4)]
    runs += [_Run(("test_missing.py",), _gtest(k)) for k in range(4)]
    runs.append(_Run(("test_empty.py",), flag=False))
    results = _pytest_side_by_side(shared_tree, runs)
    sharded, missing, unsharded = results[:4], results[4:8], results[8]
    for run in sharded:
        out = run.stdout + run.stderr
        assert run.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, out
        assert EMPTY_SHARE not in out
        line = _split_line(run.stdout)
        assert (line.group(3), line.group(4)) == ("0", "0")
    for run in missing:
        out = run.stdout + run.stderr
        assert run.returncode == pytest.ExitCode.USAGE_ERROR, out
        assert "test_missing.py" in run.stderr
    assert unsharded.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, unsharded.stdout


def test_a_collection_error_fails_every_shard(shared_tree):
    """A module that fails to import makes every shard fail (exit 2), also a
    shard whose share of the other tests is empty."""
    results = _pytest_side_by_side(shared_tree, [_Run((), _gtest(k)) for k in range(4)])
    kept = []
    for run in results:
        out = run.stdout + run.stderr
        assert run.returncode == pytest.ExitCode.INTERRUPTED, out
        assert EMPTY_SHARE not in out
        kept.append(int(_split_line(run.stdout).group(4)))
    assert sorted(kept) == [0, 0, 0, 1]


def _lines(out, text):
    return [i for i, line in enumerate(out.splitlines()) if line == text]


@pytest.mark.parametrize(
    "run, header",
    [
        (
            _Run(("-q", "test_mod.py"), _gtest(1)),
            "rocke-ctest-shard: 1/4 (GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4)",
        ),
        (
            _Run(("--no-header", "test_mod.py"), {"ROCKE_TEST_SHARD": "0/2"}),
            "rocke-ctest-shard: 0/2 (ROCKE_TEST_SHARD=0/2)",
        ),
        (_Run(("-q", "test_mod.py"), flag=False), "rocke-ctest-shard: off"),
        (
            _Run(("-v", "test_mod.py"), _gtest(1)),
            "rocke-ctest-shard: 1/4 (GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4)",
        ),
    ],
    ids=["q", "no-header", "q-off", "v"],
)
def test_the_terminal_summary_repeats_the_header_line(shared_tree, run, header):
    (result,) = _pytest_side_by_side(shared_tree, [run])
    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    found = _lines(out, header)
    assert len(found) == (2 if "-v" in run.args else 1), out
    lines = out.splitlines()
    if run.env:  # the split line follows the summary header line
        assert SPLIT_LINE.match(lines[found[-1] + 1]), out
    else:
        assert not [m for m in map(SPLIT_LINE.match, lines) if m], out


def test_ctest_passes_every_shard_of_an_entry(tmp_path):
    ctest = shutil.which("ctest")
    if ctest is None:
        pytest.skip("ctest is not on PATH")
    project = _project(tmp_path / "project", _ONE)
    python = Path(sys.executable).as_posix()
    entries = []
    for index in range(4):
        name = f"shard_{index}"
        entries.append(
            f'add_test({name} "{python}" -m pytest -p no:cacheprovider '
            f'-p {ctest_shard.PLUGIN_NAME} {FLAG} "{project.as_posix()}")\n'
            f"set_tests_properties({name} PROPERTIES "
            f'ENVIRONMENT "GTEST_SHARD_INDEX={index};GTEST_TOTAL_SHARDS=4")\n'
        )
    build = tmp_path / "build"
    build.mkdir()
    (build / "CTestTestfile.cmake").write_text("".join(entries), encoding="utf-8")
    run = subprocess.run(
        [ctest, "--output-on-failure", "--parallel", "4"],
        cwd=build,
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    out = run.stdout + run.stderr
    assert run.returncode == 0, out
    assert "100% tests passed, 0 tests failed out of 4" in out
    assert "(Skipped)" not in out


@pytest.mark.parametrize(
    "run, message",
    [
        (
            _Run(env={"GTEST_TOTAL_SHARDS": "4"}),
            "GTEST_TOTAL_SHARDS=4 needs GTEST_SHARD_INDEX",
        ),
        (
            _Run(env={"GTEST_SHARD_INDEX": "2"}),
            "GTEST_SHARD_INDEX=2 needs GTEST_TOTAL_SHARDS",
        ),
        (
            _Run(env={"GTEST_SHARD_INDEX": "2", "ROCKE_TEST_SHARD": " "}),
            "GTEST_SHARD_INDEX=2 needs GTEST_TOTAL_SHARDS",
        ),
        (
            _Run(env={"ROCKE_TEST_SHARD": "4/4"}, flag=False),
            "shard '4/4' out of range",
        ),
    ],
    ids=["total-only", "index-only", "blank-rocke-test-shard", "bad-rocke-test-shard"],
)
def test_a_malformed_shard_is_a_usage_error(shared_tree, run, message):
    (result,) = _pytest_side_by_side(shared_tree, [run])
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    assert f"rocke-ctest-shard: {message}" in result.stderr


def test_a_child_pytest_started_by_a_test_is_not_sliced(tmp_path):
    inner = _project(
        tmp_path / "inner",
        "".join(f"\n\ndef test_inner{i}():\n    pass\n" for i in range(5)),
    )
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (outer / "test_outer.py").write_text(
        textwrap.dedent(
            f"""
            import os
            import subprocess
            import sys


            def test_nested():
                assert os.environ["GTEST_TOTAL_SHARDS"] == "4"
                child = subprocess.run(
                    [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                     "-p", {ctest_shard.PLUGIN_NAME!r}],
                    cwd={inner.as_posix()!r},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert child.returncode == 0, child.stdout + child.stderr
                assert "rocke-ctest-shard: off" in child.stdout
                assert "5 passed" in child.stdout
            """
        ),
        encoding="utf-8",
    )
    index = ctest_shard.shard_of("test_outer.py::test_nested", 4)
    (run,) = _pytest_side_by_side(outer, [_Run(("test_outer.py",), _gtest(index))])
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout


def test_owns_selection_only_when_loaded_as_a_plugin(tmp_path):
    conftest = (
        "from rocke.tests import ctest_shard\n"
        "\n"
        "\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    print('owns_selection=%s' % ctest_shard.owns_selection(config))\n"
    )
    project = _project(tmp_path, _ONE, conftest)
    loaded, imported = _pytest_side_by_side(
        project,
        [_Run(("-s",), flag=False), _Run(("-s",), flag=False, plugin=False)],
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    assert "owns_selection=True" in loaded.stdout
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert "owns_selection=False" in imported.stdout


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
