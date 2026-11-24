# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocjitsu launcher and pytest plugin (Tests/emulation), with real pytest runs.

A fake ``rocjitsu`` runs the wrapped command directly, so these need no emulator.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

EMULATION = Path(__file__).parents[1] / "emulation"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EMULATION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("run_rocjitsu")
plugin = load("rocjitsu_pytest")

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")


@pytest.fixture
def rocm(tmp_path):
    """A ROCm tree with a pass-through rocjitsu and a gfx1250 device config."""
    root = tmp_path / "rocm"
    rocjitsu = root / "bin" / "rocjitsu"
    rocjitsu.parent.mkdir(parents=True)
    rocjitsu.write_text('#!/bin/sh\nwhile [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n')
    rocjitsu.chmod(0o755)
    configs = root / "share" / "rocjitsu" / "configs"
    configs.mkdir(parents=True)
    device = dict(simd_count=1024, simd_per_cu=4)
    (configs / "gfx1250_mi455x.json").write_text(json.dumps(dict(vm=dict(gpu=dict(device=device)))))
    return root


def run(tmp_path, rocm, source, *, extra=(), timeout=20, workers=0, env=()):
    test = tmp_path / "test_example.py"
    test.write_text(source)
    reports = tmp_path / "reports"
    command = [
        sys.executable,
        str(EMULATION / "run_rocjitsu.py"),
        "--",
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "rocjitsu_pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        *extra,
        str(test),
    ]
    if workers:
        command += ["-p", "xdist.plugin", "-n", str(workers)]
    # A CI job's own install, shard and summary settings must not leak in.
    ci_names = (
        "THEROCK_BIN_DIR",
        "SHARD_INDEX",
        "TOTAL_SHARDS",
        "GITHUB_STEP_SUMMARY",
        "PYTEST_NUM_WORKERS",
        "PYTEST_TEST_TIMEOUT",
    )
    run_env = {
        **{name: value for name, value in os.environ.items() if name not in ci_names},
        "ROCM_PATH": str(rocm),
        "AMDGPU_TARGETS": "gfx1250",
        "ROCJITSU_REPORT_DIR": str(reports),
        "ROCJITSU_SUITE_TIMEOUT": str(timeout),
        "ROCJITSU_SUITE_GRACE": "0.3",
        "ROCJITSU_DURATIONS": str(tmp_path / "durations.json"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        **dict(env),
    }
    run_env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(command, env=run_env, capture_output=True, text=True, timeout=60)
    execution = json.loads((reports / "execution.json").read_text())
    progress = json.loads((reports / "progress.json").read_text())
    return result, execution, progress


@linux_only
def test_xdist_reports_failures_skips_and_teardown_errors(tmp_path, rocm):
    result, execution, progress = run(
        tmp_path,
        rocm,
        """
import pytest
def test_pass(): pass
def test_fail(): assert False
def test_skip(): pytest.skip('unsupported')
@pytest.mark.xfail(reason='known issue')
def test_xfail(): assert False
@pytest.fixture
def cleanup():
    yield
    pytest.fail('teardown failed')
def test_teardown(cleanup): pass
""",
        workers=2,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert execution["complete"]
    assert (progress["selected"], progress["finished"]) == (5, 5)
    assert (progress["passed"], progress["failed"], progress["skipped"]) == (1, 2, 2)
    assert "failed: " in result.stdout


@linux_only
def test_job_summary_lists_tests_that_did_not_pass(tmp_path, rocm):
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n")
    result, execution, progress = run(
        tmp_path,
        rocm,
        "def test_pass(): pass\ndef test_fail(): assert False\n",
        env={"GITHUB_STEP_SUMMARY": str(summary)},
    )
    assert result.returncode == 1
    text = summary.read_text()
    assert text.startswith("earlier step\n### rocjitsu gfx1250 shard 1/1: failed\n")
    assert "| 2 | 1 | 1 | 0 | 0 | 0 |" in text
    assert "Failed:\n- `" in text and "::test_fail`" in text
    assert "test_pass" not in text


def test_not_passed_separates_timeouts():
    def test(outcome, detail=None):
        return dict(outcome=outcome, phases=dict(call=dict(detail=detail)))

    progress = dict(
        tests={
            "a": test("passed"),
            "b": test("failed", "E   AssertionError"),
            "c": test("failed", "E   Failed: Timeout (>7200.0s) from pytest-timeout."),
            "d": test("incomplete"),
        }
    )
    assert launcher.not_passed(progress) == {
        "failed": ["b"],
        "timed out": ["c"],
        "incomplete": ["d"],
    }


@linux_only
def test_reaps_detached_client_after_success(tmp_path, rocm):
    result, execution, progress = run(
        tmp_path,
        rocm,
        """
import subprocess, sys
from pathlib import Path
def test_client():
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    Path(__file__).with_suffix('.pid').write_text(str(child.pid))
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert execution["complete"] and progress["passed"] == 1
    pid = int((tmp_path / "test_example.pid").read_text())
    assert not Path(f"/proc/{pid}").exists(), "orphan was not reaped"


@linux_only
def test_timeout_preserves_partial_results_and_reaps_clients(tmp_path, rocm):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "progress.json").write_text('{"passed": 999}')
    result, execution, progress = run(
        tmp_path,
        rocm,
        """
import signal, subprocess, sys, time
from pathlib import Path
def test_pass(): pass
def test_hang():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    Path(__file__).with_suffix('.pid').write_text(str(child.pid))
    time.sleep(60)
""",
        timeout=3,
    )
    assert result.returncode == 124, result.stdout + result.stderr
    assert execution["timed_out"] and not execution["complete"]
    assert (progress["selected"], progress["passed"], progress["incomplete"]) == (2, 1, 1)
    pid = int((tmp_path / "test_example.pid").read_text())
    assert not Path(f"/proc/{pid}").exists(), "timed-out client was not reaped"


@linux_only
def test_durations_pair_longest_with_shortest(tmp_path, rocm):
    (tmp_path / "durations.json").write_text(
        json.dumps({"test_run[a]": 1, "test_run[b]": 3, "test_run[c]": 5})
    )
    result, execution, progress = run(
        tmp_path,
        rocm,
        """
import pytest
from pathlib import Path
@pytest.mark.parametrize('name', 'abcdz')
def test_run(name):
    with Path(__file__).with_suffix('.order').open('a') as order:
        order.write(name)
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert progress["passed"] == 5
    # Unrecorded d and z tie with the longest, c; ties keep node id order.
    assert (tmp_path / "test_example.order").read_text() == "cadbz"


@linux_only
def test_shards_are_disjoint_and_cover_the_suite(tmp_path, rocm):
    (tmp_path / "durations.json").write_text(
        json.dumps({f"test_run[{n}]": s for n, s in zip("abcdef", (60, 50, 9, 8, 2, 1))})
    )
    source = """
import pytest
from pathlib import Path
@pytest.mark.parametrize('name', 'abcdef')
def test_run(name):
    with Path(__file__).with_suffix('.ran').open('a') as ran:
        ran.write(name)
"""
    shards = []
    for index in (1, 2):
        ran = tmp_path / "test_example.ran"
        ran.unlink(missing_ok=True)
        result, execution, progress = run(
            tmp_path, rocm, source, env={"SHARD_INDEX": str(index), "TOTAL_SHARDS": "2"}
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert progress["selected"] == progress["passed"]
        shards.append(set(ran.read_text()))
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == set("abcdef")
    # The two longest tests land in different shards.
    assert {"a", "b"} - shards[0] and {"a", "b"} - shards[1]


@linux_only
def test_records_processes_of_a_long_running_test(tmp_path, rocm):
    result, execution, progress = run(
        tmp_path,
        rocm,
        """
import subprocess, sys, time
from pathlib import Path
def test_slow():
    child = subprocess.Popen([sys.executable, '-c',
        'import threading, time; threading.Thread(target=time.sleep, args=(30,)).start(); time.sleep(30)'])
    Path(__file__).with_suffix('.pid').write_text(str(child.pid))
    time.sleep(10)
    child.kill()
""",
        extra=["--rocjitsu-stall-dump-after", "1"],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    pid = (tmp_path / "test_example.pid").read_text()
    (stall,) = (tmp_path / "reports" / "stalls").iterdir()
    text = stall.read_text()
    assert "test_slow still running" in text and f"pid {pid}: " in text
    # The CI job uploads no files, so stall dumps go to the log.
    assert "--- stall dump" in result.stdout


@linux_only
def test_all_skipped_is_not_success(tmp_path, rocm):
    result, execution, progress = run(
        tmp_path, rocm, "import pytest\ndef test_skip(): pytest.skip('unsupported')\n"
    )
    assert result.returncode == 1
    assert (progress["passed"], progress["skipped"]) == (0, 1)


def items(*names):
    return [SimpleNamespace(nodeid=f"test_config.py::test_run[{n}]") for n in names]


def test_shard_items_balances_longest_first():
    durations = {f"test_run[{n}]": s for n, s in zip("abcde", (100, 90, 50, 40, 10))}
    tests = items(*"abcde")
    shards = [[t.nodeid[-2] for t in plugin.shard_items(tests, durations, i, 2)] for i in (1, 2)]
    # a(100) -> 1, b(90) -> 2, c(50) -> 2, d(40) -> 1, e(10) ties -> 1: 150 vs 140.
    assert shards == [["a", "d", "e"], ["b", "c"]]


def test_shard_items_counts_unrecorded_tests_as_the_mean():
    durations = {"test_run[a]": 30, "test_run[b]": 10}
    shards = [[t.nodeid[-2] for t in plugin.shard_items(items(*"abc"), durations, i, 2)] for i in (1, 2)]
    assert shards == [["a"], ["c", "b"]]


@pytest.mark.parametrize(
    "env, expected",
    [({}, (1, 1)), ({"SHARD_INDEX": "2", "TOTAL_SHARDS": "3"}, (2, 3))],
)
def test_env_shard(monkeypatch, env, expected):
    for name in ("SHARD_INDEX", "TOTAL_SHARDS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert plugin.env_shard() == expected


def test_env_shard_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("SHARD_INDEX", "4")
    monkeypatch.setenv("TOTAL_SHARDS", "3")
    with pytest.raises(pytest.UsageError):
        plugin.env_shard()


@pytest.mark.parametrize(
    "cores, expected", [(96, (16, 6)), (128, (16, 8)), (32, (5, 6)), (4, (1, 4))]
)
def test_split_cpus(cores, expected):
    assert launcher.split_cpus(cores) == expected


@pytest.mark.parametrize(
    "cores, gib, expected", [(96, 64, (6, 16)), (96, 512, (16, 6)), (32, 8, (1, 32))]
)
def test_split_cpus_fits_memory(cores, gib, expected):
    assert launcher.split_cpus(cores, gib * 2**30) == expected


@pytest.mark.parametrize(
    "files, expected",
    [
        ({"cpu.max": "3200000 100000\n", "memory.max": "68719476736\n"}, (32, 64 * 2**30)),
        ({"cpu.max": "max 100000\n", "memory.max": "max\n"}, (None, None)),
        (
            {
                "cpu/cpu.cfs_quota_us": "1600000\n",
                "cpu/cpu.cfs_period_us": "100000\n",
                "memory/memory.limit_in_bytes": "9223372036854771712\n",
            },
            (16, None),
        ),
        ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, (None, None)),
        ({}, (None, None)),
    ],
)
def test_cgroup_limits(tmp_path, files, expected):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    assert launcher.cgroup_limits(tmp_path) == expected


@pytest.mark.parametrize(
    "env, target",
    [
        ({"AMDGPU_TARGETS": "gfx1250"}, "gfx1250"),
        ({"AMDGPU_FAMILIES": "gfx125X-dcgpu"}, "gfx1250"),
        ({"AMDGPU_FAMILIES": "gfx94X-dcgpu"}, "gfx942"),
    ],
)
def test_emulated_target(env, target):
    assert launcher.emulated_target(env) == target


@pytest.mark.parametrize("env", [{"AMDGPU_TARGETS": "gfx1250-strict"}, {"AMDGPU_FAMILIES": "gfx110X-all"}, {}])
def test_emulated_target_rejects_unsupported(env):
    with pytest.raises(SystemExit):
        launcher.emulated_target(env)


def test_rocm_root_prefers_bin_dir_then_rocm_path(monkeypatch):
    assert launcher.rocm_root({"ROCM_PATH": "/opt/r", "THEROCK_BIN_DIR": "/x/bin"}) == Path("/x").resolve()
    assert launcher.rocm_root({"ROCM_PATH": "/opt/r"}) == Path("/opt/r")
    installed = Path("/r/share/hipblaslt/tensilelite/tensilelite/Tests/emulation")
    monkeypatch.setattr(launcher, "HERE", installed)
    assert launcher.rocm_root({}) == Path("/r")


@linux_only
def test_emulation_setup(tmp_path, rocm):
    env = {"ROCM_PATH": str(rocm), "AMDGPU_TARGETS": "gfx1250", "PYTHONPATH": "/keep"}
    prefix, run_env = launcher.emulation_setup(env, 96, tmp_path / "reports")
    config = json.loads((tmp_path / "reports" / "rocjitsu-config.json").read_text())
    assert (config["cpu_thread_budget"], config["num_threads"]) == (6, 1)
    run_config = str(tmp_path / "reports" / "rocjitsu-config.json")
    assert prefix == [str(rocm / "bin" / "rocjitsu"), "--config", run_config, "--"]
    assert run_env["CU"] == "256"
    assert (run_env["PYTEST_NUM_WORKERS"], run_env["OMP_THREAD_LIMIT"]) == ("16", "6")
    assert run_env["PYTEST_TEST_TIMEOUT"] == "2700"
    assert run_env["ROCJITSU_STALL_DUMP_AFTER"] == "2400"
    assert run_env["PYTHONPATH"] == f"{EMULATION.resolve()}{os.pathsep}/keep"
    runtime = Path(run_env["ROCJITSU_RUNTIME_DIR"])
    assert runtime.is_dir() and runtime.stat().st_mode & 0o777 == 0o700


@linux_only
def test_emulation_setup_keeps_overrides(tmp_path, rocm):
    env = {
        "ROCM_PATH": str(rocm),
        "AMDGPU_TARGETS": "gfx1250",
        "PYTEST_NUM_WORKERS": "3",
        "PYTEST_TEST_TIMEOUT": "600",
    }
    _, run_env = launcher.emulation_setup(env, 96, tmp_path)
    assert (run_env["PYTEST_NUM_WORKERS"], run_env["PYTEST_TEST_TIMEOUT"]) == ("3", "600")
    assert run_env["ROCJITSU_STALL_DUMP_AFTER"] == "300"


@linux_only
def test_emulation_setup_requires_rocjitsu(tmp_path, rocm):
    (rocm / "bin" / "rocjitsu").unlink()
    with pytest.raises(SystemExit, match="rocjitsu"):
        launcher.emulation_setup({"ROCM_PATH": str(rocm), "AMDGPU_TARGETS": "gfx1250"}, 8, tmp_path)


def test_main_requires_a_command():
    with pytest.raises(SystemExit):
        launcher.main([], {})


def test_durations_cover_rocjitsu_slow_configs():
    durations = json.loads(plugin.DEFAULT_DURATIONS.read_text())
    common = EMULATION.parent / "common"
    slow = [p for p in common.rglob("*.yaml") if "rocjitsu_slow" in p.read_text()]
    assert len(slow) == 7
    for path in slow:
        key = f"test_config[tensilelite/Tests/common/{path.relative_to(common).as_posix()}]"
        assert durations[key] > 45 * 60, key
    # And every recorded config over 45 min is marked.
    marked = {f"test_config[tensilelite/Tests/common/{p.relative_to(common).as_posix()}]" for p in slow}
    assert {key for key, seconds in durations.items() if seconds > 45 * 60} <= marked


FAKE_GTEST = """
import os, sys
from pathlib import Path
out = Path(os.environ["GTEST_OUTPUT"].removeprefix("xml:"))
out.mkdir(parents=True, exist_ok=True)
shard = os.environ["GTEST_SHARD_INDEX"] + "/" + os.environ["GTEST_TOTAL_SHARDS"]
fail = '<failure message="bad"/>' if "fail" in sys.argv else ""
(out / "fake.xml").write_text(
    '<testsuites><testsuite name="s">'
    f'<testcase classname="s" name="ok_{shard}" time="2.5"/>'
    f'<testcase classname="s" name="bad" time="0.1">{fail}</testcase>'
    '</testsuite></testsuites>')
sys.exit(1 if fail else 0)
"""


@linux_only
@pytest.mark.parametrize("args, code", [((), 0), (("fail",), 1)])
def test_gtest_mode_shards_and_summarizes(tmp_path, rocm, args, code):
    (tmp_path / "fake_gtest.py").write_text(FAKE_GTEST)
    reports, summary = tmp_path / "reports", tmp_path / "summary.md"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("GTEST_", "THEROCK_"))},
        "ROCM_PATH": str(rocm),
        "AMDGPU_TARGETS": "gfx1250",
        "ROCJITSU_GTEST": "1",
        "ROCJITSU_REPORT_DIR": str(reports),
        "SHARD_INDEX": "2",
        "TOTAL_SHARDS": "3",
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    command = [sys.executable, str(EMULATION / "run_rocjitsu.py"), "--",
               sys.executable, str(tmp_path / "fake_gtest.py"), *args]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == code, result.stdout + result.stderr
    assert "threads_per_worker=" in result.stdout and "workers=1 " in result.stdout
    assert "s.ok_1/3" in result.stdout
    text = summary.read_text()
    assert ("failed" if code else "passed") in text.splitlines()[0]
    assert ("- `s.bad`" in text) == bool(code)


def test_gtest_mode_fails_without_results(tmp_path):
    result = dict(exit_code=0, timed_out=False, wall_seconds=1)
    assert launcher.summarize_gtest(tmp_path, result) == 1
    assert "no passing tests" in result["error"]


def test_gtest_mode_reports_a_half_written_xml(tmp_path):
    (tmp_path / "gtest").mkdir()
    (tmp_path / "gtest" / "cut.xml").write_text("<testsuites><testsuite")
    result = dict(exit_code=0, timed_out=False, wall_seconds=1)
    assert launcher.summarize_gtest(tmp_path, result) == 1
    assert result["error"].startswith("unreadable gtest XML")
