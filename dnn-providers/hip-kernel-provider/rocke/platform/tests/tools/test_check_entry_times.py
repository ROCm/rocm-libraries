# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The per-entry time check projects each CI shard and fails a tier over budget."""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
from rocke.tests.lanes import SOURCE_TREE, skip_until


def _load_check_module():
    """The production check: in the source tree, or beside this test in an
    installed tree, which stages ``tools/check_entry_times.py`` and
    ``entry_times.json`` into ``tests/tools``."""
    test_file = Path(__file__).resolve()
    for candidate in (
        test_file.parents[3] / "tools" / "check_entry_times.py",
        test_file.with_name("check_entry_times.py"),
    ):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "check_entry_times", candidate
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # dataclasses resolve their module
            spec.loader.exec_module(module)
            return module
    skip_until(
        SOURCE_TREE,
        reason="rocke/tools/check_entry_times.py is absent from this platform-only tree",
    )


check = _load_check_module()
# The check puts the rocke package on the path, so these follow it.
import rocke
from rocke.tests.ci_shard import (
    ACTIVATION_FLAG,
    LIST_KEYS_FLAG,
    ShardKeyListing,
    place_keys,
    shard_of,
)
from rocke.tests.ctest_workers import WORKERS_OPTION, worker_keys

QUICK, STANDARD = check.CI_TIERS
SPLIT_COMMAND = [
    "python",
    "-m",
    "pytest",
    "-p",
    "rocke.tests.ctest_shard",
    ACTIVATION_FLAG,
    "tests",
]
SPLIT_RUNNER_COMMAND = ["python", "tests/run_golden.py", ACTIVATION_FLAG]
WHOLE_COMMAND = ["python", "tests/run_golden.py"]


def _test(name: str, labels: list[str], command: list[str], **properties) -> dict:
    """One entry as `ctest --show-only=json-v1` lists it."""
    listed = [{"name": "LABELS", "value": labels}]
    listed += [{"name": key, "value": value} for key, value in properties.items()]
    return {"name": name, "command": command, "properties": listed}


def _listing(*tests: dict) -> dict:
    return {"kind": "ctestInfo", "tests": list(tests)}


def _column(ceilings: dict | None = None, test_seconds: dict | None = None, **seconds):
    column = {"source": "test data", "seconds": seconds}
    if test_seconds is not None:
        column["test_seconds"] = test_seconds
    if ceilings is not None:
        column["ceilings"] = ceilings
    return {"linux": column}


def _verdict(verdicts, tier):
    [verdict] = [v for v in verdicts if v.tier == tier]
    return verdict


def _per_shard(verdict, name: str) -> list[float]:
    return [shard[name] for shard in verdict.shards]


def test_registered_entries_reads_labels_command_and_how_the_entry_runs() -> None:
    entries = check.registered_entries(
        [
            _test(
                "rocke_pytest_core",
                ["quick", "standard"],
                [*SPLIT_COMMAND, WORKERS_OPTION, "4"],
                ENVIRONMENT=["PYTHONPATH=."],
                WORKING_DIRECTORY="/installed/ctest",
                SKIP_RETURN_CODE=77,
            ),
            _test("rocke_golden_static", ["quick"], WHOLE_COMMAND),
            _test("rocke_golden_split", ["standard"], SPLIT_RUNNER_COMMAND),
            _test("hip_kernel_provider_tests", ["quick"], WHOLE_COMMAND),
        ]
    )
    assert [entry.name for entry in entries] == [
        "rocke_pytest_core",
        "rocke_golden_static",
        "rocke_golden_split",
    ]
    core, golden, golden_split = entries
    assert core.labels == frozenset({"quick", "standard"})
    assert core.splits_across_shards and golden_split.splits_across_shards
    assert not golden.splits_across_shards
    assert core.pytest_arguments == (*SPLIT_COMMAND[3:], WORKERS_OPTION, "4")
    assert golden_split.pytest_arguments is None
    assert (core.workers, golden.workers) == (4, 1)
    assert core.environment == ("PYTHONPATH=.",)
    assert core.working_directory == "/installed/ctest"
    assert (core.skip_return_code, golden.skip_return_code) == (77, None)


def test_the_plugin_without_the_gtest_flag_does_not_split() -> None:
    # Without the flag the plugin reads only ROCKE_TEST_SHARD, which CI does not
    # set, so the entry runs whole on every shard.
    plugin_only = ["python", "-m", "pytest", "-p", "rocke.tests.ctest_shard", "t"]
    [entry] = check.registered_entries([_test("rocke_x", ["standard"], plugin_only)])
    assert not entry.splits_across_shards


def test_registered_entries_refuses_an_entry_listed_without_its_command() -> None:
    listing = _listing({"name": "rocke_pytest_core", "properties": []})
    with pytest.raises(
        ValueError, match="registered_entries failed - rocke_pytest_core has no command"
    ):
        check.registered_entries(listing["tests"])


def test_without_keys_split_entries_divide_evenly_and_whole_entries_do_not() -> None:
    entries = check.registered_entries(
        [
            _test("rocke_pytest_core", ["quick", "standard"], SPLIT_COMMAND),
            _test("rocke_golden_static", ["quick", "standard"], WHOLE_COMMAND),
        ]
    )
    times = _column(rocke_pytest_core=400.0, rocke_golden_static=30.0)
    verdicts = check.evaluate_tiers(entries, times)
    assert _verdict(verdicts, QUICK).shards == (
        {"rocke_pytest_core": 400.0, "rocke_golden_static": 30.0},
    )
    standard = _verdict(verdicts, STANDARD)
    assert _per_shard(standard, "rocke_pytest_core") == [100.0] * 4
    assert _per_shard(standard, "rocke_golden_static") == [30.0] * 4


def _keys_on_shard(prefix: str, shard: int, count: int, shards: int = 4) -> list[str]:
    """``count`` shard keys ``test_sweep.py[<prefix>-<n>]`` that hash onto
    ``shard``."""
    keys = (f"test_sweep.py[{prefix}-{n}]" for n in range(10_000))
    return [key for key in keys if shard_of(key, shards) == shard][:count]


def _test_of(key: str) -> str:
    """The node id of the one test of ``key``."""
    module, _, parameter = key.partition("[")
    return f"{module}::test_case[{parameter}"


# Four heavy keys that hash onto the second shard, and three light keys on each
# shard: on average a shard takes (4 * 150 + 12 * 50) / 4 = 300 s.
HEAVY = _keys_on_shard("heavy", 1, 4)
LIGHT = [key for shard in range(4) for key in _keys_on_shard("light", shard, 3)]
HEAVY_SECONDS, LIGHT_SECONDS = 150.0, 50.0
SWEEP_TEST_SECONDS = {
    **{_test_of(key): HEAVY_SECONDS for key in HEAVY},
    **{_test_of(key): LIGHT_SECONDS for key in LIGHT},
}
SWEEP_WHOLE_SECONDS = sum(SWEEP_TEST_SECONDS.values())


def _sweep_keys(declared_costs: dict[str, float] | None = None):
    return ShardKeyListing(
        key_of_case={_test_of(key): key for key in HEAVY + LIGHT},
        costs=declared_costs or {},
    )


def test_heavy_keys_the_hash_puts_together_are_projected_onto_their_shard() -> None:
    per_shard = check.split_entry_shard_seconds(
        SWEEP_WHOLE_SECONDS,
        4,
        keys=_sweep_keys(),
        test_seconds=SWEEP_TEST_SECONDS,
    )
    assert SWEEP_WHOLE_SECONDS / 4 == 300.0  # the average shard fits 540 s
    assert per_shard == (150.0, 750.0, 150.0, 150.0)


def test_declared_costs_place_heavy_keys_as_ci_does() -> None:
    costs = dict.fromkeys(HEAVY, HEAVY_SECONDS)
    per_shard = check.split_entry_shard_seconds(
        SWEEP_WHOLE_SECONDS,
        4,
        keys=_sweep_keys(costs),
        test_seconds=SWEEP_TEST_SECONDS,
    )
    assert per_shard == (300.0,) * 4


def test_a_declared_cost_weighs_a_key_without_recorded_seconds() -> None:
    keys = ShardKeyListing(
        key_of_case={"t.py::test[a]": "t.py[a]", "t.py::test[b]": "t.py[b]"},
        costs={"t.py[a]": 30.0, "t.py[b]": 10.0},
    )
    per_shard = check.split_entry_shard_seconds(80.0, 2, keys=keys)
    assert sorted(per_shard) == [20.0, 60.0]


def test_declared_costs_scale_by_the_keys_that_also_have_recorded_seconds() -> None:
    keys = ShardKeyListing(
        key_of_case={"t.py::test[a]": "t.py[a]", "t.py::test[b]": "t.py[b]"},
        costs={"t.py[a]": 30.0, "t.py[b]": 10.0},
    )
    # b takes 30 s against its declared 10: a, declared 30, is taken as 90 s.
    per_shard = check.split_entry_shard_seconds(
        120.0, 2, keys=keys, test_seconds={"t.py::test[b]": 30.0}
    )
    assert sorted(per_shard) == [30.0, 90.0]


def test_keys_without_a_weight_share_what_the_weighed_keys_leave() -> None:
    # One measured key of 60 s on the second shard; the other 4 keys, all on
    # the first, share the remaining 40 s of a 100 s entry.
    measured_key = _keys_on_shard("measured", 1, 1, shards=2)[0]
    other_keys = _keys_on_shard("other", 0, 4, shards=2)
    keys = ShardKeyListing(
        key_of_case={_test_of(key): key for key in [measured_key, *other_keys]},
        costs={},
    )
    per_shard = check.split_entry_shard_seconds(
        100.0, 2, keys=keys, test_seconds={_test_of(measured_key): 60.0}
    )
    assert per_shard == (40.0, 60.0)


def test_keys_without_a_weight_share_the_rest_by_their_number_of_cases() -> None:
    # Two untimed keys: three cases on the first shard, one on the second.
    [three] = _keys_on_shard("three", 0, 1, shards=2)
    [one] = _keys_on_shard("one", 1, 1, shards=2)
    cases = {f"{three}/{n}": three for n in range(3)} | {f"{one}/0": one}
    keys = ShardKeyListing(cases)
    assert check.split_entry_shard_seconds(80.0, 2, keys=keys) == (60.0, 20.0)


def _keys_per_worker(shard: int, shards: int, workers: int, per_worker: int):
    """Keys that hash onto ``shard`` of ``shards`` and that
    ``worker_keys`` spreads ``per_worker`` to each of ``workers``."""
    chosen: dict[int, list[str]] = {worker: [] for worker in range(workers)}
    for key in _keys_on_shard("spread", shard, 1_000, shards):
        worker = worker_keys([key], {}, workers)[key]
        if len(chosen[worker]) < per_worker:
            chosen[worker].append(key)
    assert all(len(keys) == per_worker for keys in chosen.values())
    return [key for keys in chosen.values() for key in keys]


def test_a_shard_with_workers_takes_as_long_as_its_busiest_worker() -> None:
    # Shard 0: one 100 s key. Shard 1: eight 25 s keys, two per worker of four.
    heavy = _keys_on_shard("alone", 0, 1, shards=2)
    spread = _keys_per_worker(1, 2, workers=4, per_worker=2)
    test_seconds = {_test_of(heavy[0]): 100.0}
    test_seconds |= {_test_of(key): 25.0 for key in spread}
    keys = ShardKeyListing(
        key_of_case={_test_of(key): key for key in heavy + spread},
        costs={},
    )
    # The entry's whole time is the sum of its shards, 100 s + 50 s.
    with_workers = check.split_entry_shard_seconds(
        150.0, 2, workers=4, keys=keys, test_seconds=test_seconds
    )
    assert with_workers == (100.0, 50.0)
    # One process per shard: 100 s against 200 s.
    single = check.split_entry_shard_seconds(
        150.0, 2, workers=1, keys=keys, test_seconds=test_seconds
    )
    assert single == (50.0, 100.0)


def test_tests_that_run_alone_add_to_their_shard_after_the_workers() -> None:
    # Shard 0: a 40 s run-alone key and a 20 s key; shard 1: one 60 s key.
    [alone, worked] = _keys_on_shard("first", 0, 2, shards=2)
    [other] = _keys_on_shard("second", 1, 1, shards=2)
    keys = ShardKeyListing(
        key_of_case={_test_of(key): key for key in (alone, worked, other)},
        costs={},
        run_alone_cases=frozenset({_test_of(alone)}),
    )
    test_seconds = {_test_of(alone): 40.0, _test_of(worked): 20.0}
    test_seconds[_test_of(other)] = 60.0
    per_shard = check.split_entry_shard_seconds(
        120.0, 2, workers=4, keys=keys, test_seconds=test_seconds
    )
    assert per_shard == (60.0, 60.0)


@pytest.mark.parametrize("shards, workers", [(0, 1), (2, 0)])
def test_split_entry_shard_seconds_refuses_no_shard_or_no_worker(
    shards: int, workers: int
) -> None:
    name, value = ("shards", shards) if shards < 1 else ("workers", workers)
    with pytest.raises(
        ValueError,
        match=f"split_entry_shard_seconds failed - {name}={value}, expected at "
        "least 1",
    ):
        check.split_entry_shard_seconds(10.0, shards, workers=workers)


def test_a_tier_over_budget_names_the_shard_and_the_entry_alone() -> None:
    entries = check.registered_entries(
        [
            _test("rocke_pytest_core", ["quick"], SPLIT_COMMAND),
            _test("rocke_golden_static", ["quick"], WHOLE_COMMAND),
        ]
    )
    verdict = _verdict(
        check.evaluate_tiers(
            entries, _column(rocke_pytest_core=1900.0, rocke_golden_static=40.0)
        ),
        QUICK,
    )
    entry_alone = (
        "linux quick shard 1 of 1: rocke_pytest_core alone takes 1900 s, over the "
        "300 s budget"
    )
    whole_shard = (
        "linux quick shard 1 of 1 takes 1940 s, over the 300 s budget and over the "
        "1800 s step limit; largest: rocke_pytest_core 1900 s, "
        "rocke_golden_static 40 s"
    )
    assert verdict.breaches() == [entry_alone, whole_shard]


def test_only_the_shard_over_budget_breaches_while_the_average_fits() -> None:
    entries = check.registered_entries(
        [_test("rocke_library_gpu_pytest", ["standard"], SPLIT_COMMAND)]
    )
    times = _column(
        test_seconds={"rocke_library_gpu_pytest": SWEEP_TEST_SECONDS},
        rocke_library_gpu_pytest=SWEEP_WHOLE_SECONDS,
    )
    [verdict] = check.evaluate_tiers(
        entries, times, {"rocke_library_gpu_pytest": _sweep_keys()}, tiers=[STANDARD]
    )
    assert verdict.shard_seconds == (150.0, 750.0, 150.0, 150.0)
    entry_alone = (
        "linux standard shard 2 of 4: rocke_library_gpu_pytest alone takes 750 s, "
        "over the 540 s budget"
    )
    whole_shard = (
        "linux standard shard 2 of 4 takes 750 s, over the 540 s budget; largest: "
        "rocke_library_gpu_pytest 750 s"
    )
    assert verdict.breaches() == [entry_alone, whole_shard]
    assert verdict.summary() == (
        "linux standard (4 shard(s)): 150 / 750 / 150 / 150 s per shard, slowest "
        "750 s, 210 s over the 540 s budget"
    )


def test_a_tier_within_budget_has_no_breaches() -> None:
    entries = check.registered_entries(
        [_test("rocke_pytest_core", ["standard"], SPLIT_COMMAND)]
    )
    verdicts = check.evaluate_tiers(entries, _column(rocke_pytest_core=1600.0))
    assert [v.breaches() for v in verdicts] == [[], []]
    assert _verdict(verdicts, STANDARD).shard_seconds == (400.0,) * 4


def test_excluded_entries_do_not_count() -> None:
    entries = check.registered_entries(
        [_test("rocke_gpu_only", ["quick", check.EXCLUDED_LABEL], WHOLE_COMMAND)]
    )
    times = _column(rocke_gpu_only=900.0)
    verdict = _verdict(check.evaluate_tiers(entries, times), QUICK)
    assert verdict.shards == ({},)
    assert verdict.breaches() == []
    assert check.untimed_entries(entries, _column()) == []


def test_an_entry_a_tier_runs_without_a_time_fails_once_per_column() -> None:
    entries = check.registered_entries(
        [
            _test("rocke_new_entry", ["quick", "standard"], WHOLE_COMMAND),
            _test("rocke_unlabelled", ["comprehensive"], WHOLE_COMMAND),
        ]
    )
    assert check.untimed_entries(entries, _column(rocke_new_entry=1.0)) == []
    untimed = (
        "linux: rocke_new_entry has no time in the data, so no shard counts it; "
        "run it in a CI job and record its time from the job's `ctest -V` log: "
        'check_entry_times.py record --column linux --source "<which jobs and '
        'dates>" LOG...'
    )
    assert check.untimed_entries(entries, _column()) == [untimed]


# Lines in the form a CI job log shows `ctest -V` output: a timestamp, then the
# CTest line. Entry 7 takes its shard from the gtest variables and ran its 1/4
# share here.
_SHARD_LOG = """\
2026-11-17T11:50:01.3Z 7: Test command: /venv/bin/python "-m" "pytest" "-p" \
"rocke.tests.ctest_shard" "--rocke-shard-from-gtest-env" "./tests/core"
2026-11-17T11:50:02.0Z 7: rocke-ctest-shard: 3/4 (GTEST_SHARD_INDEX=3, \
GTEST_TOTAL_SHARDS=4)
2026-11-17T11:50:35.2Z  10/40 Test   #7: rocke_pytest_core ............   Passed   30.50 sec
2026-11-17T11:50:35.3Z 8: Test command: /venv/bin/python "./tests/run_golden.py"
2026-11-17T11:51:10.0Z  11/40 Test   #8: rocke_golden_static ..........***Failed   34.00 sec
2026-11-17T11:51:20.0Z  12/40 Test   #9: hip_kernel_provider_tests ....   Passed   10.00 sec
"""


def test_entry_seconds_scale_a_split_entry_by_its_shard_count(tmp_path) -> None:
    log = tmp_path / "shard3.log"
    log.write_text(_SHARD_LOG, encoding="utf-8")
    assert check.entry_seconds_from_ctest_logs([log]) == {
        "rocke_golden_static": 34.0,
        "rocke_pytest_core": 122.0,
    }


# The same entry run in worker processes (rocke.tests.ctest_workers): the entry
# prints the shard header itself, and its workers' lines carry a [w<index>]
# prefix.
_WORKERS_LOG = """\
2026-11-17T11:50:01.3Z 7: Test command: /venv/bin/python "-m" "pytest" "-p" \
"rocke.tests.ctest_shard" "--rocke-shard-from-gtest-env" "-p" \
"rocke.tests.ctest_workers" "--rocke-workers" "4" "./tests/core"
2026-11-17T11:50:01.9Z 7: rocke-ctest-shard: 3/4 (GTEST_SHARD_INDEX=3, \
GTEST_TOTAL_SHARDS=4)
2026-11-17T11:50:02.0Z 7: rocke-ctest-workers: 4 workers, 4 CPUs each
2026-11-17T11:50:02.9Z 7: [w0] rocke-ctest-shard: 3/4 (GTEST_SHARD_INDEX=3, \
GTEST_TOTAL_SHARDS=4)
2026-11-17T11:50:12.2Z  10/40 Test   #7: rocke_pytest_core ............   Passed   10.50 sec
"""


def test_entry_seconds_scale_an_entry_run_in_workers_by_its_shard_count(
    tmp_path,
) -> None:
    log = tmp_path / "workers.log"
    log.write_text(_WORKERS_LOG, encoding="utf-8")
    assert check.entry_seconds_from_ctest_logs([log]) == {"rocke_pytest_core": 42.0}


def test_entry_seconds_read_a_log_without_timestamps(tmp_path) -> None:
    log = tmp_path / "local.log"
    log.write_text(
        re.sub(r"^\S+Z ", "", _SHARD_LOG, flags=re.MULTILINE), encoding="utf-8"
    )
    assert check.entry_seconds_from_ctest_logs([log]) == {
        "rocke_golden_static": 34.0,
        "rocke_pytest_core": 122.0,
    }


def test_entry_seconds_take_the_median_over_logs_including_gzip(tmp_path) -> None:
    logs = []
    for index, golden_seconds in enumerate(("34.00", "20.00", "50.00")):
        log = tmp_path / f"job{index}.log.gz"
        with gzip.open(log, "wt", encoding="utf-8") as handle:
            handle.write(_SHARD_LOG.replace("34.00", golden_seconds))
        logs.append(log)
    assert check.entry_seconds_from_ctest_logs(logs)["rocke_golden_static"] == 34.0


def _shard_log(number: int, index: int, seconds: float) -> str:
    """One CI shard's log of a split entry, shard ``index`` of 4."""
    return (
        f'{number}: Test command: /venv/bin/python "-m" "pytest" "-p" '
        f'"rocke.tests.ctest_shard" "{ACTIVATION_FLAG}" "./tests"\n'
        f"{number}: rocke-ctest-shard: {index}/4 (GTEST_SHARD_INDEX={index}, "
        "GTEST_TOTAL_SHARDS=4)\n"
        f" 1/9 Test #{number}: rocke_library_gpu_pytest ....   Passed  "
        f"{seconds:.2f} sec\n"
    )


def test_a_split_entry_whole_time_sums_the_median_of_each_shard(tmp_path) -> None:
    logs = []
    for index, seconds in enumerate((150.0, 750.0, 150.0, 150.0)):
        logs.append(tmp_path / f"shard{index}.log")
        logs[-1].write_text(_shard_log(7, index, seconds), encoding="utf-8")
    # A second run of shard 2 takes longer; that shard's median is 760 s.
    logs.append(tmp_path / "shard1-again.log")
    logs[-1].write_text(_shard_log(7, 1, 770.0), encoding="utf-8")
    whole = check.entry_seconds_from_ctest_logs(logs)["rocke_library_gpu_pytest"]
    assert whole == 150.0 + 760.0 + 150.0 + 150.0


def test_without_every_shard_the_whole_time_is_the_count_times_the_median(
    tmp_path,
) -> None:
    logs = []
    for index, seconds in ((0, 100.0), (1, 300.0), (3, 200.0)):
        logs.append(tmp_path / f"shard{index}.log")
        logs[-1].write_text(_shard_log(7, index, seconds), encoding="utf-8")
    whole = check.entry_seconds_from_ctest_logs(logs)["rocke_library_gpu_pytest"]
    assert whole == 4 * 200.0


# A run with `--durations=0`: each worker reports its tests' setup, call and
# teardown seconds; entry 9 is not rocKE's.
_DURATIONS_LOG = """\
2026-11-17T11:50:20.0Z 7: [w0] ============ slowest durations ============
2026-11-17T11:50:20.0Z 7: [w0] 60.91s call     test_dgrad.py::TestDgrad::test_case[fp16-stride2]
2026-11-17T11:50:20.0Z 7: [w0] 0.40s setup    test_dgrad.py::TestDgrad::test_case[fp16-stride2]
2026-11-17T11:50:20.0Z 7: [w1] 0.50s call     test_dgrad.py::TestDgrad::test_case[fp16-stride1]
2026-11-17T11:50:20.0Z 9: 12.00s call     test_other.py::test_provider
2026-11-17T11:50:35.2Z  10/40 Test   #7: rocke_library_gpu_pytest .....   Passed   30.50 sec
2026-11-17T11:50:36.2Z  11/40 Test   #9: hip_kernel_provider_tests ....   Passed   12.50 sec
"""


def test_test_seconds_sum_each_slow_test_of_a_rocke_entry(tmp_path) -> None:
    log = tmp_path / "durations.log"
    log.write_text(_DURATIONS_LOG, encoding="utf-8")
    assert check.slow_test_seconds_from_ctest_logs([log]) == {
        "rocke_library_gpu_pytest": {
            "test_dgrad.py::TestDgrad::test_case[fp16-stride2]": 61.31
        }
    }


def _data_options(times: Path, committed: Path | None = None) -> list[str]:
    """Project from ``times``; ``committed`` (``times`` itself by default) is the
    data the check may not lose a registered entry's time from."""
    return ["--entry-times", str(times), "--committed-times", str(committed or times)]


def _write_inputs(tmp_path: Path, seconds: float) -> tuple[Path, Path]:
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps(_listing(_test("rocke_golden_static", ["quick"], WHOLE_COMMAND))),
        encoding="utf-8",
    )
    times = tmp_path / "entry_times.json"
    times.write_text(json.dumps(_column(rocke_golden_static=seconds)), encoding="utf-8")
    return listing, times


@pytest.mark.parametrize("seconds, exit_code", [(250.0, 0), (350.0, 1)])
def test_check_exits_nonzero_exactly_when_a_tier_is_over_budget(
    tmp_path, capsys, seconds: float, exit_code: int
) -> None:
    listing, times = _write_inputs(tmp_path, seconds)
    argv = [
        *_data_options(times),
        "check",
        "--column",
        "linux",
        "--listing",
        str(listing),
    ]
    assert check.main(argv) == exit_code
    assert ("FAIL linux quick" in capsys.readouterr().out) == bool(exit_code)


def test_check_fails_a_new_entry_until_its_time_is_recorded(tmp_path, capsys) -> None:
    listing, times = _write_inputs(tmp_path, 20.0)
    listing.write_text(
        json.dumps(
            _listing(
                _test("rocke_golden_static", ["quick"], WHOLE_COMMAND),
                _test("rocke_new_entry", ["standard"], WHOLE_COMMAND),
            )
        ),
        encoding="utf-8",
    )
    argv = [
        *_data_options(times),
        "check",
        "--column",
        "linux",
        "--listing",
        str(listing),
    ]
    assert check.main(argv) == 1
    assert "FAIL linux: rocke_new_entry has no time in the data" in (
        capsys.readouterr().out
    )
    log = tmp_path / "new_entry.log"
    log.write_text(
        " 1/1 Test   #1: rocke_new_entry ....   Passed    0.50 sec\n",
        encoding="utf-8",
    )
    record = ["--entry-times", str(times), "record", "--column", "linux"]
    assert check.main([*record, "--source", "test log", str(log)]) == 0
    assert check.main(argv) == 0


def _write_sweep_tree(tree: Path, declare_heavy_costs: bool) -> None:
    """A test module of one test per key of ``HEAVY`` and ``LIGHT``, the heavy
    ones with a ``rocke_shard_cost`` mark when ``declare_heavy_costs``."""
    heavy_mark = (
        f"marks=pytest.mark.rocke_shard_cost({HEAVY_SECONDS}), "
        if declare_heavy_costs
        else ""
    )
    cases = [
        f"pytest.param({key[len('test_sweep.py[') : -1]!r}, "
        f"{heavy_mark if key in HEAVY else ''}id={key[len('test_sweep.py[') : -1]!r})"
        for key in HEAVY + LIGHT
    ]
    tree.mkdir(exist_ok=True)
    (tree / "test_sweep.py").write_text(
        "import pytest\n\n\n"
        f"@pytest.mark.parametrize('case', [{', '.join(cases)}])\n"
        "def test_case(case):\n"
        "    pass\n",
        encoding="utf-8",
    )


_ROCKE_PATH = f"PYTHONPATH={Path(rocke.__file__).resolve().parents[1]}"


def _sweep_entry(
    tree: Path,
    arguments: tuple[str, ...] = ("test_sweep.py",),
    environment: tuple[str, ...] = (),
    workers: int = 4,
    **properties,
) -> dict:
    """A split pytest entry over ``tree`` as an installed registration lists
    it, run in ``workers`` processes as the GPU entries are (one: without the
    workers plugin), with the rocke package on its path."""
    in_workers = ("-p", "rocke.tests.ctest_workers", WORKERS_OPTION, str(workers))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "rocke.tests.ctest_shard",
        ACTIVATION_FLAG,
        *(in_workers if workers > 1 else ()),
        *arguments,
    ]
    return _test(
        "rocke_library_gpu_pytest",
        ["standard"],
        command,
        ENVIRONMENT=[_ROCKE_PATH, *environment],
        WORKING_DIRECTORY=str(tree),
        **properties,
    )


def test_collect_entry_keys_lists_keys_and_declared_costs(tmp_path) -> None:
    _write_sweep_tree(tmp_path / "tree", declare_heavy_costs=True)
    [entry] = check.registered_entries([_sweep_entry(tmp_path / "tree")])
    keys = check.collect_entry_keys(entry)
    assert keys.key_of_case == {_test_of(key): key for key in HEAVY + LIGHT}
    assert keys.costs == dict.fromkeys(HEAVY, HEAVY_SECONDS)


def test_collect_entry_keys_is_none_when_the_entry_skips_on_this_host(
    tmp_path,
) -> None:
    tree = tmp_path / "tree"
    _write_sweep_tree(tree, declare_heavy_costs=False)
    (tree / "conftest.py").write_text(
        "import pytest\n\n\n"
        "def pytest_configure(config):\n"
        "    pytest.exit('another target', returncode=77)\n",
        encoding="utf-8",
    )
    [entry] = check.registered_entries([_sweep_entry(tree, SKIP_RETURN_CODE=77)])
    assert check.collect_entry_keys(entry) is None


def test_collect_entry_keys_refuses_a_collection_that_fails(tmp_path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "test_sweep.py").write_text("def test_broken(:\n", encoding="utf-8")
    [entry] = check.registered_entries([_sweep_entry(tree)])
    with pytest.raises(
        RuntimeError,
        match="collect_entry_keys failed - rocke_library_gpu_pytest returncode=2 "
        "with the key listing written, expected 0 with the listing written",
    ):
        check.collect_entry_keys(entry)


def test_collect_entry_keys_lists_the_tests_that_run_alone(tmp_path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "test_sweep.py").write_text(
        "import pytest\n\n\n"
        "@pytest.mark.rocke_run_alone\n"
        "@pytest.mark.rocke_shard_cost(8)\n"
        "def test_large():\n"
        "    pass\n\n\n"
        "@pytest.mark.rocke_shard_cost(3)\n"
        "def test_small():\n"
        "    pass\n",
        encoding="utf-8",
    )
    [entry] = check.registered_entries([_sweep_entry(tree)])
    keys = check.collect_entry_keys(entry)
    large, small = "test_sweep.py::test_large", "test_sweep.py::test_small"
    assert keys.run_alone_cases == frozenset({large})
    assert keys.costs == {large: 8, small: 3}
    assert keys.worker_costs == {small: 3}


def test_collect_entry_keys_collects_every_test_whatever_shard_it_runs_in(
    tmp_path,
) -> None:
    _write_sweep_tree(tmp_path / "tree", declare_heavy_costs=False)
    shard_variables = (
        "GTEST_SHARD_INDEX=1",
        "GTEST_TOTAL_SHARDS=4",
        "ROCKE_TEST_SHARD=0/4",
        "ROCKE_TEST_WORKER=0/4",
    )
    entry_listing = _sweep_entry(tmp_path / "tree", environment=shard_variables)
    [entry] = check.registered_entries([entry_listing])
    keys = check.collect_entry_keys(entry)
    assert sorted(keys.key_of_case.values()) == sorted(HEAVY + LIGHT)


def test_collect_entry_keys_imports_from_the_working_directory_first(
    tmp_path,
) -> None:
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "helper_in_working_directory.py").write_text(
        "CASES = ['one', 'two']\n", encoding="utf-8"
    )
    (tree / "tests" / "test_sweep.py").write_text(
        "import pytest\n"
        "from helper_in_working_directory import CASES\n\n\n"
        "@pytest.mark.parametrize('case', CASES)\n"
        "def test_case(case):\n"
        "    pass\n",
        encoding="utf-8",
    )
    entry_listing = _sweep_entry(tree, arguments=("tests/test_sweep.py",))
    [entry] = check.registered_entries([entry_listing])
    keys = check.collect_entry_keys(entry)
    assert sorted(keys.key_of_case.values()) == [
        "tests/test_sweep.py[one]",
        "tests/test_sweep.py[two]",
    ]


def test_collect_entry_keys_refuses_an_entry_that_does_not_split() -> None:
    [entry] = check.registered_entries(
        [_test("rocke_golden_whole", ["standard"], WHOLE_COMMAND)]
    )
    with pytest.raises(
        ValueError,
        match="collect_entry_keys failed - rocke_golden_whole command="
        "'python tests/run_golden.py', expected a split entry "
        "\\(--rocke-shard-from-gtest-env\\)",
    ):
        check.collect_entry_keys(entry)


def test_collect_entry_keys_refuses_a_run_that_lists_no_selection(tmp_path) -> None:
    tree = tmp_path / "tree"
    _write_sweep_tree(tree, declare_heavy_costs=False)
    [entry] = check.registered_entries([_sweep_entry(tree, arguments=("--help",))])
    with pytest.raises(
        RuntimeError,
        match="collect_entry_keys failed - rocke_library_gpu_pytest returncode=0 "
        "with the key listing not written, expected 0 with the listing written",
    ):
        check.collect_entry_keys(entry)


def test_collect_entry_keys_refuses_a_collection_past_its_timeout(tmp_path) -> None:
    tree = tmp_path / "tree"
    _write_sweep_tree(tree, declare_heavy_costs=False)
    (tree / "conftest.py").write_text(
        "import time\n\ntime.sleep(30)\n", encoding="utf-8"
    )
    [entry] = check.registered_entries([_sweep_entry(tree)])
    with pytest.raises(
        RuntimeError,
        match="collect_entry_keys failed - rocke_library_gpu_pytest "
        "timeout_seconds=0.5, expected its listing to finish within it",
    ):
        check.collect_entry_keys(entry, timeout_seconds=0.5)


def test_registered_entries_refuses_an_environment_modification() -> None:
    listed = _test(
        "rocke_x",
        ["standard"],
        SPLIT_COMMAND,
        ENVIRONMENT_MODIFICATION=["PATH=path_list_prepend:/bin"],
    )
    with pytest.raises(
        ValueError,
        match="registered_entries failed - rocke_x has ENVIRONMENT_MODIFICATION, "
        "expected ENVIRONMENT only",
    ):
        check.registered_entries([listed])


@pytest.mark.parametrize("value", ["four", "0"])
def test_registered_entries_refuses_a_worker_count_that_is_not_one_or_more(
    value: str,
) -> None:
    listed = _test("rocke_x", ["standard"], [*SPLIT_COMMAND, WORKERS_OPTION, value])
    expected = (
        f"{WORKERS_OPTION}='four', expected a whole number"
        if value == "four"
        else f"{WORKERS_OPTION}=0, expected at least 1"
    )
    with pytest.raises(
        ValueError, match=f"RegisteredEntry failed - rocke_x {expected}"
    ):
        check.registered_entries([listed])


def _golden_runner_entry(tree: Path, case_ids: list[str]) -> dict:
    """A split entry that is not a pytest run: a runner whose cases are each
    their own shard key, as the golden runners' are, and that writes them under
    ``LIST_KEYS_FLAG``."""
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "run_golden.py").write_text(
        "import sys\n"
        "from pathlib import Path\n\n"
        "from rocke.tests.ci_shard import LIST_KEYS_FLAG, ShardKeyListing\n\n"
        f"CASES = {case_ids!r}\n"
        "path = Path(sys.argv[sys.argv.index(LIST_KEYS_FLAG) + 1])\n"
        "ShardKeyListing({case: case for case in CASES}).write(path)\n",
        encoding="utf-8",
    )
    command = [sys.executable, "run_golden.py", ACTIVATION_FLAG]
    return _test(
        "rocke_golden_split",
        ["standard"],
        command,
        ENVIRONMENT=[_ROCKE_PATH],
        WORKING_DIRECTORY=str(tree),
    )


def test_a_golden_runner_lists_its_cases_through_the_shared_flag(tmp_path) -> None:
    cases = _keys_on_shard("case", 0, 3) + _keys_on_shard("case", 2, 1)
    [entry] = check.registered_entries([_golden_runner_entry(tmp_path, cases)])
    keys = check.collect_entry_keys(entry)
    assert keys == ShardKeyListing({case: case for case in cases})
    # Cases without a time count alike: three on the first shard, one on the
    # third.
    assert check.split_entry_shard_seconds(40.0, 4, keys=keys) == (
        30.0,
        0.0,
        10.0,
        0.0,
    )


def test_a_split_runner_that_does_not_list_its_cases_fails_the_check(
    tmp_path,
) -> None:
    runner = _golden_runner_entry(tmp_path, ["case"])
    (tmp_path / "run_golden.py").write_text("print('checked')\n", encoding="utf-8")
    [entry] = check.registered_entries([runner])
    with pytest.raises(
        RuntimeError,
        match="collect_entry_keys failed - rocke_golden_split returncode=0 with "
        "the key listing not written, expected 0 with the listing written",
    ):
        check.collect_entry_keys(entry)


def test_check_fails_the_slowest_shard_until_heavy_keys_declare_costs(
    tmp_path, capsys
) -> None:
    """End to end: heavy tests the hash puts on one shard fail that shard
    though the average shard fits; declaring their costs spreads them."""
    tree = tmp_path / "tree"
    _write_sweep_tree(tree, declare_heavy_costs=False)
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps(_listing(_sweep_entry(tree, workers=1))), encoding="utf-8"
    )
    times = tmp_path / "entry_times.json"
    times.write_text(
        json.dumps(
            _column(
                test_seconds={"rocke_library_gpu_pytest": SWEEP_TEST_SECONDS},
                rocke_library_gpu_pytest=SWEEP_WHOLE_SECONDS,
            )
        ),
        encoding="utf-8",
    )
    argv = [
        *_data_options(times),
        "check",
        "--column",
        "linux",
        "--listing",
        str(listing),
    ]
    assert check.main(argv) == 1
    out = capsys.readouterr().out
    assert "150 / 750 / 150 / 150 s per shard" in out
    assert "FAIL linux standard shard 2 of 4 takes 750 s, over the 540 s budget" in out
    _write_sweep_tree(tree, declare_heavy_costs=True)
    assert check.main(argv) == 0
    assert "300 / 300 / 300 / 300 s per shard" in capsys.readouterr().out


def test_record_fills_a_new_column_from_logs(tmp_path) -> None:
    _, times = _write_inputs(tmp_path, 250.0)
    log = tmp_path / "shard3.log"
    log.write_text(_SHARD_LOG, encoding="utf-8")
    argv = ["--entry-times", str(times), "record", "--column", "windows"]
    assert check.main([*argv, "--source", "test logs", str(log)]) == 0
    recorded = json.loads(times.read_text(encoding="utf-8"))
    assert recorded["linux"]["seconds"] == {"rocke_golden_static": 250.0}
    assert recorded["windows"] == {
        "source": "test logs",
        "seconds": {"rocke_golden_static": 34.0, "rocke_pytest_core": 122.0},
    }


def test_record_updates_logged_entries_and_keeps_the_rest(tmp_path) -> None:
    times = tmp_path / "entry_times.json"
    column = _column(
        test_seconds={"rocke_not_in_logs": {"t.py::test": 9.0}},
        rocke_golden_static=250.0,
        rocke_not_in_logs=12.0,
    )
    times.write_text(json.dumps(column), encoding="utf-8")
    shard_log = tmp_path / "shard3.log"
    shard_log.write_text(_SHARD_LOG, encoding="utf-8")
    durations_log = tmp_path / "durations.log"
    durations_log.write_text(_DURATIONS_LOG, encoding="utf-8")
    argv = ["--entry-times", str(times), "record", "--column", "linux"]
    logs = [str(shard_log), str(durations_log)]
    assert check.main([*argv, "--source", "test logs", *logs]) == 0
    recorded = json.loads(times.read_text(encoding="utf-8"))["linux"]
    assert recorded["seconds"] == {
        "rocke_golden_static": 34.0,
        "rocke_library_gpu_pytest": 30.5,
        "rocke_not_in_logs": 12.0,
        "rocke_pytest_core": 122.0,
    }
    assert recorded["test_seconds"] == {
        "rocke_library_gpu_pytest": {
            "test_dgrad.py::TestDgrad::test_case[fp16-stride2]": 61.31
        },
        "rocke_not_in_logs": {"t.py::test": 9.0},
    }


def test_times_dropped_from_names_registered_entries_only() -> None:
    entries = check.registered_entries(
        [_test("rocke_golden_static", ["quick"], WHOLE_COMMAND)]
    )
    committed = _column(rocke_golden_static=30.0, rocke_unregistered=5.0)
    assert check.times_dropped_from(entries, committed, committed) == []
    dropped = (
        "linux: rocke_golden_static has a committed time but none in the projected "
        "data"
    )
    assert check.times_dropped_from(entries, committed, _column()) == [dropped]


# The line a run that checks prints first; a shard that leaves the check to
# another shard prints only the shard lines.
CHECKED = "linux quick (1 shard(s))"


def _check_entry(tmp_path: Path) -> check.RegisteredEntry:
    """The check's own entry as its registration runs it, against a
    registration that does not exist: listing its case must not read it."""
    command = [sys.executable, str(Path(check.__file__).resolve()), "check"]
    command += ["--listing", str(tmp_path / "absent.json"), ACTIVATION_FLAG]
    listed = _test("rocke_entry_time_check", ["standard"], command)
    [entry] = check.registered_entries([listed])
    return entry


def test_the_check_lists_its_one_case_through_its_registered_command(
    tmp_path,
) -> None:
    keys = check.collect_entry_keys(_check_entry(tmp_path))
    case = "rocke_entry_time_check"
    assert dict(keys.key_of_case) == {case: case}
    assert set(keys.costs) == {case}
    placed = place_keys(keys.key_of_case.values(), 4, keys.costs)[case]
    per_shard = check.split_entry_shard_seconds(12.0, 4, keys=keys)
    assert per_shard == tuple(12.0 if i == placed else 0.0 for i in range(4))


def _run_check(tmp_path: Path, monkeypatch, capsys, **shard_variables) -> str:
    """The output of a check run with the shard flag under ``shard_variables``
    and no other shard variable."""
    for name in ("GTEST_SHARD_INDEX", "GTEST_TOTAL_SHARDS", "ROCKE_TEST_SHARD"):
        monkeypatch.delenv(name, raising=False)
    for name, value in shard_variables.items():
        monkeypatch.setenv(name, value)
    listing, times = _write_inputs(tmp_path, 20.0)
    argv = [*_data_options(times), "check", "--column", "linux"]
    capsys.readouterr()
    assert check.main([*argv, "--listing", str(listing), ACTIVATION_FLAG]) == 0
    return capsys.readouterr().out


def _placed_shard(tmp_path: Path) -> int:
    keys = check.collect_entry_keys(_check_entry(tmp_path))
    [shard] = place_keys(keys.key_of_case.values(), 4, keys.costs).values()
    return shard


@pytest.mark.parametrize(
    "shard_variables",
    [{}, {"GTEST_SHARD_INDEX": "0", "GTEST_TOTAL_SHARDS": "1"}],
    ids=["unsharded", "one-shard-tier"],
)
def test_an_unsharded_or_one_shard_run_checks(
    tmp_path, monkeypatch, capsys, shard_variables: dict
) -> None:
    assert CHECKED in _run_check(tmp_path, monkeypatch, capsys, **shard_variables)


def test_a_four_shard_tier_checks_on_the_one_shard_that_keeps_the_check(
    tmp_path, monkeypatch, capsys
) -> None:
    placed = _placed_shard(tmp_path)
    checked = []
    for index in range(4):
        out = _run_check(
            tmp_path,
            monkeypatch,
            capsys,
            GTEST_SHARD_INDEX=str(index),
            GTEST_TOTAL_SHARDS="4",
        )
        split = "/".join("1" if i == placed else "0" for i in range(4))
        kept = int(index == placed)
        assert f"rocke-ctest-shard: {index}/4 (GTEST_SHARD_INDEX={index}" in out
        assert f"rocke-ctest-shard {index}/4: split {split} of 1; kept {kept}" in out
        if CHECKED in out:
            checked.append(index)
    assert checked == [placed]


def test_rocke_test_shard_wins_over_the_gtest_pair(
    tmp_path, monkeypatch, capsys
) -> None:
    placed = _placed_shard(tmp_path)
    other = (placed + 1) % 4
    gtest = {"GTEST_SHARD_INDEX": str(other), "GTEST_TOTAL_SHARDS": "4"}
    run = {"ROCKE_TEST_SHARD": f"{placed}/4", **gtest}
    assert CHECKED in _run_check(tmp_path, monkeypatch, capsys, **run)
    gtest = {"GTEST_SHARD_INDEX": str(placed), "GTEST_TOTAL_SHARDS": "4"}
    skip = {"ROCKE_TEST_SHARD": f"{other}/4", **gtest}
    assert CHECKED not in _run_check(tmp_path, monkeypatch, capsys, **skip)


@pytest.mark.parametrize(
    "flags, shard_variables, message",
    [
        (
            [LIST_KEYS_FLAG, "keys.json"],
            {},
            f"{LIST_KEYS_FLAG} needs {ACTIVATION_FLAG}",
        ),
        (
            [ACTIVATION_FLAG],
            {"GTEST_SHARD_INDEX": "4", "GTEST_TOTAL_SHARDS": "4"},
            "rocke-ctest-shard: GTEST_SHARD_INDEX='4' is not in 0..3",
        ),
    ],
    ids=["list-without-shard-flag", "bad-gtest-pair"],
)
def test_check_refuses_a_listing_or_shard_it_cannot_use(
    tmp_path, monkeypatch, capsys, flags, shard_variables, message
) -> None:
    for name in ("GTEST_SHARD_INDEX", "GTEST_TOTAL_SHARDS", "ROCKE_TEST_SHARD"):
        monkeypatch.delenv(name, raising=False)
    for name, value in shard_variables.items():
        monkeypatch.setenv(name, value)
    argv = ["check", "--listing", str(tmp_path / "absent.json"), *flags]
    with pytest.raises(SystemExit) as refused:
        check.main(argv)
    assert refused.value.code == 2
    assert message in capsys.readouterr().err


def test_check_judges_only_the_column_of_the_listed_registration(
    tmp_path, capsys
) -> None:
    """An entry registered on one operating system only has no time in the
    other's column; that column is judged on its own registration."""
    listing, times = _write_inputs(tmp_path, 20.0)
    both = {
        "linux": {"source": "test data", "seconds": {"rocke_golden_static": 20.0}},
        "windows": {"source": "test data", "seconds": {}},
    }
    times.write_text(json.dumps(both), encoding="utf-8")
    check_column = [*_data_options(times), "check", "--listing", str(listing)]
    assert (
        check.main([*check_column[:-2], "--column", "linux", *check_column[-2:]]) == 0
    )
    assert "windows" not in capsys.readouterr().out
    assert (
        check.main([*check_column[:-2], "--column", "windows", *check_column[-2:]]) == 1
    )
    assert "FAIL windows: rocke_golden_static has no time" in capsys.readouterr().out
    with pytest.raises(KeyError, match="check failed - column='macos', expected one"):
        check.main([*check_column[:-2], "--column", "macos", *check_column[-2:]])


def test_the_default_column_is_this_operating_system() -> None:
    expected = "windows" if sys.platform == "win32" else "linux"
    assert check.host_column() == expected


def test_check_fails_when_the_projection_drops_a_committed_time(
    tmp_path, capsys
) -> None:
    listing, committed = _write_inputs(tmp_path, 30.0)
    projected = tmp_path / "projected.json"
    projected.write_text(json.dumps(_column()), encoding="utf-8")
    argv = [
        *_data_options(projected, committed),
        "check",
        "--column",
        "linux",
        "--listing",
        str(listing),
    ]
    assert check.main(argv) == 1
    assert "FAIL linux: rocke_golden_static has a committed time" in (
        capsys.readouterr().out
    )


def test_record_refuses_logs_without_rocke_entries(tmp_path) -> None:
    _, times = _write_inputs(tmp_path, 250.0)
    log = tmp_path / "other.log"
    log.write_text("no ctest output here\n", encoding="utf-8")
    argv = ["--entry-times", str(times), "record", "--column", "linux"]
    with pytest.raises(ValueError, match="record failed - logs contain no rocke_"):
        check.main([*argv, "--source", "empty", str(log)])


def test_committed_entry_times_have_a_source_and_seconds_per_column() -> None:
    entry_times = check.load_entry_times()
    assert set(entry_times) == {"linux", "windows"}
    tier_labels = {tier.label for tier in check.CI_TIERS}
    for column in entry_times.values():
        assert column["source"]
        assert column["seconds"]
        assert all(seconds >= 0 for seconds in column["seconds"].values())
        assert set(column.get("ceilings", {})) <= tier_labels
        for tests in column.get("test_seconds", {}).values():
            assert all(seconds > 0 for seconds in tests.values())


def test_committed_ceilings_only_go_down() -> None:
    """Ceilings are ratchets. This guard pins the committed values, so raising
    one means editing this test as well, in plain view of review; lowering one
    (``lower-ceiling``) only needs the bound here moved down with it."""
    ceilings = {
        name: column.get("ceilings", {})
        for name, column in check.load_entry_times().items()
    }
    # The Linux standard tier fits its budget, so it has no ceiling.
    assert ceilings["linux"] == {}
    assert ceilings["windows"].keys() == {"standard"}
    assert ceilings["windows"]["standard"] <= 540.0


def _standard_verdict(seconds: float, ceiling: float | None):
    entries = check.registered_entries(
        [_test("rocke_pytest_core", ["standard"], WHOLE_COMMAND)]
    )
    ceilings = None if ceiling is None else {"standard": ceiling}
    verdicts = check.evaluate_tiers(
        entries, _column(ceilings, rocke_pytest_core=seconds), tiers=[STANDARD]
    )
    return verdicts[0]


def test_a_ceiling_replaces_the_budget_as_the_shard_limit() -> None:
    within = _standard_verdict(410.0, ceiling=560.0)
    assert within.limit_seconds == 560.0
    assert within.summary() == (
        "linux standard (4 shard(s)): 410 / 410 / 410 / 410 s per shard, slowest "
        "410 s, ceiling 560 s (margin 150 s), 130 s under the 540 s budget"
    )
    assert within.breaches() == []
    over = _standard_verdict(570.0, ceiling=560.0)
    for number in range(1, 5):
        assert (
            f"linux standard shard {number} of 4 takes 570 s, over the 560 s "
            "ceiling; largest: rocke_pytest_core 570 s"
        ) in over.breaches()


def test_a_tier_without_a_ceiling_is_held_to_its_budget() -> None:
    verdict = _standard_verdict(550.0, ceiling=None)
    assert verdict.limit_seconds == 540.0
    assert verdict.summary().endswith("slowest 550 s, 10 s over the 540 s budget")


@pytest.mark.parametrize(
    "seconds, ceiling, lowered",
    [(653.0, 680.0, 660.0), (660.0, 680.0, 660.0), (400.0, 680.0, None)],
    ids=["rounds-up", "exact-step", "fits-budget"],
)
def test_lowered_ceiling_rounds_the_projection_up_or_drops_it(
    seconds: float, ceiling: float, lowered: float | None
) -> None:
    assert check.lowered_ceiling(_standard_verdict(seconds, ceiling)) == lowered


def test_lowered_ceiling_follows_the_slowest_shard() -> None:
    entries = check.registered_entries(
        [_test("rocke_library_gpu_pytest", ["standard"], SPLIT_COMMAND)]
    )
    times = _column(
        {"standard": 800.0},
        test_seconds={"rocke_library_gpu_pytest": SWEEP_TEST_SECONDS},
        rocke_library_gpu_pytest=SWEEP_WHOLE_SECONDS,
    )
    [verdict] = check.evaluate_tiers(
        entries, times, {"rocke_library_gpu_pytest": _sweep_keys()}, tiers=[STANDARD]
    )
    assert check.lowered_ceiling(verdict) == 750.0


@pytest.mark.parametrize(
    "seconds, ceiling, message",
    [
        (555.0, 560.0, r"ceiling=560 s, expected the projection rounded up \(560 s\)"),
        (600.0, 560.0, r"ceiling=560 s, expected the projection rounded up \(600 s\)"),
        (500.0, None, r"has ceiling=None, expected a ceiling to lower"),
    ],
    ids=["equal", "raise", "no-ceiling"],
)
def test_lowered_ceiling_refuses_anything_but_a_decrease(
    seconds: float, ceiling: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match="lowered_ceiling failed - .*" + message):
        check.lowered_ceiling(_standard_verdict(seconds, ceiling))


def _write_standard_inputs(tmp_path: Path, seconds: float, ceiling: float):
    listing = tmp_path / "listing.json"
    one_case_per_shard = [_keys_on_shard("case", shard, 1)[0] for shard in range(4)]
    runner = _golden_runner_entry(tmp_path / "runner", one_case_per_shard)
    listing.write_text(json.dumps(_listing(runner)), encoding="utf-8")
    times = tmp_path / "entry_times.json"
    column = _column({"standard": ceiling}, rocke_golden_split=seconds)
    times.write_text(json.dumps(column), encoding="utf-8")
    return ["--entry-times", str(times)], ["--listing", str(listing)], times


def test_lower_ceiling_writes_the_lower_value_and_record_keeps_it(tmp_path) -> None:
    options, source, times = _write_standard_inputs(tmp_path, 2400.0, 680.0)
    lower = [*options, "lower-ceiling", "--column", "linux", "--tier", "standard"]
    assert check.main([*lower, *source]) == 0
    assert json.loads(times.read_text())["linux"]["ceilings"] == {"standard": 600.0}
    log = tmp_path / "shard3.log"
    log.write_text(_SHARD_LOG, encoding="utf-8")
    record = [*options, "record", "--column", "linux", "--source", "logs", str(log)]
    assert check.main(record) == 0
    assert json.loads(times.read_text())["linux"]["ceilings"] == {"standard": 600.0}


def test_lower_ceiling_removes_a_ceiling_once_the_tier_fits(tmp_path) -> None:
    options, source, times = _write_standard_inputs(tmp_path, 1600.0, 680.0)
    lower = [*options, "lower-ceiling", "--column", "linux", "--tier", "standard"]
    assert check.main([*lower, *source]) == 0
    assert "ceilings" not in json.loads(times.read_text())["linux"]


def test_lower_ceiling_refuses_while_an_entry_has_no_time(tmp_path) -> None:
    options, _, times = _write_standard_inputs(tmp_path, 2400.0, 680.0)
    listing = tmp_path / "untimed.json"
    listing.write_text(
        json.dumps(
            _listing(
                _test("rocke_golden_split", ["standard"], SPLIT_RUNNER_COMMAND),
                _test("rocke_new_entry", ["standard"], WHOLE_COMMAND),
            )
        ),
        encoding="utf-8",
    )
    lower = [*options, "lower-ceiling", "--column", "linux", "--tier", "standard"]
    with pytest.raises(
        ValueError,
        match="lower-ceiling failed - standard untimed entries=1, expected 0 "
        "before a ceiling moves; linux: rocke_new_entry",
    ):
        check.main([*lower, "--listing", str(listing)])
    assert json.loads(times.read_text())["linux"]["ceilings"] == {"standard": 680.0}


def test_lower_ceiling_refuses_an_unknown_column(tmp_path) -> None:
    options, source, _ = _write_standard_inputs(tmp_path, 2400.0, 680.0)
    lower = [*options, "lower-ceiling", "--column", "macos", "--tier", "standard"]
    with pytest.raises(KeyError, match="lower-ceiling failed - column='macos'"):
        check.main([*lower, *source])
