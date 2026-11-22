# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The per-entry time check projects CI shards and fails a tier over budget."""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


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
    pytest.skip("top-level check_entry_times.py is absent from this platform-only tree")


check = _load_check_module()
QUICK, STANDARD = check.CI_TIERS
SPLIT_COMMAND = [
    "python",
    "-m",
    "pytest",
    "-p",
    "rocke.tests.ctest_shard",
    check.ACTIVATION_FLAG,
    "tests",
]
SPLIT_RUNNER_COMMAND = ["python", "tests/run_golden.py", check.ACTIVATION_FLAG]
WHOLE_COMMAND = ["python", "tests/run_golden.py"]


def _test(name: str, labels: list[str], command: list[str]) -> dict:
    """One entry as `ctest --show-only=json-v1` lists it."""
    return {
        "name": name,
        "command": command,
        "properties": [{"name": "LABELS", "value": labels}],
    }


def _listing(*tests: dict) -> dict:
    return {"kind": "ctestInfo", "tests": list(tests)}


def _column(ceilings: dict | None = None, **seconds: float) -> dict:
    column = {"source": "test data", "seconds": seconds}
    if ceilings is not None:
        column["ceilings"] = ceilings
    return {"linux": column}


def _verdict(verdicts, tier):
    [verdict] = [v for v in verdicts if v.tier == tier]
    return verdict


def test_registered_entries_reads_labels_and_sharding_of_rocke_entries() -> None:
    entries = check.registered_entries(
        [
            _test("rocke_pytest_core", ["quick", "standard"], SPLIT_COMMAND),
            _test("rocke_golden_static", ["quick"], WHOLE_COMMAND),
            _test("rocke_golden_split", ["standard"], SPLIT_RUNNER_COMMAND),
            _test("hip_kernel_provider_tests", ["quick"], WHOLE_COMMAND),
        ]
    )
    assert entries == [
        check.RegisteredEntry(
            "rocke_pytest_core", frozenset({"quick", "standard"}), True
        ),
        check.RegisteredEntry("rocke_golden_static", frozenset({"quick"}), False),
        check.RegisteredEntry("rocke_golden_split", frozenset({"standard"}), True),
    ]


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


def test_split_entries_divide_across_standard_shards_and_whole_entries_do_not() -> None:
    entries = check.registered_entries(
        [
            _test("rocke_pytest_core", ["quick", "standard"], SPLIT_COMMAND),
            _test("rocke_golden_static", ["quick", "standard"], WHOLE_COMMAND),
        ]
    )
    times = _column(rocke_pytest_core=400.0, rocke_golden_static=30.0)
    verdicts = check.evaluate_tiers(entries, times)
    assert _verdict(verdicts, QUICK).seconds_per_shard == {
        "rocke_pytest_core": 400.0,
        "rocke_golden_static": 30.0,
    }
    assert _verdict(verdicts, STANDARD).seconds_per_shard == {
        "rocke_pytest_core": 100.0,
        "rocke_golden_static": 30.0,
    }


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
        "linux quick: rocke_pytest_core alone takes 1900 s of a shard, over the "
        "300 s budget"
    )
    whole_shard = (
        "linux quick: one shard takes 1940 s, over the 300 s budget and over the "
        "1800 s step limit; largest: rocke_pytest_core 1900 s, "
        "rocke_golden_static 40 s"
    )
    assert verdict.breaches() == [entry_alone, whole_shard]


def test_a_tier_within_budget_has_no_breaches() -> None:
    entries = check.registered_entries(
        [_test("rocke_pytest_core", ["standard"], SPLIT_COMMAND)]
    )
    verdicts = check.evaluate_tiers(entries, _column(rocke_pytest_core=1600.0))
    assert [v.breaches() for v in verdicts] == [[], []]
    assert _verdict(verdicts, STANDARD).shard_seconds == 400.0


def test_excluded_and_unmeasured_entries_do_not_count() -> None:
    entries = check.registered_entries(
        [
            _test("rocke_gpu_only", ["quick", check.EXCLUDED_LABEL], WHOLE_COMMAND),
            _test("rocke_new_entry", ["quick"], WHOLE_COMMAND),
        ]
    )
    verdict = _verdict(
        check.evaluate_tiers(entries, _column(rocke_gpu_only=900.0)), QUICK
    )
    assert verdict.seconds_per_shard == {}
    assert verdict.unmeasured == ("rocke_new_entry",)
    assert verdict.breaches() == []


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
    argv = [*_data_options(times), "check", "--listing", str(listing)]
    assert check.main(argv) == exit_code
    assert ("FAIL linux quick" in capsys.readouterr().out) == bool(exit_code)


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
    column = _column(rocke_golden_static=250.0, rocke_not_in_logs=12.0)
    times.write_text(json.dumps(column), encoding="utf-8")
    log = tmp_path / "shard3.log"
    log.write_text(_SHARD_LOG, encoding="utf-8")
    argv = ["--entry-times", str(times), "record", "--column", "linux"]
    assert check.main([*argv, "--source", "test logs", str(log)]) == 0
    assert json.loads(times.read_text(encoding="utf-8"))["linux"]["seconds"] == {
        "rocke_golden_static": 34.0,
        "rocke_not_in_logs": 12.0,
        "rocke_pytest_core": 122.0,
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


def test_check_fails_when_the_projection_drops_a_committed_time(
    tmp_path, capsys
) -> None:
    listing, committed = _write_inputs(tmp_path, 30.0)
    projected = tmp_path / "projected.json"
    projected.write_text(json.dumps(_column()), encoding="utf-8")
    argv = [*_data_options(projected, committed), "check", "--listing", str(listing)]
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
        "linux standard (4 shard(s)): 410 s per shard, ceiling 560 s (margin "
        "150 s), 10 s under the 420 s budget"
    )
    assert within.breaches() == []
    over = _standard_verdict(570.0, ceiling=560.0)
    entry_alone = (
        "linux standard: rocke_pytest_core alone takes 570 s of a shard, over the "
        "420 s budget"
    )
    whole_shard = (
        "linux standard: one shard takes 570 s, over the 560 s ceiling; largest: "
        "rocke_pytest_core 570 s"
    )
    assert over.breaches() == [entry_alone, whole_shard]


def test_a_tier_without_a_ceiling_is_held_to_its_budget() -> None:
    verdict = _standard_verdict(430.0, ceiling=None)
    assert verdict.limit_seconds == 420.0
    assert verdict.summary().endswith("430 s per shard, 10 s over the 420 s budget")


@pytest.mark.parametrize(
    "seconds, ceiling, lowered",
    [(533.0, 560.0, 540.0), (540.0, 560.0, 540.0), (400.0, 560.0, None)],
    ids=["rounds-up", "exact-step", "fits-budget"],
)
def test_lowered_ceiling_rounds_the_projection_up_or_drops_it(
    seconds: float, ceiling: float, lowered: float | None
) -> None:
    assert check.lowered_ceiling(_standard_verdict(seconds, ceiling)) == lowered


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
    listing.write_text(
        json.dumps(_listing(_test("rocke_pytest_core", ["standard"], SPLIT_COMMAND))),
        encoding="utf-8",
    )
    times = tmp_path / "entry_times.json"
    column = _column({"standard": ceiling}, rocke_pytest_core=seconds)
    times.write_text(json.dumps(column), encoding="utf-8")
    return ["--entry-times", str(times)], ["--listing", str(listing)], times


def test_lower_ceiling_writes_the_lower_value_and_record_keeps_it(tmp_path) -> None:
    options, source, times = _write_standard_inputs(tmp_path, 2000.0, 560.0)
    lower = [*options, "lower-ceiling", "--column", "linux", "--tier", "standard"]
    assert check.main([*lower, *source]) == 0
    assert json.loads(times.read_text())["linux"]["ceilings"] == {"standard": 500.0}
    log = tmp_path / "shard3.log"
    log.write_text(_SHARD_LOG, encoding="utf-8")
    record = [*options, "record", "--column", "linux", "--source", "logs", str(log)]
    assert check.main(record) == 0
    assert json.loads(times.read_text())["linux"]["ceilings"] == {"standard": 500.0}


def test_lower_ceiling_removes_a_ceiling_once_the_tier_fits(tmp_path) -> None:
    options, source, times = _write_standard_inputs(tmp_path, 1600.0, 560.0)
    lower = [*options, "lower-ceiling", "--column", "linux", "--tier", "standard"]
    assert check.main([*lower, *source]) == 0
    assert "ceilings" not in json.loads(times.read_text())["linux"]


def test_lower_ceiling_refuses_an_unknown_column(tmp_path) -> None:
    options, source, _ = _write_standard_inputs(tmp_path, 2000.0, 560.0)
    lower = [*options, "lower-ceiling", "--column", "macos", "--tier", "standard"]
    with pytest.raises(KeyError, match="lower-ceiling failed - column='macos'"):
        check.main([*lower, *source])
