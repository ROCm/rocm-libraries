# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.run_records records what every pytest process of a run did with
each test, loaded through the environment alone: each test's events with its
file, node name and reason, how each test that ran ended, the reason a plugin
gives for a deselection, and the records of a child pytest process a test
starts.

Each case runs real pytest processes over a small test tree it writes.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from rocke.tests.lanes import EXHAUSTIVE_CASES, SOURCE_TREE_CASES, skip_until_reason
from rocke.tests.run_records import (
    OTHER_SHARD,
    RAN_OUTCOMES,
    RECORDS_DIR_ENV,
    UNFINISHED,
    RunRecord,
    read_run_records,
)
from rocke.tests.subprocess_launcher import rocke_subprocess_env

_CHILD_TIMEOUT_SECONDS = 120


def _run_pytest(tree: Path, records: Path | None, *arguments: str) -> None:
    """Run pytest over ``tree`` with the plugin loaded through the environment,
    writing into ``records`` (None: the variable unset)."""
    environment = rocke_subprocess_env()
    environment["PYTEST_PLUGINS"] = "rocke.tests.run_records"
    environment.pop(RECORDS_DIR_ENV, None)
    if records is not None:
        environment[RECORDS_DIR_ENV] = str(records)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *arguments],
        cwd=tree,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
    assert result.returncode in (0, 1), result.stdout[-3000:] + result.stderr[-3000:]


def _write(tree: Path, name: str, source: str) -> Path:
    path = tree / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _events(records: Path) -> set[tuple[str, str, str, str]]:
    return {
        (Path(record.file).name, record.node, record.event, record.reason)
        for record in read_run_records(records)
    }


def test_each_test_event_is_recorded_with_its_node_and_reason(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(
        tree,
        "test_cases.py",
        """
        import pytest

        def test_passes():
            pass

        @pytest.mark.skipif(True, reason="needs a gfx950 GPU")
        def test_skipped_by_mark():
            pass

        def test_skipped_in_its_body():
            pytest.skip("could not import 'rocke_engine'")

        @pytest.mark.xfail(reason="known")
        def test_expected_to_fail():
            assert False

        @pytest.mark.integration
        def test_deselected():
            pass

        @pytest.mark.parametrize("width", [1, 2])
        def test_case(width):
            pass

        class TestGroup:
            def test_member(self):
                pass
        """,
    )
    _write(
        tree,
        "test_skipped_module.py",
        """
        import pytest
        pytest.importorskip("a_module_that_is_not_installed")
        """,
    )
    records = tmp_path / "records"
    _run_pytest(tree, records, "-m", "not integration", "-W", "ignore")
    events = _events(records)
    ran = {node for file, node, event, _ in events if event == "ran"}
    assert ran == {
        "test_passes",
        "test_expected_to_fail",
        "test_case[1]",
        "test_case[2]",
        "TestGroup::test_member",
    }
    assert (
        "test_cases.py",
        "test_skipped_by_mark",
        "skipped",
        "needs a gfx950 GPU",
    ) in events
    assert (
        "test_cases.py",
        "test_skipped_in_its_body",
        "skipped",
        "could not import 'rocke_engine'",
    ) in events
    assert (
        "test_cases.py",
        "test_deselected",
        "deselected",
        skip_until_reason(
            SOURCE_TREE_CASES, reason="deselected by -m 'not integration'"
        ),
    ) in events
    collected = {node for file, node, event, _ in events if event == "collected"}
    assert {
        "test_deselected",
        "test_skipped_by_mark",
        "TestGroup::test_member",
    } <= collected
    module_skips = [event for event in events if event[2] == "module_skipped"]
    assert module_skips == [
        (
            "test_skipped_module.py",
            "",
            "module_skipped",
            (
                "could not import 'a_module_that_is_not_installed': No module "
                "named 'a_module_that_is_not_installed'"
            ),
        )
    ]


def test_a_deselection_is_put_down_to_the_filter_that_made_it(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(
        tree,
        "conftest.py",
        """
        from rocke.tests.run_records import note_deselection

        def pytest_collection_modifyitems(config, items):
            dropped = [item for item in items if "gfx950" in item.name]
            for item in dropped:
                note_deselection(item, "deselected: a gfx950 case; this entry checks gfx942")
            quiet = [item for item in items if "quiet" in item.name]
            items[:] = [item for item in items if item not in dropped + quiet]
            config.hook.pytest_deselected(items=dropped + quiet)
        """,
    )
    _write(
        tree,
        "test_arches.py",
        """
        import pytest

        @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
        def test_reference(arch):
            pass

        def test_quiet():
            pass

        @pytest.mark.integration
        def test_marked():
            pass
        """,
    )
    records = tmp_path / "records"
    _run_pytest(tree, records, "-m", "not integration", "-W", "ignore")
    assert {
        (node, event, reason)
        for _, node, event, reason in _events(records)
        if event != "collected"
    } == {
        ("test_reference[gfx942]", "ran", "passed"),
        (
            "test_reference[gfx950]",
            "deselected",
            "deselected: a gfx950 case; this entry checks gfx942",
        ),
        # Removed under -m by a plugin that gave no reason: not put down to -m.
        ("test_quiet", "deselected", "deselected by a plugin that gave no reason"),
        (
            "test_marked",
            "deselected",
            skip_until_reason(
                SOURCE_TREE_CASES, reason="deselected by -m 'not integration'"
            ),
        ),
    }


def test_a_marker_deselection_declares_the_lane_whose_marker_rejects_it(tmp_path):
    """A test the -m expression rejects for a lane's marker (exhaustive,
    integration) waits for that lane; one it would reject without that marker
    too (a GPU test in a CPU entry) declares no lane."""
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(
        tree,
        "test_marks.py",
        """
        import pytest

        @pytest.mark.exhaustive
        def test_sweep():
            pass

        @pytest.mark.exhaustive
        @pytest.mark.gpu
        def test_gpu_sweep():
            pass

        @pytest.mark.gpu
        def test_on_gpu():
            pass

        def test_on_host():
            pass
        """,
    )
    records = tmp_path / "records"
    expression = "not exhaustive and not gpu"
    _run_pytest(tree, records, "-m", expression, "-W", "ignore")
    reason = f"deselected by -m {expression!r}"
    assert {
        (node, why) for _, node, event, why in _events(records) if event == "deselected"
    } == {
        ("test_sweep", skip_until_reason(EXHAUSTIVE_CASES, reason=reason)),
        ("test_gpu_sweep", reason),
        ("test_on_gpu", reason),
    }


def test_a_child_pytest_process_writes_records_of_its_own(tmp_path):
    """A child pytest that runs the product's test files (as test_portable_ir's
    test_unit and test_in_tree_suites do) inherits the records directory and
    records its tests in a file of its own."""
    tree = tmp_path / "tree"
    (tree / "child").mkdir(parents=True)
    child = _write(
        tree / "child", "test_child.py", "def test_in_the_child():\n    pass\n"
    )
    _write(
        tree,
        "test_parent.py",
        f"""
        import os, subprocess, sys

        def test_runs_a_child():
            subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", {str(child)!r}],
                env=os.environ, check=True,
            )
        """,
    )
    records = tmp_path / "records"
    _run_pytest(tree, records, "test_parent.py")
    assert len(list(records.glob("*.jsonl"))) == 2
    ran = {
        (Path(r.file).name, r.node)
        for r in read_run_records(records)
        if r.event == "ran"
    }
    assert ran == {
        ("test_parent.py", "test_runs_a_child"),
        ("test_child.py", "test_in_the_child"),
    }


def test_a_pytest_over_a_fixture_tree_records_nothing(tmp_path):
    """A test that runs pytest over a fixture tree of its own (here one whose
    test fails on purpose) starts it with fixture_session_environment: the
    run's records hold only the outer test."""
    tree = tmp_path / "tree"
    (tree / "fixture").mkdir(parents=True)
    fixture = _write(
        tree / "fixture",
        "test_fixture.py",
        "def test_meant_to_fail():\n    assert False\n",
    )
    _write(
        tree,
        "test_parent.py",
        f"""
        import subprocess, sys
        from rocke.tests.run_records import fixture_session_environment

        def test_runs_a_fixture_session():
            env = fixture_session_environment()
            assert env["PYTEST_PLUGINS"] == "rocke.tests.run_records"
            subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", {str(fixture)!r}],
                env=env,
                check=False,
            )
        """,
    )
    records = tmp_path / "records"
    _run_pytest(tree, records, "test_parent.py")
    assert len(list(records.glob("*.jsonl"))) == 1
    assert {
        (Path(record.file).name, record.node, record.event)
        for record in read_run_records(records)
    } == {
        ("test_parent.py", "test_runs_a_fixture_session", "collected"),
        ("test_parent.py", "test_runs_a_fixture_session", "ran"),
    }


def test_a_test_that_ran_is_recorded_with_how_it_ended(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(
        tree,
        "test_outcomes.py",
        """
        import pytest

        @pytest.fixture
        def broken_setup():
            raise RuntimeError("setup")

        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("teardown")

        def test_passes():
            pass

        def test_fails():
            assert False

        @pytest.mark.xfail(reason="known")
        def test_fails_as_expected():
            assert False

        @pytest.mark.xfail(reason="known", strict=False)
        def test_passes_though_expected_to_fail():
            pass

        @pytest.mark.xfail(reason="known", strict=True)
        def test_passes_though_strictly_expected_to_fail():
            pass

        def test_setup_raises(broken_setup):
            pass

        def test_passes_then_teardown_raises(broken_teardown):
            pass
        """,
    )
    records = tmp_path / "records"
    _run_pytest(tree, records)
    ran = sorted(
        (node, reason) for _, node, event, reason in _events(records) if event == "ran"
    )
    assert ran == [
        ("test_fails", "failed"),
        ("test_fails_as_expected", "xfailed"),
        ("test_passes", "passed"),
        ("test_passes_then_teardown_raises", "error"),
        ("test_passes_then_teardown_raises", "passed"),
        ("test_passes_though_expected_to_fail", "xpassed"),
        ("test_passes_though_strictly_expected_to_fail", "failed"),
        ("test_setup_raises", "error"),
    ]
    assert {reason for _, reason in ran} == RAN_OUTCOMES


def test_a_process_that_ends_before_its_session_finishes_leaves_its_tests_unfinished(
    tmp_path,
):
    """A process that a per-test time limit ends with os._exit keeps the
    records it wrote; the test it was in and the tests after it are
    ``unfinished``, not taken for tests a filter dropped."""
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(
        tree,
        "test_ends.py",
        """
        import os

        def test_before():
            pass

        def test_ends_the_process():
            os._exit(1)

        def test_after():
            pass
        """,
    )
    _run_pytest(tree, tmp_path / "records")
    assert _events(tmp_path / "records") >= {
        ("test_ends.py", "test_before", "ran", "passed"),
        ("test_ends.py", "test_ends_the_process", "unfinished", UNFINISHED),
        ("test_ends.py", "test_after", "unfinished", UNFINISHED),
    }
    assert ("test_ends.py", "test_before", "unfinished", UNFINISHED) not in (
        _events(tmp_path / "records")
    )

    _write(tree, "test_ends.py", "def test_before():\n    pass\n")
    _run_pytest(tree, tmp_path / "finished")
    assert "unfinished" not in {
        event for _, _, event, _ in _events(tmp_path / "finished")
    }


def test_nothing_is_written_without_the_records_directory(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(tree, "test_one.py", "def test_x():\n    pass\n")
    _run_pytest(tree, None)
    assert not list(tmp_path.rglob("*.jsonl"))


def test_a_record_names_a_node_exactly_for_a_test_level_event():
    with pytest.raises(ValueError, match="event='passed', expected one of"):
        RunRecord("/t/test_x.py", "test_x", "passed")
    with pytest.raises(ValueError, match="expected a node name exactly"):
        RunRecord("/t/test_x.py", "", "skipped", "why")
    with pytest.raises(ValueError, match="expected a node name exactly"):
        RunRecord("/t/test_x.py", "test_x", "module_skipped", "why")


def test_reading_a_run_needs_its_records_directory(tmp_path):
    with pytest.raises(
        ValueError, match="expected the directory ROCKE_RUN_RECORDS_DIR"
    ):
        list(read_run_records(tmp_path / "absent"))


def test_a_test_another_shard_runs_is_recorded_as_left_to_it(tmp_path):
    """The shard plugin says why it deselects a test: another shard of the
    entry runs it, so a lane run in part does not read as one that left the
    test out."""
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(
        tree, "test_many.py", "".join(f"def test_{i}():\n    pass\n" for i in range(8))
    )
    records = tmp_path / "records"
    environment = rocke_subprocess_env()
    environment.update(
        PYTEST_PLUGINS="rocke.tests.run_records",
        ROCKE_TEST_SHARD="0/2",
        **{RECORDS_DIR_ENV: str(records)},
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
        + ["-p", "rocke.tests.ctest_shard"],
        cwd=tree,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CHILD_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    events = _events(records)
    ran = {node for _, node, event, _ in events if event == "ran"}
    left = {node for _, node, event, why in events if event == "deselected"}
    assert ran and left and ran.isdisjoint(left)
    assert ran | left == {f"test_{i}" for i in range(8)}
    assert {why for _, _, event, why in events if event == "deselected"} == {
        OTHER_SHARD
    }
