# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which directory the planner of a run or audit directory uses
(lib/run_layout.py): stage02/, or stage01b/ in one written before it was
renamed."""

import json

import pytest

from lib import run_layout as rl


def _probe(rd):
    """The removed probe's outputs in a round: stage02/ without a plan."""
    (rd / "stage02" / "logs").mkdir(parents=True)
    (rd / "stage02" / "shapes_probed.yaml").write_text("- {M: 8}\n")
    (rd / "stage02.log").write_text("probe\n")


def test_a_new_directory_records_the_current_layout(tmp_path):
    run = tmp_path / "run"
    assert rl.of(run) == rl.CURRENT and not run.exists()
    assert rl.settle(run) == rl.CURRENT
    assert json.loads((run / rl.LAYOUT_FILE).read_text()) == {
        "layout": 2,
        "planner": "stage02",
    }
    before = (run / rl.LAYOUT_FILE).stat().st_mtime_ns
    assert rl.settle(run) == rl.CURRENT
    assert (run / rl.LAYOUT_FILE).stat().st_mtime_ns == before
    assert rl.plan_dir(run / "round_3") == run / "round_3" / "stage02"
    assert rl.CURRENT.dir_name("stage02") == "stage02"
    assert rl.CURRENT.dir_name("stage03b") == "stage03b"


@pytest.mark.parametrize(
    "where", ["round_0/stage01b", "round_4/stage01b", "stage01b", "probe"]
)
def test_a_directory_written_before_the_rename_keeps_stage01b(tmp_path, where):
    """A stage01b/ directory of the planner (in a round, or at the top of an
    audit set), or the removed probe's stage02/, mark a run or audit
    directory of the old layout, which settle leaves without the file."""
    run = tmp_path / "run"
    (run / "round_0" / "stage01").mkdir(parents=True)
    if where == "probe":
        _probe(run / "round_0")
    else:
        (run / where).mkdir(parents=True)
    assert rl.of(run) == rl.LEGACY and rl.LEGACY.legacy
    assert rl.settle(run) == rl.LEGACY
    assert not (run / rl.LAYOUT_FILE).exists()
    assert rl.plan_dir(run / "round_1") == run / "round_1" / "stage01b"
    assert rl.plan_dir(run) == run / "stage01b"
    assert rl.LEGACY.dir_name("stage02") == "stage01b"
    assert rl.LEGACY.dir_name("stage01") == "stage01"


def test_without_anything_of_the_old_layout_a_directory_takes_the_new_one(tmp_path):
    """Rounds with no planner directory, or with a stage02/ that holds a
    plan, are of the current layout; the file then records it."""
    run = tmp_path / "run"
    (run / "round_0" / "stage01").mkdir(parents=True)
    (run / "round_0" / "stage03").mkdir()
    (run / "stale" / "x" / "stage01b").mkdir(parents=True)
    assert rl.of(run) == rl.CURRENT
    (run / "round_1" / "stage02").mkdir(parents=True)
    (run / "round_1" / "stage02" / "plan.json").write_text("{}")
    assert rl.of(run) == rl.CURRENT
    assert rl.settle(run) == rl.CURRENT and rl.read(run) == rl.CURRENT


def test_the_recorded_layout_wins_and_an_unknown_one_is_refused(tmp_path):
    run = tmp_path / "run"
    rl.settle(run)
    _probe(run / "round_0")
    (run / "round_0" / "stage01b").mkdir()
    assert rl.of(run) == rl.CURRENT
    for text in ('{"layout": 3, "planner": "stage02"}', "[2]", "{not json"):
        (run / rl.LAYOUT_FILE).write_text(text)
        with pytest.raises(ValueError, match=rl.LAYOUT_FILE):
            rl.of(run)
        with pytest.raises(ValueError):
            rl.settle(run)
        with pytest.raises(ValueError):
            rl.plan_dir(run / "round_0")
    (run / rl.LAYOUT_FILE).write_text(json.dumps(rl.LEGACY.record()))
    assert rl.of(run) == rl.LEGACY


def test_the_plan_is_the_planners():
    from lib import planner

    assert rl.PLAN_FILE == planner.PLAN_FILE


def test_a_round_follows_its_run_and_an_audit_set_itself(tmp_path):
    run, audit = tmp_path / "run", tmp_path / "audit"
    (run / "round_2" / "stage01b").mkdir(parents=True)
    (audit / "stage01").mkdir(parents=True)
    assert rl.root_of(run / "round_2") == run and rl.root_of(audit) == audit
    assert rl.plan_dir(run / "round_7") == run / "round_7" / "stage01b"
    assert rl.plan_dir(audit) == audit / "stage02"
    assert rl.root_of(run / "round_x") == run / "round_x"
