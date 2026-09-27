# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import io
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import pr_ci_overview as sut


def _result(
    key: str,
    name: str,
    state: str,
    *,
    item_id: int,
    updated_at: str = "2026-09-09T12:00:00Z",
    url: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "key": key,
        "name": name,
        "state": state,
        "updated_at": updated_at,
        "id": item_id,
    }
    if url is not None:
        result["url"] = url
    return result


def _render(data: dict[str, object]) -> str:
    return sut.render_overview(sut.parse_overview(data))


def test_render_separates_gates_and_condenses_informational_workflows() -> None:
    data = {
        "required_checks": [
            _result("check:pre-commit", "pre-commit", "success", item_id=1),
            _result(
                "status:math",
                "Math CI Summary",
                "failure",
                item_id=2,
                url="https://ci.example.test/math/2",
            ),
            _result("check:multi-arch", "Multi-Arch CI", "pending", item_id=3),
        ],
        "workflow_runs": [
            _result("component", "Component CI", "failure", item_id=10),
            _result("asan", "hipBLASLt ASAN", "pending", item_id=11),
            _result("docs", "Documentation", "cancelled", item_id=12),
            _result("labels", "Labels", "success", item_id=13),
            _result("lint", "Extra lint", "success", item_id=14),
            _result("nightly", "Nightly tests", "skipped", item_id=15),
        ],
    }

    assert (
        _render(data)
        == """\
## CI — merge blocked: 1 required check failed

### Merge gates

| Result | Required check |
| --- | --- |
| ❌ Failed | [Math CI Summary](https://ci.example.test/math/2) |
| ⏳ Pending | Multi-Arch CI |
| ✅ Passed | pre-commit |

### Non-gating workflows

1 failed · 1 pending · 1 cancelled · 2 successful · 1 not selected

| Result | Workflow |
| --- | --- |
| ❌ Failed | Component CI |
| ⏳ Pending | hipBLASLt ASAN |
| 🚫 Cancelled | Documentation |
"""
    )


def test_optional_failure_does_not_change_all_gates_passed_headline() -> None:
    data = {
        "required_checks": [
            _result("required", "Required CI", "success", item_id=1),
        ],
        "workflow_runs": [
            _result("optional", "Experimental CI", "failure", item_id=2),
        ],
    }

    report = _render(data)

    assert report.startswith("## CI — all required checks passed\n")
    assert "1 failed" in report
    assert "| ❌ Failed | Experimental CI |" in report


def test_skipped_and_neutral_required_checks_are_satisfied_but_not_passed() -> None:
    data = {
        "required_checks": [
            _result("required", "Required CI", "success", item_id=1),
            _result("not-selected", "Platform CI", "skipped", item_id=2),
            _result("neutral", "Advisory gate", "neutral", item_id=3),
        ],
        "workflow_runs": [],
    }

    report = _render(data)

    assert report.startswith("## CI — all required checks satisfied\n")
    assert "merge blocked" not in report
    assert "| ⏭️ Not selected | Platform CI |" in report
    assert "| ➖ Neutral | Advisory gate |" in report


def test_classic_external_status_is_rendered_as_a_required_gate() -> None:
    data = {
        "required_checks": [
            _result(
                "status:Math CI Summary",
                "Math CI Summary",
                "success",
                item_id=998,
                url="https://math-ci.example.test/build/998",
            )
        ],
        "workflow_runs": [],
    }

    report = _render(data)

    assert "## CI — all required checks passed" in report
    assert (
        "| ✅ Passed | [Math CI Summary]"
        "(https://math-ci.example.test/build/998) |" in report
    )


def test_duplicate_logical_results_use_latest_timestamp_then_id() -> None:
    required_checks = [
        _result(
            "required",
            "Required CI (old run)",
            "failure",
            item_id=100,
            updated_at="2026-09-09T10:00:00Z",
        ),
        _result(
            "required",
            "Required CI",
            "success",
            item_id=101,
            updated_at="2026-09-09T11:00:00Z",
        ),
    ]
    workflow_runs = [
        _result(
            "component",
            "Component CI (first attempt)",
            "success",
            item_id=200,
            updated_at="2026-09-09T12:00:00Z",
        ),
        _result(
            "component",
            "Component CI",
            "failure",
            item_id=201,
            updated_at="2026-09-09T12:00:00Z",
        ),
    ]

    report = _render(
        {
            "required_checks": list(reversed(required_checks)),
            "workflow_runs": list(reversed(workflow_runs)),
        }
    )

    assert "## CI — all required checks passed" in report
    assert "Required CI (old run)" not in report
    assert "Component CI (first attempt)" not in report
    assert "| ❌ Failed | Component CI |" in report


def test_output_order_is_deterministic_and_table_pipes_are_escaped() -> None:
    required_checks = [
        _result("z", "Zulu | gate", "success", item_id=1),
        _result("a", "Alpha gate", "success", item_id=2),
    ]
    workflow_runs = [
        _result("z-workflow", "Zulu workflow", "pending", item_id=3),
        _result(
            "a-workflow",
            "Alpha | <details> & @team *workflow*",
            "failure",
            item_id=4,
        ),
    ]
    original = {
        "required_checks": required_checks,
        "workflow_runs": workflow_runs,
    }
    reversed_input = {
        "required_checks": list(reversed(required_checks)),
        "workflow_runs": list(reversed(workflow_runs)),
    }

    report = _render(original)

    assert report == _render(reversed_input)
    assert "Alpha \\| &lt;details&gt; &amp; &#64;team \\*workflow\\*" in report
    assert "Zulu \\| gate" in report
    assert report.index("Alpha gate") < report.index("Zulu \\| gate")


def test_required_checks_put_attention_before_success() -> None:
    data = {
        "required_checks": [
            _result("pass", "Alpha passing gate", "success", item_id=1),
            _result("pending", "Zulu pending gate", "pending", item_id=2),
            _result("failure", "Zulu failed gate", "failure", item_id=3),
        ],
        "workflow_runs": [],
    }

    report = _render(data)

    assert report.index("Zulu failed gate") < report.index("Zulu pending gate")
    assert report.index("Zulu pending gate") < report.index("Alpha passing gate")


def test_informational_detail_rows_are_bounded() -> None:
    data = {
        "required_checks": [],
        "workflow_runs": [
            _result(f"workflow-{index}", f"Workflow {index}", "failure", item_id=index)
            for index in range(25)
        ],
    }

    report = _render(data)

    assert report.count("| ❌ Failed |") == 20
    assert "_5 more workflows need attention" in report


def test_display_names_are_bounded_and_mentions_are_neutralized() -> None:
    name = "@reviewers " + ("x" * 300)
    report = _render(
        {
            "required_checks": [
                _result("gate", name, "success", item_id=1),
            ],
            "workflow_runs": [],
        }
    )

    assert "&#64;reviewers" in report
    assert "@reviewers" not in report
    assert "x" * 170 not in report
    assert "…" in report


def test_successful_and_not_selected_workflow_names_stay_condensed() -> None:
    data = {
        "required_checks": [],
        "workflow_runs": [
            _result("pass", "Passing workflow detail", "success", item_id=1),
            _result("skip", "Skipped workflow detail", "skipped", item_id=2),
        ],
    }

    report = _render(data)

    assert "1 successful · 1 not selected" in report
    assert "Passing workflow detail" not in report
    assert "Skipped workflow detail" not in report
    assert "No required checks were reported" in report


def test_job_or_shard_fields_are_rejected_by_the_workflow_only_schema() -> None:
    workflow = _result("component", "Component CI", "failure", item_id=1)
    workflow["jobs"] = [{"name": "test / gfx942 / shard 7", "state": "failure"}]

    with pytest.raises(
        sut.InputError,
        match=r"workflow_runs\[0\] has unknown fields: jobs",
    ):
        sut.parse_overview({"required_checks": [], "workflow_runs": [workflow]})


def test_cli_reads_json_from_stdin_and_writes_only_markdown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = {
        "required_checks": [
            _result("gate", "Gate", "pending", item_id=1),
        ],
        "workflow_runs": [],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(data)))

    assert sut.main([]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("## CI — waiting on 1 required check\n")


def test_conflicting_duplicate_recency_is_rejected() -> None:
    first = _result("same", "Same check", "success", item_id=1)
    second = _result("same", "Same check", "failure", item_id=1)

    with pytest.raises(
        sut.InputError,
        match="conflicting items for key 'same'",
    ):
        sut.parse_overview({"required_checks": [first, second], "workflow_runs": []})
