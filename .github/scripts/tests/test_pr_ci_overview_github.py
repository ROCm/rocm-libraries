# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import pr_ci_overview as renderer
import pr_ci_overview_github as sut


def test_required_contexts_use_effective_base_branch_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_gh_json(*args: str, **_kwargs: object) -> object:
        calls.append(args)
        return [
            {"type": "pull_request", "parameters": {}},
            {
                "type": "required_status_checks",
                "parameters": {
                    "required_status_checks": [
                        {"context": "Math CI Summary"},
                        {"context": "pre-commit", "integration_id": 15368},
                        {"context": "Math CI Summary"},
                    ]
                },
            },
        ]

    monkeypatch.setattr(sut, "gh_json", fake_gh_json)

    assert sut._required_contexts("ROCm/rocm-libraries", "release/test branch") == [
        "Math CI Summary",
        "pre-commit",
    ]
    assert calls == [
        (
            "api",
            "--paginate",
            "--slurp",
            "repos/ROCm/rocm-libraries/rules/branches/"
            "release%2Ftest%20branch?per_page=100",
        )
    ]


def test_required_workflow_rule_fails_instead_of_reporting_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sut,
        "gh_json",
        lambda *_args, **_kwargs: [[{"type": "workflows", "parameters": {}}]],
    )

    with pytest.raises(sut.GitHubError, match="required-workflow rules"):
        sut._required_contexts("ROCm/rocm-libraries", "develop")


def test_normalize_required_adds_missing_gate_and_keeps_workflow_identity() -> None:
    reported = [
        {
            "name": "pre-commit",
            "state": "SUCCESS",
            "workflow": "pre-commit",
            "completedAt": "2026-09-09T10:00:00Z",
            "link": "https://github.test/actions/runs/10/job/101",
        },
        {
            "name": "Math CI Summary",
            "state": "FAILURE",
            "workflow": "",
            "startedAt": "2026-09-09T11:00:00Z",
            "link": "https://math-ci.test/build/12",
        },
    ]

    results = sut._normalize_required(
        ["Math CI Summary", "pre-commit", "Multi-Arch CI Summary"], reported
    )

    assert results == [
        {
            "key": "required:pre-commit:pre-commit",
            "name": "pre-commit",
            "state": "success",
            "updated_at": "2026-09-09T10:00:00Z",
            "id": 101,
            "url": "https://github.test/actions/runs/10/job/101",
        },
        {
            "key": "required:external-status:Math CI Summary",
            "name": "Math CI Summary",
            "state": "failure",
            "updated_at": "2026-09-09T11:00:00Z",
            "id": 2,
            "url": "https://math-ci.test/build/12",
        },
        {
            "key": "required:missing:Multi-Arch CI Summary",
            "name": "Multi-Arch CI Summary",
            "state": "pending",
            "updated_at": "1970-01-01T00:00:00Z",
            "id": 0,
        },
    ]


def test_duplicate_required_context_names_show_their_workflow_identity() -> None:
    reported = [
        {
            "name": "Multi-Arch CI Summary",
            "state": "SUCCESS",
            "workflow": "TheRock Multi-Arch CI ASAN",
        },
        {
            "name": "Multi-Arch CI Summary",
            "state": "FAILURE",
            "workflow": "TheRock Multi-Arch CI",
        },
    ]

    results = sut._normalize_required(["Multi-Arch CI Summary"], reported)

    assert [result["name"] for result in results] == [
        "Multi-Arch CI Summary — TheRock Multi-Arch CI ASAN",
        "Multi-Arch CI Summary — TheRock Multi-Arch CI",
    ]


def test_new_required_check_attempt_wins_when_old_attempt_completes_later() -> None:
    reported = [
        {
            "name": "pre-commit",
            "state": "CANCELLED",
            "workflow": "pre-commit",
            "startedAt": "2026-09-09T10:00:00Z",
            "completedAt": "2026-09-09T10:05:00Z",
            "link": "https://github.test/actions/runs/10/job/100",
        },
        {
            "name": "pre-commit",
            "state": "PENDING",
            "workflow": "pre-commit",
            "startedAt": "2026-09-09T10:04:00Z",
            "link": "https://github.test/actions/runs/11/job/101",
        },
    ]

    normalized = sut._normalize_required(["pre-commit"], reported)
    overview = renderer.parse_overview(
        {"required_checks": normalized, "workflow_runs": []}
    )

    assert len(overview.required_checks) == 1
    assert overview.required_checks[0].state is renderer.State.PENDING
    assert overview.required_checks[0].item_id == 101


def test_workflow_runs_are_top_level_only_and_exclude_gating_runs_and_admin() -> None:
    runs = [
        {
            "id": 10,
            "workflow_id": 1,
            "name": "TheRock Multi-Arch CI",
            "status": "in_progress",
            "created_at": "2026-09-09T10:00:00Z",
            "html_url": "https://github.test/actions/runs/10",
        },
        {
            "id": 11,
            "workflow_id": 2,
            "name": "Component CI",
            "status": "completed",
            "conclusion": "failure",
            "updated_at": "2026-09-09T11:00:00Z",
            "html_url": "https://github.test/actions/runs/11",
        },
        {
            "id": 12,
            "workflow_id": 3,
            "name": "Auto Label PR",
            "status": "completed",
            "conclusion": "success",
            "updated_at": "2026-09-09T11:00:00Z",
            "html_url": "https://github.test/actions/runs/12",
        },
    ]

    assert sut._normalize_workflow_runs(runs, {10}, {1}) == [
        {
            "key": "workflow:2",
            "name": "Component CI",
            "state": "failure",
            "updated_at": "2026-09-09T11:00:00Z",
            "id": 11,
            "url": "https://github.test/actions/runs/11",
        }
    ]


def test_new_run_wins_when_older_run_is_cancelled_later() -> None:
    runs = [
        {
            "id": 10,
            "workflow_id": 2,
            "name": "Component CI",
            "status": "completed",
            "conclusion": "cancelled",
            "created_at": "2026-09-09T10:00:00Z",
            "updated_at": "2026-09-09T10:05:00Z",
            "html_url": "https://github.test/actions/runs/10",
        },
        {
            "id": 11,
            "workflow_id": 2,
            "name": "Component CI",
            "status": "in_progress",
            "created_at": "2026-09-09T10:04:00Z",
            "updated_at": "2026-09-09T10:04:00Z",
            "html_url": "https://github.test/actions/runs/11",
        },
    ]

    normalized = sut._normalize_workflow_runs(runs, set(), set())
    overview = renderer.parse_overview(
        {"required_checks": [], "workflow_runs": normalized}
    )

    assert len(overview.workflow_runs) == 1
    assert overview.workflow_runs[0].state is renderer.State.PENDING
    assert overview.workflow_runs[0].item_id == 11


def test_same_named_workflow_is_not_hidden_unless_its_run_is_gating() -> None:
    runs = [
        {
            "id": 10,
            "workflow_id": 1,
            "name": "Duplicate display name",
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-09-09T10:00:00Z",
        },
        {
            "id": 11,
            "workflow_id": 2,
            "name": "Duplicate display name",
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2026-09-09T10:01:00Z",
        },
    ]

    normalized = sut._normalize_workflow_runs(runs, {10}, {1})

    assert [item["id"] for item in normalized] == [11]


def test_newer_attempt_of_gating_workflow_stays_out_of_informational_section() -> None:
    runs = [
        {
            "id": 100,
            "workflow_id": 7,
            "name": "Required workflow",
            "status": "completed",
            "conclusion": "failure",
            "created_at": "2026-09-09T10:00:00Z",
        },
        {
            "id": 101,
            "workflow_id": 7,
            "name": "Required workflow",
            "status": "in_progress",
            "created_at": "2026-09-09T10:05:00Z",
        },
    ]

    assert sut._normalize_workflow_runs(runs, {100}, {7}) == []


def test_collect_overview_produces_condensed_gate_based_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sut,
        "gh_json",
        lambda *args, **_kwargs: {
            "number": 42,
            "headRefOid": "abc123",
            "headRefName": "feature",
            "headRepository": {"nameWithOwner": "owner/fork"},
            "baseRefName": "develop",
        },
    )
    monkeypatch.setattr(
        sut,
        "_required_contexts",
        lambda _repo, _base: ["Math CI Summary", "pre-commit"],
    )
    monkeypatch.setattr(
        sut,
        "_reported_required_checks",
        lambda _repo, _pr: [
            {
                "name": "Math CI Summary",
                "state": "SUCCESS",
                "workflow": "",
                "completedAt": "2026-09-09T12:00:00Z",
                "link": "https://math-ci.test/42",
            },
            {
                "name": "pre-commit",
                "state": "SUCCESS",
                "workflow": "pre-commit",
                "completedAt": "2026-09-09T12:01:00Z",
                "link": "https://github.test/actions/runs/1/job/2",
            },
        ],
    )
    monkeypatch.setattr(
        sut,
        "_workflow_runs",
        lambda _repo, _sha, _pr, **_kwargs: [
            {
                "id": 1,
                "workflow_id": 10,
                "name": "pre-commit",
                "status": "completed",
                "conclusion": "success",
                "updated_at": "2026-09-09T12:01:00Z",
                "html_url": "https://github.test/actions/runs/1",
            },
            {
                "id": 101,
                "workflow_id": 11,
                "name": "Component CI",
                "status": "completed",
                "conclusion": "failure",
                "updated_at": "2026-09-09T12:02:00Z",
                "html_url": "https://github.test/actions/runs/101",
            },
        ],
    )

    normalized = sut.collect_overview("ROCm/rocm-libraries", 42)
    report = renderer.render_overview(renderer.parse_overview(normalized))

    assert report.startswith("## CI — all required checks passed\n")
    assert "Math CI Summary" in report
    assert "Component CI" in report
    assert "shard" not in report.casefold()
    assert report.count("pre-commit") == 1


def test_workflow_runs_are_filtered_to_the_requested_pr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sut,
        "gh_json",
        lambda *_args, **_kwargs: [
            {
                "workflow_runs": [
                    {"id": 1, "pull_requests": [{"number": 42}]},
                    {"id": 2, "pull_requests": [{"number": 99}]},
                    {
                        "id": 3,
                        "pull_requests": [],
                        "head_repository": {"full_name": "owner/fork"},
                        "head_branch": "feature",
                    },
                    {
                        "id": 4,
                        "pull_requests": [],
                        "head_repository": {"full_name": "other/fork"},
                        "head_branch": "feature",
                    },
                ]
            }
        ],
    )

    assert sut._workflow_runs(
        "ROCm/rocm-libraries",
        "abc",
        42,
        head_repository="owner/fork",
        head_ref="feature",
    ) == [
        {"id": 1, "pull_requests": [{"number": 42}]},
        {
            "id": 3,
            "pull_requests": [],
            "head_repository": {"full_name": "owner/fork"},
            "head_branch": "feature",
        },
    ]


@pytest.mark.parametrize(
    ("status", "conclusion", "expected"),
    [
        ("queued", None, "pending"),
        ("in_progress", None, "pending"),
        ("completed", "success", "success"),
        ("completed", "skipped", "skipped"),
        ("completed", "neutral", "neutral"),
        ("completed", "timed_out", "failure"),
    ],
)
def test_workflow_state_normalization(
    status: str, conclusion: str | None, expected: str
) -> None:
    assert sut._workflow_state({"status": status, "conclusion": conclusion}) == expected


def test_resolve_pr_prefers_open_pull_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sut,
        "gh_json",
        lambda *_args, **_kwargs: [
            {"number": 1, "state": "closed"},
            {"number": 2, "state": "open"},
        ],
    )

    assert sut.resolve_pr("ROCm/rocm-libraries", "abc") == 2


def test_resolve_pr_ignores_closed_pull_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sut,
        "gh_json",
        lambda *_args, **_kwargs: [{"number": 1, "state": "closed"}],
    )

    assert sut.resolve_pr("ROCm/rocm-libraries", "abc") is None


def test_resolve_pr_requires_number_when_sha_has_multiple_open_prs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sut,
        "gh_json",
        lambda *_args, **_kwargs: [
            {"number": 1, "state": "open"},
            {"number": 2, "state": "open"},
        ],
    )

    with pytest.raises(sut.GitHubError, match="multiple open pull requests"):
        sut.resolve_pr("ROCm/rocm-libraries", "abc")


def test_resolve_targets_validates_open_state_and_current_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sut, "resolve_prs", lambda _repo, _sha: [1, 2])
    metadata = {
        1: {"number": 1, "state": "OPEN", "headRefOid": "aaa"},
        2: {"number": 2, "state": "CLOSED", "headRefOid": "bbb"},
    }
    monkeypatch.setattr(sut, "_pr_metadata", lambda _repo, number: metadata[number])

    assert sut.resolve_targets("ROCm/rocm-libraries", sha="old") == {
        "include": [{"pr": 1, "head_sha": "aaa"}]
    }


def test_resolve_targets_cli_prints_compact_actions_matrix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sut,
        "resolve_targets",
        lambda _repo, **_kwargs: {"include": [{"pr": 42, "head_sha": "abc"}]},
    )

    assert (
        sut.main(
            [
                "--repo",
                "ROCm/rocm-libraries",
                "--sha",
                "old",
                "--resolve-targets",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == '{"include":[{"pr":42,"head_sha":"abc"}]}\n'


def test_stale_head_skips_collection_and_publication(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sut, "_head_matches", lambda *_args: False)

    def unexpected_collect(*_args: object) -> object:
        raise AssertionError("stale update must not collect or publish")

    monkeypatch.setattr(sut, "collect_overview", unexpected_collect)

    assert (
        sut.main(
            [
                "--repo",
                "ROCm/rocm-libraries",
                "--pr",
                "42",
                "--expected-head-sha",
                "old",
                "--update-comment",
            ]
        )
        == 0
    )
    assert "skipping stale update" in capsys.readouterr().out


def test_upsert_comment_updates_existing_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_gh_json(*args: str, **_kwargs: object) -> object:
        calls.append(args)
        if "--paginate" in args:
            return [
                [
                    {
                        "id": 77,
                        "body": f"old\n{sut.COMMENT_MARKER}",
                        "user": {"login": sut.COMMENT_AUTHOR},
                    }
                ]
            ]
        return {"id": 77}

    monkeypatch.setattr(sut, "gh_json", fake_gh_json)

    sut.upsert_comment("ROCm/rocm-libraries", 42, "new body")

    assert calls[-1] == (
        "api",
        "--method",
        "PATCH",
        "repos/ROCm/rocm-libraries/issues/comments/77",
        "-f",
        "body=new body",
    )


def test_upsert_comment_does_not_edit_another_users_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_gh_json(*args: str, **_kwargs: object) -> object:
        calls.append(args)
        if "--paginate" in args:
            return [
                [
                    {
                        "id": 66,
                        "body": sut.COMMENT_MARKER,
                        "user": {"login": "someone-else"},
                    }
                ]
            ]
        return {"id": 88}

    monkeypatch.setattr(sut, "gh_json", fake_gh_json)

    sut.upsert_comment("ROCm/rocm-libraries", 42, "new body")

    assert calls[-1][:4] == (
        "api",
        "--method",
        "POST",
        "repos/ROCm/rocm-libraries/issues/42/comments",
    )
