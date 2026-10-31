# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for complete-group YAML partition planning and rendering."""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import split_benchmark_groups as splitter


pytestmark = pytest.mark.unit


SOURCE = """\
# original note
TestParameters:
  marks: [skip-gfx900]
GlobalParameters:
  Architecture: gfx1250
BenchmarkProblems:
  - - &base {OperationType: GEMM, DataType: s, TransposeA: false}
    - BenchmarkCommonParameters: &common
        - KernelLanguage: [Assembly]
      ForkParameters:
        - WorkGroup: [[1, 1, 1]]
      BenchmarkFinalParameters:
        - ProblemSizes: [{Exact: [4, 4, 1, 4]}]
    # second group belongs with its settings
    - BenchmarkCommonParameters: *common
      BenchmarkFinalParameters:
        - ProblemSizes: [{Exact: [8, 8, 1, 8]}]
  - - {<<: *base, TransposeA: true}
    - BenchmarkCommonParameters: *common
      BenchmarkFinalParameters:
        - ProblemSizes: [{Exact: [16, 16, 1, 16]}]
"""


def test_iterative_split_respects_runtime_and_startup_budget():
    weights = [1, 1, 1, 1]
    assert splitter.propose_sizes(weights, seconds=240) == [1, 1, 1, 1]
    assert splitter.propose_sizes(weights, seconds=24.85, target_seconds=10) == [4]
    assert splitter.propose_sizes(weights, seconds=None) == [4]
    assert splitter.propose_sizes(weights, seconds=None, untimed_max_parts=3) == [1, 1, 2]


def test_worked_example_balances_weights_without_dividing_groups():
    weights = [4, 1, 1, 2]
    assert splitter.propose_sizes(weights, seconds=240, max_parts=2) == [1, 3]
    assert splitter.propose_sizes(weights, seconds=240) == [1, 2, 1]
    assert splitter.propose_sizes([1, 1], seconds=65) == [2]
    assert splitter.propose_sizes([1, 1, 1, 1], seconds=None) == [4]


def test_partition_preserves_whole_groups_comments_and_settings():
    parts = splitter.partition_text(SOURCE, [1, 1, 1])
    source = splitter.load_config(SOURCE)
    parsed = [splitter.load_config(part) for part in parts]
    assert all("# original note" in part for part in parts)
    assert "# second group belongs with its settings" in parts[1]
    assert [len(splitter.groups(part)) for part in parsed] == [1, 1, 1]
    assert Counter(splitter.canonical(group) for group in splitter.groups(source)) == Counter(
        splitter.canonical(group) for part in parsed for group in splitter.groups(part)
    )
    assert all(
        splitter.settings_digest(part) == splitter.settings_digest(source) for part in parsed
    )
    assert all("*common" not in part for part in parts[1:])


def test_optional_anchor_expansion_removes_alias_sharing():
    parts = splitter.partition_text(SOURCE, [2, 1], expand_all_anchors=True)
    assert all(
        not any(
            isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken))
            for token in yaml.scan(part)
        )
        for part in parts
    )
    assert sum(len(splitter.groups(splitter.load_config(part))) for part in parts) == 3
    assert "# second group belongs with its settings" not in parts[1]


def test_sections_after_benchmark_problems_are_not_silently_lost():
    text = SOURCE + "LibraryLogic:\n  ScheduleName: test\n"
    with pytest.raises(splitter.SplitError, match="sections follow"):
        splitter.partition_text(text, [1, 2])


def test_path_naming_skips_collisions_and_supports_reviewed_slots(tmp_path):
    repo = tmp_path
    directory = repo / "cases"
    directory.mkdir()
    (directory / "a_gfx1250.yaml").write_text(SOURCE)
    (directory / "a_gfx1250_part2.yaml").write_text(SOURCE)
    source = "cases/a_gfx1250.yaml"
    outputs = splitter.planned_outputs(repo, [{"path": source, "sizes": [1, 1]}], None)
    assert outputs[source] == [source, "cases/a_gfx1250_part3.yaml"]
    outputs = splitter.planned_outputs(
        repo, [{"path": source, "sizes": [1, 1], "slots": [2, 1]}], None
    )
    assert outputs[source] == ["cases/a_gfx1250_part3.yaml", source]
    with pytest.raises(splitter.SplitError, match="duplicate output"):
        splitter.planned_outputs(
            repo,
            [{"path": source, "sizes": [1, 1], "names": ["a_gfx1250.yaml", "a_gfx1250.yaml"]}],
            None,
        )


def test_artifact_names_must_be_unique(tmp_path):
    first = tmp_path / "foo__bar" / "baz.yaml"
    second = tmp_path / "foo" / "bar__baz.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(SOURCE)
    second.write_text(SOURCE)
    with pytest.raises(splitter.SplitError, match="artifact names collide"):
        splitter.planned_outputs(tmp_path, [{"path": "foo__bar/baz.yaml", "sizes": [1, 1]}], None)


def test_scan_multiple_directories_emits_reviewable_plan(tmp_path, capsys):
    for directory in ("first", "second"):
        target = tmp_path / directory / "case_gfx1250.yaml"
        target.parent.mkdir()
        target.write_text(SOURCE)
    plan_path = tmp_path / "split-plan.json"
    splitter.scan(
        SimpleNamespace(
            repo=tmp_path,
            directories=["first", "second"],
            source_ref=None,
            timings=None,
            target_seconds=60,
            startup_seconds=3.5,
            max_overhead=0.05,
            max_parts=16,
            untimed_max_parts=2,
            plan_out=plan_path,
        )
    )
    plan = splitter.load_plan(plan_path)
    assert plan["directories"] == ["first", "second"]
    assert [row["path"] for row in plan["splits"]] == [
        "first/case_gfx1250.yaml",
        "second/case_gfx1250.yaml",
    ]
    assert all(sum(row["sizes"]) == 3 for row in plan["splits"])
    assert '"needs_timing": true' in capsys.readouterr().out


def test_schedule_and_expected_failure_require_review():
    doc = splitter.load_config(SOURCE)
    doc["TestParameters"]["streamk_schedules"] = [{"grid": 128, "reduction": "tree"}]
    assert "schedule assertions need per-output review" in splitter.blockers(doc, "cases/test.yaml")
    assert "path encodes a whole-file outcome" in splitter.blockers(doc, "cases/test_xfail.yaml")
    assert splitter.path_policy("cases/test_gfx1250.yaml") == splitter.path_policy(
        "cases/test_gfx1250_part2.yaml"
    )
    assert splitter.path_policy("cases/test_gfx1250.yaml") != splitter.path_policy(
        "cases/test_gfx1250-strict_part2.yaml"
    )


def test_render_uses_unsplit_git_ref_after_worktree_is_split(tmp_path):
    repo = tmp_path
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    directory = repo / "cases"
    directory.mkdir()
    source_path = directory / "test_gfx1250.yaml"
    source_path.write_text(SOURCE)
    subprocess.run(["git", "-C", str(repo), "add", "cases"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "unsplit",
        ],
        check=True,
    )
    source_ref = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    plan_path = repo / "split.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_ref": source_ref,
                "directories": ["cases"],
                "splits": [
                    {
                        "path": "cases/test_gfx1250.yaml",
                        "sizes": [1, 2],
                        "settings_sha256": splitter.settings_digest(splitter.load_config(SOURCE)),
                    }
                ],
            }
        )
    )
    command = [
        sys.executable,
        str(Path(splitter.__file__)),
        "--repo",
        str(repo),
        "render",
        str(plan_path),
    ]
    subprocess.run([*command, "--write", "--verify"], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [*command, "--verify", "--strict-bytes"], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout)["verified_outputs"] == 2
    assert len(splitter.groups(splitter.load_config(source_path.read_text()))) == 1
    assert (
        len(
            splitter.groups(
                splitter.load_config((directory / "test_gfx1250_part2.yaml").read_text())
            )
        )
        == 2
    )
    source_path.write_text(source_path.read_text() + "# local edit\n")
    refused = subprocess.run([*command, "--write"], capture_output=True, text=True)
    assert refused.returncode == 1
    assert "locally modified output" in refused.stderr
    assert source_path.read_text().endswith("# local edit\n")


def test_non_group_settings_change_requires_plan_review(tmp_path):
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "case.yaml").write_text(SOURCE.replace("skip-gfx900", "skip-gfx942"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": 1,
                "directories": ["cases"],
                "splits": [
                    {
                        "path": "cases/case.yaml",
                        "sizes": [1, 2],
                        "settings_sha256": splitter.settings_digest(splitter.load_config(SOURCE)),
                    }
                ],
            }
        )
    )
    with pytest.raises(splitter.SplitError, match="review and update plan"):
        splitter.render(
            SimpleNamespace(
                repo=tmp_path,
                plan=plan_path,
                source_ref=None,
                write=False,
                verify=False,
                strict_bytes=False,
            )
        )


@pytest.fixture
def render_case(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "case.yaml"
    source.write_text(SOURCE)
    subprocess.run(["git", "-C", str(tmp_path), "add", "case.yaml"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "unsplit",
        ],
        check=True,
    )
    source_ref = splitter.git(tmp_path, "rev-parse", "HEAD").decode().strip()
    plan = {
        "version": 1,
        "source_ref": source_ref,
        "directories": ["."],
        "splits": [
            {
                "path": "case.yaml",
                "sizes": [1, 2],
                "settings_sha256": splitter.settings_digest(splitter.load_config(SOURCE)),
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        repo=tmp_path,
        plan=plan_path,
        source_ref=None,
        write=True,
        verify=True,
        strict_bytes=True,
    )
    return args, plan


def test_write_requires_saved_unsplit_source(render_case):
    args, plan = render_case
    del plan["source_ref"]
    args.plan.write_text(json.dumps(plan))
    with pytest.raises(splitter.SplitError, match="--source-ref"):
        splitter.render(args)
    assert (args.repo / "case.yaml").read_text() == SOURCE
    assert not (args.repo / "case_part2.yaml").exists()


def test_scan_pins_named_source_ref(render_case):
    args, plan = render_case
    splitter.scan(
        SimpleNamespace(
            repo=args.repo,
            directories=["."],
            source_ref="HEAD",
            timings=None,
            target_seconds=60,
            startup_seconds=3.5,
            max_overhead=0.05,
            max_parts=16,
            untimed_max_parts=2,
            plan_out=args.plan,
        )
    )
    assert splitter.load_plan(args.plan)["source_ref"] == plan["source_ref"]


def test_custom_names_must_replace_the_unsplit_source(render_case):
    args, plan = render_case
    plan["splits"][0]["names"] = ["first.yaml", "second.yaml"]
    args.plan.write_text(json.dumps(plan))
    with pytest.raises(splitter.SplitError, match="retain the source filename"):
        splitter.render(args)
    assert (args.repo / "case.yaml").read_text() == SOURCE
    assert not (args.repo / "first.yaml").exists()


def test_write_preserves_local_output_deletion(render_case):
    args, _ = render_case
    (args.repo / "case.yaml").unlink()
    with pytest.raises(splitter.SplitError, match="locally modified output"):
        splitter.render(args)
    assert not (args.repo / "case.yaml").exists()
    assert not (args.repo / "case_part2.yaml").exists()


@pytest.mark.parametrize("change", ["reduce", "rename", "remove"])
@pytest.mark.parametrize("write", [False, True])
def test_changed_plan_rejects_obsolete_outputs_before_writing(render_case, change, write):
    args, plan = render_case
    plan["splits"][0]["sizes"] = [1, 1, 1]
    args.plan.write_text(json.dumps(plan))
    splitter.render(args)
    before = {path.name: path.read_bytes() for path in args.repo.glob("*.yaml")}

    if change == "reduce":
        plan["splits"][0]["sizes"] = [1, 2]
    elif change == "rename":
        plan["splits"][0]["names"] = ["case.yaml", "second.yaml", "third.yaml"]
    else:
        plan["splits"] = []
    args.plan.write_text(json.dumps(plan))
    args.write = write
    with pytest.raises(splitter.SplitError, match="unexpected YAML outputs"):
        splitter.render(args)
    assert {path.name: path.read_bytes() for path in args.repo.glob("*.yaml")} == before


def test_plan_rejects_sources_outside_recorded_directories(render_case):
    args, plan = render_case
    plan["directories"] = ["other"]
    args.plan.write_text(json.dumps(plan))
    with pytest.raises(splitter.SplitError, match="in a recorded directory"):
        splitter.render(args)
    assert (args.repo / "case.yaml").read_text() == SOURCE
