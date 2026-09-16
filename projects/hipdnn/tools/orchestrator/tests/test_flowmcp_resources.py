# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The `run://` surface, the flows allow-list, and reading files under a cap.

These are the server's two containment boundaries. They are tested here rather
than asserted in `server.py`, which has no tests and no coverage.
"""
from __future__ import annotations

import pytest

from flowmcp import resources, schema


def seed(run_dir, relative: str, text: str = "x"):
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "runs" / "some-flow" / "20260915T171233Z-a1b4"
    directory.mkdir(parents=True)
    return directory


# -- URIs -------------------------------------------------------------------


def test_uri_round_trips_through_a_nested_step_path():
    nested = "outer/iter-07/inner/stdout.log"
    uri = schema.format_run_uri("20260915T171233Z-a1b4", nested)

    assert schema.parse_run_uri(uri) == ("20260915T171233Z-a1b4", nested)


def test_encoded_and_unencoded_separators_name_the_same_file(run_dir):
    seed(run_dir, "outer/iter-07/inner/stdout.log", "evidence")
    plain = "outer/iter-07/inner/stdout.log"
    encoded = "outer%2Fiter-07%2Finner%2Fstdout.log"

    _, from_plain = schema.parse_run_uri(f"run://a1b4/{plain}")
    _, from_encoded = schema.parse_run_uri(f"run://a1b4/{encoded}")

    assert from_plain == from_encoded
    assert resources.resolve_in_run(run_dir, from_encoded).read_text() == "evidence"
    # A client expanding the template with plain `{path}` percent-encodes every
    # separator. We accept that on the way in and never emit it.
    assert "%2F" not in schema.format_run_uri("a1b4", from_encoded)


# -- containment ------------------------------------------------------------


def test_a_nested_artifact_inside_the_run_is_served(run_dir):
    seed(run_dir, "reports/junit/results.xml", "<testsuite/>")

    served = resources.resolve_in_run(run_dir, "reports/junit/results.xml")

    assert served.read_text() == "<testsuite/>"


@pytest.mark.parametrize(
    "escape",
    [
        "../../../etc/passwd",
        "outer/../../elsewhere.txt",
        "/etc/passwd",
        "C:/Windows/System32/drivers/etc/hosts",
    ],
)
def test_paths_that_leave_the_run_directory_are_refused(run_dir, escape):
    with pytest.raises(resources.PathRefused):
        resources.resolve_in_run(run_dir, escape)


def test_a_sibling_run_is_out_of_reach(run_dir):
    sibling = run_dir.parent / "20260915T171233Z-ffff"
    sibling.mkdir()
    (sibling / "run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(resources.PathRefused):
        resources.resolve_in_run(run_dir, "../20260915T171233Z-ffff/run.json")


# -- the flows allow-list ---------------------------------------------------


@pytest.fixture
def flows_dir(tmp_path):
    directory = tmp_path / "flows"
    (directory / "nested").mkdir(parents=True)
    (directory / "first.yaml").write_text("version: 1\n", encoding="utf-8")
    (directory / "nested" / "second.yml").write_text("version: 1\n", encoding="utf-8")
    return directory


def test_a_bare_name_and_a_contained_path_both_resolve(flows_dir):
    assert resources.resolve_flow("first", flows_dir) == flows_dir / "first.yaml"
    assert resources.resolve_flow("first.yaml", flows_dir) == flows_dir / "first.yaml"
    assert (
        resources.resolve_flow("nested/second", flows_dir)
        == flows_dir / "nested" / "second.yml"
    )


def test_an_absolute_path_is_refused_however_real_it_is(flows_dir, tmp_path):
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("version: 1\n", encoding="utf-8")

    with pytest.raises(resources.FlowRefused):
        resources.resolve_flow(str(outside), flows_dir)


def test_a_traversal_out_of_the_flows_directory_is_refused(flows_dir, tmp_path):
    (tmp_path / "elsewhere.yaml").write_text("version: 1\n", encoding="utf-8")

    with pytest.raises(resources.FlowRefused):
        resources.resolve_flow("../elsewhere.yaml", flows_dir)


def test_a_flow_that_is_not_there_is_not_found(flows_dir):
    with pytest.raises(resources.FlowNotFound):
        resources.resolve_flow("absent", flows_dir)


# -- reading ----------------------------------------------------------------


def test_a_file_over_the_cap_returns_the_cap_plus_a_marker(run_dir):
    path = seed(run_dir, "big.log", "a" * 5000)

    served = resources.read_text(path, cap=64)

    body, marker = served.rsplit("\n", 1)
    assert body == "a" * 64
    assert marker.startswith("[truncated: 64 of 5000 bytes")
    assert str(path.resolve()) in marker


def test_a_listing_reports_the_true_size_of_a_truncated_file(run_dir):
    seed(run_dir, "big.log", "a" * 5000)

    entries = {
        entry["name"]: entry
        for entry in resources.resource_entries("a1b4", run_dir, "some-flow")
    }

    assert entries["big.log"]["size"] == 5000


def test_an_absent_manifest_reads_as_absent_rather_than_raising(run_dir):
    assert resources.read_json(run_dir / schema.MANIFEST_NAME) is None


# -- discovery --------------------------------------------------------------


def test_the_walk_lists_what_exists_and_types_it_by_extension(run_dir):
    seed(run_dir, "reports/junit/results.xml")
    seed(run_dir, "descriptors/shape.json")
    seed(run_dir, "registration/CMakeLists.txt")
    seed(run_dir, "bundle/payload.kpack")
    seed(run_dir, schema.MANIFEST_NAME, "{}")
    # The engine's scratch file for the atomic replace is a fragment of a write
    # that has not landed, not an artifact.
    seed(run_dir, schema.MANIFEST_NAME + ".tmp", "{}")

    listed = resources.walk_run_dir(run_dir)

    assert listed == [
        "bundle/payload.kpack",
        "descriptors/shape.json",
        "registration/CMakeLists.txt",
        "reports/junit/results.xml",
        schema.MANIFEST_NAME,
    ]
    assert resources.mime_for("reports/junit/results.xml") == "application/xml"
    assert resources.mime_for("descriptors/shape.json") == "application/json"
    assert resources.mime_for("registration/CMakeLists.txt") == "text/x-cmake"
    # Nothing in a run directory is binary, and an unmapped extension is the
    # normal case for a flow that writes something we have never seen.
    assert resources.mime_for("bundle/payload.kpack") == "text/plain"


def test_a_run_is_found_by_id_without_being_told_its_flow(tmp_path, run_dir):
    found = resources.find_run_dir(tmp_path / "runs", run_dir.name)

    assert found == run_dir.resolve()


def test_the_supervisors_own_directory_is_never_mistaken_for_a_run(tmp_path):
    root = tmp_path / "runs"
    records = root / ".supervisor"
    records.mkdir(parents=True)
    (records / "20260915T171233Z-a1b4.json").write_text("{}", encoding="utf-8")

    assert resources.find_run_dir(root, "20260915T171233Z-a1b4.json") is None
    assert resources.newest_run_dirs(root) == []
