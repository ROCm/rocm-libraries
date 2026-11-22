# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""--build-library: one command produces a deployable merged library tree.

Persist stock + current bespoke logic, build stock-only and merged libraries with the
real TensileCreateLibrary, emit the built library's index map and stage times.
OUT/logic and OUT/library are exclusively geko-owned: any path not recorded by a
previous build (or a symlinked / non-directory root) is a usage refusal that changes
nothing; user content elsewhere under OUT survives.
"""
import json
import math
import os
from pathlib import Path

import pytest
import yaml

from tests.specialize_build_test_support import (
    assert_unchanged_refusal, build, files, gfx942_case, library_solutions, match_table, merged_case,
    seed_work, stock_case, subset_fixture, tree,
)
from tests.specialize_test_support import (
    DATA, RECIPES, assert_directories_preserved, held_directories, identity, route_document, write_rows,
)


def build_metadata(case):
    build = case["manifest"].get("build", {}).get("library", {})
    assert isinstance(build, dict), "--build-library must record the new build, not the baseline directory"
    assert build.get("command"), "--build-library must invoke TCL even for stock-only"
    return build


# Break: bespoke-only scratch staging, YAML reserialization, in-place stock changes,
# or dropping the untraced stock recipe.
@pytest.mark.parametrize("case_fixture", ["merged_case", "gfx942_case"])
def test_merged_logic_persists_byte_identical_stock_and_separate_bespoke(case_fixture, request):
    case = request.getfixturevalue(case_fixture)
    assert case["code"] == 0
    logic = Path(build_metadata(case)["command"][-3])
    assert logic.is_dir(), "TCL input must persist after specialization, not disappear with derivation scratch"
    assert logic == case["destination"] / "logic", "merged inputs belong to the output library tree"
    assert files(case["baseline"] / "source-logic") == case["before"]
    stock_directories = set()
    for relative, content in case["before"].items():
        matches = [path for path in logic.rglob(Path(relative).name)
                   if path.parts[-len(Path(relative).parts):] == Path(relative).parts]
        assert len(matches) == 1, f"stock recipe must be copied once: {relative}"
        assert matches[0].read_bytes() == content, f"stock bytes changed: {relative}"
        stock_directories.add(matches[0].relative_to(logic).parts[0])
    assert len(stock_directories) == 1
    bespoke_directories = set()
    for call in case["manifest"]["calls"]:
        matches = list(logic.rglob(call["logic_file"]))
        assert len(matches) == 1, "every generated bespoke file must enter the merge once"
        assert matches[0].read_bytes() == (case["work"] / "logic" / case["manifest"]["arch"] / call["logic_file"]).read_bytes()
        bespoke_directories.add(matches[0].relative_to(logic).parts[0])
    assert len(bespoke_directories) == 1
    assert stock_directories.isdisjoint(bespoke_directories), "stock and bespoke logic must have separate directories"
    assert not list(logic.rglob("stale-user.yaml")), "unrelated pre-existing workdir logic must not be built"


# Break: reject --stock-only with --build-library, skip TCL, or admit stale bespoke rows.
def test_stock_only_builds_the_stock_tree_without_bespoke(stock_case, merged_case):
    case = stock_case
    assert case["code"] == 0, "--stock-only --build-library must build, not reject the command"
    logic = Path(build_metadata(case)["command"][-3])
    assert logic == case["destination"] / "logic"
    assert logic.is_dir()
    assert {path.read_bytes() for path in logic.rglob("*.yaml")} == set(case["before"].values())
    assert len(list(logic.rglob("*.yaml"))) == len(case["before"])
    assert files(case["baseline"] / "source-logic") == case["before"]
    solutions = library_solutions(case["destination"] / "library/gfx950")
    assert {s["kernelName"] for s in solutions.values()} == {
        RECIPES["dp"]["kernel_name"], RECIPES["dp_other"]["kernel_name"]}
    assert all("bespokeNarrowing" not in solution for solution in solutions.values())
    merged = library_solutions(merged_case["destination"] / "library/gfx950")
    assert {s["kernelName"] for s in solutions.values()} <= {s["kernelName"] for s in merged.values()}
    assert case["manifest"]["stock_only"] is True
    assert all(call["status"] == "stock-only" for call in case["manifest"]["calls"])


# Break: build only traced stock or bespoke files, lose a lazy leaf, or dedupe by shape.
def test_built_library_contains_untraced_stock_and_each_bespoke(merged_case):
    case = merged_case
    assert case["code"] == 0
    solutions = library_solutions(case["destination"] / "library/gfx950")
    names = {solution["kernelName"] for solution in solutions.values()}
    assert RECIPES["dp"]["kernel_name"] in names, "merged library lost the traced general stock kernel"
    assert RECIPES["dp_other"]["kernel_name"] in names, "merged library lost an untraced stock kernel"
    assert len(solutions) == 4, "two stock recipes plus same-shape none/relu bespoke kernels must survive"
    for call in case["manifest"]["calls"]:
        matching = [solution for solution in solutions.values() if solution["kernelName"] == call["kernel_name"]]
        assert len(matching) == 1
        assert matching[0].get("bespokeNarrowing") is True


# Break: copy the baseline MatchTable, write the map inside the loadable library tree,
# or persist deleted scratch paths.
@pytest.mark.parametrize("case_fixture", ["merged_case", "gfx942_case", "stock_case"])
def test_built_matchtable_beside_library_resolves_current_call_identities(case_fixture, request):
    case = request.getfixturevalue(case_fixture)
    assert case["code"] == 0
    directory = case["destination"] / "library" / case["manifest"]["arch"]
    assert not (directory / "MatchTable.yaml").exists(), "the index map must not change the loadable library tree"
    table = match_table(directory.parent / "MatchTable.yaml")
    assert table, "the built library's own MatchTable.yaml must be emitted beside it"
    solutions = library_solutions(directory)
    assert set(table) == set(solutions), "built indices and serialized solutions must agree exactly"
    assert {tuple(sorted(c["identity"].items())) for c in case["manifest"]["calls"]} == {
        tuple(sorted(identity(record).items())) for record in case["records"]}
    for call in case["manifest"]["calls"]:
        source_name = call["recipe"]["source_file"] if case["manifest"]["stock_only"] else call["logic_file"]
        hits = [(index, Path(source), local) for index, (source, local) in table.items()
                if Path(source).as_posix().endswith("/" + source_name)]
        assert len(hits) == 1
        index, source, local = hits[0]
        assert source.is_file(), "built MatchTable must not name deleted staging files"
        assert source.is_relative_to(case["destination"] / "logic")
        emitted = yaml.safe_load(source.read_text())
        assert any(solution.get("SolutionIndex", position) == local
                   for position, solution in enumerate(emitted["Solutions"]))
        assert index != call["recipe"]["baseline_index"], "baseline indices are not indices of this new build"
        assert solutions[index]["kernelName"] == call["kernel_name"]
        assert solutions[index]["name"] == call["solution_name"]
        if case["manifest"]["stock_only"]:
            assert "bespokeNarrowing" not in solutions[index]
        else:
            assert solutions[index].get("bespokeNarrowing") is True
        assert solutions[index]["libraryLogicIndex"] == local


# Break: omit merge or TCL time, write a string/NaN, or a bogus zero total.
@pytest.mark.parametrize("case_fixture", ["merged_case", "gfx942_case", "stock_case"])
def test_manifest_records_each_stage_wall_time(case_fixture, request):
    case = request.getfixturevalue(case_fixture)
    assert case["code"] == 0
    timings = case["manifest"].get("timings_seconds", {})
    required = {"parse_specialize", "logic_merge", "tensile_create_library", "total"}
    assert required <= timings.keys(), "manifest must record parse/specialize, logic merge, TCL and total wall times"
    for stage in required:
        value = timings[stage]
        assert type(value) in (int, float) and math.isfinite(value) and value >= 0, stage
    assert timings["tensile_create_library"] > 0
    assert timings["total"] >= sum(timings[stage] for stage in required - {"total"})
    assert 0 < timings["total"] <= case["elapsed"] + 0.1


def canonical_map(path):
    # Fresh and reused OUT differ only by location: compare indices, source bytes, labels.
    return {index: (Path(source).read_bytes(), local) for index, (source, local) in match_table(path).items()}


# Reused OUT gets exactly the fresh build's owned artifacts and index map; user content
# outside OUT/logic and OUT/library survives. Break: keep obsolete leaves or delete it.
@pytest.mark.parametrize("next_run", ["stock_only", "one_identity"])
def test_rebuild_matches_fresh_current_library_without_obsolete_bespoke_artifacts(next_run, tmp_path, monkeypatch):
    _, logs, facts, records = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "reused"
    first = build(tmp_path, logs, facts, work, destination, monkeypatch)
    assert first["code"] == 0 and len(library_solutions(destination / "library/gfx950")) == 4
    note = destination / "user-notes.txt"
    note.write_bytes(b"not a generated library artifact\n")
    empty = destination / "user-empty"
    empty.mkdir()
    if next_run == "one_identity":
        write_rows(logs / "fixture-on/hipblaslt_101.log", records[:1])
        (facts / "fixture-on.json").write_text(json.dumps(route_document(records[:1])))
    stock_only = next_run == "stock_only"
    with held_directories([empty]) as handles:
        reused = build(tmp_path, logs, facts, work, destination, monkeypatch, stock_only=stock_only)
        assert reused["code"] == 0, "same-OUT current-run rebuild must succeed"
        assert_directories_preserved(handles)
        assert note.is_file() and note.read_bytes() == b"not a generated library artifact\n"
        fresh_out = tmp_path / "fresh"
        fresh = build(tmp_path, logs, facts, tmp_path / "fresh-work", fresh_out, monkeypatch, stock_only=stock_only)
        assert fresh["code"] == 0
        assert len(library_solutions(fresh_out / "library/gfx950")) == (2 if stock_only else 3)
        assert match_table(fresh_out / "library/MatchTable.yaml"), "fresh-build positive control has no index map"
        assert not (fresh_out / "library/gfx950/MatchTable.yaml").exists()
        assert not (destination / "library/gfx950/MatchTable.yaml").exists()
        actual = files(destination / "library/gfx950")
        expected = files(fresh_out / "library/gfx950")
        assert actual.keys() == expected.keys(), "same-OUT rebuild retained obsolete bespoke library artifacts"
        assert actual == expected, "reused and fresh library code/metadata bytes differ"
        assert canonical_map(destination / "library/MatchTable.yaml") == canonical_map(fresh_out / "library/MatchTable.yaml")
        assert len(library_solutions(destination / "library/gfx950")) == (2 if stock_only else 3)


# Break: admit pre-existing unrecorded OUT/logic content into TCL, or overwrite it.
@pytest.mark.parametrize("kind", ["file", "empty_directory"])
def test_preexisting_user_logic_content_survives_build_or_unchanged_refusal(kind, tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "built"
    seed_work(tmp_path, logs, facts, work, monkeypatch)
    logic = destination / "logic"
    logic.mkdir(parents=True)
    protected = logic / ("notes.txt" if kind == "file" else "user-empty")
    if kind == "file":
        protected.write_bytes(b"user-authored input notes\x00\xff\n")
        directories = [logic]
    else:
        protected.mkdir()
        directories = [logic, protected]
    before_work, before_destination = tree(work), tree(destination)
    with held_directories(directories) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_directories_preserved(handles)
        if kind == "file":
            assert protected.is_file() and protected.read_bytes() == b"user-authored input notes\x00\xff\n"
        assert_unchanged_refusal(result, work, destination, before_work, before_destination)


# Break: rebuild while preserving/admitting user additions made after a recorded build.
def test_user_additions_between_builds_survive_or_cause_unchanged_refusal(tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "built"
    first = build(tmp_path, logs, facts, work, destination, monkeypatch)
    assert first["code"] == 0
    assert len(library_solutions(destination / "library/gfx950")) == 4
    note = destination / "logic/stock/user-notes.txt"
    note.write_bytes(b"added after the first successful build\n")
    empty = destination / "logic/bespoke/user-empty"
    empty.mkdir()
    before_work, before_destination = tree(work), tree(destination)
    with held_directories([empty, empty.parent]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_directories_preserved(handles)
        assert note.is_file(), "rebuild deleted a user addition to generated inputs"
        assert note.read_bytes() == b"added after the first successful build\n"
        assert_unchanged_refusal(result, work, destination, before_work, before_destination)


# Break: overwrite a colliding stock path, admit unrelated YAML into TCL, or keep
# library user files through a successful build instead of refusing.
@pytest.mark.parametrize("unrecorded", ["stock_collision", "unrelated_gfx950_logic", "library_file"])
def test_unrecorded_exclusive_output_paths_are_refused_before_any_change(unrecorded, tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "built"
    # A real build is the positive control; the collision case then uses a fresh OUT.
    control_out = tmp_path / "control" if unrecorded == "stock_collision" else destination
    control = build(tmp_path, logs, facts, work, control_out, monkeypatch)
    assert control["code"] == 0
    assert len(library_solutions(control_out / "library/gfx950")) == 4
    assert len(control["manifest"]["calls"]) == 2
    if unrecorded == "stock_collision":
        protected = destination / "logic/stock" / RECIPES["dp"]["logic_file"]
        protected.parent.mkdir(parents=True)
        content = b"user-authored stock-path collision\x00\xff\n"
        assert not (destination / ".geko_specialize_build.json").exists()
    else:
        record = json.loads((destination / ".geko_specialize_build.json").read_text())
        assert record["files"], "positive control must establish actual owned output"
        if unrecorded == "unrelated_gfx950_logic":
            # Valid gfx950 logic unrelated to the bf16 workload must not become a general kernel.
            protected = destination / "logic/gfx950_user_f32.yaml"
            content = (DATA / "stock" / RECIPES["f32"]["logic_file"]).read_bytes()
            assert yaml.safe_load(content)["ArchitectureName"] == "gfx950"
        else:
            protected = destination / "library/user-notes.txt"
            content = b"user library metadata, not a geko artifact\n"
        assert str(protected.relative_to(destination)) not in record["files"]
        assert not protected.exists()
    protected.write_bytes(content)
    before_work, before_destination = tree(work), tree(destination)
    with held_directories([protected.parent]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_unchanged_refusal(result, work, destination, before_work, before_destination)
        assert_directories_preserved(handles)
        assert protected.read_bytes() == content


# OUT-root user content is preserved; library-directory content is refused unchanged.
@pytest.mark.parametrize("location", ["output_root", "library_directory"])
def test_unowned_output_content_is_preserved_or_refused_by_location(location, tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "built"
    seed_work(tmp_path, logs, facts, work, monkeypatch)
    parent = destination if location == "output_root" else destination / "library/gfx950"
    parent.mkdir(parents=True)
    note, empty = parent / "user-notes.txt", parent / "user-empty"
    note.write_bytes(b"unrelated user output\n")
    empty.mkdir()
    before_work, before_destination = tree(work), tree(destination)
    with held_directories([parent, empty]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch, stock_only=True)
        if location == "library_directory":
            assert_unchanged_refusal(result, work, destination, before_work, before_destination)
        else:
            assert result["code"] == 0
            assert len(library_solutions(destination / "library/gfx950")) == 2
        assert_directories_preserved(handles)
        assert note.is_file() and note.read_bytes() == b"unrelated user output\n"


# Break: invalidate the work/build output before rejecting invalid jobs.
def test_usage_refusal_leaves_prior_work_and_build_tree_unchanged(tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "built"
    assert build(tmp_path, logs, facts, work, destination, monkeypatch)["code"] == 0
    empty = destination / "logic/user-empty"
    empty.mkdir()
    before_work, before_destination = tree(work), tree(destination)
    with held_directories([empty]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch, jobs=0)
        assert result["code"] == 2
        assert tree(work) == before_work, "usage refusal invalidated prior specialization outputs"
        assert tree(destination) == before_destination, "usage refusal changed prior built output"
        assert_directories_preserved(handles)


# Same-tree aliases of the workdir are refused before invalidation.
@pytest.mark.parametrize("alias", ["identical", "symlink_equivalent"])
def test_workdir_build_alias_succeeds_or_refuses_without_touching_existing_contents(alias, tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work = tmp_path / "work"
    seed_work(tmp_path, logs, facts, work, monkeypatch)
    note = work / "logic/user-notes.txt"
    note.write_bytes(b"same output tree belongs to the user too\n")
    empty = work / "logic/user-empty"
    empty.mkdir()
    destination = work
    if alias == "symlink_equivalent":
        destination = tmp_path / "build-alias"
        destination.symlink_to(work, target_is_directory=True)
    before = tree(work)
    with held_directories([work / "logic", empty]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_unchanged_refusal(result, work, destination, before, before)
        assert_directories_preserved(handles)
        assert note.is_file() and note.read_bytes() == b"same output tree belongs to the user too\n"
        if alias == "symlink_equivalent":
            assert destination.is_symlink() and destination.resolve() == work.resolve(), "build changed a user path alias"


def user_directory(path):
    path.mkdir()
    (path / "notes.txt").write_bytes(b"external user notes\x00\xff\n")
    (path / "gfx950_user_f32.yaml").write_bytes((DATA / "stock" / RECIPES["f32"]["logic_file"]).read_bytes())
    return path


# Break: skip root-path validation (os.walk does not enumerate a symlinked root's
# descendants), follow the link during writes, or admit the target's YAML.
@pytest.mark.parametrize("target_contents", ["populated", "empty"])
@pytest.mark.parametrize("exclusive_root", ["logic", "library"])
def test_symlinked_exclusive_root_refuses_without_changing_external_user_directory(exclusive_root, target_contents, tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    control = build(tmp_path, logs, facts, tmp_path / "control-work", tmp_path / "control", monkeypatch)
    assert control["code"] == 0
    assert len(library_solutions(tmp_path / "control/library/gfx950")) == 4
    work, destination = tmp_path / "work", tmp_path / "built"
    seed_work(tmp_path, logs, facts, work, monkeypatch)
    destination.mkdir()
    target = tmp_path / "user-target"
    if target_contents == "populated":
        user_directory(target)
        if exclusive_root == "logic":
            # User bytes at the exact stock-copy path as well.
            collision = target / "stock" / RECIPES["dp"]["logic_file"]
            collision.parent.mkdir(parents=True)
            collision.write_bytes(b"external user stock-path collision\n")
    else:
        target.mkdir()
    link = destination / exclusive_root
    link.symlink_to(target, target_is_directory=True)
    before_work, before_destination, before_target = tree(work), tree(destination), tree(target)
    with held_directories([destination, target]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_unchanged_refusal(result, work, destination, before_work, before_destination)
        assert tree(target) == before_target, "refusal changed the external user directory"
        assert_directories_preserved(handles)
        assert link.is_symlink() and os.readlink(link) == str(target)


# Break: treat an unenumerated file root as empty and fail later, after invalidation.
@pytest.mark.parametrize("root", ["OUT", "logic", "library"])
def test_non_directory_output_root_refuses_before_changing_prior_outputs(root, tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work, destination = tmp_path / "work", tmp_path / "built"
    seed_work(tmp_path, logs, facts, work, monkeypatch)
    if root == "OUT":
        protected = destination
    else:
        destination.mkdir()
        protected = destination / root
    content = b"user file at an output-root path\x00\xff\n"
    protected.write_bytes(content)
    before_work, before_destination = tree(work), tree(destination)
    with protected.open("rb") as handle:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_unchanged_refusal(result, work, destination, before_work, before_destination)
        assert protected.is_file() and protected.read_bytes() == content
        assert os.path.samestat(protected.stat(), os.fstat(handle.fileno())), "user root file was replaced"


# An explicit OUT directory alias stays supported. Break: reject OUT aliases, delete
# root user files, or admit root-level YAML.
def test_explicit_output_directory_alias_builds_only_the_owned_library(tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    target = user_directory(tmp_path / "chosen-output")
    empty = target / "user-empty"
    empty.mkdir()
    destination = tmp_path / "output-alias"
    destination.symlink_to(target, target_is_directory=True)
    user_bytes = {name: (target / name).read_bytes() for name in ("notes.txt", "gfx950_user_f32.yaml")}
    with held_directories([target, empty]) as handles:
        result = build(tmp_path, logs, facts, tmp_path / "work", destination, monkeypatch, stock_only=True)
        assert result["code"] == 0, "an explicit OUT directory alias must remain supported"
        assert destination.is_symlink() and os.readlink(destination) == str(target)
        assert_directories_preserved(handles)
        for name, content in user_bytes.items():
            assert (target / name).is_file(), "build removed a user file outside the exclusive trees"
            assert (target / name).read_bytes() == content
        solutions = library_solutions(target / "library/gfx950")
        assert {solution["kernelName"] for solution in solutions.values()} == {
            RECIPES["dp"]["kernel_name"], RECIPES["dp_other"]["kernel_name"]}
        assert all("bespokeNarrowing" not in solution for solution in solutions.values())
        table = match_table(target / "library/MatchTable.yaml")
        assert set(table) == set(solutions)
        assert all(str(target / "logic/stock") in source for source, _ in table.values())
        assert not (target / "library/gfx950/MatchTable.yaml").exists()


# Break: skip the exclusive-tree check when OUT is spelled as an alias.
def test_explicit_output_alias_does_not_bypass_resolved_target_ownership(tmp_path, monkeypatch):
    _, logs, facts, _ = subset_fixture(tmp_path)
    work = tmp_path / "work"
    seed_work(tmp_path, logs, facts, work, monkeypatch)
    target = user_directory(tmp_path / "chosen-output")
    logic = target / "logic"
    logic.mkdir()
    (logic / "gfx950_user_f32.yaml").write_bytes((target / "gfx950_user_f32.yaml").read_bytes())
    destination = tmp_path / "output-alias"
    destination.symlink_to(target, target_is_directory=True)
    before_work, before_target = tree(work), tree(target)
    with held_directories([target, logic]) as handles:
        result = build(tmp_path, logs, facts, work, destination, monkeypatch)
        assert_unchanged_refusal(result, work, destination, before_work, before_target)
        assert tree(target) == before_target
        assert_directories_preserved(handles)
        assert destination.is_symlink() and os.readlink(destination) == str(target)
