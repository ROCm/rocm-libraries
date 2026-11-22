# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""geko --specialize baseline recipes: the MatchTable SolutionIndex is authoritative, the
stock logic root must hold the MatchTable sources, and geko's own build output is a baseline.

Pass/break pairs are recorded beside each case. Production paths are real.
"""

import json
import shutil

import pytest
import yaml

from tests.specialize_test_support import (
    RECIPES, build_fixture, input_fixture, invoke, outputs, row, tree_hash, write_rows,
)
from tests.test_specialize_recipe_unit import command


# Pass: geko's own --build-library output is a usable baseline: the map's value is the
# SolutionIndex, the recipe records the real in-file position. Break: index by position.
def test_geko_build_output_is_a_working_baseline(tmp_path, monkeypatch):
    from tests.specialize_build_test_support import match_table, subset_fixture

    first = tmp_path / "first"
    _, logs, facts, records = subset_fixture(first)
    built = tmp_path / "built"
    assert invoke(command(first, logs, facts, first / "out", extra=[
        "--keep-beta", "--build-library", str(built), "--jobs", "1"]), monkeypatch) == 0
    table = match_table(built / "library/MatchTable.yaml")
    source, value = next((s, v) for s, v in table.values() if s.endswith(RECIPES["dp"]["logic_file"]))
    index = next(i for i, (s, _) in table.items() if s == source)
    stock = yaml.safe_load(open(source))["Solutions"]
    assert value != 0 and stock[0]["SolutionIndex"] == value, "fixture must exercise SolutionIndex != position"
    trace = logs / "fixture-on/hipblaslt_101.log"
    write_rows(trace, [{**record, "solution_index": index} for record in records])
    run = logs / "fixture-on/run.json"
    metadata = json.loads(run.read_text())
    metadata["library_fp"] = tree_hash(built / "library/gfx950")
    run.write_text(json.dumps(metadata))
    out = tmp_path / "reuse"
    code = invoke(["--specialize", "--workload-log", str(logs), "--route-facts", str(facts),
                   "--hipblaslt", str(built), "--arch", "gfx950", "--workdir", str(out), "--keep-beta"], monkeypatch)
    result = outputs(out, code)
    assert code == 0
    assert len(result["manifest"]["calls"]) == 2
    recipe = result["manifest"]["calls"][0]["recipe"]
    assert recipe["in_file_index"] == 0
    assert recipe["baseline_index"] == index
    assert result["manifest"]["build"]["source_logic"] == str(built / "logic")


# Pass: a TensileCreateLibrary MatchTable (value = SolutionIndex) resolves to the solution
# carrying that SolutionIndex and records its real position. Break: use the value as position.
def test_matchtable_solution_index_value_resolves_to_real_position(tmp_path, monkeypatch):
    records = [row(call_count=3)]
    build = build_fixture(tmp_path / "baseline")
    table_path = build / "device-library/MatchTable.yaml"
    table = yaml.safe_load(table_path.read_text())
    entry = table[RECIPES["dp"]["baseline_index"]]
    solution_index = yaml.safe_load(open(entry[0]))["Solutions"][0]["SolutionIndex"]
    assert solution_index != 0
    table[RECIPES["dp"]["baseline_index"]] = [entry[0], solution_index]
    table_path.write_text(yaml.safe_dump(table))
    logs, facts = input_fixture(tmp_path, records)
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    assert result["code"] == 0
    assert result["manifest"]["calls"][0]["recipe"]["in_file_index"] == 0


# Pass: an explicit or params.txt logic root must hold every MatchTable source (symlinked
# path allowed). Break: accept any directory and copy it as the merged stock logic.
@pytest.mark.parametrize("form", ["explicit", "params"])
def test_logic_root_not_holding_matchtable_sources_is_refused(form, tmp_path, monkeypatch, capsys):
    build = build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()])
    other = tmp_path / "other-logic"
    shutil.copytree(build / "source-logic", other)
    extra = []
    if form == "explicit":
        extra = ["--logic-root", str(other)]
    else:
        (build / "params.txt").write_text(f"logic={other}\n")
    out = tmp_path / "out"
    code = invoke(command(tmp_path, logs.parent, facts.parent, out, extra=extra), monkeypatch)
    text = capsys.readouterr().err
    assert code == 2
    assert "MatchTable" in text and str(other) in text
    assert not (out / "specialize_manifest.json").exists()


def test_symlinked_logic_root_to_the_matchtable_sources_is_accepted(tmp_path, monkeypatch):
    build = build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()])
    link = tmp_path / "logic-link"
    link.symlink_to(build / "source-logic", target_is_directory=True)
    out = tmp_path / "out"
    code = invoke(command(tmp_path, logs.parent, facts.parent, out, extra=["--logic-root", str(link)]), monkeypatch)
    result = outputs(out, code)
    assert code == 0
    assert result["manifest"]["calls"][0]["recipe"]["source_file"] == RECIPES["dp"]["logic_file"]


def _two_solution_file(build, first_index, second_index):
    """dp's solution and solution98 in one file, with the given SolutionIndex values."""
    dp = yaml.safe_load((build / "source-logic" / RECIPES["dp"]["logic_file"]).read_text())
    other = yaml.safe_load((build / "source-logic" / RECIPES["dp_other"]["logic_file"]).read_text())
    first, second = dict(dp["Solutions"][0], SolutionIndex=first_index), dict(other["Solutions"][0],
                                                                              SolutionIndex=second_index)
    combined = dict(other, Solutions=[first, second])
    path = build / "source-logic" / "gfx950/gfx950/Equality/combined.yaml"
    path.write_text(yaml.safe_dump(combined, sort_keys=False))
    return path


def _point(build, index, path, value):
    table_path = build / "device-library/MatchTable.yaml"
    table = yaml.safe_load(table_path.read_text())
    table[index] = [str(path), value]
    table_path.write_text(yaml.safe_dump(table))


# Pass: the MatchTable's SolutionIndex names the recipe; names only confirm it, never pick
# another solution. Break: fall back to the value as a position when names differ.
def test_names_never_redirect_an_authoritative_solution_index(tmp_path, monkeypatch):
    build = build_fixture(tmp_path / "baseline")
    path = _two_solution_file(build, 1, 7)
    index = RECIPES["dp_other"]["baseline_index"]
    _point(build, index, path, 1)  # SolutionIndex 1 is dp's solution, at position 0
    logs, facts = input_fixture(tmp_path, [row("dp_other", call_count=4)])
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    assert result["code"] == 1
    assert result["logic"] == {}
    assert result["refusals"][0]["code"] == "recipe-name-mismatch"


def test_solution_index_selects_its_solution_at_any_position(tmp_path, monkeypatch):
    build = build_fixture(tmp_path / "baseline")
    path = _two_solution_file(build, 1, 7)
    index = RECIPES["dp_other"]["baseline_index"]
    _point(build, index, path, 7)  # positive control: solution98 carries SolutionIndex 7, at position 1
    logs, facts = input_fixture(tmp_path, [row("dp_other", call_count=4)])
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    assert result["code"] == 0
    assert result["manifest"]["calls"][0]["recipe"]["in_file_index"] == 1


@pytest.mark.parametrize("values,reason", [((5, 5), "SolutionIndex 5"), ((1, 7), "SolutionIndex 0")],
                         ids=["ambiguous", "absent"])
def test_unresolvable_solution_index_is_refused(values, reason, tmp_path, monkeypatch):
    build = build_fixture(tmp_path / "baseline")
    path = _two_solution_file(build, *values)
    index = RECIPES["dp_other"]["baseline_index"]
    _point(build, index, path, 5 if values == (5, 5) else 0)
    logs, facts = input_fixture(tmp_path, [row("dp_other", call_count=4)])
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch))
    assert result["code"] == 1
    assert result["refusals"][0]["code"] == "recipe-name-mismatch"
    assert reason in result["refusals"][0]["reason"]


# Pass: a malformed MatchTable entry names the TensileCreateLibrary form.
def test_malformed_matchtable_entry_names_solution_index(tmp_path, monkeypatch, capsys):
    build = build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()])
    table_path = build / "device-library/MatchTable.yaml"
    table = yaml.safe_load(table_path.read_text())
    table[RECIPES["dp"]["baseline_index"]] = [table[RECIPES["dp"]["baseline_index"]][0]]
    table_path.write_text(yaml.safe_dump(table))
    code = invoke(command(tmp_path, logs.parent, facts.parent, tmp_path / "out"), monkeypatch)
    err = capsys.readouterr().err
    assert code == 2
    assert "[source logic file, SolutionIndex]" in err and "position" not in err
