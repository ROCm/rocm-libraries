# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""geko --specialize usability: public input layouts, unknown facts, and public diagnostics.

Pass/break pairs are recorded beside each case. All generation and recipe
resolution are real; only GPU enumeration is forbidden by the shared helper.
"""
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from tests.specialize_test_support import (
    RECIPES, build_fixture, input_fixture, invoke, outputs, row, specialize, tree_hash,
)
from tests.test_specialize_recipe_unit import command

# Generic shapes of internal tracking references that must never reach user-visible output,
# matched case-insensitively. Public names of the same word-letter shape (identity fields
# stride_a .. stride_d and use_e, the Stream-K algorithm) are excluded by name.
PUBLIC_LETTER_TERMS = r"(?:stride_[abcd]|use_e|stream-k)\b"
INTERNAL = re.compile(
    r"§"                                                    # document section references
    r"|\b(?:tasks?|issues?|tickets?)(?:\s+|[_-])?\d+"       # numbered work items
    r"|\b[A-Z][A-Z0-9]{0,2}(?:-(?:[A-Z]\d+|\d{2,}))+\b"     # short tracker-style keys (K-12, KY-A3)
    r"|\b(?!" + PUBLIC_LETTER_TERMS + r")[a-z]{5,}[-_][a-z]\b"  # lettered milestone labels (name-A)
    , re.I)  # User-supplied absolute paths legitimately appear in outputs; paths are checked where they cannot.


# Pass: source checkout discovery or explicit --logic-root replaces params.txt.
# Break: require our private params.txt, or ignore the explicit root's precedence.
@pytest.mark.parametrize("layout", ["source_checkout", "explicit", "explicit_overrides_params", "install"])
def test_standard_baseline_finds_logic_without_private_build_metadata(layout, tmp_path, monkeypatch):
    build = build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()])
    source = build / "source-logic"
    extra = []
    if layout == "source_checkout":
        destination = tmp_path / "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
        destination.parent.mkdir(parents=True)
        shutil.move(str(source), destination)
        table = yaml.safe_load((build / "device-library/MatchTable.yaml").read_text())
        table = {i: [p.replace(str(source), str(destination)), n] for i, (p, n) in table.items()}
        (build / "device-library/MatchTable.yaml").write_text(yaml.safe_dump(table))
    else:
        extra = ["--logic-root", str(source)]
    if layout == "explicit_overrides_params":
        # Exists, but must not override the explicit source directory.
        (build / "params.txt").write_text("logic=/does-not-exist\n")
    else:
        (build / "params.txt").unlink()
    if layout == "install":
        installed = build / "lib/hipblaslt/library"
        installed.parent.mkdir(parents=True)
        shutil.copytree(build / "Tensile/library/gfx950", installed)
        shutil.copyfile(build / "device-library/MatchTable.yaml", installed / "MatchTable.yaml")
        shutil.rmtree(build / "Tensile")
        shutil.rmtree(build / "device-library")
        metadata = json.loads((logs / "run.json").read_text())
        metadata["library_fp"] = tree_hash(installed)
        (logs / "run.json").write_text(json.dumps(metadata))
    out = tmp_path / "out"
    result = outputs(out, invoke(command(tmp_path, logs.parent, facts.parent, out, extra=extra), monkeypatch))
    assert result["code"] == 0, "a standard baseline with its own MatchTable and discoverable source logic must generate"
    assert len(result["logic"]) == 1
    assert result["manifest"]["calls"][0]["recipe"]["sha256"] == RECIPES["dp"]["source_sha256"]


# Pass: never substitute a different build's MatchTable; identify the missing map.
# Break: fall back to an unrelated table or report only missing private params.
def test_standard_baseline_without_matchtable_names_the_missing_recipe_map(tmp_path, monkeypatch, capsys):
    build = build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()])
    (build / "params.txt").unlink()
    (build / "device-library/MatchTable.yaml").unlink()
    out = tmp_path / "out"
    code = invoke(command(tmp_path, logs.parent, facts.parent, out), monkeypatch)
    text = capsys.readouterr()
    assert code == 2
    assert "MatchTable" in text.out + text.err
    assert outputs(out, code)["logic"] == {}


# Pass: optional facts reach counted refusal; the ordinary trace cannot prove its
# route/amax absence. Break: require --route-facts syntactically or assume safety.
def test_omitted_route_facts_refuses_unknown_calls_instead_of_rejecting_cli(tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row(call_count=7)], run=False)
    args = command(tmp_path, logs.parent, facts.parent, tmp_path / "out")
    index = args.index("--route-facts")
    del args[index:index + 2]
    result = outputs(tmp_path / "out", invoke(args, monkeypatch))
    assert result["code"] == 1, "omitting route facts is valid CLI input, not a usage error"
    assert result["logic"] == {}
    assert len(result["refusals"]) == 1
    assert result["refusals"][0]["call_count"] == 7
    assert "route" in result["refusals"][0]["reason"].lower()


# Pass: distinguish unknown values from supported ones with useful alternatives.
# Break: silently interpret an arbitrary string as false/default/M, or give no
# supported value in the diagnostic. No full sentence is pinned.
@pytest.mark.parametrize("field,value,accepted", [
    ("call_route", "mystery-route", "ordinary matmul API"),
    ("bias_axis", "mystery-axis", "M"),
    ("output_amax", "mystery-amax", "false"),
])
def test_unknown_route_fact_values_are_refused_with_accepted_values(field, value, accepted, tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(bias_vector=True, call_count=7)], monkeypatch,
                        route_changes={field: value})
    assert result["code"] == 1
    assert result["logic"] == {}, "unknown fact strings must not silently narrow a solution"
    assert len(result["refusals"]) == 1
    reason = result["refusals"][0]["reason"]
    assert value in reason
    assert re.search(r"\b" + re.escape(accepted) + r"\b", reason, re.I), "diagnostic must name an accepted value"


# Pass: public outputs use descriptive statuses, stage labels and reasons (keep-beta).
# Break: leak internal milestone/issue identifiers through any serialized output.
@pytest.mark.parametrize("kind", ["keep_beta", "custom_kernel", "invalid_jobs", "excluded_size"])
def test_user_outputs_do_not_leak_internal_tracking_identifiers(kind, tmp_path, monkeypatch, capsys):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row("custom" if kind == "custom_kernel" else "dp")])
    extra = ["--keep-beta"] if kind == "keep_beta" else []
    if kind == "invalid_jobs":
        extra = ["--jobs", "65"]
    if kind == "excluded_size":
        exclude = tmp_path / "exclude.yaml"
        exclude.write_text(yaml.safe_dump({"exclusions": [{
            "recipe": {"source_file": RECIPES["dp"]["logic_file"], "sha256": RECIPES["dp"]["source_sha256"], "in_file_index": 0},
            "identity": {}, "facts": ["AssertSizeEqual"],
        }]}))
        extra = ["--exclude-controls", str(exclude)]
    out = tmp_path / "out"
    code = invoke(command(tmp_path, logs.parent, facts.parent, out, extra=extra), monkeypatch)
    assert code == (0 if kind == "keep_beta" else 1 if kind == "custom_kernel" else 2)
    captured = capsys.readouterr()
    surfaces = [captured.out, captured.err]
    surfaces += [p.read_text() for p in out.glob("*.json")]
    if (out / "calls.csv").is_file():
        surfaces.append((out / "calls.csv").read_text())
    for surface in surfaces:
        # Paths the test itself supplied are echoed back by design; only geko's own text is checked.
        surface = surface.replace(str(tmp_path), "<tmp>")
        assert INTERNAL.search(surface) is None, "public output contains an internal tracking identifier"
    if kind == "keep_beta":
        manifest = outputs(out, code)["manifest"]
        assert manifest["stage"] == "keep-beta"
        assert manifest["calls"][0]["keep_beta"] is True
        reason = manifest["calls"][0]["facts_not_applied"][0]["reason"]
        assert "beta" in reason.lower()


# Pass: the public instructions are standalone, not project-tracker shorthand.
# Break: leave internal identifiers in the instructions shipped to external users.
def test_readme_is_free_of_internal_tracking_identifiers():
    readme = Path(__file__).parents[1] / "README.md"
    assert INTERNAL.search(readme.read_text()) is None, "README requires private project tracking context"


# Pass: recognize geko's actual build output as a baseline, or identify the real
# unavailable source prerequisite. Break: report a missing MatchTable/params
# layout even though geko itself emitted the build-local map and stock logic.
def test_geko_build_output_is_usable_or_names_a_real_missing_prerequisite(tmp_path, monkeypatch, capsys):
    from tests.specialize_build_test_support import match_table, subset_fixture
    from tests.specialize_test_support import write_rows

    first = tmp_path / "first"
    _, logs, facts, records = subset_fixture(first)
    built = tmp_path / "built"
    code = invoke(command(first, logs, facts, first / "out", extra=[
        "--stock-only", "--keep-beta", "--build-library", str(built), "--jobs", "1",
    ]), monkeypatch)
    assert code == 0, "positive-control geko library build must succeed"
    table = match_table(built / "library/MatchTable.yaml")
    assert table, "positive-control build must emit its own MatchTable"
    index = next(i for i, (source, _) in table.items() if source.endswith(RECIPES["dp"]["logic_file"]))
    trace = logs / "fixture-on/hipblaslt_101.log"
    write_rows(trace, [{**record, "solution_index": index} for record in records])
    run = logs / "fixture-on/run.json"
    metadata = json.loads(run.read_text())
    metadata["library_fp"] = tree_hash(built / "library/gfx950")
    run.write_text(json.dumps(metadata))
    capsys.readouterr()
    out = tmp_path / "reuse"
    code = invoke(["--specialize", "--workload-log", str(logs), "--route-facts", str(facts),
                   "--hipblaslt", str(built), "--arch", "gfx950", "--workdir", str(out),
                   "--keep-beta"], monkeypatch)
    captured = capsys.readouterr()
    if code == 0:
        assert len(outputs(out, code)["manifest"]["calls"]) == 2
    else:
        text = captured.out + captured.err
        assert "logic-root" in text or "stock logic" in text.lower(), "refusal must identify a real missing source prerequisite, not reject geko's own build layout"
        assert outputs(out, code)["logic"] == {}
