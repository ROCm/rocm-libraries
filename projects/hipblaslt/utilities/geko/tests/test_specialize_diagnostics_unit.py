# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""geko --specialize diagnostics: shipped Stream-K support, unknown facts, route values,
configuration names, derivation errors, --help and missing-TensileLite messages.

Pass/break pairs are recorded beside each case. Production paths are real.
"""

import json
import shutil
import sys

import pytest

from geko import cli
from tests.specialize_test_support import (
    RECIPES, build_fixture, effective_solution, input_fixture, invoke, one_solution, outputs, row, specialize,
)
from tests.test_route_facts_producer_unit import api_trace, produce
from tests.test_specialize_recipe_unit import command
from tests.test_specialize_runtime_route_unit import edit


# Pass: the shipped generator honours beta-off on the real Stream-K recipe.
# Break: withhold beta narrowing despite known support in the shipped generator.
def test_streamk_recipe_narrows_beta_with_shipped_generator(tmp_path, monkeypatch):
    result = specialize(tmp_path, [row("sk")], monkeypatch)
    assert result["code"] == 0 and result["refusals"] == []
    logic = one_solution(result)
    assert logic["ProblemType"]["UseBeta"] is False
    assert effective_solution(logic)["BespokeNarrowing"] is True
    assert not any(n["fact"] == "UseBeta" for n in result["manifest"]["calls"][0]["facts_not_applied"])


# Pass: every unknown or non-ordinary route string lists the accepted value.
@pytest.mark.parametrize("value", ["grouped-gemm-x", "device-args-x", "rocroller-x", "hipblaslt-ext"])
def test_nonordinary_route_lists_accepted_value(value, tmp_path, monkeypatch):
    result = specialize(tmp_path, [row(call_count=7)], monkeypatch, route_changes={"call_route": value})
    assert result["code"] == 1
    reason = result["refusals"][0]["reason"]
    assert value in reason and "ordinary matmul API" in reason


# Pass: a TensileLite error names the stock recipe file, not the scratch file.
def test_derivation_error_names_the_recipe_source(tmp_path, monkeypatch):
    import tensilelite.LibraryIO as library_io

    def parse(path, *args, **kwargs):
        raise RuntimeError("synthetic TensileLite failure")

    monkeypatch.setattr(library_io, "parseLibraryLogicFile", parse)
    result = specialize(tmp_path, [row("f32", call_count=7)], monkeypatch)
    assert result["code"] == 1
    reason = result["refusals"][0]["reason"]
    assert RECIPES["f32"]["logic_file"].split("/")[-1] in reason
    assert "stock_" not in reason and "candidate.yaml" not in reason
    assert "synthetic TensileLite failure" in reason


# Pass: the producer is discoverable from geko --help.
def test_help_lists_route_facts_producer(capsys):
    assert cli.main(["--help"]) == 0
    # "--route-facts" (the option) is always listed; the producer command must be named.
    assert "geko route-facts" in capsys.readouterr().out


# Pass: the missing-TensileLite diagnostic gives generic setup, no private names.
def test_tensilelite_import_error_is_generic(tmp_path, monkeypatch, capsys):
    build_fixture(tmp_path / "baseline")
    logs, facts = input_fixture(tmp_path, [row()])
    monkeypatch.setitem(sys.modules, "tensilelite.LibraryIO", None)
    code = invoke(command(tmp_path, logs.parent, facts.parent, tmp_path / "out"), monkeypatch)
    text = capsys.readouterr().err
    assert code == 2
    # The whole diagnostic is the public setup text: nothing else (no path, host or project name).
    assert text == ("geko --specialize: usage: TensileLite is not importable (import of tensilelite.LibraryIO "
                    "halted; None in sys.modules); put the source checkout's projects/hipblaslt/tensilelite "
                    "directory and its built rocisa package on PYTHONPATH (see the README, Specialize setup)\n")


# Pass: producer and consumer name a model-run out/ directory by its run.json
# run_configuration. Break: the consumer keys on the directory basename ("out").
@pytest.mark.parametrize("form", ["directory", "file"])
def test_model_run_out_directory_uses_its_recorded_configuration_name(form, tmp_path, monkeypatch):
    build_fixture(tmp_path / "baseline")
    logs, _ = input_fixture(tmp_path, [row(call_count=5)], cfg="example-cfg")
    out_dir = tmp_path / "runs/example/out"
    out_dir.parent.mkdir(parents=True)
    shutil.move(str(logs), out_dir)
    trace_meta = json.loads((out_dir / "run.json").read_text())
    api = api_trace(tmp_path / "runs/example-api/out", 5)
    edit(api / "run.json", run_configuration="example-cfg", library_fp=trace_meta["library_fp"])
    facts = tmp_path / "facts.json"
    assert produce(out_dir, api, facts, monkeypatch) == 0
    assert json.loads(facts.read_text())["run_configuration"] == "example-cfg"
    workload = out_dir if form == "directory" else out_dir / "hipblaslt_101.log"
    work = tmp_path / "work"
    result = outputs(work, invoke(command(tmp_path, workload, facts, work, extra=["--keep-beta"]), monkeypatch))
    assert result["code"] == 0, result["refusals"]
    assert [c["run_configuration"] for c in result["manifest"]["calls"]] == ["example-cfg"]


# Pass: two directories naming the same run configuration are a usage error naming it.
# Break: silently merge or shadow one configuration.
def test_duplicate_configuration_names_are_a_usage_error(tmp_path, monkeypatch, capsys):
    build_fixture(tmp_path / "baseline")
    first, _ = input_fixture(tmp_path, [row(call_count=3)], cfg="a")
    second, _ = input_fixture(tmp_path, [row(N=619, call_count=4)], cfg="b")
    edit(second / "run.json", run_configuration="a")
    out = tmp_path / "out"
    code = invoke(command(tmp_path, first.parent, tmp_path / "route_facts", out), monkeypatch)
    err = capsys.readouterr().err
    assert code == 2
    assert "run configuration a" in err
    assert not (out / "specialize_manifest.json").exists()
