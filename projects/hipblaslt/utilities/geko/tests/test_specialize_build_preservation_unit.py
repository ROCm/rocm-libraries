# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""--build-library preservation: TCL options, markers, helpers and kernel names.

Break: change a build option/arch/jobs argument, drop the serialized bespoke marker
or helper generation, or erase the bespoke/general symbol distinction.
"""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from geko.specialize import writer
from tests.specialize_build_test_support import (
    files, gfx942_case, library_solutions, match_table, merged_case, run_tcl,
    subset_fixture, symbols,
)
from tests.specialize_test_support import RECIPES, command, identity, invoke, outputs


# Flags match the cmake build; arch and output agree; gfx942 output is labelled
# untested (manifest, calls, output-local UNTESTED-gfx942 file).
@pytest.mark.parametrize("case_fixture,arch", [("merged_case", "gfx950"), ("gfx942_case", "gfx942")])
def test_tcl_options_and_selected_architecture_are_preserved(case_fixture, arch, request):
    case = request.getfixturevalue(case_fixture)
    assert case["code"] == 0
    metadata = case["manifest"]["build"]["library"]
    argv = metadata["command"]
    assert argv[1:3] == ["-m", "tensilelite.TensileCreateLibrary"]
    assert set(argv[3:-3]) == {
        f"--architecture={arch}", "--cxx-compiler=/opt/rocm/bin/amdclang++", "--jobs=1",
        "--no-compress", "--disable-asm-comments", "--gfx1250v0"}
    assert argv[-2:] == [str(case["destination"]), "HIP"]
    assert metadata["jobs"] == 1 and metadata["exit"] == 0
    directory = case["destination"] / "library" / arch
    assert list(directory.glob("TensileLibrary_lazy_*.dat.zlib")), "default msgpack + lazy loading must be preserved"
    assert not (case["destination"] / "library" / ("gfx942" if arch == "gfx950" else "gfx950")).exists()
    if arch == "gfx942":
        assert case["manifest"].get("label") == "untested-gfx942"
        assert (case["destination"] / "UNTESTED-gfx942").is_file()
        assert case["manifest"]["calls"][0]["recipe"]["source_file"] == RECIPES["g942"]["logic_file"]
        assert all(call.get("label") == "untested-gfx942" for call in case["manifest"]["calls"])


# Pass: every expected name resolves through actual TCL index remapping to a
# serialized marked solution and a real ELF kernel descriptor. Break: omit a
# bespoke solution/marker/code object, or let none/relu share one mapped kernel.
def test_every_bespoke_identity_has_a_marked_built_kernel(merged_case):
    case = merged_case
    assert case["code"] == 0
    directory = case["destination"] / "library/gfx950"
    solutions = library_solutions(directory)
    table = match_table(case["destination"] / "library/MatchTable.yaml")
    assert table
    assert {tuple(sorted(c["identity"].items())) for c in case["manifest"]["calls"]} == {
        tuple(sorted(identity(record).items())) for record in case["records"]}
    kernel_symbols = symbols(directory)
    names = set()
    for call in case["manifest"]["calls"]:
        indices = [index for index, (source, local) in table.items()
                   if Path(source).name == call["logic_file"]]
        assert len(indices) == 1
        source, local = table[indices[0]]
        emitted = yaml.safe_load((case["work"] / "logic/gfx950" / call["logic_file"]).read_text())
        assert any(solution.get("SolutionIndex", position) == local
                   for position, solution in enumerate(emitted["Solutions"]))
        solution = solutions[indices[0]]
        assert solution.get("bespokeNarrowing") is True, "deployed bespoke solution lost its launch-guard marker"
        assert solution["kernelName"] == call["kernel_name"]
        assert solution["libraryLogicIndex"] == local
        assert solution["name"] == call["solution_name"]
        assert call["kernel_name"] + ".kd" in kernel_symbols
        names.add(call["kernel_name"])
    assert len(names) == 2, "same shape with none/relu epilogues requires different bespoke kernels"


# Pass: compile all three vector widths of each bespoke conversion helper.
# Break: suppress helper generation or accidentally retain a general epilogue.
# Literal names are hand-derived: float workspace -> bf16 D (SB), no bias/SAV,
# activation none or relu, general beta, PostGSU16, vector widths 1/2/4.
def test_every_bespoke_conversion_helper_is_in_built_code_objects(merged_case):
    directory = merged_case["destination"] / "library/gfx950"
    expected = {
        "Cijk_SB_Bespoke_PostGSU16_VW1.kd", "Cijk_SB_Bespoke_PostGSU16_VW2.kd", "Cijk_SB_Bespoke_PostGSU16_VW4.kd",
        "Cijk_SB_RELUS_Bespoke_PostGSU16_VW1.kd", "Cijk_SB_RELUS_Bespoke_PostGSU16_VW2.kd", "Cijk_SB_RELUS_Bespoke_PostGSU16_VW4.kd",
    }
    assert expected <= symbols(directory), "built library is missing a required bespoke helper descriptor"


# Pass: general and bespoke names/code/metadata coexist regardless of logic-file
# insertion order. Break: erase explicit bespoke naming, collapse same-shape
# epilogues, or overwrite a general lazy-library row while merging bespoke rows.
@pytest.mark.parametrize("bespoke_first", [False, True], ids=["stock-first", "bespoke-first"])
def test_general_and_bespoke_kernel_names_never_collide_in_either_order(bespoke_first, merged_case, tmp_path):
    case = merged_case
    logic = tmp_path / "logic/gfx950"
    logic.mkdir(parents=True)
    stock_names = {RECIPES["dp"]["kernel_name"], RECIPES["dp_other"]["kernel_name"]}
    bespoke_names = {call["kernel_name"] for call in case["manifest"]["calls"]}
    groups = [("bespoke", [case["work"] / "logic/gfx950" / call["logic_file"]
                            for call in case["manifest"]["calls"]]),
              ("stock", [case["baseline"] / "source-logic" / relative for relative in case["before"]])]
    if not bespoke_first:
        groups.reverse()
    for position, (_, paths) in enumerate(groups):
        for number, path in enumerate(paths):
            shutil.copyfile(path, logic / f"{position:02d}_{number}_gfx950_{path.name}")
    directory = run_tcl(tmp_path / "logic", tmp_path / "control-built")
    solutions = library_solutions(directory)
    assert stock_names.isdisjoint(bespoke_names)
    assert len(solutions) == 4
    assert {s["kernelName"] for s in solutions.values()} == stock_names | bespoke_names
    assert {s["kernelName"] for s in solutions.values() if s.get("bespokeNarrowing")} == bespoke_names
    assert all("bespokeNarrowing" not in s for s in solutions.values() if s["kernelName"] in stock_names)
    kernel_symbols = symbols(directory)
    assert {name + ".kd" for name in stock_names | bespoke_names} <= kernel_symbols
    assert files(case["baseline"] / "source-logic") == case["before"]


# Requested jobs reach TCL unchanged; the default is min(affinity, 64). Only the
# external compiler process is intercepted; parsing, specialization, argument
# construction and manifest writes stay real. Break: clamp, ignore or uncap --jobs.
@pytest.mark.parametrize("jobs,expected", [(1, 1), (7, 7), (64, 64), (None, 64)])
def test_jobs_limit_is_honoured_at_the_tcl_boundary(jobs, expected, tmp_path, monkeypatch):
    baseline, logs, facts, _ = subset_fixture(tmp_path)
    actual_run = subprocess.run
    commands = []

    def capture_tcl(argv, *args, **kwargs):
        if argv[1:3] == ["-m", "tensilelite.TensileCreateLibrary"]:
            commands.append(list(argv))
            return SimpleNamespace(returncode=0)
        return actual_run(argv, *args, **kwargs)

    monkeypatch.setattr(writer.subprocess, "run", capture_tcl)
    monkeypatch.setattr(writer.os, "sched_getaffinity", lambda _: set(range(128)))
    out = tmp_path / "work"
    extra = ["--keep-beta", "--build-library", str(tmp_path / "built")]
    if jobs is not None:
        extra.extend(["--jobs", str(jobs)])
    code = invoke(command(tmp_path, logs, facts, out, extra=extra), monkeypatch)
    assert code == 0
    assert len(commands) == 1
    assert [arg for arg in commands[0] if arg.startswith("--jobs=")] == [f"--jobs={expected}"]
    library = outputs(out, code)["manifest"]["build"]["library"]
    assert library["jobs"] == expected
    assert library["exit"] == 0


# Break: accept 0 or 65 workers, silently clamp them, or start building anyway.
@pytest.mark.parametrize("jobs", [0, 65])
def test_out_of_range_jobs_refuses_before_any_build(jobs, tmp_path, monkeypatch, capsys):
    _, logs, facts, _ = subset_fixture(tmp_path)
    actual_run = subprocess.run

    def forbid_tcl(argv, *args, **kwargs):
        assert argv[1:3] != ["-m", "tensilelite.TensileCreateLibrary"], "invalid jobs must never reach TCL"
        return actual_run(argv, *args, **kwargs)

    monkeypatch.setattr(writer.subprocess, "run", forbid_tcl)
    destination, work = tmp_path / "built", tmp_path / "work"
    code = invoke(command(tmp_path, logs, facts, work,
                          extra=["--build-library", str(destination), "--jobs", str(jobs)]), monkeypatch)
    assert code == 2
    assert not destination.exists()
    assert not (work / "specialize_manifest.json").exists()
    if jobs == 65:
        text = capsys.readouterr()
        assert "64" in text.out + text.err, "usage diagnostic must explain the permitted CPU-job limit"
