# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""BespokeNarrowing: assertion-derived code removal is opt-in.

Feature: a solution without `BespokeNarrowing: true` (all stock logic) keeps
its store edge checks/bodies and tail/exact-size paths, and keeps its stock
kernel name, even when AF0EM/AF1EM are multiples of the tile or AssertSizeEqual
pins exact sizes. With `BespokeNarrowing: true` the store-trim and exact-size removal applies,
and the kernel name differs from the stock name. The key is a validated bool.
"""
import copy
import json
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
HERE = Path(__file__).parent
STORE_LOGIC = HERE / "store_trim_data" / "gfx950_sk3_userargs_mt32x64.yaml"  # MT32x64
EXACT_DATA = HERE / "exact_size_data"
LOOP_LOGIC = HERE / "test_data" / "half_exit_gfx950_mt32x64.yaml"


def write_logic(tmp_path, data, name):
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def with_key(data, value):
    data = copy.deepcopy(data)
    for solution in data["Solutions"]:
        if value is None:
            solution.pop("BespokeNarrowing", None)
        else:
            solution["BespokeNarrowing"] = value
    return data


def emit(tmp_path, data, name):
    from codegen_harness import emit_kernels_from_logic, solutions_from_logic
    from tensilelite.SolutionStructs.Naming import getKernelNameMin

    path = write_logic(tmp_path, data, name)
    solutions = solutions_from_logic(path)
    assert len(solutions) == 1 and solutions[0]["Valid"], "fixture must derive normally"
    emitted = emit_kernels_from_logic(path, canonical=False)
    assert len(emitted) == 1 and emitted[0][2] == 0, "fixture must emit normally"
    return solutions[0], getKernelNameMin(solutions[0], False), emitted[0][1]


def edge_dispatch(source):
    return re.findall(r"\bs_cbranch_scc1\s+(label_GW_\w+(?:Then|Else))\b", source)


def edge_bodies(source):
    return re.findall(r"^label_GW_\w+_Then(?:_\w+)?:", source, re.M)


def store_logic(af0=32, af1=64):
    data = yaml.safe_load(STORE_LOGIC.read_text())
    for solution in data["Solutions"]:
        solution.update(AssertFree0ElementMultiple=af0, AssertFree1ElementMultiple=af1)
    return data


def exact_logic():
    data = yaml.safe_load((EXACT_DATA / "gfx950_mt160.yaml").read_text())
    data["Solutions"][0]["AssertSizeEqual"] = {0: 320, 1: 64}
    return data


def loop_logic(k=384):
    data = yaml.safe_load(LOOP_LOGIC.read_text())
    data["Solutions"][0].update(DepthU=192, TailloopInNll=False, StreamK=0, GlobalSplitU=1,
                                PrefetchGlobalRead=1, ExpandPointerSwap=True,
                                UnrollLoopSwapGlobalReadOrder=0, AssertSummationElementMultiple=1,
                                AssertSizeEqual={3: k})
    return data


@pytest.mark.parametrize("value", [None, False], ids=["absent", "false"])
def test_tile_multiple_assertions_keep_store_edges_without_opt_in(tmp_path, value):
    _, _, source = emit(tmp_path, with_key(store_logic(), value), "stock_like")
    assert {d.rsplit("_", 1)[1] for d in edge_dispatch(source)} == {"Then", "Else"}, \
        "stock-like AF0EM/AF1EM multiple of the tile lost an edge check"
    assert edge_bodies(source), "stock-like AF0EM/AF1EM multiple of the tile lost its edge store body"


def test_tile_multiple_assertions_remove_store_edges_with_opt_in(tmp_path):
    _, _, source = emit(tmp_path, with_key(store_logic(), True), "bespoke")
    assert edge_dispatch(source) == [], "opted-in full tiles still emit edge dispatch"
    assert edge_bodies(source) == [], "opted-in full tiles still emit an edge store body"


def test_name_carries_opt_in_only_when_true(tmp_path):
    _, absent, _ = emit(tmp_path, with_key(store_logic(), None), "absent")
    _, false, _ = emit(tmp_path, with_key(store_logic(), False), "false")
    _, true, _ = emit(tmp_path, with_key(store_logic(), True), "true")
    assert absent == false, "an explicit false changed the kernel name"
    assert true != absent, "opted-in (edge-free) code shares the stock kernel name"


@pytest.mark.parametrize("value", [None, False], ids=["absent", "false"])
def test_exact_sizes_keep_edges_and_stock_name_without_opt_in(tmp_path, value):
    _, name, source = emit(tmp_path, with_key(exact_logic(), value), "stock_exact")
    assert edge_dispatch(source), "AssertSizeEqual without opt-in removed the edge dispatch"
    stock = json.loads((EXACT_DATA / "stock_names.json").read_text())["160"]["symbol"]
    assert name == stock, "AssertSizeEqual without opt-in changed the stock kernel name"


def test_exact_sizes_remove_edges_and_rename_with_opt_in(tmp_path):
    _, name, source = emit(tmp_path, with_key(exact_logic(), True), "bespoke_exact")
    assert edge_dispatch(source) == [] and edge_bodies(source) == []
    stock = json.loads((EXACT_DATA / "stock_names.json").read_text())["160"]["symbol"]
    assert name != stock


def test_exact_k_keeps_tail_without_opt_in(tmp_path):
    solution, _, _ = emit(tmp_path, with_key(loop_logic(), None), "stock_k")
    assert solution["NoTailLoop"] is False, "AssertSizeEqual K without opt-in removed the tail loop"


def test_exact_k_removes_tail_with_opt_in(tmp_path):
    solution, _, _ = emit(tmp_path, with_key(loop_logic(), True), "bespoke_k")
    assert solution["NoTailLoop"] is True


def test_contradicting_exact_size_refused_only_with_opt_in(tmp_path):
    from codegen_harness import solutions_from_logic

    data = yaml.safe_load((EXACT_DATA / "gfx950_mt160.yaml").read_text())
    data["Solutions"][0].update(AssertFree0ElementMultiple=64, AssertSizeEqual={0: 480})
    solutions = solutions_from_logic(write_logic(tmp_path, data, "stock_contradiction"))
    assert len(solutions) == 1 and solutions[0]["Valid"], "stock logic without opt-in must derive as before"
    with pytest.raises(RuntimeError):
        solutions_from_logic(write_logic(tmp_path, with_key(data, True), "bespoke_contradiction"))


def test_bespoke_narrowing_is_a_validated_bool():
    from tensilelite.Common.ValidParameters import checkParametersAreValid, validParameters
    from tensilelite.SolutionStructs.Solution import validateParameterTypes

    assert validParameters["BespokeNarrowing"] == [False, True]
    checkParametersAreValid(("BespokeNarrowing", [True]), validParameters)
    with pytest.raises(Exception):
        checkParametersAreValid(("BespokeNarrowing", ["yes"]), validParameters)
    assert validateParameterTypes({"BespokeNarrowing": True}) == []
    assert validateParameterTypes({"BespokeNarrowing": 1}), "an int must not pass as the bool key"


@pytest.mark.parametrize("value", [None, False, True], ids=["absent", "false", "true"])
def test_library_solution_marks_only_opted_in_solutions(tmp_path, value):
    # The host guards only solutions whose library entry carries bespokeNarrowing: true;
    # stock entries must serialize exactly as before (no key).
    from codegen_harness import _isolated_globals, _solutions_from_logic_unguarded, \
        get_assembler, get_isa_info_map
    from tensilelite.Common import state
    from tensilelite.Contractions import Solution

    path = write_logic(tmp_path, with_key(store_logic(), value), "library_entry")
    with _isolated_globals():
        derived, = _solutions_from_logic_unguarded(path)
        entry = state(Solution.FromSolutionStruct(derived, False, False, False,
                                                  get_assembler(), get_isa_info_map()))
    assert entry.get("bespokeNarrowing") is (True if value else None)


@pytest.mark.parametrize("value", [None, False, True], ids=["absent", "false", "true"])
def test_single_activation_needs_opt_in(tmp_path, value):
    # A single ActivationType exists only in bespoke requests; without the key the
    # host would not guard the kernel, so generation refuses it.
    from codegen_harness import solutions_from_logic

    data = with_key(store_logic(), value)
    data["ProblemType"].update(Activation=True, ActivationType="gelu")
    path = write_logic(tmp_path, data, "single_gelu")
    if value:
        assert all(s["Valid"] for s in solutions_from_logic(path))
    else:
        with pytest.raises(RuntimeError):
            solutions_from_logic(path)
