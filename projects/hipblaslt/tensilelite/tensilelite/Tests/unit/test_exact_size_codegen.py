# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exact sizes: exact divisibility changes generated paths, not public multiples.

Feature: derive full tiles/divisible K using actual indices; include changed
emission facts in compile identity. Dropping those facts breaks these tests.
Preservation: nonmultiples, packed indices and odd/per-WG trips keep checks;
public assertion values and stock identities stay unchanged. The report records
production mutations that break each preservation class.
"""
import copy
import re
from pathlib import Path

import pytest
import yaml

from loop_exit_test_helpers import assemble, dec_blocks, pristine_default_solution

pytestmark = pytest.mark.unit
DATA = Path(__file__).parent / "exact_size_data"
LOOP = Path(__file__).parent / "test_data/half_exit_gfx950_mt32x64.yaml"


def document(width=160):
    return copy.deepcopy(yaml.safe_load((DATA / f"gfx950_mt{width}.yaml").read_text()))


def write_logic(tmp_path, data, name="exact"):
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def derive_emit(tmp_path, data):
    from codegen_harness import emit_kernels_from_logic, solutions_from_logic

    path = write_logic(tmp_path, data)
    solutions = solutions_from_logic(path)
    assert solutions and all(s["Valid"] for s in solutions), "fixture must derive normally"
    emitted = emit_kernels_from_logic(path, canonical=False)
    assert emitted and all(error == 0 for _, _, error in emitted), "fixture must emit normally"
    return solutions, emitted


def edge_dispatch(source):
    # Executable conditional branches into the edge store bodies, not prose.
    return re.findall(r"\bs_cbranch_scc1\s+(label_GW_\w+(?:Then|Else))\b", source)


def tail_instructions(source):
    return re.findall(r"\bs_(?:cbranch_\w+|branch)\s+(label_(?:TailLoop|LoopBeginK|LoopEndK)\w*)", source)


@pytest.mark.parametrize("width,sizes", [(160, {0: 320, 1: 64}), (240, {0: 384, 1: 480})])
def test_exact_full_nonpower_tiles_remove_edges(tmp_path, width, sizes):
    data = document(width)
    data["Solutions"][0].update(AssertSizeEqual=sizes, BespokeNarrowing=True)
    _, emitted = derive_emit(tmp_path, data)
    source = emitted[0][1]
    assert edge_dispatch(source) == [], "exact full nonpower tiles still emit edge dispatch"
    assert not re.search(r"^label_GW_\w+_Then(?:_\w+)?:", source, re.M), \
        "exact full nonpower tiles still emit an edge store body"
    assemble(tmp_path, "full", source)


@pytest.mark.parametrize("width,sizes", [(160, {0: 321, 1: 65}), (240, {0: 385, 1: 481}),
                                         (160, {0: 80, 1: 16}), (240, {0: 96, 1: 120})],
                         ids=["160-partial", "240-partial", "160-subtile", "240-subtile"])
def test_exact_nonmultiples_keep_edges(tmp_path, width, sizes):
    data = document(width)
    data["Solutions"][0].update(AssertSizeEqual=sizes, BespokeNarrowing=True)
    _, emitted = derive_emit(tmp_path, data)
    assert edge_dispatch(emitted[0][1]), "partial exact tile lost its edge dispatch"
    assemble(tmp_path, "partial", emitted[0][1])


def loop_document(k, *, streamk=0, runtime=False):
    data = copy.deepcopy(yaml.safe_load(LOOP.read_text()))
    solution = data["Solutions"][0]
    # Test-only constructed large nonpower DepthU counterexample from this
    # retained parent. The general/exact pair has precisely the same recipe.
    solution.update(DepthU=192, TailloopInNll=False, StreamK=streamk,
                    GlobalSplitU=1, PrefetchGlobalRead=1, ExpandPointerSwap=True,
                    UnrollLoopSwapGlobalReadOrder=0, AssertSummationElementMultiple=1,
                    AssertSizeEqual={3: k}, BespokeNarrowing=True)
    solution["InternalSupportParams"]["SupportUserGSU"] = runtime
    return data


@pytest.mark.parametrize("k", [192, 384, 576], ids=["one", "even", "odd"])
def test_exact_divisible_large_nonpower_depth_removes_tail(tmp_path, k):
    solutions, emitted = derive_emit(tmp_path, loop_document(k))
    assert solutions[0]["DepthU"] == 192
    assert solutions[0]["NoTailLoop"] is True, "exact K divisible by DU192 still needs a tail"
    assert not tail_instructions(emitted[0][1]), "divisible exact K still emits tail instructions"
    assemble(tmp_path, "divisible", emitted[0][1])


def test_unbatched_exact_k_uses_index_two(tmp_path):
    from codegen_harness import solutions_from_logic

    data = loop_document(384)
    data["ProblemType"].update(Batched=False, NumIndicesC=2, TotalIndices=3,
                               NumIndicesBatch=0, IndicesBatch=[], IndicesSummation=[2],
                               IndexUnroll=2, IndexAssignmentsA=[2, 0],
                               IndexAssignmentsB=[2, 1], IndexAssignmentsMetadata=[2, 0])
    data["Solutions"][0]["AssertSizeEqual"] = {2: 384}
    solutions = solutions_from_logic(write_logic(tmp_path, data))
    assert solutions and solutions[0]["Valid"], "unbatched recipe must derive normally"
    assert solutions[0]["ProblemType"]["IndicesSummation"] == [2]
    assert solutions[0]["NoTailLoop"] is True, "unbatched exact K was mistaken for batch index 2"


@pytest.mark.parametrize("k", [193, 385, 577])
def test_exact_nondivisible_k_keeps_tail(tmp_path, k):
    solutions, emitted = derive_emit(tmp_path, loop_document(k))
    assert solutions[0]["NoTailLoop"] is False, "nondivisible exact K lost its tail"
    assert tail_instructions(emitted[0][1]), "nondivisible exact K emits no tail branches"
    assemble(tmp_path, "tail", emitted[0][1])


@pytest.mark.parametrize("k,streamk,runtime", [(192, 0, False), (576, 0, False),
                                            (384, 3, False), (384, 0, True)],
                         ids=["one-trip", "odd-trip", "streamk", "runtime-gsu"])
def test_exact_k_without_per_wg_even_proof_keeps_both_exits(tmp_path, k, streamk, runtime):
    _, emitted = derive_emit(tmp_path, loop_document(k, streamk=streamk, runtime=runtime))
    decs = [(operand, following) for operand, following in dec_blocks(emitted[0][1]) if operand == "1"]
    assert len(decs) == 2 and all(operand == "1" for operand, _ in decs)
    assert all(next_instruction.startswith("s_cmp_") for _, next_instruction in decs), \
        "exact global K incorrectly removed a per-workgroup exit"
    assemble(tmp_path, "exits", emitted[0][1])


def test_exact_even_k_feeds_existing_half_exit_proof(tmp_path):
    _, emitted = derive_emit(tmp_path, loop_document(384))
    decs = dec_blocks(emitted[0][1])
    assert len(decs) == 2
    assert decs[0][1].startswith("s_cmp_")
    assert decs[1][1].startswith("s_branch label_LoopBeginL"), \
        "exact even K did not reach the existing fixed-GSU1 two-copy proof"


@pytest.mark.parametrize("field", ["AssertFree0ElementMultiple", "AssertFree1ElementMultiple", "AssertSummationElementMultiple"])
@pytest.mark.parametrize("width", [160, 240])
def test_exact_facts_do_not_overwrite_yaml_multiples(tmp_path, width, field):
    data = document(width)
    solution = data["Solutions"][0]
    solution.update(AssertSizeEqual={0: 480, 1: 480, 3: 384},
                    AssertFree0ElementMultiple=1, AssertFree1ElementMultiple=1,
                    AssertSummationElementMultiple=1, BespokeNarrowing=True)
    from codegen_harness import solutions_from_logic

    solutions = solutions_from_logic(write_logic(tmp_path, data))
    assert solutions and solutions[0]["Valid"], "fixture must derive normally"
    assert solutions[0][field] == 1, f"exact-size derivation overwrote public {field}"


@pytest.mark.parametrize("order", [(False, True), (True, False)], ids=["general-first", "exact-first"])
def test_one_library_keeps_general_and_exact_kernel_identities(tmp_path, order):
    from codegen_harness import solutions_from_logic
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions, groupKernelRecords
    from tensilelite.SolutionStructs.Naming import getKernelNameMin

    data = document(160)
    recipe = data["Solutions"][0]
    data["Solutions"] = []
    for index, exact in enumerate(order):
        sol = copy.deepcopy(recipe)
        sol["SolutionIndex"] = index
        if exact:
            sol.update(AssertSizeEqual={0: 320, 1: 64}, BespokeNarrowing=True)
        data["Solutions"].append(sol)
    solutions = solutions_from_logic(write_logic(tmp_path, data))
    assert len(solutions) == 2 and all(s["Valid"] for s in solutions)
    kernels = generateKernelObjectsFromSolutions(solutions)
    groups = groupKernelRecords(kernels, False)
    assert len(groups) == 2, "general and exact-size code paths merged in one compile group"
    assert len({getKernelNameMin(g.representative, False) for g in groups}) == 2
    _, emitted = derive_emit(tmp_path, data)
    assert len({name for name, _, _ in emitted}) == 2, "distinct kernels share an artifact name"
    assert sorted(bool(edge_dispatch(source)) for _, source, _ in emitted) == [False, True]


def test_exact_batch_index_does_not_prove_free_or_sum_tiles(tmp_path):
    data = document(160)
    data["Solutions"][0].update(AssertSizeEqual={2: 384}, BespokeNarrowing=True)
    solutions, emitted = derive_emit(tmp_path, data)
    assert solutions[0]["NoTailLoop"] is False
    assert edge_dispatch(emitted[0][1]), "batch exact size was misread as a free dimension"


@pytest.mark.parametrize("dim", [0, 1])
def test_packed_free_dimensions_keep_edge_checks(dim):
    from types import SimpleNamespace
    from gpu_test_helpers import init_rocisa
    from rocisa.code import Label
    from rocisa.container import ContinuousRegister
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    init_rocisa(target="gfx950")
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(asmCaps={"HasSCMPK": True})
    writer.do = {"EdgeWrite": True}
    writer.db = {"ForceEdgeStores": 0}
    kernel = {"AssertFree0ElementMultiple": 1, "AssertFree1ElementMultiple": 1, "BespokeNarrowing": True,
              "AssertSizeEqual": {0: 320, 1: 480, 2: 320, 3: 480},
              "MacroTile0": 160, "MacroTile1": 240,
              "PackedC0IndicesX": [0, 2] if dim == 0 else [0],
              "PackedC1IndicesX": [1, 3] if dim == 1 else [1],
              "ProblemType": {"Index0": 0, "Index1": 1, "IndicesFree": [0, 1, 2, 3],
                              "IndicesSummation": [4], "IndicesBatch": []}}
    source = str(writer.checkIsEdge(kernel, ContinuousRegister(idx=4, size=4),
                                   Label(label="edge_target", comment=""),
                                   240 if dim else 160, isSize1=bool(dim)))
    assert re.search(r"\bs_cbranch_scc1\s+label_edge_target\b", source), "packed exact index lost edge check"


def test_multi_summation_exact_sizes_do_not_prove_even_per_wg_trip():
    from tensilelite.KernelWriterAssembly import evenUnrollTripProof

    kernel = {"DepthU": 192, "AssertSummationElementMultiple": 1, "BespokeNarrowing": True,
              "AssertSizeEqual": {3: 384, 4: 384}, "StreamK": 0,
              "GlobalSplitU": 1, "InternalSupportParams": {"SupportUserGSU": False},
              "ProblemType": {"Index0": 0, "Index1": 1, "IndicesFree": [0, 1],
                              "IndicesBatch": [2], "IndicesSummation": [3, 4], "IndexUnroll": 4}}
    assert evenUnrollTripProof(kernel) is False, "multi-index exact summation gained a scalar trip proof"


def test_unsupported_tensor_contraction_refused_before_codegen(tmp_path):
    from tensilelite.SolutionStructs.Problem import ProblemType

    data = document(160)
    data["ProblemType"].update(OperationType="TensorContraction", NumIndicesC=4,
                               IndexAssignmentsA=[4, 0, 2], IndexAssignmentsB=[4, 1, 3])
    data["Solutions"][0]["AssertSizeEqual"] = {0: 320, 1: 480, 2: 320, 3: 480, 4: 384}
    with pytest.raises(Exception):
        ProblemType(yaml.safe_load(write_logic(tmp_path, data).read_text())["ProblemType"], False)


@pytest.mark.parametrize("width", [160, 240])
def test_stock_generated_kernel_names_unchanged(tmp_path, width):
    import json
    from codegen_harness import solutions_from_logic
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
    from tensilelite.SolutionStructs.Naming import getKernelNameMin, getKernelFileBase

    kernel, = generateKernelObjectsFromSolutions(solutions_from_logic(write_logic(tmp_path, document(width))))
    expected = json.loads((DATA / "stock_names.json").read_text())[str(width)]
    assert getKernelNameMin(kernel, False) == expected["symbol"], "stock generated symbol changed"
    assert getKernelFileBase(False, kernel) == expected["artifact"], "stock artifact name changed"


@pytest.mark.parametrize("sizes", [{2: 384}, {0: 321, 1: 65, 3: 193}])
def test_exact_facts_without_codegen_effect_keep_stock_identity(tmp_path, sizes):
    from codegen_harness import solutions_from_logic
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
    from tensilelite.SolutionStructs.Naming import getKernelNameMin, getKernelFileBase

    data = document(160)
    general, = generateKernelObjectsFromSolutions(solutions_from_logic(write_logic(tmp_path, data, "general")))
    data["Solutions"][0]["AssertSizeEqual"] = sizes
    exact, = generateKernelObjectsFromSolutions(solutions_from_logic(write_logic(tmp_path, data, "exact")))
    assert getKernelNameMin(exact, False) == getKernelNameMin(general, False)
    assert getKernelFileBase(False, exact) == getKernelFileBase(False, general)


@pytest.mark.parametrize("name,sizes", [("wvSpltK_bf16_tn_m1", {1: 1, 2: 1}),
                                       ("wvSpltK_bf16_tn_m2", {1: 2, 2: 1}),
                                       ("wvSpltK_bf16_tn_m4", {2: 1})])
def test_stock_custom_kernel_identity_unchanged(name, sizes):
    from tensilelite.SolutionStructs.Naming import getKernelFileBase

    kernel = {"CustomKernel": {"name": name}, "AssertSizeEqual": sizes}
    assert getKernelFileBase(False, kernel) == name, "stock source-custom artifact renamed"


@pytest.mark.parametrize("order", [(False, True), (True, False)], ids=["general-first", "exact-first"])
def test_stamped_generated_exact_kernel_refused_or_distinct_identity(tmp_path, order):
    """Feature: refuse narrowed stamped generation or name its changed code.

    Break again: return the same CustomKernel.name before naming exact facts.
    The general stamped recipe is a normal-generation positive control.
    """
    from codegen_harness import solutions_from_logic, emit_kernels_from_logic
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions, groupKernelRecords
    from tensilelite.SolutionStructs.Naming import getKernelNameMin, getKernelFileBase
    from tensilelite.CustomKernels import getCustomKernelConfig

    data = document(160)
    recipe = data["Solutions"][0]
    original = emit_kernels_from_logic(write_logic(tmp_path, data, "unstamped_control"), canonical=False)
    assert len(original) == 1 and original[0][2] == 0 and edge_dispatch(original[0][1])
    name, source, _ = original[0]
    (tmp_path / (name + ".s")).write_text(source)
    recipe["CustomKernel"] = getCustomKernelConfig(name, {}, str(tmp_path))["CustomKernel"]
    recipe["CustomKernel"]["generated"] = True
    plain = solutions_from_logic(write_logic(tmp_path, data, "stamped_control"))
    assert len(plain) == 1 and plain[0]["Valid"], "general stamped recipe must derive normally"
    control = emit_kernels_from_logic(write_logic(tmp_path, data, "stamped_control"), canonical=False)
    assert len(control) == 1 and control[0][2] == 0 and edge_dispatch(control[0][1])
    data["Solutions"] = []
    for index, exact in enumerate(order):
        solution = copy.deepcopy(recipe)
        solution["SolutionIndex"] = index
        if exact:
            solution.update(AssertSizeEqual={0: 320, 1: 64}, BespokeNarrowing=True)
        data["Solutions"].append(solution)
    path = write_logic(tmp_path, data, "stamped_pair")
    try:
        solutions = solutions_from_logic(path)
    except RuntimeError:
        return  # Normal generation may refuse code-changing exact facts on a stamped parent.
    assert len(solutions) == 2 and all(s["Valid"] for s in solutions)
    kernels = generateKernelObjectsFromSolutions(solutions)
    groups = groupKernelRecords(kernels, False)
    assert len(groups) == 2, "stamped general/exact kernels merged despite code-changing facts"
    assert len({getKernelNameMin(g.representative, False) for g in groups}) == 2
    assert len({getKernelFileBase(False, g.representative) for g in groups}) == 2
    emitted = emit_kernels_from_logic(path, canonical=False)
    assert len(emitted) == 2 and all(error == 0 for _, _, error in emitted)
    assert sorted(bool(edge_dispatch(source)) for _, source, _ in emitted) == [False, True]


@pytest.mark.parametrize("field,index,size,multiple", [
    ("AssertFree0ElementMultiple", 0, 480, 64),
    ("AssertFree1ElementMultiple", 1, 96, 64),
    ("AssertSummationElementMultiple", 3, 384, 256),
], ids=["af0", "af1", "asem"])
def test_exact_size_contradicting_declared_multiple_refused(tmp_path, field, index, size, multiple):
    """Feature: reject an empty exact/multiple contract at normal generation.

    Break again: derive exact edge/tail facts without checking public multiples.
    """
    from codegen_harness import solutions_from_logic

    data = loop_document(384) if field == "AssertSummationElementMultiple" else document(160)
    solution = data["Solutions"][0]
    solution.pop("AssertSizeEqual", None)
    solution[field] = multiple
    solution["BespokeNarrowing"] = True
    control = solutions_from_logic(write_logic(tmp_path, data, "multiple_control"))
    assert len(control) == 1 and control[0]["Valid"], "declared-multiple recipe must derive normally"
    solution["AssertSizeEqual"] = {index: size}
    with pytest.raises(RuntimeError):
        solutions_from_logic(write_logic(tmp_path, data, "contradiction"))
