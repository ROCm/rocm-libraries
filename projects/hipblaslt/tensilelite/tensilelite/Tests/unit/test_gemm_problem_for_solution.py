# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regular GEMM descriptors and explicit-index client input, without a GPU."""

import os
import subprocess
import sys
import zlib
from dataclasses import replace
from itertools import product

import msgpack
import pytest
import yaml
from tensilelite.Utilities.gemm_problem_bench import UnsupportedBenchProblem, bench_yaml
from tensilelite.Utilities.gemm_problem_for_solution import (
    UnsupportedSolution,
    load_library_metadata,
)

pytestmark = pytest.mark.unit


def solution(index=17, *, trans_a=False, trans_b=False, split=1):
    a, b = ("lik" if trans_a else "ilk"), ("jlk" if trans_b else "ljk")
    return {
        "index": index,
        "name": "solution-name",
        "kernelName": "kernel-name",
        "problemType": {
            "operationIdentifier": f"Contraction_l_A{a}_B{b}_Cijk_Dijk",
            "transA": trans_a,
            "transB": trans_b,
            "aType": "Float",
            "bType": "Float",
            "cType": "Float",
            "dType": "Float",
            "computeType": "Float",
        },
        "sizeMapping": {"macroTile": [128, 96, 1], "depthU": 32, "globalSplitU": split},
    }


def write_shard(directory, records, name="TensileLibrary_test_gfx942.dat"):
    path = directory / name
    path.write_bytes(msgpack.packb({"solutions": records, "library": {"type": "Matching"}}))
    return path


@pytest.mark.parametrize(
    "trans_a,trans_b,split", list(product((False, True), (False, True), (1, 4, 8)))
)
def test_full_tile_and_eight_blocks_per_split(tmp_path, trans_a, trans_b, split):
    write_shard(tmp_path, [solution(trans_a=trans_a, trans_b=trans_b, split=split)])
    library = load_library_metadata(tmp_path, target="gfx942")
    problem = library.get_problem(17)
    assert (problem.m, problem.n, problem.k) == (128, 96, 256 * split)
    (row,) = yaml.safe_load(bench_yaml([problem]))
    assert [row[key] for key in ("lda", "ldb", "ldc", "ldd")] == [
        problem.k if trans_a else 128,
        96 if trans_b else problem.k,
        128,
        128,
    ]
    assert [row[key] for key in ("stride_a", "stride_b", "stride_c", "stride_d")] == [
        128 * problem.k,
        96 * problem.k,
        12288,
        12288,
    ]
    assert (row["alpha"], row["beta"], row["batch_count"]) == (1, 0, 1)
    assert (row["algo_method"], row["solution_index"], row["requested_solution_num"]) == (2, 17, 1)
    assert (row["cold_iters"], row["iters"], row["adaptive"]) == (0, 0, False)
    assert row["norm_check"] and row["norm_check_assert"] and row["allclose_check"]
    assert row["activation_type"] == "none" and row["bias_vector"] is False
    # Generation reuses the loaded records, independent of the files afterward.
    for path in tmp_path.iterdir():
        path.unlink()
    assert library.get_problem(17) == problem


@pytest.mark.parametrize("library_format", ["msgpack", "msgpack-indexed"])
@pytest.mark.parametrize("compressed", [False, True])
def test_formats_selection_and_identity(tmp_path, library_format, compressed):
    from tensilelite.LibraryIO import write

    write(
        str(tmp_path / "TensileLibrary_test_gfx942"),
        {"solutions": [solution(9), solution(17)], "library": {"type": "Matching"}},
        library_format,
    )
    if not compressed:
        path = tmp_path / "TensileLibrary_test_gfx942.dat.zlib"
        path.with_suffix("").write_bytes(zlib.decompress(path.read_bytes()))
        path.unlink()
    library = load_library_metadata(tmp_path, target="gfx942", solution_indices=[17])
    assert library.indices == (17,)
    p = library.get_problem(17)
    assert (p.solution_name, p.kernel_name, p.library_directory, p.target) == (
        "solution-name",
        "kernel-name",
        tmp_path,
        "gfx942",
    )
    with pytest.raises(KeyError):
        library.get_problem(9)
    with pytest.raises(ValueError, match="same library"):
        bench_yaml([p, replace(p, target="gfx950")])


@pytest.mark.parametrize(
    "key,value",
    [
        ("tileProcessingStrategy", "StreamK"),
        ("tileProcessingStrategy", "Persistent"),
        ("globalSplitU", -1),
        ("adaptiveGemmGSUA", 1),
        ("globalSplitU", 0),
        ("depthU", 0),
        ("depthU", True),
    ],
)
def test_unsupported_scheduling_is_distinct_from_invalid_metadata(tmp_path, key, value):
    record = solution()
    record["sizeMapping"][key] = value
    write_shard(tmp_path, [record])
    if key in {"tileProcessingStrategy", "adaptiveGemmGSUA"} or value == -1:
        with pytest.raises(UnsupportedSolution) as caught:
            load_library_metadata(tmp_path, target="gfx942").get_problem(17)
        assert caught.value.solution_index == 17
        assert caught.value.reasons
    else:
        with pytest.raises(ValueError):
            load_library_metadata(tmp_path, target="gfx942").get_problem(17)


@pytest.mark.parametrize("strategy", ["StreamK", "Persistent"])
def test_zero_split_for_deferred_scheduling_is_unsupported(tmp_path, strategy):
    # Packaged persistent solutions use zero to disable fixed split-K.
    record = solution(split=0)
    record["sizeMapping"]["tileProcessingStrategy"] = strategy
    write_shard(tmp_path, [record])
    with pytest.raises(UnsupportedSolution) as caught:
        load_library_metadata(tmp_path, target="gfx942").get_problem(17)
    assert caught.value.reasons == (f"tileProcessingStrategy={strategy}",)


@pytest.mark.parametrize(
    "key,value",
    [
        ("mxBlockA", 32),
        ("groupedGemm", True),
        ("sparse", 1),
        ("useE", True),
        ("activationType", "Gelu"),
        ("operationIdentifier", "another contraction"),
    ],
)
def test_rejects_unsupported_problem_features(tmp_path, key, value):
    record = solution()
    record["problemType"][key] = value
    write_shard(tmp_path, [record])
    with pytest.raises(UnsupportedSolution) as caught:
        load_library_metadata(tmp_path, target="gfx942").get_problem(17)
    assert caught.value.solution_index == 17
    assert caught.value.reasons


def test_compressed_precedence_missing_and_duplicate_indices(tmp_path):
    directory = tmp_path / "gfx942"
    directory.mkdir()
    path = write_shard(directory, [solution(17)])
    path.with_suffix(".dat.zlib").write_bytes(zlib.compress(path.read_bytes()))
    write_shard(directory, [solution(9)])
    assert load_library_metadata(tmp_path, target="gfx942").indices == (17,)
    with pytest.raises(KeyError, match="404"):
        load_library_metadata(tmp_path, target="gfx942", solution_indices=[404])
    write_shard(directory, [solution(17)], "TensileLibrary_other_gfx942.dat")
    with pytest.raises(ValueError, match="duplicate solution index 17"):
        load_library_metadata(tmp_path, target="gfx942", solution_indices=[404])


@pytest.mark.parametrize("indices", [[17, 17.0], [1, True], [-1]])
def test_validates_requested_indices_before_deduplication(tmp_path, indices):
    with pytest.raises(ValueError, match="nonnegative integers"):
        load_library_metadata(tmp_path, target="gfx942", solution_indices=indices)


@pytest.mark.parametrize(
    "dtype,bench_type",
    [
        ("Half", "f16_r"),
        ("BFloat16", "bf16_r"),
        ("Float8", "f8_r"),
        ("BFloat8", "bf8_r"),
        ("Float8_fnuz", "f8_fnuz_r"),
        ("BFloat8_fnuz", "bf8_fnuz_r"),
        ("Int8", "i8_r"),
    ],
)
def test_storage_and_compute_types(tmp_path, dtype, bench_type):
    record = solution()
    record["problemType"].update(aType=dtype, bType=dtype)
    if dtype == "Int8":
        record["problemType"].update(cType="Int32", dType="Int32", computeType="Int32")
    write_shard(tmp_path, [record])
    (row,) = yaml.safe_load(
        bench_yaml([load_library_metadata(tmp_path, target="gfx942").get_problem(17)])
    )
    assert row["a_type"] == row["b_type"] == bench_type
    assert row["compute_type"] == ("c_i32_r" if dtype == "Int8" else "c_f32_r")
    assert row["d_type"] == ("i32_r" if dtype == "Int8" else "f32_r")


def test_generation_does_not_import_generator_or_rocisa(tmp_path):
    write_shard(tmp_path, [solution()])
    script = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('rocisa', 'tensilelite.Common', 'tensilelite.SolutionStructs')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from tensilelite.Utilities.gemm_problem_bench import UnsupportedBenchProblem, bench_yaml
from tensilelite.Utilities.gemm_problem_for_solution import UnsupportedSolution, load_library_metadata
p = load_library_metadata(sys.argv[1], target='gfx942').get_problem(17)
assert (p.m, p.n, p.k) == (128, 96, 256)
assert 'solution_index: 17' in bench_yaml([p])
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )


@pytest.mark.parametrize("blocks", [1, 3, 8, 17])
def test_reduction_work_is_configurable(tmp_path, blocks):
    write_shard(tmp_path, [solution(split=4)])
    library = load_library_metadata(tmp_path, target="gfx942")
    assert library.get_problem(17, reduction_blocks=blocks).k == blocks * 32 * 4


@pytest.mark.parametrize("blocks", [0, -1, True, 8.0])
def test_invalid_reduction_blocks_are_errors(tmp_path, blocks):
    write_shard(tmp_path, [solution()])
    with pytest.raises(ValueError, match="reduction_blocks"):
        load_library_metadata(tmp_path, target="gfx942").get_problem(17, reduction_blocks=blocks)


@pytest.mark.parametrize("target", ["gfx90a", "gfx942", "gfx950", "gfx1250-strict"])
@pytest.mark.parametrize(
    "dtype,bench_name,compute_name",
    [
        ("Double", "f64_r", "c_f64_r"),
        ("ComplexFloat", "f32_c", "c_f32_r"),
        ("ComplexDouble", "f64_c", "c_f64_r"),
    ],
)
def test_ordinary_types_and_architectures(tmp_path, target, dtype, bench_name, compute_name):
    record = solution()
    record["problemType"].update({f"{key}Type": dtype for key in "abcd"})
    record["problemType"]["computeType"] = dtype
    write_shard(tmp_path, [record], f"TensileLibrary_test_{target}.dat")
    problem = load_library_metadata(tmp_path, target=target).get_problem(17)
    assert problem.a_type == problem.compute_type == dtype
    (row,) = yaml.safe_load(bench_yaml([problem]))
    assert row["a_type"] == row["d_type"] == row["scale_type"] == bench_name
    assert row["compute_type"] == compute_name
    # Bench derives non-FP32 compute input types from its compute enum.
    assert "compute_input_typeA" not in row


@pytest.mark.parametrize("dtype", ["Int64", "Int8x4", "Half"])
def test_descriptor_support_is_independent_of_bench_adapter(tmp_path, dtype):
    record = solution()
    record["problemType"].update(aType=dtype, bType=dtype, computeType=dtype)
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").get_problem(17)
    assert problem.a_type == dtype
    with pytest.raises(UnsupportedBenchProblem) as caught:
        bench_yaml([problem])
    assert caught.value.solution_index == 17


@pytest.mark.parametrize(
    "dtype,bench_name", [("Float4", "f4_r"), ("Float6", "f6_r"), ("BFloat6", "bf6_r")]
)
def test_non_mx_packed_types_are_preserved(tmp_path, dtype, bench_name):
    record = solution()
    record["problemType"].update(aType=dtype, bType=dtype)
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").get_problem(17)
    (row,) = yaml.safe_load(bench_yaml([problem]))
    assert row["a_type"] == row["compute_input_typeA"] == bench_name


def test_complex_conjugate_transpose(tmp_path):
    record = solution(trans_a=True)
    record["problemType"].update({f"{key}Type": "ComplexFloat" for key in "abcd"})
    record["problemType"].update(
        computeType="ComplexFloat", operationIdentifier="Contraction_l_AlikC_Bljk_Cijk_Dijk"
    )
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").get_problem(17)
    assert problem.conjugate_a and problem.trans_a
    (row,) = yaml.safe_load(bench_yaml([problem]))
    assert row["transA"] == "C"


def test_sweep_can_continue_only_for_explicit_unsupported_cases(tmp_path):
    unsupported = solution(9)
    unsupported["problemType"].update(mxBlockA=32, useE=True)
    write_shard(tmp_path, [unsupported, solution(17)])
    library = load_library_metadata(tmp_path, target="gfx942")
    executed, gaps = [], {}
    for index in library.indices:
        try:
            executed.append(library.get_problem(index))
        except UnsupportedSolution as error:
            gaps[error.solution_index] = error.reasons
    assert [p.solution_index for p in executed] == [17]
    assert set(gaps[9]) == {"mxBlockA", "useE"}
    # A broken record must escape the sweep's UnsupportedSolution handler.
    broken = solution(0)
    del broken["sizeMapping"]["depthU"]
    write_shard(tmp_path, [broken])
    with pytest.raises(KeyError, match="depthU"):
        load_library_metadata(tmp_path, target="gfx942").get_problem(0)


def test_unknown_metadata_type_is_an_error(tmp_path):
    record = solution()
    record["problemType"]["aType"] = "typo"
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError, match="unknown metadata datatype"):
        load_library_metadata(tmp_path, target="gfx942").get_problem(17)


@pytest.mark.parametrize("per_operand", [False, True])
@pytest.mark.parametrize(
    "pair,a_type,b_type",
    [
        ("Float8BFloat8", "Float8", "BFloat8"),
        ("BFloat8Float8", "BFloat8", "Float8"),
        ("Float8BFloat8_fnuz", "Float8_fnuz", "BFloat8_fnuz"),
        ("BFloat8Float8_fnuz", "BFloat8_fnuz", "Float8_fnuz"),
    ],
)
def test_mixed_fp8_compute_inputs(tmp_path, per_operand, pair, a_type, b_type):
    record = solution()
    problem_type = record["problemType"]
    problem_type.update(aType=a_type, bType=b_type)
    if per_operand:
        problem_type.update(computeInputTypeA=pair, computeInputTypeB=pair)
    else:
        problem_type["computeInputType"] = pair
    write_shard(tmp_path, [record])
    problem = load_library_metadata(tmp_path, target="gfx942").get_problem(17)
    assert (problem.compute_input_a, problem.compute_input_b) == (a_type, b_type)
    (row,) = yaml.safe_load(bench_yaml([problem]))
    assert row["compute_input_typeA"] == row["a_type"]
    assert row["compute_input_typeB"] == row["b_type"]
