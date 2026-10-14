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
from tensilelite.Utilities.regular_gemm import bench_yaml, load_library_metadata

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
def test_rejects_unsupported_scheduling_and_invalid_dimensions(tmp_path, key, value):
    record = solution()
    record["sizeMapping"][key] = value
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError):
        load_library_metadata(tmp_path, target="gfx942").get_problem(17)


@pytest.mark.parametrize(
    "key,value",
    [
        ("mxBlockA", 32),
        ("groupedGemm", True),
        ("sparse", 1),
        ("useE", True),
        ("activationType", "Gelu"),
        ("operationIdentifier", "another contraction"),
        ("aType", "Float4"),
        ("computeType", "Double"),
    ],
)
def test_rejects_unsupported_problem_features(tmp_path, key, value):
    record = solution()
    record["problemType"][key] = value
    write_shard(tmp_path, [record])
    with pytest.raises(ValueError):
        load_library_metadata(tmp_path, target="gfx942").get_problem(17)


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
from tensilelite.Utilities.regular_gemm import bench_yaml, load_library_metadata
p = load_library_metadata(sys.argv[1], target='gfx942').get_problem(17)
assert (p.m, p.n, p.k) == (128, 96, 256)
assert 'solution_index: 17' in bench_yaml([p])
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
