# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidate selection and client contracts, without a GPU or generated kernels."""

import zlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

import msgpack
from tensilelite.Utilities.gemm_candidates.candidates import GemmShape, candidate_gemms
from tensilelite.Utilities.gemm_candidates.clients import bench_yaml, tensile_ini
from tensilelite.Utilities.gemm_candidates.metadata import load_library_metadata

pytestmark = pytest.mark.unit


def solution(index=17, *, predicates=(), problem=None, mapping=None):
    """Small packaged record with ordinary NN/F32 GEMM defaults."""
    return {
        "index": index,
        "name": "solution-name",
        "kernelName": "kernel-name",
        "problemType": {
            "operationIdentifier": "Contraction_l_Ailk_Bljk_Cijk_Dijk",
            "transA": False,
            "transB": False,
            "aType": "Float",
            "bType": "Float",
            "cType": "Float",
            "dType": "Float",
            "computeType": "Float",
            **(problem or {}),
        },
        "problemPredicate": {"type": "And", "value": list(predicates)},
        "sizeMapping": {
            "macroTile": [128, 96, 1],
            "depthU": 64,
            "globalSplitU": 1,
            **(mapping or {}),
        },
    }


def write_shard(directory, records, name="TensileLibrary_test_gfx942.dat", *, compressed=False):
    path = directory / (name + (".zlib" if compressed else ""))
    data = msgpack.packb({"solutions": records, "library": {"type": "Matching", "table": []}})
    path.write_bytes(zlib.compress(data) if compressed else data)
    return path


@pytest.fixture
def library(tmp_path):
    def load(*records):
        write_shard(tmp_path, records or [solution()])
        return load_library_metadata(tmp_path, target="gfx942")

    return load


def test_compressed_shards_aliases_and_architecture_objects(tmp_path):
    directory = tmp_path / "gfx942"
    directory.mkdir()
    # The compressed copy is authoritative, including when the plain copy is stale.
    write_shard(directory, [solution(999)])
    shard = write_shard(directory, [solution(17), solution(9)], compressed=True)
    for name in [
        "TensileLibrary_test_gfx942.co",
        "Kernels-gfx942.hsaco",
        "Kernels-gfx950.hsaco",
        "helper.hsaco",
    ]:
        (directory / name).touch()
    loaded = load_library_metadata(tmp_path, target="gfx942")
    assert loaded.indices == (9, 17)
    a, b = (candidate_gemms(loaded, i)[0] for i in loaded.indices)
    assert a.solution_name == b.solution_name
    assert a.kernel_name == b.kernel_name
    assert a.solution_index != b.solution_index
    assert a.metadata_file == shard
    assert {p.name for p in a.code_objects} == {
        "TensileLibrary_test_gfx942.co",
        "Kernels-gfx942.hsaco",
        "helper.hsaco",
    }
    subset = load_library_metadata(directory, target="gfx942", solution_indices=[17])
    assert subset.indices == (17,)
    with pytest.raises(KeyError):
        candidate_gemms(subset, 9)
    with pytest.raises(KeyError, match="404"):
        load_library_metadata(directory, target="gfx942", solution_indices=[404])


def test_skips_heuristic_tables_without_decoding(tmp_path):
    # Invalid UTF-8 inside a table would fail if the loader decoded that table.
    pack = msgpack.packb
    data = b"\x82" + pack("library") + b"\x91\xa1\xff" + pack("solutions") + pack([solution()])
    (tmp_path / "TensileLibrary_gfx942.dat").write_bytes(data)
    loaded = load_library_metadata(tmp_path, target="gfx942")
    assert candidate_gemms(loaded, 17)[0].shape == GemmShape(1, 1, 1)


@pytest.mark.parametrize("compressed", [False, True])
def test_rejects_truncated_and_trailing_metadata(tmp_path, compressed):
    path = write_shard(tmp_path, [solution()], compressed=compressed)
    data = path.read_bytes()
    for corrupt in (data[:-1], data + b"\x00"):
        path.write_bytes(corrupt)
        with pytest.raises(ValueError):
            load_library_metadata(tmp_path, target="gfx942")


def test_duplicate_indices_are_not_silently_overwritten(tmp_path):
    write_shard(tmp_path, [solution()])
    write_shard(tmp_path, [solution()], "TensileLibrary_other_gfx942.dat")
    with pytest.raises(ValueError, match="duplicate solution index 17"):
        load_library_metadata(tmp_path, target="gfx942", solution_indices=[9])


def test_generation_reuses_metadata_and_is_independent_of_call_order(library, monkeypatch):
    loaded = library(solution(17), solution(9, mapping={"globalSplitU": 3}))
    expected = {i: candidate_gemms(loaded, i) for i in loaded.indices}
    assert expected[17][1].shape.k == 64
    assert expected[9][1].shape.k == 192

    def no_reads(*args, **kwargs):
        pytest.fail("candidate generation reread the library")

    monkeypatch.setattr(Path, "read_bytes", no_reads)
    for i in reversed(loaded.indices):
        assert candidate_gemms(loaded, i) == expected[i]
        assert candidate_gemms(loaded, i, max_cases=4) == expected[i][:4]
    with pytest.raises(FrozenInstanceError):
        expected[17][0].shape.m = 100


def test_fixed_split_boundaries_and_partial_tiles(library):
    # Reduced from a native-validated gfx942 solution: M >= 4 is not M % 4 == 0.
    loaded = library(
        solution(
            predicates=[
                {"type": "LeadingFree0SizesGreaterOrEqual", "value": 4},
                {"type": "GlobalSplitUCheckMinK", "value": [32, 19]},
            ],
            mapping={"globalSplitU": 19},
        )
    )
    cases = {case.id: case.shape for case in candidate_gemms(loaded, 17, max_cases=20)}
    assert cases["minimum"] == GemmShape(4, 1, 608)
    assert [cases[f"k_{i}"].k for i in (1, 2, 3)] == [1216, 2432, 3648]
    assert cases["k_long"].k == 4864  # 8x exceeds the default dimension cap; use 4x.
    assert cases["k_tail"].k == 2433
    assert cases["split_remainder"].k == 2496
    assert cases["tile"] == GemmShape(128, 96, 2432)
    assert cases["mn_edge"] == GemmShape(129, 97, 2433)
    assert cases["multi_tile"] == GemmShape(256, 288, 2432)
    assert cases["batch"].batch_count == 2


@pytest.mark.parametrize("split, strategy", [(-1, "None"), (0, "StreamK"), (19, "StreamK")])
def test_automatic_or_stream_k_does_not_claim_fixed_split_work(library, split, strategy):
    loaded = library(solution(mapping={"globalSplitU": split, "tileProcessingStrategy": strategy}))
    cases = {case.id: case for case in candidate_gemms(loaded, 17, max_cases=20)}
    assert cases["k_1"].shape.k == 64
    assert cases["k_2"].shape.k == 128
    assert "split_remainder" not in cases
    assert all("configured GSU" not in c.reason for c in cases.values())


def test_large_minimum_still_gets_distinct_k_lengths(library):
    loaded = library(solution(predicates=[{"type": "SizeGreaterThan", "index": 3, "value": 1023}]))
    cases = {case.id: case.shape.k for case in candidate_gemms(loaded, 17)}
    assert cases["minimum"] == 1024
    assert cases["k_2"] == 1088
    assert cases["k_3"] == 1152
    assert cases["k_long"] == 8192


@pytest.mark.parametrize("multiple, remainder", [(128, 1664), (192, None)])
def test_split_remainder_respects_alignment_above_a_large_minimum(library, multiple, remainder):
    loaded = library(
        solution(
            predicates=[
                {"type": "SizeGreaterThan", "index": 3, "value": 1535},
                {"type": "BoundSizeMultiple", "value": multiple},
            ],
            mapping={"globalSplitU": 3},
        )
    )
    cases = {c.id: c.shape.k for c in candidate_gemms(loaded, 17, max_cases=20)}
    # 1536 / 64 = 24 blocks divide evenly across three split groups. A legal
    # increment of 128 permits 26 blocks; increments of 192 always divide evenly.
    assert cases.get("split_remainder") == remainder


def test_intersected_bounds_equalities_and_alignment(library):
    loaded = library(
        solution(
            predicates=[
                {
                    "type": "And",
                    "value": [
                        {"type": "Free0SizeMultiple", "index": 0, "value": 4},
                        {"type": "SizeMultiple", "index": 0, "value": 6},
                        {"type": "SizeInRange", "index": 0, "value": {"min": 13, "max": 37}},
                    ],
                },
                {"type": "SizeGreaterThan", "index": 1, "value": 4},
                {"type": "SizeLessThan", "index": 1, "value": 7},
                {"type": "BatchSizeEqual", "index": 0, "value": 3},
                {"type": "BoundSizeMultiple", "index": -1, "value": 128},
            ]
        )
    )
    cases = candidate_gemms(loaded, 17, max_cases=20)
    assert cases[0].shape == GemmShape(24, 5, 128, 3)
    assert all(c.shape.m % 12 == 0 and 13 <= c.shape.m < 37 for c in cases)
    assert all(4 < c.shape.n < 7 and c.shape.batch_count == 3 for c in cases)
    assert all(c.shape.k % 128 == 0 for c in cases)
    assert "k_tail" not in {c.id for c in cases}
    assert len({c.shape for c in cases}) == len(cases)


def test_disjunctions_and_negations_do_not_become_required_bounds(library):
    loaded = library(
        solution(
            predicates=[
                {
                    "type": "Or",
                    "value": [
                        {"type": "SizeEqual", "index": 0, "value": 5},
                        {"type": "SizeEqual", "index": 0, "value": 7},
                    ],
                },
                {"type": "Not", "value": {"type": "SizeEqual", "index": 3, "value": 1}},
            ]
        )
    )
    # Only independent necessary bounds are extracted; native support remains unproven.
    assert candidate_gemms(loaded, 17)[0].shape == GemmShape(1, 1, 1)


def test_caps_can_reduce_the_list_or_make_it_empty(library):
    loaded = library()
    assert [c.shape for c in candidate_gemms(loaded, 17, max_dimension=1)] == [GemmShape(1, 1, 1)]
    assert [c.shape for c in candidate_gemms(loaded, 17, max_data_bytes=16)] == [GemmShape(1, 1, 1)]
    with pytest.raises(ValueError, match="data limit"):
        candidate_gemms(loaded, 17, max_data_bytes=15)
    impossible = library(solution(predicates=[{"type": "SizeEqual", "index": 0, "value": 16}]))
    with pytest.raises(ValueError, match="dimension limit"):
        candidate_gemms(impossible, 17, max_dimension=15)


@pytest.mark.parametrize(
    "options", [{"max_cases": 0}, {"max_dimension": -1}, {"max_data_bytes": 1.5}]
)
def test_invalid_limits(library, options):
    with pytest.raises(ValueError, match="positive integers"):
        candidate_gemms(library(), 17, **options)


@pytest.mark.parametrize(
    "feature, value",
    [
        ("sparse", 1),
        ("groupedGemm", True),
        ("useE", True),
        ("useBias", 2),
        ("useScaleAlphaVec", 3),
        ("aType", "Float4"),
    ],
)
def test_unsupported_profiles_fail_explicitly(library, feature, value):
    with pytest.raises(ValueError, match="unsupported"):
        candidate_gemms(library(solution(problem={feature: value})), 17)


@pytest.mark.parametrize(
    "trans_a, trans_b, lda, ldb",
    [(False, False, 5, 11), (True, False, 11, 11), (False, True, 5, 7), (True, True, 11, 7)],
)
def test_both_adapters_preserve_shapes_types_and_packed_layouts(
    library, trans_a, trans_b, lda, ldb
):
    a, b = ("lik" if trans_a else "ilk"), ("jlk" if trans_b else "ljk")
    loaded = library(
        solution(
            problem={
                "transA": trans_a,
                "transB": trans_b,
                "operationIdentifier": f"Contraction_l_A{a}_B{b}_Cijk_Dijk",
                "aType": "Half",
                "bType": "Half",
                "cType": "Half",
                "dType": "Half",
                "highPrecisionAccumulate": True,
                "useBias": 1,
                "useScaleAB": "Scalar",
                "useScaleCD": True,
                "useScaleAlphaVec": 1,
                "activationType": "Hipblaslt_all",
            }
        )
    )
    case = replace(candidate_gemms(loaded, 17)[0], shape=GemmShape(5, 7, 11, 3))
    (row,) = yaml.safe_load(bench_yaml([case]))
    assert (row["M"], row["N"], row["K"], row["batch_count"]) == (5, 7, 11, 3)
    assert (row["lda"], row["ldb"], row["ldc"], row["ldd"]) == (lda, ldb, 5, 5)
    assert [row[f"stride_{t}"] for t in "abcd"] == [55, 77, 35, 35]
    assert row["solution_index"] == 17 and row["algo_method"] == 2
    assert row["a_type"] == "f16_r" and row["compute_type"] == "c_f32_r"
    assert row["bias_vector"] and row["scaleAlpha_vector"]
    assert row["scaleA"] == "Scalar" and row["scaleD"]
    assert row["alpha"] == 1 and row["beta"] == 0 and row["activation_type"] == "none"
    assert row["norm_check_assert"] and row["allclose_check"] == 1
    ini = dict(line.split("=", 1) for line in tensile_ini([case]).splitlines())
    assert ini["problem-size"] == "5,7,3,11"
    assert ini["a-strides"] == f"1,{lda},55"
    assert ini["b-strides"] == f"1,{ldb},77"
    assert ini["solution-start-idx"] == "17" and ini["num-solutions"] == "1"
    assert ini["a-type"] == "Half" and ini["alpha-type"] == "Float"
    assert ini["use-bias"] == "1" and ini["use-scaleAlphaVec"] == "1"
    assert ini["init-alpha"] == "One" and ini["init-beta"] == "Zero"
    assert int(ini["init-seed"]) > 0  # Zero selects the client's time-based seed.
    assert ini["num-elements-to-validate"] == "-1"


@pytest.mark.parametrize(
    "dtype, compute, bench_compute",
    [
        ("Int8", "Int32", "c_i32_r"),
        ("Float8_fnuz", "Float", "c_f32_r"),
        ("BFloat8", "Float", "c_f32_r"),
    ],
)
def test_integer_and_fp8_client_types(library, dtype, compute, bench_compute):
    loaded = library(
        solution(
            problem={
                "aType": dtype,
                "bType": dtype,
                "cType": compute,
                "dType": compute,
                "computeType": compute,
            }
        )
    )
    cases = candidate_gemms(loaded, 17, max_cases=1)
    (row,) = yaml.safe_load(bench_yaml(cases))
    assert row["compute_type"] == bench_compute
    assert ("compute_input_typeA" in row) == (compute != "Int32")
    assert f"compute-input-type-A={dtype}\n" in tensile_ini(cases)


def test_batching_preserves_solution_associations(library):
    loaded = library(solution(17), solution(9))
    a, b = [candidate_gemms(loaded, i, max_cases=4) for i in (17, 9)]
    assert [r["solution_index"] for r in yaml.safe_load(bench_yaml(a + b))] == [17] * 4 + [9] * 4
    ini = tensile_ini(a)
    assert ini.count("problem-size=") == len(a)
    assert ini.count("a-strides=") == len(a)
    assert ini.count("solution-start-idx=") == 1
    with pytest.raises(ValueError, match="only one solution"):
        tensile_ini(a + b)
    foreign = replace(a[0], metadata_file=a[0].metadata_file.parent / "other" / "library.dat")
    with pytest.raises(ValueError, match="same library directory"):
        bench_yaml([a[0], foreign])
    with pytest.raises(ValueError, match="at least one case"):
        tensile_ini([])
