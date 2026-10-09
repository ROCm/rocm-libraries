# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidate selection and client contracts, without a GPU or generated kernels."""

import zlib
from dataclasses import FrozenInstanceError, replace
from itertools import product
from pathlib import Path

import msgpack
import pytest
import yaml
from tensilelite.Utilities.gemm_candidates.candidates import candidate_gemms
from tensilelite.Utilities.gemm_candidates.clients import bench_yaml, tensile_configs, tensile_ini
from tensilelite.Utilities.gemm_candidates.metadata import load_library_metadata
from tensilelite.Utilities.gemm_candidates.models import GemmShape

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


@pytest.mark.parametrize("library_format", ["msgpack", "msgpack-indexed"])
@pytest.mark.parametrize("compressed", [False, True])
def test_loads_native_library_formats(tmp_path, library_format, compressed):
    """The library writer's two binary formats must both reach the requested solution."""
    from tensilelite.LibraryIO import write

    record = solution()
    write(
        str(tmp_path / "TensileLibrary_test_gfx942"),
        {"solutions": [record], "library": {"type": "Matching", "table": []}},
        library_format,
    )
    if not compressed:
        path = tmp_path / "TensileLibrary_test_gfx942.dat.zlib"
        path.with_suffix("").write_bytes(zlib.decompress(path.read_bytes()))
        path.unlink()
    for requested in (None, [17]):
        library = load_library_metadata(tmp_path, target="gfx942", solution_indices=requested)
        assert library.indices == (17,)
        assert candidate_gemms(library, 17)[0].solution_name == record["name"]


@pytest.mark.parametrize(
    "data_type",
    [
        "Float",
        "Half",
        "BFloat16",
        "Float8",
        "BFloat8",
        "Float8_fnuz",
        "BFloat8_fnuz",
        "Float4",
        "Float6",
        "BFloat6",
        "Int8",
        "Int32",
    ],
)
def test_storage_budget_matches_generator_type_registry(data_type):
    """The lightweight client type subset must agree with the generator's byte sizes."""
    from tensilelite.Common.DataType import DataType
    from tensilelite.Utilities.gemm_candidates import GemmProfile

    record = solution(problem={"aType": data_type, "bType": data_type})
    profile = GemmProfile.from_problem_type(record["problemType"])
    shape = GemmShape(m=3, n=5, k=32, batch_count=2)
    expected = 2 * ((3 * 32 + 5 * 32) * DataType(data_type).numBytes() + 2 * 3 * 5 * 4)
    assert profile.data_bytes(shape) == expected


@pytest.fixture
def library(tmp_path):
    def load(*records, target="gfx942"):
        write_shard(tmp_path, records or [solution()], name=f"TensileLibrary_test_{target}.dat")
        return load_library_metadata(tmp_path, target=target)

    return load


@pytest.mark.parametrize("target", ["gfx942", "gfx1250", "gfx1250-strict"])
def test_compressed_shards_aliases_and_architecture_objects(tmp_path, target):
    directory = tmp_path / target
    directory.mkdir()
    stem = f"TensileLibrary_test_{target}"
    # The compressed copy is authoritative, including when the plain copy is stale.
    write_shard(directory, [solution(999)], name=f"{stem}.dat")
    shard = write_shard(directory, [solution(17), solution(9)], name=f"{stem}.dat", compressed=True)
    # gfx1250 and gfx1250-strict share an ISA but load different code objects.
    for name in [f"{stem}.co", "helper.hsaco"] + [
        f"Kernels-{arch}.hsaco" for arch in ("gfx942", "gfx950", "gfx1250", "gfx1250-strict")
    ]:
        (directory / name).touch()
    loaded = load_library_metadata(tmp_path, target=target)
    assert loaded.indices == (9, 17)
    a, b = (candidate_gemms(loaded, i)[0] for i in loaded.indices)
    assert a.solution_name == b.solution_name
    assert a.kernel_name == b.kernel_name
    assert a.solution_index != b.solution_index
    assert a.metadata_file == shard
    assert {p.name for p in a.code_objects} == {
        f"{stem}.co",
        f"Kernels-{target}.hsaco",
        "helper.hsaco",
    }
    subset = load_library_metadata(directory, target=target, solution_indices=[17])
    assert subset.indices == (17,)
    with pytest.raises(KeyError):
        candidate_gemms(subset, 9)
    with pytest.raises(KeyError, match="404"):
        load_library_metadata(directory, target=target, solution_indices=[404])


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


def test_requested_count_extends_the_boundary_menu_and_preserves_prefixes(library):
    loaded = library()
    cases = loaded.get_problems(17, 100)
    assert len(cases) == len({c.shape for c in cases}) == 100
    for count in (1, 4, 10, 25):
        assert loaded.get_problems(17, count) == cases[:count]
    # The extension includes individual output edges and reductions on both
    # sides of DepthU, not only more repetitions of the first ten shapes.
    shapes = {c.shape for c in cases}
    assert GemmShape(128, 96, 63) in shapes
    assert GemmShape(128, 96, 65) in shapes
    assert GemmShape(129, 96, 128) in shapes
    assert GemmShape(128, 97, 128) in shapes


@pytest.fixture
def runtime_library(library):
    problem = {
        "useBias": 1,
        "biasDataTypeWhiteList": ["Float", "Half"],
        "biasSrcWhiteList": [3],
        "activationType": "Hipblaslt_all",
        "useScaleAB": "Scalar",
        "useScaleCD": True,
        "useScaleAlphaVec": 1,
    }
    predicates = [
        {
            "type": "ActivationEnumWhiteList",
            "value": ["None", "Relu", "Gelu", "Silu", "Clamp", "Sigmoid", "Abs"],
        }
    ]
    return library(*(solution(i, problem=problem, predicates=predicates) for i in (17, 9)))


def test_case_budget_includes_seeded_runtime_variation(runtime_library):
    loaded = runtime_library
    cases = loaded.get_problems(17, 1000, seed=7)
    assert len(cases) == len({case.shape for case in cases}) == 1000
    for count in (1, 4, 10, 100):
        assert loaded.get_problems(17, count, seed=7) == cases[:count]
    different = loaded.get_problems(17, 1000, seed=8)
    assert [case.shape for case in different] == [case.shape for case in cases]
    assert different != cases
    aliases = loaded.get_problems(9, 1000, seed=7)
    assert [replace(case, solution_index=17) for case in aliases] == cases
    assert [(c.alpha, c.beta) for c in cases[:8]] == [(1, 0), (2, 0), (1, 1), (2, 1)] * 2
    # A broad deterministic sample checks the documented probability without
    # requiring a particular RNG sequence or expecting exactly 5 of 10 cases.
    plain = sum(not c.profile.use_bias and c.activation == "None" for c in cases)
    assert 450 < plain < 550
    assert {c.activation for c in cases} == {"None", "Relu", "Gelu", "Silu", "Clamp"}
    assert {c.profile.bias_type for c in cases if c.profile.use_bias} == {"Float", "Half"}
    assert {c.profile.scale_ab for c in cases} == {"", "Scalar"}
    assert {c.profile.scale_cd for c in cases} == {False, True}
    assert {c.profile.scale_alpha_vector for c in cases} == {False, True}


@pytest.mark.parametrize(
    "predicates, pairs",
    [
        ([{"type": "BetaZero"}], {(1, 0), (2, 0)}),
        ([{"type": "BetaOne"}], {(1, 1), (2, 1)}),
        ([{"type": "AlphaValue", "value": "One"}], {(1, 0), (1, 1)}),
        ([{"type": "BetaValue", "value": "One"}], {(1, 1), (2, 1)}),
        ([{"type": "AlphaValue", "value": "Any"}], {(1, 0), (2, 0), (1, 1), (2, 1)}),
    ],
)
def test_runtime_sampling_respects_scalar_restrictions(library, predicates, pairs):
    cases = library(solution(predicates=predicates)).get_problems(17, 10)
    assert {(case.alpha, case.beta) for case in cases} == pairs


def test_runtime_sampling_respects_activation_lists_and_rejects_impossible_scalars(library):
    loaded = library(
        solution(
            problem={"activationType": "All"},
            predicates=[{"type": "ActivationEnumWhiteList", "value": ["Relu", "Abs"]}],
        )
    )
    cases = loaded.get_problems(17, 100)
    assert {case.activation for case in cases} == {"None", "Relu"}
    plain = next(case for case in cases if case.activation == "None")
    assert "activation-type=None\n" in tensile_ini([plain])
    loaded = library(solution(predicates=[{"type": "AlphaValue", "value": "NegativeOne"}]))
    with pytest.raises(ValueError, match="scalar restrictions"):
        loaded.get_problems(17, 10)
    with pytest.raises(ValueError, match="seed must be an integer"):
        library().get_problems(17, 10, seed=1.5)


def test_tensile_groups_run_each_case_once_with_its_own_settings(runtime_library):
    cases = runtime_library.get_problems(17, 10) + runtime_library.get_problems(9, 10)
    expected = {(c.solution_index, c.shape.tensile_sizes): c for c in cases}
    configs = tensile_configs(cases, results_file="report.csv")
    observed, result_paths = set(), set()
    for ini in configs:
        options = dict(line.split("=", 1) for line in ini.splitlines())
        result_paths.add(options["results-file"])
        for line in ini.splitlines():
            if not line.startswith("problem-size="):
                continue
            shape = tuple(map(int, line.split("=", 1)[1].split(",")))
            key = (int(options["solution-start-idx"]), shape)
            case = expected[key]
            assert key not in observed
            observed.add(key)
            assert options["num-solutions"] == "1"
            assert options["init-alpha"] == {1: "One", 2: "Two"}[case.alpha]
            assert options["init-beta"] == {0: "Zero", 1: "One"}[case.beta]
            assert options["init-c"] == ("Random" if case.beta else "Zero")
            assert options["activation-enum-args"] == case.activation
            assert options["use-bias"] == str(int(case.profile.use_bias))
            assert options["use-scaleAB"] == case.profile.scale_ab
    assert observed == expected.keys()
    assert len(result_paths) == len(configs)
    assert tensile_configs([cases[0]]) == [tensile_ini([cases[0]])]


@pytest.mark.parametrize("data_limit", [32, 128, 4096])
def test_short_result_exhausts_the_bounded_domain(library, data_limit):
    loaded = library()
    cases = loaded.get_problems(17, 1000, max_dimension=3, max_data_bytes=data_limit)
    # Independent, small exhaustive oracle. The primary menu alone cannot
    # cover all these shapes, and storage pruning must not lose a neighbor.
    expected = {
        GemmShape(m, n, k, b)
        for m, n, k, b in product(range(1, 4), repeat=4)
        if 4 * b * (m * k + n * k + 2 * m * n) <= data_limit
    }
    assert len(cases) == len(expected)
    assert {c.shape for c in cases} == expected


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
    selected = loaded.get_problems(17, 100)
    cases = {case.id: case.shape.k for case in selected}
    assert cases["minimum"] == 1024
    assert cases["k_2"] == 1088
    assert cases["k_3"] == 1152
    assert cases["k_long"] == 8192
    # Larger requests must reach both sides of a legal reduction boundary,
    # even when the minimum is many DepthU blocks long.
    assert {GemmShape(128, 96, k) for k in (2047, 2048, 2049)} <= {c.shape for c in selected}


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
        ("aType", "ComplexDouble"),
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
    case = replace(
        case,
        profile=replace(
            case.profile, use_bias=True, scale_ab="Scalar", scale_cd=True, scale_alpha_vector=True
        ),
    )
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


@pytest.mark.parametrize(
    "a_type, b_type, scale_type, block, target, layout, bench_scale",
    [
        ("Float8", "BFloat8", "E8", 32, "gfx950", 1, 1001),
        ("Float4", "Float4", "E8", 32, "gfx950", 1, 1001),
        ("Float6", "BFloat6", "E8", 32, "gfx1250", 2, 3),
        ("Float4", "Float4", "Float8", 16, "gfx1250", 2, 6),
        ("Float4", "Float4", "E5M3", 16, "gfx1250", 2, 8),
        ("Float8", "Float8", "E8", 32, "gfx950", 0, 3),
    ],
)
def test_mx_profiles_reach_both_clients(
    library, a_type, b_type, scale_type, block, target, layout, bench_scale
):
    loaded = library(
        solution(
            problem={
                "aType": a_type,
                "bType": b_type,
                "mxBlockA": block,
                "mxBlockB": block,
                "mxTypeA": scale_type,
                "mxTypeB": scale_type,
                "mxScaleFormat": layout,
                "useScaleAB": "Scalar",
            }
        ),
        target=target,
    )
    cases = loaded.get_problems(17, 10)
    rows = yaml.safe_load(bench_yaml(cases))
    assert len(rows) == len(cases) == 10
    assert all(row["scaleA"] == row["scaleB"] == bench_scale for row in rows)
    ini = dict(line.split("=", 1) for line in "".join(tensile_configs(cases)).splitlines())
    assert ini["mx-a-block"] == ini["mx-b-block"] == str(block)
    assert ini["mx-a-type"] == ini["mx-b-type"] == scale_type
    assert ini["mx-scale-format"] == str(int(layout != 0))
    assert ini["init-mx-a"] == ini["init-mx-b"] == "Random"
    assert ini["use-scaleAB"] == ""  # Optional scalar scales are disabled for both clients.
    assert ini["a-type"] == a_type and ini["b-type"] == b_type


@pytest.mark.parametrize("mx_side", ["a", "b"])
def test_mixed_mx_and_ordinary_operands(library, mx_side):
    loaded = library(
        solution(
            problem={
                "aType": "Half",
                "bType": "Half",
                f"{mx_side}Type": "Float8",
                f"mxBlock{mx_side.upper()}": 32,
                f"mxType{mx_side.upper()}": "E8",
            }
        )
    )
    (row,) = yaml.safe_load(bench_yaml(loaded.get_problems(17, 1)))
    assert row[f"scale{mx_side.upper()}"] == 3
    assert row["scaleB" if mx_side == "a" else "scaleA"] == "none"


@pytest.mark.parametrize("trans_a, trans_b", list(product((False, True), repeat=2)))
def test_packed_inputs_keep_columns_and_batches_in_whole_containers(library, trans_a, trans_b):
    a, b = ("lik" if trans_a else "ilk"), ("jlk" if trans_b else "ljk")
    loaded = library(
        solution(
            problem={
                "aType": "Float4",
                "bType": "Float6",
                "transA": trans_a,
                "transB": trans_b,
                "operationIdentifier": f"Contraction_l_A{a}_B{b}_Cijk_Dijk",
                "mxBlockA": 32,
                "mxBlockB": 32,
            }
        )
    )
    cases = loaded.get_problems(17, 40)
    assert len(cases) == 40
    for case in cases:
        s = case.shape
        assert case.leading_dimensions[0] % 2 == 0
        assert case.leading_dimensions[1] % 32 == 0
        assert case.batch_strides[0] % 2 == case.batch_strides[1] % 32 == 0
        expected = s.batch_count * (s.m * s.k // 2 + s.n * s.k * 3 // 4 + 8 * s.m * s.n)
        assert case.profile.data_bytes(s) == expected


@pytest.mark.parametrize(
    "update, message",
    [
        ({"mxBlockA": 8}, "block size"),
        ({"mxTypeA": "Half"}, "scale type"),
        ({"mxScaleFormat": 3}, "scale layout"),
    ],
)
def test_unrepresentable_mx_metadata_is_explicit(library, update, message):
    loaded = library(solution(problem={"mxBlockA": 32, "mxTypeA": "E8", **update}))
    with pytest.raises(ValueError, match=message):
        loaded.get_problems(17, 1)


def test_bench_rejects_a_scale_layout_it_cannot_upload(library):
    cases = library(
        solution(problem={"mxBlockA": 32, "mxTypeA": "E8", "mxScaleFormat": 0}),
        target="gfx1250",
    ).get_problems(17, 1)
    assert "mx-scale-format=0\n" in tensile_ini(cases)
    with pytest.raises(ValueError, match="use tensile_ini"):
        bench_yaml(cases)


@pytest.mark.parametrize(
    "activation, bench_activation, args",
    [("Relu", "relu", (0.0, 6.0)), ("Silu", "swish", (0.0, 6.0)), ("Clamp", "clamp", (-1.0, 1.0))],
)
def test_adapters_take_runtime_settings_from_the_case(library, activation, bench_activation, args):
    case = library(solution(problem={"activationType": "All"})).get_problems(17, 1)[0]
    selected = replace(case, alpha=2, beta=1, activation=activation, activation_args=args)
    (row,) = yaml.safe_load(bench_yaml([selected]))
    assert (row["alpha"], row["beta"], row["activation_type"]) == (2, 1, bench_activation)
    assert (row["activation_arg1"], row["activation_arg2"]) == args
    ini = tensile_ini([selected])
    assert "init-alpha=Two\n" in ini and "init-beta=One\n" in ini
    assert "init-c=Random\n" in ini
    assert f"activation-enum-args={activation}\n" in ini
    assert f"activation-additional-args={args[0]},{args[1]}\n" in ini
    with pytest.raises(ValueError, match="runtime profile"):
        tensile_ini([case, selected])


def test_batching_preserves_solution_associations(library):
    loaded = library(solution(17), solution(9))
    a, b = [candidate_gemms(loaded, i, max_cases=4) for i in (17, 9)]
    assert [r["solution_index"] for r in yaml.safe_load(bench_yaml(a + b))] == [17] * 4 + [9] * 4
    configs = tensile_configs(a)
    assert sum(ini.count("problem-size=") for ini in configs) == len(a)
    assert sum(ini.count("a-strides=") for ini in configs) == len(a)
    assert all(ini.count("solution-start-idx=") == 1 for ini in configs)
    with pytest.raises(ValueError, match="only one solution"):
        tensile_ini(a + b)
    foreign = replace(a[0], metadata_file=a[0].metadata_file.parent / "other" / "library.dat")
    with pytest.raises(ValueError, match="same library directory"):
        bench_yaml([a[0], foreign])
    with pytest.raises(ValueError, match="target"):
        bench_yaml([a[0], replace(a[0], target="gfx950")])
    with pytest.raises(ValueError, match="at least one case"):
        tensile_ini([])
