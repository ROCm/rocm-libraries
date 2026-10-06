# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import re
from pathlib import Path

import msgpack
import pytest

from bench_fakes import LIBRARY_STEM, OTHER_STEM, solution, write_logic
from lib import dat

BBS_NN = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_Type_BB_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx950"
)


def test_filename_decoding_keeps_mx_and_scalar_f8_apart():
    assert dat.parse_scale_mode(LIBRARY_STEM + ".dat") == 3
    assert dat.parse_scale_mode(OTHER_STEM + ".dat.zlib") == 1
    assert dat.parse_scale_mode(BBS_NN + ".dat") == 0
    assert dat.parse_scale_mode("unrelated.dat") is None


def test_mx_block_sizes():
    assert dat.mx_block_size_for_scale_mode(None) == 0
    assert dat.mx_block_size_for_scale_mode(1) == 0
    assert dat.mx_block_size_for_scale_mode(3) == 32
    assert dat.mx_block_size_for_scale_mode(4) == 16
    assert dat.mx_block_size_for_scale_mode(1001) == 32
    assert dat.mx_block_size_for_scale_mode(999) == 0


def test_contraction_logic_names_and_stems():
    assert dat.is_tensile_contraction_logic(LIBRARY_STEM + ".dat")
    assert dat.is_tensile_contraction_logic(LIBRARY_STEM + ".dat.zlib")
    assert not dat.is_tensile_contraction_logic("TensileLibrary_lazy_gfx1250.dat")
    assert not dat.is_tensile_contraction_logic(LIBRARY_STEM + ".yaml")
    assert dat.logic_stem(LIBRARY_STEM + ".dat.zlib") == LIBRARY_STEM
    assert dat.logic_stem(LIBRARY_STEM + ".dat") == LIBRARY_STEM


def test_kernel_dat_info_fields():
    info = dat.kernel_dat_info(
        solution(7, 3, (256, 128, 64), nta=4, ntb=2, occupancy=2, grvw=(8, 4), gwvw=2)
    )
    attributes = info.pop("attributes")
    assert set(attributes) == set(dat.ATTRIBUTE_NAMES)
    assert (attributes["non_temporal_a"], attributes["gwvw_c"]) == (4, 2)
    assert info == {
        "sol_idx_global": 7,
        "sol_idx_local": 3,
        "kernel_name": "Cijk_Alik_Bljk_F8BS_MT256x128x64_MI16x16x1_SN_TEST",
        "mt_m": 256,
        "mt_n": 128,
        "mt_k": 64,
        "mi_m": 16,
        "mi_n": 16,
        "mi_k": 128,
        "occupancy": 2,
        "cache_hints_a": 4,
        "cache_hints_b": 2,
        "grvw_a": 8,
        "grvw_b": 4,
        "gwvw_d": 2,
    }


@pytest.mark.parametrize("mi", [(0, 0, 0, 0), (0, 0, 0), ()])
def test_dot2_kernels_get_the_runtime_matrix_instruction(mi):
    info = dat.kernel_dat_info(solution(1, 0, mi=mi))
    assert (info["mi_m"], info["mi_n"], info["mi_k"]) == dat.DOT2_MI == (1, 1, 64)


@pytest.mark.parametrize(
    "temporal, expected", [((0, 0), (0, 0)), ((1, 2), (4, 0)), ((3, 1), (4, 4))]
)
def test_temporal_hints_replace_nontemporal(temporal, expected):
    sol = solution(1, 0, nta=2, ntb=2)
    sol["sizeMapping"].update(
        hasTemporalHint=True, temporalHintA=temporal[0], temporalHintB=temporal[1]
    )
    info = dat.kernel_dat_info(sol)
    assert (info["cache_hints_a"], info["cache_hints_b"]) == expected
    a = info["attributes"]
    assert (a["non_temporal_a"], a["non_temporal_b"], a["has_temporal_hint"]) == (
        2,
        2,
        1,
    )
    assert (a["temporal_hint_a"], a["temporal_hint_b"]) == temporal


@pytest.mark.parametrize("occ, expected", [(-1, 1), (0, 1), (1, 1), (4, 4)])
def test_occupancy_is_clamped(occ, expected):
    assert dat.kernel_dat_info(solution(1, 0, occupancy=occ))["occupancy"] == expected


def test_read_plain_and_compressed_logic(tmp_path):
    sols = [solution(10, 0), solution(11, 1)]
    plain = write_logic(tmp_path / (LIBRARY_STEM + ".dat"), sols)
    packed = write_logic(tmp_path / (OTHER_STEM + ".dat.zlib"), sols)
    assert dat.read_tensile_logic(plain)["solutions"][1]["index"] == 11
    assert dat.read_tensile_logic(packed)["solutions"][0]["index"] == 10
    bad = tmp_path / "bad_Contraction.dat"
    bad.write_bytes(b"\xc1not msgpack")
    assert dat.read_tensile_logic(bad) is None


def test_library_logic_path_and_kernels(tmp_path):
    with pytest.raises(FileNotFoundError):
        dat.library_logic_path(tmp_path, LIBRARY_STEM)
    write_logic(
        tmp_path / (LIBRARY_STEM + ".dat.zlib"), [solution(5, 1), solution(3, 0)]
    )
    assert dat.library_logic_path(tmp_path, LIBRARY_STEM).name.endswith(".dat.zlib")
    kernels = dat.load_library_kernels(tmp_path, LIBRARY_STEM)
    assert [k["sol_idx_global"] for k in kernels] == [5, 3]


def test_the_kernel_pool_is_the_prediction_table(tmp_path):
    sols = [solution(i, i, (64 * (i + 1), 64, 64)) for i in range(6)]
    path = write_logic(tmp_path / (LIBRARY_STEM + ".dat"), sols, table=[4, 2, 5, 3])
    assert dat.prediction_table(dat.read_tensile_logic(path)) == [4, 2, 5, 3]
    kernels = dat.load_library_kernels(tmp_path, LIBRARY_STEM)
    assert [k["sol_idx_global"] for k in kernels] == [4, 2, 5, 3]
    assert [k["mt_m"] for k in kernels] == [320, 192, 384, 256]
    index = dat.load_kernel_index(tmp_path, library_stem=LIBRARY_STEM)
    assert sorted(index.by_index) == list(range(6))
    assert dat.prediction_table({"solutions": []}) is None


def _two_prediction_rows(first, second):
    rows = [
        {"predicate": {"type": "PredictionMatching"}, "library": lib}
        for lib in (
            {"type": "Prediction", "table": first},
            {"type": "Prediction", "table": second},
        )
    ]
    return {"type": "Problem", "rows": rows}


def test_identical_prediction_tables_are_one_pool():
    doc = {"library": _two_prediction_rows([3, 1], [3, 1])}
    assert dat.prediction_table(doc) == [3, 1]


@pytest.mark.parametrize(
    "library, message",
    [
        (_two_prediction_rows([3, 1], [1, 3]), "2 different Prediction tables"),
        ({"type": "Prediction", "table": [1, 1]}, "repeats a solution index"),
        ({"type": "Prediction", "table": [{"index": 1}]}, "not a list of indices"),
        ({"type": "Prediction", "table": [True]}, "not a list of indices"),
        ({"type": "Prediction", "table": 5}, "not a list of indices"),
        ({"type": "Prediction", "table": [1, 9]}, "does not hold, e.g. 9"),
    ],
)
def test_prediction_tables_the_pipeline_cannot_follow(tmp_path, library, message):
    doc = {"solutions": [solution(1, 0), solution(2, 1)], "library": library}
    (tmp_path / (LIBRARY_STEM + ".dat")).write_bytes(msgpack.packb(doc))
    with pytest.raises(ValueError, match=f"{LIBRARY_STEM}.dat: .*{message}"):
        dat.load_library_kernels(tmp_path, LIBRARY_STEM)


def test_load_kernel_index_modes(tmp_path):
    write_logic(tmp_path / (LIBRARY_STEM + ".dat"), [solution(1, 0), solution(2, 1)])
    write_logic(tmp_path / (OTHER_STEM + ".dat"), [solution(2, 0), solution(9, 1)])
    (tmp_path / (BBS_NN + ".dat")).write_bytes(b"\xc1broken")

    by_stem = dat.load_kernel_index(tmp_path, library_stem=LIBRARY_STEM)
    assert sorted(by_stem.by_index) == [1, 2]
    assert by_stem.files == [LIBRARY_STEM + ".dat"]

    everything = dat.load_kernel_index(tmp_path)
    assert sorted(everything.by_index) == [1, 2, 9]
    assert everything.duplicate_indices == 1
    assert everything.unreadable == [BBS_NN + ".dat"]

    mx_only = dat.load_kernel_index(tmp_path, scale_mode=3)
    assert sorted(mx_only.by_index) == [1, 2]
    assert mx_only.skipped_scale_mode == 2

    (tmp_path / (OTHER_STEM + ".dat")).write_bytes(b"\xc1broken")
    with pytest.raises(ValueError):
        dat.load_kernel_index(tmp_path, library_stem=OTHER_STEM)


def test_sig_from_row():
    row = {
        "mt_m": "256",
        "mt_n": "128",
        "mt_k": "64",
        "mi_m": "16",
        "mi_n": "16",
        "mi_k": "128",
        "cache_hints_a": "4",
        "cache_hints_b": "",
    }
    assert dat.sig_from_row(row) == (256, 128, 64, 16, 16, 128, 4, 0)


# ── kernel attributes ────────────────────────────────────────────────────────

SCALAR_ATTRIBUTES = [
    ("wave_num", "waveNum"),
    ("gwvw_c", "gwvwC"),
    ("stagger_u", "staggerU"),
    ("stagger_u_mapping", "staggerUMapping"),
    ("global_split_u_pgr", "globalSplitUPGR"),
    ("global_split_u", "globalSplitU"),
    ("stagger_stride_shift", "staggerStrideShift"),
    ("workgroup_mapping", "workGroupMapping"),
    ("workgroup_mapping_xcc", "workGroupMappingXCC"),
    ("workgroup_mapping_xcc_group", "workGroupMappingXCCGroup"),
    ("global_split_u_coalesced", "globalSplitUCoalesced"),
    ("global_split_u_wgm_round_robin", "globalSplitUWorkGroupMappingRoundRobin"),
    ("pack_batch_dims", "packBatchDims"),
    ("pack_summation_dims", "packSummationDims"),
    ("magic_div_alg", "magicDivAlg"),
    ("prefetch_across_persistent", "prefetchAcrossPersistent"),
    ("persistent_kernel", "persistentKernel"),
    ("persistent_kernel_along_batch", "persistentKernelAlongBatch"),
    ("source_kernel", "sourceKernel"),
    ("global_accumulation", "globalAccumulation"),
    ("adaptive_gemm_gsua", "adaptiveGemmGSUA"),
    ("activation_fused", "activationFused"),
    ("prefetch_global_read", "PrefetchGlobalRead"),
    ("math_clocks_unrolled_loop", "MathClocksUnrolledLoop"),
    ("non_temporal_a", "nonTemporalA"),
    ("non_temporal_b", "nonTemporalB"),
    ("temporal_hint_a", "temporalHintA"),
    ("temporal_hint_b", "temporalHintB"),
    ("has_temporal_hint", "hasTemporalHint"),
    ("adaptive_gemm_ntab", "adaptiveGemmNTAB"),
    ("custom_main_loop_scheduling", "customMainLoopScheduling"),
    ("use_subtile_impl", "useSubtileImpl"),
    ("source_swap", "SourceSwap"),
    ("non_temporal_d", "NonTemporalD"),
    ("wave_separate_global_read_a", "WaveSeparateGlobalReadA"),
    ("wave_separate_global_read_b", "WaveSeparateGlobalReadB"),
    ("unroll_loop_swap_global_read_order", "UnrollLoopSwapGlobalReadOrder"),
    ("direct_to_vgpr_a", "DirectToVgprA"),
    ("direct_to_vgpr_b", "DirectToVgprB"),
    ("num_loads_coalesced_a", "NumLoadsCoalescedA"),
    ("num_loads_coalesced_b", "NumLoadsCoalescedB"),
    ("vector_width_a", "VectorWidthA"),
    ("vector_width_b", "VectorWidthB"),
    ("local_split_u", "LocalSplitU"),
    ("direct_to_lds_a", "DirectToLdsA"),
    ("direct_to_lds_b", "DirectToLdsB"),
    ("expert_scheduling_mode", "ExpertSchedulingMode"),
]
VECTOR_ATTRIBUTES = [
    (("work_group_x", "work_group_y", "work_group_z"), "workGroup"),
    (("thread_tile_x", "thread_tile_y", "thread_tile_z"), "threadTile"),
    (("wave_group_0", "wave_group_1"), "WaveGroup"),
    (("cluster_dim_x", "cluster_dim_y", "cluster_dim_z"), "clusterDim"),
]
POLICY_ATTRIBUTES = ("stream_k_atomic", "tile_processing_strategy", "work_assignment")
BOOL_KEYS = {
    "globalSplitUCoalesced",
    "globalSplitUWorkGroupMappingRoundRobin",
    "persistentKernelAlongBatch",
    "sourceKernel",
    "activationFused",
    "hasTemporalHint",
    "useSubtileImpl",
    "SourceSwap",
    "DirectToVgprA",
    "DirectToVgprB",
    "DirectToLdsA",
    "DirectToLdsB",
}
OPTIONAL_DEFAULTS = {
    "pack_batch_dims": 0,
    "pack_summation_dims": 0,
    "magic_div_alg": 1,
    "stream_k_atomic": 0,
    "tile_processing_strategy": 0,
    "work_assignment": 0,
    "prefetch_across_persistent": 0,
    "persistent_kernel": 0,
    "persistent_kernel_along_batch": 0,
    "adaptive_gemm_gsua": 0,
    "activation_fused": 1,
    "temporal_hint_a": 0,
    "temporal_hint_b": 0,
    "has_temporal_hint": 0,
    "adaptive_gemm_ntab": 0,
    "use_subtile_impl": 0,
    "source_swap": 0,
    "expert_scheduling_mode": 0,
    "cluster_dim_x": 1,
    "cluster_dim_y": 1,
    "cluster_dim_z": 1,
}


def test_attribute_names_are_the_engine_interface_set():
    names = [n for n, _k in SCALAR_ATTRIBUTES] + list(POLICY_ATTRIBUTES)
    names += [n for ns, _k in VECTOR_ATTRIBUTES for n in ns]
    assert len(names) == len(set(names)) == 61
    assert set(dat.ATTRIBUTE_NAMES) == set(names)
    assert len(dat.ATTRIBUTE_NAMES) == 61
    assert all(n and n == n.lower() for n in dat.ATTRIBUTE_NAMES)


RANKER_HPP = (
    Path(__file__).resolve().parents[4]
    / "projects/hipblaslt/tensilelite/include/Tensile/TilewrightRanker.hpp"
)


@pytest.mark.skipif(
    not RANKER_HPP.is_file(), reason="no TensileLite tilewright adapter"
)
def test_tensilelite_hands_the_engine_the_same_attribute_names():
    added = re.findall(r'\badd\("(\w+)",', RANKER_HPP.read_text())
    assert added == list(dat.ATTRIBUTE_NAMES)


def test_every_attribute_comes_from_its_size_mapping_key():
    sol = solution(1, 0)
    sm = sol["sizeMapping"]
    expected = {}
    for i, (name, key) in enumerate(SCALAR_ATTRIBUTES):
        value = bool(i % 2) if key in BOOL_KEYS else 100 + i
        sm[key] = value
        expected[name] = int(value)
    for j, (names, key) in enumerate(VECTOR_ATTRIBUTES):
        sm[key] = [1000 * (j + 1) + e for e in range(len(names))]
        expected.update(zip(names, sm[key]))
    sm.update(tileProcessingStrategy="StreamK", workAssignment="Hybrid")
    sm["streamKAtomic"] = 1
    expected.update(stream_k_atomic=1, tile_processing_strategy=2, work_assignment=2)
    attributes = dat.kernel_dat_info(sol)["attributes"]
    assert attributes == expected
    assert all(type(v) is int for v in attributes.values())


def test_absent_optional_keys_take_the_cpp_defaults():
    attributes = dat.kernel_dat_info(solution(1, 0))["attributes"]
    assert {k: attributes[k] for k in OPTIONAL_DEFAULTS} == OPTIONAL_DEFAULTS
    assert attributes["work_group_x"] == 256 and attributes["wave_group_1"] == 2


def test_absent_required_keys_take_the_cpp_defaults_and_are_counted(tmp_path):
    sol = solution(3, 0)
    for key in ("waveNum", "threadTile", "WaveGroup", "gwvwC", "PrefetchGlobalRead"):
        del sol["sizeMapping"][key]
    attributes = dat.kernel_dat_info(sol)["attributes"]
    assert (attributes["wave_num"], attributes["thread_tile_z"]) == (0, 0)
    assert (attributes["wave_group_0"], attributes["wave_group_1"]) == (0, 0)
    assert (attributes["gwvw_c"], attributes["prefetch_global_read"]) == (1, 2)
    assert dat.missing_size_mapping_keys(sol["sizeMapping"]) == [
        "waveNum",
        "threadTile",
        "gwvwC",
        "PrefetchGlobalRead",
        "WaveGroup",
    ]
    assert dat.missing_size_mapping_keys(solution(4, 1)["sizeMapping"]) == []
    write_logic(tmp_path / (LIBRARY_STEM + ".dat"), [sol, solution(4, 1)])
    index = dat.load_kernel_index(tmp_path, library_stem=LIBRARY_STEM)
    assert index.missing_keys == {
        "PrefetchGlobalRead": 1,
        "WaveGroup": 1,
        "gwvwC": 1,
        "threadTile": 1,
        "waveNum": 1,
    }


def test_vector_keys_pad_short_lists_with_the_defaults():
    sol = solution(1, 0, workGroup=[64], clusterDim=[2, 4])
    a = dat.kernel_dat_info(sol)["attributes"]
    assert (a["work_group_x"], a["work_group_y"], a["work_group_z"]) == (64, 0, 0)
    assert (a["cluster_dim_x"], a["cluster_dim_y"], a["cluster_dim_z"]) == (2, 4, 1)


@pytest.mark.parametrize(
    "keys, expected",
    [
        ({}, (0, 0, 0)),
        ({"streamK": 0}, (0, 0, 0)),
        ({"streamK": 0, "streamKAtomic": 1}, (0, 0, 0)),
        ({"streamK": 3}, (0, 2, 0)),
        ({"streamK": 3, "streamKAtomic": 1}, (1, 2, 0)),
        ({"streamK": 4}, (0, 2, 1)),
        ({"streamK": 5}, (0, 2, 2)),
        ({"streamK": 3, "streamKForceDPOnly": 1}, (0, 1, 0)),
        ({"streamKForceDPOnly": 0}, (0, 0, 0)),
        (
            {"streamK": 4, "tileProcessingStrategy": "StreamK"},
            (0, 2, 1),
        ),
        ({"streamK": 0, "workAssignment": "Hybrid"}, (0, 0, 0)),
        ({"tileProcessingStrategy": "DataParallel"}, (0, 1, 0)),
        (
            {
                "tileProcessingStrategy": "StreamK",
                "workAssignment": "DynamicWorkQueue",
                "streamKAtomic": 1,
            },
            (1, 2, 1),
        ),
        ({"tileProcessingStrategy": "None", "workAssignment": "Hybrid"}, (0, 0, 0)),
    ],
)
def test_legacy_stream_k_keys_normalize_like_tensilelite(keys, expected):
    a = dat.kernel_dat_info(solution(1, 0, **keys))["attributes"]
    assert tuple(a[n] for n in POLICY_ATTRIBUTES) == expected


@pytest.mark.parametrize(
    "keys, message",
    [
        ({"streamK": 1}, "unsupported legacy streamK mode"),
        ({"streamK": 3, "tileProcessingStrategy": "None"}, "conflicting"),
        ({"streamK": 4, "workAssignment": "Hybrid"}, "conflicting"),
        ({"streamKForceDPOnly": 2}, "must be 0 or 1"),
        ({"streamKForceDPOnly": 1, "streamK": 4}, "requires non-atomic streamK 3"),
        (
            {"streamKForceDPOnly": 1, "streamK": 3, "streamKAtomic": 1},
            "requires non-atomic streamK 3",
        ),
        ({"tileProcessingStrategy": "Bogus"}, "is not one of"),
        ({"workAssignment": 1}, "is not one of"),
        ({"tileProcessingStrategy": "StreamK", "streamKAtomic": 2}, "must be 0 or 1"),
        ({"streamKAtomic": 1}, "requires tileProcessingStrategy StreamK"),
        (
            {"tileProcessingStrategy": "DataParallel", "workAssignment": "Hybrid"},
            "StaticGrid only",
        ),
        ({"workGroupMapping": "8"}, "is not an integer"),
        ({"waveNum": 1 << 63}, "does not fit in int64"),
        ({"sourceKernel": 2}, "is not a bool"),
        ({"workGroup": 256}, "is not a list"),
    ],
)
def test_size_mappings_tensilelite_rejects_raise(keys, message):
    with pytest.raises(ValueError, match=f"^solution 9: .*{message}"):
        dat.kernel_dat_info(solution(9, 0, **keys))


def test_unreadable_solutions_fail_a_library_and_skip_a_directory_file(tmp_path):
    write_logic(tmp_path / (LIBRARY_STEM + ".dat"), [solution(1, 0)])
    write_logic(tmp_path / (OTHER_STEM + ".dat"), [solution(2, 0, streamK=2)])
    with pytest.raises(ValueError, match=f"{OTHER_STEM}.dat: solution 2: "):
        dat.load_library_kernels(tmp_path, OTHER_STEM)
    with pytest.raises(ValueError, match="unsupported legacy streamK mode"):
        dat.load_kernel_index(tmp_path, library_stem=OTHER_STEM)
    index = dat.load_kernel_index(tmp_path)
    assert sorted(index.by_index) == [1] and index.files == [LIBRARY_STEM + ".dat"]
    assert index.invalid == [
        f"{OTHER_STEM}.dat: solution 2: unsupported legacy streamK mode 2"
    ]


def test_kernel_attribute_file_round_trip(tmp_path):
    kernels = {
        7: dat.kernel_dat_info(solution(7, 0, workGroupMapping=-3)),
        2: dat.kernel_dat_info(solution(2, 1, streamK=4)),
    }
    path = tmp_path / dat.KERNELS_FILE
    dat.write_kernel_attributes(path, kernels, LIBRARY_STEM)
    got = dat.read_kernel_attributes(path)
    assert got == {sid: info["attributes"] for sid, info in kernels.items()}
    assert got[7]["workgroup_mapping"] == -3 and got[2]["work_assignment"] == 1
    doc = json.loads(path.read_text())
    assert doc["library_stem"] == LIBRARY_STEM
    assert doc["attribute_names"] == list(dat.ATTRIBUTE_NAMES)
    assert list(doc["kernels"]) == ["2", "7"]
    for bad in (
        '{"attribute_names": ["a"], "kernels": {"1": [1, 2]}}',
        '{"attribute_names": ["a", "a"], "kernels": {"1": [1, 2]}}',
        '{"attribute_names": ["", "b"], "kernels": {"1": [1, 2]}}',
        '{"attribute_names": ["a"], "kernels": {"1": ["x"]}}',
        '{"kernels": {}}',
        "[]",
    ):
        path.write_text(bad)
        with pytest.raises(ValueError, match="not a kernel attribute file"):
            dat.read_kernel_attributes(path)
