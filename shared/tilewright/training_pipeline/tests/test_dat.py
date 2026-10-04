# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
