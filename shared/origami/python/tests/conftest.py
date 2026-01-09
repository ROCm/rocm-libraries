# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for origami tests."""

import pytest
import origami


@pytest.fixture
def hardware():
    """Get hardware for device 0."""
    return origami.get_hardware_for_device(0)


@pytest.fixture
def mat_inst():
    """Matrix instruction configurations for different architectures."""
    mat_inst = {"gfx950": {}, "gfx942": {}}
    mat_inst["gfx950"]["f32"] = [(16, 16, 4, 1), (32, 32, 2, 1)]
    mat_inst["gfx950"]["f64"] = [(16, 16, 4, 1)]
    mat_inst["gfx950"]["f16"] = [
        (16, 16, 32, 1),  # gfx950
        (32, 32, 16, 1),  # gfx950
    ]
    mat_inst["gfx950"]["bf16"] = mat_inst["gfx950"]["f16"]
    mat_inst["gfx950"]["xf32"] = [
        (16, 16, 32, 1),  # gfx950
        (32, 32, 16, 1),  # gfx950
    ]
    mat_inst["gfx950"]["f8"] = [
        (4, 4, 4, 16),  # gfx950, gfx942
        (16, 16, 128, 1),  # gfx950
        (32, 32, 64, 1),  # gfx950
    ]
    mat_inst["gfx950"]["bf8"] = mat_inst["gfx950"]["f8"]
    return mat_inst


def create_config_list(arch, gemm_type, mat_inst):
    """Create a list of configurations for testing."""
    list_of_waves_to_include = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]
    min_mt0 = min_mt1 = 16
    max_mt0 = max_mt1 = 512

    # generate all configs for each datatype:
    bm_max = 0
    configs = []
    for mi in mat_inst[arch][gemm_type]:
        for bm in range(bm_max + 1):
            mi_block_m = 2**bm

            for wave in list_of_waves_to_include:
                wave_tile_m = 0
                wave_tile_n = 0

                while True:
                    wave_tile_m += 1
                    wave_tile_n = 0
                    matrix_inst_m = mi[0] * mi_block_m
                    mt0 = matrix_inst_m * wave_tile_m * wave[0]
                    if mt0 < min_mt0:
                        continue
                    if mt0 > max_mt0:
                        break

                    while True:
                        wave_tile_n += 1
                        matrix_inst_n = mi[1] / mi_block_m * mi[3]
                        mt1 = int(matrix_inst_n * wave_tile_n * wave[1])

                        if mt1 < min_mt1:
                            continue
                        if mt1 > max_mt1:
                            break

                        # LDS size check for LSU
                        lsu = max(1, 4 // wave[0] // wave[1])
                        if lsu > 1 and mt0 * mt1 * 4 * lsu > 256 * 256:
                            continue

                        if mt0 * mt1 > 256 * 256:
                            continue
                        for du in [16, 32, 64, 128, 256, 512, 1024]:
                            # Create config_t object
                            config = origami.config_t()
                            config.mt = origami.dim3_t(mt0, mt1, du)
                            config.mi = origami.dim3_t(mi[0], mi[1], mi[2])
                            config.occupancy = 1
                            config.workgroup_mapping = 6
                            configs.append(config)

    return configs

