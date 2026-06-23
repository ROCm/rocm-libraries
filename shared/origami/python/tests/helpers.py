# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared test utilities for origami tests."""

import origami


# Hardware objects for supported architectures
HARDWARE = {
    "gfx90a": origami.get_hardware_for_arch(
        origami.architecture_t.gfx90a, 110, 64 * 1024, 512 * 1024, 8 * 1024 * 1024, 1700000
    ),
    "gfx942": origami.get_hardware_for_arch(
        origami.architecture_t.gfx942, 228, 64 * 1024, 512 * 1024, 24 * 1024 * 1024, 1700000
    ),
    "gfx950": origami.get_hardware_for_arch(
        origami.architecture_t.gfx950, 304, 64 * 1024, 512 * 1024, 32 * 1024 * 1024, 2100000
    ),
    "gfx1100": origami.get_hardware_for_arch(
        origami.architecture_t.gfx1100, 96, 64 * 1024, 512 * 1024, 6 * 1024 * 1024, 2500000
    ),
    "gfx1201": origami.get_hardware_for_arch(
        origami.architecture_t.gfx1201, 60, 128 * 1024, 512 * 1024, 6 * 1024 * 1024, 2500000
    ),
}


def get_matrix_instructions(
    hardware: origami.hardware_t, dtype: str
) -> list[tuple[int, int, int]]:
    """Get valid matrix instructions from hardware for the given dtype."""
    dtype_enum = origami.string_to_datatype(dtype)
    instructions = hardware.get_valid_matrix_instructions(dtype_enum)
    return [(mi.m, mi.n, mi.k) for mi in instructions]


def _generate_mt_pairs(
    mi: tuple[int, int, int],
    mt_sizes: list[int] | None,
    waves: list[list[int]],
    max_mt: int,
) -> list[tuple[int, int]]:
    """Generate (mt_m, mt_n) pairs for a given matrix instruction."""
    if mt_sizes is not None:
        return [(mt_m, mt_n) for mt_m in mt_sizes for mt_n in mt_sizes]

    # Wave-based: MT = MI × wave_tile × wave
    mi_m, mi_n, _ = mi
    min_mt = 16
    pairs = []

    for wave in waves:
        wave_tile_m = 0
        while True:
            wave_tile_m += 1
            mt_m = mi_m * wave_tile_m * wave[0]
            if mt_m < min_mt:
                continue
            if mt_m > max_mt:
                break

            wave_tile_n = 0
            while True:
                wave_tile_n += 1
                mt_n = mi_n * wave_tile_n * wave[1]
                if mt_n < min_mt:
                    continue
                if mt_n > max_mt:
                    break
                pairs.append((mt_m, mt_n))

    return pairs


def create_config_list(
    hardware: origami.hardware_t,
    dtype: str,
    *,
    mt_sizes: list[int] | None = None,
    depth_unroll: list[int] | None = None,
    occupancy_values: list[int] | None = None,
    wgm_values: list[int] | None = None,
    waves: list[list[int]] | None = None,
    max_mt: int = 512,
) -> list[origami.config_t]:
    """Create a list of configurations for testing using dynamic MI discovery."""
    mi_list = get_matrix_instructions(hardware, dtype)
    if not mi_list:
        return []

    if depth_unroll is None:
        depth_unroll = [16, 32, 64, 128, 256, 512, 1024]
    if occupancy_values is None:
        occupancy_values = [1]
    if wgm_values is None:
        wgm_values = [6]
    if waves is None:
        waves = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]

    configs = []
    for mi in mi_list:
        mi_m, mi_n, mi_k = mi
        mt_pairs = _generate_mt_pairs(mi, mt_sizes, waves, max_mt)

        for mt_m, mt_n in mt_pairs:
            for mt_k in depth_unroll:
                for occ in occupancy_values:
                    for wgm in wgm_values:
                        config = origami.config_t()
                        config.mt = origami.dim3_t(mt_m, mt_n, mt_k)
                        config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
                        config.occupancy = occ
                        config.workgroup_mapping = wgm
                        configs.append(config)

    return configs
