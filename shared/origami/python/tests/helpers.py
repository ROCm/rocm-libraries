# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared test utilities for origami tests."""

import origami


# Hardware configurations for supported architectures
SUPPORTED_ARCHITECTURES = {
    "gfx90a": {
        "arch": origami.architecture_t.gfx90a,
        "N_CU": 110,
        "lds_capacity": 64 * 1024,
        "L2_capacity": 8 * 1024 * 1024,
        "compute_clock_khz": 1700000,
    },
    "gfx942": {
        "arch": origami.architecture_t.gfx942,
        "N_CU": 228,
        "lds_capacity": 64 * 1024,
        "L2_capacity": 24 * 1024 * 1024,
        "compute_clock_khz": 1700000,
    },
    "gfx950": {
        "arch": origami.architecture_t.gfx950,
        "N_CU": 304,
        "lds_capacity": 64 * 1024,
        "L2_capacity": 32 * 1024 * 1024,
        "compute_clock_khz": 2100000,
    },
    "gfx1100": {
        "arch": origami.architecture_t.gfx1100,
        "N_CU": 96,
        "lds_capacity": 64 * 1024,
        "L2_capacity": 6 * 1024 * 1024,
        "compute_clock_khz": 2500000,
    },
    "gfx1201": {
        "arch": origami.architecture_t.gfx1201,
        "N_CU": 60,
        "lds_capacity": 128 * 1024,
        "L2_capacity": 6 * 1024 * 1024,
        "compute_clock_khz": 2500000,
    },
}


def create_hardware(arch_name: str) -> origami.hardware_t:
    """Create a hardware object for the given architecture name."""
    if arch_name not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}")

    params = SUPPORTED_ARCHITECTURES[arch_name]
    return origami.get_hardware_for_arch(
        params["arch"],
        params["N_CU"],
        params["lds_capacity"],
        params["L2_capacity"],
        params["compute_clock_khz"],
    )


def get_matrix_instructions(
    hardware: origami.hardware_t, dtype: str
) -> list[tuple[int, int, int]]:
    """Get valid matrix instructions from hardware for the given dtype."""
    dtype_enum = origami.string_to_datatype(dtype)
    instructions = hardware.get_valid_matrix_instructions(dtype_enum)
    return [(mi.m, mi.n, mi.k) for mi in instructions]


def create_config_list(
    hardware: origami.hardware_t, dtype: str
) -> list[origami.config_t]:
    """Create a list of configurations for testing using dynamic MI discovery."""
    mi_list = get_matrix_instructions(hardware, dtype)
    if not mi_list:
        return []

    list_of_waves_to_include = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]
    min_mt0 = min_mt1 = 16
    max_mt0 = max_mt1 = 512

    configs = []
    for mi in mi_list:
        mi_m, mi_n, mi_k = mi

        for wave in list_of_waves_to_include:
            wave_tile_m = 0

            while True:
                wave_tile_m += 1
                mt0 = mi_m * wave_tile_m * wave[0]
                if mt0 < min_mt0:
                    continue
                if mt0 > max_mt0:
                    break

                wave_tile_n = 0
                while True:
                    wave_tile_n += 1
                    mt1 = mi_n * wave_tile_n * wave[1]

                    if mt1 < min_mt1:
                        continue
                    if mt1 > max_mt1:
                        break

                    for du in [16, 32, 64, 128, 256, 512, 1024]:
                        config = origami.config_t()
                        config.mt = origami.dim3_t(mt0, mt1, du)
                        config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
                        config.occupancy = 1
                        config.workgroup_mapping = 6
                        configs.append(config)

    return configs
