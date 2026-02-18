# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared test utilities for origami tests."""

import origami


# Hardware configurations for supported architectures
# Used by multiple test files for consistent test hardware
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
