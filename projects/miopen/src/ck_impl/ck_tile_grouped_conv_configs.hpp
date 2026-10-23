// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

namespace miopen::solver::ck_tile_detail {

enum class Pipeline
{
    CompV3,
    BasicV1,
    Memory
};

struct KernelConfig
{
    Pipeline pipeline;
    int m, n, k;
    int wave_m, wave_n;
    int vector_a, vector_b, vector_c;
};

// Native WMMA geometry and vector choices from CK's grouped convolution examples
// and full configuration rules. This is an explicit portfolio, not a Cartesian
// product; unsupported problem/configuration pairs still fail kernel support.
// Entry zero preserves the existing scalar candidate and its persisted ID.
inline constexpr std::array KernelConfigs = {
    KernelConfig{Pipeline::CompV3, 16, 64, 64, 1, 4, 1, 1, 1},
    KernelConfig{Pipeline::CompV3, 32, 64, 32, 2, 2, 1, 1, 1},
    KernelConfig{Pipeline::CompV3, 64, 16, 64, 4, 1, 1, 1, 1},
    KernelConfig{Pipeline::CompV3, 64, 64, 64, 2, 2, 1, 1, 1},
    KernelConfig{Pipeline::CompV3, 16, 64, 64, 1, 4, 2, 2, 2},
    KernelConfig{Pipeline::CompV3, 16, 64, 64, 1, 4, 4, 4, 4},
    KernelConfig{Pipeline::CompV3, 64, 64, 64, 2, 2, 2, 2, 2},
    KernelConfig{Pipeline::CompV3, 64, 64, 64, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::CompV3, 64, 128, 32, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::CompV3, 128, 128, 64, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::CompV3, 64, 64, 128, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::CompV3, 128, 128, 32, 4, 2, 4, 4, 4},
    KernelConfig{Pipeline::BasicV1, 64, 64, 64, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::BasicV1, 128, 128, 64, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::Memory, 64, 64, 64, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::Memory, 128, 128, 64, 2, 2, 4, 4, 4},
    KernelConfig{Pipeline::CompV3, 64, 128, 32, 2, 2, 8, 8, 4},
    KernelConfig{Pipeline::CompV3, 128, 128, 64, 2, 2, 8, 8, 4},
};

} // namespace miopen::solver::ck_tile_detail
