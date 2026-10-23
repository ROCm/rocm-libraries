// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <hip/hip_runtime.h>
#include "ck_tile_grouped_conv_configs.hpp"

namespace miopen::solver::ck_tile_detail {

enum class Direction
{
    Forward,
    BackwardData,
    BackwardWeight
};

enum class DataType
{
    Half,
    BFloat16,
    Float
};

// Logical X/W/Y ordering, independent of which tensor the operation computes.
// C and K are per-group counts; 2D uses the first five/two array entries.
struct Arguments
{
    int spatial_dims    = 2;
    Direction direction = Direction::Forward;
    DataType data_type  = DataType::Half;
    int config_index    = 0;
    std::array<int64_t, 6> input_lengths{}, weight_lengths{}, output_lengths{};
    std::array<int64_t, 6> input_strides{}, weight_strides{}, output_strides{};
    std::array<int64_t, 3> strides{}, dilations{}, left_pads{}, right_pads{};
    const void* input  = nullptr;
    const void* weight = nullptr;
    const void* output = nullptr;
    int split_k        = 1;
    void* workspace    = nullptr;
};

bool IsSupported(const Arguments& args);
size_t GetWorkspaceSize(const Arguments& args);
float Run(const Arguments& args, hipStream_t stream);
std::string
GetKernelId(Direction direction, DataType data_type, int spatial_dims, int config_index = 0);

} // namespace miopen::solver::ck_tile_detail
