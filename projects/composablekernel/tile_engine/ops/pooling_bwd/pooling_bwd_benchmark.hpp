// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling_bwd.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include "ck_tile/host/reference/reference_pool_bwd.hpp"

namespace ck_tile {

struct PoolBwdProblem2D
{
    index_t N, H, W, C;
    index_t Y, X;
    index_t stride_h, stride_w;
    index_t dilation_h, dilation_w;
    index_t pad_h_left, pad_h_right;
    index_t pad_w_left, pad_w_right;
    std::string datatype;

    index_t Ho() const
    {
        index_t Ys = (Y - 1) * dilation_h + 1;
        return (H + pad_h_left + pad_h_right - Ys) / stride_h + 1;
    }

    index_t Wo() const
    {
        index_t Xs = (X - 1) * dilation_w + 1;
        return (W + pad_w_left + pad_w_right - Xs) / stride_w + 1;
    }

    index_t input_elements() const { return N * H * W * C; }
    index_t output_elements() const { return N * Ho() * Wo() * C; }

    bool window_overlap() const
    {
        const index_t eff_h = (Y - 1) * dilation_h + 1;
        const index_t eff_w = (X - 1) * dilation_w + 1;
        return eff_h > stride_h || eff_w > stride_w;
    }

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "N" << N << "_H" << H << "_W" << W << "_C" << C << "_Y" << Y << "_X" << X << "_Sh"
            << stride_h << "_Sw" << stride_w << "_Dh" << dilation_h << "_Dw" << dilation_w;
        return oss.str();
    }
};

struct PoolBwdProblem3D
{
    index_t N, D, H, W, C;
    index_t Z, Y, X;
    index_t stride_d, stride_h, stride_w;
    index_t dilation_d, dilation_h, dilation_w;
    index_t pad_d_left, pad_d_right;
    index_t pad_h_left, pad_h_right;
    index_t pad_w_left, pad_w_right;
    std::string datatype;

    index_t Do() const
    {
        index_t Zs = (Z - 1) * dilation_d + 1;
        return (D + pad_d_left + pad_d_right - Zs) / stride_d + 1;
    }

    index_t Ho() const
    {
        index_t Ys = (Y - 1) * dilation_h + 1;
        return (H + pad_h_left + pad_h_right - Ys) / stride_h + 1;
    }

    index_t Wo() const
    {
        index_t Xs = (X - 1) * dilation_w + 1;
        return (W + pad_w_left + pad_w_right - Xs) / stride_w + 1;
    }

    index_t input_elements() const { return N * D * H * W * C; }
    index_t output_elements() const { return N * Do() * Ho() * Wo() * C; }

    bool window_overlap() const
    {
        const index_t eff_d = (Z - 1) * dilation_d + 1;
        const index_t eff_h = (Y - 1) * dilation_h + 1;
        const index_t eff_w = (X - 1) * dilation_w + 1;
        return eff_d > stride_d || eff_h > stride_h || eff_w > stride_w;
    }

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "N" << N << "_D" << D << "_H" << H << "_W" << W << "_C" << C << "_Z" << Z << "_Y"
            << Y << "_X" << X;
        return oss.str();
    }
};

struct PoolBwdPerformanceResult
{
    float latency_ms;
    float bandwidth_gb_s;

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "latency=" << latency_ms << "ms, bandwidth=" << bandwidth_gb_s << "GB/s";
        return oss.str();
    }
};

struct PoolBwdBenchmarkSetting
{
    int warmup      = 5;
    int repeat      = 20;
    bool verify     = true;
    int init_method = 0;
};

} // namespace ck_tile
