// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <half/half.hpp>
#include <vector>
#include "pooling2d_common.hpp"

// On gfx12 the Y and Z workgroup IDs are only 16 bits wide, so a launch with more than 65536
// workgroups in either dimension wraps blockIdx. The 2D pooling kernels map N*C to Z and the
// vertical tiles to Y; only when that grid exceeds the architecture's limit do the solvers select
// the kernel variant that linearizes all workgroups into X (otherwise the original launch is
// used). The cases cover the 65536 Z-workgroup limit on gfx1200/gfx1201, larger Z grids, and a
// grid that exceeds the limit only in Y. They are valid problems on every architecture. A 32-bit
// index type is used so that the image index mode (wsidx = 1) fits for every shape.

namespace {

std::vector<pooling2d_gtest::PoolingTestCase>
GetPoolingWorkgroupLimit2dTestCases(bool control_and_large_nc)
{
    using pooling2d_gtest::PoolingTestCase;

    struct Config
    {
        miopenPoolingMode_t mode;
        int wsidx;
    };
    const std::vector<Config> all_configs    = {{miopenPoolingMax, 0},
                                                {miopenPoolingMax, 1},
                                                {miopenPoolingAverage, 0},
                                                {miopenPoolingAverageInclusive, 0}};
    const std::vector<Config> few_configs    = {{miopenPoolingMax, 0}, {miopenPoolingAverage, 0}};
    const std::vector<Config> average_config = {{miopenPoolingAverage, 0}};

    struct Problem
    {
        std::vector<int> in_shape;
        std::vector<int> lens;
        std::vector<int> pads;
        const std::vector<Config>& configs;
    };
    std::vector<Problem> problems = {
        {{512, 256, 20, 14}, {2, 2}, {0, 0}, all_configs}}; // N*C = 131072
    if(control_and_large_nc)
    {
        problems.push_back({{256, 256, 14, 20}, {2, 2}, {0, 0}, few_configs}); // N*C = 65536
        problems.push_back({{70000, 1, 28, 20}, {3, 3}, {1, 1}, all_configs}); // N*C = 70000

        // With the default config, forward tile height is (8 / 1) * 1 = 8, so
        // ceil(524289 / 8) = 65537 Y groups. Backward uses the input height and a
        // 4 * 1 tile: ceil(1048578 / 4) = 262145 Y groups. N*C = 1, so only Y exceeds the limit.
        problems.push_back({{1, 1, 1048578, 2}, {2, 2}, {0, 0}, average_config});
    }

    std::vector<PoolingTestCase> test_cases;
    for(const auto& problem : problems)
    {
        for(const auto& config : problem.configs)
        {
            test_cases.push_back({problem.in_shape,
                                  problem.lens,
                                  problem.pads,
                                  {2, 2},
                                  miopenIndexUint32,
                                  config.mode,
                                  config.wsidx,
                                  "NCHW",
                                  "NCHW"});
        }
    }
    return test_cases;
}

} // namespace

class GPU_PoolingWorkgroupLimit2d_FP32 : public pooling2d_gtest::Pooling2dCommon<float>
{
};

class GPU_PoolingWorkgroupLimit2d_FP16 : public pooling2d_gtest::Pooling2dCommon<half_float::half>
{
};

TEST_P(GPU_PoolingWorkgroupLimit2d_FP32, FloatTest_pooling2d_workgroup_limit) { RunTest(); }

TEST_P(GPU_PoolingWorkgroupLimit2d_FP16, HalfTest_pooling2d_workgroup_limit) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_PoolingWorkgroupLimit2d_FP32,
                         testing::ValuesIn(GetPoolingWorkgroupLimit2dTestCases(true)),
                         pooling2d_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_PoolingWorkgroupLimit2d_FP16,
                         testing::ValuesIn(GetPoolingWorkgroupLimit2dTestCases(false)),
                         pooling2d_gtest::GetPoolingTestCaseName);
