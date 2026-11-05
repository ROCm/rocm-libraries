// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <half/half.hpp>
#include <vector>
#include "pooling2d_common.hpp"

// On gfx12 the Y and Z workgroup IDs are only 16 bits wide, so a launch with more than 65536
// workgroups in either dimension wraps blockIdx. The 2D pooling kernels used to map N*C to Z.
// These cases straddle that limit: N*C = 65536 is the largest count that fits, the others exceed
// it. They are valid problems on every architecture. A 32-bit index type is used so that the image
// index mode (wsidx = 1) fits for every shape.

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
    const std::vector<Config> all_configs = {{miopenPoolingMax, 0},
                                             {miopenPoolingMax, 1},
                                             {miopenPoolingAverage, 0},
                                             {miopenPoolingAverageInclusive, 0}};
    const std::vector<Config> few_configs = {{miopenPoolingMax, 0}, {miopenPoolingAverage, 0}};

    struct Problem
    {
        std::vector<int> in_shape;
        std::vector<int> lens;
        std::vector<int> pads;
        const std::vector<Config>& configs;
    };
    std::vector<Problem> problems = {
        {{512, 256, 8, 8}, {2, 2}, {0, 0}, all_configs}}; // N*C = 131072
    if(control_and_large_nc)
    {
        problems.push_back({{256, 256, 8, 8}, {2, 2}, {0, 0}, few_configs});   // N*C = 65536
        problems.push_back({{70000, 1, 28, 28}, {3, 3}, {1, 1}, all_configs}); // N*C = 70000
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

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_PoolingWorkgroupLimit2d_FP32,
                         testing::ValuesIn(GetPoolingWorkgroupLimit2dTestCases(true)),
                         pooling2d_gtest::GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_PoolingWorkgroupLimit2d_FP16,
                         testing::ValuesIn(GetPoolingWorkgroupLimit2dTestCases(false)),
                         pooling2d_gtest::GetPoolingTestCaseName);
