// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopendriver_common.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/process.hpp>

namespace {

std::vector<std::string> GetTestCases()
{
    // clang-format off
    return std::vector<std::string>{
        {std::string("convint8") + " -n 1 -c 64 -H 8 -W 8 -k 4 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -F 1 -V 1 -t 1"}
    };
    // clang-format on
}

using TestCase = decltype(GetTestCases())::value_type;

class GPU_MIOpenDriverInt8OutputDescriptorTest_I8
    : public testing::TestWithParam<std::vector<TestCase>>
{
};

void RunMIOpenDriver()
{
    using e_mask = enabled<Gpu::Default>;
    using d_mask = disabled<Gpu::Default>;
    if(!ShouldRunMIOpenDriverTest<d_mask, e_mask>())
    {
        GTEST_SKIP();
    }

    miopen::ProcessEnvironmentMap envVars;
    envVars["MIOPEN_DRIVER_USE_GPU_REFERENCE"] = "0";

    RunMIOpenDriverTestCommand(GPU_MIOpenDriverInt8OutputDescriptorTest_I8::GetParam(), envVars);
};

} // namespace

TEST_P(GPU_MIOpenDriverInt8OutputDescriptorTest_I8, MIOpenDriverInt8OutputDescriptor)
{
    RunMIOpenDriver();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_MIOpenDriverInt8OutputDescriptorTest_I8,
                         testing::Values(GetTestCases()));
