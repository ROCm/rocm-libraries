// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_arguments.hpp"
#include "hipblaslt_bench_options.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <tuple>

namespace
{
    class HostUnitBenchUniformSummationOrder
        : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, int32_t>>
    {
        int32_t saved_cli = hipblaslt_bench_options::uniform_summation_order();

        void TearDown() override
        {
            hipblaslt_bench_options::uniform_summation_order() = saved_cli;
        }
    };

    TEST_P(HostUnitBenchUniformSummationOrder, smoke_CaseOverridesCliAndUnsetStaysUnspecified)
    {
        const auto [argument, cli, expected] = GetParam();
        Arguments arg;
        arg.init();
        arg.uniform_summation_order                        = argument;
        hipblaslt_bench_options::uniform_summation_order() = cli;

        hipblaslt_bench_options::apply_to(arg);
        EXPECT_EQ(arg.uniform_summation_order, expected);
        EXPECT_EQ(hipblaslt_bench_options::uniform_summation_order(), cli);
    }

    INSTANTIATE_TEST_SUITE_P(
        ClientOptions,
        HostUnitBenchUniformSummationOrder,
        ::testing::Values(std::make_tuple(-1, -1, -1),
                          std::make_tuple(-1, 0, 0),
                          std::make_tuple(-1, 1, 1),
                          std::make_tuple(0, -1, 0),
                          std::make_tuple(0, 0, 0),
                          std::make_tuple(0, 1, 0),
                          std::make_tuple(1, -1, 1),
                          std::make_tuple(1, 0, 1),
                          std::make_tuple(1, 1, 1)),
        ([](const ::testing::TestParamInfo<HostUnitBenchUniformSummationOrder::ParamType>& info) {
            const auto [argument, cli, expected] = info.param;
            auto name                            = [](int32_t value) {
                return value < 0 ? std::string("Unset") : std::to_string(value);
            };
            return "Arg" + name(argument) + "Cli" + name(cli) + "Expected" + name(expected);
        }));
}
