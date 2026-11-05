// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_bench_options.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>
#include <tuple>

class ClientUniformSummationOrder
    : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, int32_t>>
{
    int32_t saved_cli = hipblaslt_bench_options::uniform_summation_order();

    void TearDown() override
    {
        hipblaslt_bench_options::uniform_summation_order() = saved_cli;
    }
};

TEST_P(ClientUniformSummationOrder, smoke_CaseOverridesCliAndUnsetStaysUnspecified)
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

TEST_P(ClientUniformSummationOrder, smoke_HandleUsesArgumentsWithoutReadingCli)
{
    const auto [argument, cli, resolved] = GetParam();
    Arguments arg;
    arg.init();
    arg.uniform_summation_order                        = argument;
    hipblaslt_bench_options::uniform_summation_order() = cli;

    hipblaslt_local_handle fresh;
    int32_t                library_default = -1;
    ASSERT_EQ(hipblasLtGetUniformSummationOrder(fresh, &library_default), HIPBLAS_STATUS_SUCCESS);

    hipblaslt_local_handle handle{arg};
    int32_t                value = -1;
    ASSERT_EQ(hipblasLtGetUniformSummationOrder(handle, &value), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(value, argument < 0 ? library_default : argument);
}

INSTANTIATE_TEST_SUITE_P(ClientOptions,
                         ClientUniformSummationOrder,
                         ::testing::Values(std::make_tuple(-1, -1, -1),
                                           std::make_tuple(-1, 0, 0),
                                           std::make_tuple(-1, 1, 1),
                                           std::make_tuple(0, -1, 0),
                                           std::make_tuple(0, 0, 0),
                                           std::make_tuple(0, 1, 0),
                                           std::make_tuple(1, -1, 1),
                                           std::make_tuple(1, 0, 1),
                                           std::make_tuple(1, 1, 1)));
