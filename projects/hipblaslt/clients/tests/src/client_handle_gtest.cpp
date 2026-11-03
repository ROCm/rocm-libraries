// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_bench_options.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>
#include <tuple>

class ClientHandleUniformSummationOrder
    : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, int32_t>>
{
    int32_t saved_cli = hipblaslt_bench_options::uniform_summation_order();

    void TearDown() override
    {
        hipblaslt_bench_options::uniform_summation_order() = saved_cli;
    }
};

TEST_P(ClientHandleUniformSummationOrder, ArgumentOverridesCliAndUnsetKeepsLibraryDefault)
{
    const auto [argument, cli, expected] = GetParam();
    Arguments arg;
    arg.init();
    arg.uniform_summation_order                        = argument;
    hipblaslt_bench_options::uniform_summation_order() = cli;

    hipblaslt_local_handle handle{arg};
    int32_t                value = -1;
    ASSERT_EQ(hipblasLtGetUniformSummationOrder(handle, &value), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(value, expected);
}

INSTANTIATE_TEST_SUITE_P(ClientOptions,
                         ClientHandleUniformSummationOrder,
                         ::testing::Values(std::make_tuple(-1, -1, 1),
                                           std::make_tuple(-1, 0, 0),
                                           std::make_tuple(-1, 1, 1),
                                           std::make_tuple(0, -1, 0),
                                           std::make_tuple(0, 0, 0),
                                           std::make_tuple(0, 1, 0),
                                           std::make_tuple(1, -1, 1),
                                           std::make_tuple(1, 0, 1),
                                           std::make_tuple(1, 1, 1)));
