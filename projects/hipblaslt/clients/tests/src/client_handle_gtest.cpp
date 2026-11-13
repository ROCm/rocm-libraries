// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_arguments.hpp"
#include "hipblaslt_bench_options.hpp"
#include "utility.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <tuple>

class ClientUniformSummationOrder : public ::testing::TestWithParam<std::tuple<int32_t, int32_t>>
{
    int32_t saved_cli = hipblaslt_bench_options::uniform_summation_order();

    void TearDown() override
    {
        hipblaslt_bench_options::uniform_summation_order() = saved_cli;
    }
};

TEST_P(ClientUniformSummationOrder, smoke_HandleUsesArgumentsWithoutReadingCli)
{
    const auto [argument, cli] = GetParam();
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

INSTANTIATE_TEST_SUITE_P(
    ClientOptions,
    ClientUniformSummationOrder,
    ::testing::Combine(::testing::Values(-1, 0, 1), ::testing::Values(-1, 0, 1)),
    ([](const ::testing::TestParamInfo<ClientUniformSummationOrder::ParamType>& info) {
        const auto [argument, cli] = info.param;
        auto name                  = [](int32_t value) {
            return value < 0 ? std::string("Unset") : std::to_string(value);
        };
        return "Arg" + name(argument) + "Cli" + name(cli);
    }));

TEST(ClientUniformSummationOrderError, smoke_InvalidRequestNamesSettingAndValue)
{
    Arguments arg;
    arg.init();
    arg.uniform_summation_order = 2;
    try
    {
        hipblaslt_local_handle handle{arg};
        FAIL() << "An invalid uniform_summation_order request was accepted";
    }
    catch(const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(), "uniform_summation_order=2: HIPBLAS_STATUS_INVALID_VALUE");
    }
}
