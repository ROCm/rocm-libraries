// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/handle.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/bfloat16.hpp>
#include <half/half.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include <stdexcept>
#include <ostream>

#include "get_handle.hpp"
#include "tensor_holder.hpp"

namespace {

enum class CheckNumericsTestType
{
    NormalZero,
    NormalOne,
    AbnormalNaN,
    AbnormalInf
};

template <class T>
struct CheckNumericsTestCase
{
    CheckNumericsTestType test_type;
    T value;
};

template <class T>
std::vector<CheckNumericsTestCase<T>> GetCheckNumericsTestCases()
{
    return {{CheckNumericsTestType::NormalZero, T(0.0f)},
            {CheckNumericsTestType::NormalOne, T(1.0f)},
            {CheckNumericsTestType::AbnormalNaN, std::numeric_limits<T>::quiet_NaN()},
            {CheckNumericsTestType::AbnormalInf, std::numeric_limits<T>::infinity()}};
}

template <class T>
struct GPU_CheckNumerics : public ::testing::TestWithParam<CheckNumericsTestCase<T>>
{
};

template <class T>
void RunNormalValueTest(T val)
{
    auto&& handle      = get_handle();
    constexpr int size = 42;
    miopen::TensorDescriptor desc{miopen_type<T>{}, {size}};
    std::vector<T> data(size, val);
    auto buffer = handle.Write(data);

    EXPECT_FALSE(
        miopen::checkNumericsImpl(handle, miopen::CheckNumerics::Throw, desc, buffer.get(), true));
    EXPECT_FALSE(
        miopen::checkNumericsImpl(handle, miopen::CheckNumerics::Throw, desc, buffer.get(), false));

    EXPECT_FALSE(miopen::checkNumericsImpl(handle,
                                           miopen::CheckNumerics::Throw |
                                               miopen::CheckNumerics::ComputeStats,
                                           desc,
                                           buffer.get(),
                                           true));
    EXPECT_FALSE(miopen::checkNumericsImpl(handle,
                                           miopen::CheckNumerics::Throw |
                                               miopen::CheckNumerics::ComputeStats,
                                           desc,
                                           buffer.get(),
                                           false));
}

template <class T>
void RunAbnormalValueTest(T val)
{
    auto&& handle      = get_handle();
    constexpr int size = 42;
    miopen::TensorDescriptor desc{miopen_type<T>{}, {size}};
    std::vector<T> data(size, val);
    auto buffer = handle.Write(data);

    EXPECT_TRUE(
        miopen::checkNumericsImpl(handle, miopen::CheckNumerics::Warn, desc, buffer.get(), true));
    EXPECT_TRUE(
        miopen::checkNumericsImpl(handle, miopen::CheckNumerics::Warn, desc, buffer.get(), false));

    EXPECT_THROW(
        {
            miopen::checkNumericsImpl(
                handle, miopen::CheckNumerics::Throw, desc, buffer.get(), true);
        },
        std::exception);
    EXPECT_THROW(
        {
            miopen::checkNumericsImpl(
                handle, miopen::CheckNumerics::Throw, desc, buffer.get(), false);
        },
        std::exception);

    EXPECT_TRUE(
        miopen::checkNumericsImpl(handle,
                                  miopen::CheckNumerics::Warn | miopen::CheckNumerics::ComputeStats,
                                  desc,
                                  buffer.get(),
                                  true));
    EXPECT_TRUE(
        miopen::checkNumericsImpl(handle,
                                  miopen::CheckNumerics::Warn | miopen::CheckNumerics::ComputeStats,
                                  desc,
                                  buffer.get(),
                                  false));

    EXPECT_THROW(
        {
            miopen::checkNumericsImpl(handle,
                                      miopen::CheckNumerics::Throw |
                                          miopen::CheckNumerics::ComputeStats,
                                      desc,
                                      buffer.get(),
                                      true);
        },
        std::exception);
    EXPECT_THROW(
        {
            miopen::checkNumericsImpl(handle,
                                      miopen::CheckNumerics::Throw |
                                          miopen::CheckNumerics::ComputeStats,
                                      desc,
                                      buffer.get(),
                                      false);
        },
        std::exception);
}

} // namespace

// FP32 tests
using GPU_CheckNumerics_FP32 = GPU_CheckNumerics<float>;

TEST_P(GPU_CheckNumerics_FP32, NormalAndAbnormalValues)
{
    const auto& test_case = this->GetParam();
    if(test_case.test_type == CheckNumericsTestType::NormalZero ||
       test_case.test_type == CheckNumericsTestType::NormalOne)
    {
        RunNormalValueTest(test_case.value);
    }
    else
    {
        RunAbnormalValueTest(test_case.value);
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_FP32,
                         testing::ValuesIn(GetCheckNumericsTestCases<float>()));

// FP16 tests
using GPU_CheckNumerics_FP16 = GPU_CheckNumerics<half_float::half>;

TEST_P(GPU_CheckNumerics_FP16, NormalAndAbnormalValues)
{
    const auto& test_case = this->GetParam();
    if(test_case.test_type == CheckNumericsTestType::NormalZero ||
       test_case.test_type == CheckNumericsTestType::NormalOne)
    {
        RunNormalValueTest(test_case.value);
    }
    else
    {
        RunAbnormalValueTest(test_case.value);
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_FP16,
                         testing::ValuesIn(GetCheckNumericsTestCases<half_float::half>()));

// BF16 tests
using GPU_CheckNumerics_BFP16 = GPU_CheckNumerics<bfloat16>;

TEST_P(GPU_CheckNumerics_BFP16, NormalAndAbnormalValues)
{
    const auto& test_case = this->GetParam();
    if(test_case.test_type == CheckNumericsTestType::NormalZero ||
       test_case.test_type == CheckNumericsTestType::NormalOne)
    {
        RunNormalValueTest(test_case.value);
    }
    else
    {
        RunAbnormalValueTest(test_case.value);
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_CheckNumerics_BFP16,
                         testing::ValuesIn(GetCheckNumericsTestCases<bfloat16>()));
