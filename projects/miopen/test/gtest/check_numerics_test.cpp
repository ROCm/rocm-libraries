/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or associated portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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

#include "get_handle.hpp"
#include "../tensor_holder.hpp"

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

// FP32 tests
using GPU_CheckNumerics_FP32 = GPU_CheckNumerics<float>;

TEST_P(GPU_CheckNumerics_FP32, Test)
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

TEST_P(GPU_CheckNumerics_FP16, Test)
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

TEST_P(GPU_CheckNumerics_BFP16, Test)
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

} // namespace
