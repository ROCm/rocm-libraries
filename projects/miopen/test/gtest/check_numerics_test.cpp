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
 * copies or substantial portions of the Software.
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

// Normal value tests
template <class T>
void TestNormalValue(T val)
{
    auto&& handle = get_handle();
    constexpr int size = 42;
    miopen::TensorDescriptor desc{miopen_type<T>{}, {size}};
    std::vector<T> data(size, val);
    auto buffer = handle.Write(data);

    EXPECT_FALSE(miopen::checkNumericsImpl(
        handle, miopen::CheckNumerics::Throw, desc, buffer.get(), true));
    EXPECT_FALSE(miopen::checkNumericsImpl(
        handle, miopen::CheckNumerics::Throw, desc, buffer.get(), false));

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

// Abnormal value tests (NaN, Inf)
template <class T>
void TestAbnormalValue(T val)
{
    auto&& handle = get_handle();
    constexpr int size = 42;
    miopen::TensorDescriptor desc{miopen_type<T>{}, {size}};
    std::vector<T> data(size, val);
    auto buffer = handle.Write(data);

    EXPECT_TRUE(miopen::checkNumericsImpl(
        handle, miopen::CheckNumerics::Warn, desc, buffer.get(), true));
    EXPECT_TRUE(miopen::checkNumericsImpl(
        handle, miopen::CheckNumerics::Warn, desc, buffer.get(), false));

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

    EXPECT_TRUE(miopen::checkNumericsImpl(handle,
                                         miopen::CheckNumerics::Warn |
                                             miopen::CheckNumerics::ComputeStats,
                                         desc,
                                         buffer.get(),
                                         true));
    EXPECT_TRUE(miopen::checkNumericsImpl(handle,
                                         miopen::CheckNumerics::Warn |
                                             miopen::CheckNumerics::ComputeStats,
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

// Float tests
TEST(CheckNumerics, NormalZero_FP32) { TestNormalValue<float>(0.0f); }

TEST(CheckNumerics, NormalOne_FP32) { TestNormalValue<float>(1.0f); }

TEST(CheckNumerics, AbnormalNaN_FP32) { TestAbnormalValue<float>(std::numeric_limits<float>::quiet_NaN()); }

TEST(CheckNumerics, AbnormalInf_FP32) { TestAbnormalValue<float>(std::numeric_limits<float>::infinity()); }

// Half tests
TEST(CheckNumerics, NormalZero_FP16) { TestNormalValue<half_float::half>(half_float::half(0.0f)); }

TEST(CheckNumerics, NormalOne_FP16) { TestNormalValue<half_float::half>(half_float::half(1.0f)); }

TEST(CheckNumerics, AbnormalNaN_FP16) { TestAbnormalValue<half_float::half>(std::numeric_limits<half_float::half>::quiet_NaN()); }

TEST(CheckNumerics, AbnormalInf_FP16) { TestAbnormalValue<half_float::half>(std::numeric_limits<half_float::half>::infinity()); }

// BF16 tests
TEST(CheckNumerics, NormalZero_BF16) { TestNormalValue<bfloat16>(bfloat16(0.0f)); }

TEST(CheckNumerics, NormalOne_BF16) { TestNormalValue<bfloat16>(bfloat16(1.0f)); }

TEST(CheckNumerics, AbnormalNaN_BF16) { TestAbnormalValue<bfloat16>(std::numeric_limits<bfloat16>::quiet_NaN()); }

TEST(CheckNumerics, AbnormalInf_BF16) { TestAbnormalValue<bfloat16>(std::numeric_limits<bfloat16>::infinity()); }

} // namespace
