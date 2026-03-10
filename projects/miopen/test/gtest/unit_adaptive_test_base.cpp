/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#include "gtest_common.hpp"
#include "get_handle.hpp"
#include <miopen/bfloat16.hpp>
#include <half/half.hpp>

using namespace test::adaptive;

namespace {

TEST(CPU_WrongTestConfiguration_NONE, TestConfiguration)
{
    EXPECT_FALSE(CheckTestConfiguration(UnitUnderTest::optimizedCPU, TestReference::naiveGPU));
    EXPECT_FALSE(CheckTestConfiguration(UnitUnderTest::optimizedCPU, TestReference::optimizedCPU));

    EXPECT_TRUE(CheckTestConfiguration(UnitUnderTest::naiveGPU, TestReference::naiveCPU));
}

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties>
struct VerifyChecksGPU
    : public AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>,
      public ::testing::TestWithParam<std::size_t>
{
public:
    tensor<T> naiveGPUData;
    tensor<T> optimizedCPUData;

    miopen::Allocator::ManageDataPtr naiveGPU_dev;
    miopen::Allocator::ManageDataPtr optimizedCPU_dev;

    std::size_t len;

    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

    void SetUp() override
    {
        SetUpSharedVerifyData<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>();
        prng::reset_seed();
        len              = GetParam();
        naiveGPUData     = tensor<T>{{len}};
        optimizedCPUData = tensor<T>{{len}};
    }

    void TearDown() override
    {
        TearDownSharedVerifyData<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>();
    }

    miopenStatus_t RunNaiveGPU() override
    {
        auto&& handle = get_handle();
        naiveGPUData.generate(tensor_elem_gen_integer{max_value});
        T eps = std::numeric_limits<T>::epsilon() * static_cast<T>(5);
        naiveGPUData.par_for_each([&](size_t idx) {
            naiveGPUData(idx) = (idx % 2) ? (naiveGPUData(idx) + naiveGPUData(idx) * eps)
                                          : (naiveGPUData(idx) - naiveGPUData(idx) * eps);
        });

        naiveGPU_dev = handle.Write(naiveGPUData.data);

        return miopenStatusSuccess;
    }

    miopenStatus_t RunOptimizedCPU() override
    {
        auto&& handle = get_handle();
        optimizedCPUData.generate(tensor_elem_gen_integer{max_value});
        optimizedCPU_dev = handle.Write(optimizedCPUData.data);

        return miopenStatusSuccess;
    }

    std::pair<bool, std::unordered_map<std::string, TVerify>> Verify() override
    {
        auto [res, error] =
            this->template VerifyOnGPU<T>(naiveGPU_dev.get(), optimizedCPU_dev.get(), len);

        if constexpr(CheckNumericProperties)
        {
            EXPECT_EQUAL(miopen::range_zero(naiveGPUData.data), res.all_zeros_ref);
            EXPECT_EQUAL(miopen::range_zero(optimizedCPUData.data), res.all_zeros_ref);

            auto all_finite_uut = (miopen::find_idx(naiveGPUData.data, miopen::not_finite) == -1);
            auto all_finite_ref =
                (miopen::find_idx(optimizedCPUData.data, miopen::not_finite) == -1);

            EXPECT_EQUAL(all_finite_uut, res.all_finite_and_non_nan_uut);
            EXPECT_EQUAL(all_finite_ref, res.all_finite_and_non_nan_ref);
        }

        TVerify error_ref;
        if constexpr(VER == VerifyOption::rms)
        {
            error_ref = miopen::rms_range(naiveGPUData.data, optimizedCPUData.data);
        }
        else if constexpr(VER == VerifyOption::mae)
        {
            error_ref = miopen::max_diff(naiveGPUData.data, optimizedCPUData.data);
        }
        else
        {
            error_ref = (miopen::mismatch_idx(
                             naiveGPUData.data, optimizedCPUData.data, miopen::float_equal) < len)
                            ? 1
                            : 0;
        }

        EXPECT_LE(std::abs(error_ref - error), std::numeric_limits<TVerify>::epsilon());

        return {true, {}};
    }

    void Run() { this->RunAdaptiveTest(); }
};

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties>
struct VerifyChecksCPU
    : public AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>,
      public ::testing::TestWithParam<std::size_t>
{
public:
    tensor<T> naiveGPUData;
    tensor<T> optimizedCPUData;

    std::size_t len;

    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

    void SetUp() override
    {
        SetUpSharedVerifyData<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>();
        prng::reset_seed();
        len              = GetParam();
        naiveGPUData     = tensor<T>{{len}};
        optimizedCPUData = tensor<T>{{len}};
    }

    void TearDown() override
    {
        TearDownSharedVerifyData<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>();
    }

    miopenStatus_t RunNaiveGPU() override
    {
        naiveGPUData.generate(tensor_elem_gen_integer{max_value});
        T eps = std::numeric_limits<T>::epsilon() * static_cast<T>(5);
        naiveGPUData.par_for_each([&](size_t idx) {
            naiveGPUData(idx) = (idx % 2) ? (naiveGPUData(idx) + naiveGPUData(idx) * eps)
                                          : (naiveGPUData(idx) - naiveGPUData(idx) * eps);
        });

        return miopenStatusSuccess;
    }

    miopenStatus_t RunOptimizedCPU() override
    {
        optimizedCPUData.generate(tensor_elem_gen_integer{max_value});

        return miopenStatusSuccess;
    }

    std::pair<bool, std::unordered_map<std::string, TVerify>> Verify() override
    {
        auto [res, error] = this->template VerifyOnCPU<T>(naiveGPUData.data, optimizedCPUData.data);

        if constexpr(CheckNumericProperties)
        {
            EXPECT_EQUAL(miopen::range_zero(naiveGPUData.data), res.all_zeros_ref);
            EXPECT_EQUAL(miopen::range_zero(optimizedCPUData.data), res.all_zeros_ref);

            auto all_finite_uut = (miopen::find_idx(naiveGPUData.data, miopen::not_finite) == -1);
            auto all_finite_ref =
                (miopen::find_idx(optimizedCPUData.data, miopen::not_finite) == -1);

            EXPECT_EQUAL(all_finite_uut, res.all_finite_and_non_nan_uut);
            EXPECT_EQUAL(all_finite_ref, res.all_finite_and_non_nan_ref);
        }

        TVerify error_ref;
        if constexpr(VER == VerifyOption::rms)
        {
            error_ref = miopen::rms_range(naiveGPUData.data, optimizedCPUData.data);
        }
        else if constexpr(VER == VerifyOption::mae)
        {
            error_ref = miopen::max_diff(naiveGPUData.data, optimizedCPUData.data);
        }
        else
        {
            error_ref = (miopen::mismatch_idx(
                             naiveGPUData.data, optimizedCPUData.data, miopen::float_equal) < len)
                            ? 1
                            : 0;
        }

        EXPECT_LE(std::abs(error_ref - error), std::numeric_limits<TVerify>::epsilon());

        return {true, {}};
    }

    void Run() { this->RunAdaptiveTest(); }
};

#define X_INSTANTIATE_CAST(VER_DEVICE, T_DATA, T_VERIFY, VER, VER_NAME, CHECK_NUM)                 \
    using VER_DEVICE##_Verify_##T_DATA##_##T_VERIFY##_##VER_NAME##_##CHECK_NUM =                   \
        VerifyChecks##VER_DEVICE<T_DATA,                                                           \
                                 T_VERIFY,                                                         \
                                 UnitUnderTest::naiveGPU,                                          \
                                 TestReference::optimizedCPU,                                      \
                                 AfterTestFailure::none,                                           \
                                 VER,                                                              \
                                 CHECK_NUM>;                                                       \
    TEST_P(VER_DEVICE##_Verify_##T_DATA##_##T_VERIFY##_##VER_NAME##_##CHECK_NUM,                   \
           TestVerify##VER_DEVICE)                                                                 \
    {                                                                                              \
        Run();                                                                                     \
    };                                                                                             \
                                                                                                   \
    INSTANTIATE_TEST_SUITE_P(Smoke,                                                                \
                             VER_DEVICE##_Verify_##T_DATA##_##T_VERIFY##_##VER_NAME##_##CHECK_NUM, \
                             testing::Values(8192));

// RMS GPU
X_INSTANTIATE_CAST(GPU, double, double, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(GPU, float, double, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(GPU, float, float, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(GPU, half, double, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(GPU, half, float, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(GPU, bfloat16, float, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(GPU, double, double, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(GPU, float, double, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(GPU, float, float, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(GPU, half, double, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(GPU, half, float, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(GPU, bfloat16, float, VerifyOption::rms, RMS, true);

// MAE GPU
X_INSTANTIATE_CAST(GPU, double, double, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(GPU, float, double, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(GPU, float, float, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(GPU, half, double, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(GPU, half, float, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(GPU, bfloat16, float, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(GPU, double, double, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(GPU, float, double, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(GPU, float, float, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(GPU, half, double, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(GPU, half, float, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(GPU, bfloat16, float, VerifyOption::mae, MAE, true);

// Mismatch GPU
X_INSTANTIATE_CAST(GPU, double, double, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(GPU, float, double, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(GPU, float, float, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(GPU, half, double, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(GPU, half, float, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(GPU, bfloat16, float, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(GPU, double, double, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(GPU, float, double, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(GPU, float, float, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(GPU, half, double, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(GPU, half, float, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(GPU, bfloat16, float, VerifyOption::mismatch, Mismatch, true);

// RMS CPU
X_INSTANTIATE_CAST(CPU, double, double, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(CPU, float, double, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(CPU, float, float, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(CPU, half, double, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(CPU, half, float, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(CPU, bfloat16, float, VerifyOption::rms, RMS, false);
X_INSTANTIATE_CAST(CPU, double, double, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(CPU, float, double, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(CPU, float, float, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(CPU, half, double, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(CPU, half, float, VerifyOption::rms, RMS, true);
X_INSTANTIATE_CAST(CPU, bfloat16, float, VerifyOption::rms, RMS, true);

// MAE CPU
X_INSTANTIATE_CAST(CPU, double, double, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(CPU, float, double, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(CPU, float, float, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(CPU, half, double, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(CPU, half, float, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(CPU, bfloat16, float, VerifyOption::mae, MAE, false);
X_INSTANTIATE_CAST(CPU, double, double, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(CPU, float, double, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(CPU, float, float, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(CPU, half, double, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(CPU, half, float, VerifyOption::mae, MAE, true);
X_INSTANTIATE_CAST(CPU, bfloat16, float, VerifyOption::mae, MAE, true);

// Mismatch CPU
X_INSTANTIATE_CAST(CPU, double, double, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(CPU, float, double, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(CPU, float, float, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(CPU, half, double, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(CPU, half, float, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(CPU, bfloat16, float, VerifyOption::mismatch, Mismatch, false);
X_INSTANTIATE_CAST(CPU, double, double, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(CPU, float, double, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(CPU, float, float, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(CPU, half, double, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(CPU, half, float, VerifyOption::mismatch, Mismatch, true);
X_INSTANTIATE_CAST(CPU, bfloat16, float, VerifyOption::mismatch, Mismatch, true);

} // namespace
