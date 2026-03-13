/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <numeric>
#include <array>
#include <tuple>
#include <unordered_map>
#include <type_traits>
#include <utility>

#include "miopen/miopen.h"
#include "miopen/errors.hpp"
#include "get_handle.hpp"

namespace test::adaptive {

#define ADAPTIVE_BASE_HIP_CHECK(exp)                                                   \
    if((exp) != hipSuccess)                                                            \
    {                                                                                  \
        MIOPEN_LOG_E(#exp "failed at line: " << __LINE__ << " in file: " << __FILE__); \
    }

/*! @enum class UnitUnderTest
 * Enum values for selecting unit under test (UUT).
 * There is no option to choose Naive CPU implementation as UUT because that is 'the most trusted'
 * implementation that should verify result of other implementations.
 */
enum class UnitUnderTest
{
    optimizedGPU = 0, /*!< Optimized GPU implementation as UUT. */
    naiveGPU     = 1, /*!< Naive GPU implementation as UUT. */
    optimizedCPU = 2, /*!< Optimized CPU implementation as UUT. */
};

/*! @enum class TestReference
 * Enum values for selecting test reference (REF).
 * There is no option to choose Optimized GPU as test reference because that is 'the least trusted'
 * implementation and it can only be used as UUT.
 */
enum class TestReference
{
    naiveGPU     = 0, /*!< Naive GPU implementation as reference. */
    optimizedCPU = 1, /*!< Optimized CPU implementation as reference. */
    naiveCPU     = 2, /*!< Naive CPU implementation as reference. */
    robustGPU    = 3, /*!< Robust GPU implementation as reference. */
    robustCPU    = 4, /*!< Robust CPU implementation as reference. */
};

/*! @enum class AfterTestFailure
 * Enum values for selecting an option on what to do after failure of the choosen configuration.
 * There are two options, do additional runs and provide more information about errors or move on to
 * the next, more trusted, implementations and try to verify with next reference.
 */
enum class AfterTestFailure
{
    none    = 0, /*!< Do not do anything on test failure. */
    analyze = 1, /*!< Analyze and provide more information. */
    moveOn  = 2, /*!< Move on to the next, more trusted, reference. */
};

enum class VerifyOption
{
    rms      = 0,
    mae      = 1,
    mismatch = 2,
};

constexpr bool IsValidUUT(UnitUnderTest uut)
{
    return (uut >= UnitUnderTest::optimizedGPU && uut <= UnitUnderTest::optimizedCPU);
}

constexpr bool IsValidREF(TestReference ref)
{
    return (ref >= TestReference::naiveGPU && ref <= TestReference::robustCPU);
}

constexpr bool CheckReferenceChoice(UnitUnderTest uut, TestReference ref)
{
    switch(uut)
    {
    case UnitUnderTest::optimizedGPU: return (ref >= TestReference::naiveGPU);
    case UnitUnderTest::naiveGPU: return (ref >= TestReference::optimizedCPU);
    case UnitUnderTest::optimizedCPU: return (ref == TestReference::naiveCPU);
    default: return false;
    }
}

constexpr bool CheckTestConfiguration(UnitUnderTest uut, TestReference ref)
{
    return IsValidUUT(uut) && IsValidREF(ref) && CheckReferenceChoice(uut, ref);
}

static TestReference GetNextREF(TestReference ref)
{
    if(!IsValidREF(ref))
    {
        MIOPEN_THROW("Invalid reference parameter in GetNextREF() call");
    }

    switch(ref)
    {
    case TestReference::naiveGPU: return TestReference::optimizedCPU;
    case TestReference::optimizedCPU: return TestReference::naiveCPU;
    case TestReference::naiveCPU: return TestReference::robustGPU;
    case TestReference::robustGPU: return TestReference::robustCPU;
    default: return TestReference::naiveCPU;
    }
}

static std::string GetREFName(TestReference ref)
{
    if(!IsValidREF(ref))
    {
        MIOPEN_THROW("Invalid reference parameter in getREFName() call");
    }

    switch(ref)
    {
    case TestReference::naiveGPU: return "naive GPU";
    case TestReference::optimizedCPU: return "optimized CPU";
    case TestReference::naiveCPU: return "naive CPU";
    case TestReference::robustGPU: return "robust GPU";
    case TestReference::robustCPU: return "robust CPU";
    default: return "Unknown";
    }
}

struct ChecksResult
{
    bool all_zeros_ref              = true;
    bool all_zeros_uut              = true;
    bool all_finite_and_non_nan_ref = true;
    bool all_finite_and_non_nan_uut = true;
};

/**
 * Number of runs that will be performed and analyzed when after test failure configuration is
 * AfterTestFailure::analyze
 */
constexpr int number_of_runs_after_failure = 5;

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties>
class AdaptiveTest;

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties>
static void SetUpSharedVerifyData();

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties>
static void TearDownSharedVerifyData();

/**
 * Base class for gtests, it provides interface to have different runs based on template parameters
 * that will determine the configuration.
 *
 * (UnitUnderTest)    UUT - desired unit that will be tested
 * (TestReference)    REF - desired implementation that will be used as reference
 * (AfterTestFailure) ATF - option that determine what to do (if anything) after test failure
 */
template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties = true>
class AdaptiveTest
{
private:
    /**
     * Run number_of_runs_after_failure times with selected configuration, do not change the
     * reference, after runs extract some statistical information
     */
    void AnalyzeAfterTestFailure()
    {
        std::cout << "Test failed against " << GetREFName(current_REF) << " reference." << std::endl
                  << "Doing additional runs with selected configuration." << std::endl
                  << "Collecting more information." << std::endl;

        for(int i{0}; i < number_of_runs_after_failure; i++)
        {
            std::ignore                           = RunUUT();
            std::ignore                           = RunREF();
            std::tie(test_passed, failure_errors) = Verify();

            if(!test_passed)
            {
                for(auto [key, value] : failure_errors)
                {
                    if(!info.errors.Contains(key))
                    {
                        info.errors.Insert(
                            {key, std::array<double, number_of_runs_after_failure>{}});
                    }
                    info.errors[key][info.num_of_runs_failed] = value;
                }
                info.num_of_runs_failed++;
            }
        }
        info.Analyze();
    }

    /**
     * Change to 'more trusted' reference implementation after failure, and re-test with it. Repeat
     * the process until verification succeeded or we've run with all of the 'more trusted'
     * references.
     */
    void MoveOnAfterTestFailure()
    {
        std::cout << "Test failed against " << GetREFName(current_REF) << " reference."
                  << std::endl;
        if(current_REF != TestReference::robustCPU)
        {
            std::cout << "Moving on to more trusted reference implementations." << std::endl;
        }
        else
        {
            std::cout << "No more trusted implementation than naive CPU reference, therefore "
                         "cannot move on to more trusted implementation."
                      << std::endl;
        }

        while(current_REF != TestReference::naiveCPU)
        {
            current_REF = GetNextREF(current_REF);

            auto ret = RunREF();

            if(ret == miopenStatusNotImplemented)
            {
                continue;
            }

            std::tie(test_passed, std::ignore) = Verify();

            if(test_passed)
            {
                std::cout << "Test passed against " << GetREFName(current_REF) << " reference."
                          << std::endl;
                break;
            }
            else
            {
                std::cout << "Test failed against " << GetREFName(current_REF) << " reference."
                          << std::endl;
            }
        }
    }

    miopenStatus_t RunUUT()
    {
        miopenStatus_t ret;
        if constexpr(UUT == UnitUnderTest::optimizedGPU)
        {
            ret = RunOptimizedGPU();
        }
        else if constexpr(UUT == UnitUnderTest::naiveGPU)
        {
            ret = RunNaiveGPU();
        }
        else
        {
            ret = RunOptimizedCPU();
        }

        if(ret == miopenStatusNotImplemented)
        {
            MIOPEN_THROW("Selected unit under test is not implemented.");
        }
        else
        {
            return ret;
        }
    }

    miopenStatus_t RunREF()
    {
        miopenStatus_t ret;
        if(current_REF == TestReference::naiveGPU)
        {
            ret = RunNaiveGPU();
        }
        else if(current_REF == TestReference::optimizedCPU)
        {
            ret = RunOptimizedCPU();
        }
        else
        {
            ret = RunNaiveCPU();
        }
        return ret;
    }

    /**
     * Struct to save error values for AFT == AfterTestFailure::analyze, and do analysis
     * of the values at the end
     */
    struct ErrorAnalysisInfo
    {
        int num_of_runs_failed = 0;
        std::unordered_map<std::string, std::array<double, number_of_runs_after_failure>> errors;

        void Analyze()
        {

            std::cout << num_of_runs_failed << " out of " << number_of_runs_after_failure
                      << " runs have failed." << std::endl;
            for(auto [key, value] : errors)
            {
                double mean_error = std::reduce(value.begin(), value.begin() + num_of_runs_failed) /
                                    static_cast<double>(num_of_runs_failed);
                double max_error =
                    *(std::max_element(value.begin(), value.begin() + num_of_runs_failed));
                double min_error =
                    *(std::min_element(value.begin(), value.begin() + num_of_runs_failed));

                std::cout << key << ": " << std::endl;
                std::cout << "\terrors [ ";
                for(auto el : value)
                {
                    std::cout << el << " ";
                }
                std::cout << "]" << std::endl;
                std::cout << "\tmean error: " << std::to_string(mean_error) << std::endl
                          << "\tmax error:  " << std::to_string(max_error) << std::endl
                          << "\tmin error:  " << std::to_string(min_error) << std::endl;
            }
        }
    };

    bool test_passed{true};
    std::unordered_map<std::string, TVerify> failure_errors{};
    ErrorAnalysisInfo info{};

    static constexpr int verify_block_size = 1024;

    inline static ChecksResult* res_dev = nullptr;
    inline static TVerify* rms_dev      = nullptr;
    inline static TVerify* max_dev      = nullptr;
    inline static TVerify* mae_dev      = nullptr;
    inline static TVerify* mismatch_dev = nullptr;

public:
    template <typename T_,
              typename TVerify_,
              UnitUnderTest UUT_,
              TestReference REF_,
              AfterTestFailure ATF_,
              VerifyOption VER_,
              bool CheckNumericProperties_>
    friend void SetUpSharedVerifyData();

    template <typename T_,
              typename TVerify_,
              UnitUnderTest UUT_,
              TestReference REF_,
              AfterTestFailure ATF_,
              VerifyOption VER_,
              bool CheckNumericProperties_>
    friend void TearDownSharedVerifyData();

protected:
    TestReference current_REF = REF;

    /**
     * Invoking corresponding implementation. These should be able to be called several times
     * without invoking SetUp again.
     *
     * Return value is:
     * miopenStatusNotImplemented - if corresponding implementation does not exists.
     * miopenStatusSuccess        - if correspongin implementation exists.
     */
    virtual miopenStatus_t RunOptimizedGPU() { return miopenStatusNotImplemented; }
    virtual miopenStatus_t RunNaiveGPU() { return miopenStatusNotImplemented; }
    virtual miopenStatus_t RunOptimizedCPU() { return miopenStatusNotImplemented; }
    virtual miopenStatus_t RunNaiveCPU() { return miopenStatusNotImplemented; }
    virtual miopenStatus_t RunRobustGPU() { return miopenStatusNotImplemented; }
    virtual miopenStatus_t RunRobustCPU() { return miopenStatusNotImplemented; }

    /**
     * Use EXPECT_* instead of ASSERT_* in verifying function so that on failure execution can
     * continue and perform additional work after falure
     * Data that will be verified needs to be correct for selected UUT or REF, therefore each test
     * should implement method that will give correct data in Verify() method, this is improtant
     * because in case of ATF == AfterTestFailure::moveOn the reference can be changed automatically
     * by the base class and on the next call of the Verify method results from the update reference
     * should be present
     *
     * Return value is pair where:
     * first [bool] is false if some verifying failed, true otherwise
     * second [unordered_map] map should contain small name of the value(s) that failed and error
     * value(s)
     */
    virtual std::pair<bool, std::unordered_map<std::string, TVerify>> Verify() = 0;

    template <typename UUT_Type, typename REF_Type = UUT_Type>
    std::pair<ChecksResult, TVerify> VerifyOnCPU(std::vector<UUT_Type>& uut,
                                                 std::vector<REF_Type>& ref)
    {
        bool all_zeros_uut = true, all_zeros_ref = true, all_finite_and_not_nan_uut = true,
             all_finite_and_not_nan_ref = true;
        if constexpr(CheckNumericProperties)
        {
            all_zeros_uut = miopen::range_zero(uut);
            all_zeros_ref = miopen::range_zero(ref);
            all_finite_and_not_nan_uut =
                (miopen::find_idx(uut, miopen::not_finite) == static_cast<int64_t>(-1));
            all_finite_and_not_nan_ref =
                (miopen::find_idx(ref, miopen::not_finite) == static_cast<int64_t>(-1));
        }

        TVerify error = static_cast<TVerify>(0);

        if constexpr(VER == VerifyOption::mae)
        {
            error = miopen::max_diff_v2(uut, ref);
        }
        else if constexpr(VER == VerifyOption::rms)
        {
            error = miopen::rms_range(uut, ref);
        }
        else
        {
            // static_assert(std::is_same_v<UUT_type, REF_Type>);
            if constexpr(std::is_integral_v<UUT_Type>)
            {
                error =
                    (miopen::mismatch_idx(uut, ref, [](UUT_Type v1, UUT_Type v2) { v1 == v2; }) >=
                     miopen::range_distance(uut))
                        ? static_cast<TVerify>(0)
                        : static_cast<TVerify>(1);
            }
            else
            {
                error = (miopen::mismatch_idx(uut, ref, miopen::float_equal) >=
                         miopen::range_distance(uut))
                            ? static_cast<TVerify>(0)
                            : static_cast<TVerify>(1);
            }
        }

        return std::make_pair(ChecksResult{all_zeros_ref,
                                           all_zeros_uut,
                                           all_finite_and_not_nan_ref,
                                           all_finite_and_not_nan_uut},
                              error);
    }

    template <typename UUT_Type, typename REF_Type = UUT_Type>
    std::pair<ChecksResult, TVerify>
    VerifyOnGPU(const void* uut_dev, const void* ref_dev, size_t sz)
    {
        auto&& handle = get_handle();

        size_t block_cnt = (sz / 2 + verify_block_size - 1) / verify_block_size;
        block_cnt        = (block_cnt == 0) ? 1 : block_cnt;

        if(res_dev == nullptr)
        {
            MIOPEN_THROW("Shared data used for verification is not allocated, invoke "
                         "SetUpSharedVerifyData/TearDownSharedVerifyData in "
                         "SetUpTestSuite/TearDownTestSuite.");
        }

        *res_dev = ChecksResult{true, true, true, true};

        TVerify error = static_cast<TVerify>(0);

        std::vector<size_t> vld{verify_block_size, 1, 1};
        std::vector<size_t> vgd{block_cnt * verify_block_size, 1, 1};

        std::string fp_type_verify =
            std::is_same_v<TVerify, float> ? "" : " -DMIOPEN_VERIFY_USE_DOUBLE_ACCUM=1";

        std::string algo     = "Verify_GPU";
        std::string net_conf = "";
        std::string param    = "-DBLOCK_SZ=" + std::to_string(verify_block_size) + fp_type_verify;
        std::string file     = "MIOpenVerifyGPU.cpp";
        std::string kernel_name = "VerifyGPUKernel";

        if constexpr(CheckNumericProperties)
        {
            param += " -DCHECK_NUMERIC_PROPERTIES=1";
        }
        else
        {
            param += " -DCHECK_NUMERIC_PROPERTIES=0";
        }

        if constexpr(std::is_same_v<UUT_Type, double>)
        {
            param += " -DMIOPEN_UUT_USE_FP64=1";
        }
        else if constexpr(std::is_same_v<UUT_Type, float>)
        {
            param += " -DMIOPEN_UUT_USE_FP32=1";
        }
        else if constexpr(std::is_same_v<UUT_Type, half_float::half>)
        {
            param += " -DMIOPEN_UUT_USE_FP16=1";
        }
        else if constexpr(std::is_same_v<UUT_Type, bfloat16>)
        {
            param += " -DMIOPEN_UUT_USE_BFP16=1";
        }

        if constexpr(std::is_same_v<REF_Type, double>)
        {
            param += " -DMIOPEN_REF_USE_FP64=1";
        }
        else if constexpr(std::is_same_v<REF_Type, float>)
        {
            param += " -DMIOPEN_REF_USE_FP32=1";
        }
        else if constexpr(std::is_same_v<REF_Type, half_float::half>)
        {
            param += " -DMIOPEN_REF_USE_FP16=1";
        }
        else if constexpr(std::is_same_v<REF_Type, bfloat16>)
        {
            param += " -DMIOPEN_REF_USE_BFP16=1";
        }
        // add support to other missing types

        if constexpr(VER == VerifyOption::rms)
        {
            algo += "_RMS";
            net_conf = algo + "_" + std::to_string(vld[0]) + "_" +
                       std::to_string(miopen_type<UUT_Type>{}) + "_" +
                       std::to_string(miopen_type<REF_Type>{}) + "_" +
                       std::to_string(miopen_type<TVerify>{});
            param += " -DCALCULATE_RMS=1 -DCALCULATE_MAE=0 -DFIND_MISMATCH=0";

            *rms_dev = static_cast<TVerify>(0.0);
            *max_dev = static_cast<TVerify>(0.0);

            auto&& kernels = handle.GetKernels(kernel_name, net_conf);

            miopen::KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                kernel = handle.AddKernel(algo, net_conf, file, kernel_name, vld, vgd, param);
            }

            kernel(uut_dev, ref_dev, sz, res_dev, rms_dev, max_dev);

            ADAPTIVE_BASE_HIP_CHECK(hipDeviceSynchronize());

            TVerify max = std::max({*max_dev, std::numeric_limits<TVerify>::min()});
            error       = std::sqrt(*rms_dev) / (std::sqrt(sz) * max);
        }
        else if constexpr(VER == VerifyOption::mae)
        {
            algo += "_MAE";
            net_conf = algo + "_" + std::to_string(vld[0]) + "_" + std::to_string(miopen_type<T>{});
            param += " -DCALCULATE_RMS=0 -DCALCULATE_MAE=1 -DFIND_MISMATCH=0";

            *mae_dev = 0.0;

            auto&& kernels = handle.GetKernels(kernel_name, net_conf);

            miopen::KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                kernel = handle.AddKernel(algo, net_conf, file, kernel_name, vld, vgd, param);
            }

            kernel(uut_dev, ref_dev, sz, res_dev, mae_dev);

            ADAPTIVE_BASE_HIP_CHECK(hipDeviceSynchronize());

            error = *mae_dev;
        }
        else
        {
            algo += "_MISMATCH";
            net_conf = algo + "_" + std::to_string(vld[0]) + "_" + std::to_string(miopen_type<T>{});
            param += " -DCALCULATE_RMS=0 -DCALCULATE_MAE=0 -DFIND_MISMATCH=1";

            *mismatch_dev = 0;

            auto&& kernels = handle.GetKernels(kernel_name, net_conf);

            miopen::KernelInvoke kernel;

            if(!kernels.empty())
            {
                kernel = kernels.front();
            }
            else
            {
                kernel = handle.AddKernel(algo, net_conf, file, kernel_name, vld, vgd, param);
            }

            kernel(uut_dev, ref_dev, sz, res_dev, mismatch_dev);

            ADAPTIVE_BASE_HIP_CHECK(hipDeviceSynchronize());

            error = *mismatch_dev;
        }

        return std::make_pair(*res_dev, error);
    }

    void RunAdaptiveTest()
    {
        std::ignore = RunUUT();
        auto ret    = RunREF();

        if(ret == miopenStatusNotImplemented)
        {
            MIOPEN_THROW("Selected reference is not implemented.");
        }

        std::tie(test_passed, failure_errors) = Verify();

        if(!test_passed && ATF != AfterTestFailure::none)
        {
            if constexpr(ATF == AfterTestFailure::analyze)
            {
                AnalyzeAfterTestFailure();
            }
            else if constexpr(ATF == AfterTestFailure::moveOn)
            {
                MoveOnAfterTestFailure();
            }
            else
            {
                MIOPEN_THROW("Unknown afterTestFailure option");
            }
        }
    };

    virtual ~AdaptiveTest() {}
};

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties = true>
static void SetUpSharedVerifyData()
{
    ADAPTIVE_BASE_HIP_CHECK(hipMallocManaged(
        &AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::res_dev,
        sizeof(*AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::res_dev)));
    if constexpr(VER == VerifyOption::rms)
    {
        ADAPTIVE_BASE_HIP_CHECK(hipMallocManaged(
            &AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::rms_dev,
            sizeof(TVerify)));
        ADAPTIVE_BASE_HIP_CHECK(hipMallocManaged(
            &AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::max_dev,
            sizeof(TVerify)));
    }
    else if constexpr(VER == VerifyOption::mae)
    {
        ADAPTIVE_BASE_HIP_CHECK(hipMallocManaged(
            &AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::mae_dev,
            sizeof(TVerify)));
    }
    else
    {
        ADAPTIVE_BASE_HIP_CHECK(hipMallocManaged(
            &AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::mismatch_dev,
            sizeof(TVerify)));
    }
}

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER,
          bool CheckNumericProperties = true>
static void TearDownSharedVerifyData()
{
    ADAPTIVE_BASE_HIP_CHECK(
        hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::res_dev));
    if constexpr(VER == VerifyOption::rms)
    {
        ADAPTIVE_BASE_HIP_CHECK(
            hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::rms_dev));
        ADAPTIVE_BASE_HIP_CHECK(
            hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::max_dev));
    }
    else if constexpr(VER == VerifyOption::mae)
    {
        ADAPTIVE_BASE_HIP_CHECK(
            hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::mae_dev));
    }
    else
    {
        ADAPTIVE_BASE_HIP_CHECK(hipFree(
            AdaptiveTest<T, TVerify, UUT, REF, ATF, VER, CheckNumericProperties>::mismatch_dev));
    }
}

} // namespace test::adaptive
