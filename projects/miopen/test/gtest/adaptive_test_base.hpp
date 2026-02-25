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
    noValidateAndRMS      = 0,
    noValidateAndMAE      = 1,
    noValidateAndMismatch = 2,
    validateAndRMS        = 3,
    validateAndMAE        = 4,
    validateAndMismatch   = 5,
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
    default: return TestReference::robustCPU;
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
          VerifyOption VER>
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

        /*
        if(current_REF == TestReference::naiveGPU)
        {
            // TODO: in future implementations, this will probably mean that all data is on GPU
            // therefore copy to cpu is needed.
        }
        */

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

    bool test_passed = true;
    std::unordered_map<std::string, TVerify> failure_errors;
    ErrorAnalysisInfo info;

    const int verify_block_size = 256;

public:
    inline static ChecksResult* res_dev = nullptr;
    inline static TVerify* rms_dev      = nullptr;
    inline static TVerify* max_dev      = nullptr;
    inline static TVerify* mae_dev      = nullptr;
    inline static TVerify* mismatch_dev = nullptr;

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
    virtual miopenStatus_t RunOptimizedGPU() = 0;
    virtual miopenStatus_t RunNaiveGPU()     = 0;
    virtual miopenStatus_t RunOptimizedCPU() = 0;
    virtual miopenStatus_t RunNaiveCPU()     = 0;

    /**
     * Use EXPECT_* instead of ASSERT_* in verifying function so that on failure execution can
     * continue and perform additional work after falure
     *
     * Return value is pair where:
     * first [bool] is false if some verifying failed, true otherwise
     * second [unordered_map] map should contain small name of the value(s) that failed and error
     * value(s)
     */
    virtual std::pair<bool, std::unordered_map<std::string, TVerify>> Verify() = 0;

    template <typename UUT_Type, typename REF_Type = UUT_Type>
    std::pair<ChecksResult, TVerify> VerifyOnGPU(miopen::Allocator::ManageDataPtr& uut_dev,
                                                 miopen::Allocator::ManageDataPtr& ref_dev,
                                                 size_t sz)
    {
        // This condition is based on the assumtion that verification will be performed in double
        // only for FP32 and FP64 outputs
        // static_assert((!std::is_same_v<T, float> && !std::is_same_v<T, double>) &&
        //               std::is_same_v<TVerify, float>);
        auto&& handle = get_handle();

        size_t block_cnt = (sz / 2 + verify_block_size - 1) / verify_block_size;
        block_cnt        = (block_cnt == 0) ? 1 : block_cnt;

        *res_dev = ChecksResult{true, true, true, true};

        TVerify error = static_cast<TVerify>(0);

        std::vector<size_t> vld{verify_block_size, 1, 1};
        std::vector<size_t> vgd{block_cnt * verify_block_size, 1, 1};

        std::string fp_type_verify =
            std::is_same_v<TVerify, float> ? "" : " -DMIOPEN_VERIFY_USE_DOUBLE_ACCUM=1";

        std::string algo     = "Verify_GPU";
        std::string net_conf = "";
        std::string param    = "-DBLOCK_SZ=" + std::to_string(verify_block_size) + fp_type_verify;
        std::string file     = "VerifyGPU.cpp";
        std::string kernel_name = "VerifyGPUKernel";

        if constexpr(VER == VerifyOption::noValidateAndMAE ||
                     VER == VerifyOption::noValidateAndMismatch ||
                     VER == VerifyOption::noValidateAndRMS)
        {
            param += " -DDO_VALIDATE=0";
        }
        else
        {
            param += " -DDO_VALIDATE=1";
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

        if constexpr(VER == VerifyOption::noValidateAndRMS || VER == VerifyOption::validateAndRMS)
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

            // handle.EnableProfiling();
            // handle.ResetKernelTime();

            kernel(uut_dev.get(), ref_dev.get(), sz, res_dev, rms_dev, max_dev);

            hipDeviceSynchronize();

            // std::cout << "gpu sq diff: " << *rms_dev;

            TVerify max = std::max({*max_dev, std::numeric_limits<TVerify>::min()});
            // std::cout << " max: " << max;
            // std::cout << " sz: " << sz << std::endl;
            // std::cout << handle.GetKernelTime() << std::endl;
            error = std::sqrt(*rms_dev) / (std::sqrt(sz) * max);
        }
        else if constexpr(VER == VerifyOption::noValidateAndMAE ||
                          VER == VerifyOption::validateAndMAE)
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

            kernel(uut_dev.get(), ref_dev.get(), sz, res_dev, mae_dev);

            hipDeviceSynchronize();

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

            kernel(uut_dev.get(), ref_dev.get(), sz, res_dev, mismatch_dev);

            hipDeviceSynchronize();

            error = *mismatch_dev;
        }

        return std::make_pair(*res_dev, error);
    }

    /**
     * Since there is an option to choose between different implementations that will be UUT/REF, we
     * need to use single pointer/reference to UUT/REF data, these methods are for that.
     *
     * GetREFData is called after every execution of the reference implementation, it is not
     * constexpr as the reference can be changed throughout the test execution (when ATF ==
     * AfterTestFailure::moveOn)
     *
     * SetUUTData is called once at the start of the test, it is constexpr because the UUT is set at
     * the beginning and cannot be changed throughout the test texecution
     */
    // virtual void GetREFDataDev()           = 0;
    // virtual void constexpr GetUUTDataDev() = 0;

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
          VerifyOption VER>
static void SetUpSharedVerifyData()
{
    hipMallocManaged(&AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::res_dev,
                     sizeof(*AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::res_dev));
    if constexpr(VER == VerifyOption::noValidateAndRMS || VER == VerifyOption::validateAndRMS)
    {
        hipMallocManaged(&AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::rms_dev, sizeof(TVerify));
        hipMallocManaged(&AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::max_dev, sizeof(TVerify));
    }
    else if constexpr(VER == VerifyOption::noValidateAndMAE || VER == VerifyOption::validateAndMAE)
    {
        hipMallocManaged(&AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::mae_dev, sizeof(TVerify));
    }
    else
    {
        hipMallocManaged(&AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::mismatch_dev,
                         sizeof(TVerify));
    }
}

template <typename T,
          typename TVerify,
          UnitUnderTest UUT,
          TestReference REF,
          AfterTestFailure ATF,
          VerifyOption VER>
static void TearDownSharedVerifyData()
{
    hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::res_dev);
    if constexpr(VER == VerifyOption::noValidateAndRMS || VER == VerifyOption::validateAndRMS)
    {
        hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::rms_dev);
        hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::max_dev);
    }
    else if constexpr(VER == VerifyOption::noValidateAndMAE || VER == VerifyOption::validateAndMAE)
    {
        hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::mae_dev);
    }
    else
    {
        hipFree(AdaptiveTest<T, TVerify, UUT, REF, ATF, VER>::mismatch_dev);
    }
}

} // namespace test::adaptive
