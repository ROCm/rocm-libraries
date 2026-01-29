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

#include "miopen/miopen.h"
#include "miopen/errors.hpp"

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

constexpr bool IsValidUUT(UnitUnderTest uut)
{
    return (uut >= UnitUnderTest::optimizedGPU && uut <= UnitUnderTest::optimizedCPU);
}

constexpr bool IsValidREF(TestReference ref)
{
    return (ref >= TestReference::naiveGPU && ref <= TestReference::naiveCPU);
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
        MIOPEN_THROW("Invalid reference parameter in getNextREF() call");
    }

    switch(ref)
    {
    case TestReference::naiveGPU: return TestReference::optimizedCPU;
    case TestReference::optimizedCPU: return TestReference::naiveCPU;
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
    default: return "Unknown";
    }
}

/**
 * Number of runs that will be performed and analyzed when after test failure configuration is
 * miopenAfterTestFailureAnalyze
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
template <UnitUnderTest UUT, TestReference REF, AfterTestFailure ATF>
class AdaptiveTest
{
private:
    /**
     * Run numberOfRunsAfterFailure times with selected configuration, do not change the reference,
     * after runs extract some statistical information
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
        if(current_REF != TestReference::naiveCPU)
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
        if(currentREF == TestReference::naiveGPU)
        {
            // TODO: in future implementations, this will probably mean that all data is on GPU
            // therefore copy to cpu is needed.
        }
        */

        while(current_REF != TestReference::naiveCPU)
        {
            current_REF = GetNextREF(current_REF);

            // Do we need to re-run UUT?
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
        SetREFData();
        return ret;
    }

    /**
     * Struct to save error values for AFT == miopenAfterTestFailureAnalyze, and do analysis
     * of the values at the end
     */
    struct ErrorAnalysisInfo
    {
        int num_of_runs_failed = 0;
        std::unordered_map<std::string, std::array<double, number_of_runs_after_failure>> errors;

        void analyze()
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
    std::unordered_map<std::string, double> failure_errors;
    ErrorAnalysisInfo info;

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
    virtual std::pair<bool, std::unordered_map<std::string, double>> Verify() = 0;

    /**
     * Since there is an option to choose between different implementations that will be UUT/REF, we
     * need to use single pointer/reference to UUT/REF data, these methods are for that.
     *
     * setREFData is called after every execution of the reference implementation, it is not
     * constexpr as the reference can be changed throughout the test execution (when ATF ==
     * miopenAfterTestFailureMoveOn)
     *
     * setUUTData is called once at the start of the test, it is constexpr because the UUT is set at
     * the beginning and cannot be changed throughout the test texecution
     */
    virtual void SetREFData()           = 0;
    virtual void constexpr SetUUTData() = 0;

    void RunAdaptiveTest()
    {
        SetUUTData();
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

} // namespace test::adaptive
