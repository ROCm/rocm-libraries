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

/*! @enum miopenUnitUnderTest_t
 * Enum values for selecting unit under test (UUT).
 * There is no option to choose Naive CPU implementation as UUT because that is 'the most trusted'
 * implementation that should verify result of other implementations.
 */
typedef enum
{
    miopenUnitOptimizedGPU = 0, /*!< Optimized GPU implementation as UUT. */
    miopenUnitNaiveGPU     = 1, /*!< Naive GPU implementation as UUT. */
    miopenUnitOptimizedCPU = 2, /*!< Optimized CPU implementation as UUT. */
} miopenUnitUnderTest_t;

/*! @enum miopenTestReference_t
 * Enum values for selecting test reference (REF).
 * There is no option to choose Optimized GPU as test reference because that is 'the least trusted'
 * implementation and it can only be used as UUT.
 */
typedef enum
{
    miopenTestReferenceNaiveGPU     = 0, /*!< Naive GPU implementation as reference. */
    miopenTestReferenceOptimizedCPU = 1, /*!< Optimized CPU implementation as reference. */
    miopenTestReferenceNaiveCPU     = 2, /*!< Naive CPU implementation as reference. */
} miopenTestReference_t;

/*! @enum miopenAfterTestFailure_t
 * Enum values for selecting an option on what to do after failure of the choosen configuration.
 * There are two options, do additional runs and provide more information about errors or move on to
 * the next, more trusted, implementations and try to verify with next reference.
 */
typedef enum
{
    miopenAfterTestFailureNone    = 0, /*!< Do not do anything on test failure. */
    miopenAfterTestFailureAnalyze = 1, /*!< Analyze and provide more information. */
    miopenAfterTestFailureMoveOn  = 2, /*!< Move on to the next, more trusted, reference. */
} miopenAfterTestFailure_t;

constexpr bool isValidUUT(miopenUnitUnderTest_t uut)
{
    return (uut >= miopenUnitOptimizedGPU && uut <= miopenUnitOptimizedCPU);
}

constexpr bool isValidREF(miopenTestReference_t ref)
{
    return (ref >= miopenTestReferenceNaiveGPU && ref <= miopenTestReferenceNaiveCPU);
}

constexpr bool checkReferenceChoice(miopenUnitUnderTest_t uut, miopenTestReference_t ref)
{
    switch(uut)
    {
    case miopenUnitOptimizedGPU: return (ref >= miopenTestReferenceNaiveGPU);
    case miopenUnitNaiveGPU: return (ref >= miopenTestReferenceOptimizedCPU);
    case miopenUnitOptimizedCPU: return (ref == miopenTestReferenceNaiveCPU);
    default: return false;
    }
}

constexpr bool checkTestConfiguration(miopenUnitUnderTest_t uut, miopenTestReference_t ref)
{
    return isValidUUT(uut) && isValidREF(ref) && checkReferenceChoice(uut, ref);
}

static miopenTestReference_t getNextREF(miopenTestReference_t ref)
{
    if(!isValidREF(ref))
    {
        MIOPEN_THROW("Invalid reference parameter in getNextREF() call");
    }

    switch(ref)
    {
    case miopenTestReferenceNaiveGPU: return miopenTestReferenceOptimizedCPU;
    case miopenTestReferenceOptimizedCPU: return miopenTestReferenceNaiveCPU;
    default: return miopenTestReferenceNaiveCPU;
    }
}

static std::string getREFName(miopenTestReference_t ref)
{
    if(!isValidREF(ref))
    {
        MIOPEN_THROW("Invalid reference parameter in getREFName() call");
    }

    switch(ref)
    {
    case miopenTestReferenceNaiveGPU: return "naive GPU";
    case miopenTestReferenceOptimizedCPU: return "optimized CPU";
    case miopenTestReferenceNaiveCPU: return "naive CPU";
    default: return "Unknown";
    }
}

/**
 * Number of runs that will be performed and analyzed when after test failure configuration is
 * miopenAfterTestFailureAnalyze
 */
constexpr int numberOfRunsAfterFailure = 5;

/**
 * Base class for gtests, it provides interface to have different runs based on template parameters
 * that will determine the configuration.
 *
 * (UnitUnderTest)    UUT - desired unit that will be tested
 * (TestReference)    REF - desired implementation that will be used as reference
 * (AfterTestFailure) ATF - option that determine what to do (if anything) after test failure
 */
template <miopenUnitUnderTest_t UUT, miopenTestReference_t REF, miopenAfterTestFailure_t ATF>
class GTestBase
{
private:
    /**
     * Run numberOfRunsAfterFailure times with selected configuration, do not change the reference,
     * after runs extract some statistical information
     */
    void analyzeAfterTestFailure()
    {
        std::cout << "Test failed against " << getREFName(currentREF) << " reference." << std::endl
                  << "Doing additional runs with selected configuration." << std::endl
                  << "Collecting more information." << std::endl;

        for(int i{0}; i < numberOfRunsAfterFailure; i++)
        {
            std::ignore                         = runUUT();
            std::ignore                         = runREF();
            std::tie(testPassed, failureErrors) = verify();

            if(!testPassed)
            {
                for(auto [key, value] : failureErrors)
                {
                    if(!info.errors.contains(key))
                    {
                        info.errors.insert({key, std::array<double, numberOfRunsAfterFailure>{}});
                    }
                    info.errors[key][info.numOfRunsFailed] = value;
                }
                info.numOfRunsFailed++;
            }
        }
        info.analyze();
    }

    /**
     * Change to 'more trusted' reference implementation after failure, and re-test with it. Repeat
     * the process until verification succeeded or we've run with all of the 'more trusted'
     * references.
     */
    void moveOnAfterTestFailure()
    {
        std::cout << "Test failed against " << getREFName(currentREF) << " reference." << std::endl;
        if(currentREF != miopenTestReferenceNaiveCPU)
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
        if(currentREF == miopenTestReferenceNaiveGPU)
        {
            // TODO: in future implementations, this will probably mean that all data is on GPU
            // therefore copy to cpu is needed.
        }
        */

        while(currentREF != miopenTestReferenceNaiveCPU)
        {
            currentREF = getNextREF(currentREF);

            // Do we need to re-run UUT?
            auto ret = runREF();

            if(ret == miopenStatusNotImplemented)
            {
                continue;
            }

            std::tie(testPassed, std::ignore) = verify();

            if(testPassed)
            {
                std::cout << "Test passed against " << getREFName(currentREF) << " reference."
                          << std::endl;
                break;
            }
            else
            {
                std::cout << "Test failed against " << getREFName(currentREF) << " reference."
                          << std::endl;
            }
        }
    }

    miopenStatus_t runUUT()
    {
        miopenStatus_t ret;
        if constexpr(UUT == miopenUnitOptimizedGPU)
        {
            ret = runOptimizedGPU();
        }
        else if constexpr(UUT == miopenUnitNaiveGPU)
        {
            ret = runNaiveGPU();
        }
        else
        {
            ret = runOptimizedCPU();
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

    miopenStatus_t runREF()
    {
        miopenStatus_t ret;
        if(currentREF == miopenTestReferenceNaiveGPU)
        {
            ret = runNaiveGPU();
        }
        else if(currentREF == miopenTestReferenceOptimizedCPU)
        {
            ret = runOptimizedCPU();
        }
        else
        {
            ret = runNaiveCPU();
        }
        setREFData();
        return ret;
    }

    /**
     * Struct to save error values for AFT == miopenAfterTestFailureAnalyze, and do analysis
     * of the values at the end
     */
    struct ErrorAnalysisInfo
    {
        int numOfRunsFailed = 0;
        std::unordered_map<std::string, std::array<double, numberOfRunsAfterFailure>> errors;

        void analyze()
        {

            std::cout << numOfRunsFailed << " out of " << numberOfRunsAfterFailure
                      << " runs have failed." << std::endl;
            for(auto [key, value] : errors)
            {
                double meanError = std::reduce(value.begin(), value.begin() + numOfRunsFailed) /
                                   static_cast<double>(numOfRunsFailed);
                double maxError =
                    *(std::max_element(value.begin(), value.begin() + numOfRunsFailed));
                double minError =
                    *(std::min_element(value.begin(), value.begin() + numOfRunsFailed));

                std::cout << key << ": " << std::endl;
                std::cout << "\terrors [ ";
                for(auto el : value)
                {
                    std::cout << el << " ";
                }
                std::cout << "]" << std::endl;
                std::cout << "\tmean error: " << std::to_string(meanError) << std::endl
                          << "\tmax error:  " << std::to_string(maxError) << std::endl
                          << "\tmin error:  " << std::to_string(minError) << std::endl;
            }
        }
    };

    bool testPassed = true;
    std::unordered_map<std::string, double> failureErrors;
    ErrorAnalysisInfo info;

protected:
    miopenTestReference_t currentREF = REF;
    /**
     * NOTE. These should be able to be called several times without invoking SetUp again.
     */
    virtual miopenStatus_t runOptimizedGPU() = 0;
    virtual miopenStatus_t runNaiveGPU()     = 0;
    virtual miopenStatus_t runOptimizedCPU() = 0;
    virtual miopenStatus_t runNaiveCPU()     = 0;

    /**
     * Use EXPECT_* instead of ASSERT_* in verifying function so that on failure execution can
     * continue and perform additional work after falure
     *
     * Return value is pair where:
     * first [bool] is false if some verifying failed, true otherwise
     * second [unordered_map] map should contain small name of the value(s) that failed and error
     * value(s)
     */
    virtual std::pair<bool, std::unordered_map<std::string, double>> verify() = 0;

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
    virtual void setREFData()           = 0;
    virtual void constexpr setUUTData() = 0;

    void runTest()
    {
        setUUTData();
        std::ignore = runUUT();
        auto ret    = runREF();

        if(ret == miopenStatusNotImplemented)
        {
            MIOPEN_THROW("Selected reference is not implemented.");
        }

        std::tie(testPassed, failureErrors) = verify();

        if(!testPassed && ATF != miopenAfterTestFailureNone)
        {
            if constexpr(ATF == miopenAfterTestFailureAnalyze)
            {
                analyzeAfterTestFailure();
            }
            else if constexpr(ATF == miopenAfterTestFailureMoveOn)
            {
                moveOnAfterTestFailure();
            }
            else
            {
                MIOPEN_THROW("Unknown afterTestFailure option");
            }
        }
    };

    virtual ~GTestBase() {}
};
