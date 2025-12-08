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
 * Enum values for selecting unit under test (UUT) in manual configuration mode.
 * There is no option to choose Naive CPU implementation as UUT because that is 'most trusted'
 * implementation that should verify result of other implementations.
 */
typedef enum
{
    miopenUnitBothGPU      = 0, /*!< Both GPU implementations as UUT. */
    miopenUnitOptimizedGPU = 1, /*!< Optimized GPU implementation as UUT. */
    miopenUnitNaiveGPU     = 2, /*!< Naive GPU implementation as UUT. */
    miopenUnitOptimizedCPU = 3, /*!< Optimized CPU implementation as UUT. */
} miopenUnitUnderTest_t;

/*! @enum miopenTestReference_t
 * Enum values for selecting test reference (REF) in manual configuration mode.
 * There is no option to choose Optimized GPU as test reference because that is 'least trusted'
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
 * There are two options, do additional runs and provide more information about errors or move onto
 * the next, more trusted, implementation and try to verify with next reference.
 */
typedef enum
{
    miopenAfterTestFailureNone    = 0, /*!< Do not do anything on test failure. */
    miopenAfterTestFailureAnalyze = 1, /*!< Analyze and provide more information. */
    miopenAfterTestFailureMoveOn  = 2, /*!< Move on next, more trusted, reference. */
} miopenAfterTestFailure_t;

constexpr bool isValidUUT(miopenUnitUnderTest_t uut)
{
    return (uut >= miopenUnitBothGPU && uut <= miopenUnitOptimizedCPU);
}

constexpr bool isValidREF(miopenTestReference_t ref)
{
    return (ref >= miopenTestReferenceNaiveGPU && ref <= miopenTestReferenceNaiveCPU);
}

constexpr bool checkReferenceChoice(miopenUnitUnderTest_t uut, miopenTestReference_t ref)
{
    switch(uut)
    {
    case miopenUnitNaiveGPU: [[fallthrough]];
    case miopenUnitBothGPU: return (ref >= miopenTestReferenceOptimizedCPU);
    case miopenUnitOptimizedGPU: return (ref >= miopenTestReferenceNaiveGPU);
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
    default: return miopenTestReferenceNaiveCPU; // Maybe to do something else here? Throw error?
    }
}

constexpr int numberOfRunsAfterFailure = 5;
/**
 * Base class for gtests, it provides interface to have different runs based on template parameters
 * that will determine the configuration for run.
 *
 * (UnitUnderTest)    UUT - desired unit that will be tested
 * (TestReference)    REF - desired implementation that will be treated as reference
 * (AfterTestFailure) ATF - option that determine what to do (if anything) on test failure
 */
template <miopenUnitUnderTest_t UUT, miopenTestReference_t REF, miopenAfterTestFailure_t ATF>
class GTESTBase
{
private:
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
                      << " number of runs failed." << std::endl;
            for(auto [key, value] : errors)
            {
                double meanError = std::reduce(value.begin(), value.begin() + numOfRunsFailed) /
                                   static_cast<double>(numOfRunsFailed);
                double maxError =
                    *(std::max_element(value.begin(), value.begin() + numOfRunsFailed));

                std::cout << key << ": " << std::endl
                          << "\tmean error: " << std::to_string(meanError) << std::endl
                          << "\tmax error: " << std::to_string(maxError) << std::endl;
            }
        }
    };

    miopenTestReference_t currentREF = REF;

    bool testOK = true;
    std::unordered_map<std::string, double> failureErrors;
    ErrorAnalysisInfo info;

protected:
    /**
     * Should we avoid implementing some logic that will check whether or not some UUT is
     * impelmented or not? Mostly OptimizedCPU and NaiveGPU are implemented, but not all operations
     * have OptimizedGPU. Can we consider for start that it is up to developer to take care of this.
     * For example it is resposibility of developer to not make OptimizedGPU as UUT if it is not
     * implemented. Also, we can assume that if there is OG then for sure there is NG, if there is
     * NG then for sure is OC, etc.
     *
     * These should be able to be called several times without invoking SetUp again.
     *
     * Or some checks can be implemented in test itself, maybe in SetUpTestSuite?
     */
    virtual void runOptimizedGPU() = 0;
    virtual void runNaiveGPU()     = 0;
    virtual void runOptimizedCPU() = 0;
    virtual void runNaiveCPU()     = 0;

    /**
     * Use EXPECT_* instead of ASSERT_* in verifying functions so that on failure execution can
     * continue and perform additional work after falure
     *
     * Return value is pair where:
     * first [bool] is false if some verifying failed, true otherwise
     * second [unordered_map] map should contain small name of the value(s) that failed and error
     * value(s)
     */
    virtual std::pair<bool, std::unordered_map<std::string, double>> verifyOptimizedGPU() = 0;
    virtual std::pair<bool, std::unordered_map<std::string, double>> verifyNaiveGPU()     = 0;
    virtual std::pair<bool, std::unordered_map<std::string, double>> verifyOptimizedCPU() = 0;

    /**
     * Run numberOfRunsAfterFailure times with selected configuration, do not change the reference,
     * after runs extract some statistical information
     */
    void analyzeAfterTestFailure()
    {
        std::cout << "Doing additional runs with selected configuration." << std::endl
                  << "Collecting more information." << std::endl;

        for(int i = 0; i < numberOfRunsAfterFailure; i++)
        {
            runUUT();
            runREF();
            verify();

            if(!testOK)
            {
                for(auto [key, value] : failureErrors)
                {
                    if(!info.errors.contains(key))
                    {
                        info.errors.insert({key, std::array<double, numberOfRunsAfterFailure>()});
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
     * the process until verification succeeded or we've run with all of the references.
     */
    void moveOnAfterTestFailure()
    {
        std::cout << "Moving onto more trusted reference implementation." << std::endl;
        /*
        if(currentREF == miopenTestReferenceNaiveGPU)
        {
            // TODO: in future implementations this will probably mean that all data is on GPU
            // therefore copy to cpu is needed.
        }
        */

        while(currentREF != miopenTestReferenceNaiveCPU)
        {
            currentREF = getNextREF(currentREF);
            // Print choosen reference? Or is it enough to have prints at the end of verifying
            // methods?

            // Do we need to re-run UUT?
            runREF();
            verify();

            if(testOK)
            {
                break;
            }
        }
    }

    void runUUT()
    {
        if constexpr(UUT == miopenUnitBothGPU)
        {
            runOptimizedGPU();
            runNaiveGPU();
        }
        else if constexpr(UUT == miopenUnitOptimizedGPU)
        {
            runOptimizedGPU();
        }
        else if constexpr(UUT == miopenUnitNaiveGPU)
        {
            runNaiveGPU();
        }
        else
        {
            runOptimizedCPU();
        }
    }

    void runREF()
    {
        if(currentREF == miopenTestReferenceNaiveGPU)
        {
            runNaiveGPU();
        }
        else if(currentREF == miopenTestReferenceOptimizedCPU)
        {
            runOptimizedCPU();
        }
        else
        {
            runNaiveCPU();
        }
    }

    void verify()
    {
        bool optimizedGPU_OK = true;
        bool naiveGPU_OK     = true;
        bool optimizedCPU_OK = true;
        if constexpr(UUT == miopenUnitBothGPU)
        {
            std::tie(optimizedGPU_OK, failureErrors) = verifyOptimizedGPU();
            std::tie(naiveGPU_OK, failureErrors)     = verifyNaiveGPU();
        }
        else if constexpr(UUT == miopenUnitOptimizedGPU)
        {
            std::tie(optimizedGPU_OK, failureErrors) = verifyOptimizedGPU();
        }
        else if constexpr(UUT == miopenUnitNaiveGPU)
        {
            std::tie(naiveGPU_OK, failureErrors) = verifyNaiveGPU();
        }
        else
        {
            std::tie(optimizedCPU_OK, failureErrors) = verifyOptimizedCPU();
        }
        testOK = optimizedGPU_OK && naiveGPU_OK && optimizedCPU_OK;
    }

public:
    void runTest()
    {
        runUUT();
        runREF();
        verify();

        if(!testOK && ATF != miopenAfterTestFailureNone)
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

    virtual ~GTESTBase() {}
};
