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

#include "miopen/miopen.h"

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
    miopenAfterTestFailureAnalyze = 0, /*!< Analyze and provide more information. */
    miopenAfterTestFailureMoveOn  = 0, /*!< Move on next, more trusted, implementation. */
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

constexpr int numberOfRunsAfterFailure = 5;
struct ErrorAnalysisInfo
{
    int numOfRunsFailed = 0;
    std::array<double, numberOfRunsAfterFailure> errors;

    void analyze()
    {
        double percentageFailed = static_cast<double>(numOfRunsFailed) /
                                  static_cast<double>(numberOfRunsAfterFailure) * 100.0;
        double meanError = std::reduce(errors.begin(), errors.end()) /
                           static_cast<double>(numberOfRunsAfterFailure);
        double maxError = *(std::max_element(errors.begin(), errors.end()));

        std::cout << percentageFailed << std::endl
                  << meanError << std::endl
                  << maxError << std::endl;
    }
};

template <miopenUnitUnderTest_t UUT, miopenTestReference_t REF, miopenAfterTestFailure_t ATF>
class GTESTBase
{
private:
    bool testFailed = false;
    ErrorAnalysisInfo info;

protected:
    virtual miopenStatus_t runOptimizedGPU() = 0;
    virtual miopenStatus_t runNaiveGPU()     = 0;
    virtual miopenStatus_t runOptimizedCPU() = 0;
    virtual miopenStatus_t runNaiveCPU()     = 0;

    // Use EXPECT_* instead of ASSERT_* in verifying functions so that on failure execution can
    // continue and perform analysis in runAfterFailure()
    virtual std::pair<bool, std::unordered_map<std::string, double>> verifyOptimizedGPU() = 0;
    virtual std::pair<bool, std::unordered_map<std::string, double>> verifyNaiveGPU()     = 0;
    virtual std::pair<bool, std::unordered_map<std::string, double>> verifyOptimizedCPU() = 0;

    miopenStatus_t analyzeAfterTestFailure() { return miopenStatusNotImplemented; }

    miopenStatus_t moveOnAfterTestFailure() { return miopenStatusNotImplemented; }

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
        if constexpr(REF == miopenTestReferenceNaiveGPU)
        {
            runNaiveGPU();
        }
        else if constexpr(REF == miopenTestReferenceOptimizedCPU)
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
        bool failedOG = false;
        bool failedNG = false;
        bool failedOC = false;
        if constexpr(UUT == miopenUnitBothGPU)
        {
            std::tie(failedOG, std::ignore) = verifyOptimizedGPU().first;
            std::tie(failedNG, std::ignore) = verifyNaiveGPU();
        }
        else if constexpr(UUT == miopenUnitOptimizedGPU)
        {
            std::tie(failedOG, std::ignore) = verifyOptimizedGPU();
        }
        else if constexpr(UUT == miopenUnitNaiveGPU)
        {
            std::tie(failedNG, std::ignore) = verifyNaiveGPU();
        }
        else
        {
            std::tie(failedOC, std::ignore) = verifyOptimizedCPU();
        }
        testFailed = failedOG || failedNG || failedOC;
    }

public:
    void runTest()
    {
        runUUT();
        runREF();
        verify();

        if(testFailed && ATF != miopenAfterTestFailureNone)
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
                std::cerr << "Unknown afterFailure option" << std::endl;
            }
        }
    };

    virtual ~GTESTBase() {}
};
