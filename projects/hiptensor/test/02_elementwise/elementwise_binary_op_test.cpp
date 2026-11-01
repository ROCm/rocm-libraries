/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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
#include <hiptensor/hiptensor.h>

#include <algorithm>

#include "ck_tensor_limits.hpp"
#include "common.hpp"
#include "data_types.hpp"
#include "elementwise/elementwise_cpu_reference.hpp"
#include "elementwise_binary_op_test.hpp"
#include "elementwise_type_dispatch.hpp"
#include "hiptensor_options.hpp"
#include "util.hpp"
#include "utils.hpp"

namespace hiptensor
{
    /*static*/ bool              ElementwiseBinaryOpTest::mHeaderPrinted = false;
    /*static*/ std::stringstream ElementwiseBinaryOpTest::sAPILogBuff    = std::stringstream();

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        ElementwiseBinaryOpTest::sAPILogBuff << msg;
    }

    ElementwiseBinaryOpTest::ElementwiseBinaryOpTest()
        : Base()
    {
        reset();

        // Handle our own outputs
        hiptensor::test::silenceLogger();
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different ElementwiseBinaryOp kernels have different requirements
    // True = run test
    // False = skip test
    bool ElementwiseBinaryOpTest::checkDevice(hiptensorDataType_t datatype) const
    {
        return isElementwiseTypeSupportedOnDevice(datatype);
    }

    bool ElementwiseBinaryOpTest::checkSizes() const
    {
        return true;
    }

    void ElementwiseBinaryOpTest::reset()
    {
        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs        = 0.0;
        mTotalGFlops          = 0.0;
        mMeasuredTFlopsPerSec = 0.0;
        mTotalGBytes          = 0.0;
        mGBytesPerSec         = 0.0;
    }

    std::ostream& ElementwiseBinaryOpTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream << "TypeIn, "     // 1
            << "TypeCompute, "          // 2
            << "Operators, "            // 3
            << "LogLevel, "             // 4
            << "Lengths, "              // 5
            << "memoryLayout, "         // 6
            << "PermutedOrder, "        // 7
            << "Alpha, "                // 8
            << "Gamma, "                // 9
            << "ElapsedMs, "            // 10
            << "Problem Size(GFlops), " // 11
            << "TFlops/s, "             // 12
            << "TotalGBytes, "          // 13
            << "GBytes/s, "             // 14
            << "Result"                 // 15
            << std::endl;
        // clang-format on
    }

    std::ostream& ElementwiseBinaryOpTest::printKernel(std::ostream& stream) const
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto permutedDims = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto gamma        = std::get<5>(param);
        auto operators    = std::get<6>(param);
        auto memoryLayout = std::get<7>(param);

        // clang-format off
        stream << hipTypeToString(testType[0]) << ", "                                              // 1
            << computeTypeToString(convertToComputeType(testType[1])) << ", "                       // 2
            << "[ " << opTypeToString(operators[0]) << " " << opTypeToString(operators[1]) << " " << opTypeToString(operators[2]) << "], " // 3
            << logLevelToString(logLevel) << ", ";                                                  // 4
        printContainerInCsv(lengths, stream) << ", ";                                               // 5
        stream << hipMemoryLayoutToString(memoryLayout) << ", ";                                    // 6
        printContainerInCsv(permutedDims, stream) << ", ";                                          // 7
        stream << alpha << ", ";                                                                    // 8
        stream << gamma << ", ";                                                                    // 9
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream << "n/a" << ", " // 10
                << "n/a" << ", "    // 11
                << "n/a" << ", "    // 12
                << "n/a" << ", "    // 13
                << "n/a" << ", "    // 14
                << "SKIPPED"        // 15
                << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream << mElapsedTimeMs << ", "     // 10
                << mTotalGFlops << ", "          // 11
                << mMeasuredTFlopsPerSec << ", " // 12
                << mTotalGBytes << ", "          // 13
                << mGBytesPerSec << ", "         // 14
                << result                        // 15
                << std::endl;
            // clang-format on
        }

        return stream;
    }

    ElementwiseResource* ElementwiseBinaryOpTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    void ElementwiseBinaryOpTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto permutedDims = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto gamma        = std::get<5>(param);
        auto operators    = std::get<6>(param);
        auto memoryLayout = std::get<7>(param);

        EXPECT_TRUE((lengths.size() > 1) && (lengths.size() <= 6));
        EXPECT_TRUE((permutedDims.size() > 1) && (permutedDims.size() <= 6));

        EXPECT_EQ(operators.size(), 3);
        EXPECT_TRUE((operators[0] == HIPTENSOR_OP_IDENTITY) || (operators[0] == HIPTENSOR_OP_NEG));
        EXPECT_TRUE((operators[1] == HIPTENSOR_OP_IDENTITY) || (operators[1] == HIPTENSOR_OP_NEG));
        EXPECT_TRUE(operators[2] == HIPTENSOR_OP_ADD);

        EXPECT_EQ(dataTypes.size(), 2);
        auto dataType = dataTypes[0];

        mRunFlag = mRunFlag && checkDevice(dataType);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            getResource()->setupStorage(
                lengths, dataType, ElementwiseResource::ElementwiseOp::BINARY_OP);

            // set mPrintElements to true to print element
            mPrintElements = false;
        }
    }

    void ElementwiseBinaryOpTest::reportResults(std::ostream&       stream,
                                                hiptensorDataType_t dataType,
                                                bool                omitHeader,
                                                bool                omitSkipped,
                                                bool                omitFailed,
                                                bool                omitPassed) const
    {
        if(!omitHeader)
        {
            printHeader(stream);
        }

        // Conditionally print outputs
        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            printKernel(stream);

            if(mPrintElements)
            {
                auto resource = getResource();
                printElementwiseTensors(stream,
                                        dataType,
                                        resource->getCurrentMatrixElement(),
                                        {{"A", resource->hostInput1().get()},
                                         {"C", resource->hostInput2().get()},
                                         {"D", resource->hostOutput().get()},
                                         {"ref", resource->hostReference().get()}});
            }
        }
    }

    void ElementwiseBinaryOpTest::RunKernel()
    {
        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto permutedDims = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto gamma        = std::get<5>(param);
        auto operators    = std::get<6>(param);
        auto memoryLayout = std::get<7>(param);

        auto dataType        = dataTypes[0];
        auto computeDataType = dataTypes[1];

        auto Aop  = operators[0];
        auto Cop  = operators[1];
        auto ACop = operators[2];

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        auto resource = getResource();

        if(mRunFlag)
        {
            /**********************
              B_{w, h, c, n} = 1.0 *  \textsl{IDENTITY}(A_{c, n, h, w})
             **********************/

            int nDim     = lengths.size();
            int arrDim[] = {'n', 'c', 'w', 'h', 'd', 'm'};

            std::vector<int> modeA(arrDim, arrDim + nDim);
            std::vector<int> modeC(arrDim, arrDim + nDim);
            std::vector<int> modeD;
            for(auto dim : permutedDims)
            {
                modeD.push_back(modeA[dim]);
            }

            int                              nmodeA = modeA.size();
            int                              nmodeC = modeC.size();
            int                              nmodeD = modeD.size();
            std::unordered_map<int, int64_t> extent;
            for(int i = 0; i < modeA.size(); i++)
            {
                extent[modeA[i]] = lengths[i];
            }

            std::vector<int64_t> extentA;
            for(auto mode : modeA)
                extentA.push_back(extent[mode]);
            std::vector<int64_t> extentC = extentA;
            std::vector<int64_t> extentD;
            for(auto mode : modeD)
                extentD.push_back(extent[mode]);

            std::vector<int64_t> stridesIn
                = memoryLayout == HIPTENSOR_MEMORY_LAYOUT_DEFAULT
                      ? std::vector<int64_t>{}
                      : hiptensor::stridesFromLengths(
                            extentA, memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);
            std::vector<int64_t> stridesD
                = memoryLayout == HIPTENSOR_MEMORY_LAYOUT_DEFAULT
                      ? std::vector<int64_t>{}
                      : hiptensor::stridesFromLengths(
                            extentD, memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);

            hiptensorHandle_t handle;
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(
                logLevel
                & ~HIPTENSOR_LOG_LEVEL_PERF_TRACE)); // keep ERROR/API_TRACE diagnostics but strip
                                                     // PERF_TRACE so the CK backend does not run its
                                                     // internal cold/hot timed loop

            hiptensorTensorDescriptor_t descA = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                stridesIn.empty() ? nullptr : stridesIn.data(),
                                                dataType,
                                                0));

            hiptensorTensorDescriptor_t descC = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                stridesIn.empty() ? nullptr : stridesIn.data(),
                                                dataType,
                                                0));

            hiptensorTensorDescriptor_t descD = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descD,
                                                nmodeD,
                                                extentD.data(),
                                                stridesD.empty() ? nullptr : stridesD.data(),
                                                dataType,
                                                0));

            hiptensorComputeDescriptor_t const descCompute = convertToComputeType(computeDataType);
            hiptensorOperationDescriptor_t     desc;
            CHECK_HIPTENSOR_ERROR(hiptensorCreateElementwiseBinary(handle,
                                                                   &desc,
                                                                   descA,
                                                                   modeA.data(),
                                                                   Aop,
                                                                   descC,
                                                                   modeC.data(),
                                                                   Cop,
                                                                   descD,
                                                                   modeD.data(),
                                                                   ACop,
                                                                   descCompute));

            const hiptensorAlgo_t     algo = HIPTENSOR_ALGO_DEFAULT;
            hiptensorPlanPreference_t planPref;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreatePlanPreference(handle, &planPref, algo, HIPTENSOR_JIT_MODE_NONE));

            /**************************
            * Disable Plan Cache for tests
            ***************************/
            const hiptensorCacheMode_t cacheMode = HIPTENSOR_CACHE_MODE_NONE;
            CHECK_HIPTENSOR_ERROR(
                hiptensorPlanPreferenceSetAttribute(handle,
                                                    planPref,
                                                    HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE,
                                                    &cacheMode,
                                                    sizeof(hiptensorCacheMode_t)));

            hiptensorPlan_t plan = nullptr;
            auto            planStatus
                = hiptensorCreatePlan(handle, &plan, desc, planPref, 0 /*workspaceSizeEstimate*/);
            // No plan is created for tensors the kernels cannot address, which the bench
            // configurations include.
            if(planStatus == HIPTENSOR_STATUS_NOT_SUPPORTED && exceedsCkTensorLimits(*desc))
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descA));
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descC));
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descD));
                GTEST_SKIP() << "Tensor size not supported by the kernels";
            }
            CHECK_HIPTENSOR_ERROR(planStatus);

            double alphaValue{};
            double gammaValue{};
            if(!writeElementwiseScalar(alphaValue, computeDataType, alpha)
               || !writeElementwiseScalar(gammaValue, computeDataType, gamma))
            {
                FAIL() << "Unexpected compute data type: " << computeDataType;
            }

            auto& opts     = HiptensorOptions::instance();
            int   coldRuns = opts->coldRuns();
            int   hotRuns  = std::max(1, opts->hotRuns()); // guard hot_runs <= 0
            // untimed warm-up: ramps clocks, primes plan/caches
            for(int i = 0; i < coldRuns; ++i)
                CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryExecute(handle,
                                                                        plan,
                                                                        (void*)&alphaValue,
                                                                        resource->deviceInput1().get(),
                                                                        (void*)&gammaValue,
                                                                        resource->deviceInput2().get(),
                                                                        resource->deviceOutput().get(),
                                                                        nullptr /* stream */));

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            for(int i = 0; i < hotRuns; ++i) // timed: M clean launches
                CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryExecute(handle,
                                                                        plan,
                                                                        (void*)&alphaValue,
                                                                        resource->deviceInput1().get(),
                                                                        (void*)&gammaValue,
                                                                        resource->deviceInput2().get(),
                                                                        resource->deviceOutput().get(),
                                                                        nullptr /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t sizeA = std::accumulate(extentA.begin(),
                                           extentA.end(),
                                           size_t(hiptensorDataTypeSize(dataType)),
                                           std::multiplies<size_t>());

            size_t sizeC = std::accumulate(extentC.begin(),
                                           extentC.end(),
                                           size_t(hiptensorDataTypeSize(dataType)),
                                           std::multiplies<size_t>());

            size_t sizeD = std::accumulate(extentD.begin(),
                                           extentD.end(),
                                           size_t(hiptensorDataTypeSize(dataType)),
                                           std::multiplies<size_t>());

            mElapsedTimeMs        = float64_t(timeMs) / hotRuns;
            mTotalGFlops          = 5.0 * (resource->getCurrentMatrixElement()) * 1e-9;
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            mTotalGBytes = sizeA + sizeC + sizeD;
            mTotalGBytes /= 1e9;
            mGBytesPerSec = mTotalGBytes / (mElapsedTimeMs * 1e-3);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            resource->copyOutputToHost();

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())
            {
                resource->copyOutputToHost();

                CHECK_HIPTENSOR_ERROR(
                    hiptensorElementwiseBinaryOpReference(&alphaValue,
                                                          resource->hostInput1().get(),
                                                          descA,
                                                          modeA.data(),
                                                          Aop,
                                                          &gammaValue,
                                                          resource->hostInput2().get(),
                                                          descC,
                                                          modeC.data(),
                                                          Cop,
                                                          resource->hostReference().get(),
                                                          descD,
                                                          modeD.data(),
                                                          ACop,
                                                          computeDataType,
                                                          0 /* stream */));

                resource->copyReferenceToDevice();
                std::tie(mValidationResult, mMaxRelativeError)
                    = compareElementwiseOutputWithReference(resource, dataType, computeDataType);
                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
            } // if (testOptions->performValidation())

            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));

            if(descA)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descA));
                descA = nullptr;
            }
            if(descC)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descC));
                descC = nullptr;
            }
            if(descD)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descD));
                descD = nullptr;
            }
        }

        using Options        = hiptensor::HiptensorOptions;
        auto& loggingOptions = Options::instance();

        if(!loggingOptions->omitCout())
        {
            std::cout << ElementwiseBinaryOpTest::sAPILogBuff.str();
            reportResults(std::cout,
                          dataType,
                          mHeaderPrinted,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
        }

        if(loggingOptions->logOstream().isOpen())
        {
            loggingOptions->logOstream().fstream() << ElementwiseBinaryOpTest::sAPILogBuff.str();
        }

        if(loggingOptions->ostream().isOpen())
        {
            reportResults(loggingOptions->ostream().fstream(),
                          dataType,
                          mHeaderPrinted,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
        }

        // Print the header only once
        if(!mHeaderPrinted)
        {
            mHeaderPrinted = true;
        }
    }

    void ElementwiseBinaryOpTest::TearDown() {}

} // namespace hiptensor
