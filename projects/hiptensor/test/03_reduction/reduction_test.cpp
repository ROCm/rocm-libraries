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

#include "common.hpp"
#include "data_types.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "reduction/reduction_cpu_reference.hpp"
#include "reduction_test.hpp"
#include "util.hpp"
#include "utils.hpp"

namespace
{

    template <typename T>
    void printReductionTestInputOutput(std::ostream&                 stream,
                                       hiptensor::ReductionResource* resource,
                                       size_t                        elementsA,
                                       size_t                        elementsC,
                                       size_t                        elementsD)
    {
        stream << "Tensor A elements (" << elementsA << "):\n";
        hiptensorPrintArrayElements<T>(stream, (T*)resource->hostA().get(), elementsA);
        stream << std::endl;

        stream << "Tensor C elements (" << elementsC << "):\n";
        hiptensorPrintArrayElements<T>(stream, (T*)resource->hostC().get(), elementsC);
        stream << std::endl;

        stream << "Tensor D elements (" << elementsD << "):\n";
        hiptensorPrintArrayElements<T>(stream, (T*)resource->hostD().get(), elementsD);
        stream << std::endl;

        stream << "Refenrence elements (" << elementsD << "):\n";
        hiptensorPrintArrayElements<T>(stream, (T*)resource->hostReference().get(), elementsD);
        stream << std::endl;
    }
}
namespace hiptensor
{
    /*static*/ bool              ReductionTest::mHeaderPrinted = false;
    /*static*/ std::stringstream ReductionTest::sAPILogBuff    = std::stringstream();

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        ReductionTest::sAPILogBuff << msg;
    }

    ReductionTest::ReductionTest()
        : Base()
    {
        reset();
        // Handle our own outputs
        hiptensor::test::silenceLogger();
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Reduction kernels have different requirements
    // True = run test
    // False = skip test
    bool ReductionTest::checkDevice(hiptensorDataType_t          datatype,
                                    hiptensorComputeDescriptor_t computeDataType) const
    {
        return !(((datatype == HIPTENSOR_R_16F || datatype == HIPTENSOR_R_16BF
                   || computeDataType == HIPTENSOR_COMPUTE_DESC_16F
                   || computeDataType == HIPTENSOR_COMPUTE_DESC_16BF)
                  && !isF16Supported())
                 || ((datatype == HIPTENSOR_R_32F || computeDataType == HIPTENSOR_COMPUTE_DESC_32F)
                     && !isF32Supported())
                 || ((datatype == HIPTENSOR_R_64F || computeDataType == HIPTENSOR_COMPUTE_DESC_64F)
                     && !isF64Supported()));
    }

    bool ReductionTest::checkSizes() const
    {
        return true;
    }

    void ReductionTest::reset()
    {
        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalGBytes = mGBytesPerSec = 0.0;
    }

    ReductionResource* ReductionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    std::ostream& ReductionTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream
            << "TypeIn, "               // 1
            << "TypeCompute, "          // 2
            << "OperatorA, "            // 3
            << "OperatorC, "            // 4
            << "OperatorReduce, "       // 5
            << "LogLevel, "             // 6
            << "Lengths, "              // 7
            << "memoryLayout, "         // 8
            << "ReOrder, "              // 9
            << "Alpha, "                // 10
            << "Beta, "                 // 11
            << "ElapsedMs, "            // 12
            << "Problem Size(GFlops), " // 13
            << "TFlops/s, "             // 14
            << "TotalGBytes, "          // 15
            << "GBytes/s, "             // 16
            << "Result"                 // 17
            << std::endl;
        // clang-format on
    }

    std::ostream& ReductionTest::printKernel(std::ostream& stream) const
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto outputDims   = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto beta         = std::get<5>(param);
        auto op           = std::get<6>(param);
        auto aOp          = op[0];
        auto cOp          = op[1];
        auto reduceOp     = op[2];
        auto memoryLayout = std::get<7>(param);

        // clang-format off
        stream << hipTypeToString(testType[0]) << ", "                           //1
               << computeTypeToString(convertToComputeType(testType[1])) << ", " //2
               << opTypeToString(aOp) << ", "                                    //3
               << opTypeToString(cOp) << ", "                                    //4
               << opTypeToString(reduceOp) << ", "                               //5
               << logLevelToString(logLevel) << ", ";                            //6
        printContainerInCsv(lengths, stream) << ", ";                            //7
        stream << hipMemoryLayoutToString(memoryLayout) << ", ";                 //8
        printContainerInCsv(outputDims, stream) << ", ";                         //9
        stream << alpha << ", "                                                  //10
            << beta << ", ";                                                     //11
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream << "n/a" << ", " //12
                   << "n/a" << ", " //13
                   << "n/a" << ", " //14
                   << "n/a" << ", " //15
                   << "n/a" << ", " //16
                   << "SKIPPED"     //17
                   << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream << mElapsedTimeMs << ", "     // 12
                << mTotalGFlops << ", "          // 13
                << mMeasuredTFlopsPerSec << ", " // 14
                << mTotalGBytes << ", "          // 15
                << mGBytesPerSec << ", "         // 16
                << result                        // 17
                << std::endl;
            // clang-format on
        }

        return stream;
    }

    void ReductionTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto outputDims   = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto beta         = std::get<5>(param);
        auto op           = std::get<6>(param);
        auto memoryLayout = std::get<7>(param);
        auto aOp          = op[0];
        auto cOp          = op[1];
        auto reduceOp     = op[2];

        EXPECT_TRUE((lengths.size() > 0) && (lengths.size() <= 6));
        EXPECT_TRUE((outputDims.size() >= 0) && (outputDims.size() <= 6));

        EXPECT_TRUE((reduceOp == HIPTENSOR_OP_ADD) || (reduceOp == HIPTENSOR_OP_MUL)
                    || (reduceOp == HIPTENSOR_OP_MAX) || (reduceOp == HIPTENSOR_OP_MIN));

        EXPECT_EQ(dataTypes.size(), 2); // HIPTENSOR_R_16F or HIPTENSOR_R_32F
        auto acDataType      = dataTypes[0];
        auto computeDataType = convertToComputeType(dataTypes[1]);
        EXPECT_TRUE(
            (acDataType == HIPTENSOR_R_16F && computeDataType == HIPTENSOR_COMPUTE_DESC_16F)
            || (acDataType == HIPTENSOR_R_16F && computeDataType == HIPTENSOR_COMPUTE_DESC_32F)
            || (acDataType == HIPTENSOR_R_16BF && computeDataType == HIPTENSOR_COMPUTE_DESC_16BF)
            || (acDataType == HIPTENSOR_R_16BF && computeDataType == HIPTENSOR_COMPUTE_DESC_32F)
            || (acDataType == HIPTENSOR_R_32F && computeDataType == HIPTENSOR_COMPUTE_DESC_32F)
            || (acDataType == HIPTENSOR_R_64F && computeDataType == HIPTENSOR_COMPUTE_DESC_64F));

        mRunFlag &= checkDevice(acDataType, computeDataType);
        mRunFlag &= lengths.size() >= outputDims.size();

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            std::vector<size_t> outputLengths;
            for(auto dim : outputDims)
            {
                outputLengths.push_back(lengths[dim]);
            }
            getResource()->setupStorage(lengths, outputLengths, acDataType);

            // set mPrintElements to true to print element
            mPrintElements = false;
        }
    }

    void ReductionTest::reportResults(std::ostream&       stream,
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

                auto param        = Base::GetParam();
                auto dataTypes    = std::get<0>(param);
                auto logLevel     = std::get<1>(param);
                auto lengths      = std::get<2>(param);
                auto outputDims   = std::get<3>(param);
                auto alpha        = std::get<4>(param);
                auto beta         = std::get<5>(param);
                auto op           = std::get<6>(param);
                auto memoryLayout = std::get<7>(param);
                auto aOp          = op[0];
                auto cOp          = op[1];
                auto reduceOp     = op[2];

                stream << "Input [type: " << dataTypes << ", lengths: " << lengths
                       << ", memoryLayout: " << memoryLayout << ", outputDims: " << outputDims
                       << ", alpha: " << alpha << ", beta: " << beta << ", opReduce: [" << aOp
                       << ", " << cOp << ", " << reduceOp << "]\n";

                size_t elementsA = resource->getCurrentInputElementCount();
                size_t elementsC = resource->getCurrentOutputElementCount();
                size_t elementsD = elementsC;

                if(dataType == HIPTENSOR_R_16BF)
                {
                    printReductionTestInputOutput<bfloat16_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
                else if(dataType == HIPTENSOR_R_16F)
                {
                    printReductionTestInputOutput<float16_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
                else if(dataType == HIPTENSOR_R_32F)
                {
                    printReductionTestInputOutput<float32_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
                else if(dataType == HIPTENSOR_R_64F)
                {
                    printReductionTestInputOutput<float64_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
            }
        }
    }

    void ReductionTest::RunKernel()
    {
        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto outputDims   = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto beta         = std::get<5>(param);
        auto op           = std::get<6>(param);
        auto memoryLayout = std::get<7>(param);
        auto aOp          = op[0];
        auto cOp          = op[1];
        auto reduceOp     = op[2];

        auto acDataType      = dataTypes[0];
        auto computeDataType = convertToComputeType(dataTypes[1]);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        auto resource = getResource();

        if(mRunFlag)
        {
            std::vector<int> modeA(lengths.size());
            std::iota(modeA.begin(), modeA.end(), 'a');
            std::vector<int> modeC(outputDims.cbegin(), outputDims.cend());
            std::transform(modeC.cbegin(), modeC.cend(), modeC.begin(), [&modeA](auto dim) {
                return modeA[dim];
            });
            std::vector<int> modeD(modeC);

            int nmodeA = modeA.size();
            int nmodeC = modeC.size();
            int nmodeD = nmodeC;

            std::vector<int64_t> extentA(lengths.cbegin(), lengths.cend());
            std::vector<int64_t> extentCD(outputDims.cbegin(), outputDims.cend());
            std::transform(extentCD.cbegin(),
                           extentCD.cend(),
                           extentCD.begin(),
                           [&lengths](auto dim) { return lengths[dim]; });

            std::vector<int64_t> stridesA
                = memoryLayout == HIPTENSOR_MEMORY_LAYOUT_DEFAULT
                      ? std::vector<int64_t>{}
                      : hiptensor::stridesFromLengths(
                            extentA, memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);

            std::vector<int64_t> stridesCD
                = memoryLayout == HIPTENSOR_MEMORY_LAYOUT_DEFAULT
                      ? std::vector<int64_t>{}
                      : hiptensor::stridesFromLengths(
                            extentCD, memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);

            hiptensorStatus_t err;
            hiptensorHandle_t handle;
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(logLevel));

            hiptensorTensorDescriptor_t descA = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                stridesA.empty() ? nullptr : stridesA.data(),
                                                acDataType,
                                                0));

            hiptensorTensorDescriptor_t descC = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentCD.data(),
                                                stridesCD.empty() ? nullptr : stridesCD.data(),
                                                acDataType,
                                                0));

            hiptensorTensorDescriptor_t descD = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descD,
                                                nmodeD,
                                                extentCD.data(),
                                                stridesCD.empty() ? nullptr : stridesCD.data(),
                                                acDataType,
                                                0));

            hiptensorComputeDescriptor_t const descCompute = convertToComputeType(dataTypes[1]);
            hiptensorOperationDescriptor_t     desc;
            CHECK_HIPTENSOR_ERROR(hiptensorCreateReduction(handle,
                                                           &desc,
                                                           descA,
                                                           modeA.data(),
                                                           aOp,
                                                           descC,
                                                           modeC.data(),
                                                           cOp,
                                                           descD,
                                                           modeD.data(),
                                                           reduceOp,
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

            uint64_t                            worksize      = 0;
            const hiptensorWorksizePreference_t workspacePref = HIPTENSOR_WORKSPACE_DEFAULT;
            CHECK_HIPTENSOR_ERROR(
                hiptensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &worksize));

            resource->setupWorkspace(worksize);

            hiptensorPlan_t plan;
            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

            void*  work = resource->deviceWorkspace().get();
            double alphaValue{};
            double betaValue{};
            writeVal(&alphaValue, computeDataType, {computeDataType, alpha});
            writeVal(&betaValue, computeDataType, {computeDataType, beta});

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorReduce(handle,
                                                  plan,
                                                  (const void*)&alphaValue,
                                                  resource->deviceA().get(),
                                                  (const void*)&betaValue,
                                                  resource->deviceC().get(),
                                                  resource->deviceD().get(),
                                                  work,
                                                  worksize,
                                                  0));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t sizeA = std::accumulate(extentA.begin(),
                                           extentA.end(),
                                           hiptensorDataTypeSize(acDataType),
                                           std::multiplies<size_t>());

            size_t sizeCD = std::accumulate(extentCD.begin(),
                                            extentCD.end(),
                                            hiptensorDataTypeSize(acDataType),
                                            std::multiplies<size_t>());

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = sizeA / hiptensorDataTypeSize(acDataType) * 1e-9;
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            mTotalGBytes = sizeA + sizeCD;
            mTotalGBytes += (betaValue != 0.0) ? sizeCD : 0;
            mTotalGBytes /= 1e9;
            mGBytesPerSec = mTotalGBytes / (mElapsedTimeMs * 1e-3);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())

            {
                resource->copyOutputToHost();

                CHECK_HIPTENSOR_ERROR(hiptensorReductionReference(&alphaValue,
                                                                  resource->hostA().get(),
                                                                  descA,
                                                                  modeA.data(),
                                                                  aOp,
                                                                  &betaValue,
                                                                  resource->hostC().get(),
                                                                  descC,
                                                                  modeC.data(),
                                                                  cOp,
                                                                  resource->hostReference().get(),
                                                                  descD,
                                                                  modeD.data(),
                                                                  reduceOp,
                                                                  computeDataType,
                                                                  0 /* stream */));
                resource->copyReferenceToDevice();

                if(acDataType == HIPTENSOR_R_16F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float16_t>(
                            (float16_t*)resource->deviceD().get(),
                            (float16_t*)resource->deviceReference().get(),
                            resource->getCurrentOutputElementCount(),
                            computeDataType);
                }
                else if(acDataType == HIPTENSOR_R_16BF)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<bfloat16_t>(
                            (bfloat16_t*)resource->deviceD().get(),
                            (bfloat16_t*)resource->deviceReference().get(),
                            resource->getCurrentOutputElementCount(),
                            computeDataType);
                }
                else if(acDataType == HIPTENSOR_R_32F)
                {
                    auto reducedSize = resource->getCurrentInputElementCount()
                                       / resource->getCurrentOutputElementCount();
                    double tolerance = reducedSize * getEpsilon(computeDataType);
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float32_t>(
                            (float32_t*)resource->deviceD().get(),
                            (float32_t*)resource->deviceReference().get(),
                            resource->getCurrentOutputElementCount(),
                            computeDataType,
                            tolerance);
                }
                else if(acDataType == HIPTENSOR_R_64F)
                {
                    auto reducedSize = resource->getCurrentInputElementCount()
                                       / resource->getCurrentOutputElementCount();
                    double tolerance = reducedSize * getEpsilon(computeDataType);
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float64_t>(
                            (float64_t*)resource->deviceD().get(),
                            (float64_t*)resource->deviceReference().get(),
                            resource->getCurrentOutputElementCount(),
                            computeDataType,
                            tolerance);
                }

                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
            } // if (testOptions->performValidation())

            using Options        = hiptensor::HiptensorOptions;
            auto& loggingOptions = Options::instance();

            if(!loggingOptions->omitCout())
            {
                std::cout << ReductionTest::sAPILogBuff.str();
                reportResults(std::cout,
                              acDataType,
                              mHeaderPrinted,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
            }

            if(loggingOptions->logOstream().isOpen())
            {
                loggingOptions->logOstream().fstream() << ReductionTest::sAPILogBuff.str();
            }

            if(loggingOptions->ostream().isOpen())
            {
                reportResults(loggingOptions->ostream().fstream(),
                              acDataType,
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

            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
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
    }

    void ReductionTest::TearDown() {}

} // namespace hiptensor
