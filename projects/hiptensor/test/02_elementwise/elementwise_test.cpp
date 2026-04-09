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
#include "elementwise/elementwise_cpu_reference.hpp"
#include "elementwise_test.hpp"
#include "hiptensor_options.hpp"
#include "util.hpp"
#include "utils.hpp"

namespace hiptensor
{
    /*static*/ bool              PermutationTest::mHeaderPrinted = false;
    /*static*/ std::stringstream PermutationTest::sAPILogBuff    = std::stringstream();

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        PermutationTest::sAPILogBuff << msg;
    }

    PermutationTest::PermutationTest()
        : Base()
    {
        reset();

        // Handle our own outputs
        hiptensor::test::silenceLogger();
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Permutation kernels have different requirements
    // True = run test
    // False = skip test
    bool PermutationTest::checkDevice(hiptensorDataType_t datatype) const
    {
        return (isF16Supported() && (datatype == HIPTENSOR_R_16F))
               || (isF32Supported() && (datatype == HIPTENSOR_R_32F));
    }

    bool PermutationTest::checkSizes() const
    {
        return true;
    }

    void PermutationTest::reset()
    {
        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalGBytes = mGBytesPerSec = 0.0;
    }

    std::ostream& PermutationTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream << "TypeIn, "     // 1
            << "TypeCompute, "          // 2
            << "Operators, "            // 3
            << "LogLevel, "             // 4
            << "Lengths, "              // 5
            << "MemoryLayout, "         // 6
            << "PermutedOrder, "        // 7
            << "Alpha, "                // 8
            << "ElapsedMs, "            // 9
            << "Problem Size(GFlops), " // 10
            << "TFlops/s, "             // 11
            << "TotalGBytes, "          // 12
            << "GBytes/s, "             // 13
            << "Result"                 // 14
            << std::endl;
        // clang-format on
    }

    std::ostream& PermutationTest::printKernel(std::ostream& stream) const
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto permutedDims = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto operators    = std::get<5>(param);
        auto memoryLayout = std::get<6>(param);

        // clang-format off
        stream << hipTypeToString(testType[0]) << ", "                                              // 1
            << computeTypeToString(convertToComputeType(testType[1])) << ", "                       // 2
            << "[ " << opTypeToString(operators[0]) << "], "                                        // 3
            << logLevelToString(logLevel) << ", ";                                                  // 4
        printContainerInCsv(lengths, stream) << ", ";                                               // 5
        stream << hipMemoryLayoutToString(memoryLayout) << ", ";                                    // 6
        printContainerInCsv(permutedDims, stream) << ", ";                                          // 7
        stream << alpha << ", ";                                                                    // 8
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream << "n/a" << ", " // 9
                << "n/a" << ", "    // 10
                << "n/a" << ", "    // 11
                << "n/a" << ", "    // 12
                << "n/a" << ", "    // 13
                << "SKIPPED"        // 14
                << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream << mElapsedTimeMs << ", "     // 9
                << mTotalGFlops << ", "          // 10
                << mMeasuredTFlopsPerSec << ", " // 11
                << mTotalGBytes << ", "          // 12
                << mGBytesPerSec << ", "         // 13
                << result                        // 14
                << std::endl;
            // clang-format on
        }

        return stream;
    }

    ElementwiseResource* PermutationTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    void PermutationTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto permutedDims = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto operators    = std::get<5>(param);
        auto memoryLayout = std::get<6>(param);

        EXPECT_TRUE((lengths.size() > 1) && (lengths.size() <= 6));
        EXPECT_TRUE((permutedDims.size() > 1) && (permutedDims.size() <= 6));

        EXPECT_EQ(operators.size(), 1); // HIPTENSOR_OP_IDENTITY or HIPTENSOR_OP_NEG
        auto op = operators[0];
        EXPECT_TRUE((op == HIPTENSOR_OP_IDENTITY) || (op == HIPTENSOR_OP_NEG));

        EXPECT_EQ(dataTypes.size(), 2); // HIPTENSOR_R_16F or HIPTENSOR_R_32F
        auto abDataType = dataTypes[0];
        EXPECT_TRUE((abDataType == HIPTENSOR_R_16F) || (abDataType == HIPTENSOR_R_32F));

        mRunFlag &= checkDevice(abDataType);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            getResource()->setupStorage(
                lengths, abDataType, ElementwiseResource::ElementwiseOp::PERMUTATION);

            // set mPrintElements to true to print element
            mPrintElements = false;
        }
    }

    void PermutationTest::reportResults(std::ostream&       stream,
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

                size_t elementsA   = resource->getCurrentMatrixElement();
                size_t elementsB   = elementsA;
                size_t elementsRef = elementsA;

                if(dataType == HIPTENSOR_R_32F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostInput1().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements (" << elementsB << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostOutput().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor ref elements (" << elementsRef << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostReference().get(), elementsRef);
                    stream << std::endl;
                }
                else
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostInput1().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements (" << elementsB << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostOutput().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor ref elements (" << elementsRef << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostReference().get(), elementsRef);
                    stream << std::endl;
                }
            }
        }
    }

    void PermutationTest::RunKernel()
    {
        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto logLevel     = std::get<1>(param);
        auto lengths      = std::get<2>(param);
        auto permutedDims = std::get<3>(param);
        auto alpha        = std::get<4>(param);
        auto operators    = std::get<5>(param);
        auto memoryLayout = std::get<6>(param);

        auto abDataType      = dataTypes[0];
        auto computeDataType = dataTypes[1];

        auto Aop = operators[0];

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
            std::vector<int> modeB;
            for(auto dim : permutedDims)
            {
                modeB.push_back(modeA[dim]);
            }

            int                              nmodeA = modeA.size();
            int                              nmodeB = modeB.size();
            std::unordered_map<int, int64_t> extent;
            for(auto [modeIt, i] = std::tuple{modeA.begin(), 0}; modeIt != modeA.end();
                ++modeIt, ++i)
            {
                extent[*modeIt] = lengths[i];
            }

            std::vector<int64_t> extentA;
            for(auto mode : modeA)
                extentA.push_back(extent[mode]);
            std::vector<int64_t> extentB;
            for(auto mode : modeB)
                extentB.push_back(extent[mode]);

            std::vector<int64_t> stridesA
                = memoryLayout == HIPTENSOR_MEMORY_LAYOUT_DEFAULT
                      ? std::vector<int64_t>{}
                      : hiptensor::stridesFromLengths(
                            extentA, memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);
            std::vector<int64_t> stridesB
                = memoryLayout == HIPTENSOR_MEMORY_LAYOUT_DEFAULT
                      ? std::vector<int64_t>{}
                      : hiptensor::stridesFromLengths(
                            extentB, memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);

            //hiptensorStatus_t err;
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
                                                abDataType,
                                                0));

            hiptensorTensorDescriptor_t descB = nullptr;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreateTensorDescriptor(handle,
                                                &descB,
                                                nmodeB,
                                                extentB.data(),
                                                stridesB.empty() ? nullptr : stridesB.data(),
                                                abDataType,
                                                0));

            hiptensorComputeDescriptor_t const descCompute = convertToComputeType(computeDataType);
            hiptensorOperationDescriptor_t     desc;
            CHECK_HIPTENSOR_ERROR(hiptensorCreatePermutation(
                handle, &desc, descA, modeA.data(), Aop, descB, modeB.data(), descCompute));

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

            hiptensorPlan_t plan;
            CHECK_HIPTENSOR_ERROR(
                hiptensorCreatePlan(handle, &plan, desc, planPref, 0 /* workspaceSizeLimit */));

            float alphaValue{};
            if(computeDataType == HIPTENSOR_R_16F)
            {
                *(reinterpret_cast<_Float16*>(&alphaValue)) = static_cast<_Float16>(alpha);
            }
            else
            {
                *(reinterpret_cast<float*>(&alphaValue)) = static_cast<float>(alpha);
            }

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorPermute(handle,
                                                   plan,
                                                   &alphaValue,
                                                   resource->deviceInput1().get(),
                                                   resource->deviceOutput().get(),
                                                   nullptr /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t sizeA = std::accumulate(extentA.begin(),
                                           extentA.end(),
                                           hiptensorDataTypeSize(abDataType),
                                           std::multiplies<size_t>());

            size_t sizeB = std::accumulate(extentB.begin(),
                                           extentB.end(),
                                           hiptensorDataTypeSize(abDataType),
                                           std::multiplies<size_t>());

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = 2.0 * (resource->getCurrentMatrixElement()) * 1e-9;
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            mTotalGBytes = sizeA + sizeB;
            mTotalGBytes /= 1e9;
            mGBytesPerSec = mTotalGBytes / (mElapsedTimeMs * 1e-3);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            resource->copyOutputToHost();

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())
            {
                resource->copyOutputToHost();

                if(abDataType == HIPTENSOR_R_32F)
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorElementwisePermuteReference(
                        &alphaValue,
                        (const float*)resource->hostInput1().get(),
                        descA,
                        modeA.data(),
                        Aop,
                        (float*)resource->hostReference().get(),
                        descB,
                        modeB.data(),
                        computeDataType,
                        0 /* stream */));

                    resource->copyReferenceToDevice();
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float>((float*)resource->deviceOutput().get(),
                                                          (float*)resource->deviceReference().get(),
                                                          resource->getCurrentMatrixElement(),
                                                          convertToComputeType(computeDataType));
                }
                else if(abDataType == HIPTENSOR_R_16F)
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorElementwisePermuteReference(
                        &alphaValue,
                        (const _Float16*)resource->hostInput1().get(),
                        descA,
                        modeA.data(),
                        Aop,
                        (_Float16*)resource->hostReference().get(),
                        descB,
                        modeB.data(),
                        computeDataType,
                        0 /* stream */));

                    resource->copyReferenceToDevice();

                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<_Float16>(
                            (_Float16*)resource->deviceOutput().get(),
                            (_Float16*)resource->deviceReference().get(),
                            resource->getCurrentMatrixElement(),
                            convertToComputeType(computeDataType));
                }
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
            if(descB)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descB));
                descB = nullptr;
            }
        }

        using Options        = hiptensor::HiptensorOptions;
        auto& loggingOptions = Options::instance();

        if(!loggingOptions->omitCout())
        {
            std::cout << PermutationTest::sAPILogBuff.str();
            reportResults(std::cout,
                          abDataType,
                          mHeaderPrinted,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
        }

        if(loggingOptions->logOstream().isOpen())
        {
            loggingOptions->logOstream().fstream() << PermutationTest::sAPILogBuff.str();
        }

        if(loggingOptions->ostream().isOpen())
        {
            reportResults(loggingOptions->ostream().fstream(),
                          abDataType,
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

    void PermutationTest::TearDown() {}

} // namespace hiptensor
