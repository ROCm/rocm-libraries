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
#include <hiptensor/hiptensor.hpp>

#include "data_types.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "permutation/permutation_cpu_reference.hpp"
#include "permutation_test.hpp"
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
        hiptensorLoggerOpenFile("/dev/null");
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Permutation kernels have different requirements
    // True = run test
    // False = skip test
    bool PermutationTest::checkDevice(hipDataType datatype) const
    {
        return isF32Supported() && ((datatype == HIP_R_32F) || (datatype == HIP_R_16F));
    }

    bool PermutationTest::checkSizes() const
    {
        return true;
    }

    void PermutationTest::reset()
    {
        handle = nullptr;

        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalBytes = 0.0;
    }

    std::ostream& PermutationTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream << "TypeIn,"     // 1
            << "TypeCompute,"          // 2
            << "Operators,"            // 3
            << "LogLevel,"             // 4
            << "Lengths,"              // 5
            << "PermutedOrder,"        // 6
            << "Alpha,"                // 7
            << "ElapsedMs,"            // 8
            << "Problem Size(GFlops)," // 9
            << "TFlops,"               // 10
            << "TotalBytes,"           // 11
            << "Result"                // 12
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

        // clang-format off
        stream << hipTypeToString(testType[0]) << ","                                              // 1
            << computeTypeToString(convertToComputeType(testType[1])) << ","                       // 2
            << "[ " << opTypeToString(operators[0]) << " " << opTypeToString(operators[1]) << "]," // 3
            << logLevelToString(logLevel) << ",";                                                  // 4
        printContainerInCsv(lengths, stream) << ",";                                               // 5
        printContainerInCsv(permutedDims, stream) << ",";                                          // 6
        stream << alpha << ",";                                                                    // 7
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream << "n/a" << "," // 8
                << "n/a" << ","    // 9
                << "n/a" << ","    // 10
                << "n/a" << ","    // 11
                << "SKIPPED"       // 12
                << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream << mElapsedTimeMs << ","     //8
                << mTotalGFlops << ","          //9
                << mMeasuredTFlopsPerSec << "," //10
                << mTotalBytes << ","           //11
                << result                       //12
                << std::endl;
            // clang-format on
        }

        return stream;
    }

    PermutationResource* PermutationTest::getResource() const
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

        EXPECT_TRUE((lengths.size() > 1) && (lengths.size() <= 6));
        EXPECT_TRUE((permutedDims.size() > 1) && (permutedDims.size() <= 6));

        EXPECT_EQ(operators.size(), 2); // HIPTENSOR_OP_IDENTITY or HIPTENSOR_OP_SQRT
        auto op = operators[0];
        EXPECT_TRUE((op == HIPTENSOR_OP_IDENTITY) || (op == HIPTENSOR_OP_NEG));

        EXPECT_EQ(dataTypes.size(), 2); // HIP_R_16F or HIP_R_32F
        auto abDataType = dataTypes[0];
        EXPECT_TRUE((abDataType == HIP_R_16F) || (abDataType == HIP_R_32F));

        mRunFlag &= checkDevice(abDataType);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            getResource()->setupStorage(lengths, abDataType);

            // set mPrintElements to true to print element
            mPrintElements = false;
        }
    }

    void PermutationTest::reportResults(std::ostream& stream,
                                        hipDataType   dataType,
                                        bool          omitHeader,
                                        bool          omitSkipped,
                                        bool          omitFailed,
                                        bool          omitPassed) const
    {
        if(!omitHeader)
        {
            printHeader(stream);
        }

        // Conditionally print outputs
        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            stream << PermutationTest::sAPILogBuff.str();

            printKernel(stream);

            if(mPrintElements)
            {
                auto resource = getResource();

                size_t elementsA   = resource->getCurrentMatrixElement();
                size_t elementsB   = elementsA;
                size_t elementsRef = elementsA;

                if(dataType == HIP_R_32F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements (" << elementsB << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostB().get(), elementsB);
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
                        stream, (_Float16*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements (" << elementsB << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostB().get(), elementsB);
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

        auto abDataType      = dataTypes[0];
        auto computeDataType = dataTypes[1];

        auto Aop = operators[0];
        auto Bop = operators[1];

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

            hiptensorStatus_t  err;
            hiptensorHandle_t* handle;
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            hiptensorTensorDescriptor_t descA;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle, &descA, nmodeA, extentA.data(), NULL /* stride */, abDataType, Aop));

            hiptensorTensorDescriptor_t descB;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle, &descB, nmodeB, extentB.data(), NULL /* stride */, abDataType, Bop));

            float alphaValue{};
            if(computeDataType == HIP_R_16F)
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

            CHECK_HIPTENSOR_ERROR(hiptensorPermutation(handle,
                                                       &alphaValue,
                                                       resource->deviceA().get(),
                                                       &descA,
                                                       modeA.data(),
                                                       resource->deviceB().get(),
                                                       &descB,
                                                       modeB.data(),
                                                       computeDataType,
                                                       0 /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t sizeA = std::accumulate(extentA.begin(),
                                           extentA.end(),
                                           hipDataTypeSize(abDataType),
                                           std::multiplies<size_t>());

            size_t sizeB = std::accumulate(extentB.begin(),
                                           extentB.end(),
                                           hipDataTypeSize(abDataType),
                                           std::multiplies<size_t>());

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = 2.0 * ((sizeA * sizeB) / hipDataTypeSize(abDataType));
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            mTotalBytes = sizeA + sizeB;
            mTotalBytes /= (1e9 * mElapsedTimeMs);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            resource->copyBToHost();

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())
            {
                resource->copyBToHost();

                if(abDataType == HIP_R_32F)
                {
                    CHECK_HIPTENSOR_ERROR(
                        hiptensorPermutationReference(handle,
                                                      &alphaValue,
                                                      (const float*)resource->hostA().get(),
                                                      &descA,
                                                      modeA.data(),
                                                      (float*)resource->hostReference().get(),
                                                      &descB,
                                                      modeB.data(),
                                                      computeDataType,
                                                      0 /* stream */));

                    resource->copyReferenceToDevice();
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float>((float*)resource->deviceB().get(),
                                                          (float*)resource->deviceReference().get(),
                                                          resource->getCurrentMatrixElement(),
                                                          convertToComputeType(computeDataType));
                }
                else if(abDataType == HIP_R_16F)
                {
                    CHECK_HIPTENSOR_ERROR(
                        hiptensorPermutationReference(handle,
                                                      &alphaValue,
                                                      (const _Float16*)resource->hostA().get(),
                                                      &descA,
                                                      modeA.data(),
                                                      (_Float16*)resource->hostReference().get(),
                                                      &descB,
                                                      modeB.data(),
                                                      computeDataType,
                                                      0 /* stream */));

                    resource->copyReferenceToDevice();

                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<_Float16>(
                            (_Float16*)resource->deviceB().get(),
                            (_Float16*)resource->deviceReference().get(),
                            resource->getCurrentMatrixElement(),
                            convertToComputeType(computeDataType));
                }
                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
            } // if (testOptions->performValidation())

            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }

        using Options        = hiptensor::HiptensorOptions;
        auto& loggingOptions = Options::instance();

        if(!loggingOptions->omitCout())
        {
            reportResults(std::cout,
                          abDataType,
                          mHeaderPrinted,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
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

    void PermutationTest::TearDown()
    {
        if(mRunFlag)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }
    }

} // namespace hiptensor
