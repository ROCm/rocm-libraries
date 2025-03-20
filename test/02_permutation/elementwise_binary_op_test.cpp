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
#include "elementwise_binary_op_test.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "permutation/permutation_cpu_reference.hpp"
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
        hiptensorLoggerOpenFile("/dev/null");
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different ElementwiseBinaryOp kernels have different requirements
    // True = run test
    // False = skip test
    bool ElementwiseBinaryOpTest::checkDevice(hipDataType datatype) const
    {
        return (isF32Supported() && ((datatype == HIP_R_32F) || (datatype == HIP_R_16F)))
               || (isF64Supported() && (datatype == HIP_R_64F));
    }

    bool ElementwiseBinaryOpTest::checkSizes() const
    {
        return true;
    }

    void ElementwiseBinaryOpTest::reset()
    {
        handle = nullptr;

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
            << "PermutedOrder, "        // 6
            << "Alpha, "                // 7
            << "Gamma, "                // 8
            << "ElapsedMs, "            // 9
            << "Problem Size(GFlops), " // 10
            << "TFlops/s, "             // 11
            << "TotalGBytes, "          // 12
            << "GBytes/s, "             // 13
            << "Result"                 // 14
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

        // clang-format off
        stream << hipTypeToString(testType[0]) << ", "                                              // 1
            << computeTypeToString(convertToComputeType(testType[1])) << ", "                       // 2
            << "[ " << opTypeToString(operators[0]) << " " << opTypeToString(operators[1]) << " " << opTypeToString(operators[2]) << "], " // 3
            << logLevelToString(logLevel) << ", ";                                                  // 4
        printContainerInCsv(lengths, stream) << ", ";                                               // 5
        printContainerInCsv(permutedDims, stream) << ", ";                                          // 6
        stream << alpha << ", ";                                                                    // 7
        stream << gamma << ", ";                                                                    // 8
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

    void ElementwiseBinaryOpTest::reportResults(std::ostream& stream,
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
            stream << ElementwiseBinaryOpTest::sAPILogBuff.str();

            printKernel(stream);

            if(mPrintElements)
            {
                auto resource = getResource();

                size_t elementsA   = resource->getCurrentMatrixElement();
                size_t elementsC   = elementsA;
                size_t elementsD   = elementsA;
                size_t elementsRef = elementsA;

                if(dataType == HIP_R_64F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostInput1().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostInput2().get(), elementsC);
                    stream << std::endl;

                    stream << "Tensor D elements (" << elementsD << "):\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostOutput().get(), elementsD);
                    stream << std::endl;

                    stream << "Tensor ref elements (" << elementsRef << "):\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostReference().get(), elementsRef);
                    stream << std::endl;
                }
                else if(dataType == HIP_R_32F)
                {
                    stream << "Tensor A elements (" << elementsA << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostInput1().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostInput2().get(), elementsC);
                    stream << std::endl;

                    stream << "Tensor D elements (" << elementsD << "):\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostOutput().get(), elementsD);
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

                    stream << "Tensor C elements (" << elementsC << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostInput2().get(), elementsC);
                    stream << std::endl;

                    stream << "Tensor D elements (" << elementsD << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostOutput().get(), elementsD);
                    stream << std::endl;

                    stream << "Tensor ref elements (" << elementsRef << "):\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostReference().get(), elementsRef);
                    stream << std::endl;
                }
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

            hiptensorStatus_t  err;
            hiptensorHandle_t* handle;
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            hiptensorTensorDescriptor_t descA;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle, &descA, nmodeA, extentA.data(), NULL /* stride */, dataType, Aop));

            hiptensorTensorDescriptor_t descC;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle, &descC, nmodeC, extentC.data(), NULL /* stride */, dataType, Cop));

            hiptensorTensorDescriptor_t descD;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &descD,
                                                                nmodeD,
                                                                extentD.data(),
                                                                NULL /* stride */,
                                                                dataType,
                                                                HIPTENSOR_OP_IDENTITY));

            float alphaValue{};
            if(computeDataType == HIP_R_16F)
            {
                *(reinterpret_cast<_Float16*>(&alphaValue)) = static_cast<_Float16>(alpha);
            }
            else if(computeDataType == HIP_R_32F)
            {
                *(reinterpret_cast<float*>(&alphaValue)) = static_cast<float>(alpha);
            }
            else if(computeDataType == HIP_R_64F)
            {
                *(reinterpret_cast<double*>(&alphaValue)) = static_cast<double>(alpha);
            }
            float gammaValue{};
            if(computeDataType == HIP_R_16F)
            {
                *(reinterpret_cast<_Float16*>(&gammaValue)) = static_cast<_Float16>(gamma);
            }
            else if(computeDataType == HIP_R_32F)
            {
                *(reinterpret_cast<float*>(&gammaValue)) = static_cast<float>(gamma);
            }
            else if(computeDataType == HIP_R_64F)
            {
                *(reinterpret_cast<double*>(&gammaValue)) = static_cast<double>(gamma);
            }

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinary(handle,
                                                             &alphaValue,
                                                             resource->deviceInput1().get(),
                                                             &descA,
                                                             modeA.data(),
                                                             &gammaValue,
                                                             resource->deviceInput2().get(),
                                                             &descC,
                                                             modeC.data(),
                                                             resource->deviceOutput().get(),
                                                             &descD,
                                                             modeD.data(),
                                                             ACop,
                                                             computeDataType,
                                                             0 /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t sizeA = std::accumulate(extentA.begin(),
                                           extentA.end(),
                                           hipDataTypeSize(dataType),
                                           std::multiplies<size_t>());

            size_t sizeC = std::accumulate(extentC.begin(),
                                           extentC.end(),
                                           hipDataTypeSize(dataType),
                                           std::multiplies<size_t>());

            size_t sizeD = std::accumulate(extentD.begin(),
                                           extentD.end(),
                                           hipDataTypeSize(dataType),
                                           std::multiplies<size_t>());

            mElapsedTimeMs        = float64_t(timeMs);
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

                if(dataType == HIP_R_64F)
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryOpReference(
                        handle,
                        &alphaValue,
                        (const double*)resource->hostInput1().get(),
                        &descA,
                        modeA.data(),
                        &gammaValue,
                        (const double*)resource->hostInput2().get(),
                        &descC,
                        modeC.data(),
                        (double*)resource->hostReference().get(),
                        &descD,
                        modeD.data(),
                        ACop,
                        computeDataType,
                        0 /* stream */));

                    resource->copyReferenceToDevice();
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<double>(
                            (double*)resource->deviceOutput().get(),
                            (double*)resource->deviceReference().get(),
                            resource->getCurrentMatrixElement(),
                            convertToComputeType(computeDataType));
                }
                else if(dataType == HIP_R_32F)
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryOpReference(
                        handle,
                        &alphaValue,
                        (const float*)resource->hostInput1().get(),
                        &descA,
                        modeA.data(),
                        &gammaValue,
                        (const float*)resource->hostInput2().get(),
                        &descC,
                        modeC.data(),
                        (float*)resource->hostReference().get(),
                        &descD,
                        modeD.data(),
                        ACop,
                        computeDataType,
                        0 /* stream */));

                    resource->copyReferenceToDevice();
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float>((float*)resource->deviceOutput().get(),
                                                          (float*)resource->deviceReference().get(),
                                                          resource->getCurrentMatrixElement(),
                                                          convertToComputeType(computeDataType));
                }
                else if(dataType == HIP_R_16F)
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryOpReference(
                        handle,
                        &alphaValue,
                        (const _Float16*)resource->hostInput1().get(),
                        &descA,
                        modeA.data(),
                        &gammaValue,
                        (const _Float16*)resource->hostInput2().get(),
                        &descC,
                        modeC.data(),
                        (_Float16*)resource->hostReference().get(),
                        &descD,
                        modeD.data(),
                        ACop,
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
        }

        using Options        = hiptensor::HiptensorOptions;
        auto& loggingOptions = Options::instance();

        if(!loggingOptions->omitCout())
        {
            reportResults(std::cout,
                          dataType,
                          mHeaderPrinted,
                          loggingOptions->omitSkipped(),
                          loggingOptions->omitFailed(),
                          loggingOptions->omitPassed());
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

    void ElementwiseBinaryOpTest::TearDown()
    {
        if(mRunFlag)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }
    }

} // namespace hiptensor
