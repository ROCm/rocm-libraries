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
        hiptensorLoggerOpenFile("/dev/null");
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Reduction kernels have different requirements
    // True = run test
    // False = skip test
    bool ReductionTest::checkDevice(hipDataType            datatype,
                                    hiptensorComputeType_t computeDataType) const
    {
        return !(((datatype == HIP_R_32F || computeDataType == HIP_R_32F) && !isF32Supported())
                 || ((datatype == HIP_R_64F || computeDataType == HIP_R_64F) && !isF64Supported()));
    }

    bool ReductionTest::checkSizes() const
    {
        return true;
    }

    void ReductionTest::reset()
    {
        handle = nullptr;

        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalBytes = 0.0;
    }

    ReductionResource* ReductionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    std::ostream& ReductionTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream
            << "TypeIn,"               //1
            << "TypeCompute,"          //2
            << "Operator,"             //3
            << "LogLevel,"             //4
            << "Lengths,"              //5
            << "ReOrder,"              //6
            << "Alpha,"                //7
            << "Beta,"                 //8
            << "elapsedMs,"            //9
            << "Problem Size(GFlops)," //10
            << "TFlops,"               //11
            << "TotalBytes,"           //12
            << "Result"                 //13
            << std::endl;
        // clang-format on
    }

    std::ostream& ReductionTest::printKernel(std::ostream& stream) const
    {
        auto param      = Base::GetParam();
        auto testType   = std::get<0>(param);
        auto logLevel   = std::get<1>(param);
        auto lengths    = std::get<2>(param);
        auto outputDims = std::get<3>(param);
        auto alpha      = std::get<4>(param);
        auto beta       = std::get<5>(param);
        auto op         = std::get<6>(param);

        // clang-format off
        stream << hipTypeToString(testType[0]) << ","                           //1
               << computeTypeToString(convertToComputeType(testType[1])) << "," //2
               << opTypeToString(op) << ","                                     //3
               << logLevelToString(logLevel) << ",";                            //4
        printContainerInCsv(lengths, stream) << ",";                            //5
        printContainerInCsv(outputDims, stream) << ",";                         //6
        stream << alpha << ","                                                  //7
            << beta << ",";                                                     //8
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream << "n/a" << "," //9
                   << "n/a" << "," //10
                   << "n/a" << "," //11
                   << "n/a" << "," //12
                   << "SKIPPED"     //13
                   << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream << mElapsedTimeMs << ","     //9
                << mTotalGFlops << ","          //10
                << mMeasuredTFlopsPerSec << "," //11
                << mTotalBytes << ","           //12
                << result                        //13
                << std::endl;
            // clang-format on
        }

        return stream;
    }

    void ReductionTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param      = Base::GetParam();
        auto dataTypes  = std::get<0>(param);
        auto logLevel   = std::get<1>(param);
        auto lengths    = std::get<2>(param);
        auto outputDims = std::get<3>(param);
        auto alpha      = std::get<4>(param);
        auto beta       = std::get<5>(param);
        auto op         = std::get<6>(param);

        EXPECT_TRUE((lengths.size() > 0) && (lengths.size() <= 6));
        EXPECT_TRUE((outputDims.size() >= 0) && (outputDims.size() < 6));

        EXPECT_TRUE((op == HIPTENSOR_OP_ADD) || (op == HIPTENSOR_OP_MUL) || (op == HIPTENSOR_OP_MAX)
                    || (op == HIPTENSOR_OP_MIN));

        EXPECT_EQ(dataTypes.size(), 2); // HIP_R_16F or HIP_R_32F
        auto acDataType      = dataTypes[0];
        auto computeDataType = convertToComputeType(dataTypes[1]);
        EXPECT_TRUE((acDataType == HIP_R_16F && computeDataType == HIPTENSOR_COMPUTE_16F)
                    || (acDataType == HIP_R_16F && computeDataType == HIPTENSOR_COMPUTE_32F)
                    || (acDataType == HIP_R_16BF && computeDataType == HIPTENSOR_COMPUTE_16BF)
                    || (acDataType == HIP_R_16BF && computeDataType == HIPTENSOR_COMPUTE_32F)
                    || (acDataType == HIP_R_32F && computeDataType == HIPTENSOR_COMPUTE_32F)
                    || (acDataType == HIP_R_64F && computeDataType == HIPTENSOR_COMPUTE_64F));

        mRunFlag &= checkDevice(acDataType, computeDataType);
        mRunFlag &= lengths.size() > outputDims.size();

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

    void ReductionTest::reportResults(std::ostream& stream,
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
            stream << ReductionTest::sAPILogBuff.str();

            printKernel(stream);

            if(mPrintElements)
            {
                auto resource = getResource();

                auto param      = Base::GetParam();
                auto dataTypes  = std::get<0>(param);
                auto logLevel   = std::get<1>(param);
                auto lengths    = std::get<2>(param);
                auto outputDims = std::get<3>(param);
                auto alpha      = std::get<4>(param);
                auto beta       = std::get<5>(param);
                auto op         = std::get<6>(param);
                stream << "Input [type: " << dataTypes << ", lengths: " << lengths
                       << ", outputDims: " << outputDims << ", alpha: " << alpha
                       << ", beta: " << beta << ", opReduce: " << op << "]\n";

                size_t elementsA = resource->getCurrentInputElementCount();
                size_t elementsC = resource->getCurrentOutputElementCount();
                size_t elementsD = elementsC;

                if(dataType == HIP_R_16BF)
                {
                    printReductionTestInputOutput<bfloat16_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
                else if(dataType == HIP_R_16F)
                {
                    printReductionTestInputOutput<float16_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
                else if(dataType == HIP_R_32F)
                {
                    printReductionTestInputOutput<float32_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
                else if(dataType == HIP_R_64F)
                {
                    printReductionTestInputOutput<float64_t>(
                        stream, resource, elementsA, elementsC, elementsD);
                }
            }
        }
    }

    void ReductionTest::RunKernel()
    {
        auto param      = Base::GetParam();
        auto dataTypes  = std::get<0>(param);
        auto logLevel   = std::get<1>(param);
        auto lengths    = std::get<2>(param);
        auto outputDims = std::get<3>(param);
        auto alpha      = std::get<4>(param);
        auto beta       = std::get<5>(param);
        auto opReduce   = std::get<6>(param);

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

            // Requirement of lengths and strides of output
            //
            // For example, input lengths are [3, 5, 8], output dims are [2, 1]
            //
            // CK requires that lengths of output are [5, 8], i.e. lengths of sorted output dims
            //
            // Strides of output are generated in this way:
            //   output dims are [2, 1]
            //     ==> corresponding lengths are [8(2), 5(1)]
            //     ==> strides are [5,(2) 1(1)]
            //     ==> sorted strides are [1(1), 5(2)] // sort by dim
            //
            //  strides of output are [1, 5]
            std::vector<int> sortedOutputDims(outputDims.cbegin(), outputDims.cend());
            std::sort(sortedOutputDims.begin(), sortedOutputDims.end());

            std::vector<int64_t> extentA(lengths.cbegin(), lengths.cend());
            std::vector<int64_t> extentC(sortedOutputDims.cbegin(), sortedOutputDims.cend());
            std::transform(extentC.cbegin(), extentC.cend(), extentC.begin(), [&lengths](auto dim) {
                return lengths[dim];
            });
            std::vector<int64_t> extentD(extentC);

            auto& options = HiptensorOptions::instance();

            std::vector<int64_t> strideD
                = hiptensor::stridesFromLengths(extentD, options->isColMajorStrides());
            if(!std::equal(outputDims.cbegin(), outputDims.cend(), sortedOutputDims.cbegin()))
            {
                std::unordered_map<int, int64_t> dimToStride;
                int64_t                          stride = 1;

                if(!options->isColMajorStrides())
                {
                    for(auto it = outputDims.crbegin(); it != outputDims.crend(); ++it)
                    {
                        dimToStride[*it] = stride;
                        stride *= lengths[*it];
                    }
                }
                else
                {
                    for(auto it = outputDims.cbegin(); it != outputDims.cend(); ++it)
                    {
                        dimToStride[*it] = stride;
                        stride *= lengths[*it];
                    }
                }

                std::transform(sortedOutputDims.cbegin(),
                               sortedOutputDims.cend(),
                               strideD.begin(),
                               [&dimToStride](uint64_t dim) { return dimToStride[dim]; });
            }
            std::vector<int64_t> strideC = strideD;

            hiptensorStatus_t  err;
            hiptensorHandle_t* handle;
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            hiptensorTensorDescriptor_t descA;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &descA,
                                                                nmodeA,
                                                                extentA.data(),
                                                                NULL /* stride */,
                                                                acDataType,
                                                                HIPTENSOR_OP_IDENTITY));

            hiptensorTensorDescriptor_t descC;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &descC,
                                                                nmodeC,
                                                                extentC.data(),
                                                                strideC.data(),
                                                                acDataType,
                                                                HIPTENSOR_OP_IDENTITY));

            hiptensorTensorDescriptor_t descD;
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(handle,
                                                                &descD,
                                                                nmodeD,
                                                                extentD.data(),
                                                                strideD.data(),
                                                                acDataType,
                                                                HIPTENSOR_OP_IDENTITY));

            uint64_t worksize = 0;
            CHECK_HIPTENSOR_ERROR(hiptensorReductionGetWorkspaceSize(handle,
                                                                     resource->deviceA().get(),
                                                                     &descA,
                                                                     modeA.data(),
                                                                     resource->deviceC().get(),
                                                                     &descC,
                                                                     modeC.data(),
                                                                     resource->deviceD().get(),
                                                                     &descD,
                                                                     modeD.data(),
                                                                     opReduce,
                                                                     computeDataType,
                                                                     &worksize));
            resource->setupWorkspace(worksize);

            void*  work = resource->deviceWorkspace().get();
            double alphaValue{};
            double betaValue{};
            writeVal(&alphaValue, computeDataType, {computeDataType, alpha});
            writeVal(&betaValue, computeDataType, {computeDataType, beta});

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorReduction(handle,
                                                     (const void*)&alphaValue,
                                                     resource->deviceA().get(),
                                                     &descA,
                                                     modeA.data(),
                                                     (const void*)&betaValue,
                                                     resource->deviceC().get(),
                                                     &descC,
                                                     modeC.data(),
                                                     resource->deviceD().get(),
                                                     &descD,
                                                     modeD.data(),
                                                     opReduce,
                                                     computeDataType,
                                                     work,
                                                     worksize,
                                                     0 /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t sizeA = std::accumulate(extentA.begin(),
                                           extentA.end(),
                                           hipDataTypeSize(acDataType),
                                           std::multiplies<size_t>());

            size_t sizeCD = std::accumulate(extentC.begin(),
                                            extentC.end(),
                                            hipDataTypeSize(acDataType),
                                            std::multiplies<size_t>());

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = sizeA / hipDataTypeSize(acDataType);
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            mTotalBytes = sizeA + sizeCD;
            mTotalBytes += (betaValue != 0.0) ? sizeCD : 0;
            mTotalBytes /= (1e9 * mElapsedTimeMs);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())

            {
                resource->copyOutputToHost();

                CHECK_HIPTENSOR_ERROR(hiptensorReductionReference(&alphaValue,
                                                                  resource->hostA().get(),
                                                                  &descA,
                                                                  modeA.data(),
                                                                  &betaValue,
                                                                  resource->hostC().get(),
                                                                  &descC,
                                                                  modeC.data(),
                                                                  resource->hostReference().get(),
                                                                  &descD,
                                                                  modeD.data(),
                                                                  opReduce,
                                                                  computeDataType,
                                                                  0 /* stream */));
                resource->copyReferenceToDevice();

                if(acDataType == HIP_R_16F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float16_t>(
                            (float16_t*)resource->deviceD().get(),
                            (float16_t*)resource->deviceReference().get(),
                            resource->getCurrentOutputElementCount(),
                            computeDataType);
                }
                else if(acDataType == HIP_R_16BF)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<bfloat16_t>(
                            (bfloat16_t*)resource->deviceD().get(),
                            (bfloat16_t*)resource->deviceReference().get(),
                            resource->getCurrentOutputElementCount(),
                            computeDataType);
                }
                else if(acDataType == HIP_R_32F)
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
                else if(acDataType == HIP_R_64F)
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
                reportResults(std::cout,
                              acDataType,
                              mHeaderPrinted,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
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
        }
    }

    void ReductionTest::TearDown()
    {
        if(mRunFlag)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }
    }

} // namespace hiptensor
