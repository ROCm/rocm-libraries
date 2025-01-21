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

#include "contraction/contraction_cpu_reference.hpp"
#include "contraction_test.hpp"
#include "utils.hpp"

namespace hiptensor
{
    /*static*/ bool              ContractionTest::mHeaderPrinted = false;
    /*static*/ std::stringstream ContractionTest::sAPILogBuff    = std::stringstream();

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        ContractionTest::sAPILogBuff << msg;
    }

    ContractionTest::ContractionTest()
        : Base()
    {
        reset();

        // Handle our own outputs
        hiptensorLoggerOpenFile("/dev/null");
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Contraction kernels have different requirements
    // True = run test
    // False = skip test
    bool ContractionTest::checkDevice(hipDataType datatype) const
    {
        return (isF32Supported()
                && (datatype == HIP_R_32F || datatype == HIP_R_16F || datatype == HIP_R_16BF
                    || datatype == HIP_C_32F))
               || (isF64Supported() && (datatype == HIP_R_64F || datatype == HIP_C_64F));
    }

    bool ContractionTest::checkSizes() const
    {
        return true;
    }

    void ContractionTest::reset()
    {
        handle    = nullptr;
        workspace = nullptr;

        worksize = 0u;

        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalBytes = 0.0;
    }

    ContractionResource* ContractionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    std::ostream& ContractionTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream
            << "TypeA,"                // 1
            << "TypeB,"                // 2
            << "TypeC,"                // 3
            << "TypeD,"                // 4
            << "TypeCompute,"          // 5
            << "Algorithm,"            // 6
            << "Operator,"             // 7
            << "WorkSizePreference,"   // 8
            << "LogLevel,"             // 9
            << "Lengths,"              // 10
            << "Strides,"              // 11
            << "Modes,"                // 12
            << "Alpha,"                // 13
            << "Beta,"                 // 14
            << "elapsedMs,"            // 15
            << "Problem Size(GFlops)," // 16
            << "TFlops,"               // 17
            << "TotalBytes,"           // 18
            << "Result"                // 19
            << std::endl;
        // clang-format on
    }

    std::ostream& ContractionTest::printKernel(std::ostream& stream) const
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto algorithm    = std::get<1>(param);
        auto operatorType = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto strides      = std::get<6>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);

        // clang-format off
        stream
            << hipTypeToString(testType[0]) << ","                           // 1
            << hipTypeToString(testType[1]) << ","                           // 2
            << hipTypeToString(testType[2]) << ","                           // 3
            << hipTypeToString(testType[3]) << ","                           // 4
            << computeTypeToString(convertToComputeType(testType[4])) << "," // 5
            << algoTypeToString(algorithm)  << ","                           // 6
            << opTypeToString(operatorType) << ","                           // 7
            << workSizePrefToString(workSizePref) << ","                     // 8
            << logLevelToString(logLevel) << ",";                            // 9
        printContainerInCsv(lengths, stream) << ",";                         // 10
        printContainerInCsv(strides, stream) << ",";                         // 11
        printContainerInCsv(modes, stream)   << ",";                         // 12
        printContainerInCsv(alpha, stream)   << ",";                         // 13
        printContainerInCsv(beta, stream)   << ",";                          // 14
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream
                << "n/a" << "," // 15
                << "n/a" << "," // 16
                << "n/a" << "," // 17
                << "n/a" << "," // 18
                << "SKIPPED"    // 19
                << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream
                   << mElapsedTimeMs        << ","       // 15
                   << mTotalGFlops          << ","       // 16
                   << mMeasuredTFlopsPerSec << ","       // 17
                   << mTotalBytes           << ","       // 18
                   << result                             // 19
                   << std::endl;
            // clang-format on
        }

        return stream;
    }

    void ContractionTest::SetUp()
    {
        // reset API log buffer
        sAPILogBuff.str(std::string());

        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto algorithm    = std::get<1>(param);
        auto operatorType = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto strides      = std::get<6>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);

        EXPECT_EQ(dataTypes.size(), 5);

        //Check the format of lengths, strides and Modes(Max support is 6D across M,N,K dimensions)
        EXPECT_TRUE(lengths.size() == 3); // Tensors A, B, C/D
        EXPECT_TRUE(modes.size() == 3); // Tensors A, B, C/D
        if(!strides.empty())
        {
            EXPECT_TRUE(strides.size() == 3); // Tensors A, B, C/D
        }

        for(int i = 0; i < lengths.size(); i++)
        {
            EXPECT_TRUE(lengths[i].size() <= MaxNumDimsM + MaxNumDimsN);
            if(!strides.empty())
            {
                EXPECT_TRUE(strides[i].size() == lengths[i].size());
            }
            EXPECT_TRUE(modes[i].size() == lengths[i].size());
        }

        // Separate compute type from test types
        auto computeType = convertToComputeType(dataTypes[4]);

        auto ADataType = dataTypes[0];
        auto BDataType = dataTypes[1];
        auto CDataType = dataTypes[2];
        auto DDataType = dataTypes[3];

        EXPECT_TRUE((ADataType == HIP_R_16F) || (ADataType == HIP_R_16BF)
                    || (ADataType == HIP_R_32F) || (ADataType == HIP_R_64F)
                    || (ADataType == HIP_C_32F) || (ADataType == HIP_C_64F));
        EXPECT_TRUE((BDataType == HIP_R_16F) || (BDataType == HIP_R_16BF)
                    || (BDataType == HIP_R_32F) || (BDataType == HIP_R_64F)
                    || (BDataType == HIP_C_32F) || (BDataType == HIP_C_64F));
        EXPECT_TRUE((CDataType == HIP_R_16F) || (CDataType == HIP_R_16BF)
                    || (CDataType == HIP_R_32F) || (CDataType == HIP_R_64F)
                    || (CDataType == HIP_C_32F) || (CDataType == HIP_C_64F)
                    || (CDataType == NONE_TYPE));
        EXPECT_TRUE((DDataType == HIP_R_16F) || (DDataType == HIP_R_16BF)
                    || (DDataType == HIP_R_32F) || (DDataType == HIP_R_64F)
                    || (DDataType == HIP_C_32F) || (DDataType == HIP_C_64F));
        EXPECT_TRUE(
            (computeType == HIPTENSOR_COMPUTE_16F) || (computeType == HIPTENSOR_COMPUTE_16BF)
            || (computeType == HIPTENSOR_COMPUTE_32F) || (computeType == HIPTENSOR_COMPUTE_64F)
            || (computeType == HIPTENSOR_COMPUTE_C32F) || (computeType == HIPTENSOR_COMPUTE_C64F));

        mRunFlag &= checkDevice(DDataType);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            std::vector<int64_t> a_ms_ks_lengths, a_ms_ks_strides;
            std::vector<int32_t> a_ms_ks_modes;
            for(int i = 0; i < lengths[0].size(); i++)
            {
                a_ms_ks_modes.push_back(modes[0][i]);
                a_ms_ks_lengths.push_back(lengths[0][i]);
                if(!strides.empty())
                {
                    a_ms_ks_strides.push_back(strides[0][i]);
                }
            }

            std::vector<int64_t> b_ns_ks_lengths, b_ns_ks_strides;
            std::vector<int32_t> b_ns_ks_modes;
            for(int i = 0; i < lengths[1].size(); i++)
            {
                b_ns_ks_modes.push_back(modes[1][i]);
                b_ns_ks_lengths.push_back(lengths[1][i]);
                if(!strides.empty())
                {
                    b_ns_ks_strides.push_back(strides[1][i]);
                }
            }

            std::vector<int64_t> cd_ms_ns_lengths, cd_ms_ns_strides;
            std::vector<int32_t> cd_ms_ns_modes;
            for(int i = 0; i < lengths[2].size(); i++)
            {
                cd_ms_ns_modes.push_back(modes[2][i]);
                cd_ms_ns_lengths.push_back(lengths[2][i]);
                if(!strides.empty())
                {
                    cd_ms_ns_strides.push_back(strides[2][i]);
                }
            }

            size_t elementsA  = std::accumulate(a_ms_ks_lengths.begin(),
                                               a_ms_ks_lengths.end(),
                                               size_t{1},
                                               std::multiplies<size_t>());
            size_t elementsB  = std::accumulate(b_ns_ks_lengths.begin(),
                                               b_ns_ks_lengths.end(),
                                               size_t{1},
                                               std::multiplies<size_t>());
            size_t elementsCD = std::accumulate(cd_ms_ns_lengths.begin(),
                                                cd_ms_ns_lengths.end(),
                                                size_t{1},
                                                std::multiplies<size_t>());

            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

            CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(logLevel));

            // lengths - m, n, u, v, h, k
            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle,
                &a_ms_ks,
                a_ms_ks_lengths.size(),
                a_ms_ks_lengths.data(),
                strides.empty() ? NULL : a_ms_ks_strides.data(), /*stride*/
                ADataType,
                operatorType));

            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle,
                &b_ns_ks,
                b_ns_ks_lengths.size(),
                b_ns_ks_lengths.data(),
                strides.empty() ? NULL : b_ns_ks_strides.data(), /*stride*/
                BDataType,
                operatorType));

            if(CDataType != NONE_TYPE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                    handle,
                    &c_ms_ns,
                    cd_ms_ns_lengths.size(),
                    cd_ms_ns_lengths.data(),
                    strides.empty() ? NULL : cd_ms_ns_strides.data(), /*stride*/
                    CDataType,
                    operatorType));
            }

            CHECK_HIPTENSOR_ERROR(hiptensorInitTensorDescriptor(
                handle,
                &d_ms_ns,
                cd_ms_ns_lengths.size(),
                cd_ms_ns_lengths.data(),
                strides.empty() ? NULL : cd_ms_ns_strides.data(), /*stride*/
                DDataType,
                operatorType));

            std::tuple<int32_t, int32_t, int32_t, int32_t> elementBytes(hipDataTypeSize(ADataType),
                                                                        hipDataTypeSize(BDataType),
                                                                        hipDataTypeSize(CDataType),
                                                                        hipDataTypeSize(DDataType));

            auto resource = getResource();
            resource->resizeStorage(lengths, elementBytes);

            uint32_t seed = static_cast<uint32_t>(256);

            if(ADataType == HIP_R_16F && BDataType == HIP_R_16F && DDataType == HIP_R_16F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<_Float16>(
                    (_Float16*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<_Float16>((_Float16*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIP_R_16F)
                {
                    fillLaunchKernel<_Float16>(
                        (_Float16*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<_Float16>((_Float16*)resource->deviceD().get(),
                                              elementsCD,
                                              std::numeric_limits<_Float16>::signaling_NaN());
            }
            else if(ADataType == HIP_R_16BF && BDataType == HIP_R_16BF && DDataType == HIP_R_16BF)
            {
                // Initialize matrix data on device
                fillLaunchKernel<hip_bfloat16>(
                    (hip_bfloat16*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<hip_bfloat16>(
                    (hip_bfloat16*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIP_R_16BF)
                {
                    fillLaunchKernel<hip_bfloat16>(
                        (hip_bfloat16*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<hip_bfloat16>(
                    (hip_bfloat16*)resource->deviceD().get(),
                    elementsCD,
                    std::numeric_limits<hip_bfloat16>::signaling_NaN());
            }
            else if(ADataType == HIP_R_32F && BDataType == HIP_R_32F && DDataType == HIP_R_32F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<float>((float*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<float>((float*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIP_R_32F)
                {
                    fillLaunchKernel<float>(
                        (float*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<float>((float*)resource->deviceD().get(),
                                           elementsCD,
                                           std::numeric_limits<float>::signaling_NaN());
            }
            else if(ADataType == HIP_R_64F && BDataType == HIP_R_64F && DDataType == HIP_R_64F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<double>((double*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<double>((double*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIP_R_64F)
                {
                    fillLaunchKernel<double>(
                        (double*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<double>((double*)resource->deviceD().get(),
                                            elementsCD,
                                            std::numeric_limits<double>::signaling_NaN());
            }
            else if(ADataType == HIP_C_32F && BDataType == HIP_C_32F && DDataType == HIP_C_32F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<hipFloatComplex>(
                    (hipFloatComplex*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<hipFloatComplex>(
                    (hipFloatComplex*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIP_C_32F)
                {
                    fillLaunchKernel<hipFloatComplex>(
                        (hipFloatComplex*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<hipFloatComplex>(
                    (hipFloatComplex*)resource->deviceD().get(),
                    elementsCD,
                    std::numeric_limits<hipFloatComplex>::signaling_NaN());
            }
            else if(ADataType == HIP_C_64F && BDataType == HIP_C_64F && DDataType == HIP_C_64F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<hipDoubleComplex>(
                    (hipDoubleComplex*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<hipDoubleComplex>(
                    (hipDoubleComplex*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIP_C_64F)
                {
                    fillLaunchKernel<hipDoubleComplex>(
                        (hipDoubleComplex*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<hipDoubleComplex>(
                    (hipDoubleComplex*)resource->deviceD().get(),
                    elementsCD,
                    std::numeric_limits<hipDoubleComplex>::signaling_NaN());
            }

            resource->copyDeviceToHostAll(elementBytes);

            uint32_t alignmentRequirementA;
            CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                handle, resource->deviceA().get(), &a_ms_ks, &alignmentRequirementA));

            uint32_t alignmentRequirementB;
            CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                handle, resource->deviceB().get(), &b_ns_ks, &alignmentRequirementB));

            uint32_t alignmentRequirementC = 0;
            if(CDataType != NONE_TYPE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                    handle, resource->deviceC().get(), &c_ms_ns, &alignmentRequirementC));
            }

            uint32_t alignmentRequirementD;
            CHECK_HIPTENSOR_ERROR(hiptensorGetAlignmentRequirement(
                handle, resource->deviceD().get(), &d_ms_ns, &alignmentRequirementD));

            CHECK_HIPTENSOR_ERROR(hiptensorInitContractionDescriptor(
                handle,
                &desc,
                &a_ms_ks,
                a_ms_ks_modes.data(),
                alignmentRequirementA,
                &b_ns_ks,
                b_ns_ks_modes.data(),
                alignmentRequirementB,
                (CDataType != NONE_TYPE) ? &c_ms_ns : nullptr,
                (CDataType != NONE_TYPE) ? cd_ms_ns_modes.data() : nullptr,
                alignmentRequirementC,
                &d_ms_ns,
                cd_ms_ns_modes.data(),
                alignmentRequirementD,
                computeType));
            /**************************
            * Set the algorithm to use
            ***************************/

            CHECK_HIPTENSOR_ERROR(hiptensorInitContractionFind(handle, &find, algorithm));

            /**********************
            * Query workspace
            **********************/
            CHECK_HIPTENSOR_ERROR(hiptensorContractionGetWorkspaceSize(
                handle, &desc, &find, workSizePref, &worksize));

            if(worksize > 0)
            {
                CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
            }
        }
    }

    void ContractionTest::reportResults(std::ostream&          stream,
                                        hipDataType            DDataType,
                                        hiptensorComputeType_t computeType,
                                        bool                   omitHeader,
                                        bool                   omitSkipped,
                                        bool                   omitFailed,
                                        bool                   omitPassed) const
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

            stream << ContractionTest::sAPILogBuff.str();

            if(mPrintElements)
            {
                auto resource = getResource();

                int size = hipDataTypeSize(DDataType);

                size_t elementsA  = std::accumulate(a_ms_ks.mLengths.begin(),
                                                   a_ms_ks.mLengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t elementsB  = std::accumulate(b_ns_ks.mLengths.begin(),
                                                   b_ns_ks.mLengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t elementsCD = std::accumulate(d_ms_ns.mLengths.begin(),
                                                    d_ms_ns.mLengths.end(),
                                                    size_t{1},
                                                    std::multiplies<size_t>());

                auto D = resource->allocHost(elementsCD * size);
                resource->copyData(D, resource->deviceD(), elementsCD * size);

                auto& references = resource->hostD();

                if(DDataType == HIP_R_16F)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<_Float16>(stream, (_Float16*)D.get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor reference elements:\n";
                    hiptensorPrintArrayElements<_Float16>(
                        stream, (_Float16*)references.get(), elementsCD);
                    stream << std::endl;
                }
                else if(DDataType == HIP_R_16BF)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<hip_bfloat16>(
                        stream, (hip_bfloat16*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<hip_bfloat16>(
                        stream, (hip_bfloat16*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<hip_bfloat16>(
                        stream, (hip_bfloat16*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<hip_bfloat16>(
                        stream, (hip_bfloat16*)D.get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor reference elements:\n";
                    hiptensorPrintArrayElements<hip_bfloat16>(
                        stream, (hip_bfloat16*)references.get(), elementsCD);
                    stream << std::endl;
                }
                else if(DDataType == HIP_R_32F)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<float>(stream, (float*)D.get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor reference elements:\n";
                    hiptensorPrintArrayElements<float>(
                        stream, (float*)references.get(), elementsCD);
                    stream << std::endl;
                }
                else if(DDataType == HIP_R_64F)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<double>(stream, (double*)D.get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor reference elements:\n";
                    hiptensorPrintArrayElements<double>(
                        stream, (double*)references.get(), elementsCD);
                    stream << std::endl;
                }
                else if(DDataType == HIP_C_32F)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<hipFloatComplex>(
                        stream, (hipFloatComplex*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<hipFloatComplex>(
                        stream, (hipFloatComplex*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<hipFloatComplex>(
                        stream, (hipFloatComplex*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<hipFloatComplex>(
                        stream, (hipFloatComplex*)D.get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor reference elements:\n";
                    hiptensorPrintArrayElements<hipFloatComplex>(
                        stream, (hipFloatComplex*)references.get(), elementsCD);
                    stream << std::endl;
                }
                else if(DDataType == HIP_C_64F)
                {
                    stream << "Tensor A elements:\n";
                    hiptensorPrintArrayElements<hipDoubleComplex>(
                        stream, (hipDoubleComplex*)resource->hostA().get(), elementsA);
                    stream << std::endl;

                    stream << "Tensor B elements:\n";
                    hiptensorPrintArrayElements<hipDoubleComplex>(
                        stream, (hipDoubleComplex*)resource->hostB().get(), elementsB);
                    stream << std::endl;

                    stream << "Tensor C elements:\n";
                    hiptensorPrintArrayElements<hipDoubleComplex>(
                        stream, (hipDoubleComplex*)resource->hostC().get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor D elements:\n";
                    hiptensorPrintArrayElements<hipDoubleComplex>(
                        stream, (hipDoubleComplex*)D.get(), elementsCD);
                    stream << std::endl;

                    stream << "Tensor reference elements:\n";
                    hiptensorPrintArrayElements<hipDoubleComplex>(
                        stream, (hipDoubleComplex*)references.get(), elementsCD);
                    stream << std::endl;
                }
            }
        }
    }

    void ContractionTest::RunKernel()
    {
        auto param        = Base::GetParam();
        auto dataTypes    = std::get<0>(param);
        auto algorithm    = std::get<1>(param);
        auto operatorType = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto strides      = std::get<6>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);

        if(mRunFlag)
        {
            auto ADataType = dataTypes[0];
            auto BDataType = dataTypes[1];
            auto CDataType = dataTypes[2];
            auto DDataType = dataTypes[3];

            auto computeType = convertToComputeType(dataTypes[4]);

            /*
             * `alpha` and `beta` are void pointer. hiptensor uses readVal to load the value of alpha.
             * ```
             * alphaF = hiptensor::readVal<float>(
             *      alpha, convertToComputeType(HipDataType_v<typename Traits::ComputeDataT>));
             * ```
             * Hence, the `alpha` and `bete` need to point to a ComputeData value
             */
            ScalarData alphaBuf;
            ScalarData betaBuf;
            writeVal(&alphaBuf, computeType, ScalarData(computeType, alpha[0], alpha[1]));
            writeVal(&betaBuf, computeType, ScalarData(computeType, beta[0], beta[1]));

            CHECK_HIPTENSOR_ERROR(
                hiptensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

            auto resource = getResource();

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorContraction(handle,
                                                       &plan,
                                                       (void*)&alphaBuf,
                                                       resource->deviceA().get(),
                                                       resource->deviceB().get(),
                                                       (void*)&betaBuf,
                                                       resource->deviceC().get(),
                                                       resource->deviceD().get(),
                                                       workspace,
                                                       worksize,
                                                       0 /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            size_t totalLength = std::accumulate(d_ms_ns.mLengths.begin(),
                                                 d_ms_ns.mLengths.end(),
                                                 size_t(1),
                                                 std::multiplies<size_t>());

            uint32_t hops = desc.mTensorMode[2].size() / 2;
            auto     iter = std::find(desc.mTensorMode[0].cbegin(),
                                  desc.mTensorMode[0].cend(),
                                  desc.mTensorMode[2][desc.mTensorMode[2].size() - 1]);
            if(iter != desc.mTensorMode[0].cend())
            {
                auto offset = std::distance(desc.mTensorMode[0].cbegin(), iter);
                totalLength *= std::accumulate(a_ms_ks.mLengths.begin() + offset,
                                               a_ms_ks.mLengths.begin() + offset + hops,
                                               size_t(1),
                                               std::multiplies<size_t>());
            }

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = 2.0 * totalLength;
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            size_t sizeA = std::accumulate(a_ms_ks.mLengths.begin(),
                                           a_ms_ks.mLengths.end(),
                                           hipDataTypeSize(ADataType),
                                           std::multiplies<size_t>());

            size_t sizeB = std::accumulate(b_ns_ks.mLengths.begin(),
                                           b_ns_ks.mLengths.end(),
                                           hipDataTypeSize(BDataType),
                                           std::multiplies<size_t>());

            size_t sizeD = std::accumulate(d_ms_ns.mLengths.begin(),
                                           d_ms_ns.mLengths.end(),
                                           hipDataTypeSize(DDataType),
                                           std::multiplies<size_t>());

            mTotalBytes = sizeA + sizeB + sizeD;
            mTotalBytes += (betaBuf.mReal != 0.0) ? sizeD : 0;
            mTotalBytes /= (1e9 * mElapsedTimeMs);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())
            {
                CHECK_HIPTENSOR_ERROR(hiptensorContractionReference(&plan,
                                                                    (void*)&alphaBuf,
                                                                    resource->hostA().get(),
                                                                    resource->hostB().get(),
                                                                    (void*)&betaBuf,
                                                                    resource->hostC().get(),
                                                                    resource->hostD().get(),
                                                                    a_ms_ks.mLengths,
                                                                    a_ms_ks.mStrides,
                                                                    desc.mTensorMode[0],
                                                                    b_ns_ks.mLengths,
                                                                    b_ns_ks.mStrides,
                                                                    desc.mTensorMode[1],
                                                                    d_ms_ns.mLengths,
                                                                    d_ms_ns.mStrides,
                                                                    desc.mTensorMode[2],
                                                                    d_ms_ns.mLengths,
                                                                    d_ms_ns.mStrides,
                                                                    desc.mTensorMode[2],
                                                                    ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    DDataType,
                                                                    workspace));

                auto reference = resource->allocDevice(sizeD);
                resource->copyData(reference, resource->hostD(), sizeD);

                // Compute tolerance based on compute type
                auto dimension = a_ms_ks.mLengths.size() / 2;
                auto nelems_k  = std::accumulate(a_ms_ks.mLengths.begin() + dimension,
                                                a_ms_ks.mLengths.end(),
                                                size_t{1},
                                                std::multiplies<size_t>());

                size_t elementsCD = sizeD / hipDataTypeSize(ADataType);

                auto   eps = getEpsilon(computeType == HIPTENSOR_COMPUTE_64F ? HIPTENSOR_COMPUTE_64F
                                                                           : HIPTENSOR_COMPUTE_32F);
                double tolerance = 2 * nelems_k * eps;

                // use the same default tolerance value as CK
                if(computeType == HIPTENSOR_COMPUTE_16BF || DDataType == HIP_R_16BF)
                {
                    const double epsilon = std::pow(2, -7);
                    tolerance += epsilon * 2;
                }
                else if(computeType == HIPTENSOR_COMPUTE_16F || DDataType == HIP_R_16F)
                {
                    const double epsilon = std::pow(2, -10);
                    tolerance += epsilon * 2;
                }

                if(DDataType == HIP_R_16F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<_Float16>((_Float16*)resource->deviceD().get(),
                                                             (_Float16*)reference.get(),
                                                             elementsCD,
                                                             computeType,
                                                             tolerance);
                }
                else if(DDataType == HIP_R_16BF)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<hip_bfloat16>(
                            (hip_bfloat16*)resource->deviceD().get(),
                            (hip_bfloat16*)reference.get(),
                            elementsCD,
                            computeType,
                            tolerance);
                }
                else if(DDataType == HIP_R_32F || DDataType == HIP_C_32F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float>((float*)resource->deviceD().get(),
                                                          (float*)reference.get(),
                                                          elementsCD,
                                                          computeType,
                                                          tolerance);
                }
                else if(DDataType == HIP_R_64F || DDataType == HIP_C_64F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<double>((double*)resource->deviceD().get(),
                                                           (double*)reference.get(),
                                                           elementsCD,
                                                           computeType,
                                                           tolerance);
                }

                EXPECT_TRUE(mValidationResult) << "Max relative error: " << mMaxRelativeError;
            } // if (testOptions->performValidation())

            using Options        = hiptensor::HiptensorOptions;
            auto& loggingOptions = Options::instance();

            if(!loggingOptions->omitCout())
            {
                reportResults(std::cout,
                              DDataType,
                              computeType,
                              mHeaderPrinted,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
            }

            if(loggingOptions->ostream().isOpen())
            {
                reportResults(loggingOptions->ostream().fstream(),
                              DDataType,
                              computeType,
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

    void ContractionTest::TearDown()
    {
        if(mRunFlag)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
        }
    }

} // namespace hiptensor
