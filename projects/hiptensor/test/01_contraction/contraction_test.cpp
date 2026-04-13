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

#include "data_types.hpp"
#include "hiptensor_options.hpp"

#include "contraction/contraction_cpu_reference.hpp"
#include "contraction_test.hpp"
#include "util.hpp"
#include "utils.hpp"

namespace hiptensor
{
    /*static*/ bool              ContractionTest::mHeaderPrinted = false;
    /*static*/ std::stringstream ContractionTest::sAPILogBuff    = std::stringstream();

    template <typename DataType>
    using FillLaunchKernelFn = void (*)(DataType* data, uint32_t elementSize, uint32_t seed);

    static void logMessage(int32_t logLevel, const char* funcName /*=""*/, const char* msg /*=""*/)
    {
        ContractionTest::sAPILogBuff << msg;
    }

    ContractionTest::ContractionTest()
        : Base()
    {
        reset();

        // Handle our own outputs
        hiptensor::test::silenceLogger();
        hiptensorLoggerSetCallback(logMessage);
    }

    // Kernel run checks. Virtual as different Contraction kernels have different requirements
    // True = run test
    // False = skip test
    bool ContractionTest::checkDevice(hiptensorDataType_t          dataType,
                                      hiptensorComputeDescriptor_t computeType) const
    {
        switch(computeType)
        {
        case HIPTENSOR_COMPUTE_DESC_16F:
        case HIPTENSOR_COMPUTE_DESC_16BF:
            if(isDataType16Bits(dataType))
            {
                return isF16F16MatrixCoreSupported() && !isDataTypeComplex(dataType);
            }
            else if(isDataType32Bits(dataType))
            {
                return isF32F16MatrixCoreSupported() && !isDataTypeComplex(dataType);
            }
            return false;
        case HIPTENSOR_COMPUTE_DESC_32F:
            if(isDataType16Bits(dataType))
            {
                return isF16F32MatrixCoreSupported() && !isDataTypeComplex(dataType);
            }
            else if(isDataType32Bits(dataType))
            {
                return isF32F32MatrixCoreSupported() && !isDataTypeComplex(dataType);
            }
            else if(isDataType64Bits(dataType))
            {
                return isF64F32MatrixCoreSupported() && !isDataTypeComplex(dataType);
            }
            return false;
        case HIPTENSOR_COMPUTE_DESC_64F:
            if(isDataType64Bits(dataType))
            {
                return isF64F64MatrixCoreSupported() && !isDataTypeComplex(dataType);
            }
            return false;
        case HIPTENSOR_COMPUTE_DESC_C32F:
            if(isDataType16Bits(dataType))
            {
                return isF16F32MatrixCoreSupported() && isDataTypeComplex(dataType);
            }
            else if(isDataType32Bits(dataType))
            {
                return isF32F32MatrixCoreSupported() && isDataTypeComplex(dataType);
            }
            return false;
        case HIPTENSOR_COMPUTE_DESC_C64F:
            if(isDataType64Bits(dataType))
            {
                return isF64F64MatrixCoreSupported() && isDataTypeComplex(dataType);
            }
            return false;
        default:
            return false;
        }
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

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalGBytes = mGBytesPerSec = 0.0;
    }

    ContractionResource* ContractionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    std::ostream& ContractionTest::printHeader(std::ostream& stream /* = std::cout */) const
    {
        // clang-format off
        return stream
            << "TypeA, "                // 1
            << "TypeB, "                // 2
            << "TypeC, "                // 3
            << "TypeD, "                // 4
            << "TypeCompute, "          // 5
            << "Algorithm, "            // 6
            << "Operator, "             // 7
            << "WorkSizePreference, "   // 8
            << "LogLevel, "             // 9
            << "Lengths, "              // 10
            << "MemoryLayout, "         // 11
            << "Modes, "                // 12
            << "Alpha, "                // 13
            << "Beta, "                 // 14
            << "ElapsedMs, "            // 15
            << "Problem Size(GFlops), " // 16
            << "TFlops/s, "             // 17
            << "TotalGBytes, "          // 18
            << "GBytes/s, "             // 19
            << "Result"                 // 20
            << std::endl;
        // clang-format on
    }

    std::ostream& ContractionTest::printKernel(std::ostream& stream) const
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto algorithm    = std::get<1>(param);
        auto operators    = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto strides      = std::get<6>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);
        auto memoryLayout = std::get<10>(param);

        auto usedMemoryLayout = inferMemoryLayout(strides, lengths, memoryLayout);

        // clang-format off
        stream
            << hipTypeToString(testType[0]) << ", "                           // 1
            << hipTypeToString(testType[1]) << ", "                           // 2
            << hipTypeToString(testType[2]) << ", "                           // 3
            << hipTypeToString(testType[3]) << ", "                           // 4
            << computeTypeToString(convertToComputeType(testType[4])) << ", " // 5
            << algoTypeToString(algorithm)  << ", "                           // 6
            << "[ " << opTypeToString(operators[0]) << " " << opTypeToString(operators[1]) << " " << opTypeToString(operators[2]) << "], "     // 7
            << workSizePrefToString(workSizePref) << ", "                     // 8
            << logLevelToString(logLevel) << ", ";                            // 9
        printVectorInCsv(lengths, stream) << ", ";                            // 10
        stream << hipMemoryLayoutToString(usedMemoryLayout) << ", ";          // 11
        printContainerInCsv(modes, stream)   << ", ";                         // 12
        printContainerInCsv(alpha, stream)   << ", ";                         // 13
        printContainerInCsv(beta, stream)   << ", ";                          // 14
        // clang-format on

        if(!mRunFlag)
        {
            // clang-format off
            stream
                << "n/a" << ", " // 15
                << "n/a" << ", " // 16
                << "n/a" << ", " // 17
                << "n/a" << ", " // 18
                << "n/a" << ", " // 19
                << "SKIPPED"    // 20
                << std::endl;
            // clang-format on
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream
                   << mElapsedTimeMs        << ", "       // 15
                   << mTotalGFlops          << ", "       // 16
                   << mMeasuredTFlopsPerSec << ", "       // 17
                   << mTotalGBytes          << ", "       // 18
                   << mGBytesPerSec         << ", "       // 18
                   << result                              // 20
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
        auto operators    = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto strides      = std::get<6>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);
        auto memoryLayout = std::get<10>(param);

        EXPECT_EQ(dataTypes.size(), 5);

        //Check the format of lengths, strides and Modes(Max support is 6D across M,N,K dimensions)
        EXPECT_TRUE(lengths.size() == 3); // Tensors A, B, C/D
        EXPECT_TRUE(modes.size() == 3); // Tensors A, B, C/D
        if(!strides.empty())
        {
            EXPECT_TRUE(strides.size() == 3); // Tensors A, B, C/D
        }

        // If strides are provided, use as is
        if(strides.empty() && memoryLayout != HIPTENSOR_MEMORY_LAYOUT_DEFAULT)
        {
            strides.resize(lengths.size());
            for(int t = 0; t < static_cast<int>(lengths.size()); t++)
            {
                strides[t] = stridesFromLengths(
                    lengths[t], memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);
            }
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

        EXPECT_TRUE((ADataType == HIPTENSOR_R_16F) || (ADataType == HIPTENSOR_R_16BF)
                    || (ADataType == HIPTENSOR_R_32F) || (ADataType == HIPTENSOR_R_64F)
                    || (ADataType == HIPTENSOR_C_32F) || (ADataType == HIPTENSOR_C_64F));
        EXPECT_TRUE((BDataType == HIPTENSOR_R_16F) || (BDataType == HIPTENSOR_R_16BF)
                    || (BDataType == HIPTENSOR_R_32F) || (BDataType == HIPTENSOR_R_64F)
                    || (BDataType == HIPTENSOR_C_32F) || (BDataType == HIPTENSOR_C_64F));
        EXPECT_TRUE((CDataType == HIPTENSOR_R_16F) || (CDataType == HIPTENSOR_R_16BF)
                    || (CDataType == HIPTENSOR_R_32F) || (CDataType == HIPTENSOR_R_64F)
                    || (CDataType == HIPTENSOR_C_32F) || (CDataType == HIPTENSOR_C_64F)
                    || (CDataType == NONE_TYPE));
        EXPECT_TRUE((DDataType == HIPTENSOR_R_16F) || (DDataType == HIPTENSOR_R_16BF)
                    || (DDataType == HIPTENSOR_R_32F) || (DDataType == HIPTENSOR_R_64F)
                    || (DDataType == HIPTENSOR_C_32F) || (DDataType == HIPTENSOR_C_64F));
        EXPECT_TRUE((computeType == HIPTENSOR_COMPUTE_DESC_16F)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_16BF)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_32F)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_64F)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_C32F)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_C64F));

        mRunFlag &= checkDevice(DDataType, computeType);

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

            uint32_t alignmentRequirement = 1;
            // lengths - m, n, u, v, h, k
            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle,
                &a_ms_ks,
                a_ms_ks_lengths.size(),
                a_ms_ks_lengths.data(),
                strides.empty() ? NULL : a_ms_ks_strides.data(), /*stride*/
                ADataType,
                alignmentRequirement));

            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle,
                &b_ns_ks,
                b_ns_ks_lengths.size(),
                b_ns_ks_lengths.data(),
                strides.empty() ? NULL : b_ns_ks_strides.data(), /*stride*/
                BDataType,
                alignmentRequirement));

            if(CDataType != NONE_TYPE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                    handle,
                    &c_ms_ns,
                    cd_ms_ns_lengths.size(),
                    cd_ms_ns_lengths.data(),
                    strides.empty() ? NULL : cd_ms_ns_strides.data(), /*stride*/
                    CDataType,
                    alignmentRequirement));
            }

            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle,
                &d_ms_ns,
                cd_ms_ns_lengths.size(),
                cd_ms_ns_lengths.data(),
                strides.empty() ? NULL : cd_ms_ns_strides.data(), /*stride*/
                DDataType,
                alignmentRequirement));

            std::tuple<int32_t, int32_t, int32_t, int32_t> elementBytes(
                hiptensorDataTypeSize(ADataType),
                hiptensorDataTypeSize(BDataType),
                hiptensorDataTypeSize(CDataType),
                hiptensorDataTypeSize(DDataType));

            auto resource = getResource();
            resource->resizeStorage(lengths, elementBytes);

            uint32_t seed = static_cast<uint32_t>(256);

            FillLaunchKernelFn<_Float16>     fillFuncF16;
            FillLaunchKernelFn<hip_bfloat16> fillFuncBF16;
            FillLaunchKernelFn<float>        fillFuncF32;
            FillLaunchKernelFn<double>       fillFuncF64;

            if(operators[0] != HIPTENSOR_OP_IDENTITY || operators[1] != HIPTENSOR_OP_IDENTITY
               || operators[2] != HIPTENSOR_OP_IDENTITY)
            {
                // If any of the operators is not identity, we need to initialize tensors with positive values
                fillFuncF16  = fillPositiveValLaunchKernel<_Float16>;
                fillFuncBF16 = fillPositiveValLaunchKernel<hip_bfloat16>;
                fillFuncF32  = fillPositiveValLaunchKernel<float>;
                fillFuncF64  = fillPositiveValLaunchKernel<double>;
            }
            else
            {
                fillFuncF16  = fillLaunchKernel<_Float16>;
                fillFuncBF16 = fillLaunchKernel<hip_bfloat16>;
                fillFuncF32  = fillLaunchKernel<float>;
                fillFuncF64  = fillLaunchKernel<double>;
            }

            if(ADataType == HIPTENSOR_R_16F && BDataType == HIPTENSOR_R_16F
               && DDataType == HIPTENSOR_R_16F)
            {
                // Initialize matrix data on device
                fillFuncF16((_Float16*)resource->deviceA().get(), elementsA, seed - 1);
                fillFuncF16((_Float16*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIPTENSOR_R_16F)
                {
                    fillFuncF16((_Float16*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<_Float16>((_Float16*)resource->deviceD().get(),
                                              elementsCD,
                                              std::numeric_limits<_Float16>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_R_16BF && BDataType == HIPTENSOR_R_16BF
                    && DDataType == HIPTENSOR_R_16BF)
            {
                // Initialize matrix data on device
                fillFuncBF16((hip_bfloat16*)resource->deviceA().get(), elementsA, seed - 1);
                fillFuncBF16((hip_bfloat16*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIPTENSOR_R_16BF)
                {
                    fillFuncBF16((hip_bfloat16*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<hip_bfloat16>(
                    (hip_bfloat16*)resource->deviceD().get(),
                    elementsCD,
                    std::numeric_limits<hip_bfloat16>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_R_32F && BDataType == HIPTENSOR_R_32F
                    && DDataType == HIPTENSOR_R_32F)
            {
                // Initialize matrix data on device
                fillFuncF32((float*)resource->deviceA().get(), elementsA, seed - 1);
                fillFuncF32((float*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIPTENSOR_R_32F)
                {
                    fillFuncF32((float*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<float>((float*)resource->deviceD().get(),
                                           elementsCD,
                                           std::numeric_limits<float>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_R_64F && BDataType == HIPTENSOR_R_64F
                    && DDataType == HIPTENSOR_R_64F)
            {
                // Initialize matrix data on device
                fillFuncF64((double*)resource->deviceA().get(), elementsA, seed - 1);
                fillFuncF64((double*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIPTENSOR_R_64F)
                {
                    fillFuncF64((double*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<double>((double*)resource->deviceD().get(),
                                            elementsCD,
                                            std::numeric_limits<double>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_C_32F && BDataType == HIPTENSOR_C_32F
                    && DDataType == HIPTENSOR_C_32F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<hipFloatComplex>(
                    (hipFloatComplex*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<hipFloatComplex>(
                    (hipFloatComplex*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIPTENSOR_C_32F)
                {
                    fillLaunchKernel<hipFloatComplex>(
                        (hipFloatComplex*)resource->deviceC().get(), elementsCD, seed + 1);
                }
                fillValLaunchKernel<hipFloatComplex>(
                    (hipFloatComplex*)resource->deviceD().get(),
                    elementsCD,
                    std::numeric_limits<hipFloatComplex>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_C_64F && BDataType == HIPTENSOR_C_64F
                    && DDataType == HIPTENSOR_C_64F)
            {
                // Initialize matrix data on device
                fillLaunchKernel<hipDoubleComplex>(
                    (hipDoubleComplex*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<hipDoubleComplex>(
                    (hipDoubleComplex*)resource->deviceB().get(), elementsB, seed);
                if(CDataType == HIPTENSOR_C_64F)
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

            CHECK_HIPTENSOR_ERROR(hiptensorCreateContraction(
                handle,
                &desc,
                a_ms_ks,
                a_ms_ks_modes.data(),
                operators[0],
                b_ns_ks,
                b_ns_ks_modes.data(),
                operators[1],
                (CDataType != NONE_TYPE) ? c_ms_ns : nullptr,
                (CDataType != NONE_TYPE) ? cd_ms_ns_modes.data() : nullptr,
                operators[2],
                d_ms_ns,
                cd_ms_ns_modes.data(),
                computeType));

            hiptensorDataType_t scalarType;
            CHECK_HIPTENSOR_ERROR(
                hiptensorOperationDescriptorGetAttribute(handle,
                                                         desc,
                                                         HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));
            assert(scalarType == *hiptensor::convertToHipTensorDataType(computeType));

            /**************************
            * Set the algorithm to use
            ***************************/

            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
                handle, &planPref, algorithm, HIPTENSOR_JIT_MODE_NONE));

            /**********************
            * Query workspace
            **********************/
            CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
                handle, desc, planPref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

            if(worksize > 0)
            {
                CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
            }
        }
    }

    void ContractionTest::reportResults(std::ostream&                stream,
                                        hiptensorDataType_t          DDataType,
                                        hiptensorComputeDescriptor_t computeType,
                                        bool                         omitHeader,
                                        bool                         omitSkipped,
                                        bool                         omitFailed,
                                        bool                         omitPassed) const
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

                int size = hiptensorDataTypeSize(DDataType);

                size_t elementsA  = std::accumulate(a_ms_ks->mLengths.begin(),
                                                   a_ms_ks->mLengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t elementsB  = std::accumulate(b_ns_ks->mLengths.begin(),
                                                   b_ns_ks->mLengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t elementsCD = std::accumulate(d_ms_ns->mLengths.begin(),
                                                    d_ms_ns->mLengths.end(),
                                                    size_t{1},
                                                    std::multiplies<size_t>());

                auto D = resource->allocHost(elementsCD * size);
                resource->copyData(D, resource->deviceD(), elementsCD * size);

                auto& references = resource->hostD();

                if(DDataType == HIPTENSOR_R_16F)
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
                else if(DDataType == HIPTENSOR_R_16BF)
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
                else if(DDataType == HIPTENSOR_R_32F)
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
                else if(DDataType == HIPTENSOR_R_64F)
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
                else if(DDataType == HIPTENSOR_C_32F)
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
                else if(DDataType == HIPTENSOR_C_64F)
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
        auto operators    = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto strides      = std::get<6>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);
        auto memoryLayout = std::get<10>(param);

        ContractionTest::sAPILogBuff.str("");

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
             *      alpha, convertToComputeType(HipTensorDataType_v<typename Traits::ComputeDataT>));
             * ```
             * Hence, the `alpha` and `bete` need to point to a ComputeData value
             */
            ScalarData alphaBuf;
            ScalarData betaBuf;
            writeVal(&alphaBuf, computeType, ScalarData(computeType, alpha[0], alpha[1]));
            writeVal(&betaBuf, computeType, ScalarData(computeType, beta[0], beta[1]));

            /**************************
            * Disable Plan Cache for tests
            ***************************/
            if(!mEnablePlanCache)
            {
                const hiptensorCacheMode_t cacheMode = HIPTENSOR_CACHE_MODE_NONE;
                CHECK_HIPTENSOR_ERROR(
                    hiptensorPlanPreferenceSetAttribute(handle,
                                                        planPref,
                                                        HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE,
                                                        &cacheMode,
                                                        sizeof(hiptensorCacheMode_t)));
            }

            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

            auto resource = getResource();

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorContract(handle,
                                                    plan,
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

            size_t totalLength = std::accumulate(d_ms_ns->mLengths.begin(),
                                                 d_ms_ns->mLengths.end(),
                                                 size_t(1),
                                                 std::multiplies<size_t>());

            // TODO: make totalLength calculation robust to different modes
            // currently assumes modes in the following:
            // Rank 2 ex:
            // [[modesA], [modesB], [modesD]]
            // [[0, 1, 4, 5], [2, 3, 4, 5], [0, 1, 2, 3]]

            // iterate through "K" dimension
            auto rank = a_ms_ks->mLengths.size() / 2;
            for(auto i = 0; i < rank; i++)
            {
                totalLength *= a_ms_ks->mLengths[i + rank];
            }

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = 2.0 * totalLength * 1e-9;
            mMeasuredTFlopsPerSec = mTotalGFlops / mElapsedTimeMs;

            size_t sizeA = std::accumulate(a_ms_ks->mLengths.begin(),
                                           a_ms_ks->mLengths.end(),
                                           hiptensorDataTypeSize(ADataType),
                                           std::multiplies<size_t>());

            size_t sizeB = std::accumulate(b_ns_ks->mLengths.begin(),
                                           b_ns_ks->mLengths.end(),
                                           hiptensorDataTypeSize(BDataType),
                                           std::multiplies<size_t>());

            size_t sizeD = std::accumulate(d_ms_ns->mLengths.begin(),
                                           d_ms_ns->mLengths.end(),
                                           hiptensorDataTypeSize(DDataType),
                                           std::multiplies<size_t>());

            mTotalGBytes = sizeA + sizeB + sizeD;
            mTotalGBytes += (betaBuf.mReal != 0.0) ? sizeD : 0;
            mTotalGBytes /= 1e9;
            mGBytesPerSec = mTotalGBytes / (mElapsedTimeMs * 1e-3);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())
            {
                CHECK_HIPTENSOR_ERROR(hiptensorContractionReference(
                    plan,
                    (void*)&alphaBuf,
                    resource->hostA().get(),
                    resource->hostB().get(),
                    (void*)&betaBuf,
                    resource->hostC().get(),
                    resource->hostD().get(),
                    a_ms_ks->mLengths,
                    a_ms_ks->mStrides,
                    desc->mModeA,
                    b_ns_ks->mLengths,
                    b_ns_ks->mStrides,
                    desc->mModeB,
                    d_ms_ns->mLengths,
                    d_ms_ns->mStrides,
                    desc->mModeC,
                    d_ms_ns->mLengths,
                    d_ms_ns->mStrides,
                    desc->mModeD,
                    ADataType,
                    BDataType,
                    CDataType,
                    DDataType,
                    {plan->mOpDesc->mOpA, plan->mOpDesc->mOpB, plan->mOpDesc->mOpC},
                    workspace));

                auto reference = resource->allocDevice(sizeD);
                resource->copyData(reference, resource->hostD(), sizeD);

                // Compute tolerance based on compute type
                auto dimension = a_ms_ks->mLengths.size() / 2;
                auto nelems_k  = std::accumulate(a_ms_ks->mLengths.begin() + dimension,
                                                a_ms_ks->mLengths.end(),
                                                size_t{1},
                                                std::multiplies<size_t>());

                size_t elementsCD = sizeD / hiptensorDataTypeSize(ADataType);

                auto   eps       = getEpsilon(computeType);
                double tolerance = 2 * nelems_k * eps;

                // use the same default tolerance value as CK
                if(computeType == HIPTENSOR_COMPUTE_DESC_16BF || DDataType == HIPTENSOR_R_16BF)
                {
                    const double epsilon = std::pow(2, -7);
                    tolerance += epsilon * 2;
                }
                else if(computeType == HIPTENSOR_COMPUTE_DESC_16F || DDataType == HIPTENSOR_R_16F)
                {
                    const double epsilon = std::pow(2, -10);
                    tolerance += epsilon * 2;
                }

                if(DDataType == HIPTENSOR_R_16F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<_Float16>((_Float16*)resource->deviceD().get(),
                                                             (_Float16*)reference.get(),
                                                             elementsCD,
                                                             computeType,
                                                             tolerance);
                }
                else if(DDataType == HIPTENSOR_R_16BF)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<hip_bfloat16>(
                            (hip_bfloat16*)resource->deviceD().get(),
                            (hip_bfloat16*)reference.get(),
                            elementsCD,
                            computeType,
                            tolerance);
                }
                else if(DDataType == HIPTENSOR_R_32F || DDataType == HIPTENSOR_C_32F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float>((float*)resource->deviceD().get(),
                                                          (float*)reference.get(),
                                                          elementsCD,
                                                          computeType,
                                                          tolerance);
                }
                else if(DDataType == HIPTENSOR_R_64F || DDataType == HIPTENSOR_C_64F)
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
                std::cout << ContractionTest::sAPILogBuff.str();
            }

            if(loggingOptions->logOstream().isOpen())
            {
                loggingOptions->logOstream().fstream() << ContractionTest::sAPILogBuff.str();
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

    hiptensorMemoryLayout_t
        ContractionTest::inferMemoryLayout(const std::vector<std::vector<std::size_t>>& strides,
                                           const std::vector<std::vector<std::size_t>>& lengths,
                                           hiptensorMemoryLayout_t memoryLayout) const
    {
        // If no strides are provided, return the provided memory layout
        if(strides.empty())
        {
            return memoryLayout;
        }

        // Test if strides are in Row major
        bool isRowMajor = true;
        for(int t = 0; t < static_cast<int>(lengths.size()); t++)
        {
            if(strides[t][lengths[t].size() - 1] != 1)
            {
                isRowMajor = false;
                break;
            }
            for(int i = static_cast<int>(lengths[t].size()) - 2; i >= 0; --i)
            {
                if(strides[t][i] != strides[t][i + 1] * lengths[t][i + 1])
                {
                    isRowMajor = false;
                    break;
                }
            }
            if(!isRowMajor)
            {
                break;
            }
        }

        if(isRowMajor)
        {
            return HIPTENSOR_MEMORY_LAYOUT_ROW_MAJOR;
        }

        // Test if strides are in Column major
        for(int t = 0; t < static_cast<int>(lengths.size()); t++)
        {
            if(strides[t][0] != 1)
            {
                return HIPTENSOR_MEMORY_LAYOUT_OTHER;
            }
            for(int i = 1; i < static_cast<int>(lengths[t].size()); ++i)
            {
                if(strides[t][i] != strides[t][i - 1] * lengths[t][i - 1])
                {
                    return HIPTENSOR_MEMORY_LAYOUT_OTHER;
                }
            }
        }

        return HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR;
    }

    void ContractionTest::TearDown()
    {
        if(mRunFlag)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
            CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));

            if(a_ms_ks)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(a_ms_ks));
                a_ms_ks = nullptr;
            }
            if(b_ns_ks)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(b_ns_ks));
                b_ns_ks = nullptr;
            }
            if(c_ms_ns)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(c_ms_ns));
                c_ms_ns = nullptr;
            }
            if(d_ms_ns)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(d_ms_ns));
                d_ms_ns = nullptr;
            }
            HIPTENSOR_FREE_DEVICE(workspace);
        }
    }

} // namespace hiptensor
