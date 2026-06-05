/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstring>
#include <set>
#include <unordered_map>

#include "data_types.hpp"
#include "hiptensor_options.hpp"

#include "contraction/contraction_cpu_reference.hpp"
#include "trinary_contraction_test.hpp"
#include "util.hpp"
#include "utils.hpp"

namespace hiptensor
{
    /*static*/ bool              TrinaryContractionTest::mHeaderPrinted = false;
    /*static*/ std::stringstream TrinaryContractionTest::sAPILogBuff    = std::stringstream();

    template <typename DataType>
    using FillLaunchKernelFn = void (*)(DataType* data, uint32_t elementSize, uint32_t seed);

    static void trinaryLogMessage(int32_t logLevel,
                                  const char* funcName /*=""*/,
                                  const char* msg /*=""*/)
    {
        TrinaryContractionTest::sAPILogBuff << msg;
    }

    TrinaryContractionTest::TrinaryContractionTest()
        : Base()
    {
        reset();

        hiptensor::test::silenceLogger();
        hiptensorLoggerSetCallback(trinaryLogMessage);
    }

    bool TrinaryContractionTest::checkDevice(hiptensorDataType_t          dataType,
                                             hiptensorComputeDescriptor_t computeType) const
    {
        switch(computeType)
        {
        case HIPTENSOR_COMPUTE_DESC_16F:
        case HIPTENSOR_COMPUTE_DESC_16BF:
            if(isDataType16Bits(dataType))
                return isF16F16MatrixCoreSupported() && !isDataTypeComplex(dataType);
            else if(isDataType32Bits(dataType))
                return isF32F16MatrixCoreSupported() && !isDataTypeComplex(dataType);
            return false;
        case HIPTENSOR_COMPUTE_DESC_32F:
            if(isDataType16Bits(dataType))
                return isF16F32MatrixCoreSupported() && !isDataTypeComplex(dataType);
            else if(isDataType32Bits(dataType))
                return isF32F32MatrixCoreSupported() && !isDataTypeComplex(dataType);
            else if(isDataType64Bits(dataType))
                return isF64F32MatrixCoreSupported() && !isDataTypeComplex(dataType);
            return false;
        case HIPTENSOR_COMPUTE_DESC_64F:
            if(isDataType64Bits(dataType))
                return isF64F64MatrixCoreSupported() && !isDataTypeComplex(dataType);
            return false;
        default:
            return false;
        }
    }

    bool TrinaryContractionTest::checkSizes() const
    {
        return true;
    }

    void TrinaryContractionTest::reset()
    {
        handle    = nullptr;
        workspace = nullptr;
        worksize  = 0u;

        mRepeats          = 1u;
        mRunFlag          = true;
        mValidationResult = false;
        mMaxRelativeError = 0.0;

        mElapsedTimeMs = mTotalGFlops = mMeasuredTFlopsPerSec = mTotalGBytes = mGBytesPerSec = 0.0;
    }

    TrinaryContractionResource* TrinaryContractionTest::getResource() const
    {
        return DataStorage::instance().get();
    }

    std::ostream& TrinaryContractionTest::printHeader(std::ostream& stream) const
    {
        // clang-format off
        return stream
            << "TypeA, "
            << "TypeB, "
            << "TypeC, "
            << "TypeD, "
            << "TypeE, "
            << "TypeCompute, "
            << "Algorithm, "
            << "Operator, "
            << "WorkSizePreference, "
            << "LogLevel, "
            << "Lengths, "
            << "Modes, "
            << "Alpha, "
            << "Beta, "
            << "ElapsedMs, "
            << "Problem Size(GFlops), "
            << "TFlops/s, "
            << "TotalGBytes, "
            << "GBytes/s, "
            << "Result"
            << std::endl;
        // clang-format on
    }

    std::ostream& TrinaryContractionTest::printKernel(std::ostream& stream) const
    {
        auto param        = Base::GetParam();
        auto testType     = std::get<0>(param);
        auto algorithm    = std::get<1>(param);
        auto operators    = std::get<2>(param);
        auto workSizePref = std::get<3>(param);
        auto logLevel     = std::get<4>(param);
        auto lengths      = std::get<5>(param);
        auto modes        = std::get<7>(param);
        auto alpha        = std::get<8>(param);
        auto beta         = std::get<9>(param);

        // clang-format off
        stream
            << hipTypeToString(testType[0]) << ", "
            << hipTypeToString(testType[1]) << ", "
            << hipTypeToString(testType[2]) << ", "
            << hipTypeToString(testType[3]) << ", "
            << hipTypeToString(testType[4]) << ", "
            << computeTypeToString(convertToComputeType(testType[5])) << ", "
            << algoTypeToString(algorithm)  << ", "
            << "[ ";
        for(size_t i = 0; i < operators.size(); i++)
            stream << opTypeToString(operators[i]) << " ";
        stream << "], "
            << workSizePrefToString(workSizePref) << ", "
            << logLevelToString(logLevel) << ", ";
        printVectorInCsv(lengths, stream) << ", ";
        printContainerInCsv(modes, stream)   << ", ";
        printContainerInCsv(alpha, stream)   << ", ";
        printContainerInCsv(beta, stream)   << ", ";
        // clang-format on

        if(!mRunFlag)
        {
            stream << "n/a, n/a, n/a, n/a, n/a, SKIPPED" << std::endl;
        }
        else
        {
            auto isPerformValidation = HiptensorOptions::instance()->performValidation();
            auto result = isPerformValidation ? (mValidationResult ? "PASSED" : "FAILED") : "BENCH";

            // clang-format off
            stream
                << mElapsedTimeMs        << ", "
                << mTotalGFlops          << ", "
                << mMeasuredTFlopsPerSec << ", "
                << mTotalGBytes          << ", "
                << mGBytesPerSec         << ", "
                << result
                << std::endl;
            // clang-format on
        }

        return stream;
    }

    void TrinaryContractionTest::SetUp()
    {
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

        // Trinary: [typeA, typeB, typeC, typeD, typeE, typeCompute]
        EXPECT_EQ(dataTypes.size(), 6);
        // Trinary: [A, B, C, DE]
        EXPECT_TRUE(lengths.size() == 4);
        EXPECT_TRUE(modes.size() == 4);
        if(!strides.empty())
        {
            EXPECT_TRUE(strides.size() == 4);
        }

        if(strides.empty() && memoryLayout != HIPTENSOR_MEMORY_LAYOUT_DEFAULT)
        {
            strides.resize(lengths.size());
            for(int t = 0; t < static_cast<int>(lengths.size()); t++)
            {
                strides[t] = stridesFromLengths(
                    lengths[t], memoryLayout == HIPTENSOR_MEMORY_LAYOUT_COLUMN_MAJOR);
            }
        }

        for(int i = 0; i < static_cast<int>(lengths.size()); i++)
        {
            EXPECT_TRUE(lengths[i].size() <= MaxNumDimsM + MaxNumDimsN);
            if(!strides.empty())
            {
                EXPECT_TRUE(strides[i].size() == lengths[i].size());
            }
            EXPECT_TRUE(modes[i].size() == lengths[i].size());
        }

        auto computeType = convertToComputeType(dataTypes[5]);

        auto ADataType = dataTypes[0];
        auto BDataType = dataTypes[1];
        auto CDataType = dataTypes[2];
        auto DDataType = dataTypes[3];
        auto EDataType = dataTypes[4];

        EXPECT_TRUE((ADataType == HIPTENSOR_R_16F) || (ADataType == HIPTENSOR_R_16BF)
                    || (ADataType == HIPTENSOR_R_32F) || (ADataType == HIPTENSOR_R_64F));
        EXPECT_TRUE((BDataType == HIPTENSOR_R_16F) || (BDataType == HIPTENSOR_R_16BF)
                    || (BDataType == HIPTENSOR_R_32F) || (BDataType == HIPTENSOR_R_64F));
        EXPECT_TRUE((CDataType == HIPTENSOR_R_16F) || (CDataType == HIPTENSOR_R_16BF)
                    || (CDataType == HIPTENSOR_R_32F) || (CDataType == HIPTENSOR_R_64F));
        EXPECT_TRUE((DDataType == HIPTENSOR_R_16F) || (DDataType == HIPTENSOR_R_16BF)
                    || (DDataType == HIPTENSOR_R_32F) || (DDataType == HIPTENSOR_R_64F)
                    || (DDataType == NONE_TYPE));
        EXPECT_TRUE((EDataType == HIPTENSOR_R_16F) || (EDataType == HIPTENSOR_R_16BF)
                    || (EDataType == HIPTENSOR_R_32F) || (EDataType == HIPTENSOR_R_64F));
        EXPECT_TRUE((computeType == HIPTENSOR_COMPUTE_DESC_16F)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_16BF)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_32F)
                    || (computeType == HIPTENSOR_COMPUTE_DESC_64F));

        mRunFlag &= checkDevice(EDataType, computeType);

        if(!mRunFlag)
        {
            GTEST_SKIP();
        }
        else
        {
            // Tensor A
            std::vector<int64_t> a_lengths, a_strides;
            std::vector<int32_t> a_modes;
            for(int i = 0; i < static_cast<int>(lengths[0].size()); i++)
            {
                a_modes.push_back(modes[0][i]);
                a_lengths.push_back(lengths[0][i]);
                if(!strides.empty())
                    a_strides.push_back(strides[0][i]);
            }

            // Tensor B
            std::vector<int64_t> b_lengths, b_strides;
            std::vector<int32_t> b_modes;
            for(int i = 0; i < static_cast<int>(lengths[1].size()); i++)
            {
                b_modes.push_back(modes[1][i]);
                b_lengths.push_back(lengths[1][i]);
                if(!strides.empty())
                    b_strides.push_back(strides[1][i]);
            }

            // Tensor C
            std::vector<int64_t> c_lengths, c_strides;
            std::vector<int32_t> c_modes;
            for(int i = 0; i < static_cast<int>(lengths[2].size()); i++)
            {
                c_modes.push_back(modes[2][i]);
                c_lengths.push_back(lengths[2][i]);
                if(!strides.empty())
                    c_strides.push_back(strides[2][i]);
            }

            // Tensors D/E share modes and lengths
            std::vector<int64_t> de_lengths, de_strides;
            std::vector<int32_t> de_modes;
            for(int i = 0; i < static_cast<int>(lengths[3].size()); i++)
            {
                de_modes.push_back(modes[3][i]);
                de_lengths.push_back(lengths[3][i]);
                if(!strides.empty())
                    de_strides.push_back(strides[3][i]);
            }

            size_t elementsA = std::accumulate(a_lengths.begin(), a_lengths.end(),
                                               size_t{1}, std::multiplies<size_t>());
            size_t elementsB = std::accumulate(b_lengths.begin(), b_lengths.end(),
                                               size_t{1}, std::multiplies<size_t>());
            size_t elementsC = std::accumulate(c_lengths.begin(), c_lengths.end(),
                                               size_t{1}, std::multiplies<size_t>());
            size_t elementsDE = std::accumulate(de_lengths.begin(), de_lengths.end(),
                                                size_t{1}, std::multiplies<size_t>());

            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
            CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(logLevel));

            uint32_t alignmentRequirement = 1;

            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle, &descA, a_lengths.size(), a_lengths.data(),
                strides.empty() ? NULL : a_strides.data(), ADataType, alignmentRequirement));

            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle, &descB, b_lengths.size(), b_lengths.data(),
                strides.empty() ? NULL : b_strides.data(), BDataType, alignmentRequirement));

            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle, &descC, c_lengths.size(), c_lengths.data(),
                strides.empty() ? NULL : c_strides.data(), CDataType, alignmentRequirement));

            if(DDataType != NONE_TYPE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                    handle, &descD, de_lengths.size(), de_lengths.data(),
                    strides.empty() ? NULL : de_strides.data(), DDataType, alignmentRequirement));
            }

            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(
                handle, &descE, de_lengths.size(), de_lengths.data(),
                strides.empty() ? NULL : de_strides.data(), EDataType, alignmentRequirement));

            TrinaryContractionResource::ElementBytes elementBytes(
                hiptensorDataTypeSize(ADataType),
                hiptensorDataTypeSize(BDataType),
                hiptensorDataTypeSize(CDataType),
                hiptensorDataTypeSize(DDataType),
                hiptensorDataTypeSize(EDataType));

            auto resource = getResource();
            resource->resizeStorage(lengths, elementBytes);

            uint32_t seed = static_cast<uint32_t>(256);

            if(ADataType == HIPTENSOR_R_16F && BDataType == HIPTENSOR_R_16F
               && CDataType == HIPTENSOR_R_16F && EDataType == HIPTENSOR_R_16F)
            {
                fillLaunchKernel<_Float16>((_Float16*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<_Float16>((_Float16*)resource->deviceB().get(), elementsB, seed);
                fillLaunchKernel<_Float16>((_Float16*)resource->deviceC().get(), elementsC, seed + 1);
                if(DDataType == HIPTENSOR_R_16F)
                {
                    fillLaunchKernel<_Float16>((_Float16*)resource->deviceD().get(), elementsDE, seed + 2);
                }
                fillValLaunchKernel<_Float16>((_Float16*)resource->deviceE().get(),
                                              elementsDE,
                                              std::numeric_limits<_Float16>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_R_16BF && BDataType == HIPTENSOR_R_16BF
                    && CDataType == HIPTENSOR_R_16BF && EDataType == HIPTENSOR_R_16BF)
            {
                fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)resource->deviceB().get(), elementsB, seed);
                fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)resource->deviceC().get(), elementsC, seed + 1);
                if(DDataType == HIPTENSOR_R_16BF)
                {
                    fillLaunchKernel<hip_bfloat16>((hip_bfloat16*)resource->deviceD().get(), elementsDE, seed + 2);
                }
                fillValLaunchKernel<hip_bfloat16>(
                    (hip_bfloat16*)resource->deviceE().get(),
                    elementsDE,
                    std::numeric_limits<hip_bfloat16>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_R_32F && BDataType == HIPTENSOR_R_32F
                    && CDataType == HIPTENSOR_R_32F && EDataType == HIPTENSOR_R_32F)
            {
                fillLaunchKernel<float>((float*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<float>((float*)resource->deviceB().get(), elementsB, seed);
                fillLaunchKernel<float>((float*)resource->deviceC().get(), elementsC, seed + 1);
                if(DDataType == HIPTENSOR_R_32F)
                {
                    fillLaunchKernel<float>((float*)resource->deviceD().get(), elementsDE, seed + 2);
                }
                fillValLaunchKernel<float>((float*)resource->deviceE().get(),
                                           elementsDE,
                                           std::numeric_limits<float>::signaling_NaN());
            }
            else if(ADataType == HIPTENSOR_R_64F && BDataType == HIPTENSOR_R_64F
                    && CDataType == HIPTENSOR_R_64F && EDataType == HIPTENSOR_R_64F)
            {
                fillLaunchKernel<double>((double*)resource->deviceA().get(), elementsA, seed - 1);
                fillLaunchKernel<double>((double*)resource->deviceB().get(), elementsB, seed);
                fillLaunchKernel<double>((double*)resource->deviceC().get(), elementsC, seed + 1);
                if(DDataType == HIPTENSOR_R_64F)
                {
                    fillLaunchKernel<double>((double*)resource->deviceD().get(), elementsDE, seed + 2);
                }
                fillValLaunchKernel<double>((double*)resource->deviceE().get(),
                                            elementsDE,
                                            std::numeric_limits<double>::signaling_NaN());
            }

            resource->copyDeviceToHostAll(elementBytes);

            CHECK_HIPTENSOR_ERROR(hiptensorCreateContractionTrinary(
                handle,
                &desc,
                descA,
                a_modes.data(),
                operators[0],
                descB,
                b_modes.data(),
                operators[1],
                descC,
                c_modes.data(),
                operators[2],
                (DDataType != NONE_TYPE) ? descD : nullptr,
                (DDataType != NONE_TYPE) ? de_modes.data() : nullptr,
                operators[3],
                descE,
                de_modes.data(),
                computeType));

            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
                handle, &planPref, algorithm, HIPTENSOR_JIT_MODE_NONE));

            CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
                handle, desc, planPref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

            if(worksize > 0)
            {
                CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
            }
        }
    }

    void TrinaryContractionTest::reportResults(std::ostream&                stream,
                                               hiptensorDataType_t          EDataType,
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

        if((mRunFlag || !omitSkipped) && (mValidationResult || !omitFailed)
           && (!mValidationResult || !omitPassed))
        {
            printKernel(stream);
        }
    }

    void TrinaryContractionTest::RunKernel()
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

        TrinaryContractionTest::sAPILogBuff.str("");

        if(mRunFlag)
        {
            auto ADataType = dataTypes[0];
            auto BDataType = dataTypes[1];
            auto CDataType = dataTypes[2];
            auto DDataType = dataTypes[3];
            auto EDataType = dataTypes[4];

            auto computeType = convertToComputeType(dataTypes[5]);

            ScalarData alphaBuf;
            ScalarData betaBuf;
            writeVal(&alphaBuf, computeType, ScalarData(computeType, alpha[0], alpha[1]));
            writeVal(&betaBuf, computeType, ScalarData(computeType, beta[0], beta[1]));

            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

            auto resource = getResource();

            hipEvent_t startEvent, stopEvent;
            CHECK_HIP_ERROR(hipEventCreate(&startEvent));
            CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
            CHECK_HIP_ERROR(hipEventRecord(startEvent));

            CHECK_HIPTENSOR_ERROR(hiptensorContractTrinary(
                handle,
                plan,
                (void*)&alphaBuf,
                resource->deviceA().get(),
                resource->deviceB().get(),
                resource->deviceC().get(),
                (DDataType != NONE_TYPE) ? (void*)&betaBuf : nullptr,
                (DDataType != NONE_TYPE) ? resource->deviceD().get() : nullptr,
                resource->deviceE().get(),
                workspace,
                worksize,
                0 /* stream */));

            CHECK_HIP_ERROR(hipEventRecord(stopEvent));
            CHECK_HIP_ERROR(hipEventSynchronize(stopEvent))

            auto timeMs = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeMs, startEvent, stopEvent));

            mElapsedTimeMs        = float64_t(timeMs);
            mTotalGFlops          = 0.0;
            mMeasuredTFlopsPerSec = 0.0;

            size_t sizeA = std::accumulate(descA->mLengths.begin(), descA->mLengths.end(),
                                           hiptensorDataTypeSize(ADataType),
                                           std::multiplies<size_t>());
            size_t sizeB = std::accumulate(descB->mLengths.begin(), descB->mLengths.end(),
                                           hiptensorDataTypeSize(BDataType),
                                           std::multiplies<size_t>());
            size_t sizeC = std::accumulate(descC->mLengths.begin(), descC->mLengths.end(),
                                           hiptensorDataTypeSize(CDataType),
                                           std::multiplies<size_t>());
            size_t sizeE = std::accumulate(descE->mLengths.begin(), descE->mLengths.end(),
                                           hiptensorDataTypeSize(EDataType),
                                           std::multiplies<size_t>());

            mTotalGBytes = (sizeA + sizeB + sizeC + sizeE) / 1e9;
            mGBytesPerSec = mTotalGBytes / (mElapsedTimeMs * 1e-3);

            CHECK_HIP_ERROR(hipEventDestroy(startEvent));
            CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

            auto& testOptions = HiptensorOptions::instance();

            if(testOptions->performValidation())
            {
                // Compute intermediate tensor T modes using the same logic
                // as the library's computeIntermediateModes:
                //   T = non-contracted modes of A, then unique modes of B
                std::set<int32_t> setA(desc->mModeA.begin(), desc->mModeA.end());
                std::set<int32_t> setB(desc->mModeB.begin(), desc->mModeB.end());
                std::set<int32_t> setC(desc->mModeC.begin(), desc->mModeC.end());
                std::set<int32_t> setE(desc->mModeE.begin(), desc->mModeE.end());

                std::set<int32_t> contracted;
                for(auto m : setA)
                {
                    if(setB.count(m) && !setC.count(m) && !setE.count(m))
                        contracted.insert(m);
                }

                std::vector<int32_t> t_modes;
                for(auto m : desc->mModeA)
                {
                    if(!contracted.count(m))
                        t_modes.push_back(m);
                }
                for(auto m : desc->mModeB)
                {
                    if(!setA.count(m) && !contracted.count(m))
                        t_modes.push_back(m);
                }

                std::unordered_map<int32_t, size_t> lengthMap;
                for(size_t i = 0; i < desc->mModeA.size(); i++)
                    lengthMap[desc->mModeA[i]] = descA->mLengths[i];
                for(size_t i = 0; i < desc->mModeB.size(); i++)
                {
                    if(lengthMap.find(desc->mModeB[i]) == lengthMap.end())
                        lengthMap[desc->mModeB[i]] = descB->mLengths[i];
                }

                std::vector<size_t> t_lengths;
                for(auto m : t_modes)
                    t_lengths.push_back(lengthMap.at(m));

                // Column-major packed strides (matching the library)
                std::vector<size_t> t_strides(t_lengths.size(), 1);
                for(size_t i = 1; i < t_strides.size(); i++)
                    t_strides[i] = t_strides[i - 1] * t_lengths[i - 1];

                size_t elementsT = std::accumulate(t_lengths.begin(),
                                                   t_lengths.end(),
                                                   size_t{1},
                                                   std::multiplies<size_t>());
                size_t bytesT = elementsT * hiptensorDataTypeSize(EDataType);

                auto hostT = TrinaryContractionResource::allocHost(bytesT);
                memset(hostT.get(), 0, bytesT);

                ScalarData oneVal;
                ScalarData zeroVal;
                writeVal(&oneVal, computeType, ScalarData(computeType, 1.0, 0.0));
                writeVal(&zeroVal, computeType, ScalarData(computeType, 0.0, 0.0));

                // Step 1: T = 1.0 * opA(A) * opB(B) + 0*T
                // Uses bilinear with beta=0 (matching the GPU decomposition)
                CHECK_HIPTENSOR_ERROR(hiptensorContractionReference(
                    plan,
                    (void*)&oneVal,
                    resource->hostA().get(),
                    resource->hostB().get(),
                    (void*)&zeroVal,
                    hostT.get(),
                    hostT.get(),
                    descA->mLengths,
                    descA->mStrides,
                    desc->mModeA,
                    descB->mLengths,
                    descB->mStrides,
                    desc->mModeB,
                    t_lengths,
                    t_strides,
                    t_modes,
                    t_lengths,
                    t_strides,
                    t_modes,
                    ADataType,
                    BDataType,
                    EDataType,
                    EDataType,
                    {desc->mOpA, desc->mOpB, HIPTENSOR_OP_IDENTITY},
                    workspace));

                // Step 2: E = α * T * opC(C) + β * opD(D)
                if(DDataType != NONE_TYPE)
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorContractionReference(
                        plan,
                        (void*)&alphaBuf,
                        hostT.get(),
                        resource->hostC().get(),
                        (void*)&betaBuf,
                        resource->hostD().get(),
                        resource->hostE().get(),
                        t_lengths,
                        t_strides,
                        t_modes,
                        descC->mLengths,
                        descC->mStrides,
                        desc->mModeC,
                        descD->mLengths,
                        descD->mStrides,
                        desc->mModeD,
                        descE->mLengths,
                        descE->mStrides,
                        desc->mModeE,
                        EDataType,
                        CDataType,
                        DDataType,
                        EDataType,
                        {HIPTENSOR_OP_IDENTITY, desc->mOpC, desc->mOpD},
                        workspace));
                }
                else
                {
                    CHECK_HIPTENSOR_ERROR(hiptensorContractionReference(
                        plan,
                        (void*)&alphaBuf,
                        hostT.get(),
                        resource->hostC().get(),
                        nullptr,
                        nullptr,
                        resource->hostE().get(),
                        t_lengths,
                        t_strides,
                        t_modes,
                        descC->mLengths,
                        descC->mStrides,
                        desc->mModeC,
                        descE->mLengths,
                        descE->mStrides,
                        desc->mModeE,
                        descE->mLengths,
                        descE->mStrides,
                        desc->mModeE,
                        EDataType,
                        CDataType,
                        NONE_TYPE,
                        EDataType,
                        {HIPTENSOR_OP_IDENTITY, desc->mOpC, HIPTENSOR_OP_IDENTITY},
                        workspace));
                }

                auto reference = resource->allocDevice(sizeE);
                resource->copyData(reference, resource->hostE(), sizeE);

                // Tolerance: account for two chained contractions
                size_t nelems_k1 = 1;
                for(auto m : contracted)
                    nelems_k1 *= lengthMap[m];

                size_t nelems_k2 = 1;
                for(size_t i = 0; i < t_modes.size(); i++)
                {
                    if(setC.count(t_modes[i]))
                        nelems_k2 *= t_lengths[i];
                }

                size_t elementsE = sizeE / hiptensorDataTypeSize(EDataType);
                auto   eps       = getEpsilon(computeType);
                double tolerance = 2 * (nelems_k1 + nelems_k2) * eps;

                if(computeType == HIPTENSOR_COMPUTE_DESC_16BF
                   || EDataType == HIPTENSOR_R_16BF)
                {
                    const double epsilon = std::pow(2, -7);
                    tolerance += epsilon * 2;
                }
                else if(computeType == HIPTENSOR_COMPUTE_DESC_16F
                        || EDataType == HIPTENSOR_R_16F)
                {
                    const double epsilon = std::pow(2, -10);
                    tolerance += epsilon * 2;
                }

                if(EDataType == HIPTENSOR_R_16F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<_Float16>(
                            (_Float16*)resource->deviceE().get(),
                            (_Float16*)reference.get(),
                            elementsE,
                            computeType,
                            tolerance);
                }
                else if(EDataType == HIPTENSOR_R_16BF)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<hip_bfloat16>(
                            (hip_bfloat16*)resource->deviceE().get(),
                            (hip_bfloat16*)reference.get(),
                            elementsE,
                            computeType,
                            tolerance);
                }
                else if(EDataType == HIPTENSOR_R_32F || EDataType == HIPTENSOR_C_32F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<float>(
                            (float*)resource->deviceE().get(),
                            (float*)reference.get(),
                            elementsE,
                            computeType,
                            tolerance);
                }
                else if(EDataType == HIPTENSOR_R_64F || EDataType == HIPTENSOR_C_64F)
                {
                    std::tie(mValidationResult, mMaxRelativeError)
                        = compareEqualLaunchKernel<double>(
                            (double*)resource->deviceE().get(),
                            (double*)reference.get(),
                            elementsE,
                            computeType,
                            tolerance);
                }

                EXPECT_TRUE(mValidationResult)
                    << "Max relative error: " << mMaxRelativeError;
            }
            else
            {
                mValidationResult = true;
            }

            using Options        = hiptensor::HiptensorOptions;
            auto& loggingOptions = Options::instance();

            if(!loggingOptions->omitCout())
            {
                reportResults(std::cout,
                              EDataType,
                              computeType,
                              mHeaderPrinted,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
                std::cout << TrinaryContractionTest::sAPILogBuff.str();
            }

            if(loggingOptions->ostream().isOpen())
            {
                reportResults(loggingOptions->ostream().fstream(),
                              EDataType,
                              computeType,
                              mHeaderPrinted,
                              loggingOptions->omitSkipped(),
                              loggingOptions->omitFailed(),
                              loggingOptions->omitPassed());
            }

            if(!mHeaderPrinted)
            {
                mHeaderPrinted = true;
            }
        }
    }

    void TrinaryContractionTest::TearDown()
    {
        if(mRunFlag)
        {
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
            if(descE)
            {
                CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descE));
                descE = nullptr;
            }
            HIPTENSOR_FREE_DEVICE(workspace);
        }
    }

} // namespace hiptensor
