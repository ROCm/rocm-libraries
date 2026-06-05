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

#pragma once

#include <sstream>

#include <hiptensor/hiptensor_types.h>

#include "common.hpp"
#include "contraction_test_params.hpp"
#include "trinary_contraction_resource.hpp"

#include <gtest/gtest.h>

#define MaxNumDimsM 6
#define MaxNumDimsN 6

namespace hiptensor
{
    static void trinaryLogMessage(int32_t logLevel, const char* funcName = "", const char* msg = "");

    class TrinaryContractionTest
        : public ::testing::TestWithParam<std::tuple<typename ContractionTestParams::DataTypesT,
                                                     typename ContractionTestParams::AlgorithmT,
                                                     typename ContractionTestParams::OperatorT,
                                                     typename ContractionTestParams::WorkSizePrefT,
                                                     typename ContractionTestParams::LogLevelT,
                                                     typename ContractionTestParams::LengthsT,
                                                     typename ContractionTestParams::StridesT,
                                                     typename ContractionTestParams::ModesT,
                                                     typename ContractionTestParams::AlphaT,
                                                     typename ContractionTestParams::BetaT,
                                                     typename ContractionTestParams::MemoryLayoutT>>
    {
    protected:
        using Base
            = ::testing::TestWithParam<std::tuple<typename ContractionTestParams::DataTypesT,
                                                  typename ContractionTestParams::AlgorithmT,
                                                  typename ContractionTestParams::OperatorT,
                                                  typename ContractionTestParams::WorkSizePrefT,
                                                  typename ContractionTestParams::LogLevelT,
                                                  typename ContractionTestParams::LengthsT,
                                                  typename ContractionTestParams::StridesT,
                                                  typename ContractionTestParams::ModesT,
                                                  typename ContractionTestParams::AlphaT,
                                                  typename ContractionTestParams::BetaT,
                                                  typename ContractionTestParams::MemoryLayoutT>>;

        using DataStorage = TrinaryContractionResource;

        friend void trinaryLogMessage(int32_t, const char*, const char*);

    public:
        TrinaryContractionTest();
        virtual ~TrinaryContractionTest() = default;

    protected:
        TrinaryContractionTest(TrinaryContractionTest&&)            = delete;
        TrinaryContractionTest(TrinaryContractionTest const&)       = delete;
        TrinaryContractionTest& operator=(TrinaryContractionTest&)  = delete;
        TrinaryContractionTest& operator=(TrinaryContractionTest&&) = delete;

        bool checkDevice(hiptensorDataType_t          dataType,
                         hiptensorComputeDescriptor_t computeType) const;
        bool checkSizes() const;
        void reset();

        TrinaryContractionResource* getResource() const;

        std::ostream& printHeader(std::ostream& stream) const;
        std::ostream& printKernel(std::ostream& stream) const;

        void SetUp() final;
        void TearDown() final;

        void Warmup() {}
        void RunKernel();

        void reportResults(std::ostream&                stream,
                           hiptensorDataType_t          EDataType,
                           hiptensorComputeDescriptor_t computeType,
                           bool                         omitHeader,
                           bool                         omitSkipped,
                           bool                         omitFailed,
                           bool                         omitPassed) const;

    protected:
        hiptensorHandle_t              handle = nullptr;
        hiptensorPlan_t                plan;
        hiptensorOperationDescriptor_t desc;
        hiptensorPlanPreference_t      planPref;
        uint64_t                       worksize;
        void*                          workspace = nullptr;

        hiptensorTensorDescriptor_t descA = nullptr;
        hiptensorTensorDescriptor_t descB = nullptr;
        hiptensorTensorDescriptor_t descC = nullptr;
        hiptensorTensorDescriptor_t descD = nullptr;
        hiptensorTensorDescriptor_t descE = nullptr;

        uint32_t mRepeats;
        bool     mRunFlag          = true;
        bool     mValidationResult = false;
        bool     mPrintElements    = false;
        double   mMaxRelativeError;

        static bool              mHeaderPrinted;
        static std::stringstream sAPILogBuff;

        float64_t mElapsedTimeMs, mTotalGFlops, mMeasuredTFlopsPerSec, mTotalGBytes, mGBytesPerSec;
    };

} // namespace hiptensor
