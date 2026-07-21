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

#pragma once

#include <sstream>

#include <hiptensor/hiptensor_types.h>

#include "common.hpp"
#include "contraction_resource.hpp"
#include "contraction_test_params.hpp"

#include <gtest/gtest.h>

#define MaxNumDimsM 6
#define MaxNumDimsN 6

namespace hiptensor
{
    static void logMessage(int32_t logLevel, const char* funcName = "", const char* msg = "");

    class ContractionTest
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
    protected: // Types
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

        // Shared access to Contraction storage
        using DataStorage = ContractionResource;

        friend void logMessage(int32_t, const char*, const char*);

    public:
        ContractionTest();
        virtual ~ContractionTest() = default;

        void EnablePlanCache()
        {
            mEnablePlanCache = true;
        }

    protected: // Functions
        ContractionTest(ContractionTest&&)            = delete;
        ContractionTest(ContractionTest const&)       = delete;
        ContractionTest& operator=(ContractionTest&)  = delete;
        ContractionTest& operator=(ContractionTest&&) = delete;

        bool checkDevice(hiptensorDataType_t          dataType,
                         hiptensorComputeDescriptor_t computeType) const;
        bool checkSizes() const;
        void reset();

        ContractionResource* getResource() const;

        std::ostream& printHeader(std::ostream& stream) const;
        std::ostream& printKernel(std::ostream& stream) const;

        void SetUp() final;
        void TearDown() final;

        void Warmup() {}
        void RunKernel();

        void reportResults(std::ostream&                stream,
                           hiptensorDataType_t          DDataType,
                           hiptensorComputeDescriptor_t computeType,
                           bool                         omitHeader,
                           bool                         omitSkipped,
                           bool                         omitFailed,
                           bool                         omitPassed) const;

        hiptensorMemoryLayout_t
            inferMemoryLayout(const std::vector<std::vector<std::size_t>>& strides,
                              const std::vector<std::vector<std::size_t>>& lengths,
                              hiptensorMemoryLayout_t                      memoryLayout) const;

    protected:
        // Workspace items
        hiptensorHandle_t              handle = nullptr;
        hiptensorPlan_t                plan;
        hiptensorOperationDescriptor_t desc;
        hiptensorPlanPreference_t      planPref;
        uint64_t                       worksize;
        void*                          workspace = nullptr;

        hiptensorTensorDescriptor_t a_ms_ks = nullptr;
        hiptensorTensorDescriptor_t b_ns_ks = nullptr;
        hiptensorTensorDescriptor_t c_ms_ns = nullptr;
        hiptensorTensorDescriptor_t d_ms_ns = nullptr;

        // Execution flow control
        uint32_t mRepeats;
        bool     mRunFlag          = true;
        bool     mValidationResult = false;
        bool     mPrintElements    = false;
        bool     mPrintTypes       = false;
        double   mMaxRelativeError;

        static bool mHeaderPrinted;

        //Enable plan cache
        bool mEnablePlanCache = false;

        // Output buffer
        static std::stringstream sAPILogBuff;

        // Performance
        float64_t mElapsedTimeMs, mTotalGFlops, mMeasuredTFlopsPerSec, mTotalGBytes, mGBytesPerSec;
    };

} // namespace hiptensor
