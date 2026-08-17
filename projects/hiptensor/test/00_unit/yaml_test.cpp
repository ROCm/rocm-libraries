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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <numeric>
#include <unordered_map>

// hiptensor includes
#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include "common.hpp"
#include "yaml_parser.hpp"

namespace hiptensor
{
    struct NoneType;
    static constexpr hiptensorDataType_t NONE_TYPE = (hiptensorDataType_t)31;
    struct ContractionTestParams
    {
        using DataTypesT = std::vector<hiptensorDataType_t>;

        using AlgorithmT    = hiptensorAlgo_t;
        using OperatorT     = hiptensorOperator_t;
        using WorkSizePrefT = hiptensorWorksizePreference_t;
        using LogLevelT     = hiptensorLogLevel_t;
        using MemoryLayoutT = hiptensorMemoryLayout_t;

        using LengthsT = std::vector<std::vector<std::size_t>>;
        using StridesT = std::vector<std::vector<std::size_t>>;
        using ModesT   = std::vector<std::vector<int32_t>>;
        using AlphaT   = std::vector<double>;
        using BetaT    = std::vector<double>;
        using RangesT  = std::vector<std::size_t>;

        //Data types of input and output tensors
        std::vector<DataTypesT>    mDataTypes;
        std::vector<AlgorithmT>    mAlgorithms;
        std::vector<OperatorT>     mOperators;
        std::vector<WorkSizePrefT> mWorkSizePrefs;
        LogLevelT                  mLogLevelMask;
        std::vector<LengthsT>      mProblemLengths;
        std::vector<StridesT>      mProblemStrides;
        std::vector<ModesT>        mProblemModes;
        std::vector<AlphaT>        mAlphas;
        std::vector<BetaT>         mBetas;
        std::vector<RangesT>       mProblemRanges;
        std::vector<RangesT>       mProblemRandRanges;
        std::vector<MemoryLayoutT> mMemoryLayouts;
    };
}

#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    auto yee          = hiptensor::ContractionTestParams{};
    yee.mLogLevelMask = (hiptensorLogLevel_t)(HIPTENSOR_LOG_LEVEL_OFF);
    yee.mDataTypes    = {
        // clang-format off
                {HIPTENSOR_R_32F, HIPTENSOR_R_32F, hiptensor::NONE_TYPE, HIPTENSOR_R_32F, HIPTENSOR_R_32F}, // scale F32
                {HIPTENSOR_C_32F, HIPTENSOR_C_32F, hiptensor::NONE_TYPE, HIPTENSOR_C_32F, HIPTENSOR_C_32F}, // scale F32 Complex
                {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F}, // bilinear F32
                {HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_C_32F, HIPTENSOR_C_32F}, // bilinear F32 Complex
                {HIPTENSOR_R_64F, HIPTENSOR_R_64F, hiptensor::NONE_TYPE, HIPTENSOR_R_64F, HIPTENSOR_R_64F}, // scale F64
                {HIPTENSOR_C_64F, HIPTENSOR_C_64F, hiptensor::NONE_TYPE, HIPTENSOR_C_64F, HIPTENSOR_C_64F}, // scale F64 Complex
                {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}, // bilinear F64
                {HIPTENSOR_C_64F, HIPTENSOR_C_64F, HIPTENSOR_C_64F, HIPTENSOR_C_64F, HIPTENSOR_C_64F}, // bilinear F64 Complex
        // clang-format on
    };
    yee.mAlgorithms
        = {HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT, HIPTENSOR_ALGO_ACTOR_CRITIC};
    yee.mOperators = {HIPTENSOR_OP_IDENTITY};
    yee.mWorkSizePrefs
        = {HIPTENSOR_WORKSPACE_DEFAULT, HIPTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_MAX};
    yee.mLogLevelMask
        = {hiptensorLogLevel_t(HIPTENSOR_LOG_LEVEL_ERROR | HIPTENSOR_LOG_LEVEL_PERF_TRACE)};
    yee.mProblemLengths
        = {{{5, 6, 7, 8, 4, 2, 3, 4}, {1, 2, 3, 4}, {99, 12, 44, 31, 59, 23, 54, 22}},
           {{3, 3, 3, 3}, {3, 3, 3, 3}, {33, 33, 33, 33, 33}}};
    yee.mProblemModes = {{{5, 6, 7, 8, 4, 2, 3, 4}, {1, 2, 3, 4}, {99, 12, 44, 31, 59, 23, 54, 22}},
                         {{7, 7, 7}, {7, 7, 7, 7}, {77, 77, 77}}};
    yee.mProblemStrides    = {{}};
    yee.mAlphas            = {{0}, {1}, {1}};
    yee.mBetas             = {{2}, {2}, {2}};
    yee.mProblemRanges     = {{2, 1024, 2}, {2, 2048, 2, 16}};
    yee.mProblemRandRanges = {{2, 1024, 16}};
    yee.mMemoryLayouts     = {HIPTENSOR_MEMORY_LAYOUT_DEFAULT};

    struct TmpFileWrapper
    {
        TmpFileWrapper()
        {
            // Use cross-platform filesystem approach for temporary file
            tmpFilename = hiptensor::test::generateTempFilename("hiptensor_yaml_test_");
        }
        ~TmpFileWrapper()
        {
            ::remove(tmpFilename.c_str());
        }
        std::string getTmpFile() const
        {
            return tmpFilename;
        }

    private:
        std::string tmpFilename;
    } tmpDirWrapper;

    auto tmpFile = tmpDirWrapper.getTmpFile();
    hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::storeToFile(tmpFile, yee);
    auto yee1
        = hiptensor::YamlConfigLoader<hiptensor::ContractionTestParams>::loadFromFile(tmpFile);

    int result = yee1 ? 0 : -1;

    // Remove temp file explicitly before quick_exit() bypasses the destructor.
    ::remove(tmpFile.c_str());

    // Flush LLVM's buffered streams and destroy ManagedStatic objects while all
    // libraries are still loaded, before post-main cleanup runs.
    hiptensor::llvmShutdown();

    // Use quick_exit() to bypass C++ static destructors and atexit() handlers.
    // Post-main cleanup in linked DLLs (hiptensor, HIP runtime) crashes under
    // ctest due to DLL unload order differences vs a direct run. Our cleanup is
    // already complete via llvmShutdown() above.
    std::quick_exit(result);
}
