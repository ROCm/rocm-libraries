/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <fstream>
#include <iterator>
#include <numeric>
#include <unordered_map>

// hiptensor includes
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include "common.hpp"
#include "yaml_parser.hpp"

namespace hiptensor
{
    struct NoneType;
    static constexpr hipDataType NONE_TYPE = (hipDataType)31;
    struct ContractionTestParams
    {
        using DataTypesT = std::vector<hipDataType>;

        using AlgorithmT    = hiptensorAlgo_t;
        using OperatorT     = hiptensorOperator_t;
        using WorkSizePrefT = hiptensorWorksizePreference_t;
        using LogLevelT     = hiptensorLogLevel_t;

        using LengthsT = std::vector<std::vector<std::size_t>>;
        using StridesT = std::vector<std::vector<std::size_t>>;
        using ModesT   = std::vector<std::vector<int32_t>>;
        using AlphaT   = std::vector<double>;
        using BetaT    = std::vector<double>;
        using RangesT   = std::vector<std::size_t>;

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
    };
}

#define MAX_ELEMENTS_PRINT_COUNT 512

int main(int argc, char* argv[])
{
    auto yee          = hiptensor::ContractionTestParams{};
    yee.mLogLevelMask = (hiptensorLogLevel_t)(HIPTENSOR_LOG_LEVEL_OFF);
    yee.mDataTypes    = {
        // clang-format off
                {HIP_R_32F, HIP_R_32F, hiptensor::NONE_TYPE, HIP_R_32F, HIP_R_32F}, // scale F32
                {HIP_C_32F, HIP_C_32F, hiptensor::NONE_TYPE, HIP_C_32F, HIP_C_32F}, // scale F32 Complex
                {HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F}, // bilinear F32
                {HIP_C_32F, HIP_C_32F, HIP_C_32F, HIP_C_32F, HIP_C_32F}, // bilinear F32 Complex
                {HIP_R_64F, HIP_R_64F, hiptensor::NONE_TYPE, HIP_R_64F, HIP_R_64F}, // scale F64
                {HIP_C_64F, HIP_C_64F, hiptensor::NONE_TYPE, HIP_C_64F, HIP_C_64F}, // scale F64 Complex
                {HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F}, // bilinear F64
                {HIP_C_64F, HIP_C_64F, HIP_C_64F, HIP_C_64F, HIP_C_64F}, // bilinear F64 Complex
        // clang-format on
    };
    yee.mAlgorithms
        = {HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_ALGO_DEFAULT_PATIENT, HIPTENSOR_ALGO_ACTOR_CRITIC};
    yee.mOperators = {HIPTENSOR_OP_IDENTITY};
    yee.mWorkSizePrefs
        = {HIPTENSOR_WORKSPACE_RECOMMENDED, HIPTENSOR_WORKSPACE_MIN, HIPTENSOR_WORKSPACE_MAX};
    yee.mLogLevelMask
        = {hiptensorLogLevel_t(HIPTENSOR_LOG_LEVEL_ERROR | HIPTENSOR_LOG_LEVEL_PERF_TRACE)};
    yee.mProblemLengths
        = {{{5, 6, 7, 8, 4, 2, 3, 4}, {1, 2, 3, 4}, {99, 12, 44, 31, 59, 23, 54, 22}},
           {{3, 3, 3, 3}, {3, 3, 3, 3}, {33, 33, 33, 33, 33}}};
    yee.mProblemModes = {{{5, 6, 7, 8, 4, 2, 3, 4}, {1, 2, 3, 4}, {99, 12, 44, 31, 59, 23, 54, 22}},
                         {{7, 7, 7}, {7, 7, 7, 7}, {77, 77, 77}}};
    yee.mProblemStrides = {{}};
    yee.mAlphas         = {{0}, {1}, {1}};
    yee.mBetas          = {{2}, {2}, {2}};
    yee.mProblemRanges = {{2, 1024, 2}, {2, 2048, 2, 16}};
    yee.mProblemRandRanges = {{2, 1024, 16}};

    struct TmpFileWrapper
    {
        TmpFileWrapper()
        {
            // use std C function to create tmp file since filesystem is not supported on some platforms
            char tmpFilenameBuf[L_tmpnam];
            std::tmpnam(tmpFilenameBuf);
            tmpFilename.assign(tmpFilenameBuf);
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
    if(!yee1)
    {
        return -1;
    }

    return 0;
}
