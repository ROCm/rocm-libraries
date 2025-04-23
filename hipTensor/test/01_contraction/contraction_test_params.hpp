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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_TEST_PARAMS_HPP
#define HIPTENSOR_CONTRACTION_TEST_PARAMS_HPP

#include <tuple>
#include <vector>

#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>

#include "common.hpp"
#include "utils.hpp"

namespace hiptensor
{

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
        using RangesT  = std::vector<std::size_t>;

    public:
        std::vector<DataTypesT>& dataTypes()
        {
            return mDataTypes;
        }

        std::vector<AlgorithmT>& algorithms()
        {
            return mAlgorithms;
        }

        std::vector<OperatorT>& operators()
        {
            return mOperators;
        }

        std::vector<WorkSizePrefT>& workSizePrefrences()
        {
            return mWorkSizePrefs;
        }

        LogLevelT& logLevelMask()
        {
            return mLogLevelMask;
        }

        std::vector<LengthsT>& problemLengths()
        {
            return mProblemLengths;
        }

        std::vector<StridesT>& problemStrides()
        {
            return mProblemStrides;
        }

        std::vector<ModesT>& problemModes()
        {
            return mProblemModes;
        }

        std::vector<AlphaT>& alphas()
        {
            return mAlphas;
        }

        std::vector<BetaT>& betas()
        {
            return mBetas;
        }

        std::vector<RangesT>& problemRanges()
        {
            return mProblemRanges;
        }

        std::vector<RangesT>& problemRandRanges()
        {
            return mProblemRandRanges;
        }

        void printParams()
        {
            std::cout << "DataTypes: " << mDataTypes << std::endl
                      << "Algorithms: " << mAlgorithms << std::endl
                      << "Operators: " << mOperators << std::endl
                      << "WorkSizePrefs: " << mWorkSizePrefs << std::endl
                      << "LogLevelMask: " << mLogLevelMask << std::endl
                      << "ProblemLengths: " << mProblemLengths << std::endl
                      << "ProblemStrides: " << mProblemStrides << std::endl
                      << "ProblemModes: " << mProblemModes << std::endl
                      << "Alphas: " << mAlphas << std::endl
                      << "Betas: " << mBetas << std::endl
                      << "ProblemRanges: " << mProblemRanges << std::endl;
        }

    private:
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

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_TEST_PARAMS_HPP
