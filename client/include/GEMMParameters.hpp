/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <iostream>
#include <string>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "BenchmarkSolution.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            constexpr int elementsPerMXBlock = 32;

            /**
             * @brief Indicates whether a matrix is supplied in transposed form or not
             */
            enum class TransposeType
            {
                T,
                N,

                Count
            };

            std::string toString(TransposeType trans);

            /**
             * @brief Parameters of a GEMM problem
             *  D = alpha * A * B + beta * C
             * where
             * A is a m x k matrix,
             * B is a k x n matrix,
             * C and D are m x n matrices, and
             * alpha and beta are scalars
             */
            struct ProblemParameters
            {
                int   m;
                int   n;
                int   k;
                float alpha;
                float beta;

                std::string typeA;
                std::string typeB;
                std::string typeC;
                std::string typeD;
                std::string typeAcc;

                TransposeType transA;
                TransposeType transB;

                Operations::ScaleMode scaleA;
                Operations::ScaleMode scaleB;
            };

            /**
             * @brief Solution parameters common to all approaches to solving GEMMs.
             */
            struct SolutionParameters
            {
                GPUArchitectureTarget architecture;

                // Macro tile size
                int macM;
                int macN;
                int macK;

                // MFMA instruction
                int waveM = -1;
                int waveN = -1;
                int waveK = -1;
                int waveB = -1;

                // Number of wave tiles to execute per workgroup
                int workgroupSizeX = 64;
                int workgroupSizeY = 1;

                // Datatype of inputs and outputs
                std::string typeA;
                std::string typeB;
                std::string typeC;
                std::string typeD;
                std::string typeAcc;

                TransposeType transA;
                TransposeType transB;

                Operations::ScaleMode scaleA;
                Operations::ScaleMode scaleB;

                bool loadLDSScaleA = false;
                bool loadLDSScaleB = false;

                // Other options
                bool loadLDSA  = true;
                bool loadLDSB  = true;
                bool storeLDSD = true;

                bool direct2LDSA = false;
                bool direct2LDSB = false;

                bool prefetch          = false;
                int  prefetchInFlight  = 2;
                int  prefetchLDSFactor = 0;
                bool betaInFma         = true;

                // Unroll Options
                unsigned int unrollX = 0;
                unsigned int unrollY = 0;

                std::string scheduler;
                bool        matchMemoryAccess;

                bool streamK        = false;
                bool streamKTwoTile = false;

                std::string version;

                std::string generateKernelName() const;
            };

            struct Result
            {
                ProblemParameters                   problemParams;
                SolutionParameters                  solutionParams;
                rocRoller::Client::BenchmarkResults benchmarkResults;
            };

            std::ostream& operator<<(std::ostream&, TransposeType const&);
            std::ostream& operator<<(std::ostream&, ProblemParameters const&);
            std::ostream& operator<<(std::ostream&, SolutionParameters const&);
        }
    }
}
