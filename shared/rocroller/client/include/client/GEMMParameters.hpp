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
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "client/BenchmarkSolution.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            /**
             * @brief Indicates whether a matrix is supplied in transposed form or not
             */
            enum class TransposeType
            {
                T,
                N,

                Count
            };

            struct MNKTuple
            {
                int m, n, k;
            };

            struct MNKBTuple
            {
                int m, n, k, b;
            };

            struct MKNLTuple
            {
                int m, k, n, l;
            };

            std::string toString(TransposeType trans);

            struct TypeParameters
            {
                std::string typeA   = "float";
                std::string typeB   = "float";
                std::string typeC   = "float";
                std::string typeD   = "float";
                std::string typeAcc = "float";

                Client::GEMMClient::TransposeType transA = Client::GEMMClient::TransposeType::N;
                Client::GEMMClient::TransposeType transB = Client::GEMMClient::TransposeType::N;

                Operations::ScaleMode scaleA     = Operations::ScaleMode::None;
                DataType              scaleTypeA = DataType::None;
                Operations::ScaleMode scaleB     = Operations::ScaleMode::None;
                DataType              scaleTypeB = DataType::None;

                int scaleBlockSize = -1;

                bool scaleSkipPermlane = false;

                // Order: M/N, K tile, K subtile
                std::vector<size_t> scaleShuffleTileA;
                std::vector<size_t> scaleShuffleTileB;

                std::string kernelNamePart() const;
            };

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
                size_t m;
                size_t n;
                size_t k;
                float  alpha;
                float  beta;

                TypeParameters types;

                // When scaleA/B is ScaleMode::SingleScale
                float scaleValueA, scaleValueB;

                int workgroupMappingDim;
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

                int  workgroupMappingDim    = -1;
                bool workgroupRemapXCC      = false;
                int  workgroupRemapXCCValue = -1;

                // Datatype of inputs and outputs
                TypeParameters types;

                bool loadLDSScaleA = false;
                bool loadLDSScaleB = false;

                bool      swizzleScale    = false;
                MKNLTuple swizzleTileSize = {0, 0, 0, 0};
                bool      prefetchScale   = false;

                // Other options
                Parameters::Solution::LoadPath loadPathA{
                    Parameters::Solution::LoadPath::BufferToLDSViaVGPR};
                Parameters::Solution::LoadPath loadPathB{
                    Parameters::Solution::LoadPath::BufferToLDSViaVGPR};
                bool storeLDSD = true;

                bool prefetch          = false;
                int  prefetchInFlight  = 2;
                int  prefetchLDSFactor = 0;
                bool prefetchMixMemOps = false;
                bool betaInFma         = true;

                // Unroll Options
                unsigned int unrollX = 0;
                unsigned int unrollY = 0;

                std::string scheduler;
                std::string schedulerCost;
                bool        matchMemoryAccess;

                // TODO Use StreamKConfig
                bool streamK               = false;
                bool streamKTwoTile        = false;
                bool streamKTwoTileDPFirst = false;

                std::string version;

                std::string generateKernelName() const;
            };

            struct Result
            {
                ProblemParameters                   problemParams;
                SolutionParameters                  solutionParams;
                rocRoller::Client::BenchmarkResults benchmarkResults;
            };

            std::ostream& operator<<(std::ostream& s, MNKTuple const& x);
            std::ostream& operator<<(std::ostream& s, MNKBTuple const& x);
            std::ostream& operator<<(std::ostream& s, MKNLTuple const& x);
            std::ostream& operator<<(std::ostream& s, TransposeType const& x);
            std::ostream& operator<<(std::ostream& s, TypeParameters const& x);
            std::ostream& operator<<(std::ostream& s, ProblemParameters const& x);
            std::ostream& operator<<(std::ostream& s, SolutionParameters const& x);
        }
    }
}

namespace rocRoller::Client::GEMMClient::CLI
{
    constexpr bool PARSE_SUCCESS = true;
    constexpr bool PARSE_FAILURE = false;

    /**
     * @brief Parse an MxNxK or MxNxKxB tuple from a string.
     *
     * If B is missing, it is set to 1.
     *
     * Asserts that all values are positive.
     */
    bool ParseMNKB(const std::string& arg, rocRoller::Client::GEMMClient::MNKBTuple& x);

    /**
     * @brief Parse an MxNxK tuple from a string.
     *
     * Asserts that all values are positive.
     */
    bool ParseMNK(const std::string& arg, rocRoller::Client::GEMMClient::MNKTuple& x);

    /**
     * @brief Parse an MxK/NxL tuple from a string.
     *
     * Asserts that all values are positive.
     */
    bool ParseMKNL(const std::string& arg, rocRoller::Client::GEMMClient::MKNLTuple& x);
}
