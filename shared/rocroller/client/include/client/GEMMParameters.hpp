// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Parameters/Solution/LDSBankSwizzleMode.hpp>
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Parameters/Solution/ScaleSkipPermlaneMode.hpp>
#include <rocRoller/Parameters/Solution/StoreOption.hpp>
#include <rocRoller/Parameters/Solution/StreamK.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "client/BenchmarkSolution.hpp"
#include <mxDataGenerator/DataGenerator.hpp>

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

            struct XYTuple
            {
                int x, y;
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

            struct KernelNames
            {
                std::string fullName;
                std::string shortName;
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

                ScaleSkipPermlaneMode scaleSkipPermlane = ScaleSkipPermlaneMode::None;

                std::vector<size_t> scalePretileA;
                std::vector<size_t> scalePretileB;

                // Order: M/N, K tile, K subtile
                std::vector<size_t> scaleShuffleTileA;
                std::vector<size_t> scaleShuffleTileB;

                std::vector<size_t> pretileA;
                std::vector<size_t> pretileB;

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

                DGen::DataInitMode initModeA, initModeB, initModeC;

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

                Parameters::Solution::LoadPath loadPathAScale{
                    Parameters::Solution::LoadPath::BufferToVGPR};
                Parameters::Solution::LoadPath loadPathBScale{
                    Parameters::Solution::LoadPath::BufferToVGPR};

                bool      swizzleScale    = false;
                MKNLTuple swizzleTileSize = {0, 0, 0, 0};
                bool      prefetchScale   = false;

                // Other options
                Parameters::Solution::LoadPath loadPathA{
                    Parameters::Solution::LoadPath::BufferToLDSViaVGPR};
                Parameters::Solution::LoadPath loadPathB{
                    Parameters::Solution::LoadPath::BufferToLDSViaVGPR};
                Parameters::Solution::StorePath storePath{
                    Parameters::Solution::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer};

                std::pair<int, int> padLDSA = {0, 0};
                std::pair<int, int> padLDSB = {0, 0};

                bool prefetch          = false;
                int  prefetchInFlight  = 2;
                int  prefetchLDSFactor = 0;
                bool prefetchMixMemOps = false;
                bool betaInFma         = true;

                std::string scheduler;
                std::string schedulerCost;

                bool tailLoops = true;

                StreamKMode streamK = StreamKMode::None;

                LDSBankSwizzleMode ldsBankSwizzle = LDSBankSwizzleMode::None;

                std::string version;

                KernelNames generateKernelName() const;
            };

            struct Result
            {
                ProblemParameters                   problemParams;
                SolutionParameters                  solutionParams;
                rocRoller::Client::BenchmarkResults benchmarkResults;
            };

            std::ostream& operator<<(std::ostream& s, XYTuple const& x);
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
     * @brief Parse an X,Y pair (negative OK).
     */
    bool ParseIntPair(const std::string& arg, std::pair<int, int>& x);

    /**
     * @brief Parse an XxY pair.
     */
    bool ParseXY(const std::string& arg, XYTuple& x);

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

    /**
     * @brief Parse a DataInitMode variant from a string.
     *
     * Asserts that argument is well-formed.
     */
    bool ParseInitMode(const std::string& arg, DGen::DataInitMode& result);
}
