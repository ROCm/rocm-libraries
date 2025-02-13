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

                // Datatype of inputs and outputs
                std::string typeA;
                std::string typeB;
                std::string typeC;
                std::string typeD;
                std::string typeAcc;

                TransposeType transA;
                TransposeType transB;
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
