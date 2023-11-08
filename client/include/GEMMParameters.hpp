#pragma once

#include <string>
#include <vector>

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
         *
         */
            enum class TransposeType
            {
                T,
                N,

                Count
            };

            std::string toString(TransposeType trans);

            /**
         * @brief Parameters of a GEMM problem.
         *  D = alpha * A * B + beta * C
         * where
         * A is a m x k matrix,
         * B is a k x n matrix,
         * C and D are m x n matrices, and
         * alpha and beta are scalars
         *
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
         *
         */
            struct SolutionParameters
            {
                ProblemParameters problemParams;

                // Macro Tile Size
                int macM;
                int macN;
                int macK;

                // Number of wave tiles to execute per workgroup
                int workgroupSizeX = 64;
                int workgroupSizeY = 1;

                bool loadLDSA  = true;
                bool loadLDSB  = true;
                bool storeLDSD = true;

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
                int  numWGs         = 0;
                bool streamKTwoTile = false;

                std::string generateKernelName() const;
            };

            struct Result
            {
                SolutionParameters                  solutionParams;
                rocRoller::Client::BenchmarkResults benchmarkResults;
            };
        }
    }
}
