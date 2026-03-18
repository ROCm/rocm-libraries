// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>

namespace rocRoller
{
    namespace Client
    {
        struct RunParameters
        {
            int workgroupMappingValue;
            int numWGs;
        };

        struct BenchmarkParameters
        {
            int device;

            int    numWarmUp;
            int    numOuter;
            int    numInner;
            size_t rotatingBuffSize;

            bool check;
            bool visualize;
        };

        struct BenchmarkResults
        {
            std::string         resultType{"GEMM"};
            RunParameters       runParams;
            BenchmarkParameters benchmarkParams;

            size_t              kernelGenerate;
            size_t              kernelAssemble;
            std::vector<size_t> kernelExecute;
            bool                checked = false;
            bool                correct = true;
            double              rnorm   = 1.e12;

            // Max usage
            int sgprCount = -1;
            int vgprCount = -1;
            int agprCount = -1;

            int ldsBytes = -1;
        };
    }
}
