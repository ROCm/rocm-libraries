// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
namespace rocRoller
{
    namespace Scheduling
    {
        enum class CostFunction : int
        {
            None = 0,
            Uniform,
            MinNops,
            WaitCntNop,
            LinearWeighted,
            LinearWeightedSimple,
            LinearWeightedSimpleStreamK,
            Count
        };

        class Cost;

        std::string toString(CostFunction);
    }
}
