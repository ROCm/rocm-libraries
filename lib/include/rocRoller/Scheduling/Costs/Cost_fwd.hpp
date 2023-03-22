
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
            Count
        };

        class Cost;

        std::string toString(CostFunction);
        std::string ToString(CostFunction);
    }
}
