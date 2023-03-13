
#pragma once

#include <string>
namespace rocRoller
{
    namespace Scheduling
    {
        enum class CostProcedure : int
        {
            None = 0,
            Uniform,
            MinNops,
            WaitCntNop,
            Count
        };

        class Cost;

        std::string toString(CostProcedure);
    }
}
