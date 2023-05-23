#pragma once

#include "KernelGraph_fwd.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Return value for any function that applies constraints to the KernelGraph.
         *
         */
        struct ConstraintStatus
        {
            bool        satisfied   = true;
            std::string explanation = "";

            void combine(bool inputSat, std::string inputExpl)
            {
                satisfied &= inputSat;
                if(!explanation.empty() && !inputExpl.empty())
                {
                    explanation += "\n";
                }
                explanation += inputExpl;
            }

            void combine(ConstraintStatus input)
            {
                combine(input.satisfied, input.explanation);
            }
        };

        ConstraintStatus NoDanglingMappings(const KernelGraph& k);

        using GraphConstraint = ConstraintStatus (*)(const KernelGraph& k);
    }
}
