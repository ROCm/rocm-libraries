// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>

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

        ConstraintStatus NoDanglingMappings(KernelGraph const& k);
        ConstraintStatus SingleControlRoot(KernelGraph const& k);
        ConstraintStatus NoRedundantSetCoordinates(KernelGraph const& k);
        ConstraintStatus WalkableControlGraph(KernelGraph const& k);

        using GraphConstraint = ConstraintStatus (*)(const KernelGraph& k);
    }
}
