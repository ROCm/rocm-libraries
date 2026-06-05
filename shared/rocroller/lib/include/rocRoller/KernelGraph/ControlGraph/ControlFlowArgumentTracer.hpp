// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

#include <rocRoller/KernelGraph/RegisterTagManager_fwd.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    class ControlFlowArgumentTracer
    {
    public:
        ControlFlowArgumentTracer(KernelGraph const& kgraph, AssemblyKernelPtr const& kernel);

        std::unordered_set<std::string> const& referencedArguments(int controlNode) const;

        /**
         * Map of control node -> arguments referenced by that control node.
         */
        std::unordered_map<int, std::unordered_set<std::string>> const& referencedArguments() const;

        /**
         * Any arguments that we know will never be referenced in the kernel.
         */
        std::set<std::string> const& neverReferencedArguments() const;

    private:
        std::unordered_map<int, std::unordered_set<std::string>> m_referencedArguments;

        std::set<std::string> m_neverReferencedArguments;
    };
}
