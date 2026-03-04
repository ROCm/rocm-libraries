// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>

namespace rocRoller::KernelGraph
{
    namespace SortArguments_detail
    {
        /**
         * @brief Sort the arguments by their first use in the control graph.
         */
        void sortArgumentsByFirstUse(KernelGraph const&                   graph,
                                     AssemblyKernelPtr                    kernel,
                                     std::vector<AssemblyKernelArgument>& arguments);
    }
}
