// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <rocRoller/CommandSolution_fwd.hpp>

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/ExecutableKernel_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph_fwd.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    Generator<Instruction> kernelInstructions(ContextPtr                  context,
                                              CommandPtr                  command,
                                              KernelGraph::KernelGraphPtr kernelGraph);
}
