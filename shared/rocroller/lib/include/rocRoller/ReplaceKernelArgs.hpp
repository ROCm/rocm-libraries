// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace Expression
    {
        /**
          * Replace AssemblyKernelArgument sub-expressions with registers that
          * contain their values. This may possibly require some instructions so
          * this is a Generator and it returns the modified expression in `dst`.
          */
        Generator<Instruction> replaceKernelArgs(ContextPtr const&    context,
                                                 ExpressionPtr&       dst,
                                                 ExpressionPtr const& src);
    }
}
