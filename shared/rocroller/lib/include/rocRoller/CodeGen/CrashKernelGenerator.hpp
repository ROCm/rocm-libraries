// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/AssertOpKinds_fwd.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    class CrashKernelGenerator
    {
    public:
        CrashKernelGenerator(ContextPtr);

        ~CrashKernelGenerator();

        Generator<Instruction> generateCrashSequence(AssertOpKind assertOpKind);

    private:
        std::weak_ptr<Context> m_context;

        Generator<Instruction> writeToNullPtr();
        Generator<Instruction> sTrap();
    };
}
