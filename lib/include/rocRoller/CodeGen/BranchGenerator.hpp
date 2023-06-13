/**
 * @brief
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#pragma once

#include "Instruction.hpp"

#include "../Context_fwd.hpp"
#include "../Utilities/Component.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    /**
     * @brief Generator for generating conditional and unconditional branches.
     */
    class BranchGenerator
    {
    public:
        BranchGenerator(ContextPtr);

        ~BranchGenerator();

        /**
         * @brief Generate an unconditional branch.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         */
        Generator<Instruction> branch(Register::ValuePtr destLabel, std::string comment = "");

        /**
         * @brief Generate an conditional branch.
         *
         * If condition is SCC or VCC, the corressponding branch instruction is generated.
         * If condition is a scalar register it is copied into VCC and a VCC branch is generated.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param condition The register to conditionally branch on.
         * @param zero=true If true branch zero is used, if false branch nonzero is used
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         */
        Generator<Instruction> branchConditional(Register::ValuePtr destLabel,
                                                 Register::ValuePtr condition,
                                                 bool               zero    = true,
                                                 std::string        comment = "");

        /**
         * @brief Generates a branch on zero instruction.
         *
         * Calls branchConditional with zero=true.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param condition The register to conditionally branch on.
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         * @sa branchConditional
         */
        Generator<Instruction> branchIfZero(Register::ValuePtr destLabel,
                                            Register::ValuePtr condition,
                                            std::string        comment = "");

        /**
         * @brief Generates a branch on nonzero instruction.
         *
         * Calls branchConditional with zero=false.
         *
         * @param destLabel A register value with regType() == Register::Type::Label
         * @param condition The register to conditionally branch on.
         * @param comment="" An optional comment to include in the generated instruction
         * @return Generator<Instruction>
         * @sa branchConditional
         */
        Generator<Instruction> branchIfNonZero(Register::ValuePtr destLabel,
                                               Register::ValuePtr condition,
                                               std::string        comment = "");

    private:
        std::weak_ptr<Context> m_context;
    };
}

#include "BranchGenerator_impl.hpp"
