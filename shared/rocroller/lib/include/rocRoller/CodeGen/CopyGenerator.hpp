// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    /**
     * Co-yields `mov` instructions automatically from a `src` to `dest`
     * if the copy is valid.
     */
    class CopyGenerator
    {
    public:
        CopyGenerator(ContextPtr);

        ~CopyGenerator();

        /**
         * Ensure the register is of type `t`.
         */
        Generator<Instruction>
            ensureType(Register::ValuePtr& dest, Register::ValuePtr src, Register::Type t) const;

        /**
         * Ensures `dest` is one of the `types`. Copies if nessessary.
         */
        Generator<Instruction> ensureType(Register::ValuePtr&        dest,
                                          Register::ValuePtr         src,
                                          EnumBitset<Register::Type> types) const;

        /**
         *  Prepare the LHS and RHS operands of a commutative instruction.
         *  Currently we only handle the case that if RHS is a literal,
         *  swap it with LHS or move it to a new VGPR.
         *
         *  TODO: enhance this to use GPUInstructionInfo to figure out the allowed operand types.
         *
         */
        Generator<Instruction> ensureTypeCommutative(EnumBitset<Register::Type> lhsTypes,
                                                     Register::ValuePtr&        lhs,
                                                     EnumBitset<Register::Type> rhsTypes,
                                                     Register::ValuePtr&        rhs) const;

        /**
         * Copy `src` registers to `dest` registers.
         */
        Generator<Instruction>
            copy(Register::ValuePtr dest, Register::ValuePtr src, std::string comment = "") const;

        /**
         * Conditionally copy `src` registers to `dest` registers.
         *
         * Only works for Literal->SGPR and SGPR->SGPR
         * Condition is implicitly in SCC.
         */
        Generator<Instruction> conditionalCopy(Register::ValuePtr dest,
                                               Register::ValuePtr src,
                                               std::string        comment = "") const;

        /**
         * Fill `dest` with `src` where `src` is a literal.
         */
        Generator<Instruction>
            fill(Register::ValuePtr dest, Register::ValuePtr src, std::string comment = "") const;

        /**
         * @brief Pack two 16bit values into a single 32bit register
         *
         * @param dest The register that will hold the value
         * @param loVal The register containing the value that will be stored in the low part
         * @param hiVal The register containing the value that will be stored in the high part
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @return Generator<Instruction>
         */
        Generator<Instruction> packHalf(Register::ValuePtr dest,
                                        Register::ValuePtr loVal,
                                        Register::ValuePtr hiVal,
                                        std::string        comment = "") const;

        /**
         * @brief Pack multiple values into a single 32bit register
         *
         * @param dest The register that will hold the value
         * @param values A vector containing the registers that will be stored. The lowerer indexed elements will be stored in the LSBs.
         * @param comment Comment that will be generated along with the instructions. (Default = "")
         * @return Generator<Instruction>
         */
        Generator<Instruction> pack(Register::ValuePtr              dest,
                                    std::vector<Register::ValuePtr> values,
                                    std::string                     comment = "") const;

    private:
        std::weak_ptr<Context> m_context;
    };
}

#include <rocRoller/CodeGen/CopyGenerator_impl.hpp>
