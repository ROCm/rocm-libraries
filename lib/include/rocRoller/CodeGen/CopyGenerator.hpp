/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "Instruction.hpp"

#include "../Context_fwd.hpp"
#include "../Utilities/Component.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    /**
     * Co-yields `mov` instructions automatically from a `src` to `dest`
     * if the copy is valid.
     */
    class CopyGenerator
    {
    public:
        CopyGenerator(std::shared_ptr<Context>);

        ~CopyGenerator();

        /**
         * Ensure the register is of type `t`.
         */
        Generator<Instruction>
            ensureType(Register::ValuePtr& dest, Register::ValuePtr src, Register::Type t) const;

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

    private:
        std::weak_ptr<Context> m_context;
    };
}

#include "CopyGenerator_impl.hpp"
