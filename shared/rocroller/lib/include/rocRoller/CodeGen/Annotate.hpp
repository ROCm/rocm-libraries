// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    /**
     * A functor that adds the specified comment to a given instruction.
     * Intended for use with Generator<Instruction>::map(), e.g.:
     *
     * co_yield generate(destReg, expr, context).map(AddComment("foo"));
     *
     * This will add the comment "foo" to each instruction that's part of
     * generating `expr`.
     */
    class AddComment
    {
    public:
        AddComment(std::string comment, bool skipCommentOnly = false);

        Instruction operator()(Instruction inst);

    private:
        std::string m_comment;
        bool        m_skipCommentOnly;
    };

    /**
     * A functor that adds the specified control op to a given instruction.
     * Really intended for use only from LowerFromKernelGraph, so that every
     * instruction is annotated with the control op that it came from.
     */
    class AddControlOp
    {
    public:
        AddControlOp(int op);

        Instruction operator()(Instruction inst);

    private:
        int m_op;
    };

    enum class SourceLocationPart
    {
        Function = 0,
        File,
        Line,
        Column,
        Count
    };

    inline std::string toString(SourceLocationPart part);

    class AddLocation
    {
    public:
        AddLocation(EnumBitset<SourceLocationPart> parts,
                    std::source_location           loc = std::source_location::current());

        AddLocation(std::source_location loc = std::source_location::current());

        /**
         * Returns a new object which will only annotate the first non-comment
         * instruction that it encounters.
         */
        AddLocation onlyFirst();

        std::string comment() const;

        Instruction operator()(Instruction inst);

    private:
        EnumBitset<SourceLocationPart> m_parts;
        std::source_location           m_location;

        bool m_onlyFirst = false;
        bool m_expired   = false;
    };
}

#include <rocRoller/CodeGen/Annotate_impl.hpp>
