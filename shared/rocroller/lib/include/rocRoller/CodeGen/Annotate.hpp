/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
        AddComment(std::string comment)
            : m_comment(std::move(comment))
        {
        }

        Instruction operator()(Instruction inst)
        {
            inst.addComment(m_comment);
            return inst;
        }

    private:
        std::string m_comment;
    };

    /**
     * A functor that adds the specified control op to a given instruction.
     * Really intended for use only from LowerFromKernelGraph, so that every
     * instruction is annotated with the control op that it came from.
     */
    class AddControlOp
    {
    public:
        AddControlOp(int op)
            : m_op(op)
        {
        }

        Instruction operator()(Instruction inst)
        {
            inst.addControlOp(m_op);
            return inst;
        }

    private:
        int m_op;
    };
}
