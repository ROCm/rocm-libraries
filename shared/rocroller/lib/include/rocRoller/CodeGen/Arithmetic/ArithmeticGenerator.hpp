/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    /// Base Arithmetic Generator class. All Arithmetic generators should be derived
    /// from this class.
    class ArithmeticGenerator
    {
    public:
        ArithmeticGenerator(ContextPtr context)
            : m_context(context)
        {
        }

    protected:
        ContextPtr m_context;

        /// Move a value into a single VGPR
        Generator<Instruction> moveToVGPR(Register::ValuePtr& val);

        /// Copy the sign bit from `src` into every bit of `dst`.
        Generator<Instruction> signExtendDWord(Register::ValuePtr dst, Register::ValuePtr src);

        /// Split a single register value into two registers each containing one word
        Generator<Instruction> get2DwordsScalar(Register::ValuePtr& lsd,
                                                Register::ValuePtr& msd,
                                                Register::ValuePtr  input);
        Generator<Instruction> get2DwordsVector(Register::ValuePtr& lsd,
                                                Register::ValuePtr& msd,
                                                Register::ValuePtr  input);

        /// Generate comments describing an operation that is being generated.
        Generator<Instruction> describeOpArgs(std::string const& argName0,
                                              Register::ValuePtr arg0,
                                              std::string const& argName1,
                                              Register::ValuePtr arg1,
                                              std::string const& argName2,
                                              Register::ValuePtr arg2);

        virtual std::string name() const = 0;

        /**
         * @brief For certain vector binary instructions that can not have a literal as the second source, swap.
         * Side effect: Potentially swaps the lhs and rhs registers
         *
         * TODO: Swap if RHS is anything but a VGPR
         *
         * @param lhs First source register (src0)
         * @param rhs Second source register (src1)
         * @return Generator<Instruction> May yield a move to VGPR instruction if needed
         */
        Generator<Instruction> swapIfRHSLiteral(Register::ValuePtr& lhs, Register::ValuePtr& rhs);

        /**
         * @brief Use VALU to perform a scalar comparison.
         *
         * Some vectors comparison instructions (like v_cmp_le_i64)
         * don't have scalar equivalents.  In this case, the RHS is
         * copied to VGPRs and the comparison is done with the VALU.
         *
         * @param lhs LHS of comparison (stored in SGPR)
         * @param rhs RHS of comparison (stored in SGPR)
         * @param dst Destination, can be null (in which case result is in SCC).
         */
        Generator<Instruction> scalarCompareThroughVALU(std::string const  instruction,
                                                        Register::ValuePtr dst,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr rhs);
    };

    // Unary Arithmetic Generator. Most unary generators should be derived from
    // this class.
    template <Expression::CUnary Operation>
    class UnaryArithmeticGenerator : public ArithmeticGenerator
    {
    public:
        UnaryArithmeticGenerator(ContextPtr context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction>
            generate(Register::ValuePtr dst, Register::ValuePtr arg, Operation const& expr) = 0;

        using Argument = std::tuple<ContextPtr, Register::Type, DataType>;
        using Base     = UnaryArithmeticGenerator<Operation>;
        static const std::string Basename;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CUnary Operation>
    const std::string UnaryArithmeticGenerator<Operation>::Basename
        = concatenate(Expression::ExpressionInfo<Operation>::name(), "Generator");

    // Binary Arithmetic Generator. Most binary generators should be derived from
    // this class.
    template <Expression::CBinary Operation>
    class BinaryArithmeticGenerator : public ArithmeticGenerator
    {
    public:
        BinaryArithmeticGenerator(ContextPtr context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction> generate(Register::ValuePtr dst,
                                                Register::ValuePtr lhs,
                                                Register::ValuePtr rhs,
                                                Operation const&   expr)
            = 0;

        using Argument = std::tuple<ContextPtr, Register::Type, DataType>;
        using Base     = BinaryArithmeticGenerator<Operation>;
        static const std::string Basename;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CBinary Operation>
    const std::string BinaryArithmeticGenerator<Operation>::Basename
        = concatenate(Expression::ExpressionInfo<Operation>::name(), "Generator");

    // Ternary Arithmetic Generator. Most ternary generators should be derived from
    // this class.
    template <Expression::CTernary Operation>
    class TernaryArithmeticGenerator : public ArithmeticGenerator
    {
    public:
        TernaryArithmeticGenerator(ContextPtr context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction> generate(Register::ValuePtr dst,
                                                Register::ValuePtr arg1,
                                                Register::ValuePtr arg2,
                                                Register::ValuePtr arg3,
                                                Operation const&   expr)
            = 0;

        using Argument = std::tuple<ContextPtr, Register::Type, DataType>;
        using Base     = TernaryArithmeticGenerator<Operation>;
        static const std::string Basename;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CTernary Operation>
    const std::string TernaryArithmeticGenerator<Operation>::Basename
        = concatenate(Expression::ExpressionInfo<Operation>::name(), "Generator");

    // TernaryMixed Arithmetic Generator. Only Ternary generators that can support mixed
    // airthmetic should be derived from this class.
    template <Expression::CTernaryMixed Operation>
    class TernaryMixedArithmeticGenerator : public ArithmeticGenerator
    {
    public:
        TernaryMixedArithmeticGenerator(ContextPtr context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction> generate(Register::ValuePtr dst,
                                                Register::ValuePtr arg1,
                                                Register::ValuePtr arg2,
                                                Register::ValuePtr arg3,
                                                Operation const&   expr)
            = 0;

        using Argument = std::tuple<ContextPtr, Register::Type, DataType>;
        using Base     = TernaryMixedArithmeticGenerator<Operation>;
        static const std::string Basename;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CTernaryMixed Operation>
    const std::string TernaryMixedArithmeticGenerator<Operation>::Basename
        = concatenate(Expression::ExpressionInfo<Operation>::name(), "Generator");

    // --------------------------------------------------
    // Get Functions
    // These functions are used to pick the proper Generator class for the provided
    // Expression and arguments.

    template <Expression::CUnary Operation>
    std::shared_ptr<UnaryArithmeticGenerator<Operation>>
        GetGenerator(Register::ValuePtr dst, Register::ValuePtr arg, Operation const& expr)
    {
        return nullptr;
    }

    template <Expression::CUnary Operation>
    Generator<Instruction> generateOp(Register::ValuePtr dst,
                                      Register::ValuePtr arg,
                                      Operation const&   expr = Operation{})
    {
        auto gen = GetGenerator<Operation>(dst, arg, expr);
        AssertFatal(gen != nullptr, "No generator");
        co_yield gen->generate(dst, arg, expr);
    }

    template <Expression::CBinary Operation>
    std::shared_ptr<BinaryArithmeticGenerator<Operation>> GetGenerator(Register::ValuePtr dst,
                                                                       Register::ValuePtr lhs,
                                                                       Register::ValuePtr rhs,
                                                                       Operation const&   expr)
    {
        return nullptr;
    }

    template <Expression::CBinary Operation>
    Generator<Instruction> generateOp(Register::ValuePtr const dst,
                                      Register::ValuePtr       lhs,
                                      Register::ValuePtr       rhs,
                                      Operation const&         expr = Operation{})
    {
        auto gen = GetGenerator<Operation>(dst, lhs, rhs, expr);
        AssertFatal(gen != nullptr, "No generator");
        co_yield gen->generate(dst, lhs, rhs, expr);
    }

    template <Expression::CTernary Operation>
    std::shared_ptr<TernaryArithmeticGenerator<Operation>> GetGenerator(Register::ValuePtr dst,
                                                                        Register::ValuePtr arg1,
                                                                        Register::ValuePtr arg2,
                                                                        Register::ValuePtr arg3,
                                                                        Operation const&   expr)
    {
        return nullptr;
    }

    template <Expression::CTernary Operation>
    Generator<Instruction> generateOp(Register::ValuePtr dst,
                                      Register::ValuePtr arg1,
                                      Register::ValuePtr arg2,
                                      Register::ValuePtr arg3,
                                      Operation const&   expr = Operation{})
    {
        auto gen = GetGenerator<Operation>(dst, arg1, arg2, arg3, expr);
        AssertFatal(gen != nullptr, "No generator");
        co_yield gen->generate(dst, arg1, arg2, arg3, expr);
    }
    // --------------------------------------------------
    // Helper functions

    // Return the expected datatype from the arguments to an operation.
    DataType
        promoteDataType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

    // Return the expected register type from the arguments to an operation.
    Register::Type
        promoteRegisterType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

    // Return the data of a register that will be used for arithmetic calculations.
    // If the register contains a pointer, the DataType that is returned is UInt64.
    /**
     * @brief Return the data of a register that will be used for arithmetic calculations.
     *
     * If the register contains a pointer, the DataType that is returned is UInt64.
     *
     * @param reg
     * @return DataType
     */
    DataType getArithDataType(Register::ValuePtr const reg);

    // Return the context from a list of register values.
    inline ContextPtr getContextFromValues(Register::ValuePtr const r)
    {
        AssertFatal(r != nullptr, "No context");
        return r->context();
    }

    template <typename... Args>
    inline ContextPtr getContextFromValues(Register::ValuePtr const arg, Args... args)
    {
        if(arg && arg->context())
        {
            return arg->context();
        }
        else
        {
            return getContextFromValues(args...);
        }
    }
}

#include <rocRoller/CodeGen/Arithmetic/Add.hpp>
#include <rocRoller/CodeGen/Arithmetic/AddShiftL.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticShiftR.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseAnd.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseNegate.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseOr.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseXor.hpp>
#include <rocRoller/CodeGen/Arithmetic/Conditional.hpp>
#include <rocRoller/CodeGen/Arithmetic/Convert.hpp>
#include <rocRoller/CodeGen/Arithmetic/Divide.hpp>
#include <rocRoller/CodeGen/Arithmetic/Equal.hpp>
#include <rocRoller/CodeGen/Arithmetic/Exponential2.hpp>
#include <rocRoller/CodeGen/Arithmetic/GreaterThan.hpp>
#include <rocRoller/CodeGen/Arithmetic/GreaterThanEqual.hpp>
#include <rocRoller/CodeGen/Arithmetic/LessThan.hpp>
#include <rocRoller/CodeGen/Arithmetic/LessThanEqual.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalAnd.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalNot.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalOr.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalShiftR.hpp>
#include <rocRoller/CodeGen/Arithmetic/Modulo.hpp>
#include <rocRoller/CodeGen/Arithmetic/Multiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyAdd.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyHigh.hpp>
#include <rocRoller/CodeGen/Arithmetic/Negate.hpp>
#include <rocRoller/CodeGen/Arithmetic/NotEqual.hpp>
#include <rocRoller/CodeGen/Arithmetic/RandomNumber.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftL.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftLAdd.hpp>
#include <rocRoller/CodeGen/Arithmetic/Subtract.hpp>
