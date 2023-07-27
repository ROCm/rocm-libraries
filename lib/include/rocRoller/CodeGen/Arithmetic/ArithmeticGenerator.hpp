
#pragma once

#include "CodeGen/Instruction.hpp"
#include "Utilities/Error.hpp"
#include "Utilities/Generator.hpp"

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

        virtual Generator<Instruction> generate(Register::ValuePtr dst, Register::ValuePtr arg) = 0;

        /**
         * @brief Returns true if arg will be equivalent to the result.
         *
         * @param arg
         */
        virtual bool isIdentity(Register::ValuePtr arg) const
        {
            return false;
        }

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

        virtual Generator<Instruction>
            generate(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs) = 0;

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
                                                Register::ValuePtr arg3)
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
                                                Register::ValuePtr arg3)
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
    std::shared_ptr<UnaryArithmeticGenerator<Operation>> GetGenerator(Register::ValuePtr dst,
                                                                      Register::ValuePtr arg)
    {
        return nullptr;
    }

    template <Expression::CUnary Operation>
    Generator<Instruction> generateOp(Register::ValuePtr dst, Register::ValuePtr arg)
    {
        auto gen = GetGenerator<Operation>(dst, arg);
        AssertFatal(gen != nullptr, "No generator");
        co_yield gen->generate(dst, arg);
    }

    template <Expression::CBinary Operation>
    std::shared_ptr<BinaryArithmeticGenerator<Operation>>
        GetGenerator(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        return nullptr;
    }

    template <Expression::CBinary Operation>
    Generator<Instruction>
        generateOp(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        auto gen = GetGenerator<Operation>(dst, lhs, rhs);
        AssertFatal(gen != nullptr, "No generator");
        co_yield gen->generate(dst, lhs, rhs);
    }

    template <Expression::CTernary Operation>
    std::shared_ptr<TernaryArithmeticGenerator<Operation>> GetGenerator(Register::ValuePtr dst,
                                                                        Register::ValuePtr arg1,
                                                                        Register::ValuePtr arg2,
                                                                        Register::ValuePtr arg3)
    {
        return nullptr;
    }

    template <Expression::CTernary Operation>
    Generator<Instruction> generateOp(Register::ValuePtr dst,
                                      Register::ValuePtr arg1,
                                      Register::ValuePtr arg2,
                                      Register::ValuePtr arg3)
    {
        auto gen = GetGenerator<Operation>(dst, arg1, arg2, arg3);
        AssertFatal(gen != nullptr, "No generator");
        co_yield gen->generate(dst, arg1, arg2, arg3);
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
    DataType getArithDataType(Register::ValuePtr reg);

    // Return the context from a list of register values.
    inline ContextPtr getContextFromValues(Register::ValuePtr r)
    {
        AssertFatal(r != nullptr, "No context");
        return r->context();
    }

    template <typename... Args>
    inline ContextPtr getContextFromValues(Register::ValuePtr arg, Args... args)
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

#include "Add.hpp"
#include "AddShiftL.hpp"
#include "ArithmeticShiftR.hpp"
#include "BitwiseAnd.hpp"
#include "BitwiseOr.hpp"
#include "BitwiseXor.hpp"
#include "Convert.hpp"
#include "Divide.hpp"
#include "Equal.hpp"
#include "GreaterThan.hpp"
#include "GreaterThanEqual.hpp"
#include "LessThan.hpp"
#include "LessThanEqual.hpp"
#include "LogicalAnd.hpp"
#include "LogicalShiftR.hpp"
#include "Modulo.hpp"
#include "Multiply.hpp"
#include "MultiplyAdd.hpp"
#include "MultiplyHigh.hpp"
#include "Negate.hpp"
#include "ShiftL.hpp"
#include "ShiftLAdd.hpp"
#include "Subtract.hpp"
