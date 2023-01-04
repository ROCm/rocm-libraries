
#pragma once

#include "CodeGen/Instruction.hpp"
#include "Utilities/Error.hpp"
#include "Utilities/Generator.hpp"

namespace rocRoller
{
    // Base Arithmetic Generator class. All Arithmetic generators should be derived
    // from this class.
    class ArithmeticGenerator
    {
    public:
        ArithmeticGenerator(std::shared_ptr<Context> context)
            : m_context(context)
        {
        }

    protected:
        std::shared_ptr<Context> m_context;

        // Move a value into a single VGPR
        Generator<Instruction> moveToVGPR(Register::ValuePtr& val);

        // Split a single register value into two registers each containing one word
        Generator<Instruction> get2DwordsScalar(Register::ValuePtr& lsd,
                                                Register::ValuePtr& msd,
                                                Register::ValuePtr  input);
        Generator<Instruction> get2DwordsVector(Register::ValuePtr& lsd,
                                                Register::ValuePtr& msd,
                                                Register::ValuePtr  input);

        // Generate comments describing an operation that is being generated.
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
        UnaryArithmeticGenerator(std::shared_ptr<Context> context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction> generate(Register::ValuePtr dst, Register::ValuePtr arg)
        {
            Throw<FatalError>(concatenate("Generate function not implemented for ", Name));
        }

        using Argument = std::tuple<std::shared_ptr<Context>, Register::Type, DataType>;
        using Base     = UnaryArithmeticGenerator<Operation>;
        static const std::string Name;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CUnary Operation>
    const std::string UnaryArithmeticGenerator<Operation>::Name
        = concatenate(Expression::ExpressionInfo<Operation>::name(), "Generator");

    // Binary Arithmetic Generator. Most binary generators should be derived from
    // this class.
    template <Expression::CBinary Operation>
    class BinaryArithmeticGenerator : public ArithmeticGenerator
    {
    public:
        BinaryArithmeticGenerator(std::shared_ptr<Context> context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction>
            generate(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
        {
            Throw<FatalError>(concatenate("Generate function not implemented for ", Name));
        }

        using Argument = std::tuple<std::shared_ptr<Context>, Register::Type, DataType>;
        using Base     = BinaryArithmeticGenerator<Operation>;
        static const std::string Name;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CBinary Operation>
    const std::string BinaryArithmeticGenerator<Operation>::Name
        = concatenate(Expression::ExpressionInfo<Operation>::name(), "Generator");

    // Ternary Arithmetic Generator. Most ternary generators should be derived from
    // this class.
    template <Expression::CTernary Operation>
    class TernaryArithmeticGenerator : public ArithmeticGenerator
    {
    public:
        TernaryArithmeticGenerator(std::shared_ptr<Context> context)
            : ArithmeticGenerator(context)
        {
        }

        virtual Generator<Instruction> generate(Register::ValuePtr dst,
                                                Register::ValuePtr arg1,
                                                Register::ValuePtr arg2,
                                                Register::ValuePtr arg3)
        {
            Throw<FatalError>(concatenate("Generate function not implemented for ", Name));
        }

        using Argument = std::tuple<std::shared_ptr<Context>, Register::Type, DataType>;
        using Base     = TernaryArithmeticGenerator<Operation>;
        static const std::string Name;

        std::string name() const override
        {
            return Expression::ExpressionInfo<Operation>::name();
        }
    };

    template <Expression::CTernary Operation>
    const std::string TernaryArithmeticGenerator<Operation>::Name
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
    inline std::shared_ptr<Context> getContextFromValues(Register::ValuePtr r)
    {
        AssertFatal(r != nullptr, "No context");
        return r->context();
    }

    template <typename... Args>
    inline std::shared_ptr<Context> getContextFromValues(Register::ValuePtr arg, Args... args)
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
#include "Modulo.hpp"
#include "Multiply.hpp"
#include "MultiplyHigh.hpp"
#include "Negate.hpp"
#include "ShiftL.hpp"
#include "ShiftLAdd.hpp"
#include "ShiftR.hpp"
#include "SignedShiftR.hpp"
#include "Subtract.hpp"
