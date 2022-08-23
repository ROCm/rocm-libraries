
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
    };

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

    // --------------------------------------------------
    // Get Functions
    // These functions are used to pick the proper Generator class for the provided
    // Expression and arguments.

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
        co_yield GetGenerator<Operation>(dst, lhs, rhs)->generate(dst, lhs, rhs);
    }

    // --------------------------------------------------
    // Helper functions

    // Return the expected datatype from the arguments to an operation.
    DataType
        promoteDataType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

    // Return the expected register type from the arguments to an operation.
    Register::Type
        promoteRegisterType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

    // Return the context from a list of register values.
    inline std::shared_ptr<Context> getContextFromValues(Register::ValuePtr r)
    {
        return r->context();
    }

    template <typename... Args>
    inline std::shared_ptr<Context> getContextFromValues(Register::ValuePtr arg, Args... args)
    {
        auto result = arg->context();
        if(result)
            return result;
        else
            return getContextFromValues(args...);
    }
}

#include "Add.hpp"
#include "BitwiseAnd.hpp"
#include "BitwiseOr.hpp"
#include "BitwiseXor.hpp"
#include "Divide.hpp"
#include "Modulo.hpp"
#include "Multiply.hpp"
#include "MultiplyHigh.hpp"
#include "ShiftL.hpp"
#include "ShiftR.hpp"
#include "SignedShiftR.hpp"
#include "Subtract.hpp"
