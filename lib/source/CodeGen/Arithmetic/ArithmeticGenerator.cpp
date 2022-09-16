#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>

namespace rocRoller
{
    // Move a value to a single VGPR register.
    Generator<Instruction> ArithmeticGenerator::moveToVGPR(Register::ValuePtr& val)
    {
        Register::ValuePtr tmp = val;
        val                    = Register::Value::Placeholder(
            m_context, Register::Type::Vector, tmp->variableType(), 1);

        co_yield m_context->copier()->copy(val, tmp, "");
    }

    // ----------------------------------------------
    // get2Dwords
    // Represent a single Register::Value as two Register::Values each the size of a single DWord

    void get2LiteralDwords(Register::ValuePtr& lsd,
                           Register::ValuePtr& msd,
                           Register::ValuePtr  input)
    {
        assert(input->regType() == Register::Type::Literal);
        int64_t value = std::visit(
            [](auto v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr(std::is_pointer_v<T>)
                {
                    return reinterpret_cast<int64_t>(v);
                }
                else
                {
                    return static_cast<int64_t>(v);
                }
            },
            input->getLiteralValue());

        lsd = Register::Value::Literal(static_cast<uint32_t>(value));
        msd = Register::Value::Literal(static_cast<uint32_t>(value >> 32));
    }

    Generator<Instruction> ArithmeticGenerator::get2DwordsScalar(Register::ValuePtr& lsd,
                                                                 Register::ValuePtr& msd,
                                                                 Register::ValuePtr  input)
    {
        if(input->regType() == Register::Type::Literal)
        {
            get2LiteralDwords(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Scalar)
        {
            auto varType = input->variableType();

            if(varType == DataType::Int32)
            {
                lsd = input;

                msd = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::Raw32, 1);

                auto l31 = Register::Value::Literal(31);

                co_yield generateOp<Expression::SignedShiftR>(msd, input, l31);

                co_return;
            }

            if(varType == DataType::UInt32
               || (varType == DataType::Raw32 && input->valueCount() == 1))
            {
                lsd = input->subset({0});
                msd = Register::Value::Literal(0);
                co_return;
            }

            if(varType == DataType::Raw32 && input->valueCount() >= 2)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }

            if(varType.pointerType == PointerType::PointerGlobal || varType == DataType::Int64
               || varType == DataType::UInt64)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }
        }

        Throw<FatalError>(concatenate("Conversion not implemented for register type scalar",
                                      input->regType(),
                                      "/",
                                      input->variableType()));
    }

    Generator<Instruction> ArithmeticGenerator::get2DwordsVector(Register::ValuePtr& lsd,
                                                                 Register::ValuePtr& msd,
                                                                 Register::ValuePtr  input)
    {
        if(input->regType() == Register::Type::Literal)
        {
            get2LiteralDwords(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Scalar)
        {
            co_yield get2DwordsScalar(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Vector)
        {
            auto varType = input->variableType();

            if(varType == DataType::Int32)
            {
                lsd = input;

                msd = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Raw32, 1);
                auto l31 = Register::Value::Literal(31);

                co_yield generateOp<Expression::SignedShiftR>(msd, input, l31);

                co_return;
            }

            if(varType == DataType::UInt32)
            {
                lsd = input->subset({0});
                msd = Register::Value::Literal(0);
                co_return;
            }

            if(varType.pointerType == PointerType::PointerGlobal || varType == DataType::Int64
               || varType == DataType::UInt64)
            {
                lsd = input->subset({0});
                msd = input->subset({1});
                co_return;
            }
        }

        Throw<FatalError>(concatenate("Conversion not implemented for register type dfgdgfd",
                                      input->regType(),
                                      "/",
                                      input->variableType()));
    }

    Generator<Instruction> ArithmeticGenerator::describeOpArgs(std::string const& argName0,
                                                               Register::ValuePtr arg0,
                                                               std::string const& argName1,
                                                               Register::ValuePtr arg1,
                                                               std::string const& argName2,
                                                               Register::ValuePtr arg2)
    {
        auto        opDesc = name() + ": ";
        std::string indent(opDesc.size(), ' ');

        co_yield Instruction::Comment(
            concatenate(opDesc, argName0, " (", arg0->description(), ") = "));
        co_yield Instruction::Comment(
            concatenate(indent, argName1, " (", arg1->description(), ") "));

        if(arg2)
            co_yield Instruction::Comment(
                concatenate(indent, argName2, " (", arg2->description(), ")"));
    }

    // -----------------------------
    // Helper Functions

    DataType getArithDataType(Register::ValuePtr reg)
    {
        AssertFatal(reg != nullptr, "Null argument");

        auto variableType = reg->variableType();

        if(variableType.isPointer()
           || (variableType == DataType::Raw32 && reg->registerCount() == 2))
        {
            return DataType::UInt64;
        }

        return variableType.dataType;
    }

    DataType promoteDataType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr, "Null argument");
        AssertFatal(rhs != nullptr, "Null argument");

        auto lhsVarType = lhs->variableType() == DataType::Raw32 && lhs->registerCount() == 2
                              ? DataType::UInt64
                              : lhs->variableType();
        auto rhsVarType = rhs->variableType() == DataType::Raw32 && rhs->registerCount() == 2
                              ? DataType::UInt64
                              : rhs->variableType();
        auto varType    = VariableType::Promote(lhsVarType, rhsVarType);

        if(dst)
        {
            auto dstVarType = dst->variableType();
            if(varType != dstVarType && dstVarType.dataType != DataType::Raw32)
            {
                auto const& varTypeInfo    = DataTypeInfo::Get(varType);
                auto const& dstVarTypeInfo = DataTypeInfo::Get(dstVarType);

                AssertFatal(varTypeInfo.elementSize <= dstVarTypeInfo.elementSize
                                && varTypeInfo.isIntegral == dstVarTypeInfo.isIntegral,
                            ShowValue(varType),
                            ShowValue(dstVarType));

                varType = dstVarType;
            }
        }

        return varType.isPointer() ? DataType::UInt64 : varType.dataType;
    }

    Register::Type
        promoteRegisterType(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        if(dst)
            return dst->regType();

        AssertFatal(lhs != nullptr, "Null argument");
        AssertFatal(rhs != nullptr, "Null argument");

        auto regType = Register::PromoteType(lhs->regType(), rhs->regType());

        return regType;
    }
}
