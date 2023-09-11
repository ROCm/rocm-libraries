#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyHigh.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(MultiplyHighGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::MultiplyHigh>>
        GetGenerator<Expression::MultiplyHigh>(Register::ValuePtr dst,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::MultiplyHigh>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> MultiplyHighGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr lhs,
                                                           Register::ValuePtr rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto const& dataTypeInfoLhs  = DataTypeInfo::Get(lhs->variableType());
        auto const& dataTypeInfoRhs  = DataTypeInfo::Get(rhs->variableType());
        auto const& dataTypeInfoDest = DataTypeInfo::Get(dest->variableType());

        if(dataTypeInfoLhs.elementSize == 4 && dataTypeInfoRhs.elementSize == 4)
        {
            if(dest->regType() == Register::Type::Scalar)
            {
                if(dataTypeInfoDest.isSigned)
                {
                    co_yield_(Instruction("s_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
                }
                else
                {
                    co_yield_(Instruction("s_mul_hi_u32", {dest}, {lhs, rhs}, {}, ""));
                }
            }
            else if(dest->regType() == Register::Type::Vector)
            {
                if(dataTypeInfoDest.isSigned)
                {
                    co_yield_(Instruction("v_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
                }
                else
                {
                    co_yield_(Instruction("v_mul_hi_u32", {dest}, {lhs, rhs}, {}, ""));
                }
            }
        }
        else if(dataTypeInfoDest.elementSize == 8)
        {

            if(dest->regType() == Register::Type::Vector)
            {
                // libdivide algorithm:

                // uint32_t mask = 0xFFFFFFFF;
                // uint32_t x0 = (uint32_t)(x & mask);
                // uint32_t y0 = (uint32_t)(y & mask);
                // int32_t x1 = (int32_t)(x >> 32);
                // int32_t y1 = (int32_t)(y >> 32);
                // uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0);
                // int64_t t = x1 * (int64_t)y0 + x0y0_hi;
                // int64_t w1 = x0 * (int64_t)y1 + (t & mask);

                // return x1 * (int64_t)y1 + (t >> 32) + (w1 >> 32);

                Register::ValuePtr x0, x1, y0, y1;

                co_yield get2DwordsVector(x0, x1, lhs);
                co_yield get2DwordsVector(y0, y1, rhs);

                x0->setVariableType(DataType::UInt32);
                x1->setVariableType(DataType::Int32);
                y0->setVariableType(DataType::UInt32);
                y1->setVariableType(DataType::Int32);

                auto t = std::make_shared<Register::Value>(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                {
                    auto x0y0Hi = std::make_shared<Register::Value>(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    co_yield_(Instruction("v_mul_hi_u32",
                                          {x0y0Hi->subset({0})},
                                          {x0, y0},
                                          {},
                                          "mul_hi: x0*y0 high bits"));

                    co_yield m_context->copier()->copy(x0y0Hi->subset({1}),
                                                       Register::Value::Literal(0));

                    auto x1y0 = std::make_shared<Register::Value>(
                        m_context, Register::Type::Vector, DataType::Int64, 1);
                    co_yield generateOp<Expression::Multiply>(x1y0, x1, y0);

                    co_yield generateOp<Expression::Add>(t, x1y0, x0y0Hi);
                }

                auto w1 = std::make_shared<Register::Value>(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                {
                    auto x0y1 = std::make_shared<Register::Value>(
                        m_context, Register::Type::Vector, DataType::Int64, 1);
                    co_yield generateOp<Expression::Multiply>(x0y1, x0, y1);

                    auto zero = Register::Value::Literal(0);

                    auto carry = m_context->getVCC();
                    co_yield_(Instruction("v_add_co_u32",
                                          {w1->subset({0}), carry},
                                          {x0y1->subset({0}), t->subset({0})},
                                          {},
                                          "mul_hi: w1 = x0y1 + tLo low bits")
                                  .lock(Scheduling::Dependency::VCC));
                    co_yield_(Instruction("v_addc_co_u32",
                                          {w1->subset({1}), carry},
                                          {x0y1->subset({1}), zero, carry},
                                          {},
                                          "mul_hi: w1 = x0y1 + tLo high bits")
                                  .unlock());
                }

                co_yield generateOp<Expression::ArithmeticShiftR>(
                    t, t, Register::Value::Literal(32));
                co_yield generateOp<Expression::ArithmeticShiftR>(
                    w1, w1, Register::Value::Literal(32));

                auto x1y1 = std::make_shared<Register::Value>(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield generateOp<Expression::Multiply>(x1y1, x1, y1);

                co_yield generateOp<Expression::Add>(dest, x1y1, t);
                co_yield generateOp<Expression::Add>(dest, dest, w1);
            }
            else if(dest->regType() == Register::Type::Scalar)
            {
                // libdivide algorithm:

                // uint32_t mask = 0xFFFFFFFF;
                // uint32_t x0 = (uint32_t)(x & mask);
                // uint32_t y0 = (uint32_t)(y & mask);
                // int32_t x1 = (int32_t)(x >> 32);
                // int32_t y1 = (int32_t)(y >> 32);
                // uint32_t x0y0_hi = libdivide_mullhi_u32(x0, y0);
                // int64_t t = x1 * (int64_t)y0 + x0y0_hi;
                // int64_t w1 = x0 * (int64_t)y1 + (t & mask);

                // return x1 * (int64_t)y1 + (t >> 32) + (w1 >> 32);

                Register::ValuePtr x0, x1, y0, y1;

                co_yield get2DwordsScalar(x0, x1, lhs);
                co_yield get2DwordsScalar(y0, y1, rhs);

                x0->setVariableType(DataType::UInt32);
                x1->setVariableType(DataType::Int32);
                y0->setVariableType(DataType::UInt32);
                y1->setVariableType(DataType::Int32);

                auto t = std::make_shared<Register::Value>(
                    m_context, Register::Type::Scalar, DataType::Int64, 1);
                {
                    auto x0y0Hi = std::make_shared<Register::Value>(
                        m_context, Register::Type::Scalar, DataType::Int64, 1);
                    co_yield_(Instruction("s_mul_hi_u32",
                                          {x0y0Hi->subset({0})},
                                          {x0, y0},
                                          {},
                                          "mul_hi: x0*y0 high bits"));

                    co_yield m_context->copier()->copy(x0y0Hi->subset({1}),
                                                       Register::Value::Literal(0));

                    auto x1y0 = std::make_shared<Register::Value>(
                        m_context, Register::Type::Scalar, DataType::Int64, 1);
                    co_yield generateOp<Expression::Multiply>(x1y0, x1, y0);
                    co_yield generateOp<Expression::Add>(t, x1y0, x0y0Hi);
                }

                auto w1 = std::make_shared<Register::Value>(
                    m_context, Register::Type::Scalar, DataType::Int64, 1);
                {
                    auto x0y1 = std::make_shared<Register::Value>(
                        m_context, Register::Type::Scalar, DataType::Int64, 1);
                    co_yield generateOp<Expression::Multiply>(x0y1, x0, y1);

                    auto tExtended = Register::Value::Literal(0);

                    co_yield_(Instruction("s_add_u32",
                                          {w1->subset({0})},
                                          {x0y1->subset({0}), t->subset({0})},
                                          {},
                                          "mul_hi: w1 = x0y1 + tLo low bits")
                                  .lock(Scheduling::Dependency::SCC));
                    co_yield_(Instruction("s_addc_u32",
                                          {w1->subset({1})},
                                          {x0y1->subset({1}), tExtended},
                                          {},
                                          "mul_hi: w1 = x0y1 + tLo high bits")
                                  .unlock());
                }

                co_yield generateOp<Expression::ArithmeticShiftR>(
                    t, t, Register::Value::Literal(32));
                co_yield generateOp<Expression::ArithmeticShiftR>(
                    w1, w1, Register::Value::Literal(32));

                auto x1y1 = std::make_shared<Register::Value>(
                    m_context, Register::Type::Scalar, DataType::Int64, 1);
                co_yield generateOp<Expression::Multiply>(x1y1, x1, y1);

                co_yield generateOp<Expression::Add>(dest, x1y1, t);
                co_yield generateOp<Expression::Add>(dest, dest, w1);
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for multiply high operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
