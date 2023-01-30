#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/MultiplyAdd.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(MultiplyAddGenerator);

    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::MultiplyAdd>>
        GetGenerator<Expression::MultiplyAdd>(Register::ValuePtr dst,
                                              Register::ValuePtr a,
                                              Register::ValuePtr x,
                                              Register::ValuePtr y)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::MultiplyAdd>>(
            getContextFromValues(dst, a, x, y), Register::Type::Vector, DataType::None);
    }

    Generator<Instruction> MultiplyAddGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                          std::shared_ptr<Register::Value> a,
                                                          std::shared_ptr<Register::Value> x,
                                                          std::shared_ptr<Register::Value> y)

    {
        AssertFatal(a);
        AssertFatal(x);
        AssertFatal(y);

        auto dtype = a->variableType().dataType;

        auto isVector = [](auto x) { return x->regType() == Register::Type::Vector; };
        auto isType   = [&dtype](auto x) { return x->variableType() == dtype; };

        bool uniformVector = isVector(a) && isVector(x) && isVector(y);
        bool uniformType   = isType(a) && isType(x) && isType(y);

        if(uniformVector && uniformType)
        {
            switch(dtype)
            {
            case DataType::Float:
                co_yield_(Instruction("v_fma_f32", {dest}, {a, x, y}, {}, ""));
                co_return;
            case DataType::Double:
                co_yield_(Instruction("v_fma_f64", {dest}, {a, x, y}, {}, ""));
                co_return;
            case DataType::Halfx2:
                co_yield_(Instruction("v_pk_fma_f16", {dest}, {a, x, y}, {}, ""));
                co_return;
            default:
                break;
            }
        }

        co_yield generateOp<Expression::Multiply>(dest, a, x);
        co_yield generateOp<Expression::Add>(dest, dest, y);
    }
}
