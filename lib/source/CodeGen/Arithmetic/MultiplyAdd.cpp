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
                                              Register::ValuePtr y,
                                              Expression::MultiplyAdd const&)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::MultiplyAdd>>(
            getContextFromValues(dst, a, x, y), Register::Type::Vector, DataType::None);
    }

    Generator<Instruction> MultiplyAddGenerator::generate(Register::ValuePtr dest,
                                                          Register::ValuePtr a,
                                                          Register::ValuePtr x,
                                                          Register::ValuePtr y,
                                                          Expression::MultiplyAdd const&)

    {
        AssertFatal(a);
        AssertFatal(x);
        AssertFatal(y);

        auto dTypeA = a->variableType().dataType;
        auto dTypeX = x->variableType().dataType;
        auto dTypeY = y->variableType().dataType;

        auto isVector = [](auto x) { return x->regType() == Register::Type::Vector; };
        auto isType   = [&dTypeA](auto x) { return x->variableType() == dTypeA; };

        bool uniformVector = isVector(a) && isVector(x) && isVector(y);
        bool uniformType   = isType(a) && isType(x) && isType(y);

        int valueCount = dest->valueCount();
        if(uniformVector && uniformType)
        {
            std::string operation;
            switch(dTypeA)
            {
            case DataType::Float:
                operation = "v_fma_f32";
                break;
            case DataType::Double:
                operation = "v_fma_f64";
                break;
            case DataType::Halfx2:
                operation = "v_pk_fma_f16";
                break;
            default:
                break;
            }

            if(!operation.empty())
            {
                for(size_t k = 0; k < valueCount; ++k)
                {
                    auto aVal = a->regType() == Register::Type::Literal || a->valueCount() == 1
                                    ? a
                                    : a->element({k});
                    auto xVal = x->regType() == Register::Type::Literal || x->valueCount() == 1
                                    ? x
                                    : x->element({k});
                    auto yVal = y->regType() == Register::Type::Literal || y->valueCount() == 1
                                    ? y
                                    : y->element({k});
                    auto dVal = dest->valueCount() == 1 ? dest : dest->element({k});
                    co_yield_(Instruction(operation, {dVal}, {aVal, xVal, yVal}, {}, ""));
                }
                co_return;
            }
        }

        bool mixed = false;
        int  aType = 3, xType = 3, yType = 3;
        if(dTypeA == DataType::Float || dTypeA == DataType::Halfx2)
        {
            aType = (dTypeA == DataType::Halfx2) ? 1 : 0;
        }
        if(dTypeX == DataType::Float || dTypeX == DataType::Halfx2)
        {
            xType = (dTypeX == DataType::Halfx2) ? 1 : 0;
        }
        if(dTypeY == DataType::Float || dTypeY == DataType::Halfx2)
        {
            yType = (dTypeY == DataType::Halfx2) ? 1 : 0;
        }

        //Only select combinations of FP32 or FP16
        if((aType + xType + yType) < 3 && !uniformType)
        {
            mixed = true;
        }
        //Dest x and y should be vectors, 'a' need not be a vector
        if(mixed && (isVector(x) && isVector(y)))
        {
            AssertFatal(valueCount > 1);
            std::string opSelLo     = concatenate("op_sel:[0,0,0] op_sel_hi:[",
                                              std::to_string(aType),
                                              ",",
                                              std::to_string(xType),
                                              ",",
                                              std::to_string(yType),
                                              "]");
            std::string opSelHi     = concatenate("op_sel:[",
                                              std::to_string(aType),
                                              ",",
                                              std::to_string(xType),
                                              ",",
                                              std::to_string(yType),
                                              "] op_sel_hi:[",
                                              std::to_string(aType),
                                              ",",
                                              std::to_string(xType),
                                              ",",
                                              std::to_string(yType),
                                              "]");
            int         hValueCount = valueCount / 2;
            for(size_t k = 0; k < hValueCount; ++k)
            {
                auto iter0  = 2 * k;
                auto itera0 = (aType == 1) ? k : iter0;
                auto iterx0 = (xType == 1) ? k : iter0;
                auto itery0 = (yType == 1) ? k : iter0;

                auto iter1  = 2 * k + 1;
                auto itera1 = (aType == 1) ? k : iter1;
                auto iterx1 = (xType == 1) ? k : iter1;
                auto itery1 = (yType == 1) ? k : iter1;

                auto xVal = x->regType() == Register::Type::Literal || x->valueCount() == 1
                                ? x
                                : x->element({iterx0});
                auto aVal = a->regType() == Register::Type::Literal || a->valueCount() == 1
                                ? a
                                : a->element({itera0});
                auto yVal = y->regType() == Register::Type::Literal || y->valueCount() == 1
                                ? y
                                : y->element({itery0});
                co_yield_(Instruction(
                    "v_fma_mix_f32", {dest->element({iter0})}, {aVal, xVal, yVal}, {opSelLo}, ""));
                xVal = x->regType() == Register::Type::Literal || x->valueCount() == 1
                           ? x
                           : x->element({iterx1});
                aVal = a->regType() == Register::Type::Literal || a->valueCount() == 1
                           ? a
                           : a->element({itera1});
                yVal = y->regType() == Register::Type::Literal || y->valueCount() == 1
                           ? y
                           : y->element({itery1});
                co_yield_(Instruction(
                    "v_fma_mix_f32", {dest->element({iter1})}, {aVal, xVal, yVal}, {opSelHi}, ""));
            }
            co_return;
        }

        for(size_t k = 0; k < valueCount; ++k)
        {
            auto aVal = a->regType() == Register::Type::Literal || a->valueCount() == 1
                            ? a
                            : a->element({k});
            auto xVal = x->regType() == Register::Type::Literal || x->valueCount() == 1
                            ? x
                            : x->element({k});
            auto yVal = y->regType() == Register::Type::Literal || y->valueCount() == 1
                            ? y
                            : y->element({k});
            auto dVal = dest->valueCount() == 1 ? dest : dest->element({k});
            co_yield generateOp<Expression::Multiply>(dVal, aVal, xVal);
            co_yield generateOp<Expression::Add>(dVal, dVal, yVal);
        }
    }
}
