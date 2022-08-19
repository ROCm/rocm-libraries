/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#include <rocRoller/CodeGen/Arithmetic/Double.hpp>

namespace rocRoller
{
    RegisterComponent(Arithmetic_Vector_Double);

    Arithmetic_Vector_Double::Arithmetic_Vector_Double(std::shared_ptr<Context> context)
        : DirectBase(context)
    {
    }

    std::string Arithmetic_Vector_Double::name()
    {
        return Name;
    }

    bool Arithmetic_Vector_Double::Match(Argument const& arg)
    {
        std::shared_ptr<Context> ctx;
        Register::Type           regType;
        VariableType             variableType;

        std::tie(ctx, regType, variableType) = arg;

        AssertFatal(ctx != nullptr);

        return regType == Register::Type::Vector && variableType == DataType::Double;
    }

    std::shared_ptr<Arithmetic_Vector_Double::Base>
        Arithmetic_Vector_Double::Build(Argument const& arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<Arithmetic_Vector_Double>(std::get<0>(arg));
    }

    Generator<Instruction> Arithmetic_Vector_Double::add(std::shared_ptr<Register::Value> dest,
                                                         std::shared_ptr<Register::Value> lhs,
                                                         std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_add_f64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Double::sub(std::shared_ptr<Register::Value> dest,
                                                         std::shared_ptr<Register::Value> lhs,
                                                         std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto nrhs = rhs->negate();

        co_yield_(Instruction("v_add_f64", {dest}, {lhs, nrhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Double::gt(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_gt_f64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Double::ge(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_f64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Double::lt(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_f64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Double::le(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_le_f64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Double::eq(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_f64", {dest}, {lhs, rhs}, {}, ""));
    }
}
