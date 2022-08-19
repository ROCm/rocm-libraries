/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#include <rocRoller/CodeGen/Arithmetic/Float.hpp>

#include "AssemblyKernel.hpp"
#include "Utilities/Error.hpp"
#include "Utilities/Utils.hpp"

#include <string>

namespace rocRoller
{
    RegisterComponent(Arithmetic_Vector_Float);

    Arithmetic_Vector_Float::Arithmetic_Vector_Float(std::shared_ptr<Context> context)
        : DirectBase(context)
    {
    }

    std::string Arithmetic_Vector_Float::name()
    {
        return Name;
    }

    bool Arithmetic_Vector_Float::Match(Argument const& arg)
    {
        std::shared_ptr<Context> ctx;
        Register::Type           regType;
        VariableType             variableType;

        std::tie(ctx, regType, variableType) = arg;

        AssertFatal(ctx != nullptr);

        return regType == Register::Type::Vector && variableType == DataType::Float;
    }

    std::shared_ptr<Arithmetic_Vector_Float::Base>
        Arithmetic_Vector_Float::Build(Argument const& arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<Arithmetic_Vector_Float>(std::get<0>(arg));
    }

    Generator<Instruction> Arithmetic_Vector_Float::add(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_add_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Float::sub(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_sub_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Float::gt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_gt_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Float::ge(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Float::lt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Float::le(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_le_f32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Float::eq(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_f32", {dest}, {lhs, rhs}, {}, ""));
    }
}
