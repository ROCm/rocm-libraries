/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>

namespace rocRoller
{
    RegisterComponent(Arithmetic_Vector_Int32);
    RegisterComponent(Arithmetic_Scalar_Int32);

    // -------------------------------------------------------------------
    // Vector Arithmetic
    // ...................................................................

    Arithmetic_Vector_Int32::Arithmetic_Vector_Int32(std::shared_ptr<Context> context)
        : DirectBase(context)
    {
    }

    std::string Arithmetic_Vector_Int32::name()
    {
        return Name;
    }

    bool Arithmetic_Vector_Int32::Match(Argument const& arg)
    {
        std::shared_ptr<Context> ctx;
        Register::Type           regType;
        VariableType             variableType;

        std::tie(ctx, regType, variableType) = arg;

        AssertFatal(ctx != nullptr);

        auto const& typeInfo = DataTypeInfo::Get(variableType);

        return regType == Register::Type::Vector && !typeInfo.isComplex && typeInfo.isIntegral
               && typeInfo.elementSize == sizeof(int32_t);
        // TODO: Once Arithmetic_Vector_UInt32 is implemented, add:
        // && typeInfo.isSigned
    }

    std::shared_ptr<Arithmetic_Vector_Int32::Base>
        Arithmetic_Vector_Int32::Build(Argument const& arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<Arithmetic_Vector_Int32>(std::get<0>(arg));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::add(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_add_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::sub(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_sub_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int32::shiftL(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_lshlrev_b32", {dest}, {shiftAmount, value}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int32::shiftR(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_lshrrev_b32", {dest}, {shiftAmount, value}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int32::addShiftL(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs,
                                           std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_add_lshl_u32", {dest}, {lhs, rhs, shiftAmount}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int32::shiftLAdd(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> shiftAmount,
                                           std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_lshl_add_u32", {dest}, {lhs, shiftAmount, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int32::bitwiseAnd(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_and_b32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int32::bitwiseXor(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_xor_b32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::gt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_gt_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::ge(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::lt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::le(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_le_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::eq(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    // -------------------------------------------------------------------
    // Scalar Arithmetic
    // ...................................................................

    Arithmetic_Scalar_Int32::Arithmetic_Scalar_Int32(std::shared_ptr<Context> context)
        : DirectBase(context)
    {
    }

    std::string Arithmetic_Scalar_Int32::name()
    {
        return Name;
    }

    bool Arithmetic_Scalar_Int32::Match(Argument const& arg)
    {
        std::shared_ptr<Context> ctx;
        Register::Type           regType;
        VariableType             variableType;

        std::tie(ctx, regType, variableType) = arg;

        AssertFatal(ctx != nullptr);

        auto const& typeInfo = DataTypeInfo::Get(variableType);

        return regType == Register::Type::Scalar && !typeInfo.isComplex && typeInfo.isIntegral
               && typeInfo.elementSize == sizeof(int32_t);
        // TODO: Once Arithmetic_Scalar_UInt32 is implemented, add:
        // && typeInfo.isSigned
    }

    std::shared_ptr<Arithmetic_Scalar_Int32::Base>
        Arithmetic_Scalar_Int32::Build(Argument const& arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<Arithmetic_Scalar_Int32>(std::get<0>(arg));
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::add(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_add_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int32::addShiftL(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs,
                                           std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);
        co_yield_(Instruction("s_add_u32", {dest}, {lhs, rhs}, {}, ""));
        co_yield_(Instruction("s_lshl_b32", {dest}, {dest, shiftAmount}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::sub(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_sub_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int32::shiftL(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("s_lshl_b32", {dest}, {value, shiftAmount}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int32::shiftR(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("s_lshr_b32", {dest}, {value, shiftAmount}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int32::shiftLAdd(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> shiftAmount,
                                           std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);
        if(shiftAmount->regType() == Register::Type::Literal)
        {
            auto shift = std::get<unsigned int>(shiftAmount->getLiteralValue());
            if(shift == 1u)
                co_yield_(Instruction("s_lshl1_add_u32", {dest}, {lhs, rhs}, {}, ""));
            else if(shift == 2u)
                co_yield_(Instruction("s_lshl2_add_u32", {dest}, {lhs, rhs}, {}, ""));
            else if(shift == 3u)
                co_yield_(Instruction("s_lshl3_add_u32", {dest}, {lhs, rhs}, {}, ""));
            else if(shift == 4u)
                co_yield_(Instruction("s_lshl4_add_u32", {dest}, {lhs, rhs}, {}, ""));
            else
            {
                co_yield_(Instruction("s_lshl_b32", {dest}, {lhs, shiftAmount}, {}, ""));
                co_yield_(Instruction("s_add_u32", {dest}, {dest, rhs}, {}, ""));
            }
        }
        else
        {
            co_yield_(Instruction("s_lshl_b32", {dest}, {lhs, shiftAmount}, {}, ""));
            co_yield_(Instruction("s_add_u32", {dest}, {dest, rhs}, {}, ""));
        }
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int32::bitwiseAnd(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_and_b32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int32::bitwiseXor(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_xor_b32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::gt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_gt_i32", {}, {lhs, rhs}, {}, ""));

        if(dest != nullptr && !dest->isSCC())
        {
            co_yield m_context->copier()->copy(dest, m_context->getSCC(), "");
        }
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::ge(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_ge_i32", {}, {lhs, rhs}, {}, ""));

        if(dest != nullptr && !dest->isSCC())
        {
            co_yield m_context->copier()->copy(dest, m_context->getSCC(), "");
        }
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::lt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_lt_i32", {}, {lhs, rhs}, {}, ""));

        if(dest != nullptr && !dest->isSCC())
        {
            co_yield m_context->copier()->copy(dest, m_context->getSCC(), "");
        }
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::le(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_le_i32", {}, {lhs, rhs}, {}, ""));

        if(dest != nullptr && !dest->isSCC())
        {
            co_yield m_context->copier()->copy(dest, m_context->getSCC(), "");
        }
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::eq(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_eq_i32", {}, {lhs, rhs}, {}, ""));

        if(dest != nullptr && !dest->isSCC())
        {
            co_yield m_context->copier()->copy(dest, m_context->getSCC(), "");
        }
    }
}
