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

    Generator<Instruction> Arithmetic_Vector_Int32::mul(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_mul_lo_u32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::mulHi(std::shared_ptr<Register::Value> dest,
                                                          std::shared_ptr<Register::Value> lhs,
                                                          std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::div(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        // Allocate temporary registers

        auto v_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        co_yield_(
            Instruction("v_ashrrev_i32_e32", {v_1}, {Register::Value::Literal(31), rhs}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_7}, {rhs, v_1}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_7}, {v_7, v_1}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_2}, {v_7}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_3}, {v_2}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_3}, {Register::Value::Literal(0x4f7ffffe), v_3}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_3}, {v_3}, {}, ""));
        co_yield_(
            Instruction("v_ashrrev_i32_e32", {v_4}, {Register::Value::Literal(31), lhs}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_6}, {lhs, v_4}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_1}, {v_4, v_1}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_6}, {v_6, v_4}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_4}, {Register::Value::Literal(0), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4}, {v_4, v_3}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_4}, {v_3, v_4}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_3}, {v_3, v_4}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_3}, {v_6, v_3}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4}, {v_3, v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_6}, {v_6, v_4}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_5}, {Register::Value::Literal(1), v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_6, v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_4}, {v_6, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_3}, {v_3, v_5, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_6}, {v_6, v_4, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_4}, {Register::Value::Literal(1), v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_6, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_6}, {v_3, v_4, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_6}, {v_6, v_1}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {dest}, {v_6, v_1}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int32::mod(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        // Allocate temporary registers

        auto v_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        co_yield_(
            Instruction("v_ashrrev_i32_e32", {v_1}, {Register::Value::Literal(31), rhs}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_2}, {rhs, v_1}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_3}, {v_2, v_1}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_4}, {v_3}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_5}, {Register::Value::Literal(0), v_3}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_6}, {v_4}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_6}, {Register::Value::Literal(0x4f7ffffe), v_6}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_6}, {v_6}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_5}, {v_5, v_6}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_5}, {v_6, v_5}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_6}, {v_6, v_5}, {}, ""));
        co_yield_(
            Instruction("v_ashrrev_i32_e32", {v_4}, {Register::Value::Literal(31), lhs}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_2}, {lhs, v_4}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_2}, {v_2, v_4}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_6}, {v_2, v_6}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_6}, {v_6, v_3}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_2}, {v_2, v_6}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_6}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_2}, {v_2, v_6, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_6}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_2}, {v_2, v_6, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_2}, {v_2, v_4}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {dest}, {v_2, v_4}, {}, ""));
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
        Arithmetic_Vector_Int32::signedShiftR(std::shared_ptr<Register::Value> dest,
                                              std::shared_ptr<Register::Value> value,
                                              std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_ashrrev_i32", {dest}, {shiftAmount, value}, {}, ""));
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

    Generator<Instruction> Arithmetic_Scalar_Int32::mul(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_mul_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::mulHi(std::shared_ptr<Register::Value> dest,
                                                          std::shared_ptr<Register::Value> lhs,
                                                          std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_mul_hi_i32", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::div(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        // Allocate temporary registers
        auto s_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto v_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        co_yield_(Instruction("s_ashr_i32", {s_1}, {rhs, Register::Value::Literal(31)}, {}, ""));
        co_yield_(Instruction("s_add_i32", {s_4}, {rhs, s_1}, {}, ""));
        co_yield_(Instruction("s_xor_b32", {s_4}, {s_4, s_1}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_1}, {s_4}, {}, ""));
        co_yield_(Instruction("s_ashr_i32", {s_2}, {lhs, Register::Value::Literal(31)}, {}, ""));
        co_yield_(Instruction("s_add_i32", {s_5}, {lhs, s_2}, {}, ""));
        co_yield_(Instruction("s_xor_b32", {s_3}, {s_2, s_1}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_1}, {v_1}, {}, ""));
        co_yield_(Instruction("s_xor_b32", {s_5}, {s_5, s_2}, {}, ""));
        co_yield_(Instruction("s_sub_i32", {s_2}, {Register::Value::Literal(0), s_4}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_1}, {Register::Value::Literal(0x4f7ffffe), v_1}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_1}, {v_1}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_2}, {s_2, v_1}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_2}, {v_1, v_2}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_1}, {v_1, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_1}, {s_5, v_1}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_2}, {v_1, s_4}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_2}, {s_5, v_2}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_3}, {Register::Value::Literal(1), v_1}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {s_4, v_2}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_1}, {v_1, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_subrev_u32_e32", {v_3}, {s_4, v_2}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_2}, {v_2, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_3}, {Register::Value::Literal(1), v_1}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {s_4, v_2}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_1}, {v_1, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_1}, {s_3, v_1}, {}, ""));
        co_yield_(Instruction("v_subrev_u32_e32", {v_1}, {s_3, v_1}, {}, ""));
        co_yield_(Instruction(
            "v_readlane_b32", {dest}, {v_1, Register::Value::Literal(0)}, {}, "Move value"));
    }

    Generator<Instruction> Arithmetic_Scalar_Int32::mod(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        // Allocate temporary registers
        auto s_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);

        auto v_1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        co_yield_(Instruction("s_ashr_i32", {s_2}, {rhs, Register::Value::Literal(31)}, {}, ""));
        co_yield_(Instruction("s_add_i32", {s_1}, {rhs, s_2}, {}, ""));
        co_yield_(Instruction("s_xor_b32", {s_1}, {s_1, s_2}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_1}, {s_1}, {}, ""));
        co_yield_(Instruction("s_sub_i32", {s_5}, {Register::Value::Literal(0), s_1}, {}, ""));
        co_yield_(Instruction("s_ashr_i32", {s_4}, {lhs, Register::Value::Literal(31)}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_1}, {v_1}, {}, ""));
        co_yield_(Instruction("s_add_i32", {s_3}, {lhs, s_4}, {}, ""));
        co_yield_(Instruction("s_xor_b32", {s_3}, {s_3, s_4}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_1}, {Register::Value::Literal(0x4f7ffffe), v_1}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_1}, {v_1}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_2}, {s_5, v_1}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_2}, {v_1, v_2}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_1}, {v_1, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_1}, {s_3, v_1}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_1}, {v_1, s_1}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_1}, {s_3, v_1}, {}, ""));

        co_yield_(Instruction("v_subrev_u32_e32", {v_2}, {s_1, v_1}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {s_1, v_1}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_1}, {v_1, v_2, m_context->getVCC()}, {}, ""));

        co_yield_(Instruction("v_subrev_u32_e32", {v_2}, {s_1, v_1}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {s_1, v_1}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_1}, {v_1, v_2, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_1}, {s_4, v_1}, {}, ""));
        co_yield_(Instruction("v_subrev_u32_e32", {v_1}, {s_4, v_1}, {}, ""));
        co_yield_(Instruction(
            "v_readlane_b32", {dest}, {v_1, Register::Value::Literal(0)}, {}, "Move value"));
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
        Arithmetic_Scalar_Int32::signedShiftR(std::shared_ptr<Register::Value> dest,
                                              std::shared_ptr<Register::Value> value,
                                              std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("s_ashr_i32", {dest}, {value, shiftAmount}, {}, ""));
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
