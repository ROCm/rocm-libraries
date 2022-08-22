/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#include <rocRoller/CodeGen/Arithmetic/Int64.hpp>

#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>

namespace rocRoller
{
    RegisterComponent(Arithmetic_Vector_Int64);
    RegisterComponent(Arithmetic_Scalar_Int64);

    // -------------------------------------------------------------------
    // Vector Arithmetic
    // ...................................................................

    Arithmetic_Vector_Int64::Arithmetic_Vector_Int64(std::shared_ptr<Context> context)
        : DirectBase(context)
    {
    }

    std::string Arithmetic_Vector_Int64::name()
    {
        return Name;
    }

    bool Arithmetic_Vector_Int64::Match(Argument const& arg)
    {
        std::shared_ptr<Context> ctx;
        Register::Type           regType;
        VariableType             dataType;

        std::tie(ctx, regType, dataType) = arg;

        AssertFatal(ctx != nullptr);

        auto const& typeInfo = DataTypeInfo::Get(dataType);

        return regType == Register::Type::Vector && !typeInfo.isComplex && typeInfo.isIntegral
               && typeInfo.elementSize == sizeof(int64_t);
        // TODO: Once Arithmetic_Vector_UInt64 is implemented, add:
        // && typeInfo.isSigned
    }

    std::shared_ptr<Arithmetic_Vector_Int64::Base>
        Arithmetic_Vector_Int64::Build(Argument const& arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<Arithmetic_Vector_Int64>(std::get<0>(arg));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int64::getDwords(std::vector<Register::ValuePtr>& dwords,
                                           Register::ValuePtr               input)
    {
        Register::ValuePtr lsd, msd;
        co_yield get2Dwords(lsd, msd, input);
        dwords.resize(2);
        dwords[0] = lsd;
        dwords[1] = msd;
    }

    void
        getLiteralDwords(Register::ValuePtr& lsd, Register::ValuePtr& msd, Register::ValuePtr input)
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

    Generator<Instruction> Arithmetic_Vector_Int64::get2Dwords(Register::ValuePtr& lsd,
                                                               Register::ValuePtr& msd,
                                                               Register::ValuePtr  input)
    {
        if(input->regType() == Register::Type::Literal)
        {
            getLiteralDwords(lsd, msd, input);
            co_return;
        }

        if(input->regType() == Register::Type::Scalar)
        {
            VariableType int64(DataType::Int64);
            auto scalarArith = Component::Get<Arithmetic>(m_context, Register::Type::Scalar, int64);

            std::vector<Register::ValuePtr> scalarSubsets;

            co_yield scalarArith->getDwords(scalarSubsets, input);

            assert(scalarSubsets.size() == 2);
            lsd = scalarSubsets[0];
            msd = scalarSubsets[1];

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

                auto inst
                    = Instruction("v_ashrrev_i32_e32", {msd}, {l31, input}, {}, "Sign extend");
                co_yield inst;

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

        throw std::runtime_error(concatenate("Conversion not implemented for register type ",
                                             input->regType(),
                                             "/",
                                             input->variableType()));
    }

    Generator<Instruction> moveToSingleVGPR(ContextPtr context, Register::ValuePtr& val)
    {
        Register::ValuePtr tmp = val;
        val = Register::Value::Placeholder(context, Register::Type::Vector, tmp->variableType(), 1);

        co_yield context->copier()->copy(val, tmp, "");
    }

    Generator<Instruction> Arithmetic_Vector_Int64::add(Register::ValuePtr dest,
                                                        Register::ValuePtr lhs,
                                                        Register::ValuePtr rhs)
    {
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        if(r0->regType() == Register::Type::Literal)
            std::swap(l0, r0);

        if(r0->regType() == Register::Type::Scalar)
        {
            co_yield moveToSingleVGPR(m_context, r0);
        }

        if(l1->regType() == Register::Type::Scalar
           || (l1->regType() == Register::Type::Literal
               && !m_context->targetArchitecture().isSupportedConstantValue(l1)))
        {
            co_yield moveToSingleVGPR(m_context, l1);
        }

        if(r1->regType() == Register::Type::Scalar
           || (r1->regType() == Register::Type::Literal
               && !m_context->targetArchitecture().isSupportedConstantValue(r1)))
        {
            co_yield moveToSingleVGPR(m_context, r1);
        }

        auto carry = m_context->getVCC();

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction(
            "v_add_co_u32", {dest->subset({0}), carry}, {l0, r0}, {}, "least significant half"));

        co_yield_(Instruction("v_addc_co_u32",
                              {dest->subset({1}), carry},
                              {l1, r1, carry},
                              {},
                              "most significant half"));
    }

    Generator<Instruction> Arithmetic_Vector_Int64::sub(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        auto borrow = m_context->getVCC();

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction(
            "v_sub_co_u32", {dest->subset({0}), borrow}, {l0, r0}, {}, "least significant half"));

        co_yield_(Instruction("v_subb_co_u32",
                              {dest->subset({1}), borrow},
                              {l1, r1, borrow},
                              {},
                              "most significant half"));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int64::shiftL(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_lshlrev_b64", {dest}, {shiftAmount->subset({0}), value}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int64::addShiftL(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs,
                                           std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield Arithmetic_Vector_Int64::add(dest, lhs, rhs);
        co_yield Arithmetic_Vector_Int64::shiftL(dest, dest, shiftAmount);
    }

    Generator<Instruction>
        Arithmetic_Vector_Int64::shiftLAdd(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> shiftAmount,
                                           std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield Arithmetic_Vector_Int64::shiftL(dest, lhs, shiftAmount);
        co_yield Arithmetic_Vector_Int64::add(dest, dest, rhs);
    }

    Generator<Instruction> Arithmetic_Vector_Int64::gt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_gt_i64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int64::ge(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_ge_i64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int64::lt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_lt_i64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int64::le(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_le_i64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int64::eq(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("v_cmp_eq_i64", {dest}, {lhs, rhs}, {}, ""));
    }

    // -------------------------------------------------------------------
    // Scalar Arithmetic
    // ...................................................................

    Arithmetic_Scalar_Int64::Arithmetic_Scalar_Int64(std::shared_ptr<Context> context)
        : DirectBase(context)
    {
    }

    std::string Arithmetic_Scalar_Int64::name()
    {
        return Name;
    }

    bool Arithmetic_Scalar_Int64::Match(Argument const& arg)
    {
        std::shared_ptr<Context> ctx;
        Register::Type           regType;
        VariableType             variableType;

        std::tie(ctx, regType, variableType) = arg;

        AssertFatal(ctx != nullptr);

        auto const& typeInfo = DataTypeInfo::Get(variableType);

        return regType == Register::Type::Scalar && !typeInfo.isComplex && typeInfo.isIntegral
               && typeInfo.elementSize == sizeof(int64_t);
        // TODO: Once Arithmetic_Scalar_UInt64 is implemented, add:
        // && typeInfo.isSigned
    }

    std::shared_ptr<Arithmetic_Scalar_Int64::Base>
        Arithmetic_Scalar_Int64::Build(Argument const& arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<Arithmetic_Scalar_Int64>(std::get<0>(arg));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int64::getDwords(std::vector<Register::ValuePtr>& dwords,
                                           Register::ValuePtr               input)
    {
        Register::ValuePtr lsd, msd;
        co_yield get2Dwords(lsd, msd, input);
        dwords.resize(2);
        dwords[0] = lsd;
        dwords[1] = msd;
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::get2Dwords(Register::ValuePtr& lsd,
                                                               Register::ValuePtr& msd,
                                                               Register::ValuePtr  input)
    {
        if(input->regType() == Register::Type::Literal)
        {
            getLiteralDwords(lsd, msd, input);
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

                auto inst = Instruction("s_ashr_i32", {msd}, {input, l31}, {}, "Sign extend");
                co_yield inst;

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

        throw std::runtime_error(concatenate("Conversion not implemented for register type ",
                                             input->regType(),
                                             "/",
                                             input->variableType()));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::add(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction(
            "s_add_u32", {dest->subset({0})}, {l0, r0}, {}, "least significant half; sets scc"));

        co_yield_(Instruction(
            "s_addc_u32", {dest->subset({1})}, {l1, r1}, {}, "most significant half; uses scc"));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::sub(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction(
            "s_sub_u32", {dest->subset({0})}, {l0, r0}, {}, "least significant half; sets scc"));

        co_yield_(Instruction(
            "s_subb_u32", {dest->subset({1})}, {l1, r1}, {}, "most significant half; uses scc"));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int64::shiftL(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        if(shiftAmount->regType() == Register::Type::Literal)
        {
            co_yield_(Instruction("s_lshl_b64", {dest}, {value, shiftAmount}, {}, ""));
        }
        else
        {
            co_yield_(Instruction("s_lshl_b64", {dest}, {value, shiftAmount->subset({0})}, {}, ""));
        }
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int64::addShiftL(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs,
                                           std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield Arithmetic_Scalar_Int64::add(dest, lhs, rhs);
        co_yield Arithmetic_Scalar_Int64::shiftL(dest, dest, shiftAmount);
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int64::shiftLAdd(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> shiftAmount,
                                           std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield Arithmetic_Scalar_Int64::shiftL(dest, lhs, shiftAmount);
        co_yield Arithmetic_Scalar_Int64::add(dest, dest, rhs);
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::gt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto vTemp = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 2);
        co_yield vTemp->allocate();

        co_yield m_context->copier()->copy(vTemp, rhs, "");

        co_yield_(Instruction("v_cmp_gt_i64", {dest}, {lhs, vTemp}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::ge(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto vTemp = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 2);
        co_yield vTemp->allocate();

        co_yield m_context->copier()->copy(vTemp, rhs, "");

        co_yield_(Instruction("v_cmp_ge_i64", {dest}, {lhs, vTemp}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::lt(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto vTemp = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 2);
        co_yield vTemp->allocate();

        co_yield m_context->copier()->copy(vTemp, rhs, "");

        co_yield_(Instruction("v_cmp_lt_i64", {dest}, {lhs, vTemp}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::le(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto vTemp = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 2);
        co_yield vTemp->allocate();

        co_yield m_context->copier()->copy(vTemp, rhs, "");

        co_yield_(Instruction("v_cmp_le_i64", {dest}, {lhs, vTemp}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::eq(std::shared_ptr<Register::Value> dest,
                                                       std::shared_ptr<Register::Value> lhs,
                                                       std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_cmp_eq_u64", {}, {lhs, rhs}, {}, ""));

        if(dest != nullptr)
        {
            co_yield m_context->copier()->copy(dest, m_context->getSCC(), "");
        }
    }
}
