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

    Generator<Instruction> Arithmetic_Vector_Int64::mul(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        AssertFatal(l0 != nullptr);
        AssertFatal(l1 != nullptr);
        AssertFatal(r0 != nullptr);
        AssertFatal(r1 != nullptr);

        // The high bits of the output consist of the sum of:
        // high bits of low * low,
        // low bits of low * high,
        // low bits of high * low.
        //
        // If any of these are known to be zero, that can be omitted.
        // If they are all zero, we still need to write a zero into the output.
        // Whichever of these is generated first will write into the destination,
        // additional required instructions will require registers to be allocated.

        // cppcheck-suppress nullPointer
        bool need_ll = !l0->isZeroLiteral() && !r0->isZeroLiteral();
        // cppcheck-suppress nullPointer
        bool need_lh = !l0->isZeroLiteral() && !r1->isZeroLiteral();
        // cppcheck-suppress nullPointer
        bool need_hl = !l1->isZeroLiteral() && !r0->isZeroLiteral();

        // Multiply instructions only accept constant values, not literals.

        if(l0->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(l0))
            co_yield moveToSingleVGPR(m_context, l0);

        if(l1->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(l1))
            co_yield moveToSingleVGPR(m_context, l1);

        if(r0->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(r0))
            co_yield moveToSingleVGPR(m_context, r0);

        if(r1->regType() == Register::Type::Literal
           && !m_context->targetArchitecture().isSupportedConstantValue(r1))
            co_yield moveToSingleVGPR(m_context, r1);

        // These instructions don't support scalar * scalar.
        if(l0->regType() == Register::Type::Scalar && r0->regType() == Register::Type::Scalar)
        {
            co_yield moveToSingleVGPR(m_context, l0);
            if(l1->regType() == Register::Type::Scalar)
                co_yield moveToSingleVGPR(m_context, l1);
        }

        co_yield Register::AllocateIfNeeded(dest);

        int                               numHighComponents = 0;
        std::array<Register::ValuePtr, 3> highComponents
            = {dest->subset({1}),
               Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1),
               Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1)};

        if(need_lh)
        {
            co_yield_(Instruction("v_mul_lo_u32",
                                  {highComponents[numHighComponents]},
                                  {l0, r1},
                                  {},
                                  "most significant: low of low * high"));
            numHighComponents++;
        }
        else
        {
            co_yield Instruction::Comment("low of low * high omitted due to zero input.");
        }

        if(need_hl)
        {
            co_yield_(Instruction("v_mul_lo_u32",
                                  {highComponents[numHighComponents]},
                                  {l1, r0},
                                  {},
                                  "most significant: low of high * low"));
            numHighComponents++;
        }
        else
        {
            co_yield Instruction::Comment("low of high * low omitted due to zero input.");
        }

        if(need_ll)
        {
            co_yield_(Instruction("v_mul_hi_u32",
                                  {highComponents[numHighComponents]},
                                  {l0, r0},
                                  {},
                                  "most significant: high of low * low"));
            numHighComponents++;

            co_yield_(Instruction("v_mul_lo_u32",
                                  {dest->subset({0})},
                                  {l0, r0},
                                  {},
                                  "least significant: low of low * low"));
        }
        else
        {
            co_yield m_context->copier()->copy(dest->subset({0}),
                                               Register::Value::Literal(0),
                                               "low * low optimized due to zero input.");
        }

        if(numHighComponents == 0)
        {
            co_yield m_context->copier()->copy(dest->subset({1}),
                                               Register::Value::Literal(0),
                                               "high bits of output optimized due to zero input.");
        }
        else if(numHighComponents == 1)
        {
            // The first high multiply writes into the high bits of the output, nothing to do.
            co_yield Instruction::Comment(
                "Most significant: sum omitted due to only one valid input.");
        }
        else if(numHighComponents == 2)
        {
            co_yield_(Instruction("v_add_u32",
                                  {dest->subset({1})},
                                  {highComponents[0], highComponents[1]},
                                  {},
                                  "most significant: sum"));
        }
        else if(numHighComponents == 3)
        {
            co_yield_(Instruction("v_add3_u32",
                                  {dest->subset({1})},
                                  {highComponents[0], highComponents[1], highComponents[2]},
                                  {},
                                  "most significant: sum"));
        }
        else
        {
            throw std::runtime_error(
                concatenate("Shouldn't get here: numHighComponents = ", numHighComponents));
        }
    }

    Generator<Instruction> Arithmetic_Vector_Int64::div(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        // Generated using the assembly_to_instructions.py script with the following HIP code:
        //  extern "C"
        //  __global__ void hello_world(long int * ptr, long int *value1, long int *value2)
        //  {
        //    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //    ptr[idx] = value1[idx] / value2[idx];
        //  }
        //
        // Generated code was modified to use the provided dest, lhs and rhs registers and
        // to save the result in the dest register instead of memory.

        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        auto s_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_8 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_9 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_10 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_11 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_12 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_13 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_14 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_15 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_16 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_17 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_18 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_19 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_20 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);

        auto label_21 = m_context->labelAllocator()->label("BB0_2");
        auto label_22 = m_context->labelAllocator()->label("BB0_4");

        co_yield m_context->copier()->copy(v_20, l0, "");
        co_yield m_context->copier()->copy(v_2, l1, "");
        co_yield m_context->copier()->copy(v_7, r0, "");
        co_yield m_context->copier()->copy(v_3, r1, "");

        co_yield_(Instruction("v_or_b32_e32", {dest->subset({1})}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u64_e32",
                              {m_context->getVCC()},
                              {Register::Value::Literal(0), dest},
                              {},
                              ""));
        co_yield_(Instruction("s_and_saveexec_b64", {s_5}, {m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("s_xor_b64", {s_6}, {m_context->getExec(), s_5}, {}, ""));
        co_yield_(Instruction("s_cbranch_execz", {}, {label_21}, {}, ""));
        co_yield_(Instruction(
            "v_ashrrev_i32_e32", {dest->subset({0})}, {Register::Value::Literal(31), v_3}, {}, ""));
        co_yield_(Instruction(
            "v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_7, dest->subset({0})}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_3},
                              {m_context->getVCC(), v_3, dest->subset({0}), m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_3}, {v_3, dest->subset({0})}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_7}, {v_7, dest->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {dest->subset({1})}, {v_7}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_8}, {v_3}, {}, ""));
        co_yield_(Instruction("v_sub_co_u32_e32",
                              {v_9},
                              {m_context->getVCC(), Register::Value::Literal(0), v_7},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_subb_co_u32_e32",
            {v_10},
            {m_context->getVCC(), Register::Value::Literal(0), v_3, m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_mac_f32_e32",
                              {dest->subset({1})},
                              {Register::Value::Literal(0x4f800000), v_8},
                              {},
                              ""));
        co_yield_(Instruction("v_rcp_f32_e32", {dest->subset({1})}, {dest->subset({1})}, {}, ""));
        co_yield m_context->copier()->copy(v_11, Register::Value::Literal(0), "");
        co_yield m_context->copier()->copy(v_12, Register::Value::Literal(0), "");
        co_yield_(Instruction("v_mul_f32_e32",
                              {dest->subset({1})},
                              {Register::Value::Literal(0x5f7ffffc), dest->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_f32_e32",
                              {v_8},
                              {Register::Value::Literal(0x2f800000), dest->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction("v_trunc_f32_e32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction("v_mac_f32_e32",
                              {dest->subset({1})},
                              {Register::Value::Literal(0xcf800000), v_8},
                              {},
                              ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_8}, {v_8}, {}, ""));
        co_yield_(
            Instruction("v_cvt_u32_f32_e32", {dest->subset({1})}, {dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_9, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_10, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_9, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_add3_u32", {v_14}, {v_15, v_13, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {v_9, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {dest->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {dest->subset({1}), v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {dest->subset({1}), v_14}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_17, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_11, v_15, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_18}, {v_8, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {v_8, v_16}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_13, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_8, v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_15, v_18, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_13},
                              {m_context->getVCC(), v_17, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_8, v_14}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_14}, {m_context->getVCC(), v_15, v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_11, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_add_co_u32", {dest->subset({1})}, {s_5, dest->subset({1}), v_14}, {}, ""));
        co_yield_(
            Instruction("v_addc_co_u32", {v_14}, {m_context->getVCC(), v_8, v_15, s_5}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_9, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_10}, {v_10, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_9, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_add3_u32", {v_10}, {v_16, v_13, v_10}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_9}, {v_9, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_14, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_17}, {v_14, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_19}, {dest->subset({1}), v_10}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_9}, {dest->subset({1}), v_9}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_18}, {dest->subset({1}), v_10}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_9}, {m_context->getVCC(), v_9, v_19}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_18},
                              {m_context->getVCC(), v_11, v_18, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_9}, {m_context->getVCC(), v_9, v_17}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {v_14, v_10}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_9},
                              {m_context->getVCC(), v_18, v_16, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_13},
                              {m_context->getVCC(), v_13, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_10}, {v_14, v_10}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_9}, {m_context->getVCC(), v_9, v_10}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_10},
                              {m_context->getVCC(), v_11, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_u32_e32", {v_8}, {v_8, v_15}, {}, ""));
        co_yield_(
            Instruction("v_addc_co_u32", {v_8}, {m_context->getVCC(), v_8, v_10, s_5}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32",
                              {dest->subset({1})},
                              {m_context->getVCC(), dest->subset({1}), v_9},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_addc_co_u32_e32",
            {v_8},
            {m_context->getVCC(), Register::Value::Literal(0), v_8, m_context->getVCC()},
            {},
            ""));
        co_yield_(
            Instruction("v_ashrrev_i32_e32", {v_9}, {Register::Value::Literal(31), v_2}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_20}, {m_context->getVCC(), v_20, v_9}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_20}, {v_20, v_9}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_2},
                              {m_context->getVCC(), v_2, v_9, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_20, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_20, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_2}, {v_2, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_10}, {v_20, v_8}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_14}, {m_context->getVCC(), v_15, v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_10},
                              {m_context->getVCC(), v_11, v_10, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {v_2, dest->subset({1})}, {}, ""));
        co_yield_(
            Instruction("v_mul_lo_u32", {dest->subset({1})}, {v_2, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32",
                              {dest->subset({1})},
                              {m_context->getVCC(), v_14, dest->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_2, v_8}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {dest->subset({1})},
                              {m_context->getVCC(), v_10, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_10},
                              {m_context->getVCC(), v_15, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {v_2, v_8}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32",
                              {dest->subset({1})},
                              {m_context->getVCC(), dest->subset({1}), v_8},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_11, v_10, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_10}, {v_3, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_7, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_add3_u32", {v_10}, {v_15, v_14, v_10}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {v_7, dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_14}, {v_2, v_10}, {}, ""));
        co_yield_(
            Instruction("v_sub_co_u32_e32", {v_20}, {m_context->getVCC(), v_20, v_15}, {}, ""));
        co_yield_(
            Instruction("v_subb_co_u32", {v_14}, {s_5, v_14, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_sub_co_u32", {v_15}, {s_5, v_20, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_subbrev_co_u32", {v_14}, {s_5, Register::Value::Literal(0), v_14, s_5}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32", {s_5}, {v_14, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_13},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_5},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_ge_u32", {s_5}, {v_15, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_15},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_5},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_eq_u32", {s_5}, {v_14, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_13, v_15, s_5}, {}, ""));
        co_yield_(Instruction(
            "v_add_co_u32", {v_15}, {s_5, Register::Value::Literal(2), dest->subset({1})}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {v_2},
                              {m_context->getVCC(), v_2, v_10, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_addc_co_u32", {v_13}, {s_5, Register::Value::Literal(0), v_8, s_5}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction(
            "v_add_co_u32", {v_16}, {s_5, Register::Value::Literal(1), dest->subset({1})}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_10},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_20, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_addc_co_u32", {v_17}, {s_5, Register::Value::Literal(0), v_8, s_5}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_20},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_cmp_eq_u32_e32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32", {s_5}, {Register::Value::Literal(0), v_14}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_20}, {v_10, v_20, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32_e32",
                              {m_context->getVCC()},
                              {Register::Value::Literal(0), v_20},
                              {},
                              ""));
        co_yield_(Instruction("v_cndmask_b32", {v_2}, {v_16, v_15, s_5}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_17, v_13, s_5}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32_e32", {v_2}, {dest->subset({1}), v_2, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_7}, {v_9, dest->subset({0})}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_20}, {v_8, v_14, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_2}, {v_2, v_7}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_20}, {v_20, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_sub_co_u32_e32", {dest->subset({0})}, {m_context->getVCC(), v_2, v_7}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {dest->subset({1})},
                              {m_context->getVCC(), v_20, v_7, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction::Label(label_21));
        co_yield_(Instruction("s_or_saveexec_b64", {s_5}, {s_6}, {}, ""));
        co_yield_(
            Instruction("s_xor_b64", {m_context->getExec()}, {m_context->getExec(), s_5}, {}, ""));
        co_yield_(Instruction("s_cbranch_execz", {}, {label_22}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_2}, {v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_3}, {Register::Value::Literal(0), v_7}, {}, ""));
        co_yield m_context->copier()->copy(dest->subset({1}), Register::Value::Literal(0), "");
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_2}, {v_2}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_2}, {Register::Value::Literal(0x4f7ffffe), v_2}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_2}, {v_2}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_3}, {v_3, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_3}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_2}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_2}, {v_20, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_3}, {v_2, v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_20}, {v_20, v_3}, {}, ""));
        co_yield_(Instruction(
            "v_add_u32_e32", {dest->subset({0})}, {Register::Value::Literal(1), v_2}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_3}, {v_20, v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_20, v_7}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_20}, {v_20, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32_e32", {v_2}, {v_2, dest->subset({0}), m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_3}, {Register::Value::Literal(1), v_2}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_20, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32_e32", {dest->subset({0})}, {v_2, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction::Label(label_22));
        co_yield_(
            Instruction("s_or_b64", {m_context->getExec()}, {m_context->getExec(), s_5}, {}, ""));
    }

    Generator<Instruction> Arithmetic_Vector_Int64::mod(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        // Generated using the assembly_to_instructions.py script with the following HIP code:
        //  extern "C"
        //  __global__ void hello_world(long int * ptr, long int *value1, long int *value2)
        //  {
        //    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //    ptr[idx] = value1[idx] % value2[idx];
        //  }
        //
        // Generated code was modified to use the provided dest, lhs and rhs registers and
        // to save the result in the dest register instead of memory.

        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);

        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        auto v_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_4 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 2);
        co_yield v_4->allocate();
        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        auto s_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_8 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_9 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_10 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_11 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_12 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_13 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_14 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_15 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_16 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_17 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_18 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_19 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto s_20 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);

        auto label_21 = m_context->labelAllocator()->label("BB0_2");
        auto label_22 = m_context->labelAllocator()->label("BB0_4");

        co_yield m_context->copier()->copy(v_19, l0, "");
        co_yield m_context->copier()->copy(v_2, l1, "");
        co_yield m_context->copier()->copy(v_7, r0, "");
        co_yield m_context->copier()->copy(v_3, r1, "");

        co_yield m_context->copier()->copy(v_4->subset({0}), Register::Value::Literal(0), "");
        co_yield_(Instruction("v_or_b32_e32", {v_4->subset({1})}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction(
            "v_cmp_ne_u64_e32", {m_context->getVCC()}, {Register::Value::Literal(0), v_4}, {}, ""));
        co_yield_(Instruction("s_and_saveexec_b64", {s_5}, {m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("s_xor_b64", {s_6}, {m_context->getExec(), s_5}, {}, ""));
        co_yield_(Instruction("s_cbranch_execz", {}, {label_21}, {}, ""));
        co_yield_(Instruction(
            "v_ashrrev_i32_e32", {v_4->subset({0})}, {Register::Value::Literal(31), v_3}, {}, ""));
        co_yield_(Instruction(
            "v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_7, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_3},
                              {m_context->getVCC(), v_3, v_4->subset({0}), m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_3}, {v_3, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_7}, {v_7, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_4->subset({0})}, {v_7}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_4->subset({1})}, {v_3}, {}, ""));
        co_yield_(Instruction("v_sub_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), Register::Value::Literal(0), v_7},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_subb_co_u32_e32",
            {v_9},
            {m_context->getVCC(), Register::Value::Literal(0), v_3, m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_mac_f32_e32",
                              {v_4->subset({0})},
                              {Register::Value::Literal(0x4f800000), v_4->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction("v_rcp_f32_e32", {v_4->subset({0})}, {v_4->subset({0})}, {}, ""));
        co_yield m_context->copier()->copy(v_10, Register::Value::Literal(0), "");
        co_yield m_context->copier()->copy(v_11, Register::Value::Literal(0), "");
        co_yield_(Instruction("v_mul_f32_e32",
                              {v_4->subset({0})},
                              {Register::Value::Literal(0x5f7ffffc), v_4->subset({0})},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_f32_e32",
                              {v_4->subset({1})},
                              {Register::Value::Literal(0x2f800000), v_4->subset({0})},
                              {},
                              ""));
        co_yield_(Instruction("v_trunc_f32_e32", {v_4->subset({1})}, {v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mac_f32_e32",
                              {v_4->subset({0})},
                              {Register::Value::Literal(0xcf800000), v_4->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_4->subset({1})}, {v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_4->subset({0})}, {v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_8, v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_9, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {v_8, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_add3_u32", {v_13}, {v_14, v_12, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {v_8, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_4->subset({0}), v_13}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_4->subset({0}), v_15}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {v_4->subset({0}), v_13}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_12}, {m_context->getVCC(), v_16, v_12}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_10, v_14, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_4->subset({1}), v_15}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {v_4->subset({1}), v_15}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_12}, {m_context->getVCC(), v_12, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_4->subset({1}), v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_14, v_17, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_12},
                              {m_context->getVCC(), v_16, v_11, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_4->subset({1}), v_13}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_14, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_10, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(
            Instruction("v_add_co_u32", {v_4->subset({0})}, {s_5, v_4->subset({0}), v_13}, {}, ""));
        co_yield_(Instruction(
            "v_addc_co_u32", {v_13}, {m_context->getVCC(), v_4->subset({1}), v_14, s_5}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_8, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_9}, {v_9, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_8, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_add3_u32", {v_9}, {v_15, v_12, v_9}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {v_8, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_13, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {v_13, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_18}, {v_4->subset({0}), v_9}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_8}, {v_4->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_4->subset({0}), v_9}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_8}, {m_context->getVCC(), v_8, v_18}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_17},
                              {m_context->getVCC(), v_10, v_17, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_8}, {m_context->getVCC(), v_8, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_12}, {v_13, v_9}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_17, v_15, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_12},
                              {m_context->getVCC(), v_12, v_11, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_9}, {v_13, v_9}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_8}, {m_context->getVCC(), v_8, v_9}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_9},
                              {m_context->getVCC(), v_10, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(
            Instruction("v_add_u32_e32", {v_4->subset({1})}, {v_4->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32",
                              {v_4->subset({1})},
                              {m_context->getVCC(), v_4->subset({1}), v_9, s_5},
                              {},
                              ""));
        co_yield_(Instruction("v_add_co_u32_e32",
                              {v_4->subset({0})},
                              {m_context->getVCC(), v_4->subset({0}), v_8},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_4->subset({1})},
                              {m_context->getVCC(),
                               Register::Value::Literal(0),
                               v_4->subset({1}),
                               m_context->getVCC()},
                              {},
                              ""));
        co_yield_(
            Instruction("v_ashrrev_i32_e32", {v_8}, {Register::Value::Literal(31), v_2}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_19}, {m_context->getVCC(), v_19, v_8}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_19}, {v_19, v_8}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_2},
                              {m_context->getVCC(), v_2, v_8, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_19, v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {v_19, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_2}, {v_2, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_9}, {v_19, v_4->subset({1})}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_14, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_9},
                              {m_context->getVCC(), v_10, v_9, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_12}, {v_2, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4->subset({0})}, {v_2, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32",
                              {v_4->subset({0})},
                              {m_context->getVCC(), v_13, v_4->subset({0})},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {v_2, v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_4->subset({0})},
                              {m_context->getVCC(), v_9, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_9},
                              {m_context->getVCC(), v_14, v_11, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4->subset({1})}, {v_2, v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32",
                              {v_4->subset({0})},
                              {m_context->getVCC(), v_4->subset({0}), v_4->subset({1})},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_4->subset({1})},
                              {m_context->getVCC(), v_10, v_9, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_9}, {v_3, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4->subset({1})}, {v_7, v_4->subset({1})}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {v_7, v_4->subset({0})}, {}, ""));
        co_yield_(
            Instruction("v_add3_u32", {v_4->subset({1})}, {v_13, v_4->subset({1}), v_9}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_4->subset({0})}, {v_7, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_9}, {v_2, v_4->subset({1})}, {}, ""));
        co_yield_(Instruction(
            "v_sub_co_u32_e32", {v_19}, {m_context->getVCC(), v_19, v_4->subset({0})}, {}, ""));
        co_yield_(Instruction(
            "v_subb_co_u32", {v_4->subset({0})}, {s_5, v_9, v_3, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_sub_co_u32", {v_9}, {s_5, v_19, v_7}, {}, ""));
        co_yield_(Instruction("v_subbrev_co_u32",
                              {v_13, s_20},
                              {Register::Value::Literal(0), v_4->subset({0}), s_5},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_ge_u32", {s_20}, {v_13, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_14},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_20},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_ge_u32", {s_20}, {v_9, v_7}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {v_2},
                              {m_context->getVCC(), v_2, v_4->subset({1}), m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_12},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_20},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_eq_u32", {s_20}, {v_13, v_3}, {}, ""));
        co_yield_(Instruction(
            "v_subb_co_u32", {v_4->subset({0})}, {s_5, v_4->subset({0}), v_3, s_5}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_14, v_12, s_20}, {}, ""));
        co_yield_(Instruction("v_sub_co_u32", {v_12}, {s_5, v_9, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_4->subset({1})},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_19, v_7}, {}, ""));
        co_yield_(Instruction("v_subbrev_co_u32",
                              {v_4->subset({0})},
                              {s_5, Register::Value::Literal(0), v_4->subset({0}), s_5},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_7},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("v_cmp_eq_u32_e32", {m_context->getVCC()}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32", {s_5}, {Register::Value::Literal(0), v_14}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32_e32", {v_7}, {v_4->subset({1}), v_7, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction(
            "v_cmp_ne_u32_e32", {m_context->getVCC()}, {Register::Value::Literal(0), v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_7}, {v_9, v_12, s_5}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32", {v_4->subset({0})}, {v_13, v_4->subset({0}), s_5}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_19}, {v_19, v_7, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32_e32", {v_2}, {v_2, v_4->subset({0}), m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_19}, {v_19, v_8}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_2}, {v_2, v_8}, {}, ""));
        co_yield_(Instruction(
            "v_sub_co_u32_e32", {dest->subset({0})}, {m_context->getVCC(), v_19, v_8}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {dest->subset({1})},
                              {m_context->getVCC(), v_2, v_8, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction::Label(label_21));
        co_yield_(Instruction("s_or_saveexec_b64", {s_5}, {s_6}, {}, ""));
        co_yield_(
            Instruction("s_xor_b64", {m_context->getExec()}, {m_context->getExec(), s_5}, {}, ""));
        co_yield_(Instruction("s_cbranch_execz", {}, {label_22}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_2}, {v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_3}, {Register::Value::Literal(0), v_7}, {}, ""));
        co_yield m_context->copier()->copy(dest->subset({1}), Register::Value::Literal(0), "");
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_2}, {v_2}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_2}, {Register::Value::Literal(0x4f7ffffe), v_2}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_2}, {v_2}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_3}, {v_3, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_3}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_2}, {v_2, v_3}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_2}, {v_19, v_2}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_2}, {v_2, v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_19}, {v_19, v_2}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_2}, {v_19, v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_19, v_7}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_19}, {v_19, v_2, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_2}, {v_19, v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v_19, v_7}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32_e32", {dest->subset({0})}, {v_19, v_2, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction::Label(label_22));
        co_yield_(
            Instruction("s_or_b64", {m_context->getExec()}, {m_context->getExec(), s_5}, {}, ""));
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
        Arithmetic_Vector_Int64::shiftR(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("v_ashrrev_i64", {dest}, {shiftAmount->subset({0}), value}, {}, ""));
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

    Generator<Instruction>
        Arithmetic_Vector_Int64::bitwiseAnd(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction(
            "v_and_b32", {dest->subset({0})}, {lhs->subset({0}), rhs->subset({0})}, {}, ""));
        co_yield_(Instruction(
            "v_and_b32", {dest->subset({1})}, {lhs->subset({1}), rhs->subset({1})}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Vector_Int64::bitwiseXor(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction(
            "v_xor_b32", {dest->subset({0})}, {lhs->subset({0}), rhs->subset({0})}, {}, ""));
        co_yield_(Instruction(
            "v_xor_b32", {dest->subset({1})}, {lhs->subset({1}), rhs->subset({1})}, {}, ""));
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

    Generator<Instruction> Arithmetic_Scalar_Int64::mul(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        // TODO: Optimize for cases when individual dwords are literal 0, as in the vector mul().
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        co_yield Register::AllocateIfNeeded(dest);

        co_yield_(Instruction("s_mul_i32", {dest->subset({0})}, {l0, r0}, {}, "least significant"));

        co_yield_(Instruction("s_mul_hi_u32",
                              {dest->subset({1})},
                              {l0, r0},
                              {},
                              "most significant: high of low * low"));

        auto lh = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        co_yield_(
            Instruction("s_mul_i32", {lh}, {l0, r1}, {}, "most significant: low of low * high"));

        auto hl = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        co_yield_(
            Instruction("s_mul_i32", {hl}, {l1, r0}, {}, "most significant: low of high * low"));

        co_yield_(Instruction("s_add_u32",
                              {dest->subset({1})},
                              {dest->subset({1}), lh},
                              {},
                              "most significant: sum"));

        co_yield_(Instruction("s_add_u32",
                              {dest->subset({1})},
                              {dest->subset({1}), hl},
                              {},
                              "most significant: sum"));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::div(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        // Generated using the assembly_to_instructions.py script with the following HIP code:
        //  extern "C"
        //  __global__ void hello_world(long int * ptr, long int value1, long int value2)
        //  {
        //    *ptr = value1 / value2 ;
        //  }
        //
        // Generated code was modified to use the provided dest, lhs and rhs registers and
        // to save the result in the dest register instead of memory.
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        auto s_0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        auto s_2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        auto s_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        co_yield s_6->allocate();
        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_8 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_10 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_11 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_12 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_13 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_14 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_15 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_16 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_17 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_18 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_19 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_20 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto s_22 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        co_yield s_22->allocate();
        auto s_23 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        co_yield s_23->allocate();

        auto label_24 = m_context->labelAllocator()->label("BB0_2");
        auto label_25 = m_context->labelAllocator()->label("BB0_3");
        auto label_26 = m_context->labelAllocator()->label("BB0_4");

        co_yield_(Instruction("s_or_b64", {s_0}, {lhs, rhs}, {}, ""));
        co_yield m_context->copier()->copy(s_0->subset({0}), Register::Value::Literal(0), "");
        co_yield_(Instruction("s_cmp_lg_u64", {s_0}, {Register::Value::Literal(0)}, {}, ""));
        co_yield m_context->brancher()->branchIfZero(label_24, m_context->getSCC());
        co_yield_(Instruction(
            "s_ashr_i32", {s_6->subset({0})}, {r1, Register::Value::Literal(31)}, {}, ""));
        co_yield_(Instruction("s_add_u32", {s_0->subset({0})}, {r0, s_6->subset({0})}, {}, ""));
        co_yield m_context->copier()->copy(s_6->subset({1}), s_6->subset({0}), "");
        co_yield_(Instruction("s_addc_u32", {s_0->subset({1})}, {r1, s_6->subset({0})}, {}, ""));
        co_yield_(Instruction("s_xor_b64", {s_5}, {s_0, s_6}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_7}, {s_5->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_8}, {s_5->subset({1})}, {}, ""));
        co_yield_(Instruction(
            "s_sub_u32", {s_2}, {Register::Value::Literal(0), s_5->subset({0})}, {}, ""));
        co_yield_(Instruction("s_subb_u32",
                              {s_23->subset({0})},
                              {Register::Value::Literal(0), s_5->subset({1})},
                              {},
                              ""));
        co_yield m_context->copier()->copy(v_10, Register::Value::Literal(0), "");
        co_yield_(Instruction(
            "v_mac_f32_e32", {v_7}, {Register::Value::Literal(0x4f800000), v_8}, {}, ""));
        co_yield_(Instruction("v_rcp_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield m_context->copier()->copy(v_11, Register::Value::Literal(0), "");
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_7}, {Register::Value::Literal(0x5f7ffffc), v_7}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_8}, {Register::Value::Literal(0x2f800000), v_7}, {}, ""));
        co_yield_(Instruction("v_trunc_f32_e32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction(
            "v_mac_f32_e32", {v_7}, {Register::Value::Literal(0xcf800000), v_8}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {s_2, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_23->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_12}, {v_13, v_12}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_12}, {v_12, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_7, v_12}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_7, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {v_7, v_12}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_16, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_10, v_14, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_8, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {v_8, v_15}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_13, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_8, v_12}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_14, v_17, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_13},
                              {m_context->getVCC(), v_16, v_11, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_8, v_12}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_12}, {m_context->getVCC(), v_14, v_12}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_10, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_co_u32", {v_7}, {s_0, v_7, v_12}, {}, ""));
        co_yield_(
            Instruction("v_addc_co_u32", {v_12}, {m_context->getVCC(), v_8, v_14, s_0}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {s_2, v_12}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_13}, {v_15, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {s_23->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_13}, {v_13, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {s_2, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_12, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_18}, {v_12, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_19}, {v_7, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_7, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_20}, {v_7, v_13}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_16}, {m_context->getVCC(), v_16, v_19}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_20},
                              {m_context->getVCC(), v_10, v_20, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_16}, {m_context->getVCC(), v_16, v_18}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_12, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_16},
                              {m_context->getVCC(), v_20, v_17, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_15, v_11, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_12, v_13}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_12}, {m_context->getVCC(), v_16, v_12}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_13},
                              {m_context->getVCC(), v_10, v_15, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_u32_e32", {v_8}, {v_8, v_14}, {}, ""));
        co_yield_(Instruction(
            "s_ashr_i32", {s_23->subset({0})}, {l1, Register::Value::Literal(31)}, {}, ""));
        co_yield_(
            Instruction("v_addc_co_u32", {v_8}, {m_context->getVCC(), v_8, v_13, s_0}, {}, ""));
        co_yield_(Instruction("s_add_u32", {s_0->subset({0})}, {l0, s_23->subset({0})}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_7, v_12}, {}, ""));
        co_yield m_context->copier()->copy(s_23->subset({1}), s_23->subset({0}), "");
        co_yield_(Instruction("s_addc_u32", {s_0->subset({1})}, {l1, s_23->subset({0})}, {}, ""));
        co_yield_(Instruction(
            "v_addc_co_u32_e32",
            {v_8},
            {m_context->getVCC(), Register::Value::Literal(0), v_8, m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("s_xor_b64", {s_22}, {s_0, s_23}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_22->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_22->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_12}, {s_22->subset({0}), v_8}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_14}, {m_context->getVCC(), v_13, v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_12},
                              {m_context->getVCC(), v_10, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {s_22->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_7}, {s_22->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_14, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_7},
                              {m_context->getVCC(), v_12, v_15, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_12},
                              {m_context->getVCC(), v_13, v_11, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_10, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {s_5->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_12}, {v_14, v_12}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_5->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_12}, {v_12, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_14}, {s_22->subset({1}), v_12}, {}, ""));
        co_yield m_context->copier()->copy(v_15, s_5->subset({1}), "");
        co_yield_(Instruction(
            "v_sub_co_u32_e32", {v_13}, {m_context->getVCC(), s_22->subset({0}), v_13}, {}, ""));
        co_yield_(
            Instruction("v_subb_co_u32", {v_14}, {s_0, v_14, v_15, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_subrev_co_u32", {v_15}, {s_0, s_5->subset({0}), v_13}, {}, ""));
        co_yield_(Instruction(
            "v_subbrev_co_u32", {v_14}, {s_0, Register::Value::Literal(0), v_14, s_0}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32", {s_0}, {s_5->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_16},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_0},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_le_u32", {s_0}, {s_5->subset({0}), v_15}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_15},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_0},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_eq_u32", {s_0}, {s_5->subset({1}), v_14}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_14}, {v_16, v_15, s_0}, {}, ""));
        co_yield m_context->copier()->copy(v_16, s_22->subset({1}), "");
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {v_12},
                              {m_context->getVCC(), v_16, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_cmp_le_u32_e32", {m_context->getVCC()}, {s_5->subset({1}), v_12}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32", {s_0}, {Register::Value::Literal(0), v_14}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_16},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction(
            "v_cmp_le_u32_e32", {m_context->getVCC()}, {s_5->subset({0}), v_13}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_14},
                              {Register::Value::Literal(1), Register::Value::Literal(2), s_0},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_13},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction(
            "v_cmp_eq_u32_e32", {m_context->getVCC()}, {s_5->subset({1}), v_12}, {}, ""));
        co_yield_(Instruction("v_add_co_u32", {v_14}, {s_0, v_7, v_14}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_12}, {v_16, v_13, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction(
            "v_addc_co_u32", {v_15}, {s_0, Register::Value::Literal(0), v_8, s_0}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32_e32",
                              {m_context->getVCC()},
                              {Register::Value::Literal(0), v_12},
                              {},
                              ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_7}, {v_7, v_14, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("s_xor_b64", {s_0}, {s_23, s_6}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_8}, {v_8, v_15, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_7}, {s_0->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_8}, {s_0->subset({1}), v_8}, {}, ""));
        co_yield m_context->copier()->copy(v_12, s_0->subset({1}), "");
        co_yield_(Instruction(
            "v_subrev_co_u32_e32", {v_7}, {m_context->getVCC(), s_0->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_8, v_12, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("s_cbranch_execz", {}, {label_25}, {}, ""));
        co_yield m_context->brancher()->branch(label_26);
        co_yield_(Instruction::Label(label_24));
        co_yield_(Instruction::Label(label_25));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_7}, {r0}, {}, ""));
        co_yield_(Instruction(
            "s_sub_i32", {s_0->subset({0})}, {Register::Value::Literal(0), r0}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_7}, {Register::Value::Literal(0x4f7ffffe), v_7}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_0->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_8}, {v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_7}, {v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_7}, {l0, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_12}, {v_7, r0}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_12}, {l0, v_12}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_8}, {Register::Value::Literal(1), v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {r0, v_12}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_7}, {v_7, v_8, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_subrev_u32_e32", {v_8}, {r0, v_12}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_8}, {v_12, v_8, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_14}, {Register::Value::Literal(1), v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {r0, v_8}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_7}, {v_7, v_14, m_context->getVCC()}, {}, ""));
        co_yield m_context->copier()->copy(v_8, Register::Value::Literal(0), "");
        co_yield_(Instruction::Label(label_26));
        co_yield Register::AllocateIfNeeded(dest);
        co_yield_(Instruction("v_readlane_b32",
                              {dest->subset({0})},
                              {v_7, Register::Value::Literal(0)},
                              {},
                              "Move value"));
        co_yield_(Instruction("v_readlane_b32",
                              {dest->subset({1})},
                              {v_8, Register::Value::Literal(0)},
                              {},
                              "Move value"));
    }

    Generator<Instruction> Arithmetic_Scalar_Int64::mod(std::shared_ptr<Register::Value> dest,
                                                        std::shared_ptr<Register::Value> lhs,
                                                        std::shared_ptr<Register::Value> rhs)
    {
        // Generated using the assembly_to_instructions.py script with the following HIP code:
        //  extern "C"
        //  __global__ void hello_world(long int * ptr, long int value1, long int value2)
        //  {
        //    *ptr = value1 % value2 ;
        //  }
        //
        // Generated code was modified to use the provided dest, lhs and rhs registers and
        // to save the result in the dest register instead of memory.
        co_yield describeOpArgs(__func__, "dest", dest, "lhs", lhs, "rhs", rhs);
        Register::ValuePtr l0, l1, r0, r1;
        co_yield get2Dwords(l0, l1, lhs);
        co_yield get2Dwords(r0, r1, rhs);

        auto s_0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        // co_yield s_0->allocate();
        auto s_5 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        // co_yield s_5->allocate();
        auto s_6 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        co_yield s_6->allocate();
        auto v_7 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_8 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_9 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_10 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_13 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_14 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_15 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_16 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_17 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_18 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_19 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_20 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto v_21 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        auto s_22 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        // co_yield s_22->allocate();
        auto s_23 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 2);
        co_yield s_23->allocate();

        auto label_24 = m_context->labelAllocator()->label("BB0_2");
        auto label_25 = m_context->labelAllocator()->label("BB0_3");
        auto label_26 = m_context->labelAllocator()->label("BB0_4");

        co_yield_(Instruction("s_or_b64", {s_0}, {lhs, rhs}, {}, ""));
        co_yield m_context->copier()->copy(s_0->subset({0}), Register::Value::Literal(0), "");
        co_yield_(Instruction("s_cmp_lg_u64", {s_0}, {Register::Value::Literal(0)}, {}, ""));
        co_yield m_context->brancher()->branchIfZero(label_24, m_context->getSCC());
        co_yield_(Instruction(
            "s_ashr_i32", {s_0->subset({0})}, {r1, Register::Value::Literal(31)}, {}, ""));
        co_yield_(Instruction("s_add_u32", {s_6->subset({0})}, {r0, s_0->subset({0})}, {}, ""));
        co_yield m_context->copier()->copy(s_0->subset({1}), s_0->subset({0}), "");
        co_yield_(Instruction("s_addc_u32", {s_6->subset({1})}, {r1, s_0->subset({0})}, {}, ""));
        co_yield_(Instruction("s_xor_b64", {s_5}, {s_6, s_0}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_7}, {s_5->subset({0})}, {}, ""));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_8}, {s_5->subset({1})}, {}, ""));
        co_yield_(Instruction("s_sub_u32",
                              {s_6->subset({0})},
                              {Register::Value::Literal(0), s_5->subset({0})},
                              {},
                              ""));
        co_yield_(Instruction("s_subb_u32",
                              {s_6->subset({1})},
                              {Register::Value::Literal(0), s_5->subset({1})},
                              {},
                              ""));
        co_yield m_context->copier()->copy(v_9, Register::Value::Literal(0), "");
        co_yield_(Instruction(
            "v_mac_f32_e32", {v_7}, {Register::Value::Literal(0x4f800000), v_8}, {}, ""));
        co_yield_(Instruction("v_rcp_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield m_context->copier()->copy(v_10, Register::Value::Literal(0), "");
        co_yield_(Instruction(
            "s_ashr_i32", {s_23->subset({0})}, {l1, Register::Value::Literal(31)}, {}, ""));
        co_yield m_context->copier()->copy(s_23->subset({1}), s_23->subset({0}), "");
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_7}, {Register::Value::Literal(0x5f7ffffc), v_7}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_8}, {Register::Value::Literal(0x2f800000), v_7}, {}, ""));
        co_yield_(Instruction("v_trunc_f32_e32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction(
            "v_mac_f32_e32", {v_7}, {Register::Value::Literal(0xcf800000), v_8}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_8}, {v_8}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {s_6->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {s_6->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {s_6->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_13}, {v_14, v_13}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_13}, {v_13, v_15}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {s_6->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {v_7, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_7, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_15}, {v_7, v_13}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_14}, {m_context->getVCC(), v_17, v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_9, v_15, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_18}, {v_8, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {v_8, v_16}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_14}, {m_context->getVCC(), v_14, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_8, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_15, v_18, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_17, v_10, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_8, v_13}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_15, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_15},
                              {m_context->getVCC(), v_9, v_14, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_co_u32", {v_7}, {s_0, v_7, v_13}, {}, ""));
        co_yield_(
            Instruction("v_addc_co_u32", {v_13}, {m_context->getVCC(), v_8, v_15, s_0}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_14}, {s_6->subset({0}), v_13}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {s_6->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_14}, {v_16, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_16}, {s_6->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_14}, {v_14, v_16}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_17}, {s_6->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_18}, {v_13, v_17}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_19}, {v_13, v_17}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_20}, {v_7, v_14}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_17}, {v_7, v_17}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_21}, {v_7, v_14}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_17}, {m_context->getVCC(), v_17, v_20}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_21},
                              {m_context->getVCC(), v_9, v_21, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_17}, {m_context->getVCC(), v_17, v_19}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {v_13, v_14}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_17},
                              {m_context->getVCC(), v_21, v_18, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_16},
                              {m_context->getVCC(), v_16, v_10, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {v_13, v_14}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_13}, {m_context->getVCC(), v_17, v_13}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_14},
                              {m_context->getVCC(), v_9, v_16, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_add_u32_e32", {v_8}, {v_8, v_15}, {}, ""));
        co_yield_(
            Instruction("v_addc_co_u32", {v_8}, {m_context->getVCC(), v_8, v_14, s_0}, {}, ""));
        co_yield_(Instruction("s_add_u32", {s_0->subset({0})}, {l0, s_23->subset({0})}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_7, v_13}, {}, ""));
        co_yield_(Instruction("s_addc_u32", {s_0->subset({1})}, {l1, s_23->subset({0})}, {}, ""));
        co_yield_(Instruction(
            "v_addc_co_u32_e32",
            {v_8},
            {m_context->getVCC(), Register::Value::Literal(0), v_8, m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction("s_xor_b64", {s_22}, {s_0, s_23}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_15}, {s_22->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {s_22->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_22->subset({0}), v_8}, {}, ""));
        co_yield_(
            Instruction("v_add_co_u32_e32", {v_15}, {m_context->getVCC(), v_14, v_15}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_13},
                              {m_context->getVCC(), v_9, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_16}, {s_22->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_7}, {s_22->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_15, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_14}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_7},
                              {m_context->getVCC(), v_13, v_16, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_13},
                              {m_context->getVCC(), v_14, v_10, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield_(Instruction("v_add_co_u32_e32", {v_7}, {m_context->getVCC(), v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_addc_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_9, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_5->subset({0}), v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_13}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_8}, {v_13, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_13}, {s_5->subset({1}), v_7}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_8}, {v_8, v_13}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_7}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_13}, {s_22->subset({1}), v_8}, {}, ""));
        co_yield m_context->copier()->copy(v_15, s_5->subset({1}), "");
        co_yield_(Instruction(
            "v_sub_co_u32_e32", {v_7}, {m_context->getVCC(), s_22->subset({0}), v_7}, {}, ""));
        co_yield_(
            Instruction("v_subb_co_u32", {v_13}, {s_0, v_13, v_15, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_subrev_co_u32", {v_14}, {s_0, s_5->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction(
            "v_subbrev_co_u32", {v_16}, {s_6, Register::Value::Literal(0), v_13, s_0}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32", {s_6}, {s_5->subset({1}), v_16}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_17},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_6},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_le_u32", {s_6}, {s_5->subset({0}), v_14}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32", {v_13}, {s_0, v_13, v_15, s_0}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32",
                              {v_10},
                              {Register::Value::Literal(0), Register::Value::Literal(-1), s_6},
                              {},
                              ""));
        co_yield_(Instruction("v_cmp_eq_u32", {s_6}, {s_5->subset({1}), v_16}, {}, ""));
        co_yield_(Instruction("v_subrev_co_u32", {v_15}, {s_0, s_5->subset({0}), v_14}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_17}, {v_17, v_10, s_6}, {}, ""));
        co_yield_(Instruction(
            "v_subbrev_co_u32", {v_13}, {s_0, Register::Value::Literal(0), v_13, s_0}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32", {s_0}, {Register::Value::Literal(0), v_17}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_15}, {v_14, v_15, s_0}, {}, ""));
        co_yield m_context->copier()->copy(v_14, s_22->subset({1}), "");
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_14, v_8, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction(
            "v_cmp_le_u32_e32", {m_context->getVCC()}, {s_5->subset({1}), v_8}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_14},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction(
            "v_cmp_le_u32_e32", {m_context->getVCC()}, {s_5->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32", {v_13}, {v_16, v_13, s_0}, {}, ""));
        co_yield_(Instruction(
            "v_cndmask_b32",
            {v_16},
            {Register::Value::Literal(0), Register::Value::Literal(-1), m_context->getVCC()},
            {},
            ""));
        co_yield_(Instruction(
            "v_cmp_eq_u32_e32", {m_context->getVCC()}, {s_5->subset({1}), v_8}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_14}, {v_14, v_16, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_cmp_ne_u32_e32",
                              {m_context->getVCC()},
                              {Register::Value::Literal(0), v_14},
                              {},
                              ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_7}, {v_7, v_15, m_context->getVCC()}, {}, ""));
        co_yield_(
            Instruction("v_cndmask_b32_e32", {v_8}, {v_8, v_13, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_7}, {s_23->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_xor_b32_e32", {v_8}, {s_23->subset({0}), v_8}, {}, ""));
        co_yield m_context->copier()->copy(v_13, s_23->subset({0}), "");
        co_yield_(Instruction(
            "v_subrev_co_u32_e32", {v_7}, {m_context->getVCC(), s_23->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_subb_co_u32_e32",
                              {v_8},
                              {m_context->getVCC(), v_8, v_13, m_context->getVCC()},
                              {},
                              ""));
        co_yield_(Instruction("s_cbranch_execz", {}, {label_25}, {}, ""));
        co_yield m_context->brancher()->branch(label_26);
        co_yield_(Instruction::Label(label_24));
        co_yield_(Instruction::Label(label_25));
        co_yield_(Instruction("v_cvt_f32_u32_e32", {v_7}, {r0}, {}, ""));
        co_yield_(Instruction(
            "s_sub_i32", {s_0->subset({0})}, {Register::Value::Literal(0), r0}, {}, ""));
        co_yield_(Instruction("v_rcp_iflag_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction(
            "v_mul_f32_e32", {v_7}, {Register::Value::Literal(0x4f7ffffe), v_7}, {}, ""));
        co_yield_(Instruction("v_cvt_u32_f32_e32", {v_7}, {v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_8}, {s_0->subset({0}), v_7}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_8}, {v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_add_u32_e32", {v_7}, {v_7, v_8}, {}, ""));
        co_yield_(Instruction("v_mul_hi_u32", {v_7}, {l0, v_7}, {}, ""));
        co_yield_(Instruction("v_mul_lo_u32", {v_7}, {v_7, r0}, {}, ""));
        co_yield_(Instruction("v_sub_u32_e32", {v_7}, {l0, v_7}, {}, ""));
        co_yield_(Instruction("v_subrev_u32_e32", {v_8}, {r0, v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {r0, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_7}, {v_7, v_8, m_context->getVCC()}, {}, ""));
        co_yield_(Instruction("v_subrev_u32_e32", {v_8}, {r0, v_7}, {}, ""));
        co_yield_(Instruction("v_cmp_le_u32_e32", {m_context->getVCC()}, {r0, v_7}, {}, ""));
        co_yield_(Instruction("v_cndmask_b32_e32", {v_7}, {v_7, v_8, m_context->getVCC()}, {}, ""));
        co_yield m_context->copier()->copy(v_8, Register::Value::Literal(0), "");
        co_yield_(Instruction::Label(label_26));

        co_yield Register::AllocateIfNeeded(dest);
        co_yield_(Instruction("v_readlane_b32",
                              {dest->subset({0})},
                              {v_7, Register::Value::Literal(0)},
                              {},
                              "Move value"));
        co_yield_(Instruction("v_readlane_b32",
                              {dest->subset({1})},
                              {v_8, Register::Value::Literal(0)},
                              {},
                              "Move value"));
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
        Arithmetic_Scalar_Int64::shiftR(std::shared_ptr<Register::Value> dest,
                                        std::shared_ptr<Register::Value> value,
                                        std::shared_ptr<Register::Value> shiftAmount)
    {
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        co_yield_(Instruction("s_ashr_i64", {dest}, {value, shiftAmount->subset({0})}, {}, ""));
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

    Generator<Instruction>
        Arithmetic_Scalar_Int64::bitwiseAnd(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_and_b64", {dest}, {lhs, rhs}, {}, ""));
    }

    Generator<Instruction>
        Arithmetic_Scalar_Int64::bitwiseXor(std::shared_ptr<Register::Value> dest,
                                            std::shared_ptr<Register::Value> lhs,
                                            std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        co_yield_(Instruction("s_xor_b64", {dest}, {lhs, rhs}, {}, ""));
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
