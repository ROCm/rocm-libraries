
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace CopyGeneratorTest
{
    class GPU_CopyGeneratorTest : public CurrentGPUContextFixture
    {
    };

    class CopyGeneratorTest : public GenericContextFixture
    {
        std::string targetArchitecture()
        {
            return "gfx90a";
        }
    };

    // Test if correct instructions are generated
    TEST_F(CopyGeneratorTest, Instruction)
    {
        auto vr = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        vr->allocateNow();
        auto ar = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int32, 1);
        ar->allocateNow();
        auto sr = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        auto dsr = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int64, 1);
        sr->allocateNow();
        auto ir      = Register::Value::Label("LabelRegister");
        auto literal = Register::Value::Literal(20);
        auto zero    = Register::Value::Literal(0L);

        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(sr, vr)); }, FatalError);
        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(ir, vr)); }, FatalError);
        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(literal, vr)); }, FatalError);

        m_context->schedule(m_context->copier()->copy(sr, sr));
        m_context->schedule(m_context->copier()->copy(sr, literal));
        m_context->schedule(m_context->copier()->copy(vr, vr));
        m_context->schedule(m_context->copier()->copy(ar, ar));
        m_context->schedule(m_context->copier()->copy(vr, ar));
        m_context->schedule(m_context->copier()->copy(vr, sr));
        m_context->schedule(m_context->copier()->copy(vr, literal));
        m_context->schedule(m_context->copier()->copy(ar, vr));
        m_context->schedule(m_context->copier()->copy(sr, zero));
        m_context->schedule(m_context->copier()->copy(dsr, zero));

        m_context->schedule(m_context->copier()->conditionalCopy(sr, sr));
        m_context->schedule(m_context->copier()->conditionalCopy(sr, literal));

        std::string expectedOutput = R"(
            s_mov_b32 s0, s0
            s_mov_b32 s0, 20
            v_mov_b32 v0, v0
            v_accvgpr_mov_b32 a0, a0
            v_accvgpr_read v0, a0
            v_mov_b32 v0, s0
            v_mov_b32 v0, 20
            v_accvgpr_write a0, v0
            s_mov_b32 s0, 0
            s_mov_b64 s[2:3], 0
            s_cmov_b32 s0, s0
            s_cmov_b32 s0, 20
            )";

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGeneratorTest, IteratedCopy)
    {
        int  n   = 8;
        auto vr0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, n);
        vr0->allocateNow();
        auto vr1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, n);
        vr1->allocateNow();

        auto ar0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int64, n);
        ar0->allocateNow();
        auto ar1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int64, n);
        ar1->allocateNow();

        m_context->schedule(m_context->copier()->copy(vr1, vr0));

        std::string expectedOutput = "";

        for(int i = 0; i < n; ++i)
        {
            expectedOutput
                += "v_mov_b32 v" + std::to_string(i + n) + ", v" + std::to_string(i) + "\n";
        }
        for(int i = 0; i < 2 * n; ++i)
        {
            expectedOutput += "v_accvgpr_mov_b32 a" + std::to_string(i + 2 * n) + ", a"
                              + std::to_string(i) + "\n";
        }

        m_context->schedule(m_context->copier()->copy(ar1, ar0));

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGeneratorTest, TestFillInt32)
    {
        int  n   = 8;
        auto sr0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, n);
        auto vr0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, n);
        auto ar0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int64, n);

        m_context->schedule(m_context->copier()->fill(sr0, Register::Value::Literal(11)));
        m_context->schedule(m_context->copier()->fill(vr0, Register::Value::Literal(12)));
        m_context->schedule(m_context->copier()->fill(ar0, Register::Value::Literal(13)));

        std::string expectedOutput = "";

        for(int i = 0; i < n; ++i)
        {
            expectedOutput += "s_mov_b32 s" + std::to_string(i) + ", 11\n";
        }

        for(int i = 0; i < n; ++i)
        {
            expectedOutput += "v_mov_b32 v" + std::to_string(i) + ", 12\n";
        }

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput += "v_accvgpr_write a[" + std::to_string(i) + ":" + std::to_string(i + 1)
                              + "], 13\n";
        }

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGeneratorTest, TestFillInt64)
    {
        int  n   = 8;
        auto sr0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int64, n);
        auto vr0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int64, n);
        auto ar0 = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int64, n);

        m_context->schedule(m_context->copier()->fill(sr0, Register::Value::Literal(11L)));
        m_context->schedule(
            m_context->copier()->fill(vr0, Register::Value::Literal((13L << 32) + 12)));
        m_context->schedule(m_context->copier()->fill(ar0, Register::Value::Literal(14L)));

        std::string expectedOutput = "";

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput
                += "s_mov_b64 s[" + std::to_string(i) + ":" + std::to_string(i + 1) + "], 11\n";
        }

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput += "v_mov_b32 v" + std::to_string(i) + ", 12\n";
            expectedOutput += "v_mov_b32 v" + std::to_string(i + 1) + ", 13\n";
        }

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput += "v_accvgpr_write a[" + std::to_string(i) + ":" + std::to_string(i + 1)
                              + "], 14\n";
        }

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGeneratorTest, EnsureType)
    {
        auto vr = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 4);
        vr->allocateNow();
        Register::ValuePtr dummy = nullptr;

        std::string expectedOutput = R"(
            v_accvgpr_write a0, v0
            v_accvgpr_write a1, v1
            v_accvgpr_write a2, v2
            v_accvgpr_write a3, v3
            )";

        m_context->schedule(
            m_context->copier()->ensureType(dummy, vr, Register::Type::Accumulator));

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expectedOutput));
        EXPECT_EQ(vr->regType(), Register::Type::Vector);
        EXPECT_EQ(dummy->regType(), Register::Type::Accumulator);
        EXPECT_EQ(dummy->valueCount(), vr->valueCount());
        EXPECT_EQ(dummy->variableType().dataType, vr->variableType().dataType);

        clearOutput();

        dummy = Register::Value::Label("not_an_int_vector");

        m_context->schedule(m_context->copier()->ensureType(dummy, vr, vr->regType()));

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(""));
        EXPECT_EQ(vr->regType(), Register::Type::Vector);
        EXPECT_EQ(dummy->regType(), Register::Type::Vector);
        EXPECT_EQ(dummy->valueCount(), vr->valueCount());
        EXPECT_EQ(dummy->variableType().dataType, vr->variableType().dataType);
    }

    TEST_F(CopyGeneratorTest, NoAllocation)
    {
        auto va = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 4);
        auto vb = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 4);

        EXPECT_ANY_THROW(m_context->schedule(m_context->copier()->copy(va, vb)));
    }

    TEST_F(GPU_CopyGeneratorTest, GPU_Test)
    {
        ASSERT_EQ(true, isLocalDevice());

        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto ptr_arg  = command->allocateArgument(floatPtr);
        auto val_arg  = command->allocateArgument(floatPtr);
        auto size_arg = command->allocateArgument(uintVal);

        auto ptr_exp  = std::make_shared<Expression::Expression>(ptr_arg);
        auto val_exp  = std::make_shared<Expression::Expression>(val_arg);
        auto size_exp = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"output",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        ptr_exp});
        k->addArgument({"input",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::ReadOnly,
                        val_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size_exp, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        Register::ValuePtr output_mem, input_mem;
        auto               v_value
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 4);
        auto v_addr
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Raw32, 2);
        auto v_tmp
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Float, 4);

        auto init = [&]() -> Generator<Instruction> {
            co_yield m_context->argLoader()->getValue("output", output_mem);
            co_yield m_context->argLoader()->getValue("input", input_mem);
        };

        auto copy = [&]() -> Generator<Instruction> {
            co_yield m_context->copier()->copy(v_addr, input_mem);
            co_yield m_context->mem()->loadFlat(v_value, v_addr, 0, 16);
            co_yield m_context->copier()->copy(v_tmp, v_value);
            co_yield m_context->copier()->copy(v_addr, output_mem);
            co_yield m_context->mem()->storeFlat(v_addr, v_tmp, 0, 16);
        };

        m_context->schedule(init());
        m_context->schedule(copy());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        const int size       = 4;
        auto      input_ptr  = make_shared_device<float>(size);
        auto      output_ptr = make_shared_device<float>(size);
        float     val[size]  = {2.0f, 3.0f, 5.0f, 7.0f};

        ASSERT_THAT(hipMemset(output_ptr.get(), 0, size * sizeof(float)), HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(input_ptr.get(), val, size * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        KernelArguments runtimeArgs;
        runtimeArgs.append("output", output_ptr.get());
        runtimeArgs.append("input", input_ptr.get());
        runtimeArgs.append("size", size);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        float resultValue[size]   = {0.0f, 0.0f, 0.0f, 0.0f};
        float expectedValue[size] = {2.0f, 3.0f, 5.0f, 7.0f};
        ASSERT_THAT(
            hipMemcpy(resultValue, output_ptr.get(), size * sizeof(float), hipMemcpyDefault),
            HasHipSuccess(0));

        EXPECT_EQ(std::memcmp(resultValue, expectedValue, size), 0);
    }
}
