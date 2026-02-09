/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
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

    class CopyGenerator90aTest : public GenericContextFixture
    {
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }
    };

    class CopyGenerator94xTest : public GenericContextFixture
    {
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX942};
        }
    };

    class CopyGenerator1200Test : public GenericContextFixture
    {
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX1200};
        }
    };

    static void testEnsureTypeCommutative(rocRoller::ContextPtr& context)
    {
        auto literal       = Register::Value::Literal(65536);
        auto small_literal = Register::Value::Literal(10);

        using Bitset = EnumBitset<Register::Type>;
        using namespace std::placeholders;
        auto ensureTypeCommutative = std::bind(&rocRoller::CopyGenerator::ensureTypeCommutative,
                                               context->copier().get(),
                                               _1,
                                               _2,
                                               _3,
                                               _4);

        // Throw if LHS is also a literal
        EXPECT_THROW(
            {
                context->schedule(
                    ensureTypeCommutative(Bitset{Register::Type::Vector, Register::Type::Literal},
                                          literal,
                                          Bitset{Register::Type::Vector},
                                          literal));
            },
            FatalError);

        // No swap as RHS can be Literal
        context->schedule(
            ensureTypeCommutative(Bitset{Register::Type::Vector, Register::Type::Literal},
                                  literal,
                                  Bitset{Register::Type::Literal},
                                  literal));

        // No swap as RHS can be Constant
        context->schedule(
            ensureTypeCommutative(Bitset{Register::Type::Vector, Register::Type::Literal},
                                  literal,
                                  Bitset{Register::Type::Constant},
                                  small_literal));

        // RHS (Literal) swapped with LHS
        {
            auto vgpr = std::make_shared<Register::Value>(
                context, Register::Type::Vector, DataType::Int32, 1);
            vgpr->allocateNow();

            auto literal_copy = literal;
            context->schedule(
                ensureTypeCommutative(Bitset{Register::Type::Vector, Register::Type::Literal},
                                      vgpr,
                                      Bitset{Register::Type::Vector},
                                      literal_copy));

            EXPECT_EQ(vgpr, literal);
        }

        // RHS (Constant) swapped with LHS
        {
            auto vgpr = std::make_shared<Register::Value>(
                context, Register::Type::Vector, DataType::Int32, 1);
            vgpr->allocateNow();

            auto literal_copy = small_literal;
            context->schedule(
                ensureTypeCommutative(Bitset{Register::Type::Vector, Register::Type::Constant},
                                      vgpr,
                                      Bitset{Register::Type::Vector},
                                      literal_copy));

            EXPECT_EQ(vgpr, small_literal);
        }

        // Move RHS to a new VGPR
        {
            auto vgpr = std::make_shared<Register::Value>(
                context, Register::Type::Vector, DataType::Int32, 1);
            vgpr->allocateNow();
            context->schedule(ensureTypeCommutative(
                Bitset{Register::Type::Vector}, vgpr, Bitset{Register::Type::Vector}, literal));
        }
    }

    TEST_F(CopyGenerator90aTest, ensureTypeCommutative)
    {
        testEnsureTypeCommutative(m_context);
        EXPECT_EQ(NormalizedSource("v_mov_b32 v1, 65536"), NormalizedSource(output()));
    }

    TEST_F(CopyGenerator94xTest, ensureTypeCommutative)
    {
        testEnsureTypeCommutative(m_context);
        EXPECT_EQ(NormalizedSource("v_mov_b32 v1, 65536"), NormalizedSource(output()));
    }

    TEST_F(CopyGenerator1200Test, ensureTypeCommutative)
    {
        testEnsureTypeCommutative(m_context);
        EXPECT_EQ(NormalizedSource("v_mov_b32 v1, 65536"), NormalizedSource(output()));
    }

    // Test if correct instructions are generated
    TEST_F(CopyGenerator90aTest, Instruction)
    {
        auto i32vr = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        i32vr->allocateNow();
        auto i64vr = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int64, 1);

        auto i32ar = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int32, 1);
        i32ar->allocateNow();
        auto i64ar = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Int64, 1);

        auto i32sr = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        i32sr->allocateNow();
        auto i64sr = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int64, 1);

        auto    ir       = Register::Value::Label("LabelRegister");
        auto    literal  = Register::Value::Literal(20);
        auto    izero    = Register::Value::Literal(0L);
        int32_t i32      = 2;
        int64_t i64      = 2;
        auto    i32two   = Register::Value::Literal(i32);
        auto    i64two   = Register::Value::Literal(i64);
        i32              = -2;
        i64              = -2;
        auto i32minustwo = Register::Value::Literal(i32);
        auto i64minustwo = Register::Value::Literal(i64);

        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(ir, i32vr)); }, FatalError);
        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(literal, i32vr)); },
                     FatalError);

        m_context->schedule(m_context->copier()->copy(i32sr, i32vr));

        m_context->schedule(m_context->copier()->copy(i32sr, i32sr));
        m_context->schedule(m_context->copier()->copy(i32sr, literal));
        m_context->schedule(m_context->copier()->copy(i32vr, i32vr));
        m_context->schedule(m_context->copier()->copy(i32ar, i32ar));
        m_context->schedule(m_context->copier()->copy(i32vr, i32ar));
        m_context->schedule(m_context->copier()->copy(i32vr, i32sr));
        m_context->schedule(m_context->copier()->copy(i32vr, literal));
        m_context->schedule(m_context->copier()->copy(i32ar, i32vr));

        m_context->schedule(m_context->copier()->copy(i32vr, izero));
        m_context->schedule(m_context->copier()->copy(i64vr, izero));
        m_context->schedule(m_context->copier()->copy(i64vr, i32two));
        m_context->schedule(m_context->copier()->copy(i64vr, i64two));
        m_context->schedule(m_context->copier()->copy(i64vr, i32minustwo));
        m_context->schedule(m_context->copier()->copy(i64vr, i64minustwo));

        m_context->schedule(m_context->copier()->copy(i32ar, izero));
        m_context->schedule(m_context->copier()->copy(i64ar, izero));

        m_context->schedule(m_context->copier()->copy(i32sr, izero));
        m_context->schedule(m_context->copier()->copy(i64sr, izero));

        m_context->schedule(m_context->copier()->conditionalCopy(i32sr, i32sr));
        m_context->schedule(m_context->copier()->conditionalCopy(i32sr, literal));

        m_context->schedule(m_context->copier()->copy(i32vr, m_context->getSCC()));

        std::string expectedOutput = R"(
            v_readfirstlane_b32 s0, v0
            // s_mov_b32 s0, s0         Omitted due to same registers
            s_mov_b32 s0, 20
            // v_mov_b32 v0, v0         Omitted due to same registers
            // v_accvgpr_mov_b32 a0, a0 Omitted due to same registers
            v_accvgpr_read v0, a0
            v_mov_b32 v0, s0
            v_mov_b32 v0, 20
            v_accvgpr_write a0, v0

            v_mov_b32 v0, 0
            v_mov_b32 v2, 0
            v_mov_b32 v3, 0
            v_mov_b32 v2, 2
            v_mov_b32 v3, 0
            v_mov_b32 v2, 2
            v_mov_b32 v3, 0
            // -2 == 0x 1111 1111 1111 1111 1111 1111 1111 1110
            v_mov_b32 v2, 4294967294
            v_mov_b32 v3, 4294967295
            // -2 == 0x 1111 1111 1111 1111 1111 1111 1111 1110
            v_mov_b32 v2, 4294967294
            v_mov_b32 v3, 4294967295

            v_accvgpr_write a0, 0
            v_accvgpr_write a1, 0
            v_accvgpr_write a2, 0

            s_mov_b32 s0, 0
            s_mov_b64 s[2:3], 0

            s_cmov_b32 s0, s0
            s_cmov_b32 s0, 20

            v_mov_b32 v0, scc
            )";

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGenerator90aTest, IteratedCopy)
    {
        int  n = 8;
        auto vr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Int32,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        vr0->allocateNow();
        auto vr1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Int32,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        vr1->allocateNow();

        auto ar0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        ar0->allocateNow();
        auto ar1
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
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

    TEST_F(CopyGenerator90aTest, TestFillInt32)
    {
        int  n = 8;
        auto sr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int32,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        auto vr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Int32,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        auto ar0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Int32,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());

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

        for(int i = 0; i < n; ++i)
        {
            expectedOutput += "v_accvgpr_write a" + std::to_string(i) + ", 13\n";
        }

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGenerator94xTest, TestFillInt64)
    {
        int  n = 8;
        auto sr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        auto vr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        auto ar0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());

        m_context->schedule(m_context->copier()->fill(sr0, Register::Value::Literal(11L)));
        m_context->schedule(m_context->copier()->fill(vr0, Register::Value::Literal(12L)));
        m_context->schedule(
            m_context->copier()->fill(ar0, Register::Value::Literal((15L << 32) + 14)));

        std::string expectedOutput = "";

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput
                += "s_mov_b64 s[" + std::to_string(i) + ":" + std::to_string(i + 1) + "], 11\n";
        }

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput
                += "v_mov_b64 v[" + std::to_string(i) + ":" + std::to_string(i + 1) + "], 12\n";
        }

        for(int i = 0; i < n * 2; i += 2)
        {
            expectedOutput += "v_accvgpr_write a" + std::to_string(i) + ", 14\n";
            expectedOutput += "v_accvgpr_write a" + std::to_string(i + 1) + ", 15\n";
        }

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGenerator90aTest, TestFillInt64)
    {
        int  n = 8;
        auto sr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Scalar,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        auto vr0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());
        auto ar0
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Int64,
                                                n,
                                                Register::AllocationOptions::FullyContiguous());

        m_context->schedule(m_context->copier()->fill(sr0, Register::Value::Literal(11L)));
        m_context->schedule(
            m_context->copier()->fill(vr0, Register::Value::Literal((13L << 32) + 12)));
        m_context->schedule(
            m_context->copier()->fill(ar0, Register::Value::Literal((15L << 32) + 14)));

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
            expectedOutput += "v_accvgpr_write a" + std::to_string(i) + ", 14\n";
            expectedOutput += "v_accvgpr_write a" + std::to_string(i + 1) + ", 15\n";
        }

        EXPECT_EQ(NormalizedSource(expectedOutput), NormalizedSource(output()));
    }

    TEST_F(CopyGenerator90aTest, EnsureType)
    {
        auto vr              = std::make_shared<Register::Value>(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Int32,
                                                    4,
                                                    Register::AllocationOptions::FullyContiguous());
        auto literalRegister = Register::Value::Literal(123);

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

        clearOutput();

        dummy = nullptr;

        m_context->schedule(m_context->copier()->ensureType(
            dummy, literalRegister, {Register::Type::Vector, Register::Type::Literal}));

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(""));
        EXPECT_EQ(dummy->regType(), Register::Type::Literal);
        EXPECT_EQ(dummy->valueCount(), literalRegister->valueCount());
        EXPECT_EQ(dummy->variableType().dataType, literalRegister->variableType().dataType);

        clearOutput();

        dummy = nullptr;

        m_context->schedule(m_context->copier()->ensureType(
            dummy, literalRegister, {Register::Type::Vector, Register::Type::Scalar}));

        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(R"(s_mov_b32 s0, 123)"));
        EXPECT_EQ(dummy->regType(), Register::Type::Scalar);
        EXPECT_EQ(dummy->valueCount(), 1);
        EXPECT_EQ(dummy->variableType().dataType, literalRegister->variableType().dataType);
    }

    TEST_F(CopyGenerator90aTest, NoAllocation)
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
        VariableType uintVal{DataType::UInt32, PointerType::Value};

        auto inputTag   = command->allocateTag();
        auto input_arg  = command->allocateArgument(floatPtr, inputTag, ArgumentType::Value);
        auto outputTag  = command->allocateTag();
        auto output_arg = command->allocateArgument(floatPtr, outputTag, ArgumentType::Value);
        auto sizeTag    = command->allocateTag();
        auto size_arg   = command->allocateArgument(uintVal, sizeTag, ArgumentType::Limit);

        auto input_exp  = std::make_shared<Expression::Expression>(input_arg);
        auto output_exp = std::make_shared<Expression::Expression>(output_arg);
        auto size_exp   = std::make_shared<Expression::Expression>(size_arg);

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        auto k = m_context->kernel();

        k->setKernelDimensions(1);

        k->addArgument({"input",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::ReadOnly,
                        input_exp});
        k->addArgument({"output",
                        {DataType::Float, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        output_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({size_exp, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        Register::ValuePtr output_mem, input_mem;
        auto               v_value = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::Float,
                                                    4,
                                                    Register::AllocationOptions::FullyContiguous());
        auto               v_addr  = Register::Value::Placeholder(
            m_context, Register::Type::Vector, {DataType::Float, PointerType::PointerGlobal}, 1);
        auto v_tmp = Register::Value::Placeholder(m_context,
                                                  Register::Type::Vector,
                                                  DataType::Float,
                                                  4,
                                                  Register::AllocationOptions::FullyContiguous());

        auto init = [&]() -> Generator<Instruction> {
            co_yield m_context->argLoader()->getValue("output", output_mem);
            co_yield m_context->argLoader()->getValue("input", input_mem);
        };

        auto copy = [&]() -> Generator<Instruction> {
            co_yield m_context->copier()->copy(v_addr, input_mem);
            co_yield m_context->mem()->loadGlobal(v_value, v_addr, 0, 16);
            co_yield m_context->copier()->copy(v_tmp, v_value);
            co_yield m_context->copier()->copy(v_addr, output_mem);
            co_yield m_context->mem()->storeGlobal(v_addr, v_tmp, 0, 16);
        };

        m_context->schedule(init());
        m_context->schedule(copy());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        const int size       = 4;
        auto      input_ptr  = make_shared_device<float>(size);
        auto      output_ptr = make_shared_device<float>(size);
        float     val[size]  = {2.0f, 3.0f, 5.0f, 7.0f};

        ASSERT_THAT(hipMemset(output_ptr.get(), 0, size * sizeof(float)), HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(input_ptr.get(), val, size * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();
        commandArgs.setArgument(outputTag, ArgumentType::Value, output_ptr.get());
        commandArgs.setArgument(inputTag, ArgumentType::Value, input_ptr.get());
        commandArgs.setArgument(sizeTag, ArgumentType::Limit, size);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        float resultValue[size]   = {0.0f, 0.0f, 0.0f, 0.0f};
        float expectedValue[size] = {2.0f, 3.0f, 5.0f, 7.0f};
        ASSERT_THAT(
            hipMemcpy(resultValue, output_ptr.get(), size * sizeof(float), hipMemcpyDefault),
            HasHipSuccess(0));

        EXPECT_EQ(std::memcmp(resultValue, expectedValue, size), 0);
    }

    TEST_F(CopyGenerator1200Test, NoAccVGPR)
    {
        auto literal = Register::Value::Literal(1);
        auto vgpr
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Vector,
                                                DataType::Int32,
                                                1,
                                                Register::AllocationOptions::FullyContiguous());
        auto accVGPR
            = std::make_shared<Register::Value>(m_context,
                                                Register::Type::Accumulator,
                                                DataType::Int32,
                                                1,
                                                Register::AllocationOptions::FullyContiguous());

        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(accVGPR, literal)); },
                     FatalError);
        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(accVGPR, vgpr)); },
                     FatalError);
        EXPECT_THROW({ m_context->schedule(m_context->copier()->copy(vgpr, accVGPR)); },
                     FatalError);
    }
}
