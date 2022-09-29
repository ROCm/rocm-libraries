
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace MemoryInstructionsTest
{

    class MemoryInstructionsTest : public GPUContextFixture
    {
    };

    class MemoryInstructionsExecuter : public CurrentGPUContextFixture
    {
    };

    void genBufDescTest(std::shared_ptr<rocRoller::Context> m_context)
    {
        auto k = m_context->kernel();

        k->setKernelName("BufferDescriptorTest");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result;
            co_yield m_context->argLoader()->getValue("result", s_result);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 4);

            co_yield v_a->allocate();
            co_yield v_result->allocate();
            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            auto bufDesc = rocRoller::BufferDescriptor(m_context);

            co_yield bufDesc.setup();
            co_yield bufDesc.setBasePointer(Register::Value::Literal(0x00000000));
            co_yield bufDesc.setSize(Register::Value::Literal(0x00000001));
            co_yield bufDesc.setOptions(Register::Value::Literal(131072)); //0x00020000
            co_yield bufDesc.incrementBasePointer(Register::Value::Literal(0x00000001));

            auto sRD = bufDesc.allRegisters();
            co_yield m_context->copier()->copy(v_a, sRD, "Move Value");
            co_yield m_context->mem()->storeFlat(v_result, v_a, "0", 16);

            auto bPnS = bufDesc.basePointerAndStride();
            co_yield m_context->copier()->copy(v_a->subset({0, 1}), bPnS, "Move Value");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_a->subset({0, 1}),
                                             Register::Value::Literal(16),
                                             8);

            auto size = bufDesc.size();
            co_yield m_context->copier()->copy(v_a->subset({2}), size, "Move Value");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_a->subset({2}),
                                             Register::Value::Literal(24),
                                             4);

            auto dOpt = bufDesc.descriptorOptions();
            co_yield m_context->copier()->copy(v_a->subset({3}), dOpt, "Move Value");
            co_yield m_context->mem()->store(MemoryInstructions::Flat,
                                             v_result,
                                             v_a->subset({3}),
                                             Register::Value::Literal(28),
                                             4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void genFlatTest(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        auto k = m_context->kernel();

        k->setKernelName("FlatTest");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument(
            {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, N > 4 ? N / 4 : 1);

            co_yield v_a->allocate();
            co_yield v_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(v_ptr, s_a, "Move pointer.");

            co_yield m_context->mem()->loadFlat(v_a, v_ptr, "0", N);
            co_yield m_context->mem()->storeFlat(v_result, v_a, "0", N);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeFlatTest(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        genFlatTest(m_context, N);

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        std::vector<char> a(N);
        for(int i = 0; i < N; i++)
            a[i] = i + 10;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<char>(N);

        KernelArguments kargs;
        kargs.append<void*>("result", d_result.get());
        kargs.append<void*>("a", d_a.get());
        KernelInvocation invocation;

        executableKernel->executeKernel(kargs, invocation);

        std::vector<char> result(N);
        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(char) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            EXPECT_EQ(result[i], a[i]);
    }

    void assembleFlatTest(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        genFlatTest(m_context, N);

        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);
    }

    void assembleBufDescTest(std::shared_ptr<rocRoller::Context> m_context)
    {
        genBufDescTest(m_context);

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU buffer tests for " << GPUContextFixture::GetParam();
        }

        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteFlatTest1Byte)
    {
        executeFlatTest(m_context, 1);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteFlatTest2Bytes)
    {
        executeFlatTest(m_context, 2);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteFlatTest4Bytes)
    {
        executeFlatTest(m_context, 4);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteFlatTest8Bytes)
    {
        executeFlatTest(m_context, 8);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteFlatTest12Bytes)
    {
        executeFlatTest(m_context, 12);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteFlatTest16Bytes)
    {
        executeFlatTest(m_context, 16);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_ExecuteBufDescriptor)
    {
        genBufDescTest(m_context);

        if(m_context->targetArchitecture().target().getMajorVersion() != 9)
        {
            GTEST_SKIP() << "Skipping GPU buffer tests for " << GPUContextFixture::GetParam();
        }

        if(isLocalDevice())
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            auto d_result = make_shared_device<unsigned int>(8); //Srd twice

            KernelArguments kargs;
            kargs.append<void*>("result", d_result.get());
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<unsigned int> result(8);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), sizeof(unsigned int) * 8, hipMemcpyDefault),
                HasHipSuccess(0));

            EXPECT_EQ(result[0], 0x00000001);
            EXPECT_EQ(result[1], 0x00000000);
            EXPECT_EQ(result[2], 0x00000001);
            EXPECT_EQ(result[3], 131072);
            EXPECT_EQ(result[4], 0x00000001);
            EXPECT_EQ(result[5], 0x00000000);
            EXPECT_EQ(result[6], 0x00000001);
            EXPECT_EQ(result[7], 131072);
        }
    }

    TEST_P(MemoryInstructionsTest, AssembleBufDescriptor)
    {
        assembleBufDescTest(m_context);
    }

    TEST_P(MemoryInstructionsTest, AssembleFlatTest1Byte)
    {
        assembleFlatTest(m_context, 1);
    }

    TEST_P(MemoryInstructionsTest, AssembleFlatTest2Bytes)
    {
        assembleFlatTest(m_context, 2);
    }

    TEST_P(MemoryInstructionsTest, AssembleFlatTest4Bytes)
    {
        assembleFlatTest(m_context, 4);
    }

    TEST_P(MemoryInstructionsTest, AssembleFlatTest8Bytes)
    {
        assembleFlatTest(m_context, 8);
    }

    TEST_P(MemoryInstructionsTest, AssembleFlatTest12Bytes)
    {
        assembleFlatTest(m_context, 12);
    }

    TEST_P(MemoryInstructionsTest, AssembleFlatTest16Bytes)
    {
        assembleFlatTest(m_context, 16);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_FlatTestOffset)
    {
        auto k = m_context->kernel();

        k->setKernelName("FlatTestOffset");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument(
            {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_offset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int64, 1);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_a->allocate();
            co_yield v_ptr->allocate();
            co_yield v_offset->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(
                v_offset->subset({0}), Register::Value::Literal(4), "Set offset value");
            co_yield m_context->copier()->copy(
                v_offset->subset({1}), Register::Value::Literal(0), "Set offset value");

            co_yield m_context->copier()->copy(v_ptr, s_a, "Move pointer.");

            co_yield m_context->mem()->load(MemoryInstructions::Flat, v_a, v_ptr, v_offset, 4);
            co_yield m_context->mem()->store(MemoryInstructions::Flat, v_result, v_a, v_offset, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        std::vector<int> a(2);
        a[0] = 0;
        a[1] = 123;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<int>(2);

        KernelArguments kargs;
        kargs.append("result", d_result.get());
        kargs.append("a", d_a.get());
        KernelInvocation invocation;

        executableKernel->executeKernel(kargs, invocation);

        std::vector<int> result(2);
        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(int) * 2, hipMemcpyDefault),
                    HasHipSuccess(0));

        EXPECT_EQ(result[1], a[1]);
    }

    void genLDSTest(std::shared_ptr<rocRoller::Context> m_context)
    {
        auto k = m_context->kernel();

        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        k->setKernelDimensions(1);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);
            auto workitemIndex = k->workitemIndex();

            auto lds1 = Register::Value::AllocateLDS(m_context, DataType::Int32, 2);

            auto lds2 = Register::Value::AllocateLDS(m_context, DataType::Int32, 9);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 4);

            auto lds1_offset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto lds2_offset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto twenty = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_a->allocate();

            co_yield m_context->copier()->copy(v_result, s_result);
            co_yield m_context->copier()->copy(v_ptr, s_a);

            // Get the LDS offset for each allocation
            co_yield m_context->copier()->copy(
                lds1_offset, Register::Value::Literal(lds1->getLDSAllocation()->offset()));
            co_yield m_context->copier()->copy(
                lds2_offset, Register::Value::Literal(lds2->getLDSAllocation()->offset()));
            co_yield m_context->copier()->copy(twenty, Register::Value::Literal(20));

            // Load 8 bytes into LDS1
            co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, "0", 1);
            co_yield m_context->mem()->storeLocal(lds1_offset, v_a->subset({0}), "0", 1);
            co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, "1", 1);
            co_yield m_context->mem()->storeLocal(lds1_offset, v_a->subset({0}), "1", 1);
            co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, "2", 2);
            co_yield m_context->mem()->storeLocal(
                lds1, v_a->subset({0}), "2", 2); // Use LDS1 value instead of offset register
            co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, "4", 4);
            co_yield m_context->mem()->storeLocal(lds1_offset, v_a->subset({0}), "4", 4);

            // Load 36 bytes into LDS2
            co_yield m_context->mem()->loadFlat(v_a->subset({0, 1}), v_ptr, "8", 8);
            co_yield m_context->mem()->storeLocal(lds2_offset, v_a->subset({0, 1}), "0", 8);
            co_yield m_context->mem()->loadFlat(v_a->subset({0, 1, 2}), v_ptr, "16", 12);
            co_yield m_context->mem()->store(MemoryInstructions::Local,
                                             lds2_offset,
                                             v_a->subset({0, 1, 2}),
                                             Register::Value::Literal(8),
                                             12);
            co_yield m_context->mem()->loadFlat(v_a, v_ptr, "28", 16);
            co_yield m_context->mem()->store(
                MemoryInstructions::Local, lds2_offset, v_a, twenty, 16);

            // Read 8 bytes from LDS1 and store to global data
            co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, "0", 1);
            co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), "0", 1);
            co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, "1", 1);
            co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), "1", 1);
            co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, "2", 2);
            co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), "2", 2);
            co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, "4", 4);
            co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), "4", 4);

            // Read 36 bytes from LDS2 and store to global data
            co_yield m_context->mem()->loadLocal(
                v_a->subset({0, 1}), lds2, "0", 8); // Use LDS2 value instead of offset register
            co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0, 1}), "8", 8);
            co_yield m_context->mem()->load(MemoryInstructions::Local,
                                            v_a->subset({0, 1, 2}),
                                            lds2_offset,
                                            Register::Value::Literal(8),
                                            12);
            co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0, 1, 2}), "16", 12);
            co_yield m_context->mem()->load(
                MemoryInstructions::Local, v_a, lds2_offset, twenty, 16);
            co_yield m_context->mem()->storeFlat(v_result, v_a, "28", 16);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeLDSTest(std::shared_ptr<rocRoller::Context> m_context)
    {
        genLDSTest(m_context);
        int N = 44;

        std::vector<char> a(N);
        for(int i = 0; i < N; i++)
            a[i] = i + 10;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<char>(N);

        KernelArguments kargs;
        kargs.append<void*>("result", d_result.get());
        kargs.append<void*>("a", d_a.get());
        CommandKernel commandKernel(m_context);

        commandKernel.launchKernel(kargs.runtimeArguments());

        std::vector<char> result(N);
        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(char) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            EXPECT_EQ(result[i], a[i]);
    }

    TEST_F(MemoryInstructionsExecuter, GPU_LDSTest)
    {
        executeLDSTest(m_context);
    }

    void genLDSBarrierTest(std::shared_ptr<rocRoller::Context> m_context,
                           unsigned int                        workItemCount)
    {
        auto k       = m_context->kernel();
        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::PointerGlobal}));

        auto workItemCountExpr = std::make_shared<Expression::Expression>(workItemCount);
        auto one               = std::make_shared<Expression::Expression>(1u);
        auto zero              = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Int32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});

        k->setWorkgroupSize({workItemCount, 1, 1});
        k->setWorkitemCount({workItemCountExpr, one, one});
        k->setDynamicSharedMemBytes(zero);

        k->setKernelDimensions(1);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result;
            co_yield m_context->argLoader()->getValue("result", s_result);
            auto workitemIndex = k->workitemIndex();

            auto lds3 = Register::Value::AllocateLDS(m_context, DataType::Int32, workItemCount);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto lds3_offset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto lds3_current = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            auto literal = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_a->allocate();

            co_yield m_context->copier()->copy(v_result, s_result);

            // Get the LDS offset for each allocation
            co_yield m_context->copier()->copy(
                lds3_offset, Register::Value::Literal(lds3->getLDSAllocation()->offset()));
            co_yield m_context->copier()->copy(literal,
                                               Register::Value::Literal(workItemCount - 1));

            auto arith
                = Component::Get<Arithmetic>(m_context, Register::Type::Vector, DataType::Int32);

            // Load 5 + workitemIndex.x into lds3[workitemIndex.x]
            co_yield generateOp<Expression::Add>(lds3_current, lds3_offset, workitemIndex[0]);
            co_yield generateOp<Expression::Add>(
                v_a, workitemIndex[0], Register::Value::Literal(5));
            co_yield generateOp<Expression::ShiftL>(
                lds3_current, lds3_current, Register::Value::Literal(2));
            co_yield m_context->mem()->storeLocal(lds3_current, v_a, "0", 4);

            co_yield m_context->mem()->barrier();

            // Store the contents of lds3[workitemIndex.x + 1 % workItemCount] into v_result[workitemIndex.x]
            co_yield generateOp<Expression::Add>(
                lds3_current, workitemIndex[0], Register::Value::Literal(1));
            co_yield m_context->copier()->copy(literal,
                                               Register::Value::Literal(workItemCount - 1));
            co_yield generateOp<Expression::BitwiseAnd>(lds3_current, lds3_current, literal);
            co_yield generateOp<Expression::Add>(lds3_current, lds3_offset, lds3_current);
            co_yield generateOp<Expression::ShiftL>(
                lds3_current, lds3_current, Register::Value::Literal(2));
            co_yield m_context->mem()->loadLocal(v_a, lds3_current, "0", 4);
            co_yield generateOp<Expression::ShiftL>(
                lds3_current, workitemIndex[0], Register::Value::Literal(2));
            co_yield generateOp<Expression::Add>(
                v_result->subset({0}), v_result->subset({0}), lds3_current);
            co_yield m_context->mem()->storeFlat(v_result, v_a, "0", 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeLDSBarrierTest(std::shared_ptr<rocRoller::Context> m_context)
    {
        const unsigned int N = 1024;

        genLDSBarrierTest(m_context, N);

        auto d_result = make_shared_device<int>(N);

        KernelArguments kargs;
        kargs.append<void*>("result", d_result.get());
        CommandKernel commandKernel(m_context);

        commandKernel.launchKernel(kargs.runtimeArguments());

        std::vector<int> result(N);
        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(int) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(unsigned int i = 0; i < N; i++)
        {
            EXPECT_EQ(result[i], 5 + ((i + 1) % N)) << i;
        }
    }

    TEST_F(MemoryInstructionsExecuter, GPU_LDSBarrierTest)
    {
        executeLDSBarrierTest(m_context);
    }

    INSTANTIATE_TEST_SUITE_P(
        MemoryInstructionsTests,
        MemoryInstructionsTest,
        ::testing::ValuesIn(rocRoller::GPUArchitectureLibrary::getAllSupportedISAs()));

}
