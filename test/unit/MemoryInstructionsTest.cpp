#include <memory>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace MemoryInstructionsTest
{

    struct MemoryInstructionsTest : public GPUContextFixture
    {
    };

    struct ScalarMemoryInstructionsTest : public GPUContextFixtureParam<int>
    {
        int numBytesParam()
        {
            return std::get<1>(GetParam());
        }

        void genScalarTest()
        {
            int  N = numBytesParam();
            auto k = m_context->kernel();

            k->setKernelName("ScalarTest");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a_ptr;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a_ptr);

                int                         size = (N % 4 == 0) ? N / 4 : N / 4 + 1;
                Register::AllocationOptions options;
                options.alignment            = size;
                options.contiguousChunkWidth = Register::FULLY_CONTIGUOUS;
                auto s_a                     = std::make_shared<Register::Value>(
                    m_context, Register::Type::Scalar, DataType::Int32, size, options);
                co_yield s_a->allocate();

                co_yield m_context->mem()->loadScalar(s_a, s_a_ptr, 0, N);
                co_yield m_context->mem()->storeScalar(s_result, s_a, 0, N);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

        void executeScalarTest()
        {
            genScalarTest();
            int N          = numBytesParam();
            int bufferSize = N + 20;

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            std::vector<char> a(bufferSize);
            for(int i = 0; i < N; i++)
                a[i] = i + 10;
            for(int i = N; i < bufferSize; i++)
                a[i] = -i;

            std::vector<char> initialResult(bufferSize);
            for(int i = 0; i < bufferSize; i++)
                initialResult[i] = 2 * i;

            auto d_a      = make_shared_device(a);
            auto d_result = make_shared_device<char>(initialResult);

            KernelArguments kargs;
            kargs.append<void*>("result", d_result.get());
            kargs.append<void*>("a", d_a.get());
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<char> result(bufferSize);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), sizeof(char) * bufferSize, hipMemcpyDefault),
                HasHipSuccess(0));

            for(int i = 0; i < N; i++)
                EXPECT_EQ(result[i], a[i]);
            for(int i = N; i < result.size(); i++)
                EXPECT_EQ(result[i], 2 * i);
        }

        void assembleScalarTest()
        {
            genScalarTest();

            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    };

    TEST_P(ScalarMemoryInstructionsTest, GPU_Basic)
    {
        if(!contains({4, 8, 16, 32, 64}, numBytesParam()))
        {
            EXPECT_THROW(genScalarTest(), FatalError);
            return;
        }
        else
        {
            if(isLocalDevice())
                executeScalarTest();
            else
                assembleScalarTest();
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        ScalarMemoryInstructionsTest,
        ScalarMemoryInstructionsTest,
        ::testing::Combine(::testing::Values(GPUArchitectureTarget{GPUArchitectureGFX::GFX90A},
                                             GPUArchitectureTarget{GPUArchitectureGFX::GFX908},
                                             GPUArchitectureTarget{GPUArchitectureGFX::GFX940},
                                             GPUArchitectureTarget{GPUArchitectureGFX::GFX941},
                                             GPUArchitectureTarget{GPUArchitectureGFX::GFX942}),
                           ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));

    struct FlatMemoryInstructionsTest : public GPUContextFixtureParam<int>
    {
        int numBytesParam()
        {
            return std::get<1>(GetParam());
        }

        void genFlatTest()
        {
            int  N = numBytesParam();
            auto k = m_context->kernel();

            k->setKernelName("FlatTest");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto v_ptr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Int32,
                                                   N > 4 ? N / 4 : 1,
                                                   Register::AllocationOptions::FullyContiguous());

                co_yield v_a->allocate();
                co_yield v_ptr->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

                co_yield m_context->copier()->copy(v_ptr, s_a, "Move pointer.");

                co_yield m_context->mem()->loadFlat(v_a, v_ptr, 0, N);
                co_yield m_context->mem()->storeFlat(v_result, v_a, 0, N);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

        void executeFlatTest()
        {
            genFlatTest();
            int N          = numBytesParam();
            int bufferSize = N + 20;

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            std::vector<char> a(bufferSize);
            for(int i = 0; i < N; i++)
                a[i] = i + 10;
            for(int i = N; i < bufferSize; i++)
                a[i] = -i;

            std::vector<char> initialResult(bufferSize);
            for(int i = 0; i < bufferSize; i++)
                initialResult[i] = 2 * i;

            auto d_a      = make_shared_device(a);
            auto d_result = make_shared_device<char>(initialResult);

            KernelArguments kargs;
            kargs.append<void*>("result", d_result.get());
            kargs.append<void*>("a", d_a.get());
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<char> result(bufferSize);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), sizeof(char) * bufferSize, hipMemcpyDefault),
                HasHipSuccess(0));

            for(int i = 0; i < N; i++)
                EXPECT_EQ(result[i], a[i]);
            for(int i = N; i < result.size(); i++)
                EXPECT_EQ(result[i], 2 * i);
        }

        void assembleFlatTest()
        {
            genFlatTest();

            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    };

    TEST_P(FlatMemoryInstructionsTest, GPU_Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasFlatOffset);

        if(isLocalDevice())
            executeFlatTest();
        else
            assembleFlatTest();
    }

    INSTANTIATE_TEST_SUITE_P(FlatMemoryInstructionsTest,
                             FlatMemoryInstructionsTest,
                             ::testing::Combine(supportedISAValues(),
                                                ::testing::Values(1, 2, 4, 8, 12, 16, 20, 44)));

    TEST_P(MemoryInstructionsTest, GPU_FlatTestOffset)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasFlatOffset);
        REQUIRE_NOT_ARCH_CAP(GPUCapability::HasExplicitNC);

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

            co_yield m_context->mem()->load(
                MemoryInstructions::Flat, v_a->subset({0}), v_ptr, v_offset, 4);
            co_yield m_context->mem()->store(MemoryInstructions::Flat, v_result, v_a, v_offset, 4);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(!isLocalDevice())
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
        {

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
    }

    TEST_P(MemoryInstructionsTest, GPU_BufferDescriptor)
    {
        auto generate = [&]() {
            auto k = m_context->kernel();

            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result;
                co_yield m_context->argLoader()->getValue("result", s_result);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::UInt32,
                                                   4,
                                                   Register::AllocationOptions::FullyContiguous());

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
                co_yield m_context->mem()->storeFlat(v_result, v_a, 0, 16);

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
        };

        generate();

        REQUIRE_ARCH_CAP(GPUCapability::HasFlatOffset);

        if(!isLocalDevice())
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
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

    INSTANTIATE_TEST_SUITE_P(MemoryInstructionsTests, MemoryInstructionsTest, CDNAISATuples());

    struct BufferMemoryInstructionsTest : public GPUContextFixtureParam<int>
    {
        int numBytesParam()
        {
            return std::get<1>(GetParam());
        }

        void genBufferTest()
        {
            int N = numBytesParam();

            auto k = m_context->kernel();

            k->setKernelName("BufferTest");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto vgprSerial = m_context->kernel()->workitemIndex()[0];

                int  size = (N % 4 == 0) ? N / 4 : N / 4 + 1;
                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Int32,
                                                   size,
                                                   Register::AllocationOptions::FullyContiguous());

                co_yield v_a->allocate();

                auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
                co_yield bufDesc->setup();
                co_yield bufDesc->setBasePointer(s_a);
                co_yield bufDesc->setSize(Register::Value::Literal(N));
                co_yield bufDesc->setOptions(Register::Value::Literal(131072)); //0x00020000

                auto bufInstOpts = rocRoller::BufferInstructionOptions();

                co_yield m_context->mem()->loadBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
                co_yield bufDesc->setBasePointer(s_result);
                co_yield m_context->mem()->storeBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_P(BufferMemoryInstructionsTest, GPU_Basic)
    {
        int N = numBytesParam();

        if(N % 4 == 3)
        {
            // TODO : add support for buffer loads/stores for odd number of bytes >= 3
            EXPECT_THROW(genBufferTest(), FatalError);
            GTEST_SKIP();
        }
        else
        {
            genBufferTest();
        }

        if(!isLocalDevice())
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
        {
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
            ASSERT_THAT(
                hipMemcpy(result.data(), d_result.get(), sizeof(char) * N, hipMemcpyDefault),
                HasHipSuccess(0));

            for(int i = 0; i < N; i++)
            {
                EXPECT_EQ(result[i], a[i]);
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(BufferMemoryInstructionsTest,
                             BufferMemoryInstructionsTest,
                             ::testing::Combine(mfmaSupportedISAValues(),
                                                ::testing::Values(1, 2, 3, 4, 8, 16, 20, 44, 47)));

    TEST_P(MemoryInstructionsTest, GPU_ExecuteBufDescriptor)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasFlatOffset);

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

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::Int32, PointerType::PointerGlobal},
                                               1);

            auto v_a = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::UInt32,
                                                    4,
                                                    Register::AllocationOptions::FullyContiguous());

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
            co_yield m_context->mem()->storeFlat(v_result, v_a, 0, 16);

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

        if(!isLocalDevice())
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
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

    struct MemoryInstructionsLDSTest : public CurrentGPUContextFixture
    {
        void genLDSTest()
        {
            auto k = m_context->kernel();

            auto command = std::make_shared<Command>();

            auto resultTag  = command->allocateTag();
            auto result_exp = std::make_shared<Expression::Expression>(command->allocateArgument(
                {DataType::Int32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));
            auto aTag       = command->allocateTag();
            auto a_exp      = std::make_shared<Expression::Expression>(command->allocateArgument(
                {DataType::Int32, PointerType::PointerGlobal}, aTag, ArgumentType::Value));

            auto one  = std::make_shared<Expression::Expression>(1u);
            auto zero = std::make_shared<Expression::Expression>(0u);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly,
                            result_exp});
            k->addArgument({"a",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::ReadOnly,
                            a_exp});

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

                auto lds3 = Register::Value::AllocateLDS(m_context, DataType::Int32, 11);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto v_ptr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Int32,
                                                   11,
                                                   Register::AllocationOptions::FullyContiguous());

                auto lds1_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                auto lds2_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                auto lds3_offset = Register::Value::Placeholder(
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
                co_yield m_context->copier()->copy(
                    lds3_offset, Register::Value::Literal(lds3->getLDSAllocation()->offset()));
                co_yield m_context->copier()->copy(twenty, Register::Value::Literal(20));

                // Load 8 bytes into LDS1
                co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, 0, 1);
                co_yield m_context->mem()->storeLocal(lds1_offset, v_a->subset({0}), 0, 1);
                co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, 1, 1);
                co_yield m_context->mem()->storeLocal(lds1_offset, v_a->subset({0}), 1, 1);
                co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, 2, 2);
                co_yield m_context->mem()->storeLocal(
                    lds1, v_a->subset({0}), 2, 2); // Use LDS1 value instead of offset register
                co_yield m_context->mem()->loadFlat(v_a->subset({0}), v_ptr, 4, 4);
                co_yield m_context->mem()->storeLocal(lds1_offset, v_a->subset({0}), 4, 4);

                // Load 36 bytes into LDS2
                co_yield m_context->mem()->loadFlat(v_a->subset({0, 1}), v_ptr, 8, 8);
                co_yield m_context->mem()->storeLocal(lds2_offset, v_a->subset({0, 1}), 0, 8);
                co_yield m_context->mem()->loadFlat(v_a->subset({0, 1, 2}), v_ptr, 16, 12);
                co_yield m_context->mem()->store(MemoryInstructions::Local,
                                                 lds2_offset,
                                                 v_a->subset({0, 1, 2}),
                                                 Register::Value::Literal(8),
                                                 12);
                co_yield m_context->mem()->loadFlat(v_a->subset({0, 1, 2, 3}), v_ptr, 28, 16);
                co_yield m_context->mem()->store(
                    MemoryInstructions::Local, lds2_offset, v_a->subset({0, 1, 2, 3}), twenty, 16);

                // Read 8 bytes from LDS1 and store to global data
                co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, 0, 1);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), 0, 1);
                co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, 1, 1);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), 1, 1);
                co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, 2, 2);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), 2, 2);
                co_yield m_context->mem()->loadLocal(v_a->subset({0}), lds1_offset, 4, 4);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), 4, 2);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0}), 6, 2, true);

                // Read 36 bytes from LDS2 and store to global data
                co_yield m_context->mem()->loadLocal(
                    v_a->subset({0, 1}), lds2, 0, 8); // Use LDS2 value instead of offset register
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0, 1}), 8, 8);
                co_yield m_context->mem()->load(MemoryInstructions::Local,
                                                v_a->subset({0, 1, 2}),
                                                lds2_offset,
                                                Register::Value::Literal(8),
                                                12);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0, 1, 2}), 16, 12);
                co_yield m_context->mem()->load(
                    MemoryInstructions::Local, v_a->subset({0, 1, 2, 3}), lds2_offset, twenty, 16);
                co_yield m_context->mem()->storeFlat(v_result, v_a->subset({0, 1, 2, 3}), 28, 16);

                // Load 44 bytes into LDS3
                co_yield m_context->mem()->loadFlat(v_a, v_ptr, 44, 44);
                co_yield m_context->mem()->storeLocal(lds3_offset, v_a, 0, 44);
                co_yield m_context->mem()->loadLocal(v_a, lds3_offset, 0, 44);
                co_yield m_context->mem()->storeFlat(v_result, v_a, 44, 44);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_F(MemoryInstructionsLDSTest, GPU_LDSTest)
    {
        genLDSTest();

        int N = 88;

        std::vector<char> a(N);
        for(int i = 0; i < N; i++)
            a[i] = i + 10;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<char>(N);

        KernelArguments kargs;
        kargs.append<void*>("result", d_result.get());
        kargs.append<void*>("a", d_a.get());
        CommandKernel commandKernel;
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        commandKernel.launchKernel(kargs.runtimeArguments());

        std::vector<char> result(N);
        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(char) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            EXPECT_EQ(result[i], a[i]);
    }

    struct GPU_MemoryInstructionsLDSBarrierTest : public GPUContextFixtureParam<unsigned int>
    {

        unsigned int getWorkItemCountParam()
        {
            return std::get<1>(GetParam());
        }

        void genLDSBarrierTest()
        {
            auto workItemCount = getWorkItemCountParam();

            auto k       = m_context->kernel();
            auto command = std::make_shared<Command>();

            auto resultTag  = command->allocateTag();
            auto result_exp = std::make_shared<Expression::Expression>(command->allocateArgument(
                {DataType::Int32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

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

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

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

                // Load 5 + workitemIndex.x into lds3[workitemIndex.x]
                co_yield generateOp<Expression::Add>(lds3_current, lds3_offset, workitemIndex[0]);
                co_yield generateOp<Expression::Add>(
                    v_a, workitemIndex[0], Register::Value::Literal(5));
                co_yield generateOp<Expression::ShiftL>(
                    lds3_current, lds3_current, Register::Value::Literal(2));
                co_yield m_context->mem()->storeLocal(lds3_current, v_a, 0, 4);

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
                co_yield m_context->mem()->loadLocal(v_a, lds3_current, 0, 4);
                co_yield generateOp<Expression::ShiftL>(
                    lds3_current, workitemIndex[0], Register::Value::Literal(2));
                co_yield generateOp<Expression::Add>(
                    v_result->subset({0}), v_result->subset({0}), lds3_current);
                co_yield m_context->mem()->storeFlat(v_result, v_a, 0, 4);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_P(GPU_MemoryInstructionsLDSBarrierTest, GPU_LDSBarrierTest)
    {
        const unsigned int N = getWorkItemCountParam();

        genLDSBarrierTest();

        if(!isLocalDevice())
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
        {

            auto d_result = make_shared_device<int>(N);

            KernelArguments kargs;
            kargs.append<void*>("result", d_result.get());
            CommandKernel commandKernel;
            commandKernel.setContext(m_context);
            commandKernel.generateKernel();

            commandKernel.launchKernel(kargs.runtimeArguments());

            std::vector<int> result(N);
            ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(int) * N, hipMemcpyDefault),
                        HasHipSuccess(0));

            for(unsigned int i = 0; i < N; i++)
            {
                EXPECT_EQ(result[i], 5 + ((i + 1) % N)) << i;
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(GPU_MemoryInstructionsLDSBarrierTest,
                             GPU_MemoryInstructionsLDSBarrierTest,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Values(64, 128, 256, 512, 1024)));

    TEST_P(MemoryInstructionsTest, GPU_MemoryKernelOptions)
    {
        auto v_addr_64bit
            = Register::Value::Placeholder(m_context,
                                           Register::Type::Vector,
                                           DataType::Raw32,
                                           2,
                                           Register::AllocationOptions::FullyContiguous());
        auto v_addr_32bit
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Raw32, 1);
        auto        v_data = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Int32,
                                                   4,
                                                   Register::AllocationOptions::FullyContiguous());
        std::string expected;

        auto setupRegisters = [&]() -> Generator<Instruction> {
            co_yield v_data->allocate();
            co_yield v_addr_64bit->allocate();
            co_yield v_addr_32bit->allocate();
        };
        m_context->schedule(setupRegisters());

        // Test storeGlobalWidth
        {
            auto kb = [&]() -> Generator<Instruction> {
                co_yield m_context->mem()->storeFlat(v_addr_64bit, v_data, 0, 16);
            };

            clearOutput();
            setKernelOptions({.storeGlobalWidth = 4});

            m_context->schedule(kb());
            expected = R"(flat_store_dwordx4 v[4:5], v[0:3])";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.storeGlobalWidth = 3});
            m_context->schedule(kb());
            expected = R"(
            flat_store_dwordx3 v[4:5], v[0:2]
            flat_store_dword v[4:5], v3 offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.storeGlobalWidth = 2});
            m_context->schedule(kb());
            expected = R"(
            flat_store_dwordx2 v[4:5], v[0:1]
            flat_store_dwordx2 v[4:5], v[2:3] offset:8
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.storeGlobalWidth = 1});
            m_context->schedule(kb());
            expected = R"(
            flat_store_dword v[4:5], v0
            flat_store_dword v[4:5], v1 offset:4
            flat_store_dword v[4:5], v2 offset:8
            flat_store_dword v[4:5], v3 offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));
        }

        // Test loadGlobalWidth
        {
            auto kb = [&]() -> Generator<Instruction> {
                co_yield m_context->mem()->loadFlat(v_data, v_addr_64bit, 0, 16);
            };

            clearOutput();
            setKernelOptions({.loadGlobalWidth = 4});
            m_context->schedule(kb());
            expected = R"(
            flat_load_dwordx4 v[0:3], v[4:5]
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.loadGlobalWidth = 3});
            m_context->schedule(kb());
            expected = R"(
            flat_load_dwordx3 v[0:2], v[4:5]
            flat_load_dword v3, v[4:5] offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.loadGlobalWidth = 2});
            m_context->schedule(kb());
            expected = R"(
            flat_load_dwordx2 v[0:1], v[4:5]
            flat_load_dwordx2 v[2:3], v[4:5] offset:8
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.loadGlobalWidth = 1});
            m_context->schedule(kb());
            expected = R"(
            flat_load_dword v0, v[4:5]
            flat_load_dword v1, v[4:5] offset:4
            flat_load_dword v2, v[4:5] offset:8
            flat_load_dword v3, v[4:5] offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));
        }

        // Test storeLocalWidth
        {
            auto kb = [&]() -> Generator<Instruction> {
                co_yield m_context->mem()->storeLocal(v_addr_32bit, v_data, 0, 16);
            };

            clearOutput();
            setKernelOptions({.storeLocalWidth = 4});
            m_context->schedule(kb());
            expected = R"(ds_write_b128 v6, v[0:3])";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.storeLocalWidth = 3});
            m_context->schedule(kb());
            expected = R"(
            ds_write_b96 v6, v[0:2]
            ds_write_b32 v6, v3 offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.storeLocalWidth = 2});
            m_context->schedule(kb());
            expected = R"(
            ds_write_b64 v6, v[0:1]
            ds_write_b64 v6, v[2:3] offset:8
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.storeLocalWidth = 1});
            m_context->schedule(kb());
            expected = R"(
            ds_write_b32 v6, v0
            ds_write_b32 v6, v1 offset:4
            ds_write_b32 v6, v2 offset:8
            ds_write_b32 v6, v3 offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));
        }

        // Test loadLocalWidth
        {
            auto kb = [&]() -> Generator<Instruction> {
                co_yield m_context->mem()->barrier();
                co_yield m_context->mem()->loadLocal(v_data, v_addr_32bit, 0, 16);
            };

            clearOutput();
            setKernelOptions({.loadLocalWidth = 4});
            m_context->schedule(kb());
            expected = R"(
            ds_read_b128 v[0:3], v6
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.loadLocalWidth = 3});
            m_context->schedule(kb());
            expected = R"(
            ds_read_b96 v[0:2], v6
            ds_read_b32 v3, v6 offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.loadLocalWidth = 2});
            m_context->schedule(kb());
            expected = R"(
            ds_read_b64 v[0:1], v6
            ds_read_b64 v[2:3], v6 offset:8
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));

            clearOutput();
            setKernelOptions({.loadLocalWidth = 1});
            m_context->schedule(kb());
            expected = R"(
            ds_read_b32 v0, v6
            ds_read_b32 v1, v6 offset:4
            ds_read_b32 v2, v6 offset:8
            ds_read_b32 v3, v6 offset:12
            )";
            EXPECT_THAT(NormalizedSource(output()), testing::HasSubstr(NormalizedSource(expected)));
        }
    }

    class MemoryInstructions942Test : public GPUContextFixtureParam<rocRoller::DataType>
    {
    public:
        void genByteLoadStore(rocRoller::DataType F8x4Type)
        {
            unsigned int N = 1;

            auto k       = m_context->kernel();
            auto command = std::make_shared<Command>();

            auto resultTag  = command->allocateTag();
            auto result_exp = std::make_shared<Expression::Expression>(command->allocateArgument(
                {F8x4Type, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

            auto workItemCountExpr = std::make_shared<Expression::Expression>(N);
            auto one               = std::make_shared<Expression::Expression>(1u);
            auto zero              = std::make_shared<Expression::Expression>(0u);

            k->setKernelName("PackForStore");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            k->setWorkgroupSize({N, 1, 1});
            k->setWorkitemCount({workItemCountExpr, one, one});
            k->setDynamicSharedMemBytes(zero);

            k->setKernelDimensions(1);

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto vgprSerial = m_context->kernel()->workitemIndex()[0];

                int  size = (N % 4 == 0) ? N / 4 : N / 4 + 1;
                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   F8x4Type,
                                                   size,
                                                   Register::AllocationOptions::FullyContiguous());

                co_yield v_a->allocate();

                auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
                co_yield bufDesc->setup();
                co_yield bufDesc->setBasePointer(s_a);
                co_yield bufDesc->setSize(Register::Value::Literal(N));
                co_yield bufDesc->setOptions(Register::Value::Literal(131072)); //0x00020000

                auto bufInstOpts = rocRoller::BufferInstructionOptions();

                co_yield m_context->mem()->loadBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
                co_yield bufDesc->setBasePointer(s_result);
                co_yield m_context->mem()->storeBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);

                co_yield m_context->mem()->loadBuffer(
                    v_a, vgprSerial, 0, bufDesc, bufInstOpts, N, true);
                co_yield bufDesc->setBasePointer(s_result);
                co_yield m_context->mem()->storeBuffer(
                    v_a, vgprSerial, 0, bufDesc, bufInstOpts, N, true);

                co_yield m_context->mem()->loadLocal(v_a, vgprSerial, 0, N);
                co_yield bufDesc->setBasePointer(s_result);
                co_yield m_context->mem()->storeLocal(v_a, vgprSerial, 0, N);

                co_yield m_context->mem()->loadLocal(v_a, vgprSerial, 0, N, "", true);
                co_yield bufDesc->setBasePointer(s_result);
                co_yield m_context->mem()->storeLocal(v_a, vgprSerial, 0, N, "", true);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());

            EXPECT_NE(NormalizedSource(output()).find("buffer_load_ubyte "), std::string::npos);
            EXPECT_NE(NormalizedSource(output()).find("buffer_store_byte "), std::string::npos);
            EXPECT_NE(NormalizedSource(output()).find("buffer_load_ubyte_d16_hi "),
                      std::string::npos);
            EXPECT_NE(NormalizedSource(output()).find("buffer_store_byte_d16_hi "),
                      std::string::npos);

            EXPECT_NE(NormalizedSource(output()).find("ds_read_u8 "), std::string::npos);
            EXPECT_NE(NormalizedSource(output()).find("ds_write_b8 "), std::string::npos);
            EXPECT_NE(NormalizedSource(output()).find("ds_read_u8_d16_hi "), std::string::npos);
            EXPECT_NE(NormalizedSource(output()).find("ds_write_b8_d16_hi "), std::string::npos);

            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }

        void executeByteLoadStore()
        {
            int N          = 1;
            int bufferSize = N + 20;

            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            std::vector<char> a(bufferSize);
            for(int i = 0; i < N; i++)
                a[i] = i + 10;
            for(int i = N; i < bufferSize; i++)
                a[i] = -i;

            std::vector<char> initialResult(bufferSize);
            for(int i = 0; i < bufferSize; i++)
                initialResult[i] = 2 * i;

            auto d_a      = make_shared_device(a);
            auto d_result = make_shared_device<char>(initialResult);

            KernelArguments kargs;
            kargs.append<void*>("result", d_result.get());
            kargs.append<void*>("a", d_a.get());
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<char> result(bufferSize);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), sizeof(char) * bufferSize, hipMemcpyDefault),
                HasHipSuccess(0));

            for(int i = 0; i < N; i++)
                EXPECT_EQ(result[i], a[i]);
            for(int i = N; i < result.size(); i++)
                EXPECT_EQ(result[i], 2 * i);
        }
    };

    TEST_P(MemoryInstructions942Test, GPU_ByteLoadStore)
    {
        genByteLoadStore(std::get<rocRoller::DataType>(GetParam()));

        if(isLocalDevice())
            executeByteLoadStore();
    }

    INSTANTIATE_TEST_SUITE_P(MemoryInstructions942Test,
                             MemoryInstructions942Test,
                             ::testing::Combine(::testing::Values(GPUArchitectureTarget{
                                                    GPUArchitectureGFX::GFX942, {.sramecc = true}}),
                                                ::testing::Values(rocRoller::DataType::FP8x4,
                                                                  rocRoller::DataType::BF8x4)));

    struct BufferLoad2LDSTest : public GPUContextFixtureParam<int>
    {
        int numBytesParam()
        {
            return std::get<1>(GetParam());
        }

        void genbufferLoad2LDSTest()
        {
            int N = numBytesParam();

            auto k = m_context->kernel();

            k->setKernelName("bufferLoad2LDSTest");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::Int32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"a", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto vgprSerial = m_context->kernel()->workitemIndex()[0];

                int size = (N % 4 == 0) ? N / 4 : N / 4 + 1;

                auto v_ptr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::Int32,
                                                   size,
                                                   Register::AllocationOptions::FullyContiguous());

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto s_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::Int32, 1);

                auto v_lds = Register::Value::AllocateLDS(m_context, DataType::Int32, N);

                co_yield Instruction::Comment("Allocate v_ptr");
                co_yield v_ptr->allocate();

                co_yield Instruction::Comment("Copy s_result to v_result");
                co_yield m_context->copier()->copy(v_result, s_result);

                co_yield Instruction::Comment("Copy lds offset to spgr");
                co_yield m_context->copier()->copy(
                    s_offset, Register::Value::Literal(v_lds->getLDSAllocation()->offset()));

                auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
                co_yield Instruction::Comment("setup bufDesc");
                co_yield bufDesc->setup();
                co_yield Instruction::Comment("Set base pointer");
                co_yield bufDesc->setBasePointer(s_a);
                co_yield Instruction::Comment("Set buffer size");
                co_yield bufDesc->setSize(Register::Value::Literal(N));
                co_yield Instruction::Comment("Set buffer option");
                co_yield bufDesc->setOptions(Register::Value::Literal(131072)); //0x00020000

                auto sgprSrd = bufDesc->allRegisters();

                auto bufInstOpts = rocRoller::BufferInstructionOptions();
                bufInstOpts.lds  = true;

                co_yield m_context->mem()->bufferLoad2LDS(
                    s_offset, vgprSerial, bufDesc, bufInstOpts, N);
                co_yield m_context->mem()->barrier();
                co_yield m_context->mem()->loadLocal(v_ptr, v_lds, 0, N);

                co_yield m_context->mem()->storeFlat(v_result, v_ptr, 0, N);
            };

            setKernelOptions({.alwaysWaitZeroBeforeBarrier = 1});

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }
    };

    TEST_P(BufferLoad2LDSTest, Basic)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);
        int N = numBytesParam();

        if(N % 4 == 3)
        {
            EXPECT_THROW(genbufferLoad2LDSTest(), FatalError);
            GTEST_SKIP();
        }
        else
        {
            genbufferLoad2LDSTest();
        }

        if(!isLocalDevice())
        {

            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            std::vector<unsigned char> a(N);
            for(int i = 0; i < N; i++)
                a[i] = i + 10;

            auto d_a      = make_shared_device(a);
            auto d_result = make_shared_device<unsigned char>(N);

            KernelArguments kargs;
            kargs.append<void*>("result", d_result.get());
            kargs.append<void*>("a", d_a.get());
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);

            std::vector<unsigned char> result(N);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), sizeof(unsigned char) * (N), hipMemcpyDefault),
                HasHipSuccess(0));

            for(int i = 0; i < N; i++)
            {
                EXPECT_EQ(result[i], a[i]);
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        BufferLoad2LDSTest,
        BufferLoad2LDSTest,
        ::testing::Combine(currentGPUISA(), ::testing::Values(1, 2, 4, 8, 12, 16, 32, 64, 128)));
}
