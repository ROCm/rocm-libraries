#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes_FP4_Utils.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class FP4MemoryInstructionTest : public GPUContextFixture
    {
    };

    const size_t numValuesPerByte = 2;
    const size_t numFP4PerElement = 8;

    // /**
    //  * buffer_load that into FP4x8 to GPU, buffer_store to CPU
    // */
    void genFP4x8BufferLoadAndStore(rocRoller::ContextPtr m_context, int num_fp4)
    {
        AssertFatal(num_fp4 % numFP4PerElement == 0,
                    "Number of FP4 values must be multiple times of 8");

        int N = num_fp4 / numValuesPerByte;

        auto k = m_context->kernel();
        k->setKernelName("BufferLoadAndStoreFP4x8");
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

            auto vgprSerial = m_context->kernel()->workitemIndex()[0];

            size_t size = N / 4;
            auto   v_a  = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::FP4x8,
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

    // /**
    //  * flat_load that into FP4x8 to GPU, flat_store to CPU
    // */
    void genFP4x8FlatLoadAndStore(rocRoller::ContextPtr m_context, int num_fp4)
    {
        AssertFatal(num_fp4 % numFP4PerElement == 0,
                    "Number of FP4 values must be multiple times of 8");
        int  N = num_fp4 / numValuesPerByte;
        auto k = m_context->kernel();

        k->setKernelName("FlatLoadAndStoreFP4x8");
        k->setKernelDimensions(1);

        k->addArgument(
            {"result", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
        k->addArgument(
            {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1);

            auto v_ptr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1);

            int  size = N / 4;
            auto v_a  = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::FP4x8,
                                                    size,
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

    /**
     *
     */
    void executeFP4x8LoadAndStore(rocRoller::ContextPtr m_context, int num_fp4)
    {
        AssertFatal(num_fp4 % numFP4PerElement == 0,
                    "Number of FP4 values must be multiple times of 16");

        auto rng = RandomGenerator(316473u);
        auto a   = rng.vector<uint>(num_fp4 / numFP4PerElement,
                                  std::numeric_limits<uint>::min(),
                                  std::numeric_limits<uint>::max());

        std::vector<uint32_t> result(a.size());

        std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
            = m_context->instructions()->getExecutableKernel();

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<uint32_t>(result.size());

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());
        KernelInvocation invocation;
        executableKernel->executeKernel(runtimeArgs, invocation);

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(uint32_t) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        for(int i = 0; i < a.size(); i++)
        {
            EXPECT_EQ(result[i], a[i]);
        }
    }

    void loadStoreTileFP4(ContextPtr                      m_context,
                          std::shared_ptr<CommandKernel>& commandKernel,
                          bool                            launch,
                          size_t                          nx, // tensor size x
                          size_t                          ny, // tensor size y
                          int                             m, // macro tile size x
                          int                             n, // macro tile size y
                          int                             t_m = 1, // thread tile size x
                          int                             t_n = 1) // thread tile size y
    {
        AssertFatal(nx % numFP4PerElement == 0, "Invalid FP4 Dimensions");

        int numFP4   = nx * ny;
        int numFP4x8 = numFP4 / numFP4PerElement;

        unsigned int workgroup_size_x = m / t_m;
        unsigned int workgroup_size_y = n / t_n;

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        auto rng = RandomGenerator(316473u);
        auto a   = rng.vector<uint>(
            numFP4x8, std::numeric_limits<uint>::min(), std::numeric_limits<uint>::max());

        std::vector<uint32_t> b(a.size());
        std::vector<uint32_t> r(a.size());

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);

        auto command  = std::make_shared<Command>();
        auto dataType = DataType::FP4;

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, dataType, {0, 1})); // Load A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, dataType, {0, 1})); // Store B
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagLoadA, tagTensorB));

        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)(ny));
        runtimeArgs.append("d_a_stride_1", (size_t)(1));

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_size_0", (size_t)nx);
        runtimeArgs.append("d_b_size_1", (size_t)ny);
        runtimeArgs.append("d_b_stride_0", (size_t)(ny));
        runtimeArgs.append("d_b_stride_1", (size_t)(1));

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        auto macTileVGPR
            = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

        params->setDimensionInfo(tagLoadA, macTileVGPR);

        commandKernel = std::make_shared<CommandKernel>(command, "loadStoreTileFP4", params);
        if(launch)
        {
            commandKernel->launchKernel(runtimeArgs.runtimeArguments());

            ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), numFP4x8 * sizeof(FP4x8), hipMemcpyDefault),
                        HasHipSuccess(0));

            for(size_t i = 0; i < a.size(); ++i)
            {
                EXPECT_EQ(r[i], a[i]);
            }
        }
    }

    TEST_P(FP4MemoryInstructionTest, GPU_FP4TiledLoadStore)
    {
        int workitemsPerWorkgroup = 64;
        int elementsPerWorkitem   = 8;

        int macM = workitemsPerWorkgroup * elementsPerWorkitem;
        int macN = 8;

        int M = 4 * macM;
        int N = 4 * macN;

        std::shared_ptr<CommandKernel> commandKernel;
        loadStoreTileFP4(
            m_context, commandKernel, isLocalDevice(), M, N, macM, macN, 1, elementsPerWorkitem);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numBufferLoad    = 0;
        int numBufferLoadx1  = 0;
        int numBufferStore   = 0;
        int numBufferStorex1 = 0;
        for(auto instruction : instructions)
        {
            if(instruction.starts_with("buffer_load"))
                numBufferLoad++;
            if(instruction.starts_with("buffer_load_dword "))
                numBufferLoadx1++;
            if(instruction.starts_with("buffer_store"))
                numBufferStore++;
            if(instruction.starts_with("buffer_store_dword "))
                numBufferStorex1++;
        }
        EXPECT_EQ(numBufferLoad, 1);
        EXPECT_EQ(numBufferLoadx1, 1);
        EXPECT_EQ(numBufferStore, 1);
        EXPECT_EQ(numBufferStorex1, 1);
    }

    TEST_P(FP4MemoryInstructionTest, GPU_FP4x8BufferLoadAndStore)
    {
        int num_fp4 = 8;
        genFP4x8BufferLoadAndStore(m_context, num_fp4);
        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);

        if(isLocalDevice())
        {
            executeFP4x8LoadAndStore(m_context, num_fp4);
        }

        auto instructions = NormalizedSourceLines(m_context->instructions()->toString(), false);

        int numFlatLoad   = 0;
        int numFlatLoadx1 = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load"))
                numFlatLoad++;
            if(instruction.starts_with("buffer_load_dword "))
                numFlatLoadx1++;
        }
        EXPECT_EQ(numFlatLoad, 1);
        EXPECT_EQ(numFlatLoadx1, 1);
    }

    TEST_P(FP4MemoryInstructionTest, GPU_FP4x8FlatLoadAndStore)
    {

        int num_fp4 = 8;
        genFP4x8FlatLoadAndStore(m_context, num_fp4);
        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);

        if(isLocalDevice())
        {
            executeFP4x8LoadAndStore(m_context, num_fp4);
        }

        auto instructions = NormalizedSourceLines(m_context->instructions()->toString(), false);

        int numFlatLoad   = 0;
        int numFlatLoadx1 = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("flat_load"))
                numFlatLoad++;
            if(instruction.starts_with("flat_load_dword "))
                numFlatLoadx1++;
        }
        EXPECT_EQ(numFlatLoad, 1);
        EXPECT_EQ(numFlatLoadx1, 1);
    }

    INSTANTIATE_TEST_SUITE_P(
        FP4MemoryInstructionTest,
        FP4MemoryInstructionTest,
        ::testing::Combine(::testing::Values("gfx90a:sramecc+, gfx942:sramecc+")));

    TEST(FP4ConversionTest, CPUConversions)
    {
        auto singleTest = [](auto fp64) {
            // FP4 to FP32
            rocRoller::FP4 fp4(fp64);
            float          fp32(fp4);
            EXPECT_FLOAT_EQ(fp32, fp64);

            // FP32 to FP4
            fp4 = rocRoller::FP4(fp32);
            EXPECT_FLOAT_EQ((double)fp4, fp64);
        };

        constexpr auto cases = std::to_array<double>(
            {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6});

        for(auto const& c : cases)
        {
            singleTest(c);
        }
    }

    TEST(FP4x8PackTest, CPUFP4x8Pack)
    {
        int num_fp4 = 16;

        // generate FP4 values
        std::vector<uint8_t> data(num_fp4);
        for(int i = 0; i < num_fp4; i++)
        {
            data[i] = i % 16;
        }

        // pack FP4 to FP4x8
        std::vector<uint32_t> packed(num_fp4 / numFP4PerElement);
        packFP4x8(&packed[0], &data[0], num_fp4);

        EXPECT_EQ(packed.size(), 2);
        EXPECT_EQ(packed[0], 0b01110110010101000011001000010000);
        EXPECT_EQ(packed[1], 0b11111110110111001011101010011000);

        auto result = unpackFP4x8(&packed[0], packed.size());

        for(int i = 0; i < num_fp4; i++)
            EXPECT_EQ(data[i], result[i]);
    }

    TEST(FP4x8ConversionTest, CPUFP4x8Conversion)
    {
        constexpr auto cases = std::to_array<float>(
            {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6});
        std::vector<float> f32;
        for(auto const& c : cases)
            f32.push_back(c);

        auto fp4x8 = f32_to_fp4x8(f32);

        auto floats = fp4x8_to_f32(fp4x8);

        for(int i = 0; i < floats.size(); i++)
            EXPECT_EQ(cases[i], floats[i]);
    }
}
