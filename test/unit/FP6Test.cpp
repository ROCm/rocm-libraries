#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes_FP6.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    const size_t numFP6PerFP6x16   = 16;
    const size_t numBytesPerFP6x16 = 12;

    class FP6Test : public GPUContextFixture
    {
    };

    const size_t numFP6PerElement   = 16;
    const size_t numBytesPerElement = 12;

    /*
     * Packs FP6 to FP6x16 on CPU, buffer_load that into FP6x16 to
     * GPU, buffer_store to CPU
     */
    void genFP6x16BufferLoadAndStore(rocRoller::ContextPtr m_context, int num_fp6)
    {
        int N = (num_fp6 / numFP6PerFP6x16) * numBytesPerFP6x16;

        auto k = m_context->kernel();

        k->setKernelName("BufferLoadAndStoreFP6x16");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

        auto aTag  = command->allocateTag();
        auto aExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, aTag, ArgumentType::Value));

        k->addArgument({"result",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});
        k->addArgument(
            {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly, aExpr});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto vgprSerial = m_context->kernel()->workitemIndex()[0];

            int  size = num_fp6 / numFP6PerFP6x16;
            auto v_a  = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::FP6x16,
                                                    size,
                                                    Register::AllocationOptions::FullyContiguous());

            co_yield v_a->allocate();

            auto bufDesc = std::make_shared<rocRoller::BufferDescriptor>(m_context);
            co_yield bufDesc->setup();
            co_yield bufDesc->setBasePointer(s_a);
            co_yield bufDesc->setSize(Register::Value::Literal(N));
            co_yield bufDesc->setOptions(Register::Value::Literal(0x00020000));

            auto bufInstOpts = rocRoller::BufferInstructionOptions();

            co_yield m_context->mem()->loadBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
            co_yield bufDesc->setBasePointer(s_result);
            co_yield m_context->mem()->storeBuffer(v_a, vgprSerial, 0, bufDesc, bufInstOpts, N);
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    /**
     * Packs FP6 to FP6x16 on CPU, flat_load that into FP6x16 to GPU,
     * flat_store to CPU
     */
    void genFP6x16FlatLoadAndStore(rocRoller::ContextPtr m_context, int num_fp6)
    {
        int  N = (num_fp6 / numFP6PerFP6x16) * numBytesPerFP6x16;
        auto k = m_context->kernel();

        k->setKernelName("FlatLoadAndStoreFP6x16");
        k->setKernelDimensions(1);

        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

        auto aTag  = command->allocateTag();
        auto aExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, aTag, ArgumentType::Value));

        k->addArgument({"result",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});
        k->addArgument(
            {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly, aExpr});

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

            int  size = num_fp6 / numFP6PerFP6x16;
            auto v_a  = Register::Value::Placeholder(m_context,
                                                    Register::Type::Vector,
                                                    DataType::FP6x16,
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

    void executeFP6x16FlatLoadAndStore(rocRoller::ContextPtr m_context, int num_fp6)
    {
        std::vector<uint8_t> data(num_fp6);
        for(uint32_t i = 0; i < num_fp6; i++)
        {
            data[i] = (i + 10) % 64;
        }

        auto fp6x16_data = packFP6x16(data);

        std::vector<uint32_t> result(fp6x16_data.size());

        genFP6x16FlatLoadAndStore(m_context, num_fp6);
        CommandKernel commandKernel(m_context);

        auto d_a      = make_shared_device(fp6x16_data);
        auto d_result = make_shared_device<uint32_t>(result.size());

        KernelArguments runtimeArgs;
        runtimeArgs.append<void*>("result", d_result.get());
        runtimeArgs.append<void*>("a", d_a.get());

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(uint32_t) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        auto actual_fp6 = unpackFP6x16(result);
        for(int i = 0; i < data.size(); i++)
        {
            EXPECT_EQ(actual_fp6[i], data[i]);
        }
    }

    void executeFP6x16BufferLoadAndStore(rocRoller::ContextPtr m_context, int num_fp6)
    {
        std::vector<uint8_t> data(num_fp6);
        for(uint32_t i = 0; i < num_fp6; i++)
        {
            data[i] = (i + 10) % 64;
        }

        std::vector<uint32_t> fp6x16_data = packFP6x16(data);

        std::vector<uint32_t> result(fp6x16_data.size());

        genFP6x16BufferLoadAndStore(m_context, num_fp6);
        CommandKernel commandKernel(m_context);

        auto d_a      = make_shared_device(fp6x16_data);
        auto d_result = make_shared_device<uint32_t>(result.size());

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(uint32_t) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        auto actual_fp6 = unpackFP6x16(result);
        for(int i = 0; i < data.size(); i++)
        {
            EXPECT_EQ(actual_fp6[i], data[i]);
        }
    }

    void loadStoreTileFP6(ContextPtr                      m_context,
                          std::shared_ptr<CommandKernel>& commandKernel,
                          bool                            launch,
                          size_t                          nx, // tensor size x
                          size_t                          ny, // tensor size y
                          int                             m, // macro tile size x
                          int                             n, // macro tile size y
                          int                             t_m = 1, // thread tile size x
                          int                             t_n = 1) // thread tile size y
    {
        AssertFatal(nx % numFP6PerFP6x16 == 0, "Invalid FP6 Dimensions");

        int numFP6    = nx * ny;
        int numFP6x16 = numFP6 / numFP6PerFP6x16;

        unsigned int workgroup_size_x = m / t_m;
        unsigned int workgroup_size_y = n / t_n;

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        // generate fp6 values
        std::vector<uint8_t> data(numFP6);
        for(uint32_t i = 0; i < numFP6; i++)
        {
            data[i] = (i + 10) % 64;
        }
        auto a = packFP6x16(data);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<uint32_t>(a.size());

        auto command  = std::make_shared<Command>();
        auto dataType = DataType::FP6;

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, dataType, {0, 1})); // Load A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, dataType, {0, 1})); // Store B
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagLoadA, tagTensorB));

        KernelArguments runtimeArgs;

        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)(ny));
        runtimeArgs.append("d_a_stride_1", (size_t)(1));

        runtimeArgs.append("result", d_b.get());
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

        commandKernel = std::make_shared<CommandKernel>(command, "loadStoreTileFP6", params);
        if(launch)
        {
            commandKernel->launchKernel(runtimeArgs.runtimeArguments());

            auto r = std::vector<uint32_t>(a.size());
            ASSERT_THAT(
                hipMemcpy(r.data(), d_b.get(), numFP6x16 * sizeof(FP6x16), hipMemcpyDefault),
                HasHipSuccess(0));

            for(size_t i = 0; i < a.size(); ++i)
            {
                EXPECT_EQ(r[i], a[i]);
            }
        }
    }

    TEST_P(FP6Test, GPU_FP6x16BufferLoadAndStore)
    {
        int num_fp6 = 16;
        if(isLocalDevice())
        {
            executeFP6x16BufferLoadAndStore(m_context, num_fp6);
        }
        else
        {
            genFP6x16BufferLoadAndStore(m_context, num_fp6);
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(FP6Test, GPU_FP6x16FlatLoadAndStore)
    {
        int num_fp6 = 16;
        if(isLocalDevice())
        {
            executeFP6x16FlatLoadAndStore(m_context, num_fp6);
        }
        else
        {
            genFP6x16FlatLoadAndStore(m_context, num_fp6);
            auto assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }

        auto instructions = NormalizedSourceLines(m_context->instructions()->toString(), false);

        int numFlatLoad   = 0;
        int numFlatLoadx3 = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("flat_load"))
                numFlatLoad++;
            if(instruction.starts_with("flat_load_dwordx3"))
                numFlatLoadx3++;
        }
        EXPECT_EQ(numFlatLoad, 1);
        EXPECT_EQ(numFlatLoadx3, 1);
    }

    TEST_P(FP6Test, GPU_FP6TiledLoadStore)
    {
        int workitemsPerWorkgroup = 64;
        int elementsPerWorkitem   = 16;

        int macM = workitemsPerWorkgroup * elementsPerWorkitem;
        int macN = 16;

        int M = 4 * macM;
        int N = 4 * macN;

        std::shared_ptr<CommandKernel> commandKernel;
        loadStoreTileFP6(
            m_context, commandKernel, isLocalDevice(), M, N, macM, macN, 1, elementsPerWorkitem);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numBufferLoad    = 0;
        int numBufferLoadx3  = 0;
        int numBufferStore   = 0;
        int numBufferStorex3 = 0;
        for(auto instruction : instructions)
        {
            if(instruction.starts_with("buffer_load"))
                numBufferLoad++;
            if(instruction.starts_with("buffer_load_dwordx3"))
                numBufferLoadx3++;
            if(instruction.starts_with("buffer_store"))
                numBufferStore++;
            if(instruction.starts_with("buffer_store_dwordx3"))
                numBufferStorex3++;
        }
        EXPECT_EQ(numBufferLoad, 1);
        EXPECT_EQ(numBufferLoadx3, 1);
        EXPECT_EQ(numBufferStore, 1);
        EXPECT_EQ(numBufferStorex3, 1);
    }

    TEST(FP6Test, CPUConversions)
    {
        auto singleTest = [](auto fp64) {
            // FP6 to FP32
            rocRoller::FP6 fp6(fp64); // change it to FP6 later
            float          fp32(fp6);
            EXPECT_FLOAT_EQ(fp32, fp64);

            // FP32 to FP6
            fp6 = rocRoller::FP6(fp32);
            EXPECT_FLOAT_EQ((double)fp6, fp64);
        };

        // test fp6(i.e., 2e3m)
        // TODO: add bfp6 tests (i.e., 3e2m)
        constexpr auto cases = std::to_array<double>({
            0,      0.125,  0.25,   0.375,  0.5,    0.625, 0.75,   0.875, 1,      1.125, 1.25,
            1.375,  1.5,    1.625,  1.75,   1.875,  2,     2.25,   2.5,   2.75,   3,     3.25,
            3.5,    3.75,   4,      4.5,    5,      5.5,   6,      6.5,   7,      7.5,   -0,
            -0.125, -0.25,  -0.375, -0.5,   -0.625, -0.75, -0.875, -1,    -1.125, -1.25, -1.375,
            -1.5,   -1.625, -1.75,  -1.875, -2,     -2.25, -2.5,   -2.75, -3,     -3.25, -3.5,
            -3.75,  -4,     -4.5,   -5,     -5.5,   -6,    -6.5,   -7,    -7.5,
        });

        for(auto const& c : cases)
        {
            singleTest(c);
        }
    }

    TEST(FP6Test, CPUPack)
    {
        std::vector<uint8_t> fp6bytes(64);
        for(int i = 0; i < fp6bytes.size(); i++)
        {
            fp6bytes[i] = (i + 10) % 64;
        }

        auto fp6x16 = packFP6x16(fp6bytes);
        auto result = unpackFP6x16(fp6x16);
        for(int i = 0; i < fp6bytes.size(); i++)
        {
            EXPECT_EQ(result[i], fp6bytes[i]);
        }

        auto floats = std::to_array<float>({0.125f,
                                            0.25f,
                                            0.375f,
                                            0.5f,
                                            0.625f,
                                            0.75f,
                                            0.875f,
                                            1.f,
                                            1.125f,
                                            1.25f,
                                            1.375f,
                                            1.5f,
                                            1.625f,
                                            1.75f,
                                            1.875f,
                                            2.f});
        ASSERT_EQ(floats.size(), 16);

        fp6bytes.clear();
        for(auto v : floats)
            fp6bytes.push_back(cast_to_fp6(v, FP6_FMT));
        fp6x16 = packFP6x16(fp6bytes);

        EXPECT_EQ(fp6x16.size(), 3);
        EXPECT_EQ(fp6x16[0], 0b10000101000100000011000010000001);
        EXPECT_EQ(fp6x16[1], 0b10110010100010010010000001110001);
        EXPECT_EQ(fp6x16[2], 0b01000000111100111000110100110000);
    }

    INSTANTIATE_TEST_SUITE_P(FP6Test, FP6Test, supportedISATuples());
}
