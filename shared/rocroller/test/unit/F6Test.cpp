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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes_BF6.hpp>
#include <rocRoller/DataTypes/DataTypes_FP6.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    template <typename F6Type>
    struct FloatReference
    {
    };

    template <>
    struct FloatReference<rocRoller::FP6> // E2M3
    {
        // clang-format off
        static constexpr auto Values = std::to_array<float>({
            0.000,  0.125,   0.25,   0.375,   0.50,   0.625,   0.75,   0.875,
            1.000,  1.125,   1.25,   1.375,   1.50,   1.625,   1.75,   1.875,
            2.000,  2.250,   2.50,   2.750,   3.00,   3.250,   3.50,   3.750,
            4.000,  4.500,   5.00,   5.500,   6.00,   6.500,   7.00,   7.500,

           -0.000, -0.125,  -0.25,  -0.375,  -0.50,  -0.625,  -0.75,  -0.875,
           -1.000, -1.125,  -1.25,  -1.375,  -1.50,  -1.625,  -1.75,  -1.875,
           -2.000, -2.250,  -2.50,  -2.750,  -3.00,  -3.250,  -3.50,  -3.750,
           -4.000, -4.500,  -5.00,  -5.500,  -6.00,  -6.500,  -7.00,  -7.500
        });
        // clang-format on
    };

    template <>
    struct FloatReference<rocRoller::BF6> // E3M2
    {
        // clang-format off
        static constexpr auto Values = std::to_array<float>({
            0.00,    0.0625,    0.125,    0.1875,
            0.25,    0.3125,    0.375,    0.4375,
            0.50,    0.6250,    0.750,    0.8750,
            1.00,    1.2500,    1.500,    1.7500,
            2.00,    2.5000,    3.000,    3.5000,
            4.00,    5.0000,    6.000,    7.0000,
            8.00,   10.0000,   12.000,   14.0000,
           16.00,   20.0000,   24.000,   28.0000,

           -0.00,   -0.0625,   -0.125,   -0.1875,
           -0.25,   -0.3125,   -0.375,   -0.4375,
           -0.50,   -0.6250,   -0.750,   -0.8750,
           -1.00,   -1.2500,   -1.500,   -1.7500,
           -2.00,   -2.5000,   -3.000,   -3.5000,
           -4.00,   -5.0000,   -6.000,   -7.0000,
           -8.00,  -10.0000,  -12.000,  -14.0000,
          -16.00,  -20.0000,  -24.000,  -28.0000
        });
        // clang-format on
    };

    const size_t numF6PerF6x16    = 16;
    const size_t numBytesPerF6x16 = 12;

    class F6Test : public GPUContextFixtureParam<rocRoller::DataType>
    {
    public:
        /*
         * Packs F6 to F6x16 on CPU, buffer_load that into F6x16 to
         * GPU, buffer_store to CPU
         */
        void genF6x16BufferLoadAndStore(int num_f6, DataType F6x16Type)
        {
            int N = (num_f6 / numF6PerF6x16) * numBytesPerF6x16;

            auto k = m_context->kernel();

            k->setKernelName("BufferLoadAndStoreF6x16");
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
            k->addArgument({"a",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::ReadOnly,
                            aExpr});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto vgprSerial = m_context->kernel()->workitemIndex()[0];

                int  size = num_f6 / numF6PerF6x16;
                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   F6x16Type,
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
         * Packs F6 to F6x16 on CPU, global_load that into F6x16 to GPU,
         * global_store to CPU
         */
        void genF6x16GlobalLoadAndStore(int num_f6, DataType F6x16Type)
        {
            int  N = (num_f6 / numF6PerF6x16) * numBytesPerF6x16;
            auto k = m_context->kernel();

            k->setKernelName("GlobalLoadAndStoreF6x16");
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
            k->addArgument({"a",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::ReadOnly,
                            aExpr});

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

                int  size = num_f6 / numF6PerF6x16;
                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   F6x16Type,
                                                   size,
                                                   Register::AllocationOptions::FullyContiguous());

                co_yield v_a->allocate();
                co_yield v_ptr->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

                co_yield m_context->copier()->copy(v_ptr, s_a, "Move pointer.");

                co_yield m_context->mem()->loadGlobal(v_a, v_ptr, 0, N);
                co_yield m_context->mem()->storeGlobal(v_result, v_a, 0, N);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

        void executeF6x16LoadAndStore(int num_f6, DataType F6x16Type, int isGlobal)
        {
            std::vector<uint8_t> data(num_f6);
            for(uint32_t i = 0; i < num_f6; i++)
            {
                data[i] = (i + 10) % 64;
            }

            auto f6x16_data = packF6x16(data);

            std::vector<uint32_t> result(f6x16_data.size());

            if(isGlobal)
                genF6x16GlobalLoadAndStore(num_f6, F6x16Type);
            else
                genF6x16BufferLoadAndStore(num_f6, F6x16Type);

            CommandKernel commandKernel;
            commandKernel.setContext(m_context);

            auto d_a      = make_shared_device(f6x16_data);
            auto d_result = make_shared_device<uint32_t>(result.size());

            KernelArguments runtimeArgs;
            runtimeArgs.append<void*>("result", d_result.get());
            runtimeArgs.append<void*>("a", d_a.get());

            commandKernel.launchKernel(runtimeArgs.runtimeArguments());

            ASSERT_THAT(hipMemcpy(result.data(),
                                  d_result.get(),
                                  sizeof(uint32_t) * result.size(),
                                  hipMemcpyDefault),
                        HasHipSuccess(0));

            auto actual_f6 = unpackF6x16(result);
            for(int i = 0; i < data.size(); i++)
            {
                EXPECT_EQ(actual_f6[i], data[i]);
            }
        }

        void loadStoreTileF6(std::shared_ptr<CommandKernel>& commandKernel,
                             DataType                        F6Type,
                             bool                            launch,
                             size_t                          nx, // tensor size x
                             size_t                          ny, // tensor size y
                             int                             m, // macro tile size x
                             int                             n, // macro tile size y
                             int                             t_m = 1, // thread tile size x
                             int                             t_n = 1) // thread tile size y
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasExplicitVectorCO);

            AssertFatal(nx % numF6PerF6x16 == 0, "Invalid F6 Dimensions");

            int numF6    = nx * ny;
            int numF6x16 = numF6 / numF6PerF6x16;

            unsigned int workgroup_size_x = m / t_m;
            unsigned int workgroup_size_y = n / t_n;

            // each workgroup will get one tile; since workgroup_size matches m * n
            auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
            auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
            auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

            // generate fp6 values
            std::vector<uint8_t> data(numF6);
            for(uint32_t i = 0; i < numF6; i++)
            {
                data[i] = (i + 10) % 64;
            }
            auto a = packF6x16(data);

            auto d_a = make_shared_device(a);
            auto d_b = make_shared_device<uint32_t>(a.size());

            auto command = std::make_shared<Command>();

            auto tagTensorA
                = command->addOperation(rocRoller::Operations::Tensor(2, F6Type, {0, 1})); // Load A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(
                rocRoller::Operations::Tensor(2, F6Type, {0, 1})); // Store B
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagLoadA, tagTensorB));

            auto commandArgs = command->createArguments();
            commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
            commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)nx * ny);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)nx);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)ny);
            commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)ny);
            commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)1);

            commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_b.get());
            commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)nx * ny);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)nx);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)ny);
            commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)ny);
            commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)1);

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            auto launchParams = std::make_shared<CommandLaunchParameters>();
            launchParams->setManualWorkitemCount({NX, NY, NZ});

            auto macTileVGPR
                = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

            params->setDimensionInfo(tagLoadA, macTileVGPR);
            params->transposeMemoryAccess.set(LayoutType::None, true);

            commandKernel = std::make_shared<CommandKernel>(command, "loadStoreTileF6");
            commandKernel->setContext(m_context);
            commandKernel->setCommandParameters(params);
            commandKernel->setLaunchParameters(launchParams);
            commandKernel->generateKernel();
            if(launch)
            {
                commandKernel->launchKernel(commandArgs.runtimeArguments());

                auto r = std::vector<uint32_t>(a.size());
                ASSERT_THAT(
                    hipMemcpy(r.data(), d_b.get(), numF6x16 * sizeof(FP6x16), hipMemcpyDefault),
                    HasHipSuccess(0));

                for(size_t i = 0; i < a.size(); ++i)
                {
                    EXPECT_EQ(r[i], a[i]);
                }
            }
        }
    };

    TEST_P(F6Test, GPU_F6x16BufferLoadAndStore)
    {
        auto const& arch = m_context->targetArchitecture().target();
        if(!arch.isCDNAGPU())
        {
            GTEST_SKIP() << "Test not yet supported on "
                         << m_context->targetArchitecture().target().toString() << std::endl;
        }

        int  num_f6    = 16;
        auto F6x16Type = std::get<rocRoller::DataType>(GetParam()) == DataType::FP6
                             ? DataType::FP6x16
                             : DataType::BF6x16;
        if(isLocalDevice())
        {
            executeF6x16LoadAndStore(num_f6, F6x16Type, false);
        }
        else
        {
            genF6x16BufferLoadAndStore(num_f6, F6x16Type);
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_P(F6Test, GPU_F6x16GlobalLoadAndStore)
    {
        auto const& arch = m_context->targetArchitecture().target();
        if(!arch.isCDNAGPU())
        {
            GTEST_SKIP() << "Test not yet supported on "
                         << m_context->targetArchitecture().target().toString() << std::endl;
        }

        int  num_f6    = 16;
        auto F6x16Type = std::get<rocRoller::DataType>(GetParam()) == DataType::FP6
                             ? DataType::FP6x16
                             : DataType::BF6x16;

        if(isLocalDevice())
        {
            executeF6x16LoadAndStore(num_f6, F6x16Type, true);
        }
        else
        {
            genF6x16GlobalLoadAndStore(num_f6, F6x16Type);
            auto assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }

        auto instructions = NormalizedSourceLines(m_context->instructions()->toString(), false);

        int numGlobalLoad   = 0;
        int numGlobalLoadx3 = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("global_load"))
                numGlobalLoad++;
            if(instruction.starts_with("global_load_dwordx3"))
                numGlobalLoadx3++;
        }
        EXPECT_EQ(numGlobalLoad, 1);
        EXPECT_EQ(numGlobalLoadx3, 1);
    }

    TEST_P(F6Test, GPU_F6TiledLoadStore)
    {
        auto const& arch = m_context->targetArchitecture().target();
        if(!arch.isCDNAGPU())
        {
            GTEST_SKIP() << "Test not yet supported on "
                         << m_context->targetArchitecture().target().toString() << std::endl;
        }

        int workitemsPerWorkgroup = 64;
        int elementsPerWorkitem   = 16;

        int macM = workitemsPerWorkgroup * elementsPerWorkitem;
        int macN = 16;

        int M = 4 * macM;
        int N = 4 * macN;

        std::shared_ptr<CommandKernel> commandKernel;
        loadStoreTileF6(commandKernel,
                        std::get<rocRoller::DataType>(GetParam()),
                        isLocalDevice(),
                        M,
                        N,
                        macM,
                        macN,
                        1,
                        elementsPerWorkitem);

        if(!commandKernel)
            return;

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

    INSTANTIATE_TEST_SUITE_P(F6Test,
                             F6Test,
                             ::testing::Combine(supportedISAValues(),
                                                ::testing::Values(rocRoller::DataType::FP6,
                                                                  rocRoller::DataType::BF6)));

    class CPUF6Test : public GenericContextFixture
    {
    };

    template <typename F6Type>
    void numberConversion(double fp64)
    {
        // F6 to FP32
        F6Type f6(fp64);
        float  fp32(f6);
        EXPECT_FLOAT_EQ(fp32, fp64);

        // FP32 to F6
        f6 = F6Type(fp32);
        EXPECT_FLOAT_EQ((double)f6, fp64);
    }

    TEST_F(CPUF6Test, CPUConversions)
    {
        auto const& FP6Values = FloatReference<rocRoller::FP6>::Values;
        std::for_each(FP6Values.begin(), FP6Values.end(), numberConversion<rocRoller::FP6>);

        auto const& BF6Values = FloatReference<rocRoller::BF6>::Values;
        std::for_each(BF6Values.begin(), BF6Values.end(), numberConversion<rocRoller::BF6>);
    }

    TEST_F(CPUF6Test, CPUPack)
    {
        std::vector<uint8_t> f6bytes(64);
        for(int i = 0; i < f6bytes.size(); i++)
        {
            f6bytes[i] = (i + 10) % 64;
        }

        auto f6x16  = packF6x16(f6bytes);
        auto result = unpackF6x16(f6x16);
        for(int i = 0; i < f6bytes.size(); i++)
        {
            EXPECT_EQ(result[i], f6bytes[i]);
        }

        // clang-format off
        std::array fp6 = {0.125f,  /* 000001 */
                          0.25f,   /* 000010 */
                          0.375f,  /* 000011 */
                          0.5f,    /* 000100 */
                          0.625f,  /* 000101 */
                          0.75f,   /* 000110 */
                          0.875f,  /* 000111 */
                          1.f,     /* 001000 */
                          1.125f,  /* 001001 */
                          1.25f,   /* 001010 */
                          1.375f,  /* 001011 */
                          1.5f,    /* 001100 */
                          1.625f,  /* 001101 */
                          1.75f,   /* 001110 */
                          1.875f,  /* 001111 */
                          2.f};    /* 010000 */

        std::array bf6 = {0.0625f, /* 000001 */
                          0.125f,  /* 000010 */
                          0.1875f, /* 000011 */
                          0.25f,   /* 000100 */
                          0.3125f, /* 000101 */
                          0.375f,  /* 000110 */
                          0.4375f, /* 000111 */
                          0.50f,   /* 001000 */
                          0.625f,  /* 001001 */
                          0.75f,   /* 001010 */
                          0.875f,  /* 001011 */
                          1.f,     /* 001100 */
                          1.25f,   /* 001101 */
                          1.5f,    /* 001110 */
                          1.75f,   /* 001111 */
                          2.f};    /* 010000 */
        // clang-format on

        auto testPacking = [&](auto& floats, auto fmt) {
            ASSERT_EQ(floats.size(), 16);

            f6bytes.clear();
            for(auto v : floats)
                f6bytes.push_back(cast_to_f6(v, fmt));
            f6x16 = packF6x16(f6bytes);

            EXPECT_EQ(f6x16.size(), 3);
            EXPECT_EQ(f6x16[0], 0b10000101000100000011000010000001);
            EXPECT_EQ(f6x16[1], 0b10110010100010010010000001110001);
            EXPECT_EQ(f6x16[2], 0b01000000111100111000110100110000);
        };

        testPacking(fp6, DataTypes::FP6_FMT);
        testPacking(bf6, DataTypes::BF6_FMT);
    }

    TEST_F(CPUF6Test, OutOfBoundsConversions)
    {
        rocRoller::FP6 nF6;
        nF6 = 7.6;
        EXPECT_FLOAT_EQ(nF6, 7.5);
        nF6 = 8.0;
        EXPECT_FLOAT_EQ(nF6, 7.5);
        nF6 = 128.0;
        EXPECT_FLOAT_EQ(nF6, 7.5);
        nF6 = -7.6;
        EXPECT_FLOAT_EQ(nF6, -7.5);
        nF6 = -8.0;
        EXPECT_FLOAT_EQ(nF6, -7.5);
        nF6 = -128.0;
        EXPECT_FLOAT_EQ(nF6, -7.5);

        rocRoller::BF6 nBF6;
        nBF6 = 28.1;
        EXPECT_FLOAT_EQ(nBF6, 28.0);
        nBF6 = 29.0;
        EXPECT_FLOAT_EQ(nBF6, 28.0);
        nBF6 = 128.0;
        EXPECT_FLOAT_EQ(nBF6, 28.0);
        nBF6 = -28.1;
        EXPECT_FLOAT_EQ(nBF6, -28.0);
        nBF6 = -29.0;
        EXPECT_FLOAT_EQ(nBF6, -28.0);
        nBF6 = -128.0;
        EXPECT_FLOAT_EQ(nBF6, -28.0);
    }

    template <typename F6Type>
    void checkSpecialValues(float& f32_inf, float& f32_nan, float& f32_zero)
    {
        F6Type f6_zero(f32_zero);
        F6Type f6_pos_inf(f32_inf);
        F6Type f6_neg_inf(-f32_inf);
        F6Type f6_nan(f32_nan);

        EXPECT_TRUE(std::iszero(f6_zero));
        EXPECT_FALSE(std::isnan(f6_pos_inf));
        EXPECT_FALSE(std::isinf(f6_pos_inf));
        EXPECT_FALSE(std::isnan(f6_neg_inf));
        EXPECT_FALSE(std::isinf(f6_neg_inf));
        EXPECT_FALSE(std::isnan(f6_nan));
        EXPECT_FALSE(std::isinf(f6_nan));

        constexpr float max_f6_val = std::is_same_v<F6Type, BF6> ? 28.0f : 7.5f;

        EXPECT_FLOAT_EQ(f6_pos_inf, max_f6_val);
        EXPECT_FLOAT_EQ(f6_neg_inf, -max_f6_val);
        EXPECT_FLOAT_EQ(f6_nan, max_f6_val);
    }

    TEST_F(CPUF6Test, SpecialValues)
    {
        union
        {
            uint32_t bits;
            float    val;
        } f32_inf, f32_nan, f32_zero;

        // For single-precision, if all exponent bits are 1 and
        //  - if mantissa is zero     => Inf
        //  - if mantissa is non-zero => NaN
        f32_inf.bits  = 0x7F800000;
        f32_nan.bits  = 0x7F800001;
        f32_zero.bits = 0x0;

        EXPECT_TRUE(std::isinf(f32_inf.val));
        EXPECT_TRUE(std::isnan(f32_nan.val));

        checkSpecialValues<rocRoller::FP6>(f32_inf.val, f32_nan.val, f32_zero.val);
        checkSpecialValues<rocRoller::BF6>(f32_inf.val, f32_nan.val, f32_zero.val);
    }
}
