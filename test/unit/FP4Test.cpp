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
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class FP4MemoryInstructionTest : public GPUContextFixture
    {
    public:
        const size_t numValuesPerByte = 2;
        const size_t numFP4PerElement = 8;

        /*
         * buffer_load into FP4x8 to GPU, buffer_store to CPU
         */
        void genFP4x8BufferLoadAndStore(int num_fp4)
        {
            AssertFatal(num_fp4 % numFP4PerElement == 0,
                        "Number of FP4 values must be multiple times of 8");

            int N = num_fp4 / numValuesPerByte;

            auto k = m_context->kernel();
            k->setKernelName("BufferLoadAndStoreFP4x8");
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

                size_t size = N / 4;
                auto   v_a
                    = Register::Value::Placeholder(m_context,
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

        /*
         * global_load into FP4x8 to GPU, global_store to CPU
         */
        void genFP4x8GlobalLoadAndStore(int num_fp4)
        {
            AssertFatal(num_fp4 % numFP4PerElement == 0,
                        "Number of FP4 values must be multiple times of 8");
            int  N = num_fp4 / numValuesPerByte;
            auto k = m_context->kernel();

            k->setKernelName("GlobalLoadAndStoreFP4x8");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
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
                auto v_a
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::FP4x8,
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

        void executeFP4x8LoadAndStore(int num_fp4)
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

            ASSERT_THAT(hipMemcpy(result.data(),
                                  d_result.get(),
                                  sizeof(uint32_t) * result.size(),
                                  hipMemcpyDefault),
                        HasHipSuccess(0));

            for(int i = 0; i < a.size(); i++)
            {
                EXPECT_EQ(result[i], a[i]);
            }
        }

        void loadStoreTileFP4(std::shared_ptr<CommandKernel>& commandKernel,
                              bool                            launch,
                              size_t                          nx, // tensor size x
                              size_t                          ny, // tensor size y
                              int                             m, // macro tile size x
                              int                             n, // macro tile size y
                              int                             t_m = 1, // thread tile size x
                              int                             t_n = 1) // thread tile size y
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasExplicitVectorCO);

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

            auto tagTensorA = command->addOperation(
                rocRoller::Operations::Tensor(2, dataType, {0, 1})); // Load A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(
                rocRoller::Operations::Tensor(2, dataType, {0, 1})); // Store B
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagLoadA, tagTensorB));

            auto commandArgs = command->createArguments();

            commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
            commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)nx * ny);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)nx);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)ny);
            commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)(ny));
            commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)(1));

            commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_b.get());
            commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)nx * ny);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)nx);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)ny);
            commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)(ny));
            commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)(1));

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

            auto launchParams = std::make_shared<CommandLaunchParameters>();
            launchParams->setManualWorkitemCount({NX, NY, NZ});

            auto macTileVGPR
                = KernelGraph::CoordinateGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

            params->setDimensionInfo(tagLoadA, macTileVGPR);
            params->transposeMemoryAccess[LayoutType::None] = true;

            commandKernel = std::make_shared<CommandKernel>(command, "loadStoreTileFP4");
            commandKernel->setContext(m_context);
            commandKernel->setCommandParameters(params);
            commandKernel->setLaunchParameters(launchParams);
            commandKernel->generateKernel();
            if(launch)
            {
                commandKernel->launchKernel(commandArgs.runtimeArguments());

                ASSERT_THAT(
                    hipMemcpy(r.data(), d_b.get(), numFP4x8 * sizeof(FP4x8), hipMemcpyDefault),
                    HasHipSuccess(0));

                for(size_t i = 0; i < a.size(); ++i)
                {
                    EXPECT_EQ(r[i], a[i]);
                }
            }
        }
    };

    TEST_P(FP4MemoryInstructionTest, GPU_FP4TiledLoadStore)
    {
        int workitemsPerWorkgroup = 64;
        int elementsPerWorkitem   = 8;

        int macM = workitemsPerWorkgroup * elementsPerWorkitem;
        int macN = 8;

        int M = 4 * macM;
        int N = 4 * macN;

        std::shared_ptr<CommandKernel> commandKernel;
        loadStoreTileFP4(commandKernel, isLocalDevice(), M, N, macM, macN, 1, elementsPerWorkitem);

        if(!commandKernel)
            return;

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
        auto const& arch = m_context->targetArchitecture().target();
        if(!arch.isCDNAGPU())
        {
            GTEST_SKIP() << "Test not yet supported on "
                         << m_context->targetArchitecture().target().toString() << std::endl;
        }
        int num_fp4 = 8;
        genFP4x8BufferLoadAndStore(num_fp4);
        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);

        if(isLocalDevice())
        {
            executeFP4x8LoadAndStore(num_fp4);
        }

        auto instructions = NormalizedSourceLines(m_context->instructions()->toString(), false);

        int numBufferLoad   = 0;
        int numBufferLoadx1 = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load"))
                numBufferLoad++;
            if(instruction.starts_with("buffer_load_dword "))
                numBufferLoadx1++;
        }
        EXPECT_EQ(numBufferLoad, 1);
        EXPECT_EQ(numBufferLoadx1, 1);
    }

    TEST_P(FP4MemoryInstructionTest, GPU_FP4x8GlobalLoadAndStore)
    {

        int num_fp4 = 8;
        genFP4x8GlobalLoadAndStore(num_fp4);
        std::vector<char> assembledKernel = m_context->instructions()->assemble();
        EXPECT_GT(assembledKernel.size(), 0);

        if(isLocalDevice())
        {
            executeFP4x8LoadAndStore(num_fp4);
        }

        auto instructions = NormalizedSourceLines(m_context->instructions()->toString(), false);

        int numGlobalLoad   = 0;
        int numGlobalLoadx1 = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("global_load"))
                numGlobalLoad++;
            if(instruction.starts_with("global_load_dword "))
                numGlobalLoadx1++;
        }
        EXPECT_EQ(numGlobalLoad, 1);
        EXPECT_EQ(numGlobalLoadx1, 1);
    }

    INSTANTIATE_TEST_SUITE_P(FP4MemoryInstructionTest,
                             FP4MemoryInstructionTest,
                             supportedISATuples());

    class CPUFP4Test : public GenericContextFixture
    {
    };

    TEST_F(CPUFP4Test, CPUConversions)
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

    TEST_F(CPUFP4Test, CPUFP4x8Pack)
    {
        int num_fp4 = 16;

        std::vector<uint8_t> data(num_fp4);
        for(int i = 0; i < num_fp4; i++)
        {
            data[i] = i % 16;
        }

        std::vector<uint32_t> packed = packFP4x8(data);

        EXPECT_EQ(packed.size(), 2);
        EXPECT_EQ(packed[0], 0b01110110010101000011001000010000);
        EXPECT_EQ(packed[1], 0b11111110110111001011101010011000);

        auto result = unpackFP4x8(packed);

        for(int i = 0; i < num_fp4; i++)
            EXPECT_EQ(data[i], result[i]);
    }

    TEST_F(CPUFP4Test, CPUFP4x8Conversion)
    {
        constexpr auto cases = std::to_array<float>(
            {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6});
        std::vector<float> f32(cases.begin(), cases.end());

        auto fp4x8 = f32_to_fp4x8(f32);

        auto floats = fp4x8_to_f32(fp4x8);

        for(int i = 0; i < floats.size(); i++)
            EXPECT_EQ(cases[i], floats[i]);
    }

    TEST_F(CPUFP4Test, OutOfBoundsConversions)
    {
        rocRoller::FP4 nFP4;
        nFP4 = 6.1;
        EXPECT_FLOAT_EQ(nFP4, 6.0);
        nFP4 = 7.0;
        EXPECT_FLOAT_EQ(nFP4, 6.0);
        nFP4 = 128.0;
        EXPECT_FLOAT_EQ(nFP4, 6.0);
        nFP4 = -6.1;
        EXPECT_FLOAT_EQ(nFP4, -6.0);
        nFP4 = -7.0;
        EXPECT_FLOAT_EQ(nFP4, -6.0);
        nFP4 = -128.0;
        EXPECT_FLOAT_EQ(nFP4, -6.0);
    }

    void checkSpecialValues(float& f32_inf, float& f32_nan, float& f32_zero)
    {
        rocRoller::FP4 fp4_zero(f32_zero);
        rocRoller::FP4 fp4_pos_inf(f32_inf);
        rocRoller::FP4 fp4_neg_inf(-f32_inf);
        rocRoller::FP4 fp4_nan(f32_nan);

        EXPECT_TRUE(std::iszero(fp4_zero));
        EXPECT_FALSE(std::isnan(fp4_pos_inf));
        EXPECT_FALSE(std::isinf(fp4_pos_inf));
        EXPECT_FALSE(std::isnan(fp4_neg_inf));
        EXPECT_FALSE(std::isinf(fp4_neg_inf));
        EXPECT_FALSE(std::isnan(fp4_nan));
        EXPECT_FALSE(std::isinf(fp4_nan));

        EXPECT_FLOAT_EQ(fp4_pos_inf, 6.0);
        EXPECT_FLOAT_EQ(fp4_neg_inf, -6.0);
        EXPECT_FLOAT_EQ(fp4_nan, 6.0);
    }

    TEST_F(CPUFP4Test, SpecialValues)
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

        checkSpecialValues(f32_inf.val, f32_nan.val, f32_zero.val);
    }
}
