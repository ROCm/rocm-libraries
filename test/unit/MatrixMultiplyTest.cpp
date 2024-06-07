
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/T_Execute.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "DataTypes/DataTypes.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace MatrixMultiplyTest
{

    class MatrixMultiplyTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};
    };

    class FP8MatrixMultiplyTestGPU : public GPUContextFixture
    {
    };

    template <typename T, typename ACC = T>
    void matrixMultiplyMacroTile(ContextPtr                      m_context,
                                 int                             wave_m,
                                 int                             wave_n,
                                 int                             wave_k,
                                 int                             wave_b,
                                 double                          acceptableError,
                                 std::shared_ptr<CommandKernel>& commandKernel)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        if constexpr(std::is_same_v<T, FP8_NANOO>)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        }

        auto dataTypeAB = TypeInfo<T>::Var.dataType;
        auto dataTypeD  = TypeInfo<ACC>::Var.dataType;

        // matrix size: A is MxK; B is KxN; D is MxN
        int mac_m = wave_m;
        int mac_n = wave_n;
        int mac_k = 32;

        int M = mac_m;
        int N = mac_n;
        int K = 32;

        if constexpr(std::is_same_v<T, FP8_NANOO>)
        {
            mac_k = 2 * wave_k;
            K     = 64;
        }

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");
        AssertFatal(K % mac_k == 0, "MacroTile size mismatch (K)");

        AssertFatal(mac_m == wave_m, "Single output MacroTile.");
        AssertFatal(mac_n == wave_n, "Single output MacroTile.");

        uint workgroup_size_x = 64;
        uint workgroup_size_y = 1;

        auto bpe = DataTypeInfo::Get(dataTypeAB).elementSize;
        AssertFatal(mac_m * mac_k * bpe > wave_m * wave_k, "Not enough elements.");

        auto NX = std::make_shared<Expression::Expression>(workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(9861u);

        auto A = random.vector<T>(M * K, -1.f, 1.f);
        auto B = random.vector<T>(K * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_D = make_shared_device<ACC>(M * N);

        auto command = std::make_shared<Command>();
        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB, {(size_t)1})); // A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

        auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD)); // D
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_A.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);

        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_B.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);

        commandArgs.setArgument(tagTensorD, ArgumentType::Value, d_D.get());
        commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

        auto kernelOptions                           = std::make_shared<KernelOptions>();
        kernelOptions->packMultipleElementsInto1VGPR = true;
        kernelOptions->enableLongDwordInstructions   = true;

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        // TODO: the translate step should figure out that there is a
        // T_Mul and do the right thing for the T_Load_Tiled commands
        auto macTileA = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k},
                                                                LayoutType::MATRIX_A,
                                                                {wave_m, wave_n, wave_k, wave_b},
                                                                MemoryType::WAVE_LDS);

        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        auto postParams = std::make_shared<CommandParameters>();
        postParams->setManualWavefrontCount({1u, 1u});

        commandKernel = std::make_shared<CommandKernel>(
            command, "MatrixMultiplyMacroTile", params, postParams, kernelOptions);
        commandKernel->launchKernel(commandArgs.runtimeArguments());

        std::vector<ACC> D(M * N);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(ACC), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<ACC> c_D(M * N, ACC{});
        std::vector<ACC> c_C(M * N, ACC{});
        CPUMM(c_D, c_C, A, B, M, N, K, 1.0, 0.0, false, false);

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, acceptableError);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTile)
    {
        std::shared_ptr<CommandKernel> commandKernel;
        matrixMultiplyMacroTile<float>(m_context, 32, 32, 2, 1, 2.e-6, commandKernel);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP16)
    {
        std::shared_ptr<CommandKernel> commandKernel;
        matrixMultiplyMacroTile<Half, Half>(m_context, 32, 32, 8, 1, 2.e-6, commandKernel);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int expectedLocalWriteOffset = 0;
        int numLocalRead             = 0;
        int expectedLocalReadOffset  = 0;
        for(auto const& instruction : instructions)
        {
            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 64;
            }

            if(instruction.starts_with("ds_read_u16"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 4 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 4 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 64;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 128);
        EXPECT_EQ(numLocalRead, 16);
    }

    TEST_P(FP8MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP8_16x16x32)
    {
        if(!isLocalDevice())
            return;

        std::shared_ptr<CommandKernel> commandKernel;
        matrixMultiplyMacroTile<FP8_NANOO, float>(m_context, 16, 16, 32, 1, 2.e-6, commandKernel);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int expectedLocalWriteOffset = 0;
        int numLocalRead             = 0;
        int expectedLocalReadOffset  = 0;
        int numMFMA                  = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("v_mfma_f32_16x16x32_fp8_fp8"))
                numMFMA++;

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 1024;
            }

            if(instruction.starts_with("ds_read_u8"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 8 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 8 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 16;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 1024);
        EXPECT_EQ(numLocalRead, 16);
        EXPECT_EQ(numMFMA, 2);
    }

    TEST_P(FP8MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP8_32x32x16)
    {
        if(!isLocalDevice())
            return;

        std::shared_ptr<CommandKernel> commandKernel;
        matrixMultiplyMacroTile<FP8_NANOO, float>(m_context, 32, 32, 16, 1, 2.e-6, commandKernel);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int expectedLocalWriteOffset = 0;
        int numLocalRead             = 0;
        int expectedLocalReadOffset  = 0;
        int numMFMA                  = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("v_mfma_f32_32x32x16_fp8_fp8"))
                numMFMA++;

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffset > 0)
                    EXPECT_TRUE(instruction.ends_with("offset:"
                                                      + std::to_string(expectedLocalWriteOffset)));
                expectedLocalWriteOffset += 1024;
            }

            if(instruction.starts_with("ds_read_u8"))
            {
                numLocalRead++;

                if(expectedLocalReadOffset > 0)
                    EXPECT_TRUE(
                        instruction.ends_with("offset:" + std::to_string(expectedLocalReadOffset)));

                if(numLocalRead % 8 == 0)
                {
                    expectedLocalReadOffset = numLocalRead / 8 * 512;
                }
                else
                {
                    expectedLocalReadOffset += 32;
                }
            }
        }

        EXPECT_EQ(expectedLocalWriteOffset, 1024);
        EXPECT_EQ(numLocalRead, 16);
        EXPECT_EQ(numMFMA, 2);
    }

    template <typename T, typename ACC = T>
    void matrixMultiplyAB(ContextPtr m_context,
                          int        wave_m,
                          int        wave_n,
                          int        wave_k,
                          int        wave_b,
                          double     acceptableError)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        if constexpr(std::is_same_v<T, FP8_NANOO>)
        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
        }

        auto dataTypeAB = TypeInfo<T>::Var.dataType;
        auto dataTypeD  = TypeInfo<ACC>::Var.dataType;

        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 1024;
        int N = 1024;
        int K = 512;

        // output macro tile size; we will launch 2x2 waves
        int mac_m = 2 * wave_m;
        int mac_n = 2 * wave_n;
        int mac_k = 64;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        auto bpe = DataTypeInfo::Get(dataTypeAB).elementSize;
        AssertFatal(mac_m * mac_k * bpe > wave_m * wave_k, "Not enough elements.");

        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<T>(M * K, -1.f, 1.f);
        auto B = random.vector<T>(K * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_D = make_shared_device<ACC>(M * N);

        auto command  = std::make_shared<Command>();
        auto dataType = TypeInfo<T>::Var.dataType;

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB, {(size_t)1})); // A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

        auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD)); // D
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_A.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);

        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_B.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);

        commandArgs.setArgument(tagTensorD, ArgumentType::Value, d_D.get());
        commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        // TODO: the translate step should figure out that there is a
        // T_Mul and do the right thing for the T_Load_Tiled commands
        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        auto postParams = std::make_shared<CommandParameters>();
        postParams->setManualWavefrontCount({2u, 2u});

        CommandKernel commandKernel(command, "MatrixMultiplyAB", params, postParams);
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<ACC> D(M * N);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(ACC), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<ACC> c_D(M * N, ACC{});
        std::vector<ACC> c_C(M * N, ACC{});
        CPUMM(c_D, c_C, A, B, M, N, K, 1.0, 0.0, false, false);

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, acceptableError);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyAB)
    {
        matrixMultiplyAB<float>(m_context, 32, 32, 2, 1, 2.e-6);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABFP16)
    {
        matrixMultiplyAB<Half>(m_context, 32, 32, 8, 1, 2.e-5);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABFP8_16x16x32)
    {
        matrixMultiplyAB<FP8_NANOO, float>(m_context, 16, 16, 32, 1, 2.e-5);
    }

    TEST_P(FP8MatrixMultiplyTestGPU, GPU_MatrixMultiplyABFP8_32x32x16)
    {
        if(!isLocalDevice())
            return;

        matrixMultiplyAB<FP8_NANOO, float>(m_context, 32, 32, 16, 1, 2.e-5);
    }

    template <typename T>
    void matrixMultiplyABC(ContextPtr m_context,
                           int        wave_m,
                           int        wave_n,
                           int        wave_k,
                           int        wave_b,
                           double     acceptableError)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 1024;
        int N = 1024;
        int K = 512;

        // output macro tile size
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<T>(M * K, -1.f, 1.f);
        auto B = random.vector<T>(K * N, -1.f, 1.f);
        auto C = random.vector<T>(M * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_C = make_shared_device(C);
        auto d_D = make_shared_device<T>(M * N);

        auto command  = std::make_shared<Command>();
        auto dataType = TypeInfo<T>::Var.dataType;

        auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
        auto tagLoadA   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
        auto tagLoadB   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagTensorC = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // C
        auto tagLoadC   = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

        auto tagAB
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto tagStoreD
            = execute.addXOp(rocRoller::Operations::E_Add(tagAB, tagLoadC)); // D = A * B + C
        command->addOperation(std::move(execute));

        auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // D
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_A.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);

        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_B.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);

        commandArgs.setArgument(tagTensorC, ArgumentType::Value, d_C.get());
        commandArgs.setArgument(tagTensorC, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorC, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 1, (size_t)M);

        commandArgs.setArgument(tagTensorD, ArgumentType::Value, d_D.get());
        commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
        commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        // TODO: the translate step should figure out that there is a
        // T_Mul and do the right thing for the T_Load_Tiled commands
        auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);
        params->setDimensionInfo(tagLoadC, macTileC);

        auto postParams = std::make_shared<CommandParameters>();
        postParams->setManualWavefrontCount({2u, 2u});

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "ABC", params, postParams);
        commandKernel.launchKernel(commandArgs.runtimeArguments());

        std::vector<T> D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<T> c_D(M * N, 0.f);
        CPUMM(c_D, C, A, B, M, N, K, 1.0, 1.0, false, false);

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, acceptableError);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABC)
    {
        matrixMultiplyABC<float>(m_context, 32, 32, 2, 1, 2.e-6);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABCFP16)
    {
        matrixMultiplyABC<Half>(m_context, 32, 32, 8, 1, 2.e-5);
    }

    INSTANTIATE_TEST_SUITE_P(FP8MatrixMultiplyTestGPU,
                             FP8MatrixMultiplyTestGPU,
                             ::testing::Combine(::testing::Values("gfx942:sramecc+")));

}
