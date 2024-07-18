
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ScaledMatrixMultiply.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
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

    template <typename... Ts>
    class BaseMatrixMultiplyContextFixture
        : public BaseGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<std::string, Ts...>>
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            std::string device = std::get<0>(this->GetParam());

            return this->createContextForArch(device);
        }

    public:
        std::shared_ptr<CommandKernel> commandKernel;

        template <typename T, typename ACC = T>
        void matrixMultiplyMacroTile(int         wave_m,
                                     int         wave_n,
                                     int         wave_k,
                                     int         wave_b,
                                     double      acceptableError,
                                     bool        useLDSB = true,
                                     std::string transA  = "N",
                                     std::string transB  = "N",
                                     uint8_t     scaleA  = 127,
                                     uint8_t     scaleB  = 127)
        {
            commandKernel = nullptr;

            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
            if constexpr(std::is_same_v<T, FP8> || std::is_same_v<T, BF8>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
            }
            if constexpr(std::is_same_v<T, FP6> || std::is_same_v<T, BF6> || std::is_same_v<T, FP4>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

                // TODO Remove this when we can generate FP8/F6/F4 kernels on archs that don't support FP8
                if(!isLocalDevice())
                {
                    GTEST_SKIP();
                }
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

            if constexpr(std::is_same_v<T, FP8> || std::is_same_v<T, BF8>)
            {
                mac_k = 2 * wave_k;
                K     = 2 * mac_k;
            }
            if constexpr(std::is_same_v<T, FP6> || std::is_same_v<T, BF6> || std::is_same_v<T, FP4>)
            {
                mac_k = 2 * wave_k;
                K     = 4 * mac_k;
            }

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");
            AssertFatal(K % mac_k == 0, "MacroTile size mismatch (K)");

            AssertFatal(mac_m == wave_m, "Single output MacroTile.");
            AssertFatal(mac_n == wave_n, "Single output MacroTile.");

            uint workgroup_size_x = 64;
            uint workgroup_size_y = 1;

            auto bpe = CeilDivide(DataTypeInfo::Get(dataTypeAB).elementBits, 8u);
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

            std::vector<size_t> unitStridesN = {1, 0};
            std::vector<size_t> unitStridesT = {0, 1};

            auto command    = std::make_shared<Command>();
            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeAB, transA == "N" ? unitStridesN : unitStridesT));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeAB, transB == "N" ? unitStridesN : unitStridesT));
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagStoreD = command->addOperation(
                rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

            auto tagTensorD = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD));
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(tagTensorA, ArgumentType::Value, (uint8_t*)d_A.get());
            commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
            if(transA == "N")
            {
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);
            }
            else
            {
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)K);
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)1);
            }

            commandArgs.setArgument(tagTensorB, ArgumentType::Value, (uint8_t*)d_B.get());
            commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
            if(transB == "N")
            {
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);
            }
            else
            {
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)N);
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)1);
            }

            commandArgs.setArgument(tagTensorD, ArgumentType::Value, d_D.get());
            commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
            commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
            commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
            commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
            commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

            auto kernelOptions                           = std::make_shared<KernelOptions>();
            kernelOptions->packMultipleElementsInto1VGPR = true;
            kernelOptions->enableLongDwordInstructions   = true;

            kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A] = transA == "T";
            kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B] = transB == "T";

            kernelOptions->scaleA = scaleA;
            kernelOptions->scaleB = scaleB;

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
            params->setManualWorkitemCount({NX, NY, NZ});

            // TODO: the translate step should figure out that there is a
            // T_Mul and do the right thing for the T_Load_Tiled commands
            auto macTileA
                = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k},
                                                          LayoutType::MATRIX_A,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          MemoryType::WAVE_LDS);

            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {mac_k, mac_n},
                LayoutType::MATRIX_B,
                {wave_m, wave_n, wave_k, wave_b},
                useLDSB ? MemoryType::WAVE_LDS : MemoryType::WAVE);

            params->setDimensionInfo(tagLoadA, macTileA);
            params->setDimensionInfo(tagLoadB, macTileB);

            auto postParams = std::make_shared<CommandParameters>();
            postParams->setManualWavefrontCount({1u, 1u});

            setKernelOptions(*kernelOptions);
            commandKernel = std::make_shared<CommandKernel>(
                command, testKernelName(), params, postParams, kernelOptions, m_context);

            if(isLocalDevice())
            {
                commandKernel->launchKernel(commandArgs.runtimeArguments());

                std::vector<ACC> D(M * N);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(ACC), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<ACC> c_D(M * N, ACC{});
                std::vector<ACC> c_C(M * N, ACC{});

                float alpha = 1.0f;
                alpha *= std::pow(2.0f, int(scaleA) - 127) * std::pow(2.0f, int(scaleB) - 127);

                CPUMM(c_D, c_C, A, B, M, N, K, alpha, 0.0, transA == "T", transB == "T");

                double rnorm = relativeNorm(D, c_D);
                Log::info("RNorm is {}", rnorm);
                ASSERT_LT(rnorm, acceptableError);
            }
        }

        template <typename T, typename ACC = T>
        void matrixMultiplyAB(int     wave_m,
                              int     wave_n,
                              int     wave_k,
                              int     wave_b,
                              double  acceptableError,
                              int     M      = 1024,
                              int     N      = 1024,
                              int     K      = 512,
                              uint8_t scaleA = 127,
                              uint8_t scaleB = 127)
        {
            // matrix size: A is MxK; B is KxN; D is MxN
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
            if constexpr(std::is_same_v<T, FP8> || std::is_same_v<T, BF8>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);

                // TODO Remove this when we can generate FP8 kernels on archs that don't support FP8
                if(!isLocalDevice())
                {
                    GTEST_SKIP();
                }
            }

            auto dataTypeAB = TypeInfo<T>::Var.dataType;
            auto dataTypeD  = TypeInfo<ACC>::Var.dataType;

            // output macro tile size; we will launch 2x2 waves
            int mac_m = 2 * wave_m;
            int mac_n = 2 * wave_n;
            int mac_k = 2 * wave_k;

            AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

            uint workgroup_size_x = 256;
            uint workgroup_size_y = 1;

            auto bpe = CeilDivide(DataTypeInfo::Get(dataTypeAB).elementBits, 8u);
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
                = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB, {1, 0}));
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB
                = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeAB, {1, 0})); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagStoreD = command->addOperation(
                rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

            auto tagTensorD
                = command->addOperation(rocRoller::Operations::Tensor(2, dataTypeD)); // D
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

            auto kernelOptions    = std::make_shared<KernelOptions>();
            kernelOptions->scaleA = scaleA;
            kernelOptions->scaleB = scaleB;

            CommandKernel commandKernel(
                command, "MatrixMultiplyAB", params, postParams, kernelOptions);
            if(isLocalDevice())
            {
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<ACC> D(M * N);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(ACC), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<ACC> c_D(M * N, ACC{});
                std::vector<ACC> c_C(M * N, ACC{});

                float alpha = 1.0f;
                alpha *= std::pow(2.0f, int(scaleA) - 127) * std::pow(2.0f, int(scaleB) - 127);

                CPUMM(c_D, c_C, A, B, M, N, K, alpha, 0.0, false, false);

                double rnorm = relativeNorm(D, c_D);
                Log::info("RNorm is {}", rnorm);
                ASSERT_LT(rnorm, acceptableError);
            }
        }

        template <typename T>
        void matrixMultiplyABC(
            int wave_m, int wave_n, int wave_k, int wave_b, double acceptableError)
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

            auto tagTensorA
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagTensorC
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagAB
                = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

            auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
            auto tagStoreD
                = execute.addXOp(rocRoller::Operations::E_Add(tagAB, tagLoadC)); // D = A * B + C
            command->addOperation(std::move(execute));

            auto tagTensorD
                = command->addOperation(rocRoller::Operations::Tensor(2, dataType)); // D
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            CommandArguments commandArgs = command->createArguments();

            commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_A.get());
            commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
            commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
            commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);
            // tiled?
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
            if(isLocalDevice())
            {
                commandKernel.launchKernel(commandArgs.runtimeArguments());

                std::vector<T> D(M * N, 0.f);
                ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDefault),
                            HasHipSuccess(0));

                std::vector<T> c_D(M * N, 0.f);
                CPUMM(c_D, C, A, B, M, N, K, 1.0, 1.0, false, false);

                double rnorm = relativeNorm(D, c_D);
                Log::info("RNorm is {}", rnorm);
                ASSERT_LT(rnorm, acceptableError);
            }
        }
    };

    class MatrixMultiplyTestGPU : public BaseMatrixMultiplyContextFixture<>
    {
    };

    class MatrixMultiplyTestGPUF8 : public BaseMatrixMultiplyContextFixture<rocRoller::DataType>
    {
    };

    class MatrixMultiplyTestGPUF6 : public BaseMatrixMultiplyContextFixture<rocRoller::DataType>
    {
    };

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTile)
    {
        matrixMultiplyMacroTile<float>(32, 32, 2, 1, 2.e-6);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP16)
    {
        matrixMultiplyMacroTile<Half, Half>(32, 32, 8, 1, 2.e-6, false);

        if(!commandKernel)
            return;

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

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x32_NN)
    {
        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(16, 16, 32, 1, 1.e-5, false, "N", "N");
        else
            matrixMultiplyMacroTile<BF8, float>(16, 16, 32, 1, 1.e-5, false, "N", "N");

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int               expectedLocalWriteOffset = 0;
        int               numLocalRead             = 0;
        int               expectedLocalReadOffset  = 0;
        int               numMFMA                  = 0;
        std::string const mfma_pattern
            = isFP8 ? "v_mfma_f32_16x16x32_fp8_fp8" : "v_mfma_f32_16x16x32_bf8_bf8";
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma_pattern))
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

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_32x32x16_NN)
    {
        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(32, 32, 16, 1, 1.e-5, false, "N", "N");
        else
            matrixMultiplyMacroTile<BF8, float>(32, 32, 16, 1, 1.e-5, false, "N", "N");

        if(!commandKernel)
            return;

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int               expectedLocalWriteOffset = 0;
        int               numLocalRead             = 0;
        int               expectedLocalReadOffset  = 0;
        int               numMFMA                  = 0;
        std::string const mfma_pattern
            = isFP8 ? "v_mfma_f32_32x32x16_fp8_fp8" : "v_mfma_f32_32x32x16_bf8_bf8";
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma_pattern))
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

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x32_TN)
    {
        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(16, 16, 32, 1, 1.e-5, true, "T", "N");
        else
            matrixMultiplyMacroTile<BF8, float>(16, 16, 32, 1, 1.e-5, true, "T", "N");
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(32, 32, 64, 1, 5.e-6, true, "T", "N");
        else
            matrixMultiplyMacroTile<BF8, float>(32, 32, 64, 1, 5.e-6, true, "T", "N");

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, "v_mfma_f32_32x32x64_f8f6f4"), 2);
        if(isFP8)
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b000 blgp:0b000"), 2);
        else
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b001 blgp:0b001"), 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_ScaledMatrixMultiplyMacroTileF8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(
                32, 32, 64, 1, 5.e-6, true, "T", "N", scaleA, scaleB);
        else
            matrixMultiplyMacroTile<BF8, float>(
                32, 32, 64, 1, 5.e-6, true, "T", "N", scaleA, scaleB);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, "v_mfma_scale_f32_32x32x64_f8f6f4"), 2);
        if(isFP8)
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b000 abid:1 blgp:0b000"), 2);
        else
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b001 abid:1 blgp:0b001"), 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyMacroTileF8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(16, 16, 128, 1, 1.e-5, true, "T", "N");
        else
            matrixMultiplyMacroTile<BF8, float>(16, 16, 128, 1, 1.e-5, true, "T", "N");

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, "v_mfma_f32_16x16x128_f8f6f4"), 2);
        if(isFP8)
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b000 blgp:0b000"), 2);
        else
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b001 blgp:0b001"), 2);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_ScaledMatrixMultiplyMacroTileF8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        bool const isFP8 = std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8;
        if(isFP8)
            matrixMultiplyMacroTile<FP8, float>(
                16, 16, 128, 1, 1.e-5, true, "T", "N", scaleA, scaleB);
        else
            matrixMultiplyMacroTile<BF8, float>(
                16, 16, 128, 1, 1.e-5, true, "T", "N", scaleA, scaleB);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "v_mfma"), 2);
        EXPECT_EQ(countSubstring(generatedCode, "v_mfma_scale_f32_16x16x128_f8f6f4"), 2);
        if(isFP8)
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b000 abid:1 blgp:0b000"), 2);
        else
            EXPECT_EQ(countSubstring(generatedCode, "cbsz:0b001 abid:1 blgp:0b001"), 2);
    }

    void verifyInsturctions(std::string       instructionString,
                            std::vector<int>& expectedLocalWriteOffsets,
                            std::vector<int>& expectedLocalReadOffsets,
                            int               e_numWrite,
                            int               e_numRead,
                            int               e_numMFMA,
                            std::string       mfma,
                            std::string       mfmaDataTypes)
    {
        auto instructions = NormalizedSourceLines(instructionString, false);

        int numLocalWrite = 0;
        int numLocalRead  = 0;
        int numMFMA       = 0;

        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with(mfma))
            {
                numMFMA++;
                EXPECT_NE(instruction.find(mfmaDataTypes), std::string::npos);
            }

            // Count the number of ds_write_b128 instructions and make sure they have
            // the expected offset values
            if(instruction.starts_with("ds_write_b128"))
            {
                if(expectedLocalWriteOffsets[numLocalWrite] > 0)
                    EXPECT_TRUE(instruction.ends_with(
                        "offset:" + std::to_string(expectedLocalWriteOffsets[numLocalWrite])));
                numLocalWrite++;
            }

            if(instruction.starts_with("ds_read_b128") || instruction.starts_with("ds_read_b64"))
            {
                if(expectedLocalReadOffsets[numLocalRead] > 0)
                    EXPECT_TRUE(instruction.ends_with(
                        "offset:" + std::to_string(expectedLocalReadOffsets[numLocalRead])));
                numLocalRead++;
            }
        }

        EXPECT_EQ(numLocalWrite, e_numWrite);
        EXPECT_EQ(numLocalRead, e_numRead);
        EXPECT_EQ(numMFMA, e_numMFMA);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP4_32x32x64_TN)
    {
        matrixMultiplyMacroTile<FP4, float>(32, 32, 64, 1, 2.e-6, true, "T", "N");

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 32;
        int mac_n = 32;
        int mac_k = 128;

        auto localEndA = 4 * mac_m * mac_k / 8;

        std::vector<int> expectedLocalWriteOffsets{0, 64, localEndA, localEndA + 64};

        std::vector<int> expectedLocalReadOffsets{0, localEndA, 32, localEndA + 32};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           4,
                           4,
                           2,
                           "v_mfma_f32_32x32x64_f8f6f4",
                           "cbsz:0b100 blgp:0b100");
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_ScaledMatrixMultiplyMacroTileFP4_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        matrixMultiplyMacroTile<FP4, float>(32, 32, 64, 1, 2.e-6, true, "T", "N", scaleA, scaleB);

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 32;
        int mac_n = 32;
        int mac_k = 128;

        auto localEndA = 4 * mac_m * mac_k / 8;

        std::vector<int> expectedLocalWriteOffsets{0, 64, localEndA, localEndA + 64};

        std::vector<int> expectedLocalReadOffsets{0, localEndA, 32, localEndA + 32};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           4,
                           4,
                           2,
                           "v_mfma_scale_f32_32x32x64_f8f6f4",
                           "cbsz:0b100 abid:1 blgp:0b100");
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP4_16x16x128_TN)
    {
        matrixMultiplyMacroTile<FP4, float>(16, 16, 128, 1, 2.e-6, true, "T", "N");

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 16;
        int mac_n = 16;
        int mac_k = 256;

        auto localEndA = 4 * mac_m * mac_k / 8;

        std::vector<int> expectedLocalWriteOffsets{0, 128, localEndA, localEndA + 128};

        std::vector<int> expectedLocalReadOffsets{0, localEndA, 64, localEndA + 64};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           4,
                           4,
                           2,
                           "v_mfma_f32_16x16x128_f8f6f4",
                           "cbsz:0b100 blgp:0b100");
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_ScaledMatrixMultiplyMacroTileFP4_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        matrixMultiplyMacroTile<FP4, float>(16, 16, 128, 1, 2.e-6, true, "T", "N", scaleA, scaleB);

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 16;
        int mac_n = 16;
        int mac_k = 256;

        auto localEndA = 4 * mac_m * mac_k / 8;

        std::vector<int> expectedLocalWriteOffsets{0, 128, localEndA, localEndA + 128};

        std::vector<int> expectedLocalReadOffsets{0, localEndA, 64, localEndA + 64};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           4,
                           4,
                           2,
                           "v_mfma_scale_f32_16x16x128_f8f6f4",
                           "cbsz:0b100 abid:1 blgp:0b100");
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyAB)
    {
        matrixMultiplyAB<float>(32, 32, 2, 1, 2.e-6);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABFP16)
    {
        matrixMultiplyAB<Half>(32, 32, 8, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_16x16x32)
    {
        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(16, 16, 32, 1, 2.e-5);
        else
            matrixMultiplyAB<BF8, float>(16, 16, 32, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_32x32x16)
    {
        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(32, 32, 16, 1, 2.e-5);
        else
            matrixMultiplyAB<BF8, float>(32, 32, 16, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_32x32x64)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(32, 32, 64, 1, 2.e-5);
        else
            matrixMultiplyAB<BF8, float>(32, 32, 64, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_ScaledMatrixMultiplyABF8_32x32x64)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        int wave_m = 32;
        int wave_n = 32;
        int wave_k = 64;
        int wave_b = 1;

        // For quick verification, the matrix size is kept small
        int M = 2 * wave_m;
        int N = 2 * wave_n;
        int K = 2 * wave_k;

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(
                wave_m, wave_n, wave_k, wave_b, 2.e-5, M, N, K, scaleA, scaleB);
        else
            matrixMultiplyAB<BF8, float>(
                wave_m, wave_n, wave_k, wave_b, 2.e-5, M, N, K, scaleA, scaleB);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_MatrixMultiplyABF8_16x16x128)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(16, 16, 128, 1, 2.e-5);
        else
            matrixMultiplyAB<BF8, float>(16, 16, 128, 1, 2.e-5);
    }

    TEST_P(MatrixMultiplyTestGPUF8, GPU_ScaledMatrixMultiplyABF8_16x16x128)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        int wave_m = 16;
        int wave_n = 16;
        int wave_k = 128;
        int wave_b = 1;

        // For quick verification, the matrix size is kept small
        int M = 2 * wave_m;
        int N = 2 * wave_n;
        int K = 2 * wave_k;

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP8)
            matrixMultiplyAB<FP8, float>(
                wave_m, wave_n, wave_k, wave_b, 2.e-5, M, N, K, scaleA, scaleB);
        else
            matrixMultiplyAB<BF8, float>(
                wave_m, wave_n, wave_k, wave_b, 2.e-5, M, N, K, scaleA, scaleB);
    }

    TEST_P(MatrixMultiplyTestGPUF6, GPU_MatrixMultiplyMacroTileF6_16x16x128_TN)
    {
        auto mfmaDataTypes = "cbsz:0b010 blgp:0b010";
        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP6)
            matrixMultiplyMacroTile<FP6, float>(16, 16, 128, 1, 2.e-6, true, "T", "N");
        else
        {
            matrixMultiplyMacroTile<BF6, float>(16, 16, 128, 1, 2.e-6, true, "T", "N");
            mfmaDataTypes = "cbsz:0b011 blgp:0b011";
        }

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 16;
        int mac_n = 16;
        int mac_k = 256;

        auto localEndA = 6 * mac_m * mac_k / 8;

        std::vector<int> expectedLocalWriteOffsets{
            0, 16, 32, localEndA, localEndA + 16, localEndA + 32};

        std::vector<int> expectedLocalReadOffsets{
            0, 16, localEndA, localEndA + 16, 96, 112, localEndA + 96, localEndA + 112};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           6,
                           8,
                           2,
                           "v_mfma_f32_16x16x128_f8f6f4",
                           mfmaDataTypes);
    }

    TEST_P(MatrixMultiplyTestGPUF6, GPU_ScaledMatrixMultiplyMacroTileF6_16x16x128_TN)
    {
        auto mfmaDataTypes = "cbsz:0b010 abid:1 blgp:0b010";

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP6)
            matrixMultiplyMacroTile<FP6, float>(
                16, 16, 128, 1, 2.e-6, true, "T", "N", scaleA, scaleB);
        else
        {
            matrixMultiplyMacroTile<BF6, float>(
                16, 16, 128, 1, 2.e-6, true, "T", "N", scaleA, scaleB);
            mfmaDataTypes = "cbsz:0b011 abid:1 blgp:0b011";
        }

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 16;
        int mac_n = 16;
        int mac_k = 256;

        auto localEndA = 6 * mac_m * mac_k / 8;

        std::vector<int> expectedLocalWriteOffsets{
            0, 16, 32, localEndA, localEndA + 16, localEndA + 32};

        std::vector<int> expectedLocalReadOffsets{
            0, 16, localEndA, localEndA + 16, 96, 112, localEndA + 96, localEndA + 112};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           6,
                           8,
                           2,
                           "v_mfma_scale_f32_16x16x128_f8f6f4",
                           mfmaDataTypes);
    }

    TEST_P(MatrixMultiplyTestGPUF6, GPU_MatrixMultiplyMacroTileF6_32x32x64_TN)
    {
        auto mfmaDataTypes = "cbsz:0b010 blgp:0b010";
        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP6)
            matrixMultiplyMacroTile<FP6, float>(32, 32, 64, 1, 2.e-6, true, "T", "N");
        else
        {
            matrixMultiplyMacroTile<BF6, float>(32, 32, 64, 1, 2.e-6, true, "T", "N");
            mfmaDataTypes = "cbsz:0b011 blgp:0b011";
        }

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 32;
        int mac_n = 32;
        int mac_k = 128;

        auto localEndA = 6 * mac_m * mac_k / 8;

        int numLocalWrite = 0;
        int numLocalRead  = 0;
        int numMFMA       = 0;

        std::vector<int> expectedLocalWriteOffsets{
            0, 16, 32, localEndA, localEndA + 16, localEndA + 32};

        std::vector<int> expectedLocalReadOffsets{
            0, 16, localEndA, localEndA + 16, 48, 64, localEndA + 48, localEndA + 64};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           6,
                           8,
                           2,
                           "v_mfma_f32_32x32x64_f8f6f4",
                           mfmaDataTypes);
    }

    TEST_P(MatrixMultiplyTestGPUF6, GPU_ScaledMatrixMultiplyMacroTileF6_32x32x64_TN)
    {
        auto mfmaDataTypes = "cbsz:0b010 abid:1 blgp:0b010";

        uint8_t scaleA = 128;
        uint8_t scaleB = 125;

        if(std::get<rocRoller::DataType>(GetParam()) == rocRoller::DataType::FP6)
            matrixMultiplyMacroTile<FP6, float>(
                32, 32, 64, 1, 2.e-6, true, "T", "N", scaleA, scaleB);
        else
        {
            matrixMultiplyMacroTile<BF6, float>(
                32, 32, 64, 1, 2.e-6, true, "T", "N", scaleA, scaleB);
            mfmaDataTypes = "cbsz:0b011 abid:1 blgp:0b011";
        }

        if(!commandKernel)
            return;

        std::string instructionString = commandKernel->getInstructions();

        int mac_m = 32;
        int mac_n = 32;
        int mac_k = 128;

        auto localEndA = 6 * mac_m * mac_k / 8;

        int numLocalWrite = 0;
        int numLocalRead  = 0;
        int numMFMA       = 0;

        std::vector<int> expectedLocalWriteOffsets{
            0, 16, 32, localEndA, localEndA + 16, localEndA + 32};

        std::vector<int> expectedLocalReadOffsets{
            0, 16, localEndA, localEndA + 16, 48, 64, localEndA + 48, localEndA + 64};

        verifyInsturctions(instructionString,
                           expectedLocalWriteOffsets,
                           expectedLocalReadOffsets,
                           6,
                           8,
                           2,
                           "v_mfma_scale_f32_32x32x64_f8f6f4",
                           mfmaDataTypes);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABC)
    {
        matrixMultiplyABC<float>(32, 32, 2, 1, 2.e-6);
    }

    TEST_P(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABCFP16)
    {
        matrixMultiplyABC<Half>(32, 32, 8, 1, 2.e-5);
    }

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTestGPU,
                             MatrixMultiplyTestGPU,
                             mfmaSupportedISATuples());

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTestGPUF8,
                             MatrixMultiplyTestGPUF8,
                             ::testing::Combine(mfmaSupportedISAValues(),
                                                ::testing::Values(rocRoller::DataType::FP8,
                                                                  rocRoller::DataType::BF8)));

    INSTANTIATE_TEST_SUITE_P(MatrixMultiplyTestGPUF6,
                             MatrixMultiplyTestGPUF6,
                             ::testing::Combine(mfmaSupportedISAValues(),
                                                ::testing::Values(rocRoller::DataType::FP6,
                                                                  rocRoller::DataType::BF6)));

    /**
     * Extra param: M/N tile size of instruction, allowing us to test both
     * v_mfma_scale_f32_16x16x128_f8f6f4 and
     * v_mfma_scale_f32_32x32x64_f8f6f4.
     */
    class ScaledMatrixMultiplyTestGPU : public GPUContextFixtureParam<int>
    {
    };

    TEST_P(ScaledMatrixMultiplyTestGPU, GenInstruction)
    {
        auto tileMN = std::get<1>(this->GetParam());

        auto k = m_context->kernel();

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->setKernelDimensions(1);

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto M = 16, N = 16, K = 128;
        if(tileMN == 32)
        {
            M = 32;
            N = 32;
            K = 64;
        }

        int abRegs = 8;
        int cdRegs = 4;
        if(tileMN != 16)
        {
            abRegs = 8;
            cdRegs = 16;
        }

        auto kb = [&]() -> Generator<Instruction> {
            auto fc     = Register::AllocationOptions::FullyContiguous();
            auto v_dest = Register::Value::Placeholder(
                m_context, Register::Type::Accumulator, DataType::Float, cdRegs, fc);
            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::FP8x4, abRegs, fc);
            auto v_b = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::FP8x4, abRegs, fc);
            auto v_c = Register::Value::Placeholder(
                m_context, Register::Type::Accumulator, DataType::Float, cdRegs, fc);
            auto v_a_scale = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int8, 1, fc);
            auto v_b_scale = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int8, 1, fc);
            for(auto reg : {v_a, v_b, v_c, v_a_scale, v_b_scale})
                co_yield reg->allocate();

            auto smm = Component::Get<rocRoller::InstructionGenerators::ScaledMatrixMultiply>(
                m_context, DataType::Float, DataType::FP8x4);

            co_yield smm->mul(v_dest, v_a, v_b, v_c, v_a_scale, v_b_scale, M, N, K);
        };

        m_context->schedule(kb());

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        auto mnemonic = tileMN == 16 ? "v_mfma_scale_f32_16x16x128_f8f6f4"
                                     : "v_mfma_scale_f32_32x32x64_f8f6f4";

        EXPECT_THAT(output(), testing::HasSubstr(mnemonic));

        if(isLocalDevice())
        {
            std::shared_ptr<rocRoller::ExecutableKernel> executableKernel
                = m_context->instructions()->getExecutableKernel();

            KernelArguments  kargs;
            KernelInvocation invocation;

            executableKernel->executeKernel(kargs, invocation);
        }
        else
        {
            auto assembledKernel = m_context->instructions()->assemble();
            ASSERT_GT(assembledKernel.size(), 0);
        }
    }

    INSTANTIATE_TEST_SUITE_P(ScaledMatrixMultiplyTestGPU,
                             ScaledMatrixMultiplyTestGPU,
                             ::testing::Combine(::testing::Values("gfx950"),
                                                ::testing::Values(16, 32)));
}
