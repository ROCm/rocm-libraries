
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/Double.hpp>
#include <rocRoller/CodeGen/Arithmetic/Float.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int64.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "DataTypes/DataTypes.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "Scheduling/Observers/FileWritingObserver.hpp"
#include "SourceMatcher.hpp"
#include "Utilities/Error.hpp"

using namespace rocRoller;

namespace MatrixMultiplyTest
{

    class MatrixMultiplyTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};
    };

    template <typename T>
    void matrixMultiplyMacroTile(std::shared_ptr<Context> m_context,
                                 int                      wave_m,
                                 int                      wave_n,
                                 int                      wave_k,
                                 int                      wave_b,
                                 double                   acceptableError)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 32;
        int N = 32;
        int K = 32;

        // output macro tile size
        int mac_m = 32;
        int mac_n = 32;
        int mac_k = 32;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        AssertFatal(mac_m == wave_m, "Single MacroTile.");
        AssertFatal(mac_n == wave_n, "Single MacroTile.");

        uint workgroup_size_x = 64;
        uint workgroup_size_y = 1;

        auto NX = std::make_shared<Expression::Expression>(workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(98761u);

        auto A = random.vector<T>(M * K, -1.f, 1.f);
        auto B = random.vector<T>(K * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_D = make_shared_device<T>(M * N);

        auto command  = std::make_shared<Command>();
        auto dataType = TypeInfo<T>::Var.dataType;

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // B

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1))); // D = A * B

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 2))); // D

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("A", d_A.get());
        runtimeArgs.append("d_a_limit", (size_t)M * K);
        runtimeArgs.append("d_a_size_0", (size_t)M);
        runtimeArgs.append("d_a_size_1", (size_t)K);
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)M);

        runtimeArgs.append("B", d_B.get());
        runtimeArgs.append("d_b_limit", (size_t)K * N);
        runtimeArgs.append("d_b_size_0", (size_t)K);
        runtimeArgs.append("d_b_size_1", (size_t)N);
        runtimeArgs.append("d_b_stride_0", (size_t)1);
        runtimeArgs.append("d_b_stride_1", (size_t)K);

        runtimeArgs.append("D", d_D.get());
        runtimeArgs.append("d_d_limit", (size_t)M * N);
        runtimeArgs.append("d_d_stride_0", (size_t)1);
        runtimeArgs.append("d_d_stride_1", (size_t)M);

        // TODO: remove this when for loop indexing is fixed
        command->allocateArgument(DataType::UInt32, DataDirection::ReadOnly, "UINT_MAT_K");
        runtimeArgs.append("UINT_MAT_K", static_cast<uint>(K));

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        // TODO: the translate step should figure out that there is a
        // T_Mul and do the right thing for the T_Load_Tiled commands
        auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile(
            0, {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_1 = KernelGraph::CoordinateTransform::MacroTile(
            1, {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2,
                                                          {mac_m, mac_n},
                                                          LayoutType::MATRIX_ACCUMULATOR,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 1, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, -1, four, one, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 0, two, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 1, two, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "GEMM01", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        std::vector<T> D(M * N);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<T> c_D(M * N, 0.f);
        std::vector<T> c_C(M * N, 0.f);
        CPUMM(c_D, c_C, A, B, M, N, K, 1.0, 0.0, false);

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, acceptableError);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTile)
    {
        matrixMultiplyMacroTile<float>(m_context, 32, 32, 2, 1, 2.e-6);
    }

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyMacroTileFP16)
    {
        matrixMultiplyMacroTile<Half>(m_context, 32, 32, 8, 1, 2.e-6);
    }

    template <typename T>
    void matrixMultiplyAB(std::shared_ptr<Context> m_context,
                          int                      wave_m,
                          int                      wave_n,
                          int                      wave_k,
                          int                      wave_b,
                          double                   acceptableError)
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

        // one macro tile per workgroup
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
        auto d_D = make_shared_device<T>(M * N);

        auto command  = std::make_shared<Command>();
        auto dataType = TypeInfo<T>::Var.dataType;

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // B

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1))); // D = A * B

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 2))); // D

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("A", d_A.get());
        runtimeArgs.append("d_a_limit", (size_t)M * K);
        runtimeArgs.append("d_a_size_0", (size_t)M);
        runtimeArgs.append("d_a_size_1", (size_t)K);
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)M);

        runtimeArgs.append("B", d_B.get());
        runtimeArgs.append("d_b_limit", (size_t)K * N);
        runtimeArgs.append("d_b_size_0", (size_t)K);
        runtimeArgs.append("d_b_size_1", (size_t)N);
        runtimeArgs.append("d_b_stride_0", (size_t)1);
        runtimeArgs.append("d_b_stride_1", (size_t)K);

        runtimeArgs.append("D", d_D.get());
        runtimeArgs.append("d_d_limit", (size_t)M * N);
        runtimeArgs.append("d_d_stride_0", (size_t)1);
        runtimeArgs.append("d_d_stride_1", (size_t)M);

        // TODO: remove this when for loop indexing is fixed
        command->allocateArgument(DataType::UInt32, DataDirection::ReadOnly, "UINT_MAT_K");
        runtimeArgs.append("UINT_MAT_K", static_cast<uint>(K));

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        // TODO: the translate step should figure out that there is a
        // T_Mul and do the right thing for the T_Load_Tiled commands
        auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile(
            0, {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_1 = KernelGraph::CoordinateTransform::MacroTile(
            1, {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2,
                                                          {mac_m, mac_n},
                                                          LayoutType::MATRIX_ACCUMULATOR,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 1, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, -1, four, one, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 0, two, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 1, two, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "AB", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        std::vector<T> D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<T> c_D(M * N, 0.f);
        std::vector<T> c_C(M * N, 0.f);
        CPUMM(c_D, c_C, A, B, M, N, K, 1.0, 0.0, false);

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

    TEST_F(MatrixMultiplyTestGPU, GPU_MatrixMultiplyABC)
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

        // wave tile sizes
        int wave_m = 32;
        int wave_n = 32;
        int wave_k = 2;
        int wave_b = 1;

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        // one macro tile per workgroup
        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<float>(M * K, -1.f, 1.f);
        auto B = random.vector<float>(K * N, -1.f, 1.f);
        auto C = random.vector<float>(M * N, -1.f, 1.f);

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_C = make_shared_device(C);
        auto d_D = make_shared_device<float>(M * N);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 2))); // C

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(3, 0, 1))); // D = A * B

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(4, 3, 2))); // E = D + C
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 4))); // D

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("A", d_A.get());
        runtimeArgs.append("d_a_limit", (size_t)M * K);
        runtimeArgs.append("d_a_size_0", (size_t)M);
        runtimeArgs.append("d_a_size_1", (size_t)K);
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)M);

        runtimeArgs.append("B", d_B.get());
        runtimeArgs.append("d_b_limit", (size_t)K * N);
        runtimeArgs.append("d_b_size_0", (size_t)K);
        runtimeArgs.append("d_b_size_1", (size_t)N);
        runtimeArgs.append("d_b_stride_0", (size_t)1);
        runtimeArgs.append("d_b_stride_1", (size_t)K);

        runtimeArgs.append("C", d_C.get());
        runtimeArgs.append("d_c_limit", (size_t)M * N);
        runtimeArgs.append("d_c_size_0", (size_t)M);
        runtimeArgs.append("d_c_size_1", (size_t)N);
        runtimeArgs.append("d_c_stride_0", (size_t)1);
        runtimeArgs.append("d_c_stride_1", (size_t)M);

        runtimeArgs.append("D", d_D.get());
        runtimeArgs.append("d_d_limit", (size_t)M * N);
        runtimeArgs.append("d_d_stride_0", (size_t)1);
        runtimeArgs.append("d_d_stride_1", (size_t)M);

        // TODO: remove this when for loop indexing is fixed
        command->allocateArgument(DataType::UInt32, DataDirection::ReadOnly, "UINT_MAT_K");
        runtimeArgs.append("UINT_MAT_K", static_cast<uint>(K));

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        // TODO: the translate step should figure out that there is a
        // T_Mul and do the right thing for the T_Load_Tiled commands
        auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile(
            0, {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_1 = KernelGraph::CoordinateTransform::MacroTile(
            1, {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2 = KernelGraph::CoordinateTransform::MacroTile(
            2, {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_3 = KernelGraph::CoordinateTransform::MacroTile(
            3, {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_4
            = KernelGraph::CoordinateTransform::MacroTile(4,
                                                          {mac_m, mac_n},
                                                          LayoutType::MATRIX_ACCUMULATOR,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);
        params->setDimensionInfo(mac_tile_3);
        params->setDimensionInfo(mac_tile_4);

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 1, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 1, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(2, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(2, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(2, 1, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(3, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(3, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(3, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(4, -1, four, one, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(4, 0, two, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(4, 1, two, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "ABC", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        std::vector<float> D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<float> c_D(M * N, 0.f);
        CPUMM(c_D, C, A, B, M, N, K, 1.0, 1.0, false);

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, 2.e-6);
    }

}
