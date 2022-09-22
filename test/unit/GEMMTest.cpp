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

namespace GEMMDriverTest
{
    class GEMMTestGPU : public CurrentGPUContextFixture
    {
    };

    template <typename T>
    void basicGEMM(std::shared_ptr<Context> m_context,
                   int                      wave_m,
                   int                      wave_n,
                   int                      wave_k,
                   int                      wave_b,
                   double                   acceptableError)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
        int   M     = 512;
        int   N     = 512;
        int   K     = 128;
        float alpha = 2.0f;
        float beta  = 0.5f;

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

        // Host data
        RandomGenerator random(31415u);
        std::vector<T>  h_A = random.vector<T>(M * K, -1.0, 1.0);
        std::vector<T>  h_B = random.vector<T>(K * N, -1.0, 1.0);
        std::vector<T>  h_C = random.vector<T>(M * N, -1.0, 1.0);

        // Device data
        std::shared_ptr<T> d_A = make_shared_device(h_A);
        std::shared_ptr<T> d_B = make_shared_device(h_B);
        std::shared_ptr<T> d_C = make_shared_device(h_C);
        std::shared_ptr<T> d_D = make_shared_device<T>(M * N, 0.0);

        auto command  = std::make_shared<Command>();
        auto dataType = TypeInfo<T>::Var.dataType;

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 2))); // C
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Float, 3))); // alpha
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Float, 4))); // beta

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(5, 0, 1))); // A * B

        rocRoller::Operations::T_Execute execute;
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(6, 3, 5))); // alpha * (A * B)
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(7, 4, 2))); // beta * C
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(8, 6, 7))); // alpha * (A * B) + beta * C
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 8))); // D

        KernelArguments runtimeArgs;

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

        runtimeArgs.append("alpha", alpha);

        runtimeArgs.append("beta", beta);

        runtimeArgs.append("D", d_D.get());
        runtimeArgs.append("d_d_limit", (size_t)M * N);
        runtimeArgs.append("d_d_stride_0", (size_t)1);
        runtimeArgs.append("d_d_stride_1", (size_t)M);

        // TODO: remove this when for loop indexing is fixed
        command->allocateArgument(DataType::UInt32, DataDirection::ReadOnly, "UINT_MAT_K");
        runtimeArgs.append("UINT_MAT_K", static_cast<uint>(K));

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile( // A
            0,
            {mac_m, mac_k},
            LayoutType::MATRIX_A,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_1 = KernelGraph::CoordinateTransform::MacroTile( // B
            1,
            {mac_k, mac_n},
            LayoutType::MATRIX_B,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2 = KernelGraph::CoordinateTransform::MacroTile( // C
            2,
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_5 = KernelGraph::CoordinateTransform::MacroTile( // A * B
            5,
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_6 = KernelGraph::CoordinateTransform::MacroTile( // alpha * (A * B)
            6,
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_7 = KernelGraph::CoordinateTransform::MacroTile( // beta * C
            7,
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_8 = KernelGraph::CoordinateTransform::MacroTile( // D
            8,
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b},
            true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);
        params->setDimensionInfo(mac_tile_5);
        params->setDimensionInfo(mac_tile_6);
        params->setDimensionInfo(mac_tile_7);
        params->setDimensionInfo(mac_tile_8);

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(0, -1, four, nullptr)); // A Load
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(1, -1, four, nullptr)); // B Load
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(1, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, -1, four, nullptr)); // C Load
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(2, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(2, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(8, -1, four, one, true)); // D
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(8, 0, two, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(8, 1, two, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        // Execute the GEMM kernel
        CommandKernel commandKernel(command, "GEMM", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        // Device result
        std::vector<T> d_result(M * N, 0.0);
        ASSERT_THAT(hipMemcpy(d_result.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

        // Host result
        std::vector<T> h_result(M * N, 0.0);
        rocRoller::CPUMM(h_result, h_C, h_A, h_B, M, N, K, alpha, beta, false);

        double rnorm = relativeNorm(d_result, h_result);

        ASSERT_LT(rnorm, acceptableError);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM)
    {
        basicGEMM<float>(m_context, 32, 32, 2, 1, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16)
    {
        basicGEMM<Half>(m_context, 32, 32, 8, 1, 2.e-5);
    }
}
