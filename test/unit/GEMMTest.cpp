#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

namespace GEMMDriverTest
{
    class GEMMTestGPU : public CurrentGPUContextFixture
    {
    };

    struct GEMMProblem
    {
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

        // Wave tile size
        int wave_m = 32;
        int wave_n = 32;
        int wave_k = 2;
        int wave_b = 1;

        // Workgroup size
        uint wavefront_size   = 64;
        uint workgroup_size_x = 2 * wavefront_size;
        uint workgroup_size_y = 2;
    };

    template <typename T>
    void basicGEMM(std::shared_ptr<Context>& m_context,
                   const GEMMProblem&        gemm,
                   double                    acceptableError)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
        int   M     = gemm.M;
        int   N     = gemm.N;
        int   K     = gemm.K;
        float alpha = gemm.alpha;
        float beta  = gemm.beta;

        // output macro tile size
        int mac_m = gemm.mac_m;
        int mac_n = gemm.mac_n;
        int mac_k = gemm.mac_k;

        int wave_m = gemm.wave_m;
        int wave_n = gemm.wave_n;
        int wave_k = gemm.wave_k;
        int wave_b = gemm.wave_b;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        AssertFatal(gemm.workgroup_size_x % gemm.wavefront_size == 0,
                    "Workgroup Size X must be multiply of wave front size");

        uint wavetile_per_wavefront_m
            = gemm.wavefront_size * mac_m / wave_m / gemm.workgroup_size_x;
        uint wavetile_per_wavefront_n = mac_n / wave_n / gemm.workgroup_size_y;

        AssertFatal(mac_m % (wave_m * wavetile_per_wavefront_m) == 0, "WaveTile size mismatch (M)");
        AssertFatal(mac_n % (wave_n * wavetile_per_wavefront_n) == 0, "WaveTile size mismatch (N)");

        uint workgroup_size_x = gemm.workgroup_size_x * gemm.workgroup_size_y;
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

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        // TODO: Calculate these values internally based on workgroup sizes.
        params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

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

        auto one         = Expression::literal(1u);
        auto wavefront_n = Expression::literal(static_cast<uint>(
            mac_m * mac_n / wave_m / wave_n / wavetile_per_wavefront_m / wavetile_per_wavefront_n));
        auto wavefront_nx
            = Expression::literal(static_cast<uint>(mac_m / wave_m / wavetile_per_wavefront_m));
        auto wavefront_ny
            = Expression::literal(static_cast<uint>(mac_n / wave_n / wavetile_per_wavefront_n));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(0, -1, wavefront_n, nullptr)); // A Load
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(0, 0, wavefront_nx, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(0, 1, wavefront_ny, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(1, -1, wavefront_n, nullptr)); // B Load
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(1, 0, wavefront_nx, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(1, 1, wavefront_ny, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, -1, wavefront_n, nullptr)); // C Load
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 0, wavefront_nx, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 1, wavefront_ny, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(8, -1, wavefront_n, one, true)); // D
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(8, 0, wavefront_nx, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(8, 1, wavefront_ny, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        // Execute the GEMM kernel
        CommandKernel commandKernel(command, "GEMM", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        m_context = commandKernel.getContext();

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

    template <typename T>
    void basicGEMM2(std::shared_ptr<Context>& m_context,
                    const GEMMProblem&        gemm,
                    double                    acceptableError)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

        // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
        int   M     = gemm.M;
        int   N     = gemm.N;
        int   K     = gemm.K;
        float alpha = gemm.alpha;
        float beta  = gemm.beta;

        // output macro tile size
        int mac_m = gemm.mac_m;
        int mac_n = gemm.mac_n;
        int mac_k = gemm.mac_k;

        int wave_m = gemm.wave_m;
        int wave_n = gemm.wave_n;
        int wave_k = gemm.wave_k;
        int wave_b = gemm.wave_b;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        AssertFatal(gemm.workgroup_size_x % gemm.wavefront_size == 0,
                    "Workgroup Size X must be multiply of wave front size");

        uint wavetile_per_wavefront_m
            = gemm.wavefront_size * mac_m / wave_m / gemm.workgroup_size_x;
        uint wavetile_per_wavefront_n = mac_n / wave_n / gemm.workgroup_size_y;

        AssertFatal(mac_m % (wave_m * wavetile_per_wavefront_m) == 0, "WaveTile size mismatch (M)");
        AssertFatal(mac_n % (wave_n * wavetile_per_wavefront_n) == 0, "WaveTile size mismatch (N)");

        uint workgroup_size_x = gemm.workgroup_size_x * gemm.workgroup_size_y;
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

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        // TODO: Calculate these values internally based on workgroup sizes.
        params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

        auto mac_tile_A = KernelGraph::CoordGraph::MacroTile(
            {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_B = KernelGraph::CoordGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_C = KernelGraph::CoordGraph::MacroTile(
            {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(4, mac_tile_A);
        params->setDimensionInfo(11, mac_tile_B);
        params->setDimensionInfo(18, mac_tile_C);
        params->setDimensionInfo(30, mac_tile_C);
        params->setDimensionInfo(32, mac_tile_C);
        params->setDimensionInfo(34, mac_tile_C);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        auto params2 = std::make_shared<CommandParameters>();

        auto one         = Expression::literal(1u);
        auto wavefront_n = Expression::literal(static_cast<uint>(
            mac_m * mac_n / wave_m / wave_n / wavetile_per_wavefront_m / wavetile_per_wavefront_n));
        auto wavefront_nx
            = Expression::literal(static_cast<uint>(mac_m / wave_m / wavetile_per_wavefront_m));
        auto wavefront_ny
            = Expression::literal(static_cast<uint>(mac_n / wave_n / wavetile_per_wavefront_n));

        auto WF  = KernelGraph::CoordGraph::Wavefront(-1, wavefront_n, one);
        auto WFX = KernelGraph::CoordGraph::Wavefront(0, wavefront_nx, one);
        auto WFY = KernelGraph::CoordGraph::Wavefront(1, wavefront_ny, one);

        params2->setDimensionInfo(59, WF); // A
        params2->setDimensionInfo(57, WFX);
        params2->setDimensionInfo(58, WFY);
        params2->setDimensionInfo(94, WF); // B
        params2->setDimensionInfo(92, WFX);
        params2->setDimensionInfo(93, WFY);
        params2->setDimensionInfo(129, WF); // C
        params2->setDimensionInfo(127, WFX);
        params2->setDimensionInfo(128, WFY);
        params2->setDimensionInfo(184, WF); // D
        params2->setDimensionInfo(182, WFX);
        params2->setDimensionInfo(183, WFY);

        auto context = m_context;
        context->kernel()->setKernelDimensions(2);
        context->kernel()->setWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        context->kernel()->setWorkitemCount({NX, NY, NZ});

        context->kernel()->addCommandArguments(command->getArguments());
        auto kgraph = KernelGraph::translate2(command);
        kgraph      = KernelGraph::updateParameters(kgraph, params);
        kgraph      = KernelGraph::lowerTile(kgraph, params, context);
        kgraph      = KernelGraph::cleanArguments(kgraph, context->kernel());
        kgraph      = KernelGraph::updateParameters(kgraph, params2);

        context->schedule(context->kernel()->preamble());
        context->schedule(context->kernel()->prolog());
        context->schedule(KernelGraph::generate(kgraph, context->kernel()));
        context->schedule(context->kernel()->postamble());
        context->schedule(context->kernel()->amdgpu_metadata());

        auto executableKernel = context->instructions()->getExecutableKernel();

        KernelArguments kargs;
        for(auto& arg : context->kernel()->arguments())
        {
            auto value = evaluate(arg.expression, runtimeArgs.runtimeArguments());
            kargs.append(arg.name, value);
        }

        KernelInvocation kinv;
        kinv.workgroupSize    = context->kernel()->workgroupSize();
        kinv.workitemCount[0] = num_workgroup_x * workgroup_size_x;
        kinv.workitemCount[1] = num_workgroup_y * workgroup_size_y;

        executableKernel->executeKernel(kargs, kinv);

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

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_F(GEMMTestGPU, GPU_BasicGEMM_Schedulers)
    {
        GEMMProblem gemm;
        auto        settings = Settings::getInstance();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Sequential);
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string seq = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::RoundRobin);
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string rr = m_context->instructions()->toString();

        EXPECT_NE(NormalizedSource(seq), NormalizedSource(rr));

        std::set<std::string> insts;
        std::vector<int>      seeds = {2, 4, 8, 314, 1729};
        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Random);
        for(auto seed : seeds)
        {
            settings->set(Settings::RandomSeed, seed);
            basicGEMM<float>(m_context, gemm, 1.e-6);
            std::string rand     = m_context->instructions()->toString();
            bool        not_seen = insts.insert(rand).second;
            EXPECT_EQ(not_seen, true);
        }
        // Can not compare random insts to others because non-zero chance seed generates such insts
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM)
    {
        GEMMProblem gemm;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM2)
    {
        GEMMProblem gemm;
        basicGEMM2<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16)
    {
        GEMMProblem gemm;
        gemm.wave_k = 8;
        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM2FP16)
    {
        GEMMProblem gemm;
        gemm.wave_k = 8;
        basicGEMM2<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Unroll2x4)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 16;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM2FP16Unroll2x4)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 16;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        basicGEMM2<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Unroll8x1)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 16;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 4 * gemm.wavefront_size;
        gemm.workgroup_size_y = 1;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM2FP16Unroll8x1)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 16;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 4 * gemm.wavefront_size;
        gemm.workgroup_size_y = 1;

        basicGEMM2<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Unroll1x8)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 16;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 1 * gemm.wavefront_size;
        gemm.workgroup_size_y = 8;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMM2FP16Unroll1x8)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 16;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 1 * gemm.wavefront_size;
        gemm.workgroup_size_y = 8;

        basicGEMM2<Half>(m_context, gemm, 2.e-5);
    }
}
