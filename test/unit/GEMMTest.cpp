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
#include <rocRoller/Operations/Command.hpp>
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

        std::string transA = "N";
        std::string transB = "T";

        // Unroll Sizes
        unsigned int unrollX = 0;
        unsigned int unrollY = 0;
        unsigned int unrollK = 0;

        bool loadLDSA  = true;
        bool loadLDSB  = true;
        bool storeLDSD = true;

        bool fuseLoops      = true;
        bool literalStrides = true;
        bool prefetch       = false;

        bool packMultipleElementsInto1VGPR = true;
    };

    template <typename T>
    void basicGEMM(std::shared_ptr<Context>& m_context,
                   const GEMMProblem&        gemm,
                   double                    acceptableError,
                   bool                      debuggable = false)
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

        if(gemm.unrollK > 0)
        {
            AssertFatal(K % (mac_k * gemm.unrollK) == 0, "MacroTile size mismatch (K unroll)");
        }

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
        std::vector<T> h_A;
        std::vector<T> h_B;
        std::vector<T> h_C;

        GenerateRandomInput(31415u, h_A, M * K, h_B, K * N, h_C, M * N);

        if(debuggable)
        {
            SetIdentityMatrix(h_A, K, M);
            SetIdentityMatrix(h_B, N, K);

            std::fill(h_C.begin(), h_C.end(), static_cast<T>(0.0));
        }

        // Device data.  TODO Remove padding when prefetch fixed.
        std::shared_ptr<T> d_A = make_shared_device(h_A, mac_k * M);
        std::shared_ptr<T> d_B = make_shared_device(h_B, mac_k * N);
        std::shared_ptr<T> d_C = make_shared_device(h_C);
        std::shared_ptr<T> d_D = make_shared_device<T>(M * N, 0.0);

        auto command  = std::make_shared<Command>();
        auto dataType = TypeInfo<T>::Var.dataType;

        std::vector<size_t> oneStridesN
            = gemm.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

        std::vector<size_t> oneStridesT = gemm.literalStrides
                                              ? std::vector<size_t>({(size_t)0, (size_t)1})
                                              : std::vector<size_t>({});

        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                dataType, 2, 0, gemm.transA == "N" ? oneStridesN : oneStridesT))); // A
        command->addOperation(
            std::make_shared<rocRoller::Operations::Operation>(rocRoller::Operations::T_Load_Tiled(
                dataType, 2, 1, gemm.transB == "N" ? oneStridesN : oneStridesT))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 2, oneStridesN))); // C
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
        if(gemm.transA == "N")
        {
            runtimeArgs.append("d_a_stride_0", (size_t)1);
            runtimeArgs.append("d_a_stride_1", (size_t)M);
        }
        else
        {
            runtimeArgs.append("d_a_stride_0", (size_t)K);
            runtimeArgs.append("d_a_stride_1", (size_t)1);
        }

        runtimeArgs.append("B", d_B.get());
        runtimeArgs.append("d_b_limit", (size_t)K * N);
        runtimeArgs.append("d_b_size_0", (size_t)K);
        runtimeArgs.append("d_b_size_1", (size_t)N);
        if(gemm.transB == "N")
        {
            runtimeArgs.append("d_b_stride_0", (size_t)1);
            runtimeArgs.append("d_b_stride_1", (size_t)K);
        }
        else
        {
            runtimeArgs.append("d_b_stride_0", (size_t)N);
            runtimeArgs.append("d_b_stride_1", (size_t)1);
        }

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

        auto kernelOptions                           = std::make_shared<KernelOptions>();
        kernelOptions->fuseLoops                     = gemm.fuseLoops;
        kernelOptions->unrollX                       = gemm.unrollX;
        kernelOptions->unrollY                       = gemm.unrollY;
        kernelOptions->unrollK                       = gemm.unrollK;
        kernelOptions->packMultipleElementsInto1VGPR = gemm.packMultipleElementsInto1VGPR;
        kernelOptions->prefetch                      = gemm.prefetch;
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A] = gemm.transA == "T";
        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B] = gemm.transB == "T";

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        // TODO: Calculate these values internally based on workgroup sizes.
        params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

        auto mac_tile_A = KernelGraph::CoordinateGraph::MacroTile({mac_m, mac_k},
                                                                  LayoutType::MATRIX_A,
                                                                  {wave_m, wave_n, wave_k, wave_b},
                                                                  gemm.loadLDSA ? MemoryType::LDS
                                                                                : MemoryType::WAVE);
        auto mac_tile_B = KernelGraph::CoordinateGraph::MacroTile({mac_k, mac_n},
                                                                  LayoutType::MATRIX_B,
                                                                  {wave_m, wave_n, wave_k, wave_b},
                                                                  gemm.loadLDSB ? MemoryType::LDS
                                                                                : MemoryType::WAVE);
        auto mac_tile_C = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_D = KernelGraph::CoordinateGraph::MacroTile(
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b},
            gemm.storeLDSD ? MemoryType::LDS : MemoryType::WAVE);

        params->setDimensionInfo(4, mac_tile_A);
        params->setDimensionInfo(11, mac_tile_B);
        params->setDimensionInfo(18, mac_tile_C);
        params->setDimensionInfo(28, mac_tile_C);
        params->setDimensionInfo(30, mac_tile_C);
        params->setDimensionInfo(32, mac_tile_C);
        params->setDimensionInfo(34, mac_tile_D);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        auto postParams = std::make_shared<CommandParameters>();

        auto one         = Expression::literal(1u);
        auto wavefront_n = Expression::literal(static_cast<uint>(
            mac_m * mac_n / wave_m / wave_n / wavetile_per_wavefront_m / wavetile_per_wavefront_n));
        auto wavefront_nx
            = Expression::literal(static_cast<uint>(mac_m / wave_m / wavetile_per_wavefront_m));
        auto wavefront_ny
            = Expression::literal(static_cast<uint>(mac_n / wave_n / wavetile_per_wavefront_n));

        auto WF  = KernelGraph::CoordinateGraph::Wavefront(-1, wavefront_n, one);
        auto WFX = KernelGraph::CoordinateGraph::Wavefront(0, wavefront_nx, one);
        auto WFY = KernelGraph::CoordinateGraph::Wavefront(1, wavefront_ny, one);

        std::vector<int> wavefront_ids = {58, 91, 124, 173};
        for(auto id : wavefront_ids)
        {
            postParams->setDimensionInfo(id, WF);
            postParams->setDimensionInfo(id - 2, WFX);
            postParams->setDimensionInfo(id - 1, WFY);
        }

        CommandKernel commandKernel(command, "GEMMTest", params, postParams, kernelOptions);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        m_context = commandKernel.getContext();

        // Device result
        std::vector<T> d_result(M * N, 0.0);
        ASSERT_THAT(hipMemcpy(d_result.data(), d_D.get(), M * N * sizeof(T), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

        // Host result
        std::vector<T> h_result(M * N, 0.0);
        rocRoller::CPUMM(
            h_result, h_C, h_A, h_B, M, N, K, alpha, beta, gemm.transA == "T", gemm.transB == "T");

        if(debuggable)
        {
            for(size_t i = 0; i < 32; i++)
            {
                for(size_t j = 0; j < 32; j++)
                    std::cout << d_result[i * N + j] << ", ";

                std::cout << std::endl;
            }
        }
        double rnorm = relativeNorm(d_result, h_result);

        ASSERT_LT(rnorm, acceptableError);
    }

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_F(GEMMTestGPU, GPU_BasicGEMM_Schedulers)
    {
        GEMMProblem gemm;
        gemm.mac_k = 8;

        // TODO: Re-enable LDS once LDS deallocations are fixed
        gemm.loadLDSA = false;
        gemm.loadLDSB = false;

        auto settings = Settings::getInstance();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Sequential);
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string seq = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::RoundRobin);
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string rr = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Cooperative);
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string coop_nop = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Priority);
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string priority_nop = m_context->instructions()->toString();

        EXPECT_NE(NormalizedSource(seq), NormalizedSource(rr));

        EXPECT_NE(NormalizedSource(coop_nop), NormalizedSource(rr));

        EXPECT_NE(NormalizedSource(priority_nop), NormalizedSource(rr));

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

    TEST_F(GEMMTestGPU, GPU_BasicGEMMNoLDSA)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = true;
        gemm.fuseLoops = false;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMNoLDSB)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = false;
        gemm.fuseLoops = false;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMNoLDSAB)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = false;
        gemm.fuseLoops = false;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollK)
    {
        GEMMProblem gemm;
        gemm.K         = 64 * 4 * 2;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 4;
        gemm.mac_k     = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKLDS)
    {
        GEMMProblem gemm;
        gemm.K         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 2;
        gemm.mac_k     = 4;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDS)
    {
        GEMMProblem gemm;
        gemm.K         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.mac_k     = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSA)
    {
        GEMMProblem gemm;
        gemm.K         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.mac_k     = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSB)
    {
        GEMMProblem gemm;
        gemm.K         = 64 * 4 * 2;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.mac_k     = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSPrefetch)
    {
        GEMMProblem gemm;
        gemm.K         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 2;
        gemm.mac_k     = 4;
        gemm.prefetch  = true;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16)
    {
        GEMMProblem gemm;
        gemm.wave_k = 8;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X2UnrollX)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.unrollX = 2;
        gemm.unrollY = 1;

        gemm.loadLDSA  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X2UnrollY)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.unrollX = 1;
        gemm.unrollY = 2;

        gemm.loadLDSA  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X2)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.loadLDSA  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X1)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 128;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.transA = "T";
        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 2);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X1UnrollK)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 128;
        gemm.mac_k = 16;

        gemm.unrollK = 2;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.transA = "T";
        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 4);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1X2)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 128;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 4 * gemm.wavefront_size;
        gemm.workgroup_size_y = 2;

        gemm.transA = "T";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 2);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1X2UnrollK)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 128;
        gemm.mac_k = 16;

        gemm.unrollK = 4;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 4 * gemm.wavefront_size;
        gemm.workgroup_size_y = 2;

        gemm.transA = "T";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 8);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1x8)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 4 * gemm.wavefront_size;
        gemm.workgroup_size_y = 1;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1x8UnrollK)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.unrollK = 4;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 4 * gemm.wavefront_size;
        gemm.workgroup_size_y = 1;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }
    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2x4)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 2;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 3);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2x4UnrollK)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.unrollK = 4;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 2 * gemm.wavefront_size;
        gemm.workgroup_size_y = 2;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 12);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed4x2)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 1 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.storeLDSD = false;

        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 3);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed4x2UnrollK)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.unrollK = 4;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 1 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.storeLDSD = false;

        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 12);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMLiteralStrides)
    {
        GEMMProblem gemm;
        gemm.packMultipleElementsInto1VGPR = true;
        gemm.transB                        = "N";

        gemm.literalStrides = true;
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string output_literalStrides = m_context->instructions()->toString();

        gemm.literalStrides = false;
        basicGEMM<float>(m_context, gemm, 1.e-6);
        std::string output_noLiteralStrides = m_context->instructions()->toString();

        //Since we're setting the first dimension to a literal 1, there will be less occurrences of Load_Tiled_0_stride_0.
        EXPECT_LT(countSubstring(output_literalStrides, "Load_Tiled_0_stride_0"),
                  countSubstring(output_noLiteralStrides, "Load_Tiled_0_stride_0"));
        EXPECT_LT(countSubstring(output_literalStrides, "Load_Tiled_1_stride_0"),
                  countSubstring(output_noLiteralStrides, "Load_Tiled_1_stride_0"));
        EXPECT_LT(countSubstring(output_literalStrides, "Load_Tiled_2_stride_0"),
                  countSubstring(output_noLiteralStrides, "Load_Tiled_2_stride_0"));

        //Since we're not setting the second dimension to a literal, there will be the same occurrences of Load_Tiled_X_stride_1.
        EXPECT_EQ(countSubstring(output_literalStrides, "Load_Tiled_0_stride_1"),
                  countSubstring(output_noLiteralStrides, "Load_Tiled_0_stride_1"));
        EXPECT_EQ(countSubstring(output_literalStrides, "Load_Tiled_1_stride_1"),
                  countSubstring(output_noLiteralStrides, "Load_Tiled_1_stride_1"));
        EXPECT_EQ(countSubstring(output_literalStrides, "Load_Tiled_2_stride_1"),
                  countSubstring(output_noLiteralStrides, "Load_Tiled_2_stride_1"));
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16AllLDS)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 1 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16AllLDSDebug)
    {
        GEMMProblem gemm;

        gemm.M = 256;
        gemm.N = 512;
        gemm.K = 64;

        gemm.mac_m = 128;
        gemm.mac_n = 256;
        gemm.mac_k = 16;

        gemm.wave_k = 8;

        gemm.workgroup_size_x = 1 * gemm.wavefront_size;
        gemm.workgroup_size_y = 4;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        basicGEMM<Half>(m_context, gemm, 2.e-5, true);
    }
}
