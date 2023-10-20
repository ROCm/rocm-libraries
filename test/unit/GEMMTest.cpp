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
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

namespace GEMMDriverTest
{
    struct GEMMProblem
    {
        // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
        int   m     = 512;
        int   n     = 512;
        int   k     = 128;
        float alpha = 2.0f;
        float beta  = 0.5f;

        // output macro tile size
        int macM = 64;
        int macN = 64;
        int macK = 64;

        // Wave tile size
        int waveM = 32;
        int waveN = 32;
        int waveK = 2;
        int waveB = 1;

        // Workgroup size
        uint wavefrontSize  = 64;
        uint workgroupSizeX = 2 * wavefrontSize;
        uint workgroupSizeY = 2;

        std::string transA = "N";
        std::string transB = "T";

        // Unroll Sizes
        unsigned int unrollX = 0;
        unsigned int unrollY = 0;
        unsigned int unrollK = 0;

        bool loadLDSA  = true;
        bool loadLDSB  = true;
        bool storeLDSD = true;

        bool fuseLoops                 = true;
        bool allowAmbiguousMemoryNodes = false;
        bool betaInFma                 = true;
        bool literalStrides            = true;

        bool prefetch          = false;
        int  prefetchInFlight  = 1;
        int  prefetchLDSFactor = 0;
        bool prefetchMixMemOps = false;

        bool packMultipleElementsInto1VGPR = true;

        bool loopOverTiles = false;
        bool streamK       = false;
    };

    struct GEMMTestGPU : public CurrentGPUContextFixture
    {
        template <typename T>
        void basicGEMM(ContextPtr&        m_context,
                       const GEMMProblem& gemm,
                       double             acceptableError,
                       bool               debuggable  = false,
                       bool               setIdentity = false)

        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

            hipDeviceProp_t deviceProperties;
            ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
            uint numCUs = deviceProperties.multiProcessorCount;

            // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
            int   M     = gemm.m;
            int   N     = gemm.n;
            int   K     = gemm.k;
            float alpha = gemm.alpha;
            float beta  = gemm.beta;

            AssertFatal(M % gemm.macM == 0, "MacroTile size mismatch (M)");
            AssertFatal(N % gemm.macN == 0, "MacroTile size mismatch (N)");

            if(gemm.unrollK > 0)
            {
                AssertFatal(K % (gemm.macK * gemm.unrollK) == 0,
                            "MacroTile size mismatch (K unroll)");
            }

            AssertFatal(gemm.workgroupSizeX % gemm.wavefrontSize == 0,
                        "Workgroup Size X must be multiply of wave front size");

            uint wavetilePerWavefrontM
                = gemm.wavefrontSize * gemm.macM / gemm.waveM / gemm.workgroupSizeX;
            uint wavetilePerWavefrontN = gemm.macN / gemm.waveN / gemm.workgroupSizeY;

            AssertFatal(gemm.macM % (gemm.waveM * wavetilePerWavefrontM) == 0,
                        "WaveTile size mismatch (M)");
            AssertFatal(gemm.macN % (gemm.waveN * wavetilePerWavefrontN) == 0,
                        "WaveTile size mismatch (N)");

            uint workgroupSizeX = gemm.workgroupSizeX * gemm.workgroupSizeY;
            uint workgroupSizeY = 1;

            uint numWorkgroupX;
            uint numWorkgroupY;

            if(gemm.loopOverTiles > 0)
            {
                // multiple output macro tiles per workgroup
                numWorkgroupX = M * N / gemm.macM / gemm.macN / 2;
                numWorkgroupY = 1;
            }
            else if(gemm.streamK)
            {
                numWorkgroupX = numCUs;
                numWorkgroupY = 1;
            }
            else
            {
                // one output macro tile per workgroup
                numWorkgroupX = M / gemm.macM;
                numWorkgroupY = N / gemm.macN;
            }

            auto NX = std::make_shared<Expression::Expression>(numWorkgroupX * workgroupSizeX);
            auto NY = std::make_shared<Expression::Expression>(numWorkgroupY * workgroupSizeY);
            auto NZ = std::make_shared<Expression::Expression>(1u);

            // Host data
            std::vector<T> hostA;
            std::vector<T> hostB;
            std::vector<T> hostC;

            GenerateRandomInput(31415u, hostA, M * K, hostB, K * N, hostC, M * N);

            if(setIdentity)
            {
                SetIdentityMatrix(hostA, K, M);
                SetIdentityMatrix(hostB, N, K);

                std::fill(hostC.begin(), hostC.end(), static_cast<T>(0.0));
            }

            std::shared_ptr<T> deviceA = make_shared_device(hostA);
            std::shared_ptr<T> deviceB = make_shared_device(hostB);
            std::shared_ptr<T> deviceC = make_shared_device(hostC);
            std::shared_ptr<T> deviceD = make_shared_device<T>(M * N, 0.0);

            auto command  = std::make_shared<Command>();
            auto dataType = TypeInfo<T>::Var.dataType;

            std::vector<size_t> oneStridesN
                = gemm.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

            std::vector<size_t> oneStridesT = gemm.literalStrides
                                                  ? std::vector<size_t>({(size_t)0, (size_t)1})
                                                  : std::vector<size_t>({});

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
                rocRoller::Operations::T_Load_Tiled(
                    dataType, 2, 0, gemm.transA == "N" ? oneStridesN : oneStridesT))); // A
            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
                rocRoller::Operations::T_Load_Tiled(
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
                rocRoller::Operations::E_Mul(6, 4, 2))); // beta * C
            execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
                rocRoller::Operations::E_Mul(7, 3, 5))); // alpha * (A * B)
            if(gemm.betaInFma)
            {
                execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
                    rocRoller::Operations::E_Add(8, 6, 7))); // beta * C + alpha * (A * B)
            }
            else
            {
                execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
                    rocRoller::Operations::E_Add(8, 7, 6))); // alpha * (A * B) + beta * C
            }

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
                rocRoller::Operations::T_Store_Tiled(dataType, 2, 8))); // D

            KernelArguments runtimeArgs;

            runtimeArgs.append("A", deviceA.get());
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

            runtimeArgs.append("B", deviceB.get());
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

            runtimeArgs.append("C", deviceC.get());
            runtimeArgs.append("d_c_limit", (size_t)M * N);
            runtimeArgs.append("d_c_size_0", (size_t)M);
            runtimeArgs.append("d_c_size_1", (size_t)N);
            runtimeArgs.append("d_c_stride_0", (size_t)1);
            runtimeArgs.append("d_c_stride_1", (size_t)M);

            runtimeArgs.append("alpha", alpha);

            runtimeArgs.append("beta", beta);

            runtimeArgs.append("D", deviceD.get());
            runtimeArgs.append("d_d_limit", (size_t)M * N);
            runtimeArgs.append("d_d_stride_0", (size_t)1);
            runtimeArgs.append("d_d_stride_1", (size_t)M);

            auto kernelOptions                           = std::make_shared<KernelOptions>();
            kernelOptions->fuseLoops                     = gemm.fuseLoops;
            kernelOptions->allowAmbiguousMemoryNodes     = gemm.allowAmbiguousMemoryNodes;
            kernelOptions->unrollX                       = gemm.unrollX;
            kernelOptions->unrollY                       = gemm.unrollY;
            kernelOptions->unrollK                       = gemm.unrollK;
            kernelOptions->packMultipleElementsInto1VGPR = gemm.packMultipleElementsInto1VGPR;
            kernelOptions->prefetch                      = gemm.prefetch;
            kernelOptions->prefetchInFlight              = gemm.prefetchInFlight;
            kernelOptions->prefetchLDSFactor             = gemm.prefetchLDSFactor;
            kernelOptions->prefetchMixMemOps             = gemm.prefetchMixMemOps;
            kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A] = gemm.transA == "T";
            kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B] = gemm.transB == "T";

            if(gemm.loopOverTiles > 0)
            {
                kernelOptions->loopOverOutputTilesDimensions = {0, 1};
                kernelOptions->loopOverOutputTilesCoordSizes
                    = {static_cast<uint>(M / gemm.macM), static_cast<uint>(N / gemm.macN)};
                kernelOptions->loopOverOutputTilesIteratedTiles = 2;
            }

            if(gemm.streamK)
            {
                REQUIRE_ARCH_CAP(GPUCapability::ArchAccUnifiedRegs);

                AssertFatal(
                    numWorkgroupY == 1,
                    "Current scratch space implementation assumes that the kernel is launched "
                    "with numWorkgroupY == 1");

                kernelOptions->numScratchTiles = std::min(numCUs, numWorkgroupX * numWorkgroupY);

                kernelOptions->loopOverOutputTilesDimensions = {0, 1};
                kernelOptions->loopOverOutputTilesCoordSizes
                    = {static_cast<uint>(M / gemm.macM), static_cast<uint>(N / gemm.macN)};
                kernelOptions->streamK = true;
            }

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            // TODO: Calculate these values internally based on workgroup sizes.
            params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);

            auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                {gemm.macM, gemm.macK},
                LayoutType::MATRIX_A,
                {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                gemm.loadLDSA ? MemoryType::LDS : MemoryType::WAVE);
            auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                {gemm.macK, gemm.macN},
                LayoutType::MATRIX_B,
                {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                gemm.loadLDSB ? MemoryType::LDS : MemoryType::WAVE);
            auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                {gemm.macM, gemm.macN},
                LayoutType::MATRIX_ACCUMULATOR,
                {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB});
            auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
                {gemm.macM, gemm.macN},
                LayoutType::MATRIX_ACCUMULATOR,
                {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                gemm.storeLDSD ? MemoryType::LDS : MemoryType::WAVE);

            params->setDimensionInfo(4, macTileA);
            params->setDimensionInfo(11, macTileB);
            params->setDimensionInfo(18, macTileC);
            params->setDimensionInfo(28, macTileC);
            params->setDimensionInfo(30, macTileC);
            params->setDimensionInfo(32, macTileC);
            params->setDimensionInfo(34, macTileD);

            params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});
            params->setManualWorkitemCount({NX, NY, NZ});

            rocRoller::Log::getLogger()->debug(
                "GEMM workgroup sizes {} {} {}", workgroupSizeX, workgroupSizeY, 1);
            rocRoller::Log::getLogger()->debug(
                "GEMM workitem counts {} {} {}", toString(NX), toString(NY), toString(NZ));

            auto postParams = std::make_shared<CommandParameters>();
            postParams->setManualWavefrontCount(
                {static_cast<uint>(gemm.macM / gemm.waveM / wavetilePerWavefrontM),
                 static_cast<uint>(gemm.macN / gemm.waveN / wavetilePerWavefrontN)});

            command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                      DataDirection::ReadWrite,
                                      rocRoller::SCRATCH);

            CommandKernel commandKernel(
                command, testKernelName(), params, postParams, kernelOptions);

            // Create scratch space
            auto scratchSpaceRequired = commandKernel.scratchSpaceRequired();
            auto deviceScratch        = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
            runtimeArgs.append(rocRoller::SCRATCH, static_cast<void*>(deviceScratch.get()));

            if(gemm.streamK)
            {
                runtimeArgs.append("numWGs", numCUs);
            }

            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
            m_context = commandKernel.getContext();

            // Device result
            std::vector<T> d_result(M * N, 0.0);
            ASSERT_THAT(
                hipMemcpy(d_result.data(), deviceD.get(), M * N * sizeof(T), hipMemcpyDeviceToHost),
                HasHipSuccess(0));

            // Host result
            std::vector<T> h_result(M * N, 0.0);
            rocRoller::CPUMM(h_result,
                             hostC,
                             hostA,
                             hostB,
                             M,
                             N,
                             K,
                             alpha,
                             beta,
                             gemm.transA == "T",
                             gemm.transB == "T");

            if(debuggable)
            {
                for(size_t i = 0; i < M; i++)
                {
                    for(size_t j = 0; j < N; j++)
                    {
                        auto a = d_result[i * N + j];
                        auto b = h_result[i * N + j];
                        if((a - b) * (a - b) / (b * b) > 10.0 * acceptableError)
                        {
                            std::cout << std::setw(8) << i << std::setw(8) << j << std::setw(16)
                                      << std::scientific << a << std::setw(16) << std::scientific
                                      << b << std::setw(16) << std::scientific << a - b
                                      << std::endl;
                        }
                    }
                }
            }
            double rnorm = relativeNorm(d_result, h_result);

            ASSERT_LT(rnorm, acceptableError);
        }
    };

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_F(GEMMTestGPU, GPU_BasicGEMM_Schedulers)
    {
        GEMMProblem gemm;
        gemm.macK = 8;

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

    TEST_F(GEMMTestGPU, GPU_BasicGEMMStreamK)
    {
        if(m_context->targetArchitecture().target().getVersionString() != "gfx90a")
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        uint numCUs = deviceProperties.multiProcessorCount;

        GEMMProblem gemm;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * numCUs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, numCUs);

        gemm.streamK = true;
        gemm.k       = gemm.macK * 8;

        // It works with prefetching too!!
        gemm.unrollK          = 2;
        gemm.prefetch         = true;
        gemm.prefetchInFlight = 2;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16StreamK)
    {
        if(m_context->targetArchitecture().target().getVersionString() != "gfx90a")
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        uint numCUs = deviceProperties.multiProcessorCount;

        GEMMProblem gemm;

        gemm.waveK = 8;
        gemm.macK  = 16;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * numCUs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, numCUs);

        gemm.streamK = true;
        gemm.k       = gemm.macK * 8;

        // It works with prefetching too!!
        gemm.unrollK          = 2;
        gemm.prefetch         = true;
        gemm.prefetchInFlight = 2;

        for(auto loadLDSA : {false, true})
        {
            gemm.loadLDSA = loadLDSA;
            for(auto loadLDSB : {false, true})
            {
                gemm.loadLDSB = loadLDSB;
                for(auto storeLDSD : {false, true})
                {
                    gemm.storeLDSD = storeLDSD;
                    basicGEMM<Half>(m_context, gemm, 2.e-5);
                }
            }
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMMultipleOutputTiles)
    {
        GEMMProblem gemm;
        gemm.storeLDSD     = false;
        gemm.loopOverTiles = true;
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
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 4;
        gemm.macK      = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKLDS)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDS)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSA)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSB)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSPrefetch)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        gemm.prefetch  = true;

        for(auto inflight : {1, 2})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<float>(m_context, gemm, 1.e-6);
                }
            }
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16UnrollKLDSPrefetch)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 16 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = true;
        gemm.unrollK   = 2;
        gemm.macM      = 64;
        gemm.macN      = 64;
        gemm.macK      = 16;
        gemm.prefetch  = true;
        gemm.waveK     = 8;

        for(auto inflight : {1, 2})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<Half>(m_context, gemm, 2.e-5);
                }
            }
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSMultiPrefetch)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 3;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 3;
        gemm.macK      = 4;
        gemm.prefetch  = true;

        for(auto inflight : {1, 2, 3})
        {
            gemm.prefetchInFlight = inflight;
            for(auto ldsFactor : {0, 2})
            {
                gemm.prefetchLDSFactor = ldsFactor;
                for(auto mixMemOps : {false, true})
                {
                    gemm.prefetchMixMemOps = mixMemOps;
                    basicGEMM<float>(m_context, gemm, 1.e-6);
                }
            }
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16)
    {
        GEMMProblem gemm;
        gemm.waveK = 8;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X2UnrollX)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

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

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

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

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadLDSA  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X1)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.betaInFma = false;

        gemm.transA = "T";
        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 2);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2X1UnrollK)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 16;

        gemm.unrollK = 2;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.transA = "T";
        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 4);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1X2)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 4 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.transA = "T";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 2);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1X2UnrollK)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 128;
        gemm.macK = 16;

        gemm.unrollK = 4;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 4 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.transA = "T";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 8);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1x8)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 4 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 1;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed1x8UnrollK)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.unrollK = 2;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 4 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 1;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }
    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2x4)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 3);
        EXPECT_EQ(countSubstring(generatedCode, "v_pack_B32_F16"), 24);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed2x4UnrollK)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.unrollK = 2;

        gemm.prefetchInFlight  = 2;
        gemm.prefetchLDSFactor = 2;
        gemm.prefetchMixMemOps = true;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.storeLDSD = false;

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed4x2)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.storeLDSD = false;

        gemm.transB = "N";

        basicGEMM<Half>(m_context, gemm, 2.e-5);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 3);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Jammed4x2UnrollK)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.unrollK = 4;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

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

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16AllLDSDebug)
    {
        GEMMProblem gemm;

        gemm.m = 256;
        gemm.n = 512;
        gemm.k = 64;

        gemm.macM = 128;
        gemm.macN = 256;
        gemm.macK = 16;

        gemm.waveK = 8;

        gemm.workgroupSizeX = 1 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        basicGEMM<Half>(m_context, gemm, 2.e-5, true);
    }
}
