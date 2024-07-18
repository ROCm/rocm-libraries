#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <regex>

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

#include "GEMMProblem.hpp"
#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

namespace GEMMDriverTest
{
    struct GEMMTestGPU : public CurrentGPUContextFixture
    {
        template <typename T, typename TD = T>
        void basicGEMM(ContextPtr&        m_context,
                       const GEMMProblem& gemm,
                       double             acceptableError,
                       bool               debuggable  = false,
                       bool               setIdentity = false,
                       int                numIters    = 1,
                       bool               notSetC     = false)

        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
            if constexpr(std::is_same_v<T, FP8> || std::is_same_v<T, BF8>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
            }

            if constexpr(std::is_same_v<T, FP4> || std::is_same_v<T, FP6> || std::is_same_v<T, BF6>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            auto dataTypeAB = TypeInfo<T>::Var.dataType;
            auto dataTypeD  = TypeInfo<TD>::Var.dataType;

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

            auto bpe = DataTypeInfo::Get(dataTypeAB).elementBytes;
            AssertFatal(gemm.macM * gemm.macK * bpe > gemm.waveM * gemm.waveK,
                        "Not enough elements (A).");
            AssertFatal(gemm.macN * gemm.macK * bpe > gemm.waveN * gemm.waveK,
                        "Not enough elements (B).");

            AssertFatal(gemm.workgroupSizeX % gemm.wavefrontSize == 0,
                        "Workgroup Size X must be multiply of wave front size");

            uint wavetilePerWavefrontM
                = gemm.wavefrontSize * gemm.macM / gemm.waveM / gemm.workgroupSizeX;
            uint wavetilePerWavefrontN = gemm.macN / gemm.waveN / gemm.workgroupSizeY;

            AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch (M).");
            AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch (N).");

            AssertFatal(gemm.macM % (gemm.waveM * wavetilePerWavefrontM) == 0,
                        "WaveTile size mismatch (M)");
            AssertFatal(gemm.macN % (gemm.waveN * wavetilePerWavefrontN) == 0,
                        "WaveTile size mismatch (N)");

            Log::debug("GEMMTest jamming: {}x{}", wavetilePerWavefrontM, wavetilePerWavefrontN);

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
                numWorkgroupX = gemm.numCUs;
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
            using UnsegmentedType = typename UnsegmentedTypeOf<T>::type;
            std::vector<UnsegmentedType> hostA;
            std::vector<UnsegmentedType> hostB;
            std::vector<TD>              hostC;

            GenerateRandomInput(31415u, hostA, M * K, hostB, K * N, hostC, M * N);

            if(setIdentity)
            {
                SetIdentityMatrix(hostA, K, M);
                SetIdentityMatrix(hostB, N, K);

                std::fill(hostC.begin(), hostC.end(), static_cast<T>(0.0));
            }

            auto                deviceA = make_shared_device(hostA);
            auto                deviceB = make_shared_device(hostB);
            std::shared_ptr<TD> deviceC = (notSetC) ? nullptr : make_shared_device(hostC);
            std::shared_ptr<TD> deviceD = make_shared_device<TD>(M * N, TD{});

            auto command  = std::make_shared<Command>();
            auto dataType = TypeInfo<T>::Var.dataType;

            std::vector<size_t> oneStridesN
                = gemm.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

            std::vector<size_t> oneStridesT = gemm.literalStrides
                                                  ? std::vector<size_t>({(size_t)0, (size_t)1})
                                                  : std::vector<size_t>({});

            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeAB, gemm.transA == "N" ? oneStridesN : oneStridesT)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeAB, gemm.transB == "N" ? oneStridesN : oneStridesT)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagTensorC = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeD, oneStridesN)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagScalarAlpha
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
            auto tagLoadAlpha
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

            auto tagScalarBeta
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // beta
            auto tagLoadBeta
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta));

            auto tagAB
                = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

            rocRoller::Operations::T_Execute execute(command->getNextTag());
            auto                             tagBetaC
                = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, tagLoadC)); // beta * C

            auto tagAlphaAB = execute.addXOp(
                rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)

            rocRoller::Operations::OperationTag tagStoreD;
            if(gemm.betaInFma)
            {
                tagStoreD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
            }
            else
            {
                tagStoreD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
            }

            command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

            auto tagTensorD = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeD, oneStridesN)); // D
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

            auto tagScratch = command->allocateTag();
            command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                      tagScratch,
                                      ArgumentType::Value,
                                      DataDirection::ReadWrite,
                                      rocRoller::SCRATCH);
            auto kernelOptions                           = std::make_shared<KernelOptions>();
            kernelOptions->fuseLoops                     = gemm.fuseLoops;
            kernelOptions->allowAmbiguousMemoryNodes     = gemm.allowAmbiguousMemoryNodes;
            kernelOptions->unrollK                       = gemm.unrollK;
            kernelOptions->packMultipleElementsInto1VGPR = gemm.packMultipleElementsInto1VGPR;
            kernelOptions->prefetch                      = gemm.prefetch;
            kernelOptions->prefetchInFlight              = gemm.prefetchInFlight;
            kernelOptions->prefetchLDSFactor             = gemm.prefetchLDSFactor;
            kernelOptions->prefetchMixMemOps             = gemm.prefetchMixMemOps;
            kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A] = gemm.transA == "T";
            kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B] = gemm.transB == "T";
            kernelOptions->scaleA                                      = gemm.scaleA;
            kernelOptions->scaleB                                      = gemm.scaleB;

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

                kernelOptions->numScratchTiles
                    = std::min(gemm.numCUs, numWorkgroupX * numWorkgroupY);

                kernelOptions->loopOverOutputTilesDimensions = {0, 1};
                kernelOptions->streamK                       = true;
                kernelOptions->streamKTwoTile                = gemm.streamKTwoTile;
            }

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            // TODO: Calculate these values internally based on workgroup sizes.
            params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);
            params->setSplitStoreTileIntoWaveBlocks(gemm.splitStoreTileIntoWaveBlocks);

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
                gemm.storeLDSD ? MemoryType::JAMMED_WAVE_LDS : MemoryType::WAVE);

            params->setDimensionInfo(tagLoadA, macTileA);
            params->setDimensionInfo(tagLoadB, macTileB);
            params->setDimensionInfo(tagLoadC, macTileC);
            // TODO Fix MemoryType promotion (JAMMED_WAVE_LDS)
            params->setDimensionInfo(tagStoreD, macTileD);

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

            CommandKernel commandKernel(
                command, testKernelName(), params, postParams, kernelOptions);

            CommandArguments commandArgs = command->createArguments();

            if constexpr(std::is_same_v<T, FP4> || std::is_same_v<T, FP6> || std::is_same_v<T, BF6>)
            {
                commandArgs.setArgument(tagTensorA, ArgumentType::Value, (uint8_t*)deviceA.get());
                commandArgs.setArgument(tagTensorB, ArgumentType::Value, (uint8_t*)deviceB.get());
            }
            else
            {
                commandArgs.setArgument(tagTensorA, ArgumentType::Value, deviceA.get());
                commandArgs.setArgument(tagTensorB, ArgumentType::Value, deviceB.get());
            }
            commandArgs.setArgument(tagTensorC, ArgumentType::Value, deviceC.get());
            commandArgs.setArgument(tagTensorD, ArgumentType::Value, deviceD.get());

            commandArgs.setArgument(tagTensorA, ArgumentType::Limit, (size_t)M * K);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, (size_t)M);
            commandArgs.setArgument(tagTensorA, ArgumentType::Size, 1, (size_t)K);
            if(gemm.transA == "N")
            {
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)M);
            }
            else
            {
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)K);
                commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 1, (size_t)1);
            }

            commandArgs.setArgument(tagTensorB, ArgumentType::Limit, (size_t)K * N);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, (size_t)K);
            commandArgs.setArgument(tagTensorB, ArgumentType::Size, 1, (size_t)N);
            if(gemm.transB == "N")
            {
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)K);
            }
            else
            {
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)N);
                commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 1, (size_t)1);
            }

            commandArgs.setArgument(tagTensorC, ArgumentType::Limit, (size_t)M * N);
            commandArgs.setArgument(tagTensorC, ArgumentType::Size, 0, (size_t)M);
            commandArgs.setArgument(tagTensorC, ArgumentType::Size, 1, (size_t)N);
            commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 0, (size_t)1);
            commandArgs.setArgument(tagTensorC, ArgumentType::Stride, 1, (size_t)M);

            commandArgs.setArgument(tagScalarAlpha, ArgumentType::Value, alpha);

            commandArgs.setArgument(tagScalarBeta, ArgumentType::Value, beta);

            commandArgs.setArgument(tagTensorD, ArgumentType::Limit, (size_t)M * N);
            commandArgs.setArgument(tagTensorD, ArgumentType::Size, 0, (size_t)M);
            commandArgs.setArgument(tagTensorD, ArgumentType::Size, 1, (size_t)N);
            commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 0, (size_t)1);
            commandArgs.setArgument(tagTensorD, ArgumentType::Stride, 1, (size_t)M);

            // Create scratch space
            auto scratchSpaceRequired = commandKernel.scratchSpaceRequired();
            auto deviceScratch        = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
            commandArgs.setArgument(tagScratch, ArgumentType::Value, deviceScratch.get());
            if(gemm.streamK)
            {
                commandArgs.setArgument(command->getNextTag(), ArgumentType::Value, gemm.numCUs);
            }

            // GPU is doing: D = alpha * scaleA * scaleB * (A*B) + beta*C. So we need to
            // account for the scales on CPU side as well.
            // Multiply alpha by the scales. Scale is E8M0 and has a bias of 127
            alpha
                *= std::pow(2.0f, int(gemm.scaleA) - 127) * std::pow(2.0f, int(gemm.scaleB) - 127);

            // Host result
            std::vector<TD> h_result(M * N, TD{});
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

            // Device result
            std::vector<TD> d_result(M * N);

            for(int iteration = 0; iteration < numIters; ++iteration)
            {
                ASSERT_THAT(hipMemset(deviceD.get(), 0, M * N * sizeof(TD)), HasHipSuccess(0));
                ASSERT_THAT(hipMemset(deviceScratch.get(), 0, scratchSpaceRequired),
                            HasHipSuccess(0));

                commandKernel.launchKernel(commandArgs.runtimeArguments());
                m_context = commandKernel.getContext();

                ASSERT_THAT(
                    hipMemcpy(
                        d_result.data(), deviceD.get(), M * N * sizeof(TD), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

                double rnorm = relativeNorm(d_result, h_result);
                Log::info("RNorm is {} (iteration {})", rnorm, iteration);
                if(debuggable && rnorm > acceptableError)
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
                                          << std::scientific << a << std::setw(16)
                                          << std::scientific << b << std::setw(16)
                                          << std::scientific << a - b << std::endl;
                            }
                        }
                    }
                }

                ASSERT_LT(rnorm, acceptableError) << "Iteration: " << iteration;
            }
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

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBetaIsZero)
    {
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(m_context, gemm, 1.e-6);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMNotSetC)
    {
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(m_context, gemm, 1.e-6, false, false, 1, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBetaIsZeroStreamK)
    {
        if(m_context->targetArchitecture().target().getVersionString() == "gfx908")
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMBeta0StreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numCUs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numCUs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numCUs);

        gemm.streamK = true;
        gemm.k       = gemm.macK * 8;

        // TODO: Does not work with unrolling K
        //gemm.unrollK          = 2;
        //gemm.prefetch         = true;
        //gemm.prefetchInFlight = 2;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        gemm.beta = 0;

        for(auto twoTile : {true, false})
        {
            gemm.streamKTwoTile = twoTile;
            basicGEMM<float>(m_context, gemm, 1.e-6);
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMStreamK)
    {
        if(m_context->targetArchitecture().target().getVersionString() == "gfx908")
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numCUs = deviceProperties.multiProcessorCount;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numCUs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numCUs);

        gemm.streamK = true;
        gemm.k       = gemm.macK * 8;

        // TODO: Does not work with unrolling K
        //gemm.unrollK          = 2;
        //gemm.prefetch         = true;
        //gemm.prefetchInFlight = 2;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = true;

        for(auto twoTile : {true, false})
        {
            gemm.streamKTwoTile = twoTile;
            basicGEMM<float>(m_context, gemm, 1.e-6);
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16StreamK)
    {
        if(m_context->targetArchitecture().target().getVersionString() != "gfx90a")
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numCUs = deviceProperties.multiProcessorCount;

        gemm.waveK = 8;
        gemm.macK  = 16;

        gemm.macM           = 128;
        gemm.macN           = 256;
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 2;

        gemm.m = gemm.macM * 8;
        gemm.n = gemm.macN * gemm.numCUs / 2 + gemm.macN * 2;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numCUs);

        gemm.streamK = true;
        gemm.k       = gemm.macK * 8;

        // TODO: Does not work with unrolling K
        //gemm.unrollK          = 2;
        //gemm.prefetch         = true;
        //gemm.prefetchInFlight = 2;

        for(auto twoTile : {true, false})
        {
            gemm.streamKTwoTile = twoTile;
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
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16StreamKSmall)
    {
        if(m_context->targetArchitecture().target().getVersionString() != "gfx90a")
        {
            GTEST_SKIP() << "Skipping GPU_BasicGEMMStreamK test";
        }

        GEMMProblem gemm;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        gemm.numCUs = 3;

        gemm.waveK = 8;
        gemm.macK  = 16;

        gemm.macM           = 128;
        gemm.macN           = 128;
        gemm.workgroupSizeX = 2 * gemm.wavefrontSize;
        gemm.workgroupSizeY = 4;

        gemm.m = 4 * gemm.macM;
        gemm.n = 4 * gemm.macN;

        ASSERT_GE(gemm.m * gemm.n / gemm.macM / gemm.macN, gemm.numCUs);

        gemm.streamK = true;
        gemm.k       = gemm.macK * 8;

        for(auto twoTile : {true, false})
        {
            gemm.streamKTwoTile = twoTile;
            basicGEMM<Half>(m_context, gemm, 2.e-5);
        }
    }

    TEST_F(GEMMTestGPU, DISABLED_GPU_BasicGEMMMultipleOutputTiles)
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

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16Prefetch3)
    {
        GEMMProblem gemm;
        gemm.m                 = 4096;
        gemm.n                 = 4096;
        gemm.k                 = 2048 * 3;
        gemm.loadLDSA          = true;
        gemm.loadLDSB          = true;
        gemm.storeLDSD         = false;
        gemm.fuseLoops         = false;
        gemm.unrollK           = 3;
        gemm.macM              = 128;
        gemm.macN              = 16;
        gemm.macK              = 64;
        gemm.waveM             = 16;
        gemm.waveN             = 16;
        gemm.waveK             = 16;
        gemm.workgroupSizeX    = 256;
        gemm.workgroupSizeY    = 1;
        gemm.prefetch          = true;
        gemm.prefetchInFlight  = 3;
        gemm.prefetchLDSFactor = 2;
        gemm.prefetchMixMemOps = true;
        basicGEMM<Half>(m_context, gemm, 5.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP16)
    {
        GEMMProblem gemm;
        gemm.waveK = 8;

        basicGEMM<Half>(m_context, gemm, 2.e-5);
    }

    GEMMProblem setup_GEMMF8_NT()
    {
        GEMMProblem gemm;

        // 4x2 jamming
        uint wavesPerWGX = 16;
        uint wavesPerWGY = 2;

        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 32;

        gemm.macM = wavesPerWGX * gemm.waveM;
        gemm.macN = wavesPerWGY * gemm.waveN;
        gemm.macK = 2 * gemm.waveK;

        gemm.loadLDSA = true;
        gemm.loadLDSB = true;

        gemm.workgroupSizeX = 256;
        gemm.workgroupSizeY = 1;

        gemm.m = 33 * gemm.macM;
        gemm.n = 17 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.alpha = 2.1;
        gemm.beta  = 0.75;

        gemm.transA = "N";
        gemm.transB = "T";

        return gemm;
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP8_16x16x32_NT)
    {
        auto gemm = setup_GEMMF8_NT();
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF8_16x16x32_NT)
    {
        auto gemm = setup_GEMMF8_NT();
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
    }

    GEMMProblem setup_GEMMF8F6F4_NT(int waveM, int waveN, int waveK)
    {
        GEMMProblem gemm;

        // 2x2 jamming
        uint wavesPerWGX = 4;
        uint wavesPerWGY = 4;

        gemm.waveM = waveM;
        gemm.waveN = waveN;
        gemm.waveK = waveK;

        gemm.macM = wavesPerWGX * gemm.waveM;
        gemm.macN = wavesPerWGY * gemm.waveN;
        gemm.macK = 2 * gemm.waveK;

        gemm.loadLDSA = true;
        gemm.loadLDSB = true;

        gemm.workgroupSizeX = 256;
        gemm.workgroupSizeY = 1;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.alpha = 2.1;
        gemm.beta  = 0.75;

        gemm.transA = "N";
        gemm.transB = "T";

        return gemm;
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP8_16x16x128_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(16, 16, 128);
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP8_16x16x128_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_NT(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF8_16x16x128_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(16, 16, 128);
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMBF8_16x16x128_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_NT(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP8_32x32x64_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(32, 32, 64);
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP8_32x32x64_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_NT(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF8_32x32x64_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(32, 32, 64);
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMBF8_32x32x64_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_NT(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
    }

    GEMMProblem setup_GEMMF8_TN()
    {
        GEMMProblem gemm;

        // 1x1 jamming
        uint wavesPerWGX = 4;
        uint wavesPerWGY = 1;

        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 32;

        gemm.macM = wavesPerWGX * gemm.waveM;
        gemm.macN = wavesPerWGY * gemm.waveN;
        gemm.macK = 2 * gemm.waveK;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;

        gemm.workgroupSizeX = 256;
        gemm.workgroupSizeY = 1;

        gemm.m = 33 * gemm.macM;
        gemm.n = 17 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.alpha = 2.1;
        gemm.beta  = 0.75;

        gemm.transA = "T";
        gemm.transB = "N";

        return gemm;
    }

    void check_GEMMF8_TN(rocRoller::ContextPtr m_context)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string generatedCode = m_context->instructions()->toString();

            EXPECT_EQ(countSubstring(generatedCode, "buffer_load"), 3);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4 "), 2);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 1);

            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b"), 2);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), 1);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b32 "), 1);

            EXPECT_EQ(countSubstring(generatedCode, "ds_read"), 4);
            EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64 "), 4);
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP8_16x16x32_TN)
    {
        auto gemm = setup_GEMMF8_TN();
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8_TN(m_context);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF8_16x16x32_TN)
    {
        auto gemm = setup_GEMMF8_TN();
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8_TN(m_context);
    }

    GEMMProblem setup_GEMMF8F6F4_TN(int waveM, int waveN, int waveK)
    {
        GEMMProblem gemm;

        // 2x2 jamming
        uint wavesPerWGX = 4;
        uint wavesPerWGY = 4;

        gemm.waveM = waveM;
        gemm.waveN = waveN;
        gemm.waveK = waveK;

        gemm.macM = wavesPerWGX * gemm.waveM;
        gemm.macN = wavesPerWGY * gemm.waveN;
        gemm.macK = 2 * gemm.waveK;

        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;

        gemm.workgroupSizeX = 256;
        gemm.workgroupSizeY = 1;

        gemm.m = 2 * gemm.macM;
        gemm.n = 3 * gemm.macN;
        gemm.k = 4 * gemm.macK;

        gemm.alpha = 2.1;
        gemm.beta  = 0.75;

        gemm.transA = "T";
        gemm.transB = "N";

        return gemm;
    }

    void check_GEMMF8F6F4_TN(rocRoller::ContextPtr m_context,
                             uint                  numBufferLoads,
                             uint                  numDSWrites,
                             uint                  numDSReads,
                             bool const            isF6Type = false)

    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            std::string generatedCode = m_context->instructions()->toString();

            EXPECT_EQ(countSubstring(generatedCode, "buffer_load"), numBufferLoads);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dwordx4 "), numBufferLoads);
            EXPECT_EQ(countSubstring(generatedCode, "buffer_load_dword "), 0);

            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b"), numDSWrites);
            EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128 "), numDSWrites);

            EXPECT_EQ(countSubstring(generatedCode, "ds_read"), numDSReads);
            if(!isF6Type)
            {
                EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128 "), numDSReads);
            }
            else
            {
                EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128 "), numDSReads / 2);
                EXPECT_EQ(countSubstring(generatedCode, "ds_read_b64 "), numDSReads / 2);
            }
        }
    }

    void check_mfma_f8f6f4(rocRoller::ContextPtr m_context,
                           std::string           f8f6f4_inst,
                           std::string           modifier)
    {
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasMFMA_fp8))
        {
            auto generatedCode = m_context->instructions()->toString();

            auto mfma_count     = countSubstring(generatedCode, "v_mfma_");
            auto f8f6f4_count   = countSubstring(generatedCode, f8f6f4_inst);
            auto modifier_count = countSubstring(generatedCode, modifier);

            // All mfma instructions should be f8f6f4
            EXPECT_EQ(mfma_count, f8f6f4_count);
            // All f8f6f4 instructions should use 0b100 (FP4) as input matrix format
            EXPECT_EQ(f8f6f4_count, modifier_count);
        }
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP4_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<FP4, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_16x16x128_f8f6f4", "cbsz:0b100 blgp:0b100");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 8) / 64, 4, 10);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP4_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP4, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(
            m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b100 abid:1 blgp:0b100");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 8) / 64, 4, 10);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP4_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<FP4, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_32x32x64_f8f6f4", "cbsz:0b100 blgp:0b100");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 8) / 64, 4, 10);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP4_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP4, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(
            m_context, "v_mfma_scale_f32_32x32x64_f8f6f4", "cbsz:0b100 abid:1 blgp:0b100");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 8) / 64, 4, 10);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<FP6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_16x16x128_f8f6f4", "cbsz:0b010 blgp:0b010");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(
            m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b010 abid:1 blgp:0b010");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP6_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<FP6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_32x32x64_f8f6f4", "cbsz:0b010 blgp:0b010");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP6_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(
            m_context, "v_mfma_scale_f32_32x32x64_f8f6f4", "cbsz:0b010 abid:1 blgp:0b010");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<BF6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_16x16x128_f8f6f4", "cbsz:0b011 blgp:0b011");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMBF6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<BF6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(
            m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b011 abid:1 blgp:0b011");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF6_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<BF6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_32x32x64_f8f6f4", "cbsz:0b011 blgp:0b011");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMBF6_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<BF6, float>(m_context, gemm, 2.e-5);
        check_mfma_f8f6f4(
            m_context, "v_mfma_scale_f32_32x32x64_f8f6f4", "cbsz:0b011 abid:1 blgp:0b011");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMBF8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMFP8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMFP8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<FP8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicGEMMBF8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 4) / 64, 8, 20);
    }

    TEST_F(GEMMTestGPU, GPU_BasicScaledGEMMBF8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm   = setup_GEMMF8F6F4_TN(32, 32, 64);
        gemm.scaleA = 128;
        gemm.scaleB = 125;
        basicGEMM<BF8, float>(m_context, gemm, 2.e-5);
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 4) / 64, 8, 20);
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

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 18);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
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

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 20);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
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

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 18);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
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

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 24);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
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
        EXPECT_EQ(countSubstring(generatedCode, "v_pack_B32_F16"), 152);
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
        EXPECT_LT(countSubstring(output_literalStrides, "Tensor_0_stride_0"),
                  countSubstring(output_noLiteralStrides, "Tensor_0_stride_0"));
        EXPECT_LT(countSubstring(output_literalStrides, "Tensor_2_stride_0"),
                  countSubstring(output_noLiteralStrides, "Tensor_2_stride_0"));
        EXPECT_LT(countSubstring(output_literalStrides, "Tensor_4_stride_0"),
                  countSubstring(output_noLiteralStrides, "Tensor_4_stride_0"));

        //Since we're not setting the second dimension to a literal, there will be the same occurrences of Load_Tiled_X_stride_1.
        EXPECT_EQ(countSubstring(output_literalStrides, "Tensor_0_stride_1"),
                  countSubstring(output_noLiteralStrides, "Tensor_0_stride_1"));
        EXPECT_EQ(countSubstring(output_literalStrides, "Tensor_2_stride_1"),
                  countSubstring(output_noLiteralStrides, "Tensor_2_stride_1"));
        EXPECT_EQ(countSubstring(output_literalStrides, "Tensor_4_stride_1"),
                  countSubstring(output_noLiteralStrides, "Tensor_4_stride_1"));
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

    TEST_F(GEMMTestGPU, GPU_BasicGEMMStoreDWave)
    {
        GEMMProblem gemm;

        auto nonZeroDSReadOffsets = [](auto s) {
            std::regex ds_read_offset("ds_read_b128.*offset:(\\d+)");

            auto begin = std::sregex_iterator(s.begin(), s.end(), ds_read_offset);
            auto end   = std::sregex_iterator();

            std::set<int> rv;
            for(auto i = begin; i != end; ++i)
            {
                auto m = (*i)[1].str();
                rv.insert(std::stoi(m));
            }
            return rv;
        };

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

        gemm.splitStoreTileIntoWaveBlocks = true;
        basicGEMM<Half>(m_context, gemm, 2.e-5);
        auto instructions0 = output();
        EXPECT_EQ(nonZeroDSReadOffsets(instructions0), std::set<int>{1024});

        gemm.splitStoreTileIntoWaveBlocks = false;
        basicGEMM<Half>(m_context, gemm, 2.e-5);
        auto instructions1 = output();
        EXPECT_EQ(nonZeroDSReadOffsets(instructions1), std::set<int>{64});
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
