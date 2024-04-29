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

#include "GEMMProblem.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

namespace GEMMDriverTest
{
    struct GEMMFusionGPU : public CurrentGPUContextFixture
    {
        template <typename T>
        void basicGEMMRelu(ContextPtr&        m_context,
                           const GEMMProblem& gemm,
                           double             acceptableError,
                           bool               debuggable  = false,
                           bool               setIdentity = false,
                           int                numIters    = 1)

        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

            // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
            int   M         = gemm.m;
            int   N         = gemm.n;
            int   K         = gemm.k;
            float alpha     = gemm.alpha;
            float beta      = gemm.beta;
            T     reluAlpha = 0.9;

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

            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataType, gemm.transA == "N" ? oneStridesN : oneStridesT)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataType, gemm.transB == "N" ? oneStridesN : oneStridesT)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto tagTensorC = command->addOperation(
                rocRoller::Operations::Tensor(2, dataType, oneStridesN)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagScalarAlpha
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
            auto tagLoadAlpha
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

            auto tagScalarBeta
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // beta
            auto tagLoadBeta
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta));

            auto tagScalarReluAlpha = command->addOperation(
                rocRoller::Operations::Scalar(DataType::Float)); // leaky relu alpha
            auto tagLoadReluAlpha
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarReluAlpha));

            auto tagLiteralZero
                = command->addOperation(rocRoller::Operations::Literal(0.0f)); // zero

            auto tagAB
                = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // A * B

            rocRoller::Operations::T_Execute execute(command->getNextTag());
            auto                             tagBetaC
                = execute.addXOp(rocRoller::Operations::E_Mul(tagLoadBeta, tagLoadC)); // beta * C
            auto tagAlphaAB = execute.addXOp(
                rocRoller::Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)
            auto tagD = -1;
            if(gemm.betaInFma)
            {
                tagD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
            }
            else
            {
                tagD = execute.addXOp(rocRoller::Operations::E_Add(
                    tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
            }
            auto tagDGtZero = execute.addXOp(
                rocRoller::Operations::E_GreaterThan(tagD, tagLiteralZero)); // D > 0
            auto tagDReluAlpha = execute.addXOp(
                rocRoller::Operations::E_Mul(tagD, tagLoadReluAlpha)); // D * reluAlpha
            auto tagRelu = execute.addXOp(rocRoller::Operations::E_Conditional(
                tagDGtZero, tagD, tagDReluAlpha)); // D > 0 ? D : D * reluAlpha
            command->addOperation(std::move(execute));

            auto tagTensorRelu = command->addOperation(
                rocRoller::Operations::Tensor(2, dataType, oneStridesN)); // E
            command->addOperation(rocRoller::Operations::T_Store_Tiled(tagRelu, tagTensorRelu));

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

            runtimeArgs.append("reluAlpha", static_cast<T>(reluAlpha));

            runtimeArgs.append("D", deviceD.get());
            runtimeArgs.append("d_d_limit", (size_t)M * N);
            runtimeArgs.append("d_d_size_0", (size_t)M);
            runtimeArgs.append("d_d_size_1", (size_t)N);
            runtimeArgs.append("d_d_stride_0", (size_t)1);
            runtimeArgs.append("d_d_stride_1", (size_t)M);

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
            params->setDimensionInfo(tagD, macTileD);
            params->setDimensionInfo(tagRelu, macTileD);

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
                runtimeArgs.append("numWGs", gemm.numCUs);
            }

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
            // Host leaky relu
            for(size_t i = 0; i < M; i++)
            {
                for(size_t j = 0; j < N; j++)
                {
                    auto b = h_result[i * N + j];
                    h_result[i * N + j]
                        = b > 0 ? static_cast<T>(b)
                                : static_cast<T>(static_cast<T>(reluAlpha) * static_cast<T>(b));
                }
            }

            // Device result
            std::vector<T> d_result(M * N);

            for(int iteration = 0; iteration < numIters; ++iteration)
            {
                ASSERT_THAT(hipMemset(deviceD.get(), 0, M * N * sizeof(T)), HasHipSuccess(0));
                ASSERT_THAT(hipMemset(deviceScratch.get(), 0, scratchSpaceRequired),
                            HasHipSuccess(0));

                commandKernel.launchKernel(runtimeArgs.runtimeArguments());
                m_context = commandKernel.getContext();

                ASSERT_THAT(
                    hipMemcpy(
                        d_result.data(), deviceD.get(), M * N * sizeof(T), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

                double rnorm = relativeNorm(d_result, h_result);
                if(debuggable && rnorm > acceptableError)
                {
                    for(size_t i = 0; i < M; i++)
                    {
                        for(size_t j = 0; j < N; j++)
                        {
                            auto const a = d_result[i * N + j];
                            auto       b = h_result[i * N + j];
                            if((a - b) * (a - b) / (b * b) > 100.0 * acceptableError)
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

    TEST_F(GEMMFusionGPU, GPU_GEMMRelu)
    {
        GEMMProblem gemm;
        basicGEMMRelu<float>(m_context, gemm, 1.e-6);
    }

}
