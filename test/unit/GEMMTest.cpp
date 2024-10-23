#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

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

#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "TensorDescriptor.hpp"
#include "Utilities.hpp"
#include <common/GEMMProblem.hpp>

#include "GEMMF8F6F4.hpp"

namespace GEMMDriverTest
{
    template <typename T>
    concept isF8 = std::is_same_v<T, FP8> || std::is_same_v<T, BF8>;

    template <typename T>
    concept isF6F4 = std::is_same_v<T, FP6> || std::is_same_v<T, BF6> || std::is_same_v<T, FP4>;

    template <typename... Ts>
    class BaseGEMMContextFixture
        : public BaseGPUContextFixture,
          public ::testing::WithParamInterface<std::tuple<GPUArchitectureTarget, Ts...>>
    {
    protected:
        virtual rocRoller::ContextPtr createContext() override
        {
            auto device = std::get<0>(this->GetParam());

            return this->createContextForArch(device);
        }

    public:
        template <typename TA, typename TB = TA, typename TC = TA, typename TD = TC>
        void basicGEMM(const GEMMProblem& gemm,
                       bool               debuggable  = false,
                       bool               setIdentity = false,
                       int                numIters    = 1,
                       bool               notSetC     = false)

        {
            REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
            if constexpr(isF8<TA> || isF8<TB>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_fp8);
            }

            if constexpr(isF6F4<TA> || isF6F4<TB>)
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }
            if((isF8<TA> || isF8<TB>)&&(gemm.waveK >= 64))
            {
                REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
            }

            AssertFatal(gemm.scaleAMode == gemm.scaleBMode,
                        "Scale modes must match",
                        ShowValue(gemm.scaleAMode),
                        ShowValue(gemm.scaleBMode));
            AssertFatal(gemm.scaleAMode == Operations::ScaleMode::None
                            || gemm.scaleAMode == Operations::ScaleMode::SingleScale
                            || gemm.scaleAMode == Operations::ScaleMode::Separate,
                        "Scale mode not supported!",
                        ShowValue(gemm.scaleAMode));

            auto dataTypeA = TypeInfo<TA>::Var.dataType;
            auto dataTypeB = TypeInfo<TB>::Var.dataType;
            auto dataTypeC = TypeInfo<TC>::Var.dataType;
            auto dataTypeD = TypeInfo<TD>::Var.dataType;

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

            auto bpeA = DataTypeInfo::Get(dataTypeA).elementBytes;
            auto bpeB = DataTypeInfo::Get(dataTypeB).elementBytes;
            AssertFatal(gemm.macM * gemm.macK * bpeA > gemm.waveM * gemm.waveK,
                        "Not enough elements (A).");
            AssertFatal(gemm.macN * gemm.macK * bpeB > gemm.waveN * gemm.waveK,
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
            using UnsegmentedTypeA = typename UnsegmentedTypeOf<TA>::type;
            using UnsegmentedTypeB = typename UnsegmentedTypeOf<TB>::type;
            std::vector<UnsegmentedTypeA> hostA;
            std::vector<UnsegmentedTypeB> hostB;
            std::vector<TC>               hostC;

            std::vector<uint8_t> hostScaleA, hostScaleB;

            GenerateRandomInput(31415u, hostA, M * K, hostB, K * N, hostC, M * N);

            {
                auto rng = RandomGenerator(51413u);
                if(gemm.scaleAMode == Operations::ScaleMode::Separate)
                    hostScaleA = rng.vector<uint8_t>(
                        M * K / 32, floatToScale(0.03125f), floatToScale(1024.0f));
                if(gemm.scaleBMode == Operations::ScaleMode::Separate)
                    hostScaleB = rng.vector<uint8_t>(
                        K * N / 32, floatToScale(0.03125f), floatToScale(1024.0f));
            }

            if(setIdentity)
            {
                SetIdentityMatrix(hostA, K, M);
                SetIdentityMatrix(hostB, N, K);

                std::fill(hostC.begin(), hostC.end(), static_cast<TD>(0.0));
            }

            auto deviceA = make_shared_device<TA>(hostA);
            auto deviceB = make_shared_device<TB>(hostB);

            std::shared_ptr<TC> deviceC = (notSetC) ? nullptr : make_shared_device(hostC);
            std::shared_ptr<TD> deviceD = make_shared_device<TD>(M * N, TD{});

            std::shared_ptr<uint8_t> deviceScaleA, deviceScaleB;

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
                deviceScaleA = make_shared_device(hostScaleA);
            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
                deviceScaleB = make_shared_device(hostScaleB);

            auto command = std::make_shared<Command>();

            std::vector<size_t> oneStridesN
                = gemm.literalStrides ? std::vector<size_t>({(size_t)1}) : std::vector<size_t>({});

            std::vector<size_t> oneStridesT = gemm.literalStrides
                                                  ? std::vector<size_t>({(size_t)0, (size_t)1})
                                                  : std::vector<size_t>({});

            auto tagTensorA = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeA, gemm.transA == "N" ? oneStridesN : oneStridesT)); // A
            auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

            auto tagTensorB = command->addOperation(rocRoller::Operations::Tensor(
                2, dataTypeB, gemm.transB == "N" ? oneStridesN : oneStridesT)); // B
            auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

            auto mulInputA = tagLoadA;
            auto mulInputB = tagLoadB;

            std::optional<Operations::OperationTag> tagTensorScaleA, tagLoadScaleA, tagBlockScaleA,
                tagTensorScaleB, tagLoadScaleB, tagBlockScaleB;

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(gemm.transA == "T", ShowValue(gemm.transA));

                tagTensorScaleA = command->addOperation(
                    rocRoller::Operations::Tensor(2, DataType::UInt8, oneStridesT));
                tagLoadScaleA
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleA));

                tagBlockScaleA = mulInputA = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadA, 2, tagLoadScaleA, {1, 32}));
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                AssertFatal(gemm.transB == "N", ShowValue(gemm.transB));

                tagTensorScaleB = command->addOperation(
                    rocRoller::Operations::Tensor(2, DataType::UInt8, oneStridesN));
                tagLoadScaleB
                    = command->addOperation(rocRoller::Operations::T_Load_Tiled(*tagTensorScaleB));

                tagBlockScaleB = mulInputB = command->addOperation(
                    rocRoller::Operations::BlockScale(tagLoadB, 2, tagLoadScaleB, {32, 1}));
            }

            auto tagTensorC = command->addOperation(
                rocRoller::Operations::Tensor(2, dataTypeC, oneStridesN)); // C
            auto tagLoadC = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorC));

            auto tagScalarAlpha
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // alpha
            auto tagLoadAlpha
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarAlpha));

            auto tagScalarBeta
                = command->addOperation(rocRoller::Operations::Scalar(DataType::Float)); // beta
            auto tagLoadBeta
                = command->addOperation(rocRoller::Operations::T_Load_Scalar(tagScalarBeta));

            auto tagAB = command->addOperation(
                rocRoller::Operations::T_Mul(mulInputA, mulInputB)); // A * B

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
            if constexpr(std::is_same_v<TC, TD>)
            {
                command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));
            }
            else
            {
                // If Matrix C and D are of different types, an explicit type conversion is required
                auto cvtOp  = rocRoller::Operations::T_Execute(command->getNextTag());
                auto tagCvt = cvtOp.addXOp(rocRoller::Operations::E_Cvt(tagStoreD, dataTypeD));
                command->addOperation(std::move(cvtOp)); // Convert( alpha * (A * B) + beta * C )
                command->addOperation(rocRoller::Operations::T_Store_Tiled(tagCvt, tagTensorD));
            }

            auto tagScratch = command->allocateTag();
            command->allocateArgument(VariableType(DataType::UInt32, PointerType::PointerGlobal),
                                      tagScratch,
                                      ArgumentType::Value,
                                      DataDirection::ReadWrite,
                                      rocRoller::SCRATCH);

            auto params = std::make_shared<CommandParameters>();
            params->setManualKernelDimension(2);
            // TODO: Calculate these values internally based on workgroup sizes.
            params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);
            params->setSplitStoreTileIntoWaveBlocks(gemm.splitStoreTileIntoWaveBlocks);

            params->fuseLoops                     = gemm.fuseLoops;
            params->allowAmbiguousMemoryNodes     = gemm.allowAmbiguousMemoryNodes;
            params->unrollK                       = gemm.unrollK;
            params->packMultipleElementsInto1VGPR = gemm.packMultipleElementsInto1VGPR;
            params->prefetch                      = gemm.prefetch;
            params->prefetchInFlight              = gemm.prefetchInFlight;
            params->prefetchLDSFactor             = gemm.prefetchLDSFactor;
            params->prefetchMixMemOps             = gemm.prefetchMixMemOps;
            params->transposeMemoryAccess[LayoutType::MATRIX_A] = gemm.transA == "T";
            params->transposeMemoryAccess[LayoutType::MATRIX_B] = gemm.transB == "T";

            if(gemm.loopOverTiles > 0)
            {
                params->loopOverOutputTilesDimensions = {0, 1};
                params->loopOverOutputTilesCoordSizes
                    = {static_cast<uint>(M / gemm.macM), static_cast<uint>(N / gemm.macN)};
                params->loopOverOutputTilesIteratedTiles = 2;
            }

            if(gemm.streamK)
            {
                REQUIRE_ARCH_CAP(GPUCapability::ArchAccUnifiedRegs);

                AssertFatal(
                    numWorkgroupY == 1,
                    "Current scratch space implementation assumes that the kernel is launched "
                    "with numWorkgroupY == 1");

                params->numScratchTiles = std::min(gemm.numCUs, numWorkgroupX * numWorkgroupY);

                params->loopOverOutputTilesDimensions = {0, 1};
                params->streamK                       = true;
                params->streamKTwoTile                = gemm.streamKTwoTile;
            }

            {
                auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macK},
                    LayoutType::MATRIX_A,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    gemm.loadLDSA ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(tagLoadA, macTileA);
            }

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                auto macTileAScale = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macK / 32},
                    LayoutType::MATRIX_A,
                    {gemm.waveM, gemm.waveN, gemm.waveK / 32, gemm.waveB},
                    gemm.loadLDSScaleA ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(*tagLoadScaleA, macTileAScale);
            }

            {
                auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macK, gemm.macN},
                    LayoutType::MATRIX_B,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    gemm.loadLDSB ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(tagLoadB, macTileB);
            }

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
            {
                auto macTileBScale = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macK / 32, gemm.macN},
                    LayoutType::MATRIX_B,
                    {gemm.waveM, gemm.waveN, gemm.waveK / 32, gemm.waveB},
                    gemm.loadLDSScaleB ? MemoryType::LDS : MemoryType::WAVE);
                params->setDimensionInfo(*tagLoadScaleB, macTileBScale);
            }

            {
                auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macN},
                    LayoutType::MATRIX_ACCUMULATOR,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB});
                params->setDimensionInfo(tagLoadC, macTileC);
            }

            {
                auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
                    {gemm.macM, gemm.macN},
                    LayoutType::MATRIX_ACCUMULATOR,
                    {gemm.waveM, gemm.waveN, gemm.waveK, gemm.waveB},
                    gemm.storeLDSD ? MemoryType::JAMMED_WAVE_LDS : MemoryType::WAVE);
                params->setDimensionInfo(tagStoreD, macTileD);
            }

            // TODO Fix MemoryType promotion (JAMMED_WAVE_LDS)

            params->setManualWorkgroupSize({workgroupSizeX, workgroupSizeY, 1});
            rocRoller::Log::getLogger()->debug(
                "GEMM workgroup sizes {} {} {}", workgroupSizeX, workgroupSizeY, 1);
            rocRoller::Log::getLogger()->debug(
                "GEMM workitem counts {} {} {}", toString(NX), toString(NY), toString(NZ));

            params->setManualWavefrontCount(
                {static_cast<uint>(gemm.macM / gemm.waveM / wavetilePerWavefrontM),
                 static_cast<uint>(gemm.macN / gemm.waveN / wavetilePerWavefrontN)});

            CommandKernel commandKernel(command, testKernelName());

            // TODO Some test have loops, we need to reset the context.
            m_context = createContext();

            commandKernel.setContext(m_context);
            commandKernel.setCommandParameters(params);
            commandKernel.generateKernel();

            auto launch = std::make_shared<CommandLaunchParameters>();
            launch->setManualWorkitemCount({NX, NY, NZ});
            commandKernel.setLaunchParameters(launch);

            CommandArguments commandArgs = command->createArguments();

            TensorDescriptor descA(dataTypeA, {size_t(M), size_t(K)}, gemm.transA);
            TensorDescriptor descAScale(dataTypeA, {size_t(M), size_t(K / 32)}, gemm.transA);
            TensorDescriptor descB(dataTypeB, {size_t(K), size_t(N)}, gemm.transB);
            TensorDescriptor descBScale(dataTypeB, {size_t(K / 32), size_t(N)}, gemm.transB);
            TensorDescriptor descC(dataTypeD, {size_t(M), size_t(N)}, "N");
            TensorDescriptor descD(dataTypeD, {size_t(M), size_t(N)}, "N");

            setCommandTensorArg(commandArgs, tagTensorA, descA, deviceA.get());
            setCommandTensorArg(commandArgs, tagTensorB, descB, deviceB.get());
            setCommandTensorArg(commandArgs, tagTensorC, descC, deviceC.get());
            setCommandTensorArg(commandArgs, tagTensorD, descD, deviceD.get());

            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
                setCommandTensorArg(
                    commandArgs, tagTensorScaleA.value(), descAScale, deviceScaleA.get());

            if(gemm.scaleBMode == Operations::ScaleMode::Separate)
                setCommandTensorArg(
                    commandArgs, tagTensorScaleB.value(), descBScale, deviceScaleB.get());

            commandArgs.setArgument(tagScalarAlpha, ArgumentType::Value, alpha);
            commandArgs.setArgument(tagScalarBeta, ArgumentType::Value, beta);

            // Create scratch space
            auto scratchSpaceRequired = commandKernel.scratchSpaceRequired();
            auto deviceScratch        = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
            commandArgs.setArgument(tagScratch, ArgumentType::Value, deviceScratch.get());

            if(gemm.streamK)
            {
                commandArgs.setArgument(command->getNextTag(), ArgumentType::Value, gemm.numCUs);
            }

            // Host result
            std::vector<TD> h_result(M * N, TD{});
            if(gemm.scaleAMode == Operations::ScaleMode::Separate)
            {
                if constexpr(std::same_as<TD, float>)
                {
                    rocRoller::ScaledCPUMM(h_result,
                                           hostC,
                                           hostA,
                                           hostB,
                                           hostScaleA,
                                           hostScaleB,
                                           M,
                                           N,
                                           K,
                                           alpha,
                                           beta,
                                           gemm.transA == "T",
                                           gemm.transB == "T");
                }
                else
                {
                    GTEST_SKIP() << "Scaled CPUMM only supported with C/D == float";
                }
            }
            else if constexpr(std::is_same_v<TC, TD>)
            {
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
            }
            else
            {
                std::vector<TC> hostD(M * N, TC{});
                rocRoller::CPUMM(hostD,
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
                ASSERT_EQ(hostD.size(), h_result.size());
                for(size_t i = 0; i < hostD.size(); i++)
                    h_result[i] = TD(hostD[i]);
            }

            // Device result
            std::vector<TD> d_result(M * N);

            for(int iteration = 0; iteration < numIters; ++iteration)
            {
                ASSERT_THAT(hipMemset(deviceD.get(), 0, M * N * sizeof(TD)), HasHipSuccess(0));
                ASSERT_THAT(hipMemset(deviceScratch.get(), 0, scratchSpaceRequired),
                            HasHipSuccess(0));

                commandKernel.launchKernel(commandArgs.runtimeArguments());

                ASSERT_THAT(
                    hipMemcpy(
                        d_result.data(), deviceD.get(), M * N * sizeof(TD), hipMemcpyDeviceToHost),
                    HasHipSuccess(0));

                auto tol = gemmAcceptableError<TA, TB, TD>(
                    M, N, K, m_context->targetArchitecture().target());
                auto res = compare(d_result, h_result, tol);
                Log::info("RNorm is {} (acceptable {}, iteration {})",
                          res.relativeNormL2,
                          res.acceptableError.relativeL2Tolerance,
                          iteration);
                if(debuggable && !res.ok)
                {
                    for(size_t i = 0; i < M; i++)
                    {
                        for(size_t j = 0; j < N; j++)
                        {
                            auto a = d_result[i * N + j];
                            auto b = h_result[i * N + j];
                            if((a - b) * (a - b) / (b * b)
                               > res.acceptableError.relativeL2Tolerance)
                            {
                                std::cout << std::setw(8) << i << std::setw(8) << j //
                                          << std::setw(16) << std::scientific << a //
                                          << std::setw(16) << std::scientific << b //
                                          << std::setw(16) << std::scientific << a - b //
                                          << std::endl;
                            }
                        }
                    }
                }

                EXPECT_TRUE(res.ok) << res.message();
            }
        }

        template <typename TA>
        void basicGEMMMixed(rocRoller::DataType typeB, GEMMProblem const& problem)
        {
            if(typeB == rocRoller::DataType::FP8)
                basicGEMM<TA, FP8, float>(problem);
            else if(typeB == rocRoller::DataType::BF8)
                basicGEMM<TA, BF8, float>(problem);
            else if(typeB == rocRoller::DataType::FP6)
                basicGEMM<TA, FP6, float>(problem);
            else if(typeB == rocRoller::DataType::BF6)
                basicGEMM<TA, BF6, float>(problem);
            else if(typeB == rocRoller::DataType::FP4)
                basicGEMM<TA, FP4, float>(problem);
            else
                Throw<FatalError>("Invalid type.");
        }

        void basicGEMMMixed(rocRoller::DataType typeA,
                            rocRoller::DataType typeB,
                            GEMMProblem const&  problem)
        {
            if(typeA == rocRoller::DataType::FP8)
                basicGEMMMixed<FP8>(typeB, problem);
            else if(typeA == rocRoller::DataType::BF8)
                basicGEMMMixed<BF8>(typeB, problem);
            else if(typeA == rocRoller::DataType::FP6)
                basicGEMMMixed<FP6>(typeB, problem);
            else if(typeA == rocRoller::DataType::BF6)
                basicGEMMMixed<BF6>(typeB, problem);
            else if(typeA == rocRoller::DataType::FP4)
                basicGEMMMixed<FP4>(typeB, problem);
            else
                Throw<FatalError>("Invalid type.");
        }
    };

    class GEMMTestGPU : public BaseGEMMContextFixture<>
    {
    };

    class GEMMJammedTestGPU : public BaseGEMMContextFixture<>
    {
    };

    class GEMMF8F6F4TestGPU : public BaseGEMMContextFixture<>
    {
    };

    class GEMMF8TestGPU : public BaseGEMMContextFixture<>
    {
    };

    // Params are: A type, B type, K tile size
    class GEMMMixedF8F6F4TestGPU
        : public BaseGEMMContextFixture<std::tuple<rocRoller::DataType, rocRoller::DataType, int>>
    {
    };

    // Params are: A type, B type, K tile size, Load A scale though LDS, Load B scale through LDS
    class GEMMMixedScaledTestGPU
        : public BaseGEMMContextFixture<
              std::tuple<rocRoller::DataType, rocRoller::DataType, int, bool, bool>>
    {
    };

    // This test is to ensure each scheduler properly yields insts for a basic GEMM
    TEST_P(GEMMTestGPU, GPU_BasicGEMM_Schedulers)
    {
        GEMMProblem gemm;
        gemm.macK = 8;

        // TODO: Re-enable LDS once LDS deallocations are fixed
        gemm.loadLDSA = false;
        gemm.loadLDSB = false;

        auto settings = Settings::getInstance();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Sequential);
        basicGEMM<float>(gemm);
        std::string seq = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::RoundRobin);
        basicGEMM<float>(gemm);
        std::string rr = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Cooperative);
        basicGEMM<float>(gemm);
        std::string coop_nop = m_context->instructions()->toString();

        settings->set(Settings::Scheduler, Scheduling::SchedulerProcedure::Priority);
        basicGEMM<float>(gemm);
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
            basicGEMM<float>(gemm);
            std::string rand     = m_context->instructions()->toString();
            bool        not_seen = insts.insert(rand).second;
            EXPECT_EQ(not_seen, true);
        }
        // Can not compare random insts to others because non-zero chance seed generates such insts
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMM)
    {
        GEMMProblem gemm;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBetaIsZero)
    {
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNotSetC)
    {
        GEMMProblem gemm;
        gemm.beta = 0;
        basicGEMM<float>(gemm, false, false, 1, true);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBetaIsZeroStreamK)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
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
            basicGEMM<float>(gemm);
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMStreamK)
    {
        if(m_context->targetArchitecture().target().isCDNA1GPU())
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
            basicGEMM<float>(gemm);
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16StreamK)
    {
        if(!m_context->targetArchitecture().target().isCDNA2GPU())
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
                        basicGEMM<Half>(gemm);
                    }
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16StreamKSmall)
    {
        if(!m_context->targetArchitecture().target().isCDNA2GPU())
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
            basicGEMM<Half>(gemm);
        }
    }

    TEST_P(GEMMTestGPU, DISABLED_GPU_BasicGEMMMultipleOutputTiles)
    {
        GEMMProblem gemm;
        gemm.storeLDSD     = false;
        gemm.loopOverTiles = true;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNoLDSA)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = true;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNoLDSB)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = false;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMNoLDSAB)
    {
        GEMMProblem gemm;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = false;
        gemm.fuseLoops = false;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollK)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 4;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKLDS)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 2;
        gemm.macK      = 4;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDS)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSA)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = true;
        gemm.loadLDSB  = false;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKMoreLDSB)
    {
        GEMMProblem gemm;
        gemm.k         = 64 * 4 * 2;
        gemm.loadLDSA  = false;
        gemm.loadLDSB  = true;
        gemm.storeLDSD = false;
        gemm.fuseLoops = false;
        gemm.unrollK   = 8;
        gemm.macK      = 8;
        basicGEMM<float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSPrefetch)
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
                    basicGEMM<float>(gemm);
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16UnrollKLDSPrefetch)
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
                    basicGEMM<Half>(gemm);
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMUnrollKLDSMultiPrefetch)
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
                    basicGEMM<float>(gemm);
                }
            }
        }
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16Prefetch3)
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
        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16)
    {
        GEMMProblem gemm;
        gemm.waveK = 8;

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_FP32_32x32x4)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 4;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_FP32_32x32x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_FP32_16x16x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_FP32_16x16x16)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 16;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_BF16_32x32x4)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 4;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x4);
        basicGEMM<BFloat16, BFloat16, BFloat16>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_BF16_32x32x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 32;
        gemm.waveN = 32;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_32x32x8_1k);
        basicGEMM<BFloat16, BFloat16, BFloat16>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_BF16_16x16x8)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 8;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x8);
        basicGEMM<BFloat16, BFloat16, BFloat16>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMBF16_BF16_16x16x16)
    {
        GEMMProblem gemm;
        gemm.waveM = 16;
        gemm.waveN = 16;
        gemm.waveK = 16;

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_bf16_16x16x16_1k);
        basicGEMM<BFloat16, BFloat16, float>(gemm);
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

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP8_NT)
    {
        auto gemm = setup_GEMMF8_NT();
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMF8TestGPU, GPU_BasicGEMMBF8_16x16x32_NT)
    {
        auto gemm = setup_GEMMF8_NT();
        basicGEMM<BF8, BF8, float>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMConversionFP8_NT)
    {
        auto gemm = setup_GEMMF8_NT();
        // D (FP8) = Convert( alpha * A (FP8) * B (FP8) + beta * C (F32) )
        basicGEMM<FP8, FP8, float, FP8>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMConversionBF8_NT)
    {
        auto gemm = setup_GEMMF8_NT();
        // D (BF8) = Convert( alpha * A (BF8) * B (BF8) + beta * C (F32) )
        basicGEMM<BF8, BF8, float, BF8>(gemm);
    }

    TEST_P(GEMMF8F6F4TestGPU, DISABLED_GPU_BasicGEMMFP8_16x16x128_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(16, 16, 128);
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMF8F6F4TestGPU, DISABLED_GPU_BasicGEMMBF8_16x16x128_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(16, 16, 128);
        basicGEMM<BF8, BF8, float>(gemm);
    }

    TEST_P(GEMMF8F6F4TestGPU, DISABLED_GPU_BasicGEMMFP8_32x32x64_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(32, 32, 64);
        basicGEMM<FP8, FP8, float>(gemm);
    }

    TEST_P(GEMMF8F6F4TestGPU, DISABLED_GPU_BasicGEMMBF8_32x32x64_NT)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_NT(32, 32, 64);
        basicGEMM<BF8, BF8, float>(gemm);
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

    TEST_P(GEMMF8TestGPU, GPU_BasicGEMMFP8_16x16x32_TN)
    {
        auto gemm = setup_GEMMF8_TN();
        basicGEMM<FP8, FP8, float>(gemm);
        check_GEMMF8_TN(m_context);
    }

    TEST_P(GEMMF8TestGPU, GPU_BasicGEMMBF8_16x16x32_TN)
    {
        auto gemm = setup_GEMMF8_TN();
        basicGEMM<BF8, BF8, float>(gemm);
        check_GEMMF8_TN(m_context);
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

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMFP4_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<FP4, FP4, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_16x16x128_f8f6f4", "cbsz:0b100 blgp:0b100");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 8) / 64, 4, 10);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicBlockScaledGEMMFP8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm       = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.loadLDSA      = true;
        gemm.loadLDSB      = true;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

        basicGEMM<FP8, FP8, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b000 blgp:0b000");
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicBlockScaledGEMMBF8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm       = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.loadLDSA      = true;
        gemm.loadLDSB      = true;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

        basicGEMM<BF8, BF8, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b001 blgp:0b001");
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicBlockScaledGEMMFP6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm       = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.loadLDSA      = true;
        gemm.loadLDSB      = true;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

        basicGEMM<FP6, FP6, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b010 blgp:0b010");
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicBlockScaledGEMMBF6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm       = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.loadLDSA      = true;
        gemm.loadLDSB      = true;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

        basicGEMM<BF6, BF6, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b011 blgp:0b011");
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicBlockScaledGEMMFP4_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm       = setup_GEMMF8F6F4_TN(16, 16, 128);
        gemm.scaleAMode = Operations::ScaleMode::Separate;
        gemm.scaleBMode = Operations::ScaleMode::Separate;

        gemm.loadLDSA      = true;
        gemm.loadLDSB      = true;
        gemm.loadLDSScaleA = true;
        gemm.loadLDSScaleB = true;

        basicGEMM<FP4, FP4, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_scale_f32_16x16x128_f8f6f4", "cbsz:0b100 blgp:0b100");
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMFP4_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<FP4, FP4, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_32x32x64_f8f6f4", "cbsz:0b100 blgp:0b100");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 8) / 64, 4, 10);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMFP6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<FP6, FP6, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_16x16x128_f8f6f4", "cbsz:0b010 blgp:0b010");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMFP6_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<FP6, FP6, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_32x32x64_f8f6f4", "cbsz:0b010 blgp:0b010");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMBF6_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<BF6, BF6, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_16x16x128_f8f6f4", "cbsz:0b011 blgp:0b011");
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMBF6_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<BF6, BF6, float>(gemm);
        check_mfma_f8f6f4(m_context, "v_mfma_f32_32x32x64_f8f6f4", "cbsz:0b011 blgp:0b011");
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) * 6 / 8 / 4) / 64, 6, 20, true);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMFP8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<FP8, FP8, float>(gemm);
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 4) / 64, 8, 20);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMBF8_16x16x128_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(16, 16, 128);
        basicGEMM<BF8, BF8, float>(gemm);
        check_GEMMF8F6F4_TN(m_context, (16 * 16 + (16 * 128) / 4) / 64, 8, 20);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMFP8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<FP8, FP8, float>(gemm);
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 4) / 64, 8, 20);
    }

    TEST_P(GEMMF8F6F4TestGPU, GPU_BasicGEMMBF8_32x32x64_TN)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);
        auto gemm = setup_GEMMF8F6F4_TN(32, 32, 64);
        basicGEMM<BF8, BF8, float>(gemm);
        check_GEMMF8F6F4_TN(m_context, (32 * 32 + (32 * 64) / 4) / 64, 8, 20);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed2X2)
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

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed2X1)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 18);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed2X1UnrollK)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 20);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed1X2)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 18);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed1X2UnrollK)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b64"), 24);
        EXPECT_EQ(countSubstring(generatedCode, "ds_read_b128"), 8);
        EXPECT_EQ(countSubstring(generatedCode, "buffer_store_dwordx4"), 8);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed1x8)
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

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed1x8UnrollK)
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

        basicGEMM<Half>(gemm);
    }
    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed2x4)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 3);
        EXPECT_EQ(countSubstring(generatedCode, "v_pack_b32_f16"), 152);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed2x4UnrollK)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 6);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed4x2)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 3);
    }

    TEST_P(GEMMJammedTestGPU, GPU_BasicGEMMFP16Jammed4x2UnrollK)
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

        basicGEMM<Half>(gemm);

        std::string generatedCode = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(generatedCode, "ds_write_b128"), 12);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMLiteralStrides)
    {
        GEMMProblem gemm;
        gemm.packMultipleElementsInto1VGPR = true;
        gemm.transB                        = "N";

        gemm.literalStrides = true;
        basicGEMM<float>(gemm);
        std::string output_literalStrides = m_context->instructions()->toString();

        gemm.literalStrides = false;
        basicGEMM<float>(gemm);
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

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16AllLDS)
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

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMStoreDWave)
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
        basicGEMM<Half>(gemm);
        auto instructions0 = output();
        EXPECT_EQ(nonZeroDSReadOffsets(instructions0), std::set<int>{1024});

        gemm.splitStoreTileIntoWaveBlocks = false;
        basicGEMM<Half>(gemm);
        auto instructions1 = output();
        EXPECT_EQ(nonZeroDSReadOffsets(instructions1), std::set<int>{64});
    }

    TEST_P(GEMMTestGPU, GPU_BasicGEMMFP16AllLDSDebug)
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

        basicGEMM<Half>(gemm);
    }

    TEST_P(GEMMMixedF8F6F4TestGPU, GPU_BasicGEMMMixedF8F6F4)
    {
        auto [typeA, typeB, MFMAK]
            = std::get<std::tuple<rocRoller::DataType, rocRoller::DataType, int>>(GetParam());

        int wave_m = (MFMAK == 128) ? 16 : 32;
        int wave_n = (MFMAK == 128) ? 16 : 32;
        int wave_k = MFMAK;

        auto problem = setup_GEMMF8F6F4_TN(wave_m, wave_n, wave_k);

        basicGEMMMixed(typeA, typeB, problem);

        auto mfma = (MFMAK == 128) ? "v_mfma_f32_16x16x128_f8f6f4" : "v_mfma_f32_32x32x64_f8f6f4";

        std::string modifierA = "defaultModiferA";
        std::string modifierB = "defaultModiferB";

        if(typeA == rocRoller::DataType::FP8)
            modifierA = "cbsz:0b000";
        else if(typeA == rocRoller::DataType::BF8)
            modifierA = "cbsz:0b001";
        else if(typeA == rocRoller::DataType::FP6)
            modifierA = "cbsz:0b010";
        else if(typeA == rocRoller::DataType::BF6)
            modifierA = "cbsz:0b011";
        else if(typeA == rocRoller::DataType::FP4)
            modifierA = "cbsz:0b100";
        else
            Throw<FatalError>("Unhandled data type for mixed GEMM.", ShowValue(typeA));

        if(typeB == rocRoller::DataType::FP8)
            modifierB = "blgp:0b000";
        else if(typeB == rocRoller::DataType::BF8)
            modifierB = "blgp:0b001";
        else if(typeB == rocRoller::DataType::FP6)
            modifierB = "blgp:0b010";
        else if(typeB == rocRoller::DataType::BF6)
            modifierB = "blgp:0b011";
        else if(typeB == rocRoller::DataType::FP4)
            modifierB = "blgp:0b100";
        else
            Throw<FatalError>("Unhandled data type for mixed GEMM.", ShowValue(typeB));

        check_mfma_f8f6f4(m_context, mfma, modifierA + " " + modifierB);
    }

    TEST_P(GEMMMixedScaledTestGPU, GPU_BlockScaledGEMMMixed)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA_f8f6f4);

        auto [typeA, typeB, MFMAK, loadLDSScaleA, loadLDSScaleB] = std::get<1>(GetParam());

        int wave_m = (MFMAK == 128) ? 16 : 32;
        int wave_n = (MFMAK == 128) ? 16 : 32;
        int wave_k = MFMAK;

        auto problem = setup_GEMMF8F6F4_TN(wave_m, wave_n, wave_k);

        problem.scaleAMode = rocRoller::Operations::ScaleMode::Separate;
        problem.scaleBMode = rocRoller::Operations::ScaleMode::Separate;

        problem.loadLDSScaleA = loadLDSScaleA;
        problem.loadLDSScaleB = loadLDSScaleB;

        basicGEMMMixed(typeA, typeB, problem);
    }

    INSTANTIATE_TEST_SUITE_P(GEMMTest, GEMMTestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(GEMMF8F6F4Test, GEMMF8F6F4TestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(GEMMF8Test, GEMMF8TestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(GEMMJammedTest, GEMMJammedTestGPU, currentGPUISA());

    INSTANTIATE_TEST_SUITE_P(
        GEMMMixedF8F6F4Test,
        GEMMMixedF8F6F4TestGPU,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                                rocRoller::DataType::BF8,
                                                                rocRoller::DataType::FP6,
                                                                rocRoller::DataType::BF6,
                                                                rocRoller::DataType::FP4),
                                              ::testing::Values(rocRoller::DataType::FP8,
                                                                rocRoller::DataType::BF8,
                                                                rocRoller::DataType::FP6,
                                                                rocRoller::DataType::BF6,
                                                                rocRoller::DataType::FP4),
                                              ::testing::Values(64, 128))));

    INSTANTIATE_TEST_SUITE_P(
        GEMMMixedScaledTest,
        GEMMMixedScaledTestGPU,
        ::testing::Combine(currentGPUISA(),
                           ::testing::Combine(::testing::Values(rocRoller::DataType::FP8,
                                                                rocRoller::DataType::BF8,
                                                                rocRoller::DataType::FP6,
                                                                rocRoller::DataType::BF6,
                                                                rocRoller::DataType::FP4),
                                              ::testing::Values(rocRoller::DataType::FP8,
                                                                rocRoller::DataType::BF8,
                                                                rocRoller::DataType::FP6,
                                                                rocRoller::DataType::BF6,
                                                                rocRoller::DataType::FP4),
                                              ::testing::Values(64, 128),
                                              ::testing::Values(false, true),
                                              ::testing::Values(false, true))));
}
