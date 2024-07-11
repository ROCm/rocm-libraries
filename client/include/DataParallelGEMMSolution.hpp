#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "GEMMParameters.hpp"
#include "GEMMSolution.hpp"
#include "visualize.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            template <typename A, typename B, typename C, typename D>
            class DataParallelGEMMSolution : public GEMMSolution<A, B, C, D>
            {
                Operations::OperationTag m_tagA, m_tagB, m_tagC, m_tagD;
                Operations::OperationTag m_tagTensorA, m_tagTensorB, m_tagTensorC, m_tagScalarAlpha,
                    m_tagScalarBeta, m_tagTensorD;

            public:
                DataParallelGEMMSolution(SolutionParameters const& solutionParams)
                    : GEMMSolution<A, B, C, D>(solutionParams.problemParams)
                {
                    m_solutionParams = solutionParams;
                    this->m_command  = makeCommand();

                    this->m_kernel = std::make_shared<CommandKernel>(
                        makeKernel(makeKernelOptions(),
                                   m_solutionParams.problemParams.m / m_solutionParams.macM,
                                   m_solutionParams.problemParams.n / m_solutionParams.macN));
                }

                Result benchmark(Client::RunParameters const& runParams,
                                 bool                         checkResult,
                                 bool                         doVisualize,
                                 std::vector<typename UnsegmentedTypeOf<A>::type> const& h_A,
                                 std::vector<typename UnsegmentedTypeOf<B>::type> const& h_B,
                                 std::vector<C> const&                                   h_C,
                                 std::vector<D>&                                         h_D)
                {
                    Result result;
                    result.solutionParams = m_solutionParams;

                    auto d_A = make_shared_device(h_A);
                    auto d_B = make_shared_device(h_B);
                    auto d_C = make_shared_device(h_C);
                    auto d_D = make_shared_device(h_D);

                    auto runtimeArgs = makeArgs(d_A, d_B, d_C, d_D);

                    if(doVisualize)
                    {
                        Client::visualize(this->m_command, *(this->m_kernel), runtimeArgs);
                    }

                    result.benchmarkResults
                        = GEMMSolution<A, B, C, D>::benchmark(runParams, runtimeArgs);

                    if(checkResult)
                    {
                        AssertFatal(
                            hipMemcpy(h_D.data(),
                                      d_D.get(),
                                      this->m_problemParams.m * this->m_problemParams.n * sizeof(D),
                                      hipMemcpyDeviceToHost)
                            == (hipError_t)HIP_SUCCESS);

                        result.benchmarkResults.checked = true;

                        auto [correct, rnorm]           = this->validate(h_A, h_B, h_C, h_D);
                        result.benchmarkResults.correct = correct;
                        result.benchmarkResults.rnorm   = rnorm;
                    }

                    return result;
                }

            protected:
                DataParallelGEMMSolution() {}
                SolutionParameters m_solutionParams;

                CommandPtr makeCommand()
                {
                    auto command = std::make_shared<Command>();

                    bool no_beta = m_solutionParams.problemParams.beta == 0.0
                                   && m_solutionParams.problemParams.alpha == 1.0;

                    //TODO: Handle transposed matrices more elegantly
                    switch(m_solutionParams.problemParams.transA)
                    {
                    case TransposeType::T:
                        m_tagTensorA = command->addOperation(Operations::Tensor(
                            2, TypeInfo<A>::Var.dataType, {(size_t)0, (size_t)1})); // AT
                        break;
                    case TransposeType::N:
                        m_tagTensorA = command->addOperation(
                            Operations::Tensor(2, TypeInfo<A>::Var.dataType, {(size_t)1})); // AN
                        break;
                    default:
                        Throw<FatalError>("Bad transpose option");
                    }
                    m_tagA = command->addOperation(Operations::T_Load_Tiled(m_tagTensorA));

                    //TODO: Handle transposed matrices more elegantly
                    switch(m_solutionParams.problemParams.transB)
                    {
                    case TransposeType::T:
                        m_tagTensorB = command->addOperation(Operations::Tensor(
                            2, TypeInfo<B>::Var.dataType, {(size_t)0, (size_t)1})); // BT
                        break;
                    case TransposeType::N:
                        m_tagTensorB
                            = command->addOperation(Operations::Tensor(2,
                                                                       TypeInfo<B>::Var.dataType,
                                                                       {
                                                                           (size_t)1,
                                                                       })); // BN
                        break;
                    default:
                        Throw<FatalError>("Bad transpose option");
                    }
                    m_tagB = command->addOperation(Operations::T_Load_Tiled(m_tagTensorB));

                    if(!no_beta)
                    {
                        m_tagTensorC = command->addOperation(
                            Operations::Tensor(2, TypeInfo<C>::Var.dataType, {(size_t)1})); // C
                        m_tagC = command->addOperation(Operations::T_Load_Tiled(m_tagTensorC));

                        m_tagScalarAlpha
                            = command->addOperation(Operations::Scalar(DataType::Float)); // alpha
                        auto tagLoadAlpha
                            = command->addOperation(Operations::T_Load_Scalar(m_tagScalarAlpha));

                        m_tagScalarBeta
                            = command->addOperation(Operations::Scalar(DataType::Float)); // beta
                        auto tagLoadBeta
                            = command->addOperation(Operations::T_Load_Scalar(m_tagScalarBeta));

                        auto tagAB
                            = command->addOperation(Operations::T_Mul(m_tagA, m_tagB)); // A * B

                        Operations::T_Execute execute(command->getNextTag());
                        auto                  tagBetaC
                            = execute.addXOp(Operations::E_Mul(tagLoadBeta, m_tagC)); // beta * C
                        auto tagAlphaAB = execute.addXOp(
                            Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)
                        if(m_solutionParams.betaInFma)
                        {
                            m_tagD = execute.addXOp(Operations::E_Add(
                                tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
                        }
                        else
                        {
                            m_tagD = execute.addXOp(Operations::E_Add(
                                tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
                        }
                        command->addOperation(std::move(execute));

                        m_tagTensorD = command->addOperation(
                            Operations::Tensor(2, TypeInfo<D>::Var.dataType, {(size_t)1})); // D
                        command->addOperation(Operations::T_Store_Tiled(m_tagD, m_tagTensorD));
                    }
                    else
                    {
                        m_tagD = command->addOperation(Operations::T_Mul(m_tagA, m_tagB)); // A * B
                        m_tagTensorD = command->addOperation(
                            Operations::Tensor(2, TypeInfo<D>::Var.dataType, {(size_t)1})); // D
                        command->addOperation(Operations::T_Store_Tiled(m_tagD, m_tagTensorD));
                    }

                    return command;
                }

                std::shared_ptr<KernelOptions> makeKernelOptions()
                {
                    auto kernelOptions     = std::make_shared<KernelOptions>();
                    kernelOptions->unrollX = m_solutionParams.unrollX;
                    kernelOptions->unrollY = m_solutionParams.unrollY;

                    if(m_solutionParams.prefetch)
                    {
                        kernelOptions->prefetch          = true;
                        kernelOptions->unrollK           = m_solutionParams.prefetchInFlight;
                        kernelOptions->prefetchInFlight  = m_solutionParams.prefetchInFlight;
                        kernelOptions->prefetchLDSFactor = m_solutionParams.prefetchLDSFactor;

                        if(m_solutionParams.prefetchLDSFactor != 0)
                        {
                            kernelOptions->prefetchMixMemOps = true;
                        }
                    }
                    else
                    {
                        kernelOptions->prefetch = false;
                    }

                    if(m_solutionParams.matchMemoryAccess)
                    {
                        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_A]
                            = m_solutionParams.problemParams.transA == TransposeType::T;
                        kernelOptions->transposeMemoryAccess[LayoutType::MATRIX_B]
                            = m_solutionParams.problemParams.transB == TransposeType::T;
                    }

                    kernelOptions->setNextFreeVGPRToMax = false;
                    return kernelOptions;
                }

                CommandKernel makeKernel(std::shared_ptr<KernelOptions> kernelOptions,
                                         uint                           num_workgroup_x,
                                         uint                           num_workgroup_y)
                {
                    AssertFatal(m_solutionParams.problemParams.m % m_solutionParams.macM == 0,
                                "MacroTile size mismatch (M)");
                    AssertFatal(m_solutionParams.problemParams.n % m_solutionParams.macN == 0,
                                "MacroTile size mismatch (N)");

                    AssertFatal(m_solutionParams.workgroupSizeX % wavefrontSize == 0,
                                "Workgroup Size X must be multiply of wave front size");

                    int wave_m = 0, wave_n = 0, wave_k = 0, wave_b = 0;

                    bool no_beta = m_solutionParams.problemParams.beta == 0.0
                                   && m_solutionParams.problemParams.alpha == 1.0;

                    if constexpr(std::is_same_v<A, float> && std::is_same_v<B, float>)
                    {
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 2;
                        wave_b = 1;
                    }
                    else if constexpr(std::is_same_v<A, Half> && std::is_same_v<B, Half>)
                    {
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 8;
                        wave_b = 1;
                    }
                    else if constexpr((std::is_same_v<A, FP8> && std::is_same_v<B, FP8>)
                                      || (std::is_same_v<A, BF8> && std::is_same_v<B, BF8>))
                    {
                        wave_m = 16;
                        wave_n = 16;
                        wave_k = 32;
                        wave_b = 1;
                    }
                    else if constexpr((std::is_same_v<A, FP4> && std::is_same_v<B, FP4>)
                                      || (std::is_same_v<A, FP6> && std::is_same_v<B, FP6>)
                                      || (std::is_same_v<A, BF6> && std::is_same_v<B, BF6>))
                    {
                        wave_m = 16;
                        wave_n = 16;
                        wave_k = 128;
                        wave_b = 1;
                    }
                    else
                    {
                        Throw<FatalError>("Unsupported datatype combination in client");
                    }

                    if(m_solutionParams.waveM > 0)
                        wave_m = m_solutionParams.waveM;
                    if(m_solutionParams.waveN > 0)
                        wave_n = m_solutionParams.waveN;
                    if(m_solutionParams.waveK > 0)
                        wave_k = m_solutionParams.waveK;
                    if(m_solutionParams.waveB > 0)
                        wave_b = m_solutionParams.waveB;

                    AssertFatal(m_solutionParams.macM * m_solutionParams.macK
                                        * DataTypeInfo::Get(TypeInfo<A>::Var.dataType).elementBytes
                                    > wave_m * wave_k,
                                "Not enough elements (A).");
                    AssertFatal(m_solutionParams.macN * m_solutionParams.macK
                                        * DataTypeInfo::Get(TypeInfo<A>::Var.dataType).elementBytes
                                    > wave_n * wave_k,
                                "Not enough elements (B).");

                    uint wavetilePerWavefrontM = wavefrontSize * m_solutionParams.macM / wave_m
                                                 / m_solutionParams.workgroupSizeX;
                    uint wavetilePerWavefrontN
                        = m_solutionParams.macN / wave_n / m_solutionParams.workgroupSizeY;

                    AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch.");
                    AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch.");

                    AssertFatal(m_solutionParams.macM % (wave_m * wavetilePerWavefrontM) == 0,
                                "WaveTile size mismatch (M)",
                                ShowValue(m_solutionParams.macM),
                                ShowValue(wave_m),
                                ShowValue(wavetilePerWavefrontM));
                    AssertFatal(m_solutionParams.macN % (wave_n * wavetilePerWavefrontN) == 0,
                                "WaveTile size mismatch (N)",
                                ShowValue(m_solutionParams.macN),
                                ShowValue(wave_n),
                                ShowValue(wavetilePerWavefrontN));

                    auto params = std::make_shared<CommandParameters>();
                    params->setManualKernelDimension(2);
                    // TODO: Calculate these values internally based on workgroup sizes.
                    params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);

                    auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                        {m_solutionParams.macM, m_solutionParams.macK},
                        LayoutType::MATRIX_A,
                        {wave_m, wave_n, wave_k, wave_b},
                        m_solutionParams.loadLDSA ? MemoryType::LDS : MemoryType::WAVE);
                    auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                        {m_solutionParams.macK, m_solutionParams.macN},
                        LayoutType::MATRIX_B,
                        {wave_m, wave_n, wave_k, wave_b},
                        m_solutionParams.loadLDSB ? MemoryType::LDS : MemoryType::WAVE);
                    auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                        {m_solutionParams.macM, m_solutionParams.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {wave_m, wave_n, wave_k, wave_b});
                    auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
                        {m_solutionParams.macM, m_solutionParams.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {wave_m, wave_n, wave_k, wave_b},
                        m_solutionParams.storeLDSD ? MemoryType::JAMMED_WAVE_LDS
                                                   : MemoryType::WAVE);

                    params->setDimensionInfo(m_tagA, macTileA);
                    params->setDimensionInfo(m_tagB, macTileB);
                    if(!no_beta)
                        params->setDimensionInfo(m_tagC, macTileC);
                    // TODO Fix MemoryType promotion (JAMMED_WAVE_LDS)
                    params->setDimensionInfo(m_tagD, macTileD);

                    uint workgroup_size_x
                        = m_solutionParams.workgroupSizeX * m_solutionParams.workgroupSizeY;
                    uint workgroup_size_y = 1;

                    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x
                                                                       * workgroup_size_x);
                    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y
                                                                       * workgroup_size_y);
                    auto NZ = std::make_shared<Expression::Expression>(1u);

                    params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
                    params->setManualWorkitemCount({NX, NY, NZ});

                    if(m_solutionParams.scheduler != "")
                    {
                        auto schedulerValue = fromString<Scheduling::SchedulerProcedure>(
                            m_solutionParams.scheduler);
                        Settings::getInstance()->set(Settings::Scheduler, schedulerValue);
                    }

                    auto postParams = std::make_shared<CommandParameters>();
                    postParams->setManualWavefrontCount(
                        {static_cast<uint>(m_solutionParams.macM / wave_m / wavetilePerWavefrontM),
                         static_cast<uint>(m_solutionParams.macN / wave_n
                                           / wavetilePerWavefrontN)});

                    auto kernelName = m_solutionParams.generateKernelName();

                    // Build GEMM kernel
                    return CommandKernel(BenchmarkSolution::m_command,
                                         kernelName,
                                         params,
                                         postParams,
                                         kernelOptions);
                }

                CommandArguments makeArgs(std::shared_ptr<typename UnsegmentedTypeOf<A>::type> m_dA,
                                          std::shared_ptr<typename UnsegmentedTypeOf<B>::type> m_dB,
                                          std::shared_ptr<C>                                   m_dC,
                                          std::shared_ptr<D>                                   m_dD)
                {
                    CommandArguments commandArgs = this->m_command->createArguments();

                    bool no_beta = m_solutionParams.problemParams.beta == 0.0
                                   && m_solutionParams.problemParams.alpha == 1.0;

                    commandArgs.setArgument(m_tagTensorA, ArgumentType::Value, (A*)m_dA.get());
                    commandArgs.setArgument(m_tagTensorB, ArgumentType::Value, (B*)m_dB.get());

                    commandArgs.setArgument(m_tagTensorA,
                                            ArgumentType::Limit,
                                            (size_t)m_solutionParams.problemParams.m
                                                * m_solutionParams.problemParams.k);
                    commandArgs.setArgument(m_tagTensorB,
                                            ArgumentType::Limit,
                                            (size_t)m_solutionParams.problemParams.k
                                                * m_solutionParams.problemParams.n);

                    commandArgs.setArgument(m_tagTensorA,
                                            ArgumentType::Size,
                                            0,
                                            (size_t)m_solutionParams.problemParams.m);
                    commandArgs.setArgument(m_tagTensorA,
                                            ArgumentType::Size,
                                            1,
                                            (size_t)m_solutionParams.problemParams.k);
                    commandArgs.setArgument(m_tagTensorB,
                                            ArgumentType::Size,
                                            0,
                                            (size_t)m_solutionParams.problemParams.k);
                    commandArgs.setArgument(m_tagTensorB,
                                            ArgumentType::Size,
                                            1,
                                            (size_t)m_solutionParams.problemParams.n);

                    //TODO: Handle transposed matrices more elegantly
                    if(m_solutionParams.problemParams.transA == TransposeType::T)
                    {
                        commandArgs.setArgument(m_tagTensorA,
                                                ArgumentType::Stride,
                                                0,
                                                (size_t)m_solutionParams.problemParams.k);
                        commandArgs.setArgument(m_tagTensorA, ArgumentType::Stride, 1, (size_t)1);
                    }
                    else
                    {
                        commandArgs.setArgument(m_tagTensorA, ArgumentType::Stride, 0, (size_t)1);
                        commandArgs.setArgument(m_tagTensorA,
                                                ArgumentType::Stride,
                                                1,
                                                (size_t)m_solutionParams.problemParams.m);
                    }

                    //TODO: Handle transposed matrices more elegantly
                    if(m_solutionParams.problemParams.transB == TransposeType::T)
                    {
                        commandArgs.setArgument(m_tagTensorB,
                                                ArgumentType::Stride,
                                                0,
                                                (size_t)m_solutionParams.problemParams.n);
                        commandArgs.setArgument(m_tagTensorB, ArgumentType::Stride, 1, (size_t)1);
                    }
                    else
                    {
                        commandArgs.setArgument(m_tagTensorB, ArgumentType::Stride, 0, (size_t)1);
                        commandArgs.setArgument(m_tagTensorB,
                                                ArgumentType::Stride,
                                                1,
                                                (size_t)m_solutionParams.problemParams.k);
                    }

                    if(!no_beta)
                    {
                        commandArgs.setArgument(m_tagTensorC, ArgumentType::Value, m_dC.get());
                        commandArgs.setArgument(m_tagTensorC,
                                                ArgumentType::Limit,
                                                (size_t)m_solutionParams.problemParams.m
                                                    * m_solutionParams.problemParams.n);
                        commandArgs.setArgument(m_tagTensorC,
                                                ArgumentType::Size,
                                                0,
                                                (size_t)m_solutionParams.problemParams.m);
                        commandArgs.setArgument(m_tagTensorC,
                                                ArgumentType::Size,
                                                1,
                                                (size_t)m_solutionParams.problemParams.n);
                        commandArgs.setArgument(m_tagTensorC, ArgumentType::Stride, 0, (size_t)1);
                        commandArgs.setArgument(m_tagTensorC,
                                                ArgumentType::Stride,
                                                1,
                                                (size_t)m_solutionParams.problemParams.m);
                        commandArgs.setArgument(m_tagScalarAlpha,
                                                ArgumentType::Value,
                                                m_solutionParams.problemParams.alpha);
                        commandArgs.setArgument(m_tagScalarBeta,
                                                ArgumentType::Value,
                                                m_solutionParams.problemParams.beta);
                    }

                    commandArgs.setArgument(m_tagTensorD, ArgumentType::Value, m_dD.get());
                    commandArgs.setArgument(m_tagTensorD,
                                            ArgumentType::Limit,
                                            (size_t)m_solutionParams.problemParams.m
                                                * m_solutionParams.problemParams.n);
                    commandArgs.setArgument(m_tagTensorD,
                                            ArgumentType::Size,
                                            0,
                                            (size_t)m_solutionParams.problemParams.m);
                    commandArgs.setArgument(m_tagTensorD,
                                            ArgumentType::Size,
                                            1,
                                            (size_t)m_solutionParams.problemParams.n);
                    commandArgs.setArgument(m_tagTensorD, ArgumentType::Stride, 0, (size_t)1);
                    commandArgs.setArgument(m_tagTensorD,
                                            ArgumentType::Stride,
                                            1,
                                            (size_t)m_solutionParams.problemParams.m);

                    return commandArgs;
                }
            };
        }
    }
}
