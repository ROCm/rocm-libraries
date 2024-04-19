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
                int m_tagA, m_tagB, m_tagC, m_tagD;

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
                                 std::vector<A> const&        h_A,
                                 std::vector<B> const&        h_B,
                                 std::vector<C> const&        h_C,
                                 std::vector<D>&              h_D)
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
                        result.benchmarkResults.correct = this->validate(h_A, h_B, h_C, h_D);
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

                    auto tagTensorA = -1;
                    //TODO: Handle transposed matrices more elegantly
                    switch(m_solutionParams.problemParams.transA)
                    {
                    case TransposeType::T:
                        tagTensorA = command->addOperation(Operations::Tensor(
                            2, TypeInfo<A>::Var.dataType, {(size_t)0, (size_t)1})); // AT
                        break;
                    case TransposeType::N:
                        tagTensorA = command->addOperation(
                            Operations::Tensor(2, TypeInfo<A>::Var.dataType, {(size_t)1})); // AN
                        break;
                    default:
                        Throw<FatalError>("Bad transpose option");
                    }
                    m_tagA = command->addOperation(Operations::T_Load_Tiled(tagTensorA));

                    auto tagTensorB = -1;
                    //TODO: Handle transposed matrices more elegantly
                    switch(m_solutionParams.problemParams.transB)
                    {
                    case TransposeType::T:
                        tagTensorB = command->addOperation(Operations::Tensor(
                            2, TypeInfo<B>::Var.dataType, {(size_t)0, (size_t)1})); // BT
                        break;
                    case TransposeType::N:
                        tagTensorB
                            = command->addOperation(Operations::Tensor(2,
                                                                       TypeInfo<B>::Var.dataType,
                                                                       {
                                                                           (size_t)1,
                                                                       })); // BN
                        break;
                    default:
                        Throw<FatalError>("Bad transpose option");
                    }
                    m_tagB = command->addOperation(Operations::T_Load_Tiled(tagTensorB));

                    if(!no_beta)
                    {
                        auto tagTensorC = command->addOperation(
                            Operations::Tensor(2, TypeInfo<C>::Var.dataType, {(size_t)1})); // C
                        m_tagC = command->addOperation(Operations::T_Load_Tiled(tagTensorC));

                        auto tagScalarAlpha
                            = command->addOperation(Operations::Scalar(DataType::Float)); // alpha
                        auto tagLoadAlpha
                            = command->addOperation(Operations::T_Load_Scalar(tagScalarAlpha));

                        auto tagScalarBeta
                            = command->addOperation(Operations::Scalar(DataType::Float)); // beta
                        auto tagLoadBeta
                            = command->addOperation(Operations::T_Load_Scalar(tagScalarBeta));

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

                        auto tagTensorD = command->addOperation(
                            Operations::Tensor(2, TypeInfo<D>::Var.dataType, {(size_t)1})); // D
                        command->addOperation(Operations::T_Store_Tiled(m_tagD, tagTensorD));
                    }
                    else
                    {
                        m_tagD = command->addOperation(Operations::T_Mul(m_tagA, m_tagB)); // A * B
                        auto tagTensorD = command->addOperation(
                            Operations::Tensor(2, TypeInfo<D>::Var.dataType, {(size_t)1})); // D
                        command->addOperation(Operations::T_Store_Tiled(m_tagD, tagTensorD));
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
                        kernelOptions->unrollK           = 2;
                        kernelOptions->prefetch          = true;
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
                        // wave tile sizes
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 2;
                        wave_b = 1;
                    }
                    else if constexpr(std::is_same_v<A, Half> && std::is_same_v<B, Half>)
                    {
                        // wave tile sizes
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 8;
                        wave_b = 1;
                    }
                    else
                    {
                        Throw<FatalError>("Unsupported datatype combination in client");
                    }

                    uint wavetile_per_wavefront_m = wavefrontSize * m_solutionParams.macM / wave_m
                                                    / m_solutionParams.workgroupSizeX;
                    uint wavetile_per_wavefront_n
                        = m_solutionParams.macN / wave_n / m_solutionParams.workgroupSizeY;

                    AssertFatal(m_solutionParams.macM % (wave_m * wavetile_per_wavefront_m) == 0,
                                "WaveTile size mismatch (M)",
                                ShowValue(m_solutionParams.macM),
                                ShowValue(wave_m),
                                ShowValue(wavetile_per_wavefront_m));
                    AssertFatal(m_solutionParams.macN % (wave_n * wavetile_per_wavefront_n) == 0,
                                "WaveTile size mismatch (N)",
                                ShowValue(m_solutionParams.macN),
                                ShowValue(wave_n),
                                ShowValue(wavetile_per_wavefront_n));

                    auto params = std::make_shared<CommandParameters>();
                    params->setManualKernelDimension(2);
                    // TODO: Calculate these values internally based on workgroup sizes.
                    params->setWaveTilesPerWavefront(wavetile_per_wavefront_m,
                                                     wavetile_per_wavefront_n);

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
                        {static_cast<uint>(m_solutionParams.macM / wave_m
                                           / wavetile_per_wavefront_m),
                         static_cast<uint>(m_solutionParams.macN / wave_n
                                           / wavetile_per_wavefront_n)});

                    auto kernelName = m_solutionParams.generateKernelName();

                    // Build GEMM kernel
                    return CommandKernel(BenchmarkSolution::m_command,
                                         kernelName,
                                         params,
                                         postParams,
                                         kernelOptions);
                }

                KernelArguments makeArgs(std::shared_ptr<A> m_dA,
                                         std::shared_ptr<B> m_dB,
                                         std::shared_ptr<C> m_dC,
                                         std::shared_ptr<D> m_dD)
                {
                    bool            logArgs = Log::getLogger()->should_log(spdlog::level::debug);
                    KernelArguments runtimeArgs(logArgs);

                    bool no_beta = m_solutionParams.problemParams.beta == 0.0
                                   && m_solutionParams.problemParams.alpha == 1.0;

                    runtimeArgs.append("A", m_dA.get());
                    runtimeArgs.append("d_a_limit",
                                       (size_t)m_solutionParams.problemParams.m
                                           * m_solutionParams.problemParams.k);

                    runtimeArgs.append("d_a_size_0", (size_t)m_solutionParams.problemParams.m);
                    runtimeArgs.append("d_a_size_1", (size_t)m_solutionParams.problemParams.k);

                    //TODO: Handle transposed matrices more elegantly
                    if(m_solutionParams.problemParams.transA == TransposeType::T)
                    {
                        runtimeArgs.append("d_a_stride_0",
                                           (size_t)m_solutionParams.problemParams.k);
                        runtimeArgs.append("d_a_stride_1", (size_t)1);
                    }
                    else
                    {
                        runtimeArgs.append("d_a_stride_0", (size_t)1);
                        runtimeArgs.append("d_a_stride_1",
                                           (size_t)m_solutionParams.problemParams.m);
                    }

                    runtimeArgs.append("B", m_dB.get());
                    runtimeArgs.append("d_b_limit",
                                       (size_t)m_solutionParams.problemParams.k
                                           * m_solutionParams.problemParams.n);

                    runtimeArgs.append("d_b_size_0", (size_t)m_solutionParams.problemParams.k);
                    runtimeArgs.append("d_b_size_1", (size_t)m_solutionParams.problemParams.n);

                    //TODO: Handle transposed matrices more elegantly
                    if(m_solutionParams.problemParams.transB == TransposeType::T)
                    {
                        runtimeArgs.append("d_b_stride_0",
                                           (size_t)m_solutionParams.problemParams.n);
                        runtimeArgs.append("d_b_stride_1", (size_t)1);
                    }
                    else
                    {
                        runtimeArgs.append("d_b_stride_0", (size_t)1);
                        runtimeArgs.append("d_b_stride_1",
                                           (size_t)m_solutionParams.problemParams.k);
                    }

                    if(!no_beta)
                    {
                        runtimeArgs.append("C", m_dC.get());
                        runtimeArgs.append("d_c_limit",
                                           (size_t)m_solutionParams.problemParams.m
                                               * m_solutionParams.problemParams.n);
                        runtimeArgs.append("d_c_size_0", (size_t)m_solutionParams.problemParams.m);
                        runtimeArgs.append("d_c_size_1", (size_t)m_solutionParams.problemParams.n);
                        runtimeArgs.append("d_c_stride_0", (size_t)1);
                        runtimeArgs.append("d_c_stride_1",
                                           (size_t)m_solutionParams.problemParams.m);

                        runtimeArgs.append("alpha", m_solutionParams.problemParams.alpha);

                        runtimeArgs.append("beta", m_solutionParams.problemParams.beta);
                    }

                    runtimeArgs.append("D", m_dD.get());
                    runtimeArgs.append("d_d_limit",
                                       (size_t)m_solutionParams.problemParams.m
                                           * m_solutionParams.problemParams.n);
                    runtimeArgs.append("d_d_size_0", (size_t)m_solutionParams.problemParams.m);
                    runtimeArgs.append("d_d_size_1", (size_t)m_solutionParams.problemParams.n);
                    runtimeArgs.append("d_d_stride_0", (size_t)1);
                    runtimeArgs.append("d_d_stride_1", (size_t)m_solutionParams.problemParams.m);

                    if(logArgs)
                        Log::getLogger()->debug(runtimeArgs.toString());

                    return runtimeArgs;
                }
            };
        }
    }
}
