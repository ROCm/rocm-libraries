#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "../../test/unit/TensorDescriptor.hpp"

#include "GEMMParameters.hpp"
#include "GEMMSolution.hpp"
#include "visualize.hpp"

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            class DataParallelGEMMSolution : public GEMMSolution
            {
                Operations::OperationTag m_tagA, m_tagB, m_tagC, m_tagD;
                Operations::OperationTag m_tagTensorA, m_tagTensorB, m_tagTensorC, m_tagScalarAlpha,
                    m_tagScalarBeta, m_tagTensorD;

            public:
                using GEMMSolution::GEMMSolution;

                virtual ABCDTags getABCDTags() const override
                {
                    return {m_tagTensorA, m_tagTensorB, m_tagTensorC, m_tagTensorD};
                }

            protected:
                virtual CommandPtr makeCommand(SolutionParameters const& solutionParams) override
                {
                    auto command = std::make_shared<Command>();

                    auto typeA = getDataTypeFromString(solutionParams.typeA);
                    auto typeB = getDataTypeFromString(solutionParams.typeB);
                    auto typeC = getDataTypeFromString(solutionParams.typeC);
                    auto typeD = getDataTypeFromString(solutionParams.typeD);

                    //TODO: Handle transposed matrices more elegantly
                    switch(solutionParams.transA)
                    {
                    case TransposeType::T:
                        m_tagTensorA = command->addOperation(
                            Operations::Tensor(2, typeA, {(size_t)0, (size_t)1})); // AT
                        break;
                    case TransposeType::N:
                        m_tagTensorA = command->addOperation(
                            Operations::Tensor(2, typeA, {(size_t)1})); // AN
                        break;
                    default:
                        Throw<FatalError>("Bad transpose option");
                    }
                    m_tagA = command->addOperation(Operations::T_Load_Tiled(m_tagTensorA));

                    //TODO: Handle transposed matrices more elegantly
                    switch(solutionParams.transB)
                    {
                    case TransposeType::T:
                        m_tagTensorB = command->addOperation(
                            Operations::Tensor(2, typeB, {(size_t)0, (size_t)1})); // BT
                        break;
                    case TransposeType::N:
                        m_tagTensorB = command->addOperation(Operations::Tensor(2,
                                                                                typeB,
                                                                                {
                                                                                    (size_t)1,
                                                                                })); // BN
                        break;
                    default:
                        Throw<FatalError>("Bad transpose option");
                    }
                    m_tagB = command->addOperation(Operations::T_Load_Tiled(m_tagTensorB));

                    m_tagTensorC
                        = command->addOperation(Operations::Tensor(2, typeC, {(size_t)1})); // C
                    m_tagC = command->addOperation(Operations::T_Load_Tiled(m_tagTensorC));

                    m_tagScalarAlpha
                        = command->addOperation(Operations::Scalar(DataType::Float)); // alpha
                    auto tagLoadAlpha
                        = command->addOperation(Operations::T_Load_Scalar(m_tagScalarAlpha));

                    m_tagScalarBeta
                        = command->addOperation(Operations::Scalar(DataType::Float)); // beta
                    auto tagLoadBeta
                        = command->addOperation(Operations::T_Load_Scalar(m_tagScalarBeta));

                    auto tagAB = command->addOperation(Operations::T_Mul(m_tagA, m_tagB)); // A * B

                    Operations::T_Execute execute(command->getNextTag());
                    auto                  tagBetaC
                        = execute.addXOp(Operations::E_Mul(tagLoadBeta, m_tagC)); // beta * C
                    auto tagAlphaAB
                        = execute.addXOp(Operations::E_Mul(tagLoadAlpha, tagAB)); // alpha * (A * B)
                    if(solutionParams.betaInFma)
                    {
                        m_tagD = execute.addXOp(
                            Operations::E_Add(tagBetaC, tagAlphaAB)); // beta * C + alpha * (A * B)
                    }
                    else
                    {
                        m_tagD = execute.addXOp(
                            Operations::E_Add(tagAlphaAB, tagBetaC)); // alpha * (A * B) + beta * C
                    }
                    command->addOperation(std::move(execute));

                    m_tagTensorD
                        = command->addOperation(Operations::Tensor(2, typeD, {(size_t)1})); // D
                    command->addOperation(Operations::T_Store_Tiled(m_tagD, m_tagTensorD));

                    return command;
                }

                virtual CommandParametersPtr
                    makeCommandParameters(SolutionParameters const& solutionParams) override
                {
                    auto params = std::make_shared<CommandParameters>();

                    int wave_m = 0, wave_n = 0, wave_k = 0, wave_b = 0;

                    auto typeA = getDataTypeFromString(solutionParams.typeA);
                    auto typeB = getDataTypeFromString(solutionParams.typeB);
                    auto typeC = getDataTypeFromString(solutionParams.typeC);
                    auto typeD = getDataTypeFromString(solutionParams.typeD);

                    if(typeA == DataType::Float && typeB == DataType::Float)
                    {
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 2;
                        wave_b = 1;
                    }
                    else if((typeA == DataType::Half && typeB == DataType::Half)
                            || (typeA == DataType::BFloat16 && typeB == DataType::BFloat16))
                    {
                        wave_m = 32;
                        wave_n = 32;
                        wave_k = 8;
                        wave_b = 1;
                    }
                    else if((typeA == DataType::FP8 && typeB == DataType::FP8)
                            || (typeA == DataType::BF8 && typeB == DataType::BF8))
                    {
                        wave_m = 16;
                        wave_n = 16;
                        wave_k = 32;
                        wave_b = 1;
                    }
                    else
                    {
                        Throw<FatalError>("Unsupported datatype combination in client");
                    }

                    if(solutionParams.waveM > 0)
                        wave_m = solutionParams.waveM;
                    if(solutionParams.waveN > 0)
                        wave_n = solutionParams.waveN;
                    if(solutionParams.waveK > 0)
                        wave_k = solutionParams.waveK;
                    if(solutionParams.waveB > 0)
                        wave_b = solutionParams.waveB;

                    AssertFatal(solutionParams.macM * solutionParams.macK
                                        * DataTypeInfo::Get(typeA).elementBytes
                                    > wave_m * wave_k,
                                "Not enough elements (A).");
                    AssertFatal(solutionParams.macN * solutionParams.macK
                                        * DataTypeInfo::Get(typeA).elementBytes
                                    > wave_n * wave_k,
                                "Not enough elements (B).");

                    uint wavefrontSize         = 64;
                    uint wavetilePerWavefrontM = wavefrontSize * solutionParams.macM / wave_m
                                                 / solutionParams.workgroupSizeX;
                    uint wavetilePerWavefrontN
                        = solutionParams.macN / wave_n / solutionParams.workgroupSizeY;

                    AssertFatal(wavetilePerWavefrontM > 0, "WaveTile size mismatch.");
                    AssertFatal(wavetilePerWavefrontN > 0, "WaveTile size mismatch.");

                    AssertFatal(solutionParams.macM % (wave_m * wavetilePerWavefrontM) == 0,
                                "WaveTile size mismatch (M)",
                                ShowValue(solutionParams.macM),
                                ShowValue(wave_m),
                                ShowValue(wavetilePerWavefrontM));
                    AssertFatal(solutionParams.macN % (wave_n * wavetilePerWavefrontN) == 0,
                                "WaveTile size mismatch (N)",
                                ShowValue(solutionParams.macN),
                                ShowValue(wave_n),
                                ShowValue(wavetilePerWavefrontN));

                    params->setManualKernelDimension(2);
                    params->setWaveTilesPerWavefront(wavetilePerWavefrontM, wavetilePerWavefrontN);

                    auto macTileA = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macM, solutionParams.macK},
                        LayoutType::MATRIX_A,
                        {wave_m, wave_n, wave_k, wave_b},
                        solutionParams.loadLDSA ? MemoryType::LDS : MemoryType::WAVE);
                    auto macTileB = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macK, solutionParams.macN},
                        LayoutType::MATRIX_B,
                        {wave_m, wave_n, wave_k, wave_b},
                        solutionParams.loadLDSB ? MemoryType::LDS : MemoryType::WAVE);
                    auto macTileC = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macM, solutionParams.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {wave_m, wave_n, wave_k, wave_b});
                    auto macTileD = KernelGraph::CoordinateGraph::MacroTile(
                        {solutionParams.macM, solutionParams.macN},
                        LayoutType::MATRIX_ACCUMULATOR,
                        {wave_m, wave_n, wave_k, wave_b},
                        solutionParams.storeLDSD ? MemoryType::JAMMED_WAVE_LDS : MemoryType::WAVE);

                    params->setDimensionInfo(m_tagA, macTileA);
                    params->setDimensionInfo(m_tagB, macTileB);
                    params->setDimensionInfo(m_tagC, macTileC);
                    // TODO Fix MemoryType promotion (JAMMED_WAVE_LDS)
                    params->setDimensionInfo(m_tagD, macTileD);

                    params->unrollX = solutionParams.unrollX;
                    params->unrollY = solutionParams.unrollY;

                    if(solutionParams.prefetch)
                    {
                        params->prefetch          = true;
                        params->unrollK           = solutionParams.prefetchInFlight;
                        params->prefetchInFlight  = solutionParams.prefetchInFlight;
                        params->prefetchLDSFactor = solutionParams.prefetchLDSFactor;

                        if(solutionParams.prefetchLDSFactor != 0)
                        {
                            params->prefetchMixMemOps = true;
                        }
                    }
                    else
                    {
                        params->prefetch = false;
                    }

                    if(solutionParams.matchMemoryAccess)
                    {
                        params->transposeMemoryAccess[LayoutType::MATRIX_A]
                            = solutionParams.transA == TransposeType::T;
                        params->transposeMemoryAccess[LayoutType::MATRIX_B]
                            = solutionParams.transB == TransposeType::T;
                    }

                    uint workgroup_size_x
                        = solutionParams.workgroupSizeX * solutionParams.workgroupSizeY;
                    uint workgroup_size_y = 1;

                    params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});

                    params->setManualWavefrontCount(
                        {static_cast<uint>(solutionParams.macM / wave_m / wavetilePerWavefrontM),
                         static_cast<uint>(solutionParams.macN / wave_n / wavetilePerWavefrontN)});

                    return params;
                }

                virtual CommandLaunchParametersPtr
                    makeLaunchParameters(ProblemParameters const&  problemParams,
                                         SolutionParameters const& solutionParams,
                                         RunParameters const&      runParams) override
                {
                    uint num_workgroup_x = problemParams.m / solutionParams.macM;
                    uint num_workgroup_y = problemParams.n / solutionParams.macN;

                    uint workgroup_size_x
                        = solutionParams.workgroupSizeX * solutionParams.workgroupSizeY;
                    uint workgroup_size_y = 1;

                    auto NX = std::make_shared<Expression::Expression>(num_workgroup_x
                                                                       * workgroup_size_x);
                    auto NY = std::make_shared<Expression::Expression>(num_workgroup_y
                                                                       * workgroup_size_y);
                    auto NZ = std::make_shared<Expression::Expression>(1u);

                    auto launch = std::make_shared<CommandLaunchParameters>();
                    launch->setManualWorkitemCount({NX, NY, NZ});
                    return launch;
                }

                virtual CommandArguments
                    commandArguments(ProblemParameters const& problemParams,
                                     RunParameters const&     runParams) const override
                {
                    CommandArguments commandArgs = command()->createArguments();

                    size_t M = problemParams.m;
                    size_t N = problemParams.n;
                    size_t K = problemParams.k;

                    TensorDescriptor descA(getDataTypeFromString(problemParams.typeA),
                                           {M, K},
                                           problemParams.transA == TransposeType::T ? "T" : "N");
                    TensorDescriptor descB(getDataTypeFromString(problemParams.typeB),
                                           {K, N},
                                           problemParams.transB == TransposeType::T ? "T" : "N");

                    setCommandTensorArg(commandArgs, m_tagTensorA, descA, (float*)nullptr);
                    setCommandTensorArg(commandArgs, m_tagTensorB, descB, (float*)nullptr);

                    TensorDescriptor descC(getDataTypeFromString(problemParams.typeC), {M, N}, "N");
                    setCommandTensorArg(commandArgs, m_tagTensorC, descC, (float*)nullptr);

                    commandArgs.setArgument(
                        m_tagScalarAlpha, ArgumentType::Value, problemParams.alpha);
                    commandArgs.setArgument(
                        m_tagScalarBeta, ArgumentType::Value, problemParams.beta);

                    TensorDescriptor descD(getDataTypeFromString(problemParams.typeD), {M, N}, "N");
                    setCommandTensorArg(commandArgs, m_tagTensorD, descD, (float*)nullptr);

                    return commandArgs;
                }

                virtual void setPredicates(CommandKernelPtr          commandKernel,
                                           SolutionParameters const& solutionParams) override
                {
                    using namespace rocRoller::Expression;
                    // predicate building blocks
                    // A sizes
                    auto aSizes
                        = std::get<Operations::Tensor>(*(this->m_command->findTag(m_tagTensorA)))
                              .sizes();
                    std::vector<ExpressionPtr> aSizeExps(aSizes.size());
                    std::transform(aSizes.begin(), aSizes.end(), aSizeExps.begin(), [](auto arg) {
                        return arg->expression();
                    });

                    // parameters
                    auto unrollKExp = literal(commandKernel->getCommandParameters()->unrollK);
                    auto macKExp    = literal(solutionParams.macK);

                    // constants
                    auto zero = literal(0u);
                    auto one  = literal(1u);

                    // sanitize parameters
                    auto sanUnrollKExp = convert(DataType::UInt32,
                                                 conditional(unrollKExp == zero, one, unrollKExp));

                    // predicates
                    // unrollK size match predicates
                    auto unrollKPredicate = (aSizeExps[1] % (macKExp * sanUnrollKExp) == zero);
                    setComment(unrollKPredicate,
                               "K must be a multiple of macK * unrollK (unrollK may be "
                               "set by prefetchInFlight)");
                    commandKernel->addPredicate(unrollKPredicate);
                }
            };
        }
    }
}
