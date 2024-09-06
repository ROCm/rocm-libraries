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
            template <typename A, typename B, typename C, typename D>
            class TensileGEMMSolution : public GEMMSolution<A, B, C, D>
            {
                Operations::OperationTag m_tagD, m_tagA, m_tagB, m_tagC;
                Operations::OperationTag m_offsetDTag, m_offsetATag, m_offsetBTag, m_offsetCTag;
                Operations::OperationTag m_alphaTag, m_betaTag;
                Operations::OperationTag m_sizesFree0Tag, m_sizesFree1Tag, m_sizesFree2Tag,
                    m_sizesSum0Tag;
                Operations::OperationTag m_origStaggerUIterTag, m_numWorkGroups0Tag,
                    m_numWorkGroups1Tag, m_numFullBlocksTag, m_wgmRemainder1Tag,
                    m_magicNumberWgmRemainder1Tag, m_paddingTag;

            public:
                TensileGEMMSolution(SolutionParameters const& solutionParams);

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

                        auto [correct, rnorm]           = this->validate(h_A, h_B, h_C, h_D);
                        result.benchmarkResults.correct = correct;
                        result.benchmarkResults.rnorm   = rnorm;
                    }

                    return result;
                }

            private:
                SolutionParameters m_solutionParams;
                ContextPtr         m_context;

                CommandPtr makeCommand()
                {
                    auto command = std::make_shared<Command>();

                    VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
                    VariableType floatVal{DataType::Float, PointerType::Value};
                    VariableType uintVal{DataType::UInt32, PointerType::Value};
                    VariableType ulongVal{DataType::UInt64, PointerType::Value};
                    VariableType intVal{DataType::Int32, PointerType::Value};

                    m_tagD    = command->allocateTag();
                    auto DArg = command->allocateArgument(floatPtr, m_tagD, ArgumentType::Value);
                    auto DExp = std::make_shared<Expression::Expression>(DArg);
                    auto strideDArg = command->allocateArgumentVector(
                        uintVal.dataType, 2, m_tagD, ArgumentType::Stride);
                    auto strideD0Exp = std::make_shared<Expression::Expression>(strideDArg[0]);
                    auto strideD1Exp = std::make_shared<Expression::Expression>(strideDArg[1]);

                    m_tagC    = command->allocateTag();
                    auto CArg = command->allocateArgument(floatPtr, m_tagC, ArgumentType::Value);
                    auto CExp = std::make_shared<Expression::Expression>(CArg);
                    auto limitCArg
                        = command->allocateArgument(ulongVal, m_tagC, ArgumentType::Limit);
                    auto limitCExp  = std::make_shared<Expression::Expression>(limitCArg);
                    auto strideCArg = command->allocateArgumentVector(
                        uintVal.dataType, 2, m_tagC, ArgumentType::Stride);
                    auto strideC0Exp = std::make_shared<Expression::Expression>(strideCArg[0]);
                    auto strideC1Exp = std::make_shared<Expression::Expression>(strideCArg[1]);

                    m_tagA    = command->allocateTag();
                    auto AArg = command->allocateArgument(floatPtr, m_tagA, ArgumentType::Value);
                    auto AExp = std::make_shared<Expression::Expression>(AArg);
                    auto limitAArg
                        = command->allocateArgument(ulongVal, m_tagA, ArgumentType::Limit);
                    auto limitAExp  = std::make_shared<Expression::Expression>(limitAArg);
                    auto strideAArg = command->allocateArgumentVector(
                        uintVal.dataType, 2, m_tagA, ArgumentType::Stride);
                    auto strideA0Exp = std::make_shared<Expression::Expression>(strideAArg[0]);
                    auto strideA1Exp = std::make_shared<Expression::Expression>(strideAArg[1]);

                    m_tagB    = command->allocateTag();
                    auto BArg = command->allocateArgument(floatPtr, m_tagB, ArgumentType::Value);
                    auto BExp = std::make_shared<Expression::Expression>(BArg);
                    auto limitBArg
                        = command->allocateArgument(ulongVal, m_tagB, ArgumentType::Limit);
                    auto limitBExp  = std::make_shared<Expression::Expression>(limitBArg);
                    auto strideBArg = command->allocateArgumentVector(
                        uintVal.dataType, 2, m_tagB, ArgumentType::Stride);
                    auto strideB0Exp = std::make_shared<Expression::Expression>(strideBArg[0]);
                    auto strideB1Exp = std::make_shared<Expression::Expression>(strideBArg[1]);

                    m_offsetDTag = command->allocateTag();
                    auto offsetDArg
                        = command->allocateArgument(ulongVal, m_offsetDTag, ArgumentType::Value);
                    auto offsetDExp = std::make_shared<Expression::Expression>(offsetDArg);
                    m_offsetCTag    = command->allocateTag();
                    auto offsetCArg
                        = command->allocateArgument(ulongVal, m_offsetCTag, ArgumentType::Value);
                    auto offsetCExp = std::make_shared<Expression::Expression>(offsetCArg);
                    m_offsetATag    = command->allocateTag();
                    auto offsetAArg
                        = command->allocateArgument(ulongVal, m_offsetATag, ArgumentType::Value);
                    auto offsetAExp = std::make_shared<Expression::Expression>(offsetAArg);
                    m_offsetBTag    = command->allocateTag();
                    auto offsetBArg
                        = command->allocateArgument(ulongVal, m_offsetBTag, ArgumentType::Value);
                    auto offsetBExp = std::make_shared<Expression::Expression>(offsetBArg);

                    m_alphaTag = command->allocateTag();
                    auto alphaArg
                        = command->allocateArgument(floatVal, m_alphaTag, ArgumentType::Value);
                    auto alphaExp = std::make_shared<Expression::Expression>(alphaArg);
                    m_betaTag     = command->allocateTag();
                    auto betaArg
                        = command->allocateArgument(floatVal, m_betaTag, ArgumentType::Value);
                    auto betaExp = std::make_shared<Expression::Expression>(betaArg);

                    m_sizesFree0Tag = command->allocateTag();
                    auto SizesFree0_arg
                        = command->allocateArgument(uintVal, m_sizesFree0Tag, ArgumentType::Size);
                    m_sizesFree1Tag = command->allocateTag();
                    auto SizesFree1_arg
                        = command->allocateArgument(uintVal, m_sizesFree1Tag, ArgumentType::Size);
                    m_sizesFree2Tag = command->allocateTag();
                    auto SizesFree2_arg
                        = command->allocateArgument(uintVal, m_sizesFree2Tag, ArgumentType::Size);
                    m_sizesSum0Tag = command->allocateTag();
                    auto SizesSum0_arg
                        = command->allocateArgument(uintVal, m_sizesSum0Tag, ArgumentType::Size);

                    m_origStaggerUIterTag     = command->allocateTag();
                    auto OrigStaggerUIter_arg = command->allocateArgument(
                        intVal, m_origStaggerUIterTag, ArgumentType::Value);
                    m_numWorkGroups0Tag     = command->allocateTag();
                    auto NumWorkGroups0_arg = command->allocateArgument(
                        uintVal, m_numWorkGroups0Tag, ArgumentType::Value);
                    m_numWorkGroups1Tag     = command->allocateTag();
                    auto NumWorkGroups1_arg = command->allocateArgument(
                        uintVal, m_numWorkGroups1Tag, ArgumentType::Value);
                    m_numFullBlocksTag     = command->allocateTag();
                    auto NumFullBlocks_arg = command->allocateArgument(
                        uintVal, m_numFullBlocksTag, ArgumentType::Value);
                    m_wgmRemainder1Tag     = command->allocateTag();
                    auto WgmRemainder1_arg = command->allocateArgument(
                        uintVal, m_wgmRemainder1Tag, ArgumentType::Value);
                    m_magicNumberWgmRemainder1Tag     = command->allocateTag();
                    auto MagicNumberWgmRemainder1_arg = command->allocateArgument(
                        uintVal, m_magicNumberWgmRemainder1Tag, ArgumentType::Value);
                    m_paddingTag = command->allocateTag();
                    auto padding_arg
                        = command->allocateArgument(uintVal, m_paddingTag, ArgumentType::Value);

                    auto SizesFree0_exp = std::make_shared<Expression::Expression>(SizesFree0_arg);
                    auto SizesFree1_exp = std::make_shared<Expression::Expression>(SizesFree1_arg);
                    auto SizesFree2_exp = std::make_shared<Expression::Expression>(SizesFree2_arg);
                    auto SizesSum0_exp  = std::make_shared<Expression::Expression>(SizesSum0_arg);
                    auto OrigStaggerUIter_exp
                        = std::make_shared<Expression::Expression>(OrigStaggerUIter_arg);
                    auto NumWorkGroups0_exp
                        = std::make_shared<Expression::Expression>(NumWorkGroups0_arg);
                    auto NumWorkGroups1_exp
                        = std::make_shared<Expression::Expression>(NumWorkGroups1_arg);
                    auto NumFullBlocks_exp
                        = std::make_shared<Expression::Expression>(NumFullBlocks_arg);
                    auto WgmRemainder1_exp
                        = std::make_shared<Expression::Expression>(WgmRemainder1_arg);
                    auto MagicNumberWgmRemainder1_exp
                        = std::make_shared<Expression::Expression>(MagicNumberWgmRemainder1_arg);
                    auto padding_exp = std::make_shared<Expression::Expression>(padding_arg);

                    auto k = m_context->kernel();

                    k->setKernelName("Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1");
                    k->setKernelDimensions(3);

                    k->addArgument({"limitC",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    limitCExp});
                    k->addArgument({"limitA",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    limitAExp});
                    k->addArgument({"limitB",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    limitBExp});

                    k->addArgument({"D",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadWrite,
                                    DExp});
                    k->addArgument({"C",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadOnly,
                                    CExp});
                    k->addArgument({"A",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadOnly,
                                    AExp});
                    k->addArgument({"B",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadOnly,
                                    BExp});

                    k->addArgument({"OffsetD",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    offsetDExp});
                    k->addArgument({"OffsetC",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    offsetCExp});
                    k->addArgument({"OffsetA",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    offsetAExp});
                    k->addArgument({"OffsetB",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    offsetBExp});

                    k->addArgument({"alpha",
                                    {DataType::Float, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    alphaExp});
                    k->addArgument({"beta",
                                    {DataType::Float, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    betaExp});

                    k->addArgument({"strideD0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideD0Exp});
                    k->addArgument({"strideD1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideD1Exp});
                    k->addArgument({"strideC0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideC0Exp});
                    k->addArgument({"strideC1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideC1Exp});
                    k->addArgument({"strideA0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideA0Exp});
                    k->addArgument({"strideA1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideA1Exp});
                    k->addArgument({"strideB0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideB0Exp});
                    k->addArgument({"strideB1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideB1Exp});

                    k->addArgument({"SizesFree0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    SizesFree0_exp});
                    k->addArgument({"SizesFree1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    SizesFree1_exp});
                    k->addArgument({"SizesFree2",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    SizesFree2_exp});
                    k->addArgument({"SizesSum0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    SizesSum0_exp});

                    k->addArgument({"OrigStaggerUIter",
                                    {DataType::Int32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    OrigStaggerUIter_exp});

                    k->addArgument({"NumWorkGroups0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    NumWorkGroups0_exp});
                    k->addArgument({"NumWorkGroups1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    NumWorkGroups1_exp});

                    k->addArgument({"NumFullBlocks",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    NumFullBlocks_exp});
                    k->addArgument({"WgmRemainder1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    WgmRemainder1_exp});
                    k->addArgument({"MagicNumberWgmRemainder1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    MagicNumberWgmRemainder1_exp});
                    k->addArgument({"padding",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    padding_exp});

                    auto workItem0 = std::make_shared<Expression::Expression>(60 * 256u);
                    auto workItem1 = std::make_shared<Expression::Expression>(33);
                    auto zero      = std::make_shared<Expression::Expression>(0u);
                    auto one       = std::make_shared<Expression::Expression>(1u);

                    k->setWorkgroupSize({256, 1, 1});
                    k->setWorkitemCount({workItem0, workItem1, one});

                    return command;
                }

                CommandKernel makeKernel()
                {
                    CommandKernel commandKernel(m_context);
                    commandKernel.loadKernelFromAssembly(
                        "tensile_asm/Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1.s",
                        "Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1");
                    return commandKernel;
                }

                CommandArguments makeArgs(std::shared_ptr<A> m_dA,
                                          std::shared_ptr<B> m_dB,
                                          std::shared_ptr<C> m_dC,
                                          std::shared_ptr<D> m_dD)
                {
                    CommandArguments commandArgs = this->m_command->createArguments();

                    unsigned const M = m_solutionParams.problemParams.m;
                    unsigned const N = m_solutionParams.problemParams.n;
                    unsigned const K = m_solutionParams.problemParams.k;

                    TensorDescriptor descC(
                        getDataTypeFromString(m_solutionParams.problemParams.typeC),
                        M * N,
                        {M, M * K});
                    setCommandTensorArg(commandArgs, m_tagC, descC, m_dC.get());

                    TensorDescriptor descA(
                        getDataTypeFromString(m_solutionParams.problemParams.typeA),
                        M * K,
                        {M, M * K});
                    setCommandTensorArg(commandArgs, m_tagA, descA, (A*)m_dA.get());

                    TensorDescriptor descB(
                        getDataTypeFromString(m_solutionParams.problemParams.typeB),
                        K * N,
                        {N, N * K});
                    setCommandTensorArg(commandArgs, m_tagB, descB, (B*)m_dB.get());

                    commandArgs.setArgument(m_tagD, ArgumentType::Value, m_dD.get());

                    commandArgs.setArgument(
                        m_offsetDTag, ArgumentType::Value, (unsigned long long)0);
                    commandArgs.setArgument(
                        m_offsetCTag, ArgumentType::Value, (unsigned long long)0);
                    commandArgs.setArgument(
                        m_offsetATag, ArgumentType::Value, (unsigned long long)0);
                    commandArgs.setArgument(
                        m_offsetBTag, ArgumentType::Value, (unsigned long long)0);

                    commandArgs.setArgument(
                        m_alphaTag, ArgumentType::Value, m_solutionParams.problemParams.alpha);
                    commandArgs.setArgument(
                        m_betaTag, ArgumentType::Value, m_solutionParams.problemParams.beta);

                    commandArgs.setArgument(m_tagD,
                                            ArgumentType::Stride,
                                            0,
                                            (unsigned int)m_solutionParams.problemParams.m);
                    commandArgs.setArgument(m_tagD,
                                            ArgumentType::Stride,
                                            1,
                                            (unsigned int)m_solutionParams.problemParams.m
                                                * m_solutionParams.problemParams.k);

                    commandArgs.setArgument(m_sizesFree0Tag,
                                            ArgumentType::Size,
                                            (unsigned int)m_solutionParams.problemParams.m);
                    commandArgs.setArgument(m_sizesFree1Tag,
                                            ArgumentType::Size,
                                            (unsigned int)m_solutionParams.problemParams.n);
                    commandArgs.setArgument(m_sizesFree2Tag, ArgumentType::Size, 1);
                    commandArgs.setArgument(m_sizesSum0Tag,
                                            ArgumentType::Size,
                                            (unsigned int)m_solutionParams.problemParams.k);

                    commandArgs.setArgument(m_origStaggerUIterTag, ArgumentType::Value, 0);

                    commandArgs.setArgument(m_numWorkGroups0Tag, ArgumentType::Value, 60);
                    commandArgs.setArgument(m_numWorkGroups1Tag, ArgumentType::Value, 33);
                    commandArgs.setArgument(m_numFullBlocksTag, ArgumentType::Value, 2);

                    commandArgs.setArgument(m_wgmRemainder1Tag, ArgumentType::Value, 3);
                    commandArgs.setArgument(
                        m_magicNumberWgmRemainder1Tag, ArgumentType::Value, 715827883);
                    commandArgs.setArgument(m_paddingTag, ArgumentType::Value, 0);

                    return commandArgs;
                }
            };
        }
    }
}
