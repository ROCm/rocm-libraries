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
            class TensileGEMMSolution : public GEMMSolution<A, B, C, D>
            {
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
                        result.benchmarkResults.correct = this->validate(h_A, h_B, h_C, h_D);
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

                    auto               sizeC_arg   = command->allocateArgument(ulongVal);
                    auto               sizeA_arg   = command->allocateArgument(ulongVal);
                    auto               sizeB_arg   = command->allocateArgument(ulongVal);
                    auto               D_arg       = command->allocateArgument(floatPtr);
                    auto               C_arg       = command->allocateArgument(floatPtr);
                    auto               A_arg       = command->allocateArgument(floatPtr);
                    auto               B_arg       = command->allocateArgument(floatPtr);
                    auto               OffsetD_arg = command->allocateArgument(ulongVal);
                    auto               OffsetC_arg = command->allocateArgument(ulongVal);
                    auto               OffsetA_arg = command->allocateArgument(ulongVal);
                    auto               OffsetB_arg = command->allocateArgument(ulongVal);
                    auto               alpha_arg   = command->allocateArgument(floatVal);
                    CommandArgumentPtr beta_arg;
                    beta_arg                          = command->allocateArgument(floatVal);
                    auto strideD0_arg                 = command->allocateArgument(uintVal);
                    auto strideD1_arg                 = command->allocateArgument(uintVal);
                    auto strideC0_arg                 = command->allocateArgument(uintVal);
                    auto strideC1_arg                 = command->allocateArgument(uintVal);
                    auto strideA0_arg                 = command->allocateArgument(uintVal);
                    auto strideA1_arg                 = command->allocateArgument(uintVal);
                    auto strideB0_arg                 = command->allocateArgument(uintVal);
                    auto strideB1_arg                 = command->allocateArgument(uintVal);
                    auto SizesFree0_arg               = command->allocateArgument(uintVal);
                    auto SizesFree1_arg               = command->allocateArgument(uintVal);
                    auto SizesFree2_arg               = command->allocateArgument(uintVal);
                    auto SizesSum0_arg                = command->allocateArgument(uintVal);
                    auto OrigStaggerUIter_arg         = command->allocateArgument(intVal);
                    auto NumWorkGroups0_arg           = command->allocateArgument(uintVal);
                    auto NumWorkGroups1_arg           = command->allocateArgument(uintVal);
                    auto NumFullBlocks_arg            = command->allocateArgument(uintVal);
                    auto WgmRemainder1_arg            = command->allocateArgument(uintVal);
                    auto MagicNumberWgmRemainder1_arg = command->allocateArgument(uintVal);
                    auto padding_arg                  = command->allocateArgument(uintVal);

                    auto sizeC_exp      = std::make_shared<Expression::Expression>(sizeC_arg);
                    auto sizeA_exp      = std::make_shared<Expression::Expression>(sizeA_arg);
                    auto sizeB_exp      = std::make_shared<Expression::Expression>(sizeB_arg);
                    auto D_exp          = std::make_shared<Expression::Expression>(D_arg);
                    auto C_exp          = std::make_shared<Expression::Expression>(C_arg);
                    auto A_exp          = std::make_shared<Expression::Expression>(A_arg);
                    auto B_exp          = std::make_shared<Expression::Expression>(B_arg);
                    auto alpha_exp      = std::make_shared<Expression::Expression>(alpha_arg);
                    auto strideD0_exp   = std::make_shared<Expression::Expression>(strideD0_arg);
                    auto strideD1_exp   = std::make_shared<Expression::Expression>(strideD1_arg);
                    auto strideC0_exp   = std::make_shared<Expression::Expression>(strideC0_arg);
                    auto strideC1_exp   = std::make_shared<Expression::Expression>(strideC1_arg);
                    auto strideA0_exp   = std::make_shared<Expression::Expression>(strideA0_arg);
                    auto strideA1_exp   = std::make_shared<Expression::Expression>(strideA1_arg);
                    auto strideB0_exp   = std::make_shared<Expression::Expression>(strideB0_arg);
                    auto strideB1_exp   = std::make_shared<Expression::Expression>(strideB1_arg);
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
                    auto OffsetD_exp = std::make_shared<Expression::Expression>(OffsetD_arg);
                    auto OffsetC_exp = std::make_shared<Expression::Expression>(OffsetC_arg);
                    auto OffsetA_exp = std::make_shared<Expression::Expression>(OffsetA_arg);
                    auto OffsetB_exp = std::make_shared<Expression::Expression>(OffsetB_arg);
                    auto padding_exp = std::make_shared<Expression::Expression>(padding_arg);

                    auto k = m_context->kernel();

                    k->setKernelName("Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1");
                    k->setKernelDimensions(3);

                    k->addArgument({"sizeC",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    sizeC_exp});
                    k->addArgument({"sizeA",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    sizeA_exp});
                    k->addArgument({"sizeB",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    sizeB_exp});

                    k->addArgument({"D",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadWrite,
                                    D_exp});
                    k->addArgument({"C",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadOnly,
                                    C_exp});
                    k->addArgument({"A",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadOnly,
                                    A_exp});
                    k->addArgument({"B",
                                    {DataType::Float, PointerType::PointerGlobal},
                                    DataDirection::ReadOnly,
                                    B_exp});

                    k->addArgument({"OffsetD",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    OffsetD_exp});
                    k->addArgument({"OffsetC",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    OffsetC_exp});
                    k->addArgument({"OffsetA",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    OffsetA_exp});
                    k->addArgument({"OffsetB",
                                    {DataType::UInt64, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    OffsetB_exp});

                    k->addArgument({"alpha",
                                    {DataType::Float, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    alpha_exp});

                    auto beta_exp = std::make_shared<Expression::Expression>(beta_arg);
                    k->addArgument({"beta",
                                    {DataType::Float, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    beta_exp});

                    k->addArgument({"strideD0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideD0_exp});
                    k->addArgument({"strideD1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideD1_exp});
                    k->addArgument({"strideC0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideC0_exp});
                    k->addArgument({"strideC1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideC1_exp});
                    k->addArgument({"strideA0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideA0_exp});
                    k->addArgument({"strideA1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideA1_exp});
                    k->addArgument({"strideB0",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideB0_exp});
                    k->addArgument({"strideB1",
                                    {DataType::UInt32, PointerType::Value},
                                    DataDirection::ReadOnly,
                                    strideB1_exp});

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

                KernelArguments makeArgs(std::shared_ptr<A> m_dA,
                                         std::shared_ptr<B> m_dB,
                                         std::shared_ptr<C> m_dC,
                                         std::shared_ptr<D> m_dD)
                {
                    KernelArguments kargs(true);
                    kargs.reserve(1024, 128);

                    kargs.append("sizeC",
                                 (size_t)m_solutionParams.problemParams.m
                                     * m_solutionParams.problemParams.n);
                    kargs.append("sizeA",
                                 (size_t)m_solutionParams.problemParams.m
                                     * m_solutionParams.problemParams.k);
                    kargs.append("sizeB",
                                 (size_t)m_solutionParams.problemParams.k
                                     * m_solutionParams.problemParams.n);

                    kargs.append("D", m_dD.get());
                    kargs.append("C", m_dC.get());
                    kargs.append("A", m_dA.get());
                    kargs.append("B", m_dB.get());

                    kargs.append("OffsetD", (unsigned long long)0);
                    kargs.append("OffsetC", (unsigned long long)0);
                    kargs.append("OffsetA", (unsigned long long)0);
                    kargs.append("OffsetB", (unsigned long long)0);

                    kargs.append("alpha", m_solutionParams.problemParams.alpha);
                    kargs.append("beta", m_solutionParams.problemParams.beta);

                    kargs.append("strideD0", (unsigned int)m_solutionParams.problemParams.m);
                    kargs.append("strideD1",
                                 (unsigned int)m_solutionParams.problemParams.m
                                     * m_solutionParams.problemParams.k);

                    kargs.append("strideC0", (unsigned int)m_solutionParams.problemParams.m);
                    kargs.append("strideC1",
                                 (unsigned int)m_solutionParams.problemParams.m
                                     * m_solutionParams.problemParams.k);

                    kargs.append("strideA0", (unsigned int)m_solutionParams.problemParams.m);
                    kargs.append("strideA1",
                                 (unsigned int)m_solutionParams.problemParams.m
                                     * m_solutionParams.problemParams.k);

                    kargs.append("strideB0", (unsigned int)m_solutionParams.problemParams.n);
                    kargs.append("strideB1",
                                 (unsigned int)m_solutionParams.problemParams.n
                                     * m_solutionParams.problemParams.k);

                    kargs.append("SizesFree0", (unsigned int)m_solutionParams.problemParams.m);
                    kargs.append("SizesFree1", (unsigned int)m_solutionParams.problemParams.n);
                    kargs.append("SizesFree2", 1);
                    kargs.append("SizesSum0", (unsigned int)m_solutionParams.problemParams.k);

                    kargs.append("OrigStaggerUIter", 0);

                    kargs.append("NumWorkGroups0", 60);
                    kargs.append("NumWorkGroups1", 33);
                    kargs.append("NumFullBlocks", 2);

                    kargs.append("WgmRemainder1", 3);
                    kargs.append("MagicNumberWgmRemainder1", 715827883);
                    kargs.append("padding", 0);

                    Log::getLogger()->debug(kargs.toString());

                    return kargs;
                }
            };
        }
    }
}
