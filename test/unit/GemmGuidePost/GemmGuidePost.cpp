#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <iterator>

#include <hip/amd_detail/amd_hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <rocRoller/Operations/Command.hpp>

#include "../GPUContextFixture.hpp"
#include "../GenericContextFixture.hpp"
#include "../Utilities.hpp"

#include "GemmGuidePostKernels.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    using KernelProgram = Generator<Instruction>(rocRoller::ContextPtr);

    template <typename T, typename S>
    void printMatrix(std::vector<T>& matrix, S stride0, S stride1)
    {
        for(size_t i = 0; i < matrix.size() / stride0; i++)
        {
            for(size_t j = 0; j < stride0; j++)
            {
                std::cout << matrix[i * stride0 + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    class GPU_GemmGuidePostTest : public CurrentGPUContextFixture
    {
    };

    void doMinimalMM(std::shared_ptr<rocRoller::Context> m_context,
                     KernelProgram                       Program,
                     std::vector<float>&                 hostD,
                     const std::vector<float>&           hostC,
                     const std::vector<float>&           hostA,
                     const std::vector<float>&           hostB,
                     float                               alpha,
                     float                               beta,
                     unsigned int                        strideD0,
                     unsigned int                        strideD1,
                     unsigned int                        strideC0,
                     unsigned int                        strideC1,
                     unsigned int                        strideA0,
                     unsigned int                        strideA1,
                     unsigned int                        strideB0,
                     unsigned int                        strideB1,
                     unsigned int                        SizesFree0,
                     unsigned int                        SizesFree1,
                     unsigned int                        SizesFree2,
                     unsigned int                        SizesSum0,
                     int                                 OrigStaggerUIter,
                     unsigned int                        NumWorkGroups0,
                     unsigned int                        NumWorkGroups1,
                     unsigned int                        NumFullBlocks,
                     unsigned int                        WgmRemainder1,
                     unsigned int                        MagicNumberWgmRemainder1,
                     unsigned int                        OffsetD,
                     unsigned int                        OffsetC,
                     unsigned int                        OffsetA,
                     unsigned int                        OffsetB,
                     unsigned int                        padding,
                     bool                                hasBeta = true)
    {
        KernelOptions options;
        //options.logLevel               = LogLevel::Debug;
        options.preloadKernelArguments = true;
        m_context->setKernelOptions(options);
        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};
        VariableType ulongVal{DataType::UInt64, PointerType::Value};
        VariableType intVal{DataType::Int32, PointerType::Value};

        auto                             sizeC_arg = command->allocateArgument(ulongVal);
        auto                             sizeA_arg = command->allocateArgument(ulongVal);
        auto                             sizeB_arg = command->allocateArgument(ulongVal);
        auto                             D_arg     = command->allocateArgument(floatPtr);
        auto                             C_arg     = command->allocateArgument(floatPtr);
        auto                             A_arg     = command->allocateArgument(floatPtr);
        auto                             B_arg     = command->allocateArgument(floatPtr);
        auto                             alpha_arg = command->allocateArgument(floatVal);
        std::shared_ptr<CommandArgument> beta_arg;
        if(hasBeta)
            beta_arg = command->allocateArgument(floatVal);
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
        auto OffsetD_arg                  = command->allocateArgument(uintVal);
        auto OffsetC_arg                  = command->allocateArgument(uintVal);
        auto OffsetA_arg                  = command->allocateArgument(uintVal);
        auto OffsetB_arg                  = command->allocateArgument(uintVal);
        auto padding_arg                  = command->allocateArgument(uintVal);

        auto sizeC_exp            = std::make_shared<Expression::Expression>(sizeC_arg);
        auto sizeA_exp            = std::make_shared<Expression::Expression>(sizeA_arg);
        auto sizeB_exp            = std::make_shared<Expression::Expression>(sizeB_arg);
        auto D_exp                = std::make_shared<Expression::Expression>(D_arg);
        auto C_exp                = std::make_shared<Expression::Expression>(C_arg);
        auto A_exp                = std::make_shared<Expression::Expression>(A_arg);
        auto B_exp                = std::make_shared<Expression::Expression>(B_arg);
        auto alpha_exp            = std::make_shared<Expression::Expression>(alpha_arg);
        auto strideD0_exp         = std::make_shared<Expression::Expression>(strideD0_arg);
        auto strideD1_exp         = std::make_shared<Expression::Expression>(strideD1_arg);
        auto strideC0_exp         = std::make_shared<Expression::Expression>(strideC0_arg);
        auto strideC1_exp         = std::make_shared<Expression::Expression>(strideC1_arg);
        auto strideA0_exp         = std::make_shared<Expression::Expression>(strideA0_arg);
        auto strideA1_exp         = std::make_shared<Expression::Expression>(strideA1_arg);
        auto strideB0_exp         = std::make_shared<Expression::Expression>(strideB0_arg);
        auto strideB1_exp         = std::make_shared<Expression::Expression>(strideB1_arg);
        auto SizesFree0_exp       = std::make_shared<Expression::Expression>(SizesFree0_arg);
        auto SizesFree1_exp       = std::make_shared<Expression::Expression>(SizesFree1_arg);
        auto SizesFree2_exp       = std::make_shared<Expression::Expression>(SizesFree2_arg);
        auto SizesSum0_exp        = std::make_shared<Expression::Expression>(SizesSum0_arg);
        auto OrigStaggerUIter_exp = std::make_shared<Expression::Expression>(OrigStaggerUIter_arg);
        auto NumWorkGroups0_exp   = std::make_shared<Expression::Expression>(NumWorkGroups0_arg);
        auto NumWorkGroups1_exp   = std::make_shared<Expression::Expression>(NumWorkGroups1_arg);
        auto NumFullBlocks_exp    = std::make_shared<Expression::Expression>(NumFullBlocks_arg);
        auto WgmRemainder1_exp    = std::make_shared<Expression::Expression>(WgmRemainder1_arg);
        auto MagicNumberWgmRemainder1_exp
            = std::make_shared<Expression::Expression>(MagicNumberWgmRemainder1_arg);
        auto OffsetD_exp = std::make_shared<Expression::Expression>(OffsetD_arg);
        auto OffsetC_exp = std::make_shared<Expression::Expression>(OffsetC_arg);
        auto OffsetA_exp = std::make_shared<Expression::Expression>(OffsetA_arg);
        auto OffsetB_exp = std::make_shared<Expression::Expression>(OffsetB_arg);
        auto padding_exp = std::make_shared<Expression::Expression>(padding_arg);

        auto k = m_context->kernel();

        k->setKernelName("Cijk_Ailk_Bjlk_SB_MT128x64x16_MI32x32x2x1_SN_K1");
        k->setKernelDimensions(3);

        k->addArgument(
            {"sizeC", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeC_exp});
        k->addArgument(
            {"sizeA", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeA_exp});
        k->addArgument(
            {"sizeB", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeB_exp});

        k->addArgument(
            {"D", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadWrite, D_exp});
        k->addArgument(
            {"C", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, C_exp});
        k->addArgument(
            {"A", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, A_exp});
        k->addArgument(
            {"B", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, B_exp});

        k->addArgument(
            {"alpha", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, alpha_exp});
        if(hasBeta)
        {
            auto beta_exp = std::make_shared<Expression::Expression>(beta_arg);
            k->addArgument(
                {"beta", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, beta_exp});
        }

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

        k->addArgument({"OffsetD",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetD_exp});
        k->addArgument({"OffsetC",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetC_exp});
        k->addArgument({"OffsetA",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetA_exp});
        k->addArgument({"OffsetB",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetB_exp});
        k->addArgument({"padding",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        padding_exp});

        auto workItem0 = std::make_shared<Expression::Expression>(NumWorkGroups0 * 256u);
        auto workItem1 = std::make_shared<Expression::Expression>(NumWorkGroups1);
        auto zero      = std::make_shared<Expression::Expression>(0u);
        auto one       = std::make_shared<Expression::Expression>(1u);

        k->setWorkgroupSize({256, 1, 1});
        k->setWorkitemCount({workItem0, workItem1, one});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        m_context->schedule(Program(m_context));

        auto placeholderLDS = Register::Value::AllocateLDS(m_context, DataType::Raw32, 1024);

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        unsigned long long sizeC = hostC.size(), sizeA = hostA.size(), sizeB = hostB.size();

        KernelArguments kargs(true);
        kargs.reserve(1024, 128);

        kargs.append("sizeC", sizeC);
        kargs.append("sizeA", sizeA);
        kargs.append("sizeB", sizeB);

        auto D = make_shared_device<float>(sizeC);
        auto C = make_shared_device(hostC);
        auto A = make_shared_device(hostA);
        auto B = make_shared_device(hostB);

        kargs.append("D", D.get());
        kargs.append("C", D.get());
        kargs.append("A", A.get());
        kargs.append("B", B.get());

        kargs.append("alpha", alpha);
        if(hasBeta)
            kargs.append("beta", beta);

        kargs.append("strideD0", strideD0);
        kargs.append("strideD1", strideD1);

        kargs.append("strideC0", strideC0);
        kargs.append("strideC1", strideC1);

        kargs.append("strideA0", strideA0);
        kargs.append("strideA1", strideA1);

        kargs.append("strideB0", strideB0);
        kargs.append("strideB1", strideB1);

        kargs.append("SizesFree0", SizesFree0);
        kargs.append("SizesFree1", SizesFree1);
        kargs.append("SizesFree2", SizesFree2);
        kargs.append("SizesSum0", SizesSum0);

        kargs.append("OrigStaggerUIter", OrigStaggerUIter);

        kargs.append("NumWorkGroups0", NumWorkGroups0);
        kargs.append("NumWorkGroups1", NumWorkGroups1);
        kargs.append("NumFullBlocks", NumFullBlocks);

        kargs.append("WgmRemainder1", WgmRemainder1);
        kargs.append("MagicNumberWgmRemainder1", MagicNumberWgmRemainder1);

        kargs.append("OffsetD", OffsetD);
        kargs.append("OffsetC", OffsetC);
        kargs.append("OffsetA", OffsetA);
        kargs.append("OffsetB", OffsetB);
        kargs.append("padding", padding);

        double total_time = 0;
        int    iters      = 10;
        TIMER(t_gpuMM, "GPUMM");
        TIC(t_gpuMM);
        for(int i = 0; i <= iters; i++)
        {
            hipEvent_t begin, end;
            ASSERT_THAT(hipEventCreate(&begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventCreate(&end), HasHipSuccess(0));
            ASSERT_THAT(hipEventRecord(begin, 0), HasHipSuccess(0));
            commandKernel.launchKernel(kargs.runtimeArguments());
            ASSERT_THAT(hipEventRecord(end, 0), HasHipSuccess(0));
            ASSERT_THAT(hipMemcpy(hostD.data(), D.get(), sizeC * sizeof(float), hipMemcpyDefault),
                        HasHipSuccess(0));
            float elapsed = 0.f;
            ASSERT_THAT(hipEventElapsedTime(&elapsed, begin, end), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(end), HasHipSuccess(0));
            if(i > 0)
            {
                total_time += elapsed;
            }
        }
        TOC(t_gpuMM);
        std::cout << "Average Time: " << total_time / iters << " milliseconds" << std::endl;

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    TEST_F(GPU_GemmGuidePostTest, ManualKernelSmall_Minimal)
    {
        TimerPool::clear();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        ASSERT_EQ(true, isLocalDevice());

        auto seed = 923168u;

        unsigned long long sizeC = 4096, sizeA = 4096, sizeB = 4096;
        std::vector<float> hostD(sizeC);
        std::vector<float> hostC;
        std::vector<float> hostA;
        std::vector<float> hostB;
        GenerateRandomInput(seed, hostC, sizeC, hostA, sizeA, hostB, sizeB);
        float        alpha = 1, beta = 0;
        unsigned int strideD0 = 64, strideD1 = 4096;
        unsigned int strideC0 = 64, strideC1 = 4096;
        unsigned int strideA0 = 64, strideA1 = 4096;
        unsigned int strideB0 = 64, strideB1 = 4096;
        unsigned int SizesFree0 = 64, SizesFree1 = 64, SizesFree2 = 1;
        unsigned int SizesSum0        = 64;
        int          OrigStaggerUIter = 0;
        unsigned int NumWorkGroups0 = 1, NumWorkGroups1 = 1, NumFullBlocks = 1;
        unsigned int WgmRemainder1 = 0, MagicNumberWgmRemainder1 = 0;
        unsigned int OffsetD = 0, OffsetC = 0, OffsetA = 0, OffsetB = 0, padding = 0;

        doMinimalMM(m_context,
                    SGEMM_Minimal_Program,
                    hostD,
                    hostC,
                    hostA,
                    hostB,
                    alpha,
                    beta,
                    strideD0,
                    strideD1,
                    strideC0,
                    strideC1,
                    strideA0,
                    strideA1,
                    strideB0,
                    strideB1,
                    SizesFree0,
                    SizesFree1,
                    SizesFree2,
                    SizesSum0,
                    OrigStaggerUIter,
                    NumWorkGroups0,
                    NumWorkGroups1,
                    NumFullBlocks,
                    WgmRemainder1,
                    MagicNumberWgmRemainder1,
                    OffsetD,
                    OffsetC,
                    OffsetA,
                    OffsetB,
                    padding,
                    false);

        std::vector<float> cpuD(sizeC);

        CPUMM(cpuD, hostC, hostA, hostB, strideA0, strideB0, strideB0, alpha, beta);

        double rnorm = relativeNorm(hostD, cpuD);
        EXPECT_LT(rnorm, 1.e-6);
    }

    TEST_F(GPU_GemmGuidePostTest, ManualKernel_Minimal)
    {
        TimerPool::clear();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        ASSERT_EQ(true, isLocalDevice());

        auto seed = 219678u;

        unsigned long long sizeC = 12582912, sizeA = 12582912, sizeB = 16777216;
        std::vector<float> hostD(sizeC);
        std::vector<float> hostC;
        std::vector<float> hostA;
        std::vector<float> hostB;
        GenerateRandomInput(seed, hostC, sizeC, hostA, sizeA, hostB, sizeB);
        float        alpha = 1, beta = 0;
        unsigned int strideD0 = 3072, strideD1 = 12582912;
        unsigned int strideC0 = 3072, strideC1 = 12582912;
        unsigned int strideA0 = 3072, strideA1 = 12582912;
        unsigned int strideB0 = 4096, strideB1 = 16777216;
        unsigned int SizesFree0 = 3072, SizesFree1 = 4096, SizesFree2 = 1;
        unsigned int SizesSum0        = 4096;
        int          OrigStaggerUIter = 0;
        unsigned int NumWorkGroups0 = 48, NumWorkGroups1 = 64, NumFullBlocks = 64;
        unsigned int WgmRemainder1 = 0, MagicNumberWgmRemainder1 = 0;
        unsigned int OffsetD = 0, OffsetC = 0, OffsetA = 0, OffsetB = 0, padding = 0;

        doMinimalMM(m_context,
                    SGEMM_Minimal_Program,
                    hostD,
                    hostC,
                    hostA,
                    hostB,
                    alpha,
                    beta,
                    strideD0,
                    strideD1,
                    strideC0,
                    strideC1,
                    strideA0,
                    strideA1,
                    strideB0,
                    strideB1,
                    SizesFree0,
                    SizesFree1,
                    SizesFree2,
                    SizesSum0,
                    OrigStaggerUIter,
                    NumWorkGroups0,
                    NumWorkGroups1,
                    NumFullBlocks,
                    WgmRemainder1,
                    MagicNumberWgmRemainder1,
                    OffsetD,
                    OffsetC,
                    OffsetA,
                    OffsetB,
                    padding,
                    false);

        std::vector<float> cpuD(sizeC);

        CPUMM(cpuD, hostC, hostA, hostB, strideA0, strideB0, strideB0, alpha, beta);

        double rnorm = relativeNorm(hostD, cpuD);
        EXPECT_LT(rnorm, 2.e-6);
    }

    void doMM(std::shared_ptr<rocRoller::Context> m_context,
              KernelProgram                       Program,
              std::vector<float>&                 hostD,
              const std::vector<float>&           hostC,
              const std::vector<float>&           hostA,
              const std::vector<float>&           hostB,
              float                               alpha,
              float                               beta,
              unsigned int                        strideD0,
              unsigned int                        strideD1,
              unsigned int                        strideC0,
              unsigned int                        strideC1,
              unsigned int                        strideA0,
              unsigned int                        strideA1,
              unsigned int                        strideB0,
              unsigned int                        strideB1,
              unsigned int                        SizesFree0,
              unsigned int                        SizesFree1,
              unsigned int                        SizesFree2,
              unsigned int                        SizesSum0,
              int                                 OrigStaggerUIter,
              unsigned int                        NumWorkGroups0,
              unsigned int                        NumWorkGroups1,
              unsigned int                        NumFullBlocks,
              unsigned int                        WgmRemainder1,
              unsigned int                        MagicNumberWgmRemainder1,
              unsigned int                        OffsetD,
              unsigned int                        OffsetC,
              unsigned int                        OffsetA,
              unsigned int                        OffsetB,
              unsigned int                        padding,
              bool                                hasBeta = true)
    {
        KernelOptions options;
        options.logLevel               = LogLevel::Debug;
        options.preloadKernelArguments = false;
        m_context->setKernelOptions(options);
        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};
        VariableType ulongVal{DataType::UInt64, PointerType::Value};
        VariableType intVal{DataType::Int32, PointerType::Value};

        auto                             sizeC_arg = command->allocateArgument(ulongVal);
        auto                             sizeA_arg = command->allocateArgument(ulongVal);
        auto                             sizeB_arg = command->allocateArgument(ulongVal);
        auto                             D_arg     = command->allocateArgument(floatPtr);
        auto                             C_arg     = command->allocateArgument(floatPtr);
        auto                             A_arg     = command->allocateArgument(floatPtr);
        auto                             B_arg     = command->allocateArgument(floatPtr);
        auto                             alpha_arg = command->allocateArgument(floatVal);
        std::shared_ptr<CommandArgument> beta_arg;
        if(hasBeta)
            beta_arg = command->allocateArgument(floatVal);
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
        auto OffsetD_arg                  = command->allocateArgument(uintVal);
        auto OffsetC_arg                  = command->allocateArgument(uintVal);
        auto OffsetA_arg                  = command->allocateArgument(uintVal);
        auto OffsetB_arg                  = command->allocateArgument(uintVal);
        auto padding_arg                  = command->allocateArgument(uintVal);

        auto sizeC_exp            = std::make_shared<Expression::Expression>(sizeC_arg);
        auto sizeA_exp            = std::make_shared<Expression::Expression>(sizeA_arg);
        auto sizeB_exp            = std::make_shared<Expression::Expression>(sizeB_arg);
        auto D_exp                = std::make_shared<Expression::Expression>(D_arg);
        auto C_exp                = std::make_shared<Expression::Expression>(C_arg);
        auto A_exp                = std::make_shared<Expression::Expression>(A_arg);
        auto B_exp                = std::make_shared<Expression::Expression>(B_arg);
        auto alpha_exp            = std::make_shared<Expression::Expression>(alpha_arg);
        auto strideD0_exp         = std::make_shared<Expression::Expression>(strideD0_arg);
        auto strideD1_exp         = std::make_shared<Expression::Expression>(strideD1_arg);
        auto strideC0_exp         = std::make_shared<Expression::Expression>(strideC0_arg);
        auto strideC1_exp         = std::make_shared<Expression::Expression>(strideC1_arg);
        auto strideA0_exp         = std::make_shared<Expression::Expression>(strideA0_arg);
        auto strideA1_exp         = std::make_shared<Expression::Expression>(strideA1_arg);
        auto strideB0_exp         = std::make_shared<Expression::Expression>(strideB0_arg);
        auto strideB1_exp         = std::make_shared<Expression::Expression>(strideB1_arg);
        auto SizesFree0_exp       = std::make_shared<Expression::Expression>(SizesFree0_arg);
        auto SizesFree1_exp       = std::make_shared<Expression::Expression>(SizesFree1_arg);
        auto SizesFree2_exp       = std::make_shared<Expression::Expression>(SizesFree2_arg);
        auto SizesSum0_exp        = std::make_shared<Expression::Expression>(SizesSum0_arg);
        auto OrigStaggerUIter_exp = std::make_shared<Expression::Expression>(OrigStaggerUIter_arg);
        auto NumWorkGroups0_exp   = std::make_shared<Expression::Expression>(NumWorkGroups0_arg);
        auto NumWorkGroups1_exp   = std::make_shared<Expression::Expression>(NumWorkGroups1_arg);
        auto NumFullBlocks_exp    = std::make_shared<Expression::Expression>(NumFullBlocks_arg);
        auto WgmRemainder1_exp    = std::make_shared<Expression::Expression>(WgmRemainder1_arg);
        auto MagicNumberWgmRemainder1_exp
            = std::make_shared<Expression::Expression>(MagicNumberWgmRemainder1_arg);
        auto OffsetD_exp = std::make_shared<Expression::Expression>(OffsetD_arg);
        auto OffsetC_exp = std::make_shared<Expression::Expression>(OffsetC_arg);
        auto OffsetA_exp = std::make_shared<Expression::Expression>(OffsetA_arg);
        auto OffsetB_exp = std::make_shared<Expression::Expression>(OffsetB_arg);
        auto padding_exp = std::make_shared<Expression::Expression>(padding_arg);

        auto k = m_context->kernel();

        k->setKernelName("Cijk_Ailk_Bjlk_SB_MT128x64x16_MI32x32x2x1_SN_K1");
        k->setKernelDimensions(3);

        k->addArgument(
            {"sizeC", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeC_exp});
        k->addArgument(
            {"sizeA", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeA_exp});
        k->addArgument(
            {"sizeB", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeB_exp});

        k->addArgument(
            {"D", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadWrite, D_exp});
        k->addArgument(
            {"C", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, C_exp});
        k->addArgument(
            {"A", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, A_exp});
        k->addArgument(
            {"B", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, B_exp});

        k->addArgument(
            {"alpha", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, alpha_exp});
        if(hasBeta)
        {
            auto beta_exp = std::make_shared<Expression::Expression>(beta_arg);
            k->addArgument(
                {"beta", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, beta_exp});
        }

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

        k->addArgument({"OffsetD",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetD_exp});
        k->addArgument({"OffsetC",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetC_exp});
        k->addArgument({"OffsetA",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetA_exp});
        k->addArgument({"OffsetB",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetB_exp});
        k->addArgument({"padding",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        padding_exp});

        auto workItem0 = std::make_shared<Expression::Expression>(NumWorkGroups0 * 256u);
        auto workItem1 = std::make_shared<Expression::Expression>(NumWorkGroups1);
        auto zero      = std::make_shared<Expression::Expression>(0u);
        auto one       = std::make_shared<Expression::Expression>(1u);

        k->setWorkgroupSize({256, 1, 1});
        k->setWorkitemCount({workItem0, workItem1, one});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto placeholderV = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Raw32, 220);
        auto placeholderA = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Raw32, 32);
        auto placeholderS = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Raw32, 59);
        auto placeholderLDS = Register::Value::AllocateLDS(m_context, DataType::Raw32, 7168);
        auto kb             = [&]() -> Generator<Instruction> {
            co_yield placeholderV->allocate();
            co_yield placeholderA->allocate();
            co_yield placeholderS->allocate();
        };
        m_context->schedule(kb());

        m_context->schedule(Program(m_context));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        unsigned long long sizeC = hostC.size(), sizeA = hostA.size(), sizeB = hostB.size();

        KernelArguments kargs(true);
        kargs.reserve(1024, 128);

        kargs.append("sizeC", sizeC);
        kargs.append("sizeA", sizeA);
        kargs.append("sizeB", sizeB);

        auto D = make_shared_device<float>(sizeC);
        auto C = make_shared_device(hostC);
        auto A = make_shared_device(hostA);
        auto B = make_shared_device(hostB);

        kargs.append("D", D.get());
        kargs.append("C", D.get());
        kargs.append("A", A.get());
        kargs.append("B", B.get());

        kargs.append("alpha", alpha);
        if(hasBeta)
            kargs.append("beta", beta);

        kargs.append("strideD0", strideD0);
        kargs.append("strideD1", strideD1);

        kargs.append("strideC0", strideC0);
        kargs.append("strideC1", strideC1);

        kargs.append("strideA0", strideA0);
        kargs.append("strideA1", strideA1);

        kargs.append("strideB0", strideB0);
        kargs.append("strideB1", strideB1);

        kargs.append("SizesFree0", SizesFree0);
        kargs.append("SizesFree1", SizesFree1);
        kargs.append("SizesFree2", SizesFree2);
        kargs.append("SizesSum0", SizesSum0);

        kargs.append("OrigStaggerUIter", OrigStaggerUIter);

        kargs.append("NumWorkGroups0", NumWorkGroups0);
        kargs.append("NumWorkGroups1", NumWorkGroups1);
        kargs.append("NumFullBlocks", NumFullBlocks);

        kargs.append("WgmRemainder1", WgmRemainder1);
        kargs.append("MagicNumberWgmRemainder1", MagicNumberWgmRemainder1);

        kargs.append("OffsetD", OffsetD);
        kargs.append("OffsetC", OffsetC);
        kargs.append("OffsetA", OffsetA);
        kargs.append("OffsetB", OffsetB);
        kargs.append("padding", padding);

        double total_time = 0;
        int    iters      = 10;
        TIMER(t_gpuMM, "GPUMM");
        TIC(t_gpuMM);
        for(int i = 0; i <= iters; i++)
        {
            hipEvent_t begin, end;
            ASSERT_THAT(hipEventCreate(&begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventCreate(&end), HasHipSuccess(0));
            ASSERT_THAT(hipEventRecord(begin, 0), HasHipSuccess(0));
            commandKernel.launchKernel(kargs.runtimeArguments());
            ASSERT_THAT(hipEventRecord(end, 0), HasHipSuccess(0));
            ASSERT_THAT(hipMemcpy(hostD.data(), D.get(), sizeC * sizeof(float), hipMemcpyDefault),
                        HasHipSuccess(0));
            float elapsed = 0.f;
            ASSERT_THAT(hipEventElapsedTime(&elapsed, begin, end), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(end), HasHipSuccess(0));
            if(i > 0)
            {
                total_time += elapsed;
            }
        }
        TOC(t_gpuMM);
        std::cout << "Average Time: " << total_time / iters << " milliseconds" << std::endl;

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    TEST_F(GPU_GemmGuidePostTest, ManualKernelSmall_Optimized)
    {
        TimerPool::clear();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        ASSERT_EQ(true, isLocalDevice());

        auto seed = 12679u;

        unsigned long long sizeC = 4096, sizeA = 4096, sizeB = 4096;
        std::vector<float> hostD(sizeC);
        std::vector<float> hostC;
        std::vector<float> hostA;
        std::vector<float> hostB;
        GenerateRandomInput(seed, hostC, sizeC, hostA, sizeA, hostB, sizeB);
        float        alpha = 1, beta = 0;
        unsigned int strideD0 = 64, strideD1 = 4096;
        unsigned int strideC0 = 64, strideC1 = 4096;
        unsigned int strideA0 = 64, strideA1 = 4096;
        unsigned int strideB0 = 64, strideB1 = 4096;
        unsigned int SizesFree0 = 64, SizesFree1 = 64, SizesFree2 = 1;
        unsigned int SizesSum0        = 64;
        int          OrigStaggerUIter = 0;
        unsigned int NumWorkGroups0 = 1, NumWorkGroups1 = 1, NumFullBlocks = 0;
        unsigned int WgmRemainder1 = 0, MagicNumberWgmRemainder1 = 0;
        unsigned int OffsetD = 0, OffsetC = 0, OffsetA = 0, OffsetB = 0, padding = 0;

        doMM(m_context,
             SGEMM_Optimized_Program,
             hostD,
             hostC,
             hostA,
             hostB,
             alpha,
             beta,
             strideD0,
             strideD1,
             strideC0,
             strideC1,
             strideA0,
             strideA1,
             strideB0,
             strideB1,
             SizesFree0,
             SizesFree1,
             SizesFree2,
             SizesSum0,
             OrigStaggerUIter,
             NumWorkGroups0,
             NumWorkGroups1,
             NumFullBlocks,
             WgmRemainder1,
             MagicNumberWgmRemainder1,
             OffsetD,
             OffsetC,
             OffsetA,
             OffsetB,
             padding);

        std::vector<float> cpuD(sizeC);

        CPUMM(cpuD, hostC, hostA, hostB, strideA0, strideB0, strideB0, alpha, beta);

        double rnorm = relativeNorm(hostD, cpuD);
        EXPECT_LT(rnorm, 1.e-6);
    }

    TEST_F(GPU_GemmGuidePostTest, ManualKernel_Optimized)
    {
        TimerPool::clear();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        ASSERT_EQ(true, isLocalDevice());

        auto seed = 12679u;

        unsigned long long sizeC = 12582912, sizeA = 12582912, sizeB = 16777216;
        std::vector<float> hostD(sizeC);
        std::vector<float> hostC;
        std::vector<float> hostA;
        std::vector<float> hostB;
        GenerateRandomInput(seed, hostC, sizeC, hostA, sizeA, hostB, sizeB);
        float        alpha = 1, beta = 0;
        unsigned int strideD0 = 3072, strideD1 = 12582912;
        unsigned int strideC0 = 3072, strideC1 = 12582912;
        unsigned int strideA0 = 3072, strideA1 = 12582912;
        unsigned int strideB0 = 4096, strideB1 = 16777216;
        unsigned int SizesFree0 = 3072, SizesFree1 = 4096, SizesFree2 = 1;
        unsigned int SizesSum0        = 4096;
        int          OrigStaggerUIter = 0;
        unsigned int NumWorkGroups0 = 24, NumWorkGroups1 = 64, NumFullBlocks = 64;
        unsigned int WgmRemainder1 = 8, MagicNumberWgmRemainder1 = 268435457;
        unsigned int OffsetD = 0, OffsetC = 0, OffsetA = 0, OffsetB = 0, padding = 0;

        doMM(m_context,
             SGEMM_Optimized_Program,
             hostD,
             hostC,
             hostA,
             hostB,
             alpha,
             beta,
             strideD0,
             strideD1,
             strideC0,
             strideC1,
             strideA0,
             strideA1,
             strideB0,
             strideB1,
             SizesFree0,
             SizesFree1,
             SizesFree2,
             SizesSum0,
             OrigStaggerUIter,
             NumWorkGroups0,
             NumWorkGroups1,
             NumFullBlocks,
             WgmRemainder1,
             MagicNumberWgmRemainder1,
             OffsetD,
             OffsetC,
             OffsetA,
             OffsetB,
             padding);

        std::vector<float> cpuD(sizeC);

        CPUMM(cpuD, hostC, hostA, hostB, strideA0, strideB0, strideB0, alpha, beta);

        double rnorm = relativeNorm(hostD, cpuD);
        EXPECT_LT(rnorm, 2.e-6);
    }

    void doMinimalMM_HGEMM(std::shared_ptr<rocRoller::Context> m_context,
                           KernelProgram                       Program,
                           std::vector<__half>&                hostD,
                           const std::vector<__half>&          hostC,
                           const std::vector<__half>&          hostA,
                           const std::vector<__half>&          hostB,
                           float                               alpha,
                           float                               beta,
                           unsigned int                        strideD0,
                           unsigned int                        strideD1,
                           unsigned int                        strideC0,
                           unsigned int                        strideC1,
                           unsigned int                        strideA0,
                           unsigned int                        strideA1,
                           unsigned int                        strideB0,
                           unsigned int                        strideB1,
                           unsigned int                        SizesFree0,
                           unsigned int                        SizesFree1,
                           unsigned int                        SizesFree2,
                           unsigned int                        SizesSum0,
                           int                                 OrigStaggerUIter,
                           unsigned int                        NumWorkGroups0,
                           unsigned int                        NumWorkGroups1,
                           unsigned int                        NumFullBlocks,
                           unsigned int                        WgmRemainder1,
                           unsigned int                        MagicNumberWgmRemainder1,
                           unsigned int                        OffsetD,
                           unsigned int                        OffsetC,
                           unsigned int                        OffsetA,
                           unsigned int                        OffsetB,
                           unsigned int                        padding,
                           unsigned int                        numVGPR,
                           unsigned int                        numACCGPR,
                           unsigned int                        numSGPR,
                           unsigned int                        sizeLDS,
                           bool                                hasBeta = true)
    {
        KernelOptions options;
        options.logLevel               = LogLevel::Debug;
        options.preloadKernelArguments = false;
        m_context->setKernelOptions(options);
        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};
        VariableType ulongVal{DataType::UInt64, PointerType::Value};
        VariableType intVal{DataType::Int32, PointerType::Value};

        auto                             sizeC_arg = command->allocateArgument(ulongVal);
        auto                             sizeA_arg = command->allocateArgument(ulongVal);
        auto                             sizeB_arg = command->allocateArgument(ulongVal);
        auto                             D_arg     = command->allocateArgument(floatPtr);
        auto                             C_arg     = command->allocateArgument(floatPtr);
        auto                             A_arg     = command->allocateArgument(floatPtr);
        auto                             B_arg     = command->allocateArgument(floatPtr);
        auto                             alpha_arg = command->allocateArgument(floatVal);
        std::shared_ptr<CommandArgument> beta_arg;
        if(hasBeta)
            beta_arg = command->allocateArgument(floatVal);
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
        auto OffsetD_arg                  = command->allocateArgument(uintVal);
        auto OffsetC_arg                  = command->allocateArgument(uintVal);
        auto OffsetA_arg                  = command->allocateArgument(uintVal);
        auto OffsetB_arg                  = command->allocateArgument(uintVal);
        auto padding_arg                  = command->allocateArgument(uintVal);

        auto sizeC_exp            = std::make_shared<Expression::Expression>(sizeC_arg);
        auto sizeA_exp            = std::make_shared<Expression::Expression>(sizeA_arg);
        auto sizeB_exp            = std::make_shared<Expression::Expression>(sizeB_arg);
        auto D_exp                = std::make_shared<Expression::Expression>(D_arg);
        auto C_exp                = std::make_shared<Expression::Expression>(C_arg);
        auto A_exp                = std::make_shared<Expression::Expression>(A_arg);
        auto B_exp                = std::make_shared<Expression::Expression>(B_arg);
        auto alpha_exp            = std::make_shared<Expression::Expression>(alpha_arg);
        auto strideD0_exp         = std::make_shared<Expression::Expression>(strideD0_arg);
        auto strideD1_exp         = std::make_shared<Expression::Expression>(strideD1_arg);
        auto strideC0_exp         = std::make_shared<Expression::Expression>(strideC0_arg);
        auto strideC1_exp         = std::make_shared<Expression::Expression>(strideC1_arg);
        auto strideA0_exp         = std::make_shared<Expression::Expression>(strideA0_arg);
        auto strideA1_exp         = std::make_shared<Expression::Expression>(strideA1_arg);
        auto strideB0_exp         = std::make_shared<Expression::Expression>(strideB0_arg);
        auto strideB1_exp         = std::make_shared<Expression::Expression>(strideB1_arg);
        auto SizesFree0_exp       = std::make_shared<Expression::Expression>(SizesFree0_arg);
        auto SizesFree1_exp       = std::make_shared<Expression::Expression>(SizesFree1_arg);
        auto SizesFree2_exp       = std::make_shared<Expression::Expression>(SizesFree2_arg);
        auto SizesSum0_exp        = std::make_shared<Expression::Expression>(SizesSum0_arg);
        auto OrigStaggerUIter_exp = std::make_shared<Expression::Expression>(OrigStaggerUIter_arg);
        auto NumWorkGroups0_exp   = std::make_shared<Expression::Expression>(NumWorkGroups0_arg);
        auto NumWorkGroups1_exp   = std::make_shared<Expression::Expression>(NumWorkGroups1_arg);
        auto NumFullBlocks_exp    = std::make_shared<Expression::Expression>(NumFullBlocks_arg);
        auto WgmRemainder1_exp    = std::make_shared<Expression::Expression>(WgmRemainder1_arg);
        auto MagicNumberWgmRemainder1_exp
            = std::make_shared<Expression::Expression>(MagicNumberWgmRemainder1_arg);
        auto OffsetD_exp = std::make_shared<Expression::Expression>(OffsetD_arg);
        auto OffsetC_exp = std::make_shared<Expression::Expression>(OffsetC_arg);
        auto OffsetA_exp = std::make_shared<Expression::Expression>(OffsetA_arg);
        auto OffsetB_exp = std::make_shared<Expression::Expression>(OffsetB_arg);
        auto padding_exp = std::make_shared<Expression::Expression>(padding_arg);

        auto k = m_context->kernel();

        k->setKernelName("Cijk_Ailk_Bjlk_HHS_BH_MT64x128x16_MI32x32x4x2_SN_K1");
        k->setKernelDimensions(3);

        k->addArgument(
            {"sizeC", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeC_exp});
        k->addArgument(
            {"sizeA", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeA_exp});
        k->addArgument(
            {"sizeB", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeB_exp});

        k->addArgument(
            {"D", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadWrite, D_exp});
        k->addArgument(
            {"C", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, C_exp});
        k->addArgument(
            {"A", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, A_exp});
        k->addArgument(
            {"B", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, B_exp});

        k->addArgument(
            {"alpha", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, alpha_exp});
        if(hasBeta)
        {
            auto beta_exp = std::make_shared<Expression::Expression>(beta_arg);
            k->addArgument(
                {"beta", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, beta_exp});
        }

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

        k->addArgument({"OffsetD",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetD_exp});
        k->addArgument({"OffsetC",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetC_exp});
        k->addArgument({"OffsetA",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetA_exp});
        k->addArgument({"OffsetB",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetB_exp});
        k->addArgument({"padding",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        padding_exp});

        auto workItem0 = std::make_shared<Expression::Expression>(NumWorkGroups0 * 256u);
        auto workItem1 = std::make_shared<Expression::Expression>(NumWorkGroups1);
        auto zero      = std::make_shared<Expression::Expression>(0u);
        auto one       = std::make_shared<Expression::Expression>(1u);

        k->setWorkgroupSize({256, 1, 1});
        k->setWorkitemCount({workItem0, workItem1, one});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto placeholderV = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Raw32, numVGPR);
        auto placeholderA = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Raw32, numACCGPR);
        auto placeholderS = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Raw32, numSGPR);
        auto placeholderLDS = Register::Value::AllocateLDS(m_context, DataType::Raw32, sizeLDS / 4);
        auto kb             = [&]() -> Generator<Instruction> {
            co_yield placeholderV->allocate();
            co_yield placeholderA->allocate();
            co_yield placeholderS->allocate();
        };
        m_context->schedule(kb());

        m_context->schedule(Program(m_context));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        unsigned long long sizeC = hostC.size(), sizeA = hostA.size(), sizeB = hostB.size();

        KernelArguments kargs(true);
        kargs.reserve(1024, 128);

        kargs.append("sizeC", sizeC);
        kargs.append("sizeA", sizeA);
        kargs.append("sizeB", sizeB);

        auto D = make_shared_device<__half>(sizeC, 0.0);
        auto C = make_shared_device(hostC);
        auto A = make_shared_device(hostA);
        auto B = make_shared_device(hostB);

        kargs.append("D", D.get());
        kargs.append("C", D.get());
        kargs.append("A", A.get());
        kargs.append("B", B.get());

        kargs.append("alpha", alpha);
        if(hasBeta)
            kargs.append("beta", beta);

        kargs.append("strideD0", strideD0);
        kargs.append("strideD1", strideD1);

        kargs.append("strideC0", strideC0);
        kargs.append("strideC1", strideC1);

        kargs.append("strideA0", strideA0);
        kargs.append("strideA1", strideA1);

        kargs.append("strideB0", strideB0);
        kargs.append("strideB1", strideB1);

        kargs.append("SizesFree0", SizesFree0);
        kargs.append("SizesFree1", SizesFree1);
        kargs.append("SizesFree2", SizesFree2);
        kargs.append("SizesSum0", SizesSum0);

        kargs.append("OrigStaggerUIter", OrigStaggerUIter);

        kargs.append("NumWorkGroups0", NumWorkGroups0);
        kargs.append("NumWorkGroups1", NumWorkGroups1);
        kargs.append("NumFullBlocks", NumFullBlocks);

        kargs.append("WgmRemainder1", WgmRemainder1);
        kargs.append("MagicNumberWgmRemainder1", MagicNumberWgmRemainder1);

        kargs.append("OffsetD", OffsetD);
        kargs.append("OffsetC", OffsetC);
        kargs.append("OffsetA", OffsetA);
        kargs.append("OffsetB", OffsetB);
        kargs.append("padding", padding);

        double total_time = 0;
        int    iters      = 10;
        TIMER(t_gpuMM, "GPUMM");
        TIC(t_gpuMM);
        for(int i = 0; i <= iters; i++)
        {
            hipEvent_t begin, end;
            ASSERT_THAT(hipEventCreate(&begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventCreate(&end), HasHipSuccess(0));
            ASSERT_THAT(hipEventRecord(begin, 0), HasHipSuccess(0));
            commandKernel.launchKernel(kargs.runtimeArguments());
            ASSERT_THAT(hipEventRecord(end, 0), HasHipSuccess(0));
            ASSERT_THAT(hipMemcpy(hostD.data(), D.get(), sizeC * sizeof(__half), hipMemcpyDefault),
                        HasHipSuccess(0));
            float elapsed = 0.f;
            ASSERT_THAT(hipEventElapsedTime(&elapsed, begin, end), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(end), HasHipSuccess(0));
            if(i > 0)
            {
                total_time += elapsed;
            }
        }
        TOC(t_gpuMM);
        std::cout << "Average Time: " << total_time / iters << " milliseconds" << std::endl;

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    TEST_F(GPU_GemmGuidePostTest, HGEMM_ManualKernel_Minimal)
    {
        TimerPool::clear();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        ASSERT_EQ(true, isLocalDevice());

        auto seed = 177232u;

        unsigned long long  sizeC = 64880640, sizeA = 64880640, sizeB = 71368704;
        std::vector<__half> hostD(sizeC);
        std::vector<__half> hostC;
        std::vector<__half> hostA;
        std::vector<__half> hostB;
        GenerateRandomInput(seed, hostC, sizeC, hostA, sizeA, hostB, sizeB);
        float        alpha = 1.0, beta = 0.0;
        unsigned int strideD0 = 7680, strideD1 = 64880640;
        unsigned int strideC0 = 7680, strideC1 = 64880640;
        unsigned int strideA0 = 7680, strideA1 = 64880640;
        unsigned int strideB0 = 8448, strideB1 = 71368704;
        unsigned int SizesFree0 = 7680, SizesFree1 = 8448, SizesFree2 = 1;
        unsigned int SizesSum0        = 8448;
        int          OrigStaggerUIter = 0;
        unsigned int NumWorkGroups0 = 120, NumWorkGroups1 = 66, NumFullBlocks = 66;
        unsigned int WgmRemainder1 = 0, MagicNumberWgmRemainder1 = 0;
        unsigned int OffsetD = 0, OffsetC = 0, OffsetA = 0, OffsetB = 0, padding = 0;

        doMinimalMM_HGEMM(m_context,
                          HGEMM_Minimal_Program,
                          hostD,
                          hostC,
                          hostA,
                          hostB,
                          alpha,
                          beta,
                          strideD0,
                          strideD1,
                          strideC0,
                          strideC1,
                          strideA0,
                          strideA1,
                          strideB0,
                          strideB1,
                          SizesFree0,
                          SizesFree1,
                          SizesFree2,
                          SizesSum0,
                          OrigStaggerUIter,
                          NumWorkGroups0,
                          NumWorkGroups1,
                          NumFullBlocks,
                          WgmRemainder1,
                          MagicNumberWgmRemainder1,
                          OffsetD,
                          OffsetC,
                          OffsetA,
                          OffsetB,
                          padding,
                          96,
                          32,
                          65,
                          14336,
                          false);

        std::vector<__half> cpuD(sizeC);
        CPUMM(cpuD, hostC, hostA, hostB, strideA0, strideB0, strideB0, alpha, beta);

        double rnorm2 = relativeNorm(hostD, cpuD);
        EXPECT_LT(rnorm2, 1.e-4);

        double rnormInf = relativeNormInf(hostD, cpuD);
        EXPECT_LT(rnormInf, 1.e-4);
    }

    void doMM_HGEMM(std::shared_ptr<rocRoller::Context> m_context,
                    KernelProgram                       Program,
                    std::vector<__half>&                hostD,
                    const std::vector<__half>&          hostC,
                    const std::vector<__half>&          hostA,
                    const std::vector<__half>&          hostB,
                    float                               alpha,
                    float                               beta,
                    unsigned int                        strideD0,
                    unsigned int                        strideD1,
                    unsigned int                        strideC0,
                    unsigned int                        strideC1,
                    unsigned int                        strideA0,
                    unsigned int                        strideA1,
                    unsigned int                        strideB0,
                    unsigned int                        strideB1,
                    unsigned int                        SizesFree0,
                    unsigned int                        SizesFree1,
                    unsigned int                        SizesFree2,
                    unsigned int                        SizesSum0,
                    int                                 OrigStaggerUIter,
                    unsigned int                        NumWorkGroups0,
                    unsigned int                        NumWorkGroups1,
                    unsigned int                        NumFullBlocks,
                    unsigned int                        WgmRemainder1,
                    unsigned int                        MagicNumberWgmRemainder1,
                    unsigned int                        OffsetD,
                    unsigned int                        OffsetC,
                    unsigned int                        OffsetA,
                    unsigned int                        OffsetB,
                    unsigned int                        padding,
                    unsigned int                        numVGPR,
                    unsigned int                        numACCGPR,
                    unsigned int                        numSGPR,
                    unsigned int                        sizeLDS,
                    bool                                hasBeta = true)
    {
        KernelOptions options;
        options.logLevel               = LogLevel::Debug;
        options.preloadKernelArguments = false;
        m_context->setKernelOptions(options);
        auto command = std::make_shared<Command>();

        VariableType floatPtr{DataType::Float, PointerType::PointerGlobal};
        VariableType floatVal{DataType::Float, PointerType::Value};
        VariableType uintVal{DataType::UInt32, PointerType::Value};
        VariableType ulongVal{DataType::UInt64, PointerType::Value};
        VariableType intVal{DataType::Int32, PointerType::Value};

        auto                             sizeC_arg = command->allocateArgument(ulongVal);
        auto                             sizeA_arg = command->allocateArgument(ulongVal);
        auto                             sizeB_arg = command->allocateArgument(ulongVal);
        auto                             D_arg     = command->allocateArgument(floatPtr);
        auto                             C_arg     = command->allocateArgument(floatPtr);
        auto                             A_arg     = command->allocateArgument(floatPtr);
        auto                             B_arg     = command->allocateArgument(floatPtr);
        auto                             alpha_arg = command->allocateArgument(floatVal);
        std::shared_ptr<CommandArgument> beta_arg;
        if(hasBeta)
            beta_arg = command->allocateArgument(floatVal);
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
        auto OffsetD_arg                  = command->allocateArgument(uintVal);
        auto OffsetC_arg                  = command->allocateArgument(uintVal);
        auto OffsetA_arg                  = command->allocateArgument(uintVal);
        auto OffsetB_arg                  = command->allocateArgument(uintVal);
        auto padding_arg                  = command->allocateArgument(uintVal);

        auto sizeC_exp            = std::make_shared<Expression::Expression>(sizeC_arg);
        auto sizeA_exp            = std::make_shared<Expression::Expression>(sizeA_arg);
        auto sizeB_exp            = std::make_shared<Expression::Expression>(sizeB_arg);
        auto D_exp                = std::make_shared<Expression::Expression>(D_arg);
        auto C_exp                = std::make_shared<Expression::Expression>(C_arg);
        auto A_exp                = std::make_shared<Expression::Expression>(A_arg);
        auto B_exp                = std::make_shared<Expression::Expression>(B_arg);
        auto alpha_exp            = std::make_shared<Expression::Expression>(alpha_arg);
        auto strideD0_exp         = std::make_shared<Expression::Expression>(strideD0_arg);
        auto strideD1_exp         = std::make_shared<Expression::Expression>(strideD1_arg);
        auto strideC0_exp         = std::make_shared<Expression::Expression>(strideC0_arg);
        auto strideC1_exp         = std::make_shared<Expression::Expression>(strideC1_arg);
        auto strideA0_exp         = std::make_shared<Expression::Expression>(strideA0_arg);
        auto strideA1_exp         = std::make_shared<Expression::Expression>(strideA1_arg);
        auto strideB0_exp         = std::make_shared<Expression::Expression>(strideB0_arg);
        auto strideB1_exp         = std::make_shared<Expression::Expression>(strideB1_arg);
        auto SizesFree0_exp       = std::make_shared<Expression::Expression>(SizesFree0_arg);
        auto SizesFree1_exp       = std::make_shared<Expression::Expression>(SizesFree1_arg);
        auto SizesFree2_exp       = std::make_shared<Expression::Expression>(SizesFree2_arg);
        auto SizesSum0_exp        = std::make_shared<Expression::Expression>(SizesSum0_arg);
        auto OrigStaggerUIter_exp = std::make_shared<Expression::Expression>(OrigStaggerUIter_arg);
        auto NumWorkGroups0_exp   = std::make_shared<Expression::Expression>(NumWorkGroups0_arg);
        auto NumWorkGroups1_exp   = std::make_shared<Expression::Expression>(NumWorkGroups1_arg);
        auto NumFullBlocks_exp    = std::make_shared<Expression::Expression>(NumFullBlocks_arg);
        auto WgmRemainder1_exp    = std::make_shared<Expression::Expression>(WgmRemainder1_arg);
        auto MagicNumberWgmRemainder1_exp
            = std::make_shared<Expression::Expression>(MagicNumberWgmRemainder1_arg);
        auto OffsetD_exp = std::make_shared<Expression::Expression>(OffsetD_arg);
        auto OffsetC_exp = std::make_shared<Expression::Expression>(OffsetC_arg);
        auto OffsetA_exp = std::make_shared<Expression::Expression>(OffsetA_arg);
        auto OffsetB_exp = std::make_shared<Expression::Expression>(OffsetB_arg);
        auto padding_exp = std::make_shared<Expression::Expression>(padding_arg);

        auto k = m_context->kernel();

        k->setKernelName("Cijk_Ailk_Bjlk_SB_MT128x64x16_MI32x32x2x1_SN_K1");
        k->setKernelDimensions(3);

        k->addArgument(
            {"sizeC", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeC_exp});
        k->addArgument(
            {"sizeA", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeA_exp});
        k->addArgument(
            {"sizeB", {DataType::UInt64, PointerType::Value}, DataDirection::ReadOnly, sizeB_exp});

        k->addArgument(
            {"D", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadWrite, D_exp});
        k->addArgument(
            {"C", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, C_exp});
        k->addArgument(
            {"A", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, A_exp});
        k->addArgument(
            {"B", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly, B_exp});

        k->addArgument(
            {"alpha", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, alpha_exp});
        if(hasBeta)
        {
            auto beta_exp = std::make_shared<Expression::Expression>(beta_arg);
            k->addArgument(
                {"beta", {DataType::Float, PointerType::Value}, DataDirection::ReadOnly, beta_exp});
        }

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

        k->addArgument({"OffsetD",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetD_exp});
        k->addArgument({"OffsetC",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetC_exp});
        k->addArgument({"OffsetA",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetA_exp});
        k->addArgument({"OffsetB",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        OffsetB_exp});
        k->addArgument({"padding",
                        {DataType::UInt32, PointerType::Value},
                        DataDirection::ReadOnly,
                        padding_exp});

        auto workItem0 = std::make_shared<Expression::Expression>(NumWorkGroups0 * 256u);
        auto workItem1 = std::make_shared<Expression::Expression>(NumWorkGroups1);
        auto zero      = std::make_shared<Expression::Expression>(0u);
        auto one       = std::make_shared<Expression::Expression>(1u);

        k->setWorkgroupSize({256, 1, 1});
        k->setWorkitemCount({workItem0, workItem1, one});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto placeholderV = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Raw32, numVGPR);
        auto placeholderA = std::make_shared<Register::Value>(
            m_context, Register::Type::Accumulator, DataType::Raw32, numACCGPR);
        auto placeholderS = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Raw32, numSGPR);
        auto placeholderLDS = Register::Value::AllocateLDS(m_context, DataType::Raw32, sizeLDS / 4);
        auto kb             = [&]() -> Generator<Instruction> {
            co_yield placeholderV->allocate();
            co_yield placeholderA->allocate();
            co_yield placeholderS->allocate();
        };
        m_context->schedule(kb());

        m_context->schedule(Program(m_context));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        CommandKernel commandKernel(m_context);

        unsigned long long sizeC = hostC.size(), sizeA = hostA.size(), sizeB = hostB.size();

        KernelArguments kargs(true);
        kargs.reserve(1024, 128);

        kargs.append("sizeC", sizeC);
        kargs.append("sizeA", sizeA);
        kargs.append("sizeB", sizeB);

        auto D = make_shared_device<__half>(sizeC);
        auto C = make_shared_device(hostC);
        auto A = make_shared_device(hostA);
        auto B = make_shared_device(hostB);

        kargs.append("D", D.get());
        kargs.append("C", D.get());
        kargs.append("A", A.get());
        kargs.append("B", B.get());

        kargs.append("alpha", alpha);
        if(hasBeta)
            kargs.append("beta", beta);

        kargs.append("strideD0", strideD0);
        kargs.append("strideD1", strideD1);

        kargs.append("strideC0", strideC0);
        kargs.append("strideC1", strideC1);

        kargs.append("strideA0", strideA0);
        kargs.append("strideA1", strideA1);

        kargs.append("strideB0", strideB0);
        kargs.append("strideB1", strideB1);

        kargs.append("SizesFree0", SizesFree0);
        kargs.append("SizesFree1", SizesFree1);
        kargs.append("SizesFree2", SizesFree2);
        kargs.append("SizesSum0", SizesSum0);

        kargs.append("OrigStaggerUIter", OrigStaggerUIter);

        kargs.append("NumWorkGroups0", NumWorkGroups0);
        kargs.append("NumWorkGroups1", NumWorkGroups1);
        kargs.append("NumFullBlocks", NumFullBlocks);

        kargs.append("WgmRemainder1", WgmRemainder1);
        kargs.append("MagicNumberWgmRemainder1", MagicNumberWgmRemainder1);

        kargs.append("OffsetD", OffsetD);
        kargs.append("OffsetC", OffsetC);
        kargs.append("OffsetA", OffsetA);
        kargs.append("OffsetB", OffsetB);
        kargs.append("padding", padding);

        double total_time = 0;
        int    iters      = 10;
        TIMER(t_gpuMM, "GPUMM");
        TIC(t_gpuMM);
        for(int i = 0; i <= iters; i++)
        {
            hipEvent_t begin, end;
            ASSERT_THAT(hipEventCreate(&begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventCreate(&end), HasHipSuccess(0));
            ASSERT_THAT(hipEventRecord(begin, 0), HasHipSuccess(0));
            commandKernel.launchKernel(kargs.runtimeArguments());
            ASSERT_THAT(hipEventRecord(end, 0), HasHipSuccess(0));
            ASSERT_THAT(hipMemcpy(hostD.data(), D.get(), sizeC * sizeof(__half), hipMemcpyDefault),
                        HasHipSuccess(0));
            float elapsed = 0.f;
            ASSERT_THAT(hipEventElapsedTime(&elapsed, begin, end), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(begin), HasHipSuccess(0));
            ASSERT_THAT(hipEventDestroy(end), HasHipSuccess(0));
            if(i > 0)
            {
                total_time += elapsed;
            }
        }
        TOC(t_gpuMM);
        std::cout << "Average Time: " << total_time / iters << " milliseconds" << std::endl;

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    TEST_F(GPU_GemmGuidePostTest, HGEMM_ManualKernel_Optimized)
    {
        TimerPool::clear();

        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);
        ASSERT_EQ(true, isLocalDevice());

        auto seed = 375250u;

        unsigned long long  sizeC = 64880640, sizeA = 64880640, sizeB = 71368704;
        std::vector<__half> hostD(sizeC);
        std::vector<__half> hostC;
        std::vector<__half> hostA;
        std::vector<__half> hostB;
        GenerateRandomInput(seed, hostC, sizeC, hostA, sizeA, hostB, sizeB);
        float        alpha = 1.0, beta = 0.0;
        unsigned int strideD0 = 7680, strideD1 = 64880640;
        unsigned int strideC0 = 7680, strideC1 = 64880640;
        unsigned int strideA0 = 7680, strideA1 = 64880640;
        unsigned int strideB0 = 8448, strideB1 = 71368704;
        unsigned int SizesFree0 = 7680, SizesFree1 = 8448, SizesFree2 = 1;
        unsigned int SizesSum0        = 8448;
        int          OrigStaggerUIter = 0;
        unsigned int NumWorkGroups0 = 60, NumWorkGroups1 = 33, NumFullBlocks = 2;
        unsigned int WgmRemainder1 = 3, MagicNumberWgmRemainder1 = 715827883;
        unsigned int OffsetD = 0, OffsetC = 0, OffsetA = 0, OffsetB = 0, padding = 0;

        doMM_HGEMM(m_context,
                   HGEMM_Optimized_Program,
                   hostD,
                   hostC,
                   hostA,
                   hostB,
                   alpha,
                   beta,
                   strideD0,
                   strideD1,
                   strideC0,
                   strideC1,
                   strideA0,
                   strideA1,
                   strideB0,
                   strideB1,
                   SizesFree0,
                   SizesFree1,
                   SizesFree2,
                   SizesSum0,
                   OrigStaggerUIter,
                   NumWorkGroups0,
                   NumWorkGroups1,
                   NumFullBlocks,
                   WgmRemainder1,
                   MagicNumberWgmRemainder1,
                   OffsetD,
                   OffsetC,
                   OffsetA,
                   OffsetB,
                   padding,
                   128,
                   128,
                   65,
                   28672,
                   false);

        std::vector<__half> cpuD(sizeC);
        CPUMM(cpuD, hostC, hostA, hostB, strideA0, strideB0, strideB0, alpha, beta);

        double rnorm2 = relativeNorm(hostD, cpuD);
        EXPECT_LT(rnorm2, 1.e-4);

        double rnormInf = relativeNormInf(hostD, cpuD);
        EXPECT_LT(rnormInf, 1.e-4);
    }
}
