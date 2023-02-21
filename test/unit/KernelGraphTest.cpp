
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>
#include <variant>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{

    class KernelGraphTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp()
        {
            CurrentGPUContextFixture::SetUp();
            Settings::getInstance()->set(Settings::SaveAssembly, true);

            fastArith = Expression::FastArithmetic(m_context);
        }

        static std::shared_ptr<Command> commonCommand()
        {
            auto command = std::make_shared<rocRoller::Command>();

            Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
            command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

            Operations::T_Load_Linear load_B(DataType::Int32, 1, 2);
            command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));

            Operations::T_Execute execute;
            execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 2, 0)));
            execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(4, 3)));
            execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 4)));

            command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

            Operations::T_Store_Linear store_C(1, 5);
            command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));
            return command;
        }

        void GPU_Translate04(bool reload);
    };

    class KernelGraphTest : public GenericContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp()
        {
            GenericContextFixture::SetUp();
            fastArith = Expression::FastArithmetic(m_context);
        }

        static std::shared_ptr<Command> commonCommand()
        {
            return KernelGraphTestGPU::commonCommand();
        }
    };

// TODO update this
#if 0
    class KernelGraphTestGPULoopSize : public KernelGraphTestGPU,
                                       public ::testing::WithParamInterface<int>
    {
    };

    TEST_P(KernelGraphTestGPULoopSize, MissingWorkitemCount)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        int  loopSize     = GetParam();
        auto loopSizeExpr = Expression::literal(loopSize);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / loopSizeExpr;

        ASSERT_THROW(
            {
                auto kgraph = KernelGraph::translate(command);

                kgraph = KernelGraph::lowerLinear(kgraph, m_context);

                kgraph = KernelGraph::lowerLinearLoop(kgraph, loopSizeExpr, m_context);

                kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

                m_context->kernel()->setWorkitemCount({numWorkitems, one, one});
            },
            FatalError);
    }

    TEST_P(KernelGraphTestGPULoopSize, TestForLoop)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        int  loopSize     = GetParam();
        auto loopSizeExpr = Expression::literal(loopSize);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / loopSizeExpr;

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        size_t origArgSize = m_context->kernel()->arguments().size();

        auto kgraph = KernelGraph::translate2(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

        kgraph = KernelGraph::lowerLinearLoop(kgraph, loopSizeExpr, m_context);

        kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        EXPECT_EQ(m_context->kernel()->arguments().size(), origArgSize + 1);
        ASSERT_NO_THROW(m_context->kernel()->findArgument("LAUNCH_WORKGROUPCOUNT_0"));

        CommandKernel commandKernel(command, m_context, kgraph);

        RandomGenerator random(1356);

        int              baseSize = workGroupSize * loopSize;
        std::vector<int> vecSizes = {baseSize, baseSize * 5, baseSize * 16, baseSize * 65};
        for(auto vecSize : vecSizes)
        {
            auto             a          = random.vector<int>(vecSize, -1000, 1000);
            auto             b          = random.vector<int>(vecSize, -1000, 1000);
            auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
            auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
            std::vector<int> c(vecSize);
            for(int i = 0; i < vecSize; i++)
                c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

            auto a_d = make_shared_device<int>(vecSize);
            auto b_d = make_shared_device<int>(vecSize);
            auto c_d = make_shared_device<int>(vecSize);

            ASSERT_THAT(
                hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                HasHipSuccess(0));
            ASSERT_THAT(
                hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                HasHipSuccess(0));

            KernelArguments args;
            args.append("a", a_d.get());
            args.append<int64_t>("a_extent", vecSize);
            args.append<int64_t>("a_size", vecSize);
            args.append<int64_t>("a_stride", 1);

            args.append("b", b_d.get());
            args.append<int64_t>("b_extent", vecSize);
            args.append<int64_t>("b_size", vecSize);
            args.append<int64_t>("b_stride", 1);

            args.append("c", c_d.get());
            args.append<int64_t>("c_extent", vecSize);
            // args.append<int64_t>("c_size", vecSize);
            args.append<int64_t>("c_stride", 1);

            commandKernel.launchKernel(args.runtimeArguments());

            ASSERT_THAT(
                hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
                HasHipSuccess(0));

            EXPECT_THAT(output(), testing::HasSubstr("Lock For Loop"));
            EXPECT_THAT(output(), testing::HasSubstr("Unlock For Loop"));

            for(int i = 0; i < vecSize; i++)
                EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
        }
    }

    // delete this when graph rearch complete
    TEST_P(KernelGraphTestGPULoopSize, TestForLoop2)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        int  loopSize     = GetParam();
        auto loopSizeExpr = Expression::literal(loopSize);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / loopSizeExpr;

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        size_t origArgSize = m_context->kernel()->arguments().size();

        auto kgraph = KernelGraph::translate2(command);
        kgraph      = KernelGraph::lowerLinear(kgraph, m_context);
        kgraph      = KernelGraph::lowerLinearLoop(kgraph, loopSizeExpr, m_context);
        kgraph      = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        EXPECT_EQ(m_context->kernel()->arguments().size(), origArgSize + 1);
        ASSERT_NO_THROW(m_context->kernel()->findArgument("LAUNCH_WORKGROUPCOUNT_0"));

        auto context = m_context;
        context->schedule(context->kernel()->preamble());
        context->schedule(context->kernel()->prolog());
        context->schedule(KernelGraph::generate(kgraph, context->kernel()));
        context->schedule(context->kernel()->postamble());
        context->schedule(context->kernel()->amdgpu_metadata());
        auto executableKernel = m_context->instructions()->getExecutableKernel();

        RandomGenerator random(1356);

        int              baseSize = workGroupSize * loopSize;
        std::vector<int> vecSizes = {baseSize, baseSize * 5, baseSize * 16, baseSize * 65};
        for(auto vecSize : vecSizes)
        {
            auto             a          = random.vector<int>(vecSize, -1000, 1000);
            auto             b          = random.vector<int>(vecSize, -1000, 1000);
            auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
            auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
            std::vector<int> c(vecSize);
            for(int i = 0; i < vecSize; i++)
                c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

            auto a_d = make_shared_device<int>(vecSize);
            auto b_d = make_shared_device<int>(vecSize);
            auto c_d = make_shared_device<int>(vecSize);

            ASSERT_THAT(
                hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                HasHipSuccess(0));
            ASSERT_THAT(
                hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                HasHipSuccess(0));

            KernelArguments args;
            args.append("a", a_d.get());
            args.append<int64_t>("a_extent", vecSize);
            args.append<int64_t>("a_size", vecSize);
            args.append<int64_t>("a_stride", 1);

            args.append("b", b_d.get());
            args.append<int64_t>("b_extent", vecSize);
            args.append<int64_t>("b_size", vecSize);
            args.append<int64_t>("b_stride", 1);

            args.append("c", c_d.get());
            args.append<int64_t>("c_extent", vecSize);
            args.append<int64_t>("c_stride", 1);

            args.append<int64_t>("LAUNCH_WORKGROUPCOUNT_0", vecSize / baseSize);

            KernelInvocation kinv;
            kinv.workgroupSize    = context->kernel()->workgroupSize();
            kinv.workitemCount[0] = vecSize / loopSize;

            executableKernel->executeKernel(args, kinv);

            ASSERT_THAT(
                hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
                HasHipSuccess(0));

            EXPECT_THAT(output(), testing::HasSubstr("Lock For Loop"));
            EXPECT_THAT(output(), testing::HasSubstr("Unlock For Loop"));

            for(int i = 0; i < vecSize; i++)
                EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
        }
    }

    INSTANTIATE_TEST_SUITE_P(KernelGraphTestGPULoopSize,
                             KernelGraphTestGPULoopSize,
                             ::testing::ValuesIn({1, 5, 16, 73}));

    TEST_F(KernelGraphTestGPU, TestKernelUnroll)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto unrollSize = Expression::literal(4);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / unrollSize;

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        auto kgraph = KernelGraph::translate2(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

//        kgraph = KernelGraph::lowerLinearUnroll(kgraph, unrollSize, m_context);

        kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        m_context->kernel()->setKernelGraphMeta(std::make_shared<KernelGraph::KernelGraph>(kgraph));

        CommandKernel commandKernel(command, m_context, kgraph);

        RandomGenerator random(8379);

        int vecSize = 16384;

        auto             a          = random.vector<int>(vecSize, -1000, 1000);
        auto             b          = random.vector<int>(vecSize, -1000, 1000);
        auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
        auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
        std::vector<int> c(vecSize);
        for(int i = 0; i < vecSize; i++)
            c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

        auto a_d = make_shared_device<int>(vecSize);
        auto b_d = make_shared_device<int>(vecSize);
        auto c_d = make_shared_device<int>(vecSize);

        ASSERT_THAT(hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));

        KernelArguments args;
        args.append("a", a_d.get());
        args.append<int64_t>("a_extent", vecSize);
        args.append<int64_t>("a_size", vecSize);
        args.append<int64_t>("a_stride", 1);

        args.append("b", b_d.get());
        args.append<int64_t>("b_extent", vecSize);
        args.append<int64_t>("b_size", vecSize);
        args.append<int64_t>("b_stride", 1);

        args.append("c", c_d.get());
        args.append<int64_t>("c_extent", vecSize);
        // args.append<int64_t>("c_size", vecSize);
        args.append<int64_t>("c_stride", 1);

        commandKernel.launchKernel(args.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
            HasHipSuccess(0));

        for(int i = 0; i < vecSize; i++)
            EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
    }

    TEST_F(KernelGraphTestGPU, TestKernelUnrollAndLoop)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto unrollSize = Expression::literal(4);
        auto loopSize   = Expression::literal(16);

        auto one          = Expression::literal(1u);
        auto extent       = std::make_shared<Expression::Expression>(command->getArguments()[1]);
        auto numWorkitems = extent / (unrollSize * loopSize);

        m_context->kernel()->setWorkitemCount({numWorkitems, one, one});

        auto kgraph = KernelGraph::translate(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

        kgraph = KernelGraph::lowerLinearLoop(kgraph, loopSize, m_context);
        kgraph = KernelGraph::lowerLinearUnroll(kgraph, unrollSize, m_context);

        kgraph = KernelGraph::cleanArguments(kgraph, m_context->kernel());

        m_context->kernel()->setKernelGraphMeta(std::make_shared<KernelGraph::KernelGraph>(kgraph));

        CommandKernel commandKernel(command, m_context, kgraph);

        RandomGenerator random(68103);

        int vecSize = 16384;

        auto             a          = random.vector<int>(vecSize, -1000, 1000);
        auto             b          = random.vector<int>(vecSize, -1000, 1000);
        auto             c_expected = random.vector<int>(vecSize, -1000, 1000);
        auto             c_actual   = random.vector<int>(vecSize, -1000, 1000);
        std::vector<int> c(vecSize);
        for(int i = 0; i < vecSize; i++)
            c_expected[i] = -(a[i] + b[i]) * (a[i] + b[i]);

        auto a_d = make_shared_device<int>(vecSize);
        auto b_d = make_shared_device<int>(vecSize);
        auto c_d = make_shared_device<int>(vecSize);

        ASSERT_THAT(hipMemcpy(a_d.get(), a.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));
        ASSERT_THAT(hipMemcpy(b_d.get(), b.data(), vecSize * sizeof(int), hipMemcpyHostToDevice),
                    HasHipSuccess(0));

        KernelArguments args;
        args.append("a", a_d.get());
        args.append<int64_t>("a_extent", vecSize);
        args.append<int64_t>("a_size", vecSize);
        args.append<int64_t>("a_stride", 1);

        args.append("b", b_d.get());
        args.append<int64_t>("b_extent", vecSize);
        args.append<int64_t>("b_size", vecSize);
        args.append<int64_t>("b_stride", 1);

        args.append("c", c_d.get());
        args.append<int64_t>("c_extent", vecSize);
        // args.append<int64_t>("c_size", vecSize);
        args.append<int64_t>("c_stride", 1);

        commandKernel.launchKernel(args.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(c_actual.data(), c_d.get(), vecSize * sizeof(int), hipMemcpyDeviceToHost),
            HasHipSuccess(0));

        for(int i = 0; i < vecSize; i++)
            EXPECT_EQ(c_actual[i], c_expected[i]) << i << ", " << a[i] << ", " << b[i];
    }
#endif

    TEST_F(KernelGraphTest, Translate01)
    {
        auto command = commonCommand();
        auto kgraph0 = translate(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto visitor = rocRoller::KernelGraph::BaseGraphVisitor(m_context);
        auto kgraphC = rewrite(kgraph0, visitor);

        std::string expectedC = R".(
                digraph {
                "coord1"[label="User{NA}(1)"];
                "coord2"[label="User{NA}(2)"];
                "coord3"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(3)"];
                "coord4"[label="Split(4)",shape=box];
                "coord5"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(5)"];
                "coord6"[label="Flatten(6)",shape=box];
                "coord7"[label="DataFlow(7)",shape=box];
                "coord8"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(8)"];
                "coord9"[label="Split(9)",shape=box];
                "coord10"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(10)"];
                "coord11"[label="Flatten(11)",shape=box];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord13"[label="Linear{NA}(13)"];
                "coord14"[label="DataFlow(14)",shape=box];
                "coord15"[label="Linear{NA}(15)"];
                "coord16"[label="DataFlow(16)",shape=box];
                "coord17"[label="Linear{NA}(17)"];
                "coord18"[label="DataFlow(18)",shape=box];
                "coord19"[label="SubDimension{0, NA}(19)"];
                "coord20"[label="Split(20)",shape=box];
                "coord21"[label="User{NA}(21)"];
                "coord22"[label="Join(22)",shape=box];
                "coord23"[label="DataFlow(23)",shape=box];
                "coord1" -> "coord9"
                "coord1" -> "coord12"
                "coord2" -> "coord4"
                "coord2" -> "coord7"
                "coord3" -> "coord6"
                "coord4" -> "coord3"
                "coord5" -> "coord14"
                "coord6" -> "coord5"
                "coord7" -> "coord5"
                "coord8" -> "coord11"
                "coord9" -> "coord8"
                "coord10" -> "coord14"
                "coord11" -> "coord10"
                "coord12" -> "coord10"
                "coord13" -> "coord16"
                "coord13" -> "coord18"
                "coord14" -> "coord13"
                "coord15" -> "coord18"
                "coord16" -> "coord15"
                "coord17" -> "coord20"
                "coord17" -> "coord23"
                "coord18" -> "coord17"
                "coord19" -> "coord22"
                "coord20" -> "coord19"
                "coord22" -> "coord21"
                "coord23" -> "coord21"
                {
                rank=same
                "coord5"->"coord10"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord13"->"coord15"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(5), DataFlowTag(10))(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(13))(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(13), DataFlowTag(15))(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl14"[label="StoreLinear(14)"];
                "cntrl15"[label="Sequence(15)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
                "cntrl6" -> "cntrl10"
                "cntrl6" -> "cntrl12"
                "cntrl7" -> "cntrl6"
                "cntrl8" -> "cntrl6"
                "cntrl9" -> "cntrl13"
                "cntrl10" -> "cntrl9"
                "cntrl11" -> "cntrl15"
                "cntrl12" -> "cntrl11"
                "cntrl13" -> "cntrl11"
                "cntrl15" -> "cntrl14"
                }
		"coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord5" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord10" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
                "coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
                "coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
                "coord17" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
                "coord17" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
                "coord21" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
                }).";

        EXPECT_EQ(NormalizedSource(expectedC), NormalizedSource(kgraphC.toDOT(true)));

        std::string expected0 = R".(
                digraph {
                "coord1"[label="User{NA}(1)"];
                "coord2"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(2)"];
                "coord3"[label="Split(3)",shape=box];
                "coord4"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(4)"];
                "coord5"[label="Flatten(5)",shape=box];
                "coord6"[label="DataFlow(6)",shape=box];
                "coord7"[label="User{NA}(7)"];
                "coord8"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(8)"];
                "coord9"[label="Split(9)",shape=box];
                "coord10"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(10)"];
                "coord11"[label="Flatten(11)",shape=box];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord13"[label="Linear{NA}(13)"];
                "coord14"[label="DataFlow(14)",shape=box];
                "coord15"[label="Linear{NA}(15)"];
                "coord16"[label="DataFlow(16)",shape=box];
                "coord17"[label="Linear{NA}(17)"];
                "coord18"[label="DataFlow(18)",shape=box];
                "coord19"[label="SubDimension{0, NA}(19)"];
                "coord20"[label="User{NA}(20)"];
                "coord21"[label="Split(21)",shape=box];
                "coord22"[label="Join(22)",shape=box];
                "coord23"[label="DataFlow(23)",shape=box];
                "coord1" -> "coord3"
                "coord1" -> "coord6"
                "coord2" -> "coord5"
                "coord3" -> "coord2"
                "coord4" -> "coord14"
                "coord5" -> "coord4"
                "coord6" -> "coord4"
                "coord7" -> "coord9"
                "coord7" -> "coord12"
                "coord8" -> "coord11"
                "coord9" -> "coord8"
                "coord10" -> "coord14"
                "coord11" -> "coord10"
                "coord12" -> "coord10"
                "coord13" -> "coord16"
                "coord13" -> "coord18"
                "coord14" -> "coord13"
                "coord15" -> "coord18"
                "coord16" -> "coord15"
                "coord17" -> "coord21"
                "coord17" -> "coord23"
                "coord18" -> "coord17"
                "coord19" -> "coord22"
                "coord21" -> "coord19"
                "coord22" -> "coord20"
                "coord23" -> "coord20"
                {
                rank=same
                "coord10"->"coord4"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord13"->"coord15"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(10), DataFlowTag(4))(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(13))(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(13), DataFlowTag(15))(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl14"[label="StoreLinear(14)"];
                "cntrl15"[label="Sequence(15)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl8"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl7"
                "cntrl5" -> "cntrl4"
                "cntrl6" -> "cntrl10"
                "cntrl6" -> "cntrl12"
                "cntrl7" -> "cntrl6"
                "cntrl8" -> "cntrl6"
                "cntrl9" -> "cntrl13"
                "cntrl10" -> "cntrl9"
                "cntrl11" -> "cntrl15"
                "cntrl12" -> "cntrl11"
                "cntrl13" -> "cntrl11"
                "cntrl15" -> "cntrl14"
                }
                "coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
                "coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
                "coord7" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
                "coord10" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
                "coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
                "coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
                "coord17" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
                "coord17" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
                "coord20" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
	     }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        std::string expected1 = R".(
            digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="User{NA}(2)"];
		"coord3"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(3)"];
		"coord4"[label="Split(4)",shape=box];
		"coord5"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(5)"];
		"coord6"[label="Flatten(6)",shape=box];
		"coord7"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(7)"];
		"coord8"[label="Split(8)",shape=box];
		"coord9"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(9)"];
		"coord10"[label="Flatten(10)",shape=box];
		"coord11"[label="Linear{NA}(11)"];
		"coord12"[label="SubDimension{0, NA}(12)"];
		"coord13"[label="Split(13)",shape=box];
		"coord14"[label="User{NA}(14)"];
		"coord15"[label="Join(15)",shape=box];
		"coord16"[label="VGPR{NA}(16)"];
		"coord17"[label="Workgroup{0, LAUNCH_WORKGROUPCOUNT_0}(17)"];
		"coord18"[label="Workitem{0, 32j}(18)"];
		"coord19"[label="Tile(19)",shape=box];
		"coord20"[label="Forget(20)",shape=box];
		"coord21"[label="DataFlow(21)",shape=box];
		"coord22"[label="VGPR{NA}(22)"];
		"coord23"[label="Workgroup{0, LAUNCH_WORKGROUPCOUNT_0}(23)"];
		"coord24"[label="Workitem{0, 32j}(24)"];
		"coord25"[label="Tile(25)",shape=box];
		"coord26"[label="Forget(26)",shape=box];
		"coord27"[label="DataFlow(27)",shape=box];
		"coord28"[label="VGPR{NA}(28)"];
		"coord29"[label="DataFlow(29)",shape=box];
		"coord30"[label="VGPR{NA}(30)"];
		"coord31"[label="DataFlow(31)",shape=box];
		"coord32"[label="VGPR{NA}(32)"];
		"coord33"[label="DataFlow(33)",shape=box];
		"coord34"[label="Workgroup{0, LAUNCH_WORKGROUPCOUNT_0}(34)"];
		"coord35"[label="Workitem{0, 32j}(35)"];
		"coord36"[label="Inherit(36)",shape=box];
		"coord37"[label="Flatten(37)",shape=box];
		"coord38"[label="DataFlow(38)",shape=box];
		"coord1" -> "coord8"
		"coord1" -> "coord27"
		"coord2" -> "coord4"
		"coord2" -> "coord21"
		"coord3" -> "coord6"
		"coord4" -> "coord3"
		"coord5" -> "coord19"
		"coord6" -> "coord5"
		"coord7" -> "coord10"
		"coord8" -> "coord7"
		"coord9" -> "coord25"
		"coord10" -> "coord9"
		"coord11" -> "coord13"
		"coord12" -> "coord15"
		"coord13" -> "coord12"
		"coord15" -> "coord14"
		"coord16" -> "coord29"
		"coord17" -> "coord20"
		"coord18" -> "coord20"
		"coord19" -> "coord17"
		"coord19" -> "coord18"
		"coord20" -> "coord16"
		"coord21" -> "coord16"
		"coord22" -> "coord29"
		"coord23" -> "coord26"
		"coord24" -> "coord26"
		"coord25" -> "coord23"
		"coord25" -> "coord24"
		"coord26" -> "coord22"
		"coord27" -> "coord22"
		"coord28" -> "coord31"
		"coord28" -> "coord33"
		"coord29" -> "coord28"
		"coord30" -> "coord33"
		"coord31" -> "coord30"
		"coord32" -> "coord36"
		"coord32" -> "coord38"
		"coord33" -> "coord32"
		"coord34" -> "coord37"
		"coord35" -> "coord37"
		"coord36" -> "coord34"
		"coord36" -> "coord35"
		"coord37" -> "coord11"
		"coord38" -> "coord14"
		{
		rank=same
		"coord17"->"coord18"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord17"->"coord18"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord23"->"coord24"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord23"->"coord24"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord16"->"coord22"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord28"->"coord30"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord34"->"coord35"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord34"->"coord35"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadVGPR(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadVGPR(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="Assign VGPR Add(DataFlowTag(16), DataFlowTag(22))(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="Assign VGPR Negate(DataFlowTag(28))(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="Assign VGPR Multiply(DataFlowTag(28), DataFlowTag(30))(11)"];
		"cntrl12"[label="Sequence(12)",shape=box];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="StoreVGPR(14)"];
		"cntrl15"[label="Sequence(15)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
		"cntrl5" -> "cntrl4"
		"cntrl6" -> "cntrl10"
		"cntrl6" -> "cntrl12"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		"cntrl9" -> "cntrl13"
		"cntrl10" -> "cntrl9"
		"cntrl11" -> "cntrl15"
		"cntrl12" -> "cntrl11"
		"cntrl13" -> "cntrl11"
		"cntrl15" -> "cntrl14"
		}
		"coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord16" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord22" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord28" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord30" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
		"coord32" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
		"coord14" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
		"coord32" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
        }).";

        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph1 = lowerLinear(kgraph0, m_context);
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));

        std::string expected2 = R".(
	    digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="User{NA}(2)"];
		"coord3"[label="SubDimension{0, CommandArgument(Load_Linear_2_size_0)}(3)"];
		"coord4"[label="Split(4)",shape=box];
		"coord5"[label="Linear{CommandArgument(Load_Linear_2_size_0)}(5)"];
		"coord6"[label="Flatten(6)",shape=box];
		"coord7"[label="Workgroup{0, LAUNCH_WORKGROUPCOUNT_0}(7)"];
		"coord8"[label="Workitem{0, 32j}(8)"];
		"coord9"[label="Tile(9)",shape=box];
		"coord10"[label="Linear{16i}(10)"];
		"coord11"[label="ForLoop{16i}(11)"];
		"coord12"[label="DataFlow(12)",shape=box];
		"coord13"[label="VGPR{NA}(13)"];
		"coord14"[label="Forget(14)",shape=box];
		"coord15"[label="DataFlow(15)",shape=box];
		"coord16"[label="SubDimension{0, CommandArgument(Load_Linear_0_size_0)}(16)"];
		"coord17"[label="Split(17)",shape=box];
		"coord18"[label="Linear{CommandArgument(Load_Linear_0_size_0)}(18)"];
		"coord19"[label="Flatten(19)",shape=box];
		"coord20"[label="Workgroup{0, LAUNCH_WORKGROUPCOUNT_0}(20)"];
		"coord21"[label="Workitem{0, 32j}(21)"];
		"coord22"[label="Tile(22)",shape=box];
		"coord23"[label="ForLoop{16i}(23)"];
		"coord24"[label="DataFlow(24)",shape=box];
		"coord25"[label="VGPR{NA}(25)"];
		"coord26"[label="Forget(26)",shape=box];
		"coord27"[label="DataFlow(27)",shape=box];
		"coord28"[label="VGPR{NA}(28)"];
		"coord29"[label="DataFlow(29)",shape=box];
		"coord30"[label="VGPR{NA}(30)"];
		"coord31"[label="DataFlow(31)",shape=box];
		"coord32"[label="VGPR{NA}(32)"];
		"coord33"[label="DataFlow(33)",shape=box];
		"coord34"[label="Workgroup{0, LAUNCH_WORKGROUPCOUNT_0}(34)"];
		"coord35"[label="Workitem{0, 32j}(35)"];
		"coord36"[label="Inherit(36)",shape=box];
		"coord37"[label="ForLoop{16i}(37)"];
		"coord38"[label="DataFlow(38)",shape=box];
		"coord39"[label="Linear{NA}(39)"];
		"coord40"[label="Flatten(40)",shape=box];
		"coord41"[label="SubDimension{0, NA}(41)"];
		"coord42"[label="Split(42)",shape=box];
		"coord43"[label="User{NA}(43)"];
		"coord44"[label="Join(44)",shape=box];
		"coord45"[label="DataFlow(45)",shape=box];
		"coord1" -> "coord17"
		"coord1" -> "coord27"
		"coord2" -> "coord4"
		"coord2" -> "coord15"
		"coord3" -> "coord6"
		"coord4" -> "coord3"
		"coord5" -> "coord9"
		"coord6" -> "coord5"
		"coord7" -> "coord14"
		"coord8" -> "coord14"
		"coord9" -> "coord11"
		"coord9" -> "coord7"
		"coord9" -> "coord8"
		"coord10" -> "coord12"
		"coord10" -> "coord24"
		"coord10" -> "coord38"
		"coord11" -> "coord14"
		"coord12" -> "coord11"
		"coord13" -> "coord29"
		"coord14" -> "coord13"
		"coord15" -> "coord13"
		"coord16" -> "coord19"
		"coord17" -> "coord16"
		"coord18" -> "coord22"
		"coord19" -> "coord18"
		"coord20" -> "coord26"
		"coord21" -> "coord26"
		"coord22" -> "coord23"
		"coord22" -> "coord20"
		"coord22" -> "coord21"
		"coord23" -> "coord26"
		"coord24" -> "coord23"
		"coord25" -> "coord29"
		"coord26" -> "coord25"
		"coord27" -> "coord25"
		"coord28" -> "coord31"
		"coord28" -> "coord33"
		"coord29" -> "coord28"
		"coord30" -> "coord33"
		"coord31" -> "coord30"
		"coord32" -> "coord36"
		"coord32" -> "coord45"
		"coord33" -> "coord32"
		"coord34" -> "coord40"
		"coord35" -> "coord40"
		"coord36" -> "coord37"
		"coord36" -> "coord34"
		"coord36" -> "coord35"
		"coord37" -> "coord40"
		"coord38" -> "coord37"
		"coord39" -> "coord42"
		"coord40" -> "coord39"
		"coord41" -> "coord44"
		"coord42" -> "coord41"
		"coord44" -> "coord43"
		"coord45" -> "coord43"
		{
		rank=same
		"coord11"->"coord7"->"coord8"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord11"->"coord7"->"coord8"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord23"->"coord20"->"coord21"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord23"->"coord20"->"coord21"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord13"->"coord25"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord28"->"coord30"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord37"->"coord34"->"coord35"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord37"->"coord34"->"coord35"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="ForLoopOp : LessThan(DataFlowTag(10), 16i)(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="Assign SGPR 0i(4)"];
		"cntrl5"[label="Initialize(5)",shape=box];
		"cntrl6"[label="Assign SGPR Add(DataFlowTag(10), 1i)(6)"];
		"cntrl7"[label="ForLoopIncrement(7)",shape=box];
		"cntrl8"[label="LoadVGPR(8)"];
		"cntrl9"[label="Body(9)",shape=box];
		"cntrl10"[label="LoadVGPR(10)"];
		"cntrl11"[label="Body(11)",shape=box];
		"cntrl12"[label="Assign VGPR Add(DataFlowTag(13), DataFlowTag(25))(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="Assign VGPR Negate(DataFlowTag(28))(15)"];
		"cntrl16"[label="Sequence(16)",shape=box];
		"cntrl17"[label="Assign VGPR Multiply(DataFlowTag(28), DataFlowTag(30))(17)"];
		"cntrl18"[label="Sequence(18)",shape=box];
		"cntrl19"[label="Sequence(19)",shape=box];
		"cntrl20"[label="StoreVGPR(20)"];
		"cntrl21"[label="Sequence(21)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl2" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl2" -> "cntrl9"
		"cntrl2" -> "cntrl11"
		"cntrl3" -> "cntrl2"
		"cntrl5" -> "cntrl4"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl13"
		"cntrl9" -> "cntrl8"
		"cntrl10" -> "cntrl14"
		"cntrl11" -> "cntrl10"
		"cntrl12" -> "cntrl16"
		"cntrl12" -> "cntrl18"
		"cntrl13" -> "cntrl12"
		"cntrl14" -> "cntrl12"
		"cntrl15" -> "cntrl19"
		"cntrl16" -> "cntrl15"
		"cntrl17" -> "cntrl21"
		"cntrl18" -> "cntrl17"
		"cntrl19" -> "cntrl17"
		"cntrl21" -> "cntrl20"
		}
		"coord10" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord10" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord10" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord2" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl10" [style=dotted,weight=0,arrowsize=0]
		"coord25" -> "cntrl10" [style=dotted,weight=0,arrowsize=0]
		"coord28" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord30" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord32" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
		"coord32" -> "cntrl20" [style=dotted,weight=0,arrowsize=0]
		"coord43" -> "cntrl20" [style=dotted,weight=0,arrowsize=0]
		}).";

        int  loopSize     = 16;
        auto loopSizeExpr = Expression::literal(loopSize);

        auto kgraph2 = lowerLinearLoop(kgraph1, loopSizeExpr, m_context);
        EXPECT_EQ(NormalizedSource(expected2), NormalizedSource(kgraph2.toDOT(true)));
    }

    TEST_F(KernelGraphTest, Translate01Tiled)
    {
        auto command  = std::make_shared<Command>();
        auto dataType = DataType::Int32;

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(dataType, 2, 1))); // B

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1))); // D = A * B

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(dataType, 2, 2))); // D

        auto kgraph0 = translate(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        std::string expected0 = R".(
	    digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{NA}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{NA}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="MacroTile{NA}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord17"[label="SubDimension{0, NA}(17)"];
		"coord18"[label="SubDimension{1, NA}(18)"];
		"coord19"[label="User{NA}(19)"];
		"coord20"[label="DestructTensorTile(20)",shape=box];
		"coord21"[label="Join(21)",shape=box];
		"coord22"[label="DataFlow(22)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord16"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord16"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord15" -> "coord20"
		"coord15" -> "coord22"
		"coord16" -> "coord15"
		"coord17" -> "coord21"
		"coord18" -> "coord21"
		"coord20" -> "coord17"
		"coord20" -> "coord18"
		"coord21" -> "coord19"
		"coord22" -> "coord19"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord4"->"coord11"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord17"->"coord18"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord17"->"coord18"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="StoreTiled(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
		"cntrl5" -> "cntrl4"
		"cntrl6" -> "cntrl10"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		"cntrl10" -> "cntrl9"
		}
		"coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
		"coord19" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
	    }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, Translate01Scalar)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Float, 0)));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Float, 1)));

        Operations::T_Execute execute;
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 1, 0)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Neg(4, 3)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 4)));

        command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

        auto kgraph0 = translate(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        std::string expected0 = R".(
                digraph {
                "coord1"[label="User{NA}(1)"];
                "coord2"[label="VGPR{NA}(2)"];
                "coord3"[label="DataFlow(3)",shape=box];
                "coord4"[label="User{NA}(4)"];
                "coord5"[label="VGPR{NA}(5)"];
                "coord6"[label="DataFlow(6)",shape=box];
                "coord7"[label="VGPR{NA}(7)"];
                "coord8"[label="DataFlow(8)",shape=box];
                "coord9"[label="VGPR{NA}(9)"];
                "coord10"[label="DataFlow(10)",shape=box];
                "coord11"[label="VGPR{NA}(11)"];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord1" -> "coord3"
                "coord2" -> "coord8"
                "coord3" -> "coord2"
                "coord4" -> "coord6"
                "coord5" -> "coord8"
                "coord6" -> "coord5"
                "coord7" -> "coord10"
                "coord7" -> "coord12"
                "coord8" -> "coord7"
                "coord9" -> "coord12"
                "coord10" -> "coord9"
                "coord12" -> "coord11"
                {
                rank=same
                "coord5"->"coord2"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord7"->"coord9"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadVGPR(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadVGPR(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(5), DataFlowTag(2))(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(7))(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(7), DataFlowTag(9))(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl8"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl7"
                "cntrl5" -> "cntrl4"
                "cntrl6" -> "cntrl10"
                "cntrl6" -> "cntrl12"
                "cntrl7" -> "cntrl6"
                "cntrl8" -> "cntrl6"
                "cntrl9" -> "cntrl13"
                "cntrl10" -> "cntrl9"
                "cntrl12" -> "cntrl11"
                "cntrl13" -> "cntrl11"
                }
                "coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
                "coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
                "coord4" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
                "coord5" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
                "coord7" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
                "coord9" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
                "coord11" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
             }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, LowerTensor)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 2))); // C
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Float, 3))); // alpha
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Float, 4))); // beta

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(5, 0, 1))); // A * B

        rocRoller::Operations::T_Execute execute;
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(6, 3, 5))); // alpha * (A * B)
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(7, 4, 2))); // beta * C
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(8, 6, 7))); // alpha * (A * B) + beta * C
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 8))); // D

        auto kgraph0 = translate(command);

        std::string expected0 = R".(
	    digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{NA}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{NA}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="User{NA}(15)"];
		"coord16"[label="SubDimension{0, CommandArgument(Load_Tiled_2_size_0)}(16)"];
		"coord17"[label="SubDimension{1, CommandArgument(Load_Tiled_2_size_1)}(17)"];
		"coord18"[label="MacroTile{NA}(18)"];
		"coord19"[label="Split(19)",shape=box];
		"coord20"[label="ConstructTensorTile(20)",shape=box];
		"coord21"[label="DataFlow(21)",shape=box];
		"coord22"[label="User{NA}(22)"];
		"coord23"[label="VGPR{NA}(23)"];
		"coord24"[label="DataFlow(24)",shape=box];
		"coord25"[label="User{NA}(25)"];
		"coord26"[label="VGPR{NA}(26)"];
		"coord27"[label="DataFlow(27)",shape=box];
		"coord28"[label="MacroTile{NA}(28)"];
		"coord29"[label="DataFlow(29)",shape=box];
		"coord30"[label="MacroTile{NA}(30)"];
		"coord31"[label="DataFlow(31)",shape=box];
		"coord32"[label="MacroTile{NA}(32)"];
		"coord33"[label="DataFlow(33)",shape=box];
		"coord34"[label="MacroTile{NA}(34)"];
		"coord35"[label="DataFlow(35)",shape=box];
		"coord36"[label="SubDimension{0, NA}(36)"];
		"coord37"[label="SubDimension{1, NA}(37)"];
		"coord38"[label="User{NA}(38)"];
		"coord39"[label="DestructTensorTile(39)",shape=box];
		"coord40"[label="Join(40)",shape=box];
		"coord41"[label="DataFlow(41)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord29"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord29"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord15" -> "coord19"
		"coord15" -> "coord21"
		"coord16" -> "coord20"
		"coord17" -> "coord20"
		"coord18" -> "coord33"
		"coord19" -> "coord16"
		"coord19" -> "coord17"
		"coord20" -> "coord18"
		"coord21" -> "coord18"
		"coord22" -> "coord24"
		"coord23" -> "coord31"
		"coord24" -> "coord23"
		"coord25" -> "coord27"
		"coord26" -> "coord33"
		"coord27" -> "coord26"
		"coord28" -> "coord31"
		"coord29" -> "coord28"
		"coord30" -> "coord35"
		"coord31" -> "coord30"
		"coord32" -> "coord35"
		"coord33" -> "coord32"
		"coord34" -> "coord39"
		"coord34" -> "coord41"
		"coord35" -> "coord34"
		"coord36" -> "coord40"
		"coord37" -> "coord40"
		"coord39" -> "coord36"
		"coord39" -> "coord37"
		"coord40" -> "coord38"
		"coord41" -> "coord38"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord16"->"coord17"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord16"->"coord17"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord4"->"coord11"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord23"->"coord28"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord26"->"coord18"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord30"->"coord32"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord36"->"coord37"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord36"->"coord37"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="LoadTiled(6)"];
		"cntrl7"[label="Body(7)",shape=box];
		"cntrl8"[label="LoadVGPR(8)"];
		"cntrl9"[label="Body(9)",shape=box];
		"cntrl10"[label="LoadVGPR(10)"];
		"cntrl11"[label="Body(11)",shape=box];
		"cntrl12"[label="TensorContraction(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="Assign VGPR Multiply(DataFlowTag(23), DataFlowTag(28))(15)"];
		"cntrl16"[label="Sequence(16)",shape=box];
		"cntrl17"[label="Sequence(17)",shape=box];
		"cntrl18"[label="Assign VGPR Multiply(DataFlowTag(26), DataFlowTag(18))(18)"];
		"cntrl19"[label="Sequence(19)",shape=box];
		"cntrl20"[label="Sequence(20)",shape=box];
		"cntrl21"[label="Assign VGPR Add(DataFlowTag(30), DataFlowTag(32))(21)"];
		"cntrl22"[label="Sequence(22)",shape=box];
		"cntrl23"[label="Sequence(23)",shape=box];
		"cntrl24"[label="StoreTiled(24)"];
		"cntrl25"[label="Sequence(25)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl1" -> "cntrl7"
		"cntrl1" -> "cntrl9"
		"cntrl1" -> "cntrl11"
		"cntrl2" -> "cntrl13"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl14"
		"cntrl5" -> "cntrl4"
		"cntrl6" -> "cntrl20"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl16"
		"cntrl9" -> "cntrl8"
		"cntrl10" -> "cntrl19"
		"cntrl11" -> "cntrl10"
		"cntrl12" -> "cntrl17"
		"cntrl13" -> "cntrl12"
		"cntrl14" -> "cntrl12"
		"cntrl15" -> "cntrl22"
		"cntrl16" -> "cntrl15"
		"cntrl17" -> "cntrl15"
		"cntrl18" -> "cntrl23"
		"cntrl19" -> "cntrl18"
		"cntrl20" -> "cntrl18"
		"cntrl21" -> "cntrl25"
		"cntrl22" -> "cntrl21"
		"cntrl23" -> "cntrl21"
		"cntrl25" -> "cntrl24"
		}
		"coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord18" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord22" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
		"coord23" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
		"coord25" -> "cntrl10" [style=dotted,weight=0,arrowsize=0]
		"coord26" -> "cntrl10" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord28" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord30" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord32" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
		"coord34" -> "cntrl21" [style=dotted,weight=0,arrowsize=0]
		"coord34" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord38" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
             }).";
        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        auto params = std::make_shared<CommandParameters>();

        // output macro tile size
        int mac_m = 128;
        int mac_n = 256;
        int mac_k = 16;

        // Wave tile size
        // V_MFMA_F32_32x32x8F32
        int wave_m = 32;
        int wave_n = 32;
        int wave_k = 8;
        int wave_b = 1;

        auto mac_tile_A = MacroTile({mac_m, mac_k},
                                    LayoutType::MATRIX_A,
                                    {wave_m, wave_n, wave_k, wave_b},
                                    MemoryType::LDS);
        auto mac_tile_B
            = MacroTile({mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_C = MacroTile(
            {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(4, mac_tile_A);
        params->setDimensionInfo(11, mac_tile_B);
        params->setDimensionInfo(18, mac_tile_C);
        params->setDimensionInfo(30, mac_tile_C);
        params->setDimensionInfo(32, mac_tile_C);
        params->setDimensionInfo(34, mac_tile_C);

        // Workgroup size
        uint wavefront_size   = 64;
        uint workgroup_size_x = 2 * wavefront_size;
        uint workgroup_size_y = 4;

        uint wavetile_per_wavefront_m = wavefront_size * mac_m / wave_m / workgroup_size_x;
        uint wavetile_per_wavefront_n = mac_n / wave_n / workgroup_size_y;

        params->setWaveTilesPerWavefront(wavetile_per_wavefront_m, wavetile_per_wavefront_n);

        kgraph0 = updateParameters(kgraph0, params);

        auto kgraph1 = lowerTile(kgraph0, params, m_context);

        auto kgraph_unrolled = unrollLoops(kgraph1, m_context);

        // Verify that loops have been unrolled
        auto unrolledForLoops = kgraph_unrolled.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(unrolledForLoops.size(), 7);

        auto kgraph_fused = fuseLoops(kgraph_unrolled);

        // Verify that loops have been fused
        auto fusedForLoops = kgraph_fused.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(fusedForLoops.size(), 3);

        // Verify that single iteration loops have been removed.
        auto kgraph_clean    = cleanLoops(kgraph_fused);
        auto cleanedForLoops = kgraph_clean.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(cleanedForLoops.size(), 1);

        // Verify that there is only a single StoreLDSTile node per K loop
        auto unrolled_kgraph_lds = addLDS(kgraph_unrolled, m_context);
        auto unrolledStoreLDS
            = unrolled_kgraph_lds.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(unrolledStoreLDS.size(), 4);

        unrolled_kgraph_lds = addLDS(kgraph_fused, m_context);
        auto fusedStoreLDS = unrolled_kgraph_lds.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(fusedStoreLDS.size(), 1);

        kgraph1 = addComputeIndexOperations(kgraph1);

        std::string expected1 = R".(
        digraph {
	    "coord1"[label="User{NA}(1)"];
	    "coord2"[label="User{NA}(2)"];
	    "coord3"[label="User{NA}(3)"];
	    "coord4"[label="User{NA}(4)"];
	    "coord5"[label="User{NA}(5)"];
	    "coord6"[label="VGPR{NA}(6)"];
	    "coord7"[label="DataFlow(7)",shape=box];
	    "coord8"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(8)"];
	    "coord9"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(9)"];
	    "coord10"[label="Split(10)",shape=box];
	    "coord11"[label="MacroTile{128,16}(11)"];
	    "coord12"[label="DataFlow(12)",shape=box];
	    "coord13"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(13)"];
	    "coord14"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(14)"];
	    "coord15"[label="Split(15)",shape=box];
	    "coord16"[label="MacroTile{16,256}(16)"];
	    "coord17"[label="DataFlow(17)",shape=box];
	    "coord18"[label="MacroTile{NA}(18)"];
	    "coord19"[label="DataFlow(19)",shape=box];
	    "coord20"[label="MacroTile{128,256}(20)"];
	    "coord21"[label="DataFlow(21)",shape=box];
	    "coord22"[label="VGPR{NA}(22)"];
	    "coord23"[label="DataFlow(23)",shape=box];
	    "coord24"[label="SubDimension{0, CommandArgument(Load_Tiled_2_size_0)}(24)"];
	    "coord25"[label="SubDimension{1, CommandArgument(Load_Tiled_2_size_1)}(25)"];
	    "coord26"[label="Split(26)",shape=box];
	    "coord27"[label="MacroTile{128,256}(27)"];
	    "coord28"[label="DataFlow(28)",shape=box];
	    "coord29"[label="MacroTile{128,256}(29)"];
	    "coord30"[label="DataFlow(30)",shape=box];
	    "coord31"[label="MacroTile{128,256}(31)"];
	    "coord32"[label="DataFlow(32)",shape=box];
	    "coord33"[label="SubDimension{0, NA}(33)"];
	    "coord34"[label="SubDimension{1, NA}(34)"];
	    "coord35"[label="User{NA}(35)"];
	    "coord36"[label="Join(36)",shape=box];
	    "coord37"[label="DataFlow(37)",shape=box];
	    "coord38"[label="MacroTileNumber{0, 1j}(38)"];
	    "coord39"[label="MacroTileNumber{1, 1j}(39)"];
	    "coord40"[label="MacroTileIndex{0, 128j}(40)"];
	    "coord41"[label="MacroTileIndex{1, 16j}(41)"];
	    "coord42"[label="Workgroup{0, NA}(42)"];
	    "coord44"[label="Tile(44)",shape=box];
	    "coord45"[label="Tile(45)",shape=box];
	    "coord46"[label="PassThrough(46)",shape=box];
	    "coord48"[label="Workitem{0, NA}(48)"];
	    "coord49"[label="WaveTile{256i}(49)"];
	    "coord50"[label="WaveTileNumber{0, 1j}(50)"];
	    "coord51"[label="WaveTileNumber{1, 1j}(51)"];
	    "coord52"[label="WaveTileIndex{0, 32j}(52)"];
	    "coord53"[label="WaveTileIndex{1, 8j}(53)"];
	    "coord54"[label="Tile(54)",shape=box];
	    "coord55"[label="Tile(55)",shape=box];
	    "coord56"[label="Wavefront{0, NA}(56)"];
	    "coord57"[label="Wavefront{1, NA}(57)"];
	    "coord58"[label="Wavefront{-1, NA}(58)"];
	    "coord59"[label="Lane{32j}(59)"];
	    "coord60"[label="VGPR{8j}(60)"];
	    "coord61"[label="Flatten(61)",shape=box];
	    "coord62"[label="Flatten(62)",shape=box];
	    "coord63"[label="BlockNumber{1j}(63)"];
	    "coord64"[label="BlockIndex{32j}(64)"];
	    "coord65"[label="Flatten(65)",shape=box];
	    "coord66"[label="WaveTilePerWorkGroup{0, 2j}(66)"];
	    "coord67"[label="WaveTilePerWorkGroup{1, 2j}(67)"];
	    "coord68"[label="Tile(68)",shape=box];
	    "coord69"[label="PassThrough(69)",shape=box];
	    "coord70"[label="Tile(70)",shape=box];
	    "coord71"[label="MacroTileNumber{0, 1j}(71)"];
	    "coord72"[label="MacroTileNumber{1, 1j}(72)"];
	    "coord73"[label="MacroTileIndex{0, 16j}(73)"];
	    "coord74"[label="MacroTileIndex{1, 256j}(74)"];
	    "coord76"[label="Workgroup{1, NA}(76)"];
	    "coord77"[label="Tile(77)",shape=box];
	    "coord78"[label="Tile(78)",shape=box];
	    "coord80"[label="PassThrough(80)",shape=box];
	    "coord81"[label="Workitem{0, NA}(81)"];
	    "coord82"[label="WaveTile{256i}(82)"];
	    "coord83"[label="WaveTileNumber{0, 1j}(83)"];
	    "coord84"[label="WaveTileNumber{1, 1j}(84)"];
	    "coord85"[label="WaveTileIndex{0, 8j}(85)"];
	    "coord86"[label="WaveTileIndex{1, 32j}(86)"];
	    "coord87"[label="Tile(87)",shape=box];
	    "coord88"[label="Tile(88)",shape=box];
	    "coord89"[label="Wavefront{0, NA}(89)"];
	    "coord90"[label="Wavefront{1, NA}(90)"];
	    "coord91"[label="Wavefront{-1, NA}(91)"];
	    "coord92"[label="Lane{32j}(92)"];
	    "coord93"[label="VGPR{8j}(93)"];
	    "coord94"[label="Flatten(94)",shape=box];
	    "coord95"[label="Flatten(95)",shape=box];
	    "coord96"[label="BlockNumber{1j}(96)"];
	    "coord97"[label="BlockIndex{32j}(97)"];
	    "coord98"[label="Flatten(98)",shape=box];
	    "coord99"[label="WaveTilePerWorkGroup{0, 2j}(99)"];
	    "coord100"[label="WaveTilePerWorkGroup{1, 2j}(100)"];
	    "coord101"[label="Tile(101)",shape=box];
	    "coord102"[label="PassThrough(102)",shape=box];
	    "coord103"[label="Tile(103)",shape=box];
	    "coord104"[label="MacroTileNumber{0, 1j}(104)"];
	    "coord105"[label="MacroTileNumber{1, 1j}(105)"];
	    "coord106"[label="MacroTileIndex{0, 128j}(106)"];
	    "coord107"[label="MacroTileIndex{1, 256j}(107)"];
	    "coord108"[label="Workgroup{0, NA}(108)"];
	    "coord109"[label="Workgroup{1, NA}(109)"];
	    "coord110"[label="Tile(110)",shape=box];
	    "coord111"[label="Tile(111)",shape=box];
	    "coord112"[label="PassThrough(112)",shape=box];
	    "coord113"[label="PassThrough(113)",shape=box];
	    "coord114"[label="Workitem{0, NA}(114)"];
	    "coord115"[label="WaveTile{1024i}(115)"];
	    "coord116"[label="WaveTileNumber{0, 1j}(116)"];
	    "coord117"[label="WaveTileNumber{1, 1j}(117)"];
	    "coord118"[label="WaveTileIndex{0, 32j}(118)"];
	    "coord119"[label="WaveTileIndex{1, 32j}(119)"];
	    "coord120"[label="Tile(120)",shape=box];
	    "coord121"[label="Tile(121)",shape=box];
	    "coord122"[label="Wavefront{0, NA}(122)"];
	    "coord123"[label="Wavefront{1, NA}(123)"];
	    "coord124"[label="Wavefront{-1, NA}(124)"];
	    "coord125"[label="Lane{32j}(125)"];
	    "coord126"[label="VGPR{32j}(126)"];
	    "coord127"[label="Flatten(127)",shape=box];
	    "coord128"[label="Flatten(128)",shape=box];
	    "coord129"[label="BlockNumber{1j}(129)"];
	    "coord130"[label="BlockIndex{32j}(130)"];
	    "coord131"[label="Flatten(131)",shape=box];
	    "coord132"[label="WaveTilePerWorkGroup{0, 2j}(132)"];
	    "coord133"[label="WaveTilePerWorkGroup{1, 2j}(133)"];
	    "coord134"[label="VGPRBlockNumber{8j}(134)"];
	    "coord135"[label="VGPRBlockIndex{4j}(135)"];
	    "coord136"[label="LANEBlockNumber{8j}(136)"];
	    "coord137"[label="LANEBlockIndex{4j}(137)"];
	    "coord138"[label="LinearBlock{64j}(138)"];
	    "coord139"[label="RowBlock{8j}(139)"];
	    "coord140"[label="ColBlock{8j}(140)"];
	    "coord141"[label="Flatten(141)",shape=box];
	    "coord142"[label="Tile(142)",shape=box];
	    "coord143"[label="Tile(143)",shape=box];
	    "coord144"[label="Flatten(144)",shape=box];
	    "coord145"[label="Tile(145)",shape=box];
	    "coord146"[label="Flatten(146)",shape=box];
	    "coord147"[label="Flatten(147)",shape=box];
	    "coord148"[label="Tile(148)",shape=box];
	    "coord149"[label="Tile(149)",shape=box];
	    "coord150"[label="MacroTileNumber{0, 1j}(150)"];
	    "coord151"[label="MacroTileNumber{1, 1j}(151)"];
	    "coord152"[label="MacroTileIndex{0, 128j}(152)"];
	    "coord153"[label="MacroTileIndex{1, 256j}(153)"];
	    "coord154"[label="Workgroup{0, NA}(154)"];
	    "coord155"[label="Workgroup{1, NA}(155)"];
	    "coord156"[label="Flatten(156)",shape=box];
	    "coord157"[label="Flatten(157)",shape=box];
	    "coord158"[label="PassThrough(158)",shape=box];
	    "coord159"[label="PassThrough(159)",shape=box];
	    "coord160"[label="Workitem{0, 1j}(160)"];
	    "coord161"[label="WaveTile{8192i}(161)"];
	    "coord162"[label="WaveTileNumber{0, 1j}(162)"];
	    "coord163"[label="WaveTileNumber{1, 1j}(163)"];
	    "coord164"[label="WaveTileIndex{0, 32j}(164)"];
	    "coord165"[label="WaveTileIndex{1, 32j}(165)"];
	    "coord166"[label="Join(166)",shape=box];
	    "coord167"[label="VGPRBlockNumber{8j}(167)"];
	    "coord168"[label="VGPRBlockIndex{4j}(168)"];
	    "coord169"[label="LANEBlockNumber{8j}(169)"];
	    "coord170"[label="LANEBlockIndex{4j}(170)"];
	    "coord171"[label="LinearBlock{64j}(171)"];
	    "coord172"[label="RowBlock{8j}(172)"];
	    "coord173"[label="ColBlock{8j}(173)"];
	    "coord174"[label="Flatten(174)",shape=box];
	    "coord175"[label="Flatten(175)",shape=box];
	    "coord176"[label="Wavefront{0, NA}(176)"];
	    "coord177"[label="Wavefront{1, NA}(177)"];
	    "coord178"[label="Wavefront{-1, NA}(178)"];
	    "coord179"[label="Tile(179)",shape=box];
	    "coord180"[label="Lane{32j}(180)"];
	    "coord181"[label="VGPR{32j}(181)"];
	    "coord182"[label="Tile(182)",shape=box];
	    "coord183"[label="Tile(183)",shape=box];
	    "coord184"[label="Flatten(184)",shape=box];
	    "coord185"[label="Tile(185)",shape=box];
	    "coord186"[label="WaveTilePerWorkGroup{0, 2j}(186)"];
	    "coord187"[label="WaveTilePerWorkGroup{1, 2j}(187)"];
	    "coord188"[label="Flatten(188)",shape=box];
	    "coord189"[label="Flatten(189)",shape=box];
	    "coord190"[label="Flatten(190)",shape=box];
	    "coord191"[label="Flatten(191)",shape=box];
	    "coord192"[label="Tile(192)",shape=box];
	    "coord193"[label="Linear{Divide(CommandArgument(Load_Tiled_0_size_1), 16j)}(193)"];
	    "coord194"[label="ForLoop{Divide(CommandArgument(Load_Tiled_0_size_1), 16j)}(194)"];
	    "coord195"[label="DataFlow(195)",shape=box];
	    "coord196"[label="PassThrough(196)",shape=box];
	    "coord197"[label="PassThrough(197)",shape=box];
	    "coord198"[label="Linear{2j}(198)"];
	    "coord199"[label="ForLoop{2j}(199)"];
	    "coord200"[label="DataFlow(200)",shape=box];
	    "coord201"[label="Linear{2j}(201)"];
	    "coord202"[label="ForLoop{2j}(202)"];
	    "coord203"[label="DataFlow(203)",shape=box];
	    "coord204"[label="PassThrough(204)",shape=box];
	    "coord205"[label="PassThrough(205)",shape=box];
	    "coord206"[label="PassThrough(206)",shape=box];
	    "coord207"[label="PassThrough(207)",shape=box];
	    "coord208"[label="PassThrough(208)",shape=box];
	    "coord209"[label="PassThrough(209)",shape=box];
	    "coord210"[label="PassThrough(210)",shape=box];
	    "coord211"[label="PassThrough(211)",shape=box];
	    "coord212"[label="Offset(212)",shape=box];
	    "coord213"[label="Stride(213)",shape=box];
	    "coord214"[label="Offset(214)",shape=box];
	    "coord215"[label="Stride(215)",shape=box];
	    "coord216"[label="Offset(216)",shape=box];
	    "coord217"[label="Stride(217)",shape=box];
	    "coord218"[label="Buffer(218)",shape=box];
	    "coord219"[label="Offset(219)",shape=box];
	    "coord220"[label="Stride(220)",shape=box];
	    "coord221"[label="Offset(221)",shape=box];
	    "coord222"[label="Stride(222)",shape=box];
	    "coord223"[label="Offset(223)",shape=box];
	    "coord224"[label="Stride(224)",shape=box];
	    "coord225"[label="Buffer(225)",shape=box];
	    "coord226"[label="Offset(226)",shape=box];
	    "coord227"[label="Stride(227)",shape=box];
	    "coord228"[label="Offset(228)",shape=box];
	    "coord229"[label="Stride(229)",shape=box];
	    "coord230"[label="Buffer(230)",shape=box];
	    "coord231"[label="Offset(231)",shape=box];
	    "coord232"[label="Stride(232)",shape=box];
	    "coord233"[label="Offset(233)",shape=box];
	    "coord234"[label="Stride(234)",shape=box];
	    "coord235"[label="Buffer(235)",shape=box];
	    "coord1" -> "coord10"
	    "coord1" -> "coord12"
	    "coord1" -> "coord212"
	    "coord1" -> "coord213"
	    "coord1" -> "coord214"
	    "coord1" -> "coord215"
	    "coord1" -> "coord216"
	    "coord1" -> "coord217"
	    "coord1" -> "coord218"
	    "coord2" -> "coord15"
	    "coord2" -> "coord17"
	    "coord2" -> "coord219"
	    "coord2" -> "coord220"
	    "coord2" -> "coord221"
	    "coord2" -> "coord222"
	    "coord2" -> "coord223"
	    "coord2" -> "coord224"
	    "coord2" -> "coord225"
	    "coord3" -> "coord26"
	    "coord3" -> "coord28"
	    "coord3" -> "coord226"
	    "coord3" -> "coord227"
	    "coord3" -> "coord228"
	    "coord3" -> "coord229"
	    "coord3" -> "coord230"
	    "coord4" -> "coord7"
	    "coord5" -> "coord23"
	    "coord6" -> "coord21"
	    "coord7" -> "coord6"
	    "coord8" -> "coord44"
	    "coord9" -> "coord45"
	    "coord10" -> "coord8"
	    "coord10" -> "coord9"
	    "coord11" -> "coord19"
	    "coord12" -> "coord11"
	    "coord13" -> "coord77"
	    "coord14" -> "coord78"
	    "coord15" -> "coord13"
	    "coord15" -> "coord14"
	    "coord16" -> "coord19"
	    "coord17" -> "coord16"
	    "coord18" -> "coord21"
	    "coord19" -> "coord18"
	    "coord20" -> "coord32"
	    "coord21" -> "coord20"
	    "coord22" -> "coord30"
	    "coord23" -> "coord22"
	    "coord24" -> "coord110"
	    "coord25" -> "coord111"
	    "coord26" -> "coord24"
	    "coord26" -> "coord25"
	    "coord27" -> "coord30"
	    "coord28" -> "coord27"
	    "coord29" -> "coord32"
	    "coord30" -> "coord29"
	    "coord31" -> "coord37"
	    "coord32" -> "coord31"
	    "coord33" -> "coord36"
	    "coord34" -> "coord36"
	    "coord36" -> "coord35"
	    "coord37" -> "coord35"
	    "coord38" -> "coord46"
	    "coord39" -> "coord196"
	    "coord40" -> "coord54"
	    "coord41" -> "coord55"
	    "coord44" -> "coord38"
	    "coord44" -> "coord40"
	    "coord45" -> "coord39"
	    "coord45" -> "coord41"
	    "coord46" -> "coord42"
	    "coord50" -> "coord70"
	    "coord52" -> "coord69"
	    "coord53" -> "coord68"
	    "coord54" -> "coord50"
	    "coord54" -> "coord52"
	    "coord55" -> "coord51"
	    "coord55" -> "coord53"
	    "coord56" -> "coord61"
	    "coord57" -> "coord61"
	    "coord58" -> "coord62"
	    "coord59" -> "coord62"
	    "coord61" -> "coord58"
	    "coord62" -> "coord48"
	    "coord63" -> "coord65"
	    "coord64" -> "coord65"
	    "coord65" -> "coord59"
	    "coord66" -> "coord204"
	    "coord67" -> "coord206"
	    "coord68" -> "coord63"
	    "coord68" -> "coord60"
	    "coord69" -> "coord64"
	    "coord70" -> "coord56"
	    "coord70" -> "coord66"
	    "coord71" -> "coord197"
	    "coord72" -> "coord80"
	    "coord73" -> "coord87"
	    "coord74" -> "coord88"
	    "coord77" -> "coord71"
	    "coord77" -> "coord73"
	    "coord78" -> "coord72"
	    "coord78" -> "coord74"
	    "coord80" -> "coord76"
	    "coord84" -> "coord103"
	    "coord85" -> "coord101"
	    "coord86" -> "coord102"
	    "coord87" -> "coord83"
	    "coord87" -> "coord85"
	    "coord88" -> "coord84"
	    "coord88" -> "coord86"
	    "coord89" -> "coord94"
	    "coord90" -> "coord94"
	    "coord91" -> "coord95"
	    "coord92" -> "coord95"
	    "coord94" -> "coord91"
	    "coord95" -> "coord81"
	    "coord96" -> "coord98"
	    "coord97" -> "coord98"
	    "coord98" -> "coord92"
	    "coord99" -> "coord205"
	    "coord100" -> "coord207"
	    "coord101" -> "coord96"
	    "coord101" -> "coord93"
	    "coord102" -> "coord97"
	    "coord103" -> "coord90"
	    "coord103" -> "coord100"
	    "coord104" -> "coord112"
	    "coord105" -> "coord113"
	    "coord106" -> "coord120"
	    "coord107" -> "coord121"
	    "coord110" -> "coord104"
	    "coord110" -> "coord106"
	    "coord111" -> "coord105"
	    "coord111" -> "coord107"
	    "coord112" -> "coord108"
	    "coord113" -> "coord109"
	    "coord116" -> "coord148"
	    "coord117" -> "coord149"
	    "coord118" -> "coord141"
	    "coord118" -> "coord142"
	    "coord119" -> "coord143"
	    "coord119" -> "coord141"
	    "coord120" -> "coord116"
	    "coord120" -> "coord118"
	    "coord121" -> "coord117"
	    "coord121" -> "coord119"
	    "coord122" -> "coord127"
	    "coord123" -> "coord127"
	    "coord124" -> "coord128"
	    "coord125" -> "coord128"
	    "coord127" -> "coord124"
	    "coord128" -> "coord114"
	    "coord129" -> "coord131"
	    "coord130" -> "coord131"
	    "coord131" -> "coord125"
	    "coord132" -> "coord208"
	    "coord133" -> "coord209"
	    "coord134" -> "coord146"
	    "coord135" -> "coord146"
	    "coord136" -> "coord147"
	    "coord137" -> "coord147"
	    "coord138" -> "coord145"
	    "coord139" -> "coord144"
	    "coord140" -> "coord144"
	    "coord141" -> "coord115"
	    "coord142" -> "coord139"
	    "coord142" -> "coord135"
	    "coord143" -> "coord140"
	    "coord143" -> "coord137"
	    "coord144" -> "coord138"
	    "coord145" -> "coord134"
	    "coord145" -> "coord136"
	    "coord146" -> "coord126"
	    "coord147" -> "coord125"
	    "coord148" -> "coord122"
	    "coord148" -> "coord132"
	    "coord149" -> "coord123"
	    "coord149" -> "coord133"
	    "coord150" -> "coord156"
	    "coord151" -> "coord157"
	    "coord152" -> "coord156"
	    "coord153" -> "coord157"
	    "coord154" -> "coord158"
	    "coord155" -> "coord159"
	    "coord156" -> "coord33"
	    "coord157" -> "coord34"
	    "coord158" -> "coord150"
	    "coord159" -> "coord151"
	    "coord160" -> "coord192"
	    "coord162" -> "coord174"
	    "coord163" -> "coord175"
	    "coord164" -> "coord166"
	    "coord164" -> "coord174"
	    "coord165" -> "coord166"
	    "coord165" -> "coord175"
	    "coord166" -> "coord161"
	    "coord167" -> "coord184"
	    "coord167" -> "coord231"
	    "coord167" -> "coord232"
	    "coord167" -> "coord235"
	    "coord168" -> "coord233"
	    "coord168" -> "coord234"
	    "coord168" -> "coord190"
	    "coord169" -> "coord184"
	    "coord170" -> "coord191"
	    "coord171" -> "coord185"
	    "coord172" -> "coord190"
	    "coord173" -> "coord191"
	    "coord174" -> "coord152"
	    "coord175" -> "coord153"
	    "coord176" -> "coord188"
	    "coord177" -> "coord189"
	    "coord178" -> "coord179"
	    "coord179" -> "coord176"
	    "coord179" -> "coord177"
	    "coord180" -> "coord183"
	    "coord181" -> "coord182"
	    "coord182" -> "coord167"
	    "coord182" -> "coord168"
	    "coord183" -> "coord169"
	    "coord183" -> "coord170"
	    "coord184" -> "coord171"
	    "coord185" -> "coord172"
	    "coord185" -> "coord173"
	    "coord186" -> "coord188"
	    "coord187" -> "coord189"
	    "coord188" -> "coord162"
	    "coord189" -> "coord163"
	    "coord190" -> "coord164"
	    "coord191" -> "coord165"
	    "coord192" -> "coord178"
	    "coord192" -> "coord180"
	    "coord193" -> "coord195"
	    "coord195" -> "coord194"
	    "coord196" -> "coord194"
	    "coord197" -> "coord194"
	    "coord198" -> "coord200"
	    "coord199" -> "coord210"
	    "coord200" -> "coord199"
	    "coord201" -> "coord203"
	    "coord202" -> "coord211"
	    "coord203" -> "coord202"
	    "coord204" -> "coord199"
	    "coord205" -> "coord199"
	    "coord206" -> "coord202"
	    "coord207" -> "coord202"
	    "coord208" -> "coord199"
	    "coord209" -> "coord202"
	    "coord210" -> "coord186"
	    "coord211" -> "coord187"
	    "coord212" -> "coord39"
	    "coord213" -> "coord39"
	    "coord214" -> "coord51"
	    "coord215" -> "coord51"
	    "coord216" -> "coord60"
	    "coord217" -> "coord60"
	    "coord218" -> "coord39"
	    "coord219" -> "coord71"
	    "coord220" -> "coord71"
	    "coord221" -> "coord83"
	    "coord222" -> "coord83"
	    "coord223" -> "coord93"
	    "coord224" -> "coord93"
	    "coord225" -> "coord71"
	    "coord226" -> "coord134"
	    "coord227" -> "coord134"
	    "coord228" -> "coord135"
	    "coord229" -> "coord135"
	    "coord230" -> "coord134"
	    "coord231" -> "coord35"
	    "coord232" -> "coord35"
	    "coord233" -> "coord35"
	    "coord234" -> "coord35"
	    "coord235" -> "coord35"
	    {
	    rank=same
	    "coord8"->"coord9"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord13"->"coord14"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord11"->"coord16"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord6"->"coord18"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord24"->"coord25"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord22"->"coord27"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord20"->"coord29"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord33"->"coord34"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord38"->"coord40"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord39"->"coord41"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord50"->"coord52"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord51"->"coord53"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord56"->"coord57"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord58"->"coord59"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord63"->"coord64"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord63"->"coord60"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord56"->"coord66"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord71"->"coord73"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord72"->"coord74"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord83"->"coord85"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord84"->"coord86"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord89"->"coord90"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord91"->"coord92"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord96"->"coord97"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord96"->"coord93"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord90"->"coord100"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord104"->"coord106"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord105"->"coord107"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord116"->"coord118"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord117"->"coord119"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord122"->"coord123"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord124"->"coord125"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord129"->"coord130"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord118"->"coord119"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord139"->"coord135"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord140"->"coord137"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord139"->"coord140"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord134"->"coord136"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord134"->"coord135"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord136"->"coord137"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord122"->"coord132"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord123"->"coord133"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord150"->"coord152"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord151"->"coord153"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord164"->"coord165"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord162"->"coord164"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord163"->"coord165"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord176"->"coord177"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord167"->"coord168"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord169"->"coord170"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord167"->"coord169"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord172"->"coord173"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord176"->"coord186"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord177"->"coord187"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord172"->"coord168"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord173"->"coord170"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord178"->"coord180"[style=invis]
	    rankdir=LR
	    }
	    subgraph clusterCF {label = "Control Graph";
	    "cntrl1"[label="Kernel(1)"];
	    "cntrl2"[label="LoadVGPR(2)"];
	    "cntrl3"[label="Body(3)",shape=box];
	    "cntrl4"[label="LoadTiled(4)"];
	    "cntrl6"[label="LoadTiled(6)"];
	    "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(6), DataFlowTag(18))(11)"];
	    "cntrl12"[label="Sequence(12)",shape=box];
	    "cntrl14"[label="LoadVGPR(14)"];
	    "cntrl15"[label="Body(15)",shape=box];
	    "cntrl16"[label="LoadTiled(16)"];
	    "cntrl18"[label="Assign VGPR Multiply(DataFlowTag(22), DataFlowTag(27))(18)"];
	    "cntrl19"[label="Sequence(19)",shape=box];
	    "cntrl21"[label="Assign VGPR Add(DataFlowTag(20), DataFlowTag(29))(21)"];
	    "cntrl22"[label="Sequence(22)",shape=box];
	    "cntrl23"[label="Sequence(23)",shape=box];
	    "cntrl24"[label="StoreTiled(24)"];
	    "cntrl26"[label="ForLoopOp KLoop: LessThan(DataFlowTag(193), Divide(CommandArgument(Load_Tiled_0_size_1), 16j))(26)"];
	    "cntrl27"[label="Assign SGPR 0l(27)"];
	    "cntrl28"[label="Assign SGPR Add(DataFlowTag(193), 1j)(28)"];
	    "cntrl29"[label="Initialize(29)",shape=box];
	    "cntrl30"[label="ForLoopIncrement(30)",shape=box];
	    "cntrl31"[label="Multiply(31)"];
	    "cntrl32"[label="Assign ACCVGPR 0.00000f(32)"];
	    "cntrl34"[label="Body(34)",shape=box];
	    "cntrl35"[label="Body(35)",shape=box];
	    "cntrl36"[label="Body(36)",shape=box];
	    "cntrl37"[label="ForLoopOp XLoop: LessThan(DataFlowTag(198), 2j)(37)"];
	    "cntrl38"[label="Assign SGPR 0j(38)"];
	    "cntrl39"[label="Assign SGPR Add(DataFlowTag(198), 1j)(39)"];
	    "cntrl40"[label="Initialize(40)",shape=box];
	    "cntrl41"[label="ForLoopIncrement(41)",shape=box];
	    "cntrl42"[label="ForLoopOp YLoop: LessThan(DataFlowTag(201), 2j)(42)"];
	    "cntrl43"[label="Assign SGPR 0j(43)"];
	    "cntrl44"[label="Assign SGPR Add(DataFlowTag(201), 1j)(44)"];
	    "cntrl45"[label="Initialize(45)",shape=box];
	    "cntrl46"[label="ForLoopIncrement(46)",shape=box];
	    "cntrl47"[label="Body(47)",shape=box];
	    "cntrl48"[label="Body(48)",shape=box];
	    "cntrl49"[label="Body(49)",shape=box];
	    "cntrl50"[label="Scope(50)"];
	    "cntrl51"[label="Sequence(51)",shape=box];
	    "cntrl52"[label="Sequence(52)",shape=box];
	    "cntrl53"[label="ComputeIndex(53)"];
	    "cntrl54"[label="ComputeIndex(54)"];
	    "cntrl55"[label="ComputeIndex(55)"];
	    "cntrl56"[label="Sequence(56)",shape=box];
	    "cntrl57"[label="Sequence(57)",shape=box];
	    "cntrl58"[label="Assign VGPR Add(DataFlowTag(212), DataFlowTag(213))(58)"];
	    "cntrl59"[label="Body(59)",shape=box];
	    "cntrl60"[label="Sequence(60)",shape=box];
	    "cntrl61"[label="ForLoopIncrement(61)",shape=box];
	    "cntrl62"[label="ComputeIndex(62)"];
	    "cntrl63"[label="ComputeIndex(63)"];
	    "cntrl64"[label="ComputeIndex(64)"];
	    "cntrl65"[label="Sequence(65)",shape=box];
	    "cntrl66"[label="Sequence(66)",shape=box];
	    "cntrl67"[label="Assign VGPR Add(DataFlowTag(219), DataFlowTag(220))(67)"];
	    "cntrl68"[label="Body(68)",shape=box];
	    "cntrl69"[label="Sequence(69)",shape=box];
	    "cntrl70"[label="ForLoopIncrement(70)",shape=box];
	    "cntrl71"[label="Scope(71)"];
	    "cntrl72"[label="Body(72)",shape=box];
	    "cntrl73"[label="Sequence(73)",shape=box];
	    "cntrl74"[label="ComputeIndex(74)"];
	    "cntrl75"[label="ComputeIndex(75)"];
	    "cntrl76"[label="Body(76)",shape=box];
	    "cntrl77"[label="Sequence(77)",shape=box];
	    "cntrl78"[label="Sequence(78)",shape=box];
	    "cntrl79"[label="Scope(79)"];
	    "cntrl80"[label="Sequence(80)",shape=box];
	    "cntrl81"[label="ComputeIndex(81)"];
	    "cntrl82"[label="ComputeIndex(82)"];
	    "cntrl83"[label="Body(83)",shape=box];
	    "cntrl84"[label="Sequence(84)",shape=box];
	    "cntrl85"[label="Sequence(85)",shape=box];
	    "cntrl1" -> "cntrl3"
	    "cntrl1" -> "cntrl15"
	    "cntrl1" -> "cntrl49"
	    "cntrl2" -> "cntrl12"
	    "cntrl3" -> "cntrl2"
	    "cntrl11" -> "cntrl22"
	    "cntrl12" -> "cntrl37"
	    "cntrl14" -> "cntrl19"
	    "cntrl15" -> "cntrl14"
	    "cntrl18" -> "cntrl23"
	    "cntrl19" -> "cntrl37"
	    "cntrl21" -> "cntrl80"
	    "cntrl22" -> "cntrl21"
	    "cntrl23" -> "cntrl21"
	    "cntrl26" -> "cntrl29"
	    "cntrl26" -> "cntrl30"
	    "cntrl26" -> "cntrl34"
	    "cntrl26" -> "cntrl61"
	    "cntrl26" -> "cntrl70"
	    "cntrl29" -> "cntrl27"
	    "cntrl30" -> "cntrl28"
	    "cntrl31" -> "cntrl35"
	    "cntrl31" -> "cntrl36"
	    "cntrl32" -> "cntrl51"
	    "cntrl34" -> "cntrl31"
	    "cntrl35" -> "cntrl4"
	    "cntrl36" -> "cntrl6"
	    "cntrl37" -> "cntrl40"
	    "cntrl37" -> "cntrl41"
	    "cntrl37" -> "cntrl48"
	    "cntrl40" -> "cntrl38"
	    "cntrl41" -> "cntrl39"
	    "cntrl42" -> "cntrl45"
	    "cntrl42" -> "cntrl46"
	    "cntrl42" -> "cntrl47"
	    "cntrl42" -> "cntrl72"
	    "cntrl45" -> "cntrl43"
	    "cntrl46" -> "cntrl44"
	    "cntrl47" -> "cntrl32"
	    "cntrl48" -> "cntrl42"
	    "cntrl49" -> "cntrl37"
	    "cntrl50" -> "cntrl52"
	    "cntrl50" -> "cntrl59"
	    "cntrl50" -> "cntrl68"
	    "cntrl51" -> "cntrl50"
	    "cntrl52" -> "cntrl11"
	    "cntrl53" -> "cntrl56"
	    "cntrl54" -> "cntrl57"
	    "cntrl55" -> "cntrl60"
	    "cntrl56" -> "cntrl54"
	    "cntrl57" -> "cntrl55"
	    "cntrl59" -> "cntrl53"
	    "cntrl60" -> "cntrl26"
	    "cntrl61" -> "cntrl58"
	    "cntrl62" -> "cntrl65"
	    "cntrl63" -> "cntrl66"
	    "cntrl64" -> "cntrl69"
	    "cntrl65" -> "cntrl63"
	    "cntrl66" -> "cntrl64"
	    "cntrl68" -> "cntrl62"
	    "cntrl69" -> "cntrl26"
	    "cntrl70" -> "cntrl67"
	    "cntrl71" -> "cntrl73"
	    "cntrl71" -> "cntrl76"
	    "cntrl72" -> "cntrl71"
	    "cntrl73" -> "cntrl18"
	    "cntrl74" -> "cntrl77"
	    "cntrl75" -> "cntrl78"
	    "cntrl76" -> "cntrl74"
	    "cntrl77" -> "cntrl75"
	    "cntrl78" -> "cntrl16"
	    "cntrl79" -> "cntrl83"
	    "cntrl80" -> "cntrl79"
	    "cntrl81" -> "cntrl84"
	    "cntrl82" -> "cntrl85"
	    "cntrl83" -> "cntrl81"
	    "cntrl84" -> "cntrl82"
	    "cntrl85" -> "cntrl24"
	    }
	    "coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord6" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord9" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord38" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord42" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord49" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord50" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord66" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord67" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord213" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord214" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord215" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord216" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord218" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord14" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord16" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord72" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord76" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord82" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord84" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord99" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord100" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord219" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord220" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord221" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord222" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord223" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord224" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
	    "coord16" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
	    "coord18" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
	    "coord20" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
	    "coord5" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
	    "coord22" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord24" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord25" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord27" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord104" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord105" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord108" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord109" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord115" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord116" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord117" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord126" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord132" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord133" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord227" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord228" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord229" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord230" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord29" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord31" -> "cntrl21" [style=dotted,weight=0,arrowsize=0]
	    "coord31" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord35" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord150" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord151" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord154" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord155" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord161" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord167" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord168" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord181" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord186" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord187" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord231" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord232" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord233" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord234" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord193" -> "cntrl26" [style=dotted,weight=0,arrowsize=0]
	    "coord193" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord193" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord16" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord18" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord49" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord82" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord18" -> "cntrl32" [style=dotted,weight=0,arrowsize=0]
	    "coord198" -> "cntrl37" [style=dotted,weight=0,arrowsize=0]
	    "coord198" -> "cntrl38" [style=dotted,weight=0,arrowsize=0]
	    "coord198" -> "cntrl39" [style=dotted,weight=0,arrowsize=0]
	    "coord201" -> "cntrl42" [style=dotted,weight=0,arrowsize=0]
	    "coord201" -> "cntrl43" [style=dotted,weight=0,arrowsize=0]
	    "coord201" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord213" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord218" -> "cntrl53" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord214" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord215" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord218" -> "cntrl54" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord214" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord216" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord218" -> "cntrl55" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl58" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord219" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord220" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord219" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord221" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord222" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord221" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord223" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord224" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl64" [style=dotted,weight=0,arrowsize=0]
	    "coord219" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl74" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl74" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl74" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl74" [style=dotted,weight=0,arrowsize=0]
	    "coord227" -> "cntrl74" [style=dotted,weight=0,arrowsize=0]
	    "coord230" -> "cntrl74" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord228" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord229" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord230" -> "cntrl75" [style=dotted,weight=0,arrowsize=0]
	    "coord35" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord167" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord168" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord231" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord232" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord35" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	    "coord167" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	    "coord168" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	    "coord231" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	    "coord233" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	    "coord234" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl82" [style=dotted,weight=0,arrowsize=0]
	}).";

        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));

        auto        kgraph2   = addDeallocate(kgraph1);
        std::string expected2 = R".(
           digraph {
		"1"[label="Kernel(1)"];
		"2"[label="LoadVGPR(2)"];
		"3"[label="Body(3)",shape=box];
		"4"[label="LoadTiled(4)"];
		"6"[label="LoadTiled(6)"];
		"11"[label="Assign VGPR Multiply(DataFlowTag(6), DataFlowTag(18))(11)"];
		"12"[label="Sequence(12)",shape=box];
		"14"[label="LoadVGPR(14)"];
		"15"[label="Body(15)",shape=box];
		"16"[label="LoadTiled(16)"];
		"18"[label="Assign VGPR Multiply(DataFlowTag(22), DataFlowTag(27))(18)"];
		"19"[label="Sequence(19)",shape=box];
		"21"[label="Assign VGPR Add(DataFlowTag(20), DataFlowTag(29))(21)"];
		"24"[label="StoreTiled(24)"];
		"26"[label="ForLoopOp KLoop: LessThan(DataFlowTag(193), Divide(CommandArgument(Load_Tiled_0_size_1), 16j))(26)"];
		"27"[label="Assign SGPR 0l(27)"];
		"28"[label="Assign SGPR Add(DataFlowTag(193), 1j)(28)"];
		"29"[label="Initialize(29)",shape=box];
		"30"[label="ForLoopIncrement(30)",shape=box];
		"31"[label="Multiply(31)"];
		"32"[label="Assign ACCVGPR 0.00000f(32)"];
		"34"[label="Body(34)",shape=box];
		"35"[label="Body(35)",shape=box];
		"36"[label="Body(36)",shape=box];
		"37"[label="ForLoopOp XLoop: LessThan(DataFlowTag(198), 2j)(37)"];
		"38"[label="Assign SGPR 0j(38)"];
		"39"[label="Assign SGPR Add(DataFlowTag(198), 1j)(39)"];
		"40"[label="Initialize(40)",shape=box];
		"41"[label="ForLoopIncrement(41)",shape=box];
		"42"[label="ForLoopOp YLoop: LessThan(DataFlowTag(201), 2j)(42)"];
		"43"[label="Assign SGPR 0j(43)"];
		"44"[label="Assign SGPR Add(DataFlowTag(201), 1j)(44)"];
		"45"[label="Initialize(45)",shape=box];
		"46"[label="ForLoopIncrement(46)",shape=box];
		"47"[label="Body(47)",shape=box];
		"48"[label="Body(48)",shape=box];
		"49"[label="Body(49)",shape=box];
		"50"[label="Scope(50)"];
		"51"[label="Sequence(51)",shape=box];
		"52"[label="Sequence(52)",shape=box];
		"53"[label="ComputeIndex(53)"];
		"54"[label="ComputeIndex(54)"];
		"55"[label="ComputeIndex(55)"];
		"56"[label="Sequence(56)",shape=box];
		"57"[label="Sequence(57)",shape=box];
		"58"[label="Assign VGPR Add(DataFlowTag(212), DataFlowTag(213))(58)"];
		"59"[label="Body(59)",shape=box];
		"60"[label="Sequence(60)",shape=box];
		"61"[label="ForLoopIncrement(61)",shape=box];
		"62"[label="ComputeIndex(62)"];
		"63"[label="ComputeIndex(63)"];
		"64"[label="ComputeIndex(64)"];
		"65"[label="Sequence(65)",shape=box];
		"66"[label="Sequence(66)",shape=box];
		"67"[label="Assign VGPR Add(DataFlowTag(219), DataFlowTag(220))(67)"];
		"68"[label="Body(68)",shape=box];
		"69"[label="Sequence(69)",shape=box];
		"70"[label="ForLoopIncrement(70)",shape=box];
		"71"[label="Scope(71)"];
		"72"[label="Body(72)",shape=box];
		"73"[label="Sequence(73)",shape=box];
		"74"[label="ComputeIndex(74)"];
		"75"[label="ComputeIndex(75)"];
		"76"[label="Body(76)",shape=box];
		"77"[label="Sequence(77)",shape=box];
		"78"[label="Sequence(78)",shape=box];
		"79"[label="Scope(79)"];
		"81"[label="ComputeIndex(81)"];
		"82"[label="ComputeIndex(82)"];
		"83"[label="Body(83)",shape=box];
		"84"[label="Sequence(84)",shape=box];
		"85"[label="Sequence(85)",shape=box];
		"86"[label="Deallocate(86)"];
		"88"[label="Deallocate(88)"];
		"90"[label="Deallocate(90)"];
		"91"[label="Sequence(91)",shape=box];
		"92"[label="Sequence(92)",shape=box];
		"93"[label="Deallocate(93)"];
		"94"[label="Sequence(94)",shape=box];
		"95"[label="Sequence(95)",shape=box];
		"96"[label="Deallocate(96)"];
		"98"[label="Sequence(98)",shape=box];
		"99"[label="Deallocate(99)"];
		"100"[label="Sequence(100)",shape=box];
		"101"[label="Sequence(101)",shape=box];
		"102"[label="Deallocate(102)"];
		"103"[label="Sequence(103)",shape=box];
		"104"[label="Sequence(104)",shape=box];
		"105"[label="Deallocate(105)"];
		"106"[label="Sequence(106)",shape=box];
		"107"[label="Sequence(107)",shape=box];
		"108"[label="Deallocate(108)"];
		"109"[label="Sequence(109)",shape=box];
		"1" -> "3"
		"1" -> "15"
		"1" -> "49"
		"2" -> "12"
		"3" -> "2"
		"11" -> "94"
		"12" -> "37"
		"14" -> "19"
		"15" -> "14"
		"18" -> "103"
		"19" -> "37"
		"21" -> "106"
		"24" -> "109"
		"26" -> "29"
		"26" -> "30"
		"26" -> "34"
		"26" -> "61"
		"26" -> "70"
		"29" -> "27"
		"30" -> "28"
		"31" -> "35"
		"31" -> "36"
		"31" -> "91"
		"32" -> "51"
		"34" -> "31"
		"35" -> "4"
		"36" -> "6"
		"37" -> "40"
		"37" -> "41"
		"37" -> "48"
		"40" -> "38"
		"41" -> "39"
		"42" -> "45"
		"42" -> "46"
		"42" -> "47"
		"42" -> "72"
		"42" -> "100"
		"45" -> "43"
		"46" -> "44"
		"47" -> "32"
		"48" -> "42"
		"49" -> "37"
		"50" -> "52"
		"50" -> "59"
		"50" -> "68"
		"51" -> "50"
		"52" -> "11"
		"53" -> "56"
		"54" -> "57"
		"55" -> "60"
		"56" -> "54"
		"57" -> "55"
		"59" -> "53"
		"60" -> "26"
		"61" -> "58"
		"62" -> "65"
		"63" -> "66"
		"64" -> "69"
		"65" -> "63"
		"66" -> "64"
		"68" -> "62"
		"69" -> "26"
		"70" -> "67"
		"71" -> "73"
		"71" -> "76"
		"72" -> "71"
		"73" -> "18"
		"74" -> "77"
		"75" -> "78"
		"76" -> "74"
		"77" -> "75"
		"78" -> "16"
		"79" -> "83"
		"81" -> "84"
		"82" -> "85"
		"83" -> "81"
		"84" -> "82"
		"85" -> "24"
		"90" -> "92"
		"91" -> "90"
		"92" -> "88"
		"93" -> "95"
		"94" -> "93"
		"95" -> "21"
		"96" -> "98"
		"98" -> "79"
		"99" -> "101"
		"100" -> "99"
		"101" -> "86"
		"102" -> "104"
		"103" -> "102"
		"104" -> "21"
		"105" -> "107"
		"106" -> "105"
		"107" -> "96"
		"109" -> "108"
	     }).";

        EXPECT_EQ(NormalizedSource(expected2), NormalizedSource(kgraph2.control.toDOT("", true)));

        mac_tile_A = MacroTile({mac_m, mac_k},
                               LayoutType::MATRIX_A,
                               {wave_m, wave_n, wave_k, wave_b},
                               MemoryType::LDS);
        params->setDimensionInfo(4, mac_tile_A);

        auto kgraph_lds = updateParameters(kgraph0, params);

        auto kgraph_lds_lower = lowerTile(kgraph_lds, params, m_context);
        kgraph_lds_lower      = addLDS(kgraph_lds_lower, m_context);
        kgraph_lds_lower      = addComputeIndexOperations(kgraph_lds_lower);

        std::string expected_lds = R".(
	digraph {
	    "coord1"[label="User{NA}(1)"];
	    "coord2"[label="User{NA}(2)"];
	    "coord3"[label="User{NA}(3)"];
	    "coord4"[label="User{NA}(4)"];
	    "coord5"[label="User{NA}(5)"];
	    "coord6"[label="VGPR{NA}(6)"];
	    "coord7"[label="DataFlow(7)",shape=box];
	    "coord8"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(8)"];
	    "coord9"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(9)"];
	    "coord10"[label="Split(10)",shape=box];
	    "coord11"[label="MacroTile{128,16}(11)"];
	    "coord13"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(13)"];
	    "coord14"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(14)"];
	    "coord15"[label="Split(15)",shape=box];
	    "coord16"[label="MacroTile{16,256}(16)"];
	    "coord17"[label="DataFlow(17)",shape=box];
	    "coord18"[label="MacroTile{NA}(18)"];
	    "coord19"[label="DataFlow(19)",shape=box];
	    "coord20"[label="MacroTile{128,256}(20)"];
	    "coord21"[label="DataFlow(21)",shape=box];
	    "coord22"[label="VGPR{NA}(22)"];
	    "coord23"[label="DataFlow(23)",shape=box];
	    "coord24"[label="SubDimension{0, CommandArgument(Load_Tiled_2_size_0)}(24)"];
	    "coord25"[label="SubDimension{1, CommandArgument(Load_Tiled_2_size_1)}(25)"];
	    "coord26"[label="Split(26)",shape=box];
	    "coord27"[label="MacroTile{128,256}(27)"];
	    "coord28"[label="DataFlow(28)",shape=box];
	    "coord29"[label="MacroTile{128,256}(29)"];
	    "coord30"[label="DataFlow(30)",shape=box];
	    "coord31"[label="MacroTile{128,256}(31)"];
	    "coord32"[label="DataFlow(32)",shape=box];
	    "coord33"[label="SubDimension{0, NA}(33)"];
	    "coord34"[label="SubDimension{1, NA}(34)"];
	    "coord35"[label="User{NA}(35)"];
	    "coord36"[label="Join(36)",shape=box];
	    "coord37"[label="DataFlow(37)",shape=box];
	    "coord40"[label="MacroTileIndex{0, 128j}(40)"];
	    "coord41"[label="MacroTileIndex{1, 16j}(41)"];
	    "coord48"[label="Workitem{0, NA}(48)"];
	    "coord49"[label="WaveTile{256i}(49)"];
	    "coord50"[label="WaveTileNumber{0, 1j}(50)"];
	    "coord51"[label="WaveTileNumber{1, 1j}(51)"];
	    "coord52"[label="WaveTileIndex{0, 32j}(52)"];
	    "coord53"[label="WaveTileIndex{1, 8j}(53)"];
	    "coord54"[label="Tile(54)",shape=box];
	    "coord55"[label="Tile(55)",shape=box];
	    "coord56"[label="Wavefront{0, NA}(56)"];
	    "coord57"[label="Wavefront{1, NA}(57)"];
	    "coord58"[label="Wavefront{-1, NA}(58)"];
	    "coord59"[label="Lane{32j}(59)"];
	    "coord60"[label="VGPR{8j}(60)"];
	    "coord61"[label="Flatten(61)",shape=box];
	    "coord62"[label="Flatten(62)",shape=box];
	    "coord63"[label="BlockNumber{1j}(63)"];
	    "coord64"[label="BlockIndex{32j}(64)"];
	    "coord65"[label="Flatten(65)",shape=box];
	    "coord66"[label="WaveTilePerWorkGroup{0, 2j}(66)"];
	    "coord67"[label="WaveTilePerWorkGroup{1, 2j}(67)"];
	    "coord68"[label="Tile(68)",shape=box];
	    "coord69"[label="PassThrough(69)",shape=box];
	    "coord70"[label="Tile(70)",shape=box];
	    "coord71"[label="MacroTileNumber{0, 1j}(71)"];
	    "coord72"[label="MacroTileNumber{1, 1j}(72)"];
	    "coord73"[label="MacroTileIndex{0, 16j}(73)"];
	    "coord74"[label="MacroTileIndex{1, 256j}(74)"];
	    "coord76"[label="Workgroup{1, NA}(76)"];
	    "coord77"[label="Tile(77)",shape=box];
	    "coord78"[label="Tile(78)",shape=box];
	    "coord80"[label="PassThrough(80)",shape=box];
	    "coord81"[label="Workitem{0, NA}(81)"];
	    "coord82"[label="WaveTile{256i}(82)"];
	    "coord83"[label="WaveTileNumber{0, 1j}(83)"];
	    "coord84"[label="WaveTileNumber{1, 1j}(84)"];
	    "coord85"[label="WaveTileIndex{0, 8j}(85)"];
	    "coord86"[label="WaveTileIndex{1, 32j}(86)"];
	    "coord87"[label="Tile(87)",shape=box];
	    "coord88"[label="Tile(88)",shape=box];
	    "coord89"[label="Wavefront{0, NA}(89)"];
	    "coord90"[label="Wavefront{1, NA}(90)"];
	    "coord91"[label="Wavefront{-1, NA}(91)"];
	    "coord92"[label="Lane{32j}(92)"];
	    "coord93"[label="VGPR{8j}(93)"];
	    "coord94"[label="Flatten(94)",shape=box];
	    "coord95"[label="Flatten(95)",shape=box];
	    "coord96"[label="BlockNumber{1j}(96)"];
	    "coord97"[label="BlockIndex{32j}(97)"];
	    "coord98"[label="Flatten(98)",shape=box];
	    "coord99"[label="WaveTilePerWorkGroup{0, 2j}(99)"];
	    "coord100"[label="WaveTilePerWorkGroup{1, 2j}(100)"];
	    "coord101"[label="Tile(101)",shape=box];
	    "coord102"[label="PassThrough(102)",shape=box];
	    "coord103"[label="Tile(103)",shape=box];
	    "coord104"[label="MacroTileNumber{0, 1j}(104)"];
	    "coord105"[label="MacroTileNumber{1, 1j}(105)"];
	    "coord106"[label="MacroTileIndex{0, 128j}(106)"];
	    "coord107"[label="MacroTileIndex{1, 256j}(107)"];
	    "coord108"[label="Workgroup{0, NA}(108)"];
	    "coord109"[label="Workgroup{1, NA}(109)"];
	    "coord110"[label="Tile(110)",shape=box];
	    "coord111"[label="Tile(111)",shape=box];
	    "coord112"[label="PassThrough(112)",shape=box];
	    "coord113"[label="PassThrough(113)",shape=box];
	    "coord114"[label="Workitem{0, NA}(114)"];
	    "coord115"[label="WaveTile{1024i}(115)"];
	    "coord116"[label="WaveTileNumber{0, 1j}(116)"];
	    "coord117"[label="WaveTileNumber{1, 1j}(117)"];
	    "coord118"[label="WaveTileIndex{0, 32j}(118)"];
	    "coord119"[label="WaveTileIndex{1, 32j}(119)"];
	    "coord120"[label="Tile(120)",shape=box];
	    "coord121"[label="Tile(121)",shape=box];
	    "coord122"[label="Wavefront{0, NA}(122)"];
	    "coord123"[label="Wavefront{1, NA}(123)"];
	    "coord124"[label="Wavefront{-1, NA}(124)"];
	    "coord125"[label="Lane{32j}(125)"];
	    "coord126"[label="VGPR{32j}(126)"];
	    "coord127"[label="Flatten(127)",shape=box];
	    "coord128"[label="Flatten(128)",shape=box];
	    "coord129"[label="BlockNumber{1j}(129)"];
	    "coord130"[label="BlockIndex{32j}(130)"];
	    "coord131"[label="Flatten(131)",shape=box];
	    "coord132"[label="WaveTilePerWorkGroup{0, 2j}(132)"];
	    "coord133"[label="WaveTilePerWorkGroup{1, 2j}(133)"];
	    "coord134"[label="VGPRBlockNumber{8j}(134)"];
	    "coord135"[label="VGPRBlockIndex{4j}(135)"];
	    "coord136"[label="LANEBlockNumber{8j}(136)"];
	    "coord137"[label="LANEBlockIndex{4j}(137)"];
	    "coord138"[label="LinearBlock{64j}(138)"];
	    "coord139"[label="RowBlock{8j}(139)"];
	    "coord140"[label="ColBlock{8j}(140)"];
	    "coord141"[label="Flatten(141)",shape=box];
	    "coord142"[label="Tile(142)",shape=box];
	    "coord143"[label="Tile(143)",shape=box];
	    "coord144"[label="Flatten(144)",shape=box];
	    "coord145"[label="Tile(145)",shape=box];
	    "coord146"[label="Flatten(146)",shape=box];
	    "coord147"[label="Flatten(147)",shape=box];
	    "coord148"[label="Tile(148)",shape=box];
	    "coord149"[label="Tile(149)",shape=box];
	    "coord150"[label="MacroTileNumber{0, 1j}(150)"];
	    "coord151"[label="MacroTileNumber{1, 1j}(151)"];
	    "coord152"[label="MacroTileIndex{0, 128j}(152)"];
	    "coord153"[label="MacroTileIndex{1, 256j}(153)"];
	    "coord154"[label="Workgroup{0, NA}(154)"];
	    "coord155"[label="Workgroup{1, NA}(155)"];
	    "coord156"[label="Flatten(156)",shape=box];
	    "coord157"[label="Flatten(157)",shape=box];
	    "coord158"[label="PassThrough(158)",shape=box];
	    "coord159"[label="PassThrough(159)",shape=box];
	    "coord160"[label="Workitem{0, 1j}(160)"];
	    "coord161"[label="WaveTile{8192i}(161)"];
	    "coord162"[label="WaveTileNumber{0, 1j}(162)"];
	    "coord163"[label="WaveTileNumber{1, 1j}(163)"];
	    "coord164"[label="WaveTileIndex{0, 32j}(164)"];
	    "coord165"[label="WaveTileIndex{1, 32j}(165)"];
	    "coord166"[label="Join(166)",shape=box];
	    "coord167"[label="VGPRBlockNumber{8j}(167)"];
	    "coord168"[label="VGPRBlockIndex{4j}(168)"];
	    "coord169"[label="LANEBlockNumber{8j}(169)"];
	    "coord170"[label="LANEBlockIndex{4j}(170)"];
	    "coord171"[label="LinearBlock{64j}(171)"];
	    "coord172"[label="RowBlock{8j}(172)"];
	    "coord173"[label="ColBlock{8j}(173)"];
	    "coord174"[label="Flatten(174)",shape=box];
	    "coord175"[label="Flatten(175)",shape=box];
	    "coord176"[label="Wavefront{0, NA}(176)"];
	    "coord177"[label="Wavefront{1, NA}(177)"];
	    "coord178"[label="Wavefront{-1, NA}(178)"];
	    "coord179"[label="Tile(179)",shape=box];
	    "coord180"[label="Lane{32j}(180)"];
	    "coord181"[label="VGPR{32j}(181)"];
	    "coord182"[label="Tile(182)",shape=box];
	    "coord183"[label="Tile(183)",shape=box];
	    "coord184"[label="Flatten(184)",shape=box];
	    "coord185"[label="Tile(185)",shape=box];
	    "coord186"[label="WaveTilePerWorkGroup{0, 2j}(186)"];
	    "coord187"[label="WaveTilePerWorkGroup{1, 2j}(187)"];
	    "coord188"[label="Flatten(188)",shape=box];
	    "coord189"[label="Flatten(189)",shape=box];
	    "coord190"[label="Flatten(190)",shape=box];
	    "coord191"[label="Flatten(191)",shape=box];
	    "coord192"[label="Tile(192)",shape=box];
	    "coord193"[label="Linear{Divide(CommandArgument(Load_Tiled_0_size_1), 16j)}(193)"];
	    "coord194"[label="ForLoop{Divide(CommandArgument(Load_Tiled_0_size_1), 16j)}(194)"];
	    "coord195"[label="DataFlow(195)",shape=box];
	    "coord197"[label="PassThrough(197)",shape=box];
	    "coord198"[label="Linear{2j}(198)"];
	    "coord199"[label="ForLoop{2j}(199)"];
	    "coord200"[label="DataFlow(200)",shape=box];
	    "coord201"[label="Linear{2j}(201)"];
	    "coord202"[label="ForLoop{2j}(202)"];
	    "coord203"[label="DataFlow(203)",shape=box];
	    "coord204"[label="PassThrough(204)",shape=box];
	    "coord205"[label="PassThrough(205)",shape=box];
	    "coord206"[label="PassThrough(206)",shape=box];
	    "coord207"[label="PassThrough(207)",shape=box];
	    "coord208"[label="PassThrough(208)",shape=box];
	    "coord209"[label="PassThrough(209)",shape=box];
	    "coord210"[label="PassThrough(210)",shape=box];
	    "coord211"[label="PassThrough(211)",shape=box];
	    "coord212"[label="LDS{NA}(212)"];
	    "coord213"[label="Tile(213)",shape=box];
	    "coord214"[label="MacroTile{128,16}(214)"];
	    "coord215"[label="DataFlow(215)",shape=box];
	    "coord216"[label="MacroTileNumber{0, 1j}(216)"];
	    "coord217"[label="MacroTileNumber{1, 1j}(217)"];
	    "coord218"[label="MacroTileIndex{0, 128j}(218)"];
	    "coord219"[label="MacroTileIndex{1, 16j}(219)"];
	    "coord220"[label="Tile(220)",shape=box];
	    "coord221"[label="Tile(221)",shape=box];
	    "coord222"[label="Flatten(222)",shape=box];
	    "coord223"[label="ThreadTileNumber{0, 0j}(223)"];
	    "coord224"[label="ThreadTileNumber{1, 16j}(224)"];
	    "coord225"[label="ThreadTileIndex{0, 2048j}(225)"];
	    "coord226"[label="ThreadTileIndex{1, 1j}(226)"];
	    "coord227"[label="ThreadTile{NA}(227)"];
	    "coord228"[label="Tile(228)",shape=box];
	    "coord229"[label="Workitem{0, 1j}(229)"];
	    "coord230"[label="Flatten(230)",shape=box];
	    "coord231"[label="Tile(231)",shape=box];
	    "coord232"[label="Workgroup{0, NA}(232)"];
	    "coord233"[label="PassThrough(233)",shape=box];
	    "coord234"[label="PassThrough(234)",shape=box];
	    "coord235"[label="ThreadTileIndex{0, 2048j}(235)"];
	    "coord236"[label="ThreadTileIndex{1, 1j}(236)"];
	    "coord237"[label="ThreadTile{NA}(237)"];
	    "coord238"[label="Flatten(238)",shape=box];
	    "coord239"[label="Workitem{0, 1j}(239)"];
	    "coord240"[label="Flatten(240)",shape=box];
	    "coord241"[label="PassThrough(241)",shape=box];
	    "coord242"[label="DataFlow(242)",shape=box];
	    "coord243"[label="DataFlow(243)",shape=box];
	    "coord244"[label="Offset(244)",shape=box];
	    "coord245"[label="Stride(245)",shape=box];
	    "coord246"[label="Offset(246)",shape=box];
	    "coord247"[label="Stride(247)",shape=box];
	    "coord248"[label="Offset(248)",shape=box];
	    "coord249"[label="Stride(249)",shape=box];
	    "coord250"[label="Offset(250)",shape=box];
	    "coord251"[label="Stride(251)",shape=box];
	    "coord252"[label="Offset(252)",shape=box];
	    "coord253"[label="Stride(253)",shape=box];
	    "coord254"[label="Buffer(254)",shape=box];
	    "coord255"[label="Offset(255)",shape=box];
	    "coord256"[label="Stride(256)",shape=box];
	    "coord257"[label="Offset(257)",shape=box];
	    "coord258"[label="Stride(258)",shape=box];
	    "coord259"[label="Offset(259)",shape=box];
	    "coord260"[label="Stride(260)",shape=box];
	    "coord261"[label="Buffer(261)",shape=box];
	    "coord262"[label="Offset(262)",shape=box];
	    "coord263"[label="Stride(263)",shape=box];
	    "coord264"[label="Offset(264)",shape=box];
	    "coord265"[label="Stride(265)",shape=box];
	    "coord266"[label="Buffer(266)",shape=box];
	    "coord267"[label="Offset(267)",shape=box];
	    "coord268"[label="Stride(268)",shape=box];
	    "coord269"[label="Offset(269)",shape=box];
	    "coord270"[label="Stride(270)",shape=box];
	    "coord271"[label="Buffer(271)",shape=box];
	    "coord272"[label="Offset(272)",shape=box];
	    "coord273"[label="Stride(273)",shape=box];
	    "coord274"[label="Offset(274)",shape=box];
	    "coord275"[label="Stride(275)",shape=box];
	    "coord276"[label="Buffer(276)",shape=box];
	    "coord1" -> "coord10"
	    "coord1" -> "coord215"
	    "coord1" -> "coord255"
	    "coord1" -> "coord256"
	    "coord1" -> "coord257"
	    "coord1" -> "coord258"
	    "coord1" -> "coord259"
	    "coord1" -> "coord260"
	    "coord1" -> "coord261"
	    "coord2" -> "coord15"
	    "coord2" -> "coord17"
	    "coord2" -> "coord248"
	    "coord2" -> "coord249"
	    "coord2" -> "coord250"
	    "coord2" -> "coord251"
	    "coord2" -> "coord252"
	    "coord2" -> "coord253"
	    "coord2" -> "coord254"
	    "coord3" -> "coord26"
	    "coord3" -> "coord28"
	    "coord3" -> "coord267"
	    "coord3" -> "coord268"
	    "coord3" -> "coord269"
	    "coord3" -> "coord270"
	    "coord3" -> "coord271"
	    "coord4" -> "coord7"
	    "coord5" -> "coord23"
	    "coord6" -> "coord21"
	    "coord7" -> "coord6"
	    "coord8" -> "coord220"
	    "coord9" -> "coord221"
	    "coord10" -> "coord8"
	    "coord10" -> "coord9"
	    "coord11" -> "coord19"
	    "coord13" -> "coord77"
	    "coord14" -> "coord78"
	    "coord15" -> "coord13"
	    "coord15" -> "coord14"
	    "coord16" -> "coord19"
	    "coord17" -> "coord16"
	    "coord18" -> "coord21"
	    "coord19" -> "coord18"
	    "coord20" -> "coord32"
	    "coord21" -> "coord20"
	    "coord22" -> "coord30"
	    "coord23" -> "coord22"
	    "coord24" -> "coord110"
	    "coord25" -> "coord111"
	    "coord26" -> "coord24"
	    "coord26" -> "coord25"
	    "coord27" -> "coord30"
	    "coord28" -> "coord27"
	    "coord29" -> "coord32"
	    "coord30" -> "coord29"
	    "coord31" -> "coord37"
	    "coord32" -> "coord31"
	    "coord33" -> "coord36"
	    "coord34" -> "coord36"
	    "coord36" -> "coord35"
	    "coord37" -> "coord35"
	    "coord40" -> "coord54"
	    "coord41" -> "coord55"
	    "coord50" -> "coord70"
	    "coord52" -> "coord69"
	    "coord53" -> "coord68"
	    "coord54" -> "coord50"
	    "coord54" -> "coord52"
	    "coord55" -> "coord51"
	    "coord55" -> "coord53"
	    "coord56" -> "coord61"
	    "coord57" -> "coord61"
	    "coord58" -> "coord62"
	    "coord59" -> "coord62"
	    "coord61" -> "coord58"
	    "coord62" -> "coord48"
	    "coord63" -> "coord65"
	    "coord64" -> "coord65"
	    "coord65" -> "coord59"
	    "coord66" -> "coord204"
	    "coord67" -> "coord206"
	    "coord68" -> "coord63"
	    "coord68" -> "coord60"
	    "coord69" -> "coord64"
	    "coord70" -> "coord56"
	    "coord70" -> "coord66"
	    "coord71" -> "coord197"
	    "coord72" -> "coord80"
	    "coord73" -> "coord87"
	    "coord74" -> "coord88"
	    "coord77" -> "coord71"
	    "coord77" -> "coord73"
	    "coord78" -> "coord72"
	    "coord78" -> "coord74"
	    "coord80" -> "coord76"
	    "coord84" -> "coord103"
	    "coord85" -> "coord101"
	    "coord86" -> "coord102"
	    "coord87" -> "coord83"
	    "coord87" -> "coord85"
	    "coord88" -> "coord84"
	    "coord88" -> "coord86"
	    "coord89" -> "coord94"
	    "coord90" -> "coord94"
	    "coord91" -> "coord95"
	    "coord92" -> "coord95"
	    "coord94" -> "coord91"
	    "coord95" -> "coord81"
	    "coord96" -> "coord98"
	    "coord97" -> "coord98"
	    "coord98" -> "coord92"
	    "coord99" -> "coord205"
	    "coord100" -> "coord207"
	    "coord101" -> "coord96"
	    "coord101" -> "coord93"
	    "coord102" -> "coord97"
	    "coord103" -> "coord90"
	    "coord103" -> "coord100"
	    "coord104" -> "coord112"
	    "coord105" -> "coord113"
	    "coord106" -> "coord120"
	    "coord107" -> "coord121"
	    "coord110" -> "coord104"
	    "coord110" -> "coord106"
	    "coord111" -> "coord105"
	    "coord111" -> "coord107"
	    "coord112" -> "coord108"
	    "coord113" -> "coord109"
	    "coord116" -> "coord148"
	    "coord117" -> "coord149"
	    "coord118" -> "coord141"
	    "coord118" -> "coord142"
	    "coord119" -> "coord143"
	    "coord119" -> "coord141"
	    "coord120" -> "coord116"
	    "coord120" -> "coord118"
	    "coord121" -> "coord117"
	    "coord121" -> "coord119"
	    "coord122" -> "coord127"
	    "coord123" -> "coord127"
	    "coord124" -> "coord128"
	    "coord125" -> "coord128"
	    "coord127" -> "coord124"
	    "coord128" -> "coord114"
	    "coord129" -> "coord131"
	    "coord130" -> "coord131"
	    "coord131" -> "coord125"
	    "coord132" -> "coord208"
	    "coord133" -> "coord209"
	    "coord134" -> "coord146"
	    "coord135" -> "coord146"
	    "coord136" -> "coord147"
	    "coord137" -> "coord147"
	    "coord138" -> "coord145"
	    "coord139" -> "coord144"
	    "coord140" -> "coord144"
	    "coord141" -> "coord115"
	    "coord142" -> "coord139"
	    "coord142" -> "coord135"
	    "coord143" -> "coord140"
	    "coord143" -> "coord137"
	    "coord144" -> "coord138"
	    "coord145" -> "coord134"
	    "coord145" -> "coord136"
	    "coord146" -> "coord126"
	    "coord147" -> "coord125"
	    "coord148" -> "coord122"
	    "coord148" -> "coord132"
	    "coord149" -> "coord123"
	    "coord149" -> "coord133"
	    "coord150" -> "coord156"
	    "coord151" -> "coord157"
	    "coord152" -> "coord156"
	    "coord153" -> "coord157"
	    "coord154" -> "coord158"
	    "coord155" -> "coord159"
	    "coord156" -> "coord33"
	    "coord157" -> "coord34"
	    "coord158" -> "coord150"
	    "coord159" -> "coord151"
	    "coord160" -> "coord192"
	    "coord162" -> "coord174"
	    "coord163" -> "coord175"
	    "coord164" -> "coord166"
	    "coord164" -> "coord174"
	    "coord165" -> "coord166"
	    "coord165" -> "coord175"
	    "coord166" -> "coord161"
	    "coord167" -> "coord184"
	    "coord167" -> "coord272"
	    "coord167" -> "coord273"
	    "coord167" -> "coord276"
	    "coord168" -> "coord274"
	    "coord168" -> "coord275"
	    "coord168" -> "coord190"
	    "coord169" -> "coord184"
	    "coord170" -> "coord191"
	    "coord171" -> "coord185"
	    "coord172" -> "coord190"
	    "coord173" -> "coord191"
	    "coord174" -> "coord152"
	    "coord175" -> "coord153"
	    "coord176" -> "coord188"
	    "coord177" -> "coord189"
	    "coord178" -> "coord179"
	    "coord179" -> "coord176"
	    "coord179" -> "coord177"
	    "coord180" -> "coord183"
	    "coord181" -> "coord182"
	    "coord182" -> "coord167"
	    "coord182" -> "coord168"
	    "coord183" -> "coord169"
	    "coord183" -> "coord170"
	    "coord184" -> "coord171"
	    "coord185" -> "coord172"
	    "coord185" -> "coord173"
	    "coord186" -> "coord188"
	    "coord187" -> "coord189"
	    "coord188" -> "coord162"
	    "coord189" -> "coord163"
	    "coord190" -> "coord164"
	    "coord191" -> "coord165"
	    "coord192" -> "coord178"
	    "coord192" -> "coord180"
	    "coord193" -> "coord195"
	    "coord195" -> "coord194"
	    "coord197" -> "coord194"
	    "coord198" -> "coord200"
	    "coord199" -> "coord210"
	    "coord200" -> "coord199"
	    "coord201" -> "coord203"
	    "coord202" -> "coord211"
	    "coord203" -> "coord202"
	    "coord204" -> "coord199"
	    "coord205" -> "coord199"
	    "coord206" -> "coord202"
	    "coord207" -> "coord202"
	    "coord208" -> "coord199"
	    "coord209" -> "coord202"
	    "coord210" -> "coord186"
	    "coord211" -> "coord187"
	    "coord212" -> "coord213"
	    "coord212" -> "coord243"
	    "coord212" -> "coord244"
	    "coord212" -> "coord245"
	    "coord212" -> "coord246"
	    "coord212" -> "coord247"
	    "coord213" -> "coord40"
	    "coord213" -> "coord41"
	    "coord214" -> "coord231"
	    "coord214" -> "coord241"
	    "coord214" -> "coord242"
	    "coord215" -> "coord214"
	    "coord216" -> "coord233"
	    "coord217" -> "coord234"
	    "coord218" -> "coord222"
	    "coord219" -> "coord222"
	    "coord220" -> "coord216"
	    "coord220" -> "coord218"
	    "coord221" -> "coord217"
	    "coord221" -> "coord219"
	    "coord222" -> "coord214"
	    "coord223" -> "coord230"
	    "coord224" -> "coord230"
	    "coord227" -> "coord228"
	    "coord228" -> "coord225"
	    "coord228" -> "coord226"
	    "coord230" -> "coord229"
	    "coord231" -> "coord227"
	    "coord231" -> "coord229"
	    "coord233" -> "coord232"
	    "coord234" -> "coord194"
	    "coord235" -> "coord238"
	    "coord235" -> "coord262"
	    "coord235" -> "coord263"
	    "coord235" -> "coord266"
	    "coord236" -> "coord264"
	    "coord236" -> "coord265"
	    "coord236" -> "coord238"
	    "coord237" -> "coord240"
	    "coord238" -> "coord237"
	    "coord239" -> "coord240"
	    "coord240" -> "coord214"
	    "coord241" -> "coord212"
	    "coord242" -> "coord212"
	    "coord243" -> "coord11"
	    "coord244" -> "coord51"
	    "coord245" -> "coord51"
	    "coord246" -> "coord60"
	    "coord247" -> "coord60"
	    "coord248" -> "coord71"
	    "coord249" -> "coord71"
	    "coord250" -> "coord83"
	    "coord251" -> "coord83"
	    "coord252" -> "coord93"
	    "coord253" -> "coord93"
	    "coord254" -> "coord71"
	    "coord255" -> "coord217"
	    "coord256" -> "coord217"
	    "coord257" -> "coord225"
	    "coord258" -> "coord225"
	    "coord259" -> "coord226"
	    "coord260" -> "coord226"
	    "coord261" -> "coord225"
	    "coord262" -> "coord212"
	    "coord263" -> "coord212"
	    "coord264" -> "coord212"
	    "coord265" -> "coord212"
	    "coord266" -> "coord212"
	    "coord267" -> "coord134"
	    "coord268" -> "coord134"
	    "coord269" -> "coord135"
	    "coord270" -> "coord135"
	    "coord271" -> "coord134"
	    "coord272" -> "coord35"
	    "coord273" -> "coord35"
	    "coord274" -> "coord35"
	    "coord275" -> "coord35"
	    "coord276" -> "coord35"
	    {
	    rank=same
	    "coord8"->"coord9"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord13"->"coord14"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord11"->"coord16"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord6"->"coord18"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord24"->"coord25"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord22"->"coord27"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord20"->"coord29"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord33"->"coord34"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord50"->"coord52"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord51"->"coord53"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord56"->"coord57"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord58"->"coord59"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord63"->"coord64"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord63"->"coord60"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord56"->"coord66"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord71"->"coord73"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord72"->"coord74"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord83"->"coord85"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord84"->"coord86"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord89"->"coord90"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord91"->"coord92"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord96"->"coord97"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord96"->"coord93"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord90"->"coord100"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord104"->"coord106"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord105"->"coord107"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord116"->"coord118"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord117"->"coord119"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord122"->"coord123"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord124"->"coord125"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord129"->"coord130"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord118"->"coord119"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord139"->"coord135"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord140"->"coord137"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord139"->"coord140"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord134"->"coord136"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord134"->"coord135"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord136"->"coord137"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord122"->"coord132"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord123"->"coord133"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord150"->"coord152"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord151"->"coord153"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord164"->"coord165"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord162"->"coord164"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord163"->"coord165"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord176"->"coord177"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord167"->"coord168"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord169"->"coord170"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord167"->"coord169"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord172"->"coord173"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord176"->"coord186"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord177"->"coord187"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord172"->"coord168"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord173"->"coord170"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord178"->"coord180"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord40"->"coord41"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord216"->"coord218"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord217"->"coord219"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord218"->"coord219"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord225"->"coord226"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord223"->"coord224"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord227"->"coord229"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord235"->"coord236"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord237"->"coord239"[style=invis]
	    rankdir=LR
	    }
	    subgraph clusterCF {label = "Control Graph";
	    "cntrl1"[label="Kernel(1)"];
	    "cntrl2"[label="LoadVGPR(2)"];
	    "cntrl3"[label="Body(3)",shape=box];
	    "cntrl4"[label="LoadLDSTile(4)"];
	    "cntrl6"[label="LoadTiled(6)"];
	    "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(6), DataFlowTag(18))(11)"];
	    "cntrl12"[label="Sequence(12)",shape=box];
	    "cntrl14"[label="LoadVGPR(14)"];
	    "cntrl15"[label="Body(15)",shape=box];
	    "cntrl16"[label="LoadTiled(16)"];
	    "cntrl18"[label="Assign VGPR Multiply(DataFlowTag(22), DataFlowTag(27))(18)"];
	    "cntrl19"[label="Sequence(19)",shape=box];
	    "cntrl21"[label="Assign VGPR Add(DataFlowTag(20), DataFlowTag(29))(21)"];
	    "cntrl22"[label="Sequence(22)",shape=box];
	    "cntrl23"[label="Sequence(23)",shape=box];
	    "cntrl24"[label="StoreTiled(24)"];
	    "cntrl26"[label="ForLoopOp KLoop: LessThan(DataFlowTag(193), Divide(CommandArgument(Load_Tiled_0_size_1), 16j))(26)"];
	    "cntrl27"[label="Assign SGPR 0l(27)"];
	    "cntrl28"[label="Assign SGPR Add(DataFlowTag(193), 1j)(28)"];
	    "cntrl29"[label="Initialize(29)",shape=box];
	    "cntrl30"[label="ForLoopIncrement(30)",shape=box];
	    "cntrl31"[label="Multiply(31)"];
	    "cntrl32"[label="Assign ACCVGPR 0.00000f(32)"];
	    "cntrl34"[label="Body(34)",shape=box];
	    "cntrl35"[label="Body(35)",shape=box];
	    "cntrl36"[label="Body(36)",shape=box];
	    "cntrl37"[label="ForLoopOp XLoop: LessThan(DataFlowTag(198), 2j)(37)"];
	    "cntrl38"[label="Assign SGPR 0j(38)"];
	    "cntrl39"[label="Assign SGPR Add(DataFlowTag(198), 1j)(39)"];
	    "cntrl40"[label="Initialize(40)",shape=box];
	    "cntrl41"[label="ForLoopIncrement(41)",shape=box];
	    "cntrl42"[label="ForLoopOp YLoop: LessThan(DataFlowTag(201), 2j)(42)"];
	    "cntrl43"[label="Assign SGPR 0j(43)"];
	    "cntrl44"[label="Assign SGPR Add(DataFlowTag(201), 1j)(44)"];
	    "cntrl45"[label="Initialize(45)",shape=box];
	    "cntrl46"[label="ForLoopIncrement(46)",shape=box];
	    "cntrl47"[label="Body(47)",shape=box];
	    "cntrl48"[label="Body(48)",shape=box];
	    "cntrl49"[label="Body(49)",shape=box];
	    "cntrl50"[label="LoadTiled(50)"];
	    "cntrl51"[label="StoreLDSTile(51)"];
	    "cntrl52"[label="Body(52)",shape=box];
	    "cntrl53"[label="Barrier(53)"];
	    "cntrl54"[label="Sequence(54)",shape=box];
	    "cntrl55"[label="Sequence(55)",shape=box];
	    "cntrl56"[label="Barrier(56)"];
	    "cntrl57"[label="Sequence(57)",shape=box];
	    "cntrl58"[label="Sequence(58)",shape=box];
	    "cntrl59"[label="Scope(59)"];
	    "cntrl60"[label="Sequence(60)",shape=box];
	    "cntrl61"[label="Sequence(61)",shape=box];
	    "cntrl62"[label="ComputeIndex(62)"];
	    "cntrl63"[label="ComputeIndex(63)"];
	    "cntrl64"[label="Sequence(64)",shape=box];
	    "cntrl65"[label="Body(65)",shape=box];
	    "cntrl66"[label="Sequence(66)",shape=box];
	    "cntrl67"[label="ComputeIndex(67)"];
	    "cntrl68"[label="ComputeIndex(68)"];
	    "cntrl69"[label="ComputeIndex(69)"];
	    "cntrl70"[label="Sequence(70)",shape=box];
	    "cntrl71"[label="Sequence(71)",shape=box];
	    "cntrl72"[label="Assign VGPR Add(DataFlowTag(248), DataFlowTag(249))(72)"];
	    "cntrl73"[label="Body(73)",shape=box];
	    "cntrl74"[label="Sequence(74)",shape=box];
	    "cntrl75"[label="ForLoopIncrement(75)",shape=box];
	    "cntrl76"[label="ComputeIndex(76)"];
	    "cntrl77"[label="ComputeIndex(77)"];
	    "cntrl78"[label="ComputeIndex(78)"];
	    "cntrl79"[label="Sequence(79)",shape=box];
	    "cntrl80"[label="Sequence(80)",shape=box];
	    "cntrl81"[label="Assign VGPR Add(DataFlowTag(255), DataFlowTag(256))(81)"];
	    "cntrl82"[label="Body(82)",shape=box];
	    "cntrl83"[label="Sequence(83)",shape=box];
	    "cntrl84"[label="ForLoopIncrement(84)",shape=box];
	    "cntrl85"[label="ComputeIndex(85)"];
	    "cntrl86"[label="ComputeIndex(86)"];
	    "cntrl87"[label="Body(87)",shape=box];
	    "cntrl88"[label="Sequence(88)",shape=box];
	    "cntrl89"[label="Sequence(89)",shape=box];
	    "cntrl90"[label="Scope(90)"];
	    "cntrl91"[label="Body(91)",shape=box];
	    "cntrl92"[label="Sequence(92)",shape=box];
	    "cntrl93"[label="ComputeIndex(93)"];
	    "cntrl94"[label="ComputeIndex(94)"];
	    "cntrl95"[label="Body(95)",shape=box];
	    "cntrl96"[label="Sequence(96)",shape=box];
	    "cntrl97"[label="Sequence(97)",shape=box];
	    "cntrl98"[label="Scope(98)"];
	    "cntrl99"[label="Sequence(99)",shape=box];
	    "cntrl100"[label="ComputeIndex(100)"];
	    "cntrl101"[label="ComputeIndex(101)"];
	    "cntrl102"[label="Body(102)",shape=box];
	    "cntrl103"[label="Sequence(103)",shape=box];
	    "cntrl104"[label="Sequence(104)",shape=box];
	    "cntrl1" -> "cntrl3"
	    "cntrl1" -> "cntrl15"
	    "cntrl1" -> "cntrl49"
	    "cntrl2" -> "cntrl12"
	    "cntrl3" -> "cntrl2"
	    "cntrl11" -> "cntrl22"
	    "cntrl12" -> "cntrl37"
	    "cntrl14" -> "cntrl19"
	    "cntrl15" -> "cntrl14"
	    "cntrl18" -> "cntrl23"
	    "cntrl19" -> "cntrl37"
	    "cntrl21" -> "cntrl99"
	    "cntrl22" -> "cntrl21"
	    "cntrl23" -> "cntrl21"
	    "cntrl26" -> "cntrl29"
	    "cntrl26" -> "cntrl30"
	    "cntrl26" -> "cntrl34"
	    "cntrl26" -> "cntrl52"
	    "cntrl26" -> "cntrl75"
	    "cntrl26" -> "cntrl84"
	    "cntrl29" -> "cntrl27"
	    "cntrl30" -> "cntrl28"
	    "cntrl31" -> "cntrl35"
	    "cntrl31" -> "cntrl36"
	    "cntrl32" -> "cntrl60"
	    "cntrl34" -> "cntrl31"
	    "cntrl35" -> "cntrl4"
	    "cntrl36" -> "cntrl6"
	    "cntrl37" -> "cntrl40"
	    "cntrl37" -> "cntrl41"
	    "cntrl37" -> "cntrl48"
	    "cntrl40" -> "cntrl38"
	    "cntrl41" -> "cntrl39"
	    "cntrl42" -> "cntrl45"
	    "cntrl42" -> "cntrl46"
	    "cntrl42" -> "cntrl47"
	    "cntrl42" -> "cntrl91"
	    "cntrl45" -> "cntrl43"
	    "cntrl46" -> "cntrl44"
	    "cntrl47" -> "cntrl32"
	    "cntrl48" -> "cntrl42"
	    "cntrl49" -> "cntrl37"
	    "cntrl50" -> "cntrl54"
	    "cntrl51" -> "cntrl57"
	    "cntrl52" -> "cntrl50"
	    "cntrl53" -> "cntrl55"
	    "cntrl54" -> "cntrl53"
	    "cntrl55" -> "cntrl51"
	    "cntrl56" -> "cntrl58"
	    "cntrl57" -> "cntrl56"
	    "cntrl58" -> "cntrl31"
	    "cntrl59" -> "cntrl61"
	    "cntrl59" -> "cntrl65"
	    "cntrl59" -> "cntrl73"
	    "cntrl59" -> "cntrl82"
	    "cntrl59" -> "cntrl87"
	    "cntrl60" -> "cntrl59"
	    "cntrl61" -> "cntrl11"
	    "cntrl62" -> "cntrl64"
	    "cntrl63" -> "cntrl66"
	    "cntrl64" -> "cntrl63"
	    "cntrl65" -> "cntrl62"
	    "cntrl66" -> "cntrl26"
	    "cntrl67" -> "cntrl70"
	    "cntrl68" -> "cntrl71"
	    "cntrl69" -> "cntrl74"
	    "cntrl70" -> "cntrl68"
	    "cntrl71" -> "cntrl69"
	    "cntrl73" -> "cntrl67"
	    "cntrl74" -> "cntrl26"
	    "cntrl75" -> "cntrl72"
	    "cntrl76" -> "cntrl79"
	    "cntrl77" -> "cntrl80"
	    "cntrl78" -> "cntrl83"
	    "cntrl79" -> "cntrl77"
	    "cntrl80" -> "cntrl78"
	    "cntrl82" -> "cntrl76"
	    "cntrl83" -> "cntrl26"
	    "cntrl84" -> "cntrl81"
	    "cntrl85" -> "cntrl88"
	    "cntrl86" -> "cntrl89"
	    "cntrl87" -> "cntrl85"
	    "cntrl88" -> "cntrl86"
	    "cntrl89" -> "cntrl26"
	    "cntrl90" -> "cntrl92"
	    "cntrl90" -> "cntrl95"
	    "cntrl91" -> "cntrl90"
	    "cntrl92" -> "cntrl18"
	    "cntrl93" -> "cntrl96"
	    "cntrl94" -> "cntrl97"
	    "cntrl95" -> "cntrl93"
	    "cntrl96" -> "cntrl94"
	    "cntrl97" -> "cntrl16"
	    "cntrl98" -> "cntrl102"
	    "cntrl99" -> "cntrl98"
	    "cntrl100" -> "cntrl103"
	    "cntrl101" -> "cntrl104"
	    "cntrl102" -> "cntrl100"
	    "cntrl103" -> "cntrl101"
	    "cntrl104" -> "cntrl24"
	    }
	    "coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord6" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord9" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord49" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord50" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord66" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord67" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord244" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord245" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord246" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord247" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord14" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord16" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord72" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord76" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord82" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord84" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord99" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord100" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord248" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord249" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord250" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord251" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord252" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord253" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord254" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
	    "coord16" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
	    "coord18" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
	    "coord20" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
	    "coord5" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
	    "coord22" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord24" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord25" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord27" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord104" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord105" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord108" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord109" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord115" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord116" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord117" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord126" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord132" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord133" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord267" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord268" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord269" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord270" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord271" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
	    "coord29" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord31" -> "cntrl21" [style=dotted,weight=0,arrowsize=0]
	    "coord31" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord35" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord150" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord151" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord154" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord155" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord161" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord167" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord168" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord181" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord186" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord187" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord272" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord273" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord274" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord275" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord276" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
	    "coord193" -> "cntrl26" [style=dotted,weight=0,arrowsize=0]
	    "coord193" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord193" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord16" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord18" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord49" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord82" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
	    "coord18" -> "cntrl32" [style=dotted,weight=0,arrowsize=0]
	    "coord198" -> "cntrl37" [style=dotted,weight=0,arrowsize=0]
	    "coord198" -> "cntrl38" [style=dotted,weight=0,arrowsize=0]
	    "coord198" -> "cntrl39" [style=dotted,weight=0,arrowsize=0]
	    "coord201" -> "cntrl42" [style=dotted,weight=0,arrowsize=0]
	    "coord201" -> "cntrl43" [style=dotted,weight=0,arrowsize=0]
	    "coord201" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord8" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord9" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord214" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord216" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord232" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord255" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord256" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord257" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord258" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord259" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord260" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord261" -> "cntrl50" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord214" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord236" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord262" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord263" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord264" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord265" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord266" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord244" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord245" -> "cntrl62" [style=dotted,weight=0,arrowsize=0]
	    "coord51" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord60" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord244" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord246" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord247" -> "cntrl63" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord248" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord249" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord254" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord248" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord250" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord251" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord254" -> "cntrl68" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord71" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord83" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord93" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord250" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord252" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord253" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord254" -> "cntrl69" [style=dotted,weight=0,arrowsize=0]
	    "coord248" -> "cntrl72" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord255" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord256" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord261" -> "cntrl76" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord257" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord258" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord261" -> "cntrl77" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord217" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord225" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord226" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord259" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord260" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord261" -> "cntrl78" [style=dotted,weight=0,arrowsize=0]
	    "coord255" -> "cntrl81" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl85" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl85" [style=dotted,weight=0,arrowsize=0]
	    "coord236" -> "cntrl85" [style=dotted,weight=0,arrowsize=0]
	    "coord262" -> "cntrl85" [style=dotted,weight=0,arrowsize=0]
	    "coord263" -> "cntrl85" [style=dotted,weight=0,arrowsize=0]
	    "coord266" -> "cntrl85" [style=dotted,weight=0,arrowsize=0]
	    "coord212" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord235" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord236" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord264" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord265" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord266" -> "cntrl86" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl93" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl93" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl93" [style=dotted,weight=0,arrowsize=0]
	    "coord267" -> "cntrl93" [style=dotted,weight=0,arrowsize=0]
	    "coord268" -> "cntrl93" [style=dotted,weight=0,arrowsize=0]
	    "coord271" -> "cntrl93" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord267" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord269" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord270" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord271" -> "cntrl94" [style=dotted,weight=0,arrowsize=0]
	    "coord35" -> "cntrl100" [style=dotted,weight=0,arrowsize=0]
	    "coord167" -> "cntrl100" [style=dotted,weight=0,arrowsize=0]
	    "coord168" -> "cntrl100" [style=dotted,weight=0,arrowsize=0]
	    "coord272" -> "cntrl100" [style=dotted,weight=0,arrowsize=0]
	    "coord273" -> "cntrl100" [style=dotted,weight=0,arrowsize=0]
	    "coord276" -> "cntrl100" [style=dotted,weight=0,arrowsize=0]
	    "coord35" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    "coord167" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    "coord168" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    "coord272" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    "coord274" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    "coord275" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    "coord276" -> "cntrl101" [style=dotted,weight=0,arrowsize=0]
	    }).";

        EXPECT_EQ(NormalizedSource(expected_lds), NormalizedSource(kgraph_lds_lower.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateTMul)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1)));

        auto kgraph0 = translate(command);

        std::string expected0 = R".(
	  digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{NA}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{NA}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="MacroTile{NA}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord16"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord16"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord16" -> "coord15"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord4"->"coord11"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
		"cntrl5" -> "cntrl4"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		}
		"coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
          }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateTMulB)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1)));

        auto kgraph0 = translate(command);

        // macro tile sizes
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        int t_m = 4;
        int t_n = 2;

        auto mac_tile_0 = MacroTile({mac_m, mac_k}, MemoryType::VGPR, {t_m, t_n}); // A
        auto mac_tile_1 = MacroTile({mac_k, mac_n}, MemoryType::VGPR, {t_m, t_n}); // B
        auto mac_tile_2 = MacroTile({mac_m, mac_n}, MemoryType::VGPR, {t_m, t_n}); // A * B

        auto params = std::make_shared<CommandParameters>();

        params->setDimensionInfo(4, mac_tile_0);
        params->setDimensionInfo(11, mac_tile_1);
        params->setDimensionInfo(15, mac_tile_2);

        kgraph0 = updateParameters(kgraph0, params);

        std::string expected0 = R".(
            digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{64,64}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{64,64}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="MacroTile{64,64}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord16"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord16"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord16" -> "coord15"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord4"->"coord11"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
		"cntrl5" -> "cntrl4"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		}
		"coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		}
        ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TileAdd)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 1))); // b

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(2, 0, 0))); // a + a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 1, 1))); // b + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(4, 3, 2))); // 2a + 2b

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Int32, 2, 4))); // c

        auto kgraph0 = translate(command);

        int m = 16;
        int n = 8;

        int t_m = 4;
        int t_n = 2;

        auto params = std::make_shared<CommandParameters>();

        auto mac_tile_0 = MacroTile({m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_1 = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_2 = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_3 = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_4 = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

        params->setDimensionInfo(4, mac_tile_0);
        params->setDimensionInfo(11, mac_tile_1);
        params->setDimensionInfo(15, mac_tile_2);
        params->setDimensionInfo(17, mac_tile_3);
        params->setDimensionInfo(19, mac_tile_4);
        kgraph0 = updateParameters(kgraph0, params);

        std::string expected0 = R".(
            digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{16,8}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{16,8}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="MacroTile{16,8}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord17"[label="MacroTile{16,8}(17)"];
		"coord18"[label="DataFlow(18)",shape=box];
		"coord19"[label="MacroTile{16,8}(19)"];
		"coord20"[label="DataFlow(20)",shape=box];
		"coord21"[label="SubDimension{0, NA}(21)"];
		"coord22"[label="SubDimension{1, NA}(22)"];
		"coord23"[label="User{NA}(23)"];
		"coord24"[label="DestructTensorTile(24)",shape=box];
		"coord25"[label="Join(25)",shape=box];
		"coord26"[label="DataFlow(26)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord16"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord18"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord15" -> "coord20"
		"coord16" -> "coord15"
		"coord17" -> "coord20"
		"coord18" -> "coord17"
		"coord19" -> "coord24"
		"coord19" -> "coord26"
		"coord20" -> "coord19"
		"coord21" -> "coord25"
		"coord22" -> "coord25"
		"coord24" -> "coord21"
		"coord24" -> "coord22"
		"coord25" -> "coord23"
		"coord26" -> "coord23"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord17"->"coord15"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord21"->"coord22"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord21"->"coord22"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="Assign VGPR Add(DataFlowTag(4), DataFlowTag(4))(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="Assign VGPR Add(DataFlowTag(11), DataFlowTag(11))(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="Sequence(11)",shape=box];
		"cntrl12"[label="Assign VGPR Add(DataFlowTag(17), DataFlowTag(15))(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="StoreTiled(15)"];
		"cntrl16"[label="Sequence(16)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl2" -> "cntrl8"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl10"
		"cntrl4" -> "cntrl11"
		"cntrl5" -> "cntrl4"
		"cntrl6" -> "cntrl14"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		"cntrl9" -> "cntrl13"
		"cntrl10" -> "cntrl9"
		"cntrl11" -> "cntrl9"
		"cntrl12" -> "cntrl16"
		"cntrl13" -> "cntrl12"
		"cntrl14" -> "cntrl12"
		"cntrl16" -> "cntrl15"
		}
		"coord1" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord17" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
		"coord19" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord19" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord23" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		}).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        auto kgraph1 = lowerTile(kgraph0, params, m_context);
        kgraph1      = addComputeIndexOperations(kgraph1);

        std::string expected1 = R".(
        digraph {
	    "coord1"[label="User{NA}(1)"];
	    "coord2"[label="User{NA}(2)"];
	    "coord3"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(3)"];
	    "coord4"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(4)"];
	    "coord5"[label="Split(5)",shape=box];
	    "coord6"[label="MacroTile{16,8}(6)"];
	    "coord7"[label="DataFlow(7)",shape=box];
	    "coord8"[label="MacroTile{16,8}(8)"];
	    "coord9"[label="DataFlow(9)",shape=box];
	    "coord10"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(10)"];
	    "coord11"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(11)"];
	    "coord12"[label="Split(12)",shape=box];
	    "coord13"[label="MacroTile{16,8}(13)"];
	    "coord15"[label="MacroTile{16,8}(15)"];
	    "coord16"[label="DataFlow(16)",shape=box];
	    "coord17"[label="MacroTile{16,8}(17)"];
	    "coord18"[label="DataFlow(18)",shape=box];
	    "coord19"[label="SubDimension{0, NA}(19)"];
	    "coord20"[label="SubDimension{1, NA}(20)"];
	    "coord21"[label="User{NA}(21)"];
	    "coord22"[label="Join(22)",shape=box];
	    "coord23"[label="DataFlow(23)",shape=box];
	    "coord24"[label="MacroTileNumber{0, 1j}(24)"];
	    "coord25"[label="MacroTileNumber{1, 1j}(25)"];
	    "coord26"[label="MacroTileIndex{0, 16j}(26)"];
	    "coord27"[label="MacroTileIndex{1, 8j}(27)"];
	    "coord28"[label="Workgroup{0, NA}(28)"];
	    "coord29"[label="Workgroup{1, NA}(29)"];
	    "coord30"[label="Tile(30)",shape=box];
	    "coord31"[label="Tile(31)",shape=box];
	    "coord32"[label="PassThrough(32)",shape=box];
	    "coord33"[label="PassThrough(33)",shape=box];
	    "coord34"[label="Workitem{0, NA}(34)"];
	    "coord35"[label="Workitem{1, NA}(35)"];
	    "coord36"[label="ThreadTileNumber{0, 4j}(36)"];
	    "coord37"[label="ThreadTileNumber{1, 4j}(37)"];
	    "coord38"[label="ThreadTileIndex{0, 4j}(38)"];
	    "coord39"[label="ThreadTileIndex{1, 2j}(39)"];
	    "coord40"[label="Tile(40)",shape=box];
	    "coord41"[label="Tile(41)",shape=box];
	    "coord42"[label="PassThrough(42)",shape=box];
	    "coord43"[label="PassThrough(43)",shape=box];
	    "coord46"[label="MacroTileIndex{0, 16j}(46)"];
	    "coord47"[label="MacroTileIndex{1, 8j}(47)"];
	    "coord54"[label="Workitem{0, NA}(54)"];
	    "coord55"[label="Workitem{1, NA}(55)"];
	    "coord56"[label="ThreadTileNumber{0, 4j}(56)"];
	    "coord57"[label="ThreadTileNumber{1, 4j}(57)"];
	    "coord58"[label="ThreadTileIndex{0, 4j}(58)"];
	    "coord59"[label="ThreadTileIndex{1, 2j}(59)"];
	    "coord60"[label="Tile(60)",shape=box];
	    "coord61"[label="Tile(61)",shape=box];
	    "coord62"[label="PassThrough(62)",shape=box];
	    "coord63"[label="PassThrough(63)",shape=box];
	    "coord64"[label="MacroTileNumber{0, 1j}(64)"];
	    "coord65"[label="MacroTileNumber{1, 1j}(65)"];
	    "coord66"[label="MacroTileIndex{0, 16j}(66)"];
	    "coord67"[label="MacroTileIndex{1, 8j}(67)"];
	    "coord68"[label="Workgroup{0, NA}(68)"];
	    "coord69"[label="Workgroup{1, NA}(69)"];
	    "coord70"[label="Flatten(70)",shape=box];
	    "coord71"[label="Flatten(71)",shape=box];
	    "coord72"[label="PassThrough(72)",shape=box];
	    "coord73"[label="PassThrough(73)",shape=box];
	    "coord74"[label="Workitem{0, 1j}(74)"];
	    "coord75"[label="Workitem{1, 1j}(75)"];
	    "coord76"[label="ThreadTile{NA}(76)"];
	    "coord77"[label="ThreadTileNumber{0, 4j}(77)"];
	    "coord78"[label="ThreadTileNumber{1, 4j}(78)"];
	    "coord79"[label="ThreadTileIndex{0, 4j}(79)"];
	    "coord80"[label="ThreadTileIndex{1, 2j}(80)"];
	    "coord81"[label="Split(81)",shape=box];
	    "coord82"[label="Flatten(82)",shape=box];
	    "coord83"[label="Flatten(83)",shape=box];
	    "coord84"[label="PassThrough(84)",shape=box];
	    "coord85"[label="PassThrough(85)",shape=box];
	    "coord86"[label="LDS{NA}(86)"];
	    "coord87"[label="Tile(87)",shape=box];
	    "coord88"[label="MacroTile{16,8}(88)"];
	    "coord89"[label="DataFlow(89)",shape=box];
	    "coord90"[label="MacroTileNumber{0, 1j}(90)"];
	    "coord91"[label="MacroTileNumber{1, 1j}(91)"];
	    "coord92"[label="MacroTileIndex{0, 16j}(92)"];
	    "coord93"[label="MacroTileIndex{1, 8j}(93)"];
	    "coord94"[label="Tile(94)",shape=box];
	    "coord95"[label="Tile(95)",shape=box];
	    "coord96"[label="Flatten(96)",shape=box];
	    "coord97"[label="ThreadTileNumber{0, 4j}(97)"];
	    "coord98"[label="ThreadTileNumber{1, 4j}(98)"];
	    "coord99"[label="ThreadTileIndex{0, 4j}(99)"];
	    "coord100"[label="ThreadTileIndex{1, 2j}(100)"];
	    "coord101"[label="Workitem{0, 1j}(101)"];
	    "coord102"[label="Workitem{1, 1j}(102)"];
	    "coord103"[label="Tile(103)",shape=box];
	    "coord104"[label="Tile(104)",shape=box];
	    "coord105"[label="PassThrough(105)",shape=box];
	    "coord106"[label="PassThrough(106)",shape=box];
	    "coord107"[label="Workgroup{0, NA}(107)"];
	    "coord108"[label="PassThrough(108)",shape=box];
	    "coord109"[label="Workgroup{1, NA}(109)"];
	    "coord110"[label="PassThrough(110)",shape=box];
	    "coord111"[label="ThreadTileIndex{0, 4j}(111)"];
	    "coord112"[label="ThreadTileIndex{1, 2j}(112)"];
	    "coord113"[label="Workitem{0, 1j}(113)"];
	    "coord114"[label="Workitem{1, 1j}(114)"];
	    "coord115"[label="ThreadTileNumber{0, 4j}(115)"];
	    "coord116"[label="ThreadTileNumber{1, 4j}(116)"];
	    "coord117"[label="PassThrough(117)",shape=box];
	    "coord118"[label="PassThrough(118)",shape=box];
	    "coord119"[label="MacroTileIndex{0, 16j}(119)"];
	    "coord120"[label="MacroTileIndex{1, 8j}(120)"];
	    "coord121"[label="Flatten(121)",shape=box];
	    "coord122"[label="Flatten(122)",shape=box];
	    "coord123"[label="Flatten(123)",shape=box];
	    "coord124"[label="DataFlow(124)",shape=box];
	    "coord125"[label="DataFlow(125)",shape=box];
	    "coord126"[label="Offset(126)",shape=box];
	    "coord127"[label="Stride(127)",shape=box];
	    "coord128"[label="Offset(128)",shape=box];
	    "coord129"[label="Stride(129)",shape=box];
	    "coord130"[label="Buffer(130)",shape=box];
	    "coord131"[label="Offset(131)",shape=box];
	    "coord132"[label="Stride(132)",shape=box];
	    "coord133"[label="Offset(133)",shape=box];
	    "coord134"[label="Stride(134)",shape=box];
	    "coord135"[label="Buffer(135)",shape=box];
	    "coord136"[label="Offset(136)",shape=box];
	    "coord137"[label="Stride(137)",shape=box];
	    "coord138"[label="Offset(138)",shape=box];
	    "coord139"[label="Stride(139)",shape=box];
	    "coord140"[label="Buffer(140)",shape=box];
	    "coord141"[label="Offset(141)",shape=box];
	    "coord142"[label="Stride(142)",shape=box];
	    "coord143"[label="Offset(143)",shape=box];
	    "coord144"[label="Stride(144)",shape=box];
	    "coord145"[label="Buffer(145)",shape=box];
	    "coord146"[label="Offset(146)",shape=box];
	    "coord147"[label="Stride(147)",shape=box];
	    "coord148"[label="Offset(148)",shape=box];
	    "coord149"[label="Stride(149)",shape=box];
	    "coord150"[label="Buffer(150)",shape=box];
	    "coord1" -> "coord12"
	    "coord1" -> "coord89"
	    "coord1" -> "coord131"
	    "coord1" -> "coord132"
	    "coord1" -> "coord133"
	    "coord1" -> "coord134"
	    "coord1" -> "coord135"
	    "coord2" -> "coord5"
	    "coord2" -> "coord7"
	    "coord2" -> "coord126"
	    "coord2" -> "coord127"
	    "coord2" -> "coord128"
	    "coord2" -> "coord129"
	    "coord2" -> "coord130"
	    "coord3" -> "coord30"
	    "coord4" -> "coord31"
	    "coord5" -> "coord3"
	    "coord5" -> "coord4"
	    "coord6" -> "coord9"
	    "coord7" -> "coord6"
	    "coord8" -> "coord18"
	    "coord9" -> "coord8"
	    "coord10" -> "coord94"
	    "coord11" -> "coord95"
	    "coord12" -> "coord10"
	    "coord12" -> "coord11"
	    "coord13" -> "coord16"
	    "coord15" -> "coord18"
	    "coord16" -> "coord15"
	    "coord17" -> "coord23"
	    "coord18" -> "coord17"
	    "coord19" -> "coord22"
	    "coord20" -> "coord22"
	    "coord22" -> "coord21"
	    "coord23" -> "coord21"
	    "coord24" -> "coord32"
	    "coord25" -> "coord33"
	    "coord26" -> "coord40"
	    "coord27" -> "coord41"
	    "coord30" -> "coord24"
	    "coord30" -> "coord26"
	    "coord31" -> "coord25"
	    "coord31" -> "coord27"
	    "coord32" -> "coord28"
	    "coord33" -> "coord29"
	    "coord36" -> "coord42"
	    "coord37" -> "coord43"
	    "coord40" -> "coord36"
	    "coord40" -> "coord38"
	    "coord41" -> "coord37"
	    "coord41" -> "coord39"
	    "coord42" -> "coord34"
	    "coord43" -> "coord35"
	    "coord46" -> "coord60"
	    "coord47" -> "coord61"
	    "coord56" -> "coord62"
	    "coord57" -> "coord63"
	    "coord60" -> "coord56"
	    "coord60" -> "coord58"
	    "coord61" -> "coord57"
	    "coord61" -> "coord59"
	    "coord62" -> "coord54"
	    "coord63" -> "coord55"
	    "coord64" -> "coord70"
	    "coord65" -> "coord71"
	    "coord66" -> "coord70"
	    "coord67" -> "coord71"
	    "coord68" -> "coord72"
	    "coord69" -> "coord73"
	    "coord70" -> "coord19"
	    "coord71" -> "coord20"
	    "coord72" -> "coord64"
	    "coord73" -> "coord65"
	    "coord74" -> "coord84"
	    "coord75" -> "coord85"
	    "coord76" -> "coord81"
	    "coord77" -> "coord82"
	    "coord78" -> "coord83"
	    "coord79" -> "coord141"
	    "coord79" -> "coord142"
	    "coord79" -> "coord145"
	    "coord79" -> "coord82"
	    "coord80" -> "coord143"
	    "coord80" -> "coord144"
	    "coord80" -> "coord83"
	    "coord81" -> "coord79"
	    "coord81" -> "coord80"
	    "coord82" -> "coord66"
	    "coord83" -> "coord67"
	    "coord84" -> "coord77"
	    "coord85" -> "coord78"
	    "coord86" -> "coord87"
	    "coord86" -> "coord125"
	    "coord86" -> "coord136"
	    "coord86" -> "coord137"
	    "coord86" -> "coord138"
	    "coord86" -> "coord139"
	    "coord86" -> "coord140"
	    "coord87" -> "coord46"
	    "coord87" -> "coord47"
	    "coord88" -> "coord124"
	    "coord89" -> "coord88"
	    "coord90" -> "coord108"
	    "coord91" -> "coord110"
	    "coord92" -> "coord96"
	    "coord92" -> "coord103"
	    "coord93" -> "coord104"
	    "coord93" -> "coord96"
	    "coord94" -> "coord90"
	    "coord94" -> "coord92"
	    "coord95" -> "coord91"
	    "coord95" -> "coord93"
	    "coord96" -> "coord88"
	    "coord97" -> "coord105"
	    "coord98" -> "coord106"
	    "coord103" -> "coord99"
	    "coord103" -> "coord97"
	    "coord104" -> "coord100"
	    "coord104" -> "coord98"
	    "coord105" -> "coord101"
	    "coord106" -> "coord102"
	    "coord108" -> "coord107"
	    "coord110" -> "coord109"
	    "coord111" -> "coord121"
	    "coord111" -> "coord146"
	    "coord111" -> "coord147"
	    "coord111" -> "coord150"
	    "coord112" -> "coord122"
	    "coord112" -> "coord148"
	    "coord112" -> "coord149"
	    "coord113" -> "coord117"
	    "coord114" -> "coord118"
	    "coord115" -> "coord121"
	    "coord116" -> "coord122"
	    "coord117" -> "coord115"
	    "coord118" -> "coord116"
	    "coord119" -> "coord123"
	    "coord120" -> "coord123"
	    "coord121" -> "coord119"
	    "coord122" -> "coord120"
	    "coord123" -> "coord86"
	    "coord124" -> "coord86"
	    "coord125" -> "coord13"
	    "coord126" -> "coord38"
	    "coord127" -> "coord38"
	    "coord128" -> "coord39"
	    "coord129" -> "coord39"
	    "coord130" -> "coord38"
	    "coord131" -> "coord99"
	    "coord132" -> "coord99"
	    "coord133" -> "coord100"
	    "coord134" -> "coord100"
	    "coord135" -> "coord99"
	    "coord136" -> "coord58"
	    "coord137" -> "coord58"
	    "coord138" -> "coord59"
	    "coord139" -> "coord59"
	    "coord140" -> "coord58"
	    "coord141" -> "coord21"
	    "coord142" -> "coord21"
	    "coord143" -> "coord21"
	    "coord144" -> "coord21"
	    "coord145" -> "coord21"
	    "coord146" -> "coord86"
	    "coord147" -> "coord86"
	    "coord148" -> "coord86"
	    "coord149" -> "coord86"
	    "coord150" -> "coord86"
	    {
	    rank=same
	    "coord3"->"coord4"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord10"->"coord11"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord8"->"coord15"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord19"->"coord20"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord24"->"coord26"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord25"->"coord27"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord36"->"coord38"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord37"->"coord39"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord56"->"coord58"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord57"->"coord59"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord64"->"coord66"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord65"->"coord67"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord79"->"coord80"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord77"->"coord79"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord78"->"coord80"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord46"->"coord47"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord90"->"coord92"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord91"->"coord93"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord92"->"coord93"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord99"->"coord97"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord100"->"coord98"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord111"->"coord115"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord112"->"coord116"[style=invis]
	    rankdir=LR
	    }
	    {
	    rank=same
	    "coord119"->"coord120"[style=invis]
	    rankdir=LR
	    }
	    subgraph clusterCF {label = "Control Graph";
	    "cntrl1"[label="Kernel(1)"];
	    "cntrl2"[label="LoadTiled(2)"];
	    "cntrl4"[label="Assign VGPR Add(DataFlowTag(6), DataFlowTag(6))(4)"];
	    "cntrl7"[label="LoadLDSTile(7)"];
	    "cntrl9"[label="Assign VGPR Add(DataFlowTag(13), DataFlowTag(13))(9)"];
	    "cntrl12"[label="Assign VGPR Add(DataFlowTag(8), DataFlowTag(15))(12)"];
	    "cntrl13"[label="Sequence(13)",shape=box];
	    "cntrl14"[label="Sequence(14)",shape=box];
	    "cntrl15"[label="StoreTiled(15)"];
	    "cntrl17"[label="LoadTiled(17)"];
	    "cntrl18"[label="StoreLDSTile(18)"];
	    "cntrl19"[label="Barrier(19)"];
	    "cntrl23"[label="Scope(23)"];
	    "cntrl24"[label="Body(24)",shape=box];
	    "cntrl25"[label="Sequence(25)",shape=box];
	    "cntrl26"[label="Sequence(26)",shape=box];
	    "cntrl27"[label="ComputeIndex(27)"];
	    "cntrl28"[label="ComputeIndex(28)"];
	    "cntrl29"[label="Body(29)",shape=box];
	    "cntrl30"[label="Sequence(30)",shape=box];
	    "cntrl31"[label="Sequence(31)",shape=box];
	    "cntrl32"[label="Scope(32)"];
	    "cntrl33"[label="Body(33)",shape=box];
	    "cntrl35"[label="ComputeIndex(35)"];
	    "cntrl36"[label="ComputeIndex(36)"];
	    "cntrl37"[label="Body(37)",shape=box];
	    "cntrl38"[label="Sequence(38)",shape=box];
	    "cntrl39"[label="Sequence(39)",shape=box];
	    "cntrl40"[label="Scope(40)"];
	    "cntrl41"[label="Sequence(41)",shape=box];
	    "cntrl42"[label="Sequence(42)",shape=box];
	    "cntrl43"[label="Sequence(43)",shape=box];
	    "cntrl44"[label="ComputeIndex(44)"];
	    "cntrl45"[label="ComputeIndex(45)"];
	    "cntrl46"[label="Body(46)",shape=box];
	    "cntrl47"[label="Sequence(47)",shape=box];
	    "cntrl48"[label="Sequence(48)",shape=box];
	    "cntrl49"[label="Scope(49)"];
	    "cntrl50"[label="Sequence(50)",shape=box];
	    "cntrl51"[label="ComputeIndex(51)"];
	    "cntrl52"[label="ComputeIndex(52)"];
	    "cntrl53"[label="Body(53)",shape=box];
	    "cntrl54"[label="Sequence(54)",shape=box];
	    "cntrl55"[label="Sequence(55)",shape=box];
	    "cntrl56"[label="Scope(56)"];
	    "cntrl57"[label="Sequence(57)",shape=box];
	    "cntrl58"[label="Sequence(58)",shape=box];
	    "cntrl59"[label="ComputeIndex(59)"];
	    "cntrl60"[label="ComputeIndex(60)"];
	    "cntrl61"[label="Body(61)",shape=box];
	    "cntrl62"[label="Sequence(62)",shape=box];
	    "cntrl63"[label="Sequence(63)",shape=box];
	    "cntrl1" -> "cntrl24"
	    "cntrl1" -> "cntrl33"
	    "cntrl4" -> "cntrl13"
	    "cntrl9" -> "cntrl14"
	    "cntrl12" -> "cntrl50"
	    "cntrl13" -> "cntrl12"
	    "cntrl14" -> "cntrl12"
	    "cntrl19" -> "cntrl41"
	    "cntrl23" -> "cntrl25"
	    "cntrl23" -> "cntrl26"
	    "cntrl23" -> "cntrl29"
	    "cntrl24" -> "cntrl23"
	    "cntrl25" -> "cntrl4"
	    "cntrl26" -> "cntrl4"
	    "cntrl27" -> "cntrl30"
	    "cntrl28" -> "cntrl31"
	    "cntrl29" -> "cntrl27"
	    "cntrl30" -> "cntrl28"
	    "cntrl31" -> "cntrl2"
	    "cntrl32" -> "cntrl37"
	    "cntrl32" -> "cntrl57"
	    "cntrl33" -> "cntrl32"
	    "cntrl35" -> "cntrl38"
	    "cntrl36" -> "cntrl39"
	    "cntrl37" -> "cntrl35"
	    "cntrl38" -> "cntrl36"
	    "cntrl39" -> "cntrl17"
	    "cntrl40" -> "cntrl42"
	    "cntrl40" -> "cntrl43"
	    "cntrl40" -> "cntrl46"
	    "cntrl41" -> "cntrl40"
	    "cntrl42" -> "cntrl9"
	    "cntrl43" -> "cntrl9"
	    "cntrl44" -> "cntrl47"
	    "cntrl45" -> "cntrl48"
	    "cntrl46" -> "cntrl44"
	    "cntrl47" -> "cntrl45"
	    "cntrl48" -> "cntrl7"
	    "cntrl49" -> "cntrl53"
	    "cntrl50" -> "cntrl49"
	    "cntrl51" -> "cntrl54"
	    "cntrl52" -> "cntrl55"
	    "cntrl53" -> "cntrl51"
	    "cntrl54" -> "cntrl52"
	    "cntrl55" -> "cntrl15"
	    "cntrl56" -> "cntrl58"
	    "cntrl56" -> "cntrl61"
	    "cntrl57" -> "cntrl56"
	    "cntrl58" -> "cntrl19"
	    "cntrl59" -> "cntrl62"
	    "cntrl60" -> "cntrl63"
	    "cntrl61" -> "cntrl59"
	    "cntrl62" -> "cntrl60"
	    "cntrl63" -> "cntrl18"
	    }
	    "coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord3" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord6" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord24" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord25" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord28" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord29" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord38" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord126" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord127" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord128" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord129" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord130" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
	    "coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord10" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord13" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord58" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord59" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord136" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord137" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord138" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord139" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord140" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
	    "coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
	    "coord17" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
	    "coord17" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord21" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord64" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord65" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord68" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord69" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord79" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord80" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord141" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord142" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord143" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord144" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord145" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord10" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord11" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord88" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord90" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord91" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord99" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord100" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord107" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord109" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord131" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord132" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord133" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord88" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord111" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord112" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord146" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord147" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord148" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord149" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord150" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord38" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord126" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord127" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord130" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
	    "coord2" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord38" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord38" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord39" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord128" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord129" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord130" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl35" [style=dotted,weight=0,arrowsize=0]
	    "coord99" -> "cntrl35" [style=dotted,weight=0,arrowsize=0]
	    "coord100" -> "cntrl35" [style=dotted,weight=0,arrowsize=0]
	    "coord131" -> "cntrl35" [style=dotted,weight=0,arrowsize=0]
	    "coord132" -> "cntrl35" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl35" [style=dotted,weight=0,arrowsize=0]
	    "coord1" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord99" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord99" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord100" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord133" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord134" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord135" -> "cntrl36" [style=dotted,weight=0,arrowsize=0]
	    "coord58" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord59" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord136" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord137" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord140" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	    "coord58" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord58" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord59" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord138" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord139" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord140" -> "cntrl45" [style=dotted,weight=0,arrowsize=0]
	    "coord21" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord79" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord80" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord141" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord142" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord145" -> "cntrl51" [style=dotted,weight=0,arrowsize=0]
	    "coord21" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord79" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord79" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord80" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord143" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord144" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord145" -> "cntrl52" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl59" [style=dotted,weight=0,arrowsize=0]
	    "coord111" -> "cntrl59" [style=dotted,weight=0,arrowsize=0]
	    "coord112" -> "cntrl59" [style=dotted,weight=0,arrowsize=0]
	    "coord146" -> "cntrl59" [style=dotted,weight=0,arrowsize=0]
	    "coord147" -> "cntrl59" [style=dotted,weight=0,arrowsize=0]
	    "coord150" -> "cntrl59" [style=dotted,weight=0,arrowsize=0]
	    "coord86" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	    "coord111" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	    "coord111" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	    "coord112" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	    "coord148" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	    "coord149" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	    "coord150" -> "cntrl60" [style=dotted,weight=0,arrowsize=0]
	}).";

        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));
    }

    TEST_F(KernelGraphTest, Translate02)
    {
        auto command = commonCommand();

        auto one = Expression::literal(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph0 = translate(command);
        auto kgraph1 = lowerLinear(kgraph0, m_context);

        auto user0   = 1;
        auto block0  = 23;
        auto thread0 = 24;

        // given block id and thread id, compute regular (user) index for first (0) dataflow array
        auto block_id  = Expression::literal(2);
        auto thread_id = Expression::literal(33);

        auto exprs = kgraph1.coordinates.reverse(
            {block_id, thread_id}, {user0}, {block0, thread0}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "Multiply(Add(Multiply(2i, 32j), 33i), CommandArgument(Load_Linear_0_stride_0))");

        exprs = kgraph1.coordinates.reverse(
            {block_id, thread_id}, {user0}, {block0, thread0}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "Multiply(97j, CommandArgument(Load_Linear_0_stride_0))");
    }

#if 0
    TEST_F(KernelGraphTestGPU, GPU_Translate03)
    {
        TIMER(t_total, "Translate03");
        TIMER(t_gpu, "Translate03::GPU");
        TIMER(t_hip, "Translate03::HIP");

        TIC(t_total);

        auto command = std::make_shared<rocRoller::Command>();

        Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

        Operations::T_Load_Linear load_B(DataType::Int32, 1, 2);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));

        Operations::T_Execute execute;
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 2, 0)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 0)));

        command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

        Operations::T_Store_Linear store_C(1, 5);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));

        CommandKernel commandKernel(command, "Translate03");

        auto kgraph2 = commandKernel.getKernelGraph();

        auto kernelNode = kgraph2.control.getRootOperation();

        {
            auto expected = getTag(kernelNode);
            auto outputs
                = kgraph2.control.getOutputs<Body>(getTag(kernelNode));
            EXPECT_EQ(2, outputs.size());

            auto outputs2 = kgraph2.control.getOutputs<Sequence>(
                getTag(kernelNode));
            EXPECT_EQ(0, outputs2.size());

            auto outputs3
                = kgraph2.control.getOutputs(getTag(kernelNode), Body{});

            auto outputTags3 = kgraph2.control.getOutputTags(getTag(kernelNode),
                                                             Body{});

            EXPECT_EQ(outputs3.size(), outputTags3.size());
            for(size_t i = 0; i < outputs3.size(); i++)
            {
                EXPECT_EQ(getTag(outputs3[i]), outputTags3[i]);
            }

            EXPECT_EQ(getTag(outputs[0]), getTag(outputs3[0]));

            auto inputs1
                = kgraph2.control.getInputs<Body>(getTag(outputs.at(0)));
            ASSERT_EQ(1, inputs1.size());

            auto actual1 = getTag(inputs1.at(0));
            EXPECT_EQ(actual1, expected);

            auto inputs2 = kgraph2.control.getInputs(getTag(outputs.at(0)),
                                                     Body{});
            ASSERT_EQ(1, inputs2.size());

            auto inputTags2 = kgraph2.control.getInputTags(getTag(outputs.at(0)),
                                                           Body{});

            EXPECT_EQ(inputs2.size(), inputTags2.size());
            for(size_t i = 0; i < inputs2.size(); i++)
            {
                EXPECT_EQ(getTag(inputs2[i]), inputTags2[i]);
            }

            auto actual2 = getTag(inputs2.at(0));
            EXPECT_EQ(actual1, actual2);

            auto inputs3 = kgraph2.control.getInputs<Sequence>(
                getTag(outputs.at(0)));
            EXPECT_EQ(0, inputs3.size());

            auto inputs4 = kgraph2.control.getInputs<Initialize>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs4.size());

            auto inputs5 = kgraph2.control.getInputs<ForLoopIncrement>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs5.size());
        }

        {
            std::ostringstream msg;
            msg << kgraph2.control;

            std::ostringstream msg2;
            kgraph2.control.toDOT(msg2, "krn");

            EXPECT_EQ(msg.str(), msg2.str());
        }

        TIC(t_hip);
        size_t nx = 64;

        RandomGenerator random(17629u);
        auto            a = random.vector<int>(nx, -100, 100);
        auto            b = random.vector<int>(nx, -100, 100);

        auto user0 = make_shared_device(a);
        auto user2 = make_shared_device(b);
        auto user4 = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);
        TOC(t_hip);

        KernelArguments runtimeArgs;
        runtimeArgs.append("user0", user0.get());
        runtimeArgs.append("user1", nx);
        runtimeArgs.append("user2", nx);
        runtimeArgs.append("user3", (size_t)1);
        runtimeArgs.append("user4", user2.get());
        runtimeArgs.append("user5", nx);
        runtimeArgs.append("user6", nx);
        runtimeArgs.append("user7", (size_t)1);
        runtimeArgs.append("user8", user4.get());
        runtimeArgs.append("user9", nx);
        runtimeArgs.append("user10", (size_t)1);

        TIC(t_gpu);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        TOC(t_gpu);

        TIC(t_hip);
        ASSERT_THAT(hipMemcpy(r.data(), user4.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));
        TOC(t_hip);

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i] * (a[i] + b[i]);

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);

        TIC(t_hip);
        user0.reset();
        user2.reset();
        user4.reset();
        TOC(t_hip);

        TOC(t_total);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }
#endif

    void KernelGraphTestGPU::GPU_Translate04(bool reload)
    {
        RandomGenerator random(1263u);

        size_t nx = 64;

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;
        int beta  = 33;

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 1))); // b
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar({DataType::Int32, PointerType::PointerGlobal},
                                                 2))); // alpha
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar(DataType::Int32, 3))); // beta

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(4, 0, 2))); // alpha * a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(5, 1, 3))); // beta * b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(6, 4, 5))); // add above

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Linear(1, 6)));

        CommandKernel commandKernel(command, testKernelName());

        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", nx);
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        runtimeArgs.append("user2", d_alpha.get());

        runtimeArgs.append("user3", beta);

        runtimeArgs.append("user6", d_c.get());
        runtimeArgs.append("d_c_limit", nx);
        runtimeArgs.append("d_c_stride", (size_t)1);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        // launch again, using saved assembly
        auto assemblyFileName = m_context->assemblyFileName();

        if(reload)
        {
            commandKernel.loadKernelFromAssembly(assemblyFileName, testKernelName());
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        }

        std::vector<int> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = alpha * a[i] + beta * b[i];

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);

        if(reload)
        {
            // load, using bad kernel name
            EXPECT_THROW(commandKernel.loadKernelFromAssembly(assemblyFileName, "Translate04_BAD"),
                         FatalError);

            // load, using non-existant file
            EXPECT_THROW(
                commandKernel.loadKernelFromAssembly(assemblyFileName + "_bad", testKernelName()),
                FatalError);

            std::filesystem::remove(assemblyFileName);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate04)
    {
        GPU_Translate04(false);
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate04Debug)
    {
        // Make sure Debug mode doesn't introduce bad pointer
        // references in observers
        auto settings = Settings::getInstance();
        settings->set(Settings::LogLvl, LogLevel::Debug);
        GPU_Translate04(false);
        settings->reset();
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate04LoadAssembly)
    {
        GPU_Translate04(true);
    }

    TEST_F(KernelGraphTestGPU, GPU_Translate05)
    {
        auto command = std::make_shared<rocRoller::Command>();

        Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

        Operations::T_Store_Linear store_C(1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));

        CommandKernel commandKernel(command, "Translate05");

        size_t nx = 64;

        RandomGenerator random(135679u);
        auto            a = random.vector<int>(nx, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);

        KernelArguments runtimeArgs;
        runtimeArgs.append("d_a", d_a.get());
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);
        runtimeArgs.append("d_b", d_b.get());
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i];

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopy)
    {
        size_t nx  = 256; // tensor size x
        size_t ny  = 128; // tensor size y
        int    m   = 16; // macro tile size x
        int    n   = 4; // macro tile size y
        int    t_m = 4; // thread tile size x
        int    t_n = 2; // thread tile size y

        unsigned int workgroup_size_x = 4;
        unsigned int workgroup_size_y = 2;

        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0
                        && (size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator random(193674u);
        auto            a = random.vector<int>(nx * ny, -100, 100);
        auto            r = random.vector<int>(nx * ny, -100, 100);
        auto            x = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0)));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Int32, 2, 0)));

        KernelArguments runtimeArgs;

        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("b", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        params->setDimensionInfo(4, mac_tile);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileCopy", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx * ny; ++i)
        {
            x[i] = a[i];
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyLDS)
    {
        size_t nx  = 128; // tensor size x
        size_t ny  = 256; // tensor size y
        int    m   = 8; // macro tile size x
        int    n   = 16; // macro tile size y
        int    t_m = 2; // thread tile size x
        int    t_n = 8; // thread tile size y

        unsigned int workgroup_size_x = 4;
        unsigned int workgroup_size_y = 2;

        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0
                        && (size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator  random(193674u);
        auto             a = random.vector<int>(nx * ny, -100, 100);
        std::vector<int> r(nx * ny, 0);
        std::vector<int> x(nx * ny, 0);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0)));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Int32, 2, 0)));

        KernelArguments runtimeArgs;

        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("b", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        params->setDimensionInfo(4, mac_tile);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileCopy", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx * ny; ++i)
        {
            x[i] = a[i];
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileAdd)
    {
        size_t nx  = 256; // tensor size x
        size_t ny  = 512; // tensor size y
        int    m   = 8; // macro tile size x
        int    n   = 64; // macro tile size y
        int    t_m = 2; // thread tile size x
        int    t_n = 8; // thread tile size y

        uint workgroup_size_x = 4;
        uint workgroup_size_y = 8;

        AssertFatal(m > 0 && n > 0 && t_m > 0 && t_n > 0
                        && (size_t)m * n == t_m * t_n * workgroup_size_x * workgroup_size_y,
                    "MacroTile size mismatch");

        // each workgroup will get one tile; since workgroup_size matches m * n
        auto NX = std::make_shared<Expression::Expression>(nx / t_m); // number of work items x
        auto NY = std::make_shared<Expression::Expression>(ny / t_n); // number of work items y
        auto NZ = std::make_shared<Expression::Expression>(1u); // number of work items z

        RandomGenerator random(129674u);
        auto            a = random.vector<int>(nx * ny, -100, 100);
        auto            b = random.vector<int>(nx * ny, -100, 100);
        auto            r = random.vector<int>(nx * ny, -100, 100);
        auto            x = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<int>(nx * ny);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Int32, 2, 1))); // b

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(2, 0, 0))); // a + a
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 1, 1))); // b + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(4, 3, 2))); // 2a + 2b

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Int32, 2, 4))); // c

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", (size_t)nx * ny);
        runtimeArgs.append("d_a_size_0", (size_t)nx);
        runtimeArgs.append("d_a_size_1", (size_t)ny);
        runtimeArgs.append("d_a_stride_0", (size_t)ny);
        runtimeArgs.append("d_a_stride_1", (size_t)1);

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", (size_t)nx * ny);
        runtimeArgs.append("d_b_size_0", (size_t)nx);
        runtimeArgs.append("d_b_size_1", (size_t)ny);
        runtimeArgs.append("d_b_stride_0", (size_t)ny);
        runtimeArgs.append("d_b_stride_1", (size_t)1);

        runtimeArgs.append("user2", d_c.get());
        runtimeArgs.append("d_c_limit", (size_t)nx * ny);
        runtimeArgs.append("d_c_stride_0", (size_t)ny);
        runtimeArgs.append("d_c_stride_1", (size_t)1);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        // TODO: Add a "fill" operation on the kernel graph to propagate tile sizes where appropriate
        auto mac_tile_lds  = MacroTile({m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_vgpr = MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

        params->setDimensionInfo(4, mac_tile_lds);
        params->setDimensionInfo(11, mac_tile_vgpr);
        params->setDimensionInfo(15, mac_tile_vgpr);
        params->setDimensionInfo(17, mac_tile_vgpr);
        params->setDimensionInfo(19, mac_tile_vgpr);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "TensorTileAdd", params);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx * ny; ++i)
        {
            x[i] = a[i] + a[i] + b[i] + b[i];
        }

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileScale)
    {
        // matrix size: A is MxK; B is KxN; D is MxN
        int M = 1024;
        int N = 1024;

        // output macro tile size
        int mac_m = 64;
        int mac_n = 64;

        AssertFatal(M % mac_m == 0, "MacroTile size mismatch (M)");
        AssertFatal(N % mac_n == 0, "MacroTile size mismatch (N)");

        // wave tile sizes
        int wave_m = 32;
        int wave_n = 32;
        int wave_k = 2;
        int wave_b = 1;

        uint workgroup_size_x = 256;
        uint workgroup_size_y = 1;

        // one macro tile per workgroup
        uint num_workgroup_x = M / mac_m;
        uint num_workgroup_y = N / mac_n;

        auto NX = std::make_shared<Expression::Expression>(num_workgroup_x * workgroup_size_x);
        auto NY = std::make_shared<Expression::Expression>(num_workgroup_y * workgroup_size_y);
        auto NZ = std::make_shared<Expression::Expression>(1u);

        RandomGenerator random(61u);

        auto A = random.vector<float>(M * N, -1.f, 1.f);

        std::vector<float> B = {2.12f};

        auto d_A = make_shared_device(A);
        auto d_B = make_shared_device(B);
        auto d_D = make_shared_device<float>(M * N);

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar({DataType::Float, PointerType::PointerGlobal},
                                                 1))); // B

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(2, 0, 1))); // D = B * A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 2))); // D

        KernelArguments runtimeArgs;

        // tiled?
        runtimeArgs.append("A", d_A.get());
        runtimeArgs.append("d_a_limit", (size_t)M * N);
        runtimeArgs.append("d_a_size_0", (size_t)M);
        runtimeArgs.append("d_a_size_1", (size_t)N);
        runtimeArgs.append("d_a_stride_0", (size_t)1);
        runtimeArgs.append("d_a_stride_1", (size_t)M);

        runtimeArgs.append("B", d_B.get());

        runtimeArgs.append("D", d_D.get());
        runtimeArgs.append("d_d_limit", (size_t)M * N);
        runtimeArgs.append("d_d_stride_0", (size_t)1);
        runtimeArgs.append("d_d_stride_1", (size_t)M);

        auto params = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);

        auto mac_tile = MacroTile(
            {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(4, mac_tile);
        params->setDimensionInfo(11, mac_tile);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);

        auto WF  = Wavefront(-1, four, one);
        auto WFX = Wavefront(0, two, one);
        auto WFY = Wavefront(1, two, one);

        auto postParams = std::make_shared<CommandParameters>();

        std::vector<int> wavefront_ids = {37, 91};
        for(auto id : wavefront_ids)
        {
            postParams->setDimensionInfo(id, WF);
            postParams->setDimensionInfo(id - 2, WFX);
            postParams->setDimensionInfo(id - 1, WFY);
        }

        CommandKernel commandKernel(command, "BA", params, postParams);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        std::vector<float> D(M * N, 0.f);
        ASSERT_THAT(hipMemcpy(D.data(), d_D.get(), M * N * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        std::vector<float> c_D(M * N, 0.f);
        for(size_t i = 0; i < c_D.size(); ++i)
            c_D[i] = B[0] * A[i];

        double rnorm = relativeNorm(D, c_D);
        ASSERT_LT(rnorm, 2.e-6);
    }

    TEST_F(KernelGraphTest, CleanExpression)
    {
        VariableType doubleVal{DataType::Double, PointerType::Value};
        auto         command = std::make_shared<Command>();

        auto a = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));
        auto b = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Int32, PointerType::Value}));

        m_context->kernel()->addCommandArguments(command->getArguments());

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        auto clean_expr = cleanArguments(expr2, m_context->kernel());

        EXPECT_EQ(Expression::toString(clean_expr),
                  "Multiply(user_Int32_Value_1, Add(user_Int32_Value_0, user_Int32_Value_1))");
    }

    TEST_F(KernelGraphTest, CleanArguments)
    {
        auto command = commonCommand();

        m_context->kernel()->addCommandArguments(command->getArguments());

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto kgraph = translate(command);
        kgraph      = cleanArguments(kgraph, m_context->kernel());

        auto dot = kgraph.toDOT();
        EXPECT_THAT(dot, Not(HasSubstr("SubDimension{0, CommandArgument(Load_Linear_0_size_0)}")));
        EXPECT_THAT(dot, Not(HasSubstr("SubDimension{0, CommandArgument(Load_Linear_2_size_0)}")));
        EXPECT_THAT(
            dot, Not(HasSubstr("SubDimension{0, Linear{CommandArgument(Load_Linear_0_size_0)}")));
        EXPECT_THAT(
            dot, Not(HasSubstr("SubDimension{0, Linear{CommandArgument(Load_Linear_2_size_0)}")));

        EXPECT_THAT(dot, HasSubstr("SubDimension{0, Load_Linear_0_size_0}"));
        EXPECT_THAT(dot, HasSubstr("SubDimension{0, Load_Linear_2_size_0}"));
        EXPECT_THAT(dot, HasSubstr("Linear{Load_Linear_0_size_0}"));
        EXPECT_THAT(dot, HasSubstr("Linear{Load_Linear_2_size_0}"));
    }

    TEST_F(KernelGraphTest, Basic)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        // Control Graph
        int kernel_index = kgraph.control.addElement(Kernel());
        int loadA_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int loadB_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int body1_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadA_index});
        int body2_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadB_index});

        int op1_index       = kgraph.control.addElement(Assign());
        int sequence1_index = kgraph.control.addElement(Sequence(), {loadA_index}, {op1_index});
        int sequence2_index = kgraph.control.addElement(Sequence(), {loadB_index}, {op1_index});

        int op2_index       = kgraph.control.addElement(Assign());
        int sequence3_index = kgraph.control.addElement(Sequence(), {op1_index}, {op2_index});

        int op3_index       = kgraph.control.addElement(Assign());
        int sequence4_index = kgraph.control.addElement(Sequence(), {op1_index}, {op3_index});
        int sequence5_index = kgraph.control.addElement(Sequence(), {op2_index}, {op3_index});

        int storeC_index    = kgraph.control.addElement(StoreLinear());
        int sequence6_index = kgraph.control.addElement(Sequence(), {op3_index}, {storeC_index});

        // Coordinate Graph
        int u1_index       = kgraph.coordinates.addElement(User());
        int sd1_index      = kgraph.coordinates.addElement(SubDimension());
        int split1_index   = kgraph.coordinates.addElement(Split(), {u1_index}, {sd1_index});
        int linear1_index  = kgraph.coordinates.addElement(Linear());
        int flatten1_index = kgraph.coordinates.addElement(Flatten(), {sd1_index}, {linear1_index});
        int dataflow1_index
            = kgraph.coordinates.addElement(DataFlow(), {u1_index}, {linear1_index});

        int u2_index       = kgraph.coordinates.addElement(User());
        int sd2_index      = kgraph.coordinates.addElement(SubDimension());
        int split2_index   = kgraph.coordinates.addElement(Split(), {u2_index}, {sd2_index});
        int linear2_index  = kgraph.coordinates.addElement(Linear());
        int flatten2_index = kgraph.coordinates.addElement(Flatten(), {sd2_index}, {linear2_index});
        int dataflow2_index
            = kgraph.coordinates.addElement(DataFlow(), {u2_index}, {linear2_index});

        int linear3_index   = kgraph.coordinates.addElement(Linear());
        int dataflow3_index = kgraph.coordinates.addElement(
            DataFlow(), {linear1_index, linear2_index}, {linear3_index});
        int linear4_index = kgraph.coordinates.addElement(Linear());
        int dataflow4_index
            = kgraph.coordinates.addElement(DataFlow(), {linear3_index}, {linear4_index});
        int linear5i_index  = kgraph.coordinates.addElement(Linear());
        int dataflow5_index = kgraph.coordinates.addElement(
            DataFlow(), {linear3_index, linear4_index}, {linear5i_index});

        int linear5o_index = kgraph.coordinates.addElement(Linear());
        int makeoutput1_index
            = kgraph.coordinates.addElement(MakeOutput(), {linear5i_index}, {linear5o_index});
        int sd5o_index   = kgraph.coordinates.addElement(SubDimension(0));
        int split3_index = kgraph.coordinates.addElement(Split(), {linear5o_index}, {sd5o_index});
        int u5o_index    = kgraph.coordinates.addElement(User(""));
        int join1_index  = kgraph.coordinates.addElement(Join(), {sd5o_index}, {u5o_index});
        int dataflow6_index
            = kgraph.coordinates.addElement(DataFlow(), {linear5i_index}, {u5o_index});

        std::string expected = R".(
	    digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, NA}(2)"];
		"coord3"[label="Split(3)",shape=box];
		"coord4"[label="Linear{NA}(4)"];
		"coord5"[label="Flatten(5)",shape=box];
		"coord6"[label="DataFlow(6)",shape=box];
		"coord7"[label="User{NA}(7)"];
		"coord8"[label="SubDimension{0, NA}(8)"];
		"coord9"[label="Split(9)",shape=box];
		"coord10"[label="Linear{NA}(10)"];
		"coord11"[label="Flatten(11)",shape=box];
		"coord12"[label="DataFlow(12)",shape=box];
		"coord13"[label="Linear{NA}(13)"];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="Linear{NA}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord17"[label="Linear{NA}(17)"];
		"coord18"[label="DataFlow(18)",shape=box];
		"coord19"[label="Linear{NA}(19)"];
		"coord20"[label="MakeOutput(20)",shape=box];
		"coord21"[label="SubDimension{0, NA}(21)"];
		"coord22"[label="Split(22)",shape=box];
		"coord23"[label="User{NA}(23)"];
		"coord24"[label="Join(24)",shape=box];
		"coord25"[label="DataFlow(25)",shape=box];
		"coord1" -> "coord3"
		"coord1" -> "coord6"
		"coord2" -> "coord5"
		"coord3" -> "coord2"
		"coord4" -> "coord14"
		"coord5" -> "coord4"
		"coord6" -> "coord4"
		"coord7" -> "coord9"
		"coord7" -> "coord12"
		"coord8" -> "coord11"
		"coord9" -> "coord8"
		"coord10" -> "coord14"
		"coord11" -> "coord10"
		"coord12" -> "coord10"
		"coord13" -> "coord16"
		"coord13" -> "coord18"
		"coord14" -> "coord13"
		"coord15" -> "coord18"
		"coord16" -> "coord15"
		"coord17" -> "coord20"
		"coord17" -> "coord25"
		"coord18" -> "coord17"
		"coord19" -> "coord22"
		"coord20" -> "coord19"
		"coord21" -> "coord24"
		"coord22" -> "coord21"
		"coord24" -> "coord23"
		"coord25" -> "coord23"
		{
		    rank=same
		    "coord4"->"coord10"[style=invis]
		    rankdir=LR
		}
		{
		    rank=same
		    "coord13"->"coord15"[style=invis]
		    rankdir=LR
		}

		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadLinear(2)"];
		"cntrl3"[label="LoadLinear(3)"];
		"cntrl4"[label="Body(4)",shape=box];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="Assign Literal nullptr(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="Assign Literal nullptr(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="Assign Literal nullptr(11)"];
		"cntrl12"[label="Sequence(12)",shape=box];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="StoreLinear(14)"];
		"cntrl15"[label="Sequence(15)",shape=box];
		"cntrl1" -> "cntrl4"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl8"
		"cntrl4" -> "cntrl2"
		"cntrl5" -> "cntrl3"
		"cntrl6" -> "cntrl10"
		"cntrl6" -> "cntrl12"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		"cntrl9" -> "cntrl13"
		"cntrl10" -> "cntrl9"
		"cntrl11" -> "cntrl15"
		"cntrl12" -> "cntrl11"
		"cntrl13" -> "cntrl11"
		"cntrl15" -> "cntrl14"
	    }
            }
        ).";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(kgraph.toDOT()));
    }

    TEST_F(KernelGraphTest, UpdateParamsTMul)
    {
        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 0))); // A
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Tiled(DataType::Float, 2, 1))); // B
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Mul(2, 0, 1)));

        auto kgraph0 = translate(command);

        std::string expected0 = R".(
		digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{NA}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{NA}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="MacroTile{NA}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord16"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord16"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord16" -> "coord15"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord4"->"coord11"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
		"cntrl5" -> "cntrl4"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		}
		}
	).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));

        // macro tile sizes
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        auto mac_tile_0 = MacroTile({mac_m, mac_k}, MemoryType::VGPR); // A
        auto mac_tile_1 = MacroTile({mac_k, mac_n}, MemoryType::VGPR); // B

        auto params = std::make_shared<CommandParameters>();

        params->setDimensionInfo(4, mac_tile_0);
        params->setDimensionInfo(11, mac_tile_1);

        kgraph0 = updateParameters(kgraph0, params);

        std::string expected1 = R".(
		digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, CommandArgument(Load_Tiled_0_size_0)}(2)"];
		"coord3"[label="SubDimension{1, CommandArgument(Load_Tiled_0_size_1)}(3)"];
		"coord4"[label="MacroTile{64,64}(4)"];
		"coord5"[label="Split(5)",shape=box];
		"coord6"[label="ConstructTensorTile(6)",shape=box];
		"coord7"[label="DataFlow(7)",shape=box];
		"coord8"[label="User{NA}(8)"];
		"coord9"[label="SubDimension{0, CommandArgument(Load_Tiled_1_size_0)}(9)"];
		"coord10"[label="SubDimension{1, CommandArgument(Load_Tiled_1_size_1)}(10)"];
		"coord11"[label="MacroTile{64,64}(11)"];
		"coord12"[label="Split(12)",shape=box];
		"coord13"[label="ConstructTensorTile(13)",shape=box];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="MacroTile{NA}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord1" -> "coord5"
		"coord1" -> "coord7"
		"coord2" -> "coord6"
		"coord3" -> "coord6"
		"coord4" -> "coord16"
		"coord5" -> "coord2"
		"coord5" -> "coord3"
		"coord6" -> "coord4"
		"coord7" -> "coord4"
		"coord8" -> "coord12"
		"coord8" -> "coord14"
		"coord9" -> "coord13"
		"coord10" -> "coord13"
		"coord11" -> "coord16"
		"coord12" -> "coord9"
		"coord12" -> "coord10"
		"coord13" -> "coord11"
		"coord14" -> "coord11"
		"coord16" -> "coord15"
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord2"->"coord3"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord9"->"coord10"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord4"->"coord11"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {label = "Control Graph";
		"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl8"
		"cntrl5" -> "cntrl4"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		}
		}
	).";

        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph0.toDOT()));
    }

}
