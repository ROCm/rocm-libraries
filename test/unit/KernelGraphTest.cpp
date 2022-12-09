
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
#include <rocRoller/KernelGraph/CoordGraph/CoordinateHypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "KernelGraph/KernelHypergraph.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
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

        m_context->kernel()->setKernelGraphMeta(std::make_shared<KernelGraph::KernelHypergraph>(kgraph));

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

        m_context->kernel()->setKernelGraphMeta(std::make_shared<KernelGraph::KernelHypergraph>(kgraph));

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
        auto kgraph0 = KernelGraph::translate(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto visitor = KernelGraph::BaseGraphVisitor(m_context);
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
                subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="ElementOp(10, 4)(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="ElementOp(13, -1)(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="ElementOp(13, 15)(11)"];
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
                subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="ElementOp(10, 4)(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="ElementOp(13, -1)(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="ElementOp(13, 15)(11)"];
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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadVGPR(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadVGPR(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="ElementOp(16, 22)(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="ElementOp(28, -1)(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="ElementOp(28, 30)(11)"];
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

        auto kgraph1 = KernelGraph::lowerLinear(kgraph0, m_context);
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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="ForLoopOp: LessThan(DataFlowTag(10), 16i)(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="Assign SGPR 0i(4)"];
		"cntrl5"[label="Initialize(5)",shape=box];
		"cntrl6"[label="Assign SGPR Add(DataFlowTag(10), 1i)(6)"];
		"cntrl7"[label="ForLoopIncrement(7)",shape=box];
		"cntrl8"[label="LoadVGPR(8)"];
		"cntrl9"[label="Body(9)",shape=box];
		"cntrl10"[label="LoadVGPR(10)"];
		"cntrl11"[label="Body(11)",shape=box];
		"cntrl12"[label="ElementOp(13, 25)(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="ElementOp(28, -1)(15)"];
		"cntrl16"[label="Sequence(16)",shape=box];
		"cntrl17"[label="ElementOp(28, 30)(17)"];
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

        auto kgraph2 = KernelGraph::lowerLinearLoop(kgraph1, loopSizeExpr, m_context);
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

        auto kgraph0 = KernelGraph::translate(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(4, 11)(6)"];
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

        auto kgraph0 = KernelGraph::translate(command);

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(std::holds_alternative<KernelGraph::CoordGraph::User>(
                std::get<KernelGraph::CoordGraph::Dimension>(kgraph0.coordinates.getElement(id))));
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
                subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadVGPR(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadVGPR(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="ElementOp(5, 2)(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="ElementOp(7, -1)(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="ElementOp(7, 9)(11)"];
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

        auto kgraph0 = KernelGraph::translate(command);

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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
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
		"cntrl12"[label="TensorContraction(4, 11)(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="ElementOp(23, 28)(15)"];
		"cntrl16"[label="Sequence(16)",shape=box];
		"cntrl17"[label="Sequence(17)",shape=box];
		"cntrl18"[label="ElementOp(26, 18)(18)"];
		"cntrl19"[label="Sequence(19)",shape=box];
		"cntrl20"[label="Sequence(20)",shape=box];
		"cntrl21"[label="ElementOp(30, 32)(21)"];
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

        auto mac_tile_A = KernelGraph::CoordGraph::MacroTile(
            {mac_m, mac_k}, LayoutType::MATRIX_A, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_B = KernelGraph::CoordGraph::MacroTile(
            {mac_k, mac_n}, LayoutType::MATRIX_B, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_C = KernelGraph::CoordGraph::MacroTile(
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

        auto kgraph1 = KernelGraph::lowerTile(kgraph0, params, m_context);

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
		"coord44"[label="Flatten(44)",shape=box];
		"coord45"[label="Tile(45)",shape=box];
		"coord46"[label="Tile(46)",shape=box];
		"coord47"[label="PassThrough(47)",shape=box];
		"coord49"[label="Workitem{0, NA}(49)"];
		"coord50"[label="WaveTile{256i}(50)"];
		"coord51"[label="WaveTileNumber{0, 1j}(51)"];
		"coord52"[label="WaveTileNumber{1, 1j}(52)"];
		"coord53"[label="WaveTileIndex{0, 32j}(53)"];
		"coord54"[label="WaveTileIndex{1, 8j}(54)"];
		"coord55"[label="Tile(55)",shape=box];
		"coord56"[label="Tile(56)",shape=box];
		"coord57"[label="Wavefront{0, NA}(57)"];
		"coord58"[label="Wavefront{1, NA}(58)"];
		"coord59"[label="Wavefront{-1, NA}(59)"];
		"coord60"[label="Lane{32j}(60)"];
		"coord61"[label="VGPR{8j}(61)"];
		"coord62"[label="Flatten(62)",shape=box];
		"coord63"[label="Flatten(63)",shape=box];
		"coord64"[label="BlockNumber{1j}(64)"];
		"coord65"[label="BlockIndex{32j}(65)"];
		"coord66"[label="Flatten(66)",shape=box];
		"coord67"[label="WaveTilePerWorkGroup{0, 2j}(67)"];
		"coord68"[label="WaveTilePerWorkGroup{1, 2j}(68)"];
		"coord69"[label="Flatten(69)",shape=box];
		"coord70"[label="Tile(70)",shape=box];
		"coord71"[label="PassThrough(71)",shape=box];
		"coord72"[label="Tile(72)",shape=box];
		"coord73"[label="MacroTileNumber{0, 1j}(73)"];
		"coord74"[label="MacroTileNumber{1, 1j}(74)"];
		"coord75"[label="MacroTileIndex{0, 16j}(75)"];
		"coord76"[label="MacroTileIndex{1, 256j}(76)"];
		"coord78"[label="Workgroup{1, NA}(78)"];
		"coord79"[label="Flatten(79)",shape=box];
		"coord80"[label="Tile(80)",shape=box];
		"coord81"[label="Tile(81)",shape=box];
		"coord83"[label="PassThrough(83)",shape=box];
		"coord84"[label="Workitem{0, NA}(84)"];
		"coord85"[label="WaveTile{256i}(85)"];
		"coord86"[label="WaveTileNumber{0, 1j}(86)"];
		"coord87"[label="WaveTileNumber{1, 1j}(87)"];
		"coord88"[label="WaveTileIndex{0, 8j}(88)"];
		"coord89"[label="WaveTileIndex{1, 32j}(89)"];
		"coord90"[label="Tile(90)",shape=box];
		"coord91"[label="Tile(91)",shape=box];
		"coord92"[label="Wavefront{0, NA}(92)"];
		"coord93"[label="Wavefront{1, NA}(93)"];
		"coord94"[label="Wavefront{-1, NA}(94)"];
		"coord95"[label="Lane{32j}(95)"];
		"coord96"[label="VGPR{8j}(96)"];
		"coord97"[label="Flatten(97)",shape=box];
		"coord98"[label="Flatten(98)",shape=box];
		"coord99"[label="BlockNumber{1j}(99)"];
		"coord100"[label="BlockIndex{32j}(100)"];
		"coord101"[label="Flatten(101)",shape=box];
		"coord102"[label="WaveTilePerWorkGroup{0, 2j}(102)"];
		"coord103"[label="WaveTilePerWorkGroup{1, 2j}(103)"];
		"coord104"[label="Flatten(104)",shape=box];
		"coord105"[label="Tile(105)",shape=box];
		"coord106"[label="PassThrough(106)",shape=box];
		"coord107"[label="Tile(107)",shape=box];
		"coord108"[label="MacroTileNumber{0, 1j}(108)"];
		"coord109"[label="MacroTileNumber{1, 1j}(109)"];
		"coord110"[label="MacroTileIndex{0, 128j}(110)"];
		"coord111"[label="MacroTileIndex{1, 256j}(111)"];
		"coord112"[label="Workgroup{0, NA}(112)"];
		"coord113"[label="Workgroup{1, NA}(113)"];
		"coord114"[label="Flatten(114)",shape=box];
		"coord115"[label="Tile(115)",shape=box];
		"coord116"[label="Tile(116)",shape=box];
		"coord117"[label="PassThrough(117)",shape=box];
		"coord118"[label="PassThrough(118)",shape=box];
		"coord119"[label="Workitem{0, NA}(119)"];
		"coord120"[label="WaveTile{1024i}(120)"];
		"coord121"[label="WaveTileNumber{0, 1j}(121)"];
		"coord122"[label="WaveTileNumber{1, 1j}(122)"];
		"coord123"[label="WaveTileIndex{0, 32j}(123)"];
		"coord124"[label="WaveTileIndex{1, 32j}(124)"];
		"coord125"[label="Tile(125)",shape=box];
		"coord126"[label="Tile(126)",shape=box];
		"coord127"[label="Wavefront{0, NA}(127)"];
		"coord128"[label="Wavefront{1, NA}(128)"];
		"coord129"[label="Wavefront{-1, NA}(129)"];
		"coord130"[label="Lane{32j}(130)"];
		"coord131"[label="VGPR{32j}(131)"];
		"coord132"[label="Flatten(132)",shape=box];
		"coord133"[label="Flatten(133)",shape=box];
		"coord134"[label="BlockNumber{1j}(134)"];
		"coord135"[label="BlockIndex{32j}(135)"];
		"coord136"[label="Flatten(136)",shape=box];
		"coord137"[label="WaveTilePerWorkGroup{0, 2j}(137)"];
		"coord138"[label="WaveTilePerWorkGroup{1, 2j}(138)"];
		"coord139"[label="VGPRBlockNumber{8j}(139)"];
		"coord140"[label="VGPRBlockIndex{4j}(140)"];
		"coord141"[label="LANEBlockNumber{8j}(141)"];
		"coord142"[label="LANEBlockIndex{4j}(142)"];
		"coord143"[label="LinearBlock{64j}(143)"];
		"coord144"[label="RowBlock{8j}(144)"];
		"coord145"[label="ColBlock{8j}(145)"];
		"coord146"[label="Flatten(146)",shape=box];
		"coord147"[label="Tile(147)",shape=box];
		"coord148"[label="Tile(148)",shape=box];
		"coord149"[label="Flatten(149)",shape=box];
		"coord150"[label="Tile(150)",shape=box];
		"coord151"[label="Flatten(151)",shape=box];
		"coord152"[label="Flatten(152)",shape=box];
		"coord153"[label="Tile(153)",shape=box];
		"coord154"[label="Tile(154)",shape=box];
		"coord155"[label="MacroTileNumber{0, 1j}(155)"];
		"coord156"[label="MacroTileNumber{1, 1j}(156)"];
		"coord157"[label="MacroTileIndex{0, 128j}(157)"];
		"coord158"[label="MacroTileIndex{1, 256j}(158)"];
		"coord159"[label="Workgroup{0, NA}(159)"];
		"coord160"[label="Workgroup{1, NA}(160)"];
		"coord161"[label="Workitem{0, 1j}(161)"];
		"coord162"[label="Flatten(162)",shape=box];
		"coord163"[label="Flatten(163)",shape=box];
		"coord164"[label="Flatten(164)",shape=box];
		"coord165"[label="PassThrough(165)",shape=box];
		"coord166"[label="PassThrough(166)",shape=box];
		"coord167"[label="WaveTile{8192i}(167)"];
		"coord168"[label="WaveTileNumber{0, 1j}(168)"];
		"coord169"[label="WaveTileNumber{1, 1j}(169)"];
		"coord170"[label="WaveTileIndex{0, 32j}(170)"];
		"coord171"[label="WaveTileIndex{1, 32j}(171)"];
		"coord172"[label="Join(172)",shape=box];
		"coord173"[label="VGPRBlockNumber{8j}(173)"];
		"coord174"[label="VGPRBlockIndex{4j}(174)"];
		"coord175"[label="LANEBlockNumber{8j}(175)"];
		"coord176"[label="LANEBlockIndex{4j}(176)"];
		"coord177"[label="LinearBlock{64j}(177)"];
		"coord178"[label="RowBlock{8j}(178)"];
		"coord179"[label="ColBlock{8j}(179)"];
		"coord180"[label="Flatten(180)",shape=box];
		"coord181"[label="Flatten(181)",shape=box];
		"coord182"[label="Wavefront{0, NA}(182)"];
		"coord183"[label="Wavefront{1, NA}(183)"];
		"coord184"[label="Wavefront{-1, NA}(184)"];
		"coord185"[label="Tile(185)",shape=box];
		"coord186"[label="Lane{32j}(186)"];
		"coord187"[label="VGPR{32j}(187)"];
		"coord188"[label="Tile(188)",shape=box];
		"coord189"[label="Tile(189)",shape=box];
		"coord190"[label="Flatten(190)",shape=box];
		"coord191"[label="Tile(191)",shape=box];
		"coord192"[label="WaveTilePerWorkGroup{0, 2j}(192)"];
		"coord193"[label="WaveTilePerWorkGroup{1, 2j}(193)"];
		"coord194"[label="Flatten(194)",shape=box];
		"coord195"[label="Flatten(195)",shape=box];
		"coord196"[label="Flatten(196)",shape=box];
		"coord197"[label="Flatten(197)",shape=box];
		"coord198"[label="Tile(198)",shape=box];
		"coord199"[label="Linear{Divide(CommandArgument(Load_Tiled_0_size_1), 16j)}(199)"];
		"coord200"[label="ForLoop{NA}(200)"];
		"coord201"[label="DataFlow(201)",shape=box];
		"coord202"[label="PassThrough(202)",shape=box];
		"coord203"[label="PassThrough(203)",shape=box];
		"coord204"[label="Linear{2j}(204)"];
		"coord205"[label="ForLoop{NA}(205)"];
		"coord206"[label="DataFlow(206)",shape=box];
		"coord207"[label="Linear{2j}(207)"];
		"coord208"[label="ForLoop{NA}(208)"];
		"coord209"[label="DataFlow(209)",shape=box];
		"coord210"[label="PassThrough(210)",shape=box];
		"coord211"[label="PassThrough(211)",shape=box];
		"coord212"[label="PassThrough(212)",shape=box];
		"coord213"[label="PassThrough(213)",shape=box];
		"coord214"[label="PassThrough(214)",shape=box];
		"coord215"[label="PassThrough(215)",shape=box];
		"coord216"[label="PassThrough(216)",shape=box];
		"coord217"[label="PassThrough(217)",shape=box];
		"coord218"[label="Offset(218)",shape=box];
		"coord219"[label="Stride(219)",shape=box];
		"coord220"[label="Offset(220)",shape=box];
		"coord221"[label="Stride(221)",shape=box];
		"coord222"[label="Offset(222)",shape=box];
		"coord223"[label="Stride(223)",shape=box];
		"coord224"[label="Offset(224)",shape=box];
		"coord225"[label="Stride(225)",shape=box];
		"coord226"[label="Offset(226)",shape=box];
		"coord227"[label="Stride(227)",shape=box];
		"coord228"[label="Offset(228)",shape=box];
		"coord229"[label="Stride(229)",shape=box];
		"coord230"[label="Offset(230)",shape=box];
		"coord231"[label="Stride(231)",shape=box];
		"coord232"[label="Offset(232)",shape=box];
		"coord233"[label="Stride(233)",shape=box];
		"coord234"[label="Offset(234)",shape=box];
		"coord235"[label="Stride(235)",shape=box];
		"coord236"[label="Offset(236)",shape=box];
		"coord237"[label="Stride(237)",shape=box];
		"coord1" -> "coord10"
		"coord1" -> "coord12"
		"coord1" -> "coord218"
		"coord1" -> "coord219"
		"coord1" -> "coord220"
		"coord1" -> "coord221"
		"coord1" -> "coord222"
		"coord1" -> "coord223"
		"coord2" -> "coord15"
		"coord2" -> "coord17"
		"coord2" -> "coord224"
		"coord2" -> "coord225"
		"coord2" -> "coord226"
		"coord2" -> "coord227"
		"coord2" -> "coord228"
		"coord2" -> "coord229"
		"coord3" -> "coord26"
		"coord3" -> "coord28"
		"coord3" -> "coord230"
		"coord3" -> "coord231"
		"coord3" -> "coord232"
		"coord3" -> "coord233"
		"coord4" -> "coord7"
		"coord5" -> "coord23"
		"coord6" -> "coord21"
		"coord7" -> "coord6"
		"coord8" -> "coord45"
		"coord9" -> "coord46"
		"coord10" -> "coord8"
		"coord10" -> "coord9"
		"coord11" -> "coord19"
		"coord12" -> "coord11"
		"coord13" -> "coord80"
		"coord14" -> "coord81"
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
		"coord24" -> "coord115"
		"coord25" -> "coord116"
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
		"coord35" -> "coord234"
		"coord35" -> "coord235"
		"coord35" -> "coord236"
		"coord35" -> "coord237"
		"coord36" -> "coord35"
		"coord37" -> "coord35"
		"coord38" -> "coord47"
		"coord39" -> "coord202"
		"coord40" -> "coord44"
		"coord40" -> "coord55"
		"coord41" -> "coord56"
		"coord41" -> "coord44"
		"coord44" -> "coord11"
		"coord45" -> "coord38"
		"coord45" -> "coord40"
		"coord46" -> "coord39"
		"coord46" -> "coord41"
		"coord47" -> "coord42"
		"coord51" -> "coord72"
		"coord53" -> "coord71"
		"coord53" -> "coord69"
		"coord54" -> "coord69"
		"coord54" -> "coord70"
		"coord55" -> "coord51"
		"coord55" -> "coord53"
		"coord56" -> "coord52"
		"coord56" -> "coord54"
		"coord57" -> "coord62"
		"coord58" -> "coord62"
		"coord59" -> "coord63"
		"coord60" -> "coord63"
		"coord62" -> "coord59"
		"coord63" -> "coord49"
		"coord64" -> "coord66"
		"coord65" -> "coord66"
		"coord66" -> "coord60"
		"coord67" -> "coord210"
		"coord68" -> "coord214"
		"coord69" -> "coord50"
		"coord70" -> "coord64"
		"coord70" -> "coord61"
		"coord71" -> "coord65"
		"coord72" -> "coord57"
		"coord72" -> "coord67"
		"coord73" -> "coord203"
		"coord74" -> "coord83"
		"coord75" -> "coord79"
		"coord75" -> "coord90"
		"coord76" -> "coord91"
		"coord76" -> "coord79"
		"coord79" -> "coord16"
		"coord80" -> "coord73"
		"coord80" -> "coord75"
		"coord81" -> "coord74"
		"coord81" -> "coord76"
		"coord83" -> "coord78"
		"coord87" -> "coord107"
		"coord88" -> "coord104"
		"coord88" -> "coord105"
		"coord89" -> "coord106"
		"coord89" -> "coord104"
		"coord90" -> "coord86"
		"coord90" -> "coord88"
		"coord91" -> "coord87"
		"coord91" -> "coord89"
		"coord92" -> "coord97"
		"coord93" -> "coord97"
		"coord94" -> "coord98"
		"coord95" -> "coord98"
		"coord97" -> "coord94"
		"coord98" -> "coord84"
		"coord99" -> "coord101"
		"coord100" -> "coord101"
		"coord101" -> "coord95"
		"coord102" -> "coord211"
		"coord103" -> "coord215"
		"coord104" -> "coord85"
		"coord105" -> "coord99"
		"coord105" -> "coord96"
		"coord106" -> "coord100"
		"coord107" -> "coord93"
		"coord107" -> "coord103"
		"coord108" -> "coord117"
		"coord109" -> "coord118"
		"coord110" -> "coord114"
		"coord110" -> "coord125"
		"coord111" -> "coord126"
		"coord111" -> "coord114"
		"coord114" -> "coord27"
		"coord115" -> "coord108"
		"coord115" -> "coord110"
		"coord116" -> "coord109"
		"coord116" -> "coord111"
		"coord117" -> "coord112"
		"coord118" -> "coord113"
		"coord121" -> "coord153"
		"coord122" -> "coord154"
		"coord123" -> "coord146"
		"coord123" -> "coord147"
		"coord124" -> "coord148"
		"coord124" -> "coord146"
		"coord125" -> "coord121"
		"coord125" -> "coord123"
		"coord126" -> "coord122"
		"coord126" -> "coord124"
		"coord127" -> "coord132"
		"coord128" -> "coord132"
		"coord129" -> "coord133"
		"coord130" -> "coord133"
		"coord132" -> "coord129"
		"coord133" -> "coord119"
		"coord134" -> "coord136"
		"coord135" -> "coord136"
		"coord136" -> "coord130"
		"coord137" -> "coord212"
		"coord138" -> "coord216"
		"coord139" -> "coord151"
		"coord140" -> "coord151"
		"coord141" -> "coord152"
		"coord142" -> "coord152"
		"coord143" -> "coord150"
		"coord144" -> "coord149"
		"coord145" -> "coord149"
		"coord146" -> "coord120"
		"coord147" -> "coord144"
		"coord147" -> "coord140"
		"coord148" -> "coord145"
		"coord148" -> "coord142"
		"coord149" -> "coord143"
		"coord150" -> "coord139"
		"coord150" -> "coord141"
		"coord151" -> "coord131"
		"coord152" -> "coord130"
		"coord153" -> "coord127"
		"coord153" -> "coord137"
		"coord154" -> "coord128"
		"coord154" -> "coord138"
		"coord155" -> "coord163"
		"coord156" -> "coord164"
		"coord157" -> "coord162"
		"coord157" -> "coord163"
		"coord158" -> "coord162"
		"coord158" -> "coord164"
		"coord159" -> "coord165"
		"coord160" -> "coord166"
		"coord161" -> "coord198"
		"coord162" -> "coord31"
		"coord163" -> "coord33"
		"coord164" -> "coord34"
		"coord165" -> "coord155"
		"coord166" -> "coord156"
		"coord168" -> "coord180"
		"coord169" -> "coord181"
		"coord170" -> "coord172"
		"coord170" -> "coord180"
		"coord171" -> "coord172"
		"coord171" -> "coord181"
		"coord172" -> "coord167"
		"coord173" -> "coord190"
		"coord174" -> "coord196"
		"coord175" -> "coord190"
		"coord176" -> "coord197"
		"coord177" -> "coord191"
		"coord178" -> "coord196"
		"coord179" -> "coord197"
		"coord180" -> "coord157"
		"coord181" -> "coord158"
		"coord182" -> "coord194"
		"coord183" -> "coord195"
		"coord184" -> "coord185"
		"coord185" -> "coord182"
		"coord185" -> "coord183"
		"coord186" -> "coord189"
		"coord187" -> "coord188"
		"coord188" -> "coord173"
		"coord188" -> "coord174"
		"coord189" -> "coord175"
		"coord189" -> "coord176"
		"coord190" -> "coord177"
		"coord191" -> "coord178"
		"coord191" -> "coord179"
		"coord192" -> "coord194"
		"coord193" -> "coord195"
		"coord194" -> "coord168"
		"coord195" -> "coord169"
		"coord196" -> "coord170"
		"coord197" -> "coord171"
		"coord198" -> "coord184"
		"coord198" -> "coord186"
		"coord199" -> "coord201"
		"coord201" -> "coord200"
		"coord202" -> "coord200"
		"coord203" -> "coord200"
		"coord204" -> "coord206"
		"coord205" -> "coord213"
		"coord206" -> "coord205"
		"coord207" -> "coord209"
		"coord208" -> "coord217"
		"coord209" -> "coord208"
		"coord210" -> "coord205"
		"coord211" -> "coord205"
		"coord212" -> "coord205"
		"coord213" -> "coord192"
		"coord214" -> "coord208"
		"coord215" -> "coord208"
		"coord216" -> "coord208"
		"coord217" -> "coord193"
		"coord218" -> "coord39"
		"coord219" -> "coord39"
		"coord220" -> "coord52"
		"coord221" -> "coord52"
		"coord222" -> "coord61"
		"coord223" -> "coord61"
		"coord224" -> "coord73"
		"coord225" -> "coord73"
		"coord226" -> "coord86"
		"coord227" -> "coord86"
		"coord228" -> "coord96"
		"coord229" -> "coord96"
		"coord230" -> "coord139"
		"coord231" -> "coord139"
		"coord232" -> "coord140"
		"coord233" -> "coord140"
		"coord234" -> "coord173"
		"coord235" -> "coord173"
		"coord236" -> "coord174"
		"coord237" -> "coord174"
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
		"coord40"->"coord41"[style=invis]
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
		"coord51"->"coord53"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord52"->"coord54"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord57"->"coord58"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord59"->"coord60"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord64"->"coord65"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord54"->"coord53"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord64"->"coord61"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord57"->"coord67"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord75"->"coord76"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord73"->"coord75"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord74"->"coord76"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord86"->"coord88"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord87"->"coord89"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord92"->"coord93"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord94"->"coord95"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord99"->"coord100"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord88"->"coord89"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord99"->"coord96"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord93"->"coord103"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord110"->"coord111"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord108"->"coord110"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord109"->"coord111"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord121"->"coord123"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord122"->"coord124"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord127"->"coord128"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord129"->"coord130"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord134"->"coord135"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord123"->"coord124"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord144"->"coord140"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord145"->"coord142"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord144"->"coord145"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord139"->"coord141"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord139"->"coord140"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord141"->"coord142"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord127"->"coord137"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord128"->"coord138"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord157"->"coord158"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord155"->"coord157"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord156"->"coord158"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord170"->"coord171"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord168"->"coord170"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord169"->"coord171"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord182"->"coord183"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord173"->"coord174"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord175"->"coord176"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord173"->"coord175"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord178"->"coord179"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord182"->"coord192"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord183"->"coord193"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord178"->"coord174"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord179"->"coord176"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord184"->"coord186"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadVGPR(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl6"[label="LoadTiled(6)"];
		"cntrl11"[label="ElementOp(6, 18)(11)"];
		"cntrl12"[label="Sequence(12)",shape=box];
		"cntrl14"[label="LoadVGPR(14)"];
		"cntrl15"[label="Body(15)",shape=box];
		"cntrl16"[label="LoadTiled(16)"];
		"cntrl18"[label="ElementOp(22, 27)(18)"];
		"cntrl19"[label="Sequence(19)",shape=box];
		"cntrl21"[label="ElementOp(20, 29)(21)"];
		"cntrl22"[label="Sequence(22)",shape=box];
		"cntrl23"[label="Sequence(23)",shape=box];
		"cntrl24"[label="StoreTiled(24)"];
		"cntrl26"[label="ForLoopOp: LessThan(DataFlowTag(199), Divide(CommandArgument(Load_Tiled_0_size_1), 16j))(26)"];
		"cntrl27"[label="Assign SGPR 0l(27)"];
		"cntrl28"[label="Assign SGPR Add(DataFlowTag(199), 1j)(28)"];
		"cntrl29"[label="Initialize(29)",shape=box];
		"cntrl30"[label="ForLoopIncrement(30)",shape=box];
		"cntrl31"[label="Multiply(31)"];
		"cntrl32"[label="Assign ACCVGPR 0.00000f(32)"];
		"cntrl33"[label="Initialize(33)",shape=box];
		"cntrl34"[label="Body(34)",shape=box];
		"cntrl35"[label="Body(35)",shape=box];
		"cntrl36"[label="Body(36)",shape=box];
		"cntrl37"[label="ForLoopOp: LessThan(DataFlowTag(204), 2j)(37)"];
		"cntrl38"[label="Assign SGPR 0j(38)"];
		"cntrl39"[label="Assign SGPR Add(DataFlowTag(204), 1j)(39)"];
		"cntrl40"[label="Initialize(40)",shape=box];
		"cntrl41"[label="ForLoopIncrement(41)",shape=box];
		"cntrl42"[label="ForLoopOp: LessThan(DataFlowTag(207), 2j)(42)"];
		"cntrl43"[label="Assign SGPR 0j(43)"];
		"cntrl44"[label="Assign SGPR Add(DataFlowTag(207), 1j)(44)"];
		"cntrl45"[label="Initialize(45)",shape=box];
		"cntrl46"[label="ForLoopIncrement(46)",shape=box];
		"cntrl48"[label="Body(48)",shape=box];
		"cntrl49"[label="Body(49)",shape=box];
		"cntrl50"[label="Scope(50)"];
		"cntrl51"[label="Body(51)",shape=box];
		"cntrl52"[label="Sequence(52)",shape=box];
		"cntrl53"[label="ComputeIndex(53)"];
		"cntrl54"[label="ComputeIndex(54)"];
		"cntrl55"[label="ComputeIndex(55)"];
		"cntrl56"[label="Sequence(56)",shape=box];
		"cntrl57"[label="Sequence(57)",shape=box];
		"cntrl58"[label="Assign VGPR Add(DataFlowTag(218), DataFlowTag(219))(58)"];
		"cntrl59"[label="Body(59)",shape=box];
		"cntrl60"[label="Sequence(60)",shape=box];
		"cntrl61"[label="ForLoopIncrement(61)",shape=box];
		"cntrl62"[label="ComputeIndex(62)"];
		"cntrl63"[label="ComputeIndex(63)"];
		"cntrl64"[label="ComputeIndex(64)"];
		"cntrl65"[label="Sequence(65)",shape=box];
		"cntrl66"[label="Sequence(66)",shape=box];
		"cntrl67"[label="Assign VGPR Add(DataFlowTag(224), DataFlowTag(225))(67)"];
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
		"cntrl26" -> "cntrl33"
		"cntrl26" -> "cntrl34"
		"cntrl26" -> "cntrl61"
		"cntrl26" -> "cntrl70"
		"cntrl29" -> "cntrl27"
		"cntrl30" -> "cntrl28"
		"cntrl31" -> "cntrl35"
		"cntrl31" -> "cntrl36"
		"cntrl33" -> "cntrl32"
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
		"cntrl42" -> "cntrl51"
		"cntrl42" -> "cntrl72"
		"cntrl45" -> "cntrl43"
		"cntrl46" -> "cntrl44"
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
		"coord43" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord50" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord51" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord52" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord61" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord67" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord68" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord2" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord14" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord16" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord73" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord74" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord77" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord78" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord85" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord86" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord87" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord96" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord102" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord103" -> "cntrl6" [style=dotted,weight=0,arrowsize=0]
		"coord18" -> "cntrl8" [style=dotted,weight=0,arrowsize=0]
		"coord20" -> "cntrl11" [style=dotted,weight=0,arrowsize=0]
		"coord5" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
		"coord22" -> "cntrl14" [style=dotted,weight=0,arrowsize=0]
		"coord3" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord24" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord25" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord27" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord108" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord109" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord112" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord113" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord120" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord121" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord122" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord131" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord137" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord138" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord139" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord140" -> "cntrl16" [style=dotted,weight=0,arrowsize=0]
		"coord29" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
		"coord31" -> "cntrl21" [style=dotted,weight=0,arrowsize=0]
		"coord31" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord35" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord167" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord173" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord174" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord187" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord192" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord193" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord199" -> "cntrl26" [style=dotted,weight=0,arrowsize=0]
		"coord199" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
		"coord199" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord2" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord16" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord18" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord50" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord85" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord18" -> "cntrl32" [style=dotted,weight=0,arrowsize=0]
		"coord204" -> "cntrl37" [style=dotted,weight=0,arrowsize=0]
		"coord204" -> "cntrl38" [style=dotted,weight=0,arrowsize=0]
		"coord204" -> "cntrl39" [style=dotted,weight=0,arrowsize=0]
		"coord207" -> "cntrl42" [style=dotted,weight=0,arrowsize=0]
		"coord207" -> "cntrl43" [style=dotted,weight=0,arrowsize=0]
		"coord207" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
		"coord218" -> "cntrl58" [style=dotted,weight=0,arrowsize=0]
		"coord224" -> "cntrl67" [style=dotted,weight=0,arrowsize=0]
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
		"11"[label="ElementOp(6, 18)(11)"];
		"12"[label="Sequence(12)",shape=box];
		"14"[label="LoadVGPR(14)"];
		"15"[label="Body(15)",shape=box];
		"16"[label="LoadTiled(16)"];
		"18"[label="ElementOp(22, 27)(18)"];
		"19"[label="Sequence(19)",shape=box];
		"21"[label="ElementOp(20, 29)(21)"];
		"24"[label="StoreTiled(24)"];
		"26"[label="ForLoopOp: LessThan(DataFlowTag(199), Divide(CommandArgument(Load_Tiled_0_size_1), 16j))(26)"];
		"27"[label="Assign SGPR 0l(27)"];
		"28"[label="Assign SGPR Add(DataFlowTag(199), 1j)(28)"];
		"29"[label="Initialize(29)",shape=box];
		"30"[label="ForLoopIncrement(30)",shape=box];
		"31"[label="Multiply(31)"];
		"32"[label="Assign ACCVGPR 0.00000f(32)"];
		"33"[label="Initialize(33)",shape=box];
		"34"[label="Body(34)",shape=box];
		"35"[label="Body(35)",shape=box];
		"36"[label="Body(36)",shape=box];
		"37"[label="ForLoopOp: LessThan(DataFlowTag(204), 2j)(37)"];
		"38"[label="Assign SGPR 0j(38)"];
		"39"[label="Assign SGPR Add(DataFlowTag(204), 1j)(39)"];
		"40"[label="Initialize(40)",shape=box];
		"41"[label="ForLoopIncrement(41)",shape=box];
		"42"[label="ForLoopOp: LessThan(DataFlowTag(207), 2j)(42)"];
		"43"[label="Assign SGPR 0j(43)"];
		"44"[label="Assign SGPR Add(DataFlowTag(207), 1j)(44)"];
		"45"[label="Initialize(45)",shape=box];
		"46"[label="ForLoopIncrement(46)",shape=box];
		"48"[label="Body(48)",shape=box];
		"49"[label="Body(49)",shape=box];
		"50"[label="Scope(50)"];
		"51"[label="Body(51)",shape=box];
		"52"[label="Sequence(52)",shape=box];
		"53"[label="ComputeIndex(53)"];
		"54"[label="ComputeIndex(54)"];
		"55"[label="ComputeIndex(55)"];
		"56"[label="Sequence(56)",shape=box];
		"57"[label="Sequence(57)",shape=box];
		"58"[label="Assign VGPR Add(DataFlowTag(218), DataFlowTag(219))(58)"];
		"59"[label="Body(59)",shape=box];
		"60"[label="Sequence(60)",shape=box];
		"61"[label="ForLoopIncrement(61)",shape=box];
		"62"[label="ComputeIndex(62)"];
		"63"[label="ComputeIndex(63)"];
		"64"[label="ComputeIndex(64)"];
		"65"[label="Sequence(65)",shape=box];
		"66"[label="Sequence(66)",shape=box];
		"67"[label="Assign VGPR Add(DataFlowTag(224), DataFlowTag(225))(67)"];
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
		"26" -> "33"
		"26" -> "34"
		"26" -> "61"
		"26" -> "70"
		"29" -> "27"
		"30" -> "28"
		"31" -> "35"
		"31" -> "36"
		"31" -> "91"
		"33" -> "32"
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
		"42" -> "51"
		"42" -> "72"
		"42" -> "100"
		"45" -> "43"
		"46" -> "44"
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

        auto kgraph0 = KernelGraph::translate(command);

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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(4, 11)(6)"];
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

        auto kgraph0 = KernelGraph::translate(command);

        // macro tile sizes
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        int t_m = 4;
        int t_n = 2;

        auto mac_tile_0
            = KernelGraph::CoordGraph::MacroTile({mac_m, mac_k}, MemoryType::VGPR, {t_m, t_n}); // A
        auto mac_tile_1
            = KernelGraph::CoordGraph::MacroTile({mac_k, mac_n}, MemoryType::VGPR, {t_m, t_n}); // B
        auto mac_tile_2 = KernelGraph::CoordGraph::MacroTile(
            {mac_m, mac_n}, MemoryType::VGPR, {t_m, t_n}); // A * B

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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="TensorContraction(4, 11)(6)"];
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

        auto kgraph0 = KernelGraph::translate(command);

        int m = 16;
        int n = 8;

        int t_m = 4;
        int t_n = 2;

        auto params = std::make_shared<CommandParameters>();

        auto mac_tile_0 = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_1 = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_2 = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_3 = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_4 = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

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
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="ElementOp(4, 4)(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="ElementOp(11, 11)(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="Sequence(11)",shape=box];
		"cntrl12"[label="ElementOp(17, 15)(12)"];
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

        auto kgraph1 = KernelGraph::lowerTile(kgraph0, params, m_context);

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
		"coord14"[label="DataFlow(14)",shape=box];
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
		"coord30"[label="Flatten(30)",shape=box];
		"coord31"[label="Tile(31)",shape=box];
		"coord32"[label="Tile(32)",shape=box];
		"coord33"[label="PassThrough(33)",shape=box];
		"coord34"[label="PassThrough(34)",shape=box];
		"coord35"[label="Workitem{0, NA}(35)"];
		"coord36"[label="Workitem{1, NA}(36)"];
		"coord37"[label="ThreadTileNumber{0, 1j}(37)"];
		"coord38"[label="ThreadTileNumber{1, 1j}(38)"];
		"coord39"[label="ThreadTileIndex{0, 4j}(39)"];
		"coord40"[label="ThreadTileIndex{1, 2j}(40)"];
		"coord41"[label="Tile(41)",shape=box];
		"coord42"[label="Tile(42)",shape=box];
		"coord43"[label="PassThrough(43)",shape=box];
		"coord44"[label="PassThrough(44)",shape=box];
		"coord45"[label="MacroTileNumber{0, 1j}(45)"];
		"coord46"[label="MacroTileNumber{1, 1j}(46)"];
		"coord47"[label="MacroTileIndex{0, 16j}(47)"];
		"coord48"[label="MacroTileIndex{1, 8j}(48)"];
		"coord49"[label="Workgroup{0, NA}(49)"];
		"coord50"[label="Workgroup{1, NA}(50)"];
		"coord51"[label="Flatten(51)",shape=box];
		"coord52"[label="Tile(52)",shape=box];
		"coord53"[label="Tile(53)",shape=box];
		"coord54"[label="PassThrough(54)",shape=box];
		"coord55"[label="PassThrough(55)",shape=box];
		"coord56"[label="Workitem{0, NA}(56)"];
		"coord57"[label="Workitem{1, NA}(57)"];
		"coord58"[label="ThreadTileNumber{0, 1j}(58)"];
		"coord59"[label="ThreadTileNumber{1, 1j}(59)"];
		"coord60"[label="ThreadTileIndex{0, 4j}(60)"];
		"coord61"[label="ThreadTileIndex{1, 2j}(61)"];
		"coord62"[label="Tile(62)",shape=box];
		"coord63"[label="Tile(63)",shape=box];
		"coord64"[label="PassThrough(64)",shape=box];
		"coord65"[label="PassThrough(65)",shape=box];
		"coord66"[label="MacroTileNumber{0, 1j}(66)"];
		"coord67"[label="MacroTileNumber{1, 1j}(67)"];
		"coord68"[label="MacroTileIndex{0, 16j}(68)"];
		"coord69"[label="MacroTileIndex{1, 8j}(69)"];
		"coord70"[label="Workgroup{0, NA}(70)"];
		"coord71"[label="Workgroup{1, NA}(71)"];
		"coord72"[label="Workitem{0, 1j}(72)"];
		"coord73"[label="Flatten(73)",shape=box];
		"coord74"[label="Flatten(74)",shape=box];
		"coord75"[label="Flatten(75)",shape=box];
		"coord76"[label="PassThrough(76)",shape=box];
		"coord77"[label="PassThrough(77)",shape=box];
		"coord78"[label="Workitem{0, 1j}(78)"];
		"coord79"[label="Workitem{1, 1j}(79)"];
		"coord80"[label="ThreadTile{NA}(80)"];
		"coord81"[label="ThreadTileNumber{0, 1j}(81)"];
		"coord82"[label="ThreadTileNumber{1, 1j}(82)"];
		"coord83"[label="ThreadTileIndex{0, 4j}(83)"];
		"coord84"[label="ThreadTileIndex{1, 2j}(84)"];
		"coord85"[label="Split(85)",shape=box];
		"coord86"[label="Flatten(86)",shape=box];
		"coord87"[label="Flatten(87)",shape=box];
		"coord88"[label="PassThrough(88)",shape=box];
		"coord89"[label="PassThrough(89)",shape=box];
		"coord90"[label="PassThrough(90)",shape=box];
		"coord91"[label="Offset(91)",shape=box];
		"coord92"[label="Stride(92)",shape=box];
		"coord93"[label="Offset(93)",shape=box];
		"coord94"[label="Stride(94)",shape=box];
		"coord95"[label="Offset(95)",shape=box];
		"coord96"[label="Stride(96)",shape=box];
		"coord97"[label="Offset(97)",shape=box];
		"coord98"[label="Stride(98)",shape=box];
		"coord99"[label="Offset(99)",shape=box];
		"coord100"[label="Stride(100)",shape=box];
		"coord101"[label="Offset(101)",shape=box];
		"coord102"[label="Stride(102)",shape=box];
		"coord103"[label="LDS{NA}(103)"];
		"coord104"[label="Workitem{0, 1j}(104)"];
		"coord105"[label="Workitem{1, 1j}(105)"];
		"coord106"[label="DataFlow(106)",shape=box];
		"coord107"[label="DataFlow(107)",shape=box];
		"coord108"[label="Flatten(108)",shape=box];
		"coord1" -> "coord12"
		"coord1" -> "coord14"
		"coord1" -> "coord95"
		"coord1" -> "coord96"
		"coord1" -> "coord97"
		"coord1" -> "coord98"
		"coord1" -> "coord106"
		"coord2" -> "coord5"
		"coord2" -> "coord7"
		"coord2" -> "coord91"
		"coord2" -> "coord92"
		"coord2" -> "coord93"
		"coord2" -> "coord94"
		"coord3" -> "coord31"
		"coord4" -> "coord32"
		"coord5" -> "coord3"
		"coord5" -> "coord4"
		"coord6" -> "coord9"
		"coord7" -> "coord6"
		"coord8" -> "coord18"
		"coord9" -> "coord8"
		"coord10" -> "coord52"
		"coord11" -> "coord53"
		"coord12" -> "coord10"
		"coord12" -> "coord11"
		"coord13" -> "coord16"
		"coord13" -> "coord107"
		"coord14" -> "coord13"
		"coord15" -> "coord18"
		"coord16" -> "coord15"
		"coord17" -> "coord23"
		"coord18" -> "coord17"
		"coord19" -> "coord22"
		"coord20" -> "coord22"
		"coord21" -> "coord99"
		"coord21" -> "coord100"
		"coord21" -> "coord101"
		"coord21" -> "coord102"
		"coord22" -> "coord21"
		"coord23" -> "coord21"
		"coord24" -> "coord33"
		"coord25" -> "coord34"
		"coord26" -> "coord30"
		"coord26" -> "coord41"
		"coord27" -> "coord42"
		"coord27" -> "coord30"
		"coord30" -> "coord6"
		"coord31" -> "coord24"
		"coord31" -> "coord26"
		"coord32" -> "coord25"
		"coord32" -> "coord27"
		"coord33" -> "coord28"
		"coord34" -> "coord29"
		"coord37" -> "coord43"
		"coord38" -> "coord44"
		"coord41" -> "coord37"
		"coord41" -> "coord39"
		"coord42" -> "coord38"
		"coord42" -> "coord40"
		"coord43" -> "coord35"
		"coord44" -> "coord36"
		"coord45" -> "coord54"
		"coord46" -> "coord55"
		"coord47" -> "coord51"
		"coord47" -> "coord62"
		"coord48" -> "coord63"
		"coord48" -> "coord51"
		"coord51" -> "coord13"
		"coord52" -> "coord45"
		"coord52" -> "coord47"
		"coord53" -> "coord46"
		"coord53" -> "coord48"
		"coord54" -> "coord49"
		"coord55" -> "coord50"
		"coord58" -> "coord64"
		"coord59" -> "coord65"
		"coord62" -> "coord58"
		"coord62" -> "coord60"
		"coord63" -> "coord59"
		"coord63" -> "coord61"
		"coord64" -> "coord56"
		"coord65" -> "coord57"
		"coord66" -> "coord74"
		"coord67" -> "coord75"
		"coord68" -> "coord73"
		"coord68" -> "coord74"
		"coord69" -> "coord73"
		"coord69" -> "coord75"
		"coord70" -> "coord76"
		"coord71" -> "coord77"
		"coord72" -> "coord88"
		"coord73" -> "coord17"
		"coord74" -> "coord19"
		"coord75" -> "coord20"
		"coord76" -> "coord66"
		"coord77" -> "coord67"
		"coord78" -> "coord89"
		"coord79" -> "coord90"
		"coord80" -> "coord85"
		"coord81" -> "coord86"
		"coord82" -> "coord87"
		"coord83" -> "coord86"
		"coord84" -> "coord87"
		"coord85" -> "coord83"
		"coord85" -> "coord84"
		"coord86" -> "coord68"
		"coord87" -> "coord69"
		"coord88" -> "coord83"
		"coord88" -> "coord84"
		"coord89" -> "coord81"
		"coord90" -> "coord82"
		"coord91" -> "coord39"
		"coord92" -> "coord39"
		"coord93" -> "coord40"
		"coord94" -> "coord40"
		"coord95" -> "coord60"
		"coord96" -> "coord60"
		"coord97" -> "coord61"
		"coord98" -> "coord61"
		"coord99" -> "coord83"
		"coord100" -> "coord83"
		"coord101" -> "coord84"
		"coord102" -> "coord84"
		"coord104" -> "coord108"
		"coord105" -> "coord108"
		"coord106" -> "coord13"
		"coord107" -> "coord103"
		"coord108" -> "coord103"
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
		"coord26"->"coord27"[style=invis]
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
		"coord37"->"coord39"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord38"->"coord40"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord47"->"coord48"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord45"->"coord47"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord46"->"coord48"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord58"->"coord60"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord59"->"coord61"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord68"->"coord69"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord66"->"coord68"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord67"->"coord69"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord83"->"coord84"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord81"->"coord83"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord82"->"coord84"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord83"->"coord84"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord104"->"coord105"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl4"[label="ElementOp(6, 6)(4)"];
		"cntrl7"[label="LoadTiled(7)"];
		"cntrl9"[label="ElementOp(13, 13)(9)"];
		"cntrl12"[label="ElementOp(8, 15)(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="StoreTiled(15)"];
		"cntrl17"[label="Scope(17)"];
		"cntrl18"[label="Body(18)",shape=box];
		"cntrl19"[label="Sequence(19)",shape=box];
		"cntrl20"[label="Sequence(20)",shape=box];
		"cntrl21"[label="ComputeIndex(21)"];
		"cntrl22"[label="ComputeIndex(22)"];
		"cntrl23"[label="Body(23)",shape=box];
		"cntrl24"[label="Sequence(24)",shape=box];
		"cntrl25"[label="Sequence(25)",shape=box];
		"cntrl26"[label="Scope(26)"];
		"cntrl27"[label="Body(27)",shape=box];
		"cntrl28"[label="Sequence(28)",shape=box];
		"cntrl29"[label="Sequence(29)",shape=box];
		"cntrl30"[label="ComputeIndex(30)"];
		"cntrl31"[label="ComputeIndex(31)"];
		"cntrl32"[label="Body(32)",shape=box];
		"cntrl33"[label="Sequence(33)",shape=box];
		"cntrl34"[label="Sequence(34)",shape=box];
		"cntrl35"[label="Scope(35)"];
		"cntrl36"[label="Sequence(36)",shape=box];
		"cntrl37"[label="ComputeIndex(37)"];
		"cntrl38"[label="ComputeIndex(38)"];
		"cntrl39"[label="Body(39)",shape=box];
		"cntrl40"[label="Sequence(40)",shape=box];
		"cntrl41"[label="Sequence(41)",shape=box];
		"cntrl42"[label="StoreLDSTile(42)"];
		"cntrl43"[label="Barrier(43)"];
		"cntrl44"[label="LoadLDSTile(44)"];
		"cntrl45"[label="Sequence(45)",shape=box];
		"cntrl46"[label="Sequence(46)",shape=box];
		"cntrl47"[label="Sequence(47)",shape=box];
		"cntrl1" -> "cntrl18"
		"cntrl1" -> "cntrl27"
		"cntrl4" -> "cntrl13"
		"cntrl7" -> "cntrl45"
		"cntrl9" -> "cntrl14"
		"cntrl12" -> "cntrl36"
		"cntrl13" -> "cntrl12"
		"cntrl14" -> "cntrl12"
		"cntrl17" -> "cntrl19"
		"cntrl17" -> "cntrl20"
		"cntrl17" -> "cntrl23"
		"cntrl18" -> "cntrl17"
		"cntrl19" -> "cntrl4"
		"cntrl20" -> "cntrl4"
		"cntrl21" -> "cntrl24"
		"cntrl22" -> "cntrl25"
		"cntrl23" -> "cntrl21"
		"cntrl24" -> "cntrl22"
		"cntrl25" -> "cntrl2"
		"cntrl26" -> "cntrl28"
		"cntrl26" -> "cntrl29"
		"cntrl26" -> "cntrl32"
		"cntrl27" -> "cntrl26"
		"cntrl28" -> "cntrl9"
		"cntrl29" -> "cntrl9"
		"cntrl30" -> "cntrl33"
		"cntrl31" -> "cntrl34"
		"cntrl32" -> "cntrl30"
		"cntrl33" -> "cntrl31"
		"cntrl34" -> "cntrl7"
		"cntrl35" -> "cntrl39"
		"cntrl36" -> "cntrl35"
		"cntrl37" -> "cntrl40"
		"cntrl38" -> "cntrl41"
		"cntrl39" -> "cntrl37"
		"cntrl40" -> "cntrl38"
		"cntrl41" -> "cntrl15"
		"cntrl42" -> "cntrl46"
		"cntrl43" -> "cntrl47"
		"cntrl45" -> "cntrl42"
		"cntrl46" -> "cntrl43"
		"cntrl47" -> "cntrl44"
		}
		"coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord3" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord6" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord24" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord25" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord28" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord29" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord39" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord40" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord10" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord45" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord46" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord49" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord50" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord60" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord61" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
		"coord17" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord17" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord21" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord83" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord84" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl42" [style=dotted,weight=0,arrowsize=0]
		"coord103" -> "cntrl42" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
		"coord103" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
		}).";

        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));
    }

    TEST_F(KernelGraphTest, Translate02)
    {
        auto command = commonCommand();

        auto one = Expression::literal(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph0 = KernelGraph::translate(command);
        auto kgraph1 = KernelGraph::lowerLinear(kgraph0, m_context);

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
                = kgraph2.control.getOutputs<KernelGraph::ControlGraph::Body>(getTag(kernelNode));
            EXPECT_EQ(2, outputs.size());

            auto outputs2 = kgraph2.control.getOutputs<KernelGraph::ControlGraph::Sequence>(
                getTag(kernelNode));
            EXPECT_EQ(0, outputs2.size());

            auto outputs3
                = kgraph2.control.getOutputs(getTag(kernelNode), KernelGraph::ControlGraph::Body{});

            auto outputTags3 = kgraph2.control.getOutputTags(getTag(kernelNode),
                                                             KernelGraph::ControlGraph::Body{});

            EXPECT_EQ(outputs3.size(), outputTags3.size());
            for(size_t i = 0; i < outputs3.size(); i++)
            {
                EXPECT_EQ(getTag(outputs3[i]), outputTags3[i]);
            }

            EXPECT_EQ(getTag(outputs[0]), getTag(outputs3[0]));

            auto inputs1
                = kgraph2.control.getInputs<KernelGraph::ControlGraph::Body>(getTag(outputs.at(0)));
            ASSERT_EQ(1, inputs1.size());

            auto actual1 = getTag(inputs1.at(0));
            EXPECT_EQ(actual1, expected);

            auto inputs2 = kgraph2.control.getInputs(getTag(outputs.at(0)),
                                                     KernelGraph::ControlGraph::Body{});
            ASSERT_EQ(1, inputs2.size());

            auto inputTags2 = kgraph2.control.getInputTags(getTag(outputs.at(0)),
                                                           KernelGraph::ControlGraph::Body{});

            EXPECT_EQ(inputs2.size(), inputTags2.size());
            for(size_t i = 0; i < inputs2.size(); i++)
            {
                EXPECT_EQ(getTag(inputs2[i]), inputTags2[i]);
            }

            auto actual2 = getTag(inputs2.at(0));
            EXPECT_EQ(actual1, actual2);

            auto inputs3 = kgraph2.control.getInputs<KernelGraph::ControlGraph::Sequence>(
                getTag(outputs.at(0)));
            EXPECT_EQ(0, inputs3.size());

            auto inputs4 = kgraph2.control.getInputs<KernelGraph::ControlGraph::Initialize>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs4.size());

            auto inputs5 = kgraph2.control.getInputs<KernelGraph::ControlGraph::ForLoopIncrement>(
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
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 0)));

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

        auto mac_tile = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
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
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 0)));

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

        auto mac_tile = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});
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
            rocRoller::Operations::T_Store_Tiled(DataType::Float, 2, 4))); // c

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
        auto mac_tile_lds = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_vgpr
            = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

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

        auto mac_tile = KernelGraph::CoordGraph::MacroTile(
            {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(4, mac_tile);
        params->setDimensionInfo(11, mac_tile);

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);

        auto WF  = KernelGraph::CoordGraph::Wavefront(-1, four, one);
        auto WFX = KernelGraph::CoordGraph::Wavefront(0, two, one);
        auto WFY = KernelGraph::CoordGraph::Wavefront(1, two, one);

        auto postParams = std::make_shared<CommandParameters>();
        postParams->setDimensionInfo(38, WF);
        postParams->setDimensionInfo(36, WFX);
        postParams->setDimensionInfo(37, WFY);
        postParams->setDimensionInfo(93, WF);
        postParams->setDimensionInfo(91, WFX);
        postParams->setDimensionInfo(92, WFY);

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

        auto clean_expr = rocRoller::KernelGraph::cleanArguments(expr2, m_context->kernel());

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

        auto kgraph = KernelGraph::translate(command);
        kgraph      = KernelGraph::cleanArguments(kgraph, m_context->kernel());

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

}
