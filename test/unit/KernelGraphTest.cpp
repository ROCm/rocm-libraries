
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
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

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

        auto kgraph = KernelGraph::translate(command);

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

        auto kgraph = KernelGraph::translate(command);

        kgraph = KernelGraph::lowerLinear(kgraph, m_context);

        kgraph = KernelGraph::lowerLinearUnroll(kgraph, unrollSize, m_context);

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

    TEST_F(KernelGraphTest, Translate01)
    {
        auto command = commonCommand();

        auto kgraph0 = KernelGraph::translate(command);

        auto bottom = kgraph0.coordinates.bottom();
        EXPECT_EQ(bottom.size(), 2);
        EXPECT_EQ(getTag(bottom[0]), getTag(KernelGraph::CoordinateTransform::User(0)));
        EXPECT_EQ(getTag(bottom[1]), getTag(KernelGraph::CoordinateTransform::User(2)));

        auto top = kgraph0.coordinates.top();
        EXPECT_EQ(top.size(), 1);
        EXPECT_EQ(getTag(top[0]), getTag(KernelGraph::CoordinateTransform::User(5, true)));

        std::string expected0 = R".(
          digraph {
           { "User{0, NA, i}" } -> { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } -> { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Flatten"]
           { "User{0, NA, i}" } -> { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } [color=red label="DataFlow"]
           { "User{2, NA, i}" } -> { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Flatten"]
           { "User{2, NA, i}" } -> { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } [color=red label="DataFlow"]
           { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}", "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Linear{3, NA, i}" } [color=red label="DataFlow"]
           { "Linear{3, NA, i}" } -> { "Linear{4, NA, i}" } [color=red label="DataFlow"]
           { "Linear{3, NA, i}", "Linear{4, NA, i}" } -> { "Linear{5, NA, i}" } [color=red label="DataFlow"]
           { "Linear{5, NA, i}" } -> { "Linear{5, NA, o}" } [color=blue label="MakeOutput"]
           { "Linear{5, NA, o}" } -> { "SubDimension{5, 0, NA, o}" } [color=blue label="Split"]
           { "SubDimension{5, 0, NA, o}" } -> { "User{5, NA, o}" } [color=blue label="Join"]
           { "Linear{5, NA, i}" } -> { "User{5, NA, o}" } [color=red label="DataFlow"]

          subgraph clusterCF {"krnKernel"[label="Kernel"];
          "krnLoadLinear(0)"[label="LoadLinear(0)"];
          "krnLoadLinear(2)"[label="LoadLinear(2)"];
          "krnElementOp(3)"[label="ElementOp(3)"];
          "krnElementOp(4)"[label="ElementOp(4)"];
          "krnElementOp(5)"[label="ElementOp(5)"];
          "krnStoreLinear(5)"[label="StoreLinear(5)"];
          "krnKernel" -> "krnLoadLinear(0)"[label="Body"];
          "krnKernel" -> "krnLoadLinear(2)"[label="Body"];
          "krnLoadLinear(0)" -> "krnElementOp(3)"[label="Sequence"];
          "krnLoadLinear(2)" -> "krnElementOp(3)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(4)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(4)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(5)" -> "krnStoreLinear(5)"[label="Sequence"];
          } }
        ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));

        std::string expected1 = R".(
          digraph {
           { "User{0, NA, i}" } -> { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{0, 0, CommandArgument(Load_Linear_0_size_0), i}" } -> { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } [color=blue label="Flatten"]
           { "Linear{0, CommandArgument(Load_Linear_0_size_0), i}" } -> { "Workgroup{0, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{0, 0, 32j, i}" } [color=blue label="Tile"]
           { "Workgroup{0, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{0, 0, 32j, i}" } -> { "VGPR{0, NA, i}" } [color=blue label="Forget"]
           { "User{0, NA, i}" } -> { "VGPR{0, NA, i}" } [color=red label="DataFlow"]
           { "User{2, NA, i}" } -> { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Split"]
           { "SubDimension{2, 0, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } [color=blue label="Flatten"]
           { "Linear{2, CommandArgument(Load_Linear_2_size_0), i}" } -> { "Workgroup{2, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{2, 0, 32j, i}" } [color=blue label="Tile"]
           { "Workgroup{2, 0, LAUNCH_WORKGROUPCOUNT_0, i}", "Workitem{2, 0, 32j, i}" } -> { "VGPR{2, NA, i}" } [color=blue label="Forget"]
           { "User{2, NA, i}" } -> { "VGPR{2, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{0, NA, i}", "VGPR{2, NA, i}" } -> { "VGPR{3, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{3, NA, i}" } -> { "VGPR{4, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{3, NA, i}", "VGPR{4, NA, i}" } -> { "VGPR{5, NA, i}" } [color=red label="DataFlow"]
           { "VGPR{5, NA, i}" } -> { "Workgroup{5, 0, LAUNCH_WORKGROUPCOUNT_0, o}", "Workitem{5, 0, 32j, o}" } [color=blue label="Inherit"]
           { "Workgroup{5, 0, LAUNCH_WORKGROUPCOUNT_0, o}", "Workitem{5, 0, 32j, o}" } -> { "Linear{5, NA, o}" } [color=blue label="Flatten"]
           { "Linear{5, NA, o}" } -> { "SubDimension{5, 0, NA, o}" } [color=blue label="Split"]
           { "SubDimension{5, 0, NA, o}" } -> { "User{5, NA, o}" } [color=blue label="Join"]
           { "VGPR{5, NA, i}" } -> { "User{5, NA, o}" } [color=red label="DataFlow"]

          subgraph clusterCF {"krnKernel"[label="Kernel"];
          "krnLoadVGPR(0)"[label="LoadVGPR(0)"];
          "krnLoadVGPR(2)"[label="LoadVGPR(2)"];
          "krnElementOp(3)"[label="ElementOp(3)"];
          "krnElementOp(4)"[label="ElementOp(4)"];
          "krnElementOp(5)"[label="ElementOp(5)"];
          "krnStoreVGPR(5)"[label="StoreVGPR(5)"];
          "krnKernel" -> "krnLoadVGPR(0)"[label="Body"];
          "krnKernel" -> "krnLoadVGPR(2)"[label="Body"];
          "krnLoadVGPR(0)" -> "krnElementOp(3)"[label="Sequence"];
          "krnLoadVGPR(2)" -> "krnElementOp(3)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(4)"[label="Sequence"];
          "krnElementOp(3)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(4)" -> "krnElementOp(5)"[label="Sequence"];
          "krnElementOp(5)" -> "krnStoreVGPR(5)"[label="Sequence"];
          } }
        ).";

        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto kgraph1 = KernelGraph::lowerLinear(kgraph0, m_context);
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT()));
    }

    TEST_F(KernelGraphTest, Translate01B)
    {
        auto command = commonCommand();
        auto kgraph0 = KernelGraph::translate2(command);

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

        auto visitor = KernelGraph::BaseGraphVisitor2(m_context);
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
		"coord17"[label="Workgroup{0, NA}(17)"];
		"coord18"[label="Workitem{0, 32j}(18)"];
		"coord19"[label="Tile(19)",shape=box];
		"coord20"[label="Forget(20)",shape=box];
		"coord21"[label="DataFlow(21)",shape=box];
		"coord22"[label="VGPR{NA}(22)"];
		"coord23"[label="Workgroup{0, NA}(23)"];
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
		"coord34"[label="Workgroup{0, NA}(34)"];
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
    }

    TEST_F(KernelGraphTest, Translate01BTiled)
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

        auto kgraph0 = KernelGraph::translate2(command);

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

    TEST_F(KernelGraphTest, Translate01BScalar)
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

        auto kgraph0 = KernelGraph::translate2(command);

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

        auto kgraph0 = KernelGraph::translate2(command);

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

        auto mac_tile_0 = KernelGraph::CoordGraph::MacroTile( // A
            {mac_m, mac_k},
            LayoutType::MATRIX_A,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_1 = KernelGraph::CoordGraph::MacroTile( // B
            {mac_k, mac_n},
            LayoutType::MATRIX_B,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2 = KernelGraph::CoordGraph::MacroTile( // C
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_3 = KernelGraph::CoordGraph::MacroTile( // A * B
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_4 = KernelGraph::CoordGraph::MacroTile( // alpha * (A * B)
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_5 = KernelGraph::CoordGraph::MacroTile( // beta * C
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_6 = KernelGraph::CoordGraph::MacroTile( // D
            {mac_m, mac_n},
            LayoutType::MATRIX_ACCUMULATOR,
            {wave_m, wave_n, wave_k, wave_b});

        params->setDimensionInfo(4, mac_tile_0);
        params->setDimensionInfo(11, mac_tile_1);
        params->setDimensionInfo(18, mac_tile_0);
        params->setDimensionInfo(28, mac_tile_1);
        params->setDimensionInfo(30, mac_tile_2);
        params->setDimensionInfo(32, mac_tile_3);
        params->setDimensionInfo(34, mac_tile_4);

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
		"coord18"[label="MacroTile{16,256}(18)"];
		"coord19"[label="DataFlow(19)",shape=box];
		"coord20"[label="MacroTile{128,256}(20)"];
		"coord21"[label="DataFlow(21)",shape=box];
		"coord22"[label="VGPR{NA}(22)"];
		"coord23"[label="DataFlow(23)",shape=box];
		"coord24"[label="SubDimension{0, CommandArgument(Load_Tiled_2_size_0)}(24)"];
		"coord25"[label="SubDimension{1, CommandArgument(Load_Tiled_2_size_1)}(25)"];
		"coord26"[label="Split(26)",shape=box];
		"coord27"[label="MacroTile{128,16}(27)"];
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
		"coord44"[label="Workitem{0, NA}(44)"];
		"coord45"[label="Flatten(45)",shape=box];
		"coord46"[label="Tile(46)",shape=box];
		"coord47"[label="Tile(47)",shape=box];
		"coord48"[label="PassThrough(48)",shape=box];
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
		"coord79"[label="Workitem{0, NA}(79)"];
		"coord80"[label="Flatten(80)",shape=box];
		"coord81"[label="Tile(81)",shape=box];
		"coord82"[label="Tile(82)",shape=box];
		"coord84"[label="PassThrough(84)",shape=box];
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
		"coord111"[label="MacroTileIndex{1, 16j}(111)"];
		"coord112"[label="Workgroup{0, NA}(112)"];
		"coord113"[label="Workgroup{1, NA}(113)"];
		"coord114"[label="Workitem{0, NA}(114)"];
		"coord115"[label="Flatten(115)",shape=box];
		"coord116"[label="Tile(116)",shape=box];
		"coord117"[label="Tile(117)",shape=box];
		"coord118"[label="PassThrough(118)",shape=box];
		"coord119"[label="PassThrough(119)",shape=box];
		"coord120"[label="WaveTile{256i}(120)"];
		"coord121"[label="WaveTileNumber{0, 1j}(121)"];
		"coord122"[label="WaveTileNumber{1, 1j}(122)"];
		"coord123"[label="WaveTileIndex{0, 32j}(123)"];
		"coord124"[label="WaveTileIndex{1, 8j}(124)"];
		"coord125"[label="Tile(125)",shape=box];
		"coord126"[label="Tile(126)",shape=box];
		"coord127"[label="Wavefront{0, NA}(127)"];
		"coord128"[label="Wavefront{1, NA}(128)"];
		"coord129"[label="Wavefront{-1, NA}(129)"];
		"coord130"[label="Lane{32j}(130)"];
		"coord131"[label="VGPR{8j}(131)"];
		"coord132"[label="Flatten(132)",shape=box];
		"coord133"[label="Flatten(133)",shape=box];
		"coord134"[label="BlockNumber{1j}(134)"];
		"coord135"[label="BlockIndex{32j}(135)"];
		"coord136"[label="Flatten(136)",shape=box];
		"coord137"[label="WaveTilePerWorkGroup{0, 2j}(137)"];
		"coord138"[label="WaveTilePerWorkGroup{1, 2j}(138)"];
		"coord139"[label="Flatten(139)",shape=box];
		"coord140"[label="Tile(140)",shape=box];
		"coord141"[label="PassThrough(141)",shape=box];
		"coord142"[label="Tile(142)",shape=box];
		"coord143"[label="MacroTileNumber{0, 1j}(143)"];
		"coord144"[label="MacroTileNumber{1, 1j}(144)"];
		"coord145"[label="MacroTileIndex{0, 128j}(145)"];
		"coord146"[label="MacroTileIndex{1, 256j}(146)"];
		"coord147"[label="Workgroup{0, NA}(147)"];
		"coord148"[label="Workgroup{1, NA}(148)"];
		"coord149"[label="Workitem{0, 1j}(149)"];
		"coord150"[label="Flatten(150)",shape=box];
		"coord151"[label="Flatten(151)",shape=box];
		"coord152"[label="Flatten(152)",shape=box];
		"coord153"[label="PassThrough(153)",shape=box];
		"coord154"[label="PassThrough(154)",shape=box];
		"coord155"[label="WaveTile{8192i}(155)"];
		"coord156"[label="WaveTileNumber{0, 1j}(156)"];
		"coord157"[label="WaveTileNumber{1, 1j}(157)"];
		"coord158"[label="WaveTileIndex{0, 32j}(158)"];
		"coord159"[label="WaveTileIndex{1, 32j}(159)"];
		"coord160"[label="Join(160)",shape=box];
		"coord161"[label="VGPRBlockNumber{8j}(161)"];
		"coord162"[label="VGPRBlockIndex{4j}(162)"];
		"coord163"[label="LANEBlockNumber{8j}(163)"];
		"coord164"[label="LANEBlockIndex{4j}(164)"];
		"coord165"[label="LinearBlock{64j}(165)"];
		"coord166"[label="RowBlock{8j}(166)"];
		"coord167"[label="ColBlock{8j}(167)"];
		"coord168"[label="Flatten(168)",shape=box];
		"coord169"[label="Flatten(169)",shape=box];
		"coord170"[label="Wavefront{0, NA}(170)"];
		"coord171"[label="Wavefront{1, NA}(171)"];
		"coord172"[label="Wavefront{-1, NA}(172)"];
		"coord173"[label="Tile(173)",shape=box];
		"coord174"[label="Lane{32j}(174)"];
		"coord175"[label="VGPR{32j}(175)"];
		"coord176"[label="Tile(176)",shape=box];
		"coord177"[label="Tile(177)",shape=box];
		"coord178"[label="Flatten(178)",shape=box];
		"coord179"[label="Tile(179)",shape=box];
		"coord180"[label="WaveTilePerWorkGroup{0, 2j}(180)"];
		"coord181"[label="WaveTilePerWorkGroup{1, 2j}(181)"];
		"coord182"[label="Flatten(182)",shape=box];
		"coord183"[label="Flatten(183)",shape=box];
		"coord184"[label="Flatten(184)",shape=box];
		"coord185"[label="Flatten(185)",shape=box];
		"coord186"[label="Tile(186)",shape=box];
		"coord187"[label="Linear{Divide(CommandArgument(Load_Tiled_0_size_1), 16j)}(187)"];
		"coord188"[label="ForLoop{NA}(188)"];
		"coord189"[label="DataFlow(189)",shape=box];
		"coord190"[label="PassThrough(190)",shape=box];
		"coord191"[label="PassThrough(191)",shape=box];
		"coord192"[label="Linear{2j}(192)"];
		"coord193"[label="ForLoop{NA}(193)"];
		"coord194"[label="DataFlow(194)",shape=box];
		"coord195"[label="Linear{2j}(195)"];
		"coord196"[label="ForLoop{NA}(196)"];
		"coord197"[label="DataFlow(197)",shape=box];
		"coord198"[label="PassThrough(198)",shape=box];
		"coord199"[label="PassThrough(199)",shape=box];
		"coord200"[label="PassThrough(200)",shape=box];
		"coord201"[label="PassThrough(201)",shape=box];
		"coord202"[label="PassThrough(202)",shape=box];
		"coord203"[label="PassThrough(203)",shape=box];
		"coord204"[label="PassThrough(204)",shape=box];
		"coord205"[label="PassThrough(205)",shape=box];
		"coord1" -> "coord10"
		"coord1" -> "coord12"
		"coord2" -> "coord15"
		"coord2" -> "coord17"
		"coord3" -> "coord26"
		"coord3" -> "coord28"
		"coord4" -> "coord7"
		"coord5" -> "coord23"
		"coord6" -> "coord21"
		"coord7" -> "coord6"
		"coord8" -> "coord46"
		"coord9" -> "coord47"
		"coord10" -> "coord8"
		"coord10" -> "coord9"
		"coord11" -> "coord19"
		"coord12" -> "coord11"
		"coord13" -> "coord81"
		"coord14" -> "coord82"
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
		"coord24" -> "coord116"
		"coord25" -> "coord117"
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
		"coord38" -> "coord48"
		"coord39" -> "coord190"
		"coord40" -> "coord45"
		"coord40" -> "coord55"
		"coord41" -> "coord56"
		"coord41" -> "coord45"
		"coord45" -> "coord11"
		"coord46" -> "coord38"
		"coord46" -> "coord40"
		"coord47" -> "coord39"
		"coord47" -> "coord41"
		"coord48" -> "coord42"
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
		"coord63" -> "coord44"
		"coord64" -> "coord66"
		"coord65" -> "coord66"
		"coord66" -> "coord60"
		"coord67" -> "coord198"
		"coord68" -> "coord199"
		"coord69" -> "coord50"
		"coord70" -> "coord64"
		"coord70" -> "coord61"
		"coord71" -> "coord65"
		"coord72" -> "coord57"
		"coord72" -> "coord67"
		"coord73" -> "coord191"
		"coord74" -> "coord84"
		"coord75" -> "coord80"
		"coord75" -> "coord90"
		"coord76" -> "coord91"
		"coord76" -> "coord80"
		"coord80" -> "coord16"
		"coord81" -> "coord73"
		"coord81" -> "coord75"
		"coord82" -> "coord74"
		"coord82" -> "coord76"
		"coord84" -> "coord78"
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
		"coord98" -> "coord79"
		"coord99" -> "coord101"
		"coord100" -> "coord101"
		"coord101" -> "coord95"
		"coord102" -> "coord200"
		"coord103" -> "coord201"
		"coord104" -> "coord85"
		"coord105" -> "coord99"
		"coord105" -> "coord96"
		"coord106" -> "coord100"
		"coord107" -> "coord93"
		"coord107" -> "coord103"
		"coord108" -> "coord118"
		"coord109" -> "coord119"
		"coord110" -> "coord115"
		"coord110" -> "coord125"
		"coord111" -> "coord126"
		"coord111" -> "coord115"
		"coord115" -> "coord27"
		"coord116" -> "coord108"
		"coord116" -> "coord110"
		"coord117" -> "coord109"
		"coord117" -> "coord111"
		"coord118" -> "coord112"
		"coord119" -> "coord113"
		"coord121" -> "coord142"
		"coord123" -> "coord141"
		"coord123" -> "coord139"
		"coord124" -> "coord139"
		"coord124" -> "coord140"
		"coord125" -> "coord121"
		"coord125" -> "coord123"
		"coord126" -> "coord122"
		"coord126" -> "coord124"
		"coord127" -> "coord132"
		"coord128" -> "coord132"
		"coord129" -> "coord133"
		"coord130" -> "coord133"
		"coord132" -> "coord129"
		"coord133" -> "coord114"
		"coord134" -> "coord136"
		"coord135" -> "coord136"
		"coord136" -> "coord130"
		"coord137" -> "coord202"
		"coord138" -> "coord203"
		"coord139" -> "coord120"
		"coord140" -> "coord134"
		"coord140" -> "coord131"
		"coord141" -> "coord135"
		"coord142" -> "coord127"
		"coord142" -> "coord137"
		"coord143" -> "coord151"
		"coord144" -> "coord152"
		"coord145" -> "coord150"
		"coord145" -> "coord151"
		"coord146" -> "coord150"
		"coord146" -> "coord152"
		"coord147" -> "coord153"
		"coord148" -> "coord154"
		"coord149" -> "coord186"
		"coord150" -> "coord31"
		"coord151" -> "coord33"
		"coord152" -> "coord34"
		"coord153" -> "coord143"
		"coord154" -> "coord144"
		"coord156" -> "coord168"
		"coord157" -> "coord169"
		"coord158" -> "coord160"
		"coord158" -> "coord168"
		"coord159" -> "coord160"
		"coord159" -> "coord169"
		"coord160" -> "coord155"
		"coord161" -> "coord178"
		"coord162" -> "coord184"
		"coord163" -> "coord178"
		"coord164" -> "coord185"
		"coord165" -> "coord179"
		"coord166" -> "coord184"
		"coord167" -> "coord185"
		"coord168" -> "coord145"
		"coord169" -> "coord146"
		"coord170" -> "coord182"
		"coord171" -> "coord183"
		"coord172" -> "coord173"
		"coord173" -> "coord170"
		"coord173" -> "coord171"
		"coord174" -> "coord177"
		"coord175" -> "coord176"
		"coord176" -> "coord161"
		"coord176" -> "coord162"
		"coord177" -> "coord163"
		"coord177" -> "coord164"
		"coord178" -> "coord165"
		"coord179" -> "coord166"
		"coord179" -> "coord167"
		"coord180" -> "coord182"
		"coord181" -> "coord183"
		"coord182" -> "coord156"
		"coord183" -> "coord157"
		"coord184" -> "coord158"
		"coord185" -> "coord159"
		"coord186" -> "coord172"
		"coord186" -> "coord174"
		"coord187" -> "coord189"
		"coord189" -> "coord188"
		"coord190" -> "coord188"
		"coord191" -> "coord188"
		"coord192" -> "coord194"
		"coord193" -> "coord204"
		"coord194" -> "coord193"
		"coord195" -> "coord197"
		"coord196" -> "coord205"
		"coord197" -> "coord196"
		"coord198" -> "coord193"
		"coord199" -> "coord196"
		"coord200" -> "coord193"
		"coord201" -> "coord196"
		"coord202" -> "coord193"
		"coord203" -> "coord196"
		"coord204" -> "coord180"
		"coord205" -> "coord181"
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
		"coord124"->"coord123"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord134"->"coord131"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord127"->"coord137"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord145"->"coord146"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord143"->"coord145"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord144"->"coord146"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord158"->"coord159"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord156"->"coord158"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord157"->"coord159"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord170"->"coord171"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord161"->"coord162"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord163"->"coord164"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord161"->"coord163"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord166"->"coord167"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord170"->"coord180"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord171"->"coord181"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord166"->"coord162"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord167"->"coord164"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord172"->"coord174"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadVGPR(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="LoadTiled(4)"];
		"cntrl6"[label="LoadTiled(6)"];
		"cntrl11"[label="ElementOp(6, 18)(11)"];
		"cntrl12"[label="Sequence(12)",shape=box];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="LoadVGPR(14)"];
		"cntrl15"[label="Body(15)",shape=box];
		"cntrl16"[label="LoadTiled(16)"];
		"cntrl17"[label="Body(17)",shape=box];
		"cntrl18"[label="ElementOp(22, 27)(18)"];
		"cntrl19"[label="Sequence(19)",shape=box];
		"cntrl20"[label="Sequence(20)",shape=box];
		"cntrl21"[label="ElementOp(20, 29)(21)"];
		"cntrl22"[label="Sequence(22)",shape=box];
		"cntrl23"[label="Sequence(23)",shape=box];
		"cntrl24"[label="StoreTiled(24)"];
		"cntrl25"[label="Sequence(25)",shape=box];
		"cntrl26"[label="ForLoopOp: LessThan(DataFlowTag(187), Divide(CommandArgument(Load_Tiled_0_size_1), 16j))(26)"];
		"cntrl27"[label="Assign SGPR 0l(27)"];
		"cntrl28"[label="Assign SGPR Add(DataFlowTag(187), 1j)(28)"];
		"cntrl29"[label="Initialize(29)",shape=box];
		"cntrl30"[label="ForLoopIncrement(30)",shape=box];
		"cntrl31"[label="Multiply(31)"];
		"cntrl32"[label="Assign ACCVGPR 0.00000f(32)"];
		"cntrl33"[label="Initialize(33)",shape=box];
		"cntrl34"[label="Body(34)",shape=box];
		"cntrl35"[label="Body(35)",shape=box];
		"cntrl36"[label="Body(36)",shape=box];
		"cntrl37"[label="ForLoopOp: LessThan(DataFlowTag(192), 2j)(37)"];
		"cntrl38"[label="Assign SGPR 0j(38)"];
		"cntrl39"[label="Assign SGPR Add(DataFlowTag(192), 1j)(39)"];
		"cntrl40"[label="Initialize(40)",shape=box];
		"cntrl41"[label="ForLoopIncrement(41)",shape=box];
		"cntrl42"[label="ForLoopOp: LessThan(DataFlowTag(195), 2j)(42)"];
		"cntrl43"[label="Assign SGPR 0j(43)"];
		"cntrl44"[label="Assign SGPR Add(DataFlowTag(195), 1j)(44)"];
		"cntrl45"[label="Initialize(45)",shape=box];
		"cntrl46"[label="ForLoopIncrement(46)",shape=box];
		"cntrl47"[label="Body(47)",shape=box];
		"cntrl48"[label="Body(48)",shape=box];
		"cntrl49"[label="Body(49)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl15"
		"cntrl1" -> "cntrl49"
		"cntrl2" -> "cntrl12"
		"cntrl3" -> "cntrl2"
		"cntrl11" -> "cntrl22"
		"cntrl12" -> "cntrl11"
		"cntrl13" -> "cntrl11"
		"cntrl14" -> "cntrl19"
		"cntrl15" -> "cntrl14"
		"cntrl16" -> "cntrl20"
		"cntrl17" -> "cntrl16"
		"cntrl18" -> "cntrl23"
		"cntrl19" -> "cntrl18"
		"cntrl20" -> "cntrl18"
		"cntrl21" -> "cntrl25"
		"cntrl22" -> "cntrl21"
		"cntrl23" -> "cntrl21"
		"cntrl25" -> "cntrl24"
		"cntrl26" -> "cntrl29"
		"cntrl26" -> "cntrl30"
		"cntrl26" -> "cntrl33"
		"cntrl26" -> "cntrl34"
		"cntrl26" -> "cntrl13"
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
		"cntrl42" -> "cntrl47"
		"cntrl42" -> "cntrl17"
		"cntrl45" -> "cntrl43"
		"cntrl46" -> "cntrl44"
		"cntrl47" -> "cntrl26"
		"cntrl48" -> "cntrl42"
		"cntrl49" -> "cntrl37"
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
		"coord29" -> "cntrl18" [style=dotted,weight=0,arrowsize=0]
		"coord31" -> "cntrl21" [style=dotted,weight=0,arrowsize=0]
		"coord31" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord35" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord155" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord175" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord180" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord181" -> "cntrl24" [style=dotted,weight=0,arrowsize=0]
		"coord187" -> "cntrl26" [style=dotted,weight=0,arrowsize=0]
		"coord187" -> "cntrl27" [style=dotted,weight=0,arrowsize=0]
		"coord187" -> "cntrl28" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord2" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord16" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord18" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord50" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord85" -> "cntrl31" [style=dotted,weight=0,arrowsize=0]
		"coord18" -> "cntrl32" [style=dotted,weight=0,arrowsize=0]
		"coord192" -> "cntrl37" [style=dotted,weight=0,arrowsize=0]
		"coord192" -> "cntrl38" [style=dotted,weight=0,arrowsize=0]
		"coord192" -> "cntrl39" [style=dotted,weight=0,arrowsize=0]
		"coord195" -> "cntrl42" [style=dotted,weight=0,arrowsize=0]
		"coord195" -> "cntrl43" [style=dotted,weight=0,arrowsize=0]
		"coord195" -> "cntrl44" [style=dotted,weight=0,arrowsize=0]
	     }).";
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));
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
           { "User{0, NA, i}" } -> { "SubDimension{0, 0, CommandArgument(Load_Tiled_0_size_0), i}", "SubDimension{0, 1, CommandArgument(Load_Tiled_0_size_1), i}" } [color=blue label="Split"]
           { "SubDimension{0, 0, CommandArgument(Load_Tiled_0_size_0), i}", "SubDimension{0, 1, CommandArgument(Load_Tiled_0_size_1), i}" } -> { "MacroTile{0, NA, i}" } [color=blue label="ConstructTensorTile"]
           { "User{0, NA, i}" } -> { "MacroTile{0, NA, i}" } [color=red label="DataFlow"]
           { "User{1, NA, i}" } -> { "SubDimension{1, 0, CommandArgument(Load_Tiled_1_size_0), i}", "SubDimension{1, 1, CommandArgument(Load_Tiled_1_size_1), i}" } [color=blue label="Split"]
           { "SubDimension{1, 0, CommandArgument(Load_Tiled_1_size_0), i}", "SubDimension{1, 1, CommandArgument(Load_Tiled_1_size_1), i}" } -> { "MacroTile{1, NA, i}" } [color=blue label="ConstructTensorTile"]
           { "User{1, NA, i}" } -> { "MacroTile{1, NA, i}" } [color=red label="DataFlow"]
           { "MacroTile{0, NA, i}", "MacroTile{1, NA, i}" } -> { "MacroTile{2, NA, i}" } [color=red label="DataFlow"]

          subgraph clusterCF {"krnKernel"[label="Kernel"];
          "krnLoadTiled(0)"[label="LoadTiled(0)"];
          "krnLoadTiled(1)"[label="LoadTiled(1)"];
          "krnTensorContraction(2, 0, 1)"[label="TensorContraction(2, 0, 1)"];
          "krnKernel" -> "krnLoadTiled(0)"[label="Body"];
          "krnKernel" -> "krnLoadTiled(1)"[label="Body"];
          "krnLoadTiled(0)" -> "krnTensorContraction(2, 0, 1)"[label="Sequence"];
          "krnLoadTiled(1)" -> "krnTensorContraction(2, 0, 1)"[label="Sequence"];
          } }
        ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));
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

        auto kgraph0 = KernelGraph::translate2(command);

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

        auto kgraph0 = KernelGraph::translate2(command);

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
		"coord30"[label="Workitem{0, NA}(30)"];
		"coord31"[label="Flatten(31)",shape=box];
		"coord32"[label="Tile(32)",shape=box];
		"coord33"[label="Tile(33)",shape=box];
		"coord34"[label="PassThrough(34)",shape=box];
		"coord35"[label="PassThrough(35)",shape=box];
		"coord36"[label="Workitem{0, NA}(36)"];
		"coord37"[label="Workitem{1, NA}(37)"];
		"coord38"[label="ThreadTile{NA}(38)"];
		"coord39"[label="ThreadTileNumber{0, 1j}(39)"];
		"coord40"[label="ThreadTileNumber{1, 1j}(40)"];
		"coord41"[label="ThreadTileIndex{0, 4j}(41)"];
		"coord42"[label="ThreadTileIndex{1, 2j}(42)"];
		"coord43"[label="Join(43)",shape=box];
		"coord44"[label="Tile(44)",shape=box];
		"coord45"[label="Tile(45)",shape=box];
		"coord46"[label="PassThrough(46)",shape=box];
		"coord47"[label="PassThrough(47)",shape=box];
		"coord48"[label="PassThrough(48)",shape=box];
		"coord49"[label="MacroTileNumber{0, 1j}(49)"];
		"coord50"[label="MacroTileNumber{1, 1j}(50)"];
		"coord51"[label="MacroTileIndex{0, 16j}(51)"];
		"coord52"[label="MacroTileIndex{1, 8j}(52)"];
		"coord53"[label="Workgroup{0, NA}(53)"];
		"coord54"[label="Workgroup{1, NA}(54)"];
		"coord55"[label="Workitem{0, NA}(55)"];
		"coord56"[label="Flatten(56)",shape=box];
		"coord57"[label="Tile(57)",shape=box];
		"coord58"[label="Tile(58)",shape=box];
		"coord59"[label="PassThrough(59)",shape=box];
		"coord60"[label="PassThrough(60)",shape=box];
		"coord61"[label="Workitem{0, NA}(61)"];
		"coord62"[label="Workitem{1, NA}(62)"];
		"coord63"[label="ThreadTile{NA}(63)"];
		"coord64"[label="ThreadTileNumber{0, 1j}(64)"];
		"coord65"[label="ThreadTileNumber{1, 1j}(65)"];
		"coord66"[label="ThreadTileIndex{0, 4j}(66)"];
		"coord67"[label="ThreadTileIndex{1, 2j}(67)"];
		"coord68"[label="Join(68)",shape=box];
		"coord69"[label="Tile(69)",shape=box];
		"coord70"[label="Tile(70)",shape=box];
		"coord71"[label="PassThrough(71)",shape=box];
		"coord72"[label="PassThrough(72)",shape=box];
		"coord73"[label="PassThrough(73)",shape=box];
		"coord74"[label="MacroTileNumber{0, 1j}(74)"];
		"coord75"[label="MacroTileNumber{1, 1j}(75)"];
		"coord76"[label="MacroTileIndex{0, 16j}(76)"];
		"coord77"[label="MacroTileIndex{1, 8j}(77)"];
		"coord78"[label="Workgroup{0, NA}(78)"];
		"coord79"[label="Workgroup{1, NA}(79)"];
		"coord80"[label="Workitem{0, 1j}(80)"];
		"coord81"[label="Flatten(81)",shape=box];
		"coord82"[label="Flatten(82)",shape=box];
		"coord83"[label="Flatten(83)",shape=box];
		"coord84"[label="PassThrough(84)",shape=box];
		"coord85"[label="PassThrough(85)",shape=box];
		"coord86"[label="Workitem{0, 1j}(86)"];
		"coord87"[label="Workitem{1, 1j}(87)"];
		"coord88"[label="ThreadTile{NA}(88)"];
		"coord89"[label="ThreadTileNumber{0, 1j}(89)"];
		"coord90"[label="ThreadTileNumber{1, 1j}(90)"];
		"coord91"[label="ThreadTileIndex{0, 4j}(91)"];
		"coord92"[label="ThreadTileIndex{1, 2j}(92)"];
		"coord93"[label="Split(93)",shape=box];
		"coord94"[label="Flatten(94)",shape=box];
		"coord95"[label="Flatten(95)",shape=box];
		"coord96"[label="PassThrough(96)",shape=box];
		"coord97"[label="PassThrough(97)",shape=box];
		"coord98"[label="PassThrough(98)",shape=box];
		"coord99"[label="LDS{NA}(99)"];
		"coord100"[label="Workitem{0, 1j}(100)"];
		"coord101"[label="Workitem{1, 1j}(101)"];
		"coord102"[label="DataFlow(102)",shape=box];
		"coord103"[label="DataFlow(103)",shape=box];
		"coord104"[label="Flatten(104)",shape=box];
		"coord1" -> "coord12"
		"coord1" -> "coord14"
		"coord1" -> "coord102"
		"coord2" -> "coord5"
		"coord2" -> "coord7"
		"coord3" -> "coord32"
		"coord4" -> "coord33"
		"coord5" -> "coord3"
		"coord5" -> "coord4"
		"coord6" -> "coord9"
		"coord7" -> "coord6"
		"coord8" -> "coord18"
		"coord9" -> "coord8"
		"coord10" -> "coord57"
		"coord11" -> "coord58"
		"coord12" -> "coord10"
		"coord12" -> "coord11"
		"coord13" -> "coord16"
		"coord13" -> "coord103"
		"coord14" -> "coord13"
		"coord15" -> "coord18"
		"coord16" -> "coord15"
		"coord17" -> "coord23"
		"coord18" -> "coord17"
		"coord19" -> "coord22"
		"coord20" -> "coord22"
		"coord22" -> "coord21"
		"coord23" -> "coord21"
		"coord24" -> "coord34"
		"coord25" -> "coord35"
		"coord26" -> "coord31"
		"coord26" -> "coord44"
		"coord27" -> "coord45"
		"coord27" -> "coord31"
		"coord30" -> "coord46"
		"coord31" -> "coord6"
		"coord32" -> "coord24"
		"coord32" -> "coord26"
		"coord33" -> "coord25"
		"coord33" -> "coord27"
		"coord34" -> "coord28"
		"coord35" -> "coord29"
		"coord39" -> "coord47"
		"coord40" -> "coord48"
		"coord41" -> "coord43"
		"coord42" -> "coord43"
		"coord43" -> "coord38"
		"coord44" -> "coord39"
		"coord44" -> "coord41"
		"coord45" -> "coord40"
		"coord45" -> "coord42"
		"coord46" -> "coord41"
		"coord46" -> "coord42"
		"coord47" -> "coord36"
		"coord48" -> "coord37"
		"coord49" -> "coord59"
		"coord50" -> "coord60"
		"coord51" -> "coord56"
		"coord51" -> "coord69"
		"coord52" -> "coord70"
		"coord52" -> "coord56"
		"coord55" -> "coord71"
		"coord56" -> "coord13"
		"coord57" -> "coord49"
		"coord57" -> "coord51"
		"coord58" -> "coord50"
		"coord58" -> "coord52"
		"coord59" -> "coord53"
		"coord60" -> "coord54"
		"coord64" -> "coord72"
		"coord65" -> "coord73"
		"coord66" -> "coord68"
		"coord67" -> "coord68"
		"coord68" -> "coord63"
		"coord69" -> "coord64"
		"coord69" -> "coord66"
		"coord70" -> "coord65"
		"coord70" -> "coord67"
		"coord71" -> "coord66"
		"coord71" -> "coord67"
		"coord72" -> "coord61"
		"coord73" -> "coord62"
		"coord74" -> "coord82"
		"coord75" -> "coord83"
		"coord76" -> "coord81"
		"coord76" -> "coord82"
		"coord77" -> "coord81"
		"coord77" -> "coord83"
		"coord78" -> "coord84"
		"coord79" -> "coord85"
		"coord80" -> "coord96"
		"coord81" -> "coord17"
		"coord82" -> "coord19"
		"coord83" -> "coord20"
		"coord84" -> "coord74"
		"coord85" -> "coord75"
		"coord86" -> "coord97"
		"coord87" -> "coord98"
		"coord88" -> "coord93"
		"coord89" -> "coord94"
		"coord90" -> "coord95"
		"coord91" -> "coord94"
		"coord92" -> "coord95"
		"coord93" -> "coord91"
		"coord93" -> "coord92"
		"coord94" -> "coord76"
		"coord95" -> "coord77"
		"coord96" -> "coord91"
		"coord96" -> "coord92"
		"coord97" -> "coord89"
		"coord98" -> "coord90"
		"coord100" -> "coord104"
		"coord101" -> "coord104"
		"coord102" -> "coord13"
		"coord103" -> "coord99"
		"coord104" -> "coord99"
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
		"coord41"->"coord42"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord39"->"coord41"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord40"->"coord42"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord41"->"coord42"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord51"->"coord52"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord49"->"coord51"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord50"->"coord52"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord66"->"coord67"[style=invis]
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
		"coord66"->"coord67"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord76"->"coord77"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord74"->"coord76"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord75"->"coord77"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord91"->"coord92"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord89"->"coord91"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord90"->"coord92"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord91"->"coord92"[style=invis]
		rankdir=LR
		}
		{
		rank=same
		"coord100"->"coord101"[style=invis]
		rankdir=LR
		}
		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadTiled(2)"];
		"cntrl3"[label="Body(3)",shape=box];
		"cntrl4"[label="ElementOp(6, 6)(4)"];
		"cntrl5"[label="Sequence(5)",shape=box];
		"cntrl6"[label="Sequence(6)",shape=box];
		"cntrl7"[label="LoadTiled(7)"];
		"cntrl8"[label="Body(8)",shape=box];
		"cntrl9"[label="ElementOp(13, 13)(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="Sequence(11)",shape=box];
		"cntrl12"[label="ElementOp(8, 15)(12)"];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="Sequence(14)",shape=box];
		"cntrl15"[label="StoreTiled(15)"];
		"cntrl16"[label="Sequence(16)",shape=box];
		"cntrl17"[label="StoreLDSTile(17)"];
		"cntrl18"[label="Barrier(18)"];
		"cntrl19"[label="LoadLDSTile(19)"];
		"cntrl20"[label="Sequence(20)",shape=box];
		"cntrl21"[label="Sequence(21)",shape=box];
		"cntrl22"[label="Sequence(22)",shape=box];
		"cntrl1" -> "cntrl3"
		"cntrl1" -> "cntrl8"
		"cntrl2" -> "cntrl5"
		"cntrl2" -> "cntrl6"
		"cntrl3" -> "cntrl2"
		"cntrl4" -> "cntrl13"
		"cntrl5" -> "cntrl4"
		"cntrl6" -> "cntrl4"
		"cntrl7" -> "cntrl20"
		"cntrl8" -> "cntrl7"
		"cntrl9" -> "cntrl14"
		"cntrl10" -> "cntrl9"
		"cntrl11" -> "cntrl9"
		"cntrl12" -> "cntrl16"
		"cntrl13" -> "cntrl12"
		"cntrl14" -> "cntrl12"
		"cntrl16" -> "cntrl15"
		"cntrl17" -> "cntrl21"
		"cntrl18" -> "cntrl22"
		"cntrl19" -> "cntrl10"
		"cntrl19" -> "cntrl11"
		"cntrl20" -> "cntrl17"
		"cntrl21" -> "cntrl18"
		"cntrl22" -> "cntrl19"
		}
		"coord2" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord3" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord4" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord6" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord24" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord25" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord28" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord29" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord41" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord42" -> "cntrl2" [style=dotted,weight=0,arrowsize=0]
		"coord8" -> "cntrl4" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord10" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord11" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord49" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord50" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord53" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord54" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord66" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord67" -> "cntrl7" [style=dotted,weight=0,arrowsize=0]
		"coord15" -> "cntrl9" [style=dotted,weight=0,arrowsize=0]
		"coord17" -> "cntrl12" [style=dotted,weight=0,arrowsize=0]
		"coord17" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord21" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord91" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord92" -> "cntrl15" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
		"coord99" -> "cntrl17" [style=dotted,weight=0,arrowsize=0]
		"coord1" -> "cntrl19" [style=dotted,weight=0,arrowsize=0]
		"coord13" -> "cntrl19" [style=dotted,weight=0,arrowsize=0]
		"coord99" -> "cntrl19" [style=dotted,weight=0,arrowsize=0]
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

        auto user0   = KernelGraph::CoordinateTransform::User(0);
        auto block0  = KernelGraph::CoordinateTransform::Workgroup(0);
        auto thread0 = KernelGraph::CoordinateTransform::Workitem(0);

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

        auto mac_tile_in
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_out = KernelGraph::CoordinateTransform::MacroTile(
            0, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_in);
        params->setDimensionInfo(mac_tile_out);

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

        auto mac_tile_in
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_out = KernelGraph::CoordinateTransform::MacroTile(
            0, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_in);
        params->setDimensionInfo(mac_tile_out);

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
        auto mac_tile_0
            = KernelGraph::CoordinateTransform::MacroTile(0, {m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_1
            = KernelGraph::CoordinateTransform::MacroTile(1, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_3
            = KernelGraph::CoordinateTransform::MacroTile(3, {m, n}, MemoryType::VGPR, {t_m, t_n});
        auto mac_tile_4 = KernelGraph::CoordinateTransform::MacroTile(
            4, {m, n}, MemoryType::VGPR, {t_m, t_n}, true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_1);
        params->setDimensionInfo(mac_tile_2);
        params->setDimensionInfo(mac_tile_3);
        params->setDimensionInfo(mac_tile_4);

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

    // Delete this when graph rearch complete
    TEST_F(KernelGraphTestGPU, GPU_TensorTileAdd2)
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

        // TODO: Add a "fill" operation on the kernel graph to propagate tile sizes where appropriate
        auto mac_tile_lds = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::LDS, {t_m, t_n});
        auto mac_tile_vgpr
            = KernelGraph::CoordGraph::MacroTile({m, n}, MemoryType::VGPR, {t_m, t_n});

        params->setDimensionInfo(4, mac_tile_lds);
        params->setDimensionInfo(11, mac_tile_vgpr);
        params->setDimensionInfo(15, mac_tile_vgpr);
        params->setDimensionInfo(17, mac_tile_vgpr);
        params->setDimensionInfo(19, mac_tile_vgpr);

        auto context = m_context;
        context->kernel()->setKernelDimensions(2);
        context->kernel()->setWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        context->kernel()->setWorkitemCount({NX, NY, NZ});

        context->kernel()->addCommandArguments(command->getArguments());
        auto kgraph = KernelGraph::translate2(command);
        kgraph      = KernelGraph::updateParameters(kgraph, params);
        kgraph      = KernelGraph::lowerTile(kgraph, params, context);
        kgraph      = KernelGraph::cleanArguments(kgraph, context->kernel());

        context->schedule(context->kernel()->preamble());
        context->schedule(context->kernel()->prolog());
        context->schedule(KernelGraph::generate(kgraph, context->kernel()));
        context->schedule(context->kernel()->postamble());
        context->schedule(context->kernel()->amdgpu_metadata());

        auto executableKernel = context->instructions()->getExecutableKernel();

        KernelArguments kargs;
        for(auto& arg : context->kernel()->arguments())
        {
            auto value = evaluate(arg.expression, runtimeArgs.runtimeArguments());
            kargs.append(arg.name, value);
        }

        KernelInvocation kinv;
        kinv.workgroupSize    = context->kernel()->workgroupSize();
        kinv.workitemCount[0] = nx / t_m;
        kinv.workitemCount[1] = ny / t_n;

        executableKernel->executeKernel(kargs, kinv);

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
        REQUIRE_ARCH_CAP(GPUCapability::HasMFMA);

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

        auto mac_tile_0 = KernelGraph::CoordinateTransform::MacroTile(
            0, {mac_m, mac_n}, LayoutType::MATRIX_ACCUMULATOR, {wave_m, wave_n, wave_k, wave_b});
        auto mac_tile_2
            = KernelGraph::CoordinateTransform::MacroTile(2,
                                                          {mac_m, mac_n},
                                                          LayoutType::MATRIX_ACCUMULATOR,
                                                          {wave_m, wave_n, wave_k, wave_b},
                                                          true);

        params->setDimensionInfo(mac_tile_0);
        params->setDimensionInfo(mac_tile_2);

        auto four = Expression::literal(4u);
        auto two  = Expression::literal(2u);
        auto one  = Expression::literal(1u);
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, -1, four, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 0, two, nullptr));
        params->setDimensionInfo(KernelGraph::CoordinateTransform::Wavefront(0, 1, two, nullptr));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, -1, four, one, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 0, two, nullptr, true));
        params->setDimensionInfo(
            KernelGraph::CoordinateTransform::Wavefront(2, 1, two, nullptr, true));

        params->setManualWorkgroupSize({workgroup_size_x, workgroup_size_y, 1});
        params->setManualWorkitemCount({NX, NY, NZ});

        CommandKernel commandKernel(command, "BA", params);
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

        auto kgraph = KernelGraph::translate2(command);
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
