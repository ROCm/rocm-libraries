// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/Transforms/WorkgroupRemapXCC.hpp>
#include <rocRoller/KernelGraph/Transforms/WorkgroupRemapXCC_detail.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"
#include "common/Utilities.hpp"

namespace WorkgroupRemapXCCTest
{
    class RemapWorkgroupXCCKernel : public AssemblyTestKernel
    {
        using GD = rocRoller::Graph::Direction;

    public:
        RemapWorkgroupXCCKernel(rocRoller::ContextPtr context, uint numXCC, uint size)
            : AssemblyTestKernel(context)
            , m_numXCC(numXCC)
            , m_size(size)
        {
            makeGraph();
        }

        std::tuple<rocRoller::KernelGraph::KernelGraphPtr, int, int>
            getGraphAndWorkgroups(GD direction)
        {
            // Downstream (forward) transform, return Upstream nodes
            if(direction == GD::Downstream)
                return {m_graph, m_workgroupU, m_newWorkgroupU};

            return {m_graph, m_workgroupD, m_newWorkgroupD};
        }

        rocRoller::Expression::ExpressionPtr kernelRemapWorkgroupExpression()
        {
            using namespace rocRoller::KernelGraph::CoordinateGraph;
            using GD = rocRoller::Graph::Direction;

            auto transformer = Transformer(&m_graph->coordinates);
            transformer.fillExecutionCoordinates(m_context);

            auto wgRegister = m_context->registerTagManager()->getRegister(m_newWorkgroupU);

            auto exprs = m_graph->coordinates.forward(
                {wgRegister->expression()}, {m_newWorkgroupU}, {m_workgroupU});

            return exprs[0];
        }

        uint reference(uint wg)
        {
            auto fstride = ((m_size + m_numXCC - 1) / m_numXCC);
            uint r       = (wg % m_numXCC) * fstride + (wg / m_numXCC);
            if(wg % m_numXCC > m_size % m_numXCC)
                r -= (wg % m_numXCC) - m_size % m_numXCC;
            return r;
        }

        void generate() override
        {
            using namespace rocRoller;

            auto kernel = m_context->kernel();

            kernel->addArgument({"result",
                                 {DataType::Int32, PointerType::PointerGlobal},
                                 DataDirection::WriteOnly});

            m_context->schedule(kernel->preamble());
            m_context->schedule(kernel->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result;
                co_yield m_context->argLoader()->getValue("result", s_result);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                auto expr = kernelRemapWorkgroupExpression();

                co_yield v_result->allocate();
                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");
                auto wgIndex = m_context->kernel()->workgroupIndex()[0];
                co_yield Expression::generate(v_result,
                                              v_result->expression()
                                                  + wgIndex->expression() * Expression::literal(4),
                                              m_context);

                Register::ValuePtr s_remappedWG;
                co_yield Expression::generate(s_remappedWG, expr, m_context);

                auto v_remappedWG = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, {DataType::Int32}, 1);
                co_yield m_context->copier()->copy(v_remappedWG, s_remappedWG, "Move value");

                co_yield m_context->mem()->storeGlobal(
                    v_result, v_remappedWG, 0, DataTypeInfo::Get(DataType::Int32).elementBytes);
            };

            m_context->schedule(kb());
            m_context->schedule(kernel->postamble());
            m_context->schedule(kernel->amdgpu_metadata());
        }

    private:
        void makeGraph()
        {
            using namespace rocRoller::Expression;
            using namespace rocRoller::KernelGraph;
            using namespace rocRoller::KernelGraph::CoordinateGraph;
            using namespace rocRoller::KernelGraph::WorkgroupRemapXCCDetail;

            KernelGraph graph;

            m_workgroupU = graph.coordinates.addElement(Workgroup(0, literal(m_size)));
            m_workgroupD = graph.coordinates.addElement(Workgroup(0, literal(m_size)));

            auto middleLinear = graph.coordinates.addElement(Linear());

            graph.coordinates.addElement(PassThrough(), {m_workgroupU}, {middleLinear});
            graph.coordinates.addElement(PassThrough(), {middleLinear}, {m_workgroupD});

            /* coordinate graph is:
             *                           Workgroup(0)
             *                               |
             *                           PassThrough
             *                               |
             *                             Linear
             *                               |
             *                           PassThrough
             *                               |
             *                          Workgroup(0)
             */

            m_newWorkgroupD = remapWorkgroupXCC(graph, m_workgroupD, m_numXCC);
            m_newWorkgroupU = remapWorkgroupXCC(graph, m_workgroupU, m_numXCC);

            m_graph = std::make_shared<KernelGraph>(graph);
        }

        rocRoller::KernelGraph::KernelGraphPtr m_graph;
        int m_workgroupU, m_workgroupD, m_newWorkgroupU, m_newWorkgroupD;

        uint m_size, m_numXCC;
    };

    TEST_CASE("Remap Workgroup XCC", "[kernel-graph]")
    {
        using namespace rocRoller::Expression;
        using GD = rocRoller::Graph::Direction;

        uint numXCC = 8u;
        uint size   = 55u * numXCC + 3u;

        RemapWorkgroupXCCKernel kernel(nullptr, numXCC, size);

        auto direction = GENERATE(GD::Upstream, GD::Downstream);

        std::map<int, int> coverage;
        for(uint wg = 0; wg < size; ++wg)
        {
            uint expectedWG = kernel.reference(wg);

            uint remappedWG;
            if(direction == GD::Downstream)
            {
                auto [graph, workgroup, newWorkgroup] = kernel.getGraphAndWorkgroups(direction);
                auto exprs = graph->coordinates.forward({literal(wg)}, {newWorkgroup}, {workgroup});
                remappedWG = getUnsignedInt(evaluate(exprs[0]));
            }
            else
            {
                auto [graph, workgroup, newWorkgroup] = kernel.getGraphAndWorkgroups(direction);
                auto exprs = graph->coordinates.reverse({literal(wg)}, {workgroup}, {newWorkgroup});
                remappedWG = getUnsignedInt(evaluate(exprs[0]));
            }

            CHECK(!coverage.contains(remappedWG));
            CHECK(remappedWG == expectedWG);
            coverage[remappedWG] = wg;
        }

        CHECK(coverage.size() == size);
        for(uint wg = 0; wg < size; ++wg)
            CHECK(coverage.contains(wg));
    }

    TEST_CASE("Remap Workgroup XCC GPU", "[kernel-graph][gpu]")
    {
        uint numXCC = 8u;
        uint size   = 55u * numXCC + 3u;

        auto context = TestContext::ForTestDevice();
        auto kernel  = RemapWorkgroupXCCKernel(context.get(), numXCC, size);

        // Launch kernel
        std::vector<uint> result(size);
        {
            auto d_result   = make_shared_device(result);
            auto invocation = rocRoller::KernelInvocation{{size, 1, 1}, {1, 1, 1}, 0};
            kernel(invocation, d_result.get());

            CHECK_THAT(
                hipMemcpy(result.data(), d_result.get(), sizeof(uint) * size, hipMemcpyDefault),
                HasHipSuccess(0));
        }

        // Check result
        std::map<int, int> coverage;
        for(uint wg = 0; wg < size; ++wg)
        {
            uint expectedWG = kernel.reference(wg);
            uint remappedWG = result[wg];
            CHECK(!coverage.contains(remappedWG));
            CHECK(remappedWG == expectedWG);
            coverage[remappedWG] = wg;
        }

        CHECK(coverage.size() == size);
        for(uint wg = 0; wg < size; ++wg)
            CHECK(coverage.contains(wg));
    }
}
