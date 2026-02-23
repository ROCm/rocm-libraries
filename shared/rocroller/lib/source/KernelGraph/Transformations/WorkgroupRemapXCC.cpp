// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/WorkgroupRemapXCC.hpp>
#include <rocRoller/KernelGraph/Transforms/WorkgroupRemapXCC_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using GD = rocRoller::Graph::Direction;

        namespace WorkgroupRemapXCCDetail
        {
            int remapWorkgroupXCC(rocRoller::KernelGraph::KernelGraph& graph,
                                  int                                  workgroupTag,
                                  uint                                 numXCC)
            {
                auto const direction = danglingDirection(graph, workgroupTag);
                if(not direction.has_value())
                    return -1;

                auto workgroup = graph.coordinates.get<Workgroup>(workgroupTag).value();
                auto size      = workgroup.size;
                // Skip workgroups that do not have size (e.g., these could come from
                // AddStreamK)
                if(size == nullptr)
                {
                    Log::debug("remapWorkgroupXCC: workgroup {} has no size", workgroupTag);
                    return -1;
                }

                using ExpressionPtr     = Expression::ExpressionPtr;
                using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
                using ExpressionPtrVectorPair
                    = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

                auto newWorkgroupTag = graph.coordinates.addElement(Workgroup(0, size));

                auto one           = Expression::literal(1u);
                auto numXCCLiteral = Expression::literal(numXCC);

                auto ceilDiv = [&](ExpressionPtr a, ExpressionPtr b) { return (a + b - one) / b; };

                auto xcc = graph.coordinates.addElement(Linear(numXCCLiteral, nullptr));
                auto cu
                    = graph.coordinates.addElement(Linear(ceilDiv(size, numXCCLiteral), nullptr));

                // 0 argument is XCC, 1 argument is CU
                auto condition
                    = Expression::positionalArgument(0, Register::Type::Scalar, DataType::UInt32)
                      <= (size % numXCCLiteral);

                ExpressionPtrVectorPair strides{{ceilDiv(size, numXCCLiteral), one},
                                                {size / numXCCLiteral, one}};
                ExpressionPtrPair       initialValues{nullptr, size % numXCCLiteral};

                if(direction == GD::Upstream)
                {
                    graph.coordinates.addElement(Tile(), {newWorkgroupTag}, {cu, xcc});
                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, strides, initialValues),
                        {xcc, cu},
                        {workgroupTag});
                }
                else
                {
                    graph.coordinates.addElement(
                        PiecewiseAffineJoin(condition, strides, initialValues),
                        {workgroupTag},
                        {xcc, cu});
                    graph.coordinates.addElement(Flatten(), {cu, xcc}, {newWorkgroupTag});
                }

                return newWorkgroupTag;
            }
        }

        KernelGraph WorkgroupRemapXCC::apply(KernelGraph const& original)
        {
            using namespace WorkgroupRemapXCCDetail;

            auto kgraph = original;

            if(m_workgroupRemapXCC.has_value())
            {
                auto const& arch = m_context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasXCC),
                            "XCC-aware workgroup remapping not available on: ",
                            arch.target().toString());

                auto const workgroups = kgraph.coordinates.getNodes<Workgroup>().to<std::vector>();
                for(auto tag : workgroups)
                    remapWorkgroupXCC(kgraph, tag, m_workgroupRemapXCC.value());
            }

            return kgraph;
        }
    }
}
