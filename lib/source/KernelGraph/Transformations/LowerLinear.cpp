/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerLinear.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;

        struct LowerLinearVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerLinearVisitor(ContextPtr context)
                : BaseGraphVisitor(context)
            {
            }

            virtual void visitEdge(KernelGraph&       graph,
                                   KernelGraph const& original,
                                   GraphReindexer&    reindexer,
                                   int                tag,
                                   DataFlow const&    df) override
            {
                // Don't need DataFlow edges to/from Linear anymore
                auto location = original.coordinates.getLocation(tag);

                auto check = std::vector<int>();
                check.insert(check.end(), location.incoming.cbegin(), location.incoming.cend());
                check.insert(check.end(), location.outgoing.cbegin(), location.outgoing.cend());

                bool drop
                    = std::reduce(check.cbegin(), check.cend(), false, [&](bool rv, int index) {
                          auto element   = original.coordinates.getElement(index);
                          auto dimension = std::get<Dimension>(element);
                          return rv || std::holds_alternative<Linear>(dimension);
                      });

                if(!drop)
                {
                    copyEdge(graph, original, reindexer, tag);
                }
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        Assign const&      assign) override
            {
                // if destination isn't Linear, copy this operation
                auto original_linear = original.mapper.get(tag, NaryArgument::DEST);
                auto linear          = original.coordinates.get<Linear>(original_linear);
                if(!linear)
                {
                    BaseGraphVisitor::visitOperation(graph, original, reindexer, tag, assign);
                    return;
                }

                auto new_assign = Assign{assign.regType,
                                         reindexExpression(assign.expression, reindexer),
                                         assign.valueCount};

                auto vgpr = graph.coordinates.addElement(VGPR());

                std::vector<int> coordinate_inputs;
                for(auto const& input : original.coordinates.parentNodes(original_linear))
                {
                    coordinate_inputs.push_back(reindexer.coordinates.at(input));
                }
                graph.coordinates.addElement(DataFlow(), coordinate_inputs, std::vector<int>{vgpr});

                auto new_tag = graph.control.addElement(new_assign);

                for(auto const& input : original.control.parentNodes(tag))
                {
                    graph.control.addElement(Sequence(), {reindexer.control.at(input)}, {new_tag});
                }

                graph.mapper.connect(new_tag, vgpr, NaryArgument::DEST);

                reindexer.control.insert_or_assign(tag, new_tag);
                reindexer.coordinates.insert_or_assign(original_linear, vgpr);
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        LoadLinear const&  oload) override
            {
                auto original_user   = original.mapper.get<User>(tag);
                auto original_linear = original.mapper.get<Linear>(tag);
                auto user            = reindexer.coordinates.at(original_user);
                auto linear          = reindexer.coordinates.at(original_linear);
                auto vgpr            = graph.coordinates.addElement(VGPR());

                auto one            = Expression::literal(1u);
                auto workgroupSizeX = workgroupSize()[0];
                auto numTilesX = (graph.coordinates.get<User>(user)->size + workgroupSizeX - one)
                                 / workgroupSizeX;

                auto workgroup = graph.coordinates.addElement(Workgroup(0, numTilesX));
                auto workitem  = graph.coordinates.addElement(Workitem(0, workgroupSizeX));

                graph.coordinates.addElement(Tile(), {linear}, {workgroup, workitem});
                graph.coordinates.addElement(Forget(), {workgroup, workitem}, {vgpr});
                graph.coordinates.addElement(DataFlow(), {user}, {vgpr});

                auto parent = reindexer.control.at(*original.control.parentNodes(tag).begin());
                auto load   = graph.control.addElement(LoadVGPR(oload.varType));
                graph.control.addElement(Body(), {parent}, {load});

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::lowerLinear(): LoadLinear {} -> LoadVGPR {}: {} -> {}",
                    tag,
                    load,
                    linear,
                    vgpr);

                graph.mapper.connect<User>(load, user);
                graph.mapper.connect<VGPR>(load, vgpr);

                reindexer.control.insert_or_assign(tag, load);
                reindexer.coordinates.insert_or_assign(original_linear, vgpr);
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        StoreLinear const&) override
            {
                auto original_user   = original.mapper.get<User>(tag);
                auto original_linear = original.mapper.get<Linear>(tag);
                auto user            = reindexer.coordinates.at(original_user);
                auto vgpr            = reindexer.coordinates.at(original_linear);

                // look up from User to get connected Linear (which represents an index)
                auto linear = *graph.coordinates
                                   .findNodes(
                                       user,
                                       [&](int tag) -> bool {
                                           auto linear = graph.coordinates.get<Linear>(tag);
                                           return bool(linear);
                                       },
                                       Graph::Direction::Upstream)
                                   .begin();

                auto one            = Expression::literal(1u);
                auto workgroupSizeX = workgroupSize()[0];
                auto numTilesX = (graph.coordinates.get<User>(user)->size + workgroupSizeX - one)
                                 / workgroupSizeX;
                auto workgroup = graph.coordinates.addElement(Workgroup(0, numTilesX));
                auto workitem  = graph.coordinates.addElement(Workitem(0, workgroupSizeX));

                graph.coordinates.addElement(Inherit(), {vgpr}, {workgroup, workitem});
                graph.coordinates.addElement(Flatten(), {workgroup, workitem}, {linear});
                graph.coordinates.addElement(DataFlow(), {vgpr}, {user});

                auto parent = reindexer.control.at(*original.control.parentNodes(tag).begin());
                auto store  = graph.control.addElement(StoreVGPR());
                graph.control.addElement(Sequence(), {parent}, {store});

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::lowerLinear(): StoreLinear {} -> StoreVGPR {}: {} -> {}",
                    tag,
                    store,
                    linear,
                    vgpr);

                graph.mapper.connect<User>(store, user);
                graph.mapper.connect<VGPR>(store, vgpr);

                reindexer.control.insert_or_assign(tag, store);
            }
        };

        KernelGraph LowerLinear::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::lowerLinear");
            auto visitor = LowerLinearVisitor(m_context);
            return rewrite(k, visitor);
        }

    }
}
