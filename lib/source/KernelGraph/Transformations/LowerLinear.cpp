#include "KernelGraph/ControlHypergraph/ControlEdge.hpp"
#include "KernelGraph/ControlHypergraph/Operation.hpp"
#include "KernelGraph/CoordGraph/Dimension.hpp"
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

#include <rocRoller/KernelGraph/CoordGraph/Edge.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;

        /*************************
         * KernelGraphs rewrites...
         */

        /*
         * Linear distribute
         */

        struct LowerLinearVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerLinearVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context)
            {
            }

            std::map<int, int> vgprs;

            virtual void visitEdge(KernelHypergraph&           graph,
                                   KernelHypergraph const&     original,
                                   GraphReindexer&             reindexer,
                                   int                         tag,
                                   CoordGraph::DataFlow const& df) override
            {
                // Don't need DataFlow edges to/from Linear anymore
                auto location = original.coordinates.getLocation(tag);

                auto check = std::vector<int>();
                check.insert(check.end(), location.incoming.cbegin(), location.incoming.cend());
                check.insert(check.end(), location.outgoing.cbegin(), location.outgoing.cend());

                bool drop
                    = std::reduce(check.cbegin(), check.cend(), false, [&](bool rv, int index) {
                          auto element   = original.coordinates.getElement(index);
                          auto dimension = std::get<CoordGraph::Dimension>(element);
                          return rv || std::holds_alternative<CoordGraph::Linear>(dimension);
                      });

                if(!drop)
                {
                    copyEdge(graph, original, reindexer, tag);
                }
            }

            virtual void visitOperation(KernelHypergraph&                   graph,
                                        KernelHypergraph const&             original,
                                        GraphReindexer&                     reindexer,
                                        int                                 tag,
                                        ControlHypergraph::ElementOp const& op) override
            {
                std::vector<int> coordinate_inputs, coordinate_outputs;
                std::vector<int> control_inputs;

                // if destination isn't Linear, copy this operation
                auto connections = original.mapper.getConnections(tag);
                if(connections[0].tindex != typeid(CoordGraph::Linear))
                {
                    copyOperation(graph, original, reindexer, tag);

                    auto new_tag = reindexer.control.at(tag);
                    auto new_op  = graph.control.getNode<ControlHypergraph::ElementOp>(new_tag);
                    new_op.a     = op.a > 0 ? reindexer.coordinates.at(op.a) : op.a;
                    new_op.b     = op.b > 0 ? reindexer.coordinates.at(op.b) : op.b;
                    graph.control.setElement(new_tag, new_op);
                    return;
                }

                ControlHypergraph::ElementOp newOp(op.op, -1, -1);
                newOp.a = vgprs.at(op.a);
                newOp.b = op.b > 0 ? vgprs.at(op.b) : -1;

                auto original_linear = original.mapper.get<CoordGraph::Linear>(tag);
                auto vgpr            = graph.coordinates.addElement(CoordGraph::VGPR());

                if(newOp.b > 0)
                    graph.coordinates.addElement(
                        CoordGraph::DataFlow(), {newOp.a, newOp.b}, {vgpr});
                else
                    graph.coordinates.addElement(CoordGraph::DataFlow(), {newOp.a}, {vgpr});

                auto elementOp = graph.control.addElement(newOp);

                for(auto const& input : original.control.parentNodes(tag))
                {
                    graph.control.addElement(
                        ControlHypergraph::Sequence(), {reindexer.control.at(input)}, {elementOp});
                }

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::lowerLinear(): ElementOp {} -> ElementOp {}: {} -> {}: {}/{}",
                    tag,
                    elementOp,
                    original_linear,
                    vgpr,
                    op.a,
                    newOp.a);

                graph.mapper.connect<CoordGraph::VGPR>(elementOp, vgpr);

                reindexer.control.insert_or_assign(tag, elementOp);
                vgprs.insert_or_assign(original_linear, vgpr);
            }

            virtual void visitOperation(KernelHypergraph&                    graph,
                                        KernelHypergraph const&              original,
                                        GraphReindexer&                      reindexer,
                                        int                                  tag,
                                        ControlHypergraph::LoadLinear const& oload) override
            {
                auto original_user   = original.mapper.get<CoordGraph::User>(tag);
                auto original_linear = original.mapper.get<CoordGraph::Linear>(tag);
                auto user            = reindexer.coordinates.at(original_user);
                auto linear          = reindexer.coordinates.at(original_linear);
                auto vgpr            = graph.coordinates.addElement(CoordGraph::VGPR());

                auto wg   = CoordGraph::Workgroup();
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();
                auto wi   = CoordGraph::Workitem(0, wavefrontSize());

                auto wg_tag = graph.coordinates.addElement(wg);
                auto wi_tag = graph.coordinates.addElement(wi);

                graph.coordinates.addElement(CoordGraph::Tile(), {linear}, {wg_tag, wi_tag});
                graph.coordinates.addElement(CoordGraph::Forget(), {wg_tag, wi_tag}, {vgpr});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {user}, {vgpr});

                auto parent = reindexer.control.at(*original.control.parentNodes(tag).begin());
                auto load   = graph.control.addElement(ControlHypergraph::LoadVGPR(oload.varType));
                graph.control.addElement(ControlHypergraph::Body(), {parent}, {load});

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::lowerLinear(): LoadLinear {} -> LoadVGPR {}: {} -> {}",
                    tag,
                    load,
                    linear,
                    vgpr);

                graph.mapper.connect<CoordGraph::User>(load, user);
                graph.mapper.connect<CoordGraph::VGPR>(load, vgpr);

                vgprs.insert_or_assign(original_linear, vgpr);
                reindexer.control.insert_or_assign(tag, load);
            }

            virtual void visitOperation(KernelHypergraph&                  graph,
                                        KernelHypergraph const&            original,
                                        GraphReindexer&                    reindexer,
                                        int                                tag,
                                        ControlHypergraph::LoadVGPR const& oload) override
            {
                copyOperation(graph, original, reindexer, tag);
                auto vgpr = original.mapper.get<CoordGraph::VGPR>(tag);
                vgprs.insert_or_assign(vgpr, reindexer.coordinates.at(vgpr));
            }

            virtual void visitOperation(KernelHypergraph&       graph,
                                        KernelHypergraph const& original,
                                        GraphReindexer&         reindexer,
                                        int                     tag,
                                        ControlHypergraph::StoreLinear const&) override
            {
                auto original_user   = original.mapper.get<CoordGraph::User>(tag);
                auto original_linear = original.mapper.get<CoordGraph::Linear>(tag);
                auto user            = reindexer.coordinates.at(original_user);
                auto linear          = reindexer.coordinates.at(original_linear);
                auto vgpr            = vgprs.at(original_linear);

                auto wg   = CoordGraph::Workgroup();
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();
                auto wi   = CoordGraph::Workitem(0, wavefrontSize());

                auto wg_tag = graph.coordinates.addElement(wg);
                auto wi_tag = graph.coordinates.addElement(wi);

                graph.coordinates.addElement(CoordGraph::Inherit(), {vgpr}, {wg_tag, wi_tag});
                graph.coordinates.addElement(CoordGraph::Flatten(), {wg_tag, wi_tag}, {linear});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {vgpr}, {user});

                auto parent = reindexer.control.at(*original.control.parentNodes(tag).begin());
                auto store  = graph.control.addElement(ControlHypergraph::StoreVGPR());
                graph.control.addElement(ControlHypergraph::Sequence(), {parent}, {store});

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::lowerLinear(): StoreLinear {} -> StoreVGPR {}: {} -> {}",
                    tag,
                    store,
                    linear,
                    vgpr);

                graph.mapper.connect<CoordGraph::User>(store, user);
                graph.mapper.connect<CoordGraph::VGPR>(store, vgpr);

                reindexer.control.insert_or_assign(tag, store);
            }
        };

        KernelHypergraph lowerLinear(KernelHypergraph k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinear");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerLinear()");
            auto visitor = LowerLinearVisitor(context);
            return rewrite(k, visitor);
        }

    }
}
