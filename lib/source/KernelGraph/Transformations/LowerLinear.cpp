#include "KernelGraph/ControlHypergraph/ControlEdge.hpp"
#include "KernelGraph/ControlHypergraph/Operation.hpp"
#include "KernelGraph/CoordinateTransform/Dimension.hpp"
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
        using CoordinateTransform::MacroTile;

        using namespace CoordinateTransform;
        namespace Expression = rocRoller::Expression;

        /*************************
         * KernelGraphs rewrites...
         */

        /*
         * Linear distribute
         */

        // Delete this when graph rearch complete
        struct LowerLinearVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerLinearVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context)
            {
            }

            virtual void visitEdge(HyperGraph&,
                                   ControlGraph::ControlGraph&,
                                   Location const&,
                                   MakeOutput const&) override
            {
                // NOP; don't need MakeOutput edges anymore
            }

            virtual void visitEdge(HyperGraph& coordGraph,
                                   ControlGraph::ControlGraph&,
                                   Location const& loc,
                                   DataFlow const& df) override
            {
                // Don't need DataFlow edges to/from Linear anymore
                bool drop = false;
                for(auto const& d : loc.srcDims)
                {
                    if(std::holds_alternative<Linear>(d))
                    {
                        drop = true;
                        break;
                    }
                }
                for(auto const& d : loc.dstDims)
                {
                    if(std::holds_alternative<Linear>(d))
                    {
                        drop = true;
                        break;
                    }
                }
                if(!drop)
                {
                    coordGraph.addEdge(loc.srcDims, loc.dstDims, df);
                }
            }

            virtual void visitOperation(HyperGraph&                    coordGraph,
                                        ControlGraph::ControlGraph&    controlGraph,
                                        Location const&                loc,
                                        ControlGraph::ElementOp const& op) override
            {
                std::vector<Dimension> src, dst;

                for(auto tag : Operations::Inputs()(*op.xop))
                    src.push_back(VGPR(tag));

                for(auto tag : Operations::Outputs()(*op.xop))
                    dst.push_back(VGPR(tag));

                coordGraph.addEdge(src, dst, CoordinateTransform::DataFlow());
                controlGraph.addEdge({src}, {op});
            }

            virtual void visitOperation(HyperGraph&                     coordGraph,
                                        ControlGraph::ControlGraph&     controlGraph,
                                        Location const&                 loc,
                                        ControlGraph::LoadLinear const& load) override
            {
                auto wavefront_size = wavefrontSize();

                auto user   = loc.coordGraph.getDimension(User(load.tag));
                auto linear = loc.coordGraph.getDimension(Linear(load.tag));

                auto wg   = CoordinateTransform::Workgroup(linear.tag);
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();

                auto wi = CoordinateTransform::Workitem(linear.tag, 0, wavefront_size);

                auto vgpr = CoordinateTransform::VGPR(linear.tag);

                coordGraph.addEdge({linear}, {wg, wi}, CoordinateTransform::Tile());
                coordGraph.addEdge({wg, wi}, {vgpr}, CoordinateTransform::Forget());

                coordGraph.addEdge({user}, {vgpr}, CoordinateTransform::DataFlow());
                controlGraph.addEdge(
                    loc.srcs, {ControlGraph::LoadVGPR(load.tag)}, ControlGraph::Body());
            }

            virtual void visitOperation(HyperGraph&                      coordGraph,
                                        ControlGraph::ControlGraph&      controlGraph,
                                        Location const&                  loc,
                                        ControlGraph::StoreLinear const& store) override
            {
                auto wavefront_size = wavefrontSize();

                auto linear = loc.coordGraph.getDimension(Linear(store.tag));
                auto user   = loc.coordGraph.getDimension(User(store.tag, true));

                auto wg   = CoordinateTransform::Workgroup(linear.tag, 0, true);
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();

                auto wi = CoordinateTransform::Workitem(linear.tag, 0, wavefront_size, true);

                auto vgpr = CoordinateTransform::VGPR(linear.tag);

                linear.output = true;

                coordGraph.addEdge({vgpr}, {wg, wi}, CoordinateTransform::Inherit());
                coordGraph.addEdge({wg, wi}, {linear}, CoordinateTransform::Flatten());

                coordGraph.addEdge({vgpr}, {user}, CoordinateTransform::DataFlow());
                controlGraph.addEdge({user}, {ControlGraph::StoreVGPR(store.tag)});
            }
        };

        // Delete this when graph rearch complete
        KernelGraph lowerLinear(KernelGraph k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinear");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerLinear()");
            auto visitor = LowerLinearVisitor(context);
            return rewrite(k, visitor);
        }

        // Rename this when graph rearch complete
        struct LowerLinearVisitor2 : public BaseGraphVisitor2
        {
            using BaseGraphVisitor2::visitEdge;
            using BaseGraphVisitor2::visitOperation;

            LowerLinearVisitor2(std::shared_ptr<Context> context)
                : BaseGraphVisitor2(context)
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
                auto wavefront_size = wavefrontSize();

                auto original_user   = original.mapper.get<CoordGraph::User>(tag);
                auto original_linear = original.mapper.get<CoordGraph::Linear>(tag);
                auto user            = reindexer.coordinates.at(original_user);
                auto linear          = reindexer.coordinates.at(original_linear);

                auto wg_ = CoordGraph::Workgroup();
                // wg_.stride = workgroupSize()[0];
                // wg_.size   = workgroupCountX();

                auto wg   = graph.coordinates.addElement(wg_);
                auto wi   = graph.coordinates.addElement(CoordGraph::Workitem(0, wavefront_size));
                auto vgpr = graph.coordinates.addElement(CoordGraph::VGPR());

                graph.coordinates.addElement(CoordGraph::Tile(), {linear}, {wg, wi});
                graph.coordinates.addElement(CoordGraph::Forget(), {wg, wi}, {vgpr});
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
                auto wavefront_size = wavefrontSize();

                auto original_user   = original.mapper.get<CoordGraph::User>(tag);
                auto original_linear = original.mapper.get<CoordGraph::Linear>(tag);
                auto user            = reindexer.coordinates.at(original_user);
                auto linear          = reindexer.coordinates.at(original_linear);
                auto vgpr            = vgprs.at(original_linear);

                auto wg_ = CoordGraph::Workgroup();
                // wg_.stride = workgroupSize()[0];
                // wg_.size   = workgroupCountX();

                auto wg = graph.coordinates.addElement(wg_);
                auto wi = graph.coordinates.addElement(CoordGraph::Workitem(0, wavefront_size));

                graph.coordinates.addElement(CoordGraph::Inherit(), {vgpr}, {wg, wi});
                graph.coordinates.addElement(CoordGraph::Flatten(), {wg, wi}, {linear});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {vgpr}, {user});

                auto parent = reindexer.control.at(*original.control.parentNodes(tag).begin());
                auto store  = graph.control.addElement(ControlHypergraph::StoreVGPR());
                graph.control.addElement(ControlHypergraph::Sequence(), {parent}, {store});

                graph.mapper.connect<CoordGraph::User>(store, user);
                graph.mapper.connect<CoordGraph::VGPR>(store, vgpr);

                reindexer.control.insert_or_assign(tag, store);
            }
        };

        KernelHypergraph lowerLinear(KernelHypergraph k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinear");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerLinear()");
            auto visitor = LowerLinearVisitor2(context);
            return rewrite(k, visitor);
        }

    }
}
