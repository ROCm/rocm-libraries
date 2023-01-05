
#include <rocRoller/KernelGraph/KernelGraph.hpp>
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

            LowerLinearVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context)
            {
            }

            std::map<int, int> vgprs;

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
                                        ElementOp const&   op) override
            {
                std::vector<int> coordinate_inputs, coordinate_outputs;
                std::vector<int> control_inputs;

                // if destination isn't Linear, copy this operation
                auto connections = original.mapper.getConnections(tag);
                if(connections[0].tindex != typeid(Linear))
                {
                    copyOperation(graph, original, reindexer, tag);

                    auto new_tag = reindexer.control.at(tag);
                    auto new_op  = graph.control.getNode<ElementOp>(new_tag);
                    new_op.a     = op.a > 0 ? reindexer.coordinates.at(op.a) : op.a;
                    new_op.b     = op.b > 0 ? reindexer.coordinates.at(op.b) : op.b;
                    graph.control.setElement(new_tag, new_op);
                    return;
                }

                ElementOp newOp(op.op, -1, -1);
                newOp.a = vgprs.at(op.a);
                newOp.b = op.b > 0 ? vgprs.at(op.b) : -1;

                auto original_linear = original.mapper.get<Linear>(tag);
                auto vgpr            = graph.coordinates.addElement(VGPR());

                if(newOp.b > 0)
                    graph.coordinates.addElement(DataFlow(), {newOp.a, newOp.b}, {vgpr});
                else
                    graph.coordinates.addElement(DataFlow(), {newOp.a}, {vgpr});

                auto elementOp = graph.control.addElement(newOp);

                for(auto const& input : original.control.parentNodes(tag))
                {
                    graph.control.addElement(
                        Sequence(), {reindexer.control.at(input)}, {elementOp});
                }

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::lowerLinear(): ElementOp {} -> ElementOp {}: {} -> {}: {}/{}",
                    tag,
                    elementOp,
                    original_linear,
                    vgpr,
                    op.a,
                    newOp.a);

                graph.mapper.connect<VGPR>(elementOp, vgpr);

                reindexer.control.insert_or_assign(tag, elementOp);
                vgprs.insert_or_assign(original_linear, vgpr);
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

                auto wg   = Workgroup();
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();
                auto wi   = Workitem(0, wavefrontSize());

                auto wg_tag = graph.coordinates.addElement(wg);
                auto wi_tag = graph.coordinates.addElement(wi);

                graph.coordinates.addElement(Tile(), {linear}, {wg_tag, wi_tag});
                graph.coordinates.addElement(Forget(), {wg_tag, wi_tag}, {vgpr});
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

                vgprs.insert_or_assign(original_linear, vgpr);
                reindexer.control.insert_or_assign(tag, load);
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        LoadVGPR const&    oload) override
            {
                copyOperation(graph, original, reindexer, tag);
                auto vgpr = original.mapper.get<VGPR>(tag);
                vgprs.insert_or_assign(vgpr, reindexer.coordinates.at(vgpr));
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
                auto linear          = reindexer.coordinates.at(original_linear);
                auto vgpr            = vgprs.at(original_linear);

                auto wg   = Workgroup();
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();
                auto wi   = Workitem(0, wavefrontSize());

                auto wg_tag = graph.coordinates.addElement(wg);
                auto wi_tag = graph.coordinates.addElement(wi);

                graph.coordinates.addElement(Inherit(), {vgpr}, {wg_tag, wi_tag});
                graph.coordinates.addElement(Flatten(), {wg_tag, wi_tag}, {linear});
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

        KernelGraph lowerLinear(KernelGraph k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerLinear");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerLinear()");
            auto visitor = LowerLinearVisitor(context);
            return rewrite(k, visitor);
        }

    }
}
