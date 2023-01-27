
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
                auto connections = original.mapper.getConnections(tag);
                auto linear      = original.coordinates.get<Linear>(connections[0].coordinate);
                if(!linear)
                {
                    BaseGraphVisitor::visitOperation(graph, original, reindexer, tag, assign);
                    return;
                }

                ReindexExpressionVisitor visitor(reindexer);
                auto                     new_assign
                    = Assign{assign.regType, visitor.call(assign.expression), assign.valueCount};

                auto original_linear = original.mapper.get<Linear>(tag);
                auto vgpr            = graph.coordinates.addElement(VGPR());

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

                graph.mapper.connect<VGPR>(new_tag, vgpr);

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
