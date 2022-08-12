#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

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

            virtual void visitEdge(HyperGraph&,
                                   ControlGraph::ControlGraph&,
                                   Location const&,
                                   DataFlow const&) override
            {
                // NOP; don't need DataFlow edges anymore
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

                auto wg   = CoordinateTransform::Workgroup(load.linear.tag);
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();

                auto wi = CoordinateTransform::Workitem(load.linear.tag, 0, wavefront_size);

                auto vgpr = CoordinateTransform::VGPR(load.linear.tag);

                coordGraph.addEdge({load.linear}, {wg, wi}, CoordinateTransform::Tile());
                coordGraph.addEdge({wg, wi}, {vgpr}, CoordinateTransform::Forget());

                coordGraph.addEdge({load.user}, {vgpr}, CoordinateTransform::DataFlow());
                controlGraph.addEdge(
                    loc.srcs, {ControlGraph::LoadVGPR(load.tag, load.user)}, ControlGraph::Body());
            }

            virtual void visitOperation(HyperGraph&                      coordGraph,
                                        ControlGraph::ControlGraph&      controlGraph,
                                        Location const&                  loc,
                                        ControlGraph::StoreLinear const& store) override
            {
                auto linear = store.linear;
                auto vgpr   = CoordinateTransform::VGPR(linear.tag);

                auto wavefront_size = wavefrontSize();

                auto wg   = CoordinateTransform::Workgroup(linear.tag, 0, true);
                wg.stride = workgroupSize()[0];
                wg.size   = workgroupCountX();

                auto wi = CoordinateTransform::Workitem(linear.tag, 0, wavefront_size, true);

                linear.output = true;

                coordGraph.addEdge({vgpr}, {wg, wi}, CoordinateTransform::Inherit());
                coordGraph.addEdge({wg, wi}, {linear}, CoordinateTransform::Flatten());

                coordGraph.addEdge({vgpr}, {store.user}, CoordinateTransform::DataFlow());
                controlGraph.addEdge({store.user},
                                     {ControlGraph::StoreVGPR(store.tag, store.user)});
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
