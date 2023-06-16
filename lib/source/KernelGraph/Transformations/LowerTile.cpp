
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        ConstraintStatus NoConstructDestructMT(const KernelGraph& k)
        {
            ConstraintStatus retval;
            for(auto element : k.coordinates.getEdges<CoordinateTransformEdge>())
            {
                auto dmt = k.coordinates.get<DestructMacroTile>(element);
                if(dmt)
                {
                    retval.combine(
                        false,
                        concatenate(
                            "DestructMacroTile Coordinate Edge Still Exists: ", element, "."));
                }
                else
                {
                    auto cmt = k.coordinates.get<ConstructMacroTile>(element);
                    if(cmt)
                    {
                        retval.combine(
                            false,
                            concatenate(
                                "ConstructMacroTile Coordinate Edge Still Exists: ", element, "."));
                    }
                }
            }
            return retval;
        }

        /*
         * Lower tile ops
         */

        struct LowerTileVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerTileVisitor(std::shared_ptr<CommandParameters> params, ContextPtr context)
                : BaseGraphVisitor(context)
                , m_params(params)
                , m_kernel(context->kernel())
            {
            }

            virtual void visitEdge(KernelGraph&              graph,
                                   KernelGraph const&        original,
                                   GraphReindexer&           reindexer,
                                   int                       tag,
                                   ConstructMacroTile const& edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitEdge(KernelGraph&             graph,
                                   KernelGraph const&       original,
                                   GraphReindexer&          reindexer,
                                   int                      tag,
                                   DestructMacroTile const& edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        LoadTiled const&   oload) override
            {
                auto logger = rocRoller::Log::getLogger();
                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled({})", tag);

                auto original_user     = original.mapper.get<User>(tag);
                auto original_mac_tile = original.mapper.get<MacroTile>(tag);
                auto user              = reindexer.coordinates.at(original_user);
                auto mac_tile          = reindexer.coordinates.at(original_mac_tile);

                auto sdims
                    = original.coordinates
                          .getInputNodeIndices(original_mac_tile, CT::isEdge<ConstructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto load = reindexer.control.at(tag);

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();

                loadMacroTile(graph,
                              load,
                              user,
                              mac_tile,
                              sdims,
                              workgroupSizes,
                              wavefrontSize,
                              wavetilesPerWorkgroup);
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        StoreTiled const&  ostore) override
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::LowerTileVisitor::StoreTiled({})",
                                                   tag);

                auto original_user     = original.mapper.get<User>(tag);
                auto original_mac_tile = original.mapper.get<MacroTile>(tag);
                auto user              = reindexer.coordinates.at(original_user);
                auto mac_tile          = reindexer.coordinates.at(original_mac_tile);

                auto sdims
                    = original.coordinates
                          .getOutputNodeIndices(original_mac_tile, CT::isEdge<DestructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto store = reindexer.control.at(tag);

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();

                storeMacroTile(graph,
                               store,
                               user,
                               mac_tile,
                               sdims,
                               workgroupSizes,
                               wavefrontSize,
                               wavetilesPerWorkgroup);
            }

        private:
            std::shared_ptr<AssemblyKernel>    m_kernel;
            std::shared_ptr<CommandParameters> m_params;
        };

        KernelGraph LowerTile::apply(KernelGraph const& graph)
        {
            TIMER(t, "KernelGraph::lowerTile");
            auto visitor = LowerTileVisitor(m_params, m_context);
            return rewrite(graph, visitor);
        }
    }
}
