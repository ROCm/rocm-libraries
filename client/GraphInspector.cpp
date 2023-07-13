
#include "include/GraphInspector.hpp"

namespace rocRoller
{
    namespace Client
    {

        GraphInspector::GraphInspector(CommandPtr             command,
                                       CommandKernel&         kernel,
                                       KernelArguments const& runtimeArgs)
            : m_command(command)
            , m_kernel(kernel)
            , m_runtimeArgs(runtimeArgs)
            , m_kgraph(kernel.getKernelGraph())
            , m_coords(std::make_shared<KernelGraph::CoordinateGraph::CoordinateGraph>(
                  m_kgraph.coordinates))
            , m_tx(m_coords, kernel.getContext())
            , m_invocation(kernel.getKernelInvocation(runtimeArgs.runtimeArguments()))
        {
            m_tx.fillExecutionCoordinates();
            assignLiteralSizesAndStrides();
        }

        void GraphInspector::assignLiteralSizesAndStrides()
        {
            m_argValues = m_command->readArguments(m_runtimeArgs.runtimeArguments());

            auto pred = isNode<KernelGraph::CoordinateGraph::SubDimension>(m_coords);
            for(auto const& idx : m_coords->findElements(pred))
            {
                auto& el
                    = const_cast<typename KernelGraph::CoordinateGraph::CoordinateGraph::Element&>(
                        m_coords->getElement(idx));

                auto& subdim = std::get<KernelGraph::CoordinateGraph::SubDimension>(
                    std::get<KernelGraph::CoordinateGraph::Dimension>(el));

                auto sizeName = toString(subdim.size);

                if(m_argValues.count(sizeName) > 0)
                    subdim.size = Expression::literal(m_argValues.at(sizeName));

                auto strideName = toString(subdim.stride);
                if(m_argValues.count(strideName) > 0)
                    subdim.stride = Expression::literal(m_argValues.at(strideName));
            }
        }

        int GraphInspector::findLoadStoreTile(std::string const& argName)
        {
            auto predicate = [this, &argName](int idx) {
                if(m_coords->getElementType(idx) != Graph::ElementType::Node)
                    return false;

                auto const& node
                    = std::get<KernelGraph::CoordinateGraph::Dimension>(m_coords->getElement(idx));

                if(!std::holds_alternative<KernelGraph::CoordinateGraph::User>(node))
                    return false;

                auto const& user = std::get<KernelGraph::CoordinateGraph::User>(node);

                return user.argumentName == argName;
            };

            auto coords = m_coords->findElements(predicate).to<std::vector>();
            AssertFatal(coords.size() == 1, ShowValue(coords.size()), ShowValue(argName));

            return coords[0];
        }

        int GraphInspector::findLoadTile(int userTag)
        {
            return findLoadStoreTile(concatenate("Load_Tiled_", userTag, "_pointer"));
        }

        int GraphInspector::findStoreTile(int userTag)
        {
            return findLoadStoreTile(concatenate("Store_Tiled_", userTag, "_pointer"));
        }

        void GraphInspector::inventExecutionCoordinates()
        {
            setCoordinate<KernelGraph::CoordinateGraph::ForLoop>(5);
            setCoordinate<KernelGraph::CoordinateGraph::Workgroup>(0);
        }

        size_t GraphInspector::getLoadIndex(int coord)
        {
            std::vector<int> coords{coord};
            auto             exps = m_tx.reverse(coords);
            AssertFatal(exps.size() == 1);

            auto val = Expression::evaluate(exps[0]);

            return std::visit(to_size_t, val);
        }

        size_t GraphInspector::getStoreIndex(int coord)
        {
            std::vector<int> coords{coord};
            auto             exps = m_tx.forward(coords);
            AssertFatal(exps.size() == 1);

            auto val = Expression::evaluate(exps[0]);

            return std::visit(to_size_t, val);
        }

        std::tuple<int, int, int> GraphInspector::getMacroTileSizes() const
        {
            int macM = -1;
            int macN = -1;
            int macK = -1;

            for(auto const& macroTileTag :
                m_kernel.getKernelGraph()
                    .coordinates.getNodes<KernelGraph::CoordinateGraph::MacroTile>()
                    .to<std::vector>())
            {
                auto macroTile = m_kernel.getKernelGraph()
                                     .coordinates.getNode<KernelGraph::CoordinateGraph::MacroTile>(
                                         macroTileTag);
                if(macroTile.layoutType == LayoutType::MATRIX_A)
                {
                    macM = macroTile.sizes[0];
                    AssertFatal(macK == macroTile.sizes[1] || macK == -1);
                    macK = macroTile.sizes[1];
                }
                else if(macroTile.layoutType == LayoutType::MATRIX_B)
                {
                    macN = macroTile.sizes[1];
                    AssertFatal(macK == macroTile.sizes[0] || macK == -1);
                    macK = macroTile.sizes[0];
                }
                if(macM > 0 && macK > 0 && macN > 0)
                {
                    break;
                }
            }
            AssertFatal(macM > 0);
            AssertFatal(macN > 0);
            AssertFatal(macK > 0);

            return std::tuple<int, int, int>{macM, macN, macK};
        }

    }
}
