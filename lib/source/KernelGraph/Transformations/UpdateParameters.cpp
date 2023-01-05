#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;
        namespace Expression = rocRoller::Expression;

        struct UpdateParametersVisitor
        {
            UpdateParametersVisitor(std::shared_ptr<CommandParameters> params)
            {
                m_new_dimensions = params->getDimensionInfo();
            }

            template <typename T>
            Dimension visitDimension(int tag, T const& dim)
            {
                if(m_new_dimensions.count(tag) > 0)
                    return m_new_dimensions.at(tag);
                return dim;
            }

            template <typename T>
            Operation visitOperation(T const& op)
            {
                return op;
            }

        private:
            std::map<int, Dimension> m_new_dimensions;
        };

        KernelGraph updateParameters(KernelGraph k, std::shared_ptr<CommandParameters> params)
        {
            TIMER(t, "KernelGraph::updateParameters");
            rocRoller::Log::getLogger()->debug("KernelGraph::updateParameters()");
            auto visitor = UpdateParametersVisitor(params);
            return rewriteDimensions(k, visitor);
        }
    }
}
