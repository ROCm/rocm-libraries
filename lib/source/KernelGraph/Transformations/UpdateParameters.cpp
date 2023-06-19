
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/Transforms/UpdateParameters.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;

        struct UpdateParametersVisitor
        {
            UpdateParametersVisitor(std::shared_ptr<CommandParameters> params)
            {
                m_newDimensions = params->getDimensionInfo();
            }

            template <typename T>
            Dimension visitDimension(int tag, T const& dim)
            {
                if(m_newDimensions.count(tag) > 0)
                    return m_newDimensions.at(tag);
                return dim;
            }

            template <typename T>
            Operation visitOperation(T const& op)
            {
                return op;
            }

        private:
            std::map<int, Dimension> m_newDimensions;
        };

        KernelGraph UpdateParameters::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::updateParameters");
            rocRoller::Log::getLogger()->debug("KernelGraph::updateParameters()");
            auto visitor = UpdateParametersVisitor(m_params);

            auto kgraph = rewriteDimensions(original, visitor);

            auto counts = m_params->getManualWavefrontCounts();
            if(counts)
            {
                auto wfx = std::get<0>(*counts);
                auto wfy = std::get<1>(*counts);
                auto WF  = Wavefront(-1, Expression::literal(wfx * wfy), nullptr);
                auto WFX = Wavefront(0, Expression::literal(wfx), nullptr);
                auto WFY = Wavefront(1, Expression::literal(wfy), nullptr);
                for(auto tag : kgraph.coordinates.getNodes<Wavefront>())
                {
                    auto wavefront = *kgraph.coordinates.get<Wavefront>(tag);
                    if(wavefront.dim == -1)
                        kgraph.coordinates.setElement(tag, WF);
                    if(wavefront.dim == 0)
                        kgraph.coordinates.setElement(tag, WFX);
                    if(wavefront.dim == 1)
                        kgraph.coordinates.setElement(tag, WFY);
                }
            }

            return kgraph;
        }
    }
}
