#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordinateTransform::MacroTile;

        using namespace CoordinateTransform;
        namespace Expression = rocRoller::Expression;

#define MAKE_OPERATION_VISITOR(CLS)                                     \
    ControlGraph::Operation visitOperation(ControlGraph::CLS const& op) \
    {                                                                   \
        return op;                                                      \
    }

        struct UpdateParametersVisitor
        {
            UpdateParametersVisitor(std::shared_ptr<CommandParameters> params)
            {
                for(auto const& dim : params->getDimensionInfo())
                {
                    m_new_dimensions.emplace(getTag(dim), dim);
                }
            }

            template <typename T>
            Dimension visitDimension(T const& dim)
            {
                if(m_new_dimensions.count(getTag(dim)) > 0)
                    return m_new_dimensions.at(getTag(dim));
                return dim;
            }

            MAKE_OPERATION_VISITOR(Assign);
            MAKE_OPERATION_VISITOR(Barrier);
            MAKE_OPERATION_VISITOR(ElementOp);
            MAKE_OPERATION_VISITOR(ForLoopOp);
            MAKE_OPERATION_VISITOR(Kernel);
            MAKE_OPERATION_VISITOR(LoadLDSTile);
            MAKE_OPERATION_VISITOR(LoadLinear);
            MAKE_OPERATION_VISITOR(LoadTiled);
            MAKE_OPERATION_VISITOR(LoadVGPR);
            MAKE_OPERATION_VISITOR(Multiply);
            MAKE_OPERATION_VISITOR(StoreLDSTile);
            MAKE_OPERATION_VISITOR(StoreLinear);
            MAKE_OPERATION_VISITOR(StoreTiled);
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(UnrollOp);

        private:
            std::map<TagType, Dimension> m_new_dimensions;
        };
#undef MAKE_OPERATION_VISITOR

        KernelGraph updateParameters(KernelGraph k, std::shared_ptr<CommandParameters> params)
        {
            TIMER(t, "KernelGraph::updateParameters");
            rocRoller::Log::getLogger()->debug("KernelGraph::updateParameters()");
            auto visitor = UpdateParametersVisitor(params);
            return rewriteDimensions(k, visitor);
        }
    }
}
