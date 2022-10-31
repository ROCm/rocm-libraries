#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordGraph::MacroTile;
        using namespace CoordGraph;
        namespace Expression = rocRoller::Expression;

#define MAKE_OPERATION_VISITOR(CLS)                                     \
    ControlGraph::Operation visitOperation(ControlGraph::CLS const& op) \
    {                                                                   \
        return op;                                                      \
    }

        // TODO Delete this when graph rearch complete
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
            CoordinateTransform::Dimension visitDimension(T const& dim)
            {
                if(m_new_dimensions.count(getTag(dim)) > 0)
                    return m_new_dimensions.at(getTag(dim));
                return dim;
            }

            ControlGraph::Operation visitOperation(ControlGraph::StoreTiled const& op)
            {
                return ControlGraph::StoreTiled(op.tag, op.dataType);
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
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(UnrollOp);

        private:
            std::map<TagType, CoordinateTransform::Dimension> m_new_dimensions;
        };
#undef MAKE_OPERATION_VISITOR

        // TODO Rename this when graph rearch complete
#define MAKE_OPERATION_VISITOR2(CLS)                                              \
    ControlHypergraph::Operation visitOperation(ControlHypergraph::CLS const& op) \
    {                                                                             \
        return op;                                                                \
    }
        // TODO Rename this when graph rearch complete
        struct UpdateParametersVisitor2
        {
            UpdateParametersVisitor2(std::shared_ptr<CommandParameters> params)
            {
                m_new_dimensions = params->getDimensionInfo2();
            }

            template <typename T>
            Dimension visitDimension(int tag, T const& dim)
            {
                if(m_new_dimensions.count(tag) > 0)
                    return m_new_dimensions.at(tag);
                return dim;
            }

            MAKE_OPERATION_VISITOR2(Assign);
            MAKE_OPERATION_VISITOR2(Barrier);
            MAKE_OPERATION_VISITOR2(ElementOp);
            MAKE_OPERATION_VISITOR2(ForLoopOp);
            MAKE_OPERATION_VISITOR2(Kernel);
            MAKE_OPERATION_VISITOR2(LoadLDSTile);
            MAKE_OPERATION_VISITOR2(LoadLinear);
            MAKE_OPERATION_VISITOR2(LoadTiled);
            MAKE_OPERATION_VISITOR2(LoadVGPR);
            MAKE_OPERATION_VISITOR2(Multiply);
            MAKE_OPERATION_VISITOR2(StoreLDSTile);
            MAKE_OPERATION_VISITOR2(StoreLinear);
            MAKE_OPERATION_VISITOR2(StoreTiled);
            MAKE_OPERATION_VISITOR2(StoreVGPR);
            MAKE_OPERATION_VISITOR2(TensorContraction);
            MAKE_OPERATION_VISITOR2(UnrollOp);

        private:
            std::map<int, Dimension> m_new_dimensions;
        };
#undef MAKE_OPERATION_VISITOR2

        KernelGraph updateParameters(KernelGraph k, std::shared_ptr<CommandParameters> params)
        {
            TIMER(t, "KernelGraph::updateParameters");
            rocRoller::Log::getLogger()->debug("KernelGraph::updateParameters()");
            auto visitor = UpdateParametersVisitor(params);
            return rewriteDimensions(k, visitor);
        }

        KernelHypergraph updateParameters(KernelHypergraph                   k,
                                          std::shared_ptr<CommandParameters> params)
        {
            TIMER(t, "KernelGraph::updateParameters");
            rocRoller::Log::getLogger()->debug("KernelGraph::updateParameters()");
            auto visitor = UpdateParametersVisitor2(params);
            return rewriteDimensions(k, visitor);
        }
    }
}
