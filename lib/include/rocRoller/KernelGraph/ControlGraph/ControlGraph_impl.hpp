#pragma once

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {

        template <typename T>
        inline std::vector<T> ControlGraph::findOperations() const
        {
            std::vector<T> rv;
            for(auto const& kv : m_nodes)
            {
                if(std::holds_alternative<T>(kv.second))
                    rv.push_back(std::get<T>(kv.second));
            }
            return rv;
        }

        template <CControlEdge T>
        std::vector<Operation> ControlGraph::getInputs(TagType const& dst) const
        {
            return getOperations(getInputTags<T>(dst));
        }

        template <CControlEdge T>
        std::vector<TagType> ControlGraph::getInputTags(TagType const& dst) const
        {
            std::vector<TagType> rv;

            for(auto const& kv : m_edges)
                if(kv.first.second == dst && std::holds_alternative<T>(kv.second))
                    rv.push_back(kv.first.first);

            return rv;
        }

        template <CControlEdge T>
        std::vector<Operation> ControlGraph::getOutputs(TagType const& src) const
        {
            return getOperations(getOutputTags<T>(src));
        }

        template <CControlEdge T>
        std::vector<TagType> ControlGraph::getOutputTags(TagType const& src) const
        {
            std::vector<TagType> rv;
            for(auto const& kv : m_edges)
            {
                if(kv.first.first == src && std::holds_alternative<T>(kv.second))
                {
                    rv.push_back(kv.first.second);
                }
            }
            return rv;
        }
    }
}
