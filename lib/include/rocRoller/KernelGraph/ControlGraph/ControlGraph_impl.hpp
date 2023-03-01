#pragma once

#include "ControlGraph.hpp"

namespace rocRoller::KernelGraph::ControlGraph
{

    template <typename T>
    requires(std::constructible_from<ControlGraph::Element,
                                     T>) inline std::optional<T> ControlGraph::get(int tag) const
    {
        auto x = getElement(tag);
        if constexpr(std::constructible_from<ControlEdge, T>)
        {
            if(std::holds_alternative<ControlEdge>(x))
            {
                if(std::holds_alternative<T>(std::get<ControlEdge>(x)))
                {
                    return std::get<T>(std::get<ControlEdge>(x));
                }
            }
        }
        if constexpr(std::constructible_from<Operation, T>)
        {
            if(std::holds_alternative<Operation>(x))
            {
                if(std::holds_alternative<T>(std::get<Operation>(x)))
                {
                    return std::get<T>(std::get<Operation>(x));
                }
            }
        }
        return {};
    }

    template <typename T>
    inline bool isOperation(auto const& x)
    {
        if constexpr(std::constructible_from<Operation, T>)
        {
            if(std::holds_alternative<Operation>(x))
            {
                if(std::holds_alternative<T>(std::get<Operation>(x)))
                    return true;
            }
        }
        return false;
    }

    template <typename T>
    inline bool isEdge(auto const& x)
    {
        if constexpr(std::constructible_from<ControlEdge, T>)
        {
            if(std::holds_alternative<ControlEdge>(x))
            {
                if(std::holds_alternative<T>(std::get<ControlEdge>(x)))
                    return true;
            }
        }
        return false;
    }

}
