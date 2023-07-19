#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        struct Sequence;
        struct Body; // Of kernel, for loop, if, etc.
        struct Else; // Alternative body for false conditional

        struct Initialize;
        struct ForLoopIncrement;

        using ControlEdge = std::variant<Sequence, Initialize, ForLoopIncrement, Body, Else>;

        template <typename T>
        concept CControlEdge = std::constructible_from<ControlEdge, T>;

        template <typename T>
        concept CConcreteControlEdge = (CControlEdge<T> && !std::same_as<ControlEdge, T>);

    }
}
