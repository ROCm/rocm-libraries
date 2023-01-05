#pragma once

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::ControlHypergraph
    {
        struct Sequence;
        struct Body; // Of kernel, for loop, if, etc.

        struct Initialize;
        struct ForLoopIncrement;

        using ControlEdge = std::variant<Sequence, Initialize, ForLoopIncrement, Body>;

        template <typename T>
        concept CControlEdge = std::constructible_from<ControlEdge, T>;

    }
}
