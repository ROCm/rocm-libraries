#pragma once

#include <memory>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class GraphTransform;

        using GraphTransformPtr = std::shared_ptr<GraphTransform>;
    }
}
