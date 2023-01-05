
#pragma once

#include <memory>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        class Transformer;

        using TransformerPtr = std::shared_ptr<Transformer>;
    }
}
