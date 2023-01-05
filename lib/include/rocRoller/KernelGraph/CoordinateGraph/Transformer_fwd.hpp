
#pragma once

#include <memory>

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        class Transformer;

        using TransformerPtr = std::shared_ptr<Transformer>;
    }
}
