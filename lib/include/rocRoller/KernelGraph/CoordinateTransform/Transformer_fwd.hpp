
#pragma once

#include <memory>

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
    {
        class Transformer;

        using TransformerPtr = std::shared_ptr<Transformer>;
    }
}
