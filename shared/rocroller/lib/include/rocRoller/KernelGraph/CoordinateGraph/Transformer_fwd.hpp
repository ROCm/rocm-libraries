// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
