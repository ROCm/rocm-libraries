// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocRoller
{
    namespace KernelGraph
    {
        class GraphTransform;

        using GraphTransformPtr = std::shared_ptr<GraphTransform>;
    }
}
