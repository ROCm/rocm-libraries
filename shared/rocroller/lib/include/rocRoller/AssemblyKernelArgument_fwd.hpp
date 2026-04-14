// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 */

#pragma once

#include <memory>

namespace rocRoller
{
    struct AssemblyKernelArgument;

    using AssemblyKernelArgumentPtr = std::shared_ptr<AssemblyKernelArgument>;
}
