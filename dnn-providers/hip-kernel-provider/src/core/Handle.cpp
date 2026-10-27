// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Handle.hpp"

#include "Container.hpp"

hip_kernel_provider::core::ApplicabilityManager& Handle::getEngineManager() const
{
    return container->getEngineManager();
}
