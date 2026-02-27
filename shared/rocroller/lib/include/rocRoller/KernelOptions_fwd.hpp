// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace rocRoller
{
    enum class DSObserverType
    {
        DSMEMObserver,
        WeightlessDSMemObserver,
        Count
    };

    std::string toString(DSObserverType);

    struct KernelOptionValues;

    class KernelOptions;
}
