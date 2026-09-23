// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common/mxDataGen.hpp"

namespace rocRoller
{
    void setOptions(DataGeneratorOptions& opts,
                    const float           min,
                    const float           max,
                    int                   blockScaling,
                    const DataInitMode    initMode)
    {
        opts.initMode     = initMode;
        opts.min          = min;
        opts.max          = max;
        opts.blockScaling = blockScaling;
    }
}
