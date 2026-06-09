/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#include "hipblaslt_bench_options.hpp"

namespace hipblaslt_bench_options
{
    int32_t& sm_count_target()
    {
        static int32_t v = 0;
        return v;
    }

    bool& dyn_persistent_tile_enabled()
    {
        static bool v = false;
        return v;
    }
}
