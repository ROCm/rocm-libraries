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

    int32_t& dyn_persistent_tile_mode()
    {
        static int32_t v = -1;
        return v;
    }

    std::string& dyn_persistent_tile_mode_str()
    {
        static std::string v;
        return v;
    }
}
