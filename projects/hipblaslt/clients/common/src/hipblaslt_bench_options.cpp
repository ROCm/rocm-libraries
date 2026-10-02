/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/
#include "hipblaslt_bench_options.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace hipblaslt_bench_options
{
    int32_t& sm_count_target()
    {
        static int32_t v = 0;
        return v;
    }

    int32_t& streamk_tile_scheduling_mode()
    {
        static int32_t v = -1;
        return v;
    }

    std::string& streamk_tile_scheduling_mode_str()
    {
        static std::string v;
        return v;
    }

    std::string& hybrid_assignment_policy_str()
    {
        static std::string v;
        return v;
    }

    int32_t resolve_hybrid_assignment_policy(std::string const& canonical, std::string legacy)
    {
        int32_t policy = -1;
        if(canonical == "Default")
            policy = 0;
        else if(canonical == "DynamicWorkQueue")
            policy = 1;
        else if(canonical == "Auto")
            policy = 2;
        else if(!canonical.empty())
            throw std::invalid_argument("hybrid_assignment_policy must be Default, DynamicWorkQueue, or Auto");

        std::transform(legacy.begin(), legacy.end(), legacy.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        int32_t old = -1;
        if(legacy == "off" || legacy == "0")
            old = 0;
        else if(legacy == "on" || legacy == "1")
            old = 1;
        else if(legacy == "auto" || legacy == "2")
            old = 2;
        else if(!legacy.empty())
            throw std::invalid_argument("streamk_tile_scheduling must be off|0, on|1, or auto|2");
        if(policy >= 0 && old >= 0 && policy != old)
            throw std::invalid_argument("Conflicting hybrid_assignment_policy and streamk_tile_scheduling");
        return policy >= 0 ? policy : old;
    }

    int32_t& uniform_summation_order()
    {
        static int32_t v = -1;
        return v;
    }

    std::string& uniform_summation_order_str()
    {
        static std::string v;
        return v;
    }
}
