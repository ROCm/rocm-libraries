// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief Routing of one step's rows to experts: written by the router phase, read by the MoE
/// phases of the same launch (after a megakernel dependency on the router).
///
/// The router rewrites every field each step, including the counts of experts no row picked, so
/// the table needs no reset between launches. Entries are "routes": route = row * top_k + slot,
/// so the MoE phase finds both the row (route / top_k) and its weight (topk_weights[route]).
///
/// Layout, all in global memory:
///   topk_ids     [M * top_k]            int32, expert of each route (slot order unspecified)
///   topk_weights [M * top_k]            fp32, normalised routing weight of each route
///   expert_count [num_experts]          int32, routes of each expert
///   expert_routes[num_experts * max_m]  int32, the routes of expert e at e * max_m, ascending
///   active       [num_experts]          int32, experts with expert_count > 0, ascending
///   num_active   [1]                    int32, entries of active
/// An expert receives at most one route per row, so max_m >= M bounds expert_count.
struct moe_route_table
{
    int32_t* topk_ids;
    float* topk_weights;
    int32_t* expert_count;
    int32_t* expert_routes;
    int32_t* active;
    int32_t* num_active;
    index_t M;
    index_t max_m;
    index_t top_k;
    index_t num_experts;

    CK_TILE_HOST_DEVICE static constexpr index_t row_of(index_t route, index_t top_k)
    {
        return route / top_k;
    }
};

} // namespace ck_tile
