// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/hardware.hpp"
#include <vector>

namespace origami
{
    namespace streamk
    {
        enum class reduction_type
        {
            // BasicReduction,
            Tree,
            Parallel,
            // AtomicReduction,
            Count,
            None = Count
        };

        inline reduction_type int_to_reduction_type(int rt)
        {
            return (reduction_type)rt;
        }

        size_t get_workspace(
            size_t x,
            size_t y,
            size_t mt_m,
            size_t mt_n,
            size_t bpe_c,
            size_t grid,
            size_t tiles,
            reduction_type reduction);

        reduction_type select_reduction(
            size_t x,
            size_t y,
            size_t z,
            size_t batch,
            size_t mt_m,
            size_t mt_n,
            size_t mt_k,
            const hardware_t& analytical_hardware,
            int dynamic_grid_version);

        const char* rtype_to_string(streamk::reduction_type r);

        size_t select_grid(size_t x,
                           size_t y,
                           size_t z,
                           size_t batch,
                           bool            trans_a,
                           bool            trans_b,
                           size_t          element_size_A,
                           size_t          element_size_B,
                           size_t          element_size_out,
                           data_type_t     mi_datatype,
                           size_t          workspace_size,
                           size_t          mt_m,
                           size_t          mt_n,
                           size_t          mt_k,
                           size_t          mi_m,
                           size_t          mi_n,
                           size_t          mi_k,
                           size_t          workgroup_mapping,
                           size_t          workspace_size_per_elem_c,
                           int             occupancy,
                           const hardware_t& analytical_hardware,
                           int dynamic_grid_version,
                           reduction_type reduction_strategy,
                           size_t max_cus = 0);
                           // max workspace

    } // namespace streamk
}
