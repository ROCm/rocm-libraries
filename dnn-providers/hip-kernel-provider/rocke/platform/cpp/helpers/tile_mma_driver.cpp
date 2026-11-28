// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocke/helper_rocke.helpers.tiling.mma.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "rocke/ir_internal.h"

static bool rocke_tile_mma_valid_grid(rocke_tile_mma_grid_t grid, size_t count)
{
    return grid.m > 0 && grid.n > 0 && grid.k > 0
           && static_cast<size_t>(grid.m) * static_cast<size_t>(grid.n) == count;
}

void rocke_tile_mma_accumulate_atoms(rocke_ir_builder_t* b,
                                     rocke_tile_mma_grid_t grid,
                                     const char* order,
                                     rocke_value_t* const* accs,
                                     size_t count,
                                     rocke_tile_mma_atom_fn emit,
                                     void* user,
                                     rocke_value_t** out)
{
    if(b == nullptr || b->status != ROCKE_OK)
        return;
    if(!rocke_tile_mma_valid_grid(grid, count) || order == nullptr || std::strlen(order) != 3
       || accs == nullptr || emit == nullptr || out == nullptr)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "tile MMA needs a positive grid and one accumulator per output atom");
        return;
    }

    constexpr char axis_names[] = "MNK";
    int axis[3];
    bool used[3] = {};
    for(int i = 0; i < 3; ++i)
    {
        const char* match = std::strchr(axis_names, order[i]);
        if(match == nullptr || used[match - axis_names])
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "tile MMA order must permute MNK");
            return;
        }
        axis[i] = static_cast<int>(match - axis_names);
        used[axis[i]] = true;
    }
    if(accs != out)
        std::copy_n(accs, count, out);

    const int lengths[] = {grid.m, grid.n, grid.k};
    int coord[3];
    for(coord[axis[0]] = 0; coord[axis[0]] < lengths[axis[0]]; ++coord[axis[0]])
        for(coord[axis[1]] = 0; coord[axis[1]] < lengths[axis[1]]; ++coord[axis[1]])
            for(coord[axis[2]] = 0; coord[axis[2]] < lengths[axis[2]]; ++coord[axis[2]])
            {
                const size_t idx = static_cast<size_t>(coord[0]) * grid.n + coord[1];
                out[idx] = emit(b, coord[0], coord[1], coord[2], out[idx], user);
                if(b->status != ROCKE_OK)
                    return;
            }
}

struct rocke_tile_mma_stage
{
    const rocke_tile_mma_readers_t* readers;
    std::vector<rocke_value_t*> a_rows;
    std::vector<rocke_value_t*> b_cols;
};

static rocke_value_t* rocke_tile_mma_issue_staged(
    rocke_ir_builder_t* b, int mi, int ni, int ki, rocke_value_t* acc, void* user)
{
    auto& stage = *static_cast<rocke_tile_mma_stage*>(user);
    const auto& op = *stage.readers;
    if(mi == 0 && ni == 0)
    {
        for(size_t row = 0; row < stage.a_rows.size(); ++row)
            stage.a_rows[row] = op.read_a(b, ki, static_cast<int>(row), op.user);
        for(size_t col = 0; col < stage.b_cols.size(); ++col)
            stage.b_cols[col] = op.read_b(b, ki, static_cast<int>(col), op.user);
        if(op.after_reads != nullptr)
            op.after_reads(b, ki, op.user);
    }
    if(ni == 0 && op.before_row != nullptr)
        op.before_row(b, ki, mi, op.user);
    rocke_value_t* result = op.mma(b, stage.a_rows[mi], stage.b_cols[ni], acc, op.user);
    if(static_cast<size_t>(ni + 1) == stage.b_cols.size() && op.after_row != nullptr)
        op.after_row(b, ki, mi, op.user);
    return result;
}

void rocke_tile_mma_accumulate_staged(rocke_ir_builder_t* b,
                                      rocke_tile_mma_grid_t grid,
                                      rocke_value_t* const* accs,
                                      size_t count,
                                      const rocke_tile_mma_readers_t* readers,
                                      rocke_value_t** out)
{
    if(b == nullptr || b->status != ROCKE_OK)
        return;
    if(!rocke_tile_mma_valid_grid(grid, count))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "staged MMA needs a positive grid and one accumulator per output atom");
        return;
    }
    if(readers == nullptr || readers->read_a == nullptr || readers->read_b == nullptr
       || readers->mma == nullptr)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "staged MMA needs both operand readers and an MMA callback");
        return;
    }
    rocke_tile_mma_stage stage{
        readers, std::vector<rocke_value_t*>(grid.m), std::vector<rocke_value_t*>(grid.n)};
    rocke_tile_mma_accumulate_atoms(
        b, grid, "KMN", accs, count, rocke_tile_mma_issue_staged, &stage, out);
}
