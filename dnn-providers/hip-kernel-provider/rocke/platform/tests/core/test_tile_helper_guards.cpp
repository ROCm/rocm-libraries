// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocke/helper_rocke.helpers.loads.h"

#include <cstdio>
#include <cstring>

static void
    count_chunk(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*, rocke_value_t*, void* user)
{
    ++*static_cast<int*>(user);
}

static bool check_geometry(int load_vec, int block_size, bool valid)
{
    rocke_coalesced_tile_loader_t loader{};
    loader.tile_rows = 16;
    loader.tile_cols = 16;
    loader.load_vec = load_vec;
    loader.block_size = block_size;
    int per_thread = -1;
    const auto status = rocke_coalesced_tile_loader_vecs_per_thread(&loader, &per_thread);
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "loader_geometry") != ROCKE_OK)
        return false;
    auto* tid = rocke_b_const_i32(&b, 0);
    const int before = rocke_b_current_region(&b)->num_ops;
    int visited = 0;
    rocke_coalesced_tile_loader_visit_chunks(&b, &loader, tid, nullptr, count_chunk, &visited);
    const bool passed
        = valid ? status == ROCKE_OK && per_thread == 2 && visited == per_thread
                      && rocke_ir_builder_ok(&b)
                : status == ROCKE_ERR_VALUE && per_thread == -1 && visited == 0
                      && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                      && std::strstr(rocke_ir_builder_error(&b), "invalid chunk geometry")
                      && rocke_b_current_region(&b)->num_ops == before;
    if(!passed)
        std::fprintf(stderr,
                     "unexpected chunk geometry result for load_vec=%d block_size=%d\n",
                     load_vec,
                     block_size);
    rocke_ir_builder_free(&b);
    return passed;
}

int main()
{
    return check_geometry(0, 64, false) && check_geometry(2, 0, false)
                   && check_geometry(-1, 64, false) && check_geometry(2, -1, false)
                   && check_geometry(2, 64, true)
               ? 0
               : 1;
}
