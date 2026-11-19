// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.loads.h"

#include <cstdio>
#include <cstdlib>
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

static bool check_layout()
{
    const rocke_lds_column_xor_t bad_terms[]
        = {{-1, 3, 4, -1}, {0, 32, 4, -1}, {0, 3, 32, -1}, {0, 3, 4, -2}};
    for(const auto& layout : bad_terms)
        if(rocke_lds_column_xor_shift_for(&layout, 2) != -1)
            return false;
    const auto* layout = &rocke_lds_row_xor_8x16b;
    return rocke_lds_column_xor_shift_for(layout, 0) == -1
           && rocke_lds_column_xor_shift_for(layout, 3) == -1
           && rocke_lds_column_xor_shift_for(layout, 32) == -1
           && rocke_lds_column_xor_shift_for(layout, 2) == 3;
}

static rocke_value_t*
    unexpected_read(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*, int, const void*)
{
    std::abort();
}

static bool check_empty_fragment()
{
    rocke_ir_builder_t b{};
    if(rocke_ir_builder_init(&b, "empty_fragment") != ROCKE_OK)
        return false;
    auto* zero = rocke_b_const_i32(&b, 0);
    const int before = rocke_b_current_region(&b)->num_ops;
    const auto* result = rocke_b_load_clipped_global_fragment(
        &b, zero, zero, zero, rocke_f16(), 0, unexpected_read, nullptr);
    const bool passed = result == nullptr && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                        && std::strstr(rocke_ir_builder_error(&b), "fragment needs positive width")
                        && rocke_b_current_region(&b)->num_ops == before;
    rocke_ir_builder_free(&b);
    return passed;
}

int main()
{
    return check_empty_fragment() && check_layout() && check_geometry(0, 64, false)
                   && check_geometry(2, 0, false) && check_geometry(-1, 64, false)
                   && check_geometry(2, -1, false) && check_geometry(2, 64, true)
               ? 0
               : 1;
}
