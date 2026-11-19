// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.loads.h"
#include "rocke/helper_rocke.helpers.tiling.mma.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

template <typename Fn>
static bool refuses(rocke_ir_builder_t& b, const char* diagnostic, Fn&& call)
{
    try
    {
        call();
        return rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
               && std::strcmp(rocke_ir_builder_error(&b), diagnostic) == 0;
    }
    catch(const ckc::ValueError& error)
    {
        return std::strcmp(error.what(), diagnostic) == 0;
    }
}

static void
    count_chunk(rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*, rocke_value_t*, void* user)
{
    ++*static_cast<int*>(user);
}

static bool check_geometry(int rows,
                           int cols,
                           int load_vec,
                           int block_size,
                           bool row_axis,
                           int expected,
                           const char* diagnostic = nullptr,
                           bool prebound = false)
{
    rocke_coalesced_tile_loader_t loader{};
    loader.tile_rows = rows;
    loader.tile_cols = cols;
    loader.load_vec = load_vec;
    loader.block_size = block_size;
    loader.vector_axis_row = row_axis;
    int per_thread = -1;
    const auto status = rocke_coalesced_tile_loader_vecs_per_thread(&loader, &per_thread);
    rocke_ir_builder_t b{};
    if(rocke_ir_builder_init(&b, "loader_geometry") != ROCKE_OK)
        return false;
    auto* tid = rocke_b_const_i32(&b, 0);
    int visited = 0;
    const rocke_coalesced_chunk_grid_t bogus_grid{
        nullptr, nullptr, nullptr, [](rocke_ir_builder_t*, rocke_value_t*, void*) {
            std::abort();
        }};
    const int before = rocke_b_current_region(&b)->num_ops;
    const auto visit = [&] {
        rocke_coalesced_tile_loader_visit_chunks(
            &b, &loader, tid, prebound ? &bogus_grid : nullptr, count_chunk, &visited);
    };
    bool passed;
    if(expected >= 0)
    {
        visit();
        passed = status == ROCKE_OK && per_thread == expected && visited == per_thread
                 && rocke_ir_builder_ok(&b);
    }
    else
        passed = status == ROCKE_ERR_VALUE && per_thread == -1 && refuses(b, diagnostic, visit)
                 && visited == 0 && rocke_b_current_region(&b)->num_ops == before;
    if(!passed)
        std::fprintf(stderr,
                     "unexpected %dx%d chunk geometry for load_vec=%d block_size=%d row_axis=%d\n",
                     rows,
                     cols,
                     load_vec,
                     block_size,
                     row_axis);
    rocke_ir_builder_free(&b);
    return passed;
}

static const char* const divisor_error
    = "CoalescedTileLoader: load_vec and block_size must be positive";
static const char* const extent_error = "CoalescedTileLoader: tile dimensions must be positive";
static const char* const vector_error = "CoalescedTileLoader: load_vec must divide the vector axis";
static const char* const overflow_error = "CoalescedTileLoader: chunk geometry exceeds i32";

static bool check_loader_geometry()
{
    return check_geometry(16, 16, 0, 64, false, -1, divisor_error)
           && check_geometry(16, 16, 2, 0, false, -1, divisor_error)
           && check_geometry(16, 16, -1, 64, false, -1, divisor_error)
           && check_geometry(16, 16, 2, -1, false, -1, divisor_error)
           && check_geometry(0, 0, 0, 0, false, -1, divisor_error)
           && check_geometry(4, 6, 4, 3, false, -1, vector_error)
           && check_geometry(4, 6, 4, 3, false, -1, vector_error, true)
           && check_geometry(6, 4, 4, 3, true, -1, vector_error)
           && check_geometry(4, 2, 4, 2, false, -1, vector_error)
           && check_geometry(0, 2, 4, 2, false, -1, extent_error)
           && check_geometry(8, -8, 2, 4, true, -1, extent_error)
           && check_geometry(
               8, 8, 2, 3, false, -1, "tile 8x8 / 2 = 32 not divisible by block_size 3")
           && check_geometry(65536, 65536, 1, 3, false, -1, overflow_error)
           && check_geometry(5, 8, 4, 5, false, 2) && check_geometry(8, 5, 4, 5, true, 2);
}

static bool check_other_loader_builders()
{
    rocke_coalesced_tile_loader_t loader{};
    loader.tile_rows = 4;
    loader.tile_cols = 6;
    loader.load_vec = 4;
    loader.block_size = 3;
    const auto descriptor = [](rocke_ir_builder_t*,
                               rocke_value_t*,
                               rocke_value_t*,
                               rocke_value_t**,
                               void*) -> rocke_value_t* { std::abort(); };
    const auto check = [&](auto emit) {
        rocke_ir_builder_t b{};
        if(rocke_ir_builder_init(&b, "loader_geometry") != ROCKE_OK)
            return false;
        auto* zero = rocke_b_const_i32(&b, 0);
        const int before = rocke_b_current_region(&b)->num_ops;
        const bool passed = refuses(b, vector_error, [&] { emit(&b, zero); })
                            && rocke_b_current_region(&b)->num_ops == before;
        rocke_ir_builder_free(&b);
        return passed;
    };
    return check([&](auto* b, auto* zero) {
               rocke_coalesced_tile_loader_load(
                   b, &loader, zero, nullptr, descriptor, nullptr, nullptr, zero);
           })
           && check([&](auto* b, auto* zero) {
                  rocke_ctl_staged_t staged{};
                  rocke_coalesced_tile_loader_load_global(
                      b, &loader, zero, descriptor, nullptr, nullptr, zero, &staged);
              })
           && check([&](auto* b, auto* zero) {
                  rocke_ctl_staged_t slots{};
                  rocke_coalesced_tile_loader_plan_slots(b, &loader, zero, &slots);
              });
}

static bool check_layout_refusal(const rocke_lds_column_xor_t* layout,
                                 int unit_bytes,
                                 const char* diagnostic)
{
    if(rocke_lds_column_xor_shift_for(layout, unit_bytes) != -1)
        return false;
    rocke_ir_builder_t b{};
    if(rocke_ir_builder_init(&b, "xor_layout") != ROCKE_OK)
        return false;
    auto* zero = rocke_b_const_i32(&b, 0);
    const int before = rocke_b_current_region(&b)->num_ops;
    const bool passed
        = refuses(
              b, diagnostic, [&] { rocke_lds_column_xor_emit(&b, zero, zero, layout, unit_bytes); })
          && rocke_b_current_region(&b)->num_ops == before;
    rocke_ir_builder_free(&b);
    return passed;
}

static bool check_layout()
{
    const rocke_lds_column_xor_t bad_terms[]
        = {{-1, 3, 4, -1}, {0, 32, 4, -1}, {0, 3, 32, -1}, {0, 3, 4, -2}};
    for(const auto& layout : bad_terms)
        if(!check_layout_refusal(&layout, 0, "LDS column XOR terms must be integers in [0, 31]"))
            return false;
    const auto* layout = &rocke_lds_row_xor_8x16b;
    const char* unit_error = "LDS column units must be a positive power of two bytes";
    if(!check_layout_refusal(layout, 0, unit_error) || !check_layout_refusal(layout, 3, unit_error)
       || !check_layout_refusal(layout, 32, "LDS column unit is larger than a swizzle chunk")
       || !check_layout_refusal(nullptr, 2, "LDS column XOR layout must not be null")
       || rocke_lds_column_xor_shift_for(layout, 2) != 3
       || rocke_lds_column_xor_shift_for(&rocke_lds_no_swizzle, 0) != 0)
        return false;
    rocke_ir_builder_t b{};
    if(rocke_ir_builder_init(&b, "xor_identity") != ROCKE_OK)
        return false;
    auto* col = rocke_b_const_i32(&b, 42);
    const int before = rocke_b_current_region(&b)->num_ops;
    const bool passed = rocke_lds_column_xor_emit(&b, nullptr, col, &rocke_lds_no_swizzle, 0) == col
                        && rocke_b_current_region(&b)->num_ops == before;
    rocke_ir_builder_free(&b);
    return passed;
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
    rocke_value_t* result = nullptr;
    const bool passed
        = refuses(b,
                  "fragment needs positive width",
                  [&] {
                      result = rocke_b_load_clipped_global_fragment(
                          &b, zero, zero, zero, rocke_f16(), 0, unexpected_read, nullptr);
                  })
          && result == nullptr && rocke_b_current_region(&b)->num_ops == before;
    rocke_ir_builder_free(&b);
    return passed;
}

static bool check_staged_grid(rocke_tile_mma_grid_t grid, size_t count, const char* error)
{
    rocke_ir_builder_t b{};
    if(rocke_ir_builder_init(&b, "staged_grid") != ROCKE_OK)
        return false;
    auto* zero = rocke_b_const_i32(&b, 0);
    rocke_value_t* acc[] = {zero};
    rocke_value_t* out[] = {nullptr};
    int reads = 0;
    const auto read = [](rocke_ir_builder_t* b, int, int, void* user) {
        ++*static_cast<int*>(user);
        return rocke_b_const_i32(b, 1);
    };
    const auto issue
        = [](rocke_ir_builder_t*, rocke_value_t*, rocke_value_t*, rocke_value_t* acc, void*) {
              return acc;
          };
    const rocke_tile_mma_readers_t readers{read, read, issue, nullptr, nullptr, nullptr, &reads};
    const int before = rocke_b_current_region(&b)->num_ops;
    const auto accumulate
        = [&] { rocke_tile_mma_accumulate_staged(&b, grid, acc, count, &readers, out); };
    bool passed;
    if(error == nullptr)
    {
        accumulate();
        passed = rocke_ir_builder_ok(&b) && reads == 2 && out[0] == zero;
    }
    else
        passed = refuses(b, error, accumulate) && reads == 0 && out[0] == nullptr
                 && rocke_b_current_region(&b)->num_ops == before;
    rocke_ir_builder_free(&b);
    return passed;
}

int main()
{
    const char* grid_error = "staged MMA needs a positive grid and one accumulator per output atom";
    return check_staged_grid({1, 1, 0}, 1, grid_error)
                   && check_staged_grid({2, 1, 1}, 1, grid_error)
                   && check_staged_grid({1, 1, 1}, 1, nullptr) && check_empty_fragment()
                   && check_layout() && check_loader_geometry() && check_other_loader_builders()
               ? 0
               : 1;
}
