// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.loads.h"

#include <cstdio>
#include <cstring>

static bool check_source(const rocke_type_t* pointee, const char* space, const char* error)
{
    rocke_ir_builder_t b{};
    if(rocke_ir_builder_init(&b, "tdm_validate_source") != ROCKE_OK)
        return false;
    const auto* ptr_type = rocke_ptr_type(&b, pointee, space);
    auto* zero = rocke_b_const_i32(&b, 0);
    auto* z64 = rocke_b_const_i64(&b, 0);
    auto* extent = rocke_b_const_i32(&b, 16);
    rocke_tdm_source_window_2d_t source{};
    source.ptr = rocke_b_param(&b, "ptr", ptr_type, nullptr);
    source.base_offset = z64;
    source.row_origin = rocke_b_const_i32(&b, 1);
    source.col_origin = rocke_b_const_i32(&b, 4);
    source.row_extent = extent;
    source.cols_remaining = rocke_b_const_i32(&b, 12);
    source.row_stride = extent;
    const int before = rocke_b_current_region(&b)->num_ops;
    const rocke_tdm_tile_loader_2d_t loader{16, 16, 2, -1, 0};
    rocke_tdm_tile_load_t out{};
    bool passed = false;
    try
    {
        const bool result = rocke_tdm_tile_loader_2d_prepare(&b, &loader, &source, z64, zero, &out);
        passed = error ? !result && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                             && std::strstr(rocke_ir_builder_error(&b), error)
                       : result && rocke_ir_builder_ok(&b) && out.groups[0] != nullptr;
    }
    catch(const ckc::ValueError& e)
    {
        passed = error && std::strstr(e.what(), error);
    }
    if(error)
        passed = passed && rocke_b_current_region(&b)->num_ops == before;
    if(!passed)
        std::fprintf(stderr, "unexpected TDM prepare result: %s\n", rocke_ir_builder_error(&b));
    rocke_ir_builder_free(&b);
    return passed;
}

int main()
{
    return check_source(rocke_f16(), "global", nullptr)
                   && check_source(rocke_i8(), "global", "pointee width")
                   && check_source(rocke_f16(), "constant", "global pointer")
               ? 0
               : 1;
}
