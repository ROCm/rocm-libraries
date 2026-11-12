// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_layouts.cpp -- conv-owned LDS layout helpers (C++ twin of the library's
 * kernels/common/_conv_layouts.py). See instance_conv_layouts.h.
 */
#include "rocke/instance_conv_layouts.h"

#include <stddef.h>

/* ------------------------------ ConvKOuterFragmentReader.wave32_lane_offsets */

void rocke_conv_tr_lane_offsets32(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  int n,
                                  rocke_value_t* mn_base,
                                  rocke_value_t* k_base,
                                  rocke_value_t** col_out,
                                  rocke_value_t** row0_out)
{
    /* Python binds ONE const_i32(16) and ONE const_i32(8) and reuses each;
     * a duplicate here would consume an extra SSA id and drift every later
     * number. Keep the op order identical to the Python. */
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* c8 = rocke_b_const_i32(b, 8);
    rocke_value_t* lane_mod16 = rocke_b_mod(b, lane, c16);
    rocke_value_t* col_grp = rocke_b_div(b, lane_mod16, c8);
    rocke_value_t* col = rocke_b_mul(b, col_grp, c8);
    if(mn_base != NULL)
        col = rocke_b_add(b, mn_base, col);
    rocke_value_t* lane_div16 = rocke_b_div(b, lane, c16);
    rocke_value_t* c_n = rocke_b_const_i32(b, n);
    rocke_value_t* row_mul = rocke_b_mul(b, lane_div16, c_n);
    rocke_value_t* lane_mod8 = rocke_b_mod(b, lane, c8);
    rocke_value_t* row0 = rocke_b_add(b, row_mul, lane_mod8);
    if(k_base != NULL)
        row0 = rocke_b_add(b, k_base, row0);
    *col_out = col;
    *row0_out = row0;
}
