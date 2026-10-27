// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Spec-driven native mirror of instances/gfx942/f64_mma_probe.py. */
#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/instance_f64_mma_probe.h"
#include "rocke/ir_internal.h"
#include <stdio.h>

rocke_kernel_def_t* rocke_build_f64_mma_probe(rocke_ir_builder_t* b, int m, int k)
{
    char name[64];
    snprintf(name, sizeof(name), "f64_mma_probe_m%d_k%d", m, k);
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return NULL;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(m != 16 && m != 4)
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "f64 MMA probe requires m=16 or m=4");
        if(k <= 0 || k % 4)
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "f64 MMA probe requires k to be a positive multiple of 4");
        /* m -> (B slab rows, B/C/D columns) per K step of 4. */
        int slab_rows = m == 16 ? 4 : 16;
        int cols = m == 16 ? 16 : 4;
        const rocke_arch_target_t* target = rocke_arch_target_from_gfx("gfx942");
        const rocke_mma_op_t* atom = rocke_mma_catalog_op_for_shape(
            &target->mma, "mma", "fp64", "fp64", "fp64", m, m, 4, NULL);
        rocke_param_opts_t opts = {};
        opts.align = 8;
        opts.align_set = true;
        const rocke_type_t* ptr = rocke_ptr_type(b, rocke_f64(), "global");
        rocke_value_t* a = rocke_b_param(b, "A", ptr, &opts);
        rocke_value_t* bb = rocke_b_param(b, "B", ptr, &opts);
        rocke_value_t* cc = rocke_b_param(b, "C", ptr, &opts);
        rocke_value_t* d = rocke_b_param(b, "D", ptr, &opts);
        rocke_value_t* lane = rocke_b_thread_id_x(b);
        rocke_value_t* cn = rocke_b_const_i32(b, cols);
        rocke_value_t* ck = rocke_b_const_i32(b, k);
        rocke_value_t* indices[4];
        rocke_value_t* c_values[4];
        for(int slot = 0; slot < atom->c_frag_len; ++slot)
        {
            rocke_value_t *r, *col;
            rocke_layout_map_coord(atom->c_layout, b, lane, slot, &r, &col);
            rocke_value_t* offset = rocke_b_mul(b, r, cn);
            indices[slot] = rocke_b_add(b, offset, col);
            c_values[slot] = rocke_b_global_load(b, cc, indices[slot], rocke_f64(), 8);
        }
        rocke_value_t* acc = rocke_b_vec_pack(b, c_values, atom->c_frag_len, rocke_f64());
        rocke_value_t *a_row, *a_k, *b_row, *b_col;
        rocke_layout_map_coord(atom->a_layout, b, lane, 0, &a_row, &a_k);
        rocke_layout_map_coord(atom->b_layout, b, lane, 0, &b_row, &b_col);
        for(int step = 0; step < k / 4; ++step)
        {
            rocke_value_t* a_off = rocke_b_mul(b, a_row, ck);
            rocke_value_t* a_col = rocke_b_add(b, rocke_b_const_i32(b, step * 4), a_k);
            rocke_value_t* a_index = rocke_b_add(b, a_off, a_col);
            rocke_value_t* b_slab_row
                = rocke_b_add(b, rocke_b_const_i32(b, step * slab_rows), b_row);
            rocke_value_t* b_off = rocke_b_mul(b, b_slab_row, cn);
            rocke_value_t* b_index = rocke_b_add(b, b_off, b_col);
            rocke_value_t* av = rocke_b_global_load(b, a, a_index, rocke_f64(), 8);
            rocke_value_t* bv = rocke_b_global_load(b, bb, b_index, rocke_f64(), 8);
            acc = rocke_b_mma(b, atom->op_id, av, bv, acc, NULL, 0);
        }
        for(int slot = 0; slot < atom->c_frag_len; ++slot)
        {
            rocke_value_t* value = rocke_b_vec_extract(b, acc, slot);
            rocke_b_global_store(b, d, indices[slot], value, 8);
        }
        rocke_b_ret(b);
        return rocke_ir_builder_kernel(b);
    });
}
