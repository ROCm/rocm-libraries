// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Spec-driven native mirror of instances/gfx942/tf32_mma_probe.py. */
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.mma_io.h"
#include "rocke/instance_tf32_mma_probe.h"
#include "rocke/ir_internal.h"
#include "tf32_mma_probe_internal.h"
#include <stdio.h>
#include <string.h>

rocke_kernel_def_t* rocke_build_tf32_mma_probe(rocke_ir_builder_t* b, int m, const char* mode)
{
    return ckc::build_tf32_mma_probe(b, m, mode, nullptr);
}

rocke_kernel_def_t* ckc::build_tf32_mma_probe(rocke_ir_builder_t* b,
                                              int m,
                                              const char* mode,
                                              const rocke_mma_catalog_t* catalog)
{
    char name[96];
    snprintf(name, sizeof(name), "tf32_probe_%d_%s", m, mode ? mode : "invalid");
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return NULL;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if((m != 16 && m != 32) || !mode
           || (strcmp(mode, "raw") && strcmp(mode, "carrier") && strcmp(mode, "rne")
               && strcmp(mode, "prepacked") && strcmp(mode, "fp32")))
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "TF32 probe requires m=16/32 and a known preparation");
        int k = 128 / m;
        if(!catalog)
            catalog = &rocke_arch_target_from_gfx("gfx942")->mma;
        const rocke_mma_op_t* atom
            = rocke_mma_catalog_op_for_shape(catalog, "mma", "tf32", "tf32", "fp32", m, m, k, NULL);
        const rocke_mma_op_t* full = nullptr;
        if(strcmp(mode, "fp32") == 0)
            full = rocke_mma_catalog_op_for_shape(
                catalog, "mma", "fp32", "fp32", "fp32", m, m, k / 2, nullptr);
        const int width = m == 16 ? 4 : 16;
        char expected_id[64];
        snprintf(expected_id, sizeof(expected_id), "mfma_f32_%dx%dx%d_xf32", m, m, k);
        const auto* registered
            = rocke_mma_catalog_by_op_id(&rocke_arch_target_from_gfx("gfx942")->mma, expected_id);
        auto valid = [&](const rocke_mma_op_t* selected,
                         const char* dtype,
                         int source_width,
                         int atom_k,
                         const char* suffix) {
            char id[64];
            snprintf(id, sizeof(id), "mfma_f32_%dx%dx%d_%s", m, m, atom_k, suffix);
            if(!selected || !selected->op_id || strcmp(selected->op_id, id) || !selected->family
               || strcmp(selected->family, "mma") || selected->m != m || selected->n != m
               || selected->k != atom_k || selected->wave_size != 64)
                return false;
            for(int role = 0; role < 2; ++role)
                if(!selected->srcs[role].dtype || strcmp(selected->srcs[role].dtype, dtype)
                   || selected->srcs[role].frag_len != source_width)
                    return false;
            return selected->srcs[2].dtype && selected->dst.dtype
                   && strcmp(selected->srcs[2].dtype, "fp32") == 0
                   && strcmp(selected->dst.dtype, "fp32") == 0
                   && selected->srcs[2].frag_len == width && selected->dst.frag_len == width;
        };
        auto valid_map = [&](const rocke_layout_map_t* layout, const rocke_layout_map_t* expected) {
            return layout && expected && layout->fn && layout->fn == expected->fn
                   && layout->frag_len == width && layout->wave_size == 64;
        };
        if(!valid(atom, "tf32", 2, k, "xf32")
           || (strcmp(mode, "fp32") == 0 && !valid(full, "fp32", 1, k / 2, "f32")) || !registered
           || !valid_map(atom->srcs[2].layout, registered->srcs[2].layout)
           || !valid_map(atom->dst.layout, registered->dst.layout))
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b,
                ROCKE_ERR_VALUE,
                "TF32 probe requires the fixed gfx942 instruction and layout contract");
        const rocke_type_t* ty = strcmp(mode, "prepacked") == 0 ? rocke_i32() : rocke_f32();
        rocke_param_opts_t opts = {};
        opts.align = 4;
        opts.align_set = true;
        const rocke_type_t* input_ptr = rocke_ptr_type(b, ty, "global");
        rocke_value_t* a = rocke_b_param(b, "A", input_ptr, &opts);
        rocke_value_t* bb = rocke_b_param(b, "B", input_ptr, &opts);
        const rocke_type_t* float_ptr = rocke_ptr_type(b, rocke_f32(), "global");
        rocke_value_t* cc = rocke_b_param(b, "C", float_ptr, &opts);
        rocke_value_t* d = rocke_b_param(b, "D", float_ptr, &opts);
        const rocke_type_t* bits_ptr = rocke_ptr_type(b, rocke_i32(), "global");
        rocke_value_t* pa = rocke_b_param(b, "PA", bits_ptr, &opts);
        rocke_value_t* pb = rocke_b_param(b, "PB", bits_ptr, &opts);
        rocke_value_t* lane = rocke_b_thread_id_x(b);
        rocke_value_t* batch = rocke_b_block_id_x(b);
        rocke_value_t* cm = rocke_b_const_i32(b, m);
        rocke_value_t* axis = rocke_b_mod(b, lane, cm);
        rocke_value_t* group = rocke_b_div(b, lane, cm);
        rocke_value_t* ck = rocke_b_const_i32(b, k);
        rocke_value_t* row = rocke_b_mul(b, axis, ck);
        rocke_value_t* size = rocke_b_const_i32(b, m * k);
        rocke_value_t* batch_input = rocke_b_mul(b, batch, size);
        rocke_value_t* row_base = rocke_b_add(b, batch_input, row);
        size = rocke_b_const_i32(b, m * m);
        rocke_value_t* batch_output = rocke_b_mul(b, batch, size);
        rocke_value_t* indices[16];
        rocke_value_t* c_values[16];
        for(int slot = 0; slot < atom->srcs[2].frag_len; ++slot)
        {
            rocke_value_t *r, *col;
            rocke_layout_map_coord(atom->srcs[2].layout, b, lane, slot, &r, &col);
            rocke_value_t* offset = rocke_b_mul(b, r, cm);
            offset = rocke_b_add(b, offset, col);
            indices[slot] = rocke_b_add(b, batch_output, offset);
            c_values[slot] = rocke_b_global_load(b, cc, indices[slot], rocke_f32(), 4);
        }
        rocke_value_t* acc = rocke_b_vec_pack(b, c_values, atom->srcs[2].frag_len, rocke_f32());
        if(strcmp(mode, "fp32") == 0)
        {
            for(int step = 0; step < 2; ++step)
            {
                rocke_value_t* delta = rocke_b_const_i32(b, step * (64 / m));
                delta = rocke_b_add(b, group, delta);
                rocke_value_t* index = rocke_b_add(b, row_base, delta);
                rocke_value_t* av = rocke_b_global_load(b, a, index, rocke_f32(), 4);
                rocke_value_t* bv = rocke_b_global_load(b, bb, index, rocke_f32(), 4);
                rocke_value_t* bits = rocke_b_bitcast(b, av, rocke_i32());
                rocke_b_global_store(b, pa, index, bits, 4);
                bits = rocke_b_bitcast(b, bv, rocke_i32());
                rocke_b_global_store(b, pb, index, bits, 4);
                acc = rocke_b_mma(b, full->op_id, av, bv, acc, NULL, 0);
            }
        }
        else
        {
            rocke_matrix_fragment_layout_t layout = {{{32, 32}, 2, 32, 2}, 2, 64 / m, m};
            rocke_value_t* inputs[] = {a, bb};
            rocke_value_t* prepared[] = {pa, pb};
            rocke_value_t* fragments[2];
            for(int role = 0; role < 2; ++role)
            {
                bool packed = strcmp(mode, "prepacked") == 0;
                const rocke_type_t* carrier
                    = packed || strcmp(mode, "carrier") == 0 ? rocke_i32() : rocke_f32();
                rocke_value_t* values = rocke_h_load_matrix_fragment(b,
                                                                     inputs[role],
                                                                     row_base,
                                                                     group,
                                                                     0,
                                                                     packed ? "tf32" : "fp32",
                                                                     &layout,
                                                                     carrier,
                                                                     4);
                if(strcmp(mode, "rne") == 0)
                {
                    rocke_value_t* elems[2];
                    for(int i = 0; i < 2; ++i)
                    {
                        rocke_value_t* e = rocke_b_vec_extract(b, values, i);
                        elems[i] = rocke_b_cvt_f32_to_tf32(b, e);
                    }
                    values = rocke_b_vec_pack(b, elems, 2, rocke_tf32());
                }
                else
                    values = rocke_b_bitcast(b, values, rocke_vector_type(b, rocke_tf32(), 2));
                for(int slot = 0; slot < 2; ++slot)
                {
                    rocke_value_t* two = rocke_b_const_i32(b, 2);
                    rocke_value_t* offset = rocke_b_mul(b, group, two);
                    rocke_value_t* index = rocke_b_const_i32(b, slot);
                    offset = rocke_b_add(b, offset, index);
                    index = rocke_b_add(b, row_base, offset);
                    rocke_value_t* e = rocke_b_vec_extract(b, values, slot);
                    rocke_value_t* bits = rocke_b_bitcast(b, e, rocke_i32());
                    rocke_b_global_store(b, prepared[role], index, bits, 4);
                }
                fragments[role] = values;
            }
            acc = rocke_b_mma(b, atom->op_id, fragments[0], fragments[1], acc, NULL, 0);
        }
        for(int slot = 0; slot < atom->dst.frag_len; ++slot)
        {
            rocke_value_t* value = rocke_b_vec_extract(b, acc, slot);
            rocke_b_global_store(b, d, indices[slot], value, 4);
        }
        rocke_b_ret(b);
        return rocke_ir_builder_kernel(b);
    });
}
