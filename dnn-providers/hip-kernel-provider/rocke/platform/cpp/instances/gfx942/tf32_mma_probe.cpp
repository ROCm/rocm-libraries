// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Spec-driven native mirror of instances/gfx942/tf32_mma_probe.py. */
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.tiling.h"
#include "rocke/instance_tf32_mma_probe.h"
#include "rocke/ir_internal.h"
#include "tf32_mma_probe_internal.h"

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
        rocke_tiling_tile_desc_t c_desc, d_desc;
        const auto* target = rocke_arch_target_from_gfx("gfx942");
        if(rocke_h_make_mma_accumulator_desc(b, &c_desc, atom, target, false) != ROCKE_OK
           || rocke_h_make_mma_accumulator_desc(b, &d_desc, atom, target, true) != ROCKE_OK)
            return NULL;
        int input_lengths[] = {0, m, k}, input_strides[] = {m * k, k, 1};
        int output_lengths[] = {0, m, m}, output_strides[] = {m * m, m, 1};
        bool known[] = {false, true, true};
        const auto* input_dtype = strcmp(mode, "prepacked") == 0 ? rocke_tf32() : rocke_f32();
        rocke_tiling_tensor_desc_t input_tensor, diagnostic_tensor, output_tensor;
        if(rocke_h_make_tensor_desc(
               b, &input_tensor, input_lengths, input_strides, 3, input_dtype, known)
               != ROCKE_OK
           || rocke_h_make_tensor_desc(
                  b, &diagnostic_tensor, input_lengths, input_strides, 3, rocke_i32(), known)
                  != ROCKE_OK
           || rocke_h_make_tensor_desc(
                  b, &output_tensor, output_lengths, output_strides, 3, rocke_f32(), known)
                  != ROCKE_OK)
            return NULL;
        rocke_tiling_index_t base_origin[] = {{batch, 0}, {NULL, 0}, {NULL, 0}};
        rocke_tiling_window_t output_window;
        rocke_tiling_fragment_t initial;
        if(rocke_h_make_window(b, &output_window, &output_tensor, base_origin, 3, NULL, NULL)
               != ROCKE_OK
           || rocke_h_load_fragment(b, &initial, cc, &output_window, &c_desc, lane, 4, false)
                  != ROCKE_OK)
            return NULL;
        auto* acc = initial.value;
        if(strcmp(mode, "fp32") == 0)
        {
            rocke_matrix_fragment_layout_t layout = {{{32, 32}, 1, 32, 1}, 1, 64 / m, m};
            rocke_tiling_tile_desc_t scalar_desc;
            if(rocke_h_make_matrix_fragment_desc(b, &scalar_desc, &layout) != ROCKE_OK)
                return NULL;
            for(int step = 0; step < 2; ++step)
            {
                rocke_tiling_index_t origin[] = {{batch, 0}, {NULL, 0}, {NULL, step * (64 / m)}};
                rocke_tiling_window_t window, diagnostics;
                rocke_tiling_fragment_t af, bf, prepared;
                if(rocke_h_make_window(b, &window, &input_tensor, origin, 3, NULL, NULL) != ROCKE_OK
                   || rocke_h_make_window(
                          b, &diagnostics, &diagnostic_tensor, origin, 3, NULL, NULL)
                          != ROCKE_OK
                   || rocke_h_load_fragment(b, &af, a, &window, &scalar_desc, lane, 4, false)
                          != ROCKE_OK
                   || rocke_h_load_fragment(b, &bf, bb, &window, &scalar_desc, lane, 4, false)
                          != ROCKE_OK)
                    return NULL;
                auto* av = rocke_b_vec_extract(b, af.value, 0);
                auto* bv = rocke_b_vec_extract(b, bf.value, 0);
                auto* bits = rocke_b_vector_splat(b, rocke_b_bitcast(b, av, rocke_i32()), 1);
                if(rocke_h_make_fragment(b, &prepared, &scalar_desc, rocke_i32(), bits) != ROCKE_OK
                   || rocke_h_store_fragment(b, pa, &diagnostics, &prepared, lane, 4, false)
                          != ROCKE_OK)
                    return NULL;
                bits = rocke_b_vector_splat(b, rocke_b_bitcast(b, bv, rocke_i32()), 1);
                if(rocke_h_make_fragment(b, &prepared, &scalar_desc, rocke_i32(), bits) != ROCKE_OK
                   || rocke_h_store_fragment(b, pb, &diagnostics, &prepared, lane, 4, false)
                          != ROCKE_OK)
                    return NULL;
                acc = rocke_b_mma(b, full->op_id, av, bv, acc, NULL, 0);
            }
        }
        else
        {
            rocke_matrix_fragment_layout_t layout = {{{32, 32}, 2, 32, 2}, 2, 64 / m, m};
            rocke_tiling_tile_desc_t desc;
            rocke_tiling_window_t window, diagnostics;
            if(rocke_h_make_matrix_fragment_desc(b, &desc, &layout) != ROCKE_OK
               || rocke_h_make_window(b, &window, &input_tensor, base_origin, 3, NULL, NULL)
                      != ROCKE_OK
               || rocke_h_make_window(
                      b, &diagnostics, &diagnostic_tensor, base_origin, 3, NULL, NULL)
                      != ROCKE_OK)
                return NULL;
            rocke_value_t* inputs[] = {a, bb};
            rocke_value_t* prepared_ptr[] = {pa, pb};
            rocke_value_t* fragments[2];
            for(int role = 0; role < 2; ++role)
            {
                rocke_tiling_fragment_t loaded, prepared;
                if(rocke_h_load_fragment(b, &loaded, inputs[role], &window, &desc, lane, 4, false)
                   != ROCKE_OK)
                    return NULL;
                auto* values = loaded.value;
                if(strcmp(mode, "carrier") == 0)
                    values = rocke_b_bitcast(b, values, rocke_vector_type(b, rocke_i32(), 2));
                if(strcmp(mode, "rne") == 0)
                {
                    rocke_value_t* elements[2];
                    for(int i = 0; i < 2; ++i)
                        elements[i] = rocke_b_cvt_f32_to_tf32(b, rocke_b_vec_extract(b, values, i));
                    values = rocke_b_vec_pack(b, elements, 2, rocke_tf32());
                }
                else
                    values = rocke_b_bitcast(b, values, rocke_vector_type(b, rocke_tf32(), 2));
                auto* bits = rocke_b_bitcast(b, values, rocke_vector_type(b, rocke_i32(), 2));
                if(rocke_h_make_fragment(b, &prepared, &desc, rocke_i32(), bits) != ROCKE_OK
                   || rocke_h_store_fragment(
                          b, prepared_ptr[role], &diagnostics, &prepared, lane, 4, false)
                          != ROCKE_OK)
                    return NULL;
                fragments[role] = values;
            }
            acc = rocke_b_mma(b, atom->op_id, fragments[0], fragments[1], acc, NULL, 0);
        }
        rocke_tiling_fragment_t result;
        if(rocke_h_make_fragment(b, &result, &d_desc, rocke_f32(), acc) != ROCKE_OK
           || rocke_h_store_fragment(b, d, &output_window, &result, lane, 4, false) != ROCKE_OK)
            return NULL;
        rocke_b_ret(b);
        return rocke_ir_builder_kernel(b);
    });
}
