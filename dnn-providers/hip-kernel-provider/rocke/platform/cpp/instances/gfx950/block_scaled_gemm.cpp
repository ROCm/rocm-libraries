// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <string.h>

#include <initializer_list>

#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.mma_operand.h"
#include "rocke/helper_rocke.helpers.tiling.h"
#include "rocke/instance_gfx950_block_scaled_gemm.h"
#include "rocke/ir_storage.h"
#include "rocke/mfma_scale_internal.h"

rocke_gfx950_block_scaled_gemm_spec_t rocke_gfx950_block_scaled_gemm_spec_default(void)
{
    return {"block_scaled_gemm", 0, 0, 0, "fp8e4m3", "fp8e4m3", "bf16", 32};
}

bool rocke_gfx950_block_scaled_gemm_is_valid_spec(const rocke_gfx950_block_scaled_gemm_spec_t* s,
                                                  const char* arch)
{
    if(!s || !arch || strcmp(arch, "gfx950") != 0 || !s->name || !s->dtype_a || !s->dtype_b
       || !s->dtype_c)
        return false;
    char a_scratch[64], b_scratch[64], c_scratch[64];
    const char* a = rocke_normalize_dtype(s->dtype_a, a_scratch, sizeof(a_scratch));
    const char* bb = rocke_normalize_dtype(s->dtype_b, b_scratch, sizeof(b_scratch));
    const char* c = rocke_normalize_dtype(s->dtype_c, c_scratch, sizeof(c_scratch));
    return a && bb && c && rocke_mfma_matrix_format(a) >= 0 && rocke_mfma_matrix_format(bb) >= 0
           && (strcmp(c, "bf16") == 0 || strcmp(c, "fp16") == 0) && s->block_k == 32 && s->M > 0
           && s->N > 0 && s->K > 0 && s->M % 32 == 0 && s->N % 32 == 0 && s->K % 64 == 0
           && (int64_t)s->M * s->K <= INT32_MAX && (int64_t)s->N * s->K <= INT32_MAX
           && (int64_t)s->M * s->N <= INT32_MAX;
}

rocke_kernel_def_t* rocke_build_gfx950_block_scaled_gemm_new(
    rocke_ir_builder_t* b, const rocke_gfx950_block_scaled_gemm_spec_t* s, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(!b || !s)
            return NULL;
        if(!rocke_gfx950_block_scaled_gemm_is_valid_spec(s, arch))
        {
            rocke_ir_builder_init(b, "invalid_scaled_gemm");
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid gfx950 block-scaled GEMM spec");
        }
        char a_scratch[64], b_scratch[64], c_scratch[64];
        const char* da = rocke_normalize_dtype(s->dtype_a, a_scratch, sizeof(a_scratch));
        const char* db = rocke_normalize_dtype(s->dtype_b, b_scratch, sizeof(b_scratch));
        const char* dc = rocke_normalize_dtype(s->dtype_c, c_scratch, sizeof(c_scratch));
        if(rocke_ir_builder_init(b, "") != ROCKE_OK)
            return NULL;
        char* name = rocke_arena_printf(
            &b->arena, "%s_mfma_scaled_%s_%s_%s_M%dN%dK%d", s->name, da, db, dc, s->M, s->N, s->K);
        if(!name)
            ckc::raise_status(ROCKE_ERR_OOM, "cannot allocate scaled GEMM kernel name");
        b->kernel->name = name;
        char id[160];
        const char* tokens[] = {"fp8", "bf8", "fp6", "bf6", "fp4"};
        snprintf(id,
                 sizeof(id),
                 "mfma_gfx950_f32_32x32x64_%s_%s_scale_e8m0_e8m0_k32",
                 tokens[rocke_mfma_matrix_format(da)],
                 tokens[rocke_mfma_matrix_format(db)]);
        const auto* atom = rocke_gfx950_scaled_mfma(id);
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 64);
        rocke_value_t* params[5];
        const char* names[] = {"A", "B", "A_scale", "B_scale", "C"};
        const char* dtypes[] = {da, db, "e8m0", "e8m0", dc};
        for(int i = 0; i < 5; ++i)
        {
            rocke_param_opts_t opts = {};
            opts.noalias = true;
            opts.noalias_set = true;
            if(i == 4)
            {
                opts.writeonly = true;
                opts.writeonly_set = true;
            }
            else
            {
                opts.readonly = true;
                opts.readonly_set = true;
            }
            opts.align = (i == 2 || i == 3) ? 1 : 16;
            opts.align_set = true;
            params[i] = rocke_b_param(
                b, names[i], rocke_ptr_type(b, rocke_storage_ir_type(dtypes[i]), "global"), &opts);
        }
        for(const char* n : {"M", "N", "K"})
            rocke_b_param(b, n, rocke_i32(), NULL);
        auto* c32 = rocke_b_const_i32(b, 32);
        auto* tid = rocke_b_thread_id_x(b);
        auto* c64 = rocke_b_const_i32(b, 64);
        auto* lane = rocke_b_mod(b, tid, c64);
        auto* m0 = rocke_b_mul(b, rocke_b_block_id_y(b), c32);
        auto* n0 = rocke_b_mul(b, rocke_b_block_id_x(b), c32);
        auto al = rocke_mfma_matrix_layout(da), bl = rocke_mfma_matrix_layout(db);
        rocke_tiling_tile_desc_t a_desc, b_desc, result_desc, scale_desc;
        if(rocke_h_make_matrix_fragment_desc(b, &a_desc, &al) != ROCKE_OK
           || rocke_h_make_matrix_fragment_desc(b, &b_desc, &bl) != ROCKE_OK
           || rocke_h_make_mma_accumulator_desc(
                  b, &result_desc, atom, rocke_arch_target_from_gfx(arch), true)
                  != ROCKE_OK)
            return NULL;
        rocke_tiling_tensor_desc_t a_tensor, b_tensor, a_scales, b_scales, c_tensor;
        const int groups = s->K / 32;
        int a_lengths[] = {s->M, s->K}, b_lengths[] = {s->N, s->K}, matrix_strides[] = {s->K, 1};
        int as_lengths[] = {s->M, groups}, bs_lengths[] = {s->N, groups};
        int as_strides[] = {groups, 1}, bs_strides[] = {1, s->N};
        int c_lengths[] = {s->M, s->N}, c_strides[] = {s->N, 1};
        if(rocke_h_make_tensor_desc(
               b, &a_tensor, a_lengths, matrix_strides, 2, rocke_dtype_to_ir_type(da))
               != ROCKE_OK
           || rocke_h_make_tensor_desc(
                  b, &b_tensor, b_lengths, matrix_strides, 2, rocke_dtype_to_ir_type(db))
                  != ROCKE_OK
           || rocke_h_make_tensor_desc(
                  b, &a_scales, as_lengths, as_strides, 2, rocke_dtype_to_ir_type("e8m0"))
                  != ROCKE_OK
           || rocke_h_make_tensor_desc(
                  b, &b_scales, bs_lengths, bs_strides, 2, rocke_dtype_to_ir_type("e8m0"))
                  != ROCKE_OK
           || rocke_h_make_mma_scale_desc(b, &scale_desc, atom, rocke_arch_target_from_gfx(arch))
                  != ROCKE_OK
           || rocke_h_make_tensor_desc(
                  b, &c_tensor, c_lengths, c_strides, 2, rocke_dtype_to_ir_type(dc))
                  != ROCKE_OK)
            return NULL;
        auto* acc = rocke_b_zero_vec_f32(b, 16);
        for(int step = 0; step < s->K / 64; ++step)
        {
            rocke_tiling_index_t a_origin[] = {{m0, 0}, {NULL, step * 64}};
            rocke_tiling_index_t b_origin[] = {{n0, 0}, {NULL, step * 64}};
            rocke_tiling_index_t as_origin[] = {{m0, 0}, {NULL, step * 2}};
            rocke_tiling_index_t bs_origin[] = {{n0, 0}, {NULL, step * 2}};
            rocke_tiling_window_t aw, bw, asw, bsw;
            rocke_tiling_fragment_t af, bf, saf, sbf;
            if(rocke_h_make_window(b, &aw, &a_tensor, a_origin, 2, NULL, NULL) != ROCKE_OK
               || rocke_h_make_window(b, &bw, &b_tensor, b_origin, 2, NULL, NULL) != ROCKE_OK
               || rocke_h_load_fragment(b, &af, params[0], &aw, &a_desc, lane, 16, false)
                      != ROCKE_OK
               || rocke_h_load_fragment(b, &bf, params[1], &bw, &b_desc, lane, 16, false)
                      != ROCKE_OK)
                return NULL;
            auto* a = rocke_h_prepare_scaled_mma_operand(b, atom, "a", &af);
            auto* bb = rocke_h_prepare_scaled_mma_operand(b, atom, "b", &bf);
            if(!a || !bb
               || rocke_h_make_window(b, &asw, &a_scales, as_origin, 2, NULL, NULL) != ROCKE_OK
               || rocke_h_make_window(b, &bsw, &b_scales, bs_origin, 2, NULL, NULL) != ROCKE_OK
               || rocke_h_load_fragment(b, &saf, params[2], &asw, &scale_desc, lane, 1, false)
                      != ROCKE_OK
               || rocke_h_load_fragment(b, &sbf, params[3], &bsw, &scale_desc, lane, 1, false)
                      != ROCKE_OK)
                return NULL;
            auto* sa = rocke_b_zext(b, rocke_b_vec_extract(b, saf.value, 0), rocke_i32());
            auto* sb = rocke_b_zext(b, rocke_b_vec_extract(b, sbf.value, 0), rocke_i32());
            rocke_value_t* scales[] = {sa, sb};
            acc = rocke_b_mma(b, id, a, bb, acc, scales, 2);
        }
        rocke_tiling_index_t c_origin[] = {{m0, 0}, {n0, 0}};
        rocke_tiling_window_t cw;
        rocke_tiling_fragment_t result;
        if(rocke_h_make_window(b, &cw, &c_tensor, c_origin, 2, NULL, NULL) != ROCKE_OK
           || rocke_h_make_fragment(b, &result, &result_desc, rocke_f32(), acc) != ROCKE_OK
           || rocke_h_store_fragment(b, params[4], &cw, &result, lane, 16, false) != ROCKE_OK)
            return NULL;
        return b->status == ROCKE_OK ? b->kernel : NULL;
    });
}
