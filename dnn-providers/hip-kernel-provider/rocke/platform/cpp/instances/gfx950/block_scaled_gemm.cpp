// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.mma_io.h"
#include "rocke/instance_gfx950_block_scaled_gemm.h"
#include "rocke/mfma_scale_internal.h"
#include <initializer_list>
#include <stdio.h>
#include <string.h>

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
        auto* cN = rocke_b_const_i32(b, s->N);
        auto* tid = rocke_b_thread_id_x(b);
        auto* c64 = rocke_b_const_i32(b, 64);
        auto* lane = rocke_b_mod(b, tid, c64);
        auto* row = rocke_b_mod(b, lane, c32);
        auto* half = rocke_b_div(b, lane, c32);
        auto* m0 = rocke_b_mul(b, rocke_b_block_id_y(b), c32);
        auto* n0 = rocke_b_mul(b, rocke_b_block_id_x(b), c32);
        auto* ar = rocke_b_add(b, m0, row);
        auto* br = rocke_b_add(b, n0, row);
        auto* abase = rocke_b_mul(
            b, ar, rocke_b_const_i32(b, (int64_t)s->K * rocke_dtype_info(da)->encoded_bits / 8));
        auto* bbase = rocke_b_mul(
            b, br, rocke_b_const_i32(b, (int64_t)s->K * rocke_dtype_info(db)->encoded_bits / 8));
        auto* acc = rocke_b_zero_vec_f32(b, 16);
        auto al = rocke_mfma_matrix_layout(da), bl = rocke_mfma_matrix_layout(db);
        for(int step = 0; step < s->K / 64; ++step)
        {
            auto* a = rocke_h_load_matrix_fragment(
                b, params[0], abase, half, step * 64, da, &al, rocke_i32(), 16);
            auto* bb = rocke_h_load_matrix_fragment(
                b, params[1], bbase, half, step * 64, db, &bl, rocke_i32(), 16);
            auto* group = rocke_b_add(b, rocke_b_const_i32(b, step * 2), half);
            auto* ai = rocke_b_add(b, rocke_b_mul(b, ar, rocke_b_const_i32(b, s->K / 32)), group);
            auto* bi = rocke_b_add(b, rocke_b_mul(b, group, cN), br);
            auto* sa = rocke_b_zext(
                b, rocke_b_global_load(b, params[2], ai, rocke_i8(), 1), rocke_i32());
            auto* sb = rocke_b_zext(
                b, rocke_b_global_load(b, params[3], bi, rocke_i8(), 1), rocke_i32());
            rocke_value_t* scales[] = {sa, sb};
            acc = rocke_b_mma(b, id, a, bb, acc, scales, 2);
        }
        for(int slot = 0; slot < 16; ++slot)
        {
            rocke_value_t *r, *c;
            rocke_layout_map_coord(atom->c_layout, b, lane, slot, &r, &c);
            auto* rr = rocke_b_add(b, m0, r);
            auto* offset = rocke_b_mul(b, rr, cN);
            auto* cc = rocke_b_add(b, n0, c);
            auto* index = rocke_b_add(b, offset, cc);
            rocke_b_global_store(b,
                                 params[4],
                                 index,
                                 rocke_b_cast_f32_to(b,
                                                     rocke_b_vec_extract(b, acc, slot),
                                                     rocke_storage_ir_type(dc)),
                                 2);
        }
        return b->status == ROCKE_OK ? b->kernel : NULL;
    });
}
