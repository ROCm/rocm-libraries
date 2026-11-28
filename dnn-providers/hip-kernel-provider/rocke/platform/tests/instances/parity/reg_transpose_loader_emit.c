// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of reg_transpose_loader_emit.py (config table must
 * match). */
#include "rocke/helper_rocke.helpers.loads.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[] = {"gfx1201", "gfx950"};
static const int NUM_ARCHES = sizeof(ARCHES) / sizeof(ARCHES[0]);

typedef enum
{
    PATH_LOAD = 0,
    PATH_SPLIT,
    PATH_GUARD,
    PATH_SPLIT_UNMASKED,
    PATH_GUARD_UNMASKED,
} path_t;

typedef struct
{
    const char* name;
    int tile_rows;
    int tile_cols;
    int block_size;
    int wave_size;
    int vec;
    const char* dtype;
    path_t path;
    bool valid;
    bool run;
    bool swizzle;
} case_t;

static const case_t CASES[] = {
    {"full8", 64, 64, 64, 32, 8, "f16", PATH_LOAD, false, false, false},
    {"full4_bf16", 64, 32, 64, 32, 4, "bf16", PATH_LOAD, false, false, false},
    {"multi2", 64, 64, 64, 32, 2, "f16", PATH_LOAD, true, false, false},
    {"scalar1", 16, 32, 64, 32, 1, "f16", PATH_LOAD, true, false, false},
    {"split8", 64, 64, 64, 32, 8, "f16", PATH_SPLIT, true, false, false},
    {"partial8", 64, 32, 128, 32, 8, "f16", PATH_LOAD, true, false, false},
    {"partial8_split", 64, 32, 128, 32, 8, "bf16", PATH_SPLIT, true, false, false},
    {"partial8_guard", 64, 32, 128, 32, 8, "f16", PATH_GUARD, true, false, false},
    {"run4", 64, 64, 64, 32, 4, "f16", PATH_LOAD, true, true, false},
    {"run8_guard", 64, 32, 128, 32, 8, "f16", PATH_GUARD, true, true, false},
    {"swz8", 64, 64, 64, 32, 8, "f16", PATH_SPLIT, false, false, true},
    {"swz1", 16, 32, 64, 32, 1, "f16", PATH_LOAD, false, false, true},
    {"wave64_2", 64, 64, 256, 64, 2, "f16", PATH_LOAD, false, false, false},
    /* rejected: width 3 */
    {"vec3", 48, 32, 64, 32, 3, "f16", PATH_LOAD, false, false, false},
    /* rejected: 32-bit elements */
    {"f32", 64, 64, 64, 32, 8, "f32", PATH_LOAD, false, false, false},
    /* rejected: guard on a full-participation tile */
    {"guard_full", 64, 64, 64, 32, 8, "f16", PATH_GUARD, true, false, false},
    /* rejected: guard at width 1 */
    {"guard_vec1", 16, 32, 128, 32, 1, "f16", PATH_GUARD, true, false, false},
    /* rejected: store_lds of an unmasked partial tile */
    {"partial8_unmasked", 64, 32, 128, 32, 8, "f16", PATH_SPLIT_UNMASKED, true, false, false},
    /* rejected: guard with unmasked reads */
    {"guard_unmasked", 64, 32, 128, 32, 8, "f16", PATH_GUARD_UNMASKED, true, false, false},
    /* unmasked reads of a full-participation tile (masked is moot there) */
    {"full8_unmasked", 64, 64, 64, 32, 8, "f16", PATH_SPLIT_UNMASKED, true, false, false},
};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

typedef struct
{
    const case_t* c;
    rocke_value_t* c_ld;
    rocke_value_t* c_lim;
    int kc;
} env_t;

static const rocke_type_t* dtype_of(const char* name)
{
    if(strcmp(name, "bf16") == 0)
        return rocke_bf16();
    if(strcmp(name, "f32") == 0)
        return rocke_f32();
    return rocke_f16();
}

/* Python descriptor: (row + col * ld, cmp_lt(col, lim) or None). */
static rocke_value_t* descriptor(rocke_ir_builder_t* b,
                                 rocke_value_t* row,
                                 rocke_value_t* col,
                                 rocke_value_t** out_valid,
                                 void* user)
{
    env_t* env = (env_t*)user;
    rocke_value_t* m = rocke_b_mul(b, col, env->c_ld);
    rocke_value_t* off = rocke_b_add(b, row, m);
    *out_valid = env->c->valid ? rocke_b_cmp_lt(b, col, env->c_lim) : NULL;
    return off;
}

/* Python run_descriptor: decode the first column once, step the rest. */
static void run_descriptor(rocke_ir_builder_t* b,
                           rocke_value_t* row,
                           rocke_value_t* col,
                           rocke_value_t** out_off,
                           rocke_value_t** out_valid,
                           void* user)
{
    env_t* env = (env_t*)user;
    rocke_value_t* m = rocke_b_mul(b, col, env->c_ld);
    rocke_value_t* base = rocke_b_add(b, row, m);
    for(int j = 0; j < ROCKE_REGT_K_RUN; ++j)
    {
        if(j)
        {
            rocke_value_t* step = rocke_b_const_i32(b, j * (env->c->tile_rows + 8));
            out_off[j] = rocke_b_add(b, base, step);
        }
        else
        {
            out_off[j] = base;
        }
        if(j % 2)
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j);
            rocke_value_t* cj = rocke_b_add(b, col, c_j);
            out_valid[j] = rocke_b_cmp_lt(b, cj, env->c_lim);
        }
        else
        {
            out_valid[j] = NULL;
        }
    }
}

/* Python col_swizzle: col ^ ((row & (kc - 1)) * 8). */
static rocke_value_t*
    col_swizzle(rocke_ir_builder_t* b, rocke_value_t* row, rocke_value_t* col, void* user)
{
    env_t* env = (env_t*)user;
    rocke_value_t* c_mask = rocke_b_const_i32(b, env->kc - 1);
    rocke_value_t* f = rocke_b_land(b, row, c_mask);
    rocke_value_t* c8 = rocke_b_const_i32(b, 8);
    rocke_value_t* sh = rocke_b_mul(b, f, c8);
    return rocke_b_xor(b, col, sh);
}

/* 0 on success, 1 where Python raises (the loader's __post_init__). */
static int build(rocke_ir_builder_t* b, const case_t* c)
{
    const rocke_type_t* dt = dtype_of(c->dtype);
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, dt, "global"), NULL);
    rocke_value_t* n_bytes = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, x, n_bytes);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    int shape[2] = {c->tile_rows, c->tile_cols};
    rocke_value_t* tile = rocke_b_smem_alloc(b, dt, shape, 2, "tile");
    env_t env;
    env.c = c;
    env.c_ld = rocke_b_const_i32(b, c->tile_rows + 8);
    env.c_lim = rocke_b_const_i32(b, c->tile_cols - 3);
    env.kc = c->tile_cols / ROCKE_REGT_K_RUN;

    rocke_reg_transpose_tile_loader_t ldr;
    char why[256];
    if(rocke_reg_transpose_tile_loader_init(&ldr,
                                            c->tile_rows,
                                            c->tile_cols,
                                            c->block_size,
                                            c->wave_size,
                                            c->vec,
                                            dt,
                                            why,
                                            sizeof why)
       != ROCKE_OK)
    {
        fprintf(stderr, "%s\n", why);
        return 1;
    }
    if(c->swizzle)
    {
        ldr.col_swizzle = col_swizzle;
        ldr.col_swizzle_user = &env;
    }
    rocke_loads_run_descriptor_fn run_fn = c->run ? run_descriptor : NULL;
    if(c->path == PATH_LOAD)
    {
        rocke_reg_transpose_tile_loader_load(
            b, &ldr, tid, tile, descriptor, &env, rsrc, run_fn, &env);
    }
    else
    {
        rocke_regt_staged_t* staged = (rocke_regt_staged_t*)malloc(sizeof(rocke_regt_staged_t));
        if(staged == NULL)
            return 1;
        const bool masked = c->path != PATH_SPLIT_UNMASKED && c->path != PATH_GUARD_UNMASKED;
        const bool guard = c->path == PATH_GUARD || c->path == PATH_GUARD_UNMASKED;
        rocke_reg_transpose_tile_loader_load_global(
            b, &ldr, tid, descriptor, &env, rsrc, masked, run_fn, &env, guard, staged);
        rocke_reg_transpose_tile_loader_store_lds(b, &ldr, tile, staged);
        free(staged);
    }
    rocke_b_ret(b);
    return 0;
}

int main(int argc, char** argv)
{
    int num_configs = NUM_CASES * NUM_ARCHES;
    if(argc < 2)
    {
        fprintf(
            stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], num_configs - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= num_configs)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const case_t* c = &CASES[idx / NUM_ARCHES];
    const char* arch = ARCHES[idx % NUM_ARCHES];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "reg_transpose_loader") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = block_size */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", c->block_size);
    try
    {
        if(build(&b, c) != 0)
        {
            rocke_ir_builder_free(&b);
            return 1;
        }
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "builder error: %s\n", e.what());
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return 0;
}
