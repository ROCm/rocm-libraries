// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of coalesced_loader_async_emit.py (config table must
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

static const char* const ARCH = "gfx1250";
enum
{
    TILE_ROWS = 16,
    TILE_COLS = 32,
    BLOCK_SIZE = 64,
};

typedef struct
{
    const char* name;
    const rocke_type_t* (*elem_dtype)(void);
    int load_vec;
    const rocke_type_t* (*tile_dtype)(void);
    const rocke_type_t* (*src_dtype)(void);
} case_t;

static const case_t CASES[] = {
    {"f16_b128", rocke_f16, 8, rocke_f16, rocke_f16},
    {"bf16_b128", rocke_bf16, 8, rocke_bf16, rocke_bf16},
    {"f32_b128", rocke_f32, 4, rocke_f32, rocke_f32},
    {"i32_b128", rocke_i32, 4, rocke_i32, rocke_i32},
    {"f16_b32", rocke_f16, 2, rocke_f16, rocke_f16},
    {"f32_b64", rocke_f32, 2, rocke_f32, rocke_f32},
    /* an f16 loader filling a bf16 tile: the widths match */
    {"f16_into_bf16", rocke_f16, 8, rocke_bf16, rocke_bf16},
    /* rejected: no loader width for i8 */
    {"i8", rocke_i8, 8, rocke_i8, rocke_i8},
    /* rejected: 8 x 4 B is a 32-byte chunk */
    {"f32_b256", rocke_f32, 8, rocke_f32, rocke_f32},
    /* rejected: an f16 loader filling an f32 tile */
    {"f16_into_f32_tile", rocke_f16, 8, rocke_f32, rocke_f16},
    /* rejected: an f32 loader reading an f16 source */
    {"f32_from_f16_ptr", rocke_f32, 4, rocke_f32, rocke_f16},
};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

typedef struct
{
    rocke_value_t* c_cols;
} desc_env_t;

/* Python: descriptor(b, row, col) -> (b.add(b.mul(row, c_cols), col), None) */
static rocke_value_t* descriptor(rocke_ir_builder_t* b,
                                 rocke_value_t* row,
                                 rocke_value_t* col,
                                 rocke_value_t** out_valid,
                                 void* user)
{
    desc_env_t* env = (desc_env_t*)user;
    *out_valid = NULL;
    rocke_value_t* m = rocke_b_mul(b, row, env->c_cols);
    return rocke_b_add(b, m, col);
}

/* Python CoalescedTileLoader(tile_rows=..., load_vec=..., elem_dtype=...). */
static rocke_coalesced_tile_loader_t make_loader(const case_t* c)
{
    rocke_coalesced_tile_loader_t ldr;
    memset(&ldr, 0, sizeof(ldr));
    ldr.tile_rows = TILE_ROWS;
    ldr.tile_cols = TILE_COLS;
    ldr.block_size = BLOCK_SIZE;
    ldr.load_vec = c->load_vec;
    ldr.use_buffer_rsrc = true;
    ldr.oob_sentinel = 2147483647;
    ldr.elem_dtype = c->elem_dtype();
    return ldr;
}

static void build(rocke_ir_builder_t* b, const case_t* c)
{
    rocke_coalesced_tile_loader_t ldr = make_loader(c);
    rocke_param_opts_t src_opts;
    memset(&src_opts, 0, sizeof(src_opts));
    src_opts.readonly = true;
    src_opts.readonly_set = true;
    src_opts.align = 16;
    src_opts.align_set = true;
    rocke_value_t* src
        = rocke_b_param(b, "src", rocke_ptr_type(b, c->src_dtype(), "global"), &src_opts);
    rocke_value_t* num_bytes = rocke_b_param(b, "num_bytes", rocke_i32(), NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    int shape[2] = {TILE_ROWS, TILE_COLS};
    rocke_value_t* tile = rocke_b_smem_alloc(b, c->tile_dtype(), shape, 2, "tile");
    desc_env_t env;
    env.c_cols = rocke_b_const_i32(b, TILE_COLS);
    rocke_coalesced_tile_loader_load_async_to_lds(
        b, &ldr, tid, tile, descriptor, &env, src, num_bytes, 0);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_CASES - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_CASES)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const case_t* c = &CASES[idx];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "coalesced_loader_async") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = BLOCK_SIZE */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", BLOCK_SIZE);
    try
    {
        build(&b, c);
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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, ARCH, &llvm_text, err, sizeof err);
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
