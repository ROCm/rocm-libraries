// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of coalesced_loader_col_xor_emit.py (config table must
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
    PATH_SLOTS,
} path_t;

typedef struct
{
    const char* name;
    int tile_rows;
    int tile_cols;
    int block_size;
    int load_vec; /* 0: from_tile */
    bool axis_row;
    int chunk; /* 0: no col_xor */
    path_t path;
    bool has_col_xor; /* explicit col_xor tuple (Python: not None) */
    int col_xor[4];
} case_t;

static const case_t CASES[] = {
    {"col64", 64, 64, 256, 0, false, 8, PATH_LOAD},
    {"col16", 32, 16, 64, 0, false, 8, PATH_LOAD},
    {"row32", 64, 32, 256, 8, true, 8, PATH_LOAD},
    {"scalar16", 16, 16, 256, 0, false, 8, PATH_LOAD},
    {"split64", 64, 64, 256, 0, false, 8, PATH_SPLIT},
    {"slots64", 64, 64, 256, 0, false, 8, PATH_SLOTS},
    /* rejected: a 4-element chunk under an 8-wide vector */
    {"narrow_chunk", 64, 64, 256, 0, false, 4, PATH_LOAD},
    /* control: no col_xor */
    {"plain64", 64, 64, 256, 0, false, 0, PATH_LOAD},
    /* rejected: i32 shifts by 32 (row_shift, then flip_shift) */
    {"shift32", 64, 64, 256, 8, false, 0, PATH_LOAD, true, {32, 7, 3, 3}},
    {"flip32", 64, 64, 256, 8, false, 0, PATH_LOAD, true, {0, 7, 3, 32}},
    /* rejected: 6 chunks per row under the 32-column swizzle's groups of 4 */
    {"partial48", 16, 48, 64, 8, false, 0, PATH_LOAD, true, {1, 3, 3, 3}},
    /* rejected: one 64-column chunk per row; the flip bit needs two */
    {"one_chunk64", 64, 64, 256, 8, false, 0, PATH_LOAD, true, {0, 0, 6, 3}},
    /* rejected: a ragged 36-column row (4 whole chunks of 8 plus 4 columns) */
    {"ragged36", 64, 36, 64, 4, false, 0, PATH_LOAD, true, {1, 3, 3, 3}},
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

/* Python _loader(case): 0 on success, non-zero where Python raises. */
static int make_loader(const case_t* c, rocke_coalesced_tile_loader_t* ldr)
{
    if(c->load_vec)
    {
        memset(ldr, 0, sizeof(*ldr));
        ldr->tile_rows = c->tile_rows;
        ldr->tile_cols = c->tile_cols;
        ldr->block_size = c->block_size;
        ldr->load_vec = c->load_vec;
        ldr->use_buffer_rsrc = true;
        ldr->oob_sentinel = 2147483647;
        ldr->vector_axis_row = c->axis_row;
        ldr->has_inner_dim = false;
        ldr->inner_dim = 0;
        ldr->has_col_xor = false;
    }
    else if(rocke_coalesced_tile_loader_from_tile(
                c->tile_rows, c->tile_cols, c->block_size, 8, true, ldr)
            != ROCKE_OK)
    {
        fprintf(stderr, "from_tile failed\n");
        return 1;
    }
    if(c->chunk)
    {
        int col_xor[4];
        if(rocke_lds_col_xor_params(c->tile_cols, c->chunk, col_xor) != ROCKE_OK)
        {
            fprintf(stderr, "lds_col_xor_params refused\n");
            return 1;
        }
        if(rocke_coalesced_tile_loader_set_col_xor(ldr, col_xor) != ROCKE_OK)
        {
            fprintf(stderr, "col_xor refused\n");
            return 1;
        }
    }
    if(c->has_col_xor && rocke_coalesced_tile_loader_set_col_xor(ldr, c->col_xor) != ROCKE_OK)
    {
        fprintf(stderr, "col_xor refused\n");
        return 1;
    }
    return 0;
}

static void build(rocke_ir_builder_t* b, const case_t* c, const rocke_coalesced_tile_loader_t* ldr)
{
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, rocke_f16(), "global"), NULL);
    rocke_value_t* n_bytes = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, x, n_bytes);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    int shape[2] = {c->tile_rows, c->tile_cols};
    rocke_value_t* tile = rocke_b_smem_alloc(b, rocke_f16(), shape, 2, "tile");
    desc_env_t env;
    env.c_cols = rocke_b_const_i32(b, c->tile_cols);

    if(c->path == PATH_LOAD)
    {
        rocke_coalesced_tile_loader_load(b, ldr, tid, tile, descriptor, &env, rsrc, NULL);
    }
    else if(c->path == PATH_SPLIT)
    {
        rocke_ctl_staged_t staged;
        rocke_coalesced_tile_loader_load_global(b, ldr, tid, descriptor, &env, rsrc, NULL, &staged);
        rocke_coalesced_tile_loader_store_lds(b, ldr, tile, &staged);
    }
    else
    {
        rocke_ctl_staged_t slots;
        rocke_value_t* offsets[ROCKE_CTL_MAX_VECS_PER_THREAD];
        rocke_ctl_staged_t staged;
        rocke_coalesced_tile_loader_plan_slots(b, ldr, tid, &slots);
        rocke_coalesced_tile_loader_slot_offsets(b, ldr, &slots, descriptor, &env, offsets);
        rocke_coalesced_tile_loader_issue_slots(b, ldr, rsrc, &slots, offsets, &staged);
        rocke_coalesced_tile_loader_store_lds(b, ldr, tile, &staged);
    }
    rocke_b_ret(b);
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

    rocke_coalesced_tile_loader_t ldr;
    if(make_loader(c, &ldr) != 0)
    {
        return 1;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "coalesced_loader_col_xor") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = block_size */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", c->block_size);
    try
    {
        build(&b, c, &ldr);
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
