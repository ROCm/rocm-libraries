// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native counterpart of global_read_emit.py; same authored copy and cases. */
#include "rocke/helper_rocke.helpers.global_read.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[] = {"gfx950", "gfx1151", "gfx1201"};
typedef struct
{
    const char* mode;
    const char* dtype;
    int n;
} case_t;
static const case_t CASES[] = {
    {"auto", "f16", 2},
    {"auto", "f16", 4},
    {"auto", "f16", 8},
    {"auto", "bf16", 8},
    {"auto", "f32", 2},
    {"auto", "f32", 4},
    {"auto", "i32", 4},
    {"auto", "f16", 0},
    {"auto", "bf16", 0},
    {"auto", "f32", 0},
    {"auto", "i32", 0},
    {"global64", "f16", 8},
    {"global64", "f32", 4},
    {"negative", "f16", 4},
    {"divergent", "f32", 4},
    {"wide", "f16", 8},
    {"expr", "f16", 8},
    {"auto", "f64", 2},
};
typedef struct
{
    rocke_value_t* y;
    rocke_value_t* tid;
    int n;
} body_t;

static void
    copy_body(rocke_ir_builder_t* b, const rocke_global_read_plan_t* plans, int count, void* user)
{
    body_t* ctx = (body_t*)user;
    (void)count;
    rocke_value_t* row = rocke_b_const_i32(b, 1);
    rocke_value_t* width = rocke_b_const_i32(b, ctx->n ? ctx->n : 1);
    rocke_value_t* col = rocke_b_mul(b, ctx->tid, width);
    rocke_value_t* indices[] = {row, col};
    rocke_global_read_cursor_t cursor;
    rocke_global_read_bind(b, &cursor, &plans[0], indices, 2);
    rocke_value_t* delta_row = rocke_b_const_i32(b, -1);
    rocke_value_t* delta_col = rocke_b_const_i32(b, 0);
    rocke_value_t* deltas[] = {delta_row, delta_col};
    if(ctx->n)
    {
        rocke_value_t* value = rocke_global_read_load_vec(b, &cursor, deltas, 2, ctx->n);
        rocke_b_global_store_vN(b, ctx->y, col, value, ctx->n, 0);
    }
    else
    {
        rocke_value_t* value = rocke_global_read_load_scalar(b, &cursor, deltas, 2);
        rocke_b_global_store(b, ctx->y, col, value, 1);
    }
    rocke_b_sync(b);
}

static void build_global_read(rocke_ir_builder_t* b, const case_t* c)
{
    const rocke_type_t* dtype = rocke_f16();
    if(strcmp(c->dtype, "bf16") == 0)
        dtype = rocke_bf16();
    if(strcmp(c->dtype, "f32") == 0)
        dtype = rocke_f32();
    if(strcmp(c->dtype, "i32") == 0)
        dtype = rocke_i32();
    if(strcmp(c->dtype, "f64") == 0)
        dtype = rocke_f64();
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, dtype, "global"), NULL);
    const rocke_type_t* meta = strcmp(c->mode, "wide") == 0 ? rocke_i64() : rocke_i32();
    rocke_value_t* rows = rocke_b_param(b, "rows", meta, NULL);
    rocke_value_t* stride = rocke_b_param(b, "stride", meta, NULL);
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    if(strcmp(c->mode, "divergent") == 0)
        rows = rocke_b_add(b, rows, tid);
    if(strcmp(c->mode, "expr") == 0)
    {
        rocke_value_t* bid = rocke_b_block_id_x(b);
        rows = rocke_b_add(b, rows, bid);
    }
    rocke_stride_t strides[]
        = {strcmp(c->mode, "negative") == 0 ? rocke_stride_imm(-512) : rocke_stride_value(stride),
           rocke_stride_imm(1)};
    const int shape[] = {1, 1};
    rocke_tensor_view_t view;
    rocke_make_global_view(&view, x, shape, 2, dtype, strides);
    rocke_stride_t extents[] = {rocke_stride_value(rows), rocke_stride_imm(512)};
    rocke_global_read_spec_t spec = {&view, extents, 2};
    body_t ctx = {y, tid, c->n};
    const char* addressing = strcmp(c->mode, "global64") == 0 ? "global64" : "auto";
    rocke_emit_global_read_paths(b, &spec, 1, copy_body, &ctx, addressing);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    const int count = (int)(sizeof(CASES) / sizeof(CASES[0])) * 3;
    if(argc < 2)
        return 2;
    int idx = atoi(argv[1]);
    const char* mode = argc > 2 ? argv[2] : "ll";
    if(idx < 0 || idx >= count)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "global_read") != ROCKE_OK)
        return 1;
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    try
    {
        build_global_read(&b, &CASES[idx / 3]);
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "%s\n", e.what());
        rocke_ir_builder_free(&b);
        return 1;
    }
    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "%s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    char* text = NULL;
    rocke_status_t status = ROCKE_OK;
    if(strcmp(mode, "ll") == 0)
    {
        char err[ROCKE_ERR_MSG_CAP] = {};
        status = rocke_lower_kernel_to_llvm_ex(
            b.kernel, ROCKE_LLVM_FLAVOR_AUTO, ARCHES[idx % 3], &text, err, sizeof(err));
        if(status != ROCKE_OK)
            fprintf(stderr, "%s\n", err);
    }
    else if(strcmp(mode, "ir") == 0)
        status = rocke_ir_serialize(b.kernel, &text);
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* diagnostics = NULL;
        size_t n = 0;
        rocke_verify(b.kernel, &diagnostics, &n);
        for(size_t i = 0; i < n; ++i)
        {
            char* message = rocke_diag_to_string(&diagnostics[i]);
            if(message)
            {
                puts(message);
                free(message);
            }
        }
        rocke_diags_free(diagnostics, n);
    }
    else
        status = ROCKE_ERR_VALUE;
    if(text)
    {
        fputs(text, stdout);
        free(text);
    }
    rocke_ir_builder_free(&b);
    return status == ROCKE_OK ? 0 : 1;
}
