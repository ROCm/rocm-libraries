// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of tensor_view_buffer_emit.py (config table must match):
 * the buffer address space of the shared tensor view (rocke_tv_make_buffer_resource,
 * rocke_tv_make_buffer_view and the ROCKE_ADDR_BUFFER branches of the
 * rocke_tensor_view_* / rocke_tile_window_* entry points). See that file for
 * the case-by-case rationale.
 *
 * Every builder call below is its own statement, and the statement order
 * matches the Python emitter exactly. C argument evaluation order is
 * unspecified, and since each call consumes a value id, a different order
 * emits the same IR under different SSA names. */
#include "rocke/helper_rocke.helpers.sweep.h" /* rocke_tile_window_*_f32 peers */
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[] = {"gfx950", "gfx1151", "gfx1201"};
static const int NUM_ARCHES = sizeof(ARCHES) / sizeof(ARCHES[0]);

typedef struct
{
    const char* op;
    const char* dtype;
    int n;
} case_t;

static const case_t CASES[] = {
    /* accepted (f16) */
    {"scalar", "f16", 1},
    {"vec", "f16", 2},
    {"vec", "f16", 4},
    {"vec", "f16", 8},
    {"vec_at", "f16", 2},
    {"vec_at", "f16", 4},
    {"vec_at", "f16", 8},
    {"vec_at_mask", "f16", 2},
    {"vec_at_mask", "f16", 4},
    {"vec_at_mask", "f16", 8},
    {"window", "f16", 2},
    {"window", "f16", 4},
    {"window", "f16", 8},
    {"window_f32", "f16", 1},
    {"window_f32", "f16", 2},
    {"window_f32", "f16", 4},
    {"window_f32", "f16", 8},
    /* rejected: dtype not wired on the buffer path */
    {"scalar", "bf16", 1},
    {"vec", "bf16", 4},
    {"vec_at", "bf16", 4},
    {"vec_at_mask", "bf16", 4},
    {"scalar", "f32", 1},
    {"vec", "f32", 4},
    {"scalar", "i32", 1},
    {"scalar", "i64", 1},
    {"scalar", "f64", 1},
    {"vec_at", "f64", 2},
    /* rejected: f16 width outside {2, 4, 8} */
    {"vec", "f16", 1},
    {"vec", "f16", 3},
    {"vec", "f16", 16},
    {"vec_at", "f16", 1},
    {"vec_at", "f16", 6},
    {"vec_at_mask", "f16", 3},
    /* rejected: mask on a global view; buffer view without a BufferResource */
    {"mask_global", "f16", 4},
    {"no_rsrc", "f16", 4},
};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

static const int SHAPE[2] = {4, 64};

static const rocke_type_t* type_by_name(const char* name)
{
    if(strcmp(name, "f16") == 0)
        return rocke_f16();
    if(strcmp(name, "bf16") == 0)
        return rocke_bf16();
    if(strcmp(name, "f32") == 0)
        return rocke_f32();
    if(strcmp(name, "i32") == 0)
        return rocke_i32();
    if(strcmp(name, "i64") == 0)
        return rocke_i64();
    return rocke_f64();
}

static bool op_is(const case_t* c, const char* op)
{
    return strcmp(c->op, op) == 0;
}

/* Returns false (builder error set or unknown op) when the case is rejected. */
static bool build_tensor_view_buffer(rocke_ir_builder_t* b, const case_t* c)
{
    const rocke_type_t* dtype = type_by_name(c->dtype);
    const int n = c->n;
    rocke_tv_buffer_resource_t x_rsrc, y_rsrc;
    rocke_tensor_view_t x_view, y_view;
    rocke_stride_t y_strides[2];

    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, dtype, "global"), NULL);
    rocke_value_t* n_bytes = rocke_b_param(b, "N_bytes", rocke_i32(), NULL);
    rocke_value_t* s_row = rocke_b_param(b, "S_row", rocke_i32(), NULL);
    rocke_value_t* s_off = rocke_b_param(b, "S_off", rocke_i32(), NULL);
    /* Default soffset (const_i32(0)) for X; an explicit one for Y. */
    if(rocke_tv_make_buffer_resource(b, &x_rsrc, x, n_bytes, NULL) != ROCKE_OK)
        return false;
    if(rocke_tv_make_buffer_resource(b, &y_rsrc, y, n_bytes, s_off) != ROCKE_OK)
        return false;
    /* Packed row-major for X (no BufferResource for "no_rsrc": TypeError on
     * first use); a runtime row stride for Y. */
    if(rocke_tv_make_buffer_view(
           &x_view, op_is(c, "no_rsrc") ? NULL : &x_rsrc, SHAPE, 2, dtype, NULL)
       != ROCKE_OK)
        return false;
    y_strides[0] = rocke_stride_value(s_row);
    y_strides[1] = rocke_stride_imm(1);
    if(rocke_tv_make_buffer_view(&y_view, &y_rsrc, SHAPE, 2, dtype, y_strides) != ROCKE_OK)
        return false;
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* row = rocke_b_const_i32(b, 1);
    rocke_value_t* idx[2] = {row, tid};

    if(op_is(c, "scalar"))
    {
        rocke_value_t* v = rocke_tensor_view_load_scalar(b, &x_view, idx, 2);
        rocke_tensor_view_store_scalar(b, &y_view, idx, 2, v, 0);
    }
    else if(op_is(c, "vec") || op_is(c, "no_rsrc"))
    {
        rocke_value_t* v = rocke_tensor_view_load_vec(b, &x_view, idx, 2, n);
        rocke_tensor_view_store_vec(b, &y_view, idx, 2, v, n);
    }
    else if(op_is(c, "vec_at"))
    {
        rocke_value_t* c_n = rocke_b_const_i32(b, n);
        rocke_value_t* off = rocke_b_mul(b, tid, c_n);
        rocke_value_t* v = rocke_tensor_view_load_vec_at(b, &x_view, off, n);
        rocke_tensor_view_store_vec_at(b, &y_view, off, v, n);
    }
    else if(op_is(c, "vec_at_mask"))
    {
        rocke_value_t* c_n = rocke_b_const_i32(b, n);
        rocke_value_t* off = rocke_b_mul(b, tid, c_n);
        rocke_value_t* c_lim = rocke_b_const_i32(b, 48);
        rocke_value_t* mask = rocke_b_cmp_lt(b, tid, c_lim);
        rocke_value_t* v = rocke_tensor_view_load_vec_at_masked(b, &x_view, off, n, mask);
        rocke_tensor_view_store_vec_at_masked(b, &y_view, off, v, n, mask);
    }
    else if(op_is(c, "mask_global"))
    {
        rocke_tensor_view_t g_view;
        if(rocke_make_global_view(&g_view, x, SHAPE, 2, dtype, NULL) != ROCKE_OK)
            return false;
        rocke_value_t* c_n = rocke_b_const_i32(b, n);
        rocke_value_t* off = rocke_b_mul(b, tid, c_n);
        rocke_value_t* c_lim = rocke_b_const_i32(b, 48);
        rocke_value_t* mask = rocke_b_cmp_lt(b, tid, c_lim);
        rocke_tensor_view_load_vec_at_masked(b, &g_view, off, n, mask);
    }
    else if(op_is(c, "window") || op_is(c, "window_f32"))
    {
        static const int lengths[2] = {1, 8};
        rocke_tile_window_t x_win, y_win;
        if(rocke_tensor_view_tile(&x_win, &x_view, lengths, idx, 2) != ROCKE_OK)
            return false;
        if(rocke_tensor_view_tile(&y_win, &y_view, lengths, idx, 2) != ROCKE_OK)
            return false;
        rocke_value_t* zero = rocke_b_const_i32(b, 0);
        rocke_value_t* local[2] = {zero, zero};
        if(op_is(c, "window"))
        {
            rocke_value_t* v = rocke_tile_window_load_vec(b, &x_win, local, 2, n);
            rocke_tile_window_store_vec(b, &y_win, local, 2, v, n);
        }
        else
        {
            rocke_value_t* vals[8];
            if(n < 1 || n > 8)
                return false;
            rocke_tile_window_load_vec_as_f32(b, &x_win, local, 2, n, vals);
            rocke_tile_window_store_vec_from_f32(b, &y_win, local, 2, vals, n);
        }
    }
    else
    {
        fprintf(stderr, "unknown op %s\n", c->op);
        return false;
    }
    if(!rocke_ir_builder_ok(b))
        return false;
    rocke_b_ret(b);
    return true;
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
    if(rocke_ir_builder_init(&b, "tensor_view_buffer") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    bool built;
    try
    {
        built = build_tensor_view_buffer(&b, c);
    }
    catch(const std::exception& e)
    {
        /* The helpers report a Python raise (TypeError / ValueError /
         * NotImplementedError) by throwing. */
        fprintf(stderr, "builder error: %s\n", e.what());
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(!built || !rocke_ir_builder_ok(&b))
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
        size_t nd = 0;
        rocke_verify(kernel, &d, &nd);
        for(size_t i = 0; i < nd; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, nd);
    }

    rocke_ir_builder_free(&b);
    return 0;
}
