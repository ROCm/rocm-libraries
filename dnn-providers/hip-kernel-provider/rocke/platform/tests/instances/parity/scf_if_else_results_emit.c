// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native C counterpart of scf_if_else_results_emit.py (config table must match). */
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ARCHES[] = {"gfx942", "gfx950", "gfx1201"};
static const int NUM_ARCHES = sizeof(ARCHES) / sizeof(ARCHES[0]);

static const char* CASES[] = {"branch", "loop"};
static const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

static void build_branch(rocke_ir_builder_t* b)
{
    rocke_value_t* x = rocke_b_param(b, "X", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* y = rocke_b_param(b, "Y", rocke_ptr_type(b, rocke_f32(), "global"), NULL);
    rocke_value_t* z = rocke_b_param(b, "Z", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    rocke_value_t* lane = rocke_b_thread_id_x(b);
    rocke_value_t* cond = rocke_b_cmp_lt(b, lane, rocke_b_const_i32(b, 24));
    const rocke_type_t* result_types[2] = {rocke_f32(), rocke_vector_type(b, rocke_i32(), 2)};
    rocke_if_else_t branch = rocke_b_scf_if_else_results(b, cond, result_types, 2);
    if(!rocke_ir_builder_ok(b))
        return;

    rocke_b_region_enter(b, branch.then_region);
    {
        rocke_value_t* yields[2];
        yields[0] = rocke_b_global_load(b, x, lane, rocke_f32(), 1);
        yields[1] = rocke_b_vector_splat(b, lane, 2);
        rocke_b_scf_yield(b, yields, 2);
    }
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, branch.else_region);
    {
        rocke_value_t* yields[2];
        yields[0] = rocke_b_const_f32(b, 7.5);
        yields[1] = rocke_b_vector_splat(b, rocke_b_const_i32(b, -1), 2);
        rocke_b_scf_yield(b, yields, 2);
    }
    rocke_b_region_leave(b);

    rocke_b_global_store(b, y, lane, branch.op->results[0], 1);
    rocke_b_global_store(b, z, lane, rocke_b_vec_extract(b, branch.op->results[1], 1), 1);
    rocke_b_ret(b);
}

static void build_loop(rocke_ir_builder_t* b)
{
    rocke_value_t* w = rocke_b_param(b, "W", rocke_ptr_type(b, rocke_i32(), "global"), NULL);
    rocke_value_t* lane = rocke_b_thread_id_x(b);
    rocke_value_t* taken = rocke_b_mod(b, lane, rocke_b_const_i32(b, 4));
    rocke_iter_arg_t iter_args[1];
    rocke_value_t* lo = rocke_b_const_i32(b, 0);
    rocke_value_t* hi = rocke_b_const_i32(b, 4);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    iter_args[0].name = "acc";
    iter_args[0].init = rocke_b_const_i32(b, 0);
    rocke_for_t loop = rocke_b_scf_for_iter(b, lo, hi, one, iter_args, 1, NULL, false, true);
    if(!rocke_ir_builder_ok(b))
        return;

    rocke_b_region_enter(b, loop.body);
    {
        const rocke_type_t* result_types[1] = {rocke_i32()};
        rocke_value_t* cond = rocke_b_cmp_lt(b, loop.iv, taken);
        rocke_if_else_t step = rocke_b_scf_if_else_results(b, cond, result_types, 1);
        rocke_value_t* value;
        rocke_value_t* sum;
        if(!rocke_ir_builder_ok(b))
            return;
        rocke_b_region_enter(b, step.then_region);
        value = rocke_b_const_i32(b, 1);
        rocke_b_scf_yield(b, &value, 1);
        rocke_b_region_leave(b);
        rocke_b_region_enter(b, step.else_region);
        value = rocke_b_const_i32(b, 10);
        rocke_b_scf_yield(b, &value, 1);
        rocke_b_region_leave(b);
        sum = rocke_b_add(b, loop.iter_vars[0], step.op->results[0]);
        rocke_b_scf_yield(b, &sum, 1);
    }
    rocke_b_region_leave(b);

    rocke_b_global_store(b, w, lane, loop.op->results[0], 1);
    rocke_b_ret(b);
}

static void build(rocke_ir_builder_t* b, const char* c)
{
    if(strcmp(c, "branch") == 0)
        build_branch(b);
    else
        build_loop(b);
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
    const char* c = CASES[idx / NUM_ARCHES];
    const char* arch = ARCHES[idx % NUM_ARCHES];

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "scf_if_else_results") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 64 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 64);
    try
    {
        build(&b, c);
    }
    catch(const std::exception& e)
    {
        /* The builder may report a refusal by throwing. */
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
