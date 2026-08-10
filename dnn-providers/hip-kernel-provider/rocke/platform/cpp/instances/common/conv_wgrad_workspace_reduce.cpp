/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of rocke/instances/common/conv_wgrad_workspace_reduce.py
 *
 * Stage 2 of the deterministic two-stage wgrad path.  Reads the f32 workspace
 * buffer [split_k, wg_M, wg_N] written by Stage 1 and reduces along the
 * split_k axis in a fixed sequential order (k_id = 0, 1, ..., split_k - 1).
 * The fixed iteration order guarantees bit-exact, run-to-run determinism.
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_wgrad_workspace_reduce.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type, rocke_b_store_scalar_from_f32 */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, SignatureBuilder */
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

#ifndef ROCKE_WGRAD_REDUCE_DEFAULT_TILE_M
#define ROCKE_WGRAD_REDUCE_DEFAULT_TILE_M 4
#endif
#ifndef ROCKE_WGRAD_REDUCE_DEFAULT_TILE_N
#define ROCKE_WGRAD_REDUCE_DEFAULT_TILE_N 64
#endif

/* ---- spec helpers -------------------------------------------------------- */

rocke_wgrad_reduce_spec_t rocke_wgrad_reduce_spec_default(void)
{
    rocke_wgrad_reduce_spec_t s;
    memset(&s, 0, sizeof(s));
    s.dtype_d = "fp16";
    s.tile_m = ROCKE_WGRAD_REDUCE_DEFAULT_TILE_M;
    s.tile_n = ROCKE_WGRAD_REDUCE_DEFAULT_TILE_N;
    s.name = "conv_wgrad_ws_reduce";
    s.problem_short = "";
    return s;
}

int rocke_wgrad_reduce_spec_block_size(const rocke_wgrad_reduce_spec_t* spec)
{
    return spec->tile_m * spec->tile_n;
}

rocke_status_t
    rocke_wgrad_reduce_kernel_name(const rocke_wgrad_reduce_spec_t* spec, char* out, int out_cap)
{
    /* Python: kernel_name_join(name, problem.short(), f"t{tile_m}x{tile_n}", dtype_d) */
    char t_buf[32];
    snprintf(t_buf, sizeof(t_buf), "t%dx%d", spec->tile_m, spec->tile_n);
    const char* parts[3] = {spec->problem_short ? spec->problem_short : "",
                            t_buf,
                            spec->dtype_d ? spec->dtype_d : "fp16"};
    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_wgrad_reduce_is_valid_spec(const rocke_wgrad_reduce_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      int reason_cap)
{
    (void)arch;
    if(spec->wg_M <= 0 || spec->wg_N <= 0)
    {
        if(reason)
            snprintf(reason, reason_cap, "wg_M and wg_N must be > 0");
        return false;
    }
    if(spec->tile_m <= 0 || spec->tile_n <= 0)
    {
        if(reason)
            snprintf(reason, reason_cap, "tile_m and tile_n must be > 0");
        return false;
    }
    int bs = rocke_wgrad_reduce_spec_block_size(spec);
    if(bs <= 0 || bs > 1024)
    {
        if(reason)
            snprintf(reason, reason_cap, "block_size=%d out of range [1,1024]", bs);
        return false;
    }
    const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";
    if(strcmp(dtype_d, "fp16") != 0 && strcmp(dtype_d, "fp32") != 0 && strcmp(dtype_d, "bf16") != 0)
    {
        if(reason)
            snprintf(reason, reason_cap, "dtype_d must be fp16/bf16/fp32 (got '%s')", dtype_d);
        return false;
    }
    return true;
}

/* ---- main builder -------------------------------------------------------- */

rocke_kernel_def_t* rocke_build_wgrad_workspace_reduce(rocke_ir_builder_t* b,
                                                       const rocke_wgrad_reduce_spec_t* spec,
                                                       const char* arch)
{
    if(!b || !spec)
        return NULL;
    if(!arch)
        arch = "gfx950";

    char reason[256];
    if(!rocke_wgrad_reduce_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wgrad_reduce: %s", reason);
        return NULL;
    }

    const int tile_m = spec->tile_m;
    const int tile_n = spec->tile_n;
    const int BS = tile_m * tile_n; /* one thread per output element */
    const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";
    const bool is_fp32_out = (strcmp(dtype_d, "fp32") == 0 || strcmp(dtype_d, "f32") == 0);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* ---- params ---------------------------------------------------------- */
    rocke_param_opts_t ro;
    memset(&ro, 0, sizeof(ro));
    ro.noalias = ro.noalias_set = true;
    ro.readonly = ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;

    rocke_param_opts_t wo;
    memset(&wo, 0, sizeof(wo));
    wo.noalias = wo.noalias_set = true;
    wo.writeonly = wo.writeonly_set = true;
    wo.align = 16;
    wo.align_set = true;

    rocke_value_t* ws_ptr
        = rocke_b_param(b, "ws_ptr", rocke_ptr_type(b, rocke_f32(), "global"), &ro);
    /* fp32 output: use f32 pointer directly (io_ir_type only handles f16/bf16). */
    const rocke_type_t* dw_elem = is_fp32_out ? rocke_f32() : rocke_b_io_ir_type(b, dtype_d);
    rocke_value_t* dw_ptr = rocke_b_param(b, "dw_ptr", rocke_ptr_type(b, dw_elem, "global"), &wo);
    rocke_value_t* wg_M_p = rocke_b_param(b, "wg_M", rocke_i32(), NULL);
    rocke_value_t* wg_N_p = rocke_b_param(b, "wg_N", rocke_i32(), NULL);
    rocke_value_t* sk_p = rocke_b_param(b, "split_k", rocke_i32(), NULL);
    rocke_b_param(b, "ws_bytes", rocke_i32(), NULL); /* consumed by host */
    rocke_b_param(b, "dw_bytes", rocke_i32(), NULL); /* consumed by host */

    /* ---- thread / block indices ------------------------------------------ */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* blk_m = rocke_b_block_id_y(b); /* M tiles */
    rocke_value_t* blk_n = rocke_b_block_id_x(b); /* N tiles */

    /* Each thread in the flat block owns one (m_local, n_local) element.
     * tid = t_m * tile_n + t_n  =>  t_m = tid / tile_n, t_n = tid % tile_n */
    rocke_value_t* tile_n_v = rocke_b_const_i32(b, tile_n);
    rocke_value_t* tile_m_v = rocke_b_const_i32(b, tile_m);
    rocke_value_t* t_m = rocke_b_div(b, tid, tile_n_v);
    rocke_value_t* t_n = rocke_b_mod(b, tid, tile_n_v);

    /* Global (m, n) coordinates */
    rocke_value_t* c_m = rocke_b_add(b, rocke_b_mul(b, blk_m, tile_m_v), t_m);
    rocke_value_t* c_n = rocke_b_add(b, rocke_b_mul(b, blk_n, tile_n_v), t_n);

    /* OOB guard: threads outside [0, wg_M) x [0, wg_N) do nothing */
    rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, wg_M_p);
    rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, wg_N_p);
    rocke_value_t* in_bounds = rocke_b_land(b, m_ok, n_ok);

    rocke_if_t guard = rocke_b_scf_if(b, in_bounds);
    rocke_b_region_enter(b, guard.then_region);
    {
        /* Sequential reduction over split_k slices.
         * acc carries the running f32 sum as a loop-carried value.
         * The fixed iteration order (k_id = 0..split_k-1) is the determinism
         * guarantee: the summation order is determined entirely by the loop
         * bounds and is identical across every run. */
        rocke_value_t* c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* c1 = rocke_b_const_i32(b, 1);
        rocke_value_t* acc0 = rocke_b_const_f32(b, 0.0);

        rocke_iter_arg_t iter_arg;
        iter_arg.name = "acc";
        iter_arg.init = acc0;

        rocke_for_t for_op = rocke_b_scf_for_iter(b,
                                                  c0,
                                                  sk_p,
                                                  c1,
                                                  &iter_arg,
                                                  1,
                                                  "kid",
                                                  /*unroll=*/false,
                                                  /*elide_trailing_barrier=*/true);

        rocke_b_region_enter(b, for_op.body);
        {
            rocke_value_t* kid = for_op.iv;
            rocke_value_t* acc_in = for_op.iter_vars[0];

            /* ws_off = kid * wg_M * wg_N + c_m * wg_N + c_n */
            rocke_value_t* slice_base = rocke_b_mul(b, kid, rocke_b_mul(b, wg_M_p, wg_N_p));
            rocke_value_t* elem_off
                = rocke_b_add(b, slice_base, rocke_b_add(b, rocke_b_mul(b, c_m, wg_N_p), c_n));

            rocke_value_t* partial = rocke_b_global_load_f32(b, ws_ptr, elem_off, 4);
            rocke_value_t* new_acc = rocke_b_fadd(b, acc_in, partial);

            rocke_b_scf_yield(b, &new_acc, 1);
        }
        rocke_b_region_leave(b);

        rocke_value_t* total = for_op.op->results[0];

        /* Output index: c_m * wg_N + c_n */
        rocke_value_t* dw_off = rocke_b_add(b, rocke_b_mul(b, c_m, wg_N_p), c_n);

        /* Store: fp32 output is already the right type; f16/bf16 need cast. */
        if(is_fp32_out)
            rocke_b_global_store(b, dw_ptr, dw_off, total, 4);
        else
            rocke_b_store_scalar_from_f32(b, dw_ptr, dw_off, total, dtype_d);
    }
    rocke_b_region_leave(b);

    if(!rocke_ir_builder_ok(b))
        return NULL;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_wgrad_workspace_reduce_new(rocke_ir_builder_t* b,
                                                           const rocke_wgrad_reduce_spec_t* spec,
                                                           const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        if(b == NULL || spec == NULL)
            return NULL;
        char name[256];
        rocke_wgrad_reduce_kernel_name(spec, name, sizeof(name));
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_wgrad_workspace_reduce(b, spec, arch);
    });
}

void rocke_wgrad_reduce_grid(const rocke_wgrad_reduce_spec_t* spec,
                             int* out_gx,
                             int* out_gy,
                             int* out_gz)
{
    /* x = ceil(wg_N / tile_n), y = ceil(wg_M / tile_m), z = 1 */
    *out_gx = (spec->wg_N + spec->tile_n - 1) / spec->tile_n;
    *out_gy = (spec->wg_M + spec->tile_m - 1) / spec->tile_m;
    *out_gz = 1;
}

rocke_status_t rocke_wgrad_reduce_signature(rocke_arena_t* arena,
                                            const rocke_wgrad_reduce_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
        return ROCKE_ERR_VALUE;

    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
        return st;

    const char* dtype_d = spec->dtype_d ? spec->dtype_d : "fp16";
    rocke_signature_builder_ptr(&sb, "ws_ptr", "fp32", NULL);
    rocke_signature_builder_ptr(&sb, "dw_ptr", dtype_d, NULL);
    rocke_signature_builder_scalar(&sb, "wg_M", "i32");
    rocke_signature_builder_scalar(&sb, "wg_N", "i32");
    rocke_signature_builder_scalar(&sb, "split_k", "i32");
    rocke_signature_builder_scalar(&sb, "ws_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "dw_bytes", "i32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

rocke_status_t rocke_wgrad_reduce_lower_to_llvm(const rocke_wgrad_reduce_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
        return ROCKE_ERR_VALUE;
    if(arch == NULL)
        arch = "gfx950";

    k = rocke_build_wgrad_workspace_reduce_new(&b, spec, arch);
    if(k == NULL)
    {
        /* Capture status BEFORE free (free memsets struct to zero). */
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
                m = "build_wgrad_workspace_reduce failed";
            n = strlen(m);
            if(n >= err_cap)
                n = err_cap - 1;
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(k, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
