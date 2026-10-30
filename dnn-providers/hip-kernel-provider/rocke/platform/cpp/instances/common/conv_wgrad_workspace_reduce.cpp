/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of rocke/instances/common/conv_wgrad_workspace_reduce.py
 *
 * Stage 2 of the two-stage wgrad path.  Stage 1 f32-atomic-added its partial
 * sums into a scratch of shape [groups * R, wg_M, wg_N], so the reduction over
 * split_k is already done by the time this runs; this kernel folds the R
 * replica slabs and converts f32 -> dtype_d into dW.  block_id_z encodes the
 * group index.
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_wgrad_workspace_reduce.h"

#include "rocke/arena.h" /* rocke_arena_alloc */
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
/* Scratch replica slabs to fold.  Must match the Stage 1 spec's ws_replicas;
 * mirrors _DEFAULT_WS_REPLICAS in the Python instances. */
#ifndef ROCKE_WGRAD_REDUCE_DEFAULT_WS_REPLICAS
#define ROCKE_WGRAD_REDUCE_DEFAULT_WS_REPLICAS 8
#endif

/* ---- spec helpers -------------------------------------------------------- */

rocke_wgrad_reduce_spec_t rocke_wgrad_reduce_spec_default(void)
{
    rocke_wgrad_reduce_spec_t s;
    memset(&s, 0, sizeof(s));
    s.dtype_d = "fp16";
    s.tile_m = ROCKE_WGRAD_REDUCE_DEFAULT_TILE_M;
    s.tile_n = ROCKE_WGRAD_REDUCE_DEFAULT_TILE_N;
    s.name = "conv_wgrad_ws_cast";
    s.problem_short = "";
    s.groups = 1;
    s.ws_replicas = ROCKE_WGRAD_REDUCE_DEFAULT_WS_REPLICAS;
    return s;
}

int rocke_wgrad_reduce_spec_block_size(const rocke_wgrad_reduce_spec_t* spec)
{
    return spec->tile_m * spec->tile_n;
}

rocke_status_t
    rocke_wgrad_reduce_kernel_name(const rocke_wgrad_reduce_spec_t* spec, char* out, int out_cap)
{
    /* Python: parts = [short, t{m}x{n}] + (["wsr{R}"] if R > 1) + [dtype_d] */
    char t_buf[32];
    char r_buf[32];
    snprintf(t_buf, sizeof(t_buf), "t%dx%d", spec->tile_m, spec->tile_n);
    const char* parts[4];
    int n_parts = 0;
    parts[n_parts++] = spec->problem_short ? spec->problem_short : "";
    parts[n_parts++] = t_buf;
    if(spec->ws_replicas > 1)
    {
        snprintf(r_buf, sizeof(r_buf), "wsr%d", spec->ws_replicas);
        parts[n_parts++] = r_buf;
    }
    parts[n_parts++] = spec->dtype_d ? spec->dtype_d : "fp16";
    return rocke_kernel_name_join(spec->name, parts, n_parts, NULL, NULL, 0, out, out_cap, NULL);
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
    if(spec->ws_replicas < 1)
    {
        if(reason)
            snprintf(reason, reason_cap, "ws_replicas=%d must be >= 1", spec->ws_replicas);
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
    rocke_b_param(b, "ws_bytes", rocke_i32(), NULL); /* consumed by host */
    rocke_b_param(b, "dw_bytes", rocke_i32(), NULL); /* consumed by host */
    /* groups is in the ABI so callers can pass it; the kernel uses block_id_z
     * for the group index instead, so the IR value itself is unused. */
    rocke_b_param(b, "groups", rocke_i32(), NULL);

    /* ---- thread / block indices ------------------------------------------ */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* blk_m = rocke_b_block_id_y(b); /* M tiles (local within group) */
    rocke_value_t* blk_n = rocke_b_block_id_x(b); /* N tiles */
    rocke_value_t* grp_id = rocke_b_block_id_z(b); /* group index */

    /* Each thread in the flat block owns one (m_local, n_local) element.
     * tid = t_m * tile_n + t_n  =>  t_m = tid / tile_n, t_n = tid % tile_n.
     * All constants are passed inline as arguments to match the Python IR
     * builder's constant-folding: Python inlines small ints directly into
     * binary operations without creating separate SSA values. */
    rocke_value_t* t_m = rocke_b_div(b, tid, rocke_b_const_i32(b, tile_n));
    rocke_value_t* t_n = rocke_b_mod(b, tid, rocke_b_const_i32(b, tile_n));

    /* Per-group (m, n) coordinates */
    rocke_value_t* c_m = rocke_b_add(b, rocke_b_mul(b, blk_m, rocke_b_const_i32(b, tile_m)), t_m);
    rocke_value_t* c_n = rocke_b_add(b, rocke_b_mul(b, blk_n, rocke_b_const_i32(b, tile_n)), t_n);

    /* OOB guard: threads outside [0, wg_M) x [0, wg_N) do nothing */
    rocke_value_t* m_ok = rocke_b_cmp_lt(b, c_m, wg_M_p);
    rocke_value_t* n_ok = rocke_b_cmp_lt(b, c_n, wg_N_p);
    rocke_value_t* in_bounds = rocke_b_land(b, m_ok, n_ok);

    rocke_if_t guard = rocke_b_scf_if(b, in_bounds);
    rocke_b_region_enter(b, guard.then_region);
    {
        /* Stage 1's f32 atomics already reduced over split_k; all that is
         * left is to fold this group's R replica slabs and cast.
         *
         *   dw_off = grp_id * wg_M*wg_N + c_m * wg_N + c_n
         *   ws_off = (grp_id * R + r) * wg_M*wg_N + c_m * wg_N + c_n
         *
         * At R == 1 the two coincide and the fold collapses to one load. */
        rocke_value_t* grp_stride = rocke_b_mul(b, wg_M_p, wg_N_p);
        rocke_value_t* elem_in_slab = rocke_b_add(b, rocke_b_mul(b, c_m, wg_N_p), c_n);
        rocke_value_t* dw_off = rocke_b_add(b, rocke_b_mul(b, grp_id, grp_stride), elem_in_slab);

        const int reps = spec->ws_replicas;
        rocke_value_t* total = NULL;
        if(reps == 1)
        {
            total = rocke_b_global_load_f32(b, ws_ptr, dw_off, 4);
        }
        else
        {
            /* R is compile-time, so this is a flat unrolled fold -- R
             * independent loads issued before the first add consumes one. */
            rocke_value_t* ws_base = rocke_b_add(
                b,
                rocke_b_mul(b, rocke_b_mul(b, grp_id, rocke_b_const_i32(b, reps)), grp_stride),
                elem_in_slab);
            rocke_value_t** partials = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, sizeof(rocke_value_t*) * (size_t)reps);
            if(partials == NULL)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "wgrad_reduce: partials alloc failed");
                rocke_b_region_leave(b);
                return NULL;
            }
            for(int r = 0; r < reps; ++r)
            {
                rocke_value_t* off
                    = (r == 0) ? ws_base
                               : rocke_b_add(b,
                                             ws_base,
                                             rocke_b_mul(b, rocke_b_const_i32(b, r), grp_stride));
                partials[r] = rocke_b_global_load_f32(b, ws_ptr, off, 4);
            }
            total = partials[0];
            for(int r = 1; r < reps; ++r)
                total = rocke_b_fadd(b, total, partials[r]);
        }

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
    /* x = ceil(wg_N / tile_n), y = ceil(wg_M / tile_m), z = groups */
    *out_gx = (spec->wg_N + spec->tile_n - 1) / spec->tile_n;
    *out_gy = (spec->wg_M + spec->tile_m - 1) / spec->tile_m;
    *out_gz = (spec->groups > 0) ? spec->groups : 1;
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
    rocke_signature_builder_scalar(&sb, "ws_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "dw_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "groups", "i32");
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
