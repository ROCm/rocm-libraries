// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_elementwise.c -- C99 port of
 * rocke/instances/common/elementwise.py.
 *
 * Byte-identical builder-call sequence vs the Python build_elementwise. The
 * distribution-driven load/store machinery (load_tile / store_tile /
 * make_static_distributed_tensor / TileDistribution.iterate_ys) is not yet a
 * standalone C helper, so the small slice elementwise.py actually exercises is
 * reproduced here as static helpers that mirror the Python builder-call order
 * exactly (see rocke_ew_load_tile / rocke_ew_store_tile below).
 *
 * The elementwise distribution is fixed:
 *   Hs = ((block_size, vec),)   -> num_X = 1
 *   Ps2RHs = ((1,),)/((0,),)    -> num_P = 1   (lane id feeds H level 0)
 *   Ys2RHs = (1,)/(1,)          -> num_Y = 1   (per-thread vector feeds level 1)
 * make_load_store_traits picks vector_dim_y = 0 and scalar_per_vector = vec
 * (the only Y dim is stride-1; vec in {2,4,8} is already a power of two <= 8).
 * num_access == 1, so load_tile / store_tile issue exactly one vector access at
 * y_base == (0,).
 */
#include "rocke/instance_elementwise.h"

#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.activations.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */

rocke_elementwise_spec_t rocke_elementwise_spec_default(void)
{
    rocke_elementwise_spec_t s;
    s.op = NULL;
    s.dtype = "f16";
    s.block_size = 256;
    s.vec = 8;
    s.name = "rocke_elementwise";
    return s;
}

static bool rocke_ew_streq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

bool rocke_elementwise_is_unary(const rocke_elementwise_spec_t* spec)
{
    const char* op;
    if(spec == NULL)
    {
        return false;
    }
    op = spec->op;
    return rocke_ew_streq(op, "copy") || rocke_ew_streq(op, "neg") || rocke_ew_streq(op, "abs")
           || rocke_ew_streq(op, "relu") || rocke_ew_streq(op, "gelu_tanh")
           || rocke_ew_streq(op, "quick_gelu") || rocke_ew_streq(op, "silu")
           || rocke_ew_streq(op, "swish") || rocke_ew_streq(op, "tanh")
           || rocke_ew_streq(op, "sigmoid") || rocke_ew_streq(op, "exp2");
}

bool rocke_elementwise_is_binary(const rocke_elementwise_spec_t* spec)
{
    const char* op;
    if(spec == NULL)
    {
        return false;
    }
    op = spec->op;
    return rocke_ew_streq(op, "add") || rocke_ew_streq(op, "sub") || rocke_ew_streq(op, "mul")
           || rocke_ew_streq(op, "max") || rocke_ew_streq(op, "min") || rocke_ew_streq(op, "swiglu")
           || rocke_ew_streq(op, "geglu");
}

bool rocke_elementwise_is_bias(const rocke_elementwise_spec_t* spec)
{
    /* op.startswith("bias_") */
    if(spec == NULL || spec->op == NULL)
    {
        return false;
    }
    return strncmp(spec->op, "bias_", 5) == 0;
}

int rocke_elementwise_elems_per_block(const rocke_elementwise_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->block_size * spec->vec;
}

rocke_status_t
    rocke_elementwise_kernel_name(const rocke_elementwise_spec_t* spec, char* out, size_t out_cap)
{
    /* kernel_name_join(name, op, dtype, f"b{block_size}", f"v{vec}") */
    char bstr[32];
    char vstr[32];
    const char* parts[4];
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(bstr, sizeof(bstr), "b%d", spec->block_size);
    snprintf(vstr, sizeof(vstr), "v%d", spec->vec);
    parts[0] = spec->op;
    parts[1] = spec->dtype;
    parts[2] = bstr;
    parts[3] = vstr;
    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

static bool rocke_ew_reason(char* reason, size_t cap, const char* msg)
{
    if(reason != NULL && cap > 0)
    {
        size_t n = strlen(msg);
        if(n >= cap)
        {
            n = cap - 1;
        }
        memcpy(reason, msg, n);
        reason[n] = '\0';
    }
    return false;
}

bool rocke_elementwise_is_valid_spec(const rocke_elementwise_spec_t* spec,
                                     char* reason,
                                     size_t reason_cap)
{
    char buf[128];
    if(spec == NULL)
    {
        return rocke_ew_reason(reason, reason_cap, "null spec");
    }
    if(!(rocke_elementwise_is_unary(spec) || rocke_elementwise_is_binary(spec)))
    {
        snprintf(buf, sizeof(buf), "unknown op %s", spec->op ? spec->op : "(null)");
        return rocke_ew_reason(reason, reason_cap, buf);
    }
    if(!(rocke_ew_streq(spec->dtype, "f16") || rocke_ew_streq(spec->dtype, "bf16")))
    {
        snprintf(buf, sizeof(buf), "unsupported dtype %s", spec->dtype ? spec->dtype : "(null)");
        return rocke_ew_reason(reason, reason_cap, buf);
    }
    if(!(spec->block_size == 64 || spec->block_size == 128 || spec->block_size == 256
         || spec->block_size == 512 || spec->block_size == 1024))
    {
        snprintf(
            buf, sizeof(buf), "block_size %d not in {64, 128, 256, 512, 1024}", spec->block_size);
        return rocke_ew_reason(reason, reason_cap, buf);
    }
    if(!(spec->vec == 2 || spec->vec == 4 || spec->vec == 8))
    {
        snprintf(buf, sizeof(buf), "vec %d not in {2, 4, 8}", spec->vec);
        return rocke_ew_reason(reason, reason_cap, buf);
    }
    if(reason != NULL && reason_cap > 0)
    {
        rocke_ew_reason(reason, reason_cap, "ok");
    }
    return true;
}

/* ===================================================================== *
 *  Op kernels (f32 scalar arithmetic) -- mirrors _gelu_tanh / _apply_unary /
 *  _apply_binary one builder call at a time.
 * ===================================================================== */

/* gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static rocke_value_t* rocke_ew_gelu_tanh(rocke_ir_builder_t* b, rocke_value_t* x)
{
    rocke_value_t* c_half = rocke_b_const_f32(b, 0.5);
    rocke_value_t* c_one = rocke_b_const_f32(b, 1.0);
    rocke_value_t* c_sq2_over_pi = rocke_b_const_f32(b, 0.7978845608028654);
    rocke_value_t* c_a = rocke_b_const_f32(b, 0.044715);
    rocke_value_t* x2 = rocke_b_fmul(b, x, x);
    rocke_value_t* x3 = rocke_b_fmul(b, x2, x);
    rocke_value_t* inner
        = rocke_b_fmul(b, c_sq2_over_pi, rocke_b_fadd(b, x, rocke_b_fmul(b, c_a, x3)));
    /* Python evaluates the outer fmul's arguments left-to-right, so the
     * ``0.5 * x`` half is emitted BEFORE the tanh chain. C leaves the
     * argument evaluation order of ``rocke_b_fmul(b, fmul(c_half,x), fadd(...))``
     * unspecified (compilers commonly evaluate right-to-left, emitting the
     * tanh chain first). Sequence the sub-expressions into explicit locals so
     * the builder-call order matches Python byte-for-byte. */
    rocke_value_t* half_x = rocke_b_fmul(b, c_half, x);
    rocke_value_t* one_plus_tanh = rocke_b_fadd(b, c_one, rocke_tanh_via_exp2(b, inner));
    return rocke_b_fmul(b, half_x, one_plus_tanh);
}

/* Returns the applied unary op, or NULL with a sticky ValueError set on `b`
 * for an unsupported op (mirrors the Python ``raise ValueError``). */
static rocke_value_t* rocke_ew_apply_unary(rocke_ir_builder_t* b, rocke_value_t* x, const char* op)
{
    if(rocke_ew_streq(op, "copy"))
    {
        return x;
    }
    if(rocke_ew_streq(op, "neg"))
    {
        return rocke_b_fneg(b, x);
    }
    if(rocke_ew_streq(op, "abs"))
    {
        return rocke_b_fmax(b, x, rocke_b_fneg(b, x));
    }
    if(rocke_ew_streq(op, "relu"))
    {
        return rocke_b_fmax(b, x, rocke_b_const_f32(b, 0.0));
    }
    if(rocke_ew_streq(op, "exp2"))
    {
        return rocke_b_exp2(b, x);
    }
    if(rocke_ew_streq(op, "tanh"))
    {
        return rocke_tanh_via_exp2(b, x);
    }
    if(rocke_ew_streq(op, "sigmoid"))
    {
        return rocke_sigmoid_via_exp2(b, x);
    }
    if(rocke_ew_streq(op, "silu") || rocke_ew_streq(op, "swish"))
    {
        return rocke_b_fmul(b, x, rocke_sigmoid_via_exp2(b, x));
    }
    if(rocke_ew_streq(op, "quick_gelu"))
    {
        rocke_value_t* c_1702 = rocke_b_const_f32(b, 1.702);
        return rocke_b_fmul(b, x, rocke_sigmoid_via_exp2(b, rocke_b_fmul(b, c_1702, x)));
    }
    if(rocke_ew_streq(op, "gelu_tanh"))
    {
        return rocke_ew_gelu_tanh(b, x);
    }
    {
        char msg[64];
        snprintf(msg, sizeof(msg), "unsupported unary op %s", op ? op : "(null)");
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
    }
    return NULL;
}

static rocke_value_t*
    rocke_ew_apply_binary(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, const char* op)
{
    if(rocke_ew_streq(op, "add"))
    {
        return rocke_b_fadd(b, a, c);
    }
    if(rocke_ew_streq(op, "sub"))
    {
        return rocke_b_fsub(b, a, c);
    }
    if(rocke_ew_streq(op, "mul"))
    {
        return rocke_b_fmul(b, a, c);
    }
    if(rocke_ew_streq(op, "max"))
    {
        return rocke_b_fmax(b, a, c);
    }
    if(rocke_ew_streq(op, "min"))
    {
        return rocke_b_fmin(b, a, c);
    }
    if(rocke_ew_streq(op, "swiglu"))
    {
        return rocke_b_fmul(b, rocke_b_fmul(b, a, rocke_sigmoid_via_exp2(b, a)), c);
    }
    if(rocke_ew_streq(op, "geglu"))
    {
        return rocke_b_fmul(b, rocke_ew_gelu_tanh(b, a), c);
    }
    {
        char msg[64];
        snprintf(msg, sizeof(msg), "unsupported binary op %s", op ? op : "(null)");
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
    }
    return NULL;
}

/* ===================================================================== *
 *  Distribution-driven load / store specialised for the elementwise tile.
 *
 *  Reproduces the Python load_tile / store_tile builder-call order for the
 *  fixed single-Y, single-P, single-X distribution this instance uses. The
 *  per-thread register tile holds exactly `vec` f32 scalars (storage[0..vec)).
 * ===================================================================== */

/* load_tile(window, distribution, ps=[[tid]]) for the elementwise distribution.
 *
 *   traits.iterate_accesses() yields one base y_base = (0,)
 *   x_coords = distribution.calculate_x(ys=[const_i32(0)], ps=[[tid]])
 *   scalars  = window.load_vec_as_f32(*x_coords, n=vec)
 *   dt[k] = scalars[k]
 *
 * `out_storage` must have capacity >= vec. Returns 1 on success, 0 on failure
 * (builder error). */
static int rocke_ew_load_tile(rocke_ir_builder_t* b,
                              const rocke_tile_window_t* window,
                              const rocke_tile_distribution_t* dist,
                              rocke_value_t* tid,
                              int vec,
                              rocke_value_t** out_storage)
{
    rocke_value_t* ys[1];
    rocke_value_t* ps_row[1];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* x_coords[1];
    rocke_value_t* loaded;
    int k;

    /* ys = [b.const_i32(0)] (single Y access at y_base==(0,)). */
    ys[0] = rocke_b_const_i32(b, 0);
    /* ps = [[tid]] */
    ps_row[0] = tid;
    ps[0] = ps_row;
    ps_counts[0] = 1;

    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 1, ps, ps_counts, 1, x_coords, 1))
    {
        return 0;
    }

    /* window.load_vec_as_f32(*x_coords, n=vec):
     *   v = load_vec(x_coords, n=vec)
     *   scalars[k] = cast_to_f32(vec_extract(v, k))
     * (vec is always >= 2 here, so the n==1 scalar branch never triggers). */
    loaded = rocke_tile_window_load_vec(b, window, x_coords, 1, vec);
    for(k = 0; k < vec; ++k)
    {
        out_storage[k] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, loaded, k));
    }
    return rocke_ir_builder_ok(b) ? 1 : 0;
}

/* store_tile(window, distributed, ps=[[tid]]) for the elementwise distribution.
 *
 *   x_coords = calculate_x(ys=[const_i32(0)], ps=[[tid]])
 *   scalars  = storage[0..vec)
 *   window.store_vec_from_f32(*x_coords, values=scalars):
 *       casts[k] = cast_f32_to(scalars[k], dtype)
 *       packed   = vec_pack(casts, dtype)
 *       store_vec(x_coords, packed, n=vec)
 */
static void rocke_ew_store_tile(rocke_ir_builder_t* b,
                                const rocke_tile_window_t* window,
                                const rocke_tile_distribution_t* dist,
                                rocke_value_t* tid,
                                int vec,
                                rocke_value_t** storage)
{
    rocke_value_t* ys[1];
    rocke_value_t* ps_row[1];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* x_coords[1];
    rocke_value_t* casts[8];
    rocke_value_t* packed;
    const rocke_type_t* dtype;
    int k;

    ys[0] = rocke_b_const_i32(b, 0);
    ps_row[0] = tid;
    ps[0] = ps_row;
    ps_counts[0] = 1;

    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 1, ps, ps_counts, 1, x_coords, 1))
    {
        return;
    }

    dtype = rocke_tile_window_dtype(window);
    for(k = 0; k < vec; ++k)
    {
        casts[k] = rocke_b_cast_f32_to(b, storage[k], dtype);
    }
    packed = rocke_b_vec_pack(b, casts, vec, dtype);
    rocke_tile_window_store_vec(b, window, x_coords, 1, packed, vec);
}

/* ===================================================================== *
 *  build_elementwise
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_elementwise(rocke_ir_builder_t* b,
                                            const rocke_elementwise_spec_t* spec)
{
    const rocke_type_t* io_ty;
    bool is_binary;
    int tile_elems;

    rocke_value_t* A;
    rocke_value_t* Bp = NULL;
    rocke_value_t* C;
    rocke_value_t* N;

    rocke_tile_distribution_encoding_t* encoding;
    rocke_tile_distribution_t* distribution;

    rocke_tensor_view_t a_view, b_view, c_view;
    rocke_tile_window_t a_tile, b_tile, c_tile;

    rocke_value_t* tid;
    rocke_value_t* bid;
    rocke_value_t* c_vec;
    rocke_value_t* c_chunk;
    rocke_value_t* block_base;
    rocke_value_t* thread_base;
    rocke_value_t* fast_lim;
    rocke_value_t* in_fast;

    rocke_param_opts_t opts;
    int shape1[1];

    /* Encoding arrays. */
    int h_levels[2];
    rocke_h_row_t hs[1];
    int p_major[1];
    int p_minor[1];
    rocke_p_seq_t ps_seq[1];
    int ys_major[1];
    int ys_minor[1];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    {
        char reason[128];
        if(!rocke_elementwise_is_valid_spec(spec, reason, sizeof(reason)))
        {
            char msg[160];
            snprintf(msg, sizeof(msg), "invalid elementwise spec: %s", reason);
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", msg);
            return NULL;
        }
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    is_binary = rocke_elementwise_is_binary(spec);
    tile_elems = spec->block_size * spec->vec;

    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* A = b.param("A", PtrType(io_ty,"global"), noalias=True, readonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    opts.addr_space = NULL; /* PtrType space "global" handled by rocke_ptr_type below */
    A = rocke_b_param(b, "A", rocke_ptr_type(b, io_ty, "global"), &opts);

    if(is_binary)
    {
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Bp = rocke_b_param(b, "B", rocke_ptr_type(b, io_ty, "global"), &opts);
    }

    /* C = b.param("C", PtrType(io_ty,"global"), noalias=True, writeonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    C = rocke_b_param(b, "C", rocke_ptr_type(b, io_ty, "global"), &opts);

    /* N = b.param("N", I32) */
    N = rocke_b_param(b, "N", rocke_i32(), NULL);

    /* TileDistributionEncoding(
     *     Hs=((block_size, vec),),
     *     Ps2RHs_major=((1,),), Ps2RHs_minor=((0,),),
     *     Ys2RHs_major=(1,), Ys2RHs_minor=(1,)) */
    h_levels[0] = spec->block_size;
    h_levels[1] = spec->vec;
    hs[0].levels = h_levels;
    hs[0].count = 2;
    p_major[0] = 1;
    p_minor[0] = 0;
    ps_seq[0].major = p_major;
    ps_seq[0].minor = p_minor;
    ps_seq[0].count = 1;
    ys_major[0] = 1;
    ys_minor[0] = 1;

    encoding = rocke_make_tile_distribution_encoding(b,
                                                     /*Rs*/ NULL,
                                                     0,
                                                     hs,
                                                     1,
                                                     ps_seq,
                                                     1,
                                                     ys_major,
                                                     ys_minor,
                                                     1);
    if(encoding == NULL)
    {
        return NULL;
    }
    distribution = rocke_make_static_tile_distribution(b, encoding);
    if(distribution == NULL)
    {
        return NULL;
    }

    /* 1D views over the contiguous buffer (packed strides => stride 1). */
    shape1[0] = tile_elems;
    if(rocke_make_global_view(&a_view, A, shape1, 1, io_ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(A) failed");
        return NULL;
    }
    if(rocke_make_global_view(&c_view, C, shape1, 1, io_ty, NULL) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(C) failed");
        return NULL;
    }
    if(is_binary)
    {
        if(rocke_make_global_view(&b_view, Bp, shape1, 1, io_ty, NULL) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_global_view(B) failed");
            return NULL;
        }
    }

    tid = rocke_b_thread_id_x(b);
    bid = rocke_b_block_id_x(b);
    c_vec = rocke_b_const_i32(b, spec->vec);
    c_chunk = rocke_b_const_i32(b, tile_elems);

    block_base = rocke_b_mul(b, bid, c_chunk);
    thread_base = rocke_b_add(b, block_base, rocke_b_mul(b, tid, c_vec));

    fast_lim = rocke_b_add(b, thread_base, c_vec);
    in_fast = rocke_b_cmp_le(b, fast_lim, N);

    /* Per-block tile windows anchored at this CTA's slab origin (origin =
     * (block_base,)). */
    {
        rocke_value_t* origin[1];
        int lengths1[1];
        origin[0] = block_base;
        lengths1[0] = tile_elems;
        if(rocke_make_tile_window(&a_tile, &a_view, lengths1, origin, 1) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_tile_window(A) failed");
            return NULL;
        }
        if(rocke_make_tile_window(&c_tile, &c_view, lengths1, origin, 1) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_tile_window(C) failed");
            return NULL;
        }
        if(is_binary)
        {
            if(rocke_make_tile_window(&b_tile, &b_view, lengths1, origin, 1) != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "%s", "make_tile_window(B) failed");
                return NULL;
            }
        }
    }

    /* with b.scf_if(in_fast): emit_vec_path() */
    {
        rocke_if_t gate = rocke_b_scf_if(b, in_fast);
        rocke_b_region_enter(b, gate.then_region);
        {
            /* a_dt = a_tile.load(distribution, ps=[[tid]]) */
            rocke_value_t* a_dt[8];
            rocke_value_t* out_dt[8];
            int y;
            if(!rocke_ew_load_tile(b, &a_tile, distribution, tid, spec->vec, a_dt))
            {
                rocke_b_region_leave(b);
                return NULL;
            }
            if(is_binary)
            {
                rocke_value_t* b_dt[8];
                if(!rocke_ew_load_tile(b, &b_tile, distribution, tid, spec->vec, b_dt))
                {
                    rocke_b_region_leave(b);
                    return NULL;
                }
                for(y = 0; y < spec->vec; ++y)
                {
                    out_dt[y] = rocke_ew_apply_binary(b, a_dt[y], b_dt[y], spec->op);
                }
            }
            else
            {
                for(y = 0; y < spec->vec; ++y)
                {
                    out_dt[y] = rocke_ew_apply_unary(b, a_dt[y], spec->op);
                }
            }
            /* c_tile.store(out_dt, ps=[[tid]]) */
            rocke_ew_store_tile(b, &c_tile, distribution, tid, spec->vec, out_dt);
        }
        rocke_b_region_leave(b);
    }

    /* with b.scf_if(b.lnot(in_fast)): emit_scalar_path() */
    {
        rocke_value_t* not_fast = rocke_b_lnot(b, in_fast);
        rocke_if_t gate = rocke_b_scf_if(b, not_fast);
        rocke_b_region_enter(b, gate.then_region);
        {
            int i;
            for(i = 0; i < spec->vec; ++i)
            {
                /* idx = thread_base + const_i32(i) */
                rocke_value_t* idx = rocke_b_add(b, thread_base, rocke_b_const_i32(b, i));
                rocke_value_t* in_bounds = rocke_b_cmp_lt(b, idx, N);
                rocke_if_t ib = rocke_b_scf_if(b, in_bounds);
                rocke_b_region_enter(b, ib.then_region);
                {
                    rocke_value_t* indices[1];
                    rocke_value_t* a_s;
                    rocke_value_t* r;
                    indices[0] = idx;
                    /* a = cast_to_f32(a_view.load_scalar([idx])) */
                    a_s = rocke_b_cast_to_f32(
                        b, rocke_tensor_view_load_scalar(b, &a_view, indices, 1));
                    if(is_binary)
                    {
                        rocke_value_t* bv = rocke_b_cast_to_f32(
                            b, rocke_tensor_view_load_scalar(b, &b_view, indices, 1));
                        r = rocke_ew_apply_binary(b, a_s, bv, spec->op);
                    }
                    else
                    {
                        r = rocke_ew_apply_unary(b, a_s, spec->op);
                    }
                    /* c_view.store_scalar([idx], cast_f32_to(r, io_ty)) */
                    rocke_tensor_view_store_scalar(
                        b, &c_view, indices, 1, rocke_b_cast_f32_to(b, r, io_ty), 0);
                }
                rocke_b_region_leave(b);
            }
        }
        rocke_b_region_leave(b);
    }

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_elementwise_new(rocke_ir_builder_t* b,
                                                const rocke_elementwise_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_elementwise_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_elementwise(b, spec);
    });
}

/* ===================================================================== *
 *  elementwise_grid
 * ===================================================================== */

void rocke_elementwise_grid(int numel, const rocke_elementwise_spec_t* spec, int out_grid[3])
{
    int chunk;
    if(out_grid == NULL)
    {
        return;
    }
    out_grid[0] = 0;
    out_grid[1] = 1;
    out_grid[2] = 1;
    if(spec == NULL)
    {
        return;
    }
    chunk = rocke_elementwise_elems_per_block(spec);
    if(chunk <= 0)
    {
        return;
    }
    out_grid[0] = (numel + chunk - 1) / chunk;
}

/* ===================================================================== *
 *  rocke_elementwise_lower_to_llvm -- build + lower to .ll convenience.
 * ===================================================================== */

rocke_status_t rocke_elementwise_lower_to_llvm(const rocke_elementwise_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_elementwise_new(&b, spec);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_elementwise failed";
            }
            n = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
