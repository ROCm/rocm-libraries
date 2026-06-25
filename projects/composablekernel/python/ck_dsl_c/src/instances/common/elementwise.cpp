// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ckc/instance_elementwise.c -- C99 port of
 * ck_dsl/instances/common/elementwise.py.
 *
 * Byte-identical builder-call sequence vs the Python build_elementwise. The
 * distribution-driven load/store machinery (load_tile / store_tile /
 * make_static_distributed_tensor / TileDistribution.iterate_ys) is not yet a
 * standalone C helper, so the small slice elementwise.py actually exercises is
 * reproduced here as static helpers that mirror the Python builder-call order
 * exactly (see ckc_ew_load_tile / ckc_ew_store_tile below).
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
#include "ckc/instance_elementwise.h"

#include <stdio.h>
#include <string.h>

#include "ckc/helper_ck_dsl.helpers.activations.h"
#include "ckc/helper_ck_dsl.helpers.distribution.h"
#include "ckc/helper_ck_dsl.helpers.io.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"
#include "ckc/ir_internal.h"      /* ckc_i_set_err */
#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */

ckc_elementwise_spec_t ckc_elementwise_spec_default(void)
{
    ckc_elementwise_spec_t s;
    s.op         = NULL;
    s.dtype      = "f16";
    s.block_size = 256;
    s.vec        = 8;
    s.name       = "ck_dsl_elementwise";
    return s;
}

static bool ckc_ew_streq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

bool ckc_elementwise_is_unary(const ckc_elementwise_spec_t* spec)
{
    const char* op;
    if(spec == NULL)
    {
        return false;
    }
    op = spec->op;
    return ckc_ew_streq(op, "copy") || ckc_ew_streq(op, "neg") || ckc_ew_streq(op, "abs") ||
           ckc_ew_streq(op, "relu") || ckc_ew_streq(op, "gelu_tanh") ||
           ckc_ew_streq(op, "quick_gelu") || ckc_ew_streq(op, "silu") ||
           ckc_ew_streq(op, "swish") || ckc_ew_streq(op, "tanh") || ckc_ew_streq(op, "sigmoid") ||
           ckc_ew_streq(op, "exp2");
}

bool ckc_elementwise_is_binary(const ckc_elementwise_spec_t* spec)
{
    const char* op;
    if(spec == NULL)
    {
        return false;
    }
    op = spec->op;
    return ckc_ew_streq(op, "add") || ckc_ew_streq(op, "sub") || ckc_ew_streq(op, "mul") ||
           ckc_ew_streq(op, "max") || ckc_ew_streq(op, "min") || ckc_ew_streq(op, "swiglu") ||
           ckc_ew_streq(op, "geglu");
}

bool ckc_elementwise_is_bias(const ckc_elementwise_spec_t* spec)
{
    /* op.startswith("bias_") */
    if(spec == NULL || spec->op == NULL)
    {
        return false;
    }
    return strncmp(spec->op, "bias_", 5) == 0;
}

int ckc_elementwise_elems_per_block(const ckc_elementwise_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return spec->block_size * spec->vec;
}

ckc_status_t
ckc_elementwise_kernel_name(const ckc_elementwise_spec_t* spec, char* out, size_t out_cap)
{
    /* kernel_name_join(name, op, dtype, f"b{block_size}", f"v{vec}") */
    char bstr[32];
    char vstr[32];
    const char* parts[4];
    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    snprintf(bstr, sizeof(bstr), "b%d", spec->block_size);
    snprintf(vstr, sizeof(vstr), "v%d", spec->vec);
    parts[0] = spec->op;
    parts[1] = spec->dtype;
    parts[2] = bstr;
    parts[3] = vstr;
    return ckc_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

static bool ckc_ew_reason(char* reason, size_t cap, const char* msg)
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

bool ckc_elementwise_is_valid_spec(const ckc_elementwise_spec_t* spec,
                                   char* reason,
                                   size_t reason_cap)
{
    char buf[128];
    if(spec == NULL)
    {
        return ckc_ew_reason(reason, reason_cap, "null spec");
    }
    if(!(ckc_elementwise_is_unary(spec) || ckc_elementwise_is_binary(spec)))
    {
        snprintf(buf, sizeof(buf), "unknown op %s", spec->op ? spec->op : "(null)");
        return ckc_ew_reason(reason, reason_cap, buf);
    }
    if(!(ckc_ew_streq(spec->dtype, "f16") || ckc_ew_streq(spec->dtype, "bf16")))
    {
        snprintf(buf, sizeof(buf), "unsupported dtype %s", spec->dtype ? spec->dtype : "(null)");
        return ckc_ew_reason(reason, reason_cap, buf);
    }
    if(!(spec->block_size == 64 || spec->block_size == 128 || spec->block_size == 256 ||
         spec->block_size == 512 || spec->block_size == 1024))
    {
        snprintf(
            buf, sizeof(buf), "block_size %d not in {64, 128, 256, 512, 1024}", spec->block_size);
        return ckc_ew_reason(reason, reason_cap, buf);
    }
    if(!(spec->vec == 2 || spec->vec == 4 || spec->vec == 8))
    {
        snprintf(buf, sizeof(buf), "vec %d not in {2, 4, 8}", spec->vec);
        return ckc_ew_reason(reason, reason_cap, buf);
    }
    if(reason != NULL && reason_cap > 0)
    {
        ckc_ew_reason(reason, reason_cap, "ok");
    }
    return true;
}

/* ===================================================================== *
 *  Op kernels (f32 scalar arithmetic) -- mirrors _gelu_tanh / _apply_unary /
 *  _apply_binary one builder call at a time.
 * ===================================================================== */

/* gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static ckc_value_t* ckc_ew_gelu_tanh(ckc_ir_builder_t* b, ckc_value_t* x)
{
    ckc_value_t* c_half        = ckc_b_const_f32(b, 0.5);
    ckc_value_t* c_one         = ckc_b_const_f32(b, 1.0);
    ckc_value_t* c_sq2_over_pi = ckc_b_const_f32(b, 0.7978845608028654);
    ckc_value_t* c_a           = ckc_b_const_f32(b, 0.044715);
    ckc_value_t* x2            = ckc_b_fmul(b, x, x);
    ckc_value_t* x3            = ckc_b_fmul(b, x2, x);
    ckc_value_t* inner = ckc_b_fmul(b, c_sq2_over_pi, ckc_b_fadd(b, x, ckc_b_fmul(b, c_a, x3)));
    /* Python evaluates the outer fmul's arguments left-to-right, so the
     * ``0.5 * x`` half is emitted BEFORE the tanh chain. C leaves the
     * argument evaluation order of ``ckc_b_fmul(b, fmul(c_half,x), fadd(...))``
     * unspecified (compilers commonly evaluate right-to-left, emitting the
     * tanh chain first). Sequence the sub-expressions into explicit locals so
     * the builder-call order matches Python byte-for-byte. */
    ckc_value_t* half_x        = ckc_b_fmul(b, c_half, x);
    ckc_value_t* one_plus_tanh = ckc_b_fadd(b, c_one, ckc_tanh_via_exp2(b, inner));
    return ckc_b_fmul(b, half_x, one_plus_tanh);
}

/* Returns the applied unary op, or NULL with a sticky ValueError set on `b`
 * for an unsupported op (mirrors the Python ``raise ValueError``). */
static ckc_value_t* ckc_ew_apply_unary(ckc_ir_builder_t* b, ckc_value_t* x, const char* op)
{
    if(ckc_ew_streq(op, "copy"))
    {
        return x;
    }
    if(ckc_ew_streq(op, "neg"))
    {
        return ckc_b_fneg(b, x);
    }
    if(ckc_ew_streq(op, "abs"))
    {
        return ckc_b_fmax(b, x, ckc_b_fneg(b, x));
    }
    if(ckc_ew_streq(op, "relu"))
    {
        return ckc_b_fmax(b, x, ckc_b_const_f32(b, 0.0));
    }
    if(ckc_ew_streq(op, "exp2"))
    {
        return ckc_b_exp2(b, x);
    }
    if(ckc_ew_streq(op, "tanh"))
    {
        return ckc_tanh_via_exp2(b, x);
    }
    if(ckc_ew_streq(op, "sigmoid"))
    {
        return ckc_sigmoid_via_exp2(b, x);
    }
    if(ckc_ew_streq(op, "silu") || ckc_ew_streq(op, "swish"))
    {
        return ckc_b_fmul(b, x, ckc_sigmoid_via_exp2(b, x));
    }
    if(ckc_ew_streq(op, "quick_gelu"))
    {
        ckc_value_t* c_1702 = ckc_b_const_f32(b, 1.702);
        return ckc_b_fmul(b, x, ckc_sigmoid_via_exp2(b, ckc_b_fmul(b, c_1702, x)));
    }
    if(ckc_ew_streq(op, "gelu_tanh"))
    {
        return ckc_ew_gelu_tanh(b, x);
    }
    {
        char msg[64];
        snprintf(msg, sizeof(msg), "unsupported unary op %s", op ? op : "(null)");
        ckc_i_set_err(b, CKC_ERR_VALUE, "%s", msg);
    }
    return NULL;
}

static ckc_value_t*
ckc_ew_apply_binary(ckc_ir_builder_t* b, ckc_value_t* a, ckc_value_t* c, const char* op)
{
    if(ckc_ew_streq(op, "add"))
    {
        return ckc_b_fadd(b, a, c);
    }
    if(ckc_ew_streq(op, "sub"))
    {
        return ckc_b_fsub(b, a, c);
    }
    if(ckc_ew_streq(op, "mul"))
    {
        return ckc_b_fmul(b, a, c);
    }
    if(ckc_ew_streq(op, "max"))
    {
        return ckc_b_fmax(b, a, c);
    }
    if(ckc_ew_streq(op, "min"))
    {
        return ckc_b_fmin(b, a, c);
    }
    if(ckc_ew_streq(op, "swiglu"))
    {
        return ckc_b_fmul(b, ckc_b_fmul(b, a, ckc_sigmoid_via_exp2(b, a)), c);
    }
    if(ckc_ew_streq(op, "geglu"))
    {
        return ckc_b_fmul(b, ckc_ew_gelu_tanh(b, a), c);
    }
    {
        char msg[64];
        snprintf(msg, sizeof(msg), "unsupported binary op %s", op ? op : "(null)");
        ckc_i_set_err(b, CKC_ERR_VALUE, "%s", msg);
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
static int ckc_ew_load_tile(ckc_ir_builder_t* b,
                            const ckc_tile_window_t* window,
                            const ckc_tile_distribution_t* dist,
                            ckc_value_t* tid,
                            int vec,
                            ckc_value_t** out_storage)
{
    ckc_value_t* ys[1];
    ckc_value_t* ps_row[1];
    ckc_value_t* const* ps[1];
    int ps_counts[1];
    ckc_value_t* x_coords[1];
    ckc_value_t* loaded;
    int k;

    /* ys = [b.const_i32(0)] (single Y access at y_base==(0,)). */
    ys[0] = ckc_b_const_i32(b, 0);
    /* ps = [[tid]] */
    ps_row[0]    = tid;
    ps[0]        = ps_row;
    ps_counts[0] = 1;

    if(!ckc_tile_distribution_calculate_x(b, dist, ys, 1, ps, ps_counts, 1, x_coords, 1))
    {
        return 0;
    }

    /* window.load_vec_as_f32(*x_coords, n=vec):
     *   v = load_vec(x_coords, n=vec)
     *   scalars[k] = cast_to_f32(vec_extract(v, k))
     * (vec is always >= 2 here, so the n==1 scalar branch never triggers). */
    loaded = ckc_tile_window_load_vec(b, window, x_coords, 1, vec);
    for(k = 0; k < vec; ++k)
    {
        out_storage[k] = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, loaded, k));
    }
    return ckc_ir_builder_ok(b) ? 1 : 0;
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
static void ckc_ew_store_tile(ckc_ir_builder_t* b,
                              const ckc_tile_window_t* window,
                              const ckc_tile_distribution_t* dist,
                              ckc_value_t* tid,
                              int vec,
                              ckc_value_t** storage)
{
    ckc_value_t* ys[1];
    ckc_value_t* ps_row[1];
    ckc_value_t* const* ps[1];
    int ps_counts[1];
    ckc_value_t* x_coords[1];
    ckc_value_t* casts[8];
    ckc_value_t* packed;
    const ckc_type_t* dtype;
    int k;

    ys[0]        = ckc_b_const_i32(b, 0);
    ps_row[0]    = tid;
    ps[0]        = ps_row;
    ps_counts[0] = 1;

    if(!ckc_tile_distribution_calculate_x(b, dist, ys, 1, ps, ps_counts, 1, x_coords, 1))
    {
        return;
    }

    dtype = ckc_tile_window_dtype(window);
    for(k = 0; k < vec; ++k)
    {
        casts[k] = ckc_b_cast_f32_to(b, storage[k], dtype);
    }
    packed = ckc_b_vec_pack(b, casts, vec, dtype);
    ckc_tile_window_store_vec(b, window, x_coords, 1, packed, vec);
}

/* ===================================================================== *
 *  build_elementwise
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_elementwise(ckc_ir_builder_t* b, const ckc_elementwise_spec_t* spec)
{
    const ckc_type_t* io_ty;
    bool is_binary;
    int tile_elems;

    ckc_value_t* A;
    ckc_value_t* Bp = NULL;
    ckc_value_t* C;
    ckc_value_t* N;

    ckc_tile_distribution_encoding_t* encoding;
    ckc_tile_distribution_t* distribution;

    ckc_tensor_view_t a_view, b_view, c_view;
    ckc_tile_window_t a_tile, b_tile, c_tile;

    ckc_value_t* tid;
    ckc_value_t* bid;
    ckc_value_t* c_vec;
    ckc_value_t* c_chunk;
    ckc_value_t* block_base;
    ckc_value_t* thread_base;
    ckc_value_t* fast_lim;
    ckc_value_t* in_fast;

    ckc_param_opts_t opts;
    int shape1[1];

    /* Encoding arrays. */
    int h_levels[2];
    ckc_h_row_t hs[1];
    int p_major[1];
    int p_minor[1];
    ckc_p_seq_t ps_seq[1];
    int ys_major[1];
    int ys_minor[1];

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }

    /* ok, why = is_valid_spec(spec); if not ok: raise ValueError(...) */
    {
        char reason[128];
        if(!ckc_elementwise_is_valid_spec(spec, reason, sizeof(reason)))
        {
            char msg[160];
            snprintf(msg, sizeof(msg), "invalid elementwise spec: %s", reason);
            ckc_i_set_err(b, CKC_ERR_VALUE, "%s", msg);
            return NULL;
        }
    }

    io_ty = ckc_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    is_binary  = ckc_elementwise_is_binary(spec);
    tile_elems = spec->block_size * spec->vec;

    /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* A = b.param("A", PtrType(io_ty,"global"), noalias=True, readonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias      = true;
    opts.noalias_set  = true;
    opts.readonly     = true;
    opts.readonly_set = true;
    opts.align        = 16;
    opts.align_set    = true;
    opts.addr_space   = NULL; /* PtrType space "global" handled by ckc_ptr_type below */
    A                 = ckc_b_param(b, "A", ckc_ptr_type(b, io_ty, "global"), &opts);

    if(is_binary)
    {
        memset(&opts, 0, sizeof(opts));
        opts.noalias      = true;
        opts.noalias_set  = true;
        opts.readonly     = true;
        opts.readonly_set = true;
        opts.align        = 16;
        opts.align_set    = true;
        Bp                = ckc_b_param(b, "B", ckc_ptr_type(b, io_ty, "global"), &opts);
    }

    /* C = b.param("C", PtrType(io_ty,"global"), noalias=True, writeonly=True, align=16) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias       = true;
    opts.noalias_set   = true;
    opts.writeonly     = true;
    opts.writeonly_set = true;
    opts.align         = 16;
    opts.align_set     = true;
    C                  = ckc_b_param(b, "C", ckc_ptr_type(b, io_ty, "global"), &opts);

    /* N = b.param("N", I32) */
    N = ckc_b_param(b, "N", ckc_i32(), NULL);

    /* TileDistributionEncoding(
     *     Hs=((block_size, vec),),
     *     Ps2RHs_major=((1,),), Ps2RHs_minor=((0,),),
     *     Ys2RHs_major=(1,), Ys2RHs_minor=(1,)) */
    h_levels[0]     = spec->block_size;
    h_levels[1]     = spec->vec;
    hs[0].levels    = h_levels;
    hs[0].count     = 2;
    p_major[0]      = 1;
    p_minor[0]      = 0;
    ps_seq[0].major = p_major;
    ps_seq[0].minor = p_minor;
    ps_seq[0].count = 1;
    ys_major[0]     = 1;
    ys_minor[0]     = 1;

    encoding = ckc_make_tile_distribution_encoding(b,
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
    distribution = ckc_make_static_tile_distribution(b, encoding);
    if(distribution == NULL)
    {
        return NULL;
    }

    /* 1D views over the contiguous buffer (packed strides => stride 1). */
    shape1[0] = tile_elems;
    if(ckc_make_global_view(&a_view, A, shape1, 1, io_ty, NULL) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "make_global_view(A) failed");
        return NULL;
    }
    if(ckc_make_global_view(&c_view, C, shape1, 1, io_ty, NULL) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "make_global_view(C) failed");
        return NULL;
    }
    if(is_binary)
    {
        if(ckc_make_global_view(&b_view, Bp, shape1, 1, io_ty, NULL) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "make_global_view(B) failed");
            return NULL;
        }
    }

    tid     = ckc_b_thread_id_x(b);
    bid     = ckc_b_block_id_x(b);
    c_vec   = ckc_b_const_i32(b, spec->vec);
    c_chunk = ckc_b_const_i32(b, tile_elems);

    block_base  = ckc_b_mul(b, bid, c_chunk);
    thread_base = ckc_b_add(b, block_base, ckc_b_mul(b, tid, c_vec));

    fast_lim = ckc_b_add(b, thread_base, c_vec);
    in_fast  = ckc_b_cmp_le(b, fast_lim, N);

    /* Per-block tile windows anchored at this CTA's slab origin (origin =
     * (block_base,)). */
    {
        ckc_value_t* origin[1];
        int lengths1[1];
        origin[0]   = block_base;
        lengths1[0] = tile_elems;
        if(ckc_make_tile_window(&a_tile, &a_view, lengths1, origin, 1) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "make_tile_window(A) failed");
            return NULL;
        }
        if(ckc_make_tile_window(&c_tile, &c_view, lengths1, origin, 1) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "make_tile_window(C) failed");
            return NULL;
        }
        if(is_binary)
        {
            if(ckc_make_tile_window(&b_tile, &b_view, lengths1, origin, 1) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "%s", "make_tile_window(B) failed");
                return NULL;
            }
        }
    }

    /* with b.scf_if(in_fast): emit_vec_path() */
    {
        ckc_if_t gate = ckc_b_scf_if(b, in_fast);
        ckc_b_region_enter(b, gate.then_region);
        {
            /* a_dt = a_tile.load(distribution, ps=[[tid]]) */
            ckc_value_t* a_dt[8];
            ckc_value_t* out_dt[8];
            int y;
            if(!ckc_ew_load_tile(b, &a_tile, distribution, tid, spec->vec, a_dt))
            {
                ckc_b_region_leave(b);
                return NULL;
            }
            if(is_binary)
            {
                ckc_value_t* b_dt[8];
                if(!ckc_ew_load_tile(b, &b_tile, distribution, tid, spec->vec, b_dt))
                {
                    ckc_b_region_leave(b);
                    return NULL;
                }
                for(y = 0; y < spec->vec; ++y)
                {
                    out_dt[y] = ckc_ew_apply_binary(b, a_dt[y], b_dt[y], spec->op);
                }
            }
            else
            {
                for(y = 0; y < spec->vec; ++y)
                {
                    out_dt[y] = ckc_ew_apply_unary(b, a_dt[y], spec->op);
                }
            }
            /* c_tile.store(out_dt, ps=[[tid]]) */
            ckc_ew_store_tile(b, &c_tile, distribution, tid, spec->vec, out_dt);
        }
        ckc_b_region_leave(b);
    }

    /* with b.scf_if(b.lnot(in_fast)): emit_scalar_path() */
    {
        ckc_value_t* not_fast = ckc_b_lnot(b, in_fast);
        ckc_if_t gate         = ckc_b_scf_if(b, not_fast);
        ckc_b_region_enter(b, gate.then_region);
        {
            int i;
            for(i = 0; i < spec->vec; ++i)
            {
                /* idx = thread_base + const_i32(i) */
                ckc_value_t* idx       = ckc_b_add(b, thread_base, ckc_b_const_i32(b, i));
                ckc_value_t* in_bounds = ckc_b_cmp_lt(b, idx, N);
                ckc_if_t ib            = ckc_b_scf_if(b, in_bounds);
                ckc_b_region_enter(b, ib.then_region);
                {
                    ckc_value_t* indices[1];
                    ckc_value_t* a_s;
                    ckc_value_t* r;
                    indices[0] = idx;
                    /* a = cast_to_f32(a_view.load_scalar([idx])) */
                    a_s = ckc_b_cast_to_f32(b, ckc_tensor_view_load_scalar(b, &a_view, indices, 1));
                    if(is_binary)
                    {
                        ckc_value_t* bv = ckc_b_cast_to_f32(
                            b, ckc_tensor_view_load_scalar(b, &b_view, indices, 1));
                        r = ckc_ew_apply_binary(b, a_s, bv, spec->op);
                    }
                    else
                    {
                        r = ckc_ew_apply_unary(b, a_s, spec->op);
                    }
                    /* c_view.store_scalar([idx], cast_f32_to(r, io_ty)) */
                    ckc_tensor_view_store_scalar(
                        b, &c_view, indices, 1, ckc_b_cast_f32_to(b, r, io_ty), 0);
                }
                ckc_b_region_leave(b);
            }
        }
        ckc_b_region_leave(b);
    }

    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

ckc_kernel_def_t* ckc_build_elementwise_new(ckc_ir_builder_t* b, const ckc_elementwise_spec_t* spec)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_elementwise_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_elementwise(b, spec);
    });
}

/* ===================================================================== *
 *  elementwise_grid
 * ===================================================================== */

void ckc_elementwise_grid(int numel, const ckc_elementwise_spec_t* spec, int out_grid[3])
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
    chunk = ckc_elementwise_elems_per_block(spec);
    if(chunk <= 0)
    {
        return;
    }
    out_grid[0] = (numel + chunk - 1) / chunk;
}

/* ===================================================================== *
 *  ckc_elementwise_lower_to_llvm -- build + lower to .ll convenience.
 * ===================================================================== */

ckc_status_t ckc_elementwise_lower_to_llvm(const ckc_elementwise_spec_t* spec,
                                           const char* arch,
                                           ckc_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            const char* m = "lower_to_llvm: null spec/out";
            size_t n      = strlen(m);
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_elementwise_new(&b, spec);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = ckc_ir_builder_error(&b);
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
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
