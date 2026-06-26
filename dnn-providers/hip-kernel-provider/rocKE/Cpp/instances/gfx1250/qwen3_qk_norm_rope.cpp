// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_qwen3_qk_norm_rope.c -- C99 port of
 * rocke/instances/gfx1250/qwen3_qk_norm_rope.py.
 *
 * Fused per-head RMSNorm + RoPE for one Q-or-K tensor [tokens, num_heads,
 * head_dim]. One thread per (token, head); head_dim unrolled. The build op order
 * tracks build_qwen3_qk_norm_rope() top-to-bottom so a reviewer can diff line by
 * line; emitted IR is byte-identical to the Python lowerer.
 *
 * NOTE on SSA byte-identity: Python evaluates call args left-to-right and never
 * dedupes const_i32; C arg-eval order is unspecified. Every place Python nests
 * two side-effecting calls in one expression, this port sequences them into
 * temporaries in Python's eval order.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_qwen3_qk_norm_rope.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_QKR_DEFAULT_HEAD_DIM 64
#define ROCKE_QKR_DEFAULT_DTYPE "bf16"
#define ROCKE_QKR_DEFAULT_EPS 1e-6
#define ROCKE_QKR_DEFAULT_ROPE "half"
#define ROCKE_QKR_DEFAULT_BLOCK 64
#define ROCKE_QKR_DEFAULT_NAME "rocke_gfx1250_qwen3_qk_norm_rope"

/* _dtype_ir: fp16/f16 -> f16, bf16 -> bf16. NULL on unsupported. */
static const rocke_type_t* rocke_qkr_dtype_ir(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
    {
        return rocke_f16();
    }
    if(strcmp(dtype, "bf16") == 0)
    {
        return rocke_bf16();
    }
    return NULL;
}

/* spec._pair(i): interleaved -> (2i, 2i+1); half -> (i, i + head_dim/2). */
static void
    rocke_qkr_pair(const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, int i, int* lo, int* hi)
{
    if(strcmp(spec->rope_layout, "interleaved") == 0)
    {
        *lo = 2 * i;
        *hi = 2 * i + 1;
    }
    else
    {
        *lo = i;
        *hi = i + spec->head_dim / 2;
    }
}

/* ===================================================================== *
 *  Spec accessors
 * ===================================================================== */

rocke_qwen3_qk_norm_rope_gfx1250_spec_t rocke_qwen3_qk_norm_rope_gfx1250_spec_default(void)
{
    rocke_qwen3_qk_norm_rope_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.num_heads = 0; /* no Python default; caller must set */
    s.head_dim = ROCKE_QKR_DEFAULT_HEAD_DIM;
    s.dtype = ROCKE_QKR_DEFAULT_DTYPE;
    s.eps = ROCKE_QKR_DEFAULT_EPS;
    s.rope_layout = ROCKE_QKR_DEFAULT_ROPE;
    s.block_size = ROCKE_QKR_DEFAULT_BLOCK;
    s.name = ROCKE_QKR_DEFAULT_NAME;
    return s;
}

/* Qwen3QkNormRopeSpec.kernel_name():
 *   kernel_name_join(self.name, f"h{nh}", f"d{hd}", dtype, rope_layout). */
rocke_status_t rocke_qwen3_qk_norm_rope_gfx1250_kernel_name(
    const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char h_part[32];
    char d_part[32];
    const char* parts[4];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(h_part, sizeof(h_part), "h%d", spec->num_heads);
    snprintf(d_part, sizeof(d_part), "d%d", spec->head_dim);
    parts[0] = h_part;
    parts[1] = d_part;
    parts[2] = spec->dtype;
    parts[3] = spec->rope_layout;

    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

bool rocke_qwen3_qk_norm_rope_gfx1250_is_valid_spec(
    const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    char buf[ROCKE_ERR_MSG_CAP];

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }

    /* __post_init__: dtype must be fp16/bf16. */
    if(spec->dtype == NULL
       || (strcmp(spec->dtype, "fp16") != 0 && strcmp(spec->dtype, "f16") != 0
           && strcmp(spec->dtype, "bf16") != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "dtype must be fp16/bf16");
        return false;
    }
    /* head_dim positive and even. */
    if(spec->head_dim <= 0 || (spec->head_dim % 2) != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "head_dim must be positive and even");
        return false;
    }
    /* num_heads positive. */
    if(spec->num_heads <= 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "num_heads must be positive");
        return false;
    }
    /* rope_layout half/interleaved. */
    if(spec->rope_layout == NULL
       || (strcmp(spec->rope_layout, "half") != 0 && strcmp(spec->rope_layout, "interleaved") != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "rope_layout must be 'half' or 'interleaved'");
        return false;
    }

    /* _require_supported: ArchTarget.from_gfx(arch) must resolve. */
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target '%s'", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }

    rocke_spec_set_reason(reason, reason_cap, "supported");
    return true;
}

/* ===================================================================== *
 *  build_qwen3_qk_norm_rope(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_qwen3_qk_norm_rope_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* dt;
        const rocke_type_t* f32;
        int H;
        int half;
        int nh;
        int bs;
        int d;
        int i;
        rocke_value_t* x_in;
        rocke_value_t* weight;
        rocke_value_t* cos;
        rocke_value_t* sin;
        rocke_value_t* positions;
        rocke_value_t* x_out;
        rocke_value_t* num_tokens;
        rocke_value_t* c_nh;
        rocke_value_t* c_H;
        rocke_value_t* inv_H;
        rocke_value_t* c_eps;
        rocke_value_t* c_half;
        rocke_value_t* tid;
        rocke_value_t* total;
        rocke_value_t** xs;
        rocke_value_t** xn;
        rocke_value_t** out;
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        if(!rocke_qwen3_qk_norm_rope_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "%s", reason);
            return NULL;
        }

        dt = rocke_qkr_dtype_ir(spec->dtype);
        f32 = rocke_f32();
        H = spec->head_dim;
        half = H / 2;
        nh = spec->num_heads;
        bs = spec->block_size;

        /* b.kernel.attrs["max_workgroup_size"] = bs */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", bs);

        /* ---- kernel params -- */
        {
            rocke_param_opts_t ro; /* noalias readonly align16 */
            rocke_param_opts_t wo; /* noalias writeonly align16 */
            const rocke_type_t* ptr_dt = rocke_ptr_type(b, dt, "global");
            const rocke_type_t* ptr_f32 = rocke_ptr_type(b, f32, "global");
            const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");

            memset(&ro, 0, sizeof(ro));
            ro.noalias = true;
            ro.noalias_set = true;
            ro.readonly = true;
            ro.readonly_set = true;
            ro.align = 16;
            ro.align_set = true;

            memset(&wo, 0, sizeof(wo));
            wo.noalias = true;
            wo.noalias_set = true;
            wo.writeonly = true;
            wo.writeonly_set = true;
            wo.align = 16;
            wo.align_set = true;

            x_in = rocke_b_param(b, "x_in", ptr_dt, &ro);
            weight = rocke_b_param(b, "weight", ptr_f32, &ro);
            cos = rocke_b_param(b, "cos", ptr_f32, &ro);
            sin = rocke_b_param(b, "sin", ptr_f32, &ro);
            positions = rocke_b_param(b, "positions", ptr_i32, &ro);
            x_out = rocke_b_param(b, "x_out", ptr_dt, &wo);
            num_tokens = rocke_b_param(b, "num_tokens", rocke_i32(), NULL);
        }

        /* c_nh = b.const_i32(nh); c_H = b.const_i32(H); inv_H = b.const_f32(1.0/H);
         * c_eps = b.const_f32(eps); c_half = b.const_i32(half) */
        c_nh = rocke_b_const_i32(b, nh);
        c_H = rocke_b_const_i32(b, H);
        inv_H = rocke_b_const_f32(b, 1.0 / (double)H);
        c_eps = rocke_b_const_f32(b, spec->eps);
        c_half = rocke_b_const_i32(b, half);

        /* tid = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), b.thread_id_x()) */
        {
            rocke_value_t* bid = rocke_b_block_id_x(b);
            rocke_value_t* c_bs_inline = rocke_b_const_i32(b, bs);
            rocke_value_t* bid_mul = rocke_b_mul(b, bid, c_bs_inline);
            rocke_value_t* t = rocke_b_thread_id_x(b);
            tid = rocke_b_add(b, bid_mul, t);
        }
        /* total = b.mul(num_tokens, c_nh) */
        total = rocke_b_mul(b, num_tokens, c_nh);

        xs = (rocke_value_t**)calloc((size_t)H, sizeof(rocke_value_t*));
        xn = (rocke_value_t**)calloc((size_t)H, sizeof(rocke_value_t*));
        out = (rocke_value_t**)calloc((size_t)H, sizeof(rocke_value_t*));
        if(xs == NULL || xn == NULL || out == NULL)
        {
            free(xs);
            free(xn);
            free(out);
            (void)rocke_i_set_err(b, ROCKE_ERR_VALUE, "out of memory");
            return NULL;
        }

        /* with b.scf_if(b.cmp_lt(tid, total)): */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_lt(b, tid, total));
            rocke_b_region_enter(b, iff.then_region);
            {
                rocke_value_t* token;
                rocke_value_t* row_base;
                rocke_value_t* pos;
                rocke_value_t* ss;
                rocke_value_t* inv;
                rocke_value_t* trig_base;

                /* token = b.div(tid, c_nh) */
                token = rocke_b_div(b, tid, c_nh);
                /* row_base = b.mul(tid, c_H) */
                row_base = rocke_b_mul(b, tid, c_H);
                /* pos = b.global_load_i32(positions, token) */
                pos = rocke_b_global_load_i32(b, positions, token, /*align=*/-1);

                /* ss = b.const_f32(0.0); for d: xs[d]=cast_to_f32(load(...)); ss+=xd*xd */
                ss = rocke_b_const_f32(b, 0.0);
                for(d = 0; d < H; ++d)
                {
                    rocke_value_t* off = rocke_b_add(b, row_base, rocke_b_const_i32(b, d));
                    rocke_value_t* xd = rocke_b_cast_to_f32(
                        b, rocke_b_global_load(b, x_in, off, dt, /*align=*/2));
                    xs[d] = xd;
                    ss = rocke_b_fadd(b, ss, rocke_b_fmul(b, xd, xd));
                }
                /* inv = b.rsqrt(b.fadd(b.fmul(ss, inv_H), c_eps)) */
                inv = rocke_b_rsqrt(b, rocke_b_fadd(b, rocke_b_fmul(b, ss, inv_H), c_eps));

                /* for d: wd=load(weight,const(d)); xn[d]=fmul(fmul(xs[d],inv),wd) */
                for(d = 0; d < H; ++d)
                {
                    rocke_value_t* wd
                        = rocke_b_global_load(b, weight, rocke_b_const_i32(b, d), f32, /*align=*/4);
                    xn[d] = rocke_b_fmul(b, rocke_b_fmul(b, xs[d], inv), wd);
                }

                /* trig_base = b.mul(pos, c_half) */
                trig_base = rocke_b_mul(b, pos, c_half);
                for(i = 0; i < half; ++i)
                {
                    int lo_i;
                    int hi_i;
                    rocke_value_t* c;
                    rocke_value_t* s;
                    rocke_value_t* lo;
                    rocke_value_t* hi;
                    rocke_value_t* lo_c;
                    rocke_value_t* hi_s;
                    rocke_value_t* lo_s;
                    rocke_value_t* hi_c;

                    rocke_qkr_pair(spec, i, &lo_i, &hi_i);
                    /* c = b.global_load(cos, b.add(trig_base, b.const_i32(i)), F32, align=4) */
                    c = rocke_b_global_load(
                        b, cos, rocke_b_add(b, trig_base, rocke_b_const_i32(b, i)), f32, 4);
                    /* s = b.global_load(sin, b.add(trig_base, b.const_i32(i)), F32, align=4) */
                    s = rocke_b_global_load(
                        b, sin, rocke_b_add(b, trig_base, rocke_b_const_i32(b, i)), f32, 4);
                    lo = xn[lo_i];
                    hi = xn[hi_i];
                    /* out[lo_i] = b.fsub(b.fmul(lo, c), b.fmul(hi, s))
                     * out[hi_i] = b.fadd(b.fmul(lo, s), b.fmul(hi, c))
                     * Both have two side-effecting fmul args; sequence L-to-R. */
                    lo_c = rocke_b_fmul(b, lo, c);
                    hi_s = rocke_b_fmul(b, hi, s);
                    out[lo_i] = rocke_b_fsub(b, lo_c, hi_s);
                    lo_s = rocke_b_fmul(b, lo, s);
                    hi_c = rocke_b_fmul(b, hi, c);
                    out[hi_i] = rocke_b_fadd(b, lo_s, hi_c);
                }

                /* for d: b.global_store(x_out, b.add(row_base, const(d)),
                 *                       b.cast_f32_to(out[d], dt), align=2)
                 * idx (add) and value (cast) both side-effecting; sequence idx first. */
                for(d = 0; d < H; ++d)
                {
                    rocke_value_t* soff = rocke_b_add(b, row_base, rocke_b_const_i32(b, d));
                    rocke_value_t* sval = rocke_b_cast_f32_to(b, out[d], dt);
                    rocke_b_global_store(b, x_out, soff, sval, /*align=*/2);
                }
            }
            rocke_b_region_leave(b);
        }

        free(xs);
        free(xn);
        free(out);

        /* return b.kernel */
        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

/* ===================================================================== *
 *  _new -- init builder with spec.kernel_name() then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_qwen3_qk_norm_rope_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_qwen3_qk_norm_rope_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_qwen3_qk_norm_rope_gfx1250(b, spec, arch);
    });
}

/* ===================================================================== *
 *  qwen3_qk_norm_rope_grid(num_tokens, spec)
 * ===================================================================== */
rocke_status_t rocke_qwen3_qk_norm_rope_gfx1250_grid(
    int num_tokens, const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec, int out[3])
{
    int total;
    if(out == NULL || spec == NULL || spec->block_size == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    total = num_tokens * spec->num_heads;
    out[0] = (total + spec->block_size - 1) / spec->block_size;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  lower_to_llvm convenience.
 * ===================================================================== */
rocke_status_t rocke_qwen3_qk_norm_rope_gfx1250_lower_to_llvm(
    const rocke_qwen3_qk_norm_rope_gfx1250_spec_t* spec,
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
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }

    kernel = rocke_build_qwen3_qk_norm_rope_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_qwen3_qk_norm_rope_gfx1250 failed";
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
