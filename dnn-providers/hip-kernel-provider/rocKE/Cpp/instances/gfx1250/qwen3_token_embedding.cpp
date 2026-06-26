// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_qwen3_token_embedding.c -- C99 port of
 * rocke/instances/gfx1250/qwen3_token_embedding.py.
 *
 * Token-embedding row gather: out[t, :] = table[input_ids[t], :]. Pure
 * vectorised copy, arch-neutral. The build op order tracks
 * build_qwen3_token_embedding() top-to-bottom so a reviewer can diff line by
 * line; the emitted IR is byte-identical to the Python lowerer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_qwen3_token_embedding.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_QTE_DEFAULT_HIDDEN 2048
#define ROCKE_QTE_DEFAULT_DTYPE "bf16"
#define ROCKE_QTE_DEFAULT_VEC 8
#define ROCKE_QTE_DEFAULT_BLOCK 256
#define ROCKE_QTE_DEFAULT_NAME "rocke_gfx1250_qwen3_token_embedding"

/* _dtype_ir: fp16/f16 -> f16, bf16 -> bf16. NULL on unsupported. */
static const rocke_type_t* rocke_qte_dtype_ir(const char* dtype)
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

/* ===================================================================== *
 *  Spec accessors
 * ===================================================================== */

rocke_qwen3_token_embedding_gfx1250_spec_t rocke_qwen3_token_embedding_gfx1250_spec_default(void)
{
    rocke_qwen3_token_embedding_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.hidden = ROCKE_QTE_DEFAULT_HIDDEN;
    s.dtype = ROCKE_QTE_DEFAULT_DTYPE;
    s.vec = ROCKE_QTE_DEFAULT_VEC;
    s.block_size = ROCKE_QTE_DEFAULT_BLOCK;
    s.name = ROCKE_QTE_DEFAULT_NAME;
    return s;
}

/* Qwen3TokenEmbeddingSpec.kernel_name():
 *   kernel_name_join(self.name, f"h{hidden}", dtype, f"v{vec}"). */
rocke_status_t rocke_qwen3_token_embedding_gfx1250_kernel_name(
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char h_part[32];
    char v_part[32];
    const char* parts[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(h_part, sizeof(h_part), "h%d", spec->hidden);
    snprintf(v_part, sizeof(v_part), "v%d", spec->vec);
    parts[0] = h_part;
    parts[1] = spec->dtype;
    parts[2] = v_part;

    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

bool rocke_qwen3_token_embedding_gfx1250_is_valid_spec(
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec,
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
    /* vec in {1,2,4,8}. */
    if(spec->vec != 1 && spec->vec != 2 && spec->vec != 4 && spec->vec != 8)
    {
        rocke_spec_set_reason(reason, reason_cap, "vec must be 1/2/4/8");
        return false;
    }
    /* hidden > 0 and a multiple of vec. */
    if(spec->hidden <= 0 || (spec->hidden % spec->vec) != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "hidden must be a positive multiple of vec");
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
 *  build_qwen3_token_embedding(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_qwen3_token_embedding_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_token_embedding_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* dt;
        int H;
        int vec;
        int vpr;
        int bs;
        int align;
        rocke_value_t* input_ids;
        rocke_value_t* table;
        rocke_value_t* out;
        rocke_value_t* num_tokens;
        rocke_value_t* c_vpr;
        rocke_value_t* c_vec;
        rocke_value_t* c_H;
        rocke_value_t* tid;
        rocke_value_t* total;
        rocke_if_t iff;
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        /* ok, reason = _require_supported(arch); if not ok: raise. (The full
         * __post_init__ validity gate also runs here.) */
        if(!rocke_qwen3_token_embedding_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "%s", reason);
            return NULL;
        }

        dt = rocke_qte_dtype_ir(spec->dtype);
        H = spec->hidden;
        vec = spec->vec;
        vpr = H / vec;
        bs = spec->block_size;
        align = vec * 2; /* bf16/fp16 = 2 bytes */

        /* b.kernel.attrs["max_workgroup_size"] = bs */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", bs);

        /* ---- kernel params -- */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
            const rocke_type_t* ptr_dt = rocke_ptr_type(b, dt, "global");

            /* input_ids = b.param("input_ids", PtrType(I32,"global"),
             *                     noalias, readonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            input_ids = rocke_b_param(b, "input_ids", ptr_i32, &opts);

            /* table = b.param("table", PtrType(dt,"global"),
             *                 noalias, readonly, align16) */
            table = rocke_b_param(b, "table", ptr_dt, &opts);

            /* out = b.param("out", PtrType(dt,"global"),
             *               noalias, writeonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            out = rocke_b_param(b, "out", ptr_dt, &opts);

            /* num_tokens = b.param("num_tokens", I32) */
            num_tokens = rocke_b_param(b, "num_tokens", rocke_i32(), NULL);
        }

        /* c_vpr = b.const_i32(vpr); c_vec = b.const_i32(vec); c_H = b.const_i32(H)
         * Python pre-allocates exactly 3 consts; the `const_i32(bs)` in the tid
         * expr is inlined at use-site. Mirror that exactly to preserve SSA numbering. */
        c_vpr = rocke_b_const_i32(b, vpr);
        c_vec = rocke_b_const_i32(b, vec);
        c_H = rocke_b_const_i32(b, H);

        /* tid = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), b.thread_id_x())
         * Python evals args left-to-right: block_id → const(bs) → mul → thread_id → add.
         * C arg-eval order is unspecified, so sequence the ops explicitly to match. */
        {
            rocke_value_t* bid = rocke_b_block_id_x(b);
            rocke_value_t* c_bs_inline = rocke_b_const_i32(b, bs);
            rocke_value_t* bid_mul = rocke_b_mul(b, bid, c_bs_inline);
            rocke_value_t* t = rocke_b_thread_id_x(b);
            tid = rocke_b_add(b, bid_mul, t);
        }
        /* total = b.mul(num_tokens, c_vpr) */
        total = rocke_b_mul(b, num_tokens, c_vpr);

        /* with b.scf_if(b.cmp_lt(tid, total)): */
        iff = rocke_b_scf_if(b, rocke_b_cmp_lt(b, tid, total));
        rocke_b_region_enter(b, iff.then_region);
        {
            /* token = b.div(tid, c_vpr) */
            rocke_value_t* token = rocke_b_div(b, tid, c_vpr);
            /* vcol = b.mod(tid, c_vpr) */
            rocke_value_t* vcol = rocke_b_mod(b, tid, c_vpr);
            /* col = b.mul(vcol, c_vec) */
            rocke_value_t* col = rocke_b_mul(b, vcol, c_vec);
            /* tok_id = b.global_load_i32(input_ids, token) */
            rocke_value_t* tok_id = rocke_b_global_load_i32(b, input_ids, token, /*align=*/-1);
            /* src = b.add(b.mul(tok_id, c_H), col) */
            rocke_value_t* src = rocke_b_add(b, rocke_b_mul(b, tok_id, c_H), col);
            /* dst = b.add(b.mul(token, c_H), col) */
            rocke_value_t* dst = rocke_b_add(b, rocke_b_mul(b, token, c_H), col);
            /* v = b.global_load_vN(table, src, dt, vec, align=align) */
            rocke_value_t* v = rocke_b_global_load_vN(b, table, src, dt, vec, align);
            /* b.global_store_vN(out, dst, v, vec, align=align) */
            rocke_b_global_store_vN(b, out, dst, v, vec, align);
        }
        rocke_b_region_leave(b);

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
rocke_kernel_def_t* rocke_build_qwen3_token_embedding_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_token_embedding_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_qwen3_token_embedding_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_qwen3_token_embedding_gfx1250(b, spec, arch);
    });
}

/* ===================================================================== *
 *  qwen3_token_embedding_grid(num_tokens, spec)
 * ===================================================================== */
rocke_status_t rocke_qwen3_token_embedding_gfx1250_grid(
    int num_tokens, const rocke_qwen3_token_embedding_gfx1250_spec_t* spec, int out[3])
{
    int vpr;
    int total;
    if(out == NULL || spec == NULL || spec->vec == 0 || spec->block_size == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    vpr = spec->hidden / spec->vec;
    total = num_tokens * vpr;
    out[0] = (total + spec->block_size - 1) / spec->block_size;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  lower_to_llvm convenience.
 * ===================================================================== */
rocke_status_t rocke_qwen3_token_embedding_gfx1250_lower_to_llvm(
    const rocke_qwen3_token_embedding_gfx1250_spec_t* spec,
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

    kernel = rocke_build_qwen3_token_embedding_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_qwen3_token_embedding_gfx1250 failed";
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
