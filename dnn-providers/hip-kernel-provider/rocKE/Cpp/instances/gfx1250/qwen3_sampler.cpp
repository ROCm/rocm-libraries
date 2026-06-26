// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_qwen3_sampler.c -- C99 port of
 * rocke/instances/gfx1250/qwen3_sampler.py.
 *
 * Greedy (argmax) token sampler: out[t] = argmax_v logits[t, v] with
 * deterministic lowest-index tie-break. One workgroup per token row; each thread
 * scans a strided vocab slice, then an LDS index-reduction collapses to the row
 * argmax and lane 0 writes the id. The build op order tracks
 * build_qwen3_greedy_sampler() top-to-bottom so a reviewer can diff line by line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_qwen3_sampler.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_QSAMP_DEFAULT_DTYPE "f32"
#define ROCKE_QSAMP_DEFAULT_BLOCK 256
#define ROCKE_QSAMP_DEFAULT_NAME "rocke_gfx1250_qwen3_greedy_sampler"

/* _dtype_ir: f32/fp32 -> f32, bf16 -> bf16, fp16/f16 -> f16. NULL on unsupported. */
static const rocke_type_t* rocke_qsamp_dtype_ir(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "f32") == 0 || strcmp(dtype, "fp32") == 0)
    {
        return rocke_f32();
    }
    if(strcmp(dtype, "bf16") == 0)
    {
        return rocke_bf16();
    }
    if(strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
    {
        return rocke_f16();
    }
    return NULL;
}

/* ===================================================================== *
 *  Spec accessors
 * ===================================================================== */

rocke_qwen3_sampler_gfx1250_spec_t rocke_qwen3_sampler_gfx1250_spec_default(void)
{
    rocke_qwen3_sampler_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.logits_dtype = ROCKE_QSAMP_DEFAULT_DTYPE;
    s.block_size = ROCKE_QSAMP_DEFAULT_BLOCK;
    s.name = ROCKE_QSAMP_DEFAULT_NAME;
    return s;
}

/* Qwen3GreedySamplerSpec.kernel_name():
 *   kernel_name_join(self.name, logits_dtype, f"bs{block_size}"). */
rocke_status_t rocke_qwen3_sampler_gfx1250_kernel_name(
    const rocke_qwen3_sampler_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char bs_part[32];
    const char* parts[2];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(bs_part, sizeof(bs_part), "bs%d", spec->block_size);
    parts[0] = spec->logits_dtype;
    parts[1] = bs_part;

    return rocke_kernel_name_join(spec->name, parts, 2, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec(spec, arch)
 * ===================================================================== */

bool rocke_qwen3_sampler_gfx1250_is_valid_spec(const rocke_qwen3_sampler_gfx1250_spec_t* spec,
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

    /* __post_init__: logits_dtype must be f32/fp32/bf16/fp16/f16. */
    if(spec->logits_dtype == NULL
       || (strcmp(spec->logits_dtype, "f32") != 0 && strcmp(spec->logits_dtype, "fp32") != 0
           && strcmp(spec->logits_dtype, "bf16") != 0 && strcmp(spec->logits_dtype, "fp16") != 0
           && strcmp(spec->logits_dtype, "f16") != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "logits_dtype must be f32/bf16/fp16");
        return false;
    }
    /* block_size must be a power of two. */
    if(spec->block_size <= 0 || (spec->block_size & (spec->block_size - 1)) != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "block_size must be a power of two");
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
 *  build_qwen3_greedy_sampler(spec, arch)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_qwen3_sampler_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_sampler_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* dt;
        int bs;
        rocke_value_t* logits;
        rocke_value_t* out_ids;
        rocke_value_t* vocab;
        rocke_value_t* lds_val;
        rocke_value_t* lds_idx;
        rocke_value_t* token;
        rocke_value_t* tid;
        rocke_value_t* c_bs;
        rocke_value_t* row_base;
        rocke_value_t* neg_inf;
        rocke_value_t* local_m;
        rocke_value_t* local_i;
        rocke_value_t* red_val = NULL; /* "_" in Python; helper requires non-NULL out_val */
        rocke_value_t* arg = NULL;
        rocke_for_t loop;
        rocke_iter_arg_t iter_args[2];
        int shape[1];
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx1250";
        }

        /* ok, reason = _require_supported(arch); if not ok: raise. */
        if(!rocke_qwen3_sampler_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "%s", reason);
            return NULL;
        }

        dt = rocke_qsamp_dtype_ir(spec->logits_dtype);
        bs = spec->block_size;

        /* b.kernel.attrs["max_workgroup_size"] = bs */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", bs);

        /* ---- kernel params -- */
        {
            rocke_param_opts_t opts;
            const rocke_type_t* ptr_dt = rocke_ptr_type(b, dt, "global");
            const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");

            /* logits = b.param("logits", PtrType(dt,"global"),
             *                  noalias, readonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.readonly = true;
            opts.readonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            logits = rocke_b_param(b, "logits", ptr_dt, &opts);

            /* out_ids = b.param("out_ids", PtrType(I32,"global"),
             *                   noalias, writeonly, align16) */
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            out_ids = rocke_b_param(b, "out_ids", ptr_i32, &opts);

            /* vocab = b.param("vocab", I32) */
            vocab = rocke_b_param(b, "vocab", rocke_i32(), NULL);
        }

        /* lds_val = b.smem_alloc(F32, [bs], name_hint="samp_val")
         * lds_idx = b.smem_alloc(F32, [bs], name_hint="samp_idx") */
        shape[0] = bs;
        lds_val = rocke_b_smem_alloc(b, rocke_f32(), shape, 1, "samp_val");
        lds_idx = rocke_b_smem_alloc(b, rocke_f32(), shape, 1, "samp_idx");

        /* token = b.block_id_x(); tid = b.thread_id_x(); c_bs = b.const_i32(bs) */
        token = rocke_b_block_id_x(b);
        tid = rocke_b_thread_id_x(b);
        c_bs = rocke_b_const_i32(b, bs);
        /* row_base = b.mul(token, vocab) */
        row_base = rocke_b_mul(b, token, vocab);
        /* neg_inf = b.const_f32(-3.0e38) */
        neg_inf = rocke_b_const_f32(b, -3.0e38);

        /* for_op = b.scf_for_iter(tid, vocab, c_bs,
         *              [("m", neg_inf), ("i", b.const_i32(0))], iv_name="v") */
        iter_args[0].name = "m";
        iter_args[0].init = neg_inf;
        iter_args[1].name = "i";
        iter_args[1].init = rocke_b_const_i32(b, 0);
        loop = rocke_b_scf_for_iter(b,
                                    tid,
                                    vocab,
                                    c_bs,
                                    iter_args,
                                    2,
                                    "v",
                                    /*unroll=*/false,
                                    /*elide_trailing_barrier=*/true);

        rocke_b_region_enter(b, loop.body);
        {
            rocke_value_t* v = loop.iv;
            rocke_value_t* m = loop.iter_vars[0];
            rocke_value_t* i = loop.iter_vars[1];
            rocke_value_t* lg;
            rocke_value_t* lg_f;
            rocke_value_t* better;
            rocke_value_t* sel_m;
            rocke_value_t* sel_i;
            rocke_value_t* yield_vals[2];

            /* lg = b.global_load(logits, b.add(row_base, v), dt, align=2) */
            lg = rocke_b_global_load(b, logits, rocke_b_add(b, row_base, v), dt, /*align=*/2);
            /* lg_f = lg if dt == F32 else b.cast_to_f32(lg) */
            lg_f = (dt == rocke_f32()) ? lg : rocke_b_cast_to_f32(b, lg);
            /* better = b.fcmp("olt", m, lg_f) */
            better = rocke_b_fcmp(b, "olt", m, lg_f);
            /* b.scf_yield(b.select(better, lg_f, m), b.select(better, v, i))
             * Python evals the two selects left-to-right; sequence to match. */
            sel_m = rocke_b_select(b, better, lg_f, m);
            sel_i = rocke_b_select(b, better, v, i);
            yield_vals[0] = sel_m;
            yield_vals[1] = sel_i;
            rocke_b_scf_yield(b, yield_vals, 2);
        }
        rocke_b_region_leave(b);

        /* local_m = for_op.results[0]; local_i = for_op.results[1] */
        if(!rocke_ir_builder_ok(b) || loop.op == NULL || loop.op->num_results < 2)
        {
            return NULL;
        }
        local_m = loop.op->results[0];
        local_i = loop.op->results[1];

        /* _, arg = block_lds_reduce_with_index(b, local_m, local_i, lds_val,
         *              lds_idx, tid, block_size=bs, combine="argmax") */
        if(!rocke_block_lds_reduce_with_index(
               b, local_m, local_i, lds_val, lds_idx, tid, bs, ROCKE_INDEX_ARGMAX, &red_val, &arg))
        {
            return NULL;
        }
        (void)red_val; /* "_" in Python: the reduced value is discarded */

        /* with b.scf_if(b.cmp_eq(tid, b.const_i32(0))): */
        {
            rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, tid, rocke_b_const_i32(b, 0)));
            rocke_b_region_enter(b, iff.then_region);
            /* b.global_store(out_ids, token, arg, align=4) */
            rocke_b_global_store(b, out_ids, token, arg, /*align=*/4);
            rocke_b_region_leave(b);
        }

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
rocke_kernel_def_t* rocke_build_qwen3_sampler_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_sampler_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_qwen3_sampler_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_qwen3_sampler_gfx1250(b, spec, arch);
    });
}

/* ===================================================================== *
 *  qwen3_greedy_sampler_grid(num_tokens, spec) -> (num_tokens, 1, 1)
 * ===================================================================== */
rocke_status_t rocke_qwen3_sampler_gfx1250_grid(int num_tokens,
                                                const rocke_qwen3_sampler_gfx1250_spec_t* spec,
                                                int out[3])
{
    (void)spec;
    if(out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = num_tokens;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  lower_to_llvm convenience.
 * ===================================================================== */
rocke_status_t
    rocke_qwen3_sampler_gfx1250_lower_to_llvm(const rocke_qwen3_sampler_gfx1250_spec_t* spec,
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

    kernel = rocke_build_qwen3_sampler_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n;
            if(m == NULL)
            {
                m = "build_qwen3_sampler_gfx1250 failed";
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
