// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_attention_unified_attention_unified_entry.c -- PUBLIC ENTRY / GLUE
 * bucket of the chunked C99 port of the SCALAR unified-attention kernel
 * builders (ck_dsl/instances/common/attention_unified.py).
 *
 * SCOPE (this translation unit):
 *   - the @dataclass field defaults / @property ports
 *       ckc_unified_attention_problem_default
 *       ckc_unified_attention_problem_num_queries_per_kv  (num_queries_per_kv)
 *       ckc_unified_attention_problem_all_decode          (all_decode)
 *   - the coverage gate
 *       ckc_unified_attention_supports_scalar  (supports_native_unified_attention)
 *   - the three kernel_name fns (via ckc_kernel_name_join)
 *       ckc_unified_attention_{2d,3d,reduce}_scalar_kernel_name
 *   - the three launch grid fns
 *       ckc_unified_attention_{2d,3d,reduce}_scalar_grid
 *   - the 2D signature
 *       ckc_unified_attention_2d_scalar_signature  (_attn_signature, no bt_stride)
 *   - the three build drivers, each ctx_init -> phase functions in Python
 *     execution order (see instance_attention_unified_internal.h)
 *       ckc_build_unified_attention_{2d,3d,reduce}_scalar
 *   - the _new + _lower_to_llvm convenience wrappers
 *
 * The phase functions (ckc_attn_unified_ctx_init / _declare_scalar_params /
 * _emit_find_seq_idx / _emit_prologue / _emit_2d_softmax_loop / _emit_2d_epilogue
 * / _emit_3d_segment_loop / _emit_3d_epilogue / _declare_reduce_params /
 * _emit_reduce_max_loop / _emit_reduce_combine_loop / _emit_reduce_epilogue) are
 * implemented in the sibling phase translation units; this file calls them via
 * the internal header.
 */
#include "ckc/instance_attention_unified.h"
#include "ckc/instance_attention_unified_internal.h"

#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ===================================================================== *
 *  Local helpers
 * ===================================================================== */

/* "fp16"/"bf16" -> the manifest io dtype string ("f16"/"bf16"). */
static const char* ckc__io_dtype(const char* dtype)
{
    if(dtype && strcmp(dtype, "fp16") == 0)
        return "f16";
    return "bf16";
}

/* UnifiedAttention2DSpec.dtype_ir / UnifiedAttention3DSpec.dtype_ir:
 *   fp16 -> F16, bf16 -> BF16, else ValueError -> NULL (ctx_init sets sticky). */
static const ckc_type_t* ckc__dtype_ir_2d3d(const ckc_unified_attention_problem_t* p)
{
    if(p->dtype && strcmp(p->dtype, "fp16") == 0)
        return ckc_f16();
    if(p->dtype && strcmp(p->dtype, "bf16") == 0)
        return ckc_bf16();
    return NULL;
}

/* UnifiedAttentionReduceSpec.dtype_ir:
 *   F16 if dtype == "fp16" else BF16 (never raises). */
static const ckc_type_t* ckc__dtype_ir_reduce(const ckc_unified_attention_problem_t* p)
{
    if(p->dtype && strcmp(p->dtype, "fp16") == 0)
        return ckc_f16();
    return ckc_bf16();
}

/* Write a static diagnostic into err (capacity err_cap) when non-NULL. */
static void ckc__set_err(char* err, size_t err_cap, const char* m)
{
    size_t n;
    if(err == NULL || err_cap == 0 || m == NULL)
        return;
    n = strlen(m);
    if(n >= err_cap)
        n = err_cap - 1;
    memcpy(err, m, n);
    err[n] = '\0';
}

/* ===================================================================== *
 *  @dataclass defaults + @property ports
 * ===================================================================== */

ckc_unified_attention_problem_t ckc_unified_attention_problem_default(void)
{
    ckc_unified_attention_problem_t p;
    memset(&p, 0, sizeof(p));
    p.dtype = NULL;
    p.q_dtype = NULL;
    p.sliding_window = 0;
    p.softcap = 0.0;
    p.use_sinks = false;
    p.use_alibi = false;
    p.use_qq_bias = false;
    p.use_fp8 = false;
    p.num_sms = 120;
    p.waves_per_eu = 0;
    p.waves_per_eu_set = false;
    p.compile_backend = NULL;
    p.num_kv_blocks = 0;
    return p;
}

/* @property num_queries_per_kv:
 *   if num_query_heads % num_kv_heads: raise ValueError
 *   return num_query_heads // num_kv_heads          (ValueError -> -1) */
int ckc_unified_attention_problem_num_queries_per_kv(const ckc_unified_attention_problem_t* p)
{
    if(p == NULL || p->num_kv_heads == 0)
        return -1;
    if(p->num_query_heads % p->num_kv_heads)
        return -1;
    return p->num_query_heads / p->num_kv_heads;
}

/* @property all_decode = (max_seqlen_q == 1). */
bool ckc_unified_attention_problem_all_decode(const ckc_unified_attention_problem_t* p)
{
    return p != NULL && p->max_seqlen_q == 1;
}

/* ===================================================================== *
 *  supports_native_unified_attention(problem) -- coverage gate
 * ===================================================================== */

bool ckc_unified_attention_supports_scalar(const ckc_unified_attention_problem_t* p,
                                           const char** out_reason)
{
    if(p == NULL)
    {
        if(out_reason != NULL)
            *out_reason = "null problem";
        return false;
    }
    if(p->head_size != 64 && p->head_size != 128 && p->head_size != 256)
    {
        if(out_reason != NULL)
            *out_reason = "unsupported head_size";
        return false;
    }
    if(p->block_size != 16 && p->block_size != 32 && p->block_size != 64)
    {
        if(out_reason != NULL)
            *out_reason = "unsupported block_size";
        return false;
    }
    if(!(p->dtype && (strcmp(p->dtype, "fp16") == 0 || strcmp(p->dtype, "bf16") == 0)))
    {
        if(out_reason != NULL)
            *out_reason = "unsupported dtype";
        return false;
    }
    /* FP8 K/V cache: scalar 2D backend does not implement the FP8 dequant
     * path yet (only the tiled 2D/3D backends do). */
    if(p->use_fp8 || p->q_dtype != NULL)
    {
        if(out_reason != NULL)
            *out_reason = "FP8 unified attention is not enabled in the scalar 2D path yet";
        return false;
    }
    if(p->use_alibi)
    {
        if(out_reason != NULL)
            *out_reason = "ALiBi slopes are not enabled in CK DSL attention yet";
        return false;
    }
    if(p->use_qq_bias)
    {
        if(out_reason != NULL)
            *out_reason = "QQ bias is not enabled in CK DSL attention yet";
        return false;
    }
    if(out_reason != NULL)
        *out_reason = "supported by scalar CK DSL 2D attention backend";
    return true;
}

/* ===================================================================== *
 *  kernel_name fns  (UnifiedAttention*Spec.kernel_name)
 * ===================================================================== */

ckc_status_t ckc_unified_attention_2d_scalar_kernel_name(const ckc_unified_attention_problem_t* p,
                                                         const char* name,
                                                         char* out,
                                                         size_t out_cap)
{
    char qbuf[32], hbuf[32], kvbuf[32], dbuf[32], bbuf[32];
    const char* prefix;
    if(p == NULL || out == NULL)
        return CKC_ERR_VALUE;
    prefix = name ? name : "ck_dsl_unified_attention_2d_scalar";

    snprintf(qbuf, sizeof(qbuf), "q%d", p->total_q);
    snprintf(hbuf, sizeof(hbuf), "h%d", p->num_query_heads);
    snprintf(kvbuf, sizeof(kvbuf), "kv%d", p->num_kv_heads);
    snprintf(dbuf, sizeof(dbuf), "d%d", p->head_size);
    snprintf(bbuf, sizeof(bbuf), "b%d", p->block_size);

    {
        const char* parts[6] = {qbuf, hbuf, kvbuf, dbuf, bbuf, p->dtype ? p->dtype : ""};
        /* flags={"sink":use_sinks,"sw":sliding_window>0,"softcap":softcap>0} */
        const char* flag_names[3] = {"sink", "sw", "softcap"};
        int flag_on[3]
            = {p->use_sinks ? 1 : 0, p->sliding_window > 0 ? 1 : 0, p->softcap > 0.0 ? 1 : 0};
        return ckc_kernel_name_join(prefix, parts, 6, flag_names, flag_on, 3, out, out_cap, NULL);
    }
}

ckc_status_t ckc_unified_attention_3d_scalar_kernel_name(const ckc_unified_attention_problem_t* p,
                                                         const char* name,
                                                         int num_segments,
                                                         char* out,
                                                         size_t out_cap)
{
    char qbuf[32], hbuf[32], kvbuf[32], dbuf[32], bbuf[32], segbuf[32];
    const char* prefix;
    if(p == NULL || out == NULL)
        return CKC_ERR_VALUE;
    prefix = name ? name : "ck_dsl_unified_attention_3d_scalar";

    snprintf(qbuf, sizeof(qbuf), "q%d", p->total_q);
    snprintf(hbuf, sizeof(hbuf), "h%d", p->num_query_heads);
    snprintf(kvbuf, sizeof(kvbuf), "kv%d", p->num_kv_heads);
    snprintf(dbuf, sizeof(dbuf), "d%d", p->head_size);
    snprintf(bbuf, sizeof(bbuf), "b%d", p->block_size);
    snprintf(segbuf, sizeof(segbuf), "seg%d", num_segments);

    {
        const char* parts[7] = {qbuf, hbuf, kvbuf, dbuf, bbuf, segbuf, p->dtype ? p->dtype : ""};
        return ckc_kernel_name_join(prefix, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
    }
}

ckc_status_t
    ckc_unified_attention_reduce_scalar_kernel_name(const ckc_unified_attention_problem_t* p,
                                                    const char* name,
                                                    int num_segments,
                                                    char* out,
                                                    size_t out_cap)
{
    char qbuf[32], hbuf[32], dbuf[32], segbuf[32];
    const char* prefix;
    if(p == NULL || out == NULL)
        return CKC_ERR_VALUE;
    prefix = name ? name : "ck_dsl_unified_attention_reduce_scalar";

    snprintf(qbuf, sizeof(qbuf), "q%d", p->total_q);
    snprintf(hbuf, sizeof(hbuf), "h%d", p->num_query_heads);
    snprintf(dbuf, sizeof(dbuf), "d%d", p->head_size);
    snprintf(segbuf, sizeof(segbuf), "seg%d", num_segments);

    {
        const char* parts[5] = {qbuf, hbuf, dbuf, segbuf, p->dtype ? p->dtype : ""};
        return ckc_kernel_name_join(prefix, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
    }
}

/* ===================================================================== *
 *  Launch grids  (block_id_{x,y,z} extents)
 * ===================================================================== */

void ckc_unified_attention_2d_scalar_grid(const ckc_unified_attention_problem_t* p, int out[3])
{
    if(p == NULL || out == NULL)
        return;
    out[0] = p->total_q;
    out[1] = p->num_query_heads;
    out[2] = p->head_size;
}

void ckc_unified_attention_3d_scalar_grid(const ckc_unified_attention_problem_t* p,
                                          int num_segments,
                                          int out[3])
{
    if(p == NULL || out == NULL)
        return;
    out[0] = p->total_q;
    out[1] = p->num_query_heads;
    out[2] = num_segments * p->head_size;
}

void ckc_unified_attention_reduce_scalar_grid(const ckc_unified_attention_problem_t* p, int out[3])
{
    if(p == NULL || out == NULL)
        return;
    out[0] = p->total_q;
    out[1] = p->num_query_heads;
    out[2] = p->head_size;
}

/* ===================================================================== *
 *  2D signature -- _attn_signature(dtype, include_bt_stride=False)
 * ===================================================================== */

ckc_status_t ckc_unified_attention_2d_scalar_signature(const ckc_unified_attention_problem_t* p,
                                                       ckc_arena_t* arena,
                                                       const ckc_sig_entry_t** out_items,
                                                       size_t* out_count)
{
    ckc_signature_builder_t sb;
    ckc_status_t st;
    const char* io;
    if(p == NULL || arena == NULL || out_items == NULL || out_count == NULL)
        return CKC_ERR_VALUE;
    io = ckc__io_dtype(p->dtype);

    st = ckc_signature_builder_init(&sb, arena);
    if(st != CKC_OK)
        return st;

    ckc_signature_builder_ptr(&sb, "output_ptr", io, NULL);
    ckc_signature_builder_ptr(&sb, "query_ptr", io, NULL);
    ckc_signature_builder_ptr(&sb, "key_cache_ptr", io, NULL);
    ckc_signature_builder_ptr(&sb, "value_cache_ptr", io, NULL);
    ckc_signature_builder_ptr(&sb, "sink_ptr", io, NULL);
    ckc_signature_builder_ptr(&sb, "block_tables_ptr", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "seq_lens_ptr", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "alibi_slopes_ptr", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "qq_bias_ptr", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "query_start_len_ptr", "i32", NULL);
    ckc_signature_builder_scalar(&sb, "scale", "f32");
    ckc_signature_builder_scalar(&sb, "k_scale", "f32");
    ckc_signature_builder_scalar(&sb, "v_scale", "f32");
    ckc_signature_builder_scalar(&sb, "out_scale", "f32");
    ckc_signature_builder_scalar(&sb, "softcap", "f32");
    ckc_signature_builder_scalar(&sb, "num_seqs", "i32");
    /* include_bt_stride=False, include_qq_bias_stride=False => no tail scalars */

    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  build_unified_attention_2d(spec)
 *
 *  ctx_init -> declare_scalar_params (+ 2D tail params) -> emit_find_seq_idx ->
 *  emit_prologue -> emit_2d_softmax_loop -> emit_2d_epilogue -> ctx.kernel.
 *  The 2D tail params (out_scale, softcap, num_seqs) follow the shared ABI
 *  prefix; declare_scalar_params declares only the prefix, so the tail is
 *  appended here -- matching the Python build body's param order.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_unified_attention_2d_scalar(ckc_ir_builder_t* b,
                                                        const ckc_unified_attention_problem_t* p,
                                                        const char* name)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        ckc_attn_unified_build_ctx_t ctx;
        const ckc_type_t* dtype;
        ckc_kernel_def_t* k;
        (void)name; /* the kernel_name was bound at builder init in _new */

        if(b == NULL || p == NULL)
            return NULL;
        dtype = ckc__dtype_ir_2d3d(p); /* NULL => ctx_init sets sticky error */

        if(!ckc_attn_unified_ctx_init(&ctx, b, CKC_ATTN_UNIFIED_2D, p, dtype, 0))
            return NULL;

        k = ckc_ir_builder_kernel(b);
        ckc_attr_set_int(b, &k->attrs, "max_workgroup_size", 64);

        /* output = b.param("output_ptr", PtrType(dtype,"global"), noalias, writeonly,
         * align=16) -- declared BEFORE the shared ABI prefix, as in Python. */
        {
            const ckc_type_t* ptr_ty = ckc_ptr_type(b, dtype, "global");
            ckc_param_opts_t opts;
            memset(&opts, 0, sizeof(opts));
            opts.noalias = true;
            opts.noalias_set = true;
            opts.writeonly = true;
            opts.writeonly_set = true;
            opts.align = 16;
            opts.align_set = true;
            ctx.output = ckc_b_param(b, "output_ptr", ptr_ty, &opts);
        }

        /* abi = _declare_scalar_attn_params(b, dtype) */
        ckc_attn_unified_declare_scalar_params(&ctx);

        /* 2D tail params: out_scale, softcap, num_seqs. */
        ctx.out_scale = ckc_b_param(b, "out_scale", ckc_f32(), NULL);
        ctx.softcap = ckc_b_param(b, "softcap", ckc_f32(), NULL);
        ctx.num_seqs = ckc_b_param(b, "num_seqs", ckc_i32(), NULL);

        /* prologue: grid ids + seq-idx scan + per-seq geometry + SSA constants.
         * The seq-idx scan reads q_tok, which emit_prologue computes first; the
         * prologue calls emit_find_seq_idx internally at the correct point. Calling
         * it here (before q_tok exists) poisoned the builder with a NULL operand and
         * crashed the second (in-prologue) scan on a zeroed for-struct. */
        ckc_attn_unified_emit_prologue(&ctx);

        /* online-softmax loop + guarded epilogue. */
        ckc_attn_unified_emit_2d_softmax_loop(&ctx);
        ckc_attn_unified_emit_2d_epilogue(&ctx);

        ctx.kernel = ckc_ir_builder_kernel(b);
        return ckc_ir_builder_ok(b) ? ctx.kernel : NULL;
    });
}

/* ===================================================================== *
 *  build_unified_attention_3d(spec)
 *
 *  ctx_init -> declare_scalar_params (+ segm_output/max/expsum + 3D tail) ->
 *  emit_find_seq_idx -> emit_prologue -> emit_3d_segment_loop ->
 *  emit_3d_epilogue -> ctx.kernel.
 * ===================================================================== */
ckc_kernel_def_t* ckc_build_unified_attention_3d_scalar(ckc_ir_builder_t* b,
                                                        const ckc_unified_attention_problem_t* p,
                                                        const char* name,
                                                        int num_segments)
{
    ckc_attn_unified_build_ctx_t ctx;
    const ckc_type_t* dtype;
    ckc_kernel_def_t* k;
    (void)name;

    if(b == NULL || p == NULL)
        return NULL;
    dtype = ckc__dtype_ir_2d3d(p);

    if(!ckc_attn_unified_ctx_init(&ctx, b, CKC_ATTN_UNIFIED_3D, p, dtype, num_segments))
        return NULL;

    k = ckc_ir_builder_kernel(b);
    ckc_attr_set_int(b, &k->attrs, "max_workgroup_size", 64);

    /* The three F32 per-segment output params precede the shared ABI prefix. */
    {
        const ckc_type_t* f32_ptr = ckc_ptr_type(b, ckc_f32(), "global");
        ckc_param_opts_t opts;
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        ctx.segm_output = ckc_b_param(b, "segm_output_ptr", f32_ptr, &opts);
        ctx.segm_max = ckc_b_param(b, "segm_max_ptr", f32_ptr, &opts);
        ctx.segm_expsum = ckc_b_param(b, "segm_expsum_ptr", f32_ptr, &opts);
    }

    /* abi = _declare_scalar_attn_params(b, dtype) */
    ckc_attn_unified_declare_scalar_params(&ctx);

    /* 3D tail params: softcap (unused by body) + num_seqs. */
    ctx.softcap = ckc_b_param(b, "softcap", ckc_f32(), NULL);
    ctx.num_seqs = ckc_b_param(b, "num_seqs", ckc_i32(), NULL);

    /* emit_prologue runs the seq-idx scan internally after computing q_tok;
     * do not call emit_find_seq_idx here (q_tok would still be NULL). */
    ckc_attn_unified_emit_prologue(&ctx);

    ckc_attn_unified_emit_3d_segment_loop(&ctx);
    ckc_attn_unified_emit_3d_epilogue(&ctx);

    ctx.kernel = ckc_ir_builder_kernel(b);
    return ckc_ir_builder_ok(b) ? ctx.kernel : NULL;
}

/* ===================================================================== *
 *  build_unified_attention_reduce(spec)
 *
 *  ctx_init -> declare_reduce_params -> emit_reduce_max_loop ->
 *  emit_reduce_combine_loop -> emit_reduce_epilogue -> ctx.kernel.
 * ===================================================================== */
ckc_kernel_def_t*
    ckc_build_unified_attention_reduce_scalar(ckc_ir_builder_t* b,
                                              const ckc_unified_attention_problem_t* p,
                                              int num_segments,
                                              const char* name)
{
    ckc_attn_unified_build_ctx_t ctx;
    const ckc_type_t* dtype;
    ckc_kernel_def_t* k;
    (void)name;

    if(b == NULL || p == NULL)
        return NULL;
    /* UnifiedAttentionReduceSpec.dtype_ir never raises (F16 if fp16 else BF16). */
    dtype = ckc__dtype_ir_reduce(p);

    if(!ckc_attn_unified_ctx_init(&ctx, b, CKC_ATTN_UNIFIED_REDUCE, p, dtype, num_segments))
        return NULL;

    k = ckc_ir_builder_kernel(b);
    ckc_attr_set_int(b, &k->attrs, "max_workgroup_size", 64);

    /* Reduce-specific narrow ABI declaration (output + segm_* + seq_lens + cu_q)
     * plus the grid ids. */
    ckc_attn_unified_declare_reduce_params(&ctx);

    ckc_attn_unified_emit_reduce_max_loop(&ctx);
    ckc_attn_unified_emit_reduce_combine_loop(&ctx);
    ckc_attn_unified_emit_reduce_epilogue(&ctx);

    ctx.kernel = ckc_ir_builder_kernel(b);
    return ckc_ir_builder_ok(b) ? ctx.kernel : NULL;
}

/* ===================================================================== *
 *  _new convenience wrappers: init builder via kernel_name, then build.
 * ===================================================================== */

ckc_kernel_def_t* ckc_build_unified_attention_2d_scalar_new(
    ckc_ir_builder_t* b, const ckc_unified_attention_problem_t* p, const char* name)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char kname[512];
        if(b == NULL || p == NULL)
            return NULL;
        if(ckc_unified_attention_2d_scalar_kernel_name(p, name, kname, sizeof(kname)) != CKC_OK)
            return NULL;
        if(ckc_ir_builder_init(b, kname) != CKC_OK)
            return NULL;
        return ckc_build_unified_attention_2d_scalar(b, p, name);
    });
}

ckc_kernel_def_t*
    ckc_build_unified_attention_3d_scalar_new(ckc_ir_builder_t* b,
                                              const ckc_unified_attention_problem_t* p,
                                              const char* name,
                                              int num_segments)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char kname[512];
        if(b == NULL || p == NULL)
            return NULL;
        if(ckc_unified_attention_3d_scalar_kernel_name(p, name, num_segments, kname, sizeof(kname))
           != CKC_OK)
            return NULL;
        if(ckc_ir_builder_init(b, kname) != CKC_OK)
            return NULL;
        return ckc_build_unified_attention_3d_scalar(b, p, name, num_segments);
    });
}

ckc_kernel_def_t*
    ckc_build_unified_attention_reduce_scalar_new(ckc_ir_builder_t* b,
                                                  const ckc_unified_attention_problem_t* p,
                                                  int num_segments,
                                                  const char* name)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char kname[512];
        if(b == NULL || p == NULL)
            return NULL;
        if(ckc_unified_attention_reduce_scalar_kernel_name(
               p, name, num_segments, kname, sizeof(kname))
           != CKC_OK)
            return NULL;
        if(ckc_ir_builder_init(b, kname) != CKC_OK)
            return NULL;
        return ckc_build_unified_attention_reduce_scalar(b, p, num_segments, name);
    });
}

/* ===================================================================== *
 *  _lower_to_llvm convenience: build into an internally-owned builder, lower.
 * ===================================================================== */

ckc_status_t ckc_unified_attention_2d_scalar_lower_to_llvm(const ckc_unified_attention_problem_t* p,
                                                           const char* name,
                                                           ckc_llvm_flavor_t flavor,
                                                           char** out_ll,
                                                           char* err,
                                                           size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(p == NULL || out_ll == NULL)
    {
        ckc__set_err(err, err_cap, "lower_to_llvm: null problem/out");
        return CKC_ERR_VALUE;
    }

    kernel = ckc_build_unified_attention_2d_scalar_new(&b, p, name);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        ckc__set_err(err, err_cap, ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, NULL, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t ckc_unified_attention_3d_scalar_lower_to_llvm(const ckc_unified_attention_problem_t* p,
                                                           const char* name,
                                                           int num_segments,
                                                           ckc_llvm_flavor_t flavor,
                                                           char** out_ll,
                                                           char* err,
                                                           size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(p == NULL || out_ll == NULL)
    {
        ckc__set_err(err, err_cap, "lower_to_llvm: null problem/out");
        return CKC_ERR_VALUE;
    }

    kernel = ckc_build_unified_attention_3d_scalar_new(&b, p, name, num_segments);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        ckc__set_err(err, err_cap, ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, NULL, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}

ckc_status_t
    ckc_unified_attention_reduce_scalar_lower_to_llvm(const ckc_unified_attention_problem_t* p,
                                                      int num_segments,
                                                      const char* name,
                                                      ckc_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(p == NULL || out_ll == NULL)
    {
        ckc__set_err(err, err_cap, "lower_to_llvm: null problem/out");
        return CKC_ERR_VALUE;
    }

    kernel = ckc_build_unified_attention_reduce_scalar_new(&b, p, num_segments, name);
    if(kernel == NULL)
    {
        st = ckc_ir_builder_status(&b);
        ckc__set_err(err, err_cap, ckc_ir_builder_error(&b));
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, NULL, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
