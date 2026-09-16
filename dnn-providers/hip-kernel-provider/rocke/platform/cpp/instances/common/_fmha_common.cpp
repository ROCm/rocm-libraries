// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/_fmha_common.py: FmhaShape,
 * FmhaCommonSpec, FmhaMaskMode, validate_common_spec, FmhaKernelBuilder.
 *
 * See rocke/helper_rocke.instances.common._fmha_common.h for the mapping.
 */
#include "rocke/helper_rocke.instances.common._fmha_common.h"

#include <string.h>

#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type */

/* ====================================================================== *
 * FmhaMaskMode
 * ====================================================================== */

const char* rocke_fmha_mask_mode_name(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_NONE:
        return "none";
    case ROCKE_FMHA_MASK_CAUSAL:
        return "causal";
    case ROCKE_FMHA_MASK_SLIDING_WINDOW:
        return "sliding_window";
    case ROCKE_FMHA_MASK_ALIBI:
        return "alibi";
    case ROCKE_FMHA_MASK_CUSTOM:
        return "custom";
    default:
        return NULL;
    }
}

/* ====================================================================== *
 * FmhaShape
 * ====================================================================== */

rocke_fmha_shape_t rocke_fmha_shape_make(
    int head_size, int num_query_heads, int num_kv_heads, int block_size_q, int block_size_k)
{
    rocke_fmha_shape_t s;
    s.head_size = head_size;
    s.num_query_heads = num_query_heads;
    s.num_kv_heads = num_kv_heads;
    s.block_size_q = block_size_q;
    s.block_size_k = block_size_k;
    return s;
}

rocke_fmha_shape_t rocke_fmha_shape_default(int head_size, int num_query_heads, int num_kv_heads)
{
    /* block_size_q: int = 16; block_size_k: int = 64 */
    return rocke_fmha_shape_make(head_size, num_query_heads, num_kv_heads, 16, 64);
}

rocke_status_t rocke_fmha_shape_num_queries_per_kv(const rocke_fmha_shape_t* s, int* out)
{
    if(s == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* if num_query_heads % num_kv_heads: raise ValueError(...) */
    if(s->num_kv_heads == 0 || (s->num_query_heads % s->num_kv_heads) != 0)
    {
        return ROCKE_ERR_VALUE;
    }
    if(out != NULL)
    {
        *out = s->num_query_heads / s->num_kv_heads;
    }
    return ROCKE_OK;
}

/* ====================================================================== *
 * FmhaCommonSpec
 * ====================================================================== */

rocke_fmha_common_spec_t rocke_fmha_common_spec_default(rocke_fmha_shape_t shape)
{
    rocke_fmha_common_spec_t spec;
    spec.shape = shape;
    spec.dtype = "f16";
    spec.scale_log2 = 0.0;
    spec.mask_mode = ROCKE_FMHA_MASK_NONE;
    spec.sliding_window = 0;
    spec.use_softcap = false;
    spec.use_rotary = false;
    spec.use_dropout = false;
    spec.use_sinks = false;
    return spec;
}

/* ====================================================================== *
 * validate_common_spec
 * ====================================================================== */

/* Set *out_reason (if non-NULL) to an arena-printf'd message, falling back to
 * `fallback` on OOM. Returns `ret` so callers can `return reject(...)`. */
static bool rocke_i_fmha_set_reason(rocke_arena_t* arena,
                                    const char** out_reason,
                                    bool ret,
                                    const char* formatted,
                                    const char* fallback)
{
    (void)arena;
    if(out_reason != NULL)
    {
        *out_reason = (formatted != NULL) ? formatted : fallback;
    }
    return ret;
}

bool rocke_fmha_validate_common_spec(rocke_arena_t* arena,
                                     const rocke_fmha_common_spec_t* spec,
                                     const char** out_reason)
{
    const rocke_fmha_shape_t* s;
    const char* dt;
    const char* mode;

    if(spec == NULL)
    {
        if(out_reason != NULL)
        {
            *out_reason = "invalid spec";
        }
        return false;
    }
    s = &spec->shape;
    dt = (spec->dtype != NULL) ? spec->dtype : "";

    /* if s.head_size <= 0 or s.head_size > 256: */
    if(s->head_size <= 0 || s->head_size > 256)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena, "head_size %d out of supported range (1..256)", s->head_size),
            "head_size out of supported range (1..256)");
    }
    /* if s.head_size not in (32, 64, 128, 192, 256): */
    if(s->head_size != 32 && s->head_size != 64 && s->head_size != 128 && s->head_size != 192
       && s->head_size != 256)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena,
                               "head_size %d not in the supported set "
                               "{32, 64, 128, 192, 256} (CK Tile's standard FMHA shapes)",
                               s->head_size),
            "head_size not in the supported set");
    }
    /* if s.num_query_heads <= 0 or s.num_kv_heads <= 0: */
    if(s->num_query_heads <= 0 || s->num_kv_heads <= 0)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena,
                               "num_query_heads / num_kv_heads must be > 0 "
                               "(got %d, %d)",
                               s->num_query_heads,
                               s->num_kv_heads),
            "num_query_heads / num_kv_heads must be > 0");
    }
    /* if s.num_query_heads % s.num_kv_heads != 0: */
    if((s->num_query_heads % s->num_kv_heads) != 0)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena,
                               "num_query_heads (%d) must be divisible by "
                               "num_kv_heads (%d) for GQA / MQA",
                               s->num_query_heads,
                               s->num_kv_heads),
            "num_query_heads must be divisible by num_kv_heads");
    }
    /* if s.block_size_q not in (16, 32, 64, 128): */
    if(s->block_size_q != 16 && s->block_size_q != 32 && s->block_size_q != 64
       && s->block_size_q != 128)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena, "block_size_q %d not in {16, 32, 64, 128}", s->block_size_q),
            "block_size_q not in {16, 32, 64, 128}");
    }
    /* if s.block_size_k not in (16, 32, 64, 128, 256): */
    if(s->block_size_k != 16 && s->block_size_k != 32 && s->block_size_k != 64
       && s->block_size_k != 128 && s->block_size_k != 256)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena, "block_size_k %d not in {16..256}", s->block_size_k),
            "block_size_k not in {16..256}");
    }
    /* if spec.dtype not in ("f16", "fp16", "bf16"): */
    if(strcmp(dt, "f16") != 0 && strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(arena, "dtype '%s' must be 'f16' / 'fp16' / 'bf16'", dt),
            "dtype must be 'f16' / 'fp16' / 'bf16'");
    }
    /* if spec.mask_mode not in (...): reproduced via the enum range. */
    mode = rocke_fmha_mask_mode_name(spec->mask_mode);
    if(mode == NULL)
    {
        return rocke_i_fmha_set_reason(arena, out_reason, false, NULL, "mask_mode not recognised");
    }
    /* if spec.mask_mode == "sliding_window" and spec.sliding_window <= 0: */
    if(spec->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW && spec->sliding_window <= 0)
    {
        return rocke_i_fmha_set_reason(
            arena,
            out_reason,
            false,
            rocke_arena_printf(
                arena, "sliding_window mask requires window > 0 (got %d)", spec->sliding_window),
            "sliding_window mask requires window > 0");
    }
    /* if spec.use_sinks and spec.dtype not in ("f16", "fp16"): */
    if(spec->use_sinks && strcmp(dt, "f16") != 0 && strcmp(dt, "fp16") != 0)
    {
        return rocke_i_fmha_set_reason(
            arena, out_reason, false, NULL, "use_sinks=True only supports f16 in v1");
    }
    /* return True, "ok" */
    if(out_reason != NULL)
    {
        *out_reason = "ok";
    }
    return true;
}

/* ====================================================================== *
 * FmhaKernelBuilder -- registries
 * ====================================================================== */

/* Naming convention: stride_{lower(name)}_token / stride_{lower(name)}_head.
 * _stride_param_names(name) -> (token, head); both written to the arena. */
static void rocke_i_stride_param_names(rocke_arena_t* arena,
                                       const char* name,
                                       const char** out_token,
                                       const char** out_head)
{
    char* lower;
    size_t i;
    size_t len = strlen(name);
    lower = (char*)rocke_arena_alloc(arena, len + 1);
    if(lower == NULL)
    {
        *out_token = NULL;
        *out_head = NULL;
        return;
    }
    for(i = 0; i < len; ++i)
    {
        char c = name[i];
        if(c >= 'A' && c <= 'Z')
        {
            c = (char)(c - 'A' + 'a');
        }
        lower[i] = c;
    }
    lower[len] = '\0';
    *out_token = rocke_arena_printf(arena, "stride_%s_token", lower);
    *out_head = rocke_arena_printf(arena, "stride_%s_head", lower);
}

/* Grow `*arr` (array of element size `elem_size`) so it can hold n+1 entries.
 * Arena allocations are never freed; growth copies into a fresh block. Returns 1
 * on success, 0 on OOM. */
static int
    rocke_i_reserve(rocke_arena_t* arena, void** arr, size_t* cap, size_t count, size_t elem_size)
{
    size_t new_cap;
    void* nb;
    if(count < *cap)
    {
        return 1;
    }
    new_cap = (*cap == 0) ? 8 : (*cap * 2);
    nb = rocke_arena_alloc(arena, new_cap * elem_size);
    if(nb == NULL)
    {
        return 0;
    }
    if(*arr != NULL && count > 0)
    {
        memcpy(nb, *arr, count * elem_size);
    }
    *arr = nb;
    *cap = new_cap;
    return 1;
}

static void rocke_i_sig_order_append(rocke_fmha_kernel_builder_t* kb,
                                     rocke_fmha_sig_kind_t kind,
                                     const char* name,
                                     const char* dtype)
{
    rocke_arena_t* arena = &kb->b.arena;
    if(!rocke_i_reserve(arena,
                        (void**)&kb->sig_order,
                        &kb->cap_sig_order,
                        kb->n_sig_order,
                        sizeof(rocke_fmha_sig_order_entry_t)))
    {
        return;
    }
    kb->sig_order[kb->n_sig_order].kind = kind;
    kb->sig_order[kb->n_sig_order].name = rocke_arena_strdup(arena, name);
    kb->sig_order[kb->n_sig_order].dtype = rocke_arena_strdup(arena, dtype);
    kb->n_sig_order++;
}

static void rocke_i_named_append(rocke_arena_t* arena,
                                 rocke_fmha_named_value_t** arr,
                                 size_t* n,
                                 size_t* cap,
                                 const char* name,
                                 rocke_value_t* value)
{
    if(!rocke_i_reserve(arena, (void**)arr, cap, *n, sizeof(rocke_fmha_named_value_t)))
    {
        return;
    }
    (*arr)[*n].name = rocke_arena_strdup(arena, name);
    (*arr)[*n].value = value;
    (*n)++;
}

static rocke_value_t*
    rocke_i_named_lookup(const rocke_fmha_named_value_t* arr, size_t n, const char* name)
{
    size_t i;
    for(i = 0; i < n; ++i)
    {
        if(strcmp(arr[i].name, name) == 0)
        {
            return arr[i].value;
        }
    }
    return NULL;
}

static const rocke_fmha_stride_pair_t* rocke_i_stride_lookup(const rocke_fmha_kernel_builder_t* kb,
                                                             const char* name)
{
    size_t i;
    for(i = 0; i < kb->n_stride_params; ++i)
    {
        if(strcmp(kb->stride_params[i].name, name) == 0)
        {
            return &kb->stride_params[i];
        }
    }
    return NULL;
}

/* ====================================================================== *
 * FmhaKernelBuilder -- lifecycle
 * ====================================================================== */

rocke_status_t rocke_fmha_kernel_builder_init(rocke_fmha_kernel_builder_t* kb,
                                              const char* kernel_name,
                                              const rocke_fmha_common_spec_t* common)
{
    rocke_status_t st;
    if(kb == NULL || common == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    memset(kb, 0, sizeof(*kb));
    kb->common = *common;
    /* self.b = IRBuilder(kernel_name) */
    st = rocke_ir_builder_init(&kb->b, kernel_name);
    if(st != ROCKE_OK)
    {
        return st;
    }
    return ROCKE_OK;
}

void rocke_fmha_kernel_builder_free(rocke_fmha_kernel_builder_t* kb)
{
    if(kb == NULL)
    {
        return;
    }
    rocke_ir_builder_free(&kb->b);
}

rocke_kernel_def_t* rocke_fmha_kernel_builder_kernel(rocke_fmha_kernel_builder_t* kb)
{
    return (kb != NULL) ? rocke_ir_builder_kernel(&kb->b) : NULL;
}

rocke_ir_builder_t* rocke_fmha_kernel_builder_builder(rocke_fmha_kernel_builder_t* kb)
{
    return (kb != NULL) ? &kb->b : NULL;
}

/* ====================================================================== *
 * FmhaKernelBuilder -- param declarations
 * ====================================================================== */

rocke_value_t* rocke_fmha_kernel_builder_add_tensor(rocke_fmha_kernel_builder_t* kb,
                                                    const char* name,
                                                    const char* dtype,
                                                    bool readonly,
                                                    bool writeonly,
                                                    int align)
{
    const char* actual_dtype;
    const rocke_type_t* ty;
    const rocke_type_t* ptr_ty;
    rocke_param_opts_t opts;
    rocke_value_t* p;

    if(kb == NULL)
    {
        return NULL;
    }
    /* actual_dtype = dtype or self.common.dtype */
    actual_dtype = (dtype != NULL) ? dtype : kb->common.dtype;
    if(actual_dtype == NULL)
    {
        actual_dtype = "";
    }

    if(strcmp(actual_dtype, "f16") == 0 || strcmp(actual_dtype, "fp16") == 0
       || strcmp(actual_dtype, "bf16") == 0)
    {
        ty = rocke_b_io_ir_type(&kb->b, actual_dtype);
    }
    else if(strcmp(actual_dtype, "fp8e4m3") == 0)
    {
        ty = rocke_fp8e4m3();
    }
    else if(strcmp(actual_dtype, "bf8e5m2") == 0)
    {
        ty = rocke_bf8e5m2();
    }
    else if(strcmp(actual_dtype, "i8") == 0)
    {
        ty = rocke_i8();
    }
    else
    {
        /* raise ValueError(...) -- record on the sticky-error builder. */
        kb->b.status = ROCKE_ERR_VALUE;
        return NULL;
    }
    if(ty == NULL)
    {
        return NULL;
    }

    /* PtrType(ty, "global") */
    ptr_ty = rocke_ptr_type(&kb->b, ty, "global");

    /* b.param(name, ptr, noalias=True, readonly=readonly, writeonly=writeonly,
     *         align=align) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = readonly;
    opts.readonly_set = true;
    opts.writeonly = writeonly;
    opts.writeonly_set = true;
    opts.align = align;
    opts.align_set = true;
    p = rocke_b_param(&kb->b, name, ptr_ty, &opts);

    /* self._tensor_params[name] = p */
    rocke_i_named_append(
        &kb->b.arena, &kb->tensor_params, &kb->n_tensor_params, &kb->cap_tensor_params, name, p);
    /* self._sig_order.append(("tensor", name, actual_dtype)) */
    rocke_i_sig_order_append(kb, ROCKE_FMHA_SIG_TENSOR, name, actual_dtype);
    return p;
}

rocke_value_t* rocke_fmha_kernel_builder_add_ptr(
    rocke_fmha_kernel_builder_t* kb, const char* name, const char* dtype, bool readonly, int align)
{
    const rocke_type_t* ty;
    const rocke_type_t* ptr_ty;
    rocke_param_opts_t opts;
    rocke_value_t* p;

    if(kb == NULL || dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "i32") == 0)
    {
        ty = rocke_i32();
    }
    else if(strcmp(dtype, "f32") == 0)
    {
        ty = rocke_f32();
    }
    else if(strcmp(dtype, "i8") == 0)
    {
        ty = rocke_i8();
    }
    else
    {
        ty = rocke_b_io_ir_type(&kb->b, dtype);
    }
    if(ty == NULL)
    {
        return NULL;
    }

    ptr_ty = rocke_ptr_type(&kb->b, ty, "global");

    /* b.param(name, ptr, noalias=True, readonly=readonly, align=align) */
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = readonly;
    opts.readonly_set = true;
    opts.align = align;
    opts.align_set = true;
    p = rocke_b_param(&kb->b, name, ptr_ty, &opts);

    /* self._other_params[name] = p */
    rocke_i_named_append(
        &kb->b.arena, &kb->other_params, &kb->n_other_params, &kb->cap_other_params, name, p);
    /* self._sig_order.append(("ptr", name, dtype)) */
    rocke_i_sig_order_append(kb, ROCKE_FMHA_SIG_PTR, name, dtype);
    return p;
}

rocke_value_t* rocke_fmha_kernel_builder_add_scalar(rocke_fmha_kernel_builder_t* kb,
                                                    const char* name,
                                                    const char* dtype)
{
    const rocke_type_t* ty;
    const char* dt;
    rocke_value_t* p;

    if(kb == NULL)
    {
        return NULL;
    }
    dt = (dtype != NULL) ? dtype : "i32";
    /* ty = I32 if dtype == "i32" else F32 */
    ty = (strcmp(dt, "i32") == 0) ? rocke_i32() : rocke_f32();
    /* b.param(name, ty) -- no ABI opts */
    p = rocke_b_param(&kb->b, name, ty, NULL);

    /* self._other_params[name] = p */
    rocke_i_named_append(
        &kb->b.arena, &kb->other_params, &kb->n_other_params, &kb->cap_other_params, name, p);
    /* self._sig_order.append(("scalar", name, dtype)) */
    rocke_i_sig_order_append(kb, ROCKE_FMHA_SIG_SCALAR, name, dt);
    return p;
}

void rocke_fmha_kernel_builder_add_strides(rocke_fmha_kernel_builder_t* kb,
                                           const char* const* names,
                                           size_t n)
{
    rocke_arena_t* arena;
    size_t i;

    if(kb == NULL || names == NULL)
    {
        return;
    }
    arena = &kb->b.arena;
    for(i = 0; i < n; ++i)
    {
        const char* sn_token;
        const char* sn_head;
        rocke_value_t* tok;
        rocke_value_t* hd;

        rocke_i_stride_param_names(arena, names[i], &sn_token, &sn_head);
        if(sn_token == NULL || sn_head == NULL)
        {
            kb->b.status = ROCKE_ERR_OOM;
            return;
        }
        /* tok = b.param(sn_token, I32); hd = b.param(sn_head, I32) */
        tok = rocke_b_param(&kb->b, sn_token, rocke_i32(), NULL);
        hd = rocke_b_param(&kb->b, sn_head, rocke_i32(), NULL);

        /* self._stride_params[name] = (tok, hd) */
        if(rocke_i_reserve(arena,
                           (void**)&kb->stride_params,
                           &kb->cap_stride_params,
                           kb->n_stride_params,
                           sizeof(rocke_fmha_stride_pair_t)))
        {
            kb->stride_params[kb->n_stride_params].name = rocke_arena_strdup(arena, names[i]);
            kb->stride_params[kb->n_stride_params].token = tok;
            kb->stride_params[kb->n_stride_params].head = hd;
            kb->n_stride_params++;
        }
        /* self._other_params[sn_token] = tok; self._other_params[sn_head] = hd */
        rocke_i_named_append(
            arena, &kb->other_params, &kb->n_other_params, &kb->cap_other_params, sn_token, tok);
        rocke_i_named_append(
            arena, &kb->other_params, &kb->n_other_params, &kb->cap_other_params, sn_head, hd);
        /* self._sig_order.append(("scalar", sn_token, "i32"))
         * self._sig_order.append(("scalar", sn_head, "i32")) */
        rocke_i_sig_order_append(kb, ROCKE_FMHA_SIG_SCALAR, sn_token, "i32");
        rocke_i_sig_order_append(kb, ROCKE_FMHA_SIG_SCALAR, sn_head, "i32");
    }
}

/* ====================================================================== *
 * FmhaKernelBuilder -- accessors
 * ====================================================================== */

rocke_value_t* rocke_fmha_kernel_builder_tensor(const rocke_fmha_kernel_builder_t* kb,
                                                const char* name)
{
    if(kb == NULL || name == NULL)
    {
        return NULL;
    }
    return rocke_i_named_lookup(kb->tensor_params, kb->n_tensor_params, name);
}

bool rocke_fmha_kernel_builder_stride(const rocke_fmha_kernel_builder_t* kb,
                                      const char* name,
                                      rocke_value_t** out_token,
                                      rocke_value_t** out_head)
{
    const rocke_fmha_stride_pair_t* sp;
    if(kb == NULL || name == NULL)
    {
        return false;
    }
    sp = rocke_i_stride_lookup(kb, name);
    if(sp == NULL)
    {
        return false;
    }
    if(out_token != NULL)
    {
        *out_token = sp->token;
    }
    if(out_head != NULL)
    {
        *out_head = sp->head;
    }
    return true;
}

rocke_value_t* rocke_fmha_kernel_builder_stride_token(const rocke_fmha_kernel_builder_t* kb,
                                                      const char* name)
{
    const rocke_fmha_stride_pair_t* sp;
    if(kb == NULL || name == NULL)
    {
        return NULL;
    }
    sp = rocke_i_stride_lookup(kb, name);
    return (sp != NULL) ? sp->token : NULL;
}

rocke_value_t* rocke_fmha_kernel_builder_stride_head(const rocke_fmha_kernel_builder_t* kb,
                                                     const char* name)
{
    const rocke_fmha_stride_pair_t* sp;
    if(kb == NULL || name == NULL)
    {
        return NULL;
    }
    sp = rocke_i_stride_lookup(kb, name);
    return (sp != NULL) ? sp->head : NULL;
}

rocke_value_t* rocke_fmha_kernel_builder_scalar(const rocke_fmha_kernel_builder_t* kb,
                                                const char* name)
{
    if(kb == NULL || name == NULL)
    {
        return NULL;
    }
    return rocke_i_named_lookup(kb->other_params, kb->n_other_params, name);
}

rocke_value_t* rocke_fmha_kernel_builder_ptr(const rocke_fmha_kernel_builder_t* kb,
                                             const char* name)
{
    if(kb == NULL || name == NULL)
    {
        return NULL;
    }
    return rocke_i_named_lookup(kb->other_params, kb->n_other_params, name);
}

void rocke_fmha_kernel_builder_block_size(rocke_fmha_kernel_builder_t* kb, int block_size)
{
    rocke_kernel_def_t* k;
    if(kb == NULL)
    {
        return;
    }
    /* self.b.kernel.attrs["max_workgroup_size"] = block_size */
    k = rocke_ir_builder_kernel(&kb->b);
    if(k == NULL)
    {
        return;
    }
    rocke_attr_set_int(&kb->b, &k->attrs, "max_workgroup_size", (int64_t)block_size);
}

/* ====================================================================== *
 * FmhaKernelBuilder -- grid decode
 * ====================================================================== */

void rocke_fmha_kernel_builder_appendkv_grid(rocke_fmha_kernel_builder_t* kb,
                                             int block_q,
                                             bool has_batch_axis,
                                             rocke_value_t** out_q_tile_base,
                                             rocke_value_t** out_kv_head_idx)
{
    rocke_ir_builder_t* b;
    if(kb == NULL)
    {
        return;
    }
    b = &kb->b;
    /* self.q_tile_base = b.to_sgpr_u32(b.mul(b.block_id_x(), b.const_i32(block_q))) */
    kb->q_tile_base = rocke_b_to_sgpr_u32(
        b, rocke_b_mul(b, rocke_b_block_id_x(b), rocke_b_const_i32(b, block_q)));
    /* self.kv_head_idx = b.to_sgpr_u32(b.block_id_y()) */
    kb->kv_head_idx = rocke_b_to_sgpr_u32(b, rocke_b_block_id_y(b));
    /* if has_batch_axis: self.batch_idx = b.to_sgpr_u32(b.block_id_z()) */
    if(has_batch_axis)
    {
        kb->batch_idx = rocke_b_to_sgpr_u32(b, rocke_b_block_id_z(b));
    }
    if(out_q_tile_base != NULL)
    {
        *out_q_tile_base = kb->q_tile_base;
    }
    if(out_kv_head_idx != NULL)
    {
        *out_kv_head_idx = kb->kv_head_idx;
    }
}

void rocke_fmha_kernel_builder_decode_grid(rocke_fmha_kernel_builder_t* kb,
                                           int num_queries_per_kv,
                                           bool has_batch_axis,
                                           rocke_value_t** out_q_token,
                                           rocke_value_t** out_head_idx,
                                           rocke_value_t** out_kv_head_idx)
{
    rocke_ir_builder_t* b;
    int nqkv;

    if(out_q_token != NULL)
    {
        *out_q_token = NULL;
    }
    if(out_head_idx != NULL)
    {
        *out_head_idx = NULL;
    }
    if(out_kv_head_idx != NULL)
    {
        *out_kv_head_idx = NULL;
    }
    if(kb == NULL)
    {
        return;
    }
    b = &kb->b;

    /* self.q_token = b.to_sgpr_u32(b.block_id_x()) */
    kb->q_token = rocke_b_to_sgpr_u32(b, rocke_b_block_id_x(b));
    /* self.head_idx = b.to_sgpr_u32(b.block_id_y()) */
    kb->head_idx = rocke_b_to_sgpr_u32(b, rocke_b_block_id_y(b));
    /* if has_batch_axis: self.batch_idx = b.to_sgpr_u32(b.block_id_z()) */
    if(has_batch_axis)
    {
        kb->batch_idx = rocke_b_to_sgpr_u32(b, rocke_b_block_id_z(b));
    }

    /* nqkv = num_queries_per_kv if not None else
     *        self.common.shape.num_queries_per_kv */
    if(num_queries_per_kv >= 0)
    {
        nqkv = num_queries_per_kv;
    }
    else
    {
        rocke_status_t st = rocke_fmha_shape_num_queries_per_kv(&kb->common.shape, &nqkv);
        if(st != ROCKE_OK)
        {
            /* The Python property raises ValueError on a GQA mismatch. */
            kb->b.status = ROCKE_ERR_VALUE;
            return;
        }
    }

    if(nqkv == 1)
    {
        /* self.kv_head_idx = self.head_idx */
        kb->kv_head_idx = kb->head_idx;
    }
    else
    {
        /* self.kv_head_idx = b.to_sgpr_u32(b.div(self.head_idx, b.const_i32(nqkv))) */
        kb->kv_head_idx
            = rocke_b_to_sgpr_u32(b, rocke_b_div(b, kb->head_idx, rocke_b_const_i32(b, nqkv)));
    }

    if(out_q_token != NULL)
    {
        *out_q_token = kb->q_token;
    }
    if(out_head_idx != NULL)
    {
        *out_head_idx = kb->head_idx;
    }
    if(out_kv_head_idx != NULL)
    {
        *out_kv_head_idx = kb->kv_head_idx;
    }
}

/* ====================================================================== *
 * FmhaKernelBuilder -- tensor descriptor factory
 * ====================================================================== */

rocke_tensor_descriptor_t*
    rocke_fmha_kernel_builder_tensor_descriptor(rocke_fmha_kernel_builder_t* kb,
                                                const char* tensor_name,
                                                const char* const* coord_names,
                                                const int* lengths,
                                                int n_lengths)
{
    rocke_arena_t* arena;
    static const char* default_coords[3] = {"token", "head", "d"};
    const char** coords;
    int lens[3];
    const int* lens_ptr;
    const rocke_fmha_stride_pair_t* sp;

    if(kb == NULL || tensor_name == NULL)
    {
        return NULL;
    }
    arena = &kb->b.arena;

    /* coord_names default = ("token", "head", "d") */
    coords = (coord_names != NULL) ? (const char**)coord_names : (const char**)default_coords;

    /* lengths default = (1<<24, 1<<12, max(head_size, 1)) */
    if(lengths != NULL)
    {
        lens_ptr = lengths;
    }
    else
    {
        int hs = kb->common.shape.head_size;
        lens[0] = 1 << 24;
        lens[1] = 1 << 12;
        lens[2] = (hs > 1) ? hs : 1;
        lens_ptr = lens;
        n_lengths = 3;
    }

    sp = rocke_i_stride_lookup(kb, tensor_name);
    if(sp == NULL)
    {
        /* return TensorDescriptor.naive(name, lengths=lengths,
         *                               coord_names=coord_names) */
        return rocke_tensor_descriptor_naive(
            &kb->b, tensor_name, lens_ptr, n_lengths, NULL /* strides */, coords, 3);
    }

    /* return TensorDescriptor(name=tensor_name, base_names=coord_names,
     *   base_lengths=lengths, base_strides=(0,0,1), chain=(),
     *   upper_names=coord_names) */
    {
        rocke_tensor_descriptor_t* d;
        const char** base_names;
        int* base_lengths;
        int* base_strides;
        const char** upper_names;
        int i;

        d = (rocke_tensor_descriptor_t*)rocke_arena_calloc(arena,
                                                           sizeof(rocke_tensor_descriptor_t));
        if(d == NULL)
        {
            kb->b.status = ROCKE_ERR_OOM;
            return NULL;
        }
        base_names = (const char**)rocke_arena_alloc(arena, 3 * sizeof(const char*));
        upper_names = (const char**)rocke_arena_alloc(arena, 3 * sizeof(const char*));
        base_lengths = (int*)rocke_arena_alloc(arena, 3 * sizeof(int));
        base_strides = (int*)rocke_arena_alloc(arena, 3 * sizeof(int));
        if(base_names == NULL || upper_names == NULL || base_lengths == NULL
           || base_strides == NULL)
        {
            kb->b.status = ROCKE_ERR_OOM;
            return NULL;
        }
        for(i = 0; i < 3; ++i)
        {
            base_names[i] = rocke_arena_strdup(arena, coords[i]);
            upper_names[i] = base_names[i];
            base_lengths[i] = (lens_ptr != NULL && i < n_lengths) ? lens_ptr[i] : 0;
        }
        /* base_strides=(0, 0, 1) */
        base_strides[0] = 0;
        base_strides[1] = 0;
        base_strides[2] = 1;

        d->name = rocke_arena_strdup(arena, tensor_name);
        d->base_names = base_names;
        d->base_lengths = base_lengths;
        d->base_strides = base_strides;
        d->n_base = 3;
        d->chain = NULL; /* chain=() */
        d->n_chain = 0;
        d->upper_names = upper_names;
        d->n_upper = 3;
        return d;
    }
}

/* ====================================================================== *
 * FmhaKernelBuilder -- signature builder
 * ====================================================================== */

rocke_status_t rocke_fmha_kernel_builder_signature(rocke_fmha_kernel_builder_t* kb,
                                                   rocke_arena_t* arena,
                                                   const rocke_sig_entry_t** out_items,
                                                   size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t st;
    size_t i;

    if(kb == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }
    /* for kind, name, dtype in self._sig_order:
     *     if kind in ("tensor", "ptr"): sb.ptr(name, dtype)
     *     else: sb.scalar(name, dtype) */
    for(i = 0; i < kb->n_sig_order; ++i)
    {
        const rocke_fmha_sig_order_entry_t* e = &kb->sig_order[i];
        if(e->kind == ROCKE_FMHA_SIG_TENSOR || e->kind == ROCKE_FMHA_SIG_PTR)
        {
            rocke_signature_builder_ptr(&sb, e->name, e->dtype, NULL);
        }
        else
        {
            rocke_signature_builder_scalar(&sb, e->name, e->dtype);
        }
    }
    /* return sb.build() */
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

/* ====================================================================== *
 * FmhaKernelBuilder -- row-base helpers
 * ====================================================================== */

/* _row_base(tensor_name, tok, hd) ->
 *   b.add(b.mul(tok, tok_stride), b.mul(hd, head_stride)) */
static rocke_value_t* rocke_i_row_base(rocke_fmha_kernel_builder_t* kb,
                                       const char* tensor_name,
                                       rocke_value_t* tok,
                                       rocke_value_t* hd)
{
    const rocke_fmha_stride_pair_t* sp;
    rocke_ir_builder_t* b;
    if(kb == NULL || tensor_name == NULL)
    {
        return NULL;
    }
    b = &kb->b;
    sp = rocke_i_stride_lookup(kb, tensor_name);
    if(sp == NULL)
    {
        /* self._stride_params[tensor_name] -> KeyError */
        b->status = ROCKE_ERR_KEY;
        return NULL;
    }
    /* Python evaluates the add() args left-to-right: mul(tok, tok_stride)
     * is emitted before mul(hd, head_stride). C argument evaluation order is
     * unspecified, so sequence the two muls into temporaries to pin the IR
     * emission order to match Python byte-for-byte. */
    {
        rocke_value_t* tok_mul = rocke_b_mul(b, tok, sp->token);
        rocke_value_t* head_mul = rocke_b_mul(b, hd, sp->head);
        return rocke_b_add(b, tok_mul, head_mul);
    }
}

rocke_value_t* rocke_fmha_kernel_builder_q_row_base(rocke_fmha_kernel_builder_t* kb)
{
    if(kb == NULL)
    {
        return NULL;
    }
    return rocke_i_row_base(kb, "q", kb->q_token, kb->head_idx);
}

rocke_value_t* rocke_fmha_kernel_builder_o_row_base(rocke_fmha_kernel_builder_t* kb)
{
    if(kb == NULL)
    {
        return NULL;
    }
    return rocke_i_row_base(kb, "o", kb->q_token, kb->head_idx);
}

rocke_value_t* rocke_fmha_kernel_builder_row_base(rocke_fmha_kernel_builder_t* kb,
                                                  const char* tensor_name,
                                                  rocke_value_t* tok,
                                                  rocke_value_t* hd)
{
    return rocke_i_row_base(kb, tensor_name, tok, hd);
}

rocke_value_t* rocke_fmha_kernel_builder_k_row_base(rocke_fmha_kernel_builder_t* kb,
                                                    rocke_value_t* k_idx)
{
    if(kb == NULL)
    {
        return NULL;
    }
    return rocke_i_row_base(kb, "k", k_idx, kb->kv_head_idx);
}

rocke_value_t* rocke_fmha_kernel_builder_v_row_base(rocke_fmha_kernel_builder_t* kb,
                                                    rocke_value_t* k_idx)
{
    if(kb == NULL)
    {
        return NULL;
    }
    return rocke_i_row_base(kb, "v", k_idx, kb->kv_head_idx);
}
