// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.helpers.fuse.c -- C99 port of a subset of ck_dsl/helpers/fuse.py
 * plus _MultiDEpilogue from ck_dsl/instances/common/gemm_multi_d.py.
 *
 * See ckc/helper_ck_dsl.helpers.fuse.h for the symbol map and the faithfulness
 * contract. Every builder call sequence below mirrors the Python original
 * 1:1 so the emitted IR is byte-identical.
 */

#include "ckc/helper_ck_dsl.helpers.fuse.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_set_err, ckc_i_live */

/* ================================================================== *
 * Dtype dispatch
 * ================================================================== */

const ckc_type_t* ckc_fuse_dtype_to_ir(const ckc_type_t* t)
{
    /* Python: `if isinstance(dtype, Type): return dtype`. */
    return t;
}

/* Lower-case a bounded copy of `s`, stripping a leading "torch." prefix, into
 * `buf` (capacity `cap`). Mirrors Python:
 *   s = str(dtype).lower()
 *   if s.startswith("torch."): s = s.split(".", 1)[-1]
 * Returns the normalised string, or NULL if it does not fit. */
static const char* ckc_fuse_normalise(const char* s, char* buf, size_t cap)
{
    size_t i;
    size_t n;
    if(s == NULL)
    {
        return NULL;
    }
    n = strlen(s);
    if(n + 1 > cap)
    {
        return NULL;
    }
    for(i = 0; i < n; ++i)
    {
        buf[i] = (char)tolower((unsigned char)s[i]);
    }
    buf[n] = '\0';
    /* s.startswith("torch.") -> s.split(".", 1)[-1]: everything after the FIRST
     * '.', which for "torch." is index 5; return the tail at index 6. */
    if(strncmp(buf, "torch.", 6) == 0)
    {
        return buf + 6;
    }
    return buf;
}

const ckc_type_t* ckc_fuse_dtype_to_ir_str(const char* dtype)
{
    char buf[64];
    const char* s = ckc_fuse_normalise(dtype, buf, sizeof(buf));
    if(s == NULL)
    {
        return NULL;
    }
    /* _DTYPE_STR_TO_IR table (fuse.py). */
    if(strcmp(s, "fp16") == 0 || strcmp(s, "f16") == 0 || strcmp(s, "half") == 0
       || strcmp(s, "float16") == 0)
    {
        return ckc_f16();
    }
    if(strcmp(s, "bf16") == 0 || strcmp(s, "bfloat16") == 0)
    {
        return ckc_bf16();
    }
    if(strcmp(s, "fp32") == 0 || strcmp(s, "f32") == 0 || strcmp(s, "float") == 0
       || strcmp(s, "float32") == 0)
    {
        return ckc_f32();
    }
    return NULL; /* Python: raise ValueError */
}

const ckc_type_t* ckc_fuse_b_dtype_to_ir_str(ckc_ir_builder_t* b, const char* dtype)
{
    const ckc_type_t* ty;
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    ty = ckc_fuse_dtype_to_ir_str(dtype);
    if(ty == NULL)
    {
        /* Mirror `raise ValueError(f"unsupported epilogue dtype {dtype!r}")`. */
        return (const ckc_type_t*)ckc_i_set_err(b,
                                                CKC_ERR_VALUE,
                                                "unsupported epilogue dtype %s%s%s",
                                                dtype ? "'" : "",
                                                dtype ? dtype : "None",
                                                dtype ? "'" : "");
    }
    return ty;
}

ckc_value_t* ckc_fuse_ir_dtype_zero(ckc_ir_builder_t* b, const ckc_type_t* dtype)
{
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    /* if dtype == F16: return b.trunc_f32_to_f16(b.const_f32(0.0)) */
    if(ckc_type_eq(dtype, ckc_f16()))
    {
        return ckc_b_trunc_f32_to_f16(b, ckc_b_const_f32(b, 0.0));
    }
    /* if dtype == BF16: ... try trunc_f32_to_bf16; the C builder lacks it, so
     * always the AttributeError fallback: b.cast_f32_to(b.const_f32(0.0), BF16) */
    if(ckc_type_eq(dtype, ckc_bf16()))
    {
        return ckc_b_cast_f32_to(b, ckc_b_const_f32(b, 0.0), ckc_bf16());
    }
    /* if dtype == F32: return b.const_f32(0.0) */
    if(ckc_type_eq(dtype, ckc_f32()))
    {
        return ckc_b_const_f32(b, 0.0);
    }
    /* raise NotImplementedError(f"ir_dtype_zero: unsupported {dtype}") */
    return (ckc_value_t*)ckc_i_set_err(b,
                                       CKC_ERR_NOTIMPL,
                                       "ir_dtype_zero: unsupported %s",
                                       (dtype && dtype->name) ? dtype->name : "None");
}

ckc_value_t* ckc_fuse_ir_dtype_const(ckc_ir_builder_t* b, const ckc_type_t* dtype, double value)
{
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    /* Python casts the value via float(value) before const_f32. */
    if(ckc_type_eq(dtype, ckc_f16()))
    {
        return ckc_b_trunc_f32_to_f16(b, ckc_b_const_f32(b, value));
    }
    if(ckc_type_eq(dtype, ckc_bf16()))
    {
        return ckc_b_cast_f32_to(b, ckc_b_const_f32(b, value), ckc_bf16());
    }
    if(ckc_type_eq(dtype, ckc_f32()))
    {
        return ckc_b_const_f32(b, value);
    }
    return (ckc_value_t*)ckc_i_set_err(b,
                                       CKC_ERR_NOTIMPL,
                                       "ir_dtype_const: unsupported %s",
                                       (dtype && dtype->name) ? dtype->name : "None");
}

ckc_value_t* ckc_fuse_ir_dtype_global_load(ckc_ir_builder_t* b,
                                           const ckc_type_t* dtype,
                                           ckc_value_t* ptr,
                                           ckc_value_t* idx)
{
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    /* Python: b.global_load_f16(ptr, idx) (no align kwarg -> typed default). */
    if(ckc_type_eq(dtype, ckc_f16()))
    {
        return ckc_b_global_load_f16(b, ptr, idx, 0);
    }
    if(ckc_type_eq(dtype, ckc_bf16()))
    {
        return ckc_b_global_load_bf16(b, ptr, idx, 0);
    }
    if(ckc_type_eq(dtype, ckc_f32()))
    {
        return ckc_b_global_load_f32(b, ptr, idx, 0);
    }
    return (ckc_value_t*)ckc_i_set_err(b,
                                       CKC_ERR_NOTIMPL,
                                       "ir_dtype_global_load: unsupported %s",
                                       (dtype && dtype->name) ? dtype->name : "None");
}

/* ================================================================== *
 * EpilogueOp constructors / tag / declare_params / apply_element
 * ================================================================== */

ckc_epilogue_op_t ckc_residual_add(const char* param_name, const ckc_type_t* dtype)
{
    ckc_epilogue_op_t op;
    op.kind = CKC_EOP_RESADD;
    op.param_name = (param_name != NULL) ? param_name : "residual";
    op.dtype = (dtype != NULL) ? dtype : ckc_f16();
    return op;
}

ckc_epilogue_op_t ckc_residual_mul(const char* param_name, const ckc_type_t* dtype)
{
    ckc_epilogue_op_t op;
    op.kind = CKC_EOP_RESMUL;
    op.param_name = (param_name != NULL) ? param_name : "residual_mul";
    op.dtype = (dtype != NULL) ? dtype : ckc_f16();
    return op;
}

ckc_status_t ckc_epilogue_op_tag(const ckc_epilogue_op_t* op, char* out, size_t out_cap)
{
    const char* prefix;
    const char* dn;
    int n;
    if(op == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }
    if(op->kind == CKC_EOP_RESADD)
    {
        prefix = "resadd_";
    }
    else if(op->kind == CKC_EOP_RESMUL)
    {
        prefix = "resmul_";
    }
    else
    {
        return CKC_ERR_VALUE; /* no ported tag for non-residual ops */
    }
    dn = (op->dtype && op->dtype->name) ? op->dtype->name : "";
    n = snprintf(out, out_cap, "%s%s", prefix, dn);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

ckc_value_t* ckc_epilogue_op_declare_params(ckc_ir_builder_t* b, const ckc_epilogue_op_t* op)
{
    ckc_param_opts_t opts;
    const ckc_type_t* ptr_t;
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    if(op == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "declare_params: NULL op");
    }
    /* p = b.param(name, PtrType(dtype, "global"),
     *             noalias=True, readonly=True, align=16) */
    ptr_t = ckc_ptr_type(b, op->dtype, "global");
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 16;
    opts.align_set = true;
    return ckc_b_param(b, op->param_name, ptr_t, &opts);
}

ckc_value_t* ckc_epilogue_op_apply_element(ckc_ir_builder_t* b,
                                           const ckc_epilogue_op_t* op,
                                           ckc_value_t* v,
                                           ckc_value_t* m,
                                           ckc_value_t* n,
                                           int elem_idx,
                                           const ckc_fe_params_t* params)
{
    ckc_value_t* ptr;
    ckc_value_t* n_idx;
    ckc_value_t* stride_m;
    ckc_value_t* off;
    ckc_value_t* r;
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    if(op == NULL || params == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_element: NULL op/params");
    }

    ptr = ckc_fe_params_get(params, op->param_name);
    /* n_idx = b.add(n, b.const_i32(elem_idx)) */
    n_idx = ckc_b_add(b, n, ckc_b_const_i32(b, elem_idx));

    if(op->kind == CKC_EOP_RESADD)
    {
        /* stride_m = params.get("__stride_m"); if None: params["__N"] */
        stride_m = ckc_fe_params_get(params, "__stride_m");
        if(stride_m == NULL)
        {
            stride_m = ckc_fe_params_get(params, "__N");
        }
        off = ckc_b_add(b, ckc_b_mul(b, m, stride_m), n_idx);
        r = ckc_fuse_ir_dtype_global_load(b, op->dtype, ptr, off);
        return ckc_b_fadd(b, v, r);
    }
    if(op->kind == CKC_EOP_RESMUL)
    {
        /* stride_m = params.get("__stride_m") or params["__N"] */
        stride_m = ckc_fe_params_get(params, "__stride_m");
        if(stride_m == NULL)
        {
            stride_m = ckc_fe_params_get(params, "__N");
        }
        off = ckc_b_add(b, ckc_b_mul(b, m, stride_m), n_idx);
        r = ckc_fuse_ir_dtype_global_load(b, op->dtype, ptr, off);
        return ckc_b_fmul(b, v, r);
    }
    /* EpilogueOp.apply_element base: raise NotImplementedError */
    return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_NOTIMPL, "apply_element: unsupported op kind");
}

/* ================================================================== *
 * Live-params table
 * ================================================================== */

void ckc_fe_params_init(ckc_fe_params_t* p)
{
    if(p != NULL)
    {
        p->count = 0;
    }
}

void ckc_fe_params_set(ckc_fe_params_t* p, const char* name, ckc_value_t* value)
{
    size_t i;
    if(p == NULL || name == NULL)
    {
        return;
    }
    for(i = 0; i < p->count; ++i)
    {
        if(strcmp(p->entries[i].name, name) == 0)
        {
            p->entries[i].value = value; /* dict overwrite */
            return;
        }
    }
    if(p->count >= CKC_FE_MAX_PARAMS)
    {
        return; /* capacity guard; chains never exceed CKC_FE_MAX_PARAMS */
    }
    p->entries[p->count].name = name;
    p->entries[p->count].value = value;
    p->count += 1;
}

ckc_value_t* ckc_fe_params_get(const ckc_fe_params_t* p, const char* name)
{
    size_t i;
    if(p == NULL || name == NULL)
    {
        return NULL;
    }
    for(i = 0; i < p->count; ++i)
    {
        if(strcmp(p->entries[i].name, name) == 0)
        {
            return p->entries[i].value;
        }
    }
    return NULL; /* dict.get -> None */
}

/* ================================================================== *
 * FusedEpilogue
 * ================================================================== */

void ckc_fe_init(ckc_fused_epilogue_t* fe,
                 const ckc_epilogue_op_t* ops,
                 size_t num_ops,
                 const ckc_type_t* dtype)
{
    if(fe == NULL)
    {
        return;
    }
    fe->ops = ops;
    fe->num_ops = num_ops;
    fe->dtype = (dtype != NULL) ? dtype : ckc_f16();
    ckc_fe_params_init(&fe->params);
}

const ckc_type_t* ckc_fe_ir_dtype(const ckc_fused_epilogue_t* fe)
{
    if(fe == NULL)
    {
        return NULL;
    }
    /* dtype_to_ir(self.dtype): dtype is already a Type -> identity. */
    return ckc_fuse_dtype_to_ir(fe->dtype);
}

ckc_status_t ckc_fe_declare_params(ckc_ir_builder_t* b, ckc_fused_epilogue_t* fe)
{
    size_t i;
    if(!ckc_i_live(b))
    {
        return b ? b->status : CKC_ERR_VALUE;
    }
    if(fe == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "declare_params: NULL epilogue");
        return CKC_ERR_VALUE;
    }
    /* for op in self.ops:
     *   for name, val in op.declare_params(b).items():
     *     self._live_params[name] = val */
    for(i = 0; i < fe->num_ops; ++i)
    {
        ckc_value_t* val = ckc_epilogue_op_declare_params(b, &fe->ops[i]);
        if(!ckc_i_live(b))
        {
            return b->status;
        }
        ckc_fe_params_set(&fe->params, fe->ops[i].param_name, val);
    }
    return CKC_OK;
}

void ckc_fe_record_runtime(ckc_fused_epilogue_t* fe, ckc_value_t* N, ckc_value_t* stride_m)
{
    if(fe == NULL)
    {
        return;
    }
    /* self._live_params["__N"] = N */
    ckc_fe_params_set(&fe->params, "__N", N);
    /* if stride_m is not None: self._live_params["__stride_m"] = stride_m */
    if(stride_m != NULL)
    {
        ckc_fe_params_set(&fe->params, "__stride_m", stride_m);
    }
}

ckc_value_t* ckc_fe_apply_scalar(
    ckc_ir_builder_t* b, ckc_fused_epilogue_t* fe, ckc_value_t* v, ckc_value_t* m, ckc_value_t* n)
{
    size_t i;
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    if(fe == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_scalar: NULL epilogue");
    }
    for(i = 0; i < fe->num_ops; ++i)
    {
        v = ckc_epilogue_op_apply_element(b, &fe->ops[i], v, m, n, 0, &fe->params);
    }
    return v;
}

ckc_value_t* ckc_fe_apply_vec(ckc_ir_builder_t* b,
                              ckc_fused_epilogue_t* fe,
                              ckc_value_t* v,
                              ckc_value_t* m,
                              ckc_value_t* n,
                              int n_elems)
{
    /* out[i] runs the chain on vec_extract(v, i); then vec_pack(out, dtype).
     * 64 lanes bounds the largest realistic vector store. */
    ckc_value_t* out[64];
    int i;
    size_t j;
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    if(fe == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_vec: NULL epilogue");
    }
    if(n_elems < 0 || n_elems > (int)(sizeof(out) / sizeof(out[0])))
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "apply_vec: n_elems out of range");
    }
    for(i = 0; i < n_elems; ++i)
    {
        ckc_value_t* scalar = ckc_b_vec_extract(b, v, i);
        for(j = 0; j < fe->num_ops; ++j)
        {
            scalar = ckc_epilogue_op_apply_element(b, &fe->ops[j], scalar, m, n, i, &fe->params);
        }
        out[i] = scalar;
    }
    return ckc_b_vec_pack(b, out, n_elems, ckc_fe_ir_dtype(fe));
}

ckc_status_t ckc_fe_kernel_name_suffix(const ckc_fused_epilogue_t* fe, char* out, size_t out_cap)
{
    size_t i;
    size_t used = 0;
    if(fe == NULL || out == NULL || out_cap == 0)
    {
        return CKC_ERR_VALUE;
    }
    /* "_".join(op.tag() for op in self.ops) or "id" */
    if(fe->num_ops == 0)
    {
        if(out_cap < 3)
        {
            return CKC_ERR_VALUE;
        }
        out[0] = 'i';
        out[1] = 'd';
        out[2] = '\0';
        return CKC_OK;
    }
    out[0] = '\0';
    for(i = 0; i < fe->num_ops; ++i)
    {
        char tagbuf[64];
        int n;
        ckc_status_t st = ckc_epilogue_op_tag(&fe->ops[i], tagbuf, sizeof(tagbuf));
        if(st != CKC_OK)
        {
            return st;
        }
        if(i == 0)
        {
            n = snprintf(out + used, out_cap - used, "%s", tagbuf);
        }
        else
        {
            n = snprintf(out + used, out_cap - used, "_%s", tagbuf);
        }
        if(n < 0 || (size_t)n >= out_cap - used)
        {
            return CKC_ERR_VALUE;
        }
        used += (size_t)n;
    }
    return CKC_OK;
}

/* ================================================================== *
 * _MultiDEpilogue
 * ================================================================== */

ckc_status_t ckc_mde_from_ops(ckc_multi_d_epilogue_t* mde,
                              const ckc_epilogue_op_t* ops,
                              size_t num_ops,
                              const ckc_type_t* dtype,
                              ckc_mde_load_kind_t load_kind)
{
    size_t i;
    if(mde == NULL)
    {
        return CKC_ERR_VALUE;
    }
    if(num_ops > CKC_FE_MAX_PARAMS)
    {
        return CKC_ERR_VALUE;
    }
    ckc_fe_init(&mde->base, ops, num_ops, dtype);
    mde->load_kind = load_kind;
    /* Classify each op: ResidualAdd->("add", dtype_to_ir(op.dtype)),
     * ResidualMul->("mul", ...), else (None, None). We mirror the kind onto
     * residual_kinds and resolve the dtype (already a Type on the op). */
    for(i = 0; i < num_ops; ++i)
    {
        if(ops[i].kind == CKC_EOP_RESADD)
        {
            mde->residual_kinds[i] = CKC_EOP_RESADD;
            mde->residual_dtypes[i] = ckc_fuse_dtype_to_ir(ops[i].dtype);
        }
        else if(ops[i].kind == CKC_EOP_RESMUL)
        {
            mde->residual_kinds[i] = CKC_EOP_RESMUL;
            mde->residual_dtypes[i] = ckc_fuse_dtype_to_ir(ops[i].dtype);
        }
        else
        {
            mde->residual_kinds[i] = CKC_EOP_OTHER; /* Python None */
            mde->residual_dtypes[i] = NULL; /* Python None */
        }
    }
    return CKC_OK;
}

ckc_value_t* ckc_mde_apply_vec(ckc_ir_builder_t* b,
                               ckc_multi_d_epilogue_t* mde,
                               ckc_value_t* v,
                               ckc_value_t* m,
                               ckc_value_t* n,
                               int n_elems)
{
    bool vectorize;
    bool all_residual;
    size_t k;
    ckc_value_t* stride_m;
    ckc_value_t* off_base;
    const ckc_type_t* ir_dtype;
    ckc_value_t* out[64];
    ckc_value_t* per_d_vecs[CKC_FE_MAX_PARAMS];
    int i;

    if(!ckc_i_live(b))
    {
        return NULL;
    }
    if(mde == NULL)
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "mde apply_vec: NULL epilogue");
    }

    /* vectorize = self._load_kind == "vector" */
    vectorize = (mde->load_kind == CKC_MDE_VECTOR);

    /* if vectorize and n_elems not in (2, 4, 8): return super().apply_vec(...) */
    if(vectorize && !(n_elems == 2 || n_elems == 4 || n_elems == 8))
    {
        return ckc_fe_apply_vec(b, &mde->base, v, m, n, n_elems);
    }

    /* if not all(k is not None for k in self._residual_kinds):
     *     return super().apply_vec(...) */
    all_residual = true;
    for(k = 0; k < mde->base.num_ops; ++k)
    {
        if(mde->residual_kinds[k] == CKC_EOP_OTHER) /* Python None */
        {
            all_residual = false;
            break;
        }
    }
    if(!all_residual)
    {
        return ckc_fe_apply_vec(b, &mde->base, v, m, n, n_elems);
    }

    /* stride_m = self._live_params.get("__stride_m") or self._live_params["__N"] */
    stride_m = ckc_fe_params_get(&mde->base.params, "__stride_m");
    if(stride_m == NULL)
    {
        stride_m = ckc_fe_params_get(&mde->base.params, "__N");
    }
    /* off_base = b.add(b.mul(m, stride_m), n) */
    off_base = ckc_b_add(b, ckc_b_mul(b, m, stride_m), n);
    ir_dtype = ckc_fe_ir_dtype(&mde->base);

    if(n_elems > (int)(sizeof(out) / sizeof(out[0])))
    {
        return (ckc_value_t*)ckc_i_set_err(b, CKC_ERR_VALUE, "mde apply_vec: n_elems too large");
    }

    if(!vectorize)
    {
        /* "tiled": per-element scalar D loads from the hoisted base. */
        for(i = 0; i < n_elems; ++i)
        {
            ckc_value_t* scalar = ckc_b_vec_extract(b, v, i);
            ckc_value_t* off_i
                = (i == 0) ? off_base : ckc_b_add(b, off_base, ckc_b_const_i32(b, i));
            size_t op_idx;
            for(op_idx = 0; op_idx < mde->base.num_ops; ++op_idx)
            {
                const ckc_type_t* dt = mde->residual_dtypes[op_idx];
                ckc_value_t* ptr
                    = ckc_fe_params_get(&mde->base.params, mde->base.ops[op_idx].param_name);
                ckc_value_t* r = ckc_fuse_ir_dtype_global_load(b, dt, ptr, off_i);
                if(mde->residual_kinds[op_idx] == CKC_EOP_RESADD) /* "add" */
                {
                    scalar = ckc_b_fadd(b, scalar, r);
                }
                else /* "mul" */
                {
                    scalar = ckc_b_fmul(b, scalar, r);
                }
            }
            out[i] = scalar;
        }
        return ckc_b_vec_pack(b, out, n_elems, ir_dtype);
    }

    /* "vector": one global_load_vN per D operand per slice. */
    for(k = 0; k < mde->base.num_ops; ++k)
    {
        const ckc_type_t* dt = mde->residual_dtypes[k];
        ckc_value_t* ptr = ckc_fe_params_get(&mde->base.params, mde->base.ops[k].param_name);
        ckc_value_t* dv;
        const char* dn = (dt && dt->name) ? dt->name : "";
        if(strcmp(dn, "f16") == 0 || strcmp(dn, "bf16") == 0)
        {
            dv = ckc_b_global_load_vN(b, ptr, off_base, dt, n_elems, 0);
        }
        else if(strcmp(dn, "f32") == 0)
        {
            ckc_value_t* scalars[64];
            int s;
            for(s = 0; s < n_elems; ++s)
            {
                scalars[s] = ckc_fuse_ir_dtype_global_load(
                    b, dt, ptr, ckc_b_add(b, off_base, ckc_b_const_i32(b, s)));
            }
            dv = ckc_b_vec_pack(b, scalars, n_elems, dt);
        }
        else
        {
            return ckc_fe_apply_vec(b, &mde->base, v, m, n, n_elems);
        }
        per_d_vecs[k] = dv;
    }

    for(i = 0; i < n_elems; ++i)
    {
        ckc_value_t* scalar = ckc_b_vec_extract(b, v, i);
        size_t op_idx;
        for(op_idx = 0; op_idx < mde->base.num_ops; ++op_idx)
        {
            ckc_value_t* d_elem = ckc_b_vec_extract(b, per_d_vecs[op_idx], i);
            if(mde->residual_kinds[op_idx] == CKC_EOP_RESADD) /* "add" */
            {
                scalar = ckc_b_fadd(b, scalar, d_elem);
            }
            else /* "mul" */
            {
                scalar = ckc_b_fmul(b, scalar, d_elem);
            }
        }
        out[i] = scalar;
    }
    return ckc_b_vec_pack(b, out, n_elems, ir_dtype);
}
