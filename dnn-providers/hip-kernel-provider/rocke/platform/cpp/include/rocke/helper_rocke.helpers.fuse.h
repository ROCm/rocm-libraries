/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.fuse.h -- C99 port of a SUBSET of
 * rocke/helpers/fuse.py (plus _MultiDEpilogue, which physically lives in
 * rocke/instances/common/gemm_multi_d.py but is a FusedEpilogue subclass and
 * is ported here next to its base).
 *
 *   Python                                  C99 (this header)
 *   -------------------------------------   -------------------------------------
 *   dtype_to_ir(dtype)                      rocke_fuse_dtype_to_ir / _str / _b_str
 *   ir_dtype_zero(b, dtype)                 rocke_fuse_ir_dtype_zero
 *   ir_dtype_const(b, dtype, value)         rocke_fuse_ir_dtype_const
 *   ir_dtype_global_load(b, dtype, p, i)    rocke_fuse_ir_dtype_global_load
 *   class ResidualAdd(EpilogueOp)           rocke_residual_add_* (kind RESADD)
 *   class ResidualMul(EpilogueOp)           rocke_residual_mul_* (kind RESMUL)
 *   class EpilogueOp                        rocke_epilogue_op_t (tagged vtable)
 *   class FusedEpilogue                     rocke_fused_epilogue_t + rocke_fe_*
 *   class _MultiDEpilogue(FusedEpilogue)    rocke_multi_d_epilogue_t + rocke_mde_*
 *
 * Faithfulness contract: every function reproduces the EXACT IRBuilder call
 * sequence of its Python original, so the lowered IR is byte-identical. Where
 * Python raises (ValueError / NotImplementedError) we use the builder's sticky
 * error (rocke_i_set_err) and return NULL, matching the rest of the C port.
 *
 * Object model: Python's frozen dataclasses + duck-typed EpilogueOp chain are
 * modelled as POD structs. EpilogueOp dispatch (apply_element / declare_params
 * / tag) is a small kind-tag enum: only the symbols requested for this port
 * (ResidualAdd, ResidualMul) have concrete behaviour; other kinds are reserved
 * so a heterogeneous chain can still be represented and classified, and the
 * _MultiDEpilogue fast paths fall back exactly like Python on unknown kinds.
 *
 * The Python `_live_params` dict (param_name -> SSA Value, plus the magic keys
 * "__N" and "__stride_m") is modelled by rocke_fe_params_t. declare_params /
 * record_runtime populate it; apply_* read it.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_FUSE_H
#define ROCKE_HELPER_ROCKE_HELPERS_FUSE_H

#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================== *
 * Dtype dispatch (fuse.py "Dtype dispatch" section)
 * ================================================================== */

/* dtype_to_ir when the argument is ALREADY an IR Type: the Python
 * `if isinstance(dtype, Type): return dtype` fast path. Returns `t` unchanged
 * (NULL stays NULL). */
const rocke_type_t* rocke_fuse_dtype_to_ir(const rocke_type_t* t);

/* dtype_to_ir for the string spelling path. Resolves the canonical CK Tile
 * strings + aliases to F16/BF16/F32, applying the same normalisation Python
 * does: lower-cased, and a leading "torch." prefix stripped. Recognised:
 *
 *   F16  : "fp16","f16","half","float16"
 *   BF16 : "bf16","bfloat16"
 *   F32  : "fp32","f32","float","float32"
 *
 * Returns NULL for an unsupported string (Python `raise ValueError`); no
 * builder is involved so no error state is set. */
const rocke_type_t* rocke_fuse_dtype_to_ir_str(const char* dtype);

/* Builder-aware string variant: same mapping as rocke_fuse_dtype_to_ir_str, but
 * on an unsupported string records ROCKE_ERR_VALUE with the Python-matching
 * message ("unsupported epilogue dtype <repr>") and returns NULL. No-op
 * returning NULL on an already-failed builder. */
const rocke_type_t* rocke_fuse_b_dtype_to_ir_str(rocke_ir_builder_t* b, const char* dtype);

/* ir_dtype_zero(b, dtype): element zero in `dtype`.
 *   F16  -> trunc_f32_to_f16(const_f32(0.0))
 *   BF16 -> cast_f32_to(const_f32(0.0), BF16)   (the Python AttributeError
 *           fallback path: the C builder has no trunc_f32_to_bf16)
 *   F32  -> const_f32(0.0)
 * Any other type sets ROCKE_ERR_NOTIMPL (Python NotImplementedError). */
rocke_value_t* rocke_fuse_ir_dtype_zero(rocke_ir_builder_t* b, const rocke_type_t* dtype);

/* ir_dtype_const(b, dtype, value): per-element constant in `dtype`. Same
 * dispatch as rocke_fuse_ir_dtype_zero with `value` in place of 0.0; ROCKE_ERR_NOTIMPL
 * on an unsupported type. */
rocke_value_t*
    rocke_fuse_ir_dtype_const(rocke_ir_builder_t* b, const rocke_type_t* dtype, double value);

/* ir_dtype_global_load(b, dtype, ptr, idx): single-element global load.
 *   F16 -> global_load_f16, BF16 -> global_load_bf16, F32 -> global_load_f32.
 * (The Python helper passes no align kwarg; the typed loaders default it, so
 * we pass align=0 = "use default" here too.) ROCKE_ERR_NOTIMPL otherwise. */
rocke_value_t* rocke_fuse_ir_dtype_global_load(rocke_ir_builder_t* b,
                                               const rocke_type_t* dtype,
                                               rocke_value_t* ptr,
                                               rocke_value_t* idx);

/* ================================================================== *
 * EpilogueOp chain (tagged)
 * ================================================================== */

/* The EpilogueOp kinds this port models. Only RESADD / RESMUL have ported
 * behaviour (ResidualAdd / ResidualMul); ROCKE_EOP_OTHER is the catch-all for any
 * op the fast paths must treat as "non-residual" (forcing the generic
 * per-element fallback, like Python's `isinstance` checks failing). */
typedef enum rocke_epilogue_op_kind
{
    ROCKE_EOP_OTHER = 0, /* any non-residual op (Cast/ReLU/GELU/...)            */
    ROCKE_EOP_RESADD, /* ResidualAdd                                         */
    ROCKE_EOP_RESMUL /* ResidualMul                                         */
} rocke_epilogue_op_kind_t;

/* One step in a fused op chain.
 *
 * Mirrors the frozen dataclass fields that ResidualAdd / ResidualMul carry:
 *   param_name -> .param_name   ("residual" / "residual_mul" defaults)
 *   dtype      -> .dtype        (resolved IR Type; the _ir_dtype() result)
 *
 * `dtype` here is the ALREADY-resolved IR Type (Python's op._ir_dtype()), so
 * callers building these from a string dtype should run rocke_fuse_dtype_to_ir_str
 * first -- matching with_dtype() propagation. */
typedef struct rocke_epilogue_op
{
    rocke_epilogue_op_kind_t kind;
    const char* param_name; /* borrowed; must outlive the op                */
    const rocke_type_t* dtype; /* resolved element type (op._ir_dtype())       */
} rocke_epilogue_op_t;

/* Construct a ResidualAdd op. `param_name` NULL => the Python default
 * "residual"; `dtype` NULL => F16 (the dataclass default). */
rocke_epilogue_op_t rocke_residual_add(const char* param_name, const rocke_type_t* dtype);

/* Construct a ResidualMul op. `param_name` NULL => "residual_mul"; `dtype`
 * NULL => F16. */
rocke_epilogue_op_t rocke_residual_mul(const char* param_name, const rocke_type_t* dtype);

/* ResidualAdd/.tag() / ResidualMul.tag():
 *   "resadd_<dtype.name>" / "resmul_<dtype.name>".
 * Written NUL-terminated into out (capacity out_cap). ROCKE_OK on success;
 * ROCKE_ERR_VALUE if the buffer is too small or the op kind has no ported tag. */
rocke_status_t rocke_epilogue_op_tag(const rocke_epilogue_op_t* op, char* out, size_t out_cap);

/* ResidualAdd.declare_params / ResidualMul.declare_params: declare the residual
 * pointer param. Equivalent to:
 *
 *   b.param(param_name, PtrType(dtype, "global"),
 *           noalias=True, readonly=True, align=16)
 *
 * Returns the declared SSA Value (NULL on a failed builder). The caller stores
 * it into the chain's params under `op->param_name` (rocke_fe_declare_params does
 * this for a whole chain). */
rocke_value_t* rocke_epilogue_op_declare_params(rocke_ir_builder_t* b,
                                                const rocke_epilogue_op_t* op);

/* ================================================================== *
 * Live-params table  (Python FusedEpilogue._live_params)
 * ================================================================== */

/* Capacity is small and fixed: at most one entry per chain op plus the two
 * magic runtime keys "__N" / "__stride_m". MAX_D in gemm_multi_d.py is 8. */
#define ROCKE_FE_MAX_PARAMS 16

typedef struct rocke_fe_param_entry
{
    const char* name; /* borrowed key (op param_name / "__N" / "__stride_m") */
    rocke_value_t* value; /* SSA value                                           */
} rocke_fe_param_entry_t;

typedef struct rocke_fe_params
{
    rocke_fe_param_entry_t entries[ROCKE_FE_MAX_PARAMS];
    size_t count;
} rocke_fe_params_t;

/* Reset a params table to empty. */
void rocke_fe_params_init(rocke_fe_params_t* p);

/* Insert/overwrite `name`->`value` (dict assignment semantics). NULL value is
 * stored as-is (mirrors Python storing None, e.g. an unset "__stride_m"). */
void rocke_fe_params_set(rocke_fe_params_t* p, const char* name, rocke_value_t* value);

/* dict.get(name): the stored value, or NULL if absent. */
rocke_value_t* rocke_fe_params_get(const rocke_fe_params_t* p, const char* name);

/* ================================================================== *
 * FusedEpilogue  (fuse.py)
 * ================================================================== */

/* A chain of EpilogueOps + the element dtype + the live-params table. The op
 * array is borrowed (caller owns storage that must outlive the epilogue). */
typedef struct rocke_fused_epilogue
{
    const rocke_epilogue_op_t* ops; /* borrowed array                           */
    size_t num_ops;
    const rocke_type_t* dtype; /* element type (FusedEpilogue.dtype)        */
    rocke_fe_params_t params; /* _live_params                             */
} rocke_fused_epilogue_t;

/* Initialise a FusedEpilogue over `ops`. `dtype` NULL => F16. Clears params. */
void rocke_fe_init(rocke_fused_epilogue_t* fe,
                   const rocke_epilogue_op_t* ops,
                   size_t num_ops,
                   const rocke_type_t* dtype);

/* FusedEpilogue._ir_dtype(): dtype_to_ir(self.dtype) -- here dtype is already a
 * Type, so an identity (NULL stays NULL). */
const rocke_type_t* rocke_fe_ir_dtype(const rocke_fused_epilogue_t* fe);

/* FusedEpilogue.declare_params(b): walk every op's declare_params, accumulating
 * the SSA values into _live_params keyed by op param_name. Returns ROCKE_OK / the
 * builder error. (Python returns a dict copy; here the table lives on `fe`.) */
rocke_status_t rocke_fe_declare_params(rocke_ir_builder_t* b, rocke_fused_epilogue_t* fe);

/* FusedEpilogue.record_runtime(b, N=, stride_m=): store "__N"=N always, and
 * "__stride_m"=stride_m only when stride_m != NULL (Python's `if stride_m is
 * not None`). */
void rocke_fe_record_runtime(rocke_fused_epilogue_t* fe, rocke_value_t* N, rocke_value_t* stride_m);

/* FusedEpilogue.apply_scalar(b, v, m, n): apply the chain to one scalar via
 * each op's apply_element (elem_idx=0). Returns the transformed scalar. */
rocke_value_t* rocke_fe_apply_scalar(rocke_ir_builder_t* b,
                                     rocke_fused_epilogue_t* fe,
                                     rocke_value_t* v,
                                     rocke_value_t* m,
                                     rocke_value_t* n);

/* FusedEpilogue.apply_vec(b, v, m, n, n_elems): for each of n_elems lanes,
 * vec_extract -> run the chain element-wise -> collect, then vec_pack(out,
 * _ir_dtype()). Returns the packed vector. */
rocke_value_t* rocke_fe_apply_vec(rocke_ir_builder_t* b,
                                  rocke_fused_epilogue_t* fe,
                                  rocke_value_t* v,
                                  rocke_value_t* m,
                                  rocke_value_t* n,
                                  int n_elems);

/* FusedEpilogue.kernel_name_suffix(): "_".join(op.tag() for op in ops) or "id"
 * (the empty-chain case yields "id"). NUL-terminated into out/out_cap. ROCKE_OK
 * or ROCKE_ERR_VALUE (buffer too small / an op has no ported tag). */
rocke_status_t
    rocke_fe_kernel_name_suffix(const rocke_fused_epilogue_t* fe, char* out, size_t out_cap);

/* The single ported EpilogueOp.apply_element used by the chain above. Dispatches
 * on op->kind:
 *   RESADD: n_idx = add(n, const_i32(elem_idx));
 *           stride_m = params.get("__stride_m");  if None: params["__N"];
 *           off = add(mul(m, stride_m), n_idx);
 *           r = ir_dtype_global_load(b, dtype, ptr, off);
 *           return fadd(v, r)
 *   RESMUL: identical but stride_m = params.get("__stride_m") or params["__N"]
 *           (Python `or`, so a None/zero stride falls back to __N), and the
 *           combine is fmul.
 * RESADD vs RESMUL reproduce the two source spellings byte-faithfully.
 * ROCKE_EOP_OTHER has no ported element behaviour -> ROCKE_ERR_NOTIMPL. */
rocke_value_t* rocke_epilogue_op_apply_element(rocke_ir_builder_t* b,
                                               const rocke_epilogue_op_t* op,
                                               rocke_value_t* v,
                                               rocke_value_t* m,
                                               rocke_value_t* n,
                                               int elem_idx,
                                               const rocke_fe_params_t* params);

/* ================================================================== *
 * _MultiDEpilogue  (gemm_multi_d.py)
 * ================================================================== */

/* "stock" | "tiled" | "vector" -- the d_load_kind selector. */
typedef enum rocke_mde_load_kind
{
    ROCKE_MDE_STOCK = 0,
    ROCKE_MDE_TILED,
    ROCKE_MDE_VECTOR
} rocke_mde_load_kind_t;

/* Multi-D fused epilogue: a FusedEpilogue plus the precomputed per-op residual
 * classification (_residual_kinds / _residual_dtypes) and the load strategy.
 * The base FusedEpilogue is embedded so all the base methods (declare_params,
 * record_runtime, params table, _ir_dtype) apply unchanged. */
typedef struct rocke_multi_d_epilogue
{
    rocke_fused_epilogue_t base;
    /* _residual_kinds[i]: the op kind, mirrored for fast classification.
     * _residual_dtypes[i]: the resolved D dtype, or NULL for a non-residual op
     * (Python stores None). Both arrays have base.num_ops entries. */
    rocke_epilogue_op_kind_t residual_kinds[ROCKE_FE_MAX_PARAMS];
    const rocke_type_t* residual_dtypes[ROCKE_FE_MAX_PARAMS];
    rocke_mde_load_kind_t load_kind;
} rocke_multi_d_epilogue_t;

/* _MultiDEpilogue.from_ops(ops, dtype, load_kind): classify each op into
 * (kind, resolved-dtype) -- ResidualAdd->("add", dtype_to_ir(op.dtype)),
 * ResidualMul->("mul", ...), anything else->(None, None) -- and build the
 * epilogue. `dtype` NULL => F16, load_kind defaults via the enum (caller picks;
 * VECTOR is the Python default). Returns ROCKE_OK, or ROCKE_ERR_VALUE if num_ops
 * exceeds ROCKE_FE_MAX_PARAMS. */
rocke_status_t rocke_mde_from_ops(rocke_multi_d_epilogue_t* mde,
                                  const rocke_epilogue_op_t* ops,
                                  size_t num_ops,
                                  const rocke_type_t* dtype,
                                  rocke_mde_load_kind_t load_kind);

/* _MultiDEpilogue.apply_vec(b, v, m, n, n_elems): the optimised multi-D fast
 * path with byte-faithful fallbacks:
 *
 *   - load_kind==VECTOR and n_elems not in {2,4,8}  -> base apply_vec
 *   - any op not a residual (kind classified None)  -> base apply_vec
 *   - "tiled": hoist off_base = add(mul(m, stride_m), n); per element load each
 *     D from off_i (= off_base, or add(off_base, const_i32(i)) for i>0) and
 *     fadd/fmul; vec_pack.
 *   - "vector": per D, if dtype name in {f16,bf16} use global_load_vN(ptr,
 *     off_base, dt, n_elems); if f32 build the vector from n_elems scalar loads
 *     at add(off_base, const_i32(i)) then vec_pack; any other dtype -> base
 *     apply_vec. Then per element vec_extract each D-vector and fadd/fmul; pack.
 *
 * stride_m = params.get("__stride_m") or params["__N"] (Python `or`). */
rocke_value_t* rocke_mde_apply_vec(rocke_ir_builder_t* b,
                                   rocke_multi_d_epilogue_t* mde,
                                   rocke_value_t* v,
                                   rocke_value_t* m,
                                   rocke_value_t* n,
                                   int n_elems);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_FUSE_H */
