/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/helper_ck_dsl.instances.common._fmha_common.h -- C99 port of the shared
 * FMHA scaffolding from ck_dsl/instances/common/_fmha_common.py.
 *
 *   Python (_fmha_common.py)              C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   FmhaMaskMode (Literal)                ckc_fmha_mask_mode_t (enum)
 *   @dataclass(frozen=True) FmhaShape     ckc_fmha_shape_t
 *     .num_queries_per_kv (property)      ckc_fmha_shape_num_queries_per_kv(...)
 *   @dataclass(frozen=True) FmhaCommonSpec ckc_fmha_common_spec_t
 *     .head_size (property)               ckc_fmha_common_spec_head_size(...)
 *     .use_alibi_matrix (property)        ckc_fmha_common_spec_use_alibi_matrix(...)
 *     .use_custom_mask (property)         ckc_fmha_common_spec_use_custom_mask(...)
 *   validate_common_spec(spec)            ckc_fmha_validate_common_spec(...)
 *   class FmhaKernelBuilder               ckc_fmha_kernel_builder_t (+ methods)
 *
 * FmhaShape / FmhaCommonSpec / validate_common_spec are pure value types: their
 * fields are scalars and their methods are integer / string compares, none of
 * which touch the IR builder. validate_common_spec returns (ok, reason); the
 * reason strings only ever surface through ValueError messages, never the IR, so
 * downstream emitted code is byte-identical regardless.
 *
 * FmhaKernelBuilder DOES drive the IR builder. It owns a ckc_ir_builder_t and
 * reproduces the Python builder-call sequence byte-faithfully (param() in the
 * exact declaration order, to_sgpr_u32 grid decode, add/mul row-base math). The
 * ported methods bind to ckc/ir.h plus the sibling helper ports:
 *   - helper_ck_dsl.helpers.io.h        (io_ir_type)
 *   - helper_ck_dsl.helpers.spec.h      (SignatureBuilder)
 *   - helper_ck_dsl.helpers.transforms.h(TensorDescriptor)
 *
 * The mask-mode Literal becomes an enum; the spelling is recovered with
 * ckc_fmha_mask_mode_name() so reason strings reproduce Python's repr() text.
 *
 * Error model mirrors the rest of the C port: value helpers take an out-param +
 * bool/status; builder methods route through the sticky-error IRBuilder; the
 * GQA-divisibility ValueError of the num_queries_per_kv property is surfaced via
 * an out-param + status (see ckc_fmha_shape_num_queries_per_kv).
 */
#ifndef CKC_HELPER_CK_DSL_INSTANCES_COMMON__FMHA_COMMON_H
#define CKC_HELPER_CK_DSL_INSTANCES_COMMON__FMHA_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t (signature)        */
#include "ckc/helper_ck_dsl.helpers.transforms.h" /* ckc_tensor_descriptor_t       */
#include "ckc/ir.h" /* ckc_status_t, ckc_value_t, builder */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaMaskMode
 * ------------------------------------------------------------------ *
 *
 * FmhaMaskMode = Literal["none","causal","sliding_window","alibi","custom"].
 * Stored as an enum; ckc_fmha_mask_mode_name() recovers the canonical lowercase
 * spelling so reason strings reproduce Python's `mask_mode!r` text. */
typedef enum ckc_fmha_mask_mode
{
    CKC_FMHA_MASK_NONE = 0, /* "none"            */
    CKC_FMHA_MASK_CAUSAL, /* "causal"          */
    CKC_FMHA_MASK_SLIDING_WINDOW, /* "sliding_window"  */
    CKC_FMHA_MASK_ALIBI, /* "alibi"           */
    CKC_FMHA_MASK_CUSTOM /* "custom"          */
} ckc_fmha_mask_mode_t;

/* Canonical lowercase spelling for a mask mode ("none", "causal", ...); NULL for
 * an out-of-range value. */
const char* ckc_fmha_mask_mode_name(ckc_fmha_mask_mode_t m);

/* ------------------------------------------------------------------ *
 * FmhaShape
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaShape:
 *     head_size: int
 *     num_query_heads: int
 *     num_kv_heads: int
 *     block_size_q: int = 16
 *     block_size_k: int = 64
 *
 * A pure value struct. block_size_q / block_size_k carry the dataclass defaults;
 * use ckc_fmha_shape_default() to take the defaulted fields. */
typedef struct ckc_fmha_shape
{
    int head_size;
    int num_query_heads;
    int num_kv_heads;
    int block_size_q; /* default 16 */
    int block_size_k; /* default 64 */
} ckc_fmha_shape_t;

/* FmhaShape(head_size, num_query_heads, num_kv_heads, block_size_q=16,
 * block_size_k=64): take the required fields and the dataclass defaults for the
 * two block-size fields. */
ckc_fmha_shape_t ckc_fmha_shape_default(int head_size, int num_query_heads, int num_kv_heads);

/* FmhaShape(...) with every field explicit. */
ckc_fmha_shape_t ckc_fmha_shape_make(
    int head_size, int num_query_heads, int num_kv_heads, int block_size_q, int block_size_k);

/* FmhaShape.num_queries_per_kv property:
 *   if num_query_heads % num_kv_heads: raise ValueError(...)
 *   return num_query_heads // num_kv_heads
 *
 * On the divisibility mismatch the Python raises ValueError; here returns
 * CKC_ERR_VALUE and leaves *out untouched. On success writes the quotient to
 * *out and returns CKC_OK. */
ckc_status_t ckc_fmha_shape_num_queries_per_kv(const ckc_fmha_shape_t* s, int* out);

/* ------------------------------------------------------------------ *
 * FmhaCommonSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaCommonSpec:
 *     shape: FmhaShape
 *     dtype: str = "f16"
 *     scale_log2: float = 0.0
 *     mask_mode: FmhaMaskMode = "none"
 *     sliding_window: int = 0
 *     use_softcap: bool = False
 *     use_rotary: bool = False
 *     use_dropout: bool = False
 *     use_sinks: bool = False
 *
 * dtype is referenced as-is (not copied); keep it alive for the spec's use. */
typedef struct ckc_fmha_common_spec
{
    ckc_fmha_shape_t shape;
    const char* dtype; /* default "f16"  */
    double scale_log2; /* default 0.0    */
    ckc_fmha_mask_mode_t mask_mode; /* default "none" */
    int sliding_window; /* default 0      */
    bool use_softcap; /* default false  */
    bool use_rotary; /* default false  */
    bool use_dropout; /* default false  */
    bool use_sinks; /* default false  */
} ckc_fmha_common_spec_t;

/* FmhaCommonSpec(shape, dtype="f16", scale_log2=0.0, mask_mode="none",
 * sliding_window=0, use_*=False): construct from `shape` with all the dataclass
 * defaults for the remaining fields. */
ckc_fmha_common_spec_t ckc_fmha_common_spec_default(ckc_fmha_shape_t shape);

/* FmhaCommonSpec.head_size property: spec.shape.head_size. */
static inline int ckc_fmha_common_spec_head_size(const ckc_fmha_common_spec_t* spec)
{
    return spec->shape.head_size;
}

/* FmhaCommonSpec.use_alibi_matrix property: mask_mode == "alibi". */
static inline bool ckc_fmha_common_spec_use_alibi_matrix(const ckc_fmha_common_spec_t* spec)
{
    return spec->mask_mode == CKC_FMHA_MASK_ALIBI;
}

/* FmhaCommonSpec.use_custom_mask property: mask_mode == "custom". */
static inline bool ckc_fmha_common_spec_use_custom_mask(const ckc_fmha_common_spec_t* spec)
{
    return spec->mask_mode == CKC_FMHA_MASK_CUSTOM;
}

/* ------------------------------------------------------------------ *
 * validate_common_spec
 * ------------------------------------------------------------------ *
 *
 * validate_common_spec(spec) -> (ok, reason). Returns true (ok) / false
 * (reject). *out_reason (if non-NULL) receives an arena-owned reason string
 * ("ok" on the accept path); pass a live `arena`. The reason strings reproduce
 * the Python messages byte-for-byte. On a formatting OOM the predicate result is
 * still returned faithfully and the reason is a best-effort static fallback. */
bool ckc_fmha_validate_common_spec(ckc_arena_t* arena,
                                   const ckc_fmha_common_spec_t* spec,
                                   const char** out_reason);

/* ------------------------------------------------------------------ *
 * FmhaKernelBuilder
 * ------------------------------------------------------------------ *
 *
 * The boilerplate-killing builder. Owns a ckc_ir_builder_t and three param
 * registries (tensors, strides, "other") plus the in-order param log used by
 * .signature(). Decoded grid coords are populated by decode_grid / appendkv_grid.
 *
 * Lifetime: ckc_fmha_kernel_builder_init() initialises the embedded
 * ckc_ir_builder_t (allocating its arena); the builder owns all IR nodes AND the
 * registry storage (allocated from the IR builder's arena). Call
 * ckc_fmha_kernel_builder_free() to bulk-free everything. */

/* The in-order param log entry kind (Python _sig_order tuple[0]). */
typedef enum ckc_fmha_sig_kind
{
    CKC_FMHA_SIG_TENSOR = 0, /* "tensor" */
    CKC_FMHA_SIG_PTR, /* "ptr"    */
    CKC_FMHA_SIG_SCALAR /* "scalar" */
} ckc_fmha_sig_kind_t;

/* One ("tensor"|"ptr"|"scalar", name, dtype) tuple from _sig_order. */
typedef struct ckc_fmha_sig_order_entry
{
    ckc_fmha_sig_kind_t kind;
    const char* name; /* arena-owned */
    const char* dtype; /* arena-owned (the dtype spelling for tensor/ptr; the
                        * scalar type "i32"/"f32" for scalar)               */
} ckc_fmha_sig_order_entry_t;

/* A registered (name -> Value) pair (tensor / other registries). */
typedef struct ckc_fmha_named_value
{
    const char* name; /* arena-owned */
    ckc_value_t* value;
} ckc_fmha_named_value_t;

/* A registered (name -> (stride_token, stride_head)) pair. */
typedef struct ckc_fmha_stride_pair
{
    const char* name; /* arena-owned (the un-suffixed tensor name) */
    ckc_value_t* token;
    ckc_value_t* head;
} ckc_fmha_stride_pair_t;

typedef struct ckc_fmha_kernel_builder
{
    ckc_fmha_common_spec_t common;
    ckc_ir_builder_t b; /* self.b = IRBuilder(kernel_name) */

    /* _sig_order: in-order param log. */
    ckc_fmha_sig_order_entry_t* sig_order;
    size_t n_sig_order;
    size_t cap_sig_order;

    /* _tensor_params: name -> Value. */
    ckc_fmha_named_value_t* tensor_params;
    size_t n_tensor_params;
    size_t cap_tensor_params;

    /* _stride_params: name -> (token, head). */
    ckc_fmha_stride_pair_t* stride_params;
    size_t n_stride_params;
    size_t cap_stride_params;

    /* _other_params: name -> Value (scalars, strides-by-name, ptrs). */
    ckc_fmha_named_value_t* other_params;
    size_t n_other_params;
    size_t cap_other_params;

    /* Decoded grid coords (decode_grid / appendkv_grid populate these). NULL
     * until decoded (the Python Optional[Value] = None). */
    ckc_value_t* q_token;
    ckc_value_t* head_idx;
    ckc_value_t* kv_head_idx;
    ckc_value_t* batch_idx;
    ckc_value_t* q_tile_base; /* appendkv_grid only */
} ckc_fmha_kernel_builder_t;

/* FmhaKernelBuilder.__init__(kernel_name, common): initialise the embedded IR
 * builder with `kernel_name`, copy `common`, and empty the registries + decoded
 * coords. Returns CKC_OK or the IR-builder-init status on failure. */
ckc_status_t ckc_fmha_kernel_builder_init(ckc_fmha_kernel_builder_t* kb,
                                          const char* kernel_name,
                                          const ckc_fmha_common_spec_t* common);

/* Bulk-free the embedded IR builder (and thereby every registry + IR node). */
void ckc_fmha_kernel_builder_free(ckc_fmha_kernel_builder_t* kb);

/* @property kernel -> self.b.kernel. */
ckc_kernel_def_t* ckc_fmha_kernel_builder_kernel(ckc_fmha_kernel_builder_t* kb);

/* @property builder -> self.b. The underlying IR builder. */
ckc_ir_builder_t* ckc_fmha_kernel_builder_builder(ckc_fmha_kernel_builder_t* kb);

/* ----- param declarations ----- */

/* add_tensor(name, dtype=None, readonly=True, writeonly=False, align=16):
 * declare a global tensor param and remember its dtype for descriptors.
 *   dtype NULL -> self.common.dtype.
 *   "f16"/"fp16"/"bf16" -> io_ir_type(...); "fp8e4m3" -> FP8E4M3;
 *   "bf8e5m2" -> BF8E5M2; "i8" -> I8; anything else -> ValueError (sets the
 *   builder error and returns NULL).
 * Emits b.param(name, ptr<ty,global>, noalias=True, readonly, writeonly,
 * align), records the tensor + a ("tensor", name, dtype) log entry, and returns
 * the param Value. */
ckc_value_t* ckc_fmha_kernel_builder_add_tensor(ckc_fmha_kernel_builder_t* kb,
                                                const char* name,
                                                const char* dtype, /* NULL => common.dtype */
                                                bool readonly,
                                                bool writeonly,
                                                int align /* default 16 */);

/* add_ptr(name, dtype, readonly=True, align=4): declare a non-canonical pointer
 * param (block_table, scales, ...). "i32" -> I32, "f32" -> F32, "i8" -> I8,
 * otherwise io_ir_type(dtype). Emits b.param(name, ptr<ty,global>, noalias=True,
 * readonly, align), records it under _other_params + a ("ptr", name, dtype) log
 * entry. */
ckc_value_t* ckc_fmha_kernel_builder_add_ptr(ckc_fmha_kernel_builder_t* kb,
                                             const char* name,
                                             const char* dtype,
                                             bool readonly,
                                             int align /* default 4 */);

/* add_scalar(name, dtype="i32"): declare a scalar param. dtype "i32" -> I32,
 * anything else -> F32 (mirrors the Python ternary). Records under _other_params
 * + a ("scalar", name, dtype) log entry. */
ckc_value_t* ckc_fmha_kernel_builder_add_scalar(ckc_fmha_kernel_builder_t* kb,
                                                const char* name,
                                                const char* dtype /* NULL/"i32" => i32 */);

/* add_strides(*names): for each name declare stride_{name}_token and
 * stride_{name}_head (both I32), record them in _stride_params + _other_params,
 * and append two ("scalar", stride_..., "i32") log entries (token first, then
 * head). `names` is an array of `n` un-suffixed tensor names. */
void ckc_fmha_kernel_builder_add_strides(ckc_fmha_kernel_builder_t* kb,
                                         const char* const* names,
                                         size_t n);

/* ----- accessors ----- */

/* tensor(name) -> _tensor_params[name]; NULL if absent (the Python KeyError). */
ckc_value_t* ckc_fmha_kernel_builder_tensor(const ckc_fmha_kernel_builder_t* kb, const char* name);

/* stride(name) -> (_stride_params[name]); on success writes *out_token /
 * *out_head and returns true; false if absent. */
bool ckc_fmha_kernel_builder_stride(const ckc_fmha_kernel_builder_t* kb,
                                    const char* name,
                                    ckc_value_t** out_token,
                                    ckc_value_t** out_head);

/* stride_token(name) -> _stride_params[name][0]; NULL if absent. */
ckc_value_t* ckc_fmha_kernel_builder_stride_token(const ckc_fmha_kernel_builder_t* kb,
                                                  const char* name);

/* stride_head(name) -> _stride_params[name][1]; NULL if absent. */
ckc_value_t* ckc_fmha_kernel_builder_stride_head(const ckc_fmha_kernel_builder_t* kb,
                                                 const char* name);

/* scalar(name)/ptr(name) -> _other_params[name]; NULL if absent. (One registry
 * backs both, exactly like the Python _other_params.) */
ckc_value_t* ckc_fmha_kernel_builder_scalar(const ckc_fmha_kernel_builder_t* kb, const char* name);
ckc_value_t* ckc_fmha_kernel_builder_ptr(const ckc_fmha_kernel_builder_t* kb, const char* name);

/* block_size(block_size): set kernel.attrs["max_workgroup_size"] = block_size. */
void ckc_fmha_kernel_builder_block_size(ckc_fmha_kernel_builder_t* kb, int block_size);

/* ----- grid decode ----- */

/* appendkv_grid(block_q=64, has_batch_axis=False) -> (q_tile_base, kv_head_idx).
 *   q_tile_base = to_sgpr_u32(block_id_x * block_q)
 *   kv_head_idx = to_sgpr_u32(block_id_y)
 *   if has_batch_axis: batch_idx = to_sgpr_u32(block_id_z)
 * Also populated as fields on `kb`. */
void ckc_fmha_kernel_builder_appendkv_grid(ckc_fmha_kernel_builder_t* kb,
                                           int block_q, /* default 64 */
                                           bool has_batch_axis,
                                           ckc_value_t** out_q_tile_base,
                                           ckc_value_t** out_kv_head_idx);

/* decode_grid(num_queries_per_kv=None, has_batch_axis=False) ->
 * (q_token, head_idx, kv_head_idx). Binds the grid axes through to_sgpr_u32 and
 * the GQA / MQA decode:
 *   q_token  = to_sgpr_u32(block_id_x)
 *   head_idx = to_sgpr_u32(block_id_y)
 *   if has_batch_axis: batch_idx = to_sgpr_u32(block_id_z)
 *   nqkv = num_queries_per_kv if set else common.shape.num_queries_per_kv
 *   kv_head_idx = head_idx (nqkv==1) else to_sgpr_u32(div(head_idx, const(nqkv)))
 * num_queries_per_kv < 0 means Python None (fall back to the shape property; its
 * ValueError on a GQA mismatch propagates to the builder error). On success
 * writes the three out-params (also stored on `kb`). On the shape-property
 * ValueError sets the builder error and leaves out-params NULL. */
void ckc_fmha_kernel_builder_decode_grid(ckc_fmha_kernel_builder_t* kb,
                                         int num_queries_per_kv, /* < 0 => None */
                                         bool has_batch_axis,
                                         ckc_value_t** out_q_token,
                                         ckc_value_t** out_head_idx,
                                         ckc_value_t** out_kv_head_idx);

/* ----- tensor descriptor factory ----- */

/* tensor_descriptor(tensor_name, coord_names=("token","head","d"),
 * lengths=None): build a TensorDescriptor for a registered tensor.
 *   lengths None -> (1<<24, 1<<12, max(common.shape.head_size, 1)).
 *   If the tensor has no registered strides -> TensorDescriptor.naive(name,
 *   lengths, coord_names). Otherwise -> TensorDescriptor(name, base_names=
 *   coord_names, base_lengths=lengths, base_strides=(0,0,1), chain=(),
 *   upper_names=coord_names).
 * coord_names NULL -> the default triple. lengths NULL -> the stand-in triple.
 * Returns the (arena-owned) descriptor or NULL on a builder error. */
ckc_tensor_descriptor_t* ckc_fmha_kernel_builder_tensor_descriptor(
    ckc_fmha_kernel_builder_t* kb,
    const char* tensor_name,
    const char* const* coord_names, /* NULL => {"token","head","d"} */
    const int* lengths, /* NULL => stand-in triple      */
    int n_lengths);

/* ----- signature builder ----- */

/* signature(): walk _sig_order and emit the SignatureBuilder shape (.ptr for
 * tensor/ptr, .scalar for scalar). On CKC_OK *out_items / *out_count hold the
 * arena-owned array; on failure they are untouched and the status is returned.
 * `arena` backs the SignatureBuilder's storage. */
ckc_status_t ckc_fmha_kernel_builder_signature(ckc_fmha_kernel_builder_t* kb,
                                               ckc_arena_t* arena,
                                               const ckc_sig_entry_t** out_items,
                                               size_t* out_count);

/* ----- row-base helpers ----- */

/* q_row_base() -> q_token*stride_q_token + head_idx*stride_q_head. */
ckc_value_t* ckc_fmha_kernel_builder_q_row_base(ckc_fmha_kernel_builder_t* kb);

/* o_row_base() -> q_token*stride_o_token + head_idx*stride_o_head. */
ckc_value_t* ckc_fmha_kernel_builder_o_row_base(ckc_fmha_kernel_builder_t* kb);

/* row_base(tensor_name, tok, hd) -> tok*stride_{name}_token + hd*stride_{name}_head. */
ckc_value_t* ckc_fmha_kernel_builder_row_base(ckc_fmha_kernel_builder_t* kb,
                                              const char* tensor_name,
                                              ckc_value_t* tok,
                                              ckc_value_t* hd);

/* k_row_base(k_idx) -> k_idx*stride_k_token + kv_head_idx*stride_k_head. */
ckc_value_t* ckc_fmha_kernel_builder_k_row_base(ckc_fmha_kernel_builder_t* kb, ckc_value_t* k_idx);

/* v_row_base(k_idx) -> k_idx*stride_v_token + kv_head_idx*stride_v_head. */
ckc_value_t* ckc_fmha_kernel_builder_v_row_base(ckc_fmha_kernel_builder_t* kb, ckc_value_t* k_idx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_INSTANCES_COMMON__FMHA_COMMON_H */
