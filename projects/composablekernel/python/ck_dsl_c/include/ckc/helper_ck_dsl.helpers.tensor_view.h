/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.tensor_view.h -- C99 port of the
 * ``ck_dsl.helpers.tensor_view`` module.
 *
 * Ported symbols (per phase scope):
 *   make_global_view, make_tile_window, TensorDescriptor, TensorView.
 *
 * These are pure host-side abstractions (no IR is emitted at construction
 * time). The IR-emitting members --- TensorDescriptor.offset and the
 * TensorView load/store family --- call into the C builder (ckc_b_*,
 * ckc_* in ckc/ir.h) and MUST reproduce the Python builder-call sequence
 * byte-for-byte.
 *
 * Modelling choices for the port:
 *
 *   * Python dataclasses become plain C structs. They are value types;
 *     callers allocate them on the stack and pass them by pointer.
 *   * ``strides`` entries are ``int | Value`` in Python. In C each stride
 *     is a small tagged variant (ckc_stride_t): a compile-time int OR a
 *     runtime SSA ckc_value_t*. This preserves the offset() fast-paths
 *     (literal-1 omitted, constant mul folded) verbatim.
 *   * ``addr_space`` is the enum ckc_addr_space_t.
 *   * ``dtype`` is a ckc_type_t* (same Type objects the builder uses);
 *     dispatch keys off ``dtype->name`` exactly like Python ``dtype.name``.
 *
 * Errors: where Python raises, the C port records the sticky error on the
 * builder (ckc_b_*) when a builder is in hand, and otherwise returns a
 * status / NULL. Construction-time rank checks return a ckc_status_t.
 */
#ifndef CKC_HELPER_TENSOR_VIEW_H
#define CKC_HELPER_TENSOR_VIEW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ckc/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum descriptor rank we support inline (CK Tile descriptors are small;
 * GEMM/attention use rank <= 4). Kept generous to avoid heap churn. */
#define CKC_TV_MAX_RANK 8

/* Maximum fp16/bf16 vector width handled inline by store_vec_from_f32's
 * temporary cast buffer (CK Tile wide stores top out at <8 x half>). */
#define CKC_TV_MAX_VEC 16

/* --------------------------------------------------------------- addr space */

typedef enum ckc_addr_space
{
    CKC_ADDR_GLOBAL = 0, /* "global" */
    CKC_ADDR_LDS,        /* "lds"    */
    CKC_ADDR_BUFFER      /* "buffer" */
} ckc_addr_space_t;

/* ----------------------------------------------------------------- stride */

/* One stride element: compile-time int OR runtime SSA Value (Python
 * ``StrideElem = Union[int, Value]``). */
typedef struct ckc_stride
{
    bool is_value;      /* true => runtime SSA stride in .value         */
    int64_t imm;        /* compile-time stride (valid iff !is_value)    */
    ckc_value_t* value; /* runtime SSA stride (valid iff is_value)      */
} ckc_stride_t;

static inline ckc_stride_t ckc_stride_imm(int64_t v)
{
    ckc_stride_t s;
    s.is_value = false;
    s.imm      = v;
    s.value    = NULL;
    return s;
}

static inline ckc_stride_t ckc_stride_value(ckc_value_t* v)
{
    ckc_stride_t s;
    s.is_value = true;
    s.imm      = 0;
    s.value    = v;
    return s;
}

/* ----------------------------------------------------------- TensorDescriptor */

/* Pure shape + strides + dtype; no SSA at construction. Analogue of CK Tile's
 * ``tensor_descriptor``. */
typedef struct ckc_tensor_descriptor
{
    int rank;
    int shape[CKC_TV_MAX_RANK]; /* element extents               */
    ckc_stride_t strides[CKC_TV_MAX_RANK];
    const ckc_type_t* dtype;
} ckc_tensor_descriptor_t;

/* TensorDescriptor.__init__ with rank validation (Python __post_init__).
 * Returns CKC_OK, or CKC_ERR_VALUE on shape/strides rank mismatch or empty. */
ckc_status_t ckc_tensor_descriptor_init(ckc_tensor_descriptor_t* out,
                                        const int* shape,
                                        const ckc_stride_t* strides,
                                        int rank,
                                        const ckc_type_t* dtype);

/* TensorDescriptor.packed(shape, dtype) -- row-major packed strides. */
ckc_status_t ckc_tensor_descriptor_packed(ckc_tensor_descriptor_t* out,
                                          const int* shape,
                                          int rank,
                                          const ckc_type_t* dtype);

/* TensorDescriptor.with_strides(shape, strides, dtype). */
ckc_status_t ckc_tensor_descriptor_with_strides(ckc_tensor_descriptor_t* out,
                                                const int* shape,
                                                const ckc_stride_t* strides,
                                                int rank,
                                                const ckc_type_t* dtype);

/* @property rank / numel. */
int ckc_tensor_descriptor_rank(const ckc_tensor_descriptor_t* d);
int64_t ckc_tensor_descriptor_numel(const ckc_tensor_descriptor_t* d);

/* TensorDescriptor.offset(b, indices): flat element offset (SSA). Emits the
 * same mul/add chain as Python (literal-1 stride omitted, const stride folded
 * into const_i32 mul). Returns NULL with builder error on rank mismatch. */
ckc_value_t* ckc_tensor_descriptor_offset(ckc_ir_builder_t* b,
                                          const ckc_tensor_descriptor_t* d,
                                          ckc_value_t* const* indices,
                                          int num_indices);

/* ----------------------------------------------------------------- TensorView */

/* pointer + descriptor + address space. Analogue of CK Tile's tensor_view.
 * For CKC_ADDR_GLOBAL / CKC_ADDR_LDS, ``base`` is the pointer/smem token.
 * (Buffer address space and its BufferResource are out of this phase's scope.)
 */
typedef struct ckc_tensor_view
{
    ckc_value_t* base;
    ckc_tensor_descriptor_t desc;
    ckc_addr_space_t addr_space;
} ckc_tensor_view_t;

/* @property dtype / shape / rank. */
const ckc_type_t* ckc_tensor_view_dtype(const ckc_tensor_view_t* v);
int ckc_tensor_view_rank(const ckc_tensor_view_t* v);

/* TensorView.load_scalar(b, indices). */
ckc_value_t* ckc_tensor_view_load_scalar(ckc_ir_builder_t* b,
                                         const ckc_tensor_view_t* v,
                                         ckc_value_t* const* indices,
                                         int num_indices);

/* TensorView.store_scalar(b, indices, value, align). Pass align<=0 for the
 * Python default (align=None). */
void ckc_tensor_view_store_scalar(ckc_ir_builder_t* b,
                                  const ckc_tensor_view_t* v,
                                  ckc_value_t* const* indices,
                                  int num_indices,
                                  ckc_value_t* value,
                                  int align);

/* TensorView.load_vec(b, indices, n). */
ckc_value_t* ckc_tensor_view_load_vec(ckc_ir_builder_t* b,
                                      const ckc_tensor_view_t* v,
                                      ckc_value_t* const* indices,
                                      int num_indices,
                                      int n);

/* TensorView.store_vec(b, indices, value, n). */
void ckc_tensor_view_store_vec(ckc_ir_builder_t* b,
                               const ckc_tensor_view_t* v,
                               ckc_value_t* const* indices,
                               int num_indices,
                               ckc_value_t* value,
                               int n);

/* TensorView.load_vec_at(b, elem_off, n). (No buffer mask in this phase.) */
ckc_value_t* ckc_tensor_view_load_vec_at(ckc_ir_builder_t* b,
                                         const ckc_tensor_view_t* v,
                                         ckc_value_t* elem_off,
                                         int n);

/* TensorView.store_vec_at(b, elem_off, value, n). */
void ckc_tensor_view_store_vec_at(ckc_ir_builder_t* b,
                                  const ckc_tensor_view_t* v,
                                  ckc_value_t* elem_off,
                                  ckc_value_t* value,
                                  int n);

/* ----------------------------------------------------------------- TileWindow */

/* A fixed-extent window into a TensorView. ``view`` is held by pointer (the
 * Python dataclass holds a reference); ``origin`` are SSA Values. */
typedef struct ckc_tile_window
{
    const ckc_tensor_view_t* view;
    int rank;
    int lengths[CKC_TV_MAX_RANK];
    ckc_value_t* origin[CKC_TV_MAX_RANK];
} ckc_tile_window_t;

/* @property rank / dtype / addr_space. */
int ckc_tile_window_rank(const ckc_tile_window_t* w);
const ckc_type_t* ckc_tile_window_dtype(const ckc_tile_window_t* w);
ckc_addr_space_t ckc_tile_window_addr_space(const ckc_tile_window_t* w);

/* TileWindow.load_vec(b, *local_indices, n). */
ckc_value_t* ckc_tile_window_load_vec(ckc_ir_builder_t* b,
                                      const ckc_tile_window_t* w,
                                      ckc_value_t* const* local_indices,
                                      int num_indices,
                                      int n);

/* TileWindow.store_vec(b, *local_indices, value, n). */
void ckc_tile_window_store_vec(ckc_ir_builder_t* b,
                               const ckc_tile_window_t* w,
                               ckc_value_t* const* local_indices,
                               int num_indices,
                               ckc_value_t* value,
                               int n);

/* TileWindow.load_scalar(b, *local_indices). */
ckc_value_t* ckc_tile_window_load_scalar(ckc_ir_builder_t* b,
                                         const ckc_tile_window_t* w,
                                         ckc_value_t* const* local_indices,
                                         int num_indices);

/* TileWindow.store_scalar(b, *local_indices, value, align). Pass align<=0 for
 * the Python default (align=None). */
void ckc_tile_window_store_scalar(ckc_ir_builder_t* b,
                                  const ckc_tile_window_t* w,
                                  ckc_value_t* const* local_indices,
                                  int num_indices,
                                  ckc_value_t* value,
                                  int align);

/* ---------------------------------------------------- module-level factories */

/* TensorView.tile(lengths, origin) -- the method form. */
ckc_status_t ckc_tensor_view_tile(ckc_tile_window_t* out,
                                  const ckc_tensor_view_t* view,
                                  const int* lengths,
                                  ckc_value_t* const* origin,
                                  int rank);

/* make_global_view(base, shape, dtype, strides=None).
 * Pass strides=NULL for packed row-major (the Python default). */
ckc_status_t ckc_make_global_view(ckc_tensor_view_t* out,
                                  ckc_value_t* base,
                                  const int* shape,
                                  int rank,
                                  const ckc_type_t* dtype,
                                  const ckc_stride_t* strides /* NULL => packed */);

/* make_tile_window(view, lengths, origin). Free-function alias of
 * TensorView.tile. */
ckc_status_t ckc_make_tile_window(ckc_tile_window_t* out,
                                  const ckc_tensor_view_t* view,
                                  const int* lengths,
                                  ckc_value_t* const* origin,
                                  int rank);

/* make_naive_tensor_view_packed(base, shape, dtype): CK Tile literal-name
 * alias of make_global_view with packed row-major strides. */
ckc_status_t ckc_make_naive_tensor_view_packed(
    ckc_tensor_view_t* out, ckc_value_t* base, const int* shape, int rank, const ckc_type_t* dtype);

/* make_lds_view(b, dtype, shape, name_hint, strides=NULL): allocate an
 * addrspace(3) buffer for the kernel lifetime and return a view over it.
 * strides=NULL => packed row-major. */
ckc_status_t ckc_make_lds_view(ckc_ir_builder_t* b,
                               ckc_tensor_view_t* out,
                               const ckc_type_t* dtype,
                               const int* shape,
                               int rank,
                               const char* name_hint,
                               const ckc_stride_t* strides /* NULL => packed */);

/* ----------------------------------------- compute-promoting (f32) peers */

/* TensorView.load_vec_as_f32(b, indices, n): vector load + per-lane f32
 * promotion. Writes the ``n`` f32 SSA scalars to out[0..n) (length >= n).
 * For an f32 view the per-lane cast is a no-op (elements are extracted
 * directly); n==1 routes through load_scalar. */
void ckc_tensor_view_load_vec_as_f32(ckc_ir_builder_t* b,
                                     const ckc_tensor_view_t* v,
                                     ckc_value_t* const* indices,
                                     int num_indices,
                                     int n,
                                     ckc_value_t** out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_TENSOR_VIEW_H */
