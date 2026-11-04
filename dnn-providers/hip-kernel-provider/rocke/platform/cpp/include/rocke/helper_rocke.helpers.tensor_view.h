/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_rocke.helpers.tensor_view.h -- C99 port of the
 * ``rocke.helpers.tensor_view`` module.
 *
 * Ported symbols (per phase scope):
 *   make_global_view, make_tile_window, TensorDescriptor, TensorView,
 *   BufferResource, make_buffer_resource, make_buffer_view.
 *
 * Not ported: TensorView.load_scalar_at, store_scalar_at and
 * async_load_lds_at have no C twin in any address space, so their Python
 * buffer branches have none either. The buffer branches that are ported are
 * those of load_scalar, store_scalar, load_vec, store_vec, load_vec_at and
 * store_vec_at.
 *
 * These are pure host-side abstractions (no IR is emitted at construction
 * time). The IR-emitting members --- TensorDescriptor.offset and the
 * TensorView load/store family --- call into the C builder (rocke_b_*,
 * rocke_* in rocke/ir.h) and MUST reproduce the Python builder-call sequence
 * byte-for-byte.
 *
 * Modelling choices for the port:
 *
 *   * Python dataclasses become plain C structs. They are value types;
 *     callers allocate them on the stack and pass them by pointer.
 *   * ``strides`` entries are ``int | Value`` in Python. In C each stride
 *     is a small tagged variant (rocke_stride_t): a compile-time int OR a
 *     runtime SSA rocke_value_t*. This preserves the offset() fast-paths
 *     (literal-1 omitted, constant mul folded) verbatim.
 *   * ``addr_space`` is the enum rocke_addr_space_t.
 *   * ``dtype`` is a rocke_type_t* (same Type objects the builder uses);
 *     dispatch keys off ``dtype->name`` exactly like Python ``dtype.name``.
 *
 * Errors: where Python raises, the C port records the sticky error on the
 * builder (rocke_b_*) when a builder is in hand, and otherwise returns a
 * status / NULL. Construction-time rank checks return a rocke_status_t.
 */
#ifndef ROCKE_HELPER_TENSOR_VIEW_H
#define ROCKE_HELPER_TENSOR_VIEW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum descriptor rank we support inline (CK Tile descriptors are small;
 * GEMM/attention use rank <= 4). Kept generous to avoid heap churn. */
#define ROCKE_TV_MAX_RANK 8

/* Maximum fp16/bf16 vector width handled inline by store_vec_from_f32's
 * temporary cast buffer (CK Tile wide stores top out at <8 x half>). */
#define ROCKE_TV_MAX_VEC 16

/* --------------------------------------------------------------- addr space */

typedef enum rocke_addr_space
{
    ROCKE_ADDR_GLOBAL = 0, /* "global" */
    ROCKE_ADDR_LDS, /* "lds"    */
    ROCKE_ADDR_BUFFER /* "buffer" */
} rocke_addr_space_t;

/* ----------------------------------------------------------------- stride */

/* One stride element: compile-time int OR runtime SSA Value (Python
 * ``StrideElem = Union[int, Value]``). */
typedef struct rocke_stride
{
    bool is_value; /* true => runtime SSA stride in .value         */
    int64_t imm; /* compile-time stride (valid iff !is_value)    */
    rocke_value_t* value; /* runtime SSA stride (valid iff is_value)      */
} rocke_stride_t;

static inline rocke_stride_t rocke_stride_imm(int64_t v)
{
    rocke_stride_t s;
    s.is_value = false;
    s.imm = v;
    s.value = NULL;
    return s;
}

static inline rocke_stride_t rocke_stride_value(rocke_value_t* v)
{
    rocke_stride_t s;
    s.is_value = true;
    s.imm = 0;
    s.value = v;
    return s;
}

/* ----------------------------------------------------------- TensorDescriptor */

/* Pure shape + strides + dtype; no SSA at construction. Analogue of CK Tile's
 * ``tensor_descriptor``. */
typedef struct rocke_tensor_descriptor
{
    int rank;
    int shape[ROCKE_TV_MAX_RANK]; /* element extents               */
    rocke_stride_t strides[ROCKE_TV_MAX_RANK];
    const rocke_type_t* dtype;
} rocke_tensor_descriptor_t;

/* TensorDescriptor.__init__ with rank validation (Python __post_init__).
 * Returns ROCKE_OK, or ROCKE_ERR_VALUE on shape/strides rank mismatch or empty. */
rocke_status_t rocke_tensor_descriptor_init(rocke_tensor_descriptor_t* out,
                                            const int* shape,
                                            const rocke_stride_t* strides,
                                            int rank,
                                            const rocke_type_t* dtype);

/* TensorDescriptor.packed(shape, dtype) -- row-major packed strides. */
rocke_status_t rocke_tensor_descriptor_packed(rocke_tensor_descriptor_t* out,
                                              const int* shape,
                                              int rank,
                                              const rocke_type_t* dtype);

/* TensorDescriptor.with_strides(shape, strides, dtype). */
rocke_status_t rocke_tensor_descriptor_with_strides(rocke_tensor_descriptor_t* out,
                                                    const int* shape,
                                                    const rocke_stride_t* strides,
                                                    int rank,
                                                    const rocke_type_t* dtype);

/* @property rank / numel. */
int rocke_tensor_descriptor_rank(const rocke_tensor_descriptor_t* d);
int64_t rocke_tensor_descriptor_numel(const rocke_tensor_descriptor_t* d);

/* TensorDescriptor.offset(b, indices): flat element offset (SSA). Emits the
 * same mul/add chain as Python (literal-1 stride omitted, const stride folded
 * into const_i32 mul). Returns NULL with builder error on rank mismatch. */
rocke_value_t* rocke_tensor_descriptor_offset(rocke_ir_builder_t* b,
                                              const rocke_tensor_descriptor_t* d,
                                              rocke_value_t* const* indices,
                                              int num_indices);

/* ------------------------------------------------------------- BufferResource */

/* Python ``BufferResource``: an AMDGPU 128-bit buffer-resource descriptor used
 * by the bounds-checked raw_ptr_buffer_load / raw_ptr_buffer_store family.
 * A value type (the Python dataclass is frozen). An all-zero value (rsrc ==
 * NULL) means "no buffer resource".
 *
 * Named rocke_tv_buffer_resource_t, not rocke_buffer_resource_t: pooling.cpp
 * and the conv instances keep file-private mirrors under the plain names, and
 * a shared type with the same tag but another layout would break the C++
 * one-definition rule. */
typedef struct rocke_tv_buffer_resource
{
    rocke_value_t* rsrc; /* <4 x i32> descriptor from b.buffer_rsrc(...)     */
    rocke_value_t* soffset; /* scalar byte offset; default const_i32(0)      */
    int64_t num_bytes; /* informational; Python keeps it only for a host int */
} rocke_tv_buffer_resource_t;

/* make_buffer_resource(b, ptr, num_bytes=, soffset=None). Emits
 * b.buffer_rsrc(ptr, num_bytes), then const_i32(0) when soffset is NULL (the
 * Python default), in that order. num_bytes is an SSA Value, so the
 * informational out->num_bytes is 0 (Python: int(num_bytes) only for an int). */
rocke_status_t rocke_tv_make_buffer_resource(rocke_ir_builder_t* b,
                                             rocke_tv_buffer_resource_t* out,
                                             rocke_value_t* ptr,
                                             rocke_value_t* num_bytes,
                                             rocke_value_t* soffset /* NULL => const_i32(0) */);

/* ----------------------------------------------------------------- TensorView */

/* pointer + descriptor + address space. Analogue of CK Tile's tensor_view.
 * For ROCKE_ADDR_GLOBAL / ROCKE_ADDR_LDS, ``base`` is the pointer/smem token.
 * For ROCKE_ADDR_BUFFER the Python ``base`` is a BufferResource; here it lives
 * in ``buffer`` (appended last so every existing initializer is unchanged) and
 * ``base`` is NULL. ``buffer`` is read only when addr_space ==
 * ROCKE_ADDR_BUFFER, which only rocke_tv_make_buffer_view sets.
 */
typedef struct rocke_tensor_view
{
    rocke_value_t* base;
    rocke_tensor_descriptor_t desc;
    rocke_addr_space_t addr_space;
    rocke_tv_buffer_resource_t buffer; /* addr_space == ROCKE_ADDR_BUFFER only */
} rocke_tensor_view_t;

/* @property dtype / shape / rank. */
const rocke_type_t* rocke_tensor_view_dtype(const rocke_tensor_view_t* v);
int rocke_tensor_view_rank(const rocke_tensor_view_t* v);

/* @property buffer: the view's BufferResource. Like Python, TypeError
 * (ROCKE_ERR_TYPE on ``b``) when addr_space is not ROCKE_ADDR_BUFFER or the
 * view holds no buffer resource; returns NULL then. */
const rocke_tv_buffer_resource_t* rocke_tensor_view_buffer(rocke_ir_builder_t* b,
                                                           const rocke_tensor_view_t* v);

/* TensorView.load_scalar(b, indices). */
rocke_value_t* rocke_tensor_view_load_scalar(rocke_ir_builder_t* b,
                                             const rocke_tensor_view_t* v,
                                             rocke_value_t* const* indices,
                                             int num_indices);

/* TensorView.store_scalar(b, indices, value, align). Pass align<=0 for the
 * Python default (align=None). */
void rocke_tensor_view_store_scalar(rocke_ir_builder_t* b,
                                    const rocke_tensor_view_t* v,
                                    rocke_value_t* const* indices,
                                    int num_indices,
                                    rocke_value_t* value,
                                    int align);

/* TensorView.load_vec(b, indices, n). */
rocke_value_t* rocke_tensor_view_load_vec(rocke_ir_builder_t* b,
                                          const rocke_tensor_view_t* v,
                                          rocke_value_t* const* indices,
                                          int num_indices,
                                          int n);

/* TensorView.store_vec(b, indices, value, n). */
void rocke_tensor_view_store_vec(rocke_ir_builder_t* b,
                                 const rocke_tensor_view_t* v,
                                 rocke_value_t* const* indices,
                                 int num_indices,
                                 rocke_value_t* value,
                                 int n);

/* TensorView.load_vec_at(b, elem_off, n) with mask=None. */
rocke_value_t* rocke_tensor_view_load_vec_at(rocke_ir_builder_t* b,
                                             const rocke_tensor_view_t* v,
                                             rocke_value_t* elem_off,
                                             int n);

/* TensorView.load_vec_at(b, elem_off, n, mask=mask). ``mask`` is an i1 Value
 * or NULL (Python None, identical to rocke_tensor_view_load_vec_at). A mask on
 * a non-buffer view is NotImplementedError (ROCKE_ERR_NOTIMPL), as in Python. */
rocke_value_t* rocke_tensor_view_load_vec_at_masked(rocke_ir_builder_t* b,
                                                    const rocke_tensor_view_t* v,
                                                    rocke_value_t* elem_off,
                                                    int n,
                                                    rocke_value_t* mask);

/* TensorView.store_vec_at(b, elem_off, value, n) with mask=None. */
void rocke_tensor_view_store_vec_at(rocke_ir_builder_t* b,
                                    const rocke_tensor_view_t* v,
                                    rocke_value_t* elem_off,
                                    rocke_value_t* value,
                                    int n);

/* TensorView.store_vec_at(b, elem_off, value, n, mask=mask). Same ``mask``
 * contract as rocke_tensor_view_load_vec_at_masked. */
void rocke_tensor_view_store_vec_at_masked(rocke_ir_builder_t* b,
                                           const rocke_tensor_view_t* v,
                                           rocke_value_t* elem_off,
                                           rocke_value_t* value,
                                           int n,
                                           rocke_value_t* mask);

/* ----------------------------------------------------------------- TileWindow */

/* A fixed-extent window into a TensorView. ``view`` is held by pointer (the
 * Python dataclass holds a reference); ``origin`` are SSA Values. */
typedef struct rocke_tile_window
{
    const rocke_tensor_view_t* view;
    int rank;
    int lengths[ROCKE_TV_MAX_RANK];
    rocke_value_t* origin[ROCKE_TV_MAX_RANK];
} rocke_tile_window_t;

/* @property rank / dtype / addr_space. */
int rocke_tile_window_rank(const rocke_tile_window_t* w);
const rocke_type_t* rocke_tile_window_dtype(const rocke_tile_window_t* w);
rocke_addr_space_t rocke_tile_window_addr_space(const rocke_tile_window_t* w);

/* TileWindow.load_vec(b, *local_indices, n). */
rocke_value_t* rocke_tile_window_load_vec(rocke_ir_builder_t* b,
                                          const rocke_tile_window_t* w,
                                          rocke_value_t* const* local_indices,
                                          int num_indices,
                                          int n);

/* TileWindow.store_vec(b, *local_indices, value, n). */
void rocke_tile_window_store_vec(rocke_ir_builder_t* b,
                                 const rocke_tile_window_t* w,
                                 rocke_value_t* const* local_indices,
                                 int num_indices,
                                 rocke_value_t* value,
                                 int n);

/* TileWindow.load_vec_scalars(b, *local_indices, n): vector load split into
 * ``n`` scalars in the window's own dtype (no f32 promotion). Writes them to
 * out[0..n) (caller-provided, length >= n). */
void rocke_tile_window_load_vec_scalars(rocke_ir_builder_t* b,
                                        const rocke_tile_window_t* w,
                                        rocke_value_t* const* local_indices,
                                        int num_indices,
                                        int n,
                                        rocke_value_t** out);

/* TileWindow.load_scalar(b, *local_indices). */
rocke_value_t* rocke_tile_window_load_scalar(rocke_ir_builder_t* b,
                                             const rocke_tile_window_t* w,
                                             rocke_value_t* const* local_indices,
                                             int num_indices);

/* TileWindow.store_scalar(b, *local_indices, value, align). Pass align<=0 for
 * the Python default (align=None). */
void rocke_tile_window_store_scalar(rocke_ir_builder_t* b,
                                    const rocke_tile_window_t* w,
                                    rocke_value_t* const* local_indices,
                                    int num_indices,
                                    rocke_value_t* value,
                                    int align);

/* ---------------------------------------------------- module-level factories */

/* TensorView.tile(lengths, origin) -- the method form. */
rocke_status_t rocke_tensor_view_tile(rocke_tile_window_t* out,
                                      const rocke_tensor_view_t* view,
                                      const int* lengths,
                                      rocke_value_t* const* origin,
                                      int rank);

/* make_global_view(base, shape, dtype, strides=None).
 * Pass strides=NULL for packed row-major (the Python default). */
rocke_status_t rocke_make_global_view(rocke_tensor_view_t* out,
                                      rocke_value_t* base,
                                      const int* shape,
                                      int rank,
                                      const rocke_type_t* dtype,
                                      const rocke_stride_t* strides /* NULL => packed */);

/* make_buffer_view(rsrc, shape, dtype, strides=None): wrap a BufferResource
 * as a ROCKE_ADDR_BUFFER view. Emits no IR. strides=NULL => packed row-major.
 * The resource is copied into out->buffer; rsrc=NULL leaves it empty, so a
 * later load/store raises TypeError, as Python does for a non-BufferResource. */
rocke_status_t rocke_tv_make_buffer_view(rocke_tensor_view_t* out,
                                         const rocke_tv_buffer_resource_t* rsrc,
                                         const int* shape,
                                         int rank,
                                         const rocke_type_t* dtype,
                                         const rocke_stride_t* strides /* NULL => packed */);

/* make_tile_window(view, lengths, origin). Free-function alias of
 * TensorView.tile. */
rocke_status_t rocke_make_tile_window(rocke_tile_window_t* out,
                                      const rocke_tensor_view_t* view,
                                      const int* lengths,
                                      rocke_value_t* const* origin,
                                      int rank);

/* make_naive_tensor_view_packed(base, shape, dtype): CK Tile literal-name
 * alias of make_global_view with packed row-major strides. */
rocke_status_t rocke_make_naive_tensor_view_packed(rocke_tensor_view_t* out,
                                                   rocke_value_t* base,
                                                   const int* shape,
                                                   int rank,
                                                   const rocke_type_t* dtype);

/* make_lds_view(b, dtype, shape, name_hint, strides=NULL): allocate an
 * addrspace(3) buffer for the kernel lifetime and return a view over it.
 * strides=NULL => packed row-major. */
rocke_status_t rocke_make_lds_view(rocke_ir_builder_t* b,
                                   rocke_tensor_view_t* out,
                                   const rocke_type_t* dtype,
                                   const int* shape,
                                   int rank,
                                   const char* name_hint,
                                   const rocke_stride_t* strides /* NULL => packed */);

/* Like rocke_make_lds_view but forwards `exclusive` to smem_alloc (cshuffle
 * no-alias mode): the smem-pool packer gives the buffer its own byte range.
 * rocke_make_lds_view is rocke_make_lds_view_ex(..., exclusive=0). */
rocke_status_t rocke_make_lds_view_ex(rocke_ir_builder_t* b,
                                      rocke_tensor_view_t* out,
                                      const rocke_type_t* dtype,
                                      const int* shape,
                                      int rank,
                                      const char* name_hint,
                                      const rocke_stride_t* strides /* NULL => packed */,
                                      int exclusive);

/* ----------------------------------------- compute-promoting (f32) peers */

/* TensorView.load_vec_as_f32(b, indices, n): vector load + per-lane f32
 * promotion. Writes the ``n`` f32 SSA scalars to out[0..n) (length >= n).
 * For an f32 view the per-lane cast is a no-op (elements are extracted
 * directly); n==1 routes through load_scalar. */
void rocke_tensor_view_load_vec_as_f32(rocke_ir_builder_t* b,
                                       const rocke_tensor_view_t* v,
                                       rocke_value_t* const* indices,
                                       int num_indices,
                                       int n,
                                       rocke_value_t** out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_TENSOR_VIEW_H */
