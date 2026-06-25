// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of ck_dsl.helpers.tensor_view --- the byte-identical builder-call
 * sequence for make_global_view / make_tile_window / TensorDescriptor /
 * TensorView.
 *
 * Every IR-emitting routine here mirrors its Python counterpart line for line:
 * same builder calls, same order, same arguments. Dispatch on dtype keys off
 * ``dtype->name`` exactly as Python keys off ``dtype.name``.
 */

#include "ckc/helper_ck_dsl.helpers.tensor_view.h"

#include "ckc/arena.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err for Python-NotImplementedError parity */

#include <string.h>

/* ------------------------------------------------------------------ helpers */

static bool ckc_tv_name_is(const ckc_type_t* t, const char* name)
{
    return t != NULL && t->name != NULL && strcmp(t->name, name) == 0;
}

/* Python ``_dtype_elem_bytes`` -- used only by the buffer path (out of scope
 * this phase) but kept for parity with the module surface. The unused attribute
 * documents that this is an intentionally-retained parity anchor. */
__attribute__((unused)) static int ckc_tv_dtype_elem_bytes(const ckc_type_t* dtype)
{
    if(ckc_tv_name_is(dtype, "f16") || ckc_tv_name_is(dtype, "bf16"))
        return 2;
    if(ckc_tv_name_is(dtype, "f32"))
        return 4;
    if(ckc_tv_name_is(dtype, "i32"))
        return 4;
    if(ckc_tv_name_is(dtype, "i64"))
        return 8;
    return 0; /* Python raises NotImplementedError */
}

/* ======================================================== TensorDescriptor */

ckc_status_t ckc_tensor_descriptor_init(ckc_tensor_descriptor_t* out,
                                        const int* shape,
                                        const ckc_stride_t* strides,
                                        int rank,
                                        const ckc_type_t* dtype)
{
    int i;
    if(out == NULL || shape == NULL || strides == NULL || dtype == NULL)
        return CKC_ERR_VALUE;
    /* Python __post_init__: shape rank must equal strides rank; must be >= 1. */
    if(rank <= 0) /* "TensorDescriptor must have at least one dimension" */
        return CKC_ERR_VALUE;
    if(rank > CKC_TV_MAX_RANK)
        return CKC_ERR_VALUE;
    out->rank = rank;
    out->dtype = dtype;
    for(i = 0; i < rank; ++i)
    {
        out->shape[i] = shape[i];
        out->strides[i] = strides[i];
    }
    return CKC_OK;
}

ckc_status_t ckc_tensor_descriptor_packed(ckc_tensor_descriptor_t* out,
                                          const int* shape,
                                          int rank,
                                          const ckc_type_t* dtype)
{
    /* Python TensorDescriptor.packed: stride[i] = product of dims with index>i,
     * computed by walking reversed(shape) accumulating ``prod``. */
    int i;
    int64_t prod;
    ckc_stride_t strides[CKC_TV_MAX_RANK];
    if(out == NULL || shape == NULL || dtype == NULL)
        return CKC_ERR_VALUE;
    if(rank <= 0 || rank > CKC_TV_MAX_RANK)
        return CKC_ERR_VALUE;
    prod = 1;
    for(i = rank - 1; i >= 0; --i)
    {
        strides[i] = ckc_stride_imm(prod);
        prod *= (int64_t)shape[i];
    }
    return ckc_tensor_descriptor_init(out, shape, strides, rank, dtype);
}

ckc_status_t ckc_tensor_descriptor_with_strides(ckc_tensor_descriptor_t* out,
                                                const int* shape,
                                                const ckc_stride_t* strides,
                                                int rank,
                                                const ckc_type_t* dtype)
{
    /* Python with_strides: same as init; an int stride stays int, a Value
     * stride stays Value. Our ckc_stride_t already carries the tag. */
    return ckc_tensor_descriptor_init(out, shape, strides, rank, dtype);
}

int ckc_tensor_descriptor_rank(const ckc_tensor_descriptor_t* d)
{
    return d ? d->rank : 0;
}

int64_t ckc_tensor_descriptor_numel(const ckc_tensor_descriptor_t* d)
{
    int64_t n = 1;
    int i;
    if(d == NULL)
        return 0;
    for(i = 0; i < d->rank; ++i)
        n *= (int64_t)d->shape[i];
    return n;
}

ckc_value_t* ckc_tensor_descriptor_offset(ckc_ir_builder_t* b,
                                          const ckc_tensor_descriptor_t* d,
                                          ckc_value_t* const* indices,
                                          int num_indices)
{
    /* Python TensorDescriptor.offset:
     *   off = None
     *   for idx, stride in zip(indices, self.strides):
     *       if isinstance(stride, Value):   term = b.mul(idx, stride)
     *       elif int(stride) == 1:          term = idx
     *       else:                           term = b.mul(idx, b.const_i32(stride))
     *       off = term if off is None else b.add(off, term)
     *   return off if off is not None else b.const_i32(0)
     */
    int i;
    ckc_value_t* off = NULL;
    if(b == NULL)
        return NULL;
    if(d == NULL || indices == NULL)
        return NULL;
    if(num_indices != d->rank)
    {
        /* Python: raise ValueError(f"expected {self.rank} indices, got ...") */
        return NULL;
    }
    for(i = 0; i < d->rank; ++i)
    {
        ckc_value_t* term;
        ckc_value_t* idx = indices[i];
        const ckc_stride_t* st = &d->strides[i];
        if(st->is_value)
        {
            term = ckc_b_mul(b, idx, st->value);
        }
        else if(st->imm == 1)
        {
            term = idx;
        }
        else
        {
            term = ckc_b_mul(b, idx, ckc_b_const_i32(b, st->imm));
        }
        off = (off == NULL) ? term : ckc_b_add(b, off, term);
    }
    return off != NULL ? off : ckc_b_const_i32(b, 0);
}

/* ============================================================= TensorView */

const ckc_type_t* ckc_tensor_view_dtype(const ckc_tensor_view_t* v)
{
    return v ? v->desc.dtype : NULL;
}

int ckc_tensor_view_rank(const ckc_tensor_view_t* v)
{
    return v ? v->desc.rank : 0;
}

ckc_value_t* ckc_tensor_view_load_scalar(ckc_ir_builder_t* b,
                                         const ckc_tensor_view_t* v,
                                         ckc_value_t* const* indices,
                                         int num_indices)
{
    /* Python TensorView.load_scalar. */
    ckc_value_t* off;
    const ckc_type_t* dt;
    if(b == NULL || v == NULL)
        return NULL;
    off = ckc_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    dt = v->desc.dtype;

    if(v->addr_space == CKC_ADDR_LDS)
    {
        /* LDS scalar load goes through smem_load_vN with n=1, then extract 0. */
        if(ckc_tv_name_is(dt, "f16") || ckc_tv_name_is(dt, "bf16"))
        {
            ckc_value_t* vec = ckc_b_smem_load_vN(b, v->base, indices, num_indices, dt, 1);
            return ckc_b_vec_extract(b, vec, 0);
        }
        if(ckc_tv_name_is(dt, "f32"))
        {
            ckc_value_t* vec = ckc_b_smem_load_vN_f32(b, v->base, indices, num_indices, 1);
            return ckc_b_vec_extract(b, vec, 0);
        }
        /* Python: NotImplementedError */
        return NULL;
    }
    if(v->addr_space == CKC_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view load_scalar): Python's buffer branch reads
         * self.buffer, a BufferResource carrying {rsrc, soffset, num_bytes},
         * then emits b.buffer_load_f16(rsrc.rsrc, byte_off, rsrc.soffset). The
         * builder prims exist (ckc_b_buffer_rsrc / ckc_b_buffer_load_f16), but
         * ckc_tensor_view_t.base is a bare ckc_value_t* with no soffset/
         * num_bytes slots, so there is nowhere to hold the BufferResource. A
         * faithful port REQUIRES adding buffer-resource fields to the shared
         * ckc_tensor_view_t struct (a header change in
         * helper_ck_dsl.helpers.tensor_view.h, included by many TUs). No
         * producer in the current C scope ever builds a CKC_ADDR_BUFFER view
         * through this shared path (instances roll private buffer-resource
         * structs), so this branch is unreachable; left unported to avoid a
         * cross-TU header change for dead code. */
        return NULL;
    }
    /* global */
    if(ckc_tv_name_is(dt, "f16"))
        return ckc_b_global_load_f16(b, v->base, off, 0);
    if(ckc_tv_name_is(dt, "bf16"))
        return ckc_b_global_load_bf16(b, v->base, off, 0);
    if(ckc_tv_name_is(dt, "f32"))
        return ckc_b_global_load_f32(b, v->base, off, 0);
    if(ckc_tv_name_is(dt, "i32"))
        return ckc_b_global_load_i32(b, v->base, off, 0);
    if(ckc_tv_name_is(dt, "i64"))
        return ckc_b_global_load_i64(b, v->base, off, 0);
    return ckc_b_global_load(b, v->base, off, dt, 0);
}

void ckc_tensor_view_store_scalar(ckc_ir_builder_t* b,
                                  const ckc_tensor_view_t* v,
                                  ckc_value_t* const* indices,
                                  int num_indices,
                                  ckc_value_t* value,
                                  int align)
{
    /* Python TensorView.store_scalar. */
    const ckc_type_t* dt;
    ckc_value_t* off;
    if(b == NULL || v == NULL)
        return;
    dt = v->desc.dtype;

    if(v->addr_space == CKC_ADDR_LDS)
    {
        if(ckc_tv_name_is(dt, "f16") || ckc_tv_name_is(dt, "bf16"))
        {
            ckc_b_smem_store_vN(b, v->base, indices, num_indices, value, 1);
            return;
        }
        if(ckc_tv_name_is(dt, "f32"))
        {
            ckc_b_smem_store_vN_f32(b, v->base, indices, num_indices, value, 1);
            return;
        }
        return; /* Python NotImplementedError */
    }
    if(v->addr_space == CKC_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view store_scalar): mirrors load_scalar. Python
         * emits b.buffer_store_f16(rsrc.rsrc, byte_off, rsrc.soffset, value)
         * from self.buffer (a BufferResource). Blocked on the same missing
         * buffer-resource fields in the shared ckc_tensor_view_t struct (would
         * require a cross-TU header change). Unreachable in current C scope. */
        return;
    }
    off = ckc_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    if(align <= 0)
        ckc_b_global_store(b, v->base, off, value, 0); /* Python: align default 1 */
    else
        ckc_b_global_store(b, v->base, off, value, align);
}

ckc_value_t* ckc_tensor_view_load_vec(ckc_ir_builder_t* b,
                                      const ckc_tensor_view_t* v,
                                      ckc_value_t* const* indices,
                                      int num_indices,
                                      int n)
{
    /* Python TensorView.load_vec. */
    const ckc_type_t* dt;
    ckc_value_t* off;
    if(b == NULL || v == NULL)
        return NULL;
    dt = v->desc.dtype;

    if(v->addr_space == CKC_ADDR_LDS)
    {
        if(ckc_tv_name_is(dt, "f16") || ckc_tv_name_is(dt, "bf16"))
            return ckc_b_smem_load_vN(b, v->base, indices, num_indices, dt, n);
        if(ckc_tv_name_is(dt, "f32"))
            return ckc_b_smem_load_vN_f32(b, v->base, indices, num_indices, n);
        return NULL; /* Python NotImplementedError */
    }
    if(v->addr_space == CKC_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view load_vec): Python emits
         * b.buffer_load_vN_f16(rsrc.rsrc, byte_off, rsrc.soffset, dwords=n/2)
         * for f16 (n in {2,4,8}) from self.buffer. Builder prim
         * ckc_b_buffer_load_vN_f16 exists; blocked on the missing
         * buffer-resource fields in shared ckc_tensor_view_t (cross-TU header
         * change). Unreachable in current C scope. */
        return NULL;
    }
    off = ckc_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    if(ckc_tv_name_is(dt, "f16") || ckc_tv_name_is(dt, "bf16"))
        return ckc_b_global_load_vN(b, v->base, off, dt, n, 0);
    if(ckc_tv_name_is(dt, "f32"))
    {
        /* Python TensorView.load_vec f32 branch: f32 global vec loads aren't
         * wired through global_load_vN (the vN primitive only covers 16-bit
         * elements), so fall back to n scalar global_load_f32 + a vec_pack.
         *   scalars = [b.global_load_f32(base, b.add(off, b.const_i32(i)))
         *              for i in range(n)]
         *   return b.vec_pack(scalars, self.dtype)
         */
        ckc_value_t* scalars[CKC_TV_MAX_VEC];
        int i;
        if(n < 1 || n > CKC_TV_MAX_VEC)
            return NULL;
        for(i = 0; i < n; ++i)
        {
            ckc_value_t* eoff = ckc_b_add(b, off, ckc_b_const_i32(b, i));
            scalars[i] = ckc_b_global_load_f32(b, v->base, eoff, 0);
        }
        return ckc_b_vec_pack(b, scalars, n, dt);
    }
    return NULL; /* Python NotImplementedError */
}

void ckc_tensor_view_store_vec(ckc_ir_builder_t* b,
                               const ckc_tensor_view_t* v,
                               ckc_value_t* const* indices,
                               int num_indices,
                               ckc_value_t* value,
                               int n)
{
    /* Python TensorView.store_vec. */
    ckc_value_t* off;
    if(b == NULL || v == NULL)
        return;

    if(v->addr_space == CKC_ADDR_LDS)
    {
        ckc_b_smem_store_vN(b, v->base, indices, num_indices, value, n);
        return;
    }
    if(v->addr_space == CKC_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view store_vec): Python emits
         * b.buffer_store_vN_f16(rsrc.rsrc, byte_off, rsrc.soffset, ...) from
         * self.buffer. Builder prim ckc_b_buffer_store_vN_f16 exists; blocked
         * on the missing buffer-resource fields in shared ckc_tensor_view_t
         * (cross-TU header change). Unreachable in current C scope. */
        return;
    }
    off = ckc_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    ckc_b_global_store_vN(b, v->base, off, value, n, 0);
}

ckc_value_t* ckc_tensor_view_load_vec_at(ckc_ir_builder_t* b,
                                         const ckc_tensor_view_t* v,
                                         ckc_value_t* elem_off,
                                         int n)
{
    /* Python TensorView.load_vec_at (no mask path here -- mask requires
     * addr_space="buffer", which is out of phase scope). */
    const ckc_type_t* dt;
    if(b == NULL || v == NULL)
        return NULL;
    dt = v->desc.dtype;

    if(v->addr_space == CKC_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view load_vec_at): the buffer branch additionally
         * carries the bounds-checked mask path keyed on the BufferResource.
         * Blocked on the missing buffer-resource fields in shared
         * ckc_tensor_view_t (cross-TU header change). Unreachable in current C
         * scope. */
        return NULL;
    }
    if(v->addr_space == CKC_ADDR_LDS)
    {
        ckc_value_t* idx1[1];
        idx1[0] = elem_off;
        if(!ckc_tv_name_is(dt, "f16") && !ckc_tv_name_is(dt, "bf16") && !ckc_tv_name_is(dt, "f32")
           && !ckc_tv_name_is(dt, "i32"))
            return NULL; /* Python NotImplementedError */
        if(ckc_tv_name_is(dt, "f32"))
            return ckc_b_smem_load_vN_f32(b, v->base, idx1, 1, n);
        return ckc_b_smem_load_vN(b, v->base, idx1, 1, dt, n);
    }
    /* global */
    if(ckc_tv_name_is(dt, "f16") || ckc_tv_name_is(dt, "bf16") || ckc_tv_name_is(dt, "f32")
       || ckc_tv_name_is(dt, "i32"))
        return ckc_b_global_load_vN(b, v->base, elem_off, dt, n, 0);
    return NULL; /* Python NotImplementedError */
}

void ckc_tensor_view_store_vec_at(ckc_ir_builder_t* b,
                                  const ckc_tensor_view_t* v,
                                  ckc_value_t* elem_off,
                                  ckc_value_t* value,
                                  int n)
{
    /* Python TensorView.store_vec_at (no mask path -- buffer out of scope). */
    if(b == NULL || v == NULL)
        return;
    if(v->addr_space == CKC_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view store_vec_at): mirrors load_vec_at. Blocked on
         * the missing buffer-resource fields in shared ckc_tensor_view_t
         * (cross-TU header change). Unreachable in current C scope. */
        return;
    }
    ckc_b_global_store_vN(b, v->base, elem_off, value, n, 0);
}

ckc_status_t ckc_tensor_view_tile(ckc_tile_window_t* out,
                                  const ckc_tensor_view_t* view,
                                  const int* lengths,
                                  ckc_value_t* const* origin,
                                  int rank)
{
    /* Python TensorView.tile -> TileWindow(view, lengths, origin). The
     * TileWindow.__post_init__ enforces tile rank == view rank and origin rank
     * == view rank. */
    int i;
    if(out == NULL || view == NULL || lengths == NULL || origin == NULL)
        return CKC_ERR_VALUE;
    if(rank != view->desc.rank) /* "tile rank != view rank" / "origin rank ..." */
        return CKC_ERR_VALUE;
    if(rank > CKC_TV_MAX_RANK)
        return CKC_ERR_VALUE;
    out->view = view;
    out->rank = rank;
    for(i = 0; i < rank; ++i)
    {
        out->lengths[i] = lengths[i];
        out->origin[i] = origin[i];
    }
    return CKC_OK;
}

/* ============================================================= TileWindow */

int ckc_tile_window_rank(const ckc_tile_window_t* w)
{
    return w ? w->rank : 0;
}

const ckc_type_t* ckc_tile_window_dtype(const ckc_tile_window_t* w)
{
    return (w && w->view) ? w->view->desc.dtype : NULL;
}

ckc_addr_space_t ckc_tile_window_addr_space(const ckc_tile_window_t* w)
{
    return (w && w->view) ? w->view->addr_space : CKC_ADDR_GLOBAL;
}

/* TileWindow._global_indices: per-dim add(origin, local_index). */
static int ckc_tile_window_global_indices(ckc_ir_builder_t* b,
                                          const ckc_tile_window_t* w,
                                          ckc_value_t* const* local_indices,
                                          int num_indices,
                                          ckc_value_t** out_global)
{
    int i;
    if(num_indices != w->rank)
        return 0; /* Python: ValueError "local index rank != window rank" */
    for(i = 0; i < w->rank; ++i)
        out_global[i] = ckc_b_add(b, w->origin[i], local_indices[i]);
    return 1;
}

ckc_value_t* ckc_tile_window_load_vec(ckc_ir_builder_t* b,
                                      const ckc_tile_window_t* w,
                                      ckc_value_t* const* local_indices,
                                      int num_indices,
                                      int n)
{
    ckc_value_t* gidx[CKC_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return NULL;
    if(!ckc_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return NULL;
    return ckc_tensor_view_load_vec(b, w->view, gidx, w->rank, n);
}

void ckc_tile_window_store_vec(ckc_ir_builder_t* b,
                               const ckc_tile_window_t* w,
                               ckc_value_t* const* local_indices,
                               int num_indices,
                               ckc_value_t* value,
                               int n)
{
    ckc_value_t* gidx[CKC_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return;
    if(!ckc_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return;
    ckc_tensor_view_store_vec(b, w->view, gidx, w->rank, value, n);
}

ckc_value_t* ckc_tile_window_load_scalar(ckc_ir_builder_t* b,
                                         const ckc_tile_window_t* w,
                                         ckc_value_t* const* local_indices,
                                         int num_indices)
{
    ckc_value_t* gidx[CKC_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return NULL;
    if(!ckc_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return NULL;
    return ckc_tensor_view_load_scalar(b, w->view, gidx, w->rank);
}

void ckc_tile_window_store_scalar(ckc_ir_builder_t* b,
                                  const ckc_tile_window_t* w,
                                  ckc_value_t* const* local_indices,
                                  int num_indices,
                                  ckc_value_t* value,
                                  int align)
{
    ckc_value_t* gidx[CKC_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return;
    if(!ckc_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return;
    ckc_tensor_view_store_scalar(b, w->view, gidx, w->rank, value, align);
}

/* ---- compute-promoting vector ops (TileWindow) ----
 *
 * These mirror Python TileWindow.load_vec_as_f32 / store_vec_from_f32 line for
 * line. They are declared in helper_ck_dsl.helpers.sweep.h (the sweep TU's
 * "TileWindow peers"); their home is here in the tensor_view port so the sweep
 * helpers resolve them at link time.
 */

/* C++ build: these are declared (and called) as extern "C" via
 * helper_ck_dsl.helpers.sweep.h, but that header is not included here. Re-declare
 * them extern "C" so the definitions below take C linkage and link against the
 * sweep callers without name mangling. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
void ckc_tile_window_load_vec_as_f32(ckc_ir_builder_t* b,
                                     const ckc_tile_window_t* w,
                                     ckc_value_t* const* local_indices,
                                     int num_indices,
                                     int n,
                                     ckc_value_t** out);
void ckc_tile_window_store_vec_from_f32(ckc_ir_builder_t* b,
                                        const ckc_tile_window_t* w,
                                        ckc_value_t* const* local_indices,
                                        int num_indices,
                                        ckc_value_t* const* values,
                                        int num_values);
#ifdef __cplusplus
}
#endif

void ckc_tile_window_load_vec_as_f32(ckc_ir_builder_t* b,
                                     const ckc_tile_window_t* w,
                                     ckc_value_t* const* local_indices,
                                     int num_indices,
                                     int n,
                                     ckc_value_t** out)
{
    /* Python TileWindow.load_vec_as_f32:
     *   if n == 1:
     *       scalar = self.load_scalar(b, *local_indices)
     *       return [b.cast_to_f32(scalar)]
     *   v = self.load_vec(b, *local_indices, n=n)
     *   return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)]
     */
    int i;
    if(b == NULL || w == NULL || w->view == NULL || out == NULL)
        return;
    if(n == 1)
    {
        ckc_value_t* scalar = ckc_tile_window_load_scalar(b, w, local_indices, num_indices);
        out[0] = ckc_b_cast_to_f32(b, scalar);
        return;
    }
    {
        ckc_value_t* v = ckc_tile_window_load_vec(b, w, local_indices, num_indices, n);
        for(i = 0; i < n; ++i)
            out[i] = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, v, i));
    }
}

void ckc_tile_window_store_vec_from_f32(ckc_ir_builder_t* b,
                                        const ckc_tile_window_t* w,
                                        ckc_value_t* const* local_indices,
                                        int num_indices,
                                        ckc_value_t* const* values,
                                        int num_values)
{
    /* Python TileWindow.store_vec_from_f32:
     *   if self.dtype.name not in ("f16", "bf16"):
     *       raise NotImplementedError(...)
     *   if len(values) == 1:
     *       scalar = b.cast_f32_to(values[0], self.dtype)
     *       self.store_scalar(b, *local_indices, value=scalar)
     *       return
     *   casts = [b.cast_f32_to(v, self.dtype) for v in values]
     *   packed = b.vec_pack(casts, self.dtype)
     *   self.store_vec(b, *local_indices, value=packed, n=len(values))
     */
    const ckc_type_t* dt;
    ckc_value_t** casts;
    ckc_value_t* packed;
    int i;
    if(b == NULL || w == NULL || w->view == NULL)
        return;
    dt = w->view->desc.dtype;
    if(!(ckc_tv_name_is(dt, "f16") || ckc_tv_name_is(dt, "bf16")))
    {
        /* Python: NotImplementedError "store_vec_from_f32 not wired for ...". */
        ckc_i_set_err(b,
                      CKC_ERR_NOTIMPL,
                      "store_vec_from_f32 not wired for %s; "
                      "cast manually and use store_vec",
                      dt && dt->name ? dt->name : "<null>");
        return;
    }
    if(num_values == 1)
    {
        ckc_value_t* scalar = ckc_b_cast_f32_to(b, values[0], dt);
        ckc_tile_window_store_scalar(b, w, local_indices, num_indices, scalar, 0);
        return;
    }
    if(num_values <= 0)
        return;
    casts = (ckc_value_t**)ckc_arena_alloc(&b->arena, (size_t)num_values * sizeof(ckc_value_t*));
    if(casts == NULL)
        return;
    for(i = 0; i < num_values; ++i)
        casts[i] = ckc_b_cast_f32_to(b, values[i], dt);
    packed = ckc_b_vec_pack(b, casts, num_values, dt);
    ckc_tile_window_store_vec(b, w, local_indices, num_indices, packed, num_values);
}

void ckc_tensor_view_load_vec_as_f32(ckc_ir_builder_t* b,
                                     const ckc_tensor_view_t* v,
                                     ckc_value_t* const* indices,
                                     int num_indices,
                                     int n,
                                     ckc_value_t** out)
{
    /* Python TensorView.load_vec_as_f32:
     *   if n == 1:
     *       scalar = self.load_scalar(b, indices)
     *       if self.dtype.name == "f32": return [scalar]
     *       return [b.cast_to_f32(scalar)]
     *   v = self.load_vec(b, indices, n=n)
     *   if self.dtype.name == "f32":
     *       return [b.vec_extract(v, i) for i in range(n)]
     *   return [b.cast_to_f32(b.vec_extract(v, i)) for i in range(n)]
     */
    const ckc_type_t* dt;
    int i;
    if(b == NULL || v == NULL || out == NULL)
        return;
    dt = v->desc.dtype;

    if(n == 1)
    {
        ckc_value_t* scalar = ckc_tensor_view_load_scalar(b, v, indices, num_indices);
        if(ckc_tv_name_is(dt, "f32"))
            out[0] = scalar;
        else
            out[0] = ckc_b_cast_to_f32(b, scalar);
        return;
    }

    {
        ckc_value_t* vec = ckc_tensor_view_load_vec(b, v, indices, num_indices, n);
        if(ckc_tv_name_is(dt, "f32"))
        {
            for(i = 0; i < n; ++i)
                out[i] = ckc_b_vec_extract(b, vec, i);
        }
        else
        {
            for(i = 0; i < n; ++i)
                out[i] = ckc_b_cast_to_f32(b, ckc_b_vec_extract(b, vec, i));
        }
    }
}

/* ==================================================== module-level factories */

ckc_status_t ckc_make_global_view(ckc_tensor_view_t* out,
                                  ckc_value_t* base,
                                  const int* shape,
                                  int rank,
                                  const ckc_type_t* dtype,
                                  const ckc_stride_t* strides)
{
    /* Python make_global_view:
     *   desc = packed(shape, dtype) if strides is None else with_strides(...)
     *   return TensorView(base, desc, addr_space="global")
     */
    ckc_status_t st;
    if(out == NULL)
        return CKC_ERR_VALUE;
    if(strides == NULL)
        st = ckc_tensor_descriptor_packed(&out->desc, shape, rank, dtype);
    else
        st = ckc_tensor_descriptor_with_strides(&out->desc, shape, strides, rank, dtype);
    if(st != CKC_OK)
        return st;
    out->base = base;
    out->addr_space = CKC_ADDR_GLOBAL;
    return CKC_OK;
}

ckc_status_t ckc_make_tile_window(ckc_tile_window_t* out,
                                  const ckc_tensor_view_t* view,
                                  const int* lengths,
                                  ckc_value_t* const* origin,
                                  int rank)
{
    /* Python make_tile_window -> view.tile(lengths, origin). */
    return ckc_tensor_view_tile(out, view, lengths, origin, rank);
}

ckc_status_t ckc_make_naive_tensor_view_packed(
    ckc_tensor_view_t* out, ckc_value_t* base, const int* shape, int rank, const ckc_type_t* dtype)
{
    /* Python make_naive_tensor_view_packed:
     *   return make_global_view(base, shape, dtype)   # packed row-major */
    return ckc_make_global_view(out, base, shape, rank, dtype, NULL);
}

ckc_status_t ckc_make_lds_view(ckc_ir_builder_t* b,
                               ckc_tensor_view_t* out,
                               const ckc_type_t* dtype,
                               const int* shape,
                               int rank,
                               const char* name_hint,
                               const ckc_stride_t* strides /* NULL => packed */)
{
    /* Python make_lds_view:
     *   smem = b.smem_alloc(dtype, list(shape), name_hint=name_hint)
     *   desc = packed(shape, dtype) if strides is None else with_strides(...)
     *   return TensorView(base=smem, desc=desc, addr_space="lds")
     */
    ckc_status_t st;
    ckc_value_t* smem;
    if(b == NULL || out == NULL)
        return CKC_ERR_VALUE;
    smem = ckc_b_smem_alloc(b, dtype, shape, rank, name_hint);
    if(strides == NULL)
        st = ckc_tensor_descriptor_packed(&out->desc, shape, rank, dtype);
    else
        st = ckc_tensor_descriptor_with_strides(&out->desc, shape, strides, rank, dtype);
    if(st != CKC_OK)
        return st;
    out->base = smem;
    out->addr_space = CKC_ADDR_LDS;
    return CKC_OK;
}
