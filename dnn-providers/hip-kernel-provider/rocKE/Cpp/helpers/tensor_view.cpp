// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke.helpers.tensor_view --- the byte-identical builder-call
 * sequence for make_global_view / make_tile_window / TensorDescriptor /
 * TensorView.
 *
 * Every IR-emitting routine here mirrors its Python counterpart line for line:
 * same builder calls, same order, same arguments. Dispatch on dtype keys off
 * ``dtype->name`` exactly as Python keys off ``dtype.name``.
 */

#include "rocke/helper_rocke.helpers.tensor_view.h"

#include "rocke/arena.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err for Python-NotImplementedError parity */

#include <string.h>

/* ------------------------------------------------------------------ helpers */

static bool rocke_tv_name_is(const rocke_type_t* t, const char* name)
{
    return t != NULL && t->name != NULL && strcmp(t->name, name) == 0;
}

/* Python ``_dtype_elem_bytes`` -- used only by the buffer path (out of scope
 * this phase) but kept for parity with the module surface. The unused attribute
 * documents that this is an intentionally-retained parity anchor. */
__attribute__((unused)) static int rocke_tv_dtype_elem_bytes(const rocke_type_t* dtype)
{
    if(rocke_tv_name_is(dtype, "f16") || rocke_tv_name_is(dtype, "bf16"))
        return 2;
    if(rocke_tv_name_is(dtype, "f32"))
        return 4;
    if(rocke_tv_name_is(dtype, "i32"))
        return 4;
    if(rocke_tv_name_is(dtype, "i64"))
        return 8;
    return 0; /* Python raises NotImplementedError */
}

/* ======================================================== TensorDescriptor */

rocke_status_t rocke_tensor_descriptor_init(rocke_tensor_descriptor_t* out,
                                            const int* shape,
                                            const rocke_stride_t* strides,
                                            int rank,
                                            const rocke_type_t* dtype)
{
    int i;
    if(out == NULL || shape == NULL || strides == NULL || dtype == NULL)
        return ROCKE_ERR_VALUE;
    /* Python __post_init__: shape rank must equal strides rank; must be >= 1. */
    if(rank <= 0) /* "TensorDescriptor must have at least one dimension" */
        return ROCKE_ERR_VALUE;
    if(rank > ROCKE_TV_MAX_RANK)
        return ROCKE_ERR_VALUE;
    out->rank = rank;
    out->dtype = dtype;
    for(i = 0; i < rank; ++i)
    {
        out->shape[i] = shape[i];
        out->strides[i] = strides[i];
    }
    return ROCKE_OK;
}

rocke_status_t rocke_tensor_descriptor_packed(rocke_tensor_descriptor_t* out,
                                              const int* shape,
                                              int rank,
                                              const rocke_type_t* dtype)
{
    /* Python TensorDescriptor.packed: stride[i] = product of dims with index>i,
     * computed by walking reversed(shape) accumulating ``prod``. */
    int i;
    int64_t prod;
    rocke_stride_t strides[ROCKE_TV_MAX_RANK];
    if(out == NULL || shape == NULL || dtype == NULL)
        return ROCKE_ERR_VALUE;
    if(rank <= 0 || rank > ROCKE_TV_MAX_RANK)
        return ROCKE_ERR_VALUE;
    prod = 1;
    for(i = rank - 1; i >= 0; --i)
    {
        strides[i] = rocke_stride_imm(prod);
        prod *= (int64_t)shape[i];
    }
    return rocke_tensor_descriptor_init(out, shape, strides, rank, dtype);
}

rocke_status_t rocke_tensor_descriptor_with_strides(rocke_tensor_descriptor_t* out,
                                                    const int* shape,
                                                    const rocke_stride_t* strides,
                                                    int rank,
                                                    const rocke_type_t* dtype)
{
    /* Python with_strides: same as init; an int stride stays int, a Value
     * stride stays Value. Our rocke_stride_t already carries the tag. */
    return rocke_tensor_descriptor_init(out, shape, strides, rank, dtype);
}

int rocke_tensor_descriptor_rank(const rocke_tensor_descriptor_t* d)
{
    return d ? d->rank : 0;
}

int64_t rocke_tensor_descriptor_numel(const rocke_tensor_descriptor_t* d)
{
    int64_t n = 1;
    int i;
    if(d == NULL)
        return 0;
    for(i = 0; i < d->rank; ++i)
        n *= (int64_t)d->shape[i];
    return n;
}

rocke_value_t* rocke_tensor_descriptor_offset(rocke_ir_builder_t* b,
                                              const rocke_tensor_descriptor_t* d,
                                              rocke_value_t* const* indices,
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
    rocke_value_t* off = NULL;
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
        rocke_value_t* term;
        rocke_value_t* idx = indices[i];
        const rocke_stride_t* st = &d->strides[i];
        if(st->is_value)
        {
            term = rocke_b_mul(b, idx, st->value);
        }
        else if(st->imm == 1)
        {
            term = idx;
        }
        else
        {
            term = rocke_b_mul(b, idx, rocke_b_const_i32(b, st->imm));
        }
        off = (off == NULL) ? term : rocke_b_add(b, off, term);
    }
    return off != NULL ? off : rocke_b_const_i32(b, 0);
}

/* ============================================================= TensorView */

const rocke_type_t* rocke_tensor_view_dtype(const rocke_tensor_view_t* v)
{
    return v ? v->desc.dtype : NULL;
}

int rocke_tensor_view_rank(const rocke_tensor_view_t* v)
{
    return v ? v->desc.rank : 0;
}

rocke_value_t* rocke_tensor_view_load_scalar(rocke_ir_builder_t* b,
                                             const rocke_tensor_view_t* v,
                                             rocke_value_t* const* indices,
                                             int num_indices)
{
    /* Python TensorView.load_scalar. */
    rocke_value_t* off;
    const rocke_type_t* dt;
    if(b == NULL || v == NULL)
        return NULL;
    off = rocke_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    dt = v->desc.dtype;

    if(v->addr_space == ROCKE_ADDR_LDS)
    {
        /* LDS scalar load goes through smem_load_vN with n=1, then extract 0. */
        if(rocke_tv_name_is(dt, "f16") || rocke_tv_name_is(dt, "bf16"))
        {
            rocke_value_t* vec = rocke_b_smem_load_vN(b, v->base, indices, num_indices, dt, 1);
            return rocke_b_vec_extract(b, vec, 0);
        }
        if(rocke_tv_name_is(dt, "f32"))
        {
            rocke_value_t* vec = rocke_b_smem_load_vN_f32(b, v->base, indices, num_indices, 1);
            return rocke_b_vec_extract(b, vec, 0);
        }
        /* Python: NotImplementedError */
        return NULL;
    }
    if(v->addr_space == ROCKE_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view load_scalar): Python's buffer branch reads
         * self.buffer, a BufferResource carrying {rsrc, soffset, num_bytes},
         * then emits b.buffer_load_f16(rsrc.rsrc, byte_off, rsrc.soffset). The
         * builder prims exist (rocke_b_buffer_rsrc / rocke_b_buffer_load_f16), but
         * rocke_tensor_view_t.base is a bare rocke_value_t* with no soffset/
         * num_bytes slots, so there is nowhere to hold the BufferResource. A
         * faithful port REQUIRES adding buffer-resource fields to the shared
         * rocke_tensor_view_t struct (a header change in
         * helper_rocke.helpers.tensor_view.h, included by many TUs). No
         * producer in the current C scope ever builds a ROCKE_ADDR_BUFFER view
         * through this shared path (instances roll private buffer-resource
         * structs), so this branch is unreachable; left unported to avoid a
         * cross-TU header change for dead code. */
        return NULL;
    }
    /* global */
    if(rocke_tv_name_is(dt, "f16"))
        return rocke_b_global_load_f16(b, v->base, off, 0);
    if(rocke_tv_name_is(dt, "bf16"))
        return rocke_b_global_load_bf16(b, v->base, off, 0);
    if(rocke_tv_name_is(dt, "f32"))
        return rocke_b_global_load_f32(b, v->base, off, 0);
    if(rocke_tv_name_is(dt, "i32"))
        return rocke_b_global_load_i32(b, v->base, off, 0);
    if(rocke_tv_name_is(dt, "i64"))
        return rocke_b_global_load_i64(b, v->base, off, 0);
    return rocke_b_global_load(b, v->base, off, dt, 0);
}

void rocke_tensor_view_store_scalar(rocke_ir_builder_t* b,
                                    const rocke_tensor_view_t* v,
                                    rocke_value_t* const* indices,
                                    int num_indices,
                                    rocke_value_t* value,
                                    int align)
{
    /* Python TensorView.store_scalar. */
    const rocke_type_t* dt;
    rocke_value_t* off;
    if(b == NULL || v == NULL)
        return;
    dt = v->desc.dtype;

    if(v->addr_space == ROCKE_ADDR_LDS)
    {
        if(rocke_tv_name_is(dt, "f16") || rocke_tv_name_is(dt, "bf16"))
        {
            rocke_b_smem_store_vN(b, v->base, indices, num_indices, value, 1);
            return;
        }
        if(rocke_tv_name_is(dt, "f32"))
        {
            rocke_b_smem_store_vN_f32(b, v->base, indices, num_indices, value, 1);
            return;
        }
        return; /* Python NotImplementedError */
    }
    if(v->addr_space == ROCKE_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view store_scalar): mirrors load_scalar. Python
         * emits b.buffer_store_f16(rsrc.rsrc, byte_off, rsrc.soffset, value)
         * from self.buffer (a BufferResource). Blocked on the same missing
         * buffer-resource fields in the shared rocke_tensor_view_t struct (would
         * require a cross-TU header change). Unreachable in current C scope. */
        return;
    }
    off = rocke_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    if(align <= 0)
        rocke_b_global_store(b, v->base, off, value, 0); /* Python: align default 1 */
    else
        rocke_b_global_store(b, v->base, off, value, align);
}

rocke_value_t* rocke_tensor_view_load_vec(rocke_ir_builder_t* b,
                                          const rocke_tensor_view_t* v,
                                          rocke_value_t* const* indices,
                                          int num_indices,
                                          int n)
{
    /* Python TensorView.load_vec. */
    const rocke_type_t* dt;
    rocke_value_t* off;
    if(b == NULL || v == NULL)
        return NULL;
    dt = v->desc.dtype;

    if(v->addr_space == ROCKE_ADDR_LDS)
    {
        if(rocke_tv_name_is(dt, "f16") || rocke_tv_name_is(dt, "bf16"))
            return rocke_b_smem_load_vN(b, v->base, indices, num_indices, dt, n);
        if(rocke_tv_name_is(dt, "f32"))
            return rocke_b_smem_load_vN_f32(b, v->base, indices, num_indices, n);
        return NULL; /* Python NotImplementedError */
    }
    if(v->addr_space == ROCKE_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view load_vec): Python emits
         * b.buffer_load_vN_f16(rsrc.rsrc, byte_off, rsrc.soffset, dwords=n/2)
         * for f16 (n in {2,4,8}) from self.buffer. Builder prim
         * rocke_b_buffer_load_vN_f16 exists; blocked on the missing
         * buffer-resource fields in shared rocke_tensor_view_t (cross-TU header
         * change). Unreachable in current C scope. */
        return NULL;
    }
    off = rocke_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    if(rocke_tv_name_is(dt, "f16") || rocke_tv_name_is(dt, "bf16"))
        return rocke_b_global_load_vN(b, v->base, off, dt, n, 0);
    if(rocke_tv_name_is(dt, "f32"))
    {
        /* Python TensorView.load_vec f32 branch: f32 global vec loads aren't
         * wired through global_load_vN (the vN primitive only covers 16-bit
         * elements), so fall back to n scalar global_load_f32 + a vec_pack.
         *   scalars = [b.global_load_f32(base, b.add(off, b.const_i32(i)))
         *              for i in range(n)]
         *   return b.vec_pack(scalars, self.dtype)
         */
        rocke_value_t* scalars[ROCKE_TV_MAX_VEC];
        int i;
        if(n < 1 || n > ROCKE_TV_MAX_VEC)
            return NULL;
        for(i = 0; i < n; ++i)
        {
            rocke_value_t* eoff = rocke_b_add(b, off, rocke_b_const_i32(b, i));
            scalars[i] = rocke_b_global_load_f32(b, v->base, eoff, 0);
        }
        return rocke_b_vec_pack(b, scalars, n, dt);
    }
    return NULL; /* Python NotImplementedError */
}

void rocke_tensor_view_store_vec(rocke_ir_builder_t* b,
                                 const rocke_tensor_view_t* v,
                                 rocke_value_t* const* indices,
                                 int num_indices,
                                 rocke_value_t* value,
                                 int n)
{
    /* Python TensorView.store_vec. */
    rocke_value_t* off;
    if(b == NULL || v == NULL)
        return;

    if(v->addr_space == ROCKE_ADDR_LDS)
    {
        rocke_b_smem_store_vN(b, v->base, indices, num_indices, value, n);
        return;
    }
    if(v->addr_space == ROCKE_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view store_vec): Python emits
         * b.buffer_store_vN_f16(rsrc.rsrc, byte_off, rsrc.soffset, ...) from
         * self.buffer. Builder prim rocke_b_buffer_store_vN_f16 exists; blocked
         * on the missing buffer-resource fields in shared rocke_tensor_view_t
         * (cross-TU header change). Unreachable in current C scope. */
        return;
    }
    off = rocke_tensor_descriptor_offset(b, &v->desc, indices, num_indices);
    rocke_b_global_store_vN(b, v->base, off, value, n, 0);
}

rocke_value_t* rocke_tensor_view_load_vec_at(rocke_ir_builder_t* b,
                                             const rocke_tensor_view_t* v,
                                             rocke_value_t* elem_off,
                                             int n)
{
    /* Python TensorView.load_vec_at (no mask path here -- mask requires
     * addr_space="buffer", which is out of phase scope). */
    const rocke_type_t* dt;
    if(b == NULL || v == NULL)
        return NULL;
    dt = v->desc.dtype;

    if(v->addr_space == ROCKE_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view load_vec_at): the buffer branch additionally
         * carries the bounds-checked mask path keyed on the BufferResource.
         * Blocked on the missing buffer-resource fields in shared
         * rocke_tensor_view_t (cross-TU header change). Unreachable in current C
         * scope. */
        return NULL;
    }
    if(v->addr_space == ROCKE_ADDR_LDS)
    {
        rocke_value_t* idx1[1];
        idx1[0] = elem_off;
        if(!rocke_tv_name_is(dt, "f16") && !rocke_tv_name_is(dt, "bf16")
           && !rocke_tv_name_is(dt, "f32") && !rocke_tv_name_is(dt, "i32"))
            return NULL; /* Python NotImplementedError */
        if(rocke_tv_name_is(dt, "f32"))
            return rocke_b_smem_load_vN_f32(b, v->base, idx1, 1, n);
        return rocke_b_smem_load_vN(b, v->base, idx1, 1, dt, n);
    }
    /* global */
    if(rocke_tv_name_is(dt, "f16") || rocke_tv_name_is(dt, "bf16") || rocke_tv_name_is(dt, "f32")
       || rocke_tv_name_is(dt, "i32"))
        return rocke_b_global_load_vN(b, v->base, elem_off, dt, n, 0);
    return NULL; /* Python NotImplementedError */
}

void rocke_tensor_view_store_vec_at(rocke_ir_builder_t* b,
                                    const rocke_tensor_view_t* v,
                                    rocke_value_t* elem_off,
                                    rocke_value_t* value,
                                    int n)
{
    /* Python TensorView.store_vec_at (no mask path -- buffer out of scope). */
    if(b == NULL || v == NULL)
        return;
    if(v->addr_space == ROCKE_ADDR_BUFFER)
    {
        /* NAMED GAP (buffer-view store_vec_at): mirrors load_vec_at. Blocked on
         * the missing buffer-resource fields in shared rocke_tensor_view_t
         * (cross-TU header change). Unreachable in current C scope. */
        return;
    }
    rocke_b_global_store_vN(b, v->base, elem_off, value, n, 0);
}

rocke_status_t rocke_tensor_view_tile(rocke_tile_window_t* out,
                                      const rocke_tensor_view_t* view,
                                      const int* lengths,
                                      rocke_value_t* const* origin,
                                      int rank)
{
    /* Python TensorView.tile -> TileWindow(view, lengths, origin). The
     * TileWindow.__post_init__ enforces tile rank == view rank and origin rank
     * == view rank. */
    int i;
    if(out == NULL || view == NULL || lengths == NULL || origin == NULL)
        return ROCKE_ERR_VALUE;
    if(rank != view->desc.rank) /* "tile rank != view rank" / "origin rank ..." */
        return ROCKE_ERR_VALUE;
    if(rank > ROCKE_TV_MAX_RANK)
        return ROCKE_ERR_VALUE;
    out->view = view;
    out->rank = rank;
    for(i = 0; i < rank; ++i)
    {
        out->lengths[i] = lengths[i];
        out->origin[i] = origin[i];
    }
    return ROCKE_OK;
}

/* ============================================================= TileWindow */

int rocke_tile_window_rank(const rocke_tile_window_t* w)
{
    return w ? w->rank : 0;
}

const rocke_type_t* rocke_tile_window_dtype(const rocke_tile_window_t* w)
{
    return (w && w->view) ? w->view->desc.dtype : NULL;
}

rocke_addr_space_t rocke_tile_window_addr_space(const rocke_tile_window_t* w)
{
    return (w && w->view) ? w->view->addr_space : ROCKE_ADDR_GLOBAL;
}

/* TileWindow._global_indices: per-dim add(origin, local_index). */
static int rocke_tile_window_global_indices(rocke_ir_builder_t* b,
                                            const rocke_tile_window_t* w,
                                            rocke_value_t* const* local_indices,
                                            int num_indices,
                                            rocke_value_t** out_global)
{
    int i;
    if(num_indices != w->rank)
        return 0; /* Python: ValueError "local index rank != window rank" */
    for(i = 0; i < w->rank; ++i)
        out_global[i] = rocke_b_add(b, w->origin[i], local_indices[i]);
    return 1;
}

rocke_value_t* rocke_tile_window_load_vec(rocke_ir_builder_t* b,
                                          const rocke_tile_window_t* w,
                                          rocke_value_t* const* local_indices,
                                          int num_indices,
                                          int n)
{
    rocke_value_t* gidx[ROCKE_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return NULL;
    if(!rocke_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return NULL;
    return rocke_tensor_view_load_vec(b, w->view, gidx, w->rank, n);
}

void rocke_tile_window_store_vec(rocke_ir_builder_t* b,
                                 const rocke_tile_window_t* w,
                                 rocke_value_t* const* local_indices,
                                 int num_indices,
                                 rocke_value_t* value,
                                 int n)
{
    rocke_value_t* gidx[ROCKE_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return;
    if(!rocke_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return;
    rocke_tensor_view_store_vec(b, w->view, gidx, w->rank, value, n);
}

rocke_value_t* rocke_tile_window_load_scalar(rocke_ir_builder_t* b,
                                             const rocke_tile_window_t* w,
                                             rocke_value_t* const* local_indices,
                                             int num_indices)
{
    rocke_value_t* gidx[ROCKE_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return NULL;
    if(!rocke_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return NULL;
    return rocke_tensor_view_load_scalar(b, w->view, gidx, w->rank);
}

void rocke_tile_window_store_scalar(rocke_ir_builder_t* b,
                                    const rocke_tile_window_t* w,
                                    rocke_value_t* const* local_indices,
                                    int num_indices,
                                    rocke_value_t* value,
                                    int align)
{
    rocke_value_t* gidx[ROCKE_TV_MAX_RANK];
    if(b == NULL || w == NULL || w->view == NULL)
        return;
    if(!rocke_tile_window_global_indices(b, w, local_indices, num_indices, gidx))
        return;
    rocke_tensor_view_store_scalar(b, w->view, gidx, w->rank, value, align);
}

/* ---- compute-promoting vector ops (TileWindow) ----
 *
 * These mirror Python TileWindow.load_vec_as_f32 / store_vec_from_f32 line for
 * line. They are declared in helper_rocke.helpers.sweep.h (the sweep TU's
 * "TileWindow peers"); their home is here in the tensor_view port so the sweep
 * helpers resolve them at link time.
 */

/* C++ build: these are declared (and called) as extern "C" via
 * helper_rocke.helpers.sweep.h, but that header is not included here. Re-declare
 * them extern "C" so the definitions below take C linkage and link against the
 * sweep callers without name mangling. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
void rocke_tile_window_load_vec_as_f32(rocke_ir_builder_t* b,
                                       const rocke_tile_window_t* w,
                                       rocke_value_t* const* local_indices,
                                       int num_indices,
                                       int n,
                                       rocke_value_t** out);
void rocke_tile_window_store_vec_from_f32(rocke_ir_builder_t* b,
                                          const rocke_tile_window_t* w,
                                          rocke_value_t* const* local_indices,
                                          int num_indices,
                                          rocke_value_t* const* values,
                                          int num_values);
#ifdef __cplusplus
}
#endif

void rocke_tile_window_load_vec_as_f32(rocke_ir_builder_t* b,
                                       const rocke_tile_window_t* w,
                                       rocke_value_t* const* local_indices,
                                       int num_indices,
                                       int n,
                                       rocke_value_t** out)
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
        rocke_value_t* scalar = rocke_tile_window_load_scalar(b, w, local_indices, num_indices);
        out[0] = rocke_b_cast_to_f32(b, scalar);
        return;
    }
    {
        rocke_value_t* v = rocke_tile_window_load_vec(b, w, local_indices, num_indices, n);
        for(i = 0; i < n; ++i)
            out[i] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, v, i));
    }
}

void rocke_tile_window_store_vec_from_f32(rocke_ir_builder_t* b,
                                          const rocke_tile_window_t* w,
                                          rocke_value_t* const* local_indices,
                                          int num_indices,
                                          rocke_value_t* const* values,
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
    const rocke_type_t* dt;
    rocke_value_t** casts;
    rocke_value_t* packed;
    int i;
    if(b == NULL || w == NULL || w->view == NULL)
        return;
    dt = w->view->desc.dtype;
    if(!(rocke_tv_name_is(dt, "f16") || rocke_tv_name_is(dt, "bf16")))
    {
        /* Python: NotImplementedError "store_vec_from_f32 not wired for ...". */
        rocke_i_set_err(b,
                        ROCKE_ERR_NOTIMPL,
                        "store_vec_from_f32 not wired for %s; "
                        "cast manually and use store_vec",
                        dt && dt->name ? dt->name : "<null>");
        return;
    }
    if(num_values == 1)
    {
        rocke_value_t* scalar = rocke_b_cast_f32_to(b, values[0], dt);
        rocke_tile_window_store_scalar(b, w, local_indices, num_indices, scalar, 0);
        return;
    }
    if(num_values <= 0)
        return;
    casts = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                               (size_t)num_values * sizeof(rocke_value_t*));
    if(casts == NULL)
        return;
    for(i = 0; i < num_values; ++i)
        casts[i] = rocke_b_cast_f32_to(b, values[i], dt);
    packed = rocke_b_vec_pack(b, casts, num_values, dt);
    rocke_tile_window_store_vec(b, w, local_indices, num_indices, packed, num_values);
}

void rocke_tensor_view_load_vec_as_f32(rocke_ir_builder_t* b,
                                       const rocke_tensor_view_t* v,
                                       rocke_value_t* const* indices,
                                       int num_indices,
                                       int n,
                                       rocke_value_t** out)
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
    const rocke_type_t* dt;
    int i;
    if(b == NULL || v == NULL || out == NULL)
        return;
    dt = v->desc.dtype;

    if(n == 1)
    {
        rocke_value_t* scalar = rocke_tensor_view_load_scalar(b, v, indices, num_indices);
        if(rocke_tv_name_is(dt, "f32"))
            out[0] = scalar;
        else
            out[0] = rocke_b_cast_to_f32(b, scalar);
        return;
    }

    {
        rocke_value_t* vec = rocke_tensor_view_load_vec(b, v, indices, num_indices, n);
        if(rocke_tv_name_is(dt, "f32"))
        {
            for(i = 0; i < n; ++i)
                out[i] = rocke_b_vec_extract(b, vec, i);
        }
        else
        {
            for(i = 0; i < n; ++i)
                out[i] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, vec, i));
        }
    }
}

/* ==================================================== module-level factories */

rocke_status_t rocke_make_global_view(rocke_tensor_view_t* out,
                                      rocke_value_t* base,
                                      const int* shape,
                                      int rank,
                                      const rocke_type_t* dtype,
                                      const rocke_stride_t* strides)
{
    /* Python make_global_view:
     *   desc = packed(shape, dtype) if strides is None else with_strides(...)
     *   return TensorView(base, desc, addr_space="global")
     */
    rocke_status_t st;
    if(out == NULL)
        return ROCKE_ERR_VALUE;
    if(strides == NULL)
        st = rocke_tensor_descriptor_packed(&out->desc, shape, rank, dtype);
    else
        st = rocke_tensor_descriptor_with_strides(&out->desc, shape, strides, rank, dtype);
    if(st != ROCKE_OK)
        return st;
    out->base = base;
    out->addr_space = ROCKE_ADDR_GLOBAL;
    return ROCKE_OK;
}

rocke_status_t rocke_make_tile_window(rocke_tile_window_t* out,
                                      const rocke_tensor_view_t* view,
                                      const int* lengths,
                                      rocke_value_t* const* origin,
                                      int rank)
{
    /* Python make_tile_window -> view.tile(lengths, origin). */
    return rocke_tensor_view_tile(out, view, lengths, origin, rank);
}

rocke_status_t rocke_make_naive_tensor_view_packed(rocke_tensor_view_t* out,
                                                   rocke_value_t* base,
                                                   const int* shape,
                                                   int rank,
                                                   const rocke_type_t* dtype)
{
    /* Python make_naive_tensor_view_packed:
     *   return make_global_view(base, shape, dtype)   # packed row-major */
    return rocke_make_global_view(out, base, shape, rank, dtype, NULL);
}

rocke_status_t rocke_make_lds_view(rocke_ir_builder_t* b,
                                   rocke_tensor_view_t* out,
                                   const rocke_type_t* dtype,
                                   const int* shape,
                                   int rank,
                                   const char* name_hint,
                                   const rocke_stride_t* strides /* NULL => packed */)
{
    /* Python make_lds_view:
     *   smem = b.smem_alloc(dtype, list(shape), name_hint=name_hint)
     *   desc = packed(shape, dtype) if strides is None else with_strides(...)
     *   return TensorView(base=smem, desc=desc, addr_space="lds")
     */
    rocke_status_t st;
    rocke_value_t* smem;
    if(b == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    smem = rocke_b_smem_alloc(b, dtype, shape, rank, name_hint);
    if(strides == NULL)
        st = rocke_tensor_descriptor_packed(&out->desc, shape, rank, dtype);
    else
        st = rocke_tensor_descriptor_with_strides(&out->desc, shape, strides, rank, dtype);
    if(st != ROCKE_OK)
        return st;
    out->base = smem;
    out->addr_space = ROCKE_ADDR_LDS;
    return ROCKE_OK;
}
