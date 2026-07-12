// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/loads.py: CoalescedTileLoader, AsyncTileLoader
 * (and the AsyncTileLoaderSlot that AsyncTileLoader.bind returns).
 *
 * See the header for the original Python and the contract. The builder-emitting
 * methods (load / bind / issue) reproduce the const_i32 / arith / load / store
 * call sequence byte-faithfully so the downstream IR is identical to the Python.
 */
#include "rocke/helper_rocke.helpers.loads.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ============================================================================
 * CoalescedTileLoader
 * ========================================================================== */

rocke_status_t rocke_coalesced_tile_loader_choose_vec(
    int tile_rows, int tile_cols, int block_size, int max_vec, int* out_vec)
{
    int v;

    /* Python:
     *   v = max_vec
     *   while v >= 1:
     *       if (tile_cols % v == 0
     *           and (tile_rows*tile_cols)//v >= block_size
     *           and ((tile_rows*tile_cols)//v) % block_size == 0):
     *           return v
     *       v //= 2
     *   raise ValueError(...)
     */
    v = max_vec;
    while(v >= 1)
    {
        if(tile_cols % v == 0 && (tile_rows * tile_cols) / v >= block_size
           && ((tile_rows * tile_cols) / v) % block_size == 0)
        {
            if(out_vec != NULL)
            {
                *out_vec = v;
            }
            return ROCKE_OK;
        }
        v /= 2;
    }
    return ROCKE_ERR_VALUE;
}

rocke_status_t rocke_coalesced_tile_loader_from_tile(int tile_rows,
                                                     int tile_cols,
                                                     int block_size,
                                                     int max_vec,
                                                     bool use_buffer_rsrc,
                                                     rocke_coalesced_tile_loader_t* out)
{
    int vec;
    rocke_status_t st;

    /* Python:
     *   vec = cls.choose_vec(...)
     *   return cls(tile_rows=..., tile_cols=..., block_size=...,
     *              load_vec=vec, use_buffer_rsrc=use_buffer_rsrc)
     */
    st = rocke_coalesced_tile_loader_choose_vec(tile_rows, tile_cols, block_size, max_vec, &vec);
    if(st != ROCKE_OK)
    {
        return st;
    }
    if(out != NULL)
    {
        out->tile_rows = tile_rows;
        out->tile_cols = tile_cols;
        out->block_size = block_size;
        out->load_vec = vec;
        out->use_buffer_rsrc = use_buffer_rsrc;
        /* Python (1 << 31) - 1 == 2147483647 (arbitrary-precision ints);
         * spell it as the literal to avoid the C int shift-into-sign overflow. */
        out->oob_sentinel = 2147483647; /* dataclass default */
        out->has_inner_dim = false; /* inner_dim default None */
        out->inner_dim = 0;
    }
    return ROCKE_OK;
}

rocke_status_t
    rocke_coalesced_tile_loader_vecs_per_thread(const rocke_coalesced_tile_loader_t* self, int* out)
{
    int total_vecs;

    if(self == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* Python:
     *   total_vecs = (tile_rows * tile_cols) // load_vec
     *   if total_vecs % block_size: raise ValueError(...)
     *   return total_vecs // block_size
     */
    total_vecs = (self->tile_rows * self->tile_cols) / self->load_vec;
    if(total_vecs % self->block_size)
    {
        return ROCKE_ERR_VALUE;
    }
    if(out != NULL)
    {
        *out = total_vecs / self->block_size;
    }
    return ROCKE_OK;
}

int rocke_coalesced_tile_loader_cols_per_vec(const rocke_coalesced_tile_loader_t* self)
{
    /* Python: return tile_cols // load_vec */
    if(self == NULL)
    {
        return 0;
    }
    return self->tile_cols / self->load_vec;
}

void rocke_coalesced_tile_loader_load(rocke_ir_builder_t* b,
                                      const rocke_coalesced_tile_loader_t* self,
                                      rocke_value_t* tid,
                                      rocke_value_t* smem_dst,
                                      rocke_loads_descriptor_fn descriptor,
                                      void* descriptor_user,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* ptr)
{
    rocke_value_t* c_threads;
    rocke_value_t* c_load_vec;
    rocke_value_t* c_cols_per_vec;
    rocke_value_t* c_half_bytes;
    rocke_value_t* c0;
    rocke_value_t* c_oob;
    int vecs_per_thread;
    int e;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return; /* already in error: no-op */
    }
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "CoalescedTileLoader.load: NULL loader");
        return;
    }

    /* Python:
     *   if self.use_buffer_rsrc and rsrc is None:
     *       raise ValueError("... use_buffer_rsrc=True requires rsrc")
     *   if not self.use_buffer_rsrc and ptr is None:
     *       raise ValueError("... use_buffer_rsrc=False requires ptr")
     */
    if(self->use_buffer_rsrc && rsrc == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: use_buffer_rsrc=True requires rsrc");
        return;
    }
    if(!self->use_buffer_rsrc && ptr == NULL)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "CoalescedTileLoader: use_buffer_rsrc=False requires ptr");
        return;
    }

    /* Python:
     *   c_threads      = b.const_i32(self.block_size)
     *   c_load_vec     = b.const_i32(self.load_vec)
     *   c_cols_per_vec = b.const_i32(self.cols_per_vec)
     *   c_half_bytes   = b.const_i32(2)
     *   c0             = b.const_i32(0)
     *   c_oob          = b.const_i32(self.oob_sentinel)
     */
    c_threads = rocke_b_const_i32(b, self->block_size);
    c_load_vec = rocke_b_const_i32(b, self->load_vec);
    c_cols_per_vec = rocke_b_const_i32(b, rocke_coalesced_tile_loader_cols_per_vec(self));
    c_half_bytes = rocke_b_const_i32(b, 2);
    c0 = rocke_b_const_i32(b, 0);
    c_oob = rocke_b_const_i32(b, self->oob_sentinel);

    /* self.vecs_per_thread (Python property; raises ValueError on bad divide). */
    if(rocke_coalesced_tile_loader_vecs_per_thread(self, &vecs_per_thread) != ROCKE_OK)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "CoalescedTileLoader: tile %dx%d / %d not divisible by "
                        "block_size %d",
                        self->tile_rows,
                        self->tile_cols,
                        self->load_vec,
                        self->block_size);
        return;
    }

    for(e = 0; e < vecs_per_thread; ++e)
    {
        rocke_value_t* vec_idx;
        rocke_value_t* row;
        rocke_value_t* col_v;
        rocke_value_t* col;
        rocke_value_t* off_elems;
        rocke_value_t* valid;

        /* Python:
         *   vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
         *   row     = b.div(vec_idx, c_cols_per_vec)
         *   col_v   = b.mod(vec_idx, c_cols_per_vec)
         *   col     = b.mul(col_v, c_load_vec) if self.load_vec > 1 else col_v
         */
        vec_idx = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads), tid);
        row = rocke_b_div(b, vec_idx, c_cols_per_vec);
        col_v = rocke_b_mod(b, vec_idx, c_cols_per_vec);
        col = (self->load_vec > 1) ? rocke_b_mul(b, col_v, c_load_vec) : col_v;

        /* Python: off_elems, valid = descriptor(b, row, col) */
        valid = NULL; /* default => "None" if callback leaves it NULL */
        off_elems = descriptor(b, row, col, &valid, descriptor_user);

        if(self->use_buffer_rsrc)
        {
            rocke_value_t* off_bytes;
            rocke_value_t* safe;
            rocke_value_t* indices[2];

            /* Python:
             *   off_bytes = b.mul(off_elems, c_half_bytes)
             *   safe = b.select(valid, off_bytes, c_oob) if valid is not None
             *          else off_bytes
             */
            off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
            if(valid != NULL)
            {
                safe = rocke_b_select(b, valid, off_bytes, c_oob);
            }
            else
            {
                safe = off_bytes;
            }

            indices[0] = row;
            indices[1] = col;
            if(self->load_vec == 1)
            {
                /* Python:
                 *   v = b.buffer_load_f16(rsrc, safe, c0)
                 *   b.smem_store_f16(smem_dst, [row, col], v)
                 */
                rocke_value_t* v = rocke_b_buffer_load_f16(b, rsrc, safe, c0);
                rocke_b_smem_store_f16(b, smem_dst, indices, 2, v);
            }
            else
            {
                /* Python:
                 *   dwords = self.load_vec // 2
                 *   v = b.buffer_load_vN_f16(rsrc, safe, c0, dwords)
                 *   b.smem_store_vN_f16(smem_dst, [row, col], v, self.load_vec)
                 */
                int dwords = self->load_vec / 2;
                rocke_value_t* v = rocke_b_buffer_load_vN_f16(b, rsrc, safe, c0, dwords);
                rocke_b_smem_store_vN_f16(b, smem_dst, indices, 2, v, self->load_vec);
            }
        }
        else
        {
            rocke_value_t* indices[2];
            indices[0] = row;
            indices[1] = col;
            if(self->load_vec == 1)
            {
                /* Python:
                 *   v = b.global_load_f16(ptr, off_elems)
                 *   b.smem_store_f16(smem_dst, [row, col], v)
                 */
                rocke_value_t* v = rocke_b_global_load_f16(b, ptr, off_elems, 0);
                rocke_b_smem_store_f16(b, smem_dst, indices, 2, v);
            }
            else
            {
                /* Python:
                 *   v = b.global_load_vN_f16(ptr, off_elems, self.load_vec)
                 *   b.smem_store_vN_f16(smem_dst, [row, col], v, self.load_vec)
                 */
                rocke_value_t* v = rocke_b_global_load_vN_f16(b, ptr, off_elems, self->load_vec, 0);
                rocke_b_smem_store_vN_f16(b, smem_dst, indices, 2, v, self->load_vec);
            }
        }
    }
}

/* ============================================================================
 * AsyncTileLoader / AsyncTileLoaderSlot
 * ========================================================================== */

rocke_status_t rocke_async_tile_loader_choose_dwords(
    int tile_rows, int tile_cols, int block_size, int max_dwords, int* out)
{
    /* Python:
     *   if max_dwords > 4: max_dwords = 4
     *   for d in (4, 3, 1):
     *       if d > max_dwords: continue
     *       halves = d * 2
     *       if tile_cols % halves != 0: continue
     *       chunks = (tile_rows*tile_cols) // halves
     *       if chunks < block_size: continue
     *       return d
     *   raise ValueError(...)
     */
    static const int kCandidates[3] = {4, 3, 1};
    int i;

    if(max_dwords > 4)
    {
        max_dwords = 4;
    }
    for(i = 0; i < 3; ++i)
    {
        int d = kCandidates[i];
        int halves;
        int chunks;
        if(d > max_dwords)
        {
            continue;
        }
        halves = d * 2;
        if(tile_cols % halves != 0)
        {
            continue;
        }
        chunks = (tile_rows * tile_cols) / halves;
        if(chunks < block_size)
        {
            continue;
        }
        if(out != NULL)
        {
            *out = d;
        }
        return ROCKE_OK;
    }
    return ROCKE_ERR_VALUE;
}

rocke_status_t rocke_async_tile_loader_from_tile(int tile_rows,
                                                 int tile_cols,
                                                 int block_size,
                                                 int wave_size,
                                                 int max_dwords,
                                                 rocke_async_tile_loader_t* out)
{
    int d;
    int halves;
    int chunks;
    int passes;
    rocke_status_t st;

    /* Python:
     *   d = cls.choose_dwords(...)
     *   halves = d * 2
     *   chunks = (tile_rows * tile_cols) // halves
     *   passes = (chunks + block_size - 1) // block_size
     *   return cls(... dwords=d, chunks_total=chunks,
     *              chunks_per_pass=block_size, passes=passes)
     */
    st = rocke_async_tile_loader_choose_dwords(tile_rows, tile_cols, block_size, max_dwords, &d);
    if(st != ROCKE_OK)
    {
        return st;
    }
    halves = d * 2;
    chunks = (tile_rows * tile_cols) / halves;
    passes = (chunks + block_size - 1) / block_size;
    if(out != NULL)
    {
        out->tile_rows = tile_rows;
        out->tile_cols = tile_cols;
        out->block_size = block_size;
        out->wave_size = wave_size;
        out->dwords = d;
        out->chunks_total = chunks;
        out->chunks_per_pass = block_size;
        out->passes = passes;
    }
    return ROCKE_OK;
}

int rocke_async_tile_loader_halves_per_chunk(const rocke_async_tile_loader_t* self)
{
    /* Python: return dwords * 2 */
    return (self == NULL) ? 0 : self->dwords * 2;
}

int rocke_async_tile_loader_bytes_per_chunk(const rocke_async_tile_loader_t* self)
{
    /* Python: return dwords * 4 */
    return (self == NULL) ? 0 : self->dwords * 4;
}

int rocke_async_tile_loader_cols_per_chunk(const rocke_async_tile_loader_t* self)
{
    /* Python: return self.halves_per_chunk */
    return rocke_async_tile_loader_halves_per_chunk(self);
}

int rocke_async_tile_loader_wave_bytes(const rocke_async_tile_loader_t* self)
{
    /* Python: return wave_size * self.bytes_per_chunk */
    if(self == NULL)
    {
        return 0;
    }
    return self->wave_size * rocke_async_tile_loader_bytes_per_chunk(self);
}

int rocke_async_tile_loader_pass_bytes(const rocke_async_tile_loader_t* self)
{
    /* Python: return block_size * self.bytes_per_chunk */
    if(self == NULL)
    {
        return 0;
    }
    return self->block_size * rocke_async_tile_loader_bytes_per_chunk(self);
}

rocke_status_t rocke_async_tile_loader_bind(rocke_ir_builder_t* b,
                                            const rocke_async_tile_loader_t* self,
                                            rocke_value_t* smem_dst,
                                            rocke_value_t* wave_id,
                                            rocke_async_tile_loader_slot_t* out_slot)
{
    rocke_value_t* lds_base;
    rocke_value_t* wave_byte_off_i32;
    rocke_value_t* wave_byte_off_i64;
    rocke_value_t* per_wave_lds;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return b->status; /* already in error: no-op */
    }
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "AsyncTileLoader.bind: NULL loader");
        return ROCKE_ERR_VALUE;
    }

    /* Python:
     *   lds_base = b.smem_addr_of(smem_dst)                       # i64
     *   wave_byte_off_i32 = b.mul(wave_id, b.const_i32(self.wave_bytes))
     *   wave_byte_off_i32 = b.to_sgpr_u32(wave_byte_off_i32)
     *   wave_byte_off_i64 = b.zext(wave_byte_off_i32, I64)
     *   per_wave_lds = b.smem_ptr_add(lds_base, wave_byte_off_i64)
     *   return AsyncTileLoaderSlot(loader=self, smem_dst=smem_dst,
     *                              per_wave_lds_base=per_wave_lds)
     */
    lds_base = rocke_b_smem_addr_of(b, smem_dst);
    wave_byte_off_i32
        = rocke_b_mul(b, wave_id, rocke_b_const_i32(b, rocke_async_tile_loader_wave_bytes(self)));
    wave_byte_off_i32 = rocke_b_to_sgpr_u32(b, wave_byte_off_i32);
    wave_byte_off_i64 = rocke_b_zext(b, wave_byte_off_i32, rocke_i64());
    per_wave_lds = rocke_b_smem_ptr_add(b, lds_base, wave_byte_off_i64);

    if(out_slot != NULL)
    {
        out_slot->loader = *self;
        out_slot->smem_dst = smem_dst;
        out_slot->per_wave_lds_base = per_wave_lds;
    }
    return (b != NULL) ? b->status : ROCKE_OK;
}

void rocke_async_tile_loader_slot_issue(rocke_ir_builder_t* b,
                                        const rocke_async_tile_loader_slot_t* self,
                                        rocke_value_t* tid,
                                        rocke_value_t* rsrc,
                                        rocke_loads_descriptor_fn descriptor,
                                        void* descriptor_user,
                                        int oob_sentinel,
                                        int coherency)
{
    const rocke_async_tile_loader_t* L;
    rocke_value_t* c_half_bytes;
    rocke_value_t* c_oob;
    rocke_value_t* c0;
    rocke_value_t* c_cols_per_chunk;
    int p;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return; /* already in error: no-op */
    }
    if(self == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "AsyncTileLoaderSlot.issue: NULL slot");
        return;
    }

    /* Python:
     *   L = self.loader
     *   c_half_bytes     = b.const_i32(2)
     *   c_oob            = b.const_i32(oob_sentinel)
     *   c0               = b.const_i32(0)
     *   c_cols_per_chunk = b.const_i32(L.cols_per_chunk)
     */
    L = &self->loader;
    c_half_bytes = rocke_b_const_i32(b, 2);
    c_oob = rocke_b_const_i32(b, oob_sentinel);
    c0 = rocke_b_const_i32(b, 0);
    c_cols_per_chunk = rocke_b_const_i32(b, rocke_async_tile_loader_cols_per_chunk(L));

    for(p = 0; p < L->passes; ++p)
    {
        int pass_byte_off;
        rocke_value_t* pass_base;
        rocke_value_t* chunk_idx;
        rocke_value_t* row;
        rocke_value_t* col_v;
        rocke_value_t* col;
        rocke_value_t* off_elems;
        rocke_value_t* valid;
        rocke_value_t* off_bytes;
        rocke_value_t* in_pass;
        rocke_value_t* valid_final;
        rocke_value_t* safe;

        /* Python:
         *   pass_byte_off = p * L.pass_bytes
         *   pass_base = (b.smem_ptr_add(self.per_wave_lds_base,
         *                               b.zext(b.const_i32(pass_byte_off), I64))
         *                if p > 0 else self.per_wave_lds_base)
         */
        pass_byte_off = p * rocke_async_tile_loader_pass_bytes(L);
        if(p > 0)
        {
            pass_base = rocke_b_smem_ptr_add(
                b,
                self->per_wave_lds_base,
                rocke_b_zext(b, rocke_b_const_i32(b, pass_byte_off), rocke_i64()));
        }
        else
        {
            pass_base = self->per_wave_lds_base;
        }

        /* Python:
         *   chunk_idx = b.add(tid, b.const_i32(p * L.block_size))
         *   row   = b.div(chunk_idx, c_cols_per_chunk)
         *   col_v = b.mod(chunk_idx, c_cols_per_chunk)
         *   col   = b.mul(col_v, b.const_i32(L.halves_per_chunk))
         */
        chunk_idx = rocke_b_add(b, tid, rocke_b_const_i32(b, p * L->block_size));
        row = rocke_b_div(b, chunk_idx, c_cols_per_chunk);
        col_v = rocke_b_mod(b, chunk_idx, c_cols_per_chunk);
        col = rocke_b_mul(
            b, col_v, rocke_b_const_i32(b, rocke_async_tile_loader_halves_per_chunk(L)));

        /* Python:
         *   off_elems, valid = descriptor(b, row, col)
         *   off_bytes = b.mul(off_elems, c_half_bytes)
         *   in_pass = b.cmp_lt(chunk_idx, b.const_i32(L.chunks_total))
         *   valid_final = b.land(valid, in_pass) if valid is not None else in_pass
         *   safe = b.select(valid_final, off_bytes, c_oob)
         *   b.async_buffer_load_lds_addr(rsrc, pass_base, safe, c0, L.dwords,
         *                                coherency=coherency)
         */
        valid = NULL;
        off_elems = descriptor(b, row, col, &valid, descriptor_user);
        off_bytes = rocke_b_mul(b, off_elems, c_half_bytes);
        in_pass = rocke_b_cmp_lt(b, chunk_idx, rocke_b_const_i32(b, L->chunks_total));
        if(valid != NULL)
        {
            valid_final = rocke_b_land(b, valid, in_pass);
        }
        else
        {
            valid_final = in_pass;
        }
        safe = rocke_b_select(b, valid_final, off_bytes, c_oob);
        rocke_b_async_buffer_load_lds_addr(b, rsrc, pass_base, safe, c0, L->dwords, coherency);
    }
}

int rocke_async_tile_loader_slot_required_lds_bytes(const rocke_async_tile_loader_slot_t* self)
{
    /* Python: return self.loader.passes * self.loader.pass_bytes */
    if(self == NULL)
    {
        return 0;
    }
    return self->loader.passes * rocke_async_tile_loader_pass_bytes(&self->loader);
}
