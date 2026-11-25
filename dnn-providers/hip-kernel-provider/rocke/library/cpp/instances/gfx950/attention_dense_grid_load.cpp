// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "attention_dense_grid.hpp"
#include "rocke/helper_rocke.helpers.attention.h"
#include <bit>
#include <cassert>

namespace rocke_dense_gfx950::grid
{
namespace
{
// Port of the addressing helpers in kernels/common/_attention_dense_paged.py.
std::pair<V, V> paged_seq_setup(Builder* b, V bt, V kv_lens, V bt_stride, int block_size)
{
    V seq_base = rocke_b_mul(b, bt, bt_stride);
    V kv_len = rocke_b_global_load_i32(b, kv_lens, bt, 4);
    V rounded = DENSE_OP(add, kv_len, rocke_b_const_i32(b, block_size - 1));
    return {seq_base, DENSE_OP(div, rounded, rocke_b_const_i32(b, block_size))};
}

V paged_wave_row_base(
    Builder* b, V block_tables, V seq_base, V n_pages, V wave_key0, int block_size)
{
    const int shift = std::bit_width(static_cast<unsigned>(block_size)) - 1;
    V page = DENSE_OP(lshr, wave_key0, rocke_b_const_i32(b, shift));
    V block = DENSE_OP(masked_global_load,
                       block_tables,
                       DENSE_OP(add, seq_base, page),
                       DENSE_OP(cmp_lt, page, n_pages),
                       rocke_b_const_i32(b, 0),
                       rocke_i32(),
                       4);
    return DENSE_OP(mul, block, rocke_b_const_i32(b, block_size));
}

V paged_kv_row(Builder* b, V row_base, V key, int block_size)
{
    return DENSE_OP(add, row_base, DENSE_OP(land, key, rocke_b_const_i32(b, block_size - 1)));
}
} // namespace

void Grid::prepare_loads()
{
    auto c = [&](int64_t i) { return rocke_b_const_i32(b, i); };
    // The DMA writes a contiguous group of rows. For packed D<128 only K has
    // a pad, and it goes between groups, never between rows within a group.
    K_GROUP = ROWS_PER_INSTR;
    const int k_rows = BN / K_GROUP;
    const int ldrow = K_GROUP == 1 ? D + PAD : K_GROUP * D + spec.base.lds_k_group_pad;
    const int vrow = K_GROUP == 1 ? D + spec.lds_v_row_pad : D;
    const int k_shape[] = {NBUF, k_rows, ldrow};
    const int v_shape[] = {NBUF, BN, vrow};
    K_lds = rocke_b_smem_alloc(b, dtype, k_shape, 3, "Klds");
    V_lds = rocke_b_smem_alloc(b, dtype, v_shape, 3, "Vlds");
    if(K_GROUP > 1)
    {
        k_lane_grp = DENSE_OP(div, lane_m, c(K_GROUP));
        k_sub_col = DENSE_OP(mul, DENSE_OP(mod, lane_m, c(K_GROUP)), c(D));
    }
    else
        k_lane_grp = k_sub_col = nullptr;

    // Keep Q unscaled through the QK MFMA. Only the ragged path uses a
    // bounds-checked resource; the normal aligned path loads Q directly.
    V q_rsrc = RAGGED ? DENSE_OP(buffer_rsrc, q, c(int64_t(B) * Sq * Hq * D * 2)) : nullptr;
    V q_tok = rocke_b_add(b, q_tok0, lane_m);
    for(int ks = 0; ks < K_STEPS; ++ks)
    {
        V col = DENSE_OP(add, c(ks * 16), d_base);
        V addr = DENSE_OP(add, DENSE_OP(add, q_base, DENSE_OP(mul, q_tok, stride_q())), col);
        q_packs[ks]
            = RAGGED ? DENSE_OP(buffer_load_vN, q_rsrc, DENSE_OP(mul, addr, c(2)), c(0), dtype, 8)
                     : rocke_b_global_load_vN(b, q, addr, dtype, 8, 16);
    }

    K_BYTES_PER_BUF = k_rows * ldrow * 2;
    K_GROUP_BYTES = ldrow * 2;
    V_BYTES_PER_BUF = BN * vrow * 2;
    V_GROUP_BYTES = ROWS_PER_INSTR * vrow * 2;
    ROWS_PER_WAVE = BN / WAVES;
    assert(BN % K_GROUP == 0 && ROWS_PER_WAVE % ROWS_PER_INSTR == 0);
    LOOP_VMCNT = spec.partial_vmcnt_prefetch ? (BN * D) / (WAVES * 64 * 8) : 0;
    zero_soff = c(0);
    K_lds_addr = rocke_b_smem_addr_of(b, K_lds);
    V_lds_addr = rocke_b_smem_addr_of(b, V_lds);

    auto row_bytes = [&]() {
        return runtime_shape ? DENSE_OP(mul, stride_k(), c(2)) : c(int64_t(Hkv) * D * 2);
    };
    // num_records counts bytes. Emit its expression afresh for each resource;
    // sharing the equivalent K/V bounds changes SSA numbering in the golden IR.
    auto make_resource = [&](V ptr) {
        V bound;
        if(spec.base.paged)
            bound = c(int64_t(spec.base.num_kv_blocks) * spec.base.block_size * Hkv * D * 2);
        else if(varlen)
            bound = DENSE_OP(mul, kv_seq1, row_bytes()); // End of this packed sequence.
        else
            bound = DENSE_OP(mul, DENSE_OP(mul, batch_p, seqlen_kv_p), row_bytes());
        return rocke_b_buffer_rsrc(b, ptr, bound);
    };
    if(KV_FP8)
        k_rsrc = v_rsrc = nullptr;
    else
    {
        k_rsrc = make_resource(k);
        v_rsrc = make_resource(v);
    }
    // Dead in the Python body, but its IR nodes survive lowering.
    DENSE_OP(zext, DENSE_OP(to_sgpr_u32, DENSE_OP(mul, wave, c(64 * 16))), rocke_i64());
    if(spec.base.paged)
        std::tie(pg_seq_base, pg_n_pages)
            = paged_seq_setup(b, bt, kv_lens, bt_stride, spec.base.block_size);
}

void Grid::load_tile(V buf_val, V tile_idx)
{
    auto c = [&](int64_t i) { return rocke_b_const_i32(b, i); };
    V tk0 = DENSE_OP(mul, tile_idx, c(BN));
    if(KV_FP8)
    {
        const int threads = WAVES * 64;
        assert((BN * D) % (threads * 8) == 0);
        auto sync_load = [&](V ptr, V lds, V scale_p, bool packed_k) {
            for(int it = 0; it < (BN * D) / (threads * 8); ++it)
            {
                V e0 = DENSE_OP(mul, DENSE_OP(add, c(it * threads), tid), c(8));
                V key = DENSE_OP(div, e0, c(D));
                V d0 = DENSE_OP(mod, e0, c(D));
                // The over-prefetched final tile uses unbounded global loads.
                // Redirect those keys to a valid row before computing the address.
                V kv_row = rocke_b_add(b, tk0, key);
                kv_row = DENSE_OP(select, DENSE_OP(cmp_lt, kv_row, seqlen_kv_p), kv_row, c(0));
                V addr
                    = DENSE_OP(add, DENSE_OP(add, k_base, DENSE_OP(mul, kv_row, stride_k())), d0);
                V raw = rocke_b_global_load_vN(b, ptr, addr, kv_dtype, 8, 0);
                V deq = rocke_dequant_fp8x8_to_dtype(b, raw, scale_p, dtype);
                V row = key, col = d0;
                if(packed_k)
                {
                    row = DENSE_OP(div, key, c(K_GROUP));
                    col = DENSE_OP(add, DENSE_OP(mul, DENSE_OP(mod, key, c(K_GROUP)), c(D)), d0);
                }
                V idx[] = {buf_val, row, col};
                rocke_b_smem_store_vN(b, lds, idx, 3, deq, 8);
            }
        };
        sync_load(k, K_lds, k_scale, K_GROUP > 1);
        sync_load(v, V_lds, v_scale, false);
        rocke_b_s_waitcnt(b, -1, 0, -1);
        return;
    }

    const int block_size = spec.base.block_size;
    auto async_load = [&](V rsrc, V lds_base, int bytes_per_buf, int group_bytes) {
        V buf_off = DENSE_OP(
            mul, DENSE_OP(zext, buf_val, rocke_i64()), rocke_b_const_i64(b, bytes_per_buf));
        V phys_base = nullptr;
        if(spec.base.paged)
        {
            // Admission places each wave's entire stripe (including packed
            // sub-rows) within one page; only one table lookup per K or V DMA stripe.
            V wave_key0 = DENSE_OP(add, tk0, DENSE_OP(mul, wave, c(ROWS_PER_WAVE)));
            phys_base = paged_wave_row_base(
                b, block_tables, pg_seq_base, pg_n_pages, wave_key0, block_size);
        }
        const bool packed = ROWS_PER_INSTR != 1;
        const int groups_per_wave = ROWS_PER_WAVE / ROWS_PER_INSTR;
        V sub_row = nullptr, packed_col = nullptr;
        if(packed)
        {
            sub_row = DENSE_OP(div, lane, c(D / 2));
            packed_col = DENSE_OP(mul, DENSE_OP(mod, lane, c(D / 2)), c(2));
        }
        for(int it = 0; it < groups_per_wave; ++it)
        {
            V row0 = DENSE_OP(add, DENSE_OP(mul, wave, c(ROWS_PER_WAVE)), c(it * ROWS_PER_INSTR));
            V group = packed ? DENSE_OP(add, DENSE_OP(mul, wave, c(groups_per_wave)), c(it)) : row0;
            V lds_off = DENSE_OP(
                add, buf_off, DENSE_OP(zext, DENSE_OP(mul, group, c(group_bytes)), rocke_i64()));
            V dst = rocke_b_smem_ptr_add(b, lds_base, lds_off);
            V key = rocke_b_add(b, tk0, row0);
            if(packed)
                key = rocke_b_add(b, key, sub_row);
            V col = packed ? packed_col : DENSE_OP(mul, lane, c(2));
            if(spec.base.paged)
                key = paged_kv_row(b, phys_base, key, block_size);
            V voff = DENSE_OP(add, DENSE_OP(add, k_base, DENSE_OP(mul, key, stride_k())), col);
            DENSE_OP(
                async_buffer_load_lds_addr, rsrc, dst, DENSE_OP(mul, voff, c(2)), zero_soff, 1, 0);
        }
    };
    async_load(k_rsrc, K_lds_addr, K_BYTES_PER_BUF, K_GROUP_BYTES);
    async_load(v_rsrc, V_lds_addr, V_BYTES_PER_BUF, V_GROUP_BYTES);
}
} // namespace rocke_dense_gfx950::grid
