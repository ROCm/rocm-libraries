// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// C entry points of fmha_fwd_decode_pagedkv for serving frameworks (see
// fmha_fwd_decode_pagedkv.py for the PyTorch wrapper). The launch configuration depends only on
// the batch, the kv heads, the window and an upper bound of the context lengths, so one call
// sequence can be captured in a HIP graph and replayed while the context lengths (read on the
// device) change below that bound.

#include "fmha_fwd_decode_pagedkv.hpp"

#include <exception>

extern "C" {

// The size functions take grid_rows: the kv heads times the 16-head blocks of query heads per kv
// head, num_kv_heads * ceil(num_query_heads / num_kv_heads / 16).

// Bytes of scratch workspace for these sizes, 0 for invalid sizes.
size_t ck_fmha_fwd_decode_pagedkv_workspace_bytes(
    int batch, int grid_rows, int max_seqlen_k, int window, int page_size)
{
    if(max_seqlen_k < 0 || (page_size != 16 && page_size != 64 && page_size != 128))
        return 0;
    const auto c = ck_tile::fmha_fwd_decode_pagedkv_config(
        batch, grid_rows, max_seqlen_k, window, page_size, ck_tile::get_max_cluster_size());
    return ck_tile::fmha_fwd_decode_pagedkv_workspace_bytes(
        batch, grid_rows, c.num_splits, c.num_groups);
}

// An upper bound of ck_fmha_fwd_decode_pagedkv_workspace_bytes over all context lengths, windows
// and page sizes; it grows with batch * grid_rows. 0 for invalid sizes.
size_t ck_fmha_fwd_decode_pagedkv_max_workspace_bytes(int batch, int grid_rows)
{
    return ck_tile::fmha_fwd_decode_pagedkv_max_workspace_bytes(
        batch, grid_rows, ck_tile::get_max_cluster_size());
}

// int32 counters for this batch; zero them once, calls of any shape share them. Calls that share a
// workspace and counters must not run concurrently, and the counters must be zeroed again after a
// failed call: a call that finds them non-zero can leave output rows unwritten without an error.
size_t ck_fmha_fwd_decode_pagedkv_num_counters(int batch, int grid_rows)
{
    return ck_tile::fmha_fwd_decode_pagedkv_num_counters(batch, grid_rows);
}

// q, o: [batch][nhead_q][128] bf16, or fp16 with qo_fp16 != 0, with q_strides / o_strides
// {batch stride, head stride} in elements (null: contiguous); k, v: fp8 e4m3 caches in the
// vectorized layout with page_size (16, 64 or 128) tokens per page; block_table: int32 rows of
// block_table_stride entries; seqlen_k: int32 [batch], each <= max_seqlen_k; k/v descale: one
// float each. Returns a hipError_t.
int ck_fmha_fwd_decode_pagedkv(const void* q,
                               const void* k,
                               const void* v,
                               const int32_t* block_table,
                               int block_table_stride,
                               const int32_t* seqlen_k,
                               const float* k_descale,
                               const float* v_descale,
                               void* o,
                               void* workspace,
                               int32_t* counters,
                               int batch,
                               int nhead_q,
                               int nhead_k,
                               int max_seqlen_k,
                               int window,
                               int page_size,
                               int qo_fp16,
                               float scale_s,
                               const int* q_strides,
                               const int* o_strides,
                               hipStream_t stream)
{
    if(batch <= 0 || nhead_k <= 0 || nhead_q <= 0 || max_seqlen_k < 0)
        return hipErrorInvalidValue;
    const auto c = ck_tile::fmha_fwd_decode_pagedkv_config(
        batch,
        nhead_k * ck_tile::fmha_fwd_decode_pagedkv_q_blocks(nhead_q, nhead_k),
        max_seqlen_k,
        window,
        page_size,
        ck_tile::get_max_cluster_size());

    fmha_fwd_decode_pagedkv_args args{};
    args.q_ptr              = q;
    args.k_ptr              = k;
    args.v_ptr              = v;
    args.o_ptr              = o;
    args.block_table_ptr    = block_table;
    args.block_table_stride = block_table_stride;
    args.seqlen_k_ptr       = seqlen_k;
    args.k_descale_ptr      = k_descale;
    args.v_descale_ptr      = v_descale;
    args.workspace_ptr      = workspace;
    args.counters_ptr       = counters;
    args.batch              = batch;
    args.nhead_q            = nhead_q;
    args.nhead_k            = nhead_k;
    args.nhead_stride_q     = q_strides ? q_strides[1] : 128;
    args.batch_stride_q     = q_strides ? q_strides[0] : nhead_q * 128;
    args.nhead_stride_o     = o_strides ? o_strides[1] : 128;
    args.batch_stride_o     = o_strides ? o_strides[0] : nhead_q * 128;
    args.window_size        = window;
    args.num_groups         = c.num_groups;
    args.pages_per_split    = c.pages_per_split;
    args.scale_s            = scale_s;

    fmha_fwd_decode_pagedkv_traits traits{};
    traits.hdim_q          = 128;
    traits.data_type       = qo_fp16 != 0 ? "fp16" : "bf16";
    traits.page_block_size = page_size;
    traits.wave_split      = c.wave_split;
    traits.num_splits      = c.num_splits;
    try
    {
        // invalid arguments, or no instance for this page size or device
        if(fmha_fwd_decode_pagedkv(traits, args, ck_tile::stream_config{stream}) < 0)
            return hipErrorInvalidValue;
    }
    catch(const fmha_fwd_decode_pagedkv_launch_error& e)
    {
        return e.code;
    }
    catch(const std::exception&)
    {
        return hipErrorUnknown;
    }
    return hipSuccess;
}
}
