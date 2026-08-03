// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * fp8_mqa_logits.cpp -- C99 port of
 * rocke/instances/gfx942/fp8_mqa_logits.py.
 *
 * The builder mirrors build_fp8_mqa_logits one builder call at a time. Host
 * loops and temporary arrays do not emit IR; their traversal order matches the
 * Python list construction and nested loops.
 */
#include "rocke/instance_gfx942_fp8_mqa_logits.h"

#include <stdio.h>
#include <string.h>
#include <vector>

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/ir_internal.h"

static const int ROCKE_FP8_MQA_MIN_TILES_PER_SPLIT = 8;

static void rocke_fp8_mqa_set_reason(char* reason, size_t cap, const char* message)
{
    rocke_spec_set_reason(reason, cap, message);
}

rocke_fp8_mqa_logits_spec_t rocke_fp8_mqa_logits_spec_default(void)
{
    rocke_fp8_mqa_logits_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.num_heads = 64;
    spec.head_dim = 128;
    spec.block_kv = 128;
    spec.rows_per_block = 2;
    spec.waves_per_block = 4;
    spec.has_waves_per_eu = true;
    spec.waves_per_eu = 2;
    spec.name = "rocke_fp8_mqa_logits";
    return spec;
}

int rocke_fp8_mqa_logits_block_size(const rocke_fp8_mqa_logits_spec_t* spec)
{
    return spec != NULL ? 64 * spec->waves_per_block : 0;
}

rocke_status_t rocke_fp8_mqa_logits_kernel_name(const rocke_fp8_mqa_logits_spec_t* spec,
                                                char* out,
                                                size_t out_cap)
{
    char heads[32];
    char dim[32];
    char block[32];
    char rows[32];
    char waves[32];
    const char* parts[5];

    if(spec == NULL || spec->name == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(heads, sizeof(heads), "H%d", spec->num_heads);
    snprintf(dim, sizeof(dim), "D%d", spec->head_dim);
    snprintf(block, sizeof(block), "BKV%d", spec->block_kv);
    snprintf(rows, sizeof(rows), "R%d", spec->rows_per_block);
    snprintf(waves, sizeof(waves), "W%d", spec->waves_per_block);
    parts[0] = heads;
    parts[1] = dim;
    parts[2] = block;
    parts[3] = rows;
    parts[4] = waves;
    return rocke_kernel_name_join(spec->name, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_fp8_mqa_logits_is_valid_spec(const rocke_fp8_mqa_logits_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap)
{
    const rocke_archtarget_t* target;
    const rocke_mfma_atom_t* atom;
    int block_size;
    int n_tiles;
    char buffer[192];

    if(spec == NULL)
    {
        rocke_fp8_mqa_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx942";
    }
    if(strcmp(arch, "gfx942") != 0)
    {
        snprintf(buffer,
                 sizeof(buffer),
                 "fp8_mqa_logits currently supports gfx942 only, got '%s'",
                 arch);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buffer, sizeof(buffer), "unknown gfx target %s", arch);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }
    block_size = rocke_fp8_mqa_logits_block_size(spec);
    if(block_size > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(buffer,
                 sizeof(buffer),
                 "block_size %d > %d (hardware cap) on %s",
                 block_size,
                 rocke_archtarget_max_threads_per_block(target),
                 arch);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }

    atom = rocke_mfma_atom("fp8e4m3", 16, 16, 32);
    if(atom == NULL)
    {
        rocke_fp8_mqa_set_reason(reason, reason_cap, "missing fp8e4m3 MFMA atom");
        return false;
    }
    if(spec->num_heads <= 0 || spec->num_heads % atom->m)
    {
        snprintf(buffer, sizeof(buffer), "num_heads must be a positive multiple of %d", atom->m);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }
    if(spec->head_dim <= 0 || spec->head_dim % atom->k)
    {
        snprintf(buffer, sizeof(buffer), "head_dim must be a positive multiple of %d", atom->k);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }
    if(spec->block_kv <= 0 || spec->block_kv % atom->n)
    {
        snprintf(buffer, sizeof(buffer), "block_kv must be a positive multiple of %d", atom->n);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }
    if(spec->rows_per_block <= 0)
    {
        rocke_fp8_mqa_set_reason(reason, reason_cap, "rows_per_block must be positive");
        return false;
    }
    if(spec->waves_per_block <= 0)
    {
        rocke_fp8_mqa_set_reason(reason, reason_cap, "waves_per_block must be positive");
        return false;
    }
    n_tiles = spec->block_kv / atom->n;
    if(n_tiles % spec->waves_per_block)
    {
        snprintf(buffer,
                 sizeof(buffer),
                 "block_kv / %d (%d) must be divisible by waves_per_block (%d)",
                 atom->n,
                 n_tiles,
                 spec->waves_per_block);
        rocke_fp8_mqa_set_reason(reason, reason_cap, buffer);
        return false;
    }
    if(spec->has_waves_per_eu && spec->waves_per_eu <= 0)
    {
        rocke_fp8_mqa_set_reason(reason, reason_cap, "waves_per_eu must be positive or None");
        return false;
    }
    rocke_fp8_mqa_set_reason(reason, reason_cap, "ok");
    return true;
}

static rocke_value_t*
    rocke_fp8_mqa_ceildiv(rocke_ir_builder_t* b, rocke_value_t* value, int divisor)
{
    rocke_value_t* one_less = rocke_b_const_i32(b, divisor - 1);
    rocke_value_t* divisor_value = rocke_b_const_i32(b, divisor);
    return rocke_b_div(b, rocke_b_add(b, value, one_less), divisor_value);
}

static rocke_value_t*
    rocke_fp8_mqa_ceildiv_value(rocke_ir_builder_t* b, rocke_value_t* value, rocke_value_t* divisor)
{
    rocke_value_t* one_less = rocke_b_sub(b, divisor, rocke_b_const_i32(b, 1));
    return rocke_b_div(b, rocke_b_add(b, value, one_less), divisor);
}

static rocke_kernel_def_t* rocke_build_fp8_mqa_logits_impl(rocke_ir_builder_t* b,
                                                           const rocke_fp8_mqa_logits_spec_t* spec,
                                                           const char* arch)
{
    char reason[192];
    const rocke_mfma_atom_t* atom;
    int h;
    int d;
    int bkv;
    int rpb;
    int wpb;
    int n_tiles_per_wave;
    int m_tiles;
    int k_steps;
    rocke_param_opts_t opts;
    rocke_value_t* q;
    rocke_value_t* kv;
    rocke_value_t* kv_scales;
    rocke_value_t* weights;
    rocke_value_t* cu_starts;
    rocke_value_t* cu_ends;
    rocke_value_t* logits;
    rocke_value_t* seq_len;
    rocke_value_t* seq_len_kv;
    rocke_value_t* stride_logits_s;
    rocke_value_t* num_splits;
    rocke_value_t* tid;
    rocke_value_t* bid;
    rocke_value_t* split_id;
    rocke_value_t* wave;
    rocke_value_t* lane;
    rocke_lane_decode_t lane_decode;
    rocke_value_t* n_blocks;
    rocke_value_t* reverse_bid;
    rocke_value_t* row0;
    rocke_value_t* zero_i32;
    rocke_value_t* zero_f32;
    std::vector<rocke_value_t*> starts;
    std::vector<rocke_value_t*> ends;
    std::vector<rocke_value_t*> q_fragments;
    std::vector<rocke_value_t*> weight_fragments;
    rocke_value_t* tile_start;
    rocke_value_t* tile_end;
    rocke_value_t* window_tiles;
    rocke_value_t* split_columns;
    rocke_for_t tile_loop;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx942";
    }
    if(!rocke_fp8_mqa_logits_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid fp8_mqa_logits spec: %s", reason);
    }

    atom = rocke_mfma_atom("fp8e4m3", 16, 16, 32);
    if(rocke_validate_mfma_atom_in_catalog(b, atom, arch, "fp8_mqa_logits") != ROCKE_OK)
    {
        return NULL;
    }
    h = spec->num_heads;
    d = spec->head_dim;
    bkv = spec->block_kv;
    rpb = spec->rows_per_block;
    wpb = spec->waves_per_block;
    n_tiles_per_wave = (bkv / atom->n) / wpb;
    m_tiles = h / atom->m;
    k_steps = d / atom->k;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 64 * wpb);
    if(spec->has_waves_per_eu)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    }

    memset(&opts, 0, sizeof(opts));
    opts.readonly = true;
    opts.readonly_set = true;
    opts.align = 8;
    opts.align_set = true;
    q = rocke_b_param(b, "Q", rocke_ptr_type(b, rocke_fp8e4m3(), "global"), &opts);
    kv = rocke_b_param(b, "KV", rocke_ptr_type(b, rocke_fp8e4m3(), "global"), &opts);

    opts.align = 4;
    kv_scales = rocke_b_param(b, "kv_scales", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    weights = rocke_b_param(b, "weights", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    cu_starts = rocke_b_param(b, "cu_starts", rocke_ptr_type(b, rocke_i32(), "global"), &opts);
    cu_ends = rocke_b_param(b, "cu_ends", rocke_ptr_type(b, rocke_i32(), "global"), &opts);

    memset(&opts, 0, sizeof(opts));
    opts.writeonly = true;
    opts.writeonly_set = true;
    opts.align = 4;
    opts.align_set = true;
    logits = rocke_b_param(b, "logits", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
    seq_len = rocke_b_param(b, "seq_len", rocke_i32(), NULL);
    seq_len_kv = rocke_b_param(b, "seq_len_kv", rocke_i32(), NULL);
    stride_logits_s = rocke_b_param(b, "stride_logits_s", rocke_i32(), NULL);
    num_splits = rocke_b_param(b, "num_splits", rocke_i32(), NULL);

    tid = rocke_b_thread_id_x(b);
    bid = rocke_b_block_id_x(b);
    split_id = rocke_b_block_id_y(b);
    wave = rocke_b_div(b, tid, rocke_b_const_i32(b, 64));
    lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, 64));
    lane_decode = rocke_decode_mfma_lanes(b, atom, lane);

    n_blocks = rocke_fp8_mqa_ceildiv(b, seq_len, rpb);
    {
        rocke_value_t* forward_bid = rocke_b_sub(b, n_blocks, bid);
        rocke_value_t* one = rocke_b_const_i32(b, 1);
        reverse_bid = rocke_b_sub(b, forward_bid, one);
    }
    row0 = rocke_b_mul(b, reverse_bid, rocke_b_const_i32(b, rpb));
    zero_i32 = rocke_b_const_i32(b, 0);
    zero_f32 = rocke_b_const_f32(b, 0.0);

    starts.reserve((size_t)rpb);
    ends.reserve((size_t)rpb);
    q_fragments.reserve((size_t)rpb * (size_t)m_tiles * (size_t)k_steps);
    weight_fragments.reserve((size_t)rpb * (size_t)m_tiles * (size_t)atom->c_per_lane);
    for(int row_offset = 0; row_offset < rpb; ++row_offset)
    {
        rocke_value_t* row = rocke_b_add(b, row0, rocke_b_const_i32(b, row_offset));
        rocke_value_t* start
            = rocke_b_smax(b, rocke_b_global_load_i32(b, cu_starts, row, 0), zero_i32);
        rocke_value_t* end
            = rocke_b_smin(b, rocke_b_global_load_i32(b, cu_ends, row, 0), seq_len_kv);
        starts.push_back(start);
        ends.push_back(end);

        for(int mi = 0; mi < m_tiles; ++mi)
        {
            rocke_value_t* head
                = rocke_b_add(b, rocke_b_const_i32(b, mi * atom->m), lane_decode.m_in_atom);
            rocke_value_t* row_head
                = rocke_b_add(b, rocke_b_mul(b, row, rocke_b_const_i32(b, h)), head);
            rocke_value_t* q_base = rocke_b_mul(b, row_head, rocke_b_const_i32(b, d));
            for(int kk = 0; kk < k_steps; ++kk)
            {
                rocke_value_t* k_step = rocke_b_const_i32(b, kk * atom->k);
                rocke_value_t* lane_width = rocke_b_const_i32(b, atom->a_per_lane);
                rocke_value_t* lane_offset = rocke_b_mul(b, lane_decode.k_blk, lane_width);
                rocke_value_t* k_lane = rocke_b_add(b, k_step, lane_offset);
                rocke_value_t* q_addr = rocke_b_add(b, q_base, k_lane);
                q_fragments.push_back(rocke_b_global_load_vN(
                    b, q, q_addr, rocke_fp8e4m3(), atom->a_per_lane, atom->a_per_lane));
            }

            for(int elem = 0; elem < atom->c_per_lane; ++elem)
            {
                rocke_value_t* c_width = rocke_b_const_i32(b, atom->c_per_lane);
                rocke_value_t* head_base = rocke_b_mul(b, lane_decode.k_blk, c_width);
                rocke_value_t* elem_value = rocke_b_const_i32(b, elem);
                rocke_value_t* head_offset = rocke_b_add(b, head_base, elem_value);
                rocke_value_t* weight_head
                    = rocke_b_add(b, rocke_b_const_i32(b, mi * atom->m), head_offset);
                rocke_value_t* weight_addr
                    = rocke_b_add(b, rocke_b_mul(b, row, rocke_b_const_i32(b, h)), weight_head);
                weight_fragments.push_back(rocke_b_global_load_f32(b, weights, weight_addr, 0));
            }
        }
    }

    tile_start = starts[0];
    tile_end = ends[0];
    for(int row_offset = 1; row_offset < rpb; ++row_offset)
    {
        tile_start = rocke_b_smin(b, tile_start, starts[(size_t)row_offset]);
        tile_end = rocke_b_smax(b, tile_end, ends[(size_t)row_offset]);
    }
    {
        rocke_value_t* divisor = rocke_b_const_i32(b, bkv);
        rocke_value_t* tile_index = rocke_b_div(b, tile_start, divisor);
        rocke_value_t* multiplier = rocke_b_const_i32(b, bkv);
        tile_start = rocke_b_mul(b, tile_index, multiplier);
    }

    window_tiles = rocke_fp8_mqa_ceildiv(b, rocke_b_sub(b, tile_end, tile_start), bkv);
    {
        rocke_value_t* split_tiles = rocke_fp8_mqa_ceildiv_value(b, window_tiles, num_splits);
        rocke_value_t* block_width = rocke_b_const_i32(b, bkv);
        split_columns = rocke_b_mul(b, split_tiles, block_width);
    }
    tile_start = rocke_b_add(b, tile_start, rocke_b_mul(b, split_id, split_columns));
    tile_end = rocke_b_smin(b, rocke_b_add(b, tile_start, split_columns), tile_end);

    tile_loop = rocke_b_scf_for(b, tile_start, tile_end, rocke_b_const_i32(b, bkv), "col0");
    rocke_b_region_enter(b, tile_loop.body);
    {
        rocke_value_t* col0 = tile_loop.iv;
        rocke_value_t* wave_tile_base
            = rocke_b_mul(b, wave, rocke_b_const_i32(b, n_tiles_per_wave));
        std::vector<rocke_value_t*> columns;
        std::vector<rocke_value_t*> scales;
        std::vector<rocke_value_t*> kv_fragments;
        columns.reserve((size_t)n_tiles_per_wave);
        scales.reserve((size_t)n_tiles_per_wave);
        kv_fragments.reserve((size_t)n_tiles_per_wave * (size_t)k_steps);

        for(int ni = 0; ni < n_tiles_per_wave; ++ni)
        {
            rocke_value_t* absolute_ni = rocke_b_add(b, wave_tile_base, rocke_b_const_i32(b, ni));
            rocke_value_t* column = rocke_b_add(
                b,
                rocke_b_add(b, col0, rocke_b_mul(b, absolute_ni, rocke_b_const_i32(b, atom->n))),
                lane_decode.n_in_atom);
            rocke_value_t* clamped_column
                = rocke_b_smin(b, column, rocke_b_sub(b, seq_len_kv, rocke_b_const_i32(b, 1)));
            columns.push_back(column);
            scales.push_back(rocke_b_global_load_f32(b, kv_scales, clamped_column, 0));
            rocke_value_t* kv_base = rocke_b_mul(b, clamped_column, rocke_b_const_i32(b, d));
            for(int kk = 0; kk < k_steps; ++kk)
            {
                rocke_value_t* k_step = rocke_b_const_i32(b, kk * atom->k);
                rocke_value_t* lane_width = rocke_b_const_i32(b, atom->b_per_lane);
                rocke_value_t* lane_offset = rocke_b_mul(b, lane_decode.k_blk, lane_width);
                rocke_value_t* k_lane = rocke_b_add(b, k_step, lane_offset);
                rocke_value_t* kv_addr = rocke_b_add(b, kv_base, k_lane);
                kv_fragments.push_back(rocke_b_global_load_vN(
                    b, kv, kv_addr, rocke_fp8e4m3(), atom->b_per_lane, atom->b_per_lane));
            }
        }

        for(int row_offset = 0; row_offset < rpb; ++row_offset)
        {
            rocke_value_t* row = rocke_b_add(b, row0, rocke_b_const_i32(b, row_offset));
            rocke_value_t* row_i64 = rocke_b_sext(b, row, rocke_i64());
            rocke_value_t* stride_i64 = rocke_b_sext(b, stride_logits_s, rocke_i64());
            rocke_value_t* row_stride = rocke_b_mul(b, row_i64, stride_i64);
            rocke_value_t* four = rocke_b_const_i64(b, 4);
            rocke_value_t* row_byte_offset = rocke_b_mul(b, row_stride, four);
            rocke_value_t* logits_row = rocke_b_global_ptr_add(b, logits, row_byte_offset);
            for(int ni = 0; ni < n_tiles_per_wave; ++ni)
            {
                rocke_value_t* column_sum = zero_f32;
                for(int mi = 0; mi < m_tiles; ++mi)
                {
                    rocke_value_t* accumulator = rocke_b_zero_vec_f32(b, atom->c_per_lane);
                    for(int kk = 0; kk < k_steps; ++kk)
                    {
                        size_t q_index
                            = ((size_t)row_offset * (size_t)m_tiles + (size_t)mi) * (size_t)k_steps
                              + (size_t)kk;
                        size_t kv_index = (size_t)ni * (size_t)k_steps + (size_t)kk;
                        accumulator = rocke_b_mma(b,
                                                  atom->name,
                                                  q_fragments[q_index],
                                                  kv_fragments[kv_index],
                                                  accumulator,
                                                  NULL,
                                                  0);
                    }
                    for(int elem = 0; elem < atom->c_per_lane; ++elem)
                    {
                        size_t weight_index = ((size_t)row_offset * (size_t)m_tiles + (size_t)mi)
                                                  * (size_t)atom->c_per_lane
                                              + (size_t)elem;
                        rocke_value_t* score = rocke_b_vec_extract(b, accumulator, elem);
                        rocke_value_t* relu = rocke_b_fmax(b, score, zero_f32);
                        column_sum
                            = rocke_b_fma(b, relu, weight_fragments[weight_index], column_sum);
                    }
                }
                column_sum = rocke_b_fmul(b, column_sum, scales[(size_t)ni]);
                column_sum
                    = rocke_b_fadd(b, column_sum, rocke_b_warp_shuffle_xor(b, column_sum, 16));
                column_sum
                    = rocke_b_fadd(b, column_sum, rocke_b_warp_shuffle_xor(b, column_sum, 32));

                rocke_value_t* after_start
                    = rocke_b_cmp_ge(b, columns[(size_t)ni], starts[(size_t)row_offset]);
                rocke_value_t* before_end
                    = rocke_b_cmp_lt(b, columns[(size_t)ni], ends[(size_t)row_offset]);
                rocke_value_t* in_window = rocke_b_land(b, after_start, before_end);
                rocke_value_t* first_k = rocke_b_cmp_eq(b, lane_decode.k_blk, zero_i32);
                rocke_value_t* is_writer = rocke_b_land(b, first_k, in_window);
                rocke_if_t write_if = rocke_b_scf_if(b, is_writer);
                rocke_b_region_enter(b, write_if.then_region);
                rocke_b_global_store(b, logits_row, columns[(size_t)ni], column_sum, 4);
                rocke_b_region_leave(b);
            }
        }
    }
    rocke_b_region_leave(b);

    rocke_b_ret(b);
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_fp8_mqa_logits(rocke_ir_builder_t* b,
                                               const rocke_fp8_mqa_logits_spec_t* spec,
                                               const char* arch)
{
    return ckc::guard_builder(
        b, [&]() -> rocke_kernel_def_t* { return rocke_build_fp8_mqa_logits_impl(b, spec, arch); });
}

rocke_kernel_def_t* rocke_build_fp8_mqa_logits_new(rocke_ir_builder_t* b,
                                                   const rocke_fp8_mqa_logits_spec_t* spec,
                                                   const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_fp8_mqa_logits_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_fp8_mqa_logits_impl(b, spec, arch);
    });
}

rocke_status_t rocke_fp8_mqa_logits_lower_to_llvm(const rocke_fp8_mqa_logits_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t status;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_fp8_mqa_set_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx942";
    }
    kernel = rocke_build_fp8_mqa_logits_new(&b, spec, arch);
    if(kernel == NULL)
    {
        status = rocke_ir_builder_status(&b);
        rocke_fp8_mqa_set_reason(err, err_cap, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return status == ROCKE_OK ? ROCKE_ERR_VALUE : status;
    }
    status = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return status;
}

int rocke_fp8_mqa_logits_num_splits(
    int seq_len_padded, int seq_len_kv, int rows_per_block, int block_kv, int num_cus)
{
    int grid_x = seq_len_padded / rows_per_block;
    int target_blocks;
    int max_splits;
    int needed;

    if(grid_x == 0 || seq_len_kv < 4096)
    {
        return 1;
    }
    target_blocks = 4 * num_cus;
    if(grid_x >= target_blocks)
    {
        return 1;
    }
    max_splits = (seq_len_kv / block_kv) / ROCKE_FP8_MQA_MIN_TILES_PER_SPLIT;
    if(max_splits < 1)
    {
        max_splits = 1;
    }
    needed = (target_blocks + grid_x - 1) / grid_x;
    if(needed > max_splits)
    {
        needed = max_splits;
    }
    return needed > 1 ? needed : 1;
}

rocke_status_t rocke_fp8_mqa_logits_grid(int seq_len_padded,
                                         int num_splits,
                                         const rocke_fp8_mqa_logits_spec_t* spec,
                                         int out[3])
{
    if(spec == NULL || out == NULL || spec->rows_per_block <= 0
       || seq_len_padded % spec->rows_per_block)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = seq_len_padded / spec->rows_per_block;
    out[1] = num_splits;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_fp8_mqa_logits_signature(rocke_arena_t* arena,
                                              const rocke_fp8_mqa_logits_spec_t* spec,
                                              const rocke_sig_entry_t** out_items,
                                              size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t status;
    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    status = rocke_signature_builder_init(&sb, arena);
    if(status != ROCKE_OK)
    {
        return status;
    }
    rocke_signature_builder_ptr(&sb, "Q", "fp8e4m3", NULL);
    rocke_signature_builder_ptr(&sb, "KV", "fp8e4m3", NULL);
    rocke_signature_builder_ptr(&sb, "kv_scales", "f32", NULL);
    rocke_signature_builder_ptr(&sb, "weights", "f32", NULL);
    rocke_signature_builder_ptr(&sb, "cu_starts", "i32", NULL);
    rocke_signature_builder_ptr(&sb, "cu_ends", "i32", NULL);
    rocke_signature_builder_ptr(&sb, "logits", "f32", NULL);
    rocke_signature_builder_scalar(&sb, "seq_len", "i32");
    rocke_signature_builder_scalar(&sb, "seq_len_kv", "i32");
    rocke_signature_builder_scalar(&sb, "stride_logits_s", "i32");
    rocke_signature_builder_scalar(&sb, "num_splits", "i32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}
