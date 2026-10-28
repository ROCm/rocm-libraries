// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instances/gfx1250/mxfp4_tdm_gemm.cpp -- C99 port of
 * rocke.instances.gfx1250.mxfp4_tdm_gemm.
 *
 * Python (mxfp4_tdm_gemm.py)             C99 (this file)
 * -------------------------------------  ------------------------------------
 * MxFp4TdmGemmSpec                       rocke_mxfp4_tdm_gemm_spec_t
 * build_mxfp4_tdm_gemm(spec, arch)       rocke_build_mxfp4_tdm_gemm(...)
 *
 * Every builder call is a separate statement. Their order is the Python
 * operation order and therefore determines the byte-identical SSA numbering.
 */
#include "rocke/instance_gfx1250_mxfp4_tdm_gemm.h"

#include <string.h>

#include "rocke/ir_internal.h"

typedef struct mxfp4_ctx
{
    rocke_ir_builder_t* b;
    rocke_value_t *A, *B, *AS, *BS, *M, *N;
    rocke_value_t *c0, *c1, *c16, *c32;
    rocke_value_t *packed_k, *scale_groups;
    rocke_value_t *lane, *lane_col, *lane_half, *wave;
    rocke_value_t *block_m, *block_n, *shared, *shared_addr;
    rocke_value_t *atom_row, *atom_col;
} mxfp4_ctx_t;

static void buffer_offsets(mxfp4_ctx_t* c,
                           rocke_value_t* parity,
                           rocke_value_t** a_offset,
                           rocke_value_t** b_offset)
{
    rocke_value_t* a_bytes = rocke_b_const_i32(c->b, 4096);
    *a_offset = rocke_b_mul(c->b, parity, a_bytes);
    rocke_value_t* b_region = rocke_b_const_i32(c->b, 8192);
    rocke_value_t* b_bytes = rocke_b_const_i32(c->b, 2048);
    rocke_value_t* scaled = rocke_b_mul(c->b, parity, b_bytes);
    *b_offset = rocke_b_add(c->b, b_region, scaled);
}

static void issue_tdm(mxfp4_ctx_t* c, rocke_value_t* k0, rocke_value_t* parity)
{
    rocke_value_t *a_offset, *b_offset;
    rocke_value_t *a_zext, *b_zext, *a_lds, *b_lds;
    rocke_value_t *two, *k_byte, *a_mul, *a_element, *b_mul, *b_element, *remaining;
    rocke_if_t guard;
    rocke_value_t* d[5];

    buffer_offsets(c, parity, &a_offset, &b_offset);
    a_zext = rocke_b_zext(c->b, a_offset, c->shared_addr->type);
    a_lds = rocke_b_smem_ptr_add(c->b, c->shared_addr, a_zext);
    b_zext = rocke_b_zext(c->b, b_offset, c->shared_addr->type);
    b_lds = rocke_b_smem_ptr_add(c->b, c->shared_addr, b_zext);
    two = rocke_b_const_i32(c->b, 2);
    k_byte = rocke_b_div(c->b, k0, two);
    a_mul = rocke_b_mul(c->b, c->block_m, c->packed_k);
    a_element = rocke_b_add(c->b, a_mul, k_byte);
    b_mul = rocke_b_mul(c->b, c->block_n, c->packed_k);
    b_element = rocke_b_add(c->b, b_mul, k_byte);
    remaining = rocke_b_sub(c->b, c->packed_k, k_byte);
    guard = rocke_b_scf_if(c->b, rocke_b_cmp_eq(c->b, c->wave, c->c0));
    rocke_b_region_enter(c->b, guard.then_region);
    rocke_value_t* a_ptr = rocke_b_global_ptr_add(c->b, c->A, a_element);
    rocke_value_t* a_rows = rocke_b_sub(c->b, c->M, c->block_m);
    rocke_b_tdm_descriptor_2d(
        c->b, a_ptr, a_lds, 1, remaining, a_rows, c->packed_k, 64, 64, 0, -1, 0, d);
    rocke_b_tensor_load_to_lds(c->b, d[0], d[1], d[2], d[3], d[4], 0);
    rocke_value_t* b_ptr = rocke_b_global_ptr_add(c->b, c->B, b_element);
    rocke_value_t* b_rows = rocke_b_sub(c->b, c->N, c->block_n);
    rocke_b_tdm_descriptor_2d(
        c->b, b_ptr, b_lds, 1, remaining, b_rows, c->packed_k, 64, 32, 0, -1, 0, d);
    rocke_b_tensor_load_to_lds(c->b, d[0], d[1], d[2], d[3], d[4], 0);
    rocke_b_region_leave(c->b);
}

static rocke_value_t* load_i32x4(mxfp4_ctx_t* c, rocke_value_t* byte_offset)
{
    rocke_value_t* indices[] = {byte_offset};
    return rocke_b_smem_load_vN(c->b, c->shared, indices, 1, rocke_i32(), 4);
}

static rocke_value_t*
    load_a_row(mxfp4_ctx_t* c, rocke_value_t* a_buffer, rocke_value_t* row, rocke_value_t* lane_k)
{
    rocke_value_t* row_stride = rocke_b_const_i32(c->b, 64);
    rocke_value_t* row_mul = rocke_b_mul(c->b, row, row_stride);
    rocke_value_t* row_base = rocke_b_add(c->b, a_buffer, row_mul);
    rocke_value_t* inner = rocke_b_const_i32(c->b, 0);
    rocke_value_t* inner_lane = rocke_b_add(c->b, inner, lane_k);
    rocke_value_t* first = rocke_b_add(c->b, row_base, inner_lane);
    rocke_value_t* second = rocke_b_add(c->b, first, c->c32);
    rocke_value_t* first_v = load_i32x4(c, first);
    rocke_value_t* second_v = load_i32x4(c, second);
    return rocke_b_vec_concat(c->b, first_v, second_v);
}

static rocke_value_t* load_a_fragment(mxfp4_ctx_t* c, rocke_value_t* a_buffer)
{
    rocke_value_t* row0 = rocke_b_add(c->b, c->atom_row, c->lane_col);
    rocke_value_t* row1 = rocke_b_add(c->b, row0, c->c16);
    rocke_value_t* lane_k = rocke_b_mul(c->b, c->lane_half, c->c16);
    rocke_value_t* first = load_a_row(c, a_buffer, row0, lane_k);
    rocke_value_t* second = load_a_row(c, a_buffer, row1, lane_k);
    return rocke_b_vec_concat(c->b, first, second);
}

static rocke_value_t* load_b_fragment(mxfp4_ctx_t* c, rocke_value_t* b_buffer)
{
    rocke_value_t* row = rocke_b_add(c->b, c->atom_col, c->lane_col);
    rocke_value_t* row_stride = rocke_b_const_i32(c->b, 64);
    rocke_value_t* row_mul = rocke_b_mul(c->b, row, row_stride);
    rocke_value_t* row_base = rocke_b_add(c->b, b_buffer, row_mul);
    rocke_value_t* lane_k = rocke_b_mul(c->b, c->lane_half, c->c16);
    rocke_value_t* inner = rocke_b_const_i32(c->b, 0);
    rocke_value_t* inner_lane = rocke_b_add(c->b, inner, lane_k);
    rocke_value_t* first = rocke_b_add(c->b, row_base, inner_lane);
    rocke_value_t* first_v = load_i32x4(c, first);
    rocke_value_t* second = rocke_b_add(c->b, first, c->c32);
    rocke_value_t* second_v = load_i32x4(c, second);
    return rocke_b_vec_concat(c->b, first_v, second_v);
}

static rocke_value_t* pack_a_scale(mxfp4_ctx_t* c, rocke_value_t* global_row, rocke_value_t* group0)
{
    rocke_value_t* row_base = rocke_b_mul(c->b, global_row, c->scale_groups);
    rocke_value_t* index = rocke_b_add(c->b, row_base, group0);
    rocke_value_t* raw = rocke_b_global_load_vN(c->b, c->AS, index, rocke_i8(), 4, 1);
    return rocke_b_bitcast(c->b, raw, rocke_i32());
}

static rocke_value_t* pack_b_scale(mxfp4_ctx_t* c, rocke_value_t* global_col, rocke_value_t* group0)
{
    rocke_value_t* values[4];
    for(int group = 0; group < 4; ++group)
    {
        rocke_value_t* group_c = rocke_b_const_i32(c->b, group);
        rocke_value_t* group_i = rocke_b_add(c->b, group0, group_c);
        rocke_value_t* group_base = rocke_b_mul(c->b, group_i, c->N);
        rocke_value_t* index = rocke_b_add(c->b, group_base, global_col);
        values[group] = rocke_b_global_load(c->b, c->BS, index, rocke_i8(), 1);
    }
    rocke_value_t* packed = rocke_b_vec_pack(c->b, values, 4, rocke_i8());
    return rocke_b_bitcast(c->b, packed, rocke_i32());
}

static rocke_value_t* compute_tile(mxfp4_ctx_t* c,
                                   rocke_value_t* k0,
                                   rocke_value_t* parity,
                                   rocke_value_t* accumulator)
{
    rocke_value_t *a_buffer, *b_buffer;
    buffer_offsets(c, parity, &a_buffer, &b_buffer);
    rocke_value_t* inner = rocke_b_const_i32(c->b, 0);
    rocke_value_t* k_inner = rocke_b_add(c->b, k0, inner);
    rocke_value_t* group0 = rocke_b_div(c->b, k_inner, c->c32);
    rocke_value_t* a_frag = load_a_fragment(c, a_buffer);
    rocke_value_t* row_lane = rocke_b_add(c->b, c->atom_row, c->lane);
    rocke_value_t* global_row = rocke_b_add(c->b, c->block_m, row_lane);
    rocke_value_t* a_scale = pack_a_scale(c, global_row, group0);
    rocke_value_t* b_frag = load_b_fragment(c, b_buffer);
    rocke_value_t* col_lane = rocke_b_add(c->b, c->atom_col, c->lane_col);
    rocke_value_t* global_col = rocke_b_add(c->b, c->block_n, col_lane);
    rocke_value_t* b_scale = pack_b_scale(c, global_col, group0);
    rocke_value_t* scales[] = {a_scale, b_scale};
    return rocke_b_mma(c->b, "wmma_scale_f32_32x16x128_f4", a_frag, b_frag, accumulator, scales, 2);
}

rocke_mxfp4_tdm_gemm_spec_t rocke_mxfp4_tdm_gemm_spec_default(void)
{
    rocke_mxfp4_tdm_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = "mxfp4_tdm_gemm";
    s.tile_m = 128;
    s.tile_n = 64;
    s.tile_k = 128;
    s.warp_m = 2;
    s.warp_n = 2;
    s.warp_k = 1;
    s.warp_tile_m = 32;
    s.warp_tile_n = 16;
    s.warp_tile_k = 128;
    s.block_k = 32;
    s.tdm_lds = true;
    s.tdm_prefetch = true;
    s.tdm_prefetch_depth = 2;
    return s;
}

rocke_kernel_def_t* rocke_build_mxfp4_tdm_gemm(rocke_ir_builder_t* b,
                                               const rocke_mxfp4_tdm_gemm_spec_t* spec,
                                               const char* arch)
{
    mxfp4_ctx_t c;
    rocke_param_opts_t opt;
    rocke_value_t *C, *D, *K, *alpha, *beta, *tid, *wave_n, *wave_m;
    rocke_value_t *warp_n_c, *block_y, *tile_m_c, *block_x, *tile_n_c;
    rocke_value_t *atom_row_mul, *atom_row_zero, *atom_col_mul, *atom_col_zero;
    rocke_value_t *accumulator, *upper, *step;
    rocke_iter_arg_t iter_args[2];
    rocke_for_t loop;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    if(strcmp(arch, "gfx1250") != 0 || spec->M != 64 || spec->N != 32 || spec->K != 256
       || spec->tile_m != 64 || spec->tile_n != 32 || spec->tile_k != 128 || spec->warp_m != 2
       || spec->warp_n != 2 || spec->warp_k != 1 || spec->warp_tile_m != 32
       || spec->warp_tile_n != 16 || spec->warp_tile_k != 128 || spec->block_k != 32
       || !spec->tdm_lds || !spec->tdm_prefetch || spec->tdm_prefetch_depth != 2)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "C++ mxfp4_tdm_gemm currently supports only parity config 0");
    }

    memset(&c, 0, sizeof(c));
    c.b = b;
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 128);
    memset(&opt, 0, sizeof(opt));
    opt.noalias = opt.noalias_set = opt.readonly = opt.readonly_set = true;
    opt.align = 16;
    opt.align_set = true;
    c.A = rocke_b_param(b, "A", rocke_ptr_type(b, rocke_i8(), "global"), &opt);
    c.B = rocke_b_param(b, "B", rocke_ptr_type(b, rocke_i8(), "global"), &opt);
    opt.align = 4;
    c.AS = rocke_b_param(b, "A_scale", rocke_ptr_type(b, rocke_i8(), "global"), &opt);
    c.BS = rocke_b_param(b, "B_scale", rocke_ptr_type(b, rocke_i8(), "global"), &opt);
    opt.align = 1;
    C = rocke_b_param(b, "C", rocke_ptr_type(b, rocke_fp8e4m3(), "global"), &opt);
    opt.readonly = opt.readonly_set = false;
    opt.writeonly = opt.writeonly_set = true;
    D = rocke_b_param(b, "D", rocke_ptr_type(b, rocke_fp8e4m3(), "global"), &opt);
    c.M = rocke_b_param(b, "M", rocke_i32(), NULL);
    c.N = rocke_b_param(b, "N", rocke_i32(), NULL);
    K = rocke_b_param(b, "K", rocke_i32(), NULL);
    alpha = rocke_b_param(b, "alpha", rocke_f32(), NULL);
    beta = rocke_b_param(b, "beta", rocke_f32(), NULL);

    c.c0 = rocke_b_const_i32(b, 0);
    c.c1 = rocke_b_const_i32(b, 1);
    c.c16 = rocke_b_const_i32(b, 16);
    c.c32 = rocke_b_const_i32(b, 32);
    rocke_value_t* two = rocke_b_const_i32(b, 2);
    c.packed_k = rocke_b_div(b, K, two);
    c.scale_groups = rocke_b_div(b, K, c.c32);
    tid = rocke_b_thread_id_x(b);
    c.lane = rocke_b_mod(b, tid, c.c32);
    c.lane_col = rocke_b_mod(b, c.lane, c.c16);
    c.lane_half = rocke_b_div(b, c.lane, c.c16);
    c.wave = rocke_b_div(b, tid, c.c32);
    warp_n_c = rocke_b_const_i32(b, 2);
    wave_n = rocke_b_mod(b, c.wave, warp_n_c);
    warp_n_c = rocke_b_const_i32(b, 2);
    wave_m = rocke_b_div(b, c.wave, warp_n_c);
    block_y = rocke_b_block_id_y(b);
    tile_m_c = rocke_b_const_i32(b, 64);
    c.block_m = rocke_b_mul(b, block_y, tile_m_c);
    block_x = rocke_b_block_id_x(b);
    tile_n_c = rocke_b_const_i32(b, 32);
    c.block_n = rocke_b_mul(b, block_x, tile_n_c);
    {
        const int shape[] = {12288};
        c.shared = rocke_b_smem_alloc(b, rocke_i8(), shape, 1, "ab_tdm");
    }
    c.shared_addr = rocke_b_smem_addr_of(b, c.shared);
    rocke_value_t* atom_row_stride = rocke_b_const_i32(b, 32);
    atom_row_mul = rocke_b_mul(b, wave_m, atom_row_stride);
    atom_row_zero = rocke_b_const_i32(b, 0);
    c.atom_row = rocke_b_add(b, atom_row_mul, atom_row_zero);
    rocke_value_t* atom_col_stride = rocke_b_const_i32(b, 16);
    atom_col_mul = rocke_b_mul(b, wave_n, atom_col_stride);
    atom_col_zero = rocke_b_const_i32(b, 0);
    c.atom_col = rocke_b_add(b, atom_col_mul, atom_col_zero);
    accumulator = rocke_b_zero_vec_f32(b, 16);
    issue_tdm(&c, c.c0, c.c0);
    rocke_value_t* tile_k_c = rocke_b_const_i32(b, 128);
    upper = rocke_b_sub(b, K, tile_k_c);
    iter_args[0] = (rocke_iter_arg_t){"parity", c.c0};
    iter_args[1] = (rocke_iter_arg_t){"acc0", accumulator};
    step = rocke_b_const_i32(b, 128);
    loop = rocke_b_scf_for_iter(b, c.c0, upper, step, iter_args, 2, "k0", false, false);
    rocke_b_region_enter(b, loop.body);
    {
        rocke_value_t* k0 = loop.iv;
        rocke_value_t* parity = loop.iter_vars[0];
        rocke_value_t* next = rocke_b_sub(b, c.c1, parity);
        rocke_b_s_wait_tensorcnt(b, 0);
        rocke_b_s_waitcnt(b, -1, 0, -1);
        rocke_b_s_barrier_bare(b);
        rocke_value_t* tile_k_next = rocke_b_const_i32(b, 128);
        rocke_value_t* next_k = rocke_b_add(b, k0, tile_k_next);
        issue_tdm(&c, next_k, next);
        rocke_value_t* updated = compute_tile(&c, k0, parity, loop.iter_vars[1]);
        rocke_value_t* y[] = {next, updated};
        rocke_b_scf_yield(b, y, 2);
    }
    rocke_b_region_leave(b);
    rocke_value_t* final_parity = loop.op->results[0];
    rocke_b_s_wait_tensorcnt(b, 0);
    rocke_b_s_waitcnt(b, -1, 0, -1);
    rocke_b_s_barrier_bare(b);
    rocke_value_t* final_acc = compute_tile(&c, upper, final_parity, loop.op->results[1]);
    for(int slot = 0; slot < 16; ++slot)
    {
        int slot_row = (slot / 8) * 16 + slot % 8;
        rocke_value_t* slot_c = rocke_b_const_i32(b, slot_row);
        rocke_value_t* eight = rocke_b_const_i32(b, 8);
        rocke_value_t* lane_row = rocke_b_mul(b, c.lane_half, eight);
        rocke_value_t* slot_lane = rocke_b_add(b, slot_c, lane_row);
        rocke_value_t* atom_slot = rocke_b_add(b, c.atom_row, slot_lane);
        rocke_value_t* row = rocke_b_add(b, c.block_m, atom_slot);
        rocke_value_t* atom_lane_col = rocke_b_add(b, c.atom_col, c.lane_col);
        rocke_value_t* col = rocke_b_add(b, c.block_n, atom_lane_col);
        rocke_value_t* row_base = rocke_b_mul(b, row, c.N);
        rocke_value_t* index = rocke_b_add(b, row_base, col);
        rocke_value_t* acc_value = rocke_b_vec_extract(b, final_acc, slot);
        rocke_value_t* value = rocke_b_fmul(b, alpha, acc_value);
        rocke_value_t* c_raw = rocke_b_global_load(b, C, index, rocke_fp8e4m3(), 1);
        rocke_value_t* c_value = rocke_b_cvt_fp8_to_f32(b, c_raw);
        rocke_value_t* beta_c = rocke_b_fmul(b, beta, c_value);
        value = rocke_b_fadd(b, value, beta_c);
        rocke_value_t* output = rocke_b_cvt_f32_to_fp8(b, value);
        rocke_b_global_store(b, D, index, output, 1);
    }
    rocke_b_ret(b);
    return b->kernel;
}
