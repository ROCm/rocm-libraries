// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * build_direct_depthwise_col -- column-streamed depthwise conv forward.
 *
 * Faithful C port of `build_direct_depthwise_col` in
 * library/kernels/common/conv_direct_grouped.py.  Byte-identity with the Python
 * engine is the repo's #1 invariant, so every op below is emitted in the same
 * order the Python builder emits it; the comments name the Python construct
 * each block mirrors.
 *
 * Shape of the algorithm: the KW axis is a runtime scf.for whose iter_args
 * carry a BLOCK_H x BLOCK_W accumulator band, while the input-row axis (y)
 * and the KH axis (r) are host-unrolled over one output-row tile.  Because y
 * and r are both host ints, the strided liveness test `(y - r) % stride == 0`
 * and the band row `(y - r) / stride` resolve at emission time, so a strided
 * kernel costs no runtime arithmetic -- it simply emits fewer FMAs per y.
 *
 * AOT: the kernel takes the direct-conv kernarg block (rocke_dconv_emit_params)
 * and reads the image extents, group count and tensor strides from it.  The
 * row tile is the build-time capability: block_id_z encodes
 * n * n_h_tiles + h_tile, with n_h_tiles derived from the p_Ho kernarg, and
 * every store is bounded against p_Ho / p_Wo / p_groups.
 *
 * Two deliberate differences from the preload sibling
 * (conv_direct_grouped_build_depthwise.cpp), each of which is byte-identity
 * critical rather than cosmetic:
 *
 *   1. n_iters is (BLOCK_H - 1) * stride + KH per tile, and the band row of a
 *      tap is (y - r) / stride rather than a circular KH-slot index.
 *   2. No value-side zero-fill after a load.  The hardware bounds check already
 *      returns 0 for an out-of-range buffer access, and +0.0 survives the
 *      convert in every supported dtype, so selecting over it again would be a
 *      dead cndmask.
 */

#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.fuse.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

/* Python's `//` floors; C's `/` truncates toward zero.  Every use below is on a
 * value already known to be divisible or non-negative, but the kernel's row
 * indices are derived from `y - r`, which is negative for the leading padded
 * rows, so the floor semantics are spelled out rather than assumed. */
static int dwcol_floor_div(int a, int c)
{
    int q;
    if(c == 0)
    {
        return 0;
    }
    q = a / c;
    if((a % c != 0) && ((a < 0) != (c < 0)))
    {
        --q;
    }
    return q;
}

/* Python `addr(off, cond)`: element offset -> byte offset, poisoned to the OOB
 * sentinel when the access is not valid. */
static rocke_value_t*
    dwcol_addr(rocke_dconv_dwcol_ctx_t* ctx, rocke_value_t* off, rocke_value_t* cond)
{
    return rocke_b_select(
        ctx->b, cond, rocke_b_mul(ctx->b, off, ctx->c_elem_bytes), ctx->oob_sentinel);
}

/* Python `load_elem`: one element of A or B, widened to f32.  Dispatched on DT
 * rather than routed through the generic buffer_load so the f16 path emits
 * exactly the ops it did before this kernel grew a dtype knob. */
static rocke_value_t*
    dwcol_load_elem(rocke_dconv_dwcol_ctx_t* ctx, rocke_value_t* rsrc, rocke_value_t* byte_off)
{
    rocke_ir_builder_t* b = ctx->b;
    if(strcmp(ctx->DT->name, "f16") == 0)
    {
        return rocke_b_cast_to_f32(b, rocke_b_buffer_load_f16(b, rsrc, byte_off, ctx->c0));
    }
    return rocke_b_cast_to_f32(b, rocke_b_buffer_load_bf16(b, rsrc, byte_off, ctx->c0));
}

/* Python `store_elem`: narrow the f32 accumulator back to the tensor dtype. */
static void dwcol_store_elem(rocke_dconv_dwcol_ctx_t* ctx,
                             rocke_value_t* rsrc,
                             rocke_value_t* byte_off,
                             rocke_value_t* acc)
{
    rocke_ir_builder_t* b = ctx->b;
    if(strcmp(ctx->DT->name, "f16") == 0)
    {
        rocke_b_buffer_store_f16(b, rsrc, byte_off, ctx->c0, rocke_b_trunc_f32_to_f16(b, acc));
    }
    else
    {
        rocke_b_buffer_store_bf16(b, rsrc, byte_off, ctx->c0, rocke_b_trunc_f32_to_bf16(b, acc));
    }
}

/* ===================================================================== *
 *  Prologue: validate, derive geometry, emit params / constants / ids.
 * ===================================================================== */
bool rocke_dconv_dwcol_prologue(rocke_dconv_dwcol_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_direct_depthwise_col_spec_t* spec = ctx->spec;
    char reason[ROCKE_ERR_MSG_CAP];

    /* Python: spec.validate() then is_valid_depthwise_col_spec(), both of which
     * raise before a single op is emitted. */
    if(rocke_direct_depthwise_col_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    if(!rocke_direct_depthwise_col_is_valid_spec(spec, ctx->arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }

    ctx->p = spec->problem;
    ctx->BLOCK_H = spec->block_h;
    ctx->BLOCK_W = spec->block_w;
    ctx->BLOCK_WAVES = spec->block_waves;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_depthwise_col_threads_per_block(spec);
    ctx->BLOCK_CH = rocke_direct_depthwise_col_block_ch(spec);
    /* Input rows one tile's band can reach: (BLOCK_H - 1) * stride + KH. */
    ctx->n_iters = rocke_direct_depthwise_col_n_iters(spec);

    ctx->DT = rocke_fuse_dtype_to_ir_str(spec->dtype);
    if(ctx->DT == NULL)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    /* Python: ELEM_BYTES = 2 -- the validator has already restricted DT to
     * f16/bf16, both 2 bytes wide. */
    ctx->ELEM_BYTES = 2;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    /* params: the AOT kernarg block, in conv_abi order. */
    rocke_dconv_emit_params(b, &ctx->params, "fwd", ctx->DT);
    ctx->A = ctx->params.A;
    ctx->Bp = ctx->params.Bp;
    ctx->D = ctx->params.D;
    ctx->A_bytes = ctx->params.A_bytes;
    ctx->B_bytes = ctx->params.B_bytes;
    ctx->D_bytes = ctx->params.D_bytes;

    /* Constants in Python source order: c0, c1, c_wave, c_elem_bytes,
     * oob_sentinel, zero_f32.  Note c1 is emitted SECOND here, unlike the
     * preload prologue which emits it later inside the group loop.  c_W and
     * c_groups are the p_Wo / p_groups kernargs (AOT), so emit nothing. */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c1 = rocke_b_const_i32(b, 1);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_W = ctx->params.p_Wo;
    ctx->c_groups = ctx->params.p_groups;
    ctx->c_elem_bytes = rocke_b_const_i32(b, ctx->ELEM_BYTES);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->zero_f32 = rocke_b_const_f32(b, 0.0);

    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);

    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->bz = rocke_b_block_id_z(b);
    /* H-tiling: bz encodes (n, h_tile) as n*n_h_tiles + h_tile.  The tile
     * height is a build-time capability, but how many tiles the image needs
     * follows the runtime output height, so the divisor comes from p_Ho. */
    {
        rocke_value_t* c_block_h = rocke_b_const_i32(b, ctx->BLOCK_H);
        rocke_value_t* n_h_tiles = rocke_b_div(
            b, rocke_b_add(b, ctx->params.p_Ho, rocke_b_const_i32(b, ctx->BLOCK_H - 1)), c_block_h);
        rocke_value_t* h_tile;
        ctx->n = rocke_b_div(b, ctx->bz, n_h_tiles);
        h_tile = rocke_b_mod(b, ctx->bz, n_h_tiles);
        ctx->ho_start = rocke_b_mul(b, h_tile, c_block_h);
    }
    /* First input row (in padded coordinates) of the tile's receptive field. */
    ctx->y_start = rocke_b_mul(b, ctx->ho_start, rocke_b_const_i32(b, ctx->p.stride));
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, rocke_b_const_i32(b, ctx->BLOCK_W));

    /* ch = by*BLOCK_CH + (wave_id*WAVE + lane) */
    {
        rocke_value_t* mul_by = rocke_b_mul(b, ctx->by, rocke_b_const_i32(b, ctx->BLOCK_CH));
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_wave);
        rocke_value_t* inner = rocke_b_add(b, mul_wave, ctx->lane);
        ctx->ch = rocke_b_add(b, mul_by, inner);
    }
    /* The group count is a kernarg, so whether the last channel tile is partial
     * is a launch-time fact: the guard is always emitted. */
    ctx->ch_in_range = rocke_b_cmp_lt(b, ctx->ch, ctx->c_groups);

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Descriptors: A and D take runtime extents, B stays naive.
 * ===================================================================== */
void rocke_dconv_dwcol_build_descriptors(rocke_dconv_dwcol_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

    /* A[N,H,W,C]: h = y_iter - PAD, w = wo*stride + s_off - PAD, both
     * bounds-checked against the kernargs. */
    {
        rocke_dynamic_tensor_descriptor_t* a_dyn = rocke_dconv_a_descriptor_dynamic(
            b, &ctx->params, ctx->p.PAD, ctx->p.stride, "wo", "s_off");
        if(!a_dyn)
            return;
        ctx->a_desc = &a_dyn->base;
    }
    {
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_lengths[4];
        b_lengths[0] = total_k;
        b_lengths[1] = ctx->p.KH;
        b_lengths[2] = ctx->p.KW;
        b_lengths[3] = 1;
        ctx->b_desc = rocke_tensor_descriptor_naive(b, "B", b_lengths, 4, NULL, b_coords, 4);
    }
    {
        rocke_dynamic_tensor_descriptor_t* d_dyn
            = rocke_dconv_d_descriptor_dynamic(b, &ctx->params);
        if(!d_dyn)
            return;
        ctx->d_desc = &d_dyn->base;
    }
}

/* ===================================================================== *
 *  The column loop plus the drain epilogue.
 * ===================================================================== */
rocke_kernel_def_t* rocke_dconv_dwcol_col_loop(rocke_dconv_dwcol_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const int KH = ctx->p.KH;
    const int KW = ctx->p.KW;
    const int S = ctx->p.stride;
    const int BLOCK_H = ctx->BLOCK_H;
    const int BLOCK_W = ctx->BLOCK_W;
    const int num_iargs = BLOCK_H * BLOCK_W;
    rocke_iter_arg_t* iargs;
    rocke_value_t** new_accs;
    rocke_value_t** w_col;
    rocke_value_t* const* final_accs;
    int* taps;
    rocke_for_t col_loop;
    rocke_value_t* s_iv;
    int y, r, w_out, h_out, i;

    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* Python: acc_args = [(f"dw_acc_h{h}_w{w}", zero_f32) for h ... for w ...].
     * The band is BLOCK_H x BLOCK_W and KH reaches 31+ in the supported space,
     * so these are alloca'd rather than fixed ctx arrays.  num_iargs and KH are
     * bounded rather than free: rocke_dconv_dwcol_prologue runs
     * rocke_direct_depthwise_col_is_valid_spec before any of these allocas, and
     * that gate rejects every spec whose block_h * block_w + KH exceeds
     * min(max_live_f32, vgprs * 3 / 8) -- a few hundred f32 on any supported
     * target -- so both counts stay well inside the stack frame. */
    {
        char(*name_store)[32] = (char(*)[32])alloca((size_t)num_iargs * 32 * sizeof(char));
        int idx = 0;
        iargs = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
        new_accs = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
        for(h_out = 0; h_out < BLOCK_H; ++h_out)
        {
            for(w_out = 0; w_out < BLOCK_W; ++w_out)
            {
                snprintf(name_store[idx], 32, "dw_acc_h%d_w%d", h_out, w_out);
                iargs[idx].name = name_store[idx];
                iargs[idx].init = ctx->zero_f32;
                ++idx;
            }
        }
    }
    w_col = (rocke_value_t**)alloca((size_t)KH * sizeof(rocke_value_t*));
    taps = (int*)alloca((size_t)KH * sizeof(int));

    col_loop = rocke_b_scf_for_iter(b,
                                    ctx->c0,
                                    rocke_b_const_i32(b, KW),
                                    ctx->c1,
                                    iargs,
                                    num_iargs,
                                    "dw_col",
                                    /*unroll=*/false,
                                    /*elide_trailing_barrier=*/false);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    rocke_b_region_enter(b, col_loop.body);
    s_iv = col_loop.iv;
    for(i = 0; i < num_iargs; ++i)
    {
        new_accs[i] = col_loop.iter_vars[i];
    }

    /* The KH weights of filter column `s`.  Unlike the preload sibling these
     * cannot be hoisted into the prologue: `s` is a runtime value. */
    for(r = 0; r < KH; ++r)
    {
        static const char* const w_names[4] = {"k", "r", "s", "c"};
        rocke_value_t* w_vals[4];
        rocke_value_t* w_off = NULL;
        rocke_value_t* w_valid = NULL;
        w_vals[0] = ctx->ch;
        w_vals[1] = rocke_b_const_i32(b, r);
        w_vals[2] = s_iv;
        w_vals[3] = ctx->c0;
        rocke_transforms_descriptor_offset(b, ctx->b_desc, w_names, w_vals, 4, &w_off, &w_valid);
        w_col[r] = dwcol_load_elem(ctx, ctx->b_rsrc, dwcol_addr(ctx, w_off, ctx->ch_in_range));
    }

    /* Stream the input rows.  For each y only the taps whose output row is both
     * on-grid ((y - r) divisible by stride) and in range contribute; that set is
     * computed here, at emission time, so a strided kernel simply emits fewer
     * FMAs rather than any runtime predication. */
    for(y = 0; y < ctx->n_iters; ++y)
    {
        int ntaps = 0;
        rocke_value_t* y_i;
        for(r = 0; r < KH; ++r)
        {
            int d = y - r;
            int ho;
            if(d % S != 0)
            {
                continue;
            }
            ho = dwcol_floor_div(d, S);
            if(ho >= 0 && ho < BLOCK_H)
            {
                taps[ntaps++] = r;
            }
        }
        if(ntaps == 0)
        {
            continue;
        }
        y_i = rocke_b_add(b, ctx->y_start, rocke_b_const_i32(b, y));
        for(w_out = 0; w_out < BLOCK_W; ++w_out)
        {
            static const char* const a_names[5] = {"n", "y_iter", "wo", "s_off", "c"};
            rocke_value_t* a_vals[5];
            rocke_value_t* a_off = NULL;
            rocke_value_t* a_valid = NULL;
            rocke_value_t* w_pos;
            rocke_value_t* load_ok;
            rocke_value_t* a_f32;
            int t;

            w_pos = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, w_out));
            a_vals[0] = ctx->n;
            a_vals[1] = y_i;
            a_vals[2] = w_pos;
            a_vals[3] = s_iv;
            a_vals[4] = ctx->ch;
            rocke_transforms_descriptor_offset(
                b, ctx->a_desc, a_names, a_vals, 5, &a_off, &a_valid);
            load_ok = rocke_b_land(b, a_valid, ctx->ch_in_range);
            /* No zero-fill select: the buffer bounds check already returns 0. */
            a_f32 = dwcol_load_elem(ctx, ctx->a_rsrc, dwcol_addr(ctx, a_off, load_ok));
            for(t = 0; t < ntaps; ++t)
            {
                int rr = taps[t];
                int idx = dwcol_floor_div(y - rr, S) * BLOCK_W + w_out;
                new_accs[idx] = rocke_b_fma(b, w_col[rr], a_f32, new_accs[idx]);
            }
        }
    }

    rocke_b_scf_yield(b, new_accs, num_iargs);
    rocke_b_region_leave(b);
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }

    /* Drain the band once, after the column loop.  The last row tile of an
     * image may be partial, so the rows are bounded by p_Ho. */
    final_accs = col_loop.op->results;
    for(h_out = 0; h_out < BLOCK_H; ++h_out)
    {
        rocke_value_t* out_h = rocke_b_add(b, ctx->ho_start, rocke_b_const_i32(b, h_out));
        rocke_value_t* row_ok
            = rocke_b_land(b, rocke_b_cmp_lt(b, out_h, ctx->params.p_Ho), ctx->ch_in_range);
        for(w_out = 0; w_out < BLOCK_W; ++w_out)
        {
            static const char* const d_names[4] = {"n", "h", "w", "k"};
            rocke_value_t* d_vals[4];
            rocke_value_t* d_off = NULL;
            rocke_value_t* d_valid = NULL;
            rocke_value_t* out_q;
            rocke_value_t* store_ok;

            out_q = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, w_out));
            store_ok = rocke_b_land(b, row_ok, rocke_b_cmp_lt(b, out_q, ctx->c_W));
            d_vals[0] = ctx->n;
            d_vals[1] = out_h;
            d_vals[2] = out_q;
            d_vals[3] = ctx->ch;
            rocke_transforms_descriptor_offset(
                b, ctx->d_desc, d_names, d_vals, 4, &d_off, &d_valid);
            dwcol_store_elem(ctx,
                             ctx->d_rsrc,
                             dwcol_addr(ctx, d_off, store_ok),
                             final_accs[h_out * BLOCK_W + w_out]);
        }
    }

    return rocke_ir_builder_kernel(b);
}
