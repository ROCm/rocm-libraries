/* SPDX-License-Identifier: MIT
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * build_direct_depthwise_col -- column-streamed depthwise conv forward.
 *
 * Faithful C port of `build_direct_depthwise_col` in
 * library/kernels/common/conv_direct_grouped.py.  Byte-identity with the Python
 * engine is the repo's #1 invariant, so every op below is emitted in the same
 * order the Python builder emits it; the comments name the Python construct
 * each block mirrors.
 *
 * Shape of the algorithm: the KW axis is a runtime scf.for whose iter_args
 * carry the whole Ho x BLOCK_W accumulator band, while the input-row axis (y)
 * and the KH axis (r) are host-unrolled.  Because y and r are both host ints,
 * the strided liveness test `(y - r) % stride == 0` and the row index
 * `(y - r) / stride` resolve at emission time, so a strided kernel costs no
 * runtime arithmetic -- it simply emits fewer FMAs per y.
 *
 * Three deliberate differences from the preload sibling
 * (conv_direct_grouped_build_depthwise.cpp), each of which is byte-identity
 * critical rather than cosmetic:
 *
 *   1. n_iters is (Ho - 1) * stride + KH, NOT H + KH - 1.  The two agree at
 *      stride 1 with same-padding and diverge everywhere else.
 *   2. When the channel tile divides total_c exactly (or the W tile divides Wo
 *      exactly) the corresponding guard is absent from the IR entirely -- no
 *      select, not a constant-true select.  dwcol_addr() emits a bare mul in
 *      that case.
 *   3. No value-side zero-fill after a load.  The hardware bounds check already
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

/* Python `guard_ch(cond)`: conjoin the channel-range predicate, or pass the
 * incoming condition through untouched when the tile is exact. */
static rocke_value_t* dwcol_guard_ch(rocke_dconv_dwcol_ctx_t* ctx, rocke_value_t* cond)
{
    if(ctx->ch_in_range == NULL)
    {
        return cond;
    }
    if(cond == NULL)
    {
        return ctx->ch_in_range;
    }
    return rocke_b_land(ctx->b, cond, ctx->ch_in_range);
}

/* Python `addr(off, cond)`: element offset -> byte offset, poisoned to the OOB
 * sentinel when the access is not valid.  A NULL cond emits no select at all. */
static rocke_value_t* dwcol_addr(rocke_dconv_dwcol_ctx_t* ctx,
                                 rocke_value_t* off,
                                 rocke_value_t* cond)
{
    rocke_value_t* byte_off = rocke_b_mul(ctx->b, off, ctx->c_elem_bytes);
    if(cond == NULL)
    {
        return byte_off;
    }
    return rocke_b_select(ctx->b, cond, byte_off, ctx->oob_sentinel);
}

/* Python `load_elem`: one element, widened to f32.  f32 goes through the
 * dtype-generic tile.buffer_load with no convert. */
static rocke_value_t* dwcol_load_elem(rocke_dconv_dwcol_ctx_t* ctx,
                                      rocke_value_t* rsrc,
                                      rocke_value_t* byte_off)
{
    rocke_ir_builder_t* b = ctx->b;
    if(strcmp(ctx->DT->name, "f16") == 0)
    {
        return rocke_b_cast_to_f32(b, rocke_b_buffer_load_f16(b, rsrc, byte_off, ctx->c0));
    }
    if(strcmp(ctx->DT->name, "bf16") == 0)
    {
        return rocke_b_cast_to_f32(b, rocke_b_buffer_load_bf16(b, rsrc, byte_off, ctx->c0));
    }
    return rocke_b_buffer_load(b, rsrc, byte_off, ctx->c0, rocke_f32());
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
    else if(strcmp(ctx->DT->name, "bf16") == 0)
    {
        rocke_b_buffer_store_bf16(b, rsrc, byte_off, ctx->c0, rocke_b_trunc_f32_to_bf16(b, acc));
    }
    else
    {
        rocke_b_buffer_store_f32(b, rsrc, byte_off, ctx->c0, acc);
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
    ctx->BLOCK_W = spec->block_w;
    ctx->BLOCK_WAVES = spec->block_waves;
    ctx->WAVE = spec->wave_size;
    ctx->THREADS = rocke_direct_depthwise_col_threads_per_block(spec);
    ctx->BLOCK_CH = rocke_direct_depthwise_col_block_ch(spec);
    ctx->Ho = dwcol_floor_div(ctx->p.H + 2 * ctx->p.PAD - ctx->p.KH, ctx->p.stride) + 1;
    ctx->Wo = dwcol_floor_div(ctx->p.W + 2 * ctx->p.PAD - ctx->p.KW, ctx->p.stride) + 1;
    /* Python line: n_iters = (Ho - 1) * p.stride + p.KH.  Do NOT copy the
     * preload sibling's H + KH - 1 -- the two only agree at stride 1. */
    ctx->n_iters = (ctx->Ho - 1) * ctx->p.stride + ctx->p.KH;
    ctx->ch_tile_exact = rocke_direct_depthwise_col_ch_tile_exact(spec);
    ctx->w_tile_exact = rocke_direct_depthwise_col_w_tile_exact(spec);

    ctx->DT = rocke_fuse_dtype_to_ir_str(spec->dtype);
    if(ctx->DT == NULL)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
        }
        return false;
    }
    /* Python: ELEM_BYTES = {"f16": 2, "bf16": 2, "f32": 4}[DT.name] */
    ctx->ELEM_BYTES = (strcmp(ctx->DT->name, "f32") == 0) ? 4 : 2;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", ctx->THREADS);

    {
        const rocke_type_t* dptr = rocke_ptr_type(b, ctx->DT, "global");
        rocke_param_opts_t ro;
        rocke_param_opts_t wo;
        rocke_param_opts_t none;

        ro = (rocke_param_opts_t){0};
        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;
        ctx->A = rocke_b_param(b, "A", dptr, &ro);
        ctx->Bp = rocke_b_param(b, "B", dptr, &ro);

        wo = (rocke_param_opts_t){0};
        wo.noalias = true;
        wo.noalias_set = true;
        wo.writeonly = true;
        wo.writeonly_set = true;
        wo.align = 16;
        wo.align_set = true;
        ctx->D = rocke_b_param(b, "D", dptr, &wo);

        none = (rocke_param_opts_t){0};
        ctx->A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
        ctx->B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
        ctx->D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);
    }

    /* Constants in Python source order: c0, c1, c_wave, c_W, c_groups,
     * c_elem_bytes, oob_sentinel, zero_f32.  Note c1 is emitted SECOND here,
     * unlike the preload prologue which emits it later inside the group loop. */
    ctx->c0 = rocke_b_const_i32(b, 0);
    ctx->c1 = rocke_b_const_i32(b, 1);
    ctx->c_wave = rocke_b_const_i32(b, ctx->WAVE);
    ctx->c_W = rocke_b_const_i32(b, ctx->Wo);
    ctx->c_groups = rocke_b_const_i32(b, ctx->p.groups);
    ctx->c_elem_bytes = rocke_b_const_i32(b, ctx->ELEM_BYTES);
    ctx->oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx->zero_f32 = rocke_b_const_f32(b, 0.0);

    ctx->tid = rocke_b_thread_id_x(b);
    ctx->wave_id = rocke_b_div(b, ctx->tid, ctx->c_wave);
    ctx->lane = rocke_b_mod(b, ctx->tid, ctx->c_wave);

    ctx->bx = rocke_b_block_id_x(b);
    ctx->by = rocke_b_block_id_y(b);
    ctx->n = rocke_b_block_id_z(b);
    ctx->q_tile_start = rocke_b_mul(b, ctx->bx, rocke_b_const_i32(b, ctx->BLOCK_W));

    /* ch = by*BLOCK_CH + (wave_id*WAVE + lane) */
    {
        rocke_value_t* mul_by = rocke_b_mul(b, ctx->by, rocke_b_const_i32(b, ctx->BLOCK_CH));
        rocke_value_t* mul_wave = rocke_b_mul(b, ctx->wave_id, ctx->c_wave);
        rocke_value_t* inner = rocke_b_add(b, mul_wave, ctx->lane);
        ctx->ch = rocke_b_add(b, mul_by, inner);
    }
    /* Python: ch_in_range = None if spec.ch_tile_exact else b.cmp_lt(ch, c_groups).
     * NULL here means the predicate is never emitted, not that it is true. */
    ctx->ch_in_range = ctx->ch_tile_exact ? NULL : rocke_b_cmp_lt(b, ctx->ch, ctx->c_groups);

    ctx->a_rsrc = rocke_b_buffer_rsrc(b, ctx->A, ctx->A_bytes);
    ctx->b_rsrc = rocke_b_buffer_rsrc(b, ctx->Bp, ctx->B_bytes);
    ctx->d_rsrc = rocke_b_buffer_rsrc(b, ctx->D, ctx->D_bytes);

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  Descriptors: A gains the padded/strided embeds, B and D stay naive.
 * ===================================================================== */
void rocke_dconv_dwcol_build_descriptors(rocke_dconv_dwcol_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int total_c = rocke_direct_conv_problem_total_c(&ctx->p);
    int total_k = rocke_direct_conv_problem_total_k(&ctx->p);

    {
        static const char* const a_coords[4] = {"n", "h", "w", "c"};
        int a_lengths[4];
        rocke_tensor_descriptor_t* a_naive;
        const rocke_transform_t* xforms[2];

        a_lengths[0] = ctx->p.N;
        a_lengths[1] = ctx->p.H;
        a_lengths[2] = ctx->p.W;
        a_lengths[3] = total_c;
        a_naive = rocke_tensor_descriptor_naive(b, "A", a_lengths, 4, NULL, a_coords, 4);
        {
            /* h = y_iter - PAD, clamped to [0, H) */
            static const char* const h_upper[1] = {"y_iter"};
            int h_strides[1] = {1};
            xforms[0]
                = rocke_embed_bounded(b, h_upper, 1, "h", h_strides, -ctx->p.PAD, 0, ctx->p.H);
        }
        {
            /* w = wo*stride + s_off - PAD, clamped to [0, W) */
            static const char* const w_upper[2] = {"wo", "s_off"};
            int w_strides[2];
            w_strides[0] = ctx->p.stride;
            w_strides[1] = 1;
            xforms[1]
                = rocke_embed_bounded(b, w_upper, 2, "w", w_strides, -ctx->p.PAD, 0, ctx->p.W);
        }
        ctx->a_desc = rocke_tensor_descriptor_transform(b, a_naive, xforms, 2);
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
        static const char* const d_coords[4] = {"n", "h", "w", "k"};
        int d_lengths[4];
        d_lengths[0] = ctx->p.N;
        d_lengths[1] = ctx->Ho;
        d_lengths[2] = ctx->Wo;
        d_lengths[3] = total_k;
        ctx->d_desc = rocke_tensor_descriptor_naive(b, "D", d_lengths, 4, NULL, d_coords, 4);
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
    const int BLOCK_W = ctx->BLOCK_W;
    const int Ho = ctx->Ho;
    const int num_iargs = Ho * BLOCK_W;
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
     * The band is Ho x BLOCK_W and KH reaches 31+ in the supported space, so
     * these are alloca'd rather than fixed ctx arrays. */
    {
        char(*name_store)[32] = (char(*)[32])alloca((size_t)num_iargs * 32 * sizeof(char));
        int idx = 0;
        iargs = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
        new_accs = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
        for(h_out = 0; h_out < Ho; ++h_out)
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
            if(ho >= 0 && ho < Ho)
            {
                taps[ntaps++] = r;
            }
        }
        if(ntaps == 0)
        {
            continue;
        }
        y_i = rocke_b_const_i32(b, y);
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
            load_ok = dwcol_guard_ch(ctx, a_valid);
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

    /* Drain the band once, after the column loop. */
    final_accs = col_loop.op->results;
    for(h_out = 0; h_out < Ho; ++h_out)
    {
        rocke_value_t* out_h = rocke_b_const_i32(b, h_out);
        for(w_out = 0; w_out < BLOCK_W; ++w_out)
        {
            static const char* const d_names[4] = {"n", "h", "w", "k"};
            rocke_value_t* d_vals[4];
            rocke_value_t* d_off = NULL;
            rocke_value_t* d_valid = NULL;
            rocke_value_t* out_q;
            rocke_value_t* q_ok;
            rocke_value_t* store_ok;

            out_q = rocke_b_add(b, ctx->q_tile_start, rocke_b_const_i32(b, w_out));
            q_ok = ctx->w_tile_exact ? NULL : rocke_b_cmp_lt(b, out_q, ctx->c_W);
            store_ok = dwcol_guard_ch(ctx, q_ok);
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
