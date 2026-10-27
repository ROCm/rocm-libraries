// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_dgrad_stream.cpp -- C++ port of the
 * ho-streaming depthwise dgrad in library/kernels/common/conv_direct_grouped.py:
 * DirectDepthwiseDgradStreamSpec, is_valid_depthwise_dgrad_stream_spec,
 * build_direct_depthwise_dgrad_streaming and its channel-packed form
 * _build_direct_depthwise_dgrad_streaming_packed (cpl > 1).
 *
 * The kernel streams dY rows in a runtime loop, KH rows per iteration with a
 * few rows prefetched, into stride * KH circular dX accumulator slots per
 * column; each dX row is flushed once its last tap has landed. A second loop
 * only flushes the windows past the last dY row. With cpl > 1 every lane owns
 * cpl adjacent channels: dY and dX move as one cpl-wide vector per lane and
 * every tap is cpl / 2 packed <2 x f32> FMAs over channel pairs.
 *
 * Byte-identical to the Python source: every rocke_b_* call below is issued in
 * the order Python evaluates the matching expression (arguments left to
 * right, innermost first).
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_dw_dgrad.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

#include "rocke/error_boundary.hpp"

namespace
{

using vals_t = std::vector<rocke_value_t*>;

/* Python's a % m for m > 0. */
int dgs_pymod(int a, int m)
{
    int r = a % m;
    return r < 0 ? r + m : r;
}

void dgs_reason(char* reason, size_t reason_cap, const char* text)
{
    if(reason != NULL && reason_cap > 0)
        snprintf(reason, reason_cap, "%s", text);
}

void dgs_set_err(rocke_ir_builder_t* b)
{
    if(b->status == ROCKE_OK)
        b->status = ROCKE_ERR_VALUE;
}

/* pad_w_reason(p): only -1 (PAD) or a value of at least 0 is a column
 * padding. Returns true (and fills reason) when pad_w is neither. */
bool dgs_pad_w_reason(const rocke_direct_conv_problem_t* p,
                      const char* prefix,
                      char* reason,
                      size_t reason_cap)
{
    if(p->pad_w >= -1)
        return false;
    if(reason != NULL && reason_cap > 0)
        snprintf(reason,
                 reason_cap,
                 "%spad_w must be -1 (PAD) or a column padding of at least 0 (got %d)",
                 prefix,
                 p->pad_w);
    return true;
}

/* _dw_dgrad_stream_padding_reason(p): the row-streaming dgrad kernel offsets
 * columns by PAD, so it computes no per-axis padding. */
bool dgs_stream_padding_reason(const rocke_direct_conv_problem_t* p,
                               const char* prefix,
                               char* reason,
                               size_t reason_cap)
{
    if(dgs_pad_w_reason(p, prefix, reason, reason_cap))
        return true;
    if(rocke_direct_conv_problem_pad_w(p) != p->PAD)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "%sone padding for both axes (got PAD=%dx%d)",
                     prefix,
                     p->PAD,
                     rocke_direct_conv_problem_pad_w(p));
        return true;
    }
    return false;
}

bool dgs_is_bf16(const rocke_direct_conv_problem_t* p)
{
    return p->dtype != NULL && strcmp(p->dtype, "bf16") == 0;
}

/* Iteration arguments whose names outlive the scf.for they seed. */
struct dgs_iargs
{
    std::vector<std::string> names;
    vals_t inits;

    void add(std::string name, rocke_value_t* v)
    {
        names.push_back(std::move(name));
        inits.push_back(v);
    }

    rocke_for_t open(rocke_ir_builder_t* b,
                     rocke_value_t* lo,
                     rocke_value_t* hi,
                     rocke_value_t* step,
                     const char* iv_name) const
    {
        std::vector<rocke_iter_arg_t> args(names.size());
        for(size_t i = 0; i < names.size(); ++i)
        {
            args[i].name = names[i].c_str();
            args[i].init = inits[i];
        }
        /* Python scf_for_iter defaults: unroll=False, elide_trailing_barrier=True. */
        return rocke_b_scf_for_iter(
            b, lo, hi, step, args.data(), (int)args.size(), iv_name, false, true);
    }
};

/* The values both builders share, built in the order Python emits them up to
 * the loop-invariant column terms. */
struct dgs_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_conv_problem_t* p;
    rocke_dconv_params_t params;
    int BLOCK_W;
    int stride;
    int n_slots;
    bool is_bf16;
    rocke_value_t* c0;
    rocke_value_t* c_half;
    rocke_value_t* oob_sentinel;
    rocke_value_t* wi_tile_start;
    rocke_value_t* ch_ok;
    rocke_value_t* a_rsrc;
    rocke_value_t* b_rsrc;
    rocke_value_t* d_rsrc;
    rocke_value_t* c_stride_v;
    rocke_value_t* c_KH_v;
    rocke_value_t* ch_off;
    rocke_value_t* dy_n_off;
    rocke_value_t* dx_n_off;
    vals_t st_terms;
    vals_t col_terms;
};

/* Python: the st_terms loop. */
void dgs_store_terms(dgs_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    for(int j = 0; j < c.BLOCK_W; ++j)
    {
        rocke_value_t* wi = rocke_b_add(b, c.wi_tile_start, rocke_b_const_i32(b, j));
        rocke_value_t* lt = rocke_b_cmp_lt(b, wi, c.params.p_Wi);
        rocke_value_t* wi_ok = rocke_b_land(b, lt, c.ch_ok);
        rocke_value_t* m = rocke_b_mul(b, wi, c.params.p_D_stride_wo); /* p_dX_stride_wi */
        rocke_value_t* m2 = rocke_b_mul(b, m, c.c_half);
        rocke_value_t* wi_off = rocke_b_add(b, m2, c.ch_off);
        c.st_terms.push_back(rocke_b_select(b, wi_ok, wi_off, c.oob_sentinel));
    }
}

/* Python: the col_terms loop (one term per distinct dY column d = j - s). */
void dgs_col_terms(dgs_ctx& c)
{
    rocke_ir_builder_t* b = c.b;
    const int KW = c.p->KW;
    for(int d = 1 - KW; d < c.BLOCK_W; ++d)
    {
        rocke_value_t* wi_p_s = rocke_b_add(b, c.wi_tile_start, rocke_b_const_i32(b, c.p->PAD + d));
        rocke_value_t* div_ok = NULL;
        rocke_value_t* wo;
        if(c.stride > 1)
        {
            rocke_value_t* md = rocke_b_mod(b, wi_p_s, c.c_stride_v);
            div_ok = rocke_b_cmp_eq(b, md, c.c0);
            wo = rocke_b_div(b, wi_p_s, c.c_stride_v);
        }
        else
        {
            wo = wi_p_s;
        }
        rocke_value_t* ge = rocke_b_cmp_ge(b, wi_p_s, c.c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, wo, c.params.p_Wo);
        rocke_value_t* col_ok = rocke_b_land(b, ge, lt);
        if(c.stride > 1)
            col_ok = rocke_b_land(b, col_ok, div_ok);
        rocke_value_t* m = rocke_b_mul(b, wo, c.params.p_A_stride_wi); /* p_dY_stride_wo */
        rocke_value_t* col_off = rocke_b_mul(b, m, c.c_half);
        c.col_terms.push_back(rocke_b_select(b, col_ok, col_off, c.oob_sentinel));
    }
}

/* Python: the y-th dY row offset term and its bound, shared by load_row. */
rocke_value_t* dgs_row_off(dgs_ctx& c, rocke_value_t* y_v)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* m = rocke_b_mul(b, y_v, c.params.p_A_stride_hi); /* p_dY_stride_ho */
    rocke_value_t* a = rocke_b_add(b, c.dy_n_off, m);
    return rocke_b_mul(b, a, c.c_half);
}

/* Python: the per-d prologue of flush -- hi_ok and the dX row offset. */
void dgs_flush_row(
    dgs_ctx& c, rocke_value_t* y_v, int d, rocke_value_t** hi_ok_out, rocke_value_t** row_off_out)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* ys = rocke_b_mul(b, y_v, c.c_stride_v);
    rocke_value_t* hi_v = rocke_b_add(b, ys, rocke_b_const_i32(b, d - c.p->PAD));
    rocke_value_t* ge = rocke_b_cmp_ge(b, hi_v, c.c0);
    rocke_value_t* lt = rocke_b_cmp_lt(b, hi_v, c.params.p_Hi);
    *hi_ok_out = rocke_b_land(b, ge, lt);
    rocke_value_t* m = rocke_b_mul(b, hi_v, c.params.p_D_stride_ho); /* p_dX_stride_hi */
    rocke_value_t* a = rocke_b_add(b, c.dx_n_off, m);
    *row_off_out = rocke_b_mul(b, a, c.c_half);
}

/* Python: build_direct_depthwise_dgrad_streaming, cpl == 1 body. */
rocke_kernel_def_t* dgs_build_scalar(rocke_ir_builder_t* b,
                                     const rocke_direct_depthwise_dgrad_stream_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BLOCK_W = spec->block_w;
    const int WAVE = spec->wave_size;
    const int THREADS = spec->block_waves * WAVE;
    const int BLOCK_CH = spec->block_waves * WAVE;
    const int stride = p->stride;
    const int KH = p->KH;
    const int KW = p->KW;
    const int PAD = p->PAD;

    dgs_ctx c;
    c.b = b;
    c.p = p;
    c.BLOCK_W = BLOCK_W;
    c.stride = stride;
    c.n_slots = stride * KH;
    c.is_bf16 = dgs_is_bf16(p);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    rocke_dconv_emit_params(
        b, &c.params, "dgrad", rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16"));
    rocke_value_t* p_Hi = c.params.p_Hi;

    c.c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_groups = c.params.p_groups;
    c.c_half = rocke_b_const_i32(b, 2);
    rocke_value_t* c_wave = rocke_b_const_i32(b, WAVE);
    c.oob_sentinel = rocke_b_const_i32(b, (int32_t)(((int64_t)1 << 31) - 1));
    rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* wave_id = rocke_b_div(b, tid, c_wave);
    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);

    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* n = rocke_b_block_id_z(b);
    c.wi_tile_start = rocke_b_mul(b, bx, rocke_b_const_i32(b, BLOCK_W));
    rocke_value_t* ch;
    {
        rocke_value_t* outer = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK_CH));
        rocke_value_t* m = rocke_b_mul(b, wave_id, c_wave);
        rocke_value_t* inner = rocke_b_add(b, m, lane);
        ch = rocke_b_add(b, outer, inner);
    }
    c.ch_ok = rocke_b_cmp_lt(b, ch, c_groups);

    c.a_rsrc = rocke_b_buffer_rsrc(b, c.params.A, c.params.A_bytes);
    c.b_rsrc = rocke_b_buffer_rsrc(b, c.params.Bp, c.params.B_bytes);
    c.d_rsrc = rocke_b_buffer_rsrc(b, c.params.D, c.params.D_bytes);

    /* W descriptor: B[groups, KH, KW, 1] KRSC. */
    const rocke_tensor_descriptor_t* b_desc;
    {
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_len[4] = {p->groups, KH, KW, 1};
        b_desc = rocke_tensor_descriptor_naive(b, "B", b_len, 4, NULL, b_coords, 4);
        if(b_desc == NULL)
            return NULL;
    }

    /* Preload W[ch, r, s, 0] into f32 registers. */
    std::vector<vals_t> weights_f32((size_t)KH);
    for(int r_const = 0; r_const < KH; ++r_const)
    {
        for(int s_const = 0; s_const < KW; ++s_const)
        {
            const char* w_names[4] = {"k", "r", "s", "c"};
            rocke_value_t* w_vals[4];
            w_vals[0] = ch;
            w_vals[1] = rocke_b_const_i32(b, r_const);
            w_vals[2] = rocke_b_const_i32(b, s_const);
            w_vals[3] = c.c0;
            rocke_value_t* w_off = NULL;
            rocke_transforms_descriptor_offset(b, b_desc, w_names, w_vals, 4, &w_off, NULL);
            rocke_value_t* w_bytes = rocke_b_mul(b, w_off, c.c_half);
            rocke_value_t* safe_w = rocke_b_select(b, c.ch_ok, w_bytes, c.oob_sentinel);
            rocke_value_t* w_h = c.is_bf16 ? rocke_b_buffer_load_bf16(b, c.b_rsrc, safe_w, c.c0)
                                           : rocke_b_buffer_load_f16(b, c.b_rsrc, safe_w, c.c0);
            rocke_value_t* w_cast = rocke_b_cast_to_f32(b, w_h);
            weights_f32[(size_t)r_const].push_back(rocke_b_select(b, c.ch_ok, w_cast, zero_f32));
        }
    }

    rocke_value_t* c_Ho = c.params.p_Ho;
    c.c_stride_v = rocke_b_const_i32(b, stride);
    c.c_KH_v = rocke_b_const_i32(b, KH);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* y_end_v;
    {
        rocke_value_t* a1 = rocke_b_add(b, p_Hi, rocke_b_const_i32(b, PAD));
        rocke_value_t* a2 = rocke_b_add(b, a1, rocke_b_const_i32(b, stride - 1));
        y_end_v = rocke_b_div(b, a2, c.c_stride_v);
    }
    rocke_value_t* n_groups_v
        = rocke_b_div(b, rocke_b_add(b, y_end_v, rocke_b_const_i32(b, KH - 1)), c.c_KH_v);
    rocke_value_t* n_fma_groups_v
        = rocke_b_div(b, rocke_b_add(b, c_Ho, rocke_b_const_i32(b, KH - 1)), c.c_KH_v);

    const int n_acc = c.n_slots * BLOCK_W;
    c.ch_off = rocke_b_mul(b, ch, c.c_half);
    dgs_store_terms(c);
    dgs_col_terms(c);
    const int n_cols = (int)c.col_terms.size();
    c.dy_n_off = rocke_b_mul(b, n, c.params.p_A_stride_n); /* p_dY_stride_n */
    c.dx_n_off = rocke_b_mul(b, n, c.params.p_D_stride_n); /* p_dX_stride_n */

    /* Python: load_row(y_v) -- raw dY values of row y_v, one per column. */
    auto load_row = [&](rocke_value_t* y_v, vals_t& out) {
        rocke_value_t* ho_ok = rocke_b_cmp_lt(b, y_v, c_Ho);
        rocke_value_t* row_off = dgs_row_off(c, y_v);
        for(rocke_value_t* col_term : c.col_terms)
        {
            rocke_value_t* rc = rocke_b_add(b, row_off, col_term);
            rocke_value_t* s_term = rocke_b_select(b, ho_ok, rc, c.oob_sentinel);
            rocke_value_t* v_off = rocke_b_add(b, c.ch_off, s_term);
            out.push_back(c.is_bf16 ? rocke_b_buffer_load_bf16(b, c.a_rsrc, v_off, c.c0)
                                    : rocke_b_buffer_load_f16(b, c.a_rsrc, v_off, c.c0));
        }
    };
    /* Python: flush(y_v, jj, slots_flat). */
    auto flush = [&](rocke_value_t* y_v, int jj, vals_t& slots) {
        for(int d = 0; d < stride; ++d)
        {
            const int slot = dgs_pymod(jj * stride - PAD + d, c.n_slots);
            rocke_value_t* hi_ok;
            rocke_value_t* row_off;
            dgs_flush_row(c, y_v, d, &hi_ok, &row_off);
            for(int j = 0; j < BLOCK_W; ++j)
            {
                const int flat = slot * BLOCK_W + j;
                rocke_value_t* rc = rocke_b_add(b, row_off, c.st_terms[(size_t)j]);
                rocke_value_t* safe_d = rocke_b_select(b, hi_ok, rc, c.oob_sentinel);
                if(c.is_bf16)
                    rocke_b_buffer_store_bf16(b,
                                              c.d_rsrc,
                                              safe_d,
                                              c.c0,
                                              rocke_b_trunc_f32_to_bf16(b, slots[(size_t)flat]));
                else
                    rocke_b_buffer_store_f16(b,
                                             c.d_rsrc,
                                             safe_d,
                                             c.c0,
                                             rocke_b_trunc_f32_to_f16(b, slots[(size_t)flat]));
                slots[(size_t)flat] = zero_f32;
            }
        }
    };

    /* dY rows software-pipelined PF rows ahead. */
    const int PF = KH < (16 + n_cols - 1) / n_cols ? KH : (16 + n_cols - 1) / n_cols;
    dgs_iargs fma_args;
    for(int slot = 0; slot < c.n_slots; ++slot)
        for(int j = 0; j < BLOCK_W; ++j)
            fma_args.add("dgs_acc_s" + std::to_string(slot) + "_w" + std::to_string(j), zero_f32);
    for(int k = 0; k < PF; ++k)
    {
        vals_t row;
        load_row(rocke_b_const_i32(b, k), row);
        for(int d = 0; d < n_cols; ++d)
            fma_args.add("dgs_dy_r" + std::to_string(k) + "_d" + std::to_string(d), row[(size_t)d]);
    }

    rocke_for_t fma_loop = fma_args.open(b, c.c0, n_fma_groups_v, c1, "dgs_grp");
    rocke_b_region_enter(b, fma_loop.body);
    {
        vals_t slots(fma_loop.iter_vars, fma_loop.iter_vars + n_acc);
        std::vector<vals_t> window;
        for(int k = 0; k < PF; ++k)
            window.emplace_back(fma_loop.iter_vars + n_acc + k * n_cols,
                                fma_loop.iter_vars + n_acc + (k + 1) * n_cols);
        rocke_value_t* y_base = rocke_b_mul(b, fma_loop.iv, c.c_KH_v);

        for(int jj = 0; jj < KH; ++jj)
        {
            rocke_value_t* y_v = rocke_b_add(b, y_base, rocke_b_const_i32(b, jj));
            {
                vals_t next;
                load_row(rocke_b_add(b, y_v, rocke_b_const_i32(b, PF)), next);
                window.push_back(next);
            }
            vals_t dy_row;
            for(rocke_value_t* dy_h : window.front())
                dy_row.push_back(rocke_b_cast_to_f32(b, dy_h));
            window.erase(window.begin());

            for(int r_const = 0; r_const < KH; ++r_const)
            {
                const int slot = dgs_pymod(jj * stride + r_const - PAD, c.n_slots);
                for(int s_const = 0; s_const < KW; ++s_const)
                {
                    for(int j = 0; j < BLOCK_W; ++j)
                    {
                        rocke_value_t* dy_f32 = dy_row[(size_t)(j - s_const + KW - 1)];
                        const size_t flat = (size_t)(slot * BLOCK_W + j);
                        slots[flat] = rocke_b_fma(
                            b, weights_f32[(size_t)r_const][(size_t)s_const], dy_f32, slots[flat]);
                    }
                }
            }
            flush(y_v, jj, slots);
        }

        /* Keep LLVM's SLP vectorizer from pairing the loop-carried FMA chains
         * into v_pk_fma_f32 (an empty asm, no instruction). */
        {
            const rocke_type_t* f32_ty = rocke_f32();
            rocke_inline_asm_opts_t opts;
            memset(&opts, 0, sizeof opts);
            opts.sideeffect = false;
            opts.sideeffect_set = true;
            for(size_t i = 0; i < slots.size(); ++i)
            {
                rocke_op_t* asm_op
                    = rocke_b_inline_asm(b, "", "=v,0", &slots[i], 1, &f32_ty, 1, &opts);
                slots[i] = asm_op ? asm_op->results[0] : NULL;
            }
        }
        vals_t yields(slots);
        for(const vals_t& row : window)
            yields.insert(yields.end(), row.begin(), row.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);

    /* The windows past the last dY row: no tap, only dX rows to complete. */
    dgs_iargs tail_args;
    for(int i = 0; i < n_acc; ++i)
        tail_args.add("dgs_tail_acc_" + std::to_string(i), fma_loop.op->results[i]);
    rocke_for_t flush_loop = tail_args.open(b, n_fma_groups_v, n_groups_v, c1, "dgs_tail");
    rocke_b_region_enter(b, flush_loop.body);
    {
        vals_t slots(flush_loop.iter_vars, flush_loop.iter_vars + n_acc);
        rocke_value_t* y_base = rocke_b_mul(b, flush_loop.iv, c.c_KH_v);
        for(int jj = 0; jj < KH; ++jj)
            flush(rocke_b_add(b, y_base, rocke_b_const_i32(b, jj)), jj, slots);
        rocke_b_scf_yield(b, slots.data(), (int)slots.size());
    }
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}

/* Python: _build_direct_depthwise_dgrad_streaming_packed (cpl > 1). */
rocke_kernel_def_t* dgs_build_packed(rocke_ir_builder_t* b,
                                     const rocke_direct_depthwise_dgrad_stream_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BLOCK_W = spec->block_w;
    const int WAVE = spec->wave_size;
    const int CPL = spec->cpl;
    const int NP = CPL / 2;
    const int THREADS = spec->block_waves * WAVE;
    const int stride = p->stride;
    const int KH = p->KH;
    const int KW = p->KW;
    const int PAD = p->PAD;

    dgs_ctx c;
    c.b = b;
    c.p = p;
    c.BLOCK_W = BLOCK_W;
    c.stride = stride;
    c.n_slots = stride * KH;
    c.is_bf16 = dgs_is_bf16(p);
    const rocke_type_t* elem = c.is_bf16 ? rocke_bf16() : rocke_f16();

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    rocke_dconv_emit_params(
        b, &c.params, "dgrad", rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16"));
    rocke_value_t* p_Hi = c.params.p_Hi;
    rocke_value_t* c_Ho = c.params.p_Ho;

    c.c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_groups = c.params.p_groups;
    c.c_half = rocke_b_const_i32(b, 2);
    (void)rocke_b_const_i32(b, WAVE); /* Python c_wave: emitted, unused */
    c.oob_sentinel = rocke_b_const_i32(b, (int32_t)(((int64_t)1 << 31) - 1));
    rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0);
    rocke_value_t* zero2;
    {
        rocke_value_t* z[2] = {zero_f32, zero_f32};
        zero2 = rocke_b_vec_pack(b, z, 2, rocke_f32());
    }

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* n = rocke_b_block_id_z(b);
    c.wi_tile_start = rocke_b_mul(b, bx, rocke_b_const_i32(b, BLOCK_W));
    rocke_value_t* ch;
    {
        rocke_value_t* m = rocke_b_mul(b, by, rocke_b_const_i32(b, THREADS));
        rocke_value_t* a = rocke_b_add(b, m, tid);
        ch = rocke_b_mul(b, a, rocke_b_const_i32(b, CPL));
    }
    c.ch_ok = rocke_b_cmp_lt(b, ch, c_groups);

    c.a_rsrc = rocke_b_buffer_rsrc(b, c.params.A, c.params.A_bytes);
    c.b_rsrc = rocke_b_buffer_rsrc(b, c.params.Bp, c.params.B_bytes);
    c.d_rsrc = rocke_b_buffer_rsrc(b, c.params.D, c.params.D_bytes);

    auto load1 = [&](rocke_value_t* rsrc, rocke_value_t* off) {
        return c.is_bf16 ? rocke_b_buffer_load_bf16(b, rsrc, off, c.c0)
                         : rocke_b_buffer_load_f16(b, rsrc, off, c.c0);
    };

    /* Weights W[ch + i, r, s]: per pair, a <2 x f32> per tap. */
    rocke_value_t* w_ch_off;
    {
        rocke_value_t* m = rocke_b_mul(b, ch, rocke_b_const_i32(b, KH * KW));
        w_ch_off = rocke_b_mul(b, m, c.c_half);
    }
    /* weights[r][s][k] */
    std::vector<std::vector<vals_t>> weights((size_t)KH, std::vector<vals_t>((size_t)KW));
    for(int r_const = 0; r_const < KH; ++r_const)
    {
        for(int s_const = 0; s_const < KW; ++s_const)
        {
            vals_t vals;
            for(int i = 0; i < CPL; ++i)
            {
                rocke_value_t* c_off
                    = rocke_b_const_i32(b, 2 * (i * KH * KW + r_const * KW + s_const));
                rocke_value_t* off = rocke_b_add(b, w_ch_off, c_off);
                rocke_value_t* safe = rocke_b_select(b, c.ch_ok, off, c.oob_sentinel);
                vals.push_back(rocke_b_cast_to_f32(b, load1(c.b_rsrc, safe)));
            }
            for(int k = 0; k < NP; ++k)
                weights[(size_t)r_const][(size_t)s_const].push_back(
                    rocke_b_vec_pack(b, &vals[(size_t)(2 * k)], 2, rocke_f32()));
        }
    }

    rocke_value_t* y_end_v;
    {
        rocke_value_t* a1 = rocke_b_add(b, p_Hi, rocke_b_const_i32(b, PAD));
        rocke_value_t* a2 = rocke_b_add(b, a1, rocke_b_const_i32(b, stride - 1));
        y_end_v = rocke_b_div(b, a2, rocke_b_const_i32(b, stride));
    }
    c.c_KH_v = rocke_b_const_i32(b, KH);
    c.c_stride_v = rocke_b_const_i32(b, stride);
    rocke_value_t* n_groups_v
        = rocke_b_div(b, rocke_b_add(b, y_end_v, rocke_b_const_i32(b, KH - 1)), c.c_KH_v);
    rocke_value_t* n_fma_groups_v
        = rocke_b_div(b, rocke_b_add(b, c_Ho, rocke_b_const_i32(b, KH - 1)), c.c_KH_v);

    c.ch_off = rocke_b_mul(b, ch, c.c_half);
    dgs_store_terms(c);
    dgs_col_terms(c);
    const int n_cols = (int)c.col_terms.size();
    c.dy_n_off = rocke_b_mul(b, n, c.params.p_A_stride_n); /* p_dY_stride_n */
    c.dx_n_off = rocke_b_mul(b, n, c.params.p_D_stride_n); /* p_dX_stride_n */

    /* Python: load_row(y_v) -- raw <CPL x elem> dY vectors, one per column. */
    auto load_row = [&](rocke_value_t* y_v, vals_t& out) {
        rocke_value_t* ho_ok = rocke_b_cmp_lt(b, y_v, c_Ho);
        rocke_value_t* row_off = dgs_row_off(c, y_v);
        for(rocke_value_t* col_term : c.col_terms)
        {
            rocke_value_t* rc = rocke_b_add(b, row_off, col_term);
            rocke_value_t* s_term = rocke_b_select(b, ho_ok, rc, c.oob_sentinel);
            rocke_value_t* v_off = rocke_b_add(b, c.ch_off, s_term);
            out.push_back(rocke_b_buffer_load_vN(b, c.a_rsrc, v_off, c.c0, elem, CPL));
        }
    };
    /* Python: to_pairs(v). */
    auto to_pairs = [&](rocke_value_t* v) {
        vals_t xs;
        for(int i = 0; i < CPL; ++i)
            xs.push_back(rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, v, i)));
        vals_t pairs;
        for(int k = 0; k < NP; ++k)
            pairs.push_back(rocke_b_vec_pack(b, &xs[(size_t)(2 * k)], 2, rocke_f32()));
        return pairs;
    };
    /* Python: store(off, pairs). */
    auto store = [&](rocke_value_t* off, const vals_t& pairs) {
        vals_t halves;
        for(rocke_value_t* v : pairs)
            halves.push_back(rocke_b_vec_cast_f32_to(b, v, elem));
        while(halves.size() > 1)
        {
            vals_t next;
            for(size_t i = 0; i < halves.size(); i += 2)
                next.push_back(rocke_b_vec_concat(b, halves[i], halves[i + 1]));
            halves.swap(next);
        }
        if(c.is_bf16)
            rocke_b_buffer_store_vN_bf16(b, c.d_rsrc, off, c.c0, halves[0], CPL / 2);
        else
            rocke_b_buffer_store_vN_f16(b, c.d_rsrc, off, c.c0, halves[0], CPL / 2);
    };
    /* Python: flush(y_v, jj, slots); slots[f] holds NP pairs. */
    auto flush = [&](rocke_value_t* y_v, int jj, std::vector<vals_t>& slots) {
        for(int d = 0; d < stride; ++d)
        {
            const int slot = dgs_pymod(jj * stride - PAD + d, c.n_slots);
            rocke_value_t* hi_ok;
            rocke_value_t* row_off;
            dgs_flush_row(c, y_v, d, &hi_ok, &row_off);
            for(int j = 0; j < BLOCK_W; ++j)
            {
                const size_t flat = (size_t)(slot * BLOCK_W + j);
                rocke_value_t* rc = rocke_b_add(b, row_off, c.st_terms[(size_t)j]);
                rocke_value_t* safe_d = rocke_b_select(b, hi_ok, rc, c.oob_sentinel);
                store(safe_d, slots[flat]);
                slots[flat] = vals_t((size_t)NP, zero2);
            }
        }
    };

    const int PF = KH < (16 + n_cols - 1) / n_cols ? KH : (16 + n_cols - 1) / n_cols;
    const int n_flat = c.n_slots * BLOCK_W;
    dgs_iargs fma_args;
    for(int f = 0; f < n_flat; ++f)
        for(int k = 0; k < NP; ++k)
            fma_args.add("dgp_acc_" + std::to_string(f) + "_" + std::to_string(k), zero2);
    for(int k = 0; k < PF; ++k)
    {
        vals_t row;
        load_row(rocke_b_const_i32(b, k), row);
        for(int d = 0; d < n_cols; ++d)
            fma_args.add("dgp_dy_r" + std::to_string(k) + "_d" + std::to_string(d), row[(size_t)d]);
    }
    const int n_acc = n_flat * NP;

    auto unflat = [&](rocke_value_t* const* vals) {
        std::vector<vals_t> out;
        for(int f = 0; f < n_flat; ++f)
            out.emplace_back(vals + f * NP, vals + (f + 1) * NP);
        return out;
    };

    rocke_for_t fma_loop = fma_args.open(b, c.c0, n_fma_groups_v, c1, "dgp_grp");
    rocke_b_region_enter(b, fma_loop.body);
    {
        std::vector<vals_t> slots = unflat(fma_loop.iter_vars);
        std::vector<vals_t> window;
        for(int k = 0; k < PF; ++k)
            window.emplace_back(fma_loop.iter_vars + n_acc + k * n_cols,
                                fma_loop.iter_vars + n_acc + (k + 1) * n_cols);
        rocke_value_t* y_base = rocke_b_mul(b, fma_loop.iv, c.c_KH_v);
        for(int jj = 0; jj < KH; ++jj)
        {
            rocke_value_t* y_v = rocke_b_add(b, y_base, rocke_b_const_i32(b, jj));
            {
                vals_t next;
                load_row(rocke_b_add(b, y_v, rocke_b_const_i32(b, PF)), next);
                window.push_back(next);
            }
            std::vector<vals_t> dy_row;
            for(rocke_value_t* v : window.front())
                dy_row.push_back(to_pairs(v));
            window.erase(window.begin());
            for(int r_const = 0; r_const < KH; ++r_const)
            {
                const int slot = dgs_pymod(jj * stride + r_const - PAD, c.n_slots);
                for(int s_const = 0; s_const < KW; ++s_const)
                {
                    for(int j = 0; j < BLOCK_W; ++j)
                    {
                        const vals_t& dy = dy_row[(size_t)(j - s_const + KW - 1)];
                        const size_t flat = (size_t)(slot * BLOCK_W + j);
                        const vals_t& w = weights[(size_t)r_const][(size_t)s_const];
                        vals_t acc = slots[flat];
                        for(int k = 0; k < NP; ++k)
                            acc[(size_t)k] = rocke_b_vector_fma(
                                b, w[(size_t)k], dy[(size_t)k], acc[(size_t)k]);
                        slots[flat] = acc;
                    }
                }
            }
            flush(y_v, jj, slots);
        }
        vals_t yields;
        for(const vals_t& sl : slots)
            yields.insert(yields.end(), sl.begin(), sl.end());
        for(const vals_t& row : window)
            yields.insert(yields.end(), row.begin(), row.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);

    dgs_iargs tail_args;
    for(int i = 0; i < n_acc; ++i)
        tail_args.add("dgp_tail_" + std::to_string(i), fma_loop.op->results[i]);
    rocke_for_t flush_loop = tail_args.open(b, n_fma_groups_v, n_groups_v, c1, "dgp_tail");
    rocke_b_region_enter(b, flush_loop.body);
    {
        std::vector<vals_t> slots = unflat(flush_loop.iter_vars);
        rocke_value_t* y_base = rocke_b_mul(b, flush_loop.iv, c.c_KH_v);
        for(int jj = 0; jj < KH; ++jj)
            flush(rocke_b_add(b, y_base, rocke_b_const_i32(b, jj)), jj, slots);
        vals_t yields;
        for(const vals_t& sl : slots)
            yields.insert(yields.end(), sl.begin(), sl.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}

} // namespace

/* ===================================================================== *
 *  DirectDepthwiseDgradStreamSpec
 * ===================================================================== */

rocke_direct_depthwise_dgrad_stream_spec_t rocke_direct_depthwise_dgrad_stream_spec_default(void)
{
    rocke_direct_depthwise_dgrad_stream_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_dgrad_stream";
    spec.block_w = 8;
    spec.block_waves = 1;
    spec.wave_size = 64;
    spec.cpl = 1;
    return spec;
}

int rocke_direct_depthwise_dgrad_stream_threads_per_block(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_dgrad_stream_block_ch(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec)
{
    return spec->block_waves * spec->wave_size * spec->cpl;
}

rocke_status_t rocke_direct_depthwise_dgrad_stream_kernel_name(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    const char* parts[4];
    char bw_buf[24];
    char bwv_buf[24];
    char cpl_buf[24];
    int n_parts = 3;

    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(bwv_buf, sizeof(bwv_buf), "bw%dwv", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = bw_buf;
    parts[2] = bwv_buf;
    if(spec->cpl > 1)
    {
        snprintf(cpl_buf, sizeof(cpl_buf), "cpl%d", spec->cpl);
        parts[n_parts++] = cpl_buf;
    }
    {
        const char* flag_names[1] = {"bf16"};
        int flag_on[1];
        flag_on[0] = dgs_is_bf16(&spec->problem) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, n_parts, flag_names, flag_on, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_depthwise_dgrad_stream_validate(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseDgradStreamSpec: unsupported dtype '%s'; expected fp16 "
                         "or bf16",
                         dt);
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseDgradStreamSpec requires cpg=kpg=1 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        return ROCKE_ERR_VALUE;
    }
    if(dgs_stream_padding_reason(p, "DirectDepthwiseDgradStreamSpec: ", reason, reason_cap))
        return ROCKE_ERR_VALUE;
    return ROCKE_OK;
}

bool rocke_direct_depthwise_dgrad_stream_is_valid_spec(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        dgs_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    /* dilation_reason(p) */
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "this direct conv kernel has no dilation (got %dx%d); only the non-grouped "
                     "forward kernel computes a dilated convolution",
                     p->dil_h,
                     p->dil_w);
        return false;
    }
    if(dgs_stream_padding_reason(p, "", reason, reason_cap))
        return false;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseDgradStreamSpec: unsupported dtype '%s'; expected fp16 "
                         "or bf16",
                         dt);
            return false;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
        return false;
    }
    if(spec->cpl != 1 && spec->cpl != 2 && spec->cpl != 4 && spec->cpl != 8)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "cpl must be 1, 2, 4 or 8 (got %d)", spec->cpl);
        return false;
    }
    if(p->groups % spec->cpl)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "%d channels are not a multiple of cpl=%d",
                     p->groups,
                     spec->cpl);
        return false;
    }
    if(spec->cpl > 1)
    {
        const long taps = (long)spec->cpl * p->KH * p->KW;
        const long acc = (long)spec->cpl * spec->block_w * p->KH;
        if(taps > ROCKE_DCONV_DW_DGRAD_STREAM_MAX_PRELOAD_TAPS
           || acc > ROCKE_DCONV_DW_DGRAD_CPL_MAX_ACC)
        {
            /* Hard ceilings on the packed weights and accumulator slots; the
             * AOT grid gates the specs within them that still spill. */
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "cpl=%d holds %ld weights and %ld accumulator-rows per lane "
                         "(max %d / %d)",
                         spec->cpl,
                         taps,
                         acc,
                         ROCKE_DCONV_DW_DGRAD_STREAM_MAX_PRELOAD_TAPS,
                         ROCKE_DCONV_DW_DGRAD_CPL_MAX_ACC);
            return false;
        }
    }
    dgs_reason(reason, reason_cap, "ok");
    return true;
}

rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_streaming(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_stream_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_direct_depthwise_dgrad_stream_validate(spec, reason, sizeof reason) != ROCKE_OK
       || !rocke_direct_depthwise_dgrad_stream_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        dgs_set_err(b);
        return NULL;
    }
    if(spec->cpl > 1)
        return dgs_build_packed(b, spec);
    return dgs_build_scalar(b, spec);
}

rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_streaming_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_stream_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_direct_depthwise_dgrad_stream_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_depthwise_dgrad_streaming(b, spec, arch);
    });
}

rocke_status_t rocke_direct_depthwise_dgrad_stream_lower_to_llvm(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    /* Zeroed so the failure path below may read and free it even when the
     * builder never reached rocke_ir_builder_init (e.g. the name overflowed). */
    rocke_ir_builder_t b = {};
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        dgs_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx950";
    kernel = rocke_build_direct_depthwise_dgrad_streaming_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        dgs_reason(err,
                   err_cap,
                   (m != NULL && m[0] != '\0') ? m
                                               : "build_direct_depthwise_dgrad_streaming failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  Per-axis padding and the DirectDepthwiseTiledSpec dgrad mode
 * ===================================================================== */

int rocke_direct_conv_problem_pad_w(const rocke_direct_conv_problem_t* p)
{
    return p->pad_w < 0 ? p->PAD : p->pad_w;
}

void rocke_direct_depthwise_tiled_out_hw(const rocke_direct_depthwise_tiled_spec_t* spec,
                                         int* out_h,
                                         int* out_w)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    if(spec->dgrad)
    {
        *out_h = p->H;
        *out_w = p->W;
        return;
    }
    /* DirectConvProblem.Ho / .Wo (floored; the stride is positive). */
    {
        const int eh = p->H + 2 * p->PAD - (p->KH - 1) * p->dil_h - 1;
        const int ew = p->W + 2 * rocke_direct_conv_problem_pad_w(p) - (p->KW - 1) * p->dil_w - 1;
        *out_h = (eh >= 0 ? eh / p->stride : -((-eh + p->stride - 1) / p->stride)) + 1;
        *out_w = (ew >= 0 ? ew / p->stride : -((-ew + p->stride - 1) / p->stride)) + 1;
    }
}

bool rocke_dconv_dw_tiled_padding_reason(const rocke_direct_depthwise_tiled_spec_t* spec,
                                         char* reason,
                                         size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int pad_w = rocke_direct_conv_problem_pad_w(p);
    if(dgs_pad_w_reason(p, "DirectDepthwiseTiledSpec: ", reason, reason_cap))
        return true;
    if(spec->dgrad && (p->stride != 1 || p->PAD > p->KH - 1 || pad_w > p->KW - 1))
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseTiledSpec(dgrad=True) needs stride 1 and padding below the "
                     "filter (got stride=%d, %dx%d, PAD=%dx%d)",
                     p->stride,
                     p->KH,
                     p->KW,
                     p->PAD,
                     pad_w);
        return true;
    }
    if(!spec->dgrad && pad_w != p->PAD)
    {
        dgs_reason(reason, reason_cap, "DirectDepthwiseTiledSpec: one padding for both axes");
        return true;
    }
    return false;
}
