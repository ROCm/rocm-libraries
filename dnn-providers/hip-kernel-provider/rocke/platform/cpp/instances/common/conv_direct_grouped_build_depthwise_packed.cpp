// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_packed.cpp -- C++ port of the
 * channel-packed depthwise forward kernels (cpl > 1) in
 * library/kernels/common/conv_direct_grouped.py:
 * _build_direct_depthwise_packed (row-streaming, DirectDepthwiseSpec) and
 * _build_direct_depthwise_tiled_packed (output-stationary,
 * DirectDepthwiseTiledSpec), with their shared _dw_packed_helpers.
 *
 * Each lane owns `cpl` adjacent channels: input and output move as one
 * cpl-wide vector per lane and every tap is cpl / 2 packed <2 x f32> FMAs
 * over channel pairs.
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

#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

namespace
{

using vals_t = std::vector<rocke_value_t*>;

/* Python: _dw_packed_helpers(b, p, cpl) -- load1 / loadv / to_pairs / store. */
struct dwp_helpers
{
    rocke_ir_builder_t* b;
    bool is_bf16;
    const rocke_type_t* elem;
    int cpl;
    rocke_value_t* c0;

    dwp_helpers(rocke_ir_builder_t* b_, const rocke_direct_conv_problem_t* p, int cpl_)
        : b(b_)
        , cpl(cpl_)
    {
        is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0);
        elem = is_bf16 ? rocke_bf16() : rocke_f16();
        c0 = rocke_b_const_i32(b, 0);
    }

    /* One 16-bit element. */
    rocke_value_t* load1(rocke_value_t* rsrc, rocke_value_t* off) const
    {
        return is_bf16 ? rocke_b_buffer_load_bf16(b, rsrc, off, c0)
                       : rocke_b_buffer_load_f16(b, rsrc, off, c0);
    }

    /* A cpl-wide vector. */
    rocke_value_t* loadv(rocke_value_t* rsrc, rocke_value_t* off) const
    {
        return rocke_b_buffer_load_vN(b, rsrc, off, c0, elem, cpl);
    }

    /* A raw vector as cpl / 2 <2 x f32> pairs, appended to out. */
    void to_pairs(rocke_value_t* v, vals_t& out) const
    {
        vals_t xs;
        for(int i = 0; i < cpl; ++i)
        {
            rocke_value_t* e = rocke_b_vec_extract(b, v, i);
            xs.push_back(rocke_b_cast_to_f32(b, e));
        }
        for(int k = 0; k < cpl / 2; ++k)
            out.push_back(rocke_b_vec_pack(b, &xs[2 * k], 2, rocke_f32()));
    }

    /* Pairs back to one cpl-wide 16-bit vector store. */
    void store(rocke_value_t* rsrc, rocke_value_t* off, rocke_value_t* const* pairs) const
    {
        vals_t halves;
        for(int k = 0; k < cpl / 2; ++k)
            halves.push_back(rocke_b_vec_cast_f32_to(b, pairs[k], elem));
        while(halves.size() > 1)
        {
            vals_t next;
            for(size_t i = 0; i < halves.size(); i += 2)
                next.push_back(rocke_b_vec_concat(b, halves[i], halves[i + 1]));
            halves.swap(next);
        }
        if(is_bf16)
            rocke_b_buffer_store_vN_bf16(b, rsrc, off, c0, halves[0], cpl / 2);
        else
            rocke_b_buffer_store_vN_f16(b, rsrc, off, c0, halves[0], cpl / 2);
    }
};

rocke_value_t* dwp_zero2(rocke_ir_builder_t* b)
{
    rocke_value_t* z[2];
    z[0] = rocke_b_const_f32(b, 0.0);
    z[1] = rocke_b_const_f32(b, 0.0);
    return rocke_b_vec_pack(b, z, 2, rocke_f32());
}

/* d_desc.offset(b, n=, h=, w=, k=). */
rocke_value_t* dwp_d_offset(rocke_ir_builder_t* b,
                            const rocke_tensor_descriptor_t* d_desc,
                            rocke_value_t* n,
                            rocke_value_t* h,
                            rocke_value_t* w,
                            rocke_value_t* k)
{
    const char* names[4] = {"n", "h", "w", "k"};
    rocke_value_t* vals[4] = {n, h, w, k};
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    rocke_transforms_descriptor_offset(b, d_desc, names, vals, 4, &off, &valid);
    return off;
}

/* ch = (by * THREADS + tid) * CPL. */
rocke_value_t*
    dwp_channel(rocke_ir_builder_t* b, rocke_value_t* by, rocke_value_t* tid, int threads, int cpl)
{
    rocke_value_t* c_threads = rocke_b_const_i32(b, threads);
    rocke_value_t* m = rocke_b_mul(b, by, c_threads);
    rocke_value_t* a = rocke_b_add(b, m, tid);
    rocke_value_t* c_cpl = rocke_b_const_i32(b, cpl);
    return rocke_b_mul(b, a, c_cpl);
}

/* Shared per-kernel state of the two builders (the Python closures' captures). */
struct dwp_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dconv_params_t* params;
    const dwp_helpers* h;
    int PAD;
    rocke_value_t* c0;
    rocke_value_t* c_half;
    rocke_value_t* oob_sentinel;
    rocke_value_t* a_rsrc;
    rocke_value_t* ch_off;
    rocke_value_t* n_off;
    vals_t col_terms;
};

/* Column terms of the input row: the wave-uniform offset per column, the OOB
 * sentinel off the image. wi_tile = q * stride_v - PAD. */
void dwp_col_terms(dwp_ctx& c,
                   rocke_value_t* q,
                   rocke_value_t* stride_v,
                   const std::vector<int>& cols)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* q_s = rocke_b_mul(b, q, stride_v);
    rocke_value_t* wi_tile = rocke_b_add(b, q_s, rocke_b_const_i32(b, -c.PAD));
    for(int col : cols)
    {
        rocke_value_t* wi = rocke_b_add(b, wi_tile, rocke_b_const_i32(b, col));
        rocke_value_t* ge = rocke_b_cmp_ge(b, wi, c.c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, wi, c.params->p_Wi);
        rocke_value_t* col_ok = rocke_b_land(b, ge, lt);
        rocke_value_t* w_off = rocke_b_mul(b, wi, c.params->p_A_stride_wi);
        rocke_value_t* col_off = rocke_b_mul(b, w_off, c.c_half);
        c.col_terms.push_back(rocke_b_select(b, col_ok, col_off, c.oob_sentinel));
    }
}

/* Raw <CPL x elem> input vectors of the row at row_ok / hi, one per column. */
void dwp_load_cols(dwp_ctx& c, rocke_value_t* hi, rocke_value_t* row_ok, vals_t& out)
{
    rocke_ir_builder_t* b = c.b;
    rocke_value_t* h_off = rocke_b_mul(b, hi, c.params->p_A_stride_hi);
    rocke_value_t* row_off = rocke_b_mul(b, rocke_b_add(b, c.n_off, h_off), c.c_half);
    for(rocke_value_t* t : c.col_terms)
    {
        rocke_value_t* rc = rocke_b_add(b, row_off, t);
        rocke_value_t* sel = rocke_b_select(b, row_ok, rc, c.oob_sentinel);
        out.push_back(c.h->loadv(c.a_rsrc, rocke_b_add(b, c.ch_off, sel)));
    }
}

void dwp_set_err(rocke_ir_builder_t* b)
{
    if(b->status == ROCKE_OK)
        b->status = ROCKE_ERR_VALUE;
}

} // namespace

/* Python: _build_direct_depthwise_packed(spec, arch). */
rocke_kernel_def_t* rocke_dconv_build_depthwise_packed(rocke_ir_builder_t* b,
                                                       const rocke_direct_depthwise_spec_t* spec,
                                                       const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_direct_depthwise_validate(spec, reason, sizeof reason) != ROCKE_OK
       || !rocke_direct_depthwise_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        dwp_set_err(b);
        return NULL;
    }

    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BLOCK_W = spec->block_w;
    const int CPL = spec->cpl;
    const int NP = CPL / 2;
    const int THREADS = rocke_direct_depthwise_threads_per_block(spec);
    const int S = p->stride;
    const int KH = p->KH;
    const int KW = p->KW;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    rocke_dconv_params_t params;
    rocke_dconv_emit_params(b, &params, "fwd", rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16"));
    rocke_value_t* p_Hi = params.p_Hi;
    dwp_helpers h(b, p, CPL);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_W = params.p_Wo;
    rocke_value_t* c_half = rocke_b_const_i32(b, 2);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    rocke_value_t* zero2 = dwp_zero2(b);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* n = rocke_b_block_id_z(b);
    rocke_value_t* q_tile_start = rocke_b_mul(b, bx, rocke_b_const_i32(b, BLOCK_W));
    rocke_value_t* ch = dwp_channel(b, by, tid, THREADS, CPL);
    rocke_value_t* ch_in_range = rocke_b_cmp_lt(b, ch, params.p_groups);

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);
    rocke_dynamic_tensor_descriptor_t* d_dyn = rocke_dconv_d_descriptor_dynamic(b, &params);
    if(!d_dyn)
        return NULL;
    const rocke_tensor_descriptor_t* d_desc = &d_dyn->base;

    /* Weights B[C, KH, KW, 1]: per channel pair, a <2 x f32> per tap. Lanes
     * past the channel count read the OOB sentinel (zeros) and never store. */
    rocke_value_t* w_ch_off;
    {
        rocke_value_t* m = rocke_b_mul(b, ch, rocke_b_const_i32(b, KH * KW));
        w_ch_off = rocke_b_mul(b, m, c_half);
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
                rocke_value_t* safe = rocke_b_select(b, ch_in_range, off, oob_sentinel);
                vals.push_back(rocke_b_cast_to_f32(b, h.load1(b_rsrc, safe)));
            }
            for(int k = 0; k < NP; ++k)
                weights[r_const][s_const].push_back(
                    rocke_b_vec_pack(b, &vals[2 * k], 2, rocke_f32()));
        }
    }

    /* Streaming loop bounds and column terms as in build_direct_depthwise. */
    rocke_value_t* c_KH = rocke_b_const_i32(b, KH);
    rocke_value_t* c_stride_rv = rocke_b_const_i32(b, S);
    rocke_value_t* n_iters_v = rocke_b_add(b, p_Hi, rocke_b_const_i32(b, KH - 1));
    rocke_value_t* n_fma_groups_v
        = rocke_b_div(b, rocke_b_add(b, p_Hi, rocke_b_const_i32(b, KH - 1)), c_KH);
    rocke_value_t* n_groups_v
        = rocke_b_div(b, rocke_b_add(b, p_Hi, rocke_b_const_i32(b, p->PAD + KH - 1)), c_KH);

    /* Distinct input columns in first-tap order. */
    std::vector<int> cols;
    std::vector<int> col_idx((size_t)((BLOCK_W - 1) * S + KW), -1);
    for(int w_out = 0; w_out < BLOCK_W; ++w_out)
    {
        for(int s_const = 0; s_const < KW; ++s_const)
        {
            int col = w_out * S + s_const;
            if(col_idx[(size_t)col] >= 0)
                continue;
            col_idx[(size_t)col] = (int)cols.size();
            cols.push_back(col);
        }
    }
    const int n_cols = (int)cols.size();

    dwp_ctx c;
    c.b = b;
    c.params = &params;
    c.h = &h;
    c.PAD = p->PAD;
    c.c0 = c0;
    c.c_half = c_half;
    c.oob_sentinel = oob_sentinel;
    c.a_rsrc = a_rsrc;
    c.ch_off = rocke_b_mul(b, ch, c_half);
    dwp_col_terms(c, q_tile_start, c_stride_rv, cols);
    c.n_off = rocke_b_mul(b, n, params.p_A_stride_n);

    /* Python: load_row(y) -- raw input vectors of streaming row y. */
    auto load_row = [&](rocke_value_t* y, vals_t& out) {
        rocke_value_t* hi = rocke_b_add(b, y, rocke_b_const_i32(b, -p->PAD));
        rocke_value_t* row_ok = rocke_b_cmp_lt(b, hi, p_Hi);
        dwp_load_cols(c, hi, row_ok, out);
    };
    /* Python: row_y(grp_iv, j). */
    auto row_y = [&](rocke_value_t* grp_iv, int j) {
        rocke_value_t* m = rocke_b_mul(b, grp_iv, c_KH);
        return rocke_b_add(b, m, rocke_b_const_i32(b, p->PAD + j));
    };
    /* Python: flush(grp_iv, j, y_j, slots); slots is flat [f * NP + k]. */
    auto flush = [&](rocke_value_t* grp_iv, int j, rocke_value_t* y_j, vals_t& slots) {
        rocke_value_t* j_valid = rocke_b_cmp_lt(b, y_j, n_iters_v);
        const int P_FLUSH_j = (p->PAD + j + 1) % KH;
        rocke_value_t* p_flush_rv = rocke_b_add(b, y_j, rocke_b_const_i32(b, -(KH - 1)));
        rocke_value_t* flush_ge;
        rocke_value_t* should_flush;
        if(j >= p->PAD)
            flush_ge = j_valid;
        else
            flush_ge = rocke_b_land(b, rocke_b_cmp_lt(b, c0, grp_iv), j_valid);
        if(S == 1)
        {
            should_flush = flush_ge;
        }
        else
        {
            rocke_value_t* md = rocke_b_mod(b, p_flush_rv, c_stride_rv);
            should_flush = rocke_b_land(b, flush_ge, rocke_b_cmp_eq(b, md, c0));
        }
        rocke_value_t* ho_row_j = rocke_b_div(b, p_flush_rv, c_stride_rv);
        for(int w_out = 0; w_out < BLOCK_W; ++w_out)
        {
            rocke_value_t* out_q = rocke_b_add(b, q_tile_start, rocke_b_const_i32(b, w_out));
            rocke_value_t* q_ok = rocke_b_cmp_lt(b, out_q, c_W);
            rocke_value_t* ok = rocke_b_land(b, rocke_b_land(b, q_ok, ch_in_range), should_flush);
            rocke_value_t* d_off = dwp_d_offset(b, d_desc, n, ho_row_j, out_q, ch);
            rocke_value_t* safe
                = rocke_b_select(b, ok, rocke_b_mul(b, d_off, c_half), oob_sentinel);
            const size_t f = (size_t)(P_FLUSH_j * BLOCK_W + w_out);
            h.store(d_rsrc, safe, &slots[f * NP]);
            for(int k = 0; k < NP; ++k)
                slots[f * NP + k] = zero2;
        }
    };

    int PF = (16 + n_cols - 1) / n_cols;
    if(PF > KH)
        PF = KH;
    const int n_flat = KH * BLOCK_W;
    const int n_acc = n_flat * NP;
    std::vector<std::string> names;
    vals_t inits;
    for(int f = 0; f < n_flat; ++f)
    {
        for(int k = 0; k < NP; ++k)
        {
            names.push_back("dwp_acc_" + std::to_string(f) + "_k" + std::to_string(k));
            inits.push_back(zero2);
        }
    }
    for(int k = 0; k < PF; ++k)
    {
        vals_t row;
        load_row(rocke_b_const_i32(b, p->PAD + k), row);
        for(int i = 0; i < n_cols; ++i)
        {
            names.push_back("dwp_a_r" + std::to_string(k) + "_c" + std::to_string(cols[(size_t)i]));
            inits.push_back(row[(size_t)i]);
        }
    }
    std::vector<rocke_iter_arg_t> iargs(inits.size());
    for(size_t i = 0; i < inits.size(); ++i)
    {
        iargs[i].name = names[i].c_str();
        iargs[i].init = inits[i];
    }

    rocke_for_t fma_loop = rocke_b_scf_for_iter(b,
                                                c0,
                                                n_fma_groups_v,
                                                c1,
                                                iargs.data(),
                                                (int)iargs.size(),
                                                "dwp_grp",
                                                /*unroll=*/false,
                                                /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, fma_loop.body);
    {
        vals_t slots(fma_loop.iter_vars, fma_loop.iter_vars + n_acc);
        std::vector<vals_t> window;
        for(int k = 0; k < PF; ++k)
            window.emplace_back(fma_loop.iter_vars + n_acc + k * n_cols,
                                fma_loop.iter_vars + n_acc + (k + 1) * n_cols);
        for(int j = 0; j < KH; ++j)
        {
            rocke_value_t* y_j = row_y(fma_loop.iv, j);
            {
                rocke_value_t* y_pf = rocke_b_add(b, y_j, rocke_b_const_i32(b, PF));
                vals_t row;
                load_row(y_pf, row);
                window.push_back(row);
            }
            /* a_row[col] = the column's NP pairs. */
            std::vector<vals_t> a_row;
            for(rocke_value_t* v : window.front())
            {
                vals_t pr;
                h.to_pairs(v, pr);
                a_row.push_back(pr);
            }
            window.erase(window.begin());
            for(int w_out = 0; w_out < BLOCK_W; ++w_out)
            {
                for(int s_const = 0; s_const < KW; ++s_const)
                {
                    const vals_t& a = a_row[(size_t)col_idx[(size_t)(w_out * S + s_const)]];
                    for(int r_const = 0; r_const < KH; ++r_const)
                    {
                        const int idx = ((p->PAD + j - r_const + KH) % KH) * BLOCK_W + w_out;
                        const vals_t& w = weights[r_const][s_const];
                        for(int k = 0; k < NP; ++k)
                            slots[(size_t)(idx * NP + k)] = rocke_b_vector_fma(
                                b, w[(size_t)k], a[(size_t)k], slots[(size_t)(idx * NP + k)]);
                    }
                }
            }
            flush(fma_loop.iv, j, y_j, slots);
        }
        vals_t yields(slots);
        for(const vals_t& row : window)
            yields.insert(yields.end(), row.begin(), row.end());
        rocke_b_scf_yield(b, yields.data(), (int)yields.size());
    }
    rocke_b_region_leave(b);

    /* The rows past the image: no input, only output rows to complete. */
    std::vector<std::string> tail_names;
    std::vector<rocke_iter_arg_t> tail_args((size_t)n_acc);
    for(int i = 0; i < n_acc; ++i)
        tail_names.push_back("dwp_tail_i" + std::to_string(i));
    for(int i = 0; i < n_acc; ++i)
    {
        tail_args[(size_t)i].name = tail_names[(size_t)i].c_str();
        tail_args[(size_t)i].init = fma_loop.op->results[i];
    }
    rocke_for_t flush_loop = rocke_b_scf_for_iter(b,
                                                  n_fma_groups_v,
                                                  n_groups_v,
                                                  c1,
                                                  tail_args.data(),
                                                  n_acc,
                                                  "dwp_tail",
                                                  /*unroll=*/false,
                                                  /*elide_trailing_barrier=*/false);
    rocke_b_region_enter(b, flush_loop.body);
    {
        vals_t slots(flush_loop.iter_vars, flush_loop.iter_vars + n_acc);
        for(int j = 0; j < KH; ++j)
        {
            rocke_value_t* y_j = row_y(flush_loop.iv, j);
            flush(flush_loop.iv, j, y_j, slots);
        }
        rocke_b_scf_yield(b, slots.data(), n_acc);
    }
    rocke_b_region_leave(b);

    return rocke_ir_builder_kernel(b);
}

/* Python: _build_direct_depthwise_tiled_packed(spec, arch). The caller
 * (rocke_build_direct_depthwise_tiled) has already validated the spec. */
rocke_kernel_def_t* rocke_dconv_build_depthwise_tiled_packed(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_tiled_spec_t* spec, const char* arch)
{
    (void)arch;
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BW = spec->block_w;
    const int TH = spec->block_h;
    const int CPL = spec->cpl;
    const int NP = CPL / 2;
    const int THREADS = rocke_direct_depthwise_tiled_threads_per_block(spec);
    const int S = p->stride;
    const int KH = p->KH;
    const int KW = p->KW;
    const int PAD = p->PAD;
    const int n_cols = rocke_direct_depthwise_tiled_n_cols(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    rocke_dconv_params_t params;
    rocke_dconv_emit_params(b, &params, "fwd", rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16"));
    rocke_value_t* p_Hi = params.p_Hi;
    rocke_value_t* p_Ho = params.p_Ho;
    dwp_helpers h(b, p, CPL);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_half = rocke_b_const_i32(b, 2);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    rocke_value_t* zero2 = dwp_zero2(b);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* bz = rocke_b_block_id_z(b);
    rocke_value_t* n_h_tiles;
    {
        rocke_value_t* sum = rocke_b_add(b, p_Ho, rocke_b_const_i32(b, TH - 1));
        n_h_tiles = rocke_b_div(b, sum, rocke_b_const_i32(b, TH));
    }
    rocke_value_t* n = rocke_b_div(b, bz, n_h_tiles);
    rocke_value_t* h_tile = rocke_b_mod(b, bz, n_h_tiles);
    rocke_value_t* q0 = rocke_b_mul(b, bx, rocke_b_const_i32(b, BW));
    rocke_value_t* h0 = rocke_b_mul(b, h_tile, rocke_b_const_i32(b, TH));
    rocke_value_t* ch = dwp_channel(b, by, tid, THREADS, CPL);
    rocke_value_t* ch_in_range = rocke_b_cmp_lt(b, ch, params.p_groups);

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);

    /* Weights B[C, KH, KW, 1]: channel ch + i's filter starts at
     * (ch + i) * KH * KW. Lanes past the channel count read the sentinel. */
    rocke_value_t* w_ch_off;
    {
        rocke_value_t* m = rocke_b_mul(b, ch, rocke_b_const_i32(b, KH * KW));
        w_ch_off = rocke_b_mul(b, m, c_half);
    }
    /* Python: load_w_row(r) -- raw weights of filter row r, CPL per tap. */
    auto load_w_row = [&](rocke_value_t* r, vals_t& out) {
        rocke_value_t* m = rocke_b_mul(b, r, rocke_b_const_i32(b, KW * 2));
        rocke_value_t* row_off = rocke_b_add(b, w_ch_off, m);
        for(int s = 0; s < KW; ++s)
        {
            for(int i = 0; i < CPL; ++i)
            {
                rocke_value_t* off
                    = rocke_b_add(b, row_off, rocke_b_const_i32(b, 2 * (i * KH * KW + s)));
                rocke_value_t* safe = rocke_b_select(b, ch_in_range, off, oob_sentinel);
                out.push_back(h.load1(b_rsrc, safe));
            }
        }
    };
    /* Python: w_pairs(raw) -> [s][k]. */
    auto w_pairs = [&](const vals_t& raw) {
        vals_t xs;
        for(rocke_value_t* v : raw)
            xs.push_back(rocke_b_cast_to_f32(b, v));
        std::vector<vals_t> out((size_t)KW);
        for(int s = 0; s < KW; ++s)
            for(int k = 0; k < NP; ++k)
                out[(size_t)s].push_back(
                    rocke_b_vec_pack(b, &xs[(size_t)(s * CPL + 2 * k)], 2, rocke_f32()));
        return out;
    };

    dwp_ctx c;
    c.b = b;
    c.params = &params;
    c.h = &h;
    c.PAD = PAD;
    c.c0 = c0;
    c.c_half = c_half;
    c.oob_sentinel = oob_sentinel;
    c.a_rsrc = a_rsrc;
    c.ch_off = rocke_b_mul(b, ch, c_half);
    {
        std::vector<int> cols;
        for(int col = 0; col < n_cols; ++col)
            cols.push_back(col);
        dwp_col_terms(c, q0, rocke_b_const_i32(b, S), cols);
    }
    c.n_off = rocke_b_mul(b, n, params.p_A_stride_n);
    rocke_value_t* hi_tile;
    {
        rocke_value_t* h_s = rocke_b_mul(b, h0, rocke_b_const_i32(b, S));
        hi_tile = rocke_b_add(b, h_s, rocke_b_const_i32(b, -PAD));
    }

    /* Python: load_row(hi) -- raw input vectors of row hi, one per column. */
    auto load_row = [&](rocke_value_t* hi, vals_t& out) {
        rocke_value_t* ge = rocke_b_cmp_ge(b, hi, c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, hi, p_Hi);
        rocke_value_t* row_ok = rocke_b_land(b, ge, lt);
        dwp_load_cols(c, hi, row_ok, out);
    };
    /* Python: pairs_of(vals) -- flatten raw column vectors into pairs. */
    auto pairs_of = [&](const vals_t& vals) {
        vals_t out;
        for(rocke_value_t* v : vals)
            h.to_pairs(v, out);
        return out;
    };
    /* Python: fma_row(accs, t, row, w). */
    auto fma_row
        = [&](vals_t& accs, int t, rocke_value_t* const* row, const std::vector<vals_t>& w) {
              for(int w_out = 0; w_out < BW; ++w_out)
              {
                  for(int s = 0; s < KW; ++s)
                  {
                      const int col = w_out * S + s;
                      for(int k = 0; k < NP; ++k)
                      {
                          const size_t idx = (size_t)((t * BW + w_out) * NP + k);
                          accs[idx] = rocke_b_vector_fma(
                              b, w[(size_t)s][(size_t)k], row[col * NP + k], accs[idx]);
                      }
                  }
              }
          };

    const int n_acc = TH * BW * NP;
    vals_t accs((size_t)n_acc, zero2);
    const int n_row = n_cols * NP;
    for(int ph = 0; ph < S; ++ph)
    {
        const int n_r = (KH - ph + S - 1) / S;
        rocke_value_t* phase_base = rocke_b_add(b, hi_tile, rocke_b_const_i32(b, ph));
        std::vector<std::string> names;
        vals_t inits;
        for(int i = 0; i < n_acc; ++i)
        {
            names.push_back("dwtp_acc" + std::to_string(ph) + "_i" + std::to_string(i));
            inits.push_back(accs[(size_t)i]);
        }
        for(int t = 0; t < TH - 1; ++t)
        {
            vals_t raw;
            load_row(rocke_b_add(b, phase_base, rocke_b_const_i32(b, S * t)), raw);
            vals_t row = pairs_of(raw);
            for(size_t i = 0; i < row.size(); ++i)
            {
                names.push_back("dwtp_x" + std::to_string(ph) + "_r" + std::to_string(t) + "_c"
                                + std::to_string(i));
                inits.push_back(row[i]);
            }
        }
        {
            vals_t w0;
            load_w_row(rocke_b_const_i32(b, ph), w0);
            for(size_t i = 0; i < w0.size(); ++i)
            {
                names.push_back("dwtp_w" + std::to_string(ph) + "_i" + std::to_string(i));
                inits.push_back(w0[i]);
            }
        }
        const int n_win = (TH - 1) * n_row;
        std::vector<rocke_iter_arg_t> iargs(inits.size());
        for(size_t i = 0; i < inits.size(); ++i)
        {
            iargs[i].name = names[i].c_str();
            iargs[i].init = inits[i];
        }
        const std::string iv_name = "dwtp_r" + std::to_string(ph);
        rocke_value_t* c_nr = rocke_b_const_i32(b, n_r);
        rocke_for_t loop = rocke_b_scf_for_iter(b,
                                                c0,
                                                c_nr,
                                                c1,
                                                iargs.data(),
                                                (int)iargs.size(),
                                                iv_name.c_str(),
                                                spec->unroll_rows,
                                                /*elide_trailing_barrier=*/false);
        rocke_b_region_enter(b, loop.body);
        {
            vals_t new_accs(loop.iter_vars, loop.iter_vars + n_acc);
            /* rows: TH - 1 carried rows of n_row pairs, then the new one. */
            vals_t rows(loop.iter_vars + n_acc, loop.iter_vars + n_acc + n_win);
            vals_t w_raw(loop.iter_vars + n_acc + n_win, loop.iter_vars + iargs.size());
            std::vector<vals_t> w_cur = w_pairs(w_raw);
            rocke_value_t* step = rocke_b_mul(b, loop.iv, rocke_b_const_i32(b, S));
            vals_t new_row;
            {
                rocke_value_t* base_step = rocke_b_add(b, phase_base, step);
                rocke_value_t* hi = rocke_b_add(b, base_step, rocke_b_const_i32(b, S * (TH - 1)));
                load_row(hi, new_row);
            }
            vals_t w_next;
            {
                rocke_value_t* c_ph = rocke_b_const_i32(b, ph);
                rocke_value_t* nxt = rocke_b_add(b, step, rocke_b_const_i32(b, S));
                load_w_row(rocke_b_add(b, c_ph, nxt), w_next);
            }
            for(int t = 0; t < TH - 1; ++t)
                fma_row(new_accs, t, rows.data() + (size_t)(t * n_row), w_cur);
            {
                vals_t pr = pairs_of(new_row);
                rows.insert(rows.end(), pr.begin(), pr.end());
            }
            fma_row(new_accs, TH - 1, rows.data() + (size_t)((TH - 1) * n_row), w_cur);
            vals_t yields(new_accs);
            yields.insert(yields.end(), rows.begin() + n_row, rows.end());
            yields.insert(yields.end(), w_next.begin(), w_next.end());
            rocke_b_scf_yield(b, yields.data(), (int)yields.size());
        }
        rocke_b_region_leave(b);
        for(int i = 0; i < n_acc; ++i)
            accs[(size_t)i] = loop.op->results[i];
    }

    rocke_dynamic_tensor_descriptor_t* d_dyn = rocke_dconv_d_descriptor_dynamic(b, &params);
    if(!d_dyn)
        return NULL;
    const rocke_tensor_descriptor_t* d_desc = &d_dyn->base;
    for(int t = 0; t < TH; ++t)
    {
        rocke_value_t* ho = rocke_b_add(b, h0, rocke_b_const_i32(b, t));
        rocke_value_t* h_ok = rocke_b_land(b, rocke_b_cmp_lt(b, ho, p_Ho), ch_in_range);
        for(int w_out = 0; w_out < BW; ++w_out)
        {
            rocke_value_t* wo = rocke_b_add(b, q0, rocke_b_const_i32(b, w_out));
            rocke_value_t* ok = rocke_b_land(b, h_ok, rocke_b_cmp_lt(b, wo, params.p_Wo));
            rocke_value_t* d_off = dwp_d_offset(b, d_desc, n, ho, wo, ch);
            rocke_value_t* safe
                = rocke_b_select(b, ok, rocke_b_mul(b, d_off, c_half), oob_sentinel);
            h.store(d_rsrc, safe, &accs[(size_t)((t * BW + w_out) * NP)]);
        }
    }

    return rocke_ir_builder_kernel(b);
}
