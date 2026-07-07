// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/moe_gemm_fused.py (the requested symbol
 * subset). See the header for the symbol mapping. Byte-faithful builder-call
 * order against rocke/ir.h + the sibling helper_*.h / instance_gemm_*.h.
 */
#include "rocke/helper_rocke.instances.common.moe_gemm_fused.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_gemm_internal.h" /* _emit_mfma / _emit_smem_load / ... */
#include "rocke/ir_internal.h" /* rocke_i_set_err                              */

/* Magic-division helpers from helper_rocke.helpers.transforms.h. That header
 * is NOT included here: it defines a rich `rocke_tensor_descriptor` struct that
 * collides with the (different) `rocke_tensor_descriptor` in tensor_view.h, which
 * this module needs for TensorView / TileWindow. The two cannot coexist in one
 * TU (see instance_permute_nd.c for the same constraint), so we forward-declare
 * just the two pure magic-division entry points we use. */
/* C++ build: these are cross-TU C-ABI helpers (defined with C linkage in
 * helpers/transforms.c); the forward decls must be extern "C" so C++ does not
 * mangle the references. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
bool rocke_calculate_magic_numbers(rocke_ir_builder_t* b,
                                   int divisor,
                                   uint64_t* out_multiplier,
                                   int* out_shift);
rocke_value_t* rocke_do_magic_division(rocke_ir_builder_t* b,
                                       rocke_value_t* dividend,
                                       uint64_t multiplier,
                                       int shift);
#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ guards */

#define ROCKE_MOE_MAX_MFMAS 256
#define ROCKE_MOE_MAX_VECS 256
#define ROCKE_MOE_MAX_OPERANDS 4
#define ROCKE_MOE_MAX_ITER_ARGS \
    (2 * ROCKE_MOE_MAX_MFMAS + ROCKE_MOE_MAX_OPERANDS * ROCKE_MOE_MAX_VECS)

/* _storage_dtype(spec): homogeneous A/B/C dtype -> rocke_type_t. Mirrors the
 * gemm_universal helper; re-derived from the spec's data dtype string. Shared
 * with the gate-up / down / interleaved bodies via the sibling header. */
const rocke_type_t* rocke_moe_storage_dtype(const rocke_gemm_universal_spec_t* u)
{
    const char* d = u->data.dtype_a;
    if(d == NULL)
    {
        return rocke_f16();
    }
    if(strcmp(d, "f16") == 0 || strcmp(d, "fp16") == 0)
    {
        return rocke_f16();
    }
    if(strcmp(d, "bf16") == 0)
    {
        return rocke_bf16();
    }
    return rocke_scalar_by_name(d);
}

/* _mfma_atom_widths(spec) -> (a_per_lane, b_per_lane, c_per_lane). MFMA-only
 * geometry: the warp-tile atom's per-lane fragment widths. Shared with the
 * gate-up / down / interleaved bodies via the sibling header. */
void rocke_moe_mfma_atom_widths(const rocke_gemm_universal_spec_t* u,
                                int* a_per,
                                int* b_per,
                                int* c_per)
{
    const rocke_gemm_tile_spec_t* t = &u->tile;
    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom(u->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    int wm = t->warp_tile_m;
    int wn = t->warp_tile_n;
    int wk = t->warp_tile_k;
    int wave = u->wave_size;
    /* per-lane widths: (wm*wk)/wave, (wn*wk)/wave, (wm*wn)/wave. */
    *a_per = (wm * wk) / wave;
    *b_per = (wn * wk) / wave;
    *c_per = (wm * wn) / wave;
    (void)atom;
}

/* ====================================================================== *
 *  Leaves
 * ====================================================================== */

void rocke_moe_magic_div_mod(rocke_ir_builder_t* b,
                             rocke_value_t* dividend,
                             int divisor,
                             rocke_value_t** out_quot,
                             rocke_value_t** out_rem)
{
    if(divisor == 1)
    {
        *out_quot = dividend;
        *out_rem = rocke_b_const_i32(b, 0);
        return;
    }
    uint64_t mult = 0;
    int shift = 0;
    if(!rocke_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        *out_quot = NULL;
        *out_rem = NULL;
        return;
    }
    rocke_value_t* quot = rocke_do_magic_division(b, dividend, mult, shift);
    rocke_value_t* rem
        = rocke_b_sub(b, dividend, rocke_b_mul(b, quot, rocke_b_const_i32(b, divisor)));
    *out_quot = quot;
    *out_rem = rem;
}

void rocke_moe_vec_rowcol(rocke_ir_builder_t* b,
                          int e,
                          rocke_value_t* tid,
                          rocke_value_t* c_threads,
                          int block_k_div_vec,
                          rocke_value_t* c_load_vec,
                          int load_vec,
                          rocke_value_t** out_row,
                          rocke_value_t** out_col)
{
    rocke_value_t* vec_idx
        = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, e), c_threads), tid);
    rocke_value_t* row = NULL;
    rocke_value_t* col_v = NULL;
    rocke_moe_magic_div_mod(b, vec_idx, block_k_div_vec, &row, &col_v);
    *out_row = row;
    *out_col = (load_vec > 1) ? rocke_b_mul(b, col_v, c_load_vec) : col_v;
}

rocke_value_t* rocke_moe_gemm_fused_silu_mul_f32(rocke_ir_builder_t* b,
                                                 rocke_value_t* g,
                                                 rocke_value_t* u,
                                                 rocke_value_t* one_f32,
                                                 rocke_value_t* c_neg_log2e)
{
    rocke_value_t* sig = rocke_b_rcp(
        b, rocke_b_fadd(b, one_f32, rocke_b_exp2(b, rocke_b_fmul(b, c_neg_log2e, g))));
    rocke_value_t* silu = rocke_b_fmul(b, g, sig);
    return rocke_b_fmul(b, silu, u);
}

rocke_value_t* rocke_moe_pad_in_bounds(rocke_ir_builder_t* b,
                                       rocke_value_t* c_m,
                                       rocke_value_t* c_n,
                                       rocke_value_t* M,
                                       rocke_value_t* N,
                                       bool pad_m,
                                       bool pad_n,
                                       int vec)
{
    if(!(pad_m || pad_n))
    {
        return NULL;
    }
    rocke_value_t* checks[2];
    int nc = 0;
    if(pad_m)
    {
        checks[nc++] = rocke_b_cmp_lt(b, c_m, M);
    }
    if(pad_n)
    {
        if(vec == 1)
        {
            checks[nc++] = rocke_b_cmp_lt(b, c_n, N);
        }
        else
        {
            rocke_value_t* c_n_last = rocke_b_add(b, c_n, rocke_b_const_i32(b, vec - 1));
            checks[nc++] = rocke_b_cmp_lt(b, c_n_last, N);
        }
    }
    return (nc == 1) ? checks[0] : rocke_b_land(b, checks[0], checks[1]);
}

/* ====================================================================== *
 *  _CWarpDecode
 * ====================================================================== */

int rocke_moe_cwarp_decode_init(rocke_moe_cwarp_decode_t* out,
                                rocke_ir_builder_t* b,
                                const rocke_gemm_universal_spec_t* spec,
                                rocke_value_t* warp_m_off,
                                rocke_value_t* warp_n_off,
                                rocke_value_t* lane)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    const rocke_mfma_atom_t* atom
        = rocke_mfma_atom(spec->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    if(atom == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "_CWarpDecode: no MFMA atom for warp tile");
        return 0;
    }
    const rocke_tile_distribution_encoding_t* enc = rocke_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
    {
        return 0;
    }
    const rocke_tile_distribution_t* dist = rocke_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return 0;
    }
    out->b = b;
    out->spec = spec;
    out->dist = dist;
    /* kCM1PerLane is Hs[0][2]; kCNLane is Hs[1][0]. */
    out->m1 = enc->Hs[0].levels[2];
    int n_lane = enc->Hs[1].levels[0];
    rocke_value_t* c_n_lane = rocke_b_const_i32(b, n_lane);
    out->n_in_atom = rocke_b_mod(b, lane, c_n_lane);
    out->m_blk = rocke_b_div(b, lane, c_n_lane);
    out->warp_m_off = warp_m_off;
    out->warp_n_off = warp_n_off;
    return 1;
}

/* _row_col_in_atom(i): calculate_x over (y0, y1) = (i // m1, i % m1). */
static void rocke_moe_cwarp_row_col_in_atom(const rocke_moe_cwarp_decode_t* d,
                                            int i,
                                            rocke_value_t** out_row,
                                            rocke_value_t** out_col)
{
    rocke_ir_builder_t* b = d->b;
    rocke_value_t* y0 = rocke_b_const_i32(b, i / d->m1);
    rocke_value_t* y1 = rocke_b_const_i32(b, i % d->m1);
    rocke_value_t* ys[2] = {y0, y1};
    /* ps = [[m_blk, n_in_atom]] (one P dim with two contributions). */
    rocke_value_t* p0[2] = {d->m_blk, d->n_in_atom};
    rocke_value_t* const* ps[1] = {p0};
    int ps_counts[1] = {2};
    rocke_value_t* x_out[2] = {NULL, NULL};
    if(!rocke_tile_distribution_calculate_x(b, d->dist, ys, 2, ps, ps_counts, 1, x_out, 2))
    {
        *out_row = NULL;
        *out_col = NULL;
        return;
    }
    *out_row = x_out[0];
    *out_col = x_out[1];
}

void rocke_moe_cwarp_decode_coords(const rocke_moe_cwarp_decode_t* d,
                                   int mi,
                                   int ni,
                                   int i,
                                   rocke_value_t** out_ld_m,
                                   rocke_value_t** out_ld_n)
{
    rocke_ir_builder_t* b = d->b;
    const rocke_gemm_tile_spec_t* t = &d->spec->tile;
    rocke_value_t* row_in_atom = NULL;
    rocke_value_t* col_in_atom = NULL;
    rocke_moe_cwarp_row_col_in_atom(d, i, &row_in_atom, &col_in_atom);
    *out_ld_m = rocke_b_add(
        b, d->warp_m_off, rocke_b_add(b, rocke_b_const_i32(b, mi * t->warp_tile_m), row_in_atom));
    *out_ld_n = rocke_b_add(
        b, d->warp_n_off, rocke_b_add(b, rocke_b_const_i32(b, ni * t->warp_tile_n), col_in_atom));
}

rocke_value_t* rocke_moe_cwarp_decode_warp_row(const rocke_moe_cwarp_decode_t* d, int mi, int i)
{
    rocke_ir_builder_t* b = d->b;
    const rocke_gemm_tile_spec_t* t = &d->spec->tile;
    rocke_value_t* row_in_atom = NULL;
    rocke_value_t* col_in_atom = NULL;
    rocke_moe_cwarp_row_col_in_atom(d, i, &row_in_atom, &col_in_atom);
    return rocke_b_add(
        b, d->warp_m_off, rocke_b_add(b, rocke_b_const_i32(b, mi * t->warp_tile_m), row_in_atom));
}

rocke_value_t* rocke_moe_cwarp_decode_warp_col(const rocke_moe_cwarp_decode_t* d, int ni)
{
    rocke_ir_builder_t* b = d->b;
    const rocke_gemm_tile_spec_t* t = &d->spec->tile;
    rocke_value_t* row_in_atom = NULL;
    rocke_value_t* col_in_atom = NULL;
    rocke_moe_cwarp_row_col_in_atom(d, 0, &row_in_atom, &col_in_atom);
    return rocke_b_add(
        b, d->warp_n_off, rocke_b_add(b, rocke_b_const_i32(b, ni * t->warp_tile_n), col_in_atom));
}

/* ====================================================================== *
 *  _MoeKloopPlan
 * ====================================================================== */

int rocke_moe_kloop_plan_init(rocke_moe_kloop_plan_t* out,
                              rocke_ir_builder_t* b,
                              const rocke_gemm_universal_spec_t* u,
                              rocke_value_t* tid)
{
    const rocke_gemm_tile_spec_t* t = &u->tile;
    out->b = b;
    out->u = u;
    out->tid = tid;
    out->storage_dtype = rocke_moe_storage_dtype(u);
    rocke_moe_mfma_atom_widths(u, &out->a_per_lane, &out->b_per_lane, &out->c_per_lane);
    out->block_m = t->tile_m;
    out->block_n = t->tile_n;
    out->block_k = t->tile_k;
    out->mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    out->mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    out->k_atoms = rocke_gemm_tile_k_atoms_per_tile_k(t);

    int threads = u->block_size;
    int load_vec = 0;
    if(rocke_choose_load_vec(t->tile_m, t->tile_n, t->tile_k, u->block_size, &load_vec) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "_MoeKloopPlan: choose_load_vec failed");
        return 0;
    }
    out->threads = threads;
    out->load_vec = load_vec;
    out->a_vecs_per_thread = (out->block_m * out->block_k) / load_vec / threads;
    out->b_vecs_per_thread = (out->block_n * out->block_k) / load_vec / threads;
    out->c_threads = rocke_b_const_i32(b, threads);
    out->c_load_vec = rocke_b_const_i32(b, load_vec);
    out->block_k_div_vec = out->block_k / load_vec;
    return 1;
}

static void rocke_moe_plan_rowcol(const rocke_moe_kloop_plan_t* plan,
                                  int e,
                                  rocke_value_t** out_row,
                                  rocke_value_t** out_col)
{
    rocke_moe_vec_rowcol(plan->b,
                         e,
                         plan->tid,
                         plan->c_threads,
                         plan->block_k_div_vec,
                         plan->c_load_vec,
                         plan->load_vec,
                         out_row,
                         out_col);
}

/* ====================================================================== *
 *  Shared k-loop core
 * ====================================================================== */

void rocke_moe_emit_global_load(const rocke_moe_kloop_plan_t* plan,
                                const rocke_tensor_view_t* a_view,
                                rocke_value_t* const a_mn_origin[2],
                                const rocke_moe_operand_t* operands,
                                int num_operands,
                                rocke_value_t* const b_mn_origin[2],
                                rocke_value_t* k_off,
                                rocke_value_t** out_a_regs,
                                rocke_value_t** out_b_regs)
{
    rocke_ir_builder_t* b = plan->b;
    rocke_value_t* a_origin[3] = {a_mn_origin[0], a_mn_origin[1], k_off};
    rocke_value_t* b_origin[3] = {b_mn_origin[0], b_mn_origin[1], k_off};
    int a_lengths[3] = {1, plan->block_m, plan->block_k};
    int b_lengths[3] = {1, plan->block_n, plan->block_k};

    rocke_tile_window_t a_global;
    if(rocke_make_tile_window(&a_global, a_view, a_lengths, a_origin, 3) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "_emit_moe_global_load: A window");
        return;
    }
    for(int e = 0; e < plan->a_vecs_per_thread; ++e)
    {
        rocke_value_t* row = NULL;
        rocke_value_t* col = NULL;
        rocke_moe_plan_rowcol(plan, e, &row, &col);
        /* Python builds the batch index inline at each load call site
         * (a_global.load_*(b, b.const_i32(0), row, col)) AFTER _rowcol, not from
         * a hoisted constant. Emit it here in the same order so the global SSA
         * value counter matches Python op-for-op. */
        rocke_value_t* idx[3] = {rocke_b_const_i32(b, 0), row, col};
        if(plan->load_vec == 1)
        {
            out_a_regs[e] = rocke_tile_window_load_scalar(b, &a_global, idx, 3);
        }
        else
        {
            out_a_regs[e] = rocke_tile_window_load_vec(b, &a_global, idx, 3, plan->load_vec);
        }
    }

    for(int g = 0; g < num_operands; ++g)
    {
        const rocke_moe_operand_t* op = &operands[g];
        rocke_value_t** regs = out_b_regs + (size_t)g * plan->b_vecs_per_thread;
        if(op->load_b != NULL)
        {
            for(int e = 0; e < plan->b_vecs_per_thread; ++e)
            {
                rocke_value_t* row = NULL;
                rocke_value_t* col = NULL;
                rocke_moe_plan_rowcol(plan, e, &row, &col);
                regs[e] = op->load_b(b, e, k_off, row, col, op->load_b_user);
            }
        }
        else
        {
            rocke_tile_window_t b_global;
            if(rocke_make_tile_window(&b_global, op->global_view, b_lengths, b_origin, 3)
               != ROCKE_OK)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "_emit_moe_global_load: B window");
                return;
            }
            for(int e = 0; e < plan->b_vecs_per_thread; ++e)
            {
                rocke_value_t* row = NULL;
                rocke_value_t* col = NULL;
                rocke_moe_plan_rowcol(plan, e, &row, &col);
                /* Fresh inline batch const per load (matches Python order). */
                rocke_value_t* idx[3] = {rocke_b_const_i32(b, 0), row, col};
                if(plan->load_vec == 1)
                {
                    regs[e] = rocke_tile_window_load_scalar(b, &b_global, idx, 3);
                }
                else
                {
                    regs[e] = rocke_tile_window_load_vec(b, &b_global, idx, 3, plan->load_vec);
                }
            }
        }
    }
}

void rocke_moe_emit_lds_store(const rocke_moe_kloop_plan_t* plan,
                              const rocke_tensor_view_t* a_lds_view,
                              rocke_value_t* const* a_regs,
                              const rocke_moe_operand_t* operands,
                              int num_operands,
                              rocke_value_t* const* b_reg_groups)
{
    rocke_ir_builder_t* b = plan->b;
    rocke_value_t* z[2] = {rocke_b_const_i32(b, 0), rocke_b_const_i32(b, 0)};
    int a_lengths[2] = {plan->block_m, plan->block_k};
    int b_lengths[2] = {plan->block_n, plan->block_k};

    rocke_tile_window_t a_lds;
    if(rocke_make_tile_window(&a_lds, a_lds_view, a_lengths, z, 2) != ROCKE_OK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "_emit_moe_lds_store: A lds window");
        return;
    }
    for(int e = 0; e < plan->a_vecs_per_thread; ++e)
    {
        rocke_value_t* row = NULL;
        rocke_value_t* col = NULL;
        rocke_moe_plan_rowcol(plan, e, &row, &col);
        rocke_value_t* idx[2] = {row, col};
        if(plan->load_vec == 1)
        {
            rocke_tile_window_store_scalar(b, &a_lds, idx, 2, a_regs[e], 0);
        }
        else
        {
            rocke_tile_window_store_vec(b, &a_lds, idx, 2, a_regs[e], plan->load_vec);
        }
    }
    for(int g = 0; g < num_operands; ++g)
    {
        const rocke_moe_operand_t* op = &operands[g];
        rocke_value_t* const* regs = b_reg_groups + (size_t)g * plan->b_vecs_per_thread;
        rocke_tile_window_t b_lds;
        if(rocke_make_tile_window(&b_lds, op->lds_view, b_lengths, z, 2) != ROCKE_OK)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "_emit_moe_lds_store: B lds window");
            return;
        }
        for(int e = 0; e < plan->b_vecs_per_thread; ++e)
        {
            rocke_value_t* row = NULL;
            rocke_value_t* col = NULL;
            rocke_moe_plan_rowcol(plan, e, &row, &col);
            rocke_value_t* idx[2] = {row, col};
            if(plan->load_vec == 1 && op->store_scalar_ok)
            {
                rocke_tile_window_store_scalar(b, &b_lds, idx, 2, (rocke_value_t*)regs[e], 0);
            }
            else
            {
                rocke_tile_window_store_vec(
                    b, &b_lds, idx, 2, (rocke_value_t*)regs[e], plan->load_vec);
            }
        }
    }
}

void rocke_moe_emit_mfma_phase(const rocke_moe_kloop_plan_t* plan,
                               rocke_value_t* a_smem,
                               const rocke_moe_operand_t* operands,
                               int num_operands,
                               rocke_value_t* const* const* acc_groups,
                               const int* group_sizes,
                               rocke_value_t* warp_m_idx,
                               rocke_value_t* warp_n_idx,
                               rocke_value_t* lane,
                               int sched_groups,
                               rocke_value_t** out_groups_flat)
{
    rocke_ir_builder_t* b = plan->b;
    const rocke_gemm_tile_spec_t* t = &plan->u->tile;
    rocke_value_t* m_in_atom = rocke_b_mod(b, lane, rocke_b_const_i32(b, t->warp_tile_m));
    rocke_value_t* k_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, t->warp_tile_m));
    rocke_value_t* n_in_atom = rocke_b_mod(b, lane, rocke_b_const_i32(b, t->warp_tile_n));
    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, plan->mfmas_m * t->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, plan->mfmas_n * t->warp_tile_n));

    /* new_groups starts as a copy of acc_groups (flat). Lay out per-operand
     * offsets into out_groups_flat (same order as group_sizes). */
    int off_per_group[ROCKE_MOE_MAX_OPERANDS];
    int run = 0;
    for(int g = 0; g < num_operands; ++g)
    {
        off_per_group[g] = run;
        for(int j = 0; j < group_sizes[g]; ++j)
        {
            out_groups_flat[run + j] = acc_groups[g][j];
        }
        run += group_sizes[g];
    }

    for(int kk = 0; kk < plan->k_atoms; ++kk)
    {
        /* Python emits the operands of `col_base` strictly left-to-right:
         *   b.add(b.mul(k_blk, const(a_per_lane)), const(kk*warp_tile_k))
         * i.e. const(a_per_lane) -> mul -> const(kk*warp_tile_k) -> add.
         * C argument-evaluation order is unspecified, so split the nested calls
         * into ordered statements to keep the SSA value counter byte-identical. */
        rocke_value_t* col_mul = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, plan->a_per_lane));
        rocke_value_t* col_base
            = rocke_b_add(b, col_mul, rocke_b_const_i32(b, kk * t->warp_tile_k));
        rocke_value_t* a_rows[ROCKE_MOE_MAX_MFMAS];
        for(int mi = 0; mi < plan->mfmas_m; ++mi)
        {
            rocke_value_t* a_row
                = rocke_b_add(b,
                              warp_m_off,
                              rocke_b_add(b, rocke_b_const_i32(b, mi * t->warp_tile_m), m_in_atom));
            a_rows[mi] = rocke_gemm_emit_smem_load(
                b, a_smem, a_row, col_base, plan->a_per_lane, plan->storage_dtype);
        }
        /* B fragments: one column set per operand. */
        rocke_value_t* b_cols[ROCKE_MOE_MAX_OPERANDS][ROCKE_MOE_MAX_MFMAS];
        for(int gi = 0; gi < num_operands; ++gi)
        {
            for(int ni = 0; ni < plan->mfmas_n; ++ni)
            {
                rocke_value_t* b_row = rocke_b_add(
                    b,
                    warp_n_off,
                    rocke_b_add(b, rocke_b_const_i32(b, ni * t->warp_tile_n), n_in_atom));
                b_cols[gi][ni] = rocke_gemm_emit_smem_load(
                    b, operands[gi].smem, b_row, col_base, plan->b_per_lane, plan->storage_dtype);
            }
        }
        int flat = 0;
        for(int mi = 0; mi < plan->mfmas_m; ++mi)
        {
            for(int ni = 0; ni < plan->mfmas_n; ++ni)
            {
                for(int gi = 0; gi < num_operands; ++gi)
                {
                    int slot = off_per_group[gi] + flat;
                    out_groups_flat[slot] = rocke_gemm_emit_mfma(
                        b, plan->u, a_rows[mi], b_cols[gi][ni], out_groups_flat[slot]);
                }
                flat++;
            }
        }
        if(sched_groups
           && (strcmp(plan->u->trait.pipeline, "compv3") == 0
               || strcmp(plan->u->trait.pipeline, "compv4") == 0))
        {
            rocke_b_sched_group_barrier(b, 0x100, 1, 0);
            rocke_b_sched_group_barrier(b, 0x008, sched_groups, 0);
        }
    }
}

int rocke_moe_emit_prefetch_kloop(const rocke_moe_kloop_plan_t* plan,
                                  const rocke_tensor_view_t* a_view,
                                  const rocke_tensor_view_t* a_lds_view,
                                  rocke_value_t* a_smem,
                                  rocke_value_t* const a_mn_origin[2],
                                  const rocke_moe_operand_t* operands,
                                  int num_operands,
                                  rocke_value_t* const b_mn_origin[2],
                                  rocke_value_t* const* acc_inits_flat,
                                  const char* const* acc_names_flat,
                                  const int* group_sizes,
                                  rocke_value_t* K,
                                  rocke_value_t* warp_m_idx,
                                  rocke_value_t* warp_n_idx,
                                  rocke_value_t* lane,
                                  int sched_groups,
                                  rocke_value_t** out_groups_flat)
{
    rocke_ir_builder_t* b = plan->b;
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c_block_k = rocke_b_const_i32(b, plan->block_k);

    int n_a = plan->a_vecs_per_thread;
    int n_b_per = plan->b_vecs_per_thread;
    int total_acc = 0;
    for(int g = 0; g < num_operands; ++g)
    {
        total_acc += group_sizes[g];
    }

    /* prefetch tile 0. */
    rocke_value_t* a_pre0[ROCKE_MOE_MAX_VECS];
    rocke_value_t* b_pre0[ROCKE_MOE_MAX_OPERANDS * ROCKE_MOE_MAX_VECS];
    rocke_moe_emit_global_load(
        plan, a_view, a_mn_origin, operands, num_operands, b_mn_origin, c0, a_pre0, b_pre0);

    /* Build the loop-carried iter-args: accumulators, then A prefetch, then B
     * prefetch (per operand). */
    rocke_iter_arg_t iter_args[ROCKE_MOE_MAX_ITER_ARGS];
    int n_ia = 0;
    char names[ROCKE_MOE_MAX_ITER_ARGS][32]; /* "b%d_pre%d" worst case = 28 bytes */
    for(int j = 0; j < total_acc; ++j)
    {
        /* Python carries the accumulator SSA name from the acc_groups
         * (name, init) tuples (e.g. "gate_acc_m0_n0" / "up_acc_m0_n0" /
         * "down_acc_m0_n0" / "gu_acc_m0_n0"). Use the caller-supplied name so
         * the loop-carried phi names are byte-identical to Python; fall back to
         * the generic "acc%d" only when no names are provided. */
        if(acc_names_flat != NULL && acc_names_flat[j] != NULL)
        {
            iter_args[n_ia].name = acc_names_flat[j];
        }
        else
        {
            snprintf(names[n_ia], sizeof(names[0]), "acc%d", j);
            iter_args[n_ia].name = names[n_ia];
        }
        iter_args[n_ia].init = acc_inits_flat[j];
        n_ia++;
    }
    for(int i = 0; i < n_a; ++i)
    {
        snprintf(names[n_ia], sizeof(names[0]), "a_pre%d", i);
        iter_args[n_ia].name = names[n_ia];
        iter_args[n_ia].init = a_pre0[i];
        n_ia++;
    }
    for(int gi = 0; gi < num_operands; ++gi)
    {
        for(int i = 0; i < n_b_per; ++i)
        {
            snprintf(names[n_ia], sizeof(names[0]), "b%d_pre%d", gi, i);
            iter_args[n_ia].name = names[n_ia];
            iter_args[n_ia].init = b_pre0[(size_t)gi * n_b_per + i];
            n_ia++;
        }
    }

    rocke_for_t for_op
        = rocke_b_scf_for_iter(b, c0, K, c_block_k, iter_args, n_ia, "k0", false, true);
    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* k0 = for_op.iv;
        rocke_value_t** iv = for_op.iter_vars;
        int off = 0;
        /* cur accumulator groups (pointers into iv). */
        rocke_value_t* cur_groups_storage[ROCKE_MOE_MAX_OPERANDS][ROCKE_MOE_MAX_MFMAS] = {{0}};
        rocke_value_t* const* cur_groups[ROCKE_MOE_MAX_OPERANDS] = {0};
        for(int g = 0; g < num_operands; ++g)
        {
            for(int j = 0; j < group_sizes[g]; ++j)
            {
                cur_groups_storage[g][j] = iv[off + j];
            }
            cur_groups[g] = cur_groups_storage[g];
            off += group_sizes[g];
        }
        rocke_value_t* a_regs[ROCKE_MOE_MAX_VECS] = {0};
        for(int i = 0; i < n_a; ++i)
        {
            a_regs[i] = iv[off + i];
        }
        off += n_a;
        rocke_value_t* b_reg_groups[ROCKE_MOE_MAX_OPERANDS * ROCKE_MOE_MAX_VECS] = {0};
        for(int gi = 0; gi < num_operands; ++gi)
        {
            for(int i = 0; i < n_b_per; ++i)
            {
                b_reg_groups[(size_t)gi * n_b_per + i] = iv[off + i];
            }
            off += n_b_per;
        }

        rocke_moe_emit_lds_store(plan, a_lds_view, a_regs, operands, num_operands, b_reg_groups);
        rocke_b_sync(b);
        rocke_value_t* k_next = rocke_b_add(b, k0, c_block_k);
        rocke_value_t* k_clamped = rocke_b_select(b, rocke_b_cmp_lt(b, k_next, K), k_next, k0);
        rocke_value_t* a_next[ROCKE_MOE_MAX_VECS];
        rocke_value_t* b_next[ROCKE_MOE_MAX_OPERANDS * ROCKE_MOE_MAX_VECS];
        rocke_moe_emit_global_load(plan,
                                   a_view,
                                   a_mn_origin,
                                   operands,
                                   num_operands,
                                   b_mn_origin,
                                   k_clamped,
                                   a_next,
                                   b_next);

        rocke_value_t* new_groups_flat[2 * ROCKE_MOE_MAX_MFMAS];
        rocke_moe_emit_mfma_phase(plan,
                                  a_smem,
                                  operands,
                                  num_operands,
                                  cur_groups,
                                  group_sizes,
                                  warp_m_idx,
                                  warp_n_idx,
                                  lane,
                                  sched_groups,
                                  new_groups_flat);
        rocke_b_sync(b);

        rocke_value_t* yielded[ROCKE_MOE_MAX_ITER_ARGS];
        int ny = 0;
        for(int j = 0; j < total_acc; ++j)
        {
            yielded[ny++] = new_groups_flat[j];
        }
        for(int i = 0; i < n_a; ++i)
        {
            yielded[ny++] = a_next[i];
        }
        for(int gi = 0; gi < num_operands; ++gi)
        {
            for(int i = 0; i < n_b_per; ++i)
            {
                yielded[ny++] = b_next[(size_t)gi * n_b_per + i];
            }
        }
        rocke_b_scf_yield(b, yielded, ny);
    }
    rocke_b_region_leave(b);

    /* Pull the final accumulator groups (the first total_acc results). */
    for(int j = 0; j < total_acc; ++j)
    {
        out_groups_flat[j] = (for_op.op != NULL) ? for_op.op->results[j] : NULL;
    }
    return 1;
}

/* ====================================================================== *
 *  Epilogues
 * ====================================================================== */

/* _y_x_stride(encoding, y_idx): stride a Y dim takes in its target X dim, or 1
 * for an R-mapped Y (major == 0). Mirrors distribution.py::_y_x_stride. */
static int rocke_moe_y_x_stride(const rocke_tile_distribution_encoding_t* enc, int y_idx)
{
    int major = enc->Ys_major[y_idx];
    int minor = enc->Ys_minor[y_idx];
    if(major == 0)
    {
        return 1;
    }
    const rocke_h_row_t* h = &enc->Hs[major - 1];
    int stride = 1;
    for(int level = minor + 1; level < h->count; ++level)
    {
        stride *= h->levels[level];
    }
    return stride;
}

/* Y_lengths[y_idx]: the bucket length the Y maps to (Hs[major-1][minor], or
 * Rs[minor] when major == 0). Mirrors TileDistributionEncoding.Y_lengths. */
static int rocke_moe_y_length(const rocke_tile_distribution_encoding_t* enc, int y_idx)
{
    int major = enc->Ys_major[y_idx];
    int minor = enc->Ys_minor[y_idx];
    if(major == 0)
    {
        return enc->Rs[minor];
    }
    return enc->Hs[major - 1].levels[minor];
}

/* make_load_store_traits picker (distribution.py::make_load_store_traits) for
 * the cshuffle store. Sets *out_vector_dim_y / *out_spv (max_vec=8, min_vec=1). */
static void rocke_moe_load_store_traits(const rocke_tile_distribution_encoding_t* enc,
                                        int* out_vector_dim_y,
                                        int* out_spv)
{
    int num_Y = enc->num_Y;
    /* stride-1 candidates: largest length wins, ties to highest Y index
     * (Python sorts by (length, y_idx) and takes the last). */
    int best_idx = -1;
    int best_len = -1;
    for(int y = 0; y < num_Y; ++y)
    {
        if(rocke_moe_y_x_stride(enc, y) == 1)
        {
            int len = rocke_moe_y_length(enc, y);
            if(len > best_len || (len == best_len && y > best_idx))
            {
                best_len = len;
                best_idx = y;
            }
        }
    }
    if(best_idx >= 0)
    {
        int full_len = best_len;
        int spv = (full_len < 8) ? full_len : 8;
        while(spv > 1 && (full_len % spv != 0 || (spv & (spv - 1)) != 0))
        {
            spv /= 2;
        }
        if(spv < 1)
        {
            spv = 1;
        }
        *out_vector_dim_y = best_idx;
        *out_spv = spv;
    }
    else
    {
        *out_vector_dim_y = num_Y - 1;
        *out_spv = 1;
    }
}

void rocke_moe_emit_cshuffle_stage(rocke_ir_builder_t* b,
                                   const rocke_gemm_universal_spec_t* spec,
                                   const rocke_moe_cwarp_decode_t* cdec,
                                   rocke_value_t* smem,
                                   const rocke_type_t* storage_dtype,
                                   int c_per_lane,
                                   rocke_moe_cell_value_fn cell_value,
                                   void* cell_user)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);

    /* Byte-faithful port of _emit_cshuffle_stage + store_tile_cshuffle: stage
     * each warp atom's MFMA accumulators into LDS via the C-warp tile
     * distribution's space-filling-curve (snake) store walk. For each (mi, ni):
     *   1. materialise the per-lane slot results into a StaticDistributedTensor
     *      (slot i = y0*m1 + y1, i in 0..c_per_lane, row-major) via cell_value;
     *      the cell IR is emitted in plain i-order, matching the Python dt.set
     *      loop, so the SiLU/cast ops are byte-identical;
     *   2. walk traits.iterate_accesses() (snake SFC over the non-vector Y
     *      dims); per access, for k in 0..scalar_per_vector, read the staged
     *      slot and emit b.smem_store_vN(smem, coord_fn(y_base, k), scalar, 1).
     * coord_fn(y_base, k) = cdec.coords(mi, ni, y_base[0]*m1 + k) mirrors the
     * Python _coord closure. This reproduces the exact ds_write order of the
     * cshuffle walk (the prior C body emitted a plain i-order tile-window
     * scatter at addrspace(1)); smem_store_vN targets LDS directly. */
    const rocke_tile_distribution_t* dist = cdec->dist;
    const rocke_tile_distribution_encoding_t* enc = dist->encoding;
    int m1 = cdec->m1;
    int num_Y = enc->num_Y;

    (void)storage_dtype;

    /* Python builds, ONCE before the (mi, ni) loop:
     *   lds_view  = TensorView(base=smem, desc=packed([tile_m, tile_n]), lds)
     *   z         = (b.const_i32(0), b.const_i32(0))
     *   lds_window= make_tile_window(lds_view, (tile_m, tile_n), origin=z)
     * store_tile_cshuffle then stores through lds_window.view.base (== smem)
     * via b.smem_store_vN. The Python lds_view / make_tile_window are pure
     * host-side bookkeeping (no IR), but the two `z` const_i32(0) DO advance the
     * SSA value counter (they are folded, hence never printed). Emit exactly
     * those two consts here -- and nothing else -- so the epilogue value
     * numbering stays byte-identical to Python. (Do NOT call the C
     * rocke_make_tile_window: it allocates two extra builder ids the Python free
     * function does not.) The store targets `smem` directly, which equals
     * lds_window.view.base. */
    rocke_value_t* lds_base = smem;
    (void)rocke_b_const_i32(b, 0);
    (void)rocke_b_const_i32(b, 0);

    int vector_dim_y = 0;
    int spv = 1;
    rocke_moe_load_store_traits(enc, &vector_dim_y, &spv);

    /* Y lengths + the non-vector ("outer") axis lengths in Y-index order. */
    int y_len[8];
    for(int y = 0; y < num_Y; ++y)
    {
        y_len[y] = rocke_moe_y_length(enc, y);
    }
    int outer_axis[8];
    int outer_len[8];
    int num_outer = 0;
    for(int y = 0; y < num_Y; ++y)
    {
        if(y != vector_dim_y)
        {
            outer_axis[num_outer] = y;
            outer_len[num_outer] = y_len[y];
            num_outer++;
        }
    }
    (void)outer_axis;
    int num_access = 1;
    for(int o = 0; o < num_outer; ++o)
    {
        num_access *= outer_len[o];
    }

    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            /* 1) make_static_distributed_tensor(dist, storage_dtype); fill
             * slot i in row-major i-order (dt.set([i//m1, i%m1], cell)). */
            rocke_static_distributed_tensor_t* dt
                = rocke_make_static_distributed_tensor(b, dist, storage_dtype);
            if(dt == NULL)
            {
                return;
            }
            for(int i = 0; i < c_per_lane; ++i)
            {
                dt->storage[i] = cell_value(mi, ni, i, cell_user);
            }

            /* 2) Snake SFC store walk over the non-vector Y dims. */
            for(int a = 0; a < num_access; ++a)
            {
                /* Row-major outer tuple (axis 0 slowest), then Gray-code fold:
                 * reverse axis i when the parity of the sum of slower axes is
                 * odd (LoadStoreTraits.iterate_accesses, snake=True). */
                int folded[8];
                int rem = a;
                for(int o = num_outer - 1; o >= 0; --o)
                {
                    folded[o] = rem % outer_len[o];
                    rem /= outer_len[o];
                }
                for(int axis = 1; axis < num_outer; ++axis)
                {
                    int parity = 0;
                    for(int s = 0; s < axis; ++s)
                    {
                        parity += folded[s];
                    }
                    if(parity % 2 == 1)
                    {
                        folded[axis] = outer_len[axis] - 1 - folded[axis];
                    }
                }
                /* Splice the vector-dim slot back in (at 0): full Y-base. */
                int y_base[8];
                int oi = 0;
                for(int y = 0; y < num_Y; ++y)
                {
                    y_base[y] = (y == vector_dim_y) ? 0 : folded[oi++];
                }

                for(int k = 0; k < spv; ++k)
                {
                    /* y_full = y_base with vector_dim_y := k; storage slot is the
                     * row-major linearisation of y_full over Y_lengths. */
                    int slot = 0;
                    for(int y = 0; y < num_Y; ++y)
                    {
                        int yi = (y == vector_dim_y) ? k : y_base[y];
                        slot = slot * y_len[y] + yi;
                    }
                    rocke_value_t* scalar = dt->storage[slot];
                    /* coord_fn(y_base, k): slot = y_base[0]*m1 + k. */
                    int coord_slot = y_base[0] * m1 + k;
                    rocke_value_t* ld_m = NULL;
                    rocke_value_t* ld_n = NULL;
                    rocke_moe_cwarp_decode_coords(cdec, mi, ni, coord_slot, &ld_m, &ld_n);
                    rocke_value_t* coords[2] = {ld_m, ld_n};
                    rocke_b_smem_store_vN(b, lds_base, coords, 2, scalar, 1);
                }
            }
        }
    }
}

void rocke_moe_emit_down_reduce_epilogue_atomic(rocke_ir_builder_t* b,
                                                const rocke_gemm_universal_spec_t* spec,
                                                rocke_value_t* const* accs,
                                                rocke_value_t* warp_m_idx,
                                                rocke_value_t* warp_n_idx,
                                                rocke_value_t* lane,
                                                rocke_value_t* block_m_off,
                                                rocke_value_t* block_n_off,
                                                rocke_value_t* M,
                                                rocke_value_t* N,
                                                rocke_value_t* SortedTokenIds,
                                                rocke_value_t* SortedWeights,
                                                rocke_value_t* Y,
                                                int c_per_lane,
                                                rocke_value_t* batch_bucket_off,
                                                rocke_value_t* tokens)
{
    const rocke_gemm_tile_spec_t* t = &spec->tile;
    int mfmas_m = rocke_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n = rocke_gemm_tile_mfmas_per_warp_n(t);
    rocke_value_t* warp_m_off
        = rocke_b_mul(b, warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    rocke_value_t* warp_n_off
        = rocke_b_mul(b, warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));
    bool pad_m = spec->trait.pad_m;
    bool pad_n = spec->trait.pad_n;

    rocke_moe_cwarp_decode_t cdec;
    if(!rocke_moe_cwarp_decode_init(&cdec, b, spec, warp_m_off, warp_n_off, lane))
    {
        return;
    }
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        /* Per-mi c_n list (one per ni); i-independent (hoisted). */
        rocke_value_t* c_ns[ROCKE_MOE_MAX_MFMAS];
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            c_ns[ni] = rocke_b_add(b, block_n_off, rocke_moe_cwarp_decode_warp_col(&cdec, ni));
        }
        for(int i = 0; i < c_per_lane; ++i)
        {
            rocke_value_t* c_m
                = rocke_b_add(b, block_m_off, rocke_moe_cwarp_decode_warp_row(&cdec, mi, i));
            /* emit_one_row: hoist token+weight load out of the ni loop. */
            rocke_if_t guard_m;
            bool have_guard_m = pad_m;
            if(have_guard_m)
            {
                guard_m = rocke_b_scf_if(b, rocke_b_cmp_lt(b, c_m, M));
                rocke_b_region_enter(b, guard_m.then_region);
            }
            {
                rocke_value_t* bucket = rocke_b_add(b, batch_bucket_off, c_m);
                rocke_value_t* token = rocke_b_global_load_i32(b, SortedTokenIds, bucket, 0);
                /* valid = b.land(b.cmp_ge(token, 0), b.cmp_lt(token, tokens))
                 * Python evaluates the land operands left-to-right, so cmp_ge is
                 * emitted BEFORE cmp_lt. C function-argument evaluation order is
                 * unspecified, so sequence the two compares into locals (ge
                 * first) to keep the emitted SSA order byte-identical. */
                rocke_value_t* tok_ge0 = rocke_b_cmp_ge(b, token, rocke_b_const_i32(b, 0));
                rocke_value_t* tok_lt = rocke_b_cmp_lt(b, token, tokens);
                rocke_value_t* valid = rocke_b_land(b, tok_ge0, tok_lt);
                rocke_if_t vguard = rocke_b_scf_if(b, valid);
                rocke_b_region_enter(b, vguard.then_region);
                {
                    rocke_value_t* w = rocke_b_global_load_f32(b, SortedWeights, bucket, 0);
                    for(int ni = 0; ni < mfmas_n; ++ni)
                    {
                        rocke_value_t* acc = accs[mi * mfmas_n + ni];
                        rocke_value_t* v = rocke_b_vec_extract(b, acc, i);
                        rocke_value_t* contrib = rocke_b_fmul(b, w, v);
                        rocke_value_t* y_off = rocke_b_add(b, rocke_b_mul(b, token, N), c_ns[ni]);
                        /* Python: b.global_atomic_add(Y, y_off, contrib) -- the
                         * generic memref.global_atomic_add op (plain "monotonic"
                         * f32 atomicrmw + the AMDGPU fp-atomic metadata, no
                         * syncscope("agent") / align 4). The _f32 builder emits a
                         * DIFFERENT form (syncscope agent, align 4); use the
                         * generic builder with ordering=NULL (=> monotonic). */
                        if(pad_n)
                        {
                            rocke_if_t ng = rocke_b_scf_if(b, rocke_b_cmp_lt(b, c_ns[ni], N));
                            rocke_b_region_enter(b, ng.then_region);
                            rocke_b_global_atomic_add(b, Y, y_off, contrib, NULL);
                            rocke_b_region_leave(b);
                        }
                        else
                        {
                            rocke_b_global_atomic_add(b, Y, y_off, contrib, NULL);
                        }
                    }
                }
                rocke_b_region_leave(b); /* vguard */
            }
            if(have_guard_m)
            {
                rocke_b_region_leave(b); /* guard_m */
            }
        }
    }
}

/* ====================================================================== *
 *  Spec types
 * ====================================================================== */

/* _data_spec: "f16"/"fp16" -> "fp16", else pass through. Builds a homogeneous
 * A/B/C data spec on top of the universal default (fp32 acc, RCR layout). */
static rocke_gemm_data_spec_t rocke_moe_data_spec(const char* dtype)
{
    rocke_gemm_universal_spec_t base = rocke_gemm_universal_spec_default();
    rocke_gemm_data_spec_t d = base.data;
    const char* dt = (dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
                         ? "fp16"
                         : (dtype != NULL ? dtype : "fp16");
    d.dtype_a = dt;
    d.dtype_b = dt;
    d.dtype_c = dt;
    return d;
}

static rocke_gemm_universal_spec_t rocke_moe_to_universal(const char* name,
                                                          const rocke_gemm_tile_spec_t* tile,
                                                          const rocke_gemm_trait_spec_t* trait,
                                                          int wave_size,
                                                          int block_size,
                                                          const char* dtype)
{
    rocke_gemm_universal_spec_t u = rocke_gemm_universal_spec_default();
    u.name = name;
    u.tile = *tile;
    u.trait = *trait;
    u.data = rocke_moe_data_spec(dtype);
    u.wave_size = wave_size;
    u.block_size = block_size;
    u.batched = true;
    return u;
}

static void
    rocke_moe_finalize_block_size(int* block_size, const rocke_gemm_tile_spec_t* t, int wave_size)
{
    if(*block_size == 0)
    {
        *block_size = t->warp_m * t->warp_n * t->warp_k * wave_size;
    }
}

static rocke_gemm_trait_spec_t rocke_moe_default_trait(void)
{
    /* TraitSpec(epilogue="default") (the spec field default_factory). */
    rocke_gemm_universal_spec_t base = rocke_gemm_universal_spec_default();
    rocke_gemm_trait_spec_t tr = base.trait;
    tr.epilogue = "default";
    return tr;
}

/* ---- helper: append a static suffix, replacing '/' (matches kernel_name). */
static rocke_status_t rocke_moe_kernel_name_with_suffix(const rocke_gemm_universal_spec_t* u,
                                                        const char* suffix,
                                                        char* out,
                                                        size_t out_cap)
{
    char base[1024];
    rocke_status_t st = rocke_gemm_universal_kernel_name(u, base, sizeof(base));
    if(st != ROCKE_OK)
    {
        return st;
    }
    int n = snprintf(out, out_cap, "%s%s", base, suffix);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* ----------------------------- FusedGateUpSiluGemmSpec ---------------------- */

rocke_moe_gate_up_silu_gemm_spec_t rocke_moe_gate_up_silu_gemm_spec_default(void)
{
    rocke_moe_gate_up_silu_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = NULL;
    s.tile = rocke_gemm_universal_spec_default().tile;
    s.trait = rocke_moe_default_trait();
    s.wave_size = 64;
    s.block_size = 0;
    s.dtype = "fp16";
    s.grouped = false;
    return s;
}

void rocke_moe_gate_up_silu_gemm_spec_finalize(rocke_moe_gate_up_silu_gemm_spec_t* spec)
{
    rocke_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

rocke_gemm_universal_spec_t
    rocke_moe_gate_up_silu_gemm_spec_to_universal(const rocke_moe_gate_up_silu_gemm_spec_t* spec)
{
    return rocke_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

rocke_status_t rocke_moe_gate_up_silu_gemm_spec_kernel_name(
    const rocke_moe_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u = rocke_moe_gate_up_silu_gemm_spec_to_universal(spec);
    const char* suffix = spec->grouped ? "_gate_up_silu_grouped" : "_gate_up_silu";
    return rocke_moe_kernel_name_with_suffix(&u, suffix, out, out_cap);
}

/* -------------------- FusedInterleavedGateUpSiluGemmSpec -------------------- */

rocke_moe_interleaved_gate_up_silu_gemm_spec_t
    rocke_moe_interleaved_gate_up_silu_gemm_spec_default(void)
{
    rocke_moe_interleaved_gate_up_silu_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = NULL;
    s.tile = rocke_gemm_universal_spec_default().tile;
    s.trait = rocke_moe_default_trait();
    s.wave_size = 64;
    s.block_size = 0;
    s.dtype = "fp16";
    s.grouped = false;
    return s;
}

void rocke_moe_interleaved_gate_up_silu_gemm_spec_finalize(
    rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec)
{
    rocke_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

rocke_gemm_universal_spec_t rocke_moe_interleaved_gate_up_silu_gemm_spec_to_universal(
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec)
{
    return rocke_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

rocke_status_t rocke_moe_interleaved_gate_up_silu_gemm_spec_kernel_name(
    const rocke_moe_interleaved_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u = rocke_moe_interleaved_gate_up_silu_gemm_spec_to_universal(spec);
    const char* suffix
        = spec->grouped ? "_interleaved_gate_up_silu_grouped" : "_interleaved_gate_up_silu";
    return rocke_moe_kernel_name_with_suffix(&u, suffix, out, out_cap);
}

/* --------------------------- FusedDownReduceGemmSpec ------------------------ */

rocke_moe_down_reduce_gemm_spec_t rocke_moe_down_reduce_gemm_spec_default(void)
{
    rocke_moe_down_reduce_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = NULL;
    s.tile = rocke_gemm_universal_spec_default().tile;
    s.trait = rocke_moe_default_trait();
    s.wave_size = 64;
    s.block_size = 0;
    s.dtype = "fp16";
    s.grouped = false;
    return s;
}

void rocke_moe_down_reduce_gemm_spec_finalize(rocke_moe_down_reduce_gemm_spec_t* spec)
{
    rocke_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

rocke_gemm_universal_spec_t
    rocke_moe_down_reduce_gemm_spec_to_universal(const rocke_moe_down_reduce_gemm_spec_t* spec)
{
    return rocke_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

rocke_status_t rocke_moe_down_reduce_gemm_spec_kernel_name(
    const rocke_moe_down_reduce_gemm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u = rocke_moe_down_reduce_gemm_spec_to_universal(spec);
    const char* suffix = spec->grouped ? "_down_reduce_grouped" : "_down_reduce";
    return rocke_moe_kernel_name_with_suffix(&u, suffix, out, out_cap);
}

/* ------------------------- FusedDownSiluReduceGemmSpec ---------------------- */

rocke_moe_down_silu_reduce_gemm_spec_t rocke_moe_down_silu_reduce_gemm_spec_default(void)
{
    rocke_moe_down_silu_reduce_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name = NULL;
    s.tile = rocke_gemm_universal_spec_default().tile;
    s.trait = rocke_moe_default_trait();
    s.wave_size = 64;
    s.block_size = 0;
    s.dtype = "fp16";
    return s;
}

void rocke_moe_down_silu_reduce_gemm_spec_finalize(rocke_moe_down_silu_reduce_gemm_spec_t* spec)
{
    rocke_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

rocke_gemm_universal_spec_t rocke_moe_down_silu_reduce_gemm_spec_to_universal(
    const rocke_moe_down_silu_reduce_gemm_spec_t* spec)
{
    return rocke_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

rocke_status_t rocke_moe_down_silu_reduce_gemm_spec_kernel_name(
    const rocke_moe_down_silu_reduce_gemm_spec_t* spec, char* out, size_t out_cap)
{
    rocke_gemm_universal_spec_t u = rocke_moe_down_silu_reduce_gemm_spec_to_universal(spec);
    return rocke_moe_kernel_name_with_suffix(&u, "_down_silu_reduce", out, out_cap);
}
