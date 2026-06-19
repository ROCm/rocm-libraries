// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of ck_dsl/instances/common/moe_gemm_fused.py (the requested symbol
 * subset). See the header for the symbol mapping. Byte-faithful builder-call
 * order against ckc/ir.h + the sibling helper_*.h / instance_gemm_*.h.
 */
#include "ckc/helper_ck_dsl.instances.common.moe_gemm_fused.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ckc/ir_internal.h"            /* ckc_i_set_err                              */
#include "ckc/instance_gemm_internal.h" /* _emit_mfma / _emit_smem_load / ... */

/* Magic-division helpers from helper_ck_dsl.helpers.transforms.h. That header
 * is NOT included here: it defines a rich `ckc_tensor_descriptor` struct that
 * collides with the (different) `ckc_tensor_descriptor` in tensor_view.h, which
 * this module needs for TensorView / TileWindow. The two cannot coexist in one
 * TU (see instance_permute_nd.c for the same constraint), so we forward-declare
 * just the two pure magic-division entry points we use. */
/* WS3 C++ build: these are cross-TU C-ABI helpers (defined with C linkage in
 * helpers/transforms.c); the forward decls must be extern "C" so C++ does not
 * mangle the references. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
bool ckc_calculate_magic_numbers(ckc_ir_builder_t* b,
                                 int divisor,
                                 uint64_t* out_multiplier,
                                 int* out_shift);
ckc_value_t*
ckc_do_magic_division(ckc_ir_builder_t* b, ckc_value_t* dividend, uint64_t multiplier, int shift);
#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ guards */

#define CKC_MOE_MAX_MFMAS 256
#define CKC_MOE_MAX_VECS 256
#define CKC_MOE_MAX_OPERANDS 4
#define CKC_MOE_MAX_ITER_ARGS (2 * CKC_MOE_MAX_MFMAS + CKC_MOE_MAX_OPERANDS * CKC_MOE_MAX_VECS)

/* _storage_dtype(spec): homogeneous A/B/C dtype -> ckc_type_t. Mirrors the
 * gemm_universal helper (the public C build uses an internal static); here we
 * re-derive from the spec's data dtype string. */
static const ckc_type_t* ckc_moe_storage_dtype(const ckc_gemm_universal_spec_t* u)
{
    const char* d = u->data.dtype_a;
    if(d == NULL)
    {
        return ckc_f16();
    }
    if(strcmp(d, "f16") == 0 || strcmp(d, "fp16") == 0)
    {
        return ckc_f16();
    }
    if(strcmp(d, "bf16") == 0)
    {
        return ckc_bf16();
    }
    return ckc_scalar_by_name(d);
}

/* _mfma_atom_widths(spec) -> (a_per_lane, b_per_lane, c_per_lane). MFMA-only
 * geometry: the warp-tile atom's per-lane fragment widths. */
static void
ckc_moe_mfma_atom_widths(const ckc_gemm_universal_spec_t* u, int* a_per, int* b_per, int* c_per)
{
    const ckc_gemm_tile_spec_t* t = &u->tile;
    const ckc_mfma_atom_t* atom =
        ckc_mfma_atom(u->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    int wm   = t->warp_tile_m;
    int wn   = t->warp_tile_n;
    int wk   = t->warp_tile_k;
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

void ckc_moe_magic_div_mod(ckc_ir_builder_t* b,
                           ckc_value_t* dividend,
                           int divisor,
                           ckc_value_t** out_quot,
                           ckc_value_t** out_rem)
{
    if(divisor == 1)
    {
        *out_quot = dividend;
        *out_rem  = ckc_b_const_i32(b, 0);
        return;
    }
    uint64_t mult = 0;
    int shift     = 0;
    if(!ckc_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        *out_quot = NULL;
        *out_rem  = NULL;
        return;
    }
    ckc_value_t* quot = ckc_do_magic_division(b, dividend, mult, shift);
    ckc_value_t* rem  = ckc_b_sub(b, dividend, ckc_b_mul(b, quot, ckc_b_const_i32(b, divisor)));
    *out_quot         = quot;
    *out_rem          = rem;
}

void ckc_moe_vec_rowcol(ckc_ir_builder_t* b,
                        int e,
                        ckc_value_t* tid,
                        ckc_value_t* c_threads,
                        int block_k_div_vec,
                        ckc_value_t* c_load_vec,
                        int load_vec,
                        ckc_value_t** out_row,
                        ckc_value_t** out_col)
{
    ckc_value_t* vec_idx = ckc_b_add(b, ckc_b_mul(b, ckc_b_const_i32(b, e), c_threads), tid);
    ckc_value_t* row     = NULL;
    ckc_value_t* col_v   = NULL;
    ckc_moe_magic_div_mod(b, vec_idx, block_k_div_vec, &row, &col_v);
    *out_row = row;
    *out_col = (load_vec > 1) ? ckc_b_mul(b, col_v, c_load_vec) : col_v;
}

ckc_value_t* ckc_moe_gemm_fused_silu_mul_f32(ckc_ir_builder_t* b,
                                             ckc_value_t* g,
                                             ckc_value_t* u,
                                             ckc_value_t* one_f32,
                                             ckc_value_t* c_neg_log2e)
{
    ckc_value_t* sig =
        ckc_b_rcp(b, ckc_b_fadd(b, one_f32, ckc_b_exp2(b, ckc_b_fmul(b, c_neg_log2e, g))));
    ckc_value_t* silu = ckc_b_fmul(b, g, sig);
    return ckc_b_fmul(b, silu, u);
}

ckc_value_t* ckc_moe_pad_in_bounds(ckc_ir_builder_t* b,
                                   ckc_value_t* c_m,
                                   ckc_value_t* c_n,
                                   ckc_value_t* M,
                                   ckc_value_t* N,
                                   bool pad_m,
                                   bool pad_n,
                                   int vec)
{
    if(!(pad_m || pad_n))
    {
        return NULL;
    }
    ckc_value_t* checks[2];
    int nc = 0;
    if(pad_m)
    {
        checks[nc++] = ckc_b_cmp_lt(b, c_m, M);
    }
    if(pad_n)
    {
        if(vec == 1)
        {
            checks[nc++] = ckc_b_cmp_lt(b, c_n, N);
        }
        else
        {
            ckc_value_t* c_n_last = ckc_b_add(b, c_n, ckc_b_const_i32(b, vec - 1));
            checks[nc++]          = ckc_b_cmp_lt(b, c_n_last, N);
        }
    }
    return (nc == 1) ? checks[0] : ckc_b_land(b, checks[0], checks[1]);
}

/* ====================================================================== *
 *  _CWarpDecode
 * ====================================================================== */

int ckc_moe_cwarp_decode_init(ckc_moe_cwarp_decode_t* out,
                              ckc_ir_builder_t* b,
                              const ckc_gemm_universal_spec_t* spec,
                              ckc_value_t* warp_m_off,
                              ckc_value_t* warp_n_off,
                              ckc_value_t* lane)
{
    const ckc_gemm_tile_spec_t* t = &spec->tile;
    const ckc_mfma_atom_t* atom =
        ckc_mfma_atom(spec->data.dtype_a, t->warp_tile_m, t->warp_tile_n, t->warp_tile_k);
    if(atom == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "_CWarpDecode: no MFMA atom for warp tile");
        return 0;
    }
    const ckc_tile_distribution_encoding_t* enc = ckc_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
    {
        return 0;
    }
    const ckc_tile_distribution_t* dist = ckc_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return 0;
    }
    out->b    = b;
    out->spec = spec;
    out->dist = dist;
    /* kCM1PerLane is Hs[0][2]; kCNLane is Hs[1][0]. */
    out->m1               = enc->Hs[0].levels[2];
    int n_lane            = enc->Hs[1].levels[0];
    ckc_value_t* c_n_lane = ckc_b_const_i32(b, n_lane);
    out->n_in_atom        = ckc_b_mod(b, lane, c_n_lane);
    out->m_blk            = ckc_b_div(b, lane, c_n_lane);
    out->warp_m_off       = warp_m_off;
    out->warp_n_off       = warp_n_off;
    return 1;
}

/* _row_col_in_atom(i): calculate_x over (y0, y1) = (i // m1, i % m1). */
static void ckc_moe_cwarp_row_col_in_atom(const ckc_moe_cwarp_decode_t* d,
                                          int i,
                                          ckc_value_t** out_row,
                                          ckc_value_t** out_col)
{
    ckc_ir_builder_t* b = d->b;
    ckc_value_t* y0     = ckc_b_const_i32(b, i / d->m1);
    ckc_value_t* y1     = ckc_b_const_i32(b, i % d->m1);
    ckc_value_t* ys[2]  = {y0, y1};
    /* ps = [[m_blk, n_in_atom]] (one P dim with two contributions). */
    ckc_value_t* p0[2]        = {d->m_blk, d->n_in_atom};
    ckc_value_t* const* ps[1] = {p0};
    int ps_counts[1]          = {2};
    ckc_value_t* x_out[2]     = {NULL, NULL};
    if(!ckc_tile_distribution_calculate_x(b, d->dist, ys, 2, ps, ps_counts, 1, x_out, 2))
    {
        *out_row = NULL;
        *out_col = NULL;
        return;
    }
    *out_row = x_out[0];
    *out_col = x_out[1];
}

void ckc_moe_cwarp_decode_coords(const ckc_moe_cwarp_decode_t* d,
                                 int mi,
                                 int ni,
                                 int i,
                                 ckc_value_t** out_ld_m,
                                 ckc_value_t** out_ld_n)
{
    ckc_ir_builder_t* b           = d->b;
    const ckc_gemm_tile_spec_t* t = &d->spec->tile;
    ckc_value_t* row_in_atom      = NULL;
    ckc_value_t* col_in_atom      = NULL;
    ckc_moe_cwarp_row_col_in_atom(d, i, &row_in_atom, &col_in_atom);
    *out_ld_m = ckc_b_add(
        b, d->warp_m_off, ckc_b_add(b, ckc_b_const_i32(b, mi * t->warp_tile_m), row_in_atom));
    *out_ld_n = ckc_b_add(
        b, d->warp_n_off, ckc_b_add(b, ckc_b_const_i32(b, ni * t->warp_tile_n), col_in_atom));
}

ckc_value_t* ckc_moe_cwarp_decode_warp_row(const ckc_moe_cwarp_decode_t* d, int mi, int i)
{
    ckc_ir_builder_t* b           = d->b;
    const ckc_gemm_tile_spec_t* t = &d->spec->tile;
    ckc_value_t* row_in_atom      = NULL;
    ckc_value_t* col_in_atom      = NULL;
    ckc_moe_cwarp_row_col_in_atom(d, i, &row_in_atom, &col_in_atom);
    return ckc_b_add(
        b, d->warp_m_off, ckc_b_add(b, ckc_b_const_i32(b, mi * t->warp_tile_m), row_in_atom));
}

ckc_value_t* ckc_moe_cwarp_decode_warp_col(const ckc_moe_cwarp_decode_t* d, int ni)
{
    ckc_ir_builder_t* b           = d->b;
    const ckc_gemm_tile_spec_t* t = &d->spec->tile;
    ckc_value_t* row_in_atom      = NULL;
    ckc_value_t* col_in_atom      = NULL;
    ckc_moe_cwarp_row_col_in_atom(d, 0, &row_in_atom, &col_in_atom);
    return ckc_b_add(
        b, d->warp_n_off, ckc_b_add(b, ckc_b_const_i32(b, ni * t->warp_tile_n), col_in_atom));
}

/* ====================================================================== *
 *  _MoeKloopPlan
 * ====================================================================== */

int ckc_moe_kloop_plan_init(ckc_moe_kloop_plan_t* out,
                            ckc_ir_builder_t* b,
                            const ckc_gemm_universal_spec_t* u,
                            ckc_value_t* tid)
{
    const ckc_gemm_tile_spec_t* t = &u->tile;
    out->b                        = b;
    out->u                        = u;
    out->tid                      = tid;
    out->storage_dtype            = ckc_moe_storage_dtype(u);
    ckc_moe_mfma_atom_widths(u, &out->a_per_lane, &out->b_per_lane, &out->c_per_lane);
    out->block_m = t->tile_m;
    out->block_n = t->tile_n;
    out->block_k = t->tile_k;
    out->mfmas_m = ckc_gemm_tile_mfmas_per_warp_m(t);
    out->mfmas_n = ckc_gemm_tile_mfmas_per_warp_n(t);
    out->k_atoms = ckc_gemm_tile_k_atoms_per_tile_k(t);

    int threads  = u->block_size;
    int load_vec = 0;
    if(ckc_choose_load_vec(t->tile_m, t->tile_n, t->tile_k, u->block_size, &load_vec) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "_MoeKloopPlan: choose_load_vec failed");
        return 0;
    }
    out->threads           = threads;
    out->load_vec          = load_vec;
    out->a_vecs_per_thread = (out->block_m * out->block_k) / load_vec / threads;
    out->b_vecs_per_thread = (out->block_n * out->block_k) / load_vec / threads;
    out->c_threads         = ckc_b_const_i32(b, threads);
    out->c_load_vec        = ckc_b_const_i32(b, load_vec);
    out->block_k_div_vec   = out->block_k / load_vec;
    return 1;
}

static void ckc_moe_plan_rowcol(const ckc_moe_kloop_plan_t* plan,
                                int e,
                                ckc_value_t** out_row,
                                ckc_value_t** out_col)
{
    ckc_moe_vec_rowcol(plan->b,
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

void ckc_moe_emit_global_load(const ckc_moe_kloop_plan_t* plan,
                              const ckc_tensor_view_t* a_view,
                              ckc_value_t* const a_mn_origin[2],
                              const ckc_moe_operand_t* operands,
                              int num_operands,
                              ckc_value_t* const b_mn_origin[2],
                              ckc_value_t* k_off,
                              ckc_value_t** out_a_regs,
                              ckc_value_t** out_b_regs)
{
    ckc_ir_builder_t* b      = plan->b;
    ckc_value_t* a_origin[3] = {a_mn_origin[0], a_mn_origin[1], k_off};
    ckc_value_t* b_origin[3] = {b_mn_origin[0], b_mn_origin[1], k_off};
    int a_lengths[3]         = {1, plan->block_m, plan->block_k};
    int b_lengths[3]         = {1, plan->block_n, plan->block_k};

    ckc_tile_window_t a_global;
    if(ckc_make_tile_window(&a_global, a_view, a_lengths, a_origin, 3) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "_emit_moe_global_load: A window");
        return;
    }
    for(int e = 0; e < plan->a_vecs_per_thread; ++e)
    {
        ckc_value_t* row = NULL;
        ckc_value_t* col = NULL;
        ckc_moe_plan_rowcol(plan, e, &row, &col);
        /* Python builds the batch index inline at each load call site
         * (a_global.load_*(b, b.const_i32(0), row, col)) AFTER _rowcol, not from
         * a hoisted constant. Emit it here in the same order so the global SSA
         * value counter matches Python op-for-op. */
        ckc_value_t* idx[3] = {ckc_b_const_i32(b, 0), row, col};
        if(plan->load_vec == 1)
        {
            out_a_regs[e] = ckc_tile_window_load_scalar(b, &a_global, idx, 3);
        }
        else
        {
            out_a_regs[e] = ckc_tile_window_load_vec(b, &a_global, idx, 3, plan->load_vec);
        }
    }

    for(int g = 0; g < num_operands; ++g)
    {
        const ckc_moe_operand_t* op = &operands[g];
        ckc_value_t** regs          = out_b_regs + (size_t)g * plan->b_vecs_per_thread;
        if(op->load_b != NULL)
        {
            for(int e = 0; e < plan->b_vecs_per_thread; ++e)
            {
                ckc_value_t* row = NULL;
                ckc_value_t* col = NULL;
                ckc_moe_plan_rowcol(plan, e, &row, &col);
                regs[e] = op->load_b(b, e, k_off, row, col, op->load_b_user);
            }
        }
        else
        {
            ckc_tile_window_t b_global;
            if(ckc_make_tile_window(&b_global, op->global_view, b_lengths, b_origin, 3) != CKC_OK)
            {
                ckc_i_set_err(b, CKC_ERR_VALUE, "_emit_moe_global_load: B window");
                return;
            }
            for(int e = 0; e < plan->b_vecs_per_thread; ++e)
            {
                ckc_value_t* row = NULL;
                ckc_value_t* col = NULL;
                ckc_moe_plan_rowcol(plan, e, &row, &col);
                /* Fresh inline batch const per load (matches Python order). */
                ckc_value_t* idx[3] = {ckc_b_const_i32(b, 0), row, col};
                if(plan->load_vec == 1)
                {
                    regs[e] = ckc_tile_window_load_scalar(b, &b_global, idx, 3);
                }
                else
                {
                    regs[e] = ckc_tile_window_load_vec(b, &b_global, idx, 3, plan->load_vec);
                }
            }
        }
    }
}

void ckc_moe_emit_lds_store(const ckc_moe_kloop_plan_t* plan,
                            const ckc_tensor_view_t* a_lds_view,
                            ckc_value_t* const* a_regs,
                            const ckc_moe_operand_t* operands,
                            int num_operands,
                            ckc_value_t* const* b_reg_groups)
{
    ckc_ir_builder_t* b = plan->b;
    ckc_value_t* z[2]   = {ckc_b_const_i32(b, 0), ckc_b_const_i32(b, 0)};
    int a_lengths[2]    = {plan->block_m, plan->block_k};
    int b_lengths[2]    = {plan->block_n, plan->block_k};

    ckc_tile_window_t a_lds;
    if(ckc_make_tile_window(&a_lds, a_lds_view, a_lengths, z, 2) != CKC_OK)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "_emit_moe_lds_store: A lds window");
        return;
    }
    for(int e = 0; e < plan->a_vecs_per_thread; ++e)
    {
        ckc_value_t* row = NULL;
        ckc_value_t* col = NULL;
        ckc_moe_plan_rowcol(plan, e, &row, &col);
        ckc_value_t* idx[2] = {row, col};
        if(plan->load_vec == 1)
        {
            ckc_tile_window_store_scalar(b, &a_lds, idx, 2, a_regs[e], 0);
        }
        else
        {
            ckc_tile_window_store_vec(b, &a_lds, idx, 2, a_regs[e], plan->load_vec);
        }
    }
    for(int g = 0; g < num_operands; ++g)
    {
        const ckc_moe_operand_t* op = &operands[g];
        ckc_value_t* const* regs    = b_reg_groups + (size_t)g * plan->b_vecs_per_thread;
        ckc_tile_window_t b_lds;
        if(ckc_make_tile_window(&b_lds, op->lds_view, b_lengths, z, 2) != CKC_OK)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "_emit_moe_lds_store: B lds window");
            return;
        }
        for(int e = 0; e < plan->b_vecs_per_thread; ++e)
        {
            ckc_value_t* row = NULL;
            ckc_value_t* col = NULL;
            ckc_moe_plan_rowcol(plan, e, &row, &col);
            ckc_value_t* idx[2] = {row, col};
            if(plan->load_vec == 1 && op->store_scalar_ok)
            {
                ckc_tile_window_store_scalar(b, &b_lds, idx, 2, (ckc_value_t*)regs[e], 0);
            }
            else
            {
                ckc_tile_window_store_vec(b, &b_lds, idx, 2, (ckc_value_t*)regs[e], plan->load_vec);
            }
        }
    }
}

void ckc_moe_emit_mfma_phase(const ckc_moe_kloop_plan_t* plan,
                             ckc_value_t* a_smem,
                             const ckc_moe_operand_t* operands,
                             int num_operands,
                             ckc_value_t* const* const* acc_groups,
                             const int* group_sizes,
                             ckc_value_t* warp_m_idx,
                             ckc_value_t* warp_n_idx,
                             ckc_value_t* lane,
                             int sched_groups,
                             ckc_value_t** out_groups_flat)
{
    ckc_ir_builder_t* b           = plan->b;
    const ckc_gemm_tile_spec_t* t = &plan->u->tile;
    ckc_value_t* m_in_atom        = ckc_b_mod(b, lane, ckc_b_const_i32(b, t->warp_tile_m));
    ckc_value_t* k_blk            = ckc_b_div(b, lane, ckc_b_const_i32(b, t->warp_tile_m));
    ckc_value_t* n_in_atom        = ckc_b_mod(b, lane, ckc_b_const_i32(b, t->warp_tile_n));
    ckc_value_t* warp_m_off =
        ckc_b_mul(b, warp_m_idx, ckc_b_const_i32(b, plan->mfmas_m * t->warp_tile_m));
    ckc_value_t* warp_n_off =
        ckc_b_mul(b, warp_n_idx, ckc_b_const_i32(b, plan->mfmas_n * t->warp_tile_n));

    /* new_groups starts as a copy of acc_groups (flat). Lay out per-operand
     * offsets into out_groups_flat (same order as group_sizes). */
    int off_per_group[CKC_MOE_MAX_OPERANDS];
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
        ckc_value_t* col_mul  = ckc_b_mul(b, k_blk, ckc_b_const_i32(b, plan->a_per_lane));
        ckc_value_t* col_base = ckc_b_add(b, col_mul, ckc_b_const_i32(b, kk * t->warp_tile_k));
        ckc_value_t* a_rows[CKC_MOE_MAX_MFMAS];
        for(int mi = 0; mi < plan->mfmas_m; ++mi)
        {
            ckc_value_t* a_row = ckc_b_add(
                b, warp_m_off, ckc_b_add(b, ckc_b_const_i32(b, mi * t->warp_tile_m), m_in_atom));
            a_rows[mi] = ckc_gemm_emit_smem_load(
                b, a_smem, a_row, col_base, plan->a_per_lane, plan->storage_dtype);
        }
        /* B fragments: one column set per operand. */
        ckc_value_t* b_cols[CKC_MOE_MAX_OPERANDS][CKC_MOE_MAX_MFMAS];
        for(int gi = 0; gi < num_operands; ++gi)
        {
            for(int ni = 0; ni < plan->mfmas_n; ++ni)
            {
                ckc_value_t* b_row =
                    ckc_b_add(b,
                              warp_n_off,
                              ckc_b_add(b, ckc_b_const_i32(b, ni * t->warp_tile_n), n_in_atom));
                b_cols[gi][ni] = ckc_gemm_emit_smem_load(
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
                    int slot              = off_per_group[gi] + flat;
                    out_groups_flat[slot] = ckc_gemm_emit_mfma(
                        b, plan->u, a_rows[mi], b_cols[gi][ni], out_groups_flat[slot]);
                }
                flat++;
            }
        }
        if(sched_groups && (strcmp(plan->u->trait.pipeline, "compv3") == 0 ||
                            strcmp(plan->u->trait.pipeline, "compv4") == 0))
        {
            ckc_b_sched_group_barrier(b, 0x100, 1, 0);
            ckc_b_sched_group_barrier(b, 0x008, sched_groups, 0);
        }
    }
}

int ckc_moe_emit_prefetch_kloop(const ckc_moe_kloop_plan_t* plan,
                                const ckc_tensor_view_t* a_view,
                                const ckc_tensor_view_t* a_lds_view,
                                ckc_value_t* a_smem,
                                ckc_value_t* const a_mn_origin[2],
                                const ckc_moe_operand_t* operands,
                                int num_operands,
                                ckc_value_t* const b_mn_origin[2],
                                ckc_value_t* const* acc_inits_flat,
                                const char* const* acc_names_flat,
                                const int* group_sizes,
                                ckc_value_t* K,
                                ckc_value_t* warp_m_idx,
                                ckc_value_t* warp_n_idx,
                                ckc_value_t* lane,
                                int sched_groups,
                                ckc_value_t** out_groups_flat)
{
    ckc_ir_builder_t* b    = plan->b;
    ckc_value_t* c0        = ckc_b_const_i32(b, 0);
    ckc_value_t* c_block_k = ckc_b_const_i32(b, plan->block_k);

    int n_a       = plan->a_vecs_per_thread;
    int n_b_per   = plan->b_vecs_per_thread;
    int total_acc = 0;
    for(int g = 0; g < num_operands; ++g)
    {
        total_acc += group_sizes[g];
    }

    /* prefetch tile 0. */
    ckc_value_t* a_pre0[CKC_MOE_MAX_VECS];
    ckc_value_t* b_pre0[CKC_MOE_MAX_OPERANDS * CKC_MOE_MAX_VECS];
    ckc_moe_emit_global_load(
        plan, a_view, a_mn_origin, operands, num_operands, b_mn_origin, c0, a_pre0, b_pre0);

    /* Build the loop-carried iter-args: accumulators, then A prefetch, then B
     * prefetch (per operand). */
    ckc_iter_arg_t iter_args[CKC_MOE_MAX_ITER_ARGS];
    int n_ia = 0;
    char names[CKC_MOE_MAX_ITER_ARGS][32]; /* "b%d_pre%d" worst case = 28 bytes */
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

    ckc_for_t for_op = ckc_b_scf_for_iter(b, c0, K, c_block_k, iter_args, n_ia, "k0", false, true);
    ckc_b_region_enter(b, for_op.body);
    {
        ckc_value_t* k0  = for_op.iv;
        ckc_value_t** iv = for_op.iter_vars;
        int off          = 0;
        /* cur accumulator groups (pointers into iv). */
        ckc_value_t* cur_groups_storage[CKC_MOE_MAX_OPERANDS][CKC_MOE_MAX_MFMAS] = {{0}};
        ckc_value_t* const* cur_groups[CKC_MOE_MAX_OPERANDS]                     = {0};
        for(int g = 0; g < num_operands; ++g)
        {
            for(int j = 0; j < group_sizes[g]; ++j)
            {
                cur_groups_storage[g][j] = iv[off + j];
            }
            cur_groups[g] = cur_groups_storage[g];
            off += group_sizes[g];
        }
        ckc_value_t* a_regs[CKC_MOE_MAX_VECS] = {0};
        for(int i = 0; i < n_a; ++i)
        {
            a_regs[i] = iv[off + i];
        }
        off += n_a;
        ckc_value_t* b_reg_groups[CKC_MOE_MAX_OPERANDS * CKC_MOE_MAX_VECS] = {0};
        for(int gi = 0; gi < num_operands; ++gi)
        {
            for(int i = 0; i < n_b_per; ++i)
            {
                b_reg_groups[(size_t)gi * n_b_per + i] = iv[off + i];
            }
            off += n_b_per;
        }

        ckc_moe_emit_lds_store(plan, a_lds_view, a_regs, operands, num_operands, b_reg_groups);
        ckc_b_sync(b);
        ckc_value_t* k_next    = ckc_b_add(b, k0, c_block_k);
        ckc_value_t* k_clamped = ckc_b_select(b, ckc_b_cmp_lt(b, k_next, K), k_next, k0);
        ckc_value_t* a_next[CKC_MOE_MAX_VECS];
        ckc_value_t* b_next[CKC_MOE_MAX_OPERANDS * CKC_MOE_MAX_VECS];
        ckc_moe_emit_global_load(plan,
                                 a_view,
                                 a_mn_origin,
                                 operands,
                                 num_operands,
                                 b_mn_origin,
                                 k_clamped,
                                 a_next,
                                 b_next);

        ckc_value_t* new_groups_flat[2 * CKC_MOE_MAX_MFMAS];
        ckc_moe_emit_mfma_phase(plan,
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
        ckc_b_sync(b);

        ckc_value_t* yielded[CKC_MOE_MAX_ITER_ARGS];
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
        ckc_b_scf_yield(b, yielded, ny);
    }
    ckc_b_region_leave(b);

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
static int ckc_moe_y_x_stride(const ckc_tile_distribution_encoding_t* enc, int y_idx)
{
    int major = enc->Ys_major[y_idx];
    int minor = enc->Ys_minor[y_idx];
    if(major == 0)
    {
        return 1;
    }
    const ckc_h_row_t* h = &enc->Hs[major - 1];
    int stride           = 1;
    for(int level = minor + 1; level < h->count; ++level)
    {
        stride *= h->levels[level];
    }
    return stride;
}

/* Y_lengths[y_idx]: the bucket length the Y maps to (Hs[major-1][minor], or
 * Rs[minor] when major == 0). Mirrors TileDistributionEncoding.Y_lengths. */
static int ckc_moe_y_length(const ckc_tile_distribution_encoding_t* enc, int y_idx)
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
static void ckc_moe_load_store_traits(const ckc_tile_distribution_encoding_t* enc,
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
        if(ckc_moe_y_x_stride(enc, y) == 1)
        {
            int len = ckc_moe_y_length(enc, y);
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
        int spv      = (full_len < 8) ? full_len : 8;
        while(spv > 1 && (full_len % spv != 0 || (spv & (spv - 1)) != 0))
        {
            spv /= 2;
        }
        if(spv < 1)
        {
            spv = 1;
        }
        *out_vector_dim_y = best_idx;
        *out_spv          = spv;
    }
    else
    {
        *out_vector_dim_y = num_Y - 1;
        *out_spv          = 1;
    }
}

void ckc_moe_emit_cshuffle_stage(ckc_ir_builder_t* b,
                                 const ckc_gemm_universal_spec_t* spec,
                                 const ckc_moe_cwarp_decode_t* cdec,
                                 ckc_value_t* smem,
                                 const ckc_type_t* storage_dtype,
                                 int c_per_lane,
                                 ckc_moe_cell_value_fn cell_value,
                                 void* cell_user)
{
    const ckc_gemm_tile_spec_t* t = &spec->tile;
    int mfmas_m                   = ckc_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n                   = ckc_gemm_tile_mfmas_per_warp_n(t);

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
    const ckc_tile_distribution_t* dist         = cdec->dist;
    const ckc_tile_distribution_encoding_t* enc = dist->encoding;
    int m1                                      = cdec->m1;
    int num_Y                                   = enc->num_Y;

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
     * ckc_make_tile_window: it allocates two extra builder ids the Python free
     * function does not.) The store targets `smem` directly, which equals
     * lds_window.view.base. */
    ckc_value_t* lds_base = smem;
    (void)ckc_b_const_i32(b, 0);
    (void)ckc_b_const_i32(b, 0);

    int vector_dim_y = 0;
    int spv          = 1;
    ckc_moe_load_store_traits(enc, &vector_dim_y, &spv);

    /* Y lengths + the non-vector ("outer") axis lengths in Y-index order. */
    int y_len[8];
    for(int y = 0; y < num_Y; ++y)
    {
        y_len[y] = ckc_moe_y_length(enc, y);
    }
    int outer_axis[8];
    int outer_len[8];
    int num_outer = 0;
    for(int y = 0; y < num_Y; ++y)
    {
        if(y != vector_dim_y)
        {
            outer_axis[num_outer] = y;
            outer_len[num_outer]  = y_len[y];
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
            ckc_static_distributed_tensor_t* dt =
                ckc_make_static_distributed_tensor(b, dist, storage_dtype);
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
                        slot   = slot * y_len[y] + yi;
                    }
                    ckc_value_t* scalar = dt->storage[slot];
                    /* coord_fn(y_base, k): slot = y_base[0]*m1 + k. */
                    int coord_slot    = y_base[0] * m1 + k;
                    ckc_value_t* ld_m = NULL;
                    ckc_value_t* ld_n = NULL;
                    ckc_moe_cwarp_decode_coords(cdec, mi, ni, coord_slot, &ld_m, &ld_n);
                    ckc_value_t* coords[2] = {ld_m, ld_n};
                    ckc_b_smem_store_vN(b, lds_base, coords, 2, scalar, 1);
                }
            }
        }
    }
}

void ckc_moe_emit_down_reduce_epilogue_atomic(ckc_ir_builder_t* b,
                                              const ckc_gemm_universal_spec_t* spec,
                                              ckc_value_t* const* accs,
                                              ckc_value_t* warp_m_idx,
                                              ckc_value_t* warp_n_idx,
                                              ckc_value_t* lane,
                                              ckc_value_t* block_m_off,
                                              ckc_value_t* block_n_off,
                                              ckc_value_t* M,
                                              ckc_value_t* N,
                                              ckc_value_t* SortedTokenIds,
                                              ckc_value_t* SortedWeights,
                                              ckc_value_t* Y,
                                              int c_per_lane,
                                              ckc_value_t* batch_bucket_off,
                                              ckc_value_t* tokens)
{
    const ckc_gemm_tile_spec_t* t = &spec->tile;
    int mfmas_m                   = ckc_gemm_tile_mfmas_per_warp_m(t);
    int mfmas_n                   = ckc_gemm_tile_mfmas_per_warp_n(t);
    ckc_value_t* warp_m_off =
        ckc_b_mul(b, warp_m_idx, ckc_b_const_i32(b, mfmas_m * t->warp_tile_m));
    ckc_value_t* warp_n_off =
        ckc_b_mul(b, warp_n_idx, ckc_b_const_i32(b, mfmas_n * t->warp_tile_n));
    bool pad_m = spec->trait.pad_m;
    bool pad_n = spec->trait.pad_n;

    ckc_moe_cwarp_decode_t cdec;
    if(!ckc_moe_cwarp_decode_init(&cdec, b, spec, warp_m_off, warp_n_off, lane))
    {
        return;
    }
    for(int mi = 0; mi < mfmas_m; ++mi)
    {
        /* Per-mi c_n list (one per ni); i-independent (hoisted). */
        ckc_value_t* c_ns[CKC_MOE_MAX_MFMAS];
        for(int ni = 0; ni < mfmas_n; ++ni)
        {
            c_ns[ni] = ckc_b_add(b, block_n_off, ckc_moe_cwarp_decode_warp_col(&cdec, ni));
        }
        for(int i = 0; i < c_per_lane; ++i)
        {
            ckc_value_t* c_m =
                ckc_b_add(b, block_m_off, ckc_moe_cwarp_decode_warp_row(&cdec, mi, i));
            /* emit_one_row: hoist token+weight load out of the ni loop. */
            ckc_if_t guard_m;
            bool have_guard_m = pad_m;
            if(have_guard_m)
            {
                guard_m = ckc_b_scf_if(b, ckc_b_cmp_lt(b, c_m, M));
                ckc_b_region_enter(b, guard_m.then_region);
            }
            {
                ckc_value_t* bucket = ckc_b_add(b, batch_bucket_off, c_m);
                ckc_value_t* token  = ckc_b_global_load_i32(b, SortedTokenIds, bucket, 0);
                /* valid = b.land(b.cmp_ge(token, 0), b.cmp_lt(token, tokens))
                 * Python evaluates the land operands left-to-right, so cmp_ge is
                 * emitted BEFORE cmp_lt. C function-argument evaluation order is
                 * unspecified, so sequence the two compares into locals (ge
                 * first) to keep the emitted SSA order byte-identical. */
                ckc_value_t* tok_ge0 = ckc_b_cmp_ge(b, token, ckc_b_const_i32(b, 0));
                ckc_value_t* tok_lt  = ckc_b_cmp_lt(b, token, tokens);
                ckc_value_t* valid   = ckc_b_land(b, tok_ge0, tok_lt);
                ckc_if_t vguard      = ckc_b_scf_if(b, valid);
                ckc_b_region_enter(b, vguard.then_region);
                {
                    ckc_value_t* w = ckc_b_global_load_f32(b, SortedWeights, bucket, 0);
                    for(int ni = 0; ni < mfmas_n; ++ni)
                    {
                        ckc_value_t* acc     = accs[mi * mfmas_n + ni];
                        ckc_value_t* v       = ckc_b_vec_extract(b, acc, i);
                        ckc_value_t* contrib = ckc_b_fmul(b, w, v);
                        ckc_value_t* y_off   = ckc_b_add(b, ckc_b_mul(b, token, N), c_ns[ni]);
                        /* Python: b.global_atomic_add(Y, y_off, contrib) -- the
                         * generic memref.global_atomic_add op (plain "monotonic"
                         * f32 atomicrmw + the AMDGPU fp-atomic metadata, no
                         * syncscope("agent") / align 4). The _f32 builder emits a
                         * DIFFERENT form (syncscope agent, align 4); use the
                         * generic builder with ordering=NULL (=> monotonic). */
                        if(pad_n)
                        {
                            ckc_if_t ng = ckc_b_scf_if(b, ckc_b_cmp_lt(b, c_ns[ni], N));
                            ckc_b_region_enter(b, ng.then_region);
                            ckc_b_global_atomic_add(b, Y, y_off, contrib, NULL);
                            ckc_b_region_leave(b);
                        }
                        else
                        {
                            ckc_b_global_atomic_add(b, Y, y_off, contrib, NULL);
                        }
                    }
                }
                ckc_b_region_leave(b); /* vguard */
            }
            if(have_guard_m)
            {
                ckc_b_region_leave(b); /* guard_m */
            }
        }
    }
}

/* ====================================================================== *
 *  Spec types
 * ====================================================================== */

/* _data_spec: "f16"/"fp16" -> "fp16", else pass through. Builds a homogeneous
 * A/B/C data spec on top of the universal default (fp32 acc, RCR layout). */
static ckc_gemm_data_spec_t ckc_moe_data_spec(const char* dtype)
{
    ckc_gemm_universal_spec_t base = ckc_gemm_universal_spec_default();
    ckc_gemm_data_spec_t d         = base.data;
    const char* dt = (dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
                         ? "fp16"
                         : (dtype != NULL ? dtype : "fp16");
    d.dtype_a      = dt;
    d.dtype_b      = dt;
    d.dtype_c      = dt;
    return d;
}

static ckc_gemm_universal_spec_t ckc_moe_to_universal(const char* name,
                                                      const ckc_gemm_tile_spec_t* tile,
                                                      const ckc_gemm_trait_spec_t* trait,
                                                      int wave_size,
                                                      int block_size,
                                                      const char* dtype)
{
    ckc_gemm_universal_spec_t u = ckc_gemm_universal_spec_default();
    u.name                      = name;
    u.tile                      = *tile;
    u.trait                     = *trait;
    u.data                      = ckc_moe_data_spec(dtype);
    u.wave_size                 = wave_size;
    u.block_size                = block_size;
    u.batched                   = true;
    return u;
}

static void
ckc_moe_finalize_block_size(int* block_size, const ckc_gemm_tile_spec_t* t, int wave_size)
{
    if(*block_size == 0)
    {
        *block_size = t->warp_m * t->warp_n * t->warp_k * wave_size;
    }
}

static ckc_gemm_trait_spec_t ckc_moe_default_trait(void)
{
    /* TraitSpec(epilogue="default") (the spec field default_factory). */
    ckc_gemm_universal_spec_t base = ckc_gemm_universal_spec_default();
    ckc_gemm_trait_spec_t tr       = base.trait;
    tr.epilogue                    = "default";
    return tr;
}

/* ---- helper: append a static suffix, replacing '/' (matches kernel_name). */
static ckc_status_t ckc_moe_kernel_name_with_suffix(const ckc_gemm_universal_spec_t* u,
                                                    const char* suffix,
                                                    char* out,
                                                    size_t out_cap)
{
    char base[1024];
    ckc_status_t st = ckc_gemm_universal_kernel_name(u, base, sizeof(base));
    if(st != CKC_OK)
    {
        return st;
    }
    int n = snprintf(out, out_cap, "%s%s", base, suffix);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

/* ----------------------------- FusedGateUpSiluGemmSpec ---------------------- */

ckc_moe_gate_up_silu_gemm_spec_t ckc_moe_gate_up_silu_gemm_spec_default(void)
{
    ckc_moe_gate_up_silu_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name       = NULL;
    s.tile       = ckc_gemm_universal_spec_default().tile;
    s.trait      = ckc_moe_default_trait();
    s.wave_size  = 64;
    s.block_size = 0;
    s.dtype      = "fp16";
    s.grouped    = false;
    return s;
}

void ckc_moe_gate_up_silu_gemm_spec_finalize(ckc_moe_gate_up_silu_gemm_spec_t* spec)
{
    ckc_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

ckc_gemm_universal_spec_t
ckc_moe_gate_up_silu_gemm_spec_to_universal(const ckc_moe_gate_up_silu_gemm_spec_t* spec)
{
    return ckc_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

ckc_status_t ckc_moe_gate_up_silu_gemm_spec_kernel_name(
    const ckc_moe_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap)
{
    ckc_gemm_universal_spec_t u = ckc_moe_gate_up_silu_gemm_spec_to_universal(spec);
    const char* suffix          = spec->grouped ? "_gate_up_silu_grouped" : "_gate_up_silu";
    return ckc_moe_kernel_name_with_suffix(&u, suffix, out, out_cap);
}

/* -------------------- FusedInterleavedGateUpSiluGemmSpec -------------------- */

ckc_moe_interleaved_gate_up_silu_gemm_spec_t
ckc_moe_interleaved_gate_up_silu_gemm_spec_default(void)
{
    ckc_moe_interleaved_gate_up_silu_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name       = NULL;
    s.tile       = ckc_gemm_universal_spec_default().tile;
    s.trait      = ckc_moe_default_trait();
    s.wave_size  = 64;
    s.block_size = 0;
    s.dtype      = "fp16";
    s.grouped    = false;
    return s;
}

void ckc_moe_interleaved_gate_up_silu_gemm_spec_finalize(
    ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec)
{
    ckc_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

ckc_gemm_universal_spec_t ckc_moe_interleaved_gate_up_silu_gemm_spec_to_universal(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec)
{
    return ckc_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

ckc_status_t ckc_moe_interleaved_gate_up_silu_gemm_spec_kernel_name(
    const ckc_moe_interleaved_gate_up_silu_gemm_spec_t* spec, char* out, size_t out_cap)
{
    ckc_gemm_universal_spec_t u = ckc_moe_interleaved_gate_up_silu_gemm_spec_to_universal(spec);
    const char* suffix =
        spec->grouped ? "_interleaved_gate_up_silu_grouped" : "_interleaved_gate_up_silu";
    return ckc_moe_kernel_name_with_suffix(&u, suffix, out, out_cap);
}

/* --------------------------- FusedDownReduceGemmSpec ------------------------ */

ckc_moe_down_reduce_gemm_spec_t ckc_moe_down_reduce_gemm_spec_default(void)
{
    ckc_moe_down_reduce_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name       = NULL;
    s.tile       = ckc_gemm_universal_spec_default().tile;
    s.trait      = ckc_moe_default_trait();
    s.wave_size  = 64;
    s.block_size = 0;
    s.dtype      = "fp16";
    s.grouped    = false;
    return s;
}

void ckc_moe_down_reduce_gemm_spec_finalize(ckc_moe_down_reduce_gemm_spec_t* spec)
{
    ckc_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

ckc_gemm_universal_spec_t
ckc_moe_down_reduce_gemm_spec_to_universal(const ckc_moe_down_reduce_gemm_spec_t* spec)
{
    return ckc_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

ckc_status_t ckc_moe_down_reduce_gemm_spec_kernel_name(const ckc_moe_down_reduce_gemm_spec_t* spec,
                                                       char* out,
                                                       size_t out_cap)
{
    ckc_gemm_universal_spec_t u = ckc_moe_down_reduce_gemm_spec_to_universal(spec);
    const char* suffix          = spec->grouped ? "_down_reduce_grouped" : "_down_reduce";
    return ckc_moe_kernel_name_with_suffix(&u, suffix, out, out_cap);
}

/* ------------------------- FusedDownSiluReduceGemmSpec ---------------------- */

ckc_moe_down_silu_reduce_gemm_spec_t ckc_moe_down_silu_reduce_gemm_spec_default(void)
{
    ckc_moe_down_silu_reduce_gemm_spec_t s;
    memset(&s, 0, sizeof(s));
    s.name       = NULL;
    s.tile       = ckc_gemm_universal_spec_default().tile;
    s.trait      = ckc_moe_default_trait();
    s.wave_size  = 64;
    s.block_size = 0;
    s.dtype      = "fp16";
    return s;
}

void ckc_moe_down_silu_reduce_gemm_spec_finalize(ckc_moe_down_silu_reduce_gemm_spec_t* spec)
{
    ckc_moe_finalize_block_size(&spec->block_size, &spec->tile, spec->wave_size);
}

ckc_gemm_universal_spec_t
ckc_moe_down_silu_reduce_gemm_spec_to_universal(const ckc_moe_down_silu_reduce_gemm_spec_t* spec)
{
    return ckc_moe_to_universal(
        spec->name, &spec->tile, &spec->trait, spec->wave_size, spec->block_size, spec->dtype);
}

ckc_status_t ckc_moe_down_silu_reduce_gemm_spec_kernel_name(
    const ckc_moe_down_silu_reduce_gemm_spec_t* spec, char* out, size_t out_cap)
{
    ckc_gemm_universal_spec_t u = ckc_moe_down_silu_reduce_gemm_spec_to_universal(spec);
    return ckc_moe_kernel_name_with_suffix(&u, "_down_silu_reduce", out, out_cap);
}
