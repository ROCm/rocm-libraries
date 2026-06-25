// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_fused_mega_fp8_moe_fused_mega_fp8_glue.c -- PUBLIC entry + GLUE
 * for the C99 chunked port of build_moe_fused_mega_gemm_fp8
 * (ck_dsl/instances/common/moe_fused_mega_fp8.py, lines 203-2129).
 *
 * SCOPE (this translation unit):
 *   - ckc_fused_mega_fp8_levers_default                       (module env defaults)
 *   - ckc_fused_mega_kernel_spec_fp8_default / _post_init     (dataclass defaults)
 *   - ckc_fused_mega_fp8_spec_gate_up_atom / _down_atom       (gate_up/down_atom)
 *   - ckc_fused_mega_fp8_spec_mfmas_{m,n,m_down,n_down}       (@property ports)
 *   - ckc_fused_mega_fp8_spec_kernel_name                     (kernel_name())
 *   - ckc_moe_fused_mega_fp8_grid / _persistent_grid          (grid helpers)
 *   - ckc_moe_fused_mega_fp8_signature                        (SignatureBuilder)
 *   - ckc_build_moe_fused_mega_gemm_fp8 (+ _new + lower)      (the build driver)
 *   - ckc_moe_fp8_elem_bytes_b / _b_base / _scale_base        (rebasing closures)
 *   - ckc_moe_fp8_select_item                                 (_select_item)
 *   - ckc_moe_fp8_emit_body                                   (_emit_body)
 *
 * This is the "bucket that calls phases": it reproduces the Python
 * build_moe_fused_mega_gemm_fp8 control flow byte-for-byte. The per-phase
 * emitters (gate/up fused K-loop, store-hidden Pass A, down group GEMM, down
 * atomic reduce, ...) are owned by peer TUs declared in the internal header and
 * are invoked here in Python execution order.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckc/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "ckc/helper_ck_dsl.helpers.atoms.h"
#include "ckc/helper_ck_dsl.helpers.mfma_gemm_inner.h"
#include "ckc/helper_ck_dsl.helpers.spec.h"
#include "ckc/helper_ck_dsl.helpers.tensor_view.h"
#include "ckc/instance_moe_fused_mega_fp8.h"
#include "ckc/instance_moe_fused_mega_fp8_internal.h"
#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/lower_llvm.h"

/* ===================================================================== *
 *  Optimization-lever defaults (Python module-level _USE_* / env flags)
 *
 *  Python import-time defaults (all _USE_* False, hazard_nop 8, sched "iglp1").
 * ===================================================================== */
ckc_fused_mega_fp8_levers_t ckc_fused_mega_fp8_levers_default(void)
{
    ckc_fused_mega_fp8_levers_t lv;
    lv.use_asm_agpr_mfma = false;
    lv.use_asm_agpr_mfma_down = false;
    lv.use_x_dtla = false;
    lv.use_mfma_cluster = false;
    lv.asm_mfma_hazard_nop = 8;
    lv.sched_cadence = "iglp1";
    return lv;
}

/* ===================================================================== *
 *  FusedMegaKernelSpecFp8 defaults + __post_init__  (Python 333-359)
 * ===================================================================== */
ckc_fused_mega_kernel_spec_fp8_t ckc_fused_mega_kernel_spec_fp8_default(void)
{
    ckc_fused_mega_kernel_spec_fp8_t s;

    memset(&s, 0, sizeof(s));

    s.name = NULL; /* required (Python field has no default) */

    s.tile_m = 16;
    s.tile_n_inter = 256;
    s.tile_k_gu = 32;
    s.warp_m = 1;
    s.warp_n = 4;
    s.warp_tile_m = 16;
    s.warp_tile_n = 16;
    s.warp_tile_k = 32;
    s.tile_n_down = 256;
    s.tile_k_down = 64;
    s.wave_size = 64;
    s.block_size = 0; /* resolved by __post_init__ below */
    s.dtype = "fp8e4m3";

    /* optimization-lever flags (defaults = final best) */
    s.gate_up_k = 128;
    s.down_k = 128;
    s.use_dtla = true;

    s.has_sched_cadence = false; /* Python None (defer to env) */
    s.sched_cadence = NULL;

    ckc_fused_mega_kernel_spec_fp8_post_init(&s);
    return s;
}

void ckc_fused_mega_kernel_spec_fp8_post_init(ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    if(spec == NULL)
    {
        return;
    }
    /* if self.block_size == 0: block_size = warp_m * warp_n * wave_size */
    if(spec->block_size == 0)
    {
        spec->block_size = spec->warp_m * spec->warp_n * spec->wave_size;
    }
}

/* ===================================================================== *
 *  gate_up_atom() / down_atom()  (Python 363-377)
 *
 *  gate_up_k==32 -> fp8 16x16x32 catalog atom, else the fp8 16x16x128 hero atom.
 *  MfmaAtom.fp8_16x16x{32,128}() == ckc_mfma_atom("fp8e4m3", 16, 16, {32,128}).
 * ===================================================================== */
const ckc_mfma_atom_t*
    ckc_fused_mega_fp8_spec_gate_up_atom(const ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    if(spec->gate_up_k == 32)
    {
        return ckc_mfma_atom("fp8e4m3", 16, 16, 32);
    }
    return ckc_mfma_atom("fp8e4m3", 16, 16, 128);
}

const ckc_mfma_atom_t*
    ckc_fused_mega_fp8_spec_down_atom(const ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    if(spec == NULL)
    {
        return NULL;
    }
    if(spec->down_k == 32)
    {
        return ckc_mfma_atom("fp8e4m3", 16, 16, 32);
    }
    return ckc_mfma_atom("fp8e4m3", 16, 16, 128);
}

/* ===================================================================== *
 *  mfmas_m / mfmas_n / mfmas_m_down / mfmas_n_down  (Python 379-397)
 * ===================================================================== */
int ckc_fused_mega_fp8_spec_mfmas_m(const ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    /* (tile_m // warp_m) // warp_tile_m */
    return (spec->tile_m / spec->warp_m) / spec->warp_tile_m;
}

int ckc_fused_mega_fp8_spec_mfmas_n(const ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    /* (tile_n_inter // warp_n) // warp_tile_n */
    return (spec->tile_n_inter / spec->warp_n) / spec->warp_tile_n;
}

int ckc_fused_mega_fp8_spec_mfmas_m_down(const ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    /* (tile_m // warp_m) // warp_tile_m */
    return (spec->tile_m / spec->warp_m) / spec->warp_tile_m;
}

int ckc_fused_mega_fp8_spec_mfmas_n_down(const ckc_fused_mega_kernel_spec_fp8_t* spec)
{
    /* (tile_n_down // warp_n) // warp_tile_n */
    return (spec->tile_n_down / spec->warp_n) / spec->warp_tile_n;
}

/* ===================================================================== *
 *  kernel_name()  (Python 399-403)
 *
 *  "{name}_moe_fused_mega_fp8_m{tile_m}n{tile_n_inter}k{tile_k_gu}"
 * ===================================================================== */
ckc_status_t ckc_fused_mega_fp8_spec_kernel_name(const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    int n;

    if(spec == NULL || out == NULL || spec->name == NULL)
    {
        return CKC_ERR_VALUE;
    }
    n = snprintf(out,
                 out_cap,
                 "%s_moe_fused_mega_fp8_m%dn%dk%d",
                 spec->name,
                 spec->tile_m,
                 spec->tile_n_inter,
                 spec->tile_k_gu);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

/* ===================================================================== *
 *  Grid helpers  (Python 411-452)
 * ===================================================================== */
ckc_status_t ckc_moe_fused_mega_fp8_grid(int num_m_blocks,
                                         int inter,
                                         const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                         int out_grid[3])
{
    int sub_gu;
    int gx;

    if(spec == NULL || out_grid == NULL)
    {
        return CKC_ERR_VALUE;
    }
    sub_gu = spec->tile_n_inter;
    gx = (inter + sub_gu - 1) / sub_gu;
    out_grid[0] = gx;
    out_grid[1] = num_m_blocks;
    out_grid[2] = 1;
    return CKC_OK;
}

ckc_status_t ckc_moe_fused_mega_fp8_persistent_grid(int num_m_blocks,
                                                    int inter,
                                                    const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                                    int p_cap,
                                                    int out_grid[3],
                                                    int* out_grid_x,
                                                    int* out_total_work,
                                                    int* out_P)
{
    int sub_gu;
    int grid_x;
    int total_work;
    int P;

    if(spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    if(p_cap <= 0)
    {
        p_cap = CKC_MOE_FP8_PERSISTENT_P_CAP;
    }
    sub_gu = spec->tile_n_inter;
    grid_x = (inter + sub_gu - 1) / sub_gu;
    total_work = grid_x * num_m_blocks;
    /* P = min(total_work, p_cap) if total_work > 0 else 1 */
    if(total_work > 0)
    {
        P = (total_work < p_cap) ? total_work : p_cap;
    }
    else
    {
        P = 1;
    }
    if(out_grid != NULL)
    {
        out_grid[0] = P;
        out_grid[1] = 1;
        out_grid[2] = 1;
    }
    if(out_grid_x != NULL)
    {
        *out_grid_x = grid_x;
    }
    if(out_total_work != NULL)
    {
        *out_total_work = total_work;
    }
    if(out_P != NULL)
    {
        *out_P = P;
    }
    return CKC_OK;
}

/* ===================================================================== *
 *  moe_fused_mega_fp8_signature()  (Python 455-496)
 * ===================================================================== */
ckc_status_t ckc_moe_fused_mega_fp8_signature(const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                              bool persistent,
                                              ckc_arena_t* arena,
                                              const ckc_sig_entry_t** out_items,
                                              size_t* out_count)
{
    ckc_signature_builder_t sb;

    if(spec == NULL || arena == NULL || out_items == NULL)
    {
        return CKC_ERR_VALUE;
    }
    if(ckc_signature_builder_init(&sb, arena) != CKC_OK)
    {
        return CKC_ERR_VALUE;
    }

    ckc_signature_builder_ptr(&sb, "A", "fp8e4m3", NULL);
    ckc_signature_builder_ptr(&sb, "WGate", "fp8e4m3", NULL);
    ckc_signature_builder_ptr(&sb, "WUp", "fp8e4m3", NULL);
    ckc_signature_builder_ptr(&sb, "WDown", "fp8e4m3", NULL);
    ckc_signature_builder_ptr(&sb, "AScale", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "WGateScale", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "WUpScale", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "WDownScale", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "SortedTokenIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "SortedWeights", "f32", NULL);
    ckc_signature_builder_ptr(&sb, "BlockExpertIds", "i32", NULL);
    ckc_signature_builder_ptr(&sb, "Y", "f32", NULL);
    ckc_signature_builder_scalar(&sb, "M", "i32");
    ckc_signature_builder_scalar(&sb, "N", "i32");
    ckc_signature_builder_scalar(&sb, "K", "i32");
    ckc_signature_builder_scalar(&sb, "H_out", "i32");
    ckc_signature_builder_scalar(&sb, "stride_a", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b_gate", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b_up", "i32");
    ckc_signature_builder_scalar(&sb, "stride_b_down", "i32");
    ckc_signature_builder_scalar(&sb, "stride_a_scale", "i32");
    ckc_signature_builder_scalar(&sb, "stride_gate_scale", "i32");
    ckc_signature_builder_scalar(&sb, "stride_up_scale", "i32");
    ckc_signature_builder_scalar(&sb, "stride_down_scale", "i32");
    ckc_signature_builder_scalar(&sb, "stride_gate_scale_e", "i32");
    ckc_signature_builder_scalar(&sb, "stride_up_scale_e", "i32");
    ckc_signature_builder_scalar(&sb, "stride_down_scale_e", "i32");
    ckc_signature_builder_scalar(&sb, "slot_size", "i32");
    ckc_signature_builder_scalar(&sb, "tokens", "i32");
    if(persistent)
    {
        ckc_signature_builder_scalar(&sb, "grid_x", "i32");
        ckc_signature_builder_scalar(&sb, "total_work", "i32");
        ckc_signature_builder_scalar(&sb, "P", "i32");
    }

    return ckc_signature_builder_build(&sb, out_items, out_count);
}

/* ===================================================================== *
 *  PER-EXPERT POINTER REBASING CLOSURES  (Python 1710-1772)
 *
 *  These four closures (ckc_moe_fp8_elem_bytes_b / _b_base / _scale_base /
 *  _select_item) are owned canonically by this glue part-file and declared in
 *  the shared instance_moe_fused_mega_fp8_internal.h. The select_dispatch peer
 *  TU defers to these definitions to avoid a duplicate-symbol link clash.
 * ===================================================================== */

/* _elem_bytes_b: lazy const_i64(1). */
ckc_value_t* ckc_moe_fp8_elem_bytes_b(ckc_moe_fp8_build_ctx_t* ctx)
{
    if(ctx->elem_bytes_b == NULL)
    {
        ctx->elem_bytes_b = ckc_b_const_i64(ctx->b, 1);
    }
    return ctx->elem_bytes_b;
}

/* _b_base(ptr, stride_b, expert_idx): global_ptr_add(ptr,
 * sext(expert)*sext(stride)*elem_bytes_b). */
ckc_value_t* ckc_moe_fp8_b_base(ckc_moe_fp8_build_ctx_t* ctx,
                                ckc_value_t* ptr,
                                ckc_value_t* stride_b,
                                ckc_value_t* expert_idx)
{
    /* Python: b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)) -- the
     * expert sext is emitted FIRST (left-to-right arg evaluation). C function
     * argument evaluation order is unspecified, so force it with ordered
     * temporaries to keep the SSA value numbering byte-identical. */
    ckc_value_t* expert_i64 = ckc_b_sext(ctx->b, expert_idx, ckc_i64());
    ckc_value_t* stride_i64 = ckc_b_sext(ctx->b, stride_b, ckc_i64());
    ckc_value_t* inner = ckc_b_mul(ctx->b, expert_i64, stride_i64);
    ckc_value_t* bytes_off = ckc_b_mul(ctx->b, inner, ckc_moe_fp8_elem_bytes_b(ctx));
    return ckc_b_global_ptr_add(ctx->b, ptr, bytes_off);
}

/* _scale_base(ptr, stride_e, expert_idx): global_ptr_add(ptr,
 * sext(expert)*sext(stride_e)*4). */
ckc_value_t* ckc_moe_fp8_scale_base(ckc_moe_fp8_build_ctx_t* ctx,
                                    ckc_value_t* ptr,
                                    ckc_value_t* stride_e,
                                    ckc_value_t* expert_idx)
{
    /* Python: b.mul(b.mul(b.sext(expert_idx, I64), b.sext(stride_e, I64)),
     * b.const_i64(4)) -- expert sext emitted FIRST. Force C arg-eval order with
     * ordered temporaries (see ckc_moe_fp8_b_base). */
    ckc_value_t* expert_i64 = ckc_b_sext(ctx->b, expert_idx, ckc_i64());
    ckc_value_t* stride_i64 = ckc_b_sext(ctx->b, stride_e, ckc_i64());
    ckc_value_t* inner = ckc_b_mul(ctx->b, expert_i64, stride_i64);
    ckc_value_t* bytes_off = ckc_b_mul(ctx->b, inner, ckc_b_const_i64(ctx->b, 4));
    return ckc_b_global_ptr_add(ctx->b, ptr, bytes_off);
}

/* _select_item(m_block_idx, bx_block): derive the per-work-item state into ctx.
 * bx_block NULL => emit block_id_x() HERE (default-path op-order). */
void ckc_moe_fp8_select_item(ckc_moe_fp8_build_ctx_t* ctx,
                             ckc_value_t* m_block_idx,
                             ckc_value_t* bx_block)
{
    ckc_value_t* bx;

    ctx->expert_idx = ckc_b_global_load_i32(ctx->b, ctx->BlockExpertIds, m_block_idx, 0);
    /* Create the byte-multiplier const HERE (right after expert_idx) so the
     * default-path op-counter order matches the pre-persistent baseline. */
    ckc_moe_fp8_elem_bytes_b(ctx);
    ctx->WGate = ckc_moe_fp8_b_base(ctx, ctx->WGate0, ctx->stride_b_gate, ctx->expert_idx);
    ctx->WUp = ckc_moe_fp8_b_base(ctx, ctx->WUp0, ctx->stride_b_up, ctx->expert_idx);
    ctx->WDown = ckc_moe_fp8_b_base(ctx, ctx->WDown0, ctx->stride_b_down, ctx->expert_idx);
    ctx->WGateScale
        = ckc_moe_fp8_scale_base(ctx, ctx->WGateScale0, ctx->stride_gate_scale_e, ctx->expert_idx);
    ctx->WUpScale
        = ckc_moe_fp8_scale_base(ctx, ctx->WUpScale0, ctx->stride_up_scale_e, ctx->expert_idx);
    ctx->WDownScale
        = ckc_moe_fp8_scale_base(ctx, ctx->WDownScale0, ctx->stride_down_scale_e, ctx->expert_idx);
    ctx->block_m_off = ckc_b_mul(ctx->b, m_block_idx, ctx->c_block_m);
    bx = bx_block;
    if(bx == NULL)
    {
        bx = ckc_b_block_id_x(ctx->b);
    }
    ctx->gu_n_off = ckc_b_mul(ctx->b, bx, ctx->c_block_n);
}

/* ===================================================================== *
 *  _emit_body  (Python 1850-2093)
 *
 *  Sequences STAGE 1a per-mi fused gate/up K-loop -> Pass A + amax butterfly ->
 *  per-block scale broadcast -> Pass C packed quantize -> STAGE 2 down loop,
 *  calling the phase functions from the peer buckets. Reads the per-work-item
 *  selected state already in ctx.
 * ===================================================================== */
void ckc_moe_fp8_emit_body(ckc_moe_fp8_build_ctx_t* ctx)
{
    ckc_ir_builder_t* b = ctx->b;
    const ckc_mfma_atom_t* atom = ctx->atom;
    int mfmas_m = ctx->mfmas_m;
    int mfmas_n = ctx->mfmas_n;
    int mfmas_m_down = ctx->mfmas_m_down;
    int mfmas_n_down = ctx->mfmas_n_down;
    int n_blocks = ctx->n_blocks;
    int tile_m = ctx->tile_m;
    int tile_n = ctx->tile_n;
    int warps_per_block = ctx->warps_per_block;

    /* gate_list / up_list: row-major (mi, ni), length mfmas_m*mfmas_n. */
    ckc_value_t* gate_list[CKC_MOE_FP8_MAX_ACCS];
    ckc_value_t* up_list[CKC_MOE_FP8_MAX_ACCS];
    int gate_count = 0;
    int up_count = 0;

    int mi;
    int ni;
    int wo;
    int xm;
    static const int kButterfly[6] = {1, 2, 4, 8, 16, 32};

    /* ---- DTLA bundle (L8 use_dtla): per-wave LDS base + read-row base. ---- */
    if(ctx->spec->use_dtla)
    {
        ckc_value_t* bstage_base_i64 = ckc_b_smem_addr_of(b, ctx->BStage_smem);
        int warp_rows = ctx->dtla_slots * ctx->dtla_chunks * ctx->spec->wave_size;
        int warp_wave_bytes = warp_rows * CKC_MOE_FP8_DTLA_CHUNK;
        ckc_value_t* wave_lds_base = ckc_b_smem_ptr_add(
            b,
            bstage_base_i64,
            ckc_b_sext(
                b, ckc_b_mul(b, ctx->warp_id, ckc_b_const_i32(b, warp_wave_bytes)), ckc_i64()));
        ckc_value_t* warp_row_base = ckc_b_mul(b, ctx->warp_id, ckc_b_const_i32(b, warp_rows));

        ctx->dtla.present = true;
        ctx->dtla.view = &ctx->bstage_view;
        ctx->dtla.base = wave_lds_base;
        ctx->dtla.warp_row_base = warp_row_base;
        ctx->dtla.lane = ctx->lane;
        ctx->dtla.wave_size = ctx->spec->wave_size;
        ctx->dtla.has_x_slot = ctx->has_x_slot;
        ctx->dtla.x_slot = ctx->x_slot;
    }
    else
    {
        ctx->dtla.present = false;
    }

    /* ---- STAGE 1a: gate + up fp8 GEMM -> f32 (per-128-block dequant) ---- */
    for(mi = 0; mi < mfmas_m; ++mi)
    {
        ckc_value_t* n_tile_bases[CKC_MOE_FP8_MAX_NNI];
        ckc_value_t* g_dqs[CKC_MOE_FP8_MAX_NNI];
        ckc_value_t* u_dqs[CKC_MOE_FP8_MAX_NNI];
        char tag[32];
        ckc_value_t* m_tile_base = ckc_b_add(
            b, ctx->block_m_off, ckc_b_add(b, ctx->warp_m_off, ckc_b_const_i32(b, mi * atom->m)));

        for(ni = 0; ni < mfmas_n; ++ni)
        {
            n_tile_bases[ni] = ckc_b_add(
                b, ctx->gu_n_off, ckc_b_add(b, ctx->warp_n_off, ckc_b_const_i32(b, ni * atom->n)));
        }

        snprintf(tag, sizeof(tag), "%d", mi);
        ckc_moe_fp8_emit_fp8_gateup_fused_kloop(ctx,
                                                ctx->A,
                                                ctx->WGate,
                                                ctx->WUp,
                                                ctx->AScale,
                                                ctx->WGateScale,
                                                ctx->WUpScale,
                                                m_tile_base,
                                                n_tile_bases,
                                                mfmas_n,
                                                ctx->K,
                                                ctx->stride_a_scale,
                                                ctx->stride_gate_scale,
                                                ctx->stride_up_scale,
                                                tag,
                                                ctx->dtla.present ? &ctx->dtla : NULL,
                                                ctx->cadence,
                                                g_dqs,
                                                u_dqs);

        for(ni = 0; ni < mfmas_n; ++ni)
        {
            gate_list[gate_count++] = g_dqs[ni];
            up_list[up_count++] = u_dqs[ni];
        }
    }

    /* ---- STAGE 1b Pass A (FUSED): SiLU(gate)*up -> f32 LDS + amax ---- */
    {
        ckc_value_t* amax_lane = ckc_moe_fp8_store_hidden_f32_pass(ctx,
                                                                   gate_list,
                                                                   up_list,
                                                                   &ctx->f32_view,
                                                                   ctx->warp_m_off,
                                                                   ctx->warp_n_off,
                                                                   ctx->lane,
                                                                   mfmas_m,
                                                                   mfmas_n,
                                                                   ctx->one_f32,
                                                                   ctx->c_neg_log2e,
                                                                   ctx->c_floor);

        /* 64-lane butterfly max over the warp (xor 1,2,4,8,16,32). */
        ckc_value_t* amax_warp = amax_lane;
        for(xm = 0; xm < 6; ++xm)
        {
            amax_warp
                = ckc_b_fmax(b, amax_warp, ckc_b_warp_shuffle_xor(b, amax_warp, kButterfly[xm]));
        }

        /* Lane 0 of each warp publishes its partial. */
        {
            ckc_if_t guard = ckc_b_scf_if(b, ckc_b_cmp_eq(b, ctx->lane, ctx->c0));
            ckc_value_t* idx0[1];
            idx0[0] = ctx->warp_id;
            ckc_b_region_enter(b, guard.then_region);
            ckc_b_smem_store_vN(b, ctx->WarpAmax_smem, idx0, 1, amax_warp, 1);
            ckc_b_region_leave(b);
        }
        ckc_b_sync(b);
    }

    /* ---- STAGE 1b combine: per-block amax from the 2 warps' partials ---- */
    {
        ckc_for_t sweep = ckc_b_scf_for_iter(b,
                                             ctx->tid,
                                             ckc_b_const_i32(b, n_blocks),
                                             ctx->c_threads,
                                             NULL,
                                             0,
                                             "cell",
                                             false,
                                             true);
        ckc_b_region_enter(b, sweep.body);
        {
            ckc_value_t* blk = sweep.iv;
            ckc_value_t* w0 = ckc_b_mul(b, blk, ckc_b_const_i32(b, warps_per_block));
            ckc_value_t* w0idx[1];
            ckc_value_t* amax;
            ckc_value_t* scale;
            ckc_for_t row_bc;

            w0idx[0] = w0;
            amax = ckc_b_vec_extract(
                b, ckc_b_smem_load_vN(b, ctx->WarpAmax_smem, w0idx, 1, ckc_f32(), 1), 0);
            for(wo = 1; wo < warps_per_block; ++wo)
            {
                ckc_value_t* pwidx[1];
                ckc_value_t* pw;
                pwidx[0] = ckc_b_add(b, w0, ckc_b_const_i32(b, wo));
                pw = ckc_b_vec_extract(
                    b, ckc_b_smem_load_vN(b, ctx->WarpAmax_smem, pwidx, 1, ckc_f32(), 1), 0);
                amax = ckc_b_fmax(b, amax, pw);
            }
            scale = ckc_b_fmul(b, amax, ckc_b_rcp(b, ctx->c_fp8_max)); /* amax / 448 */

            row_bc = ckc_b_scf_for_iter(
                b, ctx->c0, ctx->c_block_m, ckc_b_const_i32(b, 1), NULL, 0, "rb", false, true);
            ckc_b_region_enter(b, row_bc.body);
            {
                ckc_value_t* rridx[2];
                rridx[0] = row_bc.iv;
                rridx[1] = blk;
                ckc_b_smem_store_vN(b, ctx->scale_view.base, rridx, 2, scale, 1);
                ckc_b_scf_yield(b, NULL, 0);
            }
            ckc_b_region_leave(b);
            ckc_b_scf_yield(b, NULL, 0);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
    }

    /* ---- STAGE 1b Pass C: quantize f32 Hidden -> fp8 LDS (PACKED) ---- */
    {
        int total_q4 = (tile_m * tile_n) / 4;
        ckc_value_t* c_tile_n4 = ckc_b_const_i32(b, tile_n / 4);
        ckc_for_t qsweep = ckc_b_scf_for_iter(b,
                                              ctx->tid,
                                              ckc_b_const_i32(b, total_q4),
                                              ctx->c_threads,
                                              NULL,
                                              0,
                                              "qcell",
                                              false,
                                              true);
        ckc_b_region_enter(b, qsweep.body);
        {
            ckc_value_t* qcell = qsweep.iv;
            ckc_value_t* row = ckc_b_div(b, qcell, c_tile_n4);
            /* Python: b.mul(b.mod(qcell, c_tile_n4), b.const_i32(4)) -- the mod is
             * emitted FIRST, then the const_i32(4). Force C arg-eval order. */
            ckc_value_t* qmod = ckc_b_mod(b, qcell, c_tile_n4);
            ckc_value_t* col4 = ckc_b_mul(b, qmod, ckc_b_const_i32(b, 4));
            ckc_value_t* blk = ckc_b_div(b, col4, ctx->c_group_k);
            ckc_value_t* hvidx[2];
            ckc_value_t* scidx[2];
            ckc_value_t* hv4;
            ckc_value_t* sc;
            ckc_value_t* inv;
            ckc_value_t* comps[4];
            ckc_value_t* scaled;
            ckc_value_t* q4;
            ckc_value_t* qidx[2];
            int j;

            hvidx[0] = row;
            hvidx[1] = col4;
            scidx[0] = row;
            scidx[1] = blk;
            hv4 = ckc_b_smem_load_vN(b, ctx->f32_view.base, hvidx, 2, ckc_f32(), 4);
            sc = ckc_b_smem_load_vN(b, ctx->scale_view.base, scidx, 2, ckc_f32(), 1);
            inv = ckc_b_rcp_fast(b, ckc_b_vec_extract(b, sc, 0));
            for(j = 0; j < 4; ++j)
            {
                comps[j] = ckc_b_fmul(b, ckc_b_vec_extract(b, hv4, j), inv);
            }
            scaled = ckc_b_vec_pack(b, comps, 4, ckc_f32());
            q4 = ckc_b_cvt_pk_fp8_f32x4(b, scaled);
            qidx[0] = row;
            qidx[1] = col4;
            ckc_b_smem_store_vN(b, ctx->fp8_view.base, qidx, 2, q4, 4);
            ckc_b_scf_yield(b, NULL, 0);
        }
        ckc_b_region_leave(b);
        ckc_b_sync(b);
    }

    /* ---- STAGE 2: down fp8 GEMM (LDS-A) -> dequant -> weighted atomic Y ---- */
    {
        ckc_value_t* inter_blk_base = ckc_b_div(b, ctx->gu_n_off, ctx->c_group_k);
        ckc_for_t down_for = ckc_b_scf_for_iter(b,
                                                ctx->c0,
                                                ctx->H_out,
                                                ckc_b_const_i32(b, ctx->spec->tile_n_down),
                                                NULL,
                                                0,
                                                "ho",
                                                false,
                                                true);
        ckc_b_region_enter(b, down_for.body);
        {
            ckc_value_t* ho = down_for.iv;
            ckc_value_t* down_warp_m_off
                = ckc_b_mul(b, ctx->warp_m_idx, ckc_b_const_i32(b, mfmas_m_down * atom->m));
            ckc_value_t* down_warp_n_off
                = ckc_b_mul(b, ctx->warp_n_idx, ckc_b_const_i32(b, mfmas_n_down * atom->n));
            ckc_value_t* down_list[CKC_MOE_FP8_MAX_ACCS];
            int down_count = 0;

            for(mi = 0; mi < mfmas_m_down; ++mi)
            {
                for(ni = 0; ni < mfmas_n_down; ++ni)
                {
                    char dtag[32];
                    ckc_value_t* n_tile_base = ckc_b_add(
                        b, ho, ckc_b_add(b, down_warp_n_off, ckc_b_const_i32(b, ni * atom->n)));
                    ckc_value_t* m_row_base
                        = ckc_b_add(b, down_warp_m_off, ckc_b_const_i32(b, mi * atom->m));
                    ckc_value_t* d_dq;

                    snprintf(dtag, sizeof(dtag), "d%d_%d", mi, ni);
                    d_dq = ckc_moe_fp8_emit_fp8_down_group_gemm(ctx,
                                                                &ctx->fp8_view,
                                                                ctx->WDown,
                                                                ctx->WDownScale,
                                                                n_tile_base,
                                                                &ctx->scale_view,
                                                                tile_n,
                                                                ctx->N,
                                                                inter_blk_base,
                                                                ctx->stride_down_scale,
                                                                m_row_base,
                                                                dtag,
                                                                ctx->cadence);
                    down_list[down_count++] = d_dq;
                }
            }

            ckc_moe_fp8_emit_down_atomic_reduce(ctx,
                                                down_list,
                                                down_warp_m_off,
                                                down_warp_n_off,
                                                ctx->lane,
                                                mfmas_m_down,
                                                mfmas_n_down,
                                                ctx->block_m_off,
                                                ho,
                                                ctx->H_out,
                                                ctx->SortedTokenIds,
                                                ctx->SortedWeights,
                                                ctx->Y,
                                                ctx->tokens);
            ckc_b_scf_yield(b, NULL, 0);
        }
        ckc_b_region_leave(b);
    }
}

/* ===================================================================== *
 *  build_moe_fused_mega_gemm_fp8  (Python 1554-2129)
 *
 *  validate arch/block_size + catalog (skip k==128 hero) -> resolve atom +
 *  cadence -> emit ALL b.param() in ABI order -> derived geometry + fuse-quant
 *  invariant -> prelude -> lazy _elem_bytes_b holder -> LDS allocs + 4
 *  TensorViews + bstage_view -> decode_mfma_lanes -> populate ctx -> dispatch
 *  default vs persistent path.
 * ===================================================================== */
static ckc_value_t* moe_fp8_param(ckc_ir_builder_t* b,
                                  const char* name,
                                  const ckc_type_t* t,
                                  bool noalias,
                                  bool readonly,
                                  int align)
{
    ckc_param_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    if(noalias)
    {
        opts.noalias = true;
        opts.noalias_set = true;
    }
    if(readonly)
    {
        opts.readonly = true;
        opts.readonly_set = true;
    }
    if(align > 0)
    {
        opts.align = align;
        opts.align_set = true;
    }
    return ckc_b_param(b, name, t, &opts);
}

ckc_kernel_def_t* ckc_build_moe_fused_mega_gemm_fp8(ckc_ir_builder_t* b,
                                                    const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                                    const char* arch,
                                                    bool persistent,
                                                    const ckc_fused_mega_fp8_levers_t* levers)
{
    ckc_moe_fp8_build_ctx_t ctx;
    const char* reason = NULL;
    const ckc_type_t* fp8_global;
    const ckc_type_t* f32_global;
    const ckc_type_t* i32_global;

    int tile_m;
    int tile_n;
    int n_blocks;
    int warp_n_cols;
    int warps_per_block;
    int n_warps;
    int DTLA_SLOTS;
    int DTLA_CHUNKS;
    int bstage_rows;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* Python default */
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;
    ctx.persistent = persistent;
    ctx.levers = (levers != NULL) ? *levers : ckc_fused_mega_fp8_levers_default();

    /* ---- validate arch + block_size ---- (Python 1588-1590) */
    if(!ckc_validate_arch_and_block_size(b, arch, spec->block_size, &reason, NULL))
    {
        ckc_i_set_err(b,
                      CKC_ERR_VALUE,
                      "invalid fp8 fused-mega spec for %s: %s",
                      arch,
                      (reason != NULL) ? reason : "");
        return NULL;
    }

    /* ---- resolve atom; catalog guard only for non-hero (k != 128) ---- *
     * (Python 1591-1601) */
    ctx.atom = ckc_fused_mega_fp8_spec_gate_up_atom(spec);
    if(ctx.atom == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "fp8 fused-mega: gate_up_atom unavailable");
        return NULL;
    }
    if(ctx.atom->k != 128)
    {
        if(ckc_validate_mfma_atom_in_catalog(b, ctx.atom, arch, "moe_fused_mega_fp8") != CKC_OK)
        {
            return NULL;
        }
    }

    /* ---- L9 cadence: spec override (None => defer to env) ---- (Python 1606) */
    ctx.cadence = spec->has_sched_cadence ? spec->sched_cadence : NULL;

    /* ---- builder attrs ---- (Python 1609) */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    /* ---- params (BUILD_SPEC_FP8 Section 2.7) ---- (Python 1612-1654) */
    fp8_global = ckc_ptr_type(b, ckc_fp8e4m3(), "global");
    f32_global = ckc_ptr_type(b, ckc_f32(), "global");
    i32_global = ckc_ptr_type(b, ckc_i32(), "global");

    ctx.A = moe_fp8_param(b, "A", fp8_global, true, true, 16);
    ctx.WGate0 = moe_fp8_param(b, "WGate", fp8_global, true, true, 16);
    ctx.WUp0 = moe_fp8_param(b, "WUp", fp8_global, true, true, 16);
    ctx.WDown0 = moe_fp8_param(b, "WDown", fp8_global, true, true, 16);
    ctx.AScale = moe_fp8_param(b, "AScale", f32_global, false, true, 4);
    ctx.WGateScale0 = moe_fp8_param(b, "WGateScale", f32_global, false, true, 4);
    ctx.WUpScale0 = moe_fp8_param(b, "WUpScale", f32_global, false, true, 4);
    ctx.WDownScale0 = moe_fp8_param(b, "WDownScale", f32_global, false, true, 4);
    ctx.SortedTokenIds = moe_fp8_param(b, "SortedTokenIds", i32_global, true, true, 4);
    ctx.SortedWeights = moe_fp8_param(b, "SortedWeights", f32_global, true, true, 4);
    ctx.BlockExpertIds = moe_fp8_param(b, "BlockExpertIds", i32_global, true, true, 4);
    ctx.Y = moe_fp8_param(b, "Y", f32_global, false, false, 16);
    ctx.M = ckc_b_param(b, "M", ckc_i32(), NULL);
    ctx.N = ckc_b_param(b, "N", ckc_i32(), NULL);
    ctx.K = ckc_b_param(b, "K", ckc_i32(), NULL);
    ctx.H_out = ckc_b_param(b, "H_out", ckc_i32(), NULL);
    ctx.stride_a = ckc_b_param(b, "stride_a", ckc_i32(), NULL);
    ctx.stride_b_gate = ckc_b_param(b, "stride_b_gate", ckc_i32(), NULL);
    ctx.stride_b_up = ckc_b_param(b, "stride_b_up", ckc_i32(), NULL);
    ctx.stride_b_down = ckc_b_param(b, "stride_b_down", ckc_i32(), NULL);
    ctx.stride_a_scale = ckc_b_param(b, "stride_a_scale", ckc_i32(), NULL);
    ctx.stride_gate_scale = ckc_b_param(b, "stride_gate_scale", ckc_i32(), NULL);
    ctx.stride_up_scale = ckc_b_param(b, "stride_up_scale", ckc_i32(), NULL);
    ctx.stride_down_scale = ckc_b_param(b, "stride_down_scale", ckc_i32(), NULL);
    ctx.stride_gate_scale_e = ckc_b_param(b, "stride_gate_scale_e", ckc_i32(), NULL);
    ctx.stride_up_scale_e = ckc_b_param(b, "stride_up_scale_e", ckc_i32(), NULL);
    ctx.stride_down_scale_e = ckc_b_param(b, "stride_down_scale_e", ckc_i32(), NULL);
    ctx.slot_size = ckc_b_param(b, "slot_size", ckc_i32(), NULL);
    ctx.tokens = ckc_b_param(b, "tokens", ckc_i32(), NULL);

    /* Persistent-only params (Python 1661-1664). */
    if(persistent)
    {
        ctx.p_grid_x = ckc_b_param(b, "grid_x", ckc_i32(), NULL);
        ctx.p_total_work = ckc_b_param(b, "total_work", ckc_i32(), NULL);
        ctx.p_P = ckc_b_param(b, "P", ckc_i32(), NULL);
    }

    /* ---- derived geometry ---- (Python 1666-1687) */
    tile_m = spec->tile_m;
    tile_n = spec->tile_n_inter;
    n_blocks = tile_n / CKC_MOE_FP8_GROUP_K;
    warp_n_cols = tile_n / spec->warp_n; /* mfmas_n * atom.n */
    warps_per_block = CKC_MOE_FP8_GROUP_K / warp_n_cols;

    /* fuse-quant invariant. */
    if(spec->warp_m != 1 || warp_n_cols * spec->warp_n != tile_n
       || CKC_MOE_FP8_GROUP_K % warp_n_cols != 0 || warps_per_block * n_blocks != spec->warp_n)
    {
        ckc_i_set_err(b,
                      CKC_ERR_VALUE,
                      "fuse-quant lever requires warp_m==1 and warps to tile the "
                      "128-inter blocks evenly (got tile_n=%d, warp_n=%d, warp_n_cols=%d)",
                      tile_n,
                      spec->warp_n,
                      warp_n_cols);
        return NULL;
    }

    ctx.tile_m = tile_m;
    ctx.tile_n = tile_n;
    ctx.n_blocks = n_blocks;
    ctx.warp_n_cols = warp_n_cols;
    ctx.warps_per_block = warps_per_block;
    ctx.mfmas_m = ckc_fused_mega_fp8_spec_mfmas_m(spec);
    ctx.mfmas_n = ckc_fused_mega_fp8_spec_mfmas_n(spec);
    ctx.mfmas_m_down = ckc_fused_mega_fp8_spec_mfmas_m_down(spec);
    ctx.mfmas_n_down = ckc_fused_mega_fp8_spec_mfmas_n_down(spec);
    ctx.n_warps = spec->warp_m * spec->warp_n;

    /* ---- SSA constants (op-order) ---- (Python 1689-1693) */
    ctx.c_wave = ckc_b_const_i32(b, spec->wave_size);
    ctx.c_warps_n = ckc_b_const_i32(b, spec->warp_n);
    ctx.c_block_m = ckc_b_const_i32(b, tile_m);
    ctx.c_block_n = ckc_b_const_i32(b, tile_n);
    ctx.c0 = ckc_b_const_i32(b, 0);

    /* ---- block/thread prelude ---- (Python 1696-1700) */
    ctx.tid = ckc_b_thread_id_x(b);
    ctx.warp_id = ckc_b_div(b, ctx.tid, ctx.c_wave);
    ctx.warp_m_idx = ckc_b_div(b, ctx.warp_id, ctx.c_warps_n);
    ctx.warp_n_idx = ckc_b_mod(b, ctx.warp_id, ctx.c_warps_n);
    ctx.lane = ckc_b_mod(b, ctx.tid, ctx.c_wave);

    /* ---- lazy _elem_bytes_b holder ---- (Python 1708) */
    ctx.elem_bytes_b = NULL;

    /* ---- DEFAULT path: select the single work-item HERE ---- (Python 1778-1779) */
    if(!persistent)
    {
        ckc_moe_fp8_select_item(&ctx, ckc_b_block_id_y(b), NULL);
    }

    /* ---- LDS allocations ---- (Python 1784-1809) */
    {
        int hidden_shape[2];
        int hscale_shape[2];
        int hf32_shape[2];
        int wamax_shape[1];
        int bstage_shape[2];

        hidden_shape[0] = tile_m;
        hidden_shape[1] = tile_n;
        ctx.Hidden_smem = ckc_b_smem_alloc(b, ckc_fp8e4m3(), hidden_shape, 2, "Hidden_smem");

        hscale_shape[0] = tile_m;
        hscale_shape[1] = n_blocks;
        ctx.HiddenScale_smem = ckc_b_smem_alloc(b, ckc_f32(), hscale_shape, 2, "HiddenScale_smem");

        hf32_shape[0] = tile_m;
        hf32_shape[1] = tile_n;
        ctx.HiddenF32_smem = ckc_b_smem_alloc(b, ckc_f32(), hf32_shape, 2, "HiddenF32_smem");

        n_warps = ctx.n_warps;
        wamax_shape[0] = n_warps;
        ctx.WarpAmax_smem = ckc_b_smem_alloc(b, ckc_f32(), wamax_shape, 1, "WarpAmax_smem");

        /* DTLA landing-zone geometry. */
        DTLA_SLOTS = ctx.levers.use_x_dtla ? 5 : 4;
        ctx.dtla_slots = DTLA_SLOTS;
        ctx.has_x_slot = ctx.levers.use_x_dtla;
        ctx.x_slot = ctx.levers.use_x_dtla ? 4 : 0;
        DTLA_CHUNKS = (ctx.atom->b_per_lane + CKC_MOE_FP8_DTLA_CHUNK - 1) / CKC_MOE_FP8_DTLA_CHUNK;
        ctx.dtla_chunks = DTLA_CHUNKS;
        bstage_rows = n_warps * DTLA_SLOTS * DTLA_CHUNKS * spec->wave_size;
        ctx.bstage_rows = bstage_rows;

        bstage_shape[0] = bstage_rows;
        bstage_shape[1] = CKC_MOE_FP8_DTLA_CHUNK;
        ctx.BStage_smem = ckc_b_smem_alloc(b, ckc_fp8e4m3(), bstage_shape, 2, "BStage_smem");

        /* bstage_view (lds). */
        ctx.bstage_view.base = ctx.BStage_smem;
        ctx.bstage_view.addr_space = CKC_ADDR_LDS;
        ckc_tensor_descriptor_packed(&ctx.bstage_view.desc, bstage_shape, 2, ckc_fp8e4m3());

        /* f32_view (lds) over HiddenF32_smem. */
        ctx.f32_view.base = ctx.HiddenF32_smem;
        ctx.f32_view.addr_space = CKC_ADDR_LDS;
        ckc_tensor_descriptor_packed(&ctx.f32_view.desc, hf32_shape, 2, ckc_f32());

        /* fp8_view (lds) over Hidden_smem. */
        ctx.fp8_view.base = ctx.Hidden_smem;
        ctx.fp8_view.addr_space = CKC_ADDR_LDS;
        ckc_tensor_descriptor_packed(&ctx.fp8_view.desc, hidden_shape, 2, ckc_fp8e4m3());

        /* scale_view (lds) over HiddenScale_smem. */
        ctx.scale_view.base = ctx.HiddenScale_smem;
        ctx.scale_view.addr_space = CKC_ADDR_LDS;
        ckc_tensor_descriptor_packed(&ctx.scale_view.desc, hscale_shape, 2, ckc_f32());
    }

    /* ---- lane decode + warp offsets ---- (Python 1832-1838) */
    ctx.lane_decode = ckc_decode_mfma_lanes(b, ctx.atom, ctx.lane);
    ctx.warp_m_off = ckc_b_mul(b, ctx.warp_m_idx, ckc_b_const_i32(b, ctx.mfmas_m * ctx.atom->m));
    ctx.warp_n_off = ckc_b_mul(b, ctx.warp_n_idx, ckc_b_const_i32(b, ctx.mfmas_n * ctx.atom->n));

    /* ---- f32/const SSA constants ---- (Python 1840-1848) */
    ctx.c_neg_log2e = ckc_b_const_f32(b, -1.4426950408889634);
    ctx.one_f32 = ckc_b_const_f32(b, 1.0);
    ctx.c_fp8_max = ckc_b_const_f32(b, CKC_MOE_FP8_FP8_MAX);
    ctx.c_floor = ckc_b_const_f32(b, CKC_MOE_FP8_AMAX_FLOOR);

    ctx.c_group_k = ckc_b_const_i32(b, CKC_MOE_FP8_GROUP_K);
    ctx.c_threads = ckc_b_const_i32(b, spec->block_size);
    ctx.c_n_blocks = ckc_b_const_i32(b, n_blocks);
    ctx.c_tile_n = ckc_b_const_i32(b, tile_n);

    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }

    /* ---- dispatch: default vs persistent ---- (Python 2095-2126) */
    if(!persistent)
    {
        /* with b.scf_if(b.cmp_ge(expert_idx, c0)): _emit_body() */
        ckc_if_t guard = ckc_b_scf_if(b, ckc_b_cmp_ge(b, ctx.expert_idx, ctx.c0));
        ckc_b_region_enter(b, guard.then_region);
        ckc_moe_fp8_emit_body(&ctx);
        ckc_b_region_leave(b);
    }
    else
    {
        /* PERSISTENT path: grid-stride over linear work-ids. */
        ckc_value_t* p = ckc_b_block_id_x(b);
        ckc_for_t wloop
            = ckc_b_scf_for_iter(b, p, ctx.p_total_work, ctx.p_P, NULL, 0, "witem", false, true);
        ckc_b_region_enter(b, wloop.body);
        {
            ckc_value_t* w = wloop.iv;
            ckc_value_t* bx = ckc_b_mod(b, w, ctx.p_grid_x);
            ckc_value_t* by = ckc_b_div(b, w, ctx.p_grid_x);
            ckc_if_t guard;

            ckc_moe_fp8_select_item(&ctx, by, bx);
            /* Inter-item LDS hand-off barrier (the ONE new barrier). */
            ckc_b_sync(b);
            guard = ckc_b_scf_if(b, ckc_b_cmp_ge(b, ctx.expert_idx, ctx.c0));
            ckc_b_region_enter(b, guard.then_region);
            ckc_moe_fp8_emit_body(&ctx);
            ckc_b_region_leave(b);
            ckc_b_scf_yield(b, NULL, 0);
        }
        ckc_b_region_leave(b);
    }

    ckc_b_ret(b);

    if(!ckc_ir_builder_ok(b))
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  Convenience: init `b` with spec.kernel_name(), then build.
 * ===================================================================== */
ckc_kernel_def_t*
    ckc_build_moe_fused_mega_gemm_fp8_new(ckc_ir_builder_t* b,
                                          const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                          const char* arch,
                                          bool persistent,
                                          const ckc_fused_mega_fp8_levers_t* levers)
{
    return ckc::guard_builder(b, [&]() -> ckc_kernel_def_t* {
        char name[1024];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(ckc_fused_mega_fp8_spec_kernel_name(spec, name, sizeof(name)) != CKC_OK)
        {
            return NULL;
        }
        if(ckc_ir_builder_init(b, name) != CKC_OK)
        {
            return NULL;
        }
        return ckc_build_moe_fused_mega_gemm_fp8(b, spec, arch, persistent, levers);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM GLUE
 * ===================================================================== */
static void ckc_moe_fp8_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, msg, n);
    err[n] = '\0';
}

ckc_status_t ckc_moe_fused_mega_fp8_lower_to_llvm(const ckc_fused_mega_kernel_spec_fp8_t* spec,
                                                  const char* arch,
                                                  bool persistent,
                                                  const ckc_fused_mega_fp8_levers_t* levers,
                                                  ckc_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap)
{
    ckc_ir_builder_t b;
    ckc_kernel_def_t* kernel;
    ckc_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        ckc_moe_fp8_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return CKC_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = ckc_build_moe_fused_mega_gemm_fp8_new(&b, spec, arch, persistent, levers);
    if(kernel == NULL)
    {
        const char* m = ckc_ir_builder_error(&b);
        st = ckc_ir_builder_status(&b);
        ckc_moe_fp8_set_err(
            err, err_cap, (m != NULL && m[0] != '\0') ? m : "build_moe_fused_mega_gemm_fp8 failed");
        ckc_ir_builder_free(&b);
        return (st == CKC_OK) ? CKC_ERR_VALUE : st;
    }

    st = ckc_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    ckc_ir_builder_free(&b);
    return st;
}
