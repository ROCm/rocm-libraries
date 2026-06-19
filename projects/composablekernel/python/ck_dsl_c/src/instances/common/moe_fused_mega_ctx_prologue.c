/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * instance_moe_fused_mega_ctx_prologue.c -- C99 port of the build prologue of
 * build_moe_fused_mega_gemm (ck_dsl/instances/common/moe_fused_mega.py, lines
 * 448-634). Implements ckc_moe_mega_build_ctx_init: validity-gate the two
 * universal specs, set builder attrs, bind all 19 params, compute the storage
 * dtype / tile geometry / scalar constants, the block/thread decode, the
 * per-expert B byte rebinding, the LDS allocations, the gate/up + down tensor
 * views, the two _MoeKloopPlan inits and operand structs, the SiLU constants,
 * and the down-GEMM setup -- populating every field of ckc_moe_mega_build_ctx_t
 * in byte-identical builder-call order.
 *
 * Only this scope is implemented here; the staged body phases / down-kloop
 * helpers are peers declared in the internal header.
 */
#include "ckc/instance_moe_fused_mega_internal.h"

#include <string.h>

#include "ckc/ir_internal.h"            /* ckc_i_set_err                       */
#include "ckc/instance_gemm_internal.h" /* ckc_gemm_emit_zero_acc              */

/* ===================================================================== *
 *  file-local helpers (mirror the gemm_universal module statics the Python
 *  imports: _storage_dtype, _mfma_atom_widths). Kept identical to the
 *  moe_gemm_fused.c port so the emitted geometry matches.
 * ===================================================================== */

/* _storage_dtype(u_gu): homogeneous A/B/C dtype -> ckc_type_t. */
static const ckc_type_t* ckc_mega_storage_dtype(const ckc_gemm_universal_spec_t* u)
{
    const char* d = u->data.dtype_a;
    if (d == NULL)
    {
        return ckc_f16();
    }
    if (strcmp(d, "f16") == 0 || strcmp(d, "fp16") == 0)
    {
        return ckc_f16();
    }
    if (strcmp(d, "bf16") == 0)
    {
        return ckc_bf16();
    }
    return ckc_scalar_by_name(d);
}

/* _mfma_atom_widths(u_gu) -> (a_per_lane, b_per_lane, c_per_lane). MFMA-only:
 * the warp-tile atom's per-lane fragment widths. */
static void ckc_mega_mfma_atom_widths(const ckc_gemm_universal_spec_t* u,
                                      int* a_per,
                                      int* b_per,
                                      int* c_per)
{
    const ckc_gemm_tile_spec_t* t = &u->tile;
    int wm   = t->warp_tile_m;
    int wn   = t->warp_tile_n;
    int wk   = t->warp_tile_k;
    int wave = u->wave_size;
    *a_per = (wm * wk) / wave;
    *b_per = (wn * wk) / wave;
    *c_per = (wm * wn) / wave;
}

/* Arena-allocate one ckc_tensor_view_t the ctx can hold by const pointer. The
 * builder arena outlives the build, matching the Python "owned by the build"
 * lifetime. Returns NULL on OOM (caller sets the sticky error). */
static ckc_tensor_view_t* ckc_mega_arena_view(ckc_ir_builder_t* b)
{
    return (ckc_tensor_view_t*)ckc_arena_calloc(&b->arena, sizeof(ckc_tensor_view_t));
}

/* ===================================================================== *
 *  ckc_moe_mega_build_ctx_init -- the whole prologue (Python lines 448-634).
 * ===================================================================== */
ckc_status_t ckc_moe_mega_build_ctx_init(ckc_moe_mega_build_ctx_t* ctx,
                                         ckc_ir_builder_t* b,
                                         const ckc_moe_fused_mega_kernel_spec_t* spec,
                                         const char* arch,
                                         const ckc_gemm_universal_spec_t* u_gu,
                                         const ckc_gemm_universal_spec_t* u_down)
{
    memset(ctx, 0, sizeof(*ctx));

    ctx->b    = b;
    ctx->spec = spec;
    ctx->arch = (arch != NULL) ? arch : "gfx950";
    ctx->u_gu   = u_gu;
    ctx->u_down = u_down;

    /* ---- validity gates (lines 448-455) -------------------------------- *
     * u_gu = spec.gate_up_universal_spec(); ok, why = is_valid_gemm_spec(...)
     *   -> ValueError "invalid fused-mega gate+up GEMM spec: {why}".
     * u_down = spec.down_universal_spec(); ok, why = is_valid_gemm_spec(...)
     *   -> ValueError "invalid fused-mega down GEMM spec: {why}".
     * The two universal specs are computed by the public driver scratch (the
     * Python spec.gate_up_universal_spec()/down_universal_spec()); we validate
     * the supplied pointers. */
    {
        char why[CKC_ERR_MSG_CAP];
        why[0] = '\0';
        if (!ckc_gemm_universal_is_valid_spec(u_gu, ctx->arch, why, sizeof why))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fused-mega gate+up GEMM spec: %s", why);
            return CKC_ERR_VALUE;
        }
        why[0] = '\0';
        if (!ckc_gemm_universal_is_valid_spec(u_down, ctx->arch, why, sizeof why))
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "invalid fused-mega down GEMM spec: %s", why);
            return CKC_ERR_VALUE;
        }
    }

    /* ---- builder attrs (lines 458-460) --------------------------------- *
     * b.kernel.attrs["max_workgroup_size"] = spec.block_size
     * if spec.trait.waves_per_eu is not None: attrs["waves_per_eu"] = ... */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    if (spec->trait.waves_per_eu_set)
    {
        ckc_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->trait.waves_per_eu);
    }

    /* storage_dtype = _storage_dtype(u_gu) (line 462). */
    ctx->storage_dtype = ckc_mega_storage_dtype(u_gu);
    const ckc_type_t* storage_dtype = ctx->storage_dtype;

    /* ---- params (BUILD_SPEC Section 3.1, lines 464-496) ----------------- */
    const ckc_type_t* ptr_storage = ckc_ptr_type(b, storage_dtype, "global");
    {
        ckc_param_opts_t ro16;
        memset(&ro16, 0, sizeof ro16);
        ro16.noalias = true;
        ro16.noalias_set = true;
        ro16.readonly = true;
        ro16.readonly_set = true;
        ro16.align = 16;
        ro16.align_set = true;

        ctx->A     = ckc_b_param(b, "A", ptr_storage, &ro16);
        ctx->WGate = ckc_b_param(b, "WGate", ptr_storage, &ro16);
        ctx->WUp   = ckc_b_param(b, "WUp", ptr_storage, &ro16);
        ctx->WDown = ckc_b_param(b, "WDown", ptr_storage, &ro16);

        ckc_param_opts_t ro4;
        memset(&ro4, 0, sizeof ro4);
        ro4.noalias = true;
        ro4.noalias_set = true;
        ro4.readonly = true;
        ro4.readonly_set = true;
        ro4.align = 4;
        ro4.align_set = true;

        ctx->SortedTokenIds =
            ckc_b_param(b, "SortedTokenIds", ckc_ptr_type(b, ckc_i32(), "global"), &ro4);
        ctx->SortedWeights =
            ckc_b_param(b, "SortedWeights", ckc_ptr_type(b, ckc_f32(), "global"), &ro4);
        ctx->BlockExpertIds =
            ckc_b_param(b, "BlockExpertIds", ckc_ptr_type(b, ckc_i32(), "global"), &ro4);

        ckc_param_opts_t a16;
        memset(&a16, 0, sizeof a16);
        a16.align = 16;
        a16.align_set = true;
        ctx->Y = ckc_b_param(b, "Y", ckc_ptr_type(b, ckc_f32(), "global"), &a16);
    }

    ctx->M        = ckc_b_param(b, "M", ckc_i32(), NULL);
    ctx->N        = ckc_b_param(b, "N", ckc_i32(), NULL);
    ctx->K        = ckc_b_param(b, "K", ckc_i32(), NULL);
    ctx->H_out    = ckc_b_param(b, "H_out", ckc_i32(), NULL);
    ctx->stride_a = ckc_b_param(b, "stride_a", ckc_i32(), NULL);
    ctx->stride_b_gate = ckc_b_param(b, "stride_b_gate", ckc_i32(), NULL);
    ctx->stride_b_up   = ckc_b_param(b, "stride_b_up", ckc_i32(), NULL);
    ctx->stride_b_down = ckc_b_param(b, "stride_b_down", ckc_i32(), NULL);
    ctx->slot_size = ckc_b_param(b, "slot_size", ckc_i32(), NULL);
    ctx->tokens    = ckc_b_param(b, "tokens", ckc_i32(), NULL);

    /* ---- gate/up tile geometry + scalar consts (lines 498-509) --------- *
     * t = spec.gate_up_tile() = u_gu->tile (UniversalGemmSpec.tile mirrors
     * gate_up_tile()). _, _, c_per_lane = _mfma_atom_widths(u_gu). */
    ctx->t = u_gu->tile;
    {
        int a_per = 0, b_per = 0, c_per = 0;
        ckc_mega_mfma_atom_widths(u_gu, &a_per, &b_per, &c_per);
        ctx->c_per_lane = c_per;
    }

    ctx->block_m = ctx->t.tile_m;
    ctx->block_n = ctx->t.tile_n;
    ctx->block_k = ctx->t.tile_k;

    ctx->mfmas_m = ckc_gemm_tile_mfmas_per_warp_m(&ctx->t);
    ctx->mfmas_n = ckc_gemm_tile_mfmas_per_warp_n(&ctx->t);

    ctx->c_wave    = ckc_b_const_i32(b, spec->wave_size);
    ctx->c_warps_n = ckc_b_const_i32(b, ctx->t.warp_n);
    ctx->c_block_m = ckc_b_const_i32(b, ctx->block_m);
    ctx->c_block_n = ckc_b_const_i32(b, ctx->block_n);
    ctx->c0        = ckc_b_const_i32(b, 0);

    /* ---- block/thread prelude (lines 511-519) -------------------------- */
    ctx->tid        = ckc_b_thread_id_x(b);
    ctx->warp_id    = ckc_b_div(b, ctx->tid, ctx->c_wave);
    ctx->warp_m_idx = ckc_b_div(b, ctx->warp_id, ctx->c_warps_n);
    ctx->warp_n_idx = ckc_b_mod(b, ctx->warp_id, ctx->c_warps_n);
    ctx->lane       = ckc_b_mod(b, ctx->tid, ctx->c_wave);

    ctx->m_block_idx = ckc_b_block_id_y(b);
    ctx->expert_idx  = ckc_b_global_load_i32(b, ctx->BlockExpertIds, ctx->m_block_idx, 0);

    /* ---- per-expert B byte base (lines 521-533) ------------------------ *
     * elem_bytes_b = const_i64(2); _b_base(ptr, stride_b):
     *   bytes_off = (sext(expert_idx,I64) * sext(stride_b,I64)) * elem_bytes_b
     *   return global_ptr_add(ptr, bytes_off)
     * WGate/WUp/WDown shadowed by their per-expert byte base. */
    ctx->elem_bytes_b = ckc_b_const_i64(b, 2);
    {
        /* The Python _b_base closure re-emits sext(expert_idx,I64) on EACH of
         * the three calls (the builder does no CSE), so the .ll contains three
         * distinct sext(expert_idx) values; and Python evaluates the mul args
         * left-to-right => sext(expert_idx) THEN sext(stride_b). C does NOT
         * sequence function-call arguments, so the two sexts must be emitted as
         * ordered locals to stay byte-identical (hoisting OR relying on arg
         * eval order both diverge the SSA names). */
        ckc_value_t* eg = ckc_b_sext(b, ctx->expert_idx, ckc_i64());
        ckc_value_t* sg = ckc_b_sext(b, ctx->stride_b_gate, ckc_i64());
        ckc_value_t* off_g =
            ckc_b_mul(b, ckc_b_mul(b, eg, sg), ctx->elem_bytes_b);
        ctx->WGate = ckc_b_global_ptr_add(b, ctx->WGate, off_g);

        ckc_value_t* eu = ckc_b_sext(b, ctx->expert_idx, ckc_i64());
        ckc_value_t* su = ckc_b_sext(b, ctx->stride_b_up, ckc_i64());
        ckc_value_t* off_u =
            ckc_b_mul(b, ckc_b_mul(b, eu, su), ctx->elem_bytes_b);
        ctx->WUp = ckc_b_global_ptr_add(b, ctx->WUp, off_u);

        ckc_value_t* ed = ckc_b_sext(b, ctx->expert_idx, ckc_i64());
        ckc_value_t* sd = ckc_b_sext(b, ctx->stride_b_down, ckc_i64());
        ckc_value_t* off_d =
            ckc_b_mul(b, ckc_b_mul(b, ed, sd), ctx->elem_bytes_b);
        ctx->WDown = ckc_b_global_ptr_add(b, ctx->WDown, off_d);
    }

    /* ---- tile origins (lines 535-539) ---------------------------------- */
    ctx->batch_off_a = ctx->c0;
    ctx->batch_off_b = ctx->c0;
    ctx->block_m_off = ckc_b_mul(b, ctx->m_block_idx, ctx->c_block_m);
    ctx->gu_n_off    = ckc_b_mul(b, ckc_b_block_id_x(b), ctx->c_block_n);

    /* ---- LDS allocations (lines 541-551) ------------------------------- */
    {
        int au[2] = { ctx->block_m, ctx->block_k };
        ctx->A_smem = ckc_b_smem_alloc(b, storage_dtype, au, 2, "A_smem");
        int bn[2] = { ctx->block_n, ctx->block_k };
        ctx->Bg_smem = ckc_b_smem_alloc(b, storage_dtype, bn, 2, "Bg_smem");
        ctx->Bu_smem = ckc_b_smem_alloc(b, storage_dtype, bn, 2, "Bu_smem");
        int hm[2] = { ctx->block_m, ctx->block_n };
        ctx->Hidden_smem = ckc_b_smem_alloc(b, storage_dtype, hm, 2, "Hidden_smem");
    }

    /* ---- accumulator inits (lines 553-566) ----------------------------- *
     * mfmas_m/mfmas_n already computed. acc_init = _emit_zero_acc(b, u_gu).
     * gate_accs / up_accs are (name, acc_init) groups of length mfmas_m*mfmas_n;
     * the C port stores the shared init Value + count (num_gate_up_accs), the
     * body rebuilds the named groups (STAGE 1). */
    ctx->acc_init         = ckc_gemm_emit_zero_acc(b, u_gu);
    ctx->num_gate_up_accs = ctx->mfmas_m * ctx->mfmas_n;

    /* ---- global + LDS views (lines 568-592) ---------------------------- *
     * a_view  = make_global_view(A,    (1,1,1), storage_dtype, strides=(1,K,1))
     * wg_view = make_global_view(WGate, ...)
     * wu_view = make_global_view(WUp,   ...)
     * a/bg/bu lds views = TensorView(base=*_smem,
     *     desc=TensorDescriptor.packed((blk_m|blk_n, block_k), storage_dtype),
     *     addr_space="lds"). */
    {
        int ones3[3] = { 1, 1, 1 };
        ckc_stride_t s_kview[3];
        s_kview[0] = ckc_stride_imm(1);
        s_kview[1] = ckc_stride_value(ctx->K);
        s_kview[2] = ckc_stride_imm(1);

        ckc_tensor_view_t* a_view  = ckc_mega_arena_view(b);
        ckc_tensor_view_t* wg_view = ckc_mega_arena_view(b);
        ckc_tensor_view_t* wu_view = ckc_mega_arena_view(b);
        if (a_view == NULL || wg_view == NULL || wu_view == NULL)
        {
            ckc_i_set_err(b, CKC_ERR_OOM, "moe-mega prologue: view alloc failed");
            return CKC_ERR_OOM;
        }
        if (ckc_make_global_view(a_view, ctx->A, ones3, 3, storage_dtype, s_kview) != CKC_OK ||
            ckc_make_global_view(wg_view, ctx->WGate, ones3, 3, storage_dtype, s_kview) != CKC_OK ||
            ckc_make_global_view(wu_view, ctx->WUp, ones3, 3, storage_dtype, s_kview) != CKC_OK)
        {
            return ckc_ir_builder_status(b);
        }
        ctx->a_view  = a_view;
        ctx->wg_view = wg_view;
        ctx->wu_view = wu_view;

        ckc_tensor_view_t* a_lds  = ckc_mega_arena_view(b);
        ckc_tensor_view_t* bg_lds = ckc_mega_arena_view(b);
        ckc_tensor_view_t* bu_lds = ckc_mega_arena_view(b);
        if (a_lds == NULL || bg_lds == NULL || bu_lds == NULL)
        {
            ckc_i_set_err(b, CKC_ERR_OOM, "moe-mega prologue: lds view alloc failed");
            return CKC_ERR_OOM;
        }
        int a_shape[2] = { ctx->block_m, ctx->block_k };
        int b_shape[2] = { ctx->block_n, ctx->block_k };
        a_lds->base = ctx->A_smem;
        a_lds->addr_space = CKC_ADDR_LDS;
        bg_lds->base = ctx->Bg_smem;
        bg_lds->addr_space = CKC_ADDR_LDS;
        bu_lds->base = ctx->Bu_smem;
        bu_lds->addr_space = CKC_ADDR_LDS;
        if (ckc_tensor_descriptor_packed(&a_lds->desc, a_shape, 2, storage_dtype) != CKC_OK ||
            ckc_tensor_descriptor_packed(&bg_lds->desc, b_shape, 2, storage_dtype) != CKC_OK ||
            ckc_tensor_descriptor_packed(&bu_lds->desc, b_shape, 2, storage_dtype) != CKC_OK)
        {
            return ckc_ir_builder_status(b);
        }
        ctx->a_lds_view  = a_lds;
        ctx->bg_lds_view = bg_lds;
        ctx->bu_lds_view = bu_lds;
    }

    /* ---- gate/up k-loop plan + operands + origins (lines 594-600) ------ */
    if (!ckc_moe_kloop_plan_init(&ctx->plan, b, u_gu, ctx->tid))
    {
        return ckc_ir_builder_status(b);
    }
    /* operands = [ _MoeOperand(wg_view, bg_lds_view, Bg_smem),
     *              _MoeOperand(wu_view, bu_lds_view, Bu_smem) ] */
    memset(&ctx->operands[0], 0, sizeof ctx->operands[0]);
    ctx->operands[0].global_view = ctx->wg_view;
    ctx->operands[0].lds_view    = ctx->bg_lds_view;
    ctx->operands[0].smem        = ctx->Bg_smem;
    memset(&ctx->operands[1], 0, sizeof ctx->operands[1]);
    ctx->operands[1].global_view = ctx->wu_view;
    ctx->operands[1].lds_view    = ctx->bu_lds_view;
    ctx->operands[1].smem        = ctx->Bu_smem;

    ctx->a_mn_origin[0] = ctx->batch_off_a;
    ctx->a_mn_origin[1] = ctx->block_m_off;
    ctx->b_mn_origin[0] = ctx->batch_off_b;
    ctx->b_mn_origin[1] = ctx->gu_n_off;

    /* ---- SiLU constants (lines 602-603) -------------------------------- */
    ctx->c_neg_log2e = ckc_b_const_f32(b, -1.4426950408889634);
    ctx->one_f32     = ckc_b_const_f32(b, 1.0);

    /* ---- DOWN-GEMM setup (lines 605-634) ------------------------------- *
     * td = spec.down_tile() = u_down->tile. */
    ctx->td = u_down->tile;
    ctx->block_n_down = ctx->td.tile_n;
    ctx->block_k_down = ctx->td.tile_k;
    ctx->down_mfmas_m = ckc_gemm_tile_mfmas_per_warp_m(&ctx->td);
    ctx->down_mfmas_n = ckc_gemm_tile_mfmas_per_warp_n(&ctx->td);
    ctx->c_block_n_down = ckc_b_const_i32(b, ctx->block_n_down);

    {
        int bd_shape[2] = { ctx->block_n_down, ctx->block_k_down };
        ctx->Bd_smem = ckc_b_smem_alloc(b, storage_dtype, bd_shape, 2, "Bd_smem");

        ckc_tensor_view_t* bd_lds = ckc_mega_arena_view(b);
        if (bd_lds == NULL)
        {
            ckc_i_set_err(b, CKC_ERR_OOM, "moe-mega prologue: bd_lds view alloc failed");
            return CKC_ERR_OOM;
        }
        bd_lds->base = ctx->Bd_smem;
        bd_lds->addr_space = CKC_ADDR_LDS;
        if (ckc_tensor_descriptor_packed(&bd_lds->desc, bd_shape, 2, storage_dtype) != CKC_OK)
        {
            return ckc_ir_builder_status(b);
        }
        ctx->bd_lds_view = bd_lds;

        /* wd_view = make_global_view(WDown, (1,1,1), storage_dtype, (1,N,1)). */
        int ones3[3] = { 1, 1, 1 };
        ckc_stride_t s_nview[3];
        s_nview[0] = ckc_stride_imm(1);
        s_nview[1] = ckc_stride_value(ctx->N);
        s_nview[2] = ckc_stride_imm(1);
        ckc_tensor_view_t* wd_view = ckc_mega_arena_view(b);
        if (wd_view == NULL)
        {
            ckc_i_set_err(b, CKC_ERR_OOM, "moe-mega prologue: wd_view alloc failed");
            return CKC_ERR_OOM;
        }
        if (ckc_make_global_view(wd_view, ctx->WDown, ones3, 3, storage_dtype, s_nview) != CKC_OK)
        {
            return ckc_ir_builder_status(b);
        }
        ctx->wd_view = wd_view;
    }

    if (!ckc_moe_kloop_plan_init(&ctx->plan_down, b, u_down, ctx->tid))
    {
        return ckc_ir_builder_status(b);
    }
    /* down_operand = _MoeOperand(wd_view, bd_lds_view, Bd_smem). */
    memset(&ctx->down_operand, 0, sizeof ctx->down_operand);
    ctx->down_operand.global_view = ctx->wd_view;
    ctx->down_operand.lds_view    = ctx->bd_lds_view;
    ctx->down_operand.smem        = ctx->Bd_smem;

    /* c_down_k = const_i32(spec.tile_n_inter); down_acc_init = _emit_zero_acc. */
    ctx->c_down_k      = ckc_b_const_i32(b, spec->tile_n_inter);
    ctx->down_acc_init = ckc_gemm_emit_zero_acc(b, u_down);
    ctx->num_down_accs = ctx->down_mfmas_m * ctx->down_mfmas_n;

    return ckc_ir_builder_status(b);
}
