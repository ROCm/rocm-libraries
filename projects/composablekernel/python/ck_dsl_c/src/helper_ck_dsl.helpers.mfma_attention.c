/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of ck_dsl/helpers/mfma_attention.py -- the MFMA-tiled FMHA forward
 * inner body (and its WMMA wave32 analogue). See the header for the symbol map
 * and the byte-fidelity contract. Every ckc_b_* call sequence reproduces the
 * Python builder-call order, operands and compile-time constants op-for-op so
 * the emitted IR is byte-identical to the Python helper's emission.
 */
#include "ckc/helper_ck_dsl.helpers.mfma_attention.h"

#include <stdio.h>
#include <string.h>

#include "ckc/ir.h"
#include "ckc/ir_internal.h" /* ckc_i_set_err */
#include "ckc/helper_ck_dsl.helpers.atoms.h"
#include "ckc/helper_ck_dsl.helpers.attention.h"
#include "ckc/helper_ck_dsl.helpers.distribution.h"
#include "ckc/helper_ck_dsl.core.arch.h"
#include "ckc/arch_target.h"

/* Largest per-lane fragment / accumulator length the attention atoms produce.
 * fp8/bf8 16x16x32 -> a_per_lane=8; wmma f16 -> a_frag_len=16, c_frag_len=8.
 * head_size up to 256 -> n_pv_atoms up to 16, n_qk_atoms up to 16. */
#define CKC_ATTN_MAX_LANE 16
#define CKC_ATTN_MAX_ATOMS 16
#define CKC_ATTN_MAX_ITER_ARGS (2 * CKC_ATTN_MAX_LANE + CKC_ATTN_MAX_ATOMS)

/* ----------------------------------------------------- _ir_type_for_dtype *
 *
 * Python:
 *     if dtype in ("f16", "fp16"): return F16
 *     if dtype == "bf16":          return BF16
 *     raise ValueError(...)
 */
const ckc_type_t* ckc_mfma_attn_ir_type_for_dtype(ckc_ir_builder_t* b, const char* dtype)
{
    if (dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
    {
        return ckc_f16();
    }
    if (dtype != NULL && strcmp(dtype, "bf16") == 0)
    {
        return ckc_bf16();
    }
    if (b != NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "mfma_attention currently supports f16/bf16; got %s",
                      dtype != NULL ? dtype : "None");
    }
    return NULL;
}

/* ----------------------------------------------------- _ATOM_DTYPE_TO_CATALOG */
static const char* ckc_attn_atom_dtype_to_catalog(const char* dtype_in)
{
    if (dtype_in == NULL)
    {
        return NULL;
    }
    if (strcmp(dtype_in, "f16") == 0)
        return "f16";
    if (strcmp(dtype_in, "fp16") == 0)
        return "f16";
    if (strcmp(dtype_in, "bf16") == 0)
        return "bf16";
    if (strcmp(dtype_in, "fp8e4m3") == 0)
        return "fp8";
    if (strcmp(dtype_in, "bf8e5m2") == 0)
        return "bf8";
    if (strcmp(dtype_in, "fp4") == 0)
        return "fp4";
    if (strcmp(dtype_in, "fp6") == 0)
        return "fp6";
    return dtype_in; /* _ATOM_DTYPE_TO_CATALOG.get(x, x) */
}

/* --------------------------------------------------- _validate_attention_atom */
ckc_status_t ckc_validate_attention_atom(ckc_ir_builder_t* b,
                                         const ckc_mfma_atom_t* atom,
                                         const char* arch)
{
    const char* cat_dtype = ckc_attn_atom_dtype_to_catalog(atom->dtype_in);
    const ckc_arch_target_t* target = ckc_arch_target_from_gfx(arch);
    if (target == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "mfma_attention: no target for arch %s",
                      arch != NULL ? arch : "None");
        return CKC_ERR_VALUE;
    }
    if (!ckc_mma_catalog_has_shape(&target->mma, NULL, cat_dtype, cat_dtype, "fp32",
                                   atom->m, atom->n, atom->k))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "mfma_attention: atom %s %dx%dx%d (op_id %s) is not in the "
                      "%s MMA catalog; this kernel config is not legal on %s",
                      atom->dtype_in, atom->m, atom->n, atom->k, atom->name,
                      arch != NULL ? arch : "None", arch != NULL ? arch : "None");
        return CKC_ERR_VALUE;
    }
    return CKC_OK;
}

/* --------------------------------------------------- _load_kv_dequant_packed */
ckc_value_t* ckc_load_kv_dequant_packed(ckc_ir_builder_t* b,
                                        ckc_value_t* src,
                                        ckc_value_t* addr,
                                        int n_elems,
                                        const char* kv_dtype_eff,
                                        const ckc_type_t* kv_dtype_ir,
                                        const ckc_type_t* out_dtype_ir)
{
    bool is_fp8 = (kv_dtype_eff != NULL && strcmp(kv_dtype_eff, "fp8e4m3") == 0);

    if (n_elems % 4 != 0 || n_elems == 0)
    {
        ckc_value_t* out = ckc_b_zero_vec(b, out_dtype_ir, n_elems);
        for (int j = 0; j < n_elems; ++j)
        {
            ckc_value_t* raw =
                ckc_b_global_load(b, src, ckc_b_add(b, addr, ckc_b_const_i32(b, j)), kv_dtype_ir, 1);
            ckc_value_t* f32_v =
                is_fp8 ? ckc_b_cvt_fp8_to_f32(b, raw) : ckc_b_cvt_bf8_to_f32(b, raw);
            out = ckc_b_vec_insert(b, out, ckc_b_cast_f32_to(b, f32_v, out_dtype_ir), j);
        }
        return out;
    }

    ckc_value_t* pk_vec = ckc_b_global_load_vN(b, src, addr, kv_dtype_ir, n_elems, n_elems);
    int num_groups = n_elems / 4;
    ckc_value_t* f32_full = NULL;
    for (int grp = 0; grp < num_groups; ++grp)
    {
        ckc_value_t* chunk = ckc_b_zero_vec(b, kv_dtype_ir, 4);
        for (int j = 0; j < 4; ++j)
        {
            ckc_value_t* scalar = ckc_b_vec_extract(b, pk_vec, grp * 4 + j);
            chunk = ckc_b_vec_insert(b, chunk, scalar, j);
        }
        ckc_value_t* f32_chunk =
            is_fp8 ? ckc_b_cvt_pk_f32_fp8x4(b, chunk) : ckc_b_cvt_pk_f32_bf8x4(b, chunk);
        if (grp == 0)
        {
            f32_full = f32_chunk;
        }
        else
        {
            f32_full = ckc_b_vec_concat(b, f32_full, f32_chunk);
        }
    }
    return ckc_b_vec_cast_f32_to(b, f32_full, out_dtype_ir);
}

/* ------------------------------------------------------- _softmax_row_reduce *
 *
 * Builds the same one-element StaticDistributedTensor over the module-level
 * reduce distribution (_SOFTMAX_ROW_REDUCE_ENC: Rs=(16,), Hs=((1,),),
 * Ps2RHs_major=((0,),), Ps2RHs_minor=((0,),), Ys2RHs_major=(1,),
 * Ys2RHs_minor=(0,)) and folds it via block_tile_reduce_sync (defaults:
 * lds_buf=None, tid=None, wave_size=64). */
ckc_value_t* ckc_softmax_row_reduce(ckc_ir_builder_t* b,
                                    ckc_value_t* scalar,
                                    ckc_reduce_combine_t combine)
{
    /* Rs=(16,) */
    int rs[1] = {16};
    /* Hs=((1,),) -- one X dim, one H level of length 1. */
    int h0_levels[1] = {1};
    ckc_h_row_t hs[1];
    hs[0].levels = h0_levels;
    hs[0].count = 1;
    /* Ps2RHs_major=((0,),), Ps2RHs_minor=((0,),) -- one P dim feeding R major 0. */
    int p0_major[1] = {0};
    int p0_minor[1] = {0};
    ckc_p_seq_t ps[1];
    ps[0].major = p0_major;
    ps[0].minor = p0_minor;
    ps[0].count = 1;
    /* Ys2RHs_major=(1,), Ys2RHs_minor=(0,) -- one keep-row Y on X-dim 0, level 0. */
    int ys_major[1] = {1};
    int ys_minor[1] = {0};

    ckc_tile_distribution_encoding_t* enc =
        ckc_make_tile_distribution_encoding(b, rs, 1, hs, 1, ps, 1, ys_major, ys_minor, 1);
    if (enc == NULL)
    {
        return NULL;
    }
    ckc_tile_distribution_t* dist = ckc_make_static_tile_distribution(b, enc);
    if (dist == NULL)
    {
        return NULL;
    }
    ckc_static_distributed_tensor_t* dt = ckc_make_static_distributed_tensor(b, dist, ckc_f32());
    if (dt == NULL)
    {
        return NULL;
    }
    dt->storage[0] = scalar;
    ckc_block_tile_reduce_sync(b, dt, combine, NULL, NULL, 64);
    return dt->storage[0];
}

/* ============================== MFMA wave64 body ====================== */

static ckc_value_t* ckc_attn_opt(ckc_ir_builder_t* b, ckc_value_t* v)
{
    return v != NULL ? v : ckc_b_const_i32(b, 0);
}

ckc_status_t ckc_mfma_attention_fwd_inner_body(ckc_ir_builder_t* b,
                                               const ckc_mfma_attn_params_t* p)
{
    const char* dtype = (p->dtype != NULL) ? p->dtype : "f16";
    const char* arch = (p->arch != NULL) ? p->arch : "gfx950";
    int head_size = p->head_size;

    if (head_size % CKC_MFMA_ATTN_BLOCK_M != 0)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "mfma_attention head_size %d must be a multiple of %d",
                      head_size, CKC_MFMA_ATTN_BLOCK_M);
        return CKC_ERR_VALUE;
    }
    if (!(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "bf16") == 0))
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "mfma_attention dtype must be f16/bf16, got %s", dtype);
        return CKC_ERR_VALUE;
    }

    /* --- Atom selection (mirrors the Python if/elif cascade). --- */
    const char* kv_dtype = p->kv_dtype;
    const ckc_mfma_atom_t* atom = NULL;
    const char* kv_dtype_eff = NULL;
    if (kv_dtype == NULL || strcmp(kv_dtype, dtype) == 0)
    {
        atom = (strcmp(dtype, "bf16") == 0) ? ckc_mfma_atom("bf16", 16, 16, 16)
                                            : ckc_mfma_atom("f16", 16, 16, 16);
        kv_dtype_eff = dtype;
    }
    else if (strcmp(kv_dtype, "fp8e4m3") == 0)
    {
        atom = p->use_wider_atom ? ckc_mfma_atom("fp8e4m3", 32, 32, 16)
                                 : ckc_mfma_atom("fp8e4m3", 16, 16, 32);
        kv_dtype_eff = "fp8e4m3";
    }
    else if (strcmp(kv_dtype, "bf8e5m2") == 0)
    {
        atom = p->use_wider_atom ? ckc_mfma_atom("bf8e5m2", 32, 32, 16)
                                 : ckc_mfma_atom("bf8e5m2", 16, 16, 32);
        kv_dtype_eff = "bf8e5m2";
    }
    else
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "mfma_attention: unsupported kv_dtype %s; "
                      "expected None / 'f16' / 'fp8e4m3' / 'bf8e5m2'",
                      kv_dtype);
        return CKC_ERR_VALUE;
    }
    if (atom == NULL)
    {
        return ckc_ir_builder_status(b);
    }
    if (head_size % atom->k != 0)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE,
                      "head_size %d must be a multiple of atom.k %d for the selected atom",
                      head_size, atom->k);
        return CKC_ERR_VALUE;
    }

    /* dtype_ir / kv_dtype_ir resolution (matches the Python ternary chain). */
    bool kv_eq_dtype = (strcmp(kv_dtype_eff, dtype) == 0);
    const ckc_type_t* dtype_ir =
        ckc_mfma_attn_ir_type_for_dtype(b, kv_eq_dtype ? dtype : "f16");
    const ckc_type_t* kv_dtype_ir;
    if (kv_eq_dtype)
    {
        kv_dtype_ir = dtype_ir;
    }
    else if (strcmp(kv_dtype_eff, "f16") == 0 || strcmp(kv_dtype_eff, "fp16") == 0)
    {
        kv_dtype_ir = ckc_f16();
    }
    else if (strcmp(kv_dtype_eff, "bf16") == 0)
    {
        kv_dtype_ir = ckc_bf16();
    }
    else
    {
        kv_dtype_ir = dtype_ir;
    }

    /* native_fp8_path adjustments. */
    if (!kv_eq_dtype && !p->native_fp8_path)
    {
        atom = ckc_mfma_atom("f16", 16, 16, 16);
        dtype_ir = ckc_f16();
        kv_dtype_ir = (strcmp(kv_dtype_eff, "fp8e4m3") == 0) ? ckc_fp8e4m3() : ckc_bf8e5m2();
    }
    else if (!kv_eq_dtype && p->native_fp8_path)
    {
        dtype_ir = (strcmp(kv_dtype_eff, "fp8e4m3") == 0) ? ckc_fp8e4m3() : ckc_bf8e5m2();
        kv_dtype_ir = dtype_ir;
    }
    if (atom == NULL || dtype_ir == NULL)
    {
        return ckc_ir_builder_status(b);
    }

    bool fp8_kv = !kv_eq_dtype;

    /* --- Arch / wave dispatch. --- */
    const ckc_arch_target_t* target = ckc_arch_target_from_gfx(arch);
    if (target == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "mfma_attention: no target for arch %s", arch);
        return CKC_ERR_VALUE;
    }
    int wave_size = target->wave_size;

    if (wave_size == 32)
    {
        if (fp8_kv || p->use_wider_atom || p->native_fp8_path)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE,
                          "wave32 (WMMA) attention supports f16/bf16 KV only; "
                          "fp8 / wider-atom / native-fp8 paths are CDNA-only");
            return CKC_ERR_VALUE;
        }
        return ckc_wmma_attention_fwd_inner_body(b, p, p->wmma_v_lds_stage, target);
    }

    /* --- CDNA wave64 (MFMA) path. --- */
    ckc_status_t vst = ckc_validate_attention_atom(b, atom, arch);
    if (vst != CKC_OK)
    {
        return vst;
    }

    const char* qk_op = atom->name; /* target.mma.by_op_id(atom.name).op_id == atom.name */

    int n_qk_atoms = head_size / atom->k;
    int n_pv_atoms = head_size / atom->n;

    ckc_value_t* lane = ckc_b_thread_id_x(b);
    ckc_value_t* c16 = ckc_b_const_i32(b, 16);
    ckc_value_t* m_in_atom = ckc_b_mod(b, lane, c16);
    ckc_value_t* k_blk = ckc_b_div(b, lane, c16);
    ckc_value_t* c_a_per_lane = ckc_b_const_i32(b, atom->a_per_lane);
    ckc_value_t* k_lane_start = ckc_b_mul(b, k_blk, c_a_per_lane);

    ckc_value_t* k_off = ckc_attn_opt(b, p->k_token_offset_elems);
    ckc_value_t* v_off = ckc_attn_opt(b, p->v_token_offset_elems);

    /* ---- Pre-load Q ---- */
    ckc_value_t* q_row = ckc_b_add(b, p->q_tile_base, m_in_atom);
    /* Hoist inner ops into temporaries so emission order is Python's
     * left-to-right (C function-argument evaluation order is unspecified). */
    ckc_value_t* q_arb_t0 = ckc_b_mul(b, q_row, p->stride_q_token);
    ckc_value_t* q_arb_t1 = ckc_b_mul(b, p->head_idx, p->stride_q_head);
    ckc_value_t* q_addr_row_base = ckc_b_add(b, q_arb_t0, q_arb_t1);
    ckc_value_t* q_vecs[CKC_ATTN_MAX_ATOMS];
    for (int ka = 0; ka < n_qk_atoms; ++ka)
    {
        ckc_value_t* d_start = ckc_b_add(
            b, ckc_b_mul(b, ckc_b_const_i32(b, ka), ckc_b_const_i32(b, atom->k)), k_lane_start);
        ckc_value_t* q_addr = ckc_b_add(b, q_addr_row_base, d_start);
        q_vecs[ka] =
            ckc_b_global_load_vN(b, p->Q, q_addr, dtype_ir, atom->a_per_lane, atom->a_per_lane * 2);
    }

    /* ---- LDS for P-operand staging ---- */
    int p_lds_shape[2] = {CKC_MFMA_ATTN_BLOCK_M, CKC_MFMA_ATTN_BLOCK_K};
    ckc_value_t* P_lds = ckc_b_smem_alloc(b, dtype_ir, p_lds_shape, 2, "Pmfma");

    /* ---- Online softmax + PV accumulator iter_args ---- */
    ckc_value_t* neg_inf = ckc_b_const_f32(b, -1e30);
    ckc_value_t* zero_f = ckc_b_const_f32(b, 0.0);
    ckc_value_t* acc_zero = ckc_b_zero_vec_f32(b, atom->c_per_lane);

    ckc_iter_arg_t iter_args[CKC_ATTN_MAX_ITER_ARGS];
    char name_buf[CKC_ATTN_MAX_ITER_ARGS][16];
    int n_ia = 0;
    for (int r = 0; r < atom->c_per_lane; ++r)
    {
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "m%d", r);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = neg_inf;
        ++n_ia;
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "l%d", r);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = zero_f;
        ++n_ia;
    }
    for (int n = 0; n < n_pv_atoms; ++n)
    {
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "acc%d", n);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = acc_zero;
        ++n_ia;
    }

    ckc_value_t* c_block_k = ckc_b_const_i32(b, CKC_MFMA_ATTN_BLOCK_K);
    ckc_value_t* loop_start =
        (p->k_tile_start != NULL) ? p->k_tile_start : ckc_b_const_i32(b, 0);
    ckc_value_t* loop_stop =
        (p->k_tile_stop != NULL) ? p->k_tile_stop : ckc_b_div(b, p->seqlen_k, c_block_k);

    ckc_for_t kloop = ckc_b_scf_for_iter(b, loop_start, loop_stop, ckc_b_const_i32(b, 1),
                                         iter_args, n_ia, "kt", false, false);
    ckc_b_region_enter(b, kloop.body);
    {
        ckc_value_t* kt = kloop.iv;
        ckc_value_t* ms[CKC_ATTN_MAX_LANE];
        ckc_value_t* ls[CKC_ATTN_MAX_LANE];
        ckc_value_t* accs[CKC_ATTN_MAX_ATOMS];
        for (int r = 0; r < atom->c_per_lane; ++r)
        {
            ms[r] = kloop.iter_vars[2 * r];
            ls[r] = kloop.iter_vars[2 * r + 1];
        }
        for (int n = 0; n < n_pv_atoms; ++n)
        {
            accs[n] = kloop.iter_vars[2 * atom->c_per_lane + n];
        }

        ckc_value_t* effective_kt =
            (p->k_block_iter_fn != NULL) ? p->k_block_iter_fn(b, kt, p->k_block_iter_user) : kt;

        ckc_value_t* k_tile_base = ckc_b_mul(b, effective_kt, c_block_k);
        ckc_value_t* k_row_for_lane = ckc_b_add(b, k_tile_base, m_in_atom);

        ckc_value_t* keep_tile =
            (p->extra_mask_predicate != NULL)
                ? p->extra_mask_predicate(b, kt, p->extra_mask_predicate_user)
                : NULL;
        if (p->extra_skip_predicate != NULL)
        {
            ckc_value_t* skip_mask = p->extra_skip_predicate(b, kt, p->extra_skip_predicate_user);
            keep_tile = (keep_tile != NULL) ? ckc_b_land(b, keep_tile, skip_mask) : skip_mask;
        }

        ckc_value_t* k_addr_row_base;
        if (p->k_row_base_fn != NULL)
        {
            k_addr_row_base = p->k_row_base_fn(b, k_row_for_lane, p->k_row_base_user);
        }
        else
        {
            ckc_value_t* k_arb_t0 = ckc_b_mul(b, k_row_for_lane, p->stride_k_token);
            ckc_value_t* k_arb_t1 = ckc_b_mul(b, p->kv_head_idx, p->stride_k_head);
            ckc_value_t* k_arb_t2 = ckc_b_add(b, k_arb_t0, k_arb_t1);
            k_addr_row_base = ckc_b_add(b, k_arb_t2, k_off);
        }

        /* ---- QK MFMA chain ---- */
        ckc_value_t* score = ckc_b_zero_vec_f32(b, atom->c_per_lane);
        for (int ka = 0; ka < n_qk_atoms; ++ka)
        {
            ckc_value_t* d_start = ckc_b_add(
                b, ckc_b_mul(b, ckc_b_const_i32(b, ka), ckc_b_const_i32(b, atom->k)), k_lane_start);
            ckc_value_t* k_addr = ckc_b_add(b, k_addr_row_base, d_start);
            ckc_value_t* k_vec;
            if (fp8_kv)
            {
                k_vec = ckc_load_kv_dequant_packed(b, p->K, k_addr, atom->a_per_lane, kv_dtype_eff,
                                                   kv_dtype_ir, dtype_ir);
            }
            else
            {
                k_vec = ckc_b_global_load_vN(b, p->K, k_addr, dtype_ir, atom->a_per_lane,
                                             atom->a_per_lane * 2);
            }
            score = ckc_b_mma(b, qk_op, q_vecs[ka], k_vec, score, NULL, 0);
        }

        /* ---- Scale + mask + softmax row update ---- */
        ckc_value_t* m_blk = ckc_b_div(b, lane, c16);
        ckc_value_t* new_ms[CKC_ATTN_MAX_LANE];
        ckc_value_t* new_ls[CKC_ATTN_MAX_LANE];
        ckc_value_t* new_accs[CKC_ATTN_MAX_ATOMS];
        ckc_value_t* ps_arr[CKC_ATTN_MAX_LANE];
        for (int n = 0; n < n_pv_atoms; ++n)
        {
            new_accs[n] = accs[n];
        }
        ckc_value_t* q_pos_for_mask =
            (p->q_pos_base != NULL) ? p->q_pos_base : p->q_tile_base;
        for (int r = 0; r < atom->c_per_lane; ++r)
        {
            ckc_value_t* s_r_f32 = ckc_b_vec_extract(b, score, r);
            ckc_value_t* s_r_scaled = ckc_b_fmul(b, s_r_f32, p->scale_log2);
            ckc_value_t* rqp_t0 = ckc_b_mul(b, m_blk, ckc_b_const_i32(b, 4));
            ckc_value_t* rqp_t1 = ckc_b_add(b, q_pos_for_mask, rqp_t0);
            ckc_value_t* row_q_pos = ckc_b_add(b, rqp_t1, ckc_b_const_i32(b, r));
            ckc_value_t* k_col_pos = ckc_b_add(b, k_tile_base, m_in_atom);
            if (p->extra_score_transform != NULL)
            {
                s_r_scaled = p->extra_score_transform(b, s_r_scaled, kt, r,
                                                      p->extra_score_transform_user);
            }
            s_r_scaled = ckc_apply_attention_mask(b, s_r_scaled, p->mask_mode, k_col_pos, row_q_pos,
                                                  p->sliding_window, p->causal_ctx_offset, NULL);
            if (keep_tile != NULL)
            {
                s_r_scaled = ckc_b_select(b, keep_tile, s_r_scaled, neg_inf);
            }
            ckc_value_t* row_max = ckc_softmax_row_reduce(b, s_r_scaled, CKC_REDUCE_MAX);
            ckc_value_t* m_new_r = ckc_b_fmax(b, ms[r], row_max);
            ckc_value_t* alpha_r = ckc_b_exp2(b, ckc_b_fsub(b, ms[r], m_new_r));
            ckc_value_t* p_r = ckc_b_exp2(b, ckc_b_fsub(b, s_r_scaled, m_new_r));
            ckc_value_t* row_psum = ckc_softmax_row_reduce(b, p_r, CKC_REDUCE_SUM);
            ckc_value_t* l_new_r = ckc_b_fadd(b, ckc_b_fmul(b, ls[r], alpha_r), row_psum);

            new_ms[r] = m_new_r;
            new_ls[r] = l_new_r;
            ps_arr[r] = p_r;
            for (int n = 0; n < n_pv_atoms; ++n)
            {
                ckc_value_t* old = ckc_b_vec_extract(b, new_accs[n], r);
                ckc_value_t* rescaled = ckc_b_fmul(b, old, alpha_r);
                new_accs[n] = ckc_b_vec_insert(b, new_accs[n], rescaled, r);
            }
        }

        /* ---- P operand staging via LDS ---- */
        for (int r = 0; r < atom->c_per_lane; ++r)
        {
            ckc_value_t* p_row_t0 = ckc_b_mul(b, m_blk, ckc_b_const_i32(b, 4));
            ckc_value_t* p_row = ckc_b_add(b, p_row_t0, ckc_b_const_i32(b, r));
            ckc_value_t* p_col = m_in_atom;
            ckc_value_t* p_f16 = ckc_b_cast_f32_to(b, ps_arr[r], dtype_ir);
            ckc_value_t* idx[2] = {p_row, p_col};
            ckc_b_smem_store_vN(b, P_lds, idx, 2, p_f16, 1);
        }
        ckc_b_sync(b);

        /* ---- PV MFMA chain ---- */
        for (int nba = 0; nba < n_pv_atoms; ++nba)
        {
            ckc_value_t* p_a_vec = ckc_b_zero_vec(b, dtype_ir, atom->a_per_lane);
            for (int j = 0; j < atom->a_per_lane; ++j)
            {
                ckc_value_t* p_col_j = ckc_b_add(b, k_lane_start, ckc_b_const_i32(b, j));
                ckc_value_t* idx[2] = {m_in_atom, p_col_j};
                ckc_value_t* p_v =
                    ckc_b_vec_extract(b, ckc_b_smem_load_vN(b, P_lds, idx, 2, dtype_ir, 1), 0);
                p_a_vec = ckc_b_vec_insert(b, p_a_vec, p_v, j);
            }
            ckc_value_t* v_col_in_hd = ckc_b_add(
                b, ckc_b_mul(b, ckc_b_const_i32(b, nba), ckc_b_const_i32(b, atom->n)), m_in_atom);
            ckc_value_t* v_a_vec = ckc_b_zero_vec(b, dtype_ir, atom->b_per_lane);
            for (int j = 0; j < atom->b_per_lane; ++j)
            {
                ckc_value_t* v_row_k =
                    ckc_b_add(b, k_tile_base, ckc_b_add(b, k_lane_start, ckc_b_const_i32(b, j)));
                ckc_value_t* v_addr_row_base;
                if (p->v_row_base_fn != NULL)
                {
                    v_addr_row_base = p->v_row_base_fn(b, v_row_k, p->v_row_base_user);
                }
                else
                {
                    ckc_value_t* v_arb_t0 = ckc_b_mul(b, v_row_k, p->stride_v_token);
                    ckc_value_t* v_arb_t1 = ckc_b_mul(b, p->kv_head_idx, p->stride_v_head);
                    ckc_value_t* v_arb_t2 = ckc_b_add(b, v_arb_t0, v_arb_t1);
                    v_addr_row_base = ckc_b_add(b, v_arb_t2, v_off);
                }
                ckc_value_t* v_addr = ckc_b_add(b, v_addr_row_base, v_col_in_hd);
                ckc_value_t* v_scalar;
                if (fp8_kv)
                {
                    ckc_value_t* raw = ckc_b_global_load(b, p->V, v_addr, kv_dtype_ir, 1);
                    ckc_value_t* f32_v = (strcmp(kv_dtype_eff, "fp8e4m3") == 0)
                                             ? ckc_b_cvt_fp8_to_f32(b, raw)
                                             : ckc_b_cvt_bf8_to_f32(b, raw);
                    v_scalar = ckc_b_cast_f32_to(b, f32_v, dtype_ir);
                }
                else
                {
                    v_scalar = ckc_b_global_load(b, p->V, v_addr, dtype_ir, 2);
                }
                v_a_vec = ckc_b_vec_insert(b, v_a_vec, v_scalar, j);
            }
            new_accs[nba] = ckc_b_mma(b, qk_op, p_a_vec, v_a_vec, new_accs[nba], NULL, 0);
        }

        /* ---- Yield updated state ---- */
        ckc_value_t* yields[CKC_ATTN_MAX_ITER_ARGS];
        int ny = 0;
        for (int r = 0; r < atom->c_per_lane; ++r)
        {
            yields[ny++] = new_ms[r];
            yields[ny++] = new_ls[r];
        }
        for (int n = 0; n < n_pv_atoms; ++n)
        {
            yields[ny++] = new_accs[n];
        }
        ckc_b_scf_yield(b, yields, ny);
    }
    ckc_b_region_leave(b);

    /* ---- Pull final state ---- */
    ckc_value_t* ls_final[CKC_ATTN_MAX_LANE];
    ckc_value_t* accs_final[CKC_ATTN_MAX_ATOMS];
    for (int r = 0; r < atom->c_per_lane; ++r)
    {
        ls_final[r] = (kloop.op != NULL) ? kloop.op->results[2 * r + 1] : NULL;
    }
    for (int n = 0; n < n_pv_atoms; ++n)
    {
        accs_final[n] = (kloop.op != NULL) ? kloop.op->results[2 * atom->c_per_lane + n] : NULL;
    }

    /* ---- Epilogue ---- */
    ckc_value_t* m_blk = ckc_b_div(b, lane, c16);
    for (int nba = 0; nba < n_pv_atoms; ++nba)
    {
        for (int r = 0; r < atom->c_per_lane; ++r)
        {
            ckc_value_t* o_row_t0 = ckc_b_mul(b, m_blk, ckc_b_const_i32(b, 4));
            ckc_value_t* o_row_t1 = ckc_b_add(b, p->q_tile_base, o_row_t0);
            ckc_value_t* o_row = ckc_b_add(b, o_row_t1, ckc_b_const_i32(b, r));
            ckc_value_t* o_col_t0 = ckc_b_mul(b, ckc_b_const_i32(b, nba), ckc_b_const_i32(b, atom->n));
            ckc_value_t* o_col = ckc_b_add(b, o_col_t0, m_in_atom);
            ckc_value_t* inv_l = ckc_safe_inv_l(b, ls_final[r]);
            ckc_value_t* v_f32 = ckc_b_fmul(b, ckc_b_vec_extract(b, accs_final[nba], r), inv_l);
            if (p->v_scale != NULL)
            {
                v_f32 = ckc_b_fmul(b, v_f32, p->v_scale);
            }
            ckc_value_t* v_out = ckc_b_cast_f32_to(b, v_f32, dtype_ir);
            ckc_value_t* addr_t0 = ckc_b_mul(b, o_row, p->stride_o_token);
            ckc_value_t* addr_t1 = ckc_b_mul(b, p->head_idx, p->stride_o_head);
            ckc_value_t* addr_t2 = ckc_b_add(b, addr_t0, addr_t1);
            ckc_value_t* addr = ckc_b_add(b, addr_t2, o_col);
            ckc_b_global_store(b, p->O, addr, v_out, 2);
        }
    }

    return ckc_ir_builder_status(b);
}

/* ============================== WMMA wave32 body ====================== */

ckc_status_t ckc_wmma_attention_fwd_inner_body(ckc_ir_builder_t* b,
                                               const ckc_mfma_attn_params_t* p,
                                               bool v_lds_stage,
                                               const ckc_arch_target_t* target)
{
    const char* dtype = (p->dtype != NULL) ? p->dtype : "f16";
    const char* arch = (p->arch != NULL) ? p->arch : "gfx950";
    int head_size = p->head_size;

    if (target == NULL)
    {
        target = ckc_arch_target_from_gfx(arch);
    }
    if (target == NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "wmma_attention: no target for arch %s", arch);
        return CKC_ERR_VALUE;
    }

    const ckc_mma_op_t* op = ckc_mma_catalog_by_op_id(&target->mma, CKC_WMMA_ATTN_OP_ID);
    if (op == NULL || op->family == NULL || strcmp(op->family, "wmma") != 0)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "WMMA attention atom %s absent on %s",
                      CKC_WMMA_ATTN_OP_ID, arch);
        return CKC_ERR_VALUE;
    }
    int wave = op->wave_size;
    const ckc_type_t* dtype_ir = ckc_mfma_attn_ir_type_for_dtype(b, dtype);
    if (dtype_ir == NULL)
    {
        return ckc_ir_builder_status(b);
    }

    const ckc_layout_map_t* a_map = op->a_layout;
    const ckc_layout_map_t* c_map = op->c_layout;
    int a_frag = op->a_frag_len;
    int c_frag = op->c_frag_len;
    int n_dk = head_size / 16;

    /* Python evaluates b.mod(b.thread_id_x(), b.const_i32(wave)) left-to-right:
     * thread_id_x is created before the wave constant. C arg eval order is
     * unspecified (gcc is right-to-left), so hoist the operands into ordered
     * temporaries to match the Python value-creation order exactly. */
    ckc_value_t* tid = ckc_b_thread_id_x(b);
    ckc_value_t* lane = ckc_b_mod(b, tid, ckc_b_const_i32(b, wave));
    ckc_value_t* c16 = ckc_b_const_i32(b, 16);

    ckc_value_t* a_row = NULL;
    ckc_value_t* dummy = NULL;
    ckc_layout_map_coord(a_map, b, lane, 0, &a_row, &dummy);
    ckc_value_t* col = ckc_b_mod(b, lane, c16);

    ckc_value_t* neg_inf = ckc_b_const_f32(b, -1e30);
    ckc_value_t* zero_f = ckc_b_const_f32(b, 0.0);

    ckc_value_t* k_off = ckc_attn_opt(b, p->k_token_offset_elems);
    ckc_value_t* v_off = ckc_attn_opt(b, p->v_token_offset_elems);

    /* ---- Pre-load Q fragments ---- */
    ckc_value_t* q_row = ckc_b_add(b, p->q_tile_base, a_row);
    /* Python: b.add(b.mul(q_row, stride_q_token), b.mul(head_idx, stride_q_head))
     * -- the token mul is created before the head mul (left-to-right). Hoist to
     * fix the C arg-eval order (gcc is right-to-left). */
    ckc_value_t* q_tok_mul = ckc_b_mul(b, q_row, p->stride_q_token);
    ckc_value_t* q_hd_mul = ckc_b_mul(b, p->head_idx, p->stride_q_head);
    ckc_value_t* q_addr_row_base = ckc_b_add(b, q_tok_mul, q_hd_mul);
    ckc_value_t* q_frags[CKC_ATTN_MAX_ATOMS];
    for (int d = 0; d < n_dk; ++d)
    {
        ckc_value_t* q_addr = ckc_b_add(b, q_addr_row_base, ckc_b_const_i32(b, d * 16));
        q_frags[d] = ckc_b_global_load_vN(b, p->Q, q_addr, dtype_ir, a_frag, a_frag * 2);
    }

    /* ---- LDS staging tiles ---- */
    int p_lds_shape[2] = {16, 16};
    ckc_value_t* P_lds = ckc_b_smem_alloc(b, dtype_ir, p_lds_shape, 2, "Pwmma");
    ckc_value_t* V_lds = NULL;
    if (v_lds_stage)
    {
        int v_lds_shape[2] = {16, head_size};
        V_lds = ckc_b_smem_alloc(b, dtype_ir, v_lds_shape, 2, "Vwmma");
    }

    /* ---- Online-softmax + PV accumulator iter-args ---- */
    ckc_iter_arg_t iter_args[CKC_ATTN_MAX_ITER_ARGS];
    char name_buf[CKC_ATTN_MAX_ITER_ARGS][16];
    int n_ia = 0;
    for (int r = 0; r < c_frag; ++r)
    {
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "m%d", r);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = neg_inf;
        ++n_ia;
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "l%d", r);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = zero_f;
        ++n_ia;
    }
    for (int d = 0; d < n_dk; ++d)
    {
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "acc%d", d);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = ckc_b_zero_vec_f32(b, c_frag);
        ++n_ia;
    }

    ckc_value_t* c_block_k = ckc_b_const_i32(b, CKC_MFMA_ATTN_BLOCK_K);
    ckc_value_t* loop_start =
        (p->k_tile_start != NULL) ? p->k_tile_start : ckc_b_const_i32(b, 0);
    ckc_value_t* loop_stop =
        (p->k_tile_stop != NULL) ? p->k_tile_stop : ckc_b_div(b, p->seqlen_k, c_block_k);

    ckc_for_t kloop = ckc_b_scf_for_iter(b, loop_start, loop_stop, ckc_b_const_i32(b, 1),
                                         iter_args, n_ia, "kt", false, false);
    ckc_b_region_enter(b, kloop.body);
    {
        ckc_value_t* kt = kloop.iv;
        ckc_value_t* ms[CKC_ATTN_MAX_LANE];
        ckc_value_t* ls[CKC_ATTN_MAX_LANE];
        ckc_value_t* accs[CKC_ATTN_MAX_ATOMS];
        for (int r = 0; r < c_frag; ++r)
        {
            ms[r] = kloop.iter_vars[2 * r];
            ls[r] = kloop.iter_vars[2 * r + 1];
        }
        for (int d = 0; d < n_dk; ++d)
        {
            accs[d] = kloop.iter_vars[2 * c_frag + d];
        }

        ckc_value_t* effective_kt =
            (p->k_block_iter_fn != NULL) ? p->k_block_iter_fn(b, kt, p->k_block_iter_user) : kt;
        ckc_value_t* k_tile_base = ckc_b_mul(b, effective_kt, c_block_k);
        ckc_value_t* k_row_for_lane = ckc_b_add(b, k_tile_base, a_row);

        ckc_value_t* keep_tile =
            (p->extra_mask_predicate != NULL)
                ? p->extra_mask_predicate(b, kt, p->extra_mask_predicate_user)
                : NULL;
        if (p->extra_skip_predicate != NULL)
        {
            ckc_value_t* skip_mask = p->extra_skip_predicate(b, kt, p->extra_skip_predicate_user);
            keep_tile = (keep_tile != NULL) ? ckc_b_land(b, keep_tile, skip_mask) : skip_mask;
        }

        ckc_value_t* k_addr_row_base;
        if (p->k_row_base_fn != NULL)
        {
            k_addr_row_base = p->k_row_base_fn(b, k_row_for_lane, p->k_row_base_user);
        }
        else
        {
            /* Python: add(add(mul(k_row_for_lane, stride_k_token),
             *               mul(kv_head_idx, stride_k_head)), k_off)
             * -- token mul created before head mul. Hoist for arg-eval order. */
            ckc_value_t* k_tok_mul = ckc_b_mul(b, k_row_for_lane, p->stride_k_token);
            ckc_value_t* k_hd_mul = ckc_b_mul(b, p->kv_head_idx, p->stride_k_head);
            k_addr_row_base = ckc_b_add(b, ckc_b_add(b, k_tok_mul, k_hd_mul), k_off);
        }

        /* ---- QK^T WMMA chain ---- */
        ckc_value_t* score = ckc_b_zero_vec_f32(b, c_frag);
        for (int d = 0; d < n_dk; ++d)
        {
            ckc_value_t* k_addr = ckc_b_add(b, k_addr_row_base, ckc_b_const_i32(b, d * 16));
            ckc_value_t* k_frag = ckc_b_global_load_vN(b, p->K, k_addr, dtype_ir, a_frag, a_frag * 2);
            score = ckc_b_mma(b, op->op_id, q_frags[d], k_frag, score, NULL, 0);
        }

        /* ---- Scale + mask + per-row online softmax ---- */
        ckc_value_t* new_ms[CKC_ATTN_MAX_LANE];
        ckc_value_t* new_ls[CKC_ATTN_MAX_LANE];
        ckc_value_t* new_accs[CKC_ATTN_MAX_ATOMS];
        ckc_value_t* ps_arr[CKC_ATTN_MAX_LANE];
        for (int d = 0; d < n_dk; ++d)
        {
            new_accs[d] = accs[d];
        }
        ckc_value_t* q_pos_for_mask =
            (p->q_pos_base != NULL) ? p->q_pos_base : p->q_tile_base;
        for (int r = 0; r < c_frag; ++r)
        {
            ckc_value_t* row_rel = NULL;
            ckc_value_t* col_k = NULL;
            ckc_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
            ckc_value_t* s_r = ckc_b_fmul(b, ckc_b_vec_extract(b, score, r), p->scale_log2);
            ckc_value_t* row_q_pos = ckc_b_add(b, q_pos_for_mask, row_rel);
            ckc_value_t* k_col_pos = ckc_b_add(b, k_tile_base, col_k);
            if (p->extra_score_transform != NULL)
            {
                s_r = p->extra_score_transform(b, s_r, kt, r, p->extra_score_transform_user);
            }
            s_r = ckc_apply_attention_mask(b, s_r, p->mask_mode, k_col_pos, row_q_pos,
                                           p->sliding_window, p->causal_ctx_offset, NULL);
            if (keep_tile != NULL)
            {
                s_r = ckc_b_select(b, keep_tile, s_r, neg_inf);
            }
            ckc_value_t* row_max = ckc_softmax_row_reduce(b, s_r, CKC_REDUCE_MAX);
            ckc_value_t* m_new = ckc_b_fmax(b, ms[r], row_max);
            ckc_value_t* alpha = ckc_b_exp2(b, ckc_b_fsub(b, ms[r], m_new));
            ckc_value_t* p_r = ckc_b_exp2(b, ckc_b_fsub(b, s_r, m_new));
            ckc_value_t* row_sum = ckc_softmax_row_reduce(b, p_r, CKC_REDUCE_SUM);
            ckc_value_t* l_new = ckc_b_fadd(b, ckc_b_fmul(b, ls[r], alpha), row_sum);
            new_ms[r] = m_new;
            new_ls[r] = l_new;
            ps_arr[r] = p_r;
            for (int d = 0; d < n_dk; ++d)
            {
                ckc_value_t* old = ckc_b_vec_extract(b, new_accs[d], r);
                new_accs[d] = ckc_b_vec_insert(b, new_accs[d], ckc_b_fmul(b, old, alpha), r);
            }
        }

        /* ---- V staging into LDS ---- */
        if (v_lds_stage)
        {
            ckc_value_t* v_stage_row = ckc_b_add(b, k_tile_base, a_row);
            ckc_value_t* v_stage_base;
            if (p->v_row_base_fn != NULL)
            {
                v_stage_base = p->v_row_base_fn(b, v_stage_row, p->v_row_base_user);
            }
            else
            {
                /* Python: token mul before head mul (left-to-right). */
                ckc_value_t* vs_tok_mul = ckc_b_mul(b, v_stage_row, p->stride_v_token);
                ckc_value_t* vs_hd_mul = ckc_b_mul(b, p->kv_head_idx, p->stride_v_head);
                v_stage_base = ckc_b_add(b, ckc_b_add(b, vs_tok_mul, vs_hd_mul), v_off);
            }
            for (int e = 0; e < head_size / 8; ++e)
            {
                ckc_value_t* v_g = ckc_b_global_load_vN(
                    b, p->V, ckc_b_add(b, v_stage_base, ckc_b_const_i32(b, e * 8)), dtype_ir, 8, 16);
                ckc_value_t* idx[2] = {a_row, ckc_b_const_i32(b, e * 8)};
                ckc_b_smem_store_vN(b, V_lds, idx, 2, v_g, 8);
            }
        }

        /* ---- P staging through LDS ---- */
        for (int r = 0; r < c_frag; ++r)
        {
            ckc_value_t* row_rel = NULL;
            ckc_value_t* col_k = NULL;
            ckc_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
            ckc_value_t* idx[2] = {row_rel, col_k};
            ckc_b_smem_store_vN(b, P_lds, idx, 2, ckc_b_cast_f32_to(b, ps_arr[r], dtype_ir), 1);
        }
        ckc_b_sync(b);

        /* ---- V load + PV WMMA chain ---- */
        ckc_value_t* p_a = ckc_b_zero_vec(b, dtype_ir, a_frag);
        for (int j = 0; j < a_frag; ++j)
        {
            ckc_value_t* a_k = NULL;
            ckc_value_t* a_dummy = NULL;
            ckc_layout_map_coord(a_map, b, lane, j, &a_dummy, &a_k);
            ckc_value_t* idx[2] = {a_row, a_k};
            ckc_value_t* p_v =
                ckc_b_vec_extract(b, ckc_b_smem_load_vN(b, P_lds, idx, 2, dtype_ir, 1), 0);
            p_a = ckc_b_vec_insert(b, p_a, p_v, j);
        }

        for (int d = 0; d < n_dk; ++d)
        {
            ckc_value_t* d_col = ckc_b_add(b, ckc_b_const_i32(b, d * 16), col);
            ckc_value_t* v_b = ckc_b_zero_vec(b, dtype_ir, a_frag);
            for (int j = 0; j < a_frag; ++j)
            {
                ckc_value_t* v_elem;
                if (v_lds_stage)
                {
                    ckc_value_t* idx[2] = {ckc_b_const_i32(b, j), d_col};
                    v_elem = ckc_b_vec_extract(b, ckc_b_smem_load_vN(b, V_lds, idx, 2, dtype_ir, 1), 0);
                }
                else
                {
                    ckc_value_t* v_row = ckc_b_add(b, k_tile_base, ckc_b_const_i32(b, j));
                    ckc_value_t* v_row_base;
                    if (p->v_row_base_fn != NULL)
                    {
                        v_row_base = p->v_row_base_fn(b, v_row, p->v_row_base_user);
                    }
                    else
                    {
                        /* Python: token mul before head mul (left-to-right). */
                        ckc_value_t* v_tok_mul = ckc_b_mul(b, v_row, p->stride_v_token);
                        ckc_value_t* v_hd_mul = ckc_b_mul(b, p->kv_head_idx, p->stride_v_head);
                        v_row_base = ckc_b_add(b, ckc_b_add(b, v_tok_mul, v_hd_mul), v_off);
                    }
                    v_elem = ckc_b_global_load(b, p->V, ckc_b_add(b, v_row_base, d_col), dtype_ir, 2);
                }
                v_b = ckc_b_vec_insert(b, v_b, v_elem, j);
            }
            new_accs[d] = ckc_b_mma(b, op->op_id, p_a, v_b, new_accs[d], NULL, 0);
        }

        ckc_value_t* yields[CKC_ATTN_MAX_ITER_ARGS];
        int ny = 0;
        for (int r = 0; r < c_frag; ++r)
        {
            yields[ny++] = new_ms[r];
            yields[ny++] = new_ls[r];
        }
        for (int d = 0; d < n_dk; ++d)
        {
            yields[ny++] = new_accs[d];
        }
        ckc_b_scf_yield(b, yields, ny);
    }
    ckc_b_region_leave(b);

    ckc_value_t* ls_final[CKC_ATTN_MAX_LANE];
    ckc_value_t* accs_final[CKC_ATTN_MAX_ATOMS];
    for (int r = 0; r < c_frag; ++r)
    {
        ls_final[r] = (kloop.op != NULL) ? kloop.op->results[2 * r + 1] : NULL;
    }
    for (int d = 0; d < n_dk; ++d)
    {
        accs_final[d] = (kloop.op != NULL) ? kloop.op->results[2 * c_frag + d] : NULL;
    }

    /* ---- Epilogue ---- */
    for (int d = 0; d < n_dk; ++d)
    {
        for (int r = 0; r < c_frag; ++r)
        {
            ckc_value_t* row_rel = NULL;
            ckc_value_t* col_n = NULL;
            ckc_layout_map_coord(c_map, b, lane, r, &row_rel, &col_n);
            ckc_value_t* l_safe = ls_final[r];
            ckc_value_t* zero_mask = ckc_b_fcmp(b, "oeq", l_safe, zero_f);
            ckc_value_t* inv_l = ckc_b_select(b, zero_mask, zero_f, ckc_b_rcp(b, l_safe));
            ckc_value_t* v_f32 = ckc_b_fmul(b, ckc_b_vec_extract(b, accs_final[d], r), inv_l);
            if (p->v_scale != NULL)
            {
                v_f32 = ckc_b_fmul(b, v_f32, p->v_scale);
            }
            ckc_value_t* o_row = ckc_b_add(b, p->q_tile_base, row_rel);
            ckc_value_t* o_col = ckc_b_add(b, ckc_b_const_i32(b, d * 16), col_n);
            /* Python: token mul before head mul (left-to-right). */
            ckc_value_t* o_tok_mul = ckc_b_mul(b, o_row, p->stride_o_token);
            ckc_value_t* o_hd_mul = ckc_b_mul(b, p->head_idx, p->stride_o_head);
            ckc_value_t* o_addr =
                ckc_b_add(b, ckc_b_add(b, o_tok_mul, o_hd_mul), o_col);
            ckc_b_global_store(b, p->O, o_addr, ckc_b_cast_f32_to(b, v_f32, dtype_ir), 2);
        }
    }

    return ckc_ir_builder_status(b);
}

/* ===========================================================================
 * Additional symbols ported from ck_dsl.helpers.attention.
 * ===========================================================================
 */

/* ----------------------------------------------------- mfma_32x32x16_for_dtype *
 *
 * Python:
 *     if dtype.name == "f16":     return b.mfma_f32_32x32x16_f16(a, bv, c)
 *     if dtype.name == "bf16":    return b.mfma_f32_32x32x16_bf16(a, bv, c)
 *     if dtype.name == "fp8e4m3": return b.mfma_f32_32x32x16_fp8(a, bv, c)
 *     raise ValueError(f"unsupported MFMA 32x32x16 dtype {dtype.name}")
 */
ckc_value_t* ckc_mfma_attn_mfma_32x32x16_for_dtype(ckc_ir_builder_t* b,
                                                   const ckc_type_t* dtype,
                                                   ckc_value_t* a,
                                                   ckc_value_t* bv,
                                                   ckc_value_t* c)
{
    if (dtype == NULL || dtype->name == NULL)
    {
        if (b != NULL)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "unsupported MFMA 32x32x16 dtype (null)");
        }
        return NULL;
    }
    if (strcmp(dtype->name, "f16") == 0)
    {
        return ckc_b_mfma_f32_32x32x16_f16(b, a, bv, c);
    }
    if (strcmp(dtype->name, "bf16") == 0)
    {
        return ckc_b_mfma_f32_32x32x16_bf16(b, a, bv, c);
    }
    if (strcmp(dtype->name, "fp8e4m3") == 0)
    {
        return ckc_b_mfma_f32_32x32x16_fp8(b, a, bv, c);
    }
    if (b != NULL)
    {
        ckc_i_set_err(b, CKC_ERR_VALUE, "unsupported MFMA 32x32x16 dtype %s", dtype->name);
    }
    return NULL;
}

/* ----------------------------------------------------- dequant_fp8x8_to_dtype *
 *
 * Python:
 *     lo_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4)], FP8E4M3)
 *     hi_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4, 8)], FP8E4M3)
 *     lo_f32 = b.cvt_pk_f32_fp8x4(lo_fp8)
 *     hi_f32 = b.cvt_pk_f32_fp8x4(hi_fp8)
 *     deq = [b.cast_f32_to(b.fmul(b.vec_extract(lo_f32, i), scale), dtype) for i in range(4)]
 *         + [b.cast_f32_to(b.fmul(b.vec_extract(hi_f32, i), scale), dtype) for i in range(4)]
 *     return b.vec_pack(deq, dtype)
 *
 * FP8E4M3 is the imported singleton scalar type in the Python; bind to the ir.h
 * accessor. Order of emission matches the Python list comprehension exactly:
 * lo quad (extract*4 -> pack), hi quad (extract*4 -> pack), cvt lo then hi,
 * then 8 (vec_extract -> fmul -> cast_f32_to) triples (lo 0..3 then hi 0..3),
 * then the final vec_pack(dtype).
 */
ckc_value_t* ckc_mfma_attn_dequant_fp8x8_to_dtype(ckc_ir_builder_t* b,
                                                  ckc_value_t* fp8_vec,
                                                  ckc_value_t* scale,
                                                  const ckc_type_t* dtype)
{
    const ckc_type_t* fp8e4m3 = ckc_fp8e4m3();
    ckc_value_t* lo_comp[4];
    ckc_value_t* hi_comp[4];
    ckc_value_t* lo_fp8;
    ckc_value_t* hi_fp8;
    ckc_value_t* lo_f32;
    ckc_value_t* hi_f32;
    ckc_value_t* deq[8];
    int i;

    if (dtype == NULL)
    {
        if (b != NULL)
        {
            ckc_i_set_err(b, CKC_ERR_VALUE, "dequant_fp8x8_to_dtype: dtype is NULL");
        }
        return NULL;
    }

    /* lo_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4)], FP8E4M3) */
    for (i = 0; i < 4; ++i)
    {
        lo_comp[i] = ckc_b_vec_extract(b, fp8_vec, i);
    }
    lo_fp8 = ckc_b_vec_pack(b, lo_comp, 4, fp8e4m3);

    /* hi_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4, 8)], FP8E4M3) */
    for (i = 0; i < 4; ++i)
    {
        hi_comp[i] = ckc_b_vec_extract(b, fp8_vec, 4 + i);
    }
    hi_fp8 = ckc_b_vec_pack(b, hi_comp, 4, fp8e4m3);

    /* lo_f32 = b.cvt_pk_f32_fp8x4(lo_fp8); hi_f32 = b.cvt_pk_f32_fp8x4(hi_fp8) */
    lo_f32 = ckc_b_cvt_pk_f32_fp8x4(b, lo_fp8);
    hi_f32 = ckc_b_cvt_pk_f32_fp8x4(b, hi_fp8);

    /* deq lo lanes 0..3, then hi lanes 0..3; each: vec_extract -> fmul -> cast */
    for (i = 0; i < 4; ++i)
    {
        deq[i] = ckc_b_cast_f32_to(b, ckc_b_fmul(b, ckc_b_vec_extract(b, lo_f32, i), scale), dtype);
    }
    for (i = 0; i < 4; ++i)
    {
        deq[4 + i] =
            ckc_b_cast_f32_to(b, ckc_b_fmul(b, ckc_b_vec_extract(b, hi_f32, i), scale), dtype);
    }

    /* return b.vec_pack(deq, dtype) */
    return ckc_b_vec_pack(b, deq, 8, dtype);
}
