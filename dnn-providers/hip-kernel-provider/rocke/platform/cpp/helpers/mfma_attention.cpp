// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/mfma_attention.py -- the MFMA-tiled FMHA forward
 * inner body (and its WMMA wave32 analogue). See the header for the symbol map
 * and the byte-fidelity contract. Every rocke_b_* call sequence reproduces the
 * Python builder-call order, operands and compile-time constants op-for-op so
 * the emitted IR is byte-identical to the Python helper's emission.
 */
#include "rocke/helper_rocke.helpers.mfma_attention.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* Largest per-lane fragment / accumulator length the attention atoms produce.
 * fp8/bf8 16x16x32 -> a_per_lane=8; wmma f16 -> a_frag_len=16, c_frag_len=8.
 * head_size up to 256 -> n_pv_atoms up to 16, n_qk_atoms up to 16. */
#define ROCKE_ATTN_MAX_LANE 16
#define ROCKE_ATTN_MAX_ATOMS 16
#define ROCKE_ATTN_MAX_ITER_ARGS (2 * ROCKE_ATTN_MAX_LANE + ROCKE_ATTN_MAX_ATOMS)

/* ----------------------------------------------------- _ir_type_for_dtype *
 *
 * Python:
 *     if dtype in ("f16", "fp16"): return F16
 *     if dtype == "bf16":          return BF16
 *     raise ValueError(...)
 */
const rocke_type_t* rocke_mfma_attn_ir_type_for_dtype(rocke_ir_builder_t* b, const char* dtype)
{
    if(dtype != NULL && (strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0))
    {
        return rocke_f16();
    }
    if(dtype != NULL && strcmp(dtype, "bf16") == 0)
    {
        return rocke_bf16();
    }
    if(b != NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "mfma_attention currently supports f16/bf16; got %s",
                        dtype != NULL ? dtype : "None");
    }
    return NULL;
}

/* ----------------------------------------------------- _ATOM_DTYPE_TO_CATALOG */
static const char* rocke_attn_atom_dtype_to_catalog(const char* dtype_in)
{
    if(dtype_in == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype_in, "f16") == 0)
        return "f16";
    if(strcmp(dtype_in, "fp16") == 0)
        return "f16";
    if(strcmp(dtype_in, "bf16") == 0)
        return "bf16";
    if(strcmp(dtype_in, "fp8e4m3") == 0)
        return "fp8";
    if(strcmp(dtype_in, "bf8e5m2") == 0)
        return "bf8";
    if(strcmp(dtype_in, "fp4") == 0)
        return "fp4";
    if(strcmp(dtype_in, "fp6") == 0)
        return "fp6";
    return dtype_in; /* _ATOM_DTYPE_TO_CATALOG.get(x, x) */
}

/* --------------------------------------------------- _validate_attention_atom */
rocke_status_t rocke_validate_attention_atom(rocke_ir_builder_t* b,
                                             const rocke_mfma_atom_t* atom,
                                             const char* arch)
{
    const char* cat_dtype = rocke_attn_atom_dtype_to_catalog(atom->dtype_in);
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "mfma_attention: no target for arch %s",
                        arch != NULL ? arch : "None");
        return ROCKE_ERR_VALUE;
    }
    if(!rocke_mma_catalog_has_shape(
           &target->mma, NULL, cat_dtype, cat_dtype, "fp32", atom->m, atom->n, atom->k))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "mfma_attention: atom %s %dx%dx%d (op_id %s) is not in the "
                        "%s MMA catalog; this kernel config is not legal on %s",
                        atom->dtype_in,
                        atom->m,
                        atom->n,
                        atom->k,
                        atom->name,
                        arch != NULL ? arch : "None",
                        arch != NULL ? arch : "None");
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* --------------------------------------------------- _load_kv_dequant_packed */
rocke_value_t* rocke_load_kv_dequant_packed(rocke_ir_builder_t* b,
                                            rocke_value_t* src,
                                            rocke_value_t* addr,
                                            int n_elems,
                                            const char* kv_dtype_eff,
                                            const rocke_type_t* kv_dtype_ir,
                                            const rocke_type_t* out_dtype_ir)
{
    bool is_fp8 = (kv_dtype_eff != NULL && strcmp(kv_dtype_eff, "fp8e4m3") == 0);

    if(n_elems % 4 != 0 || n_elems == 0)
    {
        rocke_value_t* out = rocke_b_zero_vec(b, out_dtype_ir, n_elems);
        for(int j = 0; j < n_elems; ++j)
        {
            rocke_value_t* raw = rocke_b_global_load(
                b, src, rocke_b_add(b, addr, rocke_b_const_i32(b, j)), kv_dtype_ir, 1);
            rocke_value_t* f32_v
                = is_fp8 ? rocke_b_cvt_fp8_to_f32(b, raw) : rocke_b_cvt_bf8_to_f32(b, raw);
            out = rocke_b_vec_insert(b, out, rocke_b_cast_f32_to(b, f32_v, out_dtype_ir), j);
        }
        return out;
    }

    rocke_value_t* pk_vec = rocke_b_global_load_vN(b, src, addr, kv_dtype_ir, n_elems, n_elems);
    int num_groups = n_elems / 4;
    rocke_value_t* f32_full = NULL;
    for(int grp = 0; grp < num_groups; ++grp)
    {
        rocke_value_t* chunk = rocke_b_zero_vec(b, kv_dtype_ir, 4);
        for(int j = 0; j < 4; ++j)
        {
            rocke_value_t* scalar = rocke_b_vec_extract(b, pk_vec, grp * 4 + j);
            chunk = rocke_b_vec_insert(b, chunk, scalar, j);
        }
        rocke_value_t* f32_chunk
            = is_fp8 ? rocke_b_cvt_pk_f32_fp8x4(b, chunk) : rocke_b_cvt_pk_f32_bf8x4(b, chunk);
        if(grp == 0)
        {
            f32_full = f32_chunk;
        }
        else
        {
            f32_full = rocke_b_vec_concat(b, f32_full, f32_chunk);
        }
    }
    return rocke_b_vec_cast_f32_to(b, f32_full, out_dtype_ir);
}

/* ------------------------------------------------------- _softmax_row_reduce *
 *
 * Builds the same one-element StaticDistributedTensor over the module-level
 * reduce distribution (_SOFTMAX_ROW_REDUCE_ENC: Rs=(16,), Hs=((1,),),
 * Ps2RHs_major=((0,),), Ps2RHs_minor=((0,),), Ys2RHs_major=(1,),
 * Ys2RHs_minor=(0,)) and folds it via block_tile_reduce_sync (defaults:
 * lds_buf=None, tid=None, wave_size=64). */
rocke_value_t* rocke_softmax_row_reduce(rocke_ir_builder_t* b,
                                        rocke_value_t* scalar,
                                        rocke_reduce_combine_t combine)
{
    /* Rs=(16,) */
    int rs[1] = {16};
    /* Hs=((1,),) -- one X dim, one H level of length 1. */
    int h0_levels[1] = {1};
    rocke_h_row_t hs[1];
    hs[0].levels = h0_levels;
    hs[0].count = 1;
    /* Ps2RHs_major=((0,),), Ps2RHs_minor=((0,),) -- one P dim feeding R major 0. */
    int p0_major[1] = {0};
    int p0_minor[1] = {0};
    rocke_p_seq_t ps[1];
    ps[0].major = p0_major;
    ps[0].minor = p0_minor;
    ps[0].count = 1;
    /* Ys2RHs_major=(1,), Ys2RHs_minor=(0,) -- one keep-row Y on X-dim 0, level 0. */
    int ys_major[1] = {1};
    int ys_minor[1] = {0};

    rocke_tile_distribution_encoding_t* enc
        = rocke_make_tile_distribution_encoding(b, rs, 1, hs, 1, ps, 1, ys_major, ys_minor, 1);
    if(enc == NULL)
    {
        return NULL;
    }
    rocke_tile_distribution_t* dist = rocke_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return NULL;
    }
    rocke_static_distributed_tensor_t* dt
        = rocke_make_static_distributed_tensor(b, dist, rocke_f32());
    if(dt == NULL)
    {
        return NULL;
    }
    dt->storage[0] = scalar;
    rocke_block_tile_reduce_sync(b, dt, combine, NULL, NULL, 64);
    return dt->storage[0];
}

/* ============================== MFMA wave64 body ====================== */

static rocke_value_t* rocke_attn_opt(rocke_ir_builder_t* b, rocke_value_t* v)
{
    return v != NULL ? v : rocke_b_const_i32(b, 0);
}

rocke_status_t rocke_mfma_attention_fwd_inner_body(rocke_ir_builder_t* b,
                                                   const rocke_mfma_attn_params_t* p)
{
    const char* dtype = (p->dtype != NULL) ? p->dtype : "f16";
    const char* arch = (p->arch != NULL) ? p->arch : "gfx950";
    int head_size = p->head_size;

    if(head_size % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "mfma_attention head_size %d must be a multiple of %d",
                        head_size,
                        ROCKE_MFMA_ATTN_BLOCK_M);
        return ROCKE_ERR_VALUE;
    }
    if(!(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0 || strcmp(dtype, "bf16") == 0))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "mfma_attention dtype must be f16/bf16, got %s", dtype);
        return ROCKE_ERR_VALUE;
    }

    /* --- Atom selection (mirrors the Python if/elif cascade). --- */
    const char* kv_dtype = p->kv_dtype;
    const rocke_mfma_atom_t* atom = NULL;
    const char* kv_dtype_eff = NULL;
    if(kv_dtype == NULL || strcmp(kv_dtype, dtype) == 0)
    {
        atom = (strcmp(dtype, "bf16") == 0) ? rocke_mfma_atom("bf16", 16, 16, 16)
                                            : rocke_mfma_atom("f16", 16, 16, 16);
        kv_dtype_eff = dtype;
    }
    else if(strcmp(kv_dtype, "fp8e4m3") == 0)
    {
        atom = p->use_wider_atom ? rocke_mfma_atom("fp8e4m3", 32, 32, 16)
                                 : rocke_mfma_atom("fp8e4m3", 16, 16, 32);
        kv_dtype_eff = "fp8e4m3";
    }
    else if(strcmp(kv_dtype, "bf8e5m2") == 0)
    {
        atom = p->use_wider_atom ? rocke_mfma_atom("bf8e5m2", 32, 32, 16)
                                 : rocke_mfma_atom("bf8e5m2", 16, 16, 32);
        kv_dtype_eff = "bf8e5m2";
    }
    else
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "mfma_attention: unsupported kv_dtype %s; "
                        "expected None / 'f16' / 'fp8e4m3' / 'bf8e5m2'",
                        kv_dtype);
        return ROCKE_ERR_VALUE;
    }
    if(atom == NULL)
    {
        return rocke_ir_builder_status(b);
    }
    if(head_size % atom->k != 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "head_size %d must be a multiple of atom.k %d for the selected atom",
                        head_size,
                        atom->k);
        return ROCKE_ERR_VALUE;
    }

    /* dtype_ir / kv_dtype_ir resolution (matches the Python ternary chain). */
    bool kv_eq_dtype = (strcmp(kv_dtype_eff, dtype) == 0);
    const rocke_type_t* dtype_ir
        = rocke_mfma_attn_ir_type_for_dtype(b, kv_eq_dtype ? dtype : "f16");
    const rocke_type_t* kv_dtype_ir;
    if(kv_eq_dtype)
    {
        kv_dtype_ir = dtype_ir;
    }
    else if(strcmp(kv_dtype_eff, "f16") == 0 || strcmp(kv_dtype_eff, "fp16") == 0)
    {
        kv_dtype_ir = rocke_f16();
    }
    else if(strcmp(kv_dtype_eff, "bf16") == 0)
    {
        kv_dtype_ir = rocke_bf16();
    }
    else
    {
        kv_dtype_ir = dtype_ir;
    }

    /* native_fp8_path adjustments. */
    if(!kv_eq_dtype && !p->native_fp8_path)
    {
        atom = rocke_mfma_atom("f16", 16, 16, 16);
        dtype_ir = rocke_f16();
        kv_dtype_ir = (strcmp(kv_dtype_eff, "fp8e4m3") == 0) ? rocke_fp8e4m3() : rocke_bf8e5m2();
    }
    else if(!kv_eq_dtype && p->native_fp8_path)
    {
        dtype_ir = (strcmp(kv_dtype_eff, "fp8e4m3") == 0) ? rocke_fp8e4m3() : rocke_bf8e5m2();
        kv_dtype_ir = dtype_ir;
    }
    if(atom == NULL || dtype_ir == NULL)
    {
        return rocke_ir_builder_status(b);
    }

    bool fp8_kv = !kv_eq_dtype;

    /* --- Arch / wave dispatch. --- */
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "mfma_attention: no target for arch %s", arch);
        return ROCKE_ERR_VALUE;
    }
    int wave_size = target->wave_size;

    if(wave_size == 32)
    {
        if(fp8_kv || p->use_wider_atom || p->native_fp8_path)
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "wave32 (WMMA) attention supports f16/bf16 KV only; "
                            "fp8 / wider-atom / native-fp8 paths are CDNA-only");
            return ROCKE_ERR_VALUE;
        }
        return rocke_wmma_attention_fwd_inner_body(b, p, p->wmma_v_lds_stage, target);
    }

    /* --- CDNA wave64 (MFMA) path. --- */
    rocke_status_t vst = rocke_validate_attention_atom(b, atom, arch);
    if(vst != ROCKE_OK)
    {
        return vst;
    }

    const char* qk_op = atom->name; /* target.mma.by_op_id(atom.name).op_id == atom.name */

    int n_qk_atoms = head_size / atom->k;
    int n_pv_atoms = head_size / atom->n;

    rocke_value_t* lane = rocke_b_thread_id_x(b);
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* m_in_atom = rocke_b_mod(b, lane, c16);
    rocke_value_t* k_blk = rocke_b_div(b, lane, c16);
    rocke_value_t* c_a_per_lane = rocke_b_const_i32(b, atom->a_per_lane);
    rocke_value_t* k_lane_start = rocke_b_mul(b, k_blk, c_a_per_lane);

    rocke_value_t* k_off = rocke_attn_opt(b, p->k_token_offset_elems);
    rocke_value_t* v_off = rocke_attn_opt(b, p->v_token_offset_elems);

    /* ---- Pre-load Q ---- */
    rocke_value_t* q_row = rocke_b_add(b, p->q_tile_base, m_in_atom);
    /* Hoist inner ops into temporaries so emission order is Python's
     * left-to-right (C function-argument evaluation order is unspecified). */
    rocke_value_t* q_arb_t0 = rocke_b_mul(b, q_row, p->stride_q_token);
    rocke_value_t* q_arb_t1 = rocke_b_mul(b, p->head_idx, p->stride_q_head);
    rocke_value_t* q_addr_row_base = rocke_b_add(b, q_arb_t0, q_arb_t1);
    rocke_value_t* q_vecs[ROCKE_ATTN_MAX_ATOMS];
    for(int ka = 0; ka < n_qk_atoms; ++ka)
    {
        rocke_value_t* d_ka = rocke_b_const_i32(b, ka);
        rocke_value_t* d_atk = rocke_b_const_i32(b, atom->k);
        rocke_value_t* d_start = rocke_b_add(b, rocke_b_mul(b, d_ka, d_atk), k_lane_start);
        rocke_value_t* q_addr = rocke_b_add(b, q_addr_row_base, d_start);
        q_vecs[ka] = rocke_b_global_load_vN(
            b, p->Q, q_addr, dtype_ir, atom->a_per_lane, atom->a_per_lane * 2);
    }

    /* ---- LDS for P-operand staging ---- */
    int p_lds_shape[2] = {ROCKE_MFMA_ATTN_BLOCK_M, ROCKE_MFMA_ATTN_BLOCK_K};
    rocke_value_t* P_lds = rocke_b_smem_alloc(b, dtype_ir, p_lds_shape, 2, "Pmfma");

    /* ---- Online softmax + PV accumulator iter_args ---- */
    rocke_value_t* neg_inf = rocke_b_const_f32(b, -1e30);
    rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);
    rocke_value_t* acc_zero = rocke_b_zero_vec_f32(b, atom->c_per_lane);

    rocke_iter_arg_t iter_args[ROCKE_ATTN_MAX_ITER_ARGS];
    char name_buf[ROCKE_ATTN_MAX_ITER_ARGS][16];
    int n_ia = 0;
    for(int r = 0; r < atom->c_per_lane; ++r)
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
    for(int n = 0; n < n_pv_atoms; ++n)
    {
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "acc%d", n);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = acc_zero;
        ++n_ia;
    }

    rocke_value_t* c_block_k = rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_K);
    rocke_value_t* loop_start
        = (p->k_tile_start != NULL) ? p->k_tile_start : rocke_b_const_i32(b, 0);
    rocke_value_t* loop_stop
        = (p->k_tile_stop != NULL) ? p->k_tile_stop : rocke_b_div(b, p->seqlen_k, c_block_k);

    rocke_for_t kloop = rocke_b_scf_for_iter(
        b, loop_start, loop_stop, rocke_b_const_i32(b, 1), iter_args, n_ia, "kt", false, true);
    rocke_b_region_enter(b, kloop.body);
    {
        rocke_value_t* kt = kloop.iv;
        rocke_value_t* ms[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* ls[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* accs[ROCKE_ATTN_MAX_ATOMS];
        for(int r = 0; r < atom->c_per_lane; ++r)
        {
            ms[r] = kloop.iter_vars[2 * r];
            ls[r] = kloop.iter_vars[2 * r + 1];
        }
        for(int n = 0; n < n_pv_atoms; ++n)
        {
            accs[n] = kloop.iter_vars[2 * atom->c_per_lane + n];
        }

        rocke_value_t* effective_kt
            = (p->k_block_iter_fn != NULL) ? p->k_block_iter_fn(b, kt, p->k_block_iter_user) : kt;

        rocke_value_t* k_tile_base = rocke_b_mul(b, effective_kt, c_block_k);
        rocke_value_t* k_row_for_lane = rocke_b_add(b, k_tile_base, m_in_atom);

        rocke_value_t* keep_tile
            = (p->extra_mask_predicate != NULL)
                  ? p->extra_mask_predicate(b, kt, p->extra_mask_predicate_user)
                  : NULL;
        if(p->extra_skip_predicate != NULL)
        {
            rocke_value_t* skip_mask = p->extra_skip_predicate(b, kt, p->extra_skip_predicate_user);
            keep_tile = (keep_tile != NULL) ? rocke_b_land(b, keep_tile, skip_mask) : skip_mask;
        }

        rocke_value_t* k_addr_row_base;
        if(p->k_row_base_fn != NULL)
        {
            k_addr_row_base = p->k_row_base_fn(b, k_row_for_lane, p->k_row_base_user);
        }
        else
        {
            rocke_value_t* k_arb_t0 = rocke_b_mul(b, k_row_for_lane, p->stride_k_token);
            rocke_value_t* k_arb_t1 = rocke_b_mul(b, p->kv_head_idx, p->stride_k_head);
            rocke_value_t* k_arb_t2 = rocke_b_add(b, k_arb_t0, k_arb_t1);
            k_addr_row_base = rocke_b_add(b, k_arb_t2, k_off);
        }

        /* ---- QK MFMA chain ---- */
        rocke_value_t* score = rocke_b_zero_vec_f32(b, atom->c_per_lane);
        for(int ka = 0; ka < n_qk_atoms; ++ka)
        {
            rocke_value_t* d_ka = rocke_b_const_i32(b, ka);
            rocke_value_t* d_atk = rocke_b_const_i32(b, atom->k);
            rocke_value_t* d_start = rocke_b_add(b, rocke_b_mul(b, d_ka, d_atk), k_lane_start);
            rocke_value_t* k_addr = rocke_b_add(b, k_addr_row_base, d_start);
            rocke_value_t* k_vec;
            if(fp8_kv)
            {
                k_vec = rocke_load_kv_dequant_packed(
                    b, p->K, k_addr, atom->a_per_lane, kv_dtype_eff, kv_dtype_ir, dtype_ir);
            }
            else
            {
                k_vec = rocke_b_global_load_vN(
                    b, p->K, k_addr, dtype_ir, atom->a_per_lane, atom->a_per_lane * 2);
            }
            score = rocke_b_mma(b, qk_op, q_vecs[ka], k_vec, score, NULL, 0);
        }

        /* ---- Scale + mask + softmax row update ---- */
        rocke_value_t* m_blk = rocke_b_div(b, lane, c16);
        rocke_value_t* new_ms[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* new_ls[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* new_accs[ROCKE_ATTN_MAX_ATOMS];
        rocke_value_t* ps_arr[ROCKE_ATTN_MAX_LANE];
        for(int n = 0; n < n_pv_atoms; ++n)
        {
            new_accs[n] = accs[n];
        }
        rocke_value_t* q_pos_for_mask = (p->q_pos_base != NULL) ? p->q_pos_base : p->q_tile_base;
        for(int r = 0; r < atom->c_per_lane; ++r)
        {
            rocke_value_t* s_r_f32 = rocke_b_vec_extract(b, score, r);
            rocke_value_t* s_r_scaled = rocke_b_fmul(b, s_r_f32, p->scale_log2);
            rocke_value_t* rqp_t0 = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 4));
            rocke_value_t* rqp_t1 = rocke_b_add(b, q_pos_for_mask, rqp_t0);
            rocke_value_t* row_q_pos = rocke_b_add(b, rqp_t1, rocke_b_const_i32(b, r));
            rocke_value_t* k_col_pos = rocke_b_add(b, k_tile_base, m_in_atom);
            if(p->extra_score_transform != NULL)
            {
                s_r_scaled
                    = p->extra_score_transform(b, s_r_scaled, kt, r, p->extra_score_transform_user);
            }
            s_r_scaled = rocke_apply_attention_mask(b,
                                                    s_r_scaled,
                                                    p->mask_mode,
                                                    k_col_pos,
                                                    row_q_pos,
                                                    p->sliding_window,
                                                    p->causal_ctx_offset,
                                                    NULL);
            if(keep_tile != NULL)
            {
                s_r_scaled = rocke_b_select(b, keep_tile, s_r_scaled, neg_inf);
            }
            rocke_value_t* row_max = rocke_softmax_row_reduce(b, s_r_scaled, ROCKE_REDUCE_MAX);
            rocke_value_t* m_new_r = rocke_b_fmax(b, ms[r], row_max);
            rocke_value_t* alpha_r = rocke_b_exp2(b, rocke_b_fsub(b, ms[r], m_new_r));
            rocke_value_t* p_r = rocke_b_exp2(b, rocke_b_fsub(b, s_r_scaled, m_new_r));
            rocke_value_t* row_psum = rocke_softmax_row_reduce(b, p_r, ROCKE_REDUCE_SUM);
            rocke_value_t* l_new_r = rocke_b_fadd(b, rocke_b_fmul(b, ls[r], alpha_r), row_psum);

            new_ms[r] = m_new_r;
            new_ls[r] = l_new_r;
            ps_arr[r] = p_r;
            for(int n = 0; n < n_pv_atoms; ++n)
            {
                rocke_value_t* old = rocke_b_vec_extract(b, new_accs[n], r);
                rocke_value_t* rescaled = rocke_b_fmul(b, old, alpha_r);
                new_accs[n] = rocke_b_vec_insert(b, new_accs[n], rescaled, r);
            }
        }

        /* ---- P operand staging via LDS ---- */
        for(int r = 0; r < atom->c_per_lane; ++r)
        {
            rocke_value_t* p_row_t0 = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 4));
            rocke_value_t* p_row = rocke_b_add(b, p_row_t0, rocke_b_const_i32(b, r));
            rocke_value_t* p_col = m_in_atom;
            rocke_value_t* p_f16 = rocke_b_cast_f32_to(b, ps_arr[r], dtype_ir);
            rocke_value_t* idx[2] = {p_row, p_col};
            rocke_b_smem_store_vN(b, P_lds, idx, 2, p_f16, 1);
        }
        rocke_b_sync(b);

        /* ---- PV MFMA chain ---- */
        for(int nba = 0; nba < n_pv_atoms; ++nba)
        {
            rocke_value_t* p_a_vec = rocke_b_zero_vec(b, dtype_ir, atom->a_per_lane);
            for(int j = 0; j < atom->a_per_lane; ++j)
            {
                rocke_value_t* p_col_j = rocke_b_add(b, k_lane_start, rocke_b_const_i32(b, j));
                rocke_value_t* idx[2] = {m_in_atom, p_col_j};
                rocke_value_t* p_v = rocke_b_vec_extract(
                    b, rocke_b_smem_load_vN(b, P_lds, idx, 2, dtype_ir, 1), 0);
                p_a_vec = rocke_b_vec_insert(b, p_a_vec, p_v, j);
            }
            rocke_value_t* v_nba = rocke_b_const_i32(b, nba);
            rocke_value_t* v_atn = rocke_b_const_i32(b, atom->n);
            rocke_value_t* v_col_in_hd = rocke_b_add(b, rocke_b_mul(b, v_nba, v_atn), m_in_atom);
            rocke_value_t* v_a_vec = rocke_b_zero_vec(b, dtype_ir, atom->b_per_lane);
            for(int j = 0; j < atom->b_per_lane; ++j)
            {
                rocke_value_t* v_row_k = rocke_b_add(
                    b, k_tile_base, rocke_b_add(b, k_lane_start, rocke_b_const_i32(b, j)));
                rocke_value_t* v_addr_row_base;
                if(p->v_row_base_fn != NULL)
                {
                    v_addr_row_base = p->v_row_base_fn(b, v_row_k, p->v_row_base_user);
                }
                else
                {
                    rocke_value_t* v_arb_t0 = rocke_b_mul(b, v_row_k, p->stride_v_token);
                    rocke_value_t* v_arb_t1 = rocke_b_mul(b, p->kv_head_idx, p->stride_v_head);
                    rocke_value_t* v_arb_t2 = rocke_b_add(b, v_arb_t0, v_arb_t1);
                    v_addr_row_base = rocke_b_add(b, v_arb_t2, v_off);
                }
                rocke_value_t* v_addr = rocke_b_add(b, v_addr_row_base, v_col_in_hd);
                rocke_value_t* v_scalar;
                if(fp8_kv)
                {
                    rocke_value_t* raw = rocke_b_global_load(b, p->V, v_addr, kv_dtype_ir, 1);
                    rocke_value_t* f32_v = (strcmp(kv_dtype_eff, "fp8e4m3") == 0)
                                               ? rocke_b_cvt_fp8_to_f32(b, raw)
                                               : rocke_b_cvt_bf8_to_f32(b, raw);
                    v_scalar = rocke_b_cast_f32_to(b, f32_v, dtype_ir);
                }
                else
                {
                    v_scalar = rocke_b_global_load(b, p->V, v_addr, dtype_ir, 2);
                }
                v_a_vec = rocke_b_vec_insert(b, v_a_vec, v_scalar, j);
            }
            new_accs[nba] = rocke_b_mma(b, qk_op, p_a_vec, v_a_vec, new_accs[nba], NULL, 0);
        }

        /* ---- Yield updated state ---- */
        rocke_value_t* yields[ROCKE_ATTN_MAX_ITER_ARGS];
        int ny = 0;
        for(int r = 0; r < atom->c_per_lane; ++r)
        {
            yields[ny++] = new_ms[r];
            yields[ny++] = new_ls[r];
        }
        for(int n = 0; n < n_pv_atoms; ++n)
        {
            yields[ny++] = new_accs[n];
        }
        rocke_b_scf_yield(b, yields, ny);
    }
    rocke_b_region_leave(b);

    /* ---- Pull final state ---- */
    rocke_value_t* ls_final[ROCKE_ATTN_MAX_LANE];
    rocke_value_t* accs_final[ROCKE_ATTN_MAX_ATOMS];
    for(int r = 0; r < atom->c_per_lane; ++r)
    {
        ls_final[r] = (kloop.op != NULL) ? kloop.op->results[2 * r + 1] : NULL;
    }
    for(int n = 0; n < n_pv_atoms; ++n)
    {
        accs_final[n] = (kloop.op != NULL) ? kloop.op->results[2 * atom->c_per_lane + n] : NULL;
    }

    /* ---- Epilogue ---- */
    rocke_value_t* m_blk = rocke_b_div(b, lane, c16);
    for(int nba = 0; nba < n_pv_atoms; ++nba)
    {
        for(int r = 0; r < atom->c_per_lane; ++r)
        {
            rocke_value_t* o_row_t0 = rocke_b_mul(b, m_blk, rocke_b_const_i32(b, 4));
            rocke_value_t* o_row_t1 = rocke_b_add(b, p->q_tile_base, o_row_t0);
            rocke_value_t* o_row = rocke_b_add(b, o_row_t1, rocke_b_const_i32(b, r));
            rocke_value_t* o_nba = rocke_b_const_i32(b, nba);
            rocke_value_t* o_atn = rocke_b_const_i32(b, atom->n);
            rocke_value_t* o_col_t0 = rocke_b_mul(b, o_nba, o_atn);
            rocke_value_t* o_col = rocke_b_add(b, o_col_t0, m_in_atom);
            rocke_value_t* inv_l = rocke_safe_inv_l(b, ls_final[r]);
            rocke_value_t* v_f32
                = rocke_b_fmul(b, rocke_b_vec_extract(b, accs_final[nba], r), inv_l);
            if(p->v_scale != NULL)
            {
                v_f32 = rocke_b_fmul(b, v_f32, p->v_scale);
            }
            rocke_value_t* v_out = rocke_b_cast_f32_to(b, v_f32, dtype_ir);
            rocke_value_t* addr_t0 = rocke_b_mul(b, o_row, p->stride_o_token);
            rocke_value_t* addr_t1 = rocke_b_mul(b, p->head_idx, p->stride_o_head);
            rocke_value_t* addr_t2 = rocke_b_add(b, addr_t0, addr_t1);
            rocke_value_t* addr = rocke_b_add(b, addr_t2, o_col);
            rocke_b_global_store(b, p->O, addr, v_out, 2);
        }
    }

    return rocke_ir_builder_status(b);
}

/* ============================== WMMA wave32 body ====================== */

rocke_status_t rocke_wmma_attention_fwd_inner_body(rocke_ir_builder_t* b,
                                                   const rocke_mfma_attn_params_t* p,
                                                   bool v_lds_stage,
                                                   const rocke_arch_target_t* target)
{
    const char* dtype = (p->dtype != NULL) ? p->dtype : "f16";
    const char* arch = (p->arch != NULL) ? p->arch : "gfx950";
    int head_size = p->head_size;

    if(target == NULL)
    {
        target = rocke_arch_target_from_gfx(arch);
    }
    if(target == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "wmma_attention: no target for arch %s", arch);
        return ROCKE_ERR_VALUE;
    }

    /* Per-arch WMMA attention op_id (mirrors Python _wmma_attn_op_id): gfx11
     * (RDNA3/3.5) uses the cross-half-duplicated wmma_f32_16x16x16_* atom; gfx12
     * (RDNA4) uses the split-K wmma_gfx12_f32_16x16x16_* atom. The op_id also
     * selects the f16 vs bf16 intrinsic mangling, so it is keyed on dtype. */
    const char* elem = (strcmp(dtype, "bf16") == 0) ? "bf16" : "f16";
    char op_id[48];
    if(strcmp(arch, "gfx1201") == 0)
    {
        snprintf(op_id, sizeof(op_id), "wmma_gfx12_f32_16x16x16_%s", elem);
    }
    else
    {
        snprintf(op_id, sizeof(op_id), "wmma_f32_16x16x16_%s", elem);
    }

    const rocke_mma_op_t* op = rocke_mma_catalog_by_op_id(&target->mma, op_id);
    if(op == NULL || op->family == NULL || strcmp(op->family, "wmma") != 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "WMMA attention atom %s absent on %s", op_id, arch);
        return ROCKE_ERR_VALUE;
    }
    int wave = op->wave_size;
    const rocke_type_t* dtype_ir = rocke_mfma_attn_ir_type_for_dtype(b, dtype);
    if(dtype_ir == NULL)
    {
        return rocke_ir_builder_status(b);
    }

    const rocke_layout_map_t* a_map = op->a_layout;
    const rocke_layout_map_t* c_map = op->c_layout;
    int a_frag = op->a_frag_len;
    int c_frag = op->c_frag_len;
    int n_dk = head_size / 16;

    /* Python evaluates b.mod(b.thread_id_x(), b.const_i32(wave)) left-to-right:
     * thread_id_x is created before the wave constant. C arg eval order is
     * unspecified (gcc is right-to-left), so hoist the operands into ordered
     * temporaries to match the Python value-creation order exactly. */
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, wave));
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);

    rocke_value_t* a_row = NULL;
    rocke_value_t* dummy = NULL;
    rocke_layout_map_coord(a_map, b, lane, 0, &a_row, &dummy);
    /* gfx12 (RDNA4) split-K: the 16 K-elements of one WMMA step are split across
     * the two lane-halves, so each lane loads a_frag (=8) elements from K base
     * (lane // 16) * a_frag. gfx11 (RDNA3/3.5) duplicates the full K row in every
     * lane (a_frag=16, base 0). k_half_off==NULL keeps the gfx11 emission
     * byte-identical (no half-offset add); mirrors Python's split_k handling. */
    bool split_k = (a_frag * 2 == 16);
    rocke_value_t* k_half_off = NULL;
    if(split_k)
    {
        /* Python: b.mul(b.div(lane, c16), b.const_i32(a_frag)) -- div created
         * before the const (left-to-right). Hoist so the C arg-eval order (gcc is
         * right-to-left) matches the Python value-creation order. */
        rocke_value_t* half = rocke_b_div(b, lane, c16);
        rocke_value_t* c_frag_off = rocke_b_const_i32(b, a_frag);
        k_half_off = rocke_b_mul(b, half, c_frag_off);
    }
    rocke_value_t* col = rocke_b_mod(b, lane, c16);

    rocke_value_t* neg_inf = rocke_b_const_f32(b, -1e30);
    rocke_value_t* zero_f = rocke_b_const_f32(b, 0.0);

    rocke_value_t* k_off = rocke_attn_opt(b, p->k_token_offset_elems);
    rocke_value_t* v_off = rocke_attn_opt(b, p->v_token_offset_elems);

    /* ---- Pre-load Q fragments ---- */
    rocke_value_t* q_row = rocke_b_add(b, p->q_tile_base, a_row);
    /* Python: b.add(b.mul(q_row, stride_q_token), b.mul(head_idx, stride_q_head))
     * -- the token mul is created before the head mul (left-to-right). Hoist to
     * fix the C arg-eval order (gcc is right-to-left). */
    rocke_value_t* q_tok_mul = rocke_b_mul(b, q_row, p->stride_q_token);
    rocke_value_t* q_hd_mul = rocke_b_mul(b, p->head_idx, p->stride_q_head);
    rocke_value_t* q_addr_row_base = rocke_b_add(b, q_tok_mul, q_hd_mul);
    rocke_value_t* q_frags[ROCKE_ATTN_MAX_ATOMS];
    for(int d = 0; d < n_dk; ++d)
    {
        rocke_value_t* q_addr = rocke_b_add(b, q_addr_row_base, rocke_b_const_i32(b, d * 16));
        if(k_half_off != NULL)
        {
            q_addr = rocke_b_add(b, q_addr, k_half_off);
        }
        q_frags[d] = rocke_b_global_load_vN(b, p->Q, q_addr, dtype_ir, a_frag, a_frag * 2);
    }

    /* ---- LDS staging tiles ---- */
    int p_lds_shape[2] = {16, 16};
    rocke_value_t* P_lds = rocke_b_smem_alloc(b, dtype_ir, p_lds_shape, 2, "Pwmma");
    rocke_value_t* V_lds = NULL;
    if(v_lds_stage)
    {
        int v_lds_shape[2] = {16, head_size};
        V_lds = rocke_b_smem_alloc(b, dtype_ir, v_lds_shape, 2, "Vwmma");
    }

    /* ---- Online-softmax + PV accumulator iter-args ---- */
    rocke_iter_arg_t iter_args[ROCKE_ATTN_MAX_ITER_ARGS];
    char name_buf[ROCKE_ATTN_MAX_ITER_ARGS][16];
    int n_ia = 0;
    for(int r = 0; r < c_frag; ++r)
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
    for(int d = 0; d < n_dk; ++d)
    {
        snprintf(name_buf[n_ia], sizeof(name_buf[0]), "acc%d", d);
        iter_args[n_ia].name = name_buf[n_ia];
        iter_args[n_ia].init = rocke_b_zero_vec_f32(b, c_frag);
        ++n_ia;
    }

    rocke_value_t* c_block_k = rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_K);
    rocke_value_t* loop_start
        = (p->k_tile_start != NULL) ? p->k_tile_start : rocke_b_const_i32(b, 0);
    rocke_value_t* loop_stop
        = (p->k_tile_stop != NULL) ? p->k_tile_stop : rocke_b_div(b, p->seqlen_k, c_block_k);

    rocke_for_t kloop = rocke_b_scf_for_iter(
        b, loop_start, loop_stop, rocke_b_const_i32(b, 1), iter_args, n_ia, "kt", false, true);
    rocke_b_region_enter(b, kloop.body);
    {
        rocke_value_t* kt = kloop.iv;
        rocke_value_t* ms[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* ls[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* accs[ROCKE_ATTN_MAX_ATOMS];
        for(int r = 0; r < c_frag; ++r)
        {
            ms[r] = kloop.iter_vars[2 * r];
            ls[r] = kloop.iter_vars[2 * r + 1];
        }
        for(int d = 0; d < n_dk; ++d)
        {
            accs[d] = kloop.iter_vars[2 * c_frag + d];
        }

        rocke_value_t* effective_kt
            = (p->k_block_iter_fn != NULL) ? p->k_block_iter_fn(b, kt, p->k_block_iter_user) : kt;
        rocke_value_t* k_tile_base = rocke_b_mul(b, effective_kt, c_block_k);
        rocke_value_t* k_row_for_lane = rocke_b_add(b, k_tile_base, a_row);

        rocke_value_t* keep_tile
            = (p->extra_mask_predicate != NULL)
                  ? p->extra_mask_predicate(b, kt, p->extra_mask_predicate_user)
                  : NULL;
        if(p->extra_skip_predicate != NULL)
        {
            rocke_value_t* skip_mask = p->extra_skip_predicate(b, kt, p->extra_skip_predicate_user);
            keep_tile = (keep_tile != NULL) ? rocke_b_land(b, keep_tile, skip_mask) : skip_mask;
        }

        rocke_value_t* k_addr_row_base;
        if(p->k_row_base_fn != NULL)
        {
            k_addr_row_base = p->k_row_base_fn(b, k_row_for_lane, p->k_row_base_user);
        }
        else
        {
            /* Python: add(add(mul(k_row_for_lane, stride_k_token),
             *               mul(kv_head_idx, stride_k_head)), k_off)
             * -- token mul created before head mul. Hoist for arg-eval order. */
            rocke_value_t* k_tok_mul = rocke_b_mul(b, k_row_for_lane, p->stride_k_token);
            rocke_value_t* k_hd_mul = rocke_b_mul(b, p->kv_head_idx, p->stride_k_head);
            k_addr_row_base = rocke_b_add(b, rocke_b_add(b, k_tok_mul, k_hd_mul), k_off);
        }

        /* ---- QK^T WMMA chain ---- */
        rocke_value_t* score = rocke_b_zero_vec_f32(b, c_frag);
        for(int d = 0; d < n_dk; ++d)
        {
            rocke_value_t* k_addr = rocke_b_add(b, k_addr_row_base, rocke_b_const_i32(b, d * 16));
            if(k_half_off != NULL)
            {
                k_addr = rocke_b_add(b, k_addr, k_half_off);
            }
            rocke_value_t* k_frag
                = rocke_b_global_load_vN(b, p->K, k_addr, dtype_ir, a_frag, a_frag * 2);
            score = rocke_b_mma(b, op->op_id, q_frags[d], k_frag, score, NULL, 0);
        }

        /* ---- Scale + mask + per-row online softmax ---- */
        rocke_value_t* new_ms[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* new_ls[ROCKE_ATTN_MAX_LANE];
        rocke_value_t* new_accs[ROCKE_ATTN_MAX_ATOMS];
        rocke_value_t* ps_arr[ROCKE_ATTN_MAX_LANE];
        for(int d = 0; d < n_dk; ++d)
        {
            new_accs[d] = accs[d];
        }
        rocke_value_t* q_pos_for_mask = (p->q_pos_base != NULL) ? p->q_pos_base : p->q_tile_base;
        for(int r = 0; r < c_frag; ++r)
        {
            rocke_value_t* row_rel = NULL;
            rocke_value_t* col_k = NULL;
            rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
            rocke_value_t* s_r = rocke_b_fmul(b, rocke_b_vec_extract(b, score, r), p->scale_log2);
            rocke_value_t* row_q_pos = rocke_b_add(b, q_pos_for_mask, row_rel);
            rocke_value_t* k_col_pos = rocke_b_add(b, k_tile_base, col_k);
            if(p->extra_score_transform != NULL)
            {
                s_r = p->extra_score_transform(b, s_r, kt, r, p->extra_score_transform_user);
            }
            s_r = rocke_apply_attention_mask(b,
                                             s_r,
                                             p->mask_mode,
                                             k_col_pos,
                                             row_q_pos,
                                             p->sliding_window,
                                             p->causal_ctx_offset,
                                             NULL);
            if(keep_tile != NULL)
            {
                s_r = rocke_b_select(b, keep_tile, s_r, neg_inf);
            }
            rocke_value_t* row_max = rocke_softmax_row_reduce(b, s_r, ROCKE_REDUCE_MAX);
            rocke_value_t* m_new = rocke_b_fmax(b, ms[r], row_max);
            rocke_value_t* alpha = rocke_b_exp2(b, rocke_b_fsub(b, ms[r], m_new));
            rocke_value_t* p_r = rocke_b_exp2(b, rocke_b_fsub(b, s_r, m_new));
            rocke_value_t* row_sum = rocke_softmax_row_reduce(b, p_r, ROCKE_REDUCE_SUM);
            rocke_value_t* l_new = rocke_b_fadd(b, rocke_b_fmul(b, ls[r], alpha), row_sum);
            new_ms[r] = m_new;
            new_ls[r] = l_new;
            ps_arr[r] = p_r;
            for(int d = 0; d < n_dk; ++d)
            {
                rocke_value_t* old = rocke_b_vec_extract(b, new_accs[d], r);
                new_accs[d] = rocke_b_vec_insert(b, new_accs[d], rocke_b_fmul(b, old, alpha), r);
            }
        }

        /* ---- V staging into LDS ---- */
        if(v_lds_stage)
        {
            rocke_value_t* v_stage_row = rocke_b_add(b, k_tile_base, a_row);
            rocke_value_t* v_stage_base;
            if(p->v_row_base_fn != NULL)
            {
                v_stage_base = p->v_row_base_fn(b, v_stage_row, p->v_row_base_user);
            }
            else
            {
                /* Python: token mul before head mul (left-to-right). */
                rocke_value_t* vs_tok_mul = rocke_b_mul(b, v_stage_row, p->stride_v_token);
                rocke_value_t* vs_hd_mul = rocke_b_mul(b, p->kv_head_idx, p->stride_v_head);
                v_stage_base = rocke_b_add(b, rocke_b_add(b, vs_tok_mul, vs_hd_mul), v_off);
            }
            for(int e = 0; e < head_size / 8; ++e)
            {
                rocke_value_t* v_g = rocke_b_global_load_vN(
                    b,
                    p->V,
                    rocke_b_add(b, v_stage_base, rocke_b_const_i32(b, e * 8)),
                    dtype_ir,
                    8,
                    16);
                rocke_value_t* idx[2] = {a_row, rocke_b_const_i32(b, e * 8)};
                rocke_b_smem_store_vN(b, V_lds, idx, 2, v_g, 8);
            }
        }

        /* ---- P staging through LDS ---- */
        for(int r = 0; r < c_frag; ++r)
        {
            rocke_value_t* row_rel = NULL;
            rocke_value_t* col_k = NULL;
            rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_k);
            rocke_value_t* idx[2] = {row_rel, col_k};
            rocke_b_smem_store_vN(b, P_lds, idx, 2, rocke_b_cast_f32_to(b, ps_arr[r], dtype_ir), 1);
        }
        rocke_b_sync(b);

        /* ---- V load + PV WMMA chain ---- */
        rocke_value_t* p_a = rocke_b_zero_vec(b, dtype_ir, a_frag);
        for(int j = 0; j < a_frag; ++j)
        {
            rocke_value_t* a_k = NULL;
            rocke_value_t* a_dummy = NULL;
            rocke_layout_map_coord(a_map, b, lane, j, &a_dummy, &a_k);
            rocke_value_t* idx[2] = {a_row, a_k};
            rocke_value_t* p_v
                = rocke_b_vec_extract(b, rocke_b_smem_load_vN(b, P_lds, idx, 2, dtype_ir, 1), 0);
            p_a = rocke_b_vec_insert(b, p_a, p_v, j);
        }

        for(int d = 0; d < n_dk; ++d)
        {
            rocke_value_t* d_col = rocke_b_add(b, rocke_b_const_i32(b, d * 16), col);
            rocke_value_t* v_b = rocke_b_zero_vec(b, dtype_ir, a_frag);
            for(int j = 0; j < a_frag; ++j)
            {
                /* B-operand K row this lane's slot j feeds: j on gfx11 (full K
                 * per lane, byte-identical to the historical literal), or
                 * (lane // 16) * a_frag + j on gfx12 (split-K halves). Python:
                 * b.add(k_half_off, b.const_i32(j)) -- k_half_off created before
                 * the const, so it is the first add operand. */
                rocke_value_t* b_k = (k_half_off != NULL)
                                         ? rocke_b_add(b, k_half_off, rocke_b_const_i32(b, j))
                                         : rocke_b_const_i32(b, j);
                rocke_value_t* v_elem;
                if(v_lds_stage)
                {
                    rocke_value_t* idx[2] = {b_k, d_col};
                    v_elem = rocke_b_vec_extract(
                        b, rocke_b_smem_load_vN(b, V_lds, idx, 2, dtype_ir, 1), 0);
                }
                else
                {
                    rocke_value_t* v_row = rocke_b_add(b, k_tile_base, b_k);
                    rocke_value_t* v_row_base;
                    if(p->v_row_base_fn != NULL)
                    {
                        v_row_base = p->v_row_base_fn(b, v_row, p->v_row_base_user);
                    }
                    else
                    {
                        /* Python: token mul before head mul (left-to-right). */
                        rocke_value_t* v_tok_mul = rocke_b_mul(b, v_row, p->stride_v_token);
                        rocke_value_t* v_hd_mul = rocke_b_mul(b, p->kv_head_idx, p->stride_v_head);
                        v_row_base = rocke_b_add(b, rocke_b_add(b, v_tok_mul, v_hd_mul), v_off);
                    }
                    v_elem = rocke_b_global_load(
                        b, p->V, rocke_b_add(b, v_row_base, d_col), dtype_ir, 2);
                }
                v_b = rocke_b_vec_insert(b, v_b, v_elem, j);
            }
            new_accs[d] = rocke_b_mma(b, op->op_id, p_a, v_b, new_accs[d], NULL, 0);
        }

        rocke_value_t* yields[ROCKE_ATTN_MAX_ITER_ARGS];
        int ny = 0;
        for(int r = 0; r < c_frag; ++r)
        {
            yields[ny++] = new_ms[r];
            yields[ny++] = new_ls[r];
        }
        for(int d = 0; d < n_dk; ++d)
        {
            yields[ny++] = new_accs[d];
        }
        rocke_b_scf_yield(b, yields, ny);
    }
    rocke_b_region_leave(b);

    rocke_value_t* ls_final[ROCKE_ATTN_MAX_LANE];
    rocke_value_t* accs_final[ROCKE_ATTN_MAX_ATOMS];
    for(int r = 0; r < c_frag; ++r)
    {
        ls_final[r] = (kloop.op != NULL) ? kloop.op->results[2 * r + 1] : NULL;
    }
    for(int d = 0; d < n_dk; ++d)
    {
        accs_final[d] = (kloop.op != NULL) ? kloop.op->results[2 * c_frag + d] : NULL;
    }

    /* ---- Epilogue ---- */
    for(int d = 0; d < n_dk; ++d)
    {
        for(int r = 0; r < c_frag; ++r)
        {
            rocke_value_t* row_rel = NULL;
            rocke_value_t* col_n = NULL;
            rocke_layout_map_coord(c_map, b, lane, r, &row_rel, &col_n);
            rocke_value_t* l_safe = ls_final[r];
            rocke_value_t* zero_mask = rocke_b_fcmp(b, "oeq", l_safe, zero_f);
            rocke_value_t* inv_l = rocke_b_select(b, zero_mask, zero_f, rocke_b_rcp(b, l_safe));
            rocke_value_t* v_f32 = rocke_b_fmul(b, rocke_b_vec_extract(b, accs_final[d], r), inv_l);
            if(p->v_scale != NULL)
            {
                v_f32 = rocke_b_fmul(b, v_f32, p->v_scale);
            }
            rocke_value_t* o_row = rocke_b_add(b, p->q_tile_base, row_rel);
            rocke_value_t* o_col = rocke_b_add(b, rocke_b_const_i32(b, d * 16), col_n);
            /* Python: token mul before head mul (left-to-right). */
            rocke_value_t* o_tok_mul = rocke_b_mul(b, o_row, p->stride_o_token);
            rocke_value_t* o_hd_mul = rocke_b_mul(b, p->head_idx, p->stride_o_head);
            rocke_value_t* o_addr = rocke_b_add(b, rocke_b_add(b, o_tok_mul, o_hd_mul), o_col);
            rocke_b_global_store(b, p->O, o_addr, rocke_b_cast_f32_to(b, v_f32, dtype_ir), 2);
        }
    }

    return rocke_ir_builder_status(b);
}

/* ===========================================================================
 * Additional symbols ported from rocke.helpers.attention.
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
rocke_value_t* rocke_mfma_attn_mfma_32x32x16_for_dtype(rocke_ir_builder_t* b,
                                                       const rocke_type_t* dtype,
                                                       rocke_value_t* a,
                                                       rocke_value_t* bv,
                                                       rocke_value_t* c)
{
    if(dtype == NULL || dtype->name == NULL)
    {
        if(b != NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported MFMA 32x32x16 dtype (null)");
        }
        return NULL;
    }
    if(strcmp(dtype->name, "f16") == 0)
    {
        return rocke_b_mfma_f32_32x32x16_f16(b, a, bv, c);
    }
    if(strcmp(dtype->name, "bf16") == 0)
    {
        return rocke_b_mfma_f32_32x32x16_bf16(b, a, bv, c);
    }
    if(strcmp(dtype->name, "fp8e4m3") == 0)
    {
        return rocke_b_mfma_f32_32x32x16_fp8(b, a, bv, c);
    }
    if(b != NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unsupported MFMA 32x32x16 dtype %s", dtype->name);
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
rocke_value_t* rocke_mfma_attn_dequant_fp8x8_to_dtype(rocke_ir_builder_t* b,
                                                      rocke_value_t* fp8_vec,
                                                      rocke_value_t* scale,
                                                      const rocke_type_t* dtype)
{
    const rocke_type_t* fp8e4m3 = rocke_fp8e4m3();
    rocke_value_t* lo_comp[4];
    rocke_value_t* hi_comp[4];
    rocke_value_t* lo_fp8;
    rocke_value_t* hi_fp8;
    rocke_value_t* lo_f32;
    rocke_value_t* hi_f32;
    rocke_value_t* deq[8];
    int i;

    if(dtype == NULL)
    {
        if(b != NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "dequant_fp8x8_to_dtype: dtype is NULL");
        }
        return NULL;
    }

    /* lo_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4)], FP8E4M3) */
    for(i = 0; i < 4; ++i)
    {
        lo_comp[i] = rocke_b_vec_extract(b, fp8_vec, i);
    }
    lo_fp8 = rocke_b_vec_pack(b, lo_comp, 4, fp8e4m3);

    /* hi_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4, 8)], FP8E4M3) */
    for(i = 0; i < 4; ++i)
    {
        hi_comp[i] = rocke_b_vec_extract(b, fp8_vec, 4 + i);
    }
    hi_fp8 = rocke_b_vec_pack(b, hi_comp, 4, fp8e4m3);

    /* lo_f32 = b.cvt_pk_f32_fp8x4(lo_fp8); hi_f32 = b.cvt_pk_f32_fp8x4(hi_fp8) */
    lo_f32 = rocke_b_cvt_pk_f32_fp8x4(b, lo_fp8);
    hi_f32 = rocke_b_cvt_pk_f32_fp8x4(b, hi_fp8);

    /* deq lo lanes 0..3, then hi lanes 0..3; each: vec_extract -> fmul -> cast */
    for(i = 0; i < 4; ++i)
    {
        deq[i] = rocke_b_cast_f32_to(
            b, rocke_b_fmul(b, rocke_b_vec_extract(b, lo_f32, i), scale), dtype);
    }
    for(i = 0; i < 4; ++i)
    {
        deq[4 + i] = rocke_b_cast_f32_to(
            b, rocke_b_fmul(b, rocke_b_vec_extract(b, hi_f32, i), scale), dtype);
    }

    /* return b.vec_pack(deq, dtype) */
    return rocke_b_vec_pack(b, deq, 8, dtype);
}
