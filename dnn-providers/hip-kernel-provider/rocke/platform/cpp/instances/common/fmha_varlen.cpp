// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_fmha_varlen.c -- C99 port of rocke/instances/common/fmha_varlen.py.
 *
 * Variable-length FMHA forward (CK Tile 01_fmha varlen parity). Packs B
 * sequences of arbitrary lengths into one flat (total_q, H, D) tensor and uses
 * cumulative-sequence-length arrays (cu_seqlens_q, cu_seqlens_k) to address
 * into them.
 *
 * The build entry reproduces the Python build_fmha_fwd_varlen builder-call
 * sequence op-for-op (see rocke_build_fmha_fwd_varlen below), so the emitted IR
 * is byte-identical to the Python helper's emission.
 */
#include "rocke/instance_fmha_varlen.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h" /* rocke_arena_t              */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* mfma_attention_fwd body  */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join     */
#include "rocke/helper_rocke.instances.common.fmha_arch.h" /* validate_fmha_mfma_atom */

/* MFMA_ATTN_BLOCK_M / MFMA_ATTN_BLOCK_K come from helper_..mfma_attention.h:
 *   ROCKE_MFMA_ATTN_BLOCK_M == 16, ROCKE_MFMA_ATTN_BLOCK_K == 16. */

/* ------------------------------------------------------------------ *
 *  small helpers
 * ------------------------------------------------------------------ */

static void copy_reason(char* out, size_t out_cap, const char* msg)
{
    size_t n;
    if(out == NULL || out_cap == 0)
    {
        return;
    }
    if(msg == NULL)
    {
        msg = "";
    }
    n = strlen(msg);
    if(n >= out_cap)
    {
        n = out_cap - 1;
    }
    memcpy(out, msg, n);
    out[n] = '\0';
}

/* FmhaMaskMode -> attention helper mask-mode enum. The "none"/"causal"/
 * "sliding_window" spellings are the only ones the MFMA body understands;
 * alibi / custom never reach here (validate_common_spec / the body itself
 * reject them). The first three enum values align 1:1. */
static rocke_attn_mask_mode_t fmha_to_attn_mask(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_CAUSAL:
        return ROCKE_ATTN_MASK_CAUSAL;
    case ROCKE_FMHA_MASK_SLIDING_WINDOW:
        return ROCKE_ATTN_MASK_SLIDING_WINDOW;
    case ROCKE_FMHA_MASK_NONE:
    default:
        return ROCKE_ATTN_MASK_NONE;
    }
}

/* ===================================================================== *
 *  FmhaFwdVarlenSpec
 * ===================================================================== */

rocke_fmha_fwd_varlen_spec_t rocke_fmha_fwd_varlen_spec_default(rocke_fmha_common_spec_t common,
                                                                int max_seqlen_q,
                                                                int max_seqlen_k,
                                                                int batch)
{
    rocke_fmha_fwd_varlen_spec_t s;
    s.common = common;
    s.max_seqlen_q = max_seqlen_q;
    s.max_seqlen_k = max_seqlen_k;
    s.batch = batch;
    s.name = "rocke_fmha_fwd_varlen"; /* dataclass default */
    return s;
}

/* FmhaFwdVarlenSpec.kernel_name():
 *   kernel_name_join(name, "H{head_size}", "HQ{nq}", "HK{nkv}", dtype,
 *                    "Q{q}", "K{k}", "B{b}", mask_mode)            */
rocke_status_t rocke_fmha_fwd_varlen_kernel_name(const rocke_fmha_fwd_varlen_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap)
{
    char h[32], hq[32], hk[32], q[32], k[32], bb[32];
    const char* name;
    const char* dtype;
    const char* mask;
    const char* parts[8];
    const rocke_fmha_shape_t* sh;

    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    sh = &spec->common.shape;
    name = (spec->name != NULL) ? spec->name : "rocke_fmha_fwd_varlen";
    dtype = (spec->common.dtype != NULL) ? spec->common.dtype : "f16";
    mask = rocke_fmha_mask_mode_name(spec->common.mask_mode);
    if(mask == NULL)
    {
        mask = "none";
    }

    /* f"H{...}" etc. -- snprintf reproduces the Python f-string formatting. */
    snprintf(h, sizeof(h), "H%d", sh->head_size);
    snprintf(hq, sizeof(hq), "HQ%d", sh->num_query_heads);
    snprintf(hk, sizeof(hk), "HK%d", sh->num_kv_heads);
    snprintf(q, sizeof(q), "Q%d", spec->max_seqlen_q);
    snprintf(k, sizeof(k), "K%d", spec->max_seqlen_k);
    snprintf(bb, sizeof(bb), "B%d", spec->batch);

    parts[0] = h;
    parts[1] = hq;
    parts[2] = hk;
    parts[3] = dtype;
    parts[4] = q;
    parts[5] = k;
    parts[6] = bb;
    parts[7] = mask;

    return rocke_kernel_name_join(name, parts, 8, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

bool rocke_fmha_fwd_varlen_is_valid_spec(const rocke_fmha_fwd_varlen_spec_t* spec,
                                         const char* arch,
                                         char* reason,
                                         size_t reason_cap)
{
    rocke_arena_t arena;
    bool ok;
    const char* why = NULL;
    char buf[256];

    if(spec == NULL)
    {
        copy_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* ok, why = validate_common_spec(spec.common) */
    if(rocke_arena_init(&arena, 4096) != 0)
    {
        copy_reason(reason, reason_cap, "arena init failed");
        return false;
    }
    ok = rocke_fmha_validate_common_spec(&arena, &spec->common, &why);
    if(!ok)
    {
        copy_reason(reason, reason_cap, why);
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch) */
    if(!rocke_validate_fmha_mfma_atom(spec->common.dtype, arch, buf, sizeof(buf)))
    {
        copy_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.batch <= 0: ... */
    if(spec->batch <= 0)
    {
        snprintf(buf, sizeof(buf), "batch must be > 0 (got %d)", spec->batch);
        copy_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.max_seqlen_q <= 0 or spec.max_seqlen_k <= 0: ... */
    if(spec->max_seqlen_q <= 0 || spec->max_seqlen_k <= 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "max_seqlen_q / max_seqlen_k must be > 0 (got %d, %d)",
                 spec->max_seqlen_q,
                 spec->max_seqlen_k);
        copy_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.max_seqlen_q % MFMA_ATTN_BLOCK_M != 0: ... */
    if(spec->max_seqlen_q % ROCKE_MFMA_ATTN_BLOCK_M != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA varlen needs max_seqlen_q (%d) to be a multiple of BLOCK_M (%d)",
                 spec->max_seqlen_q,
                 ROCKE_MFMA_ATTN_BLOCK_M);
        copy_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.max_seqlen_k % MFMA_ATTN_BLOCK_K != 0: ... */
    if(spec->max_seqlen_k % ROCKE_MFMA_ATTN_BLOCK_K != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA varlen needs max_seqlen_k (%d) to be a multiple of BLOCK_K (%d)",
                 spec->max_seqlen_k,
                 ROCKE_MFMA_ATTN_BLOCK_K);
        copy_reason(reason, reason_cap, buf);
        return false;
    }

    /* if spec.common.shape.head_size % 16 != 0: ... */
    if(spec->common.shape.head_size % 16 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "MFMA varlen needs head_size %% 16 == 0 (got %d)",
                 spec->common.shape.head_size);
        copy_reason(reason, reason_cap, buf);
        return false;
    }

    copy_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  _declare_params -- the varlen FMHA-fwd kernel ABI (shared build + sig).
 * ===================================================================== */
static void declare_params(rocke_fmha_kernel_builder_t* kb)
{
    static const char* const stride_names[4] = {"q", "k", "v", "o"};

    /* kb.add_tensor("Q", readonly=True)                          */
    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, true, false, 16);
    /* kb.add_tensor("K", readonly=True)                          */
    rocke_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    /* kb.add_tensor("V", readonly=True)                          */
    rocke_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    /* kb.add_tensor("O", readonly=False, writeonly=True)         */
    rocke_fmha_kernel_builder_add_tensor(kb, "O", NULL, false, true, 16);
    /* kb.add_ptr("cu_seqlens_q", dtype="i32")                    */
    rocke_fmha_kernel_builder_add_ptr(kb, "cu_seqlens_q", "i32", true, 4);
    /* kb.add_ptr("cu_seqlens_k", dtype="i32")                    */
    rocke_fmha_kernel_builder_add_ptr(kb, "cu_seqlens_k", "i32", true, 4);
    /* kb.add_scalar("scale_log2", "f32")                         */
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    /* kb.add_scalar("total_q", "i32")                            */
    rocke_fmha_kernel_builder_add_scalar(kb, "total_q", "i32");
    /* kb.add_scalar("batch", "i32")                              */
    rocke_fmha_kernel_builder_add_scalar(kb, "batch", "i32");
    /* kb.add_strides("q", "k", "v", "o")                         */
    rocke_fmha_kernel_builder_add_strides(kb, stride_names, 4);
}

/* ===================================================================== *
 *  rocke_build_fmha_fwd_varlen -- the varlen FMHA forward kernel.
 *
 *  Mirrors build_fmha_fwd_varlen() op-for-op.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_fmha_fwd_varlen(rocke_fmha_kernel_builder_t* out_kb,
                                                const rocke_fmha_fwd_varlen_spec_t* spec,
                                                const char* arch,
                                                char* err,
                                                size_t err_cap)
{
    return ckc::guard_builder(
        rocke_fmha_kernel_builder_builder(out_kb), [&]() -> rocke_kernel_def_t* {
            char name[256];
            char reason[256];
            const rocke_fmha_common_spec_t* s;
            rocke_fmha_kernel_builder_t* kb;
            rocke_ir_builder_t* b;
            rocke_value_t* cu_seqlens_q;
            rocke_value_t* cu_seqlens_k;
            rocke_value_t* q_tile_idx;
            rocke_value_t* head_idx;
            rocke_value_t* kv_head_idx;
            rocke_value_t* scale_log2;
            rocke_value_t* q_tile_base;
            rocke_value_t* seq;
            rocke_value_t* seq_idx;
            rocke_value_t* cuq_base;
            rocke_value_t* local_q_tile;
            rocke_value_t* cuk_base;
            rocke_value_t* cuk_next;
            rocke_value_t* seqlen_k;
            rocke_value_t* k_token_offset;
            rocke_value_t* v_token_offset;
            rocke_value_t* causal_ctx;
            rocke_mfma_attn_params_t p;
            int i;

            if(out_kb == NULL || spec == NULL)
            {
                copy_reason(err, err_cap, "rocke_build_fmha_fwd_varlen: null spec/builder");
                return NULL;
            }
            if(arch == NULL)
            {
                arch = "gfx950";
            }

            /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
            if(!rocke_fmha_fwd_varlen_is_valid_spec(spec, arch, reason, sizeof(reason)))
            {
                char msg[320];
                snprintf(msg, sizeof(msg), "invalid fmha_fwd_varlen spec: %s", reason);
                copy_reason(err, err_cap, msg);
                return NULL;
            }

            s = &spec->common;

            /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
            if(rocke_fmha_fwd_varlen_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            {
                copy_reason(err, err_cap, "rocke_build_fmha_fwd_varlen: kernel_name too long");
                return NULL;
            }
            if(rocke_fmha_kernel_builder_init(out_kb, name, s) != ROCKE_OK)
            {
                copy_reason(
                    err, err_cap, "rocke_build_fmha_fwd_varlen: kernel builder init failed");
                return NULL;
            }
            kb = out_kb;

            /* kb.block_size(64) -- MFMA: one wave64 warp per CTA. */
            rocke_fmha_kernel_builder_block_size(kb, 64);
            /* _declare_params(kb) */
            declare_params(kb);
            /* kb.decode_grid() */
            rocke_fmha_kernel_builder_decode_grid(kb, -1, false, NULL, NULL, NULL);

            /* b = kb.builder */
            b = rocke_fmha_kernel_builder_builder(kb);

            /* cu_seqlens_q = kb.ptr("cu_seqlens_q"); cu_seqlens_k = kb.ptr("cu_seqlens_k") */
            cu_seqlens_q = rocke_fmha_kernel_builder_ptr(kb, "cu_seqlens_q");
            cu_seqlens_k = rocke_fmha_kernel_builder_ptr(kb, "cu_seqlens_k");

            /* q_tile_idx = kb.q_token */
            q_tile_idx = kb->q_token;
            /* head_idx = b.to_sgpr_u32(kb.head_idx) */
            head_idx = rocke_b_to_sgpr_u32(b, kb->head_idx);
            /* kv_head_idx = b.to_sgpr_u32(kb.kv_head_idx) */
            kv_head_idx = rocke_b_to_sgpr_u32(b, kb->kv_head_idx);
            /* scale_log2 = kb.scalar("scale_log2") */
            scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");

            /* q_tile_base = b.to_sgpr_u32(b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))) */
            q_tile_base = rocke_b_to_sgpr_u32(
                b, rocke_b_mul(b, q_tile_idx, rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_M)));

            /* Find which sequence this tile belongs to via Python-unrolled scan of
         * cu_seqlens_q (spec.batch compile-time constant => batch compares + cmovs).
         *
         *   seq = b.const_i32(0)
         *   for i in range(spec.batch):
         *       cuq_next = b.global_load_i32(cu_seqlens_q, b.const_i32(i + 1))
         *       is_in_seq = b.cmp_lt(q_tile_base, cuq_next)
         *       seq = b.select(is_in_seq, seq, b.add(seq, b.const_i32(1)))
         *   seq_idx = b.to_sgpr_u32(seq)                                          */
            seq = rocke_b_const_i32(b, 0);
            for(i = 0; i < spec->batch; ++i)
            {
                rocke_value_t* cuq_next
                    = rocke_b_global_load_i32(b, cu_seqlens_q, rocke_b_const_i32(b, i + 1), 4);
                rocke_value_t* is_in_seq = rocke_b_cmp_lt(b, q_tile_base, cuq_next);
                seq = rocke_b_select(
                    b, is_in_seq, seq, rocke_b_add(b, seq, rocke_b_const_i32(b, 1)));
            }
            seq_idx = rocke_b_to_sgpr_u32(b, seq);

            /* cuq_base = b.to_sgpr_u32(b.global_load_i32(cu_seqlens_q, seq_idx)) */
            cuq_base = rocke_b_to_sgpr_u32(b, rocke_b_global_load_i32(b, cu_seqlens_q, seq_idx, 4));
            /* local_q_tile = b.to_sgpr_u32(b.sub(q_tile_base, cuq_base)) */
            local_q_tile = rocke_b_to_sgpr_u32(b, rocke_b_sub(b, q_tile_base, cuq_base));
            /* cuk_base = b.to_sgpr_u32(b.global_load_i32(cu_seqlens_k, seq_idx)) */
            cuk_base = rocke_b_to_sgpr_u32(b, rocke_b_global_load_i32(b, cu_seqlens_k, seq_idx, 4));
            /* cuk_next = b.global_load_i32(cu_seqlens_k, b.add(seq_idx, b.const_i32(1))) */
            cuk_next = rocke_b_global_load_i32(
                b, cu_seqlens_k, rocke_b_add(b, seq_idx, rocke_b_const_i32(b, 1)), 4);
            /* seqlen_k = b.to_sgpr_u32(b.sub(cuk_next, cuk_base)) */
            seqlen_k = rocke_b_to_sgpr_u32(b, rocke_b_sub(b, cuk_next, cuk_base));

            /* k_token_offset = b.to_sgpr_u32(b.mul(cuk_base, kb.stride_token("k"))) */
            k_token_offset = rocke_b_to_sgpr_u32(
                b, rocke_b_mul(b, cuk_base, rocke_fmha_kernel_builder_stride_token(kb, "k")));
            /* v_token_offset = b.to_sgpr_u32(b.mul(cuk_base, kb.stride_token("v"))) */
            v_token_offset = rocke_b_to_sgpr_u32(
                b, rocke_b_mul(b, cuk_base, rocke_fmha_kernel_builder_stride_token(kb, "v")));

            /* causal_ctx = b.const_i32(0) if mask in ("causal","sliding_window") else None */
            causal_ctx = (s->mask_mode == ROCKE_FMHA_MASK_CAUSAL
                          || s->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW)
                             ? rocke_b_const_i32(b, 0)
                             : NULL;

            /* mfma_attention_fwd_inner_body(b, Q=..., K=..., V=..., O=..., ...) */
            memset(&p, 0, sizeof(p));
            p.Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
            p.K = rocke_fmha_kernel_builder_tensor(kb, "K");
            p.V = rocke_fmha_kernel_builder_tensor(kb, "V");
            p.O = rocke_fmha_kernel_builder_tensor(kb, "O");
            p.head_size = s->shape.head_size;
            p.seqlen_k = seqlen_k;
            p.q_tile_base = q_tile_base;
            /* q_pos_base = local_q_tile (within-sequence position for causal mask) */
            p.q_pos_base = local_q_tile;
            p.head_idx = head_idx;
            p.kv_head_idx = kv_head_idx;
            p.stride_q_token = rocke_fmha_kernel_builder_stride_token(kb, "q");
            p.stride_q_head = rocke_fmha_kernel_builder_stride_head(kb, "q");
            p.stride_k_token = rocke_fmha_kernel_builder_stride_token(kb, "k");
            p.stride_k_head = rocke_fmha_kernel_builder_stride_head(kb, "k");
            p.stride_v_token = rocke_fmha_kernel_builder_stride_token(kb, "v");
            p.stride_v_head = rocke_fmha_kernel_builder_stride_head(kb, "v");
            p.stride_o_token = rocke_fmha_kernel_builder_stride_token(kb, "o");
            p.stride_o_head = rocke_fmha_kernel_builder_stride_head(kb, "o");
            p.scale_log2 = scale_log2;
            p.dtype = s->dtype;
            p.mask_mode = fmha_to_attn_mask(s->mask_mode);
            p.sliding_window = s->sliding_window;
            p.causal_ctx_offset = causal_ctx;
            p.k_token_offset_elems = k_token_offset;
            p.v_token_offset_elems = v_token_offset;
            /* remaining fields (callbacks, kv_dtype, v_scale, flags, k_tile_start/stop,
         * codebook_ptr, wmma_v_lds_stage) are the Python defaults (None / False),
         * which memset(0) already supplies. */
            p.arch = arch;

            rocke_mfma_attention_fwd_inner_body(b, &p);

            /* b.ret() */
            rocke_b_ret(b);

            if(rocke_ir_builder_status(b) != ROCKE_OK)
            {
                copy_reason(err, err_cap, rocke_ir_builder_error(b));
                return NULL;
            }

            /* return kb.kernel */
            return rocke_fmha_kernel_builder_kernel(kb);
        });
}

/* ===================================================================== *
 *  fmha_fwd_varlen_grid
 * ===================================================================== */
void rocke_fmha_fwd_varlen_grid(const rocke_fmha_fwd_varlen_spec_t* spec,
                                int total_q,
                                int out_grid[3])
{
    if(out_grid == NULL)
    {
        return;
    }
    if(spec == NULL)
    {
        out_grid[0] = out_grid[1] = out_grid[2] = 0;
        return;
    }
    /* (total_q // BLOCK_M, num_query_heads, 1) */
    out_grid[0] = total_q / ROCKE_MFMA_ATTN_BLOCK_M;
    out_grid[1] = spec->common.shape.num_query_heads;
    out_grid[2] = 1;
}

/* ===================================================================== *
 *  rocke_fmha_fwd_varlen_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own FmhaKernelBuilder.
 * ===================================================================== */
rocke_status_t rocke_fmha_fwd_varlen_lower_to_llvm(const rocke_fmha_fwd_varlen_spec_t* spec,
                                                   const char* arch,
                                                   rocke_llvm_flavor_t flavor,
                                                   char** out_ll,
                                                   char* err,
                                                   size_t err_cap)
{
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    memset(&kb, 0, sizeof(kb));
    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        copy_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_fmha_fwd_varlen(&kb, spec, arch, err, err_cap);
    if(kernel == NULL)
    {
        /* err already populated by the build path. The kernel builder is left
         * in whatever state init reached; free it defensively. */
        rocke_fmha_kernel_builder_free(&kb);
        return ROCKE_ERR_VALUE;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}
