// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/instances/common/fmha_paged_prefill.py.
 *
 * Builder-call sequence is byte-faithful to the Python build. See the header for
 * the symbol map. Below, the comment blocks quote the Python lines each chunk
 * reproduces so the op order can be audited against the source.
 */
#include "rocke/instance_fmha_paged_prefill.h"

#include <math.h>
#include <stdio.h> /* snprintf */
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.attention.h" /* rocke_attn_mask_mode_t */
#include "rocke/helper_rocke.helpers.mfma_attention.h" /* MFMA body + consts   */
#include "rocke/helper_rocke.instances.common._fmha_warp_body.h" /* WARP_SIZE, warp body */
#include "rocke/helper_rocke.instances.common.fmha_arch.h" /* validate_fmha_mfma_atom */
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ===================================================================== *
 *  small helpers
 * ===================================================================== */

#define ROCKE_FPP_DEFAULT_NAME "rocke_fmha_fwd_paged_prefill"

static void rocke_fpp_copy_str(char* out, size_t out_cap, const char* m)
{
    size_t n;
    if(out == NULL || out_cap == 0)
    {
        return;
    }
    if(m == NULL)
    {
        m = "";
    }
    n = strlen(m);
    if(n >= out_cap)
    {
        n = out_cap - 1;
    }
    memcpy(out, m, n);
    out[n] = '\0';
}

/* The contiguous (non-NUL-suffixed) tensor name; mask_mode!r spelling reuse. */
static const char* rocke_fpp_mask_name(rocke_fmha_mask_mode_t m)
{
    const char* s = rocke_fmha_mask_mode_name(m);
    return (s != NULL) ? s : "none";
}

/* The FmhaCommonSpec mask_mode (rocke_fmha_mask_mode_t) -> the attention helper's
 * rocke_attn_mask_mode_t (only none/causal/sliding_window reach the MFMA body). */
static rocke_attn_mask_mode_t rocke_fpp_attn_mask(rocke_fmha_mask_mode_t m)
{
    switch(m)
    {
    case ROCKE_FMHA_MASK_CAUSAL:
        return ROCKE_ATTN_MASK_CAUSAL;
    case ROCKE_FMHA_MASK_SLIDING_WINDOW:
        return ROCKE_ATTN_MASK_SLIDING_WINDOW;
    default:
        return ROCKE_ATTN_MASK_NONE;
    }
}

/* ===================================================================== *
 *  spec constructors / kernel_name
 * ===================================================================== */

rocke_fmha_fwd_paged_prefill_spec_t rocke_fmha_fwd_paged_prefill_spec_default(
    rocke_fmha_common_spec_t common, int page_block_size, int max_blocks_per_seq, int batch)
{
    rocke_fmha_fwd_paged_prefill_spec_t s;
    s.common = common;
    s.page_block_size = page_block_size;
    s.max_blocks_per_seq = max_blocks_per_seq;
    s.batch = batch;
    s.name = ROCKE_FPP_DEFAULT_NAME;
    s.use_mfma_body = false;
    return s;
}

/* FmhaFwdPagedPrefillSpec.kernel_name():
 *   s = self.common.shape
 *   return kernel_name_join(
 *       self.name, f"H{s.head_size}", f"HQ{s.num_query_heads}",
 *       f"HK{s.num_kv_heads}", self.common.dtype,
 *       f"PG{self.page_block_size}", f"B{self.batch}", self.common.mask_mode)
 */
rocke_status_t rocke_fmha_fwd_paged_prefill_kernel_name(
    const rocke_fmha_fwd_paged_prefill_spec_t* spec, char* out, size_t out_cap)
{
    char pH[32], pHQ[32], pHK[32], pPG[32], pB[32];
    const char* parts[7];
    const char* prefix;
    const rocke_fmha_shape_t* sh;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    sh = &spec->common.shape;
    prefix = (spec->name != NULL) ? spec->name : ROCKE_FPP_DEFAULT_NAME;

    (void)snprintf(pH, sizeof pH, "H%d", sh->head_size);
    (void)snprintf(pHQ, sizeof pHQ, "HQ%d", sh->num_query_heads);
    (void)snprintf(pHK, sizeof pHK, "HK%d", sh->num_kv_heads);
    (void)snprintf(pPG, sizeof pPG, "PG%d", spec->page_block_size);
    (void)snprintf(pB, sizeof pB, "B%d", spec->batch);

    parts[0] = pH;
    parts[1] = pHQ;
    parts[2] = pHK;
    parts[3] = (spec->common.dtype != NULL) ? spec->common.dtype : "f16";
    parts[4] = pPG;
    parts[5] = pB;
    parts[6] = rocke_fpp_mask_name(spec->common.mask_mode);

    return rocke_kernel_name_join(prefix, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== *
 *
 * def is_valid_spec(spec, arch="gfx950") -> (bool, str):
 *     ok, why = validate_common_spec(spec.common)
 *     if not ok: return False, why
 *     if spec.use_mfma_body:
 *         ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
 *         if not ok: return False, why
 *     if spec.batch <= 0: return False, "batch must be > 0 (got ...)"
 *     if spec.page_block_size <= 0 or not power-of-two:
 *         return False, "page_block_size must be a positive power of two (got ...)"
 *     if spec.max_blocks_per_seq <= 0:
 *         return False, "max_blocks_per_seq must be > 0 (got ...)"
 *     return True, "ok"
 */
bool rocke_fmha_fwd_paged_prefill_is_valid_spec(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                                const char* arch,
                                                char* reason,
                                                size_t reason_cap)
{
    rocke_arena_t arena;
    const char* why = NULL;
    bool ok;
    int pg;

    if(spec == NULL)
    {
        rocke_fpp_copy_str(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* validate_common_spec(spec.common) */
    if(rocke_arena_init(&arena, 0) != 0)
    {
        rocke_fpp_copy_str(reason, reason_cap, "arena init failed");
        return false;
    }
    ok = rocke_fmha_validate_common_spec(&arena, &spec->common, &why);
    if(!ok)
    {
        rocke_fpp_copy_str(reason, reason_cap, why);
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* if spec.use_mfma_body: validate_fmha_mfma_atom(dtype, arch) */
    if(spec->use_mfma_body)
    {
        char atom_reason[256];
        if(!rocke_validate_fmha_mfma_atom(
               spec->common.dtype, arch, atom_reason, sizeof atom_reason))
        {
            rocke_fpp_copy_str(reason, reason_cap, atom_reason);
            return false;
        }
    }

    /* batch > 0 */
    if(spec->batch <= 0)
    {
        char buf[96];
        (void)snprintf(buf, sizeof buf, "batch must be > 0 (got %d)", spec->batch);
        rocke_fpp_copy_str(reason, reason_cap, buf);
        return false;
    }

    /* page_block_size positive power of two */
    pg = spec->page_block_size;
    if(pg <= 0 || (pg & (pg - 1)) != 0)
    {
        char buf[128];
        (void)snprintf(
            buf, sizeof buf, "page_block_size must be a positive power of two (got %d)", pg);
        rocke_fpp_copy_str(reason, reason_cap, buf);
        return false;
    }

    /* max_blocks_per_seq > 0 */
    if(spec->max_blocks_per_seq <= 0)
    {
        char buf[96];
        (void)snprintf(
            buf, sizeof buf, "max_blocks_per_seq must be > 0 (got %d)", spec->max_blocks_per_seq);
        rocke_fpp_copy_str(reason, reason_cap, buf);
        return false;
    }

    rocke_fpp_copy_str(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  _declare_params
 * ===================================================================== *
 *
 * def _declare_params(kb):
 *     kb.add_tensor("Q", readonly=True)
 *     kb.add_tensor("K_cache", readonly=True)
 *     kb.add_tensor("V_cache", readonly=True)
 *     kb.add_tensor("O", readonly=False, writeonly=True)
 *     kb.add_ptr("block_table", dtype="i32", readonly=True)
 *     kb.add_ptr("cu_seqlens_q", dtype="i32", readonly=True)
 *     kb.add_ptr("seqlens_k", dtype="i32", readonly=True)
 *     kb.add_scalar("scale_log2", "f32")
 *     kb.add_scalar("total_q", "i32")
 *     kb.add_scalar("batch", "i32")
 *     kb.add_strides("q")
 *     kb.add_scalar("stride_block", "i32")
 *     kb.add_scalar("stride_page", "i32")
 *     kb.add_scalar("stride_kv_head", "i32")
 *     kb.add_scalar("stride_v_block", "i32")
 *     kb.add_scalar("stride_v_page", "i32")
 *     kb.add_scalar("stride_v_kv_head", "i32")
 *     kb.add_strides("o")
 *     kb.add_scalar("block_table_stride", "i32")
 */
static void rocke_fpp_declare_params(rocke_fmha_kernel_builder_t* kb)
{
    static const char* const q_names[1] = {"q"};
    static const char* const o_names[1] = {"o"};

    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "K_cache", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "V_cache", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "O", NULL, false, true, 16);
    rocke_fmha_kernel_builder_add_ptr(kb, "block_table", "i32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "cu_seqlens_q", "i32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "seqlens_k", "i32", true, 4);
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "total_q", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "batch", "i32");
    rocke_fmha_kernel_builder_add_strides(kb, q_names, 1);
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_block", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_page", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_kv_head", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_v_block", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_v_page", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_v_kv_head", "i32");
    rocke_fmha_kernel_builder_add_strides(kb, o_names, 1);
    rocke_fmha_kernel_builder_add_scalar(kb, "block_table_stride", "i32");
}

/* ===================================================================== *
 *  _paged_row closure environment
 * ===================================================================== *
 *
 * The Python _paged_row(stride_blk, stride_pg, stride_h) hoists
 *   head_off = b.mul(kv_head_idx, stride_h)
 * once at closure-creation time, then returns _row(b, k_idx) which emits the
 * per-k_idx block_id load + block/page muls. This env captures the hoisted
 * head_off plus the loop-invariant constants. */
typedef struct rocke_fpp_paged_one
{
    rocke_value_t* stride_blk;
    rocke_value_t* stride_pg;
    rocke_value_t* head_off; /* hoisted kv_head_idx * stride_h */
} rocke_fpp_paged_one_t;

/* The warp body opts carry a single `user` shared by both row-base callbacks,
 * so the combined env holds BOTH the K and V closures plus the loop-invariant
 * constants. Two thin wrapper functions select the K or V closure. The MFMA
 * params have per-callback users, so they thread &env with the K / V wrapper
 * exactly the same way. */
typedef struct rocke_fpp_paged_env
{
    rocke_fpp_paged_one_t k;
    rocke_fpp_paged_one_t v;
    rocke_value_t* block_table;
    rocke_value_t* block_table_row_base;
    rocke_value_t* c_pg_log2;
    rocke_value_t* c_pg_mask;
} rocke_fpp_paged_env_t;

/* def _row(b, k_idx):
 *     block_idx_in_seq = b.lshr(k_idx, c_pg_log2)
 *     page_in_block    = b.land(k_idx, c_pg_mask)
 *     block_id = b.global_load_i32(block_table,
 *                                  b.add(block_table_row_base, block_idx_in_seq))
 *     return b.add(b.add(b.mul(block_id, stride_blk),
 *                        b.mul(page_in_block, stride_pg)),
 *                  head_off)
 *
 * Matches both the warp body (rocke_fmha_row_base_fn) and MFMA
 * (rocke_attn_row_base_fn) callback shapes: (b, k_idx/row_idx, user) -> i32. */
static rocke_value_t* rocke_fpp_paged_row_impl(rocke_ir_builder_t* b,
                                               rocke_value_t* k_idx,
                                               const rocke_fpp_paged_env_t* e,
                                               const rocke_fpp_paged_one_t* one)
{
    rocke_value_t* block_idx_in_seq = rocke_b_lshr(b, k_idx, e->c_pg_log2);
    rocke_value_t* page_in_block = rocke_b_land(b, k_idx, e->c_pg_mask);
    rocke_value_t* block_id = rocke_b_global_load_i32(
        b, e->block_table, rocke_b_add(b, e->block_table_row_base, block_idx_in_seq), 4);
    /* Python: b.add(b.add(b.mul(block_id, stride_blk),
     *                      b.mul(page_in_block, stride_pg)), head_off).
     * The block_id-mul is emitted before the page-mul; sequence into temps so
     * C's argument evaluation order matches Python's left-to-right. */
    {
        rocke_value_t* blk_mul = rocke_b_mul(b, block_id, one->stride_blk);
        rocke_value_t* pg_mul = rocke_b_mul(b, page_in_block, one->stride_pg);
        return rocke_b_add(b, rocke_b_add(b, blk_mul, pg_mul), one->head_off);
    }
}

static rocke_value_t* rocke_fpp_paged_row_k(rocke_ir_builder_t* b, rocke_value_t* k_idx, void* user)
{
    const rocke_fpp_paged_env_t* e = (const rocke_fpp_paged_env_t*)user;
    return rocke_fpp_paged_row_impl(b, k_idx, e, &e->k);
}

static rocke_value_t* rocke_fpp_paged_row_v(rocke_ir_builder_t* b, rocke_value_t* k_idx, void* user)
{
    const rocke_fpp_paged_env_t* e = (const rocke_fpp_paged_env_t*)user;
    return rocke_fpp_paged_row_impl(b, k_idx, e, &e->v);
}

/* ===================================================================== *
 *  build_fmha_fwd_paged_prefill
 * ===================================================================== */

rocke_kernel_def_t*
    rocke_build_fmha_fwd_paged_prefill(rocke_fmha_kernel_builder_t* kb,
                                       const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                       const char* arch)
{
    return ckc::guard_builder(rocke_fmha_kernel_builder_builder(kb), [&]() -> rocke_kernel_def_t* {
        char reason[256];
        char kname[256];
        rocke_ir_builder_t* b;
        const rocke_fmha_common_spec_t* s;
        int head_size;
        int pg, pg_log2, pg_mask, bs_iters;

        /* Grid coords + ABI values. */
        rocke_value_t *Q, *K_cache, *V_cache, *O;
        rocke_value_t *block_table, *cu_seqlens_q, *seqlens_k_ptr, *scale_log2;
        rocke_value_t *q_token, *head_idx, *kv_head_idx, *block_table_stride;
        rocke_value_t *stride_block, *stride_page, *stride_kv_head;
        rocke_value_t *stride_v_block, *stride_v_page, *stride_v_kv_head;

        /* binary-search loop state. */
        rocke_for_t bs_loop;
        rocke_iter_arg_t bs_args[2];
        rocke_value_t *left, *right, *done, *mid, *cuq_next_mid, *go_right, *nl, *nr;
        rocke_value_t* bs_yields[2];
        rocke_value_t *seq_idx, *cuq_base, *local_q, *seqlen_k;
        rocke_value_t *block_table_row_base, *c_pg_log2, *c_pg_mask;
        rocke_value_t* causal_ctx;

        rocke_fpp_paged_env_t env;

        if(kb == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError(...) */
        if(!rocke_fmha_fwd_paged_prefill_is_valid_spec(spec, arch, reason, sizeof reason))
        {
            /* The Python raises before the builder exists; here we init the builder
             * (so the caller has something to free) and set its sticky error. */
            if(rocke_fmha_fwd_paged_prefill_kernel_name(spec, kname, sizeof kname) != ROCKE_OK)
            {
                rocke_fpp_copy_str(kname, sizeof kname, ROCKE_FPP_DEFAULT_NAME);
            }
            if(rocke_fmha_kernel_builder_init(kb, kname, &spec->common) != ROCKE_OK)
            {
                return NULL;
            }
            b = rocke_fmha_kernel_builder_builder(kb);
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid fmha_fwd_paged_prefill spec: %s", reason);
            return NULL;
        }

        s = &spec->common;
        head_size = s->shape.head_size;

        /* kb = FmhaKernelBuilder(spec.kernel_name(), s) */
        if(rocke_fmha_fwd_paged_prefill_kernel_name(spec, kname, sizeof kname) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_fmha_kernel_builder_init(kb, kname, s) != ROCKE_OK)
        {
            return NULL;
        }

        /* kb.block_size(WARP_SIZE) */
        rocke_fmha_kernel_builder_block_size(kb, ROCKE_FMHA_WARP_SIZE);

        /* _declare_params(kb) */
        rocke_fpp_declare_params(kb);

        /* kb.decode_grid() -> (q_token, head_idx, kv_head_idx) */
        rocke_fmha_kernel_builder_decode_grid(kb, -1, false, &q_token, &head_idx, &kv_head_idx);

        /* b = kb.builder */
        b = rocke_fmha_kernel_builder_builder(kb);

        /* Q = kb.tensor("Q"); ... ; block_table = kb.ptr("block_table"); ... */
        Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
        K_cache = rocke_fmha_kernel_builder_tensor(kb, "K_cache");
        V_cache = rocke_fmha_kernel_builder_tensor(kb, "V_cache");
        O = rocke_fmha_kernel_builder_tensor(kb, "O");
        block_table = rocke_fmha_kernel_builder_ptr(kb, "block_table");
        cu_seqlens_q = rocke_fmha_kernel_builder_ptr(kb, "cu_seqlens_q");
        seqlens_k_ptr = rocke_fmha_kernel_builder_ptr(kb, "seqlens_k");
        scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
        /* q_token/head_idx/kv_head_idx already from decode_grid (== kb fields). */
        q_token = kb->q_token;
        head_idx = kb->head_idx;
        kv_head_idx = kb->kv_head_idx;
        block_table_stride = rocke_fmha_kernel_builder_scalar(kb, "block_table_stride");

        /* ---- per-q_token sequence lookup (binary search) ----
         *
         * bs_iters = max(1, int(ceil(log2(batch + 1))))
         * bs_loop = b.scf_for_iter(0, bs_iters, 1,
         *     [("bs_left", 0), ("bs_right", batch)], iv_name="bs_i") */
        bs_iters = (int)ceil(log2((double)spec->batch + 1.0));
        if(bs_iters < 1)
        {
            bs_iters = 1;
        }
        {
            rocke_value_t* bs_lb = rocke_b_const_i32(b, 0);
            rocke_value_t* bs_ub = rocke_b_const_i32(b, bs_iters);
            rocke_value_t* bs_step = rocke_b_const_i32(b, 1);
            bs_args[0].name = "bs_left";
            bs_args[0].init = rocke_b_const_i32(b, 0);
            bs_args[1].name = "bs_right";
            bs_args[1].init = rocke_b_const_i32(b, spec->batch);
            bs_loop
                = rocke_b_scf_for_iter(b, bs_lb, bs_ub, bs_step, bs_args, 2, "bs_i", false, true);
        }

        /* with bs_loop as (_iv, (left, right)): */
        rocke_b_region_enter(b, bs_loop.body);
        left = bs_loop.iter_vars[0];
        right = bs_loop.iter_vars[1];

        /* done = b.cmp_ge(left, right)
         * mid  = b.div(b.add(left, right), 2)
         * cuq_next_mid = b.global_load_i32(cu_seqlens_q, b.add(mid, 1))
         * go_right = b.cmp_le(cuq_next_mid, q_token)
         * nl = b.select(go_right, b.add(mid, 1), left)
         * nr = b.select(go_right, right, mid)
         * b.scf_yield(b.select(done, left, nl), b.select(done, right, nr)) */
        done = rocke_b_cmp_ge(b, left, right);
        /* Python: b.div(b.add(left, right), b.const_i32(2)) -- the add (and its
         * SSA id) is emitted before the const operand. Sequence into a temp so the
         * C argument-evaluation order does not allocate the const id first. */
        {
            rocke_value_t* lr_sum = rocke_b_add(b, left, right);
            mid = rocke_b_div(b, lr_sum, rocke_b_const_i32(b, 2));
        }
        cuq_next_mid = rocke_b_global_load_i32(
            b, cu_seqlens_q, rocke_b_add(b, mid, rocke_b_const_i32(b, 1)), 4);
        go_right = rocke_b_cmp_le(b, cuq_next_mid, q_token);
        nl = rocke_b_select(b, go_right, rocke_b_add(b, mid, rocke_b_const_i32(b, 1)), left);
        nr = rocke_b_select(b, go_right, right, mid);
        bs_yields[0] = rocke_b_select(b, done, left, nl);
        bs_yields[1] = rocke_b_select(b, done, right, nr);
        rocke_b_scf_yield(b, bs_yields, 2);
        rocke_b_region_leave(b);

        /* seq_idx = bs_loop.results[0]
         * cuq_base = b.global_load_i32(cu_seqlens_q, seq_idx)
         * local_q  = b.sub(q_token, cuq_base)
         * seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx) */
        seq_idx = (bs_loop.op != NULL) ? bs_loop.op->results[0] : NULL;
        cuq_base = rocke_b_global_load_i32(b, cu_seqlens_q, seq_idx, 4);
        local_q = rocke_b_sub(b, q_token, cuq_base);
        seqlen_k = rocke_b_global_load_i32(b, seqlens_k_ptr, seq_idx, 4);

        /* ---- per-k_idx paged-KV indirection setup ----
         *
         * block_table_row_base = b.mul(seq_idx, block_table_stride)
         * pg = page_block_size; pg_log2 = pg.bit_length()-1; pg_mask = pg-1
         * c_pg_log2 = b.const_i32(pg_log2); c_pg_mask = b.const_i32(pg_mask) */
        block_table_row_base = rocke_b_mul(b, seq_idx, block_table_stride);
        pg = spec->page_block_size;
        pg_log2 = 0;
        {
            int v = pg;
            while(v > 1)
            {
                v >>= 1;
                pg_log2++;
            }
        }
        pg_mask = pg - 1;
        c_pg_log2 = rocke_b_const_i32(b, pg_log2);
        c_pg_mask = rocke_b_const_i32(b, pg_mask);

        /* stride_block = kb.scalar("stride_block"); ... stride_v_kv_head = ... */
        stride_block = rocke_fmha_kernel_builder_scalar(kb, "stride_block");
        stride_page = rocke_fmha_kernel_builder_scalar(kb, "stride_page");
        stride_kv_head = rocke_fmha_kernel_builder_scalar(kb, "stride_kv_head");
        stride_v_block = rocke_fmha_kernel_builder_scalar(kb, "stride_v_block");
        stride_v_page = rocke_fmha_kernel_builder_scalar(kb, "stride_v_page");
        stride_v_kv_head = rocke_fmha_kernel_builder_scalar(kb, "stride_v_kv_head");

        /* causal_ctx = local_q if mask_mode in ("causal","sliding_window") else None */
        causal_ctx = (s->mask_mode == ROCKE_FMHA_MASK_CAUSAL
                      || s->mask_mode == ROCKE_FMHA_MASK_SLIDING_WINDOW)
                         ? local_q
                         : NULL;

        /* ---- _paged_row closure construction (call site, left-to-right) ----
         *
         * The K closure's head_off mul is emitted first (it is the first arg
         * evaluated), then the V closure's head_off mul. The shared loop-invariant
         * constants are captured by value into each env. */
        env.block_table = block_table;
        env.block_table_row_base = block_table_row_base;
        env.c_pg_log2 = c_pg_log2;
        env.c_pg_mask = c_pg_mask;
        env.k.stride_blk = stride_block;
        env.k.stride_pg = stride_page;
        env.v.stride_blk = stride_v_block;
        env.v.stride_pg = stride_v_page;
        /* The K / V closure head_off muls are emitted at the body call site (when the
         * _paged_row(...) kwargs are evaluated). In the MFMA path q_tile_base is
         * computed first, so the head_off muls are emitted inside each branch in the
         * Python evaluation order (q_tile_base, then K head_off, then V head_off). */

        if(spec->use_mfma_body)
        {
            /* P67: MFMA-tiled body.
             *
             * q_tile_base = b.mul(q_token, MFMA_ATTN_BLOCK_M)
             * mfma_attention_fwd_inner_body(b, Q=..., K=K_cache, V=V_cache, O=O, ...,
             *     stride_k_token=stride_page, stride_k_head=stride_kv_head,
             *     stride_v_token=stride_v_page, stride_v_head=stride_v_kv_head, ...,
             *     causal_ctx_offset=causal_ctx,
             *     k_row_base_fn=_paged_row(stride_block, stride_page, stride_kv_head),
             *     v_row_base_fn=_paged_row(stride_v_block, stride_v_page,
             *                              stride_v_kv_head), arch=arch) */
            rocke_mfma_attn_params_t p;
            rocke_value_t* q_tile_base
                = rocke_b_mul(b, q_token, rocke_b_const_i32(b, ROCKE_MFMA_ATTN_BLOCK_M));

            /* kwargs left-to-right: k_row_base_fn=_paged_row(stride_block,...) then
             * v_row_base_fn=_paged_row(stride_v_block,...) */
            env.k.head_off = rocke_b_mul(b, kv_head_idx, stride_kv_head);
            env.v.head_off = rocke_b_mul(b, kv_head_idx, stride_v_kv_head);

            memset(&p, 0, sizeof p);
            p.Q = Q;
            p.K = K_cache;
            p.V = V_cache;
            p.O = O;
            p.head_size = head_size;
            p.seqlen_k = seqlen_k;
            p.q_tile_base = q_tile_base;
            p.head_idx = head_idx;
            p.kv_head_idx = kv_head_idx;
            p.q_pos_base = local_q;
            p.stride_q_token = rocke_fmha_kernel_builder_stride_token(kb, "q");
            p.stride_q_head = rocke_fmha_kernel_builder_stride_head(kb, "q");
            p.stride_k_token = stride_page;
            p.stride_k_head = stride_kv_head;
            p.stride_v_token = stride_v_page;
            p.stride_v_head = stride_v_kv_head;
            p.stride_o_token = rocke_fmha_kernel_builder_stride_token(kb, "o");
            p.stride_o_head = rocke_fmha_kernel_builder_stride_head(kb, "o");
            p.scale_log2 = scale_log2;
            p.dtype = s->dtype;
            p.mask_mode = rocke_fpp_attn_mask(s->mask_mode);
            p.sliding_window = s->sliding_window;
            p.causal_ctx_offset = causal_ctx;
            p.k_row_base_fn = rocke_fpp_paged_row_k;
            p.k_row_base_user = &env;
            p.v_row_base_fn = rocke_fpp_paged_row_v;
            p.v_row_base_user = &env;
            p.arch = arch;

            rocke_mfma_attention_fwd_inner_body(b, &p);
        }
        else
        {
            /* fmha_warp_fwd_inner_body(b, Q=Q, K=K_cache, V=V_cache, O=O, ...,
             *     q_token=q_token, head_idx=head_idx, kv_head_idx=kv_head_idx,
             *     stride_k_token=stride_page, stride_k_head=stride_kv_head,
             *     stride_v_token=stride_v_page, stride_v_head=stride_v_kv_head, ...,
             *     causal_ctx_len=causal_ctx,
             *     k_row_base_fn=_paged_row(stride_block, stride_page, stride_kv_head),
             *     v_row_base_fn=_paged_row(stride_v_block, stride_v_page,
             *                              stride_v_kv_head)) */
            rocke_fmha_warp_fwd_opts_t o;

            /* kwargs left-to-right: k_row_base_fn=_paged_row(stride_block,...) then
             * v_row_base_fn=_paged_row(stride_v_block,...) */
            env.k.head_off = rocke_b_mul(b, kv_head_idx, stride_kv_head);
            env.v.head_off = rocke_b_mul(b, kv_head_idx, stride_v_kv_head);

            memset(&o, 0, sizeof o);
            o.Q = Q;
            o.K = K_cache;
            o.V = V_cache;
            o.O = O;
            o.head_size = head_size;
            o.seqlen_k = seqlen_k;
            o.q_token = q_token;
            o.head_idx = head_idx;
            o.kv_head_idx = kv_head_idx;
            o.stride_q_token = rocke_fmha_kernel_builder_stride_token(kb, "q");
            o.stride_q_head = rocke_fmha_kernel_builder_stride_head(kb, "q");
            o.stride_k_token = stride_page;
            o.stride_k_head = stride_kv_head;
            o.stride_v_token = stride_v_page;
            o.stride_v_head = stride_v_kv_head;
            o.stride_o_token = rocke_fmha_kernel_builder_stride_token(kb, "o");
            o.stride_o_head = rocke_fmha_kernel_builder_stride_head(kb, "o");
            o.scale_log2 = scale_log2;
            o.dtype = s->dtype;
            o.mask_mode = rocke_fpp_mask_name(s->mask_mode);
            o.sliding_window = s->sliding_window;
            o.causal_ctx_len = causal_ctx;
            o.k_row_base_fn = rocke_fpp_paged_row_k;
            o.v_row_base_fn = rocke_fpp_paged_row_v;
            o.user = &env; /* combined K+V env: the two wrappers select K vs V */

            rocke_fmha_warp_fwd_inner_body(b, &o);
        }

        /* b.ret() */
        rocke_b_ret(b);

        /* Python raises (e.g. load_vec n=ept not in {2,4,8} for head_size 192)
         * out of build_fmha_fwd_paged_prefill, so no kernel is produced. Mirror
         * that here: a sticky builder error => return NULL rather than a partial
         * kernel the caller would lower into bogus IR. */
        if(rocke_ir_builder_status(b) != ROCKE_OK)
        {
            return NULL;
        }

        /* return kb.kernel */
        return rocke_fmha_kernel_builder_kernel(kb);
    });
}

/* ===================================================================== *
 *  grid / signature / lower
 * ===================================================================== */

/* def fmha_fwd_paged_prefill_grid(spec, *, total_q):
 *     return (total_q, spec.common.shape.num_query_heads, 1) */
rocke_status_t rocke_fmha_fwd_paged_prefill_grid(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                                 int total_q,
                                                 int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = total_q;
    out[1] = spec->common.shape.num_query_heads;
    out[2] = 1;
    return ROCKE_OK;
}

/* def fmha_fwd_paged_prefill_signature(spec):
 *     kb = FmhaKernelBuilder("rocke_fmha_fwd_paged_prefill_sig_probe", spec.common)
 *     _declare_params(kb)
 *     return kb.signature() */
rocke_status_t
    rocke_fmha_fwd_paged_prefill_signature(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                           rocke_arena_t* arena,
                                           const rocke_sig_entry_t** out_items,
                                           size_t* out_count)
{
    rocke_fmha_kernel_builder_t kb;
    rocke_status_t st;

    if(spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_fmha_kernel_builder_init(
        &kb, "rocke_fmha_fwd_paged_prefill_sig_probe", &spec->common);
    if(st != ROCKE_OK)
    {
        return st;
    }
    rocke_fpp_declare_params(&kb);
    st = rocke_fmha_kernel_builder_signature(&kb, arena, out_items, out_count);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}

static void rocke_fpp_copy_err(char* err, size_t err_cap, const char* m)
{
    rocke_fpp_copy_str(err, err_cap, (m != NULL) ? m : "fmha_fwd_paged_prefill lower failed");
}

rocke_status_t
    rocke_fmha_fwd_paged_prefill_lower_to_llvm(const rocke_fmha_fwd_paged_prefill_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap)
{
    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* kernel;
    rocke_ir_builder_t* b;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_fpp_copy_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_fmha_fwd_paged_prefill(&kb, spec, arch);
    if(kernel == NULL)
    {
        b = rocke_fmha_kernel_builder_builder(&kb);
        st = rocke_ir_builder_status(b);
        rocke_fpp_copy_err(err, err_cap, rocke_ir_builder_error(b));
        rocke_fmha_kernel_builder_free(&kb);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_fmha_kernel_builder_free(&kb);
    return st;
}
