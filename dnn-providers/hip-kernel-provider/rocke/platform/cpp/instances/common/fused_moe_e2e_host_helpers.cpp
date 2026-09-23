// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_host_helpers.c.c -- C99 port of
 * the VALUE-PRODUCING HOST HELPERS of
 * rocke/instances/common/fused_moe_e2e.py.
 *
 * SCOPE (this TU):
 *   rocke_fmoe_workspace_specs          <- _workspace_specs (lines 1359-1382):
 *       the 15-entry WorkspaceSpec table in declaration order.
 *   rocke_fmoe_host_preshuffle_b        <- _host_preshuffle_b (lines 1053-1075):
 *       (E,N,K) -> (E,k_tiles,n_tiles,block_n,block_k) shape + the
 *       N%block_n / K%block_k ValueError precondition.
 *   rocke_fmoe_ensure_w_down_preshuffled    <- _ensure_w_down_preshuffled
 *   rocke_fmoe_ensure_gu_concat_preshuffled <- _ensure_gu_concat_preshuffled
 *   rocke_fmoe_ensure_gu_interleaved_preshuffled
 *                                         <- _ensure_gu_interleaved_preshuffled
 *   rocke_fmoe_ensure_gu_concat             <- _ensure_gu_concat
 *   rocke_fmoe_ensure_gu_interleaved        <- _ensure_gu_interleaved
 *       the data_ptr-keyed lazy weight-packing caches (cache-key arithmetic
 *       ported; the torch cat/stack/permute/contiguous is TODO(port)).
 *   rocke_fmoe_build_per_expert_problems    <- _build_per_expert_problems
 *                                                          (lines 1384-1420):
 *       per-active-expert pointer arithmetic into GroupedGemmProblem.
 *
 * Peers (ctx lifecycle, launcher ensures, forward phases) live in sibling TUs and
 * are reached only through rocke/instance_fused_moe_e2e_internal.h. This TU does
 * not emit IR; it reproduces the host-side scalar arithmetic byte-faithfully.
 */

#include <string.h>

#include "rocke/instance_fused_moe_e2e_internal.h"

/* ------------------------------------------------------------------ *
 * tensor data_ptr surrogate
 *
 * The Python cache keys are tuples of W.data_ptr() ints. rocke_tensor_t is an
 * opaque host-side handle (forward-declared in the internal header) with no
 * accessor in this codegen-only library; the real device data_ptr is a runtime
 * concern owned by a future HIP backend. For the cache-key arithmetic we use the
 * stable per-tensor handle identity as the data_ptr surrogate -- two calls with
 * the SAME tensor handle produce the SAME key (the production-inference reuse
 * pattern), and distinct handles miss the cache exactly as distinct data_ptrs
 * would. TODO(port): swap for the real device pointer once rocke_tensor_t exposes
 * one.
 * ------------------------------------------------------------------ */
static uint64_t rocke_fmoe_tensor_data_ptr(const rocke_tensor_t* t)
{
    return (uint64_t)(uintptr_t)t;
}

/* ------------------------------------------------------------------ *
 * _workspace_specs (lines 1359-1382)
 *
 * Fill the 15-entry table in the exact Python declaration order. act ==
 * _torch_dtype_for(s.dtype) is the 2-byte activation dtype (elem_bytes 2);
 * i32 / f32 are 4 bytes. s.total_pairs == tokens * topk.
 * ------------------------------------------------------------------ */
rocke_status_t rocke_fmoe_workspace_specs(const rocke_fmoe_build_ctx_t* ctx,
                                          rocke_fmoe_ws_spec_t* out,
                                          size_t cap,
                                          size_t* n_written)
{
    if(ctx == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    const rocke_fmoe_forward_spec_t* s = &ctx->spec;
    const int T = s->tokens;
    const int E = s->experts;
    const int K = s->topk;
    const int H = s->hidden;
    const int I = s->intermediate;
    const int total_pairs = T * K; /* FusedMoeForwardSpec.total_pairs */

    /* The 15-entry table, declaration order (lines 1366-1382). i32/f32 = 4 bytes,
     * act = 2 bytes (the supported f16/bf16 activation dtypes). */
    const rocke_fmoe_ws_spec_t table[ROCKE_FMOE_NUM_WORKSPACE_SPECS] = {
        {"TopkIds", {T, K}, 2, ROCKE_FMOE_WS_I32, 4},
        {"TopkWeights", {T, K}, 2, ROCKE_FMOE_WS_F32, 4},
        {"Hist", {E, 0}, 1, ROCKE_FMOE_WS_I32, 4},
        {"Counter", {E, 0}, 1, ROCKE_FMOE_WS_I32, 4},
        {"Offsets", {E, 0}, 1, ROCKE_FMOE_WS_I32, 4},
        {"Counts", {E, 0}, 1, ROCKE_FMOE_WS_I32, 4},
        {"SortedTokenIds", {total_pairs, 0}, 1, ROCKE_FMOE_WS_I32, 4},
        {"SortedTopkIds", {total_pairs, 0}, 1, ROCKE_FMOE_WS_I32, 4},
        {"SortedWeights", {total_pairs, 0}, 1, ROCKE_FMOE_WS_F32, 4},
        {"GroupedInput", {total_pairs, H}, 2, ROCKE_FMOE_WS_ACT, 2},
        {"GateOut", {total_pairs, I}, 2, ROCKE_FMOE_WS_ACT, 2},
        {"UpOut", {total_pairs, I}, 2, ROCKE_FMOE_WS_ACT, 2},
        {"Hidden", {total_pairs, I}, 2, ROCKE_FMOE_WS_ACT, 2},
        {"DownOut", {total_pairs, H}, 2, ROCKE_FMOE_WS_ACT, 2},
        {"Y_f32", {T, H}, 2, ROCKE_FMOE_WS_F32, 4},
    };

    size_t to_copy = cap < (size_t)ROCKE_FMOE_NUM_WORKSPACE_SPECS
                         ? cap
                         : (size_t)ROCKE_FMOE_NUM_WORKSPACE_SPECS;
    for(size_t i = 0; i < to_copy; ++i)
    {
        out[i] = table[i];
    }
    if(n_written != NULL)
    {
        *n_written = to_copy;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * _host_preshuffle_b (lines 1053-1075)
 *
 * (E, N, K) row-major B  ->  (E, k_tiles, n_tiles, block_n, block_k) contiguous.
 *   n_tiles = N // block_n,  k_tiles = K // block_k
 * Precondition (raise ValueError): N % block_n == 0 and K % block_k == 0.
 * The element permutation (.view/.permute/.contiguous) is a runtime concern;
 * this reproduces the shape contract + precondition exactly.
 * ------------------------------------------------------------------ */
rocke_status_t
    rocke_fmoe_host_preshuffle_b(int E, int N, int K, int block_n, int block_k, int out_shape[5])
{
    if(out_shape == NULL || block_n <= 0 || block_k <= 0)
    {
        return ROCKE_ERR_VALUE;
    }

    /* if N % block_n or K % block_k: raise ValueError (line 1064). */
    if((N % block_n) != 0 || (K % block_k) != 0)
    {
        return ROCKE_ERR_VALUE;
    }

    const int n_tiles = N / block_n;
    const int k_tiles = K / block_k;

    /* return shape (E, k_tiles, n_tiles, block_n, block_k) -- the permute order
     * .permute(0, 3, 1, 2, 4) of the (E, n_tiles, block_n, k_tiles, block_k)
     * view. */
    out_shape[0] = E;
    out_shape[1] = k_tiles;
    out_shape[2] = n_tiles;
    out_shape[3] = block_n;
    out_shape[4] = block_k;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * data_ptr-keyed lazy weight-packing caches
 *
 * Each mirrors the Python:
 *   key = (...data_ptrs..., block_n[, block_k])
 *   if cached is not None and cached_key == key: return cached
 *   cached = <pack>(...); cached_key = key; return cached
 *
 * The <pack> (torch cat/stack/permute/contiguous + _host_preshuffle_b) is a
 * runtime concern: TODO(port). The cache-key arithmetic + hit/miss decision is
 * reproduced exactly so a HIP backend only fills the pack body.
 * ------------------------------------------------------------------ */

/* _ensure_w_down_preshuffled (lines 1077-1092). key = (data_ptr, block_n,
 * block_k). */
rocke_tensor_t* rocke_fmoe_ensure_w_down_preshuffled(rocke_fmoe_build_ctx_t* ctx,
                                                     rocke_tensor_t* W_down)
{
    if(ctx == NULL || W_down == NULL)
    {
        return NULL;
    }

    const int block_n = ctx->spec.gemm_tile.tile_n;
    const int block_k = ctx->spec.gemm_tile.tile_k;
    const uint64_t key[3] = {
        rocke_fmoe_tensor_data_ptr(W_down),
        (uint64_t)block_n,
        (uint64_t)block_k,
    };

    if(ctx->w_down_preshuffled != NULL && ctx->w_down_preshuffled_key_valid
       && ctx->w_down_preshuffled_key[0] == key[0] && ctx->w_down_preshuffled_key[1] == key[1]
       && ctx->w_down_preshuffled_key[2] == key[2])
    {
        return ctx->w_down_preshuffled;
    }

    /* TODO(port): self._w_down_preshuffled =
     *   self._host_preshuffle_b(W_down, block_n, block_k)
     * (host re-arrange of the (E,N,K) B tensor). Until the runtime pack is wired
     * the cache stores no packed tensor; record the key so a future packed build
     * keys correctly. */
    ctx->w_down_preshuffled = NULL; /* TODO(port): packed tensor */
    ctx->w_down_preshuffled_key[0] = key[0];
    ctx->w_down_preshuffled_key[1] = key[1];
    ctx->w_down_preshuffled_key[2] = key[2];
    ctx->w_down_preshuffled_key_valid = true;
    return ctx->w_down_preshuffled;
}

/* _ensure_gu_concat_preshuffled (lines 1094-1113). key = (Wg_ptr, Wu_ptr,
 * block_n). Builds torch.cat([W_gate, W_up], dim=1).contiguous() then
 * _host_preshuffle_b. */
rocke_tensor_t* rocke_fmoe_ensure_gu_concat_preshuffled(rocke_fmoe_build_ctx_t* ctx,
                                                        rocke_tensor_t* W_gate,
                                                        rocke_tensor_t* W_up)
{
    if(ctx == NULL || W_gate == NULL || W_up == NULL)
    {
        return NULL;
    }

    const int block_n = ctx->spec.gemm_tile.tile_n;
    const uint64_t key[3] = {
        rocke_fmoe_tensor_data_ptr(W_gate),
        rocke_fmoe_tensor_data_ptr(W_up),
        (uint64_t)block_n,
    };

    if(ctx->gu_concat_preshuffled != NULL && ctx->gu_concat_preshuffled_key_valid
       && ctx->gu_concat_preshuffled_key[0] == key[0] && ctx->gu_concat_preshuffled_key[1] == key[1]
       && ctx->gu_concat_preshuffled_key[2] == key[2])
    {
        return ctx->gu_concat_preshuffled;
    }

    /* TODO(port): gu = torch.cat([W_gate, W_up], dim=1).contiguous();
     *   self._gu_concat_preshuffled =
     *     self._host_preshuffle_b(gu, block_n, block_k)
     * (block_k = self.spec.gemm_tile.tile_k). */
    ctx->gu_concat_preshuffled = NULL; /* TODO(port): packed tensor */
    ctx->gu_concat_preshuffled_key[0] = key[0];
    ctx->gu_concat_preshuffled_key[1] = key[1];
    ctx->gu_concat_preshuffled_key[2] = key[2];
    ctx->gu_concat_preshuffled_key_valid = true;
    return ctx->gu_concat_preshuffled;
}

/* _ensure_gu_interleaved_preshuffled (lines 1225-1251). key = (Wg_ptr, Wu_ptr,
 * block_n). Builds the interleaved (E, 2*I, H) layout
 * (torch.stack((W_gate, W_up), dim=2).reshape(...).contiguous()) then
 * _host_preshuffle_b. */
rocke_tensor_t* rocke_fmoe_ensure_gu_interleaved_preshuffled(rocke_fmoe_build_ctx_t* ctx,
                                                             rocke_tensor_t* W_gate,
                                                             rocke_tensor_t* W_up)
{
    if(ctx == NULL || W_gate == NULL || W_up == NULL)
    {
        return NULL;
    }

    const int block_n = ctx->spec.gemm_tile.tile_n;
    const uint64_t key[3] = {
        rocke_fmoe_tensor_data_ptr(W_gate),
        rocke_fmoe_tensor_data_ptr(W_up),
        (uint64_t)block_n,
    };

    if(ctx->gu_interleaved_preshuffled != NULL && ctx->gu_interleaved_preshuffled_key_valid
       && ctx->gu_interleaved_preshuffled_key[0] == key[0]
       && ctx->gu_interleaved_preshuffled_key[1] == key[1]
       && ctx->gu_interleaved_preshuffled_key[2] == key[2])
    {
        return ctx->gu_interleaved_preshuffled;
    }

    /* TODO(port): gu = torch.stack((W_gate, W_up), dim=2)
     *               .reshape(E, 2*I, H).contiguous();
     *   self._gu_interleaved_preshuffled =
     *     self._host_preshuffle_b(gu, block_n, block_k). */
    ctx->gu_interleaved_preshuffled = NULL; /* TODO(port): packed tensor */
    ctx->gu_interleaved_preshuffled_key[0] = key[0];
    ctx->gu_interleaved_preshuffled_key[1] = key[1];
    ctx->gu_interleaved_preshuffled_key[2] = key[2];
    ctx->gu_interleaved_preshuffled_key_valid = true;
    return ctx->gu_interleaved_preshuffled;
}

/* _ensure_gu_concat (lines 1295-1312). key = (Wg_ptr, Wu_ptr). Builds
 * torch.cat([W_gate, W_up], dim=1).contiguous() -> (E, 2*I, H). */
rocke_tensor_t* rocke_fmoe_ensure_gu_concat(rocke_fmoe_build_ctx_t* ctx,
                                            rocke_tensor_t* W_gate,
                                            rocke_tensor_t* W_up)
{
    if(ctx == NULL || W_gate == NULL || W_up == NULL)
    {
        return NULL;
    }

    const uint64_t key[2] = {
        rocke_fmoe_tensor_data_ptr(W_gate),
        rocke_fmoe_tensor_data_ptr(W_up),
    };

    if(ctx->gu_concat != NULL && ctx->gu_concat_key_valid && ctx->gu_concat_key[0] == key[0]
       && ctx->gu_concat_key[1] == key[1])
    {
        return ctx->gu_concat;
    }

    /* TODO(port): self._gu_concat =
     *   torch.cat([W_gate, W_up], dim=1).contiguous(). */
    ctx->gu_concat = NULL; /* TODO(port): packed tensor */
    ctx->gu_concat_key[0] = key[0];
    ctx->gu_concat_key[1] = key[1];
    ctx->gu_concat_key_valid = true;
    return ctx->gu_concat;
}

/* _ensure_gu_interleaved (lines 1314-1332). key = (Wg_ptr, Wu_ptr). Builds the
 * interleaved (E, 2*I, H) layout
 * (torch.stack((W_gate, W_up), dim=2).reshape(...).contiguous()). */
rocke_tensor_t* rocke_fmoe_ensure_gu_interleaved(rocke_fmoe_build_ctx_t* ctx,
                                                 rocke_tensor_t* W_gate,
                                                 rocke_tensor_t* W_up)
{
    if(ctx == NULL || W_gate == NULL || W_up == NULL)
    {
        return NULL;
    }

    const uint64_t key[2] = {
        rocke_fmoe_tensor_data_ptr(W_gate),
        rocke_fmoe_tensor_data_ptr(W_up),
    };

    if(ctx->gu_interleaved != NULL && ctx->gu_interleaved_key_valid
       && ctx->gu_interleaved_key[0] == key[0] && ctx->gu_interleaved_key[1] == key[1])
    {
        return ctx->gu_interleaved;
    }

    /* TODO(port): self._gu_interleaved =
     *   torch.stack((W_gate, W_up), dim=2).reshape(E, 2*I, H).contiguous(). */
    ctx->gu_interleaved = NULL; /* TODO(port): packed tensor */
    ctx->gu_interleaved_key[0] = key[0];
    ctx->gu_interleaved_key[1] = key[1];
    ctx->gu_interleaved_key_valid = true;
    return ctx->gu_interleaved;
}

/* ------------------------------------------------------------------ *
 * _build_per_expert_problems (lines 1384-1420)
 *
 * Walk the per-expert counts/offsets and emit one GroupedGemmProblem per active
 * (count > 0) expert with the exact pointer arithmetic:
 *   A_ptr = a_base + offset * a_inner_dim * elem_bytes
 *   B_ptr = b_base[e]
 *   C_ptr = c_base + offset * c_inner_dim * elem_bytes
 *   M = count, N = c_inner_dim, K = b_inner_dim.
 * ------------------------------------------------------------------ */
rocke_status_t rocke_fmoe_build_per_expert_problems(rocke_fmoe_build_ctx_t* ctx,
                                                    const int32_t* counts_cpu,
                                                    const int32_t* offsets_cpu,
                                                    uint64_t a_base,
                                                    const uint64_t* b_base,
                                                    uint64_t c_base,
                                                    int a_inner_dim,
                                                    int b_inner_dim,
                                                    int c_inner_dim,
                                                    int elem_bytes,
                                                    rocke_grouped_gemm_problem_t* out,
                                                    size_t cap,
                                                    size_t* n_written)
{
    if(ctx == NULL || counts_cpu == NULL || offsets_cpu == NULL || b_base == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    const int experts = ctx->spec.experts;
    size_t written = 0;

    /* for e in range(self.spec.experts): (line 1405). */
    for(int e = 0; e < experts; ++e)
    {
        const int count = (int)counts_cpu[e];
        if(count <= 0) /* if count <= 0: continue (line 1407). */
        {
            continue;
        }
        const int offset = (int)offsets_cpu[e];

        if(written >= cap)
        {
            /* Out of output capacity; stop appending (Python's list grows
             * unbounded -- the C cap is the caller's promise it sized `out` to at
             * least the active-expert count). */
            break;
        }

        rocke_grouped_gemm_problem_t* p = &out[written];
        p->M = count;
        p->N = c_inner_dim;
        p->K = b_inner_dim;
        p->A_ptr = a_base + (uint64_t)offset * (uint64_t)a_inner_dim * (uint64_t)elem_bytes;
        p->B_ptr = b_base[e];
        p->C_ptr = c_base + (uint64_t)offset * (uint64_t)c_inner_dim * (uint64_t)elem_bytes;
        ++written;
    }

    if(n_written != NULL)
    {
        *n_written = written;
    }
    return ROCKE_OK;
}
