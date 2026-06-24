// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of the end-to-end fused-MoE forward orchestrator
 * (ck_dsl/instances/common/fused_moe_e2e.py; the requested C file name uses
 * the `helpers.fused_moe_e2e_orchestrator` alias).
 *
 * See the header for the contract and the scope split. The three value
 * producers (_workspace_specs, _host_preshuffle_b, _build_per_expert_problems)
 * reproduce the Python arithmetic / layout exactly. The three launch entries
 * (forward / static / dynamic) reproduce the host-side dispatch decision and
 * sizing only; the HIP launch chain is a TODO(port) stub and returns
 * CKC_ERR_NOTIMPL.
 */
#include "ckc/helper_ck_dsl.helpers.fused_moe_e2e_orchestrator.h"

#include <string.h>

/* ------------------------------------------------------------------ *
 * dtype labels
 * ------------------------------------------------------------------ */

int ckc_fmoe_dtype_elem_bytes(ckc_fmoe_dtype_t dtype)
{
    /* Python _ensure_2byte_dtype: f16/fp16/bf16 -> 2; else ValueError. */
    switch(dtype)
    {
    case CKC_FMOE_DTYPE_F16:
    case CKC_FMOE_DTYPE_BF16: return 2;
    default: return 0;
    }
}

const char* ckc_fmoe_dtype_to_universal(ckc_fmoe_dtype_t dtype)
{
    /* Python _gemm_dtype_to_universal: f16/fp16 -> "fp16", bf16 -> "bf16". */
    switch(dtype)
    {
    case CKC_FMOE_DTYPE_F16: return "fp16";
    case CKC_FMOE_DTYPE_BF16: return "bf16";
    default: return NULL;
    }
}

/* ------------------------------------------------------------------ *
 * FusedMoeForwardSpec.total_pairs
 * ------------------------------------------------------------------ */

/* Integration: ckc_fmoe_forward_spec_total_pairs() is defined canonically in
 * helper_ck_dsl.helpers.fused_moe_e2e_spec.c (declared in both spec.h and this
 * module's orchestrator.h). The duplicate definition that the emitter placed
 * here would produce a multiple-definition link error, so it is omitted; the
 * canonical definition is linked in from the spec part-file. */

/* ------------------------------------------------------------------ *
 * _workspace_specs
 *
 * Reproduces the 15-entry WorkspaceSpec tuple in Python declaration order.
 * Shapes use: total_pairs = tokens*topk; act dtype = the activation dtype.
 * ------------------------------------------------------------------ */

ckc_status_t ckc_fused_moe_forward_workspace_specs(const ckc_fmoe_forward_spec_t* spec,
                                                   ckc_ws_spec_t* out,
                                                   size_t cap,
                                                   size_t* n_written)
{
    int act_bytes;
    int T, K, E, H, I, TP;
    ckc_ws_spec_t table[CKC_FMOE_NUM_WORKSPACE_SPECS];
    size_t i, n;

    if(n_written != NULL)
    {
        *n_written = 0;
    }
    if(spec == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }
    act_bytes = ckc_fmoe_dtype_elem_bytes(spec->dtype);
    if(act_bytes == 0)
    {
        return CKC_ERR_VALUE; /* unsupported activation dtype */
    }

    T  = spec->tokens;
    K  = spec->topk;
    E  = spec->experts;
    H  = spec->hidden;
    I  = spec->intermediate;
    TP = T * K; /* total_pairs */

    /* The 15 entries, in the exact Python order. Helper macros keep the
     * table declaration readable; rank/elem_bytes are derived from dtype. */
#define WSPEC1(nm, d0, dk, eb)      \
    do                              \
    {                               \
        table[n].name       = (nm); \
        table[n].shape[0]   = (d0); \
        table[n].shape[1]   = 0;    \
        table[n].rank       = 1;    \
        table[n].dtype      = (dk); \
        table[n].elem_bytes = (eb); \
        n++;                        \
    } while(0)
#define WSPEC2(nm, d0, d1, dk, eb)  \
    do                              \
    {                               \
        table[n].name       = (nm); \
        table[n].shape[0]   = (d0); \
        table[n].shape[1]   = (d1); \
        table[n].rank       = 2;    \
        table[n].dtype      = (dk); \
        table[n].elem_bytes = (eb); \
        n++;                        \
    } while(0)

    n = 0;
    WSPEC2("TopkIds", T, K, CKC_WS_I32, 4);
    WSPEC2("TopkWeights", T, K, CKC_WS_F32, 4);
    WSPEC1("Hist", E, CKC_WS_I32, 4);
    WSPEC1("Counter", E, CKC_WS_I32, 4);
    WSPEC1("Offsets", E, CKC_WS_I32, 4);
    WSPEC1("Counts", E, CKC_WS_I32, 4);
    WSPEC1("SortedTokenIds", TP, CKC_WS_I32, 4);
    WSPEC1("SortedTopkIds", TP, CKC_WS_I32, 4);
    WSPEC1("SortedWeights", TP, CKC_WS_F32, 4);
    WSPEC2("GroupedInput", TP, H, CKC_WS_ACT, act_bytes);
    WSPEC2("GateOut", TP, I, CKC_WS_ACT, act_bytes);
    WSPEC2("UpOut", TP, I, CKC_WS_ACT, act_bytes);
    WSPEC2("Hidden", TP, I, CKC_WS_ACT, act_bytes);
    WSPEC2("DownOut", TP, H, CKC_WS_ACT, act_bytes);
    WSPEC2("Y_f32", T, H, CKC_WS_F32, 4);

#undef WSPEC1
#undef WSPEC2

    /* n is now CKC_FMOE_NUM_WORKSPACE_SPECS (15). Copy up to cap. */
    for(i = 0; i < n && i < cap; ++i)
    {
        out[i] = table[i];
    }
    if(n_written != NULL)
    {
        *n_written = (i < n) ? i : n;
    }
    return CKC_OK;
}

/* ------------------------------------------------------------------ *
 * _host_preshuffle_b
 *
 * (E, N, K) row-major -> (E, k_tiles, n_tiles, block_n, block_k). Only the
 * shape contract + the divisibility precondition is reproduced (the element
 * permutation is a torch-runtime concern).
 * ------------------------------------------------------------------ */

ckc_status_t
ckc_fused_moe_host_preshuffle_b(int E, int N, int K, int block_n, int block_k, int out_shape[5])
{
    int n_tiles, k_tiles;

    if(out_shape == NULL || block_n <= 0 || block_k <= 0)
    {
        return CKC_ERR_VALUE;
    }
    /* Python: if N % block_n or K % block_k: raise ValueError(...) */
    if((N % block_n) != 0 || (K % block_k) != 0)
    {
        return CKC_ERR_VALUE;
    }
    n_tiles = N / block_n;
    k_tiles = K / block_k;

    /* Python returns a tensor of shape (E, k_tiles, n_tiles, block_n, block_k). */
    out_shape[0] = E;
    out_shape[1] = k_tiles;
    out_shape[2] = n_tiles;
    out_shape[3] = block_n;
    out_shape[4] = block_k;
    return CKC_OK;
}

/* ------------------------------------------------------------------ *
 * _build_per_expert_problems
 *
 * One GroupedGemmProblem per active expert (count > 0), with the exact Python
 * pointer arithmetic.
 * ------------------------------------------------------------------ */

ckc_status_t ckc_fused_moe_build_per_expert_problems(int experts,
                                                     const int32_t* counts_cpu,
                                                     const int32_t* offsets_cpu,
                                                     uint64_t a_base,
                                                     const uint64_t* b_base,
                                                     uint64_t c_base,
                                                     int a_inner_dim,
                                                     int b_inner_dim,
                                                     int c_inner_dim,
                                                     int elem_bytes,
                                                     ckc_grouped_gemm_problem_t* out,
                                                     size_t cap,
                                                     size_t* n_written)
{
    int e;
    size_t n = 0;

    if(n_written != NULL)
    {
        *n_written = 0;
    }
    if(counts_cpu == NULL || offsets_cpu == NULL || b_base == NULL || out == NULL)
    {
        return CKC_ERR_VALUE;
    }

    for(e = 0; e < experts; ++e)
    {
        int count = (int)counts_cpu[e];
        int offset;
        if(count <= 0)
        {
            continue; /* Python: if count <= 0: continue */
        }
        offset = (int)offsets_cpu[e];
        if(n >= cap)
        {
            /* Out of room: stop emitting but still report what fit. The
             * Python list would have grown unbounded; the C caller sizes
             * `cap` to `experts` to match. */
            break;
        }
        out[n].M = count;
        out[n].N = c_inner_dim;
        out[n].K = b_inner_dim;
        /* A_ptr = a_buf.data_ptr() + offset * a_inner_dim * elem_bytes */
        out[n].A_ptr = a_base + (uint64_t)offset * (uint64_t)a_inner_dim * (uint64_t)elem_bytes;
        /* B_ptr = b_weights[e].data_ptr() */
        out[n].B_ptr = b_base[e];
        /* C_ptr = c_buf.data_ptr() + offset * c_inner_dim * elem_bytes */
        out[n].C_ptr = c_base + (uint64_t)offset * (uint64_t)c_inner_dim * (uint64_t)elem_bytes;
        n++;
    }

    if(n_written != NULL)
    {
        *n_written = n;
    }
    return CKC_OK;
}

/* ------------------------------------------------------------------ *
 * FusedMoeForward.__init__
 *
 * Reproduces the host-side dispatch sizing (arch resolve fallback, the
 * static-vs-dynamic gate, the static slot_size). The arch-aware tile-policy
 * swap (which picks gfx942-legal MFMA atoms) is a TODO(port) stub: it depends on
 * the _default_gemm_tile / _sparse_ / _large_ tile-equality probes that live in
 * gemm_universal; the caller-supplied gemm_tile is taken as-is here.
 * ------------------------------------------------------------------ */

ckc_status_t ckc_fused_moe_forward_init(ckc_fused_moe_forward_t* self,
                                        const ckc_fmoe_forward_spec_t* spec,
                                        const char* arch)
{
    int tile_m;
    int slot;
    long long tke;

    if(self == NULL || spec == NULL)
    {
        return CKC_ERR_VALUE;
    }
    memset(self, 0, sizeof(*self));
    self->spec = *spec;

    /* Python _resolve_launch_arch: explicit arch wins; else probe device and
     * fall back to "gfx950". The device probe is a runtime concern, so an
     * absent arch resolves to the documented "gfx950" fallback. */
    self->arch = (arch != NULL) ? arch : "gfx950";

    /* NAMED GAP (port): the arch-aware default-tile swap (is_bf16 / large-hidden
     * / sparse-vs-dense density / gfx942 narrow-atom fallback) is NOT applied on
     * this legacy standalone surface; the caller-supplied tile is taken as-is.
     *
     * The authoritative, golden-verified implementation of that policy already
     * lives in ckc_fmoe_build_ctx_init (instances/common/fused_moe_e2e_ctx_init)
     * and drives the byte-identity-checked emission path. This module is the
     * older bounded mirror (its own reduced ckc_fmoe_forward_spec uses a dtype
     * ENUM and omits spec.arch / spec.name / the experimental knobs), so the
     * density branch (T*K/E <= 24) and the seven tile factories cannot be reached
     * here without reimplementing the policy. Duplicating it into this superseded
     * surface would risk a second, drifting copy. BLOCKED until this legacy
     * orchestrator is either retired or re-pointed at the ctx-based policy. */

    /* Python: _use_static_offsets = tokens * topk * experts <= 512
     * (the docstring's 256 is stale; the code uses 512). */
    tke = (long long)spec->tokens * (long long)spec->topk * (long long)spec->experts;
    self->use_static_offsets = (tke <= 512);

    /* Python: static_slot_size = ceil(T*K / tile_m) * tile_m, min tile_m. */
    tile_m = spec->gemm_tile.tile_m;
    if(tile_m <= 0)
    {
        return CKC_ERR_VALUE;
    }
    slot = ((spec->tokens * spec->topk + tile_m - 1) / tile_m) * tile_m;
    if(slot < tile_m)
    {
        slot = tile_m;
    }
    self->static_slot_size = slot;
    return CKC_OK;
}

/* ------------------------------------------------------------------ *
 * forward(): static-vs-dynamic dispatch (FusedMoeForward.forward).
 * ------------------------------------------------------------------ */

ckc_status_t ckc_fused_moe_forward_forward(ckc_fused_moe_forward_t* self)
{
    if(self == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* Python: if self._use_static_offsets: _forward_static(...) else
     * _forward_dynamic(...). */
    if(self->use_static_offsets)
    {
        return ckc_fused_moe_forward_static(self);
    }
    return ckc_fused_moe_forward_dynamic(self);
}

/* ------------------------------------------------------------------ *
 * _forward_static / _forward_dynamic
 *
 * Bounded launch-driver stubs. The faithfully-ported sizing they would use
 * (workspace table, static slot_size / total_padded, per-expert problems) is
 * available via the value producers above; the HIP launch chain, torch buffer
 * zeroing, D->H Counts/Offsets copy, and the final f32->act cast are out of
 * scope for a Python-free IR port.
 * ------------------------------------------------------------------ */

ckc_status_t ckc_fused_moe_forward_static(ckc_fused_moe_forward_t* self)
{
    if(self == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* Sizing reproduced (Python: slot_size = self._static_slot_size;
     * total_padded = experts * slot_size) so a caller can size scratch. */
    /* int total_padded = self->spec.experts * self->static_slot_size; */

    /* TODO(port): the single launch_kernel chain (router topk -> static
     * scatter/gather -> interleaved gate+up+silu GEMM -> down-reduce GEMM ->
     * f32 Y cast), workspace prepare/zero, and optional HIP-graph replay. */
    return CKC_ERR_NOTIMPL;
}

ckc_status_t ckc_fused_moe_forward_dynamic(ckc_fused_moe_forward_t* self)
{
    if(self == NULL)
    {
        return CKC_ERR_VALUE;
    }
    /* TODO(port): the 5 streaming launches + 3*E grouped-GEMM dispatch
     * (router topk -> 3-phase sort -> D->H Counts/Offsets copy -> per-expert
     * gate/up/down GEMMs via _build_per_expert_problems -> silu_mul ->
     * topk-weighted f32 reduce -> f32 Y cast), plus the optional de-padded
     * grouped-GEMM fast path (use_grouped_gemm). */
    return CKC_ERR_NOTIMPL;
}
