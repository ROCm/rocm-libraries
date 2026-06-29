// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_fused_moe_e2e_instance_fused_moe_e2e_ctx_init.c.c -- C99 port of the
 * FusedMoeForward ctx LIFECYCLE + TILE POLICY surface of
 * rocke/instances/common/fused_moe_e2e.py.
 *
 * SCOPE (this TU):
 *   rocke_fmoe_build_ctx_init   <- FusedMoeForward.__init__ (lines 692-831):
 *       _resolve_launch_arch, the bf16/large/sparse/gfx942 tile-swap policy
 *       (driven by rocke_fmoe_tile_eq + the seven tile factories), the
 *       T*K*E <= 512 static-offset gate, the ceil(T*K/tile_m)*tile_m slot size,
 *       and zeroing every launcher / cache / weight slot.
 *   rocke_fmoe_build_ctx_destroy
 *
 * Peers (launcher ensures, forward phases, host helpers) live in sibling TUs and
 * are reached only through rocke/instance_fused_moe_e2e_internal.h. This TU does
 * not emit IR; it reproduces the host-side scalar arithmetic byte-faithfully.
 */

#include <string.h>

#include "rocke/instance_fused_moe_e2e_internal.h"

/* ------------------------------------------------------------------ *
 * ctx lifecycle
 * ------------------------------------------------------------------ */

rocke_status_t rocke_fmoe_build_ctx_init(rocke_fmoe_build_ctx_t* ctx,
                                         const rocke_fmoe_forward_spec_t* spec,
                                         const char* arch)
{
    if(ctx == NULL || spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Start from a fully-zeroed ctx so every launcher / cache / weight slot is
     * NULL and every *_key_valid flag is false (mirrors __init__ setting every
     * lazy attribute to None). */
    memset(ctx, 0, sizeof(*ctx));

    /* Local working copy of the spec; the tile-swap policy mutates spec.gemm_tile
     * in place (Python `spec.gemm_tile = ...`) before it is stored as self.spec. */
    rocke_fmoe_forward_spec_t s = *spec;

    /* ---- resolve the compilation target up front (line 696) ---- *
     * _resolve_launch_arch(spec.arch): explicit arch wins; the NULL/no-device
     * arm resolves to "gfx950" inside the helper. The function arg `arch`
     * overrides spec.arch when non-NULL (matches the C build-entry contract);
     * otherwise the spec's own arch is resolved. */
    const char* resolved_arch = rocke_fmoe_resolve_launch_arch(arch != NULL ? arch : s.arch);
    /* _resolve_launch_arch never returns NULL (falls back to "gfx950"), but be
     * defensive: a NULL would only come from a contract break. */
    if(resolved_arch == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    const bool is_gfx942 = (strcmp(resolved_arch, "gfx942") == 0);

    /* ---- shape-aware tile policy (lines 698-748) ---- *
     * Only override the auto/default tile; a caller-supplied tile is respected.
     * is_bf16 = spec.dtype in ("bf16",). dtype may be NULL only on a malformed
     * spec; treat NULL as not-bf16 (it would fail the universal-dtype lookup
     * downstream anyway). */
    const bool is_bf16 = (s.dtype != NULL && strcmp(s.dtype, "bf16") == 0);

    const rocke_gemm_tile_spec_t default_tile = rocke_fmoe_default_gemm_tile();

    if(is_bf16 && rocke_fmoe_tile_eq(&s.gemm_tile, &default_tile))
    {
        /* 32x32 atom is F16-only; switch to a BF16-compatible tile.
         * gfx942 has no wide 16x16x32 bf16 atom -> use the narrow one. */
        s.gemm_tile = is_gfx942 ? rocke_fmoe_default_bf16_gemm_tile_gfx942()
                                : rocke_fmoe_default_bf16_gemm_tile();
    }
    else if(rocke_fmoe_tile_eq(&s.gemm_tile, &default_tile) && s.hidden >= 1024 && s.tokens >= 32)
    {
        /* Large-hidden, multi-token regime. Pick the tile by routing DENSITY
         * (average routed tokens per expert = T*K/E). The Python computes this
         * as a float division and compares avg_per_expert <= 24; reproduce the
         * float comparison exactly to keep the policy byte-identical. */
        const double avg_per_expert
            = (double)(s.tokens * s.topk) / (double)(s.experts > 1 ? s.experts : 1);
        const bool sparse = (avg_per_expert <= 24.0);
        if(sparse)
        {
            s.gemm_tile = is_gfx942 ? rocke_fmoe_sparse_batch_gemm_tile_gfx942()
                                    : rocke_fmoe_sparse_batch_gemm_tile();
        }
        else
        {
            s.gemm_tile = is_gfx942 ? rocke_fmoe_default_gemm_tile_gfx942()
                                    : rocke_fmoe_large_batch_gemm_tile();
        }
    }
    else if(is_gfx942 && rocke_fmoe_tile_eq(&s.gemm_tile, &default_tile))
    {
        /* f16 decode/small default uses the wide 32x32x16 atom; gfx942 needs the
         * narrow 32x32x8 atom. */
        s.gemm_tile = rocke_fmoe_default_gemm_tile_gfx942();
    }

    /* ---- store the resolved environment on the ctx (lines 749, 696-700) ---- */
    ctx->spec = s; /* self.spec (AFTER the tile-swap policy)        */
    ctx->arch = resolved_arch; /* self.arch = _resolve_launch_arch(...)         */
    ctx->is_gfx942 = is_gfx942;
    ctx->is_bf16 = is_bf16;

    /* Sub-kernel launchers + caches + cached/preshuffled weights are all left
     * NULL by the memset above (lines 750-801, 810): self._sort_launcher /
     * _fused_moe_launcher are eagerly built in Python, but in this codegen-only
     * port they are lazily compiled by their _ensure_* peers; the pool +
     * captured-graph slots are likewise initialised to NULL here. The
     * moe_gemm_launcher_cache table is empty (len 0). */
    ctx->moe_gemm_launcher_cache_len = 0;

    /* ---- static-offset gate (line 818) ---- *
     * Enable static-offset mode when T*K*E is small enough that the fixed
     * slot_size = T*K over-padding is cheap. Measured MI355X cutoff is 512. */
    ctx->use_static_offsets = (s.tokens * s.topk * s.experts) <= 512;

    /* ---- static slot_size (lines 825-830) ---- *
     * Smallest tile_m-aligned size that fits the worst-case count[e] = T*K
     * (all tokens routed to one expert), floored at tile_m. */
    const int tile_m = s.gemm_tile.tile_m;
    int slot = ((s.tokens * s.topk + tile_m - 1) / tile_m) * tile_m;
    if(slot < tile_m)
    {
        slot = tile_m;
    }
    ctx->static_slot_size = slot;

    return ROCKE_OK;
}

void rocke_fmoe_build_ctx_destroy(rocke_fmoe_build_ctx_t* ctx)
{
    if(ctx == NULL)
    {
        return;
    }

    /* Release pool / launchers / cached weights. In Python these are GC'd; in a
     * full HIP backend each carries an owned HSACO / module / device buffer that
     * must be freed here. In this codegen-only library the launcher / pool /
     * tensor peers are opaque forward-decls with no allocator wired, so there is
     * nothing to free yet -- the lifecycle hook exists so callers can rely on it
     * and a future HIP backend fills it in.
     *
     * TODO(port): free ctx->pool, every ctx->*_launcher, the
     * moe_gemm_launcher_cache[].launcher entries, and the cached/preshuffled
     * weight tensors + static_offsets once the runtime peers own resources. */

    /* Leave the struct zeroed so a double-destroy / post-destroy read is inert. */
    memset(ctx, 0, sizeof(*ctx));
}
