// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_2d_gfx950_attention_tiled_2d_public_entry_glue.c
 * -- PUBLIC ENTRY / GLUE bucket of the chunked C99 port of the gfx950 / CDNA4
 * (MI355X) WIDE-ATOM tiled-2D unified-attention kernel builder
 * (rocke/instances/gfx950/attention_tiled_2d.py).
 *
 * SCOPE (this translation unit):
 *   - the two module-level 32x32 C-distribution helpers
 *       rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row / _col  (Python 79-133).
 *     The C distribution is dtype-independent and byte-identical to the gfx942
 *     sibling's rocke__mfma_32x32_c_row/_col (same _C32_DIST.calculate_x wiring),
 *     so these forward to that already-ported, byte-faithful implementation.
 *   - the admission gate (supports_tiled_2d, Python lines 588-703)
 *       rocke_gfx950_attention_tiled_2d_supports_args_default
 *       rocke_gfx950_attention_tiled_2d_supports
 *   - the body-context prologue
 *       rocke_gfx950_attn2d_build_ctx_init  (Python build body lines 711-1151):
 *       require_tiled_attention_arch(arch) gate -> dtype fp16/bf16 gate ->
 *       WIDE-atom geometry select (Py813-814) -> the whole ALL-CAPS config
 *       derivation -> kernel attrs + param decls -> grid ids + wave
 *       decomposition -> binary-search seq_idx + cu_q bounds + Q-block geometry
 *       + early-return -> LDS layout decisions + every smem_alloc handle -> PV
 *       TransposeLdsReader bind (Py1128) -> the SSA constants block.
 *   - the build driver
 *       rocke_gfx950_build_unified_attention_2d_tiled
 *     which zero-inits the shared ctx, calls rocke_gfx950_attn2d_build_ctx_init,
 *     then drives the phase functions in the exact Python execution order:
 *         emit_q_load -> emit_loop_bounds_and_inits -> emit_preloop ->
 *         emit_q_gather -> emit_preloop_prefetch -> emit_licm_hoist ->
 *         emit_kv_step -> drive_kv_loop -> emit_epilogue (returns b->kernel).
 *   - the _new init-from-spec convenience wrapper
 *       rocke_gfx950_build_unified_attention_2d_tiled_new
 *   - the build->lower convenience
 *       rocke_gfx950_attention_tiled_2d_lower_to_llvm
 *
 * Every IR node is arena-owned (b->arena). Error model mirrors the rest of the
 * C port: build/lower route errors through the sticky-error IRBuilder; the
 * supports gate returns a bool + reason string.
 */
#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_2d_internal.h"

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_helper_rocke.helpers.attention.h" /* binary_search_seq_idx */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h" /* rocke__mfma_32x32_c_* */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join  */

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ===================================================================== *
 *  Local helpers
 * ===================================================================== */

static bool rocke_g950_streq(const char* a, const char* c)
{
    if(a == c)
        return true;
    if(!a || !c)
        return false;
    return strcmp(a, c) == 0;
}

/* Write a static diagnostic into err (capacity err_cap) when non-NULL. */
static void rocke_g950_set_err(char* err, size_t err_cap, const char* m)
{
    size_t n;
    if(err == NULL || err_cap == 0 || m == NULL)
        return;
    n = strlen(m);
    if(n >= err_cap)
        n = err_cap - 1;
    memcpy(err, m, n);
    err[n] = '\0';
}

/* Write a reason string into the supports `reason` buffer (cap reason_cap),
 * NUL-terminated. NULL reason is a no-op. Mirrors returning the structured
 * Python reason string. */
static void rocke_g950_reason(char* reason, size_t reason_cap, const char* m)
{
    rocke_g950_set_err(reason, reason_cap, m);
}

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary (ckc::guard_builder in the build entry) catches it and records
 * status + message on the builder, so the extern "C" ABI is unchanged.
 * [[noreturn]] keeps the existing `...; return false/NULL;` call sites valid --
 * the trailing return is simply never reached. */
[[noreturn]] static void rocke_g950_fail(rocke_ir_builder_t* b, rocke_status_t st, const char* msg)
{
    (void)b;
    ckc::raise_status(st, msg ? msg : "");
}

/* ===================================================================== *
 *  Module-level 32x32 C-distribution helpers (Python lines 79-133).
 *
 *  The C-warp distribution of the f16 32x32x16 atom is dtype-independent and
 *  identical across gfx942/gfx950; the gfx942 sibling already ports it
 *  byte-faithfully (same div/mod consts, same _C32_DIST.calculate_x ys/ps
 *  wiring, same trailing N-tile add). Forward to it.
 * ===================================================================== */
rocke_value_t* rocke_gfx950_attention_tiled_2d_mfma_32x32_c_row(rocke_ir_builder_t* b,
                                                                rocke_value_t* lane,
                                                                int elem_idx)
{
    return rocke__mfma_32x32_c_row(b, lane, elem_idx);
}

rocke_value_t* rocke_gfx950_attention_tiled_2d_mfma_32x32_c_col(rocke_ir_builder_t* b,
                                                                rocke_value_t* lane,
                                                                int n_tile32)
{
    return rocke__mfma_32x32_c_col(b, lane, n_tile32);
}

/* ===================================================================== *
 *  validate_tiled_attention_arch (instances/common/attention_arch.py).
 *
 *  Catalog-driven faithful port of _wide_k_mfma_available /
 *  _narrow_k_mfma_available / _NARROW_TILED_2D_ARCHES ({"gfx942"}) /
 *  validate_tiled_attention_arch. Returns true/false + writes the reason.
 * ===================================================================== */

static bool rocke_g950_wide_k_available(const rocke_arch_target_t* target)
{
    if(rocke_mma_catalog_has_shape(&target->mma, "mma", "f16", "f16", "fp32", 16, 16, 32))
        return true;
    return target->memory.has_ds_read_tr;
}

static bool rocke_g950_narrow_k_available(const rocke_arch_target_t* target)
{
    return rocke_mma_catalog_has_shape(&target->mma, "mma", "f16", "f16", "fp32", 16, 16, 16)
           && rocke_mma_catalog_has_shape(&target->mma, "mma", "bf16", "bf16", "fp32", 16, 16, 16);
}

static bool rocke_g950_arch_in_narrow_set(const char* arch)
{
    return rocke_g950_streq(arch, "gfx942");
}

/* (ok, reason) form (writes into a caller buffer); reason may be NULL. */
static bool
    rocke_g950_validate_tiled_attention_arch(const char* arch, char* reason, size_t reason_cap)
{
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        /* Python: except KeyError as e: return False, str(e). */
        char buf[128];
        snprintf(buf, sizeof(buf), "%s", arch ? arch : "None");
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    if(rocke_g950_wide_k_available(target) && target->memory.has_ds_read_tr)
    {
        rocke_g950_reason(reason, reason_cap, "ok");
        return true;
    }
    if(rocke_g950_arch_in_narrow_set(arch) && rocke_g950_narrow_k_available(target))
    {
        rocke_g950_reason(reason, reason_cap, "ok");
        return true;
    }
    if(!rocke_g950_wide_k_available(target))
    {
        char buf[320];
        snprintf(buf,
                 sizeof(buf),
                 "tiled attention requires either the wide-K MFMA atoms "
                 "(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950) or, on a narrow "
                 "variant arch (gfx942), the 16x16x16 f16/bf16 atom; neither path is "
                 "available on %s",
                 arch ? arch : "None");
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    {
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "tiled attention requires LDS transpose reads (ds_read_b64_tr_b16) "
                 "for the wide-K path, absent on %s",
                 arch ? arch : "None");
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
}

/* require_tiled_attention_arch: latches a NotImplementedError on the builder
 * (Python raise) on reject; true on accept. */
static bool rocke_g950_require_tiled_attention_arch(rocke_ir_builder_t* b, const char* arch)
{
    char reason[320];
    if(rocke_g950_validate_tiled_attention_arch(arch, reason, sizeof(reason)))
        return true;
    /* raise NotImplementedError(reason) -- caught at the public entry boundary. */
    rocke_g950_fail(b, ROCKE_ERR_NOTIMPL, reason);
}

/* ===================================================================== *
 *  supports_tiled_2d(..., arch="gfx950")   (Python lines 588-703)
 * ===================================================================== */

rocke_gfx950_attention_tiled_2d_supports_args_t
    rocke_gfx950_attention_tiled_2d_supports_args_default(void)
{
    rocke_gfx950_attention_tiled_2d_supports_args_t a;
    memset(&a, 0, sizeof(a));
    a.num_warps = 1;
    a.block_m_per_warp = 16;
    /* head_size/block_size/dtype/num_queries_per_kv set by the caller; every
     * other field defaults to 0/false/NULL == the Python keyword default
     * (q_dtype=None, kv_storage_dtype=None, tile_size=None, ...). */
    return a;
}

bool rocke_gfx950_attention_tiled_2d_supports(
    const rocke_gfx950_attention_tiled_2d_supports_args_t* args, char* reason, size_t reason_cap)
{
    const char* arch;
    int block_m;
    char buf[320];

    if(args == NULL)
    {
        rocke_g950_reason(reason, reason_cap, "null args");
        return false;
    }
    arch = args->arch ? args->arch : "gfx950";

    /* arch_ok, arch_reason = validate_tiled_attention_arch(arch) (Py622). */
    if(!rocke_g950_validate_tiled_attention_arch(arch, reason, reason_cap))
        return false;

    /* dtype gate (Py625). */
    if(!(args->dtype != NULL
         && (strcmp(args->dtype, "fp16") == 0 || strcmp(args->dtype, "bf16") == 0)))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel currently supports fp16/bf16 (got %s%s%s)",
                 args->dtype ? "'" : "",
                 args->dtype ? args->dtype : "None",
                 args->dtype ? "'" : "");
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* head_size in {64,128,256} (Py627). */
    if(!(args->head_size == 64 || args->head_size == 128 || args->head_size == 256))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel only supports head_size in {64,128,256} (got %d)",
                 args->head_size);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* head_size % 32 == 0 (Py632). */
    if(args->head_size % 32 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel requires head_size divisible by 32 (got %d)",
                 args->head_size);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* block_size in {16,32,64} (Py637). */
    if(!(args->block_size == 16 || args->block_size == 32 || args->block_size == 64))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel only supports block_size in {16,32,64} (got %d)",
                 args->block_size);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* 1 <= num_queries_per_kv <= 16 (Py642). */
    if(args->num_queries_per_kv > 16 || args->num_queries_per_kv < 1)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel needs 1<=num_queries_per_kv<=16 (got %d)",
                 args->num_queries_per_kv);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* num_queries_per_kv | BLOCK_M (Py647). */
    block_m = 16 * args->num_warps;
    if(block_m % args->num_queries_per_kv != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel needs num_queries_per_kv to divide BLOCK_M=%d "
                 "(num_warps=%d, got num_queries_per_kv=%d)",
                 block_m,
                 args->num_warps,
                 args->num_queries_per_kv);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* fp8 K/V cache pairing (Py657-666). */
    if(args->kv_storage_dtype != NULL && strcmp(args->kv_storage_dtype, "fp8e4m3") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel: unsupported kv_storage_dtype '%s'",
                 args->kv_storage_dtype);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    if(args->use_fp8 && args->kv_storage_dtype == NULL)
    {
        rocke_g950_reason(reason,
                          reason_cap,
                          "tiled 2D kernel: use_fp8=True requires kv_storage_dtype='fp8e4m3'");
        return false;
    }
    /* q_dtype check (Py667). */
    if(args->q_dtype != NULL && strcmp(args->q_dtype, "fp16") != 0
       && strcmp(args->q_dtype, "bf16") != 0)
    {
        snprintf(buf, sizeof(buf), "tiled 2D kernel: unsupported q_dtype '%s'", args->q_dtype);
        rocke_g950_reason(reason, reason_cap, buf);
        return false;
    }
    /* tile_size constraints (Py669-702). */
    if(args->has_tile_size)
    {
        int threads;
        int per_wave_tokens;
        if(args->tile_size <= 0 || args->tile_size % args->block_size != 0)
        {
            snprintf(buf,
                     sizeof(buf),
                     "tiled 2D kernel: tile_size=%d must be a positive multiple of "
                     "block_size=%d",
                     args->tile_size,
                     args->block_size);
            rocke_g950_reason(reason, reason_cap, buf);
            return false;
        }
        threads = args->num_warps * 64;
        if(args->tile_size * args->head_size < threads * 8)
        {
            snprintf(buf,
                     sizeof(buf),
                     "tiled 2D kernel: tile_size*head_size=%d too small for num_warps=%d "
                     "(need >= %d)",
                     args->tile_size * args->head_size,
                     args->num_warps,
                     threads * 8);
            rocke_g950_reason(reason, reason_cap, buf);
            return false;
        }
        per_wave_tokens = (64 * 8) / args->head_size;
        if(per_wave_tokens > args->block_size)
        {
            snprintf(buf,
                     sizeof(buf),
                     "tiled 2D kernel: per-wave tokens %d exceeds block_size=%d; would "
                     "need lane-divergent block lookup",
                     per_wave_tokens,
                     args->block_size);
            rocke_g950_reason(reason, reason_cap, buf);
            return false;
        }
    }

    /* The trailing use_mfma_32x32x8 / use_transposed_qk_32x32 / use_k_single_buffer
     * / use_conflict_free_v_store / use_k_sliced_ring args are accepted for
     * signature parity but do NOT key gfx950 admission (Py609-616). */
    rocke_g950_reason(reason, reason_cap, "supported");
    return true;
}

/* ===================================================================== *
 *  rocke_gfx950_attn2d_build_ctx_init -- the prologue (Python 711-1151)
 *
 *  INTEGRATION NOTE. The header-declared external definition of this symbol
 *  lives in the dedicated prologue TU (ctx_prologue.c). This driver TU carries
 *  its own self-consistent copy (it uses the file-local rocke_g950_* helpers and
 *  the gfx950-specific "leave ctx->target / ctx->qk_atom NULL" derivation that
 *  the phase functions this driver calls expect). To avoid a duplicate external
 *  symbol at link time, this copy is file-local (static); only this driver calls
 *  it. The cross-TU external symbol is the prologue TU's definition.
 * ===================================================================== */
static bool rocke_g950_build_ctx_init_local(rocke_gfx950_attn2d_build_ctx_t* ctx,
                                            rocke_ir_builder_t* b,
                                            const rocke_attention_tiled_2d_spec_t* spec,
                                            const char* arch)
{
    if(ctx == NULL || b == NULL || spec == NULL)
        return false;
    memset(ctx, 0, sizeof(*ctx));

    if(arch == NULL)
        arch = "gfx950"; /* Python default */

    ctx->b = b;
    ctx->spec = spec;
    ctx->arch = arch;

    /* ---- require_tiled_attention_arch(arch) (Py748) ---- */
    if(!rocke_g950_require_tiled_attention_arch(b, arch))
        return false;

    /* ---- dtype gate (Py750) ---- */
    if(!(rocke_g950_streq(spec->dtype, "fp16") || rocke_g950_streq(spec->dtype, "bf16")))
    {
        rocke_g950_fail(b, ROCKE_ERR_NOTIMPL, "tiled 2D kernel supports fp16/bf16");
    }
    const rocke_type_t* dtype = rocke_attention_tiled_2d_spec_dtype_ir(spec);
    ctx->dtype = dtype;

    /* ---- use_register_pv constraints (Python __post_init__ lines 470-485).
     * Python enforces these at dataclass construction; mirror them here so an
     * invalid register_pv spec is cleanly rejected (build fails) instead of
     * crashing during emission. */
    if(spec->use_register_pv)
    {
        if(spec->use_mfma_32x32)
        {
            rocke_g950_fail(b,
                            ROCKE_ERR_VALUE,
                            "use_register_pv currently targets the existing 16x16x32 path; "
                            "the 32x32 path has a separate register-P migration");
        }
        if(!rocke_g950_streq(spec->dtype, "bf16"))
        {
            rocke_g950_fail(b, ROCKE_ERR_VALUE, "use_register_pv v1 is restricted to dtype='bf16'");
        }
        if(spec->kv_storage_dtype != NULL)
        {
            rocke_g950_fail(
                b, ROCKE_ERR_VALUE, "use_register_pv v1 does not support fp8 K/V cache");
        }
        if(spec->use_sinks || spec->sliding_window > 0 || spec->has_softcap)
        {
            rocke_g950_fail(b,
                            ROCKE_ERR_VALUE,
                            "use_register_pv v1 requires no sinks, no sliding window, "
                            "and no softcap");
        }
        if(spec->use_alibi || spec->use_qq_bias)
        {
            rocke_g950_fail(
                b, ROCKE_ERR_VALUE, "use_register_pv v1 does not support ALiBi or QQ bias");
        }
    }

    /* ---- V-double-buffer / deep-ring / staggered-wait / q-reread.
     * Python __post_init__ accepts these flags (and the emitter wires the
     * depth-2 V-double-buffer + staggered iter-start schedule). The gfx950 C
     * twin currently emits ONLY the depth-2 single/early-V schedule, so reject
     * any spec that asks for the V-double-buffer family (and kv_ring_depth!=2 /
     * q_reread) rather than silently emit the wrong schedule. This keeps the
     * spec struct byte-identical with Python (provider passthrough) while making
     * the C emitter's narrower coverage explicit. When these schedules are
     * ported to C, replace these guards with the real emit + Python's exact
     * __post_init__ messages. */
    if(spec->use_v_double_buffer)
    {
        rocke_g950_fail(b,
                        ROCKE_ERR_NOTIMPL,
                        "use_v_double_buffer: gfx950 C twin emits only the depth-2 "
                        "single/early-V schedule (not yet ported)");
    }
    if(spec->use_staggered_iter_wait)
    {
        rocke_g950_fail(b,
                        ROCKE_ERR_NOTIMPL,
                        "use_staggered_iter_wait: gfx950 C twin emits only the depth-2 "
                        "single/early-V schedule (not yet ported)");
    }
    if(spec->kv_ring_depth != 2)
    {
        /* the Python builder wires a depth-3 deep K prefetch ring on the
         * d128 small-tile combo, but the gfx950 C twin shares the depth-2
         * single/early-V emitter and does not port the 3-slot ring + staggered
         * waits. Reject cleanly rather than silently emit a depth-2 schedule. */
        rocke_g950_fail(b,
                        ROCKE_ERR_NOTIMPL,
                        "kv_ring_depth!=2 (deep prefetch ring) not yet ported to "
                        "the gfx950 C twin");
    }
    if(spec->use_q_reread)
    {
        rocke_g950_fail(b,
                        ROCKE_ERR_NOTIMPL,
                        "use_q_reread: gfx950 C twin does not yet port the Q-reread path");
    }
    if(spec->use_q_direct_reg)
    {
        /* the direct-to-register Q gather (Triton-style, frees Q_lds) is
         * carried in the spec for byte-identity with the Python dataclass but
         * the gfx950 C twin does not yet emit this body. Reject (do not
         * silently ignore), matching the Python __post_init__ contract. */
        rocke_g950_fail(
            b,
            ROCKE_ERR_NOTIMPL,
            "use_q_direct_reg: gfx950 C twin does not yet port the direct-register Q gather");
    }
    if(spec->use_softmax_mfma_interleave)
    {
        /* the softmax-window MFMA interleave (iglp_opt / sched_group_barrier
         * at the loop top) rides on the use_q_direct_reg BLOCK_M=128 body, which
         * the gfx950 C twin does not yet emit. Carried in the spec for
         * byte-identity with the Python dataclass; reject (do not silently
         * ignore), matching the Python __post_init__ contract. */
        rocke_g950_fail(
            b,
            ROCKE_ERR_NOTIMPL,
            "use_softmax_mfma_interleave: gfx950 C twin does not yet port the "
            "softmax-window MFMA interleave (its host body use_q_direct_reg is unported)");
    }
    /* K single-buffer IS ported in the gfx950 C twin (it shares the
     * depth-2 V-single `else` schedule). Mirror Python __post_init__ guards so
     * incompatible combos reject with matching reasons rather than mis-emit. */
    if(spec->use_k_single_buffer)
    {
        if(spec->use_v_double_buffer)
            rocke_g950_fail(b,
                            ROCKE_ERR_NOTIMPL,
                            "use_k_single_buffer requires V single-buffer (it shares the "
                            "post-PV K re-issue point; V double-buffer uses a different "
                            "K/V prefetch schedule)");
        if(spec->kv_ring_depth != 2)
            rocke_g950_fail(b,
                            ROCKE_ERR_NOTIMPL,
                            "use_k_single_buffer is incompatible with kv_ring_depth!=2 "
                            "(the ring owns >2 K slots)");
        if(spec->use_grouped_kv2_softmax)
            rocke_g950_fail(b,
                            ROCKE_ERR_NOTIMPL,
                            "use_k_single_buffer does not support grouped_kv2 (needs 2 K "
                            "tiles resident for the dual-tile QK)");
        if(spec->use_early_v_schedule)
            rocke_g950_fail(
                b, ROCKE_ERR_NOTIMPL, "use_k_single_buffer does not support early_v_schedule");
        if(spec->use_staggered_iter_wait)
            rocke_g950_fail(
                b, ROCKE_ERR_NOTIMPL, "use_k_single_buffer does not support staggered_iter_wait");
    }

    /* gfx950 does NOT pre-resolve a catalog atom or ArchTarget in the prologue
     * (the kv_body calls mfma_32x32x16_for_dtype / mfma_16x16x32_for_dtype
     * directly). Leave ctx->target / ctx->qk_atom NULL, matching Python. */

    /* ---- ALL-CAPS geometry constants (Py754-767) ---- */
    const int HD = spec->head_size;
    const int T = rocke_attention_tiled_2d_spec_tile_size_eff(spec);
    const int BS = spec->block_size;
    const int N_BLOCKS_PER_TILE = rocke_attention_tiled_2d_spec_n_blocks_per_tile(spec);
    const int BLOCK_M = rocke_attention_tiled_2d_spec_block_m(spec);
    const int BLOCK_Q = rocke_attention_tiled_2d_spec_block_q(spec);
    const int NQK = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    const int NUM_KV = spec->num_kv_heads;
    const int NUM_QH = spec->num_query_heads;
    const int SLIDING_WINDOW = spec->sliding_window;

    ctx->HD = HD;
    ctx->T = T;
    ctx->BS = BS;
    ctx->N_BLOCKS_PER_TILE = N_BLOCKS_PER_TILE;
    ctx->BLOCK_M = BLOCK_M;
    ctx->BLOCK_Q = BLOCK_Q;
    ctx->NQK = NQK;
    ctx->NUM_KV = NUM_KV;
    ctx->NUM_QH = NUM_QH;
    ctx->SLIDING_WINDOW = SLIDING_WINDOW;
    ctx->USE_SOFTCAP = spec->has_softcap;
    ctx->USE_SINKS = spec->use_sinks;
    ctx->USE_ALIBI = spec->use_alibi;
    ctx->USE_QQ_BIAS = spec->use_qq_bias;

    /* transposed-softmax + experimental predicate aliases (Py768-777). */
    ctx->TRANSPOSED_SCALAR_STATE = spec->use_transposed_scalar_state;
    ctx->TRANSPOSED_INVARIANT_HOIST = spec->use_transposed_invariant_hoist;
    ctx->TRANSPOSED_MASK_ONCE = spec->use_transposed_mask_once;
    ctx->TRANSPOSED_HALF_LOCAL_PV = spec->use_transposed_half_local_pv;
    ctx->SKIP_LEGACY_QREG = spec->use_mfma32_skip_legacy_qreg;
    ctx->TRANSPOSED_MASK_LIMIT = spec->use_transposed_mask_limit;
    ctx->GROUPED_KV2 = spec->use_grouped_kv2_softmax;
    ctx->FAST_PAGED_KV_DESC = spec->use_fast_paged_kv_desc;
    ctx->I64_KV_ADDR = spec->use_i64_kv_addr;
    ctx->EARLY_V_SCHEDULE = spec->use_early_v_schedule;
    ctx->AGPR_ALLOC_ZERO = spec->use_agpr_alloc_zero;
    ctx->K_SINGLE_BUFFER = spec->use_k_single_buffer; /* K single-buffer */
    ctx->USE_SCHED_BARRIER = spec->use_sched_barrier;
    ctx->SCHED_BARRIER_MASK = spec->sched_barrier_mask;

    /* ---- fp8 K/V cache predicates (Py783-797) ---- */
    const bool KV_FP8 = rocke_g950_streq(spec->kv_storage_dtype, "fp8e4m3");
    const bool FP8_MFMA_QK = KV_FP8 && spec->use_fp8_mfma_qk;
    const bool FP8_MFMA_PV = KV_FP8 && spec->use_fp8_mfma_pv;
    /* FP8_NATIVE_QK is the documented dead path (always False, Py793). */
    const bool FP8_NATIVE_QK = false;
    ctx->KV_FP8 = KV_FP8;
    ctx->FP8_MFMA_QK = FP8_MFMA_QK;
    ctx->FP8_MFMA_PV = FP8_MFMA_PV;
    ctx->REGISTER_PV = spec->use_register_pv;
    ctx->TRANSPOSED_QK_32X32 = spec->use_transposed_qk_32x32;

    const int KV_BYTES = KV_FP8 ? 1 : 2;
    ctx->KV_BYTES = KV_BYTES;
    ctx->kv_io_dtype = KV_FP8 ? rocke_fp8e4m3() : dtype;
    /* kv_cache_aux: {"all":ALL,"global":GLOBAL,"stream":STREAM,"nt":NT}. The
     * gfx950 kernel uses the spec's kv_cache_policy for the cache hint. */
    if(rocke_g950_streq(spec->kv_cache_policy, "all"))
        ctx->kv_cache_aux = (int)ROCKE_CACHE_ALL;
    else if(rocke_g950_streq(spec->kv_cache_policy, "global"))
        ctx->kv_cache_aux = (int)ROCKE_CACHE_GLOBAL;
    else if(rocke_g950_streq(spec->kv_cache_policy, "stream"))
        ctx->kv_cache_aux = (int)ROCKE_CACHE_STREAM;
    else
        ctx->kv_cache_aux = (int)ROCKE_NON_TEMPORAL;

    /* ---- 32x32 umbrella + QK/PV MFMA geometry (Py799-819) ---- */
    const bool USE_MFMA_32X32 = spec->use_mfma_32x32;
    ctx->USE_MFMA_32X32 = USE_MFMA_32X32;

    const int MFMA_N = ROCKE_ATTN_TILED_2D_MFMA_N; /* 16 */
    const int QK_MFMA_N = USE_MFMA_32X32 ? 32 : MFMA_N;
    const int QK_K_STEP = USE_MFMA_32X32 ? 16 : 32;
    const int PV_K_STEP = (T % 32 == 0) ? 32 : 16;
    const int QK_K_ITERS = HD / QK_K_STEP;
    const int QK_N_TILES = T / QK_MFMA_N;
    const int PV_K_ITERS = T / PV_K_STEP;
    const int PV_N_TILES = HD / MFMA_N;
    ctx->QK_MFMA_N = QK_MFMA_N;
    ctx->QK_K_STEP = QK_K_STEP;
    ctx->PV_K_STEP = PV_K_STEP;
    ctx->QK_K_ITERS = QK_K_ITERS;
    ctx->QK_N_TILES = QK_N_TILES;
    ctx->PV_K_ITERS = PV_K_ITERS;
    ctx->PV_N_TILES = PV_N_TILES;

    /* ---- threads / wave / async-DMA width (Py821-839) ---- */
    const int NUM_WARPS = spec->num_warps;
    const int WAVE = 64;
    const int THREADS = NUM_WARPS * WAVE;
    ctx->NUM_WARPS = NUM_WARPS;
    ctx->WAVE = WAVE;
    ctx->THREADS = THREADS;

    /* gfx950 async-DMA (raw.ptr.buffer.load.lds) supports dwords in {1,3,4}
     * (Python line 1426); the fp8 loader picks the widest of [(4,16),(3,12),
     * (1,4)] the tile bytes allow, so the clamp is the 4-dword max. */
    ctx->ASYNC_LDS_MAX_DWORDS = 4;
    ctx->ASYNC_LDS_MAX_BYTES_PER_LANE = 16;

    const int BLOCK_M_PER_WARP = spec->block_m_per_warp;
    const int M_ATOMS_PER_WARP = BLOCK_M_PER_WARP / ROCKE_ATTN_TILED_2D_MFMA_M; /* /16 */
    const int REGS_PER_LANE = rocke_attention_tiled_2d_spec_regs_per_lane(spec);
    const int SOFTMAX_STATE_SLOTS
        = (USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32 && ctx->TRANSPOSED_SCALAR_STATE)
              ? 1
              : REGS_PER_LANE;
    ctx->BLOCK_M_PER_WARP = BLOCK_M_PER_WARP;
    ctx->M_ATOMS_PER_WARP = M_ATOMS_PER_WARP;
    ctx->REGS_PER_LANE = REGS_PER_LANE;
    ctx->SOFTMAX_STATE_SLOTS = SOFTMAX_STATE_SLOTS;

    /* ---- kernel name + attrs (Py841-847) ---- *
     * The driver inited b with spec.kernel_name(); mirror that name. */
    ctx->name = (b->kernel != NULL) ? b->kernel->name : NULL;
    if(b->kernel != NULL)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
        if(spec->has_waves_per_eu)
            rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
        if(spec->use_agpr_alloc_zero)
        {
            /* (0, 0) -- serialized as the bare-int list l:[ i:0, i:0 ] */
            const int64_t agpr_zero[2] = {0, 0};
            rocke_attr_set_int_list(b, &b->kernel->attrs, "agpr_alloc", agpr_zero, 2);
        }
    }

    /* ---- parameter declarations (Py850-889) ---- */
    {
        rocke_param_opts_t o;
        const rocke_type_t* ptr_dt = rocke_ptr_type(b, dtype, "global");
        const rocke_type_t* ptr_kvio = rocke_ptr_type(b, ctx->kv_io_dtype, "global");
        const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
        const rocke_type_t* ptr_f32 = rocke_ptr_type(b, rocke_f32(), "global");

        memset(&o, 0, sizeof(o));
        o.noalias = true;
        o.noalias_set = true;
        o.writeonly = true;
        o.writeonly_set = true;
        o.align = 16;
        o.align_set = true;
        ctx->output = rocke_b_param(b, "output_ptr", ptr_dt, &o);

        memset(&o, 0, sizeof(o));
        o.noalias = true;
        o.noalias_set = true;
        o.readonly = true;
        o.readonly_set = true;
        o.align = 16;
        o.align_set = true;
        ctx->query = rocke_b_param(b, "query_ptr", ptr_dt, &o);
        ctx->key = rocke_b_param(b, "key_cache_ptr", ptr_kvio, &o);
        ctx->value = rocke_b_param(b, "value_cache_ptr", ptr_kvio, &o);

        memset(&o, 0, sizeof(o));
        o.readonly = true;
        o.readonly_set = true;
        o.align = 16;
        o.align_set = true;
        ctx->sinks = rocke_b_param(b, "sink_ptr", ptr_dt, &o);

        memset(&o, 0, sizeof(o));
        o.readonly = true;
        o.readonly_set = true;
        o.align = 4;
        o.align_set = true;
        ctx->block_tables = rocke_b_param(b, "block_tables_ptr", ptr_i32, &o);
        ctx->seq_lens = rocke_b_param(b, "seq_lens_ptr", ptr_i32, &o);
        ctx->alibi_slopes_ptr = rocke_b_param(b, "alibi_slopes_ptr", ptr_f32, &o);
        ctx->qq_bias_ptr = rocke_b_param(b, "qq_bias_ptr", ptr_f32, &o);
        ctx->cu_q = rocke_b_param(b, "query_start_len_ptr", ptr_i32, &o);

        ctx->scale_p = rocke_b_param(b, "scale", rocke_f32(), NULL);
        ctx->k_scale_p = rocke_b_param(b, "k_scale", rocke_f32(), NULL);
        ctx->v_scale_p = rocke_b_param(b, "v_scale", rocke_f32(), NULL);
        ctx->out_scale = rocke_b_param(b, "out_scale", rocke_f32(), NULL);
        ctx->softcap_p = rocke_b_param(b, "softcap", rocke_f32(), NULL);
        ctx->num_seqs_p = rocke_b_param(b, "num_seqs", rocke_i32(), NULL);
        ctx->bt_stride_p = rocke_b_param(b, "block_table_stride", rocke_i32(), NULL);
        ctx->qq_bias_stride0_p = rocke_b_param(b, "qq_bias_stride_0", rocke_i32(), NULL);
    }

    /* ---- grid ids + wave decomposition (Py891-904) ---- */
    ctx->kv_head_idx = rocke_b_block_id_x(b);
    ctx->q_block_global_idx = rocke_b_block_id_y(b);
    ctx->tid = rocke_b_thread_id_x(b);

    if(NUM_WARPS == 1)
    {
        ctx->lane = ctx->tid;
        ctx->wave_id = NULL;
        ctx->wave_row_base = rocke_b_const_i32(b, 0);
    }
    else
    {
        ctx->lane = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, WAVE));
        ctx->wave_id = rocke_b_div(b, ctx->tid, rocke_b_const_i32(b, WAVE));
        ctx->wave_row_base = rocke_b_mul(b, ctx->wave_id, rocke_b_const_i32(b, BLOCK_M_PER_WARP));
    }

    /* ---- seq lookup + Q-block geometry (Py906-925) ---- */
    ctx->seq_idx
        = rocke_binary_search_seq_idx(b,
                                      ctx->cu_q,
                                      ctx->q_block_global_idx,
                                      ctx->num_seqs_p,
                                      BLOCK_Q,
                                      rocke_attention_tiled_2d_spec_binary_search_iters(spec),
                                      false);
    ctx->cu_q_start = rocke_b_global_load_i32(b, ctx->cu_q, ctx->seq_idx, 0);
    ctx->cu_q_stop = rocke_b_global_load_i32(
        b, ctx->cu_q, rocke_b_add(b, ctx->seq_idx, rocke_b_const_i32(b, 1)), 0);
    ctx->cur_batch_q_len = rocke_b_sub(b, ctx->cu_q_stop, ctx->cu_q_start);
    ctx->q_block_start_idx = rocke_b_add(
        b, rocke_b_div(b, ctx->cu_q_start, rocke_b_const_i32(b, BLOCK_Q)), ctx->seq_idx);
    ctx->q_block_local_idx = rocke_b_sub(b, ctx->q_block_global_idx, ctx->q_block_start_idx);
    ctx->seq_len = rocke_b_global_load_i32(b, ctx->seq_lens, ctx->seq_idx, 0);
    ctx->context_len = rocke_b_sub(b, ctx->seq_len, ctx->cur_batch_q_len);

    ctx->qb_start_pos = rocke_b_mul(b, ctx->q_block_local_idx, rocke_b_const_i32(b, BLOCK_Q));
    {
        rocke_if_t f
            = rocke_b_scf_if(b, rocke_b_cmp_ge(b, ctx->qb_start_pos, ctx->cur_batch_q_len));
        rocke_b_region_enter(b, f.then_region);
        rocke_b_ret(b);
        rocke_b_region_leave(b);
    }

    /* ---- Acc_lds stripe geometry (Py952-959) ---- */
    const int OUT_STRIPE_COLS = (HD <= 64) ? 32 : HD;
    const int OUT_STRIPES = HD / OUT_STRIPE_COLS;
    ctx->OUT_STRIPE_COLS = OUT_STRIPE_COLS;
    ctx->OUT_STRIPES = OUT_STRIPES;

    /* ---- LDS dtypes / buffering / sizes (Py1021-1037) ---- */
    const rocke_type_t* K_LDS_DTYPE = FP8_MFMA_QK ? rocke_fp8e4m3() : dtype;
    const rocke_type_t* V_LDS_DTYPE = FP8_MFMA_PV ? rocke_fp8e4m3() : dtype;
    const rocke_type_t* P_LDS_DTYPE = FP8_MFMA_PV ? rocke_fp8e4m3() : dtype;
    ctx->K_LDS_DTYPE = K_LDS_DTYPE;
    ctx->V_LDS_DTYPE = V_LDS_DTYPE;
    ctx->P_LDS_DTYPE = P_LDS_DTYPE;

    const int Q_BYTES = BLOCK_M * HD * 2;
    const int K_LDS_ELEM_BYTES = rocke_type_eq(K_LDS_DTYPE, rocke_fp8e4m3()) ? 1 : 2;
    const int K_BUF_BYTES = T * HD * K_LDS_ELEM_BYTES;
    const int K_TOTAL_BYTES = 2 * K_BUF_BYTES; /* K_lds Q-alias region (2 slots worth) */
    ctx->Q_BYTES = Q_BYTES;
    ctx->K_LDS_ELEM_BYTES = K_LDS_ELEM_BYTES;
    ctx->K_BUF_BYTES = K_BUF_BYTES;
    /* K single-buffer -> 1 slot (halves K_lds so T=64 fits 2 WG/CU). */
    ctx->K_BUFS = ctx->K_SINGLE_BUFFER ? 1 : 2;
    ctx->K_TOTAL_BYTES = K_TOTAL_BYTES;

    const bool Q_ALIAS_K = rocke_type_eq(K_LDS_DTYPE, dtype) && (Q_BYTES <= K_TOTAL_BYTES);
    const bool Q_USES_DUAL_SLOT = Q_ALIAS_K && (BLOCK_M > T);
    ctx->Q_DIRECT_GLOBAL = false; /* gfx950 has no Q_DIRECT_GLOBAL path */
    ctx->Q_ALIAS_K = Q_ALIAS_K;
    ctx->Q_USES_DUAL_SLOT = Q_USES_DUAL_SLOT;

    /* ---- K_lds smem_alloc (Py1038) ---- */
    {
        int shp[3] = {ctx->K_BUFS, T, HD}; /* 1 slot when K single-buffer */
        ctx->K_lds = rocke_b_smem_alloc(b, K_LDS_DTYPE, shp, 3, "Klds");
    }

    const int V_BUFS = 1; /* single-buffer V (race-free, Py1039) */
    ctx->V_BUFS = V_BUFS;

    /* ---- V_lds smem_alloc (Py1040-1060) ---- */
    if(FP8_MFMA_PV)
    {
        const int N_STRIPES = HD / 16;
        ctx->N_STRIPES = N_STRIPES;
        int shp[4] = {V_BUFS, N_STRIPES, T, 16};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 4, "VldsStripe");
    }
    else
    {
        int shp[3] = {V_BUFS, T, HD};
        ctx->V_lds = rocke_b_smem_alloc(b, V_LDS_DTYPE, shp, 3, "Vlds");
    }

    /* ---- P_lds (Py1061-1076) ----
     * The transposed-32x32 path keeps P entirely in registers (softmax publish
     * is a no-op, PV reads PT32_n from registers), so P_lds was dead LDS there.
     * Drop the allocation to free the occupancy-limiting LDS at HD=128.
     * Mirrors attention_tiled_2d.py P_LDS_DEAD and the ctx_prologue twin. */
    const bool P_LDS_DEAD = USE_MFMA_32X32 && ctx->TRANSPOSED_QK_32X32;
    if(!ctx->REGISTER_PV && !P_LDS_DEAD)
    {
        const int P_LDS_PAD = FP8_MFMA_PV ? 16 : 8;
        ctx->P_LDS_PAD = P_LDS_PAD;
        int shp[2] = {BLOCK_M, T + P_LDS_PAD};
        ctx->P_lds = rocke_b_smem_alloc(b, P_LDS_DTYPE, shp, 2, "Plds");
    }
    else
    {
        ctx->P_lds = NULL;
    }

    /* ---- fp8 K/V staging slabs (Py1095-1097) ---- */
    if(KV_FP8 && spec->use_fp8_mfma_qk)
    {
        int kshp[3] = {2, T, HD};
        int vshp[3] = {1, T, HD};
        ctx->K_fp8_lds = rocke_b_smem_alloc(b, rocke_fp8e4m3(), kshp, 3, "Kfp8lds");
        ctx->V_fp8_lds = rocke_b_smem_alloc(b, rocke_fp8e4m3(), vshp, 3, "Vfp8lds");
    }

    /* ---- Q_lds (Py1098-1104) ---- */
    if(Q_ALIAS_K)
    {
        ctx->Q_lds = ctx->K_lds;
    }
    else
    {
        int shp[2] = {BLOCK_M, HD};
        ctx->Q_lds = rocke_b_smem_alloc(b, dtype, shp, 2, "Qlds");
    }

    /* ---- Acc_lds (Py1113) ---- */
    {
        int shp[2] = {BLOCK_M, OUT_STRIPE_COLS};
        ctx->Acc_lds = rocke_b_smem_alloc(b, dtype, shp, 2, "Aclds");
    }

    /* ---- PV transpose-LDS reader bind (Py1128) ---- *
     * TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, lane). Cache the unbound
     * descriptor and the bound SSA view for every PV atom. */
    ctx->pv_tr_reader_desc = rocke_transpose_lds_reader_make(PV_K_STEP);
    ctx->pv_tr_reader = rocke_transpose_lds_reader_bind(b, &ctx->pv_tr_reader_desc, ctx->lane);

    /* ---- SSA constants (Py1131-1150) ---- *
     * Match Python's exact creation order: neg_inf, zero_f, one_f, rcp_ln2,
     * qk_scale, then sw_const. (pv_fp8_scale (Py1149) and z8 (Py1151) are
     * emitted by the consuming phases; they are not ctx-carried per the
     * internal contract.) */
    ctx->neg_inf_v = rocke_b_const_f32(b, -INFINITY); /* neg_inf (1132) */
    ctx->zero_f = rocke_b_const_f32(b, 0.0); /* zero_f  (1133) */
    ctx->one_f_v = rocke_b_const_f32(b, 1.0); /* one_f   (1134) */
    {
        const double rcp_ln2 = 1.4426950408889634;
        ctx->rcp_ln2_v = rocke_b_const_f32(b, rcp_ln2); /* rcp_ln2 (1135) */
        rocke_value_t* qk_scale = rocke_b_fmul(b, ctx->scale_p, ctx->rcp_ln2_v); /* (1136) */
        if(FP8_NATIVE_QK)
            qk_scale = rocke_b_fmul(b, qk_scale, ctx->k_scale_p);
        ctx->qk_scale_v = qk_scale;
    }
    /* pv_fp8_scale = fdiv(v_scale, 240.0) when FP8_MFMA_PV (line 1149). Hoisted
     * here so the SSA id matches Python; the PV epilogue reuses this value. */
    ctx->pv_fp8_scale_v
        = FP8_MFMA_PV ? rocke_b_fdiv(b, ctx->v_scale_p, rocke_b_const_f32(b, 240.0)) : NULL;
    ctx->sw_const_v = rocke_b_const_i32(b, SLIDING_WINDOW); /* sw_const (1150) */
    ctx->c0 = ctx->sw_const_v;

    /* ---- acc geometry the epilogue aliases (Py1381-1382) ---- */
    ctx->ACC_N_TILES = USE_MFMA_32X32 ? (HD / 32) : PV_N_TILES;
    ctx->ACC_M_ATOMS = USE_MFMA_32X32 ? 1 : M_ATOMS_PER_WARP;
    ctx->ml_count = 2 * SOFTMAX_STATE_SLOTS;

    return rocke_ir_builder_ok(b);
}

/* ===================================================================== *
 *  build_unified_attention_2d_tiled(spec, arch="gfx950")  (lines 711-3818)
 *
 *  The driver: populate the shared build ctx via rocke_gfx950_attn2d_build_ctx_init
 *  (the prologue), then run the phase functions in Python execution order and
 *  return b->kernel from the epilogue.
 * ===================================================================== */
rocke_kernel_def_t* rocke_gfx950_build_unified_attention_2d_tiled(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch)
{
    /* Boundary: catch any reject raised by the prologue (rocke_g950_fail ->
     * ckc::raise_status) and record it on the builder as status + message,
     * returning nullptr -- the legacy non-throwing contract for this entry. */
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        rocke_gfx950_attn2d_build_ctx_t ctx;
        rocke_kernel_def_t* kernel;

        if(b == NULL || spec == NULL)
            return NULL;

        /* Re-entrancy: clear every cross-build static this generator caches against
         * the previous build's (now-freed) arena before emitting build N. Without
         * this, the lazily-built _C32_DIST distribution and the REGISTER_PV scratch
         * carry dangling pointers from build N-1 into the new IR (a NULL operand to
         * warp_shuffle_xor on the second call). The generator is invoked once per
         * workspace-query and once per buildPlan, per graph, so this runs many times
         * per process. */
        rocke_attn2d_c32_dist_reset();
        rocke_gfx950_attn2d_reset_softmax_scratch();

        /* Prologue: arch/dtype gate, wide-atom geometry select, config derivation,
         * kernel + params, grid / seq-idx / Q-block geometry, LDS layout +
         * smem_alloc, PV TransposeLdsReader bind, SSA constants. On any reject it
         * sets b's sticky error and returns false. */
        memset(&ctx, 0, sizeof(ctx));
        if(!rocke_g950_build_ctx_init_local(&ctx, b, spec, arch))
            return NULL;

        /* The emission order below mirrors the single linear Python emitter exactly
         * (the byte-identity contract is on the lowered .ll, so the IR op stream must
         * be produced in Python source order):
         *
         *   q_load (1153) -> tile bounds + softmax iter-arg inits (1220-1398) ->
         *   pre-loop K/V buffer descriptors + _issue_k/_issue_v (1402-2249) -> Q VGPR
         *   gather (2250-2382) -> tile-0 K prefetch + cur_buf carry (2383-2400) ->
         *   LICM hoist (2402-2481) -> kv_step (2482) -> scf.for KV loop -> epilogue.
         */

        /* 1. Cooperatively stage Q[BLOCK_M, HD] global -> LDS (Py1153-1215). */
        rocke_gfx950_attn2d_emit_q_load(&ctx);

        /* 2. max_seq_prefix_len -> tile_start/tile_end + lane decomposition +
         *    online-softmax m/l/acc carry inits + named iter_args (Py1220-1398). */
        rocke_gfx950_attn2d_emit_loop_bounds_and_inits(&ctx);

        /* 3. Pre-loop: build K/V buffer descriptors + the paged-KV byte descriptor
         *    DAG + the fp8/bf16 loaders (Py1402-2249). */
        rocke_gfx950_attn2d_emit_preloop(&ctx);

        /* 4. Gather the per-lane Q MFMA A-operand to VGPRs (Py2250-2382). */
        rocke_gfx950_attn2d_emit_q_gather(&ctx);

        /* 5. tile-0 K prefetch into buffer 0 + ("cur_buf", 0) carry append
         *    (Py2383-2400). */
        rocke_gfx950_attn2d_emit_preloop_prefetch(&ctx);

        /* 6. LICM hoist of the per-reg row/pos/head/mask invariants (Py2402-2481). */
        rocke_gfx950_attn2d_emit_licm_hoist(&ctx);

        /* 7. kv_step const (Py2482). */
        rocke_gfx950_attn2d_emit_kv_step(&ctx);

        /* 8. KV-loop: build the scf.for over [tile_start, tile_end) with the named
         *    online-softmax carry and run one full KV-tile body inside it.
         *    drive_kv_loop returns the loop handle whose results are the rewritten
         *    carry the epilogue consumes; mirror them into ctx.out_carry. */
        rocke_for_t kvloop = rocke_gfx950_attn2d_drive_kv_loop(&ctx);
        if(kvloop.op != NULL)
        {
            int i;
            for(i = 0; i < kvloop.op->num_results; ++i)
                ctx.out_carry[i] = kvloop.op->results[i];
            ctx.out_carry_count = kvloop.op->num_results;
        }

        /* 9. Epilogue: drain async copies, read loop results, normalize + store;
         *    returns b->kernel on success (Py3585-3818). */
        kernel = rocke_gfx950_attn2d_emit_epilogue(&ctx);

        return rocke_ir_builder_ok(b) ? kernel : NULL;
    });
}

/* ===================================================================== *
 *  spec.kernel_name()  (Python lines 545-585), needed by the _new wrapper.
 * ===================================================================== */
static rocke_status_t rocke_g950_attn2d_kernel_name(const rocke_attention_tiled_2d_spec_t* s,
                                                    char* out,
                                                    size_t out_cap)
{
    char d_buf[32], b_buf[32], t_buf[32], hkv_buf[64], kv_buf[64];
    char sw_buf[32], w_buf[32], mw_buf[32], schedb_buf[32];
    int nqh, nkv;

    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;

    nqh = s->num_query_heads;
    nkv = s->num_kv_heads;

    snprintf(d_buf, sizeof(d_buf), "d%d", s->head_size);
    snprintf(b_buf, sizeof(b_buf), "b%d", s->block_size);
    if(rocke_attention_tiled_2d_spec_n_blocks_per_tile(s) != 1)
        snprintf(t_buf, sizeof(t_buf), "t%d", rocke_attention_tiled_2d_spec_tile_size_eff(s));
    else
        t_buf[0] = '\0';
    snprintf(hkv_buf, sizeof(hkv_buf), "h%dkv%d", nqh, nkv);
    if(s->kv_storage_dtype)
        snprintf(kv_buf, sizeof(kv_buf), "kv%s", s->kv_storage_dtype);
    else
        kv_buf[0] = '\0';
    if(s->sliding_window > 0)
        snprintf(sw_buf, sizeof(sw_buf), "sw%d", s->sliding_window);
    else
        sw_buf[0] = '\0';
    if(s->num_warps != 1)
        snprintf(w_buf, sizeof(w_buf), "w%d", s->num_warps);
    else
        w_buf[0] = '\0';
    if(s->block_m_per_warp != 16)
        snprintf(mw_buf, sizeof(mw_buf), "mw%d", s->block_m_per_warp);
    else
        mw_buf[0] = '\0';
    if(s->use_sched_barrier)
        snprintf(schedb_buf, sizeof(schedb_buf), "schedb%d", s->sched_barrier_mask);
    else
        schedb_buf[0] = '\0';

    {
        const char* parts[] = {
            d_buf,
            b_buf,
            t_buf,
            hkv_buf,
            s->dtype ? s->dtype : "",
            kv_buf,
            s->use_sinks ? "sinks" : "",
            sw_buf,
            s->has_softcap ? "softcap" : "",
            s->use_alibi ? "alibi" : "",
            s->use_qq_bias ? "qqb" : "",
            w_buf,
            mw_buf,
            s->use_mfma_32x32 ? "mfma32" : "",
            s->use_transposed_qk_32x32 ? "stqk" : "",
            s->use_transposed_scalar_state ? "s1" : "",
            s->use_transposed_mask_once ? "mask1" : "",
            s->use_transposed_invariant_hoist ? "hoist" : "",
            s->use_transposed_half_local_pv ? "hlpv" : "",
            s->use_mfma32_skip_legacy_qreg ? "skipqreg" : "",
            s->use_transposed_mask_limit ? "mlim" : "",
            s->use_grouped_kv2_softmax ? "gkv2" : "",
            s->use_fast_paged_kv_desc ? "fastkvdesc" : "",
            s->use_early_v_schedule ? "earlyv" : "",
            s->use_v_double_buffer ? "vdbuf" : "",
            s->use_k_single_buffer ? "ksb" : "", /* K single-buffer */
            s->use_staggered_iter_wait ? "stgw" : "",
            schedb_buf,
            s->use_q_reread ? "qrr" : "",
            s->use_q_direct_reg ? "qdreg" : "", /* Q direct-reg */
            s->use_agpr_alloc_zero ? "agpr0" : "",
            s->use_fp8_mfma_qk ? "fp8mfma" : "",
            s->use_fp8_mfma_pv ? "fp8pv" : "",
            s->use_register_pv ? "regpv" : "",
        };
        size_t num_parts = sizeof(parts) / sizeof(parts[0]);
        return rocke_kernel_name_join(
            "rocke_uattn2d_tiled", parts, num_parts, NULL, NULL, 0, out, out_cap, NULL);
    }
}

/* ===================================================================== *
 *  _new convenience: init builder via spec.kernel_name(), then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_gfx950_build_unified_attention_2d_tiled_new(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch)
{
    char kname[1024];
    if(b == NULL || spec == NULL)
        return NULL;
    if(rocke_g950_attn2d_kernel_name(spec, kname, sizeof(kname)) != ROCKE_OK)
        return NULL;
    if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
        return NULL;
    return rocke_gfx950_build_unified_attention_2d_tiled(b, spec, arch);
}

/* ===================================================================== *
 *  build -> lower convenience.
 * ===================================================================== */
rocke_status_t
    rocke_gfx950_attention_tiled_2d_lower_to_llvm(const rocke_attention_tiled_2d_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        rocke_g950_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    kernel = rocke_gfx950_build_unified_attention_2d_tiled_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        rocke_g950_set_err(err, err_cap, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, arch ? arch : "gfx950", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
