// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx942_attention_tiled_2d_gfx942_attention_tiled_2d_public_entry_glue.c
 * -- PUBLIC ENTRY / GLUE bucket of the chunked C99 port of the gfx942/CDNA3
 * narrow-atom tiled-2D unified-attention kernel builder
 * (rocke/instances/gfx942/attention_tiled_2d.py).
 *
 * SCOPE (this translation unit):
 *   - the build driver
 *       rocke_build_unified_attention_2d_tiled_scalar
 *     which zero-inits the shared build ctx, calls
 *     rocke_gfx942_attn2d_build_ctx_init (the prologue port: arch gate -> dtype
 *     gate -> narrow-atom select -> config -> params -> grid/seq geometry -> LDS
 *     layout + smem_alloc -> SSA constants), then drives the phase functions in
 *     the exact Python execution order:
 *         emit_q_load -> emit_q_gather -> emit_licm_hoist -> emit_preloop ->
 *         the online-softmax scf.for over KV tiles (emit_kv_body per iter) ->
 *         emit_epilogue (which returns b->kernel).
 *   - the _new init-from-spec convenience wrapper
 *       rocke_build_unified_attention_2d_tiled_scalar_new
 *   - the build->lower convenience
 *       rocke_gfx942_attention_tiled_2d_lower_to_llvm
 *   - the admission gate (supports_tiled_2d, Python lines 887-1096)
 *       rocke_gfx942_attention_tiled_2d_supports_args_default
 *       rocke_gfx942_attention_tiled_2d_supports
 *
 * The KV-loop scaffolding (tile_start/tile_end/kv_step bounds, the
 * scf_for_iter carry layout, and the per-iter unpack/repack of the online
 * softmax state) lives in rocke_gfx942_attn2d_build_ctx_init / emit_kv_body /
 * emit_epilogue per the internal contract; this driver wires the phases.
 *
 * NOTE: the spec kernel_name() port lives here (the _new convenience needs it
 * before rocke_ir_builder_init); it is a faithful transcription of
 * UnifiedAttention2DTiledSpec.kernel_name (lines 827-885) through
 * rocke_kernel_name_join.
 */
#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx942_attention_tiled_2d_internal.h"

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include <stdio.h>
#include <string.h>

/* ===================================================================== *
 *  Local helpers
 * ===================================================================== */

/* Write a static diagnostic into err (capacity err_cap) when non-NULL. */
static void rocke__set_err(char* err, size_t err_cap, const char* m)
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
static void rocke__reason(char* reason, size_t reason_cap, const char* m)
{
    rocke__set_err(reason, reason_cap, m);
}

/* ===================================================================== *
 *  attention_arch.py predicates (validate_tiled_attention_arch port).
 *
 *  These mirror instances/common/attention_arch.py: _wide_k_mfma_available,
 *  _narrow_k_mfma_available, the _NARROW_TILED_2D_ARCHES set ({"gfx942"}), and
 *  validate_tiled_attention_arch itself. Catalog-driven via rocke_arch_target.
 * ===================================================================== */

/* _wide_k_mfma_available(target): 16x16x32 f16 atom OR has_ds_read_tr. */
static bool rocke__wide_k_mfma_available(const rocke_arch_target_t* target)
{
    if(rocke_mma_catalog_has_shape(&target->mma, "mma", "f16", "f16", "fp32", 16, 16, 32))
        return true;
    return target->memory.has_ds_read_tr;
}

/* _narrow_k_mfma_available(target): 16x16x16 f16 AND bf16 atoms. */
static bool rocke__narrow_k_mfma_available(const rocke_arch_target_t* target)
{
    return rocke_mma_catalog_has_shape(&target->mma, "mma", "f16", "f16", "fp32", 16, 16, 16)
           && rocke_mma_catalog_has_shape(&target->mma, "mma", "bf16", "bf16", "fp32", 16, 16, 16);
}

/* arch in _NARROW_TILED_2D_ARCHES == frozenset({"gfx942"}). */
static bool rocke__arch_in_narrow_set(const char* arch)
{
    return arch != NULL && strcmp(arch, "gfx942") == 0;
}

/* validate_tiled_attention_arch(arch) -> (ok, reason). Returns true/false and
 * writes the reason (cap reason_cap) on every path. */
static bool rocke__validate_tiled_attention_arch(const char* arch, char* reason, size_t reason_cap)
{
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        /* Python: except KeyError as e: return False, str(e). The KeyError
         * carries the missing gfx key. */
        char buf[128];
        snprintf(buf, sizeof(buf), "%s", arch ? arch : "None");
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(rocke__wide_k_mfma_available(target) && target->memory.has_ds_read_tr)
    {
        rocke__reason(reason, reason_cap, "ok");
        return true;
    }
    if(rocke__arch_in_narrow_set(arch) && rocke__narrow_k_mfma_available(target))
    {
        rocke__reason(reason, reason_cap, "ok");
        return true;
    }
    if(!rocke__wide_k_mfma_available(target))
    {
        char buf[320];
        snprintf(buf,
                 sizeof(buf),
                 "tiled attention requires either the wide-K MFMA atoms "
                 "(mfma_f32_16x16x32 / mfma_f32_32x32x16, gfx950) or, on a "
                 "narrow variant arch (gfx942), the 16x16x16 f16/bf16 atom; "
                 "neither path is available on %s",
                 arch);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    {
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "tiled attention requires LDS transpose reads "
                 "(ds_read_b64_tr_b16) for the wide-K path, absent on %s",
                 arch);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
}

/* ===================================================================== *
 *  supports_tiled_2d(...)  (Python lines 887-1096)
 * ===================================================================== */

rocke_gfx942_attention_tiled_2d_supports_args_t
    rocke_gfx942_attention_tiled_2d_supports_args_default(void)
{
    rocke_gfx942_attention_tiled_2d_supports_args_t a;
    memset(&a, 0, sizeof(a));
    a.num_warps = 1;
    a.block_m_per_warp = 16;
    /* head_size/block_size/dtype/num_queries_per_kv set by the caller;
     * every other field defaults to 0/false/NULL == the Python keyword
     * default (q_dtype=None, kv_storage_dtype=None, tile_size=None, ...). */
    return a;
}

bool rocke_gfx942_attention_tiled_2d_supports(
    const rocke_gfx942_attention_tiled_2d_supports_args_t* args, char* reason, size_t reason_cap)
{
    const char* arch;
    int block_m;
    char buf[320];

    if(args == NULL)
    {
        rocke__reason(reason, reason_cap, "null args");
        return false;
    }
    arch = args->arch ? args->arch : "gfx942";

    /* arch_ok, arch_reason = validate_tiled_attention_arch(arch) */
    if(!rocke__validate_tiled_attention_arch(arch, reason, reason_cap))
        return false;

    /* The fp8 K/V cache path needs ds_read_tr_b8 (gfx950-only). */
    if(args->kv_storage_dtype != NULL)
    {
        rocke__reason(reason,
                      reason_cap,
                      "gfx942 tiled 2D kernel does not support the fp8 K/V cache "
                      "(ds_read_tr_b8 is gfx950-only)");
        return false;
    }
    if(!(args->dtype != NULL
         && (strcmp(args->dtype, "fp16") == 0 || strcmp(args->dtype, "bf16") == 0)))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel currently supports fp16/bf16 (got %s%s%s)",
                 args->dtype ? "'" : "",
                 args->dtype ? args->dtype : "None",
                 args->dtype ? "'" : "");
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(!(args->head_size == 64 || args->head_size == 128 || args->head_size == 256))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel only supports head_size in {64,128,256} "
                 "(got %d)",
                 args->head_size);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(args->head_size % 32 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel requires head_size divisible by 32 (got %d)",
                 args->head_size);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(!(args->block_size == 16 || args->block_size == 32 || args->block_size == 64))
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel only supports block_size in {16,32,64} "
                 "(got %d)",
                 args->block_size);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(args->num_queries_per_kv > 16 || args->num_queries_per_kv < 1)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel needs 1<=num_queries_per_kv<=16 (got %d)",
                 args->num_queries_per_kv);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    block_m = 16 * args->num_warps;
    if(block_m % args->num_queries_per_kv != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel needs num_queries_per_kv to divide "
                 "BLOCK_M=%d (num_warps=%d, got num_queries_per_kv=%d)",
                 block_m,
                 args->num_warps,
                 args->num_queries_per_kv);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    /* kv_storage_dtype is None here (early-rejected above), so the
     * "unsupported kv_storage_dtype" and "use_fp8=True requires fp8e4m3"
     * branches are unreachable given the gfx942 None-only path; reproduce them
     * faithfully anyway for parity with the shared body. */
    if(args->kv_storage_dtype != NULL && strcmp(args->kv_storage_dtype, "fp8e4m3") != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "tiled 2D kernel: unsupported kv_storage_dtype '%s'",
                 args->kv_storage_dtype);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(args->use_fp8 && args->kv_storage_dtype == NULL)
    {
        rocke__reason(reason,
                      reason_cap,
                      "tiled 2D kernel: use_fp8=True requires "
                      "kv_storage_dtype='fp8e4m3'");
        return false;
    }
    if(args->q_dtype != NULL && strcmp(args->q_dtype, "fp16") != 0
       && strcmp(args->q_dtype, "bf16") != 0)
    {
        snprintf(buf, sizeof(buf), "tiled 2D kernel: unsupported q_dtype '%s'", args->q_dtype);
        rocke__reason(reason, reason_cap, buf);
        return false;
    }
    if(args->has_tile_size)
    {
        int halves_per_lane;
        int threads;
        int per_wave_tokens;
        if(args->tile_size <= 0 || args->tile_size % args->block_size != 0)
        {
            snprintf(buf,
                     sizeof(buf),
                     "tiled 2D kernel: tile_size=%d must be a positive "
                     "multiple of block_size=%d",
                     args->tile_size,
                     args->block_size);
            rocke__reason(reason, reason_cap, buf);
            return false;
        }
        halves_per_lane = 2;
        threads = args->num_warps * 64;
        if(args->tile_size * args->head_size < threads * halves_per_lane)
        {
            snprintf(buf,
                     sizeof(buf),
                     "tiled 2D kernel: tile_size*head_size=%d too small for "
                     "num_warps=%d (need >= %d)",
                     args->tile_size * args->head_size,
                     args->num_warps,
                     threads * halves_per_lane);
            rocke__reason(reason, reason_cap, buf);
            return false;
        }
        per_wave_tokens = (64 * 2) / args->head_size;
        if(per_wave_tokens > args->block_size)
        {
            snprintf(buf,
                     sizeof(buf),
                     "tiled 2D kernel: per-wave tokens %d exceeds "
                     "block_size=%d; would need lane-divergent block lookup",
                     per_wave_tokens,
                     args->block_size);
            rocke__reason(reason, reason_cap, buf);
            return false;
        }
    }

    /* LDS-budget gate (gate only the validated fp16/bf16 path). */
    if(!args->use_fp8)
    {
        const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
        int _LDS_CAPACITY_BYTES = (t != NULL) ? t->lds_capacity_bytes : 65536;
        const int _BPE = 2; /* fp16/bf16 */
        int _t_eff = args->has_tile_size ? args->tile_size : args->block_size;
        int _block_m = args->num_warps * args->block_m_per_warp;

        if(args->use_mfma_32x32x8 && args->use_transposed_qk_32x32)
        {
            if(args->use_k_sliced_ring)
            {
                int _K_SLICE_HD = 32;
                int _K_SLICE_SLOTS = 3;
                int _k_bytes = _K_SLICE_SLOTS * _t_eff * _K_SLICE_HD * _BPE;
                int _v_kpack = 8;
                int _v_pixels = 64;
                int _v_nper_row = _v_pixels / _v_kpack;
                int _v_ngroups = args->head_size / _v_nper_row;
                int _v_kgroups = _t_eff / _v_kpack;
                int _v_bytes = _v_kgroups * _v_ngroups * (_v_pixels + _v_kpack) * _BPE;
                int _lds_x8 = _k_bytes + _v_bytes;
                if(_lds_x8 > _LDS_CAPACITY_BYTES)
                {
                    snprintf(buf,
                             sizeof(buf),
                             "gfx942 transposed-x8 sliced-K ring estimated "
                             "LDS %d B exceeds the %s %d B LDS budget",
                             _lds_x8,
                             arch,
                             _LDS_CAPACITY_BYTES);
                    rocke__reason(reason, reason_cap, buf);
                    return false;
                }
                rocke__reason(reason, reason_cap, "supported");
                return true;
            }
            {
                int k_slots = args->use_k_single_buffer ? 1 : 2;
                int q_lds = (_block_m <= 2 * _t_eff) ? 0 : _block_m * args->head_size * _BPE;
                int v_pad = args->use_conflict_free_v_store ? 8 : 0;
                int _lds_x8 = k_slots * _t_eff * args->head_size * _BPE
                              + (_t_eff + v_pad) * args->head_size * _BPE + q_lds;
                if(_lds_x8 > _LDS_CAPACITY_BYTES)
                {
                    snprintf(buf,
                             sizeof(buf),
                             "gfx942 transposed-x8 estimated LDS %d B exceeds "
                             "the %s %d B LDS budget",
                             _lds_x8,
                             arch,
                             _LDS_CAPACITY_BYTES);
                    rocke__reason(reason, reason_cap, buf);
                    return false;
                }
                rocke__reason(reason, reason_cap, "supported");
                return true;
            }
        }
        {
            int _out_stripe = (args->head_size <= 64) ? 32 : args->head_size;
            int _lds
                = 2 * _t_eff * args->head_size * _BPE /* K_lds (double) */
                  + _t_eff * args->head_size * _BPE /* V_lds          */
                  + _block_m * (_t_eff + 8) * _BPE /* P_lds          */
                  + ((_block_m <= 2 * _t_eff) ? 0 : _block_m * args->head_size * _BPE) /* Q_lds */
                  + _block_m * _out_stripe * _BPE; /* Acc_lds     */
            if(_lds > _LDS_CAPACITY_BYTES)
            {
                snprintf(buf,
                         sizeof(buf),
                         "tiled 2D kernel: estimated LDS %d B (BLOCK_M=%d, "
                         "T=%d, HD=%d) exceeds the %s %d B LDS budget; comgr "
                         "CODEGEN would fail",
                         _lds,
                         _block_m,
                         _t_eff,
                         args->head_size,
                         arch,
                         _LDS_CAPACITY_BYTES);
                rocke__reason(reason, reason_cap, buf);
                return false;
            }
        }
    }

    rocke__reason(reason, reason_cap, "supported");
    return true;
}

/* ===================================================================== *
 *  build_unified_attention_2d_tiled(spec, arch="gfx942")  (lines 1104-5287)
 *
 *  The driver: populate the shared build ctx via rocke_gfx942_attn2d_build_ctx_init
 *  (the whole prologue: arch gate -> dtype gate -> narrow-atom select -> config
 *  -> kernel/param decls -> grid/seq/Q-block geometry -> LDS layout + smem_alloc
 *  -> SSA constants -> KV-loop bounds + iter-arg carry), then run the phase
 *  functions in Python execution order and return b->kernel from the epilogue.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_unified_attention_2d_tiled_scalar(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        rocke_gfx942_attn2d_build_ctx_t ctx;
        rocke_kernel_def_t* kernel;

        if(b == NULL || spec == NULL)
            return NULL;

        /* Re-entrancy: clear every cross-build static this generator caches against
         * the previous build's (now-freed) arena before emitting build N. Without
         * this, the lazily-built _C32_DIST distribution and the REGISTER_PV scratch
         * carry dangling pointers from build N-1 into the new IR (a NULL operand to
         * warp_shuffle_xor on the second call). The generator is invoked once per
         * workspace-query and once per buildPlan, per graph, so this runs many times
         * per process. (_scalar_new funnels here too.) */
        rocke_attn2d_c32_dist_reset();
        rocke_gfx942_attn2d_reset_softmax_scratch();

        /* Prologue port: arch/dtype gate, narrow-atom select, config derivation,
         * kernel + params, grid / seq-idx / Q-block geometry, LDS layout +
         * smem_alloc, SSA constants, KV-loop bounds + the online-softmax iter-arg
         * carry. On any reject it sets b's sticky error and returns false. */
        memset(&ctx, 0, sizeof(ctx));
        if(!rocke_gfx942_attn2d_build_ctx_init(&ctx, b, spec, arch))
            return NULL;

        /* The emission order below mirrors the single linear Python emitter exactly
         * (the byte-identity contract is on the lowered .ll, so the IR op stream must
         * be produced in Python source order):
         *
         *   q_load (1913) -> tile bounds + softmax iter-arg inits (1982-2161) ->
         *   pre-loop K/V buffer descriptors (2163-2351) -> Q VGPR gather (3426-3592)
         *   -> tile-0 K prefetch + cur_buf carry (3591, 3606-3607) -> LICM hoist
         *   (3609-3688) -> kv_step (3689) -> scf.for KV loop (5050-5052) ->
         *   epilogue (5054-5287).
         */

        /* 1. Cooperatively stage Q[BLOCK_M, HD] global -> LDS (lines 1913-1980). */
        rocke_gfx942_attn2d_emit_q_load(&ctx);

        /* 2. max_seq_prefix_len -> tile_start/tile_end + online-softmax m/l/acc carry
         *    inits + named iter_args (lines 1982-2161). */
        rocke_gfx942_attn2d_emit_loop_bounds_and_inits(&ctx);

        /* 3. Pre-loop: build K/V buffer descriptors (lines 2163-2351). */
        rocke_gfx942_attn2d_emit_preloop(&ctx);

        /* 4. Gather the per-lane Q MFMA A-operand to VGPRs (lines 3426-3592). */
        rocke_gfx942_attn2d_emit_q_gather(&ctx);

        /* 5. tile-0 K prefetch into buffer 0 + ("cur_buf", 0) carry append (3591,
         *    3606-3607). */
        rocke_gfx942_attn2d_emit_preloop_prefetch(&ctx);

        /* 6. LICM hoist of the per-reg row/pos/head/mask invariants (3609-3688). */
        rocke_gfx942_attn2d_emit_licm_hoist(&ctx);

        /* 7. kv_step const (3689). */
        rocke_gfx942_attn2d_emit_kv_step(&ctx);

        /* 8. KV-loop: build the scf.for over [tile_start, tile_end) with the named
         *    online-softmax carry and run one full KV-tile body inside it (5050-5052).
         *    drive_kv_loop returns the loop handle whose results are the rewritten
         *    carry the epilogue consumes; mirror them into ctx.out_carry. */
        rocke_for_t kvloop = rocke_gfx942_attn2d_drive_kv_loop(&ctx);
        if(kvloop.op != NULL)
        {
            for(int i = 0; i < kvloop.op->num_results; ++i)
                ctx.out_carry[i] = kvloop.op->results[i];
            ctx.out_carry_count = kvloop.op->num_results;
        }

        /* 9. Epilogue: drain async copies, read loop results, normalize + store;
         * returns b->kernel on success (lines 5054-5287). */
        kernel = rocke_gfx942_attn2d_emit_epilogue(&ctx);

        return rocke_ir_builder_ok(b) ? kernel : NULL;
    });
}

/* ===================================================================== *
 *  spec.kernel_name()  (Python lines 827-885), needed by the _new wrapper.
 * ===================================================================== */
static rocke_status_t
    rocke__attn2d_kernel_name(const rocke_attention_tiled_2d_spec_t* s, char* out, size_t out_cap)
{
    /* Value-carrying optionals become conditional strings; value-less flags
     * are passed as positional parts (kernel_name_join drops empties), exactly
     * as the Python kernel_name does. */
    char d_buf[32], b_buf[32], t_buf[32], hkv_buf[64], kv_buf[64];
    char sw_buf[32], w_buf[32], wpe_buf[32], mw_buf[32], kvcp_buf[64];
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
    if(s->has_waves_per_eu)
        snprintf(wpe_buf, sizeof(wpe_buf), "wpe%d", s->waves_per_eu);
    else
        wpe_buf[0] = '\0';
    if(s->block_m_per_warp != 16)
        snprintf(mw_buf, sizeof(mw_buf), "mw%d", s->block_m_per_warp);
    else
        mw_buf[0] = '\0';
    if(s->kv_cache_policy && strcmp(s->kv_cache_policy, "stream") != 0)
        snprintf(kvcp_buf, sizeof(kvcp_buf), "kvcp%s", s->kv_cache_policy);
    else
        kvcp_buf[0] = '\0';

    {
        const char* cfvnosplit
            = (s->use_conflict_free_v_store && !s->use_conflict_free_v_store_split) ? "cfvnosplit"
                                                                                    : "";
        const char* cfvplainv
            = (s->use_conflict_free_v_store && !s->use_conflict_free_v_ck_vlds) ? "cfvplainv" : "";
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
            wpe_buf,
            mw_buf,
            s->use_mfma_32x32 ? "mfma32" : "",
            s->use_mfma_32x32x8 ? "mfma32x8" : "",
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
            s->use_q_direct_global ? "qdir" : "",
            s->use_qk_pv_sched_group_barrier ? "qsgb" : "",
            kvcp_buf,
            s->use_global_load_lds_k ? "gldlds" : "",
            s->use_q_major_grid ? "qgrid" : "",
            s->use_agpr_alloc_zero ? "agpr0" : "",
            s->use_fp8_mfma_qk ? "fp8mfma" : "",
            s->use_fp8_mfma_pv ? "fp8pv" : "",
            s->use_register_pv ? "regpv" : "",
            s->use_conflict_free_v ? "cfv" : "",
            s->use_conflict_free_v_store ? "cfvst" : "",
            cfvnosplit,
            cfvplainv,
            s->use_k_sliced_ring ? "ksring" : "",
            s->use_k_sliced_ldsseq ? "ldsseq" : "",
            s->use_iglp_opt ? "iglp1" : "",
            s->use_k_single_buffer ? "k1buf" : "",
        };
        size_t num_parts = sizeof(parts) / sizeof(parts[0]);
        return rocke_kernel_name_join(
            "rocke_uattn2d_tiled", parts, num_parts, NULL, NULL, 0, out, out_cap, NULL);
    }
}

/* ===================================================================== *
 *  _new convenience: init builder via spec.kernel_name(), then build.
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_unified_attention_2d_tiled_scalar_new(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char kname[1024];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke__attn2d_kernel_name(spec, kname, sizeof(kname)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, kname) != ROCKE_OK)
            return NULL;
        return rocke_build_unified_attention_2d_tiled_scalar(b, spec, arch);
    });
}

/* Public wrapper over the static spec.kernel_name() port, so the fastKV
 * register-P build can derive the wrapped spec's base name and append its
 * "_fastkv_regp" suffix (Python _FastKvRegisterPProxy.kernel_name). */
rocke_status_t rocke_attention_tiled_2d_spec_kernel_name(
    const rocke_attention_tiled_2d_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    return rocke__attn2d_kernel_name(spec, out, out_cap);
}

/* ===================================================================== *
 *  build -> lower convenience.
 * ===================================================================== */
rocke_status_t
    rocke_gfx942_attention_tiled_2d_lower_to_llvm(const rocke_attention_tiled_2d_spec_t* spec,
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
        rocke__set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }

    kernel = rocke_build_unified_attention_2d_tiled_scalar_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        rocke__set_err(err, err_cap, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, arch ? arch : "gfx942", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
