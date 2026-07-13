// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_helper_rocke.instances.gfx942.attention_tiled_2d.c -- C99 port of the
 * task-named symbols from rocke/instances/gfx942/attention_tiled_2d.py.
 *
 * Ported: UnifiedAttention2DTiledSpec (-> rocke_attention_tiled_2d_spec_t +
 * default/validate/derived-property helpers), the build-head config derivation
 * (-> rocke_unified_attention_2d_tiled_config_from_spec), _mfma_32x32_c_row /
 * _mfma_32x32_c_col, and the kernel build entry (stub-to-link).
 *
 * The two 32x32 C helpers reproduce the Python builder-call sequence
 * byte-faithfully: same div/mod constants, same calculate_x ys/ps wiring, same
 * trailing add for the N-tile base. _C32_DIST (built once at Python module
 * import from make_static_tile_distribution(make_c_warp_dstr_encoding(
 * MfmaAtom.f16_32x32x16()))) is reproduced as a lazily-built, process-lifetime
 * cached distribution off rocke_mfma_atom("f16", 32, 32, 16).
 *
 * Lifetime: every IR node is arena-owned (rocke_ir_builder_t.arena). Nothing is
 * freed individually. The cached _C32_DIST is built on the first call's builder
 * arena and is dtype-independent host-side analysis state (the Python caches it
 * at module scope). Because the C builder/arena is freed at the end of each
 * build, the cache is per-build (not process-lifetime): rocke_attn2d_c32_dist_reset()
 * clears it at every build entry so build N+1 never reads build N's freed arena.
 *
 * Error model: pure helpers return a sentinel (NULL/false); builder variants
 * latch the first Python ValueError/NotImplementedError onto the sticky-error
 * IRBuilder and return the sentinel. A dead/NULL builder is a no-op.
 */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"
#include "rocke/ir.h"

/* ------------------------------------------------------------- error latch */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing
 * `return (T*)rocke_attn2d_set_err(...)` call sites valid -- the cast/return is
 * simply never reached. */
[[noreturn]] static void*
    rocke_attn2d_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

static bool rocke_streq(const char* a, const char* c)
{
    if(a == NULL || c == NULL)
    {
        return a == c;
    }
    return strcmp(a, c) == 0;
}

/* ------------------------------------------------------------- spec default */

rocke_attention_tiled_2d_spec_t rocke_attention_tiled_2d_spec_default(void)
{
    rocke_attention_tiled_2d_spec_t s;
    memset(&s, 0, sizeof(s));

    /* required fields stay zero/NULL until the caller sets them. */

    /* defaulted fields (mirror the dataclass defaults). */
    s.use_alibi = false;
    s.use_qq_bias = false;
    s.num_seqs = 0;
    s.num_warps = 1;
    s.has_waves_per_eu = false;
    s.waves_per_eu = 0;
    s.kv_storage_dtype = NULL;
    s.use_fp8_mfma_qk = false;
    s.use_fp8_mfma_pv = false;
    s.use_register_pv = false;
    s.use_v_double_buffer = false;
    s.kv_ring_depth = 2;
    s.use_staggered_iter_wait = false;
    s.use_q_reread = false;
    s.use_q_direct_reg = false; /* #79 */
    s.use_sched_barrier = false;
    s.sched_barrier_mask = 0;
    s.use_softmax_mfma_interleave = false; /* #80 */
    s.softmax_interleave_mode = 1;
    s.softmax_interleave_groups = 4;
    s.has_tile_size = false;
    s.tile_size = 0;
    s.block_m_per_warp = 16;
    s.use_mfma_32x32 = false;
    s.use_mfma_32x32x8 = false;
    s.use_transposed_qk_32x32 = false;
    s.use_transposed_scalar_state = false;
    s.use_transposed_invariant_hoist = false;
    s.use_transposed_mask_once = false;
    s.use_transposed_half_local_pv = false;
    s.use_mfma32_skip_legacy_qreg = false;
    s.use_transposed_mask_limit = false;
    s.use_grouped_kv2_softmax = false;
    s.use_fast_paged_kv_desc = false;
    s.use_i64_kv_addr = false;
    s.use_early_v_schedule = false;
    s.use_agpr_alloc_zero = false;
    s.use_conflict_free_v = false;
    s.use_conflict_free_v_store = false;
    s.use_conflict_free_v_store_split = true;
    s.use_conflict_free_v_ck_vlds = true;
    s.use_k_single_buffer = false;
    s.use_k_sliced_ring = false;
    s.use_k_sliced_ldsseq = false;
    s.use_iglp_opt = false;
    s.use_qk_pv_sched_group_barrier = false;
    s.use_q_direct_global = false;
    s.kv_cache_policy = "stream";
    s.use_global_load_lds_k = false;
    s.use_q_major_grid = false;
    return s;
}

/* ------------------------------------------------- derived @property bodies */

int rocke_attention_tiled_2d_spec_num_queries_per_kv(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL || s->num_kv_heads == 0)
    {
        return 0;
    }
    return s->num_query_heads / s->num_kv_heads;
}

int rocke_attention_tiled_2d_spec_block_m(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    return s->block_m_per_warp * s->num_warps;
}

int rocke_attention_tiled_2d_spec_regs_per_lane(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    if(s->use_mfma_32x32 || s->use_mfma_32x32x8)
    {
        return 16;
    }
    return s->block_m_per_warp / 4; /* 4 for M=16, 8 for M=32 */
}

int rocke_attention_tiled_2d_spec_block_q(const rocke_attention_tiled_2d_spec_t* s)
{
    int nqk;
    if(s == NULL)
    {
        return 0;
    }
    nqk = rocke_attention_tiled_2d_spec_num_queries_per_kv(s);
    if(nqk == 0)
    {
        return 0;
    }
    return rocke_attention_tiled_2d_spec_block_m(s) / nqk;
}

int rocke_attention_tiled_2d_spec_tile_size_eff(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    return s->has_tile_size ? s->tile_size : s->block_size;
}

int rocke_attention_tiled_2d_spec_n_blocks_per_tile(const rocke_attention_tiled_2d_spec_t* s)
{
    int bs;
    if(s == NULL)
    {
        return 0;
    }
    bs = s->block_size;
    if(bs == 0)
    {
        return 0;
    }
    return rocke_attention_tiled_2d_spec_tile_size_eff(s) / bs;
}

const rocke_type_t* rocke_attention_tiled_2d_spec_dtype_ir(const rocke_attention_tiled_2d_spec_t* s)
{
    if(s != NULL && rocke_streq(s->dtype, "fp16"))
    {
        return rocke_f16();
    }
    return rocke_bf16();
}

int rocke_attention_tiled_2d_spec_binary_search_iters(const rocke_attention_tiled_2d_spec_t* s)
{
    int it;
    if(s == NULL || s->num_seqs <= 0)
    {
        return 32;
    }
    /* max(1, ceil(log2(num_seqs + 1))) */
    it = (int)ceil(log2((double)(s->num_seqs + 1)));
    return it < 1 ? 1 : it;
}

/* --------------------------------------------------------- __post_init__ */
/* Faithful reproduction of the gfx942 __post_init__ validation order/messages.
 * Returns false + latches ROCKE_ERR_VALUE on the first failing check. */
bool rocke_attention_tiled_2d_spec_validate(rocke_ir_builder_t* b,
                                            const rocke_attention_tiled_2d_spec_t* s)
{
    int block_m;
    int t_eff;

    /* Internal builder ops raise on failure, so a reachable builder here is
     * always in the OK state; only the NULL guard remains. */
    if(b == NULL)
    {
        return false;
    }
    if(s == NULL)
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "attention_tiled_2d spec is NULL");
        return false;
    }

    /* gfx950-only experimental knobs rejected up front on gfx942.
     * NOTE: use_agpr_alloc_zero is NOT an ISA feature -- it is a backend codegen
     * hint ("amdgpu-agpr-alloc"="0,0" + -amdgpu-mfma-vgpr-form) that keeps MFMA
     * accumulators in VGPR. It is legal on the gfx942 wide x8 transposed path;
     * the detailed pairing guard below enforces that (mirrors Python
     * attention_tiled_2d.py line 765). So it is gated there, not blanket-rejected
     * here. */
    if(s->use_mfma_32x32 || s->use_transposed_half_local_pv || s->use_mfma32_skip_legacy_qreg
       || s->use_grouped_kv2_softmax || s->use_fp8_mfma_qk || s->use_fp8_mfma_pv)
    {
        rocke_attn2d_set_err(b,
                             ROCKE_ERR_VALUE,
                             "gfx942 tiled-2D attention supports only the narrow 16x16x16 "
                             "default path; gfx950-only knobs are not available on gfx942");
        return false;
    }

    /* transposed orientation legal only in the x8 pairing. */
    if(s->use_transposed_qk_32x32 && !s->use_mfma_32x32x8)
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "gfx942: use_transposed_qk_32x32 requires use_mfma_32x32x8");
        return false;
    }

    /* fp8 K/V cache is gfx950-only here. */
    if(s->kv_storage_dtype != NULL)
    {
        rocke_attn2d_set_err(b,
                             ROCKE_ERR_VALUE,
                             "gfx942 tiled-2D attention has no fp8 K/V cache path "
                             "(kv_storage_dtype must be None on gfx942)");
        return false;
    }

    if(!(s->num_warps == 1 || s->num_warps == 2 || s->num_warps == 4 || s->num_warps == 8))
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "num_warps must be in {1, 2, 4, 8}");
        return false;
    }

    if(!(s->block_m_per_warp == 16 || s->block_m_per_warp == 32))
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "block_m_per_warp must be in {16, 32}");
        return false;
    }

    t_eff = rocke_attention_tiled_2d_spec_tile_size_eff(s);

    if(s->use_mfma_32x32x8)
    {
        if(s->use_mfma_32x32)
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_mfma_32x32x8 and use_mfma_32x32 are mutually exclusive");
            return false;
        }
        if(!(rocke_streq(s->dtype, "fp16") || rocke_streq(s->dtype, "bf16")))
        {
            /* The gfx942-legal K=8 32x32x8 atom exists for both f16 and bf16
             * (mfma_f32_32x32x8_{f16,bf16}); only the K=16 bf16 atom is
             * gfx950-only. Mirrors the Python use_mfma_32x32x8 dtype gate. */
            rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "use_mfma_32x32x8 requires fp16 or bf16");
            return false;
        }
        if(s->block_m_per_warp != 32)
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_mfma_32x32x8 requires block_m_per_warp=32");
            return false;
        }
        if(t_eff % 32 != 0)
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_mfma_32x32x8 requires tile_size_eff %% 32 == 0");
            return false;
        }
        if(s->head_size % 32 != 0)
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_mfma_32x32x8 requires head_size %% 32 == 0");
            return false;
        }
    }

    /* transposed sub-knob dependencies. */
    if(s->use_transposed_scalar_state && !s->use_transposed_qk_32x32)
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_transposed_scalar_state requires use_transposed_qk_32x32");
        return false;
    }
    if(s->use_transposed_invariant_hoist && !s->use_transposed_qk_32x32)
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_transposed_invariant_hoist requires use_transposed_qk_32x32");
        return false;
    }
    if(s->use_transposed_mask_once && !s->use_transposed_qk_32x32)
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_transposed_mask_once requires use_transposed_qk_32x32");
        return false;
    }

    /* conflict-free V feed requires the transposed-x8 orientation. */
    if(s->use_conflict_free_v && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_conflict_free_v requires the transposed-x8 path");
        return false;
    }
    if(s->use_conflict_free_v_store && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_conflict_free_v_store requires the transposed-x8 path");
        return false;
    }
    if(s->use_conflict_free_v_store && s->use_conflict_free_v)
    {
        rocke_attn2d_set_err(b,
                             ROCKE_ERR_VALUE,
                             "use_conflict_free_v_store and use_conflict_free_v are "
                             "mutually exclusive");
        return false;
    }

    /* K single-buffer occupancy lever. */
    if(s->use_k_single_buffer)
    {
        if(!(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_k_single_buffer requires the transposed-x8 path");
            return false;
        }
        if(!(rocke_streq(s->dtype, "fp16") || rocke_streq(s->dtype, "bf16")))
        {
            rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "use_k_single_buffer requires fp16 or bf16");
            return false;
        }
        block_m = rocke_attention_tiled_2d_spec_block_m(s);
        if(block_m > t_eff)
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_k_single_buffer requires BLOCK_M <= tile_size_eff");
            return false;
        }
    }

    /* K sliced ring + ldsseq dependencies. */
    if(s->use_k_sliced_ring
       && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32 && s->use_conflict_free_v_store))
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_k_sliced_ring requires the transposed-x8 cfvst path");
        return false;
    }
    if(s->use_k_sliced_ring)
    {
        /* The 32-wide K slices need HD %% 32 == 0; the ring is byte-size driven
         * so fp16 and bf16 (both 2-byte) are legal. Mirrors the Python gate. */
        if(!(rocke_streq(s->dtype, "fp16") || rocke_streq(s->dtype, "bf16"))
           || !(s->head_size == 64 || s->head_size == 128) || (s->head_size % 32 != 0)
           || !(t_eff == 64 || t_eff == 128))
        {
            rocke_attn2d_set_err(b,
                                 ROCKE_ERR_VALUE,
                                 "use_k_sliced_ring requires fp16/bf16, head_size in {64,128} "
                                 "(HD %% 32 == 0 for the 32-wide K slices), T in {64,128}");
            return false;
        }
    }
    if(s->use_k_sliced_ldsseq && !s->use_k_sliced_ring)
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "use_k_sliced_ldsseq requires use_k_sliced_ring");
        return false;
    }

    if(s->use_q_direct_global && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_q_direct_global currently targets transposed-x8");
        return false;
    }

    if(s->use_qk_pv_sched_group_barrier)
    {
        if(!(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
        {
            rocke_attn2d_set_err(b,
                                 ROCKE_ERR_VALUE,
                                 "use_qk_pv_sched_group_barrier requires the transposed-x8 path "
                                 "(use_mfma_32x32x8 + use_transposed_qk_32x32)");
            return false;
        }
        if(!(rocke_streq(s->dtype, "fp16") || rocke_streq(s->dtype, "bf16")))
        {
            rocke_attn2d_set_err(
                b, ROCKE_ERR_VALUE, "use_qk_pv_sched_group_barrier requires fp16 or bf16");
            return false;
        }
        if(s->use_iglp_opt)
        {
            rocke_attn2d_set_err(b,
                                 ROCKE_ERR_VALUE,
                                 "use_qk_pv_sched_group_barrier is mutually exclusive with "
                                 "use_iglp_opt (iglp_opt owns the loop schedule)");
            return false;
        }
    }

    if(s->num_warps == 8 && s->block_m_per_warp == 32
       && !(s->use_q_direct_global && s->use_conflict_free_v_store))
    {
        rocke_attn2d_set_err(b,
                             ROCKE_ERR_VALUE,
                             "num_warps=8 with block_m_per_warp=32 requires "
                             "use_q_direct_global + use_conflict_free_v_store");
        return false;
    }

    if(!(rocke_streq(s->kv_cache_policy, "all") || rocke_streq(s->kv_cache_policy, "global")
         || rocke_streq(s->kv_cache_policy, "stream") || rocke_streq(s->kv_cache_policy, "nt")))
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "kv_cache_policy must be one of {all, global, stream, nt}");
        return false;
    }

    if(s->use_global_load_lds_k && s->kv_storage_dtype != NULL)
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_global_load_lds_k v1 supports bf16/fp16 KV only");
        return false;
    }

    if(s->use_mfma32_skip_legacy_qreg && !s->use_mfma_32x32)
    {
        rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "use_mfma32_skip_legacy_qreg requires use_mfma_32x32");
        return false;
    }

    /* use_agpr_alloc_zero detailed pairing guard (mirrors Python
     * attention_tiled_2d.py line 765). On gfx942 the R4_s1mask_hlpv family is
     * gfx950-only (use_mfma_32x32 / use_transposed_half_local_pv are rejected
     * above), so the only legal pairing here is the wide x8 transposed path. */
    if(s->use_agpr_alloc_zero)
    {
        const bool agpr0_r4_s1mask_hlpv
            = s->use_mfma_32x32 && s->use_transposed_qk_32x32 && s->use_transposed_scalar_state
              && s->use_transposed_mask_once && s->use_transposed_half_local_pv;
        const bool agpr0_wide_x8 = s->use_mfma_32x32x8 && s->use_transposed_qk_32x32;
        if(!(agpr0_r4_s1mask_hlpv || agpr0_wide_x8))
        {
            rocke_attn2d_set_err(b,
                                 ROCKE_ERR_VALUE,
                                 "use_agpr_alloc_zero currently targets the R4_s1mask_hlpv "
                                 "path or the wide x8 transposed path "
                                 "(use_mfma_32x32x8 + use_transposed_qk_32x32)");
            return false;
        }
    }

    return true;
}

/* --------------------------------------------- config-from-spec (build head) */

bool rocke_unified_attention_2d_tiled_config_from_spec(
    rocke_ir_builder_t* b,
    const rocke_attention_tiled_2d_spec_t* spec,
    rocke_unified_attention_2d_tiled_config_t* out)
{
    if(b == NULL)
    {
        return false;
    }
    if(spec == NULL || out == NULL)
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "config_from_spec: NULL spec/out");
        return false;
    }

    /* __post_init__ runs at dataclass construction; reproduce it here. */
    if(!rocke_attention_tiled_2d_spec_validate(b, spec))
    {
        return false;
    }

    /* dtype gate (Python NotImplementedError). */
    if(!(rocke_streq(spec->dtype, "fp16") || rocke_streq(spec->dtype, "bf16")))
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_NOTIMPL, "tiled 2D kernel supports fp16/bf16");
        return false;
    }

    memset(out, 0, sizeof(*out));

    out->HD = spec->head_size;
    out->T = rocke_attention_tiled_2d_spec_tile_size_eff(spec);
    out->BS = spec->block_size;
    out->N_BLOCKS_PER_TILE = rocke_attention_tiled_2d_spec_n_blocks_per_tile(spec);
    out->BLOCK_M = rocke_attention_tiled_2d_spec_block_m(spec);
    out->BLOCK_Q = rocke_attention_tiled_2d_spec_block_q(spec);
    out->NQK = rocke_attention_tiled_2d_spec_num_queries_per_kv(spec);
    out->NUM_KV = spec->num_kv_heads;
    out->NUM_QH = spec->num_query_heads;
    out->SLIDING_WINDOW = spec->sliding_window;
    out->USE_SOFTCAP = spec->has_softcap;
    out->USE_SINKS = spec->use_sinks;
    out->USE_ALIBI = spec->use_alibi;
    out->USE_QQ_BIAS = spec->use_qq_bias;

    out->KV_FP8 = rocke_streq(spec->kv_storage_dtype, "fp8e4m3");
    out->FP8_MFMA_QK = out->KV_FP8 && spec->use_fp8_mfma_qk;
    out->FP8_MFMA_PV = out->KV_FP8 && spec->use_fp8_mfma_pv;
    out->FP8_NATIVE_QK = false; /* documented dead path */
    out->KV_BYTES = out->KV_FP8 ? 1 : 2;

    out->USE_MFMA_32X32X8 = spec->use_mfma_32x32x8;
    out->USE_MFMA_32X32 = spec->use_mfma_32x32 || out->USE_MFMA_32X32X8;

    out->REGISTER_PV = spec->use_register_pv;
    out->TRANSPOSED_QK_32X32 = spec->use_transposed_qk_32x32;
    out->CONFLICT_FREE_V = spec->use_conflict_free_v;
    out->CONFLICT_FREE_V_STORE = spec->use_conflict_free_v_store;
    out->K_SINGLE_BUF = spec->use_k_single_buffer;

    out->dtype = rocke_attention_tiled_2d_spec_dtype_ir(spec);
    out->kv_io_dtype = out->KV_FP8 ? rocke_fp8e4m3() : out->dtype;

    return true;
}

/* ------------------------------------------------------ _C32_DIST (cached) */
/* make_static_tile_distribution(make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16()))
 * -- a host-side distribution the Python caches at module scope. Built lazily on
 * the first 32x32-C-helper call of a build.
 *
 * Re-entrancy: the cached rocke_tile_distribution_t (and all its inner nodes) is
 * arena-allocated off the *current build's* rocke_ir_builder. When that builder is
 * freed at the end of a build, this pointer dangles; reusing it on the next
 * build feeds freed memory into calculate_x. So the cache is per-build, not
 * process-lifetime: rocke_attn2d_c32_dist_reset() clears it at each build entry
 * (see the gfx942/gfx950 public entries). */
static const rocke_tile_distribution_t* g_c32_dist = NULL;

/* Re-entrancy reset: drop the dangling per-build cache before a new build. */
void rocke_attn2d_c32_dist_reset(void)
{
    g_c32_dist = NULL;
}

static const rocke_tile_distribution_t* rocke_attn2d_c32_dist(rocke_ir_builder_t* b)
{
    const rocke_mfma_atom_t* atom;
    const rocke_tile_distribution_encoding_t* enc;
    const rocke_tile_distribution_t* dist;

    if(g_c32_dist != NULL)
    {
        return g_c32_dist;
    }
    if(b == NULL)
    {
        return NULL;
    }

    atom = rocke_mfma_atom("f16", 32, 32, 16);
    if(atom == NULL)
    {
        rocke_attn2d_set_err(b, ROCKE_ERR_VALUE, "_C32_DIST: no f16 32x32x16 MFMA atom");
        return NULL;
    }
    enc = rocke_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
    {
        return NULL;
    }
    dist = rocke_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return NULL;
    }
    g_c32_dist = dist;
    return g_c32_dist;
}

/* ------------------------------------------------ _mfma_32x32_c_row / _col */

rocke_value_t* rocke__mfma_32x32_c_row(rocke_ir_builder_t* b, rocke_value_t* lane, int elem_idx)
{
    const rocke_tile_distribution_t* dist;
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2];

    if(b == NULL)
    {
        return NULL;
    }
    /* if not (0 <= elem_idx < 16): raise ValueError */
    if(!(elem_idx >= 0 && elem_idx < 16))
    {
        return (rocke_value_t*)rocke_attn2d_set_err(
            b, ROCKE_ERR_VALUE, "mfma_32x32x16 elem_idx must be 0..15, got %d", elem_idx);
    }

    dist = rocke_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));

    /* y0 = const(elem_idx // 4); y1 = const(elem_idx % 4) */
    ys[0] = rocke_b_const_i32(b, (int64_t)(elem_idx / 4));
    ys[1] = rocke_b_const_i32(b, (int64_t)(elem_idx % 4));

    /* ps=[[m_blk, n]] */
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    /* row, _col = _C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]]) */
    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row */
}

rocke_value_t* rocke__mfma_32x32_c_col(rocke_ir_builder_t* b, rocke_value_t* lane, int n_tile32)
{
    const rocke_tile_distribution_t* dist;
    rocke_value_t* m_blk;
    rocke_value_t* n;
    rocke_value_t* ys[2];
    rocke_value_t* ps0[2];
    rocke_value_t* const* ps[1];
    int ps_counts[1];
    rocke_value_t* out_x[2];
    rocke_value_t* col;

    if(b == NULL)
    {
        return NULL;
    }

    dist = rocke_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = rocke_b_div(b, lane, rocke_b_const_i32(b, 32));
    n = rocke_b_mod(b, lane, rocke_b_const_i32(b, 32));

    /* ys=[const(0), const(0)] */
    ys[0] = rocke_b_const_i32(b, 0);
    ys[1] = rocke_b_const_i32(b, 0);

    /* ps=[[m_blk, n]] */
    ps0[0] = m_blk;
    ps0[1] = n;
    ps[0] = ps0;
    ps_counts[0] = 2;

    /* _row, col = _C32_DIST.calculate_x(b, ys=[0, 0], ps=[[m_blk, n]]) */
    if(!rocke_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    col = out_x[1];

    /* if n_tile32 == 0: return col; else return add(const(n_tile32*32), col) */
    if(n_tile32 == 0)
    {
        return col;
    }
    return rocke_b_add(b, rocke_b_const_i32(b, (int64_t)(n_tile32 * 32)), col);
}

/* ------------------------------------------------- kernel build (stub) */

/* NOTE: the kernel build entry ``rocke_build_unified_attention_2d_tiled_scalar``
 * formerly lived here as a STUB-TO-LINK placeholder. The faithful, full
 * IR-emitting port now lives in the chunked instance part-files
 * (instance_gfx942_attention_tiled_2d_*_public_entry_glue.c drives the phase
 * functions). This TU keeps only the host-side spec/config/derivation helpers
 * and the two 32x32 C-row/C-col helpers that the part-files consume
 * (rocke__mfma_32x32_c_row / rocke__mfma_32x32_c_col); the build entry is defined
 * once, by the part-file, to avoid a duplicate-symbol link error. */
