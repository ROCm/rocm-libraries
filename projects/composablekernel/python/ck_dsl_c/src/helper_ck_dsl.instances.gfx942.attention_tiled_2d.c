/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_ck_dsl.instances.gfx942.attention_tiled_2d.c -- C99 port of the
 * task-named symbols from ck_dsl/instances/gfx942/attention_tiled_2d.py.
 *
 * Ported: UnifiedAttention2DTiledSpec (-> ckc_attention_tiled_2d_spec_t +
 * default/validate/derived-property helpers), the build-head config derivation
 * (-> ckc_unified_attention_2d_tiled_config_from_spec), _mfma_32x32_c_row /
 * _mfma_32x32_c_col, and the kernel build entry (stub-to-link).
 *
 * The two 32x32 C helpers reproduce the Python builder-call sequence
 * byte-faithfully: same div/mod constants, same calculate_x ys/ps wiring, same
 * trailing add for the N-tile base. _C32_DIST (built once at Python module
 * import from make_static_tile_distribution(make_c_warp_dstr_encoding(
 * MfmaAtom.f16_32x32x16()))) is reproduced as a lazily-built, process-lifetime
 * cached distribution off ckc_mfma_atom("f16", 32, 32, 16).
 *
 * Lifetime: every IR node is arena-owned (ckc_ir_builder_t.arena). Nothing is
 * freed individually. The cached _C32_DIST is built on the first call's builder
 * arena and is dtype-independent host-side analysis state (the Python caches it
 * at module scope). Because the C builder/arena is freed at the end of each
 * build, the cache is per-build (not process-lifetime): ckc_attn2d_c32_dist_reset()
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

#include "ckc/helper_helper_ck_dsl.instances.gfx942.attention_tiled_2d.h"
#include "ckc/ir.h"

/* ------------------------------------------------------------- error latch */

static void* ckc_attn2d_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* fmt, ...)
{
    if(b == NULL)
    {
        return NULL;
    }
    if(b->status == CKC_OK)
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(b->err, (size_t)CKC_ERR_MSG_CAP, fmt, ap);
        va_end(ap);
        b->status = st;
    }
    return NULL;
}

static bool ckc_attn2d_live(const ckc_ir_builder_t* b)
{
    return b != NULL && b->status == CKC_OK;
}

static bool ckc_streq(const char* a, const char* c)
{
    if(a == NULL || c == NULL)
    {
        return a == c;
    }
    return strcmp(a, c) == 0;
}

/* ------------------------------------------------------------- spec default */

ckc_attention_tiled_2d_spec_t ckc_attention_tiled_2d_spec_default(void)
{
    ckc_attention_tiled_2d_spec_t s;
    memset(&s, 0, sizeof(s));

    /* required fields stay zero/NULL until the caller sets them. */

    /* defaulted fields (mirror the dataclass defaults). */
    s.use_alibi                       = false;
    s.use_qq_bias                     = false;
    s.num_seqs                        = 0;
    s.num_warps                       = 1;
    s.has_waves_per_eu                = false;
    s.waves_per_eu                    = 0;
    s.kv_storage_dtype                = NULL;
    s.use_fp8_mfma_qk                 = false;
    s.use_fp8_mfma_pv                 = false;
    s.use_register_pv                 = false;
    s.has_tile_size                   = false;
    s.tile_size                       = 0;
    s.block_m_per_warp                = 16;
    s.use_mfma_32x32                  = false;
    s.use_mfma_32x32x8                = false;
    s.use_transposed_qk_32x32         = false;
    s.use_transposed_scalar_state     = false;
    s.use_transposed_invariant_hoist  = false;
    s.use_transposed_mask_once        = false;
    s.use_transposed_half_local_pv    = false;
    s.use_mfma32_skip_legacy_qreg     = false;
    s.use_transposed_mask_limit       = false;
    s.use_grouped_kv2_softmax         = false;
    s.use_fast_paged_kv_desc          = false;
    s.use_i64_kv_addr                 = false;
    s.use_early_v_schedule            = false;
    s.use_agpr_alloc_zero             = false;
    s.use_conflict_free_v             = false;
    s.use_conflict_free_v_store       = false;
    s.use_conflict_free_v_store_split = true;
    s.use_conflict_free_v_ck_vlds     = true;
    s.use_k_single_buffer             = false;
    s.use_k_sliced_ring               = false;
    s.use_k_sliced_ldsseq             = false;
    s.use_iglp_opt                    = false;
    s.use_q_direct_global             = false;
    s.kv_cache_policy                 = "stream";
    s.use_global_load_lds_k           = false;
    s.use_q_major_grid                = false;
    return s;
}

/* ------------------------------------------------- derived @property bodies */

int ckc_attention_tiled_2d_spec_num_queries_per_kv(const ckc_attention_tiled_2d_spec_t* s)
{
    if(s == NULL || s->num_kv_heads == 0)
    {
        return 0;
    }
    return s->num_query_heads / s->num_kv_heads;
}

int ckc_attention_tiled_2d_spec_block_m(const ckc_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    return s->block_m_per_warp * s->num_warps;
}

int ckc_attention_tiled_2d_spec_regs_per_lane(const ckc_attention_tiled_2d_spec_t* s)
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

int ckc_attention_tiled_2d_spec_block_q(const ckc_attention_tiled_2d_spec_t* s)
{
    int nqk;
    if(s == NULL)
    {
        return 0;
    }
    nqk = ckc_attention_tiled_2d_spec_num_queries_per_kv(s);
    if(nqk == 0)
    {
        return 0;
    }
    return ckc_attention_tiled_2d_spec_block_m(s) / nqk;
}

int ckc_attention_tiled_2d_spec_tile_size_eff(const ckc_attention_tiled_2d_spec_t* s)
{
    if(s == NULL)
    {
        return 0;
    }
    return s->has_tile_size ? s->tile_size : s->block_size;
}

int ckc_attention_tiled_2d_spec_n_blocks_per_tile(const ckc_attention_tiled_2d_spec_t* s)
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
    return ckc_attention_tiled_2d_spec_tile_size_eff(s) / bs;
}

const ckc_type_t* ckc_attention_tiled_2d_spec_dtype_ir(const ckc_attention_tiled_2d_spec_t* s)
{
    if(s != NULL && ckc_streq(s->dtype, "fp16"))
    {
        return ckc_f16();
    }
    return ckc_bf16();
}

int ckc_attention_tiled_2d_spec_binary_search_iters(const ckc_attention_tiled_2d_spec_t* s)
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
 * Returns false + latches CKC_ERR_VALUE on the first failing check. */
bool ckc_attention_tiled_2d_spec_validate(ckc_ir_builder_t* b, const ckc_attention_tiled_2d_spec_t* s)
{
    int block_m;
    int t_eff;

    if(!ckc_attn2d_live(b))
    {
        return false;
    }
    if(s == NULL)
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "attention_tiled_2d spec is NULL");
        return false;
    }

    /* gfx950-only experimental knobs rejected up front on gfx942. */
    if(s->use_mfma_32x32 || s->use_transposed_half_local_pv || s->use_mfma32_skip_legacy_qreg
       || s->use_grouped_kv2_softmax || s->use_agpr_alloc_zero || s->use_fp8_mfma_qk
       || s->use_fp8_mfma_pv)
    {
        ckc_attn2d_set_err(b,
                           CKC_ERR_VALUE,
                           "gfx942 tiled-2D attention supports only the narrow 16x16x16 "
                           "default path; gfx950-only knobs are not available on gfx942");
        return false;
    }

    /* transposed orientation legal only in the x8 pairing. */
    if(s->use_transposed_qk_32x32 && !s->use_mfma_32x32x8)
    {
        ckc_attn2d_set_err(b,
                           CKC_ERR_VALUE,
                           "gfx942: use_transposed_qk_32x32 requires use_mfma_32x32x8");
        return false;
    }

    /* fp8 K/V cache is gfx950-only here. */
    if(s->kv_storage_dtype != NULL)
    {
        ckc_attn2d_set_err(b,
                           CKC_ERR_VALUE,
                           "gfx942 tiled-2D attention has no fp8 K/V cache path "
                           "(kv_storage_dtype must be None on gfx942)");
        return false;
    }

    if(!(s->num_warps == 1 || s->num_warps == 2 || s->num_warps == 4 || s->num_warps == 8))
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "num_warps must be in {1, 2, 4, 8}");
        return false;
    }

    if(!(s->block_m_per_warp == 16 || s->block_m_per_warp == 32))
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "block_m_per_warp must be in {16, 32}");
        return false;
    }

    t_eff = ckc_attention_tiled_2d_spec_tile_size_eff(s);

    if(s->use_mfma_32x32x8)
    {
        if(s->use_mfma_32x32)
        {
            ckc_attn2d_set_err(
                b, CKC_ERR_VALUE, "use_mfma_32x32x8 and use_mfma_32x32 are mutually exclusive");
            return false;
        }
        if(!ckc_streq(s->dtype, "fp16"))
        {
            ckc_attn2d_set_err(b, CKC_ERR_VALUE, "gfx942 use_mfma_32x32x8 is fp16-only");
            return false;
        }
        if(s->block_m_per_warp != 32)
        {
            ckc_attn2d_set_err(b, CKC_ERR_VALUE, "use_mfma_32x32x8 requires block_m_per_warp=32");
            return false;
        }
        if(t_eff % 32 != 0)
        {
            ckc_attn2d_set_err(
                b, CKC_ERR_VALUE, "use_mfma_32x32x8 requires tile_size_eff %% 32 == 0");
            return false;
        }
        if(s->head_size % 32 != 0)
        {
            ckc_attn2d_set_err(b, CKC_ERR_VALUE, "use_mfma_32x32x8 requires head_size %% 32 == 0");
            return false;
        }
    }

    /* transposed sub-knob dependencies. */
    if(s->use_transposed_scalar_state && !s->use_transposed_qk_32x32)
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_transposed_scalar_state requires use_transposed_qk_32x32");
        return false;
    }
    if(s->use_transposed_invariant_hoist && !s->use_transposed_qk_32x32)
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_transposed_invariant_hoist requires use_transposed_qk_32x32");
        return false;
    }
    if(s->use_transposed_mask_once && !s->use_transposed_qk_32x32)
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_transposed_mask_once requires use_transposed_qk_32x32");
        return false;
    }

    /* conflict-free V feed requires the transposed-x8 orientation. */
    if(s->use_conflict_free_v && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_conflict_free_v requires the transposed-x8 path");
        return false;
    }
    if(s->use_conflict_free_v_store && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_conflict_free_v_store requires the transposed-x8 path");
        return false;
    }
    if(s->use_conflict_free_v_store && s->use_conflict_free_v)
    {
        ckc_attn2d_set_err(b,
                           CKC_ERR_VALUE,
                           "use_conflict_free_v_store and use_conflict_free_v are "
                           "mutually exclusive");
        return false;
    }

    /* K single-buffer occupancy lever. */
    if(s->use_k_single_buffer)
    {
        if(!(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
        {
            ckc_attn2d_set_err(
                b, CKC_ERR_VALUE, "use_k_single_buffer requires the transposed-x8 path");
            return false;
        }
        if(!ckc_streq(s->dtype, "fp16"))
        {
            ckc_attn2d_set_err(b, CKC_ERR_VALUE, "use_k_single_buffer is fp16-only");
            return false;
        }
        block_m = ckc_attention_tiled_2d_spec_block_m(s);
        if(block_m > t_eff)
        {
            ckc_attn2d_set_err(
                b, CKC_ERR_VALUE, "use_k_single_buffer requires BLOCK_M <= tile_size_eff");
            return false;
        }
    }

    /* K sliced ring + ldsseq dependencies. */
    if(s->use_k_sliced_ring && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32
                                 && s->use_conflict_free_v_store))
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_k_sliced_ring requires the transposed-x8 cfvst path");
        return false;
    }
    if(s->use_k_sliced_ldsseq && !s->use_k_sliced_ring)
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "use_k_sliced_ldsseq requires use_k_sliced_ring");
        return false;
    }

    if(s->use_q_direct_global && !(s->use_mfma_32x32x8 && s->use_transposed_qk_32x32))
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "use_q_direct_global currently targets transposed-x8");
        return false;
    }

    if(s->num_warps == 8 && s->block_m_per_warp == 32
       && !(s->use_q_direct_global && s->use_conflict_free_v_store))
    {
        ckc_attn2d_set_err(b,
                           CKC_ERR_VALUE,
                           "num_warps=8 with block_m_per_warp=32 requires "
                           "use_q_direct_global + use_conflict_free_v_store");
        return false;
    }

    if(!(ckc_streq(s->kv_cache_policy, "all") || ckc_streq(s->kv_cache_policy, "global")
         || ckc_streq(s->kv_cache_policy, "stream") || ckc_streq(s->kv_cache_policy, "nt")))
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "kv_cache_policy must be one of {all, global, stream, nt}");
        return false;
    }

    if(s->use_global_load_lds_k && s->kv_storage_dtype != NULL)
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_global_load_lds_k v1 supports bf16/fp16 KV only");
        return false;
    }

    if(s->use_mfma32_skip_legacy_qreg && !s->use_mfma_32x32)
    {
        ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "use_mfma32_skip_legacy_qreg requires use_mfma_32x32");
        return false;
    }

    return true;
}

/* --------------------------------------------- config-from-spec (build head) */

bool ckc_unified_attention_2d_tiled_config_from_spec(ckc_ir_builder_t* b,
                                                     const ckc_attention_tiled_2d_spec_t* spec,
                                                     ckc_unified_attention_2d_tiled_config_t* out)
{
    if(!ckc_attn2d_live(b))
    {
        return false;
    }
    if(spec == NULL || out == NULL)
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "config_from_spec: NULL spec/out");
        return false;
    }

    /* __post_init__ runs at dataclass construction; reproduce it here. */
    if(!ckc_attention_tiled_2d_spec_validate(b, spec))
    {
        return false;
    }

    /* dtype gate (Python NotImplementedError). */
    if(!(ckc_streq(spec->dtype, "fp16") || ckc_streq(spec->dtype, "bf16")))
    {
        ckc_attn2d_set_err(b, CKC_ERR_NOTIMPL, "tiled 2D kernel supports fp16/bf16");
        return false;
    }

    memset(out, 0, sizeof(*out));

    out->HD                = spec->head_size;
    out->T                 = ckc_attention_tiled_2d_spec_tile_size_eff(spec);
    out->BS                = spec->block_size;
    out->N_BLOCKS_PER_TILE = ckc_attention_tiled_2d_spec_n_blocks_per_tile(spec);
    out->BLOCK_M           = ckc_attention_tiled_2d_spec_block_m(spec);
    out->BLOCK_Q           = ckc_attention_tiled_2d_spec_block_q(spec);
    out->NQK               = ckc_attention_tiled_2d_spec_num_queries_per_kv(spec);
    out->NUM_KV            = spec->num_kv_heads;
    out->NUM_QH            = spec->num_query_heads;
    out->SLIDING_WINDOW    = spec->sliding_window;
    out->USE_SOFTCAP       = spec->has_softcap;
    out->USE_SINKS         = spec->use_sinks;
    out->USE_ALIBI         = spec->use_alibi;
    out->USE_QQ_BIAS       = spec->use_qq_bias;

    out->KV_FP8        = ckc_streq(spec->kv_storage_dtype, "fp8e4m3");
    out->FP8_MFMA_QK   = out->KV_FP8 && spec->use_fp8_mfma_qk;
    out->FP8_MFMA_PV   = out->KV_FP8 && spec->use_fp8_mfma_pv;
    out->FP8_NATIVE_QK = false; /* documented dead path */
    out->KV_BYTES      = out->KV_FP8 ? 1 : 2;

    out->USE_MFMA_32X32X8 = spec->use_mfma_32x32x8;
    out->USE_MFMA_32X32   = spec->use_mfma_32x32 || out->USE_MFMA_32X32X8;

    out->REGISTER_PV          = spec->use_register_pv;
    out->TRANSPOSED_QK_32X32  = spec->use_transposed_qk_32x32;
    out->CONFLICT_FREE_V      = spec->use_conflict_free_v;
    out->CONFLICT_FREE_V_STORE = spec->use_conflict_free_v_store;
    out->K_SINGLE_BUF         = spec->use_k_single_buffer;

    out->dtype = ckc_attention_tiled_2d_spec_dtype_ir(spec);
    out->kv_io_dtype = out->KV_FP8 ? ckc_fp8e4m3() : out->dtype;

    return true;
}

/* ------------------------------------------------------ _C32_DIST (cached) */
/* make_static_tile_distribution(make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16()))
 * -- a host-side distribution the Python caches at module scope. Built lazily on
 * the first 32x32-C-helper call of a build.
 *
 * Re-entrancy: the cached ckc_tile_distribution_t (and all its inner nodes) is
 * arena-allocated off the *current build's* ckc_ir_builder. When that builder is
 * freed at the end of a build, this pointer dangles; reusing it on the next
 * build feeds freed memory into calculate_x. So the cache is per-build, not
 * process-lifetime: ckc_attn2d_c32_dist_reset() clears it at each build entry
 * (see the gfx942/gfx950 public entries). */
static const ckc_tile_distribution_t* g_c32_dist = NULL;

/* Re-entrancy reset: drop the dangling per-build cache before a new build. */
void ckc_attn2d_c32_dist_reset(void)
{
    g_c32_dist = NULL;
}

static const ckc_tile_distribution_t* ckc_attn2d_c32_dist(ckc_ir_builder_t* b)
{
    const ckc_mfma_atom_t* atom;
    const ckc_tile_distribution_encoding_t* enc;
    const ckc_tile_distribution_t* dist;

    if(g_c32_dist != NULL)
    {
        return g_c32_dist;
    }
    if(!ckc_attn2d_live(b))
    {
        return NULL;
    }

    atom = ckc_mfma_atom("f16", 32, 32, 16);
    if(atom == NULL)
    {
        ckc_attn2d_set_err(b, CKC_ERR_VALUE, "_C32_DIST: no f16 32x32x16 MFMA atom");
        return NULL;
    }
    enc = ckc_make_c_warp_dstr_encoding(b, atom);
    if(enc == NULL)
    {
        return NULL;
    }
    dist = ckc_make_static_tile_distribution(b, enc);
    if(dist == NULL)
    {
        return NULL;
    }
    g_c32_dist = dist;
    return g_c32_dist;
}

/* ------------------------------------------------ _mfma_32x32_c_row / _col */

ckc_value_t* ckc__mfma_32x32_c_row(ckc_ir_builder_t* b, ckc_value_t* lane, int elem_idx)
{
    const ckc_tile_distribution_t* dist;
    ckc_value_t* m_blk;
    ckc_value_t* n;
    ckc_value_t* ys[2];
    ckc_value_t* ps0[2];
    ckc_value_t* const* ps[1];
    int ps_counts[1];
    ckc_value_t* out_x[2];

    if(!ckc_attn2d_live(b))
    {
        return NULL;
    }
    /* if not (0 <= elem_idx < 16): raise ValueError */
    if(!(elem_idx >= 0 && elem_idx < 16))
    {
        return ckc_attn2d_set_err(
            b, CKC_ERR_VALUE, "mfma_32x32x16 elem_idx must be 0..15, got %d", elem_idx);
    }

    dist = ckc_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = ckc_b_div(b, lane, ckc_b_const_i32(b, 32));
    n     = ckc_b_mod(b, lane, ckc_b_const_i32(b, 32));

    /* y0 = const(elem_idx // 4); y1 = const(elem_idx % 4) */
    ys[0] = ckc_b_const_i32(b, (int64_t)(elem_idx / 4));
    ys[1] = ckc_b_const_i32(b, (int64_t)(elem_idx % 4));

    /* ps=[[m_blk, n]] */
    ps0[0]       = m_blk;
    ps0[1]       = n;
    ps[0]        = ps0;
    ps_counts[0] = 2;

    /* row, _col = _C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]]) */
    if(!ckc_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    return out_x[0]; /* row */
}

ckc_value_t* ckc__mfma_32x32_c_col(ckc_ir_builder_t* b, ckc_value_t* lane, int n_tile32)
{
    const ckc_tile_distribution_t* dist;
    ckc_value_t* m_blk;
    ckc_value_t* n;
    ckc_value_t* ys[2];
    ckc_value_t* ps0[2];
    ckc_value_t* const* ps[1];
    int ps_counts[1];
    ckc_value_t* out_x[2];
    ckc_value_t* col;

    if(!ckc_attn2d_live(b))
    {
        return NULL;
    }

    dist = ckc_attn2d_c32_dist(b);
    if(dist == NULL)
    {
        return NULL;
    }

    /* m_blk = b.div(lane, 32); n = b.mod(lane, 32) */
    m_blk = ckc_b_div(b, lane, ckc_b_const_i32(b, 32));
    n     = ckc_b_mod(b, lane, ckc_b_const_i32(b, 32));

    /* ys=[const(0), const(0)] */
    ys[0] = ckc_b_const_i32(b, 0);
    ys[1] = ckc_b_const_i32(b, 0);

    /* ps=[[m_blk, n]] */
    ps0[0]       = m_blk;
    ps0[1]       = n;
    ps[0]        = ps0;
    ps_counts[0] = 2;

    /* _row, col = _C32_DIST.calculate_x(b, ys=[0, 0], ps=[[m_blk, n]]) */
    if(!ckc_tile_distribution_calculate_x(b, dist, ys, 2, ps, ps_counts, 1, out_x, 2))
    {
        return NULL;
    }
    col = out_x[1];

    /* if n_tile32 == 0: return col; else return add(const(n_tile32*32), col) */
    if(n_tile32 == 0)
    {
        return col;
    }
    return ckc_b_add(b, ckc_b_const_i32(b, (int64_t)(n_tile32 * 32)), col);
}

/* ------------------------------------------------- kernel build (stub) */

/* NOTE: the kernel build entry ``ckc_build_unified_attention_2d_tiled_scalar``
 * formerly lived here as a STUB-TO-LINK placeholder. The faithful, full
 * IR-emitting port now lives in the chunked instance part-files
 * (instance_gfx942_attention_tiled_2d_*_public_entry_glue.c drives the phase
 * functions). This TU keeps only the host-side spec/config/derivation helpers
 * and the two 32x32 C-row/C-col helpers that the part-files consume
 * (ckc__mfma_32x32_c_row / ckc__mfma_32x32_c_col); the build entry is defined
 * once, by the part-file, to avoid a duplicate-symbol link error. */
