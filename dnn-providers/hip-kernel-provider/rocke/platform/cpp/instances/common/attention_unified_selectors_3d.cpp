// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_unified_selectors_3d.cpp -- C port of the tiled 3D (split-KV)
 * selectors of library/kernels/common/attention_unified.py and of
 * attention_spec_builder._spec_generic_3d. See the header for the conventions.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"
#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors_3d.h"
#include "rocke/helper_rocke.helpers.attention.h" /* rocke_attention_next_power_of_2 */

/* Reference CU count of the segment clamp (Python _PRE_BUMP_CUS). */
#define ROCKE_ATTN_3D_PRE_BUMP_CUS 120

static bool arch_is(const char* arch, const char* want)
{
    return arch != NULL && strcmp(arch, want) == 0;
}

static int imin(int a, int b)
{
    return a < b ? a : b;
}

static int imax(int a, int b)
{
    return a > b ? a : b;
}

static bool fp16_bf16_kv(const rocke_unified_attention_problem_t* p)
{
    return rocke_unified_attn_kv_storage_dtype(p) == NULL;
}

int rocke_unified_attn_pre_bump_segments(const rocke_unified_attention_problem_t* p)
{
    int num_2d
        = rocke_unified_attention_problem_total_num_q_blocks_upper_bound(p) * p->num_kv_heads;
    int min_seg = p->block_size <= 16 ? 16 : 8;

    if(num_2d <= 0)
    {
        return 0;
    }
    return imax(imin(rocke_attention_next_power_of_2((ROCKE_ATTN_3D_PRE_BUMP_CUS * 4 + num_2d - 1)
                                                     / num_2d),
                     128),
                min_seg);
}

bool rocke_unified_attn_gfx942_3d_decode_ctas_per_segment(
    const rocke_unified_attention_problem_t* p, const char* arch, int* ctas)
{
    int n = p->num_seqs * p->num_kv_heads;
    if(arch_is(arch, "gfx942") && rocke_unified_attention_problem_all_decode(p)
       && (p->head_size == 64 || p->head_size == 128) && p->block_size == 16
       && p->sliding_window == 0 && p->target_ctas <= 0 && fp16_bf16_kv(p)
       && (n * 16 >= p->num_cus || p->max_seqlen_k < 8192))
    {
        if(ctas != NULL)
        {
            *ctas = n;
        }
        return true;
    }
    return false;
}

bool rocke_unified_attn_gfx950_3d_decode_ctas_per_segment(
    const rocke_unified_attention_problem_t* p, const char* arch, int* ctas)
{
    int n = p->num_seqs * p->num_kv_heads;
    if(arch_is(arch, "gfx950") && rocke_unified_attention_problem_all_decode(p)
       && (p->head_size == 128 || (p->head_size == 256 && n * 32 > p->num_cus))
       && p->block_size == 16 && p->sliding_window == 0 && p->target_ctas <= 0 && fp16_bf16_kv(p))
    {
        if(ctas != NULL)
        {
            *ctas = n;
        }
        return true;
    }
    return false;
}

int rocke_unified_attn_num_segments(const rocke_unified_attention_problem_t* p, const char* arch)
{
    rocke_attention_3d_config_t attn;
    rocke_attention_3d_config_t reduce;
    int segments;
    int ctas;

    if(!rocke_unified_attention_problem_select_3d(p, &attn, &reduce))
    {
        return 0;
    }
    segments = attn.num_segments_per_seq;

    if(arch_is(arch, "gfx942") && p->sliding_window == 0)
    {
        int pre_bump;
        /* Decode split rule: about 4 CTAs per CU (6 at head size 64). */
        if(rocke_unified_attn_gfx942_3d_decode_ctas_per_segment(p, arch, &ctas))
        {
            int per_cu = p->head_size == 64 ? 6 : 4;
            int cap = p->max_seqlen_k < 8192 ? 64 : 128;
            if(ctas <= 0)
            {
                return 0;
            }
            return imax(2, imin(per_cu * p->num_cus / ctas, cap));
        }
        pre_bump = rocke_unified_attn_pre_bump_segments(p);
        if(p->max_seqlen_q == 1)
        {
            if(p->max_seqlen_k <= 2048)
            {
                if(p->head_size == 64)
                {
                    return imin(segments, 32);
                }
                if(p->head_size == 128)
                {
                    return imin(segments, 16);
                }
                if(p->head_size != 256)
                {
                    return imin(segments, 64);
                }
            }
            else if(p->head_size == 128 && p->max_seqlen_k < 32768)
            {
                return imin(segments, pre_bump);
            }
        }
        else
        {
            return imin(segments, pre_bump);
        }
    }
    if(arch_is(arch, "gfx950") && p->sliding_window == 0 && p->target_ctas <= 0)
    {
        /* Decode split rule; every other shape is clamped to the 120-CU split. */
        if(rocke_unified_attn_gfx950_3d_decode_ctas_per_segment(p, arch, &ctas))
        {
            int cus = p->num_cus;
            int kv = p->max_seqlen_k;
            if(ctas <= 0)
            {
                return 0;
            }
            if(ctas * 32 <= cus)
            {
                int n = (kv >= 65536 && ctas * 64 <= cus) ? 128 : 64;
                return imax(1, imin(n, kv / 64));
            }
            return imax(2, imin(4 * cus / ctas, kv < 8192 ? 64 : 128));
        }
        return imin(segments, rocke_unified_attn_pre_bump_segments(p));
    }
    return segments;
}

bool rocke_unified_attn_gfx942_3d_tile_size_override(const rocke_unified_attention_problem_t* p,
                                                     const char* arch,
                                                     int* tile_size)
{
    if(!(arch_is(arch, "gfx942") && p->head_size >= 128 && p->block_size >= 32))
    {
        return false;
    }
    if(tile_size != NULL)
    {
        *tile_size = p->block_size / 2;
    }
    return true;
}

bool rocke_unified_attn_gfx950_3d_tile_size_override(const rocke_unified_attention_problem_t* p,
                                                     const char* arch,
                                                     int* tile_size)
{
    int ctas;
    /* Decode too thin to fill the device even at 64 segments: 64-token tiles. */
    if(rocke_unified_attn_gfx950_3d_decode_ctas_per_segment(p, arch, &ctas)
       && ctas * 32 <= p->num_cus)
    {
        if(tile_size != NULL)
        {
            *tile_size = 64;
        }
        return true;
    }
    return false;
}

bool rocke_unified_attn_select_3d_waves_per_eu(const rocke_unified_attention_problem_t* p,
                                               const char* arch,
                                               int* waves_per_eu)
{
    int wpe;
    if(p->waves_per_eu_set)
    {
        wpe = p->waves_per_eu;
    }
    else if(arch_is(arch, "gfx1250"))
    {
        wpe = 2;
    }
    else
    {
        return false;
    }
    if(waves_per_eu != NULL)
    {
        *waves_per_eu = wpe;
    }
    return true;
}

bool rocke_unified_attn_enable_gfx942_3d_wide_kv_load(const rocke_unified_attention_problem_t* p,
                                                      const char* arch)
{
    if(!arch_is(arch, "gfx942"))
    {
        return false;
    }
    /* The fp8 path already loads wide. */
    if(!fp16_bf16_kv(p))
    {
        return false;
    }
    if(rocke_unified_attn_env_is_off("HIPDNN_GFX942_3D_WKV"))
    {
        return false;
    }
    /* At head size 256 the wide feed's loads serialize under some compilers. */
    return p->head_size <= 128;
}

bool rocke_unified_attn_enable_i64_kv_addr_3d(const rocke_unified_attention_problem_t* p,
                                              const char* arch)
{
    if(!arch_is(arch, "gfx942") || !rocke_unified_attn_enable_i64_kv_addr(p))
    {
        return rocke_unified_attn_enable_i64_kv_addr(p);
    }
    /* gfx942 tiled 3D has no i64 path, so the limit is what its K/V loader
     * reads (Python i32_kv_cache_bytes): the fp8 dequant and wide loaders are
     * flat loads indexed by an i32 byte offset, correct below 2^31 bytes; the
     * async copies read through a buffer bounded at 0x7FFF0000 bytes. The wide
     * loader only takes effect when one 64-lane pass of 8-element loads tiles
     * the T x HD tile (Python wide_kv_load_fits). */
    int tile_size = p->block_size;
    rocke_unified_attn_gfx942_3d_tile_size_override(p, arch, &tile_size);
    const bool wide = rocke_unified_attn_enable_gfx942_3d_wide_kv_load(p, arch)
                      && (tile_size * p->head_size) % (64 * 8) == 0;
    const bool flat = rocke_unified_attn_kv_storage_dtype(p) != NULL || wide;
    const uint64_t limit = flat ? 0x80000000ULL : 0x7FFF0000ULL;
    const uint64_t block_bytes = (uint64_t)p->block_size * (uint64_t)p->num_kv_heads
                                 * (uint64_t)p->head_size * (uint64_t)(p->use_fp8 ? 1 : 2);
    return (uint64_t)p->num_kv_blocks * block_bytes > limit;
}

bool rocke_unified_attn_use_decode_grid(const rocke_unified_attention_problem_t* p,
                                        const char* arch)
{
    return rocke_unified_attention_problem_all_decode(p)
           && (arch_is(arch, "gfx942") || arch_is(arch, "gfx950"));
}

const char* rocke_unified_attn_kv_cache_policy_3d(const rocke_unified_attention_problem_t* p,
                                                  const char* arch)
{
    /* All-decode reads each K/V byte once: stream it past the caches. */
    if((arch_is(arch, "gfx942") || arch_is(arch, "gfx950"))
       && rocke_unified_attention_problem_all_decode(p) && fp16_bf16_kv(p))
    {
        return "stream";
    }
    return "all";
}

bool rocke_unified_attn_wave_page_id_load_3d(const rocke_unified_attention_problem_t* p,
                                             const char* arch)
{
    if(arch_is(arch, "gfx950"))
    {
        return rocke_unified_attn_gfx950_3d_decode_ctas_per_segment(p, arch, NULL);
    }
    if(arch_is(arch, "gfx942"))
    {
        return rocke_unified_attn_kv_lds_swizzle_3d(p, arch)
               && !rocke_unified_attn_gfx942_3d_tile_size_override(p, arch, NULL);
    }
    return false;
}

bool rocke_unified_attn_kv_lds_swizzle_3d(const rocke_unified_attention_problem_t* p,
                                          const char* arch)
{
    if(arch_is(arch, "gfx950"))
    {
        return rocke_unified_attn_gfx950_3d_tile_size_override(p, arch, NULL);
    }
    if(arch_is(arch, "gfx942"))
    {
        if(!rocke_unified_attention_problem_all_decode(p))
        {
            return false;
        }
        if(p->head_size == 64 || p->head_size == 128)
        {
            return rocke_unified_attn_enable_gfx942_3d_wide_kv_load(p, arch);
        }
        return p->head_size == 256 && fp16_bf16_kv(p)
               && !rocke_unified_attn_gfx942_3d_tile_size_override(p, arch, NULL);
    }
    return false;
}

rocke_unified_attention_3d_tiled_spec_t
    rocke_unified_attn_tiled_3d_spec_from_problem(const rocke_unified_attention_problem_t* p,
                                                  const char* arch)
{
    rocke_unified_attention_3d_tiled_spec_t s = rocke_unified_attention_3d_tiled_spec_default();
    int v;

    if(arch == NULL || arch_is(arch, "gfx1250"))
    {
        return s;
    }
    s.head_size = p->head_size;
    s.block_size = p->block_size;
    s.num_query_heads = p->num_query_heads;
    s.num_kv_heads = p->num_kv_heads;
    s.dtype = p->dtype;
    s.use_sinks = p->use_sinks;
    s.sliding_window = p->sliding_window;
    s.has_softcap = p->softcap > 0;
    s.num_segments = rocke_unified_attn_num_segments(p, arch);
    s.use_alibi = p->use_alibi;
    s.use_qq_bias = p->use_qq_bias;
    s.num_seqs = p->num_seqs;
    if(rocke_unified_attn_select_3d_waves_per_eu(p, arch, &v))
    {
        s.has_waves_per_eu = true;
        s.waves_per_eu = v;
    }
    s.kv_storage_dtype = rocke_unified_attn_kv_storage_dtype(p);
    if(arch_is(arch, "gfx950") ? rocke_unified_attn_gfx950_3d_tile_size_override(p, arch, &v)
                               : rocke_unified_attn_gfx942_3d_tile_size_override(p, arch, &v))
    {
        s.has_tile_size_override = true;
        s.tile_size_override = v;
    }
    s.use_wide_kv_load = rocke_unified_attn_enable_gfx942_3d_wide_kv_load(p, arch);
    s.use_i64_kv_addr = rocke_unified_attn_enable_i64_kv_addr_3d(p, arch);
    s.use_decode_grid = rocke_unified_attn_use_decode_grid(p, arch);
    if(arch_is(arch, "gfx942") || arch_is(arch, "gfx950"))
    {
        s.kv_cache_policy = rocke_unified_attn_kv_cache_policy_3d(p, arch);
        s.use_kv_lds_swizzle = rocke_unified_attn_kv_lds_swizzle_3d(p, arch);
        s.use_wave_page_id_load = rocke_unified_attn_wave_page_id_load_3d(p, arch);
    }
    /* Only gfx950 implements the band: the gfx942 spec validator rejects a
     * non-default one (Python's gfx942 spec class has no such field). */
    s.causal_top_left = p->causal_top_left;
    s.right_bound = p->right_bound;
    return s;
}
