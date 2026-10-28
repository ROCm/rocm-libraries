// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * attention_unified_gates.cpp -- coverage gates of
 * rocke/instances/common/attention_unified.py:
 *
 *   _canonical_arch_for_selection          rocke_unified_attention_canonical_arch
 *   _mask_label                            rocke_attn_unified_mask_label
 *   _reject_mask_semantics                 rocke_attn_unified_reject_mask_semantics
 *   _reject_mask_on_arch                   (static)
 *   _reject_fp8_format_arch_mismatch       rocke_attn_unified_reject_fp8_format_arch_mismatch
 *   supports_native_unified_attention_tiled
 *                                          rocke_unified_attention_supports_tiled_2d
 *   supports_native_unified_attention_3d_tiled
 *                                          rocke_unified_attention_supports_tiled_3d
 *
 * Host-only; emits no IR.
 */

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"
#include "rocke/instance_attention_unified.h"
#include "rocke/instance_attention_unified_internal.h"
#include "rocke/instance_gfx942_attention_tiled_2d.h"
#include "rocke/instance_gfx942_attention_tiled_3d.h"
#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/instance_gfx950_attention_tiled_3d.h"

#define ROCKE_ATTN_GATE_ARCH_CAP 64

static void gate_reason(char* reason, size_t reason_cap, const char* fmt, ...)
{
    if(reason == NULL || reason_cap == 0)
        return;
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(reason, reason_cap, fmt, ap);
    va_end(ap);
}

static bool streq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

const char* rocke_unified_attention_canonical_arch(const char* arch, char* out, size_t out_cap)
{
    if(out == NULL || out_cap == 0)
        return NULL;
    if(arch == NULL || arch[0] == '\0')
    {
        snprintf(out, out_cap, "%s", rocke_unified_attn_resolve_arch());
        return out;
    }
    /* arch.strip().lower() */
    char lowered[ROCKE_ATTN_GATE_ARCH_CAP];
    const char* begin = arch;
    const char* end = arch + strlen(arch);
    while(begin < end && isspace((unsigned char)*begin))
        ++begin;
    while(end > begin && isspace((unsigned char)end[-1]))
        --end;
    size_t n = 0;
    for(; begin < end && n + 1 < sizeof(lowered); ++begin)
        lowered[n++] = (char)tolower((unsigned char)*begin);
    lowered[n] = '\0';
    return rocke_base_arch_from_target_id(lowered, out, out_cap);
}

void rocke_attn_unified_mask_label(const rocke_unified_attention_problem_t* p,
                                   char* out,
                                   size_t out_cap)
{
    if(out == NULL || out_cap == 0)
        return;
    const char* align = p->causal_top_left ? "top-left" : "bottom-right";
    if(p->right_bound == -1)
        snprintf(out, out_cap, "%s/unbounded", align);
    else if(p->right_bound == 0)
        snprintf(out, out_cap, "%s/causal", align);
    else
        snprintf(out, out_cap, "%s/right_bound=%d", align, p->right_bound);
}

bool rocke_attn_unified_reject_mask_semantics(const rocke_unified_attention_problem_t* p,
                                              char* reason,
                                              size_t reason_cap)
{
    if(p->right_bound != 0 && (p->use_alibi || p->use_qq_bias))
    {
        gate_reason(reason, reason_cap, "ALiBi / QQ-bias are only defined for causal masks");
        return true;
    }
    return false;
}

static bool reject_mask_on_arch(const rocke_unified_attention_problem_t* p,
                                const char* arch,
                                char* reason,
                                size_t reason_cap)
{
    if(!rocke_unified_attention_problem_default_mask(p) && !streq(arch, "gfx950"))
    {
        char label[64];
        rocke_attn_unified_mask_label(p, label, sizeof(label));
        gate_reason(
            reason, reason_cap, "%s tiled attention does not implement %s masks", arch, label);
        return true;
    }
    return false;
}

bool rocke_attn_unified_reject_fp8_format_arch_mismatch(const rocke_unified_attention_problem_t* p,
                                                        const char* arch,
                                                        char* reason,
                                                        size_t reason_cap)
{
    if(!p->use_fp8)
        return false;
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return true;
    }
    /* _FNUZ_FP8_TARGET_FAMILIES = {"gfx9_mfma"} */
    bool arch_is_fnuz = streq(target->target_family, "gfx9_mfma");
    if(p->fp8_fnuz == arch_is_fnuz)
        return false;
    gate_reason(reason,
                reason_cap,
                "fp8 K/V on %s (target_family='%s') decodes as %s, but the problem declares "
                "%s-quantised K/V (fp8_fnuz=%s); the mismatch would silently mis-decode K/V. "
                "Quantise K/V to the arch-native format and set fp8_fnuz to match, or run on "
                "an arch whose native fp8 format matches the data.",
                arch,
                target->target_family,
                arch_is_fnuz ? "e4m3fnuz/e5m2fnuz" : "OCP e4m3fn/e5m2",
                p->fp8_fnuz ? "fnuz" : "OCP",
                p->fp8_fnuz ? "True" : "False");
    return true;
}

/* Python admits these; the C++ builders cannot build them yet. Checked only
 * after the per-arch gate admits, so every Python reject keeps its reason. */
static bool reject_tiled_2d_not_ported(const rocke_unified_attention_problem_t* p,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap)
{
    if(streq(arch, "gfx942"))
    {
        if(rocke_unified_attn_gfx942_4warp_fast(p, arch))
        {
            gate_reason(reason, reason_cap, "gfx942 4-warp GQA tiled 2D is not ported to C++ yet");
            return true;
        }
        if(rocke_unified_attn_enable_gfx942_flash_k_sliced_ring(p, arch))
        {
            gate_reason(
                reason, reason_cap, "gfx942 tiled 2D sliced-K ring is not ported to C++ yet");
            return true;
        }
        if(rocke_unified_attn_enable_gfx942_flash_mask_limit(p, arch))
        {
            gate_reason(
                reason, reason_cap, "gfx942 tiled 2D flash mask-limit is not ported to C++ yet");
            return true;
        }
        return false;
    }
    if(p->use_additive_bias)
    {
        gate_reason(reason, reason_cap, "tiled 2D additive bias is not ported to C++ yet");
        return true;
    }
    if(!rocke_unified_attention_problem_default_mask(p))
    {
        char label[64];
        rocke_attn_unified_mask_label(p, label, sizeof(label));
        gate_reason(reason, reason_cap, "tiled 2D %s masks are not ported to C++ yet", label);
        return true;
    }
    return false;
}

/* The bf16 wide-K (32x32x8) transposed flash branch: the build geometry of
 * _tiled_spec_from_problem's non-ring bf16 spec. */
static bool supports_gfx942_bf16_wide(const rocke_unified_attention_problem_t* p,
                                      const char* arch,
                                      int nqpk,
                                      char* reason,
                                      size_t reason_cap)
{
    rocke_gfx942_attention_tiled_2d_supports_args_t a
        = rocke_gfx942_attention_tiled_2d_supports_args_default();
    a.head_size = p->head_size;
    a.block_size = p->block_size;
    a.dtype = p->dtype;
    a.num_queries_per_kv = nqpk;
    a.use_alibi = p->use_alibi;
    a.use_qq_bias = p->use_qq_bias;
    a.use_fp8 = p->use_fp8;
    a.q_dtype = p->q_dtype;
    rocke_unified_attn_gfx942_bf16_wide_geometry(p, arch, &a.num_warps, &a.use_k_single_buffer);
    a.block_m_per_warp = 32;
    a.kv_storage_dtype = rocke_unified_attn_kv_storage_dtype(p);
    a.has_tile_size = true;
    a.tile_size = rocke_unified_attn_gfx942_bf16_wide_tile_size(p, arch);
    a.arch = arch;
    a.use_mfma_32x32x8 = true;
    a.use_transposed_qk_32x32 = true;
    a.use_conflict_free_v_store = rocke_unified_attn_gfx942_bf16_wide_use_cfvst(p);
    a.ring_depth = rocke_unified_attn_select_gfx942_flash_ring_depth(p, arch);
    a.k_slice_hd = rocke_unified_attn_select_gfx942_flash_k_slice_hd(p);
    return rocke_gfx942_attention_tiled_2d_supports(&a, reason, reason_cap);
}

static bool supports_tiled_2d_general(const rocke_unified_attention_problem_t* p,
                                      const char* arch,
                                      int nqpk,
                                      char* reason,
                                      size_t reason_cap)
{
    bool flash = rocke_unified_attn_enable_gfx942_fp16_flash(p, arch);
    int num_warps = flash ? rocke_unified_attn_select_gfx942_flash_num_warps(p, arch)
                          : rocke_unified_attn_select_2d_num_warps(p, arch);
    int block_m_per_warp = rocke_unified_attn_select_2d_block_m_per_warp(p, arch);
    int tile_size = rocke_unified_attn_select_2d_tile_size(p, arch);
    bool single_k = flash && rocke_unified_attn_gfx942_flash_use_single_buffer(p);
    bool cfvst = flash && rocke_unified_attn_gfx942_flash_use_cfvst(p);
    bool ring = rocke_unified_attn_enable_gfx942_flash_k_sliced_ring(p, arch);

    if(streq(arch, "gfx942"))
    {
        rocke_gfx942_attention_tiled_2d_supports_args_t a
            = rocke_gfx942_attention_tiled_2d_supports_args_default();
        a.head_size = p->head_size;
        a.block_size = p->block_size;
        a.dtype = p->dtype;
        a.num_queries_per_kv = nqpk;
        a.use_alibi = p->use_alibi;
        a.use_qq_bias = p->use_qq_bias;
        a.use_fp8 = p->use_fp8;
        a.q_dtype = p->q_dtype;
        a.num_warps = num_warps;
        a.block_m_per_warp = block_m_per_warp;
        a.kv_storage_dtype = rocke_unified_attn_kv_storage_dtype(p);
        a.has_tile_size = true;
        a.tile_size = tile_size;
        a.arch = arch;
        a.use_mfma_32x32x8 = flash;
        a.use_transposed_qk_32x32 = flash;
        a.use_k_single_buffer = single_k;
        a.use_conflict_free_v_store = cfvst;
        a.use_k_sliced_ring = ring;
        a.ring_depth = rocke_unified_attn_select_gfx942_flash_ring_depth(p, arch);
        a.k_slice_hd = rocke_unified_attn_select_gfx942_flash_k_slice_hd(p);
        a.use_d256_fast = rocke_unified_attn_gfx942_4warp_fast(p, arch);
        return rocke_gfx942_attention_tiled_2d_supports(&a, reason, reason_cap);
    }

    /* _tiled_2d_impl: every arch other than gfx942 / gfx1250 takes the gfx950
     * variant, whose own arch check rejects the unsupported ones. */
    rocke_gfx950_attention_tiled_2d_supports_args_t a
        = rocke_gfx950_attention_tiled_2d_supports_args_default();
    a.head_size = p->head_size;
    a.block_size = p->block_size;
    a.dtype = p->dtype;
    a.num_queries_per_kv = nqpk;
    a.use_alibi = p->use_alibi;
    a.use_qq_bias = p->use_qq_bias;
    a.use_fp8 = p->use_fp8;
    a.q_dtype = p->q_dtype;
    a.num_warps = num_warps;
    a.block_m_per_warp = block_m_per_warp;
    a.kv_storage_dtype = rocke_unified_attn_kv_storage_dtype(p);
    a.has_tile_size = true;
    a.tile_size = tile_size;
    a.arch = arch;
    a.use_mfma_32x32x8 = flash;
    a.use_transposed_qk_32x32 = flash;
    a.use_k_single_buffer = single_k;
    a.use_conflict_free_v_store = cfvst;
    a.use_k_sliced_ring = ring;
    return rocke_gfx950_attention_tiled_2d_supports(&a, reason, reason_cap);
}

bool rocke_unified_attention_supports_tiled_2d(const rocke_unified_attention_problem_t* p,
                                               const char* arch_in,
                                               char* reason,
                                               size_t reason_cap)
{
    if(p == NULL)
    {
        gate_reason(reason, reason_cap, "null problem");
        return false;
    }
    char arch[ROCKE_ATTN_GATE_ARCH_CAP];
    rocke_unified_attention_canonical_arch(arch_in, arch, sizeof(arch));

    if(rocke_attn_unified_reject_mask_semantics(p, reason, reason_cap)
       || reject_mask_on_arch(p, arch, reason, reason_cap))
        return false;
    if(streq(arch, "gfx1250") && p->softcap > 0)
    {
        gate_reason(reason, reason_cap, "gfx1250 tiled 2D does not support softcap yet");
        return false;
    }
    if(p->use_additive_bias && !streq(arch, "gfx950"))
    {
        gate_reason(reason, reason_cap, "additive bias is only implemented on gfx950");
        return false;
    }
    if(streq(arch, "gfx1250"))
    {
        gate_reason(reason, reason_cap, "gfx1250 tiled 2D is not ported to C++");
        return false;
    }
    int nqpk = rocke_unified_attention_problem_num_queries_per_kv(p);
    if(nqpk <= 0)
    {
        gate_reason(reason, reason_cap, "num_query_heads must be divisible by num_kv_heads");
        return false;
    }

    bool ok = rocke_unified_attn_enable_gfx942_bf16_flash(p, arch)
                  ? supports_gfx942_bf16_wide(p, arch, nqpk, reason, reason_cap)
                  : supports_tiled_2d_general(p, arch, nqpk, reason, reason_cap);
    if(!ok)
        return false;
    return !reject_tiled_2d_not_ported(p, arch, reason, reason_cap);
}

bool rocke_unified_attention_supports_tiled_3d(const rocke_unified_attention_problem_t* p,
                                               const char* arch_in,
                                               char* reason,
                                               size_t reason_cap)
{
    if(p == NULL)
    {
        gate_reason(reason, reason_cap, "null problem");
        return false;
    }
    char arch[ROCKE_ATTN_GATE_ARCH_CAP];
    rocke_unified_attention_canonical_arch(arch_in, arch, sizeof(arch));

    if(rocke_attn_unified_reject_mask_semantics(p, reason, reason_cap)
       || reject_mask_on_arch(p, arch, reason, reason_cap)
       || rocke_attn_unified_reject_fp8_format_arch_mismatch(p, arch, reason, reason_cap))
        return false;
    if(p->use_additive_bias && !streq(arch, "gfx950"))
    {
        gate_reason(reason, reason_cap, "additive bias is only implemented on gfx950");
        return false;
    }
    if(streq(arch, "gfx1250"))
    {
        gate_reason(reason, reason_cap, "gfx1250 tiled 3D is not ported to C++");
        return false;
    }
    int nqpk = rocke_unified_attention_problem_num_queries_per_kv(p);
    if(nqpk <= 0)
    {
        gate_reason(reason, reason_cap, "num_query_heads must be divisible by num_kv_heads");
        return false;
    }

    const char* arch_reason = NULL;
    bool ok;
    if(streq(arch, "gfx942"))
        ok = rocke_gfx942_attention_tiled_3d_supports(p->head_size,
                                                      p->block_size,
                                                      p->dtype,
                                                      nqpk,
                                                      p->use_alibi,
                                                      p->use_qq_bias,
                                                      p->use_fp8,
                                                      p->q_dtype,
                                                      rocke_unified_attn_kv_storage_dtype(p),
                                                      arch,
                                                      &arch_reason);
    else
        ok = rocke_gfx950_attention_tiled_3d_supports(p->head_size,
                                                      p->block_size,
                                                      p->dtype,
                                                      nqpk,
                                                      p->use_alibi,
                                                      p->use_qq_bias,
                                                      p->use_fp8,
                                                      p->q_dtype,
                                                      rocke_unified_attn_kv_storage_dtype(p),
                                                      arch,
                                                      &arch_reason);
    if(!ok)
    {
        gate_reason(reason, reason_cap, "%s", arch_reason != NULL ? arch_reason : "");
        return false;
    }

    /* Python admits these on gfx950; the C++ tiled 3D builder does not
     * implement them. */
    if(!rocke_unified_attention_problem_default_mask(p))
    {
        char label[64];
        rocke_attn_unified_mask_label(p, label, sizeof(label));
        gate_reason(reason, reason_cap, "tiled 3D %s masks are not ported to C++ yet", label);
        return false;
    }
    if(p->use_additive_bias)
    {
        gate_reason(reason, reason_cap, "tiled 3D additive bias is not ported to C++ yet");
        return false;
    }
    gate_reason(reason, reason_cap, "%s", arch_reason != NULL ? arch_reason : "");
    return true;
}
