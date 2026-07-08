// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_helper_rocke.instances.common.attention_unified_selectors.c --
 *   C99 port of selected SELECTOR + descriptor + emit symbols from
 *   rocke/instances/common/attention_unified.py.
 *
 * The host-side selectors reproduce the Python branch structure exactly
 * (same comparisons, same order, same gate predicates). The IR-emitting
 * helpers reproduce the Python rocke_b_* builder-call sequence byte-faithfully
 * (same ops, same order, same operands).
 *
 * Lifetime: every emitted node is arena-owned (rocke_ir_builder_t.arena). Nothing
 * is freed individually; the arena bulk-frees the whole graph.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_helper_rocke.instances.common.attention_unified_selectors.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir.h"

/* ------------------------------------------------------- arch resolution */

/* Python: _resolve_attention_arch() -- query the running device arch, memoized
 * process-wide, falling back to "gfx950" when the device arch is unavailable
 * (CPU-only / cross-compile harnesses, which is exactly this static-port
 * context).
 *
 * TODO(port): wire a real runtime.hip_module.get_device_arch() query when the
 * C runtime surface exposes one. Until then this faithfully reproduces the
 * documented fallback (the only arch the tiled MFMA path supports by default)
 * and lets a host override the resolution via the setter below. */
static const char* g_resolved_attention_arch = NULL;

void rocke_unified_attn_set_resolved_arch(const char* arch)
{
    g_resolved_attention_arch = arch;
}

static const char* rocke_unified_attn_resolve_arch(void)
{
    return (g_resolved_attention_arch != NULL) ? g_resolved_attention_arch : "gfx950";
}

static bool arch_is(const char* want)
{
    return strcmp(rocke_unified_attn_resolve_arch(), want) == 0;
}

/* ----------------------------------------------------- num_queries_per_kv */

int rocke_unified_attn_num_queries_per_kv(rocke_ir_builder_t* b,
                                          const rocke_unified_attn_problem_t* p)
{
    if(p->num_kv_heads == 0 || (p->num_query_heads % p->num_kv_heads) != 0)
    {
        if(b != NULL && b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_VALUE;
            (void)snprintf(b->err,
                           (size_t)ROCKE_ERR_MSG_CAP,
                           "num_query_heads must be divisible by num_kv_heads");
        }
        return 0;
    }
    return p->num_query_heads / p->num_kv_heads;
}

/* Internal convenience: same value without a builder (the selectors call the
 * property freely; the divisibility precondition is guaranteed for any problem
 * that reached the spec stage). */
static int nqpk(const rocke_unified_attn_problem_t* p)
{
    if(p->num_kv_heads == 0)
    {
        return 0;
    }
    return p->num_query_heads / p->num_kv_heads;
}

/* ----------------------------------------------------- gate predicates */
/* These mirror the Python private predicates (_enable_*). They are static so
 * the selectors stay byte-faithful; not part of the public ABI. */

static int select_2d_tile_size(const rocke_unified_attn_problem_t* p); /* fwd */

/* Python: _enable_combo_2d(problem). */
static bool enable_combo_2d(const rocke_unified_attn_problem_t* p)
{
    if(!arch_is("gfx950"))
    {
        return false;
    }
    if(strcmp(p->dtype, "bf16") != 0)
    {
        return false;
    }
    if(p->use_alibi || p->use_qq_bias || p->softcap > 0)
    {
        return false;
    }
    if(p->head_size != 64 || p->block_size != 32)
    {
        return false;
    }
    if(nqpk(p) != 8)
    {
        return false;
    }
    if(p->max_seqlen_q <= 256)
    {
        return false;
    }
    return true;
}

/* Python: _enable_single_batch_combo(problem). Single-batch (num_seqs == 1)
 * d128/d64 prefill -> full 32x32 combo. The combinatorial autotuner proved
 * the full combo is 1.5-2.7x faster than the legacy 16x16x32 path for this cohort
 * and correctness-equal; the old gate's num_seqs>=2 restriction was stale (it was
 * measured on the bare transposed path, not the full combo). */
static bool enable_single_batch_combo(const rocke_unified_attn_problem_t* p)
{
    if(!arch_is("gfx950"))
    {
        return false;
    }
    if(p->num_seqs != 1)
    {
        return false;
    }
    if(strcmp(p->dtype, "bf16") != 0 && strcmp(p->dtype, "fp16") != 0)
    {
        return false;
    }
    if(p->use_fp8)
    {
        return false;
    }
    if(p->use_alibi || p->use_qq_bias)
    {
        return false;
    }
    if(p->softcap > 0 || p->use_sinks)
    {
        return false;
    }
    if(p->sliding_window > 0)
    {
        return false;
    }
    if(p->head_size != 64 && p->head_size != 128)
    {
        return false;
    }
    if(p->max_seqlen_q <= 256)
    {
        return false;
    }
    return true;
}

/* Python: _enable_d128_small_tile(problem). d128 occupancy lever: select
 * T = block_size (small tile) + nw=2 for the single-batch d128 combo so the
 * kernel drops from 1 -> 2 WG/CU (the d128 combo is purely LDS-bound; halving
 * the tile takes LDS 48->24 KB and crosses flash on S=2048/S=4096).
 *
 * DEFAULT-ON: enabled by default for the gfx950 single-batch d128 no-FP8
 * combo. This is a production ROUTING change (per-spec golden emit unchanged --
 * T=block_size is an existing tile_size value). ESCAPE HATCH:
 * HIPDNN_GFX950_D128_SMALL_TILE=0 (or off/no/false) force-DISABLES it. Mirrors
 * the Python helper's env gate byte-faithfully (case-folded, off-keywords only). */
static bool enable_d128_small_tile(const rocke_unified_attn_problem_t* p)
{
    const char* env = getenv("HIPDNN_GFX950_D128_SMALL_TILE");
    if(env != NULL)
    {
        /* case-fold then compare to the OFF keywords (matches Python
           .strip().lower() in ("0","false","no","off")). */
        char buf[16];
        size_t i = 0;
        for(; env[i] != '\0' && i + 1 < sizeof(buf); ++i)
        {
            char c = env[i];
            buf[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
        }
        buf[i] = '\0';
        if(strcmp(buf, "0") == 0 || strcmp(buf, "false") == 0 || strcmp(buf, "no") == 0
           || strcmp(buf, "off") == 0)
        {
            return false;
        }
    }
    return p->head_size == 128 && !p->use_fp8 && enable_single_batch_combo(p);
}

/* Python: _enable_transposed_qk_32x32(problem). */
static bool enable_transposed_qk_32x32(const rocke_unified_attn_problem_t* p)
{
    if(!arch_is("gfx950"))
    {
        return false;
    }
    if(enable_combo_2d(p))
    {
        return true;
    }
    /* Single-batch d128/d64 prefill full-combo cohort: the routing fix.
       num_seqs == 1 now gets the 32x32 transposed combo. */
    if(enable_single_batch_combo(p))
    {
        return true;
    }
    /* fp16 and bf16 both ride the validated transposed-32x32 path. Gating this
       to bf16 only left fp16 d128/d64 prefill on a legacy path that is
       numerically wrong at d128 with KV >= 1024 (mirrors the Python selector
       _enable_transposed_qk_32x32). */
    if(strcmp(p->dtype, "bf16") != 0 && strcmp(p->dtype, "fp16") != 0)
    {
        return false;
    }
    if(p->use_fp8)
    {
        return false;
    }
    if(p->use_alibi || p->use_qq_bias)
    {
        return false;
    }
    if(p->softcap > 0 || p->use_sinks)
    {
        return false;
    }
    if(p->head_size != 64 && p->head_size != 128)
    {
        return false;
    }
    bool multi_batch = (p->max_seqlen_q > 256) && (p->num_seqs >= 2);
    bool single_seq_hd64 = (p->head_size == 64) && (p->block_size == 16) && (p->num_seqs <= 1)
                           && (nqpk(p) >= 4) && (p->max_seqlen_q > 768);
    if(!(multi_batch || single_seq_hd64))
    {
        return false;
    }
    if(p->sliding_window > 0 && !p->use_fp8)
    {
        return false;
    }
    return true;
}

/* Python: _enable_early_v_schedule(problem). Long d64 single-batch combo prefill
 * (head_size==64, max_seqlen_q >= 2048) issues the V load early; bit-identical,
 * supersedes the V double-buffer. d128 long gets no V schedule flag (early-V
 * regresses d128 -- see the Python docstring's ablation). */
static bool enable_early_v_schedule(const rocke_unified_attn_problem_t* p)
{
    if(!enable_single_batch_combo(p))
    {
        return false;
    }
    return p->head_size == 64 && p->max_seqlen_q >= 2048;
}

/* Python: _enable_v_double_buffer(problem). Short single-batch combo prefill
 * (max_seqlen_q <= 1024) on d64 stacks the V[i+1] double-buffer prefetch;
 * bit-identical, mutually exclusive with early-V. d128: vdbuf is a NET DRAG
 * (joint num_warps x schedule sweep: the win path is nw=2 with NO prefetch =
 * 1.30x flash) -> OFF for d128, which also auto-disables sched_barrier (it
 * gates on this predicate). */
static bool enable_v_double_buffer(const rocke_unified_attn_problem_t* p)
{
    if(!enable_single_batch_combo(p))
    {
        return false;
    }
    if(enable_early_v_schedule(p))
    {
        return false;
    }
    if(p->head_size == 128)
    {
        return false;
    }
    return p->max_seqlen_q <= 1024;
}

/* Python: _enable_transposed_subflags(problem). The no-SW transposed-softmax
 * VALU sub-flags fire for the whole no-SW transposed-32x32 cohort (combo,
 * single-batch, AND the multi-batch transposed path the prod selector previously
 * left them off -- the autotuner's ~1.19x multi-batch miss). */
static bool enable_transposed_subflags(const rocke_unified_attn_problem_t* p)
{
    if(p->sliding_window > 0)
    {
        return false;
    }
    return enable_transposed_qk_32x32(p);
}

/* Python: _enable_gfx942_small_q_narrow(problem). */
static bool enable_gfx942_small_q_narrow(const rocke_unified_attn_problem_t* p)
{
    return arch_is("gfx942") && (strcmp(p->dtype, "fp16") == 0 || strcmp(p->dtype, "bf16") == 0)
           && !p->use_fp8 && (p->head_size == 64 || p->head_size == 128)
           && (p->max_seqlen_q > 1 && p->max_seqlen_q <= 768) && p->sliding_window == 0
           && !p->use_sinks && p->softcap == 0 && !p->use_alibi && !p->use_qq_bias;
}

/* Python: _enable_gfx942_fp16_flash(problem). */
static bool enable_gfx942_fp16_flash(const rocke_unified_attn_problem_t* p)
{
    return arch_is("gfx942") && (p->head_size == 64 || p->head_size == 128)
           && strcmp(p->dtype, "fp16") == 0 && !p->use_fp8 && p->sliding_window == 0
           && !p->use_sinks && p->softcap == 0 && !p->use_alibi && !p->use_qq_bias
           && !enable_gfx942_small_q_narrow(p);
}

/* Python: _enable_gfx942_d128_fp16_flash(problem). */
static bool enable_gfx942_d128_fp16_flash(const rocke_unified_attn_problem_t* p)
{
    return enable_gfx942_fp16_flash(p) && p->head_size == 128;
}

/* Python: _enable_gfx942_l4(problem) -- alias for the D128 fp16 flash family. */
static bool enable_gfx942_l4(const rocke_unified_attn_problem_t* p)
{
    return enable_gfx942_d128_fp16_flash(p);
}

/* Python: _gfx942_flash_wide_setting(). The HIPDNN_GFX942_FLASH_WIDE env knob
 * defaults to 4 (off/2/4 overrides). The static port honours the default; env
 * override is a host-runtime concern.
 *
 * TODO(port): consult getenv("HIPDNN_GFX942_FLASH_WIDE") for off/2/4 once the
 * port wires environment knobs. Until then returns the documented default. */
static int gfx942_flash_wide_setting(void)
{
    return 4;
}

/* Python: _select_gfx942_flash_num_warps(problem). */
static int select_gfx942_flash_num_warps(const rocke_unified_attn_problem_t* p)
{
    (void)p;
    int wide = gfx942_flash_wide_setting();
    return (wide == 2 || wide == 4) ? wide : 1;
}

/* ----------------------------------------------------- select_2d_tile_size */

/* Python: _enable_gfx942_fp16_flash gate inside _select_2d_tile_size for the
 * D64 force-T=64 branch. */
static int select_2d_tile_size(const rocke_unified_attn_problem_t* p)
{
    /* Sliding-window long-prefill FP8 exception. */
    if(p->use_fp8 && p->sliding_window > 0 && p->max_seqlen_q > 256)
    {
        return p->block_size;
    }
    /* gfx942 D64. */
    if(arch_is("gfx942") && p->head_size == 64)
    {
        if(enable_gfx942_fp16_flash(p))
        {
            return 64;
        }
        return p->block_size;
    }
    /* gfx942 D128 (ALL dtypes): T=64. */
    if(arch_is("gfx942") && p->head_size == 128)
    {
        return 64;
    }
    /* bf16 transposed-combo sliding-window. */
    if(enable_combo_2d(p) && p->sliding_window > 0)
    {
        return p->block_size;
    }
    /* Single-batch d128/d64 prefill full-combo cohort: d128 -> 2*BS,
       d64 -> 128 (paired with num_warps=4). */
    if(enable_single_batch_combo(p))
    {
        if(p->head_size == 64)
        {
            return 128;
        }
        /* d128 occupancy lever: halve the tile to T=block_size (one paged
           block per iter) -> LDS 48->24 KB -> 2 WG/CU. DEFAULT-ON;
           HIPDNN_GFX950_D128_SMALL_TILE=0 force-disables. */
        if(enable_d128_small_tile(p))
        {
            return p->block_size;
        }
        return 2 * p->block_size;
    }
    /* Qwen3-30B-A3B prefill specialization. */
    if(p->head_size == 64 && p->block_size == 16 && p->num_seqs <= 1 && !p->use_fp8
       && strcmp(p->dtype, "bf16") == 0 && nqpk(p) >= 4)
    {
        if(p->max_seqlen_q >= 512 && p->max_seqlen_q <= 768)
        {
            return 128;
        }
        if(p->max_seqlen_q > 64)
        {
            return 64;
        }
    }
    return 2 * p->block_size;
}

int rocke_unified_attn_select_2d_tile_size(const rocke_unified_attn_problem_t* p)
{
    return select_2d_tile_size(p);
}

/* ----------------------------------------------------- select_2d_num_warps */

int rocke_unified_attn_select_2d_num_warps(const rocke_unified_attn_problem_t* p)
{
    int target;

    /* Small/medium gfx942 prefill light narrow path. */
    if(enable_gfx942_small_q_narrow(p))
    {
        return (nqpk(p) == 1) ? 1 : 2;
    }
    /* gfx942 D128 fp16 flash/L4. */
    if(enable_gfx942_l4(p))
    {
        return select_gfx942_flash_num_warps(p);
    }
    /* gfx942 D64 oracle. */
    if(arch_is("gfx942") && p->head_size == 64)
    {
        return 4;
    }
    if(enable_combo_2d(p))
    {
        int t2 = (p->sliding_window > 0 && !p->use_fp8) ? 2 : 4;
        int HD = p->head_size;
        int BS = p->block_size;
        int T = select_2d_tile_size(p);
        while(t2 > 1)
        {
            if((T * HD) < 64 * t2 * 8)
            {
                t2 /= 2;
                continue;
            }
            if((64 * 8) / HD > BS)
            {
                t2 /= 2;
                continue;
            }
            break;
        }
        return (t2 > 1) ? t2 : 1;
    }
    /* Single-batch d128/d64 prefill full-combo cohort (re-tuned by the
       joint num_warps x schedule sweep): d64 -> nw=4 (T=128); d128 -> nw=2
       (S<=1024, nw=2 + V-double-buffer OFF = 1.30x flash) / nw=4 (S>=2048,
       nw=4 wpe=2 lifts S=4096 to 0.885x). Subject to the same step-down. */
    if(enable_single_batch_combo(p))
    {
        int t2;
        int HD = p->head_size;
        int BS = p->block_size;
        int T = select_2d_tile_size(p);
        if(p->head_size == 64)
        {
            t2 = 4;
        }
        /* RECONCILIATION: the d128 small-tile occupancy win (T=block_size
           -> 2 WG/CU) REQUIRES num_warps=2 for ALL seqlens; nw=4 + T=32 is
           occupancy-WORSE (56 KB LDS -> 1 WG/CU). Override the S>=2048 -> nw=4
           rule when the small-tile cohort is active. */
        else if(enable_d128_small_tile(p))
        {
            t2 = 2;
        }
        else if(p->max_seqlen_q <= 1024)
        {
            t2 = 2;
        }
        else
        {
            t2 = 4;
        }
        while(t2 > 1)
        {
            if((T * HD) < 64 * t2 * 8)
            {
                t2 /= 2;
                continue;
            }
            if((64 * 8) / HD > BS)
            {
                t2 /= 2;
                continue;
            }
            break;
        }
        return (t2 > 1) ? t2 : 1;
    }
    /* Qwen3-30B-A3B prefill specialization. */
    if(p->head_size == 64 && p->block_size == 16 && p->num_seqs <= 1 && !p->use_fp8
       && strcmp(p->dtype, "bf16") == 0 && nqpk(p) >= 4)
    {
        if(p->max_seqlen_q <= 128)
        {
            target = 1;
        }
        else if(p->max_seqlen_q <= 768)
        {
            target = 2;
        }
        else
        {
            target = 4;
        }
    }
    else if(p->max_seqlen_q <= 64)
    {
        target = 1;
    }
    else if(p->max_seqlen_q <= 128)
    {
        target = 2;
    }
    else if(p->max_seqlen_q <= 256)
    {
        target = 4;
    }
    else if(p->num_seqs <= 1)
    {
        target = 2;
    }
    else if(nqpk(p) == 1 && p->head_size == 64 && !enable_combo_2d(p))
    {
        target = (p->max_seqlen_q <= 512 || p->max_seqlen_q >= 1536) ? 2 : 4;
    }
    else
    {
        target = 4;
    }

    {
        int HD = p->head_size;
        int BS = p->block_size;
        int T = select_2d_tile_size(p);
        const int WORK_BYTES = 2;
        /* Step down until all constraints are satisfied. */
        while(target > 1)
        {
            int THREADS = 64 * target;
            int BLOCK_M = 16 * target;
            int per_wave_tokens;
            int lds_bytes;
            if((T * HD) < THREADS * 8)
            {
                target /= 2;
                continue;
            }
            per_wave_tokens = (64 * 8) / HD;
            if(per_wave_tokens > BS)
            {
                target /= 2;
                continue;
            }
            lds_bytes = BLOCK_M * HD * WORK_BYTES + 2 * T * HD * WORK_BYTES
                        + 2 * T * HD * WORK_BYTES + BLOCK_M * T * WORK_BYTES + BLOCK_M * HD * 4;
            if(lds_bytes <= 96 * 1024)
            {
                break;
            }
            target /= 2;
        }
    }
    return (target > 1) ? target : 1;
}

/* ------------------------------------------------ select_2d_block_m_per_warp */

int rocke_unified_attn_select_2d_block_m_per_warp(const rocke_unified_attn_problem_t* p)
{
    if(enable_gfx942_small_q_narrow(p))
    {
        return 16;
    }
    if(arch_is("gfx942") && p->head_size == 64)
    {
        return 32;
    }
    if(enable_gfx942_l4(p))
    {
        return 32;
    }
    if(enable_transposed_qk_32x32(p)) /* includes _enable_combo_2d */
    {
        return 32;
    }
    if(p->use_fp8 && p->max_seqlen_q > 256 && p->num_seqs >= 2)
    {
        return 32;
    }
    /* Qwen3-30B-A3B prefill specialization. */
    if(p->head_size == 64 && p->block_size == 16 && p->num_seqs <= 1 && !p->use_fp8
       && strcmp(p->dtype, "bf16") == 0 && nqpk(p) >= 4 && p->max_seqlen_q > 768
       && p->sliding_window == 0 && p->softcap == 0 && !p->use_sinks && !p->use_alibi
       && !p->use_qq_bias)
    {
        return 32;
    }
    return 16;
}

/* ----------------------------------------------------- kv_storage_dtype */

const char* rocke_unified_attn_kv_storage_dtype(const rocke_unified_attn_problem_t* p)
{
    return p->use_fp8 ? "fp8e4m3" : NULL;
}

/* ------------------------------------------------- select_2d_waves_per_eu */

/* Python: _select_2d_waves_per_eu(problem). The gfx1250 and the
 * problem.waves_per_eu host-pin branches are not reachable here (no gfx1250 in
 * this build path; the problem struct carries no waves_per_eu field), so this
 * mirrors the remaining branches: combo -> 4, fp8 long multi-seq prefill -> 3,
 * otherwise -> 2. Always returns a concrete int (Python only returns None on the
 * gfx1250 / host-pin branches this port elides), so out_wpe is always written. */
bool rocke_unified_attn_select_2d_waves_per_eu(const rocke_unified_attn_problem_t* p, int* out_wpe)
{
    int wpe = 2;
    if(enable_single_batch_combo(p))
    {
        /* Single-batch combo (re-tuned by the joint num_warps x schedule
           sweep): wpe=2 across all seqlens. The prior wpe=3-at-S>=4096 rule
           regressed the cohort (d128 S4096 nw=4: wpe=2 0.881 > wpe=3 0.856). */
        wpe = 2;
    }
    else if(enable_combo_2d(p))
    {
        wpe = 4;
    }
    else if(p->use_fp8 && p->max_seqlen_q > 256 && p->num_seqs >= 2)
    {
        wpe = 3;
    }
    if(out_wpe != NULL)
    {
        *out_wpe = wpe;
    }
    return true;
}

/* ----------------------------------------------- 2D feature-gate predicates */

/* Python: _enable_register_pv(problem). */
static bool enable_register_pv(const rocke_unified_attn_problem_t* p)
{
    if(strcmp(p->dtype, "bf16") != 0)
    {
        return false;
    }
    if(p->use_sinks)
    {
        return false;
    }
    if(p->sliding_window > 0)
    {
        return false;
    }
    if(p->softcap > 0)
    {
        return false;
    }
    if(p->use_alibi)
    {
        return false;
    }
    if(p->use_qq_bias)
    {
        return false;
    }
    if(rocke_unified_attn_kv_storage_dtype(p) != NULL)
    {
        return false;
    }
    /* use_register_pv requires the 16x16x32 path; conflicts with mfma_32x32. */
    if(enable_transposed_qk_32x32(p)) /* == _enable_mfma_32x32 */
    {
        return false;
    }
    return true;
}

bool rocke_unified_attn_enable_combo_2d(const rocke_unified_attn_problem_t* p)
{
    return enable_combo_2d(p);
}

bool rocke_unified_attn_enable_transposed_qk_32x32(const rocke_unified_attn_problem_t* p)
{
    return enable_transposed_qk_32x32(p);
}

bool rocke_unified_attn_enable_mfma_32x32(const rocke_unified_attn_problem_t* p)
{
    /* Python: _enable_mfma_32x32(problem) == _enable_transposed_qk_32x32. */
    return enable_transposed_qk_32x32(p);
}

bool rocke_unified_attn_enable_transposed_half_local_pv(const rocke_unified_attn_problem_t* p)
{
    /* Python: _enable_transposed_half_local_pv == _enable_transposed_qk_32x32. */
    return enable_transposed_qk_32x32(p);
}

bool rocke_unified_attn_enable_register_pv(const rocke_unified_attn_problem_t* p)
{
    return enable_register_pv(p);
}

bool rocke_unified_attn_enable_single_batch_combo(const rocke_unified_attn_problem_t* p)
{
    return enable_single_batch_combo(p);
}

bool rocke_unified_attn_enable_transposed_subflags(const rocke_unified_attn_problem_t* p)
{
    return enable_transposed_subflags(p);
}

bool rocke_unified_attn_enable_d128_small_tile(const rocke_unified_attn_problem_t* p)
{
    return enable_d128_small_tile(p);
}

bool rocke_unified_attn_enable_v_double_buffer(const rocke_unified_attn_problem_t* p)
{
    return enable_v_double_buffer(p);
}

bool rocke_unified_attn_enable_early_v_schedule(const rocke_unified_attn_problem_t* p)
{
    return enable_early_v_schedule(p);
}

/* ----------------------------------------------------------- magic div */

rocke_value_t*
    rocke_unified_attn_magic_div(rocke_ir_builder_t* b, rocke_value_t* dividend, int divisor)
{
    uint64_t mult = 0;
    int shift = 0;
    if(!rocke_calculate_magic_numbers(b, divisor, &mult, &shift))
    {
        return NULL;
    }
    return rocke_do_magic_division(b, dividend, mult, shift);
}

bool rocke_unified_attn_magic_div_mod(rocke_ir_builder_t* b,
                                      rocke_value_t* dividend,
                                      int divisor,
                                      rocke_value_t** out_quotient,
                                      rocke_value_t** out_remainder)
{
    rocke_value_t* quotient = rocke_unified_attn_magic_div(b, dividend, divisor);
    rocke_value_t* remainder
        = rocke_b_sub(b, dividend, rocke_b_mul(b, quotient, rocke_b_const_i32(b, divisor)));
    if(out_quotient != NULL)
    {
        *out_quotient = quotient;
    }
    if(out_remainder != NULL)
    {
        *out_remainder = remainder;
    }
    return b != NULL && b->status == ROCKE_OK;
}

/* --------------------------------------------------------- descriptors */

rocke_tensor_descriptor_t* rocke_unified_attn_q_descriptor(rocke_ir_builder_t* b,
                                                           const rocke_unified_attn_problem_t* p)
{
    int lengths[3];
    static const char* const coord_names[3] = {"token", "head", "dim"};
    lengths[0] = p->max_seqlen_q + 1;
    lengths[1] = p->num_query_heads;
    lengths[2] = p->head_size;
    return rocke_tensor_descriptor_naive(b, "Q", lengths, 3, NULL, coord_names, 3);
}

rocke_unified_attn_paged_kv_descriptor_t
    rocke_unified_attn_paged_kv_descriptor(const rocke_unified_attn_problem_t* p)
{
    rocke_unified_attn_paged_kv_descriptor_t d;
    d.block_size = p->block_size;
    d.stride_0 = p->block_size * p->num_kv_heads * p->head_size;
    d.stride_1 = p->num_kv_heads * p->head_size;
    d.stride_2 = p->head_size;
    d.stride_3 = 1;
    return d;
}

rocke_value_t* rocke_unified_attn_paged_kv_offset(rocke_ir_builder_t* b,
                                                  const rocke_unified_attn_paged_kv_descriptor_t* d,
                                                  rocke_value_t* physical_block,
                                                  rocke_value_t* token_in_block,
                                                  rocke_value_t* kv_head,
                                                  rocke_value_t* dim)
{
    rocke_value_t* off = rocke_b_mul(b, physical_block, rocke_b_const_i32(b, d->stride_0));
    off = rocke_b_add(b, off, rocke_b_mul(b, token_in_block, rocke_b_const_i32(b, d->stride_1)));
    off = rocke_b_add(b, off, rocke_b_mul(b, kv_head, rocke_b_const_i32(b, d->stride_2)));
    off = rocke_b_add(b, off, rocke_b_mul(b, dim, rocke_b_const_i32(b, d->stride_3)));
    return off;
}

bool rocke_unified_attn_segm_descriptors(rocke_ir_builder_t* b,
                                         const rocke_unified_attn_problem_t* p,
                                         int num_segments,
                                         rocke_tensor_descriptor_t** out_ml,
                                         rocke_tensor_descriptor_t** out_output)
{
    int ml_lengths[3];
    int out_lengths[4];
    static const char* const ml_coords[3] = {"token", "head", "seg"};
    static const char* const out_coords[4] = {"token", "head", "seg", "dim"};
    rocke_tensor_descriptor_t* ml;
    rocke_tensor_descriptor_t* out;

    ml_lengths[0] = p->max_seqlen_q + 1;
    ml_lengths[1] = p->num_query_heads;
    ml_lengths[2] = num_segments;
    ml = rocke_tensor_descriptor_naive(b, "segm_ml", ml_lengths, 3, NULL, ml_coords, 3);

    out_lengths[0] = p->max_seqlen_q + 1;
    out_lengths[1] = p->num_query_heads;
    out_lengths[2] = num_segments;
    out_lengths[3] = p->head_size;
    out = rocke_tensor_descriptor_naive(b, "segm_output", out_lengths, 4, NULL, out_coords, 4);

    if(out_ml != NULL)
    {
        *out_ml = ml;
    }
    if(out_output != NULL)
    {
        *out_output = out;
    }
    return ml != NULL && out != NULL;
}

/* ------------------------------------------------------- IR emit helpers */

bool rocke_unified_attn_physical_block_and_token(rocke_ir_builder_t* b,
                                                 const rocke_unified_attn_problem_t* p,
                                                 rocke_value_t* block_tables,
                                                 rocke_value_t* seq_idx,
                                                 rocke_value_t* kpos,
                                                 rocke_value_t** out_physical,
                                                 rocke_value_t** out_token_in_block)
{
    rocke_value_t* block_idx = NULL;
    rocke_value_t* token_in_block = NULL;
    int max_blocks;
    rocke_value_t* physical;

    if(!rocke_unified_attn_magic_div_mod(b, kpos, p->block_size, &block_idx, &token_in_block))
    {
        return false;
    }
    max_blocks = (p->max_seqlen_k + p->block_size - 1) / p->block_size;
    physical = rocke_b_global_load_i32(
        b,
        block_tables,
        rocke_b_add(b, rocke_b_mul(b, seq_idx, rocke_b_const_i32(b, max_blocks)), block_idx),
        0 /* align default */);

    if(out_physical != NULL)
    {
        *out_physical = physical;
    }
    if(out_token_in_block != NULL)
    {
        *out_token_in_block = token_in_block;
    }
    return b != NULL && b->status == ROCKE_OK;
}

rocke_value_t* rocke_unified_attn_emit_qk_score(rocke_ir_builder_t* b,
                                                const rocke_unified_attn_problem_t* p,
                                                const rocke_type_t* dtype,
                                                rocke_value_t* query,
                                                rocke_value_t* key,
                                                rocke_value_t* block_tables,
                                                rocke_value_t* seq_idx,
                                                rocke_value_t* q_tok,
                                                rocke_value_t* q_head,
                                                rocke_value_t* kv_head,
                                                rocke_value_t* kpos,
                                                rocke_value_t* scale,
                                                rocke_value_t* rcp_ln2)
{
    const int VEC = 8;
    rocke_value_t* score = rocke_b_const_f32(b, 0.0);
    rocke_value_t* physical = NULL;
    rocke_value_t* token_in_block = NULL;
    rocke_tensor_descriptor_t* q_desc;
    rocke_unified_attn_paged_kv_descriptor_t kv_desc;
    rocke_value_t* q_off_base = NULL;
    rocke_value_t* k_off_base;
    int n_vec;
    int d8;
    int d;

    if(!rocke_unified_attn_physical_block_and_token(
           b, p, block_tables, seq_idx, kpos, &physical, &token_in_block))
    {
        return NULL;
    }
    q_desc = rocke_unified_attn_q_descriptor(b, p);
    kv_desc = rocke_unified_attn_paged_kv_descriptor(p);

    /* q_off_base, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=const_i32(0)) */
    {
        const char* in_names[3] = {"token", "head", "dim"};
        rocke_value_t* in_values[3];
        rocke_value_t* valid = NULL;
        in_values[0] = q_tok;
        in_values[1] = q_head;
        in_values[2] = rocke_b_const_i32(b, 0);
        if(!rocke_transforms_descriptor_offset(
               b, q_desc, in_names, in_values, 3, &q_off_base, &valid))
        {
            return NULL;
        }
    }
    k_off_base = rocke_unified_attn_paged_kv_offset(
        b, &kv_desc, physical, token_in_block, kv_head, rocke_b_const_i32(b, 0));

    n_vec = p->head_size / VEC;
    for(d8 = 0; d8 < n_vec; ++d8)
    {
        rocke_value_t* d_base = rocke_b_const_i32(b, d8 * VEC);
        rocke_value_t* qv
            = rocke_b_global_load_vN(b, query, rocke_b_add(b, q_off_base, d_base), dtype, VEC, 16);
        rocke_value_t* kv
            = rocke_b_global_load_vN(b, key, rocke_b_add(b, k_off_base, d_base), dtype, VEC, 16);
        int i;
        for(i = 0; i < VEC; ++i)
        {
            score
                = rocke_b_fadd(b,
                               score,
                               rocke_b_fmul(b,
                                            rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, qv, i)),
                                            rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, kv, i))));
        }
    }
    /* Tail scalar fold for head_size not a multiple of VEC (empty for the
     * supported {64,128,256} head sizes). */
    for(d = n_vec * VEC; d < p->head_size; ++d)
    {
        rocke_value_t* d_v = rocke_b_const_i32(b, d);
        rocke_value_t* q_off = NULL;
        rocke_value_t* k_off;
        rocke_value_t* qv_s;
        rocke_value_t* kv_s;
        {
            const char* in_names[3] = {"token", "head", "dim"};
            rocke_value_t* in_values[3];
            rocke_value_t* valid = NULL;
            in_values[0] = q_tok;
            in_values[1] = q_head;
            in_values[2] = d_v;
            if(!rocke_transforms_descriptor_offset(
                   b, q_desc, in_names, in_values, 3, &q_off, &valid))
            {
                return NULL;
            }
        }
        k_off = rocke_unified_attn_paged_kv_offset(
            b, &kv_desc, physical, token_in_block, kv_head, d_v);
        qv_s = rocke_b_cast_to_f32(b, rocke_b_global_load(b, query, q_off, dtype, 2));
        kv_s = rocke_b_cast_to_f32(b, rocke_b_global_load(b, key, k_off, dtype, 2));
        score = rocke_b_fadd(b, score, rocke_b_fmul(b, qv_s, kv_s));
    }
    return rocke_b_fmul(b, rocke_b_fmul(b, score, scale), rcp_ln2);
}

rocke_value_t* rocke_unified_attn_emit_v_load(rocke_ir_builder_t* b,
                                              const rocke_unified_attn_problem_t* p,
                                              const rocke_type_t* dtype,
                                              rocke_value_t* value,
                                              rocke_value_t* block_tables,
                                              rocke_value_t* seq_idx,
                                              rocke_value_t* kv_head,
                                              rocke_value_t* kpos,
                                              rocke_value_t* dim)
{
    rocke_value_t* physical = NULL;
    rocke_value_t* token_in_block = NULL;
    rocke_unified_attn_paged_kv_descriptor_t kv_desc;
    rocke_value_t* v_off;

    if(!rocke_unified_attn_physical_block_and_token(
           b, p, block_tables, seq_idx, kpos, &physical, &token_in_block))
    {
        return NULL;
    }
    kv_desc = rocke_unified_attn_paged_kv_descriptor(p);
    v_off = rocke_unified_attn_paged_kv_offset(b, &kv_desc, physical, token_in_block, kv_head, dim);
    return rocke_b_cast_to_f32(b, rocke_b_global_load(b, value, v_off, dtype, 2));
}
