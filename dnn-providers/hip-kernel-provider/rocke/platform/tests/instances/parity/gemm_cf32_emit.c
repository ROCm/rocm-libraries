/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/gemm_cf32_emit.c -- C-side emitter for the named universal GEMM
 * bodies of the emulated-f32 routes with their fused operand prologues. Builds
 * the specs of gemm_cf32_emit.py (same order, same values) through the C API
 * (rocke_gemm_cf32_body, rocke_gemm_cf32_adjust_k, rocke_gemm_prologue_t) so
 * run_diff.py can byte-compare the two engines. The cases of make_spec() are
 * generated: `python gemm_cf32_emit.py --c-cases`.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_cf32.h"
#include "rocke/instance_gemm_prologue.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

#define ARCH "gfx950"

/* One config's base: cf32_body(body), the data types, batched and the
 * 64-bit-address adjustment (_adjust_k at the tile extents). */
static void base(rocke_gemm_universal_spec_t* spec,
                 const char* name,
                 const char* body,
                 const char* dtype,
                 bool batched,
                 bool wide)
{
    *spec = rocke_gemm_universal_spec_default();
    spec->name = name;
    if(!rocke_gemm_cf32_body(body, ARCH, spec))
    {
        fprintf(stderr, "unknown body %s\n", body);
        exit(2);
    }
    spec->data.dtype_a = dtype;
    spec->data.dtype_b = dtype;
    spec->data.dtype_c = dtype;
    spec->batched = batched;
    if(wide)
        rocke_gemm_cf32_adjust_k(spec, spec->tile.tile_m, spec->tile.tile_n, true);
}

/* 1: config idx built into spec (+ prologue); 0: past the end. */
static int
    make_spec(int idx, rocke_gemm_universal_spec_t* spec, rocke_gemm_prologue_t* pro, bool* has_pro)
{
    *pro = rocke_gemm_prologue_default();
    *has_pro = false;
    switch(idx)
    {
    case 0:
        base(spec, "pk_h16_riw", "RIW", "fp16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 1:
        base(spec, "pk_h16_riw", "RIW", "fp16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 2:
        base(spec, "pk_h16_riw_sk", "RIW_SK", "fp16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 3:
        base(spec, "pk_h16_riw_sk", "RIW_SK", "fp16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 4:
        base(spec, "pk_h16_c3", "C3", "fp16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 5:
        base(spec, "pk_h16_c3", "C3", "fp16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 6:
        base(spec, "pk_h16_q3_sk", "Q3_SK", "fp16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 7:
        base(spec, "pk_h16_q3_sk", "Q3_SK", "fp16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 8:
        base(spec, "pk_h16_riw_w64", "RIW", "fp16", false, true);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 9:
        base(spec, "pk_h16_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 10:
        base(spec, "pk_h16_riw", "RIW", "bf16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 11:
        base(spec, "pk_h16_riw_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 12:
        base(spec, "pk_h16_riw_sk", "RIW_SK", "bf16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 13:
        base(spec, "pk_h16_c3", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 14:
        base(spec, "pk_h16_c3", "C3", "bf16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 15:
        base(spec, "pk_h16_q3_sk", "Q3_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 16:
        base(spec, "pk_h16_q3_sk", "Q3_SK", "bf16", true, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 17:
        base(spec, "pk_h16_riw_w64", "RIW", "bf16", false, true);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 18:
        base(spec, "pk_x6_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 19:
        base(spec, "pk_x6_riw", "RIW", "bf16", true, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 20:
        base(spec, "pk_x6_riw_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 21:
        base(spec, "pk_x6_riw_sk", "RIW_SK", "bf16", true, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 22:
        base(spec, "pk_x6_c3", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 23:
        base(spec, "pk_x6_c3", "C3", "bf16", true, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 24:
        base(spec, "pk_x6_riw_w64", "RIW", "bf16", false, true);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 25:
        base(spec, "pk_cf32_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 26:
        base(spec, "pk_cf32_riw", "RIW", "bf16", true, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 27:
        base(spec, "pk_cf32_riw_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 28:
        base(spec, "pk_cf32_riw_sk", "RIW_SK", "bf16", true, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 29:
        base(spec, "pk_cf32_q3_sk", "Q3_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 30:
        base(spec, "pk_cf32_q3_sk", "Q3_SK", "bf16", true, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 31:
        base(spec, "pk_cf32_riw_w64", "RIW", "bf16", false, true);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 32:
        base(spec, "ex_x3_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x3";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 33:
        base(spec, "ex_x4_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x4";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 34:
        base(spec, "ex_x9_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x9";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 35:
        base(spec, "ex_load_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        return 1;
    case 36:
        base(spec, "ex_x3_c3", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x3";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 37:
        base(spec, "ex_x4_c3", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x4";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 38:
        base(spec, "ex_x9_c3", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x9";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 39:
        base(spec, "ex_load_c3", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        return 1;
    case 40:
        base(spec, "ex_bfcap", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->order = "big_first";
        pro->cap = true;
        return 1;
    case 41:
        base(spec, "ex_cvtf32", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    case 42:
        base(spec, "ex_pinv", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->pin_v = true;
        return 1;
    case 43:
        base(spec, "ex_agprout", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->agpr_out = true;
        return 1;
    case 44:
        base(spec, "ex_agprout_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->agpr_out = true;
        return 1;
    case 45:
        base(spec, "ex_noof32", "RIW", "fp16", false, false);
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        return 1;
    case 46:
        base(spec, "ex_x6_nof32", "C3", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = false;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 47:
        base(spec, "ex_cj10_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = true;
        pro->conj_b = false;
        return 1;
    case 48:
        base(spec, "ex_cj01_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = false;
        pro->conj_b = true;
        return 1;
    case 49:
        base(spec, "ex_cj11_riw", "RIW", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = true;
        pro->conj_b = true;
        return 1;
    case 50:
        base(spec, "ex_cj10_q3_sk", "Q3_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = true;
        pro->conj_b = false;
        return 1;
    case 51:
        base(spec, "ex_cj01_q3_sk", "Q3_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = false;
        pro->conj_b = true;
        return 1;
    case 52:
        base(spec, "ex_cj11_q3_sk", "Q3_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = true;
        pro->conj_b = true;
        return 1;
    case 53:
        base(spec, "ex_cj10_riw_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = true;
        pro->conj_b = false;
        return 1;
    case 54:
        base(spec, "ex_cj01_riw_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = false;
        pro->conj_b = true;
        return 1;
    case 55:
        base(spec, "ex_cj11_riw_sk", "RIW_SK", "bf16", false, false);
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        pro->conj_a = true;
        pro->conj_b = true;
        return 1;
    case 56:
        base(spec, "lv_riw", "RIW", "bf16", false, false);
        return 1;
    case 57:
        base(spec, "lv_riw_fp16_bat", "RIW", "fp16", true, false);
        return 1;
    case 58:
        base(spec, "lv_riw_sk", "RIW_SK", "fp16", false, false);
        return 1;
    case 59:
        base(spec, "lv_riw_w64", "RIW", "bf16", false, true);
        return 1;
    case 60:
        base(spec, "lv_riw_nt0", "RIW", "bf16", false, false);
        spec->trait.rve_store_nt = false;
        return 1;
    case 61:
        base(spec, "lv_rilv_scatter", "RIW", "bf16", false, false);
        spec->trait.rilv_vec_epilogue = false;
        spec->trait.rve_store_nt = false;
        return 1;
    case 62:
        base(spec, "lv_rilv_scatter_x6", "RIW", "bf16", false, false);
        spec->trait.rilv_vec_epilogue = false;
        spec->trait.rve_store_nt = false;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 63:
        base(spec, "lv_rilv_pad32", "RIW", "bf16", false, false);
        spec->trait.rilv_pad = 32;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 64:
        base(spec, "lv_rilv_pad64_sk", "RIW_SK", "fp16", false, false);
        spec->trait.rilv_pad = 64;
        return 1;
    case 65:
        base(spec, "lv_soff0_c3", "C3", "bf16", false, false);
        spec->trait.dtl_soff0 = true;
        return 1;
    case 66:
        base(spec, "lv_soff0_c3_x6", "C3", "bf16", false, false);
        spec->trait.dtl_soff0 = true;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 67:
        base(spec, "lv_soff0_riw", "RIW", "bf16", false, false);
        spec->trait.dtl_soff0 = true;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 68:
        base(spec, "lv_w64_tight", "RIW", "bf16", false, true);
        spec->trait.dtl_soff0 = false;
        spec->trait.dtl_tight_rsrc = true;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 69:
        base(spec, "lv_flat_dtl_x6", "C3", "bf16", false, false);
        spec->trait.dtl_stages = 0;
        spec->trait.dtl_frag_prefetch = false;
        spec->trait.mfma_interleave = false;
        spec->trait.ilv_merge = false;
        spec->trait.ilv_barrier_after = 0;
        spec->trait.lds_swizzle_r = -1;
        spec->trait.dtl_hoist = false;
        spec->trait.interwave_setprio = false;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 70:
        base(spec, "lv_flat_dtl_cf", "C3", "bf16", false, false);
        spec->trait.dtl_stages = 0;
        spec->trait.dtl_frag_prefetch = false;
        spec->trait.mfma_interleave = false;
        spec->trait.ilv_merge = false;
        spec->trait.ilv_barrier_after = 0;
        spec->trait.lds_swizzle_r = -1;
        spec->trait.dtl_hoist = false;
        spec->trait.interwave_setprio = false;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        pro->cplx_b = true;
        return 1;
    case 71:
        base(spec, "lv_vgpr_x6", "C3", "bf16", false, false);
        spec->trait.dtl_stages = 0;
        spec->trait.dtl_frag_prefetch = false;
        spec->trait.mfma_interleave = false;
        spec->trait.ilv_merge = false;
        spec->trait.ilv_barrier_after = 0;
        spec->trait.lds_swizzle_r = -1;
        spec->trait.dtl_hoist = false;
        spec->trait.interwave_setprio = false;
        spec->trait.direct_to_lds = false;
        spec->trait.dtl_prefetch = false;
        spec->trait.dtl_cache_b = 2;
        *has_pro = true;
        pro->src = "f32";
        pro->recipe = "x6";
        pro->out_f32 = true;
        pro->split_at = "use";
        pro->psub = true;
        return 1;
    case 72:
        base(spec, "lv_vgpr_h16", "C3", "fp16", false, false);
        spec->trait.dtl_stages = 0;
        spec->trait.dtl_frag_prefetch = false;
        spec->trait.mfma_interleave = false;
        spec->trait.ilv_merge = false;
        spec->trait.ilv_barrier_after = 0;
        spec->trait.lds_swizzle_r = -1;
        spec->trait.dtl_hoist = false;
        spec->trait.interwave_setprio = false;
        spec->trait.direct_to_lds = false;
        spec->trait.dtl_prefetch = false;
        spec->trait.dtl_cache_b = 2;
        *has_pro = true;
        pro->src = "same";
        pro->recipe = "cvt";
        pro->out_f32 = true;
        return 1;
    default:
        return 0;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: gemm_cf32_emit <config_index>\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    rocke_gemm_universal_spec_t spec;
    rocke_gemm_prologue_t pro;
    bool has_pro = false;
    if(!make_spec(idx, &spec, &pro, &has_pro))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    char reason[512];
    if(!rocke_gemm_universal_is_valid_spec(&spec, ARCH, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }
    if(has_pro)
    {
        if(!rocke_gemm_prologue_check(&pro, reason, sizeof reason))
        {
            fprintf(stderr, "prologue: %s\n", reason);
            return 1;
        }
        spec._prologue = &pro;
    }
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &spec, ARCH);
    if(kernel == NULL)
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }
    int rc = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, ARCH, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            rc = 1;
        }
        else
        {
            fputs(llvm_text, stdout);
            free(llvm_text);
        }
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rc = 1;
        }
        else
        {
            fputs(text, stdout);
            free(text);
        }
    }
    else
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    rocke_ir_builder_free(&b);
    return rc;
}
