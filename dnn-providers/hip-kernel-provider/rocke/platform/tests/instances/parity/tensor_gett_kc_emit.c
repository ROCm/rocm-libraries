/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/tensor_gett_kc_emit.c -- C-side emitter for the tensor GETT on K-contiguous (RCR)
 * bodies (rocke/instance_tensor_gett.h kc_tile / kc_trait). Builds the rows of
 * tensor_gett_kc_emit.py (same order).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/instance_tensor_gett.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

namespace
{
using namespace rocke::gett;

[[noreturn]] void reject(const std::string& why)
{
    fprintf(stderr, "%s\n", why.c_str());
    exit(1);
}

void kc_body(TensorGettSpec* s, const char* cfg)
{
    std::string why;
    if(!parse_config(cfg, &s->kc_tile, &s->kc_trait, &why))
        reject(why);
    s->has_kc_tile = true;
    s->has_kc_trait = true;
}

void mn_body(TensorGettSpec* s, const char* cfg)
{
    std::string why;
    if(!parse_config(cfg, &s->mn_tile, &s->mn_trait, &why))
        reject(why);
    s->has_mn_tile = true;
    s->has_mn_trait = true;
}

bool make(int idx, TensorGettSpec* s)
{
    switch(idx)
    {
    case 0: /* canon5m_f16_scl */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 1: /* canon5m_bf16_bil_cp8 */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 2: /* canon5m_f16_bil_cp0 */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 3: /* canon5m_f16_bil_cp2 */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 4: /* p2r5_bf16_scl */
    {
        *s = TensorGettSpec::make({"K4", "M4", "K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
                                  {"K4", "N4", "K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 5: /* p2r5_f16_bil_cp8 */
    {
        *s = TensorGettSpec::make({"K4", "M4", "K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
                                  {"K4", "N4", "K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 6: /* p1r2m2_f16_bil */
    {
        *s = TensorGettSpec::make(
            {"K1", "M1", "K0", "M0"},
            {"K1", "N1", "K0", "N0"},
            {"N1", "M1", "N0", "M0"},
            {{"K0", 96}, {"K1", 128}, {"M0", 128}, {"M1", 96}, {"N0", 160}, {"N1", 112}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 7: /* p1r4s_f16_scl_roleA */
    {
        *s = TensorGettSpec::make({"K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
                                  {"K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
                                  {"N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    }
    case 8: /* p1r4s_bf16_bil_roleA */
    {
        *s = TensorGettSpec::make({"K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
                                  {"K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
                                  {"N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"},
                                  {{"K0", 6},
                                   {"K1", 6},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"M0", 12},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        kc_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    }
    case 9: /* canon2s_ragged_roleA */
    {
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(
            s,
            "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b=0,"
            "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,"
            "ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,lds_"
            "swizzle_r=1,mfma_interleave=True,pad_m=True,pad_n=True,dtl_tight_rsrc=True");
        return true;
    }
    case 10: /* canon2s_ragged_roleA_bil */
    {
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(
            s,
            "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b=0,"
            "dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue=default,"
            "ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle=True,lds_"
            "swizzle_r=1,mfma_interleave=True,pad_m=True,pad_n=True,dtl_tight_rsrc=True");
        return true;
    }
    case 11: /* bad_strided_d_rilv_runs */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->has_strides_d = true;
        s->strides_d = ModeMap{{"M0", 1},
                               {"M1", 74},
                               {"M2", 592},
                               {"M3", 4736},
                               {"M4", 28416},
                               {"N0", 113664},
                               {"N1", 1136640},
                               {"N2", 9093120},
                               {"N3", 54558720},
                               {"N4", 436469760}};
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 12: /* canon5m_strided_d_roleA_bil */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->has_strides_d = true;
        s->strides_d = ModeMap{{"M0", 1},
                               {"M1", 74},
                               {"M2", 592},
                               {"M3", 4736},
                               {"M4", 28416},
                               {"N0", 113664},
                               {"N1", 1136640},
                               {"N2", 9093120},
                               {"N3", 54558720},
                               {"N4", 436469760}};
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 2;
        s->tile_k = 32;
        kc_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    }
    case 13: /* wide_d_f16_bil */
    {
        *s = TensorGettSpec::make(
            {"K0", "M0"}, {"K0", "N0"}, {"N0", "M0"}, {{"K0", 256}, {"M0", 40960}, {"N0", 40960}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 14: /* wide_d_bf16_roleA */
    {
        *s = TensorGettSpec::make(
            {"K0", "M0"}, {"K0", "N0"}, {"N0", "M0"}, {{"K0", 256}, {"M0", 40960}, {"N0", 40960}});
        s->dtype = "bf16";
        s->tile_k = 32;
        kc_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    }
    case 15: /* canon6L_wide_d_scl */
    {
        *s = TensorGettSpec::make(
            {"K5", "K4", "K3", "K2", "K1", "K0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {"K5", "K4", "K3", "K2", "K1", "K0", "N5", "N4", "N3", "N2", "N1", "N0"},
            {"N5", "N4", "N3", "N2", "N1", "N0", "M5", "M4", "M3", "M2", "M1", "M0"},
            {{"K0", 6},
             {"K1", 4},
             {"K2", 4},
             {"K3", 4},
             {"K4", 2},
             {"K5", 2},
             {"M0", 12},
             {"M1", 10},
             {"M2", 8},
             {"M3", 4},
             {"M4", 4},
             {"M5", 3},
             {"N0", 12},
             {"N1", 10},
             {"N2", 6},
             {"N3", 4},
             {"N4", 4},
             {"N5", 4}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 16: /* bad_c_elems */
    {
        *s = TensorGettSpec::make(
            {"K0", "M0"}, {"K0", "N0"}, {"N0", "M0"}, {{"K0", 64}, {"M0", 46592}, {"N0", 46592}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 17: /* bad_epi_v16 */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.epi_v16 = true;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 18: /* bad_epi_pair */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.epi_pair = true;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 19: /* bad_store_nt */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.store_nt = true;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 20: /* bad_with_mn */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        mn_body(s,
                "t256x256x32_w2x2x1_a16x16x32|acc_agpr=True,chiplet_swizzle=True,chiplet_wgm=4,"
                "direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_"
                "prefetch=True,dtl_stages=3,epilogue=default,ilv_barrier_after=2,ilv_merge=True,"
                "interwave_setprio=True,lds_swizzle=True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    }
    case 21: /* bad_tile_k_legacy */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->tile_k = 64;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 22: /* bad_split_k */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True,split_k=2");
        return true;
    }
    case 23: /* bad_vec_dc8 */
    {
        *s = TensorGettSpec::make({"K0", "M0"},
                                  {"K0", "N1", "N0"},
                                  {"N1", "M0", "N0"},
                                  {{"K0", 256}, {"M0", 256}, {"N0", 4}, {"N1", 64}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True");
        return true;
    }
    case 24: /* bad_ragged_no_tight */
    {
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t128x128x64_w2x4x1_a16x16x32|chiplet_swizzle=True,direct_to_lds=True,dtl_cache_b="
                "0,dtl_frag_prefetch=True,dtl_hoist=True,dtl_prefetch=True,dtl_stages=3,epilogue="
                "default,ilv_barrier_after=2,ilv_merge=True,interwave_setprio=True,lds_swizzle="
                "True,lds_swizzle_r=1,mfma_interleave=True");
        return true;
    }
    case 25: /* bad_ragged_rilv_pad */
    {
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True,pad_m=True,pad_n=True,dtl_tight_rsrc=True");
        return true;
    }
    case 26: /* canon2s_ragged_rvp2_bil */
    {
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True,pad_m=True,pad_n=True,dtl_tight_rsrc=True,rve_pad=2");
        return true;
    }
    case 27: /* canon2s_ragged_rvp1_bf16 */
    {
        *s = TensorGettSpec::make(
            {"K1", "K0", "M1", "M0"},
            {"K1", "K0", "N1", "N0"},
            {"N1", "N0", "M1", "M0"},
            {{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}});
        s->dtype = "bf16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=8,chiplet_swizzle="
                "True,chiplet_wgm=16,direct_to_lds=True,dtl_cache_b=0,dtl_frag_prefetch=True,dtl_"
                "hoist=True,dtl_prefetch=True,dtl_stages=2,epilogue=default,ilv_barrier_after=2,"
                "ilv_merge=True,lds_row_interleave=True,mfma_interleave=True,rilv_vec_epilogue="
                "True,rve_store_nt=True,pad_m=True,pad_n=True,dtl_tight_rsrc=True,rve_pad=1");
        return true;
    }
    case 28: /* beat_canon5m_f16_bil */
    {
        *s = TensorGettSpec::make({"K4", "K3", "K2", "K1", "K0", "M4", "M3", "M2", "M1", "M0"},
                                  {"K4", "K3", "K2", "K1", "K0", "N4", "N3", "N2", "N1", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(
            s,
            "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
            "das_late=True,direct_to_lds=True,dtl_addr_spread=True,dtl_cache_b=0,dtl_frag_prefetch="
            "True,dtl_hoist=True,dtl_prefetch=True,dtl_soff0=True,dtl_stages=2,epilogue=default,hb_"
            "dtl_delay=1,hb_read_every=2,hb_sched=True,hb_skew=1,hb_tile_lgkm0=1,kloop_unroll=2,"
            "lds_row_interleave=True,mfma_interleave=True,mfma_serpentine=True,persistent_ctas=256,"
            "rilv_pad=32,rilv_vec_epilogue=True,rve_store_nt=True");
        return true;
    }
    case 29: /* beat_p2r5_bf16_scl */
    {
        *s = TensorGettSpec::make({"K4", "M4", "K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
                                  {"K4", "N4", "K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
                                  {"N4", "N3", "N2", "N1", "N0", "M4", "M3", "M2", "M1", "M0"},
                                  {{"K0", 6},
                                   {"K1", 4},
                                   {"K2", 4},
                                   {"K3", 4},
                                   {"K4", 2},
                                   {"M0", 10},
                                   {"M1", 8},
                                   {"M2", 8},
                                   {"M3", 6},
                                   {"M4", 4},
                                   {"N0", 10},
                                   {"N1", 8},
                                   {"N2", 6},
                                   {"N3", 8},
                                   {"N4", 4}});
        s->dtype = "bf16";
        s->tile_k = 32;
        kc_body(
            s,
            "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle=True,"
            "das_late=True,direct_to_lds=True,dtl_addr_spread=True,dtl_cache_b=0,dtl_frag_prefetch="
            "True,dtl_hoist=True,dtl_prefetch=True,dtl_soff0=True,dtl_stages=2,epilogue=default,hb_"
            "dtl_delay=1,hb_read_every=2,hb_sched=True,hb_skew=1,hb_tile_lgkm0=1,kloop_unroll=2,"
            "lds_row_interleave=True,mfma_interleave=True,mfma_serpentine=True,persistent_ctas=256,"
            "rilv_pad=32,rilv_vec_epilogue=True,rve_store_nt=True");
        return true;
    }
    case 30: /* beat_rag_canon4m_f16_bil */
    {
        *s = TensorGettSpec::make({"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
                                  {"K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 8},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"K3", 4},
                                   {"M0", 16},
                                   {"M1", 12},
                                   {"M2", 10},
                                   {"M3", 8},
                                   {"N0", 14},
                                   {"N1", 10},
                                   {"N2", 12},
                                   {"N3", 8}});
        s->dtype = "fp16";
        s->knobs.bilinear = true;
        s->knobs.c_prefetch = 8;
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,direct_to_lds=True,dtl_addr_spread=True,dtl_cache_b=0,dtl_frag_prefetch=True,"
                "dtl_hoist=True,dtl_prefetch=True,dtl_stages=2,dtl_tight_ragged_only=True,dtl_"
                "tight_rsrc=True,epilogue=default,hb_dtl_delay=1,hb_read_every=2,hb_sched=True,hb_"
                "skew=1,kloop_unroll=2,lds_row_interleave=True,mfma_interleave=True,mfma_"
                "serpentine=True,pad_fast_tile=True,pad_m=True,persistent_ctas=256,rilv_pad=32,"
                "rilv_vec_epilogue=True,rve_pad=2,rve_store_nt=True");
        return true;
    }
    case 31: /* beat_rag_p2r4_bf16_scl */
    {
        *s = TensorGettSpec::make({"K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
                                  {"K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
                                  {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
                                  {{"K0", 8},
                                   {"K1", 8},
                                   {"K2", 6},
                                   {"K3", 4},
                                   {"M0", 16},
                                   {"M1", 12},
                                   {"M2", 10},
                                   {"M3", 8},
                                   {"N0", 14},
                                   {"N1", 10},
                                   {"N2", 12},
                                   {"N3", 8}});
        s->dtype = "bf16";
        s->tile_k = 32;
        kc_body(s,
                "t256x256x64_w2x2x1_a16x16x32|acc_agpr=True,chiplet_chunk_size=32,chiplet_swizzle="
                "True,direct_to_lds=True,dtl_addr_spread=True,dtl_cache_b=0,dtl_frag_prefetch=True,"
                "dtl_hoist=True,dtl_prefetch=True,dtl_stages=2,dtl_tight_ragged_only=True,dtl_"
                "tight_rsrc=True,epilogue=default,hb_dtl_delay=1,hb_read_every=2,hb_sched=True,hb_"
                "skew=1,kloop_unroll=2,lds_row_interleave=True,mfma_interleave=True,mfma_"
                "serpentine=True,pad_fast_tile=True,pad_m=True,persistent_ctas=256,rilv_pad=32,"
                "rilv_vec_epilogue=True,rve_pad=2,rve_store_nt=True");
        return true;
    }
    default:
        return false;
    }
}

rocke_kernel_def_t* build(rocke_ir_builder_t* b, int idx, int* known)
{
    TensorGettSpec s;
    if(!make(idx, &s))
        return NULL;
    *known = 1;
    std::string why;
    if(!is_valid_spec(s, "gfx950", &why))
        reject("invalid spec: " + why);
    return build_tensor_gett(b, s, "gfx950");
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: tensor_gett_kc_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    const char* arch = "gfx950";
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    int known = 0;
    rocke_kernel_def_t* kernel = build(&b, idx, &known);
    if(!known)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text, err, sizeof err);
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
