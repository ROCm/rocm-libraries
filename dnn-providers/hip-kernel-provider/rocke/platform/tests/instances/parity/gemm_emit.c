/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/gemm_emit.c -- C-side emitter for the universal-GEMM parity
 * harness. Selects one of 7 sampled (spec,knobs) configs by argv[1] (the
 * config index 0..6), builds rocke_gemm_universal_spec_t identically to the
 * Python emitter gemm_emit.py, lowers via rocke_gemm_universal_lower_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_gemm_universal_spec_t* spec)
{
    *spec = rocke_gemm_universal_spec_default();

    switch(idx)
    {
    case 0: /* test1 */
        spec->name = "test1";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 1: /* test2 */
        spec->name = "test2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 2: /* test3 */
        spec->name = "test3";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 512;
        spec->batched = false;
        break;
    case 3: /* test4 */
        spec->name = "test4";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 4: /* test5 */
        spec->name = "test5";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 64,
                                              .tile_k = 64,
                                              .warp_m = 1,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 64;
        spec->batched = false;
        break;
    case 5: /* test6 */
        spec->name = "test6";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 128,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = true;
        break;
    case 6: /* test7 */
        spec->name = "test7";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 192,
                                              .tile_n = 192,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.lds_swizzle = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 7: /* test8 -- split-K regression: split_k=1, byte-identical to idx 0 */
        spec->name = "test8";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->trait.split_k = 1;
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 8: /* test9 -- split-K active: split_k=8, decode shape, atomic-add */
        spec->name = "test9";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 16,
                                              .tile_n = 64,
                                              .tile_k = 64,
                                              .warp_m = 1,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.split_k = 8;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 9: /* test1 shape, lowered for gfx942 (CDNA coverage of common GEMM) */
        spec->name = "test1_gfx942";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 10: /* test_noalc: cshuffle with cshuffle_no_alias=True (idx 1 shape) */
        spec->name = "test_noalc";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.cshuffle_no_alias = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 11: /* test_pingpong: the cdna_dtl_pingpong_256x256 dispatch pick */
        spec->name = "test_pingpong";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 12: /* test_pingpong_csh: ping-pong + cshuffle (C aliased onto A/B), bf16 */
        spec->name = "test_pingpong_csh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.epilogue = "cshuffle";
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 13: /* test_abdb_csh: compv4 + cshuffle, explicit AB double buffer */
        spec->name = "test_abdb_csh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.ab_double_buffer_set = true;
        spec->trait.ab_double_buffer = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 14: /* test_absb: compv4 default, explicit single AB buffer */
        spec->name = "test_absb";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.ab_double_buffer_set = true;
        spec->trait.ab_double_buffer = false;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 15: /* test_pingpong_iw: ping-pong + interwave_setprio */
        spec->name = "test_pingpong_iw";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.interwave_setprio = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 16: /* test_pingpong_gfx942: gfx942 ping-pong analogue */
        spec->name = "test_pingpong_gfx942";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 17: /* test_iwp_csh: VGPR compv4 + cshuffle + abdb + swizzle + interwave_setprio */
        spec->name = "test_iwp_csh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.scheduler = "interwave";
        spec->trait.ab_double_buffer_set = true;
        spec->trait.ab_double_buffer = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 18: /* test_ring2_iwp: alias-scoped 2-stage DTL ring + interwave_setprio (the gfx950 dispatch pick) */
        spec->name = "test_ring2_iwp";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 19: /* test_ring2_sab_isp: ring + swapped-accumulator vector epilogue + DTL issue spread */
        spec->name = "test_ring2_sab_isp";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.mfma_swap_ab = true;
        spec->trait.dtl_issue_spread = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 20: /* test_ring3_sb: 3-stage ring (two tiles in flight, partial vmcnt) + in-stream barrier */
        spec->name = "test_ring3_sb";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_stream_barrier = true;
        spec->trait.interwave_setprio = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 21: /* test_ring2_frp_w24: 8-wave 32x32x16 ring + fragment prefetch + issue spread */
        spec->name = "test_ring2_frp_w24";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.dtl_issue_spread = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 512;
        spec->batched = false;
        break;
    case 22: /* test_ring2_pers: ring + persistent overlapped epilogue (256 CTAs) */
        spec->name = "test_ring2_pers";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.persistent_ctas = 256;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 23: /* test_ring2_iglp: ring + iglp_opt(1) */
        spec->name = "test_ring2_iglp";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.iglp_opt_set = true;
        spec->trait.iglp_opt = 1;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 24: /* test_ring2_gfx942: gfx942 ring (256x256x32, 16x16x16, 1-dword DTL; 64 KiB) */
        spec->name = "test_ring2_gfx942";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 25: /* test_ring4_csh_hints: 4-stage ring + cshuffle + sched hints, no chiplet swizzle */
        spec->name = "test_ring4_csh_hints";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = false;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 4;
        spec->trait.emit_sched_hints_set = true;
        spec->trait.emit_sched_hints = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 26: /* test_ring2_sab_padm_pers: ring + swapped epilogue with the pad_m row guard + persistent (128 CTAs) */
        spec->name = "test_ring2_sab_padm_pers";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.mfma_swap_ab = true;
        spec->trait.pad_m = true;
        spec->trait.persistent_ctas = 128;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 27: /* test_ring3_pers_nochip_iglp2: 3-stage ring + persistent without chiplet swizzle (div/mod tile origin, P=100) + iglp_opt(2) */
        spec->name = "test_ring3_pers_nochip_iglp2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = false;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.persistent_ctas = 100;
        spec->trait.iglp_opt_set = true;
        spec->trait.iglp_opt = 2;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 28: /* test_ring4_pers_iglp3: 4-stage ring + persistent (256 CTAs) + interwave_setprio + iglp_opt(3) */
        spec->name = "test_ring4_pers_iglp3";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 4;
        spec->trait.persistent_ctas = 256;
        spec->trait.interwave_setprio = true;
        spec->trait.iglp_opt_set = true;
        spec->trait.iglp_opt = 3;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 29: /* test_ring2_nolanekey_iglp0: 64x64x256 ring: swizzle span 32 > warp tile 16 (non-lane_key rows) + iglp_opt(0) */
        spec->name = "test_ring2_nolanekey_iglp0";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 64,
                                              .tile_k = 256,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.iglp_opt_set = true;
        spec->trait.iglp_opt = 0;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 30: /* test_agpr_w2x2_a32_ring2_iwp: 4-wave 2x2 32x32x16 body, AGPR accumulators, 2-stage ring + iwp */
        spec->name = "test_agpr_w2x2_a32_ring2_iwp";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.acc_agpr = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 31: /* test_agpr_w2x2_a16_pers: bf16 2x2 16x16x32 (128x128 per wave), AGPR accumulators, ring + persistent */
        spec->name = "test_agpr_w2x2_a16_pers";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.persistent_ctas = 256;
        spec->trait.interwave_setprio = true;
        spec->trait.acc_agpr = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 32: /* test_agpr_gfx942_w2x2_a32: gfx942: ping-pong (non-ring) body, 2x2 32x32x8, AGPR accumulators */
        spec->name = "test_agpr_gfx942_w2x2_a32";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.acc_agpr = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 33: /* test_agpr_a32_frp_ilv_iwp: 2x2 32x32x16 AGPR body, frag prefetch + MFMA interleave + iwp (barrier after 2 MFMAs) */
        spec->name = "test_agpr_a32_frp_ilv_iwp";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 34: /* test_agpr_a16_s3_frp_ilv: bf16 2x2 16x16x32 AGPR body, 3-stage ring at tile_k 32, frag prefetch + MFMA interleave (DTL every 2 MFMAs) */
        spec->name = "test_agpr_a16_s3_frp_ilv";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_dtl_every = 2;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 35: /* test_agpr_a16_s4_ilv_mrg_ba2: bf16 2x2 16x16x32 AGPR body, 4-stage ring at tile_k 32, frag prefetch + merged MFMA interleave + barrier after 2 MFMAs + iwp */
        spec->name = "test_agpr_a16_s4_ilv_mrg_ba2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 4;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 36: /* test_agpr_ilv: agpr_ilv exactly (S3 tk32 frp ilv mrg ba2 swr1 dh agpr iwp) */
        spec->name = "test_agpr_ilv";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 37: /* test_agpr_ilv_apin_swr0: agpr_ilv + acc_agpr_pin, lds_swizzle_r 0, bf16 */
        spec->name = "test_agpr_ilv_apin_swr0";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 0;
        spec->trait.dtl_hoist = true;
        spec->trait.acc_agpr_pin = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 38: /* test_agpr_ilv_csh_swr2: agpr_ilv with the cshuffle epilogue (AGPR hand-out), lds_swizzle_r 2 */
        spec->name = "test_agpr_ilv_csh_swr2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "cshuffle";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 2;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 39: /* test_agpr_a32_ring2_spk2: 2x2 32x32x16 AGPR ring body + split-K 2 (AGPR hand-out in the atomic epilogue) */
        spec->name = "test_agpr_a32_ring2_spk2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.acc_agpr = true;
        spec->trait.split_k = 2;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 40: /* test_s2_iwp_swr1_dh: 16-wave ring body (no AGPR) + lds_swizzle_r 1 + dtl_hoist */
        spec->name = "test_s2_iwp_swr1_dh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 41: /* test_agpr_s4_de2_apin_swr3_dh: bf16 S4 tk32 ilv de2 + apin + lds_swizzle_r 3 + dtl_hoist */
        spec->name = "test_agpr_s4_de2_apin_swr3_dh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 4;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.acc_agpr_pin = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_dtl_every = 2;
        spec->trait.lds_swizzle_r = 3;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 42: /* test_agpr_a32_ilv_ba1_swr4_dh: 2x2 32x32x16 AGPR S2 frp ilv ba1 + lds_swizzle_r 4 + dtl_hoist */
        spec->name = "test_agpr_a32_ilv_ba1_swr4_dh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 1;
        spec->trait.lds_swizzle_r = 4;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 43: /* test_gfx942_ring2_dh: gfx942: S2 ring + dtl_hoist (admitted on gfx942: GPU numerics) */
        spec->name = "test_gfx942_ring2_dh";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 44: /* test_gfx942_swr1: gfx942: lds_swizzle_r 1 (admitted on gfx942: GPU numerics) */
        spec->name = "test_gfx942_swr1";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.lds_swizzle_r = 1;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 45: /* test_gfx942_frp_ilv: gfx942: S2 frp + mfma_interleave (admitted on gfx942: GPU numerics) */
        spec->name = "test_gfx942_frp_ilv";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 46: /* test_gfx90a_agpr: gfx90a: acc_agpr (rejected: no gfx90a GPU numerics) */
        spec->name = "test_gfx90a_agpr";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 8};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.acc_agpr = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 47: /* test_gfx942_ga16_frp_ilv_mrg: gfx942: the GPU-checked GA16 body (bf16 2x2 16x16x16, S2 ring, frp + ilv + mrg + iwp + acc_agpr) */
        spec->name = "test_gfx942_ga16_frp_ilv_mrg";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 48: /* test_agpr_ilv_pc256: agpr_ilv + persistent_ctas 256 (persistent loop on the in-stream interleave body) */
        spec->name = "test_agpr_ilv_pc256";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->trait.persistent_ctas = 256;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 49: /* test_agpr_ilv_pc256_sk: agpr_ilv + persistent + stream-K (DP + 2-tile SK, linear fixup, relaxed flag spin) */
        spec->name = "test_agpr_ilv_pc256_sk";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.streamk = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 50: /* test_s2_iwp_pc256_sk: 16-wave ring (no AGPR) + persistent + stream-K, bf16 */
        spec->name = "test_s2_iwp_pc256_sk";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.streamk = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 51: /* test_agpr_ilv_pc256_sk_pad_trs_pft: agpr_ilv + persistent + stream-K + pad_m / pad_n + tight DTL resources + pad_fast_tile */
        spec->name = "test_agpr_ilv_pc256_sk_pad_trs_pft";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.streamk = true;
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 52: /* test_s2_iwp_pad_trs_pft: 16-wave ring + pad + tight resources per DTL issue (no dtl_hoist) + pad_fast_tile, bf16 */
        spec->name = "test_s2_iwp_pad_trs_pft";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 53: /* test_a32_ilv_pc256_sk: 2x2 32x32x16 AGPR ring + frp ilv mrg + persistent + stream-K (16-float partial quads) */
        spec->name = "test_a32_ilv_pc256_sk";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.streamk = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 54: /* test_agpr_ilv_padn_trs_pft: agpr_ilv + pad_n only + tight resources + pad_fast_tile (non-persistent) */
        spec->name = "test_agpr_ilv_padn_trs_pft";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->trait.pad_n = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 55: /* test_gfx942_s2_pc256_sk_pad: gfx942: 16-wave 16x16x16 ring + persistent + stream-K + pad + tight + pad_fast_tile, bf16 */
        spec->name = "test_gfx942_s2_pc256_sk_pad";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.interwave_setprio = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.streamk = true;
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 56: /* test_sk_no_pc: stream-K without persistent_ctas (both engines reject) */
        spec->name = "test_sk_no_pc";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 3;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->trait.streamk = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 57: /* test_disp_ring_small_s2_padk64: dispatch: small ring 128x128x64 2x4 waves, 2-stage, pad + trs + pft + dh (8192x8200x64 f16) */
        spec->name = "test_disp_ring_small_s2_padk64";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 2;
        spec->trait.dtl_hoist = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 512;
        spec->batched = false;
        break;
    case 58: /* test_disp_ring_small_s3ilv: dispatch: small ring 128x128x64, 3-stage frag-prefetch interleave (one wave; 2048^3 bf16) */
        spec->name = "test_disp_ring_small_s3ilv";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 512;
        spec->batched = false;
        break;
    case 59: /* test_disp_ring_small_k32: dispatch: 128x128x32 2x2-wave interleave ring, K % 64 == 32 (2048^2 x 96 f16) */
        spec->name = "test_disp_ring_small_k32";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 60: /* test_disp_ring_small_k32_agpr: dispatch: 256x128x32 AGPR interleave ring, K % 64 == 32 (4608^2 x 160 bf16) */
        spec->name = "test_disp_ring_small_k32_agpr";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 61: /* test_disp_ring_pad: dispatch: 16-wave persistent pad ring, chunk 32 (7680x6720x1536 f16) */
        spec->name = "test_disp_ring_pad";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 2;
        spec->trait.persistent_ctas = 256;
        spec->trait.dtl_hoist = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 62: /* test_disp_agpr_ilv_big_grid_pad: dispatch: AGPR interleave big-grid pad body, wgm 4 (30720x20160x7680 bf16) */
        spec->name = "test_disp_agpr_ilv_big_grid_pad";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.pad_m = true;
        spec->trait.pad_n = true;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 4;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->trait.dtl_tight_rsrc = true;
        spec->trait.pad_fast_tile = true;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 63: /* test_disp_agpr_ilv_big_grid: dispatch: AGPR interleave big-grid aligned body, wgm 4 (46080^2 x 2304 f16) */
        spec->name = "test_disp_agpr_ilv_big_grid";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 4;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 3;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.persistent_ctas = 256;
        spec->trait.acc_agpr = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_merge = true;
        spec->trait.lds_swizzle_r = 1;
        spec->trait.dtl_hoist = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 64: /* test_disp_ring_w4_iglp1: dispatch: 16-wave ring, wgm 4 + iglp_opt(1) (30720^2 x 2048 bf16) */
        spec->name = "test_disp_ring_w4_iglp1";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 4;
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_cache_b = 0;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.interwave_setprio = true;
        spec->trait.dtl_stages = 2;
        spec->trait.iglp_opt_set = true;
        spec->trait.iglp_opt = 1;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = false;
        break;
    case 65: /* test_er_a16s2_er3: early release: A16S2 er3 rl32 aq40 (tk64 S2 2x2 16x16x32 agpr frp ilv mrg dh) */
        spec->name = "test_er_a16s2_er3";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_early_release = 3;
        spec->trait.ilv_release_at = 32;
        spec->trait.ilv_acquire_at = 40;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 66: /* test_er_a16s2_er3_auto_pc: early release: er3 auto positions + c32, bf16 (persistent + frag prefetch is a Python-only extension; the C++ engine still rejects it) */
        spec->name = "test_er_a16s2_er3_auto_pc";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 32;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_early_release = 3;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 67: /* test_er_a16s2_er2_iwp: early release: er2 rl8 aq6 + interwave_setprio */
        spec->name = "test_er_a16s2_er2_iwp";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_early_release = 2;
        spec->trait.ilv_release_at = 8;
        spec->trait.ilv_acquire_at = 6;
        spec->trait.interwave_setprio = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 68: /* test_er_a16s2_er1_ba2: early release: er1 ba2 aq8 */
        spec->name = "test_er_a16s2_er1_ba2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_early_release = 1;
        spec->trait.ilv_barrier_after = 2;
        spec->trait.ilv_acquire_at = 8;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 69: /* test_er_a16s2_rs_ds_aov_sab: early release: read span 50 + dtl span 100 + acc_out_vgpr + mfma_swap_ab (agpr hand-out) */
        spec->name = "test_er_a16s2_rs_ds_aov_sab";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_read_span = 50;
        spec->trait.ilv_dtl_span = 100;
        spec->trait.acc_out_vgpr = true;
        spec->trait.mfma_swap_ab = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 70: /* test_er_a32s2_er3: early release: 32x32x16 tk64 (4 K atoms) er3 rl8 aq2 swr1, bf16 */
        spec->name = "test_er_a32s2_er3";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = true;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_early_release = 3;
        spec->trait.ilv_release_at = 8;
        spec->trait.ilv_acquire_at = 2;
        spec->trait.lds_swizzle_r = 1;
        spec->data.dtype_a = "bf16";
        spec->data.dtype_b = "bf16";
        spec->data.dtype_c = "bf16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 71: /* test_er_a16s2_er3_de2: early release: er3 rl24 aq32 without merge, ilv_dtl_every 2 */
        spec->name = "test_er_a16s2_er3_de2";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->trait.dtl_prefetch = true;
        spec->trait.lds_swizzle = true;
        spec->trait.dtl_cache_a = 0;
        spec->trait.dtl_cache_b = 0;
        spec->trait.chiplet_swizzle = true;
        spec->trait.chiplet_wgm = 8;
        spec->trait.chiplet_chunk_size = 64;
        spec->trait.dtl_stages = 2;
        spec->trait.acc_agpr = true;
        spec->trait.dtl_frag_prefetch = true;
        spec->trait.mfma_interleave = true;
        spec->trait.ilv_merge = false;
        spec->trait.dtl_hoist = true;
        spec->trait.ilv_early_release = 3;
        spec->trait.ilv_release_at = 24;
        spec->trait.ilv_acquire_at = 32;
        spec->trait.ilv_dtl_every = 2;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    default:
        return -1;
    }
    rocke_gemm_universal_spec_finalize(spec);
    return 0;
}

/* Configs 9, 16, 24, 32, 43..45, 47 and 55 exercise gfx942, config 46 gfx90a (rejected by both
 * engines); all others use the gfx950 baseline. */
static const char* arch_for(int idx)
{
    if(idx == 46)
        return "gfx90a";
    return (idx == 9 || idx == 16 || idx == 24 || idx == 32 || (idx >= 43 && idx <= 45) || idx == 47
            || idx == 55)
               ? "gfx942"
               : "gfx950";
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index 0..71>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    const char* arch = arch_for(idx);

    rocke_gemm_universal_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_gemm_universal_lower_to_llvm(
            &spec, arch, ROCKE_LLVM_FLAVOR_AUTO, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
        return 0;
    }

    /* ir / verify modes need the kernel object */
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &spec, arch);
    if(!kernel || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "build failed: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            ret = 1;
        }
        else
        {
            fputs(t, stdout);
            free(t);
        }
    }
    else if(strcmp(mode, "verify") == 0)
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
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        ret = 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
