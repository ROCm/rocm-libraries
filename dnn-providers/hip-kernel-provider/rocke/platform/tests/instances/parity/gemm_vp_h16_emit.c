/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/gemm_vp_h16_emit.c -- C-side emitter of the gemm_vp_h16 parity family
 * (the register-staged global prefetch body of the universal GEMM; see
 * _gemm_vp_common.py). Builds the specs of gemm_vp_h16_emit.py (same order, same
 * values) through the C API so run_diff.py can byte-compare the two engines.
 * The case rows are generated: `python gemm_vp_h16_emit.py --c-cases`.
 *
 * the same picks as the hipTensor h16 route builds them: tile-multiple operands (no pads, no split-K) with the out_f32 operand prologue attached.
 */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_prologue.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct vp_case
{
    const char* name;
    const char* arch;
    const char* dtype;
    int tile[9]; /* tile_m, tile_n, tile_k, warp_m, warp_n, warp_k, warp_tile_m/n/k */
    int block_size;
    const char* trait; /* "field=value,..." (non-default TraitSpec fields) */
    const char* pro; /* "field=value,..." of the Prologue, or NULL (none) */
    int batched;
} vp_case_t;

static const vp_case_t k_cases[] = {
    {"hh0",
     "gfx942",
     "fp16",
     {32, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh1",
     "gfx942",
     "fp16",
     {16, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh2",
     "gfx942",
     "fp16",
     {16, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh3",
     "gfx942",
     "fp16",
     {64, 32, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh4",
     "gfx942",
     "fp16",
     {32, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh5",
     "gfx942",
     "fp16",
     {64, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh6",
     "gfx942",
     "fp16",
     {64, 64, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh7",
     "gfx942",
     "fp16",
     {16, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh8",
     "gfx942",
     "fp16",
     {64, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh9",
     "gfx942",
     "fp16",
     {64, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh10",
     "gfx942",
     "fp16",
     {64, 128, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh11",
     "gfx942",
     "fp16",
     {32, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh12",
     "gfx942",
     "fp16",
     {64, 192, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh13",
     "gfx942",
     "fp16",
     {64, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh14",
     "gfx942",
     "fp16",
     {32, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh15",
     "gfx942",
     "fp16",
     {64, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh16",
     "gfx942",
     "fp16",
     {64, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh17",
     "gfx942",
     "fp16",
     {64, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh18",
     "gfx942",
     "fp16",
     {128, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh19",
     "gfx942",
     "fp16",
     {128, 96, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh20",
     "gfx942",
     "fp16",
     {128, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh21",
     "gfx942",
     "fp16",
     {128, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh22",
     "gfx942",
     "fp16",
     {128, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh23",
     "gfx942",
     "fp16",
     {128, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh24",
     "gfx942",
     "fp16",
     {128, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh25",
     "gfx942",
     "fp16",
     {128, 64, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh26",
     "gfx942",
     "fp16",
     {256, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh27",
     "gfx942",
     "fp16",
     {256, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh28",
     "gfx942",
     "fp16",
     {256, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh29",
     "gfx942",
     "fp16",
     {96, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh30",
     "gfx942",
     "fp16",
     {32, 32, 128, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh31",
     "gfx942",
     "fp16",
     {192, 64, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh32",
     "gfx942",
     "fp16",
     {192, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh33",
     "gfx942",
     "fp16",
     {96, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh34",
     "gfx942",
     "fp16",
     {48, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh35",
     "gfx942",
     "fp16",
     {192, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh36",
     "gfx942",
     "fp16",
     {192, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh37",
     "gfx942",
     "fp16",
     {192, 256, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh38",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh39",
     "gfx942",
     "fp16",
     {192, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh40",
     "gfx942",
     "fp16",
     {192, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh41",
     "gfx942",
     "fp16",
     {192, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh42",
     "gfx942",
     "fp16",
     {128, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh43",
     "gfx942",
     "fp16",
     {256, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh44",
     "gfx942",
     "fp16",
     {160, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh45",
     "gfx942",
     "fp16",
     {224, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh46",
     "gfx942",
     "fp16",
     {160, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh47",
     "gfx942",
     "fp16",
     {192, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh48",
     "gfx942",
     "fp16",
     {160, 384, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh49",
     "gfx942",
     "fp16",
     {48, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh50",
     "gfx942",
     "fp16",
     {256, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh51",
     "gfx942",
     "fp16",
     {256, 256, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,lds_swizzle_r=1,"
     "vgpr_prefetch=3,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh52",
     "gfx942",
     "fp16",
     {224, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh53",
     "gfx942",
     "fp16",
     {224, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh54",
     "gfx942",
     "fp16",
     {224, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh55",
     "gfx942",
     "fp16",
     {288, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh56",
     "gfx942",
     "fp16",
     {48, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh57",
     "gfx942",
     "fp16",
     {192, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh58",
     "gfx942",
     "fp16",
     {192, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh59",
     "gfx942",
     "fp16",
     {224, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh60",
     "gfx942",
     "fp16",
     {128, 384, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh61",
     "gfx942",
     "fp16",
     {128, 320, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh62",
     "gfx942",
     "fp16",
     {32, 64, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh63",
     "gfx942",
     "fp16",
     {64, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh64",
     "gfx942",
     "fp16",
     {96, 128, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh65",
     "gfx942",
     "fp16",
     {128, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh66",
     "gfx942",
     "fp16",
     {32, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh67",
     "gfx942",
     "fp16",
     {96, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh68",
     "gfx942",
     "fp16",
     {288, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh69",
     "gfx942",
     "fp16",
     {64, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh70",
     "gfx942",
     "fp16",
     {64, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh71",
     "gfx942",
     "fp16",
     {64, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh72",
     "gfx942",
     "fp16",
     {64, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh73",
     "gfx942",
     "fp16",
     {128, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh74",
     "gfx942",
     "fp16",
     {256, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh75",
     "gfx942",
     "fp16",
     {192, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh76",
     "gfx942",
     "fp16",
     {288, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh77",
     "gfx942",
     "fp16",
     {192, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh78",
     "gfx942",
     "fp16",
     {224, 320, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh79",
     "gfx942",
     "fp16",
     {288, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh80",
     "gfx942",
     "fp16",
     {256, 288, 32, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh81",
     "gfx942",
     "bf16",
     {32, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh82",
     "gfx942",
     "bf16",
     {16, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh83",
     "gfx942",
     "bf16",
     {16, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh84",
     "gfx942",
     "bf16",
     {64, 32, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh85",
     "gfx942",
     "bf16",
     {32, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh86",
     "gfx942",
     "bf16",
     {64, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh87",
     "gfx942",
     "bf16",
     {64, 64, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh88",
     "gfx942",
     "bf16",
     {16, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh89",
     "gfx942",
     "bf16",
     {64, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh90",
     "gfx942",
     "bf16",
     {64, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh91",
     "gfx942",
     "bf16",
     {64, 128, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh92",
     "gfx942",
     "bf16",
     {32, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh93",
     "gfx942",
     "bf16",
     {64, 192, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh94",
     "gfx942",
     "bf16",
     {64, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh95",
     "gfx942",
     "bf16",
     {32, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh96",
     "gfx942",
     "bf16",
     {64, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh97",
     "gfx942",
     "bf16",
     {64, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh98",
     "gfx942",
     "bf16",
     {64, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh99",
     "gfx942",
     "bf16",
     {128, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh100",
     "gfx942",
     "bf16",
     {128, 96, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh101",
     "gfx942",
     "bf16",
     {128, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh102",
     "gfx942",
     "bf16",
     {128, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh103",
     "gfx942",
     "bf16",
     {128, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh104",
     "gfx942",
     "bf16",
     {128, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh105",
     "gfx942",
     "bf16",
     {128, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh106",
     "gfx942",
     "bf16",
     {128, 64, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh107",
     "gfx942",
     "bf16",
     {256, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh108",
     "gfx942",
     "bf16",
     {256, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh109",
     "gfx942",
     "bf16",
     {256, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh110",
     "gfx942",
     "bf16",
     {32, 32, 128, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh111",
     "gfx942",
     "bf16",
     {192, 64, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh112",
     "gfx942",
     "bf16",
     {192, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh113",
     "gfx942",
     "bf16",
     {96, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh114",
     "gfx942",
     "bf16",
     {48, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh115",
     "gfx942",
     "bf16",
     {96, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh116",
     "gfx942",
     "bf16",
     {192, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh117",
     "gfx942",
     "bf16",
     {192, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh118",
     "gfx942",
     "bf16",
     {192, 256, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh119",
     "gfx942",
     "bf16",
     {192, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh120",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh121",
     "gfx942",
     "bf16",
     {288, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh122",
     "gfx942",
     "bf16",
     {192, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh123",
     "gfx942",
     "bf16",
     {128, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh124",
     "gfx942",
     "bf16",
     {256, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh125",
     "gfx942",
     "bf16",
     {160, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh126",
     "gfx942",
     "bf16",
     {224, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh127",
     "gfx942",
     "bf16",
     {160, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh128",
     "gfx942",
     "bf16",
     {192, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh129",
     "gfx942",
     "bf16",
     {160, 384, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh130",
     "gfx942",
     "bf16",
     {256, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh131",
     "gfx942",
     "bf16",
     {256, 256, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,lds_swizzle_r=1,"
     "vgpr_prefetch=3,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh132",
     "gfx942",
     "bf16",
     {192, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh133",
     "gfx942",
     "bf16",
     {224, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh134",
     "gfx942",
     "bf16",
     {288, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh135",
     "gfx942",
     "bf16",
     {224, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh136",
     "gfx942",
     "bf16",
     {128, 384, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh137",
     "gfx942",
     "bf16",
     {48, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh138",
     "gfx942",
     "bf16",
     {192, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh139",
     "gfx942",
     "bf16",
     {192, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh140",
     "gfx942",
     "bf16",
     {128, 320, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh141",
     "gfx942",
     "bf16",
     {32, 64, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh142",
     "gfx942",
     "bf16",
     {64, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh143",
     "gfx942",
     "bf16",
     {96, 128, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh144",
     "gfx942",
     "bf16",
     {64, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh145",
     "gfx942",
     "bf16",
     {128, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh146",
     "gfx942",
     "bf16",
     {32, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh147",
     "gfx942",
     "bf16",
     {96, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh148",
     "gfx942",
     "bf16",
     {224, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh149",
     "gfx942",
     "bf16",
     {224, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh150",
     "gfx942",
     "bf16",
     {288, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh151",
     "gfx942",
     "bf16",
     {64, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh152",
     "gfx942",
     "bf16",
     {64, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh153",
     "gfx942",
     "bf16",
     {64, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh154",
     "gfx942",
     "bf16",
     {64, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     "out_f32=true",
     0},
    {"hh155",
     "gfx942",
     "bf16",
     {128, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh156",
     "gfx942",
     "bf16",
     {256, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh157",
     "gfx942",
     "bf16",
     {192, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh158",
     "gfx942",
     "bf16",
     {288, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     "out_f32=true",
     0},
    {"hh159",
     "gfx942",
     "bf16",
     {192, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh160",
     "gfx942",
     "bf16",
     {224, 320, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh161",
     "gfx942",
     "bf16",
     {224, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     "out_f32=true",
     0},
    {"hh162",
     "gfx942",
     "bf16",
     {256, 288, 32, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     "out_f32=true",
     0},
};

static const char* keep_str(const char* s, size_t n)
{
    static const char* const known[]
        = {"compv4", "compv3", "mem",  "intrawave", "interwave", "default", "cshuffle", "same",
           "cvt",    "f32",    "x3",   "x4",        "x6",        "x9",      "use",      "load",
           "rne",    "trunc",  "rnef", "abs",       "neg",       "exp",     "relu"};
    for(size_t i = 0; i < sizeof(known) / sizeof(known[0]); ++i)
        if(strlen(known[i]) == n && strncmp(known[i], s, n) == 0)
            return known[i];
    return NULL;
}

static bool parse_bool(const char* v, size_t n, bool* out)
{
    if(n == 4 && strncmp(v, "true", 4) == 0)
        *out = true;
    else if(n == 5 && strncmp(v, "false", 5) == 0)
        *out = false;
    else
        return false;
    return true;
}

static bool parse_int(const char* v, size_t n, int* out)
{
    char buf[32];
    if(n == 0 || n >= sizeof buf)
        return false;
    memcpy(buf, v, n);
    buf[n] = 0;
    char* end = NULL;
    long x = strtol(buf, &end, 10);
    if(*end != 0)
        return false;
    *out = (int)x;
    return true;
}

#define KEY(s) (kn == sizeof(s) - 1 && strncmp(k, s, kn) == 0)

/* one TraitSpec field (the fields the case rows use) */
static bool
    set_trait(rocke_gemm_trait_spec_t* t, const char* k, size_t kn, const char* v, size_t vn)
{
#define B(f)    \
    if(KEY(#f)) \
        return parse_bool(v, vn, &t->f);
#define I(f)    \
    if(KEY(#f)) \
        return parse_int(v, vn, &t->f);
#define S(f)                    \
    if(KEY(#f))                 \
    {                           \
        t->f = keep_str(v, vn); \
        return t->f != NULL;    \
    }
#define OB(f)                                     \
    if(KEY(#f))                                   \
    {                                             \
        if(vn == 4 && strncmp(v, "none", 4) == 0) \
        {                                         \
            t->f##_set = false;                   \
            return true;                          \
        }                                         \
        t->f##_set = true;                        \
        return parse_bool(v, vn, &t->f);          \
    }
#define OI(f)                                     \
    if(KEY(#f))                                   \
    {                                             \
        if(vn == 4 && strncmp(v, "none", 4) == 0) \
        {                                         \
            t->f##_set = false;                   \
            return true;                          \
        }                                         \
        t->f##_set = true;                        \
        return parse_int(v, vn, &t->f);           \
    }
    S(pipeline)
    S(scheduler)
    S(epilogue)
    B(pad_m)
    B(pad_n)
    B(pad_k)
    B(chiplet_swizzle)
    I(chiplet_wgm)
    I(chiplet_num_xcds)
    I(chiplet_chunk_size)
    OI(waves_per_eu)
    B(lds_swizzle)
    I(lds_swizzle_r)
    I(split_k)
    OB(emit_sched_hints)
    OB(ab_double_buffer)
    B(interwave_setprio)
    B(mfma_swap_ab)
    B(acc_agpr)
    B(pad_fast_tile)
    B(direct_to_lds)
    B(dtl_prefetch)
    I(dtl_stages)
    B(dtl_frag_prefetch)
    B(mfma_interleave)
    B(dtl_hoist)
    B(dtl_tight_rsrc)
    I(vgpr_prefetch)
    I(vgpr_prefetch_depth)
    I(vp_unroll)
    I(vp_late_reads)
    I(vp_release_atoms)
    I(vp_tail_pack)
    I(vp_pair_reads)
    I(vp_mfma_first)
    B(split_k_aligned)
    I(split_k_serial)
    I(lds_swizzle_l)
    B(wide_addr)
    B(pad_fast_tile)
#undef B
#undef I
#undef S
#undef OB
#undef OI
    return false;
}

/* one Prologue field */
static bool set_pro(rocke_gemm_prologue_t* p, const char* k, size_t kn, const char* v, size_t vn)
{
#define PB(f)   \
    if(KEY(#f)) \
        return parse_bool(v, vn, &p->f);
#define PS(f)   \
    if(KEY(#f)) \
        return (p->f = keep_str(v, vn)) != NULL;
    PB(out_f32)
    PB(cap)
    PB(psub)
    PB(cplx_b)
    PB(conj_a)
    PB(conj_b)
    PB(pin_v)
    PB(fast_exp)
    PS(src)
    PS(recipe)
    PS(split_at)
    PS(split_mode)
    PS(unary_a)
    PS(unary_b)
#undef PB
#undef PS
    return false;
}

#undef KEY

/* "k=v,k=v" -> setter; false on an unknown field / value */
static bool apply(const char* s, void* obj, bool is_pro)
{
    while(s != NULL && *s)
    {
        const char* eq = strchr(s, '=');
        if(eq == NULL)
            return false;
        const char* end = strchr(eq + 1, ',');
        size_t vn = end ? (size_t)(end - eq - 1) : strlen(eq + 1);
        bool ok = is_pro
                      ? set_pro((rocke_gemm_prologue_t*)obj, s, (size_t)(eq - s), eq + 1, vn)
                      : set_trait((rocke_gemm_trait_spec_t*)obj, s, (size_t)(eq - s), eq + 1, vn);
        if(!ok)
        {
            fprintf(stderr, "bad case field %.*s\n", (int)(eq - s), s);
            return false;
        }
        s = end ? end + 1 : NULL;
    }
    return true;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: gemm_vp_h16_emit <config_index> [ll|ir|verify]\n");
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";
    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || (size_t)idx >= sizeof(k_cases) / sizeof(k_cases[0]))
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }
    const vp_case_t* c = &k_cases[idx];
    rocke_gemm_universal_spec_t spec = rocke_gemm_universal_spec_default();
    spec.name = c->name;
    spec.tile.tile_m = c->tile[0];
    spec.tile.tile_n = c->tile[1];
    spec.tile.tile_k = c->tile[2];
    spec.tile.warp_m = c->tile[3];
    spec.tile.warp_n = c->tile[4];
    spec.tile.warp_k = c->tile[5];
    spec.tile.warp_tile_m = c->tile[6];
    spec.tile.warp_tile_n = c->tile[7];
    spec.tile.warp_tile_k = c->tile[8];
    spec.block_size = c->block_size;
    spec.batched = c->batched != 0;
    spec.data.dtype_a = c->dtype;
    spec.data.dtype_b = c->dtype;
    spec.data.dtype_c = c->dtype;
    if(!apply(c->trait, &spec.trait, false))
        return 2;
    rocke_gemm_prologue_t pro = rocke_gemm_prologue_default();
    char reason[512];
    if(c->pro != NULL)
    {
        if(!apply(c->pro, &pro, true))
            return 2;
        if(!rocke_gemm_prologue_check(&pro, reason, sizeof reason))
        {
            fprintf(stderr, "prologue: %s\n", reason);
            return 1;
        }
        spec._prologue = &pro;
    }
    if(!rocke_gemm_universal_is_valid_spec(&spec, c->arch, reason, sizeof reason))
    {
        fprintf(stderr, "invalid spec: %s\n", reason);
        return 1;
    }
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof b);
    rocke_kernel_def_t* kernel = rocke_build_universal_gemm_new(&b, &spec, c->arch);
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
            kernel, ROCKE_LLVM_FLAVOR_AUTO, c->arch, &llvm_text, err, sizeof err);
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
