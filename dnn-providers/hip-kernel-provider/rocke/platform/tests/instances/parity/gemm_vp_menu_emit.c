/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/gemm_vp_menu_emit.c -- C-side emitter of the gemm_vp_menu parity family
 * (the register-staged global prefetch body of the universal GEMM; see
 * _gemm_vp_common.py). Builds the specs of gemm_vp_menu_emit.py (same order, same
 * values) through the C API so run_diff.py can byte-compare the two engines.
 * The case rows are generated: `python gemm_vp_menu_emit.py --c-cases`.
 *
 * every gfx942 g5_plan menu geometry once without and once with the G5P_MAINLOOP main-loop levers (vp_pair_reads, vp_release_atoms, vp_tail_pack, vp_mfma_first).
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
    {"mn0",
     "gfx942",
     "fp16",
     {112, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=3",
     NULL,
     0},
    {"mn1",
     "gfx942",
     "bf16",
     {112, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn2",
     "gfx942",
     "bf16",
     {112, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn3",
     "gfx942",
     "fp16",
     {112, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn4",
     "gfx942",
     "fp16",
     {112, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn5",
     "gfx942",
     "bf16",
     {112, 160, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn6",
     "gfx942",
     "bf16",
     {112, 160, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn7",
     "gfx942",
     "fp16",
     {112, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn8",
     "gfx942",
     "fp16",
     {112, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn9",
     "gfx942",
     "bf16",
     {112, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn10",
     "gfx942",
     "bf16",
     {112, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn11",
     "gfx942",
     "fp16",
     {112, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn12",
     "gfx942",
     "fp16",
     {112, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn13",
     "gfx942",
     "bf16",
     {112, 224, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn14",
     "gfx942",
     "bf16",
     {112, 224, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn15",
     "gfx942",
     "fp16",
     {112, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn16",
     "gfx942",
     "fp16",
     {112, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn17",
     "gfx942",
     "bf16",
     {112, 288, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn18",
     "gfx942",
     "bf16",
     {112, 288, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn19",
     "gfx942",
     "fp16",
     {112, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn20",
     "gfx942",
     "fp16",
     {112, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn21",
     "gfx942",
     "bf16",
     {112, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn22",
     "gfx942",
     "bf16",
     {112, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn23",
     "gfx942",
     "fp16",
     {112, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn24",
     "gfx942",
     "fp16",
     {112, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn25",
     "gfx942",
     "bf16",
     {112, 48, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn26",
     "gfx942",
     "bf16",
     {112, 48, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn27",
     "gfx942",
     "fp16",
     {112, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn28",
     "gfx942",
     "fp16",
     {112, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn29",
     "gfx942",
     "bf16",
     {112, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn30",
     "gfx942",
     "bf16",
     {112, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn31",
     "gfx942",
     "fp16",
     {112, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn32",
     "gfx942",
     "fp16",
     {112, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn33",
     "gfx942",
     "bf16",
     {112, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn34",
     "gfx942",
     "bf16",
     {112, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn35",
     "gfx942",
     "fp16",
     {112, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn36",
     "gfx942",
     "fp16",
     {112, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn37",
     "gfx942",
     "bf16",
     {112, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn38",
     "gfx942",
     "bf16",
     {112, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn39",
     "gfx942",
     "fp16",
     {128, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn40",
     "gfx942",
     "fp16",
     {128, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn41",
     "gfx942",
     "bf16",
     {128, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn42",
     "gfx942",
     "bf16",
     {128, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn43",
     "gfx942",
     "fp16",
     {128, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn44",
     "gfx942",
     "fp16",
     {128, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn45",
     "gfx942",
     "bf16",
     {128, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn46",
     "gfx942",
     "bf16",
     {128, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn47",
     "gfx942",
     "fp16",
     {128, 160, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn48",
     "gfx942",
     "fp16",
     {128, 160, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn49",
     "gfx942",
     "bf16",
     {128, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn50",
     "gfx942",
     "bf16",
     {128, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn51",
     "gfx942",
     "fp16",
     {128, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn52",
     "gfx942",
     "fp16",
     {128, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn53",
     "gfx942",
     "bf16",
     {128, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn54",
     "gfx942",
     "bf16",
     {128, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn55",
     "gfx942",
     "fp16",
     {128, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn56",
     "gfx942",
     "fp16",
     {128, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn57",
     "gfx942",
     "bf16",
     {128, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn58",
     "gfx942",
     "bf16",
     {128, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn59",
     "gfx942",
     "fp16",
     {128, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn60",
     "gfx942",
     "fp16",
     {128, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn61",
     "gfx942",
     "bf16",
     {128, 320, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn62",
     "gfx942",
     "bf16",
     {128, 320, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn63",
     "gfx942",
     "fp16",
     {128, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn64",
     "gfx942",
     "fp16",
     {128, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn65",
     "gfx942",
     "bf16",
     {128, 384, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn66",
     "gfx942",
     "bf16",
     {128, 384, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn67",
     "gfx942",
     "fp16",
     {128, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn68",
     "gfx942",
     "fp16",
     {128, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn69",
     "gfx942",
     "bf16",
     {128, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn70",
     "gfx942",
     "bf16",
     {128, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn71",
     "gfx942",
     "fp16",
     {128, 64, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn72",
     "gfx942",
     "fp16",
     {128, 64, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn73",
     "gfx942",
     "bf16",
     {128, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn74",
     "gfx942",
     "bf16",
     {128, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn75",
     "gfx942",
     "fp16",
     {128, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn76",
     "gfx942",
     "fp16",
     {128, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn77",
     "gfx942",
     "bf16",
     {128, 96, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn78",
     "gfx942",
     "bf16",
     {128, 96, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn79",
     "gfx942",
     "fp16",
     {160, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn80",
     "gfx942",
     "fp16",
     {160, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn81",
     "gfx942",
     "bf16",
     {160, 112, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn82",
     "gfx942",
     "bf16",
     {160, 112, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn83",
     "gfx942",
     "fp16",
     {160, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn84",
     "gfx942",
     "fp16",
     {160, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn85",
     "gfx942",
     "bf16",
     {160, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn86",
     "gfx942",
     "bf16",
     {160, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn87",
     "gfx942",
     "fp16",
     {160, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn88",
     "gfx942",
     "fp16",
     {160, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn89",
     "gfx942",
     "bf16",
     {160, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn90",
     "gfx942",
     "bf16",
     {160, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn91",
     "gfx942",
     "fp16",
     {160, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn92",
     "gfx942",
     "fp16",
     {160, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn93",
     "gfx942",
     "bf16",
     {160, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn94",
     "gfx942",
     "bf16",
     {160, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn95",
     "gfx942",
     "fp16",
     {160, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn96",
     "gfx942",
     "fp16",
     {160, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn97",
     "gfx942",
     "bf16",
     {160, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn98",
     "gfx942",
     "bf16",
     {160, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn99",
     "gfx942",
     "fp16",
     {160, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn100",
     "gfx942",
     "fp16",
     {160, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn101",
     "gfx942",
     "bf16",
     {160, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn102",
     "gfx942",
     "bf16",
     {160, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn103",
     "gfx942",
     "fp16",
     {160, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn104",
     "gfx942",
     "fp16",
     {160, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn105",
     "gfx942",
     "bf16",
     {160, 384, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn106",
     "gfx942",
     "bf16",
     {160, 384, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn107",
     "gfx942",
     "fp16",
     {160, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn108",
     "gfx942",
     "fp16",
     {160, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn109",
     "gfx942",
     "bf16",
     {160, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn110",
     "gfx942",
     "bf16",
     {160, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn111",
     "gfx942",
     "fp16",
     {160, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn112",
     "gfx942",
     "fp16",
     {160, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn113",
     "gfx942",
     "bf16",
     {160, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn114",
     "gfx942",
     "bf16",
     {160, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn115",
     "gfx942",
     "fp16",
     {160, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn116",
     "gfx942",
     "fp16",
     {160, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn117",
     "gfx942",
     "bf16",
     {160, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn118",
     "gfx942",
     "bf16",
     {160, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn119",
     "gfx942",
     "fp16",
     {16, 112, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn120",
     "gfx942",
     "fp16",
     {16, 112, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn121",
     "gfx942",
     "bf16",
     {16, 128, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn122",
     "gfx942",
     "bf16",
     {16, 128, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn123",
     "gfx942",
     "fp16",
     {16, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn124",
     "gfx942",
     "fp16",
     {16, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn125",
     "gfx942",
     "bf16",
     {16, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn126",
     "gfx942",
     "bf16",
     {16, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn127",
     "gfx942",
     "fp16",
     {16, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn128",
     "gfx942",
     "fp16",
     {16, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn129",
     "gfx942",
     "bf16",
     {16, 192, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn130",
     "gfx942",
     "bf16",
     {16, 192, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn131",
     "gfx942",
     "fp16",
     {16, 224, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn132",
     "gfx942",
     "fp16",
     {16, 224, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn133",
     "gfx942",
     "bf16",
     {16, 224, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn134",
     "gfx942",
     "bf16",
     {16, 224, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn135",
     "gfx942",
     "fp16",
     {16, 256, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn136",
     "gfx942",
     "fp16",
     {16, 256, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn137",
     "gfx942",
     "bf16",
     {16, 288, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn138",
     "gfx942",
     "bf16",
     {16, 288, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn139",
     "gfx942",
     "fp16",
     {16, 320, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn140",
     "gfx942",
     "fp16",
     {16, 320, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn141",
     "gfx942",
     "bf16",
     {16, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn142",
     "gfx942",
     "bf16",
     {16, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn143",
     "gfx942",
     "fp16",
     {16, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn144",
     "gfx942",
     "fp16",
     {16, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn145",
     "gfx942",
     "bf16",
     {16, 384, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn146",
     "gfx942",
     "bf16",
     {16, 384, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn147",
     "gfx942",
     "fp16",
     {16, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn148",
     "gfx942",
     "fp16",
     {16, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn149",
     "gfx942",
     "bf16",
     {16, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn150",
     "gfx942",
     "bf16",
     {16, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn151",
     "gfx942",
     "fp16",
     {16, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn152",
     "gfx942",
     "fp16",
     {16, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn153",
     "gfx942",
     "bf16",
     {16, 80, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn154",
     "gfx942",
     "bf16",
     {16, 80, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn155",
     "gfx942",
     "fp16",
     {16, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn156",
     "gfx942",
     "fp16",
     {16, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn157",
     "gfx942",
     "bf16",
     {192, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn158",
     "gfx942",
     "bf16",
     {192, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn159",
     "gfx942",
     "fp16",
     {192, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn160",
     "gfx942",
     "fp16",
     {192, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn161",
     "gfx942",
     "bf16",
     {192, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn162",
     "gfx942",
     "bf16",
     {192, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn163",
     "gfx942",
     "fp16",
     {192, 160, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn164",
     "gfx942",
     "fp16",
     {192, 160, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn165",
     "gfx942",
     "bf16",
     {192, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn166",
     "gfx942",
     "bf16",
     {192, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn167",
     "gfx942",
     "fp16",
     {192, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn168",
     "gfx942",
     "fp16",
     {192, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn169",
     "gfx942",
     "bf16",
     {192, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn170",
     "gfx942",
     "bf16",
     {192, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn171",
     "gfx942",
     "fp16",
     {192, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn172",
     "gfx942",
     "fp16",
     {192, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn173",
     "gfx942",
     "bf16",
     {192, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn174",
     "gfx942",
     "bf16",
     {192, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn175",
     "gfx942",
     "fp16",
     {192, 256, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn176",
     "gfx942",
     "fp16",
     {192, 256, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn177",
     "gfx942",
     "bf16",
     {192, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn178",
     "gfx942",
     "bf16",
     {192, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn179",
     "gfx942",
     "fp16",
     {192, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn180",
     "gfx942",
     "fp16",
     {192, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn181",
     "gfx942",
     "bf16",
     {192, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn182",
     "gfx942",
     "bf16",
     {192, 320, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn183",
     "gfx942",
     "fp16",
     {192, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn184",
     "gfx942",
     "fp16",
     {192, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn185",
     "gfx942",
     "bf16",
     {192, 384, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn186",
     "gfx942",
     "bf16",
     {192, 384, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn187",
     "gfx942",
     "fp16",
     {192, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn188",
     "gfx942",
     "fp16",
     {192, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn189",
     "gfx942",
     "bf16",
     {192, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn190",
     "gfx942",
     "bf16",
     {192, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn191",
     "gfx942",
     "fp16",
     {192, 64, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn192",
     "gfx942",
     "fp16",
     {192, 64, 128, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn193",
     "gfx942",
     "bf16",
     {192, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn194",
     "gfx942",
     "bf16",
     {192, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn195",
     "gfx942",
     "fp16",
     {192, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn196",
     "gfx942",
     "fp16",
     {192, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn197",
     "gfx942",
     "bf16",
     {192, 96, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn198",
     "gfx942",
     "bf16",
     {192, 96, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn199",
     "gfx942",
     "fp16",
     {224, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn200",
     "gfx942",
     "fp16",
     {224, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn201",
     "gfx942",
     "bf16",
     {224, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn202",
     "gfx942",
     "bf16",
     {224, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn203",
     "gfx942",
     "fp16",
     {224, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn204",
     "gfx942",
     "fp16",
     {224, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn205",
     "gfx942",
     "bf16",
     {224, 128, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn206",
     "gfx942",
     "bf16",
     {224, 128, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn207",
     "gfx942",
     "fp16",
     {224, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn208",
     "gfx942",
     "fp16",
     {224, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn209",
     "gfx942",
     "bf16",
     {224, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn210",
     "gfx942",
     "bf16",
     {224, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn211",
     "gfx942",
     "fp16",
     {224, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn212",
     "gfx942",
     "fp16",
     {224, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn213",
     "gfx942",
     "bf16",
     {224, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn214",
     "gfx942",
     "bf16",
     {224, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn215",
     "gfx942",
     "fp16",
     {224, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn216",
     "gfx942",
     "fp16",
     {224, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn217",
     "gfx942",
     "bf16",
     {224, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn218",
     "gfx942",
     "bf16",
     {224, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn219",
     "gfx942",
     "fp16",
     {224, 320, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn220",
     "gfx942",
     "fp16",
     {224, 320, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn221",
     "gfx942",
     "bf16",
     {224, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn222",
     "gfx942",
     "bf16",
     {224, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn223",
     "gfx942",
     "fp16",
     {224, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn224",
     "gfx942",
     "fp16",
     {224, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn225",
     "gfx942",
     "bf16",
     {224, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn226",
     "gfx942",
     "bf16",
     {224, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn227",
     "gfx942",
     "fp16",
     {224, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn228",
     "gfx942",
     "fp16",
     {224, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn229",
     "gfx942",
     "bf16",
     {224, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn230",
     "gfx942",
     "bf16",
     {224, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn231",
     "gfx942",
     "fp16",
     {224, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn232",
     "gfx942",
     "fp16",
     {224, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn233",
     "gfx942",
     "bf16",
     {256, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn234",
     "gfx942",
     "bf16",
     {256, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn235",
     "gfx942",
     "fp16",
     {256, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn236",
     "gfx942",
     "fp16",
     {256, 128, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn237",
     "gfx942",
     "bf16",
     {256, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn238",
     "gfx942",
     "bf16",
     {256, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn239",
     "gfx942",
     "fp16",
     {256, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn240",
     "gfx942",
     "fp16",
     {256, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn241",
     "gfx942",
     "bf16",
     {256, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn242",
     "gfx942",
     "bf16",
     {256, 192, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn243",
     "gfx942",
     "fp16",
     {256, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn244",
     "gfx942",
     "fp16",
     {256, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn245",
     "gfx942",
     "bf16",
     {256, 256, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,lds_swizzle_r=1,"
     "vgpr_prefetch=3,vp_unroll=1",
     NULL,
     0},
    {"mn246",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn247",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn248",
     "gfx942",
     "bf16",
     {256, 288, 32, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn249",
     "gfx942",
     "bf16",
     {256, 288, 32, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn250",
     "gfx942",
     "fp16",
     {256, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn251",
     "gfx942",
     "fp16",
     {256, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn252",
     "gfx942",
     "bf16",
     {256, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn253",
     "gfx942",
     "bf16",
     {256, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn254",
     "gfx942",
     "fp16",
     {256, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn255",
     "gfx942",
     "fp16",
     {256, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn256",
     "gfx942",
     "bf16",
     {256, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn257",
     "gfx942",
     "bf16",
     {256, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn258",
     "gfx942",
     "fp16",
     {256, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn259",
     "gfx942",
     "fp16",
     {256, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn260",
     "gfx942",
     "bf16",
     {256, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn261",
     "gfx942",
     "bf16",
     {256, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn262",
     "gfx942",
     "fp16",
     {256, 96, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn263",
     "gfx942",
     "fp16",
     {256, 96, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn264",
     "gfx942",
     "bf16",
     {288, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn265",
     "gfx942",
     "bf16",
     {288, 112, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn266",
     "gfx942",
     "fp16",
     {288, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn267",
     "gfx942",
     "fp16",
     {288, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn268",
     "gfx942",
     "bf16",
     {288, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn269",
     "gfx942",
     "bf16",
     {288, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn270",
     "gfx942",
     "fp16",
     {288, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn271",
     "gfx942",
     "fp16",
     {288, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn272",
     "gfx942",
     "bf16",
     {288, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn273",
     "gfx942",
     "bf16",
     {288, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn274",
     "gfx942",
     "fp16",
     {288, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn275",
     "gfx942",
     "fp16",
     {288, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn276",
     "gfx942",
     "bf16",
     {288, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn277",
     "gfx942",
     "bf16",
     {288, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn278",
     "gfx942",
     "fp16",
     {288, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn279",
     "gfx942",
     "fp16",
     {288, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn280",
     "gfx942",
     "bf16",
     {288, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn281",
     "gfx942",
     "bf16",
     {288, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn282",
     "gfx942",
     "fp16",
     {288, 80, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn283",
     "gfx942",
     "fp16",
     {288, 80, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn284",
     "gfx942",
     "bf16",
     {288, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn285",
     "gfx942",
     "bf16",
     {288, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn286",
     "gfx942",
     "fp16",
     {320, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn287",
     "gfx942",
     "fp16",
     {320, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn288",
     "gfx942",
     "bf16",
     {320, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn289",
     "gfx942",
     "bf16",
     {320, 32, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn290",
     "gfx942",
     "fp16",
     {320, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn291",
     "gfx942",
     "fp16",
     {320, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn292",
     "gfx942",
     "bf16",
     {320, 64, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn293",
     "gfx942",
     "bf16",
     {320, 64, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn294",
     "gfx942",
     "fp16",
     {320, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn295",
     "gfx942",
     "fp16",
     {320, 64, 64, 4, 2, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn296",
     "gfx942",
     "bf16",
     {320, 80, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn297",
     "gfx942",
     "bf16",
     {320, 80, 32, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn298",
     "gfx942",
     "fp16",
     {320, 80, 32, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn299",
     "gfx942",
     "fp16",
     {320, 80, 32, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "1,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn300",
     "gfx942",
     "bf16",
     {32, 112, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn301",
     "gfx942",
     "bf16",
     {32, 112, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn302",
     "gfx942",
     "fp16",
     {32, 128, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn303",
     "gfx942",
     "fp16",
     {32, 128, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn304",
     "gfx942",
     "bf16",
     {32, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn305",
     "gfx942",
     "bf16",
     {32, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn306",
     "gfx942",
     "fp16",
     {32, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn307",
     "gfx942",
     "fp16",
     {32, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn308",
     "gfx942",
     "bf16",
     {32, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn309",
     "gfx942",
     "bf16",
     {32, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn310",
     "gfx942",
     "fp16",
     {32, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn311",
     "gfx942",
     "fp16",
     {32, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn312",
     "gfx942",
     "bf16",
     {32, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn313",
     "gfx942",
     "bf16",
     {32, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn314",
     "gfx942",
     "fp16",
     {32, 192, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn315",
     "gfx942",
     "fp16",
     {32, 192, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn316",
     "gfx942",
     "bf16",
     {32, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn317",
     "gfx942",
     "bf16",
     {32, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn318",
     "gfx942",
     "fp16",
     {32, 224, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn319",
     "gfx942",
     "fp16",
     {32, 224, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn320",
     "gfx942",
     "bf16",
     {32, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn321",
     "gfx942",
     "bf16",
     {32, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn322",
     "gfx942",
     "fp16",
     {32, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn323",
     "gfx942",
     "fp16",
     {32, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn324",
     "gfx942",
     "bf16",
     {32, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn325",
     "gfx942",
     "bf16",
     {32, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn326",
     "gfx942",
     "fp16",
     {32, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn327",
     "gfx942",
     "fp16",
     {32, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn328",
     "gfx942",
     "bf16",
     {32, 32, 128, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn329",
     "gfx942",
     "bf16",
     {32, 32, 128, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn330",
     "gfx942",
     "fp16",
     {32, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn331",
     "gfx942",
     "fp16",
     {32, 32, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn332",
     "gfx942",
     "bf16",
     {32, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn333",
     "gfx942",
     "bf16",
     {32, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn334",
     "gfx942",
     "fp16",
     {32, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn335",
     "gfx942",
     "fp16",
     {32, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn336",
     "gfx942",
     "bf16",
     {32, 64, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn337",
     "gfx942",
     "bf16",
     {32, 64, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn338",
     "gfx942",
     "fp16",
     {32, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn339",
     "gfx942",
     "fp16",
     {32, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn340",
     "gfx942",
     "bf16",
     {32, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn341",
     "gfx942",
     "bf16",
     {32, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn342",
     "gfx942",
     "fp16",
     {32, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn343",
     "gfx942",
     "fp16",
     {32, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn344",
     "gfx942",
     "bf16",
     {32, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn345",
     "gfx942",
     "bf16",
     {32, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn346",
     "gfx942",
     "fp16",
     {48, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn347",
     "gfx942",
     "fp16",
     {48, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn348",
     "gfx942",
     "bf16",
     {48, 112, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn349",
     "gfx942",
     "bf16",
     {48, 112, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn350",
     "gfx942",
     "fp16",
     {48, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn351",
     "gfx942",
     "fp16",
     {48, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn352",
     "gfx942",
     "bf16",
     {48, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn353",
     "gfx942",
     "bf16",
     {48, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn354",
     "gfx942",
     "fp16",
     {48, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn355",
     "gfx942",
     "fp16",
     {48, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn356",
     "gfx942",
     "bf16",
     {48, 192, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn357",
     "gfx942",
     "bf16",
     {48, 192, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn358",
     "gfx942",
     "fp16",
     {48, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn359",
     "gfx942",
     "fp16",
     {48, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn360",
     "gfx942",
     "bf16",
     {48, 224, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn361",
     "gfx942",
     "bf16",
     {48, 224, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn362",
     "gfx942",
     "fp16",
     {48, 256, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn363",
     "gfx942",
     "fp16",
     {48, 256, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn364",
     "gfx942",
     "bf16",
     {48, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn365",
     "gfx942",
     "bf16",
     {48, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn366",
     "gfx942",
     "fp16",
     {48, 288, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn367",
     "gfx942",
     "fp16",
     {48, 288, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn368",
     "gfx942",
     "bf16",
     {48, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn369",
     "gfx942",
     "bf16",
     {48, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn370",
     "gfx942",
     "fp16",
     {48, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn371",
     "gfx942",
     "fp16",
     {48, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn372",
     "gfx942",
     "bf16",
     {48, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn373",
     "gfx942",
     "bf16",
     {48, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn374",
     "gfx942",
     "fp16",
     {48, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn375",
     "gfx942",
     "fp16",
     {48, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn376",
     "gfx942",
     "bf16",
     {48, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn377",
     "gfx942",
     "bf16",
     {48, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn378",
     "gfx942",
     "fp16",
     {48, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn379",
     "gfx942",
     "fp16",
     {48, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn380",
     "gfx942",
     "bf16",
     {48, 80, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn381",
     "gfx942",
     "bf16",
     {48, 80, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn382",
     "gfx942",
     "fp16",
     {48, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn383",
     "gfx942",
     "fp16",
     {48, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn384",
     "gfx942",
     "bf16",
     {64, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn385",
     "gfx942",
     "bf16",
     {64, 112, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn386",
     "gfx942",
     "fp16",
     {64, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn387",
     "gfx942",
     "fp16",
     {64, 128, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn388",
     "gfx942",
     "bf16",
     {64, 128, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn389",
     "gfx942",
     "bf16",
     {64, 128, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn390",
     "gfx942",
     "fp16",
     {64, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn391",
     "gfx942",
     "fp16",
     {64, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn392",
     "gfx942",
     "bf16",
     {64, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn393",
     "gfx942",
     "bf16",
     {64, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn394",
     "gfx942",
     "fp16",
     {64, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn395",
     "gfx942",
     "fp16",
     {64, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn396",
     "gfx942",
     "bf16",
     {64, 192, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn397",
     "gfx942",
     "bf16",
     {64, 192, 128, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn398",
     "gfx942",
     "fp16",
     {64, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn399",
     "gfx942",
     "fp16",
     {64, 192, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn400",
     "gfx942",
     "bf16",
     {64, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn401",
     "gfx942",
     "bf16",
     {64, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn402",
     "gfx942",
     "fp16",
     {64, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn403",
     "gfx942",
     "fp16",
     {64, 256, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn404",
     "gfx942",
     "bf16",
     {64, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn405",
     "gfx942",
     "bf16",
     {64, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn406",
     "gfx942",
     "fp16",
     {64, 320, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn407",
     "gfx942",
     "fp16",
     {64, 320, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn408",
     "gfx942",
     "bf16",
     {64, 32, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn409",
     "gfx942",
     "bf16",
     {64, 32, 128, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn410",
     "gfx942",
     "fp16",
     {64, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn411",
     "gfx942",
     "fp16",
     {64, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn412",
     "gfx942",
     "bf16",
     {64, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn413",
     "gfx942",
     "bf16",
     {64, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn414",
     "gfx942",
     "fp16",
     {64, 384, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn415",
     "gfx942",
     "fp16",
     {64, 384, 64, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn416",
     "gfx942",
     "bf16",
     {64, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn417",
     "gfx942",
     "bf16",
     {64, 48, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn418",
     "gfx942",
     "fp16",
     {64, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn419",
     "gfx942",
     "fp16",
     {64, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn420",
     "gfx942",
     "bf16",
     {64, 64, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn421",
     "gfx942",
     "bf16",
     {64, 64, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn422",
     "gfx942",
     "fp16",
     {64, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn423",
     "gfx942",
     "fp16",
     {64, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn424",
     "gfx942",
     "bf16",
     {64, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn425",
     "gfx942",
     "bf16",
     {64, 80, 64, 4, 1, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn426",
     "gfx942",
     "fp16",
     {64, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn427",
     "gfx942",
     "fp16",
     {64, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn428",
     "gfx942",
     "bf16",
     {64, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn429",
     "gfx942",
     "bf16",
     {64, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn430",
     "gfx942",
     "fp16",
     {80, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=3",
     NULL,
     0},
    {"mn431",
     "gfx942",
     "bf16",
     {80, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn432",
     "gfx942",
     "bf16",
     {80, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn433",
     "gfx942",
     "fp16",
     {80, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn434",
     "gfx942",
     "fp16",
     {80, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn435",
     "gfx942",
     "bf16",
     {80, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn436",
     "gfx942",
     "bf16",
     {80, 160, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn437",
     "gfx942",
     "fp16",
     {80, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn438",
     "gfx942",
     "fp16",
     {80, 16, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn439",
     "gfx942",
     "bf16",
     {80, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn440",
     "gfx942",
     "bf16",
     {80, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn441",
     "gfx942",
     "fp16",
     {80, 224, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn442",
     "gfx942",
     "fp16",
     {80, 224, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn443",
     "gfx942",
     "bf16",
     {80, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn444",
     "gfx942",
     "bf16",
     {80, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn445",
     "gfx942",
     "fp16",
     {80, 288, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn446",
     "gfx942",
     "fp16",
     {80, 288, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn447",
     "gfx942",
     "bf16",
     {80, 288, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4",
     NULL,
     0},
    {"mn448",
     "gfx942",
     "bf16",
     {80, 288, 32, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=4,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn449",
     "gfx942",
     "fp16",
     {80, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn450",
     "gfx942",
     "fp16",
     {80, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn451",
     "gfx942",
     "bf16",
     {80, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn452",
     "gfx942",
     "bf16",
     {80, 32, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn453",
     "gfx942",
     "fp16",
     {80, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn454",
     "gfx942",
     "fp16",
     {80, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn455",
     "gfx942",
     "bf16",
     {80, 48, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn456",
     "gfx942",
     "bf16",
     {80, 48, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn457",
     "gfx942",
     "fp16",
     {80, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn458",
     "gfx942",
     "fp16",
     {80, 48, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn459",
     "gfx942",
     "bf16",
     {80, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn460",
     "gfx942",
     "bf16",
     {80, 64, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn461",
     "gfx942",
     "fp16",
     {80, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn462",
     "gfx942",
     "fp16",
     {80, 64, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn463",
     "gfx942",
     "bf16",
     {80, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn464",
     "gfx942",
     "bf16",
     {80, 80, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn465",
     "gfx942",
     "fp16",
     {80, 80, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn466",
     "gfx942",
     "fp16",
     {80, 80, 64, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn467",
     "gfx942",
     "bf16",
     {80, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn468",
     "gfx942",
     "bf16",
     {80, 96, 64, 1, 2, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn469",
     "gfx942",
     "fp16",
     {96, 112, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn470",
     "gfx942",
     "fp16",
     {96, 112, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn471",
     "gfx942",
     "bf16",
     {96, 128, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn472",
     "gfx942",
     "bf16",
     {96, 128, 128, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn473",
     "gfx942",
     "fp16",
     {96, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn474",
     "gfx942",
     "fp16",
     {96, 128, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn475",
     "gfx942",
     "bf16",
     {96, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn476",
     "gfx942",
     "bf16",
     {96, 160, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn477",
     "gfx942",
     "fp16",
     {96, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn478",
     "gfx942",
     "fp16",
     {96, 160, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn479",
     "gfx942",
     "bf16",
     {96, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn480",
     "gfx942",
     "bf16",
     {96, 16, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn481",
     "gfx942",
     "fp16",
     {96, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn482",
     "gfx942",
     "fp16",
     {96, 192, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn483",
     "gfx942",
     "bf16",
     {96, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn484",
     "gfx942",
     "bf16",
     {96, 192, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn485",
     "gfx942",
     "fp16",
     {96, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn486",
     "gfx942",
     "fp16",
     {96, 224, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn487",
     "gfx942",
     "bf16",
     {96, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn488",
     "gfx942",
     "bf16",
     {96, 256, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn489",
     "gfx942",
     "fp16",
     {96, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn490",
     "gfx942",
     "fp16",
     {96, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn491",
     "gfx942",
     "bf16",
     {96, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn492",
     "gfx942",
     "bf16",
     {96, 288, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn493",
     "gfx942",
     "fp16",
     {96, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn494",
     "gfx942",
     "fp16",
     {96, 320, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn495",
     "gfx942",
     "bf16",
     {96, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn496",
     "gfx942",
     "bf16",
     {96, 32, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn497",
     "gfx942",
     "fp16",
     {96, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4",
     NULL,
     0},
    {"mn498",
     "gfx942",
     "fp16",
     {96, 384, 64, 1, 4, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,acc_agpr=true,"
     "lds_swizzle_r=0,vgpr_prefetch=4,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn499",
     "gfx942",
     "bf16",
     {96, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn500",
     "gfx942",
     "bf16",
     {96, 48, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn501",
     "gfx942",
     "fp16",
     {96, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn502",
     "gfx942",
     "fp16",
     {96, 64, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn503",
     "gfx942",
     "bf16",
     {96, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn504",
     "gfx942",
     "bf16",
     {96, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn505",
     "gfx942",
     "fp16",
     {96, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn506",
     "gfx942",
     "fp16",
     {96, 80, 64, 2, 1, 1, 16, 16, 16},
     128,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn507",
     "gfx942",
     "bf16",
     {96, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn508",
     "gfx942",
     "bf16",
     {96, 96, 128, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn509",
     "gfx942",
     "fp16",
     {96, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1",
     NULL,
     0},
    {"mn510",
     "gfx942",
     "fp16",
     {96, 96, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=false,lds_swizzle_r="
     "0,vgpr_prefetch=4,vp_unroll=1,vp_release_atoms=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_"
     "first=1",
     NULL,
     0},
    {"mn511",
     "gfx942",
     "fp16",
     {112, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn512",
     "gfx942",
     "bf16",
     {256, 256, 32, 2, 4, 1, 16, 16, 16},
     512,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,lds_swizzle_r=1,"
     "vgpr_prefetch=3,vp_unroll=1,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
     0},
    {"mn513",
     "gfx942",
     "fp16",
     {80, 112, 32, 1, 1, 1, 16, 16, 16},
     64,
     "epilogue=default,chiplet_swizzle=true,lds_swizzle=true,ab_double_buffer=true,acc_agpr=true,"
     "lds_swizzle_r=1,vgpr_prefetch=3,vp_tail_pack=3,vp_pair_reads=1,vp_mfma_first=1",
     NULL,
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
        fprintf(stderr, "usage: gemm_vp_menu_emit <config_index> [ll|ir|verify]\n");
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
