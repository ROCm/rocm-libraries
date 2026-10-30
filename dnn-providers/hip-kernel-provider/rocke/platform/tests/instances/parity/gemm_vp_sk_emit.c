/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
/*
 * tests/parity/gemm_vp_sk_emit.c -- C-side emitter of the gemm_vp_sk parity family
 * (the register-staged global prefetch body of the universal GEMM; see
 * _gemm_vp_common.py). Builds the specs of gemm_vp_sk_emit.py (same order, same
 * values) through the C API so run_diff.py can byte-compare the two engines.
 * The case rows are generated: `python gemm_vp_sk_emit.py --c-cases`.
 *
 * split_k_aligned and split_k_serial outside the register-staged body: the global-load compv4 body (whole-tile slices with pad_k) and the direct_to_lds ring.
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
    {"sk0",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk1",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk2",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk3",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk4",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk5",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk6",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,split_k_aligned=true",
     NULL,
     0},
    {"sk7",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,split_k_aligned=true",
     NULL,
     0},
    {"sk8",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,split_k_aligned=true",
     NULL,
     0},
    {"sk9",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk10",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk11",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk12",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,split_k_aligned=true",
     NULL,
     0},
    {"sk13",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,split_k_aligned=true",
     NULL,
     0},
    {"sk14",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,split_k_aligned=true",
     NULL,
     0},
    {"sk15",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk16",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk17",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk18",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk19",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk20",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk21",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,split_k_aligned=true",
     NULL,
     0},
    {"sk22",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,split_k_aligned=true",
     NULL,
     0},
    {"sk23",
     "gfx942",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,split_k_aligned=true",
     NULL,
     0},
    {"sk24",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk25",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk26",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk27",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk28",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk29",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk30",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,split_k_aligned=true",
     NULL,
     0},
    {"sk31",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,split_k_aligned=true",
     NULL,
     0},
    {"sk32",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,split_k_aligned=true",
     NULL,
     0},
    {"sk33",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk34",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk35",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk36",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk37",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk38",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk39",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,split_k_aligned=true",
     NULL,
     0},
    {"sk40",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,split_k_aligned=true",
     NULL,
     0},
    {"sk41",
     "gfx950",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,split_k_aligned=true",
     NULL,
     0},
    {"sk42",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk43",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk44",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk45",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk46",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk47",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk48",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,split_k_aligned=true",
     NULL,
     0},
    {"sk49",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,split_k_aligned=true",
     NULL,
     0},
    {"sk50",
     "gfx950",
     "fp16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,split_k_aligned=true",
     NULL,
     0},
    {"sk51",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk52",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk53",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk54",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk55",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk56",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk57",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk58",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk59",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk60",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk61",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk62",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk63",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk64",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk65",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk66",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk67",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk68",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk69",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk70",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk71",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk72",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk73",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk74",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk75",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk76",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk77",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk78",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk79",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk80",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk81",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk82",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk83",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk84",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk85",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk86",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk87",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk88",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk89",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk90",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk91",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk92",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk93",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk94",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk95",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk96",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk97",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk98",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk99",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk100",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk101",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk102",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk103",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk104",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk105",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk106",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk107",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk108",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk109",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk110",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk111",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk112",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk113",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk114",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk115",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk116",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk117",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk118",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk119",
     "gfx950",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk120",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk121",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk122",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk123",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk124",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk125",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk126",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk127",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk128",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk129",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk130",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk131",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk132",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk133",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk134",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk135",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk136",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk137",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk138",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk139",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk140",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk141",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk142",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk143",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk144",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=2,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk145",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk146",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=3,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk147",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk148",
     "gfx942",
     "fp16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_k=true,split_k=8,ab_double_buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk149",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk150",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk151",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk152",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk153",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk154",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk155",
     "gfx942",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk156",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk157",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk158",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk159",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk160",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk161",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk162",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk163",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk164",
     "gfx942",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk165",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk166",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk167",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk168",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk169",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk170",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk171",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk172",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk173",
     "gfx950",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk174",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk175",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk176",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk177",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk178",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk179",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk180",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk181",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk182",
     "gfx950",
     "bf16",
     {256, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk183",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk184",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk185",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=true,split_k_aligned=true",
     NULL,
     0},
    {"sk186",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk187",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk188",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,ab_double_"
     "buffer=false,split_k_aligned=true",
     NULL,
     0},
    {"sk189",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk190",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk191",
     "gfx950",
     "bf16",
     {64, 64, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,pad_k=true,lds_swizzle=true,split_k=8,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk192",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk193",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk194",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk195",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk196",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk197",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk198",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk199",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk200",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk201",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk202",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk203",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk204",
     "gfx950",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk205",
     "gfx950",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk206",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk207",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk208",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk209",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk210",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk211",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk212",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk213",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk214",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk215",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk216",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk217",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk218",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk219",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk220",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk221",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk222",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk223",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk224",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk225",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk226",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk227",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk228",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk229",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk230",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk231",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk232",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk233",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk234",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk235",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk236",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk237",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk238",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk239",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk240",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk241",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk242",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk243",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk244",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk245",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk246",
     "gfx942",
     "fp16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk247",
     "gfx942",
     "bf16",
     {256, 256, 64, 2, 2, 1, 16, 16, 16},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk248",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk249",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk250",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk251",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk252",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk253",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk254",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk255",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk256",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "2",
     NULL,
     0},
    {"sk257",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_serial="
     "3",
     NULL,
     0},
    {"sk258",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk259",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=2,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk260",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk261",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=2,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk262",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk263",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk264",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk265",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk266",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk267",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk268",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=2,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
     NULL,
     0},
    {"sk269",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true",
     NULL,
     0},
    {"sk270",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "2",
     NULL,
     0},
    {"sk271",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_serial="
     "3",
     NULL,
     0},
    {"sk272",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,split_k_aligned="
     "true,split_k_serial=2",
     NULL,
     0},
    {"sk273",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,pad_m=true,pad_n=true,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_"
     "stages=3,dtl_tight_rsrc=true,split_k_aligned=true,split_k_serial=3",
     NULL,
     0},
    {"sk274",
     "gfx942",
     "fp16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,acc_agpr=true,"
     "split_k_serial=2",
     NULL,
     0},
    {"sk275",
     "gfx942",
     "bf16",
     {128, 128, 64, 2, 2, 1, 16, 16, 32},
     256,
     "epilogue=default,direct_to_lds=true,dtl_prefetch=true,split_k=4,dtl_stages=3,dtl_frag_"
     "prefetch=true,split_k_serial=2",
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
        fprintf(stderr, "usage: gemm_vp_sk_emit <config_index> [ll|ir|verify]\n");
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
