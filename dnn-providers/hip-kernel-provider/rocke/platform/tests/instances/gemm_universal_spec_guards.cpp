// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only spec-validation test for the universal GEMM's gfx1250 rules: fp8/bf8
 * storage, the Tensor Data Mover and the TDM K-loop traits. It runs each rule
 * through rocke_gemm_universal_is_valid_spec and checks the verdict and the
 * reason string, so the C++ validator is under CI without the pybind
 * rocke_engine (which CI does not build).
 *
 * The cases mirror the Python validator: each reason below is the one
 * gemm_universal.is_valid_spec returns for the same spec, and
 * tests/instances/test_gemm_tdm_traits_cpp_engine.py compares the two engines
 * directly where the binding is built. The byte-identity gate only compares
 * specs both engines accept, so it cannot see an accept/reject drift.
 *
 * One case is C++-only on purpose: the C++ engine does not emit the wmma_v1
 * schedule, so it rejects that pipeline, which Python accepts.
 */
#include <stdio.h>
#include <string.h>

#include <functional>
#include <vector>

#include "rocke/instance_gemm_universal.h"

namespace
{

int g_failures = 0;

using Spec = rocke_gemm_universal_spec_t;
using Mutate = std::function<void(Spec&)>;

struct Case
{
    const char* label;
    const char* arch;
    Mutate mutate;
    const char* reason; /* expected substring; nullptr => must be accepted */
};

/* The fp8 square-sweep champion geometry with the default epilogue, as
 * test_gemm_validity_gaps._fp8_spec(epilogue="default"). Valid on gfx1250. */
Spec base_spec()
{
    Spec s = rocke_gemm_universal_spec_default();
    s.name = "spec_guard";
    s.tile.tile_m = 256;
    s.tile.tile_n = 256;
    s.tile.tile_k = 256;
    s.tile.warp_m = 4;
    s.tile.warp_n = 8;
    s.tile.warp_k = 1;
    s.tile.warp_tile_m = 16;
    s.tile.warp_tile_n = 16;
    s.tile.warp_tile_k = 64;
    s.trait.pipeline = "mem";
    s.trait.scheduler = "intrawave";
    s.trait.epilogue = "default";
    s.trait.tdm = true;
    s.trait.tdm_depth = 2;
    s.trait.lds_k_pad = 48;
    s.data.dtype_a = "fp8e4m3";
    s.data.dtype_b = "fp8e4m3";
    s.data.dtype_c = "bf16";
    s.data.dtype_acc = "fp32";
    s.data.layout = "RCR";
    s.wave_size = 32;
    s.block_size = 1024;
    return s;
}

void bf16_16x16x16(Spec& s)
{
    s.data.dtype_a = "bf16";
    s.data.dtype_b = "bf16";
    s.data.dtype_c = "bf16";
    s.tile.warp_tile_k = 16;
}

void no_tdm(Spec& s)
{
    s.trait.tdm = false;
    s.trait.tdm_depth = 1;
}

/* A gfx1201-sized bf16 WMMA tile (64x64x32, 2x2 waves, no TDM): the base
 * spec's 256x256x256 A/B tile overflows gfx1201's LDS. */
void gfx1201_tile(Spec& s)
{
    bf16_16x16x16(s);
    no_tdm(s);
    s.trait.lds_k_pad = 0;
    s.tile.tile_m = 64;
    s.tile.tile_n = 64;
    s.tile.tile_k = 32;
    s.tile.warp_m = 2;
    s.tile.warp_n = 2;
    s.block_size = 128;
}

/* Exactly 64 KiB for two padded-free A/B buffers; the default two-workgroup
 * budget cannot double-buffer it. */
void gfx1201_db1wg_tile(Spec& s)
{
    gfx1201_tile(s);
    s.tile.tile_m = 128;
    s.tile.tile_n = 128;
    s.tile.tile_k = 64;
    s.trait.pipeline = "compv4";
    s.trait.lds_db_allow_1wg = true;
    s.trait.lds_store_after_mma = true;
}

/* The FP16 128x128x32 compv4 tile the prepared-read addressing is admitted on. */
void gfx1201_prepared_read_tile(Spec& s)
{
    gfx1201_tile(s);
    s.data.dtype_a = "fp16";
    s.data.dtype_b = "fp16";
    s.data.dtype_c = "fp16";
    s.tile.tile_m = 128;
    s.tile.tile_n = 128;
    s.trait.pipeline = "compv4";
    s.trait.global_addressing = "auto";
}

/* The measured direct-A tile: 128x64x64, four waves each owning 32 M rows. */
void gfx1201_direct_a_tile(Spec& s)
{
    gfx1201_tile(s);
    s.tile.tile_m = 128;
    s.tile.tile_n = 64;
    s.tile.tile_k = 64;
    s.tile.warp_m = 4;
    s.tile.warp_n = 1;
    s.trait.pipeline = "direct_a";
    s.trait.lds_swizzle = true;
}

void run(const Case& c)
{
    Spec s = base_spec();
    c.mutate(s);
    rocke_gemm_universal_spec_finalize(&s);
    char reason[512] = {0};
    bool ok = rocke_gemm_universal_is_valid_spec(&s, c.arch, reason, sizeof(reason));
    if(c.reason == nullptr)
    {
        if(!ok)
        {
            fprintf(stderr, "FAIL %s: rejected (%s)\n", c.label, reason);
            ++g_failures;
            return;
        }
        printf("ok   %s: accepted\n", c.label);
        return;
    }
    if(ok)
    {
        fprintf(stderr, "FAIL %s: accepted, expected \"%s\"\n", c.label, c.reason);
        ++g_failures;
        return;
    }
    if(strstr(reason, c.reason) == nullptr)
    {
        fprintf(stderr, "FAIL %s: expected \"%s\" in \"%s\"\n", c.label, c.reason, reason);
        ++g_failures;
        return;
    }
    printf("ok   %s: %s\n", c.label, reason);
}

} // namespace

int main()
{
    const std::vector<Case> cases = {
        {"base", "gfx1250", [](Spec&) {}, nullptr},
        /* fp8/bf8 storage */
        {"fp8 A/B mismatch",
         "gfx1250",
         [](Spec& s) { s.data.dtype_b = "bf8e5m2"; },
         "fp8/bf8 GEMM requires matching A/B dtypes; got A=fp8e4m3, B=bf8e5m2"},
        {"fp8 16x16x128 atom", "gfx1250", [](Spec& s) { s.tile.warp_tile_k = 128; }, nullptr},
        {"bf8 16x16x64 atom",
         "gfx1250",
         [](Spec& s) {
             s.data.dtype_a = "bf8e5m2";
             s.data.dtype_b = "bf8e5m2";
         },
         nullptr},
        {"bf8 16x16x128 atom: listed, no A/B lane map",
         "gfx1250",
         [](Spec& s) {
             s.data.dtype_a = "bf8e5m2";
             s.data.dtype_b = "bf8e5m2";
             s.tile.warp_tile_k = 128;
         },
         "no verified A/B lane map for bf8e5m2 warp_tile (16, 16, 128) on gfx1250"},
        {"fp8 with fp32 C",
         "gfx1250",
         [](Spec& s) { s.data.dtype_c = "fp32"; },
         "fp8/bf8 GEMM requires an f16/bf16 C dtype (there is no fp8 accumulator), got C=fp32"},
        {"fp8 off gfx1250",
         "gfx1201",
         [](Spec& s) {
             no_tdm(s);
             s.tile.warp_tile_k = 16;
         },
         "fp8/bf8 A/B GEMM is supported only on gfx1250 (got gfx1201)"},
        /* the Tensor Data Mover */
        {"tdm without a mover",
         "gfx1201",
         bf16_16x16x16,
         "tdm requires the Tensor Data Mover, which gfx1201 lacks"},
        /* the WMMA path */
        {"wmma_v1 (C++ only: no wmma_v1 schedule)",
         "gfx1250",
         [](Spec& s) { s.trait.pipeline = "wmma_v1"; },
         "WMMA path supports only the 'mem' pipeline (got 'wmma_v1') on gfx1250"},
        {"compv4 on gfx1201 (double-buffered WMMA K loop)",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv4";
         },
         nullptr},
        {"compv3 on gfx1201",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv3";
         },
         "WMMA path supports only the 'mem' or 'compv4' pipeline (got 'compv3') on gfx1201"},
        {"lds_swizzle on gfx1201",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.lds_swizzle = true;
         },
         nullptr},
        {"K-loop traits on gfx1201 compv4",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv4";
             s.trait.lds_store_after_mma = true;
             s.trait.group_m = 4;
             s.trait.k_stagger = 8;
             s.trait.k_stagger_step = 4;
         },
         nullptr},
        {"one-workgroup double buffer at 64 KiB", "gfx1201", gfx1201_db1wg_tile, nullptr},
        {"64 KiB double buffer still needs opt-in",
         "gfx1201",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.trait.lds_db_allow_1wg = false;
         },
         "lds_store_after_mma needs the WMMA compv4 double-buffered K loop"},
        {"one-workgroup double buffer exceeds 64 KiB",
         "gfx1201",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.tile.tile_n = 192;
         },
         "lds_db_allow_1wg needs the gfx1201 WMMA compv4 double-buffered K loop"},
        {"one-workgroup double buffer charges row padding",
         "gfx1201",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.trait.lds_k_pad = 1;
         },
         "lds_db_allow_1wg needs the gfx1201 WMMA compv4 double-buffered K loop"},
        {"one-workgroup double buffer fits with row padding",
         "gfx1201",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.tile.tile_k = 32;
             s.trait.lds_k_pad = 16;
         },
         nullptr},
        {"one-workgroup policy on an already-fitting tile",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv4";
             s.trait.lds_db_allow_1wg = true;
         },
         nullptr},
        {"one-workgroup policy on mem",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.lds_db_allow_1wg = true;
             s.trait.lds_store_after_mma = true;
         },
         "lds_db_allow_1wg needs the gfx1201 WMMA compv4 double-buffered K loop"},
        {"one-workgroup policy off gfx1201",
         "gfx950",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.tile.warp_tile_m = 32;
             s.tile.warp_tile_n = 32;
             s.wave_size = 64;
             s.block_size = 256;
         },
         "lds_db_allow_1wg needs the gfx1201 WMMA compv4 double-buffered K loop"},
        /* prepared global reads: gfx1201 FP16 RCR compv4/default only */
        {"prepared reads, auto", "gfx1201", gfx1201_prepared_read_tile, nullptr},
        {"prepared reads, global64",
         "gfx1201",
         [](Spec& s) {
             gfx1201_prepared_read_tile(s);
             s.trait.global_addressing = "global64";
         },
         nullptr},
        {"prepared reads, unknown mode",
         "gfx1201",
         [](Spec& s) {
             gfx1201_prepared_read_tile(s);
             s.trait.global_addressing = "buffer";
         },
         "global_addressing must be legacy, auto, or global64"},
        {"prepared reads on mem",
         "gfx1201",
         [](Spec& s) {
             gfx1201_prepared_read_tile(s);
             s.trait.pipeline = "mem";
         },
         "global_addressing needs unpadded nonbatched gfx1201 FP16 RCR"},
        {"prepared reads on BF16",
         "gfx1201",
         [](Spec& s) {
             gfx1201_prepared_read_tile(s);
             bf16_16x16x16(s);
         },
         "global_addressing needs unpadded nonbatched gfx1201 FP16 RCR"},
        {"prepared reads with padding",
         "gfx1201",
         [](Spec& s) {
             gfx1201_prepared_read_tile(s);
             s.trait.pad_m = true;
         },
         "global_addressing needs unpadded nonbatched gfx1201 FP16 RCR"},
        /* direct_a: its own gate */
        {"direct_a, BF16", "gfx1201", gfx1201_direct_a_tile, nullptr},
        {"direct_a, BF16 in and FP32 out",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.data.dtype_c = "fp32";
         },
         nullptr},
        {"direct_a, 128x96x64",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_n = 96;
         },
         nullptr},
        {"direct_a with row strides",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.trait.row_strides = true;
         },
         nullptr},
        {"direct_a, one wave 16x16x128",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_m = 16;
             s.tile.tile_n = 16;
             s.tile.tile_k = 128;
             s.tile.warp_m = 1;
             s.block_size = 32;
         },
         nullptr},
        {"direct_a, two waves 32x32x128",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_m = 32;
             s.tile.tile_n = 32;
             s.tile.tile_k = 128;
             s.tile.warp_m = 2;
             s.block_size = 64;
         },
         nullptr},
        {"direct_a one-wave tile whose pending B vectors spill",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_m = 16;
             s.tile.tile_n = 64;
             s.tile.tile_k = 128;
             s.tile.warp_m = 1;
             s.block_size = 32;
         },
         "exceed the VGPR file"},
        {"direct_a two-wave K-256 tile spills",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_m = 32;
             s.tile.tile_n = 32;
             s.tile.tile_k = 256;
             s.tile.warp_m = 2;
             s.block_size = 64;
         },
         "exceed the VGPR file"},
        {"row strides off direct_a",
         "gfx1201",
         [](Spec& s) {
             gfx1201_prepared_read_tile(s);
             s.trait.row_strides = true;
         },
         "row_strides needs the direct_a pipeline"},
        {"direct_a off gfx1201",
         "gfx1151",
         gfx1201_direct_a_tile,
         "direct_a is a gfx1201 wave32 pipeline"},
        {"direct_a with waves sharing M rows",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.warp_m = 2;
             s.tile.warp_n = 2;
         },
         "direct_a requires warp_n=1"},
        {"direct_a with mixed A/B types",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.data.dtype_b = "fp16";
         },
         "direct_a requires FP16 or BF16 A and B of the same type"},
        {"direct_a with a C type it cannot store",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.data.dtype_c = "fp16";
         },
         "direct_a requires C in the A/B type or FP32"},
        {"direct_a with a trait it would ignore",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.trait.lds_store_after_mma = true;
         },
         "direct_a takes only group_m, k_stagger, lds_swizzle, row_strides and"},
        {"direct_a tile beyond the register file",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_n = 128;
         },
         "exceed the VGPR file"},
        {"direct_a swizzle on a row that is not a power of two of slots",
         "gfx1201",
         [](Spec& s) {
             gfx1201_direct_a_tile(s);
             s.tile.tile_k = 48;
         },
         "lds_swizzle needs a power-of-two count of 8-element slots per row"},
        {"one-workgroup policy with direct-to-LDS retains earlier rejection",
         "gfx1201",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.trait.direct_to_lds = true;
         },
         "WMMA path does not support direct_to_lds on gfx1201"},
        {"one-workgroup policy with cshuffle retains earlier rejection",
         "gfx1201",
         [](Spec& s) {
             gfx1201_db1wg_tile(s);
             s.trait.epilogue = "cshuffle";
         },
         "WMMA path supports the 'cshuffle' epilogue only on gfx1250, not gfx1201"},
        {"lds_store_after_mma on mem",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.lds_store_after_mma = true;
         },
         "lds_store_after_mma needs the WMMA compv4 double-buffered K loop"},
        {"k_stagger on mem",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.k_stagger = 8;
         },
         "k_stagger needs the WMMA compv4 double-buffered K loop"},
        {"k_stagger with split_k",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv4";
             s.trait.k_stagger = 8;
             s.trait.split_k = 2;
         },
         "k_stagger does not support split_k"},
        {"k_stagger with pad_k",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv4";
             s.trait.k_stagger = 8;
             s.trait.pad_k = true;
         },
         "k_stagger does not support pad_k"},
        {"k_stagger_step 0",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.pipeline = "compv4";
             s.trait.k_stagger = 2;
             s.trait.k_stagger_step = 0;
         },
         "k_stagger must be >= 0 and k_stagger_step >= 1 (got 2, 0)"},
        {"negative group_m",
         "gfx1201",
         [](Spec& s) {
             gfx1201_tile(s);
             s.trait.group_m = -1;
         },
         "group_m must be >= 0 (got -1)"},
        {"cshuffle off gfx1250",
         "gfx1201",
         [](Spec& s) {
             bf16_16x16x16(s);
             no_tdm(s);
             s.trait.lds_k_pad = 0;
             s.trait.epilogue = "cshuffle";
         },
         "WMMA path supports the 'cshuffle' epilogue only on gfx1250, not gfx1201"},
        {"lds_swizzle",
         "gfx1250",
         [](Spec& s) { s.trait.lds_swizzle = true; },
         "WMMA path does not support lds_swizzle on gfx1250"},
        {"direct_to_lds",
         "gfx1250",
         [](Spec& s) { s.trait.direct_to_lds = true; },
         "WMMA path does not support direct_to_lds on gfx1250"},
        /* ab_load_elem_bytes */
        {"ab_load_elem_bytes 3",
         "gfx1250",
         [](Spec& s) {
             no_tdm(s);
             s.trait.ab_load_elem_bytes_set = true;
             s.trait.ab_load_elem_bytes = 3;
         },
         "ab_load_elem_bytes must be 1, 2, 4 or 8 (got 3)"},
        {"ab_load_elem_bytes with tdm",
         "gfx1250",
         [](Spec& s) {
             s.trait.ab_load_elem_bytes_set = true;
             s.trait.ab_load_elem_bytes = 2;
         },
         "ab_load_elem_bytes has no effect with tdm"},
        /* TDM geometry and depth */
        {"tdm lds_k_pad not a dword",
         "gfx1250",
         [](Spec& s) { s.trait.lds_k_pad = 3; },
         "tdm cannot encode lds_k_pad: TDM padding is dword-granular"},
        {"tdm 4-byte row",
         "gfx1250",
         [](Spec& s) {
             s.trait.lds_k_pad = 0;
             s.tile.tile_k = 4;
         },
         "tdm needs a dword-multiple tile row (got 4B)"},
        {"tdm_depth 5",
         "gfx1250",
         [](Spec& s) { s.trait.tdm_depth = 5; },
         "tdm_depth must be in 1..4 (got 5)"},
        {"tdm_depth without tdm",
         "gfx1250",
         [](Spec& s) { s.trait.tdm = false; },
         "tdm_depth is only meaningful with tdm=True"},
        /* K-loop traits */
        {"split_barrier at depth 1",
         "gfx1250",
         [](Spec& s) {
             s.trait.split_barrier = true;
             s.trait.tdm_depth = 1;
         },
         "split_barrier needs tdm=True with tdm_depth >= 2"},
        {"split_barrier with split_k",
         "gfx1250",
         [](Spec& s) {
             s.trait.split_barrier = true;
             s.trait.split_k = 2;
         },
         "split_barrier does not support split_k"},
        {"alternate issuers at depth 3",
         "gfx1250",
         [](Spec& s) {
             s.trait.tdm_alternate_issuers = true;
             s.trait.tdm_depth = 3;
         },
         "tdm_alternate_issuers needs tdm=True with tdm_depth == 2"},
        {"alternate issuers with two waves",
         "gfx1250",
         [](Spec& s) {
             s.trait.tdm_alternate_issuers = true;
             s.tile.tile_m = 32;
             s.tile.tile_n = 32;
             s.tile.warp_m = 2;
             s.tile.warp_n = 1;
             s.block_size = 64;
         },
         "tdm_alternate_issuers needs at least four waves"},
        {"early refill without alternate issuers",
         "gfx1250",
         [](Spec& s) {
             s.trait.tdm_early_refill = true;
             s.trait.split_barrier = true;
         },
         "tdm_early_refill needs split_barrier and tdm_alternate_issuers"},
        {"early refill without split_barrier",
         "gfx1250",
         [](Spec& s) {
             s.trait.tdm_early_refill = true;
             s.trait.tdm_alternate_issuers = true;
         },
         "tdm_early_refill needs split_barrier and tdm_alternate_issuers"},
        {"all three K-loop traits",
         "gfx1250",
         [](Spec& s) {
             s.trait.split_barrier = true;
             s.trait.tdm_alternate_issuers = true;
             s.trait.tdm_early_refill = true;
         },
         nullptr},
        /* gfx1250 cshuffle LDS budget: C aliases A/B unless cshuffle_no_alias */
        {"cshuffle, C aliased onto A/B",
         "gfx1250",
         [](Spec& s) { s.trait.epilogue = "cshuffle"; },
         nullptr},
        {"cshuffle, cshuffle_no_alias",
         "gfx1250",
         [](Spec& s) {
             s.trait.epilogue = "cshuffle";
             s.trait.cshuffle_no_alias = true;
         },
         "LDS budget 442368 > 327680 cap (AB=311296, C=131072) on gfx1250"},
    };

    for(const Case& c : cases)
    {
        run(c);
    }
    if(g_failures != 0)
    {
        fprintf(stderr, "%d of %zu cases FAILED\n", g_failures, cases.size());
        return 1;
    }
    printf("all %zu cases passed\n", cases.size());
    return 0;
}
