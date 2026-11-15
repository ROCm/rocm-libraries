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
