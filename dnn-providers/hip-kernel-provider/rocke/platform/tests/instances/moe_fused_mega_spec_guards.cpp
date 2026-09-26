// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only spec-validation test for the two C++ fused-MoE mega-kernel builders.
 * C mirror of tests/instances/test_moe_fused_mega.py::TestMoeFusedMegaLdsBudget
 * and tests/instances/test_moe_fused_mega_fp8.py, so the two engines agree on
 * which specs they ACCEPT (the byte-identity gate only compares the IR of specs
 * both engines already accept, so an acceptance divergence is invisible to it).
 *
 * Pinned here:
 *   1. the whole-kernel LDS accounting equals the @smem_pool the lowerer emits,
 *      for both megas across the tilings the Python tests cover -- if the two
 *      drift, the budget gate starts rejecting valid specs;
 *   2. an over-budget tiling is rejected by the fused total even though both
 *      GEMM sub-specs pass their own validator;
 *   3. the fp8 mega rejects the scaled-f8f6f4 hero atom on a target without the
 *      instruction, and accepts the catalog 16x16x32 atom there;
 *   4. the fp8 mega rejects a wave32 target with gemm_universal's phrasing.
 *
 * The two _lds_allocs ports live in the instances' *_internal.h headers (they
 * are private to the family, like the Python module-private `_lds_allocs` the
 * Python tests import); this test is part of the family and binds to them.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_moe_fused_mega.h"
#include "rocke/instance_moe_fused_mega_fp8.h"
#include "rocke/instance_moe_fused_mega_fp8_internal.h"
#include "rocke/instance_moe_fused_mega_internal.h"
#include "rocke/lower_llvm.h"

/* Shipped f16/bf16 mega tile geometry, measured from the lowered IR's
 * addrspace(3) pool (mirrors the Python test's SHIPPED_LDS_BYTES). */
#define MEGA_SHIPPED_LDS_BYTES 74752
#define MEGA_GFX942_LDS_CAP 65536

static int g_failures = 0;

static void fail(const char* what, const char* detail)
{
    fprintf(stderr, "FAIL %s: %s\n", what, (detail != NULL) ? detail : "<none>");
    ++g_failures;
}

static void expect_contains(const char* what, const char* haystack, const char* needle)
{
    if(haystack == NULL || strstr(haystack, needle) == NULL)
    {
        char detail[1024];
        snprintf(detail,
                 sizeof(detail),
                 "expected substring \"%s\" in \"%s\"",
                 needle,
                 (haystack != NULL) ? haystack : "");
        fail(what, detail);
        return;
    }
    printf("ok   %s (contains \"%s\")\n", what, needle);
}

/* The size of the single unified addrspace(3) pool the lowerer emits:
 *   @smem_pool.<kernel> = internal unnamed_addr addrspace(3) global [N x i8] ...
 * Returns -1 when the kernel allocates no LDS at all. */
static long smem_pool_bytes_from_ll(const char* ll)
{
    const char* p = (ll != NULL) ? strstr(ll, "@smem_pool.") : NULL;
    if(p == NULL)
    {
        return -1;
    }
    p = strchr(p, '[');
    if(p == NULL)
    {
        return -1;
    }
    return strtol(p + 1, NULL, 10);
}

/* ------------------------------------------------------------------ f16/bf16 */

static rocke_moe_fused_mega_kernel_spec_t mega_spec(void)
{
    rocke_moe_fused_mega_kernel_spec_t s = rocke_moe_fused_mega_kernel_spec_default();
    s.name = "mega_lds";
    s.dtype = "bf16";
    return s;
}

static long mega_accounted_bytes(const rocke_moe_fused_mega_kernel_spec_t* spec)
{
    rocke_gemm_universal_spec_t u_gu;
    rocke_gemm_universal_spec_t u_down;
    rocke_mega_lds_alloc_t allocs[ROCKE_MOE_MEGA_LDS_ALLOCS];

    rocke_moe_fused_mega_gate_up_universal_spec(spec, &u_gu);
    rocke_moe_fused_mega_down_universal_spec(spec, &u_down);
    rocke_moe_fused_mega_lds_allocs(&u_gu, &u_down, allocs);
    return rocke_mega_lds_pool_bytes(allocs, ROCKE_MOE_MEGA_LDS_ALLOCS);
}

/* Lower `spec` for `arch`. On success returns the malloc'd .ll (caller frees)
 * and leaves err empty; on rejection returns NULL with err holding the reason. */
static char* mega_lower(const rocke_moe_fused_mega_kernel_spec_t* spec,
                        const char* arch,
                        char* err,
                        size_t err_cap)
{
    char* ll = NULL;
    err[0] = '\0';
    if(rocke_moe_fused_mega_lower_to_llvm(spec, arch, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, err_cap)
       != ROCKE_OK)
    {
        free(ll);
        return NULL;
    }
    return ll;
}

static void check_mega_accounting_matches_pool(const char* what,
                                               const rocke_moe_fused_mega_kernel_spec_t* spec)
{
    char err[ROCKE_ERR_MSG_CAP];
    char* ll = mega_lower(spec, "gfx950", err, sizeof(err));
    long emitted;
    long accounted;

    if(ll == NULL)
    {
        fail(what, err);
        return;
    }
    emitted = smem_pool_bytes_from_ll(ll);
    accounted = mega_accounted_bytes(spec);
    free(ll);
    if(emitted != accounted)
    {
        char detail[256];
        snprintf(detail, sizeof(detail), "accounted %ld != emitted pool %ld", accounted, emitted);
        fail(what, detail);
        return;
    }
    printf("ok   %s (pool %ld B)\n", what, emitted);
}

static void test_mega_lds_budget(void)
{
    rocke_moe_fused_mega_kernel_spec_t spec;
    char err[ROCKE_ERR_MSG_CAP];
    char* ll;
    long accounted;

    /* 1. The accounting must equal the bytes the smem packer actually reserves,
     * otherwise the gate rejects valid specs (or passes invalid ones). */
    spec = mega_spec();
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    check_mega_accounting_matches_pool("mega/pool/default", &spec);

    spec = mega_spec();
    spec.tile_m = 32;
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    check_mega_accounting_matches_pool("mega/pool/tile_m=32", &spec);

    spec = mega_spec();
    spec.tile_n_down = 128;
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    check_mega_accounting_matches_pool("mega/pool/tile_n_down=128", &spec);

    spec = mega_spec();
    spec.tile_k_down = 32;
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    check_mega_accounting_matches_pool("mega/pool/tile_k_down=32", &spec);

    /* 2. The shipped geometry costs the measured bytes. */
    spec = mega_spec();
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    accounted = mega_accounted_bytes(&spec);
    if(accounted != MEGA_SHIPPED_LDS_BYTES)
    {
        char detail[128];
        snprintf(detail,
                 sizeof(detail),
                 "shipped geometry costs %ld B, expected %d B",
                 accounted,
                 MEGA_SHIPPED_LDS_BYTES);
        fail("mega/shipped_geometry", detail);
    }
    else
    {
        printf("ok   mega/shipped_geometry (%ld B)\n", accounted);
    }

    /* 3. warp_tile_k=16 is the gfx942-legal atom, so both GEMM sub-specs pass;
     * only the fused total catches this one. */
    spec = mega_spec();
    spec.warp_tile_k = 16;
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    ll = mega_lower(&spec, "gfx942", err, sizeof(err));
    if(ll != NULL)
    {
        free(ll);
        fail("mega/gfx942_over_budget", "expected rejection, spec was accepted");
    }
    else
    {
        char needle[32];
        snprintf(needle, sizeof(needle), "%d", MEGA_SHIPPED_LDS_BYTES);
        expect_contains("mega/gfx942_over_budget", err, needle);
        snprintf(needle, sizeof(needle), "%d", MEGA_GFX942_LDS_CAP);
        expect_contains("mega/gfx942_over_budget", err, needle);
        expect_contains("mega/gfx942_over_budget", err, "Hidden_smem");
    }

    /* 4. One field off the shipped geometry brings the total under 64 KiB. */
    spec = mega_spec();
    spec.warp_tile_k = 16;
    spec.tile_n_down = 128;
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    ll = mega_lower(&spec, "gfx942", err, sizeof(err));
    if(ll == NULL)
    {
        fail("mega/gfx942_fits", err);
    }
    else
    {
        expect_contains("mega/gfx942_fits", ll, "[58368 x i8]");
        expect_contains("mega/gfx942_fits", ll, "atomicrmw");
        free(ll);
    }

    /* 5. The shipped geometry fits CDNA4's 160 KiB; only gfx942 is over. */
    spec = mega_spec();
    rocke_moe_fused_mega_kernel_spec_finalize(&spec);
    ll = mega_lower(&spec, "gfx950", err, sizeof(err));
    if(ll == NULL)
    {
        fail("mega/gfx950_default_builds", err);
    }
    else
    {
        expect_contains("mega/gfx950_default_builds", ll, "mfma.f32.16x16x32.bf16");
        free(ll);
    }
}

/* ---------------------------------------------------------------------- fp8 */

static rocke_fused_mega_kernel_spec_fp8_t fp8_spec(void)
{
    rocke_fused_mega_kernel_spec_fp8_t s = rocke_fused_mega_kernel_spec_fp8_default();
    s.name = "mega_fp8";
    return s;
}

static char* fp8_lower(const rocke_fused_mega_kernel_spec_fp8_t* spec,
                       const char* arch,
                       char* err,
                       size_t err_cap)
{
    char* ll = NULL;
    err[0] = '\0';
    /* levers NULL => the Python import-time module defaults. */
    if(rocke_moe_fused_mega_fp8_lower_to_llvm(
           spec, arch, false, NULL, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, err_cap)
       != ROCKE_OK)
    {
        free(ll);
        return NULL;
    }
    return ll;
}

static void check_fp8_accounting_matches_pool(const char* what,
                                              const rocke_fused_mega_kernel_spec_fp8_t* spec)
{
    rocke_mega_lds_alloc_t allocs[ROCKE_MOE_FP8_LDS_MAX_ALLOCS];
    char err[ROCKE_ERR_MSG_CAP];
    char* ll = fp8_lower(spec, "gfx950", err, sizeof(err));
    long emitted;
    long accounted;
    size_t n;

    if(ll == NULL)
    {
        fail(what, err);
        return;
    }
    emitted = smem_pool_bytes_from_ll(ll);
    free(ll);
    n = rocke_moe_fp8_lds_allocs(spec, NULL, allocs);
    accounted = rocke_mega_lds_pool_bytes(allocs, n);
    if(emitted != accounted)
    {
        char detail[256];
        snprintf(detail, sizeof(detail), "accounted %ld != emitted pool %ld", accounted, emitted);
        fail(what, detail);
        return;
    }
    printf("ok   %s (pool %ld B, %zu allocs)\n", what, emitted, n);
}

static void test_fp8_guards(void)
{
    rocke_fused_mega_kernel_spec_fp8_t spec;
    char err[ROCKE_ERR_MSG_CAP];
    char* ll;
    size_t i;
    static const char* const wave32_arches[] = {"gfx1250", "gfx1151"};

    /* 1. The K=128 hero atom lowers to mfma.scale.f32.16x16x128.f8f6f4, which
     * gfx942 does not have. It is not a catalog shape anywhere, so the shared
     * catalog guard is skipped and this needs its own rejection -- otherwise the
     * CDNA4-only instruction reaches comgr as an uncatchable LLVM abort. */
    spec = fp8_spec();
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    ll = fp8_lower(&spec, "gfx942", err, sizeof(err));
    if(ll != NULL)
    {
        free(ll);
        fail("fp8/gfx942_hero_atom", "expected rejection, spec was accepted");
    }
    else
    {
        expect_contains("fp8/gfx942_hero_atom", err, "16x16x128");
        expect_contains("fp8/gfx942_hero_atom", err, "gfx942");
    }

    /* 2. The legacy K=32 path IS a gfx942 catalog shape, so it must still build.
     * down_k selects independently of gate_up_k, so both have to be lowered. */
    spec = fp8_spec();
    spec.gate_up_k = 32;
    spec.down_k = 32;
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    ll = fp8_lower(&spec, "gfx942", err, sizeof(err));
    if(ll == NULL)
    {
        fail("fp8/gfx942_catalog_atom", err);
    }
    else
    {
        expect_contains("fp8/gfx942_catalog_atom", ll, "atomicrmw");
        free(ll);
    }

    /* 2b. A hero DOWN atom under a catalog gate/up atom is rejected too. Both
     * atoms are selectable, and Python gates both; down_k does not currently
     * reach emission (the builder drives both stages off gate_up_atom()), so
     * this case is conservative on both sides -- what it pins is that the two
     * engines are conservative in the same way. */
    spec = fp8_spec();
    spec.gate_up_k = 32;
    spec.down_k = 128;
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    ll = fp8_lower(&spec, "gfx942", err, sizeof(err));
    if(ll != NULL)
    {
        free(ll);
        fail("fp8/gfx942_hero_down_atom", "expected rejection, spec was accepted");
    }
    else
    {
        expect_contains("fp8/gfx942_hero_down_atom", err, "16x16x128");
    }

    /* 3. Every lane map in the fp8 mega is wave64 (the amax butterfly is a
     * hardcoded 6-stage xor over lanes 1..32), so a wave32 target is silently
     * wrong. The phrasing matches gemm_universal's wave-size reject. */
    for(i = 0; i < sizeof(wave32_arches) / sizeof(wave32_arches[0]); ++i)
    {
        char needle[128];
        spec = fp8_spec();
        rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
        ll = fp8_lower(&spec, wave32_arches[i], err, sizeof(err));
        snprintf(needle, sizeof(needle), "spec wave_size 64 != %s wave_size 32", wave32_arches[i]);
        if(ll != NULL)
        {
            free(ll);
            fail("fp8/wave32_reject", "expected rejection, spec was accepted");
        }
        else
        {
            expect_contains("fp8/wave32_reject", err, needle);
        }
    }

    /* 4. gfx950 default still builds the hero atom. */
    spec = fp8_spec();
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    ll = fp8_lower(&spec, "gfx950", err, sizeof(err));
    if(ll == NULL)
    {
        fail("fp8/gfx950_default_builds", err);
    }
    else
    {
        expect_contains("fp8/gfx950_default_builds", ll, "mfma.scale.f32.16x16x128.f8f6f4");
        expect_contains("fp8/gfx950_default_builds", ll, "atomicrmw");
        free(ll);
    }

    /* 5. BStage_smem is allocated unconditionally but only referenced under
     * use_dtla; the packer dead-strips it otherwise, so the accounting has to
     * track that or it over-counts 32 KiB. */
    spec = fp8_spec();
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    check_fp8_accounting_matches_pool("fp8/pool/default", &spec);

    spec = fp8_spec();
    spec.use_dtla = false;
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    check_fp8_accounting_matches_pool("fp8/pool/use_dtla=false", &spec);

    spec = fp8_spec();
    spec.gate_up_k = 32;
    spec.down_k = 32;
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    check_fp8_accounting_matches_pool("fp8/pool/k32", &spec);

    spec = fp8_spec();
    spec.tile_m = 32;
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    check_fp8_accounting_matches_pool("fp8/pool/tile_m=32", &spec);

    spec = fp8_spec();
    spec.tile_n_inter = 512;
    spec.warp_n = 4;
    rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
    check_fp8_accounting_matches_pool("fp8/pool/tile_n_inter=512", &spec);

    /* 6. Widening the inter slice scales Hidden + the f32 amax scratch + the
     * DTLA landing zone together, past CDNA4's 160 KiB. */
    {
        rocke_mega_lds_alloc_t allocs[ROCKE_MOE_FP8_LDS_MAX_ALLOCS];
        char needle[64];
        size_t n;
        long total;

        spec = fp8_spec();
        spec.tile_m = 32;
        spec.tile_n_inter = 2048;
        spec.warp_n = 16;
        rocke_fused_mega_kernel_spec_fp8_post_init(&spec);
        n = rocke_moe_fp8_lds_allocs(&spec, NULL, allocs);
        total = rocke_mega_lds_pool_bytes(allocs, n);

        ll = fp8_lower(&spec, "gfx950", err, sizeof(err));
        if(ll != NULL)
        {
            free(ll);
            fail("fp8/gfx950_over_budget", "expected rejection, spec was accepted");
        }
        else
        {
            snprintf(needle, sizeof(needle), "%ld", total);
            expect_contains("fp8/gfx950_over_budget", err, needle);
            expect_contains("fp8/gfx950_over_budget", err, "163840");
            expect_contains("fp8/gfx950_over_budget", err, "Hidden_smem");
        }
    }
}

int main(void)
{
    test_mega_lds_budget();
    test_fp8_guards();

    if(g_failures != 0)
    {
        fprintf(stderr, "%d MOE FUSED MEGA SPEC-GUARD CHECK(S) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL MOE FUSED MEGA SPEC GUARDS HELD\n");
    return 0;
}
