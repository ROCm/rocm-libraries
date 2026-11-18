// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for what the gfx950 LDS-DMA MFMA direct conv
 * validators share: the capability gate
 * rocke_direct_lds_dma_mfma_caps_reason (rocke/instance_conv_direct_grouped.h),
 * the C++ twin of Python's direct_lds_dma_mfma_caps_reason, which fwd pp, dgrad
 * pp, fwd flex and the tiled wgrad all call.
 *
 * Each stand-in target is gfx950 with exactly one capability removed (wave64,
 * the transposing LDS read, the 16-byte LDS DMA, mfma_f32_16x16x32), so each
 * check is the deciding refusal in its own case. The Python twin is
 * library/tests/test_direct_lds_dma_mfma_caps.py.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_direct_grouped.h"

#define ERR_CAP 1024

static int g_failures = 0;

#define CHECK(cond, ...)                                         \
    do                                                           \
    {                                                            \
        if(!(cond))                                              \
        {                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
            ++g_failures;                                        \
        }                                                        \
    } while(0)

static void expect_caps(const char* what,
                        const rocke_arch_target_t* t,
                        bool needs_tr,
                        bool bf16,
                        const char* expected /* NULL: no refusal */)
{
    char why[ERR_CAP];
    why[0] = '\0';
    const bool refused = rocke_direct_lds_dma_mfma_caps_reason(
        t, "gfx950", "k", 64, needs_tr, bf16, why, sizeof why);
    if(expected == NULL)
    {
        CHECK(!refused, "%s refused: %s", what, why);
        return;
    }
    CHECK(refused, "%s accepted", what);
    CHECK(strcmp(why, expected) == 0, "%s reason: got '%s', want '%s'", what, why, expected);
}

static void test_each_capability_decides(void)
{
    const rocke_arch_target_t* gfx950 = rocke_arch_target_from_gfx("gfx950");
    const rocke_arch_target_t* gfx942 = rocke_arch_target_from_gfx("gfx942");
    CHECK(gfx950 != NULL && gfx942 != NULL, "known arches");
    if(gfx950 == NULL || gfx942 == NULL)
    {
        return;
    }
    CHECK(gfx950->wave_size == 64 && gfx950->memory.has_ds_read_tr
              && gfx950->memory.async_lds_max_dwords >= 4,
          "gfx950 stand-in base");

    for(int bf16 = 0; bf16 < 2; ++bf16)
    {
        const char* ab = bf16 ? "bf16" : "f16";
        char missing[ERR_CAP];
        snprintf(missing, sizeof missing, "missing mfma_f32_16x16x32_%s on gfx950", ab);

        for(int tr = 0; tr < 2; ++tr)
        {
            expect_caps("gfx950", gfx950, tr, bf16, NULL);

            rocke_arch_target_t t = *gfx950;
            t.wave_size = 32;
            expect_caps("wave32", &t, tr, bf16, "k is a wave64 MFMA kernel");

            t = *gfx950;
            t.memory.has_ds_read_tr = false;
            expect_caps("no ds_read_tr",
                        &t,
                        tr,
                        bf16,
                        tr ? "k needs ds_read_b64_tr_b16 (absent on gfx950)" : NULL);

            t = *gfx950;
            t.memory.async_lds_max_dwords = 2;
            expect_caps(
                "async 2 dwords", &t, tr, bf16, "k needs the 16-byte LDS DMA (absent on gfx950)");

            t = *gfx950;
            t.mma = gfx942->mma; /* the catalog without mfma_f32_16x16x32 for 16-bit */
            expect_caps("no 16x16x32 mfma", &t, tr, bf16, missing);
        }
    }

    /* The kernel's own wave size is gated as well as the target's. */
    char why[ERR_CAP];
    CHECK(rocke_direct_lds_dma_mfma_caps_reason(
              gfx950, "gfx950", "k", 32, false, true, why, sizeof why),
          "kernel wave_size 32 accepted");
    CHECK(strcmp(why, "k is a wave64 MFMA kernel") == 0, "kernel wave_size reason '%s'", why);
}

int main(void)
{
    test_each_capability_decides();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_pp_mfma_host: OK\n");
    return 0;
}
