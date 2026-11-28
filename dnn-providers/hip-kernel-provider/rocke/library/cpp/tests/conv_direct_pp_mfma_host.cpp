// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for what the gfx950 LDS-DMA MFMA direct conv
 * validators share: the capability gate
 * rocke_direct_lds_dma_mfma_caps_reason (rocke/instance_conv_direct_grouped.h),
 * the C++ twin of Python's direct_lds_dma_mfma_caps_reason, which fwd pp, dgrad
 * pp, fwd flex and the tiled and gather wgrad all call. The two wgrad
 * validators have no host test of their own, so their gate calls are pinned
 * here on real arches (gfx942: no transposing LDS read; gfx1100: wave32).
 *
 * Each stand-in target is gfx950 with exactly one capability removed (wave64,
 * the transposing LDS read, the 16-byte LDS DMA, mfma_f32_16x16x32), so each
 * check is the deciding refusal in its own case. The Python twin is
 * library/tests/test_direct_lds_dma_mfma_caps.py.
 *
 * It also pins the rest of pp_is_valid_spec, which fwd pp and dgrad pp both
 * call (rocke_pp, rocke/instance_conv_direct_pp_mfma_internal.hpp): dtype,
 * extents, groups, stride, dilation, the square odd filter, "same" padding,
 * the channel granules, the LDS budget and the i32 limits. The per-family host
 * tests keep only their own gates (family name, block_k set, wiring).
 */
#include <stdio.h>
#include <string.h>

#include <initializer_list>

#include "rocke/instance_conv_direct_pp_mfma_internal.hpp"
#include "rocke/instance_conv_direct_wgrad_gather.h"
#include "rocke/instance_conv_direct_wgrad_tiled.h"

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

/* The forward-pass problem both pp families validate: "same"-padded 3x3
 * stride-1 bf16 with C = 96 and K = 136. */
static rocke_direct_conv_problem_t base_problem(void)
{
    rocke_direct_conv_problem_t p;
    memset(&p, 0, sizeof p);
    p.N = 2;
    p.H = 37;
    p.W = 41;
    p.groups = 1;
    p.cpg = 96;
    p.kpg = 136;
    p.KH = 3;
    p.KW = 3;
    p.PAD = 1;
    p.pad_w = -1;
    p.stride = 1;
    p.dil_h = 1;
    p.dil_w = 1;
    p.dtype = "bf16";
    return p;
}

static const int kFwdBlockKs[3] = {64, 128, 256};
static const int kDgradBlockKs[2] = {128, 256};
static const rocke_pp::PpFamily kFwd = {"fwd_pp_mfma", kFwdBlockKs, 3, "bk fwd", false};
static const rocke_pp::PpFamily kDgrad = {"dgrad_pp_mfma", kDgradBlockKs, 2, "bk dgrad", true};

static void expect_pp_ok(const char* what,
                         const rocke_pp::PpFamily& fam,
                         const rocke_direct_conv_problem_t& p,
                         int block_k,
                         const char* arch)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    rocke_pp::PpSpec s = {&p, "k", block_k, 64};
    CHECK(rocke_pp::pp_is_valid_spec(s, fam, arch, reason, sizeof reason),
          "%s refused: %s",
          what,
          reason);
}

static void expect_pp_refused(const char* what,
                              const rocke_pp::PpFamily& fam,
                              const rocke_direct_conv_problem_t& p,
                              int block_k,
                              const char* arch,
                              const char* expected)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    rocke_pp::PpSpec s = {&p, "k", block_k, 64};
    CHECK(!rocke_pp::pp_is_valid_spec(s, fam, arch, reason, sizeof reason), "%s accepted", what);
    CHECK(strcmp(reason, expected) == 0, "%s reason: got '%s', want '%s'", what, reason, expected);
}

static void check_problem_refusals(const rocke_pp::PpFamily& fam)
{
    const char* square = "square odd filters >= 3 only";
    struct
    {
        const char* what;
        void (*edit)(rocke_direct_conv_problem_t*);
        const char* expected;
    } cases[] = {
        {"N 0", [](rocke_direct_conv_problem_t* p) { p->N = 0; }, "N must be >= 1 (got 0)"},
        {"cpg 0", [](rocke_direct_conv_problem_t* p) { p->cpg = 0; }, "cpg must be >= 1 (got 0)"},
        {"kpg 0", [](rocke_direct_conv_problem_t* p) { p->kpg = 0; }, "kpg must be >= 1 (got 0)"},
        {"groups 2", [](rocke_direct_conv_problem_t* p) { p->groups = 2; }, "groups must be 1"},
        {"stride 2", [](rocke_direct_conv_problem_t* p) { p->stride = 2; }, "stride must be 1"},
        {"dil_h 2", [](rocke_direct_conv_problem_t* p) { p->dil_h = 2; }, "dilation must be 1"},
        {"dil_w 2", [](rocke_direct_conv_problem_t* p) { p->dil_w = 2; }, "dilation must be 1"},
        {"4x4",
         [](rocke_direct_conv_problem_t* p) {
             p->KH = p->KW = 4;
             p->PAD = 1;
         },
         square},
        {"3x5", [](rocke_direct_conv_problem_t* p) { p->KW = 5; }, square},
        {"1x1",
         [](rocke_direct_conv_problem_t* p) {
             p->KH = p->KW = 1;
             p->PAD = 0;
         },
         square},
        {"PAD 0",
         [](rocke_direct_conv_problem_t* p) { p->PAD = 0; },
         "'same' padding only: PAD must be 1 (got 0)"},
        {"pad_w 0",
         [](rocke_direct_conv_problem_t* p) { p->pad_w = 0; },
         "one padding for both axes (got PAD=1x0)"},
        {"pad_w -3",
         [](rocke_direct_conv_problem_t* p) { p->pad_w = -3; },
         "pad_w must be -1 (PAD) or a column padding of at least 0 (got -3)"},
        {"cpg 12",
         [](rocke_direct_conv_problem_t* p) { p->cpg = 12; },
         "C 12 must be a multiple of 8 (16-byte granules)"},
        {"kpg 10",
         [](rocke_direct_conv_problem_t* p) { p->kpg = 10; },
         "K 10 must be a multiple of 4 (8-byte output writes)"},
        {"7x7",
         [](rocke_direct_conv_problem_t* p) {
             p->KH = p->KW = 7;
             p->PAD = 3;
         },
         "LDS 167936 B > 163840"},
        {"input image",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 1024;
             p->W = 2048;
             p->cpg = 768;
             p->kpg = 768;
         },
         "input image exceeds 2147483647 B, the i32 buffer range"},
        {"output image",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 16;
             p->W = 16;
             p->cpg = 8;
             p->kpg = 1 << 27;
         },
         "output image exceeds 2147483647 B, the i32 buffer range"},
        {"filter",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 1;
             p->W = 1;
             p->cpg = 1 << 14;
             p->kpg = 1 << 14;
         },
         "filter exceeds 2147483647 B, the i32 buffer range"},
        {"input",
         [](rocke_direct_conv_problem_t* p) {
             p->N = 32;
             p->H = 256;
             p->W = 256;
             p->cpg = 1024;
             p->kpg = 1024;
         },
         "input exceeds 2147483647 B, the i32 byte-count kernarg"},
        {"output",
         [](rocke_direct_conv_problem_t* p) {
             p->N = 32;
             p->H = 256;
             p->W = 256;
             p->cpg = 8;
             p->kpg = 1024;
         },
         "output exceeds 2147483647 B, the i32 byte-count kernarg"},
    };
    for(const auto& c : cases)
    {
        rocke_direct_conv_problem_t p = base_problem();
        c.edit(&p);
        expect_pp_refused(c.what, fam, p, 128, "gfx950", c.expected);
    }
}

static void test_shared_refusals(void)
{
    /* Both families' problem rules and budgets are pp_is_valid_spec's: one
     * table, run for each. */
    for(const rocke_pp::PpFamily* fam : {&kFwd, &kDgrad})
    {
        rocke_direct_conv_problem_t p = base_problem();
        expect_pp_ok(fam->family, *fam, p, 128, "gfx950");
        expect_pp_ok(fam->family, *fam, p, 256, "gfx950");
        p.pad_w = 1;
        expect_pp_ok("explicit same pad_w", *fam, p, 128, "gfx950");
        p = base_problem();
        p.KH = p.KW = 5;
        p.PAD = 2;
        expect_pp_ok("5x5", *fam, p, 128, "gfx950");
        expect_pp_refused("5x5 block_k 256", *fam, p, 256, "gfx950", "LDS 167936 B > 163840");
        p = base_problem();
        p.dtype = "fp32";
        expect_pp_refused("fp32", *fam, p, 128, "gfx950", "fp16/bf16 only");
        check_problem_refusals(*fam);
    }
}

/* The wgrad validators call the gate with needs_ds_read_tr: gfx950 passes,
 * gfx942 (no ds_read_b64_tr_b16) and gfx1100 (wave32) are refused by it. The
 * problems are the ones test_direct_lds_dma_mfma_caps.py builds. */
template <typename Spec, typename Valid>
static void check_wgrad_gate(const char* family, const Spec& s, Valid is_valid)
{
    char why[ERR_CAP];
    char want[ERR_CAP];
    why[0] = '\0';
    CHECK(is_valid(&s, "gfx950", why, sizeof why), "%s refused on gfx950: %s", family, why);

    why[0] = '\0';
    snprintf(want, sizeof want, "%s needs ds_read_b64_tr_b16 (absent on gfx942)", family);
    CHECK(!is_valid(&s, "gfx942", why, sizeof why), "%s accepted on gfx942", family);
    CHECK(strcmp(why, want) == 0, "%s gfx942 reason: got '%s', want '%s'", family, why, want);

    why[0] = '\0';
    snprintf(want, sizeof want, "%s is a wave64 MFMA kernel", family);
    CHECK(!is_valid(&s, "gfx1100", why, sizeof why), "%s accepted on gfx1100", family);
    CHECK(strcmp(why, want) == 0, "%s gfx1100 reason: got '%s', want '%s'", family, why, want);
}

static void test_wgrad_callers(void)
{
    rocke_direct_wgrad_tiled_spec_t tiled = rocke_direct_wgrad_tiled_spec_default();
    tiled.problem = base_problem();
    tiled.problem.cpg = 64;
    tiled.problem.kpg = 64;
    check_wgrad_gate("wgrad_tiled", tiled, rocke_direct_wgrad_tiled_is_valid_spec);

    rocke_direct_wgrad_gather_spec_t gather = rocke_direct_wgrad_gather_spec_default();
    gather.problem = base_problem();
    gather.problem.H = 16;
    gather.problem.W = 32;
    gather.problem.cpg = 3;
    gather.problem.kpg = 24;
    gather.problem.stride = 2;
    check_wgrad_gate("wgrad_gather", gather, rocke_direct_wgrad_gather_is_valid_spec);
}

int main(void)
{
    test_each_capability_decides();
    test_shared_refusals();
    test_wgrad_callers();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_pp_mfma_host: OK\n");
    return 0;
}
