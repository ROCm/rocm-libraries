// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for the C++ validator and geometry of the gfx950
 * rows-in-registers MFMA direct forward and its weight-format helper
 * (rocke/instance_conv_direct_fwd_rr_mfma.h).
 *
 * The byte-identity gate compares only the kernels both engines build, and
 * the cross-engine describe test (library/tests/test_direct_fwd_rr_tap1d.py)
 * needs the rocke_library_engine extension, which CI does not build. So the
 * refusals of rocke_fwd_rr_mfma_is_valid_spec are pinned here to the texts of
 * Python's fwd_rr_mfma_reason (the arch gates, the problem, the filter and
 * padding, every knob's consistency rule, the i32 limits), together with the
 * grid, workspace and kernel names of both kernels.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_rr_mfma.h"

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

/* A 3x3 "same" bf16 forward, N=2, 34x45, C=128, K=256. */
static rocke_fwd_rr_mfma_spec_t base_spec(void)
{
    rocke_fwd_rr_mfma_spec_t s = rocke_fwd_rr_mfma_spec_default();
    s.problem.N = 2;
    s.problem.H = 34;
    s.problem.W = 45;
    s.problem.groups = 1;
    s.problem.cpg = 128;
    s.problem.kpg = 256;
    s.problem.KH = 3;
    s.problem.KW = 3;
    s.problem.PAD = 1;
    s.problem.pad_w = -1;
    s.problem.stride = 1;
    s.problem.dil_h = 1;
    s.problem.dil_w = 1;
    s.problem.dtype = "bf16";
    return s;
}

/* The same layer with a 3x1 filter (no column padding). */
static rocke_fwd_rr_mfma_spec_t three_by_one(void)
{
    rocke_fwd_rr_mfma_spec_t s = base_spec();
    s.problem.KW = 1;
    s.problem.pad_w = 0;
    return s;
}

static void expect_refused(const char* what,
                           const rocke_fwd_rr_mfma_spec_t* s,
                           const char* arch,
                           const char* expected)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    CHECK(!rocke_fwd_rr_mfma_is_valid_spec(s, arch, reason, sizeof reason), "%s accepted", what);
    CHECK(strcmp(reason, expected) == 0, "%s reason: got '%s', want '%s'", what, reason, expected);
}

static void test_accepted_geometry(void)
{
    char reason[ERR_CAP];
    char name[256];
    int grid[3];
    rocke_fwd_rr_mfma_spec_t s = base_spec();
    CHECK(rocke_fwd_rr_mfma_is_valid_spec(&s, "gfx950", reason, sizeof reason),
          "base refused: %s",
          reason);
    /* 2 images x 2 pixel tiles x 3 row tiles x 2 K blocks */
    CHECK(rocke_fwd_rr_mfma_tiles(&s) == 24, "tiles %lld", rocke_fwd_rr_mfma_tiles(&s));
    rocke_fwd_rr_mfma_grid(&s, grid);
    CHECK(grid[0] == 24 && grid[1] == 1 && grid[2] == 1, "grid %d", grid[0]);
    CHECK(rocke_fwd_rr_mfma_workspace_bytes(&s) == 589824,
          "workspace %lld",
          rocke_fwd_rr_mfma_workspace_bytes(&s));
    CHECK(rocke_fwd_rr_mfma_kernel_name(&s, name, sizeof name) == ROCKE_OK
              && strcmp(name, "fwd_rr_mfma_f3x3_p1_bf16") == 0,
          "name %s",
          name);
    s = three_by_one();
    s.kw16 = 2;
    s.odd_c = true;
    s.problem.cpg = 192;
    CHECK(rocke_fwd_rr_mfma_is_valid_spec(&s, "gfx950", reason, sizeof reason),
          "3x1 odd_c refused: %s",
          reason);
    rocke_fwd_rr_mfma_kernel_name(&s, name, sizeof name);
    CHECK(strcmp(name, "fwd_rr_mfma_f3x1_p1x0_k64_oddc_bf16") == 0, "name %s", name);

    rocke_fwd_rr_wfmt_spec_t w = rocke_fwd_rr_wfmt_spec_default();
    w.problem = base_spec().problem;
    w.problem.cpg = 72;
    w.problem.kpg = 192;
    w.problem.KW = 1;
    w.kw16 = 3;
    w.fw = 1;
    w.nwk = 4;
    w.dg = true;
    rocke_fwd_rr_wfmt_kernel_name(&w, name, sizeof name);
    CHECK(strcmp(name, "fwd_rr_wfmt_k96_w1_wk4_dg_bf16") == 0, "wfmt name %s", name);
    rocke_fwd_rr_wfmt_grid(&w, grid);
    CHECK(grid[0] == 36 && grid[1] == 1 && grid[2] == 1, "wfmt grid %d", grid[0]);
}

static void test_arch_refusals(void)
{
    rocke_fwd_rr_mfma_spec_t s = base_spec();
    expect_refused(
        "gfx942", &s, "gfx942", "fwd_rr_mfma needs the 16-byte LDS DMA (absent on gfx942)");
    expect_refused("gfx1250", &s, "gfx1250", "fwd_rr_mfma is a wave64 MFMA kernel");
    s.wave_size = 32;
    expect_refused("wave_size 32", &s, "gfx950", "fwd_rr_mfma is a wave64 MFMA kernel");
}

static void test_problem_refusals(void)
{
    struct
    {
        const char* what;
        void (*edit)(rocke_direct_conv_problem_t*);
        const char* expected;
    } cases[] = {
        {"N 0", [](rocke_direct_conv_problem_t* p) { p->N = 0; }, "N must be >= 1 (got 0)"},
        {"fp32", [](rocke_direct_conv_problem_t* p) { p->dtype = "fp32"; }, "fp16/bf16 only"},
        {"groups 2",
         [](rocke_direct_conv_problem_t* p) { p->groups = 2; },
         "groups must be 1 (grp takes grouped rows)"},
        {"stride 2",
         [](rocke_direct_conv_problem_t* p) { p->stride = 2; },
         "3x3 or 3x1, stride 1, dilation 1 only"},
        {"5x5",
         [](rocke_direct_conv_problem_t* p) { p->KW = 5; },
         "3x3 or 3x1, stride 1, dilation 1 only"},
        {"PAD 3",
         [](rocke_direct_conv_problem_t* p) { p->PAD = 3; },
         "padding must be 0..2 (rows) and 0..KW-1 (columns) (got 3x3)"},
        {"pad_w -2",
         [](rocke_direct_conv_problem_t* p) { p->pad_w = -2; },
         "padding must be 0..2 (rows) and 0..KW-1 (columns) (got 1x1)"},
        {"C 100",
         [](rocke_direct_conv_problem_t* p) { p->cpg = 100; },
         "C 100 must be a multiple of 64 (c_tail takes the rest)"},
        {"C 192",
         [](rocke_direct_conv_problem_t* p) { p->cpg = 192; },
         "an odd number of 64-channel chunks needs odd_c"},
        {"K 96",
         [](rocke_direct_conv_problem_t* p) { p->kpg = 96; },
         "K 96 must be a multiple of 128"},
        {"input image",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 1024;
             p->W = 2048;
             p->cpg = 1024;
         },
         "input image exceeds 2147483647 B, the i32 buffer range"},
    };
    for(const auto& c : cases)
    {
        rocke_fwd_rr_mfma_spec_t s = base_spec();
        c.edit(&s.problem);
        expect_refused(c.what, &s, "gfx950", c.expected);
    }
}

static void test_knob_refusals(void)
{
    rocke_fwd_rr_mfma_spec_t s = base_spec();
    s.kw16 = 1;
    expect_refused("kw16 1", &s, "gfx950", "kw16 must be one of (4, 3, 2)");
    s = base_spec();
    s.odd_c = true;
    expect_refused("odd_c with 2 chunks",
                   &s,
                   "gfx950",
                   "odd_c needs an odd number (>= 3) of 64-channel chunks");
    s = base_spec();
    s.c_tail = true;
    expect_refused(
        "c_tail with C 128", &s, "gfx950", "c_tail needs C a multiple of 8 but not of 64");
    s = base_spec();
    s.one_c = true;
    expect_refused("one_c with C 128", &s, "gfx950", "one_c needs C <= 64, a multiple of 8");
    s = base_spec();
    s.problem.cpg = 64;
    s.one_c = true;
    s.odd_c = true;
    expect_refused("one_c odd_c", &s, "gfx950", "one_c: c_tail exactly when C < 64, no odd_c");
    s = base_spec();
    s.grp = true;
    expect_refused("grp with groups 1",
                   &s,
                   "gfx950",
                   "grp: groups >= 2, a whole number of K blocks per group, no unfold");
    s = base_spec();
    s.unfold = true;
    expect_refused("unfold 3x3", &s, "gfx950", "unfold needs a width-1 layer with a 3x1 filter");
    s = base_spec();
    s.chunk = true;
    expect_refused(
        "chunk without k4w", &s, "gfx950", "chunk: k4w 3x3 tiles only (no unfold / grp)");
    s = three_by_one();
    s.c64 = true;
    s.kw16 = 2;
    expect_refused("c64 3x1", &s, "gfx950", "c64 needs at least 2 filter columns");
    s = three_by_one();
    s.problem.W = 8;
    expect_refused("3x1 width 8",
                   &s,
                   "gfx950",
                   "output width 8 wastes most of the 32-column tile (unfold runs width 1)");
}

int main(void)
{
    test_accepted_geometry();
    test_arch_refusals();
    test_problem_refusals();
    test_knob_refusals();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_fwd_rr_mfma_host: OK\n");
    return 0;
}
