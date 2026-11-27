// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for the C++ validator and geometry of the gfx950
 * 1-D tap-reuse MFMA direct forward (rocke/instance_conv_direct_fwd_tap1d_mfma.h).
 *
 * The cross-engine describe test (library/tests/test_direct_fwd_rr_tap1d.py)
 * needs the rocke_library_engine extension, which CI does not build. So the
 * refusals of rocke_fwd_tap1d_mfma_is_valid_spec are pinned here to the texts
 * of Python's fwd_tap1d_mfma_reason, together with the derived geometry (the
 * polyphase column count, the window and DMA counts, LDS, grid, name).
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_tap1d_mfma.h"

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

/* A 1x7 fp16 forward with dilation 3 and padding 9, N=2, 3x300, C=128,
 * K=256; bk 128 over 4 waves. */
static rocke_fwd_tap1d_mfma_spec_t base_spec(void)
{
    rocke_fwd_tap1d_mfma_spec_t s = rocke_fwd_tap1d_mfma_spec_default();
    s.problem.N = 2;
    s.problem.H = 3;
    s.problem.W = 300;
    s.problem.groups = 1;
    s.problem.cpg = 128;
    s.problem.kpg = 256;
    s.problem.KH = 1;
    s.problem.KW = 7;
    s.problem.PAD = 0;
    s.problem.pad_w = 9;
    s.problem.stride = 1;
    s.problem.dil_h = 1;
    s.problem.dil_w = 3;
    s.problem.dtype = "fp16";
    s.bk = 128;
    s.nwv = 4;
    return s;
}

static void expect_refused(const char* what,
                           const rocke_fwd_tap1d_mfma_spec_t* s,
                           const char* arch,
                           const char* expected)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    CHECK(!rocke_fwd_tap1d_mfma_is_valid_spec(s, arch, reason, sizeof reason), "%s accepted", what);
    CHECK(strcmp(reason, expected) == 0, "%s reason: got '%s', want '%s'", what, reason, expected);
}

static void test_accepted_geometry(void)
{
    char reason[ERR_CAP];
    char name[256];
    int grid[3];
    rocke_fwd_tap1d_mfma_spec_t s = base_spec();
    CHECK(rocke_fwd_tap1d_mfma_is_valid_spec(&s, "gfx950", reason, sizeof reason),
          "base refused: %s",
          reason);
    /* 1 wave along X: 16 columns, 15 a multiple of the dilation */
    CHECK(rocke_fwd_tap1d_mfma_ncol(&s) == 15, "ncol %d", rocke_fwd_tap1d_mfma_ncol(&s));
    CHECK(rocke_fwd_tap1d_mfma_bx(&s) == 135, "bx %d", rocke_fwd_tap1d_mfma_bx(&s));
    CHECK(rocke_fwd_tap1d_mfma_win(&s) == 153, "win %d", rocke_fwd_tap1d_mfma_win(&s));
    CHECK(rocke_fwd_tap1d_mfma_win_dmas(&s) == 20 && rocke_fwd_tap1d_mfma_wgt_dmas(&s) == 16,
          "dmas %d %d",
          rocke_fwd_tap1d_mfma_win_dmas(&s),
          rocke_fwd_tap1d_mfma_wgt_dmas(&s));
    CHECK(rocke_fwd_tap1d_mfma_lds_bytes(&s) == 73728,
          "lds %lld",
          rocke_fwd_tap1d_mfma_lds_bytes(&s));
    rocke_fwd_tap1d_mfma_grid(&s, grid);
    CHECK(grid[0] == 36 && grid[1] == 1 && grid[2] == 1, "grid %d", grid[0]);
    CHECK(rocke_fwd_tap1d_mfma_kernel_name(&s, name, sizeof name) == ROCKE_OK
              && strcmp(name, "fwd_tap1d_mfma_f1x7_p9_d3_k128_w4_g9_c64") == 0,
          "name %s",
          name);
}

static void test_refusals(void)
{
    rocke_fwd_tap1d_mfma_spec_t s = base_spec();
    expect_refused(
        "gfx942", &s, "gfx942", "fwd_tap1d_mfma needs the 16-byte LDS DMA (absent on gfx942)");
    expect_refused("gfx1250", &s, "gfx1250", "fwd_tap1d_mfma is a wave64 MFMA kernel");
    struct
    {
        const char* what;
        void (*edit)(rocke_fwd_tap1d_mfma_spec_t*);
        const char* expected;
    } cases[] = {
        {"groups 2",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.groups = 2; },
         "groups must be 1"},
        {"KH 3",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.KH = 3; },
         "1 x S filters, no height padding, stride 1 only"},
        {"stride 2",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.stride = 2; },
         "1 x S filters, no height padding, stride 1 only"},
        {"pad_w -2",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.pad_w = -2; },
         "pad_w must be -1 (PAD) or a column padding of at least 0 (got -2)"},
        {"pad_w 19",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.pad_w = 19; },
         "width padding must be in [0, dilation * (S - 1)]"},
        {"fp32",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.dtype = "fp32"; },
         "fp16/bf16 only"},
        {"K 100",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.kpg = 100; },
         "K 100 must be a multiple of bk 128"},
        {"C 100",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.cpg = 100; },
         "ck 64 must be 32 or 64 and divide C 100"},
        {"C 64 without one",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->problem.cpg = 64; },
         "one (single chunk) must be set exactly when C == ck (64 vs 64)"},
        {"bk 96",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->bk = 96; },
         "bk must be 32/64/128 and nwv 4/8 with nwv % (bk / 32) == 0"},
        {"nwv 6",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->nwv = 6; },
         "bk must be 32/64/128 and nwv 4/8 with nwv % (bk / 32) == 0"},
        {"seg 0",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->seg = 0; },
         "seg must be >= 1 and the dilation at most 16 * waves along X"},
        {"dilation 20",
         [](rocke_fwd_tap1d_mfma_spec_t* s) {
             s->problem.dil_w = 20;
             s->problem.pad_w = 0;
         },
         "seg must be >= 1 and the dilation at most 16 * waves along X"},
        {"one with C 128",
         [](rocke_fwd_tap1d_mfma_spec_t* s) { s->one = true; },
         "one (single chunk) must be set exactly when C == ck (128 vs 64)"},
        {"LDS",
         [](rocke_fwd_tap1d_mfma_spec_t* s) {
             s->seg = 40;
             s->nwv = 8;
             s->problem.W = 2000;
         },
         "LDS 346112 B > 163840"},
    };
    for(const auto& c : cases)
    {
        s = base_spec();
        c.edit(&s);
        expect_refused(c.what, &s, "gfx950", c.expected);
    }
}

int main(void)
{
    test_accepted_geometry();
    test_refusals();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_fwd_tap1d_mfma_host: OK\n");
    return 0;
}
