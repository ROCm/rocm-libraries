// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for the C++ validator of the gfx950 flattened-
 * pixel implicit-GEMM MFMA direct forward (rocke/instance_conv_direct_fwd_flex_mfma.h).
 *
 * The byte-identity gate compares only the kernels both engines build, and
 * the cross-engine describe test (library/tests/test_direct_fwd_flex_mfma.py)
 * needs the rocke_library_engine extension, which CI does not build. So the
 * refusals of rocke_fwd_flex_mfma_is_valid_spec are pinned here to the texts
 * of Python's fwd_flex_mfma_reason: the wave64 gate and the 16-byte LDS DMA
 * capability gate, the dtype, the extents, groups, stride-and-dilation, the
 * channel granules, the bk and split_c knobs, the tap-count limit, the i32
 * limits of the buffer descriptors, the workspace slab and the workspace size.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_direct_fwd_flex_mfma.h"

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

/* A stride-2 3x3 bf16 forward with C=192 and K=136 (the Python test's
 * _problem()). C=192 is a multiple of 64; K=136 is a multiple of 8. */
static rocke_fwd_flex_mfma_spec_t base_spec(void)
{
    rocke_fwd_flex_mfma_spec_t s = rocke_fwd_flex_mfma_spec_default();
    s.problem.N = 2;
    s.problem.H = 37;
    s.problem.W = 41;
    s.problem.groups = 1;
    s.problem.cpg = 192;
    s.problem.kpg = 136;
    s.problem.KH = 3;
    s.problem.KW = 3;
    s.problem.PAD = 1;
    s.problem.stride = 2;
    s.problem.dil_h = 1;
    s.problem.dil_w = 1;
    s.problem.dtype = "bf16";
    return s;
}

static void expect_ok(const char* what, const rocke_fwd_flex_mfma_spec_t* s)
{
    char reason[ERR_CAP];
    CHECK(rocke_fwd_flex_mfma_is_valid_spec(s, "gfx950", reason, sizeof reason),
          "%s refused: %s",
          what,
          reason);
}

static void expect_refused(const char* what,
                           const rocke_fwd_flex_mfma_spec_t* s,
                           const char* arch,
                           const char* expected)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    CHECK(!rocke_fwd_flex_mfma_is_valid_spec(s, arch, reason, sizeof reason), "%s accepted", what);
    CHECK(strcmp(reason, expected) == 0, "%s reason: got '%s', want '%s'", what, reason, expected);
}

static void test_accepted(void)
{
    rocke_fwd_flex_mfma_spec_t s = base_spec();
    expect_ok("base (bk=256 split_c=1)", &s);
    s.bk = 128;
    expect_ok("bk 128", &s);
    s = base_spec();
    s.split_c = 2;
    expect_ok("split_c 2 (C=192, C/64=3)", &s);
    s = base_spec();
    s.split_c = 3;
    expect_ok("split_c 3 (C=192, C/64=3)", &s);
    // per-axis padding: flex allows PAD_W != PAD (unlike pp)
    s = base_spec();
    s.problem.pad_w = 0;
    expect_ok("pad_w 0 with PAD 1 (per-axis ok)", &s);
    // 1x1 stride-2 layer
    s = base_spec();
    s.problem.KH = 1;
    s.problem.KW = 1;
    s.problem.PAD = 0;
    expect_ok("1x1 s2", &s);
    // 1x7 with column padding (stride 1, dilation 1)
    s = base_spec();
    s.problem.KH = 1;
    s.problem.KW = 7;
    s.problem.PAD = 0;
    s.problem.pad_w = 3;
    s.problem.stride = 1;
    expect_ok("1x7 pad_w=3 s1", &s);
}

static void test_arch_and_dtype_refusals(void)
{
    const char* wave64 = "fwd_flex_mfma is a wave64 MFMA kernel";
    rocke_fwd_flex_mfma_spec_t s = base_spec();
    expect_refused("gfx1250", &s, "gfx1250", wave64);
    expect_refused(
        "gfx942", &s, "gfx942", "fwd_flex_mfma needs the 16-byte LDS DMA (absent on gfx942)");
    expect_refused(
        "gfx90a", &s, "gfx90a", "fwd_flex_mfma needs the 16-byte LDS DMA (absent on gfx90a)");
    s.wave_size = 32;
    expect_refused("wave_size 32", &s, "gfx950", wave64);
    s = base_spec();
    s.problem.dtype = "fp32";
    expect_refused("fp32", &s, "gfx950", "fp16/bf16 only");
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
        {"cpg 0", [](rocke_direct_conv_problem_t* p) { p->cpg = 0; }, "cpg must be >= 1 (got 0)"},
        {"kpg 0", [](rocke_direct_conv_problem_t* p) { p->kpg = 0; }, "kpg must be >= 1 (got 0)"},
        {"groups 2", [](rocke_direct_conv_problem_t* p) { p->groups = 2; }, "groups must be 1"},
        {"stride 0",
         [](rocke_direct_conv_problem_t* p) { p->stride = 0; },
         "stride and dilation must be >= 1 (got s0 d1x1)"},
        {"dil_h 0",
         [](rocke_direct_conv_problem_t* p) { p->dil_h = 0; },
         "stride and dilation must be >= 1 (got s2 d0x1)"},
        {"dil_w 0",
         [](rocke_direct_conv_problem_t* p) { p->dil_w = 0; },
         "stride and dilation must be >= 1 (got s2 d1x0)"},
        {"cpg 12",
         [](rocke_direct_conv_problem_t* p) { p->cpg = 12; },
         "C 12 must be a multiple of 64"},
        {"kpg 10",
         [](rocke_direct_conv_problem_t* p) { p->kpg = 10; },
         "K 10 must be a multiple of 8 (16-byte output writes)"},
        {"pad_w -3",
         [](rocke_direct_conv_problem_t* p) { p->pad_w = -3; },
         "pad_w must be -1 (PAD) or a column padding of at least 0 (got -3)"},
        {"input image",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 1024;
             p->W = 2048;
             p->cpg = 768;
             p->kpg = 768;
         },
         "input exceeds 2147483647 B, the i32 buffer range"},
        {"output image",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 16;
             p->W = 16;
             p->cpg = 64;
             p->kpg = 1 << 27;
         },
         "output exceeds 2147483647 B, the i32 buffer range"},
        {"weights",
         [](rocke_direct_conv_problem_t* p) {
             p->H = 1;
             p->W = 1;
             p->cpg = 1 << 14;
             p->kpg = 1 << 14;
         },
         "weights exceeds 2147483647 B, the i32 buffer range"},
        {"tap count",
         [](rocke_direct_conv_problem_t* p) {
             p->KH = 6;
             p->KW = 6;
             p->PAD = 0;
             p->stride = 1;
         },
         "more than 32 taps (one valid-tap bit per tap)"},
    };
    for(const auto& c : cases)
    {
        rocke_fwd_flex_mfma_spec_t s = base_spec();
        c.edit(&s.problem);
        expect_refused(c.what, &s, "gfx950", c.expected);
    }
}

static void test_knob_refusals(void)
{
    rocke_fwd_flex_mfma_spec_t s = base_spec();
    s.bk = 64;
    expect_refused("bk 64", &s, "gfx950", "bk must be 128 or 256");
    s = base_spec();
    s.bk = 192;
    expect_refused("bk 192", &s, "gfx950", "bk must be 128 or 256");
    // split_c out of range: C=192, C/64=3, so split_c=4 is too large
    s = base_spec();
    s.split_c = 4;
    expect_refused("split_c 4 with C=192", &s, "gfx950", "split_c 4 must be in [1, C / 64]");
    // split_c=0 is also invalid
    s = base_spec();
    s.split_c = 0;
    expect_refused("split_c 0", &s, "gfx950", "split_c 0 must be in [1, C / 64]");
    // four 1 GiB slabs: each within the buffer range, 4 GiB in all
    s = base_spec();
    s.problem.N = 32;
    s.problem.H = s.problem.W = 255;
    s.problem.cpg = 256;
    s.problem.kpg = 512;
    s.problem.stride = 2;
    s.split_c = 4;
    expect_refused("workspace 4 GiB",
                   &s,
                   "gfx950",
                   "workspace exceeds 2147483647 B, the i32 byte-count kernarg");
}

int main(void)
{
    test_accepted();
    test_arch_and_dtype_refusals();
    test_problem_refusals();
    test_knob_refusals();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_fwd_flex_mfma_host: OK\n");
    return 0;
}
