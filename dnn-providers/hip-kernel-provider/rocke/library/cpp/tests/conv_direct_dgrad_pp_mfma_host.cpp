// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for the C++ validator of the gfx950 ping-pong
 * direct dgrad (rocke/instance_conv_direct_dgrad_pp_mfma.h).
 *
 * The byte-identity gate compares only the kernels both engines build, and
 * the cross-engine describe test (library/tests/test_direct_dgrad_pp_mfma.py)
 * needs the rocke_library_engine extension, which CI does not build. So the
 * refusals of rocke_dgrad_pp_mfma_is_valid_spec are pinned here to the texts of
 * Python's dgrad_pp_mfma_reason: the wave64 gate and the ds_read_b64_tr_b16 capability gate, the dtype, the
 * extents, groups, stride, dilation on either axis, the square odd filter of
 * at least 3x3, the "same" padding on both axes, the channel granules, the
 * block_k set, the LDS budget and the i32 limits of the buffer descriptors
 * and the block id.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/instance_conv_direct_dgrad_pp_mfma.h"

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

/* The forward pass of a "same"-padded 3x3 stride-1 bf16 dgrad with C = 136
 * and K = 96 (dgrad_pp_mfma_problem of the Python test's _dgrad()). */
static rocke_dgrad_pp_mfma_spec_t base_spec(void)
{
    rocke_dgrad_pp_mfma_spec_t s = rocke_dgrad_pp_mfma_spec_default();
    s.problem.N = 2;
    s.problem.H = 37;
    s.problem.W = 41;
    s.problem.groups = 1;
    s.problem.cpg = 96;
    s.problem.kpg = 136;
    s.problem.KH = 3;
    s.problem.KW = 3;
    s.problem.PAD = 1;
    s.problem.dtype = "bf16";
    return s;
}

static void expect_ok(const char* what, const rocke_dgrad_pp_mfma_spec_t* s)
{
    char reason[ERR_CAP];
    CHECK(rocke_dgrad_pp_mfma_is_valid_spec(s, "gfx950", reason, sizeof reason),
          "%s refused: %s",
          what,
          reason);
}

static void expect_refused(const char* what,
                           const rocke_dgrad_pp_mfma_spec_t* s,
                           const char* arch,
                           const char* expected)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    CHECK(!rocke_dgrad_pp_mfma_is_valid_spec(s, arch, reason, sizeof reason), "%s accepted", what);
    CHECK(strcmp(reason, expected) == 0, "%s reason: got '%s', want '%s'", what, reason, expected);
}

static void test_accepted(void)
{
    rocke_dgrad_pp_mfma_spec_t s = base_spec();
    expect_ok("base", &s);
    s.block_k = 256;
    expect_ok("block_k 256", &s);
    s = base_spec();
    s.problem.pad_w = 1;
    expect_ok("explicit same pad_w", &s);
    s = base_spec();
    s.problem.KH = s.problem.KW = 5;
    s.problem.PAD = 2;
    expect_ok("5x5 block_k 128", &s);
}

static void test_arch_and_dtype_refusals(void)
{
    const char* wave64 = "dgrad_pp_mfma is a wave64 MFMA kernel";
    rocke_dgrad_pp_mfma_spec_t s = base_spec();
    expect_refused("gfx1250", &s, "gfx1250", wave64);
    expect_refused(
        "gfx942", &s, "gfx942", "dgrad_pp_mfma needs ds_read_b64_tr_b16 (absent on gfx942)");
    expect_refused(
        "gfx90a", &s, "gfx90a", "dgrad_pp_mfma needs ds_read_b64_tr_b16 (absent on gfx90a)");
    s.wave_size = 32;
    expect_refused("wave_size 32", &s, "gfx950", wave64);
    s = base_spec();
    s.problem.dtype = "fp32";
    expect_refused("fp32", &s, "gfx950", "fp16/bf16 only");
}

static void test_problem_refusals(void)
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
        {"block id",
         [](rocke_direct_conv_problem_t* p) {
             p->N = 1 << 25;
             p->H = 256;
             p->W = 256;
         },
         "17179869184 workgroups exceed the i32 block id"},
    };
    for(const auto& c : cases)
    {
        rocke_dgrad_pp_mfma_spec_t s = base_spec();
        c.edit(&s.problem);
        expect_refused(c.what, &s, "gfx950", c.expected);
    }
}

static void test_knob_refusals(void)
{
    rocke_dgrad_pp_mfma_spec_t s = base_spec();
    s.block_k = 64;
    expect_refused("block_k 64",
                   &s,
                   "gfx950",
                   "block_k must be 128 or 256 (256-byte weight rows for the swizzle)");
    s = base_spec();
    s.problem.KH = s.problem.KW = 5;
    s.problem.PAD = 2;
    s.block_k = 256;
    expect_refused("5x5 block_k 256", &s, "gfx950", "LDS 167936 B > 163840");
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
    printf("conv_direct_dgrad_pp_mfma_host: OK\n");
    return 0;
}
