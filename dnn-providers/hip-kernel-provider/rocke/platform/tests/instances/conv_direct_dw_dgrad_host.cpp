// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only regression test for the C++ depthwise dgrad entries
 * (rocke/instance_conv_direct_grouped_dw_dgrad.h and the dgrad mode of
 * rocke_direct_depthwise_tiled_spec_t).
 *
 * The byte-identity gate compares only the kernels both engines build, so it
 * does not see:
 *   - the failure path of rocke_direct_depthwise_dgrad_stream_lower_to_llvm
 *     when the kernel name does not fit its buffer: the builder is never
 *     initialised, and the cleanup must still be well defined;
 *   - pad_w values below -1, which are no column padding and must be refused
 *     rather than read as PAD;
 *   - the kernel name of a tiled dgrad spec whose column padding differs from
 *     its row padding, which must differ from the same spec at another column
 *     padding (and stay unchanged when both paddings are equal);
 *   - a tiled dgrad spec with more than one channel per lane, which the
 *     rotated-filter mode does not implement and must refuse;
 *   - a per-axis padding given to the WMMA convolutions that offset
 *     columns by PAD, which must refuse it; the direct and small-K WMMA
 *     forwards offset columns by PAD_W and must take it (pad_w below -1
 *     is refused everywhere);
 *   - the knob sets of the direct and tap-outer WMMA forwards whose kernel
 *     would read wrong data or wait too little, which their validators must
 *     refuse with Python's reason (and the neighbouring sets they must keep):
 *     a grouped KxK strided view, a polyphase view whose column padding is
 *     below S / 2, a zero ck (refused, not divided by), a padded polyphase
 *     view whose chunk is wider than one wave zeroes, and a tap-outer TDM
 *     split over more than 2 stages;
 *   - the shared host arithmetic the forward WMMA validators and grids read:
 *     rocke_floor_div / rocke_ceil_div (Python's // and ceil_div, for any
 *     sign) and DirectConvProblem.Ho / Wo, which a too-small input must give
 *     as Python does (floor, not truncation);
 *   - the forward WMMA validator tail's refusal of an unknown scheduler, which
 *     must carry normalize_scheduler_strategy's text, choices included.
 */
#include <initializer_list>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/abi.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/instance_conv_dgrad_patch_wmma.h"
#include "rocke/instance_conv_dgrad_smallc_wmma.h"
#include "rocke/instance_conv_direct_dgrad_wmma.h"
#include "rocke/instance_conv_direct_fwd_gstream_wmma.h"
#include "rocke/instance_conv_direct_fwd_tapouter_wmma.h"
#include "rocke/instance_conv_direct_fwd_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_dw_dgrad.h"
#include "rocke/instance_conv_fwd_small_wmma.h"
#include "rocke/lower_llvm.h"

/* Layout canary, as in test_mma_catalog_lookup.cpp: growing
 * rocke_direct_fwd_wmma_spec_t fails the build here until ROCKE_ABI_VERSION in
 * rocke/abi.h moves with it. 64-bit only. */
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(rocke_direct_fwd_wmma_spec_t) == 152 && ROCKE_ABI_VERSION >= 7,
              "rocke_direct_fwd_wmma_spec_t changed size: bump ROCKE_ABI_VERSION in "
              "rocke/abi.h, then update this pin");
#endif

#define ERR_CAP 1024
#define NAME_CAP 512

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

static rocke_direct_conv_problem_t dw_problem(int KH, int KW, int PAD, int pad_w)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = 16;
    p.W = 16;
    p.groups = 64;
    p.cpg = 1;
    p.kpg = 1;
    p.KH = KH;
    p.KW = KW;
    p.PAD = PAD;
    p.pad_w = pad_w;
    return p;
}

static rocke_direct_depthwise_tiled_spec_t tiled_dgrad(rocke_direct_conv_problem_t p)
{
    rocke_direct_depthwise_tiled_spec_t s = rocke_direct_depthwise_tiled_spec_default();
    s.problem = p;
    s.block_w = 8;
    s.block_h = 2;
    s.wave_size = 32;
    s.dgrad = true;
    return s;
}

static rocke_direct_depthwise_dgrad_stream_spec_t stream(rocke_direct_conv_problem_t p)
{
    rocke_direct_depthwise_dgrad_stream_spec_t s
        = rocke_direct_depthwise_dgrad_stream_spec_default();
    s.problem = p;
    s.wave_size = 32;
    return s;
}

/* A name past the 256-byte buffer fails before rocke_ir_builder_init: the
 * entry must return an error, not read or free an uninitialised builder. */
static void test_stream_lower_name_overflow(void)
{
    char long_name[400];
    char err[ERR_CAP];
    char* ll = NULL;
    rocke_direct_depthwise_dgrad_stream_spec_t s = stream(dw_problem(3, 3, 1, -1));
    rocke_status_t st;

    memset(long_name, 'x', sizeof(long_name) - 1);
    long_name[sizeof(long_name) - 1] = '\0';
    s.name = long_name;
    err[0] = '\0';
    st = rocke_direct_depthwise_dgrad_stream_lower_to_llvm(
        &s, "gfx1250", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    CHECK(st != ROCKE_OK, "an overflowing kernel name must fail the lowering");
    CHECK(ll == NULL, "no IR on failure");
    CHECK(err[0] != '\0', "the failure carries a reason");
    free(ll);

    /* The same spec with its default name still lowers. */
    s = stream(dw_problem(3, 3, 1, -1));
    st = rocke_direct_depthwise_dgrad_stream_lower_to_llvm(
        &s, "gfx1250", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
    CHECK(st == ROCKE_OK && ll != NULL, "default-name lowering failed: %s", err);
    free(ll);
}

static void test_pad_w_below_minus_one_refused(void)
{
    char reason[ERR_CAP];
    rocke_direct_depthwise_tiled_spec_t t = tiled_dgrad(dw_problem(5, 5, 1, -2));
    rocke_direct_depthwise_dgrad_stream_spec_t s = stream(dw_problem(3, 3, 1, -3));

    CHECK(rocke_direct_depthwise_tiled_validate(&t, reason, sizeof(reason)) != ROCKE_OK,
          "tiled dgrad accepted pad_w=-2");
    CHECK(strstr(reason, "pad_w") != NULL, "reason names pad_w: %s", reason);
    t.dgrad = false;
    CHECK(rocke_direct_depthwise_tiled_validate(&t, reason, sizeof(reason)) != ROCKE_OK,
          "forward tiled accepted pad_w=-2");
    CHECK(rocke_direct_depthwise_dgrad_stream_validate(&s, reason, sizeof(reason)) != ROCKE_OK,
          "stream validate accepted pad_w=-3");
    CHECK(!rocke_direct_depthwise_dgrad_stream_is_valid_spec(&s, "gfx1250", reason, sizeof(reason)),
          "stream is_valid accepted pad_w=-3");

    /* -1 (PAD) and an explicit value equal to PAD stay valid. */
    t = tiled_dgrad(dw_problem(5, 5, 1, -1));
    CHECK(rocke_direct_depthwise_tiled_validate(&t, reason, sizeof(reason)) == ROCKE_OK,
          "tiled dgrad refused pad_w=-1: %s",
          reason);
    s = stream(dw_problem(3, 3, 1, 1));
    CHECK(rocke_direct_depthwise_dgrad_stream_is_valid_spec(&s, "gfx1250", reason, sizeof(reason)),
          "stream refused pad_w=PAD: %s",
          reason);
}

/* The row-streaming kernel offsets columns by PAD: it takes one padding. */
static void test_stream_refuses_per_axis_padding(void)
{
    char reason[ERR_CAP];
    rocke_direct_depthwise_dgrad_stream_spec_t s = stream(dw_problem(5, 5, 2, 1));
    CHECK(rocke_direct_depthwise_dgrad_stream_validate(&s, reason, sizeof(reason)) != ROCKE_OK,
          "stream validate accepted PAD=2 x PAD_W=1");
    CHECK(!rocke_direct_depthwise_dgrad_stream_is_valid_spec(&s, "gfx1250", reason, sizeof(reason)),
          "stream is_valid accepted PAD=2 x PAD_W=1");
}

static void test_tiled_dgrad_name_carries_per_axis_padding(void)
{
    char n0[NAME_CAP], n1[NAME_CAP], nsym[NAME_CAP], ndef[NAME_CAP];
    rocke_direct_depthwise_tiled_spec_t a = tiled_dgrad(dw_problem(5, 5, 2, 0));
    rocke_direct_depthwise_tiled_spec_t b = tiled_dgrad(dw_problem(5, 5, 2, 1));
    rocke_direct_depthwise_tiled_spec_t sym = tiled_dgrad(dw_problem(5, 5, 2, 2));
    rocke_direct_depthwise_tiled_spec_t def = tiled_dgrad(dw_problem(5, 5, 2, -1));

    CHECK(rocke_direct_depthwise_tiled_kernel_name(&a, n0, sizeof(n0)) == ROCKE_OK, "name a");
    CHECK(rocke_direct_depthwise_tiled_kernel_name(&b, n1, sizeof(n1)) == ROCKE_OK, "name b");
    CHECK(rocke_direct_depthwise_tiled_kernel_name(&sym, nsym, sizeof(nsym)) == ROCKE_OK,
          "name sym");
    CHECK(rocke_direct_depthwise_tiled_kernel_name(&def, ndef, sizeof(ndef)) == ROCKE_OK,
          "name def");
    CHECK(strcmp(n0, n1) != 0, "pad_w 0 and 1 share the name %s", n0);
    CHECK(strstr(n0, "_f5x5_p2x0_") != NULL, "missing padding tag: %s", n0);
    CHECK(strcmp(nsym, ndef) == 0, "equal paddings must keep one name (%s vs %s)", nsym, ndef);
    CHECK(strstr(ndef, "_p2x") == NULL, "equal paddings gained a tag: %s", ndef);
}

/* The rotated-filter dgrad mode packs one channel per lane; cpl > 1 is the
 * forward mode's packed kernel only. */
static void test_tiled_dgrad_refuses_cpl(void)
{
    char reason[ERR_CAP];
    rocke_direct_depthwise_tiled_spec_t t = tiled_dgrad(dw_problem(5, 5, 2, -1));
    t.cpl = 2;
    CHECK(rocke_direct_depthwise_tiled_validate(&t, reason, sizeof(reason)) != ROCKE_OK,
          "tiled dgrad accepted cpl=2");
    CHECK(strstr(reason, "cpl=2") != NULL, "reason names cpl: %s", reason);
    t.dgrad = false;
    CHECK(rocke_direct_depthwise_tiled_validate(&t, reason, sizeof(reason)) == ROCKE_OK,
          "forward tiled refused cpl=2: %s",
          reason);
}

static rocke_direct_conv_problem_t dense_problem(int H, int W, int C, int K)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.N = 2;
    p.H = H;
    p.W = W;
    p.groups = 1;
    p.cpg = C;
    p.kpg = K;
    p.dtype = "bf16";
    return p;
}

template <typename Spec, typename Valid>
static void check_wmma_padding(const char* kernel, Spec spec, Valid is_valid)
{
    char reason[ERR_CAP];
    char expected[ERR_CAP];
    const int pad = spec.problem.PAD;
    spec.problem.pad_w = -1;
    CHECK(is_valid(&spec, "gfx1250", reason, sizeof(reason)),
          "%s refused pad_w=-1: %s",
          kernel,
          reason);
    spec.problem.pad_w = pad;
    CHECK(is_valid(&spec, "gfx1250", reason, sizeof(reason)),
          "%s refused pad_w=PAD: %s",
          kernel,
          reason);
    spec.problem.pad_w = 1 - pad;
    CHECK(!is_valid(&spec, "gfx1250", reason, sizeof(reason)),
          "%s accepted pad_w=%d",
          kernel,
          1 - pad);
    snprintf(expected, sizeof(expected), "one padding for both axes (got PAD=%dx%d)", pad, 1 - pad);
    CHECK(strcmp(reason, expected) == 0, "%s reason: %s", kernel, reason);
    spec.problem.pad_w = -2;
    CHECK(!is_valid(&spec, "gfx1250", reason, sizeof(reason)), "%s accepted pad_w=-2", kernel);
    CHECK(strstr(reason, "pad_w") != NULL, "%s reason names pad_w: %s", kernel, reason);
}

/* The direct and small-K WMMA forwards offset columns by PAD_W: they take a
 * column padding that differs from PAD, and still refuse pad_w below -1. */
template <typename Spec, typename Valid>
static void check_wmma_per_axis_padding(const char* kernel, Spec spec, Valid is_valid)
{
    char reason[ERR_CAP];
    const int pad = spec.problem.PAD;
    for(int pad_w : {-1, pad, 1 - pad})
    {
        spec.problem.pad_w = pad_w;
        CHECK(is_valid(&spec, "gfx1250", reason, sizeof(reason)),
              "%s refused pad_w=%d: %s",
              kernel,
              pad_w,
              reason);
    }
    spec.problem.pad_w = -2;
    CHECK(!is_valid(&spec, "gfx1250", reason, sizeof(reason)), "%s accepted pad_w=-2", kernel);
    CHECK(strstr(reason, "pad_w") != NULL, "%s reason names pad_w: %s", kernel, reason);
}

static void test_wmma_kernels_refuse_per_axis_padding(void)
{
    rocke_direct_fwd_wmma_spec_t fwd = rocke_direct_fwd_wmma_spec_default();
    fwd.problem = dense_problem(13, 21, 64, 96);
    check_wmma_per_axis_padding("direct_fwd_wmma", fwd, rocke_direct_fwd_wmma_is_valid_spec);
    rocke_direct_dgrad_wmma_spec_t dgrad = rocke_direct_dgrad_wmma_spec_default();
    dgrad.problem = dense_problem(13, 21, 64, 96);
    check_wmma_padding("direct_dgrad_wmma", dgrad, rocke_direct_dgrad_wmma_is_valid_spec);
    rocke_fwd_smallc_wmma_spec_t smallc = rocke_fwd_smallc_wmma_spec_default();
    smallc.problem = dense_problem(32, 48, 3, 64);
    check_wmma_padding("fwd_smallc_wmma", smallc, rocke_fwd_smallc_wmma_is_valid_spec);
    rocke_fwd_smallk_wmma_spec_t smallk = rocke_fwd_smallk_wmma_spec_default();
    smallk.problem = dense_problem(32, 48, 64, 4);
    check_wmma_per_axis_padding("fwd_smallk_wmma", smallk, rocke_fwd_smallk_wmma_is_valid_spec);
}

/* `spec` must be refused with exactly `want` (Python's reason), or accepted
 * when `want` is NULL. */
static void
    expect_fwd_wmma(const char* what, const rocke_direct_fwd_wmma_spec_t* spec, const char* want)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    const bool ok = rocke_direct_fwd_wmma_is_valid_spec(spec, "gfx1250", reason, sizeof(reason));
    if(want == NULL)
        CHECK(ok, "direct_fwd_wmma refused %s: %s", what, reason);
    else
    {
        CHECK(!ok, "direct_fwd_wmma accepted %s", what);
        CHECK(strcmp(reason, want) == 0, "direct_fwd_wmma %s reason: %s", what, reason);
    }
}

/* A 1-D poly_view spec on one row (tile_w == block_p), as Python's _PV1. */
static rocke_direct_fwd_wmma_spec_t
    poly_view_spec(int W, int C, int K, int KW, int stride, int pad_w)
{
    rocke_direct_fwd_wmma_spec_t s = rocke_direct_fwd_wmma_spec_default();
    s.problem = dense_problem(1, W, C, K);
    s.problem.KH = 1;
    s.problem.KW = KW;
    s.problem.PAD = 0;
    s.problem.pad_w = pad_w;
    s.problem.stride = stride;
    s.poly_view = true;
    s.waves_k = 1;
    s.waves_p = 4;
    s.tiles_k = 1;
    s.tiles_p = 2;
    s.tile_w = 128;
    s.stages = 2;
    return s;
}

/* A leading-pad chunk's first view pixel is zeroed by one wave, 8 channels a
 * lane: with column padding, ck above 256 is refused. */
static void test_fwd_wmma_padded_poly_view_chunk(void)
{
    rocke_direct_fwd_wmma_spec_t s = poly_view_spec(600, 288, 64, 4, 2, 1);
    s.ck = 288;
    expect_fwd_wmma("ck 288, pad_w 1",
                    &s,
                    "a padded poly_view zeroes at most 256 channels of a chunk (ck 288)");
    s = poly_view_spec(600, 512, 64, 4, 2, 1);
    s.ck = 512;
    expect_fwd_wmma("ck 512, pad_w 1",
                    &s,
                    "a padded poly_view zeroes at most 256 channels of a chunk (ck 512)");
    s = poly_view_spec(600, 288, 64, 4, 2, 0);
    s.ck = 288;
    expect_fwd_wmma("ck 288, pad_w 0", &s, NULL);
    s = poly_view_spec(600, 256, 64, 4, 2, 1);
    s.ck = 256;
    expect_fwd_wmma("ck 256, pad_w 1", &s, NULL);
}

/* A KxK strided_view patch run is KW * cpg contiguous channels only when one
 * group holds a pixel's channels; a 1x1 one still serves grouped convs. */
static void test_fwd_wmma_strided_view_refusals(void)
{
    rocke_direct_fwd_wmma_spec_t s = rocke_direct_fwd_wmma_spec_default();
    s.strided_view = true;
    s.problem = dense_problem(13, 21, 64, 64);
    s.problem.groups = 2;
    s.problem.KH = 2;
    s.problem.KW = 2;
    s.problem.PAD = 0;
    s.problem.stride = 2;
    expect_fwd_wmma("a grouped 2x2 strided_view", &s, "a KxK strided_view serves ungrouped convs");
    s.problem.groups = 1;
    expect_fwd_wmma("a dense 2x2 strided_view", &s, NULL);
    s.problem.groups = 2;
    s.problem.KH = 1;
    s.problem.KW = 1;
    expect_fwd_wmma("a grouped 1x1 strided_view", &s, NULL);
    s.problem = dense_problem(13, 21, 64, 96);
    s.problem.KH = 2;
    s.problem.KW = 2;
    s.problem.PAD = 0;
    s.problem.stride = 2;
    s.ck = 0;
    expect_fwd_wmma("a strided_view with ck 0", &s, "ck 0 must be a positive multiple of 32");
}

/* Below S / 2 the view's last pixel is all input, not a trailing zero; a zero
 * ck is refused, not divided by. */
static void test_fwd_wmma_poly_view_refusals(void)
{
    rocke_direct_fwd_wmma_spec_t s = poly_view_spec(512, 64, 96, 8, 4, 1);
    expect_fwd_wmma(
        "poly_view S 4, pad_w 1", &s, "poly_view needs column padding 0 or at least S / 2");
    s = poly_view_spec(512, 64, 96, 8, 4, 2);
    expect_fwd_wmma("poly_view S 4, pad_w 2", &s, NULL);
    s = poly_view_spec(600, 64, 96, 4, 2, 1);
    s.ck = 0;
    expect_fwd_wmma(
        "poly_view with ck 0", &s, "a chunk must lie in one input pixel (cpg 64, ck 0)");
}

/* A slab in the top padding issues no transfer, so a partial tensorcnt wait
 * can pass early: tdm_split > 1 needs 2 stages. */
static void test_fwd_tapouter_tdm_split_stages(void)
{
    char reason[ERR_CAP];
    rocke_fwd_tapouter_wmma_spec_t s = rocke_fwd_tapouter_wmma_spec_default();
    s.problem = dense_problem(29, 37, 64, 96);
    s.problem.PAD = 12;
    s.problem.dil_h = 12;
    s.problem.dil_w = 12;
    s.tdm_split = 2;
    for(int stages : {3, 4})
    {
        s.stages = stages;
        CHECK(!rocke_fwd_tapouter_wmma_is_valid_spec(&s, "gfx1250", reason, sizeof(reason)),
              "tap-outer accepted tdm_split 2 with %d stages",
              stages);
        CHECK(strcmp(reason, "tdm_split 2 needs 2 stages (a skipped slab breaks the wait)") == 0,
              "tap-outer reason: %s",
              reason);
    }
    s.stages = 2;
    CHECK(rocke_fwd_tapouter_wmma_is_valid_spec(&s, "gfx1250", reason, sizeof(reason)),
          "tap-outer refused tdm_split 2 with 2 stages: %s",
          reason);
}

static rocke_direct_conv_problem_t stem_problem(int KH, int stride, int PAD)
{
    rocke_direct_conv_problem_t p = dense_problem(32, 48, 3, 64);
    p.KH = KH;
    p.KW = KH;
    p.stride = stride;
    p.PAD = PAD;
    return p;
}

static void test_stem_wmma_kernels_refuse_per_axis_padding(void)
{
    rocke_dgrad_patch_wmma_spec_t patch = rocke_dgrad_patch_wmma_spec_default();
    patch.problem = stem_problem(4, 4, 0);
    patch.waves_c = 3;
    patch.waves_p = 1;
    patch.tiles_c = 1;
    patch.r_blk = 4;
    check_wmma_padding("dgrad_patch_wmma", patch, rocke_dgrad_patch_wmma_is_valid_spec);
    rocke_dgrad_smallc_wmma_spec_t smallc = rocke_dgrad_smallc_wmma_spec_default();
    smallc.problem = stem_problem(3, 1, 1);
    smallc.tiles_p = 6;
    smallc.out_h = 16;
    smallc.out_w = 16;
    check_wmma_padding("dgrad_smallc_wmma", smallc, rocke_dgrad_smallc_wmma_is_valid_spec);
}

/* The validator tail (rocke_wmma_fwd_limits_ok) refuses an unknown scheduler
 * with Python's ValueError text, word for word. */
static void test_fwd_wmma_scheduler_reason(void)
{
    rocke_direct_fwd_wmma_spec_t s = rocke_direct_fwd_wmma_spec_default();
    s.problem = dense_problem(13, 21, 64, 96);
    s.sched = "bogus";
    expect_fwd_wmma("sched bogus",
                    &s,
                    "unsupported scheduler_strategy 'bogus'; expected one of: iterative-ilp, "
                    "iterative-maxocc, iterative-minreg, max-ilp, max-memory-clause, or None");
    s.sched = "iterative-ilp";
    expect_fwd_wmma("sched iterative-ilp", &s, NULL);

    /* Small C / small K write the same text (they had a shorter one). */
    const char* want = "unsupported scheduler_strategy 'bogus'; expected one of: iterative-ilp, "
                       "iterative-maxocc, iterative-minreg, max-ilp, max-memory-clause, or None";
    char reason[ERR_CAP];
    rocke_fwd_smallc_wmma_spec_t smallc = rocke_fwd_smallc_wmma_spec_default();
    smallc.problem = dense_problem(32, 48, 3, 64);
    smallc.sched = "bogus";
    CHECK(!rocke_fwd_smallc_wmma_is_valid_spec(&smallc, "gfx1250", reason, sizeof(reason))
              && strcmp(reason, want) == 0,
          "fwd_smallc_wmma sched bogus: %s",
          reason);
    rocke_fwd_smallk_wmma_spec_t smallk = rocke_fwd_smallk_wmma_spec_default();
    smallk.problem = dense_problem(32, 48, 64, 4);
    smallk.sched = "bogus";
    CHECK(!rocke_fwd_smallk_wmma_is_valid_spec(&smallk, "gfx1250", reason, sizeof(reason))
              && strcmp(reason, want) == 0,
          "fwd_smallk_wmma sched bogus: %s",
          reason);

    /* rocke_scheduler_strategy_is_valid reads ROCKE_SCHEDULER_STRATEGY_CHOICES:
     * every listed name, and no prefix, suffix or empty name. */
    const char* good[]
        = {"iterative-ilp", "iterative-maxocc", "iterative-minreg", "max-ilp", "max-memory-clause"};
    for(const char* g : good)
        CHECK(rocke_scheduler_strategy_is_valid(g), "strategy %s refused", g);
    const char* bad[] = {"", "max", "max-ilp,", "iterative-ilp, iterative-maxocc", "max-memory"};
    for(const char* x : bad)
        CHECK(!rocke_scheduler_strategy_is_valid(x), "strategy '%s' accepted", x);
    CHECK(rocke_scheduler_strategy_is_valid(NULL), "NULL strategy refused");
}

/* Python: a // b, ceil_div(a, b) = -(-a // b) and align_up(a, b) =
 * ceil_div(a, b) * b, for every sign. */
static void test_ceil_floor_div(void)
{
    const long long up[][3] = {/* n, mult, align_up(n, mult) */
                               {7, 4, 8},
                               {8, 4, 8},
                               {0, 4, 0},
                               {-7, 4, -4},
                               {13, 16, 16}};
    for(const auto& u : up)
        CHECK(rocke_align_up(u[0], u[1]) == u[2],
              "rocke_align_up(%lld, %lld) = %lld, want %lld",
              u[0],
              u[1],
              rocke_align_up(u[0], u[1]),
              u[2]);
    const long long cases[][4] = {/* a, b, a // b, ceil_div(a, b) */
                                  {7, 2, 3, 4},
                                  {8, 2, 4, 4},
                                  {0, 3, 0, 0},
                                  {-7, 2, -4, -3},
                                  {-8, 2, -4, -4},
                                  {7, -2, -4, -3},
                                  {-7, -2, 3, 4},
                                  {1, 32, 0, 1},
                                  {4294967296LL, 3, 1431655765LL, 1431655766LL}};
    for(const auto& c : cases)
    {
        CHECK(rocke_floor_div(c[0], c[1]) == c[2],
              "rocke_floor_div(%lld, %lld) = %lld, want %lld",
              c[0],
              c[1],
              rocke_floor_div(c[0], c[1]),
              c[2]);
        CHECK(rocke_ceil_div(c[0], c[1]) == c[3],
              "rocke_ceil_div(%lld, %lld) = %lld, want %lld",
              c[0],
              c[1],
              rocke_ceil_div(c[0], c[1]),
              c[3]);
    }
}

/* DirectConvProblem.Ho / Wo: strided, dilated, column padding PAD_W, and
 * Python's floor division when the input is smaller than the filter. */
static void test_problem_output_extents(void)
{
    rocke_direct_conv_problem_t p = dense_problem(35, 35, 64, 64);
    p.KH = 3;
    p.KW = 3;
    p.PAD = 0;
    p.stride = 2;
    CHECK(rocke_direct_conv_problem_Ho(&p) == 17 && rocke_direct_conv_problem_Wo(&p) == 17,
          "3x3 s2 valid on 35x35: %lldx%lld",
          rocke_direct_conv_problem_Ho(&p),
          rocke_direct_conv_problem_Wo(&p));
    p = dense_problem(1, 100, 64, 64);
    p.KH = 1;
    p.KW = 7;
    p.PAD = 0;
    p.pad_w = 9;
    p.dil_w = 3;
    /* Wo = (100 + 18 - 18 - 1) // 1 + 1 = 100 */
    CHECK(rocke_direct_conv_problem_Ho(&p) == 1 && rocke_direct_conv_problem_Wo(&p) == 100,
          "1x7 d3 pw9 on 1x100: %lldx%lld",
          rocke_direct_conv_problem_Ho(&p),
          rocke_direct_conv_problem_Wo(&p));
    p = dense_problem(2, 2, 64, 64);
    p.KH = 5;
    p.KW = 5;
    p.PAD = 0;
    p.stride = 2;
    /* (2 - 4 - 1) // 2 + 1 = -1 in Python; truncation would give 0. */
    CHECK(rocke_direct_conv_problem_Ho(&p) == -1 && rocke_direct_conv_problem_Wo(&p) == -1,
          "5x5 s2 on 2x2: %lldx%lld",
          rocke_direct_conv_problem_Ho(&p),
          rocke_direct_conv_problem_Wo(&p));
    CHECK(rocke_direct_conv_problem_total_c(&p) == 64
              && rocke_direct_conv_problem_total_k(&p) == 64,
          "total_c / total_k");
}

/* total_c / total_k are 64-bit like Python's ints: groups * cpg = 2^31 does
 * not wrap, and the forward validators refuse its tensors with Python's
 * reason (an int product wrapped negative and let these specs pass). */
static void test_total_channels_past_i32(void)
{
    rocke_direct_conv_problem_t p = dense_problem(8, 8, 16, 16);
    p.N = 1;
    p.groups = 1 << 27;
    CHECK(rocke_direct_conv_problem_total_c(&p) == 2147483648LL
              && rocke_direct_conv_problem_total_k(&p) == 2147483648LL,
          "total_c / total_k of 2^27 groups x 16: %lld / %lld",
          rocke_direct_conv_problem_total_c(&p),
          rocke_direct_conv_problem_total_k(&p));
    char reason[ERR_CAP];

    rocke_fwd_gstream_wmma_spec_t gs = rocke_fwd_gstream_wmma_spec_default();
    gs.problem = p;
    reason[0] = '\0';
    CHECK(!rocke_fwd_gstream_wmma_is_valid_spec(&gs, "gfx1250", reason, sizeof(reason)),
          "direct_fwd_gstream_wmma accepted 2^31 channels");
    CHECK(strcmp(reason, "tensors of 618475290624 B exceed the i32 byte offsets") == 0,
          "direct_fwd_gstream_wmma 2^31 channels reason: %s",
          reason);

    rocke_direct_fwd_wmma_spec_t fw = rocke_direct_fwd_wmma_spec_default();
    fw.problem = dense_problem(8, 8, 64, 64);
    fw.problem.N = 1;
    fw.problem.groups = 1 << 25;
    expect_fwd_wmma("2^31 channels", &fw, "tensors of 2473901162496 B exceed the i32 byte offsets");

    rocke_fwd_tapouter_wmma_spec_t to = rocke_fwd_tapouter_wmma_spec_default();
    to.problem = dense_problem(33, 33, 64, 64);
    to.problem.N = 1;
    to.problem.groups = 1 << 25;
    to.problem.PAD = 2;
    to.problem.dil_h = 2;
    to.problem.dil_w = 2;
    reason[0] = '\0';
    CHECK(!rocke_fwd_tapouter_wmma_is_valid_spec(&to, "gfx1250", reason, sizeof(reason)),
          "direct_fwd_tapouter_wmma accepted 2^31 channels");
    CHECK(strcmp(reason, "tensors of 4677219385344 B exceed the i32 byte offsets") == 0,
          "direct_fwd_tapouter_wmma 2^31 channels reason: %s",
          reason);
}

int main(void)
{
    test_total_channels_past_i32();
    test_ceil_floor_div();
    test_problem_output_extents();
    test_fwd_wmma_scheduler_reason();
    test_stream_lower_name_overflow();
    test_pad_w_below_minus_one_refused();
    test_stream_refuses_per_axis_padding();
    test_tiled_dgrad_name_carries_per_axis_padding();
    test_tiled_dgrad_refuses_cpl();
    test_wmma_kernels_refuse_per_axis_padding();
    test_stem_wmma_kernels_refuse_per_axis_padding();
    test_fwd_wmma_strided_view_refusals();
    test_fwd_wmma_poly_view_refusals();
    test_fwd_wmma_padded_poly_view_chunk();
    test_fwd_tapouter_tdm_split_stages();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_dw_dgrad_host: OK\n");
    return 0;
}
