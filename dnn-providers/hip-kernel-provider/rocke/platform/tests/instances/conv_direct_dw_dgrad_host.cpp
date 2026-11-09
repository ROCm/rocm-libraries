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
 *     must carry normalize_scheduler_strategy's text, choices included (and
 *     so must the dgrad WMMA and Winograd GEMM validators), the name quoted
 *     as Python's repr (rocke_py_str_repr, the engine's one port);
 *   - the direct wgrad validators' refusal of a dilated problem, which must
 *     carry conv_direct_grouped.dilation_reason's text;
 *   - a zero or negative extent (N, H, W, groups, cpg, kpg, KH, KW) given to
 *     the dw fwd and forward WMMA validators, which must refuse it first with
 *     conv_direct_grouped.problem_dims_reason's text.
 *   - the specs the Toeplitz form of the windowed depthwise dgrad refuses
 *     (parity configs 68..72 of conv_direct_dgrad_ext_emit): the CI goldens
 *     run the Python engine only, so the C++ reasons are checked here.
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
#include "rocke/instance_conv_direct_dw_fwd_wmma.h"
#include "rocke/instance_conv_direct_dw_wgrad.h"
#include "rocke/instance_conv_direct_dw_wgrad_mfma.h"
#include "rocke/instance_conv_direct_fwd_gstream_wmma.h"
#include "rocke/instance_conv_direct_fwd_tapouter_wmma.h"
#include "rocke/instance_conv_direct_fwd_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_dw_dgrad.h"
#include "rocke/instance_conv_direct_grouped_wgrad_mfma.h"
#include "rocke/instance_conv_direct_wgrad_gather.h"
#include "rocke/instance_conv_direct_wgrad_tiled.h"
#include "rocke/instance_conv_fwd_small_wmma.h"
#include "rocke/instance_conv_winograd_gemm_wmma.h"
#include "rocke/lower_llvm.h"

/* Layout canary, as in test_mma_catalog_lookup.cpp: growing
 * rocke_direct_fwd_wmma_spec_t fails the build here until ROCKE_ABI_VERSION in
 * rocke/abi.h moves with it. 64-bit only. */
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(rocke_direct_fwd_wmma_spec_t) == 152 && ROCKE_ABI_VERSION >= 8,
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

/* The Toeplitz windowed depthwise dgrad on the wave32 WMMA (gfx1250), in the
 * shape of conv_direct_dgrad_ext_emit config 62 (7x7, 8 waves, 28-row tile). */
static rocke_direct_depthwise_dgrad_win_spec_t win_wmma(int KH, int PAD, const char* dtype)
{
    rocke_direct_depthwise_dgrad_win_spec_t s = rocke_direct_depthwise_dgrad_win_spec_default();
    s.problem = rocke_direct_conv_problem_default();
    s.problem.N = 2;
    s.problem.H = 14;
    s.problem.W = 14;
    s.problem.groups = 64;
    s.problem.cpg = 1;
    s.problem.kpg = 1;
    s.problem.KH = KH;
    s.problem.KW = KH;
    s.problem.PAD = PAD;
    s.problem.dtype = dtype;
    s.block_waves = 8;
    s.block_h = 14;
    s.mfma = true;
    s.w_fold = 2;
    s.prefetch_rows = 2;
    s.wave_size = 32;
    return s;
}

static void check_win_refused(const char* what,
                              const rocke_direct_depthwise_dgrad_win_spec_t* s,
                              const char* arch,
                              const char* expected)
{
    char reason[ERR_CAP];
    reason[0] = '\0';
    CHECK(!rocke_direct_depthwise_dgrad_win_is_valid_spec(s, arch, reason, sizeof(reason)),
          "%s accepted on %s",
          what,
          arch);
    CHECK(strstr(reason, expected) != NULL, "%s reason: %s", what, reason);
}

static void test_toeplitz_wmma_refusals(void)
{
    char reason[ERR_CAP];
    rocke_direct_depthwise_dgrad_win_spec_t s = win_wmma(7, 3, "bf16");
    CHECK(rocke_direct_depthwise_dgrad_win_is_valid_spec(&s, "gfx1250", reason, sizeof(reason)),
          "the wave32 Toeplitz form refused on gfx1250: %s",
          reason);

    /* 68: a 9x9 filter's wave32 fragments exceed the VGPR cap. */
    s = win_wmma(9, 4, "bf16");
    s.block_waves = 2;
    s.w_fold = 4;
    s.prefetch_rows = 1;
    s.block_h = 5;
    check_win_refused("9x9 at wave32", &s, "gfx1250", "VGPRs (> 224)");
    /* 69: the wave32 form on the wave64 MFMA target. */
    s = win_wmma(7, 3, "bf16");
    check_win_refused(
        "wave32 on gfx950", &s, "gfx950", "wave32 mfma needs one of ('gfx1250',) (got gfx950)");
    /* 70: the wave64 MFMA form on gfx1250, with the wave32 hint. */
    s.wave_size = 64;
    check_win_refused("wave64 on gfx1250",
                      &s,
                      "gfx1250",
                      "mfma needs one of ('gfx950',) (got gfx1250); set wave_size=32");
    /* No wave32 hint on a target without the WMMA form. */
    CHECK(!rocke_direct_depthwise_dgrad_win_is_valid_spec(&s, "gfx942", reason, sizeof(reason)),
          "wave64 accepted on gfx942");
    CHECK(strcmp(reason, "mfma needs one of ('gfx950',) (got gfx942)") == 0,
          "gfx942 reason: %s",
          reason);
    /* 71 / 72: fwd at stride 1 and 2; the forward mode is wave64 only. */
    s = win_wmma(3, 1, "fp16");
    s.fwd = true;
    check_win_refused(
        "fwd at wave32", &s, "gfx1250", "fwd needs the wave64 MFMA form (got wave_size=32)");
    s.problem.dtype = "bf16";
    s.out_stride = 2;
    check_win_refused(
        "fwd s2 at wave32", &s, "gfx1250", "fwd needs the wave64 MFMA form (got wave_size=32)");
    /* The same forward spec at wave64 stays valid on gfx950. */
    s.wave_size = 64;
    CHECK(rocke_direct_depthwise_dgrad_win_is_valid_spec(&s, "gfx950", reason, sizeof(reason)),
          "the wave64 forward form refused on gfx950: %s",
          reason);
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

/* rocke_py_str_repr is Python's repr() of a str (or of None): the quote
 * choice, the escapes, the snprintf-style length and truncation. The
 * scheduler refusal quotes the name with it, as Python's {value!r}, and the
 * IR printer its region labels and list attrs. */
static void test_py_str_repr(void)
{
    const char* cases[][2] = {
        {"bogus", "'bogus'"},
        {"it's", "\"it's\""},
        {"a\\b", "'a\\\\b'"},
        {"x'\"y", "'x\\'\"y'"},
        {"say \"hi\"", "'say \"hi\"'"},
        {"t\tn\nr\r", "'t\\tn\\nr\\r'"},
        {"\x01\x1f\x7f", "'\\x01\\x1f\\x7f'"},
        {"caf\xc3\xa9", "'caf\xc3\xa9'"},
        {"", "''"},
        {NULL, "None"},
    };
    char out[64];
    for(const auto& c : cases)
    {
        const size_t n = rocke_py_str_repr(c[0], out, sizeof out);
        CHECK(strcmp(out, c[1]) == 0 && n == strlen(c[1]),
              "repr(%s) = %s",
              c[0] != NULL ? c[0] : "None",
              out);
    }
    CHECK(rocke_py_str_repr("abcdef", out, 4) == 8 && strcmp(out, "'ab") == 0,
          "truncated repr: %s",
          out);
    CHECK(rocke_py_str_repr("abc", NULL, 0) == 5, "repr length without a buffer");
    CHECK(rocke_py_str_repr(NULL, out, 3) == 4 && strcmp(out, "No") == 0,
          "truncated repr(None): %s",
          out);
    char reason[ERR_CAP];
    CHECK(rocke_scheduler_strategy_reason("it's", reason, sizeof reason)
              && strcmp(reason,
                        "unsupported scheduler_strategy \"it's\"; expected one of: iterative-ilp, "
                        "iterative-maxocc, iterative-minreg, max-ilp, max-memory-clause, or None")
                     == 0,
          "sched it's: %s",
          reason);
    char small[8];
    CHECK(rocke_scheduler_strategy_reason("bogus", small, sizeof small)
              && strcmp(small, "unsuppo") == 0,
          "truncated sched reason: %s",
          small);
    char mid[34];
    CHECK(rocke_scheduler_strategy_reason("bogus", mid, sizeof mid)
              && strcmp(mid, "unsupported scheduler_strategy 'b") == 0,
          "sched reason cut inside the name: %s",
          mid);
    const char* prefix = "unsupported scheduler_strategy 'a\\\\b'; expected";
    CHECK(rocke_scheduler_strategy_reason("a\\b", reason, sizeof reason)
              && strncmp(reason, prefix, strlen(prefix)) == 0,
          "sched a\\b: %s",
          reason);
}

/* `spec` is accepted without a scheduler and refused, with Python's
 * ValueError text, with the unknown "bogus" one. */
template <typename Spec, typename Valid>
static void check_scheduler_reason(const char* kernel, Spec spec, Valid is_valid)
{
    char reason[ERR_CAP];
    spec.sched = NULL;
    reason[0] = '\0';
    CHECK(is_valid(&spec, "gfx1250", reason, sizeof(reason)), "%s refused: %s", kernel, reason);
    spec.sched = "bogus";
    reason[0] = '\0';
    CHECK(!is_valid(&spec, "gfx1250", reason, sizeof(reason)), "%s accepted sched bogus", kernel);
    CHECK(strcmp(reason,
                 "unsupported scheduler_strategy 'bogus'; expected one of: iterative-ilp, "
                 "iterative-maxocc, iterative-minreg, max-ilp, max-memory-clause, or None")
              == 0,
          "%s sched bogus: %s",
          kernel,
          reason);
}

/* The dgrad WMMA and Winograd GEMM validators write the full text too (they
 * wrote only its first clause). */
static void test_dgrad_wmma_scheduler_reason(void)
{
    rocke_direct_dgrad_wmma_spec_t dg = rocke_direct_dgrad_wmma_spec_default();
    check_scheduler_reason("direct_dgrad_wmma", dg, rocke_direct_dgrad_wmma_is_valid_spec);
    rocke_dgrad_patch_wmma_spec_t patch = rocke_dgrad_patch_wmma_spec_default();
    patch.problem = stem_problem(4, 4, 0);
    patch.waves_c = 3;
    patch.waves_p = 1;
    patch.tiles_c = 1;
    patch.r_blk = 4;
    check_scheduler_reason("dgrad_patch_wmma", patch, rocke_dgrad_patch_wmma_is_valid_spec);
    rocke_dgrad_smallc_wmma_spec_t smallc = rocke_dgrad_smallc_wmma_spec_default();
    smallc.problem = stem_problem(3, 1, 1);
    smallc.tiles_p = 6;
    smallc.out_h = 16;
    smallc.out_w = 16;
    check_scheduler_reason("dgrad_smallc_wmma", smallc, rocke_dgrad_smallc_wmma_is_valid_spec);
    check_scheduler_reason("winograd_gemm_wmma",
                           rocke_winograd_gemm_wmma_spec_default(),
                           rocke_winograd_gemm_wmma_is_valid_spec);
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

/* Every direct wgrad validator refuses a dilated problem with Python's
 * dilation_reason text (test_direct_wgrad_specs.py pins the Python side to
 * the same literal). */
template <typename Spec, typename Valid>
static void check_wgrad_dilation(const char* kernel, Spec spec, Valid is_valid)
{
    char reason[ERR_CAP];
    spec.problem.dil_h = 2;
    reason[0] = '\0';
    CHECK(!is_valid(&spec, "gfx950", reason, sizeof(reason)), "%s accepted dilation 2x1", kernel);
    CHECK(strcmp(reason, "this direct conv kernel has no dilation (got 2x1)") == 0,
          "%s dilation reason: %s",
          kernel,
          reason);
}

static void test_wgrad_dilation_reason(void)
{
    check_wgrad_dilation("wgrad_tiled",
                         rocke_direct_wgrad_tiled_spec_default(),
                         rocke_direct_wgrad_tiled_is_valid_spec);
    check_wgrad_dilation("wgrad_gather",
                         rocke_direct_wgrad_gather_spec_default(),
                         rocke_direct_wgrad_gather_is_valid_spec);
    check_wgrad_dilation(
        "dw_wgrad_cols", rocke_dw_wgrad_cols_spec_default(), rocke_dw_wgrad_cols_is_valid_spec);
    check_wgrad_dilation(
        "dw_wgrad_lds", rocke_dw_wgrad_lds_spec_default(), rocke_dw_wgrad_lds_is_valid_spec);
    check_wgrad_dilation(
        "dw_wgrad_mfma", rocke_dw_wgrad_mfma_spec_default(), rocke_dw_wgrad_mfma_is_valid_spec);
    check_wgrad_dilation("grouped_wgrad_mfma",
                         rocke_grouped_wgrad_mfma_spec_default(),
                         rocke_grouped_wgrad_mfma_is_valid_spec);
}

/* Every validator whose builder keeps a channel total in an int refuses
 * groups * cpg or groups * kpg past INT32_MAX first, with Python's
 * channel_totals_reason text (test_conv_direct_grouped_backend_parity.py pins
 * the Python validators to the same literal). */
template <typename Spec, typename Valid>
static void check_channel_totals(const char* kernel, Spec spec, int groups, int ch, Valid is_valid)
{
    char reason[ERR_CAP];
    spec.problem.N = 1;
    spec.problem.H = 8;
    spec.problem.W = 8;
    spec.problem.groups = groups;
    spec.problem.cpg = ch;
    spec.problem.kpg = ch;
    reason[0] = '\0';
    CHECK(!is_valid(&spec, "gfx950", reason, sizeof(reason)), "%s accepted 2^31 channels", kernel);
    CHECK(strcmp(reason, "groups * cpg = 2147483648 and groups * kpg = 2147483648 must fit in i32")
              == 0,
          "%s 2^31 channels reason: %s",
          kernel,
          reason);
}

static void test_channel_totals_refused(void)
{
    check_channel_totals("16c",
                         rocke_direct_conv_16c_spec_default(),
                         1 << 27,
                         16,
                         rocke_direct_conv_16c_is_valid_spec);
    check_channel_totals(
        "4c", rocke_direct_conv_4c_spec_default(), 1 << 29, 4, rocke_direct_conv_4c_is_valid_spec);
    check_channel_totals(
        "8c", rocke_direct_conv_8c_spec_default(), 1 << 28, 8, rocke_direct_conv_8c_is_valid_spec);
    check_channel_totals("32c",
                         rocke_direct_conv_32c_spec_default(),
                         1 << 26,
                         32,
                         rocke_direct_conv_32c_is_valid_spec);
    check_channel_totals(
        "generic", rocke_direct_conv_spec_default(), 1 << 27, 16, rocke_direct_conv_is_valid_spec);
    check_channel_totals("wgrad",
                         rocke_direct_conv_wgrad_spec_default(),
                         1 << 27,
                         16,
                         rocke_direct_conv_wgrad_is_valid_spec);
    check_channel_totals("dgrad",
                         rocke_direct_conv_dgrad_spec_default(),
                         1 << 27,
                         16,
                         rocke_direct_conv_dgrad_is_valid_spec);
    /* The guard runs first, so the depthwise validators see the totals
     * before their cpg = kpg = 1 rule (a C int cannot hold 2^31 groups). */
    check_channel_totals("depthwise",
                         rocke_direct_depthwise_spec_default(),
                         1 << 30,
                         2,
                         rocke_direct_depthwise_is_valid_spec);
    check_channel_totals("depthwise_col",
                         rocke_direct_depthwise_col_spec_default(),
                         1 << 30,
                         2,
                         rocke_direct_depthwise_col_is_valid_spec);
    check_channel_totals("depthwise_spatial",
                         rocke_direct_depthwise_spatial_spec_default(),
                         1 << 30,
                         2,
                         rocke_direct_depthwise_spatial_is_valid_spec);
    check_channel_totals("grouped_wgrad_mfma",
                         rocke_grouped_wgrad_mfma_spec_default(),
                         1 << 28,
                         8,
                         rocke_grouped_wgrad_mfma_is_valid_spec);
}

/* A zero or negative extent is refused first, with Python's
 * problem_dims_reason text: groups = 0 passed every later rule of the dw fwd
 * validator (groups = -16 its divisibility too) and gave an empty grid. */
template <typename Spec, typename Valid>
static void check_problem_dims(const char* kernel, Spec spec, Valid is_valid)
{
    static const char* const names[] = {"N", "H", "W", "groups", "cpg", "kpg", "KH", "KW"};
    for(int i = 0; i < 8; ++i)
    {
        for(const int value : {0, -1, -16})
        {
            Spec s = spec;
            int* const fields[] = {&s.problem.N,
                                   &s.problem.H,
                                   &s.problem.W,
                                   &s.problem.groups,
                                   &s.problem.cpg,
                                   &s.problem.kpg,
                                   &s.problem.KH,
                                   &s.problem.KW};
            *fields[i] = value;
            char reason[ERR_CAP];
            char expected[ERR_CAP];
            snprintf(expected, sizeof(expected), "%s must be >= 1 (got %d)", names[i], value);
            reason[0] = '\0';
            CHECK(!is_valid(&s, "gfx1250", reason, sizeof(reason)),
                  "%s accepted %s = %d",
                  kernel,
                  names[i],
                  value);
            CHECK(strcmp(reason, expected) == 0,
                  "%s %s = %d reason: %s",
                  kernel,
                  names[i],
                  value,
                  reason);
        }
    }
}

static void test_problem_dims_refused(void)
{
    rocke_direct_dw_fwd_wmma_spec_t dw = rocke_direct_dw_fwd_wmma_spec_default();
    dw.problem = dense_problem(20, 37, 1, 1);
    dw.problem.groups = 64;
    check_problem_dims("direct_dw_fwd_wmma", dw, rocke_direct_dw_fwd_wmma_is_valid_spec);

    rocke_direct_fwd_wmma_spec_t fw = rocke_direct_fwd_wmma_spec_default();
    fw.problem = dense_problem(13, 21, 64, 96);
    check_problem_dims("direct_fwd_wmma", fw, rocke_direct_fwd_wmma_is_valid_spec);

    rocke_fwd_gstream_wmma_spec_t gs = rocke_fwd_gstream_wmma_spec_default();
    gs.problem = dense_problem(28, 28, 16, 16);
    gs.problem.groups = 8;
    check_problem_dims("direct_fwd_gstream_wmma", gs, rocke_fwd_gstream_wmma_is_valid_spec);

    rocke_fwd_tapouter_wmma_spec_t to = rocke_fwd_tapouter_wmma_spec_default();
    to.problem = dense_problem(29, 37, 64, 96);
    check_problem_dims("direct_fwd_tapouter_wmma", to, rocke_fwd_tapouter_wmma_is_valid_spec);

    rocke_fwd_smallc_wmma_spec_t sc = rocke_fwd_smallc_wmma_spec_default();
    sc.problem = dense_problem(32, 48, 3, 64);
    check_problem_dims("fwd_smallc_wmma", sc, rocke_fwd_smallc_wmma_is_valid_spec);

    rocke_fwd_smallk_wmma_spec_t sk = rocke_fwd_smallk_wmma_spec_default();
    sk.problem = dense_problem(32, 48, 64, 4);
    check_problem_dims("fwd_smallk_wmma", sk, rocke_fwd_smallk_wmma_is_valid_spec);
}

int main(void)
{
    test_problem_dims_refused();
    test_channel_totals_refused();
    test_wgrad_dilation_reason();
    test_total_channels_past_i32();
    test_ceil_floor_div();
    test_problem_output_extents();
    test_fwd_wmma_scheduler_reason();
    test_dgrad_wmma_scheduler_reason();
    test_py_str_repr();
    test_stream_lower_name_overflow();
    test_pad_w_below_minus_one_refused();
    test_stream_refuses_per_axis_padding();
    test_tiled_dgrad_name_carries_per_axis_padding();
    test_tiled_dgrad_refuses_cpl();
    test_wmma_kernels_refuse_per_axis_padding();
    test_stem_wmma_kernels_refuse_per_axis_padding();
    test_toeplitz_wmma_refusals();
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
