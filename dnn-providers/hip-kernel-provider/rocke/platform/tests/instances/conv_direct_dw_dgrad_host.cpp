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
 *     is refused everywhere).
 */
#include <initializer_list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_dgrad_patch_wmma.h"
#include "rocke/instance_conv_dgrad_smallc_wmma.h"
#include "rocke/instance_conv_direct_dgrad_wmma.h"
#include "rocke/instance_conv_direct_fwd_wmma.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_dw_dgrad.h"
#include "rocke/instance_conv_fwd_small_wmma.h"
#include "rocke/lower_llvm.h"

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

int main(void)
{
    test_stream_lower_name_overflow();
    test_pad_w_below_minus_one_refused();
    test_stream_refuses_per_axis_padding();
    test_tiled_dgrad_name_carries_per_axis_padding();
    test_tiled_dgrad_refuses_cpl();
    test_wmma_kernels_refuse_per_axis_padding();
    test_stem_wmma_kernels_refuse_per_axis_padding();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_dw_dgrad_host: OK\n");
    return 0;
}
