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
 *     rotated-filter mode does not implement and must refuse.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_dw_dgrad.h"
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

int main(void)
{
    test_stream_lower_name_overflow();
    test_pad_w_below_minus_one_refused();
    test_stream_refuses_per_axis_padding();
    test_tiled_dgrad_name_carries_per_axis_padding();
    test_tiled_dgrad_refuses_cpl();
    if(g_failures != 0)
    {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("conv_direct_dw_dgrad_host: OK\n");
    return 0;
}
