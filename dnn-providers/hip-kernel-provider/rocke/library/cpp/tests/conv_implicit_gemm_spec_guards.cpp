// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only spec-validation test for the C++ forward implicit-GEMM gates
 * around the gfx950 MFMA split-K forward: the split-K admission gate, the
 * NULL pipeline / epilogue refusals, the LDS budget that charges compv4 one
 * A/B tile, and the refusal of a split with fp32 output (TF32 on MFMA, plain
 * fp32 on gfx1250 WMMA, two-stage or cluster).
 *
 * CI checks the parity configs of conv_implicit_gemm_emit with the Python
 * engine only (the canonical golden), and the pybind tests that compare the
 * two engines skip where rocke_engine is not built, so without this test the
 * C++ verdicts on these specs run in no CI lane. It takes the specs from the
 * C parity emitter itself (configs 66..85), so the two cannot drift, and
 * checks that rocke_implicit_gemm_conv_is_valid_spec gives the verdict and
 * the reason text Python's is_valid_spec gives, that an accepted spec builds
 * and lowers, and that the builder refuses a rejected one.
 *
 * It also pins the refusals of the four RDNA4 forward knobs (uniform_k_decode,
 * regt_prefetch, lds_swizzle, group_fast_raster): the rejected parity configs
 * (91, 95, 100, 105) and a table of specs derived from the parity emitter's
 * RDNA4 base, each with Python's reason text. The accepted knob configs
 * (86..104) are byte-compared by the parity pair.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reuse the parity emitter's spec table (make_cfg) without its main(). */
#define main conv_implicit_gemm_emit_main
#include "conv_implicit_gemm_emit.c"
#undef main

namespace
{

int g_failures = 0;

struct Case
{
    int idx;
    const char* reason; /* Python's is_valid_spec reason; nullptr => accepted */
};

/* A refused RDNA4 knob spec: the parity emitter's RDNA4 base (gfx1201, 64x64x32
 * WMMA, mem pipeline) with `mutate` applied. Python's reason text is pinned. */
struct Refused
{
    const char* label;
    void (*mutate)(rocke_implicit_gemm_conv_spec_t*, const char**);
    const char* reason;
};

void run_refused(const Refused& r)
{
    rocke_implicit_gemm_conv_spec_t spec;
    const char* arch = "gfx1201";
    rdna4_base(&spec, &arch);
    r.mutate(&spec, &arch);

    char reason[512] = {0};
    const bool ok = rocke_implicit_gemm_conv_is_valid_spec(&spec, arch, reason, sizeof(reason));
    rocke_ir_builder_t b;
    const bool built = rocke_build_implicit_gemm_conv_new(&b, &spec, arch, NULL) != NULL;
    rocke_ir_builder_free(&b);
    if(ok || built || strcmp(reason, r.reason) != 0)
    {
        fprintf(stderr,
                "FAIL refused %s: expected \"%s\"; is_valid_spec=%d, built=%d, got \"%s\"\n",
                r.label,
                r.reason,
                (int)ok,
                (int)built,
                reason);
        ++g_failures;
        return;
    }
    printf("ok   refused %s (%s): %s\n", r.label, arch, reason);
}

void ukd_3d(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->problem
        = rocke_conv_problem_make_3d(2, 8, 8, 8, 64, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    s->uniform_k_decode = true;
}
void ukd_group_merge(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->problem = rocke_conv_problem_make(2, 14, 14, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1);
    s->problem.groups = 32;
    s->tile_n = 32;
    s->warp_n = 1;
    s->vector_size_b = 1;
    s->group_merge = 2;
    s->uniform_k_decode = true;
}
void rtp_two(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->regt_prefetch = 2;
}
void rtp_wavelet(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->regt_prefetch = 1;
    s->pipeline = "wavelet";
}
void rtp_split_k_cluster(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->regt_prefetch = 1;
    s->split_k_cluster = 1;
}
void lxs_tile_k_48(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->lds_swizzle = true;
    s->tile_k = 48;
}
void gfr_wavelet(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->group_fast_raster = true;
    s->pipeline = "wavelet";
}
/* The RDNA4-only knobs on the other targets: gfx11 WMMA, gfx1250 WMMA
 * (16x16x32 atom) and a CDNA MFMA target. */
void ukd_gfx1151(rocke_implicit_gemm_conv_spec_t* s, const char** arch)
{
    *arch = "gfx1151";
    s->uniform_k_decode = true;
}
void lxs_gfx1250(rocke_implicit_gemm_conv_spec_t* s, const char** arch)
{
    *arch = "gfx1250";
    s->warp_tile_k = 32;
    s->lds_swizzle = true;
}
void gfr_gfx950(rocke_implicit_gemm_conv_spec_t* s, const char** arch)
{
    *arch = "gfx950";
    s->problem.groups = 2;
    s->problem.C = 128;
    s->wave_size = 64;
    s->tile_k = 64;
    s->warp_tile_k = 32;
    s->group_fast_raster = true;
}
void rtp_gfx1100(rocke_implicit_gemm_conv_spec_t* s, const char** arch)
{
    *arch = "gfx1100";
    s->regt_prefetch = 1;
}
void gfr_split_k(rocke_implicit_gemm_conv_spec_t* s, const char**)
{
    s->group_fast_raster = true;
    s->split_k = 2;
}

void run(const Case& c)
{
    rocke_implicit_gemm_conv_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(c.idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "FAIL config %d: not in the parity emitter\n", c.idx);
        ++g_failures;
        return;
    }

    char reason[512] = {0};
    const bool ok = rocke_implicit_gemm_conv_is_valid_spec(&spec, arch, reason, sizeof(reason));

    rocke_ir_builder_t b;
    const bool built = rocke_build_implicit_gemm_conv_new(&b, &spec, arch, NULL) != NULL;
    rocke_ir_builder_free(&b);

    if(c.reason == nullptr)
    {
        if(!ok || !built)
        {
            fprintf(stderr,
                    "FAIL config %d: expected accepted; is_valid_spec=%d (%s), built=%d\n",
                    c.idx,
                    (int)ok,
                    reason,
                    (int)built);
            ++g_failures;
            return;
        }
        char* ll = NULL;
        char err[512] = {0};
        const rocke_status_t st = rocke_conv_implicit_gemm_lower_to_llvm(
            &spec, arch, ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof(err));
        if(st != ROCKE_OK || ll == NULL || ll[0] == '\0')
        {
            fprintf(stderr, "FAIL config %d: lowering failed (%d: %s)\n", c.idx, (int)st, err);
            ++g_failures;
        }
        else
        {
            printf("ok   config %d (%s): accepted, lowered\n", c.idx, arch);
        }
        free(ll);
        return;
    }
    if(ok || built)
    {
        fprintf(stderr,
                "FAIL config %d: expected \"%s\"; is_valid_spec=%d, built=%d\n",
                c.idx,
                c.reason,
                (int)ok,
                (int)built);
        ++g_failures;
        return;
    }
    if(strcmp(reason, c.reason) != 0)
    {
        fprintf(stderr, "FAIL config %d: expected \"%s\", got \"%s\"\n", c.idx, c.reason, reason);
        ++g_failures;
        return;
    }
    printf("ok   config %d (%s): %s\n", c.idx, arch, reason);
}

} // namespace

int main()
{
    static const char* const k_no_pipeline = "pipeline must be set (got None)";
    static const char* const k_no_epilogue = "epilogue must be set (got None)";
    static const char* const k_fp32_split
        = "split-K forward reduces through Stage 2: dtype_d must be fp16 or bf16 (got fp32)";
    const Case cases[] = {
        /* 66..73: the MFMA split-K forward (split_k=2) */
        {66, nullptr},
        {67, nullptr},
        {68, nullptr},
        {69, nullptr},
        {70, nullptr},
        {71, "split-K forward on MFMA runs the plain K loop only (mem, compv3, compv4; got basic)"},
        {72, "split-K forward does not support async_dma"},
        {73, nullptr},
        /* 74..81: the NULL pipeline / epilogue refusals and the LDS budget */
        {74, k_no_pipeline},
        {75, k_no_epilogue},
        {76, k_no_pipeline},
        {77, k_no_epilogue},
        {78, nullptr},
        {79, nullptr},
        {80, "LDS budget 196608 bytes (A/B=x2 98304, C=16384) > 163840 cap on gfx950"},
        {81, "LDS budget 196608 bytes (A/B=196608, C=131072) > 163840 cap on gfx950"},
        /* 82..84: a two-stage split with fp32 output: TF32 on gfx950 and
         * gfx942, plain fp32 on gfx1250 WMMA */
        {82, k_fp32_split},
        {83, k_fp32_split},
        {84, k_fp32_split},
        /* 85: config 84 in the cluster form, which folds with the same cast */
        {85, "split-K forward folds through the cluster: dtype_d must be fp16 or bf16 (got fp32)"},
        /* 91, 95, 100, 105: the RDNA4 forward knobs' rejected parity configs */
        {91, "uniform_k_decode does not support the pointwise fast path"},
        {95, "regt_prefetch (fwd) is enabled only on the gfx12 RDNA targets (got gfx12_cdna)"},
        {100, "lds_swizzle sets the LDS layout itself (no lds_layout / lds_k_pad)"},
        {105, "group_fast_raster is enabled only on the gfx12 RDNA targets (got gfx12_cdna)"},
    };
    for(const Case& c : cases)
        run(c);
    const Refused refused[] = {
        {"ukd_3d", ukd_3d, "uniform_k_decode does not support 3-D convolution"},
        {"ukd_group_merge", ukd_group_merge, "uniform_k_decode does not support group_merge"},
        {"rtp2", rtp_two, "regt_prefetch (fwd) must be 0 or 1 (got 2)"},
        {"rtp_wavelet",
         rtp_wavelet,
         "regt_prefetch is the 'mem' pipeline's K loop (got 'wavelet')"},
        {"rtp_split_k_cluster",
         rtp_split_k_cluster,
         "regt_prefetch does not combine with split_k_cluster"},
        {"lxs_tile_k_48", lxs_tile_k_48, "lds_swizzle needs tile_k in 16/32/64/128 (got 48)"},
        {"gfr_wavelet", gfr_wavelet, "group_fast_raster does not support the wavelet pipeline"},
        {"gfr_split_k",
         gfr_split_k,
         "group_fast_raster does not support split-K (z carries the slice)"},
        {"gfr_gfx950",
         gfr_gfx950,
         "group_fast_raster is enabled only on the gfx12 RDNA targets (got gfx950)"},
        {"rtp_gfx1100",
         rtp_gfx1100,
         "regt_prefetch (fwd) is enabled only on the gfx12 RDNA targets (got gfx11_rdna)"},
        {"ukd_gfx1151",
         ukd_gfx1151,
         "uniform_k_decode is enabled only on the gfx12 RDNA targets (got gfx11_rdna)"},
        {"lxs_gfx1250",
         lxs_gfx1250,
         "lds_swizzle is enabled only on the gfx12 RDNA targets (got gfx12_cdna)"},
    };
    for(const Refused& r : refused)
        run_refused(r);
    if(g_failures != 0)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("all cases passed\n");
    return 0;
}
