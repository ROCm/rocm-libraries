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
    };
    for(const Case& c : cases)
        run(c);
    if(g_failures != 0)
    {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("all %zu cases passed\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
}
