// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only spec-validation test for the C++ implicit-GEMM dgrad gates of the
 * gfx950 knobs: the DMA-staged two-stage K loop (dtl), the register-prefetch
 * uniform_k_decode loop (prefetch_global_regs), chiplet_swizzle on the patch
 * GEMM, the 16-element A row pad, tile_k 128 and patch split-K through the
 * cshuffle epilogue.
 *
 * CI checks the parity configs of conv_implicit_gemm_dgrad_emit with the
 * Python engine only (the canonical golden), so without this test the C++
 * verdicts on these specs run in no CI lane. It takes the specs from the C
 * parity emitter itself (configs 71..84), so the two cannot drift, and checks
 * that rocke_dgrad_conv_is_valid_spec gives the verdict and the reason text
 * Python's is_valid_dgrad_spec gives, that an accepted spec builds and lowers,
 * and that the builder refuses a rejected one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reuse the parity emitter's spec table (make_cfg) without its main(). */
#define main conv_implicit_gemm_dgrad_emit_main
#include "conv_implicit_gemm_dgrad_emit.c"
#undef main

namespace
{

int g_failures = 0;

struct Case
{
    int idx;
    const char* reason; /* Python's is_valid_dgrad_spec reason; nullptr => accepted */
};

void run(const Case& c)
{
    rocke_dgrad_conv_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(c.idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "FAIL config %d: not in the parity emitter\n", c.idx);
        ++g_failures;
        return;
    }

    char reason[512] = {0};
    const bool ok = rocke_dgrad_conv_is_valid_spec(&spec, arch, reason, sizeof(reason));

    rocke_ir_builder_t b;
    const bool built = rocke_build_implicit_gemm_conv_dgrad_new(&b, &spec, arch) != NULL;
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
        const rocke_status_t st = rocke_dgrad_conv_implicit_gemm_lower_to_llvm(
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
    const Case cases[] = {
        /* 71..79: accepted (dtl, chiplet, pgr, A row pad 16, tile_k 128, patch
         * split-K, the zero-residue fill shape) */
        {71, nullptr},
        {72, nullptr},
        {73, nullptr},
        {74, nullptr},
        {75, nullptr},
        {76, nullptr},
        {77, nullptr},
        {78, nullptr},
        {79, nullptr},
        {80, "dtl is built on the K-outer patch GEMM only"},
        {81, "prefetch_global_regs needs the uniform_k_decode loop"},
        {82, "patch with split_k > 1 needs the cshuffle epilogue"},
        {83,
         "dgrad does not implement chiplet_swizzle (the builder has no such path; the flag "
         "would be silently ignored)"},
        /* is_valid_dgrad_spec checks pgr before dtl (validate() the reverse) */
        {84, "prefetch_global_regs needs the uniform_k_decode loop"},
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
