// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/ir_serialize_roundtrip.c -- round-trip + verifier test for the C
 * ck.dsl.ir/v1 serializer/parser.
 *
 * For each buildable universal-GEMM config it asserts:
 *
 *   1. Round-trip lowering: lower(parse(serialize(k))) produces .ll that is
 *      byte-identical to lower(k). This is the strongest faithfulness gate:
 *      the parsed-back IR must lower to exactly the same code.
 *   2. Serialization idempotence: serialize(parse(serialize(k))) ==
 *      serialize(k), byte-for-byte.
 *   3. Verifier: rocke_verify(parse(serialize(k))) reports zero ERROR
 *      diagnostics (a real, well-formed kernel verifies clean).
 *
 * Exit 0 = all pass. Prints a per-config verdict.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static int make_spec(int idx, rocke_gemm_universal_spec_t* spec)
{
    *spec = rocke_gemm_universal_spec_default();
    switch(idx)
    {
    case 0:
        spec->name = "test1";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 128,
                                              .tile_k = 32,
                                              .warp_m = 2,
                                              .warp_n = 2,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->data.dtype_b = "fp16";
        spec->data.dtype_c = "fp16";
        spec->data.dtype_acc = "fp32";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 3:
        spec->name = "test4";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 128,
                                              .tile_n = 256,
                                              .tile_k = 64,
                                              .warp_m = 4,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 32};
        spec->trait.pipeline = "mem";
        spec->trait.epilogue = "default";
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 256;
        spec->batched = false;
        break;
    case 4:
        spec->name = "test5";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                              .tile_n = 64,
                                              .tile_k = 64,
                                              .warp_m = 1,
                                              .warp_n = 1,
                                              .warp_k = 1,
                                              .warp_tile_m = 16,
                                              .warp_tile_n = 16,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv3";
        spec->trait.epilogue = "cshuffle";
        spec->trait.chiplet_swizzle = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 64;
        spec->batched = false;
        break;
    case 5:
        spec->name = "test6";
        spec->tile = (rocke_gemm_tile_spec_t){.tile_m = 256,
                                              .tile_n = 256,
                                              .tile_k = 128,
                                              .warp_m = 4,
                                              .warp_n = 4,
                                              .warp_k = 1,
                                              .warp_tile_m = 32,
                                              .warp_tile_n = 32,
                                              .warp_tile_k = 16};
        spec->trait.pipeline = "compv4";
        spec->trait.epilogue = "default";
        spec->trait.direct_to_lds = true;
        spec->data.dtype_a = "fp16";
        spec->wave_size = 64;
        spec->block_size = 1024;
        spec->batched = true;
        break;
    default:
        return -1;
    }
    rocke_gemm_universal_spec_finalize(spec);
    return 0;
}

static int run_one(int idx)
{
    rocke_gemm_universal_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        return 0; /* skip non-buildable indices */
    }

    /* Build the original kernel. rocke_build_universal_gemm_new is the
     * init-from-spec wrapper: it calls rocke_ir_builder_init on b0 itself, so the
     * test must NOT pre-init b0 (doing so would leak the first arena head block
     * when the wrapper re-inits -- caught by LeakSanitizer). */
    rocke_ir_builder_t b0;
    rocke_kernel_def_t* k0 = rocke_build_universal_gemm_new(&b0, &spec, "gfx950");
    if(!k0 || !rocke_ir_builder_ok(&b0))
    {
        fprintf(stderr, "[%d] build failed: %s\n", idx, rocke_ir_builder_error(&b0));
        rocke_ir_builder_free(&b0);
        return 1;
    }

    /* serialize(k0) */
    char* s0 = NULL;
    if(rocke_ir_serialize(k0, &s0) != ROCKE_OK || !s0)
    {
        fprintf(stderr, "[%d] serialize failed\n", idx);
        rocke_ir_builder_free(&b0);
        return 1;
    }

    /* lower(k0) */
    char* ll0 = NULL;
    if(rocke_lower_kernel_to_llvm(k0, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &ll0) != ROCKE_OK || !ll0)
    {
        fprintf(stderr, "[%d] lower(original) failed\n", idx);
        free(s0);
        rocke_ir_builder_free(&b0);
        return 1;
    }

    /* parse(serialize(k0)) into a fresh builder */
    rocke_ir_builder_t b1;
    if(rocke_ir_builder_init(&b1, "parsed") != ROCKE_OK)
    {
        fprintf(stderr, "[%d] parse builder init failed\n", idx);
        free(s0);
        free(ll0);
        rocke_ir_builder_free(&b0);
        return 1;
    }
    rocke_kernel_def_t* k1 = NULL;
    if(rocke_ir_parse(s0, &b1, &k1) != ROCKE_OK || !k1)
    {
        fprintf(stderr, "[%d] parse failed: %s\n", idx, rocke_ir_builder_error(&b1));
        free(s0);
        free(ll0);
        rocke_ir_builder_free(&b0);
        rocke_ir_builder_free(&b1);
        return 1;
    }

    int rc = 0;

    /* (2) idempotence: serialize(parse(serialize(k0))) == serialize(k0) */
    char* s1 = NULL;
    if(rocke_ir_serialize(k1, &s1) != ROCKE_OK || !s1)
    {
        fprintf(stderr, "[%d] re-serialize failed\n", idx);
        rc = 1;
    }
    else if(strcmp(s0, s1) != 0)
    {
        fprintf(stderr, "[%d] FAIL: serialization not idempotent\n", idx);
        rc = 1;
    }

    /* (1) round-trip lowering: lower(parse(serialize(k0))) == lower(k0) */
    char* ll1 = NULL;
    if(rocke_lower_kernel_to_llvm(k1, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &ll1) != ROCKE_OK || !ll1)
    {
        fprintf(stderr, "[%d] lower(parsed) failed\n", idx);
        rc = 1;
    }
    else if(strcmp(ll0, ll1) != 0)
    {
        fprintf(stderr, "[%d] FAIL: round-trip .ll differs from original\n", idx);
        rc = 1;
    }

    /* (3) verifier on the parsed kernel: zero ERROR diagnostics. */
    rocke_diag_t* diags = NULL;
    size_t ndiags = 0;
    if(rocke_verify(k1, &diags, &ndiags) != ROCKE_OK)
    {
        fprintf(stderr, "[%d] verify failed (OOM)\n", idx);
        rc = 1;
    }
    else
    {
        size_t nerr = 0;
        for(size_t i = 0; i < ndiags; ++i)
        {
            if(diags[i].severity == ROCKE_DIAG_ERROR)
            {
                ++nerr;
                char* s = rocke_diag_to_string(&diags[i]);
                fprintf(stderr, "[%d] verify ERROR: %s\n", idx, s ? s : "?");
                free(s);
            }
        }
        if(nerr > 0)
        {
            rc = 1;
        }
    }
    rocke_diags_free(diags, ndiags);

    if(rc == 0)
    {
        printf("PASS  [%d] %s  roundtrip+idempotent+verify-clean (.ll %zu bytes)\n",
               idx,
               spec.name,
               strlen(ll0));
    }

    free(s0);
    free(s1);
    free(ll0);
    free(ll1);
    rocke_ir_builder_free(&b0);
    rocke_ir_builder_free(&b1);
    return rc;
}

int main(void)
{
    int rc = 0;
    int ran = 0;
    for(int i = 0; i < 7; ++i)
    {
        rocke_gemm_universal_spec_t s;
        if(make_spec(i, &s) != 0)
        {
            continue;
        }
        ran++;
        rc |= run_one(i);
    }
    if(rc == 0)
    {
        printf(">> ALL ROUND-TRIP / VERIFY CHECKS PASSED (%d configs)\n", ran);
    }
    else
    {
        printf(">> ROUND-TRIP / VERIFY FAILURES PRESENT\n");
    }
    return rc;
}
