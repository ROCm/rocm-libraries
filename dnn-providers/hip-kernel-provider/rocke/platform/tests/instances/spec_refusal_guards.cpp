// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only check of two spec refusals in the C++ engine, with their reason
 * strings, so they are under CI without the pybind rocke_engine:
 *
 *  - pooling refuses bf16: its output store goes through store_tile, which has
 *    a buffer store for f16 only;
 *  - multi-D GEMM refuses a D dtype other than the C dtype: the epilogue
 *    combines each D with the C tile in C's element type;
 *  - MatMulNBits refuses its optimized body off gfx1201, whose WMMA layout
 *    that body is written for.
 *
 * Each reason is the one the Python is_valid_spec returns for the same spec
 * (tests/instances/test_pooling_numeric.py, test_gemm_multi_d_numeric.py and
 * test_matmul_nbits_numeric.py check those). The byte-identity gate only compares what both engines build,
 * so it cannot see an accept/reject drift.
 */
#include <stdio.h>
#include <string.h>

#include "rocke/helper_rocke.instances.common.gemm_multi_d.h"
#include "rocke/instance_matmul_nbits.h"
#include "rocke/instance_pooling.h"

namespace
{

int g_failures = 0;

void expect(const char* label, bool ok, const char* reason, bool want_ok, const char* want_reason)
{
    if(ok != want_ok || strcmp(reason, want_reason) != 0)
    {
        fprintf(stderr,
                "FAIL %s: got (%s, \"%s\"), want (%s, \"%s\")\n",
                label,
                ok ? "true" : "false",
                reason,
                want_ok ? "true" : "false",
                want_reason);
        ++g_failures;
    }
}

rocke_pooling2d_spec_t pooling_spec(const char* dtype)
{
    rocke_pooling2d_spec_t spec = rocke_pooling2d_spec_default();
    spec.problem.N = 1;
    spec.problem.H = 4;
    spec.problem.W = 4;
    spec.problem.C = 4;
    spec.problem.Y = 2;
    spec.problem.X = 2;
    spec.problem.sH = 2;
    spec.problem.sW = 2;
    spec.block_size = 64;
    spec.dtype = dtype;
    return spec;
}

rocke_gemm_multi_d_spec_t multi_d_spec(const char* c_dtype, const char* d_dtype)
{
    rocke_gemm_multi_d_spec_t spec = rocke_gemm_multi_d_spec_default();
    spec.base.trait.epilogue = "cshuffle";
    spec.base.data.dtype_a = c_dtype;
    spec.base.data.dtype_b = c_dtype;
    spec.base.data.dtype_c = c_dtype;
    spec.d_operands[0].param_name = "D0";
    spec.d_operands[0].op_is_mul = false;
    spec.num_d_operands = 1;
    spec.d_dtype = d_dtype;
    return spec;
}

} // namespace

int main()
{
    char reason[256];

    rocke_pooling2d_spec_t f16 = pooling_spec("f16");
    bool ok = rocke_pooling2d_is_valid_spec(&f16, "gfx942", reason, sizeof(reason));
    expect("pooling f16", ok, reason, true, "ok");

    rocke_pooling2d_spec_t bf16 = pooling_spec("bf16");
    ok = rocke_pooling2d_is_valid_spec(&bf16, "gfx942", reason, sizeof(reason));
    expect("pooling bf16",
           ok,
           reason,
           false,
           "bf16 output is not supported (the store is f16 only); use f16");

    rocke_gemm_multi_d_spec_t mixed = multi_d_spec("fp16", "bf16");
    ok = rocke_gemm_multi_d_is_valid_spec(&mixed, "gfx942", reason, sizeof(reason));
    expect("multi-D fp16 C, bf16 D",
           ok,
           reason,
           false,
           "d_dtype 'bf16' must be the C dtype 'fp16': each D combines with C in C's type");

    rocke_gemm_multi_d_spec_t reversed = multi_d_spec("bf16", "fp16");
    ok = rocke_gemm_multi_d_is_valid_spec(&reversed, "gfx942", reason, sizeof(reason));
    expect("multi-D bf16 C, fp16 D",
           ok,
           reason,
           false,
           "d_dtype 'fp16' must be the C dtype 'bf16': each D combines with C in C's type");

    // An alias of the C dtype is the same type, so the dtype rule passes it on
    // to the remaining checks.
    rocke_gemm_multi_d_spec_t alias = multi_d_spec("fp16", "f16");
    ok = rocke_gemm_multi_d_is_valid_spec(&alias, "gfx942", reason, sizeof(reason));
    if(strstr(reason, "must be the C dtype") != NULL)
    {
        fprintf(stderr, "FAIL multi-D alias: refused as a dtype mismatch: %s\n", reason);
        ++g_failures;
    }

    rocke_matmul_nbits_spec_t nbits = rocke_matmul_nbits_spec_default();
    nbits.N = 4096;
    nbits.K = 4096;
    nbits.tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                          .tile_n = 128,
                                          .tile_k = 32,
                                          .warp_m = 2,
                                          .warp_n = 2,
                                          .warp_k = 1,
                                          .warp_tile_m = 16,
                                          .warp_tile_n = 16,
                                          .warp_tile_k = 16};
    nbits.group_size = 32;
    nbits.scale_dtype = "fp16";
    nbits.family = "large_n";
    nbits.optimized = true;
    rocke_matmul_nbits_spec_finalize(&nbits);
    ok = rocke_matmul_nbits_is_valid_spec(&nbits, "gfx1201", reason, sizeof(reason));
    expect("MatMulNBits optimized on gfx1201", ok, reason, true, "ok");
    ok = rocke_matmul_nbits_is_valid_spec(&nbits, "gfx1151", reason, sizeof(reason));
    expect("MatMulNBits optimized on gfx1151",
           ok,
           reason,
           false,
           "optimized=True is built for gfx1201 only; got gfx1151");

    if(g_failures != 0)
    {
        fprintf(stderr, "%d case(s) failed\n", g_failures);
        return 1;
    }
    printf("spec refusal guards: all cases passed\n");
    return 0;
}
