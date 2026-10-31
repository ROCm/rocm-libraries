// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_dense_f8f6f4_hip.cpp -- C++ HIP lowering of the dense gfx1250
 * 16x16x128 f8f6f4 WMMA atom.
 *
 * Builds the same fp8 GEMM as tests/core/test_dense_f8f6f4_hip.py (a 64x64x128
 * tile on the dense atom, no TDM) through the C++ engine, lowers it to HIP, and
 * checks the builtin call the Python test pins: the 6-operand form with a FP8
 * format immediate (0) per matrix and an i16 modC of 0. The LLVM side of the
 * atom is pinned byte-for-byte by the gemm parity configs; this covers the HIP
 * backend, which that gate does not reach.
 */

#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include <cstdio>
#include <cstring>

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

/* The first dense WMMA call, as the Python test pins it. */
static const char* const k_expected_hip = "= __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(0, ";
static const char* const k_expected_tail = ", (int16_t)0, acc_m0_n0);";

static rocke_gemm_universal_spec_t dense_spec()
{
    rocke_gemm_universal_spec_t spec = rocke_gemm_universal_spec_default();
    spec.name = "dense_f8f6f4_hip";
    spec.tile = (rocke_gemm_tile_spec_t){.tile_m = 64,
                                         .tile_n = 64,
                                         .tile_k = 128,
                                         .warp_m = 2,
                                         .warp_n = 2,
                                         .warp_k = 1,
                                         .warp_tile_m = 16,
                                         .warp_tile_n = 16,
                                         .warp_tile_k = 128};
    spec.trait.pipeline = "mem";
    spec.trait.epilogue = "default";
    spec.data.dtype_a = "fp8e4m3";
    spec.data.dtype_b = "fp8e4m3";
    spec.data.dtype_c = "bf16";
    spec.data.dtype_acc = "fp32";
    spec.data.layout = "RCR";
    spec.wave_size = 32;
    spec.block_size = 128;
    spec.batched = false;
    return spec;
}

static int test_dense_atom_lowers_to_hip()
{
    rocke_gemm_universal_spec_t spec = dense_spec();
    char reason[512] = {0};
    CHECK(rocke_gemm_universal_is_valid_spec(&spec, "gfx1250", reason, sizeof(reason)));

    rocke_ir_builder_t b;
    CHECK(rocke_build_universal_gemm_new(&b, &spec, "gfx1250") != NULL);
    CHECK(rocke_ir_builder_ok(&b));

    rocke_strbuf_t hip;
    CHECK(rocke_strbuf_init(&hip, 0) == 0);
    rocke_lower_hip_opts_t opts = {};
    opts.arch = "gfx1250";
    opts.include_prologue = false;
    opts.include_prologue_set = true;
    const rocke_status_t st = rocke_lower_kernel_to_hip(&b, b.kernel, &opts, &hip);
    if(st != ROCKE_OK)
        std::fprintf(stderr, "HIP lowering failed: %s\n", rocke_ir_builder_error(&b));
    CHECK(st == ROCKE_OK);

    const char* call = std::strstr(hip.data, k_expected_hip);
    if(call == NULL)
        std::fprintf(stderr, "HIP text:\n%s\n", hip.data);
    CHECK(call != NULL);
    const char* eol = std::strchr(call, '\n');
    CHECK(eol != NULL);
    const std::size_t len = (std::size_t)(eol - call);
    const std::size_t tail = std::strlen(k_expected_tail);
    CHECK(len > tail && std::memcmp(eol - tail, k_expected_tail, tail) == 0);

    rocke_strbuf_free(&hip);
    rocke_ir_builder_free(&b);
    return 0;
}

int main()
{
    if(test_dense_atom_lowers_to_hip() != 0)
        return 1;
    std::printf("dense f8f6f4 HIP lowering OK\n");
    return 0;
}
