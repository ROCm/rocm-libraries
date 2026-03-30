// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Canonical GEMM variant table — single source of truth for all kernel specs.
// Shared between device (.hip) and host (main.cpp) compilation.
//
// Device code uses gemm_variant_spec("name") to look up a spec by name at
// compile time. Host code iterates gemm_variants[] for registry/dispatch.

#pragma once

#include <rocm_ck/gemm_spec.hpp>

#include <string_view>

namespace rocm_ck {

struct GemmVariant
{
    const char* name;
    GemmSpec spec;
};

consteval GemmVariant make_variant(const char* name, Signature sig, GemmAlgorithm algo)
{
    return {name, make_spec(sig, algo)};
}

inline constexpr GemmVariant gemm_variants[] = {
    make_variant("gemm_fp32",
                 Signature{
                     .dtype = DataType::FP32,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_fp16",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_bf16",
                 Signature{
                     .dtype = DataType::BF16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_fp16_w32",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {32, 32, 16},
                 }),
    make_variant("gemm_fp16_add",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                               AddOp{.lhs = "C", .rhs = "bias", .out = "D"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_fp16_add_relu",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                               AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                               ReluOp{.in = "D", .out = "E"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    // --- Layout variants: A×B layout combinations beyond the R×C default ---
    make_variant("gemm_fp16_rr",
                 Signature{
                     .dtype   = DataType::FP16,
                     .tensors = {Tensor{.name = "B", .layout = Layout::Row}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_fp16_cr",
                 Signature{
                     .dtype   = DataType::FP16,
                     .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                 Tensor{.name = "B", .layout = Layout::Row}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_fp16_cc",
                 Signature{
                     .dtype   = DataType::FP16,
                     .tensors = {Tensor{.name = "A", .layout = Layout::Col},
                                 Tensor{.name = "B", .layout = Layout::Col}},
                     .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    // --- Split-K: partition K dimension across blockIdx.z ---
    make_variant("gemm_fp16_splitk",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                     .k_batch     = 4,
                 }),
    // --- Pipeline V3: compute-optimized pipeline ---
    make_variant("gemm_fp16_v3",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                     .pipeline    = Pipeline::V3,
                 }),
    // --- Multi-D: two D tensors (Add+Add: result = A*B + D0 + D1) ---
    make_variant("gemm_fp16_add_add",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                               AddOp{.lhs = "C", .rhs = "bias0", .out = "D"},
                               AddOp{.lhs = "D", .rhs = "bias1", .out = "E"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    // --- Batched GEMM: batch dimension via blockIdx.y (runtime, same spec as unbatched) ---
    make_variant("gemm_fp16_batched",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    // --- Architecture-adaptive: per-arch tile configs (separate variants) ---
    make_variant("gemm_fp16_gfx90a",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                 }),
    make_variant("gemm_fp16_gfx942",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {256, 256, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {32, 32, 16},
                 }),
    // --- Preshuffle: B matrix pre-rearranged for optimal LDS loads ---
    make_variant("gemm_fp16_preshuffle",
                 Signature{
                     .dtype = DataType::FP16,
                     .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}},
                 },
                 GemmAlgorithm{
                     .block_tile  = {128, 128, 32},
                     .block_waves = {2, 2, 1},
                     .mfma_tile   = {16, 16, 16},
                     .pipeline    = Pipeline::Preshuffle,
                 }),
};

inline constexpr int gemm_variant_count = sizeof(gemm_variants) / sizeof(gemm_variants[0]);

/// Compile-time variant lookup by name. Typos cause compile errors.
consteval GemmSpec gemm_variant_spec(const char* name)
{
    for(const auto& v : gemm_variants)
        if(std::string_view(v.name) == name)
            return v.spec;
    throw "unknown GEMM variant name";
}

} // namespace rocm_ck
