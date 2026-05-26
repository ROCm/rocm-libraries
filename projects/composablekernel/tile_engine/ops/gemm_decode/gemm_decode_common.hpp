// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

// Common helpers and problem descriptor for the gemm_decode tile_engine
// benchmarks. Mirrors the role of tile_engine/ops/gemm/gemm_common.hpp for
// the universal GEMM family. The full Python codegen / per-instance build
// pipeline lands in P1+ once more than one tile config is needed; for P0 the
// benchmarks include this header directly and pick the active kernel via
// `gemm_decode_universal_single_*.hpp`.

namespace gemm_decode_tile_engine {

struct DecodeProblem
{
    ck_tile::index_t M       = 1;
    ck_tile::index_t N       = 8192;
    ck_tile::index_t K       = 7168;
    ck_tile::index_t stride_a = 0; // 0 = use K
    ck_tile::index_t stride_b = 0; // 0 = use K
    ck_tile::index_t stride_c = 0; // 0 = use N
    ck_tile::index_t k_batch  = 1;
};

inline ck_tile::ArgParser create_decode_arg_parser()
{
    ck_tile::ArgParser parser;
    parser.insert("m", "1", "M dimension (decode batch)");
    parser.insert("n", "8192", "N dimension (output features)");
    parser.insert("k", "7168", "K dimension (reduction)");
    parser.insert("stride_a", "0", "row stride of A in elements (0 = K)");
    parser.insert("stride_b", "0", "row stride of B in elements (0 = K)");
    parser.insert("stride_c", "0", "row stride of C in elements (0 = N)");
    parser.insert("split_k", "1", "k_batch for AtomicAdd split-K");
    parser.insert("warmup", "20", "warmup iterations");
    parser.insert("repeat", "100", "benchmark iterations");
    parser.insert("verify", "0", "0 disables, 1 runs FP32 host reference");
    parser.insert("init", "0", "0 random, 1 monotonic, 2 constant 1.0");
    parser.insert("metric", "0", "0 ms, 1 TFLOPS, 2 GB/s");
    return parser;
}

template <typename ADataType, typename BDataType>
double DecodeFlops(const DecodeProblem& p)
{
    return 2.0 * static_cast<double>(p.M) * static_cast<double>(p.N) * static_cast<double>(p.K);
}

template <typename ADataType, typename BDataType, typename CDataType>
double DecodeBytes(const DecodeProblem& p)
{
    const double a = static_cast<double>(p.M) * static_cast<double>(p.K) * sizeof(ADataType);
    const double b = static_cast<double>(p.N) * static_cast<double>(p.K) * sizeof(BDataType);
    const double c = static_cast<double>(p.M) * static_cast<double>(p.N) * sizeof(CDataType);
    return a + b + c;
}

} // namespace gemm_decode_tile_engine
