// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

// Shared by the comp_tdm_v4 GPU tests: instance aliases, rocKE's sparse check inputs, a device
// run and the bitwise check against reference_mx_gemm_tdm_v4.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/mx_scale_kmajor.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace mx_tdm_v4_test {

using ck_tile::index_t;
using ck_tile::long_index_t;
using ck_tile::MxTdmV4Dtype;
using ck_tile::MxTdmV4OutDtype;
namespace v4 = ck_tile::mx_tdm_v4;

/// A v4 kernel: A, B and D types, tile, wave group, WMMA atom (32 or 16 rows), tuning and the
/// shape traits (edge M/N, wide A/B/D, odd tiles, tail steps, store width).
template <typename A,
          typename B,
          typename O,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t AtomM,
          typename Tuning,
          typename Traits>
using Pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV4<
    ck_tile::MxGemmTdmV4Problem<A,
                                B,
                                O,
                                ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                                       ck_tile::sequence<WarpM, WarpN, 1>,
                                                       ck_tile::sequence<AtomM, 16, 128>>,
                                Traits>,
    ck_tile::GemmPipelineAgBgCrCompTDMV4Policy<Tuning>>;

/// The kernel of a v4 pipeline: its N-major partitioner and its alpha / beta epilogue.
template <typename P>
using KernelOf =
    ck_tile::MxGemmKernel<ck_tile::GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>,
                          P,
                          ck_tile::MxAlphaBetaEpilogue<ck_tile::MxAlphaBetaEpilogueProblem<P>>>;

template <typename A,
          typename B,
          typename O,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t AtomM,
          typename Tuning,
          typename Traits>
using Kernel =
    KernelOf<Pipeline<A, B, O, TileM, TileN, TileK, WarpM, WarpN, AtomM, Tuning, Traits>>;

/// rocKE's tuning knobs with its defaults after the cluster: GL2 distance, signal WMMAs (-1 for
/// the default), local prefetch, half PLR, half-B order, segment interleave.
template <index_t CX,
          index_t CY,
          index_t Gl2                      = 4,
          index_t Signal                   = -1,
          index_t Lp                       = 1,
          ck_tile::MxTdmV4HalfPlr Plr      = ck_tile::MxTdmV4HalfPlr::B,
          ck_tile::MxTdmV4HalfBOrder Order = ck_tile::MxTdmV4HalfBOrder::Ahead,
          bool Lsi                         = true>
using Tu = ck_tile::MxTdmV4TuningT<CX, CY, Gl2, Signal, Lp, Plr, Order, Lsi>;

template <bool EdgeM, bool EdgeN, index_t OddTiles, index_t TailSteps, index_t StoreWidth>
using Shape =
    ck_tile::MxTdmV4GemmTraits<EdgeM, EdgeN, false, false, false, OddTiles, TailSteps, StoreWidth>;

/// rocKE's sparse check (mx_gemm_bench.sparse_inputs): A and B nonzero only in a few 128-K steps
/// (the first and last, both sides of the first and last tile boundaries, then random ones up to
/// 12), FP4 all 16 codes with E8M0 2^-2..2^1, 8-bit +-(1 + m / 2^bits) * 2^e, e in {-1, 0}, with
/// E8M0 2^-1..2^0. Every FP32 partial sum is exact, so D is checked bit for bit at any K.
struct Inputs
{
    std::vector<uint8_t> a;       // M rows of row_bytes(a_dtype, K)
    std::vector<uint8_t> b;       // N rows
    std::vector<uint8_t> a_scale; // K-major
    std::vector<uint8_t> b_scale; // K-major
    std::vector<uint8_t> c;       // M x N in the output type
};

inline std::vector<long_index_t> sparse_steps(long_index_t K, index_t tile_k, std::mt19937& rng)
{
    const long_index_t steps     = (K + 127) / 128;
    const long_index_t per_tile  = tile_k / 128;
    const long_index_t last_tile = (steps - 1) / per_tile * per_tile;
    std::set<long_index_t> chosen;
    for(long_index_t s :
        {long_index_t{0}, steps - 1, per_tile - 1, per_tile, last_tile - 1, last_tile})
        if(s >= 0 && s < steps)
            chosen.insert(s);
    std::vector<long_index_t> rest;
    for(long_index_t s = 0; s < steps; ++s)
        if(!chosen.count(s))
            rest.push_back(s);
    std::shuffle(rest.begin(), rest.end(), rng);
    for(std::size_t i = 0; chosen.size() < 12 && i < rest.size(); ++i)
        chosen.insert(rest[i]);
    return {chosen.begin(), chosen.end()};
}

inline uint8_t sparse_code(MxTdmV4Dtype t, std::mt19937& rng)
{
    if(t == MxTdmV4Dtype::Fp4)
        return static_cast<uint8_t>(rng() % 16);
    const int bits     = t == MxTdmV4Dtype::Fp8 ? 3 : 2;
    const int bias     = t == MxTdmV4Dtype::Fp8 ? 7 : 15;
    const int mantissa = static_cast<int>(rng() % (1u << bits));
    const int exponent = bias - static_cast<int>(rng() % 2); // 2^0 or 2^-1
    const int sign     = static_cast<int>(rng() % 2);
    return static_cast<uint8_t>((sign << 7) | (exponent << bits) | mantissa);
}

inline Inputs make_inputs(MxTdmV4Dtype a_dtype,
                          MxTdmV4Dtype b_dtype,
                          MxTdmV4OutDtype out,
                          index_t M,
                          index_t N,
                          long_index_t K,
                          index_t tile_k,
                          uint32_t seed = 0x5BA6)
{
    std::mt19937 rng(seed);
    const auto steps = sparse_steps(K, tile_k, rng);
    Inputs in;
    auto operand = [&](MxTdmV4Dtype t, index_t rows) {
        const long_index_t row_bytes = v4::row_bytes(t, K);
        std::vector<uint8_t> data(static_cast<std::size_t>(rows) * row_bytes, 0);
        for(index_t r = 0; r < rows; ++r)
            for(long_index_t s : steps)
                for(long_index_t k = s * 128; k < std::min(K, (s + 1) * 128); ++k)
                {
                    const uint8_t code = sparse_code(t, rng);
                    uint8_t* row       = data.data() + r * row_bytes;
                    if(t == MxTdmV4Dtype::Fp4)
                        row[k / 2] |= static_cast<uint8_t>(k % 2 ? code << 4 : code);
                    else
                        row[k] = code;
                }
        return data;
    };
    in.a                      = operand(a_dtype, M);
    in.b                      = operand(b_dtype, N);
    const long_index_t groups = (K + 31) / 32;
    const bool fp4            = a_dtype == MxTdmV4Dtype::Fp4;
    auto scales               = [&](index_t rows) {
        std::vector<uint8_t> natural(static_cast<std::size_t>(rows) * groups);
        for(auto& x : natural)
            x = static_cast<uint8_t>(fp4 ? 125 + rng() % 4 : 126 + rng() % 2);
        return ck_tile::mx_scale_to_kmajor(natural.data(), rows, K, groups);
    };
    in.a_scale = scales(M);
    in.b_scale = scales(N);
    in.c.resize(static_cast<std::size_t>(M) * N * v4::out_bytes(out));
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    for(std::size_t i = 0; i < static_cast<std::size_t>(M) * N; ++i)
        v4::store_out(out, in.c.data(), i, uniform(rng));
    return in;
}

/// rocKE's alpha for the sparse check.
inline constexpr float kSparseAlpha = 1.0f / 64;

/// The packed host arguments of a v4 run (k_batch 1, strides K for A and B, N for C and D).
template <typename Kernel>
typename Kernel::HostArgs host_args(const void* a,
                                    const void* b,
                                    const void* a_scale,
                                    const void* b_scale,
                                    const void* c,
                                    void* d,
                                    index_t M,
                                    index_t N,
                                    index_t K,
                                    float alpha,
                                    float beta)
{
    return typename Kernel::HostArgs{
        {a}, {a_scale}, {b}, {b_scale}, {c}, d, 1, M, N, K, {K}, {K}, {N}, N, alpha, beta};
}

/// Launches Kernel: the grid of GridSize, and with a cluster tuning the cluster of ClusterSize.
template <typename Kernel>
float launch(const ck_tile::stream_config& s, const typename Kernel::KernelArgs& kargs)
{
    if(!Kernel::IsSupportedArgument(kargs))
        throw std::runtime_error(Kernel::GetName() + ": " + Kernel::GetUnsupportedReason(kargs));
    const dim3 grids  = Kernel::GridSize(kargs.M, kargs.N, kargs.k_batch);
    const dim3 blocks = Kernel::BlockSize();
    if constexpr(Kernel::ClusterLaunch)
    {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(Kernel{}, Kernel::ClusterSize(), grids, blocks, 0, kargs));
#else
        throw std::runtime_error(Kernel::GetName() + ": this build has no cluster launch");
#endif
    }
    else
    {
        return ck_tile::launch_kernel(s,
                                      ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
    }
}

template <typename Kernel>
std::vector<uint8_t> run_device(
    const Inputs& in, index_t M, index_t N, index_t K, float alpha, float beta, bool pass_c = true)
{
#ifdef MX_TDM_V4_DRY_RUN
    // CPU-only build: check that the instance runs the problem, launch nothing.
    const auto probe = host_args<Kernel>(in.a.data(),
                                         in.b.data(),
                                         in.a_scale.data(),
                                         in.b_scale.data(),
                                         pass_c ? in.c.data() : nullptr,
                                         const_cast<uint8_t*>(in.c.data()),
                                         M,
                                         N,
                                         K,
                                         alpha,
                                         beta);
    const char* why  = Kernel::GetUnsupportedReason(probe);
    EXPECT_EQ(why, nullptr) << Kernel::GetName() << ": " << why;
    return {};
#else
    ck_tile::DeviceMem a(in.a.size()), b(in.b.size()), as(in.a_scale.size()), bs(in.b_scale.size()),
        c(in.c.size()), d(in.c.size());
    a.ToDevice(in.a.data());
    b.ToDevice(in.b.data());
    as.ToDevice(in.a_scale.data());
    bs.ToDevice(in.b_scale.data());
    c.ToDevice(in.c.data());
    d.SetZero();
    const auto args =
        Kernel::MakeKernelArgs(host_args<Kernel>(a.GetDeviceBuffer(),
                                                 b.GetDeviceBuffer(),
                                                 as.GetDeviceBuffer(),
                                                 bs.GetDeviceBuffer(),
                                                 pass_c ? c.GetDeviceBuffer() : nullptr,
                                                 d.GetDeviceBuffer(),
                                                 M,
                                                 N,
                                                 K,
                                                 alpha,
                                                 beta));
    launch<Kernel>(ck_tile::stream_config{nullptr, false}, args);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<uint8_t> out(in.c.size());
    d.FromDevice(out.data());
    return out;
#endif
}

inline std::vector<uint8_t> run_reference(MxTdmV4Dtype a_dtype,
                                          MxTdmV4Dtype b_dtype,
                                          MxTdmV4OutDtype out,
                                          const Inputs& in,
                                          index_t M,
                                          index_t N,
                                          long_index_t K,
                                          float alpha,
                                          float beta,
                                          const std::vector<index_t>* rows = nullptr)
{
    std::vector<uint8_t> d(rows ? rows->size() * N * v4::out_bytes(out) : in.c.size());
    ck_tile::reference_mx_gemm_tdm_v4(a_dtype,
                                      b_dtype,
                                      out,
                                      in.a.data(),
                                      v4::row_bytes(a_dtype, K),
                                      in.b.data(),
                                      v4::row_bytes(b_dtype, K),
                                      in.a_scale.data(),
                                      in.b_scale.data(),
                                      in.c.data(),
                                      N,
                                      d.data(),
                                      N,
                                      M,
                                      N,
                                      K,
                                      alpha,
                                      beta,
                                      0,
                                      rows ? rows->data() : nullptr,
                                      rows ? static_cast<index_t>(rows->size()) : 0);
    return d;
}

/// The first element whose bits differ, or "" if D matches bit for bit (any FP8 NaN code
/// matching any other).
inline std::string first_mismatch(const std::vector<uint8_t>& got,
                                  const std::vector<uint8_t>& want,
                                  MxTdmV4OutDtype out,
                                  index_t N)
{
    const index_t bytes = v4::out_bytes(out);
    // The bits as well as the value: NaNs of different bits all print as nan.
    const auto hex = [&](const std::vector<uint8_t>& d, std::size_t i) {
        uint32_t bits = 0;
        std::memcpy(&bits, d.data() + i * bytes, bytes);
        char text[16];
        std::snprintf(text, sizeof(text), " (0x%0*X)", static_cast<int>(bytes * 2), bits);
        return std::string(text);
    };
    // FP8 NaN is 0x7F or 0xFF; the sign of a NaN out of arithmetic is unspecified (the device
    // gives a NaN read from C sign 1, the reference keeps sign 0), so the two codes match.
    const auto fp8_nan = [](uint8_t b) { return (b & 0x7F) == 0x7F; };
    for(std::size_t i = 0; i < got.size() / bytes; ++i)
        if(std::memcmp(got.data() + i * bytes, want.data() + i * bytes, bytes) != 0 &&
           !(out == MxTdmV4OutDtype::Fp8E4M3 && fp8_nan(got[i]) && fp8_nan(want[i])))
            return "D(" + std::to_string(i / N) + ", " + std::to_string(i % N) + "): got " +
                   std::to_string(v4::load_out(out, got.data(), i)) + hex(got, i) + ", want " +
                   std::to_string(v4::load_out(out, want.data(), i)) + hex(want, i);
    return "";
}

/// The output type of a kernel's D.
template <typename Kernel>
constexpr MxTdmV4OutDtype out_of()
{
    return Kernel::GemmPipeline::kSpec.out;
}

/// Runs `Kernel` on rocKE's sparse inputs at each beta and requires bitwise equality with the
/// reference; beta 0 runs with a NaN-filled C to show C is not read.
template <typename Kernel>
void check(index_t M, index_t N, index_t K, std::initializer_list<float> betas = {0.0f, 0.5f})
{
    constexpr auto s = Kernel::GemmPipeline::kSpec;
    SCOPED_TRACE(Kernel::GetName() + " M" + std::to_string(M) + " N" + std::to_string(N) + " K" +
                 std::to_string(K));
    Inputs in        = make_inputs(s.a_dtype, s.b_dtype, s.out, M, N, K, s.tile_k);
    const auto probe = host_args<Kernel>(in.a.data(),
                                         in.b.data(),
                                         in.a_scale.data(),
                                         in.b_scale.data(),
                                         in.c.data(),
                                         in.c.data(),
                                         M,
                                         N,
                                         K,
                                         kSparseAlpha,
                                         0.5f);
    const char* why  = Kernel::GetUnsupportedReason(probe);
    ASSERT_EQ(why, nullptr) << why;
    for(float beta : betas)
    {
        SCOPED_TRACE("beta " + std::to_string(beta));
        Inputs run = in;
        if(beta == 0.0f)
            for(std::size_t i = 0; i < static_cast<std::size_t>(M) * N; ++i)
                v4::store_out(s.out, run.c.data(), i, std::numeric_limits<float>::quiet_NaN());
        const auto got = run_device<Kernel>(run, M, N, K, kSparseAlpha, beta);
#ifdef MX_TDM_V4_DRY_RUN
        continue;
#endif
        const auto want =
            run_reference(s.a_dtype, s.b_dtype, s.out, run, M, N, K, kSparseAlpha, beta);
        EXPECT_EQ(first_mismatch(got, want, s.out, N), "");
    }
}

} // namespace mx_tdm_v4_test
