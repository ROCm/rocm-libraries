// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pin represented inputs to exercise coherent P-rounding error independently of random fills.
// With V=1, O is the real-token probability mass and LSE is computed from the same logits.

#include "ck_tile/host.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

#include <cmath>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

// Built only into the fp8-input binaries that keep a wider output (see CMakeLists). An fp8
// ODataType is excluded on purpose: e4m3 carries 3 mantissa bits, so the output cast alone
// costs up to ~6% and would swamp the ~2% gain this test exists to catch.
using QDataType = FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
using ODataType = FmhaFwdTypeConfig<DataTypeConfig>::ODataType;

// Round-to-nearest-even on a format with t significand bits (stored mantissa plus the
// implicit leading 1) is bounded, relative to the true value, by
//     max |q(x) - x| / x = 2^-t / (1 + 2^-t) = 1 / (2^t + 1)
// reached at the midpoint of the first gap in a binade, where the error is a half ulp while
// the value is already 1 + 2^-t. The bound is scale-invariant: it holds in every normal
// binade, so it does not depend on where the kernel places P.
constexpr double rne_bound(int significand_bits) { return 1.0 / ((1ull << significand_bits) + 1); }

// Two casts stand between the exact attention and the stored output. P goes to e4m3 for
// gemm_1 while l accumulates the unrounded P, so O carries the mass bias
// sum(q(P))/sum(P) - 1 -- a mean of the per-element rounding errors weighted by P_j/sum(P),
// which inherits their bound. O then goes to the output type. Every output threshold below
// is this number and not a measured one: a measured threshold locks in one observation, so
// it false-fails when the value legitimately shifts and stays quiet when a real regression
// lands under it.
constexpr double kOutputBound =
    (1 + rne_bound(4)) *                                           // P -> e4m3
        (1 + rne_bound(std::is_same_v<ODataType, float> ? 24 : 8)) // O -> fp32 or bf16
    - 1;

constexpr const char* prec_string()
{
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8Fp32>)
        return "fp8fp32";
    else
        return "fp8bf16";
}

struct SinkExactResult
{
    double max_relative_error = 0; // vs exact attention, per output element
    double max_lse_error      = 0; // vs exact log-sum-exp, per row
    bool all_finite           = true;
    bool dispatched           = false;
};

// Score patterns. All three keep every q row identical, so the reference below stays a
// single scalar computed from row 0. What differs is how the per-element P-rounding errors
// combine: O/reference - 1 is sum(q(P))/sum(P) - 1, a mean of them weighted by P_j/sum(P).
//   kFlat     all scores equal, so every P is exp2(kPShift) = 256, an e4m3 grid point. The
//             rounding error is zero by construction and no mass bias is visible.
//   kPeriodic a short score cycle repeated ~seqlen_k/9 times. The e4m3 cast of k collapses
//             the nominal values to a handful, each landing above the midpoint of its gap
//             and rounding up, so the errors share a sign and no length dilutes them.
//   kRandom   q.k over N(0,1) entries with the usual 1/sqrt(d) scale, i.e. the score
//             distribution a real workload produces. Rounding errors cancel the way they
//             actually do, so this measures the bias that is really paid.
enum ScorePattern : int
{
    kFlat     = 0,
    kPeriodic = 1,
    kRandom   = 2
};

SinkExactResult run_sink_exact(
    int hdim, int seqlen_k, int seqlen_q, float sink, bool store_lse, int pattern = kFlat)
{
    SinkExactResult result;

    const bool has_sink = std::isfinite(sink);
    const float one     = 1.f;

    std::vector<QDataType> q(seqlen_q * hdim, ck_tile::type_convert<QDataType>(0.f));
    std::vector<QDataType> k(seqlen_k * hdim, ck_tile::type_convert<QDataType>(0.f));
    std::vector<QDataType> v(seqlen_k * hdim, ck_tile::type_convert<QDataType>(1.f));
    std::vector<ODataType> o(seqlen_q * hdim, ck_tile::type_convert<ODataType>(NAN));
    std::vector<float> lse(seqlen_q, NAN);
    if(pattern == kPeriodic)
    {
        for(int i = 0; i < seqlen_q; ++i)
            q[i * hdim] = ck_tile::type_convert<QDataType>(1.f);
        for(int j = 0; j < seqlen_k; ++j)
            k[j * hdim] = ck_tile::type_convert<QDataType>(-8.f + (j % 9 - 4) * 0.25f);
    }
    else if(pattern == kRandom)
    {
        // Fixed seed: the bias has to be reproducible run to run for the bound to mean
        // anything, and a fresh draw would make a failure impossible to reproduce.
        std::mt19937 gen(42);
        std::normal_distribution<float> normal(0.f, 1.f);
        std::vector<float> q_row(hdim);
        for(float& value : q_row)
            value = normal(gen);
        for(int i = 0; i < seqlen_q; ++i)
            for(int c = 0; c < hdim; ++c)
                q[i * hdim + c] = ck_tile::type_convert<QDataType>(q_row[c]);
        for(int j = 0; j < seqlen_k; ++j)
            for(int c = 0; c < hdim; ++c)
                k[j * hdim + c] = ck_tile::type_convert<QDataType>(normal(gen));
    }

    ck_tile::DeviceMem q_buf(q.size()), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem o_buf(o.size() * sizeof(ODataType)), lse_buf(lse.size() * sizeof(float));
    ck_tile::DeviceMem sink_buf(sizeof(float)), scale_buf(sizeof(float));

    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    o_buf.ToDevice(o.data());
    lse_buf.ToDevice(lse.data());
    sink_buf.ToDevice(&sink);
    scale_buf.ToDevice(&one);

    fmha_fwd_args args{};
    args.q_ptr    = q_buf.GetDeviceBuffer();
    args.k_ptr    = k_buf.GetDeviceBuffer();
    args.v_ptr    = v_buf.GetDeviceBuffer();
    args.o_ptr    = o_buf.GetDeviceBuffer();
    args.lse_ptr  = store_lse ? lse_buf.GetDeviceBuffer() : nullptr;
    args.sink_ptr = has_sink ? sink_buf.GetDeviceBuffer() : nullptr;
    // PERTENSOR with unit descales: the represented values above are the real values.
    args.q_descale_ptr = args.k_descale_ptr = args.v_descale_ptr = scale_buf.GetDeviceBuffer();

    args.batch = args.nhead_q = args.nhead_k = args.num_head_q_total = 1;
    args.seqlen_q = args.max_seqlen_q = seqlen_q;
    args.seqlen_k                     = seqlen_k;
    args.hdim_q = args.hdim_v = hdim;
    // kPeriodic puts its spread straight in the logits, so it wants an identity scale.
    args.scale_s = pattern == kPeriodic ? 1.f : 1.f / std::sqrt(static_cast<float>(hdim));

    args.stride_q = args.stride_k = args.stride_v = args.stride_o = hdim;
    args.nhead_stride_q = args.batch_stride_q = seqlen_q * hdim;
    args.nhead_stride_o = args.batch_stride_o = seqlen_q * hdim;
    args.nhead_stride_k = args.batch_stride_k = seqlen_k * hdim;
    args.nhead_stride_v = args.batch_stride_v = seqlen_k * hdim;
    args.nhead_stride_lse = args.batch_stride_lse = seqlen_q;
    args.window_size_left = args.window_size_right = -1;
    args.block_scale_size_q = args.block_scale_size_kv = 128;

    fmha_fwd_traits traits{hdim,
                           hdim,
                           prec_string(),
                           false, // batch mode
                           true,  // v row-major
                           false, // no logits soft cap
                           mask_enum::no_mask,
                           bias_enum::no_bias,
                           store_lse,
                           false, // no dropout
                           quant_scale_enum::pertensor,
                           false, // no skip_min_seqlen_q
                           has_sink};

    if(fmha_fwd(traits, args, ck_tile::stream_config{nullptr, false}) < 0)
        return result; // dispatched stays false: no instance for this trait set
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    result.dispatched = true;

    o_buf.FromDevice(o.data());
    if(store_lse)
        lse_buf.FromDevice(lse.data());

    double real_mass = 0;
    for(int j = 0; j < seqlen_k; ++j)
    {
        double score = 0;
        for(int c = 0; c < hdim; ++c)
            score += static_cast<double>(ck_tile::type_convert<float>(q[c])) *
                     ck_tile::type_convert<float>(k[j * hdim + c]);
        real_mass += std::exp(score * args.scale_s);
    }
    const double denominator = real_mass + (has_sink ? std::exp(static_cast<double>(sink)) : 0.0);
    const double reference   = real_mass / denominator;

    for(const auto& element : o)
    {
        const double value = ck_tile::type_convert<float>(element);
        result.all_finite &= std::isfinite(value);
        result.max_relative_error =
            std::max(result.max_relative_error, std::abs(value / reference - 1));
    }
    if(store_lse)
    {
        const double reference_lse = std::log(denominator);
        for(float value : lse)
        {
            result.all_finite &= std::isfinite(value);
            result.max_lse_error = std::max(result.max_lse_error,
                                            std::abs(static_cast<double>(value) - reference_lse));
        }
    }
    return result;
}

// hdim, seqlen_k, seqlen_q, sink, store_lse
class SinkExact : public ::testing::TestWithParam<std::tuple<int, int, int, float, bool>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    SinkExact,
    ::testing::Combine(::testing::Values(64, 128),  // both fp8 qr_tdm head dims
                       ::testing::Values(128, 500), // multi-tile, and a non-multiple tail
                       ::testing::Values(63, 64),   // padded and exact query tiles
                       ::testing::Values(0.5f,      // sink above the zero-valued real logits
                                         4.f,       // sink dominates the row max
                                         6.f,       // sink dominates by a wide margin
                                         -ck_tile::numeric<float>::infinity()), // no-sink control
                       ::testing::Bool()));

TEST_P(SinkExact, MatchesExactAttention)
{
    const auto [hdim, seqlen_k, seqlen_q, sink, store_lse] = GetParam();

    const auto result = run_sink_exact(hdim, seqlen_k, seqlen_q, sink, store_lse);
    if(!result.dispatched)
        GTEST_SKIP() << "No instance for current parameters";

    EXPECT_TRUE(result.all_finite);
    // kFlat puts every P on 256, a grid point, so the mass term is structurally zero here
    // and only the output cast can contribute. kOutputBound covers both and is loose for
    // this pattern on purpose: one bound, derived from the formats, no per-pattern tuning.
    EXPECT_LT(result.max_relative_error, kOutputBound);
    // LSE is the mathematical log-sum-exp: it carries the unquantized row sum, so it is
    // held to fp32 accumulation error rather than to the fp8 output bound.
    if(store_lse)
        EXPECT_LT(result.max_lse_error, 1e-4);
}

// The mass bias only survives when the per-element errors share a sign; otherwise the
// weighted mean cancels them. kPeriodic constructs the surviving case, so this is the suite
// that can actually approach kOutputBound.
// LSE is unaffected -- it reports l, the unrounded row sum.
TEST(SinkQuantizedMass, NonuniformRealScores)
{
    for(int hdim : {64, 128})
    {
        for(float sink : {-4.f, -ck_tile::numeric<float>::infinity()})
        {
            SCOPED_TRACE(::testing::Message() << "hdim=" << hdim << " sink=" << sink);
            const auto result = run_sink_exact(hdim, 500, 63, sink, true, kPeriodic);
            if(!result.dispatched)
                GTEST_SKIP() << "No instance for current parameters";
            EXPECT_TRUE(result.all_finite);
            EXPECT_LT(result.max_relative_error, kOutputBound);
            EXPECT_LT(result.max_lse_error, 1e-4);
        }
    }
}

// The periodic case above is constructed and says nothing about what a real workload pays.
// Here V=1 and there is no sink, so the exact output is 1.0 for every element and
// max_relative_error IS the quantized-mass bias sum(q(P))/sum(P) - 1, with no other error
// source mixed in, which is the point: kOutputBound is what the format permits, this is what
// a real score distribution costs. Expect it far inside the bound.
// Only the fp32-output binary can see it. Once the bias drops below half a bf16 ulp at 1.0,
// the bf16 build rounds O back to exactly 1 and reports zero no matter what the bias was.
TEST(SinkQuantizedMass, RandomRealScores)
{
    for(int seqlen_k : {512, 2048, 8192})
    {
        SCOPED_TRACE(::testing::Message() << "seqlen_k=" << seqlen_k);
        const auto result =
            run_sink_exact(128, seqlen_k, 64, -ck_tile::numeric<float>::infinity(), true, kRandom);
        if(!result.dispatched)
            GTEST_SKIP() << "No instance for current parameters";
        EXPECT_TRUE(result.all_finite);
        EXPECT_LT(result.max_relative_error, kOutputBound);
        EXPECT_LT(result.max_lse_error, 1e-4);
    }
}

} // namespace
