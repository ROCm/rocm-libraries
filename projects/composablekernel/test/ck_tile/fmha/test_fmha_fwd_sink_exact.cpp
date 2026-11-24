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

// Built only into the fp8 and fp8bf16 binaries (see CMakeLists).
using QDataType = FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
using ODataType = FmhaFwdTypeConfig<DataTypeConfig>::ODataType;

// The 1% output gain bound the quantization-scale suites use.
constexpr double kOutputBound = 0.01;

// d64 / d128 / d256 dispatch to qr_tdm_fp8_sched; every other head dim to qr_tdm.
constexpr bool is_sched_hdim(int hdim) { return hdim == 64 || hdim == 128 || hdim == 256; }

// qr_tdm_fp8_sched normalizes O by sum(q(P)) for per-tensor scales only, so no_scale keeps the
// P-rounding mass bias there and is only held to the bound on qr_tdm head dims.
bool skip_no_scale_on_sched(int hdim, quant_scale_enum qscale)
{
    return qscale == quant_scale_enum::no_scale && is_sched_hdim(hdim);
}

constexpr const char* prec_string()
{
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8>)
        return "fp8";
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
// combine: sum(q(P))/sum(P) - 1 is a mean of them weighted by P_j/sum(P).
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

SinkExactResult run_sink_exact(int hdim,
                               int seqlen_k,
                               int seqlen_q,
                               float sink,
                               bool store_lse,
                               int pattern             = kFlat,
                               quant_scale_enum qscale = quant_scale_enum::pertensor)
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
    if(qscale == quant_scale_enum::pertensor)
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
                           qscale,
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

// hdim, seqlen_k, seqlen_q, sink, store_lse, qscale
class SinkExact
    : public ::testing::TestWithParam<std::tuple<int, int, int, float, bool, quant_scale_enum>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    SinkExact,
    // see is_sched_hdim for which head dims reach qr_tdm.
    ::testing::Combine(::testing::Values(32, 64, 96, 128, 160, 256, 512),
                       ::testing::Values(128, 500), // multi-tile, and a non-multiple tail
                       ::testing::Values(63, 64),   // padded and exact query tiles
                       ::testing::Values(0.5f,      // sink above the zero-valued real logits
                                         4.f,       // sink dominates the row max
                                         6.f,       // sink dominates by a wide margin
                                         -ck_tile::numeric<float>::infinity()), // no-sink control
                       ::testing::Bool(),
                       ::testing::Values(quant_scale_enum::pertensor, quant_scale_enum::no_scale)));

TEST_P(SinkExact, MatchesExactAttention)
{
    const auto [hdim, seqlen_k, seqlen_q, sink, store_lse, qscale] = GetParam();
    if(skip_no_scale_on_sched(hdim, qscale))
        GTEST_SKIP() << "qr_tdm_fp8_sched does not normalize O by sum(q(P)) without a scale";

    const auto result = run_sink_exact(hdim, seqlen_k, seqlen_q, sink, store_lse, kFlat, qscale);
    if(!result.dispatched)
        GTEST_SKIP() << "No instance for current parameters";

    EXPECT_TRUE(result.all_finite);
    // kFlat puts every P on 256, a grid point, so P-rounding is structurally zero here.
    EXPECT_LT(result.max_relative_error, kOutputBound);
    // LSE is the mathematical log-sum-exp: it carries the unquantized row sum, so it is
    // held to fp32 accumulation error rather than to the output bound.
    if(store_lse)
        EXPECT_LT(result.max_lse_error, 1e-4);
}

// P-rounding errors that share a sign survive in the sum(q(P))/sum(P) mean instead of
// cancelling. kPeriodic constructs that case: normalizing O by sum(q(P)) must still land
// on the exact attention within the output bound.
// LSE is unaffected -- it reports the unrounded row sum of P.
// hdim, sink, store_lse, qscale
class SinkQuantizedMassNonuniform
    : public ::testing::TestWithParam<std::tuple<int, float, bool, quant_scale_enum>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    SinkQuantizedMassNonuniform,
    ::testing::Combine(::testing::Values(32, 64, 96, 128, 160, 256, 512),
                       ::testing::Values(-4.f, -ck_tile::numeric<float>::infinity()),
                       ::testing::Bool(),
                       ::testing::Values(quant_scale_enum::pertensor, quant_scale_enum::no_scale)));

TEST_P(SinkQuantizedMassNonuniform, MatchesExactAttention)
{
    const auto [hdim, sink, store_lse, qscale] = GetParam();
    if(skip_no_scale_on_sched(hdim, qscale))
        GTEST_SKIP() << "qr_tdm_fp8_sched does not normalize O by sum(q(P)) without a scale";

    const auto result = run_sink_exact(hdim, 500, 63, sink, store_lse, kPeriodic, qscale);
    if(!result.dispatched)
        GTEST_SKIP() << "No instance for current parameters";

    EXPECT_TRUE(result.all_finite);
    EXPECT_LT(result.max_relative_error, kOutputBound);
    EXPECT_LT(result.max_lse_error, 1e-4);
}

// The periodic case above is constructed and says nothing about what a real workload pays.
// Here V=1 and there is no sink, so the exact output is 1.0 for every element and
// max_relative_error is only output rounding and fp32 arithmetic: O is normalized by sum(q(P)),
// so a real score distribution stays well inside the output bound.
// seqlen_k, hdim, store_lse, qscale
class SinkQuantizedMassRandom
    : public ::testing::TestWithParam<std::tuple<int, int, bool, quant_scale_enum>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SinkQuantizedMassRandom,
                         ::testing::Combine(::testing::Values(512, 2048, 8192),
                                            ::testing::Values(96, 128, 160),
                                            ::testing::Bool(),
                                            ::testing::Values(quant_scale_enum::pertensor,
                                                              quant_scale_enum::no_scale)));

TEST_P(SinkQuantizedMassRandom, OutputCastOnly)
{
    const auto [seqlen_k, hdim, store_lse, qscale] = GetParam();
    if(skip_no_scale_on_sched(hdim, qscale))
        GTEST_SKIP() << "qr_tdm_fp8_sched does not normalize O by sum(q(P)) without a scale";

    const auto result = run_sink_exact(
        hdim, seqlen_k, 64, -ck_tile::numeric<float>::infinity(), store_lse, kRandom, qscale);
    if(!result.dispatched)
        GTEST_SKIP() << "No instance for current parameters";

    EXPECT_TRUE(result.all_finite);
    EXPECT_LT(result.max_relative_error, kOutputBound);
    EXPECT_LT(result.max_lse_error, 1e-4);
}

} // namespace
