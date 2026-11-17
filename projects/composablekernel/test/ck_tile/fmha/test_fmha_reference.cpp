// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_reference.hpp"

#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

namespace {

namespace ref = ck_tile::test::reference;

void Check(bool condition, const char* message)
{
    if(!condition)
        throw std::runtime_error(message);
}

void Near(double actual, double expected, double tolerance, const char* message)
{
    Check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void Reject(Function function, const char* message)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch(const std::invalid_argument&)
    {
        rejected = true;
    }
    Check(rejected, message);
}

struct Case
{
    std::vector<float> q, k, v, o, lse;
    ref::Sequence sequence;

    Case(std::size_t queries,
         std::size_t keys,
         std::size_t query_heads     = 1,
         std::size_t kv_heads        = 1,
         std::size_t query_dimension = 2,
         std::size_t value_dimension = 3)
        : q(queries * query_heads * query_dimension),
          k(keys * kv_heads * query_dimension),
          v(keys * kv_heads * value_dimension),
          o(queries * query_heads * value_dimension),
          lse(queries * query_heads)
    {
        sequence.q = {q.data(), q.size(), 0, query_dimension, query_heads * query_dimension, 1};
        sequence.k = {k.data(), k.size(), 0, query_dimension, kv_heads * query_dimension, 1};
        sequence.v = {v.data(), v.size(), 0, value_dimension, kv_heads * value_dimension, 1};
        sequence.query_length    = queries;
        sequence.key_length      = keys;
        sequence.query_heads     = query_heads;
        sequence.kv_heads        = kv_heads;
        sequence.query_dimension = query_dimension;
        sequence.value_dimension = value_dimension;
    }

    ref::Output Output() const
    {
        return {{o.data(),
                 o.size(),
                 0,
                 sequence.value_dimension,
                 sequence.query_heads * sequence.value_dimension,
                 1},
                ref::FloatView{lse.data(), lse.size(), 0, sequence.query_length, 1, 1}};
    }

    void SetOutput(const ref::Reference& reference)
    {
        for(const auto& row : reference.rows)
        {
            for(std::size_t dim = 0; dim < sequence.value_dimension; ++dim)
                o[(row.query * sequence.query_heads + row.head) * sequence.value_dimension + dim] =
                    row.output[dim];
            lse[row.head * sequence.query_length + row.query] = row.lse;
        }
    }

    ref::Metrics Compare(const ref::Reference& reference) const
    {
        return ref::Compare({sequence}, reference, {Output()});
    }
};

// Same public left/right-window convention as CK: negative is unlimited,
// right=0 is causal, and bottom-right shifts the diagonal by Sk-Sq.
auto WindowMask(std::ptrdiff_t queries,
                std::ptrdiff_t keys,
                std::ptrdiff_t left,
                std::ptrdiff_t right,
                bool top_left)
{
    return [=](std::size_t query, std::size_t key) {
        const auto center = static_cast<std::ptrdiff_t>(query) + (top_left ? 0 : keys - queries);
        const auto column = static_cast<std::ptrdiff_t>(key);
        return (left >= 0 && column < center - left) || (right >= 0 && column > center + right);
    };
}

void ZeroAndConstant()
{
    Case zeros(3, 7);
    const auto zero_ref = ref::Compute({zeros.sequence}, 3);
    zeros.SetOutput(zero_ref);
    for(const auto& row : zero_ref.rows)
    {
        Check(row.valid_keys == 7, "zero input key count");
        Near(row.lse, std::log(7.0), 2e-7, "zero input LSE");
        for(const auto value : row.output)
            Near(value, 0, 0, "zero input output");
    }
    const auto metrics = zeros.Compare(zero_ref);
    Check(metrics.passed && metrics.min_row_cosine == 1 && metrics.mean_row_cosine == 1 &&
              metrics.relative_rms_error == 0 && metrics.max_relative_error == 0,
          "two zero rows must have unit cosine and zero error");

    Case constant(4, 5, 1, 1, 2, 2);
    std::fill(constant.q.begin(), constant.q.end(), 2);
    std::fill(constant.k.begin(), constant.k.end(), -1);
    for(std::size_t key = 0; key < 5; ++key)
    {
        constant.v[2 * key]     = 2;
        constant.v[2 * key + 1] = -4;
    }
    constant.sequence.q_descale   = 0.5;
    constant.sequence.k_descale   = 1.25;
    constant.sequence.v_descale   = 2;
    constant.sequence.score_scale = 0.25;
    const auto constant_ref       = ref::Compute({constant.sequence}, 2);
    for(const auto& row : constant_ref.rows)
    {
        Near(row.output[0], 4, 0, "constant V positive output, descale once");
        Near(row.output[1], -8, 0, "constant V negative output, descale once");
        Near(row.lse, -0.625 + std::log(5.0), 2e-7, "Q/K descales applied exactly once");
    }
}

void MasksAndEmptyRows()
{
    Case c(5, 3, 1, 1, 1, 1);
    c.v                  = {1, 2, 3};
    c.sequence.v.data    = c.v.data();
    c.sequence.is_masked = WindowMask(5, 3, -1, 0, true);
    auto result          = ref::Compute({c.sequence}, 2);
    const std::vector<float> top_left{1, 1.5, 2, 2, 2};
    for(std::size_t row = 0; row < 5; ++row)
        Near(result.rows[row].output[0], top_left[row], 0, "top-left causal output");

    c.sequence.is_masked = WindowMask(5, 3, -1, 0, false);
    result               = ref::Compute({c.sequence}, 2);
    const std::vector<float> bottom_right{0, 0, 1, 1.5, 2};
    for(std::size_t row = 0; row < 5; ++row)
    {
        Near(result.rows[row].output[0], bottom_right[row], 0, "bottom-right causal output");
        if(row < 2)
            Check(result.rows[row].valid_keys == 0 &&
                      result.rows[row].lse == -std::numeric_limits<float>::infinity(),
                  "empty masked row needs zero output and -infinity LSE");
    }
    c.SetOutput(result);
    Check(c.Compare(result).passed, "bottom-right empty rows compare");
    c.lse[0] = 0;
    Check(!c.Compare(result).passed, "finite LSE on sampled empty row must fail");

    Case window(131, 135, 1, 1, 1, 1);
    for(std::size_t key = 0; key < 135; ++key)
        window.v[key] = static_cast<float>(key);
    window.sequence.sampled_query_rows = {0, 127, 128, 130};
    for(const bool top : {true, false})
    {
        window.sequence.is_masked = WindowMask(131, 135, 2, 1, top);
        const auto reference      = ref::Compute({window.sequence}, 128);
        for(const auto& row : reference.rows)
        {
            const std::size_t center = row.query + (top ? 0 : 4);
            const auto first         = center < 2 ? 0 : center - 2;
            const auto last          = std::min<std::size_t>(134, center + 1);
            Near(row.output[0], (first + last) * 0.5, 0, "window across key chunk boundary");
            Check(row.valid_keys == last - first + 1, "window valid key count");
        }
    }

    Case empty(2, 0);
    result = ref::Compute({empty.sequence});
    empty.SetOutput(result);
    Check(empty.Compare(result).passed && result.rows[0].valid_keys == 0,
          "zero-key input must not dereference null K/V");
}

void RaggedGqaAndStrides()
{
    // Padding is intentionally nonfinite: only logical elements may be read.
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> q(200, nan), k(150, nan), v(200, nan), o(200, nan), lse(70, nan);
    std::vector<ref::Sequence> sequences;
    std::vector<ref::Output> outputs;
    for(std::size_t batch = 0; batch < 3; ++batch)
    {
        ref::Sequence s;
        s.query_length    = batch + 1;
        s.key_length      = 3 - batch;
        s.query_heads     = 4;
        s.kv_heads        = batch == 0 ? 4 : batch == 1 ? 2 : 1;
        s.query_dimension = 1;
        s.value_dimension = 2;
        s.q               = {q.data(), q.size(), 3 + 60 * batch, 11, 2, 1};
        s.k               = {k.data(), k.size(), 2 + 45 * batch, 9, 2, 1};
        s.v               = {v.data(), v.size(), 5 + 60 * batch, 11, 2, 1};
        ref::FloatView out{o.data(), o.size(), 4 + 60 * batch, 13, 3, 1};
        ref::FloatView ls{lse.data(), lse.size(), 1 + 20 * batch, 4, 1, 1};
        for(std::size_t h = 0; h < s.query_heads; ++h)
            for(std::size_t row = 0; row < s.query_length; ++row)
            {
                q[s.q.offset + h * s.q.head_stride + row * s.q.row_stride] = 0;
                const auto kv_head = h / (s.query_heads / s.kv_heads);
                for(std::size_t d = 0; d < 2; ++d)
                    o[out.offset + h * out.head_stride + row * out.row_stride + d] =
                        static_cast<float>(100 * batch + 10 * kv_head + d + 1);
                lse[ls.offset + h * ls.head_stride + row] =
                    std::log(static_cast<float>(s.key_length));
            }
        for(std::size_t h = 0; h < s.kv_heads; ++h)
            for(std::size_t row = 0; row < s.key_length; ++row)
            {
                k[s.k.offset + h * s.k.head_stride + row * s.k.row_stride] = 0;
                for(std::size_t d = 0; d < 2; ++d)
                    v[s.v.offset + h * s.v.head_stride + row * s.v.row_stride + d] =
                        static_cast<float>(100 * batch + 10 * h + d + 1);
            }
        sequences.push_back(s);
        outputs.push_back({out, ls});
    }
    const auto result  = ref::Compute(sequences, 2);
    const auto metrics = ref::Compare(sequences, result, outputs);
    Check(metrics.passed && metrics.checked_rows == 24 && metrics.checked_elements == 48,
          "ragged MHA/GQA/MQA with distinct Q/K lengths and padded offsets");
    Check(metrics.scanned_output_elements == 48 && metrics.scanned_lse_rows == 24,
          "padding must not enter finite checks");
}

void ChunkedAgainstIndependentDense()
{
    Case c(7, 257, 4, 2, 5, 3);
    for(std::size_t i = 0; i < c.q.size(); ++i)
        c.q[i] = static_cast<float>(std::sin(static_cast<double>(i)) * 3);
    for(std::size_t i = 0; i < c.k.size(); ++i)
        c.k[i] = static_cast<float>(std::cos(static_cast<double>(i)) * 2);
    for(std::size_t i = 0; i < c.v.size(); ++i)
        c.v[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.1));
    c.sequence.score_scale = 0.7;
    c.sequence.q_descale   = 0.5;
    c.sequence.k_descale   = 1.25;
    c.sequence.v_descale   = 2;
    for(const auto chunk : {1, 3, 128, 512})
    {
        const auto result = ref::Compute({c.sequence}, chunk);
        for(const auto& row : result.rows)
        {
            std::vector<double> scores(c.sequence.key_length);
            for(std::size_t key = 0; key < scores.size(); ++key)
            {
                double dot = 0;
                for(std::size_t dim = 0; dim < c.sequence.query_dimension; ++dim)
                    dot += static_cast<double>(c.sequence.q(row.head, row.query, dim)) *
                           c.sequence.q_descale * c.sequence.k(row.head / 2, key, dim) *
                           c.sequence.k_descale;
                scores[key] = dot * c.sequence.score_scale;
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double denominator   = 0;
            std::vector<double> expected(3, 0);
            for(std::size_t key = 0; key < scores.size(); ++key)
            {
                const double weight = std::exp(scores[key] - maximum);
                denominator += weight;
                for(std::size_t dim = 0; dim < 3; ++dim)
                    expected[dim] +=
                        weight * c.sequence.v(row.head / 2, key, dim) * c.sequence.v_descale;
            }
            for(std::size_t dim = 0; dim < 3; ++dim)
                Near(row.output[dim], expected[dim] / denominator, 3e-6, "chunked vs dense O");
            Near(row.lse, maximum + std::log(denominator), 3e-6, "chunked vs dense LSE");
        }
    }
}

void LongSampledAndFiniteChecks()
{
    Case c(32768, 32768, 1, 1, 1, 2);
    std::fill(c.v.begin(), c.v.end(), 1);
    std::fill(c.o.begin(), c.o.end(), 1);
    std::fill(c.lse.begin(), c.lse.end(), std::log(32768.0f));
    c.sequence.sampled_query_rows = {0, 1, 127, 128, 16384, 32767};
    const auto result             = ref::Compute({c.sequence}, 127);
    auto metrics                  = c.Compare(result);
    Check(metrics.passed && metrics.checked_rows == 6 && result.rows.size() == 6 &&
              metrics.checked_elements == 12 && metrics.scanned_output_elements == 65536 &&
              metrics.scanned_lse_rows == 32768,
          "32K oracle must retain only sampled rows and scan all logical outputs");
    for(const auto& row : result.rows)
    {
        Check(row.valid_keys == 32768, "sampled row uses all keys");
        Near(row.output[0], 1, 0, "32K constant V output");
    }
    c.o[1000] = std::numeric_limits<float>::quiet_NaN();
    metrics   = c.Compare(result);
    Check(!metrics.passed && metrics.nonfinite_output_count == 1 && std::isinf(metrics.rms_error),
          "NaN in an unsampled output row must fail");
    c.o[1000]  = 1;
    c.lse[500] = -std::numeric_limits<float>::infinity();
    Check(!c.Compare(result).passed, "-infinity LSE in unsampled nonempty row must fail");
    c.lse[500] = std::numeric_limits<float>::quiet_NaN();
    Check(!c.Compare(result).passed, "NaN LSE in unsampled row must fail");
    c.lse[500] = std::numeric_limits<float>::infinity();
    Check(!c.Compare(result).passed, "+infinity LSE in unsampled row must fail");
}

void NegativeControlsAndMetricDefinitions()
{
    Case c(2, 1, 1, 1, 1, 2);
    c.v[0]            = 1;
    c.v[1]            = 2;
    const auto result = ref::Compute({c.sequence});
    c.SetOutput(result);
    Check(c.Compare(result).passed, "exact nonzero output must pass");
    c.o[0]       = 1.02f;
    auto metrics = c.Compare(result);
    Check(!metrics.passed && metrics.max_abs_error > 0.015625, "maxabs negative control");
    Near(metrics.rms_error, (static_cast<double>(c.o[0]) - 1) / 2, 1e-12, "absolute RMS");
    Near(metrics.relative_rms_error,
         (static_cast<double>(c.o[0]) - 1) / std::sqrt(10.0),
         1e-12,
         "relative RMS");
    Near(metrics.max_relative_error,
         static_cast<double>(c.o[0]) - 1,
         1e-12,
         "maximum relative error");

    c.SetOutput(result);
    c.lse[0] += 0.021f;
    Check(!c.Compare(result).passed, "LSE atol negative control at reference zero");
    c.SetOutput(result);
    c.o[0]  = 0;
    c.o[1]  = 0;
    metrics = c.Compare(result);
    Check(!metrics.passed && metrics.min_row_cosine == 0 && metrics.mean_row_cosine == 0.5,
          "one zero row has cosine zero, not a skipped row");
    c.SetOutput(result);
    c.o[0] = -1;
    c.o[1] = -2;
    Check(c.Compare(result).min_row_cosine == -1, "antiparallel cosine");

    Case cosine(2, 1, 1, 1, 1, 2);
    cosine.v[0]           = 0.01f;
    const auto cosine_ref = ref::Compute({cosine.sequence});
    cosine.SetOutput(cosine_ref);
    cosine.o[0] = 0;
    cosine.o[1] = 0.01f;
    metrics     = cosine.Compare(cosine_ref);
    Check(!metrics.passed && metrics.max_abs_error < 0.015625 && metrics.min_row_cosine == 0 &&
              std::isinf(metrics.max_relative_error),
          "cosine must reject small absolute but orthogonal output error");

    Case zero(1, 1);
    const auto zero_ref = ref::Compute({zero.sequence});
    zero.SetOutput(zero_ref);
    zero.o[0] = 0.001f;
    metrics   = zero.Compare(zero_ref);
    Check(!metrics.passed && metrics.min_row_cosine == 0 && std::isinf(metrics.relative_rms_error),
          "nonzero actual against zero reference must fail");
    auto no_lse = c.Output();
    no_lse.lse.reset();
    c.SetOutput(result);
    c.lse[0] = std::numeric_limits<float>::quiet_NaN();
    metrics  = ref::Compare({c.sequence}, result, {no_lse});
    Check(metrics.passed && metrics.scanned_lse_rows == 0 && metrics.checked_lse_rows == 0,
          "LSE-disabled runs must report no LSE validation");
}

void InvalidMetadataAndExtremeScores()
{
    Case c(2, 2);
    Reject([&] { ref::Compute({c.sequence}, 0); }, "zero chunk rejected");
    auto invalid     = c.sequence;
    invalid.kv_heads = 2;
    Reject([&] { ref::Compute({invalid}); }, "nonintegral GQA rejected");
    invalid          = c.sequence;
    invalid.q.offset = invalid.q.size;
    Reject([&] { ref::Compute({invalid}); }, "view allocation guard");
    invalid              = c.sequence;
    invalid.q.row_stride = std::numeric_limits<std::size_t>::max();
    Reject([&] { ref::Compute({invalid}); }, "view extent overflow guard");
    invalid                    = c.sequence;
    invalid.sampled_query_rows = {1, 1};
    Reject([&] { ref::Compute({invalid}); }, "duplicate samples rejected");
    invalid.sampled_query_rows = {2};
    Reject([&] { ref::Compute({invalid}); }, "out-of-range samples rejected");
    c.q[0] = std::numeric_limits<float>::quiet_NaN();
    Reject([&] { ref::Compute({c.sequence}); }, "nonfinite decoded input rejected");

    Case extreme(1, 3, 1, 1, 1, 1);
    extreme.q[0]      = 1;
    extreme.k[0]      = -10000;
    extreme.k[1]      = 0;
    extreme.k[2]      = 10000;
    extreme.v[0]      = 1;
    extreme.v[1]      = 2;
    extreme.v[2]      = 3;
    const auto result = ref::Compute({extreme.sequence}, 1);
    Near(result.rows[0].output[0], 3, 0, "online rescaling across extreme finite scores");
    Near(result.rows[0].lse, 10000, 0, "extreme finite LSE");
    extreme.SetOutput(result);
    extreme.lse[0] += 19;
    Check(extreme.Compare(result).passed, "LSE relative tolerance at large magnitude");
    extreme.lse[0] += 2;
    Check(!extreme.Compare(result).passed, "LSE relative tolerance negative control");
    auto broken_reference          = result;
    broken_reference.rows[0].query = 1;
    Reject([&] { extreme.Compare(broken_reference); }, "reference row identity guard");
}

void VirtualSink()
{
    Case c(2, 3, 2, 1, 1, 1);
    c.v                    = {2, 4, 6};
    c.sequence.v.data      = c.v.data();
    c.sequence.score_scale = 0.19f;
    c.sequence.sink_logits = {0, std::log(3.0f)};
    const auto result      = ref::Compute({c.sequence}, 2);
    for(const auto& row : result.rows)
    {
        Check(row.valid_keys == 3, "virtual sink is not a real key");
        Near(row.output[0], row.head == 0 ? 3 : 2, 2e-7, "V=0 sink normalization");
        Near(row.lse, std::log(row.head == 0 ? 4.0 : 6.0), 2e-7, "sink logit is already scaled");
    }
    c.SetOutput(result);
    Check(c.Compare(result).passed, "virtual sink reference output");
    c.o[0] = 4;
    Check(!c.Compare(result).passed, "omitting the sink normalization must fail");

    auto without_sink = c.sequence;
    without_sink.sink_logits.clear();
    auto negative_infinity_sink = without_sink;
    negative_infinity_sink.sink_logits.assign(2, -std::numeric_limits<float>::infinity());
    const auto plain    = ref::Compute({without_sink}, 2);
    const auto inactive = ref::Compute({negative_infinity_sink}, 2);
    for(std::size_t i = 0; i < plain.rows.size(); ++i)
        Check(plain.rows[i].output == inactive.rows[i].output &&
                  plain.rows[i].lse == inactive.rows[i].lse,
              "-infinity sink preserves no-sink output exactly");

    Case empty(2, 0, 1, 1, 1, 1);
    empty.sequence.sink_logits = {2.5f};
    const auto empty_result    = ref::Compute({empty.sequence});
    for(const auto& row : empty_result.rows)
    {
        Check(row.valid_keys == 0 && row.output[0] == 0, "empty real keys with V=0 sink");
        Near(row.lse, 2.5, 0, "sink-only LSE");
    }
    empty.SetOutput(empty_result);
    Check(empty.Compare(empty_result).passed, "sink-only output passes");
    empty.lse[1] = -std::numeric_limits<float>::infinity();
    Check(!empty.Compare(empty_result).passed, "finite sink rejects -infinity LSE");

    empty.sequence.sink_logits = {10000};
    const auto extreme         = ref::Compute({empty.sequence});
    Near(extreme.rows[0].lse, 10000, 0, "large finite sink remains stable");
    auto invalid        = c.sequence;
    invalid.sink_logits = {0};
    Reject([&] { ref::Compute({invalid}); }, "sink head count guard");
    invalid.sink_logits = {0, std::numeric_limits<float>::quiet_NaN()};
    Reject([&] { ref::Compute({invalid}); }, "NaN sink guard");
    invalid.sink_logits = {0, std::numeric_limits<float>::infinity()};
    Reject([&] { ref::Compute({invalid}); }, "+infinity sink guard");
}

void ScaleAwareAccuracyContract()
{
    const ref::AccuracyContract bf16{7, 64}, fp16{10, 128};
    Near(bf16.UnitRoundoff(), 0x1p-8, 0, "BF16 half ULP");
    Near(fp16.UnitRoundoff(), 0x1p-11, 0, "FP16 half ULP");
    Near(fp16.AbsoluteRoundoff(), 0x1p-25, 0, "FP16 subnormal rounding floor");
    Check(fp16.RequiresStrict() && !bf16.RequiresStrict(), "dtype final gate selection");
    Case constant(128, 513, 1, 1, 1, 4);
    const std::array<float, 4> values{0.75f, -1.25f, 1.5f, -1.75f};
    for(std::size_t key = 0; key < constant.sequence.key_length; ++key)
        for(std::size_t dim = 0; dim < values.size(); ++dim)
            constant.v[key * values.size() + dim] = values[dim];
    const auto expected = ref::Compute({constant.sequence}, 64);
    Check(expected.key_chunk_size == 64, "reference records its actual chunk size");
    constant.SetOutput(expected);
    auto compare = [&](const auto& contract) {
        return ref::Compare({constant.sequence}, expected, {constant.Output()}, {}, contract);
    };
    for(const auto& contract : {bf16, fp16})
    {
        auto metrics = compare(contract);
        Check(metrics.contract_passed && metrics.strict_passed && metrics.passed &&
                  metrics.max_component_ratio == 0,
              "constant V gives exact output regardless of key tiles");
        // Actual BF16 casts of (1 +/- .01) * values. Several mantissas are
        // needed: rounding 1.01 to BF16 can hide the fault for V=1.
        const std::array<std::array<float, 4>, 2> gained{{
            {0.7578125f, -1.265625f, 1.515625f, -1.765625f},
            {0.7421875f, -1.234375f, 1.484375f, -1.734375f},
        }};
        for(const auto& pattern : gained)
        {
            for(std::size_t i = 0; i < constant.o.size(); ++i)
                constant.o[i] = pattern[i % pattern.size()];
            metrics = compare(contract);
            Check(!metrics.contract_passed && metrics.component_mismatch_rows == 128 &&
                      metrics.max_component_ratio > 1,
                  "signed constant V detects post-rounded positive and negative gain");
        }
        constant.SetOutput(expected);
    }
    constant.o[0] += 0.02f;
    Check(compare(bf16).component_mismatch_count == 1, "one over-budget element must fail");
    constant.SetOutput(expected);
    constant.o[0] = std::numeric_limits<float>::quiet_NaN();
    Check(!compare(bf16).contract_passed, "nonfinite O must fail the contract");
    constant.SetOutput(expected);
    constant.lse[0] += 0.1f;
    Check(!compare(bf16).contract_passed, "LSE mismatch must fail the contract");
    constant.SetOutput(expected);
    std::fill(constant.o.begin(), constant.o.end(), 0);
    Check(!compare(bf16).contract_passed, "zero output must fail for positive constant V");

    // Signed contributions almost cancel. A small P rounding perturbation
    // can exceed a relative-to-O or cosine limit while staying inside the
    // independently computed contribution bound. Repeat identical rows:
    // deterministic rounding is not an independent statistical sample.
    Case cancellation(128, 2, 1, 1, 1, 2);
    cancellation.v               = {1, 0, -1, 0x1p-10f};
    cancellation.sequence.v.data = cancellation.v.data();
    const auto cancelling_ref    = ref::Compute({cancellation.sequence}, 1);
    cancellation.SetOutput(cancelling_ref);
    for(std::size_t row = 0; row < 128; ++row)
        cancellation.o[2 * row] += 0x1p-11f;
    const auto cancelling_metrics =
        ref::Compare({cancellation.sequence}, cancelling_ref, {cancellation.Output()}, {}, bf16);
    Check(cancelling_metrics.contract_passed && !cancelling_metrics.strict_passed &&
              cancelling_metrics.relative_rms_error > 0.5 &&
              cancelling_metrics.max_component_ratio < 1,
          "cancellation must not impose a universal global-relative-error cap");
    Check(!cancellation.Compare(cancelling_ref).passed, "default comparison stays strict");
    const auto fp16_cancellation =
        ref::Compare({cancellation.sequence}, cancelling_ref, {cancellation.Output()}, {}, fp16);
    Check(fp16_cancellation.contract_passed && !fp16_cancellation.strict_passed &&
              !fp16_cancellation.passed,
          "FP16 final status must include its strict gate");
    Near(cancelling_ref.rows[0].absolute_contribution[0], 1, 0, "independent P abs(V)");
    auto finer_chunks           = cancelling_ref;
    finer_chunks.key_chunk_size = 2;
    Check(ref::detail::ComponentLimit(cancelling_ref.rows[0], 0, 2, 1, bf16) >
              ref::detail::ComponentLimit(finer_chunks.rows[0], 0, 2, 2, bf16),
          "device and reference reduction counts enter the budget separately");

    // Large signed contributions must not bypass the original absolute cap,
    // even when cancellation makes the component budget much larger.
    Case large_cancellation(1, 2, 1, 1, 1, 1);
    large_cancellation.v               = {1000, -1000};
    large_cancellation.sequence.v.data = large_cancellation.v.data();
    const auto large_ref               = ref::Compute({large_cancellation.sequence});
    large_cancellation.SetOutput(large_ref);
    large_cancellation.o[0]   = 0.02f;
    const auto capped_metrics = ref::Compare(
        {large_cancellation.sequence}, large_ref, {large_cancellation.Output()}, {}, bf16);
    Check(capped_metrics.component_mismatch_count == 0 && capped_metrics.max_component_ratio < 1 &&
              !capped_metrics.contract_passed &&
              capped_metrics.max_abs_error > ref::Thresholds{}.max_abs_error,
          "component budget must retain the original absolute error cap");

    auto invalid           = cancelling_ref;
    invalid.key_chunk_size = 0;
    Reject(
        [&] { ref::Compare({cancellation.sequence}, invalid, {cancellation.Output()}, {}, bf16); },
        "missing reference chunk metadata rejected");
    invalid = cancelling_ref;
    invalid.rows[0].absolute_contribution.clear();
    Reject(
        [&] { ref::Compare({cancellation.sequence}, invalid, {cancellation.Output()}, {}, bf16); },
        "missing contribution metadata must not silently pass");
    Reject(
        [&] {
            ref::Compare({cancellation.sequence},
                         cancelling_ref,
                         {cancellation.Output()},
                         {},
                         ref::AccuracyContract{7, 0});
        },
        "zero key tile contract rejected");
}

void LargestKeyGainSensitivity()
{
    // Exercise the tested largest K and Dq with nonzero dot products, so
    // score roundoff contributes to the budget. This is a CPU fault control,
    // not a claim that every possible QK distribution detects a 1% gain.
    Case constant(1, 32768, 1, 1, 192, 4);
    std::fill(constant.q.begin(), constant.q.end(), 0.125f);
    std::fill(constant.k.begin(), constant.k.end(), 0.125f);
    const std::array<float, 4> values{0.75f, -1.25f, 1.5f, -1.75f};
    for(std::size_t key = 0; key < constant.sequence.key_length; ++key)
        for(std::size_t dim = 0; dim < values.size(); ++dim)
            constant.v[key * values.size() + dim] = values[dim];
    const auto expected = ref::Compute({constant.sequence});
    Check(expected.rows[0].score_roundoff > 0, "nonzero D192 score roundoff budget");
    const std::array<std::array<float, 4>, 2> gained{{
        {0.7578125f, -1.265625f, 1.515625f, -1.765625f},
        {0.7421875f, -1.234375f, 1.484375f, -1.734375f},
    }};
    for(const auto tile : {std::size_t{64}, std::size_t{128}})
    {
        const ref::AccuracyContract contract{7, tile};
        constant.SetOutput(expected);
        Check(ref::Compare({constant.sequence}, expected, {constant.Output()}, {}, contract).passed,
              "largest-key exact constant output passes");
        for(const auto& pattern : gained)
        {
            std::copy(pattern.begin(), pattern.end(), constant.o.begin());
            const auto metrics =
                ref::Compare({constant.sequence}, expected, {constant.Output()}, {}, contract);
            Check(!metrics.passed && metrics.max_component_ratio > 1,
                  "largest-key nonzero-QK signed constant detects rounded positive/negative gain");
        }
    }
}

// ---- FP8 P (fp8bf16): the quantized-P contract against a CPU model of the device arithmetic ----

// Round to nearest even onto the OCP e4m3 grid (3 mantissa bits, smallest normal 2^-6, largest
// finite 448); below 2^-6 the subnormal grid 2^-9, or zero when subnormals are flushed.
double RoundE4M3(double x, bool keep_subnormals = true)
{
    const double magnitude = std::abs(x);
    if(magnitude == 0)
        return 0;
    if(magnitude >= 448)
        return std::copysign(448.0, x);
    int exponent = 0;
    std::frexp(magnitude, &exponent); // magnitude in [2^(exponent-1), 2^exponent)
    if(exponent - 1 < -6 && !keep_subnormals)
        return 0;
    const double step = std::ldexp(1.0, std::max(exponent - 1, -6) - 3);
    return std::copysign(std::nearbyint(magnitude / step) * step, x);
}

// Round to nearest even onto the BF16 grid (finite values).
float RoundBf16(float x)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&x, &bits, sizeof(bits));
    return x;
}

// The descales and the key-to-page mapping the modelled device uses (the controls perturb them).
struct DeviceInputs
{
    float q_descale = 1, k_descale = 1, v_descale = 1;
    // physical_key[k]: the key the device reads for logical key k (a wrong page-table entry).
    std::vector<std::size_t> physical_key;
};

// The FP8 page tile's arithmetic on the CPU: per 128-key tile, FP32 scores with the descales folded
// in, P against the running row maximum, scale * P converted to e4m3, O and the mass of the
// converted P accumulated in FP32 with the online rescale, O / mass * v_descale rounded to BF16.
void ModelFp8Device(Case& c, const ref::QuantizedP& quantized, const DeviceInputs& device)
{
    const auto& s       = c.sequence;
    const std::size_t D = s.query_dimension, Dv = s.value_dimension;
    const float scale_s = s.score_scale * device.q_descale * device.k_descale;
    for(std::size_t head = 0; head < s.query_heads; ++head)
    {
        const auto kv_head = head / (s.query_heads / s.kv_heads);
        for(std::size_t row = 0; row < s.query_length; ++row)
        {
            float maximum = -std::numeric_limits<float>::infinity();
            float mass    = 0;
            std::vector<float> o(Dv, 0);
            for(std::size_t begin = 0; begin < s.key_length; begin += 128)
            {
                const auto end = std::min(begin + 128, s.key_length);
                std::vector<float> score(end - begin, -std::numeric_limits<float>::infinity());
                float tile_max = -std::numeric_limits<float>::infinity();
                for(std::size_t key = begin; key < end; ++key)
                {
                    if(s.is_masked && s.is_masked(row, key))
                        continue;
                    const auto physical =
                        device.physical_key.empty() ? key : device.physical_key[key];
                    float dot = 0;
                    for(std::size_t d = 0; d < D; ++d)
                        dot = std::fma(s.q(head, row, d), s.k(kv_head, physical, d), dot);
                    score[key - begin] = dot * scale_s;
                    tile_max           = std::max(tile_max, score[key - begin]);
                }
                if(tile_max == -std::numeric_limits<float>::infinity())
                    continue;
                const float next    = std::max(maximum, tile_max);
                const float rescale = mass == 0 ? 0 : std::exp(maximum - next);
                mass *= rescale;
                for(auto& value : o)
                    value *= rescale;
                for(std::size_t key = begin; key < end; ++key)
                {
                    if(score[key - begin] == -std::numeric_limits<float>::infinity())
                        continue;
                    const auto physical =
                        device.physical_key.empty() ? key : device.physical_key[key];
                    const double p       = std::exp(score[key - begin] - next);
                    const auto converted = static_cast<float>(
                        RoundE4M3(quantized.scale * p, quantized.keeps_subnormals) /
                        quantized.scale);
                    mass += converted;
                    for(std::size_t d = 0; d < Dv; ++d)
                        o[d] = std::fma(converted, s.v(kv_head, physical, d), o[d]);
                }
                maximum = next;
            }
            for(std::size_t d = 0; d < Dv; ++d)
                c.o[(row * s.query_heads + head) * Dv + d] =
                    mass == 0 ? 0 : RoundBf16(o[d] / mass * device.v_descale);
            c.lse[head * s.query_length + row] = 0;
        }
    }
}

// FP8-representable random inputs: Q, K and V on the e4m3 grid with distinct non-unit descales.
// q, k ~ U(-amplitude, amplitude) give scores with a standard deviation near
// amplitude^2 / 3 * q_descale * k_descale * sqrt(D) * score_scale.
void FillFp8(Case& c, float amplitude, unsigned seed)
{
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> qk(-amplitude, amplitude), value(-1, 1);
    for(auto& x : c.q)
        x = static_cast<float>(RoundE4M3(qk(generator)));
    for(auto& x : c.k)
        x = static_cast<float>(RoundE4M3(qk(generator)));
    for(auto& x : c.v)
        x = static_cast<float>(RoundE4M3(value(generator)));
    c.sequence.q_descale   = 0.75f;
    c.sequence.k_descale   = 1.25f;
    c.sequence.v_descale   = 0.625f;
    c.sequence.score_scale = 1.0f / std::sqrt(static_cast<float>(c.sequence.query_dimension));
}

// SparseMarker V (as the page-tile gtests): V = 1 on the keys of the first, quarter, middle,
// three-quarter and last 64-key pages, 0 elsewhere, so each output component is the softmax mass
// of those pages and a fault involving one of them stays large at any K.
void SparseMarkerV(Case& c)
{
    const std::size_t pages = (c.sequence.key_length + 63) / 64;
    const std::size_t Dv    = c.sequence.value_dimension;
    for(std::size_t key = 0; key < c.sequence.key_length; ++key)
    {
        const std::size_t page = key / 64;
        const bool marked      = page == 0 || page == pages / 4 || page == (pages - 1) / 2 ||
                            page == 3 * pages / 4 || page == pages - 1;
        for(std::size_t d = 0; d < Dv; ++d)
            c.v[key * Dv + d] = marked ? 1.0f : 0.0f;
    }
}

DeviceInputs CorrectDevice(const Case& c)
{
    return {c.sequence.q_descale, c.sequence.k_descale, c.sequence.v_descale, {}};
}

ref::AccuracyContract Fp8Contract(bool keep_subnormals)
{
    ref::QuantizedP quantized;
    quantized.keeps_subnormals = keep_subnormals;
    return {7, 128, quantized};
}

// The FP8 page tile writes no LSE.
ref::Output OutputWithoutLse(const Case& c)
{
    auto output = c.Output();
    output.lse.reset();
    return output;
}

ref::Metrics CompareFp8(const Case& c, const ref::Reference& reference, bool keep_subnormals)
{
    return ref::Compare(
        {c.sequence}, reference, {OutputWithoutLse(c)}, {}, Fp8Contract(keep_subnormals));
}

void QuantizedPContract()
{
    ref::QuantizedP kept, flushed;
    flushed.keeps_subnormals = false;
    Near(kept.UnitRoundoff(), 0x1p-4, 0, "e4m3 half ULP");
    Near(kept.SubnormalThreshold(), 0x1p-14, 0, "e4m3 normal range of 256 * P");
    Near(kept.SubnormalLoss(), 0x1p-18, 0, "half an e4m3 subnormal step of 256 * P");
    Near(flushed.SubnormalLoss(), 0x1p-14, 0, "a flushed key loses at most the threshold");
    Near(RoundE4M3(1.0625), 1.0, 0, "e4m3 tie rounds to the even mantissa");
    Near(RoundE4M3(1.1875), 1.25, 0, "e4m3 tie rounds to the even mantissa (up)");
    Near(RoundE4M3(0x1p-8), 0x1p-8, 0, "e4m3 subnormal kept");
    Near(RoundE4M3(0x1p-8, false), 0, 0, "e4m3 subnormal flushed");
    Near(RoundE4M3(1000), 448, 0, "e4m3 saturates at its largest finite value");
    Check(!Fp8Contract(true).RequiresStrict(), "the FP8 contract gates on the component ratio");

    Case c(4, 300, 1, 1, 64, 8);
    FillFp8(c, 2.5f, 1);
    const auto kept_ref = ref::Compute({c.sequence}, 128, kept);
    ModelFp8Device(c, kept, CorrectDevice(c));
    Check(CompareFp8(c, kept_ref, true).contract_passed, "the modelled device passes");
    const auto no_model = ref::Compute({c.sequence});
    Reject([&] { CompareFp8(c, no_model, true); },
           "a reference without the quantized-P model is rejected");
    Reject([&] { CompareFp8(c, kept_ref, false); },
           "a reference computed for another subnormal mode is rejected");
    auto fp16_out          = Fp8Contract(true);
    fp16_out.mantissa_bits = 10;
    Reject([&] { ref::Compare({c.sequence}, kept_ref, {OutputWithoutLse(c)}, {}, fp16_out); },
           "FP8 P is tested with BF16 O only");
    auto odd_scale               = Fp8Contract(true);
    odd_scale.quantized_p->scale = 200;
    const auto odd_ref           = ref::Compute({c.sequence}, 128, odd_scale.quantized_p);
    Reject([&] { ref::Compare({c.sequence}, odd_ref, {OutputWithoutLse(c)}, {}, odd_scale); },
           "the P scale must be a power of two");
    auto missing = kept_ref;
    missing.rows[0].subnormal_contribution.clear();
    Reject([&] { CompareFp8(c, missing, true); },
           "missing subnormal metadata must not silently pass");
    Reject([&] { ref::Compare({c.sequence}, kept_ref, {c.Output()}, {}, Fp8Contract(true)); },
           "the FP8 P model has no LSE");
    auto with_sink        = c.sequence;
    with_sink.sink_logits = {0.0f};
    const auto sink_ref   = ref::Compute({with_sink}, 128, kept);
    Reject(
        [&] { ref::Compare({with_sink}, sink_ref, {OutputWithoutLse(c)}, {}, Fp8Contract(true)); },
        "the FP8 P model has no sink");

    // Correct FP8 output exceeds the 16-bit absolute cap 2^-6 on short rows: two keys with V = -1
    // and +1, 256 * P of the second key at 136.5, just above the e4m3 midpoint 136, so it converts
    // to 144 (5.5 % high). Its own component limit bounds it.
    Case pair(1, 2, 1, 1, 1, 1);
    pair.q[0]                 = 1;
    pair.k[0]                 = 0;
    pair.k[1]                 = -1;
    pair.v[0]                 = -1;
    pair.v[1]                 = 1;
    pair.sequence.score_scale = 0.6289f; // 256 * exp(-0.6289) = 136.50
    const auto pair_ref       = ref::Compute({pair.sequence}, 128, kept);
    ModelFp8Device(pair, kept, CorrectDevice(pair));
    const auto pair_metrics = CompareFp8(pair, pair_ref, true);
    std::cout << "FP8 two-key row: max abs error=" << pair_metrics.max_abs_error
              << " max limit=" << pair_metrics.max_component_limit
              << " max ratio=" << pair_metrics.max_component_ratio << '\n';
    Check(pair_metrics.contract_passed && pair_metrics.max_abs_error > 0x1p-6,
          "correct FP8 output above the 16-bit cap passes its component limit");
}

// The modelled device passes the contract, with subnormals kept or flushed, from rows of two keys
// to 73728 keys (the longest chunk8k row: a 64K prefix and an 8K chunk).
void Fp8ModelPassesContract()
{
    for(const bool keep : {true, false})
    {
        ref::QuantizedP quantized;
        quantized.keeps_subnormals = keep;
        for(const std::size_t keys : {2, 8, 61, 428, 3000, 32768, 73728})
        {
            Case c(4, keys, 2, 1, 64, 8);
            FillFp8(c, 2.5f, static_cast<unsigned>(keys));
            const auto reference = ref::Compute({c.sequence}, 128, quantized);
            ModelFp8Device(c, quantized, CorrectDevice(c));
            const auto metrics = CompareFp8(c, reference, keep);
            std::cout << "FP8 model keep=" << keep << " K=" << keys
                      << " max ratio=" << metrics.max_component_ratio
                      << " max abs error=" << metrics.max_abs_error
                      << " max limit=" << metrics.max_component_limit << '\n';
            Check(metrics.contract_passed && metrics.passed && metrics.max_component_ratio <= 1,
                  "the modelled FP8 device output passes the quantized-P contract");
        }
    }
}

// Every negative control fails through the component ratio where the contract is used: K = 3000
// (Window long rows), 8192, 32768 (Deterministic) and 73728 (the longest chunk8k row). v descale
// x1.25 and one wrong page on SparseMarker V at every K; on spread scores with random V, q / k
// descale x1.5 at every K and x1.25 up to K = 8192 (from K = 32768 on, random V averages a 25 %
// score-scale error into the budget).
void Fp8ControlsFireAtLongK()
{
    for(const bool keep : {true, false})
    {
        ref::QuantizedP quantized;
        quantized.keeps_subnormals = keep;
        for(const std::size_t keys : {3000, 8192, 32768, 73728})
        {
            auto run = [&](Case& c, const DeviceInputs& device) {
                const auto reference = ref::Compute({c.sequence}, 128, quantized);
                ModelFp8Device(c, quantized, CorrectDevice(c));
                Check(CompareFp8(c, reference, keep).contract_passed,
                      "the unperturbed model passes on the control's data");
                ModelFp8Device(c, quantized, device);
                return CompareFp8(c, reference, keep);
            };
            Case spread(4, keys, 1, 1, 64, 8);
            FillFp8(spread, 2.5f, static_cast<unsigned>(keys) + 7);
            auto scaled = [&](float q_gain, float k_gain) {
                auto device = CorrectDevice(spread);
                device.q_descale *= q_gain;
                device.k_descale *= k_gain;
                return run(spread, device);
            };
            const auto q_metrics  = scaled(1.25f, 1);
            const auto k_metrics  = scaled(1, 1.25f);
            const auto q2_metrics = scaled(1.5f, 1);
            const auto k2_metrics = scaled(1, 1.5f);
            DeviceInputs device;

            Case marker(4, keys, 1, 1, 64, 8);
            FillFp8(marker, 2.5f, static_cast<unsigned>(keys) + 11);
            SparseMarkerV(marker);
            device = CorrectDevice(marker);
            device.v_descale *= 1.25f;
            const auto v_metrics = run(marker, device);
            // The middle marked page's table entry points at the (unmarked) second page.
            device = CorrectDevice(marker);
            device.physical_key.resize(keys);
            const std::size_t middle = ((keys + 63) / 64 - 1) / 2;
            for(std::size_t key = 0; key < keys; ++key)
                device.physical_key[key] =
                    key / 64 == middle && 64 + key % 64 < keys ? 64 + key % 64 : key;
            const auto page_metrics = run(marker, device);
            std::cout << "FP8 controls keep=" << keep << " K=" << keys
                      << " ratio q x1.25=" << q_metrics.max_component_ratio
                      << " k x1.25=" << k_metrics.max_component_ratio
                      << " q x1.5=" << q2_metrics.max_component_ratio
                      << " k x1.5=" << k2_metrics.max_component_ratio
                      << " v x1.25=" << v_metrics.max_component_ratio
                      << " page=" << page_metrics.max_component_ratio << '\n';
            auto fired = [](const ref::Metrics& metrics) {
                return !metrics.contract_passed && metrics.max_component_ratio > 1;
            };
            Check(fired(q2_metrics) && fired(k2_metrics) && fired(v_metrics) && fired(page_metrics),
                  "an FP8 negative control fails the component ratio at long K");
            if(keys <= 8192)
                Check(fired(q_metrics) && fired(k_metrics),
                      "a 25 % score-scale error fails the component ratio up to K = 8192");
        }
    }
}

} // namespace

TEST(FmhaReference, ZeroAndConstant) { EXPECT_NO_THROW(ZeroAndConstant()); }
TEST(FmhaReference, MasksAndEmptyRows) { EXPECT_NO_THROW(MasksAndEmptyRows()); }
TEST(FmhaReference, RaggedGqaAndStrides) { EXPECT_NO_THROW(RaggedGqaAndStrides()); }
TEST(FmhaReference, ChunkedAgainstIndependentDense)
{
    EXPECT_NO_THROW(ChunkedAgainstIndependentDense());
}
TEST(FmhaReference, LongSampledAndFiniteChecks) { EXPECT_NO_THROW(LongSampledAndFiniteChecks()); }
TEST(FmhaReference, NegativeControlsAndMetricDefinitions)
{
    EXPECT_NO_THROW(NegativeControlsAndMetricDefinitions());
}
TEST(FmhaReference, InvalidMetadataAndExtremeScores)
{
    EXPECT_NO_THROW(InvalidMetadataAndExtremeScores());
}
TEST(FmhaReference, VirtualSink) { EXPECT_NO_THROW(VirtualSink()); }
TEST(FmhaReference, ScaleAwareAccuracyContract) { EXPECT_NO_THROW(ScaleAwareAccuracyContract()); }
TEST(FmhaReference, LargestKeyGainSensitivity) { EXPECT_NO_THROW(LargestKeyGainSensitivity()); }
TEST(FmhaReference, QuantizedPContract) { EXPECT_NO_THROW(QuantizedPContract()); }
TEST(FmhaReference, Fp8ModelPassesContract) { EXPECT_NO_THROW(Fp8ModelPassesContract()); }
TEST(FmhaReference, Fp8ControlsFireAtLongK) { EXPECT_NO_THROW(Fp8ControlsFireAtLongK()); }
