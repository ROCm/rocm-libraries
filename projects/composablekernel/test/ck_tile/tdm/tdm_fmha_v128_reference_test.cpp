// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "tdm_fmha_v128_reference.hpp"

#include <iostream>
#include <string>

namespace {

namespace ref = tdm_v128_test::reference;

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

} // namespace

int main()
{
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"zero_and_constant", ZeroAndConstant},
        {"masks_and_empty_rows", MasksAndEmptyRows},
        {"ragged_gqa_and_strides", RaggedGqaAndStrides},
        {"chunked_against_independent_dense", ChunkedAgainstIndependentDense},
        {"long_sampled_and_finite_checks", LongSampledAndFiniteChecks},
        {"negative_controls_and_metric_definitions", NegativeControlsAndMetricDefinitions},
        {"invalid_metadata_and_extreme_scores", InvalidMetadataAndExtremeScores}};
    std::size_t failures = 0;
    for(const auto& [name, test] : tests)
    {
        try
        {
            test();
            std::cout << "PASS " << name << '\n';
        }
        catch(const std::exception& error)
        {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << "Host FP32 oracle tests: " << tests.size() - failures << '/' << tests.size()
              << " passed; no GPU kernels exercised\n";
    return failures == 0 ? 0 : 1;
}
