// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Qwen3.8 (Hugging Face `qwen3_5`) decoder pieces expressed as hipDNN graphs and run on the CPU
// reference executor, checked against a direct transcription of the model's math:
//   - a Gated DeltaNet layer (in-proj, causal conv1d + SiLU, gated delta rule with QK L2 norm and
//     the softplus gate, gated RMSNorm, out-proj, SwiGLU MLP),
//   - a gated full-attention layer (QK RMSNorm, partial leading-segment RoPE, paged KV cache
//     write, causal GQA SDPA over the paged cache, sigmoid output gate, SwiGLU MLP),
//   - token embedding and greedy sampling (gather, lm_head, softmax, argmax),
//   - a routed MoE block (softmax router, top-k, renormalisation, routing indices, grouped
//     GEMMs, weighted combine, sigmoid-gated shared expert).
// The layer tests prefill five tokens and then decode the sixth from the carried conv state,
// recurrent state and paged KV cache, so they also check that prefill and decode compose.
// Shapes are tiny versions of the 27B config (head-count ratios, partial rotary factor 1/2
// instead of 1/4); graph boundaries sit where a tensor changes rank, because hipDNN has no
// reshape node and a buffer is re-described with new dims between graphs.

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using hipdnn_frontend::DataType;
using hipdnn_frontend::DiagonalAlignment;
using hipdnn_frontend::NormFwdPhase;
using hipdnn_frontend::PointwiseMode;
using hipdnn_frontend::ReductionMode;
using hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::TensorAttributes;
using TensorPtr = std::shared_ptr<TensorAttributes>;
using Floats = std::vector<float>;
using Ints = std::vector<int32_t>;

constexpr float K_EPS = 1e-6F;
constexpr float K_TOLERANCE = 2e-4F;

// ---------------------------------------------------------------------------------------------
// Small helpers.

std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

// Deterministic values in [-scale, scale).
Floats randomValues(size_t count, uint32_t seed, float scale)
{
    Floats values(count);
    uint32_t state = seed * 2654435761U + 12345U;
    for(auto& value : values)
    {
        state = state * 1664525U + 1013904223U;
        const auto unit = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
        value = (2.0F * unit - 1.0F) * scale;
    }
    return values;
}

float sigmoid(float x)
{
    return 1.0F / (1.0F + std::exp(-x));
}

float silu(float x)
{
    return x * sigmoid(x);
}

float softplus(float x)
{
    return std::log1p(std::exp(x));
}

// x [rows, in] @ w [in, out].
Floats matmul(const Floats& x, const Floats& w, int64_t rows, int64_t in, int64_t out)
{
    Floats y(static_cast<size_t>(rows * out), 0.0F);
    for(int64_t r = 0; r < rows; ++r)
    {
        for(int64_t o = 0; o < out; ++o)
        {
            double acc = 0.0;
            for(int64_t i = 0; i < in; ++i)
            {
                acc += static_cast<double>(x[static_cast<size_t>(r * in + i)])
                       * static_cast<double>(w[static_cast<size_t>(i * out + o)]);
            }
            y[static_cast<size_t>(r * out + o)] = static_cast<float>(acc);
        }
    }
    return y;
}

// RMSNorm over consecutive groups of `width` values.
Floats rmsnorm(const Floats& x, const Floats& gamma, int64_t width)
{
    Floats y(x.size());
    for(size_t base = 0; base < x.size(); base += static_cast<size_t>(width))
    {
        double sum = 0.0;
        for(int64_t i = 0; i < width; ++i)
        {
            const auto v = static_cast<double>(x[base + static_cast<size_t>(i)]);
            sum += v * v;
        }
        const auto inv = static_cast<float>(
            1.0 / std::sqrt(sum / static_cast<double>(width) + static_cast<double>(K_EPS)));
        for(int64_t i = 0; i < width; ++i)
        {
            const auto idx = base + static_cast<size_t>(i);
            y[idx] = x[idx] * inv * gamma[static_cast<size_t>(i)];
        }
    }
    return y;
}

Floats add(const Floats& a, const Floats& b)
{
    Floats y(a.size());
    std::transform(a.begin(), a.end(), b.begin(), y.begin(), std::plus<>());
    return y;
}

Floats sliceRows(const Floats& x, int64_t width, int64_t first, int64_t count)
{
    return {x.begin() + static_cast<std::ptrdiff_t>(first * width),
            x.begin() + static_cast<std::ptrdiff_t>((first + count) * width)};
}

void expectClose(const Floats& actual, const Floats& expected, const std::string& what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const auto bound = K_TOLERANCE * std::max(1.0F, std::abs(expected[i]));
        ASSERT_NEAR(actual[i], expected[i], bound) << what << " element " << i;
    }
}

// One hipDNN graph plus the host buffers bound to its non-virtual tensors.
class GraphRun
{
public:
    explicit GraphRun(const std::string& name)
        : _graph(std::make_shared<Graph>())
    {
        _graph->set_name(name);
        _graph->set_io_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT);
    }

    Graph& graph()
    {
        return *_graph;
    }

    TensorPtr input(const std::string& name,
                    const std::vector<int64_t>& dims,
                    void* data,
                    DataType dataType = DataType::FLOAT,
                    std::vector<int64_t> strides = {})
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(_nextUid++)
            .set_name(name)
            .set_data_type(dataType)
            .set_dim(dims)
            .set_stride(strides.empty() ? packedStrides(dims) : std::move(strides));
        _pack[tensor->get_uid()] = data;
        return tensor;
    }

    // A compile-time scalar (no variant-pack binding).
    TensorPtr scalar(const std::string& name, float value)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(_nextUid++).set_name(name).set_value(value);
        return tensor;
    }

    // Binds a node output (or a caller-created output tensor) to a host buffer.
    void output(const TensorPtr& tensor,
                const std::vector<int64_t>& dims,
                void* data,
                DataType dataType = DataType::FLOAT,
                std::vector<int64_t> strides = {})
    {
        tensor->set_uid(_nextUid++)
            .set_output(true)
            .set_data_type(dataType)
            .set_dim(dims)
            .set_stride(strides.empty() ? packedStrides(dims) : std::move(strides));
        _pack[tensor->get_uid()] = data;
    }

    void run()
    {
        const auto validation = _graph->validate();
        ASSERT_TRUE(validation.is_good()) << validation.get_message();
        auto [serialized, error] = _graph->to_binary();
        ASSERT_TRUE(error.is_good()) << error.get_message();
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor{}.execute(
            serialized.data(), serialized.size(), _pack);
    }

private:
    std::shared_ptr<Graph> _graph;
    std::unordered_map<int64_t, void*> _pack;
    int64_t _nextUid = 1;
};

TensorPtr rmsnormNode(GraphRun& run, const TensorPtr& x, const TensorPtr& gamma)
{
    hipdnn_frontend::graph::RMSNormAttributes attributes;
    attributes.set_epsilon(run.scalar("eps", K_EPS));
    attributes.set_forward_phase(NormFwdPhase::INFERENCE);
    return run.graph().rmsnorm(x, gamma, attributes)[0];
}

TensorPtr matmulNode(GraphRun& run, const TensorPtr& a, const TensorPtr& b)
{
    return run.graph().matmul(a, b, hipdnn_frontend::graph::MatmulAttributes{});
}

TensorPtr pointwiseNode(GraphRun& run, PointwiseMode mode, const TensorPtr& a)
{
    hipdnn_frontend::graph::PointwiseAttributes attributes;
    attributes.set_mode(mode);
    if(mode == PointwiseMode::SWISH_FWD)
    {
        attributes.set_swish_beta(1.0F);
    }
    return run.graph().pointwise(a, attributes);
}

TensorPtr pointwiseNode(GraphRun& run, PointwiseMode mode, const TensorPtr& a, const TensorPtr& b)
{
    hipdnn_frontend::graph::PointwiseAttributes attributes;
    attributes.set_mode(mode);
    return run.graph().pointwise(a, b, attributes);
}

// SwiGLU MLP weights shared by both layer types.
struct Mlp
{
    int64_t hidden;
    int64_t inter;
    Floats postNorm;
    Floats gate;
    Floats up;
    Floats down;

    Mlp(int64_t hiddenSize, int64_t interSize, uint32_t seed)
        : hidden(hiddenSize)
        , inter(interSize)
        , postNorm(randomValues(static_cast<size_t>(hiddenSize), seed, 0.5F))
        , gate(randomValues(static_cast<size_t>(hiddenSize * interSize), seed + 1, 0.3F))
        , up(randomValues(static_cast<size_t>(hiddenSize * interSize), seed + 2, 0.3F))
        , down(randomValues(static_cast<size_t>(interSize * hiddenSize), seed + 3, 0.3F))
    {
        for(auto& g : postNorm)
        {
            g += 1.0F; // zero-centred (1 + w) weight folded into gamma
        }
    }

    // x1 + down(silu(gate(n)) * up(n)), n = rmsnorm(x1).
    Floats reference(const Floats& x1, int64_t tokens) const
    {
        const auto n = rmsnorm(x1, postNorm, hidden);
        const auto g = matmul(n, gate, tokens, hidden, inter);
        const auto u = matmul(n, up, tokens, hidden, inter);
        Floats act(g.size());
        for(size_t i = 0; i < g.size(); ++i)
        {
            act[i] = silu(g[i]) * u[i];
        }
        return add(x1, matmul(act, down, tokens, inter, hidden));
    }

    // Appends residual + MLP to `run`; returns the layer output tensor (virtual).
    TensorPtr build(GraphRun& run, const TensorPtr& x1)
    {
        auto norm = rmsnormNode(run, x1, run.input("post_norm", {1, 1, hidden}, postNorm.data()));
        auto g = matmulNode(run, norm, run.input("w_gate", {1, hidden, inter}, gate.data()));
        auto u = matmulNode(run, norm, run.input("w_up", {1, hidden, inter}, up.data()));
        auto act = pointwiseNode(
            run, PointwiseMode::MUL, pointwiseNode(run, PointwiseMode::SWISH_FWD, g), u);
        auto mlpOut = matmulNode(run, act, run.input("w_down", {1, inter, hidden}, down.data()));
        return pointwiseNode(run, PointwiseMode::ADD, x1, mlpOut);
    }
};

// ---------------------------------------------------------------------------------------------
// Gated DeltaNet (linear attention) layer.

struct GdnLayer
{
    static constexpr int64_t HIDDEN = 16;
    static constexpr int64_t NUM_K_HEADS = 2;
    static constexpr int64_t NUM_V_HEADS = 4;
    static constexpr int64_t HEAD_K = 4;
    static constexpr int64_t HEAD_V = 4;
    static constexpr int64_t CONV_W = 4;
    static constexpr int64_t KEY_DIM = NUM_K_HEADS * HEAD_K;
    static constexpr int64_t VALUE_DIM = NUM_V_HEADS * HEAD_V;

    Floats inNorm = randomValues(HIDDEN, 101, 0.5F);
    Floats wq = randomValues(HIDDEN * KEY_DIM, 102, 0.4F);
    Floats wk = randomValues(HIDDEN * KEY_DIM, 103, 0.4F);
    Floats wv = randomValues(HIDDEN * VALUE_DIM, 104, 0.4F);
    Floats wz = randomValues(HIDDEN * VALUE_DIM, 105, 0.4F);
    Floats wb = randomValues(HIDDEN * NUM_V_HEADS, 106, 0.4F);
    Floats wa = randomValues(HIDDEN * NUM_V_HEADS, 107, 0.4F);
    Floats convQ = randomValues(KEY_DIM * CONV_W, 108, 0.6F);
    Floats convK = randomValues(KEY_DIM * CONV_W, 109, 0.6F);
    Floats convV = randomValues(VALUE_DIM * CONV_W, 110, 0.6F);
    Floats aLog = randomValues(NUM_V_HEADS, 111, 0.5F);
    Floats dtBias = randomValues(NUM_V_HEADS, 112, 0.5F);
    Floats gatedNorm = randomValues(HEAD_V, 113, 0.5F);
    Floats wo = randomValues(VALUE_DIM * HIDDEN, 114, 0.4F);
    Mlp mlp{HIDDEN, 24, 115};

    GdnLayer()
    {
        for(auto& g : inNorm)
        {
            g += 1.0F;
        }
        for(auto& g : gatedNorm)
        {
            g += 1.0F;
        }
    }

    // Hugging Face Qwen3_5GatedDeltaNet + decoder layer, recurrent form, one sequence.
    Floats reference(const Floats& x, int64_t tokens) const
    {
        const auto h = rmsnorm(x, inNorm, HIDDEN);
        const auto conv = [&](const Floats& raw, const Floats& weight, int64_t channels) {
            Floats out(raw.size());
            for(int64_t t = 0; t < tokens; ++t)
            {
                for(int64_t c = 0; c < channels; ++c)
                {
                    float acc = 0.0F;
                    for(int64_t j = 0; j < CONV_W; ++j)
                    {
                        const auto src = t - (CONV_W - 1) + j;
                        if(src >= 0)
                        {
                            acc += weight[static_cast<size_t>(c * CONV_W + j)]
                                   * raw[static_cast<size_t>(src * channels + c)];
                        }
                    }
                    out[static_cast<size_t>(t * channels + c)] = silu(acc);
                }
            }
            return out;
        };
        const auto q = conv(matmul(h, wq, tokens, HIDDEN, KEY_DIM), convQ, KEY_DIM);
        const auto k = conv(matmul(h, wk, tokens, HIDDEN, KEY_DIM), convK, KEY_DIM);
        const auto v = conv(matmul(h, wv, tokens, HIDDEN, VALUE_DIM), convV, VALUE_DIM);
        const auto z = matmul(h, wz, tokens, HIDDEN, VALUE_DIM);
        const auto b = matmul(h, wb, tokens, HIDDEN, NUM_V_HEADS);
        const auto a = matmul(h, wa, tokens, HIDDEN, NUM_V_HEADS);

        const auto l2norm = [](Floats vec) {
            double sum = 0.0;
            for(const auto value : vec)
            {
                sum += static_cast<double>(value) * static_cast<double>(value);
            }
            const auto inv = static_cast<float>(1.0 / std::sqrt(sum + 1e-6));
            for(auto& value : vec)
            {
                value *= inv;
            }
            return vec;
        };

        Floats gated(static_cast<size_t>(tokens * VALUE_DIM));
        const auto scale = 1.0F / std::sqrt(static_cast<float>(HEAD_K));
        for(int64_t head = 0; head < NUM_V_HEADS; ++head)
        {
            const auto kHead = head / (NUM_V_HEADS / NUM_K_HEADS); // repeat_interleave
            // HF keeps the state as [K, V].
            Floats state(static_cast<size_t>(HEAD_K * HEAD_V), 0.0F);
            for(int64_t t = 0; t < tokens; ++t)
            {
                Floats qt(HEAD_K);
                Floats kt(HEAD_K);
                for(int64_t i = 0; i < HEAD_K; ++i)
                {
                    qt[static_cast<size_t>(i)]
                        = q[static_cast<size_t>(t * KEY_DIM + kHead * HEAD_K + i)];
                    kt[static_cast<size_t>(i)]
                        = k[static_cast<size_t>(t * KEY_DIM + kHead * HEAD_K + i)];
                }
                qt = l2norm(qt);
                kt = l2norm(kt);
                const auto beta = sigmoid(b[static_cast<size_t>(t * NUM_V_HEADS + head)]);
                const auto g = -std::exp(aLog[static_cast<size_t>(head)])
                               * softplus(a[static_cast<size_t>(t * NUM_V_HEADS + head)]
                                          + dtBias[static_cast<size_t>(head)]);
                const auto decay = std::exp(g);
                for(auto& s : state)
                {
                    s *= decay;
                }
                Floats out(HEAD_V, 0.0F);
                for(int64_t j = 0; j < HEAD_V; ++j)
                {
                    float kvMem = 0.0F;
                    for(int64_t i = 0; i < HEAD_K; ++i)
                    {
                        kvMem += state[static_cast<size_t>(i * HEAD_V + j)]
                                 * kt[static_cast<size_t>(i)];
                    }
                    const auto delta
                        = (v[static_cast<size_t>(t * VALUE_DIM + head * HEAD_V + j)] - kvMem)
                          * beta;
                    for(int64_t i = 0; i < HEAD_K; ++i)
                    {
                        state[static_cast<size_t>(i * HEAD_V + j)]
                            += kt[static_cast<size_t>(i)] * delta;
                    }
                }
                for(int64_t j = 0; j < HEAD_V; ++j)
                {
                    float acc = 0.0F;
                    for(int64_t i = 0; i < HEAD_K; ++i)
                    {
                        acc += state[static_cast<size_t>(i * HEAD_V + j)]
                               * qt[static_cast<size_t>(i)] * scale;
                    }
                    out[static_cast<size_t>(j)] = acc;
                }
                // Qwen3_5RMSNormGated: rmsnorm(o) * weight * silu(z).
                const auto normed = rmsnorm(out, gatedNorm, HEAD_V);
                for(int64_t j = 0; j < HEAD_V; ++j)
                {
                    const auto idx = static_cast<size_t>(t * VALUE_DIM + head * HEAD_V + j);
                    gated[idx] = normed[static_cast<size_t>(j)] * silu(z[idx]);
                }
            }
        }
        const auto x1 = add(x, matmul(gated, wo, tokens, VALUE_DIM, HIDDEN));
        return mlp.reference(x1, tokens);
    }

    // Carried decode state: conv windows and the recurrent state [1, HV, V, K].
    struct State
    {
        Floats convQ = Floats(KEY_DIM * (CONV_W - 1));
        Floats convK = Floats(KEY_DIM * (CONV_W - 1));
        Floats convV = Floats(VALUE_DIM * (CONV_W - 1));
        Floats recurrent = Floats(NUM_V_HEADS * HEAD_V * HEAD_K);
    };

    // Runs the layer on `tokens` new tokens of one sequence through four hipDNN graphs. With
    // `carried`, the conv and recurrent states start from it; both are updated in place.
    Floats run(Floats x, int64_t tokens, State& state, bool carried)
    {
        Ints cuSeqlens{0, static_cast<int32_t>(tokens)};

        // 1. Input RMSNorm and projections.
        Floats qRaw(static_cast<size_t>(tokens * KEY_DIM));
        Floats kRaw(qRaw.size());
        Floats vRaw(static_cast<size_t>(tokens * VALUE_DIM));
        Floats z(vRaw.size());
        Floats b(static_cast<size_t>(tokens * NUM_V_HEADS));
        Floats a(b.size());
        {
            GraphRun g("gdn_in_proj");
            auto xt = g.input("x", {1, tokens, HIDDEN}, x.data());
            auto h = rmsnormNode(g, xt, g.input("in_norm", {1, 1, HIDDEN}, inNorm.data()));
            const auto project = [&](const std::string& name, Floats& w, int64_t out, Floats& dst) {
                auto y = matmulNode(g, h, g.input(name, {1, HIDDEN, out}, w.data()));
                g.output(y, {1, tokens, out}, dst.data());
            };
            project("wq", wq, KEY_DIM, qRaw);
            project("wk", wk, KEY_DIM, kRaw);
            project("wv", wv, VALUE_DIM, vRaw);
            project("wz", wz, VALUE_DIM, z);
            project("wb", wb, NUM_V_HEADS, b);
            project("wa", wa, NUM_V_HEADS, a);
            g.run();
        }

        // 2. Causal conv1d + SiLU on q, k and v (one graph each: a graph must be connected),
        //    carrying the conv windows.
        Floats qConv(qRaw.size());
        Floats kConv(kRaw.size());
        Floats vConv(vRaw.size());
        const auto conv = [&](const std::string& name,
                              Floats& raw,
                              Floats& weight,
                              Floats& windows,
                              int64_t channels,
                              Floats& dst) {
            GraphRun g("gdn_conv_" + name);
            hipdnn_frontend::graph::CausalConv1dAttributes attributes;
            attributes.set_activation(hipdnn_frontend::CausalConv1dActivation::SILU);
            attributes.set_output_final_state(true);
            if(carried)
            {
                attributes.set_initial_state(
                    g.input(name + "_state_in", {1, channels, CONV_W - 1}, windows.data()));
            }
            auto [y, finalState] = g.graph().causal_conv1d(
                g.input(name, {tokens, channels}, raw.data()),
                g.input(name + "_w", {channels, CONV_W}, weight.data()),
                g.input("cu_seqlens", {2}, cuSeqlens.data(), DataType::INT32),
                attributes);
            g.output(y, {tokens, channels}, dst.data());
            // In place: the final window is written over the initial one.
            g.output(finalState, {1, channels, CONV_W - 1}, windows.data());
            g.run();
        };
        conv("q", qRaw, convQ, state.convQ, KEY_DIM, qConv);
        conv("k", kRaw, convK, state.convK, KEY_DIM, kConv);
        conv("v", vRaw, convV, state.convV, VALUE_DIM, vConv);

        // 3. Gated delta rule + gated RMSNorm.
        Floats gated(vRaw.size());
        {
            GraphRun g("gdn_core");
            hipdnn_frontend::graph::GdnAttributes attributes;
            attributes.set_use_qk_l2norm(true)
                .set_use_beta_sigmoid(true)
                .set_safe_gate(true)
                .set_gate_domain(hipdnn_frontend::GateDomain::LOG)
                .set_output_final_state(true)
                .set_a_log(g.input("a_log", {NUM_V_HEADS}, aLog.data()))
                .set_dt_bias(g.input("dt_bias", {NUM_V_HEADS}, dtBias.data()));
            if(carried)
            {
                attributes.set_initial_state(
                    g.input("state_in", {1, NUM_V_HEADS, HEAD_V, HEAD_K}, state.recurrent.data()));
            }
            auto [o, finalState, checkpoints]
                = g.graph().gdn(g.input("q", {tokens, NUM_K_HEADS, HEAD_K}, qConv.data()),
                                g.input("k", {tokens, NUM_K_HEADS, HEAD_K}, kConv.data()),
                                g.input("v", {tokens, NUM_V_HEADS, HEAD_V}, vConv.data()),
                                g.input("g", {tokens, NUM_V_HEADS}, a.data()),
                                g.input("beta", {tokens, NUM_V_HEADS}, b.data()),
                                g.input("cu_seqlens", {2}, cuSeqlens.data(), DataType::INT32),
                                attributes);
            (void)checkpoints;
            g.output(finalState, {1, NUM_V_HEADS, HEAD_V, HEAD_K}, state.recurrent.data());
            auto normed
                = rmsnormNode(g, o, g.input("gated_norm", {1, 1, HEAD_V}, gatedNorm.data()));
            auto gate = pointwiseNode(
                g, PointwiseMode::SWISH_FWD, g.input("z", {tokens, NUM_V_HEADS, HEAD_V}, z.data()));
            auto y = pointwiseNode(g, PointwiseMode::MUL, normed, gate);
            g.output(y, {tokens, NUM_V_HEADS, HEAD_V}, gated.data());
            g.run();
        }

        // 4. Out-projection, residual and MLP.
        Floats out(x.size());
        {
            GraphRun g("gdn_out");
            auto xt = g.input("x", {1, tokens, HIDDEN}, x.data());
            auto proj = matmulNode(g,
                                   g.input("gated", {1, tokens, VALUE_DIM}, gated.data()),
                                   g.input("wo", {1, VALUE_DIM, HIDDEN}, wo.data()));
            auto x1 = pointwiseNode(g, PointwiseMode::ADD, xt, proj);
            g.output(mlp.build(g, x1), {1, tokens, HIDDEN}, out.data());
            g.run();
        }
        return out;
    }
};

// ---------------------------------------------------------------------------------------------
// Gated full-attention layer with a paged KV cache.

struct AttentionLayer
{
    static constexpr int64_t HIDDEN = 16;
    static constexpr int64_t NUM_Q_HEADS = 4;
    static constexpr int64_t NUM_KV_HEADS = 2;
    static constexpr int64_t HEAD_DIM = 8;
    static constexpr int64_t ROTARY_DIM = 4; // leading segment of the head dim
    static constexpr int64_t BLOCK_SIZE = 4;
    static constexpr int64_t NUM_BLOCKS = 4;
    static constexpr int64_t MAX_BLOCKS_PER_SEQ = 2;
    static constexpr float ROPE_THETA = 10000.0F;
    static constexpr int64_t Q_DIM = NUM_Q_HEADS * HEAD_DIM;
    static constexpr int64_t KV_DIM = NUM_KV_HEADS * HEAD_DIM;

    Floats inNorm = randomValues(HIDDEN, 201, 0.5F);
    Floats wq = randomValues(HIDDEN * Q_DIM, 202, 0.4F);
    Floats wgate = randomValues(HIDDEN * Q_DIM, 203, 0.4F);
    Floats wk = randomValues(HIDDEN * KV_DIM, 204, 0.4F);
    Floats wv = randomValues(HIDDEN * KV_DIM, 205, 0.4F);
    Floats qNorm = randomValues(HEAD_DIM, 206, 0.5F);
    Floats kNorm = randomValues(HEAD_DIM, 207, 0.5F);
    Floats wo = randomValues(Q_DIM * HIDDEN, 208, 0.4F);
    Mlp mlp{HIDDEN, 24, 209};
    // Sequence 0 owns physical blocks 2 then 0.
    Ints pageTable{2, 0};

    AttentionLayer()
    {
        for(auto* gamma : {&inNorm, &qNorm, &kNorm})
        {
            for(auto& g : *gamma)
            {
                g += 1.0F;
            }
        }
    }

    static float angle(int64_t position, int64_t i)
    {
        const auto invFreq
            = std::pow(ROPE_THETA, -2.0F * static_cast<float>(i) / static_cast<float>(ROTARY_DIM));
        return static_cast<float>(position) * invFreq;
    }

    // Raw RoPE angles [1, 1, tokens, ROTARY_DIM] = cat(freqs, freqs) for positions first..first+tokens.
    static Floats freqs(int64_t first, int64_t tokens)
    {
        Floats f(static_cast<size_t>(tokens * ROTARY_DIM));
        for(int64_t t = 0; t < tokens; ++t)
        {
            for(int64_t i = 0; i < ROTARY_DIM / 2; ++i)
            {
                const auto theta = angle(first + t, i);
                f[static_cast<size_t>(t * ROTARY_DIM + i)] = theta;
                f[static_cast<size_t>(t * ROTARY_DIM + i + ROTARY_DIM / 2)] = theta;
            }
        }
        return f;
    }

    // Hugging Face Qwen3_5Attention (gated, partial rotary) + decoder layer, causal, one sequence.
    Floats reference(const Floats& x, int64_t tokens) const
    {
        const auto h = rmsnorm(x, inNorm, HIDDEN);
        auto q = rmsnorm(matmul(h, wq, tokens, HIDDEN, Q_DIM), qNorm, HEAD_DIM);
        auto k = rmsnorm(matmul(h, wk, tokens, HIDDEN, KV_DIM), kNorm, HEAD_DIM);
        const auto v = matmul(h, wv, tokens, HIDDEN, KV_DIM);
        const auto gate = matmul(h, wgate, tokens, HIDDEN, Q_DIM);
        const auto rotate = [&](Floats& t, int64_t heads) {
            for(int64_t pos = 0; pos < tokens; ++pos)
            {
                for(int64_t head = 0; head < heads; ++head)
                {
                    auto* row = &t[static_cast<size_t>((pos * heads + head) * HEAD_DIM)];
                    for(int64_t i = 0; i < ROTARY_DIM / 2; ++i)
                    {
                        const auto c = std::cos(angle(pos, i));
                        const auto s = std::sin(angle(pos, i));
                        const auto x0 = row[i];
                        const auto x1 = row[i + ROTARY_DIM / 2];
                        row[i] = x0 * c - x1 * s;
                        row[i + ROTARY_DIM / 2] = x1 * c + x0 * s;
                    }
                }
            }
        };
        rotate(q, NUM_Q_HEADS);
        rotate(k, NUM_KV_HEADS);

        Floats attn(static_cast<size_t>(tokens * Q_DIM));
        const auto scale = 1.0F / std::sqrt(static_cast<float>(HEAD_DIM));
        for(int64_t head = 0; head < NUM_Q_HEADS; ++head)
        {
            const auto kvHead = head / (NUM_Q_HEADS / NUM_KV_HEADS);
            for(int64_t t = 0; t < tokens; ++t)
            {
                Floats scores(static_cast<size_t>(t + 1));
                float maxScore = -INFINITY;
                for(int64_t s = 0; s <= t; ++s)
                {
                    float dot = 0.0F;
                    for(int64_t d = 0; d < HEAD_DIM; ++d)
                    {
                        dot += q[static_cast<size_t>(t * Q_DIM + head * HEAD_DIM + d)]
                               * k[static_cast<size_t>(s * KV_DIM + kvHead * HEAD_DIM + d)];
                    }
                    scores[static_cast<size_t>(s)] = dot * scale;
                    maxScore = std::max(maxScore, dot * scale);
                }
                float sum = 0.0F;
                for(auto& score : scores)
                {
                    score = std::exp(score - maxScore);
                    sum += score;
                }
                for(int64_t d = 0; d < HEAD_DIM; ++d)
                {
                    float acc = 0.0F;
                    for(int64_t s = 0; s <= t; ++s)
                    {
                        acc += scores[static_cast<size_t>(s)]
                               * v[static_cast<size_t>(s * KV_DIM + kvHead * HEAD_DIM + d)];
                    }
                    const auto idx = static_cast<size_t>(t * Q_DIM + head * HEAD_DIM + d);
                    attn[idx] = acc / sum * sigmoid(gate[idx]);
                }
            }
        }
        const auto x1 = add(x, matmul(attn, wo, tokens, Q_DIM, HIDDEN));
        return mlp.reference(x1, tokens);
    }

    struct Cache
    {
        Floats k = Floats(NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM, 0.0F);
        Floats v = Floats(NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM, 0.0F);
    };

    // Runs `tokens` new tokens at positions first..first+tokens of one sequence: the new K/V go
    // into the paged cache, and attention reads every cached position.
    Floats run(Floats x, int64_t first, int64_t tokens, Cache& cache)
    {
        // 1. Input RMSNorm and projections.
        Floats q(static_cast<size_t>(tokens * Q_DIM));
        Floats gate(q.size());
        Floats k(static_cast<size_t>(tokens * KV_DIM));
        Floats v(k.size());
        {
            GraphRun g("attn_in_proj");
            auto xt = g.input("x", {1, tokens, HIDDEN}, x.data());
            auto h = rmsnormNode(g, xt, g.input("in_norm", {1, 1, HIDDEN}, inNorm.data()));
            const auto project = [&](const std::string& name, Floats& w, int64_t out, Floats& dst) {
                auto y = matmulNode(g, h, g.input(name, {1, HIDDEN, out}, w.data()));
                g.output(y, {1, tokens, out}, dst.data());
            };
            project("wq", wq, Q_DIM, q);
            project("wgate", wgate, Q_DIM, gate);
            project("wk", wk, KV_DIM, k);
            project("wv", wv, KV_DIM, v);
            g.run();
        }

        // 2. QK-norm, partial RoPE, paged KV write and causal GQA SDPA, all in [B, H, S, D]
        //    dims over the token-major projection buffers.
        Floats attn(q.size());
        {
            GraphRun g("attn_core");
            auto angles = freqs(first, tokens);
            Ints slots(static_cast<size_t>(tokens));
            for(int64_t t = 0; t < tokens; ++t)
            {
                const auto position = first + t;
                slots[static_cast<size_t>(t)] = static_cast<int32_t>(
                    pageTable[static_cast<size_t>(position / BLOCK_SIZE)] * BLOCK_SIZE
                    + position % BLOCK_SIZE);
            }
            Ints seqLenQ{static_cast<int32_t>(tokens)};
            Ints seqLenKv{static_cast<int32_t>(first + tokens)};

            const auto bshd = [&](int64_t heads) {
                return std::vector<int64_t>{
                    tokens * heads * HEAD_DIM, HEAD_DIM, heads * HEAD_DIM, 1};
            };
            auto f = g.input("freqs", {1, 1, tokens, ROTARY_DIM}, angles.data());
            const auto normRope = [&](const std::string& name,
                                      Floats& data,
                                      int64_t heads,
                                      Floats& gamma) {
                auto t = g.input(
                    name, {1, heads, tokens, HEAD_DIM}, data.data(), DataType::FLOAT, bshd(heads));
                auto normed
                    = rmsnormNode(g, t, g.input(name + "_norm", {1, 1, 1, HEAD_DIM}, gamma.data()));
                hipdnn_frontend::graph::RopeAttributes rope;
                rope.set_rope_dim(ROTARY_DIM).set_rope_offset(0);
                return g.graph().rope(normed, f, rope);
            };
            auto qRot = normRope("q", q, NUM_Q_HEADS, qNorm);
            auto kRot = normRope("k", k, NUM_KV_HEADS, kNorm);

            const std::vector<int64_t> containerDims{
                NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM};
            auto slotMapping = g.input("slot_mapping", {1, tokens}, slots.data(), DataType::INT32);
            const auto store =
                [&](const std::string& name, const TensorPtr& src, Floats& container) {
                    auto dst = std::make_shared<TensorAttributes>();
                    dst->set_name(name);
                    g.output(dst, containerDims, container.data());
                    return g.graph().paged_cache_store(
                        src, slotMapping, dst, hipdnn_frontend::graph::PagedCacheStoreAttributes{});
                };
            auto kCache = store("k_cache", kRot, cache.k);
            auto vCache = store("v_cache",
                                g.input("v",
                                        {1, NUM_KV_HEADS, tokens, HEAD_DIM},
                                        v.data(),
                                        DataType::FLOAT,
                                        bshd(NUM_KV_HEADS)),
                                cache.v);

            hipdnn_frontend::graph::SdpaAttributes sdpa;
            sdpa.set_attn_scale(1.0F / std::sqrt(static_cast<float>(HEAD_DIM)))
                .set_causal_mask_bottom_right(true)
                .set_generate_stats(false)
                .set_seq_len_q(g.input("seq_len_q", {1, 1, 1, 1}, seqLenQ.data(), DataType::INT32))
                .set_seq_len_kv(
                    g.input("seq_len_kv", {1, 1, 1, 1}, seqLenKv.data(), DataType::INT32))
                .set_paged_attention_k_table(g.input("page_table_k",
                                                     {1, 1, MAX_BLOCKS_PER_SEQ, 1},
                                                     pageTable.data(),
                                                     DataType::INT32))
                .set_paged_attention_v_table(g.input("page_table_v",
                                                     {1, 1, MAX_BLOCKS_PER_SEQ, 1},
                                                     pageTable.data(),
                                                     DataType::INT32))
                .set_paged_attention_max_seq_len_kv(MAX_BLOCKS_PER_SEQ * BLOCK_SIZE);
            auto [o, stats] = g.graph().sdpa(qRot, kCache, vCache, sdpa);
            (void)stats;
            g.output(o,
                     {1, NUM_Q_HEADS, tokens, HEAD_DIM},
                     attn.data(),
                     DataType::FLOAT,
                     bshd(NUM_Q_HEADS));
            g.run();
        }

        // 3. Sigmoid output gate, out-projection, residual and MLP.
        Floats out(x.size());
        {
            GraphRun g("attn_out");
            auto xt = g.input("x", {1, tokens, HIDDEN}, x.data());
            auto gated
                = pointwiseNode(g,
                                PointwiseMode::MUL,
                                g.input("attn", {1, tokens, Q_DIM}, attn.data()),
                                pointwiseNode(g,
                                              PointwiseMode::SIGMOID_FWD,
                                              g.input("gate", {1, tokens, Q_DIM}, gate.data())));
            auto proj = matmulNode(g, gated, g.input("wo", {1, Q_DIM, HIDDEN}, wo.data()));
            auto x1 = pointwiseNode(g, PointwiseMode::ADD, xt, proj);
            g.output(mlp.build(g, x1), {1, tokens, HIDDEN}, out.data());
            g.run();
        }
        return out;
    }
};

constexpr int64_t K_PREFILL = 5;
constexpr int64_t K_TOTAL = 6;

} // namespace

TEST(IntegrationQwen38LayerGraphsFp32, GatedDeltaNetLayerPrefillThenDecode)
{
    GdnLayer layer;
    const auto x = randomValues(static_cast<size_t>(K_TOTAL * GdnLayer::HIDDEN), 7, 1.0F);
    const auto expected = layer.reference(x, K_TOTAL);

    GdnLayer::State state;
    const auto prefill
        = layer.run(sliceRows(x, GdnLayer::HIDDEN, 0, K_PREFILL), K_PREFILL, state, false);
    expectClose(prefill, sliceRows(expected, GdnLayer::HIDDEN, 0, K_PREFILL), "GDN prefill");

    const auto decode = layer.run(sliceRows(x, GdnLayer::HIDDEN, K_PREFILL, 1), 1, state, true);
    expectClose(decode, sliceRows(expected, GdnLayer::HIDDEN, K_PREFILL, 1), "GDN decode");
}

TEST(IntegrationQwen38LayerGraphsFp32, GatedAttentionLayerPrefillThenPagedDecode)
{
    AttentionLayer layer;
    const auto x = randomValues(static_cast<size_t>(K_TOTAL * AttentionLayer::HIDDEN), 8, 1.0F);
    const auto expected = layer.reference(x, K_TOTAL);

    AttentionLayer::Cache cache;
    const auto prefill
        = layer.run(sliceRows(x, AttentionLayer::HIDDEN, 0, K_PREFILL), 0, K_PREFILL, cache);
    expectClose(
        prefill, sliceRows(expected, AttentionLayer::HIDDEN, 0, K_PREFILL), "attention prefill");

    const auto decode
        = layer.run(sliceRows(x, AttentionLayer::HIDDEN, K_PREFILL, 1), K_PREFILL, 1, cache);
    expectClose(
        decode, sliceRows(expected, AttentionLayer::HIDDEN, K_PREFILL, 1), "attention decode");
}

TEST(IntegrationQwen38LayerGraphsFp32, EmbeddingAndGreedySampling)
{
    constexpr int64_t VOCAB = 11;
    constexpr int64_t HIDDEN = 8;
    constexpr int64_t TOKENS = 3;
    auto table = randomValues(VOCAB * HIDDEN, 301, 1.0F);
    auto finalNorm = randomValues(HIDDEN, 302, 0.5F);
    auto lmHead = randomValues(HIDDEN * VOCAB, 303, 1.0F);
    Ints ids{7, 0, 10};

    Floats embedded(TOKENS * HIDDEN);
    Floats probs(TOKENS * VOCAB);
    Ints next(TOKENS);
    GraphRun g("embed_and_sample");
    auto x = g.graph().gather(g.input("embed_table", {VOCAB, HIDDEN}, table.data()),
                              g.input("ids", {1, TOKENS}, ids.data(), DataType::INT32),
                              hipdnn_frontend::graph::GatherAttributes{});
    g.output(x, {1, TOKENS, HIDDEN}, embedded.data());
    auto normed = rmsnormNode(g, x, g.input("final_norm", {1, 1, HIDDEN}, finalNorm.data()));
    auto logits = matmulNode(g, normed, g.input("lm_head", {1, HIDDEN, VOCAB}, lmHead.data()));
    auto [p, stats] = g.graph().softmax(logits, hipdnn_frontend::graph::SoftmaxAttributes{});
    (void)stats;
    g.output(p, {1, TOKENS, VOCAB}, probs.data());
    hipdnn_frontend::graph::TopkAttributes topk;
    topk.set_k(1);
    auto [values, indices] = g.graph().topk(logits, topk);
    values->set_is_virtual(true);
    g.output(indices, {1, TOKENS, 1}, next.data(), DataType::INT32);
    g.run();

    Floats tokenRows;
    for(const auto id : ids)
    {
        tokenRows.insert(tokenRows.end(),
                         table.begin() + static_cast<std::ptrdiff_t>(id * HIDDEN),
                         table.begin() + static_cast<std::ptrdiff_t>((id + 1) * HIDDEN));
    }
    expectClose(embedded, tokenRows, "embedding");
    const auto expectedLogits
        = matmul(rmsnorm(tokenRows, finalNorm, HIDDEN), lmHead, TOKENS, HIDDEN, VOCAB);
    for(int64_t t = 0; t < TOKENS; ++t)
    {
        const auto row = sliceRows(expectedLogits, VOCAB, t, 1);
        const auto best = std::max_element(row.begin(), row.end()) - row.begin();
        EXPECT_EQ(next[static_cast<size_t>(t)], best) << "token " << t;
        const auto maxLogit = row[static_cast<size_t>(best)];
        float sum = 0.0F;
        for(const auto value : row)
        {
            sum += std::exp(value - maxLogit);
        }
        Floats expectedProbs(row.size());
        for(size_t i = 0; i < row.size(); ++i)
        {
            expectedProbs[i] = std::exp(row[i] - maxLogit) / sum;
        }
        expectClose(sliceRows(probs, VOCAB, t, 1), expectedProbs, "softmax row");
    }
}

TEST(IntegrationQwen38LayerGraphsFp32, RoutedMoeBlockWithSharedExpert)
{
    constexpr int64_t HIDDEN = 8;
    constexpr int64_t EXPERTS = 4;
    constexpr int64_t TOP_K = 2;
    constexpr int64_t INTER = 6;
    constexpr int64_t TOKENS = 5;
    constexpr int64_t ROUTED = TOKENS * TOP_K;
    auto h = randomValues(TOKENS * HIDDEN, 401, 1.0F);
    auto router = randomValues(HIDDEN * EXPERTS, 402, 1.0F);
    auto gateW = randomValues(EXPERTS * HIDDEN * INTER, 403, 0.4F);
    auto upW = randomValues(EXPERTS * HIDDEN * INTER, 404, 0.4F);
    auto downW = randomValues(EXPERTS * INTER * HIDDEN, 405, 0.4F);
    auto sharedGate = randomValues(HIDDEN, 406, 0.4F);
    auto sharedG = randomValues(HIDDEN * INTER, 407, 0.4F);
    auto sharedU = randomValues(HIDDEN * INTER, 408, 0.4F);
    auto sharedD = randomValues(INTER * HIDDEN, 409, 0.4F);

    // 1. Router: softmax -> top-k -> renormalise.
    Floats weights(ROUTED);
    Ints topkIds(ROUTED);
    {
        GraphRun g("moe_router");
        auto ht = g.input("h", {1, TOKENS, HIDDEN}, h.data());
        auto logits = matmulNode(g, ht, g.input("router", {1, HIDDEN, EXPERTS}, router.data()));
        auto [probs, stats]
            = g.graph().softmax(logits, hipdnn_frontend::graph::SoftmaxAttributes{});
        (void)stats;
        hipdnn_frontend::graph::TopkAttributes topk;
        topk.set_k(TOP_K);
        auto [values, indices] = g.graph().topk(probs, topk);
        g.output(indices, {1, TOKENS, TOP_K}, topkIds.data(), DataType::INT32);
        hipdnn_frontend::graph::ReductionAttributes sum;
        sum.set_mode(ReductionMode::ADD);
        auto total = g.graph().reduction(values, sum);
        total->set_dim({1, TOKENS, 1}).set_stride({TOKENS, 1, 1});
        auto normalised = pointwiseNode(g, PointwiseMode::DIV, values, total);
        g.output(normalised, {1, TOKENS, TOP_K}, weights.data());
        g.run();
    }

    // 2. Routing indices.
    Ints offsets(EXPERTS + 1);
    Ints tokenIndex(ROUTED);
    Ints tokenKs(ROUTED);
    {
        GraphRun g("moe_routing");
        hipdnn_frontend::graph::MoeRoutingAttributes attributes;
        attributes.set_num_experts(EXPERTS);
        auto [firstTokenOffset, index, ks] = g.graph().moe_routing(
            g.input("topk_ids", {TOKENS, TOP_K}, topkIds.data(), DataType::INT32), attributes);
        g.output(firstTokenOffset, {EXPERTS + 1, 1, 1}, offsets.data(), DataType::INT32);
        g.output(index, {1, ROUTED, 1}, tokenIndex.data(), DataType::INT32);
        g.output(ks, {1, ROUTED, 1}, tokenKs.data(), DataType::INT32);
        g.run();
    }

    // 3. Expert SwiGLU: GATHER gate/up, SCATTER down back to (token, k) order.
    Floats expertOut(ROUTED * HIDDEN);
    {
        GraphRun g("moe_experts");
        auto ht = g.input("h", {1, TOKENS, HIDDEN}, h.data());
        auto off
            = g.input("first_token_offset", {EXPERTS + 1, 1, 1}, offsets.data(), DataType::INT32);
        auto idx = g.input("token_index", {1, ROUTED, 1}, tokenIndex.data(), DataType::INT32);
        auto ks = g.input("token_ks", {1, ROUTED, 1}, tokenKs.data(), DataType::INT32);
        const auto grouped = [&](const TensorPtr& token,
                                 const TensorPtr& weight,
                                 hipdnn_frontend::MoeGroupedMatmulMode mode,
                                 const TensorPtr& tokenKsTensor) {
            hipdnn_frontend::graph::MoeGroupedMatmulAttributes attributes;
            attributes.set_mode(mode).set_top_k(TOP_K);
            return g.graph().moe_grouped_matmul(token, weight, off, idx, tokenKsTensor, attributes);
        };
        auto gate = grouped(ht,
                            g.input("w_gate", {EXPERTS, HIDDEN, INTER}, gateW.data()),
                            hipdnn_frontend::MoeGroupedMatmulMode::GATHER,
                            nullptr);
        auto up = grouped(ht,
                          g.input("w_up", {EXPERTS, HIDDEN, INTER}, upW.data()),
                          hipdnn_frontend::MoeGroupedMatmulMode::GATHER,
                          nullptr);
        auto act = pointwiseNode(
            g, PointwiseMode::MUL, pointwiseNode(g, PointwiseMode::SWISH_FWD, gate), up);
        auto down = grouped(act,
                            g.input("w_down", {EXPERTS, INTER, HIDDEN}, downW.data()),
                            hipdnn_frontend::MoeGroupedMatmulMode::SCATTER,
                            ks);
        g.output(down, {1, ROUTED, HIDDEN}, expertOut.data());
        g.run();
    }

    // 4. Weighted sum over k of the routed rows ([token, k] order).
    Floats routedSum(TOKENS * HIDDEN);
    {
        GraphRun g("moe_combine");
        auto weighted
            = pointwiseNode(g,
                            PointwiseMode::MUL,
                            g.input("expert_out", {1, TOKENS, TOP_K, HIDDEN}, expertOut.data()),
                            g.input("weights", {1, TOKENS, TOP_K, 1}, weights.data()));
        hipdnn_frontend::graph::ReductionAttributes sum;
        sum.set_mode(ReductionMode::ADD);
        auto routed = g.graph().reduction(weighted, sum);
        g.output(routed, {1, TOKENS, 1, HIDDEN}, routedSum.data());
        g.run();
    }

    // 5. Shared expert gated by sigmoid(h . w), plus the routed sum re-described from
    //    [1, T, 1, H] to [1, T, H].
    Floats out(TOKENS * HIDDEN);
    {
        GraphRun g("moe_shared_and_output");
        auto ht = g.input("h", {1, TOKENS, HIDDEN}, h.data());
        auto sg = matmulNode(g, ht, g.input("w_sg", {1, HIDDEN, INTER}, sharedG.data()));
        auto su = matmulNode(g, ht, g.input("w_su", {1, HIDDEN, INTER}, sharedU.data()));
        auto sact = pointwiseNode(
            g, PointwiseMode::MUL, pointwiseNode(g, PointwiseMode::SWISH_FWD, sg), su);
        auto shared = matmulNode(g, sact, g.input("w_sd", {1, INTER, HIDDEN}, sharedD.data()));
        auto gateLogit
            = matmulNode(g, ht, g.input("w_shared_gate", {1, HIDDEN, 1}, sharedGate.data()));
        auto gatedShared = pointwiseNode(
            g, PointwiseMode::MUL, shared, pointwiseNode(g, PointwiseMode::SIGMOID_FWD, gateLogit));
        auto y = pointwiseNode(g,
                               PointwiseMode::ADD,
                               g.input("routed", {1, TOKENS, HIDDEN}, routedSum.data()),
                               gatedShared);
        g.output(y, {1, TOKENS, HIDDEN}, out.data());
        g.run();
    }

    // Reference: Qwen3_5MoeSparseMoeBlock with norm_topk_prob.
    {
        const auto logits = matmul(h, router, TOKENS, HIDDEN, EXPERTS);
        Floats expected(TOKENS * HIDDEN, 0.0F);
        for(int64_t t = 0; t < TOKENS; ++t)
        {
            const auto row = sliceRows(logits, EXPERTS, t, 1);
            const auto maxLogit = *std::max_element(row.begin(), row.end());
            // Unnormalised softmax: the top-k renormalisation divides the softmax sum out.
            Floats p(EXPERTS);
            for(int64_t e = 0; e < EXPERTS; ++e)
            {
                p[static_cast<size_t>(e)] = std::exp(row[static_cast<size_t>(e)] - maxLogit);
            }
            std::vector<int64_t> order(EXPERTS);
            for(int64_t e = 0; e < EXPERTS; ++e)
            {
                order[static_cast<size_t>(e)] = e;
            }
            std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
                return p[static_cast<size_t>(lhs)] > p[static_cast<size_t>(rhs)];
            });
            const auto topSum = p[static_cast<size_t>(order[0])] + p[static_cast<size_t>(order[1])];
            const auto token = sliceRows(h, HIDDEN, t, 1);
            for(int64_t slot = 0; slot < TOP_K; ++slot)
            {
                const auto e = order[static_cast<size_t>(slot)];
                EXPECT_EQ(topkIds[static_cast<size_t>(t * TOP_K + slot)], e);
                const auto weight = p[static_cast<size_t>(e)] / topSum;
                const auto expertSlice = [&](const Floats& w, int64_t rows, int64_t cols) {
                    return Floats(w.begin() + static_cast<std::ptrdiff_t>(e * rows * cols),
                                  w.begin() + static_cast<std::ptrdiff_t>((e + 1) * rows * cols));
                };
                const auto ge = matmul(token, expertSlice(gateW, HIDDEN, INTER), 1, HIDDEN, INTER);
                const auto ue = matmul(token, expertSlice(upW, HIDDEN, INTER), 1, HIDDEN, INTER);
                Floats act(INTER);
                for(int64_t i = 0; i < INTER; ++i)
                {
                    act[static_cast<size_t>(i)]
                        = silu(ge[static_cast<size_t>(i)]) * ue[static_cast<size_t>(i)];
                }
                const auto de = matmul(act, expertSlice(downW, INTER, HIDDEN), 1, INTER, HIDDEN);
                for(int64_t i = 0; i < HIDDEN; ++i)
                {
                    expected[static_cast<size_t>(t * HIDDEN + i)]
                        += weight * de[static_cast<size_t>(i)];
                }
            }
        }

        const auto sg2 = matmul(h, sharedG, TOKENS, HIDDEN, INTER);
        const auto su2 = matmul(h, sharedU, TOKENS, HIDDEN, INTER);
        Floats sact2(sg2.size());
        for(size_t i = 0; i < sg2.size(); ++i)
        {
            sact2[i] = silu(sg2[i]) * su2[i];
        }
        const auto expectedShared = matmul(sact2, sharedD, TOKENS, INTER, HIDDEN);
        const auto gateLogits = matmul(h, sharedGate, TOKENS, HIDDEN, 1);
        for(int64_t t = 0; t < TOKENS; ++t)
        {
            for(int64_t i = 0; i < HIDDEN; ++i)
            {
                const auto idx = static_cast<size_t>(t * HIDDEN + i);
                expected[idx] += expectedShared[idx] * sigmoid(gateLogits[static_cast<size_t>(t)]);
            }
        }
        expectClose(out, expected, "MoE block output");
    }
}
