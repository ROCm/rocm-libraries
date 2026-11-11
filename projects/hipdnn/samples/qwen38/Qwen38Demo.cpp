// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Qwen3.8-27B (Hugging Face qwen3_5, text path) run end to end through hipDNN graphs. Every layer
// op (embedding gather, RMSNorm, matmul, causal conv1d, gated delta rule, RoPE, KV-cache store,
// causal GQA SDPA, pointwise, top-k) is a hipDNN graph node; this program only loads weights,
// builds graphs and moves buffers between them. Greedy decoding.
//
// --backend=cpu (default): multi-node graphs on the CPU reference executor, fp32, paged KV cache.
// --backend=gpu: bf16 weights and activations on the device, one graph per op built through a
// hipDNN Handle (engine plugins), dense per-layer KV cache; graphs no engine accepts run on the
// CPU reference executor with device<->host copies. --profile synchronizes and times each op.
//
// Usage: hipdnn_sample_qwen38 <snapshot-dir> <comma-separated prompt token ids> <max new tokens>
//                             [layers] [--backend=cpu|gpu] [--profile]
// Prints "TOKEN <id>" per generated token; run_qwen38.py tokenizes the prompt and detokenizes.

#include "../utils/Helpers.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Handle.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
// hipdnn_test_sdk is used here for more than validation: --backend=cpu, and the GPU backend's
// per-op fallback for graphs no engine accepts, execute on hipDNN's CPU reference executor.
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{

using hipdnn_frontend::DataType;
using hipdnn_frontend::NormFwdPhase;
using hipdnn_frontend::PointwiseMode;
using hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::TensorAttributes;
using TensorPtr = std::shared_ptr<TensorAttributes>;
using Floats = std::vector<float>;
using Ints = std::vector<int32_t>;

// ---------------------------------------------------------------------------------------------
// Config (Qwen/Qwen3.8-27B text_config).
constexpr int64_t HIDDEN = 5120;
constexpr int64_t INTER = 17408;
constexpr int64_t VOCAB = 248320;
constexpr int64_t LAYERS = 64;
constexpr int64_t FULL_ATTENTION_INTERVAL = 4;
constexpr float RMS_EPS = 1e-6F;
// Gated DeltaNet.
constexpr int64_t NK = 16;
constexpr int64_t NV = 48;
constexpr int64_t DK = 128;
constexpr int64_t DV = 128;
constexpr int64_t CONV_W = 4;
constexpr int64_t KEY_DIM = NK * DK;
constexpr int64_t VALUE_DIM = NV * DV;
// Gated attention.
constexpr int64_t NQ = 24;
constexpr int64_t NKV = 4;
constexpr int64_t HD = 256;
constexpr int64_t ROT = 64; // partial_rotary_factor 0.25, leading dims
constexpr double ROPE_THETA = 10000000.0;
constexpr int64_t Q_DIM = NQ * HD;
constexpr int64_t KV_DIM = NKV * HD;
// Paged KV cache.
constexpr int64_t BLOCK = 16;

double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------------------------
// Safetensors (bf16) loading into fp32.

class Checkpoint
{
public:
    explicit Checkpoint(const std::string& dir)
        : _dir(dir)
    {
        std::ifstream index(dir + "/model.safetensors.index.json");
        auto json = nlohmann::json::parse(index);
        for(const auto& [name, file] : json["weight_map"].items())
        {
            _fileOf[name] = file.get<std::string>();
        }
    }

    ~Checkpoint()
    {
        for(auto& [file, mapping] : _maps)
        {
            munmap(mapping.base, mapping.size);
        }
    }

    // Reads every shard once on `workers` detached background threads so the page cache fills
    // ahead of the loader (they own copies of what they use and die with the process).
    void prefetch(size_t workers) const
    {
        auto files = std::make_shared<std::vector<std::string>>();
        for(const auto& [name, file] : _fileOf)
        {
            if(std::find(files->begin(), files->end(), file) == files->end())
            {
                files->push_back(file);
            }
        }
        std::sort(files->begin(), files->end());
        auto next = std::make_shared<std::atomic<size_t>>(0);
        for(size_t w = 0; w < workers; ++w)
        {
            std::thread([files, next, dir = _dir] {
                std::vector<char> buffer(size_t{16} << 20);
                for(size_t i = (*next)++; i < files->size(); i = (*next)++)
                {
                    const int fd = open((dir + "/" + (*files)[i]).c_str(), O_RDONLY);
                    if(fd < 0)
                    {
                        continue;
                    }
                    while(read(fd, buffer.data(), buffer.size()) > 0)
                    {
                    }
                    close(fd);
                }
            }).detach();
        }
    }

    // Converts a bf16 tensor to fp32 (parallel); `shape` receives its dims.
    Floats load(const std::string& name, std::vector<int64_t>* shape = nullptr)
    {
        const auto& file = _fileOf.at(name);
        auto& mapping = map(file);
        const auto& meta = mapping.header.at(name);
        if(meta["dtype"] != "BF16")
        {
            throw std::runtime_error(name + " is not BF16");
        }
        const auto begin = meta["data_offsets"][0].get<size_t>();
        const auto end = meta["data_offsets"][1].get<size_t>();
        if(shape != nullptr)
        {
            *shape = meta["shape"].get<std::vector<int64_t>>();
        }
        const auto count = (end - begin) / 2;
        const auto* src = reinterpret_cast<const uint16_t*>(mapping.data + begin);
        Floats out(count);
        const auto threads = std::max<size_t>(1, std::min<size_t>(64, count / (1 << 20)));
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for(size_t t = 0; t < threads; ++t)
        {
            pool.emplace_back([&, t] {
                for(size_t i = t * count / threads; i < (t + 1) * count / threads; ++i)
                {
                    const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
                    std::memcpy(&out[i], &bits, sizeof(float));
                }
            });
        }
        for(auto& thread : pool)
        {
            thread.join();
        }
        return out;
    }

    // The tensor's bf16 payload in place (mmapped); `count` receives its element count.
    const uint16_t*
        raw(const std::string& name, size_t* count, std::vector<int64_t>* shape = nullptr)
    {
        auto& mapping = map(_fileOf.at(name));
        const auto& meta = mapping.header.at(name);
        if(meta["dtype"] != "BF16")
        {
            throw std::runtime_error(name + " is not BF16");
        }
        const auto begin = meta["data_offsets"][0].get<size_t>();
        const auto end = meta["data_offsets"][1].get<size_t>();
        if(shape != nullptr)
        {
            *shape = meta["shape"].get<std::vector<int64_t>>();
        }
        *count = (end - begin) / 2;
        return reinterpret_cast<const uint16_t*>(mapping.data + begin);
    }

private:
    struct Mapping
    {
        void* base = nullptr;
        size_t size = 0;
        const uint8_t* data = nullptr;
        nlohmann::json header;
    };

    Mapping& map(const std::string& file)
    {
        auto it = _maps.find(file);
        if(it != _maps.end())
        {
            return it->second;
        }
        const auto path = _dir + "/" + file;
        const int fd = open(path.c_str(), O_RDONLY);
        if(fd < 0)
        {
            throw std::runtime_error("cannot open " + path);
        }
        struct stat st
        {
        };
        fstat(fd, &st);
        Mapping mapping;
        mapping.size = static_cast<size_t>(st.st_size);
        mapping.base = mmap(nullptr, mapping.size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        const auto* bytes = static_cast<const uint8_t*>(mapping.base);
        uint64_t headerSize = 0;
        std::memcpy(&headerSize, bytes, sizeof(headerSize));
        mapping.header = nlohmann::json::parse(bytes + 8, bytes + 8 + headerSize);
        mapping.data = bytes + 8 + headerSize;
        return _maps.emplace(file, std::move(mapping)).first->second;
    }

    std::string _dir;
    std::unordered_map<std::string, std::string> _fileOf;
    std::map<std::string, Mapping> _maps;
};

Floats onePlus(Floats w)
{
    for(auto& v : w)
    {
        v += 1.0F;
    }
    return w;
}

// ---------------------------------------------------------------------------------------------
// hipDNN graph helpers.

std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

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
                    const void* data,
                    DataType dataType = DataType::FLOAT,
                    std::vector<int64_t> strides = {})
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(_nextUid++)
            .set_name(name)
            .set_data_type(dataType)
            .set_dim(dims)
            .set_stride(strides.empty() ? packedStrides(dims) : std::move(strides));
        _pack[tensor->get_uid()] = const_cast<void*>(data);
        return tensor;
    }

    // A weight stored row-major as [out, in] (nn.Linear) used as the matmul B operand [1, in, out].
    TensorPtr linearWeight(const std::string& name, const float* data, int64_t in, int64_t out)
    {
        return input(name, {1, in, out}, data, DataType::FLOAT, {in * out, 1, in});
    }

    TensorPtr scalar(const std::string& name, float value)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(_nextUid++).set_name(name).set_value(value);
        return tensor;
    }

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
        if(!validation.is_good())
        {
            throw std::runtime_error(_graph->get_name() + ": " + validation.get_message());
        }
        auto [serialized, error] = _graph->to_binary();
        if(!error.is_good())
        {
            throw std::runtime_error(_graph->get_name() + ": " + error.get_message());
        }
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor{}.execute(
            serialized.data(), serialized.size(), _pack);
    }

private:
    std::shared_ptr<Graph> _graph;
    std::unordered_map<int64_t, void*> _pack;
    int64_t _nextUid = 1;
};

TensorPtr rmsnorm(GraphRun& g, const TensorPtr& x, const TensorPtr& gamma)
{
    hipdnn_frontend::graph::RMSNormAttributes attributes;
    attributes.set_epsilon(g.scalar("eps", RMS_EPS));
    attributes.set_forward_phase(NormFwdPhase::INFERENCE);
    return g.graph().rmsnorm(x, gamma, attributes)[0];
}

TensorPtr matmul(GraphRun& g, const TensorPtr& a, const TensorPtr& b)
{
    return g.graph().matmul(a, b, hipdnn_frontend::graph::MatmulAttributes{});
}

TensorPtr
    pointwise(GraphRun& g, PointwiseMode mode, const TensorPtr& a, const TensorPtr& b = nullptr)
{
    hipdnn_frontend::graph::PointwiseAttributes attributes;
    attributes.set_mode(mode);
    if(mode == PointwiseMode::SWISH_FWD)
    {
        attributes.set_swish_beta(1.0F);
    }
    return b ? g.graph().pointwise(a, b, attributes) : g.graph().pointwise(a, attributes);
}

// ---------------------------------------------------------------------------------------------
// Layers.

struct Mlp
{
    Floats postNorm;
    Floats gate;
    Floats up;
    Floats down;

    void load(Checkpoint& ckpt, const std::string& prefix)
    {
        postNorm = onePlus(ckpt.load(prefix + "post_attention_layernorm.weight"));
        gate = ckpt.load(prefix + "mlp.gate_proj.weight");
        up = ckpt.load(prefix + "mlp.up_proj.weight");
        down = ckpt.load(prefix + "mlp.down_proj.weight");
    }

    // x1 + down(silu(gate(n)) * up(n)), n = rmsnorm(x1)
    TensorPtr build(GraphRun& g, const TensorPtr& x1)
    {
        auto n = rmsnorm(g, x1, g.input("post_norm", {1, 1, HIDDEN}, postNorm.data()));
        auto gt = matmul(g, n, g.linearWeight("w_gate", gate.data(), HIDDEN, INTER));
        auto u = matmul(g, n, g.linearWeight("w_up", up.data(), HIDDEN, INTER));
        auto act = pointwise(g, PointwiseMode::MUL, pointwise(g, PointwiseMode::SWISH_FWD, gt), u);
        auto mlpOut = matmul(g, act, g.linearWeight("w_down", down.data(), INTER, HIDDEN));
        return pointwise(g, PointwiseMode::ADD, x1, mlpOut);
    }
};

struct GdnLayer
{
    Floats inNorm, qkv, z, b, a, conv, aLog, dtBias, gatedNorm, outProj;
    Mlp mlp;
    // Carried state.
    Floats convQ = Floats(KEY_DIM * (CONV_W - 1), 0.0F);
    Floats convK = Floats(KEY_DIM * (CONV_W - 1), 0.0F);
    Floats convV = Floats(VALUE_DIM * (CONV_W - 1), 0.0F);
    Floats recurrent = Floats(NV * DV * DK, 0.0F);
    bool hasState = false;

    void load(Checkpoint& ckpt, const std::string& p)
    {
        inNorm = onePlus(ckpt.load(p + "input_layernorm.weight"));
        qkv = ckpt.load(p + "linear_attn.in_proj_qkv.weight"); // rows: q 2048 | k 2048 | v 6144
        z = ckpt.load(p + "linear_attn.in_proj_z.weight");
        b = ckpt.load(p + "linear_attn.in_proj_b.weight");
        a = ckpt.load(p + "linear_attn.in_proj_a.weight");
        conv = ckpt.load(p + "linear_attn.conv1d.weight"); // [10240, 1, 4]
        aLog = ckpt.load(p + "linear_attn.A_log");
        dtBias = ckpt.load(p + "linear_attn.dt_bias");
        gatedNorm = ckpt.load(p + "linear_attn.norm.weight"); // plain weight
        outProj = ckpt.load(p + "linear_attn.out_proj.weight");
        mlp.load(ckpt, p);
    }

    Floats run(Floats& x, int64_t tokens)
    {
        Ints cuSeqlens{0, static_cast<int32_t>(tokens)};
        Floats qRaw(static_cast<size_t>(tokens * KEY_DIM));
        Floats kRaw(qRaw.size());
        Floats vRaw(static_cast<size_t>(tokens * VALUE_DIM));
        Floats zOut(vRaw.size());
        Floats bOut(static_cast<size_t>(tokens * NV));
        Floats aOut(bOut.size());
        {
            GraphRun g("gdn_in_proj");
            auto h = rmsnorm(g,
                             g.input("x", {1, tokens, HIDDEN}, x.data()),
                             g.input("in_norm", {1, 1, HIDDEN}, inNorm.data()));
            const auto project = [&](const char* name, const float* w, int64_t out, Floats& dst) {
                g.output(matmul(g, h, g.linearWeight(name, w, HIDDEN, out)),
                         {1, tokens, out},
                         dst.data());
            };
            project("wq", qkv.data(), KEY_DIM, qRaw);
            project("wk", qkv.data() + KEY_DIM * HIDDEN, KEY_DIM, kRaw);
            project("wv", qkv.data() + 2 * KEY_DIM * HIDDEN, VALUE_DIM, vRaw);
            project("wz", z.data(), VALUE_DIM, zOut);
            project("wb", b.data(), NV, bOut);
            project("wa", a.data(), NV, aOut);
            g.run();
        }

        Floats qConv(qRaw.size());
        Floats kConv(kRaw.size());
        Floats vConv(vRaw.size());
        const auto convolve = [&](const char* name,
                                  Floats& raw,
                                  const float* weight,
                                  Floats& window,
                                  int64_t channels,
                                  Floats& dst) {
            GraphRun g(std::string("gdn_conv_") + name);
            hipdnn_frontend::graph::CausalConv1dAttributes attributes;
            attributes.set_activation(hipdnn_frontend::CausalConv1dActivation::SILU);
            attributes.set_output_final_state(true);
            if(hasState)
            {
                attributes.set_initial_state(
                    g.input("state_in", {1, channels, CONV_W - 1}, window.data()));
            }
            auto [y, finalState] = g.graph().causal_conv1d(
                g.input("x", {tokens, channels}, raw.data()),
                g.input("w", {channels, CONV_W}, weight),
                g.input("cu_seqlens", {2}, cuSeqlens.data(), DataType::INT32),
                attributes);
            g.output(y, {tokens, channels}, dst.data());
            g.output(finalState, {1, channels, CONV_W - 1}, window.data());
            g.run();
        };
        convolve("q", qRaw, conv.data(), convQ, KEY_DIM, qConv);
        convolve("k", kRaw, conv.data() + KEY_DIM * CONV_W, convK, KEY_DIM, kConv);
        convolve("v", vRaw, conv.data() + 2 * KEY_DIM * CONV_W, convV, VALUE_DIM, vConv);

        Floats gated(vRaw.size());
        {
            GraphRun g("gdn_core");
            hipdnn_frontend::graph::GdnAttributes attributes;
            attributes.set_use_qk_l2norm(true)
                .set_use_beta_sigmoid(true)
                .set_safe_gate(true)
                .set_gate_domain(hipdnn_frontend::GateDomain::LOG)
                .set_output_final_state(true)
                .set_a_log(g.input("a_log", {NV}, aLog.data()))
                .set_dt_bias(g.input("dt_bias", {NV}, dtBias.data()));
            if(hasState)
            {
                attributes.set_initial_state(
                    g.input("state_in", {1, NV, DV, DK}, recurrent.data()));
            }
            auto [o, finalState, checkpoints]
                = g.graph().gdn(g.input("q", {tokens, NK, DK}, qConv.data()),
                                g.input("k", {tokens, NK, DK}, kConv.data()),
                                g.input("v", {tokens, NV, DV}, vConv.data()),
                                g.input("g", {tokens, NV}, aOut.data()),
                                g.input("beta", {tokens, NV}, bOut.data()),
                                g.input("cu_seqlens", {2}, cuSeqlens.data(), DataType::INT32),
                                attributes);
            (void)checkpoints;
            g.output(finalState, {1, NV, DV, DK}, recurrent.data());
            auto normed = rmsnorm(g, o, g.input("gated_norm", {1, 1, DV}, gatedNorm.data()));
            auto gate = pointwise(
                g, PointwiseMode::SWISH_FWD, g.input("z", {tokens, NV, DV}, zOut.data()));
            g.output(
                pointwise(g, PointwiseMode::MUL, normed, gate), {tokens, NV, DV}, gated.data());
            g.run();
        }
        hasState = true;

        Floats out(x.size());
        {
            GraphRun g("gdn_out");
            auto xt = g.input("x", {1, tokens, HIDDEN}, x.data());
            auto proj = matmul(g,
                               g.input("gated", {1, tokens, VALUE_DIM}, gated.data()),
                               g.linearWeight("wo", outProj.data(), VALUE_DIM, HIDDEN));
            g.output(mlp.build(g, pointwise(g, PointwiseMode::ADD, xt, proj)),
                     {1, tokens, HIDDEN},
                     out.data());
            g.run();
        }
        return out;
    }
};

struct AttentionLayer
{
    Floats inNorm, wq, wgate, wk, wv, qNorm, kNorm, wo;
    Mlp mlp;
    int64_t numBlocks = 0;
    Ints pageTable;
    Floats kCache, vCache;

    void load(Checkpoint& ckpt, const std::string& p, int64_t maxSeq)
    {
        inNorm = onePlus(ckpt.load(p + "input_layernorm.weight"));
        // q_proj rows are per head [q (256) | gate (256)]: split host-side.
        const auto qg = ckpt.load(p + "self_attn.q_proj.weight");
        wq.resize(static_cast<size_t>(Q_DIM * HIDDEN));
        wgate.resize(wq.size());
        for(int64_t h = 0; h < NQ; ++h)
        {
            const auto* src = qg.data() + h * 2 * HD * HIDDEN;
            std::memcpy(wq.data() + h * HD * HIDDEN, src, sizeof(float) * HD * HIDDEN);
            std::memcpy(
                wgate.data() + h * HD * HIDDEN, src + HD * HIDDEN, sizeof(float) * HD * HIDDEN);
        }
        wk = ckpt.load(p + "self_attn.k_proj.weight");
        wv = ckpt.load(p + "self_attn.v_proj.weight");
        qNorm = onePlus(ckpt.load(p + "self_attn.q_norm.weight"));
        kNorm = onePlus(ckpt.load(p + "self_attn.k_norm.weight"));
        wo = ckpt.load(p + "self_attn.o_proj.weight");
        mlp.load(ckpt, p);

        numBlocks = (maxSeq + BLOCK - 1) / BLOCK;
        // Logical block i lives in physical block numBlocks-1-i, to exercise the page table.
        for(int64_t i = 0; i < numBlocks; ++i)
        {
            pageTable.push_back(static_cast<int32_t>(numBlocks - 1 - i));
        }
        kCache.assign(static_cast<size_t>(numBlocks * NKV * BLOCK * HD), 0.0F);
        vCache.assign(kCache.size(), 0.0F);
    }

    Floats run(Floats& x, int64_t first, int64_t tokens)
    {
        Floats q(static_cast<size_t>(tokens * Q_DIM));
        Floats gate(q.size());
        Floats k(static_cast<size_t>(tokens * KV_DIM));
        Floats v(k.size());
        {
            GraphRun g("attn_in_proj");
            auto h = rmsnorm(g,
                             g.input("x", {1, tokens, HIDDEN}, x.data()),
                             g.input("in_norm", {1, 1, HIDDEN}, inNorm.data()));
            const auto project = [&](const char* name, const Floats& w, int64_t out, Floats& dst) {
                g.output(matmul(g, h, g.linearWeight(name, w.data(), HIDDEN, out)),
                         {1, tokens, out},
                         dst.data());
            };
            project("wq", wq, Q_DIM, q);
            project("wgate", wgate, Q_DIM, gate);
            project("wk", wk, KV_DIM, k);
            project("wv", wv, KV_DIM, v);
            g.run();
        }

        Floats attn(q.size());
        {
            GraphRun g("attn_core");
            // Raw RoPE angles, cat(freqs, freqs) over the 64 leading dims (text-only M-RoPE).
            Floats angles(static_cast<size_t>(tokens * ROT));
            for(int64_t t = 0; t < tokens; ++t)
            {
                for(int64_t i = 0; i < ROT / 2; ++i)
                {
                    const auto invFreq = std::pow(ROPE_THETA, -2.0 * static_cast<double>(i) / ROT);
                    const auto theta = static_cast<float>(static_cast<double>(first + t) * invFreq);
                    angles[static_cast<size_t>(t * ROT + i)] = theta;
                    angles[static_cast<size_t>(t * ROT + i + ROT / 2)] = theta;
                }
            }
            Ints slots(static_cast<size_t>(tokens));
            for(int64_t t = 0; t < tokens; ++t)
            {
                const auto pos = first + t;
                slots[static_cast<size_t>(t)] = static_cast<int32_t>(
                    pageTable[static_cast<size_t>(pos / BLOCK)] * BLOCK + pos % BLOCK);
            }
            Ints seqLenQ{static_cast<int32_t>(tokens)};
            Ints seqLenKv{static_cast<int32_t>(first + tokens)};
            const auto bshd = [&](int64_t heads) {
                return std::vector<int64_t>{tokens * heads * HD, HD, heads * HD, 1};
            };
            auto f = g.input("freqs", {1, 1, tokens, ROT}, angles.data());
            const auto normRope
                = [&](const char* name, Floats& data, int64_t heads, Floats& gamma) {
                      auto t = g.input(
                          name, {1, heads, tokens, HD}, data.data(), DataType::FLOAT, bshd(heads));
                      auto normed = rmsnorm(
                          g, t, g.input(std::string(name) + "_norm", {1, 1, 1, HD}, gamma.data()));
                      hipdnn_frontend::graph::RopeAttributes rope;
                      rope.set_rope_dim(ROT).set_rope_offset(0);
                      return g.graph().rope(normed, f, rope);
                  };
            auto qRot = normRope("q", q, NQ, qNorm);
            auto kRot = normRope("k", k, NKV, kNorm);
            auto slotMapping = g.input("slot_mapping", {1, tokens}, slots.data(), DataType::INT32);
            const auto store = [&](const char* name, const TensorPtr& src, Floats& container) {
                auto dst = std::make_shared<TensorAttributes>();
                dst->set_name(name);
                g.output(dst, {numBlocks, NKV, BLOCK, HD}, container.data());
                return g.graph().paged_cache_store(
                    src, slotMapping, dst, hipdnn_frontend::graph::PagedCacheStoreAttributes{});
            };
            auto kCacheT = store("k_cache", kRot, kCache);
            auto vCacheT
                = store("v_cache",
                        g.input("v", {1, NKV, tokens, HD}, v.data(), DataType::FLOAT, bshd(NKV)),
                        vCache);
            hipdnn_frontend::graph::SdpaAttributes sdpa;
            sdpa.set_attn_scale(1.0F / std::sqrt(static_cast<float>(HD)))
                .set_causal_mask_bottom_right(true)
                .set_generate_stats(false)
                .set_seq_len_q(g.input("seq_len_q", {1, 1, 1, 1}, seqLenQ.data(), DataType::INT32))
                .set_seq_len_kv(
                    g.input("seq_len_kv", {1, 1, 1, 1}, seqLenKv.data(), DataType::INT32))
                .set_paged_attention_k_table(
                    g.input("pt_k", {1, 1, numBlocks, 1}, pageTable.data(), DataType::INT32))
                .set_paged_attention_v_table(
                    g.input("pt_v", {1, 1, numBlocks, 1}, pageTable.data(), DataType::INT32))
                .set_paged_attention_max_seq_len_kv(static_cast<int32_t>(numBlocks * BLOCK));
            auto [o, stats] = g.graph().sdpa(qRot, kCacheT, vCacheT, sdpa);
            (void)stats;
            g.output(o, {1, NQ, tokens, HD}, attn.data(), DataType::FLOAT, bshd(NQ));
            g.run();
        }

        Floats out(x.size());
        {
            GraphRun g("attn_out");
            auto xt = g.input("x", {1, tokens, HIDDEN}, x.data());
            auto gated = pointwise(g,
                                   PointwiseMode::MUL,
                                   g.input("attn", {1, tokens, Q_DIM}, attn.data()),
                                   pointwise(g,
                                             PointwiseMode::SIGMOID_FWD,
                                             g.input("gate", {1, tokens, Q_DIM}, gate.data())));
            auto proj = matmul(g, gated, g.linearWeight("wo", wo.data(), Q_DIM, HIDDEN));
            g.output(mlp.build(g, pointwise(g, PointwiseMode::ADD, xt, proj)),
                     {1, tokens, HIDDEN},
                     out.data());
            g.run();
        }
        return out;
    }
};

// =============================================================================================
// GPU backend: weights resident on the device as bf16, bf16 activations, FP32 compute. Every op
// is its own single-node hipDNN graph (two nodes for matmul + fused SiLU), built once per shape
// through a Handle and reused across layers and steps. A graph no engine accepts runs on the
// hipDNN CPU reference executor with device<->host copies of its operands.

// HIP runtime calls use the samples' HIP_CHECK (utils/Helpers.hpp).

namespace gpu
{

constexpr DataType BF16 = DataType::BFLOAT16;
constexpr DataType F32 = DataType::FLOAT;
constexpr DataType I32 = DataType::INT32;

size_t elementSize(DataType type)
{
    switch(type)
    {
    case DataType::BFLOAT16:
    case DataType::HALF:
        return 2;
    case DataType::INT64:
    case DataType::DOUBLE:
        return 8;
    default:
        return 4;
    }
}

// Elements a strided view spans from its base pointer.
size_t spanElements(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    size_t span = 1;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] == 0)
        {
            return 0;
        }
        span += static_cast<size_t>((dims[i] - 1) * strides[i]);
    }
    return span;
}

std::string joinDims(const std::vector<int64_t>& values)
{
    std::string out;
    for(size_t i = 0; i < values.size(); ++i)
    {
        out += (i > 0 ? "x" : "") + std::to_string(values[i]);
    }
    return out;
}

std::string fixed(double value, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

class DeviceArena
{
public:
    DeviceArena() = default;
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    ~DeviceArena()
    {
        for(auto* ptr : _ptrs)
        {
            (void)hipFree(ptr);
        }
        for(int i = 0; i < 2; ++i)
        {
            if(_staging[i] != nullptr)
            {
                (void)hipEventDestroy(_copied[i]);
                (void)hipHostFree(_staging[i]);
            }
        }
        if(_stream != nullptr)
        {
            (void)hipStreamDestroy(_stream);
        }
    }

    void* alloc(size_t bytes, bool zero = false)
    {
        void* ptr = nullptr;
        HIP_CHECK(hipMalloc(&ptr, std::max<size_t>(bytes, 256)));
        if(zero)
        {
            HIP_CHECK(hipMemset(ptr, 0, bytes));
        }
        _ptrs.push_back(ptr);
        _bytes += bytes;
        return ptr;
    }

    // Host -> device through two pinned staging buffers: many threads copy (and so fault in
    // the mmapped checkpoint pages concurrently) into one buffer while the other is in flight.
    void* upload(const void* src, size_t bytes)
    {
        auto* dst = static_cast<uint8_t*>(alloc(bytes));
        if(_stream == nullptr)
        {
            HIP_CHECK(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
            for(int i = 0; i < 2; ++i)
            {
                HIP_CHECK(hipHostMalloc(&_staging[i], STAGING));
                HIP_CHECK(hipEventCreate(&_copied[i]));
                HIP_CHECK(hipEventRecord(_copied[i], _stream));
            }
        }
        constexpr size_t THREADS = 32;
        for(size_t offset = 0; offset < bytes; offset += STAGING)
        {
            const auto size = std::min(STAGING, bytes - offset);
            auto* staging = static_cast<uint8_t*>(_staging[_next]);
            HIP_CHECK(hipEventSynchronize(_copied[_next]));
            std::vector<std::thread> pool;
            const auto slice = (size + THREADS - 1) / THREADS;
            for(size_t t = 0; t < THREADS && t * slice < size; ++t)
            {
                pool.emplace_back([&, t] {
                    const auto begin = t * slice;
                    std::memcpy(staging + begin,
                                static_cast<const uint8_t*>(src) + offset + begin,
                                std::min(slice, size - begin));
                });
            }
            for(auto& thread : pool)
            {
                thread.join();
            }
            HIP_CHECK(hipMemcpyAsync(dst + offset, staging, size, hipMemcpyHostToDevice, _stream));
            HIP_CHECK(hipEventRecord(_copied[_next], _stream));
            _next ^= 1;
        }
        return dst;
    }

    // Waits for the uploads in flight (the staging buffers are reused, not the sources).
    void finishUploads()
    {
        if(_stream != nullptr)
        {
            HIP_CHECK(hipStreamSynchronize(_stream));
        }
    }

    size_t bytes() const
    {
        return _bytes;
    }

private:
    static constexpr size_t STAGING = size_t{256} << 20;
    std::vector<void*> _ptrs;
    size_t _bytes = 0;
    hipStream_t _stream = nullptr;
    std::array<void*, 2> _staging = {nullptr, nullptr};
    std::array<hipEvent_t, 2> _copied = {};
    int _next = 0;
};

// One operand of an op graph, in the order the op's run() takes pointers.
struct Slot
{
    int64_t uid;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    DataType type;
    bool output;
    bool constant; // weights: the CPU fallback keeps one host copy
};

class OpBuilder
{
public:
    explicit OpBuilder(const std::string& name)
        : graph(std::make_shared<Graph>())
    {
        graph->set_name(name);
        graph->set_io_data_type(BF16).set_compute_data_type(F32).set_intermediate_data_type(F32);
    }

    Graph& g()
    {
        return *graph;
    }

    TensorPtr in(const std::string& name,
                 const std::vector<int64_t>& dims,
                 DataType type = BF16,
                 std::vector<int64_t> strides = {})
    {
        return add(name, dims, type, std::move(strides), false);
    }

    TensorPtr weight(const std::string& name,
                     const std::vector<int64_t>& dims,
                     DataType type = BF16,
                     std::vector<int64_t> strides = {})
    {
        return add(name, dims, type, std::move(strides), true);
    }

    // nn.Linear weight [out, in] used as the matmul B operand [1, in, out] (column-major).
    TensorPtr linearWeight(const std::string& name, int64_t in, int64_t out)
    {
        return weight(name, {1, in, out}, BF16, {in * out, 1, in});
    }

    TensorPtr scalar(const std::string& name, float value)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(_nextUid++).set_name(name).set_value(value);
        return tensor;
    }

    void out(const TensorPtr& tensor,
             const std::vector<int64_t>& dims,
             DataType type = BF16,
             std::vector<int64_t> strides = {})
    {
        if(strides.empty())
        {
            strides = packedStrides(dims);
        }
        tensor->set_uid(_nextUid++)
            .set_output(true)
            .set_data_type(type)
            .set_dim(dims)
            .set_stride(strides);
        slots.push_back({tensor->get_uid(), dims, strides, type, true, false});
    }

    std::shared_ptr<Graph> graph;
    std::vector<Slot> slots;

private:
    TensorPtr add(const std::string& name,
                  const std::vector<int64_t>& dims,
                  DataType type,
                  std::vector<int64_t> strides,
                  bool constant)
    {
        if(strides.empty())
        {
            strides = packedStrides(dims);
        }
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(_nextUid++)
            .set_name(name)
            .set_data_type(type)
            .set_dim(dims)
            .set_stride(strides);
        slots.push_back({tensor->get_uid(), dims, strides, type, false, constant});
        return tensor;
    }

    int64_t _nextUid = 1;
};

struct Op
{
    std::string label;
    std::shared_ptr<Graph> graph; // GPU plan
    std::vector<Slot> slots;
    std::string placement; // engine name, or "CPU reference"
    bool gpu = false;
    std::vector<uint8_t> serialized; // CPU fallback
    std::vector<std::vector<uint8_t>> staging;
};

using Builder = std::function<void(OpBuilder&)>;

class Runtime
{
public:
    explicit Runtime(bool profile)
        : _profile(profile)
    {
        auto [handle, error] = hipdnn_frontend::createHipdnnHandle();
        if(!error.is_good())
        {
            throw std::runtime_error("hipDNN handle: " + error.get_message());
        }
        _handle = std::move(handle);
    }

    ~Runtime()
    {
        if(_workspace != nullptr)
        {
            (void)hipFree(_workspace);
        }
    }

    // The op for `key`, built on first use. With allowCpu false a graph no engine accepts
    // returns nullptr, so the caller can decompose it instead.
    Op* get(const std::string& label,
            const std::string& key,
            const Builder& build,
            bool allowCpu = true)
    {
        if(auto it = _ops.find(key); it != _ops.end())
        {
            return it->second.get();
        }
        if(_declined.count(key) != 0)
        {
            return nullptr;
        }
        const auto t0 = now();
        auto op = std::make_unique<Op>();
        op->label = label;
        OpBuilder b(key);
        build(b);
        check(b, key);
        const auto error = b.graph->build(*_handle);
        std::string reason;
        if(error.is_good())
        {
            op->gpu = true;
            op->graph = b.graph;
            op->slots = b.slots;
            if(!b.graph->get_plan_name(*_handle, op->placement).is_good())
            {
                op->placement = "unnamed GPU engine";
            }
            int64_t workspace = 0;
            if(b.graph->get_workspace_size(workspace).is_good() && workspace > _workspaceSize)
            {
                HIP_CHECK(hipDeviceSynchronize());
                if(_workspace != nullptr)
                {
                    HIP_CHECK(hipFree(_workspace));
                }
                HIP_CHECK(hipMalloc(&_workspace, static_cast<size_t>(workspace)));
                _workspaceSize = workspace;
            }
        }
        else
        {
            reason = error.get_message();
            if(!allowCpu)
            {
                std::cerr << "[graph] " << label << " " << key << " -> no GPU engine, decomposing ("
                          << reason << ")\n";
                _declined.insert(key);
                return nullptr;
            }
            // Rebuild a fresh graph for the reference executor (same UIDs: the builder is
            // deterministic).
            OpBuilder cpu(key);
            build(cpu);
            check(cpu, key);
            auto [serialized, binaryError] = cpu.graph->to_binary();
            if(!binaryError.is_good())
            {
                throw std::runtime_error(key + ": " + binaryError.get_message());
            }
            op->serialized = std::move(serialized);
            op->slots = cpu.slots;
            op->staging.resize(op->slots.size());
            op->placement = "CPU reference";
        }
        std::cerr << "[graph] " << label << " " << key << " -> " << op->placement;
        if(!reason.empty())
        {
            std::cerr << " (" << reason << ")";
        }
        std::cerr << " [built in " << fixed((now() - t0) * 1e3, 1) << " ms]\n";
        _buildSeconds += now() - t0;
        return _ops.emplace(key, std::move(op)).first->second.get();
    }

    void run(Op& op, const std::vector<const void*>& ptrs)
    {
        if(ptrs.size() != op.slots.size())
        {
            throw std::runtime_error(op.label + ": expected " + std::to_string(op.slots.size())
                                     + " operands, got " + std::to_string(ptrs.size()));
        }
        double t0 = 0;
        if(_profile)
        {
            HIP_CHECK(hipDeviceSynchronize());
            t0 = now();
        }
        std::unordered_map<int64_t, void*> pack;
        if(op.gpu)
        {
            for(size_t i = 0; i < ptrs.size(); ++i)
            {
                pack[op.slots[i].uid] = const_cast<void*>(ptrs[i]);
            }
            const auto error = op.graph->execute(*_handle, pack, _workspace);
            if(!error.is_good())
            {
                throw std::runtime_error(op.label + " execute: " + error.get_message());
            }
        }
        else
        {
            for(size_t i = 0; i < ptrs.size(); ++i)
            {
                const auto& slot = op.slots[i];
                const auto bytes = spanElements(slot.dims, slot.strides) * elementSize(slot.type);
                if(slot.constant)
                {
                    auto& host = _constants[ptrs[i]];
                    if(host.size() != bytes)
                    {
                        host.resize(bytes);
                        HIP_CHECK(hipMemcpy(host.data(), ptrs[i], bytes, hipMemcpyDeviceToHost));
                    }
                    pack[slot.uid] = host.data();
                    continue;
                }
                auto& host = op.staging[i];
                host.resize(bytes);
                // Outputs too: a strided output view leaves the elements between its rows as
                // they were.
                HIP_CHECK(hipMemcpy(host.data(), ptrs[i], bytes, hipMemcpyDeviceToHost));
                pack[slot.uid] = host.data();
            }
            hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor{}.execute(
                op.serialized.data(), op.serialized.size(), pack);
            for(size_t i = 0; i < ptrs.size(); ++i)
            {
                if(op.slots[i].output)
                {
                    HIP_CHECK(hipMemcpy(const_cast<void*>(ptrs[i]),
                                        op.staging[i].data(),
                                        op.staging[i].size(),
                                        hipMemcpyHostToDevice));
                }
            }
        }
        auto& stats = _stats[op.label][op.placement];
        ++stats.runs;
        if(_profile)
        {
            HIP_CHECK(hipDeviceSynchronize());
            stats.seconds += now() - t0;
        }
    }

    void report() const
    {
        std::cerr << "\nop placement (runs per op label and engine";
        std::cerr << (_profile ? ", synchronized time" : "") << "):\n";
        for(const auto& [label, byPlacement] : _stats)
        {
            for(const auto& [placement, stats] : byPlacement)
            {
                std::cerr << "  " << std::left << std::setw(16) << label << std::setw(48)
                          << placement << std::right << std::setw(8) << stats.runs << " runs";
                if(_profile)
                {
                    std::cerr << std::setw(10) << fixed(stats.seconds, 3) << " s";
                }
                std::cerr << "\n";
            }
        }
        std::cerr << "graphs built: " << _ops.size() << " (" << _buildSeconds << " s total)\n";
    }

private:
    static void check(OpBuilder& b, const std::string& key)
    {
        const auto validation = b.graph->validate();
        if(!validation.is_good())
        {
            throw std::runtime_error(key + ": " + validation.get_message());
        }
    }

    struct Stats
    {
        int64_t runs = 0;
        double seconds = 0;
    };

    bool _profile;
    hipdnn_frontend::HipdnnHandlePtr _handle;
    void* _workspace = nullptr;
    int64_t _workspaceSize = 0;
    double _buildSeconds = 0;
    std::unordered_map<std::string, std::unique_ptr<Op>> _ops;
    std::unordered_set<std::string> _declined;
    std::unordered_map<const void*, std::vector<uint8_t>> _constants;
    std::map<std::string, std::map<std::string, Stats>> _stats;
};

std::vector<int64_t> bshdStrides(int64_t tokens, int64_t heads)
{
    return {tokens * heads * HD, HD, heads * HD, 1};
}

// The single-op graphs the model is built from. Pointers are device pointers.
class Ops
{
public:
    explicit Ops(Runtime& rt)
        : _rt(rt)
    {
    }

    // y = rmsnorm(x) * gamma over the last axis; x, y packed bf16 `dims`, gamma f32.
    void rmsnorm(const void* x, const void* gamma, void* y, const std::vector<int64_t>& dims)
    {
        auto* op = _rt.get("rmsnorm", "rmsnorm:" + joinDims(dims), [&](OpBuilder& b) {
            std::vector<int64_t> gammaDims(dims.size(), 1);
            gammaDims.back() = dims.back();
            hipdnn_frontend::graph::RMSNormAttributes attributes;
            attributes.set_epsilon(b.scalar("eps", RMS_EPS));
            attributes.set_forward_phase(NormFwdPhase::INFERENCE);
            auto xt = b.in("x", dims);
            auto gt = b.weight("gamma", gammaDims, F32);
            b.out(b.g().rmsnorm(xt, gt, attributes)[0], dims);
        });
        _rt.run(*op, {x, gamma, y});
    }

    // c[m, n] = a[m, k] * W^T, W an nn.Linear weight [n, k]; optionally c = silu(c), fused
    // into the GEMM epilogue when an engine takes it.
    void matmul(
        const void* a, const void* w, void* c, int64_t m, int64_t k, int64_t n, bool silu = false)
    {
        const auto shape = std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
        if(silu)
        {
            auto* fused = _rt.get(
                "matmul+silu",
                "matmul_silu:" + shape,
                [&](OpBuilder& b) {
                    auto mm = b.g().matmul(b.in("a", {1, m, k}),
                                           b.linearWeight("w", k, n),
                                           hipdnn_frontend::graph::MatmulAttributes{});
                    b.out(swish(b, mm), {1, m, n});
                },
                false);
            if(fused != nullptr)
            {
                _rt.run(*fused, {a, w, c});
                return;
            }
        }
        auto* op = _rt.get("matmul", "matmul:" + shape, [&](OpBuilder& b) {
            b.out(b.g().matmul(b.in("a", {1, m, k}),
                               b.linearWeight("w", k, n),
                               hipdnn_frontend::graph::MatmulAttributes{}),
                  {1, m, n});
        });
        _rt.run(*op, {a, w, c});
        if(silu)
        {
            unary(PointwiseMode::SWISH_FWD, c, c, {1, m, n});
        }
    }

    void unary(PointwiseMode mode, const void* x, void* y, const std::vector<int64_t>& dims)
    {
        const auto label = mode == PointwiseMode::SWISH_FWD ? "pw.silu" : "pw.sigmoid";
        auto* op = _rt.get(label, std::string(label) + ":" + joinDims(dims), [&](OpBuilder& b) {
            auto xt = b.in("x", dims);
            b.out(mode == PointwiseMode::SWISH_FWD ? swish(b, xt) : pointwiseOp(b, mode, xt), dims);
        });
        _rt.run(*op, {x, y});
    }

    void binary(
        PointwiseMode mode, const void* a, const void* c, void* y, const std::vector<int64_t>& dims)
    {
        const auto label = mode == PointwiseMode::ADD ? "pw.add" : "pw.mul";
        auto* op = _rt.get(label, std::string(label) + ":" + joinDims(dims), [&](OpBuilder& b) {
            auto at = b.in("a", dims);
            auto ct = b.in("b", dims);
            b.out(pointwiseOp(b, mode, at, ct), dims);
        });
        _rt.run(*op, {a, c, y});
    }

    void gather(const void* table, const void* ids, void* y, int64_t tokens)
    {
        auto* op = _rt.get("gather", "gather:" + std::to_string(tokens), [&](OpBuilder& b) {
            auto t = b.weight("table", {VOCAB, HIDDEN});
            auto i = b.in("ids", {1, tokens}, I32);
            b.out(b.g().gather(t, i, hipdnn_frontend::graph::GatherAttributes{}),
                  {1, tokens, HIDDEN});
        });
        _rt.run(*op, {table, ids, y});
    }

    void topk1(const void* logits, void* value, void* index, int64_t n)
    {
        auto* op = _rt.get("topk", "topk:" + std::to_string(n), [&](OpBuilder& b) {
            hipdnn_frontend::graph::TopkAttributes attributes;
            attributes.set_k(1);
            auto [values, indices] = b.g().topk(b.in("logits", {1, 1, n}), attributes);
            b.out(values, {1, 1, 1});
            b.out(indices, {1, 1, 1}, I32);
        });
        _rt.run(*op, {logits, value, index});
    }

    // Depthwise causal conv1d + SiLU over x [T, C], carrying the [1, C, W-1] window.
    void causalConv1d(const void* x,
                      const void* w,
                      const void* cuSeqlens,
                      const void* stateIn,
                      void* y,
                      void* stateOut,
                      int64_t tokens,
                      int64_t channels)
    {
        const auto key = "causal_conv1d:" + std::to_string(tokens) + "x" + std::to_string(channels);
        auto* op = _rt.get("causal_conv1d", key, [&](OpBuilder& b) {
            auto xt = b.in("x", {tokens, channels});
            auto wt = b.weight("w", {channels, CONV_W});
            auto cu = b.in("cu_seqlens", {2}, I32);
            hipdnn_frontend::graph::CausalConv1dAttributes attributes;
            attributes.set_activation(hipdnn_frontend::CausalConv1dActivation::SILU);
            attributes.set_output_final_state(true);
            attributes.set_initial_state(b.in("state_in", {1, channels, CONV_W - 1}));
            auto [yt, finalState] = b.g().causal_conv1d(xt, wt, cu, attributes);
            b.out(yt, {tokens, channels});
            b.out(finalState, {1, channels, CONV_W - 1});
        });
        _rt.run(*op, {x, w, cuSeqlens, stateIn, y, stateOut});
    }

    // Gated delta rule over the conv output [T, q 2048 | k 2048 | v 6144]; f32 state.
    void gdn(const void* qkv,
             const void* g,
             const void* beta,
             const void* cuSeqlens,
             const void* aLog,
             const void* dtBias,
             const void* stateIn,
             void* o,
             void* stateOut,
             int64_t tokens)
    {
        const auto* base = static_cast<const uint16_t*>(qkv);
        constexpr int64_t ROW = 2 * KEY_DIM + VALUE_DIM;
        auto* op = _rt.get("gdn", "gdn:" + std::to_string(tokens), [&](OpBuilder& b) {
            auto q = b.in("q", {tokens, NK, DK}, BF16, {ROW, DK, 1});
            auto k = b.in("k", {tokens, NK, DK}, BF16, {ROW, DK, 1});
            auto v = b.in("v", {tokens, NV, DV}, BF16, {ROW, DV, 1});
            auto gt = b.in("g", {tokens, NV});
            auto bt = b.in("beta", {tokens, NV});
            auto cu = b.in("cu_seqlens", {2}, I32);
            hipdnn_frontend::graph::GdnAttributes attributes;
            attributes.set_use_qk_l2norm(true)
                .set_use_beta_sigmoid(true)
                .set_safe_gate(true)
                .set_gate_domain(hipdnn_frontend::GateDomain::LOG)
                .set_output_final_state(true)
                .set_a_log(b.weight("a_log", {NV}, F32))
                .set_dt_bias(b.weight("dt_bias", {NV}, F32))
                .set_initial_state(b.in("state_in", {1, NV, DV, DK}, F32));
            auto [ot, finalState, checkpoints] = b.g().gdn(q, k, v, gt, bt, cu, attributes);
            (void)checkpoints;
            b.out(ot, {tokens, NV, DV});
            b.out(finalState, {1, NV, DV, DK}, F32);
        });
        _rt.run(*op,
                {base,
                 base + KEY_DIM,
                 base + 2 * KEY_DIM,
                 g,
                 beta,
                 cuSeqlens,
                 aLog,
                 dtBias,
                 stateIn,
                 o,
                 stateOut});
    }

    // Partial RoPE (leading ROT dims) of x [1, H, T, D] (BSHD memory) into y with `yStrides`.
    void rope(const void* x,
              const void* freqs,
              void* y,
              int64_t heads,
              int64_t tokens,
              const std::vector<int64_t>& yStrides)
    {
        const auto key = "rope:" + std::to_string(heads) + "x" + std::to_string(tokens) + ":"
                         + joinDims(yStrides);
        auto* op = _rt.get("rope", key, [&](OpBuilder& b) {
            auto xt = b.in("x", {1, heads, tokens, HD}, BF16, bshdStrides(tokens, heads));
            auto f = b.in("freqs", {1, 1, tokens, ROT}, F32);
            hipdnn_frontend::graph::RopeAttributes attributes;
            attributes.set_rope_dim(ROT).set_rope_offset(0);
            b.out(b.g().rope(xt, f, attributes), {1, heads, tokens, HD}, BF16, yStrides);
        });
        _rt.run(*op, {x, freqs, y});
    }

    // Causal (bottom-right) GQA attention of q [1, NQ, T, D] against the first `kvLen` rows of
    // the dense BSHD caches [Smax, NKV, D].
    void sdpa(const void* q,
              const void* kCache,
              const void* vCache,
              void* o,
              int64_t tokens,
              int64_t kvLen,
              int64_t maxSeq)
    {
        const auto key = "sdpa:" + std::to_string(tokens) + "x" + std::to_string(kvLen) + "/"
                         + std::to_string(maxSeq);
        auto* op = _rt.get("sdpa", key, [&](OpBuilder& b) {
            const std::vector<int64_t> kvStrides{maxSeq * KV_DIM, HD, KV_DIM, 1};
            auto qt = b.in("q", {1, NQ, tokens, HD}, BF16, bshdStrides(tokens, NQ));
            auto kt = b.in("k", {1, NKV, kvLen, HD}, BF16, kvStrides);
            auto vt = b.in("v", {1, NKV, kvLen, HD}, BF16, kvStrides);
            hipdnn_frontend::graph::SdpaAttributes attributes;
            attributes.set_attn_scale(1.0F / std::sqrt(static_cast<float>(HD)))
                .set_causal_mask_bottom_right(true)
                .set_generate_stats(false);
            auto [ot, stats] = b.g().sdpa(qt, kt, vt, attributes);
            (void)stats;
            b.out(ot, {1, NQ, tokens, HD}, BF16, bshdStrides(tokens, NQ));
        });
        _rt.run(*op, {q, kCache, vCache, o});
    }

private:
    static TensorPtr swish(OpBuilder& b, const TensorPtr& x)
    {
        hipdnn_frontend::graph::PointwiseAttributes attributes;
        attributes.set_mode(PointwiseMode::SWISH_FWD);
        attributes.set_swish_beta(1.0F);
        return b.g().pointwise(x, attributes);
    }

    static TensorPtr pointwiseOp(OpBuilder& b,
                                 PointwiseMode mode,
                                 const TensorPtr& x,
                                 const TensorPtr& y = nullptr)
    {
        hipdnn_frontend::graph::PointwiseAttributes attributes;
        attributes.set_mode(mode);
        return y ? b.g().pointwise(x, y, attributes) : b.g().pointwise(x, attributes);
    }

    Runtime& _rt;
};

// Weight upload helpers.
void* uploadBf16(DeviceArena& arena, Checkpoint& ckpt, const std::string& name)
{
    size_t count = 0;
    const auto* data = ckpt.raw(name, &count);
    return arena.upload(data, count * sizeof(uint16_t));
}

void* uploadF32(DeviceArena& arena, const Floats& values)
{
    return arena.upload(values.data(), values.size() * sizeof(float));
}

// Per-step activation buffers, sized for the longest step (the prompt).
struct Activations
{
    Activations(DeviceArena& arena, int64_t maxTokens)
    {
        const auto bf16 = [&](int64_t elements) {
            return static_cast<uint16_t*>(
                arena.alloc(static_cast<size_t>(maxTokens * elements) * 2));
        };
        x = bf16(HIDDEN);
        x1 = bf16(HIDDEN);
        h = bf16(HIDDEN);
        proj = bf16(HIDDEN);
        qkv = bf16(2 * KEY_DIM + VALUE_DIM);
        conv = bf16(2 * KEY_DIM + VALUE_DIM);
        z = bf16(VALUE_DIM);
        beta = bf16(NV);
        gate = bf16(NV);
        o = bf16(VALUE_DIM);
        oNorm = bf16(VALUE_DIM);
        zAct = bf16(VALUE_DIM);
        gated = bf16(Q_DIM);
        q = bf16(Q_DIM);
        qGate = bf16(Q_DIM);
        qNorm = bf16(Q_DIM);
        qRot = bf16(Q_DIM);
        k = bf16(KV_DIM);
        kNorm = bf16(KV_DIM);
        attn = bf16(Q_DIM);
        qGateAct = bf16(Q_DIM);
        mlpGate = bf16(INTER);
        mlpUp = bf16(INTER);
        mlpAct = bf16(INTER);
        logits = static_cast<uint16_t*>(arena.alloc(VOCAB * 2));
        topValue = arena.alloc(16);
        topIndex = arena.alloc(16);
        ids = arena.alloc(static_cast<size_t>(maxTokens) * 4);
        cuSeqlens = arena.alloc(8);
        freqs = arena.alloc(static_cast<size_t>(maxTokens * ROT) * 4);
    }

    uint16_t *x, *x1, *h, *proj, *qkv, *conv, *z, *beta, *gate, *o, *oNorm, *zAct, *gated, *q,
        *qGate, *qNorm, *qRot, *k, *kNorm, *attn, *qGateAct, *mlpGate, *mlpUp, *mlpAct, *logits;
    void *topValue, *topIndex, *ids, *cuSeqlens, *freqs;
};

struct GpuMlp
{
    void *postNorm, *gate, *up, *down;

    void load(DeviceArena& arena, Checkpoint& ckpt, const std::string& prefix)
    {
        postNorm = uploadF32(arena, onePlus(ckpt.load(prefix + "post_attention_layernorm.weight")));
        gate = uploadBf16(arena, ckpt, prefix + "mlp.gate_proj.weight");
        up = uploadBf16(arena, ckpt, prefix + "mlp.up_proj.weight");
        down = uploadBf16(arena, ckpt, prefix + "mlp.down_proj.weight");
    }

    // x = x1 + down(silu(gate(n)) * up(n)), n = rmsnorm(x1)
    void run(Ops& ops, Activations& a, int64_t tokens) const
    {
        const std::vector<int64_t> hidden{1, tokens, HIDDEN};
        const std::vector<int64_t> inter{1, tokens, INTER};
        ops.rmsnorm(a.x1, postNorm, a.h, hidden);
        ops.matmul(a.h, gate, a.mlpGate, tokens, HIDDEN, INTER, true);
        ops.matmul(a.h, up, a.mlpUp, tokens, HIDDEN, INTER);
        ops.binary(PointwiseMode::MUL, a.mlpGate, a.mlpUp, a.mlpAct, inter);
        ops.matmul(a.mlpAct, down, a.proj, tokens, INTER, HIDDEN);
        ops.binary(PointwiseMode::ADD, a.x1, a.proj, a.x, hidden);
    }
};

struct GpuGdnLayer
{
    void *inNorm, *qkv, *z, *b, *a, *conv, *aLog, *dtBias, *gatedNorm, *outProj;
    GpuMlp mlp;
    // Carried state, double-buffered (graphs read one and write the other).
    std::array<void*, 2> convState = {};
    std::array<void*, 2> recurrent = {};
    int current = 0;

    void load(DeviceArena& arena, Checkpoint& ckpt, const std::string& p)
    {
        inNorm = uploadF32(arena, onePlus(ckpt.load(p + "input_layernorm.weight")));
        qkv = uploadBf16(arena, ckpt, p + "linear_attn.in_proj_qkv.weight");
        z = uploadBf16(arena, ckpt, p + "linear_attn.in_proj_z.weight");
        b = uploadBf16(arena, ckpt, p + "linear_attn.in_proj_b.weight");
        a = uploadBf16(arena, ckpt, p + "linear_attn.in_proj_a.weight");
        conv = uploadBf16(arena, ckpt, p + "linear_attn.conv1d.weight"); // [10240, 1, 4]
        aLog = uploadF32(arena, ckpt.load(p + "linear_attn.A_log"));
        dtBias = uploadF32(arena, ckpt.load(p + "linear_attn.dt_bias"));
        gatedNorm = uploadF32(arena, ckpt.load(p + "linear_attn.norm.weight"));
        outProj = uploadBf16(arena, ckpt, p + "linear_attn.out_proj.weight");
        mlp.load(arena, ckpt, p);
        constexpr int64_t CHANNELS = 2 * KEY_DIM + VALUE_DIM;
        for(int i = 0; i < 2; ++i)
        {
            convState[i] = arena.alloc(static_cast<size_t>(CHANNELS * (CONV_W - 1)) * 2, true);
            recurrent[i] = arena.alloc(static_cast<size_t>(NV * DV * DK) * 4, true);
        }
    }

    void run(Ops& ops, Activations& act, int64_t tokens)
    {
        constexpr int64_t CHANNELS = 2 * KEY_DIM + VALUE_DIM;
        const std::vector<int64_t> hidden{1, tokens, HIDDEN};
        const std::vector<int64_t> value{1, tokens, VALUE_DIM};
        ops.rmsnorm(act.x, inNorm, act.h, hidden);
        ops.matmul(act.h, qkv, act.qkv, tokens, HIDDEN, CHANNELS);
        ops.matmul(act.h, z, act.z, tokens, HIDDEN, VALUE_DIM);
        ops.matmul(act.h, b, act.beta, tokens, HIDDEN, NV);
        ops.matmul(act.h, a, act.gate, tokens, HIDDEN, NV);
        const int next = 1 - current;
        ops.causalConv1d(act.qkv,
                         conv,
                         act.cuSeqlens,
                         convState[current],
                         act.conv,
                         convState[next],
                         tokens,
                         CHANNELS);
        ops.gdn(act.conv,
                act.gate,
                act.beta,
                act.cuSeqlens,
                aLog,
                dtBias,
                recurrent[current],
                act.o,
                recurrent[next],
                tokens);
        current = next;
        ops.rmsnorm(act.o, gatedNorm, act.oNorm, {tokens, NV, DV});
        ops.unary(PointwiseMode::SWISH_FWD, act.z, act.zAct, value);
        ops.binary(PointwiseMode::MUL, act.oNorm, act.zAct, act.gated, value);
        ops.matmul(act.gated, outProj, act.proj, tokens, VALUE_DIM, HIDDEN);
        ops.binary(PointwiseMode::ADD, act.x, act.proj, act.x1, hidden);
        mlp.run(ops, act, tokens);
    }
};

struct GpuAttentionLayer
{
    void *inNorm, *wq, *wgate, *wk, *wv, *qNorm, *kNorm, *wo;
    GpuMlp mlp;
    // Dense BSHD caches [Smax, NKV, HD]; SDPA reads the first `position + tokens` rows.
    uint16_t *kCache, *vCache;
    int64_t maxSeq = 0;

    void load(DeviceArena& arena, Checkpoint& ckpt, const std::string& p, int64_t maxSeqLen)
    {
        inNorm = uploadF32(arena, onePlus(ckpt.load(p + "input_layernorm.weight")));
        // q_proj rows are per head [q (256) | gate (256)]: split host-side.
        size_t count = 0;
        const auto* qg = ckpt.raw(p + "self_attn.q_proj.weight", &count);
        std::vector<uint16_t> q(static_cast<size_t>(Q_DIM * HIDDEN));
        std::vector<uint16_t> gate(q.size());
        for(int64_t h = 0; h < NQ; ++h)
        {
            const auto* src = qg + h * 2 * HD * HIDDEN;
            std::memcpy(q.data() + h * HD * HIDDEN, src, 2 * HD * HIDDEN);
            std::memcpy(gate.data() + h * HD * HIDDEN, src + HD * HIDDEN, 2 * HD * HIDDEN);
        }
        wq = arena.upload(q.data(), q.size() * 2);
        wgate = arena.upload(gate.data(), gate.size() * 2);
        wk = uploadBf16(arena, ckpt, p + "self_attn.k_proj.weight");
        wv = uploadBf16(arena, ckpt, p + "self_attn.v_proj.weight");
        qNorm = uploadF32(arena, onePlus(ckpt.load(p + "self_attn.q_norm.weight")));
        kNorm = uploadF32(arena, onePlus(ckpt.load(p + "self_attn.k_norm.weight")));
        wo = uploadBf16(arena, ckpt, p + "self_attn.o_proj.weight");
        mlp.load(arena, ckpt, p);
        maxSeq = maxSeqLen;
        kCache
            = static_cast<uint16_t*>(arena.alloc(static_cast<size_t>(maxSeq * KV_DIM) * 2, true));
        vCache
            = static_cast<uint16_t*>(arena.alloc(static_cast<size_t>(maxSeq * KV_DIM) * 2, true));
    }

    void run(Ops& ops, Activations& act, int64_t first, int64_t tokens) const
    {
        const std::vector<int64_t> hidden{1, tokens, HIDDEN};
        const std::vector<int64_t> query{1, tokens, Q_DIM};
        ops.rmsnorm(act.x, inNorm, act.h, hidden);
        ops.matmul(act.h, wq, act.q, tokens, HIDDEN, Q_DIM);
        ops.matmul(act.h, wgate, act.qGate, tokens, HIDDEN, Q_DIM);
        ops.matmul(act.h, wk, act.k, tokens, HIDDEN, KV_DIM);
        // V lands in its cache rows directly: [T, NKV * HD] is the BSHD row layout.
        ops.matmul(act.h, wv, vCache + first * KV_DIM, tokens, HIDDEN, KV_DIM);
        ops.rmsnorm(act.q, qNorm, act.qNorm, {1, tokens, NQ, HD});
        ops.rmsnorm(act.k, kNorm, act.kNorm, {1, tokens, NKV, HD});
        ops.rope(act.qNorm, act.freqs, act.qRot, NQ, tokens, bshdStrides(tokens, NQ));
        // K is rotated straight into its cache rows.
        ops.rope(act.kNorm,
                 act.freqs,
                 kCache + first * KV_DIM,
                 NKV,
                 tokens,
                 {maxSeq * KV_DIM, HD, KV_DIM, 1});
        ops.sdpa(act.qRot, kCache, vCache, act.attn, tokens, first + tokens, maxSeq);
        ops.unary(PointwiseMode::SIGMOID_FWD, act.qGate, act.qGateAct, query);
        ops.binary(PointwiseMode::MUL, act.attn, act.qGateAct, act.gated, query);
        ops.matmul(act.gated, wo, act.proj, tokens, Q_DIM, HIDDEN);
        ops.binary(PointwiseMode::ADD, act.x, act.proj, act.x1, hidden);
        mlp.run(ops, act, tokens);
    }
};

int run(const std::string& dir, const Ints& prompt, int64_t maxNew, int64_t layers, bool profile)
{
    const auto maxSeq = static_cast<int64_t>(prompt.size()) + maxNew;
    const auto maxTokens = std::max<int64_t>(1, static_cast<int64_t>(prompt.size()));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cerr << "GPU backend on " << props.gcnArchName << " (" << props.totalGlobalMem / (1 << 30)
              << " GiB)\n";

    DeviceArena arena;
    Runtime rt(profile);
    Ops ops(rt);

    auto t0 = now();
    Checkpoint ckpt(dir);
    ckpt.prefetch(4);
    const std::string base = "model.language_model.";
    auto* embed = uploadBf16(arena, ckpt, base + "embed_tokens.weight");
    auto* finalNorm = uploadF32(arena, onePlus(ckpt.load(base + "norm.weight")));
    auto* lmHead = uploadBf16(arena, ckpt, "lm_head.weight");
    std::vector<std::unique_ptr<GpuGdnLayer>> gdn(static_cast<size_t>(layers));
    std::vector<std::unique_ptr<GpuAttentionLayer>> attn(static_cast<size_t>(layers));
    for(int64_t l = 0; l < layers; ++l)
    {
        const auto prefix = base + "layers." + std::to_string(l) + ".";
        if((l + 1) % FULL_ATTENTION_INTERVAL == 0)
        {
            attn[static_cast<size_t>(l)] = std::make_unique<GpuAttentionLayer>();
            attn[static_cast<size_t>(l)]->load(arena, ckpt, prefix, maxSeq);
        }
        else
        {
            gdn[static_cast<size_t>(l)] = std::make_unique<GpuGdnLayer>();
            gdn[static_cast<size_t>(l)]->load(arena, ckpt, prefix);
        }
        std::cerr << "\rloaded layer " << l + 1 << "/" << layers << std::flush;
    }
    Activations act(arena, maxTokens);
    arena.finishUploads();
    HIP_CHECK(hipDeviceSynchronize());
    const auto loadSeconds = now() - t0;
    std::cerr << "\nweights loaded in " << loadSeconds << " s ("
              << static_cast<double>(arena.bytes()) / double(1 << 30) << " GiB on device)\n";

    Ints ids = prompt;
    int64_t position = 0;
    std::vector<double> stepSeconds;
    for(int64_t step = 0; step < maxNew; ++step)
    {
        const auto ts = now();
        const auto first = position;
        const auto tokens = static_cast<int64_t>(ids.size()) - first;

        // Step inputs: token ids, cu_seqlens and the raw RoPE angles.
        HIP_CHECK(hipMemcpy(
            act.ids, ids.data() + first, static_cast<size_t>(tokens) * 4, hipMemcpyHostToDevice));
        const Ints cuSeqlens{0, static_cast<int32_t>(tokens)};
        HIP_CHECK(hipMemcpy(act.cuSeqlens, cuSeqlens.data(), 8, hipMemcpyHostToDevice));
        Floats angles(static_cast<size_t>(tokens * ROT));
        for(int64_t t = 0; t < tokens; ++t)
        {
            for(int64_t i = 0; i < ROT / 2; ++i)
            {
                const auto invFreq = std::pow(ROPE_THETA, -2.0 * static_cast<double>(i) / ROT);
                const auto theta = static_cast<float>(static_cast<double>(first + t) * invFreq);
                angles[static_cast<size_t>(t * ROT + i)] = theta;
                angles[static_cast<size_t>(t * ROT + i + ROT / 2)] = theta;
            }
        }
        HIP_CHECK(hipMemcpy(act.freqs, angles.data(), angles.size() * 4, hipMemcpyHostToDevice));

        ops.gather(embed, act.ids, act.x, tokens);
        for(int64_t l = 0; l < layers; ++l)
        {
            if((l + 1) % FULL_ATTENTION_INTERVAL == 0)
            {
                attn[static_cast<size_t>(l)]->run(ops, act, first, tokens);
            }
            else
            {
                gdn[static_cast<size_t>(l)]->run(ops, act, tokens);
            }
        }
        ops.rmsnorm(act.x + (tokens - 1) * HIDDEN, finalNorm, act.h, {1, 1, HIDDEN});
        ops.matmul(act.h, lmHead, act.logits, 1, HIDDEN, VOCAB);
        ops.topk1(act.logits, act.topValue, act.topIndex, VOCAB);
        int32_t next = 0;
        HIP_CHECK(hipMemcpy(&next, act.topIndex, 4, hipMemcpyDeviceToHost));

        position += tokens;
        ids.push_back(next);
        stepSeconds.push_back(now() - ts);
        std::cout << "TOKEN " << next << '\n' << std::flush;
        std::cerr << "step " << step << ": " << tokens << " token(s) in " << stepSeconds.back()
                  << " s\n";
        if(next == 248046 || next == 248044)
        {
            break;
        }
    }

    rt.report();
    // Step 0 is the prefill; step 1 also builds the decode-shape graphs, so the steady decode
    // rate is taken from step 2 on.
    std::cerr << "load " << fixed(loadSeconds, 1) << " s; prefill (" << prompt.size() << " tokens) "
              << fixed(stepSeconds.front(), 3) << " s";
    if(stepSeconds.size() > 1)
    {
        std::cerr << "; first decode step " << fixed(stepSeconds[1], 3) << " s";
    }
    if(stepSeconds.size() > 2)
    {
        double decode = 0;
        for(size_t i = 2; i < stepSeconds.size(); ++i)
        {
            decode += stepSeconds[i];
        }
        const auto steps = static_cast<double>(stepSeconds.size() - 2);
        std::cerr << "; decode " << fixed(decode / steps, 3) << " s/token ("
                  << fixed(steps / decode, 2) << " tok/s) over " << stepSeconds.size() - 2
                  << " tokens";
    }
    std::cerr << "\n";
    return 0;
}

} // namespace gpu

int runSample(int argc, char** argv)
{
    const char* const usage
        = "usage: hipdnn_sample_qwen38 <snapshot-dir> <ids,comma,separated> <max-new-tokens>"
          " [layers] [--backend=cpu|gpu] [--profile]\n"
          "Runs Qwen3.8-27B through hipDNN graphs with greedy decoding and prints \"TOKEN <id>\"\n"
          "per generated token. Use run_qwen38.py to tokenize a prompt and detokenize the "
          "output.\n";
    std::vector<std::string> args;
    std::string backend = "cpu";
    bool profile = false;
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--help" || arg == "-h")
        {
            std::cout << usage;
            return 0;
        }
        if(arg.rfind("--backend=", 0) == 0)
        {
            backend = arg.substr(10);
        }
        else if(arg == "--profile")
        {
            profile = true;
        }
        else
        {
            args.push_back(arg);
        }
    }
    if(args.size() < 3 || (backend != "cpu" && backend != "gpu"))
    {
        std::cerr << usage;
        return 2;
    }
    const std::string dir = args[0];
    Ints prompt;
    {
        std::stringstream ss(args[1]);
        std::string item;
        while(std::getline(ss, item, ','))
        {
            prompt.push_back(std::stoi(item));
        }
    }
    const auto maxNew = std::stoll(args[2]);
    const int64_t layers = args.size() > 3 ? std::stoll(args[3]) : LAYERS;
    if(backend == "gpu")
    {
        return gpu::run(dir, prompt, maxNew, layers, profile);
    }
    const auto maxSeq = static_cast<int64_t>(prompt.size()) + maxNew;

    auto t0 = now();
    Checkpoint ckpt(dir);
    const std::string base = "model.language_model.";
    auto embed = ckpt.load(base + "embed_tokens.weight");
    auto finalNorm = onePlus(ckpt.load(base + "norm.weight"));
    auto lmHead = ckpt.load("lm_head.weight");
    std::vector<std::unique_ptr<GdnLayer>> gdn(static_cast<size_t>(layers));
    std::vector<std::unique_ptr<AttentionLayer>> attn(static_cast<size_t>(layers));
    for(int64_t l = 0; l < layers; ++l)
    {
        const auto prefix = base + "layers." + std::to_string(l) + ".";
        if((l + 1) % FULL_ATTENTION_INTERVAL == 0)
        {
            attn[static_cast<size_t>(l)] = std::make_unique<AttentionLayer>();
            attn[static_cast<size_t>(l)]->load(ckpt, prefix, maxSeq);
        }
        else
        {
            gdn[static_cast<size_t>(l)] = std::make_unique<GdnLayer>();
            gdn[static_cast<size_t>(l)]->load(ckpt, prefix);
        }
        std::cerr << "\rloaded layer " << l + 1 << "/" << layers << std::flush;
    }
    std::cerr << "\nweights loaded in " << now() - t0 << " s\n";

    Ints ids = prompt;
    int64_t position = 0;
    for(int64_t step = 0; step < maxNew; ++step)
    {
        const auto ts = now();
        const auto first = position;
        const auto tokens = static_cast<int64_t>(ids.size()) - first;
        Ints stepIds(ids.begin() + first, ids.end());

        // Embedding.
        Floats x(static_cast<size_t>(tokens * HIDDEN));
        {
            GraphRun g("embed");
            auto y = g.graph().gather(g.input("table", {VOCAB, HIDDEN}, embed.data()),
                                      g.input("ids", {1, tokens}, stepIds.data(), DataType::INT32),
                                      hipdnn_frontend::graph::GatherAttributes{});
            g.output(y, {1, tokens, HIDDEN}, x.data());
            g.run();
        }
        for(int64_t l = 0; l < layers; ++l)
        {
            x = (l + 1) % FULL_ATTENTION_INTERVAL == 0
                    ? attn[static_cast<size_t>(l)]->run(x, first, tokens)
                    : gdn[static_cast<size_t>(l)]->run(x, tokens);
        }
        // Final norm + lm_head on the last token, greedy argmax via top-k.
        Ints next(1);
        {
            GraphRun g("sample");
            auto last = g.input("x_last", {1, 1, HIDDEN}, x.data() + (tokens - 1) * HIDDEN);
            auto n = rmsnorm(g, last, g.input("final_norm", {1, 1, HIDDEN}, finalNorm.data()));
            auto logits = matmul(g, n, g.linearWeight("lm_head", lmHead.data(), HIDDEN, VOCAB));
            hipdnn_frontend::graph::TopkAttributes topk;
            topk.set_k(1);
            auto [values, indices] = g.graph().topk(logits, topk);
            values->set_is_virtual(true);
            g.output(indices, {1, 1, 1}, next.data(), DataType::INT32);
            g.run();
        }
        position += tokens;
        ids.push_back(next[0]);
        std::cout << "TOKEN " << next[0] << '\n' << std::flush;
        std::cerr << "step " << step << ": " << tokens << " token(s) in " << now() - ts << " s\n";
        if(next[0] == 248046 || next[0] == 248044)
        {
            break;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        return runSample(argc, argv);
    }
    catch(const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
    }
    catch(...)
    {
        std::cerr << "error: unknown exception\n";
    }
    return 1;
}
