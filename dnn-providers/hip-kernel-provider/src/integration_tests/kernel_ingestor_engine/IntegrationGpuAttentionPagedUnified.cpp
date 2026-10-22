// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

/**
 * @file IntegrationGpuAttentionPagedUnified.cpp
 * @brief End-to-end runs of hipkernel:AttentionPagedUnified through the hipDNN frontend:
 *        paged uniform decode (dense Q), ragged Q over paged K/V (mixed decode and
 *        chunked prefill), sliding windows, and HIP-graph capture/replay with changed
 *        sequence lengths, block ids and data. Every output is compared with a host
 *        gather-then-dense fp32 reference over the same page table.
 *
 * Set HIPDNN_TEST_EXPECTED_ARCH=gfx942 (or gfx950) on required runs so a missing or
 * different device fails instead of skipping.
 */
namespace
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

constexpr const char* ENGINE = "hipkernel:AttentionPagedUnified";

uint16_t toBits(float x, DataType type)
{
    if(type == DataType::BFLOAT16)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &x, 4);
        bits += 0x7fffU + ((bits >> 16) & 1U);
        return static_cast<uint16_t>(bits >> 16);
    }
    const auto half = static_cast<_Float16>(x);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, 2);
    return bits;
}

float fromBits(uint16_t bits, DataType type)
{
    if(type == DataType::BFLOAT16)
    {
        const uint32_t wide = static_cast<uint32_t>(bits) << 16;
        float value = 0;
        std::memcpy(&value, &wide, 4);
        return value;
    }
    _Float16 half = 0;
    std::memcpy(&half, &bits, 2);
    return static_cast<float>(half);
}

/// One problem. Q/O are token-major (`[tokens, Hq, D]` with a token stride of
/// `qTokenHeads * D`, so a fused QKV view is expressible); the cache is one
/// `[numBlocks, Hkv, blockSize, 2*D]` allocation viewed as K (first D) and V (last D),
/// exactly vLLM's TRITON_ATTN layout.
struct Problem
{
    DataType type = DataType::BFLOAT16;
    int64_t queryHeads = 32;
    int64_t kvHeads = 8;
    int64_t headSize = 128;
    int64_t blockSize = 16;
    int64_t numBlocks = 96;
    int64_t maxBlocks = 24; ///< page-table width M; max_seq_len_kv = M * blockSize
    bool ragged = false;
    int64_t denseQueryLength = 1; ///< Sq of a dense Q
    int64_t raggedMaxLength = 0; ///< Sq (S_max) of a ragged Q
    std::vector<int32_t> queryLengths; ///< per sequence (ragged, or dense with seq_len_q)
    std::vector<int32_t> kvLengths;
    std::vector<int32_t> gapsAfter; ///< ragged: padding tokens after each sequence
    bool withSeqLenQ = false;
    int64_t leftBound = -1;
    int64_t qTokenHeads = 0; ///< Q token stride in heads; 0 = queryHeads
    uint32_t seed = 1;

    int64_t batch() const
    {
        return static_cast<int64_t>(kvLengths.size());
    }
};

class DeviceBuffers
{
public:
    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    DeviceBuffers(DeviceBuffers&&) = delete;
    DeviceBuffers& operator=(DeviceBuffers&&) = delete;
    ~DeviceBuffers()
    {
        for(void* p : _allocations)
        {
            EXPECT_EQ(hipFree(p), hipSuccess);
        }
    }
    void* allocate(size_t bytes)
    {
        void* p = nullptr;
        if(hipMalloc(&p, std::max<size_t>(bytes, 16)) != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed");
        }
        _allocations.push_back(p);
        return p;
    }

private:
    std::vector<void*> _allocations;
};

/// Host-side state of one problem instance: data, page table and lengths.
struct HostState
{
    std::vector<uint16_t> q; ///< token-major, token stride qTokenHeads*D
    std::vector<uint16_t> cache; ///< [NB, Hkv, bs, 2D]
    std::vector<int32_t> pageTable; ///< [B, M]
    std::vector<int32_t> kvLengths;
    std::vector<int32_t> queryLengths;
    std::vector<int32_t> offsets; ///< ragged token offsets [B+1]
    int64_t qTokens = 0; ///< tokens addressed in the Q/O storage
};

int64_t queryRowsFor(const Problem& p, const HostState& s, int64_t b)
{
    if(p.ragged || p.withSeqLenQ)
    {
        return s.queryLengths[static_cast<size_t>(b)];
    }
    return p.denseQueryLength;
}

int64_t firstToken(const Problem& p, const HostState& s, int64_t b)
{
    return p.ragged ? s.offsets[static_cast<size_t>(b)] : b * p.denseQueryLength;
}

/// Random data, and a shuffled page table whose unused entries hold garbage block ids
/// the kernel must never follow.
HostState makeState(const Problem& p, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    HostState s;
    s.kvLengths = p.kvLengths;
    s.queryLengths = p.queryLengths;
    const int64_t batch = p.batch();
    if(p.ragged)
    {
        s.offsets.assign(static_cast<size_t>(batch + 1), 0);
        for(int64_t b = 0; b < batch; ++b)
        {
            const auto gap = p.gapsAfter.empty() ? 0 : p.gapsAfter[static_cast<size_t>(b)];
            s.offsets[static_cast<size_t>(b + 1)]
                = s.offsets[static_cast<size_t>(b)] + p.queryLengths[static_cast<size_t>(b)] + gap;
        }
        s.qTokens = s.offsets.back();
    }
    else
    {
        s.qTokens = batch * p.denseQueryLength;
    }
    const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
    s.q.resize(static_cast<size_t>(std::max<int64_t>(s.qTokens, 1) * tokenHeads * p.headSize));
    for(auto& x : s.q)
    {
        x = toBits(uniform(rng), p.type);
    }
    s.cache.resize(static_cast<size_t>(p.numBlocks * p.kvHeads * p.blockSize * 2 * p.headSize));
    for(auto& x : s.cache)
    {
        x = toBits(uniform(rng), p.type);
    }
    std::vector<int32_t> blocks(static_cast<size_t>(p.numBlocks));
    std::iota(blocks.begin(), blocks.end(), 0);
    std::shuffle(blocks.begin(), blocks.end(), rng);
    s.pageTable.assign(static_cast<size_t>(batch * p.maxBlocks), 0x7fffffff);
    size_t next = 0;
    for(int64_t b = 0; b < batch; ++b)
    {
        const int64_t used = (s.kvLengths[static_cast<size_t>(b)] + p.blockSize - 1) / p.blockSize;
        for(int64_t i = 0; i < used; ++i)
        {
            s.pageTable[static_cast<size_t>(b * p.maxBlocks + i)]
                = blocks.at(next++ % blocks.size());
        }
    }
    return s;
}

/// fp32 gather-then-dense reference: for each sequence, gather its K/V through the page
/// table into dense rows, then bottom-right causal (plus window) softmax attention.
std::vector<float> reference(const Problem& p, const HostState& s, float scale)
{
    const int64_t d = p.headSize;
    const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
    const int64_t group = p.queryHeads / p.kvHeads;
    std::vector<float> out(static_cast<size_t>(std::max<int64_t>(s.qTokens, 1) * p.queryHeads * d),
                           0.0F);
    for(int64_t b = 0; b < p.batch(); ++b)
    {
        const int64_t kv = s.kvLengths[static_cast<size_t>(b)];
        const int64_t rows = queryRowsFor(p, s, b);
        const int64_t first = firstToken(p, s, b);
        const auto cacheAt = [&](int64_t token, int64_t head, int64_t dim, bool value) {
            const int64_t block
                = s.pageTable[static_cast<size_t>(b * p.maxBlocks + token / p.blockSize)];
            const int64_t index
                = ((block * p.kvHeads + head) * p.blockSize + token % p.blockSize) * 2 * d
                  + (value ? d : 0) + dim;
            return fromBits(s.cache[static_cast<size_t>(index)], p.type);
        };
        for(int64_t row = 0; row < rows; ++row)
        {
            const int64_t diagonal = row + kv - rows;
            for(int64_t h = 0; h < p.queryHeads; ++h)
            {
                const int64_t kvHead = h / group;
                std::vector<float> scores;
                std::vector<int64_t> keys;
                for(int64_t key = 0; key < kv; ++key)
                {
                    if(key > diagonal || (p.leftBound >= 0 && key < diagonal - p.leftBound))
                    {
                        continue;
                    }
                    float dot = 0.0F;
                    for(int64_t i = 0; i < d; ++i)
                    {
                        dot += fromBits(s.q[static_cast<size_t>(((first + row) * tokenHeads + h) * d
                                                                + i)],
                                        p.type)
                               * cacheAt(key, kvHead, i, false);
                    }
                    scores.push_back(dot * scale);
                    keys.push_back(key);
                }
                auto* o = &out[static_cast<size_t>(((first + row) * p.queryHeads + h) * d)];
                if(scores.empty())
                {
                    continue; // no visible key: zeros
                }
                const float maximum = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0F;
                for(auto& x : scores)
                {
                    x = std::exp(x - maximum);
                    sum += x;
                }
                for(size_t j = 0; j < keys.size(); ++j)
                {
                    for(int64_t i = 0; i < d; ++i)
                    {
                        o[i] += scores[j] / sum * cacheAt(keys[j], kvHead, i, true);
                    }
                }
            }
        }
    }
    return out;
}

constexpr uint16_t SENTINEL = 0x7e00; // a NaN in fp16, a finite value in bf16; checked bitwise

/// hipDNN's own paged reference over the same host state, as float O laid out like the
/// device O ([tokens, Hq, D]).
template <typename T>
std::vector<float> hipdnnReferenceAs(const Problem& p, HostState s, float scale)
{
    using namespace hipdnn_data_sdk::utilities;
    using namespace hipdnn_test_sdk::utilities;
    const int64_t batch = p.batch();
    const int64_t d = p.headSize;
    const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
    const int64_t tokens = std::max<int64_t>(s.qTokens, 1);
    std::vector<float> out(static_cast<size_t>(tokens * p.queryHeads * d), 0.0F);
    std::unique_ptr<TensorBase<T>> q;
    std::unique_ptr<TensorBase<float>> o;
    if(p.ragged)
    {
        const std::vector<int64_t> cum(s.offsets.begin(), s.offsets.end());
        const std::vector<int64_t> dims = {batch, p.queryHeads, p.raggedMaxLength, d};
        q = std::make_unique<ShallowRaggedTensor<T>>(
            s.q.data(),
            dims,
            std::vector<int64_t>{p.raggedMaxLength * tokenHeads * d, d, tokenHeads * d, 1},
            SDPA_SEQ_AXIS,
            makeRaggedOffsetAux(cum, tokenHeads * d));
        o = std::make_unique<ShallowRaggedTensor<float>>(
            out.data(),
            dims,
            std::vector<int64_t>{p.raggedMaxLength * p.queryHeads * d, d, p.queryHeads * d, 1},
            SDPA_SEQ_AXIS,
            makeRaggedOffsetAux(cum, p.queryHeads * d));
    }
    else
    {
        const int64_t sq = p.denseQueryLength;
        const std::vector<int64_t> dims = {batch, p.queryHeads, sq, d};
        q = std::make_unique<ShallowTensor<T>>(
            s.q.data(), dims, std::vector<int64_t>{sq * tokenHeads * d, d, tokenHeads * d, 1});
        o = std::make_unique<ShallowTensor<float>>(
            out.data(), dims, std::vector<int64_t>{sq * p.queryHeads * d, d, p.queryHeads * d, 1});
    }
    const std::vector<int64_t> kvDims = {p.numBlocks, p.kvHeads, p.blockSize, d};
    const std::vector<int64_t> kvStrides
        = {p.kvHeads * p.blockSize * 2 * d, p.blockSize * 2 * d, 2 * d, 1};
    ShallowTensor<T> k(s.cache.data(), kvDims, kvStrides);
    ShallowTensor<T> v(s.cache.data() + d, kvDims, kvStrides);
    Tensor<int32_t> table({batch, 1, p.maxBlocks, 1}, {p.maxBlocks, p.maxBlocks, 1, 1});
    Tensor<int32_t> kvLengths({batch, 1, 1, 1});
    Tensor<int32_t> qLengths({batch, 1, 1, 1});
    for(int64_t b = 0; b < batch; ++b)
    {
        const int32_t kv = s.kvLengths[static_cast<size_t>(b)];
        const int64_t used = (kv + p.blockSize - 1) / p.blockSize;
        for(int64_t j = 0; j < p.maxBlocks; ++j)
        {
            // The reference checks the used ids and treats the rest as absent.
            const int32_t id
                = j < used ? s.pageTable[static_cast<size_t>(b * p.maxBlocks + j)] : -1;
            table.setHostValue(id, std::vector<int64_t>{b, 0, j, 0});
        }
        kvLengths.setHostValue(kv, std::vector<int64_t>{b, 0, 0, 0});
        qLengths.setHostValue(static_cast<int32_t>(queryRowsFor(p, s, b)),
                              std::vector<int64_t>{b, 0, 0, 0});
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpaRagged::forwardPaged<T, T, T, float, float>(
        *q,
        k,
        v,
        *o,
        table,
        table,
        kvLengths,
        (p.ragged || p.withSeqLenQ) ? &qLengths : nullptr,
        p.maxBlocks * p.blockSize,
        scale,
        p.leftBound,
        0,
        false);
    return out;
}

std::vector<float> hipdnnReference(const Problem& p, const HostState& s, float scale)
{
    if(p.type == DataType::BFLOAT16)
    {
        return hipdnnReferenceAs<hipdnn_data_sdk::types::bfloat16>(p, s, scale);
    }
    return hipdnnReferenceAs<hipdnn_data_sdk::types::half>(p, s, scale);
}

class IntegrationGpuAttentionPagedUnified : public ::testing::Test
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;

    void SetUp() override
    {
        const auto expectedArch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        if(expectedArch.empty())
        {
            SKIP_IF_NO_DEVICES();
        }
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        const auto pluginPath
            = std::filesystem::weakly_canonical(
                  hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH)
                  .string();
        const char* path = pluginPath.c_str();
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(1, &path, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
        int device = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        hipDeviceProp_t props{};
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        const std::string arch = props.gcnArchName;
        const auto base = arch.substr(0, arch.find(':'));
        if(!expectedArch.empty())
        {
            ASSERT_EQ(base, expectedArch);
        }
        if(base != "gfx942" && base != "gfx950")
        {
            GTEST_SKIP() << ENGINE << " ships for gfx942/gfx950; found " << arch;
        }
    }

    void TearDown() override
    {
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    static bool ok(const Error& error)
    {
        EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
        return error.code == ErrorCode::OK;
    }

    /// The graph, its uids and the device buffers for one problem.
    struct Built
    {
        std::unique_ptr<Graph> graph;
        DeviceBuffers memory;
        std::unordered_map<int64_t, void*> buffers;
        void* cache = nullptr;
        size_t qBytes = 0;
        size_t oBytes = 0;
        float scale = 0.0F;
    };

    enum Uid : int64_t
    {
        Q = 1,
        K = 2,
        V = 3,
        O = 4,
        PAGE_TABLE = 5,
        SEQ_LEN_KV = 6,
        SEQ_LEN_Q = 7,
        OFFSETS = 8
    };

    /// Builds the graph for @p p, planned on this engine only.
    void build(const Problem& p, Built& built)
    {
        const int64_t batch = p.batch();
        const int64_t d = p.headSize;
        const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
        const int64_t sq = p.ragged ? p.raggedMaxLength : p.denseQueryLength;
        built.graph = std::make_unique<Graph>();
        auto& graph = *built.graph;
        graph.set_io_data_type(p.type)
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT);
        const auto make = [](int64_t uid,
                             DataType type,
                             const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides) {
            return std::make_shared<TensorAttributes>(
                TensorAttributes().set_uid(uid).set_data_type(type).set_dim(dims).set_stride(
                    strides));
        };
        auto q = make(
            Q, p.type, {batch, p.queryHeads, sq, d}, {sq * tokenHeads * d, d, tokenHeads * d, 1});
        const std::vector<int64_t> kvStrides
            = {p.kvHeads * p.blockSize * 2 * d, p.blockSize * 2 * d, 2 * d, 1};
        auto k = make(K, p.type, {p.numBlocks, p.kvHeads, p.blockSize, d}, kvStrides);
        auto v = make(V, p.type, {p.numBlocks, p.kvHeads, p.blockSize, d}, kvStrides);
        auto pageTable = make(PAGE_TABLE,
                              DataType::INT32,
                              {batch, 1, p.maxBlocks, 1},
                              {p.maxBlocks, p.maxBlocks, 1, 1});
        auto seqLenKv = make(SEQ_LEN_KV, DataType::INT32, {batch, 1, 1, 1}, {1, 1, 1, 1});
        std::shared_ptr<TensorAttributes> offsets;
        if(p.ragged)
        {
            offsets = make(OFFSETS, DataType::INT32, {batch + 1, 1, 1, 1}, {1, 1, 1, 1});
            q->set_ragged_offset(offsets).set_ragged_offset_multiplier(tokenHeads * d);
        }
        built.scale = 1.0F / std::sqrt(static_cast<float>(d));
        SdpaAttributes attributes;
        attributes.set_attn_scale(built.scale)
            .set_generate_stats(false)
            .set_padding_mask(true)
            .set_seq_len_kv(seqLenKv)
            .set_paged_attention_k_table(pageTable)
            .set_paged_attention_v_table(pageTable)
            .set_paged_attention_max_seq_len_kv(static_cast<int32_t>(p.maxBlocks * p.blockSize))
            .set_diagonal_band_right_bound(0)
            .set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        if(p.leftBound >= 0)
        {
            attributes.set_diagonal_band_left_bound(p.leftBound);
        }
        if(p.withSeqLenQ || p.ragged)
        {
            attributes.set_seq_len_q(
                make(SEQ_LEN_Q, DataType::INT32, {batch, 1, 1, 1}, {1, 1, 1, 1}));
        }
        auto o = graph.sdpa(q, k, v, attributes)[0];
        o->set_uid(O)
            .set_output(true)
            .set_data_type(p.type)
            .set_dim({batch, p.queryHeads, sq, d})
            .set_stride({sq * p.queryHeads * d, d, p.queryHeads * d, 1});
        if(p.ragged)
        {
            o->set_ragged_offset(offsets).set_ragged_offset_multiplier(p.queryHeads * d);
        }
        ASSERT_TRUE(ok(graph.validate()));
        ASSERT_TRUE(ok(graph.build_operation_graph(_handle)));
        ASSERT_TRUE(ok(graph.create_execution_plan_ext(
            hipdnn_data_sdk::utilities::engineNameToId(ENGINE), {})));
        ASSERT_TRUE(ok(graph.check_support()));
        ASSERT_TRUE(ok(graph.build_plans()));
        int64_t workspace = -1;
        ASSERT_TRUE(ok(graph.get_workspace_size(workspace)));
        ASSERT_EQ(workspace, 0);

        const int64_t storageTokens = std::max<int64_t>(batch * sq, 1);
        built.qBytes = static_cast<size_t>(storageTokens * tokenHeads * d * 2);
        built.oBytes = static_cast<size_t>(storageTokens * p.queryHeads * d * 2);
        built.cache = built.memory.allocate(
            static_cast<size_t>(p.numBlocks * p.kvHeads * p.blockSize * 2 * d * 2));
        built.buffers[Q] = built.memory.allocate(built.qBytes);
        built.buffers[K] = built.cache;
        built.buffers[V] = static_cast<uint16_t*>(built.cache) + d;
        built.buffers[O] = built.memory.allocate(built.oBytes);
        built.buffers[PAGE_TABLE]
            = built.memory.allocate(static_cast<size_t>(batch * p.maxBlocks * 4));
        built.buffers[SEQ_LEN_KV] = built.memory.allocate(static_cast<size_t>(batch * 4));
        if(p.withSeqLenQ || p.ragged)
        {
            built.buffers[SEQ_LEN_Q] = built.memory.allocate(static_cast<size_t>(batch * 4));
        }
        if(p.ragged)
        {
            built.buffers[OFFSETS] = built.memory.allocate(static_cast<size_t>((batch + 1) * 4));
        }
    }

    /// Uploads @p s and poisons O with SENTINEL (stream-ordered).
    void upload(const Problem& p, const HostState& s, Built& built)
    {
        const auto copy = [&](void* dst, const void* src, size_t bytes) {
            ASSERT_EQ(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, _stream), hipSuccess);
        };
        copy(built.buffers[Q], s.q.data(), std::min(built.qBytes, s.q.size() * 2));
        copy(built.cache, s.cache.data(), s.cache.size() * 2);
        copy(built.buffers[PAGE_TABLE], s.pageTable.data(), s.pageTable.size() * 4);
        copy(built.buffers[SEQ_LEN_KV], s.kvLengths.data(), s.kvLengths.size() * 4);
        if(built.buffers.count(SEQ_LEN_Q) != 0)
        {
            copy(built.buffers[SEQ_LEN_Q], s.queryLengths.data(), s.queryLengths.size() * 4);
        }
        if(p.ragged)
        {
            copy(built.buffers[OFFSETS], s.offsets.data(), s.offsets.size() * 4);
        }
        std::vector<uint16_t> poison(built.oBytes / 2, SENTINEL);
        copy(built.buffers[O], poison.data(), built.oBytes);
    }

    /// Reads O back and compares every written row with the reference; every row the
    /// kernel must not write keeps the sentinel.
    void check(const Problem& p, const HostState& s, const Built& built)
    {
        std::vector<uint16_t> out(built.oBytes / 2);
        ASSERT_EQ(
            hipMemcpyAsync(
                out.data(), built.buffers.at(O), built.oBytes, hipMemcpyDeviceToHost, _stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        const auto expected = reference(p, s, built.scale);
        // The hipDNN paged reference (CpuFpReferenceSdpaRagged::forwardPaged) on the same
        // inputs: the test's own gather-then-dense reference must agree with it.
        const auto contract = hipdnnReference(p, s, built.scale);
        const int64_t d = p.headSize;
        std::vector<bool> written(built.oBytes / 2 / static_cast<size_t>(p.queryHeads * d), false);
        for(int64_t b = 0; b < p.batch(); ++b)
        {
            const int64_t first = firstToken(p, s, b);
            const int64_t rows
                = p.ragged ? s.queryLengths[static_cast<size_t>(b)] : p.denseQueryLength;
            const int64_t checkedRows = queryRowsFor(p, s, b);
            for(int64_t row = 0; row < rows; ++row)
            {
                written[static_cast<size_t>(first + row)] = true;
                if(row >= checkedRows)
                {
                    continue; // dense rows past seq_len_q are unspecified
                }
                for(int64_t i = 0; i < p.queryHeads * d; ++i)
                {
                    const auto index = static_cast<size_t>((first + row) * p.queryHeads * d + i);
                    const float actual = fromBits(out[index], p.type);
                    const float want = expected[index];
                    ASSERT_TRUE(std::isfinite(actual)) << "b=" << b << " row=" << row << " i=" << i;
                    if(s.kvLengths[static_cast<size_t>(b)] == 0)
                    {
                        ASSERT_EQ(actual, 0.0F) << "empty sequence b=" << b;
                    }
                    if(row < queryRowsFor(p, s, b))
                    {
                        ASSERT_NEAR(contract[index], want, 1.0e-3F + 1.0e-3F * std::abs(want))
                            << "references disagree: b=" << b << " row=" << row << " i=" << i;
                    }
                    const float error = std::abs(actual - want);
                    ASSERT_LE(error, 2.0e-2F + 2.0e-2F * std::abs(want))
                        << "b=" << b << " row=" << row << " head=" << i / d << " d=" << i % d;
                }
            }
        }
        for(size_t token = 0; token < written.size(); ++token)
        {
            if(written[token])
            {
                continue;
            }
            for(int64_t i = 0; i < p.queryHeads * d; ++i)
            {
                ASSERT_EQ(
                    out[token * static_cast<size_t>(p.queryHeads * d) + static_cast<size_t>(i)],
                    SENTINEL)
                    << "the kernel wrote unaddressed token " << token;
            }
        }
    }

    void run(const Problem& p)
    {
        Built built;
        build(p, built);
        if(::testing::Test::HasFatalFailure())
        {
            return;
        }
        const auto state = makeState(p, p.seed);
        upload(p, state, built);
        ASSERT_TRUE(ok(built.graph->execute(_handle, built.buffers, nullptr)));
        check(p, state, built);
    }
};

Problem decodeProblem(DataType type)
{
    Problem p;
    p.type = type;
    // Sequence 0 is CUDA-graph padding: no keys, and its row must come back zero.
    p.kvLengths = {0, 1, 17, 100, 255, 384};
    return p;
}

TEST_F(IntegrationGpuAttentionPagedUnified, PagedUniformDecodeBf16)
{
    run(decodeProblem(DataType::BFLOAT16));
}

TEST_F(IntegrationGpuAttentionPagedUnified, PagedUniformDecodeFp16)
{
    run(decodeProblem(DataType::HALF));
}

TEST_F(IntegrationGpuAttentionPagedUnified, SpeculativeDecodeRowsWithSeqLenQ)
{
    Problem p = decodeProblem(DataType::BFLOAT16);
    p.denseQueryLength = 3;
    p.withSeqLenQ = true;
    p.queryLengths = {3, 1, 3, 2, 3, 3};
    p.kvLengths = {0, 1, 17, 100, 255, 384};
    run(p);
}

TEST_F(IntegrationGpuAttentionPagedUnified, DecodeHeadSizes64And256)
{
    Problem small = decodeProblem(DataType::HALF);
    small.headSize = 64;
    small.queryHeads = 8;
    small.kvHeads = 8;
    small.blockSize = 64;
    small.numBlocks = 40;
    small.maxBlocks = 8;
    small.kvLengths = {511, 0, 64, 65};
    run(small);

    Problem large = decodeProblem(DataType::BFLOAT16);
    large.headSize = 256;
    large.queryHeads = 16;
    large.kvHeads = 2;
    large.blockSize = 32;
    large.numBlocks = 64;
    large.maxBlocks = 12;
    large.kvLengths = {300, 33, 0, 7};
    run(large);
}

Problem mixedProblem(DataType type)
{
    Problem p;
    p.type = type;
    p.ragged = true;
    p.raggedMaxLength = 160;
    // decode, prefill without context, a padding sequence, chunked prefill with context.
    p.queryLengths = {1, 37, 0, 64, 1};
    p.kvLengths = {200, 37, 0, 150, 9};
    p.gapsAfter = {0, 3, 0, 5, 0};
    p.withSeqLenQ = true;
    // Q as a view of a fused QKV projection.
    p.qTokenHeads = 32 + 2 * 8;
    return p;
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryOverPagedKvBf16)
{
    run(mixedProblem(DataType::BFLOAT16));
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryOverPagedKvFp16)
{
    run(mixedProblem(DataType::HALF));
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryWithSlidingWindowAndGqa5)
{
    Problem p = mixedProblem(DataType::BFLOAT16);
    p.queryHeads = 40;
    p.kvHeads = 8;
    p.qTokenHeads = 40 + 16;
    p.leftBound = 31;
    run(p);
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryWithTotalTokensBeyondSmax)
{
    // Sq (S_max) bounds each sequence, not the batch total, so the kernel's grid-stride
    // loop must reach every query block.
    Problem p = mixedProblem(DataType::HALF);
    p.raggedMaxLength = 64;
    p.queryLengths = {64, 64, 64, 64, 64};
    p.kvLengths = {64, 100, 128, 200, 64};
    p.gapsAfter = {};
    run(p);
}

/// One plan, captured once into a HIP graph and replayed with new lengths, new block ids
/// and new data: the engine reads all of them from device memory, so the replay must
/// match a fresh reference every time.
TEST_F(IntegrationGpuAttentionPagedUnified, HipGraphCaptureReplaysWithChangedLengthsAndBlocks)
{
    for(const bool ragged : {false, true})
    {
        SCOPED_TRACE(ragged ? "ragged" : "dense");
        const Problem p
            = ragged ? mixedProblem(DataType::BFLOAT16) : decodeProblem(DataType::BFLOAT16);
        Built built;
        build(p, built);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        auto state = makeState(p, 11);
        upload(p, state, built);
        // Warm-up outside capture: the first execute on a device loads the module.
        ASSERT_TRUE(ok(built.graph->execute(_handle, built.buffers, nullptr)));
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        hipGraph_t graph = nullptr;
        ASSERT_EQ(hipStreamBeginCapture(_stream, hipStreamCaptureModeThreadLocal), hipSuccess);
        const auto status = built.graph->execute(_handle, built.buffers, nullptr);
        ASSERT_EQ(hipStreamEndCapture(_stream, &graph), hipSuccess);
        ASSERT_TRUE(ok(status));
        hipGraphExec_t exec = nullptr;
        ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);

        for(uint32_t step = 0; step < 3; ++step)
        {
            SCOPED_TRACE(step);
            Problem next = p;
            // New lengths (B fixed by the plan), new shuffled blocks, new data.
            if(ragged)
            {
                next.queryLengths = {1, static_cast<int32_t>(5 + step), 1, 0, 2};
                next.kvLengths = {static_cast<int32_t>(30 + 50 * step),
                                  static_cast<int32_t>(5 + step),
                                  300,
                                  0,
                                  2};
                next.gapsAfter = {0, 0, 2, 0, 0};
            }
            else
            {
                next.kvLengths = {static_cast<int32_t>(step),
                                  0,
                                  383,
                                  static_cast<int32_t>(40 * step + 1),
                                  16,
                                  17};
            }
            state = makeState(next, 100 + step);
            upload(next, state, built);
            ASSERT_EQ(hipGraphLaunch(exec, _stream), hipSuccess);
            check(next, state, built);
        }
        ASSERT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
    }
}

} // namespace

#endif
