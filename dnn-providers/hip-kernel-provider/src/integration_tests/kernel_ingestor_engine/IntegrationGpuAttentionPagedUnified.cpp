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
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

/**
 * @file IntegrationGpuAttentionPagedUnified.cpp
 * @brief End-to-end runs of hipkernel:AttentionPagedUnified through the hipDNN frontend:
 *        paged uniform decode (dense Q), ragged Q over paged K/V (mixed decode and
 *        chunked prefill, a first origin past 0), sliding windows, separate K and V page
 *        tables, and HIP-graph capture/replay with changed sequence lengths, block ids and
 *        data. Every output is compared with a host gather-then-dense fp32 reference over
 *        the same page tables, which is itself checked against hipDNN's CPU SDPA reference.
 *        A dense Q's rows past seq_len_q, and rows with no visible key, must be zero
 *        (SdpaSemanticsRegister SEM-008, SEM-010); a ragged Q's rows past seq_len_q belong
 *        to no sequence (RFC 0014) and must keep their contents.
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
    /// Ragged origins [B+1] (non-decreasing, off[B] the end of the last block); empty =
    /// each sequence right after the previous one's live rows and gap.
    std::vector<int32_t> raggedOffsets;
    std::vector<int32_t> queryLengths; ///< seq_len_q; empty = denseQueryLength each
    std::vector<int32_t> kvLengths;
    std::vector<int32_t> gapsAfter; ///< ragged: unaddressed tokens after each sequence
    bool setMaxSeqLenKv = true; ///< false leaves max_seq_len_kv for the engine to infer
    int64_t leftBound = -1; ///< the graph's (cuDNN) left_bound; -1 = none
    int64_t qTokenHeads = 0; ///< Q token stride in heads; 0 = queryHeads
    /// V pages through its own page_table_v, which places V in other blocks than K;
    /// otherwise one tensor is both page_table_k and page_table_v.
    bool separateVTable = false;
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
    std::vector<int32_t> pageTable; ///< [B, M]: K's table (and V's, unless separate)
    std::vector<int32_t> vPageTable; ///< [B, M]: V's table
    std::vector<int32_t> kvLengths;
    std::vector<int32_t> queryLengths;
    std::vector<int32_t> offsets; ///< ragged token offsets [B+1]
    int64_t qTokens = 0; ///< tokens of the Q/O storage
};

/// seq_len_q[b]: the live query rows of sequence b.
int64_t queryRowsFor(const HostState& s, int64_t b)
{
    return s.queryLengths[static_cast<size_t>(b)];
}

/// The rows the engine writes for sequence b: Sq for a dense Q (rows past seq_len_q are
/// padding, written as zeros), the live rows for a ragged one (rows of its offset block
/// past seq_len_q belong to no sequence, so the engine leaves them alone).
int64_t rowsWritten(const Problem& p, const HostState& s, int64_t b)
{
    return p.ragged ? queryRowsFor(s, b) : p.denseQueryLength;
}

int64_t firstToken(const Problem& p, const HostState& s, int64_t b)
{
    return p.ragged ? s.offsets[static_cast<size_t>(b)] : b * p.denseQueryLength;
}

/// A ragged Q's offsets [B+1]: p.raggedOffsets, or each sequence after the previous one's
/// live rows and gap.
std::vector<int32_t> raggedOffsetsFor(const Problem& p)
{
    if(!p.raggedOffsets.empty())
    {
        return p.raggedOffsets;
    }
    std::vector<int32_t> offsets(static_cast<size_t>(p.batch() + 1), 0);
    for(int64_t b = 0; b < p.batch(); ++b)
    {
        const auto gap = p.gapsAfter.empty() ? 0 : p.gapsAfter[static_cast<size_t>(b)];
        offsets[static_cast<size_t>(b + 1)]
            = offsets[static_cast<size_t>(b)] + p.queryLengths[static_cast<size_t>(b)] + gap;
    }
    return offsets;
}

/// The tokens the Q and O storage holds: Sq rows per sequence, and for a ragged Q at
/// least Sq rows from every origin, so a write past a sequence's live rows lands in the
/// storage, where check() sees it.
int64_t storageTokens(const Problem& p)
{
    const int64_t sq = p.ragged ? p.raggedMaxLength : p.denseQueryLength;
    int64_t tokens = p.batch() * sq;
    if(p.ragged)
    {
        const auto offsets = raggedOffsetsFor(p);
        for(int64_t b = 0; b < p.batch(); ++b)
        {
            const int64_t rows = std::max<int64_t>(sq, p.queryLengths[static_cast<size_t>(b)]);
            tokens = std::max<int64_t>(tokens, offsets[static_cast<size_t>(b)] + rows);
        }
        if(p.raggedOffsets.empty())
        {
            tokens = std::max<int64_t>(tokens, offsets.back());
        }
    }
    return std::max<int64_t>(tokens, 1);
}

/// Random data, and shuffled page tables whose unused entries hold garbage block ids
/// the kernel must never follow.
HostState makeState(const Problem& p, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    HostState s;
    s.kvLengths = p.kvLengths;
    s.queryLengths
        = p.queryLengths.empty()
              ? std::vector<int32_t>(p.kvLengths.size(), static_cast<int32_t>(p.denseQueryLength))
              : p.queryLengths;
    const int64_t batch = p.batch();
    if(p.ragged)
    {
        s.offsets = raggedOffsetsFor(p);
    }
    s.qTokens = storageTokens(p);
    const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
    s.q.resize(static_cast<size_t>(s.qTokens * tokenHeads * p.headSize));
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
    s.vPageTable = s.pageTable;
    if(p.separateVTable)
    {
        // Another shuffle: V of a token sits in another block than its K.
        std::shuffle(blocks.begin(), blocks.end(), rng);
        next = 0;
        for(int64_t b = 0; b < batch; ++b)
        {
            const int64_t used
                = (s.kvLengths[static_cast<size_t>(b)] + p.blockSize - 1) / p.blockSize;
            for(int64_t i = 0; i < used; ++i)
            {
                s.vPageTable[static_cast<size_t>(b * p.maxBlocks + i)]
                    = blocks.at(next++ % blocks.size());
            }
        }
    }
    return s;
}

/// fp32 gather-then-dense reference: for each sequence, gather its K and V through their
/// page tables into dense rows, then bottom-right causal (plus window) softmax attention.
/// Rows past seq_len_q and rows with no visible key stay zero.
std::vector<float> reference(const Problem& p, const HostState& s, float scale)
{
    const int64_t d = p.headSize;
    const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
    const int64_t group = p.queryHeads / p.kvHeads;
    std::vector<float> out(static_cast<size_t>(s.qTokens * p.queryHeads * d), 0.0F);
    for(int64_t b = 0; b < p.batch(); ++b)
    {
        const int64_t kv = s.kvLengths[static_cast<size_t>(b)];
        const int64_t rows = queryRowsFor(s, b);
        const int64_t first = firstToken(p, s, b);
        const auto cacheAt = [&](int64_t token, int64_t head, int64_t dim, bool value) {
            const auto& table = value ? s.vPageTable : s.pageTable;
            const int64_t block = table[static_cast<size_t>(b * p.maxBlocks + token / p.blockSize)];
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
                    // left_bound L keeps the L keys ending at the diagonal (cuDNN).
                    if(key > diagonal || (p.leftBound > 0 && key <= diagonal - p.leftBound))
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

/// hipDNN's own SDPA reference (CpuFpReferenceSdpa, the dense path #14358's paged graphs
/// run through) over the same host state, as float O laid out like the device O
/// ([tokens, Hq, D]): K and V gathered through their page tables, Q and O dense padded
/// [B, Hq, Sq, D] with seq_len_q / seq_len_kv padding, and the ragged layout mapped by
/// the offsets.
template <typename T>
std::vector<float> hipdnnReferenceAs(const Problem& p, HostState s, float scale)
{
    using namespace hipdnn_data_sdk::utilities;
    using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;
    using hipdnn_test_sdk::utilities::SdpaFwdOptions;
    const int64_t batch = p.batch();
    const int64_t d = p.headSize;
    const int64_t tokenHeads = p.qTokenHeads > 0 ? p.qTokenHeads : p.queryHeads;
    const int64_t sq = p.ragged ? p.raggedMaxLength : p.denseQueryLength;
    const int64_t tokens = s.qTokens;

    Tensor<T> q({batch, p.queryHeads, sq, d});
    q.fillWithValue(static_cast<T>(0.0F));
    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t row = 0; row < std::min(rowsWritten(p, s, b), sq); ++row)
        {
            for(int64_t h = 0; h < p.queryHeads; ++h)
            {
                for(int64_t i = 0; i < d; ++i)
                {
                    const auto bits = s.q[static_cast<size_t>(
                        ((firstToken(p, s, b) + row) * tokenHeads + h) * d + i)];
                    q.setHostValue(static_cast<T>(fromBits(bits, p.type)),
                                   std::vector<int64_t>{b, h, row, i});
                }
            }
        }
    }
    const std::vector<int64_t> kvDims = {p.numBlocks, p.kvHeads, p.blockSize, d};
    const std::vector<int64_t> kvStrides
        = {p.kvHeads * p.blockSize * 2 * d, p.blockSize * 2 * d, 2 * d, 1};
    const ShallowTensor<T> kCache(s.cache.data(), kvDims, kvStrides);
    const ShallowTensor<T> vCache(s.cache.data() + d, kvDims, kvStrides);
    Tensor<int32_t> table({batch, 1, p.maxBlocks, 1}, {p.maxBlocks, p.maxBlocks, 1, 1});
    Tensor<int32_t> vTable({batch, 1, p.maxBlocks, 1}, {p.maxBlocks, p.maxBlocks, 1, 1});
    Tensor<int32_t> kvLengths({batch, 1, 1, 1});
    Tensor<int32_t> qLengths({batch, 1, 1, 1});
    for(int64_t b = 0; b < batch; ++b)
    {
        const int32_t kv = s.kvLengths[static_cast<size_t>(b)];
        const int64_t used = (kv + p.blockSize - 1) / p.blockSize;
        for(int64_t j = 0; j < p.maxBlocks; ++j)
        {
            // The gather reads only the used ids; the rest may hold anything.
            const auto at = static_cast<size_t>(b * p.maxBlocks + j);
            table.setHostValue(j < used ? s.pageTable[at] : -1, std::vector<int64_t>{b, 0, j, 0});
            vTable.setHostValue(j < used ? s.vPageTable[at] : -1, std::vector<int64_t>{b, 0, j, 0});
        }
        kvLengths.setHostValue(kv, std::vector<int64_t>{b, 0, 0, 0});
        qLengths.setHostValue(static_cast<int32_t>(queryRowsFor(s, b)),
                              std::vector<int64_t>{b, 0, 0, 0});
    }
    const int64_t maxSeqKv = p.maxBlocks * p.blockSize;
    const auto k = CpuFpReferenceSdpa::gatherPagedKv(kCache, table, kvLengths, maxSeqKv);
    const auto v = CpuFpReferenceSdpa::gatherPagedKv(vCache, vTable, kvLengths, maxSeqKv);
    Tensor<float> o({batch, p.queryHeads, sq, d});
    auto options = SdpaFwdOptions<float>{}
                       .setScale(scale)
                       .setBand(p.leftBound, 0, false)
                       .setSeqLens(&qLengths, &kvLengths);
    CpuFpReferenceSdpa::forward<T, T, T, float>(q, k, v, o, options);

    std::vector<float> out(static_cast<size_t>(tokens * p.queryHeads * d), 0.0F);
    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t row = 0; row < std::min(rowsWritten(p, s, b), sq); ++row)
        {
            for(int64_t h = 0; h < p.queryHeads; ++h)
            {
                for(int64_t i = 0; i < d; ++i)
                {
                    out[static_cast<size_t>(((firstToken(p, s, b) + row) * p.queryHeads + h) * d
                                            + i)]
                        = o.getHostValue(std::vector<int64_t>{b, h, row, i});
                }
            }
        }
    }
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
        OFFSETS = 8,
        PAGE_TABLE_V = 10
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
        auto pageTableV = p.separateVTable ? make(PAGE_TABLE_V,
                                                  DataType::INT32,
                                                  {batch, 1, p.maxBlocks, 1},
                                                  {p.maxBlocks, p.maxBlocks, 1, 1})
                                           : pageTable;
        auto seqLenKv = make(SEQ_LEN_KV, DataType::INT32, {batch, 1, 1, 1}, {1, 1, 1, 1});
        std::shared_ptr<TensorAttributes> offsets;
        if(p.ragged)
        {
            offsets = make(OFFSETS, DataType::INT32, {batch + 1, 1, 1, 1}, {1, 1, 1, 1});
            q->set_ragged_offset(offsets).set_ragged_offset_multiplier(tokenHeads * d);
        }
        built.scale = 1.0F / std::sqrt(static_cast<float>(d));
        SdpaAttributes attributes;
        // A paged graph carries both seq_len tensors (SdpaSemanticsRegister SEM-009).
        attributes.set_attn_scale(built.scale)
            .set_generate_stats(false)
            .set_padding_mask(true)
            .set_seq_len_q(make(SEQ_LEN_Q, DataType::INT32, {batch, 1, 1, 1}, {1, 1, 1, 1}))
            .set_seq_len_kv(seqLenKv)
            .set_paged_attention_k_table(pageTable)
            .set_paged_attention_v_table(pageTableV)
            .set_diagonal_band_right_bound(0)
            .set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        if(p.setMaxSeqLenKv)
        {
            attributes.set_paged_attention_max_seq_len_kv(
                static_cast<int32_t>(p.maxBlocks * p.blockSize));
        }
        if(p.leftBound > 0)
        {
            attributes.set_diagonal_band_left_bound(p.leftBound);
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

        const int64_t tokens = storageTokens(p);
        built.qBytes = static_cast<size_t>(tokens * tokenHeads * d * 2);
        built.oBytes = static_cast<size_t>(tokens * p.queryHeads * d * 2);
        built.cache = built.memory.allocate(
            static_cast<size_t>(p.numBlocks * p.kvHeads * p.blockSize * 2 * d * 2));
        built.buffers[Q] = built.memory.allocate(built.qBytes);
        built.buffers[K] = built.cache;
        built.buffers[V] = static_cast<uint16_t*>(built.cache) + d;
        built.buffers[O] = built.memory.allocate(built.oBytes);
        built.buffers[PAGE_TABLE]
            = built.memory.allocate(static_cast<size_t>(batch * p.maxBlocks * 4));
        if(p.separateVTable)
        {
            built.buffers[PAGE_TABLE_V]
                = built.memory.allocate(static_cast<size_t>(batch * p.maxBlocks * 4));
        }
        built.buffers[SEQ_LEN_KV] = built.memory.allocate(static_cast<size_t>(batch * 4));
        built.buffers[SEQ_LEN_Q] = built.memory.allocate(static_cast<size_t>(batch * 4));
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
        if(p.separateVTable)
        {
            copy(built.buffers[PAGE_TABLE_V], s.vPageTable.data(), s.vPageTable.size() * 4);
        }
        copy(built.buffers[SEQ_LEN_KV], s.kvLengths.data(), s.kvLengths.size() * 4);
        copy(built.buffers[SEQ_LEN_Q], s.queryLengths.data(), s.queryLengths.size() * 4);
        if(p.ragged)
        {
            copy(built.buffers[OFFSETS], s.offsets.data(), s.offsets.size() * 4);
        }
        std::vector<uint16_t> poison(built.oBytes / 2, SENTINEL);
        copy(built.buffers[O], poison.data(), built.oBytes);
    }

    /// Reads O back and compares every row the engine writes with the reference: live rows
    /// within tolerance, a dense Q's rows past seq_len_q exactly zero. Every other row (a
    /// ragged Q's gap rows included) keeps the sentinel.
    void check(const Problem& p, const HostState& s, const Built& built)
    {
        std::vector<uint16_t> out(built.oBytes / 2);
        ASSERT_EQ(
            hipMemcpyAsync(
                out.data(), built.buffers.at(O), built.oBytes, hipMemcpyDeviceToHost, _stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        const auto expected = reference(p, s, built.scale);
        // hipDNN's CPU SDPA reference on the same inputs: the test's own gather-then-dense
        // reference must agree with it.
        const auto contract = hipdnnReference(p, s, built.scale);
        const int64_t d = p.headSize;
        std::vector<bool> written(built.oBytes / 2 / static_cast<size_t>(p.queryHeads * d), false);
        for(int64_t b = 0; b < p.batch(); ++b)
        {
            const int64_t first = firstToken(p, s, b);
            const int64_t live = queryRowsFor(s, b);
            for(int64_t row = 0; row < rowsWritten(p, s, b); ++row)
            {
                written[static_cast<size_t>(first + row)] = true;
                for(int64_t i = 0; i < p.queryHeads * d; ++i)
                {
                    const auto index = static_cast<size_t>((first + row) * p.queryHeads * d + i);
                    const float actual = fromBits(out[index], p.type);
                    const float want = expected[index];
                    ASSERT_TRUE(std::isfinite(actual)) << "b=" << b << " row=" << row << " i=" << i;
                    // Bottom-right causal: live row r sees keys up to kv - live + r, so
                    // every row of an empty sequence, and the leading rows of a sequence
                    // with fewer keys than live rows, see none and must be exact zeros.
                    const int64_t kv = s.kvLengths[static_cast<size_t>(b)];
                    if(row >= live || row + kv < live)
                    {
                        ASSERT_EQ(actual, 0.0F)
                            << (row >= live ? "padding row past seq_len_q" : "row with no key")
                            << ": b=" << b << " row=" << row << " i=" << i;
                    }
                    // Both references accumulate in fp32 and differ only in summation
                    // order.
                    ASSERT_NEAR(contract[index], want, 1.0e-3F + 1.0e-3F * std::abs(want))
                        << "references disagree: b=" << b << " row=" << row << " i=" << i;
                    // The engine rounds P and O to bf16/fp16 (8 or 11 mantissa bits).
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

TEST_F(IntegrationGpuAttentionPagedUnified, PagedUniformDecodeHalf)
{
    run(decodeProblem(DataType::HALF));
}

/// Sq = 3 rows per sequence, some with fewer live rows (seq_len_q < Sq): the padded rows
/// must come back zero.
TEST_F(IntegrationGpuAttentionPagedUnified, SpeculativeDecodeRowsWithSeqLenQ)
{
    Problem p = decodeProblem(DataType::BFLOAT16);
    p.denseQueryLength = 3;
    p.queryLengths = {3, 1, 3, 2, 0, 3};
    p.kvLengths = {0, 1, 17, 100, 255, 384};
    run(p);
}

/// An unset max_seq_len_kv is valid (cuDNN infers it from the table width), and the
/// engine serves it like a set one.
TEST_F(IntegrationGpuAttentionPagedUnified, PagedDecodeWithMaxSeqLenKvUnset)
{
    Problem p = decodeProblem(DataType::HALF);
    p.setMaxSeqLenKv = false;
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
    p.gapsAfter = {0, 3, 0, 5, 2};
    // Rows past seq_len_q up to the next offset (the gaps, one after the last sequence)
    // are addressed by no sequence: the engine must not write them.
    // Q as a view of a fused QKV projection.
    p.qTokenHeads = 32 + 2 * 8;
    return p;
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryOverPagedKvBf16)
{
    run(mixedProblem(DataType::BFLOAT16));
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryOverPagedKvHalf)
{
    run(mixedProblem(DataType::HALF));
}

TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryWithSlidingWindowAndGqa5)
{
    Problem p = mixedProblem(DataType::BFLOAT16);
    p.queryHeads = 40;
    p.kvHeads = 8;
    p.qTokenHeads = 40 + 16;
    p.leftBound = 32; // a vLLM sliding window of 32 keys
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

/// A first origin past 0: the rows before it belong to no sequence and keep the sentinel,
/// and with Sq above seq_len_q the rows after the live one do too.
TEST_F(IntegrationGpuAttentionPagedUnified, RaggedQueryWithAShiftedFirstOrigin)
{
    for(const int64_t sq : {1, 4})
    {
        SCOPED_TRACE("Sq = " + std::to_string(sq));
        Problem p = mixedProblem(DataType::BFLOAT16);
        p.raggedMaxLength = sq;
        p.queryLengths = {1};
        p.kvLengths = {100};
        p.gapsAfter = {};
        p.raggedOffsets = {64, 65};
        run(p);
        if(::testing::Test::HasFatalFailure())
        {
            return;
        }
    }
}

/// 0 < seq_len_kv < seq_len_q: the bottom-right diagonal of the leading rows sits before
/// the first key, so those rows see no key and must be zero; the later rows each see the
/// keys up to their own diagonal.
TEST_F(IntegrationGpuAttentionPagedUnified, KeylessLeadingRowsWhenSeqLenKvBelowSeqLenQ)
{
    Problem dense = decodeProblem(DataType::BFLOAT16);
    dense.denseQueryLength = 5;
    dense.queryLengths = {5, 5, 3, 5, 4};
    dense.kvLengths = {2, 1, 40, 4, 0};
    run(dense);

    Problem ragged = mixedProblem(DataType::HALF);
    ragged.queryLengths = {5, 37, 0, 64, 3};
    ragged.kvLengths = {2, 37, 0, 150, 1};
    ragged.leftBound = 32;
    run(ragged);
}

/// cuDNN's form: page_table_k and page_table_v are two tensors, and the V of each token
/// sits in another block than its K.
TEST_F(IntegrationGpuAttentionPagedUnified, SeparateKAndVPageTablesDecodeBf16)
{
    Problem p = decodeProblem(DataType::BFLOAT16);
    p.separateVTable = true;
    run(p);
}

TEST_F(IntegrationGpuAttentionPagedUnified, SeparateKAndVPageTablesRaggedWithWindowHalf)
{
    Problem p = mixedProblem(DataType::HALF);
    p.separateVTable = true;
    p.leftBound = 32;
    p.setMaxSeqLenKv = false;
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
