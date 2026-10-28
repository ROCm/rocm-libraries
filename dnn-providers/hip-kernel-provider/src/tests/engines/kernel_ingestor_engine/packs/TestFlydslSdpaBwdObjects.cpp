// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdObjects.cpp
 * @brief The shipped hipkernel:flydsl_sdpa_bwd_merged and hipkernel:flydsl_sdpa_bwd_split
 *        objects, run end to end on the device and checked against a host reference.
 *
 * Each case builds one SDPA backward graph, asks the engine manager whether the engine under
 * test claims it, builds that engine's execution plan (the manager takes the engine by id, so
 * no other engine can serve the graph) and executes it on deterministic inputs. dQ, dK, dV and, on a
 * bias graph, dBias are compared with a double-precision backward pass over the same inputs:
 * the forward output and the log-sum-exp the kernels consume are computed by the reference
 * from those inputs, with the same dropout mask, and rounded to the graph's dtype. A graph
 * the engine must decline is checked to be absent from the manager's applicable engines and
 * refused when the engine is selected by id.
 *
 * The cases run on every device whose architecture has objects in the install: gfx950, and a
 * gfx11 device served by the gfx11-generic content. They skip on any other device and on an
 * install that staged no shard for the device (flydslShipsForThisDevice). On a device the
 * install ships them for, a missing object fails. The dropout reference draws from the
 * 64-bit Philox stream on both architectures.
 *
 * Tolerances are relative to the largest reference element of each gradient: a kernel that
 * is wrong, rather than rounded, misses by a good fraction of it.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

namespace uid = hip_kernel_provider::kernel_ingestor_engine::testing::sdpa_bwd_uid;

constexpr std::string_view MERGED_ENGINE = "hipkernel:flydsl_sdpa_bwd_merged";
constexpr std::string_view SPLIT_ENGINE = "hipkernel:flydsl_sdpa_bwd_split";

// The ops whose objects an engine launches; the split engine launches one per slot.
const std::vector<std::string_view> MERGED_OPS{"sdpa_bwd_merged"};
const std::vector<std::string_view> SPLIT_OPS{"sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"};

// ---------------------------------------------------------------------------
// Host data
// ---------------------------------------------------------------------------

/// What a graph computes: its shape and the features it turns on.
struct BwdCase
{
    DataType dtype = DataType::HALF;
    int64_t batch = 2;
    int64_t heads = 4;
    int64_t kvHeads = 4;
    int64_t seqQ = 96;
    int64_t seqKv = 96;
    int64_t headDim = 64;
    bool causal = false;
    bool bias = false;
    /// With a bias: whether the graph asks for its gradient, and which axes the bias (and so
    /// its gradient) broadcasts over.
    bool dbias = true;
    bool biasOverBatch = false;
    bool biasOverHead = false;
    /// Per-batch valid lengths of a padded varlen graph; empty for a dense one.
    std::vector<int32_t> lengthsQ;
    std::vector<int32_t> lengthsKv;
    float dropout = 0.0F;
    int64_t seed = 0x1234567;
    int64_t offset = 77;

    bool padded() const
    {
        return !lengthsQ.empty();
    }

    int32_t validQ(int64_t batchIndex) const
    {
        return padded() ? lengthsQ[static_cast<size_t>(batchIndex)] : static_cast<int32_t>(seqQ);
    }

    int32_t validKv(int64_t batchIndex) const
    {
        return padded() ? lengthsKv[static_cast<size_t>(batchIndex)] : static_cast<int32_t>(seqKv);
    }

    /// Elements between two rows of a (S, D) tensor and of the (Sq, Skv) bias: the width
    /// rounded up to a whole 16-byte vector, as the graph builders lay them out.
    int64_t rowStride() const
    {
        return ceilToVectorOf(dtype, headDim);
    }

    int64_t biasRowStride() const
    {
        return ceilToVectorOf(dtype, seqKv);
    }

    int64_t biasBatch() const
    {
        return biasOverBatch ? 1 : batch;
    }

    int64_t biasHeads() const
    {
        return biasOverHead ? 1 : heads;
    }

    size_t biasElements() const
    {
        return static_cast<size_t>(biasBatch() * biasHeads() * seqQ * seqKv);
    }

    double scale() const
    {
        return static_cast<double>(
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(headDim))));
    }
};

SdpaBwdSpec specOf(const BwdCase& c)
{
    SdpaBwdSpec spec;
    spec.dtype = c.dtype;
    spec.batch = c.batch;
    spec.heads = c.heads;
    spec.kvHeads = c.kvHeads;
    spec.seqQ = c.seqQ;
    spec.seqKv = c.seqKv;
    spec.headDimQk = c.headDim;
    spec.headDimV = c.headDim;
    const auto scale = static_cast<float>(c.scale());
    spec = spec.withEdit([=](SdpaBwdGraph& g) { g.attributes.attnScaleValue = scale; });
    if(c.causal)
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; });
    }
    if(c.bias)
    {
        const auto overBatch = c.biasOverBatch;
        const auto overHead = c.biasOverHead;
        const auto withDbias = c.dbias;
        spec = spec.withEdit(
            [=](SdpaBwdGraph& g) { addBroadcastBias(g, overBatch, overHead, withDbias); });
    }
    if(c.padded())
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { addSeqLens(g); });
    }
    if(c.dropout > 0.0F)
    {
        const auto probability = c.dropout;
        spec = spec.withEdit(
            [=](SdpaBwdGraph& g) { addDropout(g, probability, DataType::INT64, false); });
    }
    return spec;
}

/// Deterministic k/32 values for |k| <= 32: exact in every dtype under test, so the reference
/// reads exactly what the device does.
std::vector<double> fillValues(size_t count, uint32_t seed)
{
    std::vector<double> values(count);
    uint32_t state = seed;
    for(auto& value : values)
    {
        state = state * 1664525U + 1013904223U;
        value = static_cast<double>(static_cast<int32_t>((state >> 16) % 65U) - 32) / 32.0;
    }
    return values;
}

/// The Philox-4x32-10 block the kernels draw a dropout mask from (Triton's round).
std::array<uint32_t, 4> philox4x32(uint64_t seed, uint64_t offset)
{
    constexpr uint32_t MUL_A = 0xD2511F53U;
    constexpr uint32_t MUL_B = 0xCD9E8D57U;
    constexpr uint32_t KEY_A = 0x9E3779B9U;
    constexpr uint32_t KEY_B = 0xBB67AE85U;
    auto c0 = static_cast<uint32_t>(offset);
    auto c1 = static_cast<uint32_t>(offset >> 32U);
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    auto k0 = static_cast<uint32_t>(seed);
    auto k1 = static_cast<uint32_t>(seed >> 32U);
    for(int round = 0; round < 10; ++round)
    {
        const uint32_t previous0 = c0;
        const uint32_t previous2 = c2;
        const uint64_t productB = static_cast<uint64_t>(MUL_B) * previous2;
        const uint64_t productA = static_cast<uint64_t>(MUL_A) * previous0;
        c0 = static_cast<uint32_t>(productB >> 32U) ^ c1 ^ k0;
        c2 = static_cast<uint32_t>(productA >> 32U) ^ c3 ^ k1;
        c1 = static_cast<uint32_t>(productB);
        c3 = static_cast<uint32_t>(productA);
        k0 += KEY_A;
        k1 += KEY_B;
    }
    return {c0, c1, c2, c3};
}

/// Whether the kernels keep score (batch, head, row, column): one Philox block per four
/// columns of a row, a plane per (batch, query head), `ceil(Skv / 4)` blocks per row.
bool keeps(const BwdCase& c, int64_t batchIndex, int64_t head, int64_t row, int64_t column)
{
    if(c.dropout <= 0.0F)
    {
        return true;
    }
    const int64_t rowBlocks = (c.seqKv + 3) / 4;
    const int64_t plane = batchIndex * c.heads + head;
    const auto offset
        = static_cast<uint64_t>(c.offset + (plane * c.seqQ + row) * rowBlocks + column / 4);
    const auto block = philox4x32(static_cast<uint64_t>(c.seed), offset);
    const auto draw = static_cast<int32_t>(block[static_cast<size_t>(column % 4)]);
    const double threshold = (static_cast<double>(c.dropout) - 0.5) * 4294967295.0;
    const auto clamped = static_cast<int32_t>(std::clamp(threshold, -2147483648.0, 2147483647.0));
    return draw > clamped;
}

double dropoutScale(const BwdCase& c)
{
    return c.dropout > 0.0F ? static_cast<double>(
                                  static_cast<float>(1.0 / (1.0 - static_cast<double>(c.dropout))))
                            : 1.0;
}

/// Every operand the reference and the device share, as the device will read it.
struct Inputs
{
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> dO;
    std::vector<double> bias;
};

size_t at(int64_t b, int64_t h, int64_t s, int64_t e, int64_t heads, int64_t seq, int64_t width)
{
    return static_cast<size_t>(((b * heads + h) * seq + s) * width + e);
}

/// Index of a score's bias in the dense (Bb, Hb, Sq, Skv) bias, whose broadcast axes have
/// extent 1.
size_t biasAt(const BwdCase& c, int64_t b, int64_t h, int64_t s, int64_t e)
{
    return at(
        c.biasOverBatch ? 0 : b, c.biasOverHead ? 0 : h, s, e, c.biasHeads(), c.seqQ, c.seqKv);
}

/// Whether query row @p row may see key @p column.
bool visible(const BwdCase& c, int64_t batchIndex, int64_t row, int64_t column)
{
    return column < c.validKv(batchIndex) && (!c.causal || column <= row);
}

/// Score of one (row, key) pair: the scaled dot product plus the bias.
double scoreOf(const BwdCase& c,
               const Inputs& in,
               int64_t batchIndex,
               int64_t head,
               int64_t row,
               int64_t column)
{
    const int64_t kvHead = head / (c.heads / c.kvHeads);
    double dot = 0.0;
    for(int64_t e = 0; e < c.headDim; ++e)
    {
        dot += in.q[at(batchIndex, head, row, e, c.heads, c.seqQ, c.headDim)]
               * in.k[at(batchIndex, kvHead, column, e, c.kvHeads, c.seqKv, c.headDim)];
    }
    double score = c.scale() * dot;
    if(c.bias)
    {
        score += in.bias[biasAt(c, batchIndex, head, row, column)];
    }
    return score;
}

/// What the forward pass hands the backward: its output and the log-sum-exp of each row.
struct Forward
{
    std::vector<double> o;
    std::vector<float> lse;
};

Forward forwardOf(const BwdCase& c, const Inputs& in)
{
    Forward out;
    out.o.assign(static_cast<size_t>(c.batch * c.heads * c.seqQ * c.headDim), 0.0);
    out.lse.assign(static_cast<size_t>(c.batch * c.heads * c.seqQ), 0.0F);
    const double dropScale = dropoutScale(c);
    for(int64_t b = 0; b < c.batch; ++b)
    {
        for(int64_t h = 0; h < c.heads; ++h)
        {
            const int64_t kvHead = h / (c.heads / c.kvHeads);
            for(int64_t row = 0; row < c.validQ(b); ++row)
            {
                std::vector<double> scores(static_cast<size_t>(c.seqKv), 0.0);
                double maximum = -std::numeric_limits<double>::infinity();
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(visible(c, b, row, j))
                    {
                        scores[static_cast<size_t>(j)] = scoreOf(c, in, b, h, row, j);
                        maximum = std::max(maximum, scores[static_cast<size_t>(j)]);
                    }
                }
                double sum = 0.0;
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(visible(c, b, row, j))
                    {
                        sum += std::exp(scores[static_cast<size_t>(j)] - maximum);
                    }
                }
                const double lse = maximum + std::log(sum);
                out.lse[static_cast<size_t>((b * c.heads + h) * c.seqQ + row)]
                    = static_cast<float>(lse);
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(!visible(c, b, row, j) || !keeps(c, b, h, row, j))
                    {
                        continue;
                    }
                    const double weight
                        = std::exp(scores[static_cast<size_t>(j)] - lse) * dropScale;
                    for(int64_t e = 0; e < c.headDim; ++e)
                    {
                        out.o[at(b, h, row, e, c.heads, c.seqQ, c.headDim)]
                            += weight * in.v[at(b, kvHead, j, e, c.kvHeads, c.seqKv, c.headDim)];
                    }
                }
            }
        }
    }
    return out;
}

/// The gradients a backward pass produces, each in the graph's dense (B, H, S, D) layout.
struct Gradients
{
    std::vector<double> dq;
    std::vector<double> dk;
    std::vector<double> dv;
    std::vector<double> dbias;
};

/// The backward pass in double precision, from the (rounded) output and statistics the
/// device is given. dBias is dS, the gradient of the score before its scale.
Gradients backwardOf(const BwdCase& c,
                     const Inputs& in,
                     const std::vector<double>& o,
                     const std::vector<float>& lse)
{
    Gradients grads;
    grads.dq.assign(in.q.size(), 0.0);
    grads.dk.assign(in.k.size(), 0.0);
    grads.dv.assign(in.v.size(), 0.0);
    grads.dbias.assign(in.bias.size(), 0.0);
    const double dropScale = dropoutScale(c);
    for(int64_t b = 0; b < c.batch; ++b)
    {
        for(int64_t h = 0; h < c.heads; ++h)
        {
            const int64_t kvHead = h / (c.heads / c.kvHeads);
            for(int64_t row = 0; row < c.validQ(b); ++row)
            {
                double delta = 0.0;
                for(int64_t e = 0; e < c.headDim; ++e)
                {
                    const auto index = at(b, h, row, e, c.heads, c.seqQ, c.headDim);
                    delta += in.dO[index] * o[index];
                }
                const auto rowLse = static_cast<double>(
                    lse[static_cast<size_t>((b * c.heads + h) * c.seqQ + row)]);
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(!visible(c, b, row, j))
                    {
                        continue;
                    }
                    const double p = std::exp(scoreOf(c, in, b, h, row, j) - rowLse);
                    double dp = 0.0;
                    for(int64_t e = 0; e < c.headDim; ++e)
                    {
                        dp += in.dO[at(b, h, row, e, c.heads, c.seqQ, c.headDim)]
                              * in.v[at(b, kvHead, j, e, c.kvHeads, c.seqKv, c.headDim)];
                    }
                    const bool kept = keeps(c, b, h, row, j);
                    const double ds = p * ((kept ? dp * dropScale : 0.0) - delta);
                    const double pDropped = kept ? p * dropScale : 0.0;
                    for(int64_t e = 0; e < c.headDim; ++e)
                    {
                        const auto qIndex = at(b, h, row, e, c.heads, c.seqQ, c.headDim);
                        const auto kvIndex = at(b, kvHead, j, e, c.kvHeads, c.seqKv, c.headDim);
                        grads.dq[qIndex] += c.scale() * ds * in.k[kvIndex];
                        grads.dk[kvIndex] += c.scale() * ds * in.q[qIndex];
                        grads.dv[kvIndex] += pDropped * in.dO[qIndex];
                    }
                    if(c.bias)
                    {
                        // A broadcast axis sums the scores it repeats over.
                        grads.dbias[biasAt(c, b, h, row, j)] += ds;
                    }
                }
            }
        }
    }
    return grads;
}

// ---------------------------------------------------------------------------
// Device data
// ---------------------------------------------------------------------------

/// Device memory freed on every exit, including a failed ASSERT part-way through a case.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
        : _bytes(bytes)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_ptr, bytes));
    }

    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_ptr));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    void* get() const
    {
        return _ptr;
    }

    template <typename T>
    void upload(const std::vector<T>& values)
    {
        ASSERT_EQ(values.size() * sizeof(T), _bytes);
        ASSERT_EQ(hipSuccess, hipMemcpy(_ptr, values.data(), _bytes, hipMemcpyHostToDevice));
    }

    /// All-ones bytes: NaN in f16, bf16 and f32, so an element no kernel wrote fails the
    /// comparison instead of reading as a zero.
    void poison()
    {
        ASSERT_EQ(hipSuccess, hipMemset(_ptr, 0xFF, _bytes));
    }

    template <typename T>
    std::vector<T> download() const
    {
        std::vector<T> values(_bytes / sizeof(T));
        EXPECT_EQ(hipSuccess, hipMemcpy(values.data(), _ptr, _bytes, hipMemcpyDeviceToHost));
        return values;
    }

private:
    void* _ptr = nullptr;
    size_t _bytes;
};

template <typename T>
std::vector<T> roundedTo(const std::vector<double>& values)
{
    std::vector<T> rounded(values.size());
    std::transform(values.begin(), values.end(), rounded.begin(), [](double value) {
        return T(static_cast<float>(value));
    });
    return rounded;
}

template <typename T>
std::vector<double> widened(const std::vector<T>& values)
{
    std::vector<double> wide(values.size());
    std::transform(values.begin(), values.end(), wide.begin(), [](const T& value) {
        return static_cast<double>(static_cast<float>(value));
    });
    return wide;
}

/// @p dense, rows of @p width elements, rounded to @p T into rows @p stride apart; the
/// elements between rows are zero.
template <typename T>
std::vector<T> packRows(const std::vector<double>& dense, int64_t width, int64_t stride)
{
    const auto rows = dense.size() / static_cast<size_t>(width);
    std::vector<T> packed(rows * static_cast<size_t>(stride), T(0.0F));
    for(size_t row = 0; row < rows; ++row)
    {
        for(int64_t e = 0; e < width; ++e)
        {
            packed[row * static_cast<size_t>(stride) + static_cast<size_t>(e)]
                = T(static_cast<float>(
                    dense[row * static_cast<size_t>(width) + static_cast<size_t>(e)]));
        }
    }
    return packed;
}

/// The inverse of packRows: the @p width elements of each row of @p packed, widened.
template <typename T>
std::vector<double> unpackRows(const std::vector<T>& packed, int64_t width, int64_t stride)
{
    const auto rows = packed.size() / static_cast<size_t>(stride);
    std::vector<double> dense(rows * static_cast<size_t>(width));
    for(size_t row = 0; row < rows; ++row)
    {
        for(int64_t e = 0; e < width; ++e)
        {
            dense[row * static_cast<size_t>(width) + static_cast<size_t>(e)]
                = static_cast<double>(static_cast<float>(
                    packed[row * static_cast<size_t>(stride) + static_cast<size_t>(e)]));
        }
    }
    return dense;
}

/// Relative tolerance of a gradient in @p dtype, against the largest reference element.
double toleranceOf(DataType dtype)
{
    switch(dtype)
    {
    case DataType::FLOAT:
        return 4e-3;
    case DataType::HALF:
        return 2e-2;
    default:
        return 4e-2;
    }
}

/// Compares @p got with @p expected over the elements @p inScope names; an unwritten
/// (NaN) element is a miss.
void expectGradient(const char* name,
                    const std::vector<double>& got,
                    const std::vector<double>& expected,
                    const std::vector<bool>& inScope,
                    double relativeTolerance)
{
    ASSERT_EQ(got.size(), expected.size());
    double largest = 0.0;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        largest = inScope[i] ? std::max(largest, std::abs(expected[i])) : largest;
    }
    ASSERT_GT(largest, 0.0) << name << ": the reference is all zero, the case checks nothing";
    const double tolerance = relativeTolerance * largest;
    double worst = 0.0;
    size_t worstIndex = 0;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        if(!inScope[i])
        {
            continue;
        }
        const double error = std::isfinite(got[i]) ? std::abs(got[i] - expected[i])
                                                   : std::numeric_limits<double>::infinity();
        if(error > worst)
        {
            worst = error;
            worstIndex = i;
        }
    }
    EXPECT_LE(worst, tolerance) << name << ": worst error at element " << worstIndex << " (got "
                                << got[worstIndex] << ", expected " << expected[worstIndex]
                                << "), against a bound of " << tolerance << " ("
                                << relativeTolerance << " of the largest element " << largest
                                << ")";
}

/// Which elements of a (B, H, S, D) tensor lie inside each batch's valid length.
std::vector<bool> scopeOf(const BwdCase& c, bool queryAxis, int64_t heads, int64_t width)
{
    const int64_t seq = queryAxis ? c.seqQ : c.seqKv;
    std::vector<bool> scope(static_cast<size_t>(c.batch * heads * seq * width), false);
    for(int64_t b = 0; b < c.batch; ++b)
    {
        const int64_t valid = queryAxis ? c.validQ(b) : c.validKv(b);
        for(int64_t h = 0; h < heads; ++h)
        {
            for(int64_t s = 0; s < valid; ++s)
            {
                for(int64_t e = 0; e < width; ++e)
                {
                    scope[at(b, h, s, e, heads, seq, width)] = true;
                }
            }
        }
    }
    return scope;
}

// ---------------------------------------------------------------------------
// Engine runs
// ---------------------------------------------------------------------------

hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper
    wrap(const flatbuffers::FlatBufferBuilder& builder)
{
    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(builder.GetBufferPointer(),
                                                                      builder.GetSize());
}

int64_t idOf(std::string_view engine)
{
    return hipdnn_data_sdk::utilities::engineNameToId(engine);
}

void stubEngineId(hipdnn_test_sdk::utilities::MockEngineConfig& config, int64_t engineId)
{
    EXPECT_CALL(config, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(config, engineId()).WillRepeatedly(::testing::Return(engineId));
}

/// Executes @p c on @p engine and checks every gradient against the reference.
template <typename T>
void expectEngineMatchesReference(std::string_view engine, const BwdCase& c)
{
    const auto builder = buildSdpaBwdGraph(specOf(c));
    const auto graph = wrap(builder);

    core::Container container;
    auto& manager = container.getEngineManager();
    Handle handle;
    const auto engineId = idOf(engine);

    const auto applicable = manager.getApplicableEngineIds(handle, graph);
    ASSERT_NE(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
        << engine << " does not claim the graph";

    hipdnn_test_sdk::utilities::MockEngineConfig config;
    stubEngineId(config, engineId);
    Context context;
    ASSERT_NO_THROW(manager.initializeExecutionContext(handle, graph, config, context));
    ASSERT_TRUE(context.hasValidPlan());

    // Inputs, and the forward results the backward consumes, rounded to the graph's dtype.
    Inputs in;
    const auto qCount = static_cast<size_t>(c.batch * c.heads * c.seqQ * c.headDim);
    const auto kvCount = static_cast<size_t>(c.batch * c.kvHeads * c.seqKv * c.headDim);
    in.q = widened(roundedTo<T>(fillValues(qCount, 1U)));
    in.k = widened(roundedTo<T>(fillValues(kvCount, 2U)));
    in.v = widened(roundedTo<T>(fillValues(kvCount, 3U)));
    in.dO = widened(roundedTo<T>(fillValues(qCount, 4U)));
    if(c.bias)
    {
        in.bias = widened(roundedTo<T>(fillValues(c.biasElements(), 5U)));
    }
    // A padded graph's rows past its lengths carry no data: the gradient's input is zero
    // there, as a packed batch's padding is.
    if(c.padded())
    {
        const auto scope = scopeOf(c, true, c.heads, c.headDim);
        for(size_t i = 0; i < in.dO.size(); ++i)
        {
            in.dO[i] = scope[i] ? in.dO[i] : 0.0;
        }
    }
    const auto forward = forwardOf(c, in);
    const auto oRounded = widened(roundedTo<T>(forward.o));
    const auto expected = backwardOf(c, in, oRounded, forward.lse);

    // The graph's rows are a whole vector apart, so a head dim that is not a whole number of
    // vectors leaves unused elements between rows.
    const auto stride = c.rowStride();
    const auto paddedQ = qCount / static_cast<size_t>(c.headDim) * static_cast<size_t>(stride);
    const auto paddedKv = kvCount / static_cast<size_t>(c.headDim) * static_cast<size_t>(stride);
    DeviceBuffer q(paddedQ * sizeof(T));
    DeviceBuffer k(paddedKv * sizeof(T));
    DeviceBuffer v(paddedKv * sizeof(T));
    DeviceBuffer o(paddedQ * sizeof(T));
    DeviceBuffer dO(paddedQ * sizeof(T));
    DeviceBuffer stats(forward.lse.size() * sizeof(float));
    DeviceBuffer dq(paddedQ * sizeof(T));
    DeviceBuffer dk(paddedKv * sizeof(T));
    DeviceBuffer dv(paddedKv * sizeof(T));
    q.upload(packRows<T>(in.q, c.headDim, stride));
    k.upload(packRows<T>(in.k, c.headDim, stride));
    v.upload(packRows<T>(in.v, c.headDim, stride));
    o.upload(packRows<T>(oRounded, c.headDim, stride));
    dO.upload(packRows<T>(in.dO, c.headDim, stride));
    stats.upload(forward.lse);
    dq.poison();
    dk.poison();
    dv.poison();
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{uid::Q, q.get()},
                                                    {uid::K, k.get()},
                                                    {uid::V, v.get()},
                                                    {uid::O, o.get()},
                                                    {uid::DO, dO.get()},
                                                    {uid::STATS, stats.get()},
                                                    {uid::DQ, dq.get()},
                                                    {uid::DK, dk.get()},
                                                    {uid::DV, dv.get()}};

    const auto biasStride = c.biasRowStride();
    const auto paddedBias
        = in.bias.size() / static_cast<size_t>(c.seqKv) * static_cast<size_t>(biasStride);
    DeviceBuffer bias(paddedBias * sizeof(T));
    DeviceBuffer dbias(paddedBias * sizeof(T));
    if(c.bias)
    {
        bias.upload(packRows<T>(in.bias, c.seqKv, biasStride));
        buffers.push_back({uid::BIAS, bias.get()});
        if(c.dbias)
        {
            dbias.poison();
            buffers.push_back({uid::DBIAS, dbias.get()});
        }
    }
    DeviceBuffer lengthsQ(c.lengthsQ.size() * sizeof(int32_t));
    DeviceBuffer lengthsKv(c.lengthsKv.size() * sizeof(int32_t));
    if(c.padded())
    {
        lengthsQ.upload(c.lengthsQ);
        lengthsKv.upload(c.lengthsKv);
        buffers.push_back({uid::SEQ_LEN_Q, lengthsQ.get()});
        buffers.push_back({uid::SEQ_LEN_KV, lengthsKv.get()});
    }
    DeviceBuffer seed(sizeof(int64_t));
    DeviceBuffer offset(sizeof(int64_t));
    if(c.dropout > 0.0F)
    {
        seed.upload(std::vector<int64_t>{c.seed});
        offset.upload(std::vector<int64_t>{c.offset});
        buffers.push_back({uid::SEED, seed.get()});
        buffers.push_back({uid::OFFSET, offset.get()});
    }

    // Poisoned, so a kernel that reads workspace it did not write fails the case.
    const auto workspaceBytes = context.plan().getWorkspaceSize(handle);
    const GuardedWorkspace workspace(workspaceBytes);
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipSuccess, hipMemset(workspace.get(), 0xFF, workspaceBytes));
    }
    ASSERT_NO_THROW(context.plan().execute(
        handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    EXPECT_EQ(workspace.overwrittenGuardBytes(), 0U)
        << "the launch wrote past the " << workspaceBytes << "-byte workspace it reported";

    const double tolerance = toleranceOf(c.dtype);
    expectGradient("dQ",
                   unpackRows(dq.download<T>(), c.headDim, stride),
                   expected.dq,
                   scopeOf(c, true, c.heads, c.headDim),
                   tolerance);
    expectGradient("dK",
                   unpackRows(dk.download<T>(), c.headDim, stride),
                   expected.dk,
                   scopeOf(c, false, c.kvHeads, c.headDim),
                   tolerance);
    expectGradient("dV",
                   unpackRows(dv.download<T>(), c.headDim, stride),
                   expected.dv,
                   scopeOf(c, false, c.kvHeads, c.headDim),
                   tolerance);
    if(c.bias && c.dbias)
    {
        expectGradient("dBias",
                       unpackRows(dbias.download<T>(), c.seqKv, biasStride),
                       expected.dbias,
                       std::vector<bool>(expected.dbias.size(), true),
                       tolerance);
    }
}

void expectCase(std::string_view engine, const BwdCase& c)
{
    switch(c.dtype)
    {
    case DataType::HALF:
        expectEngineMatchesReference<hipdnn_data_sdk::types::half>(engine, c);
        break;
    case DataType::BFLOAT16:
        expectEngineMatchesReference<hipdnn_data_sdk::types::bfloat16>(engine, c);
        break;
    default:
        expectEngineMatchesReference<float>(engine, c);
        break;
    }
}

/// The graph is not among the engine's applicable graphs, and selecting the engine by id
/// for it is refused.
void expectDeclined(std::string_view engine, const SdpaBwdSpec& spec)
{
    const auto builder = buildSdpaBwdGraph(spec);
    const auto graph = wrap(builder);
    core::Container container;
    auto& manager = container.getEngineManager();
    Handle handle;
    const auto engineId = idOf(engine);

    const auto applicable = manager.getApplicableEngineIds(handle, graph);
    EXPECT_EQ(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
        << engine << " claims a graph it must decline";

    hipdnn_test_sdk::utilities::MockEngineConfig config;
    stubEngineId(config, engineId);
    Context context;
    EXPECT_THROW(manager.initializeExecutionContext(handle, graph, config, context),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Skips the case unless this install ships every op of an engine for the current device,
/// whichever architecture's content serves it.
class FlydslSdpaBwdDevice : public ::testing::Test
{
protected:
    explicit FlydslSdpaBwdDevice(const std::vector<std::string_view>& ops)
        : _ops(ops)
    {
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        for(const auto op : _ops)
        {
            if(!flydslShipsForThisDevice(op))
            {
                GTEST_SKIP() << "this install ships no " << op << " objects for "
                             << packedArchDeviceProperties().gcnArchName;
            }
        }
    }

private:
    const std::vector<std::string_view>& _ops;
};

class TestGpuFlydslSdpaBwdMergedDispatch : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdMergedDispatch()
        : FlydslSdpaBwdDevice(MERGED_OPS)
    {
    }
};

class TestGpuFlydslSdpaBwdSplitDispatch : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitDispatch()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

class TestGpuFlydslSdpaBwdSplitDispatchFp32 : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitDispatchFp32()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

BwdCase plainCase(DataType dtype)
{
    BwdCase c;
    c.dtype = dtype;
    return c;
}

BwdCase causalCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.headDim = 128;
    c.causal = true;
    return c;
}

BwdCase biasCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.bias = true;
    return c;
}

BwdCase dropoutCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.dropout = 0.2F;
    return c;
}

BwdCase groupedCase(DataType dtype)
{
    BwdCase c = causalCase(dtype);
    c.heads = 8;
    c.kvHeads = 2;
    return c;
}

BwdCase paddedCase(DataType dtype)
{
    BwdCase c = plainCase(dtype);
    c.lengthsQ = {96, 70};
    c.lengthsKv = {96, 50};
    return c;
}

/// Dropout over a query length that differs from the key length, the key length not a multiple
/// of 4: a Philox row holds a partial 4-wide draw.
BwdCase raggedDropoutCase(DataType dtype, bool causal)
{
    BwdCase c = dropoutCase(dtype);
    c.seqQ = 70;
    c.seqKv = 45;
    c.causal = causal;
    return c;
}

/// A bias, and its gradient unless @p dbias is false, broadcast over the named axes.
BwdCase broadcastBiasCase(DataType dtype, bool overBatch, bool overHead, bool dbias = true)
{
    BwdCase c = biasCase(dtype);
    c.biasOverBatch = overBatch;
    c.biasOverHead = overHead;
    c.dbias = dbias;
    return c;
}

/// A head dim that is not a power of two; the rows are a whole vector apart.
BwdCase headDimCase(DataType dtype, int64_t headDim)
{
    BwdCase c = plainCase(dtype);
    c.headDim = headDim;
    return c;
}

/// Bias beside a padding mask is outside both engines' contract.
SdpaBwdSpec biasWithPaddingSpec()
{
    return SdpaBwdSpec{}.withEdit([](SdpaBwdGraph& g) {
        addBias(g);
        addSeqLens(g);
    });
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, PlainMatchesTheReference)
{
    expectCase(MERGED_ENGINE, plainCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, CausalMatchesTheReference)
{
    expectCase(MERGED_ENGINE, causalCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, BiasAndDbiasMatchTheReference)
{
    expectCase(MERGED_ENGINE, biasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, DropoutMatchesTheReference)
{
    expectCase(MERGED_ENGINE, dropoutCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, GroupedQueryMatchesTheReference)
{
    expectCase(MERGED_ENGINE, groupedCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, PaddedVarlenMatchesTheReference)
{
    expectCase(MERGED_ENGINE, paddedCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdMergedDispatch, DeclinesAFullPrecisionGraph)
{
    expectDeclined(MERGED_ENGINE, fp32Spec());
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, PlainMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, plainCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, CausalMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, causalCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasAndDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, biasCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DropoutMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, dropoutCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, GroupedQueryMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, groupedCase(DataType::BFLOAT16));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, PaddedVarlenMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, paddedCase(DataType::HALF));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, PlainMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, plainCase(DataType::FLOAT));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, CausalMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, causalCase(DataType::FLOAT));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, BiasAndDbiasMatchTheReference)
{
    expectCase(SPLIT_ENGINE, biasCase(DataType::FLOAT));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DropoutOfUnequalLengthsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, raggedDropoutCase(DataType::HALF, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, CausalDropoutOfUnequalLengthsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, raggedDropoutCase(DataType::BFLOAT16, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasBroadcastOverBatchMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::BFLOAT16, true, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasBroadcastOverHeadsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::HALF, false, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasBroadcastOverBatchAndHeadsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::BFLOAT16, true, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, BiasWithoutDbiasMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::HALF, false, false, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, HeadDimOf72MatchesTheReference)
{
    expectCase(SPLIT_ENGINE, headDimCase(DataType::BFLOAT16, 72));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, HeadDimOf77AtAPaddedRowStrideMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, headDimCase(DataType::HALF, 77));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, DropoutOfUnequalLengthsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, raggedDropoutCase(DataType::FLOAT, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, BiasBroadcastOverBatchAndHeadsMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::FLOAT, true, true));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, BiasWithoutDbiasMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, broadcastBiasCase(DataType::FLOAT, false, false, false));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatchFp32, HeadDimOf37AtAPaddedRowStrideMatchesTheReference)
{
    expectCase(SPLIT_ENGINE, headDimCase(DataType::FLOAT, 37));
}

TEST_F(TestGpuFlydslSdpaBwdSplitDispatch, DeclinesBiasBesideAPaddingMask)
{
    expectDeclined(SPLIT_ENGINE, biasWithPaddingSpec());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
