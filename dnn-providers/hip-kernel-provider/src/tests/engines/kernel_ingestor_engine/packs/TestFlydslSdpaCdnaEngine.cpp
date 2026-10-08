// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
// The SDPA graph builder: plain hipDNN graph construction, shared with the RDNA suites.
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaCdnaEngine.cpp
 * @brief The hipkernel:flydsl_sdpa_cdna pack's seams: what its graph matcher accepts and
 *        declines, what it binds, which candidate a graph selects, and real launches of
 *        the staged gfx942/gfx950 objects checked against a host reference.
 *
 * The CDNA kernels bake the head counts, head_dim, the scale and contiguous BSHD, and
 * assume equal query and key lengths, so the accepted envelope is narrow on purpose; the
 * decline cases pin its edges. The GPU cases skip on any device this build packed no CDNA
 * object for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

// Restated rather than shared with the pack, so a rename on either side fails here.
constexpr PackSymbols FLYDSL_SDPA_CDNA{"hipkernel:flydsl_sdpa_cdna",
                                       "hipkernel.flydsl_sdpa_cdna.graph_match",
                                       "",
                                       "hipkernel.flydsl_sdpa_cdna.kernel_match",
                                       "hipkernel.flydsl_sdpa_cdna.score",
                                       "hipkernel.flydsl_sdpa_cdna.dispatch",
                                       "flydsl_sdpa_cdna.q.uid",
                                       "flydsl_sdpa_cdna.k.uid",
                                       "flydsl_sdpa_cdna.o.uid"};
/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* tryGetMetadataField(const MetadataValues& values, const char* field)
{
    const auto it = values.find(std::string(field));
    return it == values.end() ? nullptr : std::get_if<T>(&it->second);
}

constexpr const char* CDNA_CAUSAL_TOKEN = "flydsl_sdpa_cdna.causal";

constexpr const char* CDNA_DTYPE_FIELD = "dtype";
constexpr const char* CDNA_KERNEL_FIELD = "kernel";
constexpr const char* CDNA_NUM_HEADS_FIELD = "num_heads";
constexpr const char* CDNA_NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr const char* CDNA_HEAD_DIM_FIELD = "head_dim";
constexpr const char* CDNA_CAUSAL_FIELD = "causal";
constexpr const char* CDNA_BLOCK_M_FIELD = "block_m";
constexpr const char* CDNA_KV_ALIGN_FIELD = "kv_align";

/// The model shape the first CDNA objects are built for.
SdpaGraphSpec cdnaSpec()
{
    SdpaGraphSpec spec;
    spec.batch = 1;
    spec.heads = 32;
    spec.kvHeads = 8;
    spec.seqLenQ = 256;
    spec.seqLenKv = 256;
    spec.headDim = 128;
    spec.scaleKind = SdpaScale::ABSENT;
    return spec;
}

/// A kernel carrying exactly the metadata this pack's KMD declares.
KernelDefinition makeCdnaKernel(const std::string& family = "dualwave",
                                int64_t heads = 32,
                                int64_t kvHeads = 8,
                                int64_t headDim = 128,
                                int64_t causal = 0,
                                const std::string& dtype = "bf16",
                                int64_t blockM = 256,
                                int64_t kvAlign = 0)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_cdna_" + dtype + "_" + family;
    kernel.metadata = {{CDNA_DTYPE_FIELD, dtype},
                       {CDNA_KERNEL_FIELD, family},
                       {CDNA_NUM_HEADS_FIELD, heads},
                       {CDNA_NUM_KV_HEADS_FIELD, kvHeads},
                       {CDNA_HEAD_DIM_FIELD, headDim},
                       {CDNA_CAUSAL_FIELD, causal},
                       {CDNA_BLOCK_M_FIELD, blockM},
                       {CDNA_KV_ALIGN_FIELD, kvAlign}};
    kernel.priority = 100;
    return kernel;
}

std::optional<BoundTokens> cdnaGraphMatch(const SdpaGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    return matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
}

// ---------------------------------------------------------------------------
// graph_match
// ---------------------------------------------------------------------------

struct CdnaCase
{
    std::string name;
    SdpaGraphSpec spec;
};

class TestFlydslSdpaCdnaGraphAccepts : public ::testing::TestWithParam<CdnaCase>
{
};

TEST_P(TestFlydslSdpaCdnaGraphAccepts, BindsTheGraph)
{
    EXPECT_TRUE(cdnaGraphMatch(GetParam().spec).has_value());
}

std::vector<CdnaCase> acceptedCases()
{
    std::vector<CdnaCase> cases;
    cases.push_back({"NonCausal", cdnaSpec()});
    auto topLeft = cdnaSpec();
    topLeft.mask = SdpaMask::TOP_LEFT_CAUSAL;
    cases.push_back({"TopLeftCausal", topLeft});
    auto bottomRight = cdnaSpec();
    bottomRight.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    cases.push_back({"BottomRightCausal", bottomRight});
    auto scaleValue = cdnaSpec();
    scaleValue.scaleKind = SdpaScale::VALUE;
    scaleValue.scale = 1.0F / std::sqrt(128.0F);
    cases.push_back({"ScaleEqualToTheBakedOne", scaleValue});
    auto batched = cdnaSpec();
    batched.batch = 3;
    batched.seqLenQ = 333;
    batched.seqLenKv = 333;
    cases.push_back({"BatchedWithARaggedTail", batched});
    auto mha = cdnaSpec();
    mha.kvHeads = 32;
    cases.push_back({"Mha", mha});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaGraphAccepts,
                         ::testing::ValuesIn(acceptedCases()),
                         [](const auto& info) { return info.param.name; });

class TestFlydslSdpaCdnaGraphDeclines : public ::testing::TestWithParam<CdnaCase>
{
};

TEST_P(TestFlydslSdpaCdnaGraphDeclines, DeclinesTheGraph)
{
    EXPECT_FALSE(cdnaGraphMatch(GetParam().spec).has_value());
}

std::vector<CdnaCase> declinedCases()
{
    std::vector<CdnaCase> cases;
    auto bhsd = cdnaSpec();
    bhsd.layout = SdpaLayout::BHSD;
    cases.push_back({"BhsdLayout", bhsd});
    auto cross = cdnaSpec();
    cross.seqLenKv = 512;
    cases.push_back({"UnequalSequenceLengths", cross});
    auto window = cdnaSpec();
    window.mask = SdpaMask::SLIDING_WINDOW;
    cases.push_back({"SlidingWindow", window});
    auto band = cdnaSpec();
    band.mask = SdpaMask::TOP_LEFT_CAUSAL;
    band.rightBound = 16;
    cases.push_back({"RightBoundBand", band});
    auto bias = cdnaSpec();
    bias.withBias = true;
    cases.push_back({"Bias", bias});
    auto stats = cdnaSpec();
    stats.stats = SdpaStats::LSE;
    cases.push_back({"SoftmaxStats", stats});
    auto scaleTensor = cdnaSpec();
    scaleTensor.scaleKind = SdpaScale::RUNTIME_TENSOR;
    cases.push_back({"ScaleTensor", scaleTensor});
    auto otherScale = cdnaSpec();
    otherScale.scaleKind = SdpaScale::VALUE;
    otherScale.scale = 0.125F;
    cases.push_back({"ScaleOtherThanTheBakedOne", otherScale});
    auto fp32 = cdnaSpec();
    fp32.dataType = DataType::FLOAT;
    cases.push_back({"Fp32", fp32});
    auto dropout = cdnaSpec();
    dropout.withDropout = true;
    cases.push_back({"Dropout", dropout});
    auto ragged = cdnaSpec();
    ragged.raggedQuery = true;
    cases.push_back({"RaggedQuery", ragged});
    auto packed = cdnaSpec();
    packed.kvHeads = 32;
    packed.packedQkv = true;
    cases.push_back({"PackedQkv", packed});
    auto group = cdnaSpec();
    group.heads = 30;
    group.kvHeads = 8;
    cases.push_back({"HeadsNotAMultipleOfKvHeads", group});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaCdnaGraphDeclines,
                         ::testing::ValuesIn(declinedCases()),
                         [](const auto& info) { return info.param.name; });

TEST(TestFlydslSdpaCdnaBinding, BindsTheCausalFlag)
{
    auto causal = cdnaSpec();
    causal.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const auto causalBound = cdnaGraphMatch(causal);
    const auto plainBound = cdnaGraphMatch(cdnaSpec());
    ASSERT_TRUE(causalBound.has_value());
    ASSERT_TRUE(plainBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*causalBound, CDNA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(tryGetBoundInt(*plainBound, CDNA_CAUSAL_TOKEN), 0);
}

// ---------------------------------------------------------------------------
// kernel_match
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaCdnaKernelMatch, AdmitsTheCandidateBuiltForTheGraphsShape)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), makeCdnaKernel(), *bound));
    EXPECT_TRUE(matchesKernel(FLYDSL_SDPA_CDNA,
                              fixture.context(),
                              makeCdnaKernel("generic", 32, 8, 128, 0, "bf16", 128),
                              *bound));
}

TEST(TestFlydslSdpaCdnaKernelMatch, DeclinesEachBakedFieldThatDisagrees)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(cdnaSpec()));
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto declines = [&](const KernelDefinition& kernel) {
        return !matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 16, 8)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 32, 4)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 32, 8, 64)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 32, 8, 128, 1)));
    EXPECT_TRUE(declines(makeCdnaKernel("dualwave", 32, 8, 128, 0, "f16")));
    EXPECT_TRUE(declines(makeCdnaKernel("wmma")));
}

TEST(TestFlydslSdpaCdnaKernelMatch, HoldsTheKeyLengthToTheCandidatesAlignment)
{
    const auto admits = [](int64_t seqLen, const KernelDefinition& kernel) {
        auto spec = cdnaSpec();
        spec.seqLenQ = seqLen;
        spec.seqLenKv = seqLen;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
        return bound.has_value()
               && matchesKernel(FLYDSL_SDPA_CDNA, fixture.context(), kernel, *bound);
    };
    const auto aligned = makeCdnaKernel("generic", 32, 8, 128, 0, "bf16", 128, 64);
    const auto anyLength = makeCdnaKernel("generic", 32, 8, 128, 0, "bf16", 128, 0);
    EXPECT_TRUE(admits(256, aligned));
    EXPECT_FALSE(admits(333, aligned)) << "the object skips the KV tail mask";
    EXPECT_TRUE(admits(256, anyLength));
    EXPECT_TRUE(admits(333, anyLength));

    // An object that does not say which it is may be either, so it is not run.
    auto unstated = anyLength;
    unstated.metadata.erase(std::string(CDNA_KV_ALIGN_FIELD));
    EXPECT_FALSE(admits(256, unstated));
}

// ---------------------------------------------------------------------------
// GPU: the staged objects against a host reference
// ---------------------------------------------------------------------------

DeviceProperties packedArchDeviceProperties()
{
    auto properties = currentDeviceProperties();
    const auto colon = properties.gcnArchName.find(':');
    if(colon != std::string::npos)
    {
        properties.gcnArchName.resize(colon);
    }
    return properties;
}

/// Every staged object for this device's arch that the kernel matcher admits for the graph.
/// Each must compute the same answer, so the GPU cases run them all: at an aligned length
/// that is both a masked object and one that skips the KV tail mask.
std::vector<KernelDefinition> stagedCdnaCandidates(const MatchContext& context,
                                                   const BoundTokens& bound)
{
    std::vector<KernelDefinition> candidates;
    // A build for RDNA targets only stages no CDNA set at all.
    const auto* staged = findLoadedSet(FLYDSL_SDPA_CDNA.engineName);
    if(staged == nullptr)
    {
        return candidates;
    }
    const auto device = packedArchDeviceProperties().gcnArchName;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            KernelDefinition definition{kernel.id,
                                        pack.id,
                                        pack.dispatchId,
                                        kernel.source,
                                        kernel.metadata,
                                        kernel.priority,
                                        arches,
                                        kernel.originDirectory,
                                        kernel.name,
                                        kernel.treeRoot};
            if(matchesKernel(FLYDSL_SDPA_CDNA, context, definition, bound))
            {
                candidates.push_back(std::move(definition));
            }
        }
    }
    return candidates;
}

using Bf16 = hipdnn_data_sdk::types::bfloat16;

/// Deterministic k/32 values for |k| <= 32: exact in bf16, so the reference reads exactly
/// what the device does.
std::vector<Bf16> fillValues(size_t count, uint32_t seed)
{
    std::vector<Bf16> values(count);
    uint32_t state = seed;
    for(auto& value : values)
    {
        state = state * 1664525U + 1013904223U;
        const auto k = static_cast<int32_t>((state >> 16) % 65U) - 32;
        value = Bf16(static_cast<float>(k) / 32.0F);
    }
    return values;
}

/// Runs @p kernel for @p spec through prepare() and launch() and checks every output element
/// against a double-precision reference. BSHD memory throughout.
void expectCandidateMatchesReference(const MatchContext& context,
                                     const BoundTokens& bound,
                                     const KernelDefinition& kernel,
                                     const SdpaGraphSpec& spec,
                                     double absoluteTolerance)
{
    const bool causal = spec.mask != SdpaMask::NONE;
    const auto& handler = dispatchHandler(FLYDSL_SDPA_CDNA);
    const auto prepared = handler.prepare(context, bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const int64_t b = spec.batch;
    const int64_t s = spec.seqLenQ;
    const int64_t h = spec.heads;
    const int64_t hk = spec.kvHeads;
    const int64_t d = spec.headDim;
    const auto qCount = static_cast<size_t>(b * s * h * d);
    const auto kvCount = static_cast<size_t>(b * s * hk * d);
    const auto q = fillValues(qCount, 1U);
    const auto k = fillValues(kvCount, 2U);
    const auto v = fillValues(kvCount, 3U);
    std::vector<Bf16> o(qCount, Bf16(0.0F));

    void* qDevice = nullptr;
    void* kDevice = nullptr;
    void* vDevice = nullptr;
    void* oDevice = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&qDevice, qCount * sizeof(Bf16)));
    ASSERT_EQ(hipSuccess, hipMalloc(&kDevice, kvCount * sizeof(Bf16)));
    ASSERT_EQ(hipSuccess, hipMalloc(&vDevice, kvCount * sizeof(Bf16)));
    ASSERT_EQ(hipSuccess, hipMalloc(&oDevice, qCount * sizeof(Bf16)));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(qDevice, q.data(), qCount * sizeof(Bf16), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(kDevice, k.data(), kvCount * sizeof(Bf16), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(vDevice, v.data(), kvCount * sizeof(Bf16), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemset(oDevice, 0, qCount * sizeof(Bf16)));

    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{FLYDSL_SDPA_RDNA_Q_UID, qDevice},
                                                    {FLYDSL_SDPA_RDNA_K_UID, kDevice},
                                                    {FLYDSL_SDPA_RDNA_V_UID, vDevice},
                                                    {FLYDSL_SDPA_RDNA_O_UID, oDevice}};
    const Handle handle;
    handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_EQ(hipSuccess,
              hipMemcpy(o.data(), oDevice, qCount * sizeof(Bf16), hipMemcpyDeviceToHost));
    static_cast<void>(hipFree(qDevice));
    static_cast<void>(hipFree(kDevice));
    static_cast<void>(hipFree(vDevice));
    static_cast<void>(hipFree(oDevice));

    // BSHD element (batch, token, head, d).
    const auto at = [](const std::vector<Bf16>& t,
                       int64_t heads,
                       int64_t seq,
                       int64_t dim,
                       int64_t bi,
                       int64_t hi,
                       int64_t si,
                       int64_t di) {
        return static_cast<double>(
            static_cast<float>(t[static_cast<size_t>(((bi * seq + si) * heads + hi) * dim + di)]));
    };
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    const int64_t group = h / hk;
    double worst = 0.0;
    std::string where;
    std::vector<double> scores(static_cast<size_t>(s));
    for(int64_t bi = 0; bi < b; ++bi)
    {
        for(int64_t hi = 0; hi < h; ++hi)
        {
            for(int64_t si = 0; si < s; ++si)
            {
                const int64_t lastKey = causal ? si : s - 1;
                double maxScore = -std::numeric_limits<double>::infinity();
                for(int64_t j = 0; j <= lastKey; ++j)
                {
                    double dot = 0.0;
                    for(int64_t e = 0; e < d; ++e)
                    {
                        dot += at(q, h, s, d, bi, hi, si, e)
                               * at(k, hk, s, d, bi, hi / group, j, e);
                    }
                    scores[static_cast<size_t>(j)] = dot * scale;
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
                }
                double sum = 0.0;
                for(int64_t j = 0; j <= lastKey; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                for(int64_t e = 0; e < d; ++e)
                {
                    double expected = 0.0;
                    for(int64_t j = 0; j <= lastKey; ++j)
                    {
                        expected += scores[static_cast<size_t>(j)]
                                    * at(v, hk, s, d, bi, hi / group, j, e);
                    }
                    expected /= sum;
                    const double error = std::abs(at(o, h, s, d, bi, hi, si, e) - expected);
                    if(error > worst)
                    {
                        worst = error;
                        where = "b=" + std::to_string(bi) + " h=" + std::to_string(hi)
                                + " s=" + std::to_string(si) + " d=" + std::to_string(e);
                    }
                }
            }
        }
    }
    EXPECT_LE(worst, absoluteTolerance) << "worst error at " << where;
}

/// Every staged candidate the matcher admits for @p spec must match the reference.
void expectCdnaMatchesReference(const SdpaGraphSpec& spec, double absoluteTolerance)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA_CDNA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto candidates = stagedCdnaCandidates(fixture.context(), *bound);
    if(candidates.empty())
    {
        GTEST_SKIP() << "no CDNA object packed for this device ("
                     << packedArchDeviceProperties().gcnArchName << ") and shape";
    }
    for(const auto& kernel : candidates)
    {
        SCOPED_TRACE(kernel.name);
        expectCandidateMatchesReference(fixture.context(), *bound, kernel, spec, absoluteTolerance);
    }
}

/// bf16 output rounding is ~2^-8 relative on values of magnitude <= 1.
constexpr double BF16_TOLERANCE = 2e-2;

TEST(TestGpuFlydslSdpaCdnaDispatch, NonCausalMatchesTheReference)
{
    expectCdnaMatchesReference(cdnaSpec(), BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, CausalMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, BatchedRaggedTailMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 333;
    spec.seqLenKv = 333;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, NonCausalRaggedTailMatchesTheReference)
{
    // A key length that is not a multiple of the KV tile: only an object that masks the
    // last tile may run it.
    auto spec = cdnaSpec();
    spec.batch = 2;
    spec.seqLenQ = 333;
    spec.seqLenKv = 333;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaCdnaDispatch, LongerSequenceMatchesTheReference)
{
    auto spec = cdnaSpec();
    spec.seqLenQ = 1024;
    spec.seqLenKv = 1024;
    expectCdnaMatchesReference(spec, BF16_TOLERANCE);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
