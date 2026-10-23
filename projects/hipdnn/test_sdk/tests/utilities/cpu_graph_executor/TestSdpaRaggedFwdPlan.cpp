// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Checks the CPU ragged SDPA forward plan (RFC-0014: packed BSHD + ragged_offset). The plan only
// binds variant-pack buffers and graph attributes to CpuFpReferenceSdpaRagged, so each execute
// test compares it bit for bit with a direct reference call on the same host buffers. The direct
// call always uses element-unit offset tables, so a plan that mishandles token offsets, a
// separate V/O table or a packed LSE table addresses different tokens and the outputs differ.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/FloatOperandTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceNotApplicableError.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaRaggedFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS;
using hipdnn_data_sdk::utilities::ShallowRaggedTensor;
using hipdnn_data_sdk::utilities::ShallowTensor;
using hipdnn_data_sdk::utilities::TensorBase;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t STATS_UID = 5;
constexpr int64_t RAGGED_OFFSET_Q_UID = 10;
constexpr int64_t RAGGED_OFFSET_KV_UID = 11;
constexpr int64_t RAGGED_OFFSET_V_UID = 12;
constexpr int64_t RAGGED_OFFSET_O_UID = 13;
constexpr int64_t RAGGED_OFFSET_STATS_UID = 14;
constexpr int64_t SCALE_UID = 20;
constexpr int64_t DESCALE_Q_UID = 21;
constexpr int64_t DESCALE_K_UID = 22;
constexpr int64_t DESCALE_V_UID = 23;

// Not in the tensor map. Setting an unsupported-mode uid to it is enough to reject the node.
constexpr int64_t UNUSED_UID = 99;

constexpr float O_SENTINEL = -7.0f;
constexpr float LSE_SENTINEL = -99.0f;
constexpr float SCALE_VALUE = 0.3f;

using Bf16Builder = SdpaRaggedFwdPlanBuilder<DataType::BFLOAT16,
                                             DataType::BFLOAT16,
                                             DataType::BFLOAT16,
                                             DataType::BFLOAT16>;
using Fp8Builder = SdpaRaggedFwdPlanBuilder<DataType::FP8_E4M3,
                                            DataType::FP8_E4M3,
                                            DataType::FP8_E4M3,
                                            DataType::BFLOAT16>;
using Fp32Builder
    = SdpaRaggedFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>;

int64_t maxOf(const std::vector<int64_t>& values)
{
    return *std::max_element(values.begin(), values.end());
}

size_t volume(const std::vector<int64_t>& dims)
{
    return static_cast<size_t>(
        std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>()));
}

// Per-batch lengths and head geometry. Padded dims use the longest batch.
struct RaggedShape
{
    std::vector<int64_t> seqQ;
    std::vector<int64_t> seqKv;
    int64_t numHeads;
    int64_t numHeadsKv;
    int64_t headDimQk;
    int64_t headDimV;

    int64_t batch() const
    {
        return static_cast<int64_t>(seqQ.size());
    }
    std::vector<int64_t> qDims() const
    {
        return raggedDims(batch(), maxOf(seqQ), numHeads, headDimQk);
    }
    std::vector<int64_t> kDims() const
    {
        return raggedDims(batch(), maxOf(seqKv), numHeadsKv, headDimQk);
    }
    std::vector<int64_t> vDims() const
    {
        return raggedDims(batch(), maxOf(seqKv), numHeadsKv, headDimV);
    }
    std::vector<int64_t> oDims() const
    {
        return raggedDims(batch(), maxOf(seqQ), numHeads, headDimV);
    }
    std::vector<int64_t> lseDims() const
    {
        return raggedDims(batch(), maxOf(seqQ), numHeads, 1);
    }
};

// Unequal lengths, an empty batch, Sq != Skv per batch and GQA (H = 4, H_kv = 2).
const RaggedShape SMALL_SHAPE{{3, 0, 5}, {4, 0, 2}, 4, 2, 8, 8};

// Same structure with D_qk = 192 != D_v = 128, so Q/O and K/V token widths differ.
const RaggedShape WIDE_SHAPE{{3, 0, 2}, {2, 0, 4}, 4, 2, 192, 128};

// One batch for the applicability tests. The shape does not affect them.
const RaggedShape GATE_SHAPE{{4}, {4}, 2, 2, 8, 8};

// Bounds and alignment the direct call uses. The plan must derive the same from the graph.
struct MaskParams
{
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
};

enum class Runner
{
    PLAN, // SdpaRaggedFwdPlanBuilder::buildNodePlan + execute
    EXECUTOR, // CpuReferenceGraphExecutor::execute, which must route to the ragged plan
};

struct RunResult
{
    std::vector<float> oPlan;
    std::vector<float> oDirect;
    std::vector<float> lsePlan;
    std::vector<float> lseDirect;
};

flatbuffers::FlatBufferBuilder makeGraph(const RaggedShape& shape,
                                         DataType dataType,
                                         const RaggedSdpaFwdGraphOptions& options = {})
{
    return createRaggedSdpaFwdGraph(Q_UID,
                                    K_UID,
                                    V_UID,
                                    O_UID,
                                    RAGGED_OFFSET_Q_UID,
                                    RAGGED_OFFSET_KV_UID,
                                    shape.batch(),
                                    shape.qDims(),
                                    shape.kDims(),
                                    shape.vDims(),
                                    shape.oDims(),
                                    dataType,
                                    options);
}

flatbuffers::FlatBufferBuilder makeBf16GateGraph(const RaggedSdpaFwdGraphOptions& options = {})
{
    return makeGraph(GATE_SHAPE, DataType::BFLOAT16, options);
}

RaggedSdpaFwdGraphOptions withAttrs(const SdpaAttributesT& attrs)
{
    RaggedSdpaFwdGraphOptions options;
    options.attrs = attrs;
    return options;
}

FloatOperandSpec
    operandSpec(int64_t uid, OperandStorage storage, float value, std::vector<int64_t> dims = {1})
{
    FloatOperandSpec spec;
    spec.uid = uid;
    spec.storage = storage;
    spec.bakedValue = value;
    spec.dims = std::move(dims);
    return spec;
}

// fp8 inputs with bf16 output and Q/K/V descales in one storage mode.
RaggedSdpaFwdGraphOptions fp8Options(OperandStorage storage, const std::vector<int64_t>& dims)
{
    RaggedSdpaFwdGraphOptions options;
    options.oDataType = DataType::BFLOAT16;
    options.descaleQ = operandSpec(DESCALE_Q_UID, storage, 0.5f, dims);
    options.descaleK = operandSpec(DESCALE_K_UID, storage, 0.25f, dims);
    options.descaleV = operandSpec(DESCALE_V_UID, storage, 2.0f, dims);
    return options;
}

// Re-serializes a graph after editing its tensors.
flatbuffers::DetachedBuffer rewriteTensors(const flatbuffers::FlatBufferBuilder& graph,
                                           const std::function<void(TensorAttributesT&)>& edit)
{
    auto graphT = std::unique_ptr<GraphT>(GetGraph(graph.GetBufferPointer())->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        edit(*tensor);
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

// Drops the ragged_offset of the listed tensors, as a dense or partly ragged graph would have.
flatbuffers::DetachedBuffer withoutRaggedOffsets(const flatbuffers::FlatBufferBuilder& graph,
                                                 const std::vector<int64_t>& uids)
{
    return rewriteTensors(graph, [&uids](TensorAttributesT& tensor) {
        if(std::find(uids.begin(), uids.end(), tensor.uid) != uids.end())
        {
            tensor.ragged_offset_tensor_uid = flatbuffers::nullopt;
            tensor.ragged_offset_multiplier = 1;
        }
    });
}

template <typename Builder>
bool accepts(const void* graph, size_t size)
{
    const GraphWrapper graphWrapper(graph, size);
    return Builder().isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap());
}

template <typename Builder>
bool accepts(const flatbuffers::FlatBufferBuilder& graph)
{
    return accepts<Builder>(graph.GetBufferPointer(), graph.GetSize());
}

template <typename Builder>
bool accepts(const flatbuffers::DetachedBuffer& graph)
{
    return accepts<Builder>(graph.data(), graph.size());
}

// Deterministic values in [-1, 1). A local LCG keeps the data identical across platforms.
template <typename T>
std::vector<T> makeFilledBuffer(size_t count, uint32_t seed)
{
    std::vector<T> buffer(count);
    uint32_t state = seed;
    for(auto& x : buffer)
    {
        state = state * 1664525U + 1013904223U;
        const float u = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
        x = static_cast<T>(2.0f * u - 1.0f);
    }
    return buffer;
}

template <typename T>
std::vector<float> toFloat(const std::vector<T>& values)
{
    std::vector<float> out;
    out.reserve(values.size());
    for(const auto& value : values)
    {
        out.push_back(static_cast<float>(value));
    }
    return out;
}

// Host values of a FLOAT operand: bakedValue for a scalar, and distinct per element otherwise so
// a wrong (batch, head) lookup changes the output.
std::vector<float> operandValues(const FloatOperandSpec& spec)
{
    std::vector<float> values(volume(spec.dims));
    for(size_t i = 0; i < values.size(); ++i)
    {
        values[i] = spec.bakedValue + 0.125f * static_cast<float>(i);
    }
    return values;
}

// A baked operand lives in the graph, so it is left out of the variant pack on purpose. The other
// modes get a host pointer, which is what the CPU executor receives for them.
void bindOperand(std::unordered_map<int64_t, void*>& variantPack,
                 const FloatOperandSpec& spec,
                 std::vector<float>& values)
{
    if(spec.storage != OperandStorage::BAKED)
    {
        variantPack.emplace(spec.uid, values.data());
    }
}

// Views a packed buffer as a ragged tensor with element-unit offsets (cum * seq stride).
template <typename T>
ShallowRaggedTensor<T>
    wrapRagged(T* data, const std::vector<int64_t>& dims, const std::vector<int64_t>& cum)
{
    const auto strides = raggedStrides(dims);
    return ShallowRaggedTensor<T>(
        data, dims, strides, SDPA_SEQ_AXIS, makeRaggedOffsetAux(cum, raggedSeqStride(strides)));
}

// Builds the graph, runs it through the plan (or the executor) on host buffers, and runs the
// direct reference on the same inputs. O and LSE start at sentinels on both sides, so pad rows
// and empty batches must stay untouched by both.
template <typename Builder>
RunResult runPlanAndDirect(const RaggedShape& shape,
                           DataType inputDataType,
                           const RaggedSdpaFwdGraphOptions& options,
                           const MaskParams& mask = {},
                           Runner runner = Runner::PLAN)
{
    using QType = typename Builder::QDataType;
    using KType = typename Builder::KDataType;
    using VType = typename Builder::VDataType;
    using OType = typename Builder::ODataType;

    const auto qDims = shape.qDims();
    const auto kDims = shape.kDims();
    const auto vDims = shape.vDims();
    const auto oDims = shape.oDims();
    const auto lseDims = shape.lseDims();
    const auto cumQ = cumTokens(shape.seqQ);
    const auto cumKv = cumTokens(shape.seqKv);
    const bool hasLse = options.statsUid.has_value();
    const bool packedLse = hasLse && options.statsLayout == RaggedStatsLayout::PACKED;

    auto graph = makeGraph(shape, inputDataType, options);

    auto q = makeFilledBuffer<QType>(volume(qDims), 11);
    auto k = makeFilledBuffer<KType>(volume(kDims), 22);
    auto v = makeFilledBuffer<VType>(volume(vDims), 33);
    std::vector<OType> oPlan(volume(oDims), static_cast<OType>(O_SENTINEL));
    auto oDirect = oPlan;
    std::vector<float> lsePlan(hasLse ? volume(lseDims) : 0, LSE_SENTINEL);
    auto lseDirect = lsePlan;

    // Offset tables in the graph's units: tokens, or elements (tokens * seq stride).
    const auto graphOffsets
        = [&options](const std::vector<int64_t>& cum, const std::vector<int64_t>& dims) {
              const int64_t unit = options.tokenOffsets ? 1 : raggedSeqStride(raggedStrides(dims));
              std::vector<int32_t> table(cum.size());
              for(size_t i = 0; i < cum.size(); ++i)
              {
                  table[i] = static_cast<int32_t>(cum[i] * unit);
              }
              return table;
          };
    auto offQ = graphOffsets(cumQ, qDims);
    auto offKv = graphOffsets(cumKv, kDims);
    auto offV = graphOffsets(cumKv, vDims);
    auto offO = graphOffsets(cumQ, oDims);
    auto offLse = graphOffsets(cumQ, lseDims);

    std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.data()},
        {K_UID, k.data()},
        {V_UID, v.data()},
        {O_UID, oPlan.data()},
        {RAGGED_OFFSET_Q_UID, offQ.data()},
        {RAGGED_OFFSET_KV_UID, offKv.data()},
    };
    if(options.raggedOffsetVUid.has_value())
    {
        variantPack.emplace(options.raggedOffsetVUid.value(), offV.data());
    }
    if(options.raggedOffsetOUid.has_value())
    {
        variantPack.emplace(options.raggedOffsetOUid.value(), offO.data());
    }
    if(hasLse)
    {
        variantPack.emplace(options.statsUid.value(), lsePlan.data());
    }
    if(packedLse)
    {
        variantPack.emplace(options.raggedOffsetStatsUid.value(), offLse.data());
    }

    // An absent scale means 1.0. The direct call gets the value explicitly.
    std::optional<float> expectedScale = 1.0f;
    std::vector<float> scaleValues;
    if(options.scale.has_value())
    {
        scaleValues = operandValues(options.scale.value());
        bindOperand(variantPack, options.scale.value(), scaleValues);
        expectedScale = scaleValues.front();
    }
    else if(options.attrs.attn_scale_value.has_value())
    {
        expectedScale = options.attrs.attn_scale_value;
    }

    const std::array<const std::optional<FloatOperandSpec>*, 3> descaleSpecs
        = {&options.descaleQ, &options.descaleK, &options.descaleV};
    std::array<std::vector<float>, 3> descaleValues;
    std::array<std::optional<ShallowTensor<float>>, 3> descaleViews;
    std::array<TensorBase<float>*, 3> descales = {nullptr, nullptr, nullptr};
    for(size_t i = 0; i < descaleSpecs.size(); ++i)
    {
        const auto& spec = *descaleSpecs[i];
        if(!spec.has_value())
        {
            continue;
        }
        descaleValues[i] = operandValues(spec.value());
        bindOperand(variantPack, spec.value(), descaleValues[i]);
        descales[i] = &descaleViews[i].emplace(
            descaleValues[i].data(), spec->dims, generateStrides(spec->dims));
    }

    if(runner == Runner::PLAN)
    {
        const GraphWrapper graphWrapper(graph.GetBufferPointer(), graph.GetSize());
        const Builder builder;
        EXPECT_TRUE(builder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));
        auto plan = builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));
        plan->execute(variantPack);
    }
    else
    {
        CpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
        executor.execute(graph.GetBufferPointer(), graph.GetSize(), variantPack);
    }

    auto qView = wrapRagged(q.data(), qDims, cumQ);
    auto kView = wrapRagged(k.data(), kDims, cumKv);
    auto vView = wrapRagged(v.data(), vDims, cumKv);
    auto oView = wrapRagged(oDirect.data(), oDims, cumQ);
    std::optional<ShallowRaggedTensor<float>> packedLseView;
    std::optional<ShallowTensor<float>> denseLseView;
    TensorBase<float>* lse = nullptr;
    if(packedLse)
    {
        lse = &packedLseView.emplace(wrapRagged(lseDirect.data(), lseDims, cumQ));
    }
    else if(hasLse)
    {
        lse = &denseLseView.emplace(lseDirect.data(), lseDims, generateStrides(lseDims));
    }
    CpuFpReferenceSdpaRagged::forward<QType, KType, VType, OType, float>(qView,
                                                                         kView,
                                                                         vView,
                                                                         oView,
                                                                         expectedScale,
                                                                         mask.leftBound,
                                                                         mask.rightBound,
                                                                         mask.topLeftAlignment,
                                                                         lse,
                                                                         descales[0],
                                                                         descales[1],
                                                                         descales[2]);

    return {toFloat(oPlan), toFloat(oDirect), std::move(lsePlan), std::move(lseDirect)};
}

// Exact comparison: plan and direct call run the same reference on the same data. bf16 to float
// is exact, so equal floats mean equal bf16 bits.
void expectIdentical(const std::vector<float>& actual,
                     const std::vector<float>& expected,
                     const char* what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    const auto [actualIt, expectedIt]
        = std::mismatch(actual.begin(), actual.end(), expected.begin());
    EXPECT_TRUE(actualIt == actual.end())
        << what << " differs first at element " << (actualIt - actual.begin()) << ": " << *actualIt
        << " vs " << *expectedIt;
}

void expectPlanMatchesDirect(const RunResult& result)
{
    expectIdentical(result.oPlan, result.oDirect, "O");
    expectIdentical(result.lsePlan, result.lseDirect, "LSE");
}

// Bottom-right causal with a baked attn_scale_value. Sq > Skv in batch 0 leaves fully masked
// rows, and Sq < Skv in batch 2 shifts the diagonal.
RaggedSdpaFwdGraphOptions wideCausalOptions()
{
    RaggedSdpaFwdGraphOptions options;
    options.attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    options.attrs.left_bound = -1;
    options.attrs.right_bound = 0;
    options.attrs.attn_scale_value = 0.125f;
    return options;
}

constexpr MaskParams BOTTOM_RIGHT_CAUSAL{-1, 0, false};

RaggedSdpaFwdGraphOptions scaleOptions(OperandStorage storage)
{
    RaggedSdpaFwdGraphOptions options;
    options.scale = operandSpec(SCALE_UID, storage, SCALE_VALUE);
    return options;
}

RaggedSdpaFwdGraphOptions lseOptions(RaggedStatsLayout layout, bool tokenOffsets = false)
{
    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID;
    options.statsLayout = layout;
    if(layout == RaggedStatsLayout::PACKED)
    {
        options.raggedOffsetStatsUid = RAGGED_OFFSET_STATS_UID;
    }
    options.tokenOffsets = tokenOffsets;
    return options;
}

} // namespace

// Element offsets: V and O need their own tables because D_qk != D_v.
TEST(TestSdpaRaggedFwdPlan, ExecuteBottomRightCausalGqaWithElementOffsetsMatchesDirectCall)
{
    auto options = wideCausalOptions();
    options.raggedOffsetVUid = RAGGED_OFFSET_V_UID;
    options.raggedOffsetOUid = RAGGED_OFFSET_O_UID;
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        WIDE_SHAPE, DataType::BFLOAT16, options, BOTTOM_RIGHT_CAUSAL));
}

// AITER's form: Q/O share one token table and K/V share another. Each tensor's
// ragged_offset_multiplier (its seq stride) recovers element offsets.
TEST(TestSdpaRaggedFwdPlan, ExecuteBottomRightCausalGqaWithTokenOffsetsMatchesDirectCall)
{
    auto options = wideCausalOptions();
    options.tokenOffsets = true;
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        WIDE_SHAPE, DataType::BFLOAT16, options, BOTTOM_RIGHT_CAUSAL));
}

// Sliding window: left_bound = 1 masks keys that plain causal keeps (batch 0, sq = 2), so a plan
// that drops the left bound differs.
TEST(TestSdpaRaggedFwdPlan, ExecuteBottomRightSlidingWindowMatchesDirectCall)
{
    RaggedSdpaFwdGraphOptions options;
    options.attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
    options.attrs.left_bound = 1;
    options.attrs.right_bound = 0;
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, options, MaskParams{1, 0, false}));
}

// The deprecated causal_mask flag resolves to top-left causal (left = -1, right = 0).
TEST(TestSdpaRaggedFwdPlan, ExecuteDeprecatedCausalFlagMatchesDirectCall)
{
    SdpaAttributesT attrs;
    attrs.causal_mask = true;
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, withAttrs(attrs), MaskParams{-1, 0, true}));
}

// Dense [B, H, Sq_max, 1] LSE: batch b starts at b * H * Sq_max, not at its first packed token.
TEST(TestSdpaRaggedFwdPlan, ExecuteDenseLseMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, lseOptions(RaggedStatsLayout::DENSE)));
}

TEST(TestSdpaRaggedFwdPlan, ExecutePackedLseMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, lseOptions(RaggedStatsLayout::PACKED)));
}

// The packed LSE's token table is scaled by its own multiplier (H).
TEST(TestSdpaRaggedFwdPlan, ExecutePackedLseWithTokenOffsetsMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE,
        DataType::BFLOAT16,
        lseOptions(RaggedStatsLayout::PACKED, /*tokenOffsets=*/true)));
}

TEST(TestSdpaRaggedFwdPlan, ExecuteWithoutScaleMatchesDirectCallWithUnitScale)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(SMALL_SHAPE, DataType::BFLOAT16, {}));
}

TEST(TestSdpaRaggedFwdPlan, ExecuteBakedScaleTensorMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, scaleOptions(OperandStorage::BAKED)));
}

TEST(TestSdpaRaggedFwdPlan, ExecuteRuntimePassByValueScaleMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, scaleOptions(OperandStorage::RUNTIME_PASS_BY_VALUE)));
}

// An ordinary FLOAT [1] scale tensor: its variant-pack slot is a host buffer on the CPU.
TEST(TestSdpaRaggedFwdPlan, ExecuteOrdinaryScaleTensorMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(
        SMALL_SHAPE, DataType::BFLOAT16, scaleOptions(OperandStorage::TENSOR)));
}

TEST(TestSdpaRaggedFwdPlan, ExecuteFp8BakedDescalesMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Fp8Builder>(
        SMALL_SHAPE, DataType::FP8_E4M3, fp8Options(OperandStorage::BAKED, {1})));
}

TEST(TestSdpaRaggedFwdPlan, ExecuteFp8RuntimePassByValueDescalesMatchesDirectCall)
{
    expectPlanMatchesDirect(runPlanAndDirect<Fp8Builder>(
        SMALL_SHAPE, DataType::FP8_E4M3, fp8Options(OperandStorage::RUNTIME_PASS_BY_VALUE, {1})));
}

// Ordinary per-KV-head [B, H_kv, 1, 1] descales with a distinct value per (batch, head).
TEST(TestSdpaRaggedFwdPlan, ExecuteFp8PerKvHeadDescalesMatchesDirectCall)
{
    const std::vector<int64_t> perKvHead = {SMALL_SHAPE.batch(), SMALL_SHAPE.numHeadsKv, 1, 1};
    expectPlanMatchesDirect(runPlanAndDirect<Fp8Builder>(
        SMALL_SHAPE, DataType::FP8_E4M3, fp8Options(OperandStorage::TENSOR, perKvHead)));
}

// The executor must route a fully ragged node to the ragged plan, not the dense one.
TEST(TestSdpaRaggedFwdPlan, ExecutorRunsFullyRaggedGraph)
{
    expectPlanMatchesDirect(runPlanAndDirect<Bf16Builder>(SMALL_SHAPE,
                                                          DataType::BFLOAT16,
                                                          lseOptions(RaggedStatsLayout::PACKED),
                                                          MaskParams{},
                                                          Runner::EXECUTOR));
}

// fp8 E4M3 inputs with bf16 output are the other registered ragged tuple. The baked descales
// are absent from the variant pack, so the executor path must resolve them from the graph.
TEST(TestSdpaRaggedFwdPlan, ExecutorRunsFp8RaggedGraph)
{
    expectPlanMatchesDirect(runPlanAndDirect<Fp8Builder>(SMALL_SHAPE,
                                                         DataType::FP8_E4M3,
                                                         fp8Options(OperandStorage::BAKED, {1}),
                                                         MaskParams{},
                                                         Runner::EXECUTOR));
}

// Only Q is ragged. The node must be declined, never run as dense SDPA.
TEST(TestSdpaRaggedFwdPlan, ExecutorDeclinesPartlyRaggedGraph)
{
    auto raggedGraph = makeBf16GateGraph();
    CpuReferenceGraphExecutor executor;
    ASSERT_TRUE(executor.isApplicable(raggedGraph.GetBufferPointer(), raggedGraph.GetSize()));

    auto partlyRagged = withoutRaggedOffsets(raggedGraph, {K_UID, V_UID, O_UID});
    EXPECT_FALSE(executor.isApplicable(partlyRagged.data(), partlyRagged.size()));

    const std::unordered_map<int64_t, void*> emptyVariantPack;
    EXPECT_THROW(executor.execute(partlyRagged.data(), partlyRagged.size(), emptyVariantPack),
                 CpuReferenceNotApplicableError);
}

// The ragged plan registers bf16 and fp8 only, matching the GPU ragged plan.
TEST(TestSdpaRaggedFwdPlan, ExecutorDeclinesUnregisteredRaggedDtype)
{
    auto graph = makeGraph(GATE_SHAPE, DataType::FLOAT);
    ASSERT_TRUE(accepts<Fp32Builder>(graph));

    CpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsApplicableForBf16RaggedGraph)
{
    EXPECT_TRUE(accepts<Bf16Builder>(makeBf16GateGraph()));
}

TEST(TestSdpaRaggedFwdPlanBuilder, DenseBuilderDeclinesRaggedGraph)
{
    using DenseBf16Builder = SdpaFwdPlanBuilder<DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16>;
    EXPECT_FALSE(accepts<DenseBf16Builder>(makeBf16GateGraph()));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForDenseGraph)
{
    auto dense = withoutRaggedOffsets(makeBf16GateGraph(), {Q_UID, K_UID, V_UID, O_UID});
    EXPECT_FALSE(accepts<Bf16Builder>(dense));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForPartlyRaggedGraph)
{
    auto partlyRagged = withoutRaggedOffsets(makeBf16GateGraph(), {K_UID, V_UID, O_UID});
    EXPECT_FALSE(accepts<Bf16Builder>(partlyRagged));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForOtherDtypeBuilder)
{
    using HalfBuilder
        = SdpaRaggedFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>;
    EXPECT_FALSE(accepts<HalfBuilder>(makeBf16GateGraph()));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForUnsupportedAttributes)
{
    using Edit = void (*)(SdpaAttributesT&);
    const std::vector<std::pair<const char*, Edit>> cases = {
        {"seq_len_q", [](SdpaAttributesT& a) { a.seq_len_q_tensor_uid = UNUSED_UID; }},
        {"seq_len_kv", [](SdpaAttributesT& a) { a.seq_len_kv_tensor_uid = UNUSED_UID; }},
        {"alibi_mask", [](SdpaAttributesT& a) { a.alibi_mask = true; }},
        {"padding_mask", [](SdpaAttributesT& a) { a.padding_mask = true; }},
        {"attn_mask", [](SdpaAttributesT& a) { a.attn_mask_tensor_uid = UNUSED_UID; }},
        {"dropout_probability", [](SdpaAttributesT& a) { a.dropout_probability = 0.1f; }},
        {"seed", [](SdpaAttributesT& a) { a.seed_tensor_uid = UNUSED_UID; }},
        {"offset", [](SdpaAttributesT& a) { a.offset_tensor_uid = UNUSED_UID; }},
        {"dropout_mask", [](SdpaAttributesT& a) { a.dropout_mask_tensor_uid = UNUSED_UID; }},
        {"dropout_scale", [](SdpaAttributesT& a) { a.dropout_scale_tensor_uid = UNUSED_UID; }},
        {"rng_dump", [](SdpaAttributesT& a) { a.rng_dump_tensor_uid = UNUSED_UID; }},
        {"page_table_k", [](SdpaAttributesT& a) { a.page_table_k_tensor_uid = UNUSED_UID; }},
        {"page_table_v", [](SdpaAttributesT& a) { a.page_table_v_tensor_uid = UNUSED_UID; }},
        {"block_mask", [](SdpaAttributesT& a) { a.block_mask_tensor_uid = UNUSED_UID; }},
        {"sink_token", [](SdpaAttributesT& a) { a.sink_token_tensor_uid = UNUSED_UID; }},
        {"descale_s", [](SdpaAttributesT& a) { a.descale_s_tensor_uid = UNUSED_UID; }},
        {"scale_s", [](SdpaAttributesT& a) { a.scale_s_tensor_uid = UNUSED_UID; }},
        {"scale_o", [](SdpaAttributesT& a) { a.scale_o_tensor_uid = UNUSED_UID; }},
        {"amax_s", [](SdpaAttributesT& a) { a.amax_s_tensor_uid = UNUSED_UID; }},
        {"amax_o", [](SdpaAttributesT& a) { a.amax_o_tensor_uid = UNUSED_UID; }},
        {"max", [](SdpaAttributesT& a) { a.max_tensor_uid = UNUSED_UID; }},
        {"sum_exp", [](SdpaAttributesT& a) { a.sum_exp_tensor_uid = UNUSED_UID; }},
    };
    for(const auto& [name, edit] : cases)
    {
        SdpaAttributesT attrs;
        edit(attrs);
        EXPECT_FALSE(accepts<Bf16Builder>(makeBf16GateGraph(withAttrs(attrs)))) << name;
    }
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonInt32Offset)
{
    auto graph = rewriteTensors(makeBf16GateGraph(), [](TensorAttributesT& tensor) {
        if(tensor.uid == RAGGED_OFFSET_KV_UID)
        {
            tensor.data_type = DataType::INT64;
        }
    });
    EXPECT_FALSE(accepts<Bf16Builder>(graph));
}

// RFC-0014 requires ragged_offset_multiplier >= 1.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForZeroOffsetMultiplier)
{
    auto graph = rewriteTensors(makeBf16GateGraph(), [](TensorAttributesT& tensor) {
        if(tensor.uid == K_UID)
        {
            tensor.ragged_offset_multiplier = 0;
        }
    });
    EXPECT_FALSE(accepts<Bf16Builder>(graph));
}

// Contiguous [B, H, S, D] strides are not token-major, so tokens cannot be packed by offset.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonTokenMajorLayout)
{
    auto graph = rewriteTensors(makeBf16GateGraph(), [](TensorAttributesT& tensor) {
        if(tensor.uid == Q_UID)
        {
            tensor.strides = generateStrides(tensor.dims);
        }
    });
    EXPECT_FALSE(accepts<Bf16Builder>(graph));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonScalarHostDescale)
{
    for(const auto storage : {OperandStorage::BAKED, OperandStorage::RUNTIME_PASS_BY_VALUE})
    {
        auto options = fp8Options(OperandStorage::TENSOR, {1});
        options.descaleQ = operandSpec(DESCALE_Q_UID, storage, 0.5f, {2});
        EXPECT_FALSE(accepts<Fp8Builder>(makeGraph(GATE_SHAPE, DataType::FP8_E4M3, options)))
            << "storage " << static_cast<int>(storage);
    }
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonFloatStats)
{
    auto graph = rewriteTensors(makeBf16GateGraph(lseOptions(RaggedStatsLayout::DENSE)),
                                [](TensorAttributesT& tensor) {
                                    if(tensor.uid == STATS_UID)
                                    {
                                        tensor.data_type = DataType::HALF;
                                    }
                                });
    EXPECT_FALSE(accepts<Bf16Builder>(graph));
}

// A packed LSE needs the same INT32 aux and ragged_offset_multiplier >= 1 as the primaries.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForMalformedPackedStats)
{
    const auto packed = makeBf16GateGraph(lseOptions(RaggedStatsLayout::PACKED));
    ASSERT_TRUE(accepts<Bf16Builder>(packed));

    EXPECT_FALSE(accepts<Bf16Builder>(rewriteTensors(packed, [](TensorAttributesT& tensor) {
        if(tensor.uid == RAGGED_OFFSET_STATS_UID)
        {
            tensor.data_type = DataType::INT64;
        }
    })));
    EXPECT_FALSE(accepts<Bf16Builder>(rewriteTensors(packed, [](TensorAttributesT& tensor) {
        if(tensor.uid == STATS_UID)
        {
            tensor.ragged_offset_multiplier = 0;
        }
    })));
}

// The plan reads descales as float, so a non-FLOAT descale tensor is declined.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonFloatDescale)
{
    const auto graph
        = makeGraph(GATE_SHAPE, DataType::FP8_E4M3, fp8Options(OperandStorage::TENSOR, {1}));
    ASSERT_TRUE(accepts<Fp8Builder>(graph));

    EXPECT_FALSE(accepts<Fp8Builder>(rewriteTensors(graph, [](TensorAttributesT& tensor) {
        if(tensor.uid == DESCALE_K_UID)
        {
            tensor.data_type = DataType::HALF;
        }
    })));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForTwoElementScale)
{
    RaggedSdpaFwdGraphOptions options;
    options.scale = operandSpec(SCALE_UID, OperandStorage::TENSOR, SCALE_VALUE, {2});
    EXPECT_FALSE(accepts<Bf16Builder>(makeBf16GateGraph(options)));
}

// An ordinary scale tensor is read back as one float, so it must be FLOAT.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonFloatOrdinaryScale)
{
    auto graph = rewriteTensors(makeBf16GateGraph(scaleOptions(OperandStorage::TENSOR)),
                                [](TensorAttributesT& tensor) {
                                    if(tensor.uid == SCALE_UID)
                                    {
                                        tensor.data_type = DataType::HALF;
                                    }
                                });
    EXPECT_FALSE(accepts<Bf16Builder>(graph));
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsApplicableForEveryScaleStorage)
{
    for(const auto storage :
        {OperandStorage::TENSOR, OperandStorage::BAKED, OperandStorage::RUNTIME_PASS_BY_VALUE})
    {
        EXPECT_TRUE(accepts<Bf16Builder>(makeBf16GateGraph(scaleOptions(storage))))
            << "storage " << static_cast<int>(storage);
    }
}

TEST(TestSdpaRaggedFwdPlanBuilder, IsApplicableForDenseAndPackedStats)
{
    EXPECT_TRUE(accepts<Bf16Builder>(makeBf16GateGraph(lseOptions(RaggedStatsLayout::DENSE))));
    EXPECT_TRUE(accepts<Bf16Builder>(makeBf16GateGraph(lseOptions(RaggedStatsLayout::PACKED))));
}

TEST(TestSdpaRaggedFwdPlanBuilder, BuildNodePlanRejectsBoundBelowMinusOne)
{
    SdpaAttributesT attrs;
    attrs.left_bound = -2;
    auto graph = makeBf16GateGraph(withAttrs(attrs));
    const GraphWrapper graphWrapper(graph.GetBufferPointer(), graph.GetSize());
    const Bf16Builder builder;
    EXPECT_THROW(builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0)),
                 std::invalid_argument);
}

TEST(TestSdpaRaggedFwdPlanBuilder, BuildNodePlanRejectsBothCausalFlags)
{
    SdpaAttributesT attrs;
    attrs.causal_mask = true;
    attrs.causal_mask_bottom_right = true;
    auto graph = makeBf16GateGraph(withAttrs(attrs));
    const GraphWrapper graphWrapper(graph.GetBufferPointer(), graph.GetSize());
    const Bf16Builder builder;
    EXPECT_THROW(builder.buildNodePlan(graphWrapper, graphWrapper.getNode(0)),
                 std::invalid_argument);
}

// A packed LSE keeps its INT32 aux and multiplier 1 but gets contiguous [B, H, Sq, 1] strides.
// Its seq stride is then 1, too small to hold one token's H entries, so it is not token-major.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableForNonTokenMajorPackedStats)
{
    const auto packed = makeBf16GateGraph(lseOptions(RaggedStatsLayout::PACKED));
    ASSERT_TRUE(accepts<Bf16Builder>(packed));

    EXPECT_FALSE(accepts<Bf16Builder>(rewriteTensors(packed, [](TensorAttributesT& tensor) {
        if(tensor.uid == STATS_UID)
        {
            tensor.strides = generateStrides(tensor.dims);
        }
    })));
}

// The gate never throws on a malformed graph: every tensor it looks up, when missing from the
// map, makes the node not applicable.
TEST(TestSdpaRaggedFwdPlanBuilder, IsNotApplicableWhenALookedUpTensorIsMissingFromTensorMap)
{
    auto options = fp8Options(OperandStorage::TENSOR, {1});
    options.scale = operandSpec(SCALE_UID, OperandStorage::TENSOR, SCALE_VALUE);
    options.statsUid = STATS_UID;
    options.statsLayout = RaggedStatsLayout::PACKED;
    options.raggedOffsetStatsUid = RAGGED_OFFSET_STATS_UID;
    const auto graph = makeGraph(GATE_SHAPE, DataType::FP8_E4M3, options);
    const GraphWrapper graphWrapper(graph.GetBufferPointer(), graph.GetSize());
    const Fp8Builder builder;
    ASSERT_TRUE(builder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    for(const auto uid :
        {K_UID, RAGGED_OFFSET_Q_UID, SCALE_UID, DESCALE_Q_UID, STATS_UID, RAGGED_OFFSET_STATS_UID})
    {
        auto tensorMap = graphWrapper.getTensorMap();
        tensorMap.erase(uid);
        EXPECT_FALSE(builder.isApplicable(graphWrapper.getNode(0), tensorMap)) << "uid " << uid;
    }
}
