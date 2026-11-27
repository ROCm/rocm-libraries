// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/HipMlopsTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipMlopsDeterminism.cpp
 * @brief The evidence for the HIP MLOps ingestor engines' determinism claim (RFC 0025,
 *        ALMIOPEN-3066): each of the nine engines, on every graph below and with every kernel
 *        its kernel matcher admits for that graph on this device pinned through its knobs in
 *        turn, gives bitwise-identical outputs run after run.
 *
 * Per (graph, kernel) one plan is built and launched LAUNCHES times on the same inputs. Before
 * each launch every output and the workspace are filled with POISON_BYTE, which is a NaN in
 * every floating encoding, so a launch that skips an element, or reads a stale one, shows.
 * Every launch must write every output element with a finite value and match launch 0 byte for
 * byte. A plan built afresh from the same graph and knobs must then match launch 0 too, and a
 * launch with one input negated must not, so a check that compared outputs no launch wrote
 * would fail. Each engine's graphs together must pin every kernel it ships.
 *
 * The graphs cover fp32 and a half or bfloat16 case of every engine, both layouts for
 * batchnorm, the fused activations, batchnorm training with saved and running statistics,
 * batchnorm and layernorm backward with and without saved statistics, and RMSnorm backward in
 * bfloat16 at the reduction-heavy shapes HIP_MLOPS_ENGINE.toml skips as flaky. No HIP MLOps
 * kernel uses atomics, and every cross-workgroup reduction is a later launch of the same
 * plan, so nothing here should depend on scheduling; this suite is what holds the engines to
 * that.
 *
 * The suite runs on one device, so it proves the claim for the device it runs on; RFC 0025
 * claims determinism per (graph, configuration, device).
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using DataType = data_objects::DataType;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;

/// Launches of one plan per (graph, kernel), as the SDPA determinism suite launches.
constexpr size_t LAUNCHES = 20;

/// Every output and the workspace are filled with this byte before each launch.
constexpr unsigned char POISON_BYTE = 0xFF;

constexpr float CLIP = 0.75f;

// ---------------------------------------------------------------------------
// A case: a graph, and the tensors of its variant pack
// ---------------------------------------------------------------------------

size_t elementBytes(DataType type)
{
    switch(type)
    {
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    default:
        throw std::invalid_argument("elementBytes: a data type the HIP MLOps cases do not use");
    }
}

/// @p values as stored in a tensor of @p type, little-endian.
std::vector<std::byte> encode(const std::vector<float>& values, DataType type)
{
    std::vector<std::byte> bytes(values.size() * elementBytes(type));
    for(size_t i = 0; i < values.size(); ++i)
    {
        std::byte* at = bytes.data() + i * elementBytes(type);
        switch(type)
        {
        case DataType::FLOAT:
            std::memcpy(at, &values[i], sizeof(float));
            break;
        case DataType::HALF:
        {
            const uint16_t bits = hipdnn_data_sdk::types::half(values[i]).data;
            std::memcpy(at, &bits, sizeof(bits));
            break;
        }
        case DataType::BFLOAT16:
        {
            const uint16_t bits = hipdnn_data_sdk::types::bfloat16(values[i]).data;
            std::memcpy(at, &bits, sizeof(bits));
            break;
        }
        default:
            throw std::invalid_argument("encode: not a floating data type");
        }
    }
    return bytes;
}

/// What a tensor is to the plan: a buffer it reads, one it writes, a virtual intermediate, or
/// a scalar passed by value.
enum class Role
{
    INPUT,
    OUTPUT,
    VIRTUAL,
    SCALAR,
};

struct TensorSpec
{
    int64_t uid = 0;
    DataType type = DataType::FLOAT;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    Role role = Role::INPUT;
    /// An input's contents as stored on the device.
    std::vector<std::byte> bytes;
    /// A scalar's value.
    float scalar = 0.0f;
};

using NodeBuilder = std::function<std::vector<flatbuffers::Offset<data_objects::Node>>(
    flatbuffers::FlatBufferBuilder&)>;

/// One graph of an engine: its tensors, its nodes, and the input the sensitivity launch
/// negates.
struct DeterminismCase
{
    std::string name;
    std::vector<TensorSpec> tensors;
    NodeBuilder nodes;
    int64_t perturbedUid = 0;

    void input(int64_t uid,
               DataType type,
               const std::vector<int64_t>& dims,
               const std::vector<int64_t>& strides,
               const std::vector<float>& values)
    {
        if(values.size() != elementCount(dims))
        {
            throw std::invalid_argument("input " + std::to_string(uid)
                                        + ": the values do not fill the tensor");
        }
        add(uid, type, dims, strides, Role::INPUT).bytes = encode(values, type);
    }

    void indexInput(int64_t uid,
                    const std::vector<int64_t>& dims,
                    const std::vector<int64_t>& strides,
                    const std::vector<int32_t>& values)
    {
        if(values.size() != elementCount(dims))
        {
            throw std::invalid_argument("index input " + std::to_string(uid)
                                        + ": the values do not fill the tensor");
        }
        auto& tensor = add(uid, DataType::INT32, dims, strides, Role::INPUT);
        tensor.bytes.resize(values.size() * sizeof(int32_t));
        std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
    }

    void output(int64_t uid,
                DataType type,
                const std::vector<int64_t>& dims,
                const std::vector<int64_t>& strides)
    {
        add(uid, type, dims, strides, Role::OUTPUT);
    }

    /// A fused graph's intermediate, which every HIP MLOps family requires in fp32.
    void intermediate(int64_t uid,
                      const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& strides)
    {
        add(uid, DataType::FLOAT, dims, strides, Role::VIRTUAL);
    }

    void scalar(int64_t uid, float value)
    {
        add(uid, DataType::FLOAT, {1}, {1}, Role::SCALAR).scalar = value;
    }

private:
    TensorSpec& add(int64_t uid,
                    DataType type,
                    const std::vector<int64_t>& dims,
                    const std::vector<int64_t>& strides,
                    Role role)
    {
        tensors.emplace_back();
        auto& tensor = tensors.back();
        tensor.uid = uid;
        tensor.type = type;
        tensor.dims = dims;
        tensor.strides = strides;
        tensor.role = role;
        return tensor;
    }
};

flatbuffers::FlatBufferBuilder buildGraph(const DeterminismCase& testCase)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto& tensor : testCase.tensors)
    {
        if(tensor.role == Role::SCALAR)
        {
            const data_objects::Float32Value value(tensor.scalar);
            tensors.push_back(
                data_objects::CreateTensorAttributesDirect(builder,
                                                           tensor.uid,
                                                           nullptr,
                                                           DataType::FLOAT,
                                                           &tensor.strides,
                                                           &tensor.dims,
                                                           false,
                                                           data_objects::TensorValue::Float32Value,
                                                           builder.CreateStruct(value).Union()));
            continue;
        }
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     tensor.uid,
                                                                     nullptr,
                                                                     tensor.type,
                                                                     &tensor.strides,
                                                                     &tensor.dims,
                                                                     tensor.role == Role::VIRTUAL));
    }
    const auto nodes = testCase.nodes(builder);

    auto name = builder.CreateString("hip_mlops_determinism");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// @p values scaled by @p scale about @p centre: deterministicValues() lie in [-2, 2].
std::vector<float> around(std::vector<float> values, float centre, float scale)
{
    for(auto& value : values)
    {
        value = centre + scale * value;
    }
    return values;
}

/// [1, C, 1, ...]: one value per channel of @p dims.
std::vector<int64_t> perChannelDims(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> perChannel(dims.size(), 1);
    perChannel[1] = dims[1];
    return perChannel;
}

/// [N, 1, 1, ...]: one value per row of a normalization over every dimension after the first.
std::vector<int64_t> perRowDims(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> perRow(dims.size(), 1);
    perRow[0] = dims[0];
    return perRow;
}

const char* typeName(DataType type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return "Fp32";
    case DataType::HALF:
        return "Half";
    case DataType::BFLOAT16:
        return "Bfloat16";
    default:
        return "Other";
    }
}

/// A fused graph's activation, after the operation (forward) or before it (backward).
enum class Activation
{
    NONE,
    RELU,
    CLIPPED_RELU,
    IDENTITY,
};

flatbuffers::Offset<data_objects::Node> activationNode(flatbuffers::FlatBufferBuilder& builder,
                                                       Activation activation,
                                                       bool backward,
                                                       int64_t in0,
                                                       std::optional<int64_t> in1,
                                                       int64_t out0)
{
    auto mode
        = backward ? data_objects::PointwiseMode::RELU_BWD : data_objects::PointwiseMode::RELU_FWD;
    if(activation == Activation::IDENTITY)
    {
        mode = data_objects::PointwiseMode::IDENTITY;
    }
    data_objects::PointwiseAttributesBuilder attributes(builder);
    attributes.add_operation(mode);
    attributes.add_in_0_tensor_uid(in0);
    if(in1.has_value())
    {
        attributes.add_in_1_tensor_uid(*in1);
    }
    attributes.add_out_0_tensor_uid(out0);
    if(activation == Activation::CLIPPED_RELU)
    {
        attributes.add_relu_upper_clip(CLIP);
    }
    const auto pointwise = attributes.Finish().Union();
    return data_objects::CreateNodeDirect(builder,
                                          backward ? "activation_bwd" : "activation",
                                          DataType::FLOAT,
                                          data_objects::NodeAttributes::PointwiseAttributes,
                                          pointwise);
}

const char* activationName(Activation activation)
{
    switch(activation)
    {
    case Activation::RELU:
        return "Relu";
    case Activation::CLIPPED_RELU:
        return "ClippedRelu";
    case Activation::IDENTITY:
        return "Identity";
    case Activation::NONE:
        return "";
    default:
        return "Other";
    }
}

// ---------------------------------------------------------------------------
// Resample
// ---------------------------------------------------------------------------

namespace resample
{

constexpr int64_t SOURCE_UID = 1; // x (forward) or dy (backward)
constexpr int64_t RESULT_UID = 2; // y (forward) or dx (backward)
constexpr int64_t INDEX_UID = 3;

// NCHW [4, 8, 33, 35], 3x3 window, stride 2, padding 1 on each side: output [4, 8, 17, 18].
constexpr int64_t N = 4;
constexpr int64_t C = 8;
constexpr int64_t IN_H = 33;
constexpr int64_t IN_W = 35;
constexpr int64_t OUT_H = 17;
constexpr int64_t OUT_W = 18;
constexpr int64_t WINDOW = 3;
constexpr int64_t STRIDE = 2;
constexpr int64_t PAD = 1;

const std::vector<int64_t> INPUT_DIMS = {N, C, IN_H, IN_W};
const std::vector<int64_t> OUTPUT_DIMS = {N, C, OUT_H, OUT_W};

/// The index a maxpool forward writes for @p x: per output, the flattened (h, w) position of
/// the first maximum of its window.
std::vector<int32_t> maxpoolIndex(const std::vector<float>& x)
{
    std::vector<int32_t> index(elementCount(OUTPUT_DIMS));
    for(int64_t plane = 0; plane < N * C; ++plane)
    {
        for(int64_t oh = 0; oh < OUT_H; ++oh)
        {
            for(int64_t ow = 0; ow < OUT_W; ++ow)
            {
                float best = -std::numeric_limits<float>::max();
                int64_t at = 0;
                for(int64_t kh = 0; kh < WINDOW; ++kh)
                {
                    for(int64_t kw = 0; kw < WINDOW; ++kw)
                    {
                        const int64_t ih = oh * STRIDE + kh - PAD;
                        const int64_t iw = ow * STRIDE + kw - PAD;
                        if(ih < 0 || ih >= IN_H || iw < 0 || iw >= IN_W)
                        {
                            continue;
                        }
                        const float value = x[static_cast<size_t>((plane * IN_H + ih) * IN_W + iw)];
                        if(value > best)
                        {
                            best = value;
                            at = ih * IN_W + iw;
                        }
                    }
                }
                index[static_cast<size_t>((plane * OUT_H + oh) * OUT_W + ow)]
                    = static_cast<int32_t>(at);
            }
        }
    }
    return index;
}

const char* modeName(data_objects::ResampleMode mode)
{
    switch(mode)
    {
    case data_objects::ResampleMode::MAXPOOL:
        return "Maxpool";
    case data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING:
        return "AveragePoolExcludingPadding";
    case data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING:
        return "AveragePoolIncludingPadding";
    default:
        return "Other";
    }
}

/// A resample of @p ioType; a maxpool forward writes its index, a maxpool backward reads it.
DeterminismCase resampleCase(bool backward, data_objects::ResampleMode mode, DataType ioType)
{
    const bool maxpool = mode == data_objects::ResampleMode::MAXPOOL;
    const auto padding
        = maxpool ? data_objects::PaddingMode::NEG_INF_PAD : data_objects::PaddingMode::ZERO_PAD;
    const auto& sourceDims = backward ? OUTPUT_DIMS : INPUT_DIMS;
    const auto& resultDims = backward ? INPUT_DIMS : OUTPUT_DIMS;

    DeterminismCase testCase;
    testCase.name = std::string(typeName(ioType)) + modeName(mode);
    testCase.perturbedUid = SOURCE_UID;
    testCase.input(SOURCE_UID,
                   ioType,
                   sourceDims,
                   packedStrides(sourceDims),
                   deterministicValues(elementCount(sourceDims), 0.3f));
    testCase.output(RESULT_UID, ioType, resultDims, packedStrides(resultDims));
    if(maxpool && backward)
    {
        testCase.indexInput(INDEX_UID,
                            OUTPUT_DIMS,
                            packedStrides(OUTPUT_DIMS),
                            maxpoolIndex(deterministicValues(elementCount(INPUT_DIMS), 0.9f)));
    }
    else if(maxpool)
    {
        testCase.output(INDEX_UID, DataType::INT32, OUTPUT_DIMS, packedStrides(OUTPUT_DIMS));
    }

    testCase.nodes = [backward, mode, padding, maxpool](flatbuffers::FlatBufferBuilder& builder)
        -> std::vector<flatbuffers::Offset<data_objects::Node>> {
        const std::vector<int64_t> pads = {PAD, PAD};
        const std::vector<int64_t> strides = {STRIDE, STRIDE};
        const std::vector<int64_t> window = {WINDOW, WINDOW};
        const flatbuffers::Optional<int64_t> index
            = maxpool ? flatbuffers::Optional<int64_t>(INDEX_UID) : flatbuffers::nullopt;
        if(backward)
        {
            const auto attributes = data_objects::CreateResampleBwdAttributesDirect(builder,
                                                                                    SOURCE_UID,
                                                                                    RESULT_UID,
                                                                                    index,
                                                                                    &pads,
                                                                                    &pads,
                                                                                    &strides,
                                                                                    &window,
                                                                                    mode,
                                                                                    padding)
                                        .Union();
            return {
                data_objects::CreateNodeDirect(builder,
                                               "resample_bwd",
                                               DataType::FLOAT,
                                               data_objects::NodeAttributes::ResampleBwdAttributes,
                                               attributes)};
        }
        const auto attributes
            = data_objects::CreateResampleFwdAttributesDirect(
                  builder,
                  SOURCE_UID,
                  RESULT_UID,
                  index,
                  &pads,
                  &pads,
                  &strides,
                  &window,
                  mode,
                  padding,
                  maxpool ? flatbuffers::Optional<bool>(true) : flatbuffers::nullopt)
                  .Union();
        return {data_objects::CreateNodeDirect(builder,
                                               "resample",
                                               DataType::FLOAT,
                                               data_objects::NodeAttributes::ResampleFwdAttributes,
                                               attributes)};
    };
    return testCase;
}

std::vector<DeterminismCase> forwardCases()
{
    using data_objects::ResampleMode;
    return {resampleCase(false, ResampleMode::MAXPOOL, DataType::FLOAT),
            resampleCase(false, ResampleMode::AVGPOOL_EXCLUDE_PADDING, DataType::FLOAT),
            resampleCase(false, ResampleMode::AVGPOOL_INCLUDE_PADDING, DataType::FLOAT),
            resampleCase(false, ResampleMode::MAXPOOL, DataType::HALF),
            resampleCase(false, ResampleMode::AVGPOOL_INCLUDE_PADDING, DataType::BFLOAT16)};
}

std::vector<DeterminismCase> backwardCases()
{
    using data_objects::ResampleMode;
    return {resampleCase(true, ResampleMode::MAXPOOL, DataType::FLOAT),
            resampleCase(true, ResampleMode::AVGPOOL_EXCLUDE_PADDING, DataType::FLOAT),
            resampleCase(true, ResampleMode::AVGPOOL_INCLUDE_PADDING, DataType::FLOAT),
            resampleCase(true, ResampleMode::MAXPOOL, DataType::HALF),
            resampleCase(true, ResampleMode::AVGPOOL_EXCLUDE_PADDING, DataType::BFLOAT16)};
}

} // namespace resample

// ---------------------------------------------------------------------------
// Layernorm
// ---------------------------------------------------------------------------

namespace layernorm
{

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t DY_UID = 3;
constexpr int64_t DX_UID = 4;
constexpr int64_t SCALE_UID = 5;
constexpr int64_t BIAS_UID = 6;
constexpr int64_t DSCALE_UID = 7;
constexpr int64_t DBIAS_UID = 8;
constexpr int64_t MEAN_UID = 9;
constexpr int64_t INV_VARIANCE_UID = 10;
constexpr int64_t EPSILON_UID = 11;

constexpr float EPSILON = 1e-2f;

/// A layernorm over every dimension after the first, every tensor but epsilon of @p ioType.
/// Forward: @p withStatistics writes the mean and inverse variance. Backward: it reads them,
/// and without them the kernel recomputes them in the workspace.
DeterminismCase layernormCase(bool backward,
                              const std::vector<int64_t>& dims,
                              DataType ioType,
                              bool withStatistics,
                              const std::string& shapeName = "")
{
    const auto ioStrides = packedStrides(dims);
    const std::vector<int64_t> affineDims(dims.begin() + 1, dims.end());
    const auto affineStrides = packedStrides(affineDims);
    const auto statDims = perRowDims(dims);
    const auto statStrides = packedStrides(statDims);
    const size_t count = elementCount(dims);
    const size_t inner = elementCount(affineDims);

    DeterminismCase testCase;
    testCase.name = std::string(typeName(ioType)) + shapeName
                    + (withStatistics ? "WithStatistics" : "WithoutStatistics");
    testCase.input(X_UID, ioType, dims, ioStrides, deterministicValues(count, 0.4f));
    testCase.input(SCALE_UID,
                   ioType,
                   affineDims,
                   affineStrides,
                   around(deterministicValues(inner, 1.1f), 1.0f, 0.25f));
    if(backward)
    {
        testCase.input(DY_UID, ioType, dims, ioStrides, deterministicValues(count, 1.9f));
        testCase.output(DX_UID, ioType, dims, ioStrides);
        testCase.output(DSCALE_UID, ioType, affineDims, affineStrides);
        testCase.output(DBIAS_UID, ioType, affineDims, affineStrides);
        if(withStatistics)
        {
            const size_t rows = elementCount(statDims);
            testCase.input(MEAN_UID,
                           ioType,
                           statDims,
                           statStrides,
                           around(deterministicValues(rows, 1.3f), 0.0f, 0.1f));
            testCase.input(INV_VARIANCE_UID,
                           ioType,
                           statDims,
                           statStrides,
                           around(deterministicValues(rows, 2.1f), 1.0f, 0.2f));
        }
        testCase.perturbedUid = DY_UID;
    }
    else
    {
        testCase.output(Y_UID, ioType, dims, ioStrides);
        testCase.input(BIAS_UID,
                       ioType,
                       affineDims,
                       affineStrides,
                       around(deterministicValues(inner, 2.9f), 0.0f, 0.25f));
        if(withStatistics)
        {
            testCase.output(MEAN_UID, ioType, statDims, statStrides);
            testCase.output(INV_VARIANCE_UID, ioType, statDims, statStrides);
        }
        testCase.perturbedUid = X_UID;
    }
    testCase.scalar(EPSILON_UID, EPSILON);

    const auto normalizedDimCount = static_cast<int64_t>(dims.size() - 1);
    testCase.nodes
        = [backward, withStatistics, normalizedDimCount](flatbuffers::FlatBufferBuilder& builder)
        -> std::vector<flatbuffers::Offset<data_objects::Node>> {
        const auto statistic = [withStatistics](int64_t uid) {
            return withStatistics ? flatbuffers::Optional<int64_t>(uid) : flatbuffers::nullopt;
        };
        if(backward)
        {
            const auto attributes = data_objects::CreateLayernormBackwardAttributes(
                                        builder,
                                        DY_UID,
                                        X_UID,
                                        SCALE_UID,
                                        statistic(MEAN_UID),
                                        statistic(INV_VARIANCE_UID),
                                        flatbuffers::Optional<int64_t>(EPSILON_UID),
                                        DX_UID,
                                        DSCALE_UID,
                                        DBIAS_UID,
                                        normalizedDimCount)
                                        .Union();
            return {data_objects::CreateNodeDirect(
                builder,
                "layernorm_bwd",
                DataType::FLOAT,
                data_objects::NodeAttributes::LayernormBackwardAttributes,
                attributes)};
        }
        const auto attributes = data_objects::CreateLayernormAttributes(builder,
                                                                        X_UID,
                                                                        SCALE_UID,
                                                                        BIAS_UID,
                                                                        EPSILON_UID,
                                                                        Y_UID,
                                                                        normalizedDimCount,
                                                                        statistic(MEAN_UID),
                                                                        statistic(INV_VARIANCE_UID))
                                    .Union();
        return {data_objects::CreateNodeDirect(builder,
                                               "layernorm",
                                               DataType::FLOAT,
                                               data_objects::NodeAttributes::LayernormAttributes,
                                               attributes)};
    };
    return testCase;
}

// 8 rows of 2400, more than one workgroup's 1024 lanes each.
const std::vector<int64_t> DIMS = {8, 5, 16, 30};
// 64 rows of 1024: the parallel scale/bias reduction splits the rows across workgroups.
const std::vector<int64_t> MANY_ROWS_DIMS = {64, 4, 16, 16};

std::vector<DeterminismCase> forwardCases()
{
    return {layernormCase(false, DIMS, DataType::FLOAT, true),
            layernormCase(false, DIMS, DataType::FLOAT, false),
            layernormCase(false, DIMS, DataType::HALF, true),
            layernormCase(false, DIMS, DataType::BFLOAT16, false)};
}

/// Every graph runs with both reductions, `parallel` 0 and 1.
std::vector<DeterminismCase> backwardCases()
{
    return {layernormCase(true, DIMS, DataType::FLOAT, true),
            layernormCase(true, DIMS, DataType::FLOAT, false),
            layernormCase(true, MANY_ROWS_DIMS, DataType::FLOAT, true, "ManyRows"),
            layernormCase(true, DIMS, DataType::HALF, true),
            layernormCase(true, DIMS, DataType::BFLOAT16, false)};
}

} // namespace layernorm

// ---------------------------------------------------------------------------
// RMSnorm
// ---------------------------------------------------------------------------

namespace rmsnorm
{

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t SCALE_UID = 3;
constexpr int64_t BIAS_UID = 4;
constexpr int64_t INV_RMS_UID = 5;
constexpr int64_t EPSILON_UID = 6;
constexpr int64_t ACTIVATION_OUT_UID = 7; // fused forward: the activation's output
constexpr int64_t DY_UID = 8;
constexpr int64_t DX_UID = 9;
constexpr int64_t DSCALE_UID = 10;
constexpr int64_t DBIAS_UID = 11;
constexpr int64_t GRADIENT_UID = 12; // fused backward: the gradient the activation reads
constexpr int64_t PRE_ACTIVATION_UID = 13; // fused backward: the forward activation's input

constexpr float EPSILON = 1e-2f;

/// An RMSnorm over every dimension after the first: x, y, the affine tensors and their
/// gradients of @p ioType; the inverse RMS and a fused graph's intermediate in fp32, as the
/// family requires. Forward: @p withBias reads a bias and @p withInvRms writes the inverse
/// RMS. Backward: @p withBias writes dbias.
DeterminismCase rmsnormCase(bool backward,
                            const std::vector<int64_t>& dims,
                            DataType ioType,
                            bool withBias,
                            bool withInvRms,
                            Activation activation,
                            const std::string& shapeName = "")
{
    const bool fused = activation != Activation::NONE;
    const auto ioStrides = packedStrides(dims);
    auto affineDims = dims;
    affineDims[0] = 1;
    const auto affineStrides = packedStrides(affineDims);
    const auto statDims = perRowDims(dims);
    const auto statStrides = packedStrides(statDims);
    const size_t count = elementCount(dims);
    const size_t inner = elementCount(affineDims);

    DeterminismCase testCase;
    testCase.name = std::string(typeName(ioType)) + shapeName + activationName(activation)
                    + (withBias ? "WithBias" : "WithoutBias");
    testCase.input(X_UID, ioType, dims, ioStrides, deterministicValues(count, 0.4f));
    testCase.input(SCALE_UID,
                   ioType,
                   affineDims,
                   affineStrides,
                   around(deterministicValues(inner, 1.1f), 1.0f, 0.25f));
    if(backward)
    {
        testCase.input(INV_RMS_UID,
                       DataType::FLOAT,
                       statDims,
                       statStrides,
                       around(deterministicValues(elementCount(statDims), 2.1f), 1.0f, 0.2f));
        testCase.output(DX_UID, ioType, dims, ioStrides);
        testCase.output(DSCALE_UID, ioType, affineDims, affineStrides);
        if(withBias)
        {
            testCase.output(DBIAS_UID, ioType, affineDims, affineStrides);
        }
        if(fused)
        {
            testCase.intermediate(DY_UID, dims, ioStrides);
            testCase.input(GRADIENT_UID, ioType, dims, ioStrides, deterministicValues(count, 1.9f));
            if(activation != Activation::IDENTITY)
            {
                testCase.input(
                    PRE_ACTIVATION_UID, ioType, dims, ioStrides, deterministicValues(count, 2.6f));
            }
            testCase.perturbedUid = GRADIENT_UID;
        }
        else
        {
            testCase.input(DY_UID, ioType, dims, ioStrides, deterministicValues(count, 1.9f));
            testCase.perturbedUid = DY_UID;
        }
    }
    else
    {
        if(fused)
        {
            testCase.intermediate(Y_UID, dims, ioStrides);
            testCase.output(ACTIVATION_OUT_UID, ioType, dims, ioStrides);
        }
        else
        {
            testCase.output(Y_UID, ioType, dims, ioStrides);
        }
        if(withBias)
        {
            testCase.input(BIAS_UID,
                           ioType,
                           affineDims,
                           affineStrides,
                           around(deterministicValues(inner, 2.9f), 0.0f, 0.25f));
        }
        if(withInvRms)
        {
            testCase.output(INV_RMS_UID, DataType::FLOAT, statDims, statStrides);
        }
        testCase.scalar(EPSILON_UID, EPSILON);
        testCase.perturbedUid = X_UID;
    }

    testCase.nodes =
        [backward, withBias, withInvRms, activation, fused](flatbuffers::FlatBufferBuilder& builder)
        -> std::vector<flatbuffers::Offset<data_objects::Node>> {
        std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
        if(backward)
        {
            if(fused)
            {
                const auto in1 = activation == Activation::IDENTITY
                                     ? std::nullopt
                                     : std::optional<int64_t>(PRE_ACTIVATION_UID);
                nodes.push_back(
                    activationNode(builder, activation, true, GRADIENT_UID, in1, DY_UID));
            }
            data_objects::RMSNormBackwardAttributesBuilder attributes(builder);
            attributes.add_dy_tensor_uid(DY_UID);
            attributes.add_x_tensor_uid(X_UID);
            attributes.add_scale_tensor_uid(SCALE_UID);
            attributes.add_inv_rms_tensor_uid(INV_RMS_UID);
            attributes.add_dx_tensor_uid(DX_UID);
            attributes.add_dscale_tensor_uid(DSCALE_UID);
            if(withBias)
            {
                attributes.add_dbias_tensor_uid(DBIAS_UID);
            }
            const auto rmsnormBackward = attributes.Finish().Union();
            nodes.push_back(data_objects::CreateNodeDirect(
                builder,
                "rmsnorm_bwd",
                DataType::FLOAT,
                data_objects::NodeAttributes::RMSNormBackwardAttributes,
                rmsnormBackward));
            return nodes;
        }
        data_objects::RMSNormAttributesBuilder attributes(builder);
        attributes.add_x_tensor_uid(X_UID);
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_epsilon_tensor_uid(EPSILON_UID);
        attributes.add_y_tensor_uid(Y_UID);
        if(withBias)
        {
            attributes.add_bias_tensor_uid(BIAS_UID);
        }
        if(withInvRms)
        {
            attributes.add_inv_rms_tensor_uid(INV_RMS_UID);
        }
        const auto rmsnormForward = attributes.Finish().Union();
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "rmsnorm",
                                           DataType::FLOAT,
                                           data_objects::NodeAttributes::RMSNormAttributes,
                                           rmsnormForward));
        if(fused)
        {
            nodes.push_back(activationNode(
                builder, activation, false, Y_UID, std::nullopt, ACTIVATION_OUT_UID));
        }
        return nodes;
    };
    return testCase;
}

// 16 rows of 800, more than one workgroup's 256 lanes each.
const std::vector<int64_t> DIMS = {16, 4, 10, 20};
// HIP_MLOPS_ENGINE.toml's flaky pure-bfloat16 backward shapes: dscale and dbias each sum 4096
// rows, and each row of the second sums 50176 values.
const std::vector<int64_t> LONG_COLUMN_DIMS = {4096, 128, 1, 1};
const std::vector<int64_t> LONG_ROW_DIMS = {5, 256, 14, 14};

std::vector<DeterminismCase> forwardCases()
{
    return {rmsnormCase(false, DIMS, DataType::FLOAT, true, true, Activation::NONE),
            rmsnormCase(false, DIMS, DataType::FLOAT, false, false, Activation::NONE),
            rmsnormCase(false, DIMS, DataType::FLOAT, true, true, Activation::RELU),
            rmsnormCase(false, DIMS, DataType::FLOAT, false, true, Activation::CLIPPED_RELU),
            rmsnormCase(false, DIMS, DataType::HALF, true, true, Activation::NONE),
            rmsnormCase(false, DIMS, DataType::BFLOAT16, false, false, Activation::NONE)};
}

std::vector<DeterminismCase> backwardCases()
{
    return {
        rmsnormCase(true, DIMS, DataType::FLOAT, true, false, Activation::NONE),
        rmsnormCase(true, DIMS, DataType::FLOAT, false, false, Activation::NONE),
        rmsnormCase(true, DIMS, DataType::FLOAT, true, false, Activation::RELU),
        rmsnormCase(true, DIMS, DataType::FLOAT, true, false, Activation::CLIPPED_RELU),
        rmsnormCase(true, DIMS, DataType::FLOAT, false, false, Activation::IDENTITY),
        rmsnormCase(true, DIMS, DataType::HALF, true, false, Activation::NONE),
        rmsnormCase(true,
                    LONG_COLUMN_DIMS,
                    DataType::BFLOAT16,
                    true,
                    false,
                    Activation::NONE,
                    "LongColumns"),
        rmsnormCase(
            true, LONG_ROW_DIMS, DataType::BFLOAT16, true, false, Activation::NONE, "LongRows")};
}

} // namespace rmsnorm

// ---------------------------------------------------------------------------
// Batchnorm
// ---------------------------------------------------------------------------

namespace batchnorm
{

constexpr int64_t X_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t SCALE_UID = 3;
constexpr int64_t BIAS_UID = 4;
constexpr int64_t EPSILON_UID = 5;
constexpr int64_t MEAN_UID = 6; // inference: the mean; training: the saved mean
constexpr int64_t INV_VARIANCE_UID = 7; // inference: the inverse variance, or the variance
constexpr int64_t PREV_MEAN_UID = 8;
constexpr int64_t PREV_VARIANCE_UID = 9;
constexpr int64_t MOMENTUM_UID = 10;
constexpr int64_t NEXT_MEAN_UID = 11;
constexpr int64_t NEXT_VARIANCE_UID = 12;
constexpr int64_t ACTIVATION_OUT_UID = 13;
constexpr int64_t DY_UID = 14;
constexpr int64_t DX_UID = 15;
constexpr int64_t DSCALE_UID = 16;
constexpr int64_t DBIAS_UID = 17;
constexpr int64_t GRADIENT_UID = 18; // fused backward: the gradient the activation reads

constexpr float EPSILON = 1e-3f;
constexpr float MOMENTUM = 0.1f;

/// x, y and their gradients: NCHW, or channel-last.
struct Layout
{
    std::vector<int64_t> dims;
    bool channelLast = false;

    std::vector<int64_t> strides() const
    {
        return channelLast ? channelLastStrides(dims) : packedStrides(dims);
    }

    std::string name() const
    {
        return channelLast ? "Nhwc" : "Nchw";
    }
};

Layout nchw(std::vector<int64_t> dims)
{
    Layout layout;
    layout.dims = std::move(dims);
    return layout;
}

Layout nhwc(std::vector<int64_t> dims)
{
    Layout layout;
    layout.dims = std::move(dims);
    layout.channelLast = true;
    return layout;
}

/// Adds the per-channel fp32 input @p uid of @p layout: values about @p centre.
void perChannelInput(DeterminismCase& testCase,
                     int64_t uid,
                     const Layout& layout,
                     float phase,
                     float centre,
                     float scale)
{
    const auto dims = perChannelDims(layout.dims);
    testCase.input(uid,
                   DataType::FLOAT,
                   dims,
                   packedStrides(dims),
                   around(deterministicValues(elementCount(dims), phase), centre, scale));
}

void perChannelOutput(DeterminismCase& testCase, int64_t uid, const Layout& layout)
{
    const auto dims = perChannelDims(layout.dims);
    testCase.output(uid, DataType::FLOAT, dims, packedStrides(dims));
}

/// Batchnorm inference with the inverse variance or, @p withVariance, the variance and
/// epsilon; x and y of @p ioType, alone or followed by @p activation.
DeterminismCase
    inferenceCase(const Layout& layout, DataType ioType, bool withVariance, Activation activation)
{
    const bool fused = activation != Activation::NONE;
    const auto ioStrides = layout.strides();

    DeterminismCase testCase;
    testCase.name = std::string(typeName(ioType)) + (withVariance ? "Variance" : "InverseVariance")
                    + activationName(activation) + layout.name();
    testCase.input(X_UID,
                   ioType,
                   layout.dims,
                   ioStrides,
                   deterministicValues(elementCount(layout.dims), 0.4f));
    if(fused)
    {
        testCase.intermediate(Y_UID, layout.dims, ioStrides);
        testCase.output(ACTIVATION_OUT_UID, ioType, layout.dims, ioStrides);
    }
    else
    {
        testCase.output(Y_UID, ioType, layout.dims, ioStrides);
    }
    perChannelInput(testCase, MEAN_UID, layout, 1.3f, 0.0f, 0.125f);
    perChannelInput(testCase, INV_VARIANCE_UID, layout, 2.1f, 1.0f, 0.2f);
    perChannelInput(testCase, SCALE_UID, layout, 0.7f, 1.0f, 0.125f);
    perChannelInput(testCase, BIAS_UID, layout, 2.9f, 0.0f, 0.125f);
    if(withVariance)
    {
        testCase.scalar(EPSILON_UID, EPSILON);
    }
    testCase.perturbedUid = X_UID;

    testCase.nodes = [withVariance, activation, fused](flatbuffers::FlatBufferBuilder& builder)
        -> std::vector<flatbuffers::Offset<data_objects::Node>> {
        std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
        if(withVariance)
        {
            const auto attributes
                = data_objects::CreateBatchnormInferenceAttributesVarianceExt(builder,
                                                                              X_UID,
                                                                              MEAN_UID,
                                                                              INV_VARIANCE_UID,
                                                                              SCALE_UID,
                                                                              BIAS_UID,
                                                                              Y_UID,
                                                                              EPSILON_UID)
                      .Union();
            nodes.push_back(data_objects::CreateNodeDirect(
                builder,
                "batchnorm",
                DataType::FLOAT,
                data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt,
                attributes));
        }
        else
        {
            const auto attributes
                = data_objects::CreateBatchnormInferenceAttributes(
                      builder, X_UID, MEAN_UID, INV_VARIANCE_UID, SCALE_UID, BIAS_UID, Y_UID)
                      .Union();
            nodes.push_back(data_objects::CreateNodeDirect(
                builder,
                "batchnorm",
                DataType::FLOAT,
                data_objects::NodeAttributes::BatchnormInferenceAttributes,
                attributes));
        }
        if(fused)
        {
            nodes.push_back(activationNode(
                builder, activation, false, Y_UID, std::nullopt, ACTIVATION_OUT_UID));
        }
        return nodes;
    };
    return testCase;
}

/// Batchnorm training; x and y of @p ioType, the affine and statistic tensors fp32. It writes
/// the saved mean and inverse variance with @p withSaved, and updates the running statistics
/// with @p withRunning; alone or followed by @p activation.
DeterminismCase trainingCase(const Layout& layout,
                             DataType ioType,
                             bool withSaved,
                             bool withRunning,
                             Activation activation,
                             const std::string& shapeName = "")
{
    const bool fused = activation != Activation::NONE;
    const auto ioStrides = layout.strides();

    DeterminismCase testCase;
    testCase.name = std::string(typeName(ioType)) + shapeName + (withSaved ? "Saved" : "")
                    + (withRunning ? "Running" : "") + activationName(activation) + layout.name();
    testCase.input(X_UID,
                   ioType,
                   layout.dims,
                   ioStrides,
                   deterministicValues(elementCount(layout.dims), 0.4f));
    perChannelInput(testCase, SCALE_UID, layout, 0.7f, 1.0f, 0.125f);
    perChannelInput(testCase, BIAS_UID, layout, 2.9f, 0.0f, 0.125f);
    if(withSaved)
    {
        perChannelOutput(testCase, MEAN_UID, layout);
        perChannelOutput(testCase, INV_VARIANCE_UID, layout);
    }
    if(fused)
    {
        testCase.intermediate(Y_UID, layout.dims, ioStrides);
        testCase.output(ACTIVATION_OUT_UID, ioType, layout.dims, ioStrides);
    }
    else
    {
        testCase.output(Y_UID, ioType, layout.dims, ioStrides);
    }
    testCase.scalar(EPSILON_UID, EPSILON);
    if(withRunning)
    {
        perChannelInput(testCase, PREV_MEAN_UID, layout, 1.3f, 0.0f, 0.125f);
        perChannelInput(testCase, PREV_VARIANCE_UID, layout, 2.1f, 1.0f, 0.2f);
        perChannelOutput(testCase, NEXT_MEAN_UID, layout);
        perChannelOutput(testCase, NEXT_VARIANCE_UID, layout);
        testCase.scalar(MOMENTUM_UID, MOMENTUM);
    }
    testCase.perturbedUid = X_UID;

    testCase.nodes
        = [withSaved, withRunning, activation, fused](flatbuffers::FlatBufferBuilder& builder)
        -> std::vector<flatbuffers::Offset<data_objects::Node>> {
        data_objects::BatchnormAttributesBuilder attributes(builder);
        attributes.add_x_tensor_uid(X_UID);
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_bias_tensor_uid(BIAS_UID);
        attributes.add_epsilon_tensor_uid(EPSILON_UID);
        attributes.add_y_tensor_uid(Y_UID);
        if(withSaved)
        {
            attributes.add_mean_tensor_uid(MEAN_UID);
            attributes.add_inv_variance_tensor_uid(INV_VARIANCE_UID);
        }
        if(withRunning)
        {
            attributes.add_prev_running_mean_tensor_uid(PREV_MEAN_UID);
            attributes.add_prev_running_variance_tensor_uid(PREV_VARIANCE_UID);
            attributes.add_momentum_tensor_uid(MOMENTUM_UID);
            attributes.add_next_running_mean_tensor_uid(NEXT_MEAN_UID);
            attributes.add_next_running_variance_tensor_uid(NEXT_VARIANCE_UID);
        }
        const auto training = attributes.Finish().Union();
        std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "batchnorm",
                                           DataType::FLOAT,
                                           data_objects::NodeAttributes::BatchnormAttributes,
                                           training));
        if(fused)
        {
            nodes.push_back(activationNode(
                builder, activation, false, Y_UID, std::nullopt, ACTIVATION_OUT_UID));
        }
        return nodes;
    };
    return testCase;
}

/// Batchnorm backward; x, dy and dx of @p ioType, the affine and statistic tensors fp32. It
/// reads the saved mean and inverse variance with @p withSaved and recomputes them without.
/// @p fused is the three-node graph: the inference that produced the activation's input, a
/// ReLU's backward, then the backward (saved statistics required).
DeterminismCase backwardCase(const Layout& layout,
                             DataType ioType,
                             bool withSaved,
                             bool fused,
                             const std::string& shapeName = "")
{
    const auto ioStrides = layout.strides();
    const size_t count = elementCount(layout.dims);

    DeterminismCase testCase;
    testCase.name = std::string(typeName(ioType)) + shapeName + (fused ? "Relu" : "")
                    + (withSaved ? "Saved" : "Recomputing") + layout.name();
    testCase.input(X_UID, ioType, layout.dims, ioStrides, deterministicValues(count, 0.4f));
    perChannelInput(testCase, SCALE_UID, layout, 0.7f, 1.0f, 0.125f);
    if(withSaved)
    {
        perChannelInput(testCase, MEAN_UID, layout, 1.3f, 0.0f, 0.125f);
        perChannelInput(testCase, INV_VARIANCE_UID, layout, 2.1f, 1.0f, 0.2f);
    }
    testCase.output(DX_UID, ioType, layout.dims, ioStrides);
    perChannelOutput(testCase, DSCALE_UID, layout);
    perChannelOutput(testCase, DBIAS_UID, layout);
    if(fused)
    {
        perChannelInput(testCase, BIAS_UID, layout, 2.9f, 0.0f, 0.125f);
        testCase.intermediate(Y_UID, layout.dims, ioStrides);
        testCase.input(
            GRADIENT_UID, ioType, layout.dims, ioStrides, deterministicValues(count, 1.9f));
        testCase.intermediate(DY_UID, layout.dims, ioStrides);
        testCase.perturbedUid = GRADIENT_UID;
    }
    else
    {
        testCase.input(DY_UID, ioType, layout.dims, ioStrides, deterministicValues(count, 1.9f));
        testCase.perturbedUid = DY_UID;
    }

    testCase.nodes = [withSaved, fused](flatbuffers::FlatBufferBuilder& builder)
        -> std::vector<flatbuffers::Offset<data_objects::Node>> {
        std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
        if(fused)
        {
            const auto inference
                = data_objects::CreateBatchnormInferenceAttributes(
                      builder, X_UID, MEAN_UID, INV_VARIANCE_UID, SCALE_UID, BIAS_UID, Y_UID)
                      .Union();
            nodes.push_back(data_objects::CreateNodeDirect(
                builder,
                "batchnorm_inference",
                DataType::FLOAT,
                data_objects::NodeAttributes::BatchnormInferenceAttributes,
                inference));
            nodes.push_back(
                activationNode(builder, Activation::RELU, true, GRADIENT_UID, Y_UID, DY_UID));
        }
        data_objects::BatchnormBackwardAttributesBuilder attributes(builder);
        attributes.add_dy_tensor_uid(DY_UID);
        attributes.add_x_tensor_uid(X_UID);
        if(withSaved)
        {
            attributes.add_mean_tensor_uid(MEAN_UID);
            attributes.add_inv_variance_tensor_uid(INV_VARIANCE_UID);
        }
        attributes.add_scale_tensor_uid(SCALE_UID);
        attributes.add_dx_tensor_uid(DX_UID);
        attributes.add_dscale_tensor_uid(DSCALE_UID);
        attributes.add_dbias_tensor_uid(DBIAS_UID);
        const auto backward = attributes.Finish().Union();
        nodes.push_back(data_objects::CreateNodeDirect(
            builder,
            "batchnorm_bwd",
            DataType::FLOAT,
            data_objects::NodeAttributes::BatchnormBackwardAttributes,
            backward));
        return nodes;
    };
    return testCase;
}

// Vectorized along the spatial size (240, by 4) and along C (8, by 4): every vector width.
const std::vector<int64_t> INFERENCE_NCHW_DIMS = {4, 6, 12, 20};
const std::vector<int64_t> INFERENCE_NHWC_DIMS = {4, 8, 9, 11};
// The pack tests' sweep shapes: NCHW reaches variants 0, 1 and 3 with either workgroup and
// unvectorized variant 2, NHWC variant 1 and variant 2 with every vector size.
const std::vector<int64_t> SMALL_NCHW_DIMS = {2, 6, 5, 7};
const std::vector<int64_t> SMALL_NHWC_DIMS = {2, 8, 4, 4};
// More values per channel, so the multi-workgroup configurations split each channel.
const std::vector<int64_t> LARGE_NCHW_DIMS = {8, 4, 24, 24};
const std::vector<int64_t> LARGE_NHWC_DIMS = {4, 16, 12, 12};

std::vector<DeterminismCase> inferenceCases()
{
    const auto nchwLayout = nchw(INFERENCE_NCHW_DIMS);
    const auto nhwcLayout = nhwc(INFERENCE_NHWC_DIMS);
    return {inferenceCase(nchwLayout, DataType::FLOAT, false, Activation::NONE),
            inferenceCase(nhwcLayout, DataType::FLOAT, false, Activation::NONE),
            inferenceCase(nchwLayout, DataType::FLOAT, true, Activation::NONE),
            inferenceCase(nhwcLayout, DataType::FLOAT, true, Activation::NONE),
            inferenceCase(nchwLayout, DataType::FLOAT, false, Activation::RELU),
            inferenceCase(nhwcLayout, DataType::FLOAT, true, Activation::CLIPPED_RELU),
            inferenceCase(nchwLayout, DataType::HALF, false, Activation::NONE),
            inferenceCase(nhwcLayout, DataType::BFLOAT16, true, Activation::NONE)};
}

std::vector<DeterminismCase> trainingCases()
{
    const auto smallNchw = nchw(SMALL_NCHW_DIMS);
    const auto smallNhwc = nhwc(SMALL_NHWC_DIMS);
    return {
        trainingCase(smallNchw, DataType::FLOAT, true, true, Activation::NONE),
        trainingCase(smallNhwc, DataType::FLOAT, true, true, Activation::NONE),
        trainingCase(smallNchw, DataType::FLOAT, false, false, Activation::NONE),
        trainingCase(smallNhwc, DataType::FLOAT, false, false, Activation::NONE),
        trainingCase(smallNchw, DataType::FLOAT, true, true, Activation::RELU),
        trainingCase(smallNhwc, DataType::FLOAT, true, true, Activation::CLIPPED_RELU),
        trainingCase(smallNchw, DataType::HALF, true, true, Activation::NONE),
        trainingCase(smallNhwc, DataType::BFLOAT16, true, true, Activation::NONE),
        trainingCase(nchw(LARGE_NCHW_DIMS), DataType::FLOAT, true, true, Activation::NONE, "Large"),
        trainingCase(
            nhwc(LARGE_NHWC_DIMS), DataType::FLOAT, true, true, Activation::NONE, "Large")};
}

std::vector<DeterminismCase> backwardCases()
{
    const auto smallNchw = nchw(SMALL_NCHW_DIMS);
    const auto smallNhwc = nhwc(SMALL_NHWC_DIMS);
    return {backwardCase(smallNchw, DataType::FLOAT, true, false),
            backwardCase(smallNhwc, DataType::FLOAT, true, false),
            backwardCase(smallNchw, DataType::FLOAT, false, false),
            backwardCase(smallNhwc, DataType::FLOAT, false, false),
            backwardCase(smallNchw, DataType::FLOAT, true, true),
            backwardCase(smallNchw, DataType::HALF, true, false),
            backwardCase(smallNhwc, DataType::BFLOAT16, false, false),
            backwardCase(nchw(LARGE_NCHW_DIMS), DataType::FLOAT, true, false, "Large"),
            backwardCase(nhwc(LARGE_NHWC_DIMS), DataType::FLOAT, false, false, "Large")};
}

} // namespace batchnorm

/// Each engine and the graphs that are the evidence for its claim.
struct EngineCases
{
    const MlopsEngine* engine;
    std::vector<DeterminismCase> (*cases)();
};

constexpr std::array<EngineCases, 9> ENGINE_CASES
    = {{{&RESAMPLE_FWD_ENGINE, resample::forwardCases},
        {&RESAMPLE_BWD_ENGINE, resample::backwardCases},
        {&LAYERNORM_FWD_ENGINE, layernorm::forwardCases},
        {&LAYERNORM_BWD_ENGINE, layernorm::backwardCases},
        {&RMS_NORM_FWD_ENGINE, rmsnorm::forwardCases},
        {&RMS_NORM_BWD_ENGINE, rmsnorm::backwardCases},
        {&BATCHNORM_INFERENCE_ENGINE, batchnorm::inferenceCases},
        {&BATCHNORM_TRAINING_ENGINE, batchnorm::trainingCases},
        {&BATCHNORM_BACKWARD_ENGINE, batchnorm::backwardCases}}};

std::vector<DeterminismCase> casesOf(const MlopsEngine& engine)
{
    for(const auto& entry : ENGINE_CASES)
    {
        if(entry.engine->name == engine.name)
        {
            return entry.cases();
        }
    }
    return {};
}

/// The kernels of @p engine its kernel matcher admits for @p builder's graph on a device of
/// @p properties: the configurations a plan of the graph can pin. Empty when the graph match
/// declines.
std::vector<KernelDefinition>
    admittedKernels(const MlopsEngine& engine,
                    flatbuffers::FlatBufferBuilder builder,
                    hipdnn_plugin_sdk::ingestor::DeviceProperties properties)
{
    using namespace hipdnn_plugin_sdk::ingestor;
    registerNativeIngestorSymbols();
    const GraphFixture fixture(std::move(builder), std::move(properties));
    const auto context = fixture.context();
    const auto bound = GraphMatchRegistry::resolve(std::string(engine.graphMatcher))(context);
    if(!bound.has_value())
    {
        return {};
    }
    const auto kernelMatcher = KernelMatcherRegistry::resolve(engineSymbol(engine, "kernel_match"));
    std::vector<KernelDefinition> admitted;
    for(auto& kernel : loadedKernels(engine))
    {
        if(kernelMatcher(context, *bound, kernel))
        {
            admitted.push_back(std::move(kernel));
        }
    }
    return admitted;
}

// ---------------------------------------------------------------------------
// The repeat run
// ---------------------------------------------------------------------------

struct DeviceTensor
{
    const TensorSpec* spec = nullptr;
    size_t bytes = 0;
    std::unique_ptr<DeviceBuffer<std::byte>> buffer;
};

bool isPoison(const std::byte* element, size_t size)
{
    return std::all_of(
        element, element + size, [](std::byte value) { return value == std::byte{POISON_BYTE}; });
}

/// Whether the element at @p element is a NaN or an infinity of @p type.
bool isNotFinite(DataType type, const std::byte* element)
{
    switch(type)
    {
    case DataType::FLOAT:
    {
        uint32_t bits = 0;
        std::memcpy(&bits, element, sizeof(bits));
        return (bits & 0x7F800000U) == 0x7F800000U;
    }
    case DataType::HALF:
    {
        uint16_t bits = 0;
        std::memcpy(&bits, element, sizeof(bits));
        return (bits & 0x7C00U) == 0x7C00U;
    }
    case DataType::BFLOAT16:
    {
        uint16_t bits = 0;
        std::memcpy(&bits, element, sizeof(bits));
        return (bits & 0x7F80U) == 0x7F80U;
    }
    default:
        return false;
    }
}

/// Whether every element of output @p tensor, read back as @p bytes after launch @p launch, is
/// written and finite; adds a failure naming the first that is not.
bool writtenAndFinite(const DeviceTensor& tensor,
                      const std::vector<std::byte>& bytes,
                      size_t launch)
{
    const size_t size = elementBytes(tensor.spec->type);
    for(size_t element = 0; element * size < bytes.size(); ++element)
    {
        const std::byte* at = bytes.data() + element * size;
        if(isPoison(at, size))
        {
            ADD_FAILURE() << "output " << tensor.spec->uid << " element " << element
                          << " was left at the poison pattern in launch " << launch;
            return false;
        }
        if(isNotFinite(tensor.spec->type, at))
        {
            ADD_FAILURE() << "output " << tensor.spec->uid << " element " << element
                          << " is not finite in launch " << launch;
            return false;
        }
    }
    return true;
}

/// How many elements of @p elementSize bytes differ between @p reference and @p current.
size_t differingElements(const std::vector<std::byte>& reference,
                         const std::vector<std::byte>& current,
                         size_t elementSize)
{
    size_t differing = 0;
    for(size_t at = 0; at + elementSize <= reference.size(); at += elementSize)
    {
        if(std::memcmp(reference.data() + at, current.data() + at, elementSize) != 0)
        {
            ++differing;
        }
    }
    return differing;
}

/// The device side of one case: a buffer per non-virtual, non-scalar tensor, inputs uploaded.
class CaseBuffers
{
public:
    explicit CaseBuffers(const DeterminismCase& testCase)
    {
        for(const auto& spec : testCase.tensors)
        {
            if(spec.role != Role::INPUT && spec.role != Role::OUTPUT)
            {
                continue;
            }
            DeviceTensor tensor;
            tensor.spec = &spec;
            tensor.bytes = elementCount(spec.dims) * elementBytes(spec.type);
            tensor.buffer
                = spec.role == Role::INPUT
                      ? std::make_unique<DeviceBuffer<std::byte>>(tensor.bytes, spec.bytes)
                      : std::make_unique<DeviceBuffer<std::byte>>(tensor.bytes);
            _variantPack.push_back({spec.uid, tensor.buffer->get()});
            if(spec.role == Role::OUTPUT)
            {
                _outputs.push_back(_tensors.size());
            }
            _tensors.push_back(std::move(tensor));
        }
    }

    const std::vector<hipdnnPluginDeviceBuffer_t>& variantPack() const
    {
        return _variantPack;
    }

    size_t outputCount() const
    {
        return _outputs.size();
    }

    const DeviceTensor& output(size_t index) const
    {
        return _tensors.at(_outputs.at(index));
    }

    void poisonOutputs() const
    {
        for(const auto index : _outputs)
        {
            const auto& tensor = _tensors[index];
            ASSERT_EQ(hipSuccess, hipMemset(tensor.buffer->get(), POISON_BYTE, tensor.bytes));
        }
    }

    /// Every output's bytes, in the order of output().
    std::vector<std::vector<std::byte>> readOutputs() const
    {
        std::vector<std::vector<std::byte>> bytes;
        for(const auto index : _outputs)
        {
            bytes.push_back(_tensors[index].buffer->read());
        }
        return bytes;
    }

    /// Uploads input @p uid with every element negated: its sign bit flipped (the bytes are
    /// little-endian, so the sign is in each element's last byte).
    void negateInput(int64_t uid) const
    {
        for(const auto& tensor : _tensors)
        {
            if(tensor.spec->uid != uid)
            {
                continue;
            }
            ASSERT_TRUE(tensor.spec->role == Role::INPUT) << "uid " << uid << " is not an input";
            ASSERT_TRUE(tensor.spec->type != DataType::INT32) << "uid " << uid << " is an index";
            const size_t size = elementBytes(tensor.spec->type);
            auto negated = tensor.spec->bytes;
            for(size_t at = size - 1; at < negated.size(); at += size)
            {
                negated[at] ^= std::byte{0x80};
            }
            ASSERT_EQ(
                hipSuccess,
                hipMemcpy(
                    tensor.buffer->get(), negated.data(), negated.size(), hipMemcpyHostToDevice));
            return;
        }
        FAIL() << "no input of uid " << uid;
    }

private:
    std::vector<DeviceTensor> _tensors;
    /// The indices in _tensors of the outputs.
    std::vector<size_t> _outputs;
    std::vector<hipdnnPluginDeviceBuffer_t> _variantPack;
};

/// Poisons the outputs and the workspace, launches @p context's plan once on @p buffers, and
/// waits for it.
void launchOnce(const ::Context& context,
                const ::Handle& handle,
                const CaseBuffers& buffers,
                void* workspace,
                size_t workspaceBytes)
{
    ASSERT_NO_FATAL_FAILURE(buffers.poisonOutputs());
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipSuccess, hipMemset(workspace, POISON_BYTE, workspaceBytes));
    }
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    const auto& variantPack = buffers.variantPack();
    ASSERT_NO_THROW(context.plan().execute(
        handle, variantPack.data(), static_cast<uint32_t>(variantPack.size()), workspace));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));
}

/// Whether @p current, the outputs of launch @p launch, are written, finite and launch 0's
/// @p reference byte for byte; adds a failure for each output that is not.
bool matchesLaunchZero(const CaseBuffers& buffers,
                       const std::vector<std::vector<std::byte>>& reference,
                       const std::vector<std::vector<std::byte>>& current,
                       size_t launch)
{
    bool matches = true;
    for(size_t i = 0; i < buffers.outputCount(); ++i)
    {
        const auto& output = buffers.output(i);
        if(!writtenAndFinite(output, current.at(i), launch))
        {
            matches = false;
            continue;
        }
        const auto differing
            = differingElements(reference.at(i), current.at(i), elementBytes(output.spec->type));
        if(differing > 0)
        {
            ADD_FAILURE() << "output " << output.spec->uid << ": " << differing
                          << " elements differ from launch 0 in launch " << launch;
            matches = false;
        }
    }
    return matches;
}

/// Builds @p engine's plan for @p testCase's graph with @p knobs pinned (none: the engine's
/// own choice), launches it LAUNCHES times, then builds it afresh and launches once, and
/// requires every launch to write every output element and to give launch 0's bytes; a last
/// launch with the case's perturbed input negated must change an output.
void expectRepeatsBitwise(const MlopsEngine& engine,
                          const DeterminismCase& testCase,
                          const KnobSettings& knobs)
{
    const auto graph = buildGraph(testCase);
    const CaseBuffers buffers(testCase);

    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId(std::string(engine.name));
    hip_kernel_provider::core::Container container;
    auto& engineManager = container.getEngineManager();
    ::Handle handle;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graph.GetBufferPointer(), graph.GetSize());
    ASSERT_THAT(engineManager.getApplicableEngineIds(handle, wrapper),
                ::testing::Contains(engineId));

    hipdnn_test_sdk::utilities::MockEngineConfig unpinned;
    EXPECT_CALL(unpinned, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(unpinned, engineId()).WillRepeatedly(::testing::Return(engineId));
    const PinnedEngineConfig pinned(engineId, knobs);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig
        = knobs.empty()
              ? static_cast<const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&>(
                    unpinned)
              : pinned.get();

    ::Context context;
    ASSERT_NO_THROW(
        engineManager.initializeExecutionContext(handle, wrapper, engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());
    const size_t workspaceBytes = context.plan().getWorkspaceSize(handle);
    std::optional<DeviceBuffer<std::byte>> workspace;
    if(workspaceBytes > 0)
    {
        workspace.emplace(workspaceBytes);
    }
    void* workspacePointer = workspace.has_value() ? workspace->get() : nullptr;

    ASSERT_NO_FATAL_FAILURE(launchOnce(context, handle, buffers, workspacePointer, workspaceBytes));
    const auto reference = buffers.readOutputs();
    bool usable = true;
    for(size_t i = 0; i < buffers.outputCount(); ++i)
    {
        usable = writtenAndFinite(buffers.output(i), reference.at(i), 0) && usable;
    }
    ASSERT_TRUE(usable) << "launch 0 is not a usable reference";

    for(size_t launch = 1; launch < LAUNCHES; ++launch)
    {
        ASSERT_NO_FATAL_FAILURE(
            launchOnce(context, handle, buffers, workspacePointer, workspaceBytes));
        // One failing launch says it; the rest would repeat it.
        ASSERT_TRUE(matchesLaunchZero(buffers, reference, buffers.readOutputs(), launch));
    }

    // A plan built afresh, through the path the pack tests take.
    ASSERT_NO_FATAL_FAILURE(buffers.poisonOutputs());
    ASSERT_NO_FATAL_FAILURE(
        executeThroughTheEngine(engine, graph, buffers.variantPack(), nullptr, knobs));
    EXPECT_TRUE(matchesLaunchZero(buffers, reference, buffers.readOutputs(), LAUNCHES))
        << "a plan built afresh";

    // Different inputs must give different outputs, or the comparisons above proved nothing.
    ASSERT_NO_FATAL_FAILURE(buffers.negateInput(testCase.perturbedUid));
    ASSERT_NO_FATAL_FAILURE(launchOnce(context, handle, buffers, workspacePointer, workspaceBytes));
    const auto perturbed = buffers.readOutputs();
    size_t differing = 0;
    for(size_t i = 0; i < buffers.outputCount(); ++i)
    {
        differing += differingElements(
            reference.at(i), perturbed.at(i), elementBytes(buffers.output(i).spec->type));
    }
    EXPECT_GT(differing, 0U) << "negating input " << testCase.perturbedUid << " changed no output";
}

/// Runs every case of @p engine with each kernel its kernel matcher admits for the case's
/// graph on this device pinned through its knobs in turn (an engine without knobs ships one
/// kernel, which its unpinned plan builds), and expects the cases to pin every kernel the
/// engine ships.
void expectEveryConfigurationRepeatsBitwise(const MlopsEngine& engine)
{
    const auto knobs = engineKnobs(engine);
    const auto cases = casesOf(engine);
    ASSERT_FALSE(cases.empty());
    std::set<std::string> exercised;
    for(const auto& testCase : cases)
    {
        SCOPED_TRACE(testCase.name);
        const auto admitted
            = admittedKernels(engine, buildGraph(testCase), currentDeviceProperties());
        if(admitted.empty())
        {
            ADD_FAILURE() << "the engine admits no kernel for the graph";
            continue;
        }
        if(knobs.empty())
        {
            EXPECT_EQ(admitted.size(), 1U);
            expectRepeatsBitwise(engine, testCase, {});
            exercised.insert(admitted.front().name);
            continue;
        }
        for(const auto& kernel : admitted)
        {
            SCOPED_TRACE(kernel.name);
            expectRepeatsBitwise(engine, testCase, knobTuple(kernel, knobs));
            exercised.insert(kernel.name);
        }
    }

    std::set<std::string> shipped;
    for(const auto& kernel : loadedKernels(engine))
    {
        shipped.insert(kernel.name);
    }
    EXPECT_EQ(exercised, shipped) << "the graphs leave a kernel the engine ships unpinned";
}

// ---------------------------------------------------------------------------
// Host: the cases are graphs their engines serve
// ---------------------------------------------------------------------------

TEST(TestHipMlopsDeterminismCases, EveryCaseIsServedByItsEngineAlone)
{
    for(const auto& entry : ENGINE_CASES)
    {
        SCOPED_TRACE(std::string(entry.engine->name));
        const auto cases = entry.cases();
        EXPECT_FALSE(cases.empty());
        std::set<std::string> names;
        for(const auto& testCase : cases)
        {
            SCOPED_TRACE(testCase.name);
            EXPECT_TRUE(names.insert(testCase.name).second) << "two cases share a name";
            EXPECT_THAT(acceptingEngines(buildGraph(testCase)),
                        ::testing::ElementsAre(entry.engine->name));
        }
    }
}

// ---------------------------------------------------------------------------
// Device: bitwise repeats, engine by engine
// ---------------------------------------------------------------------------

TEST(TestHipMlopsDeterminismDevice, ResampleFwdRepeatsBitwise)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(RESAMPLE_FWD_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, ResampleBwdRepeatsBitwise)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(RESAMPLE_BWD_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, LayernormFwdRepeatsBitwise)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(LAYERNORM_FWD_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, LayernormBwdRepeatsBitwiseWithEitherReduction)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(LAYERNORM_BWD_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, RmsNormFwdRepeatsBitwise)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(RMS_NORM_FWD_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, RmsNormBwdRepeatsBitwise)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(RMS_NORM_BWD_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, BatchnormInferenceRepeatsBitwiseAtEveryVectorWidth)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(BATCHNORM_INFERENCE_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, BatchnormTrainingRepeatsBitwiseInEveryConfiguration)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(BATCHNORM_TRAINING_ENGINE);
}

TEST(TestHipMlopsDeterminismDevice, BatchnormBackwardRepeatsBitwiseInEveryConfiguration)
{
    SKIP_IF_NO_DEVICES();
    expectEveryConfigurationRepeatsBitwise(BATCHNORM_BACKWARD_ENGINE);
}

} // namespace

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
