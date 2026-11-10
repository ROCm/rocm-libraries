// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// One pack, so `operationMatcher` is empty: the graph matcher admits the operation itself.
inline constexpr PackSymbols FLYDSL_POINTWISE{"hipkernel:flydsl_pointwise",
                                              "hipkernel.flydsl_pointwise.graph_match",
                                              "",
                                              "hipkernel.flydsl_pointwise.kernel_match",
                                              "hipkernel.flydsl_pointwise.score",
                                              "hipkernel.flydsl_pointwise.dispatch",
                                              "flydsl_pointwise.in0.uid",
                                              "flydsl_pointwise.in1.uid",
                                              "flydsl_pointwise.out.uid"};

constexpr const char* FLYDSL_POINTWISE_ELEMENTS_TOKEN = "flydsl_pointwise.elements";

/// The kernel-metadata fields the pointwise KMD declares.
constexpr const char* FLYDSL_POINTWISE_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_POINTWISE_OPERATION_FIELD = "operation";
constexpr const char* FLYDSL_POINTWISE_IN1_DTYPE_FIELD = "in1_dtype";
constexpr const char* FLYDSL_POINTWISE_OUT_DTYPE_FIELD = "out_dtype";
constexpr const char* FLYDSL_POINTWISE_BLOCK_THREADS_FIELD = "block_threads";
constexpr const char* FLYDSL_POINTWISE_VEC_ELEMS_FIELD = "vec_elems";

/// Tensor uids buildFlydslPointwiseGraph() uses.
constexpr int64_t FLYDSL_POINTWISE_IN0_UID = 1;
constexpr int64_t FLYDSL_POINTWISE_IN1_UID = 2;
constexpr int64_t FLYDSL_POINTWISE_OUT_UID = 3;
constexpr int64_t FLYDSL_POINTWISE_IN2_UID = 4;

/// Row-major packed strides for @p dims.
inline std::vector<int64_t> flydslPointwisePackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// One operand of a test graph. Empty strides mean packed row-major over @p dims.
struct PointwiseOperandSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool isVirtual = false;
    bool runtimePassByValue = false;
    /// Builds the tensor with no strides vector at all.
    bool omitStrides = false;
};

/// A real device operand of @p dataType over @p dims; packed unless @p strides is given.
inline PointwiseOperandSpec flydslOperand(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                          std::vector<int64_t> dims,
                                          std::vector<int64_t> strides = {})
{
    PointwiseOperandSpec operand;
    operand.dataType = dataType;
    operand.dims = std::move(dims);
    operand.strides = std::move(strides);
    return operand;
}

/// A single-node pointwise graph. Defaults to a unary node; set `in1` for a binary one.
struct PointwiseGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode
        = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD;
    PointwiseOperandSpec in0;
    std::optional<PointwiseOperandSpec> in1;
    PointwiseOperandSpec out;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    std::optional<float> swishBeta;
    std::optional<float> softplusBeta;
    /// Names a third input (a real tensor), which no object reads.
    bool includeIn2 = false;
    /// Names in_1 without inserting a tensor for it.
    bool danglingIn1 = false;
    bool overrideShape = false;
    /// Adds a second, unrelated pointwise node.
    bool twoNodes = false;
};

/// A unary spec over one shape: in0 and out share @p dims, packed.
inline PointwiseGraphSpec
    flydslUnarySpec(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode,
                    const std::vector<int64_t>& dims,
                    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                    = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16)
{
    PointwiseGraphSpec spec;
    spec.mode = mode;
    spec.in0 = flydslOperand(dataType, dims);
    spec.out = flydslOperand(dataType, dims);
    return spec;
}

/// A binary spec: in0 over @p in0Dims, in1 over @p in1Dims, the output over @p outDims.
inline PointwiseGraphSpec
    flydslBinarySpec(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode,
                     const std::vector<int64_t>& in0Dims,
                     const std::vector<int64_t>& in1Dims,
                     const std::vector<int64_t>& outDims,
                     hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                     = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16)
{
    PointwiseGraphSpec spec;
    spec.mode = mode;
    spec.in0 = flydslOperand(dataType, in0Dims);
    spec.in1 = flydslOperand(dataType, in1Dims);
    spec.out = flydslOperand(dataType, outDims);
    return spec;
}

inline flatbuffers::FlatBufferBuilder buildFlydslPointwiseGraph(const PointwiseGraphSpec& spec)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto addTensor = [&](int64_t uid, const PointwiseOperandSpec& operand) {
        const auto strides = operand.strides.empty() ? flydslPointwisePackedStrides(operand.dims)
                                                     : operand.strides;
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       uid,
                                                       nullptr,
                                                       operand.dataType,
                                                       operand.omitStrides ? nullptr : &strides,
                                                       &operand.dims,
                                                       operand.isVirtual,
                                                       data_objects::TensorValue::NONE,
                                                       0,
                                                       operand.runtimePassByValue));
    };
    addTensor(FLYDSL_POINTWISE_IN0_UID, spec.in0);
    if(spec.in1.has_value() && !spec.danglingIn1)
    {
        addTensor(FLYDSL_POINTWISE_IN1_UID, *spec.in1);
    }
    addTensor(FLYDSL_POINTWISE_OUT_UID, spec.out);
    if(spec.includeIn2)
    {
        addTensor(FLYDSL_POINTWISE_IN2_UID, spec.in0);
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    const auto addNode = [&](int64_t in0, int64_t out) {
        data_objects::PointwiseAttributesBuilder attributes(builder);
        attributes.add_operation(spec.mode);
        attributes.add_in_0_tensor_uid(in0);
        if(spec.in1.has_value())
        {
            attributes.add_in_1_tensor_uid(FLYDSL_POINTWISE_IN1_UID);
        }
        if(spec.includeIn2)
        {
            attributes.add_in_2_tensor_uid(FLYDSL_POINTWISE_IN2_UID);
        }
        attributes.add_out_0_tensor_uid(out);
        if(spec.swishBeta.has_value())
        {
            attributes.add_swish_beta(*spec.swishBeta);
        }
        if(spec.softplusBeta.has_value())
        {
            attributes.add_softplus_beta(*spec.softplusBeta);
        }
        const auto finished = attributes.Finish();
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "pointwise",
                                           spec.computeType,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           finished.Union()));
    };
    addNode(FLYDSL_POINTWISE_IN0_UID, FLYDSL_POINTWISE_OUT_UID);
    if(spec.twoNodes)
    {
        addNode(FLYDSL_POINTWISE_OUT_UID, FLYDSL_POINTWISE_IN0_UID);
    }

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "flydsl_pointwise_test",
                                                   spec.computeType,
                                                   spec.computeType,
                                                   spec.out.dataType,
                                                   &tensors,
                                                   &nodes,
                                                   flatbuffers::nullopt,
                                                   spec.overrideShape));
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
