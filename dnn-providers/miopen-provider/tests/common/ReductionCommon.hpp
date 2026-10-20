// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include "MiopenApi.hpp"

namespace reduction_common
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;

struct ReductionModeCase
{
    ReductionMode mode;
    const char* name;
};

inline std::ostream& operator<<(std::ostream& os, const ReductionModeCase& modeCase)
{
    return os << modeCase.name;
}

// The modes MIOpen's legacy miopenReduceTensor serves.
inline const std::vector<ReductionModeCase>& getServedReductionModeCases()
{
    static const std::vector<ReductionModeCase> s_cases = {{ReductionMode::ADD, "Add"},
                                                           {ReductionMode::MUL, "Mul"},
                                                           {ReductionMode::MIN_OP, "MinOp"},
                                                           {ReductionMode::MAX_OP, "MaxOp"},
                                                           {ReductionMode::AMAX, "Amax"},
                                                           {ReductionMode::AVG, "Avg"},
                                                           {ReductionMode::NORM1, "Norm1"},
                                                           {ReductionMode::NORM2, "Norm2"}};
    return s_cases;
}

// Packed strides with the last axis innermost (NCHW for rank 4, NCDHW for rank 5, ...).
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    return hipdnn_data_sdk::utilities::generateStrides(dims);
}

// Packed channels-last strides (NHWC for rank 4, NDHWC for rank 5, ...), axis 1 innermost.
inline std::vector<int64_t> nhwcStrides(const std::vector<int64_t>& dims)
{
    return hipdnn_data_sdk::utilities::generateStrides(
        dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size()));
}

// One of the reduction node's two tensors.
struct ReductionTensorSpec
{
    // A dense, non-virtual tensor that is part of the graph.
    static ReductionTensorSpec
        dense(int64_t uid, std::vector<int64_t> dims, std::vector<int64_t> strides)
    {
        ReductionTensorSpec tensor;
        tensor.uid = uid;
        tensor.dims = std::move(dims);
        tensor.strides = std::move(strides);
        return tensor;
    }

    int64_t uid = 0;
    std::vector<int64_t> dims;
    // nullopt leaves the strides field out of the flatbuffer.
    std::optional<std::vector<int64_t>> strides;
    bool isVirtual = false;
    bool isRuntimePassByValue = false;
    std::optional<int64_t> raggedOffsetTensorUid;
    // false leaves the tensor out of the graph, so the node's reference to `uid` dangles.
    bool inGraph = true;
};

// Configuration for a single-node ReductionAttributes graph. The defaults describe a graph the
// MIOpen provider accepts: an fp32 ADD over the H and W axes of a packed NCHW tensor.
struct ReductionGraphSpec
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t Y_UID = 2;

    // `mode` over the axes where `yDims` is 1, with packed strides on both tensors.
    static ReductionGraphSpec packed(ReductionMode mode,
                                     const std::vector<int64_t>& xDims,
                                     const std::vector<int64_t>& yDims,
                                     DataType dataType = DataType::FLOAT)
    {
        ReductionGraphSpec spec;
        spec.mode = mode;
        spec.x.dims = xDims;
        spec.x.strides = packedStrides(xDims);
        spec.y.dims = yDims;
        spec.y.strides = packedStrides(yDims);
        spec.dataType = dataType;
        return spec;
    }

    // The data type of y: yDataType when set, otherwise the same as x.
    DataType outputDataType() const
    {
        return yDataType.value_or(dataType);
    }

    ReductionMode mode = ReductionMode::ADD;
    ReductionTensorSpec x = ReductionTensorSpec::dense(X_UID, {2, 3, 4, 5}, {60, 20, 5, 1});
    ReductionTensorSpec y = ReductionTensorSpec::dense(Y_UID, {2, 3, 1, 1}, {3, 1, 1, 1});
    // The data type of x, and of y unless yDataType overrides it.
    DataType dataType = DataType::FLOAT;
    std::optional<DataType> yDataType;
    DataType computeDataType = DataType::FLOAT;
    bool isDeterministic = false;
    bool overrideShapeEnabled = false;
};

// Builds a single-node ReductionAttributes graph from `spec`. Graph tensor uids are unique, so an
// in-place spec (y.uid == x.uid) emits the x tensor only, and the node references it twice.
inline flatbuffers::FlatBufferBuilder createReductionGraph(const ReductionGraphSpec& spec)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;

    const auto addTensor = [&](const ReductionTensorSpec& tensor, const char* name, DataType type) {
        if(!tensor.inGraph)
        {
            return;
        }
        tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            tensor.uid,
            name,
            type,
            tensor.strides.has_value() ? &tensor.strides.value() : nullptr,
            &tensor.dims,
            tensor.isVirtual,
            data_objects::TensorValue::NONE,
            0,
            tensor.isRuntimePassByValue,
            tensor.raggedOffsetTensorUid));
    };

    addTensor(spec.x, "x", spec.dataType);
    if(spec.y.uid != spec.x.uid)
    {
        addTensor(spec.y, "y", spec.outputDataType());
    }

    const auto reductionAttributes = data_objects::CreateReductionAttributes(
        builder, spec.mode, spec.x.uid, spec.y.uid, spec.isDeterministic);

    const std::vector<flatbuffers::Offset<data_objects::Node>> nodes{
        data_objects::CreateNodeDirect(builder,
                                       "reduction",
                                       spec.computeDataType,
                                       data_objects::NodeAttributes::ReductionAttributes,
                                       reductionAttributes.Union())};

    const auto graph = data_objects::CreateGraphDirect(builder,
                                                       "test",
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       DataType::FLOAT,
                                                       &tensorAttributes,
                                                       &nodes,
                                                       flatbuffers::nullopt,
                                                       spec.overrideShapeEnabled);
    builder.Finish(graph);
    return builder;
}

// The workspace size miopenGetReductionWorkspaceSize reports on `handle` for `spec` reduced with
// `op`. Built from raw MIOpen calls, so none of the provider code under test is involved. Any
// MIOpen failure is recorded as a test failure and yields 0.
inline size_t queryMiopenReductionWorkspaceSize(miopenHandle_t handle,
                                                miopenReduceTensorOp_t op,
                                                const ReductionGraphSpec& spec)
{
    using TensorDescriptor = std::unique_ptr<std::remove_pointer_t<miopenTensorDescriptor_t>,
                                             decltype(&miopenDestroyTensorDescriptor)>;
    using ReduceDescriptor = std::unique_ptr<std::remove_pointer_t<miopenReduceTensorDescriptor_t>,
                                             decltype(&miopenDestroyReduceTensorDescriptor)>;

    const auto toMiopenType
        = [](DataType type) { return type == DataType::HALF ? miopenHalf : miopenFloat; };

    const auto makeTensor = [](miopenDataType_t type, const ReductionTensorSpec& tensor) {
        miopenTensorDescriptor_t raw = nullptr;
        EXPECT_EQ(miopenCreateTensorDescriptor(&raw), miopenStatusSuccess);
        TensorDescriptor descriptor(raw, &miopenDestroyTensorDescriptor);
        const std::vector<int> dims(tensor.dims.begin(), tensor.dims.end());
        const std::vector<int> strides(tensor.strides->begin(), tensor.strides->end());
        EXPECT_EQ(miopenSetTensorDescriptor(
                      raw, type, static_cast<int>(dims.size()), dims.data(), strides.data()),
                  miopenStatusSuccess);
        return descriptor;
    };

    const auto x = makeTensor(toMiopenType(spec.dataType), spec.x);
    const auto y = makeTensor(toMiopenType(spec.outputDataType()), spec.y);

    miopenReduceTensorDescriptor_t rawReduce = nullptr;
    EXPECT_EQ(miopenCreateReduceTensorDescriptor(&rawReduce), miopenStatusSuccess);
    const ReduceDescriptor reduce(rawReduce, &miopenDestroyReduceTensorDescriptor);
    EXPECT_EQ(miopenSetReduceTensorDescriptor(rawReduce,
                                              op,
                                              miopenFloat,
                                              MIOPEN_PROPAGATE_NAN,
                                              MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                              MIOPEN_32BIT_INDICES),
              miopenStatusSuccess);

    size_t workspaceSize = 0;
    EXPECT_EQ(miopenGetReductionWorkspaceSize(handle, rawReduce, x.get(), y.get(), &workspaceSize),
              miopenStatusSuccess);
    return workspaceSize;
}

} // namespace reduction_common
