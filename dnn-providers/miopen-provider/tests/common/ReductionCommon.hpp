// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "MiopenApi.hpp"

namespace reduction_common
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
using hipdnn_test_sdk::utilities::createReductionGraph;
using hipdnn_test_sdk::utilities::ReductionGraphSpec;
using hipdnn_test_sdk::utilities::ReductionTensorSpec;

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

// Packed channels-last strides (NHWC for rank 4, NDHWC for rank 5, ...), axis 1 innermost.
inline std::vector<int64_t> nhwcStrides(const std::vector<int64_t>& dims)
{
    return hipdnn_data_sdk::utilities::generateStrides(
        dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size()));
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
