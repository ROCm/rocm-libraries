// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "HipdnnMiopenHandle.hpp"

namespace test_pointwise_execution_common
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// One tensor's dims/strides, as handed to hipdnn_test_sdk::detail::createTensor.
struct TensorShape
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

// A resolved binary pointwise mode's ground-truth math, independent of the provider's own
// op/scaleA/scaleB mapping inside MiopenBinaryPointwisePlan::execute -- computing the expected
// value the same way the provider maps it (scaleA/scaleB/op) would not catch a bug in that
// mapping.
inline double referenceBinaryOp(PointwiseMode mode, double a, double b)
{
    switch(mode)
    {
    case PointwiseMode::ADD:
        return a + b;
    case PointwiseMode::SUB:
        return a - b;
    case PointwiseMode::MUL:
        return a * b;
    case PointwiseMode::MAX_OP:
        return std::max(a, b);
    case PointwiseMode::MIN_OP:
        return std::min(a, b);
    default:
        throw std::invalid_argument("referenceBinaryOp: unsupported mode");
    }
}

inline std::vector<int64_t> unflattenRowMajor(int64_t linear, const std::vector<int64_t>& dims)
{
    std::vector<int64_t> idx(dims.size(), 0);
    for(size_t axis = dims.size(); axis-- > 0;)
    {
        idx[axis] = linear % dims[axis];
        linear /= dims[axis];
    }
    return idx;
}

// Broadcasts a c-space index down to b's index space: axes where b's dim is 1 read index 0.
inline std::vector<int64_t> broadcastIndex(const std::vector<int64_t>& cIndex,
                                           const std::vector<int64_t>& bDims)
{
    std::vector<int64_t> idx(cIndex.size(), 0);
    for(size_t axis = 0; axis < cIndex.size(); ++axis)
    {
        idx[axis] = (bDims[axis] == 1) ? 0 : cIndex[axis];
    }
    return idx;
}

// Reads one element via ITensor::getIndex (respects the tensor's own strides) so this works
// regardless of each operand's physical layout, then converts to double through T (float or
// hipdnn_data_sdk::types::half both convert explicitly to float).
template <typename T>
double readElement(hipdnn_data_sdk::utilities::ITensor& tensor, const std::vector<int64_t>& idx)
{
    const auto offset = tensor.getIndex(idx);
    const auto* ptr = static_cast<const T*>(tensor.hostDataOffsetFromIndex(offset));
    return static_cast<double>(static_cast<float>(*ptr));
}

// Absolute tolerance for numeric comparisons. Inputs are filled in [-1, 1] (|ref| <= 2 after a
// binary op), well inside fp32's ~1e-6 relative precision and fp16's ~1e-3 relative precision at
// this magnitude.
template <typename T>
constexpr double tolerance()
{
    return std::is_same_v<T, float> ? 1e-5 : 5e-2;
}

// Builds one input/output tensor triple for the given shapes and dtype, executes `plan`
// against them, and asserts every output element equals referenceBinaryOp(mode, a, b) within
// tolerance<T>(). uids follow test_pointwise_graph_common's convention: in_0 = 1, in_1 = 3,
// out_0 = 2.
template <typename T>
void executeAndVerify(const hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>& plan,
                      const HipdnnMiopenHandle& handle,
                      PointwiseMode mode,
                      hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                      const TensorShape& aShape,
                      const TensorShape& bShape,
                      const TensorShape& cShape,
                      unsigned int seed = 1234u)
{
    auto aTensor = hipdnn_test_sdk::detail::createTensor(dataType, aShape.dims, aShape.strides);
    auto bTensor = hipdnn_test_sdk::detail::createTensor(dataType, bShape.dims, bShape.strides);
    auto cTensor = hipdnn_test_sdk::detail::createTensor(dataType, cShape.dims, cShape.strides);

    aTensor->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
    bTensor->fillTensorWithRandomValues(-1.0f, 1.0f, seed + 1);
    // beta stays 0 in MiopenBinaryPointwisePlan::execute, so c's initial content does not
    // matter for correctness here (OverwritesOutputBuffer is the test that specifically
    // checks this); fill it anyway so an uninitialised read is never mistaken for zero.
    cTensor->fillTensorWithValue(0.0f);

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {{1, aTensor->rawDeviceData()},
                                                             {3, bTensor->rawDeviceData()},
                                                             {2, cTensor->rawDeviceData()}};

    plan.execute(
        handle, deviceBuffers.data(), static_cast<uint32_t>(deviceBuffers.size()), nullptr);

    // MigratableMemory is valid-flag based: rawDeviceData() marks device-valid without
    // clearing host-valid, and MIOpen writes the buffer out-of-band. Without this,
    // hostData() (used by readElement below) skips the D2H copy and returns stale data.
    cTensor->markDeviceModified();

    int64_t count = 1;
    for(const auto d : cShape.dims)
    {
        count *= d;
    }

    for(int64_t linear = 0; linear < count; ++linear)
    {
        const auto cIndex = unflattenRowMajor(linear, cShape.dims);
        const auto bIndex = broadcastIndex(cIndex, bShape.dims);

        const double aVal = readElement<T>(*aTensor, cIndex);
        const double bVal = readElement<T>(*bTensor, bIndex);
        const double expected = referenceBinaryOp(mode, aVal, bVal);
        const double actual = readElement<T>(*cTensor, cIndex);

        ASSERT_NEAR(expected, actual, tolerance<T>())
            << "mismatch at linear index " << linear << " (a=" << aVal << ", b=" << bVal << ")";
    }
}

} // namespace test_pointwise_execution_common
