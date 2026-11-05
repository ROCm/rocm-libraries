// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "ShiftedStarts.hpp"

namespace
{

using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::Tensor;
using sdpa_case_flow::ShiftedStarts;
using TensorMap = std::unordered_map<int64_t, std::shared_ptr<ITensor>>;

constexpr int64_t INPUT_UID = 1;
constexpr int64_t OUTPUT_UID = 2;
constexpr int64_t INPUT_OFFSET = 1;
constexpr int64_t OUTPUT_OFFSET = 2;

// hipMalloc returns a base aligned to at least this many bytes.
constexpr uintptr_t BASE_ALIGNMENT = 256;

// [2, 3] with a row stride of 5: 8 elements of storage, of which 6 are in the tensor.
std::shared_ptr<Tensor<float>> paddedTensor(float first)
{
    auto tensor
        = std::make_shared<Tensor<float>>(std::vector<int64_t>{2, 3}, std::vector<int64_t>{5, 1});
    auto* data = static_cast<float*>(tensor->rawHostData());
    for(size_t i = 0; i < tensor->elementSpace(); ++i)
    {
        data[i] = first + static_cast<float>(i);
    }
    tensor->markHostModified();
    return tensor;
}

std::vector<float> deviceValues(const void* device, size_t count)
{
    std::vector<float> values(count);
    EXPECT_EQ(hipMemcpy(values.data(), device, count * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    return values;
}

std::vector<float> hostValues(ITensor& tensor)
{
    const auto* data = static_cast<const float*>(tensor.rawHostData());
    return {data, data + tensor.elementSpace()};
}

class ShiftedStartsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        _input = paddedTensor(1.0f);
        _output = paddedTensor(100.0f);
        _tensors = {{INPUT_UID, _input}, {OUTPUT_UID, _output}};
        _variantPack
            = {{INPUT_UID, _input->rawDeviceData()}, {OUTPUT_UID, _output->rawDeviceData()}};
    }

    std::shared_ptr<Tensor<float>> _input;
    std::shared_ptr<Tensor<float>> _output;
    TensorMap _tensors;
    std::unordered_map<int64_t, void*> _variantPack;
};

TEST_F(ShiftedStartsTest, PointsTheVariantPackAtShiftedCopiesOfTheWholeStorage)
{
    const void* originalInput = _variantPack.at(INPUT_UID);
    const void* originalOutput = _variantPack.at(OUTPUT_UID);
    const ShiftedStarts shifted(
        _tensors, {{INPUT_UID, INPUT_OFFSET}, {OUTPUT_UID, OUTPUT_OFFSET}}, _variantPack);

    const void* input = _variantPack.at(INPUT_UID);
    const void* output = _variantPack.at(OUTPUT_UID);
    EXPECT_NE(input, originalInput);
    EXPECT_NE(output, originalOutput);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(input) % BASE_ALIGNMENT,
              static_cast<uintptr_t>(INPUT_OFFSET));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(output) % BASE_ALIGNMENT,
              static_cast<uintptr_t>(OUTPUT_OFFSET));
    EXPECT_EQ(deviceValues(input, _input->elementSpace()), hostValues(*_input));
    EXPECT_EQ(deviceValues(output, _output->elementSpace()), hostValues(*_output));

    hipDeviceptr_t base = nullptr;
    size_t allocated = 0;
    ASSERT_EQ(hipMemGetAddressRange(&base, &allocated, const_cast<void*>(output)), hipSuccess);
    EXPECT_EQ(static_cast<const char*>(output) - static_cast<const char*>(base), OUTPUT_OFFSET);
    EXPECT_GE(allocated,
              static_cast<size_t>(OUTPUT_OFFSET) + _output->elementSpace() * sizeof(float));
}

TEST_F(ShiftedStartsTest, LeavesUnlistedTensorsInPlace)
{
    const void* originalOutput = _variantPack.at(OUTPUT_UID);
    const ShiftedStarts shifted(_tensors, {{INPUT_UID, INPUT_OFFSET}}, _variantPack);

    EXPECT_EQ(_variantPack.at(OUTPUT_UID), originalOutput);
}

TEST_F(ShiftedStartsTest, CopiesOnlyTheOutputsBack)
{
    const auto inputBefore = hostValues(*_input);
    const ShiftedStarts shifted(
        _tensors, {{INPUT_UID, INPUT_OFFSET}, {OUTPUT_UID, OUTPUT_OFFSET}}, _variantPack);

    const std::vector<float> written = {-1, -2, -3, -4, -5, -6, -7, -8};
    ASSERT_EQ(written.size(), _output->elementSpace());
    const size_t bytes = written.size() * sizeof(float);
    ASSERT_EQ(hipMemcpy(_variantPack.at(OUTPUT_UID), written.data(), bytes, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(_variantPack.at(INPUT_UID), 0, bytes), hipSuccess);

    shifted.copyOutputsBack({OUTPUT_UID});

    EXPECT_EQ(deviceValues(_output->rawDeviceData(), written.size()), written);
    EXPECT_EQ(deviceValues(_input->rawDeviceData(), inputBefore.size()), inputBefore);
}

TEST_F(ShiftedStartsTest, RestoresTheVariantPackWhenATensorIsMissing)
{
    const void* originalInput = _variantPack.at(INPUT_UID);
    constexpr int64_t MISSING_UID = 3;
    _variantPack[MISSING_UID] = _input->rawDeviceData();

    EXPECT_THROW(
        static_cast<void>(ShiftedStarts(
            _tensors, {{INPUT_UID, INPUT_OFFSET}, {MISSING_UID, INPUT_OFFSET}}, _variantPack)),
        std::out_of_range);
    EXPECT_EQ(_variantPack.at(INPUT_UID), originalInput);
}

TEST_F(ShiftedStartsTest, RefusesAnOffsetThatIsNotPositive)
{
    const void* originalInput = _variantPack.at(INPUT_UID);

    EXPECT_THROW(static_cast<void>(ShiftedStarts(_tensors, {{INPUT_UID, 0}}, _variantPack)),
                 std::logic_error);
    EXPECT_EQ(_variantPack.at(INPUT_UID), originalInput);
}

} // namespace
