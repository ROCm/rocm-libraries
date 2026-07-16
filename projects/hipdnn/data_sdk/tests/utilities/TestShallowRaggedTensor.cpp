// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestRaggedTensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_ragged_test;

// ============================================================================
// Addressing math (parameterized over int32 / int64 aux)
// ============================================================================

template <typename IndexT>
class ShallowRaggedTensorTyped : public ::testing::Test
{
};

using IndexTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(ShallowRaggedTensorTyped, IndexTypes, );

TYPED_TEST(ShallowRaggedTensorTyped, Addressing)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    checkAddressing(tensor, K_DIMS, K_STRIDES, K_OFFSETS);
}

TYPED_TEST(ShallowRaggedTensorTyped, Iteration)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    checkIteration(tensor, K_OFFSETS);
}

TYPED_TEST(ShallowRaggedTensorTyped, Reporting)
{
    auto aux = makeOffsetAux<TypeParam>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    const ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    checkReporting(tensor, K_OFFSETS.back());
}

// ============================================================================
// Borrowed buffer semantics
// ============================================================================

TEST(TestShallowRaggedTensor, WrapsBorrowedBuffer)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    EXPECT_EQ(tensor.memory().hostData(), backing.data());

    tensor.setHostValue(7.0f, 1, 2, 1, 1); // physical slot 19
    EXPECT_FLOAT_EQ(backing[19], 7.0f);
}

TEST(TestShallowRaggedTensor, FillWithValueFillsBorrowedBuffer)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    std::vector<float> backing(20, -1.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    tensor.fillWithValue(3.0f);
    for(const auto& v : backing)
    {
        EXPECT_FLOAT_EQ(v, 3.0f);
    }
}

TEST(TestShallowRaggedTensor, EmptyBatchSkipped)
{
    const std::vector<int64_t> dims = {3, 3, 2, 2};
    const std::vector<int64_t> strides = {12, 4, 2, 1};
    const std::vector<int64_t> offsets = {0, 4, 4, 8};

    auto aux = makeOffsetAux<int32_t>(offsets);
    std::vector<float> backing(8, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), dims, strides, BSHD_SEQ_AXIS, aux);

    EXPECT_EQ(tensor.elementCount(), 8u);
    checkIteration(tensor, offsets);
}

// ============================================================================
// Unsupported operations throw (host-only, non-owning)
// ============================================================================

TEST(TestShallowRaggedTensor, FillWithRandomValuesThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    EXPECT_THROW(tensor.fillWithRandomValues(0.0f, 1.0f, 1337), std::runtime_error);
}

TEST(TestShallowRaggedTensor, FillWithDataThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    std::vector<float> data(20, 1.0f);
    EXPECT_THROW(tensor.fillWithData(data.data(), data.size() * sizeof(float)), std::runtime_error);
}

TEST(TestShallowRaggedTensor, DeviceAccessThrows)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    EXPECT_THROW(tensor.memory().deviceData(), std::runtime_error);
}

// ============================================================================
// Structural validation (shared with RaggedTensor)
// ============================================================================

TEST(TestShallowRaggedTensor, ValidationNullAuxThrows)
{
    std::vector<float> backing(20, 0.0f);
    EXPECT_THROW(const ShallowRaggedTensor<float> tensor(
                     backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, nullptr),
                 std::invalid_argument);
}

TEST(TestShallowRaggedTensor, ValidationWrongElementCountThrows)
{
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{2, 1, 1, 1});
    std::vector<float> backing(20, 0.0f);
    EXPECT_THROW(const ShallowRaggedTensor<float> tensor(
                     backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestShallowRaggedTensor, ValidationWrongRankThrows)
{
    auto aux = std::make_shared<Tensor<int32_t>>(std::vector<int64_t>{3, 1, 1});
    std::vector<float> backing(20, 0.0f);
    EXPECT_THROW(const ShallowRaggedTensor<float> tensor(
                     backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux),
                 std::invalid_argument);
}

TEST(TestShallowRaggedTensor, ValidationBadElementSizeThrows)
{
    auto aux = std::make_shared<Tensor<int16_t>>(std::vector<int64_t>{3, 1, 1, 1});
    std::vector<float> backing(20, 0.0f);
    EXPECT_THROW(const ShallowRaggedTensor<float> tensor(
                     backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux),
                 std::invalid_argument);
}
