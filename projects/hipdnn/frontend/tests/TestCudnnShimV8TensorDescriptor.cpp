// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/MockBackendFixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;
using hipdnn_frontend::compatibility::cudnn_frontend::DataType_t;

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class TestCudnnShimV8TensorDescriptor : public hipdnn_shim_test::ShimMockBackendFixture
{
protected:
    std::vector<cudnnBackendDescriptor_t> _owned;

    void SetUp() override
    {
        ShimMockBackendFixture::SetUp();
        lastError();
    }

    void TearDown() override
    {
        for(cudnnBackendDescriptor_t descriptor : _owned)
        {
            cudnnBackendDestroyDescriptor(descriptor);
        }
        _owned.clear();
        ShimMockBackendFixture::TearDown();
    }

    cudnnBackendDescriptor_t create()
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &descriptor),
                  CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    static cudnnStatus_t setInt64(cudnnBackendDescriptor_t descriptor,
                                  cudnnBackendAttributeName_t name,
                                  std::int64_t value)
    {
        return cudnnBackendSetAttribute(descriptor, name, CUDNN_TYPE_INT64, 1, &value);
    }

    static cudnnStatus_t setInt64s(cudnnBackendDescriptor_t descriptor,
                                   cudnnBackendAttributeName_t name,
                                   const std::vector<std::int64_t>& values)
    {
        return cudnnBackendSetAttribute(descriptor,
                                        name,
                                        CUDNN_TYPE_INT64,
                                        static_cast<std::int64_t>(values.size()),
                                        values.data());
    }

    static cudnnStatus_t
        setBool(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name, bool value)
    {
        return cudnnBackendSetAttribute(descriptor, name, CUDNN_TYPE_BOOLEAN, 1, &value);
    }

    static cudnnStatus_t setDataType(cudnnBackendDescriptor_t descriptor, cudnnDataType_t dataType)
    {
        return cudnnBackendSetAttribute(
            descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType);
    }

    // Every attribute finalize requires, as TensorBuilder_v8 sends them, minus
    // `skip`.
    cudnnBackendDescriptor_t populated(std::optional<cudnnBackendAttributeName_t> skip
                                       = std::nullopt,
                                       std::int64_t alignment = 16,
                                       std::int64_t id = 7,
                                       cudnnDataType_t dataType = CUDNN_DATA_FLOAT)
    {
        cudnnBackendDescriptor_t tensor = create();
        if(skip != CUDNN_ATTR_TENSOR_DATA_TYPE)
        {
            EXPECT_EQ(setDataType(tensor, dataType), CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_TENSOR_DIMENSIONS)
        {
            EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, {2, 3, 4, 5}),
                      CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_TENSOR_STRIDES)
        {
            EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_STRIDES, {60, 20, 5, 1}),
                      CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_TENSOR_UNIQUE_ID)
        {
            EXPECT_EQ(setInt64(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, id), CUDNN_STATUS_SUCCESS);
        }
        if(skip != CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT)
        {
            EXPECT_EQ(setInt64(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, alignment),
                      CUDNN_STATUS_SUCCESS);
        }
        return tensor;
    }

    static const backend::TensorDescriptor& object(cudnnBackendDescriptor_t descriptor)
    {
        return backend::descriptorCast<backend::TensorDescriptor>(
            *backend::liveHandle(descriptor)->object, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    }
};

TEST_F(TestCudnnShimV8TensorDescriptor, FinalizeRecordsLoweringFacts)
{
    cudnnBackendDescriptor_t tensor = populated();
    ASSERT_EQ(setBool(tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, true), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);

    const backend::TensorDescriptor& lowered = object(tensor);
    EXPECT_EQ(lowered.dataType(), DataType_t::FLOAT);
    EXPECT_EQ(lowered.dimensions(), (std::vector<std::int64_t>{2, 3, 4, 5}));
    EXPECT_EQ(lowered.strides(), (std::vector<std::int64_t>{60, 20, 5, 1}));
    EXPECT_EQ(lowered.uniqueId(), 7);
    EXPECT_TRUE(lowered.isVirtual());
    EXPECT_FALSE(lowered.isRuntimePassByValue());
    EXPECT_FALSE(lowered.constantValue().has_value());
    EXPECT_EQ(lowered.raggedOffset(), nullptr);
}

TEST_F(TestCudnnShimV8TensorDescriptor, AlignmentForwardedOnlyBelowSixteen)
{
    cudnnBackendDescriptor_t wide = populated(std::nullopt, 32);
    cudnnBackendDescriptor_t narrow = populated(std::nullopt, 4);
    cudnnBackendDescriptor_t exact = populated(std::nullopt, 16);
    ASSERT_EQ(cudnnBackendFinalize(wide), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(narrow), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(exact), CUDNN_STATUS_SUCCESS);

    EXPECT_FALSE(object(wide).forwardedAlignment().has_value());
    EXPECT_FALSE(object(exact).forwardedAlignment().has_value());
    ASSERT_TRUE(object(narrow).forwardedAlignment().has_value());
    EXPECT_EQ(*object(narrow).forwardedAlignment(), 4);

    // The cuDNN value itself is still what GET reports.
    std::int64_t alignment = 0;
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(
                  wide, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &count, &alignment),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(alignment, 32);
}

TEST_F(TestCudnnShimV8TensorDescriptor, IsByValueAloneLowersToRuntimePassByValue)
{
    cudnnBackendDescriptor_t tensor = populated();
    ASSERT_EQ(setBool(tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, true), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
    EXPECT_TRUE(object(tensor).isRuntimePassByValue());
    EXPECT_FALSE(object(tensor).constantValue().has_value());
}

TEST_F(TestCudnnShimV8TensorDescriptor, ConstantValueTakesPrecedenceOverIsByValue)
{
    // TensorBuilder_v8::setConstValue sets IS_BY_VALUE as well.
    const float value = 2.5F;
    const void* pointer = &value;
    cudnnBackendDescriptor_t tensor = populated();
    ASSERT_EQ(setBool(tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, true), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       static_cast<const void*>(&pointer)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
    EXPECT_FALSE(object(tensor).isRuntimePassByValue());
    EXPECT_TRUE(object(tensor).constantValue().has_value());
}

TEST_F(TestCudnnShimV8TensorDescriptor, ConstantValueCopiedBeforeSourceIsFreed)
{
    cudnnBackendDescriptor_t tensor = create();
    // CONSTANT_VALUE may precede DATA_TYPE; the copy happens at finalize.
    auto source = std::make_unique<double>(-3.25);
    const void* pointer = source.get();
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       static_cast<const void*>(&pointer)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setDataType(tensor, CUDNN_DATA_DOUBLE), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, {1}), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_STRIDES, {1}), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setInt64(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, 1), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setInt64(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, 8), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
    source.reset();

    const auto& constant = object(tensor).constantValue();
    ASSERT_TRUE(constant.has_value());
    ASSERT_EQ(constant->size(), sizeof(double));
    double copied = 0.0;
    std::memcpy(&copied, constant->data(), sizeof(copied));
    EXPECT_EQ(copied, -3.25);

    // GET hands back the shim's copy, never the freed builder buffer.
    const void* reported = nullptr;
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       &count,
                                       static_cast<void*>(&reported)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(reported, static_cast<const void*>(constant->data()));
}

TEST_F(TestCudnnShimV8TensorDescriptor, NullConstantValueRejected)
{
    const void* pointer = nullptr;
    cudnnBackendDescriptor_t tensor = populated();
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       static_cast<const void*>(&pointer)),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_CONSTANT_VALUE"), std::string::npos);
}

TEST_F(TestCudnnShimV8TensorDescriptor, BooleanReadAsOneByte)
{
    // Upstream passes the address of a C++ bool; the bytes after it are not
    // part of the value.
    const std::array<unsigned char, 4> isVirtual = {1, 0x7F, 0x7F, 0x7F};
    const std::array<unsigned char, 4> constant = {1, 0x7F, 0x7F, 0x7F};
    const void* pointer = constant.data();
    cudnnBackendDescriptor_t tensor = populated(std::nullopt, 16, 7, CUDNN_DATA_BOOLEAN);
    ASSERT_EQ(cudnnBackendSetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, isVirtual.data()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
                                       CUDNN_TYPE_VOID_PTR,
                                       1,
                                       static_cast<const void*>(&pointer)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);

    EXPECT_TRUE(object(tensor).isVirtual());
    ASSERT_TRUE(object(tensor).constantValue().has_value());
    EXPECT_EQ(object(tensor).constantValue()->size(), 1U);

    std::array<unsigned char, 4> read = {0x55, 0x55, 0x55, 0x55};
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &count, read.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(read, (std::array<unsigned char, 4>{1, 0x55, 0x55, 0x55}));
}

TEST_F(TestCudnnShimV8TensorDescriptor, VectorCountNotSupported)
{
    cudnnBackendDescriptor_t tensor = populated();
    ASSERT_EQ(setInt64(tensor, CUDNN_ATTR_TENSOR_VECTOR_COUNT, 4), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_VECTOR_COUNT"), std::string::npos);
}

TEST_F(TestCudnnShimV8TensorDescriptor, ReorderingNotSupported)
{
    cudnnBackendDescriptor_t tensor = populated();
    const cudnnBackendTensorReordering_t reordering = CUDNN_TENSOR_REORDERING_INT8x32;
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_REORDERING_MODE,
                                       CUDNN_TYPE_TENSOR_REORDERING_MODE,
                                       1,
                                       &reordering),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_REORDERING_MODE"), std::string::npos);
}

TEST_F(TestCudnnShimV8TensorDescriptor, ExplicitDefaultVectorCountAndReorderingAccepted)
{
    cudnnBackendDescriptor_t tensor = populated();
    const cudnnBackendTensorReordering_t reordering = CUDNN_TENSOR_REORDERING_NONE;
    ASSERT_EQ(setInt64(tensor, CUDNN_ATTR_TENSOR_VECTOR_COUNT, 1), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_REORDERING_MODE,
                                       CUDNN_TYPE_TENSOR_REORDERING_MODE,
                                       1,
                                       &reordering),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8TensorDescriptor, UnlowerableDataTypeNotSupported)
{
    cudnnBackendDescriptor_t tensor = populated(std::nullopt, 16, 7, CUDNN_DATA_INT8x4);
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_DATA_TYPE"), std::string::npos);
}

TEST_F(TestCudnnShimV8TensorDescriptor, MissingRequiredAttributeNamed)
{
    for(const cudnnBackendAttributeName_t required : {CUDNN_ATTR_TENSOR_DATA_TYPE,
                                                      CUDNN_ATTR_TENSOR_DIMENSIONS,
                                                      CUDNN_ATTR_TENSOR_STRIDES,
                                                      CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                                      CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT})
    {
        const std::string name = backend::attributeName(required);
        SCOPED_TRACE(name);
        cudnnBackendDescriptor_t tensor = populated(required);
        EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_BAD_PARAM);
        EXPECT_NE(lastError().find(name), std::string::npos);
    }
}

TEST_F(TestCudnnShimV8TensorDescriptor, ShapeRulesEnforced)
{
    cudnnBackendDescriptor_t mismatched = populated();
    ASSERT_EQ(setInt64s(mismatched, CUDNN_ATTR_TENSOR_STRIDES, {20, 5, 1}), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(mismatched), CUDNN_STATUS_BAD_PARAM);

    // Count 0 with a non-null array is a valid SET; finalize rejects it.
    const std::int64_t unused = 1;
    cudnnBackendDescriptor_t empty = populated();
    ASSERT_EQ(
        cudnnBackendSetAttribute(empty, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 0, &unused),
        CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(
        cudnnBackendSetAttribute(empty, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 0, &unused),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(empty), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_DIMENSIONS"), std::string::npos);

    cudnnBackendDescriptor_t zeroDim = populated();
    ASSERT_EQ(setInt64s(zeroDim, CUDNN_ATTR_TENSOR_DIMENSIONS, {2, 0, 4, 5}), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(zeroDim), CUDNN_STATUS_BAD_PARAM);

    cudnnBackendDescriptor_t negativeStride = populated();
    ASSERT_EQ(setInt64s(negativeStride, CUDNN_ATTR_TENSOR_STRIDES, {60, 20, -5, 1}),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(negativeStride), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8TensorDescriptor, IdAndAlignmentBoundsEnforced)
{
    cudnnBackendDescriptor_t negativeId = populated(std::nullopt, 16, -1);
    EXPECT_EQ(cudnnBackendFinalize(negativeId), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_UNIQUE_ID"), std::string::npos);

    cudnnBackendDescriptor_t zeroAlignment = populated(std::nullopt, 0);
    EXPECT_EQ(cudnnBackendFinalize(zeroAlignment), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT"), std::string::npos);

    cudnnBackendDescriptor_t zeroId = populated(std::nullopt, 16, 0);
    EXPECT_EQ(cudnnBackendFinalize(zeroId), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8TensorDescriptor, RaggedOffsetStored)
{
    cudnnBackendDescriptor_t offsets = populated(std::nullopt, 16, 1, CUDNN_DATA_INT32);
    ASSERT_EQ(cudnnBackendFinalize(offsets), CUDNN_STATUS_SUCCESS);
    cudnnBackendDescriptor_t tensor = populated();
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&offsets)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(object(tensor).raggedOffset().get(), &object(offsets));
}

} // namespace
