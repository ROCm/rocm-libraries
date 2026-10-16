// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Lifecycle, GET/SET count and type rules, and last-error behavior of the cuDNN
// v8 shim backend descriptor C API. Every descriptor type is storage-only until
// its validating class lands, so finalized descriptors here come from a
// test-only subclass whose finalize always succeeds.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;

class FinalizableDescriptor final : public backend::Descriptor
{
public:
    using Descriptor::arrayAttribute;
    using Descriptor::Descriptor;
    using Descriptor::descriptorAttribute;
    using Descriptor::realAttribute;
    using Descriptor::scalarAttribute;
    using Descriptor::stringAttribute;

protected:
    void finalizeImpl() override {}
};

// Finalize throws whatever `raise` throws, to exercise the entry-point catch ladder.
class ThrowingDescriptor final : public backend::Descriptor
{
public:
    explicit ThrowingDescriptor(std::function<void()> raise)
        : Descriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)
        , _raise(std::move(raise))
    {
    }

protected:
    void finalizeImpl() override
    {
        _raise();
    }

private:
    std::function<void()> _raise;
};

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class TestCudnnShimV8BackendApi : public hipdnn_shim_test::ShimMockBackendFixture
{
protected:
    std::vector<cudnnBackendDescriptor_t> _owned;

    void SetUp() override
    {
        ShimMockBackendFixture::SetUp();
        // The last-error buffer is thread-local and outlives a single test.
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

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t adopt(std::shared_ptr<backend::Descriptor> object)
    {
        cudnnBackendDescriptor_t descriptor = backend::makeHandle(std::move(object));
        _owned.push_back(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t createFinalizable(cudnnBackendDescriptorType_t type)
    {
        return adopt(std::make_shared<FinalizableDescriptor>(type));
    }

    void release(cudnnBackendDescriptor_t descriptor)
    {
        _owned.erase(std::remove(_owned.begin(), _owned.end(), descriptor), _owned.end());
        EXPECT_EQ(cudnnBackendDestroyDescriptor(descriptor), CUDNN_STATUS_SUCCESS);
    }

    static FinalizableDescriptor& object(cudnnBackendDescriptor_t descriptor)
    {
        return static_cast<FinalizableDescriptor&>(
            *static_cast<backend::Handle*>(descriptor)->object);
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

    cudnnBackendDescriptor_t finalizedTensor(const std::vector<std::int64_t>& dims)
    {
        cudnnBackendDescriptor_t tensor = createFinalizable(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, dims), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
        return tensor;
    }

    cudnnBackendDescriptor_t finalized(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = createFinalizable(type);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }
};

TEST_F(TestCudnnShimV8BackendApi, CreateSetDestroyLifecycle)
{
    cudnnBackendDescriptor_t tensor = nullptr;
    ASSERT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor),
              CUDNN_STATUS_SUCCESS);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, {1, 2, 3, 4}), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendDestroyDescriptor(tensor), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8BackendApi, CreateRejectsNullOutput)
{
    EXPECT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, nullptr),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("cudnnBackendCreateDescriptor"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, CreateRejectsUnknownType)
{
    // Pick an in-range value that is not an enumerator.
    int unused = 0;
    while(
        std::string(backend::descriptorTypeName(static_cast<cudnnBackendDescriptorType_t>(unused)))
        != "unknown cudnnBackendDescriptorType_t")
    {
        ++unused;
    }
    cudnnBackendDescriptor_t descriptor = nullptr;
    EXPECT_EQ(cudnnBackendCreateDescriptor(static_cast<cudnnBackendDescriptorType_t>(unused),
                                           &descriptor),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(descriptor, nullptr);
}

TEST_F(TestCudnnShimV8BackendApi, StorageOnlyFinalizeNamesType)
{
    cudnnBackendDescriptor_t heuristics = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendFinalize(heuristics), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, NullDescriptorRejected)
{
    const std::int64_t value = 1;
    std::int64_t count = 0;
    EXPECT_EQ(
        cudnnBackendSetAttribute(nullptr, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &value),
        CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendGetAttribute(
                  nullptr, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 0, &count, nullptr),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendFinalize(nullptr), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, ZeroFilledBufferRejected)
{
    alignas(std::uint64_t) std::array<std::byte, 64> zeros{};
    cudnnBackendDescriptor_t bogus = zeros.data();
    const std::int64_t value = 1;
    std::int64_t count = 0;
    EXPECT_EQ(
        cudnnBackendSetAttribute(bogus, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &value),
        CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendGetAttribute(
                  bogus, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 0, &count, nullptr),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendFinalize(bogus), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendDestroyDescriptor(bogus), CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(
        std::all_of(zeros.begin(), zeros.end(), [](std::byte b) { return b == std::byte{0}; }));
}

TEST_F(TestCudnnShimV8BackendApi, DestroyNullKeepsLastError)
{
    EXPECT_EQ(cudnnBackendFinalize(nullptr), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendDestroyDescriptor(nullptr), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("cudnnBackendFinalize"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, SetAfterFinalizeNotInitialized)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({2, 3});
    EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, {7}), CUDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCudnnShimV8BackendApi, GetBeforeFinalizeNotInitialized)
{
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    ASSERT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, {7}), CUDNN_STATUS_SUCCESS);
    std::int64_t value = 0;
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &count, &value),
              CUDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCudnnShimV8BackendApi, FinalizeTwiceBadParam)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({2});
    EXPECT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, NullArrayIsCountQueryEvenWithRequest)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({4, 32, 32, 32});
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 9, &count, nullptr),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 4);
}

TEST_F(TestCudnnShimV8BackendApi, ZeroRequestIsCountQuery)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({4, 32, 32, 32});
    std::array<std::int64_t, 4> dims = {-1, -1, -1, -1};
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 0, &count, dims.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 4);
    EXPECT_EQ(dims[0], -1);
}

TEST_F(TestCudnnShimV8BackendApi, CountQueryNeedsElementCount)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({4});
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 0, nullptr, nullptr),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, ShortBufferCopiesRequestedCount)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({4, 32, 16, 8});
    std::array<std::int64_t, 3> dims = {-1, -1, -1};
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 2, &count, dims.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 32);
    EXPECT_EQ(dims[2], -1);
}

TEST_F(TestCudnnShimV8BackendApi, OversizedRequestReportsWritten)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({4, 32, 16, 8});
    std::array<std::int64_t, 5> dims = {-1, -1, -1, -1, -1};
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, &count, dims.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 4);
    EXPECT_EQ(dims[3], 8);
    EXPECT_EQ(dims[4], -1);
}

TEST_F(TestCudnnShimV8BackendApi, ReadAllowsNullElementCount)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({6});
    std::int64_t dim = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 1, nullptr, &dim),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(dim, 6);
}

TEST_F(TestCudnnShimV8BackendApi, NegativeRequestRejected)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({6});
    std::int64_t dim = 0;
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, -1, &count, &dim),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, ScalarSetRequiresExactlyOne)
{
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, {1, 2}), CUDNN_STATUS_BAD_PARAM);
    const std::int64_t id = 1;
    EXPECT_EQ(
        cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 0, &id),
        CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, ArraySetAcceptsZeroCount)
{
    // Upstream ResampleDescBuilder sends count 0 when spatial dims were never set.
    cudnnBackendDescriptor_t resample = createFinalizable(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    const std::array<cudnnFraction_t, 1> unused = {{{1, 1}}};
    ASSERT_EQ(cudnnBackendSetAttribute(
                  resample, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, 0, unused.data()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(resample), CUDNN_STATUS_SUCCESS);
    std::int64_t count = -1;
    EXPECT_EQ(
        cudnnBackendGetAttribute(
            resample, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, 0, &count, nullptr),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 0);
}

TEST_F(TestCudnnShimV8BackendApi, ArraySetRejectsBadCounts)
{
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    const std::vector<std::int64_t> tooMany(CUDNN_DIM_MAX + 2, 1);
    EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, tooMany), CUDNN_STATUS_BAD_PARAM);
    const std::vector<std::int64_t> maximum(CUDNN_DIM_MAX + 1, 1);
    EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, maximum), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendSetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, -1, maximum.data()),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, NullArraySetRejected)
{
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(
        cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, nullptr),
        CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, NullArrayAcceptedForZeroCount)
{
    // Upstream passes data() of vectors that may be empty, which may be null.
    cudnnBackendDescriptor_t pack = createFinalizable(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendSetAttribute(
                  pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 0, nullptr),
              CUDNN_STATUS_SUCCESS);
    cudnnBackendDescriptor_t plan = createFinalizable(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendSetAttribute(
                  plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 0, nullptr),
              CUDNN_STATUS_SUCCESS);
    // A scalar still needs exactly one element.
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(
        cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 0, nullptr),
        CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, AttributeTypeMustBeAccepted)
{
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    const std::array<std::int32_t, 2> dims = {2, 2};
    EXPECT_EQ(cudnnBackendSetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT32, 2, dims.data()),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_TYPE_INT32"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, ForeignAttributeRejected)
{
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(setInt64s(tensor, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, {2}), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, UnknownAttributeRejected)
{
    int unused = 0;
    while(backend::attributeInfo(static_cast<cudnnBackendAttributeName_t>(unused)).has_value())
    {
        ++unused;
    }
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    const std::int64_t value = 1;
    EXPECT_EQ(
        cudnnBackendSetAttribute(
            tensor, static_cast<cudnnBackendAttributeName_t>(unused), CUDNN_TYPE_INT64, 1, &value),
        CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, ComputedAttributeNotSettable)
{
    cudnnBackendDescriptor_t plan = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    EXPECT_EQ(setInt64s(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, {0}),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, BooleanIsOneByte)
{
    cudnnBackendDescriptor_t tensor = createFinalizable(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    // Upstream passes the address of a C++ bool; only the first byte is valid.
    std::array<unsigned char, 4> input = {1, 0x7F, 0x7F, 0x7F};
    ASSERT_EQ(cudnnBackendSetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, input.data()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);

    std::array<unsigned char, 4> output = {0xAA, 0xAA, 0xAA, 0xAA};
    std::int64_t count = 0;
    EXPECT_EQ(
        cudnnBackendGetAttribute(
            tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &count, output.data()),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 0xAA);
    EXPECT_TRUE(object(tensor).scalarAttribute<bool>(CUDNN_ATTR_TENSOR_IS_VIRTUAL).value_or(false));
}

TEST_F(TestCudnnShimV8BackendApi, CharCountIncludesNul)
{
    cudnnBackendDescriptor_t plan = createFinalizable(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    // Upstream round-trips the buffer it read, NUL included.
    const std::array<char, 4> json = {'a', 'b', 'c', '\0'};
    ASSERT_EQ(
        cudnnBackendSetAttribute(
            plan, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR, 4, json.data()),
        CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(plan), CUDNN_STATUS_SUCCESS);

    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 4);

    std::array<char, 8> full{};
    full.fill('x');
    EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       static_cast<std::int64_t>(full.size()),
                                       &count,
                                       full.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 4);
    EXPECT_STREQ(full.data(), "abc");

    std::array<char, 2> shortBuffer{};
    EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       2,
                                       &count,
                                       shortBuffer.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(shortBuffer[0], 'a');
    EXPECT_EQ(shortBuffer[1], '\0');
    EXPECT_EQ(
        object(plan).stringAttribute(CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION).value_or(""),
        "abc");
}

TEST_F(TestCudnnShimV8BackendApi, CharSetWithoutNulStored)
{
    cudnnBackendDescriptor_t plan = createFinalizable(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    const std::string json = "{}";
    ASSERT_EQ(cudnnBackendSetAttribute(plan,
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       static_cast<std::int64_t>(json.size()),
                                       json.data()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(plan), CUDNN_STATUS_SUCCESS);
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 3);
}

TEST_F(TestCudnnShimV8BackendApi, FractionRoundTripsAndRejectsZeroDenominator)
{
    cudnnBackendDescriptor_t resample = createFinalizable(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    const std::array<cudnnFraction_t, 2> bad = {{{2, 1}, {3, 0}}};
    EXPECT_EQ(cudnnBackendSetAttribute(
                  resample, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, 2, bad.data()),
              CUDNN_STATUS_BAD_PARAM);

    const std::array<cudnnFraction_t, 2> strides = {{{2, 1}, {3, 2}}};
    ASSERT_EQ(cudnnBackendSetAttribute(
                  resample, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, 2, strides.data()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(resample), CUDNN_STATUS_SUCCESS);
    std::array<cudnnFraction_t, 2> out{};
    std::int64_t count = 0;
    EXPECT_EQ(
        cudnnBackendGetAttribute(
            resample, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, 2, &count, out.data()),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(out[1].numerator, 3);
    EXPECT_EQ(out[1].denominator, 2);
}

TEST_F(TestCudnnShimV8BackendApi, UnsetAttributesReturnCudnnDefaults)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({2});
    bool flag = true;
    std::int64_t value = 0;
    auto reordering = CUDNN_TENSOR_REORDERING_INT8x32;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, nullptr, &flag),
              CUDNN_STATUS_SUCCESS);
    EXPECT_FALSE(flag);
    flag = true;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, nullptr, &flag),
              CUDNN_STATUS_SUCCESS);
    EXPECT_FALSE(flag);
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_VECTOR_COUNT, CUDNN_TYPE_INT64, 1, nullptr, &value),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(value, 1);
    EXPECT_EQ(
        cudnnBackendGetAttribute(
            tensor, CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION, CUDNN_TYPE_INT64, 1, nullptr, &value),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(value, -1);
    EXPECT_EQ(cudnnBackendGetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_REORDERING_MODE,
                                       CUDNN_TYPE_TENSOR_REORDERING_MODE,
                                       1,
                                       nullptr,
                                       &reordering),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(reordering, CUDNN_TENSOR_REORDERING_NONE);
    EXPECT_EQ(
        object(tensor).scalarAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_VECTOR_COUNT).value_or(0),
        1);
}

TEST_F(TestCudnnShimV8BackendApi, UnsetAttributeWithoutDefaultRejected)
{
    cudnnBackendDescriptor_t tensor = finalizedTensor({2});
    std::int64_t value = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(
                  tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, nullptr, &value),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_UNIQUE_ID"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, UnsetArrayAttributesReadAsEmpty)
{
    // ExecutionPlan_v8 GETs KNOB_CHOICES even when the engine config has no knobs.
    cudnnBackendDescriptor_t config = finalized(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(config,
                                       CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 0);

    cudnnBackendDescriptor_t pack = finalized(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    std::array<std::int64_t, 2> ids = {-1, -1};
    count = -1;
    EXPECT_EQ(
        cudnnBackendGetAttribute(
            pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 2, &count, ids.data()),
        CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(ids[0], -1);

    cudnnBackendDescriptor_t plan = finalized(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    std::array<char, 4> json{};
    json.fill('x');
    count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(plan,
                                       CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                       CUDNN_TYPE_CHAR,
                                       static_cast<std::int64_t>(json.size()),
                                       &count,
                                       json.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(json[0], 'x');
}

TEST_F(TestCudnnShimV8BackendApi, AlphaAcceptsFloatOrDouble)
{
    cudnnBackendDescriptor_t floatOp
        = createFinalizable(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const float alphaFloat = 0.5F;
    ASSERT_EQ(cudnnBackendSetAttribute(floatOp,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                       CUDNN_TYPE_FLOAT,
                                       1,
                                       &alphaFloat),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(floatOp), CUDNN_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(
        object(floatOp).realAttribute(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA).value_or(0.0),
        0.5);
    double readBack = 0.0;
    EXPECT_EQ(cudnnBackendGetAttribute(floatOp,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       nullptr,
                                       &readBack),
              CUDNN_STATUS_BAD_PARAM);

    cudnnBackendDescriptor_t doubleOp
        = createFinalizable(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const double alphaDouble = 2.0;
    ASSERT_EQ(cudnnBackendSetAttribute(doubleOp,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                       CUDNN_TYPE_DOUBLE,
                                       1,
                                       &alphaDouble),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(doubleOp), CUDNN_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(object(doubleOp)
                         .realAttribute(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA)
                         .value_or(0.0),
                     2.0);
}

TEST_F(TestCudnnShimV8BackendApi, DescriptorSetRequiresFinalizedReferent)
{
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    cudnnBackendDescriptor_t tensor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendSetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&tensor)),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("not finalized"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, DescriptorSetRequiresAllowedType)
{
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    cudnnBackendDescriptor_t conv = finalized(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendSetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&conv)),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(cudnnBackendSetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&conv)),
              CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8BackendApi, DescriptorSetRejectsBadElement)
{
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    alignas(std::uint64_t) std::array<std::byte, 64> zeros{};
    cudnnBackendDescriptor_t bogus = zeros.data();
    EXPECT_EQ(cudnnBackendSetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&bogus)),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, OperationGraphAcceptsOnlyOperations)
{
    cudnnBackendDescriptor_t graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    std::array<cudnnBackendDescriptor_t, 2> ops
        = {finalized(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR),
           finalized(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)};
    EXPECT_EQ(cudnnBackendSetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(ops.data())),
              CUDNN_STATUS_SUCCESS);
    cudnnBackendDescriptor_t tensor = finalizedTensor({1});
    EXPECT_EQ(cudnnBackendSetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&tensor)),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BackendApi, UnavailableReferentNotSupported)
{
    cudnnBackendDescriptor_t heuristics = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    cudnnBackendDescriptor_t tensor = finalizedTensor({1});
    EXPECT_EQ(cudnnBackendSetAttribute(heuristics,
                                       CUDNN_ATTR_ENGINEHEUR_DEVICEPROP,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&tensor)),
              CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8BackendApi, RngSeedAcceptsValueOrDescriptor)
{
    cudnnBackendDescriptor_t rng = create(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR);
    EXPECT_EQ(setInt64s(rng, CUDNN_ATTR_OPERATION_RNG_SEED, {42}), CUDNN_STATUS_SUCCESS);
    cudnnBackendDescriptor_t seed = finalizedTensor({1});
    EXPECT_EQ(cudnnBackendSetAttribute(rng,
                                       CUDNN_ATTR_OPERATION_RNG_SEED,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&seed)),
              CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8BackendApi, InPlaceGetFillsCallerDescriptor)
{
    cudnnBackendDescriptor_t op
        = createFinalizable(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    cudnnBackendDescriptor_t x = finalizedTensor({4, 32, 16, 8});
    ASSERT_EQ(cudnnBackendSetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&x)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(object(op).descriptorAttribute(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X),
              static_cast<backend::Handle*>(x)->object);

    cudnnBackendDescriptor_t wrongType = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&wrongType)),
              CUDNN_STATUS_BAD_PARAM);

    // Errata pattern: fill a caller-created tensor, then read 5 dims of a 4-D tensor.
    cudnnBackendDescriptor_t target = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    ASSERT_EQ(cudnnBackendGetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&target)),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    std::array<std::int64_t, 5> dims{};
    EXPECT_EQ(cudnnBackendGetAttribute(
                  target, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 5, &count, dims.data()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 4);
    EXPECT_EQ(dims[1], 32);
}

TEST_F(TestCudnnShimV8BackendApi, InPlaceGetValidatesEveryTargetFirst)
{
    cudnnBackendDescriptor_t graph = createFinalizable(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    const std::array<cudnnBackendDescriptor_t, 2> ops
        = {finalized(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR),
           finalized(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)};
    ASSERT_EQ(cudnnBackendSetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(ops.data())),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(graph), CUDNN_STATUS_SUCCESS);

    std::array<cudnnBackendDescriptor_t, 2> targets
        = {create(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR),
           create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)};
    const std::shared_ptr<backend::Descriptor> original
        = static_cast<backend::Handle*>(targets[0])->object;
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       &count,
                                       static_cast<void*>(targets.data())),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("output element 1"), std::string::npos);
    EXPECT_EQ(static_cast<backend::Handle*>(targets[0])->object, original);
    EXPECT_EQ(count, -1);

    targets[1] = create(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendGetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       &count,
                                       static_cast<void*>(targets.data())),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 2);
    EXPECT_EQ(static_cast<backend::Handle*>(targets[0])->object,
              static_cast<backend::Handle*>(ops[0])->object);
    EXPECT_EQ(static_cast<backend::Handle*>(targets[1])->object,
              static_cast<backend::Handle*>(ops[1])->object);
}

TEST_F(TestCudnnShimV8BackendApi, InPlaceGetIntoSourceHandle)
{
    cudnnBackendDescriptor_t offsets = finalizedTensor({3});
    cudnnBackendDescriptor_t tensor = createFinalizable(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    ASSERT_EQ(cudnnBackendSetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&offsets)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(tensor), CUDNN_STATUS_SUCCESS);
    // Re-points the handle being read, dropping its only other reference to the source.
    ASSERT_EQ(cudnnBackendGetAttribute(tensor,
                                       CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       static_cast<void*>(&tensor)),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(static_cast<backend::Handle*>(tensor)->object,
              static_cast<backend::Handle*>(offsets)->object);
}

TEST_F(TestCudnnShimV8BackendApi, ReferenceSurvivesDestroy)
{
    cudnnBackendDescriptor_t op
        = createFinalizable(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    cudnnBackendDescriptor_t x = finalizedTensor({7, 3});
    ASSERT_EQ(cudnnBackendSetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(&x)),
              CUDNN_STATUS_SUCCESS);
    release(x);
    ASSERT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS);

    cudnnBackendDescriptor_t target = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    ASSERT_EQ(cudnnBackendGetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       nullptr,
                                       static_cast<void*>(&target)),
              CUDNN_STATUS_SUCCESS);
    const std::vector<std::int64_t> dims
        = object(target).arrayAttribute<std::int64_t>(CUDNN_ATTR_TENSOR_DIMENSIONS);
    EXPECT_EQ(dims, (std::vector<std::int64_t>{7, 3}));
}

TEST_F(TestCudnnShimV8BackendApi, LastErrorIsPerThread)
{
    std::string firstMessage;
    std::string secondMessage;
    std::thread first([&] {
        cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, nullptr);
        firstMessage = lastError();
    });
    first.join();
    std::thread second([&] {
        cudnnBackendFinalize(nullptr);
        secondMessage = lastError();
    });
    second.join();
    EXPECT_NE(firstMessage.find("cudnnBackendCreateDescriptor"), std::string::npos);
    EXPECT_NE(secondMessage.find("cudnnBackendFinalize"), std::string::npos);
    EXPECT_EQ(secondMessage.find("cudnnBackendCreateDescriptor"), std::string::npos);
    // Neither thread touched this thread's buffer.
    EXPECT_EQ(lastError(), "");
}

TEST_F(TestCudnnShimV8BackendApi, LastErrorClearedAfterRead)
{
    cudnnBackendFinalize(nullptr);
    EXPECT_FALSE(lastError().empty());
    EXPECT_EQ(lastError(), "");
}

TEST_F(TestCudnnShimV8BackendApi, LastErrorTruncatedToCallerBuffer)
{
    cudnnBackendFinalize(nullptr);
    std::array<char, 5> message{};
    message.fill('x');
    cudnnGetLastErrorString(message.data(), message.size());
    EXPECT_STREQ(message.data(), "cudn");
}

TEST_F(TestCudnnShimV8BackendApi, LastErrorFallsBackToBackend)
{
    EXPECT_CALL(*_mockBackend, getLastErrorString(::testing::_, ::testing::_))
        .WillOnce([](char* message, size_t size) {
            const std::string text = "backend failure";
            const size_t length = std::min(text.size(), size - 1);
            std::memcpy(message, text.data(), length);
            message[length] = '\0';
        });
    EXPECT_EQ(lastError(), "backend failure");
}

TEST_F(TestCudnnShimV8BackendApi, ExecuteNotSupported)
{
    EXPECT_EQ(cudnnBackendExecute(_handle, nullptr, nullptr), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("cudnnBackendExecute"), std::string::npos);
}

TEST_F(TestCudnnShimV8BackendApi, BadAllocMapsToAllocFailed)
{
    cudnnBackendDescriptor_t descriptor
        = adopt(std::make_shared<ThrowingDescriptor>([] { throw std::bad_alloc(); }));
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_ALLOC_FAILED);
    EXPECT_EQ(lastError(), "cudnnBackendFinalize: allocation failed");
}

TEST_F(TestCudnnShimV8BackendApi, UnexpectedExceptionMapsToInternalError)
{
    cudnnBackendDescriptor_t descriptor = adopt(
        std::make_shared<ThrowingDescriptor>([] { throw std::logic_error("broken invariant"); }));
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(lastError(), "cudnnBackendFinalize: broken invariant");
}

} // namespace
