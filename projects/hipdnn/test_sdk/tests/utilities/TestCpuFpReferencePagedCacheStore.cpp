// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferencePagedCacheStore.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

namespace
{

template <typename Type>
Tensor<Type> createTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

/// Writes `values` in memory order (raw host buffer), whatever the tensor's strides.
template <typename Type>
void setRawValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(tensor.elementSpace(), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = static_cast<Type>(values[i]);
    }
}

template <typename SlotType>
Tensor<SlotType> createSlots(const std::vector<int64_t>& dims, const std::vector<int64_t>& slots)
{
    auto tensor = createTensor<SlotType>(dims);
    EXPECT_EQ(tensor.elementCount(), slots.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < slots.size(); ++i)
    {
        data[i] = static_cast<SlotType>(slots[i]);
    }
    return tensor;
}

template <typename Type>
void expectRawValues(const Tensor<Type>& tensor, const std::vector<float>& expected)
{
    ASSERT_EQ(tensor.elementSpace(), expected.size());
    const auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(static_cast<float>(data[i]), expected[i]) << "Mismatch at flat index " << i;
    }
}

// T=3, H=1, D=2 into num_blocks=2, block_size=2. Slots {3, -1, 0}: token 0 lands in block 1
// offset 1, token 1 is skipped, token 2 lands in block 0 offset 0. Every other element keeps
// the -1 prefill.
template <typename DataT, typename SlotT>
void runTokenPackedStore()
{
    auto x = createTensor<DataT>({3, 1, 2});
    setRawValues(x, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto slots = createSlots<SlotT>({3}, {3, -1, 0});
    auto container = createTensor<DataT>({2, 1, 2, 2});
    container.fillTensorWithValue(-1.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {5.0F, 6.0F, -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, 2.0F});
}

} // namespace

TEST(TestCpuFpReferencePagedCacheStore, TokenPackedSkipsNegativeAndPreservesUntouchedFloat)
{
    runTokenPackedStore<float, int32_t>();
}

TEST(TestCpuFpReferencePagedCacheStore, TokenPackedInt64SlotMapping)
{
    runTokenPackedStore<float, int64_t>();
}

TEST(TestCpuFpReferencePagedCacheStore, TokenPackedHalf)
{
    runTokenPackedStore<half, int32_t>();
}

TEST(TestCpuFpReferencePagedCacheStore, TokenPackedBfloat16)
{
    runTokenPackedStore<bfloat16, int64_t>();
}

// Slots 1 and 2 straddle the block boundary (block_size 2): slot 1 is the last row of block 0,
// slot 2 the first row of block 1.
TEST(TestCpuFpReferencePagedCacheStore, BlockBoundarySlots)
{
    auto x = createTensor<float>({2, 1, 1});
    setRawValues(x, {7.0F, 8.0F});
    auto slots = createSlots<int32_t>({2}, {2, 1});
    auto container = createTensor<float>({2, 1, 2, 1});
    container.fillTensorWithValue(0.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {0.0F, 8.0F, 7.0F, 0.0F});
}

// The last slot of the last block is the highest valid slot.
TEST(TestCpuFpReferencePagedCacheStore, LastSlotOfLastBlock)
{
    auto x = createTensor<float>({1, 1, 1});
    setRawValues(x, {4.0F});
    auto slots = createSlots<int32_t>({1}, {3});
    auto container = createTensor<float>({2, 1, 2, 1});
    container.fillTensorWithValue(0.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {0.0F, 0.0F, 0.0F, 4.0F});
}

// Each head goes to its own head plane of the container: container[0, h, 1, 0].
TEST(TestCpuFpReferencePagedCacheStore, MultipleHeads)
{
    auto x = createTensor<float>({1, 2, 1});
    setRawValues(x, {1.0F, 2.0F});
    auto slots = createSlots<int32_t>({1}, {1});
    auto container = createTensor<float>({1, 2, 2, 1});
    container.fillTensorWithValue(0.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {0.0F, 1.0F, 0.0F, 2.0F});
}

TEST(TestCpuFpReferencePagedCacheStore, ScaleMultipliesStoredValues)
{
    auto x = createTensor<float>({1, 1, 2});
    setRawValues(x, {1.0F, -2.0F});
    auto slots = createSlots<int32_t>({1}, {1});
    auto container = createTensor<float>({1, 1, 2, 2});
    container.fillTensorWithValue(9.0F);
    auto scale = createTensor<float>({1});
    setRawValues(scale, {0.5F});

    CpuFpReferencePagedCacheStore::store(x, slots, container, &scale);

    expectRawValues(container, {9.0F, 9.0F, 0.5F, -1.0F});
}

// Every slot negative (an empty step): the container is left as it was.
TEST(TestCpuFpReferencePagedCacheStore, AllNegativeSlotsWriteNothing)
{
    auto x = createTensor<float>({2, 1, 1});
    setRawValues(x, {1.0F, 2.0F});
    auto slots = createSlots<int32_t>({2}, {-1, -7});
    auto container = createTensor<float>({1, 1, 2, 1});
    container.fillTensorWithValue(3.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {3.0F, 3.0F});
}

// x [B=2, H=2, S=2, D=1] laid out in BSHD memory order (strides {4, 1, 2, 1}), slot_mapping
// [B, S] = {{0, -1}, {3, 1}} into container [2, 2, 2, 1] (flat index blk*4 + h*2 + off).
TEST(TestCpuFpReferencePagedCacheStore, SdpaNativeLayoutWithBshdStrides)
{
    Tensor<float> x({2, 2, 2, 1}, {4, 1, 2, 1});
    // Memory order (b, s, h): b0s0h0, b0s0h1, b0s1h0, b0s1h1, b1s0h0, b1s0h1, b1s1h0, b1s1h1.
    setRawValues(x, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    auto slots = createSlots<int64_t>({2, 2}, {0, -1, 3, 1});
    auto container = createTensor<float>({2, 2, 2, 1});
    container.fillTensorWithValue(0.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {1.0F, 7.0F, 2.0F, 8.0F, 0.0F, 5.0F, 0.0F, 6.0F});
}

// FP8 container: x * scale is formed in float and cast once (values exactly representable).
TEST(TestCpuFpReferencePagedCacheStore, Fp8E4M3Container)
{
    auto x = createTensor<float>({1, 1, 2});
    setRawValues(x, {0.75F, -3.0F});
    auto slots = createSlots<int32_t>({1}, {0});
    auto container = createTensor<fp8_e4m3>({1, 1, 1, 2});
    container.fillTensorWithValue(1.0F);
    auto scale = createTensor<float>({1});
    setRawValues(scale, {2.0F});

    CpuFpReferencePagedCacheStore::store(x, slots, container, &scale);

    expectRawValues(container, {1.5F, -6.0F});
}

TEST(TestCpuFpReferencePagedCacheStore, Fp8E5M2ContainerFromHalf)
{
    auto x = createTensor<half>({1, 1, 2});
    setRawValues(x, {0.5F, 4.0F});
    auto slots = createSlots<int32_t>({1}, {0});
    auto container = createTensor<fp8_e5m2>({1, 1, 1, 2});
    container.fillTensorWithValue(0.0F);

    CpuFpReferencePagedCacheStore::store(x, slots, container);

    expectRawValues(container, {0.5F, 4.0F});
}

TEST(TestCpuFpReferencePagedCacheStore, SlotPastContainerThrows)
{
    auto x = createTensor<float>({1, 1, 1});
    auto slots = createSlots<int32_t>({1}, {4});
    auto container = createTensor<float>({2, 1, 2, 1});

    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x, slots, container), std::out_of_range);
}

TEST(TestCpuFpReferencePagedCacheStore, DuplicateSlotThrows)
{
    auto x = createTensor<float>({2, 1, 1});
    auto slots = createSlots<int32_t>({2}, {1, 1});
    auto container = createTensor<float>({2, 1, 2, 1});

    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x, slots, container), std::invalid_argument);
}

TEST(TestCpuFpReferencePagedCacheStore, ShapeMismatchesThrow)
{
    auto x = createTensor<float>({2, 1, 2});
    auto container = createTensor<float>({2, 1, 2, 2});

    auto wrongLength = createSlots<int32_t>({3}, {0, 1, 2});
    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x, wrongLength, container),
                 std::invalid_argument);

    auto slots = createSlots<int32_t>({2}, {0, 1});
    auto wrongHeadSize = createTensor<float>({2, 1, 2, 3});
    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x, slots, wrongHeadSize),
                 std::invalid_argument);

    auto wrongHeads = createTensor<float>({2, 2, 2, 2});
    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x, slots, wrongHeads), std::invalid_argument);

    auto x4 = createTensor<float>({1, 1, 2, 2});
    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x4, slots, container), std::invalid_argument);
}

TEST(TestCpuFpReferencePagedCacheStore, NonScalarScaleThrows)
{
    auto x = createTensor<float>({1, 1, 2});
    auto slots = createSlots<int32_t>({1}, {0});
    auto container = createTensor<float>({1, 1, 1, 2});
    auto scale = createTensor<float>({2});

    EXPECT_THROW(CpuFpReferencePagedCacheStore::store(x, slots, container, &scale),
                 std::invalid_argument);
}
