// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "TestRaggedTensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <memory>
#include <numeric>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_ragged_test;

namespace
{

// Helper to process ITensor polymorphically
template <typename T>
void processITensorErased(ITensor& tensor, const std::function<void(T&)>& func)
{
    for(auto it = tensor.begin(); it != tensor.end(); ++it)
    {
        T* value = static_cast<T*>(*it);
        func(*value);
    }
}

} // namespace

// ============================================================================
// Basic Type-Erased Iterator Tests
// ============================================================================

TEST(TestTypeErasedIterator, BasicIteration)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);
    ITensor* iTensor = &tensor;

    // Count elements using type-erased iterator
    int count = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        auto* value = static_cast<float*>(*it);
        EXPECT_EQ(*value, 1.0f);
        ++count;
    }

    EXPECT_EQ(count, 6);
}

TEST(TestTypeErasedIterator, ModifyValues)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);
    ITensor* iTensor = &tensor;

    // Modify all values through type-erased iterator
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        auto* value = static_cast<float*>(*it);
        *value = 5.0f;
    }

    for(const auto& val : *iTensor)
    {
        EXPECT_EQ(*static_cast<float*>(val), 5.0f);
    }
}

TEST(TestTypeErasedIterator, IncrementPattern)
{
    Tensor<int> tensor({3, 3});
    ITensor* iTensor = &tensor;

    // Set incremental values
    int counter = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        int* value = static_cast<int*>(*it);
        *value = counter++;
    }

    // Verify pattern
    counter = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        int* value = static_cast<int*>(*it);
        EXPECT_EQ(*value, counter++);
    }

    EXPECT_EQ(counter, 9);
}

// ============================================================================
// Const Iterator Tests
// ============================================================================

TEST(TestTypeErasedIterator, ConstIteration)
{
    Tensor<double> tensor({2, 2});
    tensor.fillWithValue(3.14);

    const ITensor* iTensor = &tensor;

    int count = 0;
    for(auto it = iTensor->cbegin(); it != iTensor->cend(); ++it)
    {
        const auto* value = static_cast<const double*>(*it);
        EXPECT_DOUBLE_EQ(*value, 3.14);
        ++count;
    }

    EXPECT_EQ(count, 4);
}

// ============================================================================
// Iterator Comparison Tests
// ============================================================================

TEST(TestTypeErasedIterator, EqualityComparison)
{
    Tensor<float> tensor({2, 2});

    ITensor* iTensor = &tensor;

    auto it1 = iTensor->begin();
    auto it2 = iTensor->begin();

    EXPECT_TRUE(it1 == it2);
    EXPECT_FALSE(it1 != it2);

    ++it1;
    EXPECT_FALSE(it1 == it2);
    EXPECT_TRUE(it1 != it2);

    ++it2;
    EXPECT_TRUE(it1 == it2);
}

TEST(TestTypeErasedIterator, EndComparison)
{
    Tensor<float> tensor({2, 2});

    ITensor* iTensor = &tensor;

    auto it = iTensor->begin();
    auto end = iTensor->end();

    EXPECT_NE(it, end);

    // Advance to end
    for(int i = 0; i < 4; ++i)
    {
        ++it;
    }

    EXPECT_EQ(it, end);
}

// ============================================================================
// Copy and Move Semantics Tests
// ============================================================================

TEST(TestTypeErasedIterator, CopyConstructor)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(2.0f);

    ITensor* iTensor = &tensor;

    auto it1 = iTensor->begin();
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto it2 = it1; // Copy

    auto* val1 = static_cast<float*>(*it1);
    auto* val2 = static_cast<float*>(*it2);

    EXPECT_EQ(val1, val2);
    EXPECT_EQ(*val1, 2.0f);
}

TEST(TestTypeErasedIterator, CopyAssignment)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(3.0f);

    ITensor* iTensor = &tensor;

    auto it1 = iTensor->begin();
    auto it2 = iTensor->end();
    EXPECT_NE(it1, it2);

    it2 = it1; // Copy assignment

    EXPECT_EQ(it1, it2);
}

TEST(TestTypeErasedIterator, MoveConstructor)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(4.0f);

    ITensor* iTensor = &tensor;

    auto it1 = iTensor->begin();
    auto it2 = std::move(it1); // Move

    auto* val = static_cast<float*>(*it2);
    EXPECT_EQ(*val, 4.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TestTypeErasedIterator, EmptyTensor)
{
    const std::vector<int64_t> dims;
    Tensor<float> tensor(dims);
    ITensor* iTensor = &tensor;

    auto it = iTensor->begin();

    EXPECT_EQ(it, iTensor->end());
    EXPECT_THROW(it++, std::out_of_range);
}

TEST(TestTensor, ThrowsWhenIncrementingPastEnd)
{
    Tensor<float> tensor({2, 2});
    auto it = tensor.begin();
    auto endIt = tensor.end();

    // Increment to the end
    ++it; // (0,1)
    ++it; // (1,0)
    ++it; // (1,1)
    ++it; // Now at 1 past end

    EXPECT_EQ(it, endIt);

    // Incrementing past the end should throw
    EXPECT_THROW(++it, std::out_of_range);
}

TEST(TestTensor, ThrowsWhenAccessingEndIterator)
{
    Tensor<float> tensor({2, 2});
    auto it = tensor.end();

    // Dereferencing the end iterator should throw
    EXPECT_THROW(*it, std::out_of_range);
}

TEST(TestTypeErasedIterator, SingleElement)
{
    Tensor<int> tensor({1});
    tensor.setHostValue(42, 0);

    ITensor* iTensor = &tensor;

    int count = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        int* value = static_cast<int*>(*it);
        EXPECT_EQ(*value, 42);
        ++count;
    }

    EXPECT_EQ(count, 1);
}

TEST(TestTypeErasedIterator, LargeTensor)
{
    Tensor<float> tensor({10, 10, 10});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;

    int count = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        ++count;
    }

    EXPECT_EQ(count, 1000);
}

// ============================================================================
// Different Data Types
// ============================================================================

TEST(TestTypeErasedIteratorDouble, BasicIteration)
{
    Tensor<double> tensor({3, 3});
    tensor.fillWithValue(2.718);

    ITensor* iTensor = &tensor;

    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        auto* value = static_cast<double*>(*it);
        EXPECT_DOUBLE_EQ(*value, 2.718);
    }
}

TEST(TestTypeErasedIteratorInt, BasicIteration)
{
    Tensor<int> tensor({4, 4});
    tensor.fillWithValue(7.0f);

    ITensor* iTensor = &tensor;

    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        int* value = static_cast<int*>(*it);
        EXPECT_EQ(*value, 7);
    }
}

// ============================================================================
// Strided Tensor Tests
// ============================================================================

TEST(TestTypeErasedIterator, StridedTensor)
{
    const std::vector<int64_t> dims = {2, 2};
    const std::vector<int64_t> strides = {3, 1}; // Non-standard strides

    Tensor<float> tensor(dims, strides);

    ITensor* iTensor = &tensor;

    // Set values using type-erased iterator
    int counter = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        auto* value = static_cast<float*>(*it);
        *value = static_cast<float>(counter++);
    }

    // Verify using indices
    EXPECT_EQ(tensor.getHostValue(0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(0, 1), 1.0f);
    EXPECT_EQ(tensor.getHostValue(1, 0), 2.0f);
    EXPECT_EQ(tensor.getHostValue(1, 1), 3.0f);
}

// ============================================================================
// Multi-Dimensional Tests
// ============================================================================

TEST(TestTypeErasedIterator, ThreeDimensionalTensor)
{
    Tensor<float> tensor({2, 3, 4});

    ITensor* iTensor = &tensor;

    // Fill with index-based values
    int counter = 0;
    for(auto it = iTensor->begin(); it != iTensor->end(); ++it)
    {
        auto* value = static_cast<float*>(*it);
        *value = static_cast<float>(counter++);
    }

    EXPECT_EQ(counter, 24);

    // Verify first and last elements
    auto begin = iTensor->begin();
    auto* first = static_cast<float*>(*begin);
    EXPECT_EQ(*first, 0.0f);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(TestTypeErasedIterator, HelperFunction)
{
    Tensor<float> tensor({3, 3});
    tensor.fillWithValue(2.0f);

    ITensor& iTensor = tensor;

    // Use helper to double all values
    processITensorErased<float>(iTensor, [](float& val) { val *= 2.0f; });

    // Verify
    for(const auto& val : tensor)
    {
        EXPECT_EQ(*static_cast<float*>(val), 4.0f);
    }
}

// ============================================================================
// Indices Access Tests
// ============================================================================

TEST(TestTypeErasedIterator, StridedIndicesAccess)
{
    Tensor<float> tensor({2, 3}, {6, 2});

    ITensor* iTensor = &tensor;

    auto it = iTensor->begin();

    // Check initial indices
    auto indices = std::get<ITensorIterator<false>::CompositeIndex>(it.index()).indices;
    EXPECT_EQ(indices.size(), 2);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 0);

    // Advance and check indices
    ++it;
    indices = std::get<ITensorIterator<false>::CompositeIndex>(it.index()).indices;
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);

    ++it;
    indices = std::get<ITensorIterator<false>::CompositeIndex>(it.index()).indices;
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 2);

    ++it;
    indices = std::get<ITensorIterator<false>::CompositeIndex>(it.index()).indices;
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 0);
}

TEST(TestTypeErasedIteratorPacked, LinearIndexAccess)
{
    Tensor<float> tensor({2, 3});

    ITensor* iTensor = &tensor;

    auto it = iTensor->begin();

    // Check initial index
    auto index = std::get<ITensorIterator<false>::LinearIndex>(it.index()).getValue();
    EXPECT_EQ(index, 0);

    // Advance and check index
    ++it;
    index = std::get<ITensorIterator<false>::LinearIndex>(it.index()).getValue();
    EXPECT_EQ(index, 1);

    ++it;
    index = std::get<ITensorIterator<false>::LinearIndex>(it.index()).getValue();
    EXPECT_EQ(index, 2);

    ++it;
    index = std::get<ITensorIterator<false>::LinearIndex>(it.index()).getValue();
    EXPECT_EQ(index, 3);
}

// ============================================================================
// Index-strategy selection: ragged vs dense (RaggedCompositeIndex hook)
// ============================================================================

TEST(TestTypeErasedIteratorRagged, UsesRaggedCompositeIndex)
{
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    RaggedTensor<float> tensor(K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    auto it = tensor.begin();

    // The iterator resolves to a RaggedCompositeIndex (not Linear/Composite).
    auto indices = std::get<ITensorIterator<false>::RaggedCompositeIndex>(it.index()).indices;
    EXPECT_EQ(indices.size(), 4);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 0);
    EXPECT_EQ(indices[2], 0);
    EXPECT_EQ(indices[3], 0);

    // Visits exactly off[B] == 20 elements.
    int count = 0;
    for(auto walk = tensor.begin(); walk != tensor.end(); ++walk)
    {
        ++count;
    }
    EXPECT_EQ(count, 20);
}

TEST(TestTypeErasedIteratorRagged, ShallowUsesRaggedCompositeIndex)
{
    // Same geometry, over a borrowed buffer.
    auto aux = makeOffsetAux<int32_t>(K_OFFSETS);
    std::vector<float> backing(20, 0.0f);
    ShallowRaggedTensor<float> tensor(backing.data(), K_DIMS, K_STRIDES, BSHD_SEQ_AXIS, aux);

    auto it = tensor.begin();

    // The iterator resolves to a RaggedCompositeIndex for the shallow type too.
    EXPECT_NO_THROW(std::ignore
                    = std::get<ITensorIterator<false>::RaggedCompositeIndex>(it.index()));

    int count = 0;
    for(auto walk = tensor.begin(); walk != tensor.end(); ++walk)
    {
        ++count;
    }
    EXPECT_EQ(count, 20);
}

TEST(TestTypeErasedIteratorRagged, DensePackedStillLinear)
{
    Tensor<float> tensor({2, 3});
    auto it = tensor.begin();

    // Dense packed tensors still resolve to LinearIndex.
    EXPECT_NO_THROW(std::ignore = std::get<ITensorIterator<false>::LinearIndex>(it.index()));
}

TEST(TestTypeErasedIteratorRagged, DenseStridedStillComposite)
{
    Tensor<float> tensor({2, 3}, {6, 2});
    auto it = tensor.begin();

    // Dense strided tensors still resolve to CompositeIndex.
    EXPECT_NO_THROW(std::ignore = std::get<ITensorIterator<false>::CompositeIndex>(it.index()));
}

// ============================================================================
// Dense regression guard for the getIndexImpl refactor
// ============================================================================

TEST(TestTypeErasedIteratorRagged, DenseAddressingUnchanged)
{
    const Tensor<float> tensor({2, 3, 4, 5});
    // 1*60 + 2*20 + 3*5 + 4*1 = 119 (unchanged by the getIndexImpl forwarding).
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0), 0);

    const std::vector<int64_t> indices = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices), 119);
}

// ============================================================================
// Prefix vs Postfix Increment
// ============================================================================

TEST(TestTypeErasedIterator, PrefixIncrement)
{
    Tensor<int> tensor({3});
    tensor.setHostValue(10, 0);
    tensor.setHostValue(20, 1);
    tensor.setHostValue(30, 2);

    ITensor* iTensor = &tensor;

    auto it = iTensor->begin();
    auto it2 = ++it; // Prefix increment

    // Both should point to same element
    int* val1 = static_cast<int*>(*it);
    int* val2 = static_cast<int*>(*it2);
    EXPECT_EQ(val1, val2);
    EXPECT_EQ(*val1, 20);
}

TEST(TestTypeErasedIterator, PostfixIncrement)
{
    Tensor<int> tensor({3});
    tensor.setHostValue(10, 0);
    tensor.setHostValue(20, 1);
    tensor.setHostValue(30, 2);

    ITensor* iTensor = &tensor;

    auto it = iTensor->begin();
    auto it2 = it++; // Postfix increment

    // it2 should point to old position
    int* val2 = static_cast<int*>(*it2);
    EXPECT_EQ(*val2, 10);

    // it should point to new position
    int* val1 = static_cast<int*>(*it);
    EXPECT_EQ(*val1, 20);
}
