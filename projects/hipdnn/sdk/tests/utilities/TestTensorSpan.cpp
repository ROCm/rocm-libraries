// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/TensorSpan.hpp>
#include <numeric>

using namespace hipdnn_sdk::utilities;

// ============================================================================
// Basic Typed Iteration Tests
// ============================================================================

TEST(TestTensorSpan, BasicIteration)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    int count = 0;
    for(auto it = span.begin(); it != span.end(); ++it)
    {
        // No cast needed! Direct typed reference
        float& value = *it;
        EXPECT_EQ(value, 1.0f);
        ++count;
    }

    EXPECT_EQ(count, 6);
}

TEST(TestTensorSpan, ModifyValues)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    // Modify all values through typed iterator
    for(auto it = span.begin(); it != span.end(); ++it)
    {
        float& value = *it;
        value = 5.0f;
    }

    // Verify modifications
    for(auto it = span.begin(); it != span.end(); ++it)
    {
        EXPECT_EQ(*it, 5.0f);
    }
}

TEST(TestTensorSpan, RangeBasedForLoop)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(3.14f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    int count = 0;
    // Clean range-based for loop without casts
    for(float& value : span)
    {
        EXPECT_FLOAT_EQ(value, 3.14f);
        value = 2.71f;
        ++count;
    }

    EXPECT_EQ(count, 4);

    // Verify modifications
    for(const float& value : span)
    {
        EXPECT_FLOAT_EQ(value, 2.71f);
    }
}

// ============================================================================
// Const Correctness Tests
// ============================================================================

TEST(TestTensorSpan, ConstIteration)
{
    Tensor<double> tensor({2, 2});
    tensor.fillWithValue(3.14);

    const ITensor* iTensor = &tensor;
    TensorSpan<double> span(*iTensor);

    int count = 0;
    for(auto it = span.begin(); it != span.end(); ++it)
    {
        const double& value = *it;
        EXPECT_DOUBLE_EQ(value, 3.14);
        ++count;
    }

    EXPECT_EQ(count, 4);
}

TEST(TestTensorSpan, ConstSpanFromConstTensor)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.5f);

    const ITensor& iTensor = tensor;
    TensorSpan<float> span(iTensor);

    // Should be able to read through const span
    for(const float& value : span)
    {
        EXPECT_FLOAT_EQ(value, 1.5f);
    }
}

TEST(TestTensorSpan, ConstRangeBasedForLoop)
{
    Tensor<int> tensor({3, 3});
    tensor.fillWithValue(42.0f);

    ITensor* iTensor = &tensor;
    const TensorSpan<int> span(*iTensor);

    int count = 0;
    for(const int& value : span)
    {
        EXPECT_EQ(value, 42);
        ++count;
    }

    EXPECT_EQ(count, 9);
}

// ============================================================================
// Iterator Comparison Tests
// ============================================================================

TEST(TestTensorSpan, EqualityComparison)
{
    Tensor<float> tensor({2, 2});

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    auto it1 = span.begin();
    auto it2 = span.begin();

    EXPECT_TRUE(it1 == it2);
    EXPECT_FALSE(it1 != it2);

    ++it1;
    EXPECT_FALSE(it1 == it2);
    EXPECT_TRUE(it1 != it2);

    ++it2;
    EXPECT_TRUE(it1 == it2);
}

TEST(TestTensorSpan, EndComparison)
{
    Tensor<float> tensor({2, 2});

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    auto it = span.begin();
    auto end = span.end();

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

TEST(TestTensorSpan, CopyConstructor)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(2.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    auto it1 = span.begin();
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto it2 = it1; // Copy

    float& val1 = *it1;
    float& val2 = *it2;

    EXPECT_EQ(&val1, &val2);
    EXPECT_EQ(val1, 2.0f);
}

TEST(TestTensorSpan, CopyAssignment)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(3.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    auto it1 = span.begin();
    auto it2 = span.end();

    it2 = it1; // Copy assignment

    EXPECT_EQ(it1, it2);
}

TEST(TestTensorSpan, MoveConstructor)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(4.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    auto it1 = span.begin();
    auto it2 = std::move(it1); // Move

    float& val = *it2;
    EXPECT_EQ(val, 4.0f);
}

TEST(TestTensorSpan, SpanCopy)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span1(*iTensor);
    TensorSpan<float> span2(span1); // Copy span

    // Both spans should iterate the same tensor
    auto it1 = span1.begin();
    auto it2 = span2.begin();

    EXPECT_EQ(*it1, *it2);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST(TestTensorSpan, AutoTypeDeduction)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(1.5f);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    // auto should correctly deduce float&
    for(auto& value : span)
    {
        static_assert(std::is_same_v<decltype(value), float&>, "Type should be float&");
    }
}

// ============================================================================
// Different Data Types
// ============================================================================

TEST(TestTensorSpanDouble, BasicIteration)
{
    Tensor<double> tensor({3, 3});
    tensor.fillWithValue(2.718);

    ITensor* iTensor = &tensor;
    TensorSpan<double> span(*iTensor);

    for(double& value : span)
    {
        EXPECT_DOUBLE_EQ(value, 2.718);
    }
}

TEST(TestTensorSpanInt, BasicIteration)
{
    Tensor<int> tensor({4, 4});
    tensor.fillWithValue(7.0f);

    ITensor* iTensor = &tensor;
    TensorSpan<int> span(*iTensor);

    for(int& value : span)
    {
        EXPECT_EQ(value, 7);
    }
}

// ============================================================================
// Strided Tensor Tests
// ============================================================================

TEST(TestTensorSpan, StridedTensor)
{
    std::vector<int64_t> dims = {2, 2};
    std::vector<int64_t> strides = {3, 1}; // Non-standard strides

    Tensor<float> tensor(dims, strides);

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    // Set values using span
    int counter = 0;
    for(float& value : span)
    {
        value = static_cast<float>(counter++);
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

TEST(TestTensorSpan, TwoDimensionalTensor)
{
    Tensor<float> tensor({3, 4});

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    // Fill with sequence
    int counter = 0;
    for(float& value : span)
    {
        value = static_cast<float>(counter++);
    }

    EXPECT_EQ(counter, 12);

    // Verify a few values
    EXPECT_EQ(tensor.getHostValue(0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(2, 3), 11.0f);
}

TEST(TestTensorSpan, ThreeDimensionalTensor)
{
    Tensor<float> tensor({2, 3, 4});

    ITensor* iTensor = &tensor;
    TensorSpan<float> span(*iTensor);

    // Fill with index-based values
    int counter = 0;
    for(float& value : span)
    {
        value = static_cast<float>(counter++);
    }

    EXPECT_EQ(counter, 24);

    // Verify first and last elements
    EXPECT_EQ(tensor.getHostValue(0, 0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(1, 2, 3), 23.0f);
}

TEST(TestTensorSpan, FourDimensionalTensor)
{
    Tensor<int> tensor({2, 2, 2, 2});

    ITensor* iTensor = &tensor;
    TensorSpan<int> span(*iTensor);

    int count = 0;
    for(int& value : span)
    {
        value = count++;
    }

    EXPECT_EQ(count, 16);
    EXPECT_EQ(tensor.getHostValue(0, 0, 0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(1, 1, 1, 1), 15.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TestTensorSpan, SingleElement)
{
    Tensor<int> tensor({1});
    tensor.setHostValue(42, 0);

    TensorSpan<int> span(tensor);

    int count = 0;
    for(int& value : span)
    {
        EXPECT_EQ(value, 42);
        ++count;
    }

    EXPECT_EQ(count, 1);
}

// ============================================================================
// Interoperability Tests
// ============================================================================

TEST(TestTensorSpan, FromITensorReference)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);

    ITensor& iTensor = tensor;
    TensorSpan<float> span(iTensor);

    for(float& value : span)
    {
        EXPECT_EQ(value, 1.0f);
    }
}

TEST(TestTensorSpan, FromConstITensorReference)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(2.0f);

    const ITensor& iTensor = tensor;
    TensorSpan<float> span(iTensor);

    for(const float& value : span)
    {
        EXPECT_EQ(value, 2.0f);
    }
}

TEST(TestTensorSpan, SpanAndDirectAccess)
{
    Tensor<float> tensor({2, 2});

    TensorSpan<float> span(tensor);
    // Set values through span
    int counter = 0;
    for(float& value : span)
    {
        value = static_cast<float>(counter++);
    }

    // Verify through direct tensor access
    EXPECT_EQ(tensor.getHostValue(0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(0, 1), 1.0f);
    EXPECT_EQ(tensor.getHostValue(1, 0), 2.0f);
    EXPECT_EQ(tensor.getHostValue(1, 1), 3.0f);
}

// // ============================================================================
// // STL Algorithm Compatibility
// // ============================================================================

TEST(TestTensorSpan, StdCount)
{
    Tensor<int> tensor({5});
    TensorSpan<int> span(tensor);

    // Fill with values
    int counter = 0;
    for(int& value : span)
    {
        value = (counter++ % 2 == 0) ? 1 : 2;
    }

    // Count occurrences of 1
    long count = std::count(span.begin(), span.end(), 1);
    EXPECT_EQ(count, 3); // Indices 0, 2, 4
}

TEST(TestTensorSpan, StdAccumulate)
{
    Tensor<int> tensor({5});
    TensorSpan<int> span(tensor);

    // Fill with sequence 1, 2, 3, 4, 5
    std::iota(span.begin(), span.end(), 1);

    // Sum all values
    int sum = std::accumulate(span.begin(), span.end(), 0);
    EXPECT_EQ(sum, 15); // 1+2+3+4+5
}

TEST(TestTensorSpan, StdTransform)
{
    Tensor<float> tensor({4});
    TensorSpan<float> span(tensor);

    // Fill with initial values
    std::iota(span.begin(), span.end(), 1.0f);

    // Double all values
    std::transform(span.begin(), span.end(), span.begin(), [](float val) { return val * 2.0f; });

    // Verify
    int idx = 0;
    for(float& value : span)
    {
        EXPECT_FLOAT_EQ(value, static_cast<float>((idx + 1) * 2));
        ++idx;
    }
}

TEST(TestTensorSpan, StdForEach)
{
    Tensor<int> tensor({3, 3});
    TensorSpan<int> span(tensor);

    // Initialize
    std::iota(span.begin(), span.end(), 1);

    // Multiply each by 3
    std::for_each(span.begin(), span.end(), [](int& val) { val *= 3; });

    // Verify
    int expected = 3;
    for(int& value : span)
    {
        EXPECT_EQ(value, expected);
        expected += 3;
    }
}

// ============================================================================
// Prefix vs Postfix Increment
// ============================================================================

TEST(TestTensorSpan, PrefixIncrement)
{
    Tensor<int> tensor({3});
    tensor.setHostValue(10, 0);
    tensor.setHostValue(20, 1);
    tensor.setHostValue(30, 2);
    TensorSpan<int> span(tensor);

    auto it = span.begin();
    auto it2 = ++it; // Prefix increment

    // Both should point to same element
    int& val1 = *it;
    int& val2 = *it2;
    EXPECT_EQ(&val1, &val2);
    EXPECT_EQ(val1, 20);
}

TEST(TestTensorSpan, PostfixIncrement)
{
    Tensor<int> tensor({3});
    tensor.setHostValue(10, 0);
    tensor.setHostValue(20, 1);
    tensor.setHostValue(30, 2);
    TensorSpan<int> span(tensor);

    auto it = span.begin();
    auto it2 = it++; // Postfix increment

    // it2 should point to old position
    int& val2 = *it2;
    EXPECT_EQ(val2, 10);

    // it should point to new position
    int& val1 = *it;
    EXPECT_EQ(val1, 20);
}
