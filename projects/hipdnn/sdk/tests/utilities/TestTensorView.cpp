// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/TensorView.hpp>
#include <numeric>

using namespace hipdnn_sdk::utilities;

// ============================================================================
// Basic Typed Iteration Tests
// ============================================================================

TEST(TestTensorView, BasicIteration)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    int count = 0;
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        // No cast needed! Direct typed reference
        float& value = *it;
        EXPECT_EQ(value, 1.0f);
        ++count;
    }

    EXPECT_EQ(count, 6);
}

TEST(TestTensorView, ModifyValues)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    // Modify all values through typed iterator
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        float& value = *it;
        value = 5.0f;
    }

    // Verify modifications
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        EXPECT_EQ(*it, 5.0f);
    }
}

TEST(TestTensorView, RangeBasedForLoop)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(3.14f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    int count = 0;
    // Clean range-based for loop without casts
    for(float& value : view)
    {
        EXPECT_FLOAT_EQ(value, 3.14f);
        value = 2.71f;
        ++count;
    }

    EXPECT_EQ(count, 4);

    // Verify modifications
    for(const float& value : view)
    {
        EXPECT_FLOAT_EQ(value, 2.71f);
    }
}

// ============================================================================
// Const Correctness Tests
// ============================================================================

TEST(TestTensorView, ConstIteration)
{
    Tensor<double> tensor({2, 2});
    tensor.fillWithValue(3.14);

    const ITensor* iTensor = &tensor;
    ConstTensorView<double> view(*iTensor);

    int count = 0;
    for(auto it = view.cbegin(); it != view.cend(); ++it)
    {
        const double& value = *it;
        EXPECT_DOUBLE_EQ(value, 3.14);
        ++count;
    }

    EXPECT_EQ(count, 4);
}

TEST(TestTensorView, ConstViewFromConstTensor)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.5f);

    const ITensor& iTensor = tensor;
    ConstTensorView<float> view(iTensor);

    // Should be able to read through const view
    for(const float& value : view)
    {
        EXPECT_FLOAT_EQ(value, 1.5f);
    }
}

TEST(TestTensorView, ConstRangeBasedForLoop)
{
    Tensor<int> tensor({3, 3});
    tensor.fillWithValue(42.0f);

    ITensor* iTensor = &tensor;
    ConstTensorView<int> view(*iTensor);

    int count = 0;
    for(const int& value : view)
    {
        EXPECT_EQ(value, 42);
        ++count;
    }

    EXPECT_EQ(count, 9);
}

// ============================================================================
// Iterator Comparison Tests
// ============================================================================

TEST(TestTensorView, EqualityComparison)
{
    Tensor<float> tensor({2, 2});

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    auto it1 = view.begin();
    auto it2 = view.begin();

    EXPECT_TRUE(it1 == it2);
    EXPECT_FALSE(it1 != it2);

    ++it1;
    EXPECT_FALSE(it1 == it2);
    EXPECT_TRUE(it1 != it2);

    ++it2;
    EXPECT_TRUE(it1 == it2);
}

TEST(TestTensorView, EndComparison)
{
    Tensor<float> tensor({2, 2});

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    auto it = view.begin();
    auto end = view.end();

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

TEST(TestTensorView, CopyConstructor)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(2.0f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    auto it1 = view.begin();
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto it2 = it1; // Copy

    float& val1 = *it1;
    float& val2 = *it2;

    EXPECT_EQ(&val1, &val2);
    EXPECT_EQ(val1, 2.0f);
}

TEST(TestTensorView, CopyAssignment)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(3.0f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    auto it1 = view.begin();
    auto it2 = view.end();

    it2 = it1; // Copy assignment

    EXPECT_EQ(it1, it2);
}

TEST(TestTensorView, MoveConstructor)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(4.0f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    auto it1 = view.begin();
    auto it2 = std::move(it1); // Move

    float& val = *it2;
    EXPECT_EQ(val, 4.0f);
}

TEST(TestTensorView, ViewCopy)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(1.0f);

    ITensor* iTensor = &tensor;
    TensorView<float> view1(*iTensor);
    TensorView<float> view2(view1); // Copy view

    // Both views should iterate the same tensor
    auto it1 = view1.begin();
    auto it2 = view2.begin();

    EXPECT_EQ(*it1, *it2);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST(TestTensorView, AutoTypeDeduction)
{
    Tensor<float> tensor({2, 2});
    tensor.fillWithValue(1.5f);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    // auto should correctly deduce float&
    for(auto& value : view)
    {
        static_assert(std::is_same_v<decltype(value), float&>, "Type should be float&");
    }
}

// ============================================================================
// Different Data Types
// ============================================================================

TEST(TestTensorViewDouble, BasicIteration)
{
    Tensor<double> tensor({3, 3});
    tensor.fillWithValue(2.718);

    ITensor* iTensor = &tensor;
    TensorView<double> view(*iTensor);

    for(double& value : view)
    {
        EXPECT_DOUBLE_EQ(value, 2.718);
    }
}

TEST(TestTensorViewInt, BasicIteration)
{
    Tensor<int> tensor({4, 4});
    tensor.fillWithValue(7.0f);

    ITensor* iTensor = &tensor;
    TensorView<int> view(*iTensor);

    for(int& value : view)
    {
        EXPECT_EQ(value, 7);
    }
}

// ============================================================================
// Strided Tensor Tests
// ============================================================================

TEST(TestTensorView, StridedTensor)
{
    std::vector<int64_t> dims = {2, 2};
    std::vector<int64_t> strides = {3, 1}; // Non-standard strides

    Tensor<float> tensor(dims, strides);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    // Set values using view
    int counter = 0;
    for(float& value : view)
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

TEST(TestTensorView, TwoDimensionalTensor)
{
    Tensor<float> tensor({3, 4});

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    // Fill with sequence
    int counter = 0;
    for(float& value : view)
    {
        value = static_cast<float>(counter++);
    }

    EXPECT_EQ(counter, 12);

    // Verify a few values
    EXPECT_EQ(tensor.getHostValue(0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(2, 3), 11.0f);
}

TEST(TestTensorView, ThreeDimensionalTensor)
{
    Tensor<float> tensor({2, 3, 4});

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    // Fill with index-based values
    int counter = 0;
    for(float& value : view)
    {
        value = static_cast<float>(counter++);
    }

    EXPECT_EQ(counter, 24);

    // Verify first and last elements
    EXPECT_EQ(tensor.getHostValue(0, 0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(1, 2, 3), 23.0f);
}

TEST(TestTensorView, FourDimensionalTensor)
{
    Tensor<int> tensor({2, 2, 2, 2});

    ITensor* iTensor = &tensor;
    TensorView<int> view(*iTensor);

    int count = 0;
    for(int& value : view)
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

TEST(TestTensorView, EmptyTensor)
{
    std::vector<int64_t> dims = {}; // Zero dimensions
    Tensor<float> tensor(dims);

    ITensor* iTensor = &tensor;
    TensorView<float> view(*iTensor);

    auto it = view.begin();

    EXPECT_EQ(it, view.end());
    EXPECT_THROW(it++, std::out_of_range);
}

TEST(TestTensorView, SingleElement)
{
    Tensor<int> tensor({1});
    tensor.setHostValue(42, 0);

    TensorView<int> view(tensor);

    int count = 0;
    for(int& value : view)
    {
        EXPECT_EQ(value, 42);
        ++count;
    }

    EXPECT_EQ(count, 1);
}

// ============================================================================
// Interoperability Tests
// ============================================================================

TEST(TestTensorView, FromITensorReference)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(1.0f);

    ITensor& iTensor = tensor;
    TensorView<float> view(iTensor);

    for(float& value : view)
    {
        EXPECT_EQ(value, 1.0f);
    }
}

TEST(TestTensorView, FromConstITensorReference)
{
    Tensor<float> tensor({2, 3});
    tensor.fillWithValue(2.0f);

    const ITensor& iTensor = tensor;
    ConstTensorView<float> view(iTensor);

    for(const float& value : view)
    {
        EXPECT_EQ(value, 2.0f);
    }
}

TEST(TestTensorView, ViewAndDirectAccess)
{
    Tensor<float> tensor({2, 2});

    TensorView<float> view(tensor);
    // Set values through view
    int counter = 0;
    for(float& value : view)
    {
        value = static_cast<float>(counter++);
    }

    // Verify through direct tensor access
    EXPECT_EQ(tensor.getHostValue(0, 0), 0.0f);
    EXPECT_EQ(tensor.getHostValue(0, 1), 1.0f);
    EXPECT_EQ(tensor.getHostValue(1, 0), 2.0f);
    EXPECT_EQ(tensor.getHostValue(1, 1), 3.0f);
}

// ============================================================================
// STL Algorithm Compatibility
// ============================================================================

TEST(TestTensorView, StdCount)
{
    Tensor<int> tensor({5});
    TensorView<int> view(tensor);

    // Fill with values
    int counter = 0;
    for(int& value : view)
    {
        value = (counter++ % 2 == 0) ? 1 : 2;
    }

    // Count occurrences of 1
    long count = std::count(view.begin(), view.end(), 1);
    EXPECT_EQ(count, 3); // Indices 0, 2, 4
}

TEST(TestTensorView, StdAccumulate)
{
    Tensor<int> tensor({5});
    TensorView<int> view(tensor);

    // Fill with sequence 1, 2, 3, 4, 5
    std::iota(view.begin(), view.end(), 1);

    // Sum all values
    int sum = std::accumulate(view.begin(), view.end(), 0);
    EXPECT_EQ(sum, 15); // 1+2+3+4+5
}

TEST(TestTensorView, StdTransform)
{
    Tensor<float> tensor({4});
    TensorView<float> view(tensor);

    // Fill with initial values
    std::iota(view.begin(), view.end(), 1.0f);

    // Double all values
    std::transform(view.begin(), view.end(), view.begin(), [](float val) { return val * 2.0f; });

    // Verify
    int idx = 0;
    for(float& value : view)
    {
        EXPECT_FLOAT_EQ(value, static_cast<float>((idx + 1) * 2));
        ++idx;
    }
}

TEST(TestTensorView, StdForEach)
{
    Tensor<int> tensor({3, 3});
    TensorView<int> view(tensor);

    // Initialize
    std::iota(view.begin(), view.end(), 1);

    // Multiply each by 3
    std::for_each(view.begin(), view.end(), [](int& val) { val *= 3; });

    // Verify
    int expected = 3;
    for(int& value : view)
    {
        EXPECT_EQ(value, expected);
        expected += 3;
    }
}

// ============================================================================
// Prefix vs Postfix Increment
// ============================================================================

TEST(TestTensorView, PrefixIncrement)
{
    Tensor<int> tensor({3});
    tensor.setHostValue(10, 0);
    tensor.setHostValue(20, 1);
    tensor.setHostValue(30, 2);
    TensorView<int> view(tensor);

    auto it = view.begin();
    auto it2 = ++it; // Prefix increment

    // Both should point to same element
    int& val1 = *it;
    int& val2 = *it2;
    EXPECT_EQ(&val1, &val2);
    EXPECT_EQ(val1, 20);
}

TEST(TestTensorView, PostfixIncrement)
{
    Tensor<int> tensor({3});
    tensor.setHostValue(10, 0);
    tensor.setHostValue(20, 1);
    tensor.setHostValue(30, 2);
    TensorView<int> view(tensor);

    auto it = view.begin();
    auto it2 = it++; // Postfix increment

    // it2 should point to old position
    int& val2 = *it2;
    EXPECT_EQ(val2, 10);

    // it should point to new position
    int& val1 = *it;
    EXPECT_EQ(val1, 20);
}
