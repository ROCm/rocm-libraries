/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Host-only unit tests for the TensorDesc RAII wrapper (miopen_utils, Layer 3).
// Exercises construction, query round-trips, copy/move/RAII semantics, C-API
// interop, and the MIOPEN_BUILD_TESTING internal-type bridge. No GPU compute.

#include <gtest/gtest.h>

#include <miopen_utils/tensor_desc.hpp>

#include <vector>

TEST(CPU_TensorDesc_NONE, ConstructFromLengths)
{
    const std::vector<size_t> lens{2, 3, 4, 5};
    TensorDesc td(miopenFloat, lens);
    EXPECT_EQ(td.GetNumDims(), 4);
    EXPECT_EQ(td.GetType(), miopenFloat);
    EXPECT_EQ(td.GetLengths(), lens);
    // Packed NCHW strides: {60, 20, 5, 1}
    const std::vector<size_t> expected_strides{60, 20, 5, 1};
    EXPECT_EQ(td.GetStrides(), expected_strides);
    EXPECT_TRUE(td.IsPacked());
    EXPECT_EQ(td.GetElementSize(), 2u * 3u * 4u * 5u);
}

TEST(CPU_TensorDesc_NONE, ConstructWithExplicitStrides)
{
    const std::vector<size_t> lens{2, 3, 4, 5};
    const std::vector<size_t> strides{60, 20, 5, 1};
    TensorDesc td(miopenFloat, lens, strides);
    EXPECT_EQ(td.GetLengths(), lens);
    EXPECT_EQ(td.GetStrides(), strides);
}

TEST(CPU_TensorDesc_NONE, ConstructWithLayout)
{
    const std::vector<size_t> lens{2, 3, 4, 5};
    TensorDesc td(miopenFloat, miopenTensorNCHW, lens);
    EXPECT_EQ(td.GetLayout(), miopenTensorNCHW);
    EXPECT_EQ(td.GetLayout_str(), "NCHW");
    EXPECT_EQ(td.GetLengths(), lens);
}

TEST(CPU_TensorDesc_NONE, Convenience4Dand5D)
{
    TensorDesc td4(miopenFloat, size_t{2}, size_t{3}, size_t{4}, size_t{5});
    EXPECT_EQ(td4.GetNumDims(), 4);
    TensorDesc td5(miopenFloat, size_t{2}, size_t{3}, size_t{4}, size_t{5}, size_t{6});
    EXPECT_EQ(td5.GetNumDims(), 5);
}

TEST(CPU_TensorDesc_NONE, CopySemantics)
{
    TensorDesc a(miopenFloat, std::vector<size_t>{2, 3, 4, 5});
    TensorDesc b(a); // copy-ctor
    EXPECT_EQ(a, b);
    EXPECT_NE(a.get(), b.get()); // independent handles

    TensorDesc c(miopenFloat, std::vector<size_t>{1, 1, 1, 1});
    c = a; // copy-assign
    EXPECT_EQ(a, c);
    EXPECT_NE(a.get(), c.get());
}

TEST(CPU_TensorDesc_NONE, MoveSemantics)
{
    TensorDesc a(miopenFloat, std::vector<size_t>{2, 3, 4, 5});
    auto* raw = a.get();
    TensorDesc b(std::move(a)); // move-ctor
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(a.get(), nullptr); // NOLINT(bugprone-use-after-move) — asserting moved-from state
}

TEST(CPU_TensorDesc_NONE, DeepCopyFromHandle)
{
    TensorDesc src(miopenFloat, std::vector<size_t>{2, 3, 4, 5});
    TensorDesc dst(src.get()); // explicit deep copy from raw handle
    EXPECT_EQ(src, dst);
    EXPECT_NE(src.get(), dst.get());
}

TEST(CPU_TensorDesc_NONE, Reshaped5Dto4D)
{
    TensorDesc td(miopenFloat, miopenTensorNCDHW, std::vector<size_t>{2, 3, 4, 5, 6});
    auto r = td.Reshaped5Dto4D();
    EXPECT_EQ(r.GetNumDims(), 4);
    // [N,C,D,H,W] -> [N,C,D*H,W] = [2,3,20,6]
    const std::vector<size_t> expected{2, 3, 20, 6};
    EXPECT_EQ(r.GetLengths(), expected);
    EXPECT_EQ(r.GetLayout(), miopenTensorNCHW);
}

TEST(CPU_TensorDesc_NONE, GetNCDHW)
{
    const std::vector<int> lens4{2, 3, 4, 5};
    auto [n2, c2, d2, h2, w2] = TensorDesc::GetNCDHW(2, lens4);
    EXPECT_EQ(n2, 2);
    EXPECT_EQ(c2, 3);
    EXPECT_EQ(d2, 1); // D defaults to 1 for 2D spatial
    EXPECT_EQ(h2, 4);
    EXPECT_EQ(w2, 5);

    const std::vector<int> lens5{2, 3, 4, 5, 6};
    auto [n3, c3, d3, h3, w3] = TensorDesc::GetNCDHW(3, lens5);
    EXPECT_EQ(d3, 4);
    EXPECT_EQ(w3, 6);
}

TEST(CPU_TensorDesc_NONE, Interop)
{
    TensorDesc td(miopenFloat, std::vector<size_t>{2, 3, 4, 5});
    // Implicit conversion to the opaque C handle usable by the public API.
    miopenTensorDescriptor_t h = td;
    int ndim                   = 0;
    EXPECT_EQ(miopenGetTensorDescriptorSize(h, &ndim), miopenStatusSuccess);
    EXPECT_EQ(ndim, 4);
}

TEST(CPU_TensorDesc_NONE, GetInnerExpandedTv)
{
    TensorDesc td(miopenFloat, std::vector<size_t>{2, 3, 4, 5});
    auto tv = GetInnerExpandedTv<4>(td.get());
    EXPECT_EQ(tv.size[0], 2u);
    EXPECT_EQ(tv.size[3], 5u);
    EXPECT_EQ(tv.stride[3], 1u);
}

#ifdef MIOPEN_BUILD_TESTING
TEST(CPU_TensorDesc_NONE, InternalTypeBridge)
{
    // Test builds define MIOPEN_BUILD_TESTING, enabling the conversion to the
    // internal miopen::TensorDescriptor. Verify it exposes matching lengths.
    TensorDesc td(miopenFloat, std::vector<size_t>{2, 3, 4, 5});
    const miopen::TensorDescriptor& internal = td;
    EXPECT_EQ(internal.GetLengths().size(), 4u);
    EXPECT_EQ(internal.GetLengths()[0], 2u);
}
#endif
