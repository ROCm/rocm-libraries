/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "gtest_base.hpp"
#include "gtest_common.hpp"

struct TernaryTensorOpsTestCase
{
    std::vector<size_t> tensorlens_ac;
    std::vector<size_t> tensorlens_b;
    std::vector<int64_t> offsets;
    std::vector<size_t> stride_a;
    std::vector<size_t> stride_b;
    std::vector<size_t> stride_c;
    std::vector<float> alphabeta;
    bool packed;
    miopenTensorOp_t operation;

    // friend std::ostream& operator<<(std::ostream& os, const TernaryTensorOpsTestCase& tc) {
    //     return os << "AC lens:" << tensor
    // }
};

template <typename T>
struct TensorOpsCommon : public GTESTBase, public testing::TestWithParam<TernaryTensorOpsTestCase>
{
};