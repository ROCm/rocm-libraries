/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include "framework/backend_param.hpp"

using namespace rpptest;

class HandleTest : public ::testing::TestWithParam<RppBackend> {};

TEST_P(HandleTest, CreateDestroy) {
    rppHandle_t handle = nullptr;
    ASSERT_EQ(rppCreate(&handle, 4, 0, nullptr, GetParam()), rppStatusSuccess);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(rppDestroy(handle, GetParam()), rppStatusSuccess);
}

TEST_P(HandleTest, BatchSizeRoundTrip) {
    rppHandle_t handle = nullptr;
    ASSERT_EQ(rppCreate(&handle, 4, 0, nullptr, GetParam()), rppStatusSuccess);

    EXPECT_EQ(rppSetBatchSize(handle, 16), rppStatusSuccess);
    size_t batchSize = 0;
    EXPECT_EQ(rppGetBatchSize(handle, &batchSize), rppStatusSuccess);
    EXPECT_EQ(batchSize, 16u);

    EXPECT_EQ(rppDestroy(handle, GetParam()), rppStatusSuccess);
}

INSTANTIATE_TEST_SUITE_P(, HandleTest, ::testing::ValuesIn(available_backends()),
                         [](const ::testing::TestParamInfo<RppBackend>& info) {
                             return backend_name(info.param);
                         });
