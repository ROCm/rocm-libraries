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

#include <string>

#include "framework/backend_param.hpp"
#include "framework/skip_list.hpp"

using namespace rpptest;

class HandleTest : public SkipListTest<RppBackend> {};

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

// RPP logs the rejection to stderr; capture it so the run output stays clean.
TEST_P(HandleTest, DestroyNullHandle) {
    ::testing::internal::CaptureStderr();
    const rppStatus_t status = rppDestroy(nullptr, GetParam());
    const std::string log = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(status, rppStatusBadParm);
    EXPECT_NE(log.find("Dereferencing nullptr"), std::string::npos) << log;
}

TEST_P(HandleTest, DestroyInvalidBackendKeepsHandle) {
    rppHandle_t handle = nullptr;
    ASSERT_EQ(rppCreate(&handle, 4, 0, nullptr, GetParam()), rppStatusSuccess);

    EXPECT_EQ(rppDestroy(handle, static_cast<RppBackend>(99)), rppStatusNotImplemented);
    size_t batchSize = 0;
    EXPECT_EQ(rppGetBatchSize(handle, &batchSize), rppStatusSuccess);
    EXPECT_EQ(batchSize, 4u);

    EXPECT_EQ(rppDestroy(handle, GetParam()), rppStatusSuccess);
}

// rppDestroy must release the handle by its own backend, not the (possibly defaulted) argument.
TEST_P(HandleTest, DestroyWithOtherBackend) {
    rppHandle_t handle = nullptr;
    ASSERT_EQ(rppCreate(&handle, 4, 0, nullptr, GetParam()), rppStatusSuccess);
    const RppBackend other = GetParam() == RPP_HOST_BACKEND ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;
    EXPECT_EQ(rppDestroy(handle, other), rppStatusSuccess);
}

INSTANTIATE_TEST_SUITE_P(, HandleTest, ::testing::ValuesIn(available_backends()),
                         [](const ::testing::TestParamInfo<RppBackend>& info) {
                             return backend_name(info.param);
                         });
