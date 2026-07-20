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

INSTANTIATE_TEST_SUITE_P(Backends, HandleTest, ::testing::ValuesIn(available_backends()),
                         [](const ::testing::TestParamInfo<RppBackend>& info) {
                             return backend_name(info.param);
                         });
