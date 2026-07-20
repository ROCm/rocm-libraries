#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include "framework/backend_param.hpp"

using namespace rpptest;

// Value-parameterized skeleton over the available backends. Fills in the
// create/destroy lifecycle only; the actual brightness call + golden compare
// are added in a later phase.
class BrightnessTest : public ::testing::TestWithParam<RppBackend> {};

TEST_P(BrightnessTest, HandleLifecycle) {
    RppBackend backend = GetParam();
    rppHandle_t handle = nullptr;
    ASSERT_EQ(rppCreate(&handle, 1, 0, nullptr, backend), rppStatusSuccess);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(rppDestroy(handle, backend), rppStatusSuccess);
}

INSTANTIATE_TEST_SUITE_P(Backends, BrightnessTest, ::testing::ValuesIn(available_backends()),
                         [](const ::testing::TestParamInfo<RppBackend>& info) {
                             return backend_name(info.param);
                         });
