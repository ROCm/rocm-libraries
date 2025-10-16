// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if MIOPEN_BACKEND_HIP

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// This test event listener ensures that HIP errors are cleaned up after every test
class HIPErrorHandler : public testing::EmptyTestEventListener
{
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        EXPECT_EQ(hipExtGetLastError(), hipSuccess) << " hipExtGetLastError returned error code after test";
        EXPECT_EQ(hipGetLastError(), hipSuccess) << " hipExtGetLastError returned error code after test";
    }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest( &argc, argv );

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    return RUN_ALL_TESTS();
}

#endif