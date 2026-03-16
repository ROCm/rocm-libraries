// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <string_view>

// This test event listener ensures that HIP errors are cleaned up after every test, and will flag
// tests that don't clean up their own errors
class HIPErrorHandler : public testing::EmptyTestEventListener
{
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        auto hipError    = hipGetLastError();
        auto hipExtError = hipExtGetLastError();

        ASSERT_EQ(hipError, hipSuccess)
            << " hipGetLastError returned error code " << hipError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipError);
        ASSERT_EQ(hipExtError, hipSuccess)
            << " hipExtGetLastError returned error code " << hipExtError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipExtError);
    }
};

int main(int argc, char** argv)
{
    // Child processes spawned by multi-process tests (e.g., perfdb) inherit
    // GTEST_TOTAL_SHARDS/GTEST_SHARD_INDEX from the parent's CI shard.
    // This causes children to run only a subset of their assigned work.
    // Clear sharding env vars so each child runs all filtered tests.
    for(int i = 1; i < argc; ++i)
    {
        if(std::string_view(argv[i]) == "--reset-sharding")
        {
#ifdef _WIN32
            _putenv_s("GTEST_TOTAL_SHARDS", "");
            _putenv_s("GTEST_SHARD_INDEX", "");
#else
            unsetenv("GTEST_TOTAL_SHARDS");
            unsetenv("GTEST_SHARD_INDEX");
#endif
            break;
        }
    }

    testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    return RUN_ALL_TESTS();
}
