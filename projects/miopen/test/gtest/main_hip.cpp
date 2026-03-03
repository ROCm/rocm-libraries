// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdlib>

// Export argc/argv for tests that need command-line argument access
// like for perfdb multi-process tests where child processes receive args via command-line
namespace miopen {
namespace tests {

int g_argc    = 0;
char** g_argv = nullptr;

} // namespace tests
} // namespace miopen

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
    // Save for tests that need arg access
    miopen::tests::g_argc = argc;
    miopen::tests::g_argv = argv;

    testing::InitGoogleTest(&argc, argv);

    // By this moment GTest has already parsed sharding env vars (GTEST_TOTAL_SHARDS, GTEST_SHARD_INDEX)
    // during InitGoogleTest(). Clear them here so child processes spawned by multiprocess tests
    // (e.g. perfdb) don't inherit sharding and skip work they're expected to perform.
#ifdef _WIN32
    _putenv_s("GTEST_TOTAL_SHARDS", "");
    _putenv_s("GTEST_SHARD_INDEX", "");
#else
    unsetenv("GTEST_TOTAL_SHARDS");
    unsetenv("GTEST_SHARD_INDEX");
#endif

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    return RUN_ALL_TESTS();
}
