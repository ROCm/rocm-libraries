// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// Fwd decl for perfdb multi-process graceful spawn support
// Rationale: perfdb is the only test that spawns a lot of child processes
// in form of invocation of own binary with special arguments, which
// means we'll invoke gtest init and it's not supposed to be called by the child process,
// Solution: associate invocations with single point of initial entry
// which is here - in main_hip.cpp this fwd decl resolves this issue providing
// single point of child process catch before the gtest init
namespace miopen {
namespace tests {
namespace perfdb {
void SetExePath(const char* path);
bool IsChildProcessMode(int argc, char** argv);
int RunChildProcess(int argc, char** argv);
} // namespace perfdb
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

        EXPECT_EQ(hipError, hipSuccess)
            << " hipGetLastError returned error code " << hipError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipError);
        EXPECT_EQ(hipExtError, hipSuccess)
            << " hipExtGetLastError returned error code " << hipExtError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipExtError);
    }
};

int main(int argc, char** argv)
{
    // Handle child process mode for perfdb multi-process tests
    // check rationale out in fwd decl comment on top of this source file
    miopen::tests::perfdb::SetExePath(argv[0]);
    if(miopen::tests::perfdb::IsChildProcessMode(argc, argv))
    {
        return miopen::tests::perfdb::RunChildProcess(argc, argv);
    }

    testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    return RUN_ALL_TESTS();
}
