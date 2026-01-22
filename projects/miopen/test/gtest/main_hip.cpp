// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// Using here"weak default implementations" for perfdb's multi-process support.
// Rationale: perfdb is the only test that spawns child processes which are not supposed to call
// gtest init. Solution: we have to check for their mode before InitGoogleTest (they have to call
// only WorkItem() and exit)
//
// These weak symbols provide default no-op stubs for tests that don't link perfdb.cpp.
// When perfdb.cpp is linked (e.g., in miopen_gtest) its strong symbols override these.
// unfortunately, this was the only way, since alternatives include either disabling combined
// mode, or creating an additional binary with it's own main() which will result in maintenance
// overhead and having separate handling for one test only.
namespace miopen {
namespace tests {
namespace perfdb {

#if defined(_MSC_VER)
// MSVC: use selectany for weak linkage
__declspec(selectany) void SetExePath(const char*) {}
__declspec(selectany) bool IsChildProcessMode(int, char**) { return false; }
__declspec(selectany) int RunChildProcess(int, char**) { return 0; }
#else
// GCC/Clang: use weak attribute
__attribute__((weak)) void SetExePath(const char*) {}
__attribute__((weak)) bool IsChildProcessMode(int, char**) { return false; }
__attribute__((weak)) int RunChildProcess(int, char**) { return 0; }
#endif

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
