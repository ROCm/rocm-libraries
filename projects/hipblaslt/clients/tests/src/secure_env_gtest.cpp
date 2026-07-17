// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for ROCM-26729 / SEC-00896 (Untrusted Search Path).
//
// hipBLASLt selects the directory it loads GPU code objects and ExtOp/msgpack
// solution libraries from using the HIPBLASLT_TENSILE_LIBPATH and
// HIPBLASLT_EXT_OP_LIBRARY_PATH environment variables. Reading those with a
// plain getenv means a set-user-ID / set-group-ID process that inherits a
// hostile environment can be redirected to an attacker-controlled directory
// and made to execute arbitrary GPU ISA. The fix routes those reads through
// rocblaslt_secure_getenv (rocblaslt_secure_env.hpp), which refuses the
// override when the process is privileged while leaving the documented,
// sanctioned HIPBLASLT_TENSILE_LIBPATH workflow untouched for normal use.
//
// This is a GPU-free, deterministic, host-only unit test that pins the full
// suppression contract in an ordinary (non-privileged) CI process:
//   * A normal process still sees the override (the workflow is not broken).
//   * The process is correctly classified as non-privileged, so nothing is
//     suppressed.
//   * When privileged, the override IS suppressed. The suppression policy is
//     factored into rocblaslt_secure_getenv_impl / _suppressed_for_security_impl
//     (rocblaslt_secure_env.hpp), which take the privilege state as a parameter,
//     so the privileged branch is exercised directly by passing is_privileged
//     =true -- no set-uid/set-gid harness required.
// The only piece that cannot be exercised without real privilege is the OS
// probe itself (rocblaslt_process_is_privileged -> getauxval(AT_SECURE) /
// real-vs-effective ID compare); that is kernel/glibc behavior rather than our
// logic, and is validated manually (see the PR test plan).
//
// Smoke tier: PR CI runs hipBLASLt with TEST_TYPE=quick, which selects
// `--gtest_filter=*smoke*` (test/therock/test_hipblaslt.py). The test names
// below carry the `smoke` token so this fast, host-only guard runs on the PR
// gate, not only in the full/nightly lane.

#include <gtest/gtest.h>

// Included by relative path on purpose. rocblaslt_secure_env.hpp is a
// dependency-free header (only standard headers), so a relative include keeps
// this white-box test self-contained without adding the internal rocblaslt
// include directory to the whole hipblaslt-test target -- doing that shadows
// the clients' own "utility.hpp" with the library's internal one and breaks the
// build of unrelated test sources.
#include "../../../library/src/amd_detail/rocblaslt/src/include/rocblaslt_secure_env.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
    constexpr const char* kTestVar = "HIPBLASLT_SECURE_ENV_GTEST";

    void setTestEnv(const char* value)
    {
#if defined(_WIN32)
        _putenv_s(kTestVar, value);
#else
        setenv(kTestVar, value, /*overwrite=*/1);
#endif
    }

    void unsetTestEnv()
    {
#if defined(_WIN32)
        _putenv_s(kTestVar, "");
#else
        unsetenv(kTestVar);
#endif
    }
}

// The hipBLASLt CI test process is not set-uid/set-gid, so it must be
// classified as non-privileged. If this ever fails the two tests below no
// longer prove what they claim, so assert it explicitly.
TEST(SecureEnv, smoke_TestProcessIsNotPrivileged)
{
    EXPECT_FALSE(rocblaslt_process_is_privileged())
        << "The test process is unexpectedly classified as privileged; the "
           "non-privileged assertions below would be vacuous.";
}

// Workflow-preservation guard: for a normal process rocblaslt_secure_getenv must
// behave exactly like getenv, so the documented HIPBLASLT_TENSILE_LIBPATH /
// HIPBLASLT_EXT_OP_LIBRARY_PATH override keeps working. A regression that
// over-broadly suppressed the env (e.g. always returning nullptr) would break
// every deployment that relies on it; this test catches that.
TEST(SecureEnv, smoke_HonorsEnvForNonPrivilegedProcess)
{
    setTestEnv("/tmp/hipblaslt-secure-env-test");

    const char* value = rocblaslt_secure_getenv(kTestVar);
    ASSERT_NE(value, nullptr)
        << "ROCM-26729: rocblaslt_secure_getenv dropped a set environment variable for a "
           "non-privileged process, which would break the documented "
           "HIPBLASLT_TENSILE_LIBPATH workflow.";
    EXPECT_STREQ(value, "/tmp/hipblaslt-secure-env-test");

    // A set variable is not "suppressed for security" in a non-privileged process.
    EXPECT_FALSE(rocblaslt_env_suppressed_for_security(kTestVar));

    unsetTestEnv();
}

// An unset variable yields nullptr (same as getenv), and a nullptr name must be
// handled without dereferencing it.
TEST(SecureEnv, smoke_ReturnsNullWhenUnsetOrNullName)
{
    unsetTestEnv();

    EXPECT_EQ(rocblaslt_secure_getenv(kTestVar), nullptr);
    EXPECT_FALSE(rocblaslt_env_suppressed_for_security(kTestVar));

    EXPECT_EQ(rocblaslt_secure_getenv(nullptr), nullptr);
    EXPECT_FALSE(rocblaslt_env_suppressed_for_security(nullptr));
}

// Core security guarantee: a privileged process must NOT see the override, even
// when the variable is set in the environment. The suppression policy is tested
// directly via the _impl entry points by forcing is_privileged=true, which is
// the exact branch a genuine set-uid/set-gid process would take -- so the
// escalation vector from ROCM-26729 is covered without a set-uid test harness.
TEST(SecureEnv, smoke_SuppressesEnvForPrivilegedProcess)
{
    setTestEnv("/tmp/hipblaslt-secure-env-test");

    // Privileged: the set variable is refused ...
    EXPECT_EQ(rocblaslt_secure_getenv_impl(kTestVar, /*is_privileged=*/true), nullptr)
        << "ROCM-26729: a privileged process was allowed to honor "
           "HIPBLASLT_TENSILE_LIBPATH-style overrides, which is the untrusted "
           "search-path escalation this fix must block.";
    // ... and reported as suppressed-for-security (for the diagnostic log).
    EXPECT_TRUE(rocblaslt_env_suppressed_for_security_impl(kTestVar, /*is_privileged=*/true));

    // Non-privileged: the same set variable is honored, confirming the two
    // branches diverge on privilege state alone (not on some other condition).
    EXPECT_STREQ(rocblaslt_secure_getenv_impl(kTestVar, /*is_privileged=*/false),
                 "/tmp/hipblaslt-secure-env-test");
    EXPECT_FALSE(rocblaslt_env_suppressed_for_security_impl(kTestVar, /*is_privileged=*/false));

    unsetTestEnv();
}

// A privileged process with the variable UNSET has nothing to suppress: the
// getenv result is nullptr and _suppressed_for_security must stay false so the
// diagnostic log does not fire spuriously. A nullptr name is handled safely.
TEST(SecureEnv, smoke_PrivilegedProcessHandlesUnsetAndNullName)
{
    unsetTestEnv();

    EXPECT_EQ(rocblaslt_secure_getenv_impl(kTestVar, /*is_privileged=*/true), nullptr);
    EXPECT_FALSE(rocblaslt_env_suppressed_for_security_impl(kTestVar, /*is_privileged=*/true));

    EXPECT_EQ(rocblaslt_secure_getenv_impl(nullptr, /*is_privileged=*/true), nullptr);
    EXPECT_FALSE(rocblaslt_env_suppressed_for_security_impl(nullptr, /*is_privileged=*/true));
}
