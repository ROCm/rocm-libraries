// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef ADDRESS_SANITIZER

#include <gtest/gtest.h>

// Declared rather than included: the ASan runtime owns this symbol and ships no header for it.
// The reserved name is the runtime's, not ours -- it only resolves under exactly this spelling.
// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __asan_default_suppressions();

TEST(TestAsanDefaultSuppressions, AsanRuntimeHookReturnsTheTensileSuppression)
{
    // Two properties in one assertion. The ASan runtime provides a weak default returning "", so
    // seeing any text at all proves the strong definition in this executable is the one the runtime
    // found -- the property the whole mechanism depends on and that no other test covers. Stating
    // the expected text here rather than sharing a constant with the definition means changing the
    // pattern has to be deliberate.
    EXPECT_STREQ(__asan_default_suppressions(), "interceptor_via_fun:*findBestKeyMatch*\n");
}

#endif // ADDRESS_SANITIZER
