// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_test_sdk/utilities/AsanDefaultSuppressions.hpp>

#include <gtest/gtest.h>

namespace asan = hipdnn_test_sdk::utilities::asan;

TEST(TestAsanDefaultSuppressions, SuppressionTextNamesTheTensileEntryPoint)
{
    // Stated independently of the constant, so changing the pattern has to be deliberate.
    EXPECT_STREQ(asan::K_DEFAULT_SUPPRESSIONS, "interceptor_via_fun:*findBestKeyMatch*\n");
}

#ifdef ADDRESS_SANITIZER

// Declared rather than included: the ASan runtime owns this symbol and ships no header for it.
// The reserved name is the runtime's, not ours -- it only resolves under exactly this spelling.
// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __asan_default_suppressions();

TEST(TestAsanDefaultSuppressions, AsanRuntimeHookResolvesToOurDefinition)
{
    // The ASan runtime provides a weak default returning "". Seeing our text here proves the
    // strong definition in this executable is the one the runtime found -- the property the whole
    // mechanism depends on and that no other test covers.
    EXPECT_STREQ(__asan_default_suppressions(), asan::K_DEFAULT_SUPPRESSIONS);
}

#endif // ADDRESS_SANITIZER
