// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef ADDRESS_SANITIZER

#include <hipdnn_test_sdk/utilities/AsanDefaultSuppressions.hpp>

// The Windows ASan runtime ships as a DLL and finds this hook by looking the symbol up in the main
// module, so it has to be exported to be seen at all.
#if defined(_WIN32)
#define HIPDNN_ASAN_HOOK_EXPORT __declspec(dllexport)
#else
#define HIPDNN_ASAN_HOOK_EXPORT
#endif

// Called by the ASan runtime during start-up. These suppressions combine with any the user supplies
// through ASAN_OPTIONS rather than being replaced by them, so setting a suppressions file does not
// turn these off.
// The reserved name is the runtime's, not ours -- it only resolves under exactly this spelling.
// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" HIPDNN_ASAN_HOOK_EXPORT const char* __asan_default_suppressions()
{
    return hipdnn_test_sdk::utilities::asan::defaultSuppressions();
}

#endif // ADDRESS_SANITIZER
