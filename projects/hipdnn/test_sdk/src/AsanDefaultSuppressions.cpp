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

// SKIP_IF_ASAN() -- named here on purpose. A maintainer auditing which tests are held back by a
// known, unfixed ASAN error greps for SKIP_IF_ASAN(); this file is the other place such an error
// can be parked, and it carries no macro of its own to find. The tree has no SKIP_IF_ASAN() call
// sites: the errors below are suppressed instead, so the affected tests run.
//
// Every pattern in the suppression text is a defect awaiting an upstream fix. Treat the list the
// way a SKIP_IF_ASAN() audit treats its hits: each entry needs a tracking issue, and it comes out
// when the fix lands.
//
// Called by the ASan runtime during start-up. These suppressions combine with any the user supplies
// through ASAN_OPTIONS rather than being replaced by them, so setting a suppressions file does not
// turn these off. Nor does any ASan flag: seeing the suppressed errors needs an edit here and a
// rebuild.
//
// The reserved name is the runtime's, not ours -- it only resolves under exactly this spelling.
// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" HIPDNN_ASAN_HOOK_EXPORT const char* __asan_default_suppressions()
{
    return hipdnn_test_sdk::utilities::asan::K_DEFAULT_SUPPRESSIONS;
}

#endif // ADDRESS_SANITIZER
