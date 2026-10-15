// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// A stand-in for libhipdnn_backend.so that hipDNN's frontend loads and then
// refuses, so script/test_forwarding_failure_modes.py can make hipDNN
// unavailable without removing the real backend. Built once per variant, each
// in its own directory for HIPDNN_BACKEND_LIBRARY_PATH to point at.

#include <cstdio>

#if defined(FAKE_BACKEND_wrong_major)
#define FAKE_BACKEND_VARIANT "wrong_major"

// A major version no real frontend has.
extern "C" __attribute__((visibility("default"))) const char* hipdnnVersionString_ext()
{
    return "9999.0.0";
}

#elif defined(FAKE_BACKEND_no_version)
#define FAKE_BACKEND_VARIANT "no_version"

// An old or unrelated library with the backend's file name: it loads, but has no
// version for the frontend to check.
extern "C" __attribute__((visibility("default"))) int fakeHipdnnBackendPlaceholder() { return 0; }

#else
#error "Define FAKE_BACKEND_wrong_major or FAKE_BACKEND_no_version"
#endif

namespace {
// Shows the test which backend was loaded, or that none was.
__attribute__((constructor)) void AnnounceLoad()
{
    std::fputs("FAKE_HIPDNN_BACKEND_LOADED:" FAKE_BACKEND_VARIANT "\n", stderr);
}
} // namespace
