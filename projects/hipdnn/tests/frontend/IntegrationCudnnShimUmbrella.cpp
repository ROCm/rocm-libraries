// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Integration-style coverage for the cuDNN-compatibility shim (RFC 0012).
// Unlike the unit-level compile/link check
// (frontend/tests/TestCudnnShimUmbrellaCompiles.cpp), this exercises the shim
// umbrella inside the *integration* executable — the same translation unit pulls
// in the real hipDNN frontend, the backend C-API, and a live device handle via
// IntegrationTestFixture — proving the shim header coexists with, and is usable
// alongside, a fully wired hipDNN stack.
//
// The umbrella is intentionally empty at the skeleton stage, so this test only
// establishes the integration scaffolding; it will grow to build/validate/execute
// a shim graph end-to-end against the in-tree test plugin as the shim's Graph and
// node wrappers land (RFC §7.2, §7.3 / §8.2).
//
// Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in tests/frontend/CMakeLists.txt.

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

namespace
{
using hipdnn_tests::IntegrationTestFixture;

class IntegrationCudnnShim : public IntegrationTestFixture
{
};

// The shim umbrella compiles and links within the integration executable, and a
// live hipDNN handle is available for the (future) shim graph path. Skips on
// hosts without a device (via the fixture's SKIP_IF_NO_DEVICES()).
TEST_F(IntegrationCudnnShim, UmbrellaUsableWithLiveHandle)
{
    ASSERT_NE(_handle, nullptr);
    SUCCEED() << "cuDNN shim umbrella usable alongside a live hipDNN backend handle.";
}

} // namespace
