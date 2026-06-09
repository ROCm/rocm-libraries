// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ALMIOPEN-2034 acceptance criterion #1: a v9-only translation unit that
// includes ONLY the cuDNN-compatibility shim umbrella header must compile and
// link. This TU is gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in the
// frontend tests CMakeLists, so it is only built when the shim is enabled.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

// The umbrella header is intentionally empty at the skeleton stage; the value of
// this test is that it compiles and links against the shim include path. The
// assertion exists so the test reports a meaningful PASS (RFC 0012 §7.1 exit
// criterion / §8.5).
TEST(TestCudnnShim, UmbrellaIncludesStandalone)
{
    SUCCEED() << "cuDNN shim umbrella header included, compiled, and linked.";
}
