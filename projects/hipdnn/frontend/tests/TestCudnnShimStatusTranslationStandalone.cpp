// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-containment regression for the cuDNN-compatibility shim's status
// translation header. This translation unit includes
// detail/status_translation.h *first*, with nothing else from the shim included
// beforehand. If the header ever regresses to depending on cudnn.h for its
// types (which would form an include cycle), this file fails to compile.
//
// Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in the frontend tests
// CMakeLists, so it is only built when the shim is enabled.
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>

#include <gtest/gtest.h>

namespace
{
namespace shim_detail = hipdnn_frontend::compatibility::cudnn_frontend::detail;

// A minimal round-trip sanity check; exhaustive mapping coverage lives in the
// parameterized suites in TestCudnnShimHandle.cpp. The point of this file is the
// standalone *compile*.
TEST(TestCudnnShimStatusTranslationStandalone, HeaderIsSelfContained)
{
    EXPECT_EQ(shim_detail::toCudnnStatus(HIPDNN_STATUS_SUCCESS), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(shim_detail::toHipdnnStatus(CUDNN_STATUS_SUCCESS), HIPDNN_STATUS_SUCCESS);
}

} // namespace
