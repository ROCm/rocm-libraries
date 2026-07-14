// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Register the chained recording callback: plugin logs reach both the
    // SHARED LogRecording instance AND stderr (logChainedRecordingCallback
    // with force=true calls simpleStderrOutputCallback regardless of
    // HIPDNN_LOG_LEVEL). This makes testing::internal::CaptureStderr() capture
    // the AOT_PROBE_LOAD_OK / AOT_PROBE_LOAD_FAILED markers emitted by
    // AotCatalog::loadForDevice() — the primary observable for the AOT load
    // integration tests.
    hipdnn_test_sdk::utilities::initializeChainedTestLogRecordingShared(
        hipdnn_test_sdk::utilities::simpleStderrOutputCallback);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    return RUN_ALL_TESTS();
}
