// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

/// Captures plugin logs for one test and restores every piece of process-global state it
/// touched. Both the global log level and the user callback registration outlive the
/// test otherwise: a raised level changes what later tests emit, and a callback keyed on
/// a destroyed fixture would stay registered. Manual teardown at the end of the body is
/// not enough, because an early ASSERT return skips it.
class ScopedPluginLogCapture
{
public:
    explicit ScopedPluginLogCapture(void* userHandle)
        : _userHandle(userHandle)
    {
        const auto levelRead = hipdnn_frontend::getGlobalLogLevel(_previousLevel);
        EXPECT_EQ(levelRead.code, hipdnn_frontend::ErrorCode::OK) << levelRead.err_msg;

        const auto registered = setCallback(HIPDNN_SEV_INFO);
        EXPECT_EQ(registered.code, hipdnn_frontend::ErrorCode::OK) << registered.err_msg;
        _registered = registered.code == hipdnn_frontend::ErrorCode::OK;

        const auto levelSet = hipdnn_frontend::setGlobalLogLevel(HIPDNN_SEV_INFO);
        EXPECT_EQ(levelSet.code, hipdnn_frontend::ErrorCode::OK) << levelSet.err_msg;
    }

    ~ScopedPluginLogCapture()
    {
        if(_registered)
        {
            static_cast<void>(setCallback(HIPDNN_SEV_OFF));
        }
        static_cast<void>(hipdnn_frontend::setGlobalLogLevel(_previousLevel));
    }

    ScopedPluginLogCapture(const ScopedPluginLogCapture&) = delete;
    ScopedPluginLogCapture& operator=(const ScopedPluginLogCapture&) = delete;
    ScopedPluginLogCapture(ScopedPluginLogCapture&&) = delete;
    ScopedPluginLogCapture& operator=(ScopedPluginLogCapture&&) = delete;

    hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder() const
    {
        return _recorder;
    }

private:
    hipdnn_frontend::Error setCallback(hipdnnSeverity_t minLevel) const
    {
        return hipdnn_frontend::setUserLogCallback(
            hipdnn_test_sdk::utilities::IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
            minLevel,
            hipdnn_frontend::LogCallbackMode::SYNC,
            _userHandle);
    }

    // Declared before the recorder so the recorder's own saved-level restore runs first.
    hipdnnSeverity_t _previousLevel = HIPDNN_SEV_OFF;
    void* _userHandle;
    bool _registered = false;
    mutable hipdnn_test_sdk::utilities::IsolatedLogRecorder _recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration
