// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>

#include <gtest/gtest.h>

namespace hipdnn_tests
{

// Installs a user log callback and raises the backend's global log level, then
// puts both back on scope exit. A failed assertion leaves the rest of the test
// body unrun, so the restoration cannot be left to the test itself.
class ScopedBackendLogCapture
{
public:
    ScopedBackendLogCapture(hipdnnUserLogCallback_t callback,
                            hipdnnSeverity_t level,
                            void* userData)
        : _callback(callback)
        , _userData(userData)
    {
        EXPECT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&_originalLevel), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            hipdnnSetUserLogCallback_ext(_callback, level, HIPDNN_LOG_CALLBACK_SYNC, _userData),
            HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(level), HIPDNN_STATUS_SUCCESS);
    }

    ~ScopedBackendLogCapture()
    {
        EXPECT_EQ(hipdnnSetUserLogCallback_ext(
                      _callback, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, _userData),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnnBackendSetGlobalLogLevel_ext(_originalLevel), HIPDNN_STATUS_SUCCESS);
    }

    ScopedBackendLogCapture(const ScopedBackendLogCapture&) = delete;
    ScopedBackendLogCapture& operator=(const ScopedBackendLogCapture&) = delete;
    ScopedBackendLogCapture(ScopedBackendLogCapture&&) = delete;
    ScopedBackendLogCapture& operator=(ScopedBackendLogCapture&&) = delete;

private:
    hipdnnUserLogCallback_t _callback;
    void* _userData;
    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};
};

} // namespace hipdnn_tests
