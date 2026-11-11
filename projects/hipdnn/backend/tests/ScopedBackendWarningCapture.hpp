// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "logging/Logging.hpp"
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

namespace hipdnn_backend::test_utilities
{

/// Routes backend warnings to the isolated log recorder for the duration of a
/// case, which is the only way to observe the resolver's diagnostics. Delivery
/// is synchronous, so the assertions need no waiting.
class ScopedBackendWarningCapture
{
public:
    ScopedBackendWarningCapture()
    {
        EXPECT_EQ(hipdnn_backend::logging::getGlobalLogLevel(_originalLevel),
                  HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(setUserCallback(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnn_backend::logging::setGlobalLogLevel(HIPDNN_SEV_WARN),
                  HIPDNN_STATUS_SUCCESS);
    }

    ~ScopedBackendWarningCapture()
    {
        EXPECT_EQ(setUserCallback(HIPDNN_SEV_OFF), HIPDNN_STATUS_SUCCESS);
        EXPECT_EQ(hipdnn_backend::logging::setGlobalLogLevel(_originalLevel),
                  HIPDNN_STATUS_SUCCESS);
    }

    ScopedBackendWarningCapture(const ScopedBackendWarningCapture&) = delete;
    ScopedBackendWarningCapture& operator=(const ScopedBackendWarningCapture&) = delete;
    ScopedBackendWarningCapture(ScopedBackendWarningCapture&&) = delete;
    ScopedBackendWarningCapture& operator=(ScopedBackendWarningCapture&&) = delete;

private:
    /// The callback and the handle together key the registration, so the same
    /// pair has to come back for the SEV_OFF removal. A null handle is rejected
    /// outright, which is why this passes the guard itself.
    hipdnnStatus_t setUserCallback(hipdnnSeverity_t minLevel)
    {
        return hipdnn_backend::logging::setUserLogCallback(
            hipdnn_test_sdk::utilities::IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
            minLevel,
            HIPDNN_LOG_CALLBACK_SYNC,
            this);
    }

    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};
};

} // namespace hipdnn_backend::test_utilities
