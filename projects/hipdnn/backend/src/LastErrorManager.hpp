// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnBackendLimits.h"
#include "HipdnnStatus.h"
#include <thread>

namespace hipdnn_backend
{

class LastErrorManager
{
private:
    // We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
    // This prevents the shared object (plugin) from being unloaded until the program terminates.
    // Note: We need to nolint this to avoid issues since static confused clang-tidy.
    // NOLINTNEXTLINE
    thread_local static char s_lastError[HIPDNN_ERROR_STRING_MAX_LENGTH];

public:
    static hipdnnStatus_t setLastError(hipdnnStatus_t status, const char* message);
    static hipdnnStatus_t setLastError(hipdnnStatus_t status, const std::string& message);
    static const char* getLastError();
};

} // namespace hipdnn_backend
