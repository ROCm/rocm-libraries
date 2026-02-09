// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <iostream>

namespace hipdnn_test_sdk::utilities
{

// Simple test logging callback that outputs to stderr.
// Log level filtering is already done by the HIPDNN_LOG_* macros before calling
// the callback, so we just output whatever we receive.
inline void testLoggingCallback([[maybe_unused]] hipdnnSeverity_t severity, const char* message)
{
#ifndef DISABLE_TEST_LOGGING
    if(message != nullptr)
    {
        std::cerr << message << '\n';
    }
#else
    (void)message;
#endif
}

} // namespace hipdnn_test_sdk::utilities
