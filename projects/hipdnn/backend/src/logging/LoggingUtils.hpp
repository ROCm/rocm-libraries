// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace hipdnn::backend::logging
{

/**
 * @brief Generate the spdlog pattern string for a component
 *
 * This is used internally by the backend for formatting log messages.
 * The pattern includes timestamp, thread ID, log level, and component name.
 *
 * @param componentName The name of the component
 * @return The spdlog pattern string
 */
inline std::string generatePatternString(const std::string& componentName)
{
    return "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] [" + componentName + "] %v";
}

/**
 * @brief Generate the spdlog pattern string for callback-received messages
 *
 * This is used for messages received via the logging callback from non-backend components.
 * The component name is already prepended to the message, so we only add timestamp, thread ID, and level.
 *
 * @return The spdlog pattern string
 */
inline std::string generateCallbackReceiverPatternString()
{
    return "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] %v";
}

} // namespace hipdnn::backend::logging
