// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <exception>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

// Runs action and expects a plugin exception. Checks the status, the message prefix and a
// phrase that names the failing check. An empty prefix matches every message. Any other
// exception fails the test. Returns the message, or an empty string when no plugin
// exception occurs.
template <typename Action>
std::string expectPluginException(Action&& action,
                                  hipdnnPluginStatus_t status,
                                  std::string_view prefix,
                                  const std::string& phrase)
{
    try
    {
        action();
        ADD_FAILURE() << "expected a refusal that contains '" << phrase << "'";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        const std::string message = error.getMessage();
        EXPECT_EQ(error.getStatus(), status) << message;
        EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
        EXPECT_NE(message.find(phrase), std::string::npos) << message;
        return message;
    }
    catch(const std::exception& error)
    {
        ADD_FAILURE() << "expected a refusal that contains '" << phrase << "', got "
                      << typeid(error).name() << ": " << error.what();
    }
    catch(...)
    {
        ADD_FAILURE() << "expected a refusal that contains '" << phrase
                      << "', got an exception of an unknown type";
    }
    return {};
}

// Runs action and expects an ingestor plan load refusal with the category prefix of status.
template <typename Action>
std::string expectIngestorPlanRefusal(Action&& action,
                                      hipdnnPluginStatus_t status,
                                      const std::string& phrase)
{
    return expectPluginException(std::forward<Action>(action),
                                 status,
                                 status == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE
                                     ? INGESTOR_PLAN_INCOMPATIBLE_PREFIX
                                     : INGESTOR_PLAN_DAMAGED_PREFIX,
                                 phrase);
}

// Runs action and expects an ingestor plan save refusal with the category prefix of status.
template <typename Action>
std::string expectIngestorPlanSaveRefusal(Action&& action,
                                          hipdnnPluginStatus_t status,
                                          const std::string& phrase)
{
    return expectPluginException(std::forward<Action>(action),
                                 status,
                                 status == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE
                                     ? INGESTOR_PLAN_SAVE_INCOMPATIBLE_PREFIX
                                     : INGESTOR_PLAN_SAVE_DAMAGED_PREFIX,
                                 phrase);
}

// Runs action and expects an internal error, which has no category prefix.
template <typename Action>
std::string expectPluginInternalError(Action&& action, const std::string& phrase)
{
    return expectPluginException(
        std::forward<Action>(action), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "", phrase);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
