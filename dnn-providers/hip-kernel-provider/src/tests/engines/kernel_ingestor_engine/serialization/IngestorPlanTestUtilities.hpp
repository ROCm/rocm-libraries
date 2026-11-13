// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

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

// Expects `read` to refuse `inputs` as a valid plan that cannot be restored, with `phrase`
// in the message.
template <typename TReader>
void expectReadRefusal(TReader read,
                       const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                       const std::string& phrase)
{
    expectIngestorPlanRefusal(
        [&]() { static_cast<void>(read(inputs)); }, HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE, phrase);
}

// The refusals every launch-value reader shares: a missing value, an extra value, an
// integer stored as a string or as an integer list, and the unversioned dispatch name,
// which is registered but names no contract.
template <typename TReader>
void expectCommonReadRefusals(TReader read,
                              const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& valid,
                              const std::string& unversionedAlias,
                              const std::string& integerName)
{
    using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

    SavedLaunchInputs missing = valid;
    missing.values.erase(integerName);
    expectReadRefusal(read, missing, "needs launch value '" + integerName + "'");

    SavedLaunchInputs extra = valid;
    extra.values["unexpected"] = int64_t{1};
    expectReadRefusal(read, extra, "launch value 'unexpected' is not part of the contract");

    SavedLaunchInputs wrongType = valid;
    wrongType.values[integerName] = std::string("1");
    expectReadRefusal(read, wrongType, "launch value '" + integerName + "'");
    expectReadRefusal(read, wrongType, "must be int in [");
    expectReadRefusal(read, wrongType, "and the plan holds string");

    SavedLaunchInputs wrongList = valid;
    wrongList.values[integerName] = std::vector<int64_t>{1};
    expectReadRefusal(read, wrongList, "and the plan holds int_list");

    SavedLaunchInputs unversioned = valid;
    unversioned.dispatchSymbol = unversionedAlias;
    expectReadRefusal(read, unversioned, "names no launch-value contract");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
