// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

// Runs action and expects an ingestor plan refusal. Checks the status, the category prefix
// and a phrase that names the failing check.
template <typename Action>
void expectIngestorPlanRefusal(Action&& action,
                               hipdnnPluginStatus_t status,
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
        const std::string prefix(status == HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE
                                     ? INGESTOR_PLAN_INCOMPATIBLE_PREFIX
                                     : INGESTOR_PLAN_DAMAGED_PREFIX);
        EXPECT_EQ(message.rfind(prefix, 0), 0U) << message;
        EXPECT_NE(message.find(phrase), std::string::npos) << message;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
