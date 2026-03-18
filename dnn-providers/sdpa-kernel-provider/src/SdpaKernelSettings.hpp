// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>

/**
 * @brief SDPA kernel provider plugin-specific execution settings.
 *
 * This structure holds settings that control SDPA kernel execution behavior.
 * Currently empty - will be extended as plan builders are added.
 */
struct SdpaKernelSettings
{
    SdpaKernelSettings(const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/)
    {
    }
    SdpaKernelSettings() = default;
};
