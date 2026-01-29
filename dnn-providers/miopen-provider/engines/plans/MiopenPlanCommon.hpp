// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <miopen/miopen.h>

#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>

struct HipdnnEnginePluginExecutionContext;

namespace miopen_legacy_plugin
{

hipdnn_data_sdk::utilities::ScopedResource<miopenSolution_t> find20Solution(
    miopenHandle_t miopenHandle,
    miopenProblem_t problem,
    const HipdnnEnginePluginExecutionContext& executionContext);

} // namespace miopen_legacy_plugin
