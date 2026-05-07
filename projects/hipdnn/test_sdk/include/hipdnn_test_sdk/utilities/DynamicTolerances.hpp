// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Umbrella header — includes all dynamic tolerance sub-headers.
// Existing code that includes DynamicTolerances.hpp continues to work unchanged.
// New operation-specific tolerance implementations should be added to their own
// header (e.g., DynamicTolerancesBatchNorm.hpp) and included here.

#include <hipdnn_test_sdk/utilities/DynamicTolerancesBatchNorm.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesLayerNorm.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesMatmul.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesPointwise.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesRMSNorm.hpp>
