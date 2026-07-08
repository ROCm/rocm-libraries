// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/MiopenReluApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

namespace miopen_plugin
{

class MiopenReluPlanBuilder
    : public MiopenUnaryActivationPlanBuilder<relu_applicability::isReluSupported>
{
public:
    MiopenReluPlanBuilder()
        : MiopenUnaryActivationPlanBuilder("Relu")
    {
    }
};

} // namespace miopen_plugin
