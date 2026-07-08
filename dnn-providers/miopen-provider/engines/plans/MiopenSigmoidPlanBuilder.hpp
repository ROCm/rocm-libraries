// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/MiopenSigmoidApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

namespace miopen_plugin
{

class MiopenSigmoidPlanBuilder
    : public MiopenUnaryActivationPlanBuilder<sigmoid_applicability::isSigmoidSupported>
{
public:
    MiopenSigmoidPlanBuilder()
        : MiopenUnaryActivationPlanBuilder("Sigmoid")
    {
    }
};

} // namespace miopen_plugin
