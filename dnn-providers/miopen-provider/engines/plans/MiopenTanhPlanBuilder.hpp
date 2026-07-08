// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/plans/MiopenTanhApplicabilityChecks.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

namespace miopen_plugin
{

class MiopenTanhPlanBuilder
    : public MiopenUnaryActivationPlanBuilder<tanh_applicability::isTanhSupported>
{
public:
    MiopenTanhPlanBuilder()
        : MiopenUnaryActivationPlanBuilder("Tanh")
    {
    }
};

} // namespace miopen_plugin
