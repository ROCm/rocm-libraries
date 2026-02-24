// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>

#include "mocks/MockPlan.hpp"

#include "HipdnnMiopenContext.hpp"

struct MockHipdnnMiopenContext : public HipdnnMiopenContext
{
    MockHipdnnMiopenContext()
        : mockPlan(std::make_unique<miopen_plugin::MockPlan>())
    {
    }

    hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>& plan() const override
    {
        return *mockPlan;
    }

    std::unique_ptr<miopen_plugin::MockPlan> mockPlan;
};
