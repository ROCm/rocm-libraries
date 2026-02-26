// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "tests/mocks/MockPlan.hpp"

#include "HipdnnHipKernelContext.hpp"

struct MockHipdnnHipKernelContext : public HipdnnHipKernelContext
{
    MockHipdnnHipKernelContext()
        : mockPlan(std::make_unique<hip_kernel_plugin::MockPlan>())
    {
    }

    hipdnn_plugin_sdk::IPlan<HipdnnHipKernelHandle>& plan() const override
    {
        return *mockPlan;
    }

    std::unique_ptr<hip_kernel_plugin::MockPlan> mockPlan;
};
