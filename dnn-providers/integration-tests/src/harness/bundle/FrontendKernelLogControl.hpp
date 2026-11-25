// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include "harness/bundle/KernelLogTap.hpp"

namespace hipdnn_integration_tests::bundle
{

/// The production IKernelLogControl: the frontend's user-callback and global-level wrappers,
/// the same ones the serialize round-trip test uses.
///
/// Kept in its own translation unit, like FrontendGraphEngineRunner: it needs a backend, and
/// the unit-test binary links the harness without one and injects a fake through
/// HarnessDependencies instead.
class FrontendKernelLogControl : public IKernelLogControl
{
public:
    bool addUserCallback(hipdnnUserLogCallback_t callback,
                         hipdnnUserLogCallbackHandle_t handle) override;

    void removeUserCallback(hipdnnUserLogCallback_t callback,
                            hipdnnUserLogCallbackHandle_t handle) override;

    std::optional<hipdnnSeverity_t> globalLevel() override;

    bool setGlobalLevel(hipdnnSeverity_t level) override;
};

} // namespace hipdnn_integration_tests::bundle
