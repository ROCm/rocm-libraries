// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/FrontendKernelLogControl.hpp"

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Logging.hpp>

namespace hipdnn_integration_tests::bundle
{

bool FrontendKernelLogControl::addUserCallback(hipdnnUserLogCallback_t callback,
                                               hipdnnUserLogCallbackHandle_t handle)
{
    return hipdnn_frontend::setUserLogCallback(
               callback, HIPDNN_SEV_INFO, hipdnn_frontend::LogCallbackMode::SYNC, handle)
        .is_good();
}

void FrontendKernelLogControl::removeUserCallback(hipdnnUserLogCallback_t callback,
                                                  hipdnnUserLogCallbackHandle_t handle)
{
    // A minimum level of off removes the registration.
    static_cast<void>(hipdnn_frontend::setUserLogCallback(
        callback, HIPDNN_SEV_OFF, hipdnn_frontend::LogCallbackMode::SYNC, handle));
}

std::optional<hipdnnSeverity_t> FrontendKernelLogControl::globalLevel()
{
    hipdnnSeverity_t level = HIPDNN_SEV_OFF;
    if(hipdnn_frontend::getGlobalLogLevel(level).is_bad())
    {
        return std::nullopt;
    }
    return level;
}

bool FrontendKernelLogControl::setGlobalLevel(hipdnnSeverity_t level)
{
    return hipdnn_frontend::setGlobalLogLevel(level).is_good();
}

} // namespace hipdnn_integration_tests::bundle
