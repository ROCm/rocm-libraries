// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslHandle.hpp"

#include "CkDslContainer.hpp"

hipdnn_plugin_sdk::EngineManager<::CkDslHandle, ck_dsl_provider::CkDslSettings,
                                 ck_dsl_provider::CkDslContext>&
CkDslHandle::getEngineManager() const {
    return container->getEngineManager();
}
