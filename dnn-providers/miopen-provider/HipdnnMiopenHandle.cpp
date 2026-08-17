// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnMiopenHandle.hpp"
#include "MiopenContainer.hpp"

hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>&
    HipdnnMiopenHandle::getEngineManager()
{
    return container->getEngineManager();
}
