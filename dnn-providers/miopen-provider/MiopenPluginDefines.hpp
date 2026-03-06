// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnMiopenHandle.hpp"
#include "MiopenContainer.hpp"
#include "version.h"

#define HIPDNN_PLUGIN_NAME "miopen_provider_plugin"
#define HIPDNN_PLUGIN_VERSION MIOPEN_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_API_VERSION "0.0.1"
#define HIPDNN_PLUGIN_CONTAINER_TYPE miopen_plugin::MiopenContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE HipdnnMiopenHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE HipdnnMiopenContext
