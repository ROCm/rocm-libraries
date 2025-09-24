// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnStatus.h"
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <memory>

namespace hipdnn_backend
{
namespace flatbuffer_utilities
{
void convertSerializedGraphToGraph(const uint8_t* buffer,
                                   size_t size,
                                   std::unique_ptr<hipdnn_sdk::data_objects::GraphT>& graphOut);
}
}
