#pragma once

#include "kv_descriptor.h"
#include "config_table.h"

namespace hipconv::cdna4::depthwise_2d_toeplitz
{
hipconv::KVDescriptor config_fields(const Config& cfg);
} // namespace hipconv::cdna4::depthwise_2d_toeplitz
