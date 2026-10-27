#pragma once

// A Direction field on a config descriptor, spelled fprop, dgrad, or wgrad.

#include "hipconv/conv_params.hpp"
#include "kv_descriptor.h"

namespace hipconv
{

// Register `value` as the "direction" field of `d`.
void direction_field(KVDescriptor& d, Direction value);

// The same, omitted from the short description at `default_value`.
void direction_field(KVDescriptor& d, Direction value, Direction default_value);

} // namespace hipconv
