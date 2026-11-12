#pragma once

// A DataType field on a config descriptor, spelled as to_string(DataType) spells it.

#include "hipconv/conv_params.hpp"
#include "kv_descriptor.h"

#include <string_view>

namespace hipconv
{

// Register `value` as the `key` field of `d`, omitted from the short description at
// `default_value`.
void data_type_field(KVDescriptor& d, std::string_view key, DataType value, DataType default_value);

} // namespace hipconv
