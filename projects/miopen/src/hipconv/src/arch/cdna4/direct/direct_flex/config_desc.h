#pragma once

#include "config_table.h"
#include "kv_descriptor.h"

namespace hipconv::cdna4::direct_flex
{
// A descriptor for the kernel's configuration class.
class ConfigMatcher : public hipconv::KVDescriptor
{
public:
    explicit ConfigMatcher(const Variant& v);
};
} // namespace hipconv::cdna4::direct_flex
