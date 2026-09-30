#pragma once

#include "config_table.h"
#include "kv_descriptor.h"

namespace hipconv::cdna4::depthwise_wgrad_hankel
{
// A descriptor for the kernel's configuration class.
class ConfigMatcher : public hipconv::KVDescriptor
{
public:
    explicit ConfigMatcher(const Config& cfg);
};
} // namespace hipconv::cdna4::depthwise_wgrad_hankel
