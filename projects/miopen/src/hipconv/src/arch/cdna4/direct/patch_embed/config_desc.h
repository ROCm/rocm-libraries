#pragma once

#include "config.h"
#include "kv_descriptor.h"

namespace hipconv::cdna4::patch_embed
{
// A descriptor for the kernel's configuration class.
class ConfigMatcher : public hipconv::KVDescriptor
{
public:
    explicit ConfigMatcher(const Config& cfg);
};
} // namespace hipconv::cdna4::patch_embed
