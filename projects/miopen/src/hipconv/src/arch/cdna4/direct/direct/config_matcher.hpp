#pragma once

#include "config.hpp"
#include "kv_descriptor.h"


namespace hipconv::cdna4::direct
{
class ConfigMatcher : public hipconv::KVDescriptor
{
public:
    explicit ConfigMatcher(const Config& cfg);
};
} // namespace hipconv::cdna4::direct
