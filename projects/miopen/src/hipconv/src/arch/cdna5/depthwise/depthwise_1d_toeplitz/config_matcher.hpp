#pragma once

#include "config_table.h"
#include "kv_descriptor.h"

namespace hipconv::cdna5::depthwise_1d_toeplitz
{
class ConfigMatcher : public hipconv::KVDescriptor
{
public:
    explicit ConfigMatcher(const Config& cfg);
};
} // namespace hipconv::cdna5::depthwise_1d_toeplitz
