// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The library always compiles in every experimental feature, so it enables the switch.
#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
#include "hipblaslt/hipblaslt-experimental.h"

#include <cstring>

namespace
{
    struct FeatureRevision
    {
        const char* name;
        int         revision;
    };

    // One entry per registered feature, with the revision this library was compiled with.
    // The final entry keeps the array non-empty when no feature is registered.
#define HIPBLASLT_EXPERIMENTAL_FEATURE(name) {#name, HIPBLASLT_EXPERIMENTAL_##name},
    constexpr FeatureRevision featureRevisions[] = {
#include "hipblaslt-experimental-registry.inc"
        {nullptr, 0}};
#undef HIPBLASLT_EXPERIMENTAL_FEATURE
}

int hipblasLtExperimentalGetFeatureRevision(const char* feature)
{
    if(feature == nullptr)
        return 0;
    for(const auto& f : featureRevisions)
    {
        if(f.name != nullptr && std::strcmp(f.name, feature) == 0)
            return f.revision;
    }
    return 0;
}
