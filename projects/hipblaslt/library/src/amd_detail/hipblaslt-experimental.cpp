// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The library always compiles in every experimental feature, so it enables the switch.
#define ROCM_ENABLE_EXPERIMENTAL_APIS 1
#include "hipblaslt/hipblaslt-experimental.h"

#include <cstring>
#include <limits>

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

hipblasStatus_t hipblasLtExperimentalDemoAlignSize(size_t                              size,
                                                   size_t                              alignment,
                                                   hipblasLtExperimentalDemoRounding_t rounding,
                                                   size_t*                             result)
{
    if(result == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0)
        return HIPBLAS_STATUS_INVALID_VALUE;

    size_t down = size & ~(alignment - 1);
    switch(rounding)
    {
    case HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_DOWN:
        *result = down;
        return HIPBLAS_STATUS_SUCCESS;
    case HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP:
        if(down == size)
        {
            *result = size;
            return HIPBLAS_STATUS_SUCCESS;
        }
        if(down > std::numeric_limits<size_t>::max() - alignment)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *result = down + alignment;
        return HIPBLAS_STATUS_SUCCESS;
    }
    // Unknown and retired values are rejected, never reinterpreted.
    return HIPBLAS_STATUS_INVALID_VALUE;
}
