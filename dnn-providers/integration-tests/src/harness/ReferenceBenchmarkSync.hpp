// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <hip/hip_runtime_api.h>

#include "harness/Benchmark.hpp"

namespace hipdnn_integration_tests
{

/// Turns a failed synchronize into a ReferenceBenchmarkError, which fails the test
/// rather than letting a fallback reference run.
inline void throwIfReferenceSyncFailed(hipError_t status)
{
    if(status != hipSuccess)
    {
        throw ReferenceBenchmarkError(std::string("reference benchmark synchronization failed: ")
                                      + hipGetErrorString(status));
    }
}

inline void synchronizeReferenceBenchmark()
{
    throwIfReferenceSyncFailed(hipDeviceSynchronize());
}

/// The timer --benchmark-reference uses in production: a real device synchronize
/// around each phase, so a GPU reference's timing covers its kernels and not only
/// their launch.
inline ReferenceBenchmarkTimer productionReferenceTimer()
{
    return ReferenceBenchmarkTimer(synchronizeReferenceBenchmark);
}

} // namespace hipdnn_integration_tests
