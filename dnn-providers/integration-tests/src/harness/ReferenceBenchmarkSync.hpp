#pragma once

#include <string>

#include <hip/hip_runtime_api.h>

#include "harness/Benchmark.hpp"

namespace hipdnn_integration_tests
{

inline void synchronizeReferenceBenchmark()
{
    const auto status = hipDeviceSynchronize();
    if(status != hipSuccess)
    {
        throw ReferenceBenchmarkError(std::string("reference benchmark synchronization failed: ")
                                      + hipGetErrorString(status));
    }
}

} // namespace hipdnn_integration_tests
