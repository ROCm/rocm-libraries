// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cotenant.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// Parse argv[i] as an integer in [lo, hi], or return def when absent.
static long parse_arg(int argc, char** argv, int i, long def, long lo, long hi, const char* error)
{
    if(argc <= i)
        return def;
    char*      end = nullptr;
    const long v   = std::strtol(argv[i], &end, 10);
    if(end == argv[i] || *end != '\0' || v < lo || v > hi)
        throw std::runtime_error(std::string("hipblaslt-cotenant-kernel: ERROR ") + error);
    return v;
}

int main(int argc, char** argv)
try
{
    const int n_cus
        = parse_arg(argc, argv, 1, 1, 1, INT_MAX, "workgroup count must be an integer >= 1");
    const int max_occupancy
        = parse_arg(argc, argv, 2, 1, 1, 64, "max occupancy must be an integer in [1, 64]");

    hipblaslt_cotenant::check(hipSetDevice(0));
    const char* hvd = std::getenv("HIP_VISIBLE_DEVICES");
    std::fprintf(stderr,
                 "hipblaslt-cotenant-kernel: pid=%d HIP_VISIBLE_DEVICES=%s\n",
                 getpid(),
                 hvd ? hvd : "<unset>");

    int* ready = nullptr;
    hipblaslt_cotenant::check(
        hipHostMalloc(reinterpret_cast<void**>(&ready), sizeof(int), hipHostMallocMapped));
    *ready         = 0;
    int* ready_dev = nullptr;
    hipblaslt_cotenant::check(
        hipHostGetDevicePointer(reinterpret_cast<void**>(&ready_dev), ready, 0));
    hipblaslt_cotenant::launch<false>(n_cus, max_occupancy, nullptr, ready_dev);
    hipblaslt_cotenant::wait_ready(ready, n_cus);
    pause();
    return 0;
}
catch(const std::exception& e)
{
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
