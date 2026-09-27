// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cotenant.hpp"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char** argv)
try
{
    int n_cus = (argc > 1) ? std::atoi(argv[1]) : 1;
    if(n_cus < 1)
    {
        std::fprintf(
            stderr, "hipblaslt-cotenant-kernel: ERROR requested %d CUs; must be >= 1\n", n_cus);
        return 1;
    }
    char*      end           = nullptr;
    const long max_occupancy = (argc > 2) ? std::strtol(argv[2], &end, 10) : 1;
    if(max_occupancy < 1 || max_occupancy > 64 || (end && *end != '\0'))
    {
        std::fprintf(
            stderr,
            "hipblaslt-cotenant-kernel: ERROR max occupancy must be an integer in [1, 64]\n");
        return 1;
    }

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
