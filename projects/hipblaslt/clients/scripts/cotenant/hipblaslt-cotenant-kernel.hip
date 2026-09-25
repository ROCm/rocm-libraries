// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Persistent LDS contention: arguments are workgroup count and max occupancy.
// Each workgroup reserves about LDS per CU / max occupancy, rounded to an LDS
// granule, then sleeps. The default of 1 reserves all LDS; larger values allow
// workgroups to share CUs. READY confirms all workgroups have started.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define HIP_CHECK(expr)                                                                           \
    do                                                                                            \
    {                                                                                             \
        hipError_t _e = (expr);                                                                   \
        if(_e != hipSuccess)                                                                      \
        {                                                                                         \
            std::fprintf(stderr, "hipblaslt-cotenant-kernel: %s failed: %s\n", #expr, hipGetErrorString(_e)); \
            return 1;                                                                             \
        }                                                                                         \
    } while(0)

// Dynamic LDS sized at launch; the single live store keeps the allocation from
// being optimised away so the per-CU LDS limiter actually fires. Thread 0 of
// each workgroup reports residency via a system-scope atomic so the host can
// observe the update on the pinned counter.
__global__ void busy_spin(int* ready)
{
    extern __shared__ volatile char pad[];
    if(threadIdx.x == 0)
    {
        pad[0] = 1;
        __scoped_atomic_fetch_add(ready, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    for(;;)
        __builtin_amdgcn_s_sleep(127);
}

int main(int argc, char** argv)
{
    int n_cus = (argc > 1) ? std::atoi(argv[1]) : 1;
    if(n_cus < 1)
    {
        std::fprintf(stderr, "hipblaslt-cotenant-kernel: ERROR requested %d CUs; must be >= 1\n", n_cus);
        return 1;
    }
    char* end = nullptr;
    const long max_occupancy = (argc > 2) ? std::strtol(argv[2], &end, 10) : 1;
    if(max_occupancy < 1 || max_occupancy > 64 || (end && *end != '\0'))
    {
        std::fprintf(stderr, "hipblaslt-cotenant-kernel: ERROR max occupancy must be an integer in [1, 64]\n");
        return 1;
    }

    // Always use device ordinal 0; the caller picks the physical GPU via
    // HIP_VISIBLE_DEVICES, so hipGetDevice would only echo this back.
    constexpr int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    // Keep the launcher's CU-count bound, including with partial LDS reservation.
    if(n_cus >= prop.multiProcessorCount)
    {
        std::fprintf(stderr,
                     "hipblaslt-cotenant-kernel: ERROR requested %d CUs but device has %d; "
                     "must be < device CU count\n",
                     n_cus,
                     prop.multiProcessorCount);
        return 1;
    }

    int lds_per_cu = 0;
    HIP_CHECK(hipDeviceGetAttribute(
        &lds_per_cu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev));
    int optin = 0;
    HIP_CHECK(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, dev));
    const int max_block_lds = (optin > 0 && optin < lds_per_cu) ? optin : lds_per_cu;
    // Round down to a gfx942/gfx950 LDS granule, then enforce the occupancy cap.
    const int divisor = 256 * max_occupancy;
    int reserve = (lds_per_cu / divisor) * 256;
    if(reserve == 0 || lds_per_cu / reserve > max_occupancy)
        reserve += 256;
    if(reserve > max_block_lds)
    {
        std::fprintf(stderr,
                     "hipblaslt-cotenant-kernel: ERROR LDS reservation %d exceeds per-block limit %d\n",
                     reserve,
                     max_block_lds);
        return 1;
    }
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&busy_spin),
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  reserve));

    int blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, reinterpret_cast<const void*>(&busy_spin), 256, reserve));
    if(blocks_per_cu < 1 || blocks_per_cu > max_occupancy)
    {
        std::fprintf(stderr,
                     "hipblaslt-cotenant-kernel: ERROR expected 1..%ld blocks/CU but device reports %d "
                     "(lds_reserved=%d of %d per CU).\n",
                     max_occupancy,
                     blocks_per_cu,
                     reserve,
                     lds_per_cu);
        return 1;
    }

    const char* hvd = std::getenv("HIP_VISIBLE_DEVICES");
    std::fprintf(stderr,
                 "hipblaslt-cotenant-kernel: pid=%d HIP_VISIBLE_DEVICES=%s device=%d (%s) "
                 "grid=%d block=256 max_occupancy=%ld max_blocks_per_cu=%d lds_reserved=%d/%d\n",
                 getpid(),
                 hvd ? hvd : "<unset>",
                 dev,
                 prop.gcnArchName,
                 n_cus,
                 max_occupancy,
                 blocks_per_cu,
                 reserve,
                 lds_per_cu);

    // Host-pinned residency counter, written by each workgroup via a
    // system-scope atomic and polled here through its device-mapped pointer.
    int* ready = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ready), sizeof(int), hipHostMallocMapped));
    *ready         = 0;
    int* ready_dev = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ready_dev), ready, 0));

    busy_spin<<<dim3(n_cus), dim3(256), reserve>>>(ready_dev);
    HIP_CHECK(hipGetLastError());

    // Wait for all workgroups, which may share CUs when max occupancy > 1.
    while(__atomic_load_n(ready, __ATOMIC_ACQUIRE) < n_cus)
        usleep(200);

    std::fprintf(stderr,
                 "hipblaslt-cotenant-kernel: READY %d/%d workgroups resident; pausing until killed\n",
                 n_cus,
                 n_cus);
    std::fflush(stderr);
    pause();
    return 0;
}
