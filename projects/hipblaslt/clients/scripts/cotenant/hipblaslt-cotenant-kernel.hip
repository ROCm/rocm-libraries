// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblaslt-cotenant-kernel: a persistent, compute-free kernel that occupies a fixed
// number of CUs, used to benchmark hipBLASLt under CU contention.
//
// Occupancy is pinned to one workgroup per CU AND the CU is made exclusive by
// reserving the ENTIRE per-CU LDS as dynamic shared memory for that one block.
// Two consequences, both from a single runtime device query (no per-arch
// constants): (1) a second cotenant block can't fit, so a grid of N workgroups
// lands on exactly N CUs; (2) zero LDS remains, so the hardware can't
// co-schedule any main-kernel workgroup onto a cotenant-occupied CU either
// (every hipBLASLt GEMM uses LDS). N is therefore an exact count of CUs removed
// from the main kernel, not merely "one competing block per CU".
//
// (Reserving only > half the LDS still pins the cotenant to 1 block/CU, but
// leaves ~half the LDS free — enough for a low-LDS main kernel to co-reside on
// the same CU and run unaffected, silently under-reporting contention. Full-LDS
// closes that gap.)
//
// The kernel body just parks each wave with s_sleep until the host kills it.
//
// Each workgroup bumps a system-scope atomic counter in host-pinned memory at
// entry; the host waits until all N have reported before printing READY. This
// confirms the grid is actually executing on the CUs (not merely that a GPU
// context exists), so callers can start their workload against full residency.

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

    // Always use device ordinal 0; the caller picks the physical GPU via
    // HIP_VISIBLE_DEVICES, so hipGetDevice would only echo this back.
    constexpr int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    // Mirror the launcher's bounds check: the count must be in [1, CU count).
    // Occupying every CU (or more than exist) leaves none free and the host wait
    // below could never observe all workgroups resident.
    if(n_cus >= prop.multiProcessorCount)
    {
        std::fprintf(stderr,
                     "hipblaslt-cotenant-kernel: ERROR requested %d CUs but device has %d; "
                     "must be < device CU count\n",
                     n_cus,
                     prop.multiProcessorCount);
        return 1;
    }

    // Reserve the ENTIRE per-CU LDS for this one block, so (a) a second cotenant
    // block can't fit (exactly one workgroup per CU) and (b) zero LDS remains for
    // any main-kernel workgroup, making the CU exclusive. hipFuncSetAttribute may
    // request up to SharedMemPerBlockOptin, which equals the per-CU total on
    // gfx942/gfx950, so a single block can legally take all of it. If a device
    // caps the per-block optin below the per-CU total, reserve the optin max (the
    // largest a single block may hold) and accept whatever sliver remains; the
    // occupancy check below still enforces exactly one block per CU.
    int lds_per_cu = 0;
    HIP_CHECK(hipDeviceGetAttribute(
        &lds_per_cu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev));
    int optin = 0;
    HIP_CHECK(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, dev));
    const int max_block_lds = (optin > 0 && optin < lds_per_cu) ? optin : lds_per_cu;
    // Round down to the 256-byte LDS granule so a non-aligned max can't be
    // rounded *up* past the per-block limit and fail the launch.
    const int reserve = (max_block_lds / 256) * 256;
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&busy_spin),
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  reserve));

    // Hard requirement: exactly one workgroup per CU, otherwise a grid of N
    // would occupy fewer than N CUs and the contention level would be wrong.
    // Fail instead of silently benchmarking against the wrong CU count.
    int blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, reinterpret_cast<const void*>(&busy_spin), 256, reserve));
    if(blocks_per_cu != 1)
    {
        std::fprintf(stderr,
                     "hipblaslt-cotenant-kernel: ERROR expected 1 block/CU but device reports %d "
                     "(lds_reserved=%d of %d per CU); cannot pin exactly one workgroup "
                     "per CU on this device.\n",
                     blocks_per_cu,
                     reserve,
                     lds_per_cu);
        return 1;
    }

    const char* hvd = std::getenv("HIP_VISIBLE_DEVICES");
    std::fprintf(stderr,
                 "hipblaslt-cotenant-kernel: pid=%d HIP_VISIBLE_DEVICES=%s device=%d (%s) "
                 "grid=%d block=256 lds_reserved=%d/%d\n",
                 getpid(),
                 hvd ? hvd : "<unset>",
                 dev,
                 prop.gcnArchName,
                 n_cus,
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

    // Wait until every workgroup has reported in, so callers start against full
    // N-CU residency. The host is killed before this returns if the grid never
    // fully lands; the kernel itself keeps the CUs busy regardless.
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
