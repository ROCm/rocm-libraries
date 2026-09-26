// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

namespace hipblaslt_cotenant
{
    inline void check(hipError_t status)
    {
        if(status != hipSuccess)
            throw std::runtime_error(std::string("cotenant: ") + hipGetErrorString(status));
    }

    template <bool Stoppable>
    __global__ void busy_spin(int* ready, const int* stop)
    {
        // Keep dynamic LDS live and report workgroup residency to the host.
        extern __shared__ volatile char pad[];
        if(threadIdx.x == 0)
        {
            pad[0] = 1;
            __scoped_atomic_fetch_add(ready, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        }
        __syncthreads();
        for(;;)
        {
            for(int i = 0; i < (Stoppable ? 256 : 1); ++i)
                __builtin_amdgcn_s_sleep(127);
            if constexpr(Stoppable)
            {
                // Check the device flag once per wave between sleep batches.
                int done = 0;
                if(threadIdx.x % warpSize == 0)
                    done = __scoped_atomic_load_n(stop, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
                if(__shfl(done, 0))
                    return;
            }
        }
    }

    static __global__ void signal_stop(int* stop)
    {
        __scoped_atomic_store_n(stop, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    }

    template <bool Stoppable>
    inline void launch(
        int n_cus, int max_occupancy, hipStream_t stream, int* ready, const int* stop = nullptr)
    {
        int dev = 0;
        check(hipGetDevice(&dev));
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, dev));
        if(n_cus < 1 || n_cus >= prop.multiProcessorCount)
            throw std::runtime_error(
                "cotenant: workgroup count must be >= 1 and < device CU count ("
                + std::to_string(prop.multiProcessorCount) + ")");
        if(max_occupancy < 1 || max_occupancy > 64)
            throw std::runtime_error("cotenant: max occupancy must be in [1, 64]");

        int lds_per_cu = 0;
        check(hipDeviceGetAttribute(
            &lds_per_cu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev));
        int optin = 0;
        check(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, dev));
        const int max_block_lds = (optin > 0 && optin < lds_per_cu) ? optin : lds_per_cu;
        // Round down to a gfx942/gfx950 LDS granule, then enforce the occupancy cap.
        const int divisor = 256 * max_occupancy;
        int       reserve = (lds_per_cu / divisor) * 256;
        if(reserve == 0 || lds_per_cu / reserve > max_occupancy)
            reserve += 256;
        if(reserve > max_block_lds)
            throw std::runtime_error("cotenant: LDS reservation exceeds the per-block limit");
        check(hipFuncSetAttribute(reinterpret_cast<const void*>(&busy_spin<Stoppable>),
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  reserve));

        int blocks_per_cu = 0;
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_cu, reinterpret_cast<const void*>(&busy_spin<Stoppable>), 256, reserve));
        if(blocks_per_cu < 1 || blocks_per_cu > max_occupancy)
            throw std::runtime_error("cotenant: reported occupancy is outside the requested cap");

        std::fprintf(stderr,
                     "cotenant: device=%d (%s) grid=%d block=256 max_occupancy=%d "
                     "max_blocks_per_cu=%d lds_reserved=%d/%d\n",
                     dev,
                     prop.gcnArchName,
                     n_cus,
                     max_occupancy,
                     blocks_per_cu,
                     reserve,
                     lds_per_cu);
        busy_spin<Stoppable><<<dim3(n_cus), dim3(256), reserve, stream>>>(ready, stop);
        check(hipGetLastError());
    }

    inline void wait_ready(int* ready, int count, double timeout_seconds = 0)
    {
        const auto start = std::chrono::steady_clock::now();
        while(__atomic_load_n(ready, __ATOMIC_ACQUIRE) < count)
        {
            if(timeout_seconds > 0
               && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      >= timeout_seconds)
                throw std::runtime_error("cotenant: timed out waiting for workgroup residency");
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        std::fprintf(stderr, "cotenant: READY %d/%d workgroups resident\n", count, count);
        std::fflush(stderr);
    }

    class Scoped
    {
        hipStream_t stream_      = nullptr;
        hipStream_t gemm_stream_ = nullptr;
        int*        control_     = nullptr;
        int*        stop_        = nullptr;

        void stop() noexcept
        {
            if(stop_)
            {
                signal_stop<<<1, 1, 0, gemm_stream_>>>(stop_);
                const auto status = hipGetLastError();
                if(status != hipSuccess)
                {
                    std::fprintf(
                        stderr, "cotenant: stop launch failed: %s\n", hipGetErrorString(status));
                    std::terminate();
                }
            }
            if(stream_)
            {
                const auto status = hipStreamSynchronize(stream_);
                if(status != hipSuccess)
                    std::fprintf(stderr, "cotenant: stop failed: %s\n", hipGetErrorString(status));
                (void)hipStreamDestroy(stream_);
            }
            if(stop_)
                (void)hipFree(stop_);
            if(control_)
                (void)hipHostFree(control_);
        }

    public:
        Scoped(int n_cus, int max_occupancy, hipStream_t gemm_stream)
            : gemm_stream_(gemm_stream)
        {
            if(n_cus == 0)
                return;
            try
            {
                check(hipStreamSynchronize(gemm_stream));
                check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
                check(hipHostMalloc(reinterpret_cast<void**>(&control_),
                                    sizeof(int),
                                    hipHostMallocMapped | hipHostMallocCoherent));
                *control_           = 0;
                int* device_control = nullptr;
                check(hipHostGetDevicePointer(
                    reinterpret_cast<void**>(&device_control), control_, 0));
                check(hipMalloc(reinterpret_cast<void**>(&stop_), sizeof(int)));
                check(hipMemset(stop_, 0, sizeof(int)));
                launch<true>(n_cus, max_occupancy, stream_, device_control, stop_);
                wait_ready(control_, n_cus, 30);
            }
            catch(...)
            {
                stop();
                throw;
            }
        }

        ~Scoped()
        {
            stop();
        }

        Scoped(const Scoped&)            = delete;
        Scoped& operator=(const Scoped&) = delete;
    };
}
