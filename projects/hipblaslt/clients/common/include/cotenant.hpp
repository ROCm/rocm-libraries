// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace hipblaslt_cotenant
{
    constexpr int kMaxOccupancy = 64;

    // Returns an error message for invalid cotenant options, or an empty string.
    inline std::string validate_options(int n_cus, int max_occupancy)
    {
#ifdef _WIN32
        if(n_cus != 0)
            return "cotenant-cus is not supported on Windows.";
#endif
        if(n_cus < 0)
            return "cotenant-cus must be >= 0.";
        if(max_occupancy < 1 || max_occupancy > kMaxOccupancy)
            return "cotenant-max-occupancy must be in [1, " + std::to_string(kMaxOccupancy)
                   + "].";
        return {};
    }

    // Dynamic LDS bytes per workgroup so that at most max_occupancy workgroups
    // fit in lds_per_cu. Rounds down to a 256-byte granule, then adds one granule
    // if the cap would be exceeded. Throws if the result exceeds the per-block limit.
    inline int lds_reservation(int lds_per_cu, int block_optin_lds, int max_occupancy)
    {
        const int max_block_lds
            = (block_optin_lds > 0 && block_optin_lds < lds_per_cu) ? block_optin_lds : lds_per_cu;
        int reserve = (lds_per_cu / (256 * max_occupancy)) * 256;
        if(reserve == 0 || lds_per_cu / reserve > max_occupancy)
            reserve += 256;
        if(reserve > max_block_lds)
            throw std::runtime_error(
                "cotenant: max occupancy " + std::to_string(max_occupancy) + " needs "
                + std::to_string(reserve) + " bytes of LDS per workgroup, above the device's "
                + std::to_string(max_block_lds) + "-byte limit; use a higher max occupancy");
        return reserve;
    }
}

#ifdef _WIN32

// POSIX-only; the bench rejects --cotenant-cus on Windows.
namespace hipblaslt_cotenant
{
    class Scoped
    {
    public:
        Scoped(int, int, hipStream_t) {}
    };
}

#else

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <thread>

// The clients provide hipblaslt_cerr; the standalone kernel build does not.
#if __has_include("hipblaslt_ostream.hpp")
#include "hipblaslt_ostream.hpp"
#define HIPBLASLT_COTENANT_USE_OSTREAM
#endif

namespace hipblaslt_cotenant
{
    inline void check(hipError_t status)
    {
        if(status != hipSuccess)
            throw std::runtime_error(std::string("cotenant: ") + hipGetErrorString(status));
    }

    // printf-style log. The clients poison stdio identifiers, so they log through
    // hipblaslt_cerr; the standalone binary writes to stderr directly.
    inline void log(const char* fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
#ifdef HIPBLASLT_COTENANT_USE_OSTREAM
        char      buf[512];
        const int n = std::vsnprintf(buf, sizeof(buf), fmt, args);
        if(n > 0)
            hipblaslt_cerr << std::string(buf, std::min<size_t>(n, sizeof(buf) - 1));
#else
        std::vfprintf(stderr, fmt, args);
#endif
        va_end(args);
    }

    // A cotenant that cannot be stopped keeps its CUs, so later measurements would
    // silently run under contention. Exit without running destructors, which could
    // block on the wedged device.
    [[noreturn]] inline void die(const char* msg)
    {
        log("cotenant: %s\n", msg);
#ifdef HIPBLASLT_COTENANT_USE_OSTREAM
        hipblaslt_cout.flush();
        hipblaslt_cerr.flush();
#endif
        std::_Exit(1);
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
    inline void launch(int         n_cus,
                       int         max_occupancy,
                       hipStream_t stream,
                       int*        ready,
                       const int*  stop    = nullptr,
                       bool        verbose = true)
    {
        int dev = 0;
        check(hipGetDevice(&dev));
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, dev));
        if(n_cus < 1 || n_cus > prop.multiProcessorCount)
            throw std::runtime_error(
                "cotenant: workgroup count must be in [1, device CU count] ("
                + std::to_string(prop.multiProcessorCount) + ")");
        if(max_occupancy < 1 || max_occupancy > kMaxOccupancy)
            throw std::runtime_error("cotenant: max occupancy must be in [1, "
                                     + std::to_string(kMaxOccupancy) + "]");

        int lds_per_cu = 0;
        check(hipDeviceGetAttribute(
            &lds_per_cu, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev));
        int optin = 0;
        check(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin, dev));
        const int reserve = lds_reservation(lds_per_cu, optin, max_occupancy);
        check(hipFuncSetAttribute(reinterpret_cast<const void*>(&busy_spin<Stoppable>),
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  reserve));

        int blocks_per_cu = 0;
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_cu, reinterpret_cast<const void*>(&busy_spin<Stoppable>), 256, reserve));
        if(blocks_per_cu < 1 || blocks_per_cu > max_occupancy)
            throw std::runtime_error("cotenant: reported occupancy is outside the requested cap");
        if(verbose && blocks_per_cu < max_occupancy)
            log("cotenant: warning: wave slots limit occupancy to %d blocks per CU (requested %d)\n",
                blocks_per_cu,
                max_occupancy);

        if(verbose)
            log("cotenant: device=%d (%s) grid=%d block=256 max_occupancy=%d "
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

    inline void wait_ready(int* ready, int count, double timeout_seconds = 0, bool verbose = true)
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
        if(verbose)
        {
            log("cotenant: READY %d/%d workgroups resident\n", count, count);
        }
    }

    // Runs a cotenant kernel on its own stream for the object's lifetime. The
    // constructor syncs the GEMM stream and waits for residency, so startup stays
    // outside any timing that begins after construction. The stop kernel is queued
    // on the GEMM stream, so shutdown follows the measured work.
    class Scoped
    {
        static constexpr double kStopTimeoutSeconds = 30;

        hipStream_t stream_      = nullptr;
        hipStream_t gemm_stream_ = nullptr;
        int*        control_     = nullptr;
        int*        stop_host_   = nullptr;
        int*        stop_dev_    = nullptr;

        // Returns true once the cotenant stream drains; false on timeout or error.
        bool wait_stopped() noexcept
        {
            const auto start = std::chrono::steady_clock::now();
            for(;;)
            {
                const auto status = hipStreamQuery(stream_);
                if(status == hipSuccess)
                    return true;
                if(status != hipErrorNotReady)
                {
                    log("cotenant: stop failed: %s\n", hipGetErrorString(status));
                    return false;
                }
                if(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                   >= kStopTimeoutSeconds)
                    return false;
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }

        void stop() noexcept
        {
            bool stopped = true;
            if(stream_)
            {
                // Order the stop after the GEMM stream's work; clear stale errors first.
                bool queued = false;
                if(stop_dev_)
                {
                    (void)hipGetLastError();
                    signal_stop<<<1, 1, 0, gemm_stream_>>>(stop_dev_);
                    const auto status = hipGetLastError();
                    queued            = status == hipSuccess;
                    if(!queued)
                        log("cotenant: stop launch failed: %s\n", hipGetErrorString(status));
                }
                // The flag is host-visible, so write it directly if the GEMM stream
                // cannot deliver the stop.
                if(!queued && stop_host_)
                    __atomic_store_n(stop_host_, 1, __ATOMIC_RELEASE);
                stopped = wait_stopped();
                if(!stopped && queued && stop_host_)
                {
                    log("cotenant: stop kernel did not run in time; stopping from the host\n");
                    __atomic_store_n(stop_host_, 1, __ATOMIC_RELEASE);
                    stopped = wait_stopped();
                }
            }
            if(!stopped)
                die("the cotenant did not stop; the device is in an unknown state");
            if(stream_)
                (void)hipStreamDestroy(stream_);
            if(control_)
                (void)hipHostFree(control_);
            if(stop_host_)
                (void)hipHostFree(stop_host_);
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
                check(hipHostMalloc(reinterpret_cast<void**>(&stop_host_),
                                    sizeof(int),
                                    hipHostMallocMapped | hipHostMallocCoherent));
                *control_   = 0;
                *stop_host_ = 0;
                int* device_control = nullptr;
                check(hipHostGetDevicePointer(
                    reinterpret_cast<void**>(&device_control), control_, 0));
                check(hipHostGetDevicePointer(
                    reinterpret_cast<void**>(&stop_dev_), stop_host_, 0));
                // Log only the first scope; the configuration is process-wide.
                static bool logged  = false;
                const bool  verbose = !logged;
                launch<true>(n_cus, max_occupancy, stream_, device_control, stop_dev_, verbose);
                wait_ready(control_, n_cus, 30, verbose);
                logged = true;
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

#endif // _WIN32
