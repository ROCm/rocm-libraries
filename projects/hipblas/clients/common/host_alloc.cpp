/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#ifdef WIN32
#include <windows.h>

#else
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#endif

#include <map>
#include <mutex>
#include <set>
#include <stdlib.h>

#include <hip/hip_runtime.h>

#include "hipblas_test.hpp"
#include "host_alloc.hpp"

// Under HSA_XNACK=1 on a discrete GPU (e.g. gfx942 MI300X), a plain pageable
// host allocation is part of the SVM address space: an H2D copy can pull its
// pages into VRAM, and a later CPU write (e.g. the OpenMP-parallel
// hipblas_init_matrix fill) then faults them back RAM<-VRAM through
// svm_migrate_to_ram. With many OpenMP threads faulting at once this serializes
// on the amdgpu SVM migration mutex (perf: ~98% osq_lock), turning client init
// into an hours-long stall. Integrated APUs (e.g. MI300A) have unified coherent
// memory and never hit this path.
//
// Pinned (page-locked) host memory is not pageable and not SVM-migratable, so
// these host-only buffers stay in RAM and the init writes never trigger
// migration -- while preserving full OpenMP parallelism. The pinned pool is
// finite, so allocation falls back to ordinary malloc/calloc when hipHostMalloc
// fails; free() therefore must know which allocator produced each pointer.
static bool host_use_pinned()
{
    // Allow opting out (e.g. to A/B against pageable memory) via env.
    static const bool disabled = [] {
        const char* e = getenv("HIPBLAS_CLIENT_NO_PINNED_HOST_ALLOC");
        return e && *e && *e != '0';
    }();
    return !disabled;
}

// light weight memory tracking for threshold limit on total use
static size_t                  mem_used{0};
static std::map<void*, size_t> mem_allocated;
static std::mutex              mem_mutex;

// Pointers allocated via hipHostMalloc; these are freed with hipHostFree.
// Guarded by mem_mutex (same critical sections as the tracking map).
static std::set<void*> pinned_ptrs;

// Allocate `size` bytes of pinned host memory, falling back to malloc/calloc on
// failure. When `zero` is true the buffer is zero-initialized (calloc semantics).
// A returned pinned pointer is recorded so free_ptr_use() releases it with
// hipHostFree instead of free.
static void* host_pinned_alloc(size_t size, bool zero)
{
    if(host_use_pinned())
    {
        void* ptr = nullptr;
        if(hipHostMalloc(&ptr, size, hipHostMallocDefault) == hipSuccess && ptr)
        {
            if(zero)
                memset(ptr, 0, size);
            {
                std::lock_guard<std::mutex> lock(mem_mutex);
                pinned_ptrs.insert(ptr);
            }
            return ptr;
        }
        // hipHostMalloc failed (pool exhausted, etc.): fall through to pageable.
    }
    return zero ? calloc(1, size) : malloc(size);
}

void alloc_ptr_use(void* ptr, size_t size)
{
    std::lock_guard<std::mutex> lock(mem_mutex);
    if(ptr)
    {
        mem_allocated[ptr] = size;
        mem_used += size;
    }
}

void free_ptr_use(void* ptr, bool call_free)
{
    bool pinned = false;
    {
        std::lock_guard<std::mutex> lock(mem_mutex);
        auto                        it = mem_allocated.find(ptr);

        if(ptr && it != mem_allocated.end())
        {
            mem_used -= it->second;
            mem_allocated.erase(it);
        }
        else if(ptr && call_free)
        {
            std::cerr << "Warning: Freeing untracked pointer " << ptr
                      << " - untracked memory released (potential double-free or memory corruption)"
                      << std::endl;
        }

        auto pit = pinned_ptrs.find(ptr);
        if(pit != pinned_ptrs.end())
        {
            pinned = true;
            pinned_ptrs.erase(pit);
        }
    }

    if(call_free)
    {
        // Pinned buffers must be released with hipHostFree, pageable with free.
        if(pinned)
            (void)hipHostFree(ptr);
        else
            free(ptr);
    }
}

size_t host_bytes_allocated()
{
    std::lock_guard<std::mutex> lock(mem_mutex);
    return mem_used;
}

//!
//! @brief Memory free helper.  Returns kB or -1 if unknown.
//!
ptrdiff_t host_bytes_available()
{
#ifdef WIN32

    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (ptrdiff_t)status.ullAvailPhys;

#else

    const int BUF_MAX = 1024;
    char      buf[BUF_MAX];

    ptrdiff_t n_bytes = -1; // unknown

    FILE* fp = popen("cat /proc/meminfo", "r");
    if(fp == NULL)
    {
        return n_bytes;
    }

    static const char* mem_token     = "MemFree";
    static auto*       mem_free_type = getenv("HIPBLAS_CLIENT_ALLOC_AVAILABLE");
    if(mem_free_type)
    {
        mem_token = "MemAvail"; // MemAvailable
    }
    int mem_token_len = strlen(mem_token);

    while(fgets(buf, BUF_MAX, fp) != NULL)
    {
        // set env HIPBLAS_CLIENT_ALLOC_AVAILABLE to use MemAvailable if too many SKIPS occur
        if(!strncmp(buf, mem_token, mem_token_len))
        {
            sscanf(buf, "%*s %td", &n_bytes); // kB assumed as 3rd column and ignored
            n_bytes *= 1024;
            break;
        }
    }

    int status = pclose(fp);
    if(status == -1)
    {
        return -1;
    }
    else
    {
        return n_bytes;
    }

#endif
}

bool host_mem_safe(size_t n_bytes)
{
#if defined(HIPBLAS_BENCH)
    return true; // roll out to hipblas-bench when CI does perf testing
#else
    static auto* no_alloc_check = getenv("HIPBLAS_CLIENT_NO_ALLOC_CHECK");
    if(no_alloc_check)
    {
        return true;
    }

    constexpr size_t threshold = 100 * 1024 * 1024; // 100 MB

    static size_t client_ram_limit = 0;

    static int once = [&] {
        auto* alloc_limit = getenv("HIPBLAS_CLIENT_RAM_GB_LIMIT");
        if(alloc_limit)
        {
            size_t mem_limit;
            client_ram_limit = sscanf(alloc_limit, "%zu", &mem_limit) == 1 ? mem_limit : 0;
            client_ram_limit <<= 30; // B to GB
        }
        return 0;
    }();

    if(n_bytes > threshold)
    {
        if(client_ram_limit)
        {
            if(host_bytes_allocated() + n_bytes > client_ram_limit)
            {
                std::cout << "Warning: skipped allocating " << n_bytes << " bytes ("
                          << (n_bytes >> 30) << " GB) as total would be more than client limit ("
                          << (client_ram_limit >> 30) << " GB)" << std::endl;

                return false;
            }
        }

        ptrdiff_t avail_bytes = host_bytes_available(); // negative if unknown
        if(avail_bytes >= 0 && n_bytes > avail_bytes)
        {
            std::cout << "Warning: skipped allocating " << n_bytes << " bytes (" << (n_bytes >> 30)
                      << " GB) as more than free memory (" << (avail_bytes >> 30) << " GB)"
                      << std::endl;

            // we don't try if it looks to push load into swap
            return false;
        }
    }
    return true;
#endif
}

void* host_malloc(size_t size)
{
    if(host_mem_safe(size))
    {
        void* ptr = host_pinned_alloc(size, false);

        static int value = -1;

        static auto once = false;
        if(!once)
        {
            auto* alloc_byte_str = getenv("HIPBLAS_CLIENT_ALLOC_FILL_HEX_BYTE");
            if(alloc_byte_str)
            {
                value = strtol(alloc_byte_str, nullptr, 16); // hex
            }
            once = true;
        }

        if(value != -1 && ptr)
            memset(ptr, value, size);

        alloc_ptr_use(ptr, size);

        return ptr;
    }
    else
        return nullptr;
}

void* host_calloc(size_t nmemb, size_t size)
{
    if(host_mem_safe(nmemb * size))
    {
        void* ptr = host_pinned_alloc(nmemb * size, true);
        alloc_ptr_use(ptr, nmemb * size);
        return ptr;
    }
    else
        return nullptr;
}

void host_free(void* ptr)
{
    free_ptr_use(ptr, true);
}
