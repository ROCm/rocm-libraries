/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
 *******************************************************************************/

#pragma once

#include "datatype_interface.hpp"
#include "hipblaslt_arguments.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_test.hpp"
#include "singletons.hpp"
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <hipblaslt/hipblaslt.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#define MEM_MAX_GUARD_PAD 8192
#define MAX_DTYPE_SIZE sizeof(double)

/* ============================================================================================ */
/*! \brief  (abstract class) wrapper around a pointer to hip device or pinned host memory, including allocation size in bytes */
class hip_memory
{
public:
    size_t bytes() const
    {
        return m_size;
    }
    size_t capacity() const
    {
        return m_capacity;
    }

    void resize(size_t s)
    {
        assert(s <= m_capacity);
        m_size = s;
    }

    bool is_managed() const
    {
        return m_managed;
    }

    bool operator<(size_t s) const
    {
        return capacity() < s;
    }

    // Smallest remaining allowance across this process's cgroup hierarchy, or SIZE_MAX
    // when nothing limits it. Needed because sysinfo().freeram is host-wide: inside a
    // MemoryMax scope the kernel will happily pin pages the scope cannot afford and then
    // OOM-kill us, so an allocation certain to be killed looks like one that fits.
    static size_t cgroup_available_memory()
    {
#ifdef __linux__
        // A v1 limit reads as a huge sentinel rather than the word "max".
        size_t const unlimited = static_cast<size_t>(1) << 62;

        auto read_size = [](std::string const& path, size_t& out) -> bool {
            FILE* f = fopen(path.c_str(), "r");
            if(!f)
                return false;
            char token[64] = {};
            bool ok        = fscanf(f, "%63s", token) == 1;
            fclose(f);
            if(!ok)
                return false;
            if(strcmp(token, "max") == 0)
            {
                out = std::numeric_limits<size_t>::max();
                return true;
            }
            char*              end   = nullptr;
            unsigned long long value = strtoull(token, &end, 10);
            if(end == token)
                return false;
            out = static_cast<size_t>(value);
            return true;
        };

        std::string v2, v1;
        if(FILE* f = fopen("/proc/self/cgroup", "r"))
        {
            char line[512];
            while(fgets(line, sizeof(line), f))
            {
                std::string entry(line);
                while(!entry.empty() && (entry.back() == '\n' || entry.back() == '\r'))
                    entry.pop_back();
                if(entry.compare(0, 3, "0::") == 0)
                    v2 = entry.substr(3);
                else
                {
                    // "<id>:<controllers>:<path>", where controllers may be co-mounted
                    // and comma-joined, so match a whole token rather than ":memory:".
                    auto ids = entry.find(':');
                    auto sep = entry.find(':', ids == std::string::npos ? 0 : ids + 1);
                    if(ids != std::string::npos && sep != std::string::npos)
                    {
                        std::string list = entry.substr(ids + 1, sep - ids - 1);
                        for(size_t at = 0; at <= list.size();)
                        {
                            auto end = list.find(',', at);
                            if(end == std::string::npos)
                                end = list.size();
                            if(list.compare(at, end - at, "memory") == 0)
                            {
                                v1 = entry.substr(sep + 1);
                                break;
                            }
                            at = end + 1;
                        }
                    }
                }
            }
            fclose(f);
        }

        size_t available = std::numeric_limits<size_t>::max();

        auto consider = [&](std::string const& dir, char const* limit, char const* usage) {
            size_t cap = 0, used = 0;
            if(!read_size(dir + "/" + limit, cap) || cap >= unlimited)
                return;
            if(!read_size(dir + "/" + usage, used))
                used = 0;
            available = std::min(available, cap > used ? cap - used : static_cast<size_t>(0));
        };

        // A MemoryMax may sit on any ancestor, so the effective allowance is the tightest.
        auto walk = [&](std::string const& base,
                        std::string const& rel,
                        char const*        limit,
                        char const*        usage) {
            if(rel.empty())
                return;
            std::string dir = base + rel;
            for(;;)
            {
                consider(dir, limit, usage);
                if(dir.size() <= base.size())
                    break;
                auto slash = dir.rfind('/');
                if(slash == std::string::npos || slash < base.size())
                    break;
                dir.erase(slash);
            }
        };

        walk("/sys/fs/cgroup", v2, "memory.max", "memory.current");
        walk("/sys/fs/cgroup/memory", v1, "memory.limit_in_bytes", "memory.usage_in_bytes");
        return available;
#else
        return std::numeric_limits<size_t>::max();
#endif
    }

    size_t get_available_host_memory()
    {
#ifdef __linux__
        struct sysinfo info;
        if(sysinfo(&info) == 0)
        {
            // In the linux system, the host memory(pinned memory)'s capacity is the same as the free memory
            // ... except under a cgroup, where freeram describes the machine and not the
            // budget this process is held to. Take the tighter of the two so an allocation
            // that would be killed is reported as one that does not fit, which lets the
            // pool-clearing retry in memory_pool::get() engage.
            return std::min(static_cast<size_t>(info.freeram), cgroup_available_memory());
        }
        else
        {
            hipblaslt_cerr << "Error getting available host memory" << std::endl;
            return 0;
        }
#elif defined(_WIN32)
        MEMORYSTATUSEX memStatus = {};
        memStatus.dwLength = sizeof(memStatus);
        if(GlobalMemoryStatusEx(&memStatus))
        {
            // In the windows system, the host memory's capacity is the half of the physical memory
            // And previous allocation of host memory is got cleared before
            return memStatus.ullTotalPhys / 2;
        }
        else
        {
            hipblaslt_cerr << "Error getting available host memory" << std::endl;
            return 0;
        }
#else
        hipblaslt_cerr << "Error getting available host memory: unsupported platform" << std::endl;
        return 0;
#endif
    }

protected:
    hip_memory(size_t size, size_t capacity, bool use_HMM = false, size_t allocated_capacity = 0)
        : m_size(size)
        , m_capacity(capacity)
        , m_managed(use_HMM)
        , m_allocated_capacity(allocated_capacity)
    {
    }
    virtual ~hip_memory() = default;

    size_t m_size     = 0;
    size_t m_capacity = 0;
    bool   m_managed  = false;
    size_t m_allocated_capacity = 0;
};

/* ============================================================================================ */
/*! \brief  wrapper around a pointer to device memory, including allocation size in bytes */
class d_memory : public hip_memory
{
public:
    d_memory()
        : hip_memory(0, 0, false)
    {
    }

    d_memory(size_t size, size_t capacity, bool use_HMM = false)
        : hip_memory(size, capacity, use_HMM)
    {
        char* d = nullptr;

        if(use_HMM)
        {
            size_t available_host_memory = get_available_host_memory();
            // Need to ensure sufficient host memory, otherwise hipMallocManaged may OOM and hip api won't return error code,
            // and will cause the gtest get aborted
            if(available_host_memory < capacity || hipMallocManaged(&d, capacity) != hipSuccess)
            {
                hipblaslt_cerr << "Error allocating (" << (capacity >> 30) << " GB) unified memory, m_size is (" << (m_size >> 30) << " GB)"
                               << std::endl;
                d      = nullptr;
                m_size = m_capacity = 0;
            }
        }
        else if(hipMalloc(&d, capacity) != hipSuccess)
        {
            size_t free_device_mem, total_device_mem;
            (void)hipMemGetInfo(&free_device_mem, &total_device_mem);
            hipblaslt_cerr << "Insufficient device memory to allocate (" << (m_size >> 30) << " GB) as the available device memory is (" << (free_device_mem >> 30) << " GB) "
                           << std::endl;
            d      = nullptr;
            m_size = m_capacity = 0;
        }
        m_d.reset(d);
    }

    char* get()
    {
        return m_d.get();
    }
    const char* get() const
    {
        return m_d.get();
    }

private:
    std::unique_ptr<char, decltype(&hipFree)> m_d{nullptr, &hipFree};
};

/* ============================================================================================ */
/*! \brief  wrapper around a pointer to pinned host memory (hipHostMalloc), including allocation size in bytes */
class h_memory : public hip_memory
{
public:
    h_memory()
        : hip_memory(0, 0, false)
    {
    }

    h_memory(size_t size, size_t capacity, bool use_HMM = false)
        : hip_memory(size, capacity, false)
    {
        char* d = nullptr;

        size_t available_host_memory = get_available_host_memory();
        // Need to ensure sufficient host memory, otherwise hipHostMalloc may OOM and hip api won't return error code,
        // and will cause the gtest get aborted
        if(available_host_memory < capacity || hipHostMalloc(&d, capacity) != hipSuccess)
        {
            hipblaslt_cerr << "Error allocating (" << (capacity >> 30) << " GB) host memory, m_size is (" << (m_size >> 30) << " GB)"
                           << std::endl;
            d      = nullptr;
            m_size = m_capacity = 0;
        }
        m_d.reset(d);
    }

    char* get()
    {
        return m_d.get();
    }
    const char* get() const
    {
        return m_d.get();
    }

private:
    std::unique_ptr<char, decltype(&hipHostFree)> m_d{nullptr, &hipHostFree};
};

/* ============================================================================================ */
/*! \brief  memory pool class to keep track of memory in either M = d_memory, or M = h_memory objects */
template <typename M>
class memory_pool
{
public:
    static M Get(size_t m_bytes, bool use_HMM = false)
    {
        std::lock_guard<std::mutex> lock(Instance().m_mutex);
        return Instance().get(m_bytes, use_HMM);
    }

    static void Restore(M& dm)
    {
        std::lock_guard<std::mutex> lock(Instance().m_mutex);
        Instance().restore(dm);
    }

private:
    std::vector<M> m_pool, m_pool_managed;
    std::mutex m_mutex;

    static memory_pool& Instance()
    {
        static memory_pool buffer;
        return buffer;
    }

    M get(size_t bytes, bool use_HMM = false)
    {
        auto& pool = use_HMM ? m_pool_managed : m_pool;
        
        // For Windows system with not enough system memory, 
        // not suitable for memory pool management when it needs another allocation
        #ifdef _WIN32
        MEMORYSTATUSEX memStatus = {};
        memStatus.dwLength = sizeof(memStatus);
        if(GlobalMemoryStatusEx(&memStatus))
        {
            // If the shared memory is less than 64GB(128 / 2), may not enough for the hipblaslt-test to run
            if(memStatus.ullTotalPhys <= (128ULL << 30))
            {
                pool.clear();
            }
        }
        #endif

        auto  it   = std::lower_bound(pool.begin(), pool.end(), bytes);
        if(it != pool.end() && // found a buffer that is large enough ..
           it->capacity() < 2 * bytes) // but not way too large
        {
            auto p = std::move(*it);
            p.resize(bytes);
            pool.erase(it);
            return p;
        }
        else
        {
            size_t free_device_mem, total_device_mem;
            (void)hipMemGetInfo(&free_device_mem, &total_device_mem);

            // Threshold for "huge" allocations — skip 20% extra allocation
            const size_t big_thresh = total_device_mem / 4;
            bool huge_request = (bytes >= big_thresh);

            // remove the (largest) buffer that was too small
            if(it != pool.begin())
                pool.erase(it - 1);

            // Allocate 20% extra if it is not huge_request for later reuse
            size_t alloc_capacity = huge_request ? bytes : static_cast<size_t>(bytes * 1.2);

            auto e = M(bytes, alloc_capacity, use_HMM);
            if(e.get())
                return e;
            hipblaslt_cerr << "Clearing memory pool and retrying" << std::endl;
            // allocation failed, so clear the pool and try again (without the 20%)
            pool.clear();

            // reset the error code from previous hipMalloc failure and try again to allocate memory
            hipError_t err = hipPeekAtLastError();
            if(err == hipErrorOutOfMemory || err == hipErrorMemoryAllocation )
                (void)hipGetLastError();

            auto retry = M(bytes, bytes, use_HMM);
            // Host allocations are unchecked by the callers -- memcheck() exists only on
            // the device buffer class -- so returning an empty one here means a null
            // pointer reaches as<T>() and gets written through. Fail attributably instead.
            if(!retry.get() && std::is_same<M, h_memory>::value)
            {
                hipblaslt_cerr << "Fatal: cannot allocate " << (bytes >> 20)
                               << " MiB of pinned host memory even with an empty pool. This "
                                  "problem does not fit in the memory available to this "
                                  "process."
                               << std::endl;
                exit(EXIT_FAILURE);
            }
            return retry;
        }
    }

    void restore(M& dm)
    {
        if(!dm.get() || !dm.capacity())
            return;
        auto& pool = dm.is_managed() ? m_pool_managed : m_pool;
        // insert in (sorted) pool
        pool.insert(std::lower_bound(pool.begin(), pool.end(), dm.capacity()), std::move(dm));
    }
};

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T>
class d_vector
{
private:
    size_t   m_size;
    size_t   m_pad, m_guard_len;
    size_t   m_bytes;
    d_memory m_mem;

    static bool m_init_guard;

protected:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    static T m_guard[MEM_MAX_GUARD_PAD];

#ifdef GOOGLE_TEST
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        , m_bytes((s + m_pad * 2) * sizeof(T))
        , use_HMM(HMM)
    {
        // Initialize m_guard with random data
        if(!m_init_guard)
        {
            hipblaslt_init_nan(m_guard, MEM_MAX_GUARD_PAD);
            m_init_guard = true;
        }
    }
#else
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * sizeof(T))
        , m_bytes(s ? s * sizeof(T) : sizeof(T))
        , use_HMM(HMM)
    {
    }
#endif

    T* device_vector_setup()
    {
        m_mem = memory_pool<d_memory>::Get(m_bytes, use_HMM);
        T* d  = reinterpret_cast<T*>(m_mem.get());
#ifdef GOOGLE_TEST
        if(d)
        {
            if(m_guard_len > 0)
            {
                // Copy m_guard to device memory before allocated memory
                EXPECT_EQ(hipMemcpy(d, m_guard, m_guard_len, hipMemcpyHostToDevice), hipSuccess);

                // Point to allocated block
                d += m_pad;

                // Copy m_guard to device memory after allocated memory
                EXPECT_EQ(hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyHostToDevice),
                          hipSuccess);
            }
        }
#endif
        return d;
    }

    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(m_guard_len > 0)
        {
            T* host = new T[m_pad];

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host, d + this->m_size, m_guard_len, hipMemcpyDeviceToHost),
                      hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

            // Point to m_guard before allocated memory
            d -= m_pad;

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

            delete[] host;
        }
#endif
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(m_pad > 0)
            {
                T* host = new T[m_pad];

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host, d + this->m_size, m_guard_len, hipMemcpyDeviceToHost),
                          hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

                // Point to m_guard before allocated memory
                d -= m_pad;

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

                delete[] host;
            }
#endif
        }
        memory_pool<d_memory>::Restore(m_mem);
    }
};

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
class d_vector_type
{
private:
    size_t      m_size;
    hipDataType m_dtype;
    size_t      m_pad, m_guard_len;
    size_t      m_bytes;
    d_memory    m_mem;

    inline static bool m_init_guard_type;

protected:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    inline static char m_guard_type[MEM_MAX_GUARD_PAD * MAX_DTYPE_SIZE];

#ifdef GOOGLE_TEST
    d_vector_type(hipDataType dtype, size_t s, bool HMM = false)
        : m_size(s)
        , m_dtype(dtype)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * realDataTypeSize(dtype))
        , m_bytes((s + m_pad * 2) * realDataTypeSize(dtype))
        , use_HMM(HMM)
    {
        // Initialize m_guard with random data
        if(!m_init_guard_type)
        {
            hipblaslt_init_nan(m_guard_type, MEM_MAX_GUARD_PAD);
            m_init_guard_type = true;
        }
    }
#else
    d_vector_type(hipDataType dtype, size_t s, bool HMM = false)
        : m_size(s)
        , m_dtype(dtype)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * realDataTypeSize(dtype))
        , m_bytes(s ? s * realDataTypeSize(dtype) : realDataTypeSize(dtype))
        , use_HMM(HMM)
    {
    }
#endif

    char* device_vector_setup()
    {
        m_mem   = memory_pool<d_memory>::Get(m_bytes, use_HMM);
        char* d = m_mem.get();
#ifdef GOOGLE_TEST
        if(d)
        {
            if(m_guard_len > 0)
            {
                // Copy m_guard to device memory before allocated memory
                EXPECT_EQ(hipMemcpy(d, m_guard_type, m_guard_len, hipMemcpyHostToDevice),
                          hipSuccess);

                // Point to allocated block
                d += m_pad * realDataTypeSize(m_dtype);

                // Copy m_guard to device memory after allocated memory
                EXPECT_EQ(hipMemcpy(d + this->m_size * realDataTypeSize(m_dtype),
                                    m_guard_type,
                                    m_guard_len,
                                    hipMemcpyHostToDevice),
                          hipSuccess);
            }
        }
#endif
        return d;
    }

    void device_vector_check(char* d)
    {
#ifdef GOOGLE_TEST
        if(m_guard_len > 0)
        {
            char* host = new char[m_pad * realDataTypeSize(m_dtype)];

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host,
                                d + this->m_size * realDataTypeSize(m_dtype),
                                m_guard_len,
                                hipMemcpyDeviceToHost),
                      hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

            // Point to m_guard before allocated memory
            d -= m_pad * realDataTypeSize(m_dtype);

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

            delete[] host;
        }
#endif
    }

    void device_vector_teardown(char* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(m_pad > 0)
            {
                char* host = new char[m_pad * realDataTypeSize(m_dtype)];

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host,
                                    d + this->m_size * realDataTypeSize(m_dtype),
                                    m_guard_len,
                                    hipMemcpyDeviceToHost),
                          hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

                // Point to m_guard before allocated memory
                d -= m_pad * realDataTypeSize(m_dtype);

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

                delete[] host;
            }
#endif
        }
        memory_pool<d_memory>::Restore(m_mem);
    }
};

template <typename T>
T d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

template <typename T>
bool d_vector<T>::m_init_guard = false;

#undef MEM_MAX_GUARD_PAD
#undef MAX_DTYPE_SIZE
