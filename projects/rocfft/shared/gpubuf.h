// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_GPUBUF_H
#define ROCFFT_GPUBUF_H

#include "device_properties.h"
#include "rocfft_hip.h"
#include "sys_mem.h"
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <vector>

struct DEVICEBUF_MEM_USAGE : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct device_memory_accountant
{
public:
    // acquire a reference to a singleton of this struct
    static device_memory_accountant& singleton()
    {
        static device_memory_accountant instance;
        return instance;
    }

    const size_t num_devices() const
    {
        return mem_account_on_device.size();
    }

    void record_used_bytes(size_t allocation_size, int dev_id)
    {
        if(dev_id < 0 || static_cast<size_t>(dev_id) >= num_devices())
            throw std::invalid_argument(
                "Invalid device ID given to device memory accountant (recording).");
        std::unique_lock lock(dev_mem_account_mutex);
        auto&            dev_account = mem_account_on_device[dev_id];
        dev_account.used_bytes += allocation_size;
        if(dev_account.is_integrated_device)
            system_memory::singleton().record_used_bytes(allocation_size);
    }

    void release_used_bytes(size_t allocation_size, int dev_id)
    {
        if(dev_id < 0 || static_cast<size_t>(dev_id) >= num_devices())
            throw std::invalid_argument(
                "Invalid device ID given to device memory accountant (releasing).");
        std::unique_lock lock(dev_mem_account_mutex);
        auto&            dev_account = mem_account_on_device[dev_id];
        // prevent underflows
        dev_account.used_bytes -= std::min(dev_account.used_bytes, allocation_size);
        if(dev_account.is_integrated_device)
            system_memory::singleton().release_used_bytes(allocation_size);
    }

    std::vector<size_t> get_usable_bytes_all_devices()
    {
        auto                ndevices = num_devices();
        std::vector<size_t> ret(ndevices);
        for(size_t dev_id = 0; dev_id < ndevices; ++dev_id)
        {
            ret[dev_id] = get_usable_bytes(dev_id);
        }
        return ret;
    }

    size_t get_usable_bytes(int dev_id)
    {
        if(dev_id < 0 || static_cast<size_t>(dev_id) >= num_devices())
            throw std::invalid_argument(
                "Invalid device ID given to device memory accountant (observation).");
        std::shared_lock lock(dev_mem_account_mutex);
        const auto&      dev_account = mem_account_on_device[dev_id];

        if(dev_account.limit_bytes <= dev_account.used_bytes)
            return 0;

        auto usable_bytes = dev_account.limit_bytes - dev_account.used_bytes;
        if(dev_account.is_integrated_device)
            usable_bytes = std::min(usable_bytes, system_memory::singleton().get_usable_bytes());
        return usable_bytes;
    }

    void set_limit_bytes_for_all_devices(size_t limit_bytes)
    {
        std::unique_lock lock(dev_mem_account_mutex);
        for(auto& account : mem_account_on_device)
            account.limit_bytes = std::min(account.total_bytes, limit_bytes);
    }

    std::string get_details(int dev_id)
    {
        if(dev_id < 0 || static_cast<size_t>(dev_id) >= num_devices())
            throw std::invalid_argument(
                "Invalid device ID given to device memory accountant (detail query).");

        const auto        usable_bytes = get_usable_bytes(dev_id);
        const auto&       dev_account  = mem_account_on_device[dev_id];
        std::shared_lock  lock(dev_mem_account_mutex);
        std::stringstream ss;
        ss << "\tUsable device memory: " << byte_size_to_str(usable_bytes) << "\n"
           << "\tUsed device memory: " << byte_size_to_str(dev_account.used_bytes) << "\n"
           << "\tLimit on device memory usage: " << byte_size_to_str(dev_account.limit_bytes);
        if(dev_account.is_integrated_device)
        {
            constexpr bool use_double_tabs = true;
            ss << "\n"
               << "\tNOTE: integrated device, system memory details below:\n"
               << system_memory::singleton().get_details(use_double_tabs);
        }
        return ss.str();
    }

    size_t get_max_total_mem_on_devices() const
    {
        size_t ret = 0;
        for(const auto& account : mem_account_on_device)
            ret = std::max(ret, account.total_bytes);
        return ret;
    }

    size_t get_limit_bytes_on_device(int dev_id) const
    {
        if(dev_id < 0 || static_cast<size_t>(dev_id) >= num_devices())
            throw std::invalid_argument("Invalid device ID given to device memory accountant "
                                        "(querying limit bytes on device).");
        return mem_account_on_device[dev_id].limit_bytes;
    }

private:
    struct mem_account_t
    {
        mem_account_t(size_t total_dev_mem, bool integrated_device)
            : total_bytes(total_dev_mem)
            , limit_bytes(total_dev_mem)
            , used_bytes(0)
            , is_integrated_device(integrated_device)
        {
        }
        const size_t total_bytes;
        size_t       limit_bytes;
        size_t       used_bytes;
        const bool   is_integrated_device;
    };

    std::vector<mem_account_t> mem_account_on_device;

    mutable std::shared_mutex dev_mem_account_mutex;

    device_memory_accountant()
    {
        int  dev_count  = 0;
        auto hip_status = hipGetDeviceCount(&dev_count);
        if(hip_status != hipSuccess || dev_count <= 0)
        {
            throw std::runtime_error("Device count failed with code " + std::to_string(hip_status)
                                     + " in initialization of device memory account (dev_count = "
                                     + std::to_string(dev_count) + ").");
        }
        mem_account_on_device.reserve(dev_count);
        for(int dev_id = 0; dev_id < dev_count; dev_id++)
        {
            rocfft_scoped_device dev(dev_id);
            const auto           dev_prop = get_curr_device_prop();
            mem_account_on_device.emplace_back(dev_prop.totalGlobalMem, dev_prop.integrated);
        }
    }
};

// Simple RAII class for GPU buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class gpubuf_t
{
public:
    gpubuf_t() {}
    // buffers are movable but not copyable
    gpubuf_t(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(owned, other.owned);
        std::swap(bsize, other.bsize);
        std::swap(device, other.device);
        std::swap(is_managed_memory, other.is_managed_memory);
    }
    gpubuf_t& operator=(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(owned, other.owned);
        std::swap(bsize, other.bsize);
        std::swap(device, other.device);
        std::swap(is_managed_memory, other.is_managed_memory);
        return *this;
    }
    gpubuf_t(const gpubuf_t&) = delete;
    gpubuf_t& operator=(const gpubuf_t&) = delete;

    static gpubuf_t make_nonowned(T* p, size_t size_bytes = 0)
    {
        gpubuf_t ret;
        ret.owned             = false;
        ret.buf               = p;
        ret.bsize             = size_bytes;
        ret.is_managed_memory = false; // irrelevant if not owned
        return ret;
    }

    ~gpubuf_t()
    {
        free();
    }

    static bool use_alloc_managed()
    {
        return std::getenv("ROCFFT_MALLOC_MANAGED");
    }

    hipError_t alloc(const size_t size, bool make_it_shared = false)
    {
        free();
        // remember the device that was current as of alloc, so we can
        // free on the correct device
        auto ret = hipGetDevice(&device);
        if(ret != hipSuccess)
            return ret;

        if(size > device_memory_accountant::singleton().get_usable_bytes(device))
        {
            std::stringstream msg;
            msg << "Unauthorized device allocation (device ID: " << device << ").\n"
                << "\tRequested size is " << byte_size_to_str(size) << "\n"
                << device_memory_accountant::singleton().get_details(device);
            throw DEVICEBUF_MEM_USAGE{msg.str()};
        }

        bsize             = size;
        is_managed_memory = use_alloc_managed() || make_it_shared;
        ret = is_managed_memory ? hipMallocManaged(&buf, bsize) : hipMalloc(&buf, bsize);
        if(ret != hipSuccess)
        {
            buf   = nullptr;
            bsize = 0;
        }

        device_memory_accountant::singleton().record_used_bytes(bsize, device);

        return ret;
    }

    size_t size() const
    {
        return bsize;
    }

    void free()
    {
        if(buf != nullptr)
        {
            if(owned)
            {
                // free on the device we allocated on
                rocfft_scoped_device dev(device);
                (void)hipFree(buf);
                device_memory_accountant::singleton().release_used_bytes(bsize, device);
            }
            buf   = nullptr;
            bsize = 0;
        }
        owned = true;
    }

    // return a pointer to the allocated memory, offset by the
    // specified number of bytes
    T* data_offset(size_t offset_bytes = 0) const
    {
        void* ptr = static_cast<char*>(buf) + offset_bytes;
        return static_cast<T*>(ptr);
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // equality/bool tests
    bool operator==(std::nullptr_t n) const
    {
        return buf == n;
    }
    bool operator!=(std::nullptr_t n) const
    {
        return buf != n;
    }
    operator bool() const
    {
        return buf;
    }

private:
    // The GPU buffer
    void* buf = nullptr;
    // whether this object owns the 'buf' pointer (and hence needs to
    // free it)
    bool   owned             = true;
    bool   is_managed_memory = false;
    size_t bsize             = 0;
    int    device            = 0;
};

// default gpubuf that gives out void* pointers
typedef gpubuf_t<> gpubuf;
#endif
