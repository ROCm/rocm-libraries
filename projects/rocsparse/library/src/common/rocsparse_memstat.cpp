/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifdef ROCSPARSE_WITH_MEMSTAT

#include "rocsparse_memstat.hpp"
#include "rocsparse-types.h"
#include "rocsparse_control.hpp"
#include "rocsparse_envariables.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

//
// STATIC UTILITY METHODS
//
static std::string relfilename(const char* tag_)
{
    //
    // Need to update to c++17 and use filesystem.
    //
    std::string tag(tag_);
    std::string thisfilename(__FILE_NAME__);
    std::string thispath(__FILE__);

    thispath = thispath.substr(0, thispath.size() - thisfilename.size());
    thispath = thispath.substr(0, thispath.size() - 12); // 12 = std::string("library/src/").size()
    auto res = tag.substr(thispath.size(), tag.size() - thispath.size());

    //
    // ..
    //
    bool v = true;
    while(v)
    {
        v = false;
        for(int i = 0; i < res.size(); ++i)
        {
            if(i > 1 && res[i] == '.' && (i + 1 < res.size()) && res[i + 1] == '.')
            {
                for(int j = i - 2; j >= 0; --j)
                {
                    if(res[j] == '/' || res[j] == '\\')
                    {
                        int k = j + 1;
                        res   = res.substr(0, k) + res.substr(i + 3);
                        v     = true;
                        break;
                    }
                }
                if(v)
                {
                    break;
                }
            }
        }
    }
    return res;
}

static double get_time_us(void)
{
    std::ignore = hipDeviceSynchronize();
    auto now    = std::chrono::steady_clock::now();
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
}

//
// ENUMERATE ALLOCATION MODE.
//
struct memstat_mode
{
    typedef enum _value
    {
        device = 0,
        host,
        managed
    } value_t;
    static constexpr size_t  size  = 3;
    static constexpr value_t all[] = {device, host, managed};
    static const char*       to_string(const value_t v)
    {
        switch(v)
        {
        case device:
            return "device";
        case host:
            return "host";
        case managed:
            return "managed";
        }
    };

    static inline hipError_t get_hipMemcpyKind(hipMemcpyKind&        kind,
                                               memstat_mode::value_t TARGET,
                                               memstat_mode::value_t SOURCE)
    {
        switch(TARGET)
        {
        case memstat_mode::host:
        {
            switch(SOURCE)
            {
            case memstat_mode::host:
            {
                kind = hipMemcpyHostToHost;
                return hipSuccess;
            }
            case memstat_mode::device:
            {
                kind = hipMemcpyDeviceToHost;
                return hipSuccess;
            }
            case memstat_mode::managed:
            {
                kind = hipMemcpyHostToHost;
                return hipSuccess;
            }
            }
        }
        case memstat_mode::device:
        {
            switch(SOURCE)
            {
            case memstat_mode::host:
            {
                kind = hipMemcpyHostToDevice;
                return hipSuccess;
            }
            case memstat_mode::device:
            {
                kind = hipMemcpyDeviceToDevice;
                return hipSuccess;
            }
            case memstat_mode::managed:
            {
                kind = hipMemcpyDeviceToDevice;
                return hipSuccess;
            }
            }
        }
        case memstat_mode::managed:
        {
            switch(SOURCE)
            {
            case memstat_mode::host:
            {
                kind = hipMemcpyHostToHost;
                return hipSuccess;
            }
            case memstat_mode::managed:
            {
                kind = hipMemcpyHostToHost;
                return hipSuccess;
            }
            case memstat_mode::device:
            {
                kind = hipMemcpyDeviceToDevice;
                return hipSuccess;
            }
            }
        }
        }
        return hipErrorInvalidValue;
    }
};

template <memstat_mode::value_t MODE>
struct memstat_allocator
{
private:
    static constexpr size_t PAD = 4096;

    static size_t     compute_nbytes(size_t s);
    static void       init_guards(char* A, size_t N);
    static char*      off_guards(char* d);
    static hipError_t install_guards(void* d, size_t size, void** res);
    static hipError_t install_guards_async(void* d, size_t size, void** res, hipStream_t stream);

public:
    static hipError_t malloc(void** mem, size_t size);
    static hipError_t malloc_async(void** mem, size_t size, hipStream_t stream);
    static hipError_t check_guards(char* d, size_t size);
    static hipError_t free(void* d_);
    static hipError_t free_async(void* d_, hipStream_t stream);
};

class memstat
{
public:
    static bool s_enabled;
    static bool s_force_managed;
    static bool s_guards_enabled;

    static memstat& instance();

private:
    memstat();
    bool check() const;
    //
    // Destructor.
    //
    ~memstat()
    {
        if(s_enabled)
        {
            if(this->m_map.size() > 0)
            {
                std::cerr << "rocsparse memstat memory leaks detected, use Python script "
                             "'rocsparse-memstat.py' to postprocess file '"
                          << this->m_report_filename << "'" << std::endl;
            }
            //
            // Force flush.
            //
            this->flush_report(true);
        }
    };

    memstat(const memstat&) = delete;
    memstat& operator=(const memstat&) = delete;

    struct stat
    {
        size_t                index;
        size_t                nbytes;
        memstat_mode::value_t mode;
        const char*           kind;
        size_t                total_nbytes[memstat_mode::size];
        const char*           tag;
        double                t;
    };

    //
    // map for hashing, vector for sorted events.
    //
    std::map<void*, stat> m_map;
    std::vector<stat>     m_data;
    double                m_start_time;
    size_t                m_total_nbytes[memstat_mode::size]{};
    int                   m_next_flush_report{};
    std::string           m_report_filename;

public:
    void set_filename(const char* filename)
    {
        if(this->m_next_flush_report > 0)
        {
            //
            // Rename the file.
            //
            std::rename(this->m_report_filename.c_str(), filename);
        }
        this->m_report_filename = std::string(filename);
    }

    void add(void* address, size_t nbytes, memstat_mode::value_t mode, const char* tag);
    void remove(void* address, const char* tag);
    bool contains(void* address) const;
    void flush_report(bool finalize = false);

private:
    void report(std::ostream& out) const;
    void report_legend(std::ostream& out) const;
    void flush_report(std::ostream& out, bool finalize) const;
};

//
// This forces instantation (but that's not the purpose of it).
//
bool memstat::s_enabled = ROCSPARSE_ENVARIABLES.get(rocsparse::envariables::MEMSTAT);
bool memstat::s_force_managed
    = ROCSPARSE_ENVARIABLES.get(rocsparse::envariables::MEMSTAT_FORCE_MANAGED);
bool memstat::s_guards_enabled = ROCSPARSE_ENVARIABLES.get(rocsparse::envariables::MEMSTAT_GUARDS);

template <memstat_mode::value_t MODE>
size_t memstat_allocator<MODE>::compute_nbytes(size_t s)
{
    return (memstat::s_guards_enabled) ? ((s + memstat_allocator<MODE>::PAD * 3)) : s;
}

template <memstat_mode::value_t MODE>
void memstat_allocator<MODE>::init_guards(char* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = i;
}

template <memstat_mode::value_t MODE>
char* memstat_allocator<MODE>::off_guards(char* d)
{
    if(PAD > 0)
    {
        d = (((char*)d) - 2 * PAD);
    }
    return d;
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::install_guards(void* d_, size_t size, void** res)
{
    char* d = static_cast<char*>(d_);
    if(d != nullptr)
    {
        if(PAD > 0)
        {
            char guard[PAD];
            init_guards(guard, PAD);
            hipMemcpyKind kind_transfer;
            hipError_t    err
                = memstat_mode::get_hipMemcpyKind(kind_transfer, MODE, memstat_mode::host);
            if(err != hipSuccess)
            {
                return err;
            }

            // Copy guard to device memory before allocated memory
            err = hipMemcpy(d, guard, sizeof(guard), kind_transfer);
            if(err != hipSuccess)
            {
                return err;
            }

            err = hipMemcpy(d + PAD, guard, sizeof(guard), kind_transfer);
            if(err != hipSuccess)
            {
                return err;
            }

            // Point to allocated block
            d += 2 * PAD;

            // Copy guard to device memory after allocated memory
            err = hipMemcpy(d + size, guard, sizeof(guard), kind_transfer);
            if(err != hipSuccess)
            {
                return err;
            }
        }
    }
    res[0] = static_cast<void*>(d);
    return hipSuccess;
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::install_guards_async(void*       d_,
                                                         size_t      size,
                                                         void**      res,
                                                         hipStream_t stream)
{
    char* d = static_cast<char*>(d_);
    if(d != nullptr)
    {
        if(PAD > 0)
        {
            char guard[PAD];
            init_guards(guard, PAD);
            hipMemcpyKind kind_transfer;
            hipError_t    err
                = memstat_mode::get_hipMemcpyKind(kind_transfer, MODE, memstat_mode::host);
            if(err != hipSuccess)
            {
                return err;
            }

            // Copy guard to device memory before allocated memory
            err = hipMemcpyAsync(d, guard, sizeof(guard), kind_transfer, stream);
            if(err != hipSuccess)
            {
                return err;
            }

            err = hipMemcpyAsync(d + PAD, guard, sizeof(guard), kind_transfer, stream);
            if(err != hipSuccess)
            {
                return err;
            }

            // Point to allocated block
            d += 2 * PAD;

            // Copy guard to device memory after allocated memory
            err = hipMemcpyAsync(d + size, guard, sizeof(guard), kind_transfer, stream);
            if(err != hipSuccess)
            {
                return err;
            }
        }
    }
    res[0] = static_cast<void*>(d);
    return hipSuccess;
}

hipError_t memstat_allocator_malloc(void** mem, size_t size, memstat_mode::value_t mode)
{
    switch(mode)
    {
    case memstat_mode::host:
    {
        return memstat_allocator<memstat_mode::host>::malloc(mem, size);
    }
    case memstat_mode::device:
    {
        return memstat_allocator<memstat_mode::device>::malloc(mem, size);
    }
    case memstat_mode::managed:
    {
        return memstat_allocator<memstat_mode::managed>::malloc(mem, size);
    }
    }
    return hipErrorInvalidValue;
}

hipError_t memstat_allocator_free(void* mem, memstat_mode::value_t mode)
{
    switch(mode)
    {
    case memstat_mode::host:
    {
        return memstat_allocator<memstat_mode::host>::free(mem);
    }
    case memstat_mode::device:
    {
        return memstat_allocator<memstat_mode::device>::free(mem);
    }
    case memstat_mode::managed:
    {
        return memstat_allocator<memstat_mode::managed>::free(mem);
    }
    }
    return hipErrorInvalidValue;
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::malloc(void** mem, size_t nbytes_)
{
    if(mem == nullptr)
    {
        return hipErrorInvalidValue;
    }
    if(nbytes_ == 0)
    {
        mem[0] = nullptr;
        return hipSuccess;
    }

    size_t nbytes = compute_nbytes(nbytes_);

    hipError_t err = hipErrorInvalidValue;
    switch(MODE)
    {
    case memstat_mode::host:
    {
        err = hipHostMalloc(mem, nbytes);
        break;
    }
    case memstat_mode::device:
    {
        err = hipMalloc(mem, nbytes);
        break;
    }
    case memstat_mode::managed:
    {
        err = hipMallocManaged(mem, nbytes);
        break;
    }
    }

    if(err != hipSuccess)
    {
        fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
        return err;
    }

    if(memstat::s_guards_enabled)
    {
        void* p;
        err = install_guards(mem[0], nbytes_, &p);
        if(err != hipSuccess)
        {
            return err;
        }
        mem[0] = p;
    }

    return hipSuccess;
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::malloc_async(void** mem, size_t nbytes_, hipStream_t stream)
{
    if(mem == nullptr)
    {
        return hipErrorInvalidValue;
    }
    if(nbytes_ == 0)
    {
        mem[0] = nullptr;
        // hipMallocAsync returns hipErrorInvalidValue when size is 0
        return hipErrorInvalidValue;
    }

    size_t nbytes = compute_nbytes(nbytes_);

    hipError_t err = hipErrorInvalidValue;
    switch(MODE)
    {
    case memstat_mode::host:
    {
        err = hipErrorInvalidValue;
        break;
    }
    case memstat_mode::device:
    {
#if HIP_VERSION >= 50300000
        err = hipMallocAsync(mem, nbytes, stream);
#else
        err = hipMalloc(mem, nbytes);
#endif
        break;
    }
    case memstat_mode::managed:
    {
        err = hipErrorInvalidValue;
        break;
    }
    }

    if(err != hipSuccess)
    {
        fprintf(stderr, "Error allocating %'zu bytes (%zu GB)\n", nbytes, nbytes >> 30);
        return err;
    }

    if(memstat::s_guards_enabled)
    {
        void* p;
        err = install_guards_async(mem[0], nbytes_, &p, stream);
        if(err != hipSuccess)
        {
            return err;
        }
        mem[0] = p;
    }

    return hipSuccess;
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::check_guards(char* d, size_t size)
{
    if(d != nullptr)
    {
        if(PAD > 0)
        {
            hipMemcpyKind kind_transfer;
            hipError_t    err
                = memstat_mode::get_hipMemcpyKind(kind_transfer, memstat_mode::host, MODE);
            if(err != hipSuccess)
            {
                return err;
            }
            char host[PAD], guard[PAD];
            // Copy device memory after allocated memory to host
            err = hipMemcpy(guard, d - 2 * PAD, sizeof(guard), kind_transfer);
            if(err != hipSuccess)
            {
                return err;
            }

            // Copy device memory after allocated memory to host
            err = hipMemcpy(host, d + size, sizeof(guard), kind_transfer);
            if(err != hipSuccess)
            {
                return err;
            }

            // Make sure no corruption has occurred
            if(memcmp(host, guard, sizeof(guard)) != 0)
            {
                FORWARD_WITH_MESSAGE_IF_HIP_ERROR(hipErrorUnknown, "Memory corruption detected");
            }

            // Point to guard before allocated memory
            d -= PAD;

            // Copy device memory after allocated memory to host
            err = hipMemcpy(host, d, sizeof(guard), kind_transfer);
            if(err != hipSuccess)
            {
                return err;
            }

            // Make sure no corruption has occurred
            if(memcmp(host, guard, sizeof(guard)) != 0)
            {
                FORWARD_WITH_MESSAGE_IF_HIP_ERROR(hipErrorUnknown, "Memory corruption detected");
            }
        }
        return hipSuccess;
    }
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::free(void* d_)
{
    char* d = static_cast<char*>(d_);
    if(d != nullptr)
    {
        if(memstat::s_guards_enabled)
        {
            d = off_guards(d);
        }

        switch(MODE)
        {
        case memstat_mode::host:
        {
            return hipHostFree(d);
        }
        case memstat_mode::managed:
        case memstat_mode::device:
        {
            // Free device memory
            return hipFree(d);
        }
        }
    }
    return hipSuccess;
}

template <memstat_mode::value_t MODE>
hipError_t memstat_allocator<MODE>::free_async(void* d_, hipStream_t stream)
{
    char* d = static_cast<char*>(d_);
    if(d != nullptr)
    {
        if(memstat::s_guards_enabled)
        {
            d = off_guards(d);
        }

        switch(MODE)
        {
        case memstat_mode::host:
        {
            return hipErrorInvalidValue;
        }
        case memstat_mode::managed:
        {
            return hipErrorInvalidValue;
        }
        case memstat_mode::device:
        {
            // Free device memory
#if HIP_VERSION >= 50300000
            if(d != nullptr)
            {
                return hipFreeAsync(d, stream);
            }
            else
            {
                return hipSuccess;
            }
#else
            return hipFree(d);
#endif
        }
        }
    }
    return hipSuccess;
}

hipError_t rocsparse_free(void* mem, memstat_mode::value_t mode, const char* tag)
{
    hipError_t err = hipErrorInvalidValue;
    switch(mode)
    {
    case memstat_mode::device:
    {
        err = memstat_allocator<memstat_mode::device>::free(mem);
        break;
    }

    case memstat_mode::host:
    {
        err = memstat_allocator<memstat_mode::host>::free(mem);
        break;
    }

    case memstat_mode::managed:
    {
        err = memstat_allocator<memstat_mode::managed>::free(mem);
        break;
    }
    }

    if(err != hipSuccess)
    {
        return err;
    }

    if(memstat::s_enabled)
    {
        memstat::instance().remove(mem, tag);
    }

    return hipSuccess;
}

hipError_t rocsparse_malloc(void** mem, size_t nbytes, memstat_mode::value_t mode, const char* tag)
{

    hipError_t err = hipErrorInvalidValue;
    switch(mode)
    {
    case memstat_mode::host:
    {
        err = memstat_allocator<memstat_mode::host>::malloc(mem, nbytes);
        break;
    }
    case memstat_mode::device:
    {
        err = memstat_allocator<memstat_mode::device>::malloc(mem, nbytes);
        break;
    }
    case memstat_mode::managed:
    {
        err = memstat_allocator<memstat_mode::managed>::malloc(mem, nbytes);
        break;
    }
    }

    if(err != hipSuccess)
    {
        return err;
    }

    if(memstat::s_enabled)
    {
        memstat::instance().add(mem[0], nbytes, mode, tag);
    }
    return hipSuccess;
}

hipError_t
    rocsparse_free_async(void* mem, hipStream_t stream, memstat_mode::value_t mode, const char* tag)
{
    hipError_t err = hipErrorInvalidValue;
    switch(mode)
    {
    case memstat_mode::host:
    {
        err = hipErrorInvalidValue;
        break;
    }
    case memstat_mode::device:
    {
        err = memstat_allocator<memstat_mode::device>::free_async(mem, stream);
        break;
    }
    case memstat_mode::managed:
    {
        err = hipErrorInvalidValue;
        break;
    }
    }

    if(err != hipSuccess)
    {
        return err;
    }

    if(memstat::s_enabled)
    {
        memstat::instance().remove(mem, tag);
    }

    return hipSuccess;
}

hipError_t rocsparse_malloc_async(
    void** mem, size_t nbytes, hipStream_t stream, memstat_mode::value_t mode, const char* tag)
{

    hipError_t err = hipErrorInvalidValue;
    switch(mode)
    {
    case memstat_mode::host:
    {
        err = hipErrorInvalidValue;
        break;
    }
    case memstat_mode::device:
    {
        err = memstat_allocator<memstat_mode::device>::malloc_async(mem, nbytes, stream);
        break;
    }
    case memstat_mode::managed:
    {
        err = hipErrorInvalidValue;
        break;
    }
    }

    if(err != hipSuccess)
    {
        return err;
    }

    if(memstat::s_enabled)
    {
        memstat::instance().add(mem[0], nbytes, mode, tag);
    }
    return hipSuccess;
}

memstat& memstat::instance()
{
    static memstat instance;
    return instance;
}

memstat::memstat()
    : m_report_filename("rocsparse_memstat.json")
{
    this->m_start_time = get_time_us();
};

void memstat::add(void* address, size_t nbytes, memstat_mode::value_t mode, const char* tag)
{
    if(address == nullptr)
        return;
    if(!contains(address))
    {
        double t = get_time_us();
        this->m_total_nbytes[mode] += nbytes;
        const size_t index = this->m_next_flush_report + (this->m_data.size() + 1);
        this->m_map[address]
            = {index,
               nbytes,
               mode,
               "malloc",
               {this->m_total_nbytes[0], this->m_total_nbytes[1], this->m_total_nbytes[2]},
               tag,
               t};
        this->m_data.push_back(
            {index,
             nbytes,
             mode,
             "malloc",
             {this->m_total_nbytes[0], this->m_total_nbytes[1], this->m_total_nbytes[2]},
             tag,
             t});
        memstat::instance().flush_report();
    }
    else
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                              "Address already exists in the memstat database.");
    }
}

void memstat::flush_report(std::ostream& out, bool finalize) const
{
    if(this->m_data.size() >= 128 || finalize)
    {
        const bool first = (0 == this->m_next_flush_report);
        if(first)
        {
            out << "{ " << std::endl;
            out << "\"legend\":";
            this->report_legend(out);
            out << "," << std::endl;
            out << "\"results\": [ " << std::endl;
        }
        else
        {
            out << ", " << std::endl;
        }
        this->report(out);
        if(finalize)
        {
            out << "], " << std::endl;
            //
            //
            //
            if(m_map.size() == 0)
            {
                out << "\"leaks\": []";
            }
            else
            {
                out << "\"leaks\": [";
                bool first = true;
                for(auto f : m_map)
                {
                    auto e = f.second;
                    if(!first)
                        out << "," << std::endl;
                    //
                    // Transform tag?
                    //
                    out << " { ";
                    out << "  \"index\": \"" << e.index << "\"";
                    out << ", "
                        << "  \"mode\": \"" << memstat_mode::to_string(e.mode) << "\""
                        << ", "
                        << "  \"op\"  : \"" << e.kind << "\""
                        << ", "
                        << "  \"nbytes\" : \"" << e.nbytes << "\""
                        << ", "
                        << "   \"tag\": \"" << relfilename(e.tag) << "\""
                        << " }";
                    first = false;
                }

                out << "]";
            }

            out << "}" << std::endl;
        }
    }
}

void memstat::flush_report(bool finalize)
{
    if(this->m_data.size() >= 128 || finalize)
    {
        const bool    first = (0 == this->m_next_flush_report);
        std::ofstream out(this->m_report_filename,
                          (first) ? std::ios_base::out : std::ios_base::app);
        this->flush_report(out, finalize);
        out.close();
        this->m_next_flush_report += rocsparse::min(m_data.size(), size_t(128));
        this->m_data.clear();
    }
}

void memstat::remove(void* address, const char* tag)
{
    if(address == nullptr)
        return;
    auto it = m_map.find(address);
    if(it != m_map.end())
    {
        this->m_total_nbytes[it->second.mode] -= it->second.nbytes;
        double       t     = get_time_us();
        const size_t index = this->m_next_flush_report + (this->m_data.size() + 1);
        this->m_data.push_back(
            {index,
             it->second.nbytes,
             it->second.mode,
             "free",
             {this->m_total_nbytes[0], this->m_total_nbytes[1], this->m_total_nbytes[2]},
             tag,
             t});
        memstat::instance().flush_report();
        m_map.erase(address);
    }
    else
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                              "Cannot remove address from the memstat database");
    }
}

bool memstat::contains(void* address) const
{
    return ((address != nullptr) && (m_map.find(address) != m_map.end()));
}

//
// Flush lines.
//
void memstat::report(std::ostream& out) const
{
    for(int i = 0; i < m_data.size(); ++i)
    {
        if(i > 0)
            out << "," << std::endl;
        out << " { ";
        out << "  \"index\": \"" << m_data[i].index << "\"";
        out << ", "
            << " \"time\": \"" << (m_data[i].t - m_start_time) / 1e3 << "\"";
        for(auto v : memstat_mode::all)
        {
            out << ", "
                << "\"nbytes_" << memstat_mode::to_string(v) << "\" : \""
                << m_data[i].total_nbytes[v] << "\"";
        }
        out << ", "
            << "  \"mode\": \"" << memstat_mode::to_string(m_data[i].mode) << "\""
            << ", "
            << "  \"op\"  : \"" << m_data[i].kind << "\""
            << ", "
            << "  \"nbytes\" : \"" << m_data[i].nbytes << "\""
            << ", "
            << "   \"tag\": \"" << relfilename(m_data[i].tag) << "\""
            << " }";
    }
}

void memstat::report_legend(std::ostream& out) const
{
    out << " [ "
        << "\"index\", \"time\"";
    for(auto v : memstat_mode::all)
    {
        out << ", "
            << "\"nbytes_" << memstat_mode::to_string(v) << "\"";
    }
    out << ", \"mode\""
        << ", "
        << "\"op\""
        << ", "
        << "\"nbytes\""
        << ", "
        << " \"tag\"";
    out << " ]";
}

extern "C" {

hipError_t rocsparse_hip_free(void* mem, const char* tag)
{
    const auto mode = memstat::s_enabled ? (memstat::s_force_managed ? memstat_mode::managed
                                                                     : memstat_mode::device)
                                         : memstat_mode::device;

    return rocsparse_free(mem, mode, tag);
}

hipError_t rocsparse_hip_malloc(void** mem, size_t nbytes, const char* tag)
{
    const auto mode = memstat::s_enabled ? (memstat::s_force_managed ? memstat_mode::managed
                                                                     : memstat_mode::device)
                                         : memstat_mode::device;

    return rocsparse_malloc(mem, nbytes, mode, tag);
}

hipError_t rocsparse_hip_free_async(void* mem, hipStream_t stream, const char* tag)
{
    return rocsparse_free_async(mem, stream, memstat_mode::device, tag);
}

hipError_t
    rocsparse_hip_malloc_async(void** mem, size_t nbytes, hipStream_t stream, const char* tag)
{
    return rocsparse_malloc_async(mem, nbytes, stream, memstat_mode::device, tag);
}

hipError_t rocsparse_hip_host_free(void* mem, const char* tag)
{
    return rocsparse_free(mem, memstat_mode::host, tag);
}

hipError_t rocsparse_hip_host_malloc(void** mem, size_t nbytes, const char* tag)
{
    return rocsparse_malloc(mem, nbytes, memstat_mode::host, tag);
}

hipError_t rocsparse_hip_free_managed(void* mem, const char* tag)
{
    return rocsparse_free(mem, memstat_mode::managed, tag);
}

hipError_t rocsparse_hip_malloc_managed(void** mem, size_t nbytes, const char* tag)
{
    return rocsparse_malloc(mem, nbytes, memstat_mode::managed, tag);
}

rocsparse_status rocsparse_memstat_report(const char* filename)
{
    if(memstat::s_enabled)
    {
        if(filename == nullptr)
        {
            return rocsparse_status_invalid_pointer;
        }
        memstat::instance().set_filename(filename);
    }
    return rocsparse_status_success;
}
}

#endif
