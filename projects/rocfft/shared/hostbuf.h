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

#ifndef ROCFFT_HOSTBUF_H
#define ROCFFT_HOSTBUF_H

#include "arithmetic.h"
#include "sys_mem.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

#ifndef WIN32
#include <stdlib.h>
#include <sys/mman.h>
#endif

struct HOSTBUF_MEM_USAGE
{
    const std::string msg;
};

// Simple RAII class for host buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class hostbuf_t
{
public:
    hostbuf_t() {}
    // buffers are movable but not copyable
    hostbuf_t(hostbuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(owned, other.owned);
        std::swap(bsize, other.bsize);
        std::swap(bsize_track, other.bsize_track);
        std::swap(is_pinned_memory, other.is_pinned_memory);
    }
    hostbuf_t& operator=(hostbuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(owned, other.owned);
        std::swap(bsize, other.bsize);
        std::swap(bsize_track, other.bsize_track);
        std::swap(is_pinned_memory, other.is_pinned_memory);
        return *this;
    }
    hostbuf_t(const hostbuf_t&) = delete;
    hostbuf_t& operator=(const hostbuf_t&) = delete;

    static hostbuf_t make_nonowned(T* p, size_t size_bytes = 0)
    {
        hostbuf_t ret;
        ret.owned            = false;
        ret.buf              = p;
        ret.bsize            = size_bytes;
        ret.bsize_track      = 0;
        ret.is_pinned_memory = false; // irrelevant anyways if not owned
        return ret;
    }

    ~hostbuf_t()
    {
        free();
    }

    void alloc(size_t size, bool make_it_pinned = false)
    {
        free();

        bsize = size;

        auto usable_mem = host_memory::singleton().get_usable_bytes();
        if(total_used_mem + size > usable_mem)
        {
            std::stringstream msg;
            msg << "Host memory usage limit exceed (used mem: "
                << bytes_to_GiB(total_used_mem + size)
                << "GiB, free mem: " << bytes_to_GiB(usable_mem) << " GiB)";
            throw HOSTBUF_MEM_USAGE{msg.str()};
        }

        if(make_it_pinned)
        {
            if(hipHostMalloc(&buf, size) != hipSuccess)
            {
                buf   = nullptr;
                bsize = 0;
            }
        }
        else
        {
            // we're aligning to multiples of 64 bytes, so round the
            // allocation size up to the nearest 64 to keep ASAN happy
            if(size % 64)
            {
                size += 64 - size % 64;
            }

            // FFTW requires aligned allocations to use faster SIMD instructions.
            // If enabling hugepages, align to 2 MiB. Otherwise, aligning to
            // 64 bytes is enough for AVX instructions up to AVX512.
#ifdef WIN32
            buf = _aligned_malloc(size, 64);
#else
            // On Linux, ask for hugepages to reduce TLB pressure and
            // improve performance.  Allocations need to be aligned to
            // the hugepage size, and rounded up to the next whole
            // hugepage.
            static const size_t TWO_MiB = 2 * 1024 * 1024;
            if(size >= TWO_MiB)
            {
                size_t rounded_size = DivRoundingUp(size, TWO_MiB) * TWO_MiB;
                buf                 = aligned_alloc(TWO_MiB, rounded_size);
                madvise(buf, rounded_size, MADV_HUGEPAGE);
            }
            else
                buf = aligned_alloc(64, size);
#endif
        }
        if(!buf)
            throw std::bad_alloc();

        is_pinned_memory = make_it_pinned;
        bsize_track      = size;
        total_used_mem += bsize_track;
    }

    size_t size() const
    {
        return bsize;
    }

    bool is_pinned() const
    {
        return is_pinned_memory;
    }

    void free()
    {
        if(buf != nullptr)
        {
            if(owned)
            {
                total_used_mem -= bsize_track;
                if(is_pinned_memory)
                {
                    (void)hipHostFree(buf);
                }
                else
                {
#ifdef WIN32
                    _aligned_free(buf);
#else
                    std::free(buf);
#endif
                }
            }
            buf   = nullptr;
            bsize = bsize_track = 0;
        }
        owned = true;
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // return a pointer to the allocated memory, offset by the
    // specified number of bytes
    T* data_offset(size_t offset_bytes = 0) const
    {
        void* ptr = static_cast<char*>(buf) + offset_bytes;
        return static_cast<T*>(ptr);
    }

    // Copy method
    hostbuf_t copy(bool make_copy_pinned = false) const
    {
        hostbuf_t copy;
        copy.alloc(bsize, make_copy_pinned);
        memcpy(copy.buf, buf, bsize);
        return copy;
    }

    // shrink the buffer to fit the new size
    void shrink(size_t new_size)
    {
        if(new_size > bsize)
            throw std::runtime_error("can't shrink hostbuf to larger size");
        // just pretend the buffer is now that size
        bsize = new_size;
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
    // The host buffer
    void* buf = nullptr;
    // whether this object owns the 'buf' pointer (and hence needs to
    // free it)
    bool   owned            = true;
    bool   is_pinned_memory = false;
    size_t bsize            = 0;

    // Buffer size for tracking total memory usage.
    // When buffer is shrunk in place, bsize_track is not changed.
    size_t bsize_track = 0;

    // Keeps track of total used memory for all hostbufs
    inline static std::atomic<size_t> total_used_mem = 0;
};

// default hostbuf that gives out void* pointers
typedef hostbuf_t<> hostbuf;
#endif
