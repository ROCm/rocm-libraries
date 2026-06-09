/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef GUARD_RPP_ALLOCATOR_HPP
#define GUARD_RPP_ALLOCATOR_HPP

#include <cassert>

#include "rpp.h"
#include "common.hpp"
#include "errors.hpp"
#include "manage_ptr.hpp"

#ifdef GPU_SUPPORT
namespace rpp {

struct AllocatorDeleter
{
    rppDeallocatorFunction deallocator;
    void* context;

    template <class T>
    void operator()(T* x) const
    {
        assert(deallocator != nullptr);
        if(x != nullptr)
        {
            deallocator(context, x);
        }
    }
};
struct Allocator
{
    rppAllocatorFunction allocator;
    rppDeallocatorFunction deallocator;
    void* context;

    using ManageDataPtr =
        std::unique_ptr<typename std::remove_pointer<Data_t>::type, AllocatorDeleter>;

    ManageDataPtr operator()(std::size_t n) const
    {
        assert(allocator != nullptr);
        assert(deallocator != nullptr);
        auto result = allocator(context, n);
        if(result == nullptr && n != 0)
        {
            RPP_THROW("Custom allocator failed to allocate memory for buffer size " +
                         std::to_string(n) + ": ");
        }
        return ManageDataPtr{DataCast(result), AllocatorDeleter{deallocator, context}};
    }
};

} // namespace rpp

#endif // GPU_SUPPORT
#endif // GUARD_RPP_ALLOCATOR_HPP
