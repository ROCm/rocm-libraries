/******************************************************************************
* Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include "kargs.h"
#include "../../shared/rocfft_hip.h"
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>

// malloc device buffer; copy host buffer to device buffer
bool KernelArgsBuffer::create(const std::vector<size_t>& length,
                              const std::vector<size_t>& inStride,
                              const std::vector<size_t>& outStride,
                              size_t                     iDist,
                              size_t                     oDist,
                              KIntType                   itype)
{
    assert(length.size() == inStride.size());
    assert(length.size() == outStride.size());

    // the dist goes one past the last stride, so the arrays need to
    // hold dim + 1 values
    if(length.size() >= KERN_ARGS_ARRAY_WIDTH)
        throw std::runtime_error("too many dimensions for kernel argument buffer");

    this->itype = itype;

    const size_t total_bytes = lengths_bytes() + 2 * strides_bytes();

    if(buf.alloc(total_bytes) != hipSuccess)
        return false;

    std::vector<char> host(total_bytes, 0);

    auto store = [&](char* array, size_t i, size_t value, const char* what) {
        if(itype == KIntType::U32)
        {
            if(value > std::numeric_limits<unsigned int>::max())
                throw std::runtime_error(std::string(what) + " overflows 32-bit kernel kint_type");
            reinterpret_cast<unsigned int*>(array)[i] = static_cast<unsigned int>(value);
        }
        else
            reinterpret_cast<unsigned long long*>(array)[i] = value;
    };

    for(size_t i = 0; i < length.size(); ++i)
        store(host.data(), i, length[i], "length");

    // NB: iDist is right after the last inStride[dim-1], i.e. inStride[dim] = batch-in-stride
    //     oDist is right after the last outStride[dim-1], i.e. outStride[dim] = batch-out-stride
    auto pack_strides = [&](size_t array_idx, const std::vector<size_t>& stride, size_t dist) {
        char* array = host.data() + lengths_bytes() + array_idx * strides_bytes();

        for(size_t i = 0; i < stride.size(); ++i)
            store(array, i, stride[i], "stride");
        store(array, stride.size(), dist, "dist");
    };

    pack_strides(0, inStride, iDist);
    pack_strides(1, outStride, oDist);

    if(hipMemcpy(buf.data(), host.data(), total_bytes, hipMemcpyHostToDevice) != hipSuccess)
        buf.free();
    return buf != nullptr;
}
