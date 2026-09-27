// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_LOCATION_H
#define ROCFFT_LOCATION_H

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

// Identifier for a location that a buffer lives on, or that a kernel
// will execute on.  this specifies a multi-process rank as well as a
// device ID.
struct rocfft_location_t
{
    rocfft_location_t() = default;
    rocfft_location_t(int _comm_rank, int _device)
        : comm_rank(_comm_rank)
        , device(_device)
    {
    }

    // return a location for the current device on comm rank 0
    static rocfft_location_t rank0_current_device()
    {
        rocfft_location_t id;
        if(hipGetDevice(&id.device) != hipSuccess)
            throw std::runtime_error("hipGetDevice failed");
        return id;
    }

    // allow locations to be sorted
    bool operator<(const rocfft_location_t& other) const
    {
        if(comm_rank != other.comm_rank)
            return comm_rank < other.comm_rank;
        return device < other.device;
    }

    bool operator==(const rocfft_location_t& other) const
    {
        return comm_rank == other.comm_rank && device == other.device;
    }

    std::string str() const
    {
        std::string ret = "comm rank ";
        ret += std::to_string(comm_rank);
        ret += " device ";
        ret += std::to_string(device);
        return ret;
    }

    int comm_rank = 0;
    int device    = 0;
};

#endif // ROCFFT_LOCATION_H
