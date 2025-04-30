/******************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "chirp.h"
#include "../../shared/arithmetic.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_complex.h"
#include "../../shared/rocfft_hip.h"
#include "rtc_cache.h"
#include "rtc_chirp_kernel.h"
#include "rtc_kernel.h"
#include <cassert>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

// this vector stores chirp for each device id.  index in the
// vector is device id.  note that this vector needs to be protected
// against concurrent access, but chirp are always accessed
// through the Repo which guarantees exclusive access.
static std::vector<hipStream_wrapper_t> chirp_streams;

void chirp_streams_cleanup()
{
    chirp_streams.clear();
}

template <typename Tcomplex>
void launch_chirp_kernel(const size_t           N,
                         rocfft_precision       precision,
                         const hipDeviceProp_t& deviceProp,
                         hipStream_t&           stream,
                         Tcomplex*              output)
{
    auto blockSize = CHIRP_THREADS;
    auto numBlocks = DivRoundingUp<size_t>(N, blockSize);

    auto          kernel = RTCKernelChirp::generate(deviceProp.gcnArchName, precision);
    RTCKernelArgs kargs;
    kargs.append_size_t(N);
    kargs.append_ptr(output);
    kernel.launch(kargs, dim3(numBlocks), dim3(blockSize), 0, deviceProp, stream);
}

template <typename Tcomplex>
gpubuf chirp_create_pr(size_t                 N,
                       rocfft_precision       precision,
                       unsigned int           deviceId,
                       const hipDeviceProp_t& deviceProp)
{
    gpubuf chirp;

    auto chirp_bytes = N * sizeof(Tcomplex);

    if(chirp.alloc(chirp_bytes) != hipSuccess)
        throw std::runtime_error("unable to allocate chirp length " + std::to_string(N));

    if(deviceId >= chirp_streams.size())
        chirp_streams.resize(deviceId + 1);
    hipStream_wrapper_t& stream = chirp_streams[deviceId];
    if(!stream)
        stream.alloc();

    auto device_chirp_ptr = static_cast<Tcomplex*>(chirp.data());

    launch_chirp_kernel(N, precision, deviceProp, stream, device_chirp_ptr);

    if(hipStreamSynchronize(stream) != hipSuccess)
        throw std::runtime_error("hipStream failure");

    return chirp;
}

gpubuf chirp_create(size_t                 N,
                    rocfft_precision       precision,
                    unsigned int           deviceId,
                    const hipDeviceProp_t& deviceProp)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return chirp_create_pr<rocfft_complex<float>>(N, precision, deviceId, deviceProp);
    case rocfft_precision_double:
        return chirp_create_pr<rocfft_complex<double>>(N, precision, deviceId, deviceProp);
    case rocfft_precision_half:
        return chirp_create_pr<rocfft_complex<rocfft_fp16>>(N, precision, deviceId, deviceProp);
    }
}
