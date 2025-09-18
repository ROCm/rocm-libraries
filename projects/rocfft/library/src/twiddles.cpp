/******************************************************************************
* Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "twiddles.h"
#include "../../shared/arithmetic.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_hip.h"
#include "function_pool.h"
#include "rtc_cache.h"
#include "rtc_kernel.h"
#include "rtc_twiddle_kernel.h"
#include <cassert>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

// this vector stores streams for each device id.  index in the
// vector is device id.  note that this vector needs to be protected
// against concurrent access, but twiddles are always accessed
// through the Repo which guarantees exclusive access.
static std::vector<hipStream_wrapper_t> twiddle_streams;

void twiddle_streams_cleanup()
{
    twiddle_streams.clear();
}

// Twiddle factors table
template <typename T>
class TwiddleTable
{
protected:
    size_t N;
    size_t half_N;
    size_t length_limit; // limit number of generated table elements

    // Attach half-N table for potential fused even-length
    // real2complex post-processing or complex2real pre-processing.
    // Pre/post processing requires a table that's a quarter of the
    // real length, but N here is our complex length.  So half-N is
    // what we need.
    bool attach_halfN;

    const rocfft_precision precision;
    hipDeviceProp_t        deviceProp;

    void GetKernelParams(const std::vector<size_t>& radices,
                         std::vector<size_t>&       radices_prod,
                         std::vector<size_t>&       radices_sum_prod,
                         size_t&                    max_radix_prod,
                         size_t&                    min_radix,
                         size_t&                    table_sz)
    {
        radices_sum_prod = {0};
        radices_prod     = {};

        size_t prod      = 1;
        size_t prod_next = radices.at(0);
        size_t sum       = 0;

        for(size_t i = 0; i < radices.size() - 1; ++i)
        {
            auto radix      = radices.at(i);
            auto radix_next = radices.at(i + 1);

            prod *= radix;
            prod_next *= radix_next;
            sum += prod * (radix_next - 1);
            radices_sum_prod.push_back(sum);
            radices_prod.push_back(prod_next);
        }

        if(radices_prod.empty())
            radices_prod.push_back(radices.at(0));

        max_radix_prod = *(std::max_element(std::begin(radices_prod), std::end(radices_prod)));
        min_radix      = *(std::min_element(std::begin(radices), std::end(radices)));

        auto M          = radices.size() - 1;
        auto last_radix = radices.at(M);
        table_sz        = M ? radices_sum_prod.at(M - 1)
                           + ((radices_prod.at(M - 1) / last_radix) - 1) * (last_radix - 1)
                           + (last_radix - 1)
                            : radices_sum_prod.at(0);
    }

    void GenerateTable(const std::vector<size_t>& radices, hipStream_t& stream, gpubuf& output)
    {
        if(radices.size() > TWIDDLES_MAX_RADICES)
            throw std::runtime_error("maximum twiddle radices exceeded");

        size_t              table_sz, maxElem, minElem;
        std::vector<size_t> radices_prod, radices_sum_prod;

        GetKernelParams(radices, radices_prod, radices_sum_prod, maxElem, minElem, table_sz);

        table_sz          = std::min(table_sz, length_limit);
        auto total_length = attach_halfN ? table_sz + half_N : table_sz;

        auto table_bytes = total_length * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(total_length));

        auto device_data_ptr = static_cast<T*>(output.data());

        launch_radices_kernel(
            radices, radices_prod, radices_sum_prod, maxElem, minElem, stream, device_data_ptr);

        if(attach_halfN)
        {
            launch_half_N_kernel(stream, device_data_ptr + table_sz, half_N, N);
        }
    }

    void GenerateTable(hipStream_t& stream, gpubuf& output)
    {
        auto length       = std::min(N, length_limit);
        auto total_length = attach_halfN ? length + half_N : length;
        auto table_bytes  = total_length * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(total_length));

        auto blockSize = TWIDDLES_THREADS;
        auto numBlocks = DivRoundingUp<size_t>(length, blockSize);

        auto device_data_ptr = static_cast<T*>(output.data());

        auto kernel = RTCKernelTwiddle::generate(
            deviceProp.gcnArchName, TwiddleTableType::LENGTH_N, precision);
        RTCKernelArgs kargs;
        kargs.append_size_t(length_limit);
        kargs.append_size_t(N);
        kargs.append_ptr(device_data_ptr);

        kernel.launch(kargs, dim3(numBlocks), dim3(blockSize), 0, deviceProp, stream);

        if(attach_halfN)
        {
            launch_half_N_kernel(stream, device_data_ptr + length, half_N, N);
        }
    }

    void launch_radices_kernel(const std::vector<size_t>& radices,
                               std::vector<size_t>&       radices_prod,
                               std::vector<size_t>&       radices_sum_prod,
                               size_t                     maxElem,
                               size_t                     minElem,
                               hipStream_t&               stream,
                               T*                         output)
    {
        auto num_radices = radices.size();

        auto blockSize  = TWIDDLES_THREADS;
        auto numBlocksX = DivRoundingUp<size_t>(num_radices, blockSize);
        auto numBlocksY = DivRoundingUp<size_t>(maxElem / minElem, blockSize);

        radices_t radices_device;
        radices_t radices_prod_device;
        radices_t radices_sum_prod_device;
        std::copy(radices.begin(), radices.end(), radices_device.data);
        std::copy(radices_prod.begin(), radices_prod.end(), radices_prod_device.data);
        std::copy(radices_sum_prod.begin(), radices_sum_prod.end(), radices_sum_prod_device.data);

        auto kernel = RTCKernelTwiddle::generate(
            deviceProp.gcnArchName, TwiddleTableType::RADICES, precision);
        RTCKernelArgs kargs;
        kargs.append_size_t(length_limit);
        kargs.append_size_t(num_radices);
        kargs.append_struct(radices_device);
        kargs.append_struct(radices_prod_device);
        kargs.append_struct(radices_sum_prod_device);
        kargs.append_ptr(output);
        kernel.launch(
            kargs, dim3(numBlocksX, numBlocksY), dim3(blockSize, blockSize), 0, deviceProp, stream);
    }

    void launch_pp_twiddle_kernel(hipStream_t& stream, T* output, size_t N)
    {
        auto blockSize = TWIDDLES_THREADS;

        auto kernel = RTCKernelTwiddle::generate(
            deviceProp.gcnArchName, TwiddleTableType::PARTIAL_PASS_N, precision);
        RTCKernelArgs kargs;
        kargs.append_size_t(N);
        kargs.append_ptr(output);

        auto numBlocks_N = DivRoundingUp<size_t>(N, blockSize);

        kernel.launch(kargs,
                      dim3(numBlocks_N, numBlocks_N),
                      dim3(blockSize, blockSize, 1),
                      0,
                      deviceProp,
                      stream);
    }

    void launch_half_N_kernel(hipStream_t& stream, T* output, size_t half_N, size_t N)
    {
        auto blockSize = TWIDDLES_THREADS;

        auto kernel = RTCKernelTwiddle::generate(
            deviceProp.gcnArchName, TwiddleTableType::HALF_N, precision);
        RTCKernelArgs kargs;
        kargs.append_size_t(half_N);
        kargs.append_size_t(N);
        kargs.append_ptr(output);

        auto numBlocks_halfN = DivRoundingUp<size_t>(half_N, blockSize);

        kernel.launch(kargs, dim3(numBlocks_halfN), dim3(blockSize), 0, deviceProp, stream);
    }

public:
    TwiddleTable(rocfft_precision       precision,
                 const hipDeviceProp_t& deviceProp,
                 size_t                 _N,
                 size_t                 _length_limit,
                 bool                   _attach_halfN)
        : N(_N)
        , length_limit(_length_limit ? _length_limit : _N)
        , attach_halfN(_attach_halfN)
        , precision(precision)
        , deviceProp(deviceProp)
    {
        half_N = attach_halfN ? (N + 1) / 2 : 0;
    }

    void GenerateTwiddleTable(const std::vector<size_t>& radices, hipStream_t& stream, gpubuf& twts)
    {
        auto use_radices = !radices.empty();
        use_radices ? GenerateTable(radices, stream, twts) : GenerateTable(stream, twts);
    }
};

template <typename T>
class TwiddleTable2D : public TwiddleTable<T>
{
private:
    size_t N1;
    size_t N2;
    bool   attach_halfN1;
    bool   attach_halfN2;

public:
    TwiddleTable2D(rocfft_precision       precision,
                   const hipDeviceProp_t& deviceProp,
                   size_t                 _N1,
                   size_t                 _N2,
                   bool                   _attach_halfN1,
                   bool                   _attach_halfN2)
        : TwiddleTable<T>(precision, deviceProp, 0, 0, false)
        , N1(_N1)
        , N2(_N2)
        , attach_halfN1(_attach_halfN1)
        , attach_halfN2(_attach_halfN2)
    {
    }

    void GenerateTwiddleTable(const std::vector<size_t>& radices1,
                              const std::vector<size_t>& radices2,
                              hipStream_t&               stream,
                              gpubuf&                    output)
    {
        // _half_N can be either halfN1 or halfN2, we don't enable both at the same time
        size_t _half_N = (attach_halfN1) ? (N1 + 1) / 2 : ((attach_halfN2) ? (N2 + 1) / 2 : 0);
        size_t _N1orN2 = (attach_halfN1) ? N1 : ((attach_halfN2) ? N2 : 0);

        if(radices1 == radices2)
            N2 = 0;

        size_t              table_sz_1, maxElem_1, minElem_1;
        std::vector<size_t> radices_prod_1, radices_sum_prod_1;

        TwiddleTable<T>::GetKernelParams(
            radices1, radices_prod_1, radices_sum_prod_1, maxElem_1, minElem_1, table_sz_1);

        size_t              table_sz_2, maxElem_2, minElem_2;
        std::vector<size_t> radices_prod_2, radices_sum_prod_2;

        if(N2)
            TwiddleTable<T>::GetKernelParams(
                radices2, radices_prod_2, radices_sum_prod_2, maxElem_2, minElem_2, table_sz_2);
        else
            table_sz_2 = N2;

        auto table_sz       = table_sz_1 + table_sz_2; // basic twd without PRE_POST
        auto total_table_sz = table_sz + _half_N; // half_N would be zero if not PRE/POST
        auto table_bytes    = total_table_sz * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(total_table_sz));

        auto device_data_ptr          = static_cast<T*>(output.data());
        TwiddleTable<T>::length_limit = N1;
        TwiddleTable<T>::launch_radices_kernel(radices1,
                                               radices_prod_1,
                                               radices_sum_prod_1,
                                               maxElem_1,
                                               minElem_1,
                                               stream,
                                               device_data_ptr);
        if(N2)
        {
            TwiddleTable<T>::length_limit = N2;
            TwiddleTable<T>::launch_radices_kernel(radices2,
                                                   radices_prod_2,
                                                   radices_sum_prod_2,
                                                   maxElem_2,
                                                   minElem_2,
                                                   stream,
                                                   device_data_ptr + table_sz_1);
        }

        if(_half_N)
        {
            TwiddleTable<T>::launch_half_N_kernel(
                stream, device_data_ptr + table_sz, _half_N, _N1orN2);
        }
    }
};

// Twiddle factors table for large N > 4096
// used in 3-step algorithm
template <typename T>
class TwiddleTableLarge
{
private:
    size_t N; // length
    size_t largeTwdBase;
    size_t X, Y;
    size_t tableSize;

    const rocfft_precision precision;
    hipDeviceProp_t        deviceProp;

public:
    TwiddleTableLarge(rocfft_precision       precision,
                      const hipDeviceProp_t& deviceProp,
                      size_t                 length,
                      size_t                 base = LTWD_BASE_DEFAULT)
        : N(length)
        , largeTwdBase(base)
        , precision(precision)
        , deviceProp(deviceProp)
    {
        X         = static_cast<size_t>(1) << largeTwdBase; // ex: 2^8 = 256
        Y         = DivRoundingUp<size_t>(CeilPo2(N), largeTwdBase);
        tableSize = X * Y;
    }

    void GenerateTwiddleTable(hipStream_t& stream, gpubuf& output)
    {
        auto table_bytes = tableSize * sizeof(T);

        if(table_bytes == 0)
            return;

        if(output.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate twiddle length "
                                     + std::to_string(tableSize));

        auto blockSize = TWIDDLES_THREADS;

        double phi = TWO_PI / double(N);

        auto numBlocksX = DivRoundingUp<size_t>(X, blockSize);
        auto numBlocksY = DivRoundingUp<size_t>(Y, blockSize);

        auto kernel = RTCKernelTwiddle::generate(
            deviceProp.gcnArchName, TwiddleTableType::LARGE, precision);
        RTCKernelArgs kargs;
        kargs.append_double(phi);
        kargs.append_size_t(largeTwdBase);
        kargs.append_size_t(X);
        kargs.append_size_t(Y);
        kargs.append_ptr(output.data());

        kernel.launch(
            kargs, dim3(numBlocksX, numBlocksY), dim3(blockSize, blockSize), 0, deviceProp, stream);
    }
};

// Twiddle table for partial pass
template <typename T>
class TwiddleTablePP
{
protected:
    size_t N;

    const rocfft_precision precision;
    hipDeviceProp_t        deviceProp;

    void launch_twiddle_kernel(hipStream_t& stream, T* output, size_t N)
    {
        auto blockSize = TWIDDLES_THREADS;

        auto kernel = RTCKernelTwiddle::generate(
            deviceProp.gcnArchName, TwiddleTableType::PARTIAL_PASS_N, precision);
        RTCKernelArgs kargs;
        kargs.append_size_t(N);
        kargs.append_ptr(output);

        auto numBlocksX = DivRoundingUp<size_t>(N, blockSize);
        auto numBlocksY = DivRoundingUp<size_t>(N, blockSize);

        kernel.launch(
            kargs, dim3(numBlocksX, numBlocksY), dim3(blockSize, blockSize), 0, deviceProp, stream);
    }

public:
    TwiddleTablePP(rocfft_precision precision, const hipDeviceProp_t& deviceProp, size_t _N)
        : N(_N)
        , precision(precision)
        , deviceProp(deviceProp)
    {
    }

    void GenerateTwiddleTable(hipStream_t& stream, gpubuf& twts)
    {
        auto table_bytes = N * N * sizeof(T);

        if(table_bytes == 0)
            return;

        if(twts.alloc(table_bytes) != hipSuccess)
            throw std::runtime_error("unable to allocate partial-pass twiddle table of length "
                                     + std::to_string(N * N));

        auto device_data_ptr = static_cast<T*>(twts.data());

        launch_twiddle_kernel(stream, device_data_ptr, N);
    }
};

template <typename T>
gpubuf twiddles_create_pr(size_t                     N,
                          size_t                     length_limit,
                          rocfft_precision           precision,
                          const hipDeviceProp_t&     deviceProp,
                          size_t                     largeTwdBase,
                          bool                       attach_halfN,
                          const std::vector<size_t>& radices,
                          unsigned int               deviceId)
{
    if(largeTwdBase && length_limit)
        throw std::runtime_error("length-limited large twiddles are not supported");

    gpubuf twts;
    if(deviceId >= twiddle_streams.size())
        twiddle_streams.resize(deviceId + 1);
    if(twiddle_streams[deviceId] == nullptr)
        twiddle_streams[deviceId].alloc();
    hipStream_wrapper_t& stream = twiddle_streams[deviceId];

    if(!stream)
        stream.alloc();

    if((N <= LARGE_TWIDDLE_THRESHOLD) && largeTwdBase == 0)
    {
        TwiddleTable<T> twTable(precision, deviceProp, N, length_limit, attach_halfN);
        twTable.GenerateTwiddleTable(radices, stream, twts);
    }
    else
    {
        assert(!attach_halfN);

        if(largeTwdBase == 0)
        {
            TwiddleTable<T> twTable(precision, deviceProp, N, length_limit, attach_halfN);
            twTable.GenerateTwiddleTable(radices, stream, twts);
        }
        else
        {
            TwiddleTableLarge<T> twTable(
                precision, deviceProp, N, largeTwdBase); // does not generate radices
            twTable.GenerateTwiddleTable(stream, twts);
        }
    }

    if(hipStreamSynchronize(stream) != hipSuccess)
        throw std::runtime_error("hipStream failure");

    return twts;
}

gpubuf twiddles_create(size_t                     N,
                       size_t                     length_limit,
                       rocfft_precision           precision,
                       const hipDeviceProp_t&     deviceProp,
                       size_t                     largeTwdBase,
                       bool                       attach_halfN,
                       const std::vector<size_t>& radices,
                       unsigned int               deviceId)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return twiddles_create_pr<rocfft_complex<float>>(
            N, length_limit, precision, deviceProp, largeTwdBase, attach_halfN, radices, deviceId);
    case rocfft_precision_double:
        return twiddles_create_pr<rocfft_complex<double>>(
            N, length_limit, precision, deviceProp, largeTwdBase, attach_halfN, radices, deviceId);
    case rocfft_precision_half:
        return twiddles_create_pr<rocfft_complex<rocfft_fp16>>(
            N, length_limit, precision, deviceProp, largeTwdBase, attach_halfN, radices, deviceId);
    }
}

template <typename T>
gpubuf twiddles_create_2D_pr(size_t                     N1,
                             size_t                     N2,
                             rocfft_precision           precision,
                             const hipDeviceProp_t&     deviceProp,
                             bool                       attach_halfN,
                             bool                       attach_halfN2,
                             const std::vector<size_t>& radices1,
                             const std::vector<size_t>& radices2,
                             unsigned int               deviceId)
{
    gpubuf twts;
    if(deviceId >= twiddle_streams.size())
        twiddle_streams.resize(deviceId + 1);
    hipStream_wrapper_t& stream = twiddle_streams[deviceId];
    if(!stream)
        stream.alloc();

    TwiddleTable2D<T> twTable(precision, deviceProp, N1, N2, attach_halfN, attach_halfN2);
    twTable.GenerateTwiddleTable(radices1, radices2, stream, twts);

    if(hipStreamSynchronize(stream) != hipSuccess)
        throw std::runtime_error("hipStream failure");

    return twts;
}

gpubuf twiddles_create_2D(size_t                     N1,
                          size_t                     N2,
                          rocfft_precision           precision,
                          const hipDeviceProp_t&     deviceProp,
                          bool                       attach_halfN,
                          bool                       attach_halfN2,
                          const std::vector<size_t>& radices1,
                          const std::vector<size_t>& radices2,
                          unsigned int               deviceId)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return twiddles_create_2D_pr<rocfft_complex<float>>(N1,
                                                            N2,
                                                            precision,
                                                            deviceProp,
                                                            attach_halfN,
                                                            attach_halfN2,
                                                            radices1,
                                                            radices2,
                                                            deviceId);
    case rocfft_precision_double:
        return twiddles_create_2D_pr<rocfft_complex<double>>(N1,
                                                             N2,
                                                             precision,
                                                             deviceProp,
                                                             attach_halfN,
                                                             attach_halfN2,
                                                             radices1,
                                                             radices2,
                                                             deviceId);
    case rocfft_precision_half:
        return twiddles_create_2D_pr<rocfft_complex<rocfft_fp16>>(N1,
                                                                  N2,
                                                                  precision,
                                                                  deviceProp,
                                                                  attach_halfN,
                                                                  attach_halfN2,
                                                                  radices1,
                                                                  radices2,
                                                                  deviceId);
    }
}

template <typename T>
gpubuf twiddles_pp_create_pr(size_t                 N,
                             rocfft_precision       precision,
                             const hipDeviceProp_t& deviceProp,
                             unsigned int           deviceId)
{
    gpubuf twts;
    if(deviceId >= twiddle_streams.size())
        twiddle_streams.resize(deviceId + 1);
    if(twiddle_streams[deviceId] == nullptr)
        twiddle_streams[deviceId].alloc();
    hipStream_wrapper_t& stream = twiddle_streams[deviceId];

    if(!stream)
        stream.alloc();

    TwiddleTablePP<T> twTable(precision, deviceProp, N);
    twTable.GenerateTwiddleTable(stream, twts);

    if(hipStreamSynchronize(stream) != hipSuccess)
        throw std::runtime_error("hipStream failure");

    return twts;
}

gpubuf twiddles_pp_create(size_t                 N,
                          rocfft_precision       precision,
                          const hipDeviceProp_t& deviceProp,
                          unsigned int           deviceId)
{
    switch(precision)
    {
    case rocfft_precision_single:
        return twiddles_pp_create_pr<rocfft_complex<float>>(N, precision, deviceProp, deviceId);
    case rocfft_precision_double:
        return twiddles_pp_create_pr<rocfft_complex<double>>(N, precision, deviceProp, deviceId);
    case rocfft_precision_half:
        return twiddles_pp_create_pr<rocfft_complex<rocfft_fp16>>(
            N, precision, deviceProp, deviceId);
    }
}
