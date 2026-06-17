// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_BENCH_H
#define ROCFFT_BENCH_H

#include "../../shared/fft_params.h"
#include "../../shared/rocfft_hip.h"

#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>
#include <vector>

class rocfft_hip_runtime_error : public std::runtime_error
{
public:
    rocfft_hip_runtime_error(const std::string& msg = "")
        : runtime_error(msg)
    {
    }
};

// This is used to either wrap a HIP function call, or to explicitly check a variable
// for an error condition.  If an error occurs, we throw.
// Note: std::runtime_error does not take unicode strings as input, so only strings
// supported
inline void
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw rocfft_hip_runtime_error(errorm);
    }
}

class rocfft_runtime_error : public std::runtime_error
{
public:
    rocfft_runtime_error(const std::string& msg = "")
        : runtime_error(msg)
    {
    }
};

inline void lib_V_Throw(rocfft_status      res,
                        const std::string& msg,
                        size_t             lineno,
                        const std::string& fileName)
{
    if(res != rocfft_status_success)
    {
        std::stringstream tmp;
        tmp << "LIB_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw rocfft_runtime_error(errorm);
    }
}

#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status, _message) lib_V_Throw(_status, _message, __LINE__, __FILE__)

// return input bricks for params, or one big brick covering the
// input field if no bricks are specified
template <typename Tparams>
std::vector<fft_params::fft_brick> get_input_bricks(const Tparams& params)
{
    std::vector<fft_params::fft_brick> bricks;
    if(!params.ifields.empty())
        bricks = params.ifields[0].bricks;
    else
    {
        auto len = params.ilength();

        // just make one big brick covering the whole input field
        bricks.resize(1);
        bricks.front().lower.resize(len.size() + 1);
        bricks.front().upper.resize(len.size() + 1);
        bricks.front().stride.resize(len.size() + 1);

        bricks.front().upper.front() = params.nbatch;
        std::copy(len.begin(), len.end(), bricks.front().upper.begin() + 1);

        bricks.front().stride.front() = params.idist;
        std::copy(params.istride.begin(), params.istride.end(), bricks.front().stride.begin() + 1);
    }
    return bricks;
}

// return output bricks for params, or one big brick covering the
// output field if no bricks are specified
template <typename Tparams>
std::vector<fft_params::fft_brick> get_output_bricks(const Tparams& params)
{
    std::vector<fft_params::fft_brick> bricks;
    if(!params.ofields.empty())
        bricks = params.ofields[0].bricks;
    else
    {
        auto len = params.olength();

        // just make one big brick covering the whole output field
        bricks.resize(1);
        bricks.front().lower.resize(len.size() + 1);
        bricks.front().upper.resize(len.size() + 1);
        bricks.front().stride.resize(len.size() + 1);

        bricks.front().upper.front() = params.nbatch;
        std::copy(len.begin(), len.end(), bricks.front().upper.begin() + 1);

        bricks.front().stride.front() = params.odist;
        std::copy(params.ostride.begin(), params.ostride.end(), bricks.front().stride.begin() + 1);
    }
    return bricks;
}

// Allocate input/output buffers for a bench run.
template <typename Tparams>
void alloc_bench_bricks(const Tparams&                            params,
                        const std::vector<fft_params::fft_brick>& ibricks,
                        const std::vector<fft_params::fft_brick>& obricks,
                        std::vector<gpubuf>&                      ibuffers,
                        std::vector<gpubuf>&                      obuffer_data,
                        std::vector<gpubuf>*&                     obuffers,
                        std::vector<hostbuf>&                     host_buffers,
                        bool                                      is_host_gen)
{
    ibuffers.clear();
    obuffer_data.clear();
    if(is_host_gen)
        host_buffers.clear();
    for(size_t b_idx = 0; b_idx < std::max(ibricks.size(), obricks.size()); b_idx++)
    {
        const auto* ibrick = b_idx < ibricks.size() ? &ibricks[b_idx] : nullptr;
        const auto* obrick = b_idx < obricks.size() ? &obricks[b_idx] : nullptr;
        size_t      isize  = ibrick ? compute_ptrdiff(ibrick->length(), ibrick->stride) : 0;
        size_t      osize  = obrick ? compute_ptrdiff(obrick->length(), obrick->stride) : 0;
        if(params.placement == fft_placement_inplace && is_real(params.transform_type))
        {
            // we may need slightly more than compute_ptrdiff's value if real innermost length
            // is not divided (typically two more real elements or one more complex element)
            const auto real_io
                = is_fwd(params.transform_type) ? fft_io::fft_io_in : fft_io::fft_io_out;
            const auto* real_brick = real_io == fft_io::fft_io_in ? ibrick : obrick;
            auto&       real_size  = real_io == fft_io::fft_io_in ? isize : osize;
            if(real_brick && real_brick->lower.back() == 0
               && real_brick->upper.back() == params.length.back())
            {
                const auto brick_length = real_brick->length();
                for(size_t dim = 0; dim < brick_length.size(); dim++)
                {
                    real_size = std::max(real_size, brick_length[dim] * real_brick->stride[dim]);
                }
            }
        }
        isize *= var_size<size_t>(params.precision, params.itype);
        osize *= var_size<size_t>(params.precision, params.otype);

        for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
        {
            const auto* iobrick = io == fft_io::fft_io_in ? ibrick : obrick;
            if(!iobrick)
                continue;
            auto& iobuffers = params.placement == fft_placement_inplace || io == fft_io::fft_io_in
                                  ? ibuffers
                                  : obuffer_data;
            const auto iobuffer_size = params.placement == fft_placement_inplace
                                           ? std::max(isize, osize)
                                           : (io == fft_io::fft_io_in ? isize : osize);

            rocfft_scoped_device dev(iobrick->device);
            const bool io_is_planar = io == fft_io::fft_io_in ? array_type_is_planar(params.itype)
                                                              : array_type_is_planar(params.otype);
            for(auto tmp = 0; tmp < (io_is_planar ? 2 : 1); tmp++)
            {
                iobuffers.emplace_back();
                if(iobuffers.back().alloc(iobuffer_size) != hipSuccess)
                    throw std::runtime_error("hipMalloc failed");
                if(is_host_gen)
                {
                    host_buffers.emplace_back();
                    host_buffers.back().alloc(iobuffer_size);
                }
            }
            if(params.placement == fft_placement_inplace)
            {
                // no need to redo the above for the output if
                // this point was reached for the input
                break;
            }
        }
    }
    obuffers = params.placement == fft_placement_inplace ? &ibuffers : &obuffer_data;
}

void copy_host_input_to_dev(std::vector<hostbuf>& host_buffers, std::vector<gpubuf>& buffers)
{
    for(size_t i = 0; i < buffers.size(); ++i)
    {
        if(hipMemcpy(buffers[i].data(),
                     host_buffers[i].data(),
                     host_buffers[i].size(),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy failure");
    }
}

template <typename Tparams>
void init_bench_input(const Tparams&                            params,
                      const std::vector<fft_params::fft_brick>& bricks,
                      std::vector<gpubuf>&                      buffers,
                      std::vector<hostbuf>&                     host_buffers,
                      bool                                      is_host_gen)
{
    auto elem_size = var_size<size_t>(params.precision, params.itype);
    if(is_host_gen)
    {
        std::vector<void*> ptrs;
        ptrs.reserve(host_buffers.size());
        for(auto& buf : host_buffers)
            ptrs.push_back(buf.data());

        init_local_input<Tparams, hostbuf>(0, params, bricks, elem_size, ptrs);
        copy_host_input_to_dev(host_buffers, buffers);
    }
    else
    {
        std::vector<void*> ptrs;
        ptrs.reserve(buffers.size());
        for(auto& buf : buffers)
            ptrs.push_back(buf.data());

        init_local_input<Tparams, gpubuf>(0, params, bricks, elem_size, ptrs);
    }
}

template <typename Tparams>
void print_device_buffer(const Tparams& params, std::vector<gpubuf>& buffer, bool input)
{
    // copy data back to host
    std::vector<hostbuf> print_buffer;
    for(auto& buf : buffer)
    {
        print_buffer.emplace_back();
        print_buffer.back().alloc(buf.size());
        if(hipMemcpy(print_buffer.back().data(), buf.data(), buf.size(), hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("hipMemcpy failed");
    }
    if(input)
        params.print_ibuffer(print_buffer);
    else
        params.print_obuffer(print_buffer);
}

#endif // ROCFFT_BENCH_H
