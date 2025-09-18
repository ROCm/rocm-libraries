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
    auto alloc_buffers = [&params, &host_buffers](const std::vector<fft_params::fft_brick>& bricks,
                                                  fft_array_type                            type,
                                                  std::vector<gpubuf>&                      output,
                                                  bool is_host_gen) {
        auto       elem_size = var_size<size_t>(params.precision, type);
        const bool is_planar
            = type == fft_array_type_complex_planar || type == fft_array_type_hermitian_planar;
        // alloc 2x buffers, each half size for planar
        if(is_planar)
            elem_size /= 2;

        for(const auto& b : bricks)
        {
            rocfft_scoped_device dev(b.device);

            size_t brick_size_bytes = compute_ptrdiff(b.length(), b.stride, 0, 0) * elem_size;
            output.emplace_back();
            if(output.back().alloc(brick_size_bytes) != hipSuccess)
                throw std::runtime_error("hipMalloc failed");
            if(is_planar)
            {
                output.emplace_back();
                if(output.back().alloc(brick_size_bytes) != hipSuccess)
                    throw std::runtime_error("hipMalloc failed");
            }
            if(is_host_gen)
            {
                host_buffers.emplace_back();
                host_buffers.back().alloc(brick_size_bytes);
                if(is_planar)
                {
                    host_buffers.emplace_back();
                    host_buffers.back().alloc(brick_size_bytes);
                }
            }
        }
    };

    // If brick shape differs, inplace is only allowed for single
    // bricks.  e.g. in-place real-complex
    if(params.placement == fft_placement_inplace)
    {
        if(ibricks.size() != 1 && obricks.size() != 1 && ibricks != obricks)
            throw std::runtime_error(
                "in-place transform to different brick shapes only allowed for single bricks");

        // allocate the larger of the two bricks
        auto isize_bytes = compute_ptrdiff(ibricks.front().length(), ibricks.front().stride, 0, 0)
                           * var_size<size_t>(params.precision, params.itype);
        auto osize_bytes = compute_ptrdiff(obricks.front().length(), obricks.front().stride, 0, 0)
                           * var_size<size_t>(params.precision, params.otype);

        alloc_buffers(isize_bytes > osize_bytes ? ibricks : obricks,
                      isize_bytes > osize_bytes ? params.itype : params.otype,
                      ibuffers,
                      is_host_gen);
        obuffers = &ibuffers;
    }
    else
    {
        alloc_buffers(ibricks, params.itype, ibuffers, is_host_gen);
        alloc_buffers(obricks, params.otype, obuffer_data, false);
        obuffers = &obuffer_data;
    }
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
#ifdef USE_HIPRAND
        std::vector<void*> ptrs;
        ptrs.reserve(buffers.size());
        for(auto& buf : buffers)
            ptrs.push_back(buf.data());

        init_local_input<Tparams, gpubuf>(0, params, bricks, elem_size, ptrs);
#endif
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
