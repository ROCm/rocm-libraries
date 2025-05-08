// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <numeric>

#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "kernel_launch.h"
#include "rtc_realcomplex_gen.h"
#include "rtc_realcomplex_kernel.h"
#include "tree_node.h"

RTCKernel::RTCGenerator RTCKernelRealComplex::generate_from_node(const LeafNode&    node,
                                                                 const std::string& gpu_arch,
                                                                 bool enable_callbacks)
{
    RTCGenerator generator;

    if(node.scheme != CS_KERNEL_COPY_R_TO_CMPLX && node.scheme != CS_KERNEL_COPY_CMPLX_TO_HERM
       && node.scheme != CS_KERNEL_COPY_HERM_TO_CMPLX && node.scheme != CS_KERNEL_COPY_CMPLX_TO_R)
    {
        return generator;
    }

    // input_size is the innermost dimension
    unsigned int input_size = node.length[0];
    // hermitian size is used for hermitian->complex copy
    if(node.scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
        input_size = node.outputLength[0] / 2 + 1;

    size_t elems = product(node.length.begin() + 1, node.length.end()) * input_size * node.batch;
    generator.gridDim
        = {static_cast<unsigned int>(DivRoundingUp<size_t>(elems, LAUNCH_BOUNDS_R2C_C2R_KERNEL)),
           1,
           1};
    generator.blockDim = {LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1};

    RealComplexSpecs specs{node.scheme,
                           node.length.size(),
                           node.precision,
                           node.inArrayType,
                           node.outArrayType,
                           node.GetCallbackType(enable_callbacks),
                           node.loadOps,
                           node.storeOps};

    generator.generate_name = [=]() { return realcomplex_rtc_kernel_name(specs); };

    generator.generate_src
        = [=](const std::string& kernel_name) { return realcomplex_rtc(kernel_name, specs); };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(
            new RTCKernelRealComplex(kernel_name, code, gridDim, blockDim));
    };
    return generator;
}

RTCKernelArgs RTCKernelRealComplex::get_launch_args(DeviceCallIn& data)
{
    // explode lengths/strides out to pass to the kernel
    std::array<size_t, 3> kern_lengths{1, 1, 1};
    std::array<size_t, 4> kern_stride_in{1, 1, 1, 1};
    std::array<size_t, 4> kern_stride_out{1, 1, 1, 1};
    auto                  dim = data.node->length.size();

    std::copy(data.node->length.begin(), data.node->length.end(), kern_lengths.begin());
    std::copy(data.node->inStride.begin(), data.node->inStride.end(), kern_stride_in.begin());
    kern_stride_in[dim] = data.node->iDist;
    std::copy(data.node->outStride.begin(), data.node->outStride.end(), kern_stride_out.begin());
    kern_stride_out[dim] = data.node->oDist;

    RTCKernelArgs kargs;
    if(data.node->scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
    {
        // dim_0 is the innermost dimension
        kern_lengths[0]       = data.node->outputLength[0];
        size_t hermitian_size = kern_lengths[0] / 2 + 1;
        kargs.append_unsigned_int(hermitian_size);
    }
    kargs.append_unsigned_int(kern_lengths[0]);
    kargs.append_unsigned_int(kern_lengths[1]);
    kargs.append_unsigned_int(kern_lengths[2]);
    kargs.append_unsigned_int(data.node->batch);
    kargs.append_unsigned_int(kern_stride_in[0]);
    kargs.append_unsigned_int(kern_stride_in[1]);
    kargs.append_unsigned_int(kern_stride_in[2]);
    kargs.append_unsigned_int(kern_stride_in[3]);
    kargs.append_unsigned_int(kern_stride_out[0]);
    kargs.append_unsigned_int(kern_stride_out[1]);
    kargs.append_unsigned_int(kern_stride_out[2]);
    kargs.append_unsigned_int(kern_stride_out[3]);

    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);
    kargs.append_ptr(data.bufOut[0]);
    if(array_type_is_planar(data.node->outArrayType))
        kargs.append_ptr(data.bufOut[1]);

    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);
    append_load_store_args(kargs, *data.node);

    return kargs;
}

RTCKernel::RTCGenerator RTCKernelRealComplexEven::generate_from_node(const LeafNode&    node,
                                                                     const std::string& gpu_arch,
                                                                     bool enable_callbacks)
{
    RTCGenerator generator;

    if(node.scheme != CS_KERNEL_R_TO_CMPLX && node.scheme != CS_KERNEL_CMPLX_TO_R)
    {
        return generator;
    }

    // Input_size is the innermost dimension
    size_t half_N;
    if(node.scheme == CS_KERNEL_R_TO_CMPLX)
    {
        // The upper level provides always N/2, that is regular complex fft size
        half_N = node.length[0];
    }
    else
    {
        // Length on the node is complex fft size.  Compute half_N as
        // half of the real size.
        half_N = node.length[0] - 1;
    }
    const bool Ndiv4 = half_N % 2 == 0;

    size_t elems = std::accumulate(
        node.length.begin() + 1, node.length.end(), half_N * node.batch, std::multiplies<size_t>());
    generator.gridDim
        = {static_cast<unsigned int>(DivRoundingUp<size_t>(elems, LAUNCH_BOUNDS_R2C_C2R_KERNEL)),
           1,
           1};

    generator.blockDim = {LAUNCH_BOUNDS_R2C_C2R_KERNEL, 1, 1};

    RealComplexEvenSpecs specs{{node.scheme,
                                node.length.size(),
                                node.precision,
                                node.inArrayType,
                                node.outArrayType,
                                node.GetCallbackType(enable_callbacks),
                                node.loadOps,
                                node.storeOps},
                               Ndiv4};

    generator.generate_name = [=]() { return realcomplex_even_rtc_kernel_name(specs); };

    generator.generate_src
        = [=](const std::string& kernel_name) { return realcomplex_even_rtc(kernel_name, specs); };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(
            new RTCKernelRealComplexEven(kernel_name, half_N, code, gridDim, blockDim));
    };
    return generator;
}

RTCKernelArgs RTCKernelRealComplexEven::get_launch_args(DeviceCallIn& data)
{
    RTCKernelArgs kargs;

    kargs.append_unsigned_int(half_N);
    if(data.node->length.size() > 1)
    {
        kargs.append_unsigned_int(data.node->inStride[1]);
        kargs.append_unsigned_int(data.node->outStride[1]);
    }
    unsigned int higherFFTLengths = product(data.node->length.begin() + 1, data.node->length.end());
    kargs.append_unsigned_int(higherFFTLengths);
    kargs.append_unsigned_int(data.node->batch);
    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);
    kargs.append_unsigned_int(data.node->iDist);
    kargs.append_ptr(data.bufOut[0]);
    if(array_type_is_planar(data.node->outArrayType))
        kargs.append_ptr(data.bufOut[1]);
    kargs.append_unsigned_int(data.node->oDist);
    kargs.append_ptr(data.node->twiddles);
    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);
    append_load_store_args(kargs, *data.node);

    return kargs;
}

RTCKernel::RTCGenerator RTCKernelRealComplexEvenTranspose::generate_from_node(
    const LeafNode& node, const std::string& gpu_arch, bool enable_callbacks)
{
    RTCGenerator generator;
    if(node.scheme != CS_KERNEL_R_TO_CMPLX_TRANSPOSE
       && node.scheme != CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        return generator;

    const unsigned int tileX = RealComplexEvenTransposeSpecs::TileX(node.scheme);
    const unsigned int tileY = RealComplexEvenTransposeSpecs::TileY();

    unsigned int m   = node.length[1];
    unsigned int n   = node.length[0];
    unsigned int dim = node.length.size();

    unsigned int gridX;
    unsigned int gridX_1d;
    unsigned int gridY;
    unsigned int gridZ = node.batch;

    if(node.scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE)
    {
        // In a 3-D grid, gridX handles 2 tiles at a time, gridY, the second
        // dimension - multiplied by the third dimension, and gridZ, the number of batches
        gridX    = (n - 1) / tileX / 2 + 1;
        gridX_1d = (tileX * tileY) * gridX;
        gridY = ((m - 1) / tileY + 1) * (dim > 2 ? static_cast<unsigned int>(node.length[2]) : 1);
    }
    else
    {
        // In a 3-D grid, gridX handles the first dimension - multiplied by the second dimension
        if(dim > 2)
        {
            n *= node.length[1];
            m = node.length[2];
        }
        gridX    = (n - 1) / tileX + 1;
        gridX_1d = (tileX * tileY) * (n - 1) / tileX + 1;

        // gridY dimension handles 2 tiles at a time, so allocate enough
        // blocks to go halfway across 'm'
        gridY = std::max<unsigned int>((((m - 1) / 2) + (tileY - 1)) / tileY, 1);
    }

    // check if it is possible to directly use a 3-D grid of GPUs
    bool grid3D = false;
    if(gridY < (1U << 16) && gridZ < (1U << 16))
    {
        // grid sizes are within limits to use a 3-D grid of GPUs
        generator.gridDim = {gridX, gridY, gridZ};
        grid3D            = true;
    }
    else
    {
        // grid sizes exceed (1U << 16) - 1 limits, then use a 1-D grid,
        // a natural remap to a 3-D grid is then performed when creating the kernel source code
        size_t elems      = gridX_1d * gridY * gridZ;
        generator.gridDim = {
            static_cast<unsigned int>(DivRoundingUp<size_t>(elems, LAUNCH_BOUNDS_R2C_C2R_KERNEL)),
            1,
            1};
    }

    generator.blockDim = {tileX, tileY, 1};

    RealComplexEvenTransposeSpecs specs{{node.scheme,
                                         node.length.size(),
                                         node.precision,
                                         node.inArrayType,
                                         node.outArrayType,
                                         node.GetCallbackType(enable_callbacks),
                                         node.loadOps,
                                         node.storeOps,
                                         grid3D}};

    generator.generate_name = [=]() { return realcomplex_even_transpose_rtc_kernel_name(specs); };

    generator.generate_src = [=](const std::string& kernel_name) {
        return realcomplex_even_transpose_rtc(kernel_name, specs);
    };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(
            new RTCKernelRealComplexEvenTranspose(kernel_name, code, gridDim, blockDim));
    };

    return generator;
}

RTCKernelArgs RTCKernelRealComplexEvenTranspose::get_launch_args(DeviceCallIn& data)
{
    RTCKernelArgs kargs;

    kargs.append_size_t(data.node->length.size());
    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);
    kargs.append_size_t(data.node->iDist);
    kargs.append_ptr(data.bufOut[0]);
    if(array_type_is_planar(data.node->outArrayType))
        kargs.append_ptr(data.bufOut[1]);
    kargs.append_size_t(data.node->oDist);
    kargs.append_ptr(data.node->twiddles);
    kargs.append_ptr(kargs_lengths(data.node->devKernArg));
    kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
    kargs.append_ptr(kargs_stride_out(data.node->devKernArg));
    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);

    // pass gridY and gridZ to restore a 3-D GPU grid, if needed for large grids
    unsigned int gridY;
    unsigned int gridZ = data.node->batch;

    if(data.node->scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE)
    {
        const unsigned int tileY = RealComplexEvenTransposeSpecs::TileY();
        unsigned int       m     = data.node->length[1];
        unsigned int       dim   = data.node->length.size();

        gridY = ((m - 1) / tileY + 1)
                * (dim > 2 ? static_cast<unsigned int>(data.node->length[2]) : 1);
    }
    else
    {
        unsigned int       m     = data.node->length[1];
        const unsigned int tileY = RealComplexEvenTransposeSpecs::TileY();

        if(data.node->length.size() > 2)
        {
            m = data.node->length[2];
        }

        gridY = std::max<unsigned int>((((m - 1) / 2) + (tileY - 1)) / tileY, 1);
    }

    kargs.append_unsigned_int(gridY);
    kargs.append_unsigned_int(gridZ);

    append_load_store_args(kargs, *data.node);

    return kargs;
}
