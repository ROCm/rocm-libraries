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
#include "rtc_transpose_gen.h"
#include "rtc_transpose_kernel.h"
#include "tree_node.h"

RTCKernel::RTCGenerator RTCKernelTranspose::generate_from_node(const LeafNode&    node,
                                                               const std::string& gpu_arch,
                                                               bool               enable_callbacks)
{
    const auto& length = node.length;

    RTCGenerator generator;
    if(node.scheme != CS_KERNEL_TRANSPOSE && node.scheme != CS_KERNEL_TRANSPOSE_XY_Z
       && node.scheme != CS_KERNEL_TRANSPOSE_Z_XY)
        return generator;

    unsigned int tileX = node.precision == rocfft_precision_single ? 64 : 32;
    unsigned int tileY = node.precision == rocfft_precision_single ? 16 : 32;

    // grid Y counts rows on dims Y+Z, sliced into tiles of tileX.
    // grid Z counts any dims beyond Y+Z, plus batch
    unsigned int gridYrows = length[1] * (length.size() > 2 ? length[2] : 1);
    auto         highdim   = std::min<size_t>(length.size(), 3);

    unsigned int gridX = DivRoundingUp<unsigned int>(length[0], tileX);
    unsigned int gridY = DivRoundingUp<unsigned int>(gridYrows, tileX);
    unsigned int gridZ = std::accumulate(
        length.begin() + highdim, length.end(), node.batch, std::multiplies<unsigned int>());

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
        generator.gridDim = {gridX * gridY * gridZ, 1, 1};
    }

    generator.blockDim = {tileX, tileY};

    size_t largeTwdSteps = 0;
    if(node.large1D > (size_t)256 * 256 * 256 * 256)
        throw std::runtime_error("large1D twiddle size too large error");
    else if(node.large1D > (size_t)256 * 256 * 256)
        largeTwdSteps = 4;
    else if(node.large1D > (size_t)256 * 256)
        largeTwdSteps = 3;
    // TODO- possibly using a smaller LargeTwdBase for transpose by large_twiddle_base
    else if(node.large1D > (size_t)256)
        largeTwdSteps = 2;
    else if(node.large1D > 0)
        largeTwdSteps = 1;

    // check the length along the fast output dimension to decide if
    // we should do diagonal block ordering
    size_t fastOut = node.length[1];
    // diagonal ordering only seems to help 2D cases, not 3D
    bool diagonal = (fastOut % 256) == 0 && (node.outStride[0] % 256 == 0)
                    && node.scheme == CS_KERNEL_TRANSPOSE;

    bool tileAligned = node.length[0] % tileX == 0 && node.length[1] % tileX == 0;

    TransposeSpecs specs{tileX,
                         tileY,
                         node.length.size(),
                         node.precision,
                         node.inArrayType,
                         node.outArrayType,
                         largeTwdSteps,
                         node.direction,
                         diagonal,
                         tileAligned,
                         node.GetCallbackType(enable_callbacks),
                         node.loadOps,
                         node.storeOps,
                         grid3D};

    generator.generate_name = [=]() { return transpose_rtc_kernel_name(specs); };

    generator.generate_src
        = [=](const std::string& kernel_name) { return transpose_rtc(kernel_name, specs); };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(
            new RTCKernelTranspose(kernel_name, code, gridDim, blockDim));
    };
    return generator;
}

RTCKernelArgs RTCKernelTranspose::get_launch_args(DeviceCallIn& data)
{
    RTCKernelArgs kargs;
    kargs.append_ptr(data.bufIn[0]);
    if(array_type_is_planar(data.node->inArrayType))
        kargs.append_ptr(data.bufIn[1]);
    kargs.append_ptr(data.bufOut[0]);
    if(array_type_is_planar(data.node->outArrayType))
        kargs.append_ptr(data.bufOut[1]);
    kargs.append_ptr(data.node->twiddles_large);

    auto num_lengths = data.node->length.size();
    kargs.append_unsigned_int(num_lengths);
    kargs.append_unsigned_int(data.node->length[0]);
    kargs.append_unsigned_int(data.node->length[1]);
    kargs.append_unsigned_int(num_lengths > 2 ? data.node->length[2] : 1);
    kargs.append_ptr(kargs_lengths(data.node->devKernArg));

    kargs.append_unsigned_int(data.node->inStride[0]);
    kargs.append_unsigned_int(data.node->inStride[1]);
    kargs.append_unsigned_int(num_lengths > 2 ? data.node->inStride[2] : 0);
    kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
    kargs.append_unsigned_int(data.node->iDist);

    kargs.append_unsigned_int(data.node->outStride[0]);
    kargs.append_unsigned_int(data.node->outStride[1]);
    kargs.append_unsigned_int(num_lengths > 2 ? data.node->outStride[2] : 0);
    kargs.append_ptr(kargs_stride_out(data.node->devKernArg));
    kargs.append_unsigned_int(data.node->oDist);

    // pass gridX, gridY and gridZ to restore a 3-D GPU grid, if needed for large grids
    unsigned int tileX = data.node->precision == rocfft_precision_single ? 64 : 32;
    unsigned int gridYrows
        = data.node->length[1] * (data.node->length.size() > 2 ? data.node->length[2] : 1);
    auto highdim = std::min<size_t>(data.node->length.size(), 3);

    unsigned int gridX = DivRoundingUp<unsigned int>(data.node->length[0], tileX);
    unsigned int gridY = DivRoundingUp<unsigned int>(gridYrows, tileX);
    unsigned int gridZ = std::accumulate(data.node->length.begin() + highdim,
                                         data.node->length.end(),
                                         data.node->batch,
                                         std::multiplies<unsigned int>());

    kargs.append_unsigned_int(gridX);
    kargs.append_unsigned_int(gridY);
    kargs.append_unsigned_int(gridZ);

    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);

    append_load_store_args(kargs, *data.node);
    return kargs;
}
