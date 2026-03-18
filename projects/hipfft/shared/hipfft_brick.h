// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPFFT_BRICK_H
#define HIPFFT_BRICK_H

#include "ptrdiff.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <iostream> // FIXME: temp


#include "fft_enums.h"
#include "data_layout.h"
#include "../library/include/hipfft/hipfft.h"
#include "../library/include/hipfft/hipfftXt.h"

// column-major ordering on indexes + strides, since these get passed
// directly to rocFFT
struct hipfft_brick
{
    // device that the brick lives on
    int device = 0;

    std::vector<size_t> field_lower;
    std::vector<size_t> field_upper;
    std::vector<size_t> brick_stride;

    // compute the length of this brick
    std::vector<size_t> length() const
    {
        std::vector<size_t> ret;
        for(size_t i = 0; i < field_lower.size(); ++i)
            ret.push_back(field_upper[i] - field_lower[i]);
        return ret;
    }

    // given a (column-major) brick index, return the offset in the field
    size_t field_offset(const std::vector<size_t>& brick_idx,
                        const std::vector<size_t>& field_stride) const
    {
        // find the index in the field
        std::vector<size_t> field_idx;
        for(size_t i = 0; i < brick_idx.size(); ++i)
            field_idx.push_back(brick_idx[i] + field_lower[i]);

        // based on the field's strides, return offset
        return std::inner_product(field_idx.begin(), field_idx.end(), field_stride.begin(), 0);
    }

    // given a (column-major) brick index, return the offset in this brick
    size_t brick_offset(const std::vector<size_t>& brick_idx) const
    {
        // based on the brick's strides, return offset
        return std::inner_product(brick_idx.begin(), brick_idx.end(), brick_stride.begin(), 0);
    }
};


// lengths include batch dimension (col-major), split_dim is counted with 0 = fastest dim.
static void set_bricks(const std::vector<size_t>& length,
                       std::vector<hipfft_brick>& bricks,
                       const size_t               split_dim)
{
    const size_t dim = length.size();

    for(size_t i = 0; i < bricks.size(); ++i)
    {
        auto& brick = bricks[i];

        // lower idx starts at origin, upper is one-past-the-end
        brick.field_lower.resize(dim);
        std::fill(brick.field_lower.begin(), brick.field_lower.end(), 0);
        brick.field_upper = length;

        // length of the brick along the split dimension
        size_t split_len             = length[split_dim] / bricks.size();
        brick.field_lower[split_dim] = split_len * i;
        if(i != bricks.size() - 1)
            brick.field_upper[split_dim] = brick.field_lower[split_dim] + split_len;
        //brick.set_contiguous_stride(); // FIXME
    }
}

// FIXME: documentation.
static void hipfftxt_bricks(const std::vector<size_t>& batchlength,
                            std::vector<hipfft_brick>& bricks,
                            const bool isrealcomplex,
                            const hipfftXtSubFormat subformat)
{

    std::cout << "isrealcomplex: " << isrealcomplex << std::endl;
    
    // We assume that the brick vector has already been allocated, but the brick data is not yet
    // computed.
    if(bricks.size() == 0)
        throw std::runtime_error("Bricks vector needs to be allocated before passing");        
    
    // Format is row-major.
    
    // batchlength includes the (single) batch dimension, so the batchlengths are {batch, X, Y, Z},
    // {batch, X, Y}, or {batch, X}.
    const size_t dim = batchlength.size();
    if(dim < 2)
        throw std::runtime_error("Need at least 1 length and batch dim");
    
    const size_t         nbatch   = batchlength[0];
    fft_result_placement placement;
    fft_io               io;
    const fft_transform_type dft_type
        = isrealcomplex ? fft_transform_type_real_forward : fft_transform_type_complex_forward;

    const bool isherm = isrealcomplex && subformat == HIPFFT_XT_FORMAT_INPLACE_SHUFFLED;
    
    // The subformat tells us which dimension is split.
    // Real in-place data needs extra padding.
    size_t splitdim = 0;
    if(nbatch == 1)
    {
        switch(subformat)
        {
        case HIPFFT_XT_FORMAT_INPUT:
            splitdim = 1; // X-axis is split
            placement = fft_placement_notinplace;
            io = fft_io_in;
            break;
        case HIPFFT_XT_FORMAT_OUTPUT:
            splitdim = 2; // Y-axis is split
            placement = fft_placement_notinplace;
            io = fft_io_out;
            break;
        case HIPFFT_XT_FORMAT_INPLACE:
            splitdim = 1; // X-axis is split
            placement = fft_placement_inplace;
            io = fft_io_in;
            break;
        case HIPFFT_XT_FORMAT_INPLACE_SHUFFLED:
            splitdim = 2; // Y-axis is split
            placement = fft_placement_inplace;
            io = fft_io_out;
            break;
        case HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED:
            // TODO: impliment 1D version.
            // TODO: what do we do with multi-gpu multi-batch 1D transforms?
            throw std::runtime_error("HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED not implimented");
            break;
        case HIPFFT_FORMAT_UNDEFINED:
            throw std::runtime_error("Format passed is HIPFFT_FORMAT_UNDEFINED");
            break;
        default:
            throw std::runtime_error("Invalid subformat");
        }
    }
    else
    {
        // Multi-batch transforms are trivially divided.
        splitdim = 0;
        throw std::runtime_error("Multi-batch multi-gpu transforms not implimented");
    }

    // We are going to put the Hermitian-symmetric length change here:
    auto batchlengthdata = batchlength;
    if(isherm)
    {
        // We have Hermitian-symmetric data
        const auto hindex = batchlengthdata.size() - 1;
        const auto hlength = batchlengthdata[hindex];
        batchlengthdata[hindex] = hlength / 2 + 1;
    }
    
    const auto ngpus = bricks.size();
    for(size_t ibrick = 0; ibrick < bricks.size(); ++ibrick)
    {
        auto& brick = bricks[ibrick];

        brick.field_lower.resize(dim);
        std::fill(brick.field_lower.begin(), brick.field_lower.end(), 0);
        if(ibrick > 0)
            brick.field_lower[splitdim] = bricks[ibrick-1].field_lower[splitdim];

        const size_t splitlen = batchlengthdata[splitdim];
        const size_t bricksplitlen = splitlen / ngpus + (ibrick < splitlen %  bricks.size()? 1 : 0);
        brick.field_upper = batchlengthdata;
        if(ibrick > 0)
        {
            brick.field_lower[splitdim] = bricks[ibrick - 1].field_upper[splitdim];
        }
        brick.field_upper[splitdim] = brick.field_lower[splitdim] + bricksplitlen;

        // FIXME: for 3D transforms, do we need to do this?
        brick.brick_stride = default_strides(isherm ? fft_transform_type_complex_forward :dft_type,
                                             placement,
                                             io,
                                             brick.field_lower,
                                             brick.field_upper);
        std::cout << "new brick_stride:";
        for(auto val : brick.brick_stride)
            std::cout << " " << val;
        std::cout << std::endl;
    }
}

// FIXME: remove this.
// length/strides are column-major.  in/out brick vectors are
// allocated by caller, but coordinates/strides of those bricks are
// filled in by this function
static void set_io_bricks(const std::vector<size_t>& inLength,
                          const std::vector<size_t>& outLength,
                          size_t                     batch,
                          std::vector<hipfft_brick>& inBricks,
                          std::vector<hipfft_brick>& outBricks)
{
    std::vector<size_t> inLengthWithBatch = inLength;
    inLengthWithBatch.push_back(batch);
    std::vector<size_t> outLengthWithBatch = outLength;
    outLengthWithBatch.push_back(batch);

    // for batched FFT, split input on batch, otherwise split input
    // on fastest FFT dim and output on slowest FFT dim
    const size_t in_split_dim
        = batch > 1 ? inLengthWithBatch.size() - 1 : inLengthWithBatch.size() - 2;
    const size_t out_split_dim
        = batch > 1 ? outLengthWithBatch.size() - 1 : outLengthWithBatch.size() - 2;

    set_bricks(inLengthWithBatch, inBricks, in_split_dim);
    set_bricks(outLengthWithBatch, outBricks, out_split_dim);
}

#endif
