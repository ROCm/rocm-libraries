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

// column-major ordering on indexes + strides, since these get passed
// directly to rocFFT
struct hipfft_brick
{
    // device that the brick lives on
    int device = 0;

    std::vector<size_t> field_lower;
    std::vector<size_t> field_upper;
    std::vector<size_t> brick_stride;

    size_t min_size = 0;

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

    // Set contiguous strides on this brick
    void set_contiguous_stride(const bool r2c)
    {
        brick_stride = {1};
        const auto len     = length();
        for(size_t idx = 0; idx < len.size() - 1; ++idx) {
            brick_stride.push_back(brick_stride[idx] * len[idx] + (idx == 0 ? 2 : 0));
        }
    }
};

// lengths include batch dimension (col-major), split_dim is counted with 0 = fastest dim.
static void set_bricks(const std::vector<size_t>& length,
                       std::vector<hipfft_brick>& bricks,
                       const size_t               split_dim,
                       const bool r2c)
{
    const size_t dim = length.size();

    for(size_t idx = 0; idx < bricks.size(); ++idx)
    {
        auto& brick = bricks[idx];

        // lower idx starts at origin, upper is one-past-the-end
        brick.field_lower.resize(dim);
        brick.field_upper.resize(dim);
        std::fill(brick.field_lower.begin(), brick.field_lower.end(), 0);

        // All of the remainders get put on the low-index bricks.
        brick.field_lower[split_dim] = length[split_dim] / bricks.size() * idx
                                       + std::min(idx, length[split_dim] % bricks.size());
        size_t split_len
            = length[split_dim] / bricks.size() + (idx < length[split_dim] % bricks.size() ? 1 : 0);

        brick.field_upper = length;
        if(idx != bricks.size() - 1)
            brick.field_upper[split_dim] = brick.field_lower[split_dim] + split_len;
        
        brick.set_contiguous_stride(r2c);

        // work out how big a buffer we need to allocate
        std::vector<size_t> brick_len(dim);
        for(size_t d = 0; d < dim; ++d)
            brick_len[d] = brick.field_upper[d] - brick.field_lower[d];
        // FIXME: r2c
        brick.min_size
            = std::max(brick.min_size, compute_ptrdiff(brick_len, brick.brick_stride, 0, 0));
    }
}

// length/strides are column-major.  in/out brick vectors are
// allocated by caller, but coordinates/strides of those bricks are
// filled in by this function
static void set_io_bricks(const std::vector<size_t>& inLength,
                          const std::vector<size_t>& outLength,
                          const size_t               batch,
                          std::vector<hipfft_brick>& inBricks,
                          std::vector<hipfft_brick>& outBricks,
                          bool r2c)
{
    // Batched transforms are split between batches.
    
    // Single-batch multi-dimensional transforms are split along the slower dimension for input, and
    // the second-slower dimension for output.

    // Single-batch 1D transforms are split in the single dimension.

    const size_t rank = inLength.size();
    const size_t in_split_dim  = batch == 1 ? rank - 1 : rank;
    const size_t out_split_dim = batch == 1 ? (rank == 1 ? 0 : rank - 2) : rank;

    std::vector<size_t> inLengthWithBatch = inLength;
    inLengthWithBatch.push_back(batch);
    std::vector<size_t> outLengthWithBatch = outLength;
    outLengthWithBatch.push_back(batch);

    // FIXME: strides?
    set_bricks(inLengthWithBatch, inBricks, in_split_dim, r2c);
    set_bricks(outLengthWithBatch, outBricks, out_split_dim, r2c);
}

#endif
