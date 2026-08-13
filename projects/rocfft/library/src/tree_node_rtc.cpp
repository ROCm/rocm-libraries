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

// TreeNode methods that kernel generation needs.  These live apart
// from tree_node.cpp because the RTC helper executables link
// rocfft-rtc-launch without the rest of the rocFFT library.

#include "tree_node.h"

#include "../../shared/ptrdiff.h"

#include <cstdint>

IndexType LeafNode::GetKernelIndexType() const
{
    auto needs_64bit_indexing = [this](io_data_label io) {
        const auto& io_stride     = io == io_data_label::INPUT ? inStride : outStride;
        const auto& io_dist       = io == io_data_label::INPUT ? iDist : oDist;
        const auto& io_array_type = io == io_data_label::INPUT ? inArrayType : outArrayType;
        const auto& io_offset     = io == io_data_label::INPUT ? iOffset : oOffset;
        const auto  io_length     = io == io_data_label::INPUT ? length : GetOutputLength();

        // Hermitian interleaved data may be re-interpreted as real data internally.
        return (io_offset + compute_ptrdiff(io_length, io_stride, batch, io_dist))
                   * (io_array_type == rocfft_array_type_hermitian_interleaved ? 2 : 1)
               > static_cast<size_t>(INT32_MAX) + 1;
    };

    return needs_64bit_indexing(io_data_label::INPUT) || needs_64bit_indexing(io_data_label::OUTPUT)
               ? IndexType::_64BIT
               : IndexType::_32BIT;
}