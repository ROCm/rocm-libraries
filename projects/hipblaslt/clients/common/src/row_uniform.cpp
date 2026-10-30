// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "row_uniform.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>

namespace
{
    constexpr unsigned long long kNone = std::numeric_limits<unsigned long long>::max();

    struct Word16
    {
        uint64_t lo, hi;
        __device__ bool operator!=(const Word16& o) const
        {
            return lo != o.lo || hi != o.hi;
        }
    };

    // Element (r, c) of every batch takes the value of (0, c) when FILL_ROWS, else of (r, 0).
    template <typename T, bool FILL_ROWS>
    __global__ void fill_kernel(
        T* buf, uint64_t rows, uint64_t cols, uint64_t ld, uint64_t stride, uint64_t batches)
    {
        const uint64_t total = rows * cols * batches;
        const uint64_t step  = uint64_t(gridDim.x) * blockDim.x;
        for(uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < total; i += step)
        {
            const uint64_t r = i % rows;
            const uint64_t c = (i / rows) % cols;
            const uint64_t b = i / (rows * cols);
            T*             m = buf + b * stride;
            if(FILL_ROWS && r != 0)
                m[c * ld + r] = m[c * ld];
            else if(!FILL_ROWS && c != 0)
                m[c * ld + r] = m[r];
        }
    }

    __global__ void reset_kernel(RowUniformDeviceResult* result)
    {
        *result = {0, kNone};
    }

    // A block per column (grid-stride over columns) and batch. Each thread scans rows in
    // increasing order and stops at its first difference, so the block minimum is the column's
    // first differing row.
    template <typename T>
    __global__ void check_kernel(const T*                d,
                                 uint64_t                rows,
                                 uint64_t                cols,
                                 uint64_t                ld,
                                 uint64_t                stride,
                                 RowUniformDeviceResult* result)
    {
        __shared__ unsigned long long first;
        for(uint64_t c = blockIdx.x; c < cols; c += gridDim.x)
        {
            if(threadIdx.x == 0)
                first = kNone;
            __syncthreads();

            const T* col = d + blockIdx.y * stride + c * ld;
            const T  ref = col[0];
            for(uint64_t r = 1 + threadIdx.x; r < rows; r += blockDim.x)
                if(col[r] != ref)
                {
                    atomicMin(&first, (unsigned long long)r);
                    break;
                }
            __syncthreads();

            if(threadIdx.x == 0 && first != kNone)
            {
                atomicAdd(&result->bad_cols, 1ull);
                atomicMin(&result->first, first << 32 | c);
            }
            __syncthreads();
        }
    }

    template <typename T, bool FILL_ROWS>
    hipError_t launch_fill(void*       buf,
                           int64_t     rows,
                           int64_t     cols,
                           int64_t     ld,
                           int64_t     stride,
                           int64_t     batches,
                           hipStream_t stream)
    {
        const uint64_t total = uint64_t(rows) * uint64_t(cols) * uint64_t(batches);
        if(total == 0)
            return hipSuccess;
        constexpr int threads = 256;
        const auto    blocks  = unsigned(std::min<uint64_t>((total + threads - 1) / threads, 65536));
        hipLaunchKernelGGL((fill_kernel<T, FILL_ROWS>),
                           dim3(blocks),
                           dim3(threads),
                           0,
                           stream,
                           static_cast<T*>(buf),
                           uint64_t(rows),
                           uint64_t(cols),
                           uint64_t(ld),
                           uint64_t(stride),
                           uint64_t(batches));
        return hipGetLastError();
    }

    template <bool FILL_ROWS>
    hipError_t fill(void*       buf,
                    size_t      element_size,
                    int64_t     rows,
                    int64_t     cols,
                    int64_t     ld,
                    int64_t     stride,
                    int64_t     batches,
                    hipStream_t stream)
    {
        switch(element_size)
        {
        case 1:
            return launch_fill<uint8_t, FILL_ROWS>(buf, rows, cols, ld, stride, batches, stream);
        case 2:
            return launch_fill<uint16_t, FILL_ROWS>(buf, rows, cols, ld, stride, batches, stream);
        case 4:
            return launch_fill<uint32_t, FILL_ROWS>(buf, rows, cols, ld, stride, batches, stream);
        case 8:
            return launch_fill<uint64_t, FILL_ROWS>(buf, rows, cols, ld, stride, batches, stream);
        case 16:
            return launch_fill<Word16, FILL_ROWS>(buf, rows, cols, ld, stride, batches, stream);
        default:
            return hipErrorInvalidValue;
        }
    }

    template <typename T>
    hipError_t launch_check(const void*             d,
                            int64_t                 rows,
                            int64_t                 cols,
                            int64_t                 ld,
                            int64_t                 stride,
                            int64_t                 batches,
                            RowUniformDeviceResult* result,
                            hipStream_t             stream)
    {
        if(rows <= 1 || cols <= 0 || batches <= 0)
            return hipSuccess;
        constexpr int     threads    = 256;
        constexpr int64_t maxBatches = 65535; // gridDim.y limit
        constexpr int64_t maxBlocksX = 1 << 20;
        for(int64_t b = 0; b < batches; b += maxBatches)
        {
            hipLaunchKernelGGL((check_kernel<T>),
                               dim3(unsigned(std::min(maxBlocksX, cols)),
                                    unsigned(std::min(maxBatches, batches - b))),
                               dim3(threads),
                               0,
                               stream,
                               static_cast<const T*>(d) + b * stride,
                               uint64_t(rows),
                               uint64_t(cols),
                               uint64_t(ld),
                               uint64_t(stride),
                               result);
            if(hipError_t err = hipGetLastError(); err != hipSuccess)
                return err;
        }
        return hipSuccess;
    }
}

hipError_t row_uniform_fill_rows_device(void*       buf,
                                        size_t      element_size,
                                        int64_t     rows,
                                        int64_t     cols,
                                        int64_t     ld,
                                        int64_t     stride,
                                        int64_t     batches,
                                        hipStream_t stream)
{
    return fill<true>(buf, element_size, rows, cols, ld, stride, batches, stream);
}

hipError_t row_uniform_fill_cols_device(void*       buf,
                                        size_t      element_size,
                                        int64_t     rows,
                                        int64_t     cols,
                                        int64_t     ld,
                                        int64_t     stride,
                                        int64_t     batches,
                                        hipStream_t stream)
{
    return fill<false>(buf, element_size, rows, cols, ld, stride, batches, stream);
}

hipError_t row_uniform_reset_device(RowUniformDeviceResult* result, hipStream_t stream)
{
    hipLaunchKernelGGL(reset_kernel, dim3(1), dim3(1), 0, stream, result);
    return hipGetLastError();
}

hipError_t row_uniform_check_device(const void*             d,
                                    size_t                  element_size,
                                    int64_t                 rows,
                                    int64_t                 cols,
                                    int64_t                 ld,
                                    int64_t                 stride,
                                    int64_t                 batches,
                                    RowUniformDeviceResult* result,
                                    hipStream_t             stream)
{
    switch(element_size)
    {
    case 1:
        return launch_check<uint8_t>(d, rows, cols, ld, stride, batches, result, stream);
    case 2:
        return launch_check<uint16_t>(d, rows, cols, ld, stride, batches, result, stream);
    case 4:
        return launch_check<uint32_t>(d, rows, cols, ld, stride, batches, result, stream);
    case 8:
        return launch_check<uint64_t>(d, rows, cols, ld, stride, batches, result, stream);
    case 16:
        return launch_check<Word16>(d, rows, cols, ld, stride, batches, result, stream);
    default:
        return hipErrorInvalidValue;
    }
}
