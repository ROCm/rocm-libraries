// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "uniform_check.hpp"

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

    __global__ void reset_kernel(UniformCheckDeviceResult* result)
    {
        *result = {0, kNone};
    }

    __device__ unsigned load_relaxed(const unsigned* p)
    {
        return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    __device__ unsigned long long load_relaxed(const unsigned long long* p)
    {
        return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    // Records a difference at packed position pos in line, a (batch, line) index. Only the thread
    // that sets the line's flag counts the line, so result->bad counts each line once.
    __device__ void record(unsigned*                 flags,
                           uint64_t                  line,
                           unsigned long long        pos,
                           UniformCheckDeviceResult* result)
    {
        unsigned*      word = flags + line / 32;
        const unsigned bit  = 1u << (line % 32);
        if(!(load_relaxed(word) & bit) && !(atomicOr(word, bit) & bit))
            atomicAdd(&result->bad, 1ull);
        if(pos < load_relaxed(&result->first))
            atomicMin(&result->first, pos);
    }

    // Grid-stride over every element of every batch in memory order. Element (r, c) of a batch
    // is compared with (0, c), flagging column c, or when COL with (r, 0), flagging row r.
    template <typename T, bool COL>
    __global__ void check_kernel(const T*                  d,
                                 uint64_t                  rows,
                                 uint64_t                  cols,
                                 uint64_t                  ld,
                                 uint64_t                  stride,
                                 uint64_t                  total,
                                 unsigned*                 flags,
                                 UniformCheckDeviceResult* result)
    {
        const uint64_t step = uint64_t(gridDim.x) * blockDim.x;
        for(uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < total; i += step)
        {
            const uint64_t r = i % rows;
            const uint64_t c = (i / rows) % cols;
            const uint64_t b = i / rows / cols;
            const T*       m = d + b * stride;
            if(!COL && r != 0 && m[c * ld + r] != m[c * ld])
                record(flags, b * cols + c, (unsigned long long)r << 32 | c, result);
            if(COL && c != 0 && m[c * ld + r] != m[r])
                record(flags, b * rows + r, (unsigned long long)c << 32 | r, result);
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
    hipError_t launch_check(const void*               d,
                            int64_t                   rows,
                            int64_t                   cols,
                            int64_t                   ld,
                            int64_t                   stride,
                            int64_t                   batches,
                            bool                      col,
                            void*                     flags,
                            UniformCheckDeviceResult* result,
                            hipStream_t               stream)
    {
        const uint64_t total   = uint64_t(rows) * uint64_t(cols) * uint64_t(batches);
        constexpr int  threads = 256;
        const auto blocks = unsigned(std::min<uint64_t>((total + threads - 1) / threads, 65536));
        hipLaunchKernelGGL((col ? check_kernel<T, true> : check_kernel<T, false>),
                           dim3(blocks),
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const T*>(d),
                           uint64_t(rows),
                           uint64_t(cols),
                           uint64_t(ld),
                           uint64_t(stride),
                           total,
                           static_cast<unsigned*>(flags),
                           result);
        return hipGetLastError();
    }
}

hipError_t uniform_check_fill_rows_device(void*       buf,
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

hipError_t uniform_check_fill_cols_device(void*       buf,
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

hipError_t uniform_check_reset_device(UniformCheckDeviceResult* result, hipStream_t stream)
{
    hipLaunchKernelGGL(reset_kernel, dim3(1), dim3(1), 0, stream, result);
    return hipGetLastError();
}

size_t uniform_check_flags_bytes(int64_t lines, int64_t batches)
{
    return (size_t(lines) * size_t(batches) + 31) / 32 * sizeof(unsigned);
}

hipError_t uniform_check_device(const void*               d,
                                size_t                    element_size,
                                int64_t                   rows,
                                int64_t                   cols,
                                int64_t                   ld,
                                int64_t                   stride,
                                int64_t                   batches,
                                bool                      col,
                                void*                     flags,
                                UniformCheckDeviceResult* result,
                                hipStream_t               stream)
{
    // A line of one element has nothing to compare.
    if((col ? cols : rows) <= 1 || (col ? rows : cols) <= 0 || batches <= 0)
        return hipSuccess;
    if(hipError_t err
       = hipMemsetAsync(flags, 0, uniform_check_flags_bytes(col ? rows : cols, batches), stream);
       err != hipSuccess)
        return err;
    switch(element_size)
    {
    case 1:
        return launch_check<uint8_t>(
            d, rows, cols, ld, stride, batches, col, flags, result, stream);
    case 2:
        return launch_check<uint16_t>(
            d, rows, cols, ld, stride, batches, col, flags, result, stream);
    case 4:
        return launch_check<uint32_t>(
            d, rows, cols, ld, stride, batches, col, flags, result, stream);
    case 8:
        return launch_check<uint64_t>(
            d, rows, cols, ld, stride, batches, col, flags, result, stream);
    case 16:
        return launch_check<Word16>(d, rows, cols, ld, stride, batches, col, flags, result, stream);
    default:
        return hipErrorInvalidValue;
    }
}
