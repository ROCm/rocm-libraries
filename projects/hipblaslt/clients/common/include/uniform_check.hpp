// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

// Support for the uniform_check argument: makes the GEMM inputs uniform along M (row direction)
// or N (column direction), so every row, or column, of D must come out bitwise identical when
// the library keeps one summation order for all of them.
//
// Matrices are column-major with leading dimension ld and batch stride stride, all in elements.
// element_size is 1, 2, 4, 8 or 16 bytes; other sizes return hipErrorInvalidValue.

// Copies stored row 0 of each column into rows 1..rows-1.
hipError_t uniform_check_fill_rows_device(void*       buf,
                                          size_t      element_size,
                                          int64_t     rows,
                                          int64_t     cols,
                                          int64_t     ld,
                                          int64_t     stride,
                                          int64_t     batches,
                                          hipStream_t stream);

// Copies stored column 0 into columns 1..cols-1.
hipError_t uniform_check_fill_cols_device(void*       buf,
                                          size_t      element_size,
                                          int64_t     rows,
                                          int64_t     cols,
                                          int64_t     ld,
                                          int64_t     stride,
                                          int64_t     batches,
                                          hipStream_t stream);

// A line is a column of D for the row direction and a row of D for the column direction.
struct UniformCheckDeviceResult
{
    unsigned long long bad; // (batch, line) pairs of D with an element that differs
    // The first differing element, packed as its position along the line << 32 | the line, so
    // row << 32 | column for the row direction and column << 32 | row for the column direction;
    // UINT64_MAX when bad is 0.
    unsigned long long first;

    int64_t first_pos() const
    {
        return bad ? int64_t(first >> 32) : -1;
    }
    int64_t first_line() const
    {
        return bad ? int64_t(first & 0xffffffffull) : -1;
    }
};

hipError_t uniform_check_reset_device(UniformCheckDeviceResult* result, hipStream_t stream);

// Bytes of flags scratch for lines x batches (batch, line) pairs.
size_t uniform_check_flags_bytes(int64_t lines, int64_t batches);

// Compares every element of d bitwise with row 0 of its column and batch (row direction, lines
// are columns) or, when col, with column 0 of its row and batch (column direction, lines are
// rows). Each (batch, line) with a difference adds 1 to result->bad, and result->first takes the
// earliest difference. flags is device scratch of uniform_check_flags_bytes(lines, batches)
// bytes, cleared here.
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
                                hipStream_t               stream);

// One direction's outcome for one solution, as printed in the bench result line.
struct UniformCheckResult
{
    enum class Status
    {
        off,
        pass,
        fail,
        skip
    };
    Status  status = Status::off;
    int64_t bad    = 0; // lines of D, over all batches, that are not uniform
    int64_t first  = -1; // position along the line of the first difference

    const char* status_name() const
    {
        switch(status)
        {
        case Status::pass:
            return "pass";
        case Status::fail:
            return "FAIL";
        case Status::skip:
            return "skip";
        default:
            return "off";
        }
    }
};

// Host-side counterparts of the fill kernels, on bytes; element_size may be any size.
inline void uniform_check_fill_rows_host(void*   buf,
                                         size_t  element_size,
                                         int64_t rows,
                                         int64_t cols,
                                         int64_t ld,
                                         int64_t stride,
                                         int64_t batches)
{
    auto* base = static_cast<unsigned char*>(buf);
    for(int64_t b = 0; b < batches; b++)
        for(int64_t c = 0; c < cols; c++)
        {
            unsigned char* col = base + (b * stride + c * ld) * element_size;
            for(int64_t r = 1; r < rows; r++)
                std::memcpy(col + r * element_size, col, element_size);
        }
}

// Copies stored column 0 into columns 1..cols-1.
inline void uniform_check_fill_cols_host(void*   buf,
                                         size_t  element_size,
                                         int64_t rows,
                                         int64_t cols,
                                         int64_t ld,
                                         int64_t stride,
                                         int64_t batches)
{
    auto* base = static_cast<unsigned char*>(buf);
    for(int64_t b = 0; b < batches; b++)
    {
        unsigned char* m = base + b * stride * element_size;
        for(int64_t c = 1; c < cols; c++)
            std::memcpy(m + c * ld * element_size, m, rows * element_size);
    }
}

// Sets all n elements to element 0.
inline void uniform_check_fill_constant_host(void* buf, size_t element_size, size_t n)
{
    auto* base = static_cast<unsigned char*>(buf);
    for(size_t i = 1; i < n; i++)
        std::memcpy(base + i * element_size, base, element_size);
}
