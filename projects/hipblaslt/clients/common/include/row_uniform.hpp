// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// Support for the row_uniform_check argument: makes the GEMM inputs uniform along M, so every
// row of D must come out bitwise identical when the library keeps one summation order per row.
//
// Matrices are column-major with leading dimension ld and batch stride stride, all in elements.
// element_size is 1, 2, 4, 8 or 16 bytes; other sizes return hipErrorInvalidValue.

// Copies stored row 0 of each column into rows 1..rows-1.
hipError_t row_uniform_fill_rows_device(void*       buf,
                                        size_t      element_size,
                                        int64_t     rows,
                                        int64_t     cols,
                                        int64_t     ld,
                                        int64_t     stride,
                                        int64_t     batches,
                                        hipStream_t stream);

// Copies stored column 0 into columns 1..cols-1.
hipError_t row_uniform_fill_cols_device(void*       buf,
                                        size_t      element_size,
                                        int64_t     rows,
                                        int64_t     cols,
                                        int64_t     ld,
                                        int64_t     stride,
                                        int64_t     batches,
                                        hipStream_t stream);

struct RowUniformDeviceResult
{
    unsigned long long bad_cols; // columns (over all batches) with a row that differs from row 0
    // The first differing element in (row, column) order, packed as row << 32 | column;
    // UINT64_MAX when bad_cols is 0.
    unsigned long long first;

    int64_t first_row() const
    {
        return bad_cols ? int64_t(first >> 32) : -1;
    }
    int64_t first_col() const
    {
        return bad_cols ? int64_t(first & 0xffffffffull) : -1;
    }
};

hipError_t row_uniform_reset_device(RowUniformDeviceResult* result, hipStream_t stream);

// Adds to *result every column of d whose rows are not all bitwise equal to row 0.
hipError_t row_uniform_check_device(const void*             d,
                                    size_t                  element_size,
                                    int64_t                 rows,
                                    int64_t                 cols,
                                    int64_t                 ld,
                                    int64_t                 stride,
                                    int64_t                 batches,
                                    RowUniformDeviceResult* result,
                                    hipStream_t             stream);

// One solution's outcome, as printed in the bench result line.
struct RowUniformResult
{
    enum class Status
    {
        off,
        pass,
        fail,
        skip
    };
    Status  status    = Status::off;
    int64_t bad_cols  = 0;
    int64_t first_row = -1;

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
inline void row_uniform_fill_rows_host(void*   buf,
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

// Sets all n elements to element 0.
inline void row_uniform_fill_constant_host(void* buf, size_t element_size, size_t n)
{
    auto* base = static_cast<unsigned char*>(buf);
    for(size_t i = 1; i < n; i++)
        std::memcpy(base + i * element_size, base, element_size);
}
