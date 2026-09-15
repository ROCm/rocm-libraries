/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_STATUS_PARAM_H
#define RPP_TEST_STATUS_PARAM_H

#include <rpp/rpp.h>

#include <ostream>

// gtest finds PrintTo via ADL; RppStatus is a global-namespace C enum, so this
// overload must live in the global namespace too.
inline void PrintTo(RppStatus status, std::ostream* os) {
    switch (status) {
        case RPP_SUCCESS:
            *os << "RPP_SUCCESS";
            return;
        case RPP_ERROR:
            *os << "RPP_ERROR";
            return;
        case RPP_ERROR_INVALID_ARGUMENTS:
            *os << "RPP_ERROR_INVALID_ARGUMENTS";
            return;
        case RPP_ERROR_LOW_OFFSET:
            *os << "RPP_ERROR_LOW_OFFSET";
            return;
        case RPP_ERROR_ZERO_DIVISION:
            *os << "RPP_ERROR_ZERO_DIVISION";
            return;
        case RPP_ERROR_HIGH_SRC_DIMENSION:
            *os << "RPP_ERROR_HIGH_SRC_DIMENSION";
            return;
        case RPP_ERROR_NOT_IMPLEMENTED:
            *os << "RPP_ERROR_NOT_IMPLEMENTED";
            return;
        case RPP_ERROR_INVALID_SRC_CHANNELS:
            *os << "RPP_ERROR_INVALID_SRC_CHANNELS";
            return;
        case RPP_ERROR_INVALID_DST_CHANNELS:
            *os << "RPP_ERROR_INVALID_DST_CHANNELS";
            return;
        case RPP_ERROR_INVALID_SRC_LAYOUT:
            *os << "RPP_ERROR_INVALID_SRC_LAYOUT";
            return;
        case RPP_ERROR_INVALID_DST_LAYOUT:
            *os << "RPP_ERROR_INVALID_DST_LAYOUT";
            return;
        case RPP_ERROR_INVALID_SRC_DATATYPE:
            *os << "RPP_ERROR_INVALID_SRC_DATATYPE";
            return;
        case RPP_ERROR_INVALID_DST_DATATYPE:
            *os << "RPP_ERROR_INVALID_DST_DATATYPE";
            return;
        case RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE:
            *os << "RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE";
            return;
        case RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH:
            *os << "RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH";
            return;
        case RPP_ERROR_INVALID_PARAMETER_DATATYPE:
            *os << "RPP_ERROR_INVALID_PARAMETER_DATATYPE";
            return;
        case RPP_ERROR_NOT_ENOUGH_MEMORY:
            *os << "RPP_ERROR_NOT_ENOUGH_MEMORY";
            return;
        case RPP_ERROR_OUT_OF_BOUND_SRC_ROI:
            *os << "RPP_ERROR_OUT_OF_BOUND_SRC_ROI";
            return;
        case RPP_ERROR_LAYOUT_MISMATCH:
            *os << "RPP_ERROR_LAYOUT_MISMATCH";
            return;
        case RPP_ERROR_INVALID_CHANNELS:
            *os << "RPP_ERROR_INVALID_CHANNELS";
            return;
        case RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH:
            *os << "RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH";
            return;
        case RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE:
            *os << "RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE";
            return;
        case RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE:
            *os << "RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE";
            return;
        case RPP_ERROR_INVALID_SRC_DIMS:
            *os << "RPP_ERROR_INVALID_SRC_DIMS";
            return;
        case RPP_ERROR_INVALID_DST_DIMS:
            *os << "RPP_ERROR_INVALID_DST_DIMS";
            return;
        case RPP_ERROR_INVALID_DIM_LENGTHS:
            *os << "RPP_ERROR_INVALID_DIM_LENGTHS";
            return;
        case RPP_ERROR_INVALID_AXIS:
            *os << "RPP_ERROR_INVALID_AXIS";
            return;
        case RPP_ERROR_INCOMPATIBLE_BACKEND:
            *os << "RPP_ERROR_INCOMPATIBLE_BACKEND";
            return;
        case RPP_ERROR_HIP_LAUNCH:
            *os << "RPP_ERROR_HIP_LAUNCH";
            return;
        case RPP_ERROR_HIP_RUNTIME:
            *os << "RPP_ERROR_HIP_RUNTIME";
            return;
        default:
            *os << "UNKNOWN_RPP_STATUS(" << static_cast<int>(status) << ")";
            return;
    }
}

#endif  // RPP_TEST_STATUS_PARAM_H
