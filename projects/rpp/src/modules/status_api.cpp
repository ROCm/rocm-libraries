/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "rpp.h"

// The switch below deliberately has no default label, so that -Wswitch flags a newly added status
// code that has not been given a string here.

extern "C" const char* rppGetStatusString(RppStatus status) {
    switch (status) {
        case RPP_SUCCESS:
            return "RPP_SUCCESS";
        case RPP_ERROR:
            return "RPP_ERROR";
        case RPP_ERROR_INVALID_ARGUMENTS:
            return "RPP_ERROR_INVALID_ARGUMENTS";
        case RPP_ERROR_LOW_OFFSET:
            return "RPP_ERROR_LOW_OFFSET";
        case RPP_ERROR_ZERO_DIVISION:
            return "RPP_ERROR_ZERO_DIVISION";
        case RPP_ERROR_HIGH_SRC_DIMENSION:
            return "RPP_ERROR_HIGH_SRC_DIMENSION";
        case RPP_ERROR_NOT_IMPLEMENTED:
            return "RPP_ERROR_NOT_IMPLEMENTED";
        case RPP_ERROR_INVALID_SRC_CHANNELS:
            return "RPP_ERROR_INVALID_SRC_CHANNELS";
        case RPP_ERROR_INVALID_DST_CHANNELS:
            return "RPP_ERROR_INVALID_DST_CHANNELS";
        case RPP_ERROR_INVALID_SRC_LAYOUT:
            return "RPP_ERROR_INVALID_SRC_LAYOUT";
        case RPP_ERROR_INVALID_DST_LAYOUT:
            return "RPP_ERROR_INVALID_DST_LAYOUT";
        case RPP_ERROR_INVALID_SRC_DATATYPE:
            return "RPP_ERROR_INVALID_SRC_DATATYPE";
        case RPP_ERROR_INVALID_DST_DATATYPE:
            return "RPP_ERROR_INVALID_DST_DATATYPE";
        case RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE:
            return "RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE";
        case RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH:
            return "RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH";
        case RPP_ERROR_INVALID_PARAMETER_DATATYPE:
            return "RPP_ERROR_INVALID_PARAMETER_DATATYPE";
        case RPP_ERROR_NOT_ENOUGH_MEMORY:
            return "RPP_ERROR_NOT_ENOUGH_MEMORY";
        case RPP_ERROR_OUT_OF_BOUND_SRC_ROI:
            return "RPP_ERROR_OUT_OF_BOUND_SRC_ROI";
        case RPP_ERROR_LAYOUT_MISMATCH:
            return "RPP_ERROR_LAYOUT_MISMATCH";
        case RPP_ERROR_INVALID_CHANNELS:
            return "RPP_ERROR_INVALID_CHANNELS";
        case RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH:
            return "RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH";
        case RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE:
            return "RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE";
        case RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE:
            return "RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE";
        case RPP_ERROR_INVALID_SRC_DIMS:
            return "RPP_ERROR_INVALID_SRC_DIMS";
        case RPP_ERROR_INVALID_DST_DIMS:
            return "RPP_ERROR_INVALID_DST_DIMS";
        case RPP_ERROR_INVALID_DIM_LENGTHS:
            return "RPP_ERROR_INVALID_DIM_LENGTHS";
        case RPP_ERROR_INVALID_AXIS:
            return "RPP_ERROR_INVALID_AXIS";
        case RPP_ERROR_INCOMPATIBLE_BACKEND:
            return "RPP_ERROR_INCOMPATIBLE_BACKEND";
        case RPP_ERROR_HIP_LAUNCH:
            return "RPP_ERROR_HIP_LAUNCH";
        case RPP_ERROR_HIP_RUNTIME:
            return "RPP_ERROR_HIP_RUNTIME";
    }
    return "RPP_STATUS_UNKNOWN";
}
