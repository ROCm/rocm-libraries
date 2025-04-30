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

#ifndef ENUM_PRINTER_H
#define ENUM_PRINTER_H

#include <string>

#include "../../../shared/rocfft_complex.h"
#include "../device/kernels/common.h"
#include "rocfft/rocfft.h"

enum OperatingBuffer
{
    OB_UNINIT              = 0b00000,
    OB_USER_IN             = 0b00001,
    OB_USER_OUT            = 0b00010,
    OB_TEMP                = 0b00100,
    OB_TEMP_CMPLX_FOR_REAL = 0b01000,
    OB_TEMP_BLUESTEIN      = 0b10000,
};

enum PlacementCode
{
    PC_IP,
    PC_OP,
    PC_UNSET,
};

// the decision strategy for buffer assigment
enum rocfft_optimize_strategy
{
    rocfft_optimize_min_buffer, // minimize number of buffers, possibly fewer fusions
    rocfft_optimize_balance, // balance between buffer and fusion
    rocfft_optimize_max_fusion, // maximize number of fusions, possibly more buffers
};

std::string PrintOperatingBuffer(const OperatingBuffer ob);
std::string PrintOperatingBufferCode(const OperatingBuffer ob);
std::string PrintOptimizeStrategy(const rocfft_optimize_strategy ros);
std::string PrintDirectToFromRegMode(const DirectRegType ty);
std::string PrintArrayType(const rocfft_array_type aryType);
std::string PrintPlacement(const rocfft_result_placement placement);
std::string PrintPlacementCode(const PlacementCode placementCode);
std::string PrintEBType(const EmbeddedType ebtype);
std::string PrintSBRCTransposeType(const SBRC_TRANSPOSE_TYPE ty);
std::string PrintPrecision(const rocfft_precision pre);

rocfft_array_type       StrToArrayType(const std::string& str);
rocfft_result_placement StrToPlacement(const std::string& str);
PlacementCode           StrToPlacementCode(const std::string& str);
EmbeddedType            StrToEBType(const std::string& str);
SBRC_TRANSPOSE_TYPE     StrToSBRCTransType(const std::string& str);
rocfft_precision        StrToPrecision(const std::string& str);

#endif
