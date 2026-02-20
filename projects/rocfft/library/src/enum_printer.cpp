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

#include "enum_printer.h"

#include <map>
#include <set>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)
#define STRENUM(x) TO_STR(x), x

//
// precision map
//
static std::map<rocfft_precision, const char*> PrecisionToStrMap()
{
    std::map<rocfft_precision, const char*> PrecisionToStr = {{rocfft_precision_single, "single"},
                                                              {rocfft_precision_double, "double"},
                                                              {rocfft_precision_half, "half"}};
    return PrecisionToStr;
}

static std::map<std::string, rocfft_precision> StrToPrecisionMap()
{
    std::map<std::string, rocfft_precision> StrToPrecision;
    for(auto i : PrecisionToStrMap())
        StrToPrecision.emplace(i.second, i.first);
    return StrToPrecision;
}

//
// sbrc trans map
//
static std::map<SBRC_TRANSPOSE_TYPE, const char*> SBRCTransTypetoStrMap()
{
    std::map<SBRC_TRANSPOSE_TYPE, const char*> SBRCTransTypeToStr = {
        {ENUMSTR(NONE)}, {ENUMSTR(DIAGONAL)}, {ENUMSTR(TILE_ALIGNED)}, {ENUMSTR(TILE_UNALIGNED)}};
    return SBRCTransTypeToStr;
}

static std::map<std::string, SBRC_TRANSPOSE_TYPE> StrToSBRCTransTypeMap()
{
    std::map<std::string, SBRC_TRANSPOSE_TYPE> StrToSBRCTransType;
    for(auto i : SBRCTransTypetoStrMap())
        StrToSBRCTransType.emplace(i.second, i.first);
    return StrToSBRCTransType;
}

//
// ebtype map
//
static std::map<EmbeddedType, const char*> EBTypeToStrMap()
{
    std::map<EmbeddedType, const char*> EBTypeToStr = {{EmbeddedType::NONE, "NONE"},
                                                       {EmbeddedType::Real2C_POST, "R2C_POST"},
                                                       {EmbeddedType::C2Real_PRE, "C2R_PRE"}};
    return EBTypeToStr;
}

static std::map<std::string, EmbeddedType> StrToEBTypeMap()
{
    std::map<std::string, EmbeddedType> StrToEBType;
    for(auto i : EBTypeToStrMap())
        StrToEBType.emplace(i.second, i.first);
    return StrToEBType;
}

//
// placement map
//
static std::map<rocfft_result_placement, const char*> PlacementToStrMap()
{
    std::map<rocfft_result_placement, const char*> PlacementToStr
        = {{rocfft_placement_inplace, "IP"}, {rocfft_placement_notinplace, "OP"}};
    return PlacementToStr;
}

static std::map<std::string, rocfft_result_placement> StrToPlacementMap()
{
    std::map<std::string, rocfft_result_placement> StrToPlacement;
    for(auto i : PlacementToStrMap())
        StrToPlacement.emplace(i.second, i.first);
    return StrToPlacement;
}

//
// (internal) placement code map
//
static std::map<PlacementCode, const char*> PlacementCodeToStrMap()
{
    std::map<PlacementCode, const char*> PlacementCodeToStr
        = {{PC_IP, "IP"}, {PC_OP, "OP"}, {PC_UNSET, "NA"}};
    return PlacementCodeToStr;
}

static std::map<std::string, PlacementCode> StrToPlacementCodeMap()
{
    std::map<std::string, PlacementCode> StrToPlacementCode;
    for(auto i : PlacementCodeToStrMap())
        StrToPlacementCode.emplace(i.second, i.first);
    return StrToPlacementCode;
}

//
// array type map
//
static std::map<rocfft_array_type, const char*> ArrayTypeToStrMap()
{
    std::map<rocfft_array_type, const char*> ArrayTypeToStr
        = {{rocfft_array_type_complex_interleaved, "CI"},
           {rocfft_array_type_complex_planar, "CP"},
           {rocfft_array_type_real, "R"},
           {rocfft_array_type_hermitian_interleaved, "HI"},
           {rocfft_array_type_hermitian_planar, "HP"},
           {rocfft_array_type_unset, "NA"}};
    return ArrayTypeToStr;
}

static std::map<std::string, rocfft_array_type> StrToArrayTypeMap()
{
    std::map<std::string, rocfft_array_type> StrToArrayType;
    for(auto i : ArrayTypeToStrMap())
        StrToArrayType.emplace(i.second, i.first);
    return StrToArrayType;
}

std::string PrintOperatingBuffer(const OperatingBuffer ob)
{
    static const std::map<OperatingBuffer, const char*> BuffertoString
        = {{ENUMSTR(OB_UNINIT)},
           {ENUMSTR(OB_USER_IN)},
           {ENUMSTR(OB_USER_OUT)},
           {ENUMSTR(OB_TEMP)},
           {ENUMSTR(OB_TEMP_CMPLX_FOR_REAL)},
           {ENUMSTR(OB_TEMP_BLUESTEIN)}};
    return BuffertoString.at(ob);
}

std::string PrintOperatingBufferCode(const OperatingBuffer ob)
{
    static const std::map<OperatingBuffer, const char*> BuffertoString
        = {{OB_UNINIT, "ERR"},
           {OB_USER_IN, "A"},
           {OB_USER_OUT, "B"},
           {OB_TEMP, "T"},
           {OB_TEMP_CMPLX_FOR_REAL, "C"},
           {OB_TEMP_BLUESTEIN, "S"}};
    return BuffertoString.at(ob);
}

std::string PrintOptimizeStrategy(const rocfft_optimize_strategy ros)
{
    static const std::map<rocfft_optimize_strategy, const char*> StrategytoString
        = {{rocfft_optimize_min_buffer, "MINIMIZE_BUFFER"},
           {rocfft_optimize_balance, "BALANCE_BUFFER_FUSION"},
           {rocfft_optimize_max_fusion, "MAXIMIZE_FUSION"}};
    return StrategytoString.at(ros);
}

std::string PrintSBRCTransposeType(const SBRC_TRANSPOSE_TYPE ty)
{
    static auto sbrc2strMap = SBRCTransTypetoStrMap();
    return sbrc2strMap.at(ty);
}

std::string PrintDirectToFromRegMode(const DirectRegType ty)
{
    static const std::map<DirectRegType, const char*> TypetoString
        = {{ENUMSTR(FORCE_OFF_OR_NOT_SUPPORT)}, {ENUMSTR(TRY_ENABLE_IF_SUPPORT)}};
    return TypetoString.at(ty);
}

std::string PrintPrecision(const rocfft_precision pre)
{
    static auto precision2strMap = PrecisionToStrMap();
    return precision2strMap.at(pre);
}

std::string PrintArrayType(const rocfft_array_type aryType)
{
    static auto aryTypeStr = ArrayTypeToStrMap();
    return aryTypeStr.at(aryType);
}

std::string PrintPlacement(const rocfft_result_placement placement)
{
    static auto placementStr = PlacementToStrMap();
    return placementStr.at(placement);
}

std::string PrintPlacementCode(const PlacementCode placementCode)
{
    static auto strMap = PlacementCodeToStrMap();
    return strMap.at(placementCode);
}

std::string PrintEBType(const EmbeddedType ebtype)
{
    static auto ebtype2strMap = EBTypeToStrMap();
    return ebtype2strMap.at(ebtype);
}

//
// String to Enum
//

rocfft_array_type StrToArrayType(const std::string& str)
{
    static auto str2ArrayTypeMap = StrToArrayTypeMap();
    return str2ArrayTypeMap.at(str);
}

rocfft_result_placement StrToPlacement(const std::string& str)
{
    static auto str2PlacementMap = StrToPlacementMap();
    return str2PlacementMap.at(str);
}

PlacementCode StrToPlacementCode(const std::string& str)
{
    static auto str2PlacementCodeMap = StrToPlacementCodeMap();
    return str2PlacementCodeMap.at(str);
}

EmbeddedType StrToEBType(const std::string& str)
{
    static auto str2ebTypeMap = StrToEBTypeMap();
    return str2ebTypeMap.at(str);
}

SBRC_TRANSPOSE_TYPE StrToSBRCTransType(const std::string& str)
{
    static auto str2sbrcMap = StrToSBRCTransTypeMap();
    return str2sbrcMap.at(str);
}

rocfft_precision StrToPrecision(const std::string& str)
{
    static auto str2precisionMap = StrToPrecisionMap();
    return str2precisionMap.at(str);
}