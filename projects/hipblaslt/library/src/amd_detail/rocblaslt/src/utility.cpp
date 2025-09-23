/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "utility.hpp"
#include <sys/types.h>
#include <time.h>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <iostream>
#include <memory>
#include <string>

std::ostream* get_logger_os()
{
    LoggerSingleton& s = LoggerSingleton::getInstance();
    return s.log_os;
}

uint32_t get_logger_layer_mode()
{
    LoggerSingleton& s = LoggerSingleton::getInstance();
    return s.env_layer_mode;
}

std::string prefix(const char* layer, const char* caller)
{
    time_t now   = time(0);
    tm*    local = localtime(&now);

    std::string             format = "[%d-%02d-%02d %02d:%02d:%02d][HIPBLASLT][%lu][%s][%s]\0";
    std::unique_ptr<char[]> buf(new char[255]);
    std::sprintf(buf.get(),
                 format.c_str(),
                 1900 + local->tm_year,
                 1 + local->tm_mon,
                 local->tm_mday,
                 local->tm_hour,
                 local->tm_min,
                 local->tm_sec,
                 getpid(),
                 layer,
                 caller);
    return std::string(buf.get());
}

const char* hipDataType_to_string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return "R_16F";
    case HIP_R_16BF:
        return "R_16BF";
    case HIP_R_32F:
        return "R_32F";
    case HIP_R_64F:
        return "R_64F";
    case HIP_R_8F_E4M3_FNUZ:
        return "R_8F_E4M3_FNUZ";
    case HIP_R_8F_E5M2_FNUZ:
        return "R_8F_E5M2_FNUZ";
    case HIP_R_8F_E4M3:
        return "R_8F_E4M3";
    case HIP_R_8F_E5M2:
        return "R_8F_E5M2";
    case HIP_R_8I:
        return "R_8I";
    case static_cast<hipDataType>(HIP_R_6F_E2M3_EXT):
        return "R_6F_E2M3";
    case static_cast<hipDataType>(HIP_R_6F_E3M2_EXT):
        return "R_6F_E3M2";
    case static_cast<hipDataType>(HIP_R_4F_E2M1_EXT):
        return "R_4F_E2M1";
    default:
        return "Invalid";
    }
}

bool rocblaslt_is_complex_datatype(hipDataType type)
{
    return type == HIP_C_32F || type == HIP_C_64F || type == HIP_C_16F || type == HIP_C_8I
           || type == HIP_C_8U || type == HIP_C_32I || type == HIP_C_32U || type == HIP_C_16BF
           || type == HIP_C_4I || type == HIP_C_4U || type == HIP_C_16I || type == HIP_C_16U
           || type == HIP_C_64I || type == HIP_C_64U;
}

const char* hipDataType_to_bench_string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return "f32_r";
    case HIP_R_64F:
        return "f64_r";
    case HIP_R_16F:
        return "f16_r";
    case HIP_R_16BF:
        return "bf16_r";
    case HIP_R_8I:
        return "i8_r";
    case HIP_R_32I:
        return "i32_r";
    case HIP_R_8F_E4M3_FNUZ:
        return "f8_r";
    case HIP_R_8F_E5M2_FNUZ:
        return "bf8_r";
    case HIP_R_8F_E4M3:
        return "f8_r";
    case HIP_R_8F_E5M2:
        return "bf8_r";
    case static_cast<hipDataType>(HIP_R_6F_E2M3_EXT):
        return "f6_r";
    case static_cast<hipDataType>(HIP_R_6F_E3M2_EXT):
        return "bf6_r";
    case static_cast<hipDataType>(HIP_R_4F_E2M1_EXT):
        return "f4_r";
    default:
        return "invalid";
    }
}

const char* rocblaslt_compute_type_to_string(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f16:
        return "COMPUTE_16F";
    case rocblaslt_compute_f32:
        return "COMPUTE_32F";
    case rocblaslt_compute_f32_fast_xf32:
        return "COMPUTE_32XF";
    case rocblaslt_compute_f64:
        return "COMPUTE_64F";
    case rocblaslt_compute_i32:
        return "COMPUTE_32I";
    case rocblaslt_compute_f32_fast_f16:
        return "COMPUTE_32F_16F";
    case rocblaslt_compute_f32_fast_bf16:
        return "COMPUTE_32F_16BF";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_matrix_layout_attributes_to_string(rocblaslt_matrix_layout_attribute_ type)
{
    switch(type)
    {
    case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
        return "MATRIX_LAYOUT_BATCH_COUNT";
    case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
        return "MATRIX_LAYOUT_STRIDED_BATCH_OFFSET";
    case ROCBLASLT_MATRIX_LAYOUT_TYPE:
        return "ROCBLASLT_MATRIX_LAYOUT_TYPE";
    case ROCBLASLT_MATRIX_LAYOUT_ORDER:
        return "ROCBLASLT_MATRIX_LAYOUT_ORDER";
    case ROCBLASLT_MATRIX_LAYOUT_ROWS:
        return "ROCBLASLT_MATRIX_LAYOUT_ROWS";
    case ROCBLASLT_MATRIX_LAYOUT_COLS:
        return "ROCBLASLT_MATRIX_LAYOUT_COLS";
    case ROCBLASLT_MATRIX_LAYOUT_LD:
        return "ROCBLASLT_MATRIX_LAYOUT_LD";
    case ROCBLASLT_MATRIX_LAYOUT_MAX:
        return "ROCBLASLT_MATRIX_LAYOUT_MAX";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_matmul_desc_attributes_to_string(rocblaslt_matmul_desc_attributes type)
{
    switch(type)
    {
    case ROCBLASLT_MATMUL_DESC_TRANSA:
        return "MATMUL_DESC_TRANSA";
    case ROCBLASLT_MATMUL_DESC_TRANSB:
        return "MATMUL_DESC_TRANSB";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE:
        return "MATMUL_DESC_EPILOGUE";
    case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
        return "MATMUL_DESC_BIAS_POINTER";
    case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
        return "MATMUL_DESC_BIAS_DATA_TYPE";
    case ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER:
        return "MATMUL_DESC_A_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER:
        return "MATMUL_DESC_B_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_C_SCALE_POINTER:
        return "MATMUL_DESC_C_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER:
        return "MATMUL_DESC_D_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
        return "MATMUL_DESC_EPILOGUE_AUX_POINTER";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
        return "MATMUL_DESC_EPILOGUE_AUX_LD";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE:
        return "MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE";
    case ROCBLASLT_MATMUL_DESC_POINTER_MODE:
        return "MATMUL_DESC_POINTER_MODE";
    case ROCBLASLT_MATMUL_DESC_AMAX_D_POINTER:
        return "MATMUL_DESC_AMAX_D_POINTER";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE:
        return "MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE";
    case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT:
        return "MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT";
    case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT:
        return "MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT:
        return "MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT:
        return "MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT";
    default:
        return "Invalid";
    }
}

const char* hipblasOperation_to_string(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return "OP_N";
    case HIPBLAS_OP_T:
        return "OP_T";
    case HIPBLAS_OP_C:
        return "OP_C";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_layer_mode2string(rocblaslt_layer_mode layer_mode)
{
    switch(layer_mode)
    {
    case rocblaslt_layer_mode_none:
        return "None";
    case rocblaslt_layer_mode_log_error:
        return "Error";
    case rocblaslt_layer_mode_log_trace:
        return "Trace";
    case rocblaslt_layer_mode_log_hints:
        return "Hints";
    case rocblaslt_layer_mode_log_info:
        return "Info";
    case rocblaslt_layer_mode_log_api:
        return "Api";
    case rocblaslt_layer_mode_log_bench:
        return "Bench";
    case rocblaslt_layer_mode_log_profile:
        return "Profile";
    case rocblaslt_layer_mode_log_extended_profile:
        return "ExtendedProfile";
    default:
        return "Invalid";
    }
}

const char* rocblaslt_epilogue_to_string(rocblaslt_epilogue epilogue)
{
    switch(epilogue)
    {
    case ROCBLASLT_EPILOGUE_DEFAULT:
        return "EPILOGUE_DEFAULT";
    case ROCBLASLT_EPILOGUE_RELU:
        return "EPILOGUE_RELU";
    case ROCBLASLT_EPILOGUE_BIAS:
        return "EPILOGUE_BIAS";
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
        return "EPILOGUE_RELU_BIAS";
    case ROCBLASLT_EPILOGUE_RELU_AUX:
        return "ROCBLASLT_EPILOGUE_RELU_AUX";
    case ROCBLASLT_EPILOGUE_RELU_AUX_BIAS:
        return "ROCBLASLT_EPILOGUE_RELU_AUX_BIAS";
    case ROCBLASLT_EPILOGUE_GELU:
        return "EPILOGUE_GELU";
    case ROCBLASLT_EPILOGUE_DGELU:
        return "EPILOGUE_DGELU";
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
        return "EPILOGUE_GELU_BIAS";
    case ROCBLASLT_EPILOGUE_GELU_AUX:
        return "EPILOGUE_GELU_AUX";
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
        return "EPILOGUE_GELU_AUX_BIAS";
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
        return "EPILOGUE_DGELU_BGRAD";
    case ROCBLASLT_EPILOGUE_BGRADA:
        return "EPILOGUE_DGELU_BGRADA";
    case ROCBLASLT_EPILOGUE_BGRADB:
        return "EPILOGUE_DGELU_BGRADB";
    case ROCBLASLT_EPILOGUE_SWISH_EXT:
        return "EPILOGUE_SWISH_EXT";
    case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
        return "EPILOGUE_SWISH_BIAS_EXT";
    case ROCBLASLT_EPILOGUE_CLAMP_EXT:
        return "EPILOGUE_CLAMP_EXT";
    case ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
        return "EPILOGUE_CLAMP_BIAS_EXT";
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT:
        return "ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT";
    case ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT:
        return "ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT";
    default:
        return "Invalid epilogue";
    }
}

std::string rocblaslt_matrix_layout_to_string(rocblaslt_matrix_layout mat)
{
    std::string             format = mat->batch_count <= 1
                                         ? "[type=%s rows=%d cols=%d ld=%d]\0"
                                         : "[type=%s rows=%d cols=%d ld=%d batch_count=%d batch_stride=%d]\0";
    std::unique_ptr<char[]> buf(new char[255]);
    if(mat->batch_count <= 1)
        std::sprintf(
            buf.get(), format.c_str(), hipDataType_to_string(mat->type), mat->m, mat->n, mat->ld);
    else
        std::sprintf(buf.get(),
                     format.c_str(),
                     hipDataType_to_string(mat->type),
                     mat->m,
                     mat->n,
                     mat->ld,
                     mat->batch_count,
                     mat->batch_stride);
    return std::string(buf.get());
}
std::string rocblaslt_matmul_desc_to_string(rocblaslt_matmul_desc matmul_desc)
{
    std::stringstream ss;
    ss << "[computeType=%s scaleType=%s transA=%s transB=%s epilogue=%s biasPointer=0x%x";
    if(is_e_enabled(matmul_desc->epilogue))
    {
        ss << " epilogueAuxPointer=0x%x epilogueAuxLd=" << matmul_desc->lde;
        if(matmul_desc->aux_type != HIPBLASLT_DATATYPE_INVALID)
            ss << " epilogueAuxDataType=" << hipDataType_to_string(matmul_desc->aux_type);
    }
    if(matmul_desc->bias_type != HIPBLASLT_DATATYPE_INVALID)
        ss << " biasType=%s";
    ss << "]";
    std::string format = ss.str();

    std::unique_ptr<char[]> buf(new char[255]);

    if(matmul_desc->bias_type == HIPBLASLT_DATATYPE_INVALID)
        if(is_e_enabled(matmul_desc->epilogue))
            std::sprintf(buf.get(),
                         format.c_str(),
                         rocblaslt_compute_type_to_string(matmul_desc->compute_type),
                         hipDataType_to_string(matmul_desc->scale_type),
                         hipblasOperation_to_string(matmul_desc->op_A),
                         hipblasOperation_to_string(matmul_desc->op_B),
                         rocblaslt_epilogue_to_string(matmul_desc->epilogue),
                         matmul_desc->bias,
                         matmul_desc->e);
        else
            std::sprintf(buf.get(),
                         format.c_str(),
                         rocblaslt_compute_type_to_string(matmul_desc->compute_type),
                         hipDataType_to_string(matmul_desc->scale_type),
                         hipblasOperation_to_string(matmul_desc->op_A),
                         hipblasOperation_to_string(matmul_desc->op_B),
                         rocblaslt_epilogue_to_string(matmul_desc->epilogue),
                         matmul_desc->bias);
    else if(is_e_enabled(matmul_desc->epilogue))
        std::sprintf(buf.get(),
                     format.c_str(),
                     rocblaslt_compute_type_to_string(matmul_desc->compute_type),
                     hipDataType_to_string(matmul_desc->scale_type),
                     hipblasOperation_to_string(matmul_desc->op_A),
                     hipblasOperation_to_string(matmul_desc->op_B),
                     rocblaslt_epilogue_to_string(matmul_desc->epilogue),
                     matmul_desc->bias,
                     matmul_desc->e,
                     hipDataType_to_string(matmul_desc->bias_type));
    else
        std::sprintf(buf.get(),
                     format.c_str(),
                     rocblaslt_compute_type_to_string(matmul_desc->compute_type),
                     hipDataType_to_string(matmul_desc->scale_type),
                     hipblasOperation_to_string(matmul_desc->op_A),
                     hipblasOperation_to_string(matmul_desc->op_B),
                     rocblaslt_epilogue_to_string(matmul_desc->epilogue),
                     matmul_desc->bias,
                     hipDataType_to_string(matmul_desc->bias_type));
    return std::string(buf.get());
}

// Define and initialize static member flush and rotatingBufferSize outside the class UserClientArguments
bool    UserClientArguments::m_flush              = false;
int32_t UserClientArguments::m_rotatingBufferSize = 0;
int32_t UserClientArguments::m_coldIterations     = 0;
int32_t UserClientArguments::m_hotIterations      = 0;

// Define and initialize static members of struct hipblasltClientPerformanceArgs
double hipblasltClientPerformanceArgs::totalGranularity = 0.0;
double hipblasltClientPerformanceArgs::tilesPerCu       = 0.0;
double hipblasltClientPerformanceArgs::tile0Granularity = 0.0; // loss due to tile0
double hipblasltClientPerformanceArgs::tile1Granularity = 0.0;
double hipblasltClientPerformanceArgs::cuGranularity    = 0.0;
double hipblasltClientPerformanceArgs::waveGranularity  = 0.0;
int    hipblasltClientPerformanceArgs::CUs              = 0;
size_t hipblasltClientPerformanceArgs::memWriteBytesD   = 0.0; //! Estimated memory writes D
size_t hipblasltClientPerformanceArgs::memReadBytes     = 0.0;
