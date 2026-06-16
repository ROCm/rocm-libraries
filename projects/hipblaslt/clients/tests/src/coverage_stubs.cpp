/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
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

// Stub definitions for internal library symbols needed by coverage tests

#include <string>
#include <iostream>
#include <ctime>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include "rocblaslt.h"
#include "handle.h"
#include "rocblaslt-types.h"

// Forward declare the struct
struct hipblasltClientPerformanceArgs
{
    static double totalGranularity;
    static double tilesPerCu;
    static double tile0Granularity;
    static double tile1Granularity;
    static double cuGranularity;
    static double waveGranularity;
    static int    CUs;
    static size_t memWriteBytesD;
    static size_t memReadBytes;
};

// Define static members of hipblasltClientPerformanceArgs
double hipblasltClientPerformanceArgs::totalGranularity = 0.0;
double hipblasltClientPerformanceArgs::tilesPerCu = 0.0;
double hipblasltClientPerformanceArgs::tile0Granularity = 0.0;
double hipblasltClientPerformanceArgs::tile1Granularity = 0.0;
double hipblasltClientPerformanceArgs::cuGranularity = 0.0;
double hipblasltClientPerformanceArgs::waveGranularity = 0.0;
int    hipblasltClientPerformanceArgs::CUs = 0;
size_t hipblasltClientPerformanceArgs::memWriteBytesD = 0;
size_t hipblasltClientPerformanceArgs::memReadBytes = 0;

// Stub implementation of prefix function for coverage tests
std::string prefix(const char* layer, const char* caller)
{
    // Get current time
    time_t now = time(nullptr);
    struct tm* local = localtime(&now);
    char time_buf[64];
    snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d",
             local->tm_year + 1900, local->tm_mon + 1, local->tm_mday,
             local->tm_hour, local->tm_min, local->tm_sec);

    // Get process ID
    pid_t pid = getpid();

    // Format: [datetime][HIPBLASLT][pid][layer][caller]
    std::string result = "[";
    result += time_buf;
    result += "][HIPBLASLT][";
    result += std::to_string(pid);
    result += "][";
    result += (layer ? layer : "");
    result += "][";
    result += (caller ? caller : "");
    result += "]";

    return result;
}

// Stub implementations of utility functions
const char* hipDataType_to_string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F: return "R_32F";
    case HIP_R_64F: return "R_64F";
    case HIP_R_16F: return "R_16F";
    case HIP_R_16BF: return "R_16BF";
    case HIP_R_8I: return "R_8I";
    case HIP_R_32I: return "R_32I";
    case HIP_C_32F: return "C_32F";
    case HIP_C_64F: return "C_64F";
    case HIP_C_16F: return "C_16F";
    case HIP_C_16BF: return "C_16BF";
    case HIP_R_8F_E4M3_FNUZ: return "R_8F_E4M3_FNUZ";
    case HIP_R_8F_E5M2_FNUZ: return "R_8F_E5M2_FNUZ";
    case HIP_R_8F_E4M3: return "R_8F_E4M3";
    case HIP_R_8F_E5M2: return "R_8F_E5M2";
    // Float6 and Float4 types
    case static_cast<hipDataType>(31): return "R_6F_E2M3"; // HIP_R_6F_E2M3_EXT
    case static_cast<hipDataType>(32): return "R_6F_E3M2"; // HIP_R_6F_E3M2_EXT
    case static_cast<hipDataType>(33): return "R_4F_E2M1"; // HIP_R_4F_E2M1_EXT
    default: return "Invalid";
    }
}

const char* hipDataType_to_bench_string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F: return "f32_r";
    case HIP_R_64F: return "f64_r";
    case HIP_R_16F: return "f16_r";
    case HIP_R_16BF: return "bf16_r";
    case HIP_R_8I: return "i8_r";
    case HIP_R_32I: return "i32_r";
    case HIP_C_32F: return "c32_r";
    case HIP_C_64F: return "c64_r";
    case HIP_C_16F: return "c16_r";
    case HIP_C_16BF: return "c16bf_r";
    case HIP_R_8F_E4M3_FNUZ: return "f8_r";
    case HIP_R_8F_E5M2_FNUZ: return "bf8_r";
    case HIP_R_8F_E4M3: return "f8_r";
    case HIP_R_8F_E5M2: return "bf8_r";
    // Float6 and Float4 types
    case static_cast<hipDataType>(31): return "f6_r";   // HIP_R_6F_E2M3_EXT
    case static_cast<hipDataType>(32): return "bf6_r";  // HIP_R_6F_E3M2_EXT
    case static_cast<hipDataType>(33): return "f4_r";   // HIP_R_4F_E2M1_EXT
    default: return "invalid";
    }
}

const char* rocblaslt_compute_type_to_string(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f16: return "COMPUTE_16F";
    case rocblaslt_compute_f16_pedantic: return "COMPUTE_16F_PEDANTIC";
    case rocblaslt_compute_f32: return "COMPUTE_32F";
    case rocblaslt_compute_f32_pedantic: return "COMPUTE_32F_PEDANTIC";
    case rocblaslt_compute_f32_fast_xf32: return "COMPUTE_32XF";
    case rocblaslt_compute_f64: return "COMPUTE_64F";
    case rocblaslt_compute_f64_pedantic: return "COMPUTE_64F_PEDANTIC";
    case rocblaslt_compute_i32: return "COMPUTE_32I";
    case rocblaslt_compute_i32_pedantic: return "COMPUTE_32I_PEDANTIC";
    case rocblaslt_compute_f32_fast_f16: return "COMPUTE_32F_16F";
    case rocblaslt_compute_f32_fast_bf16: return "COMPUTE_32F_16BF";
    case rocblaslt_compute_f32_fast_f8_fnuz: return "COMPUTE_32F_8F_FNUZ";
    default: return "Invalid";
    }
}

const char* rocblaslt_matrix_layout_attributes_to_string(rocblaslt_matrix_layout_attribute attr)
{
    switch(attr)
    {
    case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT: return "MATRIX_LAYOUT_BATCH_COUNT";
    case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: return "MATRIX_LAYOUT_STRIDED_BATCH_OFFSET";
    case ROCBLASLT_MATRIX_LAYOUT_COLS: return "ROCBLASLT_MATRIX_LAYOUT_COLS";
    case ROCBLASLT_MATRIX_LAYOUT_ROWS: return "ROCBLASLT_MATRIX_LAYOUT_ROWS";
    case ROCBLASLT_MATRIX_LAYOUT_LD: return "ROCBLASLT_MATRIX_LAYOUT_LD";
    case ROCBLASLT_MATRIX_LAYOUT_TYPE: return "ROCBLASLT_MATRIX_LAYOUT_TYPE";
    case ROCBLASLT_MATRIX_LAYOUT_ORDER: return "ROCBLASLT_MATRIX_LAYOUT_ORDER";
    case ROCBLASLT_MATRIX_LAYOUT_MAX: return "ROCBLASLT_MATRIX_LAYOUT_MAX";
    default: return "Invalid";
    }
}

const char* rocblaslt_matmul_desc_attributes_to_string(rocblaslt_matmul_desc_attributes attr)
{
    switch(attr)
    {
    case ROCBLASLT_MATMUL_DESC_TRANSA: return "MATMUL_DESC_TRANSA";
    case ROCBLASLT_MATMUL_DESC_TRANSB: return "MATMUL_DESC_TRANSB";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE: return "MATMUL_DESC_EPILOGUE";
    case ROCBLASLT_MATMUL_DESC_BIAS_POINTER: return "MATMUL_DESC_BIAS_POINTER";
    case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: return "MATMUL_DESC_BIAS_DATA_TYPE";
    case ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER: return "MATMUL_DESC_A_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER: return "MATMUL_DESC_B_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_C_SCALE_POINTER: return "MATMUL_DESC_C_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER: return "MATMUL_DESC_D_SCALE_POINTER";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER: return "MATMUL_DESC_EPILOGUE_AUX_POINTER";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD: return "MATMUL_DESC_EPILOGUE_AUX_LD";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE: return "MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE";
    case ROCBLASLT_MATMUL_DESC_POINTER_MODE: return "MATMUL_DESC_POINTER_MODE";
    case ROCBLASLT_MATMUL_DESC_AMAX_D_POINTER: return "MATMUL_DESC_AMAX_D_POINTER";
    case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE: return "MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE";
    case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT: return "MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT";
    case ROCBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT: return "MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT";
    default: return "Invalid";
    }
}

const char* hipblasOperation_to_string(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N: return "OP_N";
    case HIPBLAS_OP_T: return "OP_T";
    case HIPBLAS_OP_C: return "OP_C";
    default: return "Invalid";
    }
}

const char* rocblaslt_layer_mode2string(rocblaslt_layer_mode mode)
{
    switch(mode)
    {
    case rocblaslt_layer_mode_none: return "None";
    case rocblaslt_layer_mode_log_error: return "Error";
    case rocblaslt_layer_mode_log_trace: return "Trace";
    case rocblaslt_layer_mode_log_hints: return "Hints";
    case rocblaslt_layer_mode_log_info: return "Info";
    case rocblaslt_layer_mode_log_api: return "Api";
    case rocblaslt_layer_mode_log_bench: return "Bench";
    case rocblaslt_layer_mode_log_profile: return "Profile";
    case rocblaslt_layer_mode_log_extended_profile: return "ExtendedProfile";
    default: return "Invalid";
    }
}

const char* rocblaslt_epilogue_to_string(rocblaslt_epilogue epilogue)
{
    switch(epilogue)
    {
    case ROCBLASLT_EPILOGUE_DEFAULT: return "EPILOGUE_DEFAULT";
    case ROCBLASLT_EPILOGUE_RELU: return "EPILOGUE_RELU";
    case ROCBLASLT_EPILOGUE_BIAS: return "EPILOGUE_BIAS";
    case ROCBLASLT_EPILOGUE_RELU_BIAS: return "EPILOGUE_RELU_BIAS";
    case ROCBLASLT_EPILOGUE_GELU: return "EPILOGUE_GELU";
    case ROCBLASLT_EPILOGUE_DGELU: return "EPILOGUE_DGELU";
    case ROCBLASLT_EPILOGUE_DRELU: return "EPILOGUE_DRELU";
    case ROCBLASLT_EPILOGUE_GELU_BIAS: return "EPILOGUE_GELU_BIAS";
    case ROCBLASLT_EPILOGUE_GELU_AUX: return "EPILOGUE_GELU_AUX";
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS: return "EPILOGUE_GELU_AUX_BIAS";
    case ROCBLASLT_EPILOGUE_SIGMOID: return "EPILOGUE_SIGMOID";
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD: return "EPILOGUE_DGELU_BGRAD";
    case ROCBLASLT_EPILOGUE_BGRADA: return "EPILOGUE_BGRADA";
    case ROCBLASLT_EPILOGUE_BGRADB: return "EPILOGUE_BGRADB";
    case ROCBLASLT_EPILOGUE_DRELU_BGRAD: return "EPILOGUE_DRELU_BGRAD";
    case ROCBLASLT_EPILOGUE_SWISH_EXT: return "EPILOGUE_SWISH_EXT";
    case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT: return "EPILOGUE_SWISH_BIAS_EXT";
    default: return "Invalid epilogue";
    }
}

const char* rocblaslt_compute_type_string(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f16: return "f16_r";
    case rocblaslt_compute_f32: return "f32_r";
    case rocblaslt_compute_f32_fast_xf32: return "xf32_r";
    case rocblaslt_compute_i32: return "i32_r";
    case rocblaslt_compute_f64: return "f64_r";
    case rocblaslt_compute_f32_fast_f16: return "f32_f16_r";
    case rocblaslt_compute_f32_fast_bf16: return "f32_bf16_r";
    case rocblaslt_compute_f32_fast_f8: return "f32_f8_r";
    case rocblaslt_compute_f32_fast_f8_fnuz: return "f32_f8_fnuz_r";
    case rocblaslt_compute_f32_fast_bf8: return "f32_bf8_fnuz_r";
    case rocblaslt_compute_f32_fast_bf8_fnuz: return "f32_bf8_r";
    case rocblaslt_compute_f32_fast_f8bf8: return "f32_f8bf8_r";
    case rocblaslt_compute_f32_fast_f8bf8_fnuz: return "f32_f8bf8_fnuz_r";
    case rocblaslt_compute_f32_fast_bf8f8: return "f32_bf8f8_r";
    case rocblaslt_compute_f32_fast_bf8f8_fnuz: return "f32_bf8f8_fnuz_r";
    default: return "invalidType";
    }
}

std::string rocblaslt_matrix_layout_to_string(_rocblaslt_matrix_layout* mat)
{
    if (!mat) return "null";
    std::string result = "[type=" + std::string(hipDataType_to_string(mat->type));
    result += " order=";
    result += (mat->order == HIPBLASLT_ORDER_COL) ? "col" : "row";
    result += " rows=" + std::to_string(mat->m);
    result += " cols=" + std::to_string(mat->n);
    result += " ld=" + std::to_string(mat->ld);
    if (mat->batch_count > 1) {
        result += " batch_count=" + std::to_string(mat->batch_count);
        result += " batch_stride=" + std::to_string(mat->batch_stride);
    }
    result += "]";
    return result;
}

std::string rocblaslt_matmul_desc_to_string(_rocblaslt_matmul_desc* desc)
{
    if (!desc) return "null";
    std::string result = "[computeType=" + std::string(rocblaslt_compute_type_to_string(desc->compute_type));
    result += " scaleType=" + std::string(hipDataType_to_string(desc->scale_type));
    result += " transA=" + std::string(hipblasOperation_to_string(desc->op_A));
    result += " transB=" + std::string(hipblasOperation_to_string(desc->op_B));
    result += " epilogue=" + std::string(rocblaslt_epilogue_to_string(desc->epilogue));

    // Always include biasPointer, even if null (show as 0x0)
    result += " biasPointer=0x";
    char buf[32];
    snprintf(buf, sizeof(buf), "%lx", (uintptr_t)desc->bias);
    result += buf;

    if (desc->bias_type != HIPBLASLT_DATATYPE_INVALID)
        result += " biasType=" + std::string(hipDataType_to_string(desc->bias_type));
    if (desc->aux_type != HIPBLASLT_DATATYPE_INVALID)
        result += " epilogueAuxDataType=" + std::string(hipDataType_to_string(desc->aux_type));
    if (desc->e != nullptr) {
        result += " epilogueAuxPointer=0x";
        snprintf(buf, sizeof(buf), "%lx", (uintptr_t)desc->e);
        result += buf;
    }
    if (desc->lde != 0)
        result += " epilogueAuxLd=" + std::to_string(desc->lde);
    result += "]";
    return result;
}

// Status conversion stubs
rocblaslt_status get_rocblaslt_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    case hipSuccess: return rocblaslt_status_success;
    case hipErrorMemoryAllocation: return rocblaslt_status_memory_error;
    case hipErrorLaunchOutOfResources: return rocblaslt_status_memory_error;
    case hipErrorInvalidDevicePointer: return rocblaslt_status_invalid_pointer;
    case hipErrorInvalidDevice: return rocblaslt_status_invalid_handle;
    case hipErrorInvalidResourceHandle: return rocblaslt_status_invalid_handle;
    case hipErrorInvalidValue: return rocblaslt_status_internal_error;
    case hipErrorNoDevice: return rocblaslt_status_internal_error;
    case hipErrorUnknown: return rocblaslt_status_internal_error;
    default: return rocblaslt_status_internal_error;
    }
}

hipblasStatus_t RocBlasLtStatusToHIPStatus(rocblaslt_status status)
{
    switch(status)
    {
    case rocblaslt_status_success: return HIPBLAS_STATUS_SUCCESS;
    case rocblaslt_status_invalid_handle: return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblaslt_status_not_implemented: return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_pointer: return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_invalid_size: return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_memory_error: return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblaslt_status_internal_error: return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_value: return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_arch_mismatch: return HIPBLAS_STATUS_ARCH_MISMATCH;
    default: throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

// Logger stubs
rocblaslt_layer_mode get_logger_layer_mode()
{
    return rocblaslt_layer_mode_none;
}

std::ostream* get_logger_os()
{
    static std::ostream* null_stream = nullptr;
    return null_stream;
}

// is_act_enabled stub
inline bool is_act_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_RELU:
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
    case ROCBLASLT_EPILOGUE_GELU:
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_AUX:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_DRELU:
    case ROCBLASLT_EPILOGUE_DRELU_BGRAD:
    case ROCBLASLT_EPILOGUE_SWISH_EXT:
    case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_EXT:
    case ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
    case ROCBLASLT_EPILOGUE_SIGMOID:
        return true;
    case ROCBLASLT_EPILOGUE_DEFAULT:
    case ROCBLASLT_EPILOGUE_BIAS:
    default:
        return false;
    }
}

// RocblasltContractionProblem constructor stub
RocblasltContractionProblem::RocblasltContractionProblem(
    hipblasOperation_t trans_a_,
    hipblasOperation_t trans_b_,
    int64_t m_,
    int64_t n_,
    int64_t k_,
    const void* alpha_,
    hipDataType a_type_,
    const void* A_,
    const void* const* batch_A_,
    int64_t ld_a,
    int64_t batch_stride_a_,
    hipDataType b_type_,
    const void* B_,
    const void* const* batch_B_,
    int64_t ld_b,
    int64_t batch_stride_b_,
    const void* beta_,
    hipDataType c_type_,
    const void* C_,
    const void* const* batch_C_,
    int64_t ld_c,
    int64_t batch_stride_c_,
    hipDataType d_type_,
    void* D_,
    void* const* batch_D_,
    int64_t ld_d,
    int64_t batch_stride_d_,
    void* E_,
    void* const* batch_E_,
    int64_t ld_e,
    int64_t batch_stride_e_,
    int64_t batch_count_,
    bool strided_batch_,
    bool grouped_gemm_,
    bool gradient_,
    rocblaslt_compute_type compute_type_,
    hipDataType scale_type_,
    const void* bias_,
    const void* scaleA_,
    const void* scaleB_,
    const void* scaleC_,
    const void* scaleD_,
    const void* scaleE_,
    const void* scaleAlphaVec_,
    RocblasltContractionProblem::ScalingFormat scaleAType_,
    RocblasltContractionProblem::ScalingFormat scaleBType_,
    hipDataType bias_type_,
    hipDataType aux_type_,
    rocblaslt_epilogue epilogue_,
    void* amaxD_,
    void* workspace_,
    size_t workspaceSize_,
    float act0_,
    float act1_,
    hipStream_t stream_,
    void* Synchronizer_,
    bool swizzleA_,
    bool swizzleB_,
    hipblasLtBatchMode_t batchMode_,
    int32_t bias_stride_)
{
    // Minimal stub - just initialize the basic fields
    trans_a = trans_a_;
    trans_b = trans_b_;
    m = m_;
    n = n_;
    k = k_;
    alpha = alpha_;
    a_type = a_type_;
    A = A_;
    batch_A = batch_A_;
    row_stride_a = (trans_a == HIPBLAS_OP_N) ? 1 : ld_a;
    col_stride_a = (trans_a == HIPBLAS_OP_N) ? ld_a : 1;
    batch_stride_a = batch_stride_a_;
    b_type = b_type_;
    B = B_;
    batch_B = batch_B_;
    row_stride_b = (trans_b == HIPBLAS_OP_N) ? 1 : ld_b;
    col_stride_b = (trans_b == HIPBLAS_OP_N) ? ld_b : 1;
    batch_stride_b = batch_stride_b_;
    beta = beta_;
    c_type = c_type_;
    C = C_;
    batch_C = batch_C_;
    row_stride_c = 1;
    col_stride_c = ld_c;
    batch_stride_c = batch_stride_c_;
    d_type = d_type_;
    D = D_;
    batch_D = batch_D_;
    row_stride_d = 1;
    col_stride_d = ld_d;
    batch_stride_d = batch_stride_d_;
    E = E_;
    batch_E = batch_E_;
    row_stride_e = 1;
    col_stride_e = ld_e;
    batch_stride_e = batch_stride_e_;
    batch_count = batch_count_;
    strided_batch = strided_batch_;
    grouped_gemm = grouped_gemm_;
    gradient = gradient_;
    compute_type = compute_type_;
    scale_type = scale_type_;
    bias = bias_;
    scaleA = scaleA_;
    scaleB = scaleB_;
    scaleC = scaleC_;
    scaleD = scaleD_;
    scaleE = scaleE_;
    scaleAlphaVec = scaleAlphaVec_;
    scaleAType = scaleAType_;
    scaleBType = scaleBType_;
    bias_type = bias_type_;
    aux_type = aux_type_;
    epilogue = epilogue_;
    amaxD = amaxD_;
    workspace = workspace_;
    workspaceSize = workspaceSize_;
    act0 = act0_;
    act1 = act1_;
    stream = stream_;
    Synchronizer = Synchronizer_;
    swizzleA = swizzleA_;
    swizzleB = swizzleB_;
    batchMode = batchMode_;
    bias_stride = bias_stride_;
}
