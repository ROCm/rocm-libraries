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

#pragma once

#include "benchmark_timing.hpp"
#include "efficiency_monitor.hpp"
#include "flops.hpp"
#include "hipBuffer.hpp"
#include "hipblaslt_bench_options.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include <hipblaslt/client/MatmulTestCase.hpp>
#include <hipblaslt/host_validation/Epilogue.hpp>
#include <hipblaslt/host_validation/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_validation/MatmulValidation.hpp>
#include <hipblaslt/host_validation/Reduction.hpp>
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
#include "mxDataGen.hpp"
#endif
#include "near.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <map>
#include <numeric>
#include <omp.h>
#include <optional>
#include <set>
#include <span>
#include <vector>

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

// Convert element count to byte count, accounting for sub-byte packing.
// FP4 (4-bit) packs 2 elements per byte; all other types use realDataTypeSize.
size_t elementsToBytes(size_t numElements, hipDataType dtype)
{
    if(static_cast<int>(dtype) == HIP_R_4F_E2M1)
        return numElements / 2;
    return numElements * realDataTypeSize(dtype);
}

bool isSwizzleSupported(hipDataType datatype)
{
    switch(datatype)
    {
    case HIP_R_16BF:
    case HIP_R_16F:
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_4F_E2M1:
        return true;
    default:
        return false;
    }
}

bool MXUseRocroller()
{
#ifdef HIPBLASLT_USE_ROCROLLER
    return hipblaslt_get_arch() != 1250;
#else
    return false;
#endif
}

hipblasLtOrder_t orderForDatatype(hipDataType datatype)
{
    switch(datatype)
    {
    case HIP_R_16F:
    case HIP_R_16BF:
        return HIPBLASLT_ORDER_COL16_4R8;
    case HIP_R_8F_E4M3_FNUZ:
        return HIPBLASLT_ORDER_COL16_4R16;
    case HIP_R_4F_E2M1:
        return HIPBLASLT_ORDER_COL16_4R32;
    default:
        throw std::runtime_error("unsupported datatype in orderForDatatype");
    }
}

hipblasLtMatmulMatrixScale_t matmulScaleMode(hipblaslt_scaling_format format)
{
    switch(format)
    {
    case hipblaslt_scaling_format::Vector:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    case hipblaslt_scaling_format::Block_32_UE8M0:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    case hipblaslt_scaling_format::Block_16_UE8M0:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0_EXT;
    case hipblaslt_scaling_format::Block_32_UE4M3:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE4M3_EXT;
    case hipblaslt_scaling_format::Block_16_UE4M3:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    case hipblaslt_scaling_format::Block_32_UE5M3:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE5M3_EXT;
    case hipblaslt_scaling_format::Block_16_UE5M3:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE5M3_EXT;
    case hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
    default:
        return HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    }
}

hipblasLtEpilogue_t matmulEpilogue(const Arguments& arg)
{
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    switch(arg.activation_type)
    {
    case hipblaslt_activation_type::relu:
        epilogue = arg.bias_vector ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
        break;
    case hipblaslt_activation_type::gelu:
        epilogue = arg.bias_vector ? HIPBLASLT_EPILOGUE_GELU_BIAS : HIPBLASLT_EPILOGUE_GELU;
        break;
    case hipblaslt_activation_type::swish:
        epilogue
            = arg.bias_vector ? HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT : HIPBLASLT_EPILOGUE_SWISH_EXT;
        break;
    case hipblaslt_activation_type::clamp:
        epilogue
            = arg.bias_vector ? HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT : HIPBLASLT_EPILOGUE_CLAMP_EXT;
        break;
    default:
        if(arg.bias_vector)
            epilogue = HIPBLASLT_EPILOGUE_BIAS;
        break;
    }

    if(arg.gradient)
    {
        switch(epilogue)
        {
        case HIPBLASLT_EPILOGUE_BIAS:
            if(arg.bias_source == hipblaslt_bias_source::a)
                epilogue = HIPBLASLT_EPILOGUE_BGRADA;
            else if(arg.bias_source == hipblaslt_bias_source::b)
                epilogue = HIPBLASLT_EPILOGUE_BGRADB;
            break;
        case HIPBLASLT_EPILOGUE_GELU:
            epilogue = HIPBLASLT_EPILOGUE_DGELU;
            break;
        case HIPBLASLT_EPILOGUE_GELU_BIAS:
            epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
            break;
        case HIPBLASLT_EPILOGUE_RELU:
            epilogue = HIPBLASLT_EPILOGUE_DRELU;
            break;
        case HIPBLASLT_EPILOGUE_RELU_BIAS:
            epilogue = HIPBLASLT_EPILOGUE_DRELU_BGRAD;
            break;
        default:
            break;
        }
        if(epilogue == HIPBLASLT_EPILOGUE_DGELU || epilogue == HIPBLASLT_EPILOGUE_DGELU_BGRAD
           || epilogue == HIPBLASLT_EPILOGUE_DRELU
           || epilogue == HIPBLASLT_EPILOGUE_DRELU_BGRAD)
        {
            if(!arg.use_e)
                throw std::invalid_argument(
                    "Gradient ReLU/GELU matmul requires auxiliary E storage.");
        }
    }

    if(!arg.use_e)
        return epilogue;
    switch(epilogue)
    {
    case HIPBLASLT_EPILOGUE_RELU:
        return HIPBLASLT_EPILOGUE_RELU_AUX;
    case HIPBLASLT_EPILOGUE_RELU_BIAS:
        return HIPBLASLT_EPILOGUE_RELU_AUX_BIAS;
    case HIPBLASLT_EPILOGUE_GELU:
        return HIPBLASLT_EPILOGUE_GELU_AUX;
    case HIPBLASLT_EPILOGUE_GELU_BIAS:
        return HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
    case HIPBLASLT_EPILOGUE_CLAMP_EXT:
        return HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT;
    case HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
        return HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
    case HIPBLASLT_EPILOGUE_DGELU:
    case HIPBLASLT_EPILOGUE_DGELU_BGRAD:
    case HIPBLASLT_EPILOGUE_DRELU:
    case HIPBLASLT_EPILOGUE_DRELU_BGRAD:
        return epilogue;
    default:
        throw std::invalid_argument("Selected matmul epilogue does not support auxiliary E.");
    }
}

void calculateKforSwizzling(
    hipDataType datatype, const Arguments& arg, size_t& MiK, size_t& MiKv, size_t& PackK)
{
    switch(datatype)
    {
    case HIP_R_32F:
        if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
        {
            MiK  = 8;
            MiKv = 2;
        }
        else
        {
            MiK  = 4;
            MiKv = 1;
        }
        break;
    case HIP_R_64F:
        MiK  = 4;
        MiKv = 1;
        break;
    case HIP_R_16F:
    case HIP_R_16BF:
        MiK  = 16;
        MiKv = 4;
        break;
    case HIP_R_8I:
    case HIP_R_8F_E5M2_FNUZ:
    case HIP_R_8F_E4M3_FNUZ:
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E5M2:
        MiK  = 32;
        MiKv = 8;
        break;
    case HIP_R_4F_E2M1:
        // For fp4 viewed as uint8: matches shuffle_weight with layout=(16,16)
        // BK=32 bytes, K=16 bytes, BK/K=2
        MiK  = 16; // K inner block = 16 bytes
        MiKv = 8;
        break;
    default:
        throw std::runtime_error("unsupported datatype in calculateKforSwizzling");
    }

    PackK = 16 / MiKv / realDataTypeSize(datatype);
}

template <typename T>
void swizzle_tensor(T*               dst,
                    const T*         src,
                    hipDataType      datatype,
                    const Arguments& arg,
                    size_t           b,
                    size_t           m_n,
                    size_t           k,
                    size_t           ld,
                    bool             colMaj)
{
    if(ld < k)
        throw std::runtime_error("invalid value of ld in swizzle_tensor: ld must be >= k.");

    using roc::host_validation::Layout;
    using roc::host_validation::ScalarType;
    using roc::host_validation::Shape;
    using roc::host_validation::Tensor;

    // currently, if A then it means MiM = 16, if B then it means MiN = 16
    size_t MiM_N = 16;
    size_t MiK = 0, MiKv = 0, PackK = 0;
    calculateKforSwizzling(datatype, arg, MiK, MiKv, PackK);
    const size_t numElements = b * m_n * k;
    const ScalarType tensorType = datatype == HIP_R_4F_E2M1
                                      ? ScalarType::UInt8
                                      : hipblaslt::host_validation::scalarType(datatype);
    std::vector<T> compact(numElements);

    if(colMaj)
    {
        for(size_t i = 0; i < b * k; i++)
        {
            std::copy(src + (i * ld), src + (i * ld) + m_n, compact.data() + (i * m_n));
        }
    }
    else
    {
        for(size_t i = 0; i < b * m_n; i++)
        {
            std::copy(src + (i * ld), src + (i * ld) + k, compact.data() + (i * k));
        }
    }

    Tensor tmpTensor(
        tensorType,
        Layout::contiguous(Shape(colMaj ? std::vector<size_t>{b, k, m_n}
                                        : std::vector<size_t>{b, m_n, k})),
        std::as_bytes(std::span<const T>(compact)));
    if(colMaj)
        tmpTensor = tmpTensor.permute({0, 2, 1});

    auto       MultipleM_N = MiM_N;
    auto       MultipleK   = MiK * PackK;
    const auto paddedM_N   = (m_n / MultipleM_N + !!(m_n % MultipleM_N)) * MultipleM_N;
    const auto paddedK     = (k / MultipleK + !!(k % MultipleK)) * MultipleK;
    Tensor paddedTensor = tmpTensor.pad(Shape{b, paddedM_N, paddedK});
    Tensor reshaped     = paddedTensor.reshape(
        Shape{b, paddedM_N / MiM_N, MiM_N, paddedK / (MiK * PackK), MiK / MiKv, MiKv * PackK});
    Tensor permuted = reshaped.permute({0, 1, 3, 4, 2, 5});
    const size_t outputBytes = b * paddedM_N * paddedK * sizeof(T);
    if(permuted.storage().size() != outputBytes)
        throw std::runtime_error("swizzle_tensor produced an unexpected storage size.");
    std::memcpy(static_cast<void*>(dst), permuted.storage().data(), outputBytes);
}

void swizzle_tensor_type(HipHostBuffer&       dst,
                         const HipHostBuffer& src,
                         hipDataType          datatype,
                         const Arguments&     arg,
                         size_t               b,
                         size_t               m_n,
                         size_t               k,
                         size_t               ld,
                         bool                 colMaj)
{
    switch(datatype)
    {
    case HIP_R_32F:
        swizzle_tensor<float>(
            dst.as<float>(), src.as<float>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_16F:
        swizzle_tensor<hipblasLtHalf>(
            dst.as<hipblasLtHalf>(), src.as<hipblasLtHalf>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_16BF:
        swizzle_tensor<hip_bfloat16>(
            dst.as<hip_bfloat16>(), src.as<hip_bfloat16>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_8F_E4M3_FNUZ:
        swizzle_tensor<hipblaslt_f8_fnuz>(dst.as<hipblaslt_f8_fnuz>(),
                                          src.as<hipblaslt_f8_fnuz>(),
                                          datatype,
                                          arg,
                                          b,
                                          m_n,
                                          k,
                                          ld,
                                          colMaj);
        return;
    case HIP_R_8F_E5M2_FNUZ:
        swizzle_tensor<hipblaslt_bf8_fnuz>(dst.as<hipblaslt_bf8_fnuz>(),
                                           src.as<hipblaslt_bf8_fnuz>(),
                                           datatype,
                                           arg,
                                           b,
                                           m_n,
                                           k,
                                           ld,
                                           colMaj);
        return;
    case HIP_R_8F_E4M3:
        swizzle_tensor<hipblaslt_f8>(
            dst.as<hipblaslt_f8>(), src.as<hipblaslt_f8>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_8F_E5M2:
        swizzle_tensor<hipblaslt_bf8>(
            dst.as<hipblaslt_bf8>(), src.as<hipblaslt_bf8>(), datatype, arg, b, m_n, k, ld, colMaj);
        return;
    case HIP_R_4F_E2M1:
        // fp4: 2 elements per byte, so k_bytes = k/2, ld_bytes = ld/2
        swizzle_tensor<uint8_t>(
            dst.as<uint8_t>(), src.as<uint8_t>(), datatype, arg, b, m_n, k / 2, ld / 2, colMaj);
        return;
    default:
        hipblaslt_cerr << "Error type in swizzle_tensor_type()" << std::endl;
    }
}

inline void pre_gpu_time(bool         use_gpu_timer,
                         hipEvent_t&  event_gpu_time_start,
                         double&      gpu_time_used,
                         hipStream_t& stream)
{
    if(use_gpu_timer)
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
    else
        gpu_time_used = get_time_us_sync(stream);
}
inline void post_gpu_time(bool         use_gpu_timer,
                          hipEvent_t&  event_gpu_time_start,
                          hipEvent_t&  event_gpu_time_end,
                          double&      gpu_time_used,
                          hipStream_t& stream)
{
    if(use_gpu_timer)
    {
        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
        float gpu_time_ms;
        CHECK_HIP_ERROR(
            hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
        gpu_time_used = gpu_time_ms * 1000; // ms to us
    }
    else
    {
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
    }
}

void testing_matmul_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipblasOperation_t transA = HIPBLAS_OP_T;
    const hipblasOperation_t transB = HIPBLAS_OP_N;

    // allocate memory on device
    HipDeviceBuffer dA(arg.a_type, safe_size / 2, arg.HMM);
    HipDeviceBuffer dB(arg.b_type, safe_size, arg.HMM);
    HipDeviceBuffer dC(arg.c_type, safe_size, arg.HMM);
    HipDeviceBuffer dD(arg.d_type, safe_size, arg.HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout matA(M, K, lda, arg.a_type);
    hipblaslt_local_matrix_layout matB(K, N, ldb, arg.b_type);
    hipblaslt_local_matrix_layout matC(M, N, ldc, arg.c_type);
    hipblaslt_local_matrix_layout matD(M, N, ldc, arg.d_type);
    hipblaslt_local_matmul_descr  matmul(transA,
                                        transB,
                                        arg.compute_type,
                                        arg.scale_type,
                                        arg.compute_input_typeA,
                                        arg.compute_input_typeB);

    size_t                     workspace_size = 0;
    hipblaslt_local_preference pref;

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
}

void copy_gemm_to_host(hipStream_t                   stream,
                       const uint32_t&               problem_count,
                       std::vector<HipHostBuffer>&   hDst,
                       std::vector<HipDeviceBuffer>& dSrc)
{

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
    {
        CHECK_HIP_ERROR(synchronize(hDst[gemmIdx], dSrc[gemmIdx], 0, 0, 0, 0, 1, false, stream));
    }
}

template <typename T>
void dumpBuffer(const char* title, T* buf, size_t M, size_t N)
{
    hipblaslt_cout << "----- DUMP: " << title << " -----" << std::endl;
    for(int n = 0; n < N; n++)
    {
        for(int m = 0; m < M; m++)
        {
            hipblaslt_cout << buf[m + n * M] << " ";
        }
        hipblaslt_cout << std::endl;
    }
}

void dumpBuffer(const char* title, hipDataType To, HipHostBuffer& buf, size_t M, size_t N)
{
    switch(To)
    {
    case HIP_R_32F:
        dumpBuffer(title, buf.as<float>(), M, N);
        break;
    default:
        hipblaslt_cerr << "Unsupported dumpBuffer data type" << std::endl;
        break;
    }

    return;
}

// A function to determine the default bias_type
hipDataType derive_unset_bias_type(const Arguments& arg)
{
    // TODO: confirm if HIP_R_64F, HIP_R_32I are neccessary for biastype
    static const std::set<hipDataType> supported_bias_types
        = {HIP_R_32F, HIP_R_16F, HIP_R_16BF, HIP_R_64F, HIP_R_32I, HIP_C_32F, HIP_C_64F};

    hipDataType real_bias_type = arg.bias_type;

    // when bias type is unset
    if(arg.bias_type == HIPBLASLT_DATATYPE_INVALID)
    {
        if(arg.compute_type == HIPBLAS_COMPUTE_32I)
        {
            real_bias_type = HIP_R_32I;
        }
        else if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
        {
            real_bias_type = HIP_R_32F;
        }
        else if((arg.a_type == HIP_R_8F_E4M3_FNUZ || arg.a_type == HIP_R_8F_E5M2_FNUZ)
                && (arg.b_type == HIP_R_8F_E4M3_FNUZ || arg.b_type == HIP_R_8F_E5M2_FNUZ))
        {
            if(arg.d_type == HIP_R_32F || arg.d_type == HIP_R_16BF)
                real_bias_type = HIP_R_16BF;
            else if(arg.d_type == HIP_R_16F)
                real_bias_type = HIP_R_16F;
            else //more default cases once support C != D
                real_bias_type = HIP_R_16F;
        }
        else if((arg.a_type == HIP_R_8F_E4M3 || arg.a_type == HIP_R_8F_E5M2)
                && (arg.b_type == HIP_R_8F_E4M3 || arg.b_type == HIP_R_8F_E5M2))
        {
            if(arg.d_type == HIP_R_32F || arg.d_type == HIP_R_16BF)
                real_bias_type = HIP_R_16BF;
            else if(arg.d_type == HIP_R_16F)
                real_bias_type = HIP_R_16F;
            else //more default cases once support C != D
                real_bias_type = HIP_R_16F;
        }
        else if((arg.a_type == HIP_R_6F_E2M3 && arg.b_type == HIP_R_6F_E2M3)
                || (arg.a_type == HIP_R_6F_E3M2 && arg.b_type == HIP_R_6F_E3M2)
                || (arg.a_type == HIP_R_4F_E2M1 && arg.b_type == HIP_R_4F_E2M1))
        {
            if(arg.d_type == HIP_R_32F || arg.d_type == HIP_R_16BF)
                real_bias_type = HIP_R_16BF;
            else if(arg.d_type == HIP_R_16F)
                real_bias_type = HIP_R_16F;
            else
                real_bias_type = HIP_R_16F;
        }
        else
        {
            real_bias_type = arg.d_type;
        }
    }

    if(supported_bias_types.count(real_bias_type) == 0)
        throw std::invalid_argument("Invalid bias type "
                                    + std::string(hip_datatype_to_string(real_bias_type)));

    return real_bias_type;
}

// A function to determine the default aux_type
hipDataType derive_unset_aux_type(const Arguments& arg)
{
    static const std::set<hipDataType> supported_aux_types = {
        HIP_R_16F,
        HIP_R_16BF,
        HIP_R_8F_E4M3_FNUZ,
        HIP_R_8F_E4M3,
    };

    hipDataType real_aux_type = arg.aux_type;

    // when aux type is unset
    if(arg.aux_type == HIPBLASLT_DATATYPE_INVALID)
    {
        real_aux_type = arg.d_type;
    }

    if(real_aux_type != arg.d_type && supported_aux_types.count(real_aux_type) == 0)
        throw std::invalid_argument("Invalid aux type "
                                    + std::string(hip_datatype_to_string(real_aux_type)));

    return real_aux_type;
}

// A function to determine the default compute_input_type
std::tuple<hipDataType, hipDataType> derive_unset_compute_input_type(const Arguments& arg)
{
    static const std::set<hipDataType> supported_compute_input_types = {
        HIP_R_32F,
        HIP_R_16BF,
        HIP_R_16F,
        HIP_R_8F_E4M3,
        HIP_R_8F_E5M2,
        HIP_R_8F_E4M3_FNUZ,
        HIP_R_8F_E5M2_FNUZ,
        static_cast<hipDataType>(HIP_R_6F_E2M3),
        static_cast<hipDataType>(HIP_R_6F_E3M2),
        static_cast<hipDataType>(HIP_R_4F_E2M1),
    };

    hipDataType real_compute_input_typeA = arg.compute_input_typeA;
    hipDataType real_compute_input_typeB = arg.compute_input_typeB;

    if(real_compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
       && !supported_compute_input_types.count(real_compute_input_typeA))
        throw std::invalid_argument(
            "Invalid compute_input_typeA "
            + std::string(hip_datatype_to_string(real_compute_input_typeA)));

    if(real_compute_input_typeB != HIPBLASLT_DATATYPE_INVALID
       && !supported_compute_input_types.count(real_compute_input_typeB))
        throw std::invalid_argument(
            "Invalid compute_input_typeB "
            + std::string(hip_datatype_to_string(real_compute_input_typeB)));

    // when compute_input_type type is unset
    if(real_compute_input_typeA == HIPBLASLT_DATATYPE_INVALID)
    {
        real_compute_input_typeA = computeTypeToRealDataType(arg.compute_type);
    }

    if(real_compute_input_typeB == HIPBLASLT_DATATYPE_INVALID)
    {
        real_compute_input_typeB = computeTypeToRealDataType(arg.compute_type);
    }

    return {real_compute_input_typeA, real_compute_input_typeB};
}

std::vector<hipblaslt::host_validation::MatmulValidationCase::PointwiseTolerance>
    matmulValidationTolerances(const Arguments&         arg,
                               std::span<const int64_t> reductionLengths,
                               hipDataType              inputTypeA,
                               hipDataType              inputTypeB,
                               hipDataType              outputType,
                               hipDataType              computeType)
{
    using PointwiseTolerance = hipblaslt::host_validation::MatmulValidationCase::PointwiseTolerance;
    std::vector<PointwiseTolerance> tolerances(reductionLengths.size());
    const bool                      bfloat16Output = outputType == HIP_R_16BF;

    if(arg.unit_check && hipblaslt_get_arch_major() == 11 && realDataTypeSize(inputTypeA) == 2
       && realDataTypeSize(inputTypeB) == 2)
    {
        for(size_t i = 0; i < reductionLengths.size(); ++i)
        {
            tolerances[i].symmetricRelative
                = gfx11_low_precision_accumulation_tolerance_coefficient(computeType,
                                                                         reductionLengths[i]);
            if(bfloat16Output)
                tolerances[i].symmetricRelative
                    = std::max(tolerances[i].symmetricRelative,
                               bfloat16_output_rounding_tolerance_coefficient());
        }
    }

    if(arg.initialization == hipblaslt_initialization::fp16_accumulator_probe
       || arg.initialization == hipblaslt_initialization::norm_dist_one_special)
    {
        for(auto& tolerance : tolerances)
        {
            tolerance          = {};
            tolerance.absolute = 1e-2;
        }
    }
    return tolerances;
}

void testing_matmul_with_bias(const Arguments& arg,
                              hipDataType      TiA,
                              hipDataType      TiB,
                              hipDataType      To,
                              hipDataType      Tc,
                              hipDataType      TciA,
                              hipDataType      TciB,
                              hipDataType      Tbias,
                              hipDataType      Taux);

void testing_matmul(const Arguments& arg)
{
    hipDataType tiA = arg.a_type;
    hipDataType tiB = arg.b_type;
    hipDataType to  = arg.c_type;
    hipDataType tc  = computeTypeToRealDataType(arg.compute_type);
    hipDataType tciA, tciB;

    // after this, tciA and tciB should not be invalid
    std::tie(tciA, tciB) = derive_unset_compute_input_type(arg);

    // after this, real bias type should not be invalid
    hipDataType real_bias_type = derive_unset_bias_type(arg);
    Arguments   arg_revised    = arg;
    arg_revised.bias_type      = real_bias_type;

    hipDataType real_aux_type = derive_unset_aux_type(arg);
    arg_revised.aux_type      = real_aux_type;

    // Set the values of flush, rotating size, cold_iters and hot_iters only for internal use
    hipblasltSetFlushValue(arg.flush);
    hipblasltSetRotatingBufferSizeValue(arg.rotating);
    hipblasltSetColdIterationsValue(arg.cold_iters);
    hipblasltSetHotIterationsValue(arg.iters);

    // integer_exact: skip gfx11 only for 16-bit A (fp16/bf16)—GPU vs CPU exact match unreliable there;
    // f32/f64 (and TF32x1 f32 path) still run on Navi.
    if(arg.initialization == hipblaslt_initialization::integer_exact)
    {
        const bool is_16bit = (tiA == HIP_R_16F || tiA == HIP_R_16BF);
        if(hipblaslt_get_arch_major() == 11 && is_16bit)
        {
            hipblaslt_cout << "Skipping integer_exact on gfx11 for 16-bit float (fp16/bf16 A)"
                           << std::endl;
            return;
        }
        if(is_16bit)
        {
            // alpha=2: |2*dot|<=8K; beta=-2 adds 2*C. fp16 exact int ~2048 => K<=256 for both betas used
            const int32_t k_limit
                = (arg.alpha == 2.0f && (arg.beta == 0.0f || arg.beta == -2.0f)) ? 256 : 512;
            const int32_t problem_count = std::max(1, arg.grouped_gemm);
            for(int32_t i = 0; i < problem_count; i++)
            {
                if(arg.K[i] > k_limit)
                {
                    hipblaslt_cout << "Skipping integer_exact: 16-bit format with K=" << arg.K[i]
                                   << " > " << k_limit << " (exact representability limit)"
                                   << std::endl;
                    return;
                }
            }
        }
    }

    // FP16 full-matrix accumulator probe (see hipblaslt_init_device fp16_accumulator_probe).
    if(arg.initialization == hipblaslt_initialization::fp16_accumulator_probe)
    {
        if(tiA != HIP_R_16F || tiB != HIP_R_16F || to != HIP_R_16F || arg.d_type != HIP_R_16F
           || arg.compute_type != HIPBLAS_COMPUTE_32F)
        {
            hipblaslt_cout
                << "Skipping fp16_accumulator_probe: requires f16 A/B/C/D and HIPBLAS_COMPUTE_32F"
                << std::endl;
            return;
        }
        if(arg.transA != 'N' || arg.transB != 'N')
        {
            hipblaslt_cout << "Skipping fp16_accumulator_probe: only NN transposes supported"
                           << std::endl;
            return;
        }
        if(arg.grouped_gemm > 0)
        {
            hipblaslt_cout << "Skipping fp16_accumulator_probe: grouped_gemm not supported"
                           << std::endl;
            return;
        }
        if(arg.bias_vector || arg.activation_type != hipblaslt_activation_type::none || arg.use_e
           || arg.gradient || arg.scaleA != hipblaslt_scaling_format::none
           || arg.scaleB != hipblaslt_scaling_format::none || arg.scaleC || arg.scaleD || arg.scaleE
           || arg.scaleAlpha_vector || arg.amaxScaleA || arg.amaxScaleB || arg.amaxD)
        {
            hipblaslt_cout
                << "Skipping fp16_accumulator_probe: requires default epilogue (no bias, "
                   "activation, aux, or scaling)"
                << std::endl;
            return;
        }
        if(arg.beta != 0.0f)
        {
            hipblaslt_cout << "Skipping fp16_accumulator_probe: requires beta == 0" << std::endl;
            return;
        }
        const int32_t gemm_count_pe = std::max(1, arg.grouped_gemm);
        for(int32_t i = 0; i < gemm_count_pe; i++)
        {
            if((arg.K[i] & 1) != 0)
            {
                hipblaslt_cout << "Skipping fp16_accumulator_probe: odd K not supported (K="
                               << arg.K[i] << ")" << std::endl;
                return;
            }
        }
    }

    const bool lowPrecisionInput
        = (realDataTypeSize(tiA) == 1 || realDataTypeSize(tiB) == 1) && tc != HIP_R_32I;
    hipDataType executionBiasType = to;
    if(lowPrecisionInput || to == HIP_R_16F || to == HIP_R_16BF)
    {
        const hipDataType preferredNarrowBias
            = (to == HIP_R_16BF || to == HIP_R_32F) ? HIP_R_16BF : HIP_R_16F;
        executionBiasType
            = real_bias_type == preferredNarrowBias ? preferredNarrowBias : HIP_R_32F;
    }
    else if(to != HIP_R_32F && to != HIP_R_32I && to != HIP_R_8I && to != HIP_R_64F
            && to != HIP_C_32F && to != HIP_C_64F)
    {
        hipblaslt_test_invalid{}(arg);
        return;
    }

    testing_matmul_with_bias(
        arg_revised, tiA, tiB, to, tc, tciA, tciB, executionBiasType, real_aux_type);
}

void testing_matmul_with_bias(const Arguments& arg,
                              hipDataType      TiA,
                              hipDataType      TiB,
                              hipDataType      To,
                              hipDataType      Tc,
                              hipDataType      TciA,
                              hipDataType      TciB,
                              hipDataType      Tbias,
                              hipDataType      Taux)
{
    const auto matmulCases = hipblaslt::client::normalizeMatmulCases(arg);
    const auto& firstCase  = matmulCases.front();

    double gpu_time_used, cpu_time_used, gpu_mem_gbytes;
    gpu_time_used = cpu_time_used = gpu_mem_gbytes = 0.0;
    bool                   HMM                     = arg.HMM;
    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipEvent_t event_gpu_time_start, event_gpu_time_end;
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_end));

    const hipblasOperation_t transA = firstCase.operationA;
    const hipblasOperation_t transB = firstCase.operationB;

    // If input type is complex then alpha is set to complex datatype else compute type 
    hipDataType Talpha = (TiA == HIP_C_32F || TiA == HIP_C_64F) ?  TiA : Tc;

    const bool    do_grouped_gemm = arg.grouped_gemm > 0;
    const int32_t problem_count   = static_cast<int32_t>(matmulCases.size());
    const hipblasLtBatchMode_t batchMode = firstCase.batchMode;
    
    int64_t rotating  = arg.rotating * 1024 * 1024;

    std::vector<int64_t> M(problem_count), N(problem_count), K(problem_count), lda(problem_count),
        ldb(problem_count), ldc(problem_count), ldd(problem_count), lde(problem_count);
    std::vector<computeTypeInterface> h_alpha(problem_count, computeTypeInterface{}),
        h_beta(problem_count, computeTypeInterface{});
    std::vector<int64_t> A_row(problem_count), A_col(problem_count), B_row(problem_count), B_col(problem_count);
    std::vector<int64_t> stride_a(problem_count), stride_da(problem_count), stride_b(problem_count),
        stride_db(problem_count), stride_c(problem_count), stride_d(problem_count), stride_e(problem_count);
    std::vector<bool>   do_batched(problem_count), epilogue_on(problem_count, false);
    std::vector<int>    num_batches(problem_count);
    std::vector<size_t> size_A(problem_count), size_dA(problem_count), size_B(problem_count),
        size_dB(problem_count), size_C(problem_count), size_D(problem_count), size_D_copy(problem_count),
        size_E(problem_count), size_bias(problem_count), size_scaleAlphaVec(problem_count),
        size_scaleAVec(problem_count), size_scaleBVec(problem_count);

    std::vector<hipblasLtMatrixLayout_t> matA(problem_count), matB(problem_count), matC(problem_count),
        matD(problem_count);
    std::vector<std::vector<hipblasLtMatmulDesc_t>> matmul;
    std::vector<hipblasLtEpilogue_t> epilogue(problem_count, HIPBLASLT_EPILOGUE_DEFAULT);
    std::vector<float>               act0(problem_count), act1(problem_count);

    std::vector<HipDeviceBuffer>  dA, dB, dC, dD, dE, dBias;
    std::vector<HipDeviceBuffer>* dDp;
    std::vector<HipDeviceBuffer>  dScaleAlphaVec, dScaleA, dScaleB, dScaleC, dScaleD, dScaleE,
        dAmaxD;

    std::vector<HipHostBuffer> hE, hE_gold, hBias, hBias_gold;
    std::vector<HipHostBuffer> hA, hB, hC, hD_gold, hD_1;
    std::vector<HipHostBuffer> hScaleAlphaVec, hScaleA, hScaleB, hScaleC, hScaleD, hScaleE,
        hAmaxD_gold, hAmaxD, hD_gold_epl, hD_gold_ScaleAlpha, hBias_gold_epl;

    // These two vectors store the float values of MX data. Host validation
    // generates MX data and returns the corresponding float values. The float
    // values can be directly used for CPU verification (hipblaslt_reference_gemm) instead
    // of converting the MX data to float again.
    std::vector<std::vector<float>> refA, refB;

    std::vector<void*> alpha_in(problem_count);

    bool do_swizzle_a = arg.swizzle_a && isSwizzleSupported(TiA);
    bool do_swizzle_b = arg.swizzle_b && isSwizzleSupported(TiB);
    bool mx_use_rocroller = MXUseRocroller();

    // Need to split into two for loop to calculate the rotating buffer
    auto divideRoundUp = [](size_t value, size_t divisor) {
        return value / divisor + static_cast<size_t>(value % divisor != 0);
    };
    int64_t totalRotatingSizeNeeded = 0;
    for(int i = 0; i < problem_count; i++)
    {
        const auto& testCase = matmulCases[i];
        M[i]                 = testCase.m;
        N[i]                 = testCase.n;
        K[i]                 = testCase.k;
        set_alpha_type(h_alpha[i], arg, Tc, TiA);
        set_beta_type(h_beta[i], arg, Tc, TiA);
        lda[i] = testCase.a.leadingDimension();
        ldb[i] = testCase.b.leadingDimension();
        ldc[i] = testCase.c.leadingDimension();
        ldd[i] = testCase.d.leadingDimension();
        lde[i] = testCase.auxiliary ? testCase.auxiliary->leadingDimension() : arg.lde[i];

        A_row[i] = testCase.a.rows();
        A_col[i] = testCase.a.columns();
        B_row[i] = testCase.b.rows();
        B_col[i] = testCase.b.columns();

        do_batched[i]  = testCase.batchCount > 1;
        num_batches[i] = testCase.batchCount;

        stride_a[i] = testCase.a.batchStride();
        stride_b[i] = testCase.b.batchStride();
        stride_c[i] = testCase.c.batchStride();
        stride_d[i] = testCase.d.batchStride();
        stride_e[i] = testCase.auxiliary
                          ? testCase.auxiliary->batchStride()
                          : (batchMode == HIPBLASLT_BATCH_MODE_STRIDED && do_batched[i]
                                 ? arg.stride_e[i]
                                 : lde[i] * N[i]);

        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            size_A[i] = stride_a[i] == 0        ? lda[i] * A_col[i] * num_batches[i]
                        : lda[i] <= stride_a[i] ? stride_a[i] * num_batches[i]
                                                : lda[i] * A_col[i];
        }
        else
        {
            size_A[i] = stride_a[i];
        }
        // for (!do_swizzle_a) case, we can use size_dA and stride_da instead of size_A and stride_a
        size_dA[i]   = size_A[i];
        stride_da[i] = stride_a[i];
        if(do_swizzle_a)
        {
            size_t MiM = 16, MiK = 0, __ = 0, PackK = 0;
            calculateKforSwizzling(TiA, arg, MiK, __, PackK);
            size_t  K_block = MiK * PackK;
            int64_t stride_swizzle
                = ((M[i] + MiM - 1) / MiM) * MiM * ((K[i] + K_block - 1) / K_block) * K_block;
            if(do_batched[i] && stride_a[i] != 0)
            {
                stride_da[i] = stride_swizzle;

                //TODO: support arbitrary stride_a for both hipblaslt-bench and hipblaslt-test when swizzled
                if(stride_a[i] != lda[i] * A_col[i] && stride_a[i] != stride_swizzle)
                    hipblaslt_cerr << "Warning: swizzle_a does not yet support arbitrary stride_a!"
                                   << std::endl;
            }
            size_dA[i] = (batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) ? stride_swizzle : (num_batches[i] * stride_swizzle);
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            size_B[i] = stride_b[i] == 0        ? ldb[i] * B_col[i] * num_batches[i]
                        : ldb[i] <= stride_b[i] ? stride_b[i] * num_batches[i]
                                                : ldb[i] * B_col[i];
        }
        else
        {
            size_B[i] = stride_b[i];
        }
        // for (!do_swizzle_b) case, we can use size_dB and stride_db instead of size_B and stride_b
        size_dB[i]   = size_B[i];
        stride_db[i] = stride_b[i];
        if(do_swizzle_b)
        {
            size_t MiN = 16, MiK = 0, __ = 0, PackK = 0;
            calculateKforSwizzling(TiB, arg, MiK, __, PackK);
            size_t  K_block = MiK * PackK;
            int64_t stride_swizzle
                = ((N[i] + MiN - 1) / MiN) * MiN * ((K[i] + K_block - 1) / K_block) * K_block;
            if(do_batched[i] && stride_b[i] != 0)
            {
                stride_db[i] = stride_swizzle;

                //TODO: support arbitrary stride_b for both hipblaslt-bench and hipblaslt-test when swizzled
                if(stride_b[i] != ldb[i] * B_col[i] && stride_b[i] != stride_swizzle)
                    hipblaslt_cerr << "Warning: swizzle_b does not yet support arbitrary stride_b!"
                                   << std::endl;
            }
            size_dB[i] = (batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) ? stride_swizzle : (num_batches[i] * stride_swizzle);
        }
        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            size_C[i] = stride_c[i] == 0        ? ldc[i] * N[i] * num_batches[i]
                        : ldc[i] <= stride_c[i] ? stride_c[i] * num_batches[i]
                                                : ldc[i] * N[i];
            size_D[i] = stride_d[i] == 0        ? ldd[i] * N[i] * num_batches[i]
                        : ldd[i] <= stride_d[i] ? stride_d[i] * num_batches[i]
                                                : ldd[i] * N[i];
            size_E[i] = arg.use_e ? (stride_e[i] == 0        ? lde[i] * N[i] * num_batches[i]
                                     : lde[i] <= stride_e[i] ? stride_e[i] * num_batches[i]
                                                             : lde[i] * N[i])
                                  : 0;
        }
        else
        {
            size_C[i] = ldc[i] * N[i];
            size_D[i] = ldd[i] * N[i];
            size_E[i] = lde[i] * N[i];
        }
        if(arg.c_equal_d)
        {
            ldd[i]      = arg.ldc[i];
            stride_d[i] = stride_c[i];
            size_D[i]   = size_C[i];
        }

        size_D_copy[i] = (arg.unit_check || arg.norm_check || arg.allclose_check) ? size_D[i] : 0;
        size_scaleAlphaVec[i] = arg.scaleAlpha_vector ? M[i] : 0;
        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
                size_scaleAVec[i] = 1;
            else if(arg.scaleA == hipblaslt_scaling_format::Vector)
                size_scaleAVec[i] = M[i];
            else if(isBlockScaling(arg.scaleA))
            {
                if(!mx_use_rocroller)
                {
                    // Account for padding in the swizzled MX layout
                    size_t MXBlock_A   = blockSize(arg.scaleA);
                    size_t dimk        = 128 / MXBlock_A;
                    size_t scaleA_r
                        = transA == HIPBLAS_OP_T
                              ? divideRoundUp(static_cast<size_t>(A_row[i]), MXBlock_A)
                              : static_cast<size_t>(A_row[i]);
                    size_t scaleA_c
                        = transA == HIPBLAS_OP_T
                              ? static_cast<size_t>(A_col[i])
                              : divideRoundUp(static_cast<size_t>(A_col[i]), MXBlock_A);
                    bool   kAlongRowsA = (transA == HIPBLAS_OP_T);
                    size_t kDim        = kAlongRowsA ? scaleA_r : scaleA_c;
                    size_t mnDim       = kAlongRowsA ? scaleA_c : scaleA_r;
                    size_t padDim      = kAlongRowsA ? kDim : mnDim;
                    size_t paddedDim   = (padDim + dimk - 1) / dimk * dimk;
                    size_scaleAVec[i]  = kAlongRowsA ? (mnDim * paddedDim) : (kDim * paddedDim);
                }
                else
                {
                    size_scaleAVec[i] = scaleBufferSize(A_row[i], A_col[i], arg.scaleA);
                }
            }
            else
                size_scaleAVec[i] = 0;
            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
                size_scaleBVec[i] = 1;
            else if(arg.scaleB == hipblaslt_scaling_format::Vector)
                size_scaleBVec[i] = N[i];
            else if(isBlockScaling(arg.scaleB))
            {
                if(!mx_use_rocroller)
                {
                    // Account for padding in the swizzled MX layout
                    size_t MXBlock_B   = blockSize(arg.scaleB);
                    size_t dimk        = 128 / MXBlock_B;
                    size_t scaleB_r
                        = transB == HIPBLAS_OP_T
                              ? static_cast<size_t>(B_row[i])
                              : divideRoundUp(static_cast<size_t>(B_row[i]), MXBlock_B);
                    size_t scaleB_c
                        = transB == HIPBLAS_OP_T
                              ? divideRoundUp(static_cast<size_t>(B_col[i]), MXBlock_B)
                              : static_cast<size_t>(B_col[i]);
                    bool   kAlongRowsB = (transB == HIPBLAS_OP_N);
                    size_t kDim        = kAlongRowsB ? scaleB_r : scaleB_c;
                    size_t mnDim       = kAlongRowsB ? scaleB_c : scaleB_r;
                    size_t padDim      = kAlongRowsB ? kDim : mnDim;
                    size_t paddedDim   = (padDim + dimk - 1) / dimk * dimk;
                    size_scaleBVec[i]  = kAlongRowsB ? (mnDim * paddedDim) : (kDim * paddedDim);
                }
                else
                {
                    size_scaleBVec[i] = scaleBufferSize(B_row[i], B_col[i], arg.scaleB);
                }
            }
            else
                size_scaleBVec[i] = 0;
        }
        else
        {
            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
                size_scaleAVec[i] = 1;
            else if(arg.scaleA == hipblaslt_scaling_format::none)
                size_scaleAVec[i] = 0;
            else
            {
                hipblaslt_cout << "Only Tensorwide scaling is supported for General Batched GEMM"
                               << std::endl;
                return;
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
                size_scaleBVec[i] = 1;
            else if(arg.scaleB == hipblaslt_scaling_format::none)
                size_scaleBVec[i] = 0;
            else
            {
                hipblaslt_cout << "Only Tensorwide scaling is supported for General Batched GEMM"
                               << std::endl;
                return;
            }
        }
        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            if(arg.bias_vector)
            {
                if(arg.bias_source == hipblaslt_bias_source::a
                   || arg.bias_source == hipblaslt_bias_source::d)
                    size_bias[i] = M[i];
                else if(arg.bias_source == hipblaslt_bias_source::b)
                    size_bias[i] = N[i];

                if(arg.bias_stride > 0)
                {
                    size_bias[i] = arg.bias_stride * num_batches[i];
                }
            }
            else
            {
                size_bias[i] = 0;
            }
        }
        else
        {
            size_bias[i] = 0;
        }
        auto    biasSize = size_bias[i] * realDataTypeSize(Tbias);
        int64_t sizeC    = get_computeInterface(h_beta[i], Tc) == 0 ? 0 : size_C[i] * sizeof(To);
        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            totalRotatingSizeNeeded
                += size_dA[i] * realDataTypeSize(TiA) + size_dB[i] * realDataTypeSize(TiB) + sizeC
                   + size_D[i] * realDataTypeSize(To) + size_E[i] * realDataTypeSize(To) + biasSize
                   + size_scaleAlphaVec[i] * realDataTypeSize(Talpha)
                   + size_scaleAVec[i] * realDataTypeSize(Talpha)
                   + size_scaleBVec[i] * realDataTypeSize(Talpha);
        }
        else
        {
            // For General Batched GEMM, the Matrices aren't stored in a continuous buffer across batches.
            // Hence size_dA doesn't account for all batches.
            totalRotatingSizeNeeded += size_dA[i] * realDataTypeSize(TiA) * num_batches[i]
                                       + size_dB[i] * realDataTypeSize(TiB) * num_batches[i]
                                       + sizeC * num_batches[i]
                                       + size_D[i] * realDataTypeSize(To) * num_batches[i]
                                       + biasSize + size_scaleAlphaVec[i] * realDataTypeSize(Talpha)
                                       + size_scaleAVec[i] * realDataTypeSize(Talpha)
                                       + size_scaleBVec[i] * realDataTypeSize(Talpha);
        }
    }

    gpu_mem_gbytes = static_cast<double>(totalRotatingSizeNeeded) / (1024 * 1024 * 1024);

    // Calculating block count
    auto plan = hipblaslt_bench::compute_rotating_buffer_plan(
        arg.adaptive, arg.max_iters, arg.cold_iters, arg.iters, rotating, totalRotatingSizeNeeded);
    int32_t block_count = plan.block_count;
    if(rotating > 0)
    {
        hipblaslt_cout << "Rotating buffer " << rotating / (1024 * 1024) << " MiB. "
                       << "Needed Size: " << totalRotatingSizeNeeded / (1024 * 1024) << " MiB. "
                       << "Needed block count: " << block_count;
        if(plan.capped)
            hipblaslt_cout << " (Capped to max iters: " << plan.iter_cap << ")";
        hipblaslt_cout << std::endl;
    }
    // Calculating block count end
    matmul.resize(block_count, std::vector<hipblasLtMatmulDesc_t>(problem_count));

    for(int i = 0; i < problem_count; i++)
    {
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matA[i]), arg.a_type, A_row[i], A_col[i], lda[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matB[i]), arg.b_type, B_row[i], B_col[i], ldb[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matC[i]), arg.c_type, M[i], N[i], ldc[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matD[i]), arg.d_type, M[i], N[i], ldd[i]));

        if(do_swizzle_a)
        {
            hipblasLtOrder_t orderA = orderForDatatype(TiA);
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matA[i], HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
        }

        if(do_swizzle_b)
        {
            hipblasLtOrder_t orderB = orderForDatatype(TiB);
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matB[i], HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderB, sizeof(orderB)));
        }

        if(do_batched[i] || batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matA[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matB[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matC[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matD[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matA[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_da[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matB[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_db[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matC[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_c[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matD[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_d[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matA[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batchMode, sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matB[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batchMode, sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matC[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batchMode, sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matD[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &batchMode, sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
        }

        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatmulDescCreate(&(matmul[0][i]), arg.compute_type, arg.scale_type));

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &TciA, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &TciB, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[0][i], HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[0][i], HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));

        // Forward CLI knobs from hipblaslt-bench into the matmul descriptor.
        {
            int32_t sm = hipblaslt_bench_options::sm_count_target();
            if(sm != 0)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i],
                    HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                    &sm,
                    sizeof(sm)));
            }
            int32_t dyn = hipblaslt_bench_options::streamk_tile_scheduling_mode();
            if(dyn >= 0)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i],
                    HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                    &dyn,
                    sizeof(dyn)));
            }
            int32_t uso = hipblaslt_bench_options::uniform_summation_order();
            if(uso >= 0)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i],
                    HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT,
                    &uso,
                    sizeof(uso)));
            }
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            epilogue[i]    = matmulEpilogue(arg);
            epilogue_on[i] = epilogue[i] != HIPBLASLT_EPILOGUE_DEFAULT
                             || arg.scaleAlpha_vector;
            if(epilogue_on[i])
            {
                act0[i] = arg.activation_arg1;
                act1[i] = arg.activation_arg2;
            }

            // allocate memory on device
            dA.emplace_back(TiA, size_dA[i] * block_count, HMM);
            CHECK_DEVICE_ALLOCATION(hipGetLastError());
            dB.emplace_back(TiB, size_dB[i] * block_count, HMM);
            CHECK_DEVICE_ALLOCATION(hipGetLastError());
            dC.emplace_back(To, size_C[i] * block_count, HMM);
            CHECK_DEVICE_ALLOCATION(hipGetLastError());

            if(!arg.c_equal_d)
            {
                dD.emplace_back(To, size_D[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
                dDp = &dD;
            }
            else
                dDp = &dC;

            if(size_bias[i] * block_count != 0)
            {
                dBias.emplace_back(Tbias, size_bias[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(arg.scaleAlpha_vector)
            {
                dScaleAlphaVec.emplace_back(Talpha, size_scaleAlphaVec[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(arg.use_e)
            {
                dE.emplace_back(Taux, size_E[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                dScaleA.emplace_back(Talpha, size_scaleAVec[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else if(isBlockScaling(arg.scaleA))
            {
                // For MX format, use uint8_t for the scale (E8M0), allocate for all batches
                dScaleA.emplace_back(HIP_R_8U, size_scaleAVec[i] * num_batches[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                dScaleB.emplace_back(Talpha, size_scaleBVec[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else if(isBlockScaling(arg.scaleB))
            {
                // For MX format, use uint8_t for the scale (E8M0), allocate for all batches
                dScaleB.emplace_back(HIP_R_8U, size_scaleBVec[i] * num_batches[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleC)
            {
                dScaleC.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleD)
            {
                dScaleD.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.amaxD)
            {
                epilogue_on[i] = true;
                dAmaxD.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleE)
            {
                dScaleE.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }

            // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
            hA.emplace_back(TiA, size_A[i]);
            hB.emplace_back(TiB, size_B[i]);
            hC.emplace_back(To, size_C[i]);
            hD_gold.emplace_back(To, size_D_copy[i]);
            hD_1.emplace_back(To, size_D_copy[i]);
            if(size_bias[i] * block_count != 0)
            {
                hBias.emplace_back(Tbias, size_bias[i]);
                hBias_gold.emplace_back(Tbias, size_bias[i]);
            }

            hD_gold_epl.emplace_back(Talpha, size_D_copy[i]);
            hD_gold_ScaleAlpha.emplace_back(Talpha, size_D_copy[i]);
            hBias_gold_epl.emplace_back(Talpha, size_D_copy[i]); // Reduction for matrix D

            if(arg.scaleAlpha_vector)
                hScaleAlphaVec.emplace_back(Talpha, size_scaleAlphaVec[i]);

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                hScaleA.emplace_back(Talpha, size_scaleAVec[i]);
            }
            else if(isBlockScaling(arg.scaleA))
            {
                hScaleA.emplace_back(HIP_R_8U, size_scaleAVec[i] * num_batches[i]);
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                hScaleB.emplace_back(Talpha, size_scaleBVec[i]);
            }
            else if(isBlockScaling(arg.scaleB))
            {
                hScaleB.emplace_back(HIP_R_8U, size_scaleBVec[i] * num_batches[i]);
            }
            if(arg.scaleC)
                hScaleC.emplace_back(Talpha, 1);
            if(arg.scaleD)
                hScaleD.emplace_back(Talpha, 1);
            if(arg.amaxD)
            {
                hAmaxD_gold.emplace_back(Talpha, 1);
                hAmaxD.emplace_back(Talpha, 1);
            }
            if(arg.scaleE)
                hScaleE.emplace_back(Talpha, 1);

            if(arg.use_e)
            {
                hE.emplace_back(Taux, size_E[i]);
                if(!arg.gradient)
                {
                    hE_gold.emplace_back(Taux, size_E[i]);
                }
            }
        }
        else
        {
            for(int batchCount = 0; batchCount < arg.batch_count; batchCount++)
            {
                // allocate memory on device
                dA.emplace_back(TiA, size_dA[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
                dB.emplace_back(TiB, size_dB[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
                dC.emplace_back(To, size_C[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());

                if(!arg.c_equal_d)
                {
                    dD.emplace_back(To, size_D[i] * block_count, HMM);
                    CHECK_DEVICE_ALLOCATION(hipGetLastError());
                    dDp = &dD;
                }
                else
                    dDp = &dC;

                if(size_bias[i] * block_count != 0)
                {
                    dBias.emplace_back(Tbias, size_bias[i] * block_count, HMM);
                    CHECK_DEVICE_ALLOCATION(hipGetLastError());
                }

                if(arg.scaleAlpha_vector)
                {
                    hipblaslt_cout << "General Batched GEMM does not support scaleAlpha_vector."
                                   << std::endl;
                    return;
                }

                if(arg.use_e)
                {
                    hipblaslt_cout << "General Batched GEMM does not support use_e." << std::endl;
                    return;
                }

                // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
                hA.emplace_back(TiA, size_A[i]);
                hB.emplace_back(TiB, size_B[i]);
                hC.emplace_back(To, size_C[i]);
                hD_gold.emplace_back(To, size_D_copy[i]);
                hD_1.emplace_back(To, size_D_copy[i]);
                if(size_bias[i] * block_count != 0)
                {
                    hBias.emplace_back(Tbias, size_bias[i]);
                    hBias_gold.emplace_back(Tbias, size_bias[i]);
                }
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::none)
            {
                dScaleA.emplace_back(Talpha, size_scaleAVec[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::none)
            {
                dScaleB.emplace_back(Talpha, size_scaleBVec[i] * block_count, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleC)
            {
                dScaleC.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.scaleD)
            {
                dScaleD.emplace_back(Talpha, 1, HMM);
                CHECK_DEVICE_ALLOCATION(hipGetLastError());
            }
            if(arg.amaxD)
            {
                hipblaslt_cout << "General Batched GEMM doesn't support Epilogues."
                               << "Only Scaling and Quantization is supported for post processing"
                               << std::endl;
                return;
            }
            if(arg.scaleE)
            {
                hipblaslt_cout << "General Batched GEMM doesn't support Epilogues."
                               << "Only Scaling and Quantization is supported for post processing"
                               << std::endl;
                return;
            }
            if(arg.scaleAlpha_vector)
            {
                hipblaslt_cout << "General Batched GEMM does not support scaleAlpha_vector."
                               << std::endl;
                return;
            }

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::none)
            {
                hScaleA.emplace_back(Talpha, size_scaleAVec[i]);
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::none)
            {
                hScaleB.emplace_back(Talpha, size_scaleBVec[i]);
            }
            else
            {
                hipblaslt_cout << "General Batched GEMM only support Tensorwide scaling."
                               << std::endl;
                return;
            }
            if(arg.scaleC)
                hScaleC.emplace_back(Talpha, 1);
            if(arg.scaleD)
                hScaleD.emplace_back(Talpha, 1);
        }

        const bool positiveOnlyInitialization
            = arg.ulp_check
              && (arg.initialization == hipblaslt_initialization::hpl
                  || arg.initialization == hipblaslt_initialization::trig_float);

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
        hipDeviceProp_t mxProp{};
        if(isBlockScaling(arg.scaleA) || isBlockScaling(arg.scaleB))
            CHECK_HIP_ERROR(hipGetDeviceProperties(&mxProp, 0));
        auto mxBatchOutput = [](HipHostBuffer& buffer, size_t offset, size_t batchBytes) {
            if(offset > buffer.getNumBytes())
                throw std::invalid_argument("MX output offset exceeds host buffer capacity.");
            size_t capacity = buffer.getNumBytes() - offset;
            if(batchBytes != 0)
                capacity = std::min(capacity, batchBytes);
            return std::span<uint8_t>(buffer.as<uint8_t>() + offset, capacity);
        };
#endif

        size_t scaleA_row = ((transA == HIPBLAS_OP_T) ? blockSize(arg.scaleA) : 1);
        size_t scaleA_col = ((transA == HIPBLAS_OP_T) ? 1 : blockSize(arg.scaleA));
        if(isBlockScaling(arg.scaleA))
        {
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
            if(arg.initialization != hipblaslt_initialization::hpl
               && arg.initialization != hipblaslt_initialization::trig_float
               && arg.initialization != hipblaslt_initialization::uniform_01
               && arg.initialization != hipblaslt_initialization::zero
               && arg.initialization != hipblaslt_initialization::norm_dist
               && arg.initialization != hipblaslt_initialization::rand_int
               && arg.initialization != hipblaslt_initialization::uniform_low_precision)
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "unsupported MX initialization: "
                             << hipblaslt_initialization2string(arg.initialization);
#else
                hipblaslt_cout << "Initialization of microscaling data only allows hpl, trig_float, "
                                  "uniform_01, zero, norm_dist, rand_int or uniform_low_precision, not "
                               << hipblaslt_initialization2string(arg.initialization) << std::endl;
                return;
#endif
            }
            if(arg.algo_method == 1)
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "MX data types do not support algorithm \"all\"";
#else
                hipblaslt_cout << "MX data types do not support algorithm \"all\"" << std::endl;
                return;
#endif
            }
            MXScaleLayout const scaleLayoutA
                = mxScaleLayoutForFormat(arg.scaleA, mxProp.gcnArchName);
            size_t dataBatchBytesA  = (num_batches[i] > 1) ? elementsToBytes(stride_a[i], TiA) : 0;
            size_t scaleBatchBytesA = (num_batches[i] > 1) ? size_scaleAVec[i] : 0;
            std::vector<float> refAAll;
            refAAll.reserve(static_cast<size_t>(A_row[i]) * A_col[i] * num_batches[i]);
            for(int64_t b = 0; b < num_batches[i]; b++)
            {
                auto dataOutputA = mxBatchOutput(hA[i], b * dataBatchBytesA, dataBatchBytesA);
                auto scaleOutputA
                    = mxBatchOutput(hScaleA[i], b * scaleBatchBytesA, scaleBatchBytesA);
                auto batchRef = generateMXInput(TiA,
                                                scaleDataType(arg.scaleA),
                                                dataOutputA,
                                                scaleOutputA,
                                                A_row[i],
                                                A_col[i],
                                                lda[i],
                                                scaleA_row,
                                                scaleA_col,
                                                scaleLayoutA,
                                                hipblaslt_initialization2string(arg.initialization),
                                                /*min_val=*/-1.0f,
                                                /*max_val=*/1.0f);
                refAAll.insert(refAAll.end(), batchRef.begin(), batchRef.end());
            }
            refA.emplace_back(std::move(refAAll));
            CHECK_HIP_ERROR(synchronize(dA[i], hA[i], block_count));
            CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
#else
#ifdef GOOGLE_TEST
            GTEST_SKIP() << "MX data initialization requires HIPBLASLT_ENABLE_MXDATAGENERATOR=ON at build time";
#else
            hipblaslt_cout << "MX data initialization requires HIPBLASLT_ENABLE_MXDATAGENERATOR=ON at build time"
                           << std::endl;
            return;
#endif
#endif
        }
        else
        {
            if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                hipblaslt_init_device(ABC_dims::A,
                                      arg.initialization,
                                      alpha_isnan_type(arg, Talpha),
                                      dA[i].buf(),
                                      A_row[i],
                                      A_col[i],
                                      (do_swizzle_a) ? A_row[i] : lda[i],
                                      TiA,
                                      (do_swizzle_a && stride_a[i] != 0) ? A_row[i] * A_col[i]
                                                                         : stride_a[i],
                                      num_batches[i],
                                      positiveOnlyInitialization);
            }
            else
            {
                for(int batchCount = 0; batchCount < num_batches[i]; batchCount++)
                {
                    hipblaslt_init_device(ABC_dims::A,
                                          arg.initialization,
                                          alpha_isnan_type(arg, Talpha),
                                          dA[batchCount].buf(),
                                          A_row[i],
                                          A_col[i],
                                          (do_swizzle_a) ? A_row[i] : lda[i],
                                          TiA,
                                          (do_swizzle_a && stride_a[i] != 0) ? A_row[i] * A_col[i]
                                                                             : stride_a[i],
                                          1,
                                          positiveOnlyInitialization);
                }
            }
        }

        size_t scaleB_row = ((transB == HIPBLAS_OP_T) ? 1 : blockSize(arg.scaleB));
        size_t scaleB_col = ((transB == HIPBLAS_OP_T) ? blockSize(arg.scaleB) : 1);
        if(isBlockScaling(arg.scaleB))
        {
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
            // MX B always goes through host validation (mirrors the A side above).
            if(arg.initialization != hipblaslt_initialization::hpl
               && arg.initialization != hipblaslt_initialization::trig_float
               && arg.initialization != hipblaslt_initialization::uniform_01
               && arg.initialization != hipblaslt_initialization::zero
               && arg.initialization != hipblaslt_initialization::norm_dist
               && arg.initialization != hipblaslt_initialization::rand_int
               && arg.initialization != hipblaslt_initialization::uniform_low_precision)
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "unsupported MX initialization: "
                             << hipblaslt_initialization2string(arg.initialization);
#else
                hipblaslt_cout << "Initialization of microscaling data only allows hpl, trig_float, "
                                  "uniform_01, zero, norm_dist, rand_int or uniform_low_precision, not "
                               << hipblaslt_initialization2string(arg.initialization) << std::endl;
                return;
#endif
            }
            if(arg.algo_method == 1)
            {
#ifdef GOOGLE_TEST
                GTEST_SKIP() << "MX data types do not support algorithm \"all\"";
#else
                hipblaslt_cout << "MX data types do not support algorithm \"all\"" << std::endl;
                return;
#endif
            }
            MXScaleLayout const scaleLayoutB
                = mxScaleLayoutForFormat(arg.scaleB, mxProp.gcnArchName);
            size_t dataBatchBytesB  = (num_batches[i] > 1) ? elementsToBytes(stride_b[i], TiB) : 0;
            size_t scaleBatchBytesB = (num_batches[i] > 1) ? size_scaleBVec[i] : 0;
            std::vector<float> refBAll;
            refBAll.reserve(static_cast<size_t>(B_row[i]) * B_col[i] * num_batches[i]);
            for(int64_t b = 0; b < num_batches[i]; b++)
            {
                auto dataOutputB = mxBatchOutput(hB[i], b * dataBatchBytesB, dataBatchBytesB);
                auto scaleOutputB
                    = mxBatchOutput(hScaleB[i], b * scaleBatchBytesB, scaleBatchBytesB);
                auto batchRef = generateMXInput(TiB,
                                                scaleDataType(arg.scaleB),
                                                dataOutputB,
                                                scaleOutputB,
                                                B_row[i],
                                                B_col[i],
                                                ldb[i],
                                                scaleB_row,
                                                scaleB_col,
                                                scaleLayoutB,
                                                hipblaslt_initialization2string(arg.initialization),
                                                /*min_val=*/-1.0f,
                                                /*max_val=*/1.0f);
                refBAll.insert(refBAll.end(), batchRef.begin(), batchRef.end());
            }
            refB.emplace_back(std::move(refBAll));
            CHECK_HIP_ERROR(synchronize(dB[i], hB[i], block_count));
            CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
#else
#ifdef GOOGLE_TEST
            GTEST_SKIP() << "MX data initialization requires HIPBLASLT_ENABLE_MXDATAGENERATOR=ON at build time";
#else
            hipblaslt_cout << "MX data initialization requires HIPBLASLT_ENABLE_MXDATAGENERATOR=ON at build time"
                           << std::endl;
            return;
#endif
#endif
        }
        else
        {
            if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
            {
                hipblaslt_init_device(ABC_dims::B,
                                      arg.initialization,
                                      alpha_isnan_type(arg, Talpha),
                                      dB[i].buf(),
                                      B_row[i],
                                      B_col[i],
                                      (do_swizzle_b) ? B_row[i] : ldb[i],
                                      TiB,
                                      (do_swizzle_b && stride_b[i] != 0) ? B_row[i] * B_col[i]
                                                                         : stride_b[i],
                                      num_batches[i],
                                      positiveOnlyInitialization);
            }
            else
            {
                for(int batchCount = 0; batchCount < num_batches[i]; batchCount++)
                {
                    hipblaslt_init_device(ABC_dims::B,
                                          arg.initialization,
                                          alpha_isnan_type(arg, Talpha),
                                          dB[batchCount].buf(),
                                          B_row[i],
                                          B_col[i],
                                          (do_swizzle_b) ? B_row[i] : ldb[i],
                                          TiB,
                                          (do_swizzle_b && stride_b[i] != 0) ? B_row[i] * B_col[i]
                                                                             : stride_b[i],
                                          1,
                                          positiveOnlyInitialization);
                }
            }
        }

        if(batchMode == HIPBLASLT_BATCH_MODE_STRIDED)
        {
            hipblaslt_init_device(ABC_dims::C,
                                  arg.initialization,
                                  beta_isnan_type(arg, Talpha),
                                  dC[i].buf(),
                                  M[i],
                                  N[i],
                                  ldc[i],
                                  To,
                                  stride_c[i],
                                  num_batches[i],
                                  positiveOnlyInitialization);

        // generateMXInput already produced the reference floats and the
        // kernel-ready scale layout for both A and B; nothing to do here.
            // broadcast first block
            CHECK_HIP_ERROR(broadcast(dA[i], block_count));
            CHECK_HIP_ERROR(broadcast(dB[i], block_count));
            CHECK_HIP_ERROR(broadcast(dC[i], block_count));

            if(arg.unit_check || arg.norm_check || arg.allclose_check || do_swizzle_a
               || do_swizzle_b)
            {
                CHECK_HIP_ERROR(synchronize(hA[i],
                                            dA[i],
                                            num_batches[i],
                                            A_row[i],
                                            A_col[i],
                                            lda[i],
                                            realDataTypeSize(TiA),
                                            do_swizzle_a,
                                            stream));
                // B is always stored as K×N in memory; use (K, N, ldb) not (B_row, B_col) to avoid row > lda when transB=T
                CHECK_HIP_ERROR(synchronize(hB[i],
                                            dB[i],
                                            num_batches[i],
                                            K[i],
                                            N[i],
                                            ldb[i],
                                            realDataTypeSize(TiB),
                                            do_swizzle_b,
                                            stream));
                CHECK_HIP_ERROR(synchronize(hC[i], dC[i], 0, 0, 0, 0, 1, false, stream));

                if(arg.dump_matrix)
                {
                    for(int batchId = 0; batchId < num_batches[i]; batchId++){
                        hipblasltDispatchValuesToFile(transA,
                                                    TiA,
                                                    M[i],
                                                    K[i],
                                                    lda[i],
                                                    hA[i].buf(),
                                                    "batch_" + std::to_string(batchId) + "_" + std::to_string(i) + "_A_input.txt");
                        hipblasltDispatchValuesToFile(transB,
                                                    TiB,
                                                    K[i],
                                                    N[i],
                                                    ldb[i],
                                                    hB[i].buf(),
                                                    "batch_" + std::to_string(batchId) + "_" + std::to_string(i) + "_B_input.txt");
                        hipblasltDispatchValuesToFile(HIPBLAS_OP_N,
                                                    To,
                                                    M[i],
                                                    N[i],
                                                    ldc[i],
                                                    hC[i].buf(),
                                                    "batch_" + std::to_string(batchId) + "_" + std::to_string(i) + "_C_input.txt");
                    }
                }
            }

            if(do_swizzle_a)
            {
                HipHostBuffer tmp(TiA, size_dA[i]);
                swizzle_tensor_type(
                    tmp, hA[i], TiA, arg, num_batches[i], M[i], K[i], lda[i], false);
                CHECK_HIP_ERROR(synchronize(dA[i], tmp, block_count));
            }

            if(do_swizzle_b)
            {
                HipHostBuffer tmp(TiB, size_dB[i]);
                swizzle_tensor_type(
                    tmp, hB[i], TiB, arg, num_batches[i], N[i], K[i], ldb[i], false);
                CHECK_HIP_ERROR(synchronize(dB[i], tmp, block_count));
            }

            if(arg.gradient && arg.use_e)
            {
                hipblaslt_init(hE[i].buf(), M[i], N[i], lde[i], Taux, stride_e[i], num_batches[i]);
            }

            if(arg.bias_vector)
            {
                // Filling up unique bias values for each batch in Strided Batch
                if(arg.bias_stride > 0)
                    hipblaslt_init(hBias[i].buf(), arg.bias_stride, 1, arg.bias_stride, Tbias, arg.bias_stride, num_batches[i]);
                else
                    hipblaslt_init(hBias[i].buf(), size_bias[i], 1, size_bias[i], Tbias);
            }

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                if(arg.norm_check)
                    hipblaslt_init_small(
                        hScaleA[i].buf(), size_scaleAVec[i], 1, size_scaleAVec[i], Talpha);
                else
                    hipblaslt_init(
                        hScaleA[i].buf(), size_scaleAVec[i], 1, size_scaleAVec[i], Talpha);
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                if(arg.norm_check)
                    hipblaslt_init_small(
                        hScaleB[i].buf(), size_scaleBVec[i], 1, size_scaleBVec[i], Talpha);
                else
                    hipblaslt_init(
                        hScaleB[i].buf(), size_scaleBVec[i], 1, size_scaleBVec[i], Talpha);
            }

            if(arg.scaleC)
            {
                if(To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ)
                {
                    hipblaslt_init_small(hScaleC[i].buf(), 1, 1, 1, Talpha);
                }
                else
                {
                    hipblaslt_init(hScaleC[i].buf(), 1, 1, 1, Talpha);
                }
            }

            if(arg.scaleD)
            {
                if(To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ)
                {
                    hipblaslt_init_small(hScaleD[i].buf(), 1, 1, 1, Talpha);
                }
                else
                {
                    hipblaslt_init(hScaleD[i].buf(), 1, 1, 1, Talpha);
                }
            }

            if(arg.amaxD)
                hipblaslt_init_zero(hAmaxD_gold[i].buf(), 1, 1, 1, Talpha);

            if(arg.scaleE)
                hipblaslt_init(hScaleE[i].buf(), 1, 1, 1, Talpha);

            if(arg.scaleAlpha_vector)
                hipblaslt_init(hScaleAlphaVec[i].buf(), M[i], 1, M[i], Talpha);

            if(arg.gradient && arg.use_e)
            {
                CHECK_HIP_ERROR(synchronize(dE[i], hE[i], block_count));
            }
            if(!arg.gradient && arg.bias_vector)
            {
                CHECK_HIP_ERROR(synchronize(dBias[i], hBias[i], block_count));
            }

            if(arg.scaleAlpha_vector)
            {
                CHECK_HIP_ERROR(synchronize(dScaleAlphaVec[i], hScaleAlphaVec[i], block_count));
                alpha_in[i] = dScaleAlphaVec[i].buf();
                set_computeInterface(
                    h_alpha[i],
                    1.0,
                    Tc, 
                    TiA); // use dScaleAlphaVec instead, original alpha = 1.0 for verify
            }
            else
                alpha_in[i] = &(h_alpha[i]);

            if(arg.scaleA == hipblaslt_scaling_format::Scalar
               || arg.scaleA == hipblaslt_scaling_format::Vector)
            {
                if(arg.amaxScaleA && (arg.a_type == HIP_R_32F || arg.a_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.a_type,
                                                           HIP_R_32F,
                                                           dScaleA[i].buf(),
                                                           dA[i].buf(),
                                                           A_row[i],
                                                           A_col[i],
                                                           stream));

                    CHECK_HIP_ERROR(synchronize(hScaleA[i], dScaleA[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar
               || arg.scaleB == hipblaslt_scaling_format::Vector)
            {
                if(arg.amaxScaleB && (arg.b_type == HIP_R_32F || arg.b_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.b_type,
                                                           HIP_R_32F,
                                                           dScaleB[i].buf(),
                                                           dB[i].buf(),
                                                           B_row[i],
                                                           B_col[i],
                                                           stream));
                    CHECK_HIP_ERROR(synchronize(hScaleB[i], dScaleB[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
            }

            if(arg.scaleC)
                CHECK_HIP_ERROR(synchronize(dScaleC[i], hScaleC[i]));

            if(arg.scaleD)
                CHECK_HIP_ERROR(synchronize(dScaleD[i], hScaleD[i]));

            if(arg.scaleE)
                CHECK_HIP_ERROR(synchronize(dScaleE[i], hScaleE[i]));

            if(epilogue_on[i])
            {
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &(epilogue[i]),
                                                    sizeof(epilogue[i])),
                    HIPBLAS_STATUS_SUCCESS);
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT,
                                                    &(act0[i]),
                                                    sizeof(act0[i])));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT,
                                                    &(act1[i]),
                                                    sizeof(act1[i])));
            }

            if(arg.use_e)
            {
                void* e_addr = dE[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                    &e_addr,
                                                    sizeof(void*)));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                                                    &arg.aux_type,
                                                    sizeof(hipDataType)));
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &lde[i], sizeof(int64_t)));
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                                    &stride_e[i],
                                                    sizeof(int64_t)));
            }

            if(arg.bias_vector)
            {
                const void* bias_addr;
                int32_t bias_stride = arg.bias_stride;
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &arg.bias_type,
                                                    sizeof(hipDataType)),
                    HIPBLAS_STATUS_SUCCESS);
                bias_addr = dBias[i].buf();

                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias_addr,
                                                    sizeof(void*)),
                    HIPBLAS_STATUS_SUCCESS);
                
                if(bias_stride > 0)
                    EXPECT_HIPBLAS_STATUS(
                        hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                        HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                                                        &bias_stride,
                                                        sizeof(bias_stride)),
                        HIPBLAS_STATUS_SUCCESS);
            }

            if(arg.scaleA != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;

                void* scaleA_addr = (void*)(dScaleA[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], attr, &scaleA_addr, sizeof(void*)));

                hipblasLtMatmulMatrixScale_t mode = matmulScaleMode(arg.scaleA);

                if(mode != HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
                {
                    auto attr = HIPBLASLT_MATMUL_DESC_A_SCALE_MODE;
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[0][i], attr, &mode, sizeof(uint32_t)));
                }
            }

            if(arg.scaleB != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;

                void* scaleB_addr = (void*)(dScaleB[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], attr, &scaleB_addr, sizeof(void*)));

                hipblasLtMatmulMatrixScale_t mode = matmulScaleMode(arg.scaleB);

                if(mode != HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F)
                {
                    auto attr = HIPBLASLT_MATMUL_DESC_B_SCALE_MODE;
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[0][i], attr, &mode, sizeof(uint32_t)));
                }
            }

            if(arg.scaleC)
            {
                void* scaleC_addr = dScaleC[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                                    &scaleC_addr,
                                                    sizeof(void*)));
            }

            if(arg.scaleD)
            {
                void* scaleD_addr = dScaleD[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                    &scaleD_addr,
                                                    sizeof(void*)));
            }

            if(arg.amaxD)
            {
                void* amaxD_addr = dAmaxD[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                    &amaxD_addr,
                                                    sizeof(void*)));
            }

            if(arg.scaleE)
            {
                void* scaleE_addr = dScaleE[i].buf();
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i],
                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                    &scaleE_addr,
                    sizeof(void*)));
            }

            if(arg.scaleAlpha_vector)
            {
                hipblasLtPointerMode_t scale_mode
                    = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                    &scale_mode,
                                                    sizeof(scale_mode)),
                    HIPBLAS_STATUS_SUCCESS);
            }
        }
        else
        {
            alpha_in[i] = &(h_alpha[i]);
            for(int batchCount = 0; batchCount < num_batches[i]; batchCount++)
            {
                hipblaslt_init_device(ABC_dims::C,
                                      arg.initialization,
                                      beta_isnan_type(arg, Talpha),
                                      dC[batchCount].buf(),
                                      M[i],
                                      N[i],
                                      ldc[i],
                                      To,
                                      stride_c[i],
                                      1,
                                      positiveOnlyInitialization);
                // broadcast first block
                CHECK_HIP_ERROR(broadcast(dA[batchCount], block_count));
                CHECK_HIP_ERROR(broadcast(dB[batchCount], block_count));
                CHECK_HIP_ERROR(broadcast(dC[batchCount], block_count));

                if(arg.unit_check || arg.norm_check || arg.allclose_check || do_swizzle_a
                   || do_swizzle_b)
                {
                    CHECK_HIP_ERROR(synchronize(hA[batchCount],
                                                dA[batchCount],
                                                1,
                                                A_row[i],
                                                A_col[i],
                                                lda[i],
                                                realDataTypeSize(TiA),
                                                do_swizzle_a));
                    CHECK_HIP_ERROR(synchronize(hB[batchCount],
                                                dB[batchCount],
                                                1,
                                                B_row[i],
                                                B_col[i],
                                                ldb[i],
                                                realDataTypeSize(TiB),
                                                do_swizzle_b));
                    CHECK_HIP_ERROR(synchronize(hC[batchCount], dC[batchCount]));
                }
                if(arg.dump_matrix)
                {
                    hipblasltDispatchValuesToFile(transA,
                                                  TiA,
                                                  M[i],
                                                  K[i],
                                                  lda[i],
                                                  hA[batchCount].buf(),
                                                  "batch_" + std::to_string(i) + "_A_"
                                                      + std::to_string(batchCount) + "_input.txt");
                    hipblasltDispatchValuesToFile(transB,
                                                  TiB,
                                                  K[i],
                                                  N[i],
                                                  ldb[i],
                                                  hB[batchCount].buf(),
                                                  "batch_" + std::to_string(i) + "_B_"
                                                      + std::to_string(batchCount) + "_input.txt");
                    hipblasltDispatchValuesToFile(HIPBLAS_OP_N,
                                                  To,
                                                  M[i],
                                                  N[i],
                                                  ldc[i],
                                                  hC[batchCount].buf(),
                                                  "batch_" + std::to_string(i) + "_C_"
                                                      + std::to_string(batchCount) + "_input.txt");
                }
                if(do_swizzle_a)
                {
                    HipHostBuffer tmp(TiA, size_dA[i]);
                    swizzle_tensor_type(
                        tmp, hA[batchCount], TiA, arg, 1, M[i], K[i], lda[i], false);
                    CHECK_HIP_ERROR(synchronize(dA[batchCount], tmp, block_count));
                }

                if(do_swizzle_b)
                {
                    HipHostBuffer tmp(TiB, size_dB[i]);
                    swizzle_tensor_type(
                        tmp, hB[batchCount], TiB, arg, 1, N[i], K[i], ldb[i], false);
                    CHECK_HIP_ERROR(synchronize(dB[batchCount], tmp, block_count));
                }
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            {
                if(arg.norm_check)
                    hipblaslt_init_small(
                        hScaleA[i].buf(), size_scaleAVec[i], 1, size_scaleAVec[i], Talpha);
                else
                    hipblaslt_init(
                        hScaleA[i].buf(), size_scaleAVec[i], 1, size_scaleAVec[i], Talpha);
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            {
                if(arg.norm_check)
                    hipblaslt_init_small(
                        hScaleB[i].buf(), size_scaleBVec[i], 1, size_scaleBVec[i], Talpha);
                else
                    hipblaslt_init(
                        hScaleB[i].buf(), size_scaleBVec[i], 1, size_scaleBVec[i], Talpha);
            }
            if(arg.scaleC)
            {
                if(To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ)
                {
                    hipblaslt_init_small(hScaleC[i].buf(), 1, 1, 1, Talpha);
                }
                else
                {
                    hipblaslt_init(hScaleC[i].buf(), 1, 1, 1, Talpha);
                }
            }

            if(arg.scaleD)
            {
                if(To == HIP_R_8F_E4M3_FNUZ || To == HIP_R_8F_E5M2_FNUZ)
                {
                    hipblaslt_init_small(hScaleD[i].buf(), 1, 1, 1, Talpha);
                }
                else
                {
                    hipblaslt_init(hScaleD[i].buf(), 1, 1, 1, Talpha);
                }
            }
            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            {
                if(arg.amaxScaleA && (arg.a_type == HIP_R_32F || arg.a_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.a_type,
                                                           HIP_R_32F,
                                                           dScaleA[i].buf(),
                                                           dA[i].buf(),
                                                           A_row[i],
                                                           A_col[i],
                                                           stream));

                    CHECK_HIP_ERROR(synchronize(hScaleA[i], dScaleA[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleA[i], hScaleA[i], block_count));
            }

            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            {
                if(arg.amaxScaleB && (arg.b_type == HIP_R_32F || arg.b_type == HIP_R_16F))
                {
                    CHECK_HIPBLASLT_ERROR(hipblasltExtAMax(arg.b_type,
                                                           HIP_R_32F,
                                                           dScaleB[i].buf(),
                                                           dB[i].buf(),
                                                           B_row[i],
                                                           B_col[i],
                                                           stream));
                    CHECK_HIP_ERROR(synchronize(hScaleB[i], dScaleB[i]));
                }
                else
                    CHECK_HIP_ERROR(synchronize(dScaleB[i], hScaleB[i], block_count));
            }

            if(arg.scaleC)
                CHECK_HIP_ERROR(synchronize(dScaleC[i], hScaleC[i]));
            if(arg.scaleD)
                CHECK_HIP_ERROR(synchronize(dScaleD[i], hScaleD[i]));

            if(arg.scaleA == hipblaslt_scaling_format::Scalar)
            {
                hipblasLtMatmulDescAttributes_t attr        = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
                void*                           scaleA_addr = (void*)(dScaleA[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], attr, &scaleA_addr, sizeof(void*)));
                hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                attr                              = HIPBLASLT_MATMUL_DESC_A_SCALE_MODE;
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i], attr, &mode, sizeof(uint32_t)));
            }
            if(arg.scaleB == hipblaslt_scaling_format::Scalar)
            {
                hipblasLtMatmulDescAttributes_t attr        = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
                void*                           scaleB_addr = (void*)(dScaleB[i].buf());
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], attr, &scaleB_addr, sizeof(void*)));
                hipblasLtMatmulMatrixScale_t mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
                attr                              = HIPBLASLT_MATMUL_DESC_B_SCALE_MODE;
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i], attr, &mode, sizeof(uint32_t)));
            }
            if(arg.scaleC)
            {
                void* scaleC_addr = dScaleC[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                                    &scaleC_addr,
                                                    sizeof(void*)));
            }
            if(arg.scaleD)
            {
                void* scaleD_addr = dScaleD[i].buf();
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                    &scaleD_addr,
                                                    sizeof(void*)));
            }
        }
        for(int32_t b = 1; b < matmul.size(); b++)
        {
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescCreate(&(matmul[b][i]), arg.compute_type, arg.scale_type));
            CHECK_HIPBLASLT_ERROR(hipblaslt_ext::copyMatmul(matmul[0][i], matmul[b][i]));

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                                &TciA,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                                &TciB,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            // Forward CLI knobs from hipblaslt-bench into the matmul descriptor.
            {
                int32_t sm = hipblaslt_bench_options::sm_count_target();
                if(sm != 0)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[b][i],
                        HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                        &sm,
                        sizeof(sm)));
                }
                int32_t dyn = hipblaslt_bench_options::streamk_tile_scheduling_mode();
                if(dyn >= 0)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[b][i],
                        HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
                        &dyn,
                        sizeof(dyn)));
                }
                int32_t uso = hipblaslt_bench_options::uniform_summation_order();
                if(uso >= 0)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[b][i],
                        HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT,
                        &uso,
                        sizeof(uso)));
                }
            }

            if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
            {
                // Update bias, E
                if(arg.bias_vector)
                {
                    const void* bias_addr
                        = (const void*)(dBias[i].as<char>()
                                        + b * size_bias[i] * realDataTypeSize(Tbias));
                    EXPECT_HIPBLAS_STATUS(
                        hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &bias_addr,
                                                        sizeof(void*)),
                        HIPBLAS_STATUS_SUCCESS);
                }
                if(arg.use_e)
                {
                    void* e_addr
                        = (void*)(dE[i].as<char>() + b * size_E[i] * realDataTypeSize(Taux));
                    CHECK_HIPBLASLT_ERROR(
                        hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                        &e_addr,
                                                        sizeof(void*)));
                }
            }
            if(arg.scaleA != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
                void* scaleA_addr = (void*)(dScaleA[i].as<char>() + b * size_scaleAVec[i]);
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[b][i], attr, &scaleA_addr, sizeof(void*)));
            }

            if(arg.scaleB != hipblaslt_scaling_format::none)
            {
                hipblasLtMatmulDescAttributes_t attr = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
                void* scaleB_addr = (void*)(dScaleB[i].as<char>() + b * size_scaleBVec[i]);
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[b][i], attr, &scaleB_addr, sizeof(void*)));
            }
        }
    }

    // set preference
    size_t                     max_workspace_size = arg.user_allocated_workspace;
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)),
        HIPBLAS_STATUS_SUCCESS);

    // set workspace
    device_vector<unsigned char>* dWorkspace     = nullptr;
    size_t                        workspace_size = 0;

    // set user args
    hipblaslt_ext::UserArguments* userArgs   = nullptr;
    hipblaslt_ext::UserArguments* d_userArgs = nullptr;

    // Get Heuristic results
    int32_t requestAlgoCount = arg.requested_solution_num < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM
                                                              : arg.requested_solution_num;
    int     returnedAlgoCount = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    std::vector<size_t>                           heuristicTuningIndex;

    // Cpp API
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    std::vector<hipblaslt_ext::Gemm>                    gemmVec;
    std::vector<hipblaslt_ext::GroupedGemm>             groupedGemmVec;
    std::vector<std::vector<hipblaslt_ext::GemmInputs>> extinputs;

    // Pointer-array GEMM has one logical problem with one pointer binding per batch.
    const int32_t binding_count = batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY
                                      ? arg.batch_count
                                      : problem_count;
    // C to Cpp API for GG
    const auto groupedGemmBlockCount = do_grouped_gemm ? block_count : 0;
    std::vector<std::vector<void*>> da(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<std::vector<void*>> db(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<std::vector<void*>> dc(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<std::vector<void*>> dd(groupedGemmBlockCount,
                                       std::vector<void*>(problem_count));
    std::vector<uint64_t*> dda, ddb, ddc, ddd;
    if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        for(int i = 0; i < block_count; i++)
        {
            uint64_t* ptr = nullptr;
            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            dda.push_back(ptr);

            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            ddb.push_back(ptr);

            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            ddc.push_back(ptr);

            CHECK_HIP_ERROR(hipMalloc(&ptr, binding_count * sizeof(uint64_t*)));
            ddd.push_back(ptr);
        }
    }

    for(int32_t b = 0; b < block_count; b++)
    {
        if(!do_grouped_gemm)
            gemmVec.push_back(hipblaslt_ext::Gemm(handle,
                                                  transA,
                                                  transB,
                                                  arg.a_type,
                                                  arg.b_type,
                                                  arg.c_type,
                                                  arg.d_type,
                                                  arg.compute_type));
        else
            groupedGemmVec.push_back(hipblaslt_ext::GroupedGemm(handle,
                                                                transA,
                                                                transB,
                                                                arg.a_type,
                                                                arg.b_type,
                                                                arg.c_type,
                                                                arg.d_type,
                                                                arg.compute_type));
    }

    std::vector<hipblaslt_ext::GemmEpilogue> extepilogue;
    hipblaslt_ext::GemmProblemType           extproblemtype;
    if(arg.use_ext_setproblem && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        extinputs.resize(block_count, std::vector<hipblaslt_ext::GemmInputs>(problem_count));
        extepilogue.resize(problem_count);

        for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
        {
            auto  bias_type = HIPBLASLT_DATATYPE_INVALID;
            auto  aux_type  = HIPBLASLT_DATATYPE_INVALID;
            void* bias_addr = nullptr;
            for(int32_t b = 0; b < block_count; b++)
            {
                if(arg.bias_vector)
                {
                    bias_type = arg.bias_type;
                    bias_addr = (void*)(dBias[gemmIdx].as<char>()
                                        + b * size_bias[gemmIdx] * realDataTypeSize(bias_type));
                }
                if(arg.use_e)
                {
                    aux_type = arg.aux_type;
                }
                if(b == 0)
                {
                    extepilogue[gemmIdx].setMode(epilogue[gemmIdx]);
                    extepilogue[gemmIdx].setBiasDataType(bias_type);
                    extepilogue[gemmIdx].setAuxDataType(aux_type);
                    extepilogue[gemmIdx].setAuxLeadingDimension(lde[gemmIdx]);
                    extepilogue[gemmIdx].setAuxBatchStride(stride_e[gemmIdx]);
                    extepilogue[gemmIdx].setScalingAType(matmulScaleMode(arg.scaleA));
                    extepilogue[gemmIdx].setScalingBType(matmulScaleMode(arg.scaleB));
                }
                extinputs[b][gemmIdx].setA((void*)((dA[gemmIdx].as<char>())
                                                   + b * size_dA[gemmIdx] * realDataTypeSize(TiA)));
                extinputs[b][gemmIdx].setB((void*)((dB[gemmIdx].as<char>())
                                                   + b * size_dB[gemmIdx] * realDataTypeSize(TiB)));
                extinputs[b][gemmIdx].setC(
                    (void*)((dC[gemmIdx].as<char>()) + b * size_C[gemmIdx] * realDataTypeSize(To)));
                extinputs[b][gemmIdx].setD((void*)(((*dDp)[gemmIdx].as<char>())
                                                   + b * size_D[gemmIdx] * realDataTypeSize(To)));
                extinputs[b][gemmIdx].setAlpha(&h_alpha[gemmIdx]);
                extinputs[b][gemmIdx].setBeta(&h_beta[gemmIdx]);
                extinputs[b][gemmIdx].setBias(bias_addr);
                extinputs[b][gemmIdx].setScaleA(
                    arg.scaleA != hipblaslt_scaling_format::none
                        ? (void*)((dScaleA[gemmIdx].as<char>()) + b * size_scaleAVec[gemmIdx])
                        : nullptr);
                extinputs[b][gemmIdx].setScaleB(
                    arg.scaleB != hipblaslt_scaling_format::none
                        ? (void*)((dScaleB[gemmIdx].as<char>()) + b * size_scaleBVec[gemmIdx])
                        : nullptr);
                extinputs[b][gemmIdx].setScaleC(arg.scaleC ? dScaleC[gemmIdx].as<char>() : nullptr);
                extinputs[b][gemmIdx].setScaleD(arg.scaleD ? dScaleD[gemmIdx].as<char>() : nullptr);
                extinputs[b][gemmIdx].setScaleAux(arg.scaleE ? dScaleE[gemmIdx].as<char>()
                                                             : nullptr);
                extinputs[b][gemmIdx].setAmaxD(arg.amaxD ? dAmaxD[gemmIdx].as<char>() : nullptr);
                if(arg.use_e)
                    extinputs[b][gemmIdx].setAux(
                        (void*)((dE[gemmIdx].as<char>())
                                + b * size_E[gemmIdx] * realDataTypeSize(Taux)));
                if(arg.scaleAlpha_vector)
                    extinputs[b][gemmIdx].setScaleAlphaVec(
                        (void*)((dScaleAlphaVec[gemmIdx].as<char>())
                                + b * size_scaleAlphaVec[gemmIdx] * realDataTypeSize(Talpha)));
            }
        }
        extproblemtype.setOpA(transA);
        extproblemtype.setOpB(transB);
        extproblemtype.setTypeA(arg.a_type);
        extproblemtype.setTypeB(arg.b_type);
        extproblemtype.setTypeC(arg.c_type);
        extproblemtype.setTypeD(arg.d_type);
        extproblemtype.setTypeCompute(arg.compute_type);

        if(do_swizzle_a)
        {
            hipblasLtOrder_t orderA = orderForDatatype(TiA);
            extproblemtype.setOrderA(orderA);
        }
        if(do_swizzle_b)
        {
            hipblasLtOrder_t orderB = orderForDatatype(TiB);
            extproblemtype.setOrderB(orderB);
        }
    }
    else if(arg.grouped_gemm)
    {
        for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
        {
            for(int32_t b = 0; b < block_count; b++)
            {
                da[b][gemmIdx] = (void*)((dA[gemmIdx].as<char>())
                                         + b * size_dA[gemmIdx] * realDataTypeSize(TiA));
                db[b][gemmIdx] = (void*)((dB[gemmIdx].as<char>())
                                         + b * size_dB[gemmIdx] * realDataTypeSize(TiB));
                dc[b][gemmIdx] = (void*)((dC[gemmIdx].as<char>())
                                         + b * size_C[gemmIdx] * realDataTypeSize(To));
                dd[b][gemmIdx] = (void*)(((*dDp)[gemmIdx].as<char>())
                                         + b * size_D[gemmIdx] * realDataTypeSize(To));
            }
        }
    }
    else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        std::vector<uint64_t*> da1(binding_count), db1(binding_count), dc1(binding_count),
            dd1(binding_count);
        for(int32_t b = 0; b < block_count; b++)
        {
            for(int gemmIdx = 0; gemmIdx < binding_count; gemmIdx++)
            {
                da1[gemmIdx] = reinterpret_cast<uint64_t*>(
                    (dA[gemmIdx].as<char>()) + b * size_dA[0] * realDataTypeSize(TiA));
                db1[gemmIdx] = reinterpret_cast<uint64_t*>(
                    (dB[gemmIdx].as<char>()) + b * size_dB[0] * realDataTypeSize(TiB));
                dc1[gemmIdx] = reinterpret_cast<uint64_t*>(
                    (dC[gemmIdx].as<char>()) + b * size_C[0] * realDataTypeSize(To));
                dd1[gemmIdx] = reinterpret_cast<uint64_t*>(
                    (*dDp)[gemmIdx].as<char>() + b * size_D[0] * realDataTypeSize(To));
            }
            CHECK_HIP_ERROR(hipMemcpy(
                dda[b], da1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                ddb[b], db1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                ddc[b], dc1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                ddd[b], dd1.data(), binding_count * sizeof(uint64_t*), hipMemcpyHostToDevice));
        }
    }

    hipblaslt_ext::GemmType gemmType = do_grouped_gemm
                                           ? hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM
                                           : hipblaslt_ext::GemmType::HIPBLASLT_GEMM;

    // Remove duplicate
    std::vector<uint32_t> gsu_vector;
    std::vector<uint32_t> wgm_vector;
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.gsu_vector[i] == -1)
            break;
        gsu_vector.push_back(arg.gsu_vector[i]);
    }
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.wgm_vector[i] == -1)
            break;
        wgm_vector.push_back(arg.wgm_vector[i]);
    }
    std::set<uint32_t> remove_duplicate(gsu_vector.begin(), gsu_vector.end());
    gsu_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    remove_duplicate = std::set<uint32_t>(wgm_vector.begin(), wgm_vector.end());
    wgm_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    std::vector<hipblaslt_ext::GemmTuning> tuningVec;
    if(arg.use_ext)
    {
        for(size_t wgm = 0; wgm < wgm_vector.size(); wgm++)
            for(size_t gsu = 0; gsu < gsu_vector.size(); gsu++)
            {
                hipblaslt_ext::GemmTuning tuning;
                tuning.setSplitK(gsu_vector[gsu]);
                tuning.setWgm(wgm_vector[wgm]);
                tuningVec.push_back(tuning);
            }
    }
    else
    {
        // C API does not support
        tuningVec.push_back(hipblaslt_ext::GemmTuning());
    }

    hipblaslt_ext::GemmInstance* extProblem = nullptr;
    if(do_grouped_gemm)
    {
        if(arg.use_ext_setproblem)
        {
            auto batchCounts = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
            for(int32_t block = 0; block < block_count; ++block)
            {
                CHECK_HIPBLASLT_ERROR(groupedGemmVec[block].setProblem(M,
                                                                       N,
                                                                       K,
                                                                       batchCounts,
                                                                       lda,
                                                                       ldb,
                                                                       ldc,
                                                                       ldd,
                                                                       stride_da,
                                                                       stride_db,
                                                                       stride_c,
                                                                       stride_d,
                                                                       extepilogue,
                                                                       extinputs[block],
                                                                       extproblemtype));
            }
        }
        else
        {
            std::vector<void*> alphaPointers;
            std::vector<void*> betaPointers;
            alphaPointers.reserve(h_alpha.size());
            betaPointers.reserve(h_beta.size());
            for(size_t i = 0; i < h_alpha.size(); ++i)
            {
                alphaPointers.push_back(&h_alpha[i]);
                betaPointers.push_back(&h_beta[i]);
            }
            for(int32_t block = 0; block < block_count; ++block)
            {
                CHECK_HIPBLASLT_ERROR(groupedGemmVec[block].setProblem(matmul[block],
                                                                       alphaPointers,
                                                                       da[block],
                                                                       matA,
                                                                       db[block],
                                                                       matB,
                                                                       betaPointers,
                                                                       dc[block],
                                                                       matC,
                                                                       dd[block],
                                                                       matD));
            }
        }
        extProblem = &groupedGemmVec.front();
    }
    else if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        for(int32_t block = 0; block < block_count; ++block)
        {
            if(arg.use_ext_setproblem)
            {
                CHECK_HIPBLASLT_ERROR(gemmVec[block].setProblem(M[0],
                                                                N[0],
                                                                K[0],
                                                                num_batches[0],
                                                                lda[0],
                                                                ldb[0],
                                                                ldc[0],
                                                                ldd[0],
                                                                stride_da[0],
                                                                stride_db[0],
                                                                stride_c[0],
                                                                stride_d[0],
                                                                extepilogue[0],
                                                                extinputs[block][0],
                                                                extproblemtype));
            }
            else
            {
                CHECK_HIPBLASLT_ERROR(gemmVec[block].setProblem(
                    matmul[block][0],
                    alpha_in[0],
                    dA[0].as<char>() + block * size_dA[0] * realDataTypeSize(TiA),
                    matA[0],
                    dB[0].as<char>() + block * size_dB[0] * realDataTypeSize(TiB),
                    matB[0],
                    &h_beta[0],
                    dC[0].as<char>() + block * size_C[0] * realDataTypeSize(To),
                    matC[0],
                    (*dDp)[0].as<char>() + block * size_D[0] * realDataTypeSize(To),
                    matD[0]));
            }
        }
        extProblem = &gemmVec.front();
    }

    auto collectSupportedAlgorithms
        = [&](std::span<hipblasLtMatmulHeuristicResult_t> candidates,
              size_t                                      maximumAlgorithms) {
              size_t acceptedAlgorithms = 0;
              for(auto& candidate : candidates)
              {
                  bool accepted = false;
                  if(extProblem != nullptr)
                  {
                      for(size_t tuningIndex = 0; tuningIndex < tuningVec.size(); ++tuningIndex)
                      {
                          size_t requiredWorkspace = 0;
                          if(extProblem->isAlgoSupported(
                                 candidate.algo, tuningVec[tuningIndex], requiredWorkspace)
                             != HIPBLAS_STATUS_SUCCESS
                             || requiredWorkspace > max_workspace_size)
                              continue;
                          heuristicResult.push_back(candidate);
                          heuristicTuningIndex.push_back(tuningIndex);
                          workspace_size = std::max(workspace_size, requiredWorkspace);
                          accepted      = true;
                      }
                  }
                  else
                  {
                      size_t requiredWorkspace               = 0;
                      candidate.algo.max_workspace_bytes     = max_workspace_size;
                      const hipblasStatus_t supportStatus
                          = hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                 matmul[0][0],
                                                                 alpha_in[0],
                                                                 matA[0],
                                                                 matB[0],
                                                                 &h_beta[0],
                                                                 matC[0],
                                                                 matD[0],
                                                                 candidate.algo,
                                                                 requiredWorkspace);
                      if(supportStatus == HIPBLAS_STATUS_SUCCESS
                         && requiredWorkspace <= max_workspace_size)
                      {
                          heuristicResult.push_back(candidate);
                          heuristicTuningIndex.push_back(0);
                          workspace_size = std::max(workspace_size, requiredWorkspace);
                          accepted      = true;
                      }
                  }
                  CHECK_RETURNED_WORKSPACE_SIZE(workspace_size, max_workspace_size);
                  acceptedAlgorithms += accepted;
                  if(acceptedAlgorithms >= maximumAlgorithms)
                      break;
              }
          };

    if(arg.algo_method == 2)
    {
        heuristicResult.clear();
        heuristicTuningIndex.clear();

        const bool indicesAreDiscovered = (arg.solution_index == -1);
        std::vector<int> indices;
        if(indicesAreDiscovered)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> allAlgos;
            EXPECT_HIPBLAS_STATUS(hipblaslt_ext::getAllAlgos(handle,
                                                             gemmType,
                                                             transA,
                                                             transB,
                                                             arg.a_type,
                                                             arg.b_type,
                                                             arg.c_type,
                                                             arg.d_type,
                                                             arg.compute_type,
                                                             allAlgos),
                                  HIPBLAS_STATUS_SUCCESS);
            indices.reserve(allAlgos.size());
            for(auto& result : allAlgos)
                indices.push_back(hipblaslt_ext::getIndexFromAlgo(result.algo));
        }
        else
        {
            indices.push_back(arg.solution_index);
        }

        constexpr size_t batchSize = 100;
        for(size_t batchStart = 0; batchStart < indices.size(); batchStart += batchSize)
        {
            const size_t batchEnd = std::min(batchStart + batchSize, indices.size());
            std::vector<int> batch(indices.begin() + batchStart, indices.begin() + batchEnd);
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
            const auto status = hipblaslt_ext::getAlgosFromIndex(handle, batch, candidates);
            if(indicesAreDiscovered)
                EXPECT_HIPBLAS_STATUS(status, HIPBLAS_STATUS_SUCCESS);
            if(candidates.empty())
                break;

            const size_t previousResultCount = heuristicResult.size();
            const size_t maximumAlgorithms
                = indicesAreDiscovered && extProblem == nullptr ? candidates.size() : 1;
            collectSupportedAlgorithms(candidates, maximumAlgorithms);
            if(heuristicResult.size() != previousResultCount)
                break;
        }
    }
    else if(arg.algo_method == 1)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        EXPECT_HIPBLAS_STATUS(hipblaslt_ext::getAllAlgos(handle,
                                                         gemmType,
                                                         transA,
                                                         transB,
                                                         arg.a_type,
                                                         arg.b_type,
                                                         arg.c_type,
                                                         arg.d_type,
                                                         arg.compute_type,
                                                         tmpAlgo),
                              HIPBLAS_STATUS_SUCCESS);
        heuristicResult.clear();
        heuristicTuningIndex.clear();
        collectSupportedAlgorithms(tmpAlgo, requestAlgoCount);
    }
    else
    {
        heuristicResult.clear();
        heuristicTuningIndex.clear();
        if(extProblem != nullptr)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
            CHECK_HIPBLASLT_ERROR(
                extProblem->algoGetHeuristic(requestAlgoCount, gemmPref, candidates));
            collectSupportedAlgorithms(candidates, std::numeric_limits<size_t>::max());
        }
        else
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates(requestAlgoCount);
            EXPECT_HIPBLAS_STATUS((hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                   matmul[0][0],
                                                                   matA[0],
                                                                   matB[0],
                                                                   matC[0],
                                                                   matD[0],
                                                                   pref,
                                                                   requestAlgoCount,
                                                                   candidates.data(),
                                                                   &returnedAlgoCount)),
                                  HIPBLAS_STATUS_SUCCESS);
            candidates.resize(returnedAlgoCount);
            for(auto& candidate : candidates)
            {
                heuristicResult.push_back(candidate);
                heuristicTuningIndex.push_back(0);
                workspace_size = std::max(workspace_size, candidate.workspaceSize);
            }
            CHECK_RETURNED_WORKSPACE_SIZE(workspace_size, max_workspace_size);
        }
    }

    returnedAlgoCount = heuristicResult.size();

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    dWorkspace = new device_vector<unsigned char>(workspace_size * block_count, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());

    if(arg.use_user_args)
    {
        CHECK_HIP_ERROR(
            hipHostMalloc(&userArgs, problem_count * sizeof(hipblaslt_ext::UserArguments)));
        CHECK_HIP_ERROR(hipMalloc(&d_userArgs,
                                  block_count * problem_count * sizeof(hipblaslt_ext::UserArguments)));
    }

    auto ptrs = benchmark_allocation();

    if(arg.adaptive)
    {
        hipblaslt_bench::TimingConfig cfg;
        cfg.warmup_time         = arg.warmup_time;
        cfg.sample_time         = arg.sample_time;
        cfg.measure_time        = arg.measure_time;
        cfg.max_measure_time    = arg.max_measure_time;
        cfg.min_iters           = arg.min_iters;
        cfg.max_iters           = arg.max_iters;
        cfg.noise_threshold     = arg.noise_threshold;
        cfg.stability_threshold = arg.stability_threshold;
        cfg.stability_window    = arg.stability_window;
        cfg.stability_interval  = arg.stability_interval;
        hipblaslt_cout << hipblaslt_bench::format_adaptive_timing_summary(cfg) << std::endl;
    }

    if(arg.print_solution_found)
        hipblaslt_cout << "Is supported " << heuristicResult.size()
                       << " / Total solutions: " << returnedAlgoCount * tuningVec.size()
                       << std::endl;

    if(heuristicResult.size() != heuristicTuningIndex.size())
    {
        hipblaslt_cerr << "Internal error, heuristicResult.size() != heuristicTuningIndex.size() "
                       << heuristicResult.size() << " != " << heuristicTuningIndex.size()
                       << std::endl;
        exit(EXIT_FAILURE);
    }

    // get CPU result
    if(arg.unit_check || arg.norm_check || arg.allclose_check)
    {
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
        {
            auto                 alpha    = h_alpha[gemmIdx];
            auto                 betaTemp = h_beta[gemmIdx];
            computeTypeInterface tempSC{};
            if(arg.scaleC)
            {
                // betaTemp *= hScaleC[gemmIdx][0];
                set_computeInterface(tempSC, hScaleC[gemmIdx].buf(), Tc, TiA);
                mul_computeInterface(betaTemp, tempSC, Tc, TiA);
            }

            computeTypeInterface scale{};
            set_computeInterface(scale, 1, Talpha, TiA);
            void* scaleAVec   = (arg.scaleA == hipblaslt_scaling_format::Scalar
                               || arg.scaleA == hipblaslt_scaling_format::Vector)
                                    ? hScaleA[gemmIdx].buf()
                                    : (void*)(&scale);
            void* scaleBVec   = (arg.scaleB == hipblaslt_scaling_format::Scalar
                               || arg.scaleB == hipblaslt_scaling_format::Vector)
                                    ? hScaleB[gemmIdx].buf()
                                    : (void*)(&scale);
            void* scaleDValue = arg.scaleD ? hScaleD[gemmIdx].buf() : (void*)(&scale);
            void* scaleEValue = arg.scaleE ? hScaleE[gemmIdx].buf() : (void*)(&scale);

            bool const isScaleAMXFormat = isBlockScaling(arg.scaleA);
            bool const isScaleBMXFormat = isBlockScaling(arg.scaleB);

            for(int batchIdx = 0; batchIdx < num_batches[gemmIdx]; batchIdx++)
            {
                if(epilogue_on[gemmIdx])
                {
                    // Note: for MX types, pass the reference float instead so there is
                    //       no need to convert them to float in hipblaslt_reference_gemm
                    hipblaslt_reference_gemm(
                        transA,
                        transB,
                        M[gemmIdx],
                        N[gemmIdx],
                        K[gemmIdx],
                        alpha,
                        isScaleAMXFormat
                            ? reinterpret_cast<char*>(refA[gemmIdx].data())
                                  + stride_a[gemmIdx] * batchIdx * realDataTypeSize(HIP_R_32F)
                            : hA[gemmIdx].as<char>()
                                  + stride_a[gemmIdx] * batchIdx * realDataTypeSize(TiA),
                        lda[gemmIdx],
                        isScaleBMXFormat
                            ? reinterpret_cast<char*>(refB[gemmIdx].data())
                                  + stride_b[gemmIdx] * batchIdx * realDataTypeSize(HIP_R_32F)
                            : hB[gemmIdx].as<char>()
                                  + stride_b[gemmIdx] * batchIdx * realDataTypeSize(TiB),
                        ldb[gemmIdx],
                        betaTemp,
                        hC[gemmIdx].as<char>()
                            + stride_c[gemmIdx] * batchIdx * realDataTypeSize(To),
                        ldc[gemmIdx],
                        hD_gold_epl[gemmIdx].as<char>()
                            + stride_d[gemmIdx] * batchIdx * realDataTypeSize(Talpha),
                        ldd[gemmIdx],
                        arg.scaleAlpha_vector ? hScaleAlphaVec[gemmIdx].as<char>() + 0 : nullptr,
                        scaleAVec,
                        scaleBVec,
                        (void*)(&scale),
                        (arg.scaleA == hipblaslt_scaling_format::Vector),
                        (arg.scaleB == hipblaslt_scaling_format::Vector),
                        isScaleAMXFormat ? HIP_R_32F : TiA,
                        isScaleBMXFormat ? HIP_R_32F : TiB,
                        To,
                        Talpha,
                        Tc,
                        isScaleAMXFormat ? HIP_R_32F : TciA,
                        isScaleBMXFormat ? HIP_R_32F : TciB,
                        isBlockScaling(arg.scaleA),
                        isBlockScaling(arg.scaleB));

                    auto                        pos    = stride_d[gemmIdx] * batchIdx;
                    std::vector<HipHostBuffer>* hEInst = arg.gradient ? &hE : &hE_gold;
                    void*                       ePos
                        = ((*hEInst).size() <= gemmIdx)
                              ? nullptr
                              : ((*hEInst)[gemmIdx].as<char>() + pos * realDataTypeSize(Taux));
                    auto  applyBias = arg.gradient ? false : arg.bias_vector;
                    void* hBias_buf = ((hBias).size() <= gemmIdx) ? nullptr : hBias[gemmIdx].buf();
                    if(applyBias && arg.bias_stride > 0)
                    {
                        hBias_buf = ((char*)hBias_buf)
                                    + (arg.bias_stride * batchIdx * realDataTypeSize(Tbias));
                    }

                    hipblaslt::host_validation::EpilogueArguments epilogue;
                    epilogue.rows             = M[gemmIdx];
                    epilogue.columns          = N[gemmIdx];
                    epilogue.leadingDimension = ldd[gemmIdx];
                    epilogue.input
                        = hD_gold_epl[gemmIdx].as<char>() + pos * realDataTypeSize(Talpha);
                    epilogue.output = hD_gold[gemmIdx].as<char>() + pos * realDataTypeSize(To);
                    epilogue.rawOutput
                        = hBias_gold_epl[gemmIdx].as<char>() + pos * realDataTypeSize(Talpha);
                    epilogue.amax           = arg.amaxD ? hAmaxD_gold[gemmIdx].as<char>() : nullptr;
                    epilogue.auxiliary      = ePos;
                    epilogue.auxiliaryType  = Taux;
                    epilogue.outputScale    = scaleDValue;
                    epilogue.auxiliaryScale = scaleEValue;
                    epilogue.bias           = applyBias ? hBias_buf : nullptr;
                    epilogue.biasType       = Tbias;
                    epilogue.activationParameter0 = arg.activation_arg1;
                    epilogue.activationParameter1 = arg.activation_arg2;
                    epilogue.outputType           = To;
                    epilogue.computeType          = Talpha;

                    switch(arg.activation_type)
                    {
                    case hipblaslt_activation_type::gelu:
                        epilogue.activation = roc::host_validation::Activation::Gelu;
                        break;
                    case hipblaslt_activation_type::relu:
                        epilogue.activation = roc::host_validation::Activation::Relu;
                        break;
                    case hipblaslt_activation_type::swish:
                        // hipBLASLt's historical SWISH_EXT path implements
                        // SiLU and ignores the activation parameter.
                        epilogue.activation = roc::host_validation::Activation::Silu;
                        break;
                    case hipblaslt_activation_type::clamp:
                        epilogue.activation = roc::host_validation::Activation::Clamp;
                        break;
                    default:
                        epilogue.activation = roc::host_validation::Activation::None;
                        break;
                    }
                    epilogue.activationApplication
                        = arg.gradient
                                  && epilogue.activation != roc::host_validation::Activation::None
                              ? roc::host_validation::ActivationApplication::Gradient
                              : roc::host_validation::ActivationApplication::Forward;
                    hipblaslt::host_validation::referenceEpilogue(epilogue);

                    if(arg.gradient && arg.bias_vector && batchIdx == num_batches[gemmIdx] - 1)
                    {
                        auto* hBias_gold_buf = hBias_gold[gemmIdx].buf();
                        if(arg.bias_stride > 0 && hBias_gold_buf != nullptr)
                        {
                            hBias_gold_buf = (char*)hBias_gold_buf
                                             + arg.bias_stride * batchIdx * realDataTypeSize(Tbias);
                        }

                        auto reduceBias = [&](const void* input,
                                              hipDataType inputType,
                                              int64_t     rows,
                                              int64_t     columns,
                                              int64_t     rowStride,
                                              int64_t     columnStride) {
                            hipblaslt::host_validation::ReductionArguments reduction;
                            reduction.rows            = rows;
                            reduction.columns         = columns;
                            reduction.rowStride       = rowStride;
                            reduction.columnStride    = columnStride;
                            reduction.input           = input;
                            reduction.inputType       = inputType;
                            reduction.output          = hBias_gold_buf;
                            reduction.outputType      = Tbias;
                            reduction.accumulatorType = HIP_R_32F;
                            hipblaslt::host_validation::referenceSum(reduction);
                        };

                        if(arg.bias_source == hipblaslt_bias_source::d)
                        {
                            reduceBias(hBias_gold_epl[gemmIdx].as<char>()
                                           + pos * realDataTypeSize(Talpha),
                                       Talpha,
                                       M[gemmIdx],
                                       N[gemmIdx],
                                       1,
                                       ldd[gemmIdx]);
                        }
                        else if(arg.bias_source == hipblaslt_bias_source::a)
                        {
                            reduceBias(hA[gemmIdx].buf(),
                                       TiA,
                                       size_bias[gemmIdx],
                                       K[gemmIdx],
                                       transA == HIPBLAS_OP_N ? 1 : lda[gemmIdx],
                                       transA == HIPBLAS_OP_N ? lda[gemmIdx] : 1);
                        }
                        else if(arg.bias_source == hipblaslt_bias_source::b)
                        {
                            reduceBias(hB[gemmIdx].buf(),
                                       TiB,
                                       size_bias[gemmIdx],
                                       K[gemmIdx],
                                       transB == HIPBLAS_OP_N ? ldb[gemmIdx] : 1,
                                       transB == HIPBLAS_OP_N ? 1 : ldb[gemmIdx]);
                        }
                    }
                }
                else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    // Note: for MX types, pass the reference float instead so there is
                    //       no need to convert them to float in hipblaslt_reference_gemm
                    
                    // Added this logic to mimic the rocblas test quick_gemm_batched_bad_arg_f32_r_bad_arg_F
                    // This rocblas test passes alpha, A and B as 0 but beta as non-zero with valid C and D
                    // To mimic this behavior if --sizek is passed as 0 in hipblaslt-bench for --batch_mode 1, size_dA and size_dB
                    // will be set to 0 since A is MxK and B is KxN. In this case, we pass the pointer array A and B for 
                    // General batched GEMM as nullptr and introduced an explicit check for AddressA and AddressB != 0
                    // in KernelWriterAssembly.py since the dereference of AddressA and AddressB for 
                    // General Batched GEMM happens before the alphaNonZero check.                    
                    void *ptrA = (size_dA[0]) ? hA[batchIdx].as<char>() : nullptr;
                    void *ptrB = (size_dB[0]) ? hB[batchIdx].as<char>() : nullptr;                    
                    hipblaslt_reference_gemm(transA,
                               transB,
                               M[gemmIdx],
                               N[gemmIdx],
                               K[gemmIdx],
                               alpha,
                               ptrA,
                               lda[gemmIdx],
                               ptrB,
                               ldb[gemmIdx],
                               betaTemp,
                               hC[batchIdx].as<char>(),
                               ldc[gemmIdx],
                               hD_gold[batchIdx].as<char>(),
                               ldd[gemmIdx],
                               nullptr,
                               scaleAVec,
                               scaleBVec,
                               scaleDValue,
                               (arg.scaleA == hipblaslt_scaling_format::Vector),
                               (arg.scaleB == hipblaslt_scaling_format::Vector),
                               isScaleAMXFormat ? HIP_R_32F : TiA,
                               isScaleBMXFormat ? HIP_R_32F : TiB,
                               To,
                               To,
                               Tc,
                               isScaleAMXFormat ? HIP_R_32F : TciA,
                               isScaleBMXFormat ? HIP_R_32F : TciB,
                               isBlockScaling(arg.scaleA),
                               isBlockScaling(arg.scaleB));
                }
                else
                {
                    // Note: for MX types, pass the reference float instead so there is
                    //       no need to convert them to float in hipblaslt_reference_gemm
                    hipblaslt_reference_gemm(
                        transA,
                        transB,
                        M[gemmIdx],
                        N[gemmIdx],
                        K[gemmIdx],
                        alpha,
                        isScaleAMXFormat
                            ? reinterpret_cast<char*>(refA[gemmIdx].data())
                                  + stride_a[gemmIdx] * batchIdx * realDataTypeSize(HIP_R_32F)
                            : hA[gemmIdx].as<char>()
                                  + stride_a[gemmIdx] * batchIdx * realDataTypeSize(TiA),
                        lda[gemmIdx],
                        isScaleBMXFormat
                            ? reinterpret_cast<char*>(refB[gemmIdx].data())
                                  + stride_b[gemmIdx] * batchIdx * realDataTypeSize(HIP_R_32F)
                            : hB[gemmIdx].as<char>()
                                  + stride_b[gemmIdx] * batchIdx * realDataTypeSize(TiB),
                        ldb[gemmIdx],
                        betaTemp,
                        hC[gemmIdx].as<char>()
                            + stride_c[gemmIdx] * batchIdx * realDataTypeSize(To),
                        ldc[gemmIdx],
                        hD_gold[gemmIdx].as<char>()
                            + stride_d[gemmIdx] * batchIdx * realDataTypeSize(To),
                        ldd[gemmIdx],
                        nullptr,
                        scaleAVec,
                        scaleBVec,
                        scaleDValue,
                        (arg.scaleA == hipblaslt_scaling_format::Vector),
                        (arg.scaleB == hipblaslt_scaling_format::Vector),
                        isScaleAMXFormat ? HIP_R_32F : TiA,
                        isScaleBMXFormat ? HIP_R_32F : TiB,
                        To,
                        To,
                        Tc,
                        isScaleAMXFormat ? HIP_R_32F : TciA,
                        isScaleBMXFormat ? HIP_R_32F : TciB,
                        isBlockScaling(arg.scaleA),
                        isBlockScaling(arg.scaleB));
                }
            }
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }
    }
    void* alpha_ptr = nullptr;
    void* beta_ptr  = nullptr;

    if(problem_count > 0)
    {
        if(TiA == HIP_C_32F || TiA == HIP_C_64F)
        {
            if(TiA == HIP_C_32F)
            {
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(h_alpha[0].cf);
                beta_ptr  = (void*)&(h_beta[0].cf);
            }
            else if(TiA == HIP_C_64F)
            {

                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(h_alpha[0].cd);
                beta_ptr  = (void*)&(h_beta[0].cd);
            }
        }
        else
        {
            switch(Tc)
            {
            case HIP_R_32F:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(h_alpha[0].f32);
                beta_ptr  = (void*)&(h_beta[0].f32);
                break;
            case HIP_R_64F:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(h_alpha[0].f64);
                beta_ptr  = (void*)&(h_beta[0].f64);
                break;
            case HIP_R_16F:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(h_alpha[0].f16);
                beta_ptr  = (void*)&(h_beta[0].f16);
                break;
            case HIP_R_32I:
                alpha_ptr = arg.scaleAlpha_vector ? (void*)dScaleAlphaVec[0].buf()
                                                  : (void*)&(h_alpha[0].i32);
                beta_ptr  = (void*)&(h_beta[0].i32);
                break;
            default:
                hipblaslt_cerr << "FATAL: Unsupported type in pointer setup for hipblasLtMatmul"
                               << std::endl;
                alpha_ptr = nullptr;
                beta_ptr  = nullptr;
            }
        }
    }

    auto readValidationSideOutputs = [&] {
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
            return;

        for(int gemmIdx = 0; gemmIdx < problem_count; ++gemmIdx)
        {
            if(!arg.gradient && arg.use_e)
            {
                CHECK_HIP_ERROR(
                    synchronize(hE[gemmIdx], dE[gemmIdx], 0, 0, 0, 0, 1, false, stream));
            }
            if(arg.amaxD)
            {
                CHECK_HIP_ERROR(
                    synchronize(hAmaxD[gemmIdx], dAmaxD[gemmIdx], 0, 0, 0, 0, 1, false, stream));
            }
            if(arg.gradient && arg.bias_vector)
            {
                CHECK_HIP_ERROR(
                    synchronize(hBias[gemmIdx], dBias[gemmIdx], 0, 0, 0, 0, 1, false, stream));
            }
        }
    };

    const hipblaslt::host_validation::MatmulValidationOptions validationOptions{
        .comparePointwise = bool(arg.unit_check),
        .compareNorm = bool(arg.norm_check),
        .searchAllClose = bool(arg.allclose_check),
        .computeUlp = bool(arg.ulp_check),
        .assertNorm = arg.norm_check_assert,
        .computeType = arg.compute_type,
        .inputTypeA = arg.a_type,
        .inputTypeB = arg.b_type,
    };

    auto makeValidationCases
        = [&](const std::vector<
                  hipblaslt::host_validation::MatmulValidationCase::PointwiseTolerance>&
                  pointwiseTolerances) {
              using hipblaslt::host_validation::HostComparisonRequest;
              using hipblaslt::host_validation::MatmulValidationCase;

              auto output = [](int64_t     rows,
                               int64_t     columns,
                               int64_t     leadingDimension,
                               int64_t     batchStride,
                               int64_t     batchCount,
                               const void* expected,
                               const void* observed,
                               hipDataType type) {
                  HostComparisonRequest request;
                  request.rows             = rows;
                  request.columns          = columns;
                  request.leadingDimension = leadingDimension;
                  request.batchStride      = batchStride;
                  request.batchCount       = batchCount;
                  request.expected         = expected;
                  request.observed         = observed;
                  request.type             = type;
                  return request;
              };

              std::vector<MatmulValidationCase> cases;
              if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
              {
                  MatmulValidationCase testCase;
                  testCase.pointwiseTolerance = pointwiseTolerances.front();
                  testCase.outputs.reserve(num_batches.front());
                  for(int batch = 0; batch < num_batches.front(); ++batch)
                  {
                      testCase.outputs.push_back(output(M.front(),
                                                        N.front(),
                                                        ldd.front(),
                                                        0,
                                                        1,
                                                        hD_gold[batch].buf(),
                                                        hD_1[batch].buf(),
                                                        To));
                  }
                  cases.push_back(std::move(testCase));
                  return cases;
              }

              cases.reserve(problem_count);
              for(int gemmIdx = 0; gemmIdx < problem_count; ++gemmIdx)
              {
                  MatmulValidationCase testCase;
                  testCase.pointwiseTolerance = pointwiseTolerances[gemmIdx];
                  testCase.outputs.push_back(output(M[gemmIdx],
                                                    N[gemmIdx],
                                                    ldd[gemmIdx],
                                                    stride_d[gemmIdx],
                                                    num_batches[gemmIdx],
                                                    hD_gold[gemmIdx].buf(),
                                                    hD_1[gemmIdx].buf(),
                                                    To));
                  if(arg.amaxD)
                  {
                      auto request = output(1,
                                            1,
                                            1,
                                            1,
                                            num_batches[gemmIdx],
                                            hAmaxD_gold[gemmIdx].buf(),
                                            hAmaxD[gemmIdx].buf(),
                                            Talpha);
                      testCase.maximum
                          = MatmulValidationCase::SideOutput{request, request, false};
                  }
                  if(!arg.gradient && arg.use_e)
                  {
                      auto request = output(M[gemmIdx],
                                            N[gemmIdx],
                                            lde[gemmIdx],
                                            stride_e[gemmIdx],
                                            num_batches[gemmIdx],
                                            hE_gold[gemmIdx].buf(),
                                            hE[gemmIdx].buf(),
                                            Taux);
                      testCase.auxiliary
                          = MatmulValidationCase::SideOutput{request, request, true};
                  }
                  if(arg.gradient && arg.bias_vector)
                  {
                      auto pointwise = output(size_bias[gemmIdx],
                                              1,
                                              size_bias[gemmIdx],
                                              size_bias[gemmIdx],
                                              num_batches[gemmIdx],
                                              hBias_gold[gemmIdx].buf(),
                                              hBias[gemmIdx].buf(),
                                              Tbias);
                      auto norm = output(M[gemmIdx],
                                         1,
                                         M[gemmIdx],
                                         M[gemmIdx],
                                         num_batches[gemmIdx],
                                         hBias_gold[gemmIdx].buf(),
                                         hBias[gemmIdx].buf(),
                                         Tbias);
                      testCase.bias = MatmulValidationCase::SideOutput{pointwise, norm, false};
                  }
                  cases.push_back(std::move(testCase));
              }
              return cases;
          };

    auto validateOutputs
        = [&](hipblaslt::host_validation::MatmulValidationMetrics metrics) {
              const auto pointwiseTolerances
                  = matmulValidationTolerances(arg, K, TiA, TiB, To, Tc);
              readValidationSideOutputs();
              const auto cases = makeValidationCases(pointwiseTolerances);
              hipblaslt::host_validation::validateMatmulOutputs(
                  {.options = validationOptions, .cases = cases, .metrics = metrics});
          };

    if(!arg.timing)
    {
        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            if((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.c_equal_d)
            {
                if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) // Iterate for batch_count for General Batched GEMM
                {
                    for(int i = 0; i < arg.batch_count; i++)
                    {
                        CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                    }
                }
                else
                {
                    for(int i = 0; i < problem_count; i++)
                    {
                        CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                    }
                }
            }
            if(!do_grouped_gemm)
            {
                if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    gemmVec[0].setMaxWorkspaceBytes(workspace_size);
                    CHECK_HIPBLASLT_ERROR(
                        gemmVec[0].initialize(heuristicResult[sol].algo,
                                              tuningVec[heuristicTuningIndex[sol]],
                                              *dWorkspace));
                    CHECK_HIPBLASLT_ERROR(gemmVec[0].run(stream));
                }
                else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    // Added this logic to mimic the rocblas test quick_gemm_batched_bad_arg_f32_r_bad_arg_F
                    // This rocblas test passes alpha, A and B as 0 but beta as non-zero with valid C and D
                    // To mimic this behavior if --sizek is passed as 0 in hipblaslt-bench for --batch_mode 1, size_dA and size_dB
                    // will be set to 0 since A is MxK and B is KxN. In this case, we pass the pointer array A and B for 
                    // General batched GEMM as nullptr and introduced an explicit check for AddressA and AddressB != 0
                    // in KernelWriterAssembly.py since the dereference of AddressA and AddressB for 
                    // General Batched GEMM happens before the alphaNonZero check.
                    void *ptrA = (size_dA[0]) ? dda[0] : nullptr;
                    void *ptrB = (size_dB[0]) ? ddb[0] : nullptr;                    
                    EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                          matmul[0][0],
                                                          alpha_in[0],
                                                          ptrA,
                                                          matA[0],
                                                          ptrB,
                                                          matB[0],
                                                          &(h_beta[0]),
                                                          ddc[0],
                                                          matC[0],
                                                          ddd[0],
                                                          matD[0],
                                                          &heuristicResult[sol].algo,
                                                          *dWorkspace,
                                                          workspace_size,
                                                          stream),
                                          HIPBLAS_STATUS_SUCCESS);
                }
                else
                {
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                          matmul[0][0],
                                                          alpha_ptr,
                                                          dA[0].buf(),
                                                          matA[0],
                                                          dB[0].buf(),
                                                          matB[0],
                                                          beta_ptr,
                                                          dC[0].buf(),
                                                          matC[0],
                                                          (*dDp)[0].buf(),
                                                          matD[0],
                                                          &heuristicResult[sol].algo,
                                                          *dWorkspace,
                                                          workspace_size,
                                                          stream),
                                          HIPBLAS_STATUS_SUCCESS);
                }
            }
            else
            {
                //grouped gemm
                if(arg.use_user_args)
                {
                    groupedGemmVec[0].setMaxWorkspaceBytes(workspace_size);
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemmVec[0].initialize(heuristicResult[sol].algo,
                                                     tuningVec[heuristicTuningIndex[0]],
                                                     *dWorkspace));
                    groupedGemmVec[0].getDefaultValueForDeviceUserArguments(userArgs);
                    // Copy them to device memory
                    CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                                              userArgs,
                                              problem_count * sizeof(hipblaslt_ext::UserArguments),
                                              hipMemcpyHostToDevice));

                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(d_userArgs, stream));
                }
                else
                {
                    groupedGemmVec[0].setMaxWorkspaceBytes(workspace_size);
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemmVec[0].initialize(heuristicResult[sol].algo,
                                                     tuningVec[heuristicTuningIndex[0]],
                                                     *dWorkspace,
                                                     false,
                                                     stream));

                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(stream));
                }
            }

            double              hipblaslt_error   = 0.0;
            double              hipblaslt_atol    = 1;
            double              hipblaslt_rtol    = 1;
            double              hipblaslt_max_ulp = 0.0;
            double              hipblaslt_avg_ulp = 0.0;

            if(arg.unit_check || arg.norm_check || arg.allclose_check)
            {
                if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    copy_gemm_to_host(stream, arg.batch_count, hD_1, (*dDp));
                }
                else
                {
                    copy_gemm_to_host(stream, problem_count, hD_1, (*dDp));
                }
                validateOutputs({hipblaslt_error,
                                 hipblaslt_atol,
                                 hipblaslt_rtol,
                                 hipblaslt_max_ulp,
                                 hipblaslt_avg_ulp});
            }
        }
    }
    else
    {
        // Get device information
        hipDeviceProp_t deviceProps;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
        int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

        size_t      best_sol       = -1;
        double      best_flops     = 0.0;
        double      best_gpu_time  = std::numeric_limits<double>::max();
        double      best_warm_time = std::numeric_limits<double>::max();
        std::string best_s_name    = "";
        std::string best_k_name    = "";
        double      best_norm      = 0.0;
        double      best_atol      = 0.0;
        double      best_rtol      = 0.0;
        double      best_max_ulp   = 0.0;
        double      best_avg_ulp   = 0.0;
        int number_cold_calls
            = ((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.cold_iters == 0)
                  ? 1
                  : arg.cold_iters;
        // Adaptive timing ignores --cold_iters/--iters: warmup and batch are sized
        // inside run_measurement. Keep one cold call only when a result copy or the
        // skip-slow screen needs it.
        if(arg.adaptive)
            number_cold_calls = (arg.unit_check || arg.norm_check || arg.allclose_check
                                 || arg.skip_slow_solution_ratio != 0.0f)
                                    ? 1
                                    : 0;
        int number_hot_calls = arg.iters;

        // Adaptive timing configuration shared by every solution measured below.
        // Adaptive timing is gated on --adaptive; the per-knob settings apply only then.
        hipblaslt_bench::TimingConfig timingCfg;
        timingCfg.iters         = number_hot_calls;
        timingCfg.use_gpu_timer = arg.use_gpu_timer;
        timingCfg.adaptive      = arg.adaptive;
        if(arg.adaptive)
        {
            timingCfg.warmup_time      = arg.warmup_time;
            timingCfg.sample_time      = arg.sample_time;
            timingCfg.measure_time     = arg.measure_time;
            timingCfg.max_measure_time = arg.max_measure_time;
            timingCfg.min_iters          = arg.min_iters;
            timingCfg.max_iters          = arg.max_iters;
            timingCfg.noise_threshold    = arg.noise_threshold;
            timingCfg.stability_threshold = arg.stability_threshold;
            timingCfg.stability_window    = arg.stability_window;
            timingCfg.stability_interval  = arg.stability_interval;
        }
        if(arg.adaptive)
        {
            if(const auto err = hipblaslt_bench::validate_adaptive_config(timingCfg); !err.empty())
            {
                hipblaslt_cerr << "error: invalid adaptive timing config: " << err << std::endl;
                return;
            }
        }

        hipblaslt_bench::TimingResult timing;
        // Stop the sample loop if a launch hits a gtest fatal failure.
        auto timingAbort = []() -> bool {
#ifdef GOOGLE_TEST
            return ::testing::Test::HasFatalFailure();
#else
            return false;
#endif
        };
        hipblaslt_bench::TimingResult best_timing;

        int    flush_iter      = 100000;
        double flush_time_used = 0;
        if(arg.flush)
        {
            static std::unordered_map<std::string, double> flush_times_cache;
            static std::mutex                              mtx;
            std::lock_guard<std::mutex>                    lock(mtx);
            std::string                                    device_uuid(deviceProps.uuid.bytes);
            if(!flush_times_cache.count(device_uuid))
            {
                for(int i = 0; i < flush_iter; i++)
                    hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                pre_gpu_time(arg.use_gpu_timer, event_gpu_time_start, flush_time_used, stream);
                for(int i = 0; i < flush_iter; i++)
                    hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                post_gpu_time(arg.use_gpu_timer,
                              event_gpu_time_start,
                              event_gpu_time_end,
                              flush_time_used,
                              stream);
                flush_time_used /= flush_iter;
                flush_times_cache[device_uuid] = flush_time_used;
            }
            else
            {
                flush_time_used = flush_times_cache[device_uuid];
            }
        }
        timingCfg.flush_us = flush_time_used;

        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            // Reset per-solution so an aborted/empty measurement can't report the prior
            // solution's stats (run_measurement leaves `out` untouched on early return).
            timing = {};
            if((arg.unit_check || arg.norm_check || arg.allclose_check) && arg.c_equal_d)
            {
                if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    for(int i = 0; i < arg.batch_count; i++)
                    {
                        CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                    }
                }
                else
                {
                    for(int i = 0; i < problem_count; i++)
                    {
                        CHECK_HIP_ERROR(synchronize(dC[i], hC[i], block_count));
                    }
                }
            }
            if(!do_grouped_gemm)
            {
                auto perf_monitor = EfficiencyMonitor::create();
                if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        gemmVec[b].setMaxWorkspaceBytes(workspace_size);
                        CHECK_HIPBLASLT_ERROR(
                            gemmVec[b].initialize(heuristicResult[sol].algo,
                                                  tuningVec[heuristicTuningIndex[sol]],
                                                  *dWorkspace));
                    }
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(gemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int b = static_cast<int>(i % block_count);
                            CHECK_HIPBLASLT_ERROR(gemmVec[b].run(stream));
                            if(arg.flush)
                                hipLaunchKernelGGL(
                                    flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                }
                else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY) //For General Batch GEMM
                {
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        auto ptr_matmul = matmul[i % block_count][0];
                        auto ptr_alpha  = arg.scaleAlpha_vector
                                              ? (dScaleAlphaVec[0].as<char>())
                                                   + (i % block_count) * size_scaleAlphaVec[0]
                                              : alpha_in[0];
                        // Added this logic to mimic the rocblas test quick_gemm_batched_bad_arg_f32_r_bad_arg_F
                        // This rocblas test passes alpha, A and B as 0 but beta as non-zero with valid C and D
                        // To mimic this behavior if --sizek is passed as 0 in hipblaslt-bench for --batch_mode 1, size_dA and size_dB
                        // will be set to 0 since A is MxK and B is KxN. In this case, we pass the pointer array A and B for 
                        // General batched GEMM as nullptr and introduced an explicit check for AddressA and AddressB != 0
                        // in KernelWriterAssembly.py since the dereference of AddressA and AddressB for 
                        // General Batched GEMM happens before the alphaNonZero check.                                              
                        void *ptrA = (size_dA[0]) ? dda[i % block_count] : nullptr;
                        void *ptrB = (size_dB[0]) ? ddb[i % block_count] : nullptr;
                        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                              ptr_matmul,
                                                              ptr_alpha,
                                                              ptrA,
                                                              matA[0],
                                                              ptrB,
                                                              matB[0],
                                                              &(h_beta[0]),
                                                              ddc[i % block_count],
                                                              matC[0],
                                                              ddd[i % block_count],
                                                              matD[0],
                                                              &heuristicResult[sol].algo,
                                                              *dWorkspace,
                                                              workspace_size,
                                                              stream),
                                              HIPBLAS_STATUS_SUCCESS);
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, arg.batch_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int  b          = static_cast<int>(i % block_count);
                            auto ptr_matmul = matmul[b][0];
                            auto ptr_alpha  = arg.scaleAlpha_vector
                                                  ? (dScaleAlphaVec[0].as<char>())
                                                        + b * size_scaleAlphaVec[0]
                                                  : alpha_in[0];
                            void* ptrA = (size_dA[0]) ? dda[b] : nullptr;
                            void* ptrB = (size_dB[0]) ? ddb[b] : nullptr;
                            EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                                  ptr_matmul,
                                                                  ptr_alpha,
                                                                  ptrA,
                                                                  matA[0],
                                                                  ptrB,
                                                                  matB[0],
                                                                  &(h_beta[0]),
                                                                  ddc[b],
                                                                  matC[0],
                                                                  ddd[b],
                                                                  matD[0],
                                                                  &heuristicResult[sol].algo,
                                                                  *dWorkspace,
                                                                  workspace_size,
                                                                  stream),
                                                  HIPBLAS_STATUS_SUCCESS);
                            if(arg.flush)
                                hipLaunchKernelGGL(
                                    flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                }
                else
                {
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        auto ptr_matmul = matmul[i % block_count][0];
                        auto ptr_alpha  = arg.scaleAlpha_vector
                                              ? (dScaleAlphaVec[0].as<char>())
                                                   + (i % block_count) * size_scaleAlphaVec[0]
                                              : alpha_in[0];

                        EXPECT_HIPBLAS_STATUS(
                            hipblasLtMatmul(
                                handle,
                                ptr_matmul,
                                alpha_ptr,
                                dA[0].as<char>()
                                    + (i % block_count) * size_dA[0] * realDataTypeSize(TiA),
                                matA[0],
                                dB[0].as<char>()
                                    + (i % block_count) * size_dB[0] * realDataTypeSize(TiB),
                                matB[0],
                                beta_ptr,
                                dC[0].as<char>()
                                    + (i % block_count) * size_C[0] * realDataTypeSize(To),
                                matC[0],
                                (*dDp)[0].as<char>()
                                    + (i % block_count) * size_D[0] * realDataTypeSize(To),
                                matD[0],
                                &heuristicResult[sol].algo,
                                *dWorkspace,
                                workspace_size,
                                stream),
                            HIPBLAS_STATUS_SUCCESS);
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int  b          = static_cast<int>(i % block_count);
                            auto ptr_matmul = matmul[b][0];
                            auto ptr_alpha  = arg.scaleAlpha_vector
                                                  ? (dScaleAlphaVec[0].as<char>())
                                                        + b * size_scaleAlphaVec[0]
                                                  : alpha_in[0];
                            EXPECT_HIPBLAS_STATUS(
                                hipblasLtMatmul(
                                    handle,
                                    ptr_matmul,
                                    alpha_ptr,
                                    dA[0].as<char>() + b * size_dA[0] * realDataTypeSize(TiA),
                                    matA[0],
                                    dB[0].as<char>() + b * size_dB[0] * realDataTypeSize(TiB),
                                    matB[0],
                                    beta_ptr,
                                    dC[0].as<char>() + b * size_C[0] * realDataTypeSize(To),
                                    matC[0],
                                    (*dDp)[0].as<char>() + b * size_D[0] * realDataTypeSize(To),
                                    matD[0],
                                    &heuristicResult[sol].algo,
                                    *dWorkspace,
                                    workspace_size,
                                    stream),
                                HIPBLAS_STATUS_SUCCESS);
                            if(arg.flush)
                                hipLaunchKernelGGL(
                                    flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                }
                gpu_time_used = timing.median_us;
                perf_monitor->stop();
            }
            else
            {
                auto perf_monitor = EfficiencyMonitor::create();
                if(arg.use_user_args)
                {
                    std::vector<unsigned char*> d_userArgsVec(block_count);
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        groupedGemmVec[b].setMaxWorkspaceBytes(workspace_size);
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size)));
                        groupedGemmVec[b].getDefaultValueForDeviceUserArguments(userArgs);
                        d_userArgsVec[b] = (unsigned char*)d_userArgs
                                           + b * problem_count * sizeof(hipblaslt_ext::UserArguments);
                        // Copy them to device memory
                        CHECK_HIP_ERROR(hipMemcpy(d_userArgsVec[b],
                                                  userArgs,
                                                  problem_count * sizeof(hipblaslt_ext::UserArguments),
                                                  hipMemcpyHostToDevice));
                    }
                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(
                            d_userArgsVec[i % block_count], stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int b = static_cast<int>(i % block_count);
                            CHECK_HIPBLASLT_ERROR(
                                groupedGemmVec[b].run(d_userArgsVec[b], stream));
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                    gpu_time_used = timing.median_us;
                    perf_monitor->stop();
                }
                else
                {
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        groupedGemmVec[b].setMaxWorkspaceBytes(workspace_size);
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size),
                            false,
                            stream));
                    }

                    if(arg.skip_slow_solution_ratio)
                        pre_gpu_time(
                            arg.use_gpu_timer, event_gpu_time_start, gpu_time_used, stream);
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check || arg.allclose_check))
                            copy_gemm_to_host(stream, problem_count, hD_1, (*dDp));
                    }
                    if(arg.skip_slow_solution_ratio)
                    {
                        post_gpu_time(arg.use_gpu_timer,
                                      event_gpu_time_start,
                                      event_gpu_time_end,
                                      gpu_time_used,
                                      stream);
                        best_warm_time
                            = best_warm_time < gpu_time_used ? best_warm_time : gpu_time_used;
                        if((gpu_time_used * arg.skip_slow_solution_ratio) > best_warm_time)
                        {
                            hipblaslt_cout
                                << std::setprecision(2) << "Skip solution: " << sol
                                << " (best warm-up = " << best_warm_time / number_cold_calls
                                << " us , warm-up = " << gpu_time_used / number_cold_calls
                                << " us, skip ratio = " << arg.skip_slow_solution_ratio << ")"
                                << std::endl;
                            continue;
                        }
                    }
                    perf_monitor->start();
                    hipblaslt_bench::run_measurement(
                        [&](int64_t i) {
                            int b = static_cast<int>(i % block_count);
                            CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].run(stream));
                        },
                        timingCfg,
                        event_gpu_time_start,
                        event_gpu_time_end,
                        stream,
                        timing,
                        timingAbort);
                    gpu_time_used = timing.median_us;
                    perf_monitor->stop();
                }
            }

            double flops = 0;
            for(int gemmIdx = 0; gemmIdx < problem_count; gemmIdx++)
            {
                flops += gemm_gflop_count(M[gemmIdx], N[gemmIdx], K[gemmIdx], Talpha);
                switch(arg.activation_type)
                {
                case hipblaslt_activation_type::relu:
                    flops += relu_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                case hipblaslt_activation_type::gelu:
                    flops += gelu_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                case hipblaslt_activation_type::swish:
                    flops += silu_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                case hipblaslt_activation_type::clamp:
                    flops += clamp_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                case hipblaslt_activation_type::sigmoid:
                    flops += sigmoid_gflop_count(M[gemmIdx], N[gemmIdx], Talpha);
                    break;
                default:
                    break;
                }
            }

            double              hipblaslt_error   = 0.0;
            double              hipblaslt_atol    = 1;
            double              hipblaslt_rtol    = 1;
            double              hipblaslt_max_ulp = 0.0;
            double              hipblaslt_avg_ulp = 0.0;
            if(arg.unit_check || arg.norm_check || arg.allclose_check)
            {
                if(arg.dump_matrix)
                {
                    for(int batchId = 0; batchId < num_batches[0]; batchId++)
                    {
                        hipblasltDispatchValuesToFile(HIPBLAS_OP_N,
                                                    To,
                                                    M[0],
                                                    N[0],
                                                    ldd[0],
                                                    hD_1[0].as<char>() + batchId * stride_d[0] * realDataTypeSize(To),
                                                    "batch_" + std::to_string(batchId) + "_D_output.txt");
                        hipblasltDispatchValuesToFile(HIPBLAS_OP_N,
                                                    To,
                                                    M[0],
                                                    N[0],
                                                    ldd[0],
                                                    hD_gold[0].as<char>() + batchId * stride_d[0] * realDataTypeSize(To),
                                                    "batch_" + std::to_string(batchId) + "_D_Gold_output.txt");
                    }
                }
                validateOutputs({hipblaslt_error,
                                 hipblaslt_atol,
                                 hipblaslt_rtol,
                                 hipblaslt_max_ulp,
                                 hipblaslt_avg_ulp});
            }

#define argument_param                                                                            \
    e_transA, e_transB, e_grouped_gemm, e_batch_count, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a, \
        e_beta, e_ldb, e_stride_b, e_ldc, e_stride_c, e_ldd, e_stride_d, e_a_type, e_b_type,      \
        e_c_type, e_d_type, e_compute_type, e_scaleA, e_scaleB, e_scaleC, e_scaleD, e_amaxD,      \
        e_swizzle_a, e_swizzle_b, e_activation_type, e_bias_vector, e_bias_type, e_aux_type

            const char* tuningEnv     = getenv("HIPBLASLT_TUNING_FILE");
            int32_t     solutionIndex = ((tuningEnv && heuristicResult.size() == 1)
                                     || (arg.print_solution_found && arg.print_kernel_info))
                                            ? hipblaslt_ext::getIndexFromAlgo(heuristicResult[sol].algo)
                                            : -1;
            std::string solutionName  = "";
            std::string kernelName    = "";
            std::string archName      = "";
            std::string cuNum         = "";

            if(tuningEnv && heuristicResult.size() == 1)
            {
                archName = deviceProps.gcnArchName;
                cuNum    = std::to_string(deviceProps.multiProcessorCount);
            }

            if(arg.print_solution_found)
            {
                if(arg.print_kernel_info)
                {
                    if(arg.use_ext && batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                    {
                        if(!do_grouped_gemm)
                        {
                            solutionName = gemmVec[0].getSolutionName();
                            kernelName   = gemmVec[0].getKernelName();
                        }
                        else
                        {
                            solutionName = groupedGemmVec[0].getSolutionName();
                            kernelName   = groupedGemmVec[0].getKernelName();
                        }
                    }
                    else
                    {
                        solutionName = hipblaslt_ext::getSolutionNameFromAlgo(
                            handle, heuristicResult[sol].algo);
                        kernelName = hipblaslt_ext::getKernelNameFromAlgo(
                            handle, heuristicResult[sol].algo);
                    }
                }
                ArgumentModel<argument_param>{}.log_args(
                    Talpha,
                    hipblaslt_cout,
                    sol,
                    solutionIndex,
                    solutionName,
                    kernelName,
                    archName,
                    cuNum,
                    arg,
                    (uint32_t)tuningVec[heuristicTuningIndex[sol]].getSplitK(),
                    (uint32_t)tuningVec[heuristicTuningIndex[sol]].getWgm(),
                    gpu_time_used,
                    flops,
                    gpu_mem_gbytes,
                    cpu_time_used,
                    hipblaslt_error,
                    hipblaslt_atol,
                    hipblaslt_rtol,
                    hipblaslt_max_ulp,
                    hipblaslt_avg_ulp,
                    timing);
            }
            if(best_gpu_time > gpu_time_used)
            {
                best_sol      = sol;
                best_flops    = flops;
                best_gpu_time = gpu_time_used;
                best_s_name   = solutionName;
                best_k_name   = kernelName;
                best_norm     = hipblaslt_error;
                best_atol     = hipblaslt_atol;
                best_rtol     = hipblaslt_rtol;
                best_max_ulp  = hipblaslt_max_ulp;
                best_avg_ulp  = hipblaslt_avg_ulp;
                best_timing   = timing;
            }
        }

        if(heuristicResult.size() > 1)
        {
            const char* tuningEnv = getenv("HIPBLASLT_TUNING_FILE");
            int32_t     solutionIndex
                = (tuningEnv || arg.print_kernel_info)
                      ? hipblaslt_ext::getIndexFromAlgo(heuristicResult[best_sol].algo)
                      : -1;
            std::string solutionName = "";
            std::string kernelName   = "";
            std::string archName     = "";
            std::string cuNum        = "";
            if(tuningEnv)
            {
                archName = deviceProps.gcnArchName;
                cuNum    = std::to_string(deviceProps.multiProcessorCount);
            }

            if(arg.print_kernel_info)
            {
                solutionName = best_s_name;
                kernelName   = best_k_name;
            }

            hipblaslt_cout << "Winner: " << std::endl;
            ArgumentModel<argument_param>{}.log_args(
                Talpha,
                hipblaslt_cout,
                best_sol,
                solutionIndex,
                solutionName,
                kernelName,
                archName,
                cuNum,
                arg,
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].getSplitK(),
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].getWgm(),
                best_gpu_time,
                best_flops,
                gpu_mem_gbytes,
                cpu_time_used,
                best_norm,
                best_atol,
                best_rtol,
                best_max_ulp,
                best_avg_ulp,
                best_timing);
        }
    }

    for(auto it : ptrs)
    {
        CHECK_HIP_ERROR(hipFree(it));
    }

    //Freeing the device memory allocated for the General Batched GEMM Pointer Arrays
    if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
    {
        for(int i = 0; i < block_count; i++)
        {
            CHECK_HIP_ERROR(hipFree(dda[i]));
            CHECK_HIP_ERROR(hipFree(ddb[i]));
            CHECK_HIP_ERROR(hipFree(ddc[i]));
            CHECK_HIP_ERROR(hipFree(ddd[i]));
        }
    }

    if(dWorkspace != nullptr)
        delete dWorkspace;
    if(userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(userArgs));
    if(d_userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(d_userArgs));

    // Explicitly destroy opaque handles to avoid leaks.
    for(auto& h : matA)
        if(h)
            (void)hipblasLtMatrixLayoutDestroy(h);
    for(auto& h : matB)
        if(h)
            (void)hipblasLtMatrixLayoutDestroy(h);
    for(auto& h : matC)
        if(h)
            (void)hipblasLtMatrixLayoutDestroy(h);
    for(auto& h : matD)
        if(h)
            (void)hipblasLtMatrixLayoutDestroy(h);

    for(auto& block : matmul)
        for(auto& h : block)
            if(h)
                (void)hipblasLtMatmulDescDestroy(h);

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_end));
}
