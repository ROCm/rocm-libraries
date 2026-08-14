/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// The implementation of the rocblaslt<->Tensile interface layer.

#include "rocblaslt.h"

/*****************************************************************************
 * This is the only file in rocblaslt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

#include "Debug.hpp"
#include "include/check_numerics_matrix.hpp"
#include "rocblaslt-types.h"
#include "rocblaslt_mat_utils.hpp"
#include "rocblaslt_secure_env.hpp"
#include "tensile_host.hpp"

#ifdef HIPBLASLT_USE_ROCROLLER
#include "rocroller_host.hpp"
#endif

#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define HIPBLASLT_LIB_PATH "/opt/rocm/lib"

#ifdef ENABLE_ROCTX
#include <roctracer/roctx.h>
#endif

#define INTERNAL_HIPHOSTMEM_SIZE 32768

RocblasltContractionProblem::RocblasltContractionProblem(hipblasOperation_t     trans_a,
                                                         hipblasOperation_t     trans_b,
                                                         int64_t                m,
                                                         int64_t                n,
                                                         int64_t                k,
                                                         const void*            alpha,
                                                         hipDataType            a_type,
                                                         const void*            A,
                                                         const void* const*     batch_A,
                                                         int64_t                ld_a,
                                                         int64_t                batch_stride_a,
                                                         int64_t                batch_offset_a,
                                                         hipDataType            b_type,
                                                         const void*            B,
                                                         const void* const*     batch_B,
                                                         int64_t                ld_b,
                                                         int64_t                batch_stride_b,
                                                         int64_t                batch_offset_b,
                                                         const void*            beta,
                                                         hipDataType            c_type,
                                                         const void*            C,
                                                         const void* const*     batch_C,
                                                         int64_t                ld_c,
                                                         int64_t                batch_stride_c,
                                                         int64_t                batch_offset_c,
                                                         hipDataType            d_type,
                                                         void*                  D,
                                                         void* const*           batch_D,
                                                         int64_t                ld_d,
                                                         int64_t                batch_stride_d,
                                                         int64_t                batch_offset_d,
                                                         void*                  E,
                                                         void* const*           batch_E,
                                                         int64_t                ld_e,
                                                         int64_t                batch_stride_e,
                                                         int64_t                batch_count,
                                                         bool                   strided_batch,
                                                         bool                   grouped_gemm,
                                                         bool                   gradient,
                                                         rocblaslt_compute_type compute_type,
                                                         hipDataType            scale_type,
                                                         const void*            bias,
                                                         const void*            scaleA,
                                                         const void*            scaleB,
                                                         const void*            scaleC,
                                                         const void*            scaleD,
                                                         const void*            scaleE,
                                                         const void*            scaleAlphaVec,
                                                         ScalingFormat          scaleAType,
                                                         ScalingFormat          scaleBType,
                                                         hipDataType            bias_type,
                                                         hipDataType            aux_type,
                                                         rocblaslt_epilogue     epilogue,
                                                         void*                  amaxD,
                                                         void*                  workspace,
                                                         size_t                 workspaceSize,
                                                         float                  act0,
                                                         float                  act1,
                                                         hipStream_t            stream,
                                                         void*                  Synchronizer,
                                                         bool                   swizzleA,
                                                         bool                   swizzleB,
                                                         hipblasLtBatchMode_t   batchMode,
                                                         int32_t                bias_stride,
                                                         int32_t                streamk_tile_scheduling_ext,
                                                         int32_t                sm_count_target)
    : trans_a(trans_a)
    , trans_b(trans_b)
    , m(m)
    , n(n)
    , k(k)
    , alpha(alpha)
    , a_type(a_type)
    , A(A)
    , batch_A(batch_A)
    , row_stride_a(1)
    , col_stride_a(ld_a)
    , batch_stride_a(batch_stride_a)
    , batch_offset_a(batch_offset_a)
    , b_type(b_type)
    , B(B)
    , batch_B(batch_B)
    , row_stride_b(1)
    , col_stride_b(ld_b)
    , batch_stride_b(batch_stride_b)
    , batch_offset_b(batch_offset_b)
    , beta(beta)
    , c_type(c_type)
    , C(C)
    , batch_C(batch_C)
    , row_stride_c(1)
    , col_stride_c(ld_c)
    , batch_stride_c(batch_stride_c)
    , batch_offset_c(batch_offset_c)
    , d_type(d_type)
    , D(D)
    , batch_D(batch_D)
    , row_stride_d(1)
    , col_stride_d(ld_d)
    , batch_stride_d(batch_stride_d)
    , batch_offset_d(batch_offset_d)
    , E(E)
    , batch_E(batch_E)
    , row_stride_e(1)
    , col_stride_e(ld_e)
    , batch_stride_e(batch_stride_e)
    , batch_count(batch_count)
    , strided_batch(strided_batch)
    , grouped_gemm(grouped_gemm)
    , gradient(gradient)
    , compute_type(compute_type)
    , bias(bias)
    , scale_type(scale_type)
    , scaleA(scaleA)
    , scaleB(scaleB)
    , scaleC(scaleC)
    , scaleD(scaleD)
    , scaleE(scaleE)
    , scaleAlphaVec(scaleAlphaVec)
    , scaleAType(scaleAType)
    , scaleBType(scaleBType)
    , bias_type(bias_type)
    , aux_type(aux_type)
    , epilogue(epilogue)
    , amaxD(amaxD)
    , workspace(workspace)
    , workspaceSize(workspaceSize)
    , act0(act0)
    , act1(act1)
    , stream(stream)
    , Synchronizer(Synchronizer)
    , swizzleA(swizzleA)
    , swizzleB(swizzleB)
    , batchMode(batchMode)
    , bias_stride(bias_stride)
    , streamk_tile_scheduling_ext(streamk_tile_scheduling_ext)
    , sm_count_target(sm_count_target)
{
    if(this->bias_type == HIPBLASLT_DATATYPE_INVALID)
    {
        // FIXME: Currently the default bias_type is set to match the yamls' configuration, should add the default type when the yamls are fixed.
        if(this->compute_type == rocblaslt_compute_i32)
        {
            this->bias_type = HIP_R_32I;
        }
        else if(this->compute_type == rocblaslt_compute_f32_fast_xf32)
        {
            this->bias_type = HIP_R_32F;
        }
        else if((this->a_type == HIP_R_8F_E4M3_FNUZ || this->a_type == HIP_R_8F_E5M2_FNUZ)
                && (this->b_type == HIP_R_8F_E4M3_FNUZ || this->b_type == HIP_R_8F_E5M2_FNUZ))
        {
            if(this->d_type == HIP_R_32F || this->d_type == HIP_R_16BF)
                this->bias_type = HIP_R_16BF;
            else if(this->d_type == HIP_R_16F)
                this->bias_type = HIP_R_16F;
            else //more default cases once support C != D
                this->bias_type = HIP_R_16F;
        }
        else if((this->a_type == HIP_R_8F_E4M3 || this->a_type == HIP_R_8F_E5M2)
                && (this->b_type == HIP_R_8F_E4M3 || this->b_type == HIP_R_8F_E5M2))
        {
            if(this->d_type == HIP_R_32F || this->d_type == HIP_R_16BF)
                this->bias_type = HIP_R_16BF;
            else if(this->d_type == HIP_R_16F)
                this->bias_type = HIP_R_16F;
            else //more default cases once support C != D
                this->bias_type = HIP_R_16F;
        }
        else
        {
            this->bias_type = this->d_type;
        }
    }

    if(this->aux_type == HIPBLASLT_DATATYPE_INVALID)
    {
        this->aux_type = this->d_type;
    }
}

namespace
{
    template <typename T>
    inline const T& read_scalar_from_ptr(const void* ptr)
    {
        return *(reinterpret_cast<const T*>(ptr));
    }

    // Classify alpha/beta via its storage type (alphaBetaType), not the matrix type.
    static TensileLite::ScalarValue get_scalar_value_from_void_ptr(const void*      ptr,
                                                                   rocisa::DataType type)
    {
        if(!ptr)
            return TensileLite::ScalarValue::Any; // Safety check

        switch(type)
        {
        case rocisa::DataType::ComplexDouble:
            return TensileLite::toScalarValueEnum(
                *reinterpret_cast<const hipblaslt_complex_double*>(ptr));
        case rocisa::DataType::ComplexFloat:
            return TensileLite::toScalarValueEnum(
                *reinterpret_cast<const hipblaslt_complex_float*>(ptr));
        case rocisa::DataType::Double:
            return TensileLite::toScalarValueEnum(*reinterpret_cast<const double*>(ptr));
        case rocisa::DataType::Int32:
            return TensileLite::toScalarValueEnum(*reinterpret_cast<const int32_t*>(ptr));
        case rocisa::DataType::Half:
            return TensileLite::toScalarValueEnum(*reinterpret_cast<const hipblasLtHalf*>(ptr));
        case rocisa::DataType::Float:
        case rocisa::DataType::XFloat32:
            return TensileLite::toScalarValueEnum(*reinterpret_cast<const float*>(ptr));
        default:
            throw std::runtime_error(
                "get_scalar_value_from_void_ptr: unsupported alpha/beta storage type.");
        }
    }

    static void assignAlphaBeta(rocisa::DataType computeType,
                                rocisa::DataType typeA,
                                const void*      alphaPtr,
                                const void*      betaPtr,
                                double*          alpha,
                                double*          beta)
    {

        if(typeA == rocisa::DataType::ComplexFloat || typeA == rocisa::DataType::ComplexDouble)
        {
            if(typeA == rocisa::DataType::ComplexDouble)
            {
                const hipblaslt_complex_double alpha_val
                    = read_scalar_from_ptr<hipblaslt_complex_double>(alphaPtr);
                const hipblaslt_complex_double beta_val
                    = read_scalar_from_ptr<hipblaslt_complex_double>(betaPtr);
                // Calculate magnitude and assign to double pointers
                *alpha = std::abs(alpha_val);
                *beta  = std::abs(beta_val);
            }
            else // typeA == rocisa::DataType::ComplexFloat
            {
                const hipblaslt_complex_float alpha_val
                    = read_scalar_from_ptr<hipblaslt_complex_float>(alphaPtr);
                const hipblaslt_complex_float beta_val
                    = read_scalar_from_ptr<hipblaslt_complex_float>(betaPtr);
                // Calculate magnitude and assign to double pointers
                *alpha = std::abs(alpha_val);
                *beta  = std::abs(beta_val);
            }
        }
        else
        {
            switch(computeType)
            {
            case rocisa::DataType::Float:
            case rocisa::DataType::XFloat32:
            case rocisa::DataType::BFloat16:
            case rocisa::DataType::Float8_fnuz:
            case rocisa::DataType::BFloat8_fnuz:
            case rocisa::DataType::Float8:
            case rocisa::DataType::BFloat8:
            {
                *alpha = read_scalar_from_ptr<float>(alphaPtr);
                *beta  = read_scalar_from_ptr<float>(betaPtr);
                break;
            }
            case rocisa::DataType::Double:
            {
                *alpha = read_scalar_from_ptr<double>(alphaPtr);
                *beta  = read_scalar_from_ptr<double>(betaPtr);
                break;
            }
            case rocisa::DataType::Half:
            {
                *alpha = read_scalar_from_ptr<hipblasLtHalf>(alphaPtr);
                *beta  = read_scalar_from_ptr<hipblasLtHalf>(betaPtr);
                break;
            }
            case rocisa::DataType::Int32:
            {
                *alpha = read_scalar_from_ptr<int32_t>(alphaPtr);
                *beta  = read_scalar_from_ptr<int32_t>(betaPtr);
                break;
            }
            default:
                throw std::runtime_error("Unsupported alpha/beta conversion type.");
            }
        }
    }
    inline TensileLite::ActivationType getTensileActivationType(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_RELU:
        case ROCBLASLT_EPILOGUE_RELU_BIAS:
        case ROCBLASLT_EPILOGUE_RELU_AUX:
        case ROCBLASLT_EPILOGUE_RELU_AUX_BIAS:
            return TensileLite::ActivationType::Relu;
            break;
        case ROCBLASLT_EPILOGUE_GELU:
        case ROCBLASLT_EPILOGUE_GELU_BIAS:
        case ROCBLASLT_EPILOGUE_GELU_AUX:
        case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
            return TensileLite::ActivationType::Gelu;
            break;
        case ROCBLASLT_EPILOGUE_DGELU:
        case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
            return TensileLite::ActivationType::DGelu;
        case ROCBLASLT_EPILOGUE_DRELU:
        case ROCBLASLT_EPILOGUE_DRELU_BGRAD:
            return TensileLite::ActivationType::DRelu;
        case ROCBLASLT_EPILOGUE_SWISH_EXT:
        case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
            return TensileLite::ActivationType::Silu;
        case ROCBLASLT_EPILOGUE_CLAMP_EXT:
        case ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
        case ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT:
        case ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT:
            return TensileLite::ActivationType::Clamp;
        case ROCBLASLT_EPILOGUE_SIGMOID:
            return TensileLite::ActivationType::Sigmoid;
        case ROCBLASLT_EPILOGUE_BIAS:
        case ROCBLASLT_EPILOGUE_DEFAULT:
        case ROCBLASLT_EPILOGUE_BGRADA:
        case ROCBLASLT_EPILOGUE_BGRADB:
            break;
        }
        return TensileLite::ActivationType::None;
    }

    inline TensileLite::ContractionProblemGemm::TENSOR getBiasSrc(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_BGRADA:
            return TensileLite::ContractionProblemGemm::TENSOR::A;
            break;
        case ROCBLASLT_EPILOGUE_BGRADB:
            return TensileLite::ContractionProblemGemm::TENSOR::B;
            break;
        default:
            break;
        }
        return TensileLite::ContractionProblemGemm::TENSOR::D;
    }

    inline bool tensileUseBias(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_RELU_BIAS:
        case ROCBLASLT_EPILOGUE_GELU_BIAS:
        case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
        case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
        case ROCBLASLT_EPILOGUE_DRELU_BGRAD:
        case ROCBLASLT_EPILOGUE_BIAS:
        case ROCBLASLT_EPILOGUE_BGRADA:
        case ROCBLASLT_EPILOGUE_BGRADB:
        case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
        case ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT:
            return true;
            break;
        default:
            return false;
            break;
        }
        return false;
    }

    rocisa::DataType hip2TensileType(hipDataType type)
    {
        switch(type)
        {
        case HIP_R_32F:
            return rocisa::DataType::Float;
        case HIP_R_16F:
            return rocisa::DataType::Half;
        case HIP_R_64F:
            return rocisa::DataType::Double;
        case HIP_R_16BF:
            return rocisa::DataType::BFloat16;
        case HIP_R_8F_E4M3_FNUZ:
            return rocisa::DataType::Float8_fnuz;
        case HIP_R_8F_E5M2_FNUZ:
            return rocisa::DataType::BFloat8_fnuz;
        case HIP_R_8F_E4M3:
            return rocisa::DataType::Float8;
        case HIP_R_8F_E5M2:
            return rocisa::DataType::BFloat8;
        case HIP_R_8I:
            return rocisa::DataType::Int8;
        case HIP_R_32I:
            return rocisa::DataType::Int32;
        case HIP_C_32F:
            return rocisa::DataType::ComplexFloat;
        case HIP_C_64F:
            return rocisa::DataType::ComplexDouble;    
        case HIP_R_6F_E2M3:
            return rocisa::DataType::Float6;
        case HIP_R_6F_E3M2:
            return rocisa::DataType::BFloat6;
        case HIP_R_4F_E2M1:
            return rocisa::DataType::Float4;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return rocisa::DataType::None;
    }

    hipDataType tensile2HipType(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return HIP_R_32F;
        case rocisa::DataType::Half:
            return HIP_R_16F;
        case rocisa::DataType::Double:
            return HIP_R_64F;
        case rocisa::DataType::BFloat16:
            return HIP_R_16BF;
        case rocisa::DataType::Float8_fnuz:
            return HIP_R_8F_E4M3_FNUZ;
        case rocisa::DataType::BFloat8_fnuz:
            return HIP_R_8F_E5M2_FNUZ;
        case rocisa::DataType::Float8:
            return HIP_R_8F_E4M3;
        case rocisa::DataType::BFloat8:
            return HIP_R_8F_E5M2;
        case rocisa::DataType::Int8:
            return HIP_R_8I;
        case rocisa::DataType::Int32:
            return HIP_R_32I;
        case rocisa::DataType::ComplexFloat:
            return HIP_C_32F;
        case rocisa::DataType::ComplexDouble:
            return HIP_C_64F;    
        case rocisa::DataType::Float6:
            return static_cast<hipDataType>(HIP_R_6F_E2M3);
        case rocisa::DataType::BFloat6:
            return static_cast<hipDataType>(HIP_R_6F_E3M2);
        case rocisa::DataType::Float4:
            return static_cast<hipDataType>(HIP_R_4F_E2M1);
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return HIP_R_32F;
    }

    // Adapter: extract shape from a Tensile problem and forward to scan_D.
    rocblaslt_status maybe_check_numerics_problem(
        rocblaslt_handle                           handle,
        hipStream_t                                stream,
        const TensileLite::ContractionProblemGemm& prob,
        const void*                                D,
        uint32_t                                   call_id)
    {
        if(!handle || !handle->check_numerics || !D || call_id == 0)
            return rocblaslt_status_success;

        const int64_t m        = prob.c().sizes()[0];
        const int64_t n        = prob.c().sizes()[1];
        const int64_t ldd      = prob.d().strides()[1];
        const int64_t stride_d = prob.d().strides()[2];
        const int32_t batch    = static_cast<int32_t>(prob.batchSize(0));
        // Tensile builds column-major strides; row-major D has strides()[0] != 1.
        const bool        row_major = (prob.d().strides()[0] != 1);
        const hipDataType type_d    = tensile2HipType(prob.d().dataType());

        return hipblaslt_check_numerics_scan_D(handle,
                                               stream,
                                               call_id,
                                               m, n, batch,
                                               type_d,
                                               D,
                                               ldd,
                                               stride_d,
                                               row_major);
    }

    // Post-launch hook shared by single- and grouped-GEMM. Acquires one
    // call_id then invokes enumerate(call_id, scan_one) for each sub-problem.
    template <typename Enumerate>
    rocblaslt_status check_numerics_post_launch(rocblaslt_handle handle,
                                                hipStream_t      stream,
                                                Enumerate&&      enumerate)
    {
        const uint32_t call_id = hipblaslt_check_numerics_begin_call(handle);
        if(call_id == 0)
            return rocblaslt_status_success;
        rocblaslt_status agg = rocblaslt_status_success;
        enumerate(call_id,
                  [&](const TensileLite::ContractionProblemGemm& prob, const void* d) {
                      const auto st = maybe_check_numerics_problem(
                          handle, stream, prob, d, call_id);
                      if(st != rocblaslt_status_success && agg == rocblaslt_status_success)
                          agg = st;
                  });
        return agg;
    }

    rocisa::DataType roc2TensileType(rocblaslt_compute_type type, bool fallback = true)
    {
        switch(type)
        {
        case rocblaslt_compute_f16: // setting compute_type to f16_r will fallback to f32_r
            return fallback ? rocisa::DataType::Float : rocisa::DataType::Half;
        case rocblaslt_compute_f32:
        case rocblaslt_compute_f32_fast_xf32:
        case rocblaslt_compute_f32_fast_f16:
        case rocblaslt_compute_f32_fast_bf16:
        case rocblaslt_compute_f32_fast_f8_fnuz:
        case rocblaslt_compute_f32_fast_bf8_fnuz:
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
        case rocblaslt_compute_f32_fast_f8:
        case rocblaslt_compute_f32_fast_bf8:
        case rocblaslt_compute_f32_fast_f8bf8:
        case rocblaslt_compute_f32_fast_bf8f8:
            return rocisa::DataType::Float;
        case rocblaslt_compute_f64:
            return rocisa::DataType::Double;
        case rocblaslt_compute_i32:
            return rocisa::DataType::Int32;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return rocisa::DataType::None;
    }

    inline const rocisa::DataType
        roc2TensileComputeInputTypeA(const rocisa::DataType&       typeA,
                                     const rocisa::DataType&       typeB,
                                     const rocblaslt_compute_type& typeCompute)
    {
        switch(typeCompute)
        {
        case rocblaslt_compute_f32_fast_f16:
            return rocisa::DataType::Half;
        case rocblaslt_compute_f32_fast_bf16:
            return rocisa::DataType::BFloat16;
        case rocblaslt_compute_f32_fast_f8_fnuz:
            return rocisa::DataType::Float8_fnuz;
        case rocblaslt_compute_f32_fast_bf8_fnuz:
            return rocisa::DataType::BFloat8_fnuz;
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
            return rocisa::DataType::Float8_fnuz;
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
            return rocisa::DataType::BFloat8_fnuz;
        case rocblaslt_compute_f32_fast_f8:
            return rocisa::DataType::Float8;
        case rocblaslt_compute_f32_fast_bf8:
            return rocisa::DataType::BFloat8;
        case rocblaslt_compute_f32_fast_f8bf8:
            return rocisa::DataType::Float8;
        case rocblaslt_compute_f32_fast_bf8f8:
            return rocisa::DataType::BFloat8;
        default:;
        }

        if(typeA == rocisa::DataType::Float8_fnuz && typeB == rocisa::DataType::BFloat8_fnuz)
        {
            return rocisa::DataType::Float8_fnuz;
        }
        else if(typeA == rocisa::DataType::BFloat8_fnuz && typeB == rocisa::DataType::Float8_fnuz)
        {
            return rocisa::DataType::BFloat8_fnuz;
        }

        if(typeA == rocisa::DataType::Float8 && typeB == rocisa::DataType::BFloat8)
        {
            return rocisa::DataType::Float8;
        }
        else if(typeA == rocisa::DataType::BFloat8 && typeB == rocisa::DataType::Float8)
        {
            return rocisa::DataType::BFloat8;
        }

        if(typeA == rocisa::DataType::Float8 || typeA == rocisa::DataType::BFloat8 || typeA == rocisa::DataType::Float8_fnuz || typeA == rocisa::DataType::BFloat8_fnuz || typeA == rocisa::DataType::Float6 || typeA == rocisa::DataType::BFloat6 || typeA == rocisa::DataType::Float4) return typeA;
        

        return TensileLite::DataTypeInfo::Get(typeA).elementSize
                       <= TensileLite::DataTypeInfo::Get(typeB).elementSize
                   ? typeA
                   : typeB;
    }
	
	inline const rocisa::DataType
        roc2TensileComputeInputTypeB(const rocisa::DataType&       typeA,
                                     const rocisa::DataType&       typeB,
                                     const rocblaslt_compute_type& typeCompute)
    {
        switch(typeCompute)
        {
        case rocblaslt_compute_f32_fast_f16:
            return rocisa::DataType::Half;
        case rocblaslt_compute_f32_fast_bf16:
            return rocisa::DataType::BFloat16;
        case rocblaslt_compute_f32_fast_f8_fnuz:
            return rocisa::DataType::Float8_fnuz;
        case rocblaslt_compute_f32_fast_bf8_fnuz:
            return rocisa::DataType::BFloat8_fnuz;
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
            return rocisa::DataType::BFloat8_fnuz;
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
            return rocisa::DataType::Float8_fnuz;
        case rocblaslt_compute_f32_fast_f8:
            return rocisa::DataType::Float8;
        case rocblaslt_compute_f32_fast_bf8:
            return rocisa::DataType::BFloat8;
        case rocblaslt_compute_f32_fast_f8bf8:
            return rocisa::DataType::BFloat8;
        case rocblaslt_compute_f32_fast_bf8f8:
            return rocisa::DataType::Float8;
        default:;
        }

        if(typeA == rocisa::DataType::Float8_fnuz && typeB == rocisa::DataType::BFloat8_fnuz)
        {
            return rocisa::DataType::BFloat8_fnuz;
        }
        else if(typeA == rocisa::DataType::BFloat8_fnuz && typeB == rocisa::DataType::Float8_fnuz)
        {
            return rocisa::DataType::Float8_fnuz;
        }

        if(typeA == rocisa::DataType::Float8 && typeB == rocisa::DataType::BFloat8)
        {
            return rocisa::DataType::BFloat8;
        }
        else if(typeA == rocisa::DataType::BFloat8 && typeB == rocisa::DataType::Float8)
        {
            return rocisa::DataType::Float8;
        }

        if(typeB == rocisa::DataType::Float8 || typeB == rocisa::DataType::BFloat8 || typeB == rocisa::DataType::Float8_fnuz || typeB == rocisa::DataType::BFloat8_fnuz ||typeB == rocisa::DataType::Float6 || typeB == rocisa::DataType::BFloat6 || typeB == rocisa::DataType::Float4) return typeB;
        

        return TensileLite::DataTypeInfo::Get(typeA).elementSize
                       <= TensileLite::DataTypeInfo::Get(typeB).elementSize
                   ? typeA
                   : typeB;
    }

    rocblaslt_status hip2RocStatus(hipError_t status)
    {
        switch(status)
        {
        case hipSuccess:
            return rocblaslt_status_success;
        case hipErrorUnknown:
        case hipErrorRuntimeOther:
        case hipErrorInvalidDevice:
            return rocblaslt_status_internal_error;
        default:
            return rocblaslt_status_not_implemented;
        }
    }

    inline auto CreateTensileProblem(hipblasOperation_t     opA,
                                     hipblasOperation_t     opB,
                                     hipDataType            typeA,
                                     hipDataType            typeB,
                                     hipDataType            typeC,
                                     hipDataType            typeD,
                                     rocblaslt_compute_type typeCompute,
                                     float                  alpha,
                                     float                  beta,
                                     bool                   isGroupedGemm,
                                     size_t                 maxWorkspaceBytes)
    {
        auto                          typeATensile = hip2TensileType(typeA);
        auto                          typeBTensile = hip2TensileType(typeB);
        std::vector<rocisa::DataType> biasDataTypeWhiteList; // dummy
        std::vector<int>              biasSrcWhiteList; // dummy

        TensileLite::TensorOps aOps, bOps, cOps, dOps;

        if(opA == HIPBLAS_OP_C)
            aOps = {TensileLite::TensorOp::ComplexConjugate()};

        if(opB == HIPBLAS_OP_C)
            bOps = {TensileLite::TensorOp::ComplexConjugate()};

        bool isComplexInput = (typeATensile == rocisa::DataType::ComplexFloat
                               || typeATensile == rocisa::DataType::ComplexDouble);

        auto alphaBetaType = isComplexInput ? typeATensile : roc2TensileType(typeCompute);

        return TensileLite::ContractionProblemGemm::createDefaultProblem(
            (opA != HIPBLAS_OP_N),
            (opB != HIPBLAS_OP_N),
            typeATensile,
            typeBTensile,
            hip2TensileType(typeC),
            hip2TensileType(typeD),
            alphaBetaType,
            alphaBetaType,
            roc2TensileComputeInputTypeA(typeATensile, typeBTensile, typeCompute),
            roc2TensileComputeInputTypeB(typeATensile, typeBTensile, typeCompute),
            roc2TensileType(typeCompute),
            alpha,
            beta,
            false,
            false,
            biasDataTypeWhiteList,
            biasSrcWhiteList,
            isGroupedGemm,
            maxWorkspaceBytes,
            aOps,
            bOps,
            cOps,
            dOps);
    }

    const char* tensileComputeInputType_to_bench_string(rocisa::DataType typeCompute,
                                                        rocisa::DataType F32XdlMathOp,
                                                        rocisa::DataType typeComputeInputA,
                                                        rocisa::DataType typeComputeInputB,
                                                        rocisa::DataType typeA,
                                                        rocisa::DataType typeB)
    {
        switch(typeCompute)
        {
        case rocisa::DataType::Float:
            break;
        case rocisa::DataType::Double:
            return "f64_r";
            break;
        case rocisa::DataType::Int32:
            return "i32_r";
            break;
        case rocisa::DataType::ComplexFloat:
            return "f32_r";
            break;
        case rocisa::DataType::ComplexDouble:
            return "f64_r";
            break;       
        default:
            throw std::runtime_error("Unsupported type.");
        }

        if(F32XdlMathOp == rocisa::DataType::XFloat32)
        {
            return "xf32_r";
        }
        else if(typeComputeInputA == rocisa::DataType::BFloat16
                && typeComputeInputB == rocisa::DataType::BFloat16
                && (typeA == rocisa::DataType::Half && typeB == rocisa::DataType::Half
                    || typeA == rocisa::DataType::Float && typeB == rocisa::DataType::Float))
        {
            return "f32_bf16_r";
        }
        else if(typeComputeInputA == rocisa::DataType::Half
                && typeComputeInputB == rocisa::DataType::Half
                && (typeA == rocisa::DataType::Float8_fnuz && typeB == rocisa::DataType::Half
                    || typeA == rocisa::DataType::Half && typeB == rocisa::DataType::Float8_fnuz))
        {
            return "f32_f16_r";
        }
        else
        {
            return "f32_r";
        }
    }

    const char* tensileComputeInputType_to_profile_string(rocisa::DataType typeCompute,
                                                          rocisa::DataType F32XdlMathOp,
                                                          rocisa::DataType typeComputeInputA,
                                                          rocisa::DataType typeComputeInputB,
                                                          rocisa::DataType typeA,
                                                          rocisa::DataType typeB)
    {
        switch(typeCompute)
        {
        case rocisa::DataType::Float:
            break;
        case rocisa::DataType::Double:
            return "c_f64_r";
            break;
        case rocisa::DataType::Int32:
            return "c_i32_r";
            break;
        case rocisa::DataType::ComplexFloat:
            return "c_f32_r";
            break;
        case rocisa::DataType::ComplexDouble:
            return "c_f64_r";
            break;    
        default:
            throw std::runtime_error("Unsupported type.");
        }

        if(F32XdlMathOp == rocisa::DataType::XFloat32)
        {
            return "c_xf32_r";
        }
        else if(typeComputeInputA == rocisa::DataType::BFloat16
                && typeComputeInputB == rocisa::DataType::BFloat16
                && (typeA == rocisa::DataType::Half && typeB == rocisa::DataType::Half
                    || typeA == rocisa::DataType::Float && typeB == rocisa::DataType::Float))
        {
            return "c_f32_fast_bf16_r";
        }
        else if(typeComputeInputA == rocisa::DataType::Half
                && typeComputeInputB == rocisa::DataType::Half
                && (typeA == rocisa::DataType::Float8_fnuz && typeB == rocisa::DataType::Half
                    || typeA == rocisa::DataType::Half && typeB == rocisa::DataType::Float8_fnuz))
        {
            return "c_f32_fast_f16_r";
        }
        else
        {
            return "c_f32_r";
        }
    }

    const char* tensileActivationtType_to_bench_string(TensileLite::ActivationType activation)
    {
        switch(activation)
        {
        case TensileLite::ActivationType::DGelu:
        case TensileLite::ActivationType::Gelu:
            return "gelu";
            break;
        case TensileLite::ActivationType::DRelu:
        case TensileLite::ActivationType::Relu:
            return "relu";
            break;
        case TensileLite::ActivationType::Silu:
        case TensileLite::ActivationType::Swish:
            return "swish";
        case TensileLite::ActivationType::Clamp:
            return "clamp";
        case TensileLite::ActivationType::None:
        default:
            return "none";
            break;
        }
    }

    inline std::string getAlphaRealAsString(const TensileLite::ContractionInputs& inputs, bool isComplex)
    {
        return isComplex 
            ?  (std::holds_alternative<hipblaslt_complex_float>(inputs.alpha)
                ?  std::to_string(std::get<hipblaslt_complex_float>(inputs.alpha).real())
                : std::to_string(std::get<hipblaslt_complex_double>(inputs.alpha).real()))
            : ToString(inputs.alpha);
    }

    inline std::string getAlphaImagAsString(const TensileLite::ContractionInputs& inputs, bool isComplex)
    {
        return isComplex
            ?  (std::holds_alternative<hipblaslt_complex_float>(inputs.alpha)
                ?  std::to_string(std::get<hipblaslt_complex_float>(inputs.alpha).imag())
                : std::to_string(std::get<hipblaslt_complex_double>(inputs.alpha).imag()))
            : std::to_string(0.0);
    }

    inline std::string getBetaRealAsString(const TensileLite::ContractionInputs& inputs, bool isComplex)
    {
        return isComplex
            ?  (std::holds_alternative<hipblaslt_complex_float>(inputs.beta)
                ?  std::to_string(std::get<hipblaslt_complex_float>(inputs.beta).real())
                : std::to_string(std::get<hipblaslt_complex_double>(inputs.beta).real()))
            : ToString(inputs.beta);
    }

    inline std::string getBetaImagAsString(const TensileLite::ContractionInputs& inputs, bool isComplex)
    {
        return isComplex
            ?  (std::holds_alternative<hipblaslt_complex_float>(inputs.beta)
                ?  std::to_string(std::get<hipblaslt_complex_float>(inputs.beta).imag())
                : std::to_string(std::get<hipblaslt_complex_double>(inputs.beta).imag()))
            : std::to_string(0.0f);
    }


    inline void logBenchFromTensileDataGemm(const TensileLite::ContractionProblemGemm& problem,
                                            const TensileLite::ContractionInputs&      inputs,
                                            const int&     solutionIndex,
                                            bool           flush,
                                            const int32_t& rotatingBufferSize,
                                            const int32_t& coldIterations,
                                            const int32_t& hotIterations,
                                            bool           isCpp)
    {
        bool isComplexInput = (problem.a().dataType() == rocisa::DataType::ComplexFloat
                               || problem.a().dataType() == rocisa::DataType::ComplexDouble);
        auto s = log_str(
            __func__,
            "--api_method",
            isCpp ? "cpp" : "c",
            "-m",
            problem.c().sizes()[0],
            "-n",
            problem.c().sizes()[1],
            "-k",
            problem.a().sizes()[problem.boundIndices()[0].a],
            "--lda",
            problem.a().strides()[1],
            "--ldb",
            problem.b().strides()[1],
            "--ldc",
            problem.c().strides()[1],
            "--ldd",
            problem.d().strides()[1],
            problem.tensor(TensileLite::ContractionProblemGemm::TENSOR::E).strides().size()
                ? "--lde"
                : "",
            problem.tensor(TensileLite::ContractionProblemGemm::TENSOR::E).strides().size()
                ? std::to_string(
                      problem.tensor(TensileLite::ContractionProblemGemm::TENSOR::E).strides()[1])
                : "",
            "--stride_a",
            problem.a().strides()[2],
            "--stride_b",
            problem.b().strides()[2],
            "--stride_c",
            problem.c().strides()[2],
            "--stride_d",
            problem.d().strides()[2],
            problem.tensor(TensileLite::ContractionProblemGemm::TENSOR::E).strides().size()
                ? "--stride_e"
                : "",
            problem.tensor(TensileLite::ContractionProblemGemm::TENSOR::E).strides().size()
                ? std::to_string(
                      problem.tensor(TensileLite::ContractionProblemGemm::TENSOR::E).strides()[2])
                : "",
            "--alpha",
            getAlphaRealAsString(inputs, isComplexInput),
            "--alphai",
            getAlphaImagAsString(inputs, isComplexInput),
            "--beta",
            getBetaRealAsString(inputs, isComplexInput),
            "--betai",
            getBetaImagAsString(inputs, isComplexInput),
            "--transA",
            problem.transA() ? "T" : "N",
            "--transB",
            problem.transB() ? "T" : "N",
            "--batch_count",
            problem.batchSize(0),
			"--batch_mode",
			problem.batchMode(),
            "--scaleA",
            problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
            "--scaleB",
            problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
            problem.useScaleCD() ? "--scaleC" : "",
            problem.useScaleCD() ? "--scaleD" : "",
            problem.swizzleTensorA() ? "--swizzleA" : "",
            problem.swizzleTensorB() ? "--swizzleB" : "",
            problem.useScaleAlphaVec() ? "--scaleAlpha_vector" : "",
            problem.useGradient() ? "--gradient" : "",
            problem.useE() ? "--use_e" : "",
            problem.useBias() ? "--bias_vector" : "",
            problem.useBias() ? "--bias_source" : "",
            problem.useBias() ? problem.tensor(problem.biasSrc()).getName() : "",
			problem.useBias() ? "--bias_stride" : "",
			problem.useBias() ? std::to_string((problem.bias().strides())[2]) : "",
            "--a_type",
            hipDataType_to_bench_string(tensile2HipType(problem.a().dataType())),
            "--b_type",
            hipDataType_to_bench_string(tensile2HipType(problem.b().dataType())),
            "--c_type",
            hipDataType_to_bench_string(tensile2HipType(problem.c().dataType())),
            "--d_type",
            hipDataType_to_bench_string(tensile2HipType(problem.d().dataType())),
            "--scale_type",
            hipDataType_to_bench_string(tensile2HipType(problem.alphaType())),
            "--bias_type",
            hipDataType_to_bench_string(tensile2HipType(problem.bias().dataType())),
            problem.useE() ? "--aux_type" : "",
            problem.useE() ? hipDataType_to_bench_string(tensile2HipType(problem.e().dataType()))
                           : "",
            problem.getParams().gsu() ? "--splitk" : "",
            problem.getParams().gsu() ? std::to_string(problem.getParams().gsu()) : "",
            problem.getParams().wgm() ? "--wgm" : "",
            problem.getParams().wgm() ? std::to_string(problem.getParams().wgm()) : "",
            "--compute_type",
            tensileComputeInputType_to_bench_string(problem.computeType(),
                                                    problem.f32XdlMathOp(),
                                                    problem.computeInputTypeA(),
                                                    problem.computeInputTypeB(),
                                                    problem.a().dataType(),
                                                    problem.b().dataType()),
            "--algo_method",
            "index",
            "--solution_index",
            solutionIndex,
            "--activation_type",
            tensileActivationtType_to_bench_string(problem.getParams().activationEnum()),
            flush ? "--flush" : "",
            "--any_stride",
            "--rotating",
            rotatingBufferSize,
            "--cold_iters",
            coldIterations,
            "--iters",
            hotIterations);

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
            log_bench_from_str(s);
        if(rocblaslt::Debug::Instance().printLogAsMarker())
        {
            rocblaslt::Debug::Instance().logMarkerStart(s.c_str());
        }
        if(rocblaslt::Debug::Instance().benchPrintCommand())
        {
            std::cout << s << std::endl;
            rocblaslt::Debug::Instance().setBenchPrint(false);
        }
    }

    inline void logProfileFromTensileDataGemm(const TensileLite::ContractionProblemGemm& problem,
                                              const TensileLite::ContractionInputs&      inputs,
                                              const int&     solutionIndex,
                                              bool           flush,
                                              const int32_t& rotatingBufferSize,
                                              const int32_t& coldIterations,
                                              const int32_t& hotIterations,
                                              bool           isCpp)
    {
        bool isComplexInput = (problem.a().dataType() == rocisa::DataType::ComplexFloat
                               || problem.a().dataType() == rocisa::DataType::ComplexDouble);

        log_profile("matmul",
                    "M",
                    problem.c().sizes()[0],
                    "N",
                    problem.c().sizes()[1],
                    "K",
                    problem.a().sizes()[problem.boundIndices()[0].a],
                    "lda",
                    problem.a().strides()[1],
                    "ldb",
                    problem.b().strides()[1],
                    "ldc",
                    problem.c().strides()[1],
                    "ldd",
                    problem.d().strides()[1],
                    "stride_a",
                    problem.a().strides()[2],
                    "stride_b",
                    problem.b().strides()[2],
                    "stride_c",
                    problem.c().strides()[2],
                    "stride_d",
                    problem.d().strides()[2],
                    "alpha",
                    getAlphaRealAsString(inputs, isComplexInput),
                    "alphai",
                    getAlphaImagAsString(inputs, isComplexInput),
                    "beta",
                    getBetaRealAsString(inputs, isComplexInput),
                    "betai",
                    getBetaImagAsString(inputs, isComplexInput),
                    "transA",
                    problem.transA() ? "T" : "N",
                    "transB",
                    problem.transB() ? "T" : "N",
                    "batch_count",
                    problem.batchSize(0),
					"batch_mode",
					problem.batchMode(),
                    "scaleA",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
                    "scaleB",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
                    "scaleC",
                    problem.useScaleCD() ? 1 : 0,
                    "scaleD",
                    problem.useScaleCD() ? 1 : 0,
                    "swizzleA",
                    problem.swizzleTensorA() ? "true" : "false",
                    "swizzleB",
                    problem.swizzleTensorB() ? "true" : "false",
                    "scaleAlpha_vector",
                    problem.useScaleAlphaVec() ? "true" : "false",
                    "gradient",
                    problem.useGradient() ? "true" : "false",
                    "use_e",
                    problem.useE() ? "true" : "false",
                    "bias_vector",
                    problem.useBias() ? "true" : "false",
                    "bias_source",
                    problem.useBias() ? problem.tensor(problem.biasSrc()).getName() : "d",
					"bias_stride",
					problem.useBias() ? (problem.bias().strides())[2] : 0,
                    "a_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.a().dataType())),
                    "b_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.b().dataType())),
                    "c_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.c().dataType())),
                    "d_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.d().dataType())),
                    "scale_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.alphaType())),
                    "bias_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.bias().dataType())),
                    "aux_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.e().dataType())),
                    "compute_type",
                    tensileComputeInputType_to_profile_string(problem.computeType(),
                                                              problem.f32XdlMathOp(),
                                                              problem.computeInputTypeA(),
                                                              problem.computeInputTypeB(),
                                                              problem.a().dataType(),
                                                              problem.b().dataType()),
                    "algo_method",
                    2,
                    "solution_index",
                    solutionIndex,
                    "activation_type",
                    tensileActivationtType_to_bench_string(problem.getParams().activationEnum()),
                    "flush",
                    flush ? "true" : "false",
                    "any_stride",
                    "true",
                    "rotating",
                    rotatingBufferSize,
                    "cold_iters",
                    coldIterations,
                    "iters",
                    hotIterations);
    }

    inline void
        logExtendedProfileFromTensileDataGemm(const TensileLite::ContractionProblemGemm& problem,
                                              const TensileLite::ContractionInputs&      inputs,
                                              const int&         solutionIndex,
                                              const std::string& kernelName,
                                              const std::string& solutionName,
                                              bool               flush,
                                              const int32_t&     rotatingBufferSize,
                                              const int32_t&     coldIterations,
                                              const int32_t&     hotIterations,
                                              bool               isCpp)
    {
        bool isComplexInput = (problem.a().dataType() == rocisa::DataType::ComplexFloat
                               || problem.a().dataType() == rocisa::DataType::ComplexDouble);
                               
        log_profile("matmul",
                    "M",
                    problem.c().sizes()[0],
                    "N",
                    problem.c().sizes()[1],
                    "K",
                    problem.a().sizes()[problem.boundIndices()[0].a],
                    "lda",
                    problem.a().strides()[1],
                    "ldb",
                    problem.b().strides()[1],
                    "ldc",
                    problem.c().strides()[1],
                    "ldd",
                    problem.d().strides()[1],
                    "stride_a",
                    problem.a().strides()[2],
                    "stride_b",
                    problem.b().strides()[2],
                    "stride_c",
                    problem.c().strides()[2],
                    "stride_d",
                    problem.d().strides()[2],
                    "alpha",
                    getAlphaRealAsString(inputs, isComplexInput),
                    "alphai",
                    getAlphaImagAsString(inputs, isComplexInput),
                    "beta",
                    getBetaRealAsString(inputs, isComplexInput),
                    "betai",
                    getBetaImagAsString(inputs, isComplexInput),
                    "transA",
                    problem.transA() ? "T" : "N",
                    "transB",
                    problem.transB() ? "T" : "N",
                    "batch_count",
                    problem.batchSize(0),
					"batch_mode",
					problem.batchMode(),
                    "scaleA",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
                    "scaleB",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
                    "scaleC",
                    problem.useScaleCD() ? 1 : 0,
                    "scaleD",
                    problem.useScaleCD() ? 1 : 0,
                    "swizzleA",
                    problem.swizzleTensorA() ? "true" : "false",
                    "swizzleB",
                    problem.swizzleTensorB() ? "true" : "false",
                    "scaleAlpha_vector",
                    problem.useScaleAlphaVec() ? "true" : "false",
                    "gradient",
                    problem.useGradient() ? "true" : "false",
                    "use_e",
                    problem.useE() ? "true" : "false",
                    "bias_vector",
                    problem.useBias() ? "true" : "false",
                    "bias_source",
                    problem.useBias() ? problem.tensor(problem.biasSrc()).getName() : "d",
					"bias_stride",
					problem.useBias() ? (problem.bias().strides())[2] : 0,
                    "a_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.a().dataType())),
                    "b_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.b().dataType())),
                    "c_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.c().dataType())),
                    "d_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.d().dataType())),
                    "scale_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.alphaType())),
                    "bias_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.bias().dataType())),
                    "aux_type",
                    hipDataType_to_bench_string(tensile2HipType(problem.e().dataType())),
                    "compute_type",
                    tensileComputeInputType_to_profile_string(problem.computeType(),
                                                              problem.f32XdlMathOp(),
                                                              problem.computeInputTypeA(),
                                                              problem.computeInputTypeB(),
                                                              problem.a().dataType(),
                                                              problem.b().dataType()),
                    "activation_type",
                    tensileActivationtType_to_bench_string(problem.getParams().activationEnum()),
                    "flush",
                    flush ? "true" : "false",
                    "any_stride",
                    "true",
                    "rotating",
                    rotatingBufferSize,
                    "cold_iters",
                    coldIterations,
                    "iters",
                    hotIterations,
                    "solution_index",
                    solutionIndex,
                    "solution_Name",
                    solutionName,
                    "kernel_name",
                    kernelName);
    }

    inline void
        logBenchFromTensileDataGemm(const TensileLite::ContractionProblemGroupedGemm& problem,
                                    const TensileLite::ContractionGroupedInputs&      inputs,
                                    const int&                                        solutionIndex,
                                    bool                                              flush,
                                    const int32_t& rotatingBufferSize,
                                    const int32_t& coldIterations,
                                    const int32_t& hotIterations,
                                    bool           isCpp)
    {
        size_t            gemmCount = problem.gemms.size();
        std::stringstream grouped_gemm_bench_string;
        for(int i = 0; i < gemmCount; ++i)
        {
            grouped_gemm_bench_string << " -m " << problem.gemms[i].c().sizes()[0];
            grouped_gemm_bench_string << " -n " << problem.gemms[i].c().sizes()[1];
            grouped_gemm_bench_string
                << " -k " << problem.gemms[i].a().sizes()[problem.gemms[i].boundIndices()[0].a];
            grouped_gemm_bench_string << " --lda " << problem.gemms[i].a().strides()[1];
            grouped_gemm_bench_string << " --ldb " << problem.gemms[i].b().strides()[1];
            grouped_gemm_bench_string << " --ldc " << problem.gemms[i].c().strides()[1];
            grouped_gemm_bench_string << " --ldd " << problem.gemms[i].d().strides()[1];
            if(problem.gemms[i]
                   .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                   .strides()
                   .size())
                grouped_gemm_bench_string
                    << " --lde "
                    << problem.gemms[i]
                           .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                           .strides()[1];
            grouped_gemm_bench_string << " --stride_a " << problem.gemms[i].a().strides()[2];
            grouped_gemm_bench_string << " --stride_b " << problem.gemms[i].b().strides()[2];
            grouped_gemm_bench_string << " --stride_c " << problem.gemms[i].c().strides()[2];
            grouped_gemm_bench_string << " --stride_d " << problem.gemms[i].d().strides()[2];
            if(problem.gemms[i]
                   .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                   .strides()
                   .size())
                grouped_gemm_bench_string
                    << " --stride_e "
                    << problem.gemms[i]
                           .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                           .strides()[2];
        }
        auto s = log_str(
            __func__,
            "--api_method",
            isCpp ? "cpp" : "c",
            "--grouped_gemm",
            grouped_gemm_bench_string.str(),
            "--alpha",
            ToString(inputs.grouped[0].alpha),
            "--beta",
            ToString(inputs.grouped[0].beta),
            "--transA",
            problem.gemms[0].transA() ? "T" : "N",
            "--transB",
            problem.gemms[0].transB() ? "T" : "N",
            "--batch_count",
            problem.gemms[0].batchSize(0),
            "--scaleA",
            problem.gemms[0].useScaleAB().empty()
                ? 0
                : (problem.gemms[0].useScaleAB() == "Vector" ? 2 : 1),
            "--scaleB",
            problem.gemms[0].useScaleAB().empty()
                ? 0
                : (problem.gemms[0].useScaleAB() == "Vector" ? 2 : 1),
            problem.gemms[0].useScaleCD() ? "--scaleC" : "",
            problem.gemms[0].useScaleCD() ? "--scaleD" : "",
            problem.gemms[0].swizzleTensorA() ? "--swizzleA" : "",
            problem.gemms[0].swizzleTensorB() ? "--swizzleB" : "",
            problem.gemms[0].useScaleAlphaVec() ? "--scaleAlpha_vector" : "",
            problem.gemms[0].useGradient() ? "--gradient" : "",
            problem.gemms[0].useE() ? "--use_e" : "",
            problem.gemms[0].useBias() ? "--bias_vector" : "",
            problem.gemms[0].useBias() ? "--bias_source" : "",
            problem.gemms[0].useBias()
                ? problem.gemms[0].tensor(problem.gemms[0].biasSrc()).getName()
                : "",
            "--a_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].a().dataType())),
            "--b_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].b().dataType())),
            "--c_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].c().dataType())),
            "--d_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].d().dataType())),
            "--scale_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].alphaType())),
            "--bias_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].bias().dataType())),
            problem.gemms[0].useE() ? "--aux_type" : "",
            problem.gemms[0].useE()
                ? hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].e().dataType()))
                : "",
            problem.gemms[0].getParams().gsu() ? "--splitk" : "",
            problem.gemms[0].getParams().gsu() ? std::to_string(problem.gemms[0].getParams().gsu())
                                               : "",
            problem.gemms[0].getParams().wgm() ? "--wgm" : "",
            problem.gemms[0].getParams().wgm() ? std::to_string(problem.gemms[0].getParams().wgm())
                                               : "",
            "--compute_type",
            tensileComputeInputType_to_bench_string(problem.gemms[0].computeType(),
                                                    problem.gemms[0].f32XdlMathOp(),
                                                    problem.gemms[0].computeInputTypeA(),
                                                    problem.gemms[0].computeInputTypeB(),
                                                    problem.gemms[0].a().dataType(),
                                                    problem.gemms[0].b().dataType()),
            "--algo_method",
            "index",
            "--solution_index",
            solutionIndex,
            "--activation_type",
            tensileActivationtType_to_bench_string(problem.gemms[0].getParams().activationEnum()),
            flush ? "--flush" : "",
            "--any_stride",
            "--rotating",
            rotatingBufferSize,
            "--cold_iters",
            coldIterations,
            "--iters",
            hotIterations);

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
            log_bench_from_str(s);
        if(rocblaslt::Debug::Instance().printLogAsMarker())
        {
            rocblaslt::Debug::Instance().logMarkerStart(s.c_str());
        }
        if(rocblaslt::Debug::Instance().benchPrintCommand())
        {
            std::cout << s << std::endl;
            rocblaslt::Debug::Instance().setBenchPrint(false);
        }
    }

    inline void
        logProfileFromTensileDataGemm(const TensileLite::ContractionProblemGroupedGemm& problem,
                                      const TensileLite::ContractionGroupedInputs&      inputs,
                                      const int&     solutionIndex,
                                      bool           flush,
                                      const int32_t& rotatingBufferSize,
                                      const int32_t& coldIterations,
                                      const int32_t& hotIterations,
                                      bool           isCpp)
    {
        size_t            gemmCount = problem.gemms.size();
        std::stringstream grouped_gemm_profile_string;
        for(int i = 0; i < gemmCount; ++i)
        {
            grouped_gemm_profile_string << " m: " << problem.gemms[i].c().sizes()[0] << ",";
            grouped_gemm_profile_string << " n: " << problem.gemms[i].c().sizes()[1] << ",";
            grouped_gemm_profile_string
                << " k: " << problem.gemms[i].a().sizes()[problem.gemms[i].boundIndices()[0].a]
                << ",";
            grouped_gemm_profile_string << " lda: " << problem.gemms[i].a().strides()[1] << ",";
            grouped_gemm_profile_string << " ldb: " << problem.gemms[i].b().strides()[1] << ",";
            grouped_gemm_profile_string << " ldc: " << problem.gemms[i].c().strides()[1] << ",";
            grouped_gemm_profile_string << " ldd: " << problem.gemms[i].d().strides()[1] << ",";
            if(problem.gemms[i]
                   .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                   .strides()
                   .size())
                grouped_gemm_profile_string
                    << " lde: "
                    << problem.gemms[i]
                           .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                           .strides()[1]
                    << ",";
            grouped_gemm_profile_string << " stride_a: " << problem.gemms[i].a().strides()[2]
                                        << ",";
            grouped_gemm_profile_string << " stride_b: " << problem.gemms[i].b().strides()[2]
                                        << ",";
            grouped_gemm_profile_string << " stride_c: " << problem.gemms[i].c().strides()[2]
                                        << ",";
            if(i != (gemmCount - 1))
            {
                grouped_gemm_profile_string << " stride_d: " << problem.gemms[i].d().strides()[2]
                                            << ",";
                if(problem.gemms[i]
                       .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                       .strides()
                       .size())
                    grouped_gemm_profile_string
                        << " stride_e: "
                        << problem.gemms[i]
                               .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                               .strides()[2]
                        << ",";
            }
            else
            {
                grouped_gemm_profile_string << " stride_d: " << problem.gemms[i].d().strides()[2];
                if(problem.gemms[i]
                       .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                       .strides()
                       .size())
                    grouped_gemm_profile_string
                        << " stride_e: "
                        << problem.gemms[i]
                               .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                               .strides()[2];
            }
        }
        log_profile(
            "matmul",
            "grouped_gemm",
            grouped_gemm_profile_string.str(),
            "alpha",
            ToString(inputs.grouped[0].alpha),
            "beta",
            ToString(inputs.grouped[0].beta),
            "transA",
            problem.gemms[0].transA() ? "T" : "N",
            "transB",
            problem.gemms[0].transB() ? "T" : "N",
            "batch_count",
            problem.gemms[0].batchSize(0),
            "scaleA",
            problem.gemms[0].useScaleAB().empty()
                ? 0
                : (problem.gemms[0].useScaleAB() == "Vector" ? 2 : 1),
            "scaleB",
            problem.gemms[0].useScaleAB().empty()
                ? 0
                : (problem.gemms[0].useScaleAB() == "Vector" ? 2 : 1),
            "scaleC",
            problem.gemms[0].useScaleCD() ? 1 : 0,
            "scaleD",
            problem.gemms[0].useScaleCD() ? 1 : 0,
            "swizzleA",
            problem.gemms[0].swizzleTensorA() ? "true" : "false",
            "swizzleB",
            problem.gemms[0].swizzleTensorB() ? "true" : "false",
            "scaleAlpha_vector",
            problem.gemms[0].useScaleAlphaVec() ? "true" : "false",
            "gradient",
            problem.gemms[0].useGradient() ? "true" : "false",
            "use_e",
            problem.gemms[0].useE() ? "true" : "false",
            "bias_vector",
            problem.gemms[0].useBias() ? "true" : "false",
            "bias_source",
            problem.gemms[0].useBias()
                ? problem.gemms[0].tensor(problem.gemms[0].biasSrc()).getName()
                : "d",
            "a_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].a().dataType())),
            "b_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].b().dataType())),
            "c_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].c().dataType())),
            "d_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].d().dataType())),
            "scale_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].alphaType())),
            "bias_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].bias().dataType())),
            "aux_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].e().dataType())),
            "compute_type",
            tensileComputeInputType_to_profile_string(problem.gemms[0].computeType(),
                                                      problem.gemms[0].f32XdlMathOp(),
                                                      problem.gemms[0].computeInputTypeA(),
                                                      problem.gemms[0].computeInputTypeB(),
                                                      problem.gemms[0].a().dataType(),
                                                      problem.gemms[0].b().dataType()),
            "algo_method",
            2,
            "solution_index",
            solutionIndex,
            "activation_type",
            tensileActivationtType_to_bench_string(problem.gemms[0].getParams().activationEnum()),
            "flush",
            flush ? "true" : "false",
            "any_stride",
            "true",
            "rotating",
            rotatingBufferSize,
            "cold_iters",
            coldIterations,
            "iters",
            hotIterations);
    }
    inline void logExtendedProfileFromTensileDataGemm(
        const TensileLite::ContractionProblemGroupedGemm& problem,
        const TensileLite::ContractionGroupedInputs&      inputs,
        const int&                                        solutionIndex,
        const std::string&                                kernelName,
        const std::string&                                solutionName,
        bool                                              flush,
        const int32_t&                                    rotatingBufferSize,
        const int32_t&                                    coldIterations,
        const int32_t&                                    hotIterations,
        bool                                              isCpp)
    {
        size_t            gemmCount = problem.gemms.size();
        std::stringstream grouped_gemm_profile_string;
        for(int i = 0; i < gemmCount; ++i)
        {
            grouped_gemm_profile_string << " m: " << problem.gemms[i].c().sizes()[0] << ",";
            grouped_gemm_profile_string << " n: " << problem.gemms[i].c().sizes()[1] << ",";
            grouped_gemm_profile_string
                << " k: " << problem.gemms[i].a().sizes()[problem.gemms[i].boundIndices()[0].a]
                << ",";
            grouped_gemm_profile_string << " lda: " << problem.gemms[i].a().strides()[1] << ",";
            grouped_gemm_profile_string << " ldb: " << problem.gemms[i].b().strides()[1] << ",";
            grouped_gemm_profile_string << " ldc: " << problem.gemms[i].c().strides()[1] << ",";
            grouped_gemm_profile_string << " ldd: " << problem.gemms[i].d().strides()[1] << ",";
            if(problem.gemms[i]
                   .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                   .strides()
                   .size())
                grouped_gemm_profile_string
                    << " lde: "
                    << problem.gemms[i]
                           .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                           .strides()[1]
                    << ",";
            grouped_gemm_profile_string << " stride_a: " << problem.gemms[i].a().strides()[2]
                                        << ",";
            grouped_gemm_profile_string << " stride_b: " << problem.gemms[i].b().strides()[2]
                                        << ",";
            grouped_gemm_profile_string << " stride_c: " << problem.gemms[i].c().strides()[2]
                                        << ",";
            if(i != (gemmCount - 1))
            {
                grouped_gemm_profile_string << " stride_d: " << problem.gemms[i].d().strides()[2]
                                            << ",";
                if(problem.gemms[i]
                       .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                       .strides()
                       .size())
                    grouped_gemm_profile_string
                        << " stride_e: "
                        << problem.gemms[i]
                               .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                               .strides()[2]
                        << ",";
            }
            else
            {
                grouped_gemm_profile_string << " stride_d: " << problem.gemms[i].d().strides()[2];
                if(problem.gemms[i]
                       .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                       .strides()
                       .size())
                    grouped_gemm_profile_string
                        << " stride_e: "
                        << problem.gemms[i]
                               .tensor(TensileLite::ContractionProblemGemm::TENSOR::E)
                               .strides()[2];
            }
        }
        log_profile(
            "matmul",
            "grouped_gemm",
            grouped_gemm_profile_string.str(),
            "alpha",
            ToString(inputs.grouped[0].alpha),
            "beta",
            ToString(inputs.grouped[0].beta),
            "transA",
            problem.gemms[0].transA() ? "T" : "N",
            "transB",
            problem.gemms[0].transB() ? "T" : "N",
            "batch_count",
            problem.gemms[0].batchSize(0),
            "scaleA",
            problem.gemms[0].useScaleAB().empty()
                ? 0
                : (problem.gemms[0].useScaleAB() == "Vector" ? 2 : 1),
            "scaleB",
            problem.gemms[0].useScaleAB().empty()
                ? 0
                : (problem.gemms[0].useScaleAB() == "Vector" ? 2 : 1),
            "scaleC",
            problem.gemms[0].useScaleCD() ? 1 : 0,
            "scaleD",
            problem.gemms[0].useScaleCD() ? 1 : 0,
            "swizzleA",
            problem.gemms[0].swizzleTensorA() ? "true" : "false",
            "swizzleB",
            problem.gemms[0].swizzleTensorB() ? "true" : "false",
            "scaleAlpha_vector",
            problem.gemms[0].useScaleAlphaVec() ? "true" : "false",
            "gradient",
            problem.gemms[0].useGradient() ? "true" : "false",
            "use_e",
            problem.gemms[0].useE() ? "true" : "false",
            "bias_vector",
            problem.gemms[0].useBias() ? "true" : "false",
            "bias_source",
            problem.gemms[0].useBias()
                ? problem.gemms[0].tensor(problem.gemms[0].biasSrc()).getName()
                : "d",
            "a_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].a().dataType())),
            "b_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].b().dataType())),
            "c_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].c().dataType())),
            "d_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].d().dataType())),
            "scale_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].alphaType())),
            "bias_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].bias().dataType())),
            "aux_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].e().dataType())),
            "compute_type",
            tensileComputeInputType_to_profile_string(problem.gemms[0].computeType(),
                                                      problem.gemms[0].f32XdlMathOp(),
                                                      problem.gemms[0].computeInputTypeA(),
                                                      problem.gemms[0].computeInputTypeB(),
                                                      problem.gemms[0].a().dataType(),
                                                      problem.gemms[0].b().dataType()),
            "activation_type",
            tensileActivationtType_to_bench_string(problem.gemms[0].getParams().activationEnum()),
            "flush",
            flush ? "true" : "false",
            "any_stride",
            "true",
            "rotating",
            rotatingBufferSize,
            "cold_iters",
            coldIterations,
            "iters",
            hotIterations,
            "solution_index",
            solutionIndex,
            "solution_Name",
            solutionName,
            "kernel_name",
            kernelName);
    }
#undef GEN_BENCH_ARG

    bool mxScaleTensorNeedsPaddingFreeDim()
    {
        static std::mutex                    cacheMutex;
        static std::unordered_map<int, bool> cache;

        int deviceId = 0;
        HIP_CHECK_EXC(hipGetDevice(&deviceId));

        std::lock_guard<std::mutex> lock(cacheMutex);
        auto                        it = cache.find(deviceId);
        if(it != cache.end())
            return it->second;

        hipDeviceProp_t prop;
        HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));
        const bool needsPadFreeDim = std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
        cache[deviceId]            = needsPadFreeDim;
        return needsPadFreeDim;
    }

    /****************************************************************
 * Construct a Tensile Problem from a RocblasltContractionProblem *
 ****************************************************************/
    auto ConstructTensileProblem(const RocblasltContractionProblem& prob)
    {
        auto a_type       = hipDataType_to_tensile_type(prob.a_type);
        auto b_type       = hipDataType_to_tensile_type(prob.b_type);
        auto c_type       = hipDataType_to_tensile_type(prob.c_type);
        auto d_type       = hipDataType_to_tensile_type(prob.d_type);
        auto compute_type = roc2TensileType(prob.compute_type, false);

        // Tensor descriptors for a, b
        TensileLite::TensorDescriptor a, b;

        // Tensile Indices for contraction problem
        TensileLite::ContractionProblemGemm::FreeIndices  freeIndex(2);
        TensileLite::ContractionProblemGemm::BoundIndices boundIndex(1);
        TensileLite::ContractionProblemGemm::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the
        // inputs. It optimizes all problems with alpha==0 into K=0 and alpha=(don't
        // care)
        double alpha = 0, beta = 0;
        assignAlphaBeta(compute_type, a_type, prob.alpha, prob.beta, &alpha, &beta);
        auto k = prob.k && alpha ? prob.k : 0;

        // fallback to f32 for f16 compute type after alpha/beta assignment
        if(prob.compute_type == rocblaslt_compute_f16)
        {
            compute_type = roc2TensileType(prob.compute_type);
        }

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != HIPBLAS_OP_N)
        {
            a = {
                    "a",
                    a_type,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    "a",
                    a_type,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a}
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != HIPBLAS_OP_N)
        {
            b = {
                    "b",
                    b_type,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    "b",
                    b_type,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b}
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        TensileLite::TensorDescriptor c{
            "c",
            c_type,
            {prob.m, prob.n, prob.batch_count},
            {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c}};

        // Descriptor for output matrix D
        TensileLite::TensorDescriptor d{
            "d",
            d_type,
            {prob.m, prob.n, prob.batch_count},
            {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d}};

        TensileLite::TensorDescriptor e{"e"};
        TensileLite::TensorDescriptor bias{"bias"};
        TensileLite::TensorDescriptor scaleA{"scaleA"};
        TensileLite::TensorDescriptor scaleB{"scaleB"};
        TensileLite::TensorDescriptor scaleC{"scaleC"};
        TensileLite::TensorDescriptor scaleD{"scaleD"};
        TensileLite::TensorDescriptor scaleAlphaVec{"scaleAlphaVec"};

        // The ContractionProblemGemm
        TensileLite::ContractionProblemGemm tensileProblem{a,
                                                           b,
                                                           c,
                                                           d,
                                                           e,
                                                           bias,
                                                           scaleA,
                                                           scaleB,
                                                           scaleC,
                                                           scaleD,
                                                           scaleAlphaVec,
                                                           freeIndex,
                                                           batchIndex,
                                                           boundIndex,
                                                           value_category(beta),
                                                           prob.workspaceSize};

        tensileProblem.setComputeInputTypeA(
            roc2TensileComputeInputTypeA(a_type, b_type, prob.compute_type));
        tensileProblem.setComputeInputTypeB(
            roc2TensileComputeInputTypeB(a_type, b_type, prob.compute_type));

        bool isComplexInput = (a_type == rocisa::DataType::ComplexFloat
                               || a_type == rocisa::DataType::ComplexDouble);

        auto alphaBetaType = isComplexInput ? a_type : compute_type;

        tensileProblem.setAlphaType(alphaBetaType);
        tensileProblem.setBetaType(alphaBetaType);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(
            TensileLite::DataTypeInfo::Get(compute_type).elementSize
            > TensileLite::DataTypeInfo::Get(a_type).elementSize);

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setBatchMode(static_cast<TensileLite::ContractionProblemGemm::BATCHMODE>(static_cast<int>(prob.batchMode)));        
        tensileProblem.setGroupedGemm(prob.grouped_gemm);
        if(prob.grouped_gemm)
            tensileProblem.setUseDeviceUserArguments(true);
        else
            tensileProblem.setUseDeviceUserArguments(false);

        if(prob.k == 0)
            tensileProblem.setAlphaRestriction(TensileLite::toScalarValueEnum(0.0));
        else
            tensileProblem.setAlphaRestriction(
                get_scalar_value_from_void_ptr(prob.alpha, alphaBetaType));

        tensileProblem.setBetaRestriction(
            get_scalar_value_from_void_ptr(prob.beta, alphaBetaType));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        if(is_e_enabled(prob.epilogue))
        {
            bool isOutput = prob.gradient ? false : true;
            auto aux_type = hipDataType_to_tensile_type(prob.aux_type);
            tensileProblem.setUseE(true);
            tensileProblem.setE(aux_type,
                                {prob.m, prob.n, prob.batch_count},
                                {prob.row_stride_e, prob.col_stride_e, prob.batch_stride_e},
                                isOutput);
        }

        // set bias mode
        auto biasSrc  = getBiasSrc(prob.epilogue);
        auto biasSize = (biasSrc == TensileLite::ContractionProblemGemm::TENSOR::B) ? d.sizes()[1]
                                                                                    : d.sizes()[0];
        tensileProblem.setUseBias(prob.bias != nullptr);
        auto biasType = hipDataType_to_tensile_type(prob.bias_type);
        tensileProblem.setBias(biasType, biasSize, prob.bias_stride, prob.gradient, biasSrc);
        tensileProblem.setParams().setBiasEnum(
            tensileUseBias(prob.epilogue) ? biasType : rocisa::DataType::None);

        const bool padMXScaleTensorFreeDim = mxScaleTensorNeedsPaddingFreeDim();

        switch(prob.scaleAType)
        {
        case RocblasltContractionProblem::ScalingFormat::None:
        case RocblasltContractionProblem::ScalingFormat::Scalar:
        case RocblasltContractionProblem::ScalingFormat::Vector:
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0:
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT:
	    // Block_32_UE8M0_32_8_EXT (commit fe9a04d) is pre-swizzled scale data in `32x8` tile
            tensileProblem.setMXScaleA(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0:
            tensileProblem.setMXScaleA(rocisa::DataType::E8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3:
            tensileProblem.setMXScaleA(rocisa::DataType::Float8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3:
            tensileProblem.setMXScaleA(rocisa::DataType::Float8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3:
            tensileProblem.setMXScaleA(rocisa::DataType::E5M3, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3:
            tensileProblem.setMXScaleA(rocisa::DataType::E5M3, 16, {}, padMXScaleTensorFreeDim);
            break;
        }

        switch(prob.scaleBType)
        {
        case RocblasltContractionProblem::ScalingFormat::None:
        case RocblasltContractionProblem::ScalingFormat::Scalar:
        case RocblasltContractionProblem::ScalingFormat::Vector:
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0:
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT:
	    // Block_32_UE8M0_32_8_EXT (commit fe9a04d) is pre-swizzled scale data in `32x8` tile
            tensileProblem.setMXScaleB(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0:
            tensileProblem.setMXScaleB(rocisa::DataType::E8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3:
            tensileProblem.setMXScaleB(rocisa::DataType::Float8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3:
            tensileProblem.setMXScaleB(rocisa::DataType::Float8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3:
            tensileProblem.setMXScaleB(rocisa::DataType::E5M3, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3:
            tensileProblem.setMXScaleB(rocisa::DataType::E5M3, 16, {}, padMXScaleTensorFreeDim);
            break;
        }

        if (prob.scaleA == nullptr && prob.scaleB == nullptr)
            tensileProblem.setUseScaleAB("");
        else if (prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Vector
                 || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Vector)
            tensileProblem.setUseScaleAB("Vector");
        else if (prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Scalar
                 || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Scalar)
            tensileProblem.setUseScaleAB("Scalar");
        else
            tensileProblem.setUseScaleAB("");

        tensileProblem.setUseScaleCD(prob.scaleC != nullptr || prob.scaleD != nullptr);
        tensileProblem.setUseScaleAlphaVec(prob.scaleAlphaVec != nullptr);
        tensileProblem.setScaleAlphaVec(compute_type, d.sizes()[0]);
        tensileProblem.setScaleA(compute_type, 1);
        tensileProblem.setScaleB(compute_type, 1);
        tensileProblem.setScaleC(compute_type);
        tensileProblem.setScaleD(compute_type);

        // set Actvation
        tensileProblem.setActivationType(is_act_enabled(prob.epilogue)
                                             ? TensileLite::ActivationType::Hipblaslt_all
                                             : TensileLite::ActivationType::None);
        tensileProblem.setActivationComputeType(compute_type);
        tensileProblem.setParams().setActivationEnum(getTensileActivationType(prob.epilogue));
        // set use gradient
        tensileProblem.setUseGradient(is_grad_enabled(prob.epilogue));

        // Forward HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT. Tri-state
        // {OFF=0, ON=1, AUTO=2}. The mode is consumed in
        // ContractionSolution::solve's SK5 arg-pack: AUTO delegates to
        // origami::streamk::select_hybrid_mode using sm_count_target as
        // the effective CU budget. Non-StreamK=5 solutions ignore it.
        tensileProblem.setParams().setStreamKTileSchedulingMode(prob.streamk_tile_scheduling_ext);
        tensileProblem.setParams().setSmCountTarget(prob.sm_count_target);

        // set AmaxD
        tensileProblem.setOutputAmaxD(prob.amaxD != nullptr);
        tensileProblem.setAmaxD(compute_type, true);

        if(prob.compute_type == rocblaslt_compute_f32_fast_xf32)
            tensileProblem.setF32XdlMathOp(rocisa::DataType::XFloat32);

        tensileProblem.setSwizzleTensorA(prob.swizzleA);
        tensileProblem.setSwizzleTensorB(prob.swizzleB);

        if(prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0 or
            prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT)
          tensileProblem.setMXScaleA(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
        if(prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0 or
            prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT)
          tensileProblem.setMXScaleB(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);

        return tensileProblem;
    }

    void updateTensileProblem(const RocblasltContractionProblem&   prob,
                              TensileLite::ContractionProblemGemm& tensileProblem)
    {
        auto a_type       = hipDataType_to_tensile_type(prob.a_type);
        auto b_type       = hipDataType_to_tensile_type(prob.b_type);
        auto c_type       = hipDataType_to_tensile_type(prob.c_type);
        auto d_type       = hipDataType_to_tensile_type(prob.d_type);
        auto compute_type = roc2TensileType(prob.compute_type, false);

        // Tensile Indices for contraction problem
        TensileLite::ContractionProblemGemm::FreeIndices  freeIndex(2);
        TensileLite::ContractionProblemGemm::BoundIndices boundIndex(1);
        TensileLite::ContractionProblemGemm::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the
        // inputs. It optimizes all problems with alpha==0 into K=0 and alpha=(don't
        // care)
        auto k = prob.k; // && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != HIPBLAS_OP_N)
        {
            tensileProblem.resetTensor(TensileLite::ContractionProblemGemm::TENSOR::A,
                    a_type,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a});
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            tensileProblem.resetTensor(TensileLite::ContractionProblemGemm::TENSOR::A,
                    a_type,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a});
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != HIPBLAS_OP_N)
        {
            tensileProblem.resetTensor(TensileLite::ContractionProblemGemm::TENSOR::B,
                    b_type,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b});
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            tensileProblem.resetTensor(TensileLite::ContractionProblemGemm::TENSOR::B,
                    b_type,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b});
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        tensileProblem.resetTensor(TensileLite::ContractionProblemGemm::TENSOR::C,
                                   c_type,
                                   {prob.m, prob.n, prob.batch_count},
                                   {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c});

        // Descriptor for output matrix D
        tensileProblem.resetTensor(TensileLite::ContractionProblemGemm::TENSOR::D,
                                   d_type,
                                   {prob.m, prob.n, prob.batch_count},
                                   {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d});

        if(prob.trans_a == HIPBLAS_OP_C)
            tensileProblem.setAOps({TensileLite::TensorOp::ComplexConjugate()});

        if(prob.trans_b == HIPBLAS_OP_C)
            tensileProblem.setBOps({TensileLite::TensorOp::ComplexConjugate()});

        double alpha = 0, beta = 0;
        assignAlphaBeta(compute_type, a_type, prob.alpha, prob.beta, &alpha, &beta);

        // fallback to f32 for f16 compute type after alpha/beta assignment
        if(prob.compute_type == rocblaslt_compute_f16)
        {
            compute_type = roc2TensileType(prob.compute_type);
        }

        tensileProblem.updateProblem(freeIndex, batchIndex, boundIndex, beta, prob.workspaceSize);

        tensileProblem.setComputeInputTypeA(
            roc2TensileComputeInputTypeA(a_type, b_type, prob.compute_type));
        tensileProblem.setComputeInputTypeB(
            roc2TensileComputeInputTypeB(a_type, b_type, prob.compute_type));

        bool isComplexInput = (a_type == rocisa::DataType::ComplexFloat
                               || a_type == rocisa::DataType::ComplexDouble);

        auto alphaBetaType = isComplexInput ? a_type : compute_type;

        tensileProblem.setAlphaType(alphaBetaType);
        tensileProblem.setBetaType(alphaBetaType);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(
            TensileLite::DataTypeInfo::Get(compute_type).elementSize
            > TensileLite::DataTypeInfo::Get(a_type).elementSize);

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setGroupedGemm(prob.grouped_gemm);
        tensileProblem.setBatchMode(static_cast<TensileLite::ContractionProblemGemm::BATCHMODE>(static_cast<int>(prob.batchMode)));
        if(prob.grouped_gemm)
            tensileProblem.setUseDeviceUserArguments(true);
        else
            tensileProblem.setUseDeviceUserArguments(false);

        if(prob.k == 0)
            tensileProblem.setAlphaRestriction(TensileLite::toScalarValueEnum(0.0));
        else
            tensileProblem.setAlphaRestriction(
                get_scalar_value_from_void_ptr(prob.alpha, alphaBetaType));

        tensileProblem.setBetaRestriction(
            get_scalar_value_from_void_ptr(prob.beta, alphaBetaType));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        auto tensileAct = getTensileActivationType(prob.epilogue);

        auto& d = tensileProblem.tensor(TensileLite::ContractionProblemGemm::TENSOR::D);
        // set bias mode
        auto biasSrc  = getBiasSrc(prob.epilogue);
        auto biasSize = (biasSrc == TensileLite::ContractionProblemGemm::TENSOR::B) ? d.sizes()[1]
                                                                                    : d.sizes()[0];

        tensileProblem.setUseBias(prob.bias != nullptr);
        auto biasType = hipDataType_to_tensile_type(prob.bias_type);
        tensileProblem.setBias(biasType, biasSize, prob.bias_stride, prob.gradient, biasSrc);
        tensileProblem.setParams().setBiasEnum(
            tensileUseBias(prob.epilogue) ? biasType : rocisa::DataType::None);

        const bool padMXScaleTensorFreeDim = mxScaleTensorNeedsPaddingFreeDim();

        switch(prob.scaleAType)
        {
        case RocblasltContractionProblem::ScalingFormat::None:
        case RocblasltContractionProblem::ScalingFormat::Scalar:
        case RocblasltContractionProblem::ScalingFormat::Vector:
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0:
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT:
            tensileProblem.setMXScaleA(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0:
            tensileProblem.setMXScaleA(rocisa::DataType::E8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3:
            tensileProblem.setMXScaleA(rocisa::DataType::Float8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3:
            tensileProblem.setMXScaleA(rocisa::DataType::Float8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3:
            tensileProblem.setMXScaleA(rocisa::DataType::E5M3, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3:
            tensileProblem.setMXScaleA(rocisa::DataType::E5M3, 16, {}, padMXScaleTensorFreeDim);
            break;
        }

        switch(prob.scaleBType)
        {
        case RocblasltContractionProblem::ScalingFormat::None:
        case RocblasltContractionProblem::ScalingFormat::Scalar:
        case RocblasltContractionProblem::ScalingFormat::Vector:
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0:
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT:
            tensileProblem.setMXScaleB(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0:
            tensileProblem.setMXScaleB(rocisa::DataType::E8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3:
            tensileProblem.setMXScaleB(rocisa::DataType::Float8, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3:
            tensileProblem.setMXScaleB(rocisa::DataType::Float8, 16, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3:
            tensileProblem.setMXScaleB(rocisa::DataType::E5M3, 32, {}, padMXScaleTensorFreeDim);
            break;
        case RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3:
            tensileProblem.setMXScaleB(rocisa::DataType::E5M3, 16, {}, padMXScaleTensorFreeDim);
            break;
        }

        if (prob.scaleA == nullptr && prob.scaleB == nullptr)
            tensileProblem.setUseScaleAB("");
        else if (prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Vector
                 || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Vector)
            tensileProblem.setUseScaleAB("Vector");
        else if (prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Scalar
                 || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Scalar)
            tensileProblem.setUseScaleAB("Scalar");
        else
            tensileProblem.setUseScaleAB("");

        tensileProblem.setUseScaleCD(prob.scaleC != nullptr || prob.scaleD != nullptr);
        tensileProblem.setUseScaleAlphaVec(prob.scaleAlphaVec != nullptr);
        tensileProblem.setScaleAlphaVec(compute_type, d.sizes()[0]);
        tensileProblem.setScaleA(compute_type, 1);
        tensileProblem.setScaleB(compute_type, 1);
        tensileProblem.setScaleC(compute_type);
        tensileProblem.setScaleD(compute_type);

        // set Actvation
        tensileProblem.setActivationType(is_act_enabled(prob.epilogue)
                                             ? TensileLite::ActivationType::Hipblaslt_all
                                             : TensileLite::ActivationType::None);
        tensileProblem.setActivationComputeType(compute_type);
        tensileProblem.setParams().setActivationEnum(getTensileActivationType(prob.epilogue));

        // Forward HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT. See
        // companion block in ConstructTensileProblem for details.
        tensileProblem.setParams().setStreamKTileSchedulingMode(prob.streamk_tile_scheduling_ext);
        tensileProblem.setParams().setSmCountTarget(prob.sm_count_target);

        // set E
        if(is_e_enabled(prob.epilogue))
        {
            bool isOutput = prob.gradient ? false : true;
            auto aux_type = hipDataType_to_tensile_type(prob.aux_type);
            tensileProblem.setUseE(true);
            tensileProblem.setE(aux_type,
                                {prob.m, prob.n, prob.batch_count},
                                {prob.row_stride_e, prob.col_stride_e, prob.batch_stride_e},
                                isOutput);
        }

        // set gradient
        tensileProblem.setUseGradient(is_grad_enabled(prob.epilogue));

        // set AmaxD
        tensileProblem.setOutputAmaxD(prob.amaxD != nullptr);
        tensileProblem.setAmaxD(compute_type, true);

        if(prob.compute_type == rocblaslt_compute_f32_fast_xf32)
            tensileProblem.setF32XdlMathOp(rocisa::DataType::XFloat32);
        else
            tensileProblem.setF32XdlMathOp(rocisa::DataType::Float);

        tensileProblem.setSwizzleTensorA(prob.swizzleA);
        tensileProblem.setSwizzleTensorB(prob.swizzleB);

	if(prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0 or
   	   prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT)
	    tensileProblem.setMXScaleA(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
	if(prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0 or
   	   prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT)
	    tensileProblem.setMXScaleB(rocisa::DataType::E8, 32, {}, padMXScaleTensorFreeDim);
    }

    rocisa::DataType computeTypeToRocisaDataType(rocblaslt_compute_type compute_type)
    {
        switch(compute_type)
        {
        case rocblaslt_compute_f16:
            return rocisa::DataType::Half;

        case rocblaslt_compute_f32:
            // Case f32 is often the base type for f32_r compute
            return rocisa::DataType::Float;

        case rocblaslt_compute_f32_fast_xf32:
            return rocisa::DataType::XFloat32;

        case rocblaslt_compute_f64:
            return rocisa::DataType::Double;

        case rocblaslt_compute_i32:
            return rocisa::DataType::Int32;

        case rocblaslt_compute_f32_fast_f16:
        case rocblaslt_compute_f32_fast_bf16:
        case rocblaslt_compute_f32_fast_f8_fnuz:
        case rocblaslt_compute_f32_fast_bf8_fnuz:
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
        case rocblaslt_compute_f32_fast_f8:
        case rocblaslt_compute_f32_fast_bf8:
        case rocblaslt_compute_f32_fast_f8bf8:
        case rocblaslt_compute_f32_fast_bf8f8:
            return rocisa::DataType::Float;

        default:
            return rocisa::DataType::None;
        }
    }

    /***************************************************************
 * Construct the inputs to a Tensile ContractionProblemGemm        *
 ***************************************************************/
    auto GetTensileInputs(const RocblasltContractionProblem& prob)
    {
        auto compute_type = roc2TensileType(prob.compute_type, false);

        // Structure describing the inputs (A, B, C, D, alpha, beta)
        TensileLite::ContractionInputs inputs;

        // Set the A, B, C, D matrices pointers in Tensile
        inputs.a = reinterpret_cast<const void*>(prob.A);
        inputs.b = reinterpret_cast<const void*>(prob.B);
        inputs.c = reinterpret_cast<const void*>(prob.C);
        inputs.d = reinterpret_cast<void*>(prob.D);
        inputs.e = reinterpret_cast<void*>(prob.E);

        inputs.batchA = reinterpret_cast<void const* const*>(prob.batch_A);
        inputs.batchB = reinterpret_cast<void const* const*>(prob.batch_B);
        inputs.batchC = reinterpret_cast<void const* const*>(prob.batch_C);
        inputs.batchD = reinterpret_cast<void* const*>(prob.batch_D);

        // The batch offsets are specified by the user in elements; convert them to
        // bytes here so the kernel/assembly can add them straight to byte addresses.
        // Only data types whose element size is at least one byte are supported
        // (sub-byte types such as fp4/fp6 are rejected during argument validation).
        inputs.batchOffsetA
            = prob.batch_offset_a
              * size_t(TensileLite::DataTypeInfo::Get(hip2TensileType(prob.a_type)).elementSize);
        inputs.batchOffsetB
            = prob.batch_offset_b
              * size_t(TensileLite::DataTypeInfo::Get(hip2TensileType(prob.b_type)).elementSize);
        inputs.batchOffsetC
            = prob.batch_offset_c
              * size_t(TensileLite::DataTypeInfo::Get(hip2TensileType(prob.c_type)).elementSize);
        inputs.batchOffsetD
            = prob.batch_offset_d
              * size_t(TensileLite::DataTypeInfo::Get(hip2TensileType(prob.d_type)).elementSize);

        // Set the GSU workspace
        inputs.ws            = prob.workspace;
        inputs.workspaceSize = prob.workspaceSize;

        inputs.Synchronizer = prob.Synchronizer;

        // set bias vector
        if(is_bias_enabled(prob.epilogue))
            inputs.bias = reinterpret_cast<const void*>(prob.bias);
        else
            inputs.bias = nullptr;

        if(prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0
            || prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT
            || prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3
            || prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0
            || prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3
            || prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3
            || prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3)
        {
            inputs.scaleA = nullptr;
            inputs.mxsa   = reinterpret_cast<const void*>(prob.scaleA);
        }
        else
        {
            inputs.scaleA = reinterpret_cast<const void*>(prob.scaleA);
            inputs.mxsa   = nullptr;
        }

        if(prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0
            || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT
            || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_16_UE4M3
            || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_16_UE8M0
            || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE4M3
            || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_32_UE5M3
            || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block_16_UE5M3)
        {
            inputs.scaleB = nullptr;
            inputs.mxsb   = reinterpret_cast<const void*>(prob.scaleB);
        }
        else
        {
            inputs.scaleB = reinterpret_cast<const void*>(prob.scaleB);
            inputs.mxsb   = nullptr;
        }

        inputs.scaleC        = reinterpret_cast<const void*>(prob.scaleC);
        inputs.scaleD        = reinterpret_cast<const void*>(prob.scaleD);
        inputs.scaleAlphaVec = reinterpret_cast<const void*>(prob.scaleAlphaVec);
        inputs.amaxD         = reinterpret_cast<void*>(prob.amaxD);

        static const std::map<rocisa::DataType, TensileLite::ConstantVariant> argument_vals = {
            {rocisa::DataType::Float, 0.0f},
            {rocisa::DataType::XFloat32, 0.0f},
            {rocisa::DataType::Half, (hipblasLtHalf)0.0},
            {rocisa::DataType::Int32, (int32_t)0},
            {rocisa::DataType::Double, (double)0.0},
        };

        if(argument_vals.find(compute_type) == argument_vals.end())
        {
            log_error(__func__, "Unsupported compute type");
            throw std::runtime_error("[GetTensileInputs] unsupported compute type.");
        }

        {
            const rocisa::DataType complex_check_type
                = (prob.a_type == HIP_C_64F)   ? rocisa::DataType::ComplexDouble
                  : (prob.a_type == HIP_C_32F) ? rocisa::DataType::ComplexFloat
                                               : rocisa::DataType::None;

            TensileLite::ConstantVariant visitable_alpha_beta_arg;

            // If the matrix type is complex, manually inject the complex zero variant.
            if(complex_check_type != rocisa::DataType::None)
            {
                if(complex_check_type == rocisa::DataType::ComplexDouble)
                {
                    visitable_alpha_beta_arg = hipblaslt_complex_double(0.0, 0.0);
                }
                else
                {
                    visitable_alpha_beta_arg = hipblaslt_complex_float(0.0f, 0.0f);
                }
            }
            else
            {
                // If not complex, fall back to the compute type map.
                rocisa::DataType key_type = computeTypeToRocisaDataType(prob.compute_type);

                auto it = argument_vals.find(key_type);
                if(it == argument_vals.end())
                {
                    // Handle error/default case for unsupported compute types here
                    log_error(__func__, "Unsupported compute type in argument_vals map.");
                    throw std::runtime_error("[GetTensileInputs] unsupported compute type.");
                }
                visitable_alpha_beta_arg = it->second;
            }

            // Now run the visit with the correctly type-resolved variant
            std::visit(
                [&inputs, &prob](auto val) {
                    using T = decltype(val);
                    if(prob.k)
                        inputs.alpha = *(decltype(val)*)(prob.alpha);
                    else
                        inputs.alpha = val;
                    inputs.beta = *(decltype(val)*)(prob.beta);
                },
                visitable_alpha_beta_arg);
        }

        {
            rocisa::DataType key_type = computeTypeToRocisaDataType(prob.compute_type);

            auto it = argument_vals.find(key_type);

            if(it == argument_vals.end())
            {
                // Handle error as before
                log_error(__func__, "Unsupported compute type for activation args");
                throw std::runtime_error("[GetTensileInputs] unsupported compute type.");
            }

            std::visit(
                [&inputs, &prob](auto val) {
                    using ValType = decltype(val);
                    if constexpr (std::is_constructible_v<ValType, float>)
                    {
                        inputs.activationArgs.push_back(static_cast<ValType>(prob.act0));
                        inputs.activationArgs.push_back(static_cast<ValType>(prob.act1));
                    }
                    else
                    {
                        inputs.activationArgs.push_back(prob.act0);
                        inputs.activationArgs.push_back(prob.act1);
                    }
                },
                it->second);
        }

        // convert alpha and beta to float if compute type is half
        if(prob.compute_type == rocblaslt_compute_f16)
        {
            inputs.activationArgs = {prob.act0, prob.act1};
            inputs.alpha          = static_cast<float>(std::get<hipblasLtHalf>(inputs.alpha));
            inputs.beta           = static_cast<float>(std::get<hipblasLtHalf>(inputs.beta));
        }

        return inputs;
    }

    TensileLite::LazyLoadingInit getLazyLoadingArch(int deviceID)
    {
        hipDeviceProp_t deviceProperties;
        HIP_CHECK_EXC(hipGetDeviceProperties(&deviceProperties, deviceID));
        // strip out xnack/ecc from name
        std::string deviceFullString(deviceProperties.gcnArchName);
        std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

        if(deviceString.find("gfx803") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx803;
        }
        else if(deviceString.find("gfx900") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx900;
        }
        else if(deviceString.find("gfx906") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx906;
        }
        else if(deviceString.find("gfx908") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx908;
        }
        else if(deviceString.find("gfx90a") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx90a;
        }
        else if(deviceString.find("gfx942") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx942;
        }
        else if(deviceString.find("gfx950") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx950;
        }
        else if(deviceString.find("gfx1010") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1010;
        }
        else if(deviceString.find("gfx1011") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1011;
        }
        else if(deviceString.find("gfx1012") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1012;
        }
        else if(deviceString.find("gfx1030") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1030;
        }
        else if(deviceString.find("gfx1100") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1100;
        }
        else if(deviceString.find("gfx1101") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1101;
        }
        else if(deviceString.find("gfx1102") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1102;
        }
        else if(deviceString.find("gfx1103") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1103;
        }
        else if(deviceString.find("gfx1150") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1150;
        }
        else if(deviceString.find("gfx1151") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1151;
        }
        else if(deviceString.find("gfx1152") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1152;
        }
        else if(deviceString.find("gfx1153") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1153;
        }
        else if(deviceString.find("gfx1200") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1200;
        }
        else if(deviceString.find("gfx1201") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1201;
        }
        else if(deviceString.find("gfx1250") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1250;
        }
        return TensileLite::LazyLoadingInit::None;
    }

    /**************************************************
 * The TensileHost struct interfaces with Tensile *
 **************************************************/
    class TensileHost
    {
        // The library object
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
            m_library;
#if ROCBLASLT_TENSILE_LAZY_LOAD
        std::unordered_set<TensileLite::LazyLoadingInit>                  m_deviceSet;
        std::unordered_map<std::string, std::shared_ptr<hipDeviceProp_t>> m_devicePropMap;
        std::unordered_map<std::string, std::shared_ptr<TensileLite::Hardware>> m_hardwareMap;
#else
        std::shared_ptr<hipDeviceProp_t>       m_deviceProp;
        std::shared_ptr<TensileLite::Hardware> m_hardware;
#endif
        std::string m_tensileLibPath;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<TensileLite::hip::SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                                      mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        TensileHost()
            : m_adapters(GetDeviceCount())
        {
            // We mark TensileHost as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocblaslt_internal_tensile_is_initialized() = true;
        }

        // TensileHost is not copyable or assignable
        TensileHost(const TensileHost&)            = delete;
        TensileHost& operator=(const TensileHost&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile host: No "
                             "devices found"
                          << std::endl;
                // rocblaslt_abort();
            }
            return count;
        }

        ~TensileHost()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_library() const
        {
            return m_library;
        }
#if ROCBLASLT_TENSILE_LAZY_LOAD
        auto& get_device_property(const std::string& deviceName) const
        {
            return m_devicePropMap.at(deviceName);
        }
        auto get_hardware(const std::string& deviceName) const
        {
            return m_hardwareMap.at(deviceName);
        }
#else
        auto& get_device_property() const
        {
            return m_deviceProp;
        }
        auto get_hardware() const
        {
            return m_hardware;
        }
#endif
        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*********************************************************************
   * Initialize adapter and library according to environment variables *
   * and default paths based on librocblaslt.so location and GPU         *
   *********************************************************************/
        void initialize(TensileLite::hip::SolutionAdapter& adapter, int32_t deviceId)
        {
            bool enableYaml = false;
            bool staticLib  = false;
            bool lazyLoad   = ROCBLASLT_TENSILE_LAZY_LOAD;
#ifdef TENSILE_YAML
            enableYaml = true;
#endif
#ifdef HIPBLASLT_STATIC_LIB
            staticLib = true;
#endif

            std::filesystem::path path;

            // The name of the current GPU platform
            std::string processor = rocblaslt_internal_get_arch_name();

            // ROCM-26729 / SEC-00896: use the privilege-aware accessor so a
            // process in a secure execution context cannot be redirected to an
            // attacker-controlled code-object directory via inherited
            // environment. Probe the privilege state once and reuse it for both
            // the lookup and the suppression diagnostic.
            const bool  is_privileged = rocblaslt_process_is_privileged();
            const char* env
                = rocblaslt_secure_getenv_impl("HIPBLASLT_TENSILE_LIBPATH", is_privileged);
            if(env)
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "Using HIPBLASLT_TENSILE_LIBPATH=" << env << std::endl;
                    log_info(__func__, msg.str());
                }
                path = env;
            }
            else
            {
                if(rocblaslt_env_suppressed_for_security_impl("HIPBLASLT_TENSILE_LIBPATH",
                                                              is_privileged))
                {
                    std::ostringstream msg;
                    msg << "Ignoring HIPBLASLT_TENSILE_LIBPATH because the process is running "
                           "in a secure execution context (set-uid/set-gid or another "
                           "credential-changing exec, such as file capabilities); falling back "
                           "to the default library location."
                        << std::endl;
                    log_error(__func__, msg.str());
                }
                // Find the location of librocblaslt.so
                // Fall back on hard-coded path if static library or not found
                std::optional<std::filesystem::path> default_lib_path;
                if(staticLib)
                {
                    default_lib_path = HIPBLASLT_LIB_PATH;
                }
                if(auto maybe_path = rocblaslt_find_library_relative_path(
                       /*relpath=*/std::nullopt, default_lib_path))
                    path = std::move(*maybe_path);
                // Optionally, look for a `processor` sub-directory under the library path.
                // Only use the subdir if a Tensile mapping file is actually present there;
                // otherwise the directory may have been created by ExtOp/Transform installs
                // without a corresponding Tensile library (multi-arch non-TheRock builds).
                {
                    auto processor_path     = path / processor;
                    auto mapping_msgpack    = processor_path / ("TensileLibrary_lazy_" + processor + ".dat");
                    auto mapping_msgpack_gz = processor_path / ("TensileLibrary_lazy_" + processor + ".dat.zlib");
                    auto mapping_yaml       = processor_path / ("TensileLibrary_lazy_" + processor + ".yaml");
                    if(std::filesystem::exists(mapping_msgpack) || std::filesystem::exists(mapping_msgpack_gz)
                       || std::filesystem::exists(mapping_yaml))
                        path = std::move(processor_path);
                }

                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "HIPBLASLT_TENSILE_LIBPATH not set: Using " << path << std::endl;
                    log_info(__func__, msg.str());
                }
            }

            // only load modules for the current architecture (contains the processor
            // string and ends in "co").
            if(!lazyLoad)
            {
                bool no_match = true;
                for(const auto& entry : std::filesystem::directory_iterator(path))
                {
                    auto filename = entry.path().filename();
                    if(filename.string().find(processor) != std::string::npos
                       && filename.extension().string() == ".co")
                    {
                        static_cast<void>(adapter.loadCodeObjectFile(entry.path().string()));
                        no_match = false;
                    }
                }
                if(no_match)
                {
                    // static rocblaslt_internal_ostream& once
                    //    = rocblaslt_cerr
                    std::cerr << "\nrocblaslt warning: No paths matched " << path
                              << ". Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly."
                              << std::endl;
                }
            }

            // We initialize a local static variable with a lambda function call to
            // avoid race conditions when multiple threads with different device IDs try
            // to initialize library. This ensures that only one thread initializes
            // library, and other threads trying to initialize library wait for it to
            // complete.
            static int once = [&] {
                // Determine library path. This is always the logical name with
                // a single ".dat"/".yaml" extension, never ".dat.zlib": the
                // loader resolves a compressed variant by appending ".zlib", and
                // downstream filename parsing (initLibraryMapping, placeholder
                // suffix derivation) assumes a single extension.
                std::filesystem::path tensileLibPath;
                if(lazyLoad)
                {
                    if(enableYaml)
                    {
                        tensileLibPath
                            = path / (std::string("TensileLibrary_lazy_") + processor + ".yaml");
                    }
                    else
                    {
                        tensileLibPath
                            = path / (std::string("TensileLibrary_lazy_") + processor + ".dat");
                    }
                }
                else
                {
                    if(enableYaml)
                    {
                        tensileLibPath
                            = path / (std::string("TensileLibrary_") + processor + ".yaml");
                    }
                    else
                    {
                        tensileLibPath
                            = path / (std::string("TensileLibrary_") + processor + ".dat");
                    }
                }
                if(!std::filesystem::exists(tensileLibPath)
                   && !std::filesystem::exists(tensileLibPath.string() + ".zlib"))
                {
                    std::cerr << "\nrocblaslt error: Cannot read " << tensileLibPath
                              << " (or .zlib variant): " << strerror(errno) << std::endl;
                    // rocblaslt_abort();
                }

#if ROCBLASLT_TENSILE_LAZY_LOAD
                // Get devices
                hipDeviceProp_t prop;
                int             count;
                HIP_CHECK_EXC(hipGetDeviceCount(&count));
                for(int devId = 0; devId < count; devId++)
                {
                    auto deviceArch = getLazyLoadingArch(devId);
                    if(m_deviceSet.find(deviceArch) == m_deviceSet.end())
                    {
                        // populate the arch list for lazy loading
                        m_deviceSet.insert(deviceArch);
                        // populate device property map, used in finding solutions based on arch
                        HIP_CHECK_EXC(hipGetDeviceProperties(&prop, devId));
                        // strip out xnack/ecc from name
                        std::string deviceFullString(prop.gcnArchName);
                        std::string deviceString
                            = deviceFullString.substr(0, deviceFullString.find(":"));
                        m_devicePropMap[deviceString] = std::make_shared<hipDeviceProp_t>(prop);
                        m_hardwareMap[deviceString] = TensileLite::hip::GetDevice(devId);
                    }
                }

                // Load library
                auto lib = TensileLite::LoadLibraryFilePreload<TensileLite::ContractionProblemGemm>(
                    tensileLibPath.string(), std::vector<TensileLite::LazyLoadingInit>{});
#else
                // Get device prop
                hipDeviceProp_t prop;
                HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));
                m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
                m_hardware = TensileLite::hip::GetDevice(deviceId);

                // Load library
                auto lib = TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(
                    tensileLibPath.string());
#endif
                if(!lib)
                    std::cerr << "\nrocblaslt error: Could not load " << tensileLibPath
                              << std::endl;
                else
                {
                    using MSL
                        = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;
                    m_library = std::dynamic_pointer_cast<MSL>(lib);
                    if(!m_library->initLibraryMapping(tensileLibPath.string()))
                    {
                        std::cerr << "\nrocblaslt error: Could not initialize Tensile library "
                                     "mapping"
                                  << std::endl;
                    }
                    m_tensileLibPath = tensileLibPath.string();
                }
                return 0;
            }();

            static_cast<void>(adapter.initializeLazyLoading(processor, path.string()));

            if(!m_library && once != 0)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile library" << std::endl;
                // rocblaslt_abort();
            }
        }
    };

    // Return the library and adapter for the current HIP device
    TensileLite::hip::SolutionAdapter* get_library_and_adapter(
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>*
            library
        = nullptr,
        std::shared_ptr<hipDeviceProp_t>*       deviceProp = nullptr,
        std::shared_ptr<TensileLite::Hardware>* hardware   = nullptr,
        int                                     device     = -1)
    try
    {
        // TensileHost is initialized on the first call
        static TensileHost host;

        if(device == -1)
            static_cast<void>(hipGetDevice(&device));

        // Adapter entry for the current HIP device ID
        auto& a       = host.get_adapters().at(device);
        auto* adapter = a.adapter.load(std::memory_order_acquire);

        // Once set, a.adapter contains the adapter for the current HIP device ID
        if(!adapter)
        {
            // Lock so that only one thread performs initialization of the adapter
            std::lock_guard<std::mutex> lock(a.mutex);

            adapter = a.adapter.load(std::memory_order_relaxed);
            if(!adapter)
            {
                // Allocate a new adapter using the current HIP device
                adapter = new TensileLite::hip::SolutionAdapter;

                // Initialize the adapter and possibly the library
                host.initialize(*adapter, device);

                // Atomically change the adapter stored for this device ID
                a.adapter.store(adapter, std::memory_order_release);
            }
        }

        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.get_library();
#if ROCBLASLT_TENSILE_LAZY_LOAD
        if(deviceProp)
            *deviceProp = host.get_device_property(rocblaslt_internal_get_arch_name());
        if(hardware)
            *hardware = host.get_hardware(rocblaslt_internal_get_arch_name());
#else
        if(deviceProp)
            *deviceProp = host.get_device_property();
        if(hardware)
            *hardware = host.get_hardware();
#endif

        return adapter;
    }
    catch(const std::exception& e)
    {
        std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\n"
                  << e.what() << std::endl;
        return nullptr;
    }
    catch(...)
    {
        std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\nUnknown "
                     "exception thrown"
                  << std::endl;
        return nullptr;
    }

#if 0
    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const std::ostream& msg)
    {
        if(rocblaslt_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCBLASLT_VERBOSE_TENSILE_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = std::cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
        }
        else
            std::cerr << msg << std::endl;
    }
#endif
} // namespace

struct TensileDataGemm
{
    bool                                       enableEpilogue = true;
    TensileLite::ContractionProblemGemm        problem;
    TensileLite::ContractionInputs             inputs;
    std::vector<TensileLite::KernelInvocation> kernels;
    int                                        algoIndex = std::numeric_limits<int>::max();

    // Built once from the RocblasltContractionProblem this object was created
    // from, so the C++ path looks up the same key the C path would. Deriving it
    // from the Tensile problem instead would mean a second builder that has to
    // stay in agreement with the first, which is how c_type versus d_type
    // already diverged between the two paths.
    TensileLite::ProblemOverride tuningKey;
};

struct TensileDataGroupedGemm
{
    bool                                       enableEpilogue = true;
    TensileLite::ContractionProblemGroupedGemm problem;
    TensileLite::ContractionGroupedInputs      inputs;
    std::vector<TensileLite::KernelInvocation> kernels;
    int                                        algoIndex = std::numeric_limits<int>::max();
    std::shared_ptr<void>                      hipHostMemory;
    size_t                                     hipHostMemorySize;
    bool                                       useUserArgs = false;
};

namespace
{
    struct DeviceIdentity
    {
        std::string archName;
        int32_t     cuCount = 0;
    };

    /**
     * The full gcnArchName, not the colon-stripped form
     * rocblaslt_internal_get_arch_name() returns. sramecc and xnack can change
     * which kernels apply, and the bench writer already records the full
     * string, so keeping it whole makes both sides agree and loses nothing.
     */
    const DeviceIdentity& getDeviceIdentity()
    {
        static std::mutex                    mtx;
        static std::map<int, DeviceIdentity> cache;

        int deviceId = 0;
        static_cast<void>(hipGetDevice(&deviceId));

        std::lock_guard<std::mutex> lock(mtx);

        auto found = cache.find(deviceId);
        if(found != cache.end())
            return found->second;

        DeviceIdentity  identity;
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, deviceId) == hipSuccess)
        {
            identity.archName = props.gcnArchName;
            identity.cuCount  = props.multiProcessorCount;
        }

        return cache.emplace(deviceId, std::move(identity)).first->second;
    }
} // namespace

/**
 * The single canonical key builder.
 *
 * Both the C and the C++ paths route through this, the latter by way of the
 * key cached on TensileDataGemm at create time. Anything that needs a key and
 * does not have a RocblasltContractionProblem is a bug waiting to happen; the
 * previous second builder keyed c_type where this one keys c and d separately,
 * so the two paths could not match each other's entries.
 */
TensileLite::ProblemOverride
    RocblasltContractionProblem2ProblemOverride(const RocblasltContractionProblem& problem)
{
    TensileLite::ProblemOverride po;

    po.transA    = problem.trans_a != HIPBLAS_OP_N;
    po.transB    = problem.trans_b != HIPBLAS_OP_N;
    po.m         = problem.m;
    po.n         = problem.n;
    po.k         = problem.k;
    po.batchSize = problem.batch_count;

    po.inputTypeA  = hipDataType_to_tensile_type(problem.a_type);
    po.inputTypeB  = hipDataType_to_tensile_type(problem.b_type);
    po.outputTypeC = hipDataType_to_tensile_type(problem.c_type);
    po.outputTypeD = hipDataType_to_tensile_type(problem.d_type);
    po.computeType = rocComputeType_to_tensile_type(problem.compute_type);
    po.computeInputTypeA
        = static_cast<int32_t>(roc2TensileComputeInputTypeA(hipDataType_to_tensile_type(problem.a_type),
                                                            hipDataType_to_tensile_type(problem.b_type),
                                                            problem.compute_type));
    po.computeInputTypeB
        = static_cast<int32_t>(roc2TensileComputeInputTypeB(hipDataType_to_tensile_type(problem.a_type),
                                                            hipDataType_to_tensile_type(problem.b_type),
                                                            problem.compute_type));

    po.colStrideA   = problem.col_stride_a;
    po.colStrideB   = problem.col_stride_b;
    po.colStrideC   = problem.col_stride_c;
    po.colStrideD   = problem.col_stride_d;
    po.batchStrideA = problem.batch_stride_a;
    po.batchStrideB = problem.batch_stride_b;
    po.batchStrideC = problem.batch_stride_c;
    po.batchStrideD = problem.batch_stride_d;
    po.batchMode    = static_cast<int32_t>(problem.batchMode);

    po.epilogue   = static_cast<int32_t>(problem.epilogue);
    po.gradient   = problem.gradient;
    po.biasType   = static_cast<int32_t>(problem.bias_type);
    po.biasStride = problem.bias_stride;
    po.hasBias    = problem.bias != nullptr;
    po.auxType    = static_cast<int32_t>(problem.aux_type);

    po.scaleAFormat     = static_cast<int32_t>(problem.scaleAType);
    po.scaleBFormat     = static_cast<int32_t>(problem.scaleBType);
    po.hasScaleA        = problem.scaleA != nullptr;
    po.hasScaleB        = problem.scaleB != nullptr;
    po.hasScaleC        = problem.scaleC != nullptr;
    po.hasScaleD        = problem.scaleD != nullptr;
    po.hasScaleE        = problem.scaleE != nullptr;
    po.hasScaleAlphaVec = problem.scaleAlphaVec != nullptr;
    po.hasAmaxD         = problem.amaxD != nullptr;

    po.swizzleA              = problem.swizzleA;
    po.swizzleB              = problem.swizzleB;
    po.streamkTileScheduling = problem.streamk_tile_scheduling_ext;
    po.smCountTarget         = problem.sm_count_target;

    const auto& device = getDeviceIdentity();
    po.archName        = device.archName;
    po.cuCount         = device.cuCount;

    // Behind its own switch, not the general info level. This runs on every key
    // build, which is every heuristic call on the hit path, and the formatting
    // alone measured a 7.7x slowdown when info logging was enabled.
    static const bool dumpKeys = getenv("HIPBLASLT_TUNING_DEBUG_KEY") != nullptr;
    if(dumpKeys && (get_logger_layer_mode() & rocblaslt_layer_mode_log_info))
    {
        std::ostringstream msg;
        msg << "tuning-cache: key hash=" << std::hash<TensileLite::ProblemOverride>{}(po)
            << " mnk=" << po.m << "x" << po.n << "x" << po.k << " ld=" << po.colStrideA << ","
            << po.colStrideB << "," << po.colStrideC << "," << po.colStrideD
            << " st=" << po.batchStrideA << "," << po.batchStrideB << "," << po.batchStrideC << ","
            << po.batchStrideD << " epi=" << po.epilogue << " bias=" << po.hasBias
            << " biasT=" << po.biasType << " auxT=" << po.auxType << " bm=" << po.batchMode
            << " sk=" << po.streamkTileScheduling << " sm=" << po.smCountTarget
            << " tci=" << po.computeInputTypeA << "," << po.computeInputTypeB
            << " types=" << (int)po.inputTypeA << "," << (int)po.inputTypeB << ","
            << (int)po.outputTypeC << "," << (int)po.outputTypeD << "," << (int)po.computeType
            << " scale=" << po.scaleAFormat << "," << po.scaleBFormat << "," << po.hasScaleA
            << po.hasScaleB << po.hasScaleC << po.hasScaleD << po.hasScaleE
            << po.hasScaleAlphaVec << " amax=" << po.hasAmaxD << " grad=" << po.gradient;
        log_info(__func__, msg.str());
    }

    return po;
}

TensileLite::ProblemOverride TensileDataGemm2ProblemOverride(std::shared_ptr<void> gemmData)
{
    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    return data->tuningKey;
}

TensileLite::ContractionProblemGemm* ExtractProblemGemm(std::shared_ptr<void> gemmData)
{
    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);

    return &data->problem;
}

// Apply the GemmPreference-supplied StreamK tile scheduling mode onto every
// contraction problem currently carried by gemmData. Called from
// rocblaslt_algo_get_heuristic_cpp before solution ranking so the SK5
// arg-pack and the heuristic-selection paths see the same mode value.
// Defined here because gemmData's concrete type (TensileDataGemm /
// TensileDataGroupedGemm) only exists in this translation unit.
void applyStreamKTileSchedulingMode(std::shared_ptr<void>  gemmData,
                                rocblaslt::RocGemmType gemmType,
                                int32_t                mode)
{
    if(!gemmData)
        return;
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        auto data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        if(data)
            data->problem.setParams().setStreamKTileSchedulingMode(mode);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        auto data = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        if(data)
        {
            for(auto& g : data->problem.gemms)
                g.setParams().setStreamKTileSchedulingMode(mode);
        }
    }
}

void initTensileGemmData(rocblaslt_handle       handle,
                         rocblaslt::RocGemmType gemmType,
                         hipblasOperation_t     opA,
                         hipblasOperation_t     opB,
                         hipDataType            typeA,
                         hipDataType            typeB,
                         hipDataType            typeC,
                         hipDataType            typeD,
                         rocblaslt_compute_type typeCompute,
                         size_t                 maxWorkspaceBytes,
                         std::shared_ptr<void>& gemmData)
{
    float alpha = 1.0;
    float beta  = 1.0;
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        TensileDataGemm data;
        data.problem = CreateTensileProblem(opA,
                                            opB,
                                            typeA,
                                            typeB,
                                            typeC,
                                            typeD,
                                            typeCompute,
                                            alpha,
                                            beta,
                                            false,
                                            maxWorkspaceBytes);
        gemmData     = std::static_pointer_cast<void>(std::make_shared<TensileDataGemm>(data));
        return;
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        TensileDataGroupedGemm                      data;
        TensileLite::ContractionProblemGroupedGemm& tensile_probs = data.problem;
        TensileLite::ContractionGroupedInputs&      groupedInputs = data.inputs;

        tensile_probs.gemms.push_back(CreateTensileProblem(opA,
                                                           opB,
                                                           typeA,
                                                           typeB,
                                                           typeC,
                                                           typeD,
                                                           typeCompute,
                                                           alpha,
                                                           beta,
                                                           true,
                                                           maxWorkspaceBytes));
        groupedInputs.grouped.resize(1);

        void* tmp = nullptr;
        static_cast<void>(hipHostMalloc(&tmp, INTERNAL_HIPHOSTMEM_SIZE, 0));
        data.hipHostMemory
            = std::shared_ptr<void>(tmp, [](auto p) { static_cast<void>(hipFree(p)); });
        data.hipHostMemorySize = INTERNAL_HIPHOSTMEM_SIZE;

        gemmData = std::static_pointer_cast<void>(std::make_shared<TensileDataGroupedGemm>(data));
        return;
    }

    throw std::runtime_error("Gemm problem type initialization not implemented.");
}

#ifdef HIPBLASLT_USE_ROCROLLER
bool useRocRoller(rocblaslt_handle handle, const RocblasltContractionProblem& prob)
{
    // Do not use rocRoller for FP4 A + FP4 B with pre-swizzled (shuffled) scale layout
    bool isFp4A = (prob.a_type == static_cast<hipDataType>(HIP_R_4F_E2M1));
    bool isFp4B = (prob.b_type == static_cast<hipDataType>(HIP_R_4F_E2M1));
    bool isShuffledScale
        = (prob.scaleAType
               == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT
           && prob.scaleBType
                  == RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT);
    if(isFp4A && isFp4B && isShuffledScale)
        return false;

    // Do not use rocRoller for FP8 E4M3 A + FP8 E4M3 B with pre-swizzled (shuffled) scale layout
    bool isFp8A = (prob.a_type == static_cast<hipDataType>(HIP_R_8F_E4M3));
    bool isFp8B = (prob.b_type == static_cast<hipDataType>(HIP_R_8F_E4M3));
    if(isFp8A && isFp8B && isShuffledScale)
        return false;

    return handle->useRocRoller == 1
           || (handle->useRocRoller == -1
               && (isBlockScaling(prob.scaleAType) || isBlockScaling(prob.scaleBType)));
}
#endif

/**
 * Invalidate every compute unit's instruction cache.
 *
 * Byte-for-byte the bench client's flush_icache kernel, and launched with the
 * same geometry, because the point of having it here is that the two measure
 * the same thing. Without it a candidate is timed with its own code already
 * resident after the first launch, which is not how a kernel runs the first
 * time a production workload reaches it.
 *
 * The nops are what the client uses: s_icache_inv has no hardware interlock, so
 * the wave has to be kept from fetching across the invalidate before it lands.
 */
__global__ void hipblasltTuningFlushICache()
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

namespace
{
    /**
     * How much work a single shape's tuning does.
     *
     * Deliberately the same measurement hipblaslt-bench performs: a fixed count
     * of untimed launches followed by a fixed count of timed ones, over rotating
     * buffers, with every candidate measured the same way. Matching it is the
     * point, since the two are expected to reach the same winner on the same
     * shape, and a cheaper or differently-shaped measurement here would show up
     * as the tuner and the bench client disagreeing.
     *
     * An earlier version derived iteration counts from a probe so each span hit
     * a target duration, on the theory that a fixed count means something
     * different for an 11 us kernel than for a 2 ms one. It was dropped: bench
     * does not do it, it made the two harder to compare, and the probe itself
     * cost roughly what it saved.
     */
    struct TuningPolicy
    {
        static int envInt(const char* name, int fallback)
        {
            if(const char* env = getenv(name))
            {
                try
                {
                    return std::stoi(env);
                }
                catch(...)
                {
                }
            }
            return fallback;
        }

        /**
         * Measure every kernel that can run the problem rather than the ranked
         * prefix the prediction model returns.
         *
         * The prefix is a hard ceiling on quality: on a 1024x512x1024 fp16 case
         * the fastest kernel sat outside the top 128 entirely, so no amount of
         * timing care could reach it. Enumerating removes that ceiling and makes
         * the per-shape budget the only bound.
         */
        static bool allKernels()
        {
            return envInt("HIPBLASLT_TUNING_ALL_KERNELS", 1) != 0;
        }
        /** Ranked-prefix size. Ignored when allKernels() is on. */
        static int candidateCap()
        {
            return std::max(1, envInt("HIPBLASLT_TUNING_MAX_CANDIDATES", 128));
        }

        /**
         * Invalidate the instruction cache between timed launches, as the bench
         * client does when tuning.
         *
         * On by default because it demonstrably improves agreement with bench:
         * across three fp16 shapes the tuner reached bench's exact winner on
         * two of them with the flush on, and repeated runs picked the same
         * kernel on those two. It is not a cure-all. On a shape whose leading
         * kernels sit within a couple of percent of each other the choice stays
         * unstable with the flush on, because at that separation the ordering
         * is below what either tool can resolve.
         *
         * The cost is about 5% of tuning time (135.6s to 142.5s on
         * 2048x1024x2048), far less than the ~10 us per flush suggests, because
         * the flush launches pipeline behind the GEMMs rather than serialising
         * with them.
         */
        static bool flushICache()
        {
            return envInt("HIPBLASLT_TUNING_FLUSH_ICACHE", 1) != 0;
        }
        /**
         * Untimed launches before measuring, then launches in the timed batch.
         *
         * The same fixed counts hipblaslt-bench uses when tuning, so a candidate
         * is measured over exactly the work bench would measure it over. Counts
         * rather than durations on purpose: matching bench's numbers is what
         * makes the two comparable, and the tuner still finishes sooner because
         * its launches cost about an eighth of what bench's do.
         */
        static int coldIterations()
        {
            return std::max(0, envInt("HIPBLASLT_TUNING_COLD_ITERS", 1000));
        }
        static int hotIterations()
        {
            return std::max(1, envInt("HIPBLASLT_TUNING_HOT_ITERS", 1000));
        }
        /**
         * Wall-clock ceiling for one shape, or zero for no ceiling.
         *
         * Unlimited by default, because the default search is exhaustive at
         * bench's iteration counts and any finite ceiling silently turns that
         * into a partial one. A 10 second default measured 140 of 782
         * candidates and then persisted the best of that prefix as if it were
         * the answer; since the candidate list is unranked, which 140 got
         * measured carries no meaning. A budget is now something a caller opts
         * into, and a search it cuts short is discarded rather than cached.
         */
        static double perShapeBudgetUs()
        {
            return 1000.0 * std::max(0, envInt("HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE", 0));
        }
        /**
         * Device memory the rotation may use, so successive launches do not read
         * and write the same lines.
         *
         * Every buffer a candidate touches is rotated, not just the operands:
         * A, B, C, D, the aux tensor, a gradient bias and the workspace. Leaving
         * any of them unrotated leaves that part of the footprint resident and
         * partially defeats the point.
         *
         * Without this the inputs stay resident after the first iteration and
         * every candidate is measured cache-hot, which is not how a production
         * GEMM runs and which measurably reorders the field: on MI300X, with
         * 256 MB of last-level cache to defeat, a hot measurement picked a
         * different kernel than hipblaslt-bench did on the same shape. 512 MB
         * matches what the bench client uses when tuning. Zero disables
         * rotation and restores the cache-hot behaviour.
         */
        static size_t rotatingBytes()
        {
            return size_t(std::max(0, envInt("HIPBLASLT_TUNING_ROTATING_MB", 512))) * 1024 * 1024;
        }
    };

    /**
     * Instruction-cache flush, sized and costed the way the bench client does.
     *
     * The grid is CU count times 60 with 64 threads, so every compute unit is
     * covered several times over; that geometry is copied from the client
     * rather than derived, because a smaller grid would not invalidate the same
     * set of caches and the two would stop measuring the same thing.
     *
     * costUs() is the average wall time of one such launch, measured once per
     * device and remembered. The timed span contains one flush per iteration,
     * so that average is subtracted back out to leave the GEMM time. Candidates
     * all pay the same flush, so the ranking would survive without it, but the
     * number written to the cache and compared against the baseline would be
     * the GEMM plus a flush rather than the GEMM.
     */
    class ICacheFlush
    {
    public:
        static ICacheFlush& instance()
        {
            static ICacheFlush gInstance;
            return gInstance;
        }

        bool launch(hipStream_t stream)
        {
            const Device* device = current();
            if(device == nullptr)
                return false;

            hipLaunchKernelGGL(hipblasltTuningFlushICache, dim3(device->grid), dim3(64), 0, stream);
            return hipGetLastError() == hipSuccess;
        }

        /** Microseconds one flush launch costs on the current device, or 0. */
        double costUs(hipStream_t stream)
        {
            const Device* device = current(stream);
            return device ? device->costUs : 0.0;
        }

    private:
        struct Device
        {
            unsigned grid   = 0;
            double   costUs = 0.0;
        };

        // Enough launches for the average to be stable without the 100k the
        // client uses; the flush is a fixed-cost launch, not a measurement that
        // needs converging, and this runs before a tuning pass that is about to
        // spend far longer than the calibration itself.
        static constexpr int kCalibrationIters = 2000;

        const Device* current(hipStream_t stream = nullptr)
        {
            int id = 0;
            if(hipGetDevice(&id) != hipSuccess)
                return nullptr;

            auto found = m_devices.find(id);
            if(found != m_devices.end())
            {
                // A launch-only caller can have created this entry before
                // anyone had a stream to calibrate on, which would otherwise
                // leave the cost pinned at zero for the life of the process and
                // silently stop the flush being subtracted.
                if(found->second.costUs == 0.0 && stream != nullptr)
                    found->second.costUs = calibrate(found->second.grid, stream);
                return &found->second;
            }

            hipDeviceProp_t props{};
            if(hipGetDeviceProperties(&props, id) != hipSuccess)
                return nullptr;

            Device device;
            device.grid = static_cast<unsigned>(props.multiProcessorCount) * 60u;

            // Only the geometry is needed to launch, so publish it first and
            // let a caller that has no stream to calibrate on use it at zero
            // cost rather than blocking.
            if(stream != nullptr)
                device.costUs = calibrate(device.grid, stream);

            return &(m_devices[id] = device);
        }

        static double calibrate(unsigned grid, hipStream_t stream)
        {
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            if(hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess)
            {
                if(start)
                    static_cast<void>(hipEventDestroy(start));
                if(stop)
                    static_cast<void>(hipEventDestroy(stop));
                return 0.0;
            }

            auto burst = [&] {
                for(int i = 0; i < kCalibrationIters; i++)
                    hipLaunchKernelGGL(hipblasltTuningFlushICache, dim3(grid), dim3(64), 0, stream);
            };

            burst(); // warm the queue and the flush kernel's own code object

            double us = 0.0;
            if(hipEventRecord(start, stream) == hipSuccess)
            {
                burst();
                float ms = 0.0f;
                if(hipEventRecord(stop, stream) == hipSuccess
                   && hipEventSynchronize(stop) == hipSuccess
                   && hipEventElapsedTime(&ms, start, stop) == hipSuccess)
                    us = (static_cast<double>(ms) * 1000.0) / kCalibrationIters;
            }

            static_cast<void>(hipEventDestroy(start));
            static_cast<void>(hipEventDestroy(stop));
            static_cast<void>(hipGetLastError());
            return us;
        }

        // Only ever touched with the tuning lock held.
        std::map<int, Device> m_devices;
    };

    /**
     * Device memory owned by the tuning path.
     *
     * Candidates must never write the caller's buffers, and the library has no
     * general workspace allocator to borrow, so tuning brings its own. Gating on
     * the caller's workspace being large enough to cover isolation instead would
     * mean tuning essentially never runs under a framework.
     *
     * One allocation per device, grown on demand, freed at process exit. All
     * tuning on a device is serialised, so a single buffer is enough.
     */
    class TuningScratch
    {
    public:
        static TuningScratch& instance()
        {
            static TuningScratch gInstance;
            return gInstance;
        }

        size_t cap() const { return m_cap; }

        /**
         * Null when the request exceeds the cap or allocation fails.
         *
         * One buffer per device, keyed by whichever device is current, because
         * that is the device hipMalloc will allocate on. A single process-wide
         * buffer handed the pointer allocated on the first device tuned to every
         * later device, so a candidate on device 1 read and wrote device 0
         * memory: on a peer-capable pair that silently measures cross-device
         * traffic, and otherwise it faults.
         */
        void* acquire(size_t bytes)
        {
            if(bytes == 0 || bytes > m_cap)
                return nullptr;

            int device = 0;
            if(hipGetDevice(&device) != hipSuccess)
                return nullptr;

            Slot& slot = m_slots[device];

            if(bytes <= slot.size)
                return slot.ptr;

            if(slot.ptr)
                static_cast<void>(hipFree(slot.ptr));
            slot.ptr  = nullptr;
            slot.size = 0;

            void* fresh = nullptr;
            if(hipMalloc(&fresh, bytes) != hipSuccess)
                return nullptr;

            slot.ptr  = fresh;
            slot.size = bytes;
            return slot.ptr;
        }

    private:
        TuningScratch()
        {
            if(const char* env = getenv("HIPBLASLT_TUNING_SCRATCH_MAX_BYTES"))
            {
                try
                {
                    m_cap = static_cast<size_t>(std::stoull(env));
                }
                catch(...)
                {
                }
            }
        }

        struct Slot
        {
            void*  ptr  = nullptr;
            size_t size = 0;
        };

        // Only ever touched with the tuning lock held.
        std::map<int, Slot> m_slots;
        size_t              m_cap = size_t(1) << 30;
    };

    /**
     * Byte span each output tensor covers, computed from the strides actually in
     * the problem rather than assumed to be dense.
     *
     * Returns 0 when the span cannot be established, which makes the caller skip
     * tuning. Guessing here would mean handing a kernel a buffer smaller than it
     * writes, so every uncertain case declines instead.
     *
     * `expanded` reports that a stride was rounded up to keep the result an
     * upper bound, which happens for broadcast layouts such as a zero batch
     * stride. Over-allocating scratch to write into is harmless, but the same
     * span used as a read length against the caller's buffer runs off the end of
     * an allocation that only ever held one batch, so rotation asks for this
     * flag on the tensors it copies from.
     */
    size_t tensorSpanBytes(size_t elementSize,
                           size_t rows,
                           size_t cols,
                           size_t colStride,
                           size_t batchCount,
                           size_t batchStride,
                           bool*  expanded = nullptr)
    {
        if(expanded)
            *expanded = false;

        if(elementSize == 0 || rows == 0 || cols == 0)
            return 0;

        const size_t batches = std::max<size_t>(batchCount, 1);

        // A column stride below the row count would overlap columns; treat as
        // dense rather than under-allocating.
        const size_t effectiveColStride = std::max(colStride, rows);
        if(expanded && effectiveColStride != colStride)
            *expanded = true;

        // Every guard precedes the multiplication it protects. Checking after
        // the fact reads an already-wrapped value and lets a too-small buffer
        // through.
        if(cols > 1 && effectiveColStride > (SIZE_MAX - rows) / (cols - 1))
            return 0;

        const size_t perBatch = effectiveColStride * (cols - 1) + rows;
        const size_t effectiveBatchStride
            = (batches > 1) ? std::max(batchStride, perBatch) : perBatch;
        if(expanded && batches > 1 && effectiveBatchStride != batchStride)
            *expanded = true;

        if(batches > 1 && effectiveBatchStride > (SIZE_MAX - perBatch) / (batches - 1))
            return 0;

        const size_t elements = effectiveBatchStride * (batches - 1) + perBatch;
        if(elements > (SIZE_MAX / elementSize))
            return 0;

        return elements * elementSize;
    }

    size_t elementSizeOf(hipDataType type)
    {
        return static_cast<size_t>(
            TensileLite::DataTypeInfo::Get(hipDataType_to_tensile_type(type)).elementSize);
    }

    /** Same shape as the Synchronizer the handle allocates. */
    constexpr size_t kSynchronizerBytes = 16 * 409600 * sizeof(int);

    struct ScratchLayout
    {
        bool   usable = false;
        size_t total  = 0;

        size_t offsetD            = 0;
        size_t bytesD             = 0;
        size_t bytesInPlaceC      = 0;
        size_t offsetE            = 0;
        size_t bytesE             = 0;
        size_t offsetBias         = 0;
        size_t bytesBias          = 0;
        size_t offsetAmaxD        = 0;
        size_t bytesAmaxD         = 0;
        size_t offsetWorkspace    = 0;
        size_t bytesWorkspace     = 0;
        size_t offsetSynchronizer = 0;

        // Rotation. Every buffer a candidate touches gets blockCount copies,
        // strided by its aligned span; block 0 of each input holds a copy of the
        // caller's data and the rest are broadcast from it. blockCount is always
        // at least 1, so the unrotated case is just the degenerate one.
        //
        // C is rotated whenever it is a distinct buffer, not only when beta is
        // non-zero: at beta zero the kernel never reads it so the copies are
        // merely unused, and testing beta here would mean decoding the scalar
        // through its storage type for no benefit beyond a few blocks of space.
        // When C aliases D it is covered by D's rotation instead.
        size_t blockCount      = 1;
        size_t offsetA         = 0;
        size_t bytesA          = 0;
        size_t strideA         = 0;
        size_t offsetB         = 0;
        size_t bytesB          = 0;
        size_t strideB         = 0;
        size_t offsetC         = 0;
        size_t bytesC          = 0;
        size_t strideC         = 0;
        size_t strideD         = 0;
        size_t strideE         = 0;
        size_t strideBias      = 0;
        size_t strideWorkspace = 0;
    };

    /**
     * Ceiling on rotation blocks. See the sizing comment in planScratch: the
     * per-block cost is paid once per candidate, so it has to stay bounded even
     * when the rotating budget divided by a tiny problem says otherwise.
     */
    constexpr size_t kMaxRotationBlocks = 128;

    // Saturating, so an overflowing layout ends up obviously too large rather
    // than wrapping to a plausible-looking small number.
    size_t addSat(size_t a, size_t b)
    {
        return (a > SIZE_MAX - b) ? SIZE_MAX : a + b;
    }

    size_t mulSat(size_t a, size_t b)
    {
        if(a == 0 || b == 0)
            return 0;
        return (a > SIZE_MAX / b) ? SIZE_MAX : a * b;
    }

    size_t alignUp(size_t v, size_t a = 256)
    {
        if(v > SIZE_MAX - (a - 1))
            return SIZE_MAX;
        return (v + a - 1) / a * a;
    }

    /**
     * Everything a candidate can write, laid out in one buffer.
     *
     * D is the obvious one, but an epilogue can also write the aux tensor, the
     * bias buffer when it is a gradient, and amaxD; a solution can write its own
     * workspace; and Stream-K writes the Synchronizer, which is shared across
     * every matmul on the handle.
     */
    ScratchLayout planScratch(const RocblasltContractionProblem& prob)
    {
        ScratchLayout layout;

        // Every span below is sized from the extents and strides alone, and the
        // measurement rebases A, B, C and D onto those spans. A batch offset is
        // added to the rebased pointer by the kernel and is not part of any of
        // that arithmetic, so a nonzero one would read or write outside the
        // block it was given.
        //
        // Today this cannot happen: rocblaslt_matmul_valid_args rejects a
        // nonzero offset unless the batch mode is POINTER_ARRAY, and tuning
        // already declines that mode. The check is here because that invariant
        // lives in another file, and if it ever relaxes this code must decline
        // rather than silently run off the end of a block.
        if(prob.batch_offset_a != 0 || prob.batch_offset_b != 0 || prob.batch_offset_c != 0
           || prob.batch_offset_d != 0)
            return layout;

        layout.bytesD = tensorSpanBytes(elementSizeOf(prob.d_type),
                                        prob.m,
                                        prob.n,
                                        prob.col_stride_d,
                                        prob.batch_count,
                                        prob.batch_stride_d);
        if(layout.bytesD == 0)
            return layout;

        // An in-place problem reads C through the D block, so that block has to
        // cover whichever tensor reaches further. C and D share a type but not
        // necessarily a leading dimension or batch stride, and sizing from D
        // alone lets a wider C read off the end of its block into whatever the
        // layout placed next.
        if(prob.C != nullptr && prob.C == prob.D)
        {
            bool expandedInPlaceC = false;
            layout.bytesInPlaceC  = tensorSpanBytes(elementSizeOf(prob.c_type),
                                                   prob.m,
                                                   prob.n,
                                                   prob.col_stride_c,
                                                   prob.batch_count,
                                                   prob.batch_stride_c,
                                                   &expandedInPlaceC);
            if(layout.bytesInPlaceC == 0 || expandedInPlaceC)
                return layout;

            layout.bytesD = std::max(layout.bytesD, layout.bytesInPlaceC);
        }

        if(is_e_enabled(prob.epilogue))
        {
            layout.bytesE = tensorSpanBytes(elementSizeOf(prob.aux_type),
                                            prob.m,
                                            prob.n,
                                            prob.col_stride_e,
                                            prob.batch_count,
                                            prob.batch_stride_e);
            if(layout.bytesE == 0)
                return layout;
        }

        // A gradient epilogue writes the bias buffer instead of reading it. The
        // length is a full D row or column depending on the source tensor, so
        // take the larger and stay conservative.
        if(is_grad_enabled(prob.epilogue) && prob.bias != nullptr)
        {
            // Through the same span helper as D and E. Multiplying length by
            // batch count ignores the stride, so a bias_stride larger than the
            // vector length under-allocates and the kernel writes past the end.
            const size_t biasElements = std::max(prob.m, prob.n);
            layout.bytesBias          = tensorSpanBytes(elementSizeOf(prob.bias_type),
                                               biasElements,
                                               1,
                                               biasElements,
                                               prob.batch_count,
                                               static_cast<size_t>(prob.bias_stride));
            if(layout.bytesBias == 0)
                return layout;
        }

        if(prob.amaxD != nullptr)
            layout.bytesAmaxD = 256;

        layout.bytesWorkspace = prob.workspaceSize;

        // op(A) is m by k and op(B) is k by n, so the stored extents depend on
        // the transposes; col_stride is the leading dimension.
        const size_t rowsA = (prob.trans_a == HIPBLAS_OP_N) ? prob.m : prob.k;
        const size_t colsA = (prob.trans_a == HIPBLAS_OP_N) ? prob.k : prob.m;
        const size_t rowsB = (prob.trans_b == HIPBLAS_OP_N) ? prob.k : prob.n;
        const size_t colsB = (prob.trans_b == HIPBLAS_OP_N) ? prob.n : prob.k;

        bool expandedA = false, expandedB = false, expandedC = false;

        layout.bytesA = tensorSpanBytes(elementSizeOf(prob.a_type),
                                        rowsA,
                                        colsA,
                                        prob.col_stride_a,
                                        prob.batch_count,
                                        prob.batch_stride_a,
                                        &expandedA);
        layout.bytesB = tensorSpanBytes(elementSizeOf(prob.b_type),
                                        rowsB,
                                        colsB,
                                        prob.col_stride_b,
                                        prob.batch_count,
                                        prob.batch_stride_b,
                                        &expandedB);

        // Only when C is its own buffer. An in-place problem reaches C through
        // D, which rotates already.
        if(prob.C != nullptr && prob.C != prob.D && prob.batch_C == nullptr)
        {
            layout.bytesC = tensorSpanBytes(elementSizeOf(prob.c_type),
                                            prob.m,
                                            prob.n,
                                            prob.col_stride_c,
                                            prob.batch_count,
                                            prob.batch_stride_c,
                                            &expandedC);
        }

        layout.strideD         = alignUp(layout.bytesD);
        layout.strideA         = alignUp(layout.bytesA);
        layout.strideB         = alignUp(layout.bytesB);
        layout.strideC         = alignUp(layout.bytesC);
        layout.strideE         = alignUp(layout.bytesE);
        layout.strideBias      = alignUp(layout.bytesBias);
        layout.strideWorkspace = alignUp(layout.bytesWorkspace);

        // Rotation needs a readable source for A and B, and spans it can size.
        // C is optional: a problem with no separate C still rotates the rest.
        //
        // A span that had to be expanded is an upper bound, not the size of the
        // caller's allocation, so copying it would read past the end of a
        // broadcast input. Swizzled A and B are declined for the mirror-image
        // reason: their physical layout is a re-tiled, padded form whose size is
        // not the logical span at all, so both the copy length and the per-block
        // stride would be wrong. Neither case is worth guessing at, and
        // declining only costs a cache-hot measurement.
        const bool rotatable
            = layout.bytesA != 0 && layout.bytesB != 0 && prob.A != nullptr && prob.B != nullptr
              && prob.batch_A == nullptr && prob.batch_B == nullptr && !expandedA && !expandedB
              && !(expandedC && layout.bytesC != 0) && !prob.swizzleA && !prob.swizzleB;

        // Everything a candidate reads or writes, so no iteration finds any of
        // it already resident. The scale vectors are deliberately left out: they
        // are O(m) or O(n) against O(mk), O(kn) and O(mn) for the tensors, so
        // their cache footprint is negligible, and sizing them safely means
        // handling every block-scaling variant.
        size_t perBlock = 0;
        for(size_t stride : {layout.strideA,
                             layout.strideB,
                             layout.strideC,
                             layout.strideD,
                             layout.strideE,
                             layout.strideBias,
                             layout.strideWorkspace})
            perBlock = addSat(perBlock, stride);

        size_t blocks = 1;
        if(rotatable && perBlock != 0 && perBlock != SIZE_MAX)
        {
            blocks = std::max<size_t>(1, TuningPolicy::rotatingBytes() / perBlock);

            // Each block costs a solve() and a seeding copy per candidate, so
            // the count has to stay bounded independently of how small the
            // problem is. A few-KB GEMM divides into hundreds of thousands of
            // blocks, and building that many kernel-invocation sets for every
            // one of hundreds of candidates dwarfs the measurement it is meant
            // to inform. Capping gives up on defeating the cache for problems
            // far smaller than it, which are launch-bound rather than
            // bandwidth-bound anyway.
            blocks = std::min<size_t>(blocks, kMaxRotationBlocks);
        }

        // Offsets are a pure function of the block count, so sizing is just
        // "lay it out, and if it does not fit, lay out fewer blocks".
        //
        // Saturating arithmetic throughout: a wrapped cursor would produce a
        // total small enough to pass the cap check below and offsets pointing
        // outside the allocation that total then reserves.
        auto place = [&](size_t n) {
            layout.blockCount = n;

            // A single block never has its A, B or C pointer redirected into
            // scratch, so reserving room for them would be dead space that can
            // push an otherwise fine shape over the cap.
            const bool rotating = n > 1;

            size_t cursor  = 0;
            auto   reserve = [&](size_t& offset, size_t span, size_t count) {
                offset = cursor;
                cursor = addSat(cursor, mulSat(span, count));
                cursor = alignUp(cursor);
            };

            reserve(layout.offsetD, layout.strideD, n);
            reserve(layout.offsetE, layout.strideE, n);
            reserve(layout.offsetBias, layout.strideBias, n);
            reserve(layout.offsetAmaxD, layout.bytesAmaxD, 1);
            reserve(layout.offsetWorkspace, layout.strideWorkspace, n);
            reserve(layout.offsetSynchronizer, kSynchronizerBytes, 1);
            reserve(layout.offsetA, layout.strideA, rotating ? n : 0);
            reserve(layout.offsetB, layout.strideB, rotating ? n : 0);
            reserve(layout.offsetC, layout.strideC, rotating ? n : 0);

            return cursor;
        };

        // Trim rather than decline when the full rotation exceeds the cap:
        // fewer blocks still beats measuring everything cache-hot.
        const size_t cap   = TuningScratch::instance().cap();
        size_t       total = place(blocks);
        while(blocks > 1 && total > cap)
        {
            // At least one block goes each pass, so this terminates.
            const size_t over   = total - cap;
            const size_t excess = over / perBlock + (over % perBlock != 0);
            blocks              = (excess >= blocks) ? 1 : blocks - excess;
            total               = place(blocks);
        }

        // A single block over the cap, including a saturated total, is left for
        // acquire to reject so it is reported as the capacity skip it is.
        layout.total  = total;
        layout.usable = true;
        return layout;
    }

    /**
     * Serialises all tuning. try_lock rather than lock so a second thread
     * reaching an untuned shape runs normally instead of stalling behind a
     * benchmark, and so the shared scratch has exactly one writer.
     */
    std::mutex& tuningLock()
    {
        static std::mutex gLock;
        return gLock;
    }

    bool streamIsCapturing(hipStream_t stream)
    {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        return hipStreamIsCapturing(stream, &status) == hipSuccess
               && status != hipStreamCaptureStatusNone;
    }

    /**
     * Benchmark candidates on isolated scratch and return the winning solution
     * index, or -1 to leave selection alone.
     *
     * Nothing here touches the caller's buffers. The winner is launched once by
     * the normal path afterwards, on the real ones.
     */
    int benchmarkAndSelectWinner(
        rocblaslt_handle                   handle,
        const RocblasltContractionProblem& prob,
        std::shared_ptr<void>              gemmData,
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>&
                                                library,
        std::shared_ptr<TensileLite::Hardware>& hardware,
        TensileLite::hip::SolutionAdapter*      adapter,
        TensileLite::TunedEntry&                winnerOut)
    {
        // Wall clock from function entry, not the sum of the timed spans.
        // Scratch allocation, rotation seeding, enumeration, support checks,
        // solve() and code-object loading are most of the cost and none of them
        // land in a span, so a span-only total said 10 seconds while the call
        // took 17. Starting the clock at entry also stops a slow setup from
        // handing the measurement loop a budget that is already spent.
        const double budgetUs  = TuningPolicy::perShapeBudgetUs();
        const auto   started   = std::chrono::steady_clock::now();
        auto         elapsedUs = [&] {
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()
                                                             - started)
                .count();
        };
        auto outOfBudget = [&] { return budgetUs > 0.0 && elapsedUs() > budgetUs; };

        const bool inPlace = (prob.C == prob.D);
        if(inPlace)
        {
            double alpha = 0.0, beta = 0.0;
            assignAlphaBeta(roc2TensileType(prob.compute_type, false),
                            hipDataType_to_tensile_type(prob.a_type),
                            prob.alpha,
                            prob.beta,
                            &alpha,
                            &beta);
            if(beta != 0.0)
            {
                // With C and D aliased, every launch overwrites the input for
                // the next launch. The default 2,000 launches reuse at most 128
                // blocks, so a nonzero beta compounds C repeatedly and quickly
                // measures Inf/NaN rather than the caller's problem. Resetting C
                // inside the one event span would time the copies, so decline
                // this case until the timing loop can provide an untimed reset
                // before every reuse.
                TensileLite::TuningCounters::instance().skipped++;
                log_info(__func__,
                         "tuning-cache: skipped, in-place C==D with nonzero beta "
                         "cannot be measured without mutating its input");
                return -1;
            }
        }

        const ScratchLayout layout = planScratch(prob);
        if(!layout.usable)
        {
            TensileLite::TuningCounters::instance().skipped++;
            log_info(__func__, "tuning-cache: skipped, output extent could not be established");
            return -1;
        }

        void* base = TuningScratch::instance().acquire(layout.total);
        if(base == nullptr)
        {
            TensileLite::TuningCounters::instance().skipped++;
            log_info(__func__, "tuning-cache: skipped, scratch exceeds cap or allocation failed");
            return -1;
        }

        auto* bytes = static_cast<uint8_t*>(base);

        // Candidates run against a copy of the problem whose outputs point into
        // scratch. C is left alone unless it aliases D, in which case both move
        // so setCEqualsD stays true and the predicate set does not change.
        RocblasltContractionProblem scratchProb = prob;

        scratchProb.D         = bytes + layout.offsetD;
        scratchProb.E         = layout.bytesE ? bytes + layout.offsetE : prob.E;
        scratchProb.workspace = layout.bytesWorkspace ? bytes + layout.offsetWorkspace : nullptr;
        scratchProb.Synchronizer = bytes + layout.offsetSynchronizer;
        if(layout.bytesAmaxD)
            scratchProb.amaxD = bytes + layout.offsetAmaxD;
        if(layout.bytesBias)
            scratchProb.bias = bytes + layout.offsetBias;
        if(inPlace)
            scratchProb.C = scratchProb.D;

        auto resetSynchronizer = [&] {
            return hipMemsetAsync(
                       bytes + layout.offsetSynchronizer, 0, kSynchronizerBytes, prob.stream)
                   == hipSuccess;
        };

        // Zero the whole buffer before anything reads it.
        //
        // Scratch comes from a raw hipMalloc, and not every span in it gets
        // seeded: an in-place problem reads C through D, a backward epilogue
        // reads the aux tensor it would otherwise only write, and a solution can
        // read its own workspace. Those reads would land on whatever the
        // allocator handed back, where a denormal pattern changes the timing of
        // the very thing being measured and a NaN or Inf can change it a great
        // deal. Zero is uniform, cheap and cannot denormal.
        if(hipMemsetAsync(bytes, 0, layout.total, prob.stream) != hipSuccess)
        {
            static_cast<void>(hipGetLastError());
            return -1;
        }

        // Every path after the first scratch operation must drain this stream
        // before the function releases the process-wide tuning lock. Most
        // successful paths are already synchronized by the timing events, but
        // enumeration failures, allocation exceptions and an exhausted budget
        // can return earlier. Without this guard, another thread can acquire the
        // lock on a different stream and reuse the same scratch while the
        // memset/copies above are still in flight.
        struct StreamDrain
        {
            hipStream_t stream;
            ~StreamDrain()
            {
                static_cast<void>(hipStreamSynchronize(stream));
                static_cast<void>(hipGetLastError());
            }
        } streamDrain{prob.stream};

        // Seed the rotation inputs over the zeros. Block 0 takes the caller's A
        // and B so the data is representative of what the kernel will really
        // see, and the rest are broadcast from it.
        size_t blocks = layout.blockCount;
        if(blocks > 1)
        {
            auto seed = [&](size_t offset, size_t stride, const void* src, size_t span) {
                if(hipMemcpyAsync(bytes + offset, src, span, hipMemcpyDeviceToDevice, prob.stream)
                   != hipSuccess)
                    return false;
                for(size_t b = 1; b < blocks; b++)
                    if(hipMemcpyAsync(bytes + offset + b * stride,
                                      bytes + offset,
                                      span,
                                      hipMemcpyDeviceToDevice,
                                      prob.stream)
                       != hipSuccess)
                        return false;
                return true;
            };

            const bool seeded = seed(layout.offsetA, layout.strideA, prob.A, layout.bytesA)
                                && seed(layout.offsetB, layout.strideB, prob.B, layout.bytesB)
                                && (layout.bytesC == 0
                                    || seed(layout.offsetC, layout.strideC, prob.C, layout.bytesC));

            if(!seeded || hipStreamSynchronize(prob.stream) != hipSuccess)
            {
                // Fall back to a single block rather than abandoning the shape.
                static_cast<void>(hipGetLastError());
                blocks = 1;
            }
        }

        if(blocks > 1 && (get_logger_layer_mode() & rocblaslt_layer_mode_log_info))
        {
            std::ostringstream msg;
            msg << "tuning-cache: rotating over " << blocks << " blocks, " << (layout.total >> 20)
                << " MiB scratch";
            log_info(__func__, msg.str());
        }

        // Default selection's pick, needed whichever way candidates are
        // enumerated. With the ranked prefix it is simply the first entry, but
        // an unranked enumeration has no such entry, so ask for it separately
        // and measure it first, which keeps the recorded baseline comparable to
        // the winner rather than depending on where it landed in the order.
        int  baselineIndex = -1;
        auto ranked        = getBestRawSolutions(prob, handle, gemmData, 1, prob.workspaceSize);
        if(!ranked.empty())
            baselineIndex = ranked.front()->index;

        // Candidate indexes. All-kernel enumeration reuses the public
        // getAllSolutions, which already dedups repeated indexes and drops
        // custom kernels that cannot run pointer-array batch.
        std::vector<int> candidateIndexes;
        if(TuningPolicy::allKernels())
        {
            std::vector<rocblaslt_matmul_heuristic_result> all;
            if(getAllSolutions(gemmData,
                               handle,
                               rocblaslt::RocGemmType::ROCBLASLT_GEMM,
                               all,
                               prob.workspaceSize)
               != rocblaslt_status_success)
                return -1;

            candidateIndexes.reserve(all.size());
            for(const auto& result : all)
                candidateIndexes.push_back(*(int*)result.algo.data);
        }
        else
        {
            auto prefix = getBestRawSolutions(
                prob, handle, gemmData, TuningPolicy::candidateCap(), prob.workspaceSize);
            candidateIndexes.reserve(prefix.size());
            for(const auto& solution : prefix)
                candidateIndexes.push_back(solution->index);
        }

        if(baselineIndex >= 0)
        {
            auto at = std::find(candidateIndexes.begin(), candidateIndexes.end(), baselineIndex);
            if(at != candidateIndexes.end())
                std::iter_swap(candidateIndexes.begin(), at);
            else
                candidateIndexes.insert(candidateIndexes.begin(), baselineIndex);
        }

        // Resolve through the same path replay uses. Enumeration and
        // getSolutionByIndex are different lookups and can disagree for one
        // index depending on which placeholder libraries have been
        // materialised; benchmarking the enumerated object and replaying the
        // resolved one would measure a kernel the cache never runs. Resolving
        // here makes the two identical by construction.
        std::vector<std::shared_ptr<TensileLite::ContractionSolution>> candidates;
        candidates.reserve(candidateIndexes.size());
        for(int index : candidateIndexes)
            if(auto solution = library->getSolutionByIndex(*hardware, index))
                candidates.push_back(std::move(solution));

        if(candidates.empty())
            return -1;

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
        {
            std::ostringstream msg;
            msg << "tuning-cache: " << candidates.size() << " candidates"
                << (TuningPolicy::allKernels() ? " (all kernels)" : " (ranked prefix)")
                << ", baseline index " << baselineIndex;
            log_info(__func__, msg.str());
        }

        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);

        // Owned, because the exits below are not all returns. isSolutionSupported
        // is called outside the per-candidate try block and can throw, and that
        // exception unwinds all the way to the tuning fallback in the caller. A
        // pair of raw handles destroyed only on the success path leaked two
        // events per such failure until event creation itself started failing.
        struct EventPair
        {
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;

            ~EventPair()
            {
                if(start)
                    static_cast<void>(hipEventDestroy(start));
                if(stop)
                    static_cast<void>(hipEventDestroy(stop));
            }
        } events;

        if(hipEventCreate(&events.start) != hipSuccess
           || hipEventCreate(&events.stop) != hipSuccess)
            return -1;

        hipEvent_t start = events.start;
        hipEvent_t stop  = events.stop;

        /**
         * Time n back-to-back launches as a single span, returning the total in
         * microseconds, or a negative value if any step failed.
         *
         * One event pair around the whole batch, not one per launch. Bracketing
         * and host-synchronising every launch drains the queue each iteration,
         * so each sample carries a full launch latency and the GPU never stays
         * fed; at a launch-overhead-bound size that overhead is a large part of
         * what gets compared between candidates.
         */
        // Advances across every span so blocks keep cycling rather than each
        // span replaying the same few.
        size_t rotor = 0;

        // Calibrated once here rather than per candidate, and only when the
        // flush is actually going to run, so a disabled flush costs nothing.
        const bool   flushICache = TuningPolicy::flushICache();
        const double flushUs     = flushICache ? ICacheFlush::instance().costUs(prob.stream) : 0.0;

        if(flushICache && (get_logger_layer_mode() & rocblaslt_layer_mode_log_info))
        {
            std::ostringstream msg;
            msg << "tuning-cache: icache flush enabled, " << flushUs << " us per launch";
            log_info(__func__, msg.str());
        }

        auto timeBatch
            = [&](const std::vector<std::vector<TensileLite::KernelInvocation>>& perBlock,
                  int                                                            n,
                  bool                                                           flush) -> double {
            if(hipEventRecord(start, prob.stream) != hipSuccess)
                return -1.0;

            for(int i = 0; i < n; i++)
            {
                const auto& kernels = perBlock[rotor++ % perBlock.size()];
                if(adapter->launchKernels(kernels, prob.stream, nullptr, nullptr) != hipSuccess)
                    return -1.0;

                // Inside the span, exactly as the bench client does it: the
                // flush has to land between one launch and the next to be worth
                // anything, and its cost comes back out of the mean below.
                if(flush && !ICacheFlush::instance().launch(prob.stream))
                    return -1.0;
            }

            if(hipEventRecord(stop, prob.stream) != hipSuccess)
                return -1.0;
            if(hipEventSynchronize(stop) != hipSuccess)
                return -1.0;

            float ms = 0.0f;
            if(hipEventElapsedTime(&ms, start, stop) != hipSuccess)
                return -1.0;

            return static_cast<double>(ms) * 1000.0;
        };

        int    bestIndex = -1;
        double bestUs    = std::numeric_limits<double>::max();

        /**
         * Per-launch time for one candidate, or a negative value if it cannot
         * run. `required` receives its workspace need.
         *
         * Deliberately the same shape as hipblaslt-bench's fixed-count mode:
         * coldIterations untimed launches, then one timed batch of
         * hotIterations, reported as the batch mean. Every candidate gets this,
         * so there is no shortlist and no ranking pass that could drop the real
         * winner before it is measured properly.
         */
        auto measure = [&](const std::shared_ptr<TensileLite::ContractionSolution>& solution,
                           size_t& required) -> double {
            required = 0;

            try
            {
                static_cast<void>(resetSynchronizer());

                rocblaslt_matmul_algo algo{};
                *(int*)algo.data         = solution->index;
                algo.max_workspace_bytes = prob.workspaceSize;

                // Inside the guard, not before it. isSolutionSupported resolves
                // and predicate-checks the candidate, and a lazily materialised
                // library can throw while doing so. Checking outside meant one
                // bad candidate aborted the whole search rather than being
                // skipped, which on the default exhaustive path threw away
                // every remaining candidate for that shape.
                if(rocblaslt_status_success
                   != isSolutionSupported(handle, scratchProb, gemmData, &algo, &required))
                    return -1.0;
                if(required > prob.workspaceSize)
                    return -1.0;

                // One kernel set per rotation block. Only the buffer pointers
                // differ, so the problem description built by
                // isSolutionSupported is shared; solve() bakes the pointers from
                // inputs into the invocation, which is why each block needs its
                // own set rather than one set relaunched.
                std::vector<std::vector<TensileLite::KernelInvocation>> perBlock;
                perBlock.reserve(blocks);
                for(size_t b = 0; b < blocks; b++)
                {
                    RocblasltContractionProblem blockProb = scratchProb;
                    blockProb.D = bytes + layout.offsetD + b * layout.strideD;
                    if(layout.bytesE)
                        blockProb.E = bytes + layout.offsetE + b * layout.strideE;
                    if(layout.bytesBias)
                        blockProb.bias = bytes + layout.offsetBias + b * layout.strideBias;
                    if(layout.bytesWorkspace)
                        blockProb.workspace
                            = bytes + layout.offsetWorkspace + b * layout.strideWorkspace;
                    if(inPlace)
                        blockProb.C = blockProb.D;
                    if(blocks > 1)
                    {
                        blockProb.A = bytes + layout.offsetA + b * layout.strideA;
                        blockProb.B = bytes + layout.offsetB + b * layout.strideB;
                        if(!inPlace && layout.bytesC != 0)
                            blockProb.C = bytes + layout.offsetC + b * layout.strideC;
                    }

                    auto inputs = GetTensileInputs(blockProb);
                    perBlock.push_back(solution->solve(data->problem, inputs, *hardware));
                }

                const int cold = TuningPolicy::coldIterations();
                const int hot  = TuningPolicy::hotIterations();

                // Warm-up does not flush, matching the bench client: its cold
                // loop launches the GEMM alone and only the timed lambda adds a
                // flush. Flushing here would just pay for evicting code that
                // the first timed flush evicts anyway.
                if(cold > 0 && timeBatch(perBlock, cold, false) < 0.0)
                    return -1.0;

                const double spanUs = timeBatch(perBlock, hot, flushICache);
                if(spanUs < 0.0)
                    return -1.0;

                // The span holds one flush per iteration. Subtracting its
                // measured cost leaves the GEMM, so the recorded time stays
                // comparable to a bench number and to entries tuned without the
                // flush. Clamped because a kernel faster than the flush's own
                // jitter could otherwise land at or below zero.
                return std::max(1e-3, spanUs / hot - flushUs);
            }
            catch(...)
            {
                // solve() can throw. One bad candidate must not fail the call.
                return -1.0;
            }
        };

        auto discardCandidate = [&] {
            static_cast<void>(hipStreamSynchronize(prob.stream));

            // Clear any sticky async error so it cannot be attributed to the
            // caller's real launch further down.
            static_cast<void>(hipGetLastError());
        };

        // One pass, every candidate measured the same way. There is no cheap
        // ranking round to drop a candidate before it has been measured
        // properly, which is what a shortlist did: on a large shape the eventual
        // winner screened 396th of 1025 and never reached the decision round.
        size_t measured  = 0;
        bool   truncated = false;
        for(const auto& solution : candidates)
        {
            if(outOfBudget())
            {
                truncated = true;
                break;
            }

            size_t       required = 0;
            const double us       = measure(solution, required);
            if(us < 0.0)
            {
                discardCandidate();
                continue;
            }
            measured++;

            if(solution->index == baselineIndex)
                winnerOut.baselineTimeUs = us;

            if(us < bestUs)
            {
                bestUs                           = us;
                bestIndex                        = solution->index;
                winnerOut.solutionIndex          = solution->index;
                winnerOut.requiredWorkspaceBytes = required;
                winnerOut.winnerTimeUs           = us;

                // From the measured object, which candidate resolution above
                // guarantees is the one getSolutionByIndex returns for this
                // index, so the name written here is the name replay will
                // compare against via getKernelNameFromAlgoIndex.
                winnerOut.kernelName = solution->kernelName;
            }
        }

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
        {
            std::ostringstream msg;
            msg << "tuning-cache: measured " << measured << " of " << candidates.size()
                << " candidates in " << static_cast<long long>(elapsedUs() / 1000.0) << " ms";
            log_info(__func__, msg.str());
        }

        // A search the budget cut short is thrown away rather than cached.
        //
        // The candidate list is not ordered by expected performance, so the
        // prefix that happened to fit in the budget is an arbitrary subset and
        // its best member is not the shape's best kernel. Persisting it would
        // freeze that arbitrary pick into the cache permanently, and because the
        // entry also marks the shape as tuned, no later run would revisit it.
        // Discarding costs a re-tune next process start and leaves the shape on
        // default selection until a run is allowed to finish.
        if(truncated)
        {
            TensileLite::TuningCounters::instance().skipped++;
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "tuning-cache: discarded, time budget stopped the search after " << measured
                    << " of " << candidates.size()
                    << " candidates; raise or unset HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE";
                log_info(__func__, msg.str());
            }
            return -1;
        }

        if(bestIndex < 0)
            return -1;

        // The first candidate is what default selection would have returned, so
        // recording it costs nothing and cannot be reconstructed later.
        winnerOut.baselineIndex = candidates.front()->index;

        // Fastest measured candidate wins outright, which is what
        // hipblaslt-bench does: its selection is a plain
        // "if(best_gpu_time > gpu_time_used)" with no floor on the improvement.
        //
        // An earlier version required the winner to beat default selection by a
        // margin. That was compensating for a timing loop which bracketed and
        // host-synchronised every launch, compressing the whole field to the
        // launch-overhead floor where the ordering really was noise. With
        // batched spans over rotating buffers the field separates, so the
        // margin only suppressed real wins.
        //
        // Note this persists an entry even when the winner is the default pick,
        // which is deliberate: it records that the shape was measured, so tune
        // mode does not re-benchmark it on every process start.
        return bestIndex;
    }
} // namespace

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocblasltContractionProblem *
 ******************************************************************************/
rocblaslt_status runContractionProblem(rocblaslt_handle                   handle,
                                       const rocblaslt_matmul_algo*       algo,
                                       const RocblasltContractionProblem& prob,
                                       std::shared_ptr<void>              gemmData)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
#ifdef HIPBLASLT_USE_ROCROLLER
        if(useRocRoller(handle, prob))
            return runRocRollerContractionProblem(handle, algo, prob);
#endif
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        rocblaslt_matmul_heuristic_result heuristicResult;

        if(prob.trans_a == HIPBLAS_OP_C)
            data->problem.setAOps({TensileLite::TensorOp::ComplexConjugate()});
        if(prob.trans_b == HIPBLAS_OP_C)
            data->problem.setBOps({TensileLite::TensorOp::ComplexConjugate()});

        if(algo == nullptr)
        {
            int returnAlgoCount;
            status = getBestSolutions(
                prob, handle, gemmData, 1, &heuristicResult, &returnAlgoCount, prob.workspaceSize);
            if(returnAlgoCount == 0)
                return rocblaslt_status_not_implemented;
            algo = &heuristicResult.algo;
        }
        updateTensileProblem(prob, data->problem);

        // Get the values of static member variables flush and rotating size from UserClientArguments
        UserClientArguments ClientArguments;
        bool                flush              = ClientArguments.GetFlushValue();
        int32_t             rotatingBufferSize = ClientArguments.GetRotatingBufferSizeValue();
        int32_t             hotIterations      = ClientArguments.GetHotIterationsValue();
        int32_t             coldIterations     = ClientArguments.GetColdIterationsValue();

        int* solutionIndex = (int*)algo->data;

        // Online tuning.
        //
        // This decides for itself rather than acting on a flag the heuristic
        // set, because hipblasLtMatmul accepts algo == nullptr and then runs its
        // own getBestSolutions above without ever entering
        // hipblasLtMatmulAlgoGetHeuristic. A shape reaching execution that way
        // would otherwise never be tuned.
        //
        // The winner is kept in a local. The caller's algo is const and must not
        // be rewritten; only the index used for this launch changes.
        int  tunedIndex  = -1;
        bool benchmarked = false;
        try
        {
            const auto& tuning = TensileLite::TuningModeSingleton::getInstance();

            const bool eligible = tuning.writes() && !prob.grouped_gemm
                                  && prob.batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY
                                  && !streamIsCapturing(prob.stream);

            if(eligible)
            {
                // The file has to be on disk-to-memory before deciding anything:
                // a matmul-only caller never enters the heuristic entry point,
                // which is the only other place that loads it, and would
                // otherwise re-benchmark every shape on every process start.
                TensileLite::getContractionProblemsFromFile(tuning.cachePath());

                const TensileLite::ProblemOverride key
                    = RocblasltContractionProblem2ProblemOverride(prob);
                TensileLite::OverrideMap& cache = TensileLite::OverrideMap::getMap();

                // "No usable entry", not "no entry". Entries that failed name
                // validation after a rebuild stay in the map, and testing for
                // mere presence made them permanently un-retunable.
                if(!tuning_cache_has_valid_entry(handle, key, prob, gemmData, prob.workspaceSize))
                {
                    // try_lock, not lock: a second thread meeting an untuned
                    // shape runs normally rather than stalling behind a
                    // benchmark, and the shared scratch keeps one writer.
                    std::unique_lock<std::mutex> guard(tuningLock(), std::try_to_lock);
                    if(guard.owns_lock()
                       && !tuning_cache_has_valid_entry(
                           handle, key, prob, gemmData, prob.workspaceSize))
                    {
                        TensileLite::TunedEntry winner;
                        benchmarked = true;
                        tunedIndex  = benchmarkAndSelectWinner(
                            handle, prob, gemmData, library, hardware, adapter, winner);

                        if(tunedIndex >= 0)
                        {
                            // solutionName is already set from the benchmarked
                            // solution object; re-deriving it from the index
                            // here is what previously recorded the wrong kernel.
                            winner.schemaVersion = TensileLite::TuningSchemaVersion::Current;
                            winner.source        = TensileLite::TuningEntrySource::OnlineTuning;
                            winner.buildStamp    = TensileLite::currentBuildStamp();

                            // Replace rather than add: any rows still here
                            // failed validation, and addIfAbsent would refuse
                            // outright if the fresh winner reused their index.
                            cache.replaceAll(key, winner);

                            const bool persisted
                                = TensileLite::appendTunedEntry(tuning.cachePath(), prob, winner);

                            auto& counters = TensileLite::TuningCounters::instance();
                            if(persisted)
                                counters.tuned++;
                            else
                                log_error(__func__,
                                          "tuning-cache: failed to write winner to "
                                              + tuning.cachePath()
                                              + "; it will be lost when this process exits");

                            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                            {
                                std::ostringstream msg;
                                msg << "tuning-cache: tuned shape, winner index " << tunedIndex
                                    << " at " << winner.winnerTimeUs << " us vs baseline "
                                    << winner.baselineTimeUs << " us [" << counters.summary()
                                    << "]";
                                log_info(__func__, msg.str());
                            }
                        }
                    }
                }
            }
        }
        catch(...)
        {
            // Tuning is an optimisation, so it fails quietly into normal
            // selection rather than failing the matmul.
            //
            // It must be caught *here* and not by the function-level handler
            // below. Lazy library loading, enumeration and solve() can all
            // throw, and when the caller passed algo == nullptr the status
            // variable already holds success from getBestSolutions above; an
            // escape would return that success to a caller whose GEMM never
            // launched, which is silent wrong output rather than an error. The
            // benchmarked flag stays set so data->problem is restored below.
            static_cast<void>(hipGetLastError());
            tunedIndex = -1;
            log_error(__func__, "tuning-cache: benchmarking threw; using default selection");
        }

        if(benchmarked)
        {
            // Unconditionally, not only on success. isSolutionSupported rewrites
            // data->problem from whatever it is handed, so after benchmarking it
            // describes the scratch problem whose D, and possibly C, point
            // elsewhere. That has to be undone even when no winner emerged, or
            // the caller's launch is built from the scratch description.
            updateTensileProblem(prob, data->problem);
        }

        if(tunedIndex >= 0)
            solutionIndex = &tunedIndex;

        data->algoIndex    = *solutionIndex;
        data->inputs       = GetTensileInputs(prob);

        if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
           || rocblaslt::Debug::Instance().printLogAsMarker()
           || rocblaslt::Debug::Instance().benchPrintCommand())
        {
            logBenchFromTensileDataGemm(data->problem,
                                        data->inputs,
                                        data->algoIndex,
                                        flush,
                                        rotatingBufferSize,
                                        coldIterations,
                                        hotIterations,
                                        false);
        }

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_profile)
        {
            logProfileFromTensileDataGemm(data->problem,
                                          data->inputs,
                                          data->algoIndex,
                                          flush,
                                          rotatingBufferSize,
                                          coldIterations,
                                          hotIterations,
                                          false);
        }

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_extended_profile)
        {
            std::string kernel_name   = getKernelNameFromAlgoIndex(handle, *algo);
            std::string Solution_name = getSolutionNameFromAlgoIndex(handle, *algo);

            logExtendedProfileFromTensileDataGemm(data->problem,
                                                  data->inputs,
                                                  data->algoIndex,
                                                  kernel_name,
                                                  Solution_name,
                                                  flush,
                                                  rotatingBufferSize,
                                                  coldIterations,
                                                  hotIterations,
                                                  false);
        }

        auto solution = library->getSolutionByIndex(data->problem, *hardware, *solutionIndex);
        if(prob.workspaceSize < solution->requiredWorkspaceSize(data->problem, *hardware))
        {
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "Input workspace size " << prob.workspaceSize
                    << " is less than the required workspace size ";
                msg << solution->requiredWorkspaceSize(data->problem, *hardware) << std::endl;
                log_info(__func__, msg.str());
            }
            return rocblaslt_status_invalid_value;
        }

        if(getenv("HIPBLASLT_BENCH_PERF") != nullptr)
        {
            auto autoGsuVal  = solution->calculateAutoGSU(data->problem, &(*hardware));
            auto Granularity = solution->computeGranularities(
                *hardware,
                data->problem.c().sizes()[0],
                data->problem.c().sizes()[1],
                data->problem.a().sizes()[data->problem.boundIndices()[0].a],
                data->problem.batchSize(0),
                autoGsuVal);

            hipblasltClientPerformanceArgs::totalGranularity = Granularity.totalGranularity;
            hipblasltClientPerformanceArgs::tilesPerCu       = Granularity.tilesPerCu;
            hipblasltClientPerformanceArgs::tile0Granularity
                = Granularity.tile0Granularity; // loss due to tile0
            hipblasltClientPerformanceArgs::tile1Granularity = Granularity.tile1Granularity;
            hipblasltClientPerformanceArgs::cuGranularity    = Granularity.cuGranularity;
            hipblasltClientPerformanceArgs::waveGranularity  = Granularity.waveGranularity;
            hipblasltClientPerformanceArgs::CUs              = Granularity.CUs;

            auto staticPerformanceModel = solution->staticPerformanceModel(
                data->problem.c().sizes()[0],
                data->problem.c().sizes()[1],
                data->problem.a().sizes()[data->problem.boundIndices()[0].a],
                data->problem.batchSize(0),
                Granularity.MT0,
                Granularity.MT1,
                Granularity.CUs,
                Granularity.totalGranularity,
                solution->sizeMapping.globalSplitU);

            hipblasltClientPerformanceArgs::memWriteBytesD
                = staticPerformanceModel.memWriteBytesD; //! Estimated memory writes D
            hipblasltClientPerformanceArgs::memReadBytes = staticPerformanceModel.memReadBytes;
        }

        if(!solution)
        {
#if 0
            std::ostream msg;
            print_once(msg << "\nrocblaslt error: No Tensile solution found for " << prob);
#endif
            status = rocblaslt_status_not_implemented;
        }
        else
        {
            // cu-fallback detection
            bool isCUFallback = solution->isFallbackForHW(*hardware);
            if(isCUFallback)
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "The solution is a cu-fallback for current HW. Use XCC=1 kernelArg."
                        << std::endl;
                    log_info(__func__, msg.str());
                }
            }
            // set XCC=1 to param when this is a fallback solution
            data->problem.setParams().setWGMXCC((isCUFallback ? 1 : 0));

            auto kernels = solution->solve(data->problem, GetTensileInputs(prob), *hardware);
            // Remove this after supports getting comgr buffers from hip.
            bool isPreloaded = false;
            if(rocblaslt::Debug::Instance().preload())
            {
                for(size_t i = 0; i < kernels.size(); i++)
                {
                    if(!kernels[i].codeObjectFile.empty())
                    {
                        auto isAlreadyLoaded = adapter->FindCodeObject(kernels[i].codeObjectFile);
                        if(!isAlreadyLoaded || !kernels[i].isSingleCall)
                        {
                            if(kernels[i].isSingleCall)
                            {
                                auto solutions = library->findAllSolutions(
                                    data->problem,
                                    *hardware,
                                    TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
                                std::vector<std::string> kernelNames;
                                for(auto s : solutions)
                                {
                                    kernelNames.push_back(s->KernelName());
                                }
                                static_cast<void>(adapter->initKernels(kernelNames));
                            }
                            else
                                static_cast<void>(adapter->initKernel(kernels[i].kernelName));
                        }
                    }
                }
                isPreloaded = true;
            }
            status = hip2RocStatus(
                adapter->launchKernels(kernels, prob.stream, nullptr, nullptr, isPreloaded));
            if(rocblaslt::Debug::Instance().printLogAsMarker())
                rocblaslt::Debug::Instance().logMarkerStop();
        }
    }
    catch(const std::exception& e)
    {
        status = rocblaslt_status_internal_error;
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
        status = rocblaslt_status_internal_error;
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status gemmCreate(RocblasltContractionProblem const& problem,
                            std::shared_ptr<void>&             gemmData,
                            size_t&                            gemmCount)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        // Check if pointer is valid
        // Update for the valid case: (alpha=0 && (A=NULL || B=NULL))
        if(problem.alpha == nullptr || problem.beta == nullptr || problem.C == nullptr
           || problem.D == nullptr
           || ((*((float*)problem.alpha)) && (problem.A == nullptr || problem.B == nullptr)))
        {
            log_error(__func__, "invalid data pointer");
            return rocblaslt_status_invalid_pointer;
        }
        gemmCount = 1;

        // Only when something will actually look it up. Building the key means a
        // device query and a string copy, and this runs on every gemm object
        // creation; callers who never asked for tuning should not pay for it.
        const bool cacheTuningKey = TensileLite::selectTuningFile().active;

        if(gemmData)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);
            updateTensileProblem(problem, data->problem);
            data->inputs         = GetTensileInputs(problem);
            data->enableEpilogue = problem.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;
            if(cacheTuningKey)
                data->tuningKey = RocblasltContractionProblem2ProblemOverride(problem);
        }
        else
        {
            TensileDataGemm data;
            data.problem        = ConstructTensileProblem(problem);
            data.inputs         = GetTensileInputs(problem);
            data.enableEpilogue = problem.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;
            if(cacheTuningKey)
                data.tuningKey = RocblasltContractionProblem2ProblemOverride(problem);

            gemmData = std::static_pointer_cast<void>(std::make_shared<TensileDataGemm>(data));
        }

        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status groupedGemmCreate(std::vector<RocblasltContractionProblem>& probs,
                                   std::shared_ptr<void>&                    gemmData,
                                   size_t&                                   gemmCount)
{
    gemmCount = probs.size();
    if(gemmCount == 0)
        return rocblaslt_status_success;
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        bool enableEpilogue = false;
        if(gemmData)
        {
            // Need to check if is same type?
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            TensileLite::ContractionProblemGroupedGemm& tensile_probs = data->problem;
            TensileLite::ContractionGroupedInputs&      groupedInputs = data->inputs;

            groupedInputs.grouped.clear();
            if(tensile_probs.gemms.size() != probs.size())
                tensile_probs.gemms.clear();

            for(int i = 0; i < probs.size(); i++)
            {
                // Check if pointer is valid for n != 0
                if(probs[i].n)
                {
                    if(probs[i].alpha == nullptr || probs[i].beta == nullptr
                       || probs[i].A == nullptr || probs[i].B == nullptr || probs[i].C == nullptr
                       || probs[i].D == nullptr)
                    {
                        log_error(__func__, "invalid data pointer");
                        return rocblaslt_status_invalid_pointer;
                    }
                }
                if(tensile_probs.gemms.size() != probs.size())
                    tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
                else
                    updateTensileProblem(probs[i], tensile_probs.gemms[i]);
                groupedInputs.grouped.push_back(GetTensileInputs(probs[i]));
                if(probs[i].epilogue != ROCBLASLT_EPILOGUE_DEFAULT)
                    enableEpilogue = true;
            }
            data->enableEpilogue = enableEpilogue;
        }
        else
        {
            TensileDataGroupedGemm                      data;
            TensileLite::ContractionProblemGroupedGemm& tensile_probs = data.problem;
            TensileLite::ContractionGroupedInputs&      groupedInputs = data.inputs;

            for(int i = 0; i < probs.size(); i++)
            {
                // Check if pointer is valid for n != 0
                if(probs[i].n)
                {
                    if(probs[i].alpha == nullptr || probs[i].beta == nullptr
                       || probs[i].A == nullptr || probs[i].B == nullptr || probs[i].C == nullptr
                       || probs[i].D == nullptr)
                    {
                        log_error(__func__, "invalid data pointer");
                        return rocblaslt_status_invalid_pointer;
                    }
                }
                tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
                groupedInputs.grouped.push_back(GetTensileInputs(probs[i]));
                if(probs[i].epilogue != ROCBLASLT_EPILOGUE_DEFAULT)
                    enableEpilogue = true;
            }
            data.enableEpilogue = enableEpilogue;

            gemmData
                = std::static_pointer_cast<void>(std::make_shared<TensileDataGroupedGemm>(data));
        }
        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

template <typename Tuning>
rocblaslt_status makeArgument(rocblaslt_handle             handle,
                              const rocblaslt::RocGemmType gemmType,
                              const rocblaslt_matmul_algo& algo,
                              const Tuning*                tuning,
                              void*                        workspace,
                              size_t                       workspaceSizeInBytes,
                              bool                         useUserArgs,
                              hipStream_t                  stream,
                              std::shared_ptr<void>        gemmData)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        int* solutionIndex = (int*)algo.data;
        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);

            data->algoIndex = *solutionIndex;
            auto solution   = library->getSolutionByIndex(data->problem, *hardware, *solutionIndex);

            if(tuning)
            {
                data->problem.setParams().setGSU(tuning->gsu);
                data->problem.setParams().setWgm(tuning->wgm);
                std::stringstream ss;
                if(!solution->checkInternalArgumentsSupport(data->problem, ss, true))
                {
                    data->problem.setParams().resetInternalArgs();
                    log_error(__func__, ss.str().c_str());
                    return rocblaslt_status_invalid_value;
                }
            }
            else
            {
                data->problem.setParams().resetInternalArgs();
            }

            // cu-fallback detection
            bool isCUFallback = solution->isFallbackForHW(*hardware);
            if(isCUFallback)
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "The solution is a cu-fallback for current HW. Use XCC=1 kernelArg."
                        << std::endl;
                    log_info(__func__, msg.str());
                }
            }
            // set XCC=1 to param when this is a fallback solution
            data->problem.setParams().setWGMXCC((isCUFallback ? 1 : 0));

            data->inputs.ws = workspace;

            // set workspace size from argument
            data->inputs.workspaceSize = workspaceSizeInBytes;
            data->problem.setWorkspaceSize(workspaceSizeInBytes);

            data->kernels = solution->solve(data->problem, data->inputs, *hardware);
        }
        else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);

            data->algoIndex = *solutionIndex;
            auto solution
                = library->getSolutionByIndex(data->problem.gemms[0], *hardware, *solutionIndex);

            if(tuning)
            {
                data->problem.gemms[0].setParams().setGSU(tuning->gsu);
                data->problem.gemms[0].setParams().setWgm(tuning->wgm);
                std::stringstream ss;
                if(!solution->checkInternalArgumentsSupport(data->problem.gemms[0], ss, true))
                {
                    data->problem.gemms[0].setParams().resetInternalArgs();
                    log_error(__func__, ss.str().c_str());
                    return rocblaslt_status_invalid_value;
                }
                for(size_t i = 1; i < data->problem.gemms.size(); i++)
                {
                    data->problem.gemms[i].setParams().setGSU(tuning->gsu);
                    data->problem.gemms[i].setParams().setWgm(tuning->wgm);
                }
            }
            else
            {
                for(size_t i = 0; i < data->problem.gemms.size(); i++)
                {
                    data->problem.gemms[i].setParams().resetInternalArgs();
                }
            }

            // cu-fallback detection
            bool isCUFallback = solution->isFallbackForHW(*hardware);
            if(isCUFallback)
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "The solution is a cu-fallback for current HW. Use XCC=1 kernelArg."
                        << std::endl;
                    log_info(__func__, msg.str()); // set xcc to 1 in the for-loop below
                }
            }
            uint16_t xcc_param = isCUFallback ? 1 : 0;
            for(size_t i = 0; i < data->problem.gemms.size(); i++)
            {
                // set XCC=1 to param when this is a fallback solution
                data->problem.gemms[i].setParams().setWGMXCC(xcc_param);
            }

            for(int i = 0; i < data->inputs.grouped.size(); i++)
            {
                data->inputs.grouped[i].ws = workspace;
            }
            data->inputs.ws = workspace;

            // set workspace size from argument
            data->problem.setWorkspaceSizeGroupedGemm(workspaceSizeInBytes);
            data->problem.setWorkspaceSize(workspaceSizeInBytes);
            for(int i = 0; i < data->inputs.grouped.size(); i++)
            {
                data->inputs.grouped[i].workspaceSize = workspaceSizeInBytes;
            }
            for(size_t i = 0; i < data->problem.gemms.size(); i++)
            {
                data->problem.gemms[i].setWorkspaceSizeGroupedGemm(workspaceSizeInBytes);
                data->problem.gemms[i].setWorkspaceSize(workspaceSizeInBytes);
            }

            data->useUserArgs = useUserArgs;
            if(useUserArgs)
            {
                data->kernels = solution->solveGroupedGemmGPU(
                    data->problem.gemms, data->inputs, *hardware, nullptr, workspace, stream);
            }
            else
            {
                size_t requiedHostSize
                    = solution->requiredHostWorkspaceSizePerProblem * data->problem.gemms.size();
                if(requiedHostSize > data->hipHostMemorySize)
                {
                    void* tmp = nullptr;
                    static_cast<void>(hipHostMalloc(&tmp, requiedHostSize, 0));
                    data->hipHostMemory
                        = std::shared_ptr<void>(tmp, [](auto p) { static_cast<void>(hipFree(p)); });
                    data->hipHostMemorySize = requiedHostSize;
                }

                data->kernels = solution->solveGroupedGemm(data->problem.gemms,
                                                           data->inputs,
                                                           *hardware,
                                                           data->hipHostMemory.get(),
                                                           data->hipHostMemorySize,
                                                           stream);
            }
        }
        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status runKernelFromInvocation(rocblaslt_handle       handle,
                                         rocblaslt::RocGemmType gemmType,
                                         std::shared_ptr<void>  gemmData,
                                         hipStream_t            stream,
                                         hipEvent_t             start,
                                         hipEvent_t             stop)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        // Get the values of static member variables flush and rotating size from UserClientArguments
        UserClientArguments ClientArguments;
        bool                flush              = ClientArguments.GetFlushValue();
        int32_t             rotatingBufferSize = ClientArguments.GetRotatingBufferSizeValue();
        int32_t             hotIterations      = ClientArguments.GetHotIterationsValue();
        int32_t             coldIterations     = ClientArguments.GetColdIterationsValue();

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);
            if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
               || rocblaslt::Debug::Instance().printLogAsMarker()
               || rocblaslt::Debug::Instance().benchPrintCommand())
            {
                logBenchFromTensileDataGemm(data->problem,
                                            data->inputs,
                                            data->algoIndex,
                                            flush,
                                            rotatingBufferSize,
                                            coldIterations,
                                            hotIterations,
                                            true);
            }
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_profile)
            {
                logProfileFromTensileDataGemm(data->problem,
                                              data->inputs,
                                              data->algoIndex,
                                              flush,
                                              rotatingBufferSize,
                                              coldIterations,
                                              hotIterations,
                                              true);
            }
            status = hip2RocStatus(adapter->launchKernels(data->kernels, stream, start, stop));
            if(rocblaslt::Debug::Instance().printLogAsMarker())
                rocblaslt::Debug::Instance().logMarkerStop();

            if(status == rocblaslt_status_success)
            {
                auto st = check_numerics_post_launch(
                    handle, stream, [&](uint32_t /*cid*/, auto visit) {
                        visit(data->problem, data->inputs.d);
                    });
                if(st != rocblaslt_status_success)
                    status = st;
            }
        }
        else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            if(data->useUserArgs)
            {
                log_error(__func__,
                          "GG is initialized with useUserArgs = true, workspace has no arguments.");
                return rocblaslt_status_not_initialized;
            }

            if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
               || rocblaslt::Debug::Instance().printLogAsMarker()
               || rocblaslt::Debug::Instance().benchPrintCommand())
            {
                logBenchFromTensileDataGemm(data->problem,
                                            data->inputs,
                                            data->algoIndex,
                                            flush,
                                            rotatingBufferSize,
                                            coldIterations,
                                            hotIterations,
                                            true);
            }
            if((get_logger_layer_mode() & rocblaslt_layer_mode_log_profile))
            {
                logProfileFromTensileDataGemm(data->problem,
                                              data->inputs,
                                              data->algoIndex,
                                              flush,
                                              rotatingBufferSize,
                                              coldIterations,
                                              hotIterations,
                                              false);
            }
            auto solution = library->getSolutionByIndex(*hardware, data->algoIndex);
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_extended_profile)
            {
                logExtendedProfileFromTensileDataGemm(data->problem,
                                                      data->inputs,
                                                      data->algoIndex,
                                                      solution->kernelName,
                                                      solution->solutionName,
                                                      flush,
                                                      rotatingBufferSize,
                                                      coldIterations,
                                                      hotIterations,
                                                      false);
            }

            status = hip2RocStatus(adapter->launchKernels(data->kernels, stream, start, stop));
            if(rocblaslt::Debug::Instance().printLogAsMarker())
                rocblaslt::Debug::Instance().logMarkerStop();

            if(status == rocblaslt_status_success)
            {
                // One launchKernels = one matmul = one call_id, shared across
                // all sub-problems so the first NaN'd sub-problem claims the
                // slot at the matmul-level id.
                auto st = check_numerics_post_launch(
                    handle, stream, [&](uint32_t /*cid*/, auto visit) {
                        const size_t N = std::min(data->problem.gemms.size(),
                                                  data->inputs.grouped.size());
                        for(size_t i = 0; i < N; ++i)
                            visit(data->problem.gemms[i], data->inputs.grouped[i].d);
                    });
                if(st != rocblaslt_status_success && status == rocblaslt_status_success)
                    status = st;
            }
        }
        else
        {
            return rocblaslt_status_invalid_value;
        }
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

rocblaslt_status getDeviceUserArgumentsValuesFromContractionProblem(rocblaslt_handle       handle,
                                                                    rocblaslt::RocGemmType gemmType,
                                                                    std::shared_ptr<void>  gemmData,
                                                                    void* hostDeviceUserArgs)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            auto  solution = library->getSolutionByIndex(*hardware, data->algoIndex);
            auto& problem  = data->problem.gemms[0];
            if(problem.activationComputeType() == rocisa::DataType::Float)
            {
                setDeviceUserArgs(data->problem.gemms,
                                  data->inputs,
                                  (TensileLite::DeviceUserArguments<float>*)hostDeviceUserArgs);
            }
            else
            {
                throw std::runtime_error("Currently only supports DeviceUserArguments<float>");
            }
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
        status = rocblaslt_status_success;
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: "
                       << "Is hostDeviceUserArgs not match the size of the problem type? " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: "
                       << "Is hostDeviceUserArgs not match the size of the problem type? " << prob);
#endif
    }

    return status;
}

// HIPBLASLT_CHECK_NUMERICS is intentionally NOT wired here: the kernel reads
// per-gemm D pointers from `deviceUserArgs` (a GPU buffer), which may differ
// from the host-side `data->inputs.grouped[i].d` captured at create time.
// Scanning the wrong buffer is worse than not scanning. Tracked as known gap.
rocblaslt_status runKernelFromNewDeviceUserArguments(rocblaslt_handle       handle,
                                                     rocblaslt::RocGemmType gemmType,
                                                     std::shared_ptr<void>  gemmData,
                                                     void*                  deviceUserArgs,
                                                     hipStream_t            stream)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        // Get the values of static member variables flush and rotating size from UserClientArguments
        UserClientArguments ClientArguments;
        bool                flush              = ClientArguments.GetFlushValue();
        int32_t             rotatingBufferSize = ClientArguments.GetRotatingBufferSizeValue();
        int32_t             hotIterations      = ClientArguments.GetHotIterationsValue();
        int32_t             coldIterations     = ClientArguments.GetColdIterationsValue();

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
               || rocblaslt::Debug::Instance().printLogAsMarker()
               || rocblaslt::Debug::Instance().benchPrintCommand())
            {
                logBenchFromTensileDataGemm(data->problem,
                                            data->inputs,
                                            data->algoIndex,
                                            flush,
                                            rotatingBufferSize,
                                            coldIterations,
                                            hotIterations,
                                            true);
            }
            if((get_logger_layer_mode() & rocblaslt_layer_mode_log_profile))
            {
                logProfileFromTensileDataGemm(data->problem,
                                              data->inputs,
                                              data->algoIndex,
                                              flush,
                                              rotatingBufferSize,
                                              coldIterations,
                                              hotIterations,
                                              false);
            }
            auto solution = library->getSolutionByIndex(*hardware, data->algoIndex);
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_extended_profile)
            {
                logExtendedProfileFromTensileDataGemm(data->problem,
                                                      data->inputs,
                                                      data->algoIndex,
                                                      solution->kernelName,
                                                      solution->solutionName,
                                                      flush,
                                                      rotatingBufferSize,
                                                      coldIterations,
                                                      hotIterations,
                                                      false);
            }

            for(auto& it : data->kernels)
            {
                uint8_t* arg = it.args.rawdata();
                if(solution->internalArgsSupport.useUniversalArgs)
                {
                    if(deviceUserArgs != nullptr)
                    {
                        int gemmCount = 0;
                        memcpy(&gemmCount, arg, sizeof(int));
                        gemmCount = gemmCount & 0x3FFFFFFF;
                        gemmCount = gemmCount | (2 << 30);
                        memcpy(arg, &gemmCount, sizeof(int));
                    }
                    memcpy(arg + TENSILE_COMMON_KERNEL_ARGS_SIZE, &deviceUserArgs, sizeof(void*));
                }
                else
                {
                    memcpy(arg + 4, &deviceUserArgs, sizeof(void*));
                }
            }
            status = hip2RocStatus(adapter->launchKernels(data->kernels, stream, nullptr, nullptr));
            if(rocblaslt::Debug::Instance().printLogAsMarker())
                rocblaslt::Debug::Instance().logMarkerStop();
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

// HIPBLASLT_CHECK_NUMERICS is intentionally NOT wired here: same reason as
// runKernelFromNewDeviceUserArguments above -- D pointers come from
// `deviceUserArgs` and may not match host-side records.
rocblaslt_status runKernelFromDeviceUserArguments(rocblaslt_handle             handle,
                                                  rocblaslt::RocGemmType       gemmType,
                                                  size_t                       gemmCount,
                                                  std::shared_ptr<void>        gemmData,
                                                  const rocblaslt_matmul_algo& algo,
                                                  void*                        deviceUserArgs,
                                                  void*                        workspace,
                                                  hipStream_t                  stream)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        int* solutionIndex = (int*)algo.data;
        // don't overwrite data->algoIndex = *solutionIndex; here
        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            auto solution = library->getSolutionByIndex(*hardware, *solutionIndex);
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            auto kernel = solution->solveGroupedGemmGPU(
                data->problem.gemms, data->inputs, *hardware, deviceUserArgs, workspace, stream);
            status = hip2RocStatus(adapter->launchKernels(kernel, stream, nullptr, nullptr));
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

/******************************************************************************
 * getBestSolutions calls Tensile's findTopSolutions and converts to          *
 * rocblaslt_matmul_heuristic_result.                                         *
 ******************************************************************************/

void _convertToHeuristicResultArray(
    std::vector<std::shared_ptr<TensileLite::ContractionSolution>>& solutions,
    int                                                             requestedAlgoCount,
    rocblaslt_matmul_heuristic_result                               heuristicResultsArray[],
    int*                                                            returnAlgoCount,
    size_t                                                          maxWorkSpaceBytes,
    const TensileLite::ContractionProblemGemm&                      problem,
    const TensileLite::Hardware&                                    hardware)
{
    *returnAlgoCount = std::min((int)solutions.size(), requestedAlgoCount);
    for(size_t i = 0; i < *returnAlgoCount; i++)
    {
        auto solution = solutions[i];
        memset(heuristicResultsArray[i].algo.data, 0, sizeof(heuristicResultsArray[i].algo.data));
        int* solutionIndex = (int*)(heuristicResultsArray[i].algo.data);
        *solutionIndex     = solution->index;
        heuristicResultsArray[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResultsArray[i].algo.fallback            = false;
        heuristicResultsArray[i].state                    = rocblaslt_status_success;
        heuristicResultsArray[i].workspaceSize = solution->requiredWorkspaceSize(problem, hardware);
    }
    for(size_t i = *returnAlgoCount; i < requestedAlgoCount; i++)
    {
        heuristicResultsArray[i].state = rocblaslt_status_invalid_value;
    }
}

template <typename T>
inline auto getSolutions(
    const T& inputs,
    const std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>&
                                                  library,
    const std::shared_ptr<TensileLite::Hardware>& hardware,
    TensileLite::ContractionProblemGemm&          tensile_prob,
    bool                                          enableEpilogue,
    const int&                                    requestedAlgoCount)
{
    auto solutions = library->findTopSolutions(tensile_prob, *hardware, requestedAlgoCount);
    return solutions;
}

std::vector<std::shared_ptr<TensileLite::ContractionSolution>>
    getBestRawSolutions(RocblasltContractionProblem const& prob,
                        rocblaslt_handle                   handle,
                        std::shared_ptr<void>              gemmData,
                        int                                requestedAlgoCount,
                        size_t                             maxWorkSpaceBytes)
{
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    static_cast<void>(get_library_and_adapter(&library, &deviceProp, &hardware, handle->device));

    if(!library)
    {
        return {};
    }

    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(prob, data->problem);

    bool enableEpilogue = prob.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

    auto solutions
        = getSolutions(prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if(solutions.size() == 0 && prob.compute_type == rocblaslt_compute_f32_fast_xf32)
    {
        log_api(__func__, "no solutions found, try to fallback");
        data->problem.setF32XdlMathOp(rocisa::DataType::Float);
        solutions = getSolutions(
            prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);
    }

    return solutions;
}

rocblaslt_status getBestSolutions(RocblasltContractionProblem const& prob,
                                  rocblaslt_handle                   handle,
                                  std::shared_ptr<void>              gemmData,
                                  int                                requestedAlgoCount,
                                  rocblaslt_matmul_heuristic_result  heuristicResultsArray[],
                                  int*                               returnAlgoCount,
                                  size_t                             maxWorkSpaceBytes)
{
#ifdef HIPBLASLT_USE_ROCROLLER
    if(useRocRoller(handle, prob))
        return getRocRollerBestSolutions(handle,
                                         prob,
                                         requestedAlgoCount,
                                         heuristicResultsArray,
                                         maxWorkSpaceBytes,
                                         returnAlgoCount);
#endif
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, &hardware, handle->device));

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(prob, data->problem);

    bool enableEpilogue = prob.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

    auto solutions
        = getSolutions(prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if(solutions.size() == 0 && prob.compute_type == rocblaslt_compute_f32_fast_xf32)
    {
        log_api(__func__, "no xf32 solutions found, try to fallback fp32");
        data->problem.setF32XdlMathOp(rocisa::DataType::Float);
        solutions = getSolutions(
            prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);
    }

    auto algoCount = min(static_cast<size_t>(requestedAlgoCount), solutions.size());
    memset(heuristicResultsArray, 0, sizeof(rocblaslt_matmul_heuristic_result) * algoCount);

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
    {
        std::ostringstream msg;
        for(size_t i = 0; i < algoCount; ++i)
        {
            auto& solution = solutions[i];
            msg << "getBestSolutions(): sol-idx = " << solution->index << ", (require TENSILE_DB set 0x2|0x4) sol-tag = " << solution->matchingTag() << std::endl;
        }
        log_info(__func__, msg.str());
    }

    _convertToHeuristicResultArray(solutions,
                                   requestedAlgoCount,
                                   heuristicResultsArray,
                                   returnAlgoCount,
                                   maxWorkSpaceBytes,
                                   data->problem,
                                   *hardware);

    return rocblaslt_status_success;
}

template <typename MyProblem>
rocblaslt_status getAllSolutions(MyProblem&                                      prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    log_api(__func__, "Entering function");

    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, &hardware, handle->device));

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    std::string deviceFullString(deviceProp->gcnArchName);
    std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

    std::set<std::shared_ptr<TensileLite::ContractionSolution>> solutions;
    std::shared_ptr<void>                                       tensile_prob;

    if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGemm>::value)
    {
        solutions = library->findAllSolutions(
            prob, *hardware, TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
    }
    else if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGroupedGemm>::value)
    {
        solutions = library->findAllSolutionsGroupedGemm(
            prob.gemms, *hardware, TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
    }
    log_api(__func__, "Found hardware solutions: ", solutions.size());

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if(solutions.size() == 0 && prob.f32XdlMathOp() == rocisa::DataType::XFloat32)
    {
        prob.setF32XdlMathOp(rocisa::DataType::Float);
        if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGemm>::value)
        {
            solutions = library->findAllSolutions(
                prob, *hardware, TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
        }
        else if constexpr(std::is_same<MyProblem,
                                       TensileLite::ContractionProblemGroupedGemm>::value)
        {
            solutions = library->findAllSolutionsGroupedGemm(
                prob.gemms, *hardware, TensileLite::SolutionLibrarySearchType::GEMM_TYPE_ONLY);
        }
    }

    heuristicResults.resize(solutions.size());

    int i                 = 0;
    int duplicated_counts = 0;
    for(auto solution : solutions)
    {
        // Custom kernels don't support general batched mode (pointer arrays)
        // Only check for ContractionProblemGemm (grouped gemm doesn't use batchMode)
        if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGemm>::value)
        {
            if(prob.batchMode() == TensileLite::ContractionProblemGemm::BATCHMODE::POINTER_ARRAY
               && !solution->sizeMapping.customKernelName.empty())
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "Skipping custom kernel " << solution->sizeMapping.customKernelName
                        << " - does not support batch_mode=POINTER_ARRAY" << std::endl;
                    log_info(__func__, msg.str());
                }
                continue;
            }
        }

        //workaround: findAllSolutions should get all solutions without duplications
        bool duplicated_sol = false;
        for(int j = 0; j < i; j++)
            if(*(int*)(heuristicResults[j].algo.data) == solution->index)
                duplicated_sol = true;
        if(duplicated_sol)
        {
            ++duplicated_counts;
            continue;
        }
        memset(&heuristicResults[i], 0, sizeof(rocblaslt_matmul_heuristic_result));
        memset(heuristicResults[i].algo.data, 0, sizeof(heuristicResults[i].algo.data));
        int* solutionIndex                           = (int*)(heuristicResults[i].algo.data);
        *solutionIndex                               = solution->index;
        heuristicResults[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResults[i].algo.fallback            = false;
        heuristicResults[i].state                    = rocblaslt_status_success;
        if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGemm>::value)
            heuristicResults[i].workspaceSize = solution->requiredWorkspaceSize(prob, *hardware);
        else
            heuristicResults[i].workspaceSize = 0;

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
        {
            std::ostringstream msg;
            msg << "getAllSolutions(): sol-idx = " << solution->index << ", (require TENSILE_DB set 0x2|0x4) sol-tag = " << solution->matchingTag() << std::endl;
            log_info(__func__, msg.str());
        }

        i++;
    }
    heuristicResults.resize(i);
    log_api(__func__, "Final hardware solutions: ", heuristicResults.size());
    log_api(__func__, "Leaving function, duplicated counts: ", duplicated_counts);

    return rocblaslt_status_success;
}

rocblaslt_status getAllSolutions(RocblasltContractionProblem&                    prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
#ifdef HIPBLASLT_USE_ROCROLLER
    if(useRocRoller(handle, prob))
        return getAllSolutionsRocRoller(prob, handle, heuristicResults, maxWorkSpaceBytes);
#endif
    auto tensile_prob = ConstructTensileProblem(prob);
    return getAllSolutions(tensile_prob, handle, heuristicResults, maxWorkSpaceBytes);
}

rocblaslt_status getAllSolutions(std::vector<RocblasltContractionProblem>&       probs,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    TensileLite::ContractionProblemGroupedGemm tensile_probs;
    for(int i = 0; i < probs.size(); i++)
    {
        tensile_probs.gemms.push_back(ConstructTensileProblem(probs[i]));
        tensile_probs.gemms[i].setGroupedGemm(true);
    }
    return getAllSolutions(tensile_probs, handle, heuristicResults, maxWorkSpaceBytes);
}

rocblaslt_status getAllSolutions(std::shared_ptr<void>                           gemmData,
                                 rocblaslt_handle                                handle,
                                 rocblaslt::RocGemmType                          gemmType,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
    rocblaslt_status status = rocblaslt_status_success;
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        status = getAllSolutions(data->problem, handle, heuristicResults, maxWorkSpaceBytes);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        status = getAllSolutions(data->problem, handle, heuristicResults, maxWorkSpaceBytes);
    }
    else
    {
        log_api(__func__, "Invalid gemm type", static_cast<int>(gemmType));
        status = rocblaslt_status_not_implemented;
    }
    return status;
}

using GemmMasterLibraryPtr
    = std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>;

static rocblaslt_status validateGemmLibraryAndHardware(const GemmMasterLibraryPtr&            library,
                                                       const std::shared_ptr<TensileLite::Hardware>& hardware,
                                                       const char* caller,
                                                       const char* nullHardwareMessage)
{
    if(!library)
        return rocblaslt_status_invalid_pointer;
    if(!hardware)
    {
        log_error(caller, nullHardwareMessage);
        return rocblaslt_status_invalid_pointer;
    }
    return rocblaslt_status_success;
}

rocblaslt_status
    getSolutionsFromIndex(rocblaslt_handle                                handle,
                          std::vector<int>&                               solutionIndex,
                          std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                          size_t                                          maxWorkSpaceBytes)
{
    GemmMasterLibraryPtr                   library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

    if(rocblaslt_status const st = validateGemmLibraryAndHardware(
           library,
           hardware,
           __func__,
           "Tensile Hardware is null; cannot resolve solution indices (library not "
           "initialized for this device?)");
       st != rocblaslt_status_success)
        return st;

    bool isOutOfBound = false;
    int  i            = 0;
    for(auto index : solutionIndex)
    {
#ifdef HIPBLASLT_USE_ROCROLLER
        if(index < 0)
        {
            isOutOfBound = false;
            getRocRollerSolutionsFromIndex(handle, index, heuristicResults, maxWorkSpaceBytes);
            continue;
        }

#endif
        auto solution = library->getSolutionByIndex(*hardware, index);
        if(!solution)
        {
            isOutOfBound = true;
            continue;
        }
        rocblaslt_matmul_heuristic_result result;
        memset(&result, 0, sizeof(rocblaslt_matmul_heuristic_result));
        memset(result.algo.data, 0, sizeof(result.algo.data));
        int* const pAlgoIndex = reinterpret_cast<int*>(result.algo.data);
        *pAlgoIndex           = solution->index;
        result.algo.max_workspace_bytes = maxWorkSpaceBytes;
        result.algo.fallback            = false;
        result.state                    = rocblaslt_status_success;
        result.workspaceSize            = 0;
        i++;
        heuristicResults.push_back(result);
    }
    if(isOutOfBound)
        return rocblaslt_status_invalid_value;
    return rocblaslt_status_success;
}

template <typename MyProblem, typename Inputs, typename Tuning>
rocblaslt_status isSolutionSupported(rocblaslt_handle       handle,
                                     MyProblem&             tensile_prob,
                                     Inputs&                inputs,
                                     rocblaslt_matmul_algo* algo,
                                     const Tuning*          tuning,
                                     size_t*                workspaceSizeInBytes)
{
    GemmMasterLibraryPtr                   library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

    if(rocblaslt_status const st = validateGemmLibraryAndHardware(
           library,
           hardware,
           __func__,
           "Tensile Hardware is null; cannot evaluate solution support (library not "
           "initialized for this device?)");
       st != rocblaslt_status_success)
        return st;

    *workspaceSizeInBytes = 0;

    int* const solutionIndex = reinterpret_cast<int*>(algo->data);
    // don't overwrite data->algoIndex = *solutionIndex; here
    if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGemm>::value)
    {
        auto solution = library->getSolutionByIndex(tensile_prob, *hardware, *solutionIndex);
        if(!solution)
        {
            std::ostringstream msg;
            msg << "getSolutionByIndex returned null for index " << *solutionIndex
                << " (solution missing from library map; check Tensile packaging or version "
                   "skew)";
            log_error(__func__, msg.str());
            return rocblaslt_status_invalid_value;
        }

        if(tuning)
        {
            tensile_prob.setParams().setGSU(tuning->gsu);
            tensile_prob.setParams().setWgm(tuning->wgm);
            std::stringstream ss;
            if(!solution->checkInternalArgumentsSupport(tensile_prob, ss, true))
            {
                tensile_prob.setParams().resetInternalArgs();
                log_error(__func__, ss.str().c_str());
                return rocblaslt_status_invalid_value;
            }
        }
        else
        {
            tensile_prob.setParams().resetInternalArgs();
        }

        // cu-fallback detection
        bool isCUFallback = solution->isFallbackForHW(*hardware);
        if(isCUFallback)
        {
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "The solution is a cu-fallback for current HW. Use XCC=1 for predicate."
                    << std::endl;
                log_info(__func__, msg.str());
            }
        }
        // set this flag for SW predicate
        tensile_prob.setParams().setFallbackStatus(isCUFallback);

        TensileLite::Task task(*hardware, tensile_prob, *solution);
        tensile_prob.setWorkspaceSize(algo->max_workspace_bytes);
        if(!(*solution->hardwarePredicate)(*hardware))
        {
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "Hardware match: " << solution->description();
                solution->hardwarePredicate->debugEval(*hardware, msg);
                msg << std::endl;
                log_info(__func__, msg.str());
            }
            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
        if(!(*solution->problemPredicate)(tensile_prob))
        {
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "Software match: " << solution->description();
                solution->problemPredicate->debugEval(tensile_prob, msg);
                msg << std::endl;
                log_info(__func__, msg.str());
            }

            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
        if(!(*solution->taskPredicate)(task))
        {
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "Software match: " << solution->description();
                solution->taskPredicate->debugEval(task, msg);
                msg << std::endl;
                log_info(__func__, msg.str());
            }

            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
        else
        {
            *workspaceSizeInBytes = solution->requiredWorkspaceSize(tensile_prob, *hardware);
        }
    }
    else if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGroupedGemm>::value)
    {
        auto solution
            = library->getSolutionByIndex(tensile_prob.gemms[0], *hardware, *solutionIndex);
        if(!solution)
        {
            std::ostringstream msg;
            msg << "getSolutionByIndex returned null for grouped gemm, index " << *solutionIndex
                << " (solution missing from library map; check Tensile packaging or version "
                   "skew)";
            log_error(__func__, msg.str());
            return rocblaslt_status_invalid_value;
        }

        if(tuning)
        {
            tensile_prob.gemms[0].setParams().setGSU(tuning->gsu);
            tensile_prob.gemms[0].setParams().setWgm(tuning->wgm);
            std::stringstream ss;
            if(!solution->checkInternalArgumentsSupport(tensile_prob.gemms[0], ss, true))
            {
                tensile_prob.gemms[0].setParams().resetInternalArgs();
                log_error(__func__, ss.str().c_str());
                return rocblaslt_status_invalid_value;
            }
            for(size_t i = 1; i < tensile_prob.gemms.size(); i++)
            {
                tensile_prob.gemms[i].setParams().setGSU(tuning->gsu);
                tensile_prob.gemms[i].setParams().setWgm(tuning->wgm);
            }
        }
        else
        {
            for(size_t i = 0; i < tensile_prob.gemms.size(); i++)
            {
                tensile_prob.gemms[i].setParams().resetInternalArgs();
            }
        }

        bool isSupported  = true;
        bool isNormalGemm = true;
        // cu-fallback detection
        bool isCUFallback = solution->isFallbackForHW(*hardware);
        if(isCUFallback)
        {
            if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
            {
                std::ostringstream msg;
                msg << "The solution is a cu-fallback for current HW. Use XCC=1 for predicate."
                    << std::endl;
                log_info(__func__, msg.str()); // will set status in the for-loop below
            }
        }
        auto problemWs = solution->requiredWorkspaceSizeGroupedGemm(tensile_prob.gemms, *hardware);
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            tensile_prob.gemms[i].setWorkspaceSize(algo->max_workspace_bytes);
            tensile_prob.gemms[i].setWorkspaceSizeGroupedGemm(problemWs);
            tensile_prob.gemms[i].setGroupedGemmCount(tensile_prob.gemms.size());
            // set this flag for SW predicate
            tensile_prob.gemms[i].setParams().setFallbackStatus(isCUFallback);
        }
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            if(!((*solution->hardwarePredicate)(*hardware)
                 && (*solution->problemPredicate)(tensile_prob.gemms[i])))
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "Match " << "[" << i << "]: " << solution->description();
                    solution->problemPredicate->debugEval(tensile_prob.gemms[i], msg);
                    msg << std::endl;
                    log_info(__func__, msg.str());
                }
                isSupported = false;
            }
        }
        if(!isSupported)
        {
            log_error(__func__, "Solution is not supported");
            return rocblaslt_status_invalid_value;
        }
        *workspaceSizeInBytes = problemWs;
    }
    return rocblaslt_status_success;
}

rocblaslt_status isSolutionSupported(rocblaslt_handle             handle,
                                     RocblasltContractionProblem& prob,
                                     std::shared_ptr<void>        gemmData,
                                     rocblaslt_matmul_algo*       algo,
                                     size_t*                      workspaceSizeInBytes)
{
#ifdef HIPBLASLT_USE_ROCROLLER
    if(useRocRoller(handle, prob))
        return isRocRollerSolutionSupported(handle, prob, algo, workspaceSizeInBytes);
#endif
    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(prob, data->problem);
    rocblaslt::RocTuningV2* tuning = nullptr;
    return isSolutionSupported(handle, data->problem, prob, algo, tuning, workspaceSizeInBytes);
}

rocblaslt_status isSolutionSupportedNoMutation(rocblaslt_handle                   handle,
                                               const RocblasltContractionProblem& prob,
                                               std::shared_ptr<void>              gemmData,
                                               rocblaslt_matmul_algo*             algo,
                                               size_t* workspaceSizeInBytes)
{
#ifdef HIPBLASLT_USE_ROCROLLER
    if(useRocRoller(handle, prob))
    {
        RocblasltContractionProblem probe = prob;
        return isRocRollerSolutionSupported(handle, probe, algo, workspaceSizeInBytes);
    }
#endif

    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    auto                             savedProblem = data->problem;
    RocblasltContractionProblem      probe        = prob;

    try
    {
        const rocblaslt_status status
            = isSolutionSupported(handle, probe, gemmData, algo, workspaceSizeInBytes);
        data->problem = std::move(savedProblem);
        return status;
    }
    catch(...)
    {
        data->problem = std::move(savedProblem);
        throw;
    }
}

template <typename T>
void setRestrictions(TensileLite::ContractionProblemGemm& tensile_prob,
                     const T*                             alpha,
                     const T*                             beta)
{
    tensile_prob.setAlphaRestriction(TensileLite::toScalarValueEnum(*alpha));
    tensile_prob.setBetaRestriction(TensileLite::toScalarValueEnum(*beta));
}

// Centralized type dispatch: maps rocisa::DataType to a pointer tag.
template <typename F>
rocblaslt_status dispatchByComputeType(rocisa::DataType dt, F&& f)
{
    switch(dt)
    {
    case rocisa::DataType::Float:
        return f(static_cast<float*>(nullptr));
    case rocisa::DataType::Double:
        return f(static_cast<double*>(nullptr));
    // Extend as needed:
    default:
        return rocblaslt_status_not_implemented;
    }
}

template <typename Tuning>
rocblaslt_status isSolutionSupported(rocblaslt_handle              handle,
                                     const rocblaslt::RocGemmType& gemmType,
                                     std::shared_ptr<void>         gemmData,
                                     rocblaslt_matmul_algo&        algo,
                                     const Tuning*                 tuning,
                                     size_t&                       workspaceSizeInBytes)
{
    if(!gemmData)
        return rocblaslt_status_invalid_pointer;
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        auto data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        if(!data)
            return rocblaslt_status_invalid_pointer;

        auto checkSupportForTypeTag = [&](auto tag) -> rocblaslt_status {
            using T = std::remove_pointer_t<decltype(tag)>;

            const T* a = std::get_if<T>(&data->inputs.alpha);
            const T* b = std::get_if<T>(&data->inputs.beta);
            if(!a || !b)
                return rocblaslt_status_not_implemented;

            setRestrictions<T>(data->problem, a, b);

            return isSolutionSupported(
                handle, data->problem, data->inputs, &algo, tuning, &workspaceSizeInBytes);
        };

        return dispatchByComputeType(data->problem.computeType(), checkSupportForTypeTag);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        auto data = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        if(!data)
            return rocblaslt_status_invalid_pointer;

        if(data->problem.gemms.empty())
            return rocblaslt_status_invalid_size;

        const rocisa::DataType dt = data->problem.gemms[0].computeType();

        // If mixed compute types are unsupported, enforce uniformity.
        for(const auto& p : data->problem.gemms)
        {
            if(p.computeType() != dt)
                return rocblaslt_status_not_implemented;
        }

        auto checkGroupedSupportForTypeTag = [&](auto tag) -> rocblaslt_status {
            using T = std::remove_pointer_t<decltype(tag)>;

            if(data->problem.gemms.size() != data->inputs.grouped.size())
                return rocblaslt_status_invalid_size;

            for(size_t i = 0; i < data->problem.gemms.size(); ++i)
            {
                TensileLite::ContractionProblemGemm&  prob = data->problem.gemms[i];
                const TensileLite::ContractionInputs& in   = data->inputs.grouped[i];

                const T* a = std::get_if<T>(&in.alpha);
                const T* b = std::get_if<T>(&in.beta);
                if(!a || !b)
                    return rocblaslt_status_not_implemented;

                setRestrictions<T>(prob, a, b);
            }

            return isSolutionSupported(
                handle, data->problem, data->inputs, &algo, tuning, &workspaceSizeInBytes);
        };

        return dispatchByComputeType(dt, checkGroupedSupportForTypeTag);
    }
    return rocblaslt_status_not_implemented;
}

rocblaslt_status getBestSolutions(rocblaslt_handle       handle,
                                  rocblaslt::RocGemmType gemmType,
                                  std::shared_ptr<void>  gemmData,
                                  const int              workspaceBytes,
                                  const int              requestedAlgoCount,
                                  std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults)
{
    log_api(__func__, "Entering function");

    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, &hardware, handle->device));

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        data->problem.setWorkspaceSize(workspaceBytes);
        auto solutions = getSolutions(data->inputs,
                                      library,
                                      hardware,
                                      data->problem,
                                      data->enableEpilogue,
                                      requestedAlgoCount);

        // when there is no solution for xfloat32, fallback comput_type to fp32
        if(solutions.size() == 0 && data->problem.f32XdlMathOp() == rocisa::DataType::XFloat32)
        {
            data->problem.setF32XdlMathOp(rocisa::DataType::Float);
            solutions = getSolutions(data->inputs,
                                     library,
                                     hardware,
                                     data->problem,
                                     data->enableEpilogue,
                                     requestedAlgoCount);
        }

        auto algoCount       = min(static_cast<size_t>(requestedAlgoCount), solutions.size());
        int  returnAlgoCount = 0;
        heuristicResults.clear();
        heuristicResults.resize(algoCount);

        if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
        {
            std::ostringstream msg;
            for(size_t i = 0; i < algoCount; ++i)
            {
                auto& solution = solutions[i];
                msg << "getBestSolutions(): sol-idx = " << solution->index << ", (require TENSILE_DB set 0x2|0x4) sol-tag = " << solution->matchingTag() << std::endl;
            }
            log_info(__func__, msg.str());
        }

        _convertToHeuristicResultArray(solutions,
                                       algoCount,
                                       heuristicResults.data(),
                                       &returnAlgoCount,
                                       workspaceBytes,
                                       data->problem,
                                       *hardware);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        for(int i = 0; i < data->problem.gemms.size(); i++)
        {
            data->problem.gemms[i].setWorkspaceSize(workspaceBytes);
            data->problem.gemms[i].setGroupedGemmCount(data->problem.gemms.size());
        }

        auto solutions = library->findTopSolutionsGroupedGemm(
            data->problem.gemms, *hardware, requestedAlgoCount);

        auto algoCount       = min(static_cast<size_t>(requestedAlgoCount), solutions.size());
        int  returnAlgoCount = 0;
        heuristicResults.clear();
        heuristicResults.resize(algoCount);

        _convertToHeuristicResultArray(solutions,
                                       algoCount,
                                       heuristicResults.data(),
                                       &returnAlgoCount,
                                       workspaceBytes,
                                       data->problem.gemms[0],
                                       *hardware);
    }

    return rocblaslt_status_success;
}

std::string getKernelNameFromData(rocblaslt_handle             handle,
                                  const rocblaslt::RocGemmType gemmType,
                                  std::shared_ptr<void>        gemmData)
{
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                     library;
    std::shared_ptr<hipDeviceProp_t> deviceProp;

    auto adapter = get_library_and_adapter(&library, &deviceProp, nullptr, handle->device);

    if(!library)
    {
        return std::string();
    }

    int                                        gsu = 0;
    int                                        wgm = 0;
    std::vector<TensileLite::KernelInvocation> kernels;

    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        kernels                               = data->kernels;
        gsu                                   = data->problem.getParams().gsu();
        wgm                                   = data->problem.getParams().wgm();
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        kernels = data->kernels;
        gsu     = data->problem.gemms[0].getParams().gsu();
        wgm     = data->problem.gemms[0].getParams().wgm();
    }
    std::string kernelName = "";
    if(kernels.empty())
        return kernelName;
    kernelName += kernels[0].kernelName;
    for(size_t i = 1; i < kernels.size(); i++)
    {
        kernelName += "; " + kernels[i].kernelName;
    }
    return kernelName;
}

std::string getSolutionNameFromData(rocblaslt_handle             handle,
                                    const rocblaslt::RocGemmType gemmType,
                                    std::shared_ptr<void>        gemmData)
{
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

    if(!library)
    {
        return std::string();
    }

    int gsu           = 0;
    int wgm           = 0;
    int solutionIndex = -1;

    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        solutionIndex                         = data->algoIndex;
        gsu                                   = data->problem.getParams().gsu();
        wgm                                   = data->problem.getParams().wgm();
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        solutionIndex = data->algoIndex;
        gsu           = data->problem.gemms[0].getParams().gsu();
        wgm           = data->problem.gemms[0].getParams().wgm();
    }
    if(solutionIndex == -1)
        return "";

#ifdef HIPBLASLT_USE_ROCROLLER
    if(solutionIndex < 0)
    {
        return rocRollerShortKernelNameFromEncodedSolutionIndex(solutionIndex);
    }
#endif

    auto solution = library->getSolutionByIndex(*hardware, solutionIndex);
    if(!solution)
        return "";

    std::string modifiedString = "";
    if(gsu != solution->sizeMapping.globalSplitU && gsu != 0)
    {
        modifiedString += "GSU: " + std::to_string(gsu);
    }

    if(wgm != solution->sizeMapping.workGroupMapping && wgm != 0)
    {
        if(modifiedString != "")
            modifiedString += ", ";
        modifiedString += "WGM: " + std::to_string(wgm);
    }
    auto solutionName = solution->solutionName;
    if(modifiedString != "")
        solutionName += " (Custom tuning: " + modifiedString + ")";
    return solutionName;
}

std::string getKernelNameFromAlgoIndex(rocblaslt_handle handle, const rocblaslt_matmul_algo& algo)
{
    int* solutionIndex = (int*)algo.data;

#ifdef HIPBLASLT_USE_ROCROLLER
    if(*solutionIndex < 0)
    {
        return rocRollerShortKernelNameFromEncodedSolutionIndex(*solutionIndex);
    }
#endif

    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

    if(!library)
    {
        return std::string();
    }

    auto solution = library->getSolutionByIndex(*hardware, *solutionIndex);
    return solution->kernelName;
}

std::string getSolutionNameFromAlgoIndex(rocblaslt_handle handle, const rocblaslt_matmul_algo& algo)
{
    int* solutionIndex = (int*)algo.data;
 
#ifdef HIPBLASLT_USE_ROCROLLER
    if(*solutionIndex < 0)
    {
        return rocRollerShortKernelNameFromEncodedSolutionIndex(*solutionIndex);
    }
#endif

    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    auto adapter = get_library_and_adapter(&library, &deviceProp, &hardware, handle->device);

    if(!library)
    {
        return std::string();
    }

    auto solution = library->getSolutionByIndex(*hardware, *solutionIndex);
    return solution->solutionName;
}

/***************************************************************
 * ! \brief  Initialize rocblaslt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocblaslt_createialize()
{
    static_cast<void>(get_library_and_adapter());
}

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool& rocblaslt_internal_tensile_is_initialized()
{
    static std::atomic_bool init;
    return init;
}

/***********************************************************************************
 * Templates for backward compatibility with old rocBLASLt API
 ***********************************************************************************/
// clang-format off
#define CREATECOMPATIBILITYFUNCTION(Tuning)                                                    \
    template rocblaslt_status makeArgument<Tuning>(rocblaslt_handle             handle,                        \
                                                   const rocblaslt::RocGemmType gemmType,                      \
                                                   const rocblaslt_matmul_algo& algo,                          \
                                                   const Tuning*                tuning,                        \
                                                   void*                        workspace,                     \
                                                   size_t                       workspaceSizeInBytes,          \
                                                   bool                         useUserArgs,                   \
                                                   hipStream_t                  stream,                        \
                                                   std::shared_ptr<void>        gemmData);                     \
    template rocblaslt_status isSolutionSupported<Tuning>(rocblaslt_handle       handle,                       \
                                                          const rocblaslt::RocGemmType& gemmType,              \
                                                          std::shared_ptr<void>         gemmData,              \
                                                          rocblaslt_matmul_algo&        algo,                  \
                                                          const Tuning*                 tuning,                \
                                                          size_t&                       workspaceSizeInBytes);
// clang-format on
CREATECOMPATIBILITYFUNCTION(rocblaslt::RocTuningV2)
