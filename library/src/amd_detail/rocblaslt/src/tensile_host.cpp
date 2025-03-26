/* ************************************************************************
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
#include "rocblaslt-types.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"

#ifdef USE_ROCROLLER
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

#include <atomic>
#include <complex>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <link.h>
#include <regex>
#include <string_view>
#include <unistd.h>

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
                                                         hipDataType            b_type,
                                                         const void*            B,
                                                         const void* const*     batch_B,
                                                         int64_t                ld_b,
                                                         int64_t                batch_stride_b,
                                                         const void*            beta,
                                                         hipDataType            c_type,
                                                         const void*            C,
                                                         const void* const*     batch_C,
                                                         int64_t                ld_c,
                                                         int64_t                batch_stride_c,
                                                         hipDataType            d_type,
                                                         void*                  D,
                                                         void* const*           batch_D,
                                                         int64_t                ld_d,
                                                         int64_t                batch_stride_d,
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
                                                         size_t                 scaleABlockRowSize,
                                                         size_t                 scaleABlockColSize,
                                                         size_t                 scaleBBlockRowSize,
                                                         size_t                 scaleBBlockColSize,
                                                         hipDataType            bias_type,
                                                         hipDataType            aux_type,
                                                         rocblaslt_epilogue     epilogue,
                                                         void*                  amaxD,
                                                         void*                  workspace,
                                                         size_t                 workspaceSize,
                                                         hipStream_t            stream,
                                                         void*                  Synchronizer,
                                                         bool                   swizzleA,
                                                         bool                   swizzleB)
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
    , b_type(b_type)
    , B(B)
    , batch_B(batch_B)
    , row_stride_b(1)
    , col_stride_b(ld_b)
    , batch_stride_b(batch_stride_b)
    , beta(beta)
    , c_type(c_type)
    , C(C)
    , batch_C(batch_C)
    , row_stride_c(1)
    , col_stride_c(ld_c)
    , batch_stride_c(batch_stride_c)
    , d_type(d_type)
    , D(D)
    , batch_D(batch_D)
    , row_stride_d(1)
    , col_stride_d(ld_d)
    , batch_stride_d(batch_stride_d)
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
    , scaleA(scaleA)
    , scaleB(scaleB)
    , scaleC(scaleC)
    , scaleD(scaleD)
    , scaleE(scaleE)
    , scaleAlphaVec(scaleAlphaVec)
    , scaleAType(scaleAType)
    , scaleBType(scaleBType)
    , scaleABlockRowSize(scaleABlockRowSize)
    , scaleABlockColSize(scaleABlockColSize)
    , scaleBBlockRowSize(scaleBBlockRowSize)
    , scaleBBlockColSize(scaleBBlockColSize)
    , bias_type(bias_type)
    , aux_type(aux_type)
    , epilogue(epilogue)
    , amaxD(amaxD)
    , workspace(workspace)
    , workspaceSize(workspaceSize)
    , stream(stream)
    , Synchronizer(Synchronizer)
    , swizzleA(swizzleA)
    , swizzleB(swizzleB)
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
#ifdef ROCM_USE_FLOAT8
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
#endif
        else
        {
            this->bias_type = this->d_type;
        }

        if(this->aux_type == HIPBLASLT_DATATYPE_INVALID)
        {
            this->aux_type = this->d_type;
        }

        if(this->trans_a == HIPBLAS_OP_C)
        {
            if(rocblaslt_is_complex_datatype(this->a_type))
                this->trans_a = HIPBLAS_OP_T;
        }
        if(this->trans_b == HIPBLAS_OP_C)
        {
            if(rocblaslt_is_complex_datatype(this->b_type))
                this->trans_b = HIPBLAS_OP_T;
        }
    }
}

namespace
{
    std::string getHipblasltSoPath()
    {
        return rocblaslt_internal_get_so_path("libhipblaslt");
    }

    static void assignAlphaBeta(TensileLite::DataType type,
                                const void*           alphaPtr,
                                const void*           betaPtr,
                                double*               alpha,
                                double*               beta)
    {
        switch(type)
        {
        case TensileLite::DataType::Half:
            *alpha = *(hipblasLtHalf*)alphaPtr;
            *beta  = *(hipblasLtHalf*)betaPtr;
            break;
        case TensileLite::DataType::Float:
        case TensileLite::DataType::XFloat32:
            *alpha = *(float*)alphaPtr;
            *beta  = *(float*)betaPtr;
            break;
        case TensileLite::DataType::Double:
            *alpha = *(double*)alphaPtr;
            *beta  = *(double*)betaPtr;
            break;
        case TensileLite::DataType::Int32:
            *alpha = *(int32_t*)alphaPtr;
            *beta  = *(int32_t*)betaPtr;
            break;
        default:
            throw std::runtime_error("Unsupported alpha, beta type.");
        }
    }

    inline bool gpu_arch_match(std::string_view gpu_arch, std::string_view pattern)
    {
        if(!pattern.length())
        {
            return true;
        }

        constexpr char    prefix[]   = "gfx";
        const std::size_t prefix_len = std::string_view(prefix).length();
        gpu_arch.remove_prefix(prefix_len);
        std::regex arch_regex(pattern.data());
        return std::regex_search(gpu_arch.data(), arch_regex);
    }

    inline TensileLite::ActivationType getTensileActivationType(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_RELU:
        case ROCBLASLT_EPILOGUE_RELU_BIAS:
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
        case ROCBLASLT_EPILOGUE_SWISH_EXT:
        case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
            return TensileLite::ActivationType::Silu;
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
        case ROCBLASLT_EPILOGUE_BIAS:
        case ROCBLASLT_EPILOGUE_BGRADA:
        case ROCBLASLT_EPILOGUE_BGRADB:
        case ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT:
            return true;
            break;
        default:
            return false;
            break;
        }
        return false;
    }

    TensileLite::DataType hip2TensileType(hipDataType type)
    {
        switch(type)
        {
        case HIP_R_32F:
            return TensileLite::DataType::Float;
        case HIP_R_16F:
            return TensileLite::DataType::Half;
        case HIP_R_64F:
            return TensileLite::DataType::Double;
        case HIP_R_16BF:
            return TensileLite::DataType::BFloat16;
        case HIP_R_8F_E4M3_FNUZ:
            return TensileLite::DataType::Float8_fnuz;
        case HIP_R_8F_E5M2_FNUZ:
            return TensileLite::DataType::BFloat8_fnuz;
#ifdef ROCM_USE_FLOAT8
        case HIP_R_8F_E4M3:
            return TensileLite::DataType::Float8;
        case HIP_R_8F_E5M2:
            return TensileLite::DataType::BFloat8;
#endif
        case HIP_R_8I:
            return TensileLite::DataType::Int8;
        case HIP_R_32I:
            return TensileLite::DataType::Int32;
        case HIP_R_6F_E2M3_EXT: // FIXME: fix this when tensile provide FP6 type
            return TensileLite::DataType::Float8;
        case HIP_R_6F_E3M2_EXT: // FIXME: fix this when tensile provide BF6 type
            return TensileLite::DataType::Float8;
        case HIP_R_4F_E2M1_EXT: // FIXME: fix this when tensile provide FP4 type
            return TensileLite::DataType::Float8;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return TensileLite::DataType::None;
    }

    hipDataType tensile2HipType(TensileLite::DataType type)
    {
        switch(type)
        {
        case TensileLite::DataType::Float:
            return HIP_R_32F;
        case TensileLite::DataType::Half:
            return HIP_R_16F;
        case TensileLite::DataType::Double:
            return HIP_R_64F;
        case TensileLite::DataType::BFloat16:
            return HIP_R_16BF;
        case TensileLite::DataType::Float8_fnuz:
            return HIP_R_8F_E4M3_FNUZ;
        case TensileLite::DataType::BFloat8_fnuz:
            return HIP_R_8F_E5M2_FNUZ;
#ifdef ROCM_USE_FLOAT8
        case TensileLite::DataType::Float8:
            return HIP_R_8F_E4M3;
        case TensileLite::DataType::BFloat8:
            return HIP_R_8F_E5M2;
#endif
        case TensileLite::DataType::Int8:
            return HIP_R_8I;
        case TensileLite::DataType::Int32:
            return HIP_R_32I;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return HIP_R_32F;
    }

    TensileLite::DataType roc2TensileType(rocblaslt_compute_type type, bool fallback = true)
    {
        switch(type)
        {
        case rocblaslt_compute_f16: // setting compute_type to f16_r will fallback to f32_r
            return fallback ? TensileLite::DataType::Float : TensileLite::DataType::Half;
        case rocblaslt_compute_f32:
        case rocblaslt_compute_f32_fast_xf32:
        case rocblaslt_compute_f32_fast_f16:
        case rocblaslt_compute_f32_fast_bf16:
        case rocblaslt_compute_f32_fast_f8_fnuz:
        case rocblaslt_compute_f32_fast_bf8_fnuz:
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
#ifdef ROCM_USE_FLOAT8
        case rocblaslt_compute_f32_fast_f8:
        case rocblaslt_compute_f32_fast_bf8:
        case rocblaslt_compute_f32_fast_f8bf8:
        case rocblaslt_compute_f32_fast_bf8f8:
#endif
            return TensileLite::DataType::Float;
        case rocblaslt_compute_f64:
            return TensileLite::DataType::Double;
        case rocblaslt_compute_i32:
            return TensileLite::DataType::Int32;
        default:
            throw std::runtime_error("Unsupported type.");
        }
        return TensileLite::DataType::None;
    }

    inline const TensileLite::DataType
        roc2TensileComputeInputType(const TensileLite::DataType&  typeA,
                                    const TensileLite::DataType&  typeB,
                                    const rocblaslt_compute_type& typeCompute)
    {
        switch(typeCompute)
        {
        case rocblaslt_compute_f32_fast_f16:
            return TensileLite::DataType::Half;
        case rocblaslt_compute_f32_fast_bf16:
            return TensileLite::DataType::BFloat16;
        case rocblaslt_compute_f32_fast_f8_fnuz:
            return TensileLite::DataType::Float8_fnuz;
        case rocblaslt_compute_f32_fast_bf8_fnuz:
            return TensileLite::DataType::BFloat8_fnuz;
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
            return TensileLite::DataType::Float8BFloat8_fnuz;
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
            return TensileLite::DataType::BFloat8Float8_fnuz;
#ifdef ROCM_USE_FLOAT8
        case rocblaslt_compute_f32_fast_f8:
            return TensileLite::DataType::Float8;
        case rocblaslt_compute_f32_fast_bf8:
            return TensileLite::DataType::BFloat8;
        case rocblaslt_compute_f32_fast_f8bf8:
            return TensileLite::DataType::Float8BFloat8;
        case rocblaslt_compute_f32_fast_bf8f8:
            return TensileLite::DataType::BFloat8Float8;
#endif
        default:;
        }

        if(typeA == TensileLite::DataType::Float8_fnuz
           && typeB == TensileLite::DataType::BFloat8_fnuz)
        {
            return TensileLite::DataType::Float8BFloat8_fnuz;
        }
        else if(typeA == TensileLite::DataType::BFloat8_fnuz
                && typeB == TensileLite::DataType::Float8_fnuz)
        {
            return TensileLite::DataType::BFloat8Float8_fnuz;
        }

#ifdef ROCM_USE_FLOAT8
        if(typeA == TensileLite::DataType::Float8 && typeB == TensileLite::DataType::BFloat8)
        {
            return TensileLite::DataType::Float8BFloat8;
        }
        else if(typeA == TensileLite::DataType::BFloat8 && typeB == TensileLite::DataType::Float8)
        {
            return TensileLite::DataType::BFloat8Float8;
        }
#endif

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
        auto                               typeATensile = hip2TensileType(typeA);
        auto                               typeBTensile = hip2TensileType(typeB);
        std::vector<TensileLite::DataType> biasDataTypeWhiteList; // dummy
        std::vector<int>                   biasSrcWhiteList; // dummy
        return TensileLite::ContractionProblemGemm::createDefaultProblem(
            (opA != HIPBLAS_OP_N),
            (opB != HIPBLAS_OP_N),
            typeATensile,
            typeBTensile,
            hip2TensileType(typeC),
            hip2TensileType(typeD),
            roc2TensileType(typeCompute),
            roc2TensileType(typeCompute),
            roc2TensileComputeInputType(typeATensile, typeBTensile, typeCompute),
            roc2TensileType(typeCompute),
            alpha,
            beta,
            false,
            false,
            biasDataTypeWhiteList,
            biasSrcWhiteList,
            isGroupedGemm,
            maxWorkspaceBytes);
    }

    const char* tensileComputeInputType_to_bench_string(TensileLite::DataType typeCompute,
                                                        TensileLite::DataType F32XdlMathOp,
                                                        TensileLite::DataType typeComputeInput,
                                                        TensileLite::DataType typeA,
                                                        TensileLite::DataType typeB)
    {
        switch(typeCompute)
        {
        case TensileLite::DataType::Float:
            break;
        case TensileLite::DataType::Double:
            return "f64_r";
            break;
        case TensileLite::DataType::Int32:
            return "i32_r";
            break;
        default:
            throw std::runtime_error("Unsupported type.");
        }

        if(F32XdlMathOp == TensileLite::DataType::XFloat32)
        {
            return "xf32_r";
        }
        else if(typeComputeInput == TensileLite::DataType::BFloat16
                && typeA == TensileLite::DataType::Half && typeB == TensileLite::DataType::Half)
        {
            return "f32_bf16_r";
        }
        else if(typeComputeInput == TensileLite::DataType::Half
                && (typeA == TensileLite::DataType::Float8_fnuz
                        && typeB == TensileLite::DataType::Half
                    || typeA == TensileLite::DataType::Half
                           && typeB == TensileLite::DataType::Float8_fnuz))
        {
            return "f32_f16_r";
        }
        else
        {
            return "f32_r";
        }
    }

    const char* tensileComputeInputType_to_profile_string(TensileLite::DataType typeCompute,
                                                          TensileLite::DataType F32XdlMathOp,
                                                          TensileLite::DataType typeComputeInput,
                                                          TensileLite::DataType typeA,
                                                          TensileLite::DataType typeB)
    {
        switch(typeCompute)
        {
        case TensileLite::DataType::Float:
            break;
        case TensileLite::DataType::Double:
            return "c_f64_r";
            break;
        case TensileLite::DataType::Int32:
            return "c_i32_r";
            break;
        default:
            throw std::runtime_error("Unsupported type.");
        }

        if(F32XdlMathOp == TensileLite::DataType::XFloat32)
        {
            return "c_xf32_r";
        }
        else if(typeComputeInput == TensileLite::DataType::BFloat16
                && typeA == TensileLite::DataType::Half && typeB == TensileLite::DataType::Half)
        {
            return "c_f32_fast_bf16_r";
        }
        else if(typeComputeInput == TensileLite::DataType::Half
                && (typeA == TensileLite::DataType::Float8_fnuz
                        && typeB == TensileLite::DataType::Half
                    || typeA == TensileLite::DataType::Half
                           && typeB == TensileLite::DataType::Float8_fnuz))
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
        case TensileLite::ActivationType::Relu:
            return "relu";
            break;
        case TensileLite::ActivationType::Silu:
        case TensileLite::ActivationType::Swish:
            return "swish";
        case TensileLite::ActivationType::None:
        default:
            return "none";
            break;
        }
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
            ToString(inputs.alpha),
            "--beta",
            ToString(inputs.beta),
            "--transA",
            problem.transA() ? "T" : "N",
            "--transB",
            problem.transB() ? "T" : "N",
            "--batch_count",
            problem.batchSize(0),
            "--scaleA",
            problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
            "--scaleB",
            problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
            problem.useScaleCD() ? "--scaleC" : "",
            problem.useScaleCD() ? "--scaleD" : "",
            problem.useScaleAlphaVec() ? "--scaleAlpha_vector" : "",
            problem.useGradient() ? "--gradient" : "",
            problem.useE() ? "--use_e" : "",
            problem.useBias() ? "--bias_vector" : "",
            problem.useBias() ? "--bias_source" : "",
            problem.useBias() ? problem.tensor(problem.biasSrc()).getName() : "",
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
            "--aux_type",
            hipDataType_to_bench_string(tensile2HipType(problem.e().dataType())),
            problem.getParams().gsu() ? "--splitk" : "",
            problem.getParams().gsu() ? std::to_string(problem.getParams().gsu()) : "",
            problem.getParams().wgm() ? "--wgm" : "",
            problem.getParams().wgm() ? std::to_string(problem.getParams().wgm()) : "",
            "--compute_type",
            tensileComputeInputType_to_bench_string(problem.computeType(),
                                                    problem.f32XdlMathOp(),
                                                    problem.computeInputType(),
                                                    problem.a().dataType(),
                                                    problem.b().dataType()),
            "--algo_method",
            "index",
            "--solution_index",
            solutionIndex,
            "--activation_type",
            tensileActivationtType_to_bench_string(problem.getParams().activationEnum()),
            flush ? "--flush" : "",
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
            rocblaslt::Debug::Instance().logMarkerStop();
        }
    }

    inline void logProfileFromTensileDataGemm(const TensileLite::ContractionProblemGemm& problem,
                                              const TensileLite::ContractionInputs&      inputs,
                                              bool                                       flush,
                                              const int32_t& rotatingBufferSize,
                                              const int32_t& coldIterations,
                                              const int32_t& hotIterations,
                                              bool           isCpp)
    {
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
                    ToString(inputs.alpha),
                    "beta",
                    ToString(inputs.beta),
                    "transA",
                    problem.transA() ? "T" : "N",
                    "transB",
                    problem.transB() ? "T" : "N",
                    "batch_count",
                    problem.batchSize(0),
                    "scaleA",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
                    "scaleB",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
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
                                                              problem.computeInputType(),
                                                              problem.a().dataType(),
                                                              problem.b().dataType()),
                    "activation_type",
                    tensileActivationtType_to_bench_string(problem.getParams().activationEnum()),
                    "flush",
                    flush ? "true" : "false",
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
                    ToString(inputs.alpha),
                    "beta",
                    ToString(inputs.beta),
                    "transA",
                    problem.transA() ? "T" : "N",
                    "transB",
                    problem.transB() ? "T" : "N",
                    "batch_count",
                    problem.batchSize(0),
                    "scaleA",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
                    "scaleB",
                    problem.useScaleAB().empty() ? 0 : (problem.useScaleAB() == "Vector" ? 2 : 1),
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
                                                              problem.computeInputType(),
                                                              problem.a().dataType(),
                                                              problem.b().dataType()),
                    "activation_type",
                    tensileActivationtType_to_bench_string(problem.getParams().activationEnum()),
                    "flush",
                    flush ? "true" : "false",
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
            "--aux_type",
            hipDataType_to_bench_string(tensile2HipType(problem.gemms[0].e().dataType())),
            problem.gemms[0].getParams().gsu() ? "--splitk" : "",
            problem.gemms[0].getParams().gsu() ? std::to_string(problem.gemms[0].getParams().gsu())
                                               : "",
            problem.gemms[0].getParams().wgm() ? "--wgm" : "",
            problem.gemms[0].getParams().wgm() ? std::to_string(problem.gemms[0].getParams().wgm())
                                               : "",
            "--compute_type",
            tensileComputeInputType_to_bench_string(problem.gemms[0].computeType(),
                                                    problem.gemms[0].f32XdlMathOp(),
                                                    problem.gemms[0].computeInputType(),
                                                    problem.gemms[0].a().dataType(),
                                                    problem.gemms[0].b().dataType()),
            "--algo_method",
            "index",
            "--solution_index",
            solutionIndex,
            "--activation_type",
            tensileActivationtType_to_bench_string(problem.gemms[0].getParams().activationEnum()),
            flush ? "--flush" : "",
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
            rocblaslt::Debug::Instance().logMarkerStop();
        }
    }

    inline void
        logProfileFromTensileDataGemm(const TensileLite::ContractionProblemGroupedGemm& problem,
                                      const TensileLite::ContractionGroupedInputs&      inputs,
                                      bool                                              flush,
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
                                                      problem.gemms[0].computeInputType(),
                                                      problem.gemms[0].a().dataType(),
                                                      problem.gemms[0].b().dataType()),
            "activation_type",
            tensileActivationtType_to_bench_string(problem.gemms[0].getParams().activationEnum()),
            "flush",
            flush ? "true" : "false",
            "rotating",
            rotatingBufferSize,
            "cold_iters",
            coldIterations,
            "iters",
            hotIterations);
    }
#undef GEN_BENCH_ARG

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
        assignAlphaBeta(compute_type, prob.alpha, prob.beta, &alpha, &beta);
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

        tensileProblem.setComputeInputType(
            roc2TensileComputeInputType(a_type, b_type, prob.compute_type));
        tensileProblem.setAlphaType(compute_type);
        tensileProblem.setBetaType(compute_type);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(
            TensileLite::DataTypeInfo::Get(compute_type).elementSize
            > TensileLite::DataTypeInfo::Get(a_type).elementSize);

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setGroupedGemm(prob.grouped_gemm);
        if(prob.grouped_gemm)
            tensileProblem.setUseDeviceUserArguments(true);
        else
            tensileProblem.setUseDeviceUserArguments(false);

        // alpha and beta are stored by value in TensileLite::TypedContractionInputs
        // alpha and beta are copied from host to TensileLite::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        double alphaRestriction = 0;
        if(prob.k)
            alphaRestriction = alpha;
        tensileProblem.setAlphaRestriction(TensileLite::toScalarValueEnum(alphaRestriction));

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
        tensileProblem.setBias(biasType, biasSize, 0, prob.gradient, biasSrc);
        tensileProblem.setParams().setBiasEnum(
            tensileUseBias(prob.epilogue) ? biasType : TensileLite::DataType::None);

        tensileProblem.setUseScaleAB(
            (prob.scaleA == nullptr && prob.scaleB == nullptr)
                ? ""
                : ((prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Vector)
                       ? "Vector"
                       : "Scalar"));
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

        // set AmaxD
        tensileProblem.setOutputAmaxD(prob.amaxD != nullptr);
        tensileProblem.setAmaxD(compute_type, true);

        if(prob.compute_type == rocblaslt_compute_f32_fast_xf32)
            tensileProblem.setF32XdlMathOp(TensileLite::DataType::XFloat32);

        tensileProblem.setSwizzleTensorA(prob.swizzleA);
        tensileProblem.setSwizzleTensorB(prob.swizzleB);

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

        double alpha = 0, beta = 0;
        assignAlphaBeta(compute_type, prob.alpha, prob.beta, &alpha, &beta);

        // fallback to f32 for f16 compute type after alpha/beta assignment
        if(prob.compute_type == rocblaslt_compute_f16)
        {
            compute_type = roc2TensileType(prob.compute_type);
        }

        tensileProblem.updateProblem(freeIndex, batchIndex, boundIndex, beta, prob.workspaceSize);

        tensileProblem.setComputeInputType(
            roc2TensileComputeInputType(a_type, b_type, prob.compute_type));
        tensileProblem.setAlphaType(compute_type);
        tensileProblem.setBetaType(compute_type);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(
            TensileLite::DataTypeInfo::Get(compute_type).elementSize
            > TensileLite::DataTypeInfo::Get(a_type).elementSize);

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);
        tensileProblem.setGroupedGemm(prob.grouped_gemm);
        if(prob.grouped_gemm)
            tensileProblem.setUseDeviceUserArguments(true);
        else
            tensileProblem.setUseDeviceUserArguments(false);

        // alpha and beta are stored by value in TensileLite::TypedContractionInputs
        // alpha and beta are copied from host to TensileLite::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        double alphaRestriction = 0;
        if(prob.k)
            alphaRestriction = alpha;
        tensileProblem.setAlphaRestriction(TensileLite::toScalarValueEnum(alphaRestriction));

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
        tensileProblem.setBias(biasType, biasSize, 0, prob.gradient, biasSrc);
        tensileProblem.setParams().setBiasEnum(
            tensileUseBias(prob.epilogue) ? biasType : TensileLite::DataType::None);

        tensileProblem.setUseScaleAB(
            (prob.scaleA == nullptr && prob.scaleB == nullptr)
                ? ""
                : ((prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Vector)
                       ? "Vector"
                       : "Scalar"));
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
            tensileProblem.setF32XdlMathOp(TensileLite::DataType::XFloat32);

        tensileProblem.setSwizzleTensorA(prob.swizzleA);
        tensileProblem.setSwizzleTensorB(prob.swizzleB);
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

        // Set the GSU workspace
        inputs.ws = prob.workspace;

        inputs.Synchronizer = prob.Synchronizer;

        // set bias vector
        if(is_bias_enabled(prob.epilogue))
            inputs.bias = reinterpret_cast<const void*>(prob.bias);
        else
            inputs.bias = nullptr;
        inputs.scaleA        = reinterpret_cast<const void*>(prob.scaleA);
        inputs.scaleB        = reinterpret_cast<const void*>(prob.scaleB);
        inputs.scaleC        = reinterpret_cast<const void*>(prob.scaleC);
        inputs.scaleD        = reinterpret_cast<const void*>(prob.scaleD);
        inputs.scaleAlphaVec = reinterpret_cast<const void*>(prob.scaleAlphaVec);
        inputs.amaxD         = reinterpret_cast<void*>(prob.amaxD);

        static const std::map<TensileLite::DataType, TensileLite::ConstantVariant> argument_vals = {
            {TensileLite::DataType::Float, 0.0f},
            {TensileLite::DataType::XFloat32, 0.0f},
            {TensileLite::DataType::Half, (hipblasLtHalf)0.0},
            {TensileLite::DataType::Int32, (int32_t)0},
            {TensileLite::DataType::Double, (double)0.0},
        };

        if(argument_vals.find(compute_type) == argument_vals.end())
        {
            log_error(__func__, "Unsupported compute type");
            throw std::runtime_error("[GetTensileInputs] unsupported compute type.");
        }

        // push 2 activation arguments
        std::visit(
            [&inputs, &prob](auto val) {
                inputs.activationArgs.push_back(val);
                inputs.activationArgs.push_back(val);
                if(prob.k)
                    inputs.alpha = *(decltype(val)*)(prob.alpha);
                else
                    inputs.alpha = val;
                inputs.beta = *(decltype(val)*)(prob.beta);
            },
            argument_vals.at(compute_type));

        // convert alpha and beta to float if compute type is half
        if(prob.compute_type == rocblaslt_compute_f16)
        {
            inputs.activationArgs = {0.0f, 0.0f};
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
        else if(deviceString.find("gfx1200") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1200;
        }
        else if(deviceString.find("gfx1201") != std::string::npos)
        {
            return TensileLite::LazyLoadingInit::gfx1201;
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
#else
        std::shared_ptr<hipDeviceProp_t> m_deviceProp;
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
#else
        auto&                            get_device_property() const
        {
            return m_deviceProp;
        }
#endif
        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*******************************************************
   * Testpath() tests that a path exists and is readable *
   *******************************************************/
        static bool TestPath(const std::string& path)
        {
            return rocblaslt_internal_test_path(path);
        }

        /*********************************************************************
   * Initialize adapter and library according to environment variables *
   * and default paths based on librocblaslt.so location and GPU         *
   *********************************************************************/
        void initialize(TensileLite::hip::SolutionAdapter& adapter, int32_t deviceId)
        {
            std::string path;
#ifndef WIN32
            path.reserve(PATH_MAX);
#endif

            // The name of the current GPU platform
            std::string processor = rocblaslt_internal_get_arch_name();

            const char* env = getenv("HIPBLASLT_TENSILE_LIBPATH");
            if(env)
            {
                std::cout << "rocblaslt info: Using HIPBLASLT_TENSILE_LIBPATH=" << env
                          << std::endl;
                path = env;
            }
            else
            {
                path = HIPBLASLT_LIB_PATH;

                // Find the location of librocblaslt.so
                // Fall back on hard-coded path if static library or not found

#ifndef HIPBLASLT_STATIC_LIB
                auto hipblaslt_so_path = getHipblasltSoPath();

                if(hipblaslt_so_path.size())
                    path = std::string{dirname(&hipblaslt_so_path[0])};
#endif // ifndef HIPBLASLT_STATIC_LIB

                // Find the location of the libraries
                if(TestPath(path + "/../Tensile/library"))
                    path += "/../Tensile/library";
                else if(TestPath(path + "library"))
                    path += "/library";
                else
                    path += "/hipblaslt/library";

                if(TestPath(path + "/" + processor))
                    path += "/" + processor;

                std::cout << "rocblaslt info: HIPBLASLT_TENSILE_LIBPATH not set: Using " << path 
                          << std::endl;
            }

            // only load modules for the current architecture
            auto dir = path + "/*" + processor + "*co";
#if ROCBLASLT_TENSILE_LAZY_LOAD == 0
            bool no_match = false;
#ifdef WIN32
            std::replace(dir.begin(), dir.end(), '/', '\\');
            WIN32_FIND_DATAA finddata;
            HANDLE           hfine = FindFirstFileA(dir.c_str(), &finddata);
            if(hfine != INVALID_HANDLE_VALUE)
            {
                do
                {
                    std::string codeObjectFile = path + "\\" + finddata.cFileName;
                    static_cast<void>(adapter.loadCodeObjectFile(codeObjectFile.c_str()));
                } while(FindNextFileA(hfine, &finddata));
            }
            else
            {
                no_match = true;
            }
            FindClose(hfine);
#else
            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    static_cast<void>(adapter.loadCodeObjectFile(glob_result.gl_pathv[i]));
            }
            else if(g == GLOB_NOMATCH)
            {
                no_match = true;
            }
            else
            {
#if 0
                // clang-format off
                static std::ostream& once = std::cerr
                                    << "\nrocblaslt warning: glob(\"" << dir << "\", ...) returned "
                                    << (g == GLOB_ABORTED ? "GLOB_ABORTED"
                                                          : g == GLOB_NOSPACE ? "GLOB_NOSPACE"
                                                                              : "an unknown error")
                                    << "." << std::endl;
                // clang-format on
#endif
            }
            globfree(&glob_result);
#endif
            if(no_match)
            {
                // static rocblaslt_internal_ostream& once
                //    = rocblaslt_cerr
                std::cerr << "\nrocblaslt warning: No paths matched " << dir
                          << ". Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly."
                          << std::endl;
            }
#endif
            // We initialize a local static variable with a lambda function call to
            // avoid race conditions when multiple threads with different device IDs try
            // to initialize library. This ensures that only one thread initializes
            // library, and other threads trying to initialize library wait for it to
            // complete.
            static int once = [&] {
                // Determine library path
                std::string tensileLibPath;
#if ROCBLASLT_TENSILE_LAZY_LOAD
#ifdef TENSILE_YAML
                tensileLibPath = path + "/TensileLibrary_lazy_" + processor + ".yaml";
#else
                tensileLibPath = path + "/TensileLibrary_lazy_" + processor + ".dat";
#endif
#else
#ifdef TENSILE_YAML
                tensileLibPath = path + "/TensileLibrary_" + processor + ".yaml";
#else
                tensileLibPath = path + "/TensileLibrary_" + processor + ".dat";
#endif
#endif
                if(!TestPath(tensileLibPath))
                {
                    std::cerr << "\nrocblaslt error: Cannot read " << tensileLibPath << ": "
                              << strerror(errno) << std::endl;
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
                    }
                }

                // Load library
                auto lib = TensileLite::LoadLibraryFilePreload<TensileLite::ContractionProblemGemm>(
                    tensileLibPath, std::vector<TensileLite::LazyLoadingInit>{});
#else
                // Get device prop
                hipDeviceProp_t prop;
                HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));
                m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);

                // Load library
                auto lib = TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(
                    tensileLibPath);
#endif
                if(!lib)
                    std::cerr << "\nrocblaslt error: Could not load " << tensileLibPath
                              << std::endl;
                else
                {
                    using MSL
                        = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;
                    m_library        = std::dynamic_pointer_cast<MSL>(lib);
                    m_tensileLibPath = tensileLibPath;
                }
                return 0;
            }();

            static_cast<void>(adapter.initializeLazyLoading(processor, path));

            if(!m_library && once != 0)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile library" << std::endl;
                // rocblaslt_abort();
            }
        }

#if ROCBLASLT_TENSILE_LAZY_LOAD
        // A workaround for getSolutionsFromIndex and isSolutionSupported with lazy_lib_load.
        // preload() shouldn't be called more than once.
        void preload()
        {
            auto lib = TensileLite::LoadLibraryFilePreload<TensileLite::ContractionProblemGemm>(
                m_tensileLibPath,
                std::vector<TensileLite::LazyLoadingInit>{m_deviceSet.begin(), m_deviceSet.end()});
            using MSL = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;
            m_library = std::dynamic_pointer_cast<MSL>(lib);
        }
#endif
    };

    // Return the library and adapter for the current HIP device
    TensileLite::hip::SolutionAdapter* get_library_and_adapter(
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>*
            library
        = nullptr,
        std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr,
        int                               device     = -1
#if ROCBLASLT_TENSILE_LAZY_LOAD
        ,
        bool isPreload = false
#endif
    )
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

#if ROCBLASLT_TENSILE_LAZY_LOAD
        // A workaround for getSolutionsFromIndex and isSolutionSupported when lazy_lib_load is on.
        // preload() shouldn't be called more than once.
        if(isPreload)
            static int once = [&] {
                host.preload();
                *library = host.get_library();
                return 0;
            }();
#endif
        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.get_library();
        if(deviceProp)
#if ROCBLASLT_TENSILE_LAZY_LOAD
            *deviceProp = host.get_device_property(rocblaslt_internal_get_arch_name());
#else
            *deviceProp = host.get_device_property();
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

TensileLite::ProblemOverride
    RocblasltContractionProblem2ProblemOverride(const RocblasltContractionProblem& problem)
{
    return TensileLite::ProblemOverride(problem.trans_a == HIPBLAS_OP_N ? false : true,
                                        problem.trans_b == HIPBLAS_OP_N ? false : true,
                                        hipDataType_to_tensile_type(problem.a_type),
                                        hipDataType_to_tensile_type(problem.b_type),
                                        roc2TensileType(problem.compute_type),
                                        hipDataType_to_tensile_type(problem.c_type),
                                        problem.m,
                                        problem.n,
                                        problem.k,
                                        problem.batch_count);
}

TensileLite::ProblemOverride TensileDataGemm2ProblemOverride(std::shared_ptr<void> gemmData)
{
    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);

    return TensileLite::ProblemOverride(data->problem.transA(),
                                        data->problem.transB(),
                                        data->problem.a().dataType(),
                                        data->problem.b().dataType(),
                                        data->problem.computeInputType(),
                                        data->problem.c().dataType(),
                                        data->problem.freeSizeA(0),
                                        data->problem.freeSizeB(0),
                                        data->problem.boundSize(0),
                                        data->problem.batchSize(0));
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

#ifdef USE_ROCROLLER
bool useRocRoller(rocblaslt_handle handle, const RocblasltContractionProblem& prob)
{
    return handle->useRocRoller == 1
           || (handle->useRocRoller == -1
               && (prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block
                   || prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block));
}
#endif

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
#ifdef USE_ROCROLLER
        if(useRocRoller(handle, prob))
            return runRocRollerContractionProblem(handle, algo, prob);
#endif
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                               library;
        std::shared_ptr<hipDeviceProp_t>       deviceProp;
        std::shared_ptr<TensileLite::Hardware> hardware;

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        hardware = TensileLite::hip::GetDevice(*deviceProp);

        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        rocblaslt_matmul_heuristic_result heuristicResult;
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
        data->algoIndex    = *solutionIndex;
        data->inputs       = GetTensileInputs(prob);

        if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
           || rocblaslt::Debug::Instance().printLogAsMarker())
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
        if(gemmData)
        {
            std::shared_ptr<TensileDataGemm> data
                = std::static_pointer_cast<TensileDataGemm>(gemmData);
            updateTensileProblem(problem, data->problem);
            data->inputs         = GetTensileInputs(problem);
            data->enableEpilogue = problem.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;
        }
        else
        {
            TensileDataGemm data;
            data.problem        = ConstructTensileProblem(problem);
            data.inputs         = GetTensileInputs(problem);
            data.enableEpilogue = problem.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

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

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        hardware = TensileLite::hip::GetDevice(*deviceProp);

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

            data->inputs.ws = workspace;

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

            for(int i = 0; i < data->inputs.grouped.size(); i++)
            {
                data->inputs.grouped[i].ws = workspace;
            }
            data->inputs.ws = workspace;

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

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

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
               || rocblaslt::Debug::Instance().printLogAsMarker())
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
                                              flush,
                                              rotatingBufferSize,
                                              coldIterations,
                                              hotIterations,
                                              true);
            }
            status = hip2RocStatus(adapter->launchKernels(data->kernels, stream, start, stop));
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
               || rocblaslt::Debug::Instance().printLogAsMarker())
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
            //TODO: add profile logging for grouped gemm
            /*if(get_logger_layer_mode() & rocblaslt_layer_mode_log_profile)
            {
                logProfileFromTensileDataGemm(data->problem, data->inputs, true);
            }*/
            status = hip2RocStatus(adapter->launchKernels(data->kernels, stream, start, stop));
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

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

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
            if(problem.activationComputeType() == TensileLite::DataType::Float)
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

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

        if(!library)
        {
            return rocblaslt_status_invalid_pointer;
        }

        if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::shared_ptr<TensileDataGroupedGemm> data
                = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
            for(auto& it : data->kernels)
            {
                uint8_t* arg      = it.args.rawdata();
                auto     solution = library->getSolutionByIndex(*hardware, data->algoIndex);
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

        auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

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

    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    if(!library)
    {
        return {};
    }

    hardware = TensileLite::hip::GetDevice(*deviceProp);

    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(prob, data->problem);

    bool enableEpilogue = prob.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

    auto solutions
        = getSolutions(prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if(solutions.size() == 0 && prob.compute_type == rocblaslt_compute_f32_fast_xf32)
    {
        log_api(__func__, "no solutions found, try to fallback");
        data->problem.setF32XdlMathOp(TensileLite::DataType::Float);
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
#ifdef USE_ROCROLLER
    if(useRocRoller(handle, prob))
        return getRocRollerBestSolutions(
            handle, prob, requestedAlgoCount, heuristicResultsArray, returnAlgoCount);
#endif
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    hardware = TensileLite::hip::GetDevice(*deviceProp);

    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(prob, data->problem);

    bool enableEpilogue = prob.epilogue == ROCBLASLT_EPILOGUE_DEFAULT ? false : true;

    auto solutions
        = getSolutions(prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);

    // when there is no solution for xfloat32, fallback comput_type to fp32
    if(solutions.size() == 0 && prob.compute_type == rocblaslt_compute_f32_fast_xf32)
    {
        log_api(__func__, "no xf32 solutions found, try to fallback fp32");
        data->problem.setF32XdlMathOp(TensileLite::DataType::Float);
        solutions = getSolutions(
            prob, library, hardware, data->problem, enableEpilogue, requestedAlgoCount);
    }

    memset(
        heuristicResultsArray, 0, sizeof(rocblaslt_matmul_heuristic_result) * requestedAlgoCount);
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
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    std::string deviceFullString(deviceProp->gcnArchName);
    std::string deviceString = deviceFullString.substr(0, deviceFullString.find(":"));

    hardware = TensileLite::hip::GetDevice(*deviceProp);

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
    if(solutions.size() == 0 && prob.f32XdlMathOp() == TensileLite::DataType::XFloat32)
    {
        prob.setF32XdlMathOp(TensileLite::DataType::Float);
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

    int i = 0;
    for(auto solution : solutions)
    {
        //workaround: findAllSolutions should get all solutions without duplications
        bool duplicated_sol = false;
        for(int j = 0; j < i; j++)
            if(*(int*)(heuristicResults[j].algo.data) == solution->index)
                duplicated_sol = true;
        if(duplicated_sol)
            continue;
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
        i++;
    }
    heuristicResults.resize(i);
    log_api(__func__, "Final hardware solutions: ", heuristicResults.size());

    return rocblaslt_status_success;
}

rocblaslt_status getAllSolutions(RocblasltContractionProblem&                    prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes)
{
#ifdef USE_ROCROLLER
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

rocblaslt_status
    getSolutionsFromIndex(rocblaslt_handle                                handle,
                          std::vector<int>&                               solutionIndex,
                          std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                          size_t                                          maxWorkSpaceBytes)
{
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

#if ROCBLASLT_TENSILE_LAZY_LOAD
    // isPreload = true is to load placeholder libraries except code objects
    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device, true);
#else
    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
#endif

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    hardware = TensileLite::hip::GetDevice(*deviceProp);

    int  lastSolutionIndex = library->solutions.rbegin()->first;
    bool isOutOfBound      = true;
    int  i                 = 0;
    for(auto index : solutionIndex)
    {
#ifdef USE_ROCROLLER
        if(index < 0)
        {
            isOutOfBound = false;
            getRocRollerSolutionsFromIndex(handle, index, heuristicResults, maxWorkSpaceBytes);
            continue;
        }

#endif
        isOutOfBound  = isOutOfBound && (index > lastSolutionIndex);
        auto solution = library->getSolutionByIndex(*hardware, index);
        if(!solution)
            continue;
        rocblaslt_matmul_heuristic_result result;
        memset(&result, 0, sizeof(rocblaslt_matmul_heuristic_result));
        memset(result.algo.data, 0, sizeof(result.algo.data));
        int* solutionIndex              = (int*)(result.algo.data);
        *solutionIndex                  = solution->index;
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
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

#if ROCBLASLT_TENSILE_LAZY_LOAD
    // isPreload = true is a workaround for lazy_lib_load
    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device, true);
#else
    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
#endif

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    hardware              = TensileLite::hip::GetDevice(*deviceProp);
    *workspaceSizeInBytes = 0;

    int* solutionIndex = (int*)algo->data;
    // don't overwrite data->algoIndex = *solutionIndex; here
    if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGemm>::value)
    {
        auto solution = library->getSolutionByIndex(tensile_prob, *hardware, *solutionIndex);

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
        else
        {
            *workspaceSizeInBytes = solution->requiredWorkspaceSize(tensile_prob, *hardware);
        }
    }
    else if constexpr(std::is_same<MyProblem, TensileLite::ContractionProblemGroupedGemm>::value)
    {
        auto solution
            = library->getSolutionByIndex(tensile_prob.gemms[0], *hardware, *solutionIndex);

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
        auto problemWs = solution->requiredWorkspaceSizeGroupedGemm(tensile_prob.gemms, *hardware);
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            tensile_prob.gemms[i].setWorkspaceSize(algo->max_workspace_bytes);
            tensile_prob.gemms[i].setWorkspaceSizeGroupedGemm(problemWs);
            tensile_prob.gemms[i].setGroupedGemmCount(tensile_prob.gemms.size());
        }
        for(int i = 0; i < tensile_prob.gemms.size(); i++)
        {
            if(!((*solution->hardwarePredicate)(*hardware)
                 && (*solution->problemPredicate)(tensile_prob.gemms[i])))
            {
                if(get_logger_layer_mode() & rocblaslt_layer_mode_log_info)
                {
                    std::ostringstream msg;
                    msg << "Match "
                        << "[" << i << "]: " << solution->description();
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
#ifdef USE_ROCROLLER
    if(useRocRoller(handle, prob))
        return isRocRollerSolutionSupported(handle, prob, algo, workspaceSizeInBytes);
#endif
    std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
    updateTensileProblem(prob, data->problem);
    rocblaslt::RocTuningV2* tuning = nullptr;
    return isSolutionSupported(handle, data->problem, prob, algo, tuning, workspaceSizeInBytes);
}

template <typename T>
void setRestrictions(TensileLite::ContractionProblemGemm& tensile_prob,
                     const T*                             alpha,
                     const T*                             beta)
{
    tensile_prob.setAlphaRestriction(TensileLite::toScalarValueEnum(*alpha));
    tensile_prob.setBetaRestriction(TensileLite::toScalarValueEnum(*beta));
}

template <typename Tuning>
rocblaslt_status isSolutionSupported(rocblaslt_handle              handle,
                                     const rocblaslt::RocGemmType& gemmType,
                                     std::shared_ptr<void>         gemmData,
                                     rocblaslt_matmul_algo&        algo,
                                     const Tuning*                 tuning,
                                     size_t&                       workspaceSizeInBytes)
{
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
    {
        std::shared_ptr<TensileDataGemm> data = std::static_pointer_cast<TensileDataGemm>(gemmData);
        if(data->problem.computeType() == TensileLite::DataType::Float)
        {
            setRestrictions<float>(data->problem,
                                   std::get_if<float>(&data->inputs.alpha),
                                   std::get_if<float>(&data->inputs.beta));
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
        return isSolutionSupported(
            handle, data->problem, data->inputs, &algo, tuning, &workspaceSizeInBytes);
    }
    else if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        std::shared_ptr<TensileDataGroupedGemm> data
            = std::static_pointer_cast<TensileDataGroupedGemm>(gemmData);
        if(data->problem.gemms[0].computeType() == TensileLite::DataType::Float)
        {
            for(int i = 0; i < data->problem.gemms.size(); i++)
            {
                auto& tensile_prob = data->problem.gemms[i];
                setRestrictions<float>(tensile_prob,
                                       std::get_if<float>(&data->inputs.grouped[i].alpha),
                                       std::get_if<float>(&data->inputs.grouped[i].beta));
            }
        }
        else
        {
            return rocblaslt_status_not_implemented;
        }
        return isSolutionSupported(
            handle, data->problem, data->inputs, &algo, tuning, &workspaceSizeInBytes);
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
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                           library;
    std::shared_ptr<hipDeviceProp_t>       deviceProp;
    std::shared_ptr<TensileLite::Hardware> hardware;

    // auto &adapter =
    static_cast<void>(get_library_and_adapter(&library, &deviceProp, handle->device));

    if(!library)
    {
        return rocblaslt_status_invalid_pointer;
    }

    hardware = TensileLite::hip::GetDevice(*deviceProp);

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
        if(solutions.size() == 0 && data->problem.f32XdlMathOp() == TensileLite::DataType::XFloat32)
        {
            data->problem.setF32XdlMathOp(TensileLite::DataType::Float);
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

    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

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
    std::shared_ptr<hipDeviceProp_t> deviceProp;

    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);

    if(!library)
    {
        return std::string();
    }

    int gsu           = 0;
    int wgm           = 0;
    int solutionIndex = -1;

    std::shared_ptr<TensileLite::Hardware> hardware;

    hardware = TensileLite::hip::GetDevice(*deviceProp);

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
    auto        solution       = library->getSolutionByIndex(*hardware, solutionIndex);
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
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                     library;
    std::shared_ptr<hipDeviceProp_t> deviceProp;

    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
    std::shared_ptr<TensileLite::Hardware> hardware;
    hardware = TensileLite::hip::GetDevice(*deviceProp);

    if(!library)
    {
        return std::string();
    }

    int* solutionIndex = (int*)algo.data;
    auto solution      = library->getSolutionByIndex(*hardware, *solutionIndex);
    return solution->kernelName;
}

std::string getSolutionNameFromAlgoIndex(rocblaslt_handle handle, const rocblaslt_matmul_algo& algo)
{
    std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                     library;
    std::shared_ptr<hipDeviceProp_t> deviceProp;

    auto adapter = get_library_and_adapter(&library, &deviceProp, handle->device);
    std::shared_ptr<TensileLite::Hardware> hardware;
    hardware = TensileLite::hip::GetDevice(*deviceProp);

    if(!library)
    {
        return std::string();
    }

    int* solutionIndex = (int*)algo.data;
    auto solution      = library->getSolutionByIndex(*hardware, *solutionIndex);
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
CREATECOMPATIBILITYFUNCTION(rocblaslt::RocTuning)
CREATECOMPATIBILITYFUNCTION(rocblaslt::RocTuningV2)
