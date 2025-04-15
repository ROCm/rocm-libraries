/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_ROCBLAS_REFERENCE_HPP
#define ROCWMMA_ROCBLAS_REFERENCE_HPP

#define ROCM_USE_FLOAT16

#define CHECK_ROCBLAS_ERROR(expression)                              \
    if(auto status = (expression); status != rocblas_status_success) \
    {                                                                \
        fprintf(stderr,                                              \
                "rocBLAS error: '%s'(%d) at %s:%d\n",                \
                rocblas_status_to_string(status),                    \
                status,                                              \
                __FILE__,                                            \
                __LINE__);                                           \
        exit(EXIT_FAILURE);                                          \
    }

// BETA_FEATURES_API needs to be defined to access the beta features of rocBLAS which includes float8/bfloat8 support.
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

#if(ROCBLAS_VERSION_MAJOR >= 3) || ((ROCBLAS_VERSION_MAJOR >= 2) && (ROCBLAS_VERSION_MINOR > 45))
#define ROCBLAS_DATA_TYPE_INVALID
#endif

//#if(((ROCBLAS_VERSION_MAJOR == 3) && (ROCBLAS_VERSION_MINOR >= 1)) || (ROCBLAS_VERSION_MAJOR > 3))
//#define ROCBLAS_DATA_TYPE_FLOAT8
//#endif

#include "common.hpp"
#include <rocwmma/internal/types.hpp>

namespace rocwmma
{

    constexpr const char* rocblas_datatype2string(rocblas_datatype type)
    {
        switch(type)
        {
        case rocblas_datatype_f16_r:
            return "f16_r";
        case rocblas_datatype_f32_r:
            return "f32_r";
        case rocblas_datatype_f64_r:
            return "f64_r";
        case rocblas_datatype_f16_c:
            return "f16_c";
        case rocblas_datatype_f32_c:
            return "f32_c";
        case rocblas_datatype_f64_c:
            return "f64_c";
        case rocblas_datatype_i8_r:
            return "i8_r";
        case rocblas_datatype_u8_r:
            return "u8_r";
        case rocblas_datatype_i32_r:
            return "i32_r";
        case rocblas_datatype_u32_r:
            return "u32_r";
        case rocblas_datatype_i8_c:
            return "i8_c";
        case rocblas_datatype_u8_c:
            return "u8_c";
        case rocblas_datatype_i32_c:
            return "i32_c";
        case rocblas_datatype_u32_c:
            return "u32_c";
        case rocblas_datatype_bf16_r:
            return "bf16_r";
        case rocblas_datatype_bf16_c:
            return "bf16_c";
#if defined(ROCBLAS_DATA_TYPE_FLOAT8)
        case rocblas_datatype_f8_r:
            return "f8_r";
        case rocblas_datatype_bf8_r:
            return "bf8_r";
#endif
#if defined(ROCBLAS_DATA_TYPE_INVALID)
        case rocblas_datatype_invalid:;
#endif // ROCBLAS_DATA_TYPE_INVALID
        }
        return "invalid";
    }

    template <typename DataT>
    struct rocblas_types
#if defined(ROCBLAS_DATA_TYPE_INVALID)
    {
        using DataType = DataT;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_invalid;
        }
    }
#endif //ROCBLAS_DATA_TYPE_INVALID
    ;

    template <>
    struct rocblas_types<int8_t>
    {
        using DataType = int8_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_i8_r;
        }
    };

    template <>
    struct rocblas_types<uint8_t>
    {
        using DataType = uint8_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_u8_r;
        }
    };

    template <>
    struct rocblas_types<int32_t>
    {
        using DataType = rocblas_int;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_i32_r;
        }
    };

    template <>
    struct rocblas_types<uint32_t>
    {
        using DataType = uint32_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_u32_r;
        }
    };

    template <>
    struct rocblas_types<float16_t>
    {
        using DataType = rocblas_half;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_f16_r;
        }
    };

#if !ROCWMMA_TESTS_NO_HALF
    template <>
    struct rocblas_types<hfloat16_t> : public rocblas_types<float16_t>
    {
    };
#endif // !ROCWMMA_TESTS_NO_HALF

    template <>
    struct rocblas_types<bfloat16_t>
    {
        using DataType = rocblas_bfloat16;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_bf16_r;
        }
    };

    template <>
    struct rocblas_types<float32_t>
    {
        using DataType = float32_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_f32_r;
        }
    };

    template <>
    struct rocblas_types<xfloat32_t>
    {
        using DataType = xfloat32_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_f32_r;
        }
    };

    template <>
    struct rocblas_types<float64_t>
    {
        using DataType = float64_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_f64_r;
        }
    };

#if defined(ROCBLAS_DATA_TYPE_FLOAT8)
    template <>
    struct rocblas_types<bfloat8_t>
    {
        using DataType = bfloat8_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_bf8_r;
        }
    };

    template <>
    struct rocblas_types<float8_t>
    {
        using DataType = float8_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_f8_r;
        }
    };

    template <>
    struct rocblas_types<bfloat8_fnuz_t>
    {
        using DataType = bfloat8_fnuz_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_bf8_r;
        }
    };

    template <>
    struct rocblas_types<float8_fnuz_t>
    {
        using DataType = float8_fnuz_t;
        constexpr static inline rocblas_datatype type()
        {
            return rocblas_datatype_f8_r;
        }
    };
#endif

    template <typename DataLayoutT>
    struct rocblas_layout;

    template <>
    struct rocblas_layout<row_major>
    {
        using Layout = row_major;
        constexpr static inline rocblas_operation operation()
        {
            return rocblas_operation_transpose;
        }
    };

    template <>
    struct rocblas_layout<col_major>
    {
        using Layout = col_major;
        constexpr static inline rocblas_operation operation()
        {
            return rocblas_operation_none;
        }
    };

    //! @brief Dispatcher for rocBLAS.
    //! Need to call different interface for f8 validation
    auto dispatch_rocBLAS(rocblas_handle    handle,
                          rocblas_operation opA,
                          rocblas_operation opB,
                          rocblas_int       m,
                          rocblas_int       n,
                          rocblas_int       k,
                          const void*       alpha,
                          const void*       a,
                          rocblas_datatype  a_type,
                          rocblas_int       lda,
                          const void*       b,
                          rocblas_datatype  b_type,
                          rocblas_int       ldb,
                          const void*       beta,
                          const void*       c,
                          rocblas_datatype  c_type,
                          rocblas_int       ldc,
                          void*             d,
                          rocblas_datatype  d_type,
                          rocblas_int       ldd,
                          rocblas_datatype  compute_type,
                          rocblas_gemm_algo algo,
                          int32_t           solution_index,
                          uint32_t          flags)
    {
#if defined(ROCBLAS_DATA_TYPE_FLOAT8)
        if(a_type == rocblas_datatype_f8_r || b_type == rocblas_datatype_f8_r
           || a_type == rocblas_datatype_bf8_r || b_type == rocblas_datatype_bf8_r)
        {
            if(compute_type != rocblas_datatype_f32_r)
            {
                std::cerr << "float8_t and bfloat8_t input types must have f32 compute type with "
                             "this version of rocBLAS"
                          << std::endl;
                return rocblas_status_not_implemented;
            }

            // rocblas_gemm_ex3 needs a rocblas_computetype
            rocblas_computetype compute_type_f32 = rocblas_compute_type_f32;
            if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_f8_r)
            {
                compute_type_f32 = rocblas_compute_type_f8_f8_f32;
            }
            else if(a_type == rocblas_datatype_f8_r && b_type == rocblas_datatype_bf8_r)
            {
                compute_type_f32 = rocblas_compute_type_f8_bf8_f32;
            }
            else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_f8_r)
            {
                compute_type_f32 = rocblas_compute_type_bf8_f8_f32;
            }
            else if(a_type == rocblas_datatype_bf8_r && b_type == rocblas_datatype_bf8_r)
            {
                compute_type_f32 = rocblas_compute_type_bf8_bf8_f32;
            }

            return rocblas_gemm_ex3(handle,
                                    opA,
                                    opB,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    a,
                                    a_type,
                                    lda,
                                    b,
                                    b_type,
                                    ldb,
                                    beta,
                                    c,
                                    c_type,
                                    ldc,
                                    d,
                                    d_type,
                                    ldd,
                                    compute_type_f32,
                                    algo,
                                    solution_index,
                                    flags);
        }
#endif

        return rocblas_gemm_ex(handle,
                               opA,
                               opB,
                               m,
                               n,
                               k,
                               alpha,
                               a,
                               a_type,
                               lda,
                               b,
                               b_type,
                               ldb,
                               beta,
                               c,
                               c_type,
                               ldc,
                               d,
                               d_type,
                               ldd,
                               compute_type,
                               algo,
                               solution_index,
                               flags);
    }
    /*
    * Rocblas notes:
    * Layouts C and D are always assumed as col_major
    * Non-transpose (N) is col-major
    * Transpose (T) is row-major
    */
    template <typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB>
    void gemm_rocBLAS(uint32_t       m,
                      uint32_t       n,
                      uint32_t       k,
                      InputT const*  ha,
                      InputT const*  hb,
                      OutputT const* hc,
                      OutputT*       hd,
                      ComputeT       alpha,
                      ComputeT       beta)
    {
        rocblas_datatype a_type       = rocblas_types<InputT>::type();
        rocblas_datatype b_type       = rocblas_types<InputT>::type();
        rocblas_datatype c_type       = rocblas_types<OutputT>::type();
        rocblas_datatype d_type       = rocblas_types<OutputT>::type();
        rocblas_datatype compute_type = rocblas_types<ComputeT>::type();

        using a_t = typename rocblas_types<InputT>::DataType;
        using b_t = typename rocblas_types<InputT>::DataType;
        using c_t = typename rocblas_types<OutputT>::DataType;
        using d_t = typename rocblas_types<OutputT>::DataType;

        size_t size_a = m * k;
        size_t size_b = k * n;
        size_t size_c = m * n;
        size_t size_d = m * n;

        rocblas_operation opA = rocblas_layout<LayoutA>::operation();
        rocblas_operation opB = rocblas_layout<LayoutB>::operation();

        rocblas_int lda = (opA == rocblas_operation_none ? m : k);
        rocblas_int ldb = (opB == rocblas_operation_none ? k : n);
        rocblas_int ldc = m;
        rocblas_int ldd = m;

        // allocate memory on device
        a_t* da;
        b_t* db;
        c_t* dc;
        d_t* dd;

        CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(a_t)));
        CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(b_t)));
        CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(c_t)));
        CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(d_t)));

        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(da, &ha[0], sizeof(a_t) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, &hb[0], sizeof(b_t) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, &hc[0], sizeof(c_t) * size_c, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dd, &hd[0], sizeof(d_t) * size_d, hipMemcpyHostToDevice));

        rocblas_handle handle;
        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

        auto     algo           = rocblas_gemm_algo_standard;
        int32_t  solution_index = 0;
        uint32_t flags          = 0;

        if((std::is_same<InputT, float8_t>::value)
           || (std::is_same<InputT, bfloat8_t>::value
               || (std::is_same<InputT, float8_fnuz_t>::value)
               || (std::is_same<InputT, bfloat8_fnuz_t>::value)))
        {
#if !defined(ROCBLAS_DATA_TYPE_FLOAT8)
            std::cerr
                << "float8_t and bfloat8_t types are not supported with this version of rocBLAS"
                << std::endl;
            return;
#endif
        }

        CHECK_ROCBLAS_ERROR(dispatch_rocBLAS(handle,
                                             opA,
                                             opB,
                                             m,
                                             n,
                                             k,
                                             &alpha,
                                             da,
                                             a_type,
                                             lda,
                                             db,
                                             b_type,
                                             ldb,
                                             &beta,
                                             dc,
                                             c_type,
                                             ldc,
                                             dd,
                                             d_type,
                                             ldd,
                                             compute_type,
                                             algo,
                                             solution_index,
                                             flags));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(&hd[0], dd, sizeof(d_t) * size_d, hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipFree(da));
        CHECK_HIP_ERROR(hipFree(db));
        CHECK_HIP_ERROR(hipFree(dc));
        CHECK_HIP_ERROR(hipFree(dd));
        CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    }

} // namespace rocwmma

#endif // ROCWMMA_ROCBLAS_REFERENCE_HPP
