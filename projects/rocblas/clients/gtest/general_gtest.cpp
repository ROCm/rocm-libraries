/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocblas_data.hpp"
#include "rocblas_matrix.hpp"
#include "rocblas_test.hpp"
#include "rocblas_vector.hpp"
#include "type_dispatch.hpp"
#include "unit.hpp"

#include "client_utility.hpp"

#include "../../library/src/include/check_numerics_matrix.hpp"
#include "../../library/src/include/check_numerics_vector.hpp"

namespace
{
    template <typename T>
    void expect_decimals_eq(T a, T b, int decimals)
    {
        EXPECT_EQ(std::round(a * pow(10, decimals)), std::round(b * pow(10, decimals)));
    }

    //
    // half floats

    template <typename T>
    void testing_half_operators(const Arguments& arg)
    {

        T c(0.5);
        T s(2.0);

        T result = -(c + c) * s;
        result /= s;
        EXPECT_EQ(result, T(-1.0));

        c      = T(0.5);
        s      = T(2.0);
        result = c * s + s / c;
        EXPECT_EQ((float)result, 5.0f);

        // unique harmonic convergence
        // search half-precision-arithmetic-fp16-versus-bfloat16 harmonic
        result = T(0);
        if(std::is_same_v<T, rocblas_half>)
        {
            for(int i = 1; i <= 513; i++)
                result += T(1.0) / T(i);
            expect_decimals_eq((float)result, 7.08594f, 5);
        }
        else if(std::is_same_v<T, rocblas_bfloat16>)
        {
            for(int i = 1; i <= 65; i++)
                result += T(1.0) / T(i);
            expect_decimals_eq((float)result, 5.0625f, 4);
        }
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct half_operators_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct half_operators_testing<
        T,
        std::enable_if_t<std::is_same_v<T, rocblas_half> || std::is_same_v<T, rocblas_bfloat16>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "half_operators"))
                testing_half_operators<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct half_operators : RocBLAS_Test<half_operators, half_operators_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "half_operators");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<half_operators> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            return std::move(name);
        }
    };

    TEST_P(half_operators, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<half_operators_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(half_operators);

    //
    // complex

    template <typename T>
    void testing_complex_operators(const Arguments& arg)
    {
        using R = real_t<T>;

        T c(0.5, 0.25);
        R s(2.0);

        T result = c * s;
        EXPECT_EQ(result, T(1.0, 0.5));

        result /= s;
        EXPECT_EQ(result, c);

        T val(1.0, -2.0);
        result = (s - val) / s;
        EXPECT_EQ(result, T(0.5, 1.0));

        result = T(20.0, -4.0) / T(3.0, 2.0);
        EXPECT_EQ(result, T(4.0, -4.0));

        result = 1.0 / T(1.0, 0.0);
        EXPECT_EQ(result, T(1.0, 0.0));
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct complex_operators_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct complex_operators_testing<
        T,
        std::enable_if_t<
            std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "complex_operators"))
                testing_complex_operators<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct complex_operators : RocBLAS_Test<complex_operators, complex_operators_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "complex_operators");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<complex_operators> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            return std::move(name);
        }
    };

    TEST_P(complex_operators, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<complex_operators_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(complex_operators);

    //
    // helper utilities

    template <typename T>
    void testing_helper_utilities(const Arguments& arg)
    {
        // just want to ensure 64 bit math

        uint32_t block  = 8;
        int64_t  stride = int64_t(1) << 33; // int32 overflow
        int64_t  offset = stride + 2;

        T        val(42);
        const T* valptr = &val - block * stride;
        T        result = load_scalar(valptr, block, stride);
        EXPECT_EQ(result, val);

        const T* dst = load_ptr_batch(&val, block, offset, stride);
        valptr       = &val + block * stride + offset;
        EXPECT_EQ(dst, valptr);

        dst    = load_ptr_batch(&val, block, stride);
        valptr = &val + block * stride;
        EXPECT_EQ(dst, valptr);
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct helper_utilities_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct helper_utilities_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex> || std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "helper_utilities"))
                testing_helper_utilities<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct helper_utilities : RocBLAS_Test<helper_utilities, helper_utilities_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "helper_utilities");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<helper_utilities> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            return std::move(name);
        }
    };

    TEST_P(helper_utilities, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<helper_utilities_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(helper_utilities);

    //
    // check numerics

    //
    //Testing a vector for NaN/zero/Inf/denormal values
    template <typename T>
    void testing_check_numerics_vector(const Arguments& arg)
    {
        int64_t        N           = arg.N;
        int64_t        inc_x       = arg.incx;
        rocblas_stride offset_x    = 0;
        rocblas_stride stride_x    = arg.stride_x;
        int64_t        batch_count = arg.batch_count;

        // for initialization we don't care about computation order
        size_t abs_inc_x = std::abs(inc_x);

        //Creating a rocBLAS handle
        rocblas_local_handle handle{arg};

        rocblas_status status          = rocblas_status_success;
        const char     function_name[] = "testing_check_numerics_vector";
        bool           is_input        = true;

        //Hard-code the enum `check_numerics` to `rocblas_check_numerics_mode_fail` which will return `rocblas_status_check_numerics_fail` if the vector contains a NaN/Inf/denormal value
        rocblas_check_numerics_mode check_numerics = rocblas_check_numerics_mode_fail;

        //Argument sanity check before allocating invalid memory
        if(N <= 0 || batch_count <= 0)
        {
            status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                     handle,
                                                                     N,
                                                                     (T*)nullptr,
                                                                     offset_x,
                                                                     inc_x,
                                                                     stride_x,
                                                                     1,
                                                                     check_numerics,
                                                                     is_input);
            EXPECT_EQ(status, rocblas_status_success);
            return;
        }

        //Allocating memory for the host vector
        host_vector<T> h_x(N, inc_x);

        // allocate memory on device
        device_vector<T> d_x(N, inc_x);
        CHECK_DEVICE_ALLOCATION(d_x.memcheck());

        //==============================================================================================
        // Initializing random values in the vector
        //==============================================================================================
        rocblas_init_vector(h_x, arg, rocblas_client_never_set_nan, true);

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_x.transfer_from(h_x));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 (T*)d_x,
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for zero in the vector
        //==============================================================================================
        rocblas_init_zero_range<T>((T*)h_x, 0, 1); // for inc_x==0 case we set first element

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_x.transfer_from(h_x));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 (T*)d_x,
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for Inf in the vector
        //==============================================================================================
        rocblas_init_vector(h_x, arg, rocblas_client_never_set_nan, true);
        rocblas_init_inf_range<T>((T*)h_x, 0, 1);

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_x.transfer_from(h_x));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 (T*)d_x,
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for NaN in the vector
        //==============================================================================================
        rocblas_init_vector(h_x, arg, rocblas_client_never_set_nan, true);
        rocblas_init_nan_range<T>((T*)h_x, 0, 1);

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_x.transfer_from(h_x));
        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 (T*)d_x,
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for denorm values in the vector
        //==============================================================================================
        rocblas_init_vector(h_x, arg, rocblas_client_never_set_nan, true);
        rocblas_init_denorm_range<T>((T*)h_x, 0, 1);

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_x.transfer_from(h_x));
        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 (T*)d_x,
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing random values in batched vectors
        //==============================================================================================
        //Allocate device and host batched vectors
        device_batch_vector<T> d_x_batch(N, inc_x, batch_count);
        host_batch_vector<T>   h_x_batch(N, inc_x, batch_count);

        //Initialize Data on CPU
        rocblas_init_vector(h_x_batch, arg, rocblas_client_never_set_nan, true);

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_x_batch.transfer_from(h_x_batch));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 d_x_batch.const_batch_ptr(),
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for zero in batched vectors
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_vector(h_x_batch, arg, rocblas_client_never_set_nan, true);
        for(size_t i = batch_count - 1; i < batch_count; i++)
            for(size_t j = N - 1; j < N; j++)
                h_x_batch[i][j * abs_inc_x] = T(rocblas_zero_rng());

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_x_batch.transfer_from(h_x_batch));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 d_x_batch.const_batch_ptr(),
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for Inf in batched vectors
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_vector(h_x_batch, arg, rocblas_client_never_set_nan, true);
        for(size_t i = batch_count - 1; i < batch_count; i++)
            for(size_t j = N - 1; j < N; j++)
                h_x_batch[i][j * abs_inc_x] = T(rocblas_inf_rng());

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_x_batch.transfer_from(h_x_batch));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 d_x_batch.const_batch_ptr(),
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for NaN in batched vectors
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_vector(h_x_batch, arg, rocblas_client_never_set_nan, true);
        for(size_t i = batch_count - 1; i < batch_count; i++)
            for(size_t j = N - 1; j < N; j++)
                h_x_batch[i][j * abs_inc_x] = T(rocblas_nan_rng());

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_x_batch.transfer_from(h_x_batch));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 d_x_batch.const_batch_ptr(),
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for denorm values in batched vectors
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_vector(h_x_batch, arg, rocblas_client_never_set_nan, true);
        for(size_t i = batch_count - 1; i < batch_count; i++)
            for(size_t j = N - 1; j < N; j++)
                h_x_batch[i][j * abs_inc_x] = T(rocblas_denorm_rng());

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_x_batch.transfer_from(h_x_batch));

        status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                 handle,
                                                                 N,
                                                                 d_x_batch.const_batch_ptr(),
                                                                 offset_x,
                                                                 inc_x,
                                                                 stride_x,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);

        EXPECT_EQ(status, rocblas_status_check_numerics_fail);
    };

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct check_numerics_vector_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct check_numerics_vector_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex> || std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "check_numerics_vector"))
                testing_check_numerics_vector<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct check_numerics_vector
        : RocBLAS_Test<check_numerics_vector, check_numerics_vector_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "check_numerics_vector");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<check_numerics_vector> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            name << '_' << arg.N << '_' << arg.incx;
            name << '_' << arg.batch_count;
            return std::move(name);
        }
    };

    TEST_P(check_numerics_vector, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<check_numerics_vector_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(check_numerics_vector);

    //Testing a matrix for NaN/zero/Inf/denormal values
    template <typename T>
    void testing_check_numerics_matrix(const Arguments& arg)
    {
        int64_t           M           = arg.M;
        int64_t           N           = arg.N;
        int64_t           lda         = arg.lda;
        rocblas_stride    offset_a    = 0;
        rocblas_stride    stride_a    = arg.stride_a;
        int64_t           batch_count = arg.batch_count;
        rocblas_fill      uplo        = char2rocblas_fill(arg.uplo);
        rocblas_operation transA      = char2rocblas_operation(arg.transA);

        int64_t row = transA == rocblas_operation_none ? M : N;
        int64_t col = transA == rocblas_operation_none ? N : M;

        lda = std::max(M, lda);

        rocblas_local_handle handle{arg};

        //Hard-code the enum `check_numerics` to `rocblas_check_numerics_mode_fail` which will return `rocblas_status_check_numerics_fail` if the matrix contains a NaN/Inf/denormal value
        rocblas_check_numerics_mode check_numerics = rocblas_check_numerics_mode_fail;

        rocblas_status status          = rocblas_status_success;
        const char     function_name[] = "testing_check_numerics_matrix";
        bool           is_input        = true;

        //Argument sanity check before allocating invalid memory
        if(!M || !N || !batch_count)
            return;

        //Allocating memory for the host matrix
        host_matrix<T> h_A(row, col, lda);
        // triangular and symmetric will be based on columns matching rocblas_init_matrix
        host_matrix<T> h_A_symmetric(col, col, lda);
        host_matrix<T> h_A_triangular(col, col, lda);

        // Allocate memory on device
        device_matrix<T> d_A(row, col, lda);
        CHECK_DEVICE_ALLOCATION(d_A.memcheck());

        device_matrix<T> d_A_triangular(col, col, lda);
        CHECK_DEVICE_ALLOCATION(d_A_triangular.memcheck());

        device_matrix<T> d_A_symmetric(col, col, lda);
        CHECK_DEVICE_ALLOCATION(d_A_symmetric.memcheck());

        //==============================================================================================
        // Initializing random values in the matrix
        //==============================================================================================
        rocblas_init_matrix(
            h_A, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_triangular,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_triangular_matrix,
                            true);
        rocblas_init_matrix(h_A_symmetric,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_symmetric_matrix,
                            true);

        CHECK_HIP_ERROR(d_A.transfer_from(h_A));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 (T*)d_A,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        CHECK_HIP_ERROR(d_A_triangular.transfer_from(h_A_triangular));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 uplo,
                                                                 rocblas_client_triangular_matrix,
                                                                 col,
                                                                 col,
                                                                 (T*)d_A_triangular,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        CHECK_HIP_ERROR(d_A_symmetric.transfer_from(h_A_symmetric));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 uplo,
                                                                 rocblas_client_symmetric_matrix,
                                                                 col,
                                                                 col,
                                                                 (T*)d_A_symmetric,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for zero in the matrix
        //==============================================================================================
        rocblas_init_zero<T>((T*)h_A, row, col, lda);
        rocblas_init_zero_range<T>((T*)h_A_triangular, 0, 1);

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_A.transfer_from(h_A));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 (T*)d_A,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        CHECK_HIP_ERROR(d_A_triangular.transfer_from(h_A_triangular));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 uplo,
                                                                 rocblas_client_triangular_matrix,
                                                                 col,
                                                                 col,
                                                                 (T*)d_A_triangular,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for Inf in the matrix
        //==============================================================================================
        rocblas_init_matrix(
            h_A, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_inf_range<T>((T*)h_A, row - 1, row);
        rocblas_init_inf_range<T>((T*)h_A_symmetric, 0, col);

        CHECK_HIP_ERROR(d_A.transfer_from(h_A));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 (T*)d_A,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        CHECK_HIP_ERROR(d_A_symmetric.transfer_from(h_A_symmetric));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 uplo,
                                                                 rocblas_client_symmetric_matrix,
                                                                 col,
                                                                 col,
                                                                 (T*)d_A_symmetric,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for NaN in the matrix
        //==============================================================================================
        rocblas_init_matrix(
            h_A, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_triangular,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_triangular_matrix,
                            true);

        rocblas_init_nan<T>((T*)h_A, row, col, lda);
        rocblas_init_nan<T>((T*)h_A_triangular, col, col, lda);

        CHECK_HIP_ERROR(d_A.transfer_from(h_A));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 (T*)d_A,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        CHECK_HIP_ERROR(d_A_triangular.transfer_from(h_A_triangular));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 uplo,
                                                                 rocblas_client_triangular_matrix,
                                                                 col,
                                                                 col,
                                                                 (T*)d_A_triangular,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for denorm values in the matrix
        //==============================================================================================
        rocblas_init_matrix(
            h_A, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_symmetric,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_symmetric_matrix,
                            true);

        rocblas_init_denorm<T>((T*)h_A, row, col, lda);
        rocblas_init_denorm<T>((T*)h_A_symmetric, col, col, lda);

        // copy data from CPU to device
        CHECK_HIP_ERROR(d_A.transfer_from(h_A));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 (T*)d_A,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        CHECK_HIP_ERROR(d_A_symmetric.transfer_from(h_A_symmetric));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 uplo,
                                                                 rocblas_client_symmetric_matrix,
                                                                 col,
                                                                 col,
                                                                 (T*)d_A_symmetric,
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 1,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);
    }

    //Testing a matrix for NaN/zero/Inf/denormal values
    template <typename T>
    void testing_check_numerics_matrix_batched(const Arguments& arg)
    {
        int64_t           M           = arg.M;
        int64_t           N           = arg.N;
        int64_t           lda         = arg.lda;
        rocblas_stride    offset_a    = 0;
        rocblas_stride    stride_a    = arg.stride_a;
        int64_t           batch_count = arg.batch_count;
        rocblas_fill      uplo        = char2rocblas_fill(arg.uplo);
        rocblas_operation transA      = char2rocblas_operation(arg.transA);

        int64_t row = transA == rocblas_operation_none ? M : N;
        int64_t col = transA == rocblas_operation_none ? N : M;

        lda = std::max(M, lda);

        rocblas_local_handle handle{arg};

        //Hard-code the enum `check_numerics` to `rocblas_check_numerics_mode_fail` which will return `rocblas_status_check_numerics_fail` if the matrix contains a NaN/Inf/denormal value
        rocblas_check_numerics_mode check_numerics = rocblas_check_numerics_mode_fail;

        rocblas_status status          = rocblas_status_success;
        const char     function_name[] = "testing_check_numerics_matrix";
        bool           is_input        = true;

        //Argument sanity check before allocating invalid memory
        if(!M || !N || !batch_count)
            return;

        //==============================================================================================
        // Initializing random values in batched matrices
        //==============================================================================================
        //Allocating memory for the host batch matrix
        host_batch_matrix<T> h_A_batch(row, col, lda, batch_count);
        // triangular and symmetric will be based on columns matching rocblas_init_matrix
        host_batch_matrix<T> h_A_batch_triangular(col, col, lda, batch_count);
        host_batch_matrix<T> h_A_batch_symmetric(col, col, lda, batch_count);

        // Allocate memory on device
        device_batch_matrix<T> d_A_batch(row, col, lda, batch_count);
        CHECK_DEVICE_ALLOCATION(d_A_batch.memcheck());

        device_batch_matrix<T> d_A_batch_triangular(col, col, lda, batch_count);
        CHECK_DEVICE_ALLOCATION(d_A_batch_triangular.memcheck());

        device_batch_matrix<T> d_A_batch_symmetric(col, col, lda, batch_count);
        CHECK_DEVICE_ALLOCATION(d_A_batch_symmetric.memcheck());

        //Initialize Data on CPU
        rocblas_init_matrix(
            h_A_batch, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);

        rocblas_init_matrix(h_A_batch_triangular,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_triangular_matrix,
                            true);

        rocblas_init_matrix(h_A_batch_symmetric,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_symmetric_matrix,
                            true);

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_A_batch.transfer_from(h_A_batch));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 d_A_batch.const_batch_ptr(),
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        CHECK_HIP_ERROR(d_A_batch_symmetric.transfer_from(h_A_batch_symmetric));
        status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              transA,
                                                              uplo,
                                                              rocblas_client_symmetric_matrix,
                                                              col,
                                                              col,
                                                              d_A_batch_symmetric.const_batch_ptr(),
                                                              offset_a,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for zero in batched matrices
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_matrix(
            h_A_batch, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_batch_triangular,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_triangular_matrix,
                            true);

        for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
            for(size_t j = 0; j < col; ++j)
                for(size_t i = 0; i < row; ++i)
                {
                    h_A_batch[i_batch][i + j * lda] = T(rocblas_zero_rng());
                    if(i < col)
                        h_A_batch_triangular[i_batch][i + j * lda] = T(rocblas_zero_rng());
                }

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_A_batch.transfer_from(h_A_batch));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 d_A_batch.const_batch_ptr(),
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_success);

        CHECK_HIP_ERROR(d_A_batch_triangular.transfer_from(h_A_batch_triangular));
        status = rocblas_internal_check_numerics_matrix_template(
            function_name,
            handle,
            transA,
            uplo,
            rocblas_client_triangular_matrix,
            col,
            col,
            d_A_batch_triangular.const_batch_ptr(),
            offset_a,
            lda,
            stride_a,
            batch_count,
            check_numerics,
            is_input);
        EXPECT_EQ(status, rocblas_status_success);

        //==============================================================================================
        // Initializing and testing for Inf in batched matrices
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_matrix(
            h_A_batch, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_batch_symmetric,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_symmetric_matrix,
                            true);

        for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
            for(size_t j = 0; j < col; ++j)
                for(size_t i = 0; i < row; ++i)
                {
                    h_A_batch[i_batch][i + j * lda] = T(rocblas_inf_rng());
                    if(i < col)
                        h_A_batch_symmetric[i_batch][i + j * lda] = T(rocblas_inf_rng());
                }

        //Transferring data from host to device
        CHECK_HIP_ERROR(d_A_batch.transfer_from(h_A_batch));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 d_A_batch.const_batch_ptr(),
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        CHECK_HIP_ERROR(d_A_batch_symmetric.transfer_from(h_A_batch_symmetric));
        status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              transA,
                                                              uplo,
                                                              rocblas_client_symmetric_matrix,
                                                              col,
                                                              col,
                                                              d_A_batch_symmetric.const_batch_ptr(),
                                                              offset_a,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for NaN in batched matrices
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_matrix(
            h_A_batch, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_batch_triangular,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_triangular_matrix,
                            true);

        for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
            for(size_t j = 0; j < col; ++j)
                for(size_t i = 0; i < row; ++i)
                {
                    h_A_batch[i_batch][i + j * lda] = T(rocblas_nan_rng());
                    if(i < col)
                        h_A_batch_triangular[i_batch][i + j * lda] = T(rocblas_nan_rng());
                }

        CHECK_HIP_ERROR(d_A_batch.transfer_from(h_A_batch));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 d_A_batch.const_batch_ptr(),
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        CHECK_HIP_ERROR(d_A_batch_triangular.transfer_from(h_A_batch_triangular));
        status = rocblas_internal_check_numerics_matrix_template(
            function_name,
            handle,
            transA,
            uplo,
            rocblas_client_triangular_matrix,
            col,
            col,
            d_A_batch_triangular.const_batch_ptr(),
            offset_a,
            lda,
            stride_a,
            batch_count,
            check_numerics,
            is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        //==============================================================================================
        // Initializing and testing for denorm values in batched matrices
        //==============================================================================================
        //Initialize Data on CPU
        rocblas_init_matrix(
            h_A_batch, arg, rocblas_client_never_set_nan, rocblas_client_general_matrix, true);
        rocblas_init_matrix(h_A_batch_symmetric,
                            arg,
                            rocblas_client_never_set_nan,
                            rocblas_client_symmetric_matrix,
                            true);

        for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
            for(size_t j = 0; j < col; ++j)
                for(size_t i = 0; i < row; ++i)
                {
                    h_A_batch[i_batch][i + j * lda] = T(rocblas_denorm_rng());
                    if(i < col)
                        h_A_batch_symmetric[i_batch][i + j * lda] = T(rocblas_denorm_rng());
                }

        CHECK_HIP_ERROR(d_A_batch.transfer_from(h_A_batch));
        status = rocblas_internal_check_numerics_matrix_template(function_name,
                                                                 handle,
                                                                 transA,
                                                                 rocblas_fill_full,
                                                                 rocblas_client_general_matrix,
                                                                 M,
                                                                 N,
                                                                 d_A_batch.const_batch_ptr(),
                                                                 offset_a,
                                                                 lda,
                                                                 stride_a,
                                                                 batch_count,
                                                                 check_numerics,
                                                                 is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);

        CHECK_HIP_ERROR(d_A_batch_symmetric.transfer_from(h_A_batch_symmetric));
        status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              transA,
                                                              uplo,
                                                              rocblas_client_symmetric_matrix,
                                                              col,
                                                              col,
                                                              d_A_batch_symmetric.const_batch_ptr(),
                                                              offset_a,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        EXPECT_EQ(status, rocblas_status_check_numerics_fail);
    }

    // By default, arbitrary type combinations are invalid.
    // The unnamed second parameter is used for enable_if_t below.
    template <typename, typename = void>
    struct check_numerics_matrix_testing : rocblas_test_invalid
    {
    };

    template <typename T>
    struct check_numerics_matrix_testing<
        T,
        std::enable_if_t<
            std::is_same_v<
                T,
                rocblas_half> || std::is_same_v<T, rocblas_bfloat16> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex> || std::is_same_v<T, float> || std::is_same_v<T, double>>>
        : rocblas_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "check_numerics_matrix"))
                testing_check_numerics_matrix<T>(arg);
            else if(!strcmp(arg.function, "check_numerics_matrix_batched"))
                testing_check_numerics_matrix_batched<T>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct check_numerics_matrix
        : RocBLAS_Test<check_numerics_matrix, check_numerics_matrix_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "check_numerics_matrix");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<check_numerics_matrix> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            name << '_' << (char)std::toupper(arg.uplo) << '_' << (char)std::toupper(arg.transA)
                 << '_' << arg.M << '_' << arg.N;
            name << '_' << arg.batch_count;
            return std::move(name);
        }
    };

    TEST_P(check_numerics_matrix, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<check_numerics_matrix_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(check_numerics_matrix);

    struct check_numerics_matrix_batched
        : RocBLAS_Test<check_numerics_matrix, check_numerics_matrix_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return true;
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "check_numerics_matrix_batched");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBLAS_TestName<check_numerics_matrix_batched> name(arg.name);
            name << rocblas_datatype2string(arg.a_type);
            name << '_' << (char)std::toupper(arg.uplo) << '_' << (char)std::toupper(arg.transA)
                 << '_' << arg.M << '_' << arg.N;
            name << '_' << arg.batch_count;
            return std::move(name);
        }
    };

    TEST_P(check_numerics_matrix_batched, auxiliary)
    {
        CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(
            rocblas_simple_dispatch<check_numerics_matrix_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(check_numerics_matrix_batched);

} // namespace
