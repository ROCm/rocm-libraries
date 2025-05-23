/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasDotExModel
    = ArgumentModel<e_a_type, e_b_type, e_c_type, e_compute_type, e_N, e_incx, e_incy>;

inline void testname_dot_ex(const Arguments& arg, std::string& name)
{
    hipblasDotExModel{}.test_name(arg, name);
}

inline void testname_dotc_ex(const Arguments& arg, std::string& name)
{
    hipblasDotExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_ex_bad_arg(const Arguments& arg)
{
    auto hipblasDotExFn = arg.api == FORTRAN ? (CONJ ? hipblasDotcExFortran : hipblasDotExFortran)
                                             : (CONJ ? hipblasDotcEx : hipblasDotEx);
    auto hipblasDotExFn_64 = arg.api == FORTRAN_64
                                 ? (CONJ ? hipblasDotcEx_64Fortran : hipblasDotEx_64Fortran)
                                 : (CONJ ? hipblasDotcEx_64 : hipblasDotEx_64);

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<Tx> dx(N, incx);
        device_vector<Ty> dy(N, incy);
        device_vector<Tr> d_res(1);
        host_vector<Tr>   h_res(1);

        // None of these test cases will write to result so using device pointer is fine for both modes
        DAPI_EXPECT(
            HIPBLAS_STATUS_NOT_INITIALIZED,
            hipblasDotExFn,
            (nullptr, N, dx, xType, incx, dy, yType, incy, d_res, resultType, executionType));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotExFn,
                        (handle,
                         N,
                         nullptr,
                         xType,
                         incx,
                         dy,
                         yType,
                         incy,
                         d_res,
                         resultType,
                         executionType));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasDotExFn,
                        (handle,
                         N,
                         dx,
                         xType,
                         incx,
                         nullptr,
                         yType,
                         incy,
                         d_res,
                         resultType,
                         executionType));
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasDotExFn,
                (handle, N, dx, xType, incx, dy, yType, incy, nullptr, resultType, executionType));

            // This is a little different than the checks for L2. In rocBLAS implementation n <= 0 is a quick-return success before other arg checks.
            // Here, for 32-bit API, I'm counting on the rollover to return success, and for the 64-bit API I'm passing in invalid
            // pointers to get invalid_value returns. Note that result has to be valid as quick-return sets the result
            DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_INVALID_VALUE
                                             : HIPBLAS_STATUS_SUCCESS,
                        hipblasDotExFn,
                        (handle,
                         c_i32_overflow,
                         nullptr,
                         xType,
                         1,
                         nullptr,
                         yType,
                         incy,
                         pointer_mode == HIPBLAS_POINTER_MODE_HOST ? h_res.internal_type() : d_res,
                         resultType,
                         executionType));
        }
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_ex_bad_arg(const Arguments& arg)
{
    testing_dot_ex_bad_arg<Tx, Ty, Tr, Tex, true>(arg);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_ex(const Arguments& arg)
{
    auto hipblasDotExFn = arg.api == FORTRAN ? (CONJ ? hipblasDotcExFortran : hipblasDotExFortran)
                                             : (CONJ ? hipblasDotcEx : hipblasDotEx);
    auto hipblasDotExFn_64 = arg.api == FORTRAN_64
                                 ? (CONJ ? hipblasDotcEx_64Fortran : hipblasDotEx_64Fortran)
                                 : (CONJ ? hipblasDotcEx_64 : hipblasDotEx_64);

    int64_t N    = arg.N;
    int64_t incx = arg.incx;
    int64_t incy = arg.incy;

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(1);
        host_vector<Tr>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasDotExFn,
                   (handle,
                    N,
                    nullptr,
                    xType,
                    incx,
                    nullptr,
                    yType,
                    incy,
                    d_hipblas_result_0,
                    resultType,
                    executionType));

        host_vector<Tr> cpu_0(1);
        host_vector<Tr> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);
        return;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(N, incx);
    host_vector<Ty> hy(N, incy);

    device_vector<Tx> dx(N, incx);
    device_vector<Ty> dy(N, incy);
    device_vector<Tr> d_hipblas_result(1);

    CHECK_DEVICE_ALLOCATION(dx.memcheck());
    CHECK_DEVICE_ALLOCATION(dy.memcheck());
    CHECK_DEVICE_ALLOCATION(d_hipblas_result.memcheck());

    Tr cpu_result, hipblas_result_host, hipblas_result_device;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true, false);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan, false, true);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasDotExFn,
                   (handle,
                    N,
                    dx,
                    xType,
                    incx,
                    dy,
                    yType,
                    incy,
                    &hipblas_result_host,
                    resultType,
                    executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasDotExFn,
                   (handle,
                    N,
                    dx,
                    xType,
                    incx,
                    dy,
                    yType,
                    incy,
                    d_hipblas_result,
                    resultType,
                    executionType));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? ref_dotc<Tx> : ref_dot<Tx>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        if(arg.unit_check)
        {
            if(std::is_same<Tr, hipblasHalf>{})
            {
                double tol = error_tolerance<Tr> * N;
                near_check_general(1, 1, 1, &cpu_result, &hipblas_result_host, tol);
                near_check_general(1, 1, 1, &cpu_result, &hipblas_result_device, tol);
            }
            else
            {
                unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host);
                unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_device);
        }

    } // end of if unit/norm check

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasDotExFn,
                          (handle,
                           N,
                           dx,
                           xType,
                           incx,
                           dy,
                           yType,
                           incy,
                           d_hipblas_result,
                           resultType,
                           executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotExModel{}.log_args<Tx>(std::cout,
                                         arg,
                                         gpu_time_used,
                                         dot_gflop_count<CONJ, Tx>(N),
                                         dot_gbyte_count<Tx>(N),
                                         hipblas_error_host,
                                         hipblas_error_device);
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_ex(const Arguments& arg)
{
    testing_dot_ex<Tx, Ty, Tr, Tex, true>(arg);
}
