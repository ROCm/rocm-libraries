/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

template <typename I, typename X, typename Y, typename T>
void testing_axpby_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle      local_handle;
    rocsparse_handle            handle = local_handle;
    rocsparse_const_spvec_descr x      = (rocsparse_const_spvec_descr)0x4;
    rocsparse_dnvec_descr       y      = (rocsparse_dnvec_descr)0x4;
    const void*                 alpha  = (const void*)0x4;
    const void*                 beta   = (const void*)0x4;
    bad_arg_analysis(rocsparse_axpby, handle, alpha, x, beta, y);
}

template <typename I, typename X, typename Y, typename T>
void testing_axpby(const Arguments& arg)
{
    I size = arg.M;
    I nnz  = arg.nnz;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocsparse_index_base base = arg.baseA;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  xtype = get_datatype<X>();
    rocsparse_datatype  ytype = get_datatype<Y>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory for matrix
    host_vector<I> hx_ind(nnz);
    host_vector<X> hx_val(nnz);
    host_vector<Y> hy_1(size);
    host_vector<Y> hy_2(size);
    host_vector<Y> hy_gold(size);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, size + base);
    rocsparse_init<X>(hx_val, 1, nnz, 1, arg.convert_to_int);
    rocsparse_init<Y>(hy_1, 1, size, 1, arg.convert_to_int);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<I> dx_ind(nnz);
    device_vector<X> dx_val(nnz);
    device_vector<Y> dy_1(size);
    device_vector<Y> dy_2(size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(X) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(Y) * size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_2, dy_1, sizeof(Y) * size, hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, xtype);
    rocsparse_local_dnvec y1(size, dy_1, ytype);
    rocsparse_local_dnvec y2(size, dy_2, ytype);

    if(arg.unit_check)
    {
        // axpby - host pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_axpby(handle, &h_alpha, x, &h_beta, y1));

        // axpby - device pointer mode
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_axpby(handle, d_alpha, x, d_beta, y2));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(Y) * size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(Y) * size, hipMemcpyDeviceToHost));

        // CPU axpby
        host_axpby<T, I, X, Y>(size, nnz, h_alpha, hx_val, hx_ind, h_beta, hy_gold, base);

        hy_gold.unit_check(hy_1);
        hy_gold.unit_check(hy_2);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save(
                "Y with host pointer", dy_1, "Y with device pointer", dy_2);
        }
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_axpby, handle, &h_alpha, x, &h_beta, y1);

        double gflop_count = axpby_gflop_count(nnz);
        double gbyte_count = axpby_gbyte_count<T>(nnz);

        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);

        display_timing_info(display_key_t::size,
                            size,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, XTYPE, YTYPE, TTYPE)                                            \
    template void testing_axpby_bad_arg<ITYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg); \
    template void testing_axpby<ITYPE, XTYPE, YTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, _Float16, _Float16, float);
INSTANTIATE(int32_t, float, float, float);
INSTANTIATE(int32_t, double, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int64_t, _Float16, _Float16, float);
INSTANTIATE(int64_t, float, float, float);
INSTANTIATE(int64_t, double, double, double);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex, rocsparse_double_complex);
void testing_axpby_extra(const Arguments& arg)
{
    // Regression test for AISPARSE-650.
    //
    // Before the fix, axpyi_device computed the element index as
    //   hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x
    // entirely in 32-bit unsigned arithmetic. Once nnz reaches 2^32 the
    // block-index multiply wraps around, so threads in the high blocks operate
    // on wrapped (low) elements while the tail of the sparse vector is never
    // touched. The fix casts the block index to the (64-bit) index type before
    // the multiply and iterates with a grid-stride loop.
    //
    // This drives the 64-bit-index path of rocsparse_axpby (which dispatches to
    // axpyi_template) with nnz just past the 2^32 boundary and checks that an
    // element beyond that boundary is actually accumulated into y. To stay
    // within a single device allocation (host mirrors of the full arrays would
    // need tens of GB) everything is initialized on the device and a single
    // element is probed.
    using I = int64_t;
    using T = float;

    static constexpr int64_t two_pow_32 = static_cast<int64_t>(1) << 32;

    // nnz just beyond 2^32 so at least one block has a block index whose
    // (blockIdx * BLOCKSIZE) product overflows 32-bit arithmetic.
    const I nnz  = two_pow_32 + 512;
    const I size = 2;

    const rocsparse_index_base base = rocsparse_index_base_zero;

    rocsparse_local_handle handle(arg);

    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy(size);

    // Filler elements all reference dense entry 0 and start at value 0.
    CHECK_HIP_ERROR(hipMemset(dx_ind, 0, sizeof(I) * nnz));
    CHECK_HIP_ERROR(hipMemset(dx_val, 0, sizeof(T) * nnz));
    CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * size));

    // The probe lives past the 2^32 boundary. It carries the only non-zero value
    // and references its own dense entry (index 1) so its result cannot be
    // perturbed by the accumulations the filler elements make to dense entry 0.
    const I probe_idx = two_pow_32 + 5;
    const I probe_ind = 1;
    const T x_in      = static_cast<T>(1);
    CHECK_HIP_ERROR(hipMemcpy(
        static_cast<I*>(dx_ind) + probe_idx, &probe_ind, sizeof(I), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(static_cast<T*>(dx_val) + probe_idx, &x_in, sizeof(T), hipMemcpyHostToDevice));

    // beta == 1 leaves the existing y untouched; alpha scales the gathered x.
    const T halpha = static_cast<T>(2);
    const T hbeta  = static_cast<T>(1);

    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, get_indextype<I>(), base, get_datatype<T>());
    rocsparse_local_dnvec y(size, dy, get_datatype<T>());

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(testing::rocsparse_axpby(handle, &halpha, x, &hbeta, y));

    // y[1] = beta * 0 + alpha * x_in = 2. Before the fix the wrapped block index
    // leaves the probe element unprocessed, so y[1] stays 0.
    T y_out = static_cast<T>(0);
    CHECK_HIP_ERROR(hipMemcpy(&y_out, static_cast<T*>(dy) + 1, sizeof(T), hipMemcpyDeviceToHost));

    unit_check_scalar<T>(static_cast<T>(2), y_out);
}
