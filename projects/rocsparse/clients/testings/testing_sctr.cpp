/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename T>
void testing_sctr_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle   = local_handle;
    rocsparse_int          nnz      = 2;
    const T*               x_val    = (const T*)0x4;
    const rocsparse_int*   x_ind    = (const rocsparse_int*)0x4;
    T*                     y        = (T*)0x4;
    rocsparse_index_base   idx_base = rocsparse_index_base_zero;
    bad_arg_analysis(rocsparse_sctr<T>, handle, nnz, x_val, x_ind, y, idx_base);
}

template <typename T>
void testing_sctr(const Arguments& arg)
{
    rocsparse_int        M    = arg.M;
    rocsparse_int        nnz  = arg.nnz;
    rocsparse_index_base base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory
    host_vector<rocsparse_int> hx_ind(nnz);
    host_vector<T>             hx_val(nnz);
    host_vector<T>             hy_1(M);
    host_vector<T>             hy_2(M);
    host_vector<T>             hy_gold(M);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, M + base);
    rocsparse_init<T>(hx_val, 1, nnz, 1, arg.convert_to_int);
    hy_2    = hy_1;
    hy_gold = hy_1;

    // Allocate device memory
    device_vector<rocsparse_int> dx_ind(nnz);
    device_vector<T>             dx_val(nnz);
    device_vector<T>             dy_1(M);
    device_vector<T>             dy_2(M);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val, hx_val, sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1, sizeof(T) * M, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2, sizeof(T) * M, hipMemcpyHostToDevice));

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_sctr<T>(handle, nnz, dx_val, dx_ind, dy_1, base));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_sctr<T>(handle, nnz, dx_val, dx_ind, dy_2, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hy_1, dy_1, sizeof(T) * M, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2, dy_2, sizeof(T) * M, hipMemcpyDeviceToHost));

        // CPU sctr
        host_sctr<rocsparse_int, T>(nnz, hx_val, hx_ind, hy_gold, base);

        hy_gold.unit_check(hy_1);
        hy_gold.unit_check(hy_2);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save(
                "Y pointer mode host", hy_1, "Y pointer mode device", hy_2);
        }
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_sctr<T>, handle, nnz, dx_val, dx_ind, dy_1, base);

        double gbyte_count = sctr_gbyte_count<T>(nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);
        display_timing_info(display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                           \
    template void testing_sctr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_sctr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_sctr_extra(const Arguments& arg)
{
    // Regression test for AISPARSE-652.
    //
    // Before the fix, sctr_device computed the element index as
    //   hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x
    // entirely in 32-bit unsigned arithmetic. Once nnz reaches 2^32 the
    // block-index multiply wraps around, so the tail of the sparse vector past
    // the wrap point is never scattered into y. The fix casts the block index
    // to the (64-bit) index type before the multiply and iterates with a
    // grid-stride loop.
    //
    // This drives the 64-bit-index path of rocsparse_scatter (which dispatches
    // to sctr_template) with nnz just past the 2^32 boundary and checks that an
    // element beyond that boundary is actually scattered. To stay within a
    // single device allocation everything is initialized on the device and a
    // single element is probed.
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

    // Filler elements all reference dense entry 0 and scatter value 0; y is 0.
    CHECK_HIP_ERROR(hipMemset(dx_ind, 0, sizeof(I) * nnz));
    CHECK_HIP_ERROR(hipMemset(dx_val, 0, sizeof(T) * nnz));
    CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * size));

    // The probe lives past the 2^32 boundary. It scatters the only non-zero
    // value into dense entry 1, which no filler element writes, so its result
    // is isolated from the filler scatters to dense entry 0.
    const I probe_idx = two_pow_32 + 5;
    const I probe_ind = 1;
    const T x_in      = static_cast<T>(3);
    CHECK_HIP_ERROR(hipMemcpy(
        static_cast<I*>(dx_ind) + probe_idx, &probe_ind, sizeof(I), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(static_cast<T*>(dx_val) + probe_idx, &x_in, sizeof(T), hipMemcpyHostToDevice));

    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, get_indextype<I>(), base, get_datatype<T>());
    rocsparse_local_dnvec y(size, dy, get_datatype<T>());

    CHECK_ROCSPARSE_ERROR(testing::rocsparse_scatter(handle, x, y));

    // y[1] = x_in = 3. Before the fix the wrapped block index leaves the probe
    // element unscattered, so y[1] stays 0.
    T y_out = static_cast<T>(0);
    CHECK_HIP_ERROR(hipMemcpy(&y_out, static_cast<T*>(dy) + 1, sizeof(T), hipMemcpyDeviceToHost));

    unit_check_scalar<T>(x_in, y_out);
}
