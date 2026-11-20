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

template <typename I, typename T>
void testing_gather_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle      local_handle;
    rocsparse_handle            handle = local_handle;
    rocsparse_const_dnvec_descr y      = (rocsparse_const_dnvec_descr)0x4;
    rocsparse_spvec_descr       x      = (rocsparse_spvec_descr)0x4;
    bad_arg_analysis(rocsparse_gather, handle, y, x);
}

template <typename I, typename T>
void testing_gather(const Arguments& arg)
{
    I size = arg.M;
    I nnz  = arg.nnz;

    rocsparse_index_base base = arg.baseA;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory for matrix
    host_vector<I> hx_ind(nnz);
    host_vector<T> hx_val(nnz);
    host_vector<T> hx_val_gold(nnz);
    host_vector<T> hy(size);

    // Initialize data on CPU
    rocsparse_seedrand();
    rocsparse_init_index(hx_ind, nnz, base, size + base);
    rocsparse_init<T>(hy, 1, size, 1, arg.convert_to_int);

    // Allocate device memory
    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy(size);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind, sizeof(I) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size, hipMemcpyHostToDevice));

    // Create descriptors
    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, itype, base, ttype);
    rocsparse_local_dnvec y(size, dy, ttype);

    if(arg.unit_check)
    {
        // Gather
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gather(handle, y, x));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hx_val, dx_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        // CPU coomv
        host_gthr<I, T>(nnz, hy, hx_val_gold, hx_ind, base);

        hx_val_gold.unit_check(hx_val);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("X", hx_val);
        }
    }

    if(arg.timing)
    {

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg, rocsparse_gather, handle, y, x);

        double gbyte_count = gthr_gbyte_count<T>(nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, TTYPE)                                             \
    template void testing_gather_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_gather<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int8_t);
INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int8_t);
INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

void testing_gather_extra(const Arguments& arg)
{
    // Regression test for gthr with nnz past the 32-bit dispatch limit.
    //
    // Before the fix the gather launched (nnz - 1) / 512 + 1 blocks of 512
    // threads. For nnz = 2^32 + 512 that is 8,388,609 blocks, so
    // grid.x * blockDim.x exceeded 2^32 - 1 and the launch was rejected. The
    // fix clamps grid.x with get_grid_size_x and grid-strides over the rest
    // with a 64-bit index.
    //
    // This drives the 64-bit-index path of rocsparse_gather (which dispatches
    // to gthr_template) with nnz just past 2^32 and checks that an element
    // beyond that boundary is gathered. Everything is initialized on the
    // device and a single element is probed.
    using I = int64_t;
    using T = float;

    static constexpr int64_t two_pow_32 = static_cast<int64_t>(1) << 32;

    // nnz just beyond 2^32, so the unclamped grid exceeds the dispatch limit.
    const I nnz = two_pow_32 + 512;

    // rocsparse_create_spvec_descr rejects nnz > size, so the dense vector has
    // to be at least as long as the sparse one.
    const I size = nnz;

    const rocsparse_index_base base = rocsparse_index_base_zero;

    rocsparse_local_handle handle(arg);

    device_vector<I> dx_ind(nnz);
    device_vector<T> dx_val(nnz);
    device_vector<T> dy(size);

    // Filler elements all gather dense entry 0; x_val starts at 0.
    CHECK_HIP_ERROR(hipMemset(dx_ind, 0, sizeof(I) * nnz));
    CHECK_HIP_ERROR(hipMemset(dx_val, 0, sizeof(T) * nnz));
    CHECK_HIP_ERROR(hipMemset(dy, 0, sizeof(T) * size));

    // y[1] holds the only non-zero dense value; y[0] stays 0.
    const T y_one = static_cast<T>(3);
    CHECK_HIP_ERROR(hipMemcpy(static_cast<T*>(dy) + 1, &y_one, sizeof(T), hipMemcpyHostToDevice));

    // The probe lives past the 2^32 boundary and references dense entry 1, so
    // after the gather its x_val must equal y[1].
    const I probe_idx = two_pow_32 + 5;
    const I probe_ind = 1;
    CHECK_HIP_ERROR(hipMemcpy(
        static_cast<I*>(dx_ind) + probe_idx, &probe_ind, sizeof(I), hipMemcpyHostToDevice));

    rocsparse_local_spvec x(size, nnz, dx_ind, dx_val, get_indextype<I>(), base, get_datatype<T>());
    rocsparse_local_dnvec y(size, dy, get_datatype<T>());

    CHECK_ROCSPARSE_ERROR(testing::rocsparse_gather(handle, y, x));

    // The probe is past 2^32, so it is only gathered if the grid-stride loop
    // reaches the elements the clamped grid does not cover.
    T x_out = static_cast<T>(0);
    CHECK_HIP_ERROR(
        hipMemcpy(&x_out, static_cast<T*>(dx_val) + probe_idx, sizeof(T), hipMemcpyDeviceToHost));

    unit_check_scalar<T>(y_one, x_out);
}
