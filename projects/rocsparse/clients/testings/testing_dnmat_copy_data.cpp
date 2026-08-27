/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_clients_dnmat_descr.hpp"
#include "rocsparse_clients_dnvec_descr.hpp"
#include "testing.hpp"

extern "C" rocsparse_status rocsparse_dnmat_copy_data(rocsparse_handle            handle,
                                                      rocsparse_const_dnvec_descr alpha,
                                                      rocsparse_const_dnmat_descr X,
                                                      rocsparse_dnmat_descr       Y,
                                                      rocsparse_error*            p_error);

template <typename T>
void testing_dnmat_copy_data_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;
    rocsparse_error*       p_error{};

    for(rocsparse_order t_order : {rocsparse_order_row, rocsparse_order_column})

    {
        for(rocsparse_order s_order : {rocsparse_order_row, rocsparse_order_column})

        {
            device_dense_vector<T> dalpha(1);
            rocsparse_local_dnvec  ALPHA(dalpha);

            for(rocsparse_dnvec_descr alpha :
                {((rocsparse_dnvec_descr) nullptr), ((rocsparse_dnvec_descr)ALPHA)})
            {
                {
                    device_dense_vector<float>  dfalpha(1);
                    rocsparse_local_dnvec       FALPHA(dfalpha);
                    rocsparse_dnvec_descr       falpha = FALPHA;
                    device_dense_matrix<double> dsource(3, 2, s_order);
                    device_dense_matrix<double> dtarget(3, 2, t_order);
                    rocsparse_local_dnmat       SOURCE(dsource);
                    rocsparse_local_dnmat       TARGET(dtarget);
                    rocsparse_dnmat_descr       source = SOURCE;
                    rocsparse_dnmat_descr       target = TARGET;
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_dnmat_copy_data(handle, falpha, source, target, p_error),
                        rocsparse_status_not_implemented);
                }

                {
                    if(alpha)
                    {
                        device_dense_matrix<T> dsource(3, 2, s_order);
                        device_dense_matrix<T> dtarget(3, 2, t_order);
                        rocsparse_local_dnmat  SOURCE(dsource);
                        rocsparse_local_dnmat  TARGET(dtarget);
                        rocsparse_dnmat_descr  source = SOURCE;
                        rocsparse_dnmat_descr  target = TARGET;

                        device_dense_vector<T> dfalpha(1);
                        rocsparse_local_dnvec  FALPHA(dfalpha);
                        rocsparse_dnvec_descr  falpha = FALPHA;

                        CHECK_ROCSPARSE_ERROR(rocsparse_dnvec_set_strided_batch(falpha, 12, 6));

                        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(source, 2, 6));

                        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(target, 2, 6));

                        EXPECT_ROCSPARSE_STATUS(
                            rocsparse_dnmat_copy_data(handle, falpha, source, target, p_error),
                            rocsparse_status_invalid_value);
                    }
                }

                {

                    device_dense_matrix<T>   dsource(3, 2, s_order);
                    device_dense_matrix<T>   dtarget(3, 2, t_order);
                    rocsparse_local_dnmat    SOURCE(dsource);
                    rocsparse_local_dnmat    TARGET(dtarget);
                    rocsparse_dnmat_descr    source              = SOURCE;
                    rocsparse_dnmat_descr    target              = TARGET;
                    static constexpr int32_t nexcludes           = 2;
                    static constexpr int32_t excludes[nexcludes] = {1, 4};
                    select_bad_arg_analysis(rocsparse_dnmat_copy_data,
                                            nexcludes,
                                            excludes,
                                            handle,
                                            alpha,
                                            source,
                                            target,
                                            p_error);
                }

                {
                    device_dense_matrix<float>  dsource(3, 2, s_order);
                    device_dense_matrix<double> dtarget(3, 2, t_order);
                    rocsparse_local_dnmat       SOURCE(dsource);
                    rocsparse_local_dnmat       TARGET(dtarget);
                    rocsparse_dnmat_descr       source = SOURCE;
                    rocsparse_dnmat_descr       target = TARGET;
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_dnmat_copy_data(handle, alpha, source, target, p_error),
                        rocsparse_status_not_implemented);
                }

                {
                    device_dense_matrix<T> dsource(3, 2, s_order);
                    device_dense_matrix<T> dtarget(4, 2, t_order);
                    rocsparse_local_dnmat  SOURCE(dsource);
                    rocsparse_local_dnmat  TARGET(dtarget);
                    rocsparse_dnmat_descr  source = SOURCE;
                    rocsparse_dnmat_descr  target = TARGET;
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_dnmat_copy_data(handle, alpha, source, target, p_error),
                        rocsparse_status_invalid_value);
                }

                {
                    device_dense_matrix<T> dsource(3, 2, s_order);
                    device_dense_matrix<T> dtarget(3, 4, t_order);
                    rocsparse_local_dnmat  SOURCE(dsource);
                    rocsparse_local_dnmat  TARGET(dtarget);
                    rocsparse_dnmat_descr  source = SOURCE;
                    rocsparse_dnmat_descr  target = TARGET;
                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_dnmat_copy_data(handle, alpha, source, target, p_error),
                        rocsparse_status_invalid_value);
                }

                {
                    device_dense_matrix<T> dsource(3, 2, s_order);
                    device_dense_matrix<T> dtarget(3, 2, t_order);
                    rocsparse_local_dnmat  SOURCE(dsource);
                    rocsparse_local_dnmat  TARGET(dtarget);
                    rocsparse_dnmat_descr  source = SOURCE;
                    rocsparse_dnmat_descr  target = TARGET;

                    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(source, 1, 6));

                    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(target, 2, 6));

                    EXPECT_ROCSPARSE_STATUS(
                        rocsparse_dnmat_copy_data(handle, alpha, source, target, p_error),
                        rocsparse_status_invalid_value);
                }
            }
        }
    }
}

template <typename T>
void testing_dnmat_copy_data(const Arguments& arg)
{
    static constexpr bool verbose = true;

    const int64_t M = arg.M;
    const int64_t N = arg.N;
    const int64_t batch_count
        = (arg.batch_count_C == -1) ? 1 : ((arg.batch_count_C == 0) ? 1 : arg.batch_count_C);
    const int64_t X_batch_count     = (arg.batch_count_B == -1) ? batch_count : arg.batch_count_B;
    const int64_t alpha_batch_count = (arg.batch_count_A == -1) ? batch_count : arg.batch_count_A;
    const int64_t Y_batch_count     = batch_count;

    const auto                X_order   = arg.orderB;
    const auto                Y_order   = arg.orderC;
    const int64_t             MxN       = int64_t(M) * N;
    const rocsparse_direction direction = arg.direction;

    const bool is_test_direction_column = (direction == rocsparse_direction_column);
    const bool is_X_order_row           = (X_order == rocsparse_order_row);
    const bool is_Y_order_row           = (Y_order == rocsparse_order_row);

    const int64_t global_x_M = (is_test_direction_column) ? M * X_batch_count : M;
    const int64_t global_x_N = (is_test_direction_column) ? N : N * X_batch_count;

    const int64_t global_y_M = (is_test_direction_column) ? M * batch_count : M;
    const int64_t global_y_N = (is_test_direction_column) ? N : N * batch_count;

    const int64_t X_ld = std::max(int64_t(1), ((is_X_order_row) ? global_x_N : global_x_M));

    const int64_t X_batch_stride = (X_batch_count > 1)
                                       ? ((is_test_direction_column) ? ((is_X_order_row) ? MxN : M)
                                                                     : ((is_X_order_row) ? N : MxN))
                                       : 0;

    const int64_t Y_ld = std::max(int64_t(1), (is_Y_order_row) ? global_y_N : global_y_M);

    const int64_t Y_batch_stride
        = (is_test_direction_column) ? ((is_Y_order_row) ? MxN : M) : ((is_Y_order_row) ? N : MxN);

    const int64_t alpha_batch_stride = (alpha_batch_count == 1) ? 0 : 1;

    if(verbose)
    {
        std::cout << "Info test:" << std::endl;
        std::cout << " M                 : " << M << std::endl;
        std::cout << " N                 : " << N << std::endl;
        std::cout << std::endl;
        std::cout << " alpha               " << std::endl;
        std::cout << "  - batch_count    : " << alpha_batch_count << std::endl;
        std::cout << "  - batch_stride   : " << alpha_batch_stride << std::endl;
        std::cout << std::endl;
        std::cout << " X                   " << std::endl;
        std::cout << "  - batch_count     : " << X_batch_count << std::endl;
        std::cout << "  - order           : " << rocsparse_order2string(X_order) << std::endl;
        std::cout << "  - ld              : " << X_ld << std::endl;
        std::cout << "  - batch_stride    : " << X_batch_stride << std::endl;
        std::cout << std::endl;
        std::cout << " Y                   " << std::endl;
        std::cout << "  - order           : " << rocsparse_order2string(Y_order) << std::endl;
        std::cout << "  - ld              : " << Y_ld << std::endl;
        std::cout << "  - batch_stride    : " << Y_batch_stride << std::endl;
        std::cout << "  - batch_count     : " << Y_batch_count << std::endl;
        std::cout << std::endl;
        std::cout << " test                   " << std::endl;
        std::cout << "  - direction       : " << rocsparse_direction2string(direction) << std::endl;
        std::cout << "  - global M        : " << global_x_M << std::endl;
        std::cout << "  - global N        : " << global_x_N << std::endl;
    }
    rocsparse_error*       p_error{};
    rocsparse_local_handle handle{};
    const bool             no_scale = (arg.get_alpha<T>() == static_cast<T>(1));

    //
    // We need to wait the scalar C-APint64_t to test the host pointer mode.
    //

    host_dense_vector<T> halphamem(alpha_batch_count);
    host_dense_vector<T> hbetamem(alpha_batch_count);
    halphamem[0] = arg.get_alpha<T>();
    for(int64_t i = 1; i < alpha_batch_count; ++i)
    {
        halphamem[i] = halphamem[i - 1] * static_cast<T>(2);
    }

    for(int64_t i = 0; i < alpha_batch_count; ++i)
    {
        hbetamem[i] = static_cast<T>(1) / halphamem[i];
    }

    device_dense_vector<T> dalphamem(halphamem);
    device_dense_vector<T> dbetamem(hbetamem);

    rocsparse_local_dnvec dn_alpha(1, dalphamem, get_datatype<T>());
    rocsparse_local_dnvec dn_beta(1, dbetamem, get_datatype<T>());
    if(alpha_batch_count > 1)
    {
        CHECK_ROCSPARSE_ERROR(
            rocsparse_dnvec_set_strided_batch(dn_alpha, alpha_batch_count, alpha_batch_stride));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_dnvec_set_strided_batch(dn_beta, alpha_batch_count, alpha_batch_stride));
    }

    device_dense_matrix<T, int64_t> global_x(global_x_M, global_x_N, X_order);
    device_dense_matrix<T, int64_t> global_y(global_y_M, global_y_N, Y_order);

    device_dense_matrix_view<T, int64_t> dX(M, N, global_x, X_ld, X_order);
    rocsparse_local_dnmat                X(dX);

    host_dense_matrix<T, int64_t> hxmem(global_x_M, global_x_N, X_order);
    rocsparse_matrix_utils::init<T, int64_t>(hxmem);
    global_x.transfer_from(hxmem);

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(X, X_batch_count, X_batch_stride));

    device_dense_matrix_view<T, int64_t> dY(M, N, global_y, Y_ld, Y_order);
    rocsparse_local_dnmat                Y(dY);

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(Y, Y_batch_count, Y_batch_stride));

    rocsparse_dnvec_descr alpha
        = (no_scale) ? ((rocsparse_dnvec_descr) nullptr) : ((rocsparse_dnvec_descr)dn_alpha);
    rocsparse_dnvec_descr beta
        = (no_scale) ? ((rocsparse_dnvec_descr) nullptr) : ((rocsparse_dnvec_descr)dn_beta);

    if(arg.unit_check)
    {
        const auto Z_order        = X_order;
        const bool is_Z_order_row = (X_order == rocsparse_order_row);

        const int64_t global_z_M = (is_test_direction_column) ? M * batch_count : M;
        const int64_t global_z_N = (is_test_direction_column) ? N : N * batch_count;
        const int64_t Z_ld       = std::max(int64_t(1), (is_Z_order_row) ? global_z_N : global_z_M);

        const int64_t Z_batch_stride = (is_test_direction_column) ? ((is_Z_order_row) ? MxN : M)
                                                                  : ((is_Z_order_row) ? N : MxN);

        device_dense_matrix<T, int64_t> global_z(global_z_M, global_z_N, Z_order);

        device_dense_matrix_view<T, int64_t> dZ(M, N, global_z, Z_ld, Z_order);
        rocsparse_local_dnmat                Z(dZ);
        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_strided_batch(Z, batch_count, Z_batch_stride));

        host_dense_matrix<T, int64_t> hmem(global_z_M, global_z_N, Z_order);
        if((X_batch_count == 1) && (batch_count > 1))
        {
            const auto stride = (is_test_direction_column) ? ((is_Z_order_row) ? MxN : M)
                                                           : ((is_Z_order_row) ? N : MxN);

            switch(X_order)
            {
            case rocsparse_order_row:
            {
                for(int64_t k = 0; k < batch_count; ++k)
                {
                    for(int64_t i = 0; i < M; ++i)
                    {
                        for(int64_t j = 0; j < N; ++j)
                        {
                            hmem[k * stride + i * hmem.ld + j] = hxmem[i * hxmem.ld + j];
                        }
                    }
                }
                break;
            }
            case rocsparse_order_column:
            {

                for(int64_t k = 0; k < batch_count; ++k)
                {
                    for(int64_t j = 0; j < N; ++j)
                    {
                        for(int64_t i = 0; i < M; ++i)
                        {
                            hmem[k * stride + j * hmem.ld + i] = hxmem[j * hxmem.ld + i];
                        }
                    }
                }
                break;
            }
            }
        }
        else
        {
            hmem.transfer_from(hxmem);
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_copy_data(handle, alpha, X, Y, p_error));

        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_copy_data(handle, beta, Y, Z, p_error));

        hmem.unit_check(global_z);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_dnmat_copy_data, handle, alpha, X, Y, p_error);

        const double gflop_count = ((alpha) ? (double(size_t(M) * N * batch_count)) : 0) / 1e9;
        const double gpu_gflops  = get_gpu_gflops(gpu_time_used, gflop_count);

        const double gbyte_count
            = (size_t(M) * N * sizeof(T) * (batch_count + X_batch_count)) / 1e9;
        const double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::batch_count_C,
                            batch_count,
                            display_key_t::batch_count_A,
                            alpha_batch_count,
                            display_key_t::batch_count_B,
                            X_batch_count,
                            display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::order_B,
                            X_order,
                            display_key_t::order_C,
                            Y_order,
                            display_key_t::alpha,
                            halphamem[0],
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TTYPE)                                                      \
    template void testing_dnmat_copy_data_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_dnmat_copy_data<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_dnmat_copy_data_extra(const Arguments& arg) {}
