/*! \file */
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

#include "testing.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>

namespace
{
    template <typename I, typename T>
    void host_spsort_coo(rocsparse_direction dir, host_coo_matrix<T, I>& A)
    {
        const int64_t nnz = A.nnz;
        const I*      row = A.row_ind.data();
        const I*      col = A.col_ind.data();
        const T*      val = A.val.data();

        std::vector<int64_t> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);

        if(dir == rocsparse_direction_row)
        {
            std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return std::tie(row[a], col[a]) < std::tie(row[b], col[b]);
            });
        }
        else
        {
            std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return std::tie(col[a], row[a]) < std::tie(col[b], row[b]);
            });
        }

        std::vector<I> sorted_row(nnz);
        std::vector<I> sorted_col(nnz);
        std::vector<T> sorted_val(nnz);
        for(int64_t k = 0; k < nnz; ++k)
        {
            sorted_row[k] = row[perm[k]];
            sorted_col[k] = col[perm[k]];
            sorted_val[k] = val[perm[k]];
        }

        std::copy(sorted_row.begin(), sorted_row.end(), A.row_ind.data());
        std::copy(sorted_col.begin(), sorted_col.end(), A.col_ind.data());
        std::copy(sorted_val.begin(), sorted_val.end(), A.val.data());
    }

    template <typename I, typename T>
    void host_shuffle_coo(host_coo_matrix<T, I>& A)
    {
        const int64_t nnz = A.nnz;
        I*            row = A.row_ind.data();
        I*            col = A.col_ind.data();
        T*            val = A.val.data();

        for(int64_t i = 0; i < nnz; ++i)
        {
            const int64_t j = rand() % nnz;
            std::swap(row[i], row[j]);
            std::swap(col[i], col[j]);
            std::swap(val[i], val[j]);
        }
    }

    void set_spsort_inputs(rocsparse_handle       handle,
                           rocsparse_spsort_descr descr,
                           rocsparse_spsort_alg   alg,
                           rocsparse_direction    dir)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr));
    }
}

template <typename I, typename T>
void testing_spsort_coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    // Pointer and enum checks, with dummy descriptors.
    {
        rocsparse_spsort_descr      descr   = (rocsparse_spsort_descr)0x4;
        rocsparse_const_spmat_descr source  = (rocsparse_const_spmat_descr)0x4;
        rocsparse_spmat_descr       target  = (rocsparse_spmat_descr)0x4;
        rocsparse_spsort_stage      stage   = rocsparse_spsort_stage_analysis;
        rocsparse_error*            p_error = nullptr;

        {
            size_t* buffer_size_in_bytes = (size_t*)0x4;
#define PARAMS_BUFFER_SIZE handle, descr, source, target, stage, buffer_size_in_bytes, p_error
            static constexpr int nex     = 1;
            static const int     ex[nex] = {6};
            select_bad_arg_analysis(rocsparse_spsort_buffer_size, nex, ex, PARAMS_BUFFER_SIZE);
#undef PARAMS_BUFFER_SIZE
        }

        {
            const size_t buffer_size_in_bytes = 10;
            void*        temp_buffer          = (void*)0x4;
#define PARAMS handle, descr, source, target, stage, buffer_size_in_bytes, temp_buffer, p_error
            static constexpr int nex     = 2;
            static const int     ex[nex] = {5, 7};
            select_bad_arg_analysis(rocsparse_spsort, nex, ex, PARAMS);
#undef PARAMS
        }

        {
            rocsparse_spsort_input input              = rocsparse_spsort_input_alg;
            const void*            data               = (const void*)0x4;
            const size_t           data_size_in_bytes = sizeof(rocsparse_spsort_alg);
#define PARAMS_SET_INPUT handle, descr, input, data, data_size_in_bytes, p_error
            static constexpr int nex     = 2;
            static const int     ex[nex] = {4, 5};
            select_bad_arg_analysis(rocsparse_spsort_set_input, nex, ex, PARAMS_SET_INPUT);
#undef PARAMS_SET_INPUT
        }
    }

    {
        rocsparse_spsort_descr descr{};
        EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_descr_create(nullptr, &descr, nullptr),
                                rocsparse_status_invalid_handle);
    }
    EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_descr_create(handle, nullptr, nullptr),
                            rocsparse_status_invalid_pointer);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_descr_destroy(nullptr, nullptr, nullptr),
                            rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_descr_destroy(handle, nullptr, nullptr),
                            rocsparse_status_success);

    // Descriptor input and stage ordering checks, with a real descriptor and matrix.
    {
        device_vector<I> d_coo_row_ind(safe_size);
        device_vector<I> d_coo_col_ind(safe_size);
        device_vector<T> d_coo_val(safe_size);

        rocsparse_local_spmat mat(safe_size,
                                  safe_size,
                                  safe_size,
                                  d_coo_row_ind,
                                  d_coo_col_ind,
                                  d_coo_val,
                                  get_indextype<I>(),
                                  rocsparse_index_base_zero,
                                  get_datatype<T>());

        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        const rocsparse_direction  dir = rocsparse_direction_row;

        rocsparse_spsort_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
        set_spsort_inputs(handle, descr, alg, dir);

        // The output matrix must match the input matrix.
        {
            size_t buffer_size;
            auto   expect_mismatch = [&](int64_t              m,
                                       int64_t              n,
                                       int64_t              nnz,
                                       rocsparse_indextype  itype,
                                       rocsparse_index_base base,
                                       rocsparse_datatype   ttype,
                                       rocsparse_status     status) {
                rocsparse_local_spmat mat_B(
                    m, n, nnz, d_coo_row_ind, d_coo_col_ind, d_coo_val, itype, base, ttype);
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_spsort_buffer_size(handle,
                                                 descr,
                                                 mat,
                                                 mat_B,
                                                 rocsparse_spsort_stage_analysis,
                                                 &buffer_size,
                                                 nullptr),
                    status);
                EXPECT_ROCSPARSE_STATUS(rocsparse_spsort(handle,
                                                         descr,
                                                         mat,
                                                         mat_B,
                                                         rocsparse_spsort_stage_analysis,
                                                         0,
                                                         nullptr,
                                                         nullptr),
                                        status);
            };

            const rocsparse_indextype other_itype = (get_indextype<I>() == rocsparse_indextype_i32)
                                                        ? rocsparse_indextype_i64
                                                        : rocsparse_indextype_i32;
            const rocsparse_datatype  other_ttype = (get_datatype<T>() == rocsparse_datatype_f32_r)
                                                        ? rocsparse_datatype_f64_r
                                                        : rocsparse_datatype_f32_r;

            // clang-format off
            expect_mismatch(safe_size - 1, safe_size, safe_size, get_indextype<I>(), rocsparse_index_base_zero, get_datatype<T>(), rocsparse_status_invalid_size);
            expect_mismatch(safe_size, safe_size - 1, safe_size, get_indextype<I>(), rocsparse_index_base_zero, get_datatype<T>(), rocsparse_status_invalid_size);
            expect_mismatch(safe_size, safe_size, safe_size - 1, get_indextype<I>(), rocsparse_index_base_zero, get_datatype<T>(), rocsparse_status_invalid_size);
            expect_mismatch(safe_size, safe_size, safe_size, other_itype, rocsparse_index_base_zero, get_datatype<T>(), rocsparse_status_invalid_value);
            expect_mismatch(safe_size, safe_size, safe_size, get_indextype<I>(), rocsparse_index_base_one, get_datatype<T>(), rocsparse_status_invalid_value);
            expect_mismatch(safe_size, safe_size, safe_size, get_indextype<I>(), rocsparse_index_base_zero, other_ttype, rocsparse_status_invalid_value);
            // clang-format on

            rocsparse_local_spmat mat_B_csr(safe_size,
                                            safe_size,
                                            safe_size,
                                            d_coo_row_ind,
                                            d_coo_col_ind,
                                            d_coo_val,
                                            get_indextype<I>(),
                                            get_indextype<I>(),
                                            rocsparse_index_base_zero,
                                            get_datatype<T>());
            EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_buffer_size(handle,
                                                                 descr,
                                                                 mat,
                                                                 mat_B_csr,
                                                                 rocsparse_spsort_stage_analysis,
                                                                 &buffer_size,
                                                                 nullptr),
                                    rocsparse_status_invalid_value);

            auto expect_batch_status = [&](int64_t          batch_count_A,
                                           int64_t          batch_stride_A,
                                           int64_t          batch_count_B,
                                           int64_t          batch_stride_B,
                                           rocsparse_status status) {
                rocsparse_local_spmat mat_A(safe_size,
                                            safe_size,
                                            safe_size,
                                            d_coo_row_ind,
                                            d_coo_col_ind,
                                            d_coo_val,
                                            get_indextype<I>(),
                                            rocsparse_index_base_zero,
                                            get_datatype<T>());
                rocsparse_local_spmat mat_B(safe_size,
                                            safe_size,
                                            safe_size,
                                            d_coo_row_ind,
                                            d_coo_col_ind,
                                            d_coo_val,
                                            get_indextype<I>(),
                                            rocsparse_index_base_zero,
                                            get_datatype<T>());
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_coo_set_strided_batch(mat_A, batch_count_A, batch_stride_A));
                CHECK_ROCSPARSE_ERROR(
                    rocsparse_coo_set_strided_batch(mat_B, batch_count_B, batch_stride_B));
                EXPECT_ROCSPARSE_STATUS(
                    rocsparse_spsort_buffer_size(handle,
                                                 descr,
                                                 mat_A,
                                                 mat_B,
                                                 rocsparse_spsort_stage_analysis,
                                                 &buffer_size,
                                                 nullptr),
                    status);
                EXPECT_ROCSPARSE_STATUS(rocsparse_spsort(handle,
                                                         descr,
                                                         mat_A,
                                                         mat_B,
                                                         rocsparse_spsort_stage_analysis,
                                                         0,
                                                         nullptr,
                                                         nullptr),
                                        status);
            };

            // Different batch counts.
            expect_batch_status(1, 0, 2, safe_size, rocsparse_status_invalid_value);
            expect_batch_status(2, safe_size, 3, safe_size, rocsparse_status_invalid_value);

            // Batch strides that make the batches overlap.
            expect_batch_status(2, safe_size - 1, 2, safe_size, rocsparse_status_invalid_size);
            expect_batch_status(2, safe_size, 2, 0, rocsparse_status_invalid_size);
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));

        // Wrong input sizes.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg) + 1, nullptr),
            rocsparse_status_invalid_size);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir) + 1, nullptr),
            rocsparse_status_invalid_size);

        // The algorithm has not been set yet.
        size_t buffer_size;
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_buffer_size(
                handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
            rocsparse_status_invalid_value);

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr));

        // The direction has not been set yet.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_buffer_size(
                handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr),
            rocsparse_status_invalid_value);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort(
                handle, descr, mat, mat, rocsparse_spsort_stage_analysis, 0, nullptr, nullptr),
            rocsparse_status_invalid_value);

        // Invalid direction value.
        const rocsparse_direction invalid_dir = (rocsparse_direction)-1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_set_input(handle,
                                                           descr,
                                                           rocsparse_spsort_input_direction,
                                                           &invalid_dir,
                                                           sizeof(invalid_dir),
                                                           nullptr),
                                rocsparse_status_invalid_value);

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr));

        // Compute cannot be executed before analysis.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort(
                handle, descr, mat, mat, rocsparse_spsort_stage_compute, 0, nullptr, nullptr),
            rocsparse_status_invalid_value);

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
            handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr));
        void* dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(handle,
                                               descr,
                                               mat,
                                               mat,
                                               rocsparse_spsort_stage_analysis,
                                               buffer_size,
                                               dbuffer,
                                               nullptr));

        // Analysis cannot be executed twice.
        EXPECT_ROCSPARSE_STATUS(rocsparse_spsort(handle,
                                                 descr,
                                                 mat,
                                                 mat,
                                                 rocsparse_spsort_stage_analysis,
                                                 buffer_size,
                                                 dbuffer,
                                                 nullptr),
                                rocsparse_status_invalid_value);

        // The algorithm cannot be changed after analysis.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr),
            rocsparse_status_internal_error);

        // The direction cannot be changed after analysis.
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_spsort_set_input(
                handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), nullptr),
            rocsparse_status_internal_error);

        CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
    }
}

template <typename I, typename T>
void testing_spsort_coo(const Arguments& arg)
{
    I                    M    = arg.M;
    I                    N    = arg.N;
    rocsparse_index_base base = arg.baseA;
    rocsparse_direction  dir  = arg.direction;
    rocsparse_spsort_alg alg  = rocsparse_spsort_alg_default;

    const int64_t batch_count = std::max<int64_t>(arg.batch_count, 1);

    rocsparse_local_handle handle(arg);

    rocsparse_matrix_factory<T, I, I> matrix_factory(arg);

    host_coo_matrix<T, I> hA_single;
    matrix_factory.init_coo(hA_single, M, N, base);

    const int64_t nnz = hA_single.nnz;

    // A is padded between batches to check that its stride is honoured, B is packed.
    const int64_t batch_stride_A = (batch_count > 1) ? nnz + 3 : 0;
    const int64_t batch_stride_B = (batch_count > 1) ? nnz : 0;
    const int64_t size_A         = (batch_count - 1) * batch_stride_A + nnz;
    const int64_t size_B         = (batch_count - 1) * batch_stride_B + nnz;

    // Every batch is a differently shuffled and scaled copy of the same matrix, so that a
    // mix up between batches shows up in the result.
    host_dense_vector<I> hA_row(size_A);
    host_dense_vector<I> hA_col(size_A);
    host_dense_vector<T> hA_val(size_A);
    host_dense_vector<I> hB_row_gold(size_B);
    host_dense_vector<I> hB_col_gold(size_B);
    host_dense_vector<T> hB_val_gold(size_B);

    std::fill(hA_row.data(), hA_row.data() + size_A, static_cast<I>(-1));
    std::fill(hA_col.data(), hA_col.data() + size_A, static_cast<I>(-1));
    std::fill(hA_val.data(), hA_val.data() + size_A, static_cast<T>(-1));

    rocsparse_seedrand();
    for(int64_t batch = 0; batch < batch_count; ++batch)
    {
        host_coo_matrix<T, I> hA_batch(hA_single);
        for(int64_t k = 0; k < nnz; ++k)
        {
            hA_batch.val[k] = hA_batch.val[k] * static_cast<T>(batch + 1);
        }

        host_coo_matrix<T, I> hB_batch(hA_batch);
        host_spsort_coo(dir, hB_batch);
        host_shuffle_coo(hA_batch);

        std::copy(hA_batch.row_ind.data(),
                  hA_batch.row_ind.data() + nnz,
                  hA_row.data() + batch * batch_stride_A);
        std::copy(hA_batch.col_ind.data(),
                  hA_batch.col_ind.data() + nnz,
                  hA_col.data() + batch * batch_stride_A);
        std::copy(
            hA_batch.val.data(), hA_batch.val.data() + nnz, hA_val.data() + batch * batch_stride_A);
        std::copy(hB_batch.row_ind.data(),
                  hB_batch.row_ind.data() + nnz,
                  hB_row_gold.data() + batch * batch_stride_B);
        std::copy(hB_batch.col_ind.data(),
                  hB_batch.col_ind.data() + nnz,
                  hB_col_gold.data() + batch * batch_stride_B);
        std::copy(hB_batch.val.data(),
                  hB_batch.val.data() + nnz,
                  hB_val_gold.data() + batch * batch_stride_B);
    }

    // The in place sort keeps the padding of A.
    host_dense_vector<I> hA_row_gold(hA_row);
    host_dense_vector<I> hA_col_gold(hA_col);
    host_dense_vector<T> hA_val_gold(hA_val);
    for(int64_t batch = 0; batch < batch_count; ++batch)
    {
        std::copy(hB_row_gold.data() + batch * batch_stride_B,
                  hB_row_gold.data() + batch * batch_stride_B + nnz,
                  hA_row_gold.data() + batch * batch_stride_A);
        std::copy(hB_col_gold.data() + batch * batch_stride_B,
                  hB_col_gold.data() + batch * batch_stride_B + nnz,
                  hA_col_gold.data() + batch * batch_stride_A);
        std::copy(hB_val_gold.data() + batch * batch_stride_B,
                  hB_val_gold.data() + batch * batch_stride_B + nnz,
                  hA_val_gold.data() + batch * batch_stride_A);
    }

    device_dense_vector<I> dA_row(hA_row);
    device_dense_vector<I> dA_col(hA_col);
    device_dense_vector<T> dA_val(hA_val);
    device_dense_vector<I> dB_row(size_B);
    device_dense_vector<I> dB_col(size_B);
    device_dense_vector<T> dB_val(size_B);

    rocsparse_local_spmat matA(
        M, N, nnz, dA_row, dA_col, dA_val, get_indextype<I>(), base, get_datatype<T>());
    rocsparse_local_spmat matB(
        M, N, nnz, dB_row, dB_col, dB_val, get_indextype<I>(), base, get_datatype<T>());
    CHECK_ROCSPARSE_ERROR(rocsparse_coo_set_strided_batch(matA, batch_count, batch_stride_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_coo_set_strided_batch(matB, batch_count, batch_stride_B));

    rocsparse_spsort_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
    set_spsort_inputs(handle, descr, alg, dir);

    // Analysis
    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
        handle, descr, matA, matB, rocsparse_spsort_stage_analysis, &buffer_size, nullptr));

    void* dbuffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort(
        handle, descr, matA, matB, rocsparse_spsort_stage_analysis, buffer_size, dbuffer, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    dbuffer = nullptr;

    // Compute
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
        handle, descr, matA, matB, rocsparse_spsort_stage_compute, &buffer_size, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Out of place: B holds the sorted matrix and A is left unchanged.
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(handle,
                                               descr,
                                               matA,
                                               matB,
                                               rocsparse_spsort_stage_compute,
                                               buffer_size,
                                               dbuffer,
                                               nullptr));

        hB_row_gold.unit_check(dB_row);
        hB_col_gold.unit_check(dB_col);
        hB_val_gold.unit_check(dB_val);
        hA_row.unit_check(dA_row);
        hA_col.unit_check(dA_col);
        hA_val.unit_check(dA_val);

        // In place: A is sorted into itself.
        size_t in_place_buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(handle,
                                                           descr,
                                                           matA,
                                                           matA,
                                                           rocsparse_spsort_stage_compute,
                                                           &in_place_buffer_size,
                                                           nullptr));
        void* in_place_dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&in_place_dbuffer, in_place_buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(handle,
                                               descr,
                                               matA,
                                               matA,
                                               rocsparse_spsort_stage_compute,
                                               in_place_buffer_size,
                                               in_place_dbuffer,
                                               nullptr));
        CHECK_HIP_ERROR(rocsparse_hipFree(in_place_dbuffer));

        hA_row_gold.unit_check(dA_row);
        hA_col_gold.unit_check(dA_col);
        hA_val_gold.unit_check(dA_val);
    }

    if(arg.timing)
    {
        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_spsort,
                                               handle,
                                               descr,
                                               (rocsparse_const_spmat_descr)matA,
                                               (rocsparse_spmat_descr)matB,
                                               rocsparse_spsort_stage_compute,
                                               buffer_size,
                                               dbuffer,
                                               nullptr);

        const double gbyte_count = batch_count * spsort_coo_gbyte_count<I, T>(nnz);
        const double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::batch_count,
                            batch_count,
                            display_key_t::dir,
                            rocsparse_direction2string(dir),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
}

#define INSTANTIATE(ITYPE, TTYPE)                                                 \
    template void testing_spsort_coo_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spsort_coo<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_spsort_coo_extra(const Arguments& arg) {}
