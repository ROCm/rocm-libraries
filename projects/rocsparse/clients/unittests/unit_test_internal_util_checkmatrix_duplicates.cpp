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

// COO, ELL and HYB matrix checkers: duplicate coordinates, row ordering, an
// ELL failure that the HYB checker must not replace with the COO result, and
// the row batches of the unsorted ELL duplicate pass.
#include "unit_test_utils.hpp"

#include "../../library/src/util/check_matrix_ell_unsorted.hpp"
#include "rocsparse_hyb_mat.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr auto k_ok   = rocsparse_data_status_success;
    constexpr auto k_dup  = rocsparse_data_status_duplicate_entry;
    constexpr auto k_sort = rocsparse_data_status_invalid_sorting;
    constexpr auto k_fill = rocsparse_data_status_invalid_fill;
    constexpr auto k_idx  = rocsparse_data_status_invalid_index;
    constexpr auto k_nan  = rocsparse_data_status_nan;

    template <typename T>
    T nan_of()
    {
        return scalar<T>(std::numeric_limits<float>::quiet_NaN());
    }

    int base_of(rocsparse_index_base base)
    {
        return base == rocsparse_index_base_one ? 1 : 0;
    }

    // Every check in this file gets a temporary buffer of exactly the queried size,
    // followed by k_canary bytes that the check must leave unchanged.
    constexpr size_t        k_canary      = 4096;
    constexpr unsigned char k_canary_byte = 0xA5;

    struct guarded_buffer
    {
        size_t              bytes;
        device_vector<char> mem;

        explicit guarded_buffer(size_t size)
            : bytes(size)
            , mem(size + k_canary)
        {
            EXPECT_EQ(hipMemset(mem.ptr, k_canary_byte, bytes + k_canary), hipSuccess);
        }

        void* ptr() const
        {
            return mem.ptr;
        }

        void expect_canary() const
        {
            const std::vector<char> tail  = to_host(mem.ptr + bytes, k_canary);
            const auto              moved = std::count_if(tail.begin(), tail.end(), [](char c) {
                return static_cast<unsigned char>(c) != k_canary_byte;
            });
            EXPECT_EQ(moved, 0) << "bytes written past the queried buffer size " << bytes;
        }
    };

    template <typename T>
    rocsparse_status coo_buffer(rocsparse_handle       handle,
                                rocsparse_int          m,
                                rocsparse_int          n,
                                rocsparse_int          nnz,
                                const T*               val,
                                const rocsparse_int*   row,
                                const rocsparse_int*   col,
                                rocsparse_index_base   base,
                                rocsparse_matrix_type  type,
                                rocsparse_fill_mode    uplo,
                                rocsparse_storage_mode storage,
                                size_t*                bytes)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scheck_matrix_coo_buffer_size(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, bytes);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcheck_matrix_coo_buffer_size(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, bytes);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccheck_matrix_coo_buffer_size(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, bytes);
        else
            return rocsparse_zcheck_matrix_coo_buffer_size(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, bytes);
    }

    template <typename T>
    rocsparse_status coo_check(rocsparse_handle       handle,
                               rocsparse_int          m,
                               rocsparse_int          n,
                               rocsparse_int          nnz,
                               const T*               val,
                               const rocsparse_int*   row,
                               const rocsparse_int*   col,
                               rocsparse_index_base   base,
                               rocsparse_matrix_type  type,
                               rocsparse_fill_mode    uplo,
                               rocsparse_storage_mode storage,
                               rocsparse_data_status* data,
                               void*                  buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scheck_matrix_coo(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, data, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcheck_matrix_coo(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, data, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccheck_matrix_coo(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, data, buffer);
        else
            return rocsparse_zcheck_matrix_coo(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, data, buffer);
    }

    template <typename T>
    rocsparse_status ell_buffer(rocsparse_handle       handle,
                                rocsparse_int          m,
                                rocsparse_int          n,
                                rocsparse_int          width,
                                const T*               val,
                                const rocsparse_int*   col,
                                rocsparse_index_base   base,
                                rocsparse_matrix_type  type,
                                rocsparse_fill_mode    uplo,
                                rocsparse_storage_mode storage,
                                size_t*                bytes)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scheck_matrix_ell_buffer_size(
                handle, m, n, width, val, col, base, type, uplo, storage, bytes);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcheck_matrix_ell_buffer_size(
                handle, m, n, width, val, col, base, type, uplo, storage, bytes);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccheck_matrix_ell_buffer_size(
                handle, m, n, width, val, col, base, type, uplo, storage, bytes);
        else
            return rocsparse_zcheck_matrix_ell_buffer_size(
                handle, m, n, width, val, col, base, type, uplo, storage, bytes);
    }

    template <typename T>
    rocsparse_status ell_check(rocsparse_handle       handle,
                               rocsparse_int          m,
                               rocsparse_int          n,
                               rocsparse_int          width,
                               const T*               val,
                               const rocsparse_int*   col,
                               rocsparse_index_base   base,
                               rocsparse_matrix_type  type,
                               rocsparse_fill_mode    uplo,
                               rocsparse_storage_mode storage,
                               rocsparse_data_status* data,
                               void*                  buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scheck_matrix_ell(
                handle, m, n, width, val, col, base, type, uplo, storage, data, buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcheck_matrix_ell(
                handle, m, n, width, val, col, base, type, uplo, storage, data, buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccheck_matrix_ell(
                handle, m, n, width, val, col, base, type, uplo, storage, data, buffer);
        else
            return rocsparse_zcheck_matrix_ell(
                handle, m, n, width, val, col, base, type, uplo, storage, data, buffer);
    }

    template <typename T>
    rocsparse_status csr2hyb(rocsparse_handle          handle,
                             rocsparse_int             m,
                             rocsparse_int             n,
                             const rocsparse_mat_descr descr,
                             const T*                  val,
                             const rocsparse_int*      row,
                             const rocsparse_int*      col,
                             rocsparse_hyb_mat         hyb,
                             rocsparse_int             ell_width,
                             rocsparse_hyb_partition   part)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsr2hyb(handle, m, n, descr, val, row, col, hyb, ell_width, part);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsr2hyb(handle, m, n, descr, val, row, col, hyb, ell_width, part);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsr2hyb(handle, m, n, descr, val, row, col, hyb, ell_width, part);
        else
            return rocsparse_zcsr2hyb(handle, m, n, descr, val, row, col, hyb, ell_width, part);
    }

    struct spmat_guard
    {
        rocsparse_spmat_descr p = nullptr;
        ~spmat_guard()
        {
            if(p)
                rocsparse_destroy_spmat_descr(p);
        }
    };

    struct descr_guard
    {
        rocsparse_mat_descr p = nullptr;
        descr_guard()
        {
            EXPECT_EQ(rocsparse_create_mat_descr(&p), rocsparse_status_success);
        }
        ~descr_guard()
        {
            if(p)
                rocsparse_destroy_mat_descr(p);
        }
    };

    struct hyb_guard
    {
        rocsparse_hyb_mat p = nullptr;
        hyb_guard()
        {
            EXPECT_EQ(rocsparse_create_hyb_mat(&p), rocsparse_status_success);
        }
        ~hyb_guard()
        {
            if(p)
                rocsparse_destroy_hyb_mat(p);
        }
    };

    void set_spmat(rocsparse_spmat_descr  descr,
                   rocsparse_matrix_type  type,
                   rocsparse_fill_mode    uplo,
                   rocsparse_storage_mode storage)
    {
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(descr, rocsparse_spmat_matrix_type, &type, sizeof(type)),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(descr, rocsparse_spmat_fill_mode, &uplo, sizeof(uplo)),
            rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmat_set_attribute(
                      descr, rocsparse_spmat_storage_mode, &storage, sizeof(storage)),
                  rocsparse_status_success);
    }

    rocsparse_data_status run_spmat(rocsparse_handle       handle,
                                    rocsparse_spmat_descr  descr,
                                    rocsparse_matrix_type  type,
                                    rocsparse_fill_mode    uplo,
                                    rocsparse_storage_mode storage)
    {
        set_spmat(descr, type, uplo, storage);
        rocsparse_data_status data  = k_fill;
        size_t                bytes = 0;
        const auto            query = rocsparse_check_spmat(
            handle, descr, &data, rocsparse_check_spmat_stage_buffer_size, &bytes, nullptr);
        EXPECT_EQ(query, rocsparse_status_success);
        if(query != rocsparse_status_success)
            return k_fill;
        guarded_buffer work(bytes);
        data             = k_fill;
        const auto check = rocsparse_check_spmat(
            handle, descr, &data, rocsparse_check_spmat_stage_compute, &bytes, work.ptr());
        EXPECT_EQ(check, rocsparse_status_success);
        work.expect_canary();
        if(check != rocsparse_status_success)
            return k_fill;
        return data;
    }

    template <typename T>
    T* upload(const std::vector<T>& host)
    {
        if(host.empty())
            return nullptr;
        T* p = nullptr;
        EXPECT_EQ(hipMalloc(&p, host.size() * sizeof(T)), hipSuccess);
        if(p == nullptr)
            return nullptr;
        EXPECT_EQ(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
        return p;
    }

    struct device_free
    {
        void* p = nullptr;
        ~device_free()
        {
            if(p)
            {
                (void)hipFree(p);
            }
        }
    };

    template <typename T>
    void expect_coo_public(rocsparse_handle                  handle,
                           rocsparse_int                     m,
                           rocsparse_int                     n,
                           const std::vector<rocsparse_int>& rows,
                           const std::vector<rocsparse_int>& cols,
                           rocsparse_index_base              base,
                           rocsparse_matrix_type             type,
                           rocsparse_fill_mode               uplo,
                           rocsparse_storage_mode            storage,
                           rocsparse_data_status             expected)
    {
        const rocsparse_int  nnz = static_cast<rocsparse_int>(rows.size());
        std::vector<T>       vals(static_cast<size_t>(nnz), scalar<T>(1));
        device_free          row_mem{upload(rows)};
        device_free          col_mem{upload(cols)};
        device_free          val_mem{upload(vals)};
        const T*             val   = static_cast<const T*>(val_mem.p);
        const rocsparse_int* row   = static_cast<const rocsparse_int*>(row_mem.p);
        const rocsparse_int* col   = static_cast<const rocsparse_int*>(col_mem.p);
        size_t               bytes = 0;
        ASSERT_EQ(
            coo_buffer<T>(handle, m, n, nnz, val, row, col, base, type, uplo, storage, &bytes),
            rocsparse_status_success);
        guarded_buffer        work(bytes);
        rocsparse_data_status data = k_fill;
        ASSERT_EQ(
            coo_check<T>(
                handle, m, n, nnz, val, row, col, base, type, uplo, storage, &data, work.ptr()),
            rocsparse_status_success);
        EXPECT_EQ(data, expected);
        work.expect_canary();
    }

    template <typename T, typename I>
    void expect_coo_spmat(rocsparse_handle       handle,
                          int64_t                m,
                          int64_t                n,
                          const std::vector<I>&  rows,
                          const std::vector<I>&  cols,
                          rocsparse_index_base   base,
                          rocsparse_matrix_type  type,
                          rocsparse_fill_mode    uplo,
                          rocsparse_storage_mode storage,
                          rocsparse_data_status  expected)
    {
        std::vector<T> vals(rows.size(), scalar<T>(1));
        // create_coo_descr rejects null index pointers, including nnz = 0.
        std::vector<I>   row_host = rows.empty() ? std::vector<I>{I(0)} : rows;
        std::vector<I>   col_host = cols.empty() ? std::vector<I>{I(0)} : cols;
        std::vector<T>   val_host = vals.empty() ? std::vector<T>{scalar<T>(0)} : vals;
        device_vector<I> drow(row_host);
        device_vector<I> dcol(col_host);
        device_vector<T> dval(val_host);
        spmat_guard      descr;
        ASSERT_EQ(rocsparse_create_coo_descr(&descr.p,
                                             m,
                                             n,
                                             static_cast<int64_t>(rows.size()),
                                             drow.ptr,
                                             dcol.ptr,
                                             dval.ptr,
                                             it_of<I>(),
                                             base,
                                             dt_of<T>()),
                  rocsparse_status_success);
        EXPECT_EQ(run_spmat(handle, descr.p, type, uplo, storage), expected);
    }

    std::vector<rocsparse_int>
        shift_base(const std::vector<int>& zero_based, int base, bool allow_negative)
    {
        std::vector<rocsparse_int> out(zero_based.size());
        for(size_t i = 0; i < zero_based.size(); ++i)
        {
            const int v = zero_based[i];
            out[i]      = (v < 0 && allow_negative) ? static_cast<rocsparse_int>(v)
                                                    : static_cast<rocsparse_int>(v + base);
        }
        return out;
    }

    struct coo_case
    {
        const char*           name;
        std::vector<int>      rows;
        std::vector<int>      cols;
        rocsparse_data_status sorted;
        rocsparse_data_status unsorted;
        bool                  general_only;
    };

    // Sorted storage is ordered by row, then by column, with no repeated coordinate.
    // Unsorted storage has no order, so a duplicate pair can be separated by entries
    // in the same row or in the same column.
    const std::vector<coo_case>& general_coo_cases()
    {
        static const std::vector<coo_case> cases = {
            {"valid", {0, 0, 1}, {0, 1, 1}, k_ok, k_ok, true},
            {"duplicate", {0, 0, 1}, {0, 0, 1}, k_dup, k_dup, true},
            {"duplicate_last", {0, 1, 1}, {0, 1, 1}, k_dup, k_dup, true},
            {"descending_row", {1, 0, 0}, {1, 0, 1}, k_sort, k_ok, true},
            {"descending_col", {0, 0, 1}, {1, 0, 1}, k_sort, k_ok, true},
            {"duplicate_unsorted", {0, 1, 0}, {0, 1, 0}, k_sort, k_dup, true},
            {"duplicate_row_separated", {0, 0, 0}, {0, 1, 0}, k_sort, k_dup, true},
            {"duplicate_col_separated", {0, 1, 0}, {0, 0, 0}, k_sort, k_dup, true},
            {"invalid_index", {-1, 0, 1}, {0, 1, 1}, k_idx, k_idx, true},
        };
        return cases;
    }

    std::vector<coo_case> triangle_coo_cases(rocsparse_fill_mode uplo)
    {
        if(uplo == rocsparse_fill_mode_lower)
        {
            return {{"valid", {0, 1, 1}, {0, 0, 1}, k_ok, k_ok, false},
                    {"duplicate", {1, 1, 1}, {0, 0, 1}, k_dup, k_dup, false},
                    {"duplicate_last", {0, 1, 1}, {0, 0, 0}, k_dup, k_dup, false},
                    {"descending_row", {1, 0}, {0, 0}, k_sort, k_ok, false},
                    {"descending_col", {1, 1}, {1, 0}, k_sort, k_ok, false},
                    {"duplicate_separated", {1, 0, 1}, {0, 0, 0}, k_sort, k_dup, false},
                    {"duplicate_row_separated", {1, 1, 1}, {0, 1, 0}, k_sort, k_dup, false},
                    {"wrong_triangle", {0}, {1}, k_fill, k_fill, false}};
        }
        return {{"valid", {0, 0, 1}, {0, 1, 1}, k_ok, k_ok, false},
                {"duplicate", {0, 0, 1}, {1, 1, 1}, k_dup, k_dup, false},
                {"duplicate_last", {0, 0, 0}, {0, 1, 1}, k_dup, k_dup, false},
                {"descending_row", {1, 0}, {1, 1}, k_sort, k_ok, false},
                {"descending_col", {0, 0}, {1, 0}, k_sort, k_ok, false},
                {"duplicate_separated", {0, 1, 0}, {1, 1, 1}, k_sort, k_dup, false},
                {"duplicate_row_separated", {0, 0, 0}, {0, 1, 0}, k_sort, k_dup, false},
                {"wrong_triangle", {1}, {0}, k_fill, k_fill, false}};
    }

    template <typename T>
    void run_coo_matrix(rocsparse_handle handle, const char* type_name, bool with_spmat_i64)
    {
        const rocsparse_matrix_type types[] = {rocsparse_matrix_type_general,
                                               rocsparse_matrix_type_symmetric,
                                               rocsparse_matrix_type_hermitian,
                                               rocsparse_matrix_type_triangular};
        const rocsparse_fill_mode  uplos[] = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};
        const rocsparse_index_base bases[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
        const rocsparse_storage_mode storages[]
            = {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted};

        for(auto type : types)
            for(auto uplo : uplos)
                for(auto base : bases)
                    for(auto storage : storages)
                    {
                        const int                   b       = base_of(base);
                        const bool                  general = type == rocsparse_matrix_type_general;
                        const std::vector<coo_case> cases
                            = general ? std::vector<coo_case>(general_coo_cases())
                                      : triangle_coo_cases(uplo);
                        for(const auto& item : cases)
                        {
                            if(item.general_only && !general)
                                continue;
                            if(!item.general_only && general)
                                continue;
                            SCOPED_TRACE(std::string(type_name) + " " + item.name + " type "
                                         + std::to_string(type) + " uplo " + std::to_string(uplo)
                                         + " base " + std::to_string(b) + " storage "
                                         + std::to_string(storage));
                            const auto rows     = shift_base(item.rows, b, true);
                            const auto cols     = shift_base(item.cols, b, true);
                            const auto expected = storage == rocsparse_storage_mode_sorted
                                                      ? item.sorted
                                                      : item.unsorted;
                            expect_coo_public<T>(
                                handle, 2, 2, rows, cols, base, type, uplo, storage, expected);
                            if(with_spmat_i64)
                            {
                                std::vector<int64_t> r64(rows.begin(), rows.end());
                                std::vector<int64_t> c64(cols.begin(), cols.end());
                                expect_coo_spmat<T, int64_t>(
                                    handle, 2, 2, r64, c64, base, type, uplo, storage, expected);
                            }
                            else
                            {
                                std::vector<int32_t> r32(rows.begin(), rows.end());
                                std::vector<int32_t> c32(cols.begin(), cols.end());
                                expect_coo_spmat<T, int32_t>(
                                    handle, 2, 2, r32, c32, base, type, uplo, storage, expected);
                            }
                        }
                    }
    }

    template <typename T>
    void expect_ell_public(rocsparse_handle                  handle,
                           rocsparse_int                     m,
                           rocsparse_int                     n,
                           rocsparse_int                     width,
                           const std::vector<rocsparse_int>& cols,
                           const std::vector<T>&             vals,
                           rocsparse_index_base              base,
                           rocsparse_storage_mode            storage,
                           rocsparse_data_status             expected)
    {
        device_free          col_mem{upload(cols)};
        device_free          val_mem{upload(vals)};
        const T*             dval  = static_cast<const T*>(val_mem.p);
        const rocsparse_int* dcol  = static_cast<const rocsparse_int*>(col_mem.p);
        size_t               bytes = 0;
        ASSERT_EQ(ell_buffer<T>(handle,
                                m,
                                n,
                                width,
                                dval,
                                dcol,
                                base,
                                rocsparse_matrix_type_general,
                                rocsparse_fill_mode_lower,
                                storage,
                                &bytes),
                  rocsparse_status_success);
        guarded_buffer        work(bytes);
        rocsparse_data_status data = k_fill;
        ASSERT_EQ(ell_check<T>(handle,
                               m,
                               n,
                               width,
                               dval,
                               dcol,
                               base,
                               rocsparse_matrix_type_general,
                               rocsparse_fill_mode_lower,
                               storage,
                               &data,
                               work.ptr()),
                  rocsparse_status_success);
        EXPECT_EQ(data, expected);
        work.expect_canary();
    }

    // Column-major ELL. `rows[row]` lists that row's columns in slot order.
    // A negative entry is padding and stays -1 for both index bases.
    template <typename I>
    void ell_pack(const std::vector<std::vector<int>>& rows,
                  int                                  width,
                  int                                  base,
                  std::vector<I>&                      cols,
                  std::vector<float>&                  vals_unused)
    {
        (void)vals_unused;
        const int64_t m = static_cast<int64_t>(rows.size());
        cols.assign(static_cast<size_t>(m * width), static_cast<I>(-1));
        for(int64_t row = 0; row < m; ++row)
            for(int j = 0; j < static_cast<int>(rows[row].size()) && j < width; ++j)
            {
                const int c = rows[row][j];
                cols[static_cast<size_t>(m * j + row)]
                    = c < 0 ? static_cast<I>(-1) : static_cast<I>(c + base);
            }
    }

    template <typename T, typename I>
    void expect_ell(rocsparse_handle                     handle,
                    const char*                          type_name,
                    const char*                          name,
                    int                                  m,
                    int                                  n,
                    int                                  width,
                    const std::vector<std::vector<int>>& rows,
                    rocsparse_index_base                 base,
                    rocsparse_storage_mode               storage,
                    rocsparse_data_status                expected,
                    bool                                 spmat)
    {
        SCOPED_TRACE(std::string(type_name) + " " + name + " base " + std::to_string(base_of(base))
                     + " storage " + std::to_string(storage) + (spmat ? " spmat" : " api"));
        std::vector<I>     cols;
        std::vector<float> ignore;
        ell_pack<I>(rows, width, base_of(base), cols, ignore);
        std::vector<T> vals(cols.size(), scalar<T>(1));
        if(!spmat)
        {
            std::vector<rocsparse_int> cols32(cols.begin(), cols.end());
            expect_ell_public<T>(handle, m, n, width, cols32, vals, base, storage, expected);
            return;
        }
        std::vector<I>   col_host = cols.empty() ? std::vector<I>{I(0)} : cols;
        std::vector<T>   val_host = vals.empty() ? std::vector<T>{scalar<T>(0)} : vals;
        device_vector<I> dcol(col_host);
        device_vector<T> dval(val_host);
        spmat_guard      descr;
        ASSERT_EQ(rocsparse_create_ell_descr(
                      &descr.p, m, n, dcol.ptr, dval.ptr, width, it_of<I>(), base, dt_of<T>()),
                  rocsparse_status_success);
        EXPECT_EQ(
            run_spmat(
                handle, descr.p, rocsparse_matrix_type_general, rocsparse_fill_mode_lower, storage),
            expected);
    }

    template <typename T>
    void run_ell(rocsparse_handle handle, const char* type_name, bool i64)
    {
        const rocsparse_index_base bases[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
        const rocsparse_storage_mode storages[]
            = {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted};
        struct ell_case
        {
            const char*                   name;
            int                           n;
            int                           width;
            std::vector<std::vector<int>> rows;
            rocsparse_data_status         sorted;
            rocsparse_data_status         unsorted;
        };
        // A repeated column is a duplicate entry in both storage modes; in unsorted
        // storage the pair need not be adjacent. Every slot from the first -1 of a row
        // onward is padding, whatever it holds.
        const std::vector<ell_case> cases = {
            {"valid", 2, 2, {{0, 1}, {0, 1}}, k_ok, k_ok},
            {"duplicate_adjacent", 2, 2, {{0, 0}, {0, 1}}, k_dup, k_dup},
            {"duplicate_separated", 4, 3, {{0, 1, 0}, {1, 2, 3}}, k_sort, k_dup},
            {"decreasing_unique", 4, 3, {{2, 0, 1}, {3, 1, 0}}, k_sort, k_ok},
            {"padded", 4, 3, {{0}, {1}}, k_ok, k_ok},
            {"repeat_after_padding", 4, 3, {{2, -1, 2}, {1, -1, -1}}, k_ok, k_ok},
            {"repeats_after_padding", 5, 5, {{0, 1, -1, 1, 1}, {3}}, k_ok, k_ok},
            {"duplicate_before_padding", 4, 3, {{1, 1, -1}, {0}}, k_dup, k_dup},
            {"separated_duplicate_before_padding", 5, 5, {{2, 1, 2, -1, 1}, {0}}, k_sort, k_dup},
            {"invalid_index", 2, 2, {{0, 5}, {0, 1}}, k_idx, k_idx},
        };
        for(const auto& item : cases)
            for(auto base : bases)
                for(auto storage : storages)
                {
                    const auto expected
                        = storage == rocsparse_storage_mode_sorted ? item.sorted : item.unsorted;
                    const int m = static_cast<int>(item.rows.size());
                    if(i64)
                        expect_ell<T, int64_t>(handle,
                                               type_name,
                                               item.name,
                                               m,
                                               item.n,
                                               item.width,
                                               item.rows,
                                               base,
                                               storage,
                                               expected,
                                               true);
                    else
                    {
                        expect_ell<T, int32_t>(handle,
                                               type_name,
                                               item.name,
                                               m,
                                               item.n,
                                               item.width,
                                               item.rows,
                                               base,
                                               storage,
                                               expected,
                                               false);
                        expect_ell<T, int32_t>(handle,
                                               type_name,
                                               item.name,
                                               m,
                                               item.n,
                                               item.width,
                                               item.rows,
                                               base,
                                               storage,
                                               expected,
                                               true);
                    }
                }
    }

    // An ELL slot `index` (column-major) of the converted HYB matrix is set to `col`.
    struct ell_poke
    {
        int64_t index;
        int     col;
    };

    template <typename T>
    void expect_hyb(rocsparse_handle             handle,
                    const char*                  type_name,
                    const char*                  name,
                    rocsparse_index_base         base,
                    rocsparse_storage_mode       storage,
                    rocsparse_int                n,
                    const std::vector<int>&      row_ptr,
                    const std::vector<int>&      col_ind,
                    const std::vector<T>&        vals,
                    rocsparse_int                ell_width,
                    rocsparse_hyb_partition      part,
                    rocsparse_data_status        expected,
                    const std::vector<ell_poke>& pokes = {})
    {
        SCOPED_TRACE(std::string(type_name) + " " + name + " base " + std::to_string(base_of(base))
                     + " storage " + std::to_string(storage));
        const int                  b = base_of(base);
        const rocsparse_int        m = static_cast<rocsparse_int>(row_ptr.size() - 1);
        std::vector<rocsparse_int> hrow(row_ptr.size());
        std::vector<rocsparse_int> hcol(col_ind.size());
        for(size_t i = 0; i < row_ptr.size(); ++i)
            hrow[i] = row_ptr[i] + b;
        for(size_t i = 0; i < col_ind.size(); ++i)
            hcol[i] = col_ind[i] + b;
        device_vector<rocsparse_int> drow(hrow);
        device_vector<rocsparse_int> dcol(hcol);
        device_vector<T>             dval(vals);
        descr_guard                  descr;
        hyb_guard                    hyb;
        ASSERT_EQ(descr.p == nullptr, false);
        ASSERT_EQ(hyb.p == nullptr, false);
        ASSERT_EQ(rocsparse_set_mat_index_base(descr.p, base), rocsparse_status_success);
        ASSERT_EQ(
            csr2hyb<T>(handle, m, n, descr.p, dval.ptr, drow.ptr, dcol.ptr, hyb.p, ell_width, part),
            rocsparse_status_success);
        for(const auto& poke : pokes)
        {
            ASSERT_LT(poke.index, static_cast<int64_t>(hyb.p->m) * hyb.p->ell_width);
            const rocsparse_int value
                = poke.col < 0 ? rocsparse_int(-1) : static_cast<rocsparse_int>(poke.col + b);
            UT_CHECK_HIP(hipMemcpy(
                hyb.p->ell_col_ind + poke.index, &value, sizeof(value), hipMemcpyHostToDevice));
        }
        size_t bytes = 0;
        ASSERT_EQ(rocsparse_check_matrix_hyb_buffer_size(handle,
                                                         hyb.p,
                                                         base,
                                                         rocsparse_matrix_type_general,
                                                         rocsparse_fill_mode_lower,
                                                         storage,
                                                         &bytes),
                  rocsparse_status_success);
        guarded_buffer        work(bytes);
        rocsparse_data_status data = k_fill;
        ASSERT_EQ(rocsparse_check_matrix_hyb(handle,
                                             hyb.p,
                                             base,
                                             rocsparse_matrix_type_general,
                                             rocsparse_fill_mode_lower,
                                             storage,
                                             &data,
                                             work.ptr()),
                  rocsparse_status_success);
        EXPECT_EQ(data, expected);
        work.expect_canary();
    }

    template <typename T>
    void run_hyb(rocsparse_handle handle, const char* type_name)
    {
        const T                    one     = scalar<T>(1);
        const T                    nan     = nan_of<T>();
        const auto                 user    = rocsparse_hyb_partition_user;
        const rocsparse_index_base bases[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
        const rocsparse_storage_mode storages[]
            = {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted};
        for(auto base : bases)
            for(auto storage : storages)
            {
                const bool sorted   = storage == rocsparse_storage_mode_sorted;
                const auto separate = sorted ? k_sort : k_dup;
                // clang-format off
                expect_hyb<T>(handle, type_name, "finite_ell_only", base, storage, 2,
                              {0, 1, 2}, {0, 1}, {one, one},
                              0, rocsparse_hyb_partition_max, k_ok);
                expect_hyb<T>(handle, type_name, "nan_ell_only", base, storage, 2,
                              {0, 1, 2}, {0, 1}, {nan, one},
                              0, rocsparse_hyb_partition_max, k_nan);
                expect_hyb<T>(handle, type_name, "finite_coo_tail", base, storage, 2,
                              {0, 2, 3}, {0, 1, 1}, {one, one, one},
                              1, user, k_ok);
                expect_hyb<T>(handle, type_name, "nan_ell_valid_coo", base, storage, 2,
                              {0, 2, 3}, {0, 1, 1}, {nan, one, one},
                              1, user, k_nan);
                expect_hyb<T>(handle, type_name, "valid_ell_nan_coo", base, storage, 2,
                              {0, 2, 3}, {0, 1, 1}, {one, nan, one},
                              1, user, k_nan);
                expect_hyb<T>(handle, type_name, "nan_ell_bad_coo", base, storage, 2,
                              {0, 2, 3}, {0, 5, 1}, {nan, one, one},
                              1, user, k_nan);
                // An ELL failure is reported without checking the COO part.
                expect_hyb<T>(handle, type_name, "bad_ell_nan_coo", base, storage, 2,
                              {0, 2, 3}, {0, 1, 1}, {one, nan, one},
                              1, user, k_idx, {{0, 5}});
                expect_hyb<T>(handle, type_name, "nan_ell_separated_duplicate_coo_tail", base,
                              storage, 4,
                              {0, 4, 5}, {0, 1, 2, 1, 1}, {nan, one, one, one, one},
                              1, user, k_nan);
                expect_hyb<T>(handle, type_name, "bad_ell_separated_duplicate_coo_tail", base,
                              storage, 4,
                              {0, 4, 5}, {0, 1, 2, 1, 1}, {one, one, one, one, one},
                              1, user, k_idx, {{1, 9}});
                expect_hyb<T>(handle, type_name, "duplicate_ell", base, storage, 2,
                              {0, 2, 3}, {0, 0, 1}, {one, one, one},
                              2, user, k_dup);

                // Every ELL slot from the first -1 of a row onward is padding.
                expect_hyb<T>(handle, type_name, "ell_repeat_after_padding", base, storage, 4,
                              {0, 1, 4}, {2, 0, 1, 3}, {one, one, one, one},
                              3, user, k_ok, {{4, 2}});
                expect_hyb<T>(handle, type_name, "ell_duplicate_before_padding", base, storage, 4,
                              {0, 2, 3}, {1, 1, 0}, {one, one, one},
                              3, user, k_dup);

                // Rows 0 and 1 keep their first entry in the ELL part, the rest is the COO part.
                expect_hyb<T>(handle, type_name, "valid_coo_tail", base, storage, 4,
                              {0, 3, 5}, {0, 1, 2, 0, 3}, {one, one, one, one, one},
                              1, user, k_ok);
                expect_hyb<T>(handle, type_name, "duplicate_coo_tail", base, storage, 4,
                              {0, 3, 4}, {0, 1, 1, 1}, {one, one, one, one},
                              1, user, k_dup);
                expect_hyb<T>(handle, type_name, "separated_duplicate_coo_tail", base, storage, 4,
                              {0, 4, 5}, {0, 1, 2, 1, 1}, {one, one, one, one, one},
                              1, user, separate);

                // The ELL and COO parts are checked separately, so a coordinate stored
                // once in each part is not reported.
                expect_hyb<T>(handle, type_name, "ell_coo_cross_duplicate", base, storage, 4,
                              {0, 2, 3}, {0, 0, 1}, {one, one, one},
                              1, user, k_ok);
                // clang-format on
            }
    }

    struct coo_coords
    {
        std::vector<int64_t> rows;
        std::vector<int64_t> cols;
    };

    constexpr int64_t k_scale_dim = 1000;
    constexpr int64_t k_scale_nnz = 100000;

    // Distinct coordinates of a k_scale_dim x k_scale_dim matrix, 100 per row, ordered by
    // row, then by column. Every column index is a multiple of 10.
    coo_coords scale_coo_sorted()
    {
        coo_coords c;
        c.rows.resize(k_scale_nnz);
        c.cols.resize(k_scale_nnz);
        for(int64_t k = 0; k < k_scale_nnz; ++k)
        {
            c.rows[k] = k / 100;
            c.cols[k] = (k % 100) * 10;
        }
        return c;
    }

    coo_coords shuffled(const coo_coords& in)
    {
        std::vector<size_t> perm(in.rows.size());
        std::iota(perm.begin(), perm.end(), size_t{0});
        std::mt19937 gen(1094);
        std::shuffle(perm.begin(), perm.end(), gen);
        coo_coords out = in;
        for(size_t k = 0; k < perm.size(); ++k)
        {
            out.rows[k] = in.rows[perm[k]];
            out.cols[k] = in.cols[perm[k]];
        }
        return out;
    }

    coo_coords with_copy(coo_coords c, int64_t from, int64_t to)
    {
        c.rows[to] = c.rows[from];
        c.cols[to] = c.cols[from];
        return c;
    }

    template <typename I>
    std::vector<I> as_index(const std::vector<int64_t>& zero_based, int base)
    {
        std::vector<I> out(zero_based.size());
        for(size_t k = 0; k < zero_based.size(); ++k)
            out[k] = static_cast<I>(zero_based[k] + base);
        return out;
    }

    // The public 32-bit API and rocsparse_check_spmat with 32-bit and 64-bit indices.
    void expect_coo_scale(rocsparse_handle       handle,
                          const char*            name,
                          const coo_coords&      c,
                          rocsparse_index_base   base,
                          rocsparse_storage_mode storage,
                          rocsparse_data_status  expected)
    {
        SCOPED_TRACE(std::string("coo scale ") + name + " base " + std::to_string(base_of(base))
                     + " storage " + std::to_string(storage));
        const int  b       = base_of(base);
        const auto general = rocsparse_matrix_type_general;
        const auto lower   = rocsparse_fill_mode_lower;
        expect_coo_public<float>(handle,
                                 k_scale_dim,
                                 k_scale_dim,
                                 as_index<rocsparse_int>(c.rows, b),
                                 as_index<rocsparse_int>(c.cols, b),
                                 base,
                                 general,
                                 lower,
                                 storage,
                                 expected);
        expect_coo_spmat<float, int32_t>(handle,
                                         k_scale_dim,
                                         k_scale_dim,
                                         as_index<int32_t>(c.rows, b),
                                         as_index<int32_t>(c.cols, b),
                                         base,
                                         general,
                                         lower,
                                         storage,
                                         expected);
        expect_coo_spmat<float, int64_t>(handle,
                                         k_scale_dim,
                                         k_scale_dim,
                                         as_index<int64_t>(c.rows, b),
                                         as_index<int64_t>(c.cols, b),
                                         base,
                                         general,
                                         lower,
                                         storage,
                                         expected);
    }

    constexpr int k_ell_scale_m     = 1000;
    constexpr int k_ell_scale_n     = 64;
    constexpr int k_ell_scale_width = 8;

    // Distinct columns in every row, in an order that is not increasing.
    std::vector<std::vector<int>> scale_ell_unsorted()
    {
        std::vector<std::vector<int>> rows(k_ell_scale_m, std::vector<int>(k_ell_scale_width));
        for(int r = 0; r < k_ell_scale_m; ++r)
            for(int j = 0; j < k_ell_scale_width; ++j)
                rows[r][j] = (r * 7 + j * 11) % k_ell_scale_n;
        return rows;
    }

    std::vector<std::vector<int>> scale_ell_sorted()
    {
        auto rows = scale_ell_unsorted();
        for(auto& row : rows)
            std::sort(row.begin(), row.end());
        return rows;
    }

    void expect_ell_scale(rocsparse_handle                     handle,
                          const char*                          name,
                          const std::vector<std::vector<int>>& rows,
                          rocsparse_index_base                 base,
                          rocsparse_storage_mode               storage,
                          rocsparse_data_status                expected)
    {
        for(bool spmat : {false, true})
            expect_ell<float, int32_t>(handle,
                                       "s",
                                       name,
                                       k_ell_scale_m,
                                       k_ell_scale_n,
                                       k_ell_scale_width,
                                       rows,
                                       base,
                                       storage,
                                       expected,
                                       spmat);
        expect_ell<float, int64_t>(handle,
                                   "s64",
                                   name,
                                   k_ell_scale_m,
                                   k_ell_scale_n,
                                   k_ell_scale_width,
                                   rows,
                                   base,
                                   storage,
                                   expected,
                                   true);
    }

    // CSR rows of 8 to 12 increasing columns; with an ELL width of 8 the remainder of a
    // row goes to the COO part.
    void scale_hyb_csr(bool duplicate, std::vector<int>& row_ptr, std::vector<int>& col_ind)
    {
        row_ptr.assign(1, 0);
        col_ind.clear();
        for(int r = 0; r < k_ell_scale_m; ++r)
        {
            const int count = 8 + r % 5;
            for(int j = 0; j < count; ++j)
                col_ind.push_back(j * 5);
            row_ptr.push_back(static_cast<int>(col_ind.size()));
        }
        if(duplicate)
        {
            // Row 504 has 12 entries; its last two are in the COO part.
            const int end    = row_ptr[505];
            col_ind[end - 1] = col_ind[end - 2];
        }
    }

    void expect_hyb_scale(rocsparse_handle       handle,
                          const char*            name,
                          bool                   duplicate,
                          rocsparse_index_base   base,
                          rocsparse_storage_mode storage,
                          rocsparse_data_status  expected)
    {
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        scale_hyb_csr(duplicate, row_ptr, col_ind);
        const std::vector<float> vals(col_ind.size(), 1.0f);
        expect_hyb<float>(handle,
                          "s",
                          name,
                          base,
                          storage,
                          k_ell_scale_n,
                          row_ptr,
                          col_ind,
                          vals,
                          k_ell_scale_width,
                          rocsparse_hyb_partition_user,
                          expected);
    }

    // The duplicate pass of the unsorted ELL check on its own, sorting batches of at most
    // batch_limit entries. Returns the status it reports.
    template <typename I>
    rocsparse_status run_ell_duplicate_pass(rocsparse_handle                     handle,
                                            I                                    n,
                                            I                                    width,
                                            const std::vector<std::vector<int>>& rows,
                                            rocsparse_index_base                 base,
                                            int64_t                              batch_limit,
                                            rocsparse_data_status&               data)
    {
        const I            m = static_cast<I>(rows.size());
        std::vector<I>     cols;
        std::vector<float> ignore;
        ell_pack<I>(rows, static_cast<int>(width), base_of(base), cols, ignore);
        device_vector<I> dcol(cols);

        rocsparse::check_matrix_ell_workspace ws;
        const rocsparse_status query = rocsparse::check_matrix_ell_workspace_query<I>(
            handle, m, n, width, rocsparse_storage_mode_unsorted, ws, batch_limit);
        if(query != rocsparse_status_success)
            return query;

        guarded_buffer work(ws.total);
        EXPECT_EQ(hipMemset(work.ptr(), 0, ws.status_bytes), hipSuccess);
        data                         = k_fill;
        const rocsparse_status check = rocsparse::check_matrix_ell_unsorted_duplicates<I>(
            handle, m, n, width, dcol.ptr, base, &data, work.ptr(), batch_limit);
        work.expect_canary();
        return check;
    }

    // Rows of 4 distinct columns that are not in increasing order. Each full row ends with
    // the column that the next row starts with, so a sort segment that crosses the boundary
    // between two full rows holds a repeated column. Row 5 ends with padding.
    std::vector<std::vector<int>> batch_ell_rows()
    {
        std::vector<std::vector<int>> rows;
        for(int r = 0; r < 7; ++r)
            rows.push_back({r, r + 2, r + 3, r + 1});
        rows[5][3] = -1;
        return rows;
    }

    template <typename I>
    void expect_ell_batches(rocsparse_handle handle, const char* type_name)
    {
        const I    n     = 16;
        const I    width = 4;
        const auto valid = batch_ell_rows();
        const int  m     = static_cast<int>(valid.size());

        // name, rows, expected
        std::vector<std::tuple<std::string, std::vector<std::vector<int>>, rocsparse_data_status>>
            cases;
        cases.emplace_back("valid", valid, k_ok);
        for(int r = 0; r < m; ++r)
        {
            auto rows  = valid;
            rows[r][2] = rows[r][0];
            cases.emplace_back("duplicate_row_" + std::to_string(r), rows, k_dup);
        }
        // With 2 rows per batch, rows 3 and 4 are on either side of a batch boundary.
        auto both  = valid;
        both[3][2] = both[3][0];
        both[4][2] = both[4][0];
        cases.emplace_back("duplicate_rows_3_4", both, k_dup);

        // Batches of 1, 2, 3 and 6 rows, with and without unused entries below the limit,
        // one batch of all rows, and the library default.
        const int64_t limits[]
            = {4, 5, 8, 11, 12, 24, 27, 28, rocsparse::check_matrix_ell_sort_batch_limit};

        for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            for(const auto& item : cases)
                for(int64_t limit : limits)
                {
                    SCOPED_TRACE(std::string(type_name) + " " + std::get<0>(item) + " base "
                                 + std::to_string(base_of(base)) + " limit "
                                 + std::to_string(limit));
                    rocsparse_data_status data = k_fill;
                    EXPECT_EQ(run_ell_duplicate_pass<I>(
                                  handle, n, width, std::get<1>(item), base, limit, data),
                              rocsparse_status_success);
                    EXPECT_EQ(data, std::get<2>(item));
                }

        // A limit below the ELL width cannot hold one row.
        rocsparse_data_status data = k_fill;
        EXPECT_EQ(run_ell_duplicate_pass<I>(
                      handle, n, width, valid, rocsparse_index_base_zero, width - 1, data),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(rocsparse::check_matrix_ell_unsorted_duplicates<I>(handle,
                                                                     I(m),
                                                                     n,
                                                                     width,
                                                                     nullptr,
                                                                     rocsparse_index_base_zero,
                                                                     &data,
                                                                     nullptr,
                                                                     width - 1),
                  rocsparse_status_not_implemented);
    }

    // The unsorted COO duplicate pass sorts keys of as many bits as max(m, n) - 1 + idx_base
    // needs, so a wide or a tall matrix has keys above the bits of its short side.
    void expect_coo_shape(rocsparse_handle            handle,
                          const char*                 name,
                          int64_t                     m,
                          int64_t                     n,
                          const std::vector<int64_t>& rows,
                          const std::vector<int64_t>& cols,
                          rocsparse_data_status       sorted,
                          rocsparse_data_status       unsorted)
    {
        const auto general = rocsparse_matrix_type_general;
        const auto lower   = rocsparse_fill_mode_lower;
        for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            for(auto storage : {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
            {
                SCOPED_TRACE(std::string("coo shape ") + name + " base "
                             + std::to_string(base_of(base)) + " storage "
                             + std::to_string(storage));
                const int  b        = base_of(base);
                const auto expected = storage == rocsparse_storage_mode_sorted ? sorted : unsorted;
                expect_coo_public<float>(handle,
                                         static_cast<rocsparse_int>(m),
                                         static_cast<rocsparse_int>(n),
                                         as_index<rocsparse_int>(rows, b),
                                         as_index<rocsparse_int>(cols, b),
                                         base,
                                         general,
                                         lower,
                                         storage,
                                         expected);
                expect_coo_spmat<float, int32_t>(handle,
                                                 m,
                                                 n,
                                                 as_index<int32_t>(rows, b),
                                                 as_index<int32_t>(cols, b),
                                                 base,
                                                 general,
                                                 lower,
                                                 storage,
                                                 expected);
                expect_coo_spmat<float, int64_t>(handle,
                                                 m,
                                                 n,
                                                 as_index<int64_t>(rows, b),
                                                 as_index<int64_t>(cols, b),
                                                 base,
                                                 general,
                                                 lower,
                                                 storage,
                                                 expected);
            }
    }

    // The public 32-bit API and rocsparse_check_spmat with 32-bit and 64-bit indices.
    void expect_ell_shape(rocsparse_handle                     handle,
                          const char*                          name,
                          int                                  n,
                          int                                  width,
                          const std::vector<std::vector<int>>& rows,
                          rocsparse_data_status                sorted,
                          rocsparse_data_status                unsorted)
    {
        const int m = static_cast<int>(rows.size());
        for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            for(auto storage : {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
            {
                const auto expected = storage == rocsparse_storage_mode_sorted ? sorted : unsorted;
                for(bool spmat : {false, true})
                    expect_ell<float, int32_t>(
                        handle, "s", name, m, n, width, rows, base, storage, expected, spmat);
                expect_ell<float, int64_t>(
                    handle, "s64", name, m, n, width, rows, base, storage, expected, true);
            }
    }

    template <typename T>
    void run_shapes(rocsparse_handle handle, const char* type_name)
    {
        const rocsparse_index_base bases[] = {rocsparse_index_base_zero, rocsparse_index_base_one};
        const rocsparse_storage_mode storages[]
            = {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted};
        const rocsparse_matrix_type types[] = {rocsparse_matrix_type_general,
                                               rocsparse_matrix_type_triangular,
                                               rocsparse_matrix_type_symmetric,
                                               rocsparse_matrix_type_hermitian};
        for(auto base : bases)
            for(auto storage : storages)
                for(auto type : types)
                    for(auto uplo : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                    {
                        SCOPED_TRACE(std::string(type_name) + " empty type "
                                     + std::to_string(type));
                        expect_coo_public<T>(handle, 0, 0, {}, {}, base, type, uplo, storage, k_ok);
                        expect_coo_public<T>(handle, 2, 2, {}, {}, base, type, uplo, storage, k_ok);
                        const int b = base_of(base);
                        expect_coo_public<T>(handle,
                                             1,
                                             1,
                                             {static_cast<rocsparse_int>(b)},
                                             {static_cast<rocsparse_int>(b)},
                                             base,
                                             type,
                                             uplo,
                                             storage,
                                             k_ok);
                    }
        for(auto base : bases)
            for(auto storage : storages)
            {
                SCOPED_TRACE(std::string(type_name) + " ell empty");
                expect_ell_public<T>(handle, 0, 0, 0, {}, {}, base, storage, k_ok);
                expect_ell_public<T>(handle, 2, 2, 0, {}, {}, base, storage, k_ok);
                const int                  b   = base_of(base);
                std::vector<rocsparse_int> col = {static_cast<rocsparse_int>(b)};
                std::vector<T>             val = {scalar<T>(1)};
                expect_ell_public<T>(handle, 1, 1, 1, col, val, base, storage, k_ok);
            }
    }
}

class CheckMatrixDefects : public HandleTest
{
};

TEST_F(CheckMatrixDefects, CooFloat)
{
    run_coo_matrix<float>(handle, "s", false);
}
TEST_F(CheckMatrixDefects, CooDouble)
{
    run_coo_matrix<double>(handle, "d", false);
}
TEST_F(CheckMatrixDefects, CooComplexFloat)
{
    run_coo_matrix<rocsparse_float_complex>(handle, "c", false);
}
TEST_F(CheckMatrixDefects, CooComplexDouble)
{
    run_coo_matrix<rocsparse_double_complex>(handle, "z", false);
}
TEST_F(CheckMatrixDefects, CooFloatIndex64)
{
    run_coo_matrix<float>(handle, "s64", true);
}
TEST_F(CheckMatrixDefects, CooComplexDoubleIndex64)
{
    run_coo_matrix<rocsparse_double_complex>(handle, "z64", true);
}

TEST_F(CheckMatrixDefects, EllFloat)
{
    run_ell<float>(handle, "s", false);
}
TEST_F(CheckMatrixDefects, EllDouble)
{
    run_ell<double>(handle, "d", false);
}
TEST_F(CheckMatrixDefects, EllComplexFloat)
{
    run_ell<rocsparse_float_complex>(handle, "c", false);
}
TEST_F(CheckMatrixDefects, EllComplexDouble)
{
    run_ell<rocsparse_double_complex>(handle, "z", false);
}
TEST_F(CheckMatrixDefects, EllFloatIndex64)
{
    run_ell<float>(handle, "s64", true);
}
TEST_F(CheckMatrixDefects, EllComplexDoubleIndex64)
{
    run_ell<rocsparse_double_complex>(handle, "z64", true);
}

TEST_F(CheckMatrixDefects, EllRejectsNonGeneral)
{
    device_vector<rocsparse_int> col(std::vector<rocsparse_int>{0});
    device_vector<float>         val(std::vector<float>{1.0f});
    size_t                       bytes = 0;
    EXPECT_EQ(rocsparse_scheck_matrix_ell_buffer_size(handle,
                                                      1,
                                                      1,
                                                      1,
                                                      val.ptr,
                                                      col.ptr,
                                                      rocsparse_index_base_zero,
                                                      rocsparse_matrix_type_triangular,
                                                      rocsparse_fill_mode_lower,
                                                      rocsparse_storage_mode_sorted,
                                                      &bytes),
              rocsparse_status_invalid_value);
    EXPECT_EQ(ell_buffer<float>(handle,
                                1,
                                1,
                                1,
                                val.ptr,
                                col.ptr,
                                rocsparse_index_base_zero,
                                rocsparse_matrix_type_general,
                                rocsparse_fill_mode_upper,
                                rocsparse_storage_mode_unsorted,
                                &bytes),
              rocsparse_status_success);
}

TEST_F(CheckMatrixDefects, HybFloat)
{
    run_hyb<float>(handle, "s");
}
TEST_F(CheckMatrixDefects, HybDouble)
{
    run_hyb<double>(handle, "d");
}
TEST_F(CheckMatrixDefects, HybComplexFloat)
{
    run_hyb<rocsparse_float_complex>(handle, "c");
}
TEST_F(CheckMatrixDefects, HybComplexDouble)
{
    run_hyb<rocsparse_double_complex>(handle, "z");
}

TEST_F(CheckMatrixDefects, EmptyAndSingle)
{
    run_shapes<float>(handle, "s");
    run_shapes<double>(handle, "d");
    run_shapes<rocsparse_float_complex>(handle, "c");
    run_shapes<rocsparse_double_complex>(handle, "z");
}

// The kernels run more than one block of 256 threads, so pairs are planted at the
// first and last entries and across the boundary between entries 255 and 256.
TEST_F(CheckMatrixDefects, CooScale)
{
    const coo_coords sorted   = scale_coo_sorted();
    const coo_coords unsorted = shuffled(sorted);
    const int64_t    last     = k_scale_nnz - 1;

    coo_coords descending = sorted;
    descending.rows[256]  = descending.rows[255] - 1;
    descending.cols[256]  = 5;

    const auto s_mode = rocsparse_storage_mode_sorted;
    const auto u_mode = rocsparse_storage_mode_unsorted;
    for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(auto storage : {s_mode, u_mode})
        {
            expect_coo_scale(handle, "valid", sorted, base, storage, k_ok);
            expect_coo_scale(handle, "dup_first", with_copy(sorted, 0, 1), base, storage, k_dup);
            expect_coo_scale(
                handle, "dup_last", with_copy(sorted, last - 1, last), base, storage, k_dup);
            expect_coo_scale(
                handle, "dup_boundary", with_copy(sorted, 255, 256), base, storage, k_dup);
        }
        expect_coo_scale(handle, "descending_row", descending, base, s_mode, k_sort);
        expect_coo_scale(handle, "descending_row", descending, base, u_mode, k_ok);

        expect_coo_scale(handle, "shuffled", unsorted, base, s_mode, k_sort);
        expect_coo_scale(handle, "shuffled", unsorted, base, u_mode, k_ok);
        expect_coo_scale(
            handle, "shuffled_dup_first", with_copy(unsorted, 0, 1), base, u_mode, k_dup);
        expect_coo_scale(
            handle, "shuffled_dup_last", with_copy(unsorted, last - 1, last), base, u_mode, k_dup);
        expect_coo_scale(
            handle, "shuffled_dup_boundary", with_copy(unsorted, 255, 256), base, u_mode, k_dup);
        expect_coo_scale(
            handle, "shuffled_dup_first_last", with_copy(unsorted, 0, last), base, u_mode, k_dup);
    }
}

// The entry between the two entries of a pair, and the entries that differ only in bit 19,
// need more key bits than the short side of a 2 x 2^20 or 2^20 x 2 matrix.
TEST_F(CheckMatrixDefects, CooWideAndTall)
{
    const int64_t long_side = int64_t(1) << 20;
    const int64_t high      = int64_t(1) << 19;
    expect_coo_shape(
        handle, "wide_duplicate", 2, long_side, {0, 0, 0}, {0, high, 0}, k_sort, k_dup);
    expect_coo_shape(
        handle, "tall_duplicate", long_side, 2, {5, high, 5}, {1, 1, 1}, k_sort, k_dup);
    // The entry between the pair has the same low bits as the pair.
    expect_coo_shape(handle,
                     "tall_duplicate_low_bits",
                     long_side,
                     2,
                     {5, high + 5, 5},
                     {1, 1, 1},
                     k_sort,
                     k_dup);
    expect_coo_shape(handle, "wide_high_bit", 2, long_side, {0, 0}, {high + 1, 1}, k_sort, k_ok);
    expect_coo_shape(handle, "tall_high_bit", long_side, 2, {high + 5, 5}, {1, 1}, k_sort, k_ok);
}

// Columns of a row with 2^20 columns that differ only in bit 19 are distinct, and the last
// column sorts before the padding.
TEST_F(CheckMatrixDefects, EllHighBits)
{
    const int n    = 1 << 20;
    const int high = 1 << 19;
    expect_ell_shape(handle, "high_bit_increasing", n, 2, {{1, high + 1}, {0, high}}, k_ok, k_ok);
    expect_ell_shape(handle, "high_bit_unordered", n, 2, {{high + 1, 1}, {high, 0}}, k_sort, k_ok);
    expect_ell_shape(
        handle, "high_bit_duplicate", n, 3, {{high + 1, 1, high + 1}, {0}}, k_sort, k_dup);
    expect_ell_shape(handle, "last_column", n, 3, {{n - 1, 0, -1}, {0, n - 1}}, k_sort, k_ok);
}

// One thread checks one row, so pairs are planted in the first and last rows and in
// rows 255 and 256 on either side of the block boundary.
TEST_F(CheckMatrixDefects, EllScale)
{
    const auto s_mode = rocsparse_storage_mode_sorted;
    const auto u_mode = rocsparse_storage_mode_unsorted;
    const auto copy   = [](std::vector<std::vector<int>> rows, int row, int from, int to) {
        rows[row][to] = rows[row][from];
        return rows;
    };
    const auto sorted   = scale_ell_sorted();
    const auto unsorted = scale_ell_unsorted();
    const int  last_row = k_ell_scale_m - 1;
    const int  last     = k_ell_scale_width - 1;
    for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(auto storage : {s_mode, u_mode})
        {
            expect_ell_scale(handle, "valid", sorted, base, storage, k_ok);
            expect_ell_scale(handle, "dup_first_row", copy(sorted, 0, 0, 1), base, storage, k_dup);
            expect_ell_scale(handle,
                             "dup_last_row",
                             copy(sorted, last_row, last - 1, last),
                             base,
                             storage,
                             k_dup);
            expect_ell_scale(handle, "dup_row_255", copy(sorted, 255, 3, 4), base, storage, k_dup);
            expect_ell_scale(handle, "dup_row_256", copy(sorted, 256, 3, 4), base, storage, k_dup);
        }
        expect_ell_scale(handle, "unordered", unsorted, base, s_mode, k_sort);
        expect_ell_scale(handle, "unordered", unsorted, base, u_mode, k_ok);
        expect_ell_scale(
            handle, "unordered_dup_first_row", copy(unsorted, 0, 0, last), base, u_mode, k_dup);
        expect_ell_scale(
            handle, "unordered_dup_last_row", copy(unsorted, last_row, 0, 1), base, u_mode, k_dup);
        expect_ell_scale(
            handle, "unordered_dup_row_255", copy(unsorted, 255, 1, 6), base, u_mode, k_dup);
        expect_ell_scale(
            handle, "unordered_dup_row_256", copy(unsorted, 256, 1, 6), base, u_mode, k_dup);
    }
}

// The unsorted ELL duplicate pass sorts whole rows in batches that fit the 32-bit sizes of
// the segmented sort. A lowered batch limit splits a small matrix into several batches,
// which must report the same status as a single batch.
TEST_F(CheckMatrixDefects, EllUnsortedBatches)
{
    expect_ell_batches<int32_t>(handle, "i32");
    expect_ell_batches<int64_t>(handle, "i64");
}

// The helpers allocate the queried size plus a canary and check that the canary is
// unchanged. This runs every format and storage mode, with sizes above one block.
TEST_F(CheckMatrixDefects, TemporaryBufferCanary)
{
    const coo_coords sorted = scale_coo_sorted();
    for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
        for(auto storage : {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
        {
            expect_coo_scale(handle, "valid", sorted, base, storage, k_ok);
            expect_coo_scale(handle, "dup", with_copy(sorted, 255, 256), base, storage, k_dup);
            expect_ell_scale(handle, "valid", scale_ell_sorted(), base, storage, k_ok);
            expect_hyb_scale(handle, "valid", false, base, storage, k_ok);
            expect_hyb_scale(handle, "dup_coo_tail", true, base, storage, k_dup);
            expect_coo_public<float>(handle,
                                     2,
                                     2,
                                     shift_base({0, 1, 0}, base_of(base), false),
                                     shift_base({0, 1, 0}, base_of(base), false),
                                     base,
                                     rocsparse_matrix_type_general,
                                     rocsparse_fill_mode_lower,
                                     storage,
                                     storage == rocsparse_storage_mode_sorted ? k_sort : k_dup);
            expect_ell<float, int32_t>(
                handle, "s", "small", 2, 4, 3, {{2, -1, 2}, {0, 1, 3}}, base, storage, k_ok, false);
        }
}
