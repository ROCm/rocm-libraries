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

// A const sparse descriptor stores its arrays in const_row_data / const_col_data /
// const_val_data and leaves the mutable pointers null. These tests run rocsparse_extract
// (CSR and CSC) and the BSR-to-CSR, CSR-to-BSR, CSR-to-ELL, CSR-to-COO and CSR-to-COO AoS
// conversions of rocsparse_sparse_to_sparse on a const source. Extract calls the shared run_extract helper,
// which builds the source with the const constructor and checks the output against its host
// reference, including the sentinel and the guard tail. The other conversions run a mutable
// source and a const source that share the same arrays; both must return the same statuses
// and write the output of a host reference, and neither may write past the end of the
// output arrays.

#include "unit_test_device_conv_common.hpp"

#include <functional>
#include <string>

using namespace convut;

namespace
{
    using ConvConstSource = ConvTest;

    // Elements appended after every output array, holding the sentinel.
    constexpr size_t s_guard = 4;

    bool failed()
    {
        return ::testing::Test::HasFailure();
    }

    template <typename T>
    T numbered(int64_t k)
    {
        if constexpr(is_complex_v<T>)
        {
            using real_t
                = std::conditional_t<std::is_same_v<T, rocsparse_float_complex>, float, double>;
            return T(static_cast<real_t>(k + 1), static_cast<real_t>(k + 3));
        }
        else
        {
            return static_cast<T>(k + 1);
        }
    }

    template <typename T>
    std::vector<T> numbered_vector(int64_t n)
    {
        std::vector<T> v(static_cast<size_t>(n));
        for(int64_t k = 0; k < n; ++k)
            v[static_cast<size_t>(k)] = numbered<T>(k);
        return v;
    }

    template <typename I>
    std::vector<I> shift(std::initializer_list<int64_t> xs, int base)
    {
        std::vector<I> out;
        out.reserve(xs.size());
        for(int64_t x : xs)
            out.push_back(static_cast<I>(x + base));
        return out;
    }

    // Initial content of every output element. No routine under test writes this value.
    template <typename T>
    T sentinel()
    {
        if constexpr(std::is_integral_v<T>)
            return T(-7);
        else
            return make_val<T>(-77, -55);
    }

    // Host content of an output array of `n` elements followed by the guard.
    template <typename T>
    std::vector<T> sentinel_vector(int64_t n)
    {
        return std::vector<T>(static_cast<size_t>(n) + s_guard, sentinel<T>());
    }

    // `buf` holds `want` and still holds the sentinel everywhere after it.
    template <typename T>
    void expect_written(const DevBuf<T>& buf, const std::vector<T>& want, const std::string& what)
    {
        const std::vector<T> got      = buf.read();
        std::vector<T>       expected = want;
        if(expected.size() < got.size())
            expected.resize(got.size(), sentinel<T>());
        EXPECT_EQ(got, expected) << what;
    }

    const char* dtype_name(rocsparse_datatype dt)
    {
        switch(dt)
        {
        case rocsparse_datatype_f32_r:
            return "f32";
        case rocsparse_datatype_f64_r:
            return "f64";
        case rocsparse_datatype_f32_c:
            return "c32";
        case rocsparse_datatype_f64_c:
            return "c64";
        default:
            return "other";
        }
    }

    template <typename P, typename J, typename T>
    std::string type_label()
    {
        return std::string(dtype_name(dt_of<T>())) + " P" + (sizeof(P) == 4 ? "32" : "64") + " J"
               + (sizeof(J) == 4 ? "32" : "64");
    }

    // Statuses of the analysis stage, the compute stage, and the synchronize that
    // follows them. sync_c is meaningful only when the analysis stage left the queue healthy.
    struct StageStatus
    {
        rocsparse_status analysis = static_cast<rocsparse_status>(-1);
        rocsparse_status compute  = static_cast<rocsparse_status>(-1);
        hipError_t       sync_a   = hipErrorUnknown;
        hipError_t       sync_c   = hipErrorUnknown;
    };

    bool healthy(const StageStatus& s)
    {
        return s.analysis == st_ok && s.compute == st_ok && s.sync_a == hipSuccess
               && s.sync_c == hipSuccess;
    }

    void
        expect_same_status(const std::string& label, const StageStatus& mut, const StageStatus& cst)
    {
        EXPECT_EQ(mut.analysis, st_ok) << label;
        EXPECT_EQ(mut.sync_a, hipSuccess) << label;
        EXPECT_EQ(mut.compute, st_ok) << label;
        EXPECT_EQ(mut.sync_c, hipSuccess) << label;
        EXPECT_EQ(cst.analysis, mut.analysis) << label;
        EXPECT_EQ(cst.sync_a, mut.sync_a) << label;
        if(cst.sync_a == hipSuccess && mut.sync_a == hipSuccess)
        {
            EXPECT_EQ(cst.compute, mut.compute) << label;
            EXPECT_EQ(cst.sync_c, mut.sync_c) << label;
        }
    }

    // Status of a run and the number of non-zeros it reports.
    struct Run
    {
        StageStatus st;
        int64_t     nnz = -1;
    };

    StageStatus sparse_to_sparse(rocsparse_handle             handle,
                                 rocsparse_const_spmat_descr  source,
                                 rocsparse_spmat_descr        target,
                                 const std::function<void()>& after_analysis = nullptr)
    {
        StageStatus st;
        S2sDescr    descr;
        if(rocsparse_create_sparse_to_sparse_descr(
               &descr, source, target, rocsparse_sparse_to_sparse_alg_default)
           != st_ok)
            return st;
        if(rocsparse_sparse_to_sparse_permissive(descr) != st_ok)
            return st;

        const auto one = [&](rocsparse_sparse_to_sparse_stage stage, rocsparse_status& out) {
            size_t bytes = 0;
            out          = rocsparse_sparse_to_sparse_buffer_size(
                handle, descr, source, target, stage, &bytes);
            if(out != st_ok)
                return;
            DevBuf<char> buffer(bytes);
            out = rocsparse_sparse_to_sparse(handle, descr, source, target, stage, bytes, buffer);
        };
        one(rocsparse_sparse_to_sparse_stage_analysis, st.analysis);
        st.sync_a = hipDeviceSynchronize();
        if(st.analysis != st_ok || st.sync_a != hipSuccess)
            return st;
        if(after_analysis)
            after_analysis();
        one(rocsparse_sparse_to_sparse_stage_compute, st.compute);
        st.sync_c = hipDeviceSynchronize();
        return st;
    }

    // Const CSR and CSC extract. The host reference, the const constructor, and the
    // sentinel guard live in run_extract.
    template <typename P, typename J, typename T>
    void extract_sweep(rocsparse_handle handle)
    {
        const std::string kind = type_label<P, J, T>();
        const auto        dt   = dt_of<T>();
        for(int base : {0, 1})
        {
            for(bool csc : {false, true})
            {
                const std::string label
                    = kind + (csc ? " csc" : " csr") + " base " + std::to_string(base);
                // 3x3 with entries on, above and below the diagonal. The CSC arrays hold the
                // transposed pattern, which is the same pattern.
                for(auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
                {
                    for(auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                    {
                        if(failed())
                            return;
                        SCOPED_TRACE(label + " 3x3");
                        run_extract<P, J, true>(handle,
                                                csc,
                                                3,
                                                3,
                                                base,
                                                dt,
                                                shift<P>({0, 2, 3, 5}, base),
                                                shift<J>({0, 2, 1, 0, 2}, base),
                                                fill,
                                                diag);
                        // Two compressed rows (columns for CSC) of length four.
                        SCOPED_TRACE(label + " rectangular");
                        run_extract<P, J, true>(handle,
                                                csc,
                                                csc ? 4 : 2,
                                                csc ? 2 : 4,
                                                base,
                                                dt,
                                                shift<P>({0, 2, 4}, base),
                                                shift<J>({1, 3, 0, 2}, base),
                                                fill,
                                                diag);
                    }
                }
                // No non-zeros, with and without rows and columns. The pointer array of the
                // source is read whenever there is a row (a column for CSC).
                for(const auto& [rows, cols] : {std::pair<int64_t, int64_t>{3, 3},
                                                std::pair<int64_t, int64_t>{0, 0},
                                                std::pair<int64_t, int64_t>{3, 0},
                                                std::pair<int64_t, int64_t>{0, 3}})
                {
                    if(failed())
                        return;
                    const int64_t outer = csc ? cols : rows;
                    SCOPED_TRACE(label + " empty " + std::to_string(rows) + "x"
                                 + std::to_string(cols));
                    run_extract<P, J, true>(
                        handle,
                        csc,
                        rows,
                        cols,
                        base,
                        dt,
                        std::vector<P>(static_cast<size_t>(outer + 1), static_cast<P>(base)),
                        std::vector<J>{},
                        rocsparse_fill_mode_lower,
                        rocsparse_diag_type_non_unit);
                }
            }
        }
    }

    template <typename P, typename J, typename T>
    void run_bsr_to_csr(rocsparse_handle      handle,
                        rocsparse_direction   dir,
                        int                   base,
                        int64_t               block_dim,
                        int64_t               mb,
                        int64_t               nb,
                        const std::vector<P>& ptr_h,
                        const std::vector<J>& ind_h,
                        const std::vector<T>& val_h,
                        const std::string&    label)
    {
        SCOPED_TRACE(label + (dir == dir_r ? " row" : " column") + " bd "
                     + std::to_string(block_dim));
        const int64_t bd   = block_dim;
        const int64_t nnzb = static_cast<int64_t>(ind_h.size());
        const int64_t rows = mb * bd;
        const int64_t cols = nb * bd;
        const int64_t nnz  = nnzb * bd * bd;

        // Each block row expands to `bd` CSR rows that take the blocks in order.
        std::vector<P> want_ptr{static_cast<P>(base)};
        std::vector<J> want_ind;
        std::vector<T> want_val;
        for(int64_t br = 0; br < mb; ++br)
        {
            for(int64_t r = 0; r < bd; ++r)
            {
                for(int64_t k = ptr_h[br] - base; k < ptr_h[br + 1] - base; ++k)
                {
                    for(int64_t c = 0; c < bd; ++c)
                    {
                        want_ind.push_back(static_cast<J>((ind_h[k] - base) * bd + c + base));
                        want_val.push_back(
                            val_h[k * bd * bd + (dir == dir_r ? r * bd + c : c * bd + r)]);
                    }
                }
                want_ptr.push_back(static_cast<P>(base + want_ind.size()));
            }
        }

        DevBuf<P>  src_ptr(ptr_h);
        DevBuf<J>  src_ind(ind_h);
        DevBuf<T>  src_val(val_h);
        DevBuf<P>  mut_ptr(sentinel_vector<P>(rows + 1)), cst_ptr(sentinel_vector<P>(rows + 1));
        DevBuf<J>  mut_ind(sentinel_vector<J>(nnz)), cst_ind(sentinel_vector<J>(nnz));
        DevBuf<T>  mut_val(sentinel_vector<T>(nnz)), cst_val(sentinel_vector<T>(nnz));
        const auto ib = base_of(base);
        const auto pt = it_of<P>();
        const auto jt = it_of<J>();
        const auto dt = dt_of<T>();

        const auto once
            = [&](bool as_const, DevBuf<P>& dst_ptr, DevBuf<J>& dst_ind, DevBuf<T>& dst_val) {
                  Spmat      mutable_source, target;
                  ConstSpmat const_source;
                  const auto created = as_const ? rocsparse_create_const_bsr_descr(&const_source,
                                                                                   mb,
                                                                                   nb,
                                                                                   nnzb,
                                                                                   dir,
                                                                                   bd,
                                                                                   src_ptr.get(),
                                                                                   src_ind.get(),
                                                                                   src_val.get(),
                                                                                   pt,
                                                                                   jt,
                                                                                   ib,
                                                                                   dt)
                                                : rocsparse_create_bsr_descr(&mutable_source,
                                                                             mb,
                                                                             nb,
                                                                             nnzb,
                                                                             dir,
                                                                             bd,
                                                                             src_ptr.get(),
                                                                             src_ind.get(),
                                                                             src_val.get(),
                                                                             pt,
                                                                             jt,
                                                                             ib,
                                                                             dt);
                  EXPECT_EQ(created, st_ok);
                  EXPECT_EQ(rocsparse_create_csr_descr(&target,
                                                       rows,
                                                       cols,
                                                       0,
                                                       dst_ptr.get(),
                                                       dst_ind.get(),
                                                       dst_val.get(),
                                                       pt,
                                                       jt,
                                                       ib,
                                                       dt),
                            st_ok);
                  if(created != st_ok || target.get() == nullptr)
                      return Run{};
                  rocsparse_const_spmat_descr source
                      = as_const ? const_source.get() : mutable_source.get();
                  Run     run;
                  int64_t r, c;
                  run.st = sparse_to_sparse(handle, source, target);
                  EXPECT_EQ(rocsparse_spmat_get_size(target, &r, &c, &run.nnz), st_ok);
                  return run;
              };

        const Run mut = once(false, mut_ptr, mut_ind, mut_val);
        const Run cst = once(true, cst_ptr, cst_ind, cst_val);
        expect_same_status(label, mut.st, cst.st);
        if(!healthy(mut.st) || !healthy(cst.st))
            return;
        EXPECT_EQ(mut.nnz, nnz) << "mutable target nnz";
        EXPECT_EQ(cst.nnz, nnz) << "const target nnz";
        expect_written(mut_ptr, want_ptr, "mutable ptr");
        expect_written(cst_ptr, want_ptr, "const ptr");
        expect_written(mut_ind, want_ind, "mutable ind");
        expect_written(cst_ind, want_ind, "const ind");
        expect_written(mut_val, want_val, "mutable val");
        expect_written(cst_val, want_val, "const val");
    }

    template <typename P, typename J, typename T>
    void bsr_to_csr_sweep(rocsparse_handle handle)
    {
        const std::string kind = type_label<P, J, T>();
        for(int base : {0, 1})
        {
            for(auto dir : {dir_r, dir_c})
            {
                for(int64_t bd : {1, 2, 8, 33})
                {
                    if(failed())
                        return;
                    const std::string label = kind + " base " + std::to_string(base);
                    // 2-by-2 block grid with the two diagonal blocks.
                    run_bsr_to_csr<P, J, T>(handle,
                                            dir,
                                            base,
                                            bd,
                                            2,
                                            2,
                                            shift<P>({0, 1, 2}, base),
                                            shift<J>({0, 1}, base),
                                            numbered_vector<T>(2 * bd * bd),
                                            label + " diagonal");
                    // 2-by-3 block grid: two blocks in the first block row, one in the second.
                    run_bsr_to_csr<P, J, T>(handle,
                                            dir,
                                            base,
                                            bd,
                                            2,
                                            3,
                                            shift<P>({0, 2, 3}, base),
                                            shift<J>({0, 2, 1}, base),
                                            numbered_vector<T>(3 * bd * bd),
                                            label + " rectangular");
                    for(const auto& [mb, nb] : {std::pair<int64_t, int64_t>{2, 2},
                                                std::pair<int64_t, int64_t>{0, 0},
                                                std::pair<int64_t, int64_t>{2, 0},
                                                std::pair<int64_t, int64_t>{0, 2}})
                    {
                        run_bsr_to_csr<P, J, T>(
                            handle,
                            dir,
                            base,
                            bd,
                            mb,
                            nb,
                            std::vector<P>(static_cast<size_t>(mb + 1), static_cast<P>(base)),
                            {},
                            {},
                            label + " empty " + std::to_string(mb) + "x" + std::to_string(nb));
                    }
                }
            }
        }
    }

    template <typename P, typename J, typename T>
    void run_csr_to_ell(rocsparse_handle      handle,
                        int                   base,
                        int64_t               rows,
                        int64_t               cols,
                        const std::vector<P>& ptr_h,
                        const std::vector<J>& ind_h,
                        const std::vector<T>& val_h,
                        const std::string&    label)
    {
        SCOPED_TRACE(label);
        const int64_t nnz = static_cast<int64_t>(ind_h.size());

        // Column-major ELL: entry p of row r is at p * rows + r. Padding has column -1 and
        // value 0.
        int64_t want_width = 0;
        for(int64_t r = 0; r < rows; ++r)
            want_width = std::max<int64_t>(want_width, ptr_h[r + 1] - ptr_h[r]);
        std::vector<J> want_ind(static_cast<size_t>(rows * want_width), J(-1));
        std::vector<T> want_val(static_cast<size_t>(rows * want_width), make_val<T>(0));
        for(int64_t r = 0; r < rows; ++r)
        {
            for(int64_t k = ptr_h[r] - base; k < ptr_h[r + 1] - base; ++k)
            {
                const size_t idx = static_cast<size_t>((k - (ptr_h[r] - base)) * rows + r);
                want_ind[idx]    = ind_h[k];
                want_val[idx]    = val_h[k];
            }
        }

        // The target is created with the largest width the matrix allows, which the
        // analysis stage replaces with the computed width.
        DevBuf<P> src_ptr(ptr_h);
        DevBuf<J> src_ind(ind_h);
        DevBuf<T> src_val(val_h);
        DevBuf<J> mut_ind(sentinel_vector<J>(rows * cols)),
            cst_ind(sentinel_vector<J>(rows * cols));
        DevBuf<T> mut_val(sentinel_vector<T>(rows * cols)),
            cst_val(sentinel_vector<T>(rows * cols));
        const auto ib = base_of(base);
        const auto pt = it_of<P>();
        const auto jt = it_of<J>();
        const auto dt = dt_of<T>();

        const auto once = [&](bool as_const, DevBuf<J>& dst_ind, DevBuf<T>& dst_val) {
            Spmat      mutable_source, target;
            ConstSpmat const_source;
            const auto created = as_const ? rocsparse_create_const_csr_descr(&const_source,
                                                                             rows,
                                                                             cols,
                                                                             nnz,
                                                                             src_ptr.get(),
                                                                             src_ind.get(),
                                                                             src_val.get(),
                                                                             pt,
                                                                             jt,
                                                                             ib,
                                                                             dt)
                                          : rocsparse_create_csr_descr(&mutable_source,
                                                                       rows,
                                                                       cols,
                                                                       nnz,
                                                                       src_ptr.get(),
                                                                       src_ind.get(),
                                                                       src_val.get(),
                                                                       pt,
                                                                       jt,
                                                                       ib,
                                                                       dt);
            EXPECT_EQ(created, st_ok);
            EXPECT_EQ(rocsparse_create_ell_descr(
                          &target, rows, cols, dst_ind.get(), dst_val.get(), cols, jt, ib, dt),
                      st_ok);
            if(created != st_ok || target.get() == nullptr)
                return Run{};
            rocsparse_const_spmat_descr source
                = as_const ? const_source.get() : mutable_source.get();
            Run run;
            run.st = sparse_to_sparse(handle, source, target);
            int64_t              r, c;
            void *               ci, *cv;
            rocsparse_indextype  it;
            rocsparse_index_base b;
            rocsparse_datatype   d;
            EXPECT_EQ(rocsparse_ell_get(target, &r, &c, &ci, &cv, &run.nnz, &it, &b, &d), st_ok);
            return run;
        };
        const Run mut = once(false, mut_ind, mut_val);
        const Run cst = once(true, cst_ind, cst_val);
        expect_same_status(label, mut.st, cst.st);
        if(!healthy(mut.st) || !healthy(cst.st))
            return;
        EXPECT_EQ(mut.nnz, want_width) << "mutable ell_width";
        EXPECT_EQ(cst.nnz, want_width) << "const ell_width";
        expect_written(mut_ind, want_ind, "mutable ind");
        expect_written(cst_ind, want_ind, "const ind");
        expect_written(mut_val, want_val, "mutable val");
        expect_written(cst_val, want_val, "const val");
    }

    template <typename P, typename J, typename T>
    void csr_to_ell_sweep(rocsparse_handle handle)
    {
        const std::string kind = type_label<P, J, T>();
        for(int base : {0, 1})
        {
            if(failed())
                return;
            const std::string label = kind + " base " + std::to_string(base);
            // Rows of length 1, 2 and 0.
            run_csr_to_ell<P, J, T>(handle,
                                    base,
                                    3,
                                    4,
                                    shift<P>({0, 1, 3, 3}, base),
                                    shift<J>({0, 1, 3}, base),
                                    numbered_vector<T>(3),
                                    label + " 3x4");
            for(const auto& [rows, cols] : {std::pair<int64_t, int64_t>{3, 4},
                                            std::pair<int64_t, int64_t>{0, 0},
                                            std::pair<int64_t, int64_t>{3, 0},
                                            std::pair<int64_t, int64_t>{0, 4}})
            {
                run_csr_to_ell<P, J, T>(
                    handle,
                    base,
                    rows,
                    cols,
                    std::vector<P>(static_cast<size_t>(rows + 1), static_cast<P>(base)),
                    {},
                    {},
                    label + " empty " + std::to_string(rows) + "x" + std::to_string(cols));
            }
        }
    }

    template <typename P, typename J, typename T>
    void run_csr_to_bsr(rocsparse_handle      handle,
                        rocsparse_direction   dir,
                        int                   base,
                        int64_t               block_dim,
                        int64_t               rows,
                        int64_t               cols,
                        const std::vector<P>& ptr_h,
                        const std::vector<J>& ind_h,
                        const std::vector<T>& val_h,
                        const std::string&    label)
    {
        SCOPED_TRACE(label + (dir == dir_r ? " row" : " column") + " bd "
                     + std::to_string(block_dim));
        const int64_t bd  = block_dim;
        const int64_t nnz = static_cast<int64_t>(ind_h.size());
        const int64_t mb  = (rows + bd - 1) / bd;
        const int64_t nb  = (cols + bd - 1) / bd;
        const int64_t cap = mb * nb;

        // A block is stored when it holds an entry; block columns are in increasing order
        // and the block's other entries are zero.
        std::vector<P> want_ptr{static_cast<P>(base)};
        std::vector<J> want_ind;
        std::vector<T> want_val;
        for(int64_t br = 0; br < mb; ++br)
        {
            std::vector<int64_t> bcols;
            for(int64_t r = br * bd; r < std::min(rows, (br + 1) * bd); ++r)
                for(int64_t k = ptr_h[r] - base; k < ptr_h[r + 1] - base; ++k)
                    bcols.push_back((ind_h[k] - base) / bd);
            std::sort(bcols.begin(), bcols.end());
            bcols.erase(std::unique(bcols.begin(), bcols.end()), bcols.end());
            const size_t first = want_val.size();
            want_val.resize(first + bcols.size() * bd * bd, make_val<T>(0));
            for(int64_t bc : bcols)
                want_ind.push_back(static_cast<J>(bc + base));
            for(int64_t r = br * bd; r < std::min(rows, (br + 1) * bd); ++r)
            {
                for(int64_t k = ptr_h[r] - base; k < ptr_h[r + 1] - base; ++k)
                {
                    const int64_t c = ind_h[k] - base;
                    const int64_t block
                        = std::lower_bound(bcols.begin(), bcols.end(), c / bd) - bcols.begin();
                    const int64_t lr = r % bd, lc = c % bd;
                    want_val[first + block * bd * bd + (dir == dir_r ? lr * bd + lc : lc * bd + lr)]
                        = val_h[k];
                }
            }
            want_ptr.push_back(static_cast<P>(base + want_ind.size()));
        }
        const int64_t want_nnzb = static_cast<int64_t>(want_ind.size());

        DevBuf<P> src_ptr(ptr_h);
        DevBuf<J> src_ind(ind_h);
        DevBuf<T> src_val(val_h);
        DevBuf<P> mut_ptr(sentinel_vector<P>(mb + 1)), cst_ptr(sentinel_vector<P>(mb + 1));
        DevBuf<J> mut_ind(sentinel_vector<J>(cap)), cst_ind(sentinel_vector<J>(cap));
        DevBuf<T> mut_val(sentinel_vector<T>(cap * bd * bd)),
            cst_val(sentinel_vector<T>(cap * bd * bd));
        const auto ib = base_of(base);
        const auto pt = it_of<P>();
        const auto jt = it_of<J>();
        const auto dt = dt_of<T>();

        const auto once = [&](bool       as_const,
                              DevBuf<P>& dst_ptr,
                              DevBuf<J>& dst_ind,
                              DevBuf<T>& dst_val) {
            Spmat      mutable_source, target;
            ConstSpmat const_source;
            const auto created = as_const ? rocsparse_create_const_csr_descr(&const_source,
                                                                             rows,
                                                                             cols,
                                                                             nnz,
                                                                             src_ptr.get(),
                                                                             src_ind.get(),
                                                                             src_val.get(),
                                                                             pt,
                                                                             jt,
                                                                             ib,
                                                                             dt)
                                          : rocsparse_create_csr_descr(&mutable_source,
                                                                       rows,
                                                                       cols,
                                                                       nnz,
                                                                       src_ptr.get(),
                                                                       src_ind.get(),
                                                                       src_val.get(),
                                                                       pt,
                                                                       jt,
                                                                       ib,
                                                                       dt);
            EXPECT_EQ(created, st_ok);
            // The block count is known after the analysis stage, which writes the row pointer;
            // the column and value arrays are attached for the compute stage.
            EXPECT_EQ(
                rocsparse_create_bsr_descr(
                    &target, mb, nb, 0, dir, bd, dst_ptr.get(), nullptr, nullptr, pt, jt, ib, dt),
                st_ok);
            if(created != st_ok || target.get() == nullptr)
                return Run{};
            rocsparse_const_spmat_descr source
                = as_const ? const_source.get() : mutable_source.get();
            Run run;
            run.st = sparse_to_sparse(handle, source, target, [&] {
                int64_t r, c;
                EXPECT_EQ(rocsparse_spmat_get_size(target, &r, &c, &run.nnz), st_ok);
                EXPECT_EQ(
                    rocsparse_bsr_set_pointers(target, dst_ptr.get(), dst_ind.get(), dst_val.get()),
                    st_ok);
            });
            return run;
        };

        const Run mut = once(false, mut_ptr, mut_ind, mut_val);
        const Run cst = once(true, cst_ptr, cst_ind, cst_val);
        expect_same_status(label, mut.st, cst.st);
        if(!healthy(mut.st) || !healthy(cst.st))
            return;
        EXPECT_EQ(mut.nnz, want_nnzb) << "mutable target nnzb";
        EXPECT_EQ(cst.nnz, want_nnzb) << "const target nnzb";
        expect_written(mut_ptr, want_ptr, "mutable ptr");
        expect_written(cst_ptr, want_ptr, "const ptr");
        expect_written(mut_ind, want_ind, "mutable ind");
        expect_written(cst_ind, want_ind, "const ind");
        expect_written(mut_val, want_val, "mutable val");
        expect_written(cst_val, want_val, "const val");
    }

    template <typename P, typename J, typename T>
    void csr_to_bsr_sweep(rocsparse_handle handle)
    {
        const std::string kind = type_label<P, J, T>();
        for(int base : {0, 1})
        {
            for(auto dir : {dir_r, dir_c})
            {
                if(failed())
                    return;
                const std::string label = kind + " base " + std::to_string(base);
                run_csr_to_bsr<P, J, T>(handle,
                                        dir,
                                        base,
                                        1,
                                        3,
                                        3,
                                        shift<P>({0, 2, 3, 5}, base),
                                        shift<J>({0, 2, 1, 0, 2}, base),
                                        numbered_vector<T>(5),
                                        label + " 3x3");
                // Three of the four 2x2 blocks hold entries; the upper-right one is partly filled.
                run_csr_to_bsr<P, J, T>(handle,
                                        dir,
                                        base,
                                        2,
                                        4,
                                        4,
                                        shift<P>({0, 3, 5, 7, 9}, base),
                                        shift<J>({0, 1, 2, 0, 1, 2, 3, 2, 3}, base),
                                        numbered_vector<T>(9),
                                        label + " 4x4");
                for(const auto& [rows, cols] : {std::pair<int64_t, int64_t>{4, 4},
                                                std::pair<int64_t, int64_t>{0, 0},
                                                std::pair<int64_t, int64_t>{4, 0},
                                                std::pair<int64_t, int64_t>{0, 4}})
                {
                    run_csr_to_bsr<P, J, T>(
                        handle,
                        dir,
                        base,
                        2,
                        rows,
                        cols,
                        std::vector<P>(static_cast<size_t>(rows + 1), static_cast<P>(base)),
                        {},
                        {},
                        label + " empty " + std::to_string(rows) + "x" + std::to_string(cols));
                }
            }
        }
    }

    // CSR to COO (and COO AoS): the row pointer is expanded and the columns and values are
    // copied. The COO index type is the column type J of the source; the row pointer type
    // P may differ. Both index bases; with base one a column array left unwritten (or copied
    // from a null mutable pointer) cannot pass for the right one.
    template <typename P, typename J, typename T>
    void run_csr_to_coo(rocsparse_handle      handle,
                        bool                  aos,
                        int                   base,
                        int64_t               rows,
                        int64_t               cols,
                        const std::vector<P>& ptr_h,
                        const std::vector<J>& ind_h,
                        const std::vector<T>& val_h,
                        const std::string&    label)
    {
        SCOPED_TRACE(label + (aos ? " coo_aos" : " coo"));
        const int64_t nnz = static_cast<int64_t>(ind_h.size());

        std::vector<J> want_row, want_col, want_aos;
        for(int64_t r = 0; r < rows; ++r)
            for(int64_t k = ptr_h[r] - base; k < ptr_h[r + 1] - base; ++k)
            {
                want_row.push_back(static_cast<J>(r + base));
                want_col.push_back(ind_h[k]);
                want_aos.push_back(static_cast<J>(r + base));
                want_aos.push_back(ind_h[k]);
            }

        DevBuf<P> src_ptr(ptr_h);
        DevBuf<J> src_ind(ind_h);
        DevBuf<T> src_val(val_h);
        // Only the arrays of the chosen target layout are allocated.
        const std::vector<J> soa_init = aos ? std::vector<J>() : sentinel_vector<J>(nnz);
        const std::vector<J> aos_init = aos ? sentinel_vector<J>(2 * nnz) : std::vector<J>();
        DevBuf<J>  mut_row(soa_init), cst_row(soa_init), mut_col(soa_init), cst_col(soa_init);
        DevBuf<J>  mut_aos(aos_init), cst_aos(aos_init);
        DevBuf<T>  mut_val(sentinel_vector<T>(nnz)), cst_val(sentinel_vector<T>(nnz));
        const auto ib = base_of(base);
        const auto pt = it_of<P>();
        const auto jt = it_of<J>();
        const auto dt = dt_of<T>();

        const auto once = [&](bool       as_const,
                              DevBuf<J>& dst_row,
                              DevBuf<J>& dst_col,
                              DevBuf<J>& dst_aos,
                              DevBuf<T>& dst_val) {
            Spmat      mutable_source, target;
            ConstSpmat const_source;
            const auto created = as_const ? rocsparse_create_const_csr_descr(&const_source,
                                                                             rows,
                                                                             cols,
                                                                             nnz,
                                                                             src_ptr.get(),
                                                                             src_ind.get(),
                                                                             src_val.get(),
                                                                             pt,
                                                                             jt,
                                                                             ib,
                                                                             dt)
                                          : rocsparse_create_csr_descr(&mutable_source,
                                                                       rows,
                                                                       cols,
                                                                       nnz,
                                                                       src_ptr.get(),
                                                                       src_ind.get(),
                                                                       src_val.get(),
                                                                       pt,
                                                                       jt,
                                                                       ib,
                                                                       dt);
            EXPECT_EQ(created, st_ok);
            EXPECT_EQ(aos ? rocsparse_create_coo_aos_descr(
                                &target, rows, cols, nnz, dst_aos.get(), dst_val.get(), jt, ib, dt)
                          : rocsparse_create_coo_descr(&target,
                                                       rows,
                                                       cols,
                                                       nnz,
                                                       dst_row.get(),
                                                       dst_col.get(),
                                                       dst_val.get(),
                                                       jt,
                                                       ib,
                                                       dt),
                      st_ok);
            if(created != st_ok || target.get() == nullptr)
                return Run{};
            rocsparse_const_spmat_descr source
                = as_const ? const_source.get() : mutable_source.get();
            Run run;
            run.st = sparse_to_sparse(handle, source, target);
            int64_t r, c;
            EXPECT_EQ(rocsparse_spmat_get_size(target, &r, &c, &run.nnz), st_ok);
            return run;
        };

        const Run mut = once(false, mut_row, mut_col, mut_aos, mut_val);
        const Run cst = once(true, cst_row, cst_col, cst_aos, cst_val);
        expect_same_status(label, mut.st, cst.st);
        if(!healthy(mut.st) || !healthy(cst.st))
            return;
        EXPECT_EQ(mut.nnz, nnz) << "mutable target nnz";
        EXPECT_EQ(cst.nnz, nnz) << "const target nnz";
        if(aos)
        {
            expect_written(mut_aos, want_aos, "mutable ind");
            expect_written(cst_aos, want_aos, "const ind");
        }
        else
        {
            expect_written(mut_row, want_row, "mutable row");
            expect_written(cst_row, want_row, "const row");
            expect_written(mut_col, want_col, "mutable col");
            expect_written(cst_col, want_col, "const col");
        }
        expect_written(mut_val, val_h, "mutable val");
        expect_written(cst_val, val_h, "const val");
    }

    template <typename P, typename J, typename T>
    void csr_to_coo_sweep(rocsparse_handle handle)
    {
        const std::string kind = type_label<P, J, T>();
        for(bool aos : {false, true})
            for(int base : {0, 1})
            {
                if(failed())
                    return;
                const std::string label = kind + " base " + std::to_string(base);
                // Rows of length 2, 0, 1 and 3.
                run_csr_to_coo<P, J, T>(handle,
                                        aos,
                                        base,
                                        4,
                                        5,
                                        shift<P>({0, 2, 2, 3, 6}, base),
                                        shift<J>({1, 4, 0, 1, 2, 4}, base),
                                        numbered_vector<T>(6),
                                        label + " 4x5");
                run_csr_to_coo<P, J, T>(handle,
                                        aos,
                                        base,
                                        3,
                                        4,
                                        std::vector<P>(4, static_cast<P>(base)),
                                        {},
                                        {},
                                        label + " empty 3x4");
            }
    }

    // Pointer and index widths, in the style of ConvExtract.ExtractIndexTypesAndFormats.
    // The null mutable pointers do not depend on the value type, so each conversion runs
    // as float. The four widths are 32/32, 64/32, 32/64 and 64/64.
    template <typename SWEEP>
    void type_sweep(rocsparse_handle handle)
    {
        SWEEP::template run<int32_t, int32_t, float>(handle);
        SWEEP::template run<int64_t, int32_t, float>(handle);
        SWEEP::template run<int32_t, int64_t, float>(handle);
        SWEEP::template run<int64_t, int64_t, float>(handle);
    }

#define CONST_SOURCE_SWEEP(NAME, FUNCTION)            \
    struct NAME                                       \
    {                                                 \
        template <typename P, typename J, typename T> \
        static void run(rocsparse_handle handle)      \
        {                                             \
            if(!failed())                             \
                FUNCTION<P, J, T>(handle);            \
        }                                             \
    }

    CONST_SOURCE_SWEEP(ExtractSweep, extract_sweep);
    CONST_SOURCE_SWEEP(Bsr2CsrSweep, bsr_to_csr_sweep);
    CONST_SOURCE_SWEEP(Csr2EllSweep, csr_to_ell_sweep);
    CONST_SOURCE_SWEEP(Csr2BsrSweep, csr_to_bsr_sweep);
    CONST_SOURCE_SWEEP(Csr2CooSweep, csr_to_coo_sweep);
#undef CONST_SOURCE_SWEEP

    // rocm-libraries#15888: one 2x2 row-major block of a const BSR matrix.
    TEST_F(ConvConstSource, Bsr2CsrSingleBlock)
    {
        run_bsr_to_csr<int32_t, int32_t, float>(
            handle, dir_r, 0, 2, 1, 1, {0, 1}, {0}, {1.f, 2.f, 3.f, 4.f}, "single block");
    }

    TEST_F(ConvConstSource, Bsr2CsrMatchesReference)
    {
        type_sweep<Bsr2CsrSweep>(handle);
    }

    // The CSR-to-ELL and CSR-to-BSR conversions already passed the const arrays to their
    // kernels; these tests guard that.
    TEST_F(ConvConstSource, Csr2EllMatchesReference)
    {
        type_sweep<Csr2EllSweep>(handle);
    }

    TEST_F(ConvConstSource, Csr2BsrMatchesReference)
    {
        type_sweep<Csr2BsrSweep>(handle);
    }

    // The CSR-to-COO conversion copies the columns from the source descriptor's column
    // pointer, which only the const field holds for a const source.
    TEST_F(ConvConstSource, Csr2CooMatchesReference)
    {
        type_sweep<Csr2CooSweep>(handle);
    }

    // A single-precision source and a double-precision target. The conversion rejects the
    // pair for both kinds of source, whether or not the outer data type guard sees the
    // values.
    TEST_F(ConvConstSource, Csr2EllDatatypeMismatch)
    {
        DevBuf<int32_t> ptr(std::vector<int32_t>{0, 1, 1});
        DevBuf<int32_t> ind(std::vector<int32_t>{0});
        DevBuf<float>   val(std::vector<float>{1.f});
        DevBuf<int32_t> ell_ind(2);
        DevBuf<double>  ell_val(2);
        const auto      once = [&](bool as_const) {
            Spmat      mutable_source, target;
            ConstSpmat const_source;
            if(as_const)
            {
                EXPECT_EQ(rocsparse_create_const_csr_descr(
                              &const_source, 2, 2, 1, ptr, ind, val, i32, i32, base0, f32),
                          st_ok);
            }
            else
            {
                EXPECT_EQ(rocsparse_create_csr_descr(
                              &mutable_source, 2, 2, 1, ptr, ind, val, i32, i32, base0, f32),
                          st_ok);
            }
            EXPECT_EQ(rocsparse_create_ell_descr(
                          &target, 2, 2, ell_ind, ell_val, 1, i32, base0, rocsparse_datatype_f64_r),
                      st_ok);
            rocsparse_const_spmat_descr source
                = as_const ? const_source.get() : mutable_source.get();
            return sparse_to_sparse(handle, source, target);
        };
        const StageStatus mut = once(false);
        const StageStatus cst = once(true);
        EXPECT_EQ(mut.analysis, st_ok);
        EXPECT_EQ(cst.analysis, mut.analysis);
        EXPECT_EQ(mut.compute, rocsparse_status_not_implemented);
        EXPECT_EQ(cst.compute, mut.compute);
    }

    // The target is created with no value array, so the outer datatype guard never
    // runs and only the inner check is tested.
    TEST_F(ConvConstSource, Csr2BsrDatatypeMismatch)
    {
        DevBuf<int32_t> ptr(std::vector<int32_t>{0, 1, 1});
        DevBuf<int32_t> ind(std::vector<int32_t>{0});
        DevBuf<float>   val(std::vector<float>{1.f});
        DevBuf<int32_t> bptr(3);
        const auto      once = [&](bool as_const) {
            Spmat      mutable_source, target;
            ConstSpmat const_source;
            if(as_const)
            {
                EXPECT_EQ(rocsparse_create_const_csr_descr(
                              &const_source, 2, 2, 1, ptr, ind, val, i32, i32, base0, f32),
                          st_ok);
            }
            else
            {
                EXPECT_EQ(rocsparse_create_csr_descr(
                              &mutable_source, 2, 2, 1, ptr, ind, val, i32, i32, base0, f32),
                          st_ok);
            }
            EXPECT_EQ(rocsparse_create_bsr_descr(&target,
                                                 2,
                                                 2,
                                                 0,
                                                 dir_r,
                                                 1,
                                                 bptr,
                                                 nullptr,
                                                 nullptr,
                                                 i32,
                                                 i32,
                                                 base0,
                                                 rocsparse_datatype_f64_r),
                      st_ok);
            rocsparse_const_spmat_descr source
                = as_const ? const_source.get() : mutable_source.get();
            return sparse_to_sparse(handle, source, target);
        };
        // The analysis stage only counts blocks.
        const StageStatus mut = once(false);
        const StageStatus cst = once(true);
        EXPECT_EQ(mut.analysis, st_ok);
        EXPECT_EQ(cst.analysis, mut.analysis);
        EXPECT_EQ(mut.compute, rocsparse_status_not_implemented);
        EXPECT_EQ(cst.compute, mut.compute);
    }

    // rocm-libraries#15896: the 2x2 diagonal of a const CSR and a const CSC matrix.
    TEST_F(ConvConstSource, ExtractDiagonal)
    {
        for(bool csc : {false, true})
        {
            SCOPED_TRACE(csc ? "csc" : "csr");
            run_extract<int32_t, int32_t, true>(handle,
                                                csc,
                                                2,
                                                2,
                                                0,
                                                f32,
                                                std::vector<int32_t>{0, 1, 2},
                                                std::vector<int32_t>{0, 1},
                                                rocsparse_fill_mode_lower,
                                                rocsparse_diag_type_non_unit);
        }
    }

    TEST_F(ConvConstSource, ExtractMatchesReference)
    {
        type_sweep<ExtractSweep>(handle);
    }
} // namespace
