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

// Triangular-solve data of rocsparse_mat_info (rocsparse::trm_info_t and
// rocsparse::trm_data_t):
//  * every accessor, the copy constructor and the assignment (which refuses a
//    destination of another size) of trm_info_t, and the per-routine info objects of a
//    mat_info (csrsv, csrsm, csrilu0, csric0, csrildlt0, bsrsv, bsrsm, bsrilu0, bsric0,
//    ellsv): creation, the four (operation, fill mode) slots, sharing and clearing;
//  * rocsparse_copy_mat_info of an info that carries triangular-solve analysis data (the
//    copy constructor when the destination is empty, the assignment when it already
//    holds the data). The analysis of the transposed operations fills every array of the
//    info, and each copy is used for the solve and compared with a dense reference.

#include "unit_test_device_precond_common.hpp"

#include "rocsparse_mat_info.hpp"
#include "rocsparse_trm_data_t.hpp"
#include "rocsparse_trm_info.hpp"

#include <string>
#include <vector>

using namespace precond_ut;

namespace
{
    // Device copy of `host`; the trm_info_t that adopts the pointer frees it.
    template <typename T>
    T* make_device_array(const std::vector<T>& host)
    {
        T* p = nullptr;
        if(hipMalloc(&p, host.size() * sizeof(T)) != hipSuccess)
            return nullptr;
        if(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
        {
            (void)hipFree(p);
            return nullptr;
        }
        return p;
    }

    template <typename T>
    std::vector<T> read_device_array(const void* p, size_t n)
    {
        std::vector<T> out(n);
        EXPECT_EQ(hipMemcpy(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
        return out;
    }

    // A trm_info_t of order m = 4 with nnz = 6 and 32-bit indices whose five device arrays
    // are filled with `base`, `base + 1`, ...
    void fill_trm_info(rocsparse::trm_info_t& info, int32_t base)
    {
        const int32_t m   = 4;
        const int32_t nnz = 6;
        info.set_m(m);
        info.set_nnz(nnz);
        info.set_max_nnz(3);
        info.set_offset_indextype(rocsparse_indextype_i32);
        info.set_index_indextype(rocsparse_indextype_i32);
        auto fill = [&](void** slot, int32_t count, int32_t first) {
            std::vector<int32_t> host(static_cast<size_t>(count));
            for(int32_t i = 0; i < count; ++i)
                host[i] = first + i;
            *slot = make_device_array(host);
            ASSERT_NE(*slot, nullptr);
        };
        fill(info.get_ref_row_map(), m, base);
        fill(info.get_ref_diag_ind(), m, base + 10);
        fill(info.get_ref_transposed_perm(), nnz, base + 20);
        fill(info.get_ref_transposed_row_ptr(), m + 1, base + 30);
        fill(info.get_ref_transposed_col_ind(), nnz, base + 40);
    }

    void expect_same_arrays(rocsparse::trm_info_t& a, rocsparse::trm_info_t& b)
    {
        const size_t m   = static_cast<size_t>(a.get_m());
        const size_t nnz = static_cast<size_t>(a.get_nnz());
        EXPECT_EQ(read_device_array<int32_t>(a.get_row_map(), m),
                  read_device_array<int32_t>(b.get_row_map(), m));
        EXPECT_EQ(read_device_array<int32_t>(a.get_diag_ind(), m),
                  read_device_array<int32_t>(b.get_diag_ind(), m));
        EXPECT_EQ(read_device_array<int32_t>(a.get_transposed_perm(), nnz),
                  read_device_array<int32_t>(b.get_transposed_perm(), nnz));
        EXPECT_EQ(read_device_array<int32_t>(a.get_transposed_row_ptr(), m + 1),
                  read_device_array<int32_t>(b.get_transposed_row_ptr(), m + 1));
        EXPECT_EQ(read_device_array<int32_t>(a.get_transposed_col_ind(), nnz),
                  read_device_array<int32_t>(b.get_transposed_col_ind(), nnz));
    }

    // The four storage slots of a trm data holder: no-transpose / transpose times lower /
    // upper (conjugate transpose shares the transpose slot).
    struct slot
    {
        rocsparse_operation op;
        rocsparse_fill_mode fill;
    };
    constexpr slot slots[] = {{rocsparse_operation_none, rocsparse_fill_mode_lower},
                              {rocsparse_operation_none, rocsparse_fill_mode_upper},
                              {rocsparse_operation_transpose, rocsparse_fill_mode_lower},
                              {rocsparse_operation_transpose, rocsparse_fill_mode_upper}};

    // Drives the accessors of one family of mat_info: `get()` returns the family object,
    // `get_slot` / `set_slot` its slots.
    template <typename Get, typename GetSlot, typename SetSlot, typename Shared, typename Clear>
    void exercise_family(Get get, GetSlot get_slot, SetSlot set_slot, Shared shared, Clear clear)
    {
        // Created on first use.
        auto* object = get();
        ASSERT_NE(object, nullptr);
        EXPECT_EQ(get(), object);
        EXPECT_NE(shared().get(), nullptr);
        EXPECT_EQ(shared().get(), object);

        // Nothing is analyzed yet.
        for(const auto& s : slots)
            EXPECT_EQ(get_slot(s.op, s.fill), nullptr);

        // Every slot takes (and later releases) its own data.
        rocsparse::trm_info_t* stored[4] = {};
        for(size_t i = 0; i < 4; ++i)
        {
            stored[i] = new rocsparse::trm_info_t();
            set_slot(slots[i].op, slots[i].fill, stored[i]);
        }
        for(size_t i = 0; i < 4; ++i)
            EXPECT_EQ(get_slot(slots[i].op, slots[i].fill), stored[i]);
        // The conjugate transpose shares the slot of the transpose.
        EXPECT_EQ(get_slot(rocsparse_operation_conjugate_transpose, rocsparse_fill_mode_lower),
                  stored[2]);
        EXPECT_EQ(get_slot(rocsparse_operation_conjugate_transpose, rocsparse_fill_mode_upper),
                  stored[3]);

        // Clearing drops the object; the next access creates a new empty one.
        clear();
        for(const auto& s : slots)
            EXPECT_EQ(get_slot(s.op, s.fill), nullptr);
    }

    // Dense triangle of `a` (lower or upper, with the diagonal) as a row-major matrix.
    std::vector<double> dense_triangle(const csr_matrix<double>& a, bool lower)
    {
        const int32_t       n = a.n;
        std::vector<double> t(static_cast<size_t>(n) * n, 0.0);
        for(int32_t i = 0; i < n; ++i)
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                if(lower ? (a.ind[p] <= i) : (a.ind[p] >= i))
                    t[static_cast<size_t>(i) * n + a.ind[p]] = a.val[p];
        return t;
    }

    // Solves op(T) y = rhs for the dense triangular matrix T.
    std::vector<double> solve_dense(const std::vector<double>& t,
                                    int32_t                    n,
                                    bool                       lower,
                                    bool                       transpose,
                                    const std::vector<double>& rhs)
    {
        const auto at = [&](int32_t i, int32_t j) {
            return transpose ? t[static_cast<size_t>(j) * n + i]
                             : t[static_cast<size_t>(i) * n + j];
        };
        // op(T) is lower triangular for (lower, no transpose) and (upper, transpose).
        const bool          forward = (lower != transpose);
        std::vector<double> y(static_cast<size_t>(n), 0.0);
        for(int32_t s = 0; s < n; ++s)
        {
            const int32_t i   = forward ? s : n - 1 - s;
            double        sum = rhs[i];
            for(int32_t j = forward ? 0 : i + 1; j < (forward ? i : n); ++j)
                sum -= at(i, j) * y[j];
            y[i] = sum / at(i, i);
        }
        return y;
    }

    // Owns a rocsparse_mat_info: it is released on every exit path, including the
    // early returns of failed ASSERT_* macros.
    struct info_guard
    {
        rocsparse_mat_info info = nullptr;

        info_guard()
        {
            EXPECT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        }

        // Destroys the info now (the guard is empty afterwards).
        void reset()
        {
            if(info)
                EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
            info = nullptr;
        }

        ~info_guard()
        {
            reset();
        }

        info_guard(const info_guard&) = delete;
        info_guard& operator=(const info_guard&) = delete;
    };

    struct trm_fixture
    {
        csr_matrix<double>           a;
        device_vector<rocsparse_int> row;
        device_vector<rocsparse_int> col;
        device_vector<double>        val;
        rocsparse_mat_descr          lower = nullptr;
        rocsparse_mat_descr          upper = nullptr;

        trm_fixture()
            : a(make_grid_matrix<double>(4, true))
        {
            upload(row, as_rocsparse_int(a.ptr));
            upload(col, as_rocsparse_int(a.ind));
            upload(val, a.val);
            EXPECT_EQ(rocsparse_create_mat_descr(&lower), rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_mat_descr(&upper), rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_fill_mode(lower, rocsparse_fill_mode_lower),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_set_mat_fill_mode(upper, rocsparse_fill_mode_upper),
                      rocsparse_status_success);
        }

        ~trm_fixture()
        {
            if(lower)
                (void)rocsparse_destroy_mat_descr(lower);
            if(upper)
                (void)rocsparse_destroy_mat_descr(upper);
        }

        trm_fixture(const trm_fixture&) = delete;
        trm_fixture& operator=(const trm_fixture&) = delete;
    };

    // Buffer size and analysis of one triangle / operation into `info`.
    void analyze(rocsparse_handle    handle,
                 trm_fixture&        f,
                 rocsparse_mat_info  info,
                 bool                lower,
                 rocsparse_operation trans,
                 size_t&             buffer_size,
                 void*               buffer)
    {
        rocsparse_mat_descr descr = lower ? f.lower : f.upper;
        size_t              size  = 0;
        ASSERT_EQ(rocsparse_dcsrsv_buffer_size(
                      handle, trans, f.a.n, f.a.nnz(), descr, f.val, f.row, f.col, info, &size),
                  rocsparse_status_success);
        if(buffer == nullptr)
        {
            buffer_size = std::max(buffer_size, size);
            return;
        }
        ASSERT_LE(size, buffer_size);
        ASSERT_EQ(rocsparse_dcsrsv_analysis(handle,
                                            trans,
                                            f.a.n,
                                            f.a.nnz(),
                                            descr,
                                            f.val,
                                            f.row,
                                            f.col,
                                            info,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            buffer),
                  rocsparse_status_success);
    }

    // Solve with the analysis stored in `info` and compare with the dense reference.
    void check_solve(rocsparse_handle    handle,
                     trm_fixture&        f,
                     rocsparse_mat_info  info,
                     bool                lower,
                     rocsparse_operation trans,
                     void*               buffer,
                     const std::string&  what)
    {
        SCOPED_TRACE(what);
        const int32_t       n = f.a.n;
        std::vector<double> rhs(static_cast<size_t>(n));
        for(int32_t i = 0; i < n; ++i)
            rhs[i] = 1.0 + 0.25 * (i % 5);
        device_vector<double> dx{rhs};
        device_vector<double> dy{static_cast<size_t>(n)};
        ASSERT_TRUE(dx.ptr && dy.ptr);
        const double alpha = 1.5;
        ASSERT_EQ(rocsparse_dcsrsv_solve(handle,
                                         trans,
                                         n,
                                         f.a.nnz(),
                                         &alpha,
                                         lower ? f.lower : f.upper,
                                         f.val,
                                         f.row,
                                         f.col,
                                         info,
                                         dx,
                                         dy,
                                         rocsparse_solve_policy_auto,
                                         buffer),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<double> scaled = rhs;
        for(double& v : scaled)
            v *= alpha;
        const auto want = solve_dense(
            dense_triangle(f.a, lower), n, lower, trans == rocsparse_operation_transpose, scaled);
        expect_close(to_host(dy), want);
    }
}

class PrecondTrmInfo : public HandleTest
{
};

TEST_F(PrecondTrmInfo, trm_info_accessors)
{
    descr_info d;
    ASSERT_NE(d.descr, nullptr);
    rocsparse::trm_info_t info;
    EXPECT_EQ(info.get_m(), 0);
    EXPECT_EQ(info.get_nnz(), 0);
    EXPECT_EQ(info.get_max_nnz(), 0);
    EXPECT_EQ(info.get_row_map(), nullptr);
    EXPECT_EQ(info.get_diag_ind(), nullptr);
    EXPECT_EQ(info.get_transposed_perm(), nullptr);
    EXPECT_EQ(info.get_transposed_row_ptr(), nullptr);
    EXPECT_EQ(info.get_transposed_col_ind(), nullptr);
    EXPECT_EQ(info.get_row_ptr(), nullptr);
    EXPECT_EQ(info.get_col_ind(), nullptr);
    EXPECT_EQ(info.get_descr(), nullptr);

    const int32_t marker[2] = {0, 0};
    info.set_m(11);
    info.set_nnz(22);
    info.set_max_nnz(33);
    info.set_row_ptr(&marker[0]);
    info.set_col_ind(&marker[1]);
    info.set_descr(d.descr);
    info.set_offset_indextype(rocsparse_indextype_i64);
    info.set_index_indextype(rocsparse_indextype_i32);
    EXPECT_EQ(info.get_m(), 11);
    EXPECT_EQ(info.get_nnz(), 22);
    EXPECT_EQ(info.get_max_nnz(), 33);
    EXPECT_EQ(info.get_row_ptr(), &marker[0]);
    EXPECT_EQ(info.get_col_ind(), &marker[1]);
    EXPECT_EQ(info.get_descr(), d.descr);
    EXPECT_EQ(info.get_offset_indextype(), rocsparse_indextype_i64);
    EXPECT_EQ(info.get_index_indextype(), rocsparse_indextype_i32);

    // The same through a constant reference.
    const rocsparse::trm_info_t& cinfo = info;
    EXPECT_EQ(cinfo.get_m(), 11);
    EXPECT_EQ(cinfo.get_row_map(), nullptr);
    EXPECT_EQ(cinfo.get_diag_ind(), nullptr);
    EXPECT_EQ(cinfo.get_transposed_perm(), nullptr);
    EXPECT_EQ(cinfo.get_transposed_row_ptr(), nullptr);
    EXPECT_EQ(cinfo.get_transposed_col_ind(), nullptr);

    // destroy() accepts both a live object and a null pointer.
    rocsparse::trm_info_t::destroy(new rocsparse::trm_info_t());
    rocsparse::trm_info_t::destroy(nullptr);
}

TEST_F(PrecondTrmInfo, trm_info_copy_and_assignment)
{
    rocsparse::trm_info_t source;
    fill_trm_info(source, 100);
    ASSERT_NE(source.get_row_map(), nullptr);

    // Copy construction duplicates the device arrays.
    rocsparse::trm_info_t copy(source);
    EXPECT_EQ(copy.get_m(), source.get_m());
    EXPECT_EQ(copy.get_nnz(), source.get_nnz());
    EXPECT_EQ(copy.get_max_nnz(), source.get_max_nnz());
    EXPECT_NE(copy.get_row_map(), source.get_row_map());
    EXPECT_NE(copy.get_transposed_col_ind(), source.get_transposed_col_ind());
    expect_same_arrays(source, copy);

    // Assignment to an info of the same size allocates what the destination lacks ...
    rocsparse::trm_info_t empty;
    empty.set_m(source.get_m());
    empty.set_nnz(source.get_nnz());
    empty.set_max_nnz(source.get_max_nnz());
    empty.set_offset_indextype(rocsparse_indextype_i32);
    empty.set_index_indextype(rocsparse_indextype_i32);
    empty = source;
    ASSERT_NE(empty.get_row_map(), nullptr);
    ASSERT_NE(empty.get_transposed_row_ptr(), nullptr);
    expect_same_arrays(source, empty);

    // ... and overwrites the arrays of a destination that has them.
    rocsparse::trm_info_t other;
    fill_trm_info(other, 500);
    other = source;
    expect_same_arrays(source, other);

    // A destination of another size is refused.
    rocsparse::trm_info_t smaller;
    smaller.set_m(source.get_m() - 1);
    smaller.set_nnz(source.get_nnz());
    smaller.set_max_nnz(source.get_max_nnz());
    smaller.set_offset_indextype(rocsparse_indextype_i32);
    smaller.set_index_indextype(rocsparse_indextype_i32);
    EXPECT_ANY_THROW(smaller = source);
}

TEST_F(PrecondTrmInfo, mat_info_trm_slots_of_every_routine)
{
    info_guard g;
    ASSERT_NE(g.info, nullptr);
    rocsparse_mat_info info = g.info;

#define EXERCISE_FAMILY(NAME)                                                                   \
    exercise_family([&] { return info->get_##NAME##_info(); },                                  \
                    [&](auto op, auto fill) { return info->get_##NAME##_info(op, fill); },      \
                    [&](auto op, auto fill, auto* t) { info->set_##NAME##_info(op, fill, t); }, \
                    [&] { return info->get_shared_##NAME##_info(); },                           \
                    [&] { info->clear_##NAME##_info(); })
    EXERCISE_FAMILY(csrsv);
    EXERCISE_FAMILY(csrsm);
    EXERCISE_FAMILY(csrilu0);
    EXERCISE_FAMILY(csric0);
    EXERCISE_FAMILY(csrildlt0);
    EXERCISE_FAMILY(bsrsv);
    EXERCISE_FAMILY(bsrsm);
    EXERCISE_FAMILY(bsrilu0);
    EXERCISE_FAMILY(bsric0);
    EXERCISE_FAMILY(ellsv);
#undef EXERCISE_FAMILY

    // The numeric boost holder, through a constant reference too.
    const rocsparse_mat_info cinfo = info;
    EXPECT_NE(cinfo->get_boost(), nullptr);
    EXPECT_EQ(cinfo->get_boost(), info->get_boost());
}

// The four (operation, fill mode) slots of the data of a triangular routine: the index
// type of the slots, copies that duplicate every filled slot, and the release of slots
// that two holders share.
TEST_F(PrecondTrmInfo, trm_data_slots_copy_and_uncouple)
{
    rocsparse::trm_data_t        empty;
    const rocsparse::trm_data_t& const_empty = empty;
    EXPECT_EQ(empty.get_indextype_J(), rocsparse_indextype_i32);
    EXPECT_EQ(const_empty.get(rocsparse_operation_none, rocsparse_fill_mode_lower), nullptr);
    EXPECT_EQ(const_empty.get(rocsparse_operation_transpose, rocsparse_fill_mode_upper), nullptr);

    // The index type is the one of the first slot that is filled.
    {
        rocsparse::trm_data_t wide;
        auto*                 info = new rocsparse::trm_info_t();
        info->set_index_indextype(rocsparse_indextype_i64);
        wide.set(rocsparse_operation_transpose, rocsparse_fill_mode_upper, info);
        EXPECT_EQ(wide.get_indextype_J(), rocsparse_indextype_i64);
        EXPECT_EQ(wide.get(rocsparse_operation_transpose, rocsparse_fill_mode_upper), info);
    }

    // A copy duplicates the filled slots and leaves the others empty.
    rocsparse::trm_data_t source;
    auto*                 lower = new rocsparse::trm_info_t();
    auto*                 trans = new rocsparse::trm_info_t();
    fill_trm_info(*lower, 10);
    fill_trm_info(*trans, 200);
    source.set(rocsparse_operation_none, rocsparse_fill_mode_lower, lower);
    source.set(rocsparse_operation_transpose, rocsparse_fill_mode_upper, trans);

    rocsparse::trm_data_t copy;
    copy.copy(&source, handle->stream);
    auto* copied_lower = copy.get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    auto* copied_trans = copy.get(rocsparse_operation_transpose, rocsparse_fill_mode_upper);
    ASSERT_NE(copied_lower, nullptr);
    ASSERT_NE(copied_trans, nullptr);
    EXPECT_NE(copied_lower, lower);
    EXPECT_NE(copied_trans, trans);
    EXPECT_EQ(copy.get(rocsparse_operation_none, rocsparse_fill_mode_upper), nullptr);
    EXPECT_EQ(copy.get(rocsparse_operation_transpose, rocsparse_fill_mode_lower), nullptr);
    expect_same_arrays(*lower, *copied_lower);
    expect_same_arrays(*trans, *copied_trans);
    copy.copy(nullptr, handle->stream);

    // Uncoupling releases the slots that are shared with the other holder only.
    rocsparse::trm_data_t holder_a;
    rocsparse::trm_data_t holder_b;
    auto*                 shared = new rocsparse::trm_info_t();
    auto*                 own    = new rocsparse::trm_info_t();
    fill_trm_info(*shared, 30);
    fill_trm_info(*own, 40);
    holder_a.set(rocsparse_operation_none, rocsparse_fill_mode_upper, shared);
    holder_b.set(rocsparse_operation_none, rocsparse_fill_mode_upper, shared);
    holder_a.set(rocsparse_operation_transpose, rocsparse_fill_mode_lower, own);

    holder_a.uncouple(nullptr);
    holder_a.uncouple(&holder_a);
    EXPECT_EQ(holder_a.get(rocsparse_operation_none, rocsparse_fill_mode_upper), shared);
    holder_a.uncouple(&holder_b);
    EXPECT_EQ(holder_a.get(rocsparse_operation_none, rocsparse_fill_mode_upper), nullptr);
    EXPECT_EQ(holder_a.get(rocsparse_operation_transpose, rocsparse_fill_mode_lower), own);
    EXPECT_EQ(holder_b.get(rocsparse_operation_none, rocsparse_fill_mode_upper), shared);
}

TEST_F(PrecondTrmInfo, copy_mat_info_with_triangular_analysis)
{
    for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
    {
        trm_fixture f;
        ASSERT_NE(f.lower, nullptr);

        info_guard src;
        info_guard constructed; // empty destination: the trm info is copy constructed
        info_guard assigned; // destination that already holds the data: assignment
        ASSERT_TRUE(src.info && constructed.info && assigned.info);

        size_t buffer_size = 0;
        analyze(handle, f, src.info, true, trans, buffer_size, nullptr);
        analyze(handle, f, src.info, false, trans, buffer_size, nullptr);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        analyze(handle, f, src.info, true, trans, buffer_size, buffer.ptr);
        analyze(handle, f, src.info, false, trans, buffer_size, buffer.ptr);

        const std::string tag = (trans == rocsparse_operation_none) ? "none" : "transpose";
        check_solve(handle, f, src.info, true, trans, buffer.ptr, "src lower " + tag);
        check_solve(handle, f, src.info, false, trans, buffer.ptr, "src upper " + tag);

        ASSERT_EQ(rocsparse_copy_mat_info(constructed.info, src.info), rocsparse_status_success);
        check_solve(handle, f, constructed.info, true, trans, buffer.ptr, "copy lower " + tag);
        check_solve(handle, f, constructed.info, false, trans, buffer.ptr, "copy upper " + tag);

        ASSERT_EQ(rocsparse_copy_mat_info(assigned.info, src.info), rocsparse_status_success);
        ASSERT_EQ(rocsparse_copy_mat_info(assigned.info, src.info), rocsparse_status_success);
        check_solve(handle, f, assigned.info, true, trans, buffer.ptr, "assigned lower " + tag);
        check_solve(handle, f, assigned.info, false, trans, buffer.ptr, "assigned upper " + tag);

        // The source is unaffected by the copies.
        check_solve(handle, f, src.info, true, trans, buffer.ptr, "src again " + tag);

        // Release the source's analysis data and the source itself. A copy that shared the
        // device arrays of the source would now solve with freed memory; a deep copy does
        // not notice.
        EXPECT_EQ(rocsparse_csrsv_clear(handle, f.lower, src.info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_csrsv_clear(handle, f.upper, src.info), rocsparse_status_success);
        src.reset();
        // Scribble over freed allocations a shallow copy might still point to.
        {
            device_vector<char> filler{buffer_size ? buffer_size : size_t{1}};
            ASSERT_TRUE(filler.ptr);
            ASSERT_EQ(hipMemset(filler.ptr, 0xff, buffer_size ? buffer_size : size_t{1}),
                      hipSuccess);
        }
        check_solve(handle,
                    f,
                    constructed.info,
                    true,
                    trans,
                    buffer.ptr,
                    "constructed after src gone lower " + tag);
        check_solve(handle,
                    f,
                    constructed.info,
                    false,
                    trans,
                    buffer.ptr,
                    "constructed after src gone upper " + tag);
        check_solve(handle,
                    f,
                    assigned.info,
                    true,
                    trans,
                    buffer.ptr,
                    "assigned after src gone lower " + tag);
        check_solve(handle,
                    f,
                    assigned.info,
                    false,
                    trans,
                    buffer.ptr,
                    "assigned after src gone upper " + tag);

        for(const auto info : {constructed.info, assigned.info})
        {
            EXPECT_EQ(rocsparse_csrsv_clear(handle, f.lower, info), rocsparse_status_success);
            EXPECT_EQ(rocsparse_csrsv_clear(handle, f.upper, info), rocsparse_status_success);
        }
    }
}
