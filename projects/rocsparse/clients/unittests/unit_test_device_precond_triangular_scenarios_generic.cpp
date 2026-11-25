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

// Scenarios of the triangular analysis data that the generic routines keep on the
// mat_info of a sparse matrix: sptrsv, sptrsm, spilu0, spic0 and spildlt0 with the reuse
// policy share the analysis of the matrix, with the force policy their descriptor
// keeps one of its own; spsv and spsm keep theirs on the matrix. The tests cover one
// routine reusing the analysis of another, outputs that belong to their descriptor,
// a forced analysis leaving the matrix alone, preprocess and compute of spsv and spsm,
// and descriptors outliving the matrix. The conventions are those of
// unit_test_device_precond_triangular_scenarios.cpp.

#include "unit_test_device_precond_triangular_common.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace triangular_scenarios;

namespace
{
    using triangular_scenarios::name;

    constexpr rocsparse_datatype f64 = rocsparse_datatype_f64_r;

    // The matrix with the entry (row, from) moved to (row, to), keeping the row sorted:
    // the same size and number of entries on another pattern.
    csr_matrix<double> with_entry_moved(csr_matrix<double> a, int32_t row, int32_t from, int32_t to)
    {
        std::vector<std::pair<int32_t, double>> entries;
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
            entries.emplace_back(a.ind[p] == from ? to : a.ind[p], a.val[p]);
        std::sort(entries.begin(), entries.end());
        for(int32_t p = a.ptr[row]; p < a.ptr[row + 1]; ++p)
        {
            a.ind[p] = entries[p - a.ptr[row]].first;
            a.val[p] = entries[p - a.ptr[row]].second;
        }
        return a;
    }

    // The matrix with the values `val` on its pattern.
    csr_matrix<double> with_values(csr_matrix<double> a, const std::vector<double>& val)
    {
        a.val = val;
        return a;
    }

    struct dense_vector
    {
        device_vector<double> data;
        rocsparse_dnvec_descr d = nullptr;

        explicit dense_vector(const std::vector<double>& v)
            : data(v)
        {
            EXPECT_EQ(
                rocsparse_create_dnvec_descr(&d, static_cast<int64_t>(v.size()), data.ptr, f64),
                rocsparse_status_success);
        }

        dense_vector(const dense_vector&) = delete;
        dense_vector& operator=(const dense_vector&) = delete;

        ~dense_vector()
        {
            if(d)
                (void)rocsparse_destroy_dnvec_descr(d);
        }

        std::vector<double> values() const
        {
            return to_host(data);
        }
    };

    // A column major matrix of `cols` columns of `rows` entries.
    struct dense_matrix
    {
        device_vector<double> data;
        rocsparse_dnmat_descr d = nullptr;

        dense_matrix(const std::vector<double>& v, int64_t rows, int64_t cols)
            : data(v)
        {
            EXPECT_EQ(rocsparse_create_dnmat_descr(
                          &d, rows, cols, rows, data.ptr, f64, rocsparse_order_column),
                      rocsparse_status_success);
        }

        dense_matrix(const dense_matrix&) = delete;
        dense_matrix& operator=(const dense_matrix&) = delete;

        ~dense_matrix()
        {
            if(d)
                (void)rocsparse_destroy_dnmat_descr(d);
        }

        std::vector<double> values() const
        {
            return to_host(data);
        }
    };

    // An sptrsv descriptor solving with op none and alpha 1.
    struct sptrsv_solver
    {
        rocsparse_handle       handle;
        rocsparse_sptrsv_descr descr = nullptr;
        double                 alpha = 1.0;

        sptrsv_solver(rocsparse_handle h, rocsparse_analysis_policy policy)
            : handle(h)
        {
            EXPECT_EQ(rocsparse_create_sptrsv_descr(&descr), rocsparse_status_success);
            const rocsparse_sptrsv_alg alg = rocsparse_sptrsv_alg_default;
            const rocsparse_operation  op  = rocsparse_operation_none;
            EXPECT_EQ(set(rocsparse_sptrsv_input_alg, &alg, sizeof(alg)), rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsv_input_operation, &op, sizeof(op)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsv_input_scalar_datatype, &f64, sizeof(f64)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsv_input_compute_datatype, &f64, sizeof(f64)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsv_input_analysis_policy, &policy, sizeof(policy)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsv_input_scalar_alpha, &alpha, sizeof(const void*)),
                      rocsparse_status_success);
        }

        sptrsv_solver(const sptrsv_solver&) = delete;
        sptrsv_solver& operator=(const sptrsv_solver&) = delete;

        ~sptrsv_solver()
        {
            if(descr)
                (void)rocsparse_destroy_sptrsv_descr(descr);
        }

        rocsparse_status set(rocsparse_sptrsv_input in, const void* data, size_t size)
        {
            return rocsparse_sptrsv_set_input(handle, descr, in, data, size, nullptr);
        }

        rocsparse_status run(rocsparse_spmat_descr  A,
                             const dense_vector&    x,
                             dense_vector&          y,
                             rocsparse_sptrsv_stage stage)
        {
            size_t           size = 0;
            rocsparse_status status
                = rocsparse_sptrsv_buffer_size(handle, descr, A, x.d, y.d, stage, &size, nullptr);
            if(status != rocsparse_status_success)
                return status;
            device_vector<char> buffer(size ? size : size_t{1});
            return rocsparse_sptrsv(
                handle, descr, A, x.d, y.d, stage, size, size ? buffer.ptr : nullptr, nullptr);
        }

        rocsparse_status analysis(rocsparse_spmat_descr A, const dense_vector& x, dense_vector& y)
        {
            return run(A, x, y, rocsparse_sptrsv_stage_analysis);
        }

        rocsparse_status solve(rocsparse_spmat_descr A, const dense_vector& x, dense_vector& y)
        {
            return run(A, x, y, rocsparse_sptrsv_stage_compute);
        }

        int64_t singularity_position()
        {
            int64_t position = -7;
            EXPECT_EQ(rocsparse_sptrsv_get_output(handle,
                                                  descr,
                                                  rocsparse_sptrsv_output_singularity_position,
                                                  &position,
                                                  sizeof(position),
                                                  nullptr),
                      rocsparse_status_success);
            return position;
        }

        rocsparse_singularity singularity()
        {
            rocsparse_singularity s = rocsparse_singularity_numeric_near;
            EXPECT_EQ(
                rocsparse_sptrsv_get_output(
                    handle, descr, rocsparse_sptrsv_output_singularity, &s, sizeof(s), nullptr),
                rocsparse_status_success);
            return s;
        }
    };

    // An sptrsm descriptor solving with op none for A and X and alpha 1.
    struct sptrsm_solver
    {
        rocsparse_handle       handle;
        rocsparse_sptrsm_descr descr = nullptr;
        double                 alpha = 1.0;

        sptrsm_solver(rocsparse_handle h, rocsparse_analysis_policy policy)
            : handle(h)
        {
            EXPECT_EQ(rocsparse_create_sptrsm_descr(&descr), rocsparse_status_success);
            const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
            const rocsparse_operation  op  = rocsparse_operation_none;
            EXPECT_EQ(set(rocsparse_sptrsm_input_alg, &alg, sizeof(alg)), rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsm_input_operation_A, &op, sizeof(op)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsm_input_operation_X, &op, sizeof(op)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsm_input_scalar_datatype, &f64, sizeof(f64)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsm_input_compute_datatype, &f64, sizeof(f64)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsm_input_analysis_policy, &policy, sizeof(policy)),
                      rocsparse_status_success);
            EXPECT_EQ(set(rocsparse_sptrsm_input_scalar_alpha, &alpha, sizeof(const void*)),
                      rocsparse_status_success);
        }

        sptrsm_solver(const sptrsm_solver&) = delete;
        sptrsm_solver& operator=(const sptrsm_solver&) = delete;

        ~sptrsm_solver()
        {
            if(descr)
                (void)rocsparse_destroy_sptrsm_descr(descr);
        }

        rocsparse_status set(rocsparse_sptrsm_input in, const void* data, size_t size)
        {
            return rocsparse_sptrsm_set_input(handle, descr, in, data, size, nullptr);
        }

        rocsparse_status run(rocsparse_spmat_descr  A,
                             const dense_matrix&    X,
                             dense_matrix&          Y,
                             rocsparse_sptrsm_stage stage)
        {
            size_t           size = 0;
            rocsparse_status status
                = rocsparse_sptrsm_buffer_size(handle, descr, A, X.d, Y.d, stage, &size, nullptr);
            if(status != rocsparse_status_success)
                return status;
            device_vector<char> buffer(size ? size : size_t{1});
            return rocsparse_sptrsm(
                handle, descr, A, X.d, Y.d, stage, size, size ? buffer.ptr : nullptr, nullptr);
        }
    };

    // An spilu0, spic0 or spildlt0 descriptor factorizing A in place (P = A).
    template <Kind K>
    struct factorization
    {
        rocsparse_handle handle;
        descr_guard<K>   guard;

        factorization(rocsparse_handle h, rocsparse_analysis_policy policy)
            : handle(h)
            , guard(h)
        {
            EXPECT_EQ(api<K>::create(handle, &guard.descr), rocsparse_status_success);
            const rocsparse_datatype dtype = f64;
            EXPECT_EQ((set_value<K>(handle, guard.descr, In::dtype, &dtype, sizeof(dtype))),
                      rocsparse_status_success);
            const int32_t alg = 0;
            EXPECT_EQ((set_value<K>(handle, guard.descr, In::alg, &alg, sizeof(alg))),
                      rocsparse_status_success);
            EXPECT_EQ((set_value<K>(handle, guard.descr, In::policy, &policy, sizeof(policy))),
                      rocsparse_status_success);
        }

        rocsparse_status run(rocsparse_spmat_descr A, Stage s)
        {
            size_t           size = 0;
            rocsparse_status status
                = api<K>::buffer_size(handle, guard.descr, A, A, api<K>::stage(s), &size);
            if(status != rocsparse_status_success)
                return status;
            device_vector<char> buffer(size ? size : size_t{1});
            return api<K>::run(
                handle, guard.descr, A, A, api<K>::stage(s), size, size ? buffer.ptr : nullptr);
        }

        rocsparse_status analysis(rocsparse_spmat_descr A)
        {
            return run(A, Stage::analysis);
        }

        rocsparse_status compute(rocsparse_spmat_descr A)
        {
            return run(A, Stage::compute);
        }

        rocsparse_singularity singularity()
        {
            rocsparse_singularity s = rocsparse_singularity_numeric_near;
            EXPECT_EQ(api<K>::get(handle, guard.descr, false, &s, sizeof(s)),
                      rocsparse_status_success);
            return s;
        }
    };

    const char* name(Kind k)
    {
        return (k == Kind::ilu) ? "spilu0" : (k == Kind::ic) ? "spic0" : "spildlt0";
    }

    using device_csr = dev_csr<double, int32_t, int32_t>;

    // The factorization of A by K, checked against the host.
    template <Kind K>
    void check_factorization(rocsparse_handle          handle,
                             device_csr&               d,
                             rocsparse_analysis_policy policy)
    {
        factorization<K> f(handle, policy);
        ASSERT_EQ(f.analysis(d.A), rocsparse_status_success);
        ASSERT_EQ(f.compute(d.A), rocsparse_status_success);
        check_factor<K>(d.base, to_host(d.val), run_options{}, name(K));
        EXPECT_EQ(f.singularity(), rocsparse_singularity_none);
    }

    // The solve with the lower triangle of A (with its current values), checked
    // against the host.
    void check_lower_solve(sptrsv_solver& s, device_csr& d)
    {
        const auto   a = with_values(d.base, to_host(d.val));
        const auto   b = right_hand_side(a.n);
        dense_vector x(b), y(std::vector<double>(a.n, 0.0));
        ASSERT_EQ(s.analysis(d.A, x, y), rocsparse_status_success);
        ASSERT_EQ(s.solve(d.A, x, y), rocsparse_status_success);
        expect_close(y.values(), host_lower_solve(a, b, 1), tolerance<double>(max_row_length(a)));
        EXPECT_EQ(s.singularity(), rocsparse_singularity_none);
    }

    // spsv (or spsm with `nrhs` columns) of op(L), L the lower triangle of A: the stages
    // given, then the result, whose entries start at `initial`. Returns the status of the
    // compute stage.
    template <typename D>
    rocsparse_status run_spsv(rocsparse_handle     handle,
                              D&                   d,
                              bool                 preprocess,
                              rocsparse_operation  op,
                              std::vector<double>  b,
                              std::vector<double>& y,
                              double               initial = 0.0)
    {
        const double alpha = 1.0;
        dense_vector x(b), out(std::vector<double>(b.size(), initial));
        size_t       size = 0;
        const auto   call = [&](rocsparse_spsv_stage stage, void* buffer) {
            return rocsparse_spsv(handle,
                                  op,
                                  &alpha,
                                  d.A,
                                  x.d,
                                  out.d,
                                  f64,
                                  rocsparse_spsv_alg_default,
                                  stage,
                                  &size,
                                  buffer);
        };
        EXPECT_EQ(call(rocsparse_spsv_stage_buffer_size, nullptr), rocsparse_status_success);
        device_vector<char> buffer(size ? size : size_t{1});
        if(preprocess)
            EXPECT_EQ(call(rocsparse_spsv_stage_preprocess, buffer.ptr), rocsparse_status_success);
        const rocsparse_status status = call(rocsparse_spsv_stage_compute, buffer.ptr);
        y                             = out.values();
        return status;
    }

    template <typename D>
    rocsparse_status run_spsm(rocsparse_handle     handle,
                              D&                   d,
                              bool                 preprocess,
                              rocsparse_operation  op,
                              int64_t              nrhs,
                              std::vector<double>  b,
                              std::vector<double>& c)
    {
        const double  alpha = 1.0;
        const int64_t n     = d.base.n;
        dense_matrix  B(b, n, nrhs), C(std::vector<double>(b.size(), 0.0), n, nrhs);
        size_t        size = 0;
        const auto    call = [&](rocsparse_spsm_stage stage, void* buffer) {
            return rocsparse_spsm(handle,
                                  op,
                                  rocsparse_operation_none,
                                  &alpha,
                                  d.A,
                                  B.d,
                                  C.d,
                                  f64,
                                  rocsparse_spsm_alg_default,
                                  stage,
                                  &size,
                                  buffer);
        };
        EXPECT_EQ(call(rocsparse_spsm_stage_buffer_size, nullptr), rocsparse_status_success);
        device_vector<char> buffer(size ? size : size_t{1});
        if(preprocess)
            EXPECT_EQ(call(rocsparse_spsm_stage_preprocess, buffer.ptr), rocsparse_status_success);
        const rocsparse_status status = call(rocsparse_spsm_stage_compute, buffer.ptr);
        c                             = C.values();
        return status;
    }

    // Row 5 of the 4 x 4 grid holds 1, 4, 5, 6 and 9; moving 6 to 0 adds the entry
    // (5, 0) to the lower triangle.
    csr_matrix<double> another_pattern(const csr_matrix<double>& a)
    {
        return with_entry_moved(a, 5, 6, 0);
    }

    // Points the matrix descriptor of `d` at the arrays of `m` in `base`, uploaded into
    // `arrays`.
    template <typename D>
    void set_pointers(D& d, const csr_matrix<double>& m, rocsparse_index_base base, D& arrays)
    {
        arrays.init(m, 1, true, base);
        ASSERT_NE(arrays.A, nullptr);
        ASSERT_EQ(rocsparse_csr_set_pointers(d.A, arrays.row.ptr, arrays.col.ptr, arrays.val.ptr),
                  rocsparse_status_success);
        d.base = m;
    }

    template <typename I, typename J>
    std::string index_types()
    {
        const auto width = [](size_t size) { return (size == 8) ? std::string("i64") : "i32"; };
        return width(sizeof(I)) + " row pointers, " + width(sizeof(J)) + " column indices";
    }
}

class PrecondTriangularScenariosGeneric : public HandleTest
{
};

// ---------------------------------------------------------------------------
// Outputs and forced analyses
// ---------------------------------------------------------------------------

namespace
{
    // The index types decide the sizes of the shared analysis arrays, and the position is
    // reported in the index base of the matrix.
    template <typename I, typename J>
    void check_sptrsv_outputs_belong_to_the_descriptor(rocsparse_handle     handle,
                                                       rocsparse_index_base base)
    {
        SCOPED_TRACE((index_types<I, J>()));
        const int32_t k    = 4;
        const auto    want = static_cast<int32_t>(position_in_base(k, base));
        const auto    a    = make_grid_matrix<double>(4, true);
        const auto    b    = right_hand_side(a.n);
        for(const auto policy : policies)
        {
            SCOPED_TRACE(name(policy));
            dev_csr<double, I, J> d;
            d.init(with_zero_diagonal(a, k), 1, true, base);
            ASSERT_NE(d.A, nullptr);

            sptrsv_solver first(handle, policy);
            dense_vector  x(b), y(std::vector<double>(a.n, 0.0));
            ASSERT_EQ(first.analysis(d.A, x, y), rocsparse_status_success);
            ASSERT_EQ(first.solve(d.A, x, y), rocsparse_status_success);
            ASSERT_EQ(first.singularity_position(), want);

            ASSERT_EQ(
                hipMemcpy(
                    d.val.ptr, a.val.data(), sizeof(double) * a.val.size(), hipMemcpyHostToDevice),
                hipSuccess);
            sptrsv_solver second(handle, policy);
            ASSERT_EQ(second.analysis(d.A, x, y), rocsparse_status_success);
            ASSERT_EQ(second.solve(d.A, x, y), rocsparse_status_success);
            EXPECT_EQ(second.singularity_position(), -1);

            const auto position = static_cast<int32_t>(first.singularity_position());
            if(policy == rocsparse_analysis_policy_reuse)
                check(position,
                      known_defect(no_pivot,
                                   want,
                                   "with reuse both descriptors read the singularity of the "
                                   "csrsv data of the matrix, which the second solve resets"),
                      "singularity_position of the first descriptor");
            else
                check(position, is(want), "singularity_position of the first descriptor");
        }
    }
}

// The outputs of an sptrsv descriptor are those of its own solve: a later solve by
// another descriptor on the same matrix does not change them.
TEST_F(PrecondTriangularScenariosGeneric, sptrsv_outputs_belong_to_the_descriptor)
{
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        check_sptrsv_outputs_belong_to_the_descriptor<int32_t, int32_t>(handle, base);
        check_sptrsv_outputs_belong_to_the_descriptor<int64_t, int32_t>(handle, base);
        check_sptrsv_outputs_belong_to_the_descriptor<int64_t, int64_t>(handle, base);
    }
}

// A forced analysis keeps its data in the descriptor: the matrix has no analysis
// afterwards, and the reuse policy of another routine runs an analysis of its own.
TEST_F(PrecondTriangularScenariosGeneric, forced_analysis_leaves_the_matrix_alone)
{
    const auto a = make_grid_matrix<double>(4, false);
    {
        SCOPED_TRACE("spilu0 reusing after a forced sptrsv");
        device_csr d;
        d.init(a, 1, true);
        sptrsv_solver s(handle, rocsparse_analysis_policy_force);
        ASSERT_NO_FATAL_FAILURE(check_lower_solve(s, d));
        ASSERT_NO_FATAL_FAILURE(
            check_factorization<Kind::ilu>(handle, d, rocsparse_analysis_policy_reuse));
    }
    for(const auto k : {Kind::ilu, Kind::ic, Kind::ildlt})
    {
        SCOPED_TRACE(::testing::Message() << "sptrsv reusing after a forced " << name(k));
        device_csr d;
        d.init(a, 1, true);
        if(k == Kind::ilu)
            ASSERT_NO_FATAL_FAILURE(
                check_factorization<Kind::ilu>(handle, d, rocsparse_analysis_policy_force));
        else if(k == Kind::ic)
            ASSERT_NO_FATAL_FAILURE(
                check_factorization<Kind::ic>(handle, d, rocsparse_analysis_policy_force));
        else
            ASSERT_NO_FATAL_FAILURE(
                check_factorization<Kind::ildlt>(handle, d, rocsparse_analysis_policy_force));
        sptrsv_solver s(handle, rocsparse_analysis_policy_reuse);
        ASSERT_NO_FATAL_FAILURE(check_lower_solve(s, d));
    }
}

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
// sptrsm with one column after an sptrsv analysis of the matrix: with reuse the
// analysis is stored for csrsv, so the compute of sptrsm finds none.
TEST_F(PrecondTriangularScenariosGeneric, sptrsm_one_column_after_sptrsv)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n);
    device_csr d;
    d.init(a, 1, true);
    sptrsv_solver s(handle, rocsparse_analysis_policy_reuse);
    dense_vector  x(b), y(std::vector<double>(a.n, 0.0));
    ASSERT_EQ(s.analysis(d.A, x, y), rocsparse_status_success);

    sptrsm_solver m(handle, rocsparse_analysis_policy_reuse);
    dense_matrix  X(b, a.n, 1), Y(std::vector<double>(a.n, 0.0), a.n, 1);
    ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_analysis), rocsparse_status_success);
    check(m.run(d.A, X, Y, rocsparse_sptrsm_stage_compute),
          known_defect(rocsparse_status_invalid_pointer,
                       rocsparse_status_success,
                       "with one column the reused analysis is stored for csrsv, so the "
                       "compute of sptrsm finds none"),
          "sptrsm compute with one column");
}
#endif

// ---------------------------------------------------------------------------
// spsv and spsm
// ---------------------------------------------------------------------------

// The compute stage of spsv needs the preprocess: without it spsv returns
// rocsparse_status_invalid_pointer and leaves y alone.
TEST_F(PrecondTriangularScenariosGeneric, spsv_compute_needs_the_preprocess)
{
    const auto a = make_grid_matrix<double>(4, true);
    device_csr d;
    d.init(a, 1, true);
    std::vector<double> y;
    check(run_spsv(handle, d, false, rocsparse_operation_none, right_hand_side(a.n), y, 7.0),
          is(rocsparse_status_invalid_pointer),
          "spsv compute without the preprocess");
    EXPECT_EQ(y, std::vector<double>(a.n, 7.0));
}

namespace
{
    template <typename I, typename J>
    void check_preprocess_of_a_new_pattern(rocsparse_handle handle, rocsparse_index_base base)
    {
        SCOPED_TRACE((index_types<I, J>()));
        const auto a     = make_grid_matrix<double>(4, true);
        const auto a_new = another_pattern(a);
        const auto tol   = tolerance<double>(max_row_length(a));
        const auto op    = rocsparse_operation_transpose;
        for(const int32_t nrhs : {0, 2})
        {
            const bool spsv = (nrhs == 0);
            SCOPED_TRACE(spsv ? "spsv" : "spsm");
            const int32_t         cols = spsv ? 1 : nrhs;
            const auto            b    = right_hand_side(a.n, cols);
            dev_csr<double, I, J> d, arrays;
            std::vector<double>   y;
            d.init(a, 1, true, base);
            ASSERT_NE(d.A, nullptr);
            const auto run = [&] {
                return spsv ? run_spsv(handle, d, true, op, b, y)
                            : run_spsm(handle, d, true, op, nrhs, b, y);
            };
            ASSERT_EQ(run(), rocsparse_status_success);
            expect_close(y, host_lower_transposed_solve(a, b, cols), tol);

            ASSERT_NO_FATAL_FAILURE(set_pointers(d, a_new, base, arrays));
            ASSERT_EQ(run(), rocsparse_status_success);
            const bool documented = all_close(y, host_lower_transposed_solve(a_new, b, cols), tol);
            check(documented,
                  known_defect(false,
                               true,
                               "the preprocess skips the analysis when the matrix already has "
                               "one, so the compute uses the analysis of the old pattern"),
                  "solve with the new pattern matches the host");
            // The stale analysis gathers the new values through the permutation of the
            // old pattern: the solve is that of the old pattern holding the new values
            // position by position.
            if(!documented)
            {
                SCOPED_TRACE("solve with the analysis of the old pattern");
                expect_close(
                    y, host_lower_transposed_solve(with_values(a, a_new.val), b, cols), tol);
            }
        }
    }
}

// The preprocess analyses the matrix it is given: after rocsparse_csr_set_pointers to
// another pattern, the preprocess and compute solve with the new pattern. The solve
// with the transpose reads the transposed structure that the analysis stored (the
// solve with op none reads the structure from the matrix).
TEST_F(PrecondTriangularScenariosGeneric, spsv_and_spsm_preprocess_a_new_pattern)
{
    for(const auto base : bases)
    {
        SCOPED_TRACE(name(base));
        check_preprocess_of_a_new_pattern<int32_t, int32_t>(handle, base);
        check_preprocess_of_a_new_pattern<int64_t, int32_t>(handle, base);
        check_preprocess_of_a_new_pattern<int64_t, int64_t>(handle, base);
    }
}

// ---------------------------------------------------------------------------
// Descriptors outliving the matrix
// ---------------------------------------------------------------------------

// A descriptor that shares the analysis of the matrix keeps it when the matrix
// descriptor (and with it its mat_info) is destroyed first.
TEST_F(PrecondTriangularScenariosGeneric, descriptors_outlive_the_matrix)
{
    const auto a = make_grid_matrix<double>(4, false);
    const auto b = right_hand_side(a.n);
    {
        SCOPED_TRACE("sptrsv");
        device_csr d;
        d.init(a, 1, true);
        sptrsv_solver s(handle, rocsparse_analysis_policy_reuse);
        dense_vector  x(b), y(std::vector<double>(a.n, 0.0));
        ASSERT_EQ(s.analysis(d.A, x, y), rocsparse_status_success);
        ASSERT_EQ(s.solve(d.A, x, y), rocsparse_status_success);
        ASSERT_EQ(rocsparse_destroy_spmat_descr(d.A), rocsparse_status_success);
        d.A = nullptr;
        EXPECT_EQ(s.singularity(), rocsparse_singularity_none);
        EXPECT_EQ(s.singularity_position(), -1);
        EXPECT_EQ(rocsparse_destroy_sptrsv_descr(s.descr), rocsparse_status_success);
        s.descr = nullptr;
    }
    {
        SCOPED_TRACE("spilu0");
        device_csr d;
        d.init(a, 1, true);
        factorization<Kind::ilu> f(handle, rocsparse_analysis_policy_reuse);
        ASSERT_EQ(f.analysis(d.A), rocsparse_status_success);
        ASSERT_EQ(f.compute(d.A), rocsparse_status_success);
        ASSERT_EQ(rocsparse_destroy_spmat_descr(d.A), rocsparse_status_success);
        d.A = nullptr;
        EXPECT_EQ(f.singularity(), rocsparse_singularity_none);
        EXPECT_EQ(api<Kind::ilu>::destroy(handle, f.guard.descr), rocsparse_status_success);
        f.guard.descr = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Zero pivot and boost outputs
// ---------------------------------------------------------------------------

// The zero pivot output of sptrsm comes with rocsparse_status_zero_pivot when there is
// one, and with success and -1 otherwise.
TEST_F(PrecondTriangularScenariosGeneric, sptrsm_zero_pivot_position)
{
    const int32_t r     = 4;
    const auto    clean = make_grid_matrix<double>(3, false);
    for(const bool singular : {true, false})
    {
        SCOPED_TRACE(singular ? "missing diagonal" : "regular");
        const auto a = singular ? without_diagonal(clean, r) : clean;
        const auto b = right_hand_side(a.n, 2);
        device_csr d;
        d.init(a, 1, true);
        sptrsm_solver m(handle, rocsparse_analysis_policy_force);
        dense_matrix  X(b, a.n, 2), Y(std::vector<double>(b.size(), 0.0), a.n, 2);
        ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_analysis), rocsparse_status_success);
        ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_compute), rocsparse_status_success);
        int64_t position = -7;
        check(rocsparse_sptrsm_get_output(handle,
                                          m.descr,
                                          rocsparse_sptrsm_output_zero_pivot_position,
                                          &position,
                                          sizeof(position),
                                          nullptr),
              is(singular ? rocsparse_status_zero_pivot : rocsparse_status_success),
              "sptrsm zero pivot output");
        EXPECT_EQ(position, singular ? r : -1);
    }
}

namespace
{
    // The factor of `a` by K (forced analysis), with the first pivot boosted when
    // `boost` is set.
    template <Kind K>
    std::vector<double> factor(rocsparse_handle handle, const csr_matrix<double>& a, bool boost)
    {
        device_csr d;
        d.init(a, 1, true);
        run_options opt;
        opt.policy    = rocsparse_analysis_policy_force;
        opt.boost     = boost;
        opt.boost_tol = 0.1;
        opt.boost_val = 4.0;
        run_result<double> out;
        run_factorization<K, double>(handle, d.A, d.val, 1, opt, out);
        EXPECT_EQ(out.compute_status, rocsparse_status_success) << name(K);
        return out.values;
    }
}

// The boost inputs replace a pivot with |pivot| <= boost tolerance: the factor with the
// boost differs from the one without, for spilu0 and for spic0.
TEST_F(PrecondTriangularScenariosGeneric, numeric_boost_takes_effect)
{
    csr_matrix<double> a;
    a.n   = 3;
    a.ptr = {0, 2, 5, 7};
    a.ind = {0, 1, 0, 1, 2, 1, 2};
    a.val = {1e-4, 1e-3, 1e-3, 4, 1, 1, 4};
    check(factor<Kind::ilu>(handle, a, true) == factor<Kind::ilu>(handle, a, false),
          is(false),
          "the spilu0 factor with the boost equals the one without");
    check(factor<Kind::ic>(handle, a, true) == factor<Kind::ic>(handle, a, false),
          known_defect(true, false, "spic0 stores the boost inputs but factorizes without them"),
          "the spic0 factor with the boost equals the one without");
}

// ---------------------------------------------------------------------------
// Statuses
// ---------------------------------------------------------------------------

namespace
{
    constexpr rocsparse_operation conj_transpose = rocsparse_operation_conjugate_transpose;

    // spsv through the stage `last`: the status of `last`, or of the first stage before
    // it that fails.
    rocsparse_status spsv_until(rocsparse_handle      handle,
                                rocsparse_operation   trans,
                                rocsparse_spmat_descr A,
                                rocsparse_dnvec_descr x,
                                rocsparse_dnvec_descr y,
                                rocsparse_spsv_stage  last,
                                rocsparse_spsv_alg    alg = rocsparse_spsv_alg_default)
    {
        const double alpha = 1.0;
        size_t       size  = 0;
        const auto   call  = [&](rocsparse_spsv_stage stage, void* buffer) {
            return rocsparse_spsv(handle, trans, &alpha, A, x, y, f64, alg, stage, &size, buffer);
        };
        rocsparse_status status = call(rocsparse_spsv_stage_buffer_size, nullptr);
        if(status != rocsparse_status_success || last == rocsparse_spsv_stage_buffer_size)
            return status;
        device_vector<char> buffer(size ? size : size_t{1});
        status = call(rocsparse_spsv_stage_preprocess, buffer.ptr);
        if(status != rocsparse_status_success || last == rocsparse_spsv_stage_preprocess)
            return status;
        return call(rocsparse_spsv_stage_compute, buffer.ptr);
    }

    rocsparse_status spsm_until(rocsparse_handle      handle,
                                rocsparse_operation   trans_A,
                                rocsparse_operation   trans_B,
                                rocsparse_spmat_descr A,
                                rocsparse_dnmat_descr B,
                                rocsparse_dnmat_descr C,
                                rocsparse_spsm_stage  last,
                                rocsparse_spsm_alg    alg = rocsparse_spsm_alg_default)
    {
        const double alpha = 1.0;
        size_t       size  = 0;
        const auto   call  = [&](rocsparse_spsm_stage stage, void* buffer) {
            return rocsparse_spsm(
                handle, trans_A, trans_B, &alpha, A, B, C, f64, alg, stage, &size, buffer);
        };
        rocsparse_status status = call(rocsparse_spsm_stage_buffer_size, nullptr);
        if(status != rocsparse_status_success || last == rocsparse_spsm_stage_buffer_size)
            return status;
        device_vector<char> buffer(size ? size : size_t{1});
        status = call(rocsparse_spsm_stage_preprocess, buffer.ptr);
        if(status != rocsparse_status_success || last == rocsparse_spsm_stage_preprocess)
            return status;
        return call(rocsparse_spsm_stage_compute, buffer.ptr);
    }

    // set_input and get_output of K with a wrong size.
    template <Kind K>
    void check_factorization_sizes(rocsparse_handle handle)
    {
        SCOPED_TRACE(name(K));
        device_csr d;
        d.init(make_grid_matrix<double>(3, false), 1, true);
        factorization<K> f(handle, rocsparse_analysis_policy_force);
        const auto       expected
            = known_defect(rocsparse_status_invalid_value,
                           rocsparse_status_invalid_size,
                           "a wrong size that is not zero is reported as an invalid value");
        const int32_t alg = 0;
        check(set_value<K>(handle, f.guard.descr, In::alg, &alg, 1),
              expected,
              "set_input(alg) of 1 byte");
        ASSERT_EQ(f.analysis(d.A), rocsparse_status_success);
        ASSERT_EQ(f.compute(d.A), rocsparse_status_success);
        rocsparse_singularity s = rocsparse_singularity_none;
        check(api<K>::get(handle, f.guard.descr, false, &s, 1),
              expected,
              "get_output(singularity) of 1 byte");
    }
}

// A non-null buffer of size zero is documented as invalid_pointer, like a null buffer
// of a non-zero size.
TEST_F(PrecondTriangularScenariosGeneric, sptrsv_buffer_arguments)
{
    const auto a = make_grid_matrix<double>(3, false);
    device_csr d;
    d.init(a, 1, true);
    sptrsv_solver       s(handle, rocsparse_analysis_policy_force);
    dense_vector        x(right_hand_side(a.n)), y(std::vector<double>(a.n, 0.0));
    device_vector<char> buffer(64);
    check(rocsparse_sptrsv(handle,
                           s.descr,
                           d.A,
                           x.d,
                           y.d,
                           rocsparse_sptrsv_stage_analysis,
                           0,
                           buffer.ptr,
                           nullptr),
          known_defect(rocsparse_status_invalid_size,
                       rocsparse_status_invalid_pointer,
                       "a non-null buffer of size zero is reported as an invalid size"),
          "sptrsv with a buffer of size 0");
    check(
        rocsparse_sptrsv(
            handle, s.descr, d.A, x.d, y.d, rocsparse_sptrsv_stage_analysis, 64, nullptr, nullptr),
        is(rocsparse_status_invalid_pointer),
        "sptrsv with a null buffer of size 64");
}

// A wrong size of an input or an output is documented as invalid_size; get_output
// reports a wrong size that is not zero as invalid_value. Once a stage ran, the analysis
// policy can no longer be set.
TEST_F(PrecondTriangularScenariosGeneric, sptrsv_input_and_output_sizes)
{
    const auto a = make_grid_matrix<double>(3, false);
    device_csr d;
    d.init(a, 1, true);
    sptrsv_solver              s(handle, rocsparse_analysis_policy_force);
    const rocsparse_sptrsv_alg alg = rocsparse_sptrsv_alg_default;
    check(s.set(rocsparse_sptrsv_input_alg, &alg, 1),
          is(rocsparse_status_invalid_size),
          "set_input(alg) of 1 byte");

    dense_vector x(right_hand_side(a.n)), y(std::vector<double>(a.n, 0.0));
    ASSERT_EQ(s.analysis(d.A, x, y), rocsparse_status_success);
    ASSERT_EQ(s.solve(d.A, x, y), rocsparse_status_success);
    const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
    check(s.set(rocsparse_sptrsv_input_analysis_policy, &policy, sizeof(policy)),
          is(rocsparse_status_invalid_value),
          "set_input(analysis_policy) after the analysis");

    rocsparse_singularity singularity = rocsparse_singularity_none;
    int64_t               position    = -7;
    const auto            get         = [&](rocsparse_sptrsv_output out, void* data, size_t size) {
        return rocsparse_sptrsv_get_output(handle, s.descr, out, data, size, nullptr);
    };
    check(get(rocsparse_sptrsv_output_singularity, &singularity, 0),
          is(rocsparse_status_invalid_size),
          "get_output(singularity) of 0 bytes");
    const auto expected
        = known_defect(rocsparse_status_invalid_value,
                       rocsparse_status_invalid_size,
                       "a wrong size that is not zero is reported as an invalid value");
    check(get(rocsparse_sptrsv_output_singularity, &singularity, 1),
          expected,
          "get_output(singularity) of 1 byte");
    check(get(rocsparse_sptrsv_output_singularity_position, &position, 4),
          expected,
          "get_output(singularity_position) of 4 bytes");
}

// The same for sptrsm. A non-null buffer of size zero gives invalid_size, which is not
// among the documented statuses of rocsparse_sptrsm.
TEST_F(PrecondTriangularScenariosGeneric, sptrsm_input_and_output_sizes)
{
    const auto a = make_grid_matrix<double>(3, false);
    device_csr d;
    d.init(a, 1, true);
    sptrsm_solver              m(handle, rocsparse_analysis_policy_force);
    const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
    check(m.set(rocsparse_sptrsm_input_alg, &alg, 1),
          is(rocsparse_status_invalid_size),
          "set_input(alg) of 1 byte");

    const auto   b = right_hand_side(a.n, 2);
    dense_matrix X(b, a.n, 2), Y(std::vector<double>(b.size(), 0.0), a.n, 2);
#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
    device_vector<char> buffer(64);
    check(rocsparse_sptrsm(handle,
                           m.descr,
                           d.A,
                           X.d,
                           Y.d,
                           rocsparse_sptrsm_stage_analysis,
                           0,
                           buffer.ptr,
                           nullptr),
          known_defect(rocsparse_status_invalid_size,
                       rocsparse_status_invalid_pointer,
                       "a non-null buffer of size zero is reported as an invalid size"),
          "sptrsm with a buffer of size 0");
#endif
    ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_analysis), rocsparse_status_success);
    ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_compute), rocsparse_status_success);
    const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
    check(m.set(rocsparse_sptrsm_input_analysis_policy, &policy, sizeof(policy)),
          is(rocsparse_status_invalid_value),
          "set_input(analysis_policy) after the analysis");

    const auto get = [&](rocsparse_sptrsm_output out, void* data, size_t size) {
        return rocsparse_sptrsm_get_output(handle, m.descr, out, data, size, nullptr);
    };
#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
    rocsparse_singularity singularity = rocsparse_singularity_none;
    check(get(rocsparse_sptrsm_output_singularity, &singularity, 1),
          known_defect(rocsparse_status_invalid_value,
                       rocsparse_status_invalid_size,
                       "a wrong size that is not zero is reported as an invalid value"),
          "get_output(singularity) of 1 byte");
#else
    int64_t position = -7;
    check(get(rocsparse_sptrsm_output_zero_pivot_position, &position, 4),
          is(rocsparse_status_invalid_size),
          "get_output(zero_pivot_position) of 4 bytes");
#endif
}

TEST_F(PrecondTriangularScenariosGeneric, factorization_input_and_output_sizes)
{
    check_factorization_sizes<Kind::ilu>(handle);
    check_factorization_sizes<Kind::ic>(handle);
    check_factorization_sizes<Kind::ildlt>(handle);
}

// The documented statuses of rocsparse_spilu0 have no invalid_size, which a non-null
// buffer of size zero gives.
TEST_F(PrecondTriangularScenariosGeneric, spilu0_buffer_arguments)
{
    device_csr d;
    d.init(make_grid_matrix<double>(3, false), 1, true);
    factorization<Kind::ilu> f(handle, rocsparse_analysis_policy_force);
    device_vector<char>      buffer(64);
    check(api<Kind::ilu>::run(
              handle, f.guard.descr, d.A, d.A, rocsparse_spilu0_stage_analysis, 0, buffer.ptr),
          known_defect(rocsparse_status_invalid_size,
                       rocsparse_status_invalid_pointer,
                       "a non-null buffer of size zero is reported as an invalid size"),
          "spilu0 with a buffer of size 0");
}

// The pages contradict themselves and the routines: they support only none and
// transpose, not_implemented otherwise, while the sptrsv page defines op(A) = A^H for
// conjugate_transpose. sptrsv, sptrsm, spsv and spsm implement conjugate_transpose and
// solve with it (with double values, the solve with the transpose). The pages need
// reconciling with the routines; the expectations pin what they do today.
TEST_F(PrecondTriangularScenariosGeneric, conjugate_transpose)
{
    const auto a   = make_grid_matrix<double>(3, false);
    const auto b   = right_hand_side(a.n);
    const auto tol = tolerance<double>(max_row_length(a));
    {
        SCOPED_TRACE("sptrsv");
        device_csr d;
        d.init(a, 1, true);
        sptrsv_solver s(handle, rocsparse_analysis_policy_force);
        ASSERT_EQ(s.set(rocsparse_sptrsv_input_operation, &conj_transpose, sizeof(conj_transpose)),
                  rocsparse_status_success);
        dense_vector x(b), y(std::vector<double>(a.n, 0.0));
        check(s.analysis(d.A, x, y), is(rocsparse_status_success), "sptrsv analysis");
        check(s.solve(d.A, x, y), is(rocsparse_status_success), "sptrsv compute");
        expect_close(y.values(), host_lower_transposed_solve(a, b, 1), tol);
    }
    {
        SCOPED_TRACE("spsv");
        device_csr d;
        d.init(a, 1, true);
        dense_vector x(b), y(std::vector<double>(a.n, 0.0));
        check(spsv_until(handle, conj_transpose, d.A, x.d, y.d, rocsparse_spsv_stage_compute),
              is(rocsparse_status_success),
              "spsv");
        expect_close(y.values(), host_lower_transposed_solve(a, b, 1), tol);
    }
    const int64_t nrhs = 2;
    const auto    bm   = right_hand_side(a.n, nrhs);
    for(const bool on_A : {true, false})
    {
        SCOPED_TRACE(on_A ? "operation of A" : "operation of the right-hand sides");
        const auto rows = on_A ? a.n : nrhs;
        const auto cols = on_A ? nrhs : a.n;
        // The right-hand sides are `rows` x `cols`, so op(B) is their transpose when the
        // operation applies to them.
        const auto want = on_A ? host_lower_transposed_solve(a, bm, nrhs)
                               : host_lower_solve(a, transposed(bm, nrhs, a.n), nrhs);
        {
            device_csr d;
            d.init(a, 1, true);
            sptrsm_solver m(handle, rocsparse_analysis_policy_force);
            ASSERT_EQ(m.set(on_A ? rocsparse_sptrsm_input_operation_A
                                 : rocsparse_sptrsm_input_operation_X,
                            &conj_transpose,
                            sizeof(conj_transpose)),
                      rocsparse_status_success);
            dense_matrix X(bm, rows, cols), Y(std::vector<double>(bm.size(), 0.0), a.n, nrhs);
            check(m.run(d.A, X, Y, rocsparse_sptrsm_stage_analysis),
                  is(rocsparse_status_success),
                  "sptrsm analysis");
            check(m.run(d.A, X, Y, rocsparse_sptrsm_stage_compute),
                  is(rocsparse_status_success),
                  "sptrsm compute");
            expect_close(Y.values(), want, tol);
        }
        {
            device_csr d;
            d.init(a, 1, true);
            dense_matrix B(bm, rows, cols), C(std::vector<double>(bm.size(), 0.0), a.n, nrhs);
            check(spsm_until(handle,
                             on_A ? conj_transpose : rocsparse_operation_none,
                             on_A ? rocsparse_operation_none : conj_transpose,
                             d.A,
                             B.d,
                             C.d,
                             rocsparse_spsm_stage_compute),
                  is(rocsparse_status_success),
                  "spsm");
            expect_close(C.values(), want, tol);
        }
    }
}

// A trans, alg or stage outside its enumeration is rejected as invalid_value, the status
// ROCSPARSE_CHECKARG_ENUM gives throughout the library; the spsv and spsm pages list no
// invalid_value and need it added. A valid compute type that the routines do not
// support is the documented not_implemented.
TEST_F(PrecondTriangularScenariosGeneric, spsv_and_spsm_invalid_and_unsupported_values)
{
    const auto a = make_grid_matrix<double>(3, false);
    device_csr d;
    d.init(a, 1, true);
    const auto          expected = is(rocsparse_status_invalid_value);
    const auto          bad_op   = static_cast<rocsparse_operation>(77);
    const auto          none     = rocsparse_operation_none;
    const auto          f32      = rocsparse_datatype_f32_r;
    const double        alpha    = 1.0;
    size_t              size     = 64;
    device_vector<char> buffer(size);
    {
        SCOPED_TRACE("spsv");
        dense_vector x(right_hand_side(a.n)), y(std::vector<double>(a.n, 0.0));
        check(rocsparse_spsv(handle,
                             none,
                             &alpha,
                             d.A,
                             x.d,
                             y.d,
                             f32,
                             rocsparse_spsv_alg_default,
                             rocsparse_spsv_stage_buffer_size,
                             &size,
                             nullptr),
              is(rocsparse_status_not_implemented),
              "compute type f32 for double data");
        check(spsv_until(handle, bad_op, d.A, x.d, y.d, rocsparse_spsv_stage_buffer_size),
              expected,
              "trans 77");
        check(spsv_until(handle,
                         none,
                         d.A,
                         x.d,
                         y.d,
                         rocsparse_spsv_stage_buffer_size,
                         static_cast<rocsparse_spsv_alg>(99)),
              expected,
              "alg 99");
        check(rocsparse_spsv(handle,
                             none,
                             &alpha,
                             d.A,
                             x.d,
                             y.d,
                             f64,
                             rocsparse_spsv_alg_default,
                             static_cast<rocsparse_spsv_stage>(99),
                             &size,
                             buffer.ptr),
              expected,
              "stage 99");
    }
    {
        SCOPED_TRACE("spsm");
        const auto   bm = right_hand_side(a.n, 2);
        dense_matrix B(bm, a.n, 2), C(std::vector<double>(bm.size(), 0.0), a.n, 2);
        check(rocsparse_spsm(handle,
                             none,
                             none,
                             &alpha,
                             d.A,
                             B.d,
                             C.d,
                             f32,
                             rocsparse_spsm_alg_default,
                             rocsparse_spsm_stage_buffer_size,
                             &size,
                             nullptr),
              is(rocsparse_status_not_implemented),
              "compute type f32 for double data");
        check(spsm_until(handle, bad_op, none, d.A, B.d, C.d, rocsparse_spsm_stage_buffer_size),
              expected,
              "trans_A 77");
        check(spsm_until(handle,
                         none,
                         none,
                         d.A,
                         B.d,
                         C.d,
                         rocsparse_spsm_stage_buffer_size,
                         static_cast<rocsparse_spsm_alg>(99)),
              expected,
              "alg 99");
        check(rocsparse_spsm(handle,
                             none,
                             none,
                             &alpha,
                             d.A,
                             B.d,
                             C.d,
                             f64,
                             rocsparse_spsm_alg_default,
                             static_cast<rocsparse_spsm_stage>(99),
                             &size,
                             buffer.ptr),
              expected,
              "stage 99");
    }
}

// The buffer holds nothing between the preprocess and the compute, for every operation:
// the compute with another buffer, after the first one is freed, solves correctly.
TEST_F(PrecondTriangularScenariosGeneric, spsv_buffer_is_not_persistent)
{
    const auto a   = make_grid_matrix<double>(4, false);
    const auto b   = right_hand_side(a.n);
    const auto tol = tolerance<double>(max_row_length(a));
    for(const auto op : {rocsparse_operation_none, rocsparse_operation_transpose})
    {
        SCOPED_TRACE(op == rocsparse_operation_none ? "op none" : "op transpose");
        device_csr d;
        d.init(a, 1, true);
        dense_vector x(b), y(std::vector<double>(a.n, 0.0));
        const double alpha = 1.0;
        size_t       size  = 0;
        const auto   call  = [&](rocsparse_spsv_stage stage, void* buffer) {
            return rocsparse_spsv(handle,
                                  op,
                                  &alpha,
                                  d.A,
                                  x.d,
                                  y.d,
                                  f64,
                                  rocsparse_spsv_alg_default,
                                  stage,
                                  &size,
                                  buffer);
        };
        ASSERT_EQ(call(rocsparse_spsv_stage_buffer_size, nullptr), rocsparse_status_success);
        const size_t                         n = size ? size : size_t{1};
        std::unique_ptr<device_vector<char>> first(new device_vector<char>(n));
        ASSERT_EQ(call(rocsparse_spsv_stage_preprocess, first->ptr), rocsparse_status_success);
        device_vector<char> second(std::vector<char>(n, static_cast<char>(0xff)));
        first.reset();
        ASSERT_EQ(call(rocsparse_spsv_stage_compute, second.ptr), rocsparse_status_success);
        expect_close(y.values(),
                     (op == rocsparse_operation_none) ? host_lower_solve(a, b, 1)
                                                      : host_lower_transposed_solve(a, b, 1),
                     tol);
    }
}

// spsv and spsm do not support batches: a strided batch is not_implemented.
TEST_F(PrecondTriangularScenariosGeneric, spsv_and_spsm_reject_batches)
{
    const auto a = make_grid_matrix<double>(3, false);
    const auto n = static_cast<int64_t>(a.n);
    device_csr d;
    d.init(a, 1, true);
    {
        device_vector<double> xs(right_hand_side(a.n, 2)), ys(std::vector<double>(2 * n, 0.0));
        rocsparse_dnvec_descr x = nullptr, y = nullptr;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&x, n, xs.ptr, f64), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&y, n, ys.ptr, f64), rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnvec_set_strided_batch(x, 2, n), rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnvec_set_strided_batch(y, 2, n), rocsparse_status_success);
        check(spsv_until(handle, rocsparse_operation_none, d.A, x, y, rocsparse_spsv_stage_compute),
              is(rocsparse_status_not_implemented),
              "spsv of a batch of 2");
        (void)rocsparse_destroy_dnvec_descr(x);
        (void)rocsparse_destroy_dnvec_descr(y);
    }
    {
        const int64_t         nrhs = 2;
        device_vector<double> bs(right_hand_side(a.n, 2 * nrhs)),
            cs(std::vector<double>(2 * nrhs * n, 0.0));
        rocsparse_dnmat_descr B = nullptr, C = nullptr;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&B, n, nrhs, n, bs.ptr, f64, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&C, n, nrhs, n, cs.ptr, f64, rocsparse_order_column),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnmat_set_strided_batch(B, 2, nrhs * n), rocsparse_status_success);
        EXPECT_EQ(rocsparse_dnmat_set_strided_batch(C, 2, nrhs * n), rocsparse_status_success);
        check(spsm_until(handle,
                         rocsparse_operation_none,
                         rocsparse_operation_none,
                         d.A,
                         B,
                         C,
                         rocsparse_spsm_stage_compute),
              is(rocsparse_status_not_implemented),
              "spsm of a batch of 2");
        (void)rocsparse_destroy_dnmat_descr(B);
        (void)rocsparse_destroy_dnmat_descr(C);
    }
}

// ---------------------------------------------------------------------------
// Sequences that crash
// ---------------------------------------------------------------------------

namespace
{
    // Each reuse below copies the analysis that another routine left on the matrix and
    // returns before allocating the zero pivot of the routine that reuses it, which
    // then faults on it.
    template <Kind K>
    void check_sptrsv_after(rocsparse_handle handle)
    {
        device_csr d;
        d.init(make_grid_matrix<double>(4, false), 1, true);
        ASSERT_NO_FATAL_FAILURE(check_factorization<K>(handle, d, rocsparse_analysis_policy_reuse));
        sptrsv_solver s(handle, rocsparse_analysis_policy_reuse);
        ASSERT_NO_FATAL_FAILURE(check_lower_solve(s, d));
    }

    template <Kind K>
    void check_factorization_after_sptrsv(rocsparse_handle handle)
    {
        const auto a = make_grid_matrix<double>(4, false);
        const auto b = right_hand_side(a.n);
        device_csr d;
        d.init(a, 1, true);
        sptrsv_solver s(handle, rocsparse_analysis_policy_reuse);
        dense_vector  x(b), y(std::vector<double>(a.n, 0.0));
        ASSERT_EQ(s.analysis(d.A, x, y), rocsparse_status_success);
        ASSERT_NO_FATAL_FAILURE(check_factorization<K>(handle, d, rocsparse_analysis_policy_reuse));
    }
}

TEST_F(PrecondTriangularScenariosGeneric, DISABLED_sptrsv_reusing_an_spilu0_analysis)
{
    check_sptrsv_after<Kind::ilu>(handle);
}

TEST_F(PrecondTriangularScenariosGeneric, DISABLED_sptrsv_reusing_an_spic0_analysis)
{
    check_sptrsv_after<Kind::ic>(handle);
}

TEST_F(PrecondTriangularScenariosGeneric, DISABLED_spilu0_reusing_an_sptrsv_analysis)
{
    check_factorization_after_sptrsv<Kind::ilu>(handle);
}

TEST_F(PrecondTriangularScenariosGeneric, DISABLED_spic0_reusing_an_sptrsv_analysis)
{
    check_factorization_after_sptrsv<Kind::ic>(handle);
}

TEST_F(PrecondTriangularScenariosGeneric, DISABLED_spildlt0_reusing_an_sptrsv_analysis)
{
    check_factorization_after_sptrsv<Kind::ildlt>(handle);
}

// sptrsm with two columns reuses the analysis of sptrsv and faults on its own zero
// pivot.
TEST_F(PrecondTriangularScenariosGeneric, DISABLED_sptrsm_reusing_an_sptrsv_analysis)
{
    const auto a = make_grid_matrix<double>(4, true);
    const auto b = right_hand_side(a.n, 2);
    device_csr d;
    d.init(a, 1, true);
    sptrsv_solver s(handle, rocsparse_analysis_policy_reuse);
    dense_vector  x(right_hand_side(a.n)), y(std::vector<double>(a.n, 0.0));
    ASSERT_EQ(s.analysis(d.A, x, y), rocsparse_status_success);

    sptrsm_solver m(handle, rocsparse_analysis_policy_reuse);
    dense_matrix  X(b, a.n, 2), Y(std::vector<double>(b.size(), 0.0), a.n, 2);
    ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_analysis), rocsparse_status_success);
    ASSERT_EQ(m.run(d.A, X, Y, rocsparse_sptrsm_stage_compute), rocsparse_status_success);
    expect_close(Y.values(), host_lower_solve(a, b, 2), tolerance<double>(max_row_length(a)));
}

// The compute stage of spsm without the preprocess dereferences the analysis it does
// not have (spsv returns rocsparse_status_invalid_pointer in that case).
TEST_F(PrecondTriangularScenariosGeneric, DISABLED_spsm_compute_without_the_preprocess)
{
    const auto a = make_grid_matrix<double>(4, true);
    device_csr d;
    d.init(a, 1, true);
    std::vector<double> c;
    check(run_spsm(handle, d, false, rocsparse_operation_none, 2, right_hand_side(a.n, 2), c),
          is(rocsparse_status_invalid_pointer),
          "spsm compute without the preprocess");
}
