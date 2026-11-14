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

// The HIP CSR matrix beyond what unit_test_matrix_csr.cpp covers: copies with
// host matrices and with matrices that have no arrays, matrices without
// entries, the options of the iterative ILU(0), the shared analysis buffer, and
// the remaining AMG and ordering kernels, checked against the host or against
// references.

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/hip/hip_allocate_free.hpp"
#include "base/hip/hip_matrix_csr.hpp"
#include "solvers/preconditioners/preconditioner.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;

namespace
{
    const Format csr_format{rocalution::CSR, 1};

    // Tridiagonal, plus an entry three columns to the right of the diagonal,
    // wrapped around, in every even row, with a dominant diagonal
    template <typename T>
    Csr<T> dominant(int n)
    {
        std::vector<T> D(static_cast<size_t>(n) * n, static_cast<T>(0));
        for(int i = 0; i < n; ++i)
        {
            for(int j : {i - 1, i + 1, i % 2 == 0 ? (i + 3) % n : -1})
            {
                if(j >= 0 && j < n && j != i)
                {
                    D[i * n + j] = from_int<T>(-1 - (i + 2 * j) % 3);
                }
            }
            D[i * n + i] = from_int<T>(20 + i);
        }
        return from_dense(n, n, D);
    }

    template <typename T>
    std::unique_ptr<BaseMatrix<T>> accelerator(const Csr<T>& ref)
    {
        return make_matrix(Backend::Accelerator, csr_format, *ref.host_csr());
    }

    template <typename T>
    std::unique_ptr<BaseMatrix<T>> empty_accelerator()
    {
        return empty_matrix<T>(Backend::Accelerator, csr_format);
    }

    template <typename T>
    rocalution::AcceleratorMatrix<T>& as_accelerator(BaseMatrix<T>& A)
    {
        return dynamic_cast<rocalution::AcceleratorMatrix<T>&>(A);
    }

    // The iterative ILU(0) stops at a tolerance of 1e-14, which float cannot reach
    template <typename T>
    double iterative_tol()
    {
        return loose_tol<T>(1e-10);
    }

    // The ILU(0) factors on the pattern of ref
    template <typename T>
    Csr<T> ilu0(const Csr<T>& ref)
    {
        const std::vector<T> D = ilu0_dense(ref);
        return transformed(ref,
                           [&](int i, int j, T) { return D[static_cast<size_t>(i) * ref.n + j]; });
    }

    template <typename T>
    void expect_csr_near(const Csr<T>& A, const Csr<T>& ref, double tolerance)
    {
        ASSERT_EQ(A.n, ref.n);
        ASSERT_EQ(A.m, ref.m);
        EXPECT_EQ(A.row_offset, ref.row_offset);
        EXPECT_EQ(A.col, ref.col);
        expect_all_near(A.val, ref.val, tolerance);
    }

    template <typename T>
    void expect_empty(const BaseMatrix<T>& A, int nrow, int ncol)
    {
        EXPECT_EQ(A.GetM(), nrow);
        EXPECT_EQ(A.GetN(), ncol);
        EXPECT_EQ(A.GetNnz(), 0);
    }
}

template <typename P>
class hip_matrix_csr_backend : public TypedBackendTest<P>
{
};

TYPED_TEST_SUITE(hip_matrix_csr_backend, AcceleratorOnly<FloatingTypes>, TypedName);

// The copies allocate an empty destination and overwrite one of the same size
TYPED_TEST(hip_matrix_csr_backend, copies_with_host_matrices)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = dominant<T>(9);
    const Csr<T> other = transformed(ref, [](int, int, T v) { return -v; });

    auto A = accelerator(ref);
    ASSERT_NE(A, nullptr);
    auto& hip_A = as_accelerator(*A);

    {
        SCOPED_TRACE("CopyFromHost into a matrix of the same size");
        auto B = accelerator(other);
        ASSERT_NE(B, nullptr);
        as_accelerator(*B).CopyFromHost(*ref.host_csr());
        expect_csr_near(csr_of(*B), ref, 0.0);
    }
    {
        SCOPED_TRACE("CopyToHost into a matrix of the same size");
        auto H = other.host_csr();
        hip_A.CopyToHost(H.get());
        expect_csr_near(csr_of(*H), ref, 0.0);
    }
    {
        SCOPED_TRACE("CopyToAsync into an empty host matrix");
        HostMatrixCSR<T> H(descriptor());
        A->CopyToAsync(&H);
        rocalution::_rocalution_sync();
        expect_csr_near(csr_of(H), ref, 0.0);
    }
    {
        SCOPED_TRACE("CopyToAsync into a host matrix of the same size");
        auto H = other.host_csr();
        A->CopyToAsync(H.get());
        rocalution::_rocalution_sync();
        expect_csr_near(csr_of(*H), ref, 0.0);
    }
}

// A matrix that never had arrays copies as a 0 x 0 matrix
TYPED_TEST(hip_matrix_csr_backend, copies_of_a_matrix_without_arrays)
{
    using T = typename TypeParam::value_type;

    {
        SCOPED_TRACE("CopyFromHostAsync of a host matrix without arrays");
        auto             B = empty_accelerator<T>();
        HostMatrixCSR<T> H(descriptor());
        as_accelerator(*B).CopyFromHostAsync(H);
        rocalution::_rocalution_sync();
        expect_empty(*B, 0, 0);
    }

    using Copy = std::function<void(const BaseMatrix<T>&, BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Copy>> copies = {
        {"CopyFrom", [](const BaseMatrix<T>& src, BaseMatrix<T>& dst) { dst.CopyFrom(src); }},
        {"CopyFromAsync",
         [](const BaseMatrix<T>& src, BaseMatrix<T>& dst) { dst.CopyFromAsync(src); }},
        {"CopyTo", [](const BaseMatrix<T>& src, BaseMatrix<T>& dst) { src.CopyTo(&dst); }},
        {"CopyToAsync",
         [](const BaseMatrix<T>& src, BaseMatrix<T>& dst) { src.CopyToAsync(&dst); }},
    };
    for(const auto& copy : copies)
    {
        SCOPED_TRACE(copy.first);
        auto src = empty_accelerator<T>();
        auto dst = empty_accelerator<T>();
        copy.second(*src, *dst);
        rocalution::_rocalution_sync();
        expect_empty(*dst, 0, 0);
        expect_empty(*host_csr_of(*dst), 0, 0);
    }

    {
        SCOPED_TRACE("CopyToHostAsync");
        auto             src = empty_accelerator<T>();
        HostMatrixCSR<T> H(descriptor());
        as_accelerator(*src).CopyToHostAsync(&H);
        rocalution::_rocalution_sync();
        expect_empty(H, 0, 0);
    }
}

TYPED_TEST(hip_matrix_csr_backend, set_data_ptr_without_entries)
{
    using T = typename TypeParam::value_type;

    auto source = empty_accelerator<T>();
    source->AllocateCSR(0, 4, 6);

    PtrType* row_offset = nullptr;
    int*     col        = nullptr;
    T*       val        = nullptr;
    source->LeaveDataPtrCSR(&row_offset, &col, &val);
    ASSERT_NE(row_offset, nullptr);

    auto A = empty_accelerator<T>();
    A->SetDataPtrCSR(&row_offset, &col, &val, 0, 4, 6);
    expect_empty(*A, 4, 6);

    const Csr<T> csr = csr_of(*A);
    EXPECT_EQ(csr.row_offset, std::vector<PtrType>(5, 0));
}

// The operations on the values and the extractions leave a matrix without
// entries as it is. The extractions zero their outputs.
TYPED_TEST(hip_matrix_csr_backend, operations_on_a_matrix_without_entries)
{
    using T       = typename TypeParam::value_type;
    const int n   = 5;
    const T   two = static_cast<T>(2);

    using Operation = std::function<bool(BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Operation>> operations = {
        {"Zeros", [](BaseMatrix<T>& A) { return A.Zeros(); }},
        {"Sort", [](BaseMatrix<T>& A) { return A.Sort(); }},
        {"Scale", [&](BaseMatrix<T>& A) { return A.Scale(two); }},
        {"ScaleDiagonal", [&](BaseMatrix<T>& A) { return A.ScaleDiagonal(two); }},
        {"ScaleOffDiagonal", [&](BaseMatrix<T>& A) { return A.ScaleOffDiagonal(two); }},
        {"AddScalar", [&](BaseMatrix<T>& A) { return A.AddScalar(two); }},
        {"AddScalarDiagonal", [&](BaseMatrix<T>& A) { return A.AddScalarDiagonal(two); }},
        {"AddScalarOffDiagonal", [&](BaseMatrix<T>& A) { return A.AddScalarOffDiagonal(two); }},
        {"DiagonalMatrixMultR",
         [&](BaseMatrix<T>& A) {
             return A.DiagonalMatrixMultR(
                 *make_vector(Backend::Accelerator, std::vector<T>(n, two)));
         }},
        {"DiagonalMatrixMultL",
         [&](BaseMatrix<T>& A) {
             return A.DiagonalMatrixMultL(
                 *make_vector(Backend::Accelerator, std::vector<T>(n, two)));
         }},
        {"Compress", [](BaseMatrix<T>& A) { return A.Compress(0.5); }},
        {"Transpose", [](BaseMatrix<T>& A) { return A.Transpose(); }},
        {"Transpose into a matrix",
         [&](BaseMatrix<T>& A) {
             auto T_A = empty_accelerator<T>();
             T_A->AllocateCSR(0, n, n);
             bool ok = A.Transpose(T_A.get());
             expect_empty(*T_A, n, n);
             return ok;
         }},
        {"Permute",
         [&](BaseMatrix<T>& A) {
             return A.Permute(*make_vector(Backend::Accelerator, std::vector<int>{4, 3, 2, 1, 0}));
         }},
        {"ApplyAdd",
         [&](BaseMatrix<T>& A) {
             auto in  = make_vector(Backend::Accelerator, std::vector<T>(n, two));
             auto out = make_vector(Backend::Accelerator, from_ints<T>({1, 2, 3, 4, 5}));
             A.ApplyAdd(*in, two, out.get());
             EXPECT_EQ(values_of(*out), from_ints<T>({1, 2, 3, 4, 5}));
             return true;
         }},
        {"CopyFromCSR and CopyToCSR",
         [&](BaseMatrix<T>& A) {
             auto source = empty_accelerator<T>();
             source->AllocateCSR(0, n, n);
             PtrType* row_offset = nullptr;
             int*     col        = nullptr;
             T*       val        = nullptr;
             source->LeaveDataPtrCSR(&row_offset, &col, &val);
             A.CopyFromCSR(row_offset, col, val);

             // CopyToCSR writes the n + 1 zero row offsets over the 7s
             const size_t         bytes = sizeof(PtrType) * (n + 1);
             std::vector<PtrType> copied(n + 1, 7);
             EXPECT_EQ(hipMemcpy(row_offset, copied.data(), bytes, hipMemcpyHostToDevice),
                       hipSuccess);
             A.CopyToCSR(row_offset, col, val);
             rocalution::_rocalution_sync();
             EXPECT_EQ(hipMemcpy(copied.data(), row_offset, bytes, hipMemcpyDeviceToHost),
                       hipSuccess);
             EXPECT_EQ(copied, std::vector<PtrType>(n + 1, 0));

             source->SetDataPtrCSR(&row_offset, &col, &val, 0, n, n);
             return true;
         }},
    };

    for(const auto& operation : operations)
    {
        SCOPED_TRACE(operation.first);
        auto A = empty_accelerator<T>();
        A->AllocateCSR(0, n, n);
        EXPECT_TRUE(operation.second(*A));
        expect_empty(*A, n, n);
    }

    using Extraction = std::function<bool(const BaseMatrix<T>&, BaseVector<T>*)>;
    const std::vector<std::pair<std::string, Extraction>> extractions = {
        {"ExtractDiagonal",
         [](const BaseMatrix<T>& A, BaseVector<T>* v) { return A.ExtractDiagonal(v); }},
        {"ExtractColumnVector",
         [](const BaseMatrix<T>& A, BaseVector<T>* v) { return A.ExtractColumnVector(2, v); }},
        {"ExtractRowVector",
         [](const BaseMatrix<T>& A, BaseVector<T>* v) { return A.ExtractRowVector(2, v); }},
    };

    std::vector<std::vector<T>> extracted;
    for(const auto& extraction : extractions)
    {
        SCOPED_TRACE(extraction.first);
        auto A = empty_accelerator<T>();
        A->AllocateCSR(0, n, n);
        auto v = make_vector(Backend::Accelerator, std::vector<T>(n, from_int<T>(7)));
        EXPECT_TRUE(extraction.second(*A, v.get()));
        expect_empty(*A, n, n);
        extracted.push_back(values_of(*v));
    }

    KNOWN_BUG("the extractions return early without entries");
    for(size_t k = 0; k < extractions.size(); ++k)
    {
        SCOPED_TRACE(extractions[k].first);
        EXPECT_EQ(extracted[k], std::vector<T>(n, static_cast<T>(0)));
    }
}

// The column becomes the only entries of the matrix. The code for matrices with
// entries has a known bug that kills the process.
TYPED_TEST(hip_matrix_csr_backend, replace_column_vector_of_a_matrix_without_entries)
{
    using T     = typename TypeParam::value_type;
    const int n = 5;

    KNOWN_BUG("ReplaceColumnVector returns early without entries");
    expect_in_child([&] {
        const std::vector<T> column = from_ints<T>({1, 2, 3, 4, 5});
        Csr<T>               expected(n);
        for(const T& v : column)
        {
            expected.add(2, v);
            expected.end_row();
        }

        auto A = empty_accelerator<T>();
        A->AllocateCSR(0, n, n);
        EXPECT_TRUE(A->ReplaceColumnVector(2, *make_vector(Backend::Accelerator, column)));
        expect_csr_near(csr_of(*A), expected, 0.0);
    });
}

// The factorizations and solves of a 0 x 0 matrix have nothing to do. The
// solves need no analysis then.
TYPED_TEST(hip_matrix_csr_backend, factorizations_and_solves_of_a_matrix_without_rows)
{
    using T = typename TypeParam::value_type;

    auto in  = make_vector(Backend::Accelerator, std::vector<T>{});
    auto inv = make_vector(Backend::Accelerator, std::vector<T>{});

    using Operation = std::function<bool(BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Operation>> operations = {
        {"ILU0Factorize", [](BaseMatrix<T>& A) { return A.ILU0Factorize(); }},
        {"ICFactorize", [&](BaseMatrix<T>& A) { return A.ICFactorize(inv.get()); }},
        {"ItILU0Factorize",
         [](BaseMatrix<T>& A) {
             int niter = -1;
             return A.ItILU0Factorize(rocalution::Default, 0, 10, 1e-8, &niter, nullptr);
         }},
        {"Apply",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             A.Apply(*in, out.get());
             return true;
         }},
        {"LUSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.LUSolve(*in, out.get());
         }},
        {"LLSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.LLSolve(*in, out.get()) && A.LLSolve(*in, *inv, out.get());
         }},
        {"LSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.LSolve(*in, out.get());
         }},
        {"USolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.USolve(*in, out.get());
         }},
        {"ItLUSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.ItLUSolve(10, 0.0, false, *in, out.get());
         }},
        {"ItLLSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.ItLLSolve(10, 0.0, false, *in, out.get())
                    && A.ItLLSolve(10, 0.0, false, *in, *inv, out.get());
         }},
        {"ItLSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.ItLSolve(10, 0.0, false, *in, out.get());
         }},
        {"ItUSolve",
         [&](BaseMatrix<T>& A) {
             auto out = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.ItUSolve(10, 0.0, false, *in, out.get());
         }},
        {"ExtractInverseDiagonal",
         [&](BaseMatrix<T>& A) {
             auto d = make_vector(Backend::Accelerator, std::vector<T>{});
             return A.ExtractInverseDiagonal(d.get());
         }},
    };

    for(const auto& operation : operations)
    {
        SCOPED_TRACE(operation.first);
        auto A = empty_accelerator<T>();
        A->AllocateCSR(0, 0, 0);
        EXPECT_TRUE(operation.second(*A));
        expect_empty(*A, 0, 0);
    }
}

// The AMG setup steps leave their outputs alone for a matrix without entries
TYPED_TEST(hip_matrix_csr_backend, amg_steps_of_a_matrix_without_entries)
{
    using T     = typename TypeParam::value_type;
    const int n = 4;

    auto A = empty_accelerator<T>();
    A->AllocateCSR(0, n, n);
    auto ghost = empty_accelerator<T>();

    auto none_int64 = make_vector(Backend::Accelerator, std::vector<int64_t>{});
    auto none_int   = make_vector(Backend::Accelerator, std::vector<int>{});
    auto none_ptr   = make_vector(Backend::Accelerator, std::vector<PtrType>{});

    {
        SCOPED_TRACE("AMGComputeStrongConnections");
        auto diag = make_vector(Backend::Accelerator, std::vector<T>(n, static_cast<T>(1)));
        auto conn = make_vector(Backend::Accelerator, std::vector<bool>{true});
        EXPECT_TRUE(A->AMGComputeStrongConnections(
            static_cast<T>(0.5), *diag, *none_int64, conn.get(), *ghost));
        EXPECT_EQ(values_of(*conn), std::vector<bool>{true});
    }
    {
        SCOPED_TRACE("AMGPMISFindMaxNeighbourNode");
        auto conn      = make_vector(Backend::Accelerator, std::vector<bool>{});
        auto state     = make_vector(Backend::Accelerator, std::vector<int>{1, 2, 3, 4});
        auto hash      = make_vector(Backend::Accelerator, std::vector<int>{4, 3, 2, 1});
        auto max_state = make_vector(Backend::Accelerator, std::vector<int>{1, 2, 3, 4});
        auto agg       = make_vector(Backend::Accelerator, std::vector<int64_t>{-1, -1, -1, -1});
        bool undecided = false;
        EXPECT_TRUE(A->AMGPMISFindMaxNeighbourNode(0,
                                                   n,
                                                   undecided,
                                                   *conn,
                                                   *state,
                                                   *hash,
                                                   *none_ptr,
                                                   *none_int64,
                                                   *none_int,
                                                   *none_int,
                                                   max_state.get(),
                                                   agg.get(),
                                                   *ghost));
        EXPECT_EQ(values_of(*max_state), (std::vector<int>{1, 2, 3, 4}));
        EXPECT_EQ(values_of(*agg), (std::vector<int64_t>{-1, -1, -1, -1}));
    }
    {
        SCOPED_TRACE("RSInterpolationTruncation");
        EXPECT_TRUE(A->RSInterpolationTruncation(0.5f, 2));
        expect_empty(*A, n, n);
    }
}

// Without a truncation factor and an entry limit the interpolation is kept
TYPED_TEST(hip_matrix_csr_backend, rs_interpolation_truncation_without_limits_keeps_the_matrix)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = dominant<T>(8);

    auto A = accelerator(ref);
    ASSERT_NE(A, nullptr);
    EXPECT_TRUE(A->RSInterpolationTruncation(0.0f, 0));
    expect_csr_near(csr_of(*A), ref, 0.0);
}

// The options leave the result alone: it is the ILU(0) in each case. An unknown
// algorithm is the default one: with the stopping criteria it stops after as many
// iterations, which SyncSplit and SyncSplitFusion do not.
TYPED_TEST(hip_matrix_csr_backend, iterative_ilu0_options_and_unknown_algorithms)
{
    using T              = typename TypeParam::value_type;
    const Csr<T> ref     = dominant<T>(10);
    const Csr<T> factors = ilu0(ref);

    const int stopping = rocalution::StoppingCriteria | rocalution::ComputeNrmResidual;
    struct Case
    {
        std::string                 name;
        rocalution::ItILU0Algorithm alg;
        int                         option;
    };
    const std::vector<Case> cases = {
        {"default", rocalution::Default, stopping},
        {"unknown algorithm", static_cast<rocalution::ItILU0Algorithm>(99), stopping},
        {"verbose", rocalution::Default, rocalution::Verbose},
        {"COO format", rocalution::SyncSplit, rocalution::COOFormat},
    };

    std::vector<int> niters;
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.name);
        auto A = accelerator(ref);
        ASSERT_NE(A, nullptr);

        int  niter = -1;
        bool ok    = false;
        logged([&] { ok = A->ItILU0Factorize(c.alg, c.option, 200, 1e-14, &niter, nullptr); });
        EXPECT_TRUE(ok);
        EXPECT_GE(niter, 1);
        EXPECT_LE(niter, 200);
        expect_csr_near(csr_of(*A), factors, iterative_tol<T>());
        niters.push_back(niter);
    }
    EXPECT_LT(niters[0], 200);
    EXPECT_EQ(niters[1], niters[0]);
}

// The history of float and complex<float> goes through a float buffer. The
// options are those of the rocsparse_csritilu0_compute example, see
// matrix_csr_itilu0.records_the_convergence_history.
TYPED_TEST(hip_matrix_csr_backend, iterative_ilu0_records_the_convergence_history)
{
    using T              = typename TypeParam::value_type;
    const Csr<T> ref     = dominant<T>(10);
    const Csr<T> factors = ilu0(ref);

    auto A = accelerator(ref);
    ASSERT_NE(A, nullptr);

    const int option = rocalution::StoppingCriteria | rocalution::ComputeNrmResidual
                       | rocalution::ConvergenceHistory;
    const int    max_iter  = 200;
    const double untouched = -1.0;
    // complex<float> does not reach 1e-14 in SyncSplit
    const double tolerance = std::is_same_v<real_t<T>, float> ? 1e-6 : 1e-14;

    int                 niter = -1;
    std::vector<double> history(2 * max_iter, untouched);
    ASSERT_TRUE(A->ItILU0Factorize(
        rocalution::SyncSplit, option, max_iter, tolerance, &niter, history.data()));
    expect_csr_near(csr_of(*A), factors, iterative_tol<T>());

    ASSERT_GE(niter, 1);
    ASSERT_LT(niter, max_iter);
    for(int i = niter; i < 2 * niter; ++i)
    {
        EXPECT_TRUE(std::isfinite(history[i])) << "residual norm " << i - niter;
        EXPECT_GE(history[i], 0.0) << "residual norm " << i - niter;
    }
    EXPECT_GT(history[niter], 0.0) << "the first residual norm";
    EXPECT_LE(history[2 * niter - 1], tolerance) << "the last residual norm";
}

// IC(0) grows the analysis buffer that a smaller analysis left, and reuses a
// larger one: the iterative analysis needs more than IC(0) from about 1000 rows
// on, except for float
TYPED_TEST(hip_matrix_csr_backend, ic0_after_an_iterative_analysis)
{
    using T    = typename TypeParam::value_type;
    auto lower = [](int i, int j, T) { return j <= i; };

    for(int n : {10, 1000})
    {
        SCOPED_TRACE("n = " + std::to_string(n));
        Csr<T> ref(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = std::max(0, i - 1); j <= std::min(n - 1, i + 1); ++j)
            {
                ref.add(j, j == i ? static_cast<T>(4 + i % 7) : static_cast<T>(-1));
            }
            ref.end_row();
        }

        const std::vector<T> L        = ic0_dense(ref);
        const Csr<T>         expected = transformed(filtered(ref, lower), [&](int i, int j, T) {
            return L[static_cast<size_t>(i) * n + j];
        });

        auto A = accelerator(ref);
        ASSERT_IN_LOOP(A != nullptr);
        A->ItLAnalyse(false);
        auto unused = make_vector(Backend::Accelerator, std::vector<T>(n, static_cast<T>(0)));
        ASSERT_IN_LOOP(A->ICFactorize(unused.get()));
        A->ItLAnalyseClear();

        expect_csr_near(filtered(csr_of(*A), lower), expected, scaled_tol<T>(1e-14));
    }
}

// The second factorization reuses the buffer of the first one
TYPED_TEST(hip_matrix_csr_backend, repeated_ilu0_factorization)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = dominant<T>(10);

    auto A = accelerator(ref);
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->ILU0Factorize());
    ASSERT_TRUE(A->ILU0Factorize());

    expect_csr_near(csr_of(*A), ilu0(ilu0(ref)), scaled_tol<T>(1e-13));
}

// The transpose of an n x m matrix without entries is an empty m x n matrix
TYPED_TEST(hip_matrix_csr_backend, transpose_of_a_matrix_without_entries)
{
    using T = typename TypeParam::value_type;

    auto A = empty_accelerator<T>();
    A->AllocateCSR(0, 3, 5);
    auto T_A = accelerator(dominant<T>(4));
    ASSERT_NE(T_A, nullptr);
    EXPECT_TRUE(A->Transpose(T_A.get()));
    EXPECT_TRUE(A->Transpose());

    KNOWN_BUG("the transpose of a matrix without entries keeps the sizes");
    expect_empty(*T_A, 5, 3);
    expect_empty(*A, 5, 3);
}

// In float the buffer of the iterative analysis is smaller than csrilu0 needs.
// The assert on the buffer size is off in Release, where rocSPARSE writes past
// the buffer without a visible effect, so only a Debug build can confirm a fix.
TYPED_TEST(hip_matrix_csr_backend, ilu0_after_an_iterative_analysis)
{
    using T = typename TypeParam::value_type;

    auto checks = [] {
        const Csr<T> ref = dominant<T>(1000);
        auto         A   = accelerator(ref);
        ASSERT_NE(A, nullptr);
        A->ItLUAnalyse();
        ASSERT_TRUE(A->ILU0Factorize());
        A->ItLUAnalyseClear();
        expect_csr_near(csr_of(*A), ilu0(ref), scaled_tol<T>(1e-13));
    };

    if constexpr(std::is_same_v<T, float>)
    {
        KNOWN_BUG("ILU0Factorize keeps a shared buffer that is too small");
        expect_in_child(checks);
        return;
    }
    checks();
}

TYPED_TEST(hip_matrix_csr_backend, ilu0_after_a_triangular_analysis)
{
    using T = typename TypeParam::value_type;

    KNOWN_BUG("csrilu0 with the reuse policy after a csrsv analysis hangs or faults");
    expect_in_child(
        [] {
            const Csr<T> ref = dominant<T>(10);
            auto         A   = accelerator(ref);
            ASSERT_NE(A, nullptr);
            A->LUAnalyse();
            ASSERT_TRUE(A->ILU0Factorize());
            A->LUAnalyseClear();
            expect_csr_near(csr_of(*A), ilu0(ref), scaled_tol<T>(1e-13));
        },
        60);
}

TYPED_TEST(hip_matrix_csr_backend, multi_coloring_of_isolated_nodes)
{
    using T = typename TypeParam::value_type;

    // Nodes 2 and 5 are isolated, the others form the path 0-1-3-4-6
    std::vector<T> D(49, static_cast<T>(0));
    for(int i : {0, 1, 3, 4, 6})
    {
        D[i * 7 + i] = static_cast<T>(4);
    }
    for(const auto& e : std::vector<std::pair<int, int>>{{0, 1}, {1, 3}, {3, 4}, {4, 6}})
    {
        D[e.first * 7 + e.second] = static_cast<T>(-1);
        D[e.second * 7 + e.first] = static_cast<T>(-1);
    }
    const Csr<T> ref = from_dense(7, 7, D);

    auto A = make_matrix(Backend::Accelerator, csr_format, *ref.host_csr());
    ASSERT_NE(A, nullptr);
    auto perm       = make_vector(Backend::Accelerator, std::vector<int>{});
    int  num_colors = 0;
    int* size       = nullptr;
    ASSERT_TRUE(A->MultiColoring(num_colors, &size, perm.get()));
    const std::vector<int> sizes(size, size + num_colors);
    rocalution::free_host(&size);

    // The path needs two colors, and the isolated nodes need no color of their own
    EXPECT_EQ(num_colors, 2);
    const std::vector<int> p = values_of(*perm);
    ASSERT_EQ(p.size(), static_cast<size_t>(7));
    std::vector<int> sorted = p;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, (std::vector<int>{0, 1, 2, 3, 4, 5, 6}));
    expect_valid_coloring(ref, num_colors, sizes, p);
}

// Row i of the prolongation has a one in the column of its aggregate, and no
// entry if node i has none
TYPED_TEST(hip_matrix_csr_backend, unsmoothed_aggregation_prolongation)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = dominant<T>(7);

    const std::vector<std::vector<int64_t>> cases
        = {{0, 0, 1, 2, 1, 2, 3}, {1, -1, 0, 0, -1, 2, 1}};
    for(const auto& aggregates : cases)
    {
        int64_t ncol = 0;
        for(int64_t a : aggregates)
        {
            ncol = std::max(ncol, a + 1);
        }
        Csr<T> expected(ref.n, static_cast<int>(ncol));
        for(int64_t a : aggregates)
        {
            if(a >= 0)
            {
                expected.add(static_cast<int>(a), static_cast<T>(1));
            }
            expected.end_row();
        }

        auto A = accelerator(ref);
        ASSERT_IN_LOOP(A != nullptr);
        auto agg = make_vector(Backend::Accelerator, aggregates);
        auto P   = empty_accelerator<T>();
        ASSERT_IN_LOOP(A->AMGUnsmoothedAggregation(*agg, P.get()));
        expect_csr_near(csr_of(*P), expected, 0.0);
    }
}

TYPED_TEST(hip_matrix_csr_backend, info_names_the_class)
{
    using T = typename TypeParam::value_type;
    auto A  = empty_accelerator<T>();
    EXPECT_NE(logged([&] { A->Info(); }).find("HIPAcceleratorMatrixCSR"), std::string::npos);
}

// A HIP matrix needs the backend descriptor
TEST(hip_matrix_csr_backend_DeathTest, default_constructor_ends_the_program)
{
    REQUIRE_BACKEND(Backend::Accelerator);
    EXPECT_FATAL_ERROR({ rocalution::HIPAcceleratorMatrixCSR<double> A; }, "hip_matrix_csr\\.cpp");
}
