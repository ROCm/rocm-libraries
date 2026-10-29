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

// The LocalMatrix operations on both backends and in every format: the result,
// the format and backend the matrix ends in, and the fallback warnings

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "unit_test_amg_reference.hpp"
#include "unit_test_common.hpp"
#include "unit_test_local.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/local_matrix.hpp"
#include "base/local_stencil.hpp"
#include "base/local_vector.hpp"
#include "solvers/preconditioners/preconditioner.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;
using rocalution::LocalMatrix;
using rocalution::LocalVector;

namespace
{
    using rocalution::BCSR;
    using rocalution::COO;
    using rocalution::CSR;
    using rocalution::DENSE;
    using rocalution::DIA;
    using rocalution::ELL;
    using rocalution::HYB;
    using rocalution::MCSR;

    const Format csr_format = {CSR, 1};

    template <typename T>
    std::unique_ptr<LocalMatrix<T>> empty_local_matrix(Backend b)
    {
        auto A = std::make_unique<LocalMatrix<T>>();
        if(b == Backend::Accelerator)
        {
            A->MoveToAccelerator();
        }
        return A;
    }

    template <typename T>
    Csr<T> nonzeros(const Csr<T>& A)
    {
        return filtered(A, [](int, int, T v) { return v != static_cast<T>(0); });
    }

    // Relative to the reference for magnitudes above 1, absolute below
    template <typename T>
    void expect_close(const std::vector<T>& a, const std::vector<T>& ref, double tol)
    {
        ASSERT_EQ(a.size(), ref.size());
        for(size_t i = 0; i < a.size(); ++i)
        {
            const double scale = std::max(1.0, static_cast<double>(std::abs(ref[i])));
            EXPECT_LE(static_cast<double>(std::abs(a[i] - ref[i])), tol * scale)
                << "at index " << i;
        }
    }

    // M x = b up to a backward error: each residual relative to the sum of |M_ij x_j| of its row
    template <typename T>
    void
        expect_solves(const Csr<T>& M, const std::vector<T>& x, const std::vector<T>& b, double tol)
    {
        ASSERT_EQ(x.size(), static_cast<size_t>(M.m));
        ASSERT_EQ(b.size(), static_cast<size_t>(M.n));
        for(int i = 0; i < M.n; ++i)
        {
            T sum   = 0;
            T scale = 1;
            for(PtrType k = M.row_offset[i]; k < M.row_offset[i + 1]; ++k)
            {
                sum += M.val[k] * x[M.col[k]];
                scale += std::abs(M.val[k] * x[M.col[k]]);
            }
            EXPECT_LE(static_cast<double>(std::abs(sum - b[i])), tol * static_cast<double>(scale))
                << "in row " << i;
        }
    }

    // The same non-zeros, independent of explicit zeros; exact for tol 0
    template <typename T>
    void expect_entries(const Csr<T>& A, const Csr<T>& ref, double tol = 0)
    {
        ASSERT_EQ(A.n, ref.n);
        ASSERT_EQ(A.m, ref.m);
        if(tol == 0)
        {
            const Csr<T> a = nonzeros(A);
            const Csr<T> r = nonzeros(ref);
            EXPECT_EQ(a.row_offset, r.row_offset);
            EXPECT_EQ(a.col, r.col);
            EXPECT_EQ(a.val, r.val);
        }
        else
        {
            expect_close(dense(A), dense(ref), tol);
        }
    }

    // In format f on backend b, holding the non-zeros of ref
    template <typename T>
    void expect_matrix(
        const LocalMatrix<T>& A, const Csr<T>& ref, Format f, Backend b, double tol = 0)
    {
        EXPECT_EQ(rocalution::_matrix_format_names[A.GetFormat()], format_name(f));
        if(f.id == BCSR)
        {
            EXPECT_EQ(A.GetBlockDimension(), f.blockdim);
        }
        EXPECT_EQ(backend_of(A), backend_name(b));
        expect_entries(csr_of(A), ref, tol);
    }

    template <typename T>
    void expect_vector(const LocalVector<T>& v, const std::vector<T>& ref, Backend b, double tol)
    {
        EXPECT_EQ(backend_of(v), backend_name(b));
        expect_close(values_of(v), ref, tol);
    }

    // LocalMatrix warns when an operation falls back to another format or to
    // the host, and only then
    void expect_fallback_warning(const std::string& log, const std::string& operation, bool native)
    {
        const bool warned = log.find("*** warning: LocalMatrix::" + operation + "() is performed")
                            != std::string::npos;
        EXPECT_EQ(warned, !native) << operation << " logged:\n" << log;
    }

    // The formats with an analysis for the triangular solves. In the others the
    // analysis ends the program, and the solves fall back to the host.
    bool has_analysis(Backend b, unsigned int format, const std::string& solve)
    {
        if(b == Backend::Accelerator)
        {
            return format == CSR || format == BCSR;
        }
        return format == CSR || (solve == "LU" && format == MCSR);
    }

    // Nonsymmetric and diagonally dominant, with lower bandwidth 1, so that ILU(0)
    // is the exact LU factorization. DIA stores zeros and ELL pads every other row.
    template <typename T = double>
    Csr<T> banded(int n = 12)
    {
        return transformed(banded_with_gaps<T>(n), [](int i, int j, T v) {
            return i == j ? static_cast<T>(50 * i + 150) : v;
        });
    }

    // Symmetric positive definite
    template <typename T = double>
    Csr<T> spd(int n = 12)
    {
        return Tridiagonal<T>(n);
    }

    // n x m, with the entries of row i in the columns i, i + 2 and 3 i, modulo m
    template <typename T = double>
    Csr<T> rectangular(int n, int m)
    {
        Csr<T> A(n, m);
        for(int i = 0; i < n; ++i)
        {
            for(int j : std::set<int>{i % m, (i + 2) % m, (3 * i) % m})
            {
                A.add(j, static_cast<T>(i + 2 * j + 1));
            }
            A.end_row();
        }
        return A;
    }

    // Entries (i, i - 2) and (i, i), so that the union with banded has more diagonals
    template <typename T = double>
    Csr<T> two_diagonals(int n = 12)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            if(i >= 2)
            {
                A.add(i - 2, static_cast<T>(i + 1));
            }
            A.add(i, static_cast<T>(1));
            A.end_row();
        }
        return A;
    }

    // Symmetric pattern with |i - j| <= 2, diagonally dominant
    template <typename T>
    Csr<T> pentadiagonal(int n = 12)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = std::max(0, i - 2); j <= std::min(n - 1, i + 2); ++j)
            {
                A.add(j, static_cast<T>(i == j ? 10 : -1));
            }
            A.end_row();
        }
        return A;
    }

    template <typename T>
    Csr<T> lower(const Csr<T>& A, bool diagonal)
    {
        return filtered(A, [=](int i, int j, T) { return j < i || (diagonal && j == i); });
    }

    template <typename T>
    Csr<T> upper(const Csr<T>& A, bool diagonal)
    {
        return filtered(A, [=](int i, int j, T) { return j > i || (diagonal && j == i); });
    }

    // The columns of each row in reverse order
    template <typename T>
    Csr<T> unsorted(Csr<T> A)
    {
        for(int i = 0; i < A.n; ++i)
        {
            std::reverse(A.col.begin() + A.row_offset[i], A.col.begin() + A.row_offset[i + 1]);
            std::reverse(A.val.begin() + A.row_offset[i], A.val.begin() + A.row_offset[i + 1]);
        }
        return A;
    }

    template <typename T>
    std::vector<std::set<int>> pattern_of(const Csr<T>& A)
    {
        std::vector<std::set<int>> p(A.n);
        for(int i = 0; i < A.n; ++i)
        {
            p[i].insert(A.col.begin() + A.row_offset[i], A.col.begin() + A.row_offset[i + 1]);
        }
        return p;
    }

    // The pattern of |A|^2, with zero values
    template <typename T>
    Csr<T> squared_pattern(const Csr<T>& A)
    {
        const Csr<T> ones = transformed(A, [](int, int, T) { return static_cast<T>(1); });
        return from_dense(A.n, A.n, dense_product(dense(ones), dense(ones), A.n, A.n, A.n));
    }

    template <typename T>
    Csr<T> product(const Csr<T>& A, const Csr<T>& B)
    {
        return from_dense(A.n, B.m, dense_product(dense(A), dense(B), A.n, A.m, B.m));
    }

    // The strict triangle with a unit diagonal
    template <typename T>
    Csr<T> unit_triangle(const Csr<T>& strict)
    {
        std::vector<T> D = dense(strict);
        for(int i = 0; i < strict.n; ++i)
        {
            D[static_cast<size_t>(i) * strict.n + i] = 1;
        }
        return from_dense(strict.n, strict.n, D);
    }

    // The rows [r0, r0 + nr) and columns [c0, c0 + nc)
    template <typename T>
    Csr<T> block(const Csr<T>& A, int r0, int c0, int nr, int nc)
    {
        Csr<T> B(nr, nc);
        for(int i = r0; i < r0 + nr; ++i)
        {
            for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
            {
                if(A.col[k] >= c0 && A.col[k] < c0 + nc)
                {
                    B.add(A.col[k] - c0, A.val[k]);
                }
            }
            B.end_row();
        }
        return B;
    }

    template <typename T>
    std::vector<T> column(const Csr<T>& A, int j)
    {
        std::vector<T>       c(A.n, 0);
        const std::vector<T> D = dense(A);
        for(int i = 0; i < A.n; ++i)
        {
            c[i] = D[static_cast<size_t>(i) * A.m + j];
        }
        return c;
    }

    template <typename T>
    std::vector<T> row(const Csr<T>& A, int i)
    {
        const std::vector<T> D = dense(A);
        return std::vector<T>(D.begin() + static_cast<size_t>(i) * A.m,
                              D.begin() + static_cast<size_t>(i + 1) * A.m);
    }

    void expect_permutation(const std::vector<int>& perm, int n)
    {
        std::vector<int> sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        std::vector<int> identity(n);
        std::iota(identity.begin(), identity.end(), 0);
        EXPECT_EQ(sorted, identity) << "not a permutation of 0.." << n - 1;
    }

    template <typename T>
    bool connected(const Csr<T>& A, int i, int j)
    {
        for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
        {
            if(A.col[k] == j)
            {
                return true;
            }
        }
        return false;
    }

    std::vector<int> scrambled(int n)
    {
        std::vector<int> perm(n);
        for(int i = 0; i < n; ++i)
        {
            perm[i] = (5 * i + 3) % n;
        }
        return perm;
    }

    template <typename P>
    class local_matrix : public TypedBackendTest<P>
    {
    protected:
        static Backend backend()
        {
            return P::backend;
        }

        // The other backend if it is available, else the same
        static Backend other()
        {
            return backend() == Backend::Host && available(Backend::Accelerator)
                       ? Backend::Accelerator
                       : Backend::Host;
        }
    };

    using FormatParam = std::tuple<Backend, Format>;

    class local_matrix_formats : public testing::TestWithParam<FormatParam>
    {
    protected:
        void SetUp() override
        {
            REQUIRE_BACKEND(backend());
        }

        Backend backend() const
        {
            return std::get<0>(GetParam());
        }

        Format format() const
        {
            return std::get<1>(GetParam());
        }

        bool is(unsigned int id) const
        {
            return format().id == id;
        }

        bool native(const std::string& operation) const
        {
            return implements(backend(), format().id, operation);
        }

        // For inputs the format cannot hold, which make() leaves in CSR
        template <typename T>
        bool native(const LocalMatrix<T>& A, const std::string& operation) const
        {
            return implements(backend(), A.GetFormat(), operation);
        }

        template <typename T>
        std::unique_ptr<LocalMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_local_matrix(backend(), ref, format());
        }

        template <typename T>
        std::unique_ptr<LocalMatrix<T>> empty() const
        {
            auto A = empty_local_matrix<T>(backend());
            A->ConvertTo(format().id, format().blockdim);
            return A;
        }

        template <typename T>
        std::unique_ptr<LocalVector<T>> vector_of(const std::vector<T>& v) const
        {
            return make_local_vector(backend(), v);
        }

        template <typename T>
        void expect_kept(const LocalMatrix<T>& A, const Csr<T>& ref, double tol = 0) const
        {
            expect_matrix(A, ref, format(), backend(), tol);
        }

        // A result the format cannot hold falls back to CSR, with a warning in the log
        template <typename T>
        void expect_kept_or_csr(const LocalMatrix<T>& A,
                                const Csr<T>&         ref,
                                const std::string&    log) const
        {
            if(A.GetFormat() == format().id)
            {
                expect_kept(A, ref);
                return;
            }
            expect_matrix(A, ref, csr_format, backend());
            EXPECT_NE(log.find("Matrix conversion to " + format_name(format())
                               + " failed, falling back to CSR format"),
                      std::string::npos)
                << log;
        }
    };

    class local_matrix_data_ptr : public local_matrix_formats
    {
    };

    const auto all_formats = testing::Values(Format{CSR, 1},
                                             Format{MCSR, 1},
                                             Format{BCSR, 2},
                                             Format{COO, 1},
                                             Format{ELL, 1},
                                             Format{DIA, 1},
                                             Format{HYB, 1},
                                             Format{DENSE, 1});
}

TYPED_TEST_SUITE(local_matrix, WithBackends<RealTypes>, TypedName);

// Construction, allocation and Info

TYPED_TEST(local_matrix, info_reports_the_sizes_the_format_and_the_backend)
{
    using T = typename TypeParam::value_type;

    const std::string current = this->backend() == Backend::Host ? "current=CPU" : "current=HIP";

    auto E = empty_local_matrix<T>(this->backend());
    EXPECT_EQ(E->GetM(), 0);
    EXPECT_EQ(E->GetN(), 0);
    EXPECT_EQ(E->GetNnz(), 0);
    EXPECT_EQ(E->GetFormat(), CSR);
    EXPECT_EQ(E->GetBlockDimension(), 1);
    EXPECT_EQ(backend_of(*E), backend_name(this->backend()));

    std::string info = logged([&] { E->Info(); });
    EXPECT_NE(info.find("rows=0; cols=0; nnz=0;"), std::string::npos) << info;
    EXPECT_NE(info.find("format=CSR(" + std::to_string(8 * sizeof(PtrType)) + ","
                        + std::to_string(8 * sizeof(int)) + ")"),
              std::string::npos)
        << info;
    EXPECT_NE(info.find(current), std::string::npos) << info;

    auto A = make_local_matrix(this->backend(), rectangular<T>(8, 12), Format{COO, 1});
    info   = logged([&] { A->Info(); });
    EXPECT_NE(info.find("rows=8; cols=12; nnz=" + std::to_string(A->GetNnz()) + ";"),
              std::string::npos)
        << info;
    EXPECT_NE(info.find("prec=" + std::to_string(8 * sizeof(T)) + "bit;"), std::string::npos)
        << info;
    EXPECT_NE(info.find("format=COO;"), std::string::npos) << info;
    EXPECT_NE(info.find(current), std::string::npos) << info;
}

TYPED_TEST(local_matrix, allocate_sets_the_format_and_the_sizes)
{
    using T = typename TypeParam::value_type;

    struct Case
    {
        std::string                          name;
        std::function<void(LocalMatrix<T>&)> allocate;
        Format                               format;
        int64_t                              m;
        int64_t                              n;
        int64_t                              nnz;
    };

    // Without entries CSR keeps the sizes. The other formats drop them, which is
    // not documented, so their sizes (-1) are not checked.
    const std::vector<Case> cases = {
        {"CSR", [](LocalMatrix<T>& A) { A.AllocateCSR("A", 7, 4, 5); }, {CSR, 1}, 4, 5, 7},
        {"MCSR", [](LocalMatrix<T>& A) { A.AllocateMCSR("A", 7, 4, 4); }, {MCSR, 1}, 4, 4, 7},
        {"COO", [](LocalMatrix<T>& A) { A.AllocateCOO("A", 7, 4, 5); }, {COO, 1}, 4, 5, 7},
        {"BCSR", [](LocalMatrix<T>& A) { A.AllocateBCSR("A", 3, 2, 3, 2); }, {BCSR, 2}, 4, 6, 12},
        {"DIA", [](LocalMatrix<T>& A) { A.AllocateDIA("A", 12, 4, 4, 3); }, {DIA, 1}, 4, 4, 12},
        {"ELL", [](LocalMatrix<T>& A) { A.AllocateELL("A", 8, 4, 5, 2); }, {ELL, 1}, 4, 5, 8},
        {"HYB", [](LocalMatrix<T>& A) { A.AllocateHYB("A", 8, 3, 2, 4, 5); }, {HYB, 1}, 4, 5, 11},
        {"DENSE", [](LocalMatrix<T>& A) { A.AllocateDENSE("A", 4, 5); }, {DENSE, 1}, 4, 5, 20},
        {"empty CSR", [](LocalMatrix<T>& A) { A.AllocateCSR("A", 0, 4, 5); }, {CSR, 1}, 4, 5, 0},
        {"empty MCSR",
         [](LocalMatrix<T>& A) { A.AllocateMCSR("A", 0, 4, 4); },
         {MCSR, 1},
         -1,
         -1,
         0},
        {"empty COO", [](LocalMatrix<T>& A) { A.AllocateCOO("A", 0, 4, 5); }, {COO, 1}, -1, -1, 0},
        {"empty BCSR",
         [](LocalMatrix<T>& A) { A.AllocateBCSR("A", 0, 2, 3, 2); },
         {BCSR, 2},
         -1,
         -1,
         0},
        {"empty DIA",
         [](LocalMatrix<T>& A) { A.AllocateDIA("A", 0, 4, 4, 0); },
         {DIA, 1},
         -1,
         -1,
         0},
        {"empty ELL",
         [](LocalMatrix<T>& A) { A.AllocateELL("A", 0, 4, 5, 0); },
         {ELL, 1},
         -1,
         -1,
         0},
        {"empty HYB",
         [](LocalMatrix<T>& A) { A.AllocateHYB("A", 0, 0, 0, 4, 5); },
         {HYB, 1},
         -1,
         -1,
         0},
        {"empty DENSE",
         [](LocalMatrix<T>& A) { A.AllocateDENSE("A", 0, 5); },
         {DENSE, 1},
         -1,
         -1,
         0},
    };

    auto A = empty_local_matrix<T>(this->backend());
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.name);
        c.allocate(*A);
        EXPECT_EQ(rocalution::_matrix_format_names[A->GetFormat()], format_name(c.format));
        EXPECT_EQ(A->GetBlockDimension(), c.format.blockdim);
        if(c.m >= 0)
        {
            EXPECT_EQ(A->GetM(), c.m);
            EXPECT_EQ(A->GetN(), c.n);
        }
        EXPECT_EQ(A->GetNnz(), c.nnz);
        EXPECT_EQ(backend_of(*A), backend_name(this->backend()));
    }

    A->AllocateCSR("A", 7, 4, 5);
    A->Clear();
    EXPECT_EQ(A->GetM(), 0);
    EXPECT_EQ(A->GetN(), 0);
    EXPECT_EQ(A->GetNnz(), 0);
    EXPECT_EQ(backend_of(*A), backend_name(this->backend()));

    auto B = empty_local_matrix<T>(this->backend());
    B->AllocateBCSR("B", 3, 2, 3, 3);
    KNOWN_BUG("AllocateBCSR of a matrix not in BCSR uses block dimension 2");
    EXPECT_EQ(B->GetBlockDimension(), 3);
    EXPECT_EQ(B->GetM(), 6);
    EXPECT_EQ(B->GetN(), 9);
    EXPECT_EQ(B->GetNnz(), 27);
}

// Only a GlobalMatrix has a ghost part. As an Operator, a LocalMatrix is all local.
TYPED_TEST(local_matrix, as_an_operator_is_local_and_has_no_ghost_part)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref = rectangular<T>(8, 12);
    auto         A   = make_local_matrix(this->backend(), ref);

    const rocalution::Operator<T>& op = *A;
    EXPECT_EQ(op.GetLocalM(), 8);
    EXPECT_EQ(op.GetLocalN(), 12);
    EXPECT_EQ(op.GetLocalNnz(), ref.nnz());
    EXPECT_EQ(op.GetGhostM(), 0);
    EXPECT_EQ(op.GetGhostN(), 0);
    EXPECT_EQ(op.GetGhostNnz(), 0);
}

// Data pointers and copies

TEST_P(local_matrix_data_ptr, leave_and_set_data_ptr_hand_over_the_arrays)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>  ref = banded<T>();
        auto          A   = make(ref);
        const int64_t m   = A->GetM();
        const int64_t n   = A->GetN();
        const int64_t nnz = A->GetNnz();
        auto          B   = empty_local_matrix<T>(backend());

        int* row = nullptr;
        int* col = nullptr;
        T*   val = nullptr;
        switch(format().id)
        {
        case CSR:
        {
            PtrType* row_offset = nullptr;
            A->LeaveDataPtrCSR(&row_offset, &col, &val);
            if(backend() == Backend::Host)
            {
                EXPECT_EQ(std::vector<PtrType>(row_offset, row_offset + m + 1), ref.row_offset);
                EXPECT_EQ(std::vector<int>(col, col + nnz), ref.col);
                EXPECT_EQ(std::vector<T>(val, val + nnz), ref.val);
            }
            B->SetDataPtrCSR(&row_offset, &col, &val, "B", nnz, m, n);
            EXPECT_EQ(row_offset, nullptr);
            break;
        }
        case MCSR:
            A->LeaveDataPtrMCSR(&row, &col, &val);
            B->SetDataPtrMCSR(&row, &col, &val, "B", nnz, m, n);
            break;
        case BCSR:
        {
            int blockdim = format().blockdim;
            A->LeaveDataPtrBCSR(&row, &col, &val, blockdim);
            EXPECT_EQ(blockdim, format().blockdim);
            B->SetDataPtrBCSR(&row,
                              &col,
                              &val,
                              "B",
                              nnz / (blockdim * blockdim),
                              (m + blockdim - 1) / blockdim,
                              (n + blockdim - 1) / blockdim,
                              blockdim);
            break;
        }
        case COO:
            A->LeaveDataPtrCOO(&row, &col, &val);
            B->SetDataPtrCOO(&row, &col, &val, "B", nnz, m, n);
            break;
        case DIA:
        {
            int num_diag = 0;
            A->LeaveDataPtrDIA(&row, &val, num_diag);
            EXPECT_EQ(num_diag, 4);
            B->SetDataPtrDIA(&row, &val, "B", nnz, m, n, num_diag);
            break;
        }
        case ELL:
        {
            int max_row = 0;
            A->LeaveDataPtrELL(&col, &val, max_row);
            EXPECT_EQ(max_row, 4);
            B->SetDataPtrELL(&col, &val, "B", nnz, m, n, max_row);
            break;
        }
        case DENSE:
            A->LeaveDataPtrDENSE(&val);
            B->SetDataPtrDENSE(&val, "B", m, n);
            break;
        }

        EXPECT_EQ(row, nullptr);
        EXPECT_EQ(col, nullptr);
        EXPECT_EQ(val, nullptr);
        EXPECT_EQ(A->GetM(), 0);
        EXPECT_EQ(A->GetNnz(), 0);
        expect_kept(*B, ref);
    });
}

TEST_P(local_matrix_data_ptr, leave_data_ptr_converts_to_the_requested_format)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();
        auto         A   = make_local_matrix(backend(), ref);
        auto         B   = empty_local_matrix<T>(backend());

        int* row = nullptr;
        int* col = nullptr;
        T*   val = nullptr;
        switch(format().id)
        {
        case CSR:
        {
            A->ConvertToCOO();
            PtrType* row_offset = nullptr;
            A->LeaveDataPtrCSR(&row_offset, &col, &val);
            B->SetDataPtrCSR(&row_offset, &col, &val, "B", ref.nnz(), ref.n, ref.m);
            break;
        }
        case MCSR:
            A->LeaveDataPtrMCSR(&row, &col, &val);
            B->SetDataPtrMCSR(&row, &col, &val, "B", ref.nnz(), ref.n, ref.m);
            break;
        case BCSR:
        {
            int blockdim = format().blockdim;
            A->LeaveDataPtrBCSR(&row, &col, &val, blockdim);
            EXPECT_EQ(blockdim, format().blockdim);
            auto blocks = make_local_matrix(Backend::Host, ref, format());
            B->SetDataPtrBCSR(&row,
                              &col,
                              &val,
                              "B",
                              blocks->GetNnz() / (blockdim * blockdim),
                              ref.n / blockdim,
                              ref.m / blockdim,
                              blockdim);
            break;
        }
        case COO:
            A->LeaveDataPtrCOO(&row, &col, &val);
            B->SetDataPtrCOO(&row, &col, &val, "B", ref.nnz(), ref.n, ref.m);
            break;
        case DIA:
        {
            int num_diag = 0;
            A->LeaveDataPtrDIA(&row, &val, num_diag);
            B->SetDataPtrDIA(&row, &val, "B", num_diag * ref.n, ref.n, ref.m, num_diag);
            break;
        }
        case ELL:
        {
            int max_row = 0;
            A->LeaveDataPtrELL(&col, &val, max_row);
            B->SetDataPtrELL(&col, &val, "B", max_row * ref.n, ref.n, ref.m, max_row);
            break;
        }
        case DENSE:
            A->LeaveDataPtrDENSE(&val);
            B->SetDataPtrDENSE(&val, "B", ref.n, ref.m);
            break;
        }

        EXPECT_EQ(A->GetNnz(), 0);
        expect_kept(*B, ref);
    });
}

TYPED_TEST(local_matrix, copies_from_and_to_csr_and_coo_arrays)
{
    using T = typename TypeParam::value_type;

    const Csr<T>  ref     = banded<T>();
    const Csr<T>  doubled = transformed(ref, [](int, int, T v) { return 2 * v; });
    const int64_t nnz     = ref.nnz();

    // Arrays of the backend, device memory on the accelerator
    PtrType* row_offset = nullptr;
    int*     col        = nullptr;
    T*       val        = nullptr;
    make_local_matrix(this->backend(), ref)->LeaveDataPtrCSR(&row_offset, &col, &val);

    auto B = empty_local_matrix<T>(this->backend());
    B->AllocateCSR("B", nnz, ref.n, ref.m);
    B->CopyFromCSR(row_offset, col, val);
    expect_matrix(*B, ref, csr_format, this->backend());

    make_local_matrix(this->backend(), doubled)->CopyToCSR(row_offset, col, val);
    auto C = empty_local_matrix<T>(this->backend());
    C->SetDataPtrCSR(&row_offset, &col, &val, "C", nnz, ref.n, ref.m);
    expect_matrix(*C, doubled, csr_format, this->backend());

    int* row = nullptr;
    make_local_matrix(this->backend(), ref)->LeaveDataPtrCOO(&row, &col, &val);

    auto D = empty_local_matrix<T>(this->backend());
    D->AllocateCOO("D", nnz, ref.n, ref.m);
    D->CopyFromCOO(row, col, val);
    expect_matrix(*D, ref, Format{COO, 1}, this->backend());

    make_local_matrix(this->backend(), doubled, Format{COO, 1})->CopyToCOO(row, col, val);
    auto E = empty_local_matrix<T>(this->backend());
    E->SetDataPtrCOO(&row, &col, &val, "E", nnz, ref.n, ref.m);
    expect_matrix(*E, doubled, Format{COO, 1}, this->backend());

    std::string info = logged([&] { B->Info(); });
    EXPECT_NE(info.find("Imported from CSR matrix"), std::string::npos) << info;
    info = logged([&] { D->Info(); });
    EXPECT_NE(info.find("Imported from COO matrix"), std::string::npos) << info;
}

TYPED_TEST(local_matrix, copy_from_host_csr_keeps_the_backend)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref = rectangular<T>(8, 12);

    auto A = empty_local_matrix<T>(this->backend());
    A->CopyFromHostCSR(
        ref.row_offset.data(), ref.col.data(), ref.val.data(), "A", ref.nnz(), ref.n, ref.m);
    expect_matrix(*A, ref, csr_format, this->backend());

    A->ConvertToCOO();
    A->CopyFromHostCSR(
        ref.row_offset.data(), ref.col.data(), ref.val.data(), "A", ref.nnz(), ref.n, ref.m);
    expect_matrix(*A, ref, csr_format, this->backend());

    // Unlike AllocateCSR, the sizes of a matrix without entries are kept
    const std::vector<PtrType> no_entries(5, 0);
    A->CopyFromHostCSR(no_entries.data(), nullptr, nullptr, "A", 0, 4, 6);
    EXPECT_EQ(A->GetM(), 4);
    EXPECT_EQ(A->GetN(), 6);
    EXPECT_EQ(A->GetNnz(), 0);
    EXPECT_EQ(backend_of(*A), backend_name(this->backend()));

    auto x = make_local_vector(this->backend(), std::vector<T>(6, 1.0));
    auto y = make_local_vector(this->backend(), std::vector<T>(4, 3.0));
    A->Apply(*x, y.get());
    expect_vector(*y, std::vector<T>(4, 0.0), this->backend(), 0);
    A->ApplyAdd(*x, 2.0, y.get());
    expect_vector(*y, std::vector<T>(4, 0.0), this->backend(), 0);
}

TEST_P(local_matrix_formats, clone_and_copy_between_backends)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>  ref = banded<T>();
        auto          A   = make(ref);
        const Backend to  = backend() == Backend::Host && available(Backend::Accelerator)
                                ? Backend::Accelerator
                                : Backend::Host;

        // CloneFrom takes the format and the backend of the source
        auto C = empty_local_matrix<T>(to);
        C->CloneFrom(*A);
        expect_kept(*C, ref);
        C->Scale(2.0);
        expect_kept(*A, ref);

        // CopyFrom keeps the backend of the destination
        auto D = empty_local_matrix<T>(to);
        D->ConvertTo(format().id, format().blockdim);
        D->CopyFrom(*A);
        expect_matrix(*D, ref, format(), to);

        auto E = empty_local_matrix<T>(to);
        E->ConvertTo(format().id, format().blockdim);
        E->CopyFromAsync(*A);
        E->Sync();
        expect_matrix(*E, ref, format(), to);
    });
}

TYPED_TEST(local_matrix, update_values_csr_keeps_the_pattern)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref    = banded<T>();
    const Csr<T> scaled = transformed(ref, [](int i, int j, T v) { return v + i - j; });

    auto A = make_local_matrix(this->backend(), ref);
    A->UpdateValuesCSR(const_cast<T*>(scaled.val.data()));
    expect_matrix(*A, scaled, csr_format, this->backend());
}

TEST_P(local_matrix_formats, check_accepts_valid_matrices)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        EXPECT_TRUE(make(banded<T>())->Check());
        EXPECT_TRUE(make(spd<T>())->Check());
        EXPECT_TRUE(empty<T>()->Check());

        if(is(CSR) && backend() == Backend::Host)
        {
            Csr<T> bad = banded<T>(4);
            bad.col[1] = 7;
            auto A     = empty_local_matrix<T>(backend());
            A->AllocateCSR("A", bad.nnz(), bad.n, bad.m);
            A->CopyFromCSR(bad.row_offset.data(), bad.col.data(), bad.val.data());
            EXPECT_FALSE(A->Check());
        }
    });
}

// Conversions and moves

TEST_P(local_matrix_formats, converts_between_formats_on_its_backend)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();

        auto        A   = make_local_matrix(backend(), ref);
        std::string log = logged([&] { A->ConvertTo(format().id, format().blockdim); });
        expect_kept(*A, ref);
        EXPECT_EQ(log.find("ConvertTo() is performed on the host") == std::string::npos,
                  backend() == Backend::Host || native("ConvertFromCSR"))
            << log;

        // Between two formats other than CSR
        A->ConvertToDENSE();
        expect_matrix(*A, ref, Format{DENSE, 1}, backend());
        A->ConvertTo(format().id, format().blockdim);
        expect_kept(*A, ref);

        A->ConvertToCSR();
        expect_matrix(*A, ref, csr_format, backend());

        auto B = make(ref);
        B->ConvertToCOO();
        B->ConvertToMCSR();
        B->ConvertToELL();
        B->ConvertToDIA();
        B->ConvertToHYB();
        B->ConvertToBCSR(3);
        expect_matrix(*B, ref, Format{BCSR, 3}, backend());
        B->ConvertTo(format().id, format().blockdim);
        if(is(BCSR))
        {
            KNOWN_BUG("ConvertTo keeps the block dimension of a BCSR matrix");
        }
        expect_kept(*B, ref);
    });
}

TYPED_TEST(local_matrix, conversion_falls_back_to_csr_for_unsuitable_matrices)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref = arrow<T>(12);

    for(unsigned int format : {DIA, ELL})
    {
        SCOPED_TRACE(rocalution::_matrix_format_names[format]);
        auto        A   = make_local_matrix(this->backend(), ref);
        std::string log = logged([&] { A->ConvertTo(format); });
        expect_matrix(*A, ref, csr_format, this->backend());
        EXPECT_NE(log.find(std::string("Matrix conversion to ")
                           + rocalution::_matrix_format_names[format]
                           + " failed, falling back to CSR format"),
                  std::string::npos)
            << log;
    }
}

TEST_P(local_matrix_formats, moves_keep_the_format_and_the_values)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();
        auto         A   = make(ref);

        A->MoveToHost();
        expect_matrix(*A, ref, format(), Backend::Host);
        A->MoveToHost();
        expect_matrix(*A, ref, format(), Backend::Host);

        if(available(Backend::Accelerator))
        {
            A->MoveToAccelerator();
            expect_matrix(*A, ref, format(), Backend::Accelerator);
            A->MoveToAccelerator();
            expect_matrix(*A, ref, format(), Backend::Accelerator);

            A->MoveToHost();
            A->MoveToAcceleratorAsync();
            A->Sync();
            expect_matrix(*A, ref, format(), Backend::Accelerator);

            KNOWN_BUG("Sync after MoveToHostAsync deletes the matrix");
            expect_in_child([&] {
                auto B = make(ref);
                B->MoveToAccelerator();
                B->MoveToHostAsync();
                B->Sync();
                expect_matrix(*B, ref, format(), Backend::Host);
            });
        }
    });
}

TYPED_TEST(local_matrix, async_moves_complete_on_sync)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref = banded<T>();

    auto A = make_local_matrix(Backend::Host, ref);
    A->MoveToAcceleratorAsync();
    A->Sync();
    expect_matrix(*A,
                  ref,
                  csr_format,
                  available(Backend::Accelerator) ? Backend::Accelerator : Backend::Host);

    auto B = make_local_matrix(this->other(), ref);
    auto C = empty_local_matrix<T>(this->backend());
    C->CopyFromAsync(*B);
    C->Sync();
    expect_matrix(*C, ref, csr_format, this->backend());

    if(available(Backend::Accelerator))
    {
        KNOWN_BUG("Sync after MoveToHostAsync deletes the matrix");
        expect_in_child([&] {
            auto D = make_local_matrix(Backend::Accelerator, ref);
            D->MoveToHostAsync();
            D->Sync();
            expect_matrix(*D, ref, csr_format, Backend::Host);
        });
    }
}

// Apply

TEST_P(local_matrix_formats, apply_and_apply_add_match_the_reference)
{
    const bool apply_add_bug = backend() == Backend::Accelerator && (is(HYB) || is(DENSE));

    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        for(const auto& named : matrices<T>(format()))
        {
            SCOPED_TRACE(named.first);
            const Csr<T>&        ref = named.second;
            const std::vector<T> x   = rhs<T>(ref.m);
            const std::vector<T> y0(ref.n, 0.5);

            auto A  = make(ref);
            auto vx = vector_of(x);
            auto vy = vector_of(y0);

            A->Apply(*vx, vy.get());
            expect_vector(*vy, ref.apply(x), backend(), scaled_tol<T>(1e-14));

            std::vector<T> expected = ref.apply(x);
            for(int i = 0; i < ref.n; ++i)
            {
                expected[i] = static_cast<T>(0.5) + 3 * expected[i];
            }
            vy = vector_of(y0);
            A->ApplyAdd(*vx, 3.0, vy.get());
            expect_kept(*A, ref);
            if(apply_add_bug && !run_known_bugs())
            {
                continue;
            }
            expect_vector(*vy, expected, backend(), scaled_tol<T>(1e-14));
        }
    });

    if(apply_add_bug)
    {
        KNOWN_BUG("HIP HYB and DENSE ApplyAdd ignore the output");
    }
}

TEST_P(local_matrix_formats, apply_of_a_rectangular_matrix)
{
    if(is(MCSR))
    {
        GTEST_SKIP() << "MCSR holds square matrices";
    }

    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>         ref = rectangular<T>(8, 12);
        const std::vector<T> x   = rhs<T>(12);
        auto                 A   = make(ref);
        auto                 vx  = vector_of(x);
        auto                 vy  = vector_of(std::vector<T>(8, 0.0));

        A->Apply(*vx, vy.get());
        if(backend() == Backend::Host && A->GetFormat() == DIA)
        {
            KNOWN_BUG("host DIA Apply drops the entries in the columns from nrow on");
        }
        expect_vector(*vy, ref.apply(x), backend(), scaled_tol<T>(1e-14));
    });
}

// Values

TEST_P(local_matrix_formats, zeros_keeps_the_pattern)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>  ref = banded<T>();
        auto          A   = make(ref);
        const int64_t nnz = A->GetNnz();

        A->Zeros();
        EXPECT_EQ(A->GetNnz(), nnz);
        EXPECT_EQ(rocalution::_matrix_format_names[A->GetFormat()], format_name(format()));
        EXPECT_EQ(backend_of(*A), backend_name(backend()));
        for(T v : csr_of(*A).val)
        {
            EXPECT_EQ(v, 0.0);
        }
    });
}

TEST_P(local_matrix_formats, scales_and_adds_scalars)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        using Transform = std::function<T(int, int, T)>;
        struct Case
        {
            std::string                          name;
            std::function<void(LocalMatrix<T>&)> apply;
            Transform                            f;
        };

        const std::vector<Case> cases = {
            {"Scale", [](LocalMatrix<T>& A) { A.Scale(2.0); }, [](int, int, T v) { return 2 * v; }},
            {"ScaleDiagonal",
             [](LocalMatrix<T>& A) { A.ScaleDiagonal(3.0); },
             [](int i, int j, T v) { return i == j ? 3 * v : v; }},
            {"ScaleOffDiagonal",
             [](LocalMatrix<T>& A) { A.ScaleOffDiagonal(-2.0); },
             [](int i, int j, T v) { return i == j ? v : -2 * v; }},
            {"AddScalar",
             [](LocalMatrix<T>& A) { A.AddScalar(1.5); },
             [](int, int, T v) { return v + 1.5; }},
            {"AddScalarDiagonal",
             [](LocalMatrix<T>& A) { A.AddScalarDiagonal(4.0); },
             [](int i, int j, T v) { return i == j ? v + 4 : v; }},
            {"AddScalarOffDiagonal",
             [](LocalMatrix<T>& A) { A.AddScalarOffDiagonal(-0.5); },
             [](int i, int j, T v) { return i == j ? v : v - 0.5; }},
        };

        for(const Case& c : cases)
        {
            SCOPED_TRACE(c.name);
            auto         A        = make(banded<T>());
            const Csr<T> expected = transformed(csr_of(*A), c.f);

            std::string log = logged([&] { c.apply(*A); });
            expect_kept(*A, expected);
            expect_fallback_warning(log, c.name, native(c.name));
        }
    });
}

// Extraction and replacement

TEST_P(local_matrix_formats, extracts_the_diagonal_and_its_inverse)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>   ref = banded<T>();
        std::vector<T> diagonal(ref.n);
        std::vector<T> inverse(ref.n);
        for(int i = 0; i < ref.n; ++i)
        {
            diagonal[i] = static_cast<T>(50 * i + 150);
            inverse[i]  = 1 / diagonal[i];
        }

        auto           A = make(ref);
        LocalVector<T> d;
        LocalVector<T> inv;
        if(backend() == Backend::Accelerator)
        {
            d.MoveToAccelerator();
            inv.MoveToAccelerator();
        }

        std::string log = logged([&] { A->ExtractDiagonal(&d); });
        expect_vector(d, diagonal, backend(), 0);
        expect_fallback_warning(log, "ExtractDiagonal", native("ExtractDiagonal"));

        A->ExtractInverseDiagonal(&inv);
        expect_vector(inv, inverse, backend(), scaled_tol<T>(1e-15));
        expect_kept(*A, ref);
    });
}

TEST_P(local_matrix_formats, extracts_sub_matrices)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();
        auto         A   = make(ref);

        auto        S   = empty_local_matrix<T>(backend());
        std::string log = logged([&] { A->ExtractSubMatrix(2, 3, 6, 8, S.get()); });
        expect_kept_or_csr(*S, block(ref, 2, 3, 6, 8), log);

        if(!is(BCSR))
        {
            auto R = empty_local_matrix<T>(backend());
            log    = logged([&] { A->ExtractSubMatrix(5, 0, 1, 12, R.get()); });
            expect_kept_or_csr(*R, block(ref, 5, 0, 1, 12), log);
        }

        auto extract_blocks = [&] {
            const std::vector<int> row_offset = {0, 6, 12};
            const std::vector<int> col_offset = {0, 4, 8, 12};

            LocalMatrix<T>*** blocks = new LocalMatrix<T>**[2];
            for(int i = 0; i < 2; ++i)
            {
                blocks[i] = new LocalMatrix<T>*[3];
                for(int j = 0; j < 3; ++j)
                {
                    blocks[i][j] = new LocalMatrix<T>;
                    if(backend() == Backend::Accelerator)
                    {
                        blocks[i][j]->MoveToAccelerator();
                    }
                }
            }

            auto        M         = make(ref);
            std::string block_log = logged(
                [&] { M->ExtractSubMatrices(2, 3, row_offset.data(), col_offset.data(), blocks); });
            for(int i = 0; i < 2; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    SCOPED_TRACE("block " + std::to_string(i) + "," + std::to_string(j));
                    expect_kept_or_csr(*blocks[i][j], block(ref, 6 * i, 4 * j, 6, 4), block_log);
                    delete blocks[i][j];
                }
                delete[] blocks[i];
            }
            delete[] blocks;
        };

        if(is(DIA))
        {
            KNOWN_BUG("csr_to_dia writes past the values of a matrix with more rows than columns");
            expect_in_child(extract_blocks);
        }
        else
        {
            extract_blocks();
        }
    });
}

TEST_P(local_matrix_formats, extracts_the_triangular_parts)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();
        auto         A   = make(ref);

        for(bool diagonal : {true, false})
        {
            SCOPED_TRACE(diagonal ? "with the diagonal" : "without the diagonal");
            auto        L     = empty_local_matrix<T>(backend());
            auto        U     = empty_local_matrix<T>(backend());
            std::string log_l = logged([&] { A->ExtractL(L.get(), diagonal); });
            std::string log_u = logged([&] { A->ExtractU(U.get(), diagonal); });
            expect_kept_or_csr(*L, lower(ref, diagonal), log_l);
            expect_kept_or_csr(*U, upper(ref, diagonal), log_u);
        }
        expect_kept(*A, ref);
    });
}

TEST_P(local_matrix_formats, extracts_and_replaces_rows_and_columns)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();
        auto         A   = make(ref);

        auto c = vector_of(std::vector<T>(ref.n, 0.0));
        auto r = vector_of(std::vector<T>(ref.m, 0.0));
        A->ExtractColumnVector(4, c.get());
        A->ExtractRowVector(7, r.get());
        expect_vector(*c, column(ref, 4), backend(), 0);
        expect_vector(*r, row(ref, 7), backend(), 0);
        expect_kept(*A, ref);

        // The new row and column only change existing entries
        std::vector<T> new_row = row(ref, 7);
        for(T& v : new_row)
        {
            v = v == 0 ? 0 : -v;
        }
        std::vector<T> D = dense(ref);
        std::copy(new_row.begin(), new_row.end(), D.begin() + 7 * ref.m);
        const Csr<T> row_replaced = from_dense(ref.n, ref.m, D);

        A->ReplaceRowVector(7, *vector_of(new_row));
        expect_kept(*A, row_replaced);

        std::vector<T> new_column = column(row_replaced, 4);
        for(T& v : new_column)
        {
            v *= 2;
        }
        for(int i = 0; i < ref.n; ++i)
        {
            D[static_cast<size_t>(i) * ref.m + 4] = new_column[i];
        }
        const Csr<T> column_replaced = from_dense(ref.n, ref.m, D);

        // A column with new entries
        std::vector<T> full_column(ref.n, 1.0);
        for(int i = 0; i < ref.n; ++i)
        {
            D[static_cast<size_t>(i) * ref.m + 9] = 1.0;
        }
        const Csr<T> column_inserted = from_dense(ref.n, ref.m, D);

        if(backend() == Backend::Accelerator && is(CSR))
        {
            KNOWN_BUG("HIP CSR ReplaceColumnVector kills the process");
            expect_in_child([&] {
                auto M = make(row_replaced);
                M->ReplaceColumnVector(4, *vector_of(new_column));
                expect_kept(*M, column_replaced);
                M->ReplaceColumnVector(9, *vector_of(full_column));
                expect_kept(*M, column_inserted);
            });
            return;
        }
        A->ReplaceColumnVector(4, *vector_of(new_column));
        expect_kept(*A, column_replaced);

        // The BCSR blocks already store the column, so only the other formats insert entries
        if(is(DENSE) || is(BCSR))
        {
            A->ReplaceColumnVector(9, *vector_of(full_column));
            expect_kept(*A, column_inserted);
            return;
        }

        // The slot of the dropped entry stays uninitialized, which can corrupt memory
        // when the result is converted back
        KNOWN_BUG("host CSR ReplaceColumnVector drops the entry after an inserted one");
        expect_in_child([&] {
            auto M = make(column_replaced);
            M->ReplaceColumnVector(9, *vector_of(full_column));
            expect_kept(*M, column_inserted);
        });
    });
}

// Factorizations and triangular solves

TEST_P(local_matrix_formats, lu_factorizations_solve_a_banded_system)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>         ref   = banded<T>();
        const std::vector<T> b     = rhs<T>(ref.n);
        int                  niter = 0;

        using Factorize = std::function<void(LocalMatrix<T>&)>;
        const std::vector<std::tuple<std::string, std::string, Factorize>> factorizations = {
            {"ILU0Factorize", "ILU0Factorize", [](LocalMatrix<T>& A) { A.ILU0Factorize(); }},
            {"ILUTFactorize", "ILUTFactorize", [](LocalMatrix<T>& A) { A.ILUTFactorize(0, 12); }},
            {"LUFactorize", "LUFactorize", [](LocalMatrix<T>& A) { A.LUFactorize(); }},
            {"ILUpFactorize(0)", "", [](LocalMatrix<T>& A) { A.ILUpFactorize(0); }},
            {"ItILU0Factorize",
             "",
             [&](LocalMatrix<T>& A) {
                 A.ItILU0Factorize(
                     rocalution::Default, 0, 200, scaled_tol<T>(1e-14), &niter, nullptr);
             }},
            {"ILUpFactorize(1) with levels", "", [](LocalMatrix<T>& A) { A.ILUpFactorize(1); }},
            {"ILUpFactorize(1) without levels",
             "",
             [](LocalMatrix<T>& A) { A.ILUpFactorize(1, false); }},
        };

        auto factorize_and_solve = [&](const auto& factorization) {
            SCOPED_TRACE(std::get<0>(factorization));
            auto        A   = make(ref);
            std::string log = logged([&] { std::get<2>(factorization)(*A); });
            EXPECT_EQ(rocalution::_matrix_format_names[A->GetFormat()], format_name(format()));
            EXPECT_EQ(backend_of(*A), backend_name(backend()));
            if(!std::get<1>(factorization).empty())
            {
                expect_fallback_warning(
                    log, std::get<1>(factorization), native(std::get<1>(factorization)));
            }

            const bool analysed = has_analysis(backend(), format().id, "LU");
            if(analysed)
            {
                A->LUAnalyse();
            }
            auto x = vector_of(std::vector<T>(ref.n, 0.0));
            log    = logged([&] { A->LUSolve(*vector_of(b), x.get()); });
            EXPECT_EQ(backend_of(*x), backend_name(backend()));
            expect_close(ref.apply(values_of(*x)), b, loose_tol<T>(1e-10));
            expect_fallback_warning(log, "LUSolve", native("LUSolve"));
            if(analysed)
            {
                A->LUAnalyseClear();
            }
        };

        // The cases a known bug ends the process in run last, each in a child process
        struct Killed
        {
            const char* reason;
            size_t      index;
        };
        std::vector<Killed> killed;
        for(size_t k = 0; k < factorizations.size(); ++k)
        {
            const std::string& name = std::get<0>(factorizations[k]);
            if(name.find("ILUpFactorize(1)") == 0 && (is(DIA) || is(DENSE)))
            {
                killed.push_back(
                    {"the SymbolicPower structure loses its zero entries in DIA and DENSE", k});
            }
            else if(name == "ILUpFactorize(1) without levels" && is(BCSR))
            {
                killed.push_back(
                    {"the MatrixAdd fallback converts back to BCSR with block dimension 1", k});
            }
            else
            {
                factorize_and_solve(factorizations[k]);
            }
        }
        for(const Killed& known : killed)
        {
            KNOWN_BUG(known.reason);
            expect_in_child([&] { factorize_and_solve(factorizations[known.index]); });
        }
    });
}

TEST_P(local_matrix_formats, incomplete_cholesky_solves_a_tridiagonal_system)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>         ref = spd<T>();
        const std::vector<T> b   = rhs<T>(ref.n);

        // As in the IC preconditioner, the factorization takes the lower triangle
        auto A   = make(lower(ref, true));
        auto inv = vector_of(std::vector<T>());
        A->ICFactorize(inv.get());
        EXPECT_EQ(rocalution::_matrix_format_names[A->GetFormat()], format_name(format()));
        EXPECT_EQ(backend_of(*A), backend_name(backend()));

        const bool analysed = has_analysis(backend(), format().id, "LL");
        if(analysed)
        {
            A->LLAnalyse();
        }
        auto        x   = vector_of(std::vector<T>(ref.n, 0.0));
        std::string log = logged([&] { A->LLSolve(*vector_of(b), x.get()); });
        EXPECT_EQ(backend_of(*x), backend_name(backend()));
        expect_fallback_warning(log, "LLSolve", native("LLSolve"));

        auto y = vector_of(std::vector<T>(ref.n, 0.0));
        A->LLSolve(*vector_of(b), *inv, y.get());
        EXPECT_EQ(backend_of(*y), backend_name(backend()));
        if(analysed)
        {
            A->LLAnalyseClear();
        }

        // The BCSR blocks store zeros above the diagonal
        if(backend() == Backend::Host && is(BCSR))
        {
            KNOWN_BUG("host LLSolve takes the last entry of a row as its diagonal");
        }
        expect_close(ref.apply(values_of(*x)), b, loose_tol<T>(1e-10));
        expect_close(ref.apply(values_of(*y)), b, loose_tol<T>(1e-10));

        if(backend() == Backend::Host)
        {
            EXPECT_EQ(inv->GetSize(), ref.n);
            return;
        }

        // Unlike the HIP one, the host LLSolve uses the inverse diagonal
        auto M = make(lower(ref, true));
        auto d = vector_of(std::vector<T>());
        M->ICFactorize(d.get());
        EXPECT_EQ(d->GetSize(), ref.n);
        M->MoveToHost();
        d->MoveToHost();
        auto z = make_local_vector(Backend::Host, std::vector<T>(ref.n, 0.0));
        M->LLSolve(*make_local_vector(Backend::Host, b), *d, z.get());
        expect_close(ref.apply(values_of(*z)), b, loose_tol<T>(1e-10));
    });
}

TEST_P(local_matrix_formats, triangular_solves_use_the_triangular_parts)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>         ref      = banded<T>();
        const std::vector<T> b        = rhs<T>(ref.n);
        const bool           analysed = has_analysis(backend(), format().id, "L");

        // Only the analysis sets a unit diagonal
        for(bool unit : {false, true})
        {
            if(unit && !analysed)
            {
                continue;
            }
            SCOPED_TRACE(unit ? "unit diagonal" : "stored diagonal");
            const Csr<T> L = unit ? unit_triangle(lower(ref, false)) : lower(ref, true);
            const Csr<T> U = unit ? unit_triangle(upper(ref, false)) : upper(ref, true);

            auto A = make(ref);
            if(analysed)
            {
                A->LAnalyse(unit);
                A->UAnalyse(unit);
            }

            auto        x   = vector_of(std::vector<T>(ref.n, 0.0));
            std::string log = logged([&] { A->LSolve(*vector_of(b), x.get()); });
            EXPECT_EQ(backend_of(*x), backend_name(backend()));
            expect_solves(L, values_of(*x), b, tol<T>());
            expect_fallback_warning(log, "LSolve", native("LSolve"));

            auto y = vector_of(std::vector<T>(ref.n, 0.0));
            log    = logged([&] { A->USolve(*vector_of(b), y.get()); });
            EXPECT_EQ(backend_of(*y), backend_name(backend()));
            expect_solves(U, values_of(*y), b, tol<T>());
            expect_fallback_warning(log, "USolve", native("USolve"));

            if(analysed)
            {
                A->LAnalyseClear();
                A->UAnalyseClear();
            }
            expect_kept(*A, ref);
        }
    });
}

TYPED_TEST(local_matrix, iterative_triangular_solves_converge)
{
    using T = typename TypeParam::value_type;

    const Csr<T>         ref   = banded<T>();
    const std::vector<T> b     = rhs<T>(ref.n);
    const int            iters = 2 * ref.n;

    for(bool unit : {false, true})
    {
        SCOPED_TRACE(unit ? "unit diagonal" : "stored diagonal");
        auto A = make_local_matrix(this->backend(), ref);
        auto x = make_local_vector(this->backend(), std::vector<T>(ref.n, 0.0));

        A->ItLAnalyse(unit);
        A->ItLSolve(iters, 0, false, *make_local_vector(this->backend(), b), x.get());
        expect_solves(
            unit ? unit_triangle(lower(ref, false)) : lower(ref, true), values_of(*x), b, tol<T>());
        auto xt = make_local_vector(this->backend(), std::vector<T>(ref.n, 0.0));
        A->ItLSolve(
            iters, scaled_tol<T>(1e-14), true, *make_local_vector(this->backend(), b), xt.get());
        expect_solves(unit ? unit_triangle(lower(ref, false)) : lower(ref, true),
                      values_of(*xt),
                      b,
                      loose_tol<T>(1e-10));
        A->ItLAnalyseClear();

        A->ItUAnalyse(unit);
        A->ItUSolve(iters, 0, false, *make_local_vector(this->backend(), b), x.get());
        expect_solves(
            unit ? unit_triangle(upper(ref, false)) : upper(ref, true), values_of(*x), b, tol<T>());
        auto yt = make_local_vector(this->backend(), std::vector<T>(ref.n, 0.0));
        A->ItUSolve(
            iters, scaled_tol<T>(1e-14), true, *make_local_vector(this->backend(), b), yt.get());
        expect_solves(unit ? unit_triangle(upper(ref, false)) : upper(ref, true),
                      values_of(*yt),
                      b,
                      loose_tol<T>(1e-10));
        A->ItUAnalyseClear();
        EXPECT_EQ(backend_of(*x), backend_name(this->backend()));
    }

    auto LU = make_local_matrix(this->backend(), ref);
    LU->ILU0Factorize();
    LU->ItLUAnalyse();
    auto x = make_local_vector(this->backend(), std::vector<T>(ref.n, 0.0));
    LU->ItLUSolve(iters, 0, false, *make_local_vector(this->backend(), b), x.get());
    expect_close(ref.apply(values_of(*x)), b, tol<T>());

    const Csr<T> s   = spd<T>();
    auto         LL  = make_local_matrix(this->backend(), lower(s, true));
    auto         inv = make_local_vector(this->backend(), std::vector<T>());
    LL->ICFactorize(inv.get());
    LL->ItLLAnalyse();
    auto y = make_local_vector(this->backend(), std::vector<T>(s.n, 0.0));
    LL->ItLLSolve(iters, 0, false, *make_local_vector(this->backend(), b), y.get());
    expect_close(s.apply(values_of(*y)), b, tol<T>());

    auto z = make_local_vector(this->backend(), std::vector<T>(s.n, 0.0));
    LL->ItLLSolve(iters, 0, false, *make_local_vector(this->backend(), b), *inv, z.get());
    expect_close(s.apply(values_of(*z)), b, tol<T>());

    // With a tolerance the iteration may stop early, at the requested accuracy
    auto w = make_local_vector(this->backend(), std::vector<T>(ref.n, 0.0));
    LU->ItLUSolve(
        iters, scaled_tol<T>(1e-14), true, *make_local_vector(this->backend(), b), w.get());
    auto v = make_local_vector(this->backend(), std::vector<T>(s.n, 0.0));
    LL->ItLLSolve(
        iters, scaled_tol<T>(1e-14), true, *make_local_vector(this->backend(), b), v.get());
    auto u = make_local_vector(this->backend(), std::vector<T>(s.n, 0.0));
    LL->ItLLSolve(
        iters, scaled_tol<T>(1e-14), true, *make_local_vector(this->backend(), b), *inv, u.get());
    LU->ItLUAnalyseClear();
    LL->ItLLAnalyseClear();
    if(this->backend() == Backend::Host)
    {
        KNOWN_BUG(
            "host ItLUSolve and ItLLSolve give the second solve the first one's iteration count");
    }
    expect_close(ref.apply(values_of(*w)), b, loose_tol<T>(1e-10));
    if(this->backend() == Backend::Host)
    {
        KNOWN_BUG("the transposed host iterative triangular solve does not keep the largest "
                  "residual, so with a tolerance ItLLSolve may stop early");
    }
    expect_close(s.apply(values_of(*v)), b, loose_tol<T>(1e-10));
    expect_close(s.apply(values_of(*u)), b, loose_tol<T>(1e-10));
}

TYPED_TEST(local_matrix, iterative_solves_of_other_formats_fall_back_to_csr)
{
    using T = typename TypeParam::value_type;

    const Csr<T>         ref = banded<T>();
    const std::vector<T> b   = rhs<T>(ref.n);

    // The analysis only exists for CSR, so other formats solve without one
    KNOWN_BUG("the It*Solve fallback runs without the analysis and with the wrong vector");
    expect_in_child([&] {
        auto A = make_local_matrix(this->backend(), ref, Format{COO, 1});
        auto x = make_local_vector(this->backend(), std::vector<T>(ref.n, 0.0));
        A->ItLSolve(2 * ref.n, 0, false, *make_local_vector(this->backend(), b), x.get());
        expect_solves(lower(ref, true), values_of(*x), b, tol<T>());
        EXPECT_EQ(backend_of(*x), backend_name(this->backend()));
    });
}

// Dense operations

TEST_P(local_matrix_formats, qr_decomposition_solves_and_inverts)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>         ref = banded<T>();
        const std::vector<T> b   = rhs<T>(ref.n);

        // The factors and the inverse are dense, which DIA and ELL may reject
        auto expect_format = [&](const LocalMatrix<T>& A) {
            const std::string name = rocalution::_matrix_format_names[A.GetFormat()];
            if(is(DIA) || is(ELL))
            {
                EXPECT_TRUE(name == format_name(format()) || name == "CSR") << name;
            }
            else
            {
                EXPECT_EQ(name, format_name(format()));
            }
            EXPECT_EQ(backend_of(A), backend_name(backend()));
        };

        auto        A   = make(ref);
        std::string log = logged([&] { A->QRDecompose(); });
        expect_format(*A);
        expect_fallback_warning(log, "QRDecompose", native("QRDecompose"));

        auto x = vector_of(std::vector<T>(ref.n, 0.0));
        A->QRSolve(*vector_of(b), x.get());
        EXPECT_EQ(backend_of(*x), backend_name(backend()));
        expect_close(ref.apply(values_of(*x)), b, loose_tol<T>(1e-10));

        auto Inv = make(ref);
        log      = logged([&] { Inv->Invert(); });
        expect_format(*Inv);
        expect_fallback_warning(log, "Invert", native("Invert"));

        std::vector<T> identity(static_cast<size_t>(ref.n) * ref.n, 0);
        for(int i = 0; i < ref.n; ++i)
        {
            identity[static_cast<size_t>(i) * ref.n + i] = 1;
        }
        expect_close(dense_product(dense(csr_of(*Inv)), dense(ref), ref.n, ref.n, ref.n),
                     identity,
                     tol<T>());
    });
}

TEST_P(local_matrix_formats, householder_reflects_a_column_onto_the_diagonal)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();
        const int    idx = 3;
        const int    len = ref.n - idx;

        auto A    = make(ref);
        auto v    = vector_of(std::vector<T>(len, 0.0));
        T    beta = 0;

        std::string log = logged([&] { A->Householder(idx, beta, v.get()); });
        expect_fallback_warning(log, "Householder", native("Householder"));
        EXPECT_EQ(backend_of(*v), backend_name(backend()));
        expect_kept(*A, ref);

        const std::vector<T> c = column(ref, idx);
        const std::vector<T> u = values_of(*v);
        EXPECT_EQ(u[0], 1.0);

        // (I - beta u u^T) x is a multiple of the first unit vector
        T ux   = 0;
        T norm = 0;
        for(int i = 0; i < len; ++i)
        {
            ux += u[i] * c[idx + i];
            norm += c[idx + i] * c[idx + i];
        }
        std::vector<T> reflected(len);
        for(int i = 0; i < len; ++i)
        {
            reflected[i] = c[idx + i] - beta * u[i] * ux;
        }
        EXPECT_NEAR(std::abs(reflected[0]), std::sqrt(norm), tol<T>() * std::sqrt(norm));
        for(int i = 1; i < len; ++i)
        {
            EXPECT_NEAR(reflected[i], 0.0, tol<T>() * std::sqrt(norm)) << "at index " << i;
        }
    });
}

// Permutations and orderings

TEST_P(local_matrix_formats, permute_moves_rows_and_columns)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>           ref  = banded<T>();
        const std::vector<int> perm = scrambled(ref.n);

        auto        A   = make(ref);
        std::string log = logged([&] { A->Permute(*vector_of(perm)); });
        expect_fallback_warning(log, "Permute", native("Permute"));
        if(!is(COO))
        {
            expect_kept(*A, permuted(ref, perm));
            return;
        }

        // The check converts the unsorted entries to CSR, where an assert ends the process
        KNOWN_BUG("COO Permute leaves the entries unsorted");
        expect_in_child([&] {
            auto M = make(ref);
            M->Permute(*vector_of(perm));
            expect_kept(*M, permuted(ref, perm));
        });
    });
}

TEST_P(local_matrix_formats, permute_backward_undoes_permute)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>           ref  = banded<T>();
        const std::vector<int> perm = scrambled(ref.n);

        // PermuteBackward only exists for COO, so every format goes through it, and the unsorted
        // entries break the conversion back: an assert ends the process
        auto permute_backward = [&] {
            auto        B   = make(permuted(ref, perm));
            std::string log = logged([&] { B->PermuteBackward(*vector_of(perm)); });
            expect_fallback_warning(log, "PermuteBackward", is(COO));
            EXPECT_EQ(rocalution::_matrix_format_names[B->GetFormat()], format_name(format()));
            EXPECT_EQ(backend_of(*B), backend_name(backend()));
            return B;
        };
        if(is(COO))
        {
            permute_backward();
        }
        KNOWN_BUG("COO PermuteBackward leaves the entries unsorted");
        expect_in_child([&] { expect_kept(*permute_backward(), ref); });
    });
}

TEST_P(local_matrix_formats, cmk_orderings_reduce_the_bandwidth)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = permuted(spd<T>(), scrambled(12));
        ASSERT_EQ(bandwidth(ref), 7);

        auto             A = make(ref);
        LocalVector<int> cmk;
        LocalVector<int> rcmk;
        if(backend() == Backend::Accelerator)
        {
            cmk.MoveToAccelerator();
            rcmk.MoveToAccelerator();
        }

        std::string log = logged([&] { A->CMK(&cmk); });
        expect_fallback_warning(log, "CMK", native("CMK"));
        log = logged([&] { A->RCMK(&rcmk); });
        expect_fallback_warning(log, "RCMK", native("RCMK"));
        EXPECT_EQ(backend_of(cmk), backend_name(backend()));
        EXPECT_EQ(backend_of(rcmk), backend_name(backend()));
        expect_kept(*A, ref);

        const std::vector<int> c = values_of(cmk);
        const std::vector<int> r = values_of(rcmk);
        expect_permutation(c, ref.n);
        ASSERT_EQ(r.size(), c.size());
        for(int i = 0; i < ref.n; ++i)
        {
            EXPECT_EQ(r[i], ref.n - 1 - c[i]) << "at index " << i;
        }

        // The ordering follows the stored entries, which for BCSR include the zeros of the blocks
        const Csr<T>     stored = csr_of(*A);
        auto             S      = make_local_matrix(Backend::Host, stored);
        LocalVector<int> host_cmk;
        S->CMK(&host_cmk);
        EXPECT_EQ(c, values_of(host_cmk));
        if(!is(BCSR))
        {
            EXPECT_LE(bandwidth(permuted(ref, c)), 2);
            EXPECT_LE(bandwidth(permuted(ref, r)), 2);
        }
    });
}

TEST_P(local_matrix_formats, connectivity_order_sorts_the_rows_by_their_non_zeros)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = permuted(arrow<T>(12), scrambled(12));

        auto             A = make(ref);
        LocalVector<int> perm;
        if(backend() == Backend::Accelerator)
        {
            perm.MoveToAccelerator();
        }
        std::string log = logged([&] { A->ConnectivityOrder(&perm); });
        expect_fallback_warning(log, "ConnectivityOrder", native(*A, "ConnectivityOrder"));
        EXPECT_EQ(backend_of(perm), backend_name(backend()));
        const std::vector<int> p = values_of(perm);
        expect_permutation(p, ref.n);

        // The documented usage: ConnectivityOrder, then Permute
        KNOWN_BUG("ConnectivityOrder returns perm[new] = old");
        const Csr<T> ordered = permuted(ref, p);
        for(int i = 1; i < ordered.n; ++i)
        {
            EXPECT_LE(ordered.row_offset[i] - ordered.row_offset[i - 1],
                      ordered.row_offset[i + 1] - ordered.row_offset[i])
                << "at row " << i;
        }
    });
}

TEST_P(local_matrix_formats, multi_coloring_gives_independent_color_classes)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = pentadiagonal<T>();

        auto             A           = make(ref);
        int              num_colors  = 0;
        int*             size_colors = nullptr;
        LocalVector<int> perm;
        if(backend() == Backend::Accelerator)
        {
            perm.MoveToAccelerator();
        }
        A->MultiColoring(num_colors, &size_colors, &perm);
        EXPECT_EQ(backend_of(perm), backend_name(backend()));
        ASSERT_NE(size_colors, nullptr);
        EXPECT_GE(num_colors, 3);

        std::vector<int> offset(num_colors + 1, 0);
        for(int c = 0; c < num_colors; ++c)
        {
            offset[c + 1] = offset[c] + size_colors[c];
        }
        rocalution::free_host(&size_colors);
        EXPECT_EQ(offset[num_colors], ref.n);

        const std::vector<int> p = values_of(perm);
        expect_permutation(p, ref.n);
        auto color = [&](int i) {
            return static_cast<int>(std::upper_bound(offset.begin(), offset.end(), p[i])
                                    - offset.begin())
                   - 1;
        };
        for(int i = 0; i < ref.n; ++i)
        {
            for(PtrType k = ref.row_offset[i]; k < ref.row_offset[i + 1]; ++k)
            {
                if(ref.col[k] != i)
                {
                    EXPECT_NE(color(i), color(ref.col[k]))
                        << "nodes " << i << " and " << ref.col[k];
                }
            }
        }
        expect_kept(*A, ref);
    });
}

TEST_P(local_matrix_formats, maximal_independent_set_comes_first)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = pentadiagonal<T>();

        auto             A    = make(ref);
        int              size = 0;
        LocalVector<int> perm;
        if(backend() == Backend::Accelerator)
        {
            perm.MoveToAccelerator();
        }
        A->MaximalIndependentSet(size, &perm);
        EXPECT_EQ(backend_of(perm), backend_name(backend()));

        // Independent in the graph of the stored entries, which for BCSR include the zeros of the blocks
        const Csr<T>           graph = csr_of(*A);
        const std::vector<int> p     = values_of(perm);
        expect_permutation(p, ref.n);
        ASSERT_GT(size, 0);
        for(int i = 0; i < ref.n; ++i)
        {
            bool member_neighbour = false;
            for(int j = 0; j < ref.n; ++j)
            {
                if(j != i && connected(graph, i, j) && p[j] < size)
                {
                    member_neighbour = true;
                    EXPECT_FALSE(p[i] < size) << "members " << i << " and " << j << " are adjacent";
                }
            }
            if(p[i] >= size)
            {
                EXPECT_TRUE(member_neighbour) << "node " << i << " could join the set";
            }
        }
    });
}

TEST_P(local_matrix_formats, zero_block_permutation_moves_zero_diagonals_last)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref
            = filtered(pentadiagonal<T>(), [](int i, int j, T) { return !(i == j && i % 3 == 1); });

        auto             A    = make(ref);
        int              size = 0;
        LocalVector<int> perm;
        if(backend() == Backend::Accelerator)
        {
            perm.MoveToAccelerator();
        }
        std::string log = logged([&] { A->ZeroBlockPermutation(size, &perm); });
        expect_fallback_warning(log, "ZeroBlockPermutation", native(*A, "ZeroBlockPermutation"));
        EXPECT_EQ(backend_of(perm), backend_name(backend()));

        // A stored zero counts as a diagonal entry, so BCSR blocks fill in every diagonal here
        const std::vector<std::set<int>> stored = pattern_of(csr_of(*A));
        std::vector<int>                 expected(ref.n);
        int                              with_diagonal = 0;
        for(int i = 0; i < ref.n; ++i)
        {
            with_diagonal += stored[i].count(i) > 0;
        }
        for(int i = 0, first = 0, last = with_diagonal; i < ref.n; ++i)
        {
            expected[i] = stored[i].count(i) > 0 ? first++ : last++;
        }
        EXPECT_EQ(with_diagonal, is(BCSR) ? ref.n : 8);
        EXPECT_EQ(size, with_diagonal);
        EXPECT_EQ(values_of(perm), expected);
    });
}

// Products and structure

TEST_P(local_matrix_formats, matrix_mult_takes_the_format_of_the_factors)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> a = banded<T>();
        const Csr<T> b = two_diagonals<T>();

        auto multiply_into = [&](LocalMatrix<T>& C) {
            C.MatrixMult(*make(a), *make(b));
            expect_kept(C, product(a, b), scaled_tol<T>(1e-14));
        };

        // The second product goes into a matrix of the same size, which MatrixMult
        // clears first
        auto multiply_twice = [&] {
            auto C = empty<T>();
            multiply_into(*C);
            multiply_into(*C);
        };

        if(backend() == Backend::Accelerator && is(MCSR))
        {
            KNOWN_BUG("HIP MCSR CopyToHost copies the row offsets of an empty matrix from NULL");
            expect_in_child(multiply_twice);
            return;
        }

        auto C = empty<T>();
        multiply_into(*C);
        if(backend() == Backend::Accelerator && is(BCSR))
        {
            KNOWN_BUG("BCSR Clear keeps the block counts, so the cleared matrix copies them");
            expect_in_child(multiply_twice);
            return;
        }
        multiply_into(*C);
    });
}

TEST_P(local_matrix_formats, matrix_add_scales_and_adds)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> a = banded<T>();
        const Csr<T> b
            = transformed(a, [](int i, int j, T) { return static_cast<T>((i + 2 * j) % 5 - 1.5); });
        const Csr<T> c = two_diagonals<T>();

        auto add = [](const Csr<T>& x, T alpha, const Csr<T>& y, T beta) {
            std::vector<T> D  = dense(x);
            std::vector<T> Dy = dense(y);
            for(size_t k = 0; k < D.size(); ++k)
            {
                D[k] = alpha * D[k] + beta * Dy[k];
            }
            return from_dense(x.n, x.m, D);
        };

        auto add_to = [&](const Csr<T>& x, T alpha, const Csr<T>& y, T beta, bool structure) {
            auto A = make(x);
            A->MatrixAdd(*make(y), alpha, beta, structure);
            expect_kept(*A, add(x, alpha, y, beta), scaled_tol<T>(1e-14));
        };

        // The pattern of the summand is part of the pattern of the matrix
        const Csr<T> part = lower(a, true);

        // A known bug ends the process in every BCSR MatrixAdd
        if(is(BCSR))
        {
            KNOWN_BUG("the MatrixAdd fallback converts back to BCSR with block dimension 1");
            expect_in_child([&] {
                add_to(a, 2, b, -1, false);
                add_to(a, 0.5, c, 3, true);
            });
            KNOWN_BUG("MatrixAdd does not scale the entries outside the summand's pattern");
            expect_in_child([&] { add_to(a, 2, part, 1, false); });
            return;
        }

        add_to(a, 2, b, -1, false);
        add_to(a, 0.5, c, 3, true);

        auto A = make(a);
        A->MatrixAdd(*make(part), 2.0, 1.0, false);
        KNOWN_BUG("MatrixAdd does not scale the entries outside the summand's pattern");
        expect_kept(*A, add(a, 2, part, 1), scaled_tol<T>(1e-14));
    });
}

TEST_P(local_matrix_formats, triple_matrix_product_in_csr)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> r        = rectangular<T>(6, 12);
        const Csr<T> a        = banded<T>();
        const Csr<T> p        = transposed(rectangular<T>(6, 12));
        const Csr<T> expected = product(product(r, a), p);

        // The factors may come in any format
        const Format other = is(CSR) ? Format{COO, 1} : csr_format;
        auto         R     = make_local_matrix(backend(), r, other);
        auto         A     = make(a);
        auto         P     = make_local_matrix(backend(), p, other);
        auto         C     = empty<T>();
        C->TripleMatrixProduct(*R, *A, *P);

        const std::string name = rocalution::_matrix_format_names[C->GetFormat()];
        if(is(DIA) || is(ELL))
        {
            EXPECT_TRUE(name == format_name(format()) || name == "CSR") << name;
        }
        else
        {
            EXPECT_EQ(name, format_name(format()));
        }
        EXPECT_EQ(backend_of(*C), backend_name(backend()));
        expect_entries(csr_of(*C), expected, scaled_tol<T>(1e-13));

        if(is(CSR))
        {
            auto D = empty<T>();
            D->TripleMatrixProduct(
                *make_local_matrix(backend(), r), *A, *make_local_matrix(backend(), p));
            expect_kept(*D, expected, scaled_tol<T>(1e-13));
        }
    });
}

TYPED_TEST(local_matrix, triple_matrix_product_converts_each_factor_to_csr)
{
    using T = typename TypeParam::value_type;

    const Csr<T> r        = rectangular<T>(6, 12);
    const Csr<T> a        = banded<T>();
    const Csr<T> p        = transposed(rectangular<T>(6, 12));
    const Csr<T> expected = product(product(r, a), p);
    const Format coo      = {COO, 1};

    for(const std::string& factor : {"A", "P"})
    {
        SCOPED_TRACE(factor + " in COO");
        auto R = make_local_matrix(this->backend(), r);
        auto A = make_local_matrix(this->backend(), a, factor == "A" ? coo : csr_format);
        auto P = make_local_matrix(this->backend(), p, factor == "P" ? coo : csr_format);
        auto C = empty_local_matrix<T>(this->backend());

        std::string log = logged([&] { C->TripleMatrixProduct(*R, *A, *P); });
        expect_matrix(*C, expected, csr_format, this->backend(), scaled_tol<T>(1e-13));
        EXPECT_NE(log.find("TripleMatrixProduct() is performed in CSR format"), std::string::npos)
            << log;
    }
}

// Row map[i] of the restriction holds a one in column i
TYPED_TEST(local_matrix, create_from_map_gives_the_restriction)
{
    using T                    = typename TypeParam::value_type;
    const std::vector<int> map = {0, 2, 1, 1, 0, 2, 1, 2, 0};
    const int              n   = static_cast<int>(map.size());
    const int              m   = 3;

    Csr<T> restriction(m, n);
    for(int r = 0; r < m; ++r)
    {
        for(int i = 0; i < n; ++i)
        {
            if(map[i] == r)
            {
                restriction.add(i, T(1));
            }
        }
        restriction.end_row();
    }

    auto R = make_local_matrix(this->backend(), banded<T>(4));
    R->CreateFromMap(*make_local_vector(this->backend(), map), n, m);
    expect_matrix(*R, restriction, csr_format, this->backend());

    KNOWN_BUG("CreateFromMap(map, n, m) leaves a matrix without entries empty");

    auto E = empty_local_matrix<T>(this->backend());
    E->CreateFromMap(*make_local_vector(this->backend(), map), n, m);
    expect_matrix(*E, restriction, csr_format, this->backend());
}

TEST_P(local_matrix_formats, diagonal_matrix_mult_scales_rows_and_columns)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>   ref = banded<T>();
        std::vector<T> d(ref.n);
        for(int i = 0; i < ref.n; ++i)
        {
            d[i] = static_cast<T>(i + 1);
        }

        auto A = make(ref);
        A->DiagonalMatrixMultL(*vector_of(d));
        expect_kept(*A, transformed(ref, [&](int i, int, T v) { return d[i] * v; }));

        A = make(ref);
        A->DiagonalMatrixMultR(*vector_of(d));
        expect_kept(*A, transformed(ref, [&](int, int j, T v) { return v * d[j]; }));

        A = make(ref);
        A->DiagonalMatrixMult(*vector_of(d));
        expect_kept(*A, transformed(ref, [&](int, int j, T v) { return v * d[j]; }));
    });
}

TEST_P(local_matrix_formats, transposes)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();

        auto A = make(ref);
        A->Transpose();
        expect_kept(*A, transposed(ref));

        auto transpose = empty_local_matrix<T>(backend());
        make(ref)->Transpose(transpose.get());
        expect_kept(*transpose, transposed(ref));

        if(!is(DIA) && !is(MCSR))
        {
            const Csr<T> rect = rectangular<T>(8, 12);
            auto         R    = make(rect);
            R->Transpose();
            expect_kept(*R, transposed(rect));
        }
    });
}

TEST_P(local_matrix_formats, compress_drops_small_entries)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = transformed(
            banded<T>(), [](int i, int j, T v) { return j == i + 1 ? static_cast<T>(0.25) : v; });

        auto A = make(ref);
        A->Compress(0.5);
        expect_kept(*A, filtered(ref, [](int, int, T v) { return std::abs(v) > 0.5; }));
    });
}

TEST_P(local_matrix_formats, gershgorin_bounds_the_spectrum)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = transformed(spd<T>(), [](int i, int j, T v) {
            return i == j ? static_cast<T>(i < 6 ? -5 : 7) : v;
        });

        auto        A          = make(ref);
        T           lambda_min = 0;
        T           lambda_max = 0;
        std::string log        = logged([&] { A->Gershgorin(lambda_min, lambda_max); });
        expect_fallback_warning(log, "Gershgorin", native("Gershgorin"));
        EXPECT_EQ(lambda_min, -7.0);
        EXPECT_EQ(lambda_max, 9.0);
    });
}

TEST_P(local_matrix_formats, symbolic_power_gives_the_pattern_with_zero_values)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();

        auto        A   = make(ref);
        std::string log = logged([&] { A->SymbolicPower(2); });
        expect_fallback_warning(log, "SymbolicPower", native("SymbolicPower"));
        EXPECT_EQ(rocalution::_matrix_format_names[A->GetFormat()], format_name(format()));
        EXPECT_EQ(backend_of(*A), backend_name(backend()));

        const Csr<T> result = csr_of(*A);
        for(T v : result.val)
        {
            EXPECT_EQ(v, 0.0);
        }
        if(is(CSR) || is(COO))
        {
            EXPECT_EQ(pattern_of(result), pattern_of(squared_pattern(ref)));
        }
    });
}

TEST_P(local_matrix_formats, fsai_gives_an_approximate_inverse_factor)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = spd<T>();

        // Row i of G solves A(J, J) g = c e_i on its pattern J, with (G A G^T)(i, i) = 1
        auto expect_factor = [&](const LocalMatrix<T>& G, const Csr<T>& pattern) {
            EXPECT_EQ(rocalution::_matrix_format_names[G.GetFormat()], format_name(format()));
            EXPECT_EQ(backend_of(G), backend_name(backend()));
            const Csr<T>         g   = csr_of(G);
            const std::vector<T> GA  = dense_product(dense(g), dense(ref), ref.n, ref.n, ref.n);
            const std::vector<T> GAG = dense_product(GA, dense(transposed(g)), ref.n, ref.n, ref.n);
            EXPECT_EQ(pattern_of(nonzeros(g)), pattern_of(lower(pattern, true)));
            const std::vector<std::set<int>> strict = pattern_of(lower(pattern, false));
            for(int i = 0; i < ref.n; ++i)
            {
                EXPECT_NEAR(GAG[static_cast<size_t>(i) * ref.n + i], 1.0, tol<T>())
                    << "at row " << i;
                for(int j : strict[i])
                {
                    EXPECT_NEAR(GA[static_cast<size_t>(i) * ref.n + j], 0.0, tol<T>())
                        << "at " << i << "," << j;
                }
            }
        };

        // Without a pattern matrix, the pattern of the stored entries, which for BCSR include the
        // zeros of the blocks
        const Csr<T> stored
            = transformed(csr_of(*make(ref)), [](int, int, T) { return static_cast<T>(1); });

        auto G = make(ref);
        G->FSAI(1, nullptr);
        expect_factor(*G, stored);

        G = make(ref);
        G->FSAI(2, nullptr);
        expect_factor(*G, transformed(squared_pattern(stored), [](int, int, T) {
            return static_cast<T>(1);
        }));

        G            = make(ref);
        auto pattern = make_local_matrix(backend(), ref);
        G->FSAI(1, pattern.get());
        expect_factor(*G, ref);
    });
}

TEST_P(local_matrix_formats, spai_minimizes_the_residual_on_the_pattern)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();

        auto M = make(ref);
        M->SPAI();
        EXPECT_EQ(rocalution::_matrix_format_names[M->GetFormat()], format_name(format()));
        EXPECT_EQ(backend_of(*M), backend_name(backend()));

        // Column i of M solves min ||A m - e_i|| on the pattern J of column i of A:
        // the residual is orthogonal to the columns J of A
        // The pattern is the one of the stored entries, which for BCSR include the zeros of the blocks
        const Csr<T>         stored = csr_of(*make(ref));
        const Csr<T>         m      = csr_of(*M);
        const Csr<T>         at     = transposed(stored);
        const std::vector<T> AM     = dense_product(dense(ref), dense(m), ref.n, ref.n, ref.n);
        const std::vector<T> A      = dense(ref);
        EXPECT_EQ(pattern_of(nonzeros(m)), pattern_of(stored));
        const std::vector<std::set<int>> columns = pattern_of(at);
        for(int i = 0; i < ref.n; ++i)
        {
            for(int j : columns[i])
            {
                T dot = 0;
                for(int k = 0; k < ref.n; ++k)
                {
                    const T r = AM[static_cast<size_t>(k) * ref.n + i] - (k == i ? 1 : 0);
                    dot += A[static_cast<size_t>(k) * ref.n + j] * r;
                }
                // Absolute, with terms A_kj r_k of |A_kj| up to 700, so tol<T>() is below one
                // rounding in float
                EXPECT_NEAR(dot, 0.0, scaled_tol<T>(1e-12))
                    << "column " << i << ", pattern column " << j;
            }
        }
    });
}

TEST_P(local_matrix_formats, key_depends_on_the_csr_entries)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();

        long int    row_key = 0;
        long int    col_key = 0;
        long int    val_key = 0;
        auto        A       = make(ref);
        std::string log     = logged([&] { A->Key(row_key, col_key, val_key); });
        expect_fallback_warning(log, "Key", native("Key"));

        long int row_ref = 0;
        long int col_ref = 0;
        long int val_ref = 0;
        make_local_matrix(Backend::Host, csr_of(*A))->Key(row_ref, col_ref, val_ref);
        EXPECT_EQ(row_key, row_ref);
        EXPECT_EQ(col_key, col_ref);
        EXPECT_EQ(val_key, val_ref);

        Csr<T> changed = ref;
        changed.val[5] += 1;
        long int row_changed = 0;
        long int col_changed = 0;
        long int val_changed = 0;
        make(changed)->Key(row_changed, col_changed, val_changed);
        EXPECT_EQ(row_changed, row_key);
        EXPECT_EQ(col_changed, col_key);
        EXPECT_NE(val_changed, val_key);
    });
}

TYPED_TEST(local_matrix, sort_orders_the_columns_of_each_row)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref = banded<T>();
    const Csr<T> mix = unsorted(ref);

    auto A = empty_local_matrix<T>(Backend::Host);
    A->AllocateCSR("A", mix.nnz(), mix.n, mix.m);
    A->CopyFromCSR(mix.row_offset.data(), mix.col.data(), mix.val.data());
    if(this->backend() == Backend::Accelerator)
    {
        A->MoveToAccelerator();
    }
    A->Sort();
    EXPECT_EQ(csr_of(*A).col, ref.col);
    expect_matrix(*A, ref, csr_format, this->backend());

    std::vector<int> row;
    for(int i = 0; i < mix.n; ++i)
    {
        row.insert(row.end(), mix.row_offset[i + 1] - mix.row_offset[i], i);
    }
    auto B = empty_local_matrix<T>(Backend::Host);
    B->AllocateCOO("B", mix.nnz(), mix.n, mix.m);
    B->CopyFromCOO(row.data(), mix.col.data(), mix.val.data());
    if(this->backend() == Backend::Accelerator)
    {
        B->MoveToAccelerator();
    }
    B->Sort();
    expect_matrix(*B, ref, Format{COO, 1}, this->backend());
}

// Files

TEST_P(local_matrix_formats, writes_and_reads_matrix_market_and_csr_files)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T> ref = banded<T>();

        auto read_mtx = [&] {
            const std::string mtx = temp_file("mtx");
            make(ref)->WriteFileMTX(mtx);
            auto B = empty<T>();
            B->ReadFileMTX(mtx);
            expect_kept(*B, ref);
            std::remove(mtx.c_str());
        };

        // Deprecated, but still part of the interface. The read clears the matrix first.
        auto read_csr = [&] {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            const std::string csr = temp_file("csr");
            make(ref)->WriteFileCSR(csr);
            auto C = make(spd<T>(4));
            C->ReadFileCSR(csr);
#pragma GCC diagnostic pop
            expect_kept(*C, ref);
            std::remove(csr.c_str());
        };

        // Both reads clear the matrix and then move it to the host, to read it there
        if(backend() == Backend::Accelerator && is(MCSR))
        {
            KNOWN_BUG("HIP MCSR CopyToHost copies the row offsets of an empty matrix from NULL");
            expect_in_child(read_mtx);
            expect_in_child(read_csr);
            return;
        }

        read_mtx();
        if(backend() == Backend::Accelerator && is(BCSR))
        {
            KNOWN_BUG("BCSR Clear keeps the block counts, so the cleared matrix copies them");
            expect_in_child(read_csr);
            return;
        }
        read_csr();
    });
}

TEST_P(local_matrix_formats, writes_and_reads_rocsparseio_files)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        const Csr<T>      ref  = banded<T>();
        const std::string file = temp_file("rsio");

        std::string log = logged([&] { make(ref)->WriteFileRSIO(file); });
        EXPECT_NE(log.find("WriteFileRSIO: filename=" + file + "; done"), std::string::npos) << log;

        // The file keeps the format, unless the matrix keeps its own
        auto B = make_local_matrix(backend(), spd<T>(4));
        log    = logged([&] { B->ReadFileRSIO(file); });
        EXPECT_EQ(backend_of(*B), backend_name(backend()));
        expect_entries(csr_of(*B), ref);

        auto C = make_local_matrix(backend(), spd<T>(4));
        C->ReadFileRSIO(file, true);
        expect_matrix(*C, ref, csr_format, backend());

        if(backend() == Backend::Accelerator && !is(CSR))
        {
            KNOWN_BUG("WriteFileRSIO of an accelerator matrix rewrites the file in CSR");
        }
        EXPECT_EQ(log.find("conversion from CSR to") != std::string::npos, !is(CSR)) << log;
        expect_kept(*B, ref);
        std::remove(file.c_str());
    });
}

// The AMG setup of a matrix in another format runs on a CSR copy, with a warning,
// and gives the results of that copy
TEST_P(local_matrix_formats, amg_setup_runs_on_a_csr_copy)
{
    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        auto expect_warning = [&](const std::string& log, const std::string& operation) {
            const bool warned
                = log.find("LocalMatrix::" + operation + "() is performed in CSR format")
                  != std::string::npos;
            EXPECT_EQ(warned, !is(CSR)) << operation << " logged:\n" << log;
        };
        auto like = [](const LocalMatrix<T>& A, auto& v) { v.CloneBackend(A); };

        // A CSR copy with the explicit zeros of BCSR blocks
        auto A = make(grid<T>(8, 6, 1.0, 0.1, true));
        auto C = make_local_matrix(backend(), csr_of(*A));

        using Aggregate = void (LocalMatrix<T>::*)(
            T, LocalVector<bool>*, LocalVector<int64_t>*, LocalVector<int64_t>*) const;
        const std::vector<std::pair<std::string, Aggregate>> aggregations
            = {{"AMGPMISAggregate", &LocalMatrix<T>::AMGPMISAggregate}};
        for(const auto& aggregation : aggregations)
        {
            SCOPED_TRACE(aggregation.first);
            LocalVector<bool>    connections, connections_ref;
            LocalVector<int64_t> aggregates, aggregates_ref, roots, roots_ref;
            like(*A, connections);
            like(*A, aggregates);
            like(*A, roots);
            like(*C, connections_ref);
            like(*C, aggregates_ref);
            like(*C, roots_ref);

            std::string log = logged(
                [&] { ((*A).*(aggregation.second))(T(0.08), &connections, &aggregates, &roots); });
            ((*C).*(aggregation.second))(T(0.08), &connections_ref, &aggregates_ref, &roots_ref);
            expect_warning(log, aggregation.first);
            EXPECT_EQ(values_of(connections), values_of(connections_ref));
            EXPECT_EQ(values_of(aggregates), values_of(aggregates_ref));
            EXPECT_EQ(values_of(roots), values_of(roots_ref));
        }

        LocalVector<bool>    connections;
        LocalVector<int64_t> aggregates, roots;
        like(*C, connections);
        like(*C, aggregates);
        like(*C, roots);
        C->AMGPMISAggregate(T(0.08), &connections, &aggregates, &roots);

        {
            SCOPED_TRACE("AMGUnsmoothedAggregation");
            LocalMatrix<T> P, P_ref;
            P.CloneBackend(*A);
            P_ref.CloneBackend(*C);
            std::string log = logged([&] { A->AMGUnsmoothedAggregation(aggregates, roots, &P); });
            C->AMGUnsmoothedAggregation(aggregates, roots, &P_ref);
            expect_warning(log, "AMGUnsmoothedAggregation");
            expect_near(csr_of(P), csr_of(P_ref), 0.0);
        }
        {
            SCOPED_TRACE("AMGSmoothedAggregation");
            LocalMatrix<T> P, P_ref;
            P.CloneBackend(*A);
            P_ref.CloneBackend(*C);
            std::string log = logged([&] {
                A->AMGSmoothedAggregation(T(2.0 / 3.0), connections, aggregates, roots, &P);
            });
            C->AMGSmoothedAggregation(T(2.0 / 3.0), connections, aggregates, roots, &P_ref);
            expect_warning(log, "AMGSmoothedAggregation");
            expect_near(csr_of(P), csr_of(P_ref), scaled_tol<T>(1e-14));
        }

        auto R  = make(grid<T>(8, 6, 1.0, 0.1, false));
        auto RC = make_local_matrix(backend(), csr_of(*R));

        LocalVector<int>  cf;
        LocalVector<bool> S;
        like(*RC, cf);
        like(*RC, S);
        RC->RSPMISCoarsening(0.25f, &cf, &S);

        {
            SCOPED_TRACE("RSPMISCoarsening");
            LocalVector<int>  cf_other;
            LocalVector<bool> S_other;
            like(*R, cf_other);
            like(*R, S_other);
            std::string log = logged([&] { R->RSPMISCoarsening(0.25f, &cf_other, &S_other); });
            expect_warning(log, "RSPMISCoarsening");
            EXPECT_EQ(values_of(cf_other), values_of(cf));
            EXPECT_EQ(values_of(S_other), values_of(S));
        }

        using Interpolation = std::function<void(const LocalMatrix<T>&, LocalMatrix<T>*)>;
        const std::vector<std::pair<std::string, Interpolation>> interpolations
            = {{"RSDirectInterpolation",
                [&](const LocalMatrix<T>& M, LocalMatrix<T>* P) {
                    M.RSDirectInterpolation(cf, S, P);
                }},
               {"RSExtPIInterpolation",
                [&](const LocalMatrix<T>& M, LocalMatrix<T>* P) {
                    M.RSExtPIInterpolation(cf, S, false, P);
                }},
               {"RSMMExtPIInterpolation",
                [&](const LocalMatrix<T>& M, LocalMatrix<T>* P) {
                    M.RSMMExtPIInterpolation(cf, S, P);
                }},
               {"RSMMExtPEInterpolation", [&](const LocalMatrix<T>& M, LocalMatrix<T>* P) {
                    M.RSMMExtPEInterpolation(cf, S, P);
                }}};
        for(const auto& interpolation : interpolations)
        {
            SCOPED_TRACE(interpolation.first);
            LocalMatrix<T> P, P_ref;
            P.CloneBackend(*R);
            P_ref.CloneBackend(*RC);
            std::string log = logged([&] { interpolation.second(*R, &P); });
            interpolation.second(*RC, &P_ref);
            expect_warning(log, interpolation.first);
            expect_near(csr_of(P), csr_of(P_ref), scaled_tol<T>(1e-14));
        }
    });
}

// The greedy aggregation runs on the host, on a CSR copy
TEST_P(local_matrix_formats, amg_greedy_aggregate_runs_on_a_csr_copy)
{
    auto checks = [&] {
        for_each_type<RealTypes>([&](auto type) {
            using T = typename decltype(type)::type;

            auto A = make(grid<T>(8, 6, 1.0, 0.1, true));
            auto C = make_local_matrix(backend(), csr_of(*A));

            LocalVector<bool>    connections, connections_ref;
            LocalVector<int64_t> aggregates, aggregates_ref, roots, roots_ref;
            connections.CloneBackend(*A);
            aggregates.CloneBackend(*A);
            roots.CloneBackend(*A);
            connections_ref.CloneBackend(*C);
            aggregates_ref.CloneBackend(*C);
            roots_ref.CloneBackend(*C);

            std::string log = logged(
                [&] { A->AMGGreedyAggregate(T(0.08), &connections, &aggregates, &roots); });
            C->AMGGreedyAggregate(T(0.08), &connections_ref, &aggregates_ref, &roots_ref);
            EXPECT_EQ(log.find("LocalMatrix::AMGGreedyAggregate() is performed in CSR format")
                          != std::string::npos,
                      !is(CSR))
                << log;
            EXPECT_EQ(values_of(connections), values_of(connections_ref));
            EXPECT_EQ(values_of(aggregates), values_of(aggregates_ref));
            EXPECT_EQ(values_of(roots), values_of(roots_ref));
        });
    };

    if(backend() == Backend::Accelerator && !is(CSR))
    {
        KNOWN_BUG("the host fallback copies the matrix in its own format into a CSR host matrix");
        expect_in_child(checks);
    }
    else
    {
        checks();
    }
}

// Matrices without entries

// The rows and columns of a matrix without entries are zero
TYPED_TEST(local_matrix, rows_and_columns_of_a_matrix_without_entries)
{
    using T = typename TypeParam::value_type;

    const int                  n = 4;
    const std::vector<PtrType> no_entries(n + 1, 0);
    const std::vector<T>       values = {1, 0, 2, 3};
    const std::vector<T>       zeros(n, T(0));

    KNOWN_BUG("rows and columns of a matrix without entries are neither extracted nor replaced");

    auto A = empty_local_matrix<T>(this->backend());
    A->CopyFromHostCSR(no_entries.data(), nullptr, nullptr, "E", 0, n, n);
    auto x = make_local_vector(this->backend(), std::vector<T>(n, T(7)));
    A->ExtractColumnVector(1, x.get());
    expect_vector(*x, zeros, this->backend(), 0);

    x = make_local_vector(this->backend(), std::vector<T>(n, T(7)));
    A->ExtractRowVector(1, x.get());
    expect_vector(*x, zeros, this->backend(), 0);

    std::vector<T> D(n * n, T(0));
    for(int i = 0; i < n; ++i)
    {
        D[i * n + 1] = values[i];
    }
    A->ReplaceColumnVector(1, *make_local_vector(this->backend(), values));
    expect_matrix(*A, from_dense(n, n, D), csr_format, this->backend());

    auto B = empty_local_matrix<T>(this->backend());
    B->CopyFromHostCSR(no_entries.data(), nullptr, nullptr, "E", 0, n, n);
    std::fill(D.begin(), D.end(), T(0));
    std::copy(values.begin(), values.end(), D.begin() + 2 * n);
    B->ReplaceRowVector(2, *make_local_vector(this->backend(), values));
    expect_matrix(*B, from_dense(n, n, D), csr_format, this->backend());
}

// The operations skip a matrix without entries and leave their outputs as they
// were. For the solves of this singular system, the eigenvalue bounds, the
// Householder vector, the orderings and the iteration count that is a convention
// of the library, not a result that follows from the input; the test pins it.
TYPED_TEST(local_matrix, operations_on_a_matrix_without_entries_change_nothing)
{
    using T = typename TypeParam::value_type;

    const int                  n = 4;
    const std::vector<PtrType> no_entries(n + 1, 0);
    const std::vector<T>       sentinel(n, 7.0);
    const std::vector<int>     identity = {0, 1, 2, 3};

    auto A = empty_local_matrix<T>(this->backend());
    A->CopyFromHostCSR(no_entries.data(), nullptr, nullptr, "E", 0, n, n);
    auto x    = make_local_vector(this->backend(), sentinel);
    auto b    = make_local_vector(this->backend(), std::vector<T>(n, 1.0));
    auto perm = make_local_vector(this->backend(), identity);

    A->Zeros();
    A->Scale(2.0);
    A->ScaleDiagonal(2.0);
    A->ScaleOffDiagonal(2.0);
    A->AddScalar(2.0);
    A->AddScalarDiagonal(2.0);
    A->AddScalarOffDiagonal(2.0);
    A->ILU0Factorize();
    A->ILUTFactorize(0.1, 4);
    A->ILUpFactorize(1);
    A->LUFactorize();
    A->QRDecompose();
    A->Invert();
    A->SymbolicPower(2);
    A->Compress(0.1);
    A->Transpose();
    A->Sort();
    A->FSAI(1, nullptr);
    A->SPAI();
    A->Permute(*perm);
    A->PermuteBackward(*perm);
    A->DiagonalMatrixMultL(*b);
    A->DiagonalMatrixMultR(*b);

    int niter = -1;
    A->ItILU0Factorize(rocalution::Default, 0, 10, 1e-8, &niter, nullptr);
    EXPECT_EQ(niter, -1);

    auto inv = make_local_vector(this->backend(), std::vector<T>());
    A->ICFactorize(inv.get());
    EXPECT_EQ(inv->GetSize(), 0);

    A->LUAnalyse();
    A->LLAnalyse();
    A->LAnalyse(false);
    A->UAnalyse(false);
    A->ItLUAnalyse();
    A->ItLLAnalyse();
    A->ItLAnalyse(false);
    A->ItUAnalyse(false);
    A->LUSolve(*b, x.get());
    A->LLSolve(*b, x.get());
    A->LLSolve(*b, *b, x.get());
    A->LSolve(*b, x.get());
    A->USolve(*b, x.get());
    A->QRSolve(*b, x.get());
    A->ItLUSolve(10, 0, false, *b, x.get());
    A->ItLLSolve(10, 0, false, *b, x.get());
    A->ItLLSolve(10, 0, false, *b, *b, x.get());
    A->ItLSolve(10, 0, false, *b, x.get());
    A->ItUSolve(10, 0, false, *b, x.get());
    A->LUAnalyseClear();
    A->LLAnalyseClear();
    A->LAnalyseClear();
    A->UAnalyseClear();
    A->ItLUAnalyseClear();
    A->ItLLAnalyseClear();
    A->ItLAnalyseClear();
    A->ItUAnalyseClear();
    expect_vector(*x, sentinel, this->backend(), 0);

    T beta = 5;
    A->Householder(0, beta, x.get());
    EXPECT_EQ(beta, 5.0);

    T lambda_min = 3;
    T lambda_max = 3;
    A->Gershgorin(lambda_min, lambda_max);
    EXPECT_EQ(lambda_min, 3.0);
    EXPECT_EQ(lambda_max, 3.0);

    long int keys[3] = {1, 2, 3};
    A->Key(keys[0], keys[1], keys[2]);
    EXPECT_EQ(keys[0], 1);
    EXPECT_EQ(keys[1], 2);
    EXPECT_EQ(keys[2], 3);

    auto d = make_local_vector(this->backend(), std::vector<T>());
    A->ExtractDiagonal(d.get());
    A->ExtractInverseDiagonal(d.get());
    EXPECT_EQ(d->GetSize(), 0);

    auto L = empty_local_matrix<T>(this->backend());
    auto U = empty_local_matrix<T>(this->backend());
    A->ExtractL(L.get(), true);
    A->ExtractU(U.get(), false);
    EXPECT_EQ(L->GetNnz(), 0);
    EXPECT_EQ(U->GetNnz(), 0);

    auto p          = make_local_vector(this->backend(), std::vector<int>());
    int  size       = -1;
    int  num_colors = -1;
    int* sizes      = nullptr;
    A->CMK(p.get());
    A->RCMK(p.get());
    A->ConnectivityOrder(p.get());
    A->MultiColoring(num_colors, &sizes, p.get());
    A->MaximalIndependentSet(size, p.get());
    A->ZeroBlockPermutation(size, p.get());
    EXPECT_EQ(p->GetSize(), 0);
    EXPECT_EQ(size, -1);
    EXPECT_EQ(num_colors, -1);
    EXPECT_EQ(sizes, nullptr);

    const int        offsets[2] = {0, n};
    LocalMatrix<T>   block;
    LocalMatrix<T>*  row[1]    = {&block};
    LocalMatrix<T>** blocks[1] = {row};
    A->ExtractSubMatrices(1, 1, offsets, offsets, blocks);
    EXPECT_EQ(block.GetNnz(), 0);

    auto transpose = empty_local_matrix<T>(this->backend());
    A->Transpose(transpose.get());
    EXPECT_EQ(transpose->GetNnz(), 0);

    std::vector<PtrType> row_offset(n + 1, 0);
    A->CopyFromCSR(row_offset.data(), nullptr, nullptr);
    A->CopyToCSR(row_offset.data(), nullptr, nullptr);

    auto C     = empty_local_matrix<T>(this->backend());
    int  index = 0;
    T    value = 0;
    C->ConvertToCOO();
    C->CopyFromCOO(&index, &index, &value);
    C->CopyToCOO(&index, &index, &value);
    EXPECT_EQ(C->GetNnz(), 0);
    EXPECT_EQ(rocalution::_matrix_format_names[C->GetFormat()], std::string("COO"));

    EXPECT_EQ(A->GetNnz(), 0);
    EXPECT_EQ(A->GetM(), n);
    EXPECT_EQ(A->GetN(), n);
    EXPECT_EQ(rocalution::_matrix_format_names[A->GetFormat()], format_name(csr_format));
    EXPECT_EQ(backend_of(*A), backend_name(this->backend()));
}

TYPED_TEST(local_matrix, amg_setup_of_an_empty_matrix_gives_empty_results)
{
    using T = typename TypeParam::value_type;

    auto A = empty_local_matrix<T>(this->backend());

    LocalVector<bool>    connections;
    LocalVector<int64_t> aggregates, roots;
    connections.CloneBackend(*A);
    aggregates.CloneBackend(*A);
    roots.CloneBackend(*A);
    A->AMGGreedyAggregate(T(0.08), &connections, &aggregates, &roots);
    A->AMGPMISAggregate(T(0.08), &connections, &aggregates, &roots);
    EXPECT_EQ(connections.GetSize(), 0);
    EXPECT_EQ(aggregates.GetSize(), 0);
    EXPECT_EQ(roots.GetSize(), 0);

    LocalMatrix<T> P;
    P.CloneBackend(*A);
    A->AMGUnsmoothedAggregation(aggregates, roots, &P);
    EXPECT_EQ(P.GetNnz(), 0);

    LocalVector<int>  cf;
    LocalVector<bool> S;
    cf.CloneBackend(*A);
    S.CloneBackend(*A);
    A->RSCoarsening(0.25f, &cf, &S);
    A->RSPMISCoarsening(0.25f, &cf, &S);
    EXPECT_EQ(cf.GetSize(), 0);
    EXPECT_EQ(S.GetSize(), 0);

    LocalVector<int> G;
    G.CloneBackend(*A);
    int  nc     = 0;
    int  Gsize  = 0;
    int  rGsize = 0;
    int* rG     = nullptr;
    A->InitialPairwiseAggregation(T(0.25), nc, &G, Gsize, &rG, rGsize, 0);
    EXPECT_EQ(G.GetSize(), 0);
    EXPECT_EQ(rG, nullptr);
    EXPECT_EQ(backend_of(*A), backend_name(this->backend()));
}

TYPED_TEST(local_matrix, set_data_ptr_csr_without_entries_keeps_the_sizes)
{
    using T = typename TypeParam::value_type;

    const int n          = 4;
    PtrType*  row_offset = nullptr;
    int*      col        = nullptr;
    T*        val        = nullptr;
    rocalution::allocate_host(n + 1, &row_offset);
    std::fill(row_offset, row_offset + n + 1, 0);

    // The arrays are host arrays, so the matrix moves after taking them
    auto A = empty_local_matrix<T>(Backend::Host);
    A->SetDataPtrCSR(&row_offset, &col, &val, "E", 0, n, n);
    EXPECT_EQ(row_offset, nullptr);
    if(this->backend() == Backend::Accelerator)
    {
        A->MoveToAccelerator();
    }
    EXPECT_EQ(A->GetM(), n);
    EXPECT_EQ(A->GetN(), n);
    EXPECT_EQ(A->GetNnz(), 0);
    EXPECT_EQ(backend_of(*A), backend_name(this->backend()));

    auto x = make_local_vector(this->backend(), std::vector<T>(n, 7.0));
    A->Apply(*make_local_vector(this->backend(), std::vector<T>(n, 1.0)), x.get());
    expect_vector(*x, std::vector<T>(n, 0.0), this->backend(), 0);
}

// The value array of DIA holds num_diag columns of max(nrow, ncol) values, (row, d) at d * nrow + row
TYPED_TEST(local_matrix, set_data_ptr_dia_of_a_wide_matrix)
{
    using T = typename TypeParam::value_type;

    const int nrow   = 4;
    const int ncol   = 6;
    const int ndiag  = 2;
    int*      offset = nullptr;
    T*        val    = nullptr;
    rocalution::allocate_host(ndiag, &offset);
    rocalution::allocate_host(ncol * ndiag, &val);
    std::fill(val, val + ncol * ndiag, 0.0);
    offset[0] = 0;
    offset[1] = 2;

    Csr<T> ref(nrow, ncol);
    for(int i = 0; i < nrow; ++i)
    {
        for(int d = 0; d < ndiag; ++d)
        {
            const T v = static_cast<T>(10 * i + d + 1);
            ref.add(i + offset[d], v);
            val[d * nrow + i] = v;
        }
        ref.end_row();
    }

    auto A = empty_local_matrix<T>(Backend::Host);
    A->SetDataPtrDIA(&offset, &val, "wide", ncol * ndiag, nrow, ncol, ndiag);
    EXPECT_EQ(offset, nullptr);
    EXPECT_EQ(val, nullptr);
    if(this->backend() == Backend::Accelerator)
    {
        A->MoveToAccelerator();
    }
    expect_matrix(*A, ref, Format{DIA, 1}, this->backend());

    const std::vector<T> x = {1, 2, 3, 4, 5, 6};
    auto                 y = make_local_vector(this->backend(), std::vector<T>(nrow, 0.0));
    A->Apply(*make_local_vector(this->backend(), x), y.get());
    if(this->backend() == Backend::Host)
    {
        KNOWN_BUG("host DIA Apply drops the entries in the columns from nrow on");
    }
    expect_vector(*y, ref.apply(x), this->backend(), 0);
}

// Preconditions

TEST(local_matrix_preconditions, violations_end_the_program)
{
    const Csr<double> ref = banded(4);

    LocalMatrix<double> E;
    EXPECT_ASSERT_DEATH(E.AllocateCSR("A", -1, 4, 4), "nnz >= 0");
    EXPECT_ASSERT_DEATH(E.AllocateCSR("A", 4, 0, 4), "nrow > 0");
    EXPECT_ASSERT_DEATH(E.AllocateBCSR("A", 1, 1, 1, 1), "blockdim > 1");
    EXPECT_ASSERT_DEATH(
        {
            int*    row = nullptr;
            int*    col = nullptr;
            double* val = nullptr;
            E.LeaveDataPtrCOO(&row, &col, &val);
        },
        "GetM");

    auto A = make_local_matrix(Backend::Host, ref);
    auto x = make_local_vector(Backend::Host, std::vector<double>(3, 1.0));
    auto y = make_local_vector(Backend::Host, std::vector<double>(4, 1.0));
    EXPECT_ASSERT_DEATH(A->Apply(*x, y.get()), "in.GetSize");
    EXPECT_ASSERT_DEATH(A->CopyFromCOO(ref.col.data(), ref.col.data(), ref.val.data()),
                        "GetFormat");
    EXPECT_ASSERT_DEATH(make_local_matrix(Backend::Host, ref, Format{COO, 1})
                            ->UpdateValuesCSR(const_cast<double*>(ref.val.data())),
                        "GetFormat");
    EXPECT_ASSERT_DEATH(A->MatrixAdd(*make_local_matrix(Backend::Host, ref, Format{COO, 1})),
                        "GetFormat");
    EXPECT_ASSERT_DEATH(A->ILUpFactorize(-1), "p >= 0");
    EXPECT_ASSERT_DEATH(A->FSAI(0, nullptr), "power > 0");
    EXPECT_ASSERT_DEATH(A->ItLSolve(0, 0, false, *y, y.get()), "max_iter > 0");

    LocalMatrix<double> S;
    EXPECT_ASSERT_DEATH(A->ExtractSubMatrix(5, 0, 1, 1, &S), "row_offset");
    EXPECT_ASSERT_DEATH(S.MatrixMult(*A, *make_local_matrix(Backend::Host, rectangular(3, 4))),
                        "B.GetM");
}

// LocalStencil, the other local Operator, keeps the async moves and Sync of
// BaseRocalution

TEST(local_stencil, sync_keeps_the_stencil_and_it_has_no_ghost_part)
{
    // On the 2 x 2 grid every point has two neighbours: y_i = 4 x_i - x_j - x_k
    rocalution::LocalStencil<double> S(rocalution::Laplace2D);
    S.SetGrid(2);
    S.Sync();

    const rocalution::Operator<double>& op = S;
    EXPECT_EQ(op.GetLocalM(), 4);
    EXPECT_EQ(op.GetLocalN(), 4);
    EXPECT_EQ(op.GetGhostM(), 0);
    EXPECT_EQ(op.GetGhostN(), 0);
    EXPECT_EQ(op.GetGhostNnz(), 0);

    auto x = make_local_vector(Backend::Host, std::vector<double>{1, 2, 3, 4});
    auto y = make_local_vector(Backend::Host, std::vector<double>(4, 0.0));
    S.Apply(*x, y.get());
    EXPECT_EQ(values_of(*y), (std::vector<double>{-1, 3, 7, 11}));
}

// The default async moves call MoveToAccelerator and MoveToHost, which
// LocalStencil does not implement
TEST(local_stencil, async_moves_end_the_program_as_the_moves_do)
{
    rocalution::LocalStencil<double> S(rocalution::Laplace2D);
    S.SetGrid(2);
    EXPECT_FATAL_ERROR(S.MoveToAcceleratorAsync(), "local_stencil\\.cpp");
    EXPECT_FATAL_ERROR(S.MoveToHostAsync(), "local_stencil\\.cpp");
}

INSTANTIATE_TEST_SUITE_P(formats,
                         local_matrix_formats,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          all_formats),
                         backend_format_name);

INSTANTIATE_TEST_SUITE_P(formats,
                         local_matrix_data_ptr,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{CSR, 1},
                                                          Format{MCSR, 1},
                                                          Format{BCSR, 2},
                                                          Format{COO, 1},
                                                          Format{ELL, 1},
                                                          Format{DIA, 1},
                                                          Format{DENSE, 1})),
                         backend_format_name);

// The typed rocSPARSE wrappers of the other precisions, reached through
// LocalMatrix on the accelerator. Complex needs SUPPORT_COMPLEX.

namespace
{
    template <typename T>
    class local_matrix_precision : public testing::Test
    {
    protected:
        void SetUp() override
        {
            REQUIRE_BACKEND(Backend::Accelerator);
        }
    };

    struct PrecisionName
    {
        template <typename T>
        static std::string GetName(int)
        {
            if(std::is_same<T, float>::value)
            {
                return "float";
            }
            return std::is_same<T, std::complex<float>>::value ? "complex_float" : "complex_double";
        }
    };

    template <typename T>
    class local_matrix_host_precision : public testing::Test
    {
    };

#ifdef SUPPORT_COMPLEX
    using Precisions = testing::Types<float, std::complex<float>, std::complex<double>>;
#else
    using Precisions = testing::Types<float>;
#endif

    template <typename T>
    T with_imaginary_part(double re, double im)
    {
        if constexpr(is_complex_v<T>)
        {
            return T(static_cast<real_t<T>>(re), static_cast<real_t<T>>(im));
        }
        else
        {
            return static_cast<T>(re);
        }
    }

    // For complex T, imaginary parts of both signs, so that a missing or an extra
    // conjugation changes the results. hermitian keeps a symmetric A Hermitian, with
    // a real diagonal and conjugate entries at (i, j) and (j, i); the parts stay
    // small enough to keep spd() diagonally dominant.
    template <typename T>
    Csr<T> complex_valued(const Csr<double>& A, bool hermitian = false)
    {
        Csr<T> B(A.n, A.m);
        B.row_offset = A.row_offset;
        B.col        = A.col;
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType k = A.row_offset[i]; k < A.row_offset[i + 1]; ++k)
            {
                const int    j    = A.col[k];
                const double sign = j > i ? 1 : j < i ? -1 : 0;
                const double im
                    = hermitian ? sign * 0.25 * (1 + (i + j) % 3) : 0.5 * ((i + 2 * j) % 4) - 0.75;
                B.val.push_back(with_imaginary_part<T>(A.val[k], im));
            }
        }
        return B;
    }

    template <typename T>
    std::vector<T> complex_valued(const std::vector<double>& v)
    {
        std::vector<T> w;
        for(size_t k = 0; k < v.size(); ++k)
        {
            w.push_back(with_imaginary_part<T>(v[k], 0.5 * ((3 * k) % 4) - 0.75));
        }
        return w;
    }
}

TYPED_TEST_SUITE(local_matrix_precision, Precisions, PrecisionName);

TYPED_TEST(local_matrix_precision, applies_and_converts_on_the_accelerator)
{
    using T                  = TypeParam;
    const Backend        acc = Backend::Accelerator;
    const std::vector<T> x   = complex_valued<T>(rhs(12));

    for(const Format f : {Format{CSR, 1},
                          Format{COO, 1},
                          Format{ELL, 1},
                          Format{HYB, 1},
                          Format{DIA, 1},
                          Format{BCSR, 2},
                          Format{DENSE, 1}})
    {
        SCOPED_TRACE(format_name(f));
        const Csr<T> ref = complex_valued<T>(f.id == HYB ? arrow<double>(12) : banded());

        auto apply_and_convert_to_csr = [&](LocalMatrix<T>& A) {
            auto vx = make_local_vector(acc, x);
            auto vy = make_local_vector(acc, std::vector<T>(12));
            A.Apply(*vx, vy.get());
            expect_close(values_of(*vy), ref.apply(x), scaled_tol<T>(1e-14));

            A.ConvertToCSR();
            expect_matrix(A, ref, csr_format, acc);
        };

        auto A = make_local_matrix(acc, ref);
        A->ConvertTo(f.id, f.blockdim);
        expect_matrix(*A, ref, f, acc);

        if(f.id == DENSE && is_complex_v<T>)
        {
            KNOWN_BUG("the complex rocblasTgemv of HIP DENSE Apply is a FATAL_ERROR stub");
            expect_in_child([&] {
                auto M = make_local_matrix(acc, ref, f);
                apply_and_convert_to_csr(*M);
            });
            continue;
        }
        apply_and_convert_to_csr(*A);
    }
}

TYPED_TEST(local_matrix_precision, factorizes_and_solves_on_the_accelerator)
{
    using T                         = TypeParam;
    const Backend        acc        = Backend::Accelerator;
    const Csr<T>         ref        = complex_valued<T>(banded());
    const Csr<T>         s          = complex_valued<T>(spd(), true);
    const std::vector<T> b          = complex_valued<T>(rhs(12));
    const double         direct_tol = loose_tol<T>(1e-10);

    auto solution = [&]() { return make_local_vector(acc, std::vector<T>(12)); };

    // Checked last, so that their known bug skips only their checks
    struct Solution
    {
        std::string    solve;
        std::vector<T> x;
        double         tolerance;
    };
    std::vector<Solution> cholesky_solutions;

    for(const Format f : {Format{CSR, 1}, Format{BCSR, 2}})
    {
        SCOPED_TRACE(format_name(f));
        auto A = make_local_matrix(acc, ref, f);
        A->ILU0Factorize();
        A->LUAnalyse();
        auto x = solution();
        A->LUSolve(*make_local_vector(acc, b), x.get());
        expect_close(ref.apply(values_of(*x)), b, direct_tol);
        A->LUAnalyseClear();

        auto T_ = make_local_matrix(acc, ref, f);
        T_->LAnalyse(false);
        T_->UAnalyse(false);
        auto y = solution();
        T_->LSolve(*make_local_vector(acc, b), y.get());
        expect_close(lower(ref, true).apply(values_of(*y)), b, direct_tol);
        T_->USolve(*make_local_vector(acc, b), y.get());
        expect_close(upper(ref, true).apply(values_of(*y)), b, direct_tol);
        T_->LAnalyseClear();
        T_->UAnalyseClear();

        auto S   = make_local_matrix(acc, lower(s, true), f);
        auto inv = make_local_vector(acc, std::vector<T>());
        S->ICFactorize(inv.get());
        S->LLAnalyse();
        auto z = solution();
        S->LLSolve(*make_local_vector(acc, b), z.get());
        cholesky_solutions.push_back({"LLSolve in " + format_name(f), values_of(*z), direct_tol});
        S->LLAnalyseClear();
    }

    auto LU = make_local_matrix(acc, ref);
    LU->ILU0Factorize();
    LU->ItLUAnalyse();
    auto x = solution();
    LU->ItLUSolve(24, 0, false, *make_local_vector(acc, b), x.get());
    expect_close(ref.apply(values_of(*x)), b, tol<T>());
    LU->ItLUAnalyseClear();

    auto Tri = make_local_matrix(acc, ref);
    Tri->ItLAnalyse(false);
    Tri->ItLSolve(24, 0, false, *make_local_vector(acc, b), x.get());
    expect_close(lower(ref, true).apply(values_of(*x)), b, tol<T>());
    Tri->ItLAnalyseClear();
    Tri->ItUAnalyse(false);
    Tri->ItUSolve(24, 0, false, *make_local_vector(acc, b), x.get());
    expect_close(upper(ref, true).apply(values_of(*x)), b, tol<T>());
    Tri->ItUAnalyseClear();

    auto S   = make_local_matrix(acc, lower(s, true));
    auto inv = make_local_vector(acc, std::vector<T>());
    S->ICFactorize(inv.get());
    S->ItLLAnalyse();
    S->ItLLSolve(24, 0, false, *make_local_vector(acc, b), x.get());
    cholesky_solutions.push_back({"ItLLSolve", values_of(*x), tol<T>()});
    S->ItLLAnalyseClear();

    int  niter = 0;
    auto It    = make_local_matrix(acc, ref);
    It->ItILU0Factorize(rocalution::Default, 0, 200, scaled_tol<T>(1e-14), &niter, nullptr);
    It->LUAnalyse();
    It->LUSolve(*make_local_vector(acc, b), x.get());
    expect_close(ref.apply(values_of(*x)), b, loose_tol<T>(1e-10));
    It->LUAnalyseClear();

    if(is_complex_v<T>)
    {
        KNOWN_BUG("the HIP LLSolve and ItLLSolve solve L L^T x = b, but the rocSPARSE IC(0) "
                  "factor L is the one of L L^H");
    }
    for(const Solution& computed : cholesky_solutions)
    {
        SCOPED_TRACE(computed.solve);
        expect_close(s.apply(computed.x), b, computed.tolerance);
    }
}

TYPED_TEST_SUITE(local_matrix_host_precision, Precisions, PrecisionName);

// IC(0) of a Hermitian positive definite matrix is the lower factor L of L L^H
TYPED_TEST(local_matrix_host_precision, incomplete_cholesky_factorizes_and_solves_on_the_host)
{
    using T                 = TypeParam;
    const Backend        h  = Backend::Host;
    const Csr<T>         s  = complex_valued<T>(spd(), true);
    const Csr<T>         sl = lower(s, true);
    const int            n  = s.n;
    const std::vector<T> b  = complex_valued<T>(rhs(12));

    // IC(0) of the tridiagonal s has no fill, so it is the Cholesky factor, and the
    // dense solution of s x = b solves with L and then with L^H
    const std::vector<T> L = ic0_dense(s);
    std::vector<T>       LH(L.size());
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            LH[i * n + j] = conj_entry(L[j * n + i]);
        }
    }
    const std::vector<T> solution = upper_solve(LH, n, lower_solve(L, n, b, false), false);
    expect_close(s.apply(solution), b, loose_tol<T>(1e-10));

    auto S   = make_local_matrix(h, sl);
    auto inv = make_local_vector(h, std::vector<T>());
    S->ICFactorize(inv.get());
    EXPECT_EQ(rocalution::_matrix_format_names[S->GetFormat()], format_name(csr_format));
    EXPECT_EQ(backend_of(*S), backend_name(h));

    const Csr<T> factor = csr_of(*S);
    EXPECT_EQ(factor.row_offset, sl.row_offset);
    EXPECT_EQ(factor.col, sl.col);
    const std::vector<T> F        = dense(factor);
    const std::vector<T> inv_diag = values_of(*inv);
    ASSERT_EQ(inv_diag.size(), static_cast<size_t>(n));
    std::vector<T> diag_times_inverse, first_column, first_column_ref;
    for(int i = 0; i < n; ++i)
    {
        diag_times_inverse.push_back(F[i * n + i] * inv_diag[i]);
        first_column.push_back(F[i * n]);
        first_column_ref.push_back(L[i * n]);
    }
    expect_close(diag_times_inverse, std::vector<T>(n, 1.0), tol<T>());
    // The first column does not depend on the conjugation
    expect_close(first_column, first_column_ref, loose_tol<T>(1e-10));

    auto x = make_local_vector(h, std::vector<T>(n));
    auto y = make_local_vector(h, std::vector<T>(n));
    S->LLAnalyse();
    S->LLSolve(*make_local_vector(h, b), x.get());
    S->LLSolve(*make_local_vector(h, b), *inv, y.get());
    S->LLAnalyseClear();

    auto z = make_local_vector(h, std::vector<T>(n));
    S->ItLLAnalyse();
    S->ItLLSolve(24, 0, false, *make_local_vector(h, b), z.get());
    S->ItLLAnalyseClear();

    if(is_complex_v<T>)
    {
        KNOWN_BUG("the host ICFactorize computes the complex symmetric factor, with the "
                  "square roots of magnitudes on the diagonal, and the host LLSolve and "
                  "ItLLSolve solve with L L^T");
    }
    {
        SCOPED_TRACE("factor");
        expect_close(F, L, loose_tol<T>(1e-10));
    }
    {
        SCOPED_TRACE("LLSolve");
        expect_close(values_of(*x), solution, loose_tol<T>(1e-10));
    }
    {
        SCOPED_TRACE("LLSolve with the inverse diagonal");
        expect_close(values_of(*y), solution, loose_tol<T>(1e-10));
    }
    {
        SCOPED_TRACE("ItLLSolve");
        expect_close(values_of(*z), solution, loose_tol<T>(1e-10));
    }
}

TYPED_TEST(local_matrix_precision, multiplies_adds_transposes_and_sorts_on_the_accelerator)
{
    using T           = TypeParam;
    const Backend acc = Backend::Accelerator;
    const Csr<T>  a   = complex_valued<T>(banded());
    const Csr<T>  c   = complex_valued<T>(two_diagonals());

    auto A = make_local_matrix(acc, a);
    auto B = make_local_matrix(acc, c);
    auto C = empty_local_matrix<T>(acc);
    C->MatrixMult(*A, *B);
    expect_matrix(*C,
                  from_dense(12, 12, dense_product(dense(a), dense(c), 12, 12, 12)),
                  csr_format,
                  acc,
                  scaled_tol<T>(1e-14));

    std::vector<T> sum = dense(a);
    std::vector<T> dc  = dense(c);
    for(size_t k = 0; k < sum.size(); ++k)
    {
        sum[k] = static_cast<T>(2) * sum[k] - dc[k];
    }
    A->MatrixAdd(*B, static_cast<T>(2), static_cast<T>(-1), true);
    expect_matrix(*A, from_dense(12, 12, sum), csr_format, acc, scaled_tol<T>(1e-14));

    const Csr<T> r = complex_valued<T>(rectangular(8, 12));
    auto         R = make_local_matrix(acc, r);
    R->Transpose();
    expect_matrix(*R, transposed(r), csr_format, acc);

    const Csr<T> mix = unsorted(a);
    auto         U   = empty_local_matrix<T>(Backend::Host);
    U->AllocateCSR("U", mix.nnz(), mix.n, mix.m);
    U->CopyFromCSR(mix.row_offset.data(), mix.col.data(), mix.val.data());
    U->MoveToAccelerator();
    U->Sort();
    EXPECT_EQ(csr_of(*U).col, a.col);
    expect_matrix(*U, a, csr_format, acc);
}
