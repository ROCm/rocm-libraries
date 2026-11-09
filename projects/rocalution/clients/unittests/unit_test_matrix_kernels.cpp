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

// Kernel selection branches and raw-array interfaces of the matrix formats
// other than CSR, conversions between the accelerator formats, and the host
// kernels outside the matrix classes: the iterative triangular solve and the
// Laplace stencil

#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <tuple>

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/host/host_matrix_coo.hpp"
#include "base/host/host_matrix_dia.hpp"
#include "base/host/host_matrix_ell.hpp"
#include "base/host/host_matrix_hyb.hpp"
#include "base/host/host_sparse.hpp"
#include "base/host/host_stencil_laplace2d.hpp"
#include "base/local_stencil.hpp"
#include "base/local_vector.hpp"
#include "base/stencil_types.hpp"

#ifdef SUPPORT_HIP
#include "base/hip/hip_matrix_dia.hpp"
#include "base/hip/hip_matrix_ell.hpp"
#include "base/hip/hip_matrix_hyb.hpp"
#endif

using namespace rocalution_unit;

namespace
{
    using Param = std::tuple<Backend, Format>;

    std::string backend_param_name(const testing::TestParamInfo<Backend>& info)
    {
        return backend_name(info.param);
    }

    // n x n band of half bandwidth w with a dominant diagonal and small positive
    // integers off it, so that products are exact
    template <typename T>
    Csr<T> band(int n, int w)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = std::max(0, i - w); j <= std::min(n - 1, i + w); ++j)
            {
                A.add(j, static_cast<T>(j == i ? 100 + i % 9 : 1 + (i + 2 * j) % 5));
            }
            A.end_row();
        }
        return A;
    }

    // n x m matrix without zeros
    template <typename T>
    Csr<T> full(int n, int m)
    {
        Csr<T> A(n, m);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                A.add(j, static_cast<T>(1 + (3 * i + 5 * j) % 7));
            }
            A.end_row();
        }
        return A;
    }

    template <typename T>
    std::vector<T> small_integers(int n)
    {
        std::vector<T> x(n);
        for(int i = 0; i < n; ++i)
        {
            x[i] = static_cast<T>((3 * i + 1) % 7 - 3);
        }
        return x;
    }

    template <typename T>
    std::vector<T> ramp(int n)
    {
        std::vector<T> y(n);
        for(int i = 0; i < n; ++i)
        {
            y[i] = static_cast<T>(i - 3);
        }
        return y;
    }

    // y + scalar * A x
    template <typename T>
    std::vector<T> apply_add(const Csr<T>& A, const std::vector<T>& x, T scalar, std::vector<T> y)
    {
        const std::vector<T> Ax = A.apply(x);
        for(int i = 0; i < A.n; ++i)
        {
            y[i] += scalar * Ax[i];
        }
        return y;
    }

    // BCSR keeps the zeros of its blocks
    template <typename T>
    void expect_matrix(const BaseMatrix<T>& A, Format f, const Csr<T>& ref)
    {
        EXPECT_EQ(A.GetMatFormat(), f.id);
        expect_csr_eq(*host_csr_of(A), ref, f.id == rocalution::BCSR);
    }

    template <typename T>
    Csr<T> lower_part(const Csr<T>& A)
    {
        return filtered(A, [](int i, int j, T) { return j <= i; });
    }

    template <typename T>
    Csr<T> upper_part(const Csr<T>& A)
    {
        return filtered(A, [](int i, int j, T) { return j >= i; });
    }
}

// HIP MCSR Apply and ApplyAdd pick kernel_mcsr_spmv<N> by nnz / nrow, with
// wider bins on wave64 devices
namespace
{
    struct RowLengthBin
    {
        int half_bandwidth;
        int from;
        int below;
    };

    std::vector<RowLengthBin> mcsr_bins()
    {
        std::vector<RowLengthBin> bins = {{1, 0, 4}, {2, 4, 8}, {5, 8, 16}, {10, 16, 32}};
        if(descriptor().HIP_warp == 64)
        {
            bins.push_back({20, 32, 64});
            bins.push_back({40, 64, INT_MAX});
        }
        else
        {
            bins.push_back({20, 32, INT_MAX});
        }
        return bins;
    }

    template <typename P>
    class matrix_kernels_mcsr : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(matrix_kernels_mcsr, WithBackends<RealTypes>, TypedName);

TYPED_TEST(matrix_kernels_mcsr, apply_in_every_row_length_bin)
{
    using T = typename TypeParam::value_type;
    for(const RowLengthBin& bin : mcsr_bins())
    {
        SCOPED_TRACE("half bandwidth " + std::to_string(bin.half_bandwidth));
        const Csr<T>  ref         = band<T>(300, bin.half_bandwidth);
        const int64_t nnz_per_row = ref.nnz() / ref.n;
        EXPECT_GE(nnz_per_row, bin.from);
        EXPECT_LT(nnz_per_row, bin.below);

        auto A = make_matrix(TypeParam::backend, Format{rocalution::MCSR, 1}, *ref.host_csr());
        ASSERT_IN_LOOP(A != nullptr);

        const std::vector<T> x = small_integers<T>(ref.n);
        auto out = make_vector(TypeParam::backend, std::vector<T>(ref.n, static_cast<T>(-7)));
        A->Apply(*make_vector(TypeParam::backend, x), out.get());
        EXPECT_EQ(values_of(*out), ref.apply(x));
    }
}

TYPED_TEST(matrix_kernels_mcsr, apply_add_in_every_row_length_bin)
{
    using T = typename TypeParam::value_type;
    for(const RowLengthBin& bin : mcsr_bins())
    {
        SCOPED_TRACE("half bandwidth " + std::to_string(bin.half_bandwidth));
        const Csr<T>  ref         = band<T>(300, bin.half_bandwidth);
        const int64_t nnz_per_row = ref.nnz() / ref.n;
        EXPECT_GE(nnz_per_row, bin.from);
        EXPECT_LT(nnz_per_row, bin.below);

        auto A = make_matrix(TypeParam::backend, Format{rocalution::MCSR, 1}, *ref.host_csr());
        ASSERT_IN_LOOP(A != nullptr);

        const std::vector<T> x   = small_integers<T>(ref.n);
        auto                 out = make_vector(TypeParam::backend, ramp<T>(ref.n));
        A->ApplyAdd(*make_vector(TypeParam::backend, x), static_cast<T>(0.5), out.get());
        EXPECT_EQ(values_of(*out), apply_add(ref, x, static_cast<T>(0.5), ramp<T>(ref.n)));
    }
}

namespace
{
    class matrix_kernels_formats : public testing::TestWithParam<Param>
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
    };

    class matrix_kernels_data_ptr : public matrix_kernels_formats
    {
    };

    // The arrays of a matrix left to the caller, with what SetDataPtr needs to take
    // them back. extra is the ELL width, the number of DIA diagonals or the BCSR
    // block dimension.
    template <typename T>
    struct Arrays
    {
        int*    row   = nullptr;
        int*    col   = nullptr;
        T*      val   = nullptr;
        int64_t nnz   = 0;
        int     nrow  = 0;
        int     ncol  = 0;
        int     extra = 0;
    };

    template <typename T>
    Arrays<T> leave(BaseMatrix<T>& A)
    {
        Arrays<T> a;
        a.nnz  = A.GetNnz();
        a.nrow = A.GetM();
        a.ncol = A.GetN();
        switch(A.GetMatFormat())
        {
        case rocalution::COO:
            A.LeaveDataPtrCOO(&a.row, &a.col, &a.val);
            break;
        case rocalution::ELL:
            A.LeaveDataPtrELL(&a.col, &a.val, a.extra);
            break;
        case rocalution::DIA:
            A.LeaveDataPtrDIA(&a.col, &a.val, a.extra);
            break;
        case rocalution::MCSR:
            A.LeaveDataPtrMCSR(&a.row, &a.col, &a.val);
            break;
        case rocalution::BCSR:
            A.LeaveDataPtrBCSR(&a.row, &a.col, &a.val, a.extra);
            break;
        case rocalution::DENSE:
            A.LeaveDataPtrDENSE(&a.val);
            break;
        default:
            ADD_FAILURE() << "no data pointer interface for format " << A.GetMatFormat();
        }
        return a;
    }

    template <typename T>
    void set(BaseMatrix<T>& A, Arrays<T>& a)
    {
        switch(A.GetMatFormat())
        {
        case rocalution::COO:
            A.SetDataPtrCOO(&a.row, &a.col, &a.val, a.nnz, a.nrow, a.ncol);
            break;
        case rocalution::ELL:
            A.SetDataPtrELL(&a.col, &a.val, a.nnz, a.nrow, a.ncol, a.extra);
            break;
        case rocalution::DIA:
            A.SetDataPtrDIA(&a.col, &a.val, a.nnz, a.nrow, a.ncol, a.extra);
            break;
        case rocalution::MCSR:
            A.SetDataPtrMCSR(&a.row, &a.col, &a.val, a.nnz, a.nrow, a.ncol);
            break;
        case rocalution::BCSR:
            A.SetDataPtrBCSR(&a.row,
                             &a.col,
                             &a.val,
                             a.nnz / (a.extra * a.extra),
                             a.nrow / a.extra,
                             a.ncol / a.extra,
                             a.extra);
            break;
        case rocalution::DENSE:
            A.SetDataPtrDENSE(&a.val, a.nrow, a.ncol);
            break;
        default:
            ADD_FAILURE() << "no data pointer interface for format " << A.GetMatFormat();
        }
    }

    template <typename T>
    int longest_row(const Csr<T>& A)
    {
        int longest = 0;
        for(int i = 0; i < A.n; ++i)
        {
            longest = std::max(longest, static_cast<int>(A.row_offset[i + 1] - A.row_offset[i]));
        }
        return longest;
    }

    template <typename T>
    int diagonals(const Csr<T>& A)
    {
        std::vector<bool> used(A.n + A.m, false);
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                used[A.col[j] - i + A.n] = true;
            }
        }
        return static_cast<int>(std::count(used.begin(), used.end(), true));
    }

    // The ELL width or the number of DIA diagonals as the matrix reports it
    template <typename T>
    int reported_width(const BaseMatrix<T>& A)
    {
        if(auto ell = dynamic_cast<const rocalution::HostMatrixELL<T>*>(&A))
        {
            return ell->GetMaxRow();
        }
        if(auto dia = dynamic_cast<const rocalution::HostMatrixDIA<T>*>(&A))
        {
            return dia->GetNDiag();
        }
#ifdef SUPPORT_HIP
        if(auto ell = dynamic_cast<const rocalution::HIPAcceleratorMatrixELL<T>*>(&A))
        {
            return ell->GetMaxRow();
        }
        if(auto dia = dynamic_cast<const rocalution::HIPAcceleratorMatrixDIA<T>*>(&A))
        {
            return dia->GetNDiag();
        }
#endif
        ADD_FAILURE() << "not an ELL or DIA matrix";
        return -1;
    }

    template <typename T>
    void expect_leave_and_set(Backend b, Format f, const Csr<T>& ref)
    {
        // The accelerator converts to MCSR only through the host
        std::unique_ptr<BaseMatrix<T>> A;
        if(f.id == rocalution::MCSR)
        {
            A = make_matrix(b, f, *ref.host_csr());
            ASSERT_NE(A, nullptr);
        }
        else
        {
            auto csr = make_matrix(b, Format{rocalution::CSR, 1}, *ref.host_csr());
            ASSERT_NE(csr, nullptr);
            A = empty_matrix<T>(b, f);
            ASSERT_TRUE(A->ConvertFrom(*csr));
        }

        const bool has_width = f.id == rocalution::ELL || f.id == rocalution::DIA;
        const int  width     = has_width ? reported_width(*A) : 0;
        Arrays<T>  arrays    = leave(*A);
        EXPECT_EQ(A->GetM(), 0);
        EXPECT_EQ(A->GetN(), 0);
        EXPECT_EQ(A->GetNnz(), 0);
        EXPECT_NE(arrays.val, nullptr);

        switch(f.id)
        {
        case rocalution::ELL:
            EXPECT_EQ(width, longest_row(ref));
            EXPECT_EQ(arrays.extra, longest_row(ref));
            break;
        case rocalution::DIA:
            EXPECT_EQ(width, diagonals(ref));
            EXPECT_EQ(arrays.extra, diagonals(ref));
            EXPECT_EQ(arrays.nnz, std::max(ref.n, ref.m) * arrays.extra);
            break;
        case rocalution::BCSR:
            EXPECT_EQ(arrays.extra, f.blockdim);
            break;
        }

        auto B = empty_matrix<T>(b, f);
        set(*B, arrays);
        EXPECT_EQ(B->GetNnz(), arrays.nnz);
        expect_matrix(*B, f, ref);
    }

    // HYB has no data pointer interface, MCSR holds only square matrices
    template <typename T>
    void test_leave_and_set(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        {
            SCOPED_TRACE("square");
            expect_leave_and_set(b, f, banded_with_gaps<T>(12));
        }
        if(f.id == rocalution::MCSR)
        {
            return;
        }

        SCOPED_TRACE("wide");
        const Csr<T> wide = full<T>(6, 12);
        if(f.id == rocalution::DIA && b == Backend::Host)
        {
            KNOWN_BUG("csr_to_dia sizes the values by the smaller dimension, so LeaveDataPtrDIA "
                      "aborts");
            expect_in_child([&] { expect_leave_and_set(b, f, wide); });
            return;
        }
        expect_leave_and_set(b, f, wide);
    }
}

// gtest has no typed value-parameterized tests, so each case runs both value types
TEST_P(matrix_kernels_data_ptr, leave_and_set_the_backend_arrays)
{
    test_leave_and_set<float>(backend(), format());
    test_leave_and_set<double>(backend(), format());
}

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_kernels_data_ptr,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::COO, 1},
                                                          Format{rocalution::ELL, 1},
                                                          Format{rocalution::DIA, 1},
                                                          Format{rocalution::MCSR, 1},
                                                          Format{rocalution::BCSR, 2},
                                                          Format{rocalution::BCSR, 3},
                                                          Format{rocalution::DENSE, 1})),
                         backend_format_name);

namespace
{
    class matrix_kernels_without_entries : public matrix_kernels_formats
    {
    };

    // n x n without non-zeros, and n x 0 in DENSE, which stores every entry.
    // BCSR is allocated, as the conversion of an empty matrix ignores the block
    // dimension (a known bug).
    template <typename T>
    std::unique_ptr<BaseMatrix<T>> without_entries(Backend b, Format f, int n)
    {
        auto A = empty_matrix<T>(b, f);
        if(f.id == rocalution::DENSE)
        {
            A->AllocateDENSE(n, 0);
        }
        else if(f.id == rocalution::BCSR)
        {
            A->AllocateBCSR(0, n / f.blockdim, n / f.blockdim, f.blockdim);
        }
        else
        {
            auto csr = empty_matrix<T>(b, Format{rocalution::CSR, 1});
            csr->AllocateCSR(0, n, n);
            EXPECT_TRUE(A->ConvertFrom(*csr));
        }
        EXPECT_EQ(A->GetMatFormat(), f.id);
        EXPECT_EQ(A->GetNnz(), 0);
        return A;
    }

    // Apply of an empty matrix either zeroes the output or skips the matrix and
    // leaves it (only host COO and DENSE zero it); LocalMatrix zeroes it itself
    template <typename T>
    void test_apply_without_entries(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        auto A = without_entries<T>(b, f, 6);
        ASSERT_EQ(A->GetM(), 6);

        auto in  = make_vector(b, std::vector<T>(A->GetN(), static_cast<T>(1)));
        auto out = make_vector(b, ramp<T>(6));
        A->Apply(*in, out.get());
        const std::vector<T> applied = values_of(*out);
        EXPECT_TRUE(applied == std::vector<T>(6, static_cast<T>(0)) || applied == ramp<T>(6))
            << testing::PrintToString(applied);

        out = make_vector(b, ramp<T>(6));
        A->ApplyAdd(*in, static_cast<T>(0.5), out.get());
        EXPECT_EQ(values_of(*out), ramp<T>(6));

        std::vector<int> identity(6);
        for(int i = 0; i < 6; ++i)
        {
            identity[i] = i;
        }
        auto p = make_vector(b, identity);
        if(implements(b, f.id, "Permute"))
        {
            EXPECT_TRUE(A->Permute(*p));
        }
        if(implements(b, f.id, "PermuteBackward"))
        {
            EXPECT_TRUE(A->PermuteBackward(*p));
        }
        if(implements(b, f.id, "Sort"))
        {
            EXPECT_TRUE(A->Sort());
        }
        EXPECT_EQ(A->GetNnz(), 0);
        EXPECT_EQ(A->GetM(), 6);
    }

    // Within the backend and between the backends
    template <typename T>
    void test_copies_without_entries(Backend b, Format f, const BaseMatrix<T>* A)
    {
        SCOPED_TRACE(type_name<T>());
        auto expect = [&](const std::string& copy, const BaseMatrix<T>& B) {
            SCOPED_TRACE(copy);
            EXPECT_EQ(B.GetMatFormat(), f.id);
            EXPECT_EQ(B.GetM(), A->GetM());
            EXPECT_EQ(B.GetN(), A->GetN());
            EXPECT_EQ(B.GetNnz(), 0);
        };

        auto B = empty_matrix<T>(b, f);
        B->CopyFrom(*A);
        expect("CopyFrom", *B);
        B = empty_matrix<T>(b, f);
        B->CopyFromAsync(*A);
        rocalution::_rocalution_sync();
        expect("CopyFromAsync", *B);
        B = empty_matrix<T>(b, f);
        A->CopyTo(B.get());
        expect("CopyTo", *B);
        B = empty_matrix<T>(b, f);
        A->CopyToAsync(B.get());
        rocalution::_rocalution_sync();
        expect("CopyToAsync", *B);

        if(b == Backend::Accelerator)
        {
            const auto& acc = dynamic_cast<const rocalution::AcceleratorMatrix<T>&>(*A);
            auto        H   = empty_host_matrix<T>(f);
            acc.CopyToHost(H.get());
            expect("CopyToHost", *H);
            auto async_host = empty_host_matrix<T>(f);
            acc.CopyToHostAsync(async_host.get());
            rocalution::_rocalution_sync();
            expect("CopyToHostAsync", *async_host);

            auto C = empty_matrix<T>(b, f);
            dynamic_cast<rocalution::AcceleratorMatrix<T>&>(*C).CopyFromHost(*H);
            expect("CopyFromHost", *C);
            C = empty_matrix<T>(b, f);
            dynamic_cast<rocalution::AcceleratorMatrix<T>&>(*C).CopyFromHostAsync(*H);
            rocalution::_rocalution_sync();
            expect("CopyFromHostAsync", *C);
        }
    }

    template <typename T>
    void test_solves_without_entries(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        auto                 A         = without_entries<T>(b, f, 6);
        auto                 in        = make_vector(b, std::vector<T>(6, static_cast<T>(1)));
        const std::vector<T> untouched = ramp<T>(6);
        auto                 out       = make_vector(b, untouched);

        if(implements(b, f.id, "ILU0Factorize"))
        {
            EXPECT_TRUE(A->ILU0Factorize());
        }
        for(const std::string& solve : {"LUSolve", "LLSolve", "LSolve", "USolve"})
        {
            if(!implements(b, f.id, solve))
            {
                continue;
            }
            SCOPED_TRACE(solve);
            const bool done = solve == "LUSolve"   ? A->LUSolve(*in, out.get())
                              : solve == "LLSolve" ? A->LLSolve(*in, out.get())
                              : solve == "LSolve"  ? A->LSolve(*in, out.get())
                                                   : A->USolve(*in, out.get());
            EXPECT_TRUE(done);
            EXPECT_EQ(values_of(*out), untouched);
        }
        EXPECT_EQ(A->GetNnz(), 0);
    }

    template <typename T>
    void test_leave_and_set_without_entries(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        auto      A      = without_entries<T>(b, f, 6);
        const int ncol   = A->GetN();
        Arrays<T> arrays = leave(*A);
        EXPECT_EQ(A->GetM(), 0);
        EXPECT_EQ(arrays.nnz, 0);

        auto B = empty_matrix<T>(b, f);
        set(*B, arrays);
        EXPECT_EQ(B->GetM(), 6);
        EXPECT_EQ(B->GetN(), ncol);
        EXPECT_EQ(B->GetNnz(), 0);
    }
}

TEST_P(matrix_kernels_without_entries, apply_zeroes_or_leaves_the_output_and_apply_add_leaves_it)
{
    test_apply_without_entries<float>(backend(), format());
    test_apply_without_entries<double>(backend(), format());
}

TEST_P(matrix_kernels_without_entries, copies_have_no_entries)
{
    test_copies_without_entries<float>(
        backend(), format(), without_entries<float>(backend(), format(), 6).get());
    test_copies_without_entries<double>(
        backend(), format(), without_entries<double>(backend(), format(), 6).get());
}

TEST_P(matrix_kernels_without_entries, copies_of_a_never_allocated_matrix_are_empty)
{
    auto checks = [&] {
        test_copies_without_entries<float>(
            backend(), format(), empty_matrix<float>(backend(), format()).get());
        test_copies_without_entries<double>(
            backend(), format(), empty_matrix<double>(backend(), format()).get());
    };

    if(format().id == rocalution::BCSR)
    {
        KNOWN_BUG("the BCSR constructors leave the block counts uninitialised, so the copy "
                  "asserts or exits with a HIP error");
        expect_in_child(checks);
        return;
    }
    if(format().id == rocalution::MCSR && backend() == Backend::Accelerator)
    {
        KNOWN_BUG("the HIP MCSR copies read the row offsets of an unallocated source and exit "
                  "with a HIP error");
        expect_in_child(checks);
        return;
    }
    checks();
}

// HYB has no data pointer interface
TEST_P(matrix_kernels_without_entries, leave_and_set_the_backend_arrays)
{
    if(format().id == rocalution::HYB)
    {
        GTEST_SKIP() << "HYB has no data pointer interface";
    }
    test_leave_and_set_without_entries<float>(backend(), format());
    test_leave_and_set_without_entries<double>(backend(), format());
}

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_kernels_without_entries,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::COO, 1},
                                                          Format{rocalution::ELL, 1},
                                                          Format{rocalution::DIA, 1},
                                                          Format{rocalution::HYB, 1},
                                                          Format{rocalution::MCSR, 1},
                                                          Format{rocalution::BCSR, 2},
                                                          Format{rocalution::BCSR, 3},
                                                          Format{rocalution::DENSE, 1})),
                         backend_format_name);

namespace
{
    class matrix_kernels_formats_accelerator : public matrix_kernels_formats
    {
    };

    template <typename T>
    void test_async_copies(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T> ref   = banded_with_gaps<T>(12);
        const Csr<T> other = transformed(ref, [](int, int, T v) { return -v; });

        auto A = make_matrix(b, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);
        auto host = host_copy(*A);

        {
            SCOPED_TRACE("CopyFromAsync into an empty matrix");
            auto B = empty_matrix<T>(b, f);
            B->CopyFromAsync(*host);
            rocalution::_rocalution_sync();
            expect_matrix(*B, f, ref);
        }
        {
            SCOPED_TRACE("CopyFromAsync into a matrix of the same size");
            auto B = make_matrix(b, f, *other.host_csr());
            ASSERT_NE(B, nullptr);
            B->CopyFromAsync(*host);
            rocalution::_rocalution_sync();
            expect_matrix(*B, f, ref);
        }
        {
            SCOPED_TRACE("CopyToAsync into an empty matrix");
            auto H = empty_host_matrix<T>(f);
            A->CopyToAsync(H.get());
            rocalution::_rocalution_sync();
            expect_matrix(*H, f, ref);
        }
        {
            SCOPED_TRACE("CopyToAsync into a matrix of the same size");
            auto O = make_matrix(b, f, *other.host_csr());
            ASSERT_NE(O, nullptr);
            auto H = host_copy(*O);
            A->CopyToAsync(H.get());
            rocalution::_rocalution_sync();
            expect_matrix(*H, f, ref);
        }
    }
}

// The asynchronous copies of an accelerator matrix take a host matrix of the
// same format, empty or of the same size
TEST_P(matrix_kernels_formats_accelerator, async_copies_from_and_to_a_host_matrix)
{
    test_async_copies<float>(backend(), format());
    test_async_copies<double>(backend(), format());
}

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_kernels_formats_accelerator,
                         testing::Combine(testing::Values(Backend::Accelerator),
                                          testing::Values(Format{rocalution::COO, 1},
                                                          Format{rocalution::ELL, 1},
                                                          Format{rocalution::DIA, 1},
                                                          Format{rocalution::HYB, 1},
                                                          Format{rocalution::MCSR, 1},
                                                          Format{rocalution::BCSR, 2},
                                                          Format{rocalution::DENSE, 1})),
                         backend_format_name);

namespace
{
    class matrix_kernels_conversion : public testing::TestWithParam<Backend>
    {
    protected:
        void SetUp() override
        {
            REQUIRE_BACKEND(GetParam());
        }
    };
}

// Each backend converts a format only from CSR and from the format itself.
// Without a native conversion ConvertFrom returns false and leaves the
// destination empty, so that LocalMatrix can convert through CSR.
TEST_P(matrix_kernels_conversion, unimplemented_operations_return_false_and_change_nothing)
{
    const Csr<double>         ref     = Tridiagonal<double>(12);
    const std::vector<Format> formats = {Format{rocalution::COO, 1},
                                         Format{rocalution::ELL, 1},
                                         Format{rocalution::DIA, 1},
                                         Format{rocalution::HYB, 1},
                                         Format{rocalution::MCSR, 1},
                                         Format{rocalution::BCSR, 2},
                                         Format{rocalution::DENSE, 1}};

    for(Format from : formats)
    {
        for(Format to : formats)
        {
            if(from.id == to.id)
            {
                continue;
            }
            SCOPED_TRACE(format_name(from) + " to " + format_name(to));

            auto src = make_matrix(GetParam(), from, *ref.host_csr());
            ASSERT_IN_LOOP(src != nullptr);

            auto dst = empty_matrix(GetParam(), to);
            EXPECT_FALSE(dst->ConvertFrom(*src));
            EXPECT_EQ(dst->GetMatFormat(), to.id);
            EXPECT_EQ(dst->GetNnz(), 0);
            expect_matrix(*src, from, ref);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(backends,
                         matrix_kernels_conversion,
                         testing::Values(Backend::Host, Backend::Accelerator),
                         backend_param_name);

namespace
{
    template <typename P>
    class matrix_kernels_hip_conversion : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(matrix_kernels_hip_conversion, AcceleratorOnly<RealTypes>, TypedName);

// ELL and DIA reject matrices that would need more than 5 times the CSR
// storage, on the accelerator as on the host. A dense first row gives an ELL
// width of n and n diagonals at 2 non-zeros per row.
TYPED_TEST(matrix_kernels_hip_conversion, ell_and_dia_reject_oversized_matrices)
{
    using T = typename TypeParam::value_type;

    const int n = 12;
    Csr<T>    ref(n);
    for(int j = 0; j < n; ++j)
    {
        ref.add(j, static_cast<T>(j + 1));
    }
    ref.end_row();
    for(int i = 1; i < n; ++i)
    {
        ref.add(i, static_cast<T>(i + 1));
        ref.end_row();
    }
    EXPECT_GT(longest_row(ref), 5 * (ref.nnz() / n));
    EXPECT_GT(diagonals(ref), 5 * (ref.nnz() / n));

    auto csr = make_matrix(Backend::Accelerator, Format{rocalution::CSR, 1}, *ref.host_csr());
    ASSERT_NE(csr, nullptr);

    for(Format f : {Format{rocalution::ELL, 1}, Format{rocalution::DIA, 1}})
    {
        SCOPED_TRACE(format_name(f));
        auto X = empty_matrix<T>(Backend::Accelerator, f);
        EXPECT_FALSE(X->ConvertFrom(*csr));
        EXPECT_EQ(X->GetNnz(), 0);
    }

    auto hyb = empty_matrix<T>(Backend::Accelerator, Format{rocalution::HYB, 1});
    ASSERT_TRUE(hyb->ConvertFrom(*csr));
    expect_matrix(*hyb, Format{rocalution::HYB, 1}, ref);
}

TEST(matrix_kernels_hip_conversion, bcsr_requires_dimensions_divisible_by_the_block_dimension)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    const Csr<double> ref = Tridiagonal<double>(12);
    auto csr = make_matrix(Backend::Accelerator, Format{rocalution::CSR, 1}, *ref.host_csr());
    ASSERT_NE(csr, nullptr);

    auto X = empty_matrix(Backend::Accelerator, Format{rocalution::BCSR, 5});
    EXPECT_NE(ref.n % 5, 0);
    EXPECT_FALSE(X->ConvertFrom(*csr));
    EXPECT_EQ(X->GetNnz(), 0);
}

namespace
{
    class matrix_kernels_apply_add : public matrix_kernels_formats
    {
    };

    // HYB stores ceil(nnz / n) entries of every row in ELL and the rest in COO
    template <typename T>
    int64_t hyb_ell_width(const Csr<T>& A)
    {
        return (A.nnz() - 1) / A.n + 1;
    }

    template <typename T>
    int64_t hyb_coo_nnz(const Csr<T>& A)
    {
        int64_t coo = 0;
        for(int i = 0; i < A.n; ++i)
        {
            coo += std::max<int64_t>(0, A.row_offset[i + 1] - A.row_offset[i] - hyb_ell_width(A));
        }
        return coo;
    }

    // ELL width, ELL non-zeros and COO non-zeros of a HYB matrix of any backend
    template <typename T>
    std::tuple<int64_t, int64_t, int64_t> hyb_split(const BaseMatrix<T>& A)
    {
        if(auto host = dynamic_cast<const rocalution::HostMatrixHYB<T>*>(&A))
        {
            return {host->GetELLMaxRow(), host->GetELLNnz(), host->GetCOONnz()};
        }
#ifdef SUPPORT_HIP
        if(auto accel = dynamic_cast<const rocalution::HIPAcceleratorMatrixHYB<T>*>(&A))
        {
            return {accel->GetELLMaxRow(), accel->GetELLNnz(), accel->GetCOONnz()};
        }
#endif
        ADD_FAILURE() << "not a HYB matrix";
        return {-1, -1, -1};
    }

    template <typename T>
    std::vector<std::pair<std::string, Csr<T>>> apply_add_cases(Format f)
    {
        if(f.id == rocalution::HYB)
        {
            return {{"ELL part only", Tridiagonal<T>(12)}, {"ELL and COO parts", arrow<T>(12)}};
        }
        return {{"wide", full<T>(5, 7)}, {"tall", full<T>(7, 5)}};
    }

    // The accelerator HYB and DENSE ApplyAdd overwrite the output (a known bug), which
    // does not show when it starts out zero
    template <typename T>
    void test_apply_add(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const T half = static_cast<T>(0.5);

        for(const auto& matrix : apply_add_cases<T>(f))
        {
            SCOPED_TRACE(matrix.first + " into a zero vector");
            const Csr<T>& ref = matrix.second;

            auto A = make_matrix(b, f, *ref.host_csr());
            ASSERT_IN_LOOP(A != nullptr);
            if(f.id == rocalution::HYB)
            {
                EXPECT_EQ((matrix.first == "ELL and COO parts"), hyb_coo_nnz(ref) > 0);
                EXPECT_EQ(hyb_split(*A),
                          std::make_tuple(
                              hyb_ell_width(ref), hyb_ell_width(ref) * ref.n, hyb_coo_nnz(ref)));
            }

            const std::vector<T> x   = small_integers<T>(ref.m);
            auto                 out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(0)));
            A->ApplyAdd(*make_vector(b, x), half, out.get());
            EXPECT_EQ(values_of(*out),
                      apply_add(ref, x, half, std::vector<T>(ref.n, static_cast<T>(0))));
        }

        if(b == Backend::Accelerator)
        {
            KNOWN_BUG("ellmv and gemv get beta 0, so the output is overwritten");
        }

        for(const auto& matrix : apply_add_cases<T>(f))
        {
            SCOPED_TRACE(matrix.first);
            const Csr<T>& ref = matrix.second;

            auto A = make_matrix(b, f, *ref.host_csr());
            ASSERT_IN_LOOP(A != nullptr);

            const std::vector<T> x   = small_integers<T>(ref.m);
            auto                 out = make_vector(b, ramp<T>(ref.n));
            A->ApplyAdd(*make_vector(b, x), half, out.get());
            EXPECT_EQ(values_of(*out), apply_add(ref, x, half, ramp<T>(ref.n)));
        }
    }
}

TEST_P(matrix_kernels_apply_add, apply_add_adds_the_scaled_product_to_the_output)
{
    test_apply_add<float>(backend(), format());
    test_apply_add<double>(backend(), format());
}

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_kernels_apply_add,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::HYB, 1},
                                                          Format{rocalution::DENSE, 1})),
                         backend_format_name);

namespace
{
    template <typename P>
    class matrix_kernels_coo : public TypedBackendTest<P>
    {
    };

    template <typename T>
    using Entry = std::tuple<int, int, T>;

    // The entries of a COO matrix of any backend, sorted, because Permute leaves
    // them in their old order (a known bug)
    template <typename T>
    std::vector<Entry<T>> sorted_coo_entries(const BaseMatrix<T>& A)
    {
        auto host = host_copy(A);
        auto coo  = dynamic_cast<rocalution::HostMatrixCOO<T>*>(host.get());
        EXPECT_NE(coo, nullptr);
        if(coo == nullptr)
        {
            return {};
        }

        const int64_t    nnz = coo->GetNnz();
        std::vector<int> row(nnz), col(nnz);
        std::vector<T>   val(nnz);
        coo->CopyToCOO(row.data(), col.data(), val.data());

        std::vector<Entry<T>> entries;
        for(int64_t k = 0; k < nnz; ++k)
        {
            entries.emplace_back(row[k], col[k], val[k]);
        }
        std::sort(entries.begin(), entries.end());
        return entries;
    }

    template <typename T>
    std::vector<Entry<T>> entries_of(const Csr<T>& A)
    {
        std::vector<Entry<T>> entries;
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                entries.emplace_back(i, A.col[j], A.val[j]);
            }
        }
        return entries;
    }
}

TYPED_TEST_SUITE(matrix_kernels_coo, WithBackends<RealTypes>, TypedName);

// The arrays of a backend matrix are left to the caller, copied and handed back
TYPED_TEST(matrix_kernels_coo, copies_into_and_out_of_backend_arrays)
{
    using T = typename TypeParam::value_type;

    const Backend b       = TypeParam::backend;
    const Csr<T>  ref     = banded_with_gaps<T>(12);
    const Csr<T>  negated = transformed(ref, [](int, int, T v) { return -v; });
    const Format  coo{rocalution::COO, 1};

    auto A = make_matrix(b, coo, *ref.host_csr());
    ASSERT_NE(A, nullptr);
    Arrays<T> arrays = leave(*A);
    ASSERT_NE(arrays.row, nullptr);

    auto B = empty_matrix<T>(b, coo);
    B->AllocateCOO(arrays.nnz, arrays.nrow, arrays.ncol);
    B->CopyFromCOO(arrays.row, arrays.col, arrays.val);
    expect_matrix(*B, coo, ref);

    auto N = make_matrix(b, coo, *negated.host_csr());
    ASSERT_NE(N, nullptr);
    N->CopyToCOO(arrays.row, arrays.col, arrays.val);

    auto C = empty_matrix<T>(b, coo);
    set(*C, arrays);
    expect_matrix(*C, coo, negated);
}

namespace
{
    template <typename P>
    class matrix_kernels_hip_coo : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(matrix_kernels_hip_coo, AcceleratorOnly<RealTypes>, TypedName);

// The HIP COO permutation caps its grid at HIP_block_size * HIP_max_threads
// non-zeros, about 16 million with the device values, so the cap is lowered
TYPED_TEST(matrix_kernels_hip_coo, permute_with_a_capped_grid_moves_every_entry)
{
    using T = typename TypeParam::value_type;

    rocalution::Rocalution_Backend_Descriptor capped = descriptor();
    capped.HIP_max_threads                           = 1;

    const Csr<T>  ref = band<T>(200, 1);
    const int64_t k   = (ref.nnz() / capped.HIP_block_size) / capped.HIP_max_threads + 1;
    EXPECT_GT(k, 1);

    std::vector<int> perm(ref.n), inverse(ref.n);
    for(int i = 0; i < ref.n; ++i)
    {
        perm[i]          = (7 * i + 3) % ref.n;
        inverse[perm[i]] = i;
    }
    auto p = make_vector(Backend::Accelerator, perm);

    auto host = empty_host_matrix<T>(Format{rocalution::COO, 1});
    ASSERT_TRUE(host->ConvertFrom(*ref.host_csr()));

    using Permute = std::function<bool(BaseMatrix<T>&)>;
    const std::vector<std::tuple<std::string, Permute, std::vector<int>>> permutations
        = {{"Permute", [&](BaseMatrix<T>& A) { return A.Permute(*p); }, perm},
           {"PermuteBackward", [&](BaseMatrix<T>& A) { return A.PermuteBackward(*p); }, inverse}};

    for(const auto& permutation : permutations)
    {
        SCOPED_TRACE(std::get<0>(permutation));
        std::unique_ptr<rocalution::AcceleratorMatrix<T>> A(
            rocalution::_rocalution_init_base_backend_matrix<T>(capped, rocalution::COO, 1));
        A->CopyFromHost(*host);

        ASSERT_IN_LOOP(std::get<1>(permutation)(*A));
        EXPECT_EQ(sorted_coo_entries(*A), entries_of(permuted(ref, std::get<2>(permutation))));
    }
}

namespace
{
    class matrix_kernels_bcsr : public matrix_kernels_formats
    {
    };

    // LUAnalyse and UAnalyse of a matrix that was not factorized allocate the
    // buffer that ILU0Factorize otherwise provides
    template <typename T>
    void test_lu_solve_of_an_unfactorized_matrix(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T>         ref = band<T>(12, 2);
        const std::vector<T> D   = dense(ref);
        const std::vector<T> rb  = rhs<T>(ref.n);

        auto A = make_matrix(b, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);
        auto out = make_vector(b, std::vector<T>(ref.n));

        A->LUAnalyse();
        ASSERT_TRUE(A->LUSolve(*make_vector(b, rb), out.get()));
        A->LUAnalyseClear();
        expect_near(values_of(*out),
                    upper_solve(D, ref.n, lower_solve(D, ref.n, rb, true), false),
                    tol<T>());
    }

    template <typename T>
    void test_upper_solve(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T>         ref = band<T>(12, 2);
        const std::vector<T> D   = dense(ref);
        const std::vector<T> rb  = rhs<T>(ref.n);

        auto A = make_matrix(b, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);

        for(bool unit : {false, true})
        {
            SCOPED_TRACE(unit ? "unit diagonal" : "diagonal of the matrix");
            auto out = make_vector(b, std::vector<T>(ref.n));
            A->UAnalyse(unit);
            ASSERT_IN_LOOP(A->USolve(*make_vector(b, rb), out.get()));
            A->UAnalyseClear();
            expect_near(values_of(*out), upper_solve(D, ref.n, rb, unit), tol<T>());
        }
    }

    template <typename T>
    void test_ll_solve(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T>         ref = band<T>(12, 2);
        const std::vector<T> L   = dense(lower_part(ref));
        const std::vector<T> Lt  = dense(transposed(lower_part(ref)));
        const std::vector<T> rb  = rhs<T>(ref.n);

        std::vector<T> inv_diag(ref.n);
        for(int i = 0; i < ref.n; ++i)
        {
            inv_diag[i] = static_cast<T>(1) / L[i * ref.n + i];
        }

        auto A = make_matrix(b, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);
        auto out = make_vector(b, std::vector<T>(ref.n));

        A->LLAnalyse();
        ASSERT_TRUE(A->LLSolve(*make_vector(b, rb), *make_vector(b, inv_diag), out.get()));
        A->LLAnalyseClear();
        expect_near(values_of(*out),
                    upper_solve(Lt, ref.n, lower_solve(L, ref.n, rb, false), false),
                    tol<T>());
    }

    // A with the explicit zeros of its blockdim x blockdim blocks
    template <typename T>
    Csr<T> blocked(const Csr<T>& A, int blockdim)
    {
        const int         nb = (A.n + blockdim - 1) / blockdim;
        std::vector<bool> block(static_cast<size_t>(nb) * nb, false);
        std::vector<T>    D = dense(A);
        for(int i = 0; i < A.n; ++i)
        {
            for(PtrType j = A.row_offset[i]; j < A.row_offset[i + 1]; ++j)
            {
                block[(i / blockdim) * nb + A.col[j] / blockdim] = true;
            }
        }

        Csr<T> B(A.n, A.m);
        for(int i = 0; i < A.n; ++i)
        {
            for(int j = 0; j < A.m; ++j)
            {
                if(block[(i / blockdim) * nb + j / blockdim])
                {
                    B.add(j, D[i * A.m + j]);
                }
            }
            B.end_row();
        }
        return B;
    }

    // The factors of ref, which BCSR computes on the pattern of its blocks, and
    // the LU solve with them
    template <typename T>
    void expect_factors_and_solve(BaseMatrix<T>&        A,
                                  Format                f,
                                  const Csr<T>&         ref,
                                  const std::vector<T>& rb,
                                  const BaseVector<T>&  out)
    {
        const std::vector<T> F = ilu0_dense(blocked(ref, f.blockdim));
        expect_near(dense(csr_of(A)), F, tol<T>());
        expect_near(values_of(out),
                    upper_solve(F, ref.n, lower_solve(F, ref.n, rb, true), false),
                    tol<T>());
    }

    // A second ILU0Factorize after new values were copied in reuses the buffer of
    // the first
    template <typename T>
    void test_refactorization(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T> ref = band<T>(12, 2);
        const Csr<T> scaled
            = transformed(ref, [](int i, int, T v) { return static_cast<T>(i % 3 + 2) * v; });
        const std::vector<T> rb = rhs<T>(ref.n);

        auto A   = make_matrix(b, f, *scaled.host_csr());
        auto out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(-7)));
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->ILU0Factorize());

        A->CopyFrom(*make_matrix(b, f, *ref.host_csr()));
        ASSERT_TRUE(A->ILU0Factorize());
        A->LUAnalyse();
        ASSERT_TRUE(A->LUSolve(*make_vector(b, rb), out.get()));
        expect_factors_and_solve(*A, f, ref, rb, *out);
        A->LUAnalyseClear();
    }

    // The factorization runs on the analysis of the previous factors. On the
    // accelerator, ILU0Factorize after LUAnalyse hangs or ends in an illegal
    // memory access (a known bug).
    template <typename T>
    void test_refactorization_after_an_analysis(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T> ref = band<T>(12, 2);
        const Csr<T> scaled
            = transformed(ref, [](int i, int, T v) { return static_cast<T>(i % 3 + 2) * v; });
        const std::vector<T> rb = rhs<T>(ref.n);

        auto A   = make_matrix(b, f, *scaled.host_csr());
        auto out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(-7)));
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->ILU0Factorize());
        A->LUAnalyse();
        ASSERT_TRUE(A->LUSolve(*make_vector(b, rb), out.get()));
        expect_factors_and_solve(*A, f, scaled, rb, *out);

        A->CopyFrom(*make_matrix(b, f, *ref.host_csr()));
        ASSERT_TRUE(A->ILU0Factorize());
        out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(-7)));
        ASSERT_TRUE(A->LUSolve(*make_vector(b, rb), out.get()));
        expect_factors_and_solve(*A, f, ref, rb, *out);
        A->LUAnalyseClear();
    }

    template <typename T>
    void test_empty_conversion(Backend b, Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const int nrow = 2 * f.blockdim;
        const int ncol = 3 * f.blockdim;

        auto csr = empty_matrix<T>(b, Format{rocalution::CSR, 1});
        csr->AllocateCSR(0, nrow, ncol);

        auto A = empty_matrix<T>(b, f);
        ASSERT_TRUE(A->ConvertFrom(*csr));
        EXPECT_EQ(A->GetNnz(), 0);

        if(f.blockdim != 2)
        {
            KNOWN_BUG("an empty matrix gets block dimension 2 and even dimensions");
        }

        EXPECT_EQ(A->GetM(), nrow);
        EXPECT_EQ(A->GetN(), ncol);
        EXPECT_EQ(A->GetMatBlockDimension(), f.blockdim);
    }
}

TEST_P(matrix_kernels_bcsr, lu_solve_of_an_unfactorized_matrix)
{
    REQUIRE_NATIVE(backend(), format().id, "LUSolve");

    test_lu_solve_of_an_unfactorized_matrix<float>(backend(), format());
    test_lu_solve_of_an_unfactorized_matrix<double>(backend(), format());
}

TEST_P(matrix_kernels_bcsr, upper_solve_analysed_alone_and_again_after_clearing)
{
    REQUIRE_NATIVE(backend(), format().id, "USolve");

    test_upper_solve<float>(backend(), format());
    test_upper_solve<double>(backend(), format());
}

TEST_P(matrix_kernels_bcsr, ll_solve_with_the_inverse_diagonal)
{
    REQUIRE_NATIVE(backend(), format().id, "LLSolve");

    test_ll_solve<float>(backend(), format());
    test_ll_solve<double>(backend(), format());
}

TEST_P(matrix_kernels_bcsr, ilu0_factorize_again_after_new_values)
{
    REQUIRE_NATIVE(backend(), format().id, "ILU0Factorize");
    REQUIRE_NATIVE(backend(), format().id, "LUSolve");

    test_refactorization<float>(backend(), format());
    test_refactorization<double>(backend(), format());
}

TEST_P(matrix_kernels_bcsr, ilu0_factorize_after_an_analysis)
{
    REQUIRE_NATIVE(backend(), format().id, "ILU0Factorize");
    REQUIRE_NATIVE(backend(), format().id, "LUSolve");

    auto checks = [&] {
        test_refactorization_after_an_analysis<float>(backend(), format());
        test_refactorization_after_an_analysis<double>(backend(), format());
    };

    if(backend() == Backend::Accelerator)
    {
        KNOWN_BUG("BCSR ILU0Factorize after LUAnalyse hangs or accesses illegal memory");
        expect_in_child(checks, 60);
        return;
    }
    checks();
}

TEST_P(matrix_kernels_bcsr, factorization_and_solves_without_entries_change_nothing)
{
    REQUIRE_NATIVE(backend(), format().id, "ILU0Factorize");

    test_solves_without_entries<float>(backend(), format());
    test_solves_without_entries<double>(backend(), format());
}

TEST_P(matrix_kernels_bcsr, conversion_of_an_empty_matrix_keeps_the_dimensions)
{
    test_empty_conversion<float>(backend(), format());
    test_empty_conversion<double>(backend(), format());
}

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_kernels_bcsr,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::BCSR, 2},
                                                          Format{rocalution::BCSR, 3})),
                         backend_format_name);

namespace
{
    template <typename P>
    class matrix_kernels_iterative_solves : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_matrix(P::backend, Format{rocalution::CSR, 1}, *ref.host_csr());
        }
    };
}

TYPED_TEST_SUITE(matrix_kernels_iterative_solves, WithBackends<RealTypes>, TypedName);

// Without a tolerance every solve runs max_iter iterations, so the known bug of the
// iteration limit does not apply
TYPED_TEST(matrix_kernels_iterative_solves, ll_solve_without_a_tolerance)
{
    using T = typename TypeParam::value_type;

    const Backend        b        = TypeParam::backend;
    const Csr<T>         ref      = lower_part(band<T>(10, 2));
    const std::vector<T> L        = dense(ref);
    const std::vector<T> Lt       = dense(transposed(ref));
    const std::vector<T> rb       = rhs<T>(ref.n);
    const std::vector<T> expected = upper_solve(Lt, ref.n, lower_solve(L, ref.n, rb, false), false);

    std::vector<T> inv_diag(ref.n);
    for(int i = 0; i < ref.n; ++i)
    {
        inv_diag[i] = static_cast<T>(1) / L[i * ref.n + i];
    }

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto in = make_vector(b, rb);

    A->ItLLAnalyse();
    auto out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(0)));
    ASSERT_TRUE(A->ItLLSolve(30, 0.0, false, *in, out.get()));
    expect_near(values_of(*out), expected, tol<T>());

    auto out_inv = make_vector(b, std::vector<T>(ref.n, static_cast<T>(0)));
    ASSERT_TRUE(A->ItLLSolve(30, 0.0, false, *in, *make_vector(b, inv_diag), out_inv.get()));
    expect_near(values_of(*out_inv), expected, tol<T>());
    A->ItLLAnalyseClear();
}

// The unit lower solve of an upper triangular matrix is the identity
TYPED_TEST(matrix_kernels_iterative_solves,
           lu_solve_of_an_upper_triangular_matrix_without_a_tolerance)
{
    using T = typename TypeParam::value_type;

    const Backend        b   = TypeParam::backend;
    const Csr<T>         ref = upper_part(band<T>(10, 2));
    const std::vector<T> rb  = rhs<T>(ref.n);

    auto A = this->make(ref);
    ASSERT_NE(A, nullptr);
    auto out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(0)));

    A->ItLUAnalyse();
    ASSERT_TRUE(A->ItLUSolve(30, 0.0, false, *make_vector(b, rb), out.get()));
    A->ItLUAnalyseClear();
    expect_near(values_of(*out), upper_solve(dense(ref), ref.n, rb, false), tol<T>());
}

// host_csritsv_solve, called directly for the operations, matrix types and
// error cases that HostMatrixCSR does not use
namespace
{
    template <typename T>
    struct Itsv
    {
        bool           status     = false;
        int            iterations = 0;
        int            zero_pivot = -2;
        std::vector<T> y;
        std::vector<T> history;
    };

    template <typename T>
    struct ItsvCall
    {
        host_sparse_operation   trans    = host_sparse_operation_none;
        host_sparse_fill_mode   fill     = host_sparse_fill_mode_lower;
        host_sparse_diag_type   diag     = host_sparse_diag_type_non_unit;
        host_sparse_matrix_type type     = host_sparse_matrix_type_general;
        int                     max_iter = 50;
        const T*                tol      = nullptr;
        T                       alpha    = 1;
    };

    template <typename T>
    Itsv<T> itsv(const Csr<T>& A, const ItsvCall<T>& c, const std::vector<T>& x, std::vector<T> y0)
    {
        size_t size = 0;
        EXPECT_TRUE(rocalution::host_csritsv_buffer_size(c.trans,
                                                         A.n,
                                                         static_cast<PtrType>(A.nnz()),
                                                         c.fill,
                                                         c.diag,
                                                         c.type,
                                                         A.val.data(),
                                                         A.row_offset.data(),
                                                         A.col.data(),
                                                         &size));
        std::vector<T> buffer(size / sizeof(T) + 1);

        Itsv<T> r;
        r.iterations = c.max_iter;
        r.y          = std::move(y0);
        r.history.assign(c.max_iter, static_cast<T>(-1));
        r.status = rocalution::host_csritsv_solve(&r.iterations,
                                                  c.tol,
                                                  r.history.data(),
                                                  c.trans,
                                                  A.n,
                                                  static_cast<PtrType>(A.nnz()),
                                                  &c.alpha,
                                                  c.fill,
                                                  c.diag,
                                                  c.type,
                                                  A.val.data(),
                                                  A.row_offset.data(),
                                                  A.col.data(),
                                                  x.data(),
                                                  r.y.data(),
                                                  buffer.data(),
                                                  &r.zero_pivot);
        r.history.resize(std::min(r.iterations, c.max_iter));
        return r;
    }

    template <typename T>
    struct Jacobi
    {
        std::vector<T> y;
        int            iterations;
        std::vector<T> history;
    };

    // The documented iteration of csritsv on the dense n x n matrix A, from y = 0:
    // y += diag(A)^-1 (x - A y). It stops after the first iteration whose residual
    // max |x - A y| is at most tol, and records max |change of y| per iteration.
    template <typename T>
    Jacobi<T> jacobi(const std::vector<T>& A, int n, const std::vector<T>& x, int max_iter, T tol)
    {
        Jacobi<T> r{std::vector<T>(n, static_cast<T>(0)), max_iter, {}};
        for(int k = 0; k < max_iter; ++k)
        {
            std::vector<T> next(n);
            T              residual = 0;
            T              change   = 0;
            for(int i = 0; i < n; ++i)
            {
                T ri = x[i];
                for(int j = 0; j < n; ++j)
                {
                    ri -= A[i * n + j] * r.y[j];
                }
                next[i]  = r.y[i] + ri / A[i * n + i];
                residual = std::max(residual, std::abs(ri));
                change   = std::max(change, std::abs(next[i] - r.y[i]));
            }
            r.y = next;
            r.history.push_back(change);
            if(residual <= tol)
            {
                r.iterations = k + 1;
                break;
            }
        }
        return r;
    }

    template <typename P>
    class matrix_kernels_host_csritsv : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(matrix_kernels_host_csritsv, HostOnly<RealTypes>, TypedName);

TYPED_TEST(matrix_kernels_host_csritsv, transposed_solves_use_the_transpose_of_the_lower_triangle)
{
    using T = typename TypeParam::value_type;

    const Csr<T>         L  = lower_part(band<T>(10, 2));
    const std::vector<T> Lt = dense(transposed(L));
    const std::vector<T> x  = rhs<T>(L.n);

    for(auto trans : {host_sparse_operation_transpose, host_sparse_operation_conjugate_transpose})
    {
        for(auto diag : {host_sparse_diag_type_non_unit, host_sparse_diag_type_unit})
        {
            const bool unit = diag == host_sparse_diag_type_unit;
            SCOPED_TRACE(std::string(trans == host_sparse_operation_transpose
                                         ? "transpose, "
                                         : "conjugate transpose, ")
                         + (unit ? "unit diagonal" : "diagonal of the matrix"));

            ItsvCall<T> c;
            c.trans = trans;
            c.diag  = diag;

            Itsv<T> r = itsv(L, c, x, std::vector<T>(L.n, static_cast<T>(0)));
            ASSERT_IN_LOOP(r.status);
            EXPECT_EQ(r.zero_pivot, -1);
            expect_near(r.y, upper_solve(Lt, L.n, x, unit), tol<T>());
        }
    }
}

// With the triangular matrix type the diagonal is the last entry of a lower and
// the first entry of an upper row. A unit lower matrix stores no diagonal.
TYPED_TEST(matrix_kernels_host_csritsv, triangular_matrix_type_solves_with_the_stored_triangle)
{
    using T = typename TypeParam::value_type;

    const Csr<T>         A = band<T>(10, 2);
    const std::vector<T> x = rhs<T>(A.n);

    struct Case
    {
        std::string           name;
        Csr<T>                matrix;
        host_sparse_fill_mode fill;
        host_sparse_diag_type diag;
        std::vector<T>        expected;
    };
    const Csr<T>            strict = filtered(A, [](int i, int j, T) { return j < i; });
    const std::vector<Case> cases  = {{"lower",
                                       lower_part(A),
                                       host_sparse_fill_mode_lower,
                                       host_sparse_diag_type_non_unit,
                                       lower_solve(dense(A), A.n, x, false)},
                                      {"upper",
                                       upper_part(A),
                                       host_sparse_fill_mode_upper,
                                       host_sparse_diag_type_non_unit,
                                       upper_solve(dense(A), A.n, x, false)},
                                      {"strictly lower with unit diagonal",
                                       strict,
                                       host_sparse_fill_mode_lower,
                                       host_sparse_diag_type_unit,
                                       lower_solve(dense(A), A.n, x, true)}};

    for(const Case& t : cases)
    {
        SCOPED_TRACE(t.name);
        ItsvCall<T> c;
        c.fill = t.fill;
        c.diag = t.diag;
        c.type = host_sparse_matrix_type_triangular;

        Itsv<T> r = itsv(t.matrix, c, x, std::vector<T>(A.n, static_cast<T>(0)));
        ASSERT_IN_LOOP(r.status);
        EXPECT_EQ(r.zero_pivot, -1);
        expect_near(r.y, t.expected, tol<T>());
    }
}

// A missing or zero diagonal entry is reported as the first such row, the solve
// returns true and leaves y alone
TYPED_TEST(matrix_kernels_host_csritsv, zero_pivots_are_reported_and_leave_the_solution_alone)
{
    using T = typename TypeParam::value_type;

    const Csr<T>         A = band<T>(10, 2);
    const std::vector<T> x = rhs<T>(A.n), untouched(A.n, static_cast<T>(-7));

    auto without_diagonal = [](const Csr<T>& M, std::vector<int> rows) {
        return filtered(M, [rows](int i, int j, T) {
            return i != j || std::find(rows.begin(), rows.end(), i) == rows.end();
        });
    };

    struct Case
    {
        std::string             name;
        Csr<T>                  matrix;
        host_sparse_fill_mode   fill;
        host_sparse_matrix_type type;
        int                     pivot;
    };
    const std::vector<Case> cases
        = {{"general lower without diagonal in rows 6 and 3",
            without_diagonal(lower_part(A), {6, 3}),
            host_sparse_fill_mode_lower,
            host_sparse_matrix_type_general,
            3},
           {"general upper without diagonal in rows 5 and 2",
            without_diagonal(upper_part(A), {5, 2}),
            host_sparse_fill_mode_upper,
            host_sparse_matrix_type_general,
            2},
           {"triangular lower without diagonal in row 4",
            without_diagonal(lower_part(A), {4}),
            host_sparse_fill_mode_lower,
            host_sparse_matrix_type_triangular,
            4},
           {"triangular upper without diagonal in row 1",
            without_diagonal(upper_part(A), {1}),
            host_sparse_fill_mode_upper,
            host_sparse_matrix_type_triangular,
            1},
           {"general lower with a zero on the diagonal of row 3",
            transformed(lower_part(A),
                        [](int i, int j, T v) { return i == j && i == 3 ? static_cast<T>(0) : v; }),
            host_sparse_fill_mode_lower,
            host_sparse_matrix_type_general,
            3}};

    for(const Case& t : cases)
    {
        SCOPED_TRACE(t.name);
        ItsvCall<T> c;
        c.fill = t.fill;
        c.type = t.type;

        Itsv<T> r = itsv(t.matrix, c, x, untouched);
        EXPECT_TRUE(r.status);
        EXPECT_EQ(r.zero_pivot, t.pivot);
        EXPECT_EQ(r.y, untouched);
    }
}

// Without entries the unit triangle is the identity, so y = alpha x after one
// iteration. Without a diagonal nothing is solved.
TYPED_TEST(matrix_kernels_host_csritsv, matrix_without_entries)
{
    using T = typename TypeParam::value_type;

    Csr<T> E(4);
    for(int i = 0; i < E.n; ++i)
    {
        E.end_row();
    }
    const std::vector<T> x = {1, -2, 3, -4}, untouched(4, static_cast<T>(-7));

    ItsvCall<T> c;
    c.alpha    = 2;
    c.max_iter = 10;

    c.diag       = host_sparse_diag_type_unit;
    Itsv<T> unit = itsv(E, c, x, untouched);
    EXPECT_TRUE(unit.status);
    EXPECT_EQ(unit.iterations, 1);
    EXPECT_EQ(unit.zero_pivot, -1);
    EXPECT_EQ(unit.y, (std::vector<T>{2, -4, 6, -8}));

    // The iteration count is also the limit, so it cannot be preset. As with a zero
    // pivot, the solve returns before it iterates and leaves the count at the limit.
    c.diag           = host_sparse_diag_type_non_unit;
    Itsv<T> non_unit = itsv(E, c, x, untouched);
    EXPECT_TRUE(non_unit.status);
    EXPECT_EQ(non_unit.iterations, 10);
    EXPECT_EQ(non_unit.y, untouched);

    // Every row lacks its diagonal, so the first zero pivot is row 0
    KNOWN_BUG("without entries the non-unit solve returns before it looks for zero pivots and "
              "reports none");
    EXPECT_EQ(non_unit.zero_pivot, 0);
}

// Integer entries and a unit diagonal keep every iterate exact. The stored
// diagonal is ignored.
TYPED_TEST(matrix_kernels_host_csritsv, records_the_largest_change_of_every_iteration)
{
    using T = typename TypeParam::value_type;

    const int n = 6;
    Csr<T>    L(n);
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j <= i; ++j)
        {
            L.add(j, static_cast<T>(j == i ? 9 : 1 + (i + j) % 3));
        }
        L.end_row();
    }
    const std::vector<T> x = {1, -2, 3, -1, 2, -3};

    std::vector<T> A = dense(L);
    for(int i = 0; i < n; ++i)
    {
        A[i * n + i] = 1;
    }
    const T         tol      = 0;
    const Jacobi<T> expected = jacobi(A, n, x, 20, tol);
    ASSERT_LT(expected.iterations, 20);

    ItsvCall<T> c;
    c.diag     = host_sparse_diag_type_unit;
    c.max_iter = 20;
    c.tol      = &tol;

    Itsv<T> r = itsv(L, c, x, std::vector<T>(n, static_cast<T>(0)));
    ASSERT_TRUE(r.status);
    EXPECT_EQ(r.iterations, expected.iterations);
    EXPECT_EQ(r.history, expected.history);
    EXPECT_EQ(r.y, expected.y);
}

// The iteration stops when the largest residual is at most the tolerance. In
// the first transposed iteration the residual of the second row is 5, but its
// update only 0.5.
TYPED_TEST(matrix_kernels_host_csritsv, the_tolerance_bounds_the_largest_residual)
{
    using T = typename TypeParam::value_type;

    Csr<T> L(3);
    L.add(0, 10);
    L.end_row();
    L.add(0, 10);
    L.add(1, 10);
    L.end_row();
    L.add(2, 10);
    L.end_row();

    const T     bound = 1;
    ItsvCall<T> c;
    c.tol = &bound;

    {
        SCOPED_TRACE("lower");
        const std::vector<T> x        = {5, 0, 0};
        const Jacobi<T>      expected = jacobi(dense(L), L.n, x, c.max_iter, bound);
        EXPECT_EQ(expected.iterations, 3);

        Itsv<T> r = itsv(L, c, x, std::vector<T>(L.n, static_cast<T>(0)));
        ASSERT_TRUE(r.status);
        EXPECT_EQ(r.iterations, expected.iterations);
        expect_near(r.y, lower_solve(dense(L), L.n, x, false), tol<T>());
    }

    KNOWN_BUG("the transposed solve takes the largest update of the previous rows as the residual");

    {
        SCOPED_TRACE("transposed");
        const std::vector<T> x        = {0, 5, 0};
        const std::vector<T> Lt       = dense(transposed(L));
        const Jacobi<T>      expected = jacobi(Lt, L.n, x, c.max_iter, bound);
        EXPECT_EQ(expected.iterations, 3);

        c.trans   = host_sparse_operation_transpose;
        Itsv<T> r = itsv(L, c, x, std::vector<T>(L.n, static_cast<T>(0)));
        ASSERT_TRUE(r.status);
        EXPECT_EQ(r.iterations, expected.iterations);
        expect_near(r.y, upper_solve(Lt, L.n, x, false), tol<T>());
    }
}

// The 2D Laplace stencil is the only stencil, and exists only on the host
namespace
{
    // 5-point Laplacian with Dirichlet boundary on a size x size grid, row major
    template <typename T>
    Csr<T> laplace_2d(int size)
    {
        Csr<T> A(size * size);
        for(int i = 0; i < size; ++i)
        {
            for(int j = 0; j < size; ++j)
            {
                if(i > 0)
                {
                    A.add((i - 1) * size + j, -1);
                }
                if(j > 0)
                {
                    A.add(i * size + j - 1, -1);
                }
                A.add(i * size + j, 4);
                if(j < size - 1)
                {
                    A.add(i * size + j + 1, -1);
                }
                if(i < size - 1)
                {
                    A.add((i + 1) * size + j, -1);
                }
                A.end_row();
            }
        }
        return A;
    }

    rocalution::Rocalution_Backend_Descriptor with_threads(int threads, int threshold)
    {
        rocalution::Rocalution_Backend_Descriptor d = descriptor();
        d.OpenMP_threads                            = threads;
        d.OpenMP_threshold                          = threshold;
        return d;
    }

    // Sizes 2 and 3 have no interior and no boundary layer points respectively
    const std::vector<int> grid_sizes = {2, 3, 7};

    template <typename P>
    class matrix_kernels_stencil : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(matrix_kernels_stencil, HostOnly<RealTypes>, TypedName);

TYPED_TEST(matrix_kernels_stencil, laplace_2d_apply_matches_the_five_point_matrix)
{
    using T = typename TypeParam::value_type;
    DescriptorGuard guard;

    for(int size : grid_sizes)
    {
        SCOPED_TRACE("grid " + std::to_string(size));
        const Csr<T> ref = laplace_2d<T>(size);

        rocalution::HostStencilLaplace2D<T> S(with_threads(1, 0));
        S.SetGrid(size);
        EXPECT_EQ(S.GetM(), ref.n);
        EXPECT_EQ(S.GetN(), ref.n);
        EXPECT_EQ(S.GetNDim(), 2);
        EXPECT_EQ(S.GetNnz(), 5);
        EXPECT_EQ(S.GetStencilId(), static_cast<unsigned int>(rocalution::Laplace2D));

        const std::vector<T> x   = small_integers<T>(ref.n);
        auto                 out = host_vector(std::vector<T>(ref.n, static_cast<T>(-7)));
        S.Apply(*host_vector(x), out.get());
        EXPECT_EQ(values_of(*out), ref.apply(x));
    }
}

TYPED_TEST(matrix_kernels_stencil, laplace_2d_apply_add_adds_the_scaled_product)
{
    using T = typename TypeParam::value_type;
    DescriptorGuard guard;

    for(T scalar : {static_cast<T>(1), static_cast<T>(0.5)})
    {
        if(scalar != static_cast<T>(1))
        {
            KNOWN_BUG("ApplyAdd ignores the scalar");
        }

        for(int size : grid_sizes)
        {
            SCOPED_TRACE("grid " + std::to_string(size) + ", scalar " + std::to_string(scalar));
            const Csr<T> ref = laplace_2d<T>(size);

            rocalution::HostStencilLaplace2D<T> S(with_threads(1, 0));
            S.SetGrid(size);

            const std::vector<T> x   = small_integers<T>(ref.n);
            auto                 out = host_vector(ramp<T>(ref.n));
            S.ApplyAdd(*host_vector(x), scalar, out.get());
            EXPECT_EQ(values_of(*out), apply_add(ref, x, scalar, ramp<T>(ref.n)));
        }
    }
}

// Above the OpenMP threshold the grid runs on several threads
TYPED_TEST(matrix_kernels_stencil, laplace_2d_on_several_threads)
{
    using T = typename TypeParam::value_type;
    DescriptorGuard guard;

    const int    size = 300;
    const Csr<T> ref  = laplace_2d<T>(size);
    const auto   d    = with_threads(8, 1000);
    ASSERT_GT(ref.n, d.OpenMP_threshold);

    rocalution::HostStencilLaplace2D<T> S(d);
    S.SetGrid(size);

    const std::vector<T> x = small_integers<T>(ref.n);
    {
        SCOPED_TRACE("Apply");
        auto out = host_vector(std::vector<T>(ref.n, static_cast<T>(-7)));
        S.Apply(*host_vector(x), out.get());
        EXPECT_EQ(values_of(*out), ref.apply(x));
    }
    {
        SCOPED_TRACE("ApplyAdd");
        auto out = host_vector(ramp<T>(ref.n));
        S.ApplyAdd(*host_vector(x), static_cast<T>(1), out.get());
        EXPECT_EQ(values_of(*out), apply_add(ref, x, static_cast<T>(1), ramp<T>(ref.n)));
    }
}

namespace
{
    template <typename T>
    std::vector<T> values_of(const rocalution::LocalVector<T>& v)
    {
        std::vector<T> values(v.GetSize());
        v.CopyToData(values.data());
        return values;
    }

    template <typename P>
    class matrix_kernels_local_stencil : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(matrix_kernels_local_stencil, HostOnly<RealTypes>, TypedName);

// LocalStencil keeps the stencil on the host
TYPED_TEST(matrix_kernels_local_stencil, apply_and_apply_add_on_the_host)
{
    using T = typename TypeParam::value_type;

    const int            size = 4;
    const Csr<T>         ref  = laplace_2d<T>(size);
    const std::vector<T> x = small_integers<T>(ref.n), y = ramp<T>(ref.n);

    rocalution::LocalStencil<T> S(rocalution::Laplace2D);
    S.SetGrid(size);
    EXPECT_EQ(S.GetM(), ref.n);
    EXPECT_EQ(S.GetN(), ref.n);
    EXPECT_EQ(S.GetNDim(), 2);
    EXPECT_EQ(S.GetNnz(), 5);

    rocalution::LocalVector<T> in, out;
    in.Allocate("in", ref.n);
    in.CopyFromData(x.data());
    out.Allocate("out", ref.n);

    S.Apply(in, &out);
    EXPECT_EQ(values_of(out), ref.apply(x));

    out.Zeros();
    S.ApplyAdd(in, static_cast<T>(1), &out);
    EXPECT_EQ(values_of(out), ref.apply(x));

    KNOWN_BUG("LocalStencil::ApplyAdd calls Apply, which overwrites the output");

    out.CopyFromData(y.data());
    S.ApplyAdd(in, static_cast<T>(1), &out);
    EXPECT_EQ(values_of(out), apply_add(ref, x, static_cast<T>(1), y));

    KNOWN_BUG("ApplyAdd ignores the scalar");

    out.CopyFromData(y.data());
    S.ApplyAdd(in, static_cast<T>(0.5), &out);
    EXPECT_EQ(values_of(out), apply_add(ref, x, static_cast<T>(0.5), y));
}
