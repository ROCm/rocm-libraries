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

// The MCSR backend matrices: copies into allocated matrices, empty matrices, conversions
// and the host ILU(0). The SpMV kernels are in unit_test_matrix_kernels.cpp.

#include <memory>
#include <string>
#include <vector>

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

using namespace rocalution_unit;

namespace
{
    const Format mcsr_format{rocalution::MCSR, 1};

    template <typename P>
    class matrix_mcsr : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        static Backend backend()
        {
            return P::backend;
        }

        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_matrix(backend(), mcsr_format, *ref.host_csr());
        }

        // An nrow x nrow matrix without entries, set from data pointers
        std::unique_ptr<BaseMatrix<T>> without_entries(int nrow) const
        {
            auto source = empty_matrix<T>(backend(), mcsr_format);
            source->AllocateMCSR(0, nrow, nrow);

            int* row_offset = nullptr;
            int* col        = nullptr;
            T*   val        = nullptr;
            source->LeaveDataPtrMCSR(&row_offset, &col, &val);

            auto A = empty_matrix<T>(backend(), mcsr_format);
            A->SetDataPtrMCSR(&row_offset, &col, &val, 0, nrow, nrow);
            return A;
        }
    };

    // Diagonal 10 + i, with off-diagonals that scale with i + j
    template <typename T>
    Csr<T> pentadiagonal(int n, double scale)
    {
        Csr<T> A(n);
        for(int i = 0; i < n; ++i)
        {
            for(int j = i - 2; j <= i + 2; ++j)
            {
                if(j >= 0 && j < n)
                {
                    A.add(j, static_cast<T>(j == i ? 10.0 + i : -scale * (1 + (i + j) % 3)));
                }
            }
            A.end_row();
        }
        return A;
    }

    // Rows 1 and 3 depend on row 0, whose upper part reaches columns outside their
    // patterns: (1, 2) and (3, 1) would be fill-in
    template <typename T>
    Csr<T> matrix_with_fill_in()
    {
        Csr<T> A(4);
        A.add(0, 4.0);
        A.add(1, 1.0);
        A.add(2, 1.0);
        A.add(3, -1.0);
        A.end_row();
        A.add(0, 1.0);
        A.add(1, 5.0);
        A.add(3, 2.0);
        A.end_row();
        A.add(1, -1.0);
        A.add(2, 6.0);
        A.end_row();
        A.add(0, 2.0);
        A.add(2, 1.0);
        A.add(3, 7.0);
        A.end_row();
        return A;
    }
}

TYPED_TEST_SUITE(matrix_mcsr, WithBackends<RealTypes>, TypedName);

TYPED_TEST(matrix_mcsr, info_names_the_backend_class)
{
    using T             = typename TypeParam::value_type;
    auto              A = this->make(Tridiagonal<T>(4));
    const std::string name
        = TypeParam::backend == Backend::Host ? "HostMatrixMCSR" : "HIPAcceleratorMatrixMCSR";
    EXPECT_NE(logged([&] { A->Info(); }).find(name), std::string::npos);
}

TYPED_TEST(matrix_mcsr, converts_from_mcsr)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = pentadiagonal<T>(9, 1.0);
    auto         A   = this->make(ref);
    auto         B   = this->make(Tridiagonal<T>(3));

    ASSERT_TRUE(B->ConvertFrom(*A));
    expect_near(csr_of(*B), ref, 0.0);
}

// Copies into a matrix of the same size reuse its arrays
TYPED_TEST(matrix_mcsr, copies_from_and_to_the_host_into_an_allocated_matrix)
{
    using T            = typename TypeParam::value_type;
    const Csr<T> ref   = pentadiagonal<T>(9, 1.0);
    const Csr<T> other = pentadiagonal<T>(9, 2.0);
    auto         src   = make_matrix(Backend::Host, mcsr_format, *ref.host_csr());

    auto A = this->make(other);
    if(auto accel = dynamic_cast<rocalution::AcceleratorMatrix<T>*>(A.get()))
    {
        accel->CopyFromHost(*dynamic_cast<HostMatrix<T>*>(src.get()));
    }
    else
    {
        A->CopyFrom(*src);
    }
    expect_near(csr_of(*A), ref, 0.0);

    auto dst = make_matrix(Backend::Host, mcsr_format, *other.host_csr());
    if(auto accel = dynamic_cast<rocalution::AcceleratorMatrix<T>*>(A.get()))
    {
        accel->CopyToHost(dynamic_cast<HostMatrix<T>*>(dst.get()));
    }
    else
    {
        A->CopyTo(dst.get());
    }
    expect_near(csr_of(*dst), ref, 0.0);
}

// A matrix without entries adds nothing. The 0 x 0 matrix applies to empty vectors; that it
// returns without an error and keeps the output empty is all there is to check.
TYPED_TEST(matrix_mcsr, matrices_without_entries_apply_as_zero)
{
    using T = typename TypeParam::value_type;
    auto A  = this->without_entries(4);
    EXPECT_EQ(A->GetM(), 4);
    EXPECT_EQ(A->GetN(), 4);
    EXPECT_EQ(A->GetNnz(), 0);

    const std::vector<T> y   = {1.0, -2.0, 3.0, 0.5};
    auto                 out = make_vector(this->backend(), y);
    A->ApplyAdd(*make_vector(this->backend(), std::vector<T>{1.0, 2.0, 3.0, 4.0}),
                static_cast<T>(2),
                out.get());
    EXPECT_EQ(values_of(*out), y);

    auto Z     = this->without_entries(0);
    auto empty = make_vector(this->backend(), std::vector<T>{});
    Z->Apply(*make_vector(this->backend(), std::vector<T>{}), empty.get());
    EXPECT_EQ(empty->GetSize(), 0);
}

// A matrix that was never allocated copies as the 0 x 0 matrix
TYPED_TEST(matrix_mcsr, copy_of_an_unallocated_matrix_is_empty)
{
    using T     = typename TypeParam::value_type;
    auto checks = [] {
        auto src = empty_matrix<T>(Backend::Host, mcsr_format);
        auto dst = empty_matrix<T>(TypeParam::backend, mcsr_format);
        if(auto accel = dynamic_cast<rocalution::AcceleratorMatrix<T>*>(dst.get()))
        {
            accel->CopyFromHost(*dynamic_cast<HostMatrix<T>*>(src.get()));
        }
        else
        {
            dst->CopyFrom(*src);
        }
        EXPECT_EQ(dst->GetM(), 0);
        EXPECT_EQ(dst->GetN(), 0);
        EXPECT_EQ(dst->GetNnz(), 0);
    };

    if(TypeParam::backend == Backend::Accelerator)
    {
        KNOWN_BUG("the HIP MCSR copies read the row offsets of an unallocated source and exit "
                  "with a HIP error");
        expect_in_child(checks);
        return;
    }
    checks();
}

namespace
{
    // ILU0Factorize, LUSolve and ReadFileRSIO of MCSR exist only on the host
    template <typename P>
    class host_matrix_mcsr : public matrix_mcsr<P>
    {
    };
}

TYPED_TEST_SUITE(host_matrix_mcsr, HostOnly<RealTypes>, TypedName);

TYPED_TEST(host_matrix_mcsr, ilu0_drops_the_fill_in)
{
    using T          = typename TypeParam::value_type;
    const Csr<T> ref = matrix_with_fill_in<T>();
    auto         A   = this->make(ref);
    ASSERT_TRUE(A->ILU0Factorize());

    const Csr<T> LU = csr_of(*A);
    EXPECT_EQ(LU.row_offset, ref.row_offset);
    EXPECT_EQ(LU.col, ref.col);
    const std::vector<T> D = ilu0_dense(ref);
    expect_near(dense(LU), D, scaled_tol<T>(1e-15));

    const std::vector<T> b   = rhs<T>(ref.n);
    auto                 out = make_vector(this->backend(), std::vector<T>(ref.n));
    A->LUAnalyse();
    ASSERT_TRUE(A->LUSolve(*make_vector(this->backend(), b), out.get()));
    A->LUAnalyseClear();
    expect_near(values_of(*out),
                upper_solve(D, ref.n, lower_solve(D, ref.n, b, true), false),
                scaled_tol<T>(1e-14));
}

TYPED_TEST(host_matrix_mcsr, read_file_rsio_of_a_missing_file_fails)
{
    using T                = typename TypeParam::value_type;
    auto              A    = this->make(Tridiagonal<T>(4));
    const std::string file = temp_file("missing.rsio");
    bool              read = true;
    EXPECT_NE(logged([&] { read = A->ReadFileRSIO(file); }).find("cannot open file"),
              std::string::npos);
    EXPECT_FALSE(read);
    EXPECT_EQ(A->GetM(), 4);
    EXPECT_EQ(A->GetNnz(), Tridiagonal<T>(4).nnz());
}
