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

#include "unit_test_common.hpp"
#include "unit_test_types.hpp"

#include "base/host/host_conversion.hpp"
#include "base/matrix_formats_ind.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;

namespace
{
    // Converts ref from CSR to format f and back, both through the host matrix
    // classes. BCSR keeps the zeros of its blocks.
    template <typename T>
    void expect_round_trip(Format f, Csr<T> ref)
    {
        SCOPED_TRACE(format_name(f) + " with block dimension " + std::to_string(f.blockdim));

        auto A = make_matrix(Backend::Host, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);
        ASSERT_EQ(A->GetMatFormat(), f.id);

        rocalution::HostMatrixCSR<T> back(descriptor());
        ASSERT_TRUE(back.ConvertFrom(*A));
        expect_csr_eq(back, ref, f.id == rocalution::BCSR);
    }

    template <typename T>
    void test_round_trips()
    {
        const std::vector<Format> formats = {Format{rocalution::COO, 1},
                                             Format{rocalution::ELL, 1},
                                             Format{rocalution::DIA, 1},
                                             Format{rocalution::HYB, 1},
                                             Format{rocalution::MCSR, 1},
                                             Format{rocalution::BCSR, 2},
                                             Format{rocalution::BCSR, 3},
                                             Format{rocalution::DENSE, 1}};

        for(Format f : formats)
        {
            {
                SCOPED_TRACE("tridiagonal");
                expect_round_trip(f, Csr<T>(Tridiagonal<T>(12)));
            }
            {
                SCOPED_TRACE("banded with gaps");
                expect_round_trip(f, banded_with_gaps<T>(12));
            }

            // ELL and DIA reject the arrow, see ell_and_dia_reject_oversized_matrices
            if(f.id != rocalution::ELL && f.id != rocalution::DIA)
            {
                SCOPED_TRACE("arrow");
                expect_round_trip(f, arrow<T>(12));
            }
        }
    }

    // n x m with three diagonals, so that ELL and DIA accept it
    template <typename T>
    Csr<T> rectangular(int n, int m)
    {
        Csr<T> A(n, m);
        for(int i = 0; i < n; ++i)
        {
            for(int d : {-5, 0, 2})
            {
                const int j = i + d;
                if(j >= 0 && j < m)
                {
                    A.add(j, static_cast<T>(i + 2 * j + 1));
                }
            }
            A.end_row();
        }
        return A;
    }

    // Output arrays that a conversion allocates
    template <typename T>
    struct OwnedCsr
    {
        ~OwnedCsr()
        {
            rocalution::free_host(&mat.row_offset);
            rocalution::free_host(&mat.col);
            rocalution::free_host(&mat.val);
        }

        rocalution::MatrixCSR<T, int, PtrType> mat{};
    };

    template <typename T>
    struct OwnedHyb
    {
        ~OwnedHyb()
        {
            rocalution::free_host(&mat.ELL.col);
            rocalution::free_host(&mat.ELL.val);
            rocalution::free_host(&mat.COO.row);
            rocalution::free_host(&mat.COO.col);
            rocalution::free_host(&mat.COO.val);
        }

        rocalution::MatrixHYB<T, int> mat{};
    };

    // Non-owning view of a reference matrix as the raw CSR struct
    template <typename T>
    rocalution::MatrixCSR<T, int, PtrType> csr_view(Csr<T>& ref)
    {
        rocalution::MatrixCSR<T, int, PtrType> src{};
        src.row_offset = ref.row_offset.data();
        src.col        = ref.col.data();
        src.val        = ref.val.data();
        return src;
    }

    // COO output of a conversion. The conversion allocates the arrays and asserts
    // that they start out empty.
    template <typename T>
    struct OwnedCoo
    {
        ~OwnedCoo()
        {
            rocalution::free_host(&mat.row);
            rocalution::free_host(&mat.col);
            rocalution::free_host(&mat.val);
        }

        rocalution::MatrixCOO<T, int> mat{};
    };

    template <typename P>
    class host_conversion : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(host_conversion, HostOnly<RealTypes>, TypedName);

// Free function on raw format structs, without any matrix object
TYPED_TEST(host_conversion, csr_to_coo)
{
    using T = typename TypeParam::value_type;
    Tridiagonal<T> ref(12);
    OwnedCoo<T>    dst;

    ASSERT_TRUE(rocalution::csr_to_coo(
        descriptor().OpenMP_threads, ref.nnz(), ref.n, ref.n, csr_view(ref), &dst.mat));

    for(int i = 0; i < ref.n; ++i)
    {
        for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
        {
            EXPECT_EQ(dst.mat.row[j], i) << "at entry " << j;
            EXPECT_EQ(dst.mat.col[j], ref.col[j]) << "at entry " << j;
            EXPECT_EQ(dst.mat.val[j], ref.val[j]) << "at entry " << j;
        }
    }
}

TYPED_TEST(host_conversion, round_trip_through_every_format)
{
    test_round_trips<typename TypeParam::value_type>();
}

// MCSR holds only square matrices, see mcsr_requires_a_square_matrix
TYPED_TEST(host_conversion, round_trip_of_wide_and_tall_matrices)
{
    using T = typename TypeParam::value_type;
    const std::vector<std::pair<std::string, Csr<T>>> matrices
        = {{"wide", rectangular<T>(6, 12)}, {"tall", rectangular<T>(12, 6)}};
    const std::vector<Format> formats = {Format{rocalution::COO, 1},
                                         Format{rocalution::ELL, 1},
                                         Format{rocalution::HYB, 1},
                                         Format{rocalution::BCSR, 2},
                                         Format{rocalution::BCSR, 3},
                                         Format{rocalution::DENSE, 1}};

    for(Format f : formats)
    {
        for(const auto& matrix : matrices)
        {
            SCOPED_TRACE(matrix.first);
            expect_round_trip(f, matrix.second);
        }
    }

    {
        SCOPED_TRACE("wide");
        expect_round_trip(Format{rocalution::DIA, 1}, matrices[0].second);
    }

    // DIA stores max(n, m) entries per diagonal, which SetDataPtrDIA, LeaveDataPtrDIA
    // and the accelerator rely on. The matrices have 3 diagonals.
    KNOWN_BUG("csr_to_dia sizes the values by the smaller dimension, and a tall matrix writes "
              "past them, which can end the process");
    expect_in_child([&] {
        for(const auto& matrix : matrices)
        {
            SCOPED_TRACE(matrix.first);
            auto A
                = make_matrix(Backend::Host, Format{rocalution::DIA, 1}, *matrix.second.host_csr());
            ASSERT_NE(A, nullptr);
            EXPECT_EQ(A->GetNnz(), 3 * std::max(matrix.second.n, matrix.second.m));
        }
        SCOPED_TRACE("tall");
        expect_round_trip(Format{rocalution::DIA, 1}, matrices[1].second);
    });
}

TYPED_TEST(host_conversion, conversion_from_the_same_format_copies_the_matrix)
{
    using T                           = typename TypeParam::value_type;
    const std::vector<Format> formats = {Format{rocalution::COO, 1},
                                         Format{rocalution::ELL, 1},
                                         Format{rocalution::DIA, 1},
                                         Format{rocalution::HYB, 1},
                                         Format{rocalution::MCSR, 1},
                                         Format{rocalution::BCSR, 2},
                                         Format{rocalution::BCSR, 3},
                                         Format{rocalution::DENSE, 1}};
    const Csr<T>              ref     = banded_with_gaps<T>(12);

    for(Format f : formats)
    {
        SCOPED_TRACE(format_name(f) + " with block dimension " + std::to_string(f.blockdim));
        auto A = make_matrix(Backend::Host, f, *ref.host_csr());
        ASSERT_IN_LOOP(A != nullptr);

        auto B = empty_matrix<T>(Backend::Host, f);
        ASSERT_IN_LOOP(B->ConvertFrom(*A));
        EXPECT_EQ(B->GetMatFormat(), f.id);
        EXPECT_EQ(B->GetNnz(), A->GetNnz());
        expect_csr_eq(*host_csr_of(*B), ref, f.id == rocalution::BCSR);
    }
}

// A 4 x 6 matrix with two non-zeros per row
namespace
{
    Csr<double> wide()
    {
        Csr<double> A(4, 6);
        for(int i = 0; i < 4; ++i)
        {
            A.add(i, 1.0);
            A.add(i + 2, 2.0);
            A.end_row();
        }
        return A;
    }
}

TEST(host_conversion, mcsr_requires_a_square_matrix)
{
    Csr<double>                         A = wide();
    rocalution::MatrixMCSR<double, int> dst{};

    EXPECT_FALSE(
        rocalution::csr_to_mcsr(descriptor().OpenMP_threads, A.nnz(), A.n, A.m, csr_view(A), &dst));
    EXPECT_EQ(dst.row_offset, nullptr);

    OwnedCsr<double> back;
    EXPECT_FALSE(
        rocalution::mcsr_to_csr(descriptor().OpenMP_threads, A.nnz(), A.n, A.m, dst, &back.mat));
    EXPECT_EQ(back.mat.row_offset, nullptr);
}

TEST(host_conversion, mcsr_requires_every_diagonal_entry)
{
    Tridiagonal<double> full(12);
    Csr<double>         A(12);
    for(int i = 0; i < A.n; ++i)
    {
        for(PtrType j = full.row_offset[i]; j < full.row_offset[i + 1]; ++j)
        {
            if(i != 5 || full.col[j] != 5)
            {
                A.add(full.col[j], full.val[j]);
            }
        }
        A.end_row();
    }

    std::unique_ptr<BaseMatrix<double>> out;
    EXPECT_FALSE(try_make_matrix(Backend::Host, Format{rocalution::MCSR, 1}, *A.host_csr(), &out));
}

TEST(host_conversion, bcsr_requires_dimensions_divisible_by_the_block_dimension)
{
    // 12 rows are not divisible by 5
    Tridiagonal<double>                 square(12);
    std::unique_ptr<BaseMatrix<double>> out;
    EXPECT_FALSE(
        try_make_matrix(Backend::Host, Format{rocalution::BCSR, 5}, *square.host_csr(), &out));

    // 4 rows are divisible by 4, 6 columns are not
    Csr<double>                         A = wide();
    rocalution::MatrixBCSR<double, int> dst{};
    dst.blockdim = 4;

    EXPECT_FALSE(
        rocalution::csr_to_bcsr(descriptor().OpenMP_threads, A.nnz(), A.n, A.m, csr_view(A), &dst));
    EXPECT_EQ(dst.row_offset, nullptr);
}

TYPED_TEST(host_conversion, coo_to_csr_sorts_the_columns_of_each_row)
{
    using T = typename TypeParam::value_type;

    std::vector<int> row = {0, 0, 0, 1, 1};
    std::vector<int> col = {2, 0, 1, 1, 0};
    std::vector<T>   val = {1, 2, 3, 4, 5};

    rocalution::MatrixCOO<T, int> src{row.data(), col.data(), val.data()};
    OwnedCsr<T>                   dst;

    ASSERT_TRUE(rocalution::coo_to_csr(descriptor().OpenMP_threads, 5, 2, 3, src, &dst.mat));

    EXPECT_EQ(std::vector<PtrType>(dst.mat.row_offset, dst.mat.row_offset + 3),
              (std::vector<PtrType>{0, 3, 5}));
    EXPECT_EQ(std::vector<int>(dst.mat.col, dst.mat.col + 5), (std::vector<int>{0, 1, 2, 0, 1}));
    EXPECT_EQ(std::vector<T>(dst.mat.val, dst.mat.val + 5), (std::vector<T>{2, 3, 1, 5, 4}));
}

// ELL entries are stored if their column is inside the matrix. The padding of
// short rows has column -1.
namespace
{
    template <typename T>
    struct Ell2x2
    {
        Ell2x2()
        {
            set(0, 0, 0, 1);
            set(0, 1, 5, 9);
            set(1, 0, 0, 2);
            set(1, 1, -1, 0);
        }

        void set(int i, int n, int j, T v)
        {
            col[ELL_IND(i, n, 2, 2)] = j;
            val[ELL_IND(i, n, 2, 2)] = v;
        }

        std::vector<int> col = std::vector<int>(4);
        std::vector<T>   val = std::vector<T>(4);
    };
}

// Not only the -1 padding: every column outside the matrix is dropped
TYPED_TEST(host_conversion, ell_to_csr_skips_columns_outside_the_matrix)
{
    using T = typename TypeParam::value_type;

    Ell2x2<T>                     ell;
    rocalution::MatrixELL<T, int> src{2, ell.col.data(), ell.val.data()};
    OwnedCsr<T>                   dst;
    int64_t                       nnz_csr = 0;

    ASSERT_TRUE(
        rocalution::ell_to_csr(descriptor().OpenMP_threads, 4, 2, 2, src, &dst.mat, &nnz_csr));

    ASSERT_EQ(nnz_csr, 2);
    EXPECT_EQ(std::vector<PtrType>(dst.mat.row_offset, dst.mat.row_offset + 3),
              (std::vector<PtrType>{0, 1, 2}));
    EXPECT_EQ(std::vector<int>(dst.mat.col, dst.mat.col + 2), (std::vector<int>{0, 0}));
    EXPECT_EQ(std::vector<T>(dst.mat.val, dst.mat.val + 2), (std::vector<T>{1, 2}));
}

// As ell_to_csr, every ELL column outside the matrix is dropped
TYPED_TEST(host_conversion, hyb_to_csr_skips_ell_columns_outside_the_matrix)
{
    using T = typename TypeParam::value_type;

    Ell2x2<T>        ell;
    std::vector<int> coo_row = {1};
    std::vector<int> coo_col = {1};
    std::vector<T>   coo_val = {3};

    rocalution::MatrixHYB<T, int> src{};
    src.ELL     = {2, ell.col.data(), ell.val.data()};
    src.COO     = {coo_row.data(), coo_col.data(), coo_val.data()};
    int64_t nnz = 0;

    OwnedCsr<T> dst;
    ASSERT_TRUE(
        rocalution::hyb_to_csr(descriptor().OpenMP_threads, 5, 2, 2, 4, 1, src, &dst.mat, &nnz));

    ASSERT_EQ(nnz, 3);
    EXPECT_EQ(std::vector<PtrType>(dst.mat.row_offset, dst.mat.row_offset + 3),
              (std::vector<PtrType>{0, 1, 3}));
    EXPECT_EQ(std::vector<int>(dst.mat.col, dst.mat.col + 3), (std::vector<int>{0, 0, 1}));
    EXPECT_EQ(std::vector<T>(dst.mat.val, dst.mat.val + 3), (std::vector<T>{1, 2, 3}));
}

// An ELL width set by the caller is kept; the rest of each row goes to COO
TYPED_TEST(host_conversion, csr_to_hyb_keeps_a_given_ell_width)
{
    using T = typename TypeParam::value_type;
    Tridiagonal<T> ref(12);
    OwnedHyb<T>    hyb;
    hyb.mat.ELL.max_row = 1;
    int64_t nnz_hyb = 0, nnz_ell = 0, nnz_coo = 0;

    ASSERT_TRUE(rocalution::csr_to_hyb(descriptor().OpenMP_threads,
                                       ref.nnz(),
                                       ref.n,
                                       ref.n,
                                       csr_view(ref),
                                       &hyb.mat,
                                       &nnz_hyb,
                                       &nnz_ell,
                                       &nnz_coo));

    EXPECT_EQ(hyb.mat.ELL.max_row, 1);
    EXPECT_EQ(nnz_ell, ref.n);
    EXPECT_EQ(nnz_coo, ref.nnz() - ref.n);

    OwnedCsr<T> back;
    int64_t     nnz_csr = 0;
    ASSERT_TRUE(rocalution::hyb_to_csr(descriptor().OpenMP_threads,
                                       nnz_hyb,
                                       ref.n,
                                       ref.n,
                                       nnz_ell,
                                       nnz_coo,
                                       hyb.mat,
                                       &back.mat,
                                       &nnz_csr));

    ASSERT_EQ(nnz_csr, ref.nnz());
    EXPECT_EQ(std::vector<PtrType>(back.mat.row_offset, back.mat.row_offset + ref.n + 1),
              ref.row_offset);
    EXPECT_EQ(std::vector<int>(back.mat.col, back.mat.col + nnz_csr), ref.col);
    EXPECT_EQ(std::vector<T>(back.mat.val, back.mat.val + nnz_csr), ref.val);
}

// Preconditions of the conversion functions are asserts, which stay enabled in
// release builds
TEST(host_conversion_DeathTest, csr_to_dia_requires_non_zeros)
{
    Tridiagonal<double>                ref(12);
    rocalution::MatrixDIA<double, int> dst{};
    int64_t                            nnz_dia = 0;

    EXPECT_ASSERT_DEATH(
        rocalution::csr_to_dia(
            descriptor().OpenMP_threads, 0, ref.n, ref.n, csr_view(ref), &dst, &nnz_dia),
        "nnz > 0");
}
