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

#include <tuple>

#include "unit_test_common.hpp"
#include "unit_test_types.hpp"

#include "base/host/host_matrix_coo.hpp"

using namespace rocalution_unit;

namespace
{
    using Param = std::tuple<Backend, Format>;

    template <typename T>
    void test_extract_diagonal(Backend b, Format f)
    {
        const T        untouched = static_cast<T>(-7);
        Tridiagonal<T> ref(12);

        auto A = make_matrix(b, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);
        EXPECT_EQ(A->GetMatFormat(), f.id);

        auto d = make_vector(b, std::vector<T>(ref.n, untouched));

        bool native = A->ExtractDiagonal(d.get());

        ASSERT_EQ(native, implements(b, f.id, "ExtractDiagonal"));

        std::vector<T> result = values_of(*d);
        for(int i = 0; i < ref.n; ++i)
        {
            EXPECT_EQ(result[i], native ? ref.diagonal(i) : untouched) << "at row " << i;
        }
    }

    // Every format implements Apply natively on every backend. Apply overwrites
    // the output.
    template <typename T>
    void test_apply(Backend b, Format f)
    {
        Tridiagonal<T> ref(12);
        std::vector<T> x(ref.n);
        for(int i = 0; i < ref.n; ++i)
        {
            x[i] = static_cast<T>(i % 3 + 1);
        }

        auto A = make_matrix(b, f, *ref.host_csr());
        ASSERT_NE(A, nullptr);

        auto in  = make_vector(b, x);
        auto out = make_vector(b, std::vector<T>(ref.n, static_cast<T>(-7)));

        A->Apply(*in, out.get());

        EXPECT_EQ(values_of(*out), ref.apply(x));
    }
}

class backend_matrix : public testing::TestWithParam<Param>
{
protected:
    void SetUp() override
    {
        REQUIRE_BACKEND(std::get<0>(GetParam()));
    }
};

TEST_P(backend_matrix, extract_diagonal_float)
{
    test_extract_diagonal<float>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(backend_matrix, extract_diagonal_double)
{
    test_extract_diagonal<double>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(backend_matrix, apply_float)
{
    test_apply<float>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(backend_matrix, apply_double)
{
    test_apply<double>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(formats,
                         backend_matrix,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::CSR, 1},
                                                          Format{rocalution::COO, 1},
                                                          Format{rocalution::ELL, 1},
                                                          Format{rocalution::DIA, 1},
                                                          Format{rocalution::HYB, 1},
                                                          Format{rocalution::MCSR, 1},
                                                          Format{rocalution::BCSR, 2},
                                                          Format{rocalution::DENSE, 1})),
                         backend_format_name);

// ELL and DIA reject matrices that would need more than 5 times the CSR storage.
// A dense first row gives an ELL width of n and n diagonals at 2 non-zeros per row.
TEST(host_conversion, ell_and_dia_reject_oversized_matrices)
{
    const int   n = 12;
    Csr<double> A(n);
    for(int j = 0; j < n; ++j)
    {
        A.add(j, 1.0);
    }
    A.end_row();
    for(int i = 1; i < n; ++i)
    {
        A.add(i, 1.0);
        A.end_row();
    }
    auto csr = A.host_csr();

    std::unique_ptr<BaseMatrix<double>> out;
    EXPECT_FALSE(try_make_matrix(Backend::Host, Format{rocalution::ELL, 1}, *csr, &out));
    EXPECT_FALSE(try_make_matrix(Backend::Host, Format{rocalution::DIA, 1}, *csr, &out));
    EXPECT_TRUE(try_make_matrix(Backend::Host, Format{rocalution::HYB, 1}, *csr, &out));
}

namespace
{
    template <typename T>
    void test_coo_convert_from_csr()
    {
        SCOPED_TRACE(type_name<T>());
        Tridiagonal<T> ref(12);
        auto           csr = ref.host_csr();

        rocalution::HostMatrixCOO<T> coo(descriptor());
        ASSERT_TRUE(coo.ConvertFrom(*csr));

        EXPECT_EQ(coo.GetM(), ref.n);
        EXPECT_EQ(coo.GetN(), ref.n);
        EXPECT_EQ(coo.GetNnz(), ref.nnz());

        std::vector<int> row(ref.nnz());
        std::vector<int> col(ref.nnz());
        std::vector<T>   val(ref.nnz());
        coo.CopyToCOO(row.data(), col.data(), val.data());

        for(int i = 0; i < ref.n; ++i)
        {
            for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
            {
                EXPECT_EQ(row[j], i) << "at entry " << j;
                EXPECT_EQ(col[j], ref.col[j]) << "at entry " << j;
                EXPECT_EQ(val[j], ref.val[j]) << "at entry " << j;
            }
        }
    }
}

// A single backend class, constructed directly. The test suite is shared with
// unit_test_matrix_formats.cpp, so it is not a typed test suite.
TEST(host_matrix_coo, convert_from_csr)
{
    test_coo_convert_from_csr<float>();
    test_coo_convert_from_csr<double>();
}
