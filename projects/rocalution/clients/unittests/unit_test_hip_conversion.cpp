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

// Conversions between the formats of the accelerator, checked through the
// host conversion back to CSR

#include "unit_test_common.hpp"
#include "unit_test_types.hpp"

#include "base/hip/hip_allocate_free.hpp"
#include "base/hip/hip_conversion.hpp"
#include "base/matrix_formats_ind.hpp"

using namespace rocalution_unit;

namespace
{
    template <typename T>
    T* to_device(const std::vector<T>& host)
    {
        T* device = nullptr;
        rocalution::allocate_hip(static_cast<int64_t>(host.size()), &device);
        rocalution::copy_h2d(static_cast<int64_t>(host.size()), host.data(), device);
        return device;
    }

    template <typename T>
    std::vector<T> from_device(const T* device, int64_t n)
    {
        std::vector<T> host(n);
        rocalution::copy_d2h(n, device, host.data());
        return host;
    }

    // Device arrays of a CSR source and of the HYB result of a conversion, freed
    // also after a failed assertion
    template <typename T>
    struct DeviceCsrAndHyb
    {
        ~DeviceCsrAndHyb()
        {
            rocalution::free_hip(&csr.row_offset);
            rocalution::free_hip(&csr.col);
            rocalution::free_hip(&csr.val);
            rocalution::free_hip(&hyb.ELL.col);
            rocalution::free_hip(&hyb.ELL.val);
            rocalution::free_hip(&hyb.COO.row);
            rocalution::free_hip(&hyb.COO.col);
            rocalution::free_hip(&hyb.COO.val);
        }

        rocalution::MatrixCSR<T, int, PtrType> csr{};
        rocalution::MatrixHYB<T, int>          hyb{};
    };

    template <typename T>
    std::unique_ptr<rocalution::AcceleratorMatrix<T>> accelerator_matrix(Format f)
    {
        return std::unique_ptr<rocalution::AcceleratorMatrix<T>>(
            rocalution::_rocalution_init_base_backend_matrix<T>(descriptor(), f.id, f.blockdim));
    }

    // BCSR keeps the zeros of its blocks
    bool zero_fill(Format f)
    {
        return f.id == rocalution::BCSR;
    }

    class hip_conversion : public testing::TestWithParam<Format>
    {
    protected:
        void SetUp() override
        {
            REQUIRE_BACKEND(Backend::Accelerator);
        }

        Format format() const
        {
            return GetParam();
        }
    };

    template <typename T>
    void test_from_csr(Format f)
    {
        SCOPED_TRACE(type_name<T>());
        for(const auto& matrix : matrices<T>(f))
        {
            SCOPED_TRACE(matrix.first);

            auto A = make_matrix(
                Backend::Accelerator, Format{rocalution::CSR, 1}, *matrix.second.host_csr());
            ASSERT_IN_LOOP(A != nullptr);

            auto X = accelerator_matrix<T>(f);
            ASSERT_IN_LOOP(X->ConvertFrom(*A));
            EXPECT_EQ(X->GetMatFormat(), f.id);
            expect_csr_eq(*host_csr_of(*X), matrix.second, zero_fill(f));
        }
    }

    template <typename T>
    void test_to_csr(Format f)
    {
        SCOPED_TRACE(type_name<T>());
        for(const auto& matrix : matrices<T>(f))
        {
            SCOPED_TRACE(matrix.first);

            auto X = make_matrix(Backend::Accelerator, f, *matrix.second.host_csr());
            ASSERT_IN_LOOP(X != nullptr);

            auto A = accelerator_matrix<T>(Format{rocalution::CSR, 1});
            ASSERT_IN_LOOP(A->ConvertFrom(*X));
            expect_csr_eq(*host_csr_of(*A), matrix.second, zero_fill(f));
        }
    }

    template <typename T>
    void test_unimplemented_conversions(Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T> ref = banded_with_gaps<T>(12);

        if(!implements(Backend::Accelerator, f.id, "ConvertFromCSR"))
        {
            SCOPED_TRACE("from CSR");
            auto A = make_matrix(Backend::Accelerator, Format{rocalution::CSR, 1}, *ref.host_csr());
            ASSERT_NE(A, nullptr);

            auto X = accelerator_matrix<T>(f);
            EXPECT_FALSE(X->ConvertFrom(*A));
            EXPECT_EQ(X->GetMatFormat(), f.id);
            EXPECT_EQ(X->GetNnz(), 0);
            expect_csr_eq(*host_csr_of(*A), ref, false);
        }

        if(!implements(Backend::Accelerator, f.id, "ConvertToCSR"))
        {
            SCOPED_TRACE("to CSR");
            auto X = make_matrix(Backend::Accelerator, f, *ref.host_csr());
            ASSERT_NE(X, nullptr);

            auto A = accelerator_matrix<T>(Format{rocalution::CSR, 1});
            EXPECT_FALSE(A->ConvertFrom(*X));
            EXPECT_EQ(A->GetNnz(), 0);
            expect_csr_eq(*host_csr_of(*X), ref, zero_fill(f));
        }
    }

    template <typename T>
    void test_from_the_same_format(Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const Csr<T> ref = banded_with_gaps<T>(12);

        auto X = make_matrix(Backend::Accelerator, f, *ref.host_csr());
        ASSERT_NE(X, nullptr);

        auto Y = accelerator_matrix<T>(f);
        ASSERT_TRUE(Y->ConvertFrom(*X));
        expect_csr_eq(*host_csr_of(*Y), ref, zero_fill(f));
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

    template <typename T>
    void test_rectangular_from_csr(Format f)
    {
        SCOPED_TRACE(type_name<T>());
        const std::vector<std::pair<std::string, Csr<T>>> cases
            = {{"wide", rectangular<T>(6, 12)}, {"tall", rectangular<T>(12, 6)}};
        for(const auto& matrix : cases)
        {
            SCOPED_TRACE(matrix.first);
            const Csr<T>& ref = matrix.second;

            auto A = make_matrix(Backend::Accelerator, Format{rocalution::CSR, 1}, *ref.host_csr());
            ASSERT_IN_LOOP(A != nullptr);

            auto X = accelerator_matrix<T>(f);
            ASSERT_IN_LOOP(X->ConvertFrom(*A));
            EXPECT_EQ(X->GetM(), ref.n);
            EXPECT_EQ(X->GetN(), ref.m);
            expect_csr_eq(*host_csr_of(*X), ref, zero_fill(f));

            std::vector<T> x(ref.m);
            for(int j = 0; j < ref.m; ++j)
            {
                x[j] = static_cast<T>(j % 4 - 1);
            }
            auto out = make_vector(Backend::Accelerator, std::vector<T>(ref.n, static_cast<T>(-7)));
            X->Apply(*make_vector(Backend::Accelerator, x), out.get());
            EXPECT_EQ(values_of(*out), ref.apply(x));
        }
    }

    template <typename T>
    void test_from_an_empty_csr_matrix(Format f)
    {
        SCOPED_TRACE(type_name<T>());
        auto A = accelerator_matrix<T>(Format{rocalution::CSR, 1});
        A->AllocateCSR(0, 6, 12);

        auto X = accelerator_matrix<T>(f);
        ASSERT_TRUE(X->ConvertFrom(*A));
        EXPECT_EQ(X->GetM(), 6);
        EXPECT_EQ(X->GetN(), 12);
        EXPECT_EQ(X->GetNnz(), f.id == rocalution::DENSE ? 72 : 0);
    }
}

// gtest has no typed value-parameterized tests, so each case runs both value types
TEST_P(hip_conversion, from_csr)
{
    REQUIRE_NATIVE(Backend::Accelerator, format().id, "ConvertFromCSR");

    test_from_csr<float>(format());
    test_from_csr<double>(format());
}

TEST_P(hip_conversion, to_csr)
{
    REQUIRE_NATIVE(Backend::Accelerator, format().id, "ConvertToCSR");

    test_to_csr<float>(format());
    test_to_csr<double>(format());
}

// DIA stores max(n, m) entries per diagonal
TEST_P(hip_conversion, rectangular_matrices_from_csr)
{
    REQUIRE_NATIVE(Backend::Accelerator, format().id, "ConvertFromCSR");

    test_rectangular_from_csr<float>(format());
    test_rectangular_from_csr<double>(format());
}

// A conversion that the accelerator does not implement returns false and leaves
// the destination empty, so that LocalMatrix can convert on the host
TEST_P(hip_conversion, unimplemented_conversions_return_false_and_change_nothing)
{
    if(implements(Backend::Accelerator, format().id, "ConvertFromCSR")
       && implements(Backend::Accelerator, format().id, "ConvertToCSR"))
    {
        GTEST_SKIP() << "every conversion is implemented";
    }

    test_unimplemented_conversions<float>(format());
    test_unimplemented_conversions<double>(format());
}

TEST_P(hip_conversion, from_the_same_format)
{
    test_from_the_same_format<float>(format());
    test_from_the_same_format<double>(format());
}

// Every format, MCSR included, takes an empty matrix. DENSE stores all entries.
TEST_P(hip_conversion, from_an_empty_csr_matrix)
{
    test_from_an_empty_csr_matrix<float>(format());
    test_from_an_empty_csr_matrix<double>(format());
}

INSTANTIATE_TEST_SUITE_P(formats,
                         hip_conversion,
                         testing::Values(Format{rocalution::COO, 1},
                                         Format{rocalution::ELL, 1},
                                         Format{rocalution::DIA, 1},
                                         Format{rocalution::HYB, 1},
                                         Format{rocalution::MCSR, 1},
                                         Format{rocalution::BCSR, 2},
                                         Format{rocalution::BCSR, 3},
                                         Format{rocalution::DENSE, 1}),
                         [](const testing::TestParamInfo<Format>& info) {
                             return format_name(info.param) + std::to_string(info.param.blockdim);
                         });

TEST(hip_conversion_dense, a_dense_matrix_of_zeros_converts_to_csr_without_entries)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    auto empty = accelerator_matrix<double>(Format{rocalution::CSR, 1});
    empty->AllocateCSR(0, 6, 12);
    auto D = accelerator_matrix<double>(Format{rocalution::DENSE, 1});
    ASSERT_TRUE(D->ConvertFrom(*empty));
    ASSERT_EQ(D->GetNnz(), 72);

    auto A = accelerator_matrix<double>(Format{rocalution::CSR, 1});
    ASSERT_TRUE(A->ConvertFrom(*D));
    EXPECT_EQ(A->GetM(), 6);
    EXPECT_EQ(A->GetN(), 12);
    EXPECT_EQ(A->GetNnz(), 0);

    Csr<double> zeros(6, 12);
    for(int i = 0; i < 6; ++i)
    {
        zeros.end_row();
    }
    expect_csr_eq(*host_csr_of(*A), zeros, false);
}

// Free function on raw device arrays: the ELL width is the average row length
// unless the caller presets it
TEST(hip_conversion_hyb, csr_to_hyb_keeps_a_given_ell_width)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    const Tridiagonal<double> ref(12);
    DeviceCsrAndHyb<double>   arrays;
    auto&                     src = arrays.csr;
    auto&                     hyb = arrays.hyb;
    src.row_offset                = to_device(ref.row_offset);
    src.col                       = to_device(ref.col);
    src.val                       = to_device(ref.val);

    hyb.ELL.max_row = 1;
    int64_t nnz_hyb = 0, nnz_ell = 0, nnz_coo = 0;
    ASSERT_TRUE(rocalution::csr_to_hyb_hip(rocalution::_get_backend_descriptor(),
                                           ref.nnz(),
                                           ref.n,
                                           ref.n,
                                           src,
                                           &hyb,
                                           &nnz_hyb,
                                           &nnz_ell,
                                           &nnz_coo));
    EXPECT_EQ(hyb.ELL.max_row, 1);
    EXPECT_EQ(nnz_ell, ref.n);
    EXPECT_EQ(nnz_coo, ref.nnz() - ref.n);
    EXPECT_EQ(nnz_hyb, ref.nnz());

    using Entry = std::tuple<int, int, double>;
    std::vector<Entry>        entries;
    const std::vector<int>    ell_col = from_device(hyb.ELL.col, nnz_ell);
    const std::vector<double> ell_val = from_device(hyb.ELL.val, nnz_ell);
    for(int i = 0; i < ref.n; ++i)
    {
        const int k = ELL_IND(i, 0, ref.n, 1);
        if(ell_col[k] >= 0)
        {
            entries.emplace_back(i, ell_col[k], ell_val[k]);
        }
    }
    const std::vector<int>    coo_row = from_device(hyb.COO.row, nnz_coo);
    const std::vector<int>    coo_col = from_device(hyb.COO.col, nnz_coo);
    const std::vector<double> coo_val = from_device(hyb.COO.val, nnz_coo);
    for(int64_t k = 0; k < nnz_coo; ++k)
    {
        entries.emplace_back(coo_row[k], coo_col[k], coo_val[k]);
    }

    std::vector<Entry> expected;
    for(int i = 0; i < ref.n; ++i)
    {
        for(PtrType j = ref.row_offset[i]; j < ref.row_offset[i + 1]; ++j)
        {
            expected.emplace_back(i, ref.col[j], ref.val[j]);
        }
    }
    std::sort(entries.begin(), entries.end());
    EXPECT_EQ(entries, expected);
}

namespace
{
    template <typename P>
    class hip_matrix_csr : public TypedBackendTest<P>
    {
    };
}

TYPED_TEST_SUITE(hip_matrix_csr, AcceleratorOnly<RealTypes>, TypedName);

// Asynchronous copies between the host and the accelerator
TYPED_TEST(hip_matrix_csr, async_copies_to_and_from_the_host)
{
    using T = typename TypeParam::value_type;

    const Csr<T> ref  = banded_with_gaps<T>(12);
    auto         host = ref.host_csr();

    auto A = accelerator_matrix<T>(Format{rocalution::CSR, 1});
    A->CopyFromHostAsync(*host);
    rocalution::_rocalution_sync();
    expect_csr_eq(*host_csr_of(*A), ref, false);

    HostMatrixCSR<T> back(descriptor());
    A->CopyToHostAsync(&back);
    rocalution::_rocalution_sync();
    expect_csr_eq(back, ref, false);

    // Into matrices of the same size
    auto B = accelerator_matrix<T>(Format{rocalution::CSR, 1});
    B->AllocateCSR(ref.nnz(), ref.n, ref.m);
    B->CopyFromHostAsync(*host);
    rocalution::_rocalution_sync();
    expect_csr_eq(*host_csr_of(*B), ref, false);

    HostMatrixCSR<T> again(descriptor());
    again.AllocateCSR(ref.nnz(), ref.n, ref.m);
    B->CopyToHostAsync(&again);
    rocalution::_rocalution_sync();
    expect_csr_eq(again, ref, false);
}
