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

// Operations of the matrix formats other than CSR, checked against CSR

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <tuple>

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"
#include "unit_test_types.hpp"

#include "base/host/host_matrix_coo.hpp"

using namespace rocalution_unit;

namespace
{
    using Param = std::tuple<Backend, Format>;

    class matrix_formats : public testing::TestWithParam<Param>
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

        template <typename T>
        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_matrix(backend(), format(), *ref.host_csr());
        }

        template <typename T>
        std::unique_ptr<BaseMatrix<T>> empty() const
        {
            return empty_matrix<T>(backend(), format());
        }

        // BCSR keeps the zeros of its blocks
        template <typename T>
        void expect_matrix(const BaseMatrix<T>& A, const Csr<T>& ref) const
        {
            EXPECT_EQ(A.GetMatFormat(), format().id);
            expect_csr_eq(*host_csr_of(A), ref, format().id == rocalution::BCSR);
        }

        bool is_native(const std::string& operation) const
        {
            return implements(backend(), format().id, operation);
        }
    };

}

TEST_P(matrix_formats, apply_add)
{
    if(backend() == Backend::Accelerator
       && (format().id == rocalution::HYB || format().id == rocalution::DENSE))
    {
        KNOWN_BUG("ellmv and gemv get beta 0, so the output is overwritten");
    }

    for_each_type<RealTypes>([&](auto type) {
        using T = typename decltype(type)::type;
        for(const auto& matrix : matrices<T>(format()))
        {
            SCOPED_TRACE(matrix.first);
            const Csr<T>& ref = matrix.second;

            auto A = make(ref);
            ASSERT_IN_LOOP(A != nullptr);

            const std::vector<T> x = rhs<T>(ref.n);
            std::vector<T>       y(ref.n);
            for(int i = 0; i < ref.n; ++i)
            {
                y[i] = static_cast<T>(i - 3);
            }
            std::vector<T> expected = ref.apply(x);
            for(int i = 0; i < ref.n; ++i)
            {
                expected[i] = y[i] + T(0.5) * expected[i];
            }

            auto out = make_vector(backend(), y);
            A->ApplyAdd(*make_vector(backend(), x), 0.5, out.get());
            EXPECT_EQ(values_of(*out), expected);
        }
    });
}

// Into an empty matrix and into one of the same size
TEST_P(matrix_formats, copies_within_the_backend)
{
    for_each_type<RealTypes>([&](auto type) {
        using T            = typename decltype(type)::type;
        const Csr<T> ref   = banded_with_gaps<T>(12);
        const Csr<T> other = transformed(ref, [](int, int, T v) { return -v; });

        auto A = make(ref);
        ASSERT_NE(A, nullptr);

        using Copy = std::function<void(BaseMatrix<T>&)>;
        const std::vector<std::pair<std::string, Copy>> copies
            = {{"CopyFrom", [&](BaseMatrix<T>& B) { B.CopyFrom(*A); }},
               {"CopyFromAsync", [&](BaseMatrix<T>& B) { B.CopyFromAsync(*A); }},
               {"CopyTo", [&](BaseMatrix<T>& B) { A->CopyTo(&B); }},
               {"CopyToAsync", [&](BaseMatrix<T>& B) { A->CopyToAsync(&B); }}};

        for(const auto& copy : copies)
        {
            SCOPED_TRACE(copy.first);

            auto C = make(other);
            ASSERT_IN_LOOP(C != nullptr);
            copy.second(*C);
            rocalution::_rocalution_sync();
            expect_matrix(*C, ref);

            auto into_empty = [&] {
                auto B = empty<T>();
                copy.second(*B);
                rocalution::_rocalution_sync();
                expect_matrix(*B, ref);
            };

            // CopyToAsync comes last, as the skip ends the current value type
            if(copy.first == "CopyToAsync" && backend() == Backend::Accelerator
               && (format().id == rocalution::ELL || format().id == rocalution::BCSR))
            {
                KNOWN_BUG("into an empty matrix it tests the source for being empty and copies "
                          "into an unallocated matrix, which exits the process");
                expect_in_child(into_empty);
                continue;
            }
            into_empty();
        }
    });
}

namespace
{
    class matrix_formats_accelerator : public matrix_formats
    {
    };

    class matrix_coo : public matrix_formats
    {
    };
}

TEST_P(matrix_formats_accelerator, copies_between_host_and_accelerator)
{
    for_each_type<RealTypes>([&](auto type) {
        using T          = typename decltype(type)::type;
        const Csr<T> ref = banded_with_gaps<T>(12);
        auto         A   = make(ref);
        ASSERT_NE(A, nullptr);
        auto& accel = dynamic_cast<rocalution::AcceleratorMatrix<T>&>(*A);

        auto host = host_copy(*A);

        using Copy = std::function<void(rocalution::AcceleratorMatrix<T>&, HostMatrix<T>&)>;
        const std::vector<std::pair<std::string, Copy>> uploads = {
            {"CopyFromHost",
             [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { d.CopyFromHost(h); }},
            {"CopyFromHostAsync",
             [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { d.CopyFromHostAsync(h); }},
            {"CopyFrom",
             [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { d.CopyFrom(h); }}};
        for(const auto& upload : uploads)
        {
            SCOPED_TRACE(upload.first);
            auto B = empty<T>();
            upload.second(dynamic_cast<rocalution::AcceleratorMatrix<T>&>(*B), *host);
            rocalution::_rocalution_sync();
            expect_matrix(*B, ref);
        }

        const std::vector<std::pair<std::string, Copy>> downloads = {
            {"CopyToHost",
             [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { d.CopyToHost(&h); }},
            {"CopyToHostAsync",
             [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { d.CopyToHostAsync(&h); }},
            {"CopyTo", [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { d.CopyTo(&h); }},
            {"host CopyFrom",
             [](rocalution::AcceleratorMatrix<T>& d, HostMatrix<T>& h) { h.CopyFrom(d); }}};
        for(const auto& download : downloads)
        {
            SCOPED_TRACE(download.first);
            auto H = empty_host_matrix<T>(format());
            download.second(accel, *H);
            rocalution::_rocalution_sync();
            HostMatrixCSR<T> csr(descriptor());
            ASSERT_IN_LOOP(csr.ConvertFrom(*H));
            expect_csr_eq(csr, ref, format().id == rocalution::BCSR);
        }
    });
}

TEST_P(matrix_formats_accelerator, copies_between_host_and_accelerator_of_the_same_size)
{
    for_each_type<RealTypes>([&](auto type) {
        using T            = typename decltype(type)::type;
        const Csr<T> ref   = banded_with_gaps<T>(12);
        const Csr<T> other = transformed(ref, [](int, int, T v) { return -v; });

        auto A = make(ref);
        ASSERT_NE(A, nullptr);
        auto host = host_copy(*A);

        {
            SCOPED_TRACE("CopyFromHost");
            auto B = make(other);
            ASSERT_NE(B, nullptr);
            dynamic_cast<rocalution::AcceleratorMatrix<T>&>(*B).CopyFromHost(*host);
            expect_matrix(*B, ref);
        }
        {
            SCOPED_TRACE("CopyToHost");
            auto O = make(other);
            ASSERT_NE(O, nullptr);
            auto H = host_copy(*O);
            dynamic_cast<rocalution::AcceleratorMatrix<T>&>(*A).CopyToHost(H.get());
            HostMatrixCSR<T> csr(descriptor());
            ASSERT_TRUE(csr.ConvertFrom(*H));
            expect_csr_eq(csr, ref, format().id == rocalution::BCSR);
        }
    });
}

TEST_P(matrix_formats, info_names_the_backend_and_the_format)
{
    auto A = make(banded_with_gaps<double>(12));
    ASSERT_NE(A, nullptr);
    const std::string info = logged([&] { A->Info(); });
    const std::string name = (backend() == Backend::Host ? "HostMatrix" : "HIPAcceleratorMatrix")
                             + format_name(format());
    EXPECT_NE(info.find(name), std::string::npos) << info;
}

TEST_P(matrix_formats, reading_a_missing_rocsparseio_file_fails_and_changes_nothing)
{
    REQUIRE_NATIVE(backend(), format().id, "ReadFileRSIO");

    const Csr<double> ref = banded_with_gaps<double>(12);
    auto              A   = make(ref);
    ASSERT_NE(A, nullptr);
    const std::string output
        = logged([&] { EXPECT_FALSE(A->ReadFileRSIO(temp_file("missing.rsio"))); });
    EXPECT_NE(output.find("cannot open"), std::string::npos) << output;
    expect_matrix(*A, ref);
}

TEST_P(matrix_formats, value_operations)
{
    for_each_type<RealTypes>([&](auto type) {
        using T          = typename decltype(type)::type;
        const Csr<T> ref = banded_with_gaps<T>(12);

        using Operation = std::function<bool(BaseMatrix<T>&)>;
        using Reference = std::function<T(int, int, T)>;
        const std::vector<std::tuple<std::string, Operation, Reference>> operations
            = {{"Scale",
                [](BaseMatrix<T>& A) { return A.Scale(3.0); },
                [](int, int, T v) { return 3 * v; }},
               {"ScaleDiagonal",
                [](BaseMatrix<T>& A) { return A.ScaleDiagonal(3.0); },
                [](int i, int j, T v) { return i == j ? 3 * v : v; }},
               {"ScaleOffDiagonal",
                [](BaseMatrix<T>& A) { return A.ScaleOffDiagonal(3.0); },
                [](int i, int j, T v) { return i != j ? 3 * v : v; }},
               {"AddScalar",
                [](BaseMatrix<T>& A) { return A.AddScalar(0.5); },
                [](int, int, T v) { return v + T(0.5); }},
               {"AddScalarDiagonal",
                [](BaseMatrix<T>& A) { return A.AddScalarDiagonal(0.5); },
                [](int i, int j, T v) { return i == j ? v + T(0.5) : v; }},
               {"AddScalarOffDiagonal",
                [](BaseMatrix<T>& A) { return A.AddScalarOffDiagonal(0.5); },
                [](int i, int j, T v) { return i != j ? v + T(0.5) : v; }}};

        for(const auto& operation : operations)
        {
            SCOPED_TRACE(std::get<0>(operation));
            REQUIRE_NATIVE(backend(), format().id, std::get<0>(operation));

            auto A = make(ref);
            ASSERT_IN_LOOP(A != nullptr);
            ASSERT_IN_LOOP(std::get<1>(operation)(*A));
            expect_matrix(*A, transformed(ref, std::get<2>(operation)));
        }
    });
}

namespace
{
    template <typename T>
    using Entry = std::tuple<int, int, T>;

    // The entries of a COO matrix of any backend, in their stored order
    template <typename T>
    std::vector<Entry<T>> coo_entries(const BaseMatrix<T>& A)
    {
        auto host = host_copy(A);
        auto coo  = dynamic_cast<rocalution::HostMatrixCOO<T>*>(host.get());
        EXPECT_NE(coo, nullptr);

        const int64_t    nnz = coo->GetNnz();
        std::vector<int> row(nnz), col(nnz);
        std::vector<T>   val(nnz);
        coo->CopyToCOO(row.data(), col.data(), val.data());

        std::vector<Entry<T>> entries;
        for(int64_t k = 0; k < nnz; ++k)
        {
            entries.emplace_back(row[k], col[k], val[k]);
        }
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

    template <typename T>
    bool sorted_by_row_and_column(const std::vector<Entry<T>>& entries)
    {
        return std::is_sorted(
            entries.begin(), entries.end(), [](const Entry<T>& a, const Entry<T>& b) {
                return std::make_pair(std::get<0>(a), std::get<1>(a))
                       < std::make_pair(std::get<0>(b), std::get<1>(b));
            });
    }

    // Permute and PermuteBackward of a COO matrix, compared on the entries
    // because the conversion to CSR requires them to be sorted
    template <typename T>
    void expect_permuted(const BaseMatrix<T>& A, const Csr<T>& ref)
    {
        std::vector<Entry<T>> entries = coo_entries(A);
        std::sort(entries.begin(), entries.end());
        EXPECT_EQ(entries, entries_of(ref));
    }
}

// PermuteBackward permutes with the inverse permutation
TEST_P(matrix_formats, permute_and_permute_backward)
{
    REQUIRE_NATIVE(backend(), format().id, "Permute");
    REQUIRE_NATIVE(backend(), format().id, "PermuteBackward");

    for_each_type<RealTypes>([&](auto type) {
        using T                     = typename decltype(type)::type;
        const Csr<T>           ref  = banded_with_gaps<T>(12);
        const std::vector<int> perm = {3, 0, 11, 6, 1, 9, 7, 2, 10, 5, 8, 4};
        std::vector<int>       inverse(perm.size());
        for(size_t i = 0; i < perm.size(); ++i)
        {
            inverse[perm[i]] = static_cast<int>(i);
        }

        auto p = make_vector(backend(), perm);

        auto A = make(ref);
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->Permute(*p));
        expect_permuted(*A, permuted(ref, perm));

        auto B = make(ref);
        ASSERT_NE(B, nullptr);
        ASSERT_TRUE(B->PermuteBackward(*p));
        expect_permuted(*B, permuted(ref, inverse));
    });
}

// A COO matrix is documented to be sorted by row, then by column
TEST_P(matrix_coo, permute_keeps_the_coo_entries_sorted)
{
    for_each_type<RealTypes>([&](auto type) {
        using T          = typename decltype(type)::type;
        const Csr<T> ref = banded_with_gaps<T>(12);
        auto         A   = make(ref);
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->Permute(
            *make_vector(backend(), std::vector<int>{3, 0, 11, 6, 1, 9, 7, 2, 10, 5, 8, 4})));

        KNOWN_BUG("Permute leaves the entries in their old order");

        EXPECT_TRUE(sorted_by_row_and_column(coo_entries(*A)));
    });
}

TEST_P(matrix_formats, sort)
{
    REQUIRE_NATIVE(backend(), format().id, "Permute");
    REQUIRE_NATIVE(backend(), format().id, "Sort");

    for_each_type<RealTypes>([&](auto type) {
        using T                     = typename decltype(type)::type;
        const Csr<T>           ref  = banded_with_gaps<T>(12);
        const std::vector<int> perm = {3, 0, 11, 6, 1, 9, 7, 2, 10, 5, 8, 4};

        auto A = make(ref);
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->Permute(*make_vector(backend(), perm)));
        ASSERT_TRUE(A->Sort());
        EXPECT_TRUE(sorted_by_row_and_column(coo_entries(*A)));
        expect_matrix(*A, permuted(ref, perm));
    });
}

TEST_P(matrix_formats, write_and_read_rocsparseio_files)
{
    REQUIRE_NATIVE(backend(), format().id, "WriteFileRSIO");
    REQUIRE_NATIVE(backend(), format().id, "ReadFileRSIO");

    for_each_type<RealTypes>([&](auto type) {
        using T                = typename decltype(type)::type;
        const Csr<T>      ref  = banded_with_gaps<T>(12);
        const std::string file = temp_file("rsio");

        auto A = make(ref);
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->WriteFileRSIO(file));

        auto B = empty<T>();
        ASSERT_TRUE(B->ReadFileRSIO(file));
        expect_matrix(*B, ref);
        std::remove(file.c_str());
    });
}

// An operation that the backend does not implement for the format returns
// false and leaves the matrix unchanged, so that LocalMatrix can retry in CSR
// or on the host
TEST_P(matrix_formats, unimplemented_operations_return_false_and_change_nothing)
{
    for_each_type<RealTypes>([&](auto type) {
        using T                = typename decltype(type)::type;
        const Csr<T>      ref  = banded_with_gaps<T>(12);
        const std::string file = temp_file("fallback.rsio");
        ASSERT_TRUE(ref.host_csr()->WriteFileRSIO(file));

        auto p = make_vector(backend(), std::vector<int>{3, 0, 11, 6, 1, 9, 7, 2, 10, 5, 8, 4});

        using Operation = std::function<bool(BaseMatrix<T>&)>;
        const std::vector<std::pair<std::string, Operation>> operations = {
            {"Scale", [](BaseMatrix<T>& A) { return A.Scale(3.0); }},
            {"ScaleDiagonal", [](BaseMatrix<T>& A) { return A.ScaleDiagonal(3.0); }},
            {"ScaleOffDiagonal", [](BaseMatrix<T>& A) { return A.ScaleOffDiagonal(3.0); }},
            {"AddScalar", [](BaseMatrix<T>& A) { return A.AddScalar(0.5); }},
            {"AddScalarDiagonal", [](BaseMatrix<T>& A) { return A.AddScalarDiagonal(0.5); }},
            {"AddScalarOffDiagonal", [](BaseMatrix<T>& A) { return A.AddScalarOffDiagonal(0.5); }},
            {"Permute", [&](BaseMatrix<T>& A) { return A.Permute(*p); }},
            {"PermuteBackward", [&](BaseMatrix<T>& A) { return A.PermuteBackward(*p); }},
            {"Sort", [](BaseMatrix<T>& A) { return A.Sort(); }},
            {"WriteFileRSIO",
             [&](BaseMatrix<T>& A) {
                 const std::string out    = file + ".out";
                 bool              native = A.WriteFileRSIO(out);
                 EXPECT_FALSE(std::ifstream(out).good()) << out << " was written";
                 std::remove(out.c_str());
                 return native;
             }},
            {"ReadFileRSIO", [&](BaseMatrix<T>& A) { return A.ReadFileRSIO(file); }},
        };

        int checked = 0;
        for(const auto& operation : operations)
        {
            if(is_native(operation.first))
            {
                continue;
            }
            SCOPED_TRACE(operation.first);

            auto A = make(ref);
            ASSERT_IN_LOOP(A != nullptr);
            EXPECT_FALSE(operation.second(*A));
            expect_matrix(*A, ref);
            ++checked;
        }

        std::remove(file.c_str());
        if(checked == 0)
        {
            GTEST_SKIP() << "every operation is implemented";
        }
    });
}

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_formats,
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

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_formats_accelerator,
                         testing::Combine(testing::Values(Backend::Accelerator),
                                          testing::Values(Format{rocalution::COO, 1},
                                                          Format{rocalution::ELL, 1},
                                                          Format{rocalution::DIA, 1},
                                                          Format{rocalution::HYB, 1},
                                                          Format{rocalution::MCSR, 1},
                                                          Format{rocalution::BCSR, 2},
                                                          Format{rocalution::BCSR, 3},
                                                          Format{rocalution::DENSE, 1})),
                         backend_format_name);

INSTANTIATE_TEST_SUITE_P(formats,
                         matrix_coo,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::COO, 1})),
                         backend_format_name);

// Not a typed test: unit_test_backend_matrix.cpp adds plain tests to this suite
TEST(host_matrix_coo, write_and_read_matrix_market_files)
{
    for_each_type<RealTypes>([&](auto type) {
        using T                = typename decltype(type)::type;
        const Csr<T>      ref  = banded_with_gaps<T>(12);
        const std::string file = temp_file("mtx");

        auto A = make_matrix(Backend::Host, Format{rocalution::COO, 1}, *ref.host_csr());
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->WriteFileMTX(file));

        rocalution::HostMatrixCOO<T> B(descriptor());
        ASSERT_TRUE(B.ReadFileMTX(file));
        expect_csr_eq(*host_csr_of(B), ref, false);
        std::remove(file.c_str());

        EXPECT_FALSE(B.ReadFileMTX(file));
    });
}

TEST(host_matrix_coo, writing_a_matrix_market_file_into_a_missing_directory_fails)
{
    const Csr<double> ref = banded_with_gaps<double>(12);
    auto              A   = make_matrix(Backend::Host, Format{rocalution::COO, 1}, *ref.host_csr());
    ASSERT_NE(A, nullptr);
    const std::string file   = temp_file("missing") + "/matrix.mtx";
    const std::string output = logged([&] { EXPECT_FALSE(A->WriteFileMTX(file)); });
    EXPECT_NE(output.find("cannot open"), std::string::npos) << output;
}

namespace
{
    // Diagonally dominant, so that LU without pivoting is stable
    template <typename T>
    Csr<T> full(int n, int m, int seed)
    {
        Csr<T> A(n, m);
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < m; ++j)
            {
                T v = static_cast<T>(i == j ? 20.0 + i : ((3 * i + 5 * j + seed) % 7) - 3.0);
                if(v != T(0))
                {
                    A.add(j, v);
                }
            }
            A.end_row();
        }
        return A;
    }

    // Row major, of a matrix of any backend and format
    template <typename T>
    std::vector<T> dense_of(const BaseMatrix<T>& A)
    {
        return dense(csr_of(A));
    }

    template <typename P>
    class matrix_dense : public TypedBackendTest<P>
    {
    protected:
        using T = typename P::value_type;

        std::unique_ptr<BaseMatrix<T>> make(const Csr<T>& ref) const
        {
            return make_matrix(P::backend, Format{rocalution::DENSE, 1}, *ref.host_csr());
        }

        bool is_native(const std::string& operation) const
        {
            return implements(P::backend, rocalution::DENSE, operation);
        }
    };
}

TYPED_TEST_SUITE(matrix_dense, WithBackends<RealTypes>, TypedName);

// MatMatMult writes into a matrix of the size of the product
TYPED_TEST(matrix_dense, mat_mat_mult)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> a = full<T>(4, 6, 0);
    const Csr<T> b = full<T>(6, 5, 2);

    auto A = this->make(a);
    auto B = this->make(b);
    auto C = this->make(full<T>(4, 5, 1));
    ASSERT_NE(A, nullptr);
    ASSERT_NE(B, nullptr);
    ASSERT_NE(C, nullptr);
    ASSERT_TRUE(C->MatMatMult(*A, *B));

    EXPECT_EQ(dense_of(*C), dense_product(dense(a), dense(b), 4, 6, 5));
}

// A wide matrix, to tell rows from columns
TYPED_TEST(matrix_dense, extract_and_replace_rows_and_columns)
{
    using T           = typename TypeParam::value_type;
    const Csr<T>   a  = full<T>(5, 7, 0);
    std::vector<T> da = dense(a);
    auto           A  = this->make(a);
    ASSERT_NE(A, nullptr);

    auto column = make_vector(TypeParam::backend, std::vector<T>(5));
    ASSERT_TRUE(A->ExtractColumnVector(3, column.get()));
    EXPECT_EQ(values_of(*column), (std::vector<T>{da[3], da[10], da[17], da[24], da[31]}));

    const std::vector<T> new_row = {1, 2, 3, 4, 5, 6, 7};
    ASSERT_TRUE(A->ReplaceRowVector(1, *make_vector(TypeParam::backend, new_row)));
    std::copy(new_row.begin(), new_row.end(), da.begin() + 7);

    const std::vector<T> new_column = {-1, -2, -3, -4, -5};
    ASSERT_TRUE(A->ReplaceColumnVector(4, *make_vector(TypeParam::backend, new_column)));
    for(int i = 0; i < 5; ++i)
    {
        da[i * 7 + 4] = new_column[i];
    }

    EXPECT_EQ(dense_of(*A), da);

    if(TypeParam::backend == Backend::Host)
    {
        KNOWN_BUG("ExtractRowVector stops after as many columns as there are rows");
    }

    auto row = make_vector(TypeParam::backend, std::vector<T>(7));
    ASSERT_TRUE(A->ExtractRowVector(2, row.get()));
    EXPECT_EQ(values_of(*row), std::vector<T>(da.begin() + 14, da.begin() + 21));
}

TYPED_TEST(matrix_dense, lu_factorize_and_solve)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "LUFactorize");
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "LUSolve");

    const Csr<T>         a = full<T>(8, 8, 0);
    const std::vector<T> x = {1, -2, 3, -4, 5, -6, 7, -8};

    auto A   = this->make(a);
    auto out = make_vector(TypeParam::backend, std::vector<T>(8));
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->LUFactorize());
    ASSERT_TRUE(A->LUSolve(*make_vector(TypeParam::backend, a.apply(x)), out.get()));
    expect_near(values_of(*out), x, tol<T>());
}

TYPED_TEST(matrix_dense, qr_decompose_and_solve)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "QRDecompose");
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "QRSolve");

    const Csr<T>         a = full<T>(8, 8, 3);
    const std::vector<T> x = {1, -2, 3, -4, 5, -6, 7, -8};

    auto A   = this->make(a);
    auto out = make_vector(TypeParam::backend, std::vector<T>(8));
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->QRDecompose());
    ASSERT_TRUE(A->QRSolve(*make_vector(TypeParam::backend, a.apply(x)), out.get()));
    expect_near(values_of(*out), x, tol<T>());
}

TYPED_TEST(matrix_dense, invert)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "Invert");

    const int    n = 7;
    const Csr<T> a = full<T>(n, n, 5);

    auto A = this->make(a);
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->Invert());

    std::vector<T> identity(n * n, T(0));
    for(int i = 0; i < n; ++i)
    {
        identity[i * n + i] = 1;
    }
    expect_near(dense_product(dense_of(*A), dense(a), n, n, n), identity, tol<T>());
}

// (I - beta v v^T) maps column idx below the diagonal onto a multiple of the first unit vector
TYPED_TEST(matrix_dense, householder)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "Householder");

    const int    n   = 6;
    const int    idx = 2;
    const Csr<T> a   = full<T>(n, n, 4);

    auto A    = this->make(a);
    auto v    = make_vector(TypeParam::backend, std::vector<T>(n));
    T    beta = 0;
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->Householder(idx, beta, v.get()));

    const std::vector<T> da = dense(a);
    std::vector<T>       column(n - idx);
    for(int i = idx; i < n; ++i)
    {
        column[i - idx] = da[i * n + idx];
    }

    const std::vector<T> vv = values_of(*v);
    EXPECT_EQ(vv[0], T(1));
    T dot = 0;
    for(int i = 0; i < n - idx; ++i)
    {
        dot += vv[i] * column[i];
    }

    T norm = 0;
    for(T c : column)
    {
        norm += c * c;
    }

    std::vector<T> reflected(n - idx), expected(n - idx, T(0));
    for(int i = 0; i < n - idx; ++i)
    {
        reflected[i] = column[i] - beta * dot * vv[i];
    }
    expected[0] = column[0] > 0 ? -std::sqrt(norm) : std::sqrt(norm);
    expect_near(reflected, expected, tol<T>());
}

// With a zero diagonal entry the reflection maps the column onto -|column| e_1
TYPED_TEST(matrix_dense, householder_of_a_column_with_a_zero_diagonal_entry)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "Householder");

    const int    n   = 6;
    const int    idx = 2;
    const Csr<T> a   = filtered(full<T>(n, n, 4), [](int i, int j, T) { return i != 2 || j != 2; });

    auto A    = this->make(a);
    auto v    = make_vector(TypeParam::backend, std::vector<T>(n));
    T    beta = 0;
    ASSERT_NE(A, nullptr);
    ASSERT_TRUE(A->Householder(idx, beta, v.get()));

    const std::vector<T> da = dense(a);
    std::vector<T>       column(n - idx);
    for(int i = idx; i < n; ++i)
    {
        column[i - idx] = da[i * n + idx];
    }
    ASSERT_EQ(column[0], T(0));

    const std::vector<T> vv = values_of(*v);
    EXPECT_EQ(vv[0], T(1));
    T dot = 0, norm = 0;
    for(int i = 0; i < n - idx; ++i)
    {
        dot += vv[i] * column[i];
        norm += column[i] * column[i];
    }

    std::vector<T> reflected(n - idx), expected(n - idx, T(0));
    for(int i = 0; i < n - idx; ++i)
    {
        reflected[i] = column[i] - beta * dot * vv[i];
    }
    expected[0] = -std::sqrt(norm);
    expect_near(reflected, expected, tol<T>());
}

// The QR decomposition of an n x m matrix has min(n, m) reflections. A wide
// system has many solutions; any of them passes.
TYPED_TEST(matrix_dense, qr_solves_tall_and_wide_systems)
{
    using T = typename TypeParam::value_type;
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "QRDecompose");
    REQUIRE_NATIVE(TypeParam::backend, rocalution::DENSE, "QRSolve");
    const Backend b = TypeParam::backend;

    {
        SCOPED_TRACE("tall");
        const Csr<T>         tall = full<T>(8, 5, 1);
        const std::vector<T> x    = {1, -2, 3, -4, 5};

        auto A   = this->make(tall);
        auto out = make_vector(b, std::vector<T>(5));
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->QRDecompose());
        ASSERT_TRUE(A->QRSolve(*make_vector(b, tall.apply(x)), out.get()));
        expect_near(values_of(*out), x, tol<T>());
    }
    {
        SCOPED_TRACE("wide");
        const Csr<T>         wide = full<T>(5, 8, 2);
        const std::vector<T> rb   = wide.apply(std::vector<T>{1, -2, 3, -4, 5, -6, 7, -8});

        auto A   = this->make(wide);
        auto out = make_vector(b, std::vector<T>(8, T(-7)));
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->QRDecompose());
        ASSERT_TRUE(A->QRSolve(*make_vector(b, rb), out.get()));

        expect_near(wide.apply(values_of(*out)), rb, tol<T>());
    }
}

// Row operations on a 5 x 0 matrix and column operations on a 0 x 5 matrix
TYPED_TEST(matrix_dense, rows_and_columns_of_matrices_without_entries)
{
    using T         = typename TypeParam::value_type;
    const Backend b = TypeParam::backend;
    auto          e = make_vector(b, std::vector<T>());

    // The row and column vectors are empty, and the matrices keep their shape
    const auto expect_shape = [&](const BaseMatrix<T>& A, int m, int n) {
        EXPECT_EQ(A.GetM(), m);
        EXPECT_EQ(A.GetN(), n);
        EXPECT_EQ(A.GetNnz(), 0);
        EXPECT_EQ(A.GetMatFormat(), rocalution::DENSE);
        EXPECT_EQ(e->GetSize(), 0);
    };

    auto rows = empty_matrix<T>(b, Format{rocalution::DENSE, 1});
    rows->AllocateDENSE(5, 0);
    EXPECT_TRUE(rows->ExtractRowVector(2, e.get()));
    expect_shape(*rows, 5, 0);
    EXPECT_TRUE(rows->ReplaceRowVector(2, *e));
    expect_shape(*rows, 5, 0);

    auto columns = empty_matrix<T>(b, Format{rocalution::DENSE, 1});
    columns->AllocateDENSE(0, 5);
    EXPECT_TRUE(columns->ExtractColumnVector(2, e.get()));
    expect_shape(*columns, 0, 5);
    EXPECT_TRUE(columns->ReplaceColumnVector(2, *e));
    expect_shape(*columns, 0, 5);
}

TYPED_TEST(matrix_dense, conversion_from_dense_copies_the_matrix)
{
    using T        = typename TypeParam::value_type;
    const Csr<T> a = full<T>(4, 6, 0);

    auto A = this->make(a);
    ASSERT_NE(A, nullptr);
    auto B = empty_matrix<T>(TypeParam::backend, Format{rocalution::DENSE, 1});
    ASSERT_TRUE(B->ConvertFrom(*A));
    EXPECT_EQ(dense_of(*B), dense(a));
}

TYPED_TEST(matrix_dense, unimplemented_operations_return_false_and_change_nothing)
{
    using T                = typename TypeParam::value_type;
    const int            n = 6;
    const Csr<T>         a = full<T>(n, n, 4);
    const std::vector<T> b(n, T(1)), untouched(n, T(-7));

    auto untouched_out = [&](const std::function<bool(BaseVector<T>*)>& f) {
        auto out    = make_vector(TypeParam::backend, untouched);
        bool native = f(out.get());
        EXPECT_EQ(values_of(*out), untouched);
        return native;
    };

    using Operation = std::function<bool(BaseMatrix<T>&)>;
    const std::vector<std::pair<std::string, Operation>> operations = {
        {"LUFactorize", [](BaseMatrix<T>& A) { return A.LUFactorize(); }},
        {"LUSolve",
         [&](BaseMatrix<T>& A) {
             return untouched_out([&](BaseVector<T>* out) {
                 return A.LUSolve(*make_vector(TypeParam::backend, b), out);
             });
         }},
        {"QRDecompose", [](BaseMatrix<T>& A) { return A.QRDecompose(); }},
        {"QRSolve",
         [&](BaseMatrix<T>& A) {
             return untouched_out([&](BaseVector<T>* out) {
                 return A.QRSolve(*make_vector(TypeParam::backend, b), out);
             });
         }},
        {"Invert", [](BaseMatrix<T>& A) { return A.Invert(); }},
        {"Householder",
         [&](BaseMatrix<T>& A) {
             T    beta = -7;
             bool native
                 = untouched_out([&](BaseVector<T>* v) { return A.Householder(2, beta, v); });
             EXPECT_EQ(beta, -7);
             return native;
         }},
    };

    int checked = 0;
    for(const auto& operation : operations)
    {
        if(this->is_native(operation.first))
        {
            continue;
        }
        SCOPED_TRACE(operation.first);

        auto A = this->make(a);
        ASSERT_IN_LOOP(A != nullptr);
        EXPECT_FALSE(operation.second(*A));
        EXPECT_EQ(dense_of(*A), dense(a));
        ++checked;
    }

    if(checked == 0)
    {
        GTEST_SKIP() << "every operation is implemented";
    }
}

namespace
{
    // The pattern of banded_with_gaps with a dominant diagonal
    template <typename T>
    Csr<T> dominant_banded_with_gaps(int n)
    {
        const Csr<T> pattern = banded_with_gaps<T>(n);
        return transformed(pattern, [](int i, int j, T) {
            return static_cast<T>(i == j ? 50.0 + i : ((i + 2 * j) % 4) - 1.5);
        });
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

    class block_solvers : public matrix_formats
    {
    };

    // Row major transpose of the dense n x n matrix D
    template <typename T>
    std::vector<T> dense_transposed(const std::vector<T>& D, int n)
    {
        std::vector<T> D_t(D.size());
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                D_t[j * n + i] = D[i * n + j];
            }
        }
        return D_t;
    }
}

// The block formats factorize on the pattern of their blocks, explicit zeros
// included, which is the scalar ILU(0) of the blocked pattern
TEST_P(block_solvers, ilu0_factorize_and_lu_solve_match_the_reference)
{
    REQUIRE_NATIVE(backend(), format().id, "ILU0Factorize");
    REQUIRE_NATIVE(backend(), format().id, "LUSolve");

    for_each_type<RealTypes>([&](auto type) {
        using T                  = typename decltype(type)::type;
        const Csr<T>         ref = dominant_banded_with_gaps<T>(12);
        const std::vector<T> b   = rhs<T>(12);
        const std::vector<T> F   = ilu0_dense(blocked(ref, format().blockdim));

        auto A   = make(ref);
        auto out = make_vector(backend(), std::vector<T>(12, T(-7)));
        ASSERT_NE(A, nullptr);
        ASSERT_TRUE(A->ILU0Factorize());
        expect_near(dense_of(*A), F, tol<T>());

        A->LUAnalyse();
        ASSERT_TRUE(A->LUSolve(*make_vector(backend(), b), out.get()));
        expect_near(
            values_of(*out), upper_solve(F, 12, lower_solve(F, 12, b, true), false), tol<T>());
    });
}

// LL solves with the lower triangle L of the matrix and with L^T
TEST_P(block_solvers, triangular_and_ll_solves_match_the_reference)
{
    REQUIRE_NATIVE(backend(), format().id, "LSolve");
    REQUIRE_NATIVE(backend(), format().id, "USolve");
    REQUIRE_NATIVE(backend(), format().id, "LLSolve");

    for_each_type<RealTypes>([&](auto type) {
        using T                  = typename decltype(type)::type;
        const Csr<T>         ref = dominant_banded_with_gaps<T>(12);
        const std::vector<T> b   = rhs<T>(12);
        const std::vector<T> D   = dense(ref);

        auto A = make(ref);
        ASSERT_NE(A, nullptr);

        using Analyse = std::function<void(BaseMatrix<T>&)>;
        using Solve
            = std::function<bool(const BaseMatrix<T>&, const BaseVector<T>&, BaseVector<T>*)>;
        const std::vector<std::tuple<std::string, Analyse, Solve, std::vector<T>>> solves
            = {{"L",
                [](BaseMatrix<T>& M) { M.LAnalyse(false); },
                [](const BaseMatrix<T>& M, const BaseVector<T>& in, BaseVector<T>* x) {
                    return M.LSolve(in, x);
                },
                lower_solve(D, 12, b, false)},
               {"L unit diagonal",
                [](BaseMatrix<T>& M) { M.LAnalyse(true); },
                [](const BaseMatrix<T>& M, const BaseVector<T>& in, BaseVector<T>* x) {
                    return M.LSolve(in, x);
                },
                lower_solve(D, 12, b, true)},
               {"U",
                [](BaseMatrix<T>& M) { M.UAnalyse(false); },
                [](const BaseMatrix<T>& M, const BaseVector<T>& in, BaseVector<T>* x) {
                    return M.USolve(in, x);
                },
                upper_solve(D, 12, b, false)},
               {"U unit diagonal",
                [](BaseMatrix<T>& M) { M.UAnalyse(true); },
                [](const BaseMatrix<T>& M, const BaseVector<T>& in, BaseVector<T>* x) {
                    return M.USolve(in, x);
                },
                upper_solve(D, 12, b, true)},
               {"LL",
                [](BaseMatrix<T>& M) { M.LLAnalyse(); },
                [](const BaseMatrix<T>& M, const BaseVector<T>& in, BaseVector<T>* x) {
                    return M.LLSolve(in, x);
                },
                upper_solve(dense_transposed(D, 12), 12, lower_solve(D, 12, b, false), false)}};

        for(const auto& solve : solves)
        {
            SCOPED_TRACE(std::get<0>(solve));

            auto out = make_vector(backend(), std::vector<T>(12, T(-7)));
            std::get<1>(solve)(*A);
            ASSERT_IN_LOOP(std::get<2>(solve)(*A, *make_vector(backend(), b), out.get()));
            expect_near(values_of(*out), std::get<3>(solve), tol<T>());
        }
    });
}

// The solves run without their analysis, which is a fatal error where they are
// not implemented
TEST_P(block_solvers, unimplemented_operations_return_false_and_change_nothing)
{
    for_each_type<RealTypes>([&](auto type) {
        using T                  = typename decltype(type)::type;
        const Csr<T>         ref = dominant_banded_with_gaps<T>(12);
        const std::vector<T> b   = rhs<T>(12), untouched(12, T(-7));

        using Solve = std::function<bool(BaseMatrix<T>&, const BaseVector<T>&, BaseVector<T>*)>;
        const std::vector<std::pair<std::string, Solve>> operations = {
            {"ILU0Factorize",
             [](BaseMatrix<T>& A, const BaseVector<T>&, BaseVector<T>*) {
                 return A.ILU0Factorize();
             }},
            {"LUSolve",
             [](BaseMatrix<T>& A, const BaseVector<T>& in, BaseVector<T>* x) {
                 return A.LUSolve(in, x);
             }},
            {"LSolve",
             [](BaseMatrix<T>& A, const BaseVector<T>& in, BaseVector<T>* x) {
                 return A.LSolve(in, x);
             }},
            {"USolve",
             [](BaseMatrix<T>& A, const BaseVector<T>& in, BaseVector<T>* x) {
                 return A.USolve(in, x);
             }},
            {"LLSolve",
             [](BaseMatrix<T>& A, const BaseVector<T>& in, BaseVector<T>* x) {
                 return A.LLSolve(in, x);
             }},
        };

        int checked = 0;
        for(const auto& operation : operations)
        {
            if(is_native(operation.first))
            {
                continue;
            }
            SCOPED_TRACE(operation.first);

            auto A   = make(ref);
            auto out = make_vector(backend(), untouched);
            ASSERT_IN_LOOP(A != nullptr);
            EXPECT_FALSE(operation.second(*A, *make_vector(backend(), b), out.get()));
            EXPECT_EQ(values_of(*out), untouched);
            expect_matrix(*A, ref);
            ++checked;
        }

        if(checked == 0)
        {
            GTEST_SKIP() << "every operation is implemented";
        }
    });
}

INSTANTIATE_TEST_SUITE_P(formats,
                         block_solvers,
                         testing::Combine(testing::Values(Backend::Host, Backend::Accelerator),
                                          testing::Values(Format{rocalution::MCSR, 1},
                                                          Format{rocalution::BCSR, 2},
                                                          Format{rocalution::BCSR, 3})),
                         backend_format_name);
