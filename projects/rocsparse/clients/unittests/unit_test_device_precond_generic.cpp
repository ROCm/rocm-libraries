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

// Generic incomplete-factorization API (rocsparse_spilu0 / spic0 / spildlt0):
// descriptor inputs and outputs, argument validation, stage ordering, CSR (every
// value and index type, batched) and BSR matrices, numeric boost, singularity
// reporting, and the kernel regimes selected by the row length (hash vs binary
// search). Every factorization is compared with a host reference and with the
// residual of the factorization on the sparsity pattern.

#include "unit_test_device_precond_common.hpp"

#include <algorithm>
#include <string>

using namespace precond_ut;

class PrecondGeneric : public HandleTest
{
};

template <typename KindTag>
class PrecondIluIc : public KindFixture<KindTag>
{
};
TYPED_TEST_SUITE(PrecondIluIc, IluIc);

// ---------------------------------------------------------------------------
// CSR
// ---------------------------------------------------------------------------

// On the grid matrix a row never shares a column below the diagonal with a row it
// depends on, so every IC(0) inner sum is zero (the band tests below have non-zero
// sums).
TYPED_TEST(PrecondIluIc, csr_types_policies_batches_and_index_base)
{
    constexpr Kind K      = TestFixture::K;
    const auto     handle = this->handle;
    const auto     one    = rocsparse_index_base_one;
    run_options    opt;
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        check_csr_case<K, T>(handle, 6, 1, true, opt);
        check_csr_case<K, T>(handle, 5, 1, true, opt, one);
    });
    opt.policy = rocsparse_analysis_policy_force;
    check_csr_case<K, double>(handle, 5, 1, true, opt);
    opt.policy = rocsparse_analysis_policy_reuse;
    // Batched, with a shared and with one row pointer per batch entry.
    for(const bool shared : {true, false})
        check_csr_case<K, double>(
            handle, 4, 3, shared, opt, shared ? one : rocsparse_index_base_zero);
    check_csr_case<K, rocsparse_float_complex>(handle, 4, 2, false, opt);
    // 64-bit row offsets and column indices.
    check_csr_case<K, double, int64_t, int32_t>(handle, 5, 1, true, opt, one);
    check_csr_case<K, float, int64_t, int64_t>(handle, 5, 1, true, opt);
}

// IC(0) inner sums with several non-zero terms (rows that share columns below the
// diagonal) through the hash and the binary-search kernel, with real-valued entries.
TEST_F(PrecondGeneric, spic0_csr_nonzero_inner_sums_hash)
{
    run_options opt;
    // Longest rows of 3, 6, 21 and 41 entries.
    for(const int32_t w : {2, 5, 20, 40})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            check_csr_matrix<Kind::ic, T>(
                handle, make_lower_band_matrix<T>(60, w, false), opt, "band");
        });
    }
    // A dense last row over narrow band rows: table sizes 32 .. 512 (lengths below
    // the sizes) and band 3 so that the sums have three terms.
    for(const int32_t fat : {8, 40, 100, 300, 511})
    {
        check_csr_matrix<Kind::ic, double>(
            handle, make_fat_row_matrix<double>(fat, false, 3), opt, "fat " + std::to_string(fat));
        check_csr_matrix<Kind::ic, float>(
            handle, make_fat_row_matrix<float>(fat, false, 3), opt, "fat " + std::to_string(fat));
    }
    // Band and fat-row matrices (non-zero inner sums) in the other index base.
    const auto one = rocsparse_index_base_one;
    check_csr_matrix<Kind::ic, double>(
        handle, make_lower_band_matrix<double>(60, 5, false), opt, "band", one);
    check_csr_matrix<Kind::ic, double>(
        handle, make_fat_row_matrix<double>(520, false, 3), opt, "fat 520", one);
    check_csr_matrix<Kind::ilu, double>(
        handle, with_upper_triangle(make_lower_band_matrix<double>(60, 5)), opt, "band", one);
    check_csr_matrix<Kind::ilu, double>(
        handle,
        with_upper_triangle(make_fat_row_matrix<double>(520, true, 3)),
        opt,
        "fat 520",
        one);
    check_csr_matrix<Kind::ildlt, double>(
        handle, make_fat_row_matrix<double>(40, true, 3), opt, "fat 40", one);
}

// ---------------------------------------------------------------------------
// Singularity reporting
// ---------------------------------------------------------------------------
// The singularity position is reported in the index base of the matrix.
TYPED_TEST(PrecondIluIc, symbolic_and_exact_zero_pivots)
{
    constexpr Kind K      = TestFixture::K;
    const auto     handle = this->handle;
    run_options    opt;
    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        // A missing diagonal entry (row 5) is a symbolic singularity.
        const auto symbolic = without_diagonal(make_grid_matrix<double>(4, false), 5);
        singular_case<K, double>(handle, symbolic, opt, rocsparse_singularity_symbolic, 5, base);
        // The diagonal entry of row 0 is zero, and row 0 has no row above it to
        // change that, so its pivot is exactly 0.
        for_all_types([&](auto tag) {
            using T         = decltype(tag);
            auto a          = make_grid_matrix<T>(4, false);
            a.val[a.ptr[0]] = make_value<T>(0.0);
            singular_case<K, T>(handle, a, opt, rocsparse_singularity_numeric_exact, 0, base);
        });
    }
    opt.policy = rocsparse_analysis_policy_force;
    singular_case<K, double>(handle,
                             without_diagonal(make_grid_matrix<double>(4, false), 5),
                             opt,
                             rocsparse_singularity_symbolic,
                             5);
}

// ILU(0): a pivot is near-singular when |u_ii| <= tol (inclusive). IC(0): when
// l_ii^2 <= tol^2 on the pivot before the square root. The first row whose pivot is
// within the tolerance is reported. The pivot of row 0 is the diagonal entry (nothing
// is subtracted from it); the others are smaller and differ from it by far more than
// the rounding of the factorization.
TYPED_TEST(PrecondIluIc, near_singularity_tolerance_boundaries)
{
    constexpr Kind K      = TestFixture::K;
    const auto     handle = this->handle;
    const auto     a      = make_grid_matrix<double>(4, false);
    const auto     pivots = pivots_of<K>(a);
    const double   first  = pivots[0];
    const double   last   = *std::min_element(pivots.begin(), pivots.end());
    ASSERT_NEAR(first, (K == Kind::ic) ? std::sqrt(5.0) : 5.0, 1.0e-12);
    ASSERT_LT(last, first - 0.05);

    struct tolerance_case
    {
        double tol;
        bool   on_boundary; // the tolerance is meant to sit on a pivot
    };
    // Everything flagged, nothing flagged, between the smallest pivot and the others
    // and below all of them. For ILU(0) also exactly the first pivot (flagged) and the
    // largest tolerance below it (row 0 skipped).
    std::vector<tolerance_case> cases
        = {{1.0e3, false}, {1.0e-12, false}, {0.5 * (first + last), false}, {0.5 * last, false}};
    if constexpr(K == Kind::ilu)
    {
        cases.push_back({first, true});
        cases.push_back({std::nextafter(first, 0.0), true});
    }
    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(const bool after : {false, true})
        {
            for(const auto& c : cases)
            {
                const int64_t want = first_pivot_at_most(pivots, c.tol);
                if(!c.on_boundary)
                    ASSERT_TRUE(clear_of_pivots(pivots, c.tol)) << "tol " << c.tol;
                near_singular_case<K>(handle, a, c.tol, after, want, base);
            }
        }
    }
    // Skipping row 0 reports a later row.
    EXPECT_GT(first_pivot_at_most(pivots, 0.5 * (first + last)), 0);
}

// ---------------------------------------------------------------------------
// Numeric boost
// ---------------------------------------------------------------------------
namespace
{
    // On the grid matrix (positive pivots) a boost tolerance above every pivot replaces
    // each of them by the boost value, one below every pivot replaces none.
    template <typename T>
    void check_ilu_boost(rocsparse_handle handle)
    {
        const auto           a     = make_grid_matrix<T>(4, true);
        const std::vector<T> plain = host_ilu0(a);
        std::vector<double>  pivots;
        for(int32_t r = 0; r < a.n; ++r)
            pivots.push_back(std::abs(to_cd(plain[diagonal_position(a, r)])));
        const double largest  = *std::max_element(pivots.begin(), pivots.end());
        const double smallest = *std::min_element(pivots.begin(), pivots.end());
        for(const double tol : {2.0 * largest, 0.5 * smallest})
        {
            run_options opt;
            opt.boost     = true;
            opt.boost_tol = tol;
            opt.boost_val = 7.0;
            check_csr_matrix<Kind::ilu, T>(handle, a, opt, "boost tol " + std::to_string(tol));
        }
    }

    // |u_ii| of ILU(0) of `a` before any boost, with the boost tolerance `boost_tol` and
    // the boost value `boost_val` (no boost by default).
    template <typename T>
    std::vector<double>
        ilu_pivots(const csr_matrix<T>& a, double boost_tol = -1.0, double boost_val = 0.0)
    {
        std::vector<double> pivots;
        host_ilu0(a, boost_tol, boost_val, &pivots);
        return pivots;
    }

    // Calls `check(opt, what)` with tolerances between the pivots of ILU(0) of `a` and
    // below every pivot, each passed as a float or a double, in host or device memory. The
    // float is followed by 0x7fefffff, so an 8-byte read boosts every pivot: a tolerance
    // above every pivot would not tell it apart.
    template <typename T, typename F>
    void for_ilu_boost_inputs(const csr_matrix<T>& a, F&& check)
    {
        const auto   pivots   = ilu_pivots(a);
        const double largest  = *std::max_element(pivots.begin(), pivots.end());
        const double smallest = *std::min_element(pivots.begin(), pivots.end());
        // The boosted pivots change the later ones, so the margin to the tolerance is
        // checked on the pivots of the boosted factorization.
        const double between        = 0.5 * (smallest + largest);
        const auto   boosted_pivots = ilu_pivots(a, between, 7.0);
        const auto   boosted_count  = std::count_if(
            boosted_pivots.begin(), boosted_pivots.end(), [&](double p) { return p <= between; });
        ASSERT_TRUE(clear_of_pivots(boosted_pivots, between));
        ASSERT_GT(boosted_count, 0);
        ASSERT_LT(boosted_count, a.n);
        for(const double tol : {between, 0.5 * smallest})
        {
            for(const bool float_tol : {false, true})
            {
                for(const bool device : {false, true})
                {
                    run_options opt;
                    opt.boost           = true;
                    opt.boost_tol       = tol;
                    opt.boost_val       = 7.0;
                    opt.boost_tol_float = float_tol;
                    opt.boost_device    = device;
                    check(opt,
                          "tol " + std::to_string(tol) + " float_tol " + std::to_string(float_tol)
                              + " device " + std::to_string(device));
                }
            }
        }
    }

    // CSR through the hash kernel (the grid matrix) and the binary-search kernel (a last
    // row of 1100 entries, at least 1024 on a wave64 device).
    template <typename T>
    void check_ilu_boost_inputs(rocsparse_handle handle)
    {
        for(const auto& a :
            {make_grid_matrix<T>(4, true), with_upper_triangle(make_fat_row_matrix<T>(1100))})
        {
            for_ilu_boost_inputs(a, [&](const run_options& opt, const std::string& what) {
                check_csr_matrix<Kind::ilu, T>(
                    handle, a, opt, "n " + std::to_string(a.n) + " " + what);
            });
        }
    }

    // BSR through the bsrilu0 kernels: block_dim 2, 9 and 33 select the 2_8, 9_32 and
    // 33_64 kernels on a wave64 device (33 the general kernel for double complex), every
    // block_dim selects the general kernel on a wave32 device.
    template <typename T>
    void check_ilu_bsr_boost_inputs(rocsparse_handle handle)
    {
        for(const int32_t block_dim : {2, 9, 33})
        {
            for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
            {
                const auto m = make_block_matrix<T>(2, block_dim, dir, true);
                for_ilu_boost_inputs(
                    expand_bsr(m), [&](const run_options& opt, const std::string& what) {
                        check_bsr_matrix<Kind::ilu, T>(
                            handle, m, opt, "block_dim " + std::to_string(block_dim) + " " + what);
                    });
            }
        }
    }

    // Pivots (the entries of D, signed) of ILDL^T(0) of `a` before any boost, with the
    // boost tolerance `boost_tol` and the boost value `boost_val` (no boost by default).
    template <typename T>
    std::vector<double>
        ildlt_pivots(const csr_matrix<T>& a, double boost_tol = -1.0, double boost_val = 0.0)
    {
        std::vector<double> pivots;
        host_ildlt0(a, boost_tol, boost_val, &pivots);
        return pivots;
    }

    inline std::vector<double> moduli(std::vector<double> v)
    {
        for(auto& x : v)
            x = std::abs(x);
        return v;
    }

    // spildlt0 with the numeric boost on the grid matrix, whose pivots are positive and
    // whose D_0 is exactly A_00 = 5:
    // - tolerances above every pivot, below every pivot, and equal to D_0 (so the test is
    //   |D_i| <= tol, not <);
    // - 4.7, between the pivots: only some rows are boosted, their D feeds the later
    //   rows, and every A_ii (5) is above it, so D_i and not real(A_ii) is compared;
    // - a negative boost value, which must not flip the positive pivots;
    // - a matrix with a small negative pivot, which stays negative once boosted.
    template <typename T>
    void check_ildlt_boost(rocsparse_handle handle)
    {
        const auto   a        = make_grid_matrix<T>(4, false);
        const auto   pivots   = moduli(ildlt_pivots(a));
        const double largest  = *std::max_element(pivots.begin(), pivots.end());
        const double smallest = *std::min_element(pivots.begin(), pivots.end());
        const auto   check
            = [&](const csr_matrix<T>& m, double tol, double val, const std::string& what) {
                  run_options opt;
                  opt.boost     = true;
                  opt.boost_tol = tol;
                  opt.boost_val = val;
                  check_csr_matrix<Kind::ildlt, T>(handle, m, opt, what);
              };
        for(const double tol : {2.0 * largest, 0.5 * smallest, 5.0})
            check(a, tol, 7.0, "boost tol " + std::to_string(tol));

        // The boosted pivots change the later ones, so the margin to the tolerance is
        // checked on the pivots of the boosted factorization.
        const double between        = 4.7;
        const auto   boosted_pivots = moduli(ildlt_pivots(a, between, 7.0));
        const auto   boosted_count  = std::count_if(
            boosted_pivots.begin(), boosted_pivots.end(), [&](double p) { return p <= between; });
        ASSERT_TRUE(clear_of_pivots(boosted_pivots, between));
        ASSERT_GT(boosted_count, 0);
        ASSERT_LT(boosted_count, a.n);
        check(a, between, 7.0, "boost tol between the pivots");

        check(a, 2.0 * largest, -7.0, "negative boost value");

        // A_55 = 0.3 makes D_5 about -0.12; every other pivot stays above 1 in modulus.
        csr_matrix<T> indefinite                    = a;
        indefinite.val[diagonal_position(a, 5)]     = make_value<T>(0.3);
        const std::vector<double> indefinite_pivots = ildlt_pivots(indefinite, 1.0, 7.0);
        ASSERT_LT(indefinite_pivots[5], 0.0);
        ASSERT_GT(indefinite_pivots[5], -1.0);
        for(int32_t r = 0; r < a.n; ++r)
        {
            if(r != 5)
                ASSERT_GT(std::abs(indefinite_pivots[r]), 1.0) << "row " << r;
        }
        check(indefinite, 1.0, 7.0, "negative pivot");

        // Batch b scales A, hence D, by 1 + 0.25 * b: a tolerance of 6.5 boosts every
        // pivot of batches 0 and 1 (at most 6.25) and none of batch 2 (at least 6.8).
        run_options batched;
        batched.boost     = true;
        batched.boost_tol = 6.5;
        batched.boost_val = 7.0;
        for(const bool shared_row_ptr : {true, false})
            check_csr_case<Kind::ildlt, T>(handle, 4, 3, shared_row_ptr, batched);
    }

    // The boost of every pivot or of none, with the tolerance passed as a float or a
    // double, in host or device memory, through the hash kernel (the grid matrix) and the
    // binary-search kernel (a last row of 600 entries).
    template <typename T>
    void check_ildlt_boost_inputs(rocsparse_handle handle)
    {
        for(const auto& a : {make_grid_matrix<T>(4, false), make_fat_row_matrix<T>(600)})
        {
            const auto   pivots   = moduli(ildlt_pivots(a));
            const double largest  = *std::max_element(pivots.begin(), pivots.end());
            const double smallest = *std::min_element(pivots.begin(), pivots.end());
            for(const double tol : {2.0 * largest, 0.5 * smallest})
            {
                for(const bool float_tol : {false, true})
                {
                    for(const bool device : {false, true})
                    {
                        run_options opt;
                        opt.boost           = true;
                        opt.boost_tol       = tol;
                        opt.boost_val       = 7.0;
                        opt.boost_tol_float = float_tol;
                        opt.boost_device    = device;
                        check_csr_matrix<Kind::ildlt, T>(handle,
                                                         a,
                                                         opt,
                                                         "n " + std::to_string(a.n) + " tol "
                                                             + std::to_string(tol) + " float_tol "
                                                             + std::to_string(float_tol)
                                                             + " device " + std::to_string(device));
                    }
                }
            }
        }
    }

    // The boost enabled without a boost tolerance, without a boost value, or without both:
    // the analysis succeeds, and the compute stage returns rocsparse_status_invalid_pointer
    // without touching the values. Once the missing inputs are set, the same descriptor
    // computes the factor with every pivot boosted.
    template <Kind K, typename M>
    void check_boost_unset(rocsparse_handle handle, const M& m)
    {
        constexpr bool bsr = std::is_same_v<M, bsr_matrix<double>>;
        using D = std::conditional_t<bsr, dev_bsr<double>, dev_csr<double, int32_t, int32_t>>;
        const auto scalar = [&](const std::vector<double>& val) {
            M f   = m;
            f.val = val;
            if constexpr(bsr)
                return expand_bsr(f);
            else
                return f;
        };
        for(const bool set_tol : {false, true})
        {
            for(const bool set_val : {false, true})
            {
                if(set_tol && set_val)
                    continue;
                SCOPED_TRACE(::testing::Message()
                             << "set_tol " << set_tol << " set_val " << set_val);
                D d;
                if constexpr(bsr)
                    d.init(m);
                else
                    d.init(m, 1, true);
                ASSERT_NE(d.A, nullptr);
                run_options opt;
                opt.boost          = true;
                opt.boost_tol      = 1.0e3;
                opt.boost_val      = 7.0;
                opt.set_boost_tol  = set_tol;
                opt.set_boost_val  = set_val;
                opt.set_boost_late = true;
                run_result<double> out;
                run_factorization<K, double>(handle, d.A, d.val, 1, opt, out);
                EXPECT_EQ(out.early_compute_status, rocsparse_status_invalid_pointer);
                EXPECT_EQ(out.early_values, m.val);
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                check_factor<K>(scalar(m.val), scalar(out.values).val, opt, "missing inputs set");
            }
        }
    }
}

TEST_F(PrecondGeneric, spilu0_numeric_boost)
{
    for_all_types([&](auto tag) { check_ilu_boost<decltype(tag)>(handle); });
}

TEST_F(PrecondGeneric, spilu0_numeric_boost_inputs)
{
    for_all_types([&](auto tag) { check_ilu_boost_inputs<decltype(tag)>(handle); });
}

TEST_F(PrecondGeneric, spilu0_bsr_numeric_boost_inputs)
{
    for_all_types([&](auto tag) { check_ilu_bsr_boost_inputs<decltype(tag)>(handle); });
}

// The 3x3 of the AISPARSE-1063 reproducer: first pivot 1e-4, tolerance 1e-3, boost
// value 1. A float tolerance must boost the same pivot a double tolerance does.
TEST_F(PrecondGeneric, spilu0_numeric_boost_float_tolerance)
{
    csr_matrix<float> a;
    a.n   = 3;
    a.ptr = {0, 2, 4, 5};
    a.ind = {0, 1, 0, 1, 2};
    a.val = {1.0e-4f, 1.0f, 1.0f, 2.0f, 3.0f};
    for(const bool float_tol : {false, true})
    {
        for(const bool device : {false, true})
        {
            run_options opt;
            opt.boost           = true;
            opt.boost_tol       = 1.0e-3;
            opt.boost_val       = 1.0;
            opt.boost_tol_float = float_tol;
            opt.boost_device    = device;
            check_csr_matrix<Kind::ilu, float>(handle,
                                               a,
                                               opt,
                                               std::string(float_tol ? "float" : "double")
                                                   + (device ? " device" : " host"));
        }
    }
}

TEST_F(PrecondGeneric, spilu0_numeric_boost_unset)
{
    check_boost_unset<Kind::ilu>(handle, make_grid_matrix<double>(4, true));
    check_boost_unset<Kind::ilu>(handle,
                                 make_block_matrix<double>(3, 3, rocsparse_direction_row, true));
}

TEST_F(PrecondGeneric, spildlt0_numeric_boost)
{
    for_all_types([&](auto tag) { check_ildlt_boost<decltype(tag)>(handle); });
}

TEST_F(PrecondGeneric, spildlt0_numeric_boost_inputs)
{
    for_all_types([&](auto tag) { check_ildlt_boost_inputs<decltype(tag)>(handle); });
}

// With the boost enabled, spildlt0 still reports a pivot at or below the singularity
// tolerance (numeric_near, first such row) and then boosts it; spilu0 reports none.
TEST_F(PrecondGeneric, numeric_boost_and_singularity_tolerance)
{
    run_options opt;
    opt.boost        = true;
    opt.boost_tol    = 100.0;
    opt.boost_val    = 7.0;
    opt.set_sing_tol = true;
    opt.sing_tol     = 5.0;
    singular_case<Kind::ildlt, double>(
        handle, make_grid_matrix<double>(4, false), opt, rocsparse_singularity_numeric_near, 0);
    singular_case<Kind::ilu, double>(
        handle, make_grid_matrix<double>(4, true), opt, rocsparse_singularity_none, -1);
}

TEST_F(PrecondGeneric, spildlt0_numeric_boost_unset)
{
    check_boost_unset<Kind::ildlt>(handle, make_grid_matrix<double>(4, false));
}

// ---------------------------------------------------------------------------
// BSR
// ---------------------------------------------------------------------------

// The IC(0) matrices are block tridiagonal with Hermitian diagonal blocks; the scalar
// inner sums are non-zero (a scalar row shares columns with the rows it depends on
// inside a block), and the complex types have non-real products.
TYPED_TEST(PrecondIluIc, bsr_block_dims_and_index_base)
{
    constexpr Kind K      = TestFixture::K;
    const auto     handle = this->handle;
    const auto     one    = rocsparse_index_base_one;
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        run_options opt;
        for(const int32_t block_dim : {1, 2, 3, 4, 8, 9, 16, 17, 32})
            check_bsr_case<K, double>(handle, 4, block_dim, dir, opt);
        check_bsr_case<K, float>(handle, 4, 5, dir, opt);
        check_bsr_case<K, rocsparse_float_complex>(handle, 3, 6, dir, opt);
        check_bsr_case<K, rocsparse_double_complex>(handle, 3, 7, dir, opt);
        check_bsr_case<K, double>(handle, 4, 3, dir, opt, one);
        check_bsr_case<K, float>(handle, 3, 9, dir, opt, one);
    }
    run_options opt;
    opt.policy = rocsparse_analysis_policy_force;
    check_bsr_case<K, double>(handle, 4, 4, rocsparse_direction_row, opt);
}

// A 64-lane wavefront selects the specialised block kernels (2_8 / 9_32 / 33_64 for
// ILU, 2_8_unrolled / 2_8 / 9_16 / 17_32 for IC), so this only runs on a wave64 GPU.
TYPED_TEST(PrecondIluIc, bsr_wave64_kernels)
{
    if(device_warp_size() != 64)
        GTEST_SKIP() << "needs a GPU with a wavefront of 64";
    constexpr Kind             K      = TestFixture::K;
    const auto                 handle = this->handle;
    const std::vector<int32_t> wide = (K == Kind::ilu) ? std::vector<int32_t>{8, 9, 16, 32, 33, 64}
                                                       : std::vector<int32_t>{8, 9, 16, 17, 32};
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        const run_options opt;
        for(const int32_t block_dim : {2, 4})
        {
            check_bsr_case<K, double>(handle, 4, block_dim, dir, opt);
            check_bsr_case<K, float>(handle, 3, block_dim, dir, opt);
        }
        check_bsr_case<K, rocsparse_float_complex>(handle, 3, 3, dir, opt);
        check_bsr_case<K, rocsparse_double_complex>(handle, 3, 2, dir, opt);
        for(const int32_t block_dim : wide)
            check_bsr_case<K, double>(handle, 3, block_dim, dir, opt);
        check_bsr_case<K, float>(handle, 3, 20, dir, opt);
        check_bsr_case<K, rocsparse_float_complex>(handle, 3, 12, dir, opt);
        check_bsr_case<K, rocsparse_double_complex>(handle, 3, 20, dir, opt);
    }
}

// Block rows of 33 to 128 blocks and block_dim <= 8 select the plain (not unrolled) 2_8
// bsric0 kernel on a wave64 GPU and the general kernel on a wave32 GPU; the lower block
// pattern has a last block row of `fat` blocks.
TEST_F(PrecondGeneric, spic0_bsr_long_block_rows)
{
    const run_options opt;
    for(const auto dir : {rocsparse_direction_row, rocsparse_direction_column})
    {
        for(const int32_t block_dim : {2, 3, 4, 8})
        {
            check_bsr_matrix<Kind::ic, double>(
                handle,
                make_fat_block_row_matrix<double>(40, block_dim, dir),
                opt,
                "fat block row, block_dim=" + std::to_string(block_dim));
        }
        check_bsr_matrix<Kind::ic, float>(
            handle, make_fat_block_row_matrix<float>(100, 3, dir), opt, "fat block row 100");
        check_bsr_matrix<Kind::ic, rocsparse_float_complex>(
            handle,
            make_fat_block_row_matrix<rocsparse_float_complex>(40, 4, dir),
            opt,
            "fat block row, float complex");
        check_bsr_matrix<Kind::ic, rocsparse_double_complex>(
            handle,
            make_fat_block_row_matrix<rocsparse_double_complex>(40, 2, dir),
            opt,
            "fat block row, complex");
    }
}

// ---------------------------------------------------------------------------
// Row-length regimes of ILDLT(0): hash kernel up to 512 entries (the binary-search kernels
// are covered by the value and index type combinations below)
// ---------------------------------------------------------------------------
TEST_F(PrecondGeneric, spildlt0_csr_hash_row_lengths)
{
    // Row lengths up to 512 select the hash kernel; the table size grows with the length.
    run_options opt;
    for(const int32_t fat : {8, 31, 33, 63, 65, 127, 129, 255, 257, 511})
    {
        check_csr_matrix<Kind::ildlt, double>(
            handle, make_fat_row_matrix<double>(fat), opt, "fat=" + std::to_string(fat));
        check_csr_matrix<Kind::ildlt, double>(handle,
                                              make_fat_row_matrix<double>(fat, true, 3),
                                              opt,
                                              "band 3 fat=" + std::to_string(fat));
    }
}

// ---------------------------------------------------------------------------
// Descriptor, input, output and stage validation
// ---------------------------------------------------------------------------
namespace
{
    // The header documents invalid_size for a wrong data size, the library returns
    // invalid_value for a non-zero one: either is accepted (AISPARSE-928).
    void expect_size_error(rocsparse_status st)
    {
        EXPECT_TRUE(st == rocsparse_status_invalid_size || st == rocsparse_status_invalid_value)
            << "status " << st;
    }

    template <Kind K>
    void validate_descriptor_api(rocsparse_handle handle)
    {
        using A                   = api<K>;
        typename A::descr_t descr = nullptr;

        // create / destroy
        EXPECT_EQ(A::create(handle, nullptr), rocsparse_status_invalid_pointer);
        ASSERT_EQ(A::create(handle, &descr), rocsparse_status_success);
        ASSERT_NE(descr, nullptr);

        // set_input argument checks
        const rocsparse_datatype dtype = rocsparse_datatype_f64_r;
        EXPECT_EQ(A::set(nullptr, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::set(handle, nullptr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(
            A::set(
                handle, descr, static_cast<decltype(A::input(In::alg))>(99), &dtype, sizeof(dtype)),
            rocsparse_status_invalid_value);
        EXPECT_EQ(A::set(handle, descr, A::input(In::dtype), nullptr, sizeof(dtype)),
                  rocsparse_status_invalid_pointer);

        // Wrong sizes.
        const int32_t alg = 0;
        expect_size_error(A::set(handle, descr, A::input(In::alg), &alg, 1));
        expect_size_error(A::set(handle, descr, A::input(In::dtype), &dtype, 1));
        const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
        expect_size_error(A::set(handle, descr, A::input(In::policy), &policy, 1));
        expect_size_error(A::set(handle, descr, A::input(In::boost_enable), &alg, 1));
        const double value = 1.0;
        expect_size_error(A::set(handle, descr, A::input(In::boost_tol), &value, 3));
        expect_size_error(A::set(handle, descr, A::input(In::boost_val), &value, 1));
        expect_size_error(A::set(handle, descr, A::input(In::sing_tol), &value, 1));

        // Valid inputs: the boost tolerance may be given as a float or a double, and a
        // sizeof(float) tolerance is recorded as f32. spic0 never reads its boost
        // (AISPARSE-927). It is only validated here, no factorization of this test uses it.
        const float ftol = 1.0f;
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_tol), &ftol, sizeof(float)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_tol), &value, sizeof(double)),
                  rocsparse_status_success);
        const int32_t enable = 1;
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_enable), &enable, sizeof(int32_t)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::boost_val), &value, sizeof(void*)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::sing_tol), &value, sizeof(void*)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::alg), &alg, sizeof(alg)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::policy), &policy, sizeof(policy)),
                  rocsparse_status_success);
        EXPECT_EQ(A::set(handle, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_success);

        // get_output argument checks
        rocsparse_singularity singularity = rocsparse_singularity_none;
        int64_t               position    = 0;
        EXPECT_EQ(A::get(nullptr, descr, false, &singularity, sizeof(singularity)),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::get(handle, nullptr, false, &singularity, sizeof(singularity)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::get(handle, descr, false, nullptr, sizeof(singularity)),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::get(handle, descr, false, &singularity, 0), rocsparse_status_invalid_size);
        EXPECT_EQ(A::get(handle, descr, true, &position, 0), rocsparse_status_invalid_size);
        expect_size_error(A::get(handle, descr, false, &singularity, 1));
        expect_size_error(A::get(handle, descr, true, &position, 1));

        // Destroying a null descriptor is allowed.
        EXPECT_EQ(A::destroy(handle, nullptr), rocsparse_status_success);
        EXPECT_EQ(A::destroy(handle, descr), rocsparse_status_success);
    }

    template <Kind K>
    void validate_stage_api(rocsparse_handle handle)
    {
        using A = api<K>;
        dev_csr<double, int32_t, int32_t> d;
        d.init(make_grid_matrix<double>(3, false), 1, true);
        ASSERT_NE(d.A, nullptr);
        dev_csr<double, int32_t, int32_t> other;
        other.init(make_grid_matrix<double>(3, false), 1, true);
        ASSERT_NE(other.A, nullptr);

        descr_guard<K> guard(handle);
        auto&          descr = guard.descr;
        ASSERT_EQ(A::create(handle, &descr), rocsparse_status_success);

        size_t size = 0;
        // buffer_size argument checks.
        EXPECT_EQ(A::buffer_size(nullptr, descr, d.A, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::buffer_size(handle, nullptr, d.A, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::buffer_size(handle, descr, nullptr, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::buffer_size(handle, descr, d.A, nullptr, A::stage(Stage::analysis), &size),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::buffer_size(handle, descr, d.A, other.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(A::buffer_size(handle,
                                 descr,
                                 d.A,
                                 d.A,
                                 static_cast<decltype(A::stage(Stage::analysis))>(7),
                                 &size),
                  rocsparse_status_invalid_value);
        EXPECT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::analysis), nullptr),
                  rocsparse_status_invalid_pointer);
        ASSERT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::analysis), &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);

        // run argument checks.
        EXPECT_EQ(A::run(nullptr, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(A::run(handle, nullptr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::run(handle, descr, nullptr, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::run(handle, descr, d.A, nullptr, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_invalid_pointer);
        EXPECT_EQ(A::run(handle, descr, d.A, other.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_not_implemented);
        EXPECT_EQ(A::run(handle,
                         descr,
                         d.A,
                         d.A,
                         static_cast<decltype(A::stage(Stage::analysis))>(7),
                         size,
                         buffer.ptr),
                  rocsparse_status_invalid_value);
        // The misuse below is rejected, with a status the header does not specify.
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), 0, buffer.ptr),
                  rocsparse_status_success);
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, nullptr),
                  rocsparse_status_success);

        // The descriptor datatype must match the matrix.
        const rocsparse_datatype wrong = rocsparse_datatype_f32_r;
        ASSERT_EQ(A::set(handle, descr, A::input(In::dtype), &wrong, sizeof(wrong)),
                  rocsparse_status_success);
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_success);
        const rocsparse_datatype dtype = rocsparse_datatype_f64_r;
        ASSERT_EQ(A::set(handle, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_success);

        // The compute stage needs the analysis first; the analysis needs a policy.
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::compute), size, buffer.ptr),
                  rocsparse_status_success);
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_success);
        const rocsparse_analysis_policy policy = rocsparse_analysis_policy_reuse;
        ASSERT_EQ(A::set(handle, descr, A::input(In::policy), &policy, sizeof(policy)),
                  rocsparse_status_success);
        ASSERT_EQ(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_success);

        // The analysis cannot run twice and the descriptor can no longer be reconfigured.
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_success);
        const int32_t alg = 0;
        EXPECT_NE(A::set(handle, descr, A::input(In::alg), &alg, sizeof(alg)),
                  rocsparse_status_success);
        EXPECT_NE(A::set(handle, descr, A::input(In::policy), &policy, sizeof(policy)),
                  rocsparse_status_success);
        EXPECT_NE(A::set(handle, descr, A::input(In::dtype), &dtype, sizeof(dtype)),
                  rocsparse_status_success);

        size_t compute_size = 0;
        ASSERT_EQ(A::buffer_size(handle, descr, d.A, d.A, A::stage(Stage::compute), &compute_size),
                  rocsparse_status_success);
        device_vector<char> compute_buffer{compute_size ? compute_size : size_t{1}};
        ASSERT_TRUE(compute_buffer.ptr);
        ASSERT_EQ(A::run(handle,
                         descr,
                         d.A,
                         d.A,
                         A::stage(Stage::compute),
                         compute_size,
                         compute_size ? compute_buffer.ptr : nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        check_factor<K>(d.base, to_host(d.val), run_options{}, "stage api");

        // After the compute stage the analysis cannot run again.
        EXPECT_NE(A::run(handle, descr, d.A, d.A, A::stage(Stage::analysis), size, buffer.ptr),
                  rocsparse_status_success);

        EXPECT_EQ(A::destroy(handle, descr), rocsparse_status_success);
        descr = nullptr;
    }
}

TEST_F(PrecondGeneric, spilu0_descriptor_and_input_validation)
{
    validate_descriptor_api<Kind::ilu>(handle);
}

TEST_F(PrecondGeneric, spic0_descriptor_and_input_validation)
{
    validate_descriptor_api<Kind::ic>(handle);
}

TEST_F(PrecondGeneric, spildlt0_descriptor_and_input_validation)
{
    validate_descriptor_api<Kind::ildlt>(handle);
}

TEST_F(PrecondGeneric, spilu0_stage_validation)
{
    validate_stage_api<Kind::ilu>(handle);
}

TEST_F(PrecondGeneric, spic0_stage_validation)
{
    validate_stage_api<Kind::ic>(handle);
}

TEST_F(PrecondGeneric, spildlt0_stage_validation)
{
    validate_stage_api<Kind::ildlt>(handle);
}

namespace
{
    template <Kind K, typename T, typename I, typename J>
    void check_all_row_shapes(rocsparse_handle handle, const std::string& what)
    {
        const run_options opt;
        // spic0 gets no imaginary parts (AISPARSE-903).
        constexpr bool real_only = (K == Kind::ic);
        // ILU(0) gets the upper triangle as well, or its update would have nothing to do.
        const auto shape = [](const csr_matrix<T>& lower) {
            return (K == Kind::ilu) ? with_upper_triangle(lower) : lower;
        };

        // Short rows: hash kernel with the smallest table.
        check_csr_matrix<K, T, I, J>(
            handle, shape(make_lower_band_matrix<T>(12, 3, !real_only)), opt, what + " short rows");
        // A row of 100 entries: hash kernel with a table of 128.
        check_csr_matrix<K, T, I, J>(
            handle, shape(make_fat_row_matrix<T>(100, !real_only, 3)), opt, what + " fat row 100");
        // A row of 520 entries: binary-search kernel.
        // Not for spildlt0 i32 rows / i64 cols: find_csrildlt0_kernel_binsearch_launch
        // dispatches <int32_t, int32_t> whenever row_type is i32 without looking at
        // col_type, so 64-bit column indices are read as 32-bit (AISPARSE-920).
        if constexpr(!(K == Kind::ildlt && sizeof(I) == 4 && sizeof(J) == 8))
            check_csr_matrix<K, T, I, J>(handle,
                                         shape(make_fat_row_matrix<T>(520, !real_only, 3)),
                                         opt,
                                         what + " fat row 520");
    }

    template <Kind K, typename T>
    void check_index_types(rocsparse_handle handle, const std::string& what)
    {
        check_all_row_shapes<K, T, int32_t, int32_t>(handle, what + " i32/i32");
        check_all_row_shapes<K, T, int64_t, int32_t>(handle, what + " i64/i32");
        check_all_row_shapes<K, T, int32_t, int64_t>(handle, what + " i32/i64");
        check_all_row_shapes<K, T, int64_t, int64_t>(handle, what + " i64/i64");
    }

    template <Kind K>
    void check_value_types(rocsparse_handle handle, const std::string& what)
    {
        check_index_types<K, float>(handle, what + " float");
        check_index_types<K, double>(handle, what + " double");
        check_index_types<K, rocsparse_float_complex>(handle, what + " cfloat");
        check_index_types<K, rocsparse_double_complex>(handle, what + " cdouble");
    }
}

TEST_F(PrecondGeneric, spilu0_csr_value_and_index_type_combinations)
{
    check_value_types<Kind::ilu>(handle, "spilu0");
}

TEST_F(PrecondGeneric, spic0_csr_value_and_index_type_combinations)
{
    check_value_types<Kind::ic>(handle, "spic0");
}

TEST_F(PrecondGeneric, spildlt0_csr_value_and_index_type_combinations)
{
    check_value_types<Kind::ildlt>(handle, "spildlt0");
}

namespace
{
    // Tiny diagonal 2 x 2 matrices of the sparse formats the factorizations do not
    // support. The device arrays live as long as the object.
    struct other_format_matrices
    {
        device_vector<int32_t> ptr{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> ind{std::vector<int32_t>{0, 1}};
        device_vector<int32_t> ind_aos{std::vector<int32_t>{0, 0, 1, 1}};
        device_vector<double>  val{std::vector<double>{2.0, 3.0}};

        rocsparse_spmat_descr coo     = nullptr;
        rocsparse_spmat_descr coo_aos = nullptr;
        rocsparse_spmat_descr csc     = nullptr;
        rocsparse_spmat_descr ell     = nullptr;
        rocsparse_spmat_descr bell    = nullptr;

        other_format_matrices()
        {
            const auto i32  = rocsparse_indextype_i32;
            const auto base = rocsparse_index_base_zero;
            const auto f64  = rocsparse_datatype_f64_r;
            EXPECT_EQ(rocsparse_create_coo_descr(&coo, 2, 2, 2, ind, ind, val, i32, base, f64),
                      rocsparse_status_success);
            EXPECT_EQ(
                rocsparse_create_coo_aos_descr(&coo_aos, 2, 2, 2, ind_aos, val, i32, base, f64),
                rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_csc_descr(&csc, 2, 2, 2, ptr, ind, val, i32, i32, base, f64),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_ell_descr(&ell, 2, 2, ind, val, 1, i32, base, f64),
                      rocsparse_status_success);
            EXPECT_EQ(rocsparse_create_bell_descr(
                          &bell, 2, 2, rocsparse_direction_row, 1, 2, ind, val, i32, base, f64),
                      rocsparse_status_success);
        }

        ~other_format_matrices()
        {
            for(auto* d : {&coo, &coo_aos, &csc, &ell, &bell})
                if(*d)
                    (void)rocsparse_destroy_spmat_descr(*d);
        }

        other_format_matrices(const other_format_matrices&) = delete;
        other_format_matrices& operator=(const other_format_matrices&) = delete;
    };

    template <Kind K>
    void check_unsupported_formats(rocsparse_handle handle)
    {
        other_format_matrices mats;
        const struct
        {
            rocsparse_spmat_descr A;
            const char*           name;
        } formats[] = {{mats.coo, "coo"},
                       {mats.coo_aos, "coo_aos"},
                       {mats.csc, "csc"},
                       {mats.ell, "ell"},
                       {mats.bell, "bell"}};
        for(const auto& f : formats)
        {
            SCOPED_TRACE(f.name);
            ASSERT_NE(f.A, nullptr);
            descr_guard<K> guard(handle);
            auto&          descr = guard.descr;
            ASSERT_EQ(api<K>::create(handle, &descr), rocsparse_status_success);
            const rocsparse_datatype dtype = rocsparse_datatype_f64_r;
            ASSERT_EQ((set_value<K>(handle, descr, In::dtype, &dtype, sizeof(dtype))),
                      rocsparse_status_success);

            // The format is not supported by any stage.
            size_t size = 0;
            EXPECT_EQ(
                api<K>::buffer_size(handle, descr, f.A, f.A, api<K>::stage(Stage::analysis), &size),
                rocsparse_status_not_implemented);
            EXPECT_EQ(
                api<K>::buffer_size(handle, descr, f.A, f.A, api<K>::stage(Stage::compute), &size),
                rocsparse_status_not_implemented);
            device_vector<char> buffer{size_t{64}};
            ASSERT_TRUE(buffer.ptr);
            EXPECT_NE(api<K>::run(
                          handle, descr, f.A, f.A, api<K>::stage(Stage::analysis), 64, buffer.ptr),
                      rocsparse_status_success);

            // Whatever was recorded, reading the outputs must not fault.
            rocsparse_singularity singularity = rocsparse_singularity_numeric_near;
            int64_t               position    = -2;
            (void)api<K>::get(handle, descr, false, &singularity, sizeof(rocsparse_singularity));
            (void)api<K>::get(handle, descr, true, &position, sizeof(int64_t));
            EXPECT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
        }
    }
}

TEST_F(PrecondGeneric, unsupported_sparse_formats_are_rejected)
{
    check_unsupported_formats<Kind::ilu>(handle);
    check_unsupported_formats<Kind::ic>(handle);
    check_unsupported_formats<Kind::ildlt>(handle);
}

// A batch larger than the number of positions the handle buffer can hold is read back in
// several chunks. The device memory of the handle is shrunk so that a small batch
// needs three chunks. Every batch entry has its own singular pivot: the values of entry
// b are those of the first entry times s_b = 1 + 0.25 b, so the pivots scale the same way.
TEST_F(PrecondGeneric, spilu0_batched_singularity_positions_in_chunks)
{
    const csr_matrix<double> grid   = make_grid_matrix<double>(4, true);
    const auto               factor = host_ilu0(grid);
    std::vector<double>      pivots;
    for(int32_t r = 0; r < grid.n; ++r)
        for(int32_t p = grid.ptr[r]; p < grid.ptr[r + 1]; ++p)
            if(grid.ind[p] == r)
                pivots.push_back(std::abs(factor[p]));
    const double smallest = *std::min_element(pivots.begin(), pivots.end());
    const int    batch    = 5;

    for(const size_t memory : {size_t{16}, size_t{24}, size_t{32}})
    {
        for(const double scale : {1.15, 1.6, 0.5})
        {
            const double tol = scale * smallest;
            SCOPED_TRACE("memory " + std::to_string(memory) + " scale " + std::to_string(scale));
            // Expected zero based position per batch entry, and no pivot on the boundary.
            std::vector<int64_t> want(batch, -1);
            for(int b = 0; b < batch; ++b)
            {
                const double s = 1.0 + 0.25 * b;
                for(int32_t i = 0; i < grid.n; ++i)
                {
                    ASSERT_GT(std::abs(s * pivots[i] - tol), 1.0e-6 * tol);
                    if(want[b] < 0 && s * pivots[i] <= tol)
                        want[b] = i;
                }
            }

            dev_csr<double, int32_t, int32_t> d;
            d.init(grid, batch, true);
            ASSERT_NE(d.A, nullptr);
            run_options opt;
            opt.set_sing_tol  = true;
            opt.sing_tol      = tol;
            opt.device_memory = memory;
            run_result<double> out;
            run_factorization<Kind::ilu, double>(handle, d.A, d.val, batch, opt, out);
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            ASSERT_EQ(out.position.size(), static_cast<size_t>(batch));
            ASSERT_EQ(out.singularity.size(), static_cast<size_t>(batch));
            const size_t nnz = static_cast<size_t>(d.base.nnz());
            for(int b = 0; b < batch; ++b)
            {
                EXPECT_EQ(out.position[b], want[b]) << "batch " << b;
                EXPECT_EQ(out.singularity[b],
                          want[b] < 0 ? rocsparse_singularity_none
                                      : rocsparse_singularity_numeric_near)
                    << "batch " << b;
                const std::vector<double> got(out.values.begin() + b * nnz,
                                              out.values.begin() + (b + 1) * nnz);
                check_factor<Kind::ilu>(d.batches[b], got, opt, "batch " + std::to_string(b));
            }
        }
    }
}
