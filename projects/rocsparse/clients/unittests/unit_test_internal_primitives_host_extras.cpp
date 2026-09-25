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

//
// Host-path unit tests for the RESIDUAL in-scope rocSPARSE internal HOST helpers
// (the "compile-in" / header-exposer family), compiled into the
// rocsparse-unit-test-device binary (it needs a GPU handle + device memory and
// its include chain requires the HIP toolchain).
//
// Routines covered here:
//   * rocsparse::primitives::sort_csr_column_indices (both overloads) plus its
//     buffer-size query. Their defining TU
//     (library/src/primitives/rocsparse_segmented_radix_sort_keys.cpp) is ALREADY
//     compiled into this binary because unit_test_primitives_support.cpp
//     #includes it directly, and it is explicitly instantiated for
//     (I=int32,J=int32), (I=int64,J=int32), (I=int64,J=int64). No new CMake
//     compile-in is required.
//   * rocsparse::find_V<K> and rocsparse::find_radix_sort_pairs_buffer_size --
//     the radix dispatch selectors (rocsparse_primitives.hpp ~261,282). They are
//     header-inline and return pointers into
//     rocsparse::primitives::radix_sort_pairs_buffer_size<K,V>, whose
//     instantiations are already compiled in via
//     ROCSPARSE_UNIT_TEST_PRIMITIVE_SOURCES (rocsparse_radix_sort_pairs.cpp).
//
// Deliberately SKIPPED (documented, not silently dropped):
//   * enum_utils::is_invalid / to_string RUNTIME paths -- see the SKIP note at
//     the bottom of this file. Reaching them requires compiling a heavy library
//     TU with no gc-sections, which pulls unresolved hidden symbols; there is no
//     dedicated enum-only TU to compile in, and a clean exposure would require a
//     library-side change (out of scope).
//   * host format-default selectors "beyond csrmm": there is exactly one
//     *_select_default_alg selector in the whole tree (csrmm_select_default_alg),
//     already covered by unit_test_internal_hostblocks.cpp. No other candidates
//     exist; see the SKIP note.
//
#include "unit_test_utils.hpp"

#include "rocsparse_primitives.hpp" // primitives::sort_csr_column_indices, find_V, ...

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::HandleTest;
using rocsparse_ut::to_host;

// ===========================================================================
// primitives::sort_csr_column_indices : sorts the column indices within each
// CSR row (segmented radix sort keyed by row). Two overloads:
//   (a) 3-buffer form (const input, result in buffer2)
//   (b) 2-buffer form (mutable input, result written back into csr_col_ind)
// ===========================================================================
namespace
{
    // Fixture name deliberately carries the "primitives_host_extras" token so the
    // whole file is selected by the --gtest_filter '*primitives_host_extras*'.
    struct internal_primitives_host_extras_sort_csr : HandleTest
    {
    };

    // Build a small unsorted CSR structure and its per-row-sorted reference.
    template <typename J>
    struct csr_case
    {
        std::vector<J> row_ptr; // length m+1 (offset type filled by caller as I)
        std::vector<J> col_ind; // unsorted
        std::vector<J> sorted; // per-row ascending reference
        J              m;
        J              n;
    };

    // Column layout (m=4, n=10):
    //   row0: 5 2 8      -> 2 5 8
    //   row1: 1          -> 1
    //   row2: 9 0 3 7    -> 0 3 7 9
    //   row3: (empty)
    template <typename I, typename J>
    void run_sort_csr_column_indices(rocsparse_handle handle, bool three_buffer)
    {
        const J              m       = 4;
        const J              n       = 10;
        const std::vector<I> row_ptr = {0, 3, 4, 8, 8};
        const std::vector<J> col_ind = {5, 2, 8, 1, 9, 0, 3, 7};
        const std::vector<J> sorted  = {2, 5, 8, 1, 0, 3, 7, 9};
        const I              nnz     = static_cast<I>(col_ind.size());
        const size_t         nnz_sz  = col_ind.size();

        device_vector<I> d_row_ptr(row_ptr);
        device_vector<J> d_col_ind(col_ind);
        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);

        size_t buffer_size = 0;
        ASSERT_EQ((rocsparse::primitives::sort_csr_column_indices_buffer_size<I, J>(
                      handle, m, n, nnz, d_row_ptr.ptr, &buffer_size)),
                  rocsparse_status_success);
        // The segmented radix sort always needs a non-trivial scratch buffer.
        EXPECT_GT(buffer_size, size_t{0});

        device_vector<char> d_buffer(buffer_size ? buffer_size : size_t{1});
        ASSERT_NE(d_buffer.ptr, nullptr);

        if(three_buffer)
        {
            device_vector<J> d_buf1(nnz_sz), d_buf2(nnz_sz);
            ASSERT_NE(d_buf1.ptr, nullptr);
            ASSERT_NE(d_buf2.ptr, nullptr);
            ASSERT_EQ((rocsparse::primitives::sort_csr_column_indices<I, J>(handle,
                                                                            m,
                                                                            n,
                                                                            nnz,
                                                                            d_row_ptr.ptr,
                                                                            d_col_ind.ptr,
                                                                            d_buf1.ptr,
                                                                            d_buf2.ptr,
                                                                            d_buffer.ptr)),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(to_host(d_buf2), sorted);
            // The const input must be untouched.
            EXPECT_EQ(to_host(d_col_ind), col_ind);
        }
        else
        {
            device_vector<J> d_buf2(nnz_sz);
            ASSERT_NE(d_buf2.ptr, nullptr);
            ASSERT_EQ(
                (rocsparse::primitives::sort_csr_column_indices<I, J>(
                    handle, m, n, nnz, d_row_ptr.ptr, d_col_ind.ptr, d_buf2.ptr, d_buffer.ptr)),
                rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(to_host(d_col_ind), sorted);
        }
    }
} // namespace

TEST_F(internal_primitives_host_extras_sort_csr, three_buffer_i32_i32)
{
    run_sort_csr_column_indices<int32_t, int32_t>(handle, true);
}
TEST_F(internal_primitives_host_extras_sort_csr, three_buffer_i64_i32)
{
    run_sort_csr_column_indices<int64_t, int32_t>(handle, true);
}
TEST_F(internal_primitives_host_extras_sort_csr, three_buffer_i64_i64)
{
    run_sort_csr_column_indices<int64_t, int64_t>(handle, true);
}
TEST_F(internal_primitives_host_extras_sort_csr, two_buffer_i32_i32)
{
    run_sort_csr_column_indices<int32_t, int32_t>(handle, false);
}
TEST_F(internal_primitives_host_extras_sort_csr, two_buffer_i64_i32)
{
    run_sort_csr_column_indices<int64_t, int32_t>(handle, false);
}
TEST_F(internal_primitives_host_extras_sort_csr, two_buffer_i64_i64)
{
    run_sort_csr_column_indices<int64_t, int64_t>(handle, false);
}

// ===========================================================================
// find_V<K> / find_radix_sort_pairs_buffer_size : the radix-sort dispatch
// selectors. They map (K, V) index types to the matching
// radix_sort_pairs_buffer_size<K,V> instantiation, returning nullptr for the
// deprecated u16 index type.
// ===========================================================================
TEST(internal_primitives_host_extras_find_V, selects_expected_instantiation)
{
    // K = int32
    EXPECT_EQ(rocsparse::find_V<int32_t>(rocsparse_indextype_i32),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int32_t>));
    EXPECT_EQ(rocsparse::find_V<int32_t>(rocsparse_indextype_i64),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int64_t>));
    EXPECT_EQ(rocsparse::find_V<int32_t>(deprecated_rocsparse_indextype_u16), nullptr);

    // K = int64
    EXPECT_EQ(rocsparse::find_V<int64_t>(rocsparse_indextype_i32),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int64_t, int32_t>));
    EXPECT_EQ(rocsparse::find_V<int64_t>(rocsparse_indextype_i64),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int64_t, int64_t>));
    EXPECT_EQ(rocsparse::find_V<int64_t>(deprecated_rocsparse_indextype_u16), nullptr);
}

TEST(internal_primitives_host_extras_find_radix, dispatch_matrix)
{
    // Valid (K, V) combinations resolve to the matching typed selector.
    EXPECT_EQ(rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32,
                                                           rocsparse_indextype_i32),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int32_t>));
    EXPECT_EQ(rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32,
                                                           rocsparse_indextype_i64),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int64_t>));
    EXPECT_EQ(rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i64,
                                                           rocsparse_indextype_i32),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int64_t, int32_t>));
    EXPECT_EQ(rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i64,
                                                           rocsparse_indextype_i64),
              (&rocsparse::primitives::radix_sort_pairs_buffer_size<int64_t, int64_t>));

    // Any u16 on either axis is unsupported -> nullptr.
    EXPECT_EQ(rocsparse::find_radix_sort_pairs_buffer_size(deprecated_rocsparse_indextype_u16,
                                                           rocsparse_indextype_i32),
              nullptr);
    EXPECT_EQ(rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32,
                                                           deprecated_rocsparse_indextype_u16),
              nullptr);
}

// The selected function pointer is not just identity-equal, it is callable and
// produces a valid buffer size against a real handle (exercises the runtime
// dispatch path end-to-end).
TEST_F(internal_primitives_host_extras_sort_csr, find_radix_selected_pointer_is_callable)
{
    auto fn = rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32,
                                                           rocsparse_indextype_i32);
    ASSERT_NE(fn, nullptr);
    size_t buffer_size = 0;
    ASSERT_EQ(fn(handle, /*length*/ 128, /*startbit*/ 0, /*endbit*/ 32, &buffer_size, true),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, size_t{0});
}

// ===========================================================================
// SKIP notes (in-scope candidates that proved unreachable without a library
// change or a risky build change):
//
// 1. enum_utils::is_invalid / to_string (src/include/rocsparse_enum_utils.hpp):
//    only the primary template is declared in the header; every runtime
//    definition is an explicit specialization living in a library .cpp and is
//    compiled with hidden visibility (verified with `nm`: the symbols are local
//    `t`, absent from the .so dynamic table), so they cannot be linked from this
//    binary against roc::rocsparse. The specializations for the standard public
//    enums live in library/src/auxiliary/rocsparse_auxiliary.cpp (5.6k lines,
//    which also defines the public handle C API + logging), and every other
//    defining TU (spmm/spgemm/spgeam/check_spmat/...) embeds a full compute
//    chain. Compiling any of them in fails to link because the test target links
//    WITHOUT --gc-sections, so the many unused heavy functions in those TUs drag
//    unresolved hidden symbols. There is no dedicated enum-only TU (analogous to
//    csrmm_default_alg.cpp) to compile in. A clean exposure would need a
//    library-side exposer TU (a library change) or enabling --gc-sections on the
//    test target (risks dropping gtest static registrations) -- both out of
//    scope. These paths remain covered indirectly through the public C API in
//    the existing host_* suites.
//
// 2. host format-default selectors "beyond csrmm": a tree-wide search finds
//    exactly one *_select_default_alg (rocsparse::csrmm_select_default_alg),
//    already tested by unit_test_internal_hostblocks.cpp. No other
//    *_select_default_alg selectors exist, so there is nothing additional to
//    cover here.
// ===========================================================================
