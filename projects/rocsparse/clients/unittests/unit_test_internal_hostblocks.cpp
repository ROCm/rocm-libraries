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
// Unit tests for rocSPARSE internal host building-block routines.
//
// NOTE ON TARGET: these routines are pure *host* logic (selectors/partitioners),
// but their headers pull in rocsparse_control.hpp -> rocsparse_common.hpp, which
// uses HIP device intrinsics (__ldg, ...) that only compile under `-x hip`. This
// file therefore builds into the GPU test binary (rocsparse-unit-test-device),
// NOT the host-only rocsparse-unit-test, and the matching library TUs
// (rocsparse_csrmm_default_alg.cpp, rocsparse_determine_indextype.cpp) are
// compiled in via ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES. The tests themselves
// run host code and do not need to launch kernels.
//
// Exercises: csrmm_select_default_alg, determine_I/J_indextype, clz, host fnp2,
// line_nnz_profile guard logic, and itilu0 assign_b/unassign_b/buffer_layout.
// (The flp2 case is parked pending AISPARSE-642; see note below.)
//
#include "unit_test_utils.hpp"

// NOTE: the `flp2` helper lives in library/src/include/rocsparse_csrmv_adaptive_analysis.hpp,
// a header that is NOT present on this stack's base (ut-infra); it is introduced
// by the sibling level2 SpMV series (AISPARSE-642). To keep this PR
// self-contained on ut-infra, the flp2 case is PARKED (see internal_ut_findings.md)
// and this header is intentionally not included here. It will be reinstated once
// AISPARSE-642 lands in the base.

#include "rocsparse_csrmm.hpp" // csrmm_select_default_alg + line_nnz_profile
#include "rocsparse_determine_indextype.hpp" // determine_I/J_indextype

#include "rocsparse_common.hpp" // fnp2
#include "rocsparse_utility.hpp" // clz

// itilu0 buffer helpers (assign_b / unassign_b / buffer_layout_contiguous_t).
// Reached by an explicit relative path: library/src/precond/itilu0 is NOT on the
// unit-test include search path, so this needs no CMake change and avoids any
// ambiguity with other headers literally named "common.hpp".
#include "../../library/src/precond/itilu0/common.hpp"

#include <gtest/gtest.h>
#include <vector>

// ===========================================================================
// clz  (library/src/include/rocsparse_utility.hpp)
// ===========================================================================
// clz(n) returns the 1-based position of the most-significant set bit
// (floor(log2(n)) + 1) for n > 0, and 0 for n == 0. This identity holds for
// both the 32-bit and the ILP64 (64-bit) definitions of rocsparse_int, so the
// expected values below are configuration-independent.
TEST(internal_hostblocks, clz)
{
    EXPECT_EQ(rocsparse::clz(0), 0); // documented n == 0 special case
    EXPECT_EQ(rocsparse::clz(1), 1);
    EXPECT_EQ(rocsparse::clz(2), 2);
    EXPECT_EQ(rocsparse::clz(3), 2);
    EXPECT_EQ(rocsparse::clz(4), 3);
    EXPECT_EQ(rocsparse::clz(5), 3);
    EXPECT_EQ(rocsparse::clz(7), 3);
    EXPECT_EQ(rocsparse::clz(8), 4);
    EXPECT_EQ(rocsparse::clz(15), 4);
    EXPECT_EQ(rocsparse::clz(16), 5);
    EXPECT_EQ(rocsparse::clz(17), 5);
    EXPECT_EQ(rocsparse::clz(255), 8);
    EXPECT_EQ(rocsparse::clz(256), 9);
    EXPECT_EQ(rocsparse::clz(1023), 10);
    EXPECT_EQ(rocsparse::clz(1024), 11);
    // Highest bit representable without touching the sign bit of a 32-bit
    // rocsparse_int (bit 30).
    EXPECT_EQ(rocsparse::clz(static_cast<rocsparse_int>(1) << 30), 31);
}

// ===========================================================================
// fnp2  (library/src/include/rocsparse_common.hpp) - next power of two
// ===========================================================================
// fnp2(x) rounds x up to the next power of two. Powers of two map to
// themselves; the empty/zero input wraps to 0 (x-- underflows then x++ returns
// to 0), which is the documented edge-case behavior we lock in here.
TEST(internal_hostblocks, fnp2)
{
    EXPECT_EQ(rocsparse::fnp2(0u), 0u); // edge case: wraps back to 0
    EXPECT_EQ(rocsparse::fnp2(1u), 1u);
    EXPECT_EQ(rocsparse::fnp2(2u), 2u);
    EXPECT_EQ(rocsparse::fnp2(3u), 4u);
    EXPECT_EQ(rocsparse::fnp2(4u), 4u);
    EXPECT_EQ(rocsparse::fnp2(5u), 8u);
    EXPECT_EQ(rocsparse::fnp2(7u), 8u);
    EXPECT_EQ(rocsparse::fnp2(8u), 8u);
    EXPECT_EQ(rocsparse::fnp2(9u), 16u);
    EXPECT_EQ(rocsparse::fnp2(15u), 16u);
    EXPECT_EQ(rocsparse::fnp2(16u), 16u);
    EXPECT_EQ(rocsparse::fnp2(17u), 32u);
    EXPECT_EQ(rocsparse::fnp2(1000u), 1024u);
    EXPECT_EQ(rocsparse::fnp2(1024u), 1024u);
    EXPECT_EQ(rocsparse::fnp2(1u << 30), 1u << 30);
    EXPECT_EQ(rocsparse::fnp2(1u << 31), 1u << 31); // 2^31 is already a power of 2
}

// NOTE: the `flp2` case is PARKED for self-containment on ut-infra. It depends
// on rocsparse::flp2 from rocsparse_csrmv_adaptive_analysis.hpp, a header added
// by the sibling AISPARSE-642 level2 SpMV series that is not in this stack's
// base. Reinstate this case (and the header include above) once AISPARSE-642
// lands in ut-infra. See kim/internal_ut_findings.md.

// ===========================================================================
// csrmm_select_default_alg  (library/src/level3/rocsparse_csrmm_default_alg.cpp)
// + line_nnz_profile guard logic
// ===========================================================================
// The selector only ever upgrades the *format default* to the nnz-split kernel,
// and only when the profile is present and the longest-line imbalance test
// (profile.max * cu_count >= 3 * profile.nnz) fires. Every guard that keeps the
// historical row-split default is checked below.
namespace
{
    rocsparse_csrmm_alg select_alg(rocsparse_operation                trans_a,
                                   bool                               is_batched,
                                   int32_t                            cu_count,
                                   const rocsparse::line_nnz_profile& profile,
                                   rocsparse_csrmm_alg                start_alg)
    {
        rocsparse_csrmm_alg alg = start_alg;
        EXPECT_EQ(rocsparse::csrmm_select_default_alg(trans_a, is_batched, cu_count, profile, alg),
                  rocsparse_status_success);
        return alg;
    }
}

TEST(internal_hostblocks, csrmm_select_default_alg_explicit_alg_unchanged)
{
    // A concrete profile that WOULD trip the imbalance test if the algorithm
    // were the default; an explicit (non-default) choice must be preserved.
    rocsparse::line_nnz_profile profile{};
    profile.known = true;
    profile.nnz   = 100;
    profile.max   = 100;

    EXPECT_EQ(select_alg(rocsparse_operation_none,
                         false,
                         64,
                         profile,
                         rocsparse_csrmm_alg_nnz_split),
              rocsparse_csrmm_alg_nnz_split);
}

TEST(internal_hostblocks, csrmm_select_default_alg_guards_keep_default)
{
    rocsparse::line_nnz_profile tripping{};
    tripping.known = true;
    tripping.nnz   = 100;
    tripping.max   = 100; // 100 * cu_count >= 3 * 100 once cu_count >= 3

    // Transposed multiply: profile is not applicable -> stays default.
    EXPECT_EQ(select_alg(rocsparse_operation_transpose, false, 64, tripping,
                         rocsparse_csrmm_alg_default),
              rocsparse_csrmm_alg_default);

    // Batched multiply -> stays default.
    EXPECT_EQ(select_alg(rocsparse_operation_none, true, 64, tripping,
                         rocsparse_csrmm_alg_default),
              rocsparse_csrmm_alg_default);

    // Unknown profile -> stays default.
    {
        rocsparse::line_nnz_profile unknown{};
        unknown.known = false;
        unknown.nnz   = 100;
        unknown.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 64, unknown,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Non-positive nnz -> stays default.
    {
        rocsparse::line_nnz_profile zero_nnz{};
        zero_nnz.known = true;
        zero_nnz.nnz   = 0;
        zero_nnz.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 64, zero_nnz,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Non-positive compute-unit count -> stays default.
    EXPECT_EQ(select_alg(rocsparse_operation_none, false, 0, tripping,
                         rocsparse_csrmm_alg_default),
              rocsparse_csrmm_alg_default);
}

TEST(internal_hostblocks, csrmm_select_default_alg_imbalance_threshold)
{
    // Balanced enough to keep row-split: max * cu = 1 * 1 = 1 < 3 * 100.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 1;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 1, p,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Just below the crossover: max * cu = 100 * 2 = 200 < 3 * 100 = 300.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 2, p,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Exactly at the crossover (>=): 100 * 3 = 300 >= 300 -> upgrade to nnz-split.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 3, p,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_nnz_split);
    }

    // Clearly imbalanced -> nnz-split.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 90;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 64, p,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_nnz_split);
    }
}

// ===========================================================================
// determine_I_indextype / determine_J_indextype
// (library/src/common/rocsparse_determine_indextype.cpp)
// ===========================================================================
// These map a sparse-matrix descriptor's storage index types onto the (I, J)
// index roles. For CSR the row-pointer type is I and the column-index type is
// J; CSC swaps them; single-index formats (COO) return the same type for both.
namespace
{
    // Small, valid, non-null device allocations so descriptor creation passes its
    // array/size argument checks. Contents are irrelevant to the index-type query.
    struct dummy_device_arrays
    {
        rocsparse_ut::device_vector<char> a{size_t{256}};
        rocsparse_ut::device_vector<char> b{size_t{256}};
        rocsparse_ut::device_vector<char> c{size_t{256}};
    };
}

TEST(internal_hostblocks, determine_indextype_csr)
{
    dummy_device_arrays d;

    // Distinct row-pointer / column-index types so the two roles are separable.
    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr,
                                         d.b.ptr,
                                         d.c.ptr,
                                         rocsparse_indextype_i32, // row_ptr  -> I
                                         rocsparse_indextype_i64, // col_ind  -> J
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i32);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST(internal_hostblocks, determine_indextype_csr_uniform)
{
    dummy_device_arrays d;

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr,
                                         d.b.ptr,
                                         d.c.ptr,
                                         rocsparse_indextype_i64,
                                         rocsparse_indextype_i64,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i64);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST(internal_hostblocks, determine_indextype_csc_swaps_roles)
{
    dummy_device_arrays d;

    // CSC stores col_ptr as the I-role and row_ind as the J-role, i.e. the
    // opposite of CSR.
    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_csc_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr, // csc_col_ptr
                                         d.b.ptr, // csc_row_ind
                                         d.c.ptr, // csc_val
                                         rocsparse_indextype_i32, // col_ptr -> I
                                         rocsparse_indextype_i64, // row_ind -> J
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i32);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

TEST(internal_hostblocks, determine_indextype_coo)
{
    dummy_device_arrays d;

    rocsparse_spmat_descr mat = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&mat,
                                         4,
                                         4,
                                         4,
                                         d.a.ptr, // coo_row_ind
                                         d.b.ptr, // coo_col_ind
                                         d.c.ptr, // coo_val
                                         rocsparse_indextype_i64,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);

    EXPECT_EQ(rocsparse::determine_I_indextype(mat), rocsparse_indextype_i64);
    EXPECT_EQ(rocsparse::determine_J_indextype(mat), rocsparse_indextype_i64);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
}

// ===========================================================================
// itilu0 buffer bookkeeping (library/src/precond/itilu0/common.hpp)
// ===========================================================================
// assign_b carves a T[nitems] slice off the front of a running buffer, advancing
// the cursor and shrinking the remaining size; it strictly requires
// buffer_size > sizeof(T) * nitems and returns nullptr otherwise. unassign_b is
// the exact inverse of that bookkeeping.
TEST(internal_hostblocks, itilu0_assign_b)
{
    std::vector<char> mem(1024);
    void*             buffer      = mem.data();
    size_t            buffer_size = mem.size();

    int32_t* p = rocsparse::assign_b<int32_t>(buffer_size, buffer, 10);
    EXPECT_EQ(reinterpret_cast<void*>(p), reinterpret_cast<void*>(mem.data()));
    EXPECT_EQ(buffer, reinterpret_cast<void*>(mem.data() + 10 * sizeof(int32_t)));
    EXPECT_EQ(buffer_size, mem.size() - 10 * sizeof(int32_t));

    // A second slice continues contiguously from the advanced cursor.
    char* q = rocsparse::assign_b<char>(buffer_size, buffer, 5);
    EXPECT_EQ(reinterpret_cast<void*>(q),
              reinterpret_cast<void*>(mem.data() + 10 * sizeof(int32_t)));
    EXPECT_EQ(buffer, reinterpret_cast<void*>(mem.data() + 10 * sizeof(int32_t) + 5));
    EXPECT_EQ(buffer_size, mem.size() - 10 * sizeof(int32_t) - 5);
}

TEST(internal_hostblocks, itilu0_assign_b_exact_boundary_fails)
{
    // Release-only: in a debug build the failure path prints and exit(1)s, so we
    // only probe the strict-inequality boundary when asserts are compiled out.
#ifdef NDEBUG
    std::vector<char> mem(64);
    void*             buffer      = mem.data();
    size_t            buffer_size = 40; // exactly sizeof(int32_t) * 10

    int32_t* p = rocsparse::assign_b<int32_t>(buffer_size, buffer, 10);
    EXPECT_EQ(p, nullptr); // '>' is strict, so an exact fit is rejected
    EXPECT_EQ(buffer, reinterpret_cast<void*>(mem.data())); // cursor untouched
    EXPECT_EQ(buffer_size, 40u); // size untouched
#else
    GTEST_SKIP() << "assign_b failure path abort()s when NDEBUG is not defined";
#endif
}

TEST(internal_hostblocks, itilu0_unassign_b_is_inverse_of_assign_b)
{
    std::vector<char> mem(1024);
    void*             buffer      = mem.data();
    size_t            buffer_size = mem.size();

    void* const  buffer0 = buffer;
    const size_t size0   = buffer_size;

    (void)rocsparse::assign_b<double>(buffer_size, buffer, 7);
    EXPECT_NE(buffer, buffer0);
    EXPECT_NE(buffer_size, size0);

    rocsparse::unassign_b<double>(buffer_size, buffer, 7);
    EXPECT_EQ(buffer, buffer0); // cursor restored
    EXPECT_EQ(buffer_size, size0); // size restored
}

// buffer_layout_contiguous_t::init lays out all of the itilu0 working arrays
// contiguously inside one user buffer. We verify the per-array sizes and that
// every array starts exactly where the previous one ended (the partition
// invariant), plus the reserved double-aligned header and the trailing
// "remaining buffer" accounting.
TEST(internal_hostblocks, itilu0_buffer_layout_contiguous_init)
{
    using layout_t = rocsparse::buffer_layout_contiguous_t;
    using I        = int32_t;
    using J        = int32_t;

    const I m   = 8;
    const I nnz = 20;

    // Generously sized, double-aligned backing store so every assign_b succeeds.
    std::vector<double> backing(4096, 0.0);
    void*               buffer      = backing.data();
    size_t              buffer_size = backing.size() * sizeof(double);

    void* const  base      = buffer;
    const size_t base_size = buffer_size;

    layout_t layout;
    layout.init<I, J>(m, nnz, rocsparse_datatype_f64_r, buffer_size, buffer);

    // Header reserved at the front: get_sizeof_double() doubles.
    const size_t header_bytes = layout_t::get_sizeof_double() * sizeof(double);
    char* const  after_header = reinterpret_cast<char*>(base) + header_bytes;

    // Expected per-array sizes.
    EXPECT_EQ(layout.get_size(layout_t::perm), sizeof(I) * nnz);
    EXPECT_EQ(layout.get_size(layout_t::lnnz), sizeof(I) * 1);
    EXPECT_EQ(layout.get_size(layout_t::lptr), sizeof(I) * (m + 1));
    EXPECT_EQ(layout.get_size(layout_t::unnz), sizeof(I) * 1);
    EXPECT_EQ(layout.get_size(layout_t::uptr), sizeof(I) * (m + 1));
    EXPECT_EQ(layout.get_size(layout_t::ind), sizeof(J) * nnz);
    EXPECT_EQ(layout.get_size(layout_t::x), sizeof(double) * nnz); // f64_r

    // Contiguous placement in allocation order: perm, lnnz, lptr, unnz, uptr,
    // ind, x - each starting exactly where the previous ended.
    char* cursor = after_header;
    EXPECT_EQ(layout.get_pointer(layout_t::perm), reinterpret_cast<void*>(cursor));
    cursor += sizeof(I) * nnz;
    EXPECT_EQ(layout.get_pointer(layout_t::lnnz), reinterpret_cast<void*>(cursor));
    cursor += sizeof(I) * 1;
    EXPECT_EQ(layout.get_pointer(layout_t::lptr), reinterpret_cast<void*>(cursor));
    cursor += sizeof(I) * (m + 1);
    EXPECT_EQ(layout.get_pointer(layout_t::unnz), reinterpret_cast<void*>(cursor));
    cursor += sizeof(I) * 1;
    EXPECT_EQ(layout.get_pointer(layout_t::uptr), reinterpret_cast<void*>(cursor));
    cursor += sizeof(I) * (m + 1);
    EXPECT_EQ(layout.get_pointer(layout_t::ind), reinterpret_cast<void*>(cursor));
    cursor += sizeof(J) * nnz;
    EXPECT_EQ(layout.get_pointer(layout_t::x), reinterpret_cast<void*>(cursor));
    cursor += sizeof(double) * nnz;

    // lptr_end / uptr_end point one element past lptr / uptr.
    EXPECT_EQ(layout.get_pointer(layout_t::lptr_end),
              reinterpret_cast<void*>(reinterpret_cast<I*>(layout.get_pointer(layout_t::lptr)) + 1));
    EXPECT_EQ(layout.get_pointer(layout_t::uptr_end),
              reinterpret_cast<void*>(reinterpret_cast<I*>(layout.get_pointer(layout_t::uptr)) + 1));

    // The trailing "remaining buffer" region starts at the cursor and its size
    // matches the leftover buffer_size reported back to the caller.
    EXPECT_EQ(layout.get_pointer(layout_t::buffer), reinterpret_cast<void*>(cursor));
    EXPECT_EQ(layout.get_size(layout_t::buffer), buffer_size);

    // Total consumed = header + all arrays; leftover is the rest of the store.
    const size_t consumed = header_bytes + sizeof(I) * nnz + sizeof(I) * 1 + sizeof(I) * (m + 1)
                            + sizeof(I) * 1 + sizeof(I) * (m + 1) + sizeof(J) * nnz
                            + sizeof(double) * nnz;
    EXPECT_EQ(buffer_size, base_size - consumed);
    EXPECT_EQ(buffer, reinterpret_cast<void*>(cursor));
}

TEST(internal_hostblocks, harness_smoke)
{
    SUCCEED();
}
