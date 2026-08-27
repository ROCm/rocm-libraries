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
// Unit tests for the rocSPARSE internal itilu0 buffer-bookkeeping helpers
// (library/src/precond/itilu0/common.hpp): assign_b / unassign_b and the
// buffer_layout_contiguous_t partitioner. Split out of
// unit_test_internal_hostblocks.cpp by topic.
//
// assign_b carves a T[nitems] slice off the front of a running buffer,
// advancing the cursor and shrinking the remaining size; it strictly requires
// buffer_size > sizeof(T) * nitems and returns nullptr otherwise. unassign_b is
// the exact inverse of that bookkeeping.
//
// NOTE ON TARGET: these are header-only host helpers, but the include chain
// requires the HIP compile mode, so this file builds into the GPU test binary
// (rocsparse-unit-test-device). No kernels are launched.
//
#include "unit_test_utils.hpp"

// itilu0 buffer helpers (assign_b / unassign_b / buffer_layout_contiguous_t).
// Reached by an explicit relative path: library/src/precond/itilu0 is NOT on the
// unit-test include search path, so this needs no CMake change and avoids any
// ambiguity with other headers literally named "common.hpp".
#include "../../library/src/precond/itilu0/common.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(internal_hostblocks_itilu0, assign_b)
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

TEST(internal_hostblocks_itilu0, assign_b_exact_boundary_fails)
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

TEST(internal_hostblocks_itilu0, unassign_b_is_inverse_of_assign_b)
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
TEST(internal_hostblocks_itilu0, buffer_layout_contiguous_init)
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
    EXPECT_EQ(
        layout.get_pointer(layout_t::lptr_end),
        reinterpret_cast<void*>(reinterpret_cast<I*>(layout.get_pointer(layout_t::lptr)) + 1));
    EXPECT_EQ(
        layout.get_pointer(layout_t::uptr_end),
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
