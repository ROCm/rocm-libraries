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
// Device (GPU) unit tests for rocSPARSE internal rocsparse::assign_ilu0_boost_value
// plus the host-side rocsparse::primitives::double_buffer utility. Split out of
// unit_test_internal_collectives.cpp by collective family.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp" // assign_ilu0_boost_value
#include "rocsparse_primitives.hpp" // rocsparse::primitives::double_buffer (host-side)

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    // ---- ilu0 boost --------------------------------------------------------
    template <typename T>
    __global__ void k_assign_boost(T value, T boost, T* out)
    {
        out[0] = rocsparse::assign_ilu0_boost_value<T>(value, boost);
    }
    template <typename T>
    void run_assign_boost(T value, T boost, T expected)
    {
        device_vector<T> d_out(size_t{1});
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_assign_boost<T>, 1u, value, boost, d_out.ptr), hipSuccess);
        expect_close(to_host(d_out)[0], expected);
    }
} // namespace

// ===========================================================================
// assign_ilu0_boost_value
// ===========================================================================
//
// PR #10290 (merged to upstream/develop) makes ROCSPARSE_ENABLE_ILU0_BOOST_SIGN
// default ON, so ROCSPARSE_WITH_ILU0_BOOST_SIGN is defined in this build and the
// sign/inertia-preserving path is the default. In that path the routine returns
// the boost *magnitude* carried along the sign/phase of the original pivot:
//
//   real:    (|value| > 0) ? copysign(|boost|, value) : |boost|
//   complex: (|value| > 0) ? |boost| * value / |value| : |boost|
//
// i.e. a negative boost can never flip the pivot's sign (inertia is preserved),
// and a zero pivot receives the pure boost magnitude. The sign-aware result is
// therefore the DEFAULT expectation below; a verbatim-boost expectation is only
// retained under an explicit #ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN fallback for
// a hypothetical no-sign build.

// --- Primary, same-direction cases (identical under both configurations) -----
TEST(internal_collectives_assign_ilu0_boost, f64)
{
    run_assign_boost<double>(2.0, 0.25, 0.25);
}
TEST(internal_collectives_assign_ilu0_boost, f32)
{
    run_assign_boost<float>(2.0f, 0.25f, 0.25f);
}
TEST(internal_collectives_assign_ilu0_boost, complex_f32)
{
    run_assign_boost<rocsparse_float_complex>(rocsparse_float_complex(2.0f, 0.0f),
                                              rocsparse_float_complex(0.25f, 0.0f),
                                              rocsparse_float_complex(0.25f, 0.0f));
}
TEST(internal_collectives_assign_ilu0_boost, complex_f64)
{
    run_assign_boost<rocsparse_double_complex>(rocsparse_double_complex(2.0, 0.0),
                                               rocsparse_double_complex(0.25, 0.0),
                                               rocsparse_double_complex(0.25, 0.0));
}

// --- Opposite-direction cases: sign-aware follows the pivot sign -------------
// value=-4.0, boost=0.25 -> copysign(|0.25|, -4.0) == -0.25 (sign-aware default).
TEST(internal_collectives_assign_ilu0_boost, f64_negative_value)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<double>(-4.0, 0.25, 0.25); // no-sign fallback: verbatim boost
#else
    run_assign_boost<double>(-4.0, 0.25, -0.25);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, f32_negative_value)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<float>(-4.0f, 0.25f, 0.25f);
#else
    run_assign_boost<float>(-4.0f, 0.25f, -0.25f);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, complex_f32_negative_value)
{
    const rocsparse_float_complex value(-4.0f, 0.0f);
    const rocsparse_float_complex boost(0.25f, 0.0f);
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<rocsparse_float_complex>(value, boost, rocsparse_float_complex(0.25f, 0.0f));
#else
    // |0.25| * (-4,0)/|(-4,0)| == (-0.25, 0)
    run_assign_boost<rocsparse_float_complex>(value, boost, rocsparse_float_complex(-0.25f, 0.0f));
#endif
}
TEST(internal_collectives_assign_ilu0_boost, complex_f64_negative_value)
{
    const rocsparse_double_complex value(-4.0, 0.0);
    const rocsparse_double_complex boost(0.25, 0.0);
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(0.25, 0.0));
#else
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(-0.25, 0.0));
#endif
}

// --- Negative boost: magnitude used, so the pivot sign is never flipped ------
// value=4.0, boost=-0.25 -> copysign(|-0.25|, 4.0) == +0.25 (sign-aware default).
TEST(internal_collectives_assign_ilu0_boost, f64_negative_boost_uses_magnitude)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<double>(4.0, -0.25, -0.25); // no-sign fallback: verbatim boost
#else
    run_assign_boost<double>(4.0, -0.25, 0.25);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, f32_negative_boost_uses_magnitude)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<float>(4.0f, -0.25f, -0.25f);
#else
    run_assign_boost<float>(4.0f, -0.25f, 0.25f);
#endif
}

// --- Zero pivot: |value| == 0 returns the pure boost magnitude |boost| -------
TEST(internal_collectives_assign_ilu0_boost, f64_zero_value_returns_abs_boost)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<double>(0.0, -0.25, -0.25); // no-sign fallback: verbatim boost
#else
    run_assign_boost<double>(0.0, -0.25, 0.25);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, f32_zero_value_returns_abs_boost)
{
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<float>(0.0f, -0.25f, -0.25f);
#else
    run_assign_boost<float>(0.0f, -0.25f, 0.25f);
#endif
}
TEST(internal_collectives_assign_ilu0_boost, complex_f64_zero_value_returns_abs_boost)
{
    // |value| == 0 -> returns |boost| as a pure-real complex.
    const rocsparse_double_complex value(0.0, 0.0);
    const rocsparse_double_complex boost(0.0, -0.25); // |boost| == 0.25
#ifndef ROCSPARSE_WITH_ILU0_BOOST_SIGN
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(0.0, -0.25));
#else
    run_assign_boost<rocsparse_double_complex>(value, boost, rocsparse_double_complex(0.25, 0.0));
#endif
}
// ===========================================================================
// host-side primitives::double_buffer
// ===========================================================================
// double_buffer<T>: current()/alternate() selectors with swap(); a pure host
// utility (no device launch).
TEST(internal_collectives_double_buffer, select_and_swap)
{
    int                                       a = 0, b = 0;
    rocsparse::primitives::double_buffer<int> db(&a, &b);
    EXPECT_EQ(db.current(), &a);
    EXPECT_EQ(db.alternate(), &b);
    db.swap();
    EXPECT_EQ(db.current(), &b);
    EXPECT_EQ(db.alternate(), &a);
    db.swap();
    EXPECT_EQ(db.current(), &a);
    EXPECT_EQ(db.alternate(), &b);
}
TEST(internal_collectives_double_buffer, default_ctor_is_null)
{
    rocsparse::primitives::double_buffer<double> db;
    EXPECT_EQ(db.current(), nullptr);
    EXPECT_EQ(db.alternate(), nullptr);
}
TEST(internal_collectives_double_buffer, void_ctor)
{
    int                                       a = 0, b = 0;
    rocsparse::primitives::double_buffer<int> db(static_cast<void*>(&a), static_cast<void*>(&b));
    EXPECT_EQ(db.current(), &a);
    EXPECT_EQ(db.alternate(), &b);
}
