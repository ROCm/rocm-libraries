/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

#if defined(ROCALUTION_WITH_OPENMP) && defined(_OPENMP) && _OPENMP >= 200805 && !defined(_WIN32)
#include <omp.h>
#endif

using namespace rocalution;

template <typename T>
void testing_amax_ties(void)
{
    // Above the OpenMP threshold, with 2 threads, so that the host backend runs threaded and
    // the static schedule splits the vector at size / 2
    int64_t size = 1 << 16;
    int64_t half = size / 2;

    set_omp_threads_rocalution(2);
    set_omp_threshold_rocalution(0);

#if defined(ROCALUTION_WITH_OPENMP) && defined(_OPENMP) && _OPENMP >= 200805 && !defined(_WIN32)
    // With dynamic teams the runtime may run the scan on one thread, which hides the race
    int def_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif

    LocalVector<T> vec;
    vec.Allocate("TestVector", size);

    T       max_value;
    int64_t max_index;

    // All entries equal: the first one is the maximum
    vec.Ones();
    max_value = 0;
    max_index = vec.Amax(max_value);
    EXPECT_EQ(max_index, 0);
    EXPECT_EQ(max_value, static_cast<T>(1));

    // Tied maxima on both sides of the chunk boundary
    vec[half - 1] = static_cast<T>(2);
    vec[half]     = static_cast<T>(2);
    max_value     = 0;
    max_index     = vec.Amax(max_value);
    EXPECT_EQ(max_index, half - 1);
    EXPECT_EQ(max_value, static_cast<T>(2));

    // Tied in magnitude only
    vec[half - 1] = static_cast<T>(-2);
    max_value     = 0;
    max_index     = vec.Amax(max_value);
    EXPECT_EQ(max_index, half - 1);
    EXPECT_EQ(max_value, static_cast<T>(2));

    vec[half - 1] = static_cast<T>(2);
    vec[half]     = static_cast<T>(-2);
    max_value     = 0;
    max_index     = vec.Amax(max_value);
    EXPECT_EQ(max_index, half - 1);
    EXPECT_EQ(max_value, static_cast<T>(2));

#if defined(ROCALUTION_WITH_OPENMP) && defined(_OPENMP) && _OPENMP >= 200805 && !defined(_WIN32)
    omp_set_dynamic(def_dynamic);
#endif

    // Default threshold; the thread count is reset by the next init_rocalution()
    set_omp_threshold_rocalution(10000);
}

#define TEST_CASE(function, type, disable_accelerator)   \
    disable_accelerator_rocalution(disable_accelerator); \
    set_device_rocalution(0);                            \
    init_rocalution();                                   \
    function<type>();                                    \
    stop_rocalution();                                   \
    disable_accelerator_rocalution(false);

// rocalution-unit-test checks the same tie rule, but a clients-only build does not build it
// and the rtest.xml sets do not run it.
// Host-only and cheap: run in every tier, including the emulation tiers used by CI
TEST(local_vector_amax_ties, host_float)
{
    TEST_CASE(testing_amax_ties, float, true);
}

TEST(local_vector_amax_ties, host_double)
{
    TEST_CASE(testing_amax_ties, double, true);
}
